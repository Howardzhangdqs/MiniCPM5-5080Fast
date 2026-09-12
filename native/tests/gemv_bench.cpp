// gemv_bench.cpp — P3+ 自研 GEMV 微基准（CTest: gemv_bench，计划 §10.2）。
//
// 内容：
//  ① 正确性：5 个 decode M=1 shape，随机 bf16 权重/输入，CPU fp32 参考
//    （double 累加 → bf16 舍入；lm_head 抽样 4096 行，其余全量）。
//    自研 rows / split-K 与 cuBLASLt 三者都对拍（rel-L2 < 1e-2、
//    cos > 0.9999、逐元素 ≤ 2 bf16 ulp——fp32 累加序差异 + 单次 RNE）。
//  ② 微基准：每 shape 各引擎 3 预热 + 计量（事件夹循环），GB/s 按实际
//    权重字节（N×K×2B）计；表 + benchmark/results/gemv_ab.json。
//  ③ tactic 提示：打印每 shape 赢家（与 runtime 建期快测同准则：min ms）。
// 共享 GPU 注记：util 记入 JSON；绝对 GB/s 受争用影响，A/B 相对有效。
#include "kernels/sm120/kernels.h"
#include "runtime/error.h"
#include "runtime/gemm.h"
#include "runtime/gemv_tactic.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

#define REQUIRE(cond)                                                          \
    do {                                                                       \
        if (!(cond)) {                                                         \
            fprintf(stderr, "FAIL %s:%d: %s (last_error: %s)\n", __FILE__,     \
                    __LINE__, #cond, mc_last_error());                         \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

namespace {

uint32_t g_lcg = 1;
float lcg_uniform(float lo, float hi) {
    g_lcg = g_lcg * 1664525u + 1013904223u;
    return lo + (hi - lo) * ((float)(g_lcg >> 8) / (float)0x1000000u);
}
uint16_t f2bf(float f) {
    // RNE 舍入（host 侧参考用；够用即可——对拍容差 2 ulp）
    uint32_t x = 0;
    std::memcpy(&x, &f, 4);
    const uint32_t lsb = (x >> 16) & 1u;
    const uint32_t y = x + 0x7FFFu + lsb;
    return (uint16_t)(y >> 16);
}
float bf2f(uint16_t h) {
    const uint32_t x = (uint32_t)h << 16;
    float f = 0.f;
    std::memcpy(&f, &x, 4);
    return f;
}

struct DevBuf {
    void* p = nullptr;
    size_t n = 0;
    ~DevBuf() {
        if (p != nullptr) cudaFree(p);
    }
    bool alloc(size_t bytes) {
        n = bytes;
        return cudaMalloc(&p, bytes ? bytes : 1) == cudaSuccess && p != nullptr;
    }
};

struct ShapeSpec {
    uint32_t n, k;
};

struct ShapeRow {
    const char* name;
    uint32_t n, k;
    double cublas_ms, rows_ms, split_ms; // <0 = 未跑
    double cublas_gbs, rows_gbs, split_gbs;
    int winner; // 0=cublas 1=rows 2=splitK
    uint32_t split_s;
    // P4：fp8 GEMV（合成 e4m3 + 行 scale；DRAM-honest 同 bf16 口径）
    double fp8_ms = -1.0, fp8_gbs = 0.0;
};

// 引擎计时（事件夹计量循环）。DRAM-honest：wcopies 为同一权重的多份拷贝
//（总容量 ≥ ~2×L2），逐迭代轮换 → 每次 ODE 读的都不是上一轮驻留 L2 的那
// 份 —— 小 shape 重复读单份缓冲只会测出 L2 带宽（首次实测 3300GB/s 的
// 教训），decode 真实条件是 4.5GB 权重流式过 DRAM。
double time_engine(mc::GemmEngine& eng, void* ws, size_t ws_size, int tactic,
                   uint32_t S, const uint16_t** wcopies, uint32_t n_copies,
                   const uint16_t* x, uint16_t* y, float* part, uint32_t N,
                   uint32_t K, int warm, int reps) {
    int it = 0;
    auto run_one = [&]() -> mc_status_t {
        const uint16_t* w = wcopies[(uint32_t)it++ % n_copies];
        if (tactic == mc::kGemvTacticCublas)
            return eng.gemm_bf16(0u, ws, ws_size, 1u, N, K, w, x, y);
        return launch_decode_gemv(w, x, y, N, K, tactic, S, part, 0u);
    };
    for (int i = 0; i < warm; ++i)
        if (run_one() != MC_OK) return -1.0;
    cudaEvent_t e0, e1;
    if (cudaEventCreate(&e0) != cudaSuccess || cudaEventCreate(&e1) != cudaSuccess)
        return -1.0;
    (void)cudaEventRecord(e0, 0u);
    for (int i = 0; i < reps; ++i)
        if (run_one() != MC_OK) return -1.0;
    (void)cudaEventRecord(e1, 0u);
    if (cudaEventSynchronize(e1) != cudaSuccess) return -1.0;
    float ms = 0.f;
    (void)cudaEventElapsedTime(&ms, e0, e1);
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    return (double)ms / (double)reps;
}

// CPU 参考 + 对拍（rows：抽样的行集合）
bool verify(const std::vector<uint16_t>& w, const std::vector<uint16_t>& x,
            const std::vector<uint16_t>& y, uint32_t K,
            const std::vector<uint32_t>& rows, const char* tag) {
    double sum_aa = 0, sum_rr = 0, sum_dd = 0;
    int bad = 0;
    uint32_t first_bad_row = 0;
    double first_got = 0, first_ref = 0;
    for (uint32_t row : rows) {
        double acc = 0;
        const uint16_t* wr = w.data() + (size_t)row * K;
        for (uint32_t k = 0; k < K; ++k)
            acc += (double)bf2f(wr[k]) * (double)bf2f(x[k]);
        const float ref = bf2f(f2bf((float)acc)); // fp32→bf16（与 kernel 同舍入位）
        const float got = bf2f(y[row]);
        const double d = (double)got - (double)ref;
        sum_aa += (double)got * (double)got;
        sum_rr += (double)ref * (double)ref;
        sum_dd += d * d;
        const float ulp = (float)std::ldexp(1.0, (int)std::floor(std::log2(
                              std::fabs((double)ref) + 1e-30)) - 7);
        if (std::fabs(d) > 2.0 * (double)ulp + 1e-3) {
            if (bad == 0) {
                first_bad_row = row;
                first_got = got;
                first_ref = ref;
            }
            ++bad;
        }
    }
    const double rl = std::sqrt(sum_dd / (sum_rr + 1e-30));
    const double cos = sum_aa > 0 && sum_rr > 0
                           ? (sum_aa + sum_rr - sum_dd) /
                                 (2.0 * std::sqrt(sum_aa * sum_rr))
                           : 1.0; // 恒等式近似（残差小量级时准确）
    printf("    %-10s relL2=%.3e cos=%.8f bad=%d/%zu", tag, rl, cos, bad,
           rows.size());
    if (bad > 0) printf(" first: row=%u got=%.5f ref=%.5f", first_bad_row,
                        first_got, first_ref);
    printf("\n");
    return rl < 1e-2 && bad == 0;
}

} // namespace

int main() {
    fprintf(stderr, "gemv_bench: begin\n");
    REQUIRE(cudaFree(0) == cudaSuccess); // 初始化上下文

    int util0 = -1;
    {
        // 记录基准时的 GPU util（共享机器注记）
        FILE* f = popen("nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null", "r");
        if (f != nullptr) {
            if (fscanf(f, "%d", &util0) != 1) util0 = -1;
            pclose(f);
        }
    }

    mc::GemmEngine eng;
    REQUIRE(eng.init() == MC_OK);
    constexpr size_t kWs = 16u << 20;
    DevBuf ws;
    REQUIRE(ws.alloc(kWs));
    REQUIRE(eng.prewarm_decode_shapes(kWs) == MC_OK);

    const ShapeSpec shapes[5] = {
        {2560, 2048}, {2048, 2048}, {12288, 2048}, {2048, 6144}, {130560, 2048}};
    const char* names[5] = {"qkv", "o", "gate_up", "down", "lm_head"};

    std::vector<ShapeRow> rows_out;
    bool all_ok = true;

    for (int si = 0; si < 5; ++si) {
        const uint32_t N = shapes[si].n, K = shapes[si].k;
        const size_t w_elems = (size_t)N * K;
        printf("== %s: N=%u K=%u (%.1f MB) ==\n", names[si], N, K,
               (double)w_elems * 2.0 / 1e6);

        // host 数据（LCG bf16 [-1,1]；lm_head 268M 元素 ~0.5s 可接受）
        g_lcg = 0xC0FFEEu + (uint32_t)si;
        std::vector<uint16_t> h_w(w_elems), h_x(K);
        for (auto& v : h_w) v = f2bf(lcg_uniform(-1.0f, 1.0f));
        for (auto& v : h_x) v = f2bf(lcg_uniform(-1.0f, 1.0f));

        // DRAM-honest：多份权重拷贝（总 ≥ ~2×L2 ≈ 256MB），计时逐份轮换
        const double wbytes = (double)w_elems * 2.0;
        uint32_t n_copies =
            (uint32_t)((wbytes >= 268435456.0) ? 1 : (268435456.0 / wbytes) + 1);
        DevBuf dw;
        REQUIRE(dw.alloc((size_t)wbytes * n_copies));
        for (uint32_t c = 0; c < n_copies; ++c)
            REQUIRE(cudaMemcpy((char*)dw.p + (size_t)wbytes * c, h_w.data(),
                               (size_t)wbytes, cudaMemcpyHostToDevice) ==
                    cudaSuccess);
        std::vector<const uint16_t*> wcopies(n_copies);
        for (uint32_t c = 0; c < n_copies; ++c)
            wcopies[c] = (const uint16_t*)((char*)dw.p + (size_t)wbytes * c);
        const uint16_t* w0 = wcopies[0];

        DevBuf dx, dy_cublas, dy_rows, dy_split, dpart;
        REQUIRE(dx.alloc((size_t)K * 2) && dy_cublas.alloc((size_t)N * 2) &&
                dy_rows.alloc((size_t)N * 2) && dy_split.alloc((size_t)N * 2) &&
                dpart.alloc((size_t)8 * N * 4));
        REQUIRE(cudaMemcpy(dx.p, h_x.data(), (size_t)K * 2,
                           cudaMemcpyHostToDevice) == cudaSuccess);

        // CPU 参考行集合：全量（≤ 12288 行）或抽样 4096
        std::vector<uint32_t> ref_rows;
        if (N <= 12288u) {
            ref_rows.resize(N);
            for (uint32_t i = 0; i < N; ++i) ref_rows[i] = i;
        } else {
            const uint32_t step = N / 4096u;
            for (uint32_t i = 0; i < N; i += step) ref_rows.push_back(i);
        }

        // ---- 正确性（split-K 按 S=2 与 S=8 双档验证）----
        REQUIRE(eng.gemm_bf16(0u, ws.p, kWs, 1u, N, K, w0,
                              (const uint16_t*)dx.p, (uint16_t*)dy_cublas.p) == MC_OK);
        REQUIRE(launch_decode_gemv(w0, (const uint16_t*)dx.p,
                                   (uint16_t*)dy_rows.p, N, K, mc::kGemvTacticRows,
                                   1, (float*)dpart.p, 0u) == MC_OK);
        REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
        std::vector<uint16_t> y_cublas(N), y_rows(N), y_split(N);
        REQUIRE(cudaMemcpy(y_cublas.data(), dy_cublas.p, (size_t)N * 2,
                           cudaMemcpyDeviceToHost) == cudaSuccess);
        REQUIRE(cudaMemcpy(y_rows.data(), dy_rows.p, (size_t)N * 2,
                           cudaMemcpyDeviceToHost) == cudaSuccess);
        bool ok = verify(h_w, h_x, y_cublas, K, ref_rows, "cublasLt");
        ok = verify(h_w, h_x, y_rows, K, ref_rows, "gemv-rows") && ok;
        for (uint32_t S : {2u, 8u}) {
            if (K % S != 0u || (K / S) % 8u != 0u) continue;
            REQUIRE(launch_decode_gemv(w0, (const uint16_t*)dx.p,
                                       (uint16_t*)dy_split.p, N, K,
                                       mc::kGemvTacticSplitK, S, (float*)dpart.p,
                                       0u) == MC_OK);
            REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
            REQUIRE(cudaMemcpy(y_split.data(), dy_split.p, (size_t)N * 2,
                               cudaMemcpyDeviceToHost) == cudaSuccess);
            char tag[32];
            snprintf(tag, sizeof(tag), "gemv-splitK(S=%u)", S);
            ok = verify(h_w, h_x, y_split, K, ref_rows, tag) && ok;
        }
        all_ok = ok && all_ok;

        // ---- 微基准（DRAM-honest，多拷贝轮换）----
        const int reps = wbytes > 256e6 ? 10 : 24;
        const int warm = 3;
        ShapeRow r{};
        r.name = names[si];
        r.n = N;
        r.k = K;
        r.split_s = 0;
        r.cublas_ms =
            time_engine(eng, ws.p, kWs, mc::kGemvTacticCublas, 1, wcopies.data(),
                        n_copies, (const uint16_t*)dx.p, (uint16_t*)dy_cublas.p,
                        (float*)dpart.p, N, K, warm, reps);
        r.rows_ms =
            time_engine(eng, ws.p, kWs, mc::kGemvTacticRows, 1, wcopies.data(),
                        n_copies, (const uint16_t*)dx.p, (uint16_t*)dy_rows.p,
                        (float*)dpart.p, N, K, warm, reps);
        // split-K 扫 S ∈ {2,8}，取最优
        r.split_ms = -1.0;
        for (uint32_t S : {2u, 8u}) {
            if (K % S != 0u || (K / S) % 8u != 0u) continue;
            const double t = time_engine(eng, ws.p, kWs, mc::kGemvTacticSplitK, S,
                                         wcopies.data(), n_copies,
                                         (const uint16_t*)dx.p,
                                         (uint16_t*)dy_split.p, (float*)dpart.p, N,
                                         K, warm, reps);
            if (t >= 0.0 && (r.split_ms < 0.0 || t < r.split_ms)) {
                r.split_ms = t;
                r.split_s = S;
            }
        }
        r.cublas_gbs = r.cublas_ms > 0 ? wbytes / (r.cublas_ms * 1e6) : 0;
        r.rows_gbs = r.rows_ms > 0 ? wbytes / (r.rows_ms * 1e6) : 0;
        r.split_gbs = r.split_ms > 0 ? wbytes / (r.split_ms * 1e6) : 0;
        r.winner = 0;
        double best = r.cublas_ms;
        if (r.rows_ms > 0 && r.rows_ms < best) {
            best = r.rows_ms;
            r.winner = 1;
        }
        if (r.split_ms > 0 && r.split_ms < best) {
            best = r.split_ms;
            r.winner = 2;
        }

        // ---- P4：fp8 GEMV 微基准（仅层内 4 shape；合成随机字节 e4m3 +
        //      随机行 scale；与 bf16 同多拷贝轮换 DRAM-honest 口径）----
        if (si < 4) {
            g_lcg = 0x0F0800u + (uint32_t)si;
            const size_t fp8_bytes = (size_t)w_elems; // 1B/元素
            const uint32_t n8 = (uint32_t)((fp8_bytes >= 268435456ull)
                                              ? 1
                                              : (268435456ull / fp8_bytes) + 1);
            DevBuf dw8, dsc8;
            REQUIRE(dw8.alloc(fp8_bytes * n8) && dsc8.alloc((size_t)N * 4));
            std::vector<uint8_t> h8((size_t)std::min<uint64_t>(fp8_bytes, 1 << 20));
            std::vector<const uint16_t*> unused8; // 复用 time_engine 需 uint16 指针——
            // 直接内联计时（launch_gemv_fp8 签名不同）
            for (uint32_t c = 0; c < n8; ++c) {
                for (size_t off = 0; off < fp8_bytes; off += h8.size()) {
                    const size_t n = std::min(h8.size(), fp8_bytes - off);
                    for (size_t i = 0; i < n; ++i)
                        h8[i] = (uint8_t)(g_lcg = g_lcg * 1664525u + 1013904223u,
                                          (uint8_t)(g_lcg >> 24));
                    REQUIRE(cudaMemcpy((char*)dw8.p + (size_t)fp8_bytes * c + off,
                                       h8.data(), n, cudaMemcpyHostToDevice) ==
                            cudaSuccess);
                }
            }
            std::vector<float> hsc(N);
            for (auto& v : hsc) v = 1e-4f * lcg_uniform(1.0f, 10.0f);
            REQUIRE(cudaMemcpy(dsc8.p, hsc.data(), (size_t)N * 4,
                               cudaMemcpyHostToDevice) == cudaSuccess);
            std::vector<const uint8_t*> w8(n8);
            for (uint32_t c = 0; c < n8; ++c)
                w8[c] = (const uint8_t*)((char*)dw8.p + (size_t)fp8_bytes * c);
            const uint8_t* w80 = w8[0];
            REQUIRE(launch_gemv_fp8(w80, (const float*)dsc8.p,
                                    (const uint16_t*)dx.p, (uint16_t*)dy_rows.p, N,
                                    K, 0u) == MC_OK); // 功能冒烟
            REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
            for (int i = 0; i < warm; ++i)
                REQUIRE(launch_gemv_fp8(w8[(uint32_t)i % n8], (const float*)dsc8.p,
                                        (const uint16_t*)dx.p,
                                        (uint16_t*)dy_rows.p, N, K, 0u) == MC_OK);
            cudaEvent_t e0, e1;
            REQUIRE(cudaEventCreate(&e0) == cudaSuccess &&
                    cudaEventCreate(&e1) == cudaSuccess);
            (void)cudaEventRecord(e0, 0u);
            for (int i = 0; i < reps; ++i)
                REQUIRE(launch_gemv_fp8(w8[(uint32_t)i % n8], (const float*)dsc8.p,
                                        (const uint16_t*)dx.p,
                                        (uint16_t*)dy_rows.p, N, K, 0u) == MC_OK);
            (void)cudaEventRecord(e1, 0u);
            REQUIRE(cudaEventSynchronize(e1) == cudaSuccess);
            float ms8 = 0.f;
            (void)cudaEventElapsedTime(&ms8, e0, e1);
            cudaEventDestroy(e0);
            cudaEventDestroy(e1);
            r.fp8_ms = (double)ms8 / (double)reps;
            r.fp8_gbs = r.fp8_ms > 0 ? (double)w_elems / (r.fp8_ms * 1e6) : 0;
            (void)unused8;
        }

        printf("    cublasLt %7.1f GB/s | gemv-rows %7.1f GB/s | gemv-splitK "
               "(best S=%u) %7.1f GB/s",
               r.cublas_gbs, r.rows_gbs, r.split_s, r.split_gbs);
        if (r.fp8_ms > 0)
            printf(" | gemv-fp8 %7.1f GB/s (by fp8 bytes)", r.fp8_gbs);
        printf(" -> winner: %s%s\n",
               r.winner == 0    ? "cublasLt"
               : r.winner == 1  ? "gemv-rows"
                                : "gemv-splitK",
               r.winner == 0 ? " (custom did NOT win)" : "");
        rows_out.push_back(r);
    }
    REQUIRE(all_ok);

    // ---- JSON ----
    {
        const std::string dir = std::string(MC_REPO_ROOT) + "/benchmark/results";
        std::error_code ec;
        fs::create_directories(dir, ec);
        const std::string path = dir + "/gemv_ab.json";
        std::ofstream f(path);
        if (!f) {
            fprintf(stderr, "gemv_bench: FAIL — cannot write %s\n", path.c_str());
            return 1;
        }
        time_t t = time(nullptr);
        char ts[32];
        strftime(ts, sizeof(ts), "%Y-%m-%dT%H:%M:%S", localtime(&t));
        f << "{\n  \"timestamp\": \"" << ts << "\",\n";
        f << "  \"gpu_util_at_start\": " << util0 << ",\n";
        f << "  \"nominal_peak_gbs\": 960.0,\n";
        f << "  \"shapes\": [\n";
        for (size_t i = 0; i < rows_out.size(); ++i) {
            const auto& r = rows_out[i];
            f << "    {\"name\": \"" << r.name << "\", \"N\": " << r.n
              << ", \"K\": " << r.k << ", \"weight_mb\": "
              << (double)r.n * (double)r.k * 2.0 / 1e6 << ",\n";
            f << "     \"cublaslt\": {\"ms\": " << r.cublas_ms
              << ", \"gbs\": " << r.cublas_gbs << "},\n";
            f << "     \"gemv_rows\": {\"ms\": " << r.rows_ms
              << ", \"gbs\": " << r.rows_gbs << "},\n";
            f << "     \"gemv_splitk\": {\"ms\": " << r.split_ms
              << ", \"gbs\": " << r.split_gbs << ", \"S\": " << r.split_s << "},\n";
            f << "     \"gemv_fp8\": {\"ms\": " << r.fp8_ms
              << ", \"gbs_by_fp8_bytes\": " << r.fp8_gbs << "},\n";
            const double wg = r.winner == 0   ? r.cublas_gbs
                              : r.winner == 1 ? r.rows_gbs
                                              : r.split_gbs;
            f << "     \"winner\": \"" << (r.winner == 0    ? "cublaslt"
                                           : r.winner == 1  ? "gemv_rows"
                                                            : "gemv_splitk")
              << "\", \"winner_gbs\": " << wg << ", \"pct_of_peak\": "
              << wg / 960.0 * 100.0 << "}" << (i + 1 < rows_out.size() ? "," : "")
              << "\n";
        }
        f << "  ]\n}\n";
        printf("  results written: %s\n", path.c_str());
    }

    fprintf(stderr, "gemv_bench: all done\n");
    return 0;
}
