// gemv_tactic.cpp — P3+ decode GEMV tactic 建期快测 + dispatch 实现。
//
// 快测（select_decode_gemv_tactics，session_create 期、prewarm 之后）：
//   对 5 个 decode M=1 shape 用真实权重指针各跑「3 预热 + N 计量」，
//   cudaEvent 计时（事件夹住整个计量循环，除以轮数得单次）；
//   候选：cuBLASLt（已预热的 plan）/ rows / split-K（仅小 N shape）。
//   赢家写入 ModelImpl（进程内一次，mutex 保护；后续 session 复用）。
//   x/y 缓冲复用 session activation scratch（a.gate_up 行 0 当 x ——
//   宽 12288 ≥ 所有 K；a.logits 当 y —— 宽 vocab ≥ 所有 N）。计时与数值
//   无关，x 先清零保证无 NaN。
//
//   注记（测量学）：真实权重只有一份，4/5 shape（≤50MB）反复读会驻留
//   L2 → 快测偏向 L2 带宽条件（gemv_bench 的多拷贝 DRAM-honest 对照见
//   benchmark/results/gemv_ab.json）。两侧引擎同等条件对比，结论仍有效
//   （实测选定结果与 DRAM-honest 赢家一致：rows 胜/平 4 shape，
//   lm_head 因 535MB 无法驻留而天然 DRAM-honest → cuBLASLt）。
//
// 失败语义：任一引擎计时出错 → 该 shape 保底 cuBLASLt（tactic 0），
// 不让 session 失败（自研 GEMV 是优化，不是正确性依赖）。
#include "runtime/gemv_tactic.h"

#include <cstdio>
#include <mutex>

#include <cuda_runtime.h>

#include "runtime/error.h"
#include "runtime/internal_state.h"

namespace mc {

namespace {

struct ShapeSpec {
    uint32_t n;
    uint32_t k;
};

// 与 prewarm_decode_shapes / decode_shape_index 一致的形状表。
void decode_shape_table(ShapeSpec out[5]) {
    using namespace mc::cfg;
    const uint32_t qkv_n = kNumQueryHeads * kHeadDim + 2 * kNumKvHeads * kHeadDim;
    out[0] = {qkv_n, kHiddenSize};
    out[1] = {kHiddenSize, kHiddenSize};
    out[2] = {2 * kIntermediateSize, kHiddenSize};
    out[3] = {kHiddenSize, kIntermediateSize};
    out[4] = {kVocabSize, kHiddenSize};
}

// 单引擎计时：warm 预热 + reps 计量，返回单次 ms（<0 = 失败）。
double time_engine(int tactic, uint32_t split_s, const ModelImpl& m,
                   const uint16_t* w, const uint16_t* x, uint16_t* y, float* part,
                   uint32_t N, uint32_t K, cudaStream_t st, int warm, int reps) {
    auto run_one = [&]() -> mc_status_t {
        if (tactic == kGemvTacticCublas)
            return m.gemm.gemm_bf16(st, m.gemm_ws, m.gemm_ws_bytes, 1u, N, K, w, x, y);
        return launch_decode_gemv(w, x, y, N, K, tactic, split_s, part, st);
    };
    for (int i = 0; i < warm; ++i)
        if (run_one() != MC_OK) return -1.0;
    cudaEvent_t e0 = nullptr, e1 = nullptr;
    if (cudaEventCreate(&e0) != cudaSuccess || cudaEventCreate(&e1) != cudaSuccess) {
        if (e0 != nullptr) cudaEventDestroy(e0);
        if (e1 != nullptr) cudaEventDestroy(e1);
        cudaGetLastError();
        return -1.0;
    }
    (void)cudaEventRecord(e0, st);
    for (int i = 0; i < reps; ++i)
        if (run_one() != MC_OK) {
            cudaEventDestroy(e0);
            cudaEventDestroy(e1);
            return -1.0;
        }
    (void)cudaEventRecord(e1, st);
    if (cudaEventSynchronize(e1) != cudaSuccess) {
        cudaEventDestroy(e0);
        cudaEventDestroy(e1);
        cudaGetLastError();
        return -1.0;
    }
    float ms = 0.f;
    (void)cudaEventElapsedTime(&ms, e0, e1);
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    return (double)ms / (double)reps;
}

} // namespace

mc_status_t select_decode_gemv_tactics(const ModelImpl& m, SessionImpl& s) {
    // 进程内首个真实 session 测定一次；其余 session 直接复用（eager/graph
    // 与全部 session 同 kernel → bit 一致性）。
    {
        std::lock_guard<std::mutex> lk(m.gemv_mu);
        if (m.gemv_tactic_ready) return MC_OK;
    }

    // debug/基准开关：MC_GEMV_OFF=1 → 全部 shape 强制 cuBLASLt（对照组）。
    if (getenv("MC_GEMV_OFF") != nullptr) {
        std::lock_guard<std::mutex> lk(m.gemv_mu);
        for (int i = 0; i < 5; ++i) {
            m.gemv_tactic[i] = kGemvTacticCublas;
            m.gemv_split_s[i] = 1;
        }
        m.gemv_tactic_ready = true;
        fprintf(stderr, "[minicpm-native] gemv tactic: MC_GEMV_OFF set — all "
                        "decode GEMV on cuBLASLt (control group)\n");
        return MC_OK;
    }

    ShapeSpec shapes[5];
    decode_shape_table(shapes);
    // 计时缓冲：x = a.gate_up 行 0（宽 12288 ≥ K max 6144）；y = a.logits
    //（宽 vocab ≥ N max 130560）；split partial = a.gemv_partial。
    uint16_t* x = s.act.gate_up;
    uint16_t* y = s.act.logits;
    float* part = s.act.gemv_partial;
    const uint16_t* w[5] = {m.weights.qkv[0],       m.weights.o[0],
                            m.weights.gate_up[0],   m.weights.down[0],
                            m.weights.lm_head};
    // P4：fp8 双拷贝模式 —— shapes 0-3（qkv/o/gate_up/down）固定走自研
    // fp8 GEMV（无 cuBLASLt 对照）：快测只做功能校验（跑通 + 无 CUDA 错）
    // 并记录 GB/s；lm_head（shape 4）无 fp8 拷贝，照常 bf16 选型。
    const bool fp8 = m.weights.has_fp8;
    const uint8_t* w8[5] = {m.weights.fp8_qkv[0], m.weights.fp8_o[0],
                            m.weights.fp8_gate_up[0], m.weights.fp8_down[0],
                            nullptr};
    const float* w8s[5] = {m.weights.scale_qkv[0], m.weights.scale_o[0],
                           m.weights.scale_gate_up[0], m.weights.scale_down[0],
                           nullptr};
    (void)cudaMemsetAsync(x, 0, (size_t)cfg::kIntermediateSize * 2u * 2u, s.stream);

    // 计量轮数按 shape 流量自适应：大 shape 少跑几轮（lm_head 单次 ~0.6ms），
    // 小 shape 多跑（提高事件分辨率）。
    int32_t tactics[5] = {kGemvTacticCublas, kGemvTacticCublas, kGemvTacticCublas,
                          kGemvTacticCublas, kGemvTacticCublas};
    uint32_t splits[5] = {1, 1, 1, 1, 1};
    double best_ms[5] = {-1.0, -1.0, -1.0, -1.0, -1.0};

    for (int i = 0; i < 5; ++i) {
        const uint32_t N = shapes[i].n, K = shapes[i].k;
        const double wbytes = (double)N * (double)K * 2.0;
        const int reps = wbytes > 256e6 ? 10 : 24;
        const int warm = 3;

        if (fp8 && w8[i] != nullptr) {
            // 功能校验 + 计时（信息性 GB/s；tactic 固定 fp8）
            bool ok = true;
            for (int it = 0; it < warm + reps && ok; ++it) {
                if (launch_gemv_fp8(w8[i], w8s[i], x, y, N, K, s.stream) != MC_OK)
                    ok = false;
                if (cudaStreamSynchronize(s.stream) != cudaSuccess) ok = false;
            }
            if (ok) {
                // 事件计时一轮（功能轮已 warmup）
                cudaEvent_t e0, e1;
                if (cudaEventCreate(&e0) == cudaSuccess &&
                    cudaEventCreate(&e1) == cudaSuccess) {
                    (void)cudaEventRecord(e0, s.stream);
                    for (int it = 0; it < reps; ++it)
                        (void)launch_gemv_fp8(w8[i], w8s[i], x, y, N, K, s.stream);
                    (void)cudaEventRecord(e1, s.stream);
                    (void)cudaEventSynchronize(e1);
                    float ms = 0.f;
                    (void)cudaEventElapsedTime(&ms, e0, e1);
                    best_ms[i] = (double)ms / (double)reps;
                    cudaEventDestroy(e0);
                    cudaEventDestroy(e1);
                }
                tactics[i] = kGemvTacticFp8;
            } else {
                cudaGetLastError();
                mc::clear_error();
                tactics[i] = kGemvTacticCublas; // 功能异常 → 保底 bf16 cuBLASLt
                fprintf(stderr,
                        "[minicpm-native] gemv tactic: fp8 functional check FAILED "
                        "for shape %d -- falling back to bf16\n",
                        i);
            }
            continue;
        }

        const double t_cublas =
            time_engine(kGemvTacticCublas, 1, m, w[i], x, y, part, N, K, s.stream,
                        warm, reps);
        if (t_cublas >= 0.0) best_ms[i] = t_cublas;

        const double t_rows =
            time_engine(kGemvTacticRows, 1, m, w[i], x, y, part, N, K, s.stream,
                        warm, reps);
        // P5：rows 距 cuBLASLt ≤2% 即选 rows —— 自研 kernel 带 PDL wait/
        // trigger（图手术要求全链 wait-instrumented；cuBLASLt kernel 无法
        // 插桩），且 PDL 重叠下 rows 实际更快。lm_head 两者孤立速度差 <0.1%。
        const double t_ref = best_ms[i] > 0.0 ? best_ms[i] * 1.02 : -1.0;
        if (t_rows >= 0.0 &&
            (best_ms[i] < 0.0 || t_rows < best_ms[i] || t_rows <= t_ref)) {
            best_ms[i] = best_ms[i] < 0.0 ? t_rows : best_ms[i];
            tactics[i] = kGemvTacticRows;
        }

        const uint32_t S = gemv_pick_split_s(N, K);
        if (S > 1u && part != nullptr) {
            const double t_split =
                time_engine(kGemvTacticSplitK, S, m, w[i], x, y, part, N, K,
                            s.stream, warm, reps);
            if (t_split >= 0.0 && t_split < best_ms[i]) {
                best_ms[i] = t_split;
                tactics[i] = kGemvTacticSplitK;
                splits[i] = S;
            }
        }
    }

    // 清理快测产生的错误状态（计时失败已按保底处理），不污染 session。
    cudaGetLastError();
    mc::clear_error();

    std::lock_guard<std::mutex> lk(m.gemv_mu);
    for (int i = 0; i < 5; ++i) {
        m.gemv_tactic[i] = tactics[i];
        m.gemv_split_s[i] = splits[i];
    }
    m.gemv_tactic_ready = true;

    static const char* kShapeNames[5] = {"qkv", "o", "gate_up", "down", "lm_head"};
    for (int i = 0; i < 5; ++i) {
        // fp8 按 1B/元素计带宽（信息性；bf16 按 2B）
        const double bytes_per_elem = tactics[i] == kGemvTacticFp8 ? 1.0 : 2.0;
        const double gbs =
            best_ms[i] > 0.0
                ? (double)shapes[i].n * (double)shapes[i].k * bytes_per_elem /
                      (best_ms[i] * 1e6)
                : 0.0;
        const char* eng = tactics[i] == kGemvTacticCublas   ? "cublasLt"
                          : tactics[i] == kGemvTacticRows   ? "gemv-rows"
                          : tactics[i] == kGemvTacticSplitK ? "gemv-splitK"
                                                            : "gemv-fp8";
        fprintf(stderr,
                "[minicpm-native] gemv tactic %-8s N=%-6u K=%-4u -> %-11s "
                "(%.1f GB/s, %.3f ms)%s\n",
                kShapeNames[i], shapes[i].n, shapes[i].k, eng, gbs, best_ms[i],
                tactics[i] == kGemvTacticFp8 ? " [fp8 fixed tactic]" : "");
    }
    return MC_OK;
}

mc_status_t run_decode_gemv(const ModelImpl& m, float* split_partial, uint32_t N,
                            uint32_t K, const uint16_t* d_w, const uint16_t* d_x,
                            uint16_t* d_y, cudaStream_t stream, bool pdl) {
    const int si = decode_shape_index(N, K);
    if (si < 0 || m.gemv_tactic[si] == kGemvTacticCublas) {
        mc::set_error("run_decode_gemv: shape (%u,%u) not on gemv tactic", N, K);
        return MC_E_INTERNAL; // forward.cpp 只在 tactic≠0 时路由到此
    }
    return launch_decode_gemv(d_w, d_x, d_y, N, K, m.gemv_tactic[si],
                              m.gemv_split_s[si], split_partial, stream, pdl);
}

} // namespace mc
