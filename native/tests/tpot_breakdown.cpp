// tpot_breakdown.cpp — P5：decode TPOT 分段计时 harness（CTest: tpot_breakdown）。
//
// 产物 benchmark/results/tpot_breakdown.json：BF16 与 FP8 两模式 × ctx16/2048，
// 每个 arm 给出：
//   tpot_clean_ms     —— 无插桩 session 的 200 步平均 TPOT（graph replay）
//   tpot_instr_ms     —— MC_BREAKDOWN=1 session（图内含 298 个 event record
//                        节点）的 200 步平均 TPOT（插桩开销可见）
//   segments_ms       —— 10 桶段分解（emb / 7 层段 / lm_head / sample，
//                        42 层聚合，事件相邻差求和）
//   layer0/layer41    —— 首尾层单列（7 段各自均值）
//   unaccounted_ms    —— tpot_instr − Σsegments（D2H/D2D/replay/sync/host）
//   eager 段          —— 同一插桩 session 关闭 graph 的 eager 分解（launch
//                        开销归入段间隙的对照）
// 模型解析与 fp8_inference 一致（MC_MODEL_DIR / MC_FP8_MODEL_DIR > repo
// models > build fixture；fixture 生成共用 fp8_fixture.hpp，含 lm_head fp8 段）。
// 缺模型 → 醒目 SKIP+PASS。
#include "minicpm_runtime.h"

extern "C" void mc_debug_dump_set_steps(int steps);
extern "C" void mc_debug_session_set_graph_enabled(mc_session_t session, int enabled);
extern "C" int mc_debug_breakdown_raw_segments(mc_session_t session);
extern "C" int mc_debug_breakdown_bucket_count();
extern "C" const char* mc_debug_breakdown_bucket_name(int i);
extern "C" int mc_debug_breakdown_collect(mc_session_t session, double* out_ms, int n);

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "tests/fp8_fixture.hpp"

namespace fs = std::filesystem;

#define REQUIRE(cond)                                                          \
    do {                                                                       \
        if (!(cond)) {                                                          \
            fprintf(stderr, "FAIL %s:%d: %s (last_error: %s)\n", __FILE__,     \
                    __LINE__, #cond, mc_last_error());                         \
            std::exit(1);                                                       \
        }                                                                       \
    } while (0)

struct mc_model { void* impl; };
struct mc_session { void* impl; };

namespace {

bool file_exists(const std::string& p) {
    std::error_code ec;
    return fs::exists(p, ec);
}

std::string timestamp() {
    const time_t t = time(nullptr);
    char buf[32];
    strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%S", localtime(&t));
    return buf;
}

int gpu_util_now() {
    FILE* f = popen(
        "nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits "
        "2>/dev/null",
        "r");
    if (f == nullptr) return -1;
    int u = -1;
    if (fscanf(f, "%d", &u) != 1) u = -1;
    pclose(f);
    return u;
}

// fp8 包是否携带 lm_head dtype=2 段（决定 decode lm_head 走 gemv_fp8 与否）
bool fp8_pack_has_lm_head(const std::string& dir) {
    FILE* f = fopen((dir + "/model_fp8.wpk").c_str(), "rb");
    if (f == nullptr) return false;
    fp8fix::WpkHeaderIo hdr{};
    if (fread(&hdr, sizeof(hdr), 1, f) != 1) {
        fclose(f);
        return false;
    }
    std::vector<fp8fix::WpkSectionIo> toc(hdr.section_count);
    if (fseek(f, (long)hdr.toc_offset, SEEK_SET) != 0 ||
        fread(toc.data(), sizeof(fp8fix::WpkSectionIo), hdr.section_count, f) !=
            (size_t)hdr.section_count) {
        fclose(f);
        return false;
    }
    fclose(f);
    const uint64_t h = fp8fix::fnv1a64_io("lm_head");
    for (const auto& t : toc)
        if (t.name_hash == h && t.dtype == 2) return true;
    return false;
}

mc_model_options_t base_options(mc_precision_t prec) {
    mc_model_options_t o{};
    o.struct_size = sizeof(o);
    o.abi_version = MC_ABI_VERSION;
    o.device_id = 0;
    o.max_sessions = 16;
    o.max_context_tokens = 8448;
    o.precision = prec;
    o.kv_precision = MC_KV_BF16;
    o.enable_cuda_graph = 1;
    o.reserve_vram_bytes = 0;
    return o;
}

struct ArmResult {
    double tpot_clean_ms = -1.0;
    double tpot_graph_instr_ms = -1.0;
    double tpot_eager_instr_ms = -1.0;
    std::vector<double> graph_buckets_ms; // [10]（每步均值）
    std::vector<double> eager_buckets_ms; // [10]
    std::vector<double> layer0_ms;        // [7]
    std::vector<double> layer41_ms;       // [7]
    double graph_unaccounted_ms = 0.0;
};

// 一个插桩 session：graph 段分解 → eager 段分解（同一 session，先 graph 后关）
ArmResult run_breakdown(mc_model_t m, uint32_t ctx) {
    ArmResult r;
    const int nb = mc_debug_breakdown_bucket_count();
    const int nraw = 297; // mc::kBdownRawSegs（与库共同编译期常量一致）
    std::vector<double> bucket_sum(nb, 0.0), raw(nraw, 0.0);

    // ---- Phase 1：干净 TPOT（无插桩；MC_BREAKDOWN 建期读取，先 unset）----
    {
        unsetenv("MC_BREAKDOWN");
        mc_session_t s{nullptr};
        REQUIRE(mc_session_create(m, &s) == MC_OK);
        std::vector<int32_t> prompt(ctx);
        for (uint32_t i = 0; i < ctx; ++i)
            prompt[i] = (int32_t)((i * 2654435761u + 7u) % 130560u);
        REQUIRE(mc_prefill(s, prompt.data(), ctx) == MC_OK);
        mc_generation_config_t g{};
        g.struct_size = sizeof(g);
        g.max_new_tokens = 100000;
        g.temperature = 0.f;
        g.top_k = 0;
        g.seed = 1;
        int32_t tok = -1;
        for (int i = 0; i < 20; ++i) REQUIRE(mc_decode_one(s, &g, &tok) == MC_OK);
        mc_runtime_stats_t st0{}, st1{};
        REQUIRE(mc_get_stats(s, &st0) == MC_OK);
        for (int i = 0; i < 200; ++i) REQUIRE(mc_decode_one(s, &g, &tok) == MC_OK);
        REQUIRE(mc_get_stats(s, &st1) == MC_OK);
        r.tpot_clean_ms = (double)(st1.decode_us - st0.decode_us) / 1000.0 / 200.0;
        mc_session_destroy(s);
    }

    // ---- Phase 2：插桩 session（事件入图；graph → eager 两轮分解）----
    setenv("MC_BREAKDOWN", "1", 1);
    r.layer0_ms.assign(7, 0.0);
    r.layer41_ms.assign(7, 0.0);
    mc_session_t s{nullptr};
    REQUIRE(mc_session_create(m, &s) == MC_OK);
    REQUIRE(mc_debug_breakdown_raw_segments(s) == nraw);
    std::vector<int32_t> prompt(ctx);
    for (uint32_t i = 0; i < ctx; ++i)
        prompt[i] = (int32_t)((i * 2654435761u + 7u) % 130560u);
    REQUIRE(mc_prefill(s, prompt.data(), ctx) == MC_OK);
    mc_generation_config_t g{};
    g.struct_size = sizeof(g);
    g.max_new_tokens = 100000;
    g.temperature = 0.f;
    g.top_k = 0;
    g.seed = 1;
    int32_t tok = -1;

    auto run_phase = [&](bool graph_on, double& tpot_out,
                         std::vector<double>& buckets_out) {
        mc_debug_session_set_graph_enabled(s, graph_on ? 1 : 0);
        for (int i = 0; i < 20; ++i) REQUIRE(mc_decode_one(s, &g, &tok) == MC_OK);
        std::fill(bucket_sum.begin(), bucket_sum.end(), 0.0);
        mc_runtime_stats_t st0{}, st1{};
        REQUIRE(mc_get_stats(s, &st0) == MC_OK);
        for (int i = 0; i < 200; ++i) {
            REQUIRE(mc_decode_one(s, &g, &tok) == MC_OK);
            const int got = mc_debug_breakdown_collect(s, raw.data(), nraw);
            REQUIRE(got == nraw); // 完整 42 层链（prefill 后恒成立）
            for (int k = 0; k < nraw; ++k) {
                // 与库内 bdown_bucket 同构：0=emb；1..294 层段循环；295/296 尾段
                uint32_t b;
                if (k == 0) b = 0;
                else if (k == nraw - 2) b = (uint32_t)nb - 2;
                else if (k == nraw - 1) b = (uint32_t)nb - 1;
                else b = 1u + (uint32_t)((k - 1) % 7);
                bucket_sum[b] += raw[k];
            }
        }
        REQUIRE(mc_get_stats(s, &st1) == MC_OK);
        tpot_out = (double)(st1.decode_us - st0.decode_us) / 1000.0 / 200.0;
        buckets_out = bucket_sum; // 已是 200 步总和 → 调用方 /200
    };

    run_phase(true, r.tpot_graph_instr_ms, r.graph_buckets_ms);
    // 首尾层：重跑一轮 graph 只收集 layer0/layer41 原始段（1+1+7L+7 段）
    {
        std::fill(r.layer0_ms.begin(), r.layer0_ms.end(), 0.0);
        std::fill(r.layer41_ms.begin(), r.layer41_ms.end(), 0.0);
        for (int i = 0; i < 100; ++i) {
            REQUIRE(mc_decode_one(s, &g, &tok) == MC_OK);
            REQUIRE(mc_debug_breakdown_collect(s, raw.data(), nraw) == nraw);
            for (int j = 0; j < 7; ++j) {
                r.layer0_ms[j] += raw[1 + j];                     // layer0 段
                r.layer41_ms[j] += raw[1 + 41 * 7 + j];           // layer41 段
            }
        }
        for (int j = 0; j < 7; ++j) {
            r.layer0_ms[j] /= 100.0;
            r.layer41_ms[j] /= 100.0;
        }
    }
    run_phase(false, r.tpot_eager_instr_ms, r.eager_buckets_ms);
    for (auto& v : r.graph_buckets_ms) v /= 200.0;
    for (auto& v : r.eager_buckets_ms) v /= 200.0;
    double seg_sum = 0.0;
    for (auto v : r.graph_buckets_ms) seg_sum += v;
    r.graph_unaccounted_ms = r.tpot_graph_instr_ms - seg_sum;
    mc_session_destroy(s);
    return r;
}

void write_json(const std::string& path, const std::string& bf16_dir,
                const std::string& fp8_dir, bool lmhead8,
                const std::vector<ArmResult>& arms,
                const std::vector<std::string>& arm_tags, int util0) {
    const int nb = mc_debug_breakdown_bucket_count();
    std::ofstream f(path);
    if (!f) return;
    f << "{\n  \"timestamp\": \"" << timestamp() << "\",\n";
    f << "  \"gpu_util_at_start\": " << util0 << ",\n";
    f << "  \"note\": \"segments = per-step mean over 200 decode steps; graph "
         "arms replay CUDA graph with in-graph event nodes; eager arm = same "
         "instrumented session with graph dispatch off; unaccounted = tpot_instr "
         "- sum(segments) (D2H/D2D/replay/sync/host)\",\n";
    f << "  \"bf16_model_dir\": \"" << bf16_dir << "\",\n";
    f << "  \"fp8_model_dir\": \"" << fp8_dir << "\",\n";
    f << "  \"fp8_lm_head_quantized\": " << (lmhead8 ? "true" : "false") << ",\n";
    f << "  \"arms\": [\n";
    for (size_t a = 0; a < arms.size(); ++a) {
        const ArmResult& r = arms[a];
        f << "    {\"arm\": \"" << arm_tags[a] << "\",\n";
        f << "     \"tpot_clean_ms\": " << r.tpot_clean_ms << ",\n";
        f << "     \"tpot_graph_instr_ms\": " << r.tpot_graph_instr_ms << ",\n";
        f << "     \"tpot_eager_instr_ms\": " << r.tpot_eager_instr_ms << ",\n";
        f << "     \"graph_unaccounted_ms\": " << r.graph_unaccounted_ms << ",\n";
        f << "     \"graph_segments_ms\": {\n";
        for (int b = 0; b < nb; ++b)
            f << "       \"" << mc_debug_breakdown_bucket_name(b)
              << "\": " << r.graph_buckets_ms[b] << (b + 1 < nb ? "," : "") << "\n";
        f << "     },\n";
        f << "     \"eager_segments_ms\": {\n";
        for (int b = 0; b < nb; ++b)
            f << "       \"" << mc_debug_breakdown_bucket_name(b)
              << "\": " << r.eager_buckets_ms[b] << (b + 1 < nb ? "," : "") << "\n";
        f << "     },\n";
        f << "     \"layer0_segments_ms\": {";
        for (int j = 0; j < 7; ++j)
            f << "\"" << mc_debug_breakdown_bucket_name(1 + j) << "\": "
              << r.layer0_ms[j] << (j + 1 < 7 ? ", " : "");
        f << "},\n";
        f << "     \"layer41_segments_ms\": {";
        for (int j = 0; j < 7; ++j)
            f << "\"" << mc_debug_breakdown_bucket_name(1 + j) << "\": "
              << r.layer41_ms[j] << (j + 1 < 7 ? ", " : "");
        f << "}}" << (a + 1 < arms.size() ? "," : "") << "\n";
    }
    f << "  ]\n}\n";
}

} // namespace

int main() {
    fprintf(stderr, "tpot_breakdown: begin\n");
    mc_debug_dump_set_steps(0);
    const int util0 = gpu_util_now();

    // ---- 模型解析（与 fp8_inference 同序）----
    std::string bf16_dir;
    if (const char* env = getenv("MC_MODEL_DIR"))
        bf16_dir = env;
    else if (file_exists(std::string(MC_REPO_ROOT) +
                         "/models/minicpm5-2b/model.wpk"))
        bf16_dir = std::string(MC_REPO_ROOT) + "/models/minicpm5-2b";
    else if (file_exists(std::string(MC_BUILD_DIR) + "/wpk_model/model.wpk"))
        bf16_dir = std::string(MC_BUILD_DIR) + "/wpk_model";

    std::string fp8_dir;
    if (const char* env = getenv("MC_FP8_MODEL_DIR"))
        fp8_dir = env;
    else if (file_exists(std::string(MC_REPO_ROOT) +
                         "/models/minicpm5-2b/model_fp8.wpk"))
        fp8_dir = std::string(MC_REPO_ROOT) + "/models/minicpm5-2b";

    if (bf16_dir.empty() && fp8_dir.empty()) {
        fprintf(stderr,
                "== tpot_breakdown: SKIP — no model package (need model.wpk / "
                "model_fp8.wpk; set MC_MODEL_DIR / MC_FP8_MODEL_DIR) ==\n");
        return 0;
    }
    // fp8 需要同目录 bf16 源（双拷贝）
    if (!fp8_dir.empty() &&
        !file_exists(fp8_dir + "/model.wpk")) {
        fprintf(stderr,
                "== tpot_breakdown: fp8 dir %s lacks model.wpk — fp8 arm "
                "skipped ==\n",
                fp8_dir.c_str());
        fp8_dir.clear();
    }
    // bf16 缺但 fp8 在（其目录含 model.wpk）→ bf16 用同目录
    if (bf16_dir.empty() && !fp8_dir.empty()) bf16_dir = fp8_dir;

    std::vector<ArmResult> arms;
    std::vector<std::string> tags;
    bool lmhead8 = false;
    if (!fp8_dir.empty()) lmhead8 = fp8_pack_has_lm_head(fp8_dir);

    // ---- FP8 arm（先跑：显存高峰在 fp8 双拷贝）----
    if (!fp8_dir.empty()) {
        printf("== tpot_breakdown: FP8 mode (%s)%s ==\n", fp8_dir.c_str(),
               lmhead8 ? " [lm_head fp8]" : "");
        mc_model_t mf{nullptr};
        mc_model_options_t of = base_options(MC_FP8);
        REQUIRE(mc_model_load(fp8_dir.c_str(), &of, &mf) == MC_OK);
        arms.push_back(run_breakdown(mf, 16));
        tags.push_back("fp8_ctx16");
        arms.push_back(run_breakdown(mf, 2048));
        tags.push_back("fp8_ctx2048");
        mc_model_destroy(mf);
    }
    // ---- BF16 arm ----
    if (!bf16_dir.empty()) {
        printf("== tpot_breakdown: BF16 mode (%s) ==\n", bf16_dir.c_str());
        mc_model_t mb{nullptr};
        mc_model_options_t ob = base_options(MC_BF16);
        REQUIRE(mc_model_load(bf16_dir.c_str(), &ob, &mb) == MC_OK);
        arms.push_back(run_breakdown(mb, 16));
        tags.push_back("bf16_ctx16");
        arms.push_back(run_breakdown(mb, 2048));
        tags.push_back("bf16_ctx2048");
        mc_model_destroy(mb);
    }

    // ---- 报告 + JSON ----
    for (size_t a = 0; a < arms.size(); ++a) {
        const ArmResult& r = arms[a];
        double seg_sum = 0;
        for (auto v : r.graph_buckets_ms) seg_sum += v;
        printf("  %-14s clean %7.3f ms | instr(graph) %7.3f ms | eager %7.3f ms | "
               "segs %7.3f | unacc %6.3f\n",
               tags[a].c_str(), r.tpot_clean_ms, r.tpot_graph_instr_ms,
               r.tpot_eager_instr_ms, seg_sum, r.graph_unaccounted_ms);
        printf("      graph segs:");
        for (int b = 0; b < mc_debug_breakdown_bucket_count(); ++b)
            printf(" %s=%.3f", mc_debug_breakdown_bucket_name(b),
                   r.graph_buckets_ms[b]);
        printf("\n");
    }
    const std::string dir = std::string(MC_REPO_ROOT) + "/benchmark/results";
    std::error_code ec;
    fs::create_directories(dir, ec);
    write_json(dir + "/tpot_breakdown.json", bf16_dir, fp8_dir, lmhead8, arms,
               tags, util0);
    printf("  results written: %s/tpot_breakdown.json\n", dir.c_str());
    fprintf(stderr, "tpot_breakdown: done\n");
    return 0;
}
