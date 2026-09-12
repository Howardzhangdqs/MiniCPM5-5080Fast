// graph_bench.cpp — P3 CUDA Graph A/B 测试（CTest: graph_bench）。
//
// 前置：model.wpk（MC_MODEL_DIR > 仓库 models/minicpm5-2b > 构建目录
// wpk_model）+ golden 目录（MC_GOLDEN_DIR > 仓库 tests/runtime/golden）。
// 缺任一 → FAIL（return 1）：graph A/B 无权重/无 golden 无从验证。
//
// 内容：
//  ① 正确性：三个 golden case 各跑两遍（同一 model：session A 用
//     mc_debug_session_set_graph_enabled(0) 走 eager，session B 走 graph
//     replay）。断言：graph 与 eager 的 decode 发放 token 序列逐位一致
//     （bit-exact；两模式跑的是同一批 kernel、同一地址、同一顺序——
//     greedy 确定性下 token 一致即数值一致）。eager vs golden 复用
//     tests/golden_compare.h 的分叉豁免策略（gap ≤ max(4×bf16ulp,0.25)
//     且该步 logits cos≥0.99 → 合法翻转；case_mixed step6 为 P2 已接受
//     的 documented flip）。发放语义：decode 调用 k 发放 step{k-1} 的
//     决策（prefill argmax = step0.token），故 S 次调用覆盖全部 S 个
//     golden token。
//  ② 性能：ctx=16/2048/8192 三组，各 20 步预热 + 200 步计量（stats 的
//     decode_us 差分），输出 Graph Off/On TPOT（ms 与 tok/s）与加速比，
//     写入 benchmark/results/graph_ab.json（时间戳/ctx/模式/TPOT/加速比/
//     graph 节点数）。
//  ③ 纪律断言：graph session 的 stats.cuda_graph_launches == 220（20+200）；
//     eager session == 0。
#include "minicpm_runtime.h"

// additive debug 钩子（libminicpm_native 导出；不属于冻结 ABI）。
extern "C" void mc_debug_dump_set_steps(int steps);
extern "C" void mc_debug_session_set_graph_enabled(mc_session_t session, int enabled);
extern "C" size_t mc_debug_session_graph_nodes(mc_session_t session);

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "tests/golden_compare.h"

namespace fs = std::filesystem;

#define REQUIRE(cond)                                                          \
    do {                                                                       \
        if (!(cond)) {                                                         \
            fprintf(stderr, "FAIL %s:%d: %s (last_error: %s)\n", __FILE__,     \
                    __LINE__, #cond, mc_last_error());                         \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

struct mc_model { void* impl; };
struct mc_session { void* impl; };

namespace {

using golden_cmp::read_i32_file;

bool file_exists(const std::string& p) {
    std::error_code ec;
    return fs::exists(p, ec);
}

// 与 real_inference 相同的目录解析（MC_MODEL_DIR > repo models > build wpk）。
std::string resolve_model_dir() {
    if (const char* env = getenv("MC_MODEL_DIR")) return env;
    if (file_exists(std::string(MC_REPO_ROOT) + "/models/minicpm5-2b/model.wpk"))
        return std::string(MC_REPO_ROOT) + "/models/minicpm5-2b";
    if (file_exists(std::string(MC_BUILD_DIR) + "/wpk_model/model.wpk"))
        return std::string(MC_BUILD_DIR) + "/wpk_model";
    return "";
}

std::string resolve_golden_dir() {
    if (const char* env = getenv("MC_GOLDEN_DIR")) return env;
    const std::string repo = std::string(MC_REPO_ROOT) + "/tests/runtime/golden";
    if (file_exists(repo)) return repo;
    return "";
}

// 跑一个 golden case 的 S 步 decode，返回发放的 token 序列（长度 S：
// 调用 k 发放 step{k-1} 的决策，S 次覆盖 step0..step{S-1}）。
// dump_steps > 0 时启用层级 dump（eager 基线取证：分叉步 logits 供
// golden_cmp::evaluate_divergence 认证；graph replay 不产 dump）。
std::vector<int32_t> run_case(mc_model_t m, const std::vector<int32_t>& ids,
                              uint32_t steps, bool graph_on, int dump_steps = 0) {
    mc_session_t s{nullptr};
    REQUIRE(mc_session_create(m, &s) == MC_OK);
    mc_debug_session_set_graph_enabled(s, graph_on ? 1 : 0);
    mc_debug_dump_set_steps(dump_steps);
    REQUIRE(mc_prefill(s, ids.data(), (uint32_t)ids.size()) == MC_OK);

    mc_generation_config_t g{};
    g.struct_size = sizeof(g);
    g.max_new_tokens = steps + 8; // 不因 force-eos 截断
    g.temperature = 0.f; // golden 对拍必须 greedy 分支（激活规则见 decode_one）
    g.top_k = 0;
    g.seed = 1;
    std::vector<int32_t> toks;
    int32_t tok = -1;
    for (uint32_t k = 0; k < steps; ++k) {
        REQUIRE(mc_decode_one(s, &g, &tok) == MC_OK);
        toks.push_back(tok);
    }
    mc_debug_dump_set_steps(0);
    mc_session_destroy(s);
    return toks;
}

// 性能：prefill(prompt) → 20 步预热 → 200 步计量，返回 TPOT(ms) 与
// 计量段 cuda_graph_launches。固定 prompt（确定性 LCG），各 ctx 独立 session。
constexpr int kPerfWarmSteps = 20;
constexpr int kPerfMeasuredSteps = 200;

struct PerfResult { double tpot_ms; uint64_t graph_launches; };

PerfResult run_perf(mc_model_t m, uint32_t ctx, bool graph_on) {
    mc_session_t s{nullptr};
    REQUIRE(mc_session_create(m, &s) == MC_OK);
    mc_debug_session_set_graph_enabled(s, graph_on ? 1 : 0);

    std::vector<int32_t> prompt(ctx);
    for (uint32_t i = 0; i < ctx; ++i)
        prompt[i] = (int32_t)((i * 2654435761u + 7u) % 130560u);
    REQUIRE(mc_prefill(s, prompt.data(), ctx) == MC_OK);

    mc_generation_config_t g{};
    g.struct_size = sizeof(g);
    g.max_new_tokens = 100000; // 计量段绝不触发 force-eos
    g.seed = 1;
    g.temperature = 0.f; // 显式 greedy 分支（激活规则：temp ≤0 或 top_k ≤1；
    g.top_k = 0;         // 零初始化字段写明，防未来默认值变化）
    int32_t tok = -1;
    for (int i = 0; i < kPerfWarmSteps; ++i)
        REQUIRE(mc_decode_one(s, &g, &tok) == MC_OK);
    mc_runtime_stats_t st0{};
    REQUIRE(mc_get_stats(s, &st0) == MC_OK);
    for (int i = 0; i < kPerfMeasuredSteps; ++i)
        REQUIRE(mc_decode_one(s, &g, &tok) == MC_OK);
    mc_runtime_stats_t st1{};
    REQUIRE(mc_get_stats(s, &st1) == MC_OK);
    REQUIRE(st1.generated_tokens ==
            st0.generated_tokens + (uint64_t)kPerfMeasuredSteps);
    // 纪律：graph session 每步恰一次 cudaGraphLaunch；eager 恒 0
    const uint64_t launches = st1.cuda_graph_launches - st0.cuda_graph_launches;
    REQUIRE(launches == (graph_on ? (uint64_t)kPerfMeasuredSteps : 0ull));
    if (graph_on)
        REQUIRE(st0.cuda_graph_launches == (uint64_t)kPerfWarmSteps); // 预热段已计

    PerfResult r;
    r.tpot_ms = (double)(st1.decode_us - st0.decode_us) / 1000.0 /
                (double)kPerfMeasuredSteps;
    r.graph_launches = st1.cuda_graph_launches - st0.cuda_graph_launches;
    mc_session_destroy(s);
    return r;
}

std::string timestamp() {
    const auto now = std::chrono::system_clock::now();
    const std::time_t t = std::chrono::system_clock::to_time_t(now);
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%S", std::localtime(&t));
    return buf;
}

} // namespace

int main() {
    fprintf(stderr, "graph_bench: begin\n");
    // dump env 必须在任何 dump 配置访问前设置（静态缓存一次）。默认步数 0
    //（关闭）；golden 段的 eager 基线按 case 临时开启以取证分叉步 logits。
    const std::string dump_dir = std::string(MC_BUILD_DIR) + "/graph_bench_dump";
    std::error_code ec;
    fs::remove_all(dump_dir, ec);
    fs::create_directories(dump_dir, ec);
    setenv("MC_DEBUG_DUMP_DIR", dump_dir.c_str(), 1);
    setenv("MC_DEBUG_DUMP_STEPS", "0", 1);
    mc_debug_dump_set_steps(0);

    const std::string model_dir = resolve_model_dir();
    if (model_dir.empty() || !file_exists(model_dir + "/model.wpk")) {
        fprintf(stderr, "graph_bench: FAIL — model.wpk not found\n");
        return 1;
    }
    const std::string golden_dir = resolve_golden_dir();
    if (golden_dir.empty()) {
        fprintf(stderr, "graph_bench: FAIL — golden dir not found\n");
        return 1;
    }

    // 单一 model 载入（权重 ~4.7GB 显存，A/B 共享；session 级切 graph 开关）
    mc_model_t m{nullptr};
    mc_model_options_t o{};
    o.struct_size = sizeof(o);
    o.abi_version = MC_ABI_VERSION;
    o.device_id = 0;
    o.max_sessions = 16;
    o.max_context_tokens = 8448; // 覆盖 8192+20+200 与 golden case
    o.precision = MC_BF16;
    o.kv_precision = MC_KV_BF16;
    o.enable_cuda_graph = 1; // A/B 的前提
    o.reserve_vram_bytes = 0;
    REQUIRE(mc_model_load(model_dir.c_str(), &o, &m) == MC_OK);

    size_t graph_nodes = 0;
    int failures = 0;
    bool sampling_smoke = false;

    // ================= ① golden 正确性：graph vs eager vs golden =================
    // 判定策略（复用 tests/golden_compare.h，与 real_inference 一致）：
    //   - graph vs eager：逐 step bit-exact（硬性 —— P3 的核心正确性要求）；
    //   - eager vs golden：全一致，或首个失配步经 evaluate_divergence 认证为
    //     近平局合法翻转（gap ≤ max(4×bf16ulp(|top1|), 0.25) 且该步 logits
    //     cos ≥ 0.99；翻转后的后续 step 为确定性分叉，豁免）。case_mixed
    //     step6 是 P2 已接受的 documented flip（gap=0.125）。
    printf("== graph_bench: golden A/B (graph vs eager vs golden tokens) ==\n");
    struct GoldenRow {
        const char* case_name;
        bool bit_exact;
        bool golden_ok;
        bool had_flip;
    };
    std::vector<GoldenRow> golden_rows;

    std::vector<std::string> cases;
    for (const auto& e : fs::directory_iterator(golden_dir))
        if (e.is_directory() && file_exists(e.path().string() + "/input_ids.bin"))
            cases.push_back(e.path().filename().string());
    std::sort(cases.begin(), cases.end());
    REQUIRE(!cases.empty());

    for (const auto& cname : cases) {
        const std::string cdir = golden_dir + "/" + cname;
        const auto ids = read_i32_file(cdir + "/input_ids.bin");
        REQUIRE(!ids.empty());
        uint32_t steps = 0;
        for (uint32_t t = 0;; ++t) {
            char buf[64];
            snprintf(buf, sizeof(buf), "/step%u.token", t);
            if (!file_exists(cdir + buf)) break;
            steps = t + 1;
        }
        REQUIRE(steps > 0);
        std::vector<int32_t> golden_toks;
        for (uint32_t t = 0; t < steps; ++t) {
            char buf[64];
            snprintf(buf, sizeof(buf), "/step%u.token", t);
            const auto v = read_i32_file(cdir + buf);
            REQUIRE(v.size() == 1);
            golden_toks.push_back(v[0]);
        }

        // eager 基线（开 dump 取证 logits）；graph 重跑（不 dump）
        const auto toks_eager =
            run_case(m, ids, steps, /*graph_on=*/false, /*dump_steps=*/(int)steps + 1);
        const auto toks_graph = run_case(m, ids, steps, /*graph_on=*/true);
        REQUIRE(toks_eager.size() == steps && toks_graph.size() == steps);

        bool bit_exact = true;
        int first_diff = -1;
        for (uint32_t k = 0; k < steps; ++k) {
            if (toks_graph[k] != toks_eager[k]) {
                bit_exact = false;
                if (first_diff < 0) first_diff = (int)k;
            }
        }

        // eager vs golden：全一致 或 单次合法翻转（翻转步后豁免）
        bool golden_ok = true;
        bool had_flip = false;
        int t_div = -1;
        for (uint32_t k = 0; k < steps; ++k) {
            if (toks_eager[k] != golden_toks[k]) {
                t_div = (int)k;
                break;
            }
        }
        if (t_div >= 0) {
            const auto div = golden_cmp::evaluate_divergence(
                cdir, dump_dir, t_div, golden_toks[t_div], toks_eager[t_div]);
            had_flip = true;
            if (!div.evaluated) {
                golden_ok = false;
                printf("  case %s: divergence @step%d NOT CERTIFIABLE (golden "
                       "logits unreadable)\n",
                       cname.c_str(), t_div);
            } else if (div.legitimate) {
                printf("  case %s: divergence @step%d golden=%d got=%d — top1 "
                       "%.4f / top2 %.4f gap=%.4f <= th %.4f, logits cos=%.6f "
                       "—> LEGITIMATE FLIP (steps>%d exempt; same policy as "
                       "real_inference)\n",
                       cname.c_str(), t_div, div.golden_tok, div.got_tok, div.top1,
                       div.top2, div.gap, div.threshold, div.logits_cos, t_div);
            } else {
                golden_ok = false;
                printf("  case %s: divergence @step%d golden=%d got=%d — gap=%.4f "
                       "vs th %.4f, logits cos=%.6f —> NOT excusable\n",
                       cname.c_str(), t_div, div.golden_tok, div.got_tok, div.gap,
                       div.threshold, div.logits_cos);
            }
        }
        if (!bit_exact || !golden_ok) failures++;
        printf("  case %-10s steps=%-3u graph==eager: %s", cname.c_str(), steps,
               bit_exact ? "BIT-EXACT" : "DIFF");
        if (!bit_exact) printf(" (first diff @step%d: eager=%d graph=%d)", first_diff,
                               toks_eager[first_diff], toks_graph[first_diff]);
        printf(" | vs golden: %s%s\n", golden_ok ? "OK" : "MISMATCH",
               had_flip ? " (1 legitimate flip)" : " (all match)");
        golden_rows.push_back({cname.c_str(), bit_exact, golden_ok, had_flip});
        // 清理本 case 的 dump 取证（flat 暂存区）
        for (const auto& e : fs::directory_iterator(dump_dir))
            if (e.is_regular_file(ec)) fs::remove(e.path(), ec);
    }
    REQUIRE(failures == 0);

    // ================= ② 性能 A/B：ctx=16/2048/8192 =================
    printf("== graph_bench: TPOT A/B (20 warmup + 200 measured) ==\n");
    struct PerfRow { uint32_t ctx; PerfResult off, on; double speedup; };
    std::vector<PerfRow> perf_rows;
    for (uint32_t ctx : {16u, 2048u, 8192u}) {
        // 先各跑一次 off/on（off 在前，散热条件等同；共享 GPU 公平性靠背靠背）
        const PerfResult off = run_perf(m, ctx, /*graph_on=*/false);
        const PerfResult on = run_perf(m, ctx, /*graph_on=*/true);
        REQUIRE(off.graph_launches == 0);
        REQUIRE(on.graph_launches == (uint64_t)kPerfMeasuredSteps);
        const double speedup = off.tpot_ms / on.tpot_ms;
        perf_rows.push_back({ctx, off, on, speedup});
        printf("  ctx=%-4u Graph Off: %7.3f ms/tok (%6.1f tok/s) | Graph On: "
               "%7.3f ms/tok (%6.1f tok/s) | speedup %.2fx\n",
               ctx, off.tpot_ms, 1000.0 / off.tpot_ms, on.tpot_ms,
               1000.0 / on.tpot_ms, speedup);
    }
    if (graph_nodes == 0) {
        // 节点数统一取证一次（graph 在 create 期捕获，各 session 相同）
        mc_session_t s{nullptr};
        REQUIRE(mc_session_create(m, &s) == MC_OK);
        graph_nodes = mc_debug_session_graph_nodes(s);
        mc_session_destroy(s);
    }
    REQUIRE(graph_nodes > 0);
    printf("  decode graph nodes: %zu\n", graph_nodes);

    // ================= ④ 采样冒烟（§14.2：temperature/top-k/top-p/rep-penalty）==
    // 无 golden：验证不崩、token 合法、eager-vs-graph bit 一致、同 seed 可
    // 复现 / 异 seed 发散、TPOT 相对 greedy 无明显退步（<5%）。
    {
        printf("== graph_bench: sampling smoke (temp=0.7 top_p=0.9 k=128 rp=1.15) "
               "==\n");
        const uint32_t prompt_n = 64;
        std::vector<int32_t> prompt(prompt_n);
        for (uint32_t i = 0; i < prompt_n; ++i)
            prompt[i] = (int32_t)((i * 2654435761u + 13u) % 130560u);
        auto samp_cfg = [&](uint64_t seed) {
            mc_generation_config_t g{};
            g.struct_size = sizeof(g);
            g.max_new_tokens = 100000;
            g.temperature = 0.7f;
            g.top_p = 0.9f;
            g.top_k = 128u; // >kSampMaxK? 128 = 上限内；显式开采样分支
            g.repetition_penalty = 1.15f;
            g.seed = seed;
            return g;
        };
        auto run_sampling = [&](bool graph_on, uint64_t seed, int steps,
                                std::vector<int32_t>* toks) {
            mc_session_t s{nullptr};
            REQUIRE(mc_session_create(m, &s) == MC_OK);
            mc_debug_session_set_graph_enabled(s, graph_on ? 1 : 0);
            mc_debug_dump_set_steps(0);
            REQUIRE(mc_prefill(s, prompt.data(), prompt_n) == MC_OK);
            mc_generation_config_t g = samp_cfg(seed);
            int32_t tok = -1;
            if (toks != nullptr) toks->clear();
            for (int k = 0; k < steps; ++k) {
                REQUIRE(mc_decode_one(s, &g, &tok) == MC_OK);
                REQUIRE(tok >= 0 && tok < 130560); // token 合法
                if (toks != nullptr) toks->push_back(tok);
            }
            mc_session_destroy(s);
        };
        // ① eager vs graph：同 seed 32 步 bit 一致（两模式同 kernel + 同 RNG）
        std::vector<int32_t> te, tg;
        run_sampling(false, 1234u, 32, &te);
        run_sampling(true, 1234u, 32, &tg);
        REQUIRE(te == tg);
        printf("  eager==graph (sampling): BIT-EXACT (%d steps)\n", (int)te.size());
        // ② 确定性：同 seed 重跑一致；异 seed 发散
        std::vector<int32_t> tg2, td;
        run_sampling(true, 1234u, 32, &tg2);
        run_sampling(true, 4321u, 32, &td);
        REQUIRE(tg == tg2);
        REQUIRE(tg != td);
        printf("  rng determinism: same-seed identical, diff-seed diverges\n");
        // ②b OpenAI 语义：top_k 未指定(0) + temp>0 → 采样激活（k 规范化为
        //    上限），token 合法且同 seed 确定性
        {
            auto run_k0 = [&](uint64_t seed, std::vector<int32_t>* toks) {
                mc_session_t s{nullptr};
                REQUIRE(mc_session_create(m, &s) == MC_OK);
                mc_debug_dump_set_steps(0);
                REQUIRE(mc_prefill(s, prompt.data(), prompt_n) == MC_OK);
                mc_generation_config_t g{};
                g.struct_size = sizeof(g);
                g.max_new_tokens = 100000;
                g.temperature = 0.7f;
                g.top_p = 0.9f;
                g.top_k = 0u; // 未指定 → host 侧规范化
                g.seed = seed;
                int32_t tok = -1;
                toks->clear();
                for (int k = 0; k < 16; ++k) {
                    REQUIRE(mc_decode_one(s, &g, &tok) == MC_OK);
                    REQUIRE(tok >= 0 && tok < 130560);
                    toks->push_back(tok);
                }
                mc_session_destroy(s);
            };
            std::vector<int32_t> tk0a, tk0b;
            run_k0(99u, &tk0a);
            run_k0(99u, &tk0b);
            REQUIRE(!tk0a.empty());
            REQUIRE(tk0a == tk0b);
            printf("  top_k unspecified (0): sampling active, deterministic "
                   "(%zu toks)\n",
                   tk0a.size());
        }
        // ③ TPOT：同 session 先 greedy 后采样（ctx=16 等价的小 prompt 已就绪
        //    —— 用新 session 各测一次，背靠背）
        auto tpot_of = [&](bool sampling) {
            mc_session_t s{nullptr};
            REQUIRE(mc_session_create(m, &s) == MC_OK);
            REQUIRE(mc_prefill(s, prompt.data(), prompt_n) == MC_OK);
            int32_t tok = -1;
            if (sampling) {
                mc_generation_config_t g = samp_cfg(777u);
                for (int i = 0; i < 20; ++i) REQUIRE(mc_decode_one(s, &g, &tok) == MC_OK);
                mc_runtime_stats_t s0{}, s1{};
                REQUIRE(mc_get_stats(s, &s0) == MC_OK);
                for (int i = 0; i < 200; ++i) REQUIRE(mc_decode_one(s, &g, &tok) == MC_OK);
                REQUIRE(mc_get_stats(s, &s1) == MC_OK);
                mc_session_destroy(s);
                return (double)(s1.decode_us - s0.decode_us) / 1000.0 / 200.0;
            }
            mc_generation_config_t gg{};
            gg.struct_size = sizeof(gg);
            gg.max_new_tokens = 100000;
            gg.temperature = 0.f; // greedy 分支
            gg.seed = 1;
            for (int i = 0; i < 20; ++i) REQUIRE(mc_decode_one(s, &gg, &tok) == MC_OK);
            mc_runtime_stats_t s0{}, s1{};
            REQUIRE(mc_get_stats(s, &s0) == MC_OK);
            for (int i = 0; i < 200; ++i) REQUIRE(mc_decode_one(s, &gg, &tok) == MC_OK);
            REQUIRE(mc_get_stats(s, &s1) == MC_OK);
            mc_session_destroy(s);
            return (double)(s1.decode_us - s0.decode_us) / 1000.0 / 200.0;
        };
        const double t_greedy = tpot_of(false);
        const double t_samp = tpot_of(true);
        const double ratio = t_samp / t_greedy;
        printf("  TPOT ctx=64: greedy %.3f ms | sampling %.3f ms | ratio %.3f "
               "(<1.05 required)\n",
               t_greedy, t_samp, ratio);
        REQUIRE(ratio < 1.05);
        sampling_smoke = true;
    }

    // ================= ③ 写 benchmark/results/graph_ab.json =================
    {
        const std::string dir = std::string(MC_REPO_ROOT) + "/benchmark/results";
        std::error_code ec;
        fs::create_directories(dir, ec);
        const std::string path = dir + "/graph_ab.json";
        std::ofstream f(path);
        if (!f) {
            fprintf(stderr, "graph_bench: FAIL — cannot write %s\n", path.c_str());
            return 1;
        }
        f << "{\n";
        f << "  \"timestamp\": \"" << timestamp() << "\",\n";
        f << "  \"model_dir\": \"" << model_dir << "\",\n";
        f << "  \"graph_nodes\": " << graph_nodes << ",\n";
        f << "  \"sampling_smoke_passed\": " << (sampling_smoke ? "true" : "false")
          << ",\n";
        f << "  \"golden\": [\n";
        for (size_t i = 0; i < golden_rows.size(); ++i) {
            const auto& r = golden_rows[i];
            f << "    {\"case\": \"" << r.case_name
              << "\", \"bit_exact_vs_eager\": " << (r.bit_exact ? "true" : "false")
              << ", \"tokens_match_golden\": " << (r.golden_ok ? "true" : "false")
              << ", \"legitimate_flips\": " << (r.had_flip ? 1 : 0)
              << "}" << (i + 1 < golden_rows.size() ? "," : "") << "\n";
        }
        f << "  ],\n";
        f << "  \"perf\": [\n";
        for (size_t i = 0; i < perf_rows.size(); ++i) {
            const auto& r = perf_rows[i];
            f << "    {\"ctx\": " << r.ctx
              << ", \"tpot_ms_graph_off\": " << r.off.tpot_ms
              << ", \"tok_per_s_graph_off\": " << 1000.0 / r.off.tpot_ms
              << ", \"tpot_ms_graph_on\": " << r.on.tpot_ms
              << ", \"tok_per_s_graph_on\": " << 1000.0 / r.on.tpot_ms
              << ", \"speedup\": " << r.speedup
              << "}" << (i + 1 < perf_rows.size() ? "," : "") << "\n";
        }
        f << "  ]\n";
        f << "}\n";
        printf("  results written: %s\n", path.c_str());
    }

    mc_model_destroy(m);
    fprintf(stderr, "graph_bench: all done (failures=%d)\n", failures);
    return failures == 0 ? 0 : 1;
}
