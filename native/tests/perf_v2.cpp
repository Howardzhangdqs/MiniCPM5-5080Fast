// perf_v2.cpp — P5 验收基准：空闲 GPU 背靠背 5 轮 × 200 token 取中位数。
//
// 纪律（任务书）：测前查 nvidia-smi util，被占用（>5%）时等待重试；
// 每轮独立 20 预热 + 200 计量（graph 模式，统计 decode_us 口径同 fp8_e2e）；
// 产物 benchmark/results/perf_v2.json：FP8 ctx16/2048/8192 + BF16 同步报告，
// 含每轮原始值、中位数、tok/s 与环境 util 注记。
// 模型解析同 tpot_breakdown（MC_FP8_MODEL_DIR / MC_MODEL_DIR > repo models）。
//
// P8 长上下文扫档（env MC_PERF_CTXS 设置且合法时进入 sweep 模式）：
//   - MC_PERF_CTXS：逗号分隔 ctx 列表（uint32，升序排序去重）；
//   - MC_PERF_OUT_TOKENS（默认 200）/ MC_PERF_ROUNDS（默认 5）/
//     MC_PERF_ROUNDS_BIG（默认 3，ctx ≥ 65536 的档位使用）；
//   - max_context_tokens = min(max(ctxs)+512, 模型上限 kMaxContextTokens)；
//   - 输出写 MC_PERF_OUT（默认 benchmark/results/perf_sweep.json），每行含
//     mode/ctx/out_tokens/rounds_ms 原始值/mean_ms/median_ms/tok_per_s
//    （mean 与 median 各一）/util_before_pct；kBaseMs 基线对照段仅对默认
//     三档有效（sweep 跳过）；printf 轮次循环打印；
//   - 单个 ctx 档位失败（如 OOM/显存不足）记 {"error": "..."} 行后继续下一档。
#include "minicpm_runtime.h"
#include "runtime/generated_model_config.h" // kMaxContextTokens（容量钳制）

extern "C" void mc_debug_dump_set_steps(int steps);

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

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

// P8-A2 additive debug 钩子（不在公共头；同 soak_test.cpp 的本地声明方式）：
// 本会话的 decode attention chunk 档位（64/128/256）。
extern "C" uint32_t mc_debug_session_attn_chunk(mc_session_t session);

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

// 等待空闲（util ≤ 5%），最多 ~10 分钟；返回等待后的 util（-1 = 查询失败，按空闲继续）
int wait_idle(int limit_pct = 5, int tries = 120) {
    for (int i = 0; i < tries; ++i) {
        const int u = gpu_util_now();
        if (u < 0 || u <= limit_pct) return u;
        if (i % 10 == 0)
            fprintf(stderr, "[perf_v2] GPU busy (%d%%) — waiting...\n", u);
        struct timespec ts {0, 5000 * 1000 * 1000}; // 5s
        nanosleep(&ts, nullptr);
    }
    return gpu_util_now();
}

// ---- P8 env 解析（非法值回退默认并 stderr 警告；同 runtime 的纪律）----

uint32_t env_u32(const char* name, uint32_t defv) {
    if (const char* v = getenv(name)) {
        char* end = nullptr;
        const unsigned long parsed = strtoul(v, &end, 10);
        if (end != v && *end == '\0' && parsed >= 1 && parsed <= 100000000ul)
            return (uint32_t)parsed;
        fprintf(stderr, "[perf_v2] %s=\"%s\" invalid -- using default %u\n",
                name, v, defv);
    }
    return defv;
}

// 逗号分隔 ctx 列表（容忍空格）；任一元素非法 → 整体非法（false）。
// 成功时升序排序去重。
bool parse_ctx_list(const char* s, std::vector<uint32_t>& out) {
    out.clear();
    std::string str(s);
    size_t start = 0;
    for (;;) {
        const size_t comma = str.find(',', start);
        std::string part = str.substr(
            start, comma == std::string::npos ? std::string::npos : comma - start);
        // 去首尾空格
        const size_t b = part.find_first_not_of(" \t");
        if (b == std::string::npos) return false;
        const size_t epos = part.find_last_not_of(" \t");
        part = part.substr(b, epos - b + 1);
        char* end = nullptr;
        const unsigned long v = strtoul(part.c_str(), &end, 10);
        if (*end != '\0' || v == 0 || v > 100000000ul) return false;
        out.push_back((uint32_t)v);
        if (comma == std::string::npos) break;
        start = comma + 1;
    }
    if (out.empty()) return false;
    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
    return true;
}

// P8：max_context_tokens = min(max(ctxs) + 512, 模型上限)。默认三档时
// = 8192+512 = 8448（与旧硬编码一致）；131072 顶格档位由钳制保住
// model_load 的 ABI 校验（该档 prefill 后 decode 无余量，属正常失败路径）。
uint32_t context_cap(const std::vector<uint32_t>& ctxs) {
    uint32_t mx = 0;
    for (uint32_t c : ctxs) mx = c > mx ? c : mx;
    const uint32_t want = mx + 512u;
    return want > mc::cfg::kMaxContextTokens ? mc::cfg::kMaxContextTokens : want;
}

mc_model_options_t base_options(mc_precision_t prec, uint32_t max_ctx) {
    mc_model_options_t o{};
    o.struct_size = sizeof(o);
    o.abi_version = MC_ABI_VERSION;
    o.device_id = 0;
    o.max_sessions = 16;
    o.max_context_tokens = max_ctx;
    o.precision = prec;
    o.kv_precision = MC_KV_BF16;
    o.enable_cuda_graph = 1;
    o.reserve_vram_bytes = 0;
    return o;
}

// 一轮的成败结果（P8：单档失败不中止整个 run，错误上抛为行级记录）。
struct RoundOutcome {
    bool ok = false;
    double ms = 0.0;
    std::string error;
};

// 一轮：prefill + 20 预热 + out_tokens 计量，返回 TPOT ms（或失败原因）。
// chunk_out 非 null 时回报本 session 的 attention chunk 档位（P8-A2）。
RoundOutcome run_round(mc_model_t m, uint32_t ctx, uint32_t out_tokens,
                       uint32_t* chunk_out = nullptr) {
    RoundOutcome oc;
    mc_session_t s{nullptr};
    if (mc_session_create(m, &s) != MC_OK) {
        oc.error = mc_last_error();
        return oc;
    }
    if (chunk_out) *chunk_out = mc_debug_session_attn_chunk(s);
    std::vector<int32_t> prompt(ctx);
    for (uint32_t i = 0; i < ctx; ++i)
        prompt[i] = (int32_t)((i * 2654435761u + 7u) % 130560u);
    if (mc_prefill(s, prompt.data(), ctx) != MC_OK) {
        oc.error = mc_last_error();
        mc_session_destroy(s);
        return oc;
    }
    mc_generation_config_t g{};
    g.struct_size = sizeof(g);
    g.max_new_tokens = 100000;
    g.temperature = 0.f;
    g.top_k = 0;
    g.seed = 1;
    int32_t tok = -1;
    for (int i = 0; i < 20; ++i) {
        if (mc_decode_one(s, &g, &tok) != MC_OK) {
            oc.error = mc_last_error();
            mc_session_destroy(s);
            return oc;
        }
    }
    mc_runtime_stats_t st0{}, st1{};
    if (mc_get_stats(s, &st0) != MC_OK || st0.cuda_graph_launches < 20) {
        oc.error = (st0.cuda_graph_launches < 20)
                       ? "graph mode broken: launches < warmup"
                       : mc_last_error();
        mc_session_destroy(s);
        return oc;
    }
    for (uint32_t i = 0; i < out_tokens; ++i) {
        if (mc_decode_one(s, &g, &tok) != MC_OK) {
            oc.error = mc_last_error();
            mc_session_destroy(s);
            return oc;
        }
    }
    if (mc_get_stats(s, &st1) != MC_OK ||
        st1.cuda_graph_launches - st0.cuda_graph_launches != out_tokens) {
        oc.error = "graph launch count mismatch over measured window";
        mc_session_destroy(s);
        return oc;
    }
    oc.ms = (double)(st1.decode_us - st0.decode_us) / 1000.0 / (double)out_tokens;
    mc_session_destroy(s);
    oc.ok = true;
    return oc;
}

} // namespace

int main() {
    fprintf(stderr, "perf_v2: begin\n");
    mc_debug_dump_set_steps(0);

    std::string bf16_dir, fp8_dir;
    if (const char* env = getenv("MC_MODEL_DIR"))
        bf16_dir = env;
    else if (file_exists(std::string(MC_REPO_ROOT) +
                         "/models/minicpm5-2b/model.wpk"))
        bf16_dir = std::string(MC_REPO_ROOT) + "/models/minicpm5-2b";
    else if (file_exists(std::string(MC_BUILD_DIR) + "/wpk_model/model.wpk"))
        bf16_dir = std::string(MC_BUILD_DIR) + "/wpk_model";
    if (const char* env = getenv("MC_FP8_MODEL_DIR"))
        fp8_dir = env;
    else if (file_exists(std::string(MC_REPO_ROOT) +
                         "/models/minicpm5-2b/model_fp8.wpk"))
        fp8_dir = std::string(MC_REPO_ROOT) + "/models/minicpm5-2b";
    if (!fp8_dir.empty() && !file_exists(fp8_dir + "/model.wpk"))
        fp8_dir.clear();
    if (bf16_dir.empty() && !fp8_dir.empty()) bf16_dir = fp8_dir;
    if (bf16_dir.empty() && fp8_dir.empty()) {
        fprintf(stderr,
                "== perf_v2: SKIP — no model package (need model.wpk / "
                "model_fp8.wpk) ==\n");
        return 0;
    }

    // P5-BF16 交付模式：MC_PERF_BF16=1 → 只跑 BF16，写 perf_bf16.json
    //（before = 本轮任务起点基线，同纪律实测：5×200 中位、空闲 GPU）
    const bool bf16_only = getenv("MC_PERF_BF16") != nullptr;

    // ---- P8 参数化：MC_PERF_CTXS 设置且合法 → sweep 模式；否则默认三档
    //      保持原样（perf_v2.json 原格式 + kBaseMs 基线对照）----
    std::vector<uint32_t> ctxs = {16u, 2048u, 8192u};
    bool sweep = getenv("MC_PERF_CTXS") != nullptr;
    if (sweep) {
        std::vector<uint32_t> parsed;
        if (!parse_ctx_list(getenv("MC_PERF_CTXS"), parsed)) {
            fprintf(stderr,
                    "[perf_v2] MC_PERF_CTXS=\"%s\" invalid -- fallback to "
                    "default {16,2048,8192} (non-sweep mode)\n",
                    getenv("MC_PERF_CTXS"));
            sweep = false;
        } else {
            ctxs = parsed;
        }
    }
    const uint32_t out_tokens = env_u32("MC_PERF_OUT_TOKENS", 200u);
    const uint32_t rounds_def = env_u32("MC_PERF_ROUNDS", 5u);
    const uint32_t rounds_big = env_u32("MC_PERF_ROUNDS_BIG", 3u);
    const uint32_t max_ctx = context_cap(ctxs);

    struct Arm {
        const char* mode;
        std::string dir;
        mc_precision_t prec;
    };
    std::vector<Arm> arms;
    if (!bf16_only && !fp8_dir.empty()) arms.push_back({"fp8", fp8_dir, MC_FP8});
    if (!bf16_dir.empty()) arms.push_back({"bf16", bf16_dir, MC_BF16});

    struct Row {
        std::string mode;
        uint32_t ctx;
        uint32_t out_tokens;
        uint32_t chunk = 0; // P8-A2：本档实际 attention CHUNK（sweep 记录）
        std::vector<double> rounds;
        double mean = 0.0, median = 0.0;
        double tps_mean = 0.0, tps_median = 0.0;
        int util_before;
        bool ok = false;
        std::string error;
    };
    std::vector<Row> rows;

    // 单档测量主体：model m 上跑 ctx 档 rounds 轮，填 Row。
    auto run_tier = [&](mc_model_t m, const Arm& arm, uint32_t ctx) {
        Row r;
        r.mode = arm.mode;
        r.ctx = ctx;
        r.out_tokens = out_tokens;
        // 首轮前等空闲；轮间复检。ctx ≥ 65536 用 ROUNDS_BIG（长上下文
        // 单轮耗时长，轮数收敛防占机过久）。
        const uint32_t rounds = ctx >= 65536u ? rounds_big : rounds_def;
        r.util_before = wait_idle();
        bool all_ok = true;
        for (uint32_t k = 0; k < rounds; ++k) {
            if (k > 0) (void)wait_idle();
            uint32_t chunk = 0;
            RoundOutcome oc = run_round(m, ctx, out_tokens, &chunk);
            if (chunk != 0) r.chunk = chunk;
            if (!oc.ok) {
                r.error = oc.error;
                all_ok = false;
                break; // 该档位已失败（如 OOM）：记录后继续下一档
            }
            r.rounds.push_back(oc.ms);
        }
        if (all_ok) {
            std::vector<double> sorted = r.rounds;
            std::sort(sorted.begin(), sorted.end());
            r.median = sorted[sorted.size() / 2];
            double sum = 0.0;
            for (double v : r.rounds) sum += v;
            r.mean = sum / (double)r.rounds.size();
            r.tps_median = 1000.0 / r.median;
            r.tps_mean = 1000.0 / r.mean;
            r.ok = true;
            printf("  %-4s ctx=%-6u chunk=%-3u median %7.3f ms (%6.1f tok/s) | "
                   "rounds:",
                   r.mode.c_str(), ctx, r.chunk, r.median, r.tps_median);
            for (size_t k = 0; k < r.rounds.size(); ++k)
                printf(" %.3f", r.rounds[k]);
            printf(" | util_before=%d%%\n", r.util_before);
        } else {
            printf("  %-4s ctx=%-6u chunk=%-3u ERROR: %s\n", r.mode.c_str(), ctx,
                   r.chunk, r.error.c_str());
        }
        rows.push_back(r);
    };

    if (sweep) {
        // ---- P8-A3：sweep 模式最外层循环 ctx 档，每档独立 model_load
        //      （max_context_tokens = min(ctx+512, 模型上限)）+ session +
        //      graph 捕获 —— 每档拿到自己的 CHUNK 档位（贴合按需建 ctx 的
        //      部署形态；每档 load 约 5-10s 可接受）。----
        for (uint32_t ctx : ctxs) {
            const uint32_t tier_cap =
                (ctx + 512u > mc::cfg::kMaxContextTokens)
                    ? mc::cfg::kMaxContextTokens
                    : (ctx + 512u);
            for (const auto& arm : arms) {
                printf("== perf_v2[sweep]: ctx=%u %s (%s) max_ctx=%u ==\n", ctx,
                       arm.mode, arm.dir.c_str(), tier_cap);
                mc_model_t m{nullptr};
                mc_model_options_t o = base_options(arm.prec, tier_cap);
                REQUIRE(mc_model_load(arm.dir.c_str(), &o, &m) == MC_OK);
                run_tier(m, arm, ctx);
                mc_model_destroy(m);
            }
        }
    } else {
        // ---- 默认模式：arm 外层、单 model（max_ctx = max(ctxs)+512），
        //      行为与 P5 验收基线一致。----
        for (const auto& arm : arms) {
            printf("== perf_v2: %s (%s) ==\n", arm.mode, arm.dir.c_str());
            mc_model_t m{nullptr};
            mc_model_options_t o = base_options(arm.prec, max_ctx);
            REQUIRE(mc_model_load(arm.dir.c_str(), &o, &m) == MC_OK);
            for (uint32_t ctx : ctxs) run_tier(m, arm, ctx);
            mc_model_destroy(m);
        }
    }

    // ---- JSON ----
    {
        const std::string dir = std::string(MC_REPO_ROOT) + "/benchmark/results";
        std::error_code ec;
        fs::create_directories(dir, ec);
        if (sweep) {
            // P8 sweep 模式：全部输出写 MC_PERF_OUT（默认 perf_sweep.json）；
            // kBaseMs 基线对照段仅对默认三档有效，此处跳过。
            const char* out_env = getenv("MC_PERF_OUT");
            std::string path = (out_env != nullptr && *out_env != '\0')
                                   ? std::string(out_env)
                                   : dir + "/perf_sweep.json";
            std::ofstream f(path);
            if (!f) return 1;
            f << "{\n  \"timestamp\": \"" << timestamp() << "\",\n";
            f << "  \"discipline\": \"idle-GPU back-to-back ctx sweep, rounds x "
              << out_tokens << " tokens, mean+median; util checked before each "
                 "round (wait if >5%)\",\n";
            f << "  \"max_context_tokens\": " << max_ctx << ",\n";
            f << "  \"fp8_model_dir\": \"" << fp8_dir << "\",\n";
            f << "  \"bf16_model_dir\": \"" << bf16_dir << "\",\n";
            f << "  \"rows\": [\n";
            for (size_t i = 0; i < rows.size(); ++i) {
                const Row& r = rows[i];
                f << "    {\"mode\": \"" << r.mode << "\", \"ctx\": " << r.ctx
                  << ", \"chunk\": " << r.chunk;
                if (r.ok) {
                    f << ", \"out_tokens\": " << r.out_tokens
                      << ", \"rounds_ms\": [";
                    for (size_t k = 0; k < r.rounds.size(); ++k)
                        f << r.rounds[k] << (k + 1 < r.rounds.size() ? ", " : "");
                    f << "], \"mean_ms\": " << r.mean
                      << ", \"median_ms\": " << r.median
                      << ", \"tok_per_s_mean\": " << r.tps_mean
                      << ", \"tok_per_s_median\": " << r.tps_median
                      << ", \"util_before_pct\": " << r.util_before << "}";
                } else {
                    // 失败档位（OOM/显存不足等）：记录错误继续
                    f << ", \"error\": \"" << r.error << "\"}";
                }
                f << (i + 1 < rows.size() ? "," : "") << "\n";
            }
            f << "  ]\n}\n";
            printf("  results written: %s\n", path.c_str());
        } else {
            // 默认模式：perf_v2.json 原格式（口径与 P5 验收基线一致）
            std::ofstream f(dir + "/perf_v2.json");
            if (!f) return 1;
            f << "{\n  \"timestamp\": \"" << timestamp() << "\",\n";
            f << "  \"discipline\": \"idle-GPU back-to-back, 5 rounds x 200 "
                 "tokens, median; util checked before each round (wait if >5%)\",\n";
            f << "  \"fp8_model_dir\": \"" << fp8_dir << "\",\n";
            f << "  \"bf16_model_dir\": \"" << bf16_dir << "\",\n";
            f << "  \"rows\": [\n";
            for (size_t i = 0; i < rows.size(); ++i) {
                const Row& r = rows[i];
                if (r.ok) {
                    f << "    {\"mode\": \"" << r.mode << "\", \"ctx\": " << r.ctx
                      << ", \"median_ms\": " << r.median
                      << ", \"tok_per_s\": " << r.tps_median
                      << ", \"util_before_pct\": " << r.util_before
                      << ", \"rounds_ms\": [";
                    for (size_t k = 0; k < r.rounds.size(); ++k)
                        f << r.rounds[k] << (k + 1 < r.rounds.size() ? ", " : "");
                    f << "]}";
                } else {
                    f << "    {\"mode\": \"" << r.mode << "\", \"ctx\": " << r.ctx
                      << ", \"error\": \"" << r.error << "\"}";
                }
                f << (i + 1 < rows.size() ? "," : "") << "\n";
            }
            f << "  ]\n}\n";
            printf("  results written: %s/perf_v2.json\n", dir.c_str());

            if (bf16_only) {
                // ---- perf_bf16.json：前后对照（基线为本任务起点实测值）----
                static const double kBaseMs[3] = {6.01235, 7.13242, 7.09717};
                std::ofstream b(dir + "/perf_bf16.json");
                if (!b) return 1;
                b << "{\n  \"timestamp\": \"" << timestamp() << "\",\n";
                b << "  \"discipline\": \"idle-GPU back-to-back, 5 rounds x 200 "
                     "tokens, median; util checked before each round (wait if >5%)\",\n";
                b << "  \"baseline_note\": \"baseline_ms = task-start measurement "
                     "(same discipline, same harness)\",\n";
                b << "  \"rows\": [\n";
                bool first = true;
                for (size_t i = 0; i < rows.size(); ++i) {
                    const Row& r = rows[i];
                    if (r.mode != "bf16" || !r.ok) continue;
                    const int ci = r.ctx == 16 ? 0 : (r.ctx == 2048 ? 1 : 2);
                    b << (first ? "    " : ",   ") << "{\"ctx\": " << r.ctx
                      << ", \"baseline_ms\": " << kBaseMs[ci]
                      << ", \"median_ms\": " << r.median
                      << ", \"tok_per_s\": " << r.tps_median
                      << ", \"baseline_tok_per_s\": " << 1000.0 / kBaseMs[ci]
                      << ", \"speedup\": " << kBaseMs[ci] / r.median
                      << ", \"util_before_pct\": " << r.util_before
                      << ", \"rounds_ms\": [";
                    for (size_t k = 0; k < r.rounds.size(); ++k)
                        b << r.rounds[k] << (k + 1 < r.rounds.size() ? ", " : "");
                    b << "]}\n";
                    first = false;
                }
                b << "  ]\n}\n";
                printf("  results written: %s/perf_bf16.json\n", dir.c_str());
            }
        }
    }
    fprintf(stderr, "perf_v2: done\n");
    return 0;
}
