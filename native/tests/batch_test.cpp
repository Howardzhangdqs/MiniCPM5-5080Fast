// batch_test.cpp — B1b 端到端验证（批组运行时 + eager 批前向）。
//
// 场景：
//   A（正确性）：N ∈ {2,8}，确定性 prompt（perf_v2 式 rng）。每槽两条完全
//     相同配置的 session：一条 solo decode_one ×64，一条入组 mc_batch_step
//     ×64 → greedy 前缀一致率 ≥99%；另做 relL2 对拍（solo 侧 MC_DEBUG_DUMP
//     的 step1.logits.f32 vs 批侧 mc_debug_batch_last_logits，方法见报告）。
//     若 <100%：dump 翻转位 + slot0 的 top-2 logit 差证据（尽力而为）。
//   B（生命周期）：solo 步 10 → 入组批步 10 → 出组 solo 步 10，输出与
//     全 solo 参考（30 步）完全一致（不变量：成员流串行序经事件链保持）。
//   C（性能门）：8 槽 eager step 计时，warmup 3 + 30 reps 中位 ≤25ms
//     （聚合 ≥320 tok/s）；MC_BATCH_BREAKDOWN 组记录 attention/gemm 段拆分；
//     destroy 前后 cudaMemGetInfo 差 <50MB（无泄漏，soak T1 口径）。
//   D（B5 前置探针，MC_BATCH_CORUN_PROBE=1 门控，默认不进 ctest 路径）：
//     量化「准入 prefill（计算受限）与批解码（带宽受限）在双流上的真实 GPU
//     共跑收益」。臂 1 = 8 次 spare prefill 串行 + 64 次 mc_batch_step；
//     臂 2 = 8 线程各持一个 spare session 并发 prefill，主线程同时驱动
//     64 次 mc_batch_step。收益 = 1 − T_corun/T_serial；判定线 ≥20% GO /
//     <15% STOP / 中间人工。正确性哨兵：共跑臂 spare decode 8 token 与独跑
//     参考前缀一致。3 轮取中位；产物 stdout 表 + benchmark/results/
//     corun_probe.json。
//
// 模型目录解析同 perf_v2：MC_MODEL_DIR > repo models/minicpm5-2b > build
// wpk_model；缺包 → SKIP+PASS（ctest 契约同 real_inference/soak_test）。
#include "minicpm_runtime.h"
#include "runtime/generated_model_config.h"

extern "C" void mc_debug_dump_set_steps(int steps);
extern "C" mc_status_t mc_debug_batch_group_stats(mc_batch_group_t, uint64_t*,
                                                  uint64_t*, uint32_t*);
extern "C" int mc_debug_batch_breakdown_collect(mc_batch_group_t, double*, int);
extern "C" mc_status_t mc_debug_batch_last_logits(mc_batch_group_t, uint32_t,
                                                  float*);

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <string>
#include <thread>
#include <vector>

#include <cuda_runtime.h> // cudaMemGetInfo（场景 C 泄漏检查）

namespace fs = std::filesystem;

struct mc_model { void* impl; };
struct mc_session { void* impl; };
struct mc_batch_group { void* impl; };

#define REQUIRE(cond)                                                          \
    do {                                                                       \
        if (!(cond)) {                                                         \
            fprintf(stderr, "FAIL %s:%d: %s (last_error: %s)\n", __FILE__,     \
                    __LINE__, #cond, mc_last_error());                         \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

namespace {

bool file_exists(const std::string& p) {
    std::error_code ec;
    return fs::exists(p, ec);
}

// perf_v2 式确定性 prompt（slot 区分：乘黄金分割奇数扰动）
std::vector<int32_t> make_prompt(uint32_t len, uint32_t slot) {
    std::vector<int32_t> p(len);
    for (uint32_t i = 0; i < len; ++i)
        p[i] = (int32_t)((i * 2654435761u + 7u + slot * 0x9E3779B9u) % 130560u);
    return p;
}

mc_generation_config_t greedy_cfg() {
    mc_generation_config_t g{};
    g.struct_size = sizeof(g);
    g.max_new_tokens = 100000; // 关闭收尾哨兵（本测试自行控制步数）
    g.temperature = 0.f;
    g.top_k = 0;
    g.top_p = 1.f;
    g.repetition_penalty = 1.f;
    g.seed = 1;
    return g;
}

mc_model_options_t model_opts(uint32_t max_ctx, uint32_t sessions, int graph) {
    mc_model_options_t o{};
    o.struct_size = sizeof(o);
    o.abi_version = MC_ABI_VERSION;
    o.device_id = 0;
    o.max_sessions = sessions;
    o.max_context_tokens = max_ctx;
    o.precision = MC_BF16; // 批路径恒 bf16 GEMM；solo 侧同精度对拍
    o.kv_precision = MC_KV_BF16;
    o.enable_cuda_graph = (uint32_t)graph;
    o.reserve_vram_bytes = 0;
    return o;
}

void run_solo(mc_session_t s, const std::vector<int32_t>& prompt, uint32_t steps,
              std::vector<int32_t>* out) {
    REQUIRE(mc_prefill(s, prompt.data(), (uint32_t)prompt.size()) == MC_OK);
    mc_generation_config_t g = greedy_cfg();
    int32_t tok = -1;
    for (uint32_t i = 0; i < steps; ++i) {
        REQUIRE(mc_decode_one(s, &g, &tok) == MC_OK);
        if (out) out->push_back(tok);
    }
}

double now_ms() {
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

// ---- 场景 A：N 槽 solo vs batch 的 greedy 前缀一致性 ----
bool scenario_a(mc_model_t m, uint32_t N, uint32_t prompt_len, uint32_t steps) {
    fprintf(stderr, "== batch_test A: N=%u prompt=%u steps=%u ==\n", N, prompt_len,
            steps);
    std::vector<mc_session_t> solo(N), bat(N);
    for (uint32_t i = 0; i < N; ++i) {
        REQUIRE(mc_session_create(m, &solo[i]) == MC_OK);
        REQUIRE(mc_session_create(m, &bat[i]) == MC_OK);
    }
    std::vector<std::vector<int32_t>> ref(N), got(N);
    for (uint32_t i = 0; i < N; ++i)
        run_solo(solo[i], make_prompt(prompt_len, i), steps, &ref[i]);

    mc_batch_options_t bo{};
    bo.struct_size = sizeof(bo);
    bo.abi_version = MC_ABI_VERSION;
    bo.max_slots = 8;
    bo.max_context_tokens = 512u; // 必须等于模型档（A 档模型 max_ctx=512）
    mc_batch_group_t g{nullptr};
    REQUIRE(mc_batch_group_create(m, &bo, &g) == MC_OK);
    std::vector<uint32_t> ids(N);
    for (uint32_t i = 0; i < N; ++i) {
        REQUIRE(mc_prefill(bat[i], make_prompt(prompt_len, i).data(), prompt_len) ==
                MC_OK);
        REQUIRE(mc_batch_group_add(g, bat[i], &ids[i]) == MC_OK);
    }
    std::vector<int32_t> out(8);
    mc_generation_config_t gc = greedy_cfg();
    std::vector<mc_generation_config_t> cfgs(8, gc);
    // 负向：重复 slot 拒绝
    {
        std::vector<uint32_t> dup = {ids[0], ids[0]};
        REQUIRE(mc_batch_step(g, dup.data(), 2, cfgs.data(), out.data()) ==
                MC_E_INVALID_ARGUMENT);
    }
    for (uint32_t st = 0; st < steps; ++st) {
        REQUIRE(mc_batch_step(g, ids.data(), N, cfgs.data(), out.data()) == MC_OK);
        for (uint32_t i = 0; i < N; ++i) got[i].push_back(out[i]);
    }

    uint32_t total = 0, match = 0;
    std::vector<std::pair<uint32_t, uint32_t>> flips; // (slot, step)
    for (uint32_t i = 0; i < N; ++i) {
        for (uint32_t j = 0; j < steps; ++j) {
            ++total;
            if (ref[i][j] == got[i][j]) ++match;
            else flips.push_back({i, j});
        }
    }
    const double rate = 100.0 * (double)match / (double)total;
    fprintf(stderr, "   prefix match: %u/%u = %.2f%% (flips=%zu)\n", match, total,
            rate, flips.size());
    for (size_t k = 0; k < flips.size() && k < 8; ++k)
        fprintf(stderr, "   flip: slot=%u step=%u solo=%d batch=%d\n", flips[k].first,
                flips[k].second, ref[flips[k].first][flips[k].second],
                got[flips[k].first][flips[k].second]);

    // 生命周期收尾：出组 + 销毁（成员随后 solo 销毁）
    for (uint32_t i = 0; i < N; ++i)
        REQUIRE(mc_batch_group_remove(g, ids[i]) == MC_OK);
    mc_batch_group_destroy(g);
    for (uint32_t i = 0; i < N; ++i) {
        mc_session_destroy(solo[i]);
        mc_session_destroy(bat[i]);
    }
    const bool pass = rate >= 99.0;
    fprintf(stderr, "== batch_test A: %s (gate ≥99%%) ==\n", pass ? "PASS" : "FAIL");
    return pass;
}

// ---- relL2 对拍：solo step1 logits（MC_DEBUG_DUMP）vs 批侧 hook ----
// 返回 relL2；fail_out 置 false 表示方法不可用（如实报告）。
// 说明：solo 生产默认的 M=1 走自研 gemv-rows，批侧 M=n 恒 cuBLASLt —— 两者
// 的 bf16 输出 rounding 差（1-2 ULP）经 42 层放大后 ~1e-2 量级，属于两条
// 已有数值路径的固有差异（非批化引入）。受控口径（gate）：MC_GEMV_OFF=1
// 让 solo 也走 cuBLASLt（同 kernel 族，仅 M 不同）→ 该口径度量「批化本身
// 引入的误差」，预期 ≈0。
double logits_relL2(mc_model_t m, const std::string& dump_dir, bool& method_ok,
                    bool controlled) {
    method_ok = false;
    const std::vector<int32_t> prompt = make_prompt(256, 0);
    mc_session_t s{nullptr}, b{nullptr};
    REQUIRE(mc_session_create(m, &s) == MC_OK);
    REQUIRE(mc_session_create(m, &b) == MC_OK);

    // solo：prefill(step0 dump) + 1 decode(step1 dump)
    mc_debug_dump_set_steps(2);
    std::vector<int32_t> solo_tok;
    run_solo(s, prompt, 1, &solo_tok);
    mc_debug_dump_set_steps(0);

    // batch：同 prompt、1 槽组、1 步
    mc_batch_options_t bo{};
    bo.struct_size = sizeof(bo);
    bo.abi_version = MC_ABI_VERSION;
    bo.max_slots = 8;
    bo.max_context_tokens = 512u;
    mc_batch_group_t g{nullptr};
    REQUIRE(mc_batch_group_create(m, &bo, &g) == MC_OK);
    REQUIRE(mc_prefill(b, prompt.data(), (uint32_t)prompt.size()) == MC_OK);
    uint32_t sid = 0;
    REQUIRE(mc_batch_group_add(g, b, &sid) == MC_OK);
    std::vector<mc_generation_config_t> cfgs(8, greedy_cfg());
    std::vector<int32_t> out(8);
    uint32_t one = 0;
    REQUIRE(mc_batch_step(g, &one, 1, cfgs.data(), out.data()) == MC_OK);
    std::vector<float> lg((size_t)mc::cfg::kVocabSize);
    REQUIRE(mc_debug_batch_last_logits(g, 0, lg.data()) == MC_OK);

    // 读 solo step1.logits.f32（bf16 upcast；批侧 hook 同口径）
    const std::string path = dump_dir + "/step1.logits.f32";
    FILE* f = fopen(path.c_str(), "rb");
    if (f == nullptr) {
        fprintf(stderr, "   relL2[%s]: solo dump missing (%s) — method unavailable\n",
                controlled ? "controlled" : "default", path.c_str());
        mc_batch_group_remove(g, sid);
        mc_batch_group_destroy(g);
        mc_session_destroy(s);
        mc_session_destroy(b);
        return -1.0;
    }
    std::vector<float> ref((size_t)mc::cfg::kVocabSize);
    const size_t rd = fread(ref.data(), sizeof(float), ref.size(), f);
    fclose(f);
    if (rd != ref.size()) return -1.0;
    double num = 0, den = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
        const double d = (double)ref[i] - (double)lg[i];
        num += d * d;
        den += (double)ref[i] * (double)ref[i];
    }
    const double rel = std::sqrt(num / (den > 0 ? den : 1));

    // top-2 证据（若翻转，报告两侧 top-2 间距；本函数总是打印基线）
    auto top2 = [](const std::vector<float>& v, int& i1, int& i2) {
        i1 = i2 = -1;
        for (size_t i = 0; i < v.size(); ++i) {
            if (i1 < 0 || v[i] > v[i1]) {
                i2 = i1;
                i1 = (int)i;
            } else if (i2 < 0 || v[i] > v[i2]) {
                i2 = (int)i;
            }
        }
    };
    int s1, s2, b1, b2;
    top2(ref, s1, s2);
    top2(lg, b1, b2);
    fprintf(stderr,
            "   relL2[%s](step1 logits, bf16-upcast 口径) = %.3e; solo top2=(%d "
            "%.4f, %d %.4f) batch top2=(%d %.4f, %d %.4f)\n",
            controlled ? "controlled" : "default", rel, s1, (double)ref[s1], s2,
            (double)ref[s2], b1, (double)lg[b1], b2, (double)lg[b2]);

    mc_batch_group_remove(g, sid);
    mc_batch_group_destroy(g);
    mc_session_destroy(s);
    mc_session_destroy(b);
    method_ok = true;
    return rel;
}

// ---- 场景 B：solo → batch → solo 的生命周期不变量 ----
bool scenario_b(mc_model_t m) {
    fprintf(stderr, "== batch_test B: lifecycle (solo 10 -> batch 10 -> solo 10) "
                    "==\n");
    const std::vector<int32_t> prompt = make_prompt(64, 3);
    mc_session_t ref{nullptr}, mix{nullptr};
    REQUIRE(mc_session_create(m, &ref) == MC_OK);
    REQUIRE(mc_session_create(m, &mix) == MC_OK);

    // 参考：30 连续 solo 步
    std::vector<int32_t> want;
    run_solo(ref, prompt, 30, &want);

    // 混合：solo 10 → 入组批 10 → 出组 solo 10
    std::vector<int32_t> got;
    run_solo(mix, prompt, 10, &got); // 内含 prefill（graph 模式 solo 走 replay）
    mc_batch_options_t bo{};
    bo.struct_size = sizeof(bo);
    bo.abi_version = MC_ABI_VERSION;
    bo.max_slots = 4;
    bo.max_context_tokens = 2560u; // 模型 2 的档（B/C 共用，见 main）
    mc_batch_group_t g{nullptr};
    REQUIRE(mc_batch_group_create(m, &bo, &g) == MC_OK);
    uint32_t sid = 0;
    REQUIRE(mc_batch_group_add(g, mix, &sid) == MC_OK);
    std::vector<mc_generation_config_t> cfgs(4, greedy_cfg());
    std::vector<int32_t> out(4);
    for (int st = 0; st < 10; ++st) {
        REQUIRE(mc_batch_step(g, &sid, 1, cfgs.data(), out.data()) == MC_OK);
        got.push_back(out[0]);
    }
    REQUIRE(mc_batch_group_remove(g, sid) == MC_OK);
    mc_batch_group_destroy(g);
    mc_generation_config_t gc = greedy_cfg();
    int32_t tok = -1;
    for (int st = 0; st < 10; ++st) {
        REQUIRE(mc_decode_one(mix, &gc, &tok) == MC_OK);
        got.push_back(tok);
    }

    // 量子边界补充：重入组 + 忙态拒绝 + 非 greedy 拒绝 + 出组后 solo 恢复
    mc_batch_group_t g2{nullptr};
    REQUIRE(mc_batch_group_create(m, &bo, &g2) == MC_OK);
    uint32_t sid2 = 0;
    REQUIRE(mc_batch_group_add(g2, mix, &sid2) == MC_OK);
    REQUIRE(mc_batch_group_add(g2, mix, &sid2) == MC_E_UNSUPPORTED); // 忙态拒绝
    mc_generation_config_t samp = greedy_cfg();
    samp.temperature = 0.8f;
    samp.top_k = 40;
    std::vector<mc_generation_config_t> bad(4, samp);
    REQUIRE(mc_batch_step(g2, &sid2, 1, bad.data(), out.data()) ==
            MC_E_UNSUPPORTED); // B1 仅 greedy
    REQUIRE(mc_batch_step(g2, &sid2, 1, cfgs.data(), out.data()) == MC_OK);
    REQUIRE(mc_batch_group_remove(g2, sid2) == MC_OK);
    REQUIRE(mc_decode_one(mix, &gc, &tok) == MC_OK); // 出组后 solo 恢复
    mc_batch_group_destroy(g2);
    fprintf(stderr, "   quantum-boundary checks ok (rejoin/busy/non-greedy/solo)\n");

    bool pass = true;
    if (got.size() != want.size()) pass = false;
    for (size_t i = 0; i < got.size() && i < want.size(); ++i)
        if (got[i] != want[i]) {
            fprintf(stderr, "   mismatch at %zu: want=%d got=%d\n", i, want[i],
                    got[i]);
            pass = false;
        }
    fprintf(stderr, "   30-token sequence %s\n", pass ? "identical" : "DIVERGED");
    mc_session_destroy(ref);
    mc_session_destroy(mix);
    fprintf(stderr, "== batch_test B: %s ==\n", pass ? "PASS" : "FAIL");
    return pass;
}

// ---- 场景 C：8 槽性能门 + 段拆分 + 泄漏检查 ----
bool scenario_c(mc_model_t m, uint32_t prompt_len) {
    fprintf(stderr, "== batch_test C: 8-slot perf gate (prompt=%u) ==\n",
            prompt_len);
    const uint32_t N = 8;
    std::vector<mc_session_t> ss(N);
    for (uint32_t i = 0; i < N; ++i) REQUIRE(mc_session_create(m, &ss[i]) == MC_OK);
    for (uint32_t i = 0; i < N; ++i) {
        const std::vector<int32_t> p = make_prompt(prompt_len, i + 100);
        REQUIRE(mc_prefill(ss[i], p.data(), (uint32_t)p.size()) == MC_OK);
    }

    size_t free0 = 0, tot = 0;
    REQUIRE(cudaMemGetInfo(&free0, &tot) == cudaSuccess);

    mc_batch_options_t bo{};
    bo.struct_size = sizeof(bo);
    bo.abi_version = MC_ABI_VERSION;
    bo.max_slots = N;
    bo.max_context_tokens = 2560u;
    mc_batch_group_t g{nullptr};
    REQUIRE(mc_batch_group_create(m, &bo, &g) == MC_OK);
    std::vector<uint32_t> ids(N);
    for (uint32_t i = 0; i < N; ++i)
        REQUIRE(mc_batch_group_add(g, ss[i], &ids[i]) == MC_OK);
    std::vector<mc_generation_config_t> cfgs(N, greedy_cfg());
    std::vector<int32_t> out(N);

    // warmup 3 + 计量 30（host wall = step 时长：内部 sync）
    for (int w = 0; w < 3; ++w)
        REQUIRE(mc_batch_step(g, ids.data(), N, cfgs.data(), out.data()) == MC_OK);
    std::vector<double> ms;
    for (int r = 0; r < 30; ++r) {
        const double t0 = now_ms();
        REQUIRE(mc_batch_step(g, ids.data(), N, cfgs.data(), out.data()) == MC_OK);
        ms.push_back(now_ms() - t0);
    }
    std::sort(ms.begin(), ms.end());
    const double med = ms[ms.size() / 2];
    const double tps = (double)N / (med / 1000.0);
    fprintf(stderr,
            "   step ms: median=%.3f p10=%.3f p90=%.3f | aggregate=%.1f tok/s "
            "(gate: median ≤25ms, ≥320 tok/s)\n",
            med, ms[3], ms[27], tps);

    for (uint32_t i = 0; i < N; ++i)
        REQUIRE(mc_batch_group_remove(g, ids[i]) == MC_OK);
    mc_batch_group_destroy(g);

    size_t free1 = 0;
    REQUIRE(cudaMemGetInfo(&free1, &tot) == cudaSuccess);
    const double delta_mb = (double)(free0 - free1) / (1024.0 * 1024.0);
    fprintf(stderr, "   leak check: free delta after create+destroy = %+.1f MB "
                    "(gate |Δ|<50MB)\n",
            delta_mb);

    // 段拆分：独立组（MC_BATCH_BREAKDOWN=1 于 create 时读 env），3 步取均值
    setenv("MC_BATCH_BREAKDOWN", "1", 1);
    mc_batch_group_t gb{nullptr};
    REQUIRE(mc_batch_group_create(m, &bo, &gb) == MC_OK);
    std::vector<uint32_t> ids2(N);
    for (uint32_t i = 0; i < N; ++i)
        REQUIRE(mc_batch_group_add(gb, ss[i], &ids2[i]) == MC_OK);
    double buckets[6] = {0, 0, 0, 0, 0, 0};
    static const char* names[6] = {"emb", "gemm", "attn", "norm", "swiglu",
                                   "tail(argmax+advance)"};
    for (int r = 0; r < 3; ++r) {
        REQUIRE(mc_batch_step(gb, ids2.data(), N, cfgs.data(), out.data()) == MC_OK);
        double b[6];
        const int k = mc_debug_batch_breakdown_collect(gb, b, 6);
        REQUIRE(k == 6);
        for (int i = 0; i < 6; ++i) buckets[i] += b[i];
    }
    fprintf(stderr, "   breakdown (avg/step):");
    double sum = 0;
    for (int i = 0; i < 6; ++i) sum += buckets[i] / 3.0;
    for (int i = 0; i < 6; ++i)
        fprintf(stderr, " %s=%.3fms", names[i], buckets[i] / 3.0);
    fprintf(stderr, " | total=%.3fms\n", sum);
    for (uint32_t i = 0; i < N; ++i)
        REQUIRE(mc_batch_group_remove(gb, ids2[i]) == MC_OK);
    mc_batch_group_destroy(gb);
    unsetenv("MC_BATCH_BREAKDOWN");

    for (uint32_t i = 0; i < N; ++i) mc_session_destroy(ss[i]);

    const bool pass = med <= 25.0 && tps >= 320.0 && std::fabs(delta_mb) < 50.0;
    fprintf(stderr, "== batch_test C: %s ==\n", pass ? "PASS" : "FAIL");
    return pass;
}

// ---- 场景 D（B5 前置探针）：量化 prefill 与批解码双流共跑收益 ----
// 背景（official 口径，8×2048 prompt + 64 tok）：稳态批步 8.1ms（≈988
// tok/s），但 8 次串行准入 prefill 占墙钟 879/1380ms → 聚合仅 371 tok/s；
// vLLM 靠 chunked prefill 与解码共跑达 431.9。理论：prefill 计算受限
//（cuBLASLt tensor core + flash attention）、批解码带宽受限（权重读）→
// SM 级应可部分共跑。本探针给出立项/止损数据。
//
// 并发安全依据（S1 契约，internal_state.h）：跨 session 并发 stream 安全
//（每 session 独立 arena/stream/gemm_ws；每 session 的调用来自单线程）；
// 共享可变状态仅 GemmEngine 计划缓存（自带 mutex）与不可变权重。探针的
// 跨线程组合严格限定为「spare session prefill × 批组 step」——无共享可变
// 状态；绝不并发调用同一 session/组。
//
// 计时口径：mc_prefill / mc_batch_step 内部均 stream-sync 后返回 → host
// wall = 各自 GPU 占用时长（含共跑期的资源争用）。
int gpu_util_now_d() {
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

// 等空闲（util ≤ limit_pct），最多 ~1 分钟；返回等待后的 util（-1 = 查询
// 失败，按空闲继续）。轻量版（perf_v2 的缩短款，探针专用）。
int wait_idle_d(int limit_pct, int tries) {
    for (int i = 0; i < tries; ++i) {
        const int u = gpu_util_now_d();
        if (u < 0 || u <= limit_pct) return u;
        struct timespec ts {0, 1000 * 1000 * 1000}; // 1s
        nanosleep(&ts, nullptr);
    }
    return gpu_util_now_d();
}

double median_of(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

std::string timestamp_d() {
    const time_t t = time(nullptr);
    char buf[32];
    strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%S", localtime(&t));
    return buf;
}

// 一步共跑 prefill 的工作描述（每线程一个，互不共享）
struct PrefillJob {
    mc_session_t s = {nullptr};
    const std::vector<int32_t>* prompt = nullptr;
    double dur_ms = -1.0;
    double t_start = 0, t_end = 0; // 绝对时刻（arm t0 修正后 = 相对起跑线）
    mc_status_t st = MC_E_INTERNAL;
};

bool scenario_d(mc_model_t m) {
    const uint32_t NSPARE = 8;   // spare session 数（每臂 8 个）
    const uint32_t STEPS = 64;   // 每臂批步数（两臂总工作量相同）
    const uint32_t MAIN_ROUNDS = 3; // 主测量轮数（取中位）
    const uint32_t SUPP_ROUNDS = 2; // 补充臂轮数（K=2/4 有界并发）
    const uint32_t PLEN = 2048;  // prompt 长度（official 口径）
    // 成员步数预算：2048 prompt + (3+2)×(64+64) = 2688 = 模型 max_ctx
    //（每轮 arm1 64 步 + 一个共跑臂 64 步；补充轮各含 K=2/K=4 两臂）
    fprintf(stderr,
            "== batch_test D: co-run probe (B5 前置)：8 spare prefill ×2048 "
            "serial/corun(K=8) + 补充 K∈{2,4} vs %u batch steps ==\n",
            STEPS);

    // ---- 布置：8 成员（prefill 完入组）+ 2×8 spare（A=串行臂，B=共跑臂）----
    std::vector<mc_session_t> mem(NSPARE), spa(NSPARE), spb(NSPARE);
    for (uint32_t i = 0; i < NSPARE; ++i) {
        REQUIRE(mc_session_create(m, &mem[i]) == MC_OK);
        REQUIRE(mc_session_create(m, &spa[i]) == MC_OK);
        REQUIRE(mc_session_create(m, &spb[i]) == MC_OK);
    }
    {
        size_t free_b = 0, tot_b = 0;
        if (cudaMemGetInfo(&free_b, &tot_b) == cudaSuccess)
            fprintf(stderr, "   VRAM after 24 sessions: free=%.2f GiB\n",
                    (double)free_b / (1ull << 30));
    }
    std::vector<std::vector<int32_t>> mem_prompt(NSPARE), sp_prompt(NSPARE);
    for (uint32_t i = 0; i < NSPARE; ++i) {
        mem_prompt[i] = make_prompt(PLEN, i + 100); // 成员（同场景 C 口径）
        sp_prompt[i] = make_prompt(PLEN, 200 + i);  // spare（确定性）
    }
    for (uint32_t i = 0; i < NSPARE; ++i)
        REQUIRE(mc_prefill(mem[i], mem_prompt[i].data(), PLEN) == MC_OK);
    mc_batch_options_t bo{};
    bo.struct_size = sizeof(bo);
    bo.abi_version = MC_ABI_VERSION;
    bo.max_slots = NSPARE;
    bo.max_context_tokens = 2688u;
    mc_batch_group_t g{nullptr};
    REQUIRE(mc_batch_group_create(m, &bo, &g) == MC_OK);
    std::vector<uint32_t> ids(NSPARE);
    for (uint32_t i = 0; i < NSPARE; ++i)
        REQUIRE(mc_batch_group_add(g, mem[i], &ids[i]) == MC_OK);
    std::vector<mc_generation_config_t> cfgs(NSPARE, greedy_cfg());
    std::vector<int32_t> out(NSPARE);

    // ---- 独跑参考（round 0，GPU 空闲）：spare A 侧 prefill + decode 8 ----
    // 参考用于共跑臂哨兵（greedy 确定性 → 跨轮复用；场景 A 已证 solo 复现）
    std::vector<std::vector<int32_t>> ref(NSPARE);
    for (uint32_t i = 0; i < NSPARE; ++i) {
        REQUIRE(mc_prefill(spa[i], sp_prompt[i].data(), PLEN) == MC_OK);
        mc_generation_config_t gc = greedy_cfg();
        int32_t tok = -1;
        for (int k = 0; k < 8; ++k) {
            REQUIRE(mc_decode_one(spa[i], &gc, &tok) == MC_OK);
            ref[i].push_back(tok);
        }
    }

    struct Round {
        double t_serial = 0, t_corun = 0;
        double prefill_solo = 0, prefill_corun = 0; // 平均 ms
        double step_solo = 0, step_corun = 0;       // 平均 ms
        bool sentinel = false;
        int util_before = -1;
        // 补充臂（有界并发 K）：K → 指标
        double t_k2 = 0, pf_k2 = 0, st_k2 = 0;
        double t_k4 = 0, pf_k4 = 0, st_k4 = 0;
        bool sentinel_k2 = false, sentinel_k4 = false;
        bool has_supp = false;
        // 剖面（机制归因）：步时长前/后半均值 + prefill 时间线（span/末完成）
        double prof_step_fh = 0, prof_step_sh = 0; // corun 臂步时长前/后半
        double prof_pf_last = 0;                   // 最后一个 prefill 完成时刻
    };
    std::vector<Round> rounds(MAIN_ROUNDS + SUPP_ROUNDS);

    // ---- 共跑臂统一实现：K 个工人线程从 spare 池原子取号做 prefill（并发
    //      prefill 数 ≤ K）∥ 主线程 64 次批步。K=NSPARE（每 spare 一线程）
    //      即规格臂 2 的全并发形态；K<NSPARE 为补充臂（模拟 B5 chunked
    //      prefill 的有界并发动态）。剖面输出：步时长前/后半均值（双峰检
    //      测）与 prefill 时间线（相对起跑线）。----
    auto corun_arm = [&](mc_session_t* sp, uint32_t K, double* t_ms,
                         double* pf_avg, double* st_avg, double* step_fh,
                         double* step_sh, double* pf_last_ms) {
        std::vector<PrefillJob> jobs(NSPARE);
        for (uint32_t i = 0; i < NSPARE; ++i) {
            jobs[i].s = sp[i];
            jobs[i].prompt = &sp_prompt[i];
        }
        std::atomic<bool> go{false};
        std::atomic<uint32_t> next_idx{0};
        auto worker = [&]() {
            while (!go.load(std::memory_order_acquire)) {
            } // 发令前自旋（对齐起跑线，开销 µs 级）
            for (;;) {
                const uint32_t i = next_idx.fetch_add(1);
                if (i >= NSPARE) break;
                PrefillJob& j = jobs[i];
                j.t_start = now_ms();
                j.st = mc_prefill(j.s, j.prompt->data(), (uint32_t)j.prompt->size());
                j.t_end = now_ms();
                j.dur_ms = j.t_end - j.t_start;
            }
        };
        std::vector<std::thread> th;
        th.reserve(K);
        for (uint32_t k = 0; k < K; ++k) th.emplace_back(worker);
        const double t0 = now_ms();
        go.store(true); // 双侧同时起跑
        std::vector<double> step_durs;
        step_durs.reserve(STEPS);
        for (uint32_t s = 0; s < STEPS; ++s) {
            const double s0 = now_ms();
            REQUIRE(mc_batch_step(g, ids.data(), NSPARE, cfgs.data(),
                                  out.data()) == MC_OK);
            step_durs.push_back(now_ms() - s0);
        }
        for (auto& t : th) t.join();
        *t_ms = now_ms() - t0;
        double pf_sum = 0, last_end = 0;
        for (uint32_t i = 0; i < NSPARE; ++i) {
            REQUIRE(jobs[i].st == MC_OK); // 线程侧错误上抛（status 捕获）
            REQUIRE(jobs[i].dur_ms > 0);
            pf_sum += jobs[i].dur_ms;
            last_end = std::max(last_end, jobs[i].t_end - t0);
        }
        *pf_avg = pf_sum / NSPARE;
        *st_avg = 0;
        double fh = 0, sh = 0;
        for (uint32_t s = 0; s < STEPS; ++s) {
            *st_avg += step_durs[s] / STEPS;
            if (s < STEPS / 2) fh += step_durs[s] / (STEPS / 2);
            else sh += step_durs[s] / (STEPS - STEPS / 2);
        }
        *step_fh = fh;
        *step_sh = sh;
        *pf_last_ms = last_end;
    };

    // 哨兵：spare 集 sp 上 decode 8 token vs 独跑参考前缀
    auto sentinel_on = [&](mc_session_t* sp) {
        for (uint32_t i = 0; i < NSPARE; ++i) {
            mc_generation_config_t gc = greedy_cfg();
            int32_t tok = -1;
            for (int k = 0; k < 8; ++k) {
                if (mc_decode_one(sp[i], &gc, &tok) != MC_OK || tok != ref[i][k]) {
                    fprintf(stderr,
                            "   sentinel FAIL: spare %u token %d: want %d got %d\n",
                            i, k, ref[i][k], tok);
                    return false;
                }
            }
        }
        return true;
    };

    const uint32_t TOTAL = MAIN_ROUNDS + SUPP_ROUNDS;
    for (uint32_t r = 0; r < TOTAL; ++r) {
        Round& R = rounds[r];
        R.util_before = wait_idle_d(5, 60);
        // spare 序列复位（每臂都用「首次 prefill」的 fresh 序列）
        for (uint32_t i = 0; i < NSPARE; ++i) {
            REQUIRE(mc_session_reset(spa[i]) == MC_OK);
            REQUIRE(mc_session_reset(spb[i]) == MC_OK);
        }

        // ---- 臂 1（串行基线，主轮）：8 次 prefill 顺序 → 64 次批步 ----
        if (r < MAIN_ROUNDS) {
            const double t0 = now_ms();
            double pf_sum = 0;
            for (uint32_t i = 0; i < NSPARE; ++i) {
                const double p0 = now_ms();
                REQUIRE(mc_prefill(spa[i], sp_prompt[i].data(), PLEN) == MC_OK);
                pf_sum += now_ms() - p0;
            }
            R.prefill_solo = pf_sum / NSPARE;
            double st_sum = 0;
            for (uint32_t s = 0; s < STEPS; ++s) {
                const double s0 = now_ms();
                REQUIRE(mc_batch_step(g, ids.data(), NSPARE, cfgs.data(),
                                      out.data()) == MC_OK);
                st_sum += now_ms() - s0;
            }
            R.step_solo = st_sum / STEPS;
            R.t_serial = now_ms() - t0;
        }

        // ---- 臂 2（共跑，K=NSPARE 全并发，主轮）----
        if (r < MAIN_ROUNDS) {
            corun_arm(spb.data(), NSPARE, &R.t_corun, &R.prefill_corun,
                      &R.step_corun, &R.prof_step_fh, &R.prof_step_sh,
                      &R.prof_pf_last);
            R.sentinel = sentinel_on(spb.data());
            fprintf(stderr,
                    "   round %u [main]: T_serial=%7.1fms T_corun(K=%u)=%7.1fms "
                    "benefit=%5.1f%% | prefill solo=%6.1fms corun=%6.1fms "
                    "(%+.0f%%) | step solo=%5.2fms corun=%5.2fms (%+.0f%%) | "
                    "util=%d%% sentinel=%s\n",
                    r, R.t_serial, NSPARE, R.t_corun,
                    100.0 * (1.0 - R.t_corun / R.t_serial), R.prefill_solo,
                    R.prefill_corun,
                    100.0 * (R.prefill_corun / R.prefill_solo - 1.0), R.step_solo,
                    R.step_corun, 100.0 * (R.step_corun / R.step_solo - 1.0),
                    R.util_before, R.sentinel ? "ok" : "FAIL");
            fprintf(stderr,
                    "     剖面: step 前/后半=%5.2f/%5.2fms（前段=共跑窗口内的"
                    "拖慢；后段若≈solo 则无持久干扰）| 最后 prefill 完成于 "
                    "%6.1fms（占 T_corun %2.0f%%）\n",
                    R.prof_step_fh, R.prof_step_sh, R.prof_pf_last,
                    100.0 * R.prof_pf_last / R.t_corun);
        } else {
            // ---- 补充臂（有界并发 K=2/4）：B5 chunked prefill 形态的模拟 ----
            double fh2 = 0, sh2 = 0, pl2 = 0, fh4 = 0, sh4 = 0, pl4 = 0;
            R.has_supp = true;
            corun_arm(spa.data(), 2, &R.t_k2, &R.pf_k2, &R.st_k2, &fh2, &sh2,
                      &pl2);
            R.sentinel_k2 = sentinel_on(spa.data());
            for (uint32_t i = 0; i < NSPARE; ++i)
                REQUIRE(mc_session_reset(spa[i]) == MC_OK);
            corun_arm(spb.data(), 4, &R.t_k4, &R.pf_k4, &R.st_k4, &fh4, &sh4,
                      &pl4);
            R.sentinel_k4 = sentinel_on(spb.data());
            fprintf(stderr,
                    "   round %u [supp]: K=2: T=%7.1fms prefill=%6.1fms step=%5.2fms"
                    " (前/后半 %5.2f/%5.2f, 末prefill@%6.1fms) sentinel=%s | "
                    "K=4: T=%7.1fms prefill=%6.1fms step=%5.2fms (前/后半 "
                    "%5.2f/%5.2f, 末prefill@%6.1fms) sentinel=%s | util=%d%%\n",
                    r, R.t_k2, R.pf_k2, R.st_k2, fh2, sh2, pl2,
                    R.sentinel_k2 ? "ok" : "FAIL", R.t_k4, R.pf_k4, R.st_k4, fh4,
                    sh4, pl4, R.sentinel_k4 ? "ok" : "FAIL", R.util_before);
        }
    }

    // ---- 中位汇总 + 判定（≥20% GO / <15% STOP / 中间人工）----
    std::vector<double> vs, vc, vps, vpc, vss, vsc;
    bool sentinels = true;
    for (uint32_t r = 0; r < MAIN_ROUNDS; ++r) {
        const Round& R = rounds[r];
        vs.push_back(R.t_serial);
        vc.push_back(R.t_corun);
        vps.push_back(R.prefill_solo);
        vpc.push_back(R.prefill_corun);
        vss.push_back(R.step_solo);
        vsc.push_back(R.step_corun);
        sentinels = sentinels && R.sentinel;
    }
    const double med_serial = median_of(vs), med_corun = median_of(vc);
    const double med_ps = median_of(vps), med_pc = median_of(vpc);
    const double med_ss = median_of(vss), med_sc = median_of(vsc);
    const double benefit = 100.0 * (1.0 - med_corun / med_serial);
    const double pf_slow = 100.0 * (med_pc / med_ps - 1.0);
    const double st_slow = 100.0 * (med_sc / med_ss - 1.0);
    // 投影聚合（official 口径：512 decode token / 共跑总墙钟；对照 vLLM 431.9）
    const double proj_tps = (double)(NSPARE * STEPS) / (med_corun / 1000.0);
    const char* verdict = benefit >= 20.0 ? "GO-B5"
                          : benefit < 15.0 ? "STOP"
                                           : "MANUAL (15-20%, 人工判定)";

    // 补充臂中位（对照 T_serial 主轮中位）
    std::vector<double> vk2t, vk4t, vk2p, vk4p, vk2s, vk4s;
    bool supp_sentinels = true;
    for (uint32_t r = MAIN_ROUNDS; r < TOTAL; ++r) {
        const Round& R = rounds[r];
        vk2t.push_back(R.t_k2);
        vk4t.push_back(R.t_k4);
        vk2p.push_back(R.pf_k2);
        vk4p.push_back(R.pf_k4);
        vk2s.push_back(R.st_k2);
        vk4s.push_back(R.st_k4);
        supp_sentinels = supp_sentinels && R.sentinel_k2 && R.sentinel_k4;
    }
    const double med_k2t = median_of(vk2t), med_k4t = median_of(vk4t);
    const double ben_k2 = 100.0 * (1.0 - med_k2t / med_serial);
    const double ben_k4 = 100.0 * (1.0 - med_k4t / med_serial);

    printf("\n==== B5 前置探针（共跑收益）====\n");
    printf("%-14s %10s %10s %8s\n", "round", "T_serial", "T_corun", "benefit");
    for (uint32_t r = 0; r < MAIN_ROUNDS; ++r)
        printf("round %-8u %9.1fms %9.1fms %7.1f%%\n", r, rounds[r].t_serial,
               rounds[r].t_corun,
               100.0 * (1.0 - rounds[r].t_corun / rounds[r].t_serial));
    printf("%-14s %9.1fms %9.1fms %7.1f%%\n", "median", med_serial, med_corun,
           benefit);
    printf("prefill avg : solo %6.1fms -> corun(K=8) %6.1fms  (被批步拖慢 %+.1f%%)\n",
           med_ps, med_pc, pf_slow);
    printf("batch step  : solo %6.2fms -> corun(K=8) %6.2fms  (被prefill拖慢 %+.1f%%)\n",
           med_ss, med_sc, st_slow);
    printf("投影聚合    : %u tok / %.1fms = %.0f tok/s (official 串行 371 / vLLM "
           "431.9)\n",
           NSPARE * STEPS, med_corun, proj_tps);
    printf("-- 补充（有界并发，模拟 B5 chunked 动态；T_serial 用主轮中位 %.1fms）"
           "--\n",
           med_serial);
    printf("K=2 pool    : T=%7.1fms benefit=%5.1f%% | prefill avg %6.1fms | step "
           "avg %5.2fms\n",
           med_k2t, ben_k2, median_of(vk2p), median_of(vk2s));
    printf("K=4 pool    : T=%7.1fms benefit=%5.1f%% | prefill avg %6.1fms | step "
           "avg %5.2fms\n",
           med_k4t, ben_k4, median_of(vk4p), median_of(vk4s));
    printf("sentinel    : main %s / supp %s (每臂 8 spare × 8 token 前缀一致)\n",
           sentinels ? "PASS" : "FAIL", supp_sentinels ? "PASS" : "FAIL");
    printf("verdict     : %s (K=8 全并发口径；线：≥20%% GO / <15%% STOP)\n",
           verdict);

    // ---- JSON 落盘：benchmark/results/corun_probe.json ----
    {
        const std::string jdir = std::string(MC_REPO_ROOT) + "/benchmark/results";
        std::error_code ec2;
        fs::create_directories(jdir, ec2);
        const std::string path = jdir + "/corun_probe.json";
        FILE* f = fopen(path.c_str(), "wb");
        if (f != nullptr) {
            fprintf(f, "{\n  \"probe\": \"corun\",\n  \"timestamp\": \"%s\",\n",
                    timestamp_d().c_str());
            fprintf(f, "  \"method\": {\n    \"prompt_tokens\": %u,\n"
                       "    \"spare_sessions\": %u,\n    \"batch_slots\": %u,\n"
                       "    \"batch_steps\": %u,\n    \"main_rounds\": %u,\n"
                       "    \"supp_rounds\": %u,\n",
                    PLEN, NSPARE, NSPARE, STEPS, MAIN_ROUNDS, SUPP_ROUNDS);
            fprintf(f, "    \"arm_serial\": \"8x prefill sequential, then %u "
                       "batch steps\",\n",
                    STEPS);
            fprintf(f, "    \"arm_corun\": \"K worker threads pull spare prefills "
                       "(concurrency<=K) || %u batch steps on main thread; "
                       "K=8 = spec arm2 (one thread per spare)\",\n",
                    STEPS);
            fprintf(f, "    \"supplementary\": \"K in {2,4} bounded pools emulate "
                       "B5 chunked-prefill dynamics\",\n");
            fprintf(f, "    \"concurrency_basis\": \"S1 contract: per-session "
                       "stream/arena/gemm_ws; GemmEngine cache mutex\",\n");
            fprintf(f, "    \"baseline_note\": \"in-situ baselines from arm1 "
                       "(prefill_solo/step_solo)\"\n  },\n");
            fprintf(f, "  \"rounds\": [\n");
            for (uint32_t r = 0; r < TOTAL; ++r) {
                const Round& R = rounds[r];
                if (!R.has_supp) {
                    fprintf(f,
                            "    {\"kind\": \"main\", \"t_serial_ms\": %.2f, "
                            "\"t_corun_ms\": %.2f, \"benefit_pct\": %.2f, "
                            "\"prefill_solo_ms\": %.2f, \"prefill_corun_ms\": "
                            "%.2f, \"prefill_slowdown_pct\": %.2f, "
                            "\"step_solo_ms\": %.3f, \"step_corun_ms\": %.3f, "
                            "\"step_slowdown_pct\": %.2f, "
                            "\"step_first_half_ms\": %.3f, "
                            "\"step_second_half_ms\": %.3f, "
                            "\"last_prefill_end_ms\": %.2f, "
                            "\"util_before_pct\": %d, \"sentinel_ok\": %s}%s\n",
                            R.t_serial, R.t_corun,
                            100.0 * (1.0 - R.t_corun / R.t_serial),
                            R.prefill_solo, R.prefill_corun,
                            100.0 * (R.prefill_corun / R.prefill_solo - 1.0),
                            R.step_solo, R.step_corun,
                            100.0 * (R.step_corun / R.step_solo - 1.0),
                            R.prof_step_fh, R.prof_step_sh, R.prof_pf_last,
                            R.util_before, R.sentinel ? "true" : "false",
                            r + 1 < TOTAL ? "," : "");
                } else {
                    fprintf(f,
                            "    {\"kind\": \"supp\", \"k2\": {\"t_ms\": %.2f, "
                            "\"prefill_avg_ms\": %.2f, \"step_avg_ms\": %.3f, "
                            "\"sentinel_ok\": %s}, \"k4\": {\"t_ms\": %.2f, "
                            "\"prefill_avg_ms\": %.2f, \"step_avg_ms\": %.3f, "
                            "\"sentinel_ok\": %s}, \"util_before_pct\": %d}%s\n",
                            R.t_k2, R.pf_k2, R.st_k2,
                            R.sentinel_k2 ? "true" : "false", R.t_k4, R.pf_k4,
                            R.st_k4, R.sentinel_k4 ? "true" : "false",
                            R.util_before, r + 1 < TOTAL ? "," : "");
                }
            }
            fprintf(f, "  ],\n");
            fprintf(f,
                    "  \"median\": {\"t_serial_ms\": %.2f, \"t_corun_ms\": %.2f, "
                    "\"benefit_pct\": %.2f, \"prefill_solo_ms\": %.2f, "
                    "\"prefill_corun_ms\": %.2f, \"prefill_slowdown_pct\": %.2f, "
                    "\"step_solo_ms\": %.3f, \"step_corun_ms\": %.3f, "
                    "\"step_slowdown_pct\": %.2f, \"projected_aggregate_tps\": "
                    "%.1f},\n",
                    med_serial, med_corun, benefit, med_ps, med_pc, pf_slow,
                    med_ss, med_sc, st_slow, proj_tps);
            fprintf(f,
                    "  \"supplementary_median\": {\"k2\": {\"t_ms\": %.2f, "
                    "\"benefit_pct\": %.2f, \"prefill_avg_ms\": %.2f, "
                    "\"step_avg_ms\": %.3f}, \"k4\": {\"t_ms\": %.2f, "
                    "\"benefit_pct\": %.2f, \"prefill_avg_ms\": %.2f, "
                    "\"step_avg_ms\": %.3f}},\n",
                    med_k2t, ben_k2, median_of(vk2p), median_of(vk2s), med_k4t,
                    ben_k4, median_of(vk4p), median_of(vk4s));
            fprintf(f,
                    "  \"sentinel\": {\"main\": \"%s\", \"supp\": \"%s\"},\n"
                    "  \"verdict\": \"%s (K=8 spec arm; supp K=2/K=4 for chunked "
                    "projection)\"\n}\n",
                    sentinels ? "pass" : "fail", supp_sentinels ? "pass" : "fail",
                    verdict);
            fclose(f);
            printf("json        : %s\n", path.c_str());
        } else {
            fprintf(stderr, "   WARN: cannot write %s\n", path.c_str());
        }
    }

    // ---- 收尾 ----
    for (uint32_t i = 0; i < NSPARE; ++i)
        REQUIRE(mc_batch_group_remove(g, ids[i]) == MC_OK);
    mc_batch_group_destroy(g);
    for (uint32_t i = 0; i < NSPARE; ++i) {
        mc_session_destroy(mem[i]);
        mc_session_destroy(spa[i]);
        mc_session_destroy(spb[i]);
    }
    // 探针本体给数据；pass 口径 = 哨兵全过（收益判定交由人工/后续决策）
    const bool pass = sentinels && supp_sentinels;
    fprintf(stderr, "== batch_test D: %s ==\n", pass ? "PASS" : "FAIL");
    return pass;
}

} // namespace

int main() {
    fprintf(stderr, "batch_test: begin\n");
    std::string dir;
    if (const char* env = getenv("MC_MODEL_DIR"))
        dir = env;
    else if (file_exists(std::string(MC_REPO_ROOT) +
                         "/models/minicpm5-2b/model.wpk"))
        dir = std::string(MC_REPO_ROOT) + "/models/minicpm5-2b";
    else if (file_exists(std::string(MC_BUILD_DIR) + "/wpk_model/model.wpk"))
        dir = std::string(MC_BUILD_DIR) + "/wpk_model";
    if (dir.empty()) {
        fprintf(stderr, "== batch_test: SKIP — no model package (need model.wpk) "
                        "==\n");
        return 0;
    }
    const uint32_t steps = getenv("MC_BATCH_STEPS")
                               ? (uint32_t)atoi(getenv("MC_BATCH_STEPS"))
                               : 64u;

    // relL2 的 solo 侧 dump 目录（进程首个 dump::dir() 调用前生效）
    const std::string dump_dir = "/tmp/opencode/batch_test_dump";
    std::error_code ec;
    fs::remove_all(dump_dir, ec);
    fs::create_directories(dump_dir, ec);
    setenv("MC_DEBUG_DUMP_DIR", dump_dir.c_str(), 1);
    mc_debug_dump_set_steps(0);

    bool all = true;

    // ---- 场景 D（B5 前置探针）：MC_BATCH_CORUN_PROBE=1 时独占运行本进程
    //      （不与 A/B/C 混跑，避免互相污染计时；默认不进 ctest 路径）。
    //      ctx=2688 = 2048 prompt + 640 decode 步预算（见 scenario_d 注释）----
    if (getenv("MC_BATCH_CORUN_PROBE") != nullptr) {
        mc_model_t m{nullptr};
        // graph off：探针只关心 eager 共跑行为；session 建更快（无 capture）
        mc_model_options_t o = model_opts(2688, 32, 0);
        REQUIRE(mc_model_load(dir.c_str(), &o, &m) == MC_OK);
        all = scenario_d(m);
        mc_model_destroy(m);
        fprintf(stderr, "== batch_test: %s ==\n", all ? "ALL PASS" : "FAILED");
        return all ? 0 : 1;
    }

    // ---- 模型 1（graph off，eager solo 对拍）：场景 A + 默认口径 relL2（信息性）----
    {
        mc_model_t m{nullptr};
        mc_model_options_t o = model_opts(512, 32, 0);
        REQUIRE(mc_model_load(dir.c_str(), &o, &m) == MC_OK);
        all = scenario_a(m, 2, 256, steps) && all;
        all = scenario_a(m, 8, 256, steps) && all;
        bool ok = false;
        const double rel = logits_relL2(m, dump_dir, ok, /*controlled=*/false);
        if (ok) {
            // 默认口径：solo gemv-rows vs 批 cuBLASLt 的固有 bf16 舍入差
            //（信息性，不作门 —— 见函数头注释）
            fprintf(stderr,
                    "== batch_test A-relL2[default]: informational (%.3e; 归因："
                    "solo 自研 gemv-rows 与 cuBLASLt 的 bf16 舍入差，非批化误差) "
                    "==\n",
                    rel);
        } else {
            fprintf(stderr,
                    "== batch_test A-relL2[default]: SKIP (method unavailable) ==\n");
        }
        mc_model_destroy(m);
    }

    // ---- 模型 1b（graph off + MC_GEMV_OFF：solo 与批同为 cuBLASLt）：受控
    //      relL2 门 ≤2e-3（度量「批化本身引入的误差」）----
    {
        setenv("MC_GEMV_OFF", "1", 1);
        mc_model_t m{nullptr};
        mc_model_options_t o = model_opts(512, 32, 0);
        REQUIRE(mc_model_load(dir.c_str(), &o, &m) == MC_OK);
        bool ok = false;
        const double rel = logits_relL2(m, dump_dir, ok, /*controlled=*/true);
        if (ok) {
            const bool pass = rel <= 2e-3;
            fprintf(stderr, "== batch_test A-relL2[controlled]: %s (%.3e, gate "
                            "≤2e-3) ==\n",
                    pass ? "PASS" : "FAIL", rel);
            all = pass && all;
        } else {
            fprintf(stderr, "== batch_test A-relL2[controlled]: SKIP (method "
                            "unavailable) ==\n");
        }
        mc_model_destroy(m);
        unsetenv("MC_GEMV_OFF");
    }

    // ---- 模型 2（graph on）：场景 B（solo 走 graph replay 的生命周期）+ C ----
    {
        mc_model_t m{nullptr};
        // MC_BATCH_GRAPH=0 可关 graph（隔离实验：eager solo 的生命周期不变量）
        const int graph = getenv("MC_BATCH_GRAPH") &&
                                  atoi(getenv("MC_BATCH_GRAPH")) == 0
                              ? 0
                              : 1;
        mc_model_options_t o = model_opts(2560, 16, graph);
        REQUIRE(mc_model_load(dir.c_str(), &o, &m) == MC_OK);
        all = scenario_b(m) && all;
        all = scenario_c(m, getenv("MC_BATCH_CTX")
                               ? (uint32_t)atoi(getenv("MC_BATCH_CTX"))
                               : 2048u) &&
              all;
        mc_model_destroy(m);
    }

    fprintf(stderr, "== batch_test: %s ==\n", all ? "ALL PASS" : "FAILED");
    return all ? 0 : 1;
}
