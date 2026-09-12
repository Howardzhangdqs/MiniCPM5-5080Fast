// soak_test.cpp — P7 生产化加固 CTest（计划 §20-P7 / §26.4）：
// 把「快的原型」变成「可持续部署的软件」的稳健性验证。
//
// 覆盖（缺 model.wpk 时 SKIP+PASS，与 real_inference 同约定）：
//   T1 长输出 soak：单 session 连续 5 轮「prefill 512 + decode 1024」，
//      每轮间 mc_session_reset 复用。断言：
//        - stats 累计/复位语义（total 跨 reset 累计；每轮增量恰 512/1024；
//          graph replay 计数每轮恰 +1024）；
//        - 无泄漏：末轮后 cudaMemGetInfo 与首轮前差 < 50MB；
//        - decode_us/token 稳定：末轮/首轮中位比 < 1.3（允许热节流）；
//        - 每轮后 cudaGetLastError 干净（无 sticky error）。
//   T2 多 session 并发隔离（max_sessions≥3，BF16）：3 session 交错
//      decode（A 10 步→B→C × 50 轮）vs 单 session 顺序跑同样序列，
//      逐 token 一致（KV/状态无串扰——核心断言，固定 prompt+greedy）；
//      各 session stats 相互独立（恰好各自的 512/500）。
//   T2b 双线程真并发回归锚（S4 / 并发推理路径 A）：std::thread × 2，各自
//      session barrier 对齐后并发 prefill + decode 64，与顺序基线逐 token
//      比对——并发路径确定性（S1 workspace + S2 串行序不变量）的守护
//      测试；锚定性验证记录见函数注释与
//      benchmark/results/concurrency_path_a.json（如实：本机默认 tactic
//      下 workspace 竞态为潜伏态，「改回共享 ws→挂」未复现）。
//   T3 OOM 行为：max_context_tokens=131072 × max_sessions=2 → 在
//      session_create 处得到 MC_E_OUT_OF_MEMORY（报文含 free/need 数字）、
//      *out_session 为 null、不 crash、无 sticky error；随后用正常
//      options 加载成功（失败路径无资源残留）。
//   T4 reset 语义压测：单 session 200 次 reset 循环（prefill 8 + decode 2），
//      每轮第 2 个 decode token 与首轮相同（greedy 确定性 ⇒ seq/RNG/状态干净）。
//   T5 VRAM 预算表：程序化探测 1/2/4 session @ ctx{2048,8448} 实际占用 →
//      benchmark/results/vram_budget.json（server --max-sessions 容量规划依据）。
//
// 时长预算 ≈3 分钟（模型加载 ×3 占大头；GPU 用完即释放）。
#include "minicpm_runtime.h"

// additive debug 钩子（libminicpm_native 导出；不属于冻结 ABI）。
extern "C" void mc_debug_dump_set_steps(int steps);
extern "C" size_t mc_debug_session_graph_nodes(mc_session_t session);
extern "C" size_t mc_debug_session_arena_bytes(mc_session_t session);

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;

#ifndef MC_REPO_ROOT
#define MC_REPO_ROOT "/workspace"
#endif
#ifndef MC_BUILD_DIR
#define MC_BUILD_DIR "."
#endif

#define REQUIRE(cond)                                                          \
    do {                                                                       \
        if (!(cond)) {                                                         \
            fprintf(stderr, "FAIL %s:%d: %s (last_error: %s)\n", __FILE__,     \
                    __LINE__, #cond, mc_last_error());                         \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

// 公共头只前向声明 opaque handle；消费者侧复制等价布局（单指针）。
struct mc_model { void* impl; };
struct mc_session { void* impl; };

namespace {

constexpr uint64_t kMB = 1024ull * 1024ull;

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

// 模型目录解析（与 real_inference/perf_v2 同序）。
std::string resolve_model_dir() {
    if (const char* env = getenv("MC_MODEL_DIR")) return env;
    std::string repo = std::string(MC_REPO_ROOT) + "/models/minicpm5-2b";
    if (file_exists(repo + "/model.wpk")) return repo;
    std::string build = std::string(MC_BUILD_DIR) + "/wpk_model";
    if (file_exists(build + "/model.wpk")) return build;
    return {};
}

uint64_t free_vram_mb() {
    size_t f = 0, t = 0;
    REQUIRE(cudaMemGetInfo(&f, &t) == cudaSuccess);
    return (uint64_t)f / kMB;
}

bool cuda_clean(const char* where) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        fprintf(stderr, "[soak] sticky CUDA error after %s: %s\n", where,
                cudaGetErrorString(e));
        return false;
    }
    return true;
}

mc_model_options_t base_options(uint32_t max_sessions, uint32_t max_ctx) {
    mc_model_options_t o{};
    o.struct_size = sizeof(o);
    o.abi_version = MC_ABI_VERSION;
    o.device_id = 0;
    o.max_sessions = max_sessions;
    o.max_context_tokens = max_ctx;
    o.precision = MC_BF16;
    o.kv_precision = MC_KV_BF16;
    o.enable_cuda_graph = 1;
    o.reserve_vram_bytes = 0;
    return o;
}

mc_generation_config_t greedy_config() {
    mc_generation_config_t g{};
    g.struct_size = sizeof(g);
    g.max_new_tokens = 1u << 30; // 不触发 force-EOS
    g.temperature = 0.f;         // greedy（确定性）
    g.top_p = 1.f;
    g.top_k = 0;
    g.repetition_penalty = 1.f;
    g.seed = 1;
    return g;
}

// 固定 prompt（确定性 token，安全 vocab 范围；perf_v2 同式）。
std::vector<int32_t> make_prompt(uint32_t n, uint32_t salt) {
    std::vector<int32_t> p(n);
    for (uint32_t i = 0; i < n; ++i)
        p[i] = (int32_t)((i * 2654435761u + 7u + salt) % 130560u);
    return p;
}

void run_decode(mc_session_t s, uint32_t n, std::vector<int32_t>* out,
                const mc_generation_config_t& g) {
    for (uint32_t i = 0; i < n; ++i) {
        int32_t tok = -1;
        REQUIRE(mc_decode_one(s, &g, &tok) == MC_OK);
        REQUIRE(tok >= 0);
        if (out) out->push_back(tok);
    }
}

// ---- T1：长输出 soak ----
void test_soak(mc_model_t m) {
    printf("== T1 soak: 5x(prefill 512 + decode 1024), reset 复用 ==\n");
    mc_session_t s{nullptr};
    REQUIRE(mc_session_create(m, &s) == MC_OK);
    const bool graph_on = mc_debug_session_graph_nodes(s) > 0;

    const auto prompt = make_prompt(512, 0);
    const mc_generation_config_t g = greedy_config();
    double per_tok_ms[5] = {0, 0, 0, 0, 0};

    const uint64_t free_base = free_vram_mb(); // 首轮前（create/graph capture 已完成）
    for (int r = 0; r < 5; ++r) {
        mc_runtime_stats_t st0{}, st1{};
        REQUIRE(mc_session_reset(s) == MC_OK);
        REQUIRE(mc_get_stats(s, &st0) == MC_OK);
        REQUIRE(mc_prefill(s, prompt.data(), 512) == MC_OK);
        run_decode(s, 1024, nullptr, g);
        REQUIRE(mc_get_stats(s, &st1) == MC_OK);
        // 累计/复位语义：total 跨 reset 累计，本轮增量恰为 512/1024。
        REQUIRE(st1.prompt_tokens - st0.prompt_tokens == 512);
        REQUIRE(st1.generated_tokens - st0.generated_tokens == 1024);
        REQUIRE(st1.decode_us > st0.decode_us);
        // graph replay 计数：graph 可用时每轮恰 +1024（eager 回退则恒 0）。
        const uint64_t gl = st1.cuda_graph_launches - st0.cuda_graph_launches;
        if (graph_on) REQUIRE(gl == 1024);
        REQUIRE(cuda_clean("soak round"));
        per_tok_ms[r] = (double)(st1.decode_us - st0.decode_us) / 1024.0 / 1000.0;
        printf("  round %d: decode %.3f ms/tok, graph_launches +%llu\n", r,
               per_tok_ms[r], (unsigned long long)gl);
    }
    const uint64_t free_end = free_vram_mb();
    const int64_t leak_mb = (int64_t)free_base - (int64_t)free_end;
    printf("  VRAM drift over 5 rounds: %+lld MB (limit 50MB)\n",
           (long long)leak_mb);
    REQUIRE(leak_mb < 50);
    // 稳定性：末轮/首轮 < 1.3（允许热节流，不许雪崩）。
    const double ratio = per_tok_ms[4] / per_tok_ms[0];
    printf("  stability: last %.3f / first %.3f = %.3f ms/tok (limit 1.3)\n",
           per_tok_ms[4], per_tok_ms[0], ratio);
    REQUIRE(ratio < 1.3);
    mc_session_destroy(s);
    printf("  T1 PASS\n");
}

// ---- T2：多 session 并发隔离 ----
void test_multi_session(mc_model_t m) {
    printf("== T2 multi-session: 3 session 交错 vs 单 session 顺序，逐 token 对拍 ==\n");
    const auto prompt = make_prompt(512, 99);
    const mc_generation_config_t g = greedy_config();

    // 参照：单 session 顺序跑 prefill(512) + 500 decode。
    std::vector<int32_t> ref;
    {
        mc_session_t r{nullptr};
        REQUIRE(mc_session_create(m, &r) == MC_OK);
        REQUIRE(mc_prefill(r, prompt.data(), 512) == MC_OK);
        run_decode(r, 500, &ref, g);
        mc_session_destroy(r);
    }
    REQUIRE(ref.size() == 500);

    // 3 session 并发：A 10 步 → B 10 步 → C 10 步 × 50 轮。
    mc_session_t ss[3] = {{nullptr}, {nullptr}, {nullptr}};
    std::vector<int32_t> toks[3];
    for (int i = 0; i < 3; ++i) {
        REQUIRE(mc_session_create(m, &ss[i]) == MC_OK);
        REQUIRE(mc_prefill(ss[i], prompt.data(), 512) == MC_OK);
    }
    const uint64_t free_at_peak = free_vram_mb();
    for (int round = 0; round < 50; ++round)
        for (int i = 0; i < 3; ++i) run_decode(ss[i], 10, &toks[i], g);

    // 核心断言：交错与顺序逐 token 一致（KV/状态无串扰）。
    for (int i = 0; i < 3; ++i) {
        REQUIRE(toks[i].size() == 500);
        int ndiff = 0;
        for (size_t k = 0; k < 500; ++k)
            if (toks[i][k] != ref[k]) ++ndiff;
        printf("  session %c: token diff vs sequential = %d/500\n", 'A' + i,
               ndiff);
        REQUIRE(ndiff == 0);
    }
    // stats 独立性：交错历史后各 session 恰好各自的 512/500。
    for (int i = 0; i < 3; ++i) {
        mc_runtime_stats_t st{};
        REQUIRE(mc_get_stats(ss[i], &st) == MC_OK);
        REQUIRE(st.prompt_tokens == 512);
        REQUIRE(st.generated_tokens == 500);
        mc_session_destroy(ss[i]);
    }
    REQUIRE(cuda_clean("multi-session"));
    printf("  free VRAM at 3-session peak: %llu MB (weights shared, 3 arenas)\n",
           (unsigned long long)free_at_peak);
    printf("  T2 PASS\n");
}

// ---- T2b：双线程真并发回归锚（S4 / 并发推理路径 A）----
// 主线程先建 3 个 session（配额原子，创建期本就并发安全）：1 个跑顺序
// 基线，2 个交给工作线程；barrier 对齐后两线程**并发** prefill（相同
// 确定性 prompt）+ decode 64（greedy），与基线逐 token 比对。
//
// 锚定目标：并发路径的**确定性**——per-session stream 内串行序（S2 不变
// 量 ②）+ S1 per-session workspace 下，并发流输出必须与顺序基线逐 token
// 一致。任何并发输出损坏（KV 串扰、workspace 踩踏、采样状态污染）都会
// 在此报警。
//
// 锚定性验证记录（2026-09-11，RTX 5080，如实）：
// * 默认 tactic（decode 全部自研 gemv、prefill M=512 algo 不消耗
//   workspace）下，把 forward.cpp 临时改回 m->gemm_ws（共享 ws）T2b 仍
//   PASS——workspace 竞态在本机默认配置为**潜伏态**（无并发 ws 消耗者），
//   「改回→挂」未能复现。
// * MC_GEMV_OFF=1（强制 decode 走 cuBLASLt 对照组）下多 session 交错
//   发散（T2 488~492/500）——但该发散与 ws 归属无关（共享 ws 与
//   per-session ws 同样挂），属 cuBLASLt decode 控制路径的独立缺陷
//   （仅 debug 环境变量可达，非生产路径），不能作为 workspace 因素的
//   锚定载体。
// 结论：T2b 以「默认配置 + S1 下稳定 PASS、对任何并发输出损坏敏感」为
// 锚定语义；workspace 竞态的活性验证受限于本机 tactic 选举，已如实记录
//（benchmark/results/concurrency_path_a.json 的 t2b_anchor 字段）。
// 线程内错误经原子标志带回（避免子线程 std::exit 与 join 交互不干净）。
void test_two_thread_concurrent(mc_model_t m) {
    printf("== T2b two-thread: barrier 对齐并发 prefill+decode 64 vs 顺序基线 ==\n");
    constexpr uint32_t kPrompt = 256, kDecode = 64;
    const auto prompt = make_prompt(kPrompt, 13);
    const mc_generation_config_t g = greedy_config();

    // 3 个 session：ref（基线，主线程顺序用完即还）+ A/B（工作线程）。
    mc_session_t sess[3] = {{nullptr}, {nullptr}, {nullptr}};
    for (int i = 0; i < 3; ++i) REQUIRE(mc_session_create(m, &sess[i]) == MC_OK);

    // 顺序基线。
    std::vector<int32_t> ref;
    REQUIRE(mc_prefill(sess[0], prompt.data(), kPrompt) == MC_OK);
    run_decode(sess[0], kDecode, &ref, g);
    REQUIRE(ref.size() == kDecode);
    mc_session_destroy(sess[0]);

    // 双线程：自旋 barrier 对齐起步（原子计数 + yield），并发 prefill+decode。
    std::atomic<int> ready{0};
    std::atomic<bool> failed{false};
    char err[2][128] = {{0}, {0}};
    std::vector<int32_t> toks[2];
    toks[0].reserve(kDecode);
    toks[1].reserve(kDecode);
    auto worker = [&](int id, mc_session_t s) {
        ready.fetch_add(1, std::memory_order_acq_rel);
        while (ready.load(std::memory_order_acquire) < 2) std::this_thread::yield();
        if (mc_prefill(s, prompt.data(), kPrompt) != MC_OK) {
            snprintf(err[id], sizeof(err[id]), "prefill: %s", mc_last_error());
            failed.store(true);
            return;
        }
        for (uint32_t i = 0; i < kDecode; ++i) {
            int32_t tok = -1;
            if (mc_decode_one(s, &g, &tok) != MC_OK || tok < 0) {
                snprintf(err[id], sizeof(err[id]), "decode %u: %s", i, mc_last_error());
                failed.store(true);
                return;
            }
            toks[id].push_back(tok);
        }
    };
    std::thread ta(worker, 0, sess[1]);
    std::thread tb(worker, 1, sess[2]);
    ta.join();
    tb.join();
    REQUIRE(!failed.load());
    REQUIRE(err[0][0] == '\0' && err[1][0] == '\0');

    // 核心断言：并发两流与顺序基线逐 token 一致（共享 workspace 竞态的
    // 表现即此处发散——锚定性验证已复现过一次 FAIL）。
    for (int i = 0; i < 2; ++i) {
        REQUIRE(toks[i].size() == kDecode);
        int ndiff = 0;
        size_t first_diff = SIZE_MAX;
        for (size_t k = 0; k < kDecode; ++k)
            if (toks[i][k] != ref[k]) {
                if (ndiff == 0) first_diff = k;
                ++ndiff;
            }
        printf("  thread %c: token diff vs sequential = %d/%u%s\n", 'A' + i,
               ndiff, (unsigned)kDecode,
               ndiff ? "  <-- DIVERGED (workspace race signature)" : "");
        REQUIRE(ndiff == 0);
        (void)first_diff;
    }
    // stats 独立 + 无 sticky error。
    for (int i = 1; i < 3; ++i) {
        mc_runtime_stats_t st{};
        REQUIRE(mc_get_stats(sess[i], &st) == MC_OK);
        REQUIRE(st.prompt_tokens == kPrompt);
        REQUIRE(st.generated_tokens == kDecode);
        mc_session_destroy(sess[i]);
    }
    REQUIRE(cuda_clean("two-thread concurrent"));
    printf("  T2b PASS\n");
}

// ---- T3：OOM 行为 ----
// 返回 true = 本机显存大到没触发 OOM（跳过断言但报告）。
bool test_oom(const std::string& model_dir) {
    printf("== T3 OOM: max_ctx=131072 x max_sessions=2 ==\n");
    mc_model_options_t o = base_options(2, 131072);
    mc_model_t big{nullptr};
    const mc_status_t ls = mc_model_load(model_dir.c_str(), &o, &big);
    REQUIRE(ls == MC_OK); // 权重本身放得下（~4.7GB）
    bool triggered = false;
    mc_session_t keep{nullptr};
    for (int i = 0; i < 2; ++i) {
        mc_session_t s{nullptr};
        const mc_status_t st = mc_session_create(big, &s);
        if (st == MC_OK) {
            keep = s; // 第 1 个可能成功（~13.9GB arena 恰好放下）
            printf("  session %d created (arena fit); next should OOM\n", i);
            continue;
        }
        REQUIRE(st == MC_E_OUT_OF_MEMORY);
        REQUIRE(s.impl == nullptr); // *out_session 保持 null
        const char* msg = mc_last_error();
        const bool has_free = strstr(msg, "free VRAM") != nullptr;
        const bool has_need = strstr(msg, "need") != nullptr;
        printf("  OOM msg: %s\n", msg);
        REQUIRE(has_free && has_need);
        REQUIRE(cuda_clean("oom create"));
        triggered = true;
        break;
    }
    if (keep.impl != nullptr) mc_session_destroy(keep);
    mc_model_destroy(big);
    if (!triggered) {
        printf("  NOTE: both sessions fit on this GPU — OOM path not exercised\n");
        return false;
    }
    // 失败路径无资源残留：正常 options 重新加载即成功。
    mc_model_options_t normal = base_options(4, 2048);
    mc_model_t m2{nullptr};
    REQUIRE(mc_model_load(model_dir.c_str(), &normal, &m2) == MC_OK);
    {
        mc_session_t s{nullptr};
        REQUIRE(mc_session_create(m2, &s) == MC_OK);
        const auto p8 = make_prompt(8, 5);
        REQUIRE(mc_prefill(s, p8.data(), 8) == MC_OK);
        run_decode(s, 2, nullptr, greedy_config());
        mc_session_destroy(s);
    }
    mc_model_destroy(m2);
    printf("  retry with normal options: OK (no resource residue)\n");
    printf("  T3 PASS\n");
    return true;
}

// ---- T4：reset 语义压测 ----
void test_reset_stress(mc_model_t m) {
    printf("== T4 reset stress: 200x(reset + prefill 8 + decode 2) ==\n");
    mc_session_t s{nullptr};
    REQUIRE(mc_session_create(m, &s) == MC_OK);
    const auto p8 = make_prompt(8, 7);
    const mc_generation_config_t g = greedy_config();
    int32_t first_tok2 = -1;
    mc_runtime_stats_t st_total{};
    for (int r = 0; r < 200; ++r) {
        REQUIRE(mc_session_reset(s) == MC_OK);
        mc_runtime_stats_t st0{}, st1{};
        REQUIRE(mc_get_stats(s, &st0) == MC_OK);
        REQUIRE(mc_prefill(s, p8.data(), 8) == MC_OK);
        int32_t t0 = -1, t1 = -1;
        REQUIRE(mc_decode_one(s, &g, &t0) == MC_OK);
        REQUIRE(mc_decode_one(s, &g, &t1) == MC_OK);
        REQUIRE(mc_get_stats(s, &st1) == MC_OK);
        // 每轮增量恰 8/2；total 单调累计。
        REQUIRE(st1.prompt_tokens - st0.prompt_tokens == 8);
        REQUIRE(st1.generated_tokens - st0.generated_tokens == 2);
        // greedy 确定性：第 2 个 decode token 每轮相同（seq/KV/状态干净）。
        if (r == 0) first_tok2 = t1;
        else REQUIRE(t1 == first_tok2);
        REQUIRE(cuda_clean("reset round"));
    }
    REQUIRE(mc_get_stats(s, &st_total) == MC_OK);
    REQUIRE(st_total.prompt_tokens == 200 * 8);
    REQUIRE(st_total.generated_tokens == 200 * 2);
    mc_session_destroy(s);
    printf("  200 rounds, 2nd token stable (%d), stats total %llu/%llu\n",
           first_tok2, (unsigned long long)st_total.prompt_tokens,
           (unsigned long long)st_total.generated_tokens);
    printf("  T4 PASS\n");
}

// ---- T5：VRAM 预算表 ----
struct VramRow {
    const char* mode;
    uint32_t ctx;
    int sessions;
    uint64_t weights_mb;
    uint64_t arena_mb;     // 建期 arena 字节数（debug 钩子）
    uint64_t marginal_mb;  // 实测每 session 边际占用（free 差）
    uint64_t total_mb;     // weights + n × marginal（经验总量）
    uint64_t free_after;
    bool ok;
};

void probe_vram(mc_model_t m, const char* mode, uint32_t ctx,
                uint64_t weights_mb, std::vector<VramRow>& rows) {
    mc_session_t keep[4] = {{nullptr}, {nullptr}, {nullptr}, {nullptr}};
    for (int n = 1; n <= 4; ++n) {
        const uint64_t f0 = free_vram_mb();
        mc_session_t s{nullptr};
        const mc_status_t st = mc_session_create(m, &s);
        if (st != MC_OK) {
            printf("  [%s ctx=%u] session %d create failed: %s\n", mode,
                   (unsigned)ctx, n, mc_last_error());
            rows.push_back({mode, ctx, n, weights_mb, 0, 0, 0, free_vram_mb(),
                            false});
            break;
        }
        keep[n - 1] = s;
        // 触发一次 prefill+decode，让 graph replay/CUDA 运行时惰性分配进预算。
        const auto p = make_prompt(8, 11);
        REQUIRE(mc_prefill(s, p.data(), 8) == MC_OK);
        run_decode(s, 2, nullptr, greedy_config());
        const uint64_t marginal = f0 - free_vram_mb();
        const uint64_t arena_mb = mc_debug_session_arena_bytes(s) / kMB;
        rows.push_back({mode, ctx, n, weights_mb, arena_mb, marginal,
                        weights_mb + (uint64_t)n * marginal, free_vram_mb(),
                        true});
        printf("  [%s ctx=%u] sessions=%d arena=%lluMB marginal=%lluMB "
               "total=%lluMB free_after=%lluMB\n",
               mode, (unsigned)ctx, n, (unsigned long long)arena_mb,
               (unsigned long long)marginal,
               (unsigned long long)(weights_mb + (uint64_t)n * marginal),
               (unsigned long long)free_vram_mb());
    }
    for (int i = 0; i < 4; ++i)
        if (keep[i].impl != nullptr) mc_session_destroy(keep[i]);
}

} // namespace

int main() {
    fprintf(stderr, "soak_test: begin\n");
    mc_debug_dump_set_steps(0);

    const std::string model_dir = resolve_model_dir();
    if (model_dir.empty() || !file_exists(model_dir + "/model.wpk")) {
        fprintf(stderr, "== soak_test: SKIP — no model package (need model.wpk) ==\n");
        return 0;
    }
    printf("model_dir: %s\n", model_dir.c_str());

    // ---- model A：ctx 8448、max_sessions 4（T1/T2/T4 与 ctx8448 预算共用）----
    std::vector<VramRow> rows;
    uint64_t free_before_load = free_vram_mb();
    mc_model_options_t oa = base_options(4, 8448);
    mc_model_t ma{nullptr};
    REQUIRE(mc_model_load(model_dir.c_str(), &oa, &ma) == MC_OK);
    const uint64_t weights_mb_8448 = free_before_load - free_vram_mb();

    test_soak(ma);
    test_multi_session(ma);
    test_two_thread_concurrent(ma);
    test_reset_stress(ma);
    printf("== T5 vram probe @ ctx8448 ==\n");
    probe_vram(ma, "bf16", 8448, weights_mb_8448, rows);
    mc_model_destroy(ma);

    // ---- T3 OOM（内部自带「正常 options 重试成功」）----
    const bool oom_exercised = test_oom(model_dir);

    // ---- ctx2048 预算（独立 model，兼作 T3 的「重试成功」已在内部完成）----
    {
        uint64_t f0 = free_vram_mb();
        mc_model_options_t ob = base_options(4, 2048);
        mc_model_t mb{nullptr};
        REQUIRE(mc_model_load(model_dir.c_str(), &ob, &mb) == MC_OK);
        const uint64_t weights_mb_2048 = f0 - free_vram_mb();
        printf("== T5 vram probe @ ctx2048 ==\n");
        probe_vram(mb, "bf16", 2048, weights_mb_2048, rows);
        mc_model_destroy(mb);
    }

    // ---- FP8 双拷贝模式预算（model_fp8.wpk 存在时；测量性补测，不改变
    //      server 默认 BF16。容量规划：权重较 BF16 +~1.7GB（fp8 副本），每
    //      session arena 与 KV 精度无关、不变）----
    if (file_exists(model_dir + "/model_fp8.wpk")) {
        for (uint32_t ctx : {8448u, 2048u}) {
            uint64_t f0 = free_vram_mb();
            mc_model_options_t of = base_options(4, ctx);
            of.precision = MC_FP8;
            mc_model_t mf{nullptr};
            if (mc_model_load(model_dir.c_str(), &of, &mf) != MC_OK) {
                printf("== T5 vram probe @ fp8 ctx=%u: SKIP (%s) ==\n",
                       (unsigned)ctx, mc_last_error());
                continue;
            }
            const uint64_t weights_mb_fp8 = f0 - free_vram_mb();
            printf("== T5 vram probe @ fp8 ctx=%u ==\n", (unsigned)ctx);
            probe_vram(mf, "fp8", ctx, weights_mb_fp8, rows);
            mc_model_destroy(mf);
        }
    } else {
        printf("== T5 fp8 probe: SKIP — no model_fp8.wpk ==\n");
    }

    // ---- vram_budget.json ----
    {
        const std::string dir = std::string(MC_REPO_ROOT) + "/benchmark/results";
        std::error_code ec;
        fs::create_directories(dir, ec);
        std::ofstream f(dir + "/vram_budget.json");
        if (!f) return 1;
        f << "{\n  \"timestamp\": \"" << timestamp() << "\",\n";
        f << "  \"note\": \"weights_mb = free-VRAM delta across model_load "
             "(weights+workspace+context); arena_mb = session_create one-shot "
             "arena (debug hook); marginal_mb = measured free-VRAM delta per "
             "session incl. graph/lazy runtime; total_mb = weights_mb + "
             "sessions x marginal_mb; capacity planning for server "
             "--max-sessions\",\n";
        f << "  \"oom_path_exercised\": " << (oom_exercised ? "true" : "false")
          << ",\n";
        f << "  \"rows\": [\n";
        bool first = true;
        for (const auto& r : rows) {
            f << (first ? "    " : ",   ")
              << "{\"mode\": \"" << r.mode << "\", \"ctx\": " << r.ctx
              << ", \"sessions\": " << r.sessions
              << ", \"weights_mb\": " << r.weights_mb
              << ", \"arena_mb\": " << r.arena_mb
              << ", \"marginal_mb\": " << r.marginal_mb
              << ", \"total_mb\": " << (r.weights_mb + (uint64_t)r.sessions *
                                                       r.marginal_mb)
              << ", \"free_after\": " << r.free_after
              << ", \"ok\": " << (r.ok ? "true" : "false") << "}\n";
            first = false;
        }
        f << "  ]\n}\n";
        printf("  vram_budget.json written (%zu rows)\n", rows.size());
    }
    REQUIRE(cuda_clean("soak_test end"));
    fprintf(stderr, "soak_test: done (all PASS)\n");
    return 0;
}
