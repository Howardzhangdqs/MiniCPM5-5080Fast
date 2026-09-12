// sampling_regression.cpp — 采样路径持久污染回归（2026-09 事故）。
//
// 事故根因（已修，见 session.cpp 采样 scratch 切分的回归锚注释）：采样
// pipeline 的候选列表 list_i 与 RoPE inv_freq 表在 session scratch 中重叠，
// 每个采样步的 compact kernel 用候选 token id（int32 位型）写穿建期只读的
// rope 表 → 之后本 session 一切 forward（含 prefill）位置编码全错；表建期
// 一次写入、reset 不重写 → 「采样请求之后 greedy 也坏、reset 救不回」。
//
// 覆盖（缺 model.wpk 时 SKIP+PASS，与 soak_test 同约定）：
//   R1 reset 恢复力（核心行为回归）：
//      a) 同 session：prefill→greedy 记 ref → 采样 8 步 → reset →
//         prefill→greedy 24 步 == ref 逐 token（graph 路径）；
//      b) 跨 session（eager，事故配置 enable_cuda_graph=0 的等价路径：
//         mc_debug_session_set_graph_enabled 关 replay）：同样 == ref。
//   R2 一步采样即污染探测：temperature→0⁺（采样管线激活但决策≈argmax）
//      连续 24 步 vs 纯 greedy —— 逐 token 比对，出现分叉时用 dump 的
//      step{t}.logits 判定是否 bf16 并列（并列则容忍并停止比对；非并列
//      分叉 = 采样步写穿了持久状态 → FAIL）。
//   R3 采样输出合法性 + top-k 成员性（python 参照的成员性断言代理）：
//      temp=1.0 采样 24 步，每步 token ∈ [0,vocab) 且 logit ≥ 该步
//      logits 的第 128 大值（top_k 规范化为 128、rep_penalty=1 恒等 →
//      dump 的 bf16-upcast logits 即采样器实际输入）。
//   R4 采样确定性：同 seed 两个 session 序列完全一致（device RNG 播种
//      确定性）；不同 seed 打印差异量（信息输出，不作硬断言）。
//   全程穿插 mc_debug_session_rope_intact==1（机制级断言：rope 表在任意
//   decode/采样/reset 序列后与建期快照位级一致）。
#include "minicpm_runtime.h"

// additive debug 钩子（libminicpm_native 导出；不属于冻结 ABI）。
extern "C" void mc_debug_dump_set_steps(int steps);
extern "C" void mc_debug_session_set_graph_enabled(mc_session_t session, int enabled);
extern "C" int mc_debug_session_rope_intact(mc_session_t session);

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
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

constexpr uint32_t kVocab = 130560;
constexpr uint32_t kTopK = 128; // decode_one 对 top_k=0 的规范化值
constexpr uint32_t kSteps = 24;

bool file_exists(const std::string& p) {
    std::error_code ec;
    return fs::exists(p, ec);
}

std::string resolve_model_dir() {
    if (const char* env = getenv("MC_MODEL_DIR")) return env;
    std::string repo = std::string(MC_REPO_ROOT) + "/models/minicpm5-2b";
    if (file_exists(repo + "/model.wpk")) return repo;
    std::string build = std::string(MC_BUILD_DIR) + "/wpk_model";
    if (file_exists(build + "/model.wpk")) return build;
    return {};
}

// 32 token：golden case_en 的 11 token + 本模型特殊 token（<think>=8、
// </think>=9、<|im_sep|>=4、</s>=1）与对话常见 id 的混合。
std::vector<int32_t> make_prompt() {
    std::vector<int32_t> p = {0,    24934, 280,  21417, 1323, 317,
                              12167, 9032,  62669, 4330,  35};
    const int32_t extra[] = {4, 8,  24934, 280, 9,   1,   4,   9032,
                             198, 27, 416,   1323, 1153, 1802, 305, 12612,
                             331, 390, 4314, 316,  2021};
    for (int32_t t : extra) p.push_back(t);
    for (int32_t t : p) REQUIRE(t >= 0 && (uint32_t)t < kVocab);
    return p;
}

mc_generation_config_t base_config() {
    mc_generation_config_t g{};
    g.struct_size = sizeof(g);
    g.max_new_tokens = 1u << 30; // 不触发 force-EOS
    g.top_p = 1.f;
    g.top_k = 0;                 // → 规范化为 128
    g.repetition_penalty = 1.f;  // penalty 恒等（成员性断言的前提）
    return g;
}

void run_decode(mc_session_t s, uint32_t n, const mc_generation_config_t& g,
                std::vector<int32_t>* out) {
    for (uint32_t i = 0; i < n; ++i) {
        int32_t tok = -1;
        REQUIRE(mc_decode_one(s, &g, &tok) == MC_OK);
        REQUIRE(tok >= 0 && (uint32_t)tok < kVocab);
        if (out) out->push_back(tok);
    }
}

int count_diff(const std::vector<int32_t>& a, const std::vector<int32_t>& b) {
    REQUIRE(a.size() == b.size());
    int d = 0;
    for (size_t i = 0; i < a.size(); ++i)
        if (a[i] != b[i]) ++d;
    return d;
}

// dump 的 step{t}.logits.f32（130560 fp32，bf16 upcast）→ host。
std::vector<float> read_step_logits(const std::string& dir, uint32_t t) {
    std::string path = dir + "/step" + std::to_string(t) + ".logits.f32";
    std::ifstream f(path, std::ios::binary);
    REQUIRE(f);
    std::vector<float> v(kVocab);
    f.read(reinterpret_cast<char*>(v.data()), (std::streamsize)(kVocab * 4));
    REQUIRE(f.gcount() == (std::streamsize)(kVocab * 4));
    return v;
}

float kth_largest_value(std::vector<float> v, uint32_t k) {
    REQUIRE(k >= 1 && k <= v.size());
    std::nth_element(v.begin(), v.begin() + (k - 1), v.end(),
                     std::greater<float>());
    return v[k - 1];
}

} // namespace

int main() {
    fprintf(stderr, "sampling_regression: begin\n");

    const std::string model_dir = resolve_model_dir();
    if (model_dir.empty() || !file_exists(model_dir + "/model.wpk")) {
        fprintf(stderr,
                "== sampling_regression: SKIP — no model package (need model.wpk) ==\n");
        return 0;
    }
    printf("model_dir: %s\n", model_dir.c_str());

    // dump 目录（R2/R3 用；env 只在首次访问时读 → 必须在一切 mc_* 调用前
    // setenv）。步数运行时用 mc_debug_dump_set_steps 分段开关。
    const std::string dump_dir =
        (fs::temp_directory_path() / "mc_sampling_regression_dump").string();
    std::error_code ec;
    fs::remove_all(dump_dir, ec);
    fs::create_directories(dump_dir, ec);
    REQUIRE(fs::is_directory(dump_dir));
    setenv("MC_DEBUG_DUMP_DIR", dump_dir.c_str(), 1);
    setenv("MC_DEBUG_DUMP_STEPS", "64", 1);
    mc_debug_dump_set_steps(0); // 默认关；R2/R3 段内开

    const auto prompt = make_prompt();
    const uint32_t np = (uint32_t)prompt.size();

    mc_model_options_t o{};
    o.struct_size = sizeof(o);
    o.abi_version = MC_ABI_VERSION;
    o.device_id = 0;
    o.max_sessions = 3;
    o.max_context_tokens = 2048;
    o.precision = MC_BF16;
    o.kv_precision = MC_KV_BF16;
    o.enable_cuda_graph = 1; // capture 失败自动回退 eager（语义一致）
    o.reserve_vram_bytes = 0;
    mc_model_t m{nullptr};
    REQUIRE(mc_model_load(model_dir.c_str(), &o, &m) == MC_OK);

    const mc_generation_config_t g_greedy = [] {
        auto g = base_config();
        g.temperature = 0.f;
        g.seed = 1;
        return g;
    }();
    const mc_generation_config_t g_sample = [] {
        auto g = base_config();
        g.temperature = 1.f;
        g.seed = 7;
        return g;
    }();

    // ---- R1a：同 session（graph 路径）采样→reset→greedy == 纯 greedy ----
    std::vector<int32_t> ref;
    {
        printf("== R1a same-session: greedy ref -> sample 8 -> reset -> greedy ==\n");
        mc_session_t s{nullptr};
        REQUIRE(mc_session_create(m, &s) == MC_OK);
        REQUIRE(mc_debug_session_rope_intact(s) == 1); // 基线：建期表完好
        REQUIRE(mc_prefill(s, prompt.data(), np) == MC_OK);
        run_decode(s, kSteps, g_greedy, &ref);
        REQUIRE(ref.size() == kSteps);
        REQUIRE(mc_debug_session_rope_intact(s) == 1); // greedy 不碰采样 scratch

        run_decode(s, 8, g_sample, nullptr);
        // 机制级断言：采样步之后 rope 表必须仍与建期快照位级一致
        //（事故版本在此被 list_i 写穿）。
        const int rope_ok = mc_debug_session_rope_intact(s);
        printf("  rope table intact after 8 sampling steps: %d\n", rope_ok);
        REQUIRE(rope_ok == 1);

        REQUIRE(mc_session_reset(s) == MC_OK);
        REQUIRE(mc_prefill(s, prompt.data(), np) == MC_OK);
        std::vector<int32_t> after;
        run_decode(s, kSteps, g_greedy, &after);
        const int d = count_diff(ref, after);
        printf("  reset-after-sampling vs pure-greedy diff: %d/%u\n", d, kSteps);
        REQUIRE(mc_debug_session_rope_intact(s) == 1);
        mc_session_destroy(s);
        REQUIRE(d == 0);
        printf("  R1a PASS\n");
    }

    // ---- R1b：跨 session + eager（事故配置：graph replay 关闭）----
    {
        printf("== R1b eager session: sample 8 -> reset -> greedy ==\n");
        mc_session_t s{nullptr};
        REQUIRE(mc_session_create(m, &s) == MC_OK);
        mc_debug_session_set_graph_enabled(s, 0); // 强制 eager（A/B 钩子）
        REQUIRE(mc_prefill(s, prompt.data(), np) == MC_OK);
        run_decode(s, 8, g_sample, nullptr);
        REQUIRE(mc_debug_session_rope_intact(s) == 1);
        REQUIRE(mc_session_reset(s) == MC_OK);
        REQUIRE(mc_prefill(s, prompt.data(), np) == MC_OK);
        std::vector<int32_t> after;
        run_decode(s, kSteps, g_greedy, &after);
        const int d = count_diff(ref, after);
        printf("  eager reset-after-sampling vs pure-greedy diff: %d/%u\n", d, kSteps);
        REQUIRE(mc_debug_session_rope_intact(s) == 1);
        mc_session_destroy(s);
        REQUIRE(d == 0);
        printf("  R1b PASS\n");
    }

    // ---- R2：temperature→0⁺ 采样 ≈ greedy（一步采样即污染探测器）----
    // 采样管线激活（temperature>0）但 exp((v-max)/T)→{0,1}：非并列时决策
    // 确定性等于 argmax。出现分叉仅当该步 logits 最大值 bf16 并列（greedy
    // 取小下标、采样按 u 取并列成员之一）—— 用 dump logits 验证并列后
    // 容忍并停止比对（其后轨迹合法分叉）。非并列分叉 = 持久状态被写穿。
    {
        printf("== R2 temp->0+ sampling vs greedy (tie-tolerant) ==\n");
        mc_debug_dump_set_steps(64);
        // 参照也带 dump 生成（与被测同 kernel 路径，逐 bit 可比）。两 session
        // 均强制 eager：graph replay 的图中无 dump 节点（capture 期 dump 被
        // 禁用），逐步 logits 需要 eager 路径。
        std::vector<int32_t> ref_d;
        {
            mc_session_t s{nullptr};
            REQUIRE(mc_session_create(m, &s) == MC_OK);
            mc_debug_session_set_graph_enabled(s, 0);
            REQUIRE(mc_prefill(s, prompt.data(), np) == MC_OK);
            run_decode(s, kSteps, g_greedy, &ref_d);
            mc_session_destroy(s);
        }
        mc_generation_config_t g_t0 = g_greedy;
        g_t0.temperature = 1e-30f; // >0 → 采样管线；决策≈argmax
        g_t0.seed = 3;
        std::vector<int32_t> got;
        {
            mc_session_t s{nullptr};
            REQUIRE(mc_session_create(m, &s) == MC_OK);
            mc_debug_session_set_graph_enabled(s, 0);
            REQUIRE(mc_prefill(s, prompt.data(), np) == MC_OK);
            run_decode(s, kSteps, g_t0, &got);
            REQUIRE(mc_debug_session_rope_intact(s) == 1);
            mc_session_destroy(s);
        }
        // emitted[i] = step{i}.token（prefill 预决策 = step0；decode t 的
        // 决策 dump 为 step{t}，由第 t+1 次调用发放 —— 与 out 收集对齐）。
        bool tie_stop = false;
        for (uint32_t i = 0; i < kSteps; ++i) {
            if (got[i] == ref_d[i]) continue;
            const auto lg = read_step_logits(dump_dir, i);
            const float va = lg[(size_t)ref_d[i]];
            const float vb = lg[(size_t)got[i]];
            const float vmax = *std::max_element(lg.begin(), lg.end());
            REQUIRE(va == vmax);        // greedy 参照取到最大值
            REQUIRE(vb == va);          // 采样分叉仅容忍于最大值并列
            printf("  tie at step %u: tokens %d/%d share max logit %.6f "
                   "(tolerated, stop comparing)\n",
                   i, ref_d[i], got[i], va);
            tie_stop = true;
            break;
        }
        if (!tie_stop) printf("  all %u tokens match greedy\n", kSteps);
        mc_debug_dump_set_steps(0);
        printf("  R2 PASS\n");
    }

    // ---- R3：采样输出合法性 + top-k 成员性 ----
    {
        printf("== R3 sampling membership: temp=1.0 x %u steps ==\n", kSteps);
        mc_debug_dump_set_steps(64);
        mc_session_t s{nullptr};
        REQUIRE(mc_session_create(m, &s) == MC_OK);
        mc_debug_session_set_graph_enabled(s, 0); // eager：逐步 logits dump
        REQUIRE(mc_prefill(s, prompt.data(), np) == MC_OK);
        mc_generation_config_t g = g_sample;
        g.seed = 11;
        std::vector<int32_t> got;
        run_decode(s, kSteps, g, &got);
        REQUIRE(mc_debug_session_rope_intact(s) == 1);
        mc_debug_dump_set_steps(0);
        int n_argmax = 0;
        for (uint32_t i = 0; i < kSteps; ++i) {
            const auto lg = read_step_logits(dump_dir, i);
            const float kth = kth_largest_value(lg, kTopK);
            REQUIRE(lg[(size_t)got[i]] >= kth); // ∈ top-128（值口径）
            if (lg[(size_t)got[i]] == *std::max_element(lg.begin(), lg.end()))
                ++n_argmax;
        }
        printf("  all %u sampled tokens in top-%u; argmax hits: %d/%u\n", kSteps,
               kTopK, n_argmax, kSteps);
        mc_session_destroy(s);
        printf("  R3 PASS\n");
    }

    // ---- R4：同 seed 确定性（跨 session）/ 不同 seed 差异（信息）----
    {
        printf("== R4 determinism: same seed across sessions ==\n");
        mc_generation_config_t g = g_sample;
        g.seed = 42;
        std::vector<int32_t> seq[2];
        for (int k = 0; k < 2; ++k) {
            mc_session_t s{nullptr};
            REQUIRE(mc_session_create(m, &s) == MC_OK);
            REQUIRE(mc_prefill(s, prompt.data(), np) == MC_OK);
            run_decode(s, 16, g, &seq[k]);
            mc_session_destroy(s);
        }
        REQUIRE(count_diff(seq[0], seq[1]) == 0);
        printf("  same-seed sequences identical (%zu tokens)\n", seq[0].size());
        mc_generation_config_t g2 = g;
        g2.seed = 43;
        std::vector<int32_t> other;
        {
            mc_session_t s{nullptr};
            REQUIRE(mc_session_create(m, &s) == MC_OK);
            REQUIRE(mc_prefill(s, prompt.data(), np) == MC_OK);
            run_decode(s, 16, g2, &other);
            mc_session_destroy(s);
        }
        printf("  seed=43 vs seed=42 diff: %d/16 (info)\n", count_diff(seq[0], other));
        printf("  R4 PASS\n");
    }

    // ---- R5：max_new_tokens 收尾语义（2026-09 对拍修复，OpenAI/vLLM 对齐）----
    // a) length：max=8 greedy → 恰 8 个真实内容 token；第 9 次调用返回
    //    eos 哨兵（幂等）；stats.generated == 8（哨兵不跑 forward、不推进
    //    d_seq/seq_len、不计数）。旧缺陷：第 8 次即强制 eos → 只流出 7 个
    //    内容 token 且被 host 误判 stop。
    // b) 自然 eos：France 首都问答（预填 </think>，enable_thinking=false
    //    模板等价），greedy 第 2 次调用自然产出 130073 —— 早于上限、原样
    //    返回（host 判 stop）；该步是真实 forward，计入 generated。
    {
        constexpr int32_t kEosIds[] = {1, 130073}; // generation_config.json
        printf("== R5a finish=length: max=8 greedy -> 8 content + sentinel ==\n");
        mc_session_t s{nullptr};
        REQUIRE(mc_session_create(m, &s) == MC_OK);
        REQUIRE(mc_prefill(s, prompt.data(), np) == MC_OK);
        mc_generation_config_t g8 = g_greedy;
        g8.max_new_tokens = 8;
        std::vector<int32_t> content;
        run_decode(s, 8, g8, &content);
        for (int32_t t : content)
            REQUIRE(t != kEosIds[0] && t != kEosIds[1]); // 8 个全是内容 token
        int32_t tok = -1;
        REQUIRE(mc_decode_one(s, &g8, &tok) == MC_OK);
        REQUIRE(tok == kEosIds[1]);      // 第 9 次调用 = 越界哨兵
        REQUIRE(mc_decode_one(s, &g8, &tok) == MC_OK);
        REQUIRE(tok == kEosIds[1]);      // 哨兵幂等
        mc_runtime_stats_t st{};
        REQUIRE(mc_get_stats(s, &st) == MC_OK);
        REQUIRE(st.prompt_tokens == np);
        REQUIRE(st.generated_tokens == 8); // 哨兵不计数
        REQUIRE(mc_debug_session_rope_intact(s) == 1);
        mc_session_destroy(s);
        printf("  R5a PASS (8 content tokens, stats.generated==8)\n");

        printf("== R5b natural eos: France prompt, eos at call 2 (< max) ==\n");
        // <s><|im_start|>user\nWhat is the capital of France? Answer with
        // only the city name.<|im_end|>\n<|im_start|>assistant\n
        // <think>\n\n</think>\n\n   （tokenizers 离线编码，greedy 确定
        // 性：1 个内容 token 后自然 <|im_end|>）
        const std::vector<int32_t> eos_prompt = {
            0,     130072, 8448, 220, 2928, 357,  285,  4894, 304,
            6918,  52,     10893, 401, 950,  285,  4321, 2546, 35,
            130073, 220,   130072, 130071, 220, 8,   220,  220,  9,
            220,   220};
        mc_session_t e{nullptr};
        REQUIRE(mc_session_create(m, &e) == MC_OK);
        mc_debug_session_set_graph_enabled(e, 0); // eager 路径同语义
        REQUIRE(mc_prefill(e, eos_prompt.data(), (uint32_t)eos_prompt.size()) ==
                MC_OK);
        int32_t t0 = -1, t1 = -1;
        REQUIRE(mc_decode_one(e, &g8, &t0) == MC_OK);
        REQUIRE(t0 != kEosIds[0] && t0 != kEosIds[1]); // 内容 token
        REQUIRE(mc_decode_one(e, &g8, &t1) == MC_OK);
        REQUIRE(t1 == kEosIds[1]); // 自然 eos（第 2 步 << max=8，非哨兵）
        mc_runtime_stats_t st2{};
        REQUIRE(mc_get_stats(e, &st2) == MC_OK);
        REQUIRE(st2.generated_tokens == 2); // 自然 eos = 真实 decode 步
        // eos 后 host 判停；若 host 继续调用仍产出真实 token（不闩锁），
        // 序列只随真实 token 前进。
        int32_t t2 = -1;
        REQUIRE(mc_decode_one(e, &g8, &t2) == MC_OK);
        REQUIRE(t2 >= 0 && (uint32_t)t2 < kVocab);
        REQUIRE(mc_debug_session_rope_intact(e) == 1);
        mc_session_destroy(e);
        printf("  R5b PASS (content %d then natural eos, generated==2)\n", t0);
    }

    mc_model_destroy(m);
    const cudaError_t e = cudaGetLastError();
    REQUIRE(e == cudaSuccess);
    fs::remove_all(dump_dir, ec);
    fprintf(stderr, "sampling_regression: done (all PASS)\n");
    return 0;
}
