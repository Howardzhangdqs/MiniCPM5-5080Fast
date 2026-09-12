// abi_smoke.cpp — ABI 冒烟测试（CTest: abi_smoke）。真跑 GPU（RTX 5080）。
//
// 覆盖：
//  ① 正常流：load(mock) → create session → prefill{0,100,200} → decode 循环
//     直到 eos（max_new_tokens=32）→ stats 断言 prompt==3 / generated==32 →
//     reset → 再来一轮验证复用与确定性 → destroy。
//  ② 参数校验：struct_size 错、null 指针、越界 token、超长 prompt、
//     enable_cuda_graph=1 → load 成功（P3 已落地；mock 不 capture）等。
//  ③ steady-state：session 建立后连续 100 次 decode 无分配错误
//     （说明性断言：decode_us 累计 > 0）。
#include "minicpm_runtime.h"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <vector>

#define REQUIRE(cond)                                                          \
    do {                                                                       \
        if (!(cond)) {                                                         \
            fprintf(stderr, "FAIL %s:%d: %s (last_error: %s)\n", __FILE__,     \
                    __LINE__, #cond, mc_last_error());                         \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

// 公共头只前向声明 opaque handle；ABI 语义是「按值传递的单指针 handle」。
// 消费者侧复制等价布局（单指针，全局作用域）即可创建/检查 null handle。
struct mc_model { void* impl; };
struct mc_session { void* impl; };

namespace {

constexpr uint32_t kTestMaxContext = 8192; // 控制 KV pool 尺寸（BF16 约 336MiB）
constexpr int32_t  kEosStream = 130073;

mc_model_options_t base_options() {
    mc_model_options_t o{};
    o.struct_size = sizeof(o);
    o.abi_version = MC_ABI_VERSION;
    o.device_id = 0;
    o.max_sessions = 1;
    o.max_context_tokens = kTestMaxContext;
    o.precision = MC_BF16;
    o.kv_precision = MC_KV_BF16;
    o.enable_cuda_graph = 0;
    o.reserve_vram_bytes = 0;
    return o;
}

// 注：mock 链路（无 wpk）的采样字段不生效（mock_next_token 用 host 参数，
// 不读 device cfg）；真实采样路径由 fp8/graph_bench 的 sampling smoke 覆盖。
mc_generation_config_t base_gen(uint32_t max_new, uint64_t seed) {
    mc_generation_config_t g{};
    g.struct_size = sizeof(g);
    g.max_new_tokens = max_new;
    g.temperature = 0.8f;
    g.top_p = 0.95f;
    g.top_k = 0;
    g.repetition_penalty = 1.0f;
    g.seed = seed;
    return g;
}

void make_mock_model_dir(const std::filesystem::path& dir) {
    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
    std::filesystem::create_directories(dir, ec);
    // 放两个哑文件，验证 weight_loader 的枚举/报告链路（无 JSON 解析）。
    { std::ofstream f(dir / "config.json"); f << "{}"; }
    { std::ofstream f(dir / "tokenizer.json"); f << "{}"; }
    // 注意：不放 model.wpk —— 保证 mock 模式。
}

void section(const char* name) { fprintf(stderr, "== abi_smoke: %s ==\n", name); }

// --------------------------------------------------------------- ② 参数校验
void test_parameter_validation(const std::filesystem::path& model_dir) {
    section("parameter validation");

    mc_model_t m{nullptr};

    { // struct_size 错 → MC_E_INVALID_ARGUMENT，*out_model 保持 nullptr
        mc_model_options_t o = base_options();
        o.struct_size = sizeof(o) + 4;
        REQUIRE(mc_model_load(model_dir.string().c_str(), &o, &m) ==
                MC_E_INVALID_ARGUMENT);
        REQUIRE(m.impl == nullptr);
    }
    { // abi_version 错
        mc_model_options_t o = base_options();
        o.abi_version = 0xDEADu;
        REQUIRE(mc_model_load(model_dir.string().c_str(), &o, &m) ==
                MC_E_INVALID_ARGUMENT);
        REQUIRE(m.impl == nullptr);
    }
    { // null options / null model_dir / null out_model
        mc_model_options_t o = base_options();
        REQUIRE(mc_model_load(model_dir.string().c_str(), nullptr, &m) ==
                MC_E_INVALID_ARGUMENT);
        REQUIRE(m.impl == nullptr);
        REQUIRE(mc_model_load(nullptr, &o, &m) == MC_E_INVALID_ARGUMENT);
        REQUIRE(m.impl == nullptr);
        REQUIRE(mc_model_load(model_dir.string().c_str(), &o, nullptr) ==
                MC_E_INVALID_ARGUMENT);
    }
    { // 精度不支持（FP16/NVFP4 仍拒绝）→ MC_E_UNSUPPORTED
        mc_model_options_t o = base_options();
        o.precision = MC_FP16;
        REQUIRE(mc_model_load(model_dir.string().c_str(), &o, &m) ==
                MC_E_UNSUPPORTED);
        REQUIRE(m.impl == nullptr);
    }
    { // P4：precision=MC_FP8 已支持——但 mock 目录无 model_fp8.wpk →
      // MC_E_MODEL_MISMATCH（缺包显式报错，不静默降级）。真实 fp8 包的
      // 成功路径由 fp8_inference ctest 覆盖（SKIP 语义）。
        mc_model_options_t o = base_options();
        o.precision = MC_FP8;
        REQUIRE(mc_model_load(model_dir.string().c_str(), &o, &m) ==
                MC_E_MODEL_MISMATCH);
        REQUIRE(m.impl == nullptr);
        const char* msg = mc_last_error();
        REQUIRE(msg != nullptr && strstr(msg, "model_fp8.wpk") != nullptr);
    }
    { // KV 精度不支持（当前仅 BF16 KV）→ MC_E_UNSUPPORTED
        mc_model_options_t o = base_options();
        o.kv_precision = MC_KV_FP8;
        REQUIRE(mc_model_load(model_dir.string().c_str(), &o, &m) ==
                MC_E_UNSUPPORTED);
        REQUIRE(m.impl == nullptr);
    }
    { // enable_cuda_graph=1 → P3 已落地：load 必须成功（mock 模型无真实
      // forward，不 capture graph，session 仍为 eager；真实 wpk 模型在
      // session_create 尾部 capture，失败自动回退）
        mc_model_options_t o = base_options();
        o.enable_cuda_graph = 1;
        REQUIRE(mc_model_load(model_dir.string().c_str(), &o, &m) == MC_OK);
        REQUIRE(m.impl != nullptr);
        mc_model_destroy(m);
        m = mc_model_t{nullptr};
    }
    { // 不存在的目录 → IO 错误
        mc_model_options_t o = base_options();
        REQUIRE(mc_model_load("/nonexistent/dir/for/minicpm", &o, &m) == MC_E_IO);
        REQUIRE(m.impl == nullptr);
    }
    { // session_create 参数校验
        mc_model_options_t o = base_options();
        REQUIRE(mc_model_load(model_dir.string().c_str(), &o, &m) == MC_OK);
        REQUIRE(m.impl != nullptr);
        mc_session_t s{nullptr};
        REQUIRE(mc_session_create(mc_model_t{nullptr}, &s) == MC_E_INVALID_ARGUMENT);
        REQUIRE(s.impl == nullptr);
        REQUIRE(mc_session_create(m, nullptr) == MC_E_INVALID_ARGUMENT);

        REQUIRE(mc_session_create(m, &s) == MC_OK);
        REQUIRE(s.impl != nullptr);

        // decode 前置校验：null config / null out_token / 错 struct_size / 空 seq
        int32_t tok = -1;
        mc_generation_config_t g = base_gen(4, 7);
        REQUIRE(mc_decode_one(s, nullptr, &tok) == MC_E_INVALID_ARGUMENT);
        REQUIRE(mc_decode_one(s, &g, nullptr) == MC_E_INVALID_ARGUMENT);
        mc_generation_config_t bad = g;
        bad.struct_size = 8;
        REQUIRE(mc_decode_one(s, &bad, &tok) == MC_E_INVALID_ARGUMENT);
        REQUIRE(mc_decode_one(s, &g, &tok) == MC_E_INVALID_ARGUMENT); // seq 为空

        // prefill 校验：null / 0 长度 / 越界 token
        const int32_t ok3[3] = {0, 100, 200};
        const int32_t oob[2] = {0, 130560};
        REQUIRE(mc_prefill(s, nullptr, 3) == MC_E_INVALID_ARGUMENT);
        REQUIRE(mc_prefill(s, ok3, 0) == MC_E_INVALID_ARGUMENT);
        REQUIRE(mc_prefill(s, oob, 2) == MC_E_INVALID_ARGUMENT);
        REQUIRE(mc_prefill(mc_session_t{nullptr}, ok3, 3) == MC_E_INVALID_ARGUMENT);

        // 超过 max_context_tokens → MC_E_OUT_OF_MEMORY
        std::vector<int32_t> too_long(kTestMaxContext + 1, 0);
        REQUIRE(mc_prefill(s, too_long.data(), (uint32_t)too_long.size()) ==
                MC_E_OUT_OF_MEMORY);

        // stats 参数校验
        mc_runtime_stats_t st{};
        REQUIRE(mc_get_stats(mc_session_t{nullptr}, &st) == MC_E_INVALID_ARGUMENT);
        REQUIRE(mc_get_stats(s, nullptr) == MC_E_INVALID_ARGUMENT);

        // 正常 prefill/decode 供后续步骤（同时验证上述失败未破坏状态）
        REQUIRE(mc_prefill(s, ok3, 3) == MC_OK);
        REQUIRE(mc_decode_one(s, &g, &tok) == MC_OK);

        mc_session_destroy(s);
        mc_model_destroy(m);
    }
}

// -------------------------------------------------------------- ① 正常生命周期
void test_normal_lifecycle(const std::filesystem::path& model_dir) {
    section("normal lifecycle (mock forward)");

    mc_model_options_t o = base_options();
    mc_model_t m{nullptr};
    REQUIRE(mc_model_load(model_dir.string().c_str(), &o, &m) == MC_OK);
    REQUIRE(m.impl != nullptr);

    mc_session_t s{nullptr};
    REQUIRE(mc_session_create(m, &s) == MC_OK);
    REQUIRE(s.impl != nullptr);

    // max_sessions=1：第二个 session 必须被拒绝（宽松断言：非 MC_OK）
    mc_session_t s2{nullptr};
    REQUIRE(mc_session_create(m, &s2) != MC_OK);
    REQUIRE(s2.impl == nullptr);

    // ---- 第一轮：prefill 3 tokens，decode 到 eos（max_new_tokens=32）----
    // 收尾语义（OpenAI/vLLM 对齐）：32 次调用全部返回真实 mock token，
    // 第 33 次调用返回 eos 哨兵（不计数、不推进序列）。
    const int32_t prompt[3] = {0, 100, 200};
    REQUIRE(mc_prefill(s, prompt, 3) == MC_OK);

    mc_generation_config_t g = base_gen(32, 42);
    std::vector<int32_t> round1;
    int32_t tok = -1;
    for (uint32_t guard = 0; guard < 256; ++guard) {
        REQUIRE(mc_decode_one(s, &g, &tok) == MC_OK);
        round1.push_back(tok);
        if (tok == kEosStream) break;
    }
    REQUIRE(round1.size() == 33);             // 32 个真实 token + eos 哨兵
    REQUIRE(round1.back() == kEosStream);     // 哨兵在第 33 次调用到达
    for (size_t i = 0; i + 1 < round1.size(); ++i) {
        REQUIRE(round1[i] >= 100 && round1[i] <= 1099); // mock 输出域
    }

    mc_runtime_stats_t st{};
    REQUIRE(mc_get_stats(s, &st) == MC_OK);
    REQUIRE(st.prompt_tokens == 3);
    REQUIRE(st.generated_tokens == 32);       // 哨兵调用不计数
    REQUIRE(st.cuda_graph_launches == 0);     // mock session 不走 graph replay
    fprintf(stderr,
            "  round1 stats: prompt=%llu generated=%llu prefill_us=%llu decode_us=%llu "
            "peak_vram_MiB=%.1f\n",
            (unsigned long long)st.prompt_tokens,
            (unsigned long long)st.generated_tokens,
            (unsigned long long)st.prefill_us, (unsigned long long)st.decode_us,
            (double)st.peak_vram_bytes / (1024.0 * 1024.0));

    // ---- reset：复用全部内存只清状态；再来一轮验证复用 + 确定性 ----
    REQUIRE(mc_session_reset(s) == MC_OK);
    REQUIRE(mc_prefill(s, prompt, 3) == MC_OK);

    mc_generation_config_t g5 = base_gen(5, 42); // 同 seed → 确定性复现
    for (uint32_t i = 0; i < 5; ++i) {
        REQUIRE(mc_decode_one(s, &g5, &tok) == MC_OK);
        REQUIRE(tok == round1[i]); // 5 个全部真实，与第一轮完全一致
    }
    REQUIRE(mc_decode_one(s, &g5, &tok) == MC_OK);
    REQUIRE(tok == kEosStream); // 第 6 次调用 = 越界哨兵（5 个真实之后）

    // 累计统计保留（reset 不清 total）：3+3 prompt，32+5 generated
    //（两轮的哨兵调用均不计数）
    REQUIRE(mc_get_stats(s, &st) == MC_OK);
    REQUIRE(st.prompt_tokens == 6);
    REQUIRE(st.generated_tokens == 37);

    mc_session_destroy(s);
    mc_model_destroy(m);
}

// -------------------------------------------------------------- ③ steady-state
void test_steady_state(const std::filesystem::path& model_dir) {
    section("steady-state decode (100 steps, zero allocation)");

    mc_model_options_t o = base_options();
    o.max_sessions = 2;
    mc_model_t m{nullptr};
    REQUIRE(mc_model_load(model_dir.string().c_str(), &o, &m) == MC_OK);
    REQUIRE(m.impl != nullptr);

    mc_session_t s{nullptr};
    REQUIRE(mc_session_create(m, &s) == MC_OK);

    const int32_t prompt[8] = {0, 1, 2, 3, 4, 5, 6, 7};
    REQUIRE(mc_prefill(s, prompt, 8) == MC_OK);

    // session 建成后连续 100 次 decode：无任何分配/错误（若 steady-state 里
    // 发生 cudaMalloc，正确性无从直接断言，但 DeviceArena 纪律 + 下面的
    // 说明性断言覆盖编排链路）。
    mc_generation_config_t g = base_gen(200, 1234); // 100 < 200，不会触发 eos
    int32_t tok = -1;
    for (int i = 0; i < 100; ++i) {
        REQUIRE(mc_decode_one(s, &g, &tok) == MC_OK);
        REQUIRE(tok >= 0 && tok < 130560);
    }

    mc_runtime_stats_t st{};
    REQUIRE(mc_get_stats(s, &st) == MC_OK);
    REQUIRE(st.generated_tokens == 100);
    REQUIRE(st.decode_us > 0); // 说明性断言：计时链路真实在工作
    fprintf(stderr, "  steady-state: 100 decodes, decode_us total=%llu\n",
            (unsigned long long)st.decode_us);

    mc_session_destroy(s);
    mc_model_destroy(m);
}

} // namespace

int main() {
    fprintf(stderr, "abi_smoke: begin (device 0, mock forward)\n");

    const std::filesystem::path model_dir = "mc_smoke_model_dir";
    make_mock_model_dir(model_dir);

    test_parameter_validation(model_dir);
    test_normal_lifecycle(model_dir);
    test_steady_state(model_dir);

    // null 安全性 + 释放后不崩。
    // 注：CUDA 12.8 无 cudaGetDeviceContextFree API，用 device sync +
    // MemGetInfo 作为等价的"至少不崩"检查。
    mc_session_destroy(mc_session_t{nullptr});
    mc_model_destroy(mc_model_t{nullptr});
    cudaError_t e = cudaDeviceSynchronize();
    REQUIRE(e == cudaSuccess);
    size_t free_b = 0, total_b = 0;
    REQUIRE(cudaMemGetInfo(&free_b, &total_b) == cudaSuccess);
    fprintf(stderr, "abi_smoke: all sections passed (free VRAM %.1f MiB of %.1f MiB)\n",
            (double)free_b / (1024.0 * 1024.0), (double)total_b / (1024.0 * 1024.0));
    return 0;
}
