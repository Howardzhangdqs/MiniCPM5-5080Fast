// internal_state.h — mc_model / mc_session 内部实现（Pimpl 风格）。
//
// ABI 头文件按值传递 mc_model_t / mc_session_t（struct mc_model 等 opaque
// typedef），因此 C++ 侧把这些 struct 补全为「单个 impl 指针」的 8 字节
// 可拷贝 handle（x86_64 SysV 下等价于传指针，一个寄存器）；
// 真正的状态在 ModelImpl / SessionImpl 中。
//
// 线程模型（ABI 契约 §5）：
//   - ModelImpl：load 完成后不可变，可被多 session / 多线程共享
//   - SessionImpl：非线程安全。一个 session 是一个独占的可变生成状态对象
//   - gemm workspace 已 per-session 化（SessionImpl::gemm_ws，session arena
//     内切片）：多 session 各自的 stream 并发执行 cuBLASLt 时互不踩踏
//     workspace —— 跨 session 并发 stream 安全（前提：每 session 的全部
//     native 调用来自单线程，即每 session 单线程调用）。
//   - ModelImpl::gemm_ws（model 级 workspace）仅供建期 gemv tactic 快测
//     （gemv_tactic.cpp，session_create 期 + mutex 保护，无竞态）。
//     热路径（forward）禁用 —— 多 stream 并用同一块 workspace 会互相
//     覆盖中间结果，跨 stream 不安全。
#pragma once

#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>

#include <cuda_runtime.h>

#include "kernels/sm120/kernels.h" // SamplingConfigDevice / splitmix64
#include "minicpm_runtime.h"
#include "runtime/cuda_graph.h"
#include "runtime/generated_model_config.h"
#include "runtime/gemm.h"
#include "runtime/kv_cache.h"
#include "runtime/memory_arena.h"
#include "runtime/weight_loader.h"

struct ModelImpl;

// ---- ABI handle（补全 minicpm_runtime.h 的 opaque struct）----
struct mc_model {
    ModelImpl* impl;
};

struct SessionImpl;

// B1b：批组前置声明（batch_group.cpp）。session 在组内期间由该指针标记
// 忙态（add 校验/清空于 remove）；生产 solo 路径不读写此字段。
struct BatchGroupImpl;

// B1b：批组 ABI handle（同 mc_model/mc_session 的单指针可拷贝款式）。
struct mc_batch_group {
    BatchGroupImpl* impl;
};

struct mc_session {
    SessionImpl* impl;
};

// 计时/计数（chrono steady_clock 微秒累计）。
struct SessionStats {
    uint64_t prompt_tokens    = 0;
    uint64_t generated_tokens = 0;
    uint64_t prefill_us       = 0;
    uint64_t decode_us        = 0;
};

// device 侧固定地址的 session 状态（计划 §11.1/§12.2/§12.4）。
// seq_len 变化只改本结构内容，不改变任何 tensor pointer。
// &seq_len 是 decode 热路径 kernel（rope/attention/append）读取位置的固定
// device 地址（P3 CUDA Graph 可捕获性）。
// rng_state：64-bit device RNG（PCG32）状态 —— 显存驻留，kernel 内推进，
// graph replay 安全（§12.4「Sampling 参数变化只改值」同源）。
// cfg：采样配置（§14.2）——decode_one 每次调用从 pinned 暂存 H2D 到
// &d_state->cfg（graph 外小拷贝），采样 kernel 运行时读取。
struct SessionStateDevice {
    uint32_t seq_len;      // offset 0
    uint32_t next_token;   // offset 4（与前 8B 连续：eager/prefill 的状态
                           // 回写只拷这 8 字节，不触碰 rng/cfg）
    uint64_t rng_state;    // offset 8（8B 对齐）
    SamplingConfigDevice cfg; // offset 16（kernels.h）
};

struct ModelImpl {
    // ---- load 后不可变 ----
    int         device_id = 0;
    uint32_t    max_sessions = 1;
    uint32_t    max_context_tokens = 0;
    uint64_t    reserve_vram_bytes = 0;
    std::string model_dir;
    bool        has_wpk = false; // model.wpk 存在 → 真实 BF16 权重；缺失 → mock
    bool        enable_cuda_graph = false; // P3：session 期尝试 decode graph capture
    uint64_t    initial_free_vram = 0;
    uint64_t    total_vram = 0;
    uint64_t    weight_bytes = 0;
    WeightScanResult weight_scan;

    // 共享 runtime workspace（16MB 对 BF16 heuristic 足够——preference 以
    // 可用字节过滤 algo）。
    // S1 起热路径不再使用：forward 的 gemm workspace 已 per-session 化
    //（SessionImpl::gemm_ws）。此块仅供建期 gemv tactic 快测使用
    //（gemv_tactic.cpp：单线程 + mutex，无竞态）。
    DeviceArena workspace;
    void*       gemm_ws = nullptr; // = workspace 基址
    size_t      gemm_ws_bytes = 0;

    // P2 真实权重（has_wpk 时有效）
    DeviceArena weight_arena; // 单次 cudaMalloc 全部权重
    DeviceArena fp8_arena;    // P4：fp8 量化权重（precision=MC_FP8；否则空）
    WeightRefs  weights;

    // model 级 cuBLASLt（load 创建 / destroy 释放；可被多 session 共享）
    mc::GemmEngine gemm;

    // ---- 少量原子状态（多线程 create/destroy session 用；mutable：model
    //      整体不可变，唯独这个配额计数例外）----
    mutable std::atomic<uint32_t> active_sessions{0};

    // P3+ decode GEMV tactic（计划 §8.2/§10.2）：首个真实 session 建期快测
    //（runtime/gemv_tactic.cpp），此后进程内只读 —— 所有 session / eager /
    // graph 路径同 kernel，bit 一致性由此保证。mutable 同上（惰性初始化）。
    // 值：0=cuBLASLt（该 shape 自研未胜出）、1=gemv-rows、2=gemv-splitK。
    mutable std::mutex gemv_mu;
    mutable bool gemv_tactic_ready = false;
    mutable int32_t gemv_tactic[5] = {0, 0, 0, 0, 0};
    mutable uint32_t gemv_split_s[5] = {1, 1, 1, 1, 1};

    // cudaMemGetInfo 包装（纯查询，不是分配，steady-state 允许）。
    mc_status_t sample_free_vram(uint64_t& free_out) const;
};

// P2 activation scratch（全部 bf16，按 max_context_tokens 一次预留）：
// 每层流水线中间量复用固定 buffer，steady-state 零分配（计划 §7.2/§10.3）。
// 元素数/token：residual 2048 + normed 2048 + qkv 2560 + ctx 2048 +
// o_out 2048 + gate_up 12288 + mlp_mid 6144 + down_out 2048 = 31232
// + logits(vocab) —— logits 只需一行，见下。embedding gather 直接写
// residual 流起点（无独立 emb 切片）。
struct ActivationScratch {
    uint16_t* residual  = nullptr; // [T, 2048]（residual stream，原地累加；gather 起点）
    uint16_t* residual_alt = nullptr; // [2048] P5 decode ping-pong 行（融合 prologue 的
                                      // 交替缓冲；prefill/独立 kernel 路径不使用）
    uint16_t* normed    = nullptr; // [T, 2048]
    uint16_t* qkv       = nullptr; // [T, 2560]
    uint16_t* ctx       = nullptr; // [T, 2048]（attention ctx，o 投影输入）
    uint16_t* o_out     = nullptr; // [T, 2048]（o 投影输出 = attn_out dump）
    uint16_t* gate_up   = nullptr; // [T, 12288]
    uint16_t* mlp_mid   = nullptr; // [T, 6144]（swiglu 输出）
    uint16_t* down_out  = nullptr; // [T, 2048]（down 投影输出 = mlp_out dump）
    uint16_t* logits    = nullptr; // [vocab]（末行 logits，bf16）
    // P3+ attention 两段式中间量（kernels.h kAttnPartialStride=130）：
    // [16 q head][attn_max_chunks][(m, l, acc[128])] fp32。固定地址
    // （CUDA Graph 兼容，§12.2）；42 层间按流序复用同一 buffer。
    // 尺寸量级：max_ctx 8192 → 16×32×130×4B ≈ 266KB。
    float*    attn_partials = nullptr;
    // P8-A1 两级 merge 的 level-2 partials [16][kMergeSplit2=32][130] fp32
    //（266KB；merge_a 写 → merge_b 读，42 层流序复用；固定地址 graph 安全）。
    float*    attn_partials2 = nullptr;
    // P3 split-KV prefill flash partials [T·16·S_kv][130] fp32（S_kv =
    // prefill_flash_split_kv(T)，kernels.h）：flash 写 → prefill_merge 读，
    // 42 层流序复用；固定地址（eager 路径，无 graph 约束）。40MB 固定切片
    //（T·S_kv ≤ 4096 → 峰值 34.07MB，见 session.cpp compute_sizes）。
    float*    prefill_partials = nullptr;
    // P3+ GEMV split-K partial（[S, N] fp32，gemv_tactic.h 的口径）：
    // 固定地址 scratch；仅 split-K tactic 的 shape 使用。~80KB。
    float*    gemv_partial  = nullptr;
    // P4+ 采样（§14.2）：rep-penalty 字节位图 [vocab]（每步 memset）+
    // penalty 后的 fp32 logits [vocab]（采样主体统一在 fp32 上工作）。
    // 固定地址（graph 兼容）；~0.65MB。
    uint8_t* rep_bitmap   = nullptr;
    float*   logits_f32   = nullptr;
    // 采样两段式的 partial（[kSampBlocks][kSampMaxK] 值/下标，64KB）
    float*   samp_partial_v = nullptr;
    int32_t* samp_partial_i = nullptr;
    // P5：RoPE inv_freq 表 [64]（建期一次写入，热路径只读）
    float*   rope_inv_freq = nullptr;
};

// ---- P5 perf 分解（MC_BREAKDOWN=1，session_create 期创建；基准/诊断专用，
//      默认关闭不进 hot path）----
// forward.cpp 在每个段边界 cudaEventRecord（eager：流内事件；graph capture：
// 成为 event record 节点入图 —— replay 后 host 读相邻事件差即为段时长）。
// 事件布局：[start][emb][7×42 层内段][lm_head][sample] = 298 事件 / 297 段。
namespace mc {
inline constexpr uint32_t kBdownLayerSegs = 7;  // qkv/rope/attn/o_res/gate_up/swiglu/down_res
inline constexpr uint32_t kBdownSegs = 1 /*emb*/ + kBdownLayerSegs + 2 /*lm_head+sample*/; // 10 桶
inline constexpr uint32_t kBdownEvents =
    2 + cfg::kNumLayers * kBdownLayerSegs + 2;   // 298（含 start）
inline constexpr uint32_t kBdownRawSegs = kBdownEvents - 1; // 297 相邻事件对
// 段桶名（collect 的聚合口径；层数学见 forward.cpp 的记录点注释）
inline const char* const kBdownSegNames[kBdownSegs] = {
    "emb",          "norm1+qkv",      "rope+kvwrite", "attn(chunk+merge)",
    "o+resnorm2",   "gate_up",        "swiglu",       "down+resnorm(next)",
    "lm_head",      "sample"};
// 原始段索引 → 桶索引（0=emb；1..294 层段循环；295=lm_head；296=sample）
inline uint32_t bdown_bucket(uint32_t raw) {
    if (raw == 0u) return 0u;
    if (raw == kBdownRawSegs - 1u) return kBdownSegs - 2u; // lm_head
    if (raw == kBdownRawSegs) return kBdownSegs - 1u;      // sample
    return 1u + ((raw - 1u) % kBdownLayerSegs);
}
} // namespace mc

// 分解计时状态（事件建期创建、跨 step 复用；n_used 每次 forward 重置）
struct BreakdownState {
    bool        enabled = false;
    uint32_t    n_used  = 0;
    cudaEvent_t events[mc::kBdownEvents] = {};
};

struct SessionImpl {
    // ---- 建期固定，此后只读 ----
    const ModelImpl* model = nullptr;
    cudaStream_t     stream = nullptr; // session 独占 stream

    DeviceArena         arena;  // 一次性大块：seq/page table/KV pool/scratch
    PagedKvAllocator    kv;     // page table + ownership（P3 真实读写）
    GraphManager        graph;  // P3 decode CUDA Graph（capture 失败则 available()=false）
    ActivationScratch   act;    // P2 real forward 中间量（mock 模式为空）

    // P3 graph 状态：graph_mode = capture+instantiate 成功且未被 debug 钩子
    // 关闭；in_capture 标记 capture 区间（forward 内的调试同步/dump 禁用）。
    bool graph_mode  = false;
    bool in_capture  = false;

    // P4+ 采样（§14.2）：per-call host 模式（temperature>0 且 top_k>1 时为
    // true；graph 双图选一 / eager 分支共用）；device RNG 播种簿记。
    bool     sampling_step  = false;
    bool     dev_rng_seeded = false;
    uint64_t dev_rng_state  = 0;   // splitmix64(seed) 的初始状态（播种用）
    SamplingConfigDevice* h_cfg = nullptr; // pinned 暂存（建期 cudaHostAlloc）

    // P4+：采样变体 decode graph（capture 失败则采样回退 eager；
    // greedy graph 语义/节点序列不变）。
    GraphManager graph_sample;

    // P5 perf 分解（MC_BREAKDOWN=1；默认禁用，零开销）
    BreakdownState bdown;

    // P5：采样配置去重（内容未变时跳过每步的 pinned H2D 小拷贝）
    SamplingConfigDevice last_cfg{};
    bool cfg_dirty = true; // 首次必写

    // P3+ attention 两段式：chunk 上限（= attention_decode_max_chunks(
    // max_context_tokens, attn_chunk_pos)，建期固定 → grid.y 与 partials
    // 寻址恒定）。
    uint32_t attn_max_chunks = 0;
    // P8-A2：本会话的 decode attention chunk 位置数（建期按
    // attention_decode_chunk_for_ctx(max_context_tokens) 分档：≤16K→64、
    // ≤64K→128、>64K→256；session 内恒定 → launch 参数不变，graph 安全）。
    uint32_t attn_chunk_pos = 0;

    // 固定 device 地址（计划 §12.2，CUDA Graph 友好）
    SessionStateDevice* d_state = nullptr;
    int32_t*            d_seq = nullptr;        // [max_context_tokens]
    int32_t*            d_next_token = nullptr; // 采样结果常驻（§10.4）
    uint32_t*           d_page_table = nullptr; // [max_pages]
    void*               kv_pool = nullptr;      // Paged KV 数据区（P2 真实使用）
    void*               scratch = nullptr;      // 兼容字段（real 模式改用 act.*）

    // S1 per-session gemm workspace（session arena 内独立 256 对齐切片，
    // 建期 carve 一次，steady-state 复用）。多 session 的 stream 并发跑
    // cuBLASLt 时各用各的 workspace，互不踩踏。
    // 关键不变量：大小恒 16MB（与 model 级 workspace 同尺寸，real 模式；
    // mock 模式 0 字节保持空指针）→ cublasLt heuristic 的
    // MAX_WORKSPACE_BYTES 过滤条件不变 → 选出同一 algo → 数值逐 bit 不变
    //（gemm.cpp pick_algo 以 ws_size 过滤）。
    void*               gemm_ws = nullptr;
    size_t              gemm_ws_bytes = 0;

    // 回归锚（2026-09 采样持久污染事故）：RoPE inv_freq 表建期的 host 位级
    // 快照。表本应「建期一次写入、热路径只读」；采样 scratch 越界写穿表
    // 曾是持久污染根因。additive 钩子 mc_debug_session_rope_intact 用它
    // 断言任意 decode/采样/reset 序列后表未被改动。
    uint32_t rope_snapshot[64] = {};
    bool     rope_snapshot_valid = false;

    // pinned host buffer（建期 cudaHostAlloc，steady-state 只读写内容）
    int32_t* h_tokens = nullptr;      // prefill staging：max_context_tokens*4B
    int32_t* h_next_token = nullptr;  // 4B D2H 落点（§10.4 流式输出链路）

    // ---- 可变序列状态（host 权威，d_state 为 device 镜像）----
    uint32_t seq_len = 0;
    bool     rng_seeded = false;
    uint32_t rng_state = 0;
    SessionStateDevice h_state{}; // d_state 的 host 镜像（async 拷贝源）
    uint32_t debug_step = 0;      // debug dump 步号（0=prefill/step0；reset 清零）

    // B1b：所在批组（null = 空闲/不在组内）。仅 batch_group.cpp 读写；
    // solo 生产路径（forward/decode_executor/prefill）不触碰。
    BatchGroupImpl* batch_group = nullptr;

    // 统计：total 为生命周期累计（reset 不清）；since_reset 每轮清零。
    SessionStats total;
    SessionStats since_reset;

    uint64_t min_free_vram = ~0ull; // peak_vram_bytes = total_vram - min_free
};
