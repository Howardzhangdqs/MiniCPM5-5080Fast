// minicpm_runtime.h — MiniCPM5 native GPU runtime C ABI v0.1
//
// Stable Host/GPU boundary between the Rust Host Runtime and the C++/CUDA
// GPU Runtime. See MiniCPM5_RTX5080_Rust_CUDA_实施计划.md §5 for the
// ABI contract:
//   - extern "C" only; fixed-width ints; POD structs; opaque handles
//   - no C++ exceptions, std::string/std::vector across the boundary
//   - model is immutable & shared; session is NOT thread-safe
//   - evolving structs carry struct_size (+ abi_version where noted)
//   - mc_last_error() is thread-local diagnostics, not an error protocol
//
// Changelog:
//   v0.1 (0x00020000) — initial ABI per plan §5.3: model load/destroy,
//     session create/reset/destroy, prefill, decode_one, stats, last_error.
//   v0.1-batch (B1b, 2026-09) — additive batch-group surface (only-add,
//     nothing above changes): mc_batch_options_t + mc_batch_group_create/
//     add/remove/step/destroy. B1 语义：eager 精确 n 步、仅 greedy。

#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define MC_ABI_VERSION 0x00020000u

typedef struct mc_model mc_model_t;
typedef struct mc_session mc_session_t;

typedef enum mc_status_t {
    MC_OK = 0,
    MC_E_INVALID_ARGUMENT = 1,
    MC_E_IO = 2,
    MC_E_CUDA = 3,
    MC_E_MODEL_MISMATCH = 4,
    MC_E_OUT_OF_MEMORY = 5,
    MC_E_GRAPH = 6,
    MC_E_UNSUPPORTED = 7,
    MC_E_INTERNAL = 100
} mc_status_t;

typedef enum mc_precision_t {
    MC_BF16 = 0,
    MC_FP16 = 1,
    MC_FP8 = 2,
    MC_NVFP4 = 3
} mc_precision_t;

typedef enum mc_kv_precision_t {
    MC_KV_BF16 = 0,
    MC_KV_FP8 = 1
} mc_kv_precision_t;

// sizeof(mc_model_options_t) == 40 on x86_64; required at call time.
typedef struct mc_model_options_t {
    uint32_t struct_size;        /* caller must set to sizeof(this struct) */
    uint32_t abi_version;        /* caller must set to MC_ABI_VERSION      */
    int32_t device_id;
    uint32_t max_sessions;
    uint32_t max_context_tokens;
    mc_precision_t precision;
    mc_kv_precision_t kv_precision;
    uint32_t enable_cuda_graph;
    uint64_t reserve_vram_bytes;
} mc_model_options_t;

// sizeof(mc_generation_config_t) == 32 on x86_64; required at call time.
typedef struct mc_generation_config_t {
    uint32_t struct_size;        /* caller must set to sizeof(this struct) */
    uint32_t max_new_tokens;
    float temperature;
    float top_p;
    uint32_t top_k;
    float repetition_penalty;
    uint64_t seed;
} mc_generation_config_t;

typedef struct mc_runtime_stats_t {
    uint64_t prompt_tokens;
    uint64_t generated_tokens;
    uint64_t prefill_us;
    uint64_t decode_us;
    uint64_t peak_vram_bytes;
    uint64_t cuda_graph_launches;
} mc_runtime_stats_t;

/* ---- lifecycle ---- */

// Loads a model package. `model_dir` is a UTF-8, NUL-terminated directory
// path. On failure *out_model stays NULL and a non-MC_OK status is returned.
mc_status_t mc_model_load(
    const char* model_dir,
    const mc_model_options_t* options,
    mc_model_t* out_model);

void mc_model_destroy(mc_model_t model);

mc_status_t mc_session_create(
    mc_model_t model,
    mc_session_t* out_session);

// Reuses all allocations (arena, stream, graphs, page tables); only clears
// sequence state, KV page ownership and RNG state.
mc_status_t mc_session_reset(mc_session_t session);

void mc_session_destroy(mc_session_t session);

/* ---- inference ---- */

// Host token ids in [0, vocab_size). Performs pinned staging + H2D copy.
mc_status_t mc_prefill(
    mc_session_t session,
    const int32_t* token_ids,
    uint32_t token_count);

// One decode step. Returns the sampled token via *out_token; the same token
// also stays in a fixed device buffer for the next embedding lookup (plan
// §10.4) — the host never needs to send it back.
mc_status_t mc_decode_one(
    mc_session_t session,
    const mc_generation_config_t* config,
    int32_t* out_token);

/* ---- observability ---- */

mc_status_t mc_get_stats(
    mc_session_t session,
    mc_runtime_stats_t* out_stats);

const char* mc_last_error(void);

/* ---- batched decode（B1b：多 session 合并解码，路径 B）----
 * 批组（batch group）持有一条组 stream + 组集中激活/logits/partials arena；
 * 成员 session 以 slot（0..max_slots-1）加入，保留各自的 KV 池/页表/序列
 * 状态（组经 device 指针表按 slot 寻址成员私有量）。
 * 线程契约：全部组调用来自调用者单线程（同 session 的 solo 调用一致）；
 * 成员在组内期间禁止并发 solo 使用（add 时校验忙态）。
 * 生命周期：session 在组内期间不得 destroy（调用方契约）。
 * B1 语义：eager 精确 n 步（slot_ids 即本步成员集，无 bucket/padding，
 * 图捕获是 B2）；mc_batch_step 仅 greedy —— 任一 cfg 的采样参数非 greedy
 * （temperature>0 且 top_k!=1）返回 MC_E_UNSUPPORTED（B2 再批采样；
 * rep_penalty 字段在 greedy 步与 solo 一致地被忽略）。
 * mc_batch_step 的发放语义沿用 decode_one 的预决策：返回的 out_tokens[i]
 * 是成员「上一步已决策」token（首步 = prefill 末 token 的预决策）。
 */

typedef struct mc_batch_group mc_batch_group_t;

/* sizeof(mc_batch_options_t) == 32 on x86_64; required at call time.
 * max_slots ≤ 8；max_context_tokens 必须与成员 session 的模型档一致
 * （同 ctx 档 → 同 KV 布局与 attention chunk 档）。 */
typedef struct mc_batch_options_t {
    uint32_t struct_size;        /* caller must set to sizeof(this struct) */
    uint32_t abi_version;        /* caller must set to MC_ABI_VERSION      */
    uint32_t max_slots;          /* 1..8 */
    uint32_t max_context_tokens; /* 与成员模型一致（ctx/chunk 档匹配校验） */
    uint32_t reserved[4];        /* 必须清零（自描述 ABI 的兼容扩展先例） */
} mc_batch_options_t;

/* 创建批组：组 stream、组 arena（激活族 + logits[max][vocab] + partials×2 +
 * gemm workspace 16MB + device 指针表固定地址 + pinned 收割缓冲）、停车槽
 * （1 页 dummy KV/页表/pos；B2 padding 用，B1 eager 不使用）、gemm 计划预热。
 * 仅真实权重（model.wpk）模型可用。 */
mc_status_t mc_batch_group_create(
    mc_model_t model,
    const mc_batch_options_t* options,
    mc_batch_group_t* out_group);

/* 成员入组（分配一个空闲 slot）。拒绝（MC_E_UNSUPPORTED）：成员模型与组
 * 模型不同、ctx/chunk 档不匹配、容量满、session 已在（任一）组内（忙态）。
 * add 建立成员流→组流的单向事件链（成员此前的 prefill/KV 写对组可见），
 * 并把成员私有量指针写入组 device 指针表对应槽（≤64B 小 H2D，表地址固定）。 */
mc_status_t mc_batch_group_add(
    mc_batch_group_t group,
    mc_session_t session,
    uint32_t* slot_id);

/* 成员出组（quantum 边界调用：两次 batch_step 之间）。建立组流→成员流的
 * 单向事件链（组的 KV/序列写对此后的 solo 调用可见）。 */
mc_status_t mc_batch_group_remove(
    mc_batch_group_t group,
    uint32_t slot_id);

/* 批 decode 一步（eager，精确 n）：slot_ids[0..n) 为本步成员集（不可重复），
 * cfgs 为逐成员生成配置（B1 仅 greedy），out_tokens[0..n) 收割发放 token。
 * 内部流序执行 + sync + pinned 收割（同步语义同 decode_one）。 */
mc_status_t mc_batch_step(
    mc_batch_group_t group,
    const uint32_t* slot_ids,
    uint32_t n,
    const mc_generation_config_t* cfgs,
    int32_t* out_tokens);

/* 销毁批组：仍在组的成员自动经历 remove 语义（组流 record → 成员流 wait）
 * 后释放全部组资源。幂等（null 安全 no-op）。 */
void mc_batch_group_destroy(mc_batch_group_t group);

#ifdef __cplusplus
} /* extern "C" */
#endif
