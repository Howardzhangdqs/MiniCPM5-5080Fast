// forward.h — P2 真实 BF16 forward（embedding → 42 层 → final_norm → lm_head →
// argmax），prefill 与 decode 共用（计划 §8.1 单层执行图 / §9 / §10）。
#pragma once

#include "minicpm_runtime.h"

struct SessionImpl;

namespace mc {

// 执行 T 个 token 的完整 forward（T=1 即 decode step）。token ids 在 device
// （prefill: d_seq+base；decode: d_next_token）。位置 = seq_len .. seq_len+T-1
//（调用时 seq_len 尚未累加）。KV 写入这些位置；attention causal 覆盖
// 0..seq_len+T-1。结束时末行 logits 的 greedy argmax 写入 s.d_next_token
//（prefill 语义 = step0 预决策；decode 语义 = 下一 token 预决策，§10.4）。
// 真实 greedy 自然产生的 eos 不在此处理——由 decode 编排层原样返回。
mc_status_t run_real_forward(SessionImpl& s, const int32_t* d_ids, uint32_t T,
                             bool allow_layer_dump);

} // namespace mc
