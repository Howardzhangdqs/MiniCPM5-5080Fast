// prefill_executor.h — prefill 编排（计划 §5.4、§9）。
// pinned staging → cudaMemcpyAsync H2D → seq_len 累加 → KV page 增长 → 计时。
// 真实 layer forward（42 层）P2 接入；本阶段保证校验/拷贝/统计链路真实。
#pragma once

#include "minicpm_runtime.h"

struct SessionImpl;

namespace mc {

mc_status_t prefill_run(SessionImpl& s, const int32_t* token_ids, uint32_t token_count);

// 有效 prefill 分块 token 数（单次 forward 允许的最大 T）。
// 默认 8448；env MC_PREFILL_CHUNK 可覆盖（一次读取，非法值回退默认并
// stderr 警告）。session.cpp 的激活 scratch 分配与本文件的分块循环共用
// 同一口径（min(max_ctx, chunk)），见实现处注释。
uint32_t prefill_chunk_tokens();

} // namespace mc
