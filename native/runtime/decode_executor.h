// decode_executor.h — decode 编排（计划 §8.1 单层图结构、§10.4 device token 链路）。
// MC_NATIVE_MOCK_FORWARD=1：mock_next_token kernel <<<1,1>>> 产生确定性 token；
//                           P2 用真实 embedding/42 层/lm_head/采样替换。
// P3：capture_decode_graph 在 session_create 尾部 capture decode CUDA Graph
//（计划 §12）；成功后 decode_one_run 走 graph replay 路径。
#pragma once

#include "minicpm_runtime.h"

struct SessionImpl;

namespace mc {

mc_status_t decode_one_run(SessionImpl& s, const mc_generation_config_t* config,
                           int32_t* out_token);

// P3（§12.3）：capture + instantiate decode CUDA Graph。成功后 s.graph_mode
// 置位且 s.graph.available()；失败返回非 MC_OK（调用方回退 eager，session
// 不失效）。仅真实（has_wpk）路径定义；mock 构建下无此符号。
mc_status_t capture_decode_graph(SessionImpl& s);

} // namespace mc
