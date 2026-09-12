// debug_dump.h — §17.2 层级 dump（仅 debug，Release 默认关闭不进 hot path）。
//
// 开关：环境变量 MC_DEBUG_DUMP_DIR=<dir> 且 MC_DEBUG_DUMP_STEPS=N（默认 0）。
// 语义：N>0 时 dump step0（prefill：层张量 + logits/final_hidden/token）及
// 后续 decode 步 step{t}（logits/final_hidden/token），直到步号 ≥ N。
// 文件布局与 golden 完全一致（flat，case 目录内）：
//   layer{i}.hidden_in.f32 / layer{i}.attn_out.f32 / layer{i}.mlp_out.f32（2048，仅 step0）
//   step{t}.logits.f32（130560）、step{t}.final_hidden.f32（2048）、step{t}.token（i32）
// 数值：f32 = bf16 upcast（D2H bf16 → host 转 fp32 写盘）。
// 张量语义（与 python 权威 tools/reference_forward.py 一致）：
//   hidden_in=层输入（residual 流入）；attn_out=o 投影并入残差后的 residual
//   末行（add_bf16(x, ao) 之后）；mlp_out=down 投影并入残差后的 residual 末行
//   （add_bf16(x, d) 之后）；final_hidden=final_norm 之后的末行（hf[-1]）。
#pragma once

#include <cstdint>

#include <cuda_runtime.h>

#include "minicpm_runtime.h"

namespace mc::dump {

// 开关状态（getenv 只读一次，静态缓存；默认全关，hot path 一次原子判断）。
bool enabled();
// 允许 dump 的最大步号（不含）：step0..step{N-1}；0 = 关。
int max_steps();
const char* dir();
// 分窗种子输出：MC_DEBUG_DUMP_WINDOW_SEED=1 时，截断窗（layer_max<42）在
// prefill 末尾把 residual 全行以 bf16 原位 dump 到 {dir}/window_seed.bf16
//（gpu_layerwise 链式窗口的下一窗种子）。
bool seed_out();
// 运行时改步数上限（0=关）：供测试在压力段前关 dump，避免数千步 ×522KB
// 磁盘洪水。非线程安全（与 dump 本身一样仅编排期调用）。setenv 无效
// 是因为 env 只在首次访问时读一次。
void set_steps(int n);

// 写盘原语（内部 D2H + bf16→fp32 转换 + fwrite；仅 debug 模式调用）。
// 文件已存在则覆盖。
mc_status_t write_bf16_as_f32(const char* path, const uint16_t* d_data,
                              size_t count, cudaStream_t stream);
// bf16 原位写盘（无转换；窗口种子链要求位级精确，禁止 f32 往返）。
mc_status_t write_bf16_raw(const char* path, const uint16_t* d_data, size_t count,
                           cudaStream_t stream);
mc_status_t write_token(const char* path, int32_t token);

} // namespace mc::dump

// 编排期 dump 步数开关（0=关闭后续步）。additive debug 钩子：不属于冻结
// ABI 契约（minicpm_runtime.h）的一部分，仅供测试/编排进程经 .so 调用
//（real_inference 压力段前关 dump，避免数千步 ×522KB 磁盘洪水）。
#ifdef __cplusplus
extern "C" {
#endif
void mc_debug_dump_set_steps(int steps);

// 分窗种子钩子（gpu_layerwise）：把 host_bf16_rows（rows×hidden 个 bf16，
// 行主序）写入 session 的 residual 流 device buffer，作为本窗 forward 的
// 输入（embedding gather 被跳过——loader 分窗未加载 embedding 时）。
// 同为 additive debug 钩子，不属于冻结 ABI。
mc_status_t mc_debug_seed_residual(mc_session_t session, const void* host_bf16_rows,
                                   uint32_t rows);
#ifdef __cplusplus
}
#endif
