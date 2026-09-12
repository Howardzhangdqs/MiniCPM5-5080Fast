// gemm.h — cuBLASLt BF16 GEMM 封装（计划 §8.2：QKV/O/gate_up/down/lm_head）。
//
// 语义：C[M,N] = activation[M,K] · weight[N,K]ᵀ（两者均行主）。参数序与
// cublasLt 列主映射一致（实现权威，kernel_ops §5.2 按此对拍）：
//   d_a = 权重 [N,K] 行主（GEMM 的 B 矩阵）；
//   d_b = activation [M,K] 行主（GEMM 的 A 矩阵）。
// 数值：A/B/C 均 CUDA_R_16BF，compute CUBLAS_COMPUTE_32F（fp32 累加，任务书
// 数值规范）。列主映射：cublasLtMatmul(opA=T, opB=N, m=N, n=M, k=K)：
//   A := W_col[K,N] ld=K（权重），OP_T → [N,K]
//   B := x_col[K,M] ld=K（activation），OP_N → [K,M]
//   C := C_col[N,M] ld=N  ≡ C row-major [M,N]
//
// Heuristic 计划按 (M,N,K) 缓存（decode M=1 的 5 个固定 shape 永远命中缓存，
// steady-state 零 host 分配；prefill M 变化允许查询，超上限后走未缓存路径）。
// workspace 由调用方传入：热路径（forward）= per-session workspace
//（SessionImpl::gemm_ws，S1 起），建期 gemv tactic 快测 = model 级
// shared_workspace。heuristic 以可用字节数过滤，不会超申请。
#pragma once

#include <cstddef>
#include <cstdint>

#include <cublasLt.h>
#include <cuda_runtime.h>

#include "minicpm_runtime.h"

namespace mc {

// model 级 GEMM 引擎：持有 cublasLtHandle 与 heuristic 计划缓存。
// model_load 创建 / model_destroy 释放；可被多 session 共享（内部有锁，
// 但单 session 串行调用为主，锁只保护缓存表）。
class GemmEngine {
public:
    GemmEngine() = default;
    GemmEngine(const GemmEngine&) = delete;
    GemmEngine& operator=(const GemmEngine&) = delete;

    mc_status_t init();    // cublasLtCreate
    void destroy();        // cublasLtDestroy + 释放缓存

    // 预热 decode 稳态的 5 个固定 M=1 shape（qkv/o/gate_up/down/lm_head）：
    // 建期（session_create 尾部）完成 build_plan + heuristic 查询并写入
    // 缓存，首个 decode step 不再发生 heuristic/host 分配（§7.2 纪律）。
    // 已缓存的 shape 跳过（多 session 重复调用幂等）。ws_size 用于
    // heuristic 的 workspace 过滤（与实际执行一致）。
    mc_status_t prewarm_decode_shapes(size_t ws_size);

    // C = A·Bᵀ（见上）。workspace/ws_size 由调用方提供（热路径 =
    // per-session workspace；见文件头注释）。
    mc_status_t gemm_bf16(cudaStream_t stream, void* workspace, size_t ws_size,
                          uint32_t M, uint32_t N, uint32_t K,
                          const void* d_a, const void* d_b, void* d_c) const;

    // ---- F2a：FP8 W8A8 GEMM（批解码 F2b / prefill F3 共享引擎层）----
    // 语义：Y[M,N] = (W_fp8[N,K] ⊙ w_scale[N]) · (X_fp8[M,K] ⊙ x_scale[M])ᵀ，
    // D 直接 bf16。d_w_fp8/d_w_scale = wpk 的 fp8 权重副本 + per-row scale
    //（solo gemv-fp8 同源，不改包格式）。
    //
    // scale 模式（建期探测一次，日志注明所选模式）：
    //   1) 首选 CUBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F：w_scale[n]×
    //      x_scale[m] 外积缩放核内完成，无后处理 kernel；
    //   2) 回退 per-tensor scalar（A/B scale=1.0）+ launch_post_scale_rows
    //      补偿（额外 2·M·N·2B 读写/GEMM）。
    //
    // gemm_fp8：内部含 per-token 量化（launch_quant_rows → d_x_fp8_scratch
    //   [M,K] + d_x_scale_scratch [M]，由调用方提供）→ GEMM（→ 回退时
    //   post-scale）。F2b 接线的直接形态。
    // gemm_fp8_q：免量化变体（激活已 e4m3 + scale，F2b 若上游融合量化时用）。
    //
    // 约束：K%16==0 且 d_w/d_x/d_y 16B 对齐（fp8 GEMM 的 ld/指针要求）；
    // plan 按 (M,N,K) 独立缓存（与 bf16 计划分表）；heuristic 以 ws_size
    // 过滤（同 gemm_bf16 口径）。数值：CUBLAS_COMPUTE_32F 累加。
    mc_status_t gemm_fp8(cudaStream_t stream, void* workspace, size_t ws_size,
                         uint32_t M, uint32_t N, uint32_t K,
                         const void* d_w_fp8, const float* d_w_scale,
                         const uint16_t* d_x_bf16, uint16_t* d_y_bf16,
                         void* d_x_fp8_scratch, float* d_x_scale_scratch) const;

    mc_status_t gemm_fp8_q(cudaStream_t stream, void* workspace, size_t ws_size,
                           uint32_t M, uint32_t N, uint32_t K,
                           const void* d_w_fp8, const float* d_w_scale,
                           const void* d_x_fp8, const float* d_x_scale,
                           uint16_t* d_y_bf16) const;

    // 建期探测结果：0=未探测，1=OUTER_VEC_32F（首选），2=per-tensor+post
    //（回退）。首次 gemm_fp8 调用时探测并缓存（std::printf 日志）。
    int fp8_scale_mode() const;

    cublasLtHandle_t handle() const { return handle_; }
    size_t cache_size() const;

private:
    cublasLtHandle_t handle_ = nullptr;
    struct Impl;
    Impl* impl_ = nullptr; // 计划缓存（pimpl，避免在头里拖 STL/锁）
};

} // namespace mc
