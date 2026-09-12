// gemv_tactic.h — P3+ decode GEMV 的 tactic 选型与 dispatch（计划 §8.2/§10.2）。
//
// session 建期（prewarm 之后、graph capture 之前）对 5 个 decode M=1 shape
// 做一次快测（cuBLASLt vs 自研 rows vs split-K，cudaEvent 计时，真实权重
// 指针），按每 shape 选赢家；首个真实 session 测定后缓存于 ModelImpl
// （同进程所有 session 复用 → eager/graph 两条路径与所有 session 必然同
// kernel，bit 一致性由此保证）。decode/lm_head 的 M=1 GEMM 走赢家；
// M>1（prefill）保持 cuBLASLt 不动。
//
// 环境开关（debug/基准用，建期读一次，不进 hot path）：
//   MC_GEMV_OFF=1 → 全部 shape 强制 cuBLASLt（E2E 前后对比的对照组）。
#pragma once

#include <cstdint>

#include <cuda_runtime.h>

#include "kernels/sm120/kernels.h"
#include "minicpm_runtime.h"
#include "runtime/generated_model_config.h"

struct ModelImpl;
struct SessionImpl;

namespace mc {

// decode 稳态 5 个固定 M=1 shape 的索引；-1 = 非固定 shape（走 cuBLASLt）。
// 与 GemmEngine::prewarm_decode_shapes 的形状表一致。
inline int decode_shape_index(uint32_t N, uint32_t K) {
    using namespace mc::cfg;
    const uint32_t qkv_n = kNumQueryHeads * kHeadDim + 2 * kNumKvHeads * kHeadDim;
    if (N == qkv_n && K == kHiddenSize) return 0;              // qkv
    if (N == kHiddenSize && K == kHiddenSize) return 1;        // o
    if (N == 2 * kIntermediateSize && K == kHiddenSize) return 2; // gate_up
    if (N == kHiddenSize && K == kIntermediateSize) return 3;  // down
    if (N == kVocabSize && K == kHiddenSize) return 4;         // lm_head
    return -1;
}

// split-K 候选段数（上限）：N/8 blocks 已 ≥ kGemvSplitTargetBlocks 的 shape
// 不需要 split（候选只有 rows）；小 N shape 从大到小试 {8,4,2}。
inline constexpr uint32_t kGemvSplitMaxS = 8;
inline constexpr uint32_t kGemvSplitTargetBlocks = 168; // ~2 blocks/SM（84 SM）
inline constexpr uint32_t kGemvSplitMaxN = 4096;        // split-K 只考虑小 N shape

// 该 shape 的 split-K 段数（1 = 不 split，用 rows 变体）。
inline uint32_t gemv_pick_split_s(uint32_t N, uint32_t K) {
    if (N > kGemvSplitMaxN) return 1;
    const uint32_t blocks = (N + 7u) / 8u; // 8 warps/block
    if (blocks >= kGemvSplitTargetBlocks) return 1;
    for (uint32_t S = kGemvSplitMaxS; S >= 2u; S >>= 1) {
        if (K % S == 0u && (K / S) % 8u == 0u && blocks * S >= kGemvSplitTargetBlocks)
            return S;
    }
    return 1u;
}

// split-K partial scratch 的 float 数（session 建期预留尺寸的口径）。
inline uint32_t gemv_partial_floats() {
    using namespace mc::cfg;
    const uint32_t qkv_n = kNumQueryHeads * kHeadDim + 2 * kNumKvHeads * kHeadDim;
    uint32_t max_n = qkv_n > kHiddenSize ? qkv_n : kHiddenSize; // split 候选 shape 的最大 N
    return kGemvSplitMaxS * max_n;
}

// 建期快测：对 5 个 shape 逐一计时（cuBLASLt / rows / split-K），把赢家
// 写入 ModelImpl::gemv_tactic[]（进程内首个真实 session 执行，其余复用）。
// 幂等、线程安全（model 级互斥）；失败返回非 MC_OK（调用方决定是否回退
// 全 cuBLASLt——默认 tactic 初值即全 0=cuBLASLt，安全）。
mc_status_t select_decode_gemv_tactics(const ModelImpl& m, SessionImpl& s);

// decode/lm_head 的 M=1 dispatch：按 m.gemv_tactic[decode_shape_index(N,K)]
// 调 launch_decode_gemv（split_partial 为 session 建期预留的固定地址 scratch）。
mc_status_t run_decode_gemv(const ModelImpl& m, float* split_partial, uint32_t N,
                            uint32_t K, const uint16_t* d_w, const uint16_t* d_x,
                            uint16_t* d_y, cudaStream_t stream,
                            bool pdl = false); // P5 PDL

} // namespace mc
