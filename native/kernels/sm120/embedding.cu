// embedding.cu — embedding gather（真实可用，计划 §8.2 第一版：Custom Gather）。
// out[t*hidden + d] = table[id[t]*hidden + d]
#include "kernels/sm120/kernels.h"

#include <cuda_bf16.h>

#include "runtime/error.h"

namespace {

using bf16 = __nv_bfloat16;

__global__ void embedding_gather_kernel(const bf16* __restrict__ table,
                                        const int32_t* __restrict__ ids,
                                        bf16* __restrict__ out,
                                        int32_t hidden,
                                        int32_t n_tokens) {
    const int t = blockIdx.x;
    if (t >= n_tokens) return;
    const int64_t id = ids[t]; // 调用方保证 id ∈ [0, vocab)
    const bf16* __restrict__ row = table + id * (int64_t)hidden;
    bf16* __restrict__ dst = out + (int64_t)t * hidden;
    for (int d = threadIdx.x; d < hidden; d += blockDim.x) {
        dst[d] = row[d];
    }
}

// B1b：批间接 gather —— token 不在 host，也不在连续 device 数组里，而是
// 每 slot 各自常驻的成员 d_next_token（device 指针表寻址）。每 slot 一
// block；token 由全 block 广播读（同一地址，只发一条 loads）。
// out 即组激活 residual 的第 slot 行 [slot][hidden]。
__global__ void embedding_gather_indirect_kernel(
    const bf16* __restrict__ table,
    const int32_t* const* __restrict__ d_slot_tok, // [n]：每 slot 的 token 地址
    bf16* __restrict__ out,                        // [n][hidden]
    int32_t hidden, int32_t n) {
    const int s = blockIdx.x;
    if (s >= n) return;
    const int64_t id = *d_slot_tok[s]; // 调用方保证 ∈ [0, vocab)
    const bf16* __restrict__ row = table + id * (int64_t)hidden;
    bf16* __restrict__ dst = out + (int64_t)s * hidden;
    for (int d = threadIdx.x; d < hidden; d += blockDim.x) {
        dst[d] = row[d];
    }
}

} // namespace

mc_status_t launch_embedding_gather(const uint16_t* d_table,
                                    const int32_t* d_ids,
                                    uint16_t* d_out,
                                    int32_t hidden,
                                    int32_t n_tokens,
                                    cudaStream_t stream) {
    if (d_table == nullptr || d_ids == nullptr || d_out == nullptr ||
        hidden <= 0 || n_tokens <= 0) {
        mc::set_error("launch_embedding_gather: invalid args (table=%p ids=%p out=%p "
                      "hidden=%d n_tokens=%d)",
                      (const void*)d_table, (const void*)d_ids, (const void*)d_out,
                      hidden, n_tokens);
        return MC_E_INVALID_ARGUMENT;
    }
    MC_NVTX_PUSH("embedding_gather");
    embedding_gather_kernel<<<dim3(n_tokens), dim3(256), 0, stream>>>(
        reinterpret_cast<const bf16*>(d_table), d_ids,
        reinterpret_cast<bf16*>(d_out), hidden, n_tokens);
    MC_NVTX_POP();
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        mc::set_error("embedding_gather launch failed: %s", cudaGetErrorString(e));
        return MC_E_CUDA;
    }
    return MC_OK;
}

// B1b：批间接 embedding gather（组 decode 步首算子）。ids 来自每 slot 的
// 成员 d_next_token（device 指针表 [n]，表地址组内固定）；out 写组集中
// residual 行 [n][hidden]。
mc_status_t launch_embedding_gather_indirect(const uint16_t* d_table,
                                             const int32_t* const* d_slot_tok,
                                             uint16_t* d_out, int32_t hidden,
                                             int32_t n, cudaStream_t stream) {
    if (d_table == nullptr || d_slot_tok == nullptr || d_out == nullptr ||
        hidden <= 0 || n <= 0) {
        mc::set_error("launch_embedding_gather_indirect: invalid args (table=%p "
                      "slot_tok=%p out=%p hidden=%d n=%d)",
                      (const void*)d_table, (const void*)d_slot_tok,
                      (const void*)d_out, hidden, n);
        return MC_E_INVALID_ARGUMENT;
    }
    MC_NVTX_PUSH("embedding_gather_indirect");
    embedding_gather_indirect_kernel<<<dim3(n), dim3(256), 0, stream>>>(
        reinterpret_cast<const bf16*>(d_table), d_slot_tok,
        reinterpret_cast<bf16*>(d_out), hidden, n);
    MC_NVTX_POP();
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        mc::set_error("embedding_gather_indirect launch failed: %s",
                      cudaGetErrorString(e));
        return MC_E_CUDA;
    }
    return MC_OK;
}
