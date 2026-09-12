// swiglu.cu — SiLU(gate) * up elementwise（真实可用，计划 §8.3 优化点 3）。
// BF16 in/out，fp32 中间计算；silu(x) = x / (1 + expf(-x))。
#include "kernels/sm120/kernels.h"

#include <cuda_bf16.h>

#include "runtime/error.h"

namespace {

using bf16 = __nv_bfloat16;

__global__ void swiglu_kernel(const bf16* __restrict__ gate,
                              const bf16* __restrict__ up,
                              bf16* __restrict__ out,
                              int64_t n) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float g = __bfloat162float(gate[i]);
    const float u = __bfloat162float(up[i]);
    const float silu = g / (1.0f + expf(-g));
    out[i] = __float2bfloat16(silu * u);
}

} // namespace

mc_status_t launch_swiglu(const uint16_t* d_gate, const uint16_t* d_up,
                          uint16_t* d_out, int64_t n, cudaStream_t stream) {
    if (d_gate == nullptr || d_up == nullptr || d_out == nullptr || n <= 0) {
        mc::set_error("launch_swiglu: invalid args (gate=%p up=%p out=%p n=%lld)",
                      (const void*)d_gate, (const void*)d_up, (const void*)d_out,
                      (long long)n);
        return MC_E_INVALID_ARGUMENT;
    }
    const int64_t blocks = (n + 255) / 256;
    if (blocks > 2147483647ll) {
        mc::set_error("launch_swiglu: element count %lld too large for grid", (long long)n);
        return MC_E_INVALID_ARGUMENT;
    }
    MC_NVTX_PUSH("swiglu");
    swiglu_kernel<<<dim3((unsigned)blocks), dim3(256), 0, stream>>>(
        reinterpret_cast<const bf16*>(d_gate), reinterpret_cast<const bf16*>(d_up),
        reinterpret_cast<bf16*>(d_out), n);
    MC_NVTX_POP();
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        mc::set_error("swiglu launch failed: %s", cudaGetErrorString(e));
        return MC_E_CUDA;
    }
    return MC_OK;
}

// ---- P2：gate/up 行拼输入（wpk gate_up [*,12288] 行拼序：gate|up）----
namespace {

// 一维元素索引 i = m*inter + d；gate=src[m*2*inter+d]，up=src[m*2*inter+inter+d]
__global__ void swiglu_gated_kernel(const bf16* __restrict__ src,
                                    bf16* __restrict__ out, int64_t inter,
                                    int64_t total) {
    MC_PDL_TRIGGER(); // 入口放行后继（后继消费本 kernel 输出前必 wait）
    MC_PDL_WAIT(); // src 由前驱 gate_up 写出
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const int64_t m = i / inter;
    const int64_t d = i - m * inter;
    const float g = __bfloat162float(src[m * 2 * inter + d]);
    const float u = __bfloat162float(src[m * 2 * inter + inter + d]);
    const float silu = g / (1.0f + expf(-g));
    out[m * inter + d] = __float2bfloat16(silu * u);
}

} // namespace

mc_status_t launch_swiglu_gated(const uint16_t* d_src, uint16_t* d_out, uint32_t rows,
                                int64_t inter, cudaStream_t stream, bool pdl /* = false */) {
    if (d_src == nullptr || d_out == nullptr || rows == 0 || inter <= 0) {
        mc::set_error("launch_swiglu_gated: invalid args (rows=%u inter=%lld)", rows,
                      (long long)inter);
        return MC_E_INVALID_ARGUMENT;
    }
    const int64_t total = (int64_t)rows * inter;
    const int64_t blocks = (total + 255) / 256;
    if (blocks > 2147483647ll) {
        mc::set_error("launch_swiglu_gated: element count %lld too large", (long long)total);
        return MC_E_INVALID_ARGUMENT;
    }
    MC_NVTX_PUSH("swiglu_gated");
    MC_LAUNCH_PDL(swiglu_gated_kernel, dim3((unsigned)blocks), dim3(256), stream,
                  pdl, reinterpret_cast<const bf16*>(d_src),
                  reinterpret_cast<bf16*>(d_out), inter, total);
    MC_NVTX_POP();
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        mc::set_error("swiglu_gated launch failed: %s", cudaGetErrorString(e));
        return MC_E_CUDA;
    }
    return MC_OK;
}
