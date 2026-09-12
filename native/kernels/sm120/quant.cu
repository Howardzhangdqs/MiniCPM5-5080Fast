// quant.cu — F2a：bf16 → e4m3 per-row/per-token 动态量化 + fp8 GEMM 回退模式
// 的 post-scale 补偿 kernel（批解码 F2b / prefill F3 共享机械）。
//
// quant_rows_kernel：bf16 [M,K] → e4m3 [M,K] + scale[M]。
//   每行两遍：pass1 行 amax（分摊装载 + warp/block max 归约）；
//   scale = amax/448（e4m3 satfinite 最大值），全零行 scale=1（避免除 0，
//   量化值恒 0）；pass2 q = x·(448/amax) 经硬件 cvt（RNE + satfinite，
//   __nv_cvt_float_to_fp8——与 solo gemv-fp8 的解码路径互逆，solo 已锚定
//   e4m3 全码空间语义）。|q| ≤ 448 按构造成立（amax 归一），satfinite 仅
//   兜底。M≤8（批解码）与 M=2048（prefill）同一路径：块 ↔ 行 grid-stride。
//
// post_scale_rows_kernel（仅回退模式可达）：cuBLASLt per-tensor scalar 缩放
// 无法表达 per-row/per-token 向量 → GEMM 输出原始 fp8 点积，本 kernel 补
//   y[m,n] = bf16(f32(y[m,n]) · w_scale[n] · x_scale[m])
//（首选 OUTER_VEC_32F 模式下核内已完成，本 kernel 不发射）。代价
// 2·M·N·2B 读写/层（M=8 五 shape 中最大 gate_up ≈ 0.4MB/层，可接受）。
//
// 数值口径：量化误差由调用方锚吸收（W8A8 relL2 ≤1e-2，kernel_ops
// §5.14）；本 kernel 无归约顺序歧义（逐元素）。
#include "kernels/sm120/kernels.h"

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include "runtime/error.h"

namespace {

using bf16 = __nv_bfloat16;

constexpr uint32_t kQuantBlock = 256; // 8 warp（与 resnorm/gemv 族一致）

// bf16 [M,K] → e4m3 [M,K] + scale[M]（per-row amax 动态量化）
__global__ void quant_rows_kernel(const uint16_t* __restrict__ x,  // [M,K]
                                  uint8_t* __restrict__ xq,        // [M,K] e4m3
                                  float* __restrict__ scale,       // [M]
                                  uint32_t M, uint32_t K) {
    const uint32_t t = threadIdx.x;
    __shared__ float wmax[kQuantBlock / 32u]; // 每 warp 的行 amax
    __shared__ float s_row;                   // 行 amax（归约结果）
    for (uint32_t row = blockIdx.x; row < M; row += gridDim.x) {
        const uint16_t* xr = x + (size_t)row * K;
        // pass1：行 amax
        float am = 0.f;
        for (uint32_t k = t; k < K; k += blockDim.x) {
            const float v =
                __bfloat162float(reinterpret_cast<const bf16&>(xr[k]));
            am = fmaxf(am, fabsf(v));
        }
#pragma unroll
        for (int s = 16; s > 0; s >>= 1)
            am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, s));
        if ((t & 31u) == 0u) wmax[t >> 5] = am;
        __syncthreads();
        if (t < kQuantBlock / 32u) { // 8 个 warp max → 首 warp 归约
            float v = wmax[t];
#pragma unroll
            for (int s = 4; s > 0; s >>= 1)
                v = fmaxf(v, __shfl_xor_sync(0xffu, v, s));
            if (t == 0u) s_row = v;
        }
        __syncthreads();
        // scale = amax/448；inv = 448/amax（全零行 inv=0 → 量化值恒 0，
        // scale 记 1 保持反量化恒等 0×1=0）
        const float inv = s_row > 0.f ? 448.f / s_row : 0.f;
        if (t == 0u) scale[row] = s_row > 0.f ? s_row / 448.f : 1.f;
        uint8_t* qr = xq + (size_t)row * K;
        // pass2：RNE + satfinite 编码（硬件 cvt，与 e4m3 解码互逆）
        for (uint32_t k = t; k < K; k += blockDim.x) {
            const float v =
                __bfloat162float(reinterpret_cast<const bf16&>(xr[k])) * inv;
            qr[k] = (uint8_t)__nv_cvt_float_to_fp8(v, __NV_SATFINITE,
                                                   __NV_E4M3);
        }
        __syncthreads(); // smem（wmax/s_row）跨行复用前同步
    }
}

// y[m,n] ×= w_scale[n]·x_scale[m]（x_scale 可空 → 仅 w_scale；元素级，无歧义）
__global__ void post_scale_rows_kernel(uint16_t* __restrict__ y,    // [M,N] bf16
                                       const float* __restrict__ ws, // [N]
                                       const float* __restrict__ xs, // [M] 或 null
                                       uint32_t M, uint32_t N) {
    const size_t tot = (size_t)M * N;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < tot;
         i += (size_t)gridDim.x * blockDim.x) {
        const uint32_t m = (uint32_t)(i / N), n = (uint32_t)(i % N);
        float s = ws[n];
        if (xs != nullptr) s *= xs[m];
        const float v =
            __bfloat162float(reinterpret_cast<const bf16&>(y[i])) * s;
        y[i] = __bfloat16_as_ushort(__float2bfloat16(v));
    }
}

} // namespace

mc_status_t launch_quant_rows(const uint16_t* d_x, uint8_t* d_xq,
                              float* d_scale, uint32_t M, uint32_t K,
                              cudaStream_t stream) {
    if (d_x == nullptr || d_xq == nullptr || d_scale == nullptr || M == 0 ||
        K == 0) {
        mc::set_error("launch_quant_rows: invalid args (M=%u K=%u)", M, K);
        return MC_E_INVALID_ARGUMENT;
    }
    const uint32_t grid = M < 1024u ? M : 1024u; // 块 ↔ 行（grid-stride 兜底）
    quant_rows_kernel<<<grid, dim3(kQuantBlock), 0, stream>>>(d_x, d_xq, d_scale,
                                                              M, K);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        mc::set_error("quant_rows launch failed: %s", cudaGetErrorString(e));
        return MC_E_CUDA;
    }
    return MC_OK;
}

mc_status_t launch_post_scale_rows(uint16_t* d_y, const float* d_w_scale,
                                   const float* d_x_scale, uint32_t M,
                                   uint32_t N, cudaStream_t stream) {
    if (d_y == nullptr || d_w_scale == nullptr || M == 0 || N == 0) {
        mc::set_error("launch_post_scale_rows: invalid args (M=%u N=%u)", M, N);
        return MC_E_INVALID_ARGUMENT;
    }
    const size_t tot = (size_t)M * N;
    const uint32_t grid =
        (uint32_t)((tot / kQuantBlock + 3u) / 4u + 1u); // ~4 元素/线程，上限内
    post_scale_rows_kernel<<<grid < 4096u ? grid : 4096u, dim3(kQuantBlock), 0,
                             stream>>>(d_y, d_w_scale, d_x_scale, M, N);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        mc::set_error("post_scale_rows launch failed: %s",
                      cudaGetErrorString(e));
        return MC_E_CUDA;
    }
    return MC_OK;
}
