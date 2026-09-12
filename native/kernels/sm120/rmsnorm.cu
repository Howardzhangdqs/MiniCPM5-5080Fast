// rmsnorm.cu — RMSNorm（真实可用）：fp32 累加，eps 参数，两档简单实现。
//   档 1（n <= 1024）：单 block，线程一一对应 + block reduce
//   档 2（n 任意）  ：单 block 512 线程循环两遍（sumsq → normalize）
// 同一 block 内归约，无需跨块；MiniCPM5-2B hidden=2048 走档 2。
#include "kernels/sm120/kernels.h"

#include <cuda_bf16.h>

#include "runtime/error.h"

namespace {

using bf16 = __nv_bfloat16;

__global__ void rmsnorm_small_kernel(const bf16* __restrict__ x,
                                     const bf16* __restrict__ w,
                                     bf16* __restrict__ out,
                                     int32_t n,
                                     float eps) {
    extern __shared__ float s[]; // blockDim.x floats
    const int t = threadIdx.x;
    float v = 0.f;
    if (t < n) v = __bfloat162float(x[t]);
    s[t] = v * v;
    __syncthreads();
    for (unsigned stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if ((unsigned)t < stride) s[t] += s[t + stride];
        __syncthreads();
    }
    const float inv = rsqrtf(s[0] / (float)n + eps);
    if (t < n) out[t] = __float2bfloat16(v * inv * __bfloat162float(w[t]));
}

__global__ void rmsnorm_large_kernel(const bf16* __restrict__ x,
                                     const bf16* __restrict__ w,
                                     bf16* __restrict__ out,
                                     int32_t n,
                                     float eps) {
    extern __shared__ float s[]; // blockDim.x floats
    const int t = threadIdx.x;
    float acc = 0.f;
    for (int i = t; i < n; i += blockDim.x) {
        const float v = __bfloat162float(x[i]);
        acc += v * v;
    }
    s[t] = acc;
    __syncthreads();
    for (unsigned stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if ((unsigned)t < stride) s[t] += s[t + stride];
        __syncthreads();
    }
    const float inv = rsqrtf(s[0] / (float)n + eps);
    for (int i = t; i < n; i += blockDim.x) {
        out[i] = __float2bfloat16(__bfloat162float(x[i]) * inv * __bfloat162float(w[i]));
    }
}

} // namespace

mc_status_t launch_rmsnorm(const uint16_t* d_x, const uint16_t* d_weight,
                           uint16_t* d_out, int32_t n, float eps, cudaStream_t stream) {
    if (d_x == nullptr || d_weight == nullptr || d_out == nullptr || n <= 0) {
        mc::set_error("launch_rmsnorm: invalid args (x=%p w=%p out=%p n=%d)",
                      (const void*)d_x, (const void*)d_weight, (const void*)d_out, n);
        return MC_E_INVALID_ARGUMENT;
    }
    const bf16* x = reinterpret_cast<const bf16*>(d_x);
    const bf16* w = reinterpret_cast<const bf16*>(d_weight);
    bf16* out = reinterpret_cast<bf16*>(d_out);

    MC_NVTX_PUSH("rmsnorm");
    if (n <= 1024) {
        int threads = 32;
        while (threads < n) threads <<= 1; // block 归约要求 2 的幂
        rmsnorm_small_kernel<<<dim3(1), dim3(threads), threads * sizeof(float), stream>>>(
            x, w, out, n, eps);
    } else {
        constexpr int kThreads = 512;
        rmsnorm_large_kernel<<<dim3(1), dim3(kThreads), kThreads * sizeof(float), stream>>>(
            x, w, out, n, eps);
    }
    MC_NVTX_POP();
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        mc::set_error("rmsnorm launch failed: %s", cudaGetErrorString(e));
        return MC_E_CUDA;
    }
    return MC_OK;
}

// ==================== P2 追加：行批量 + 融合残差 ====================

namespace {

// 单行 RMSNorm 的行批量版本（每 block 一行，256 线程循环 2048 元素）。
// 语义与上面单行版一致：fp32 连乘 x*inv*w 后单次 RNE 舍入 bf16
// （与 python 参考 tools/reference_forward.py rmsnorm_bf16 一致；
// 无中间 bf16 舍入）。
__global__ void rmsnorm_rows_kernel(const bf16* __restrict__ x,
                                     const bf16* __restrict__ w,
                                     bf16* __restrict__ out, int n, float eps) {
    extern __shared__ float s[]; // blockDim.x floats
    const uint32_t row = blockIdx.x;
    const int t = threadIdx.x;
    const bf16* xr = x + (uint64_t)row * n;
    bf16* orow = out + (uint64_t)row * n;

    float acc = 0.f;
    for (int i = t; i < n; i += blockDim.x) {
        const float v = __bfloat162float(xr[i]);
        acc += v * v;
    }
    s[t] = acc;
    __syncthreads();
    for (unsigned stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if ((unsigned)t < stride) s[t] += s[t + stride];
        __syncthreads();
    }
    const float inv = rsqrtf(s[0] / (float)n + eps);
    for (int i = t; i < n; i += blockDim.x) {
        orow[i] = __float2bfloat16(__bfloat162float(xr[i]) * inv *
                                   __bfloat162float(w[i]));
    }
}

// 融合：res = bf16(f32(res)+f32(x))（原地）+ RMSNorm(res, w) → out。
// pass1 算和 + 落 bf16 残差（原地写，与参考 add_bf16 的存储边界一致），
// pass2 读舍入后的残差做归一化 —— norm 作用的是「舍入后的残差」；
// 归一化输出为 fp32 连乘 x*inv*w 单次 RNE 舍入（同上，无中间舍入）。
__global__ void residual_add_rmsnorm_kernel(bf16* __restrict__ res,
                                             const bf16* __restrict__ x,
                                             const bf16* __restrict__ w,
                                             bf16* __restrict__ out, int n,
                                             float eps) {
    MC_PDL_TRIGGER(); // 入口放行后继（后继消费本 kernel 输出前必 wait）
    MC_PDL_WAIT(); // x（down_out 等）由前驱写出
    extern __shared__ float s[]; // blockDim.x floats
    const uint32_t row = blockIdx.x;
    const int t = threadIdx.x;
    bf16* rrow = res + (uint64_t)row * n;
    const bf16* xrow = x + (uint64_t)row * n;
    bf16* orow = out + (uint64_t)row * n;

    for (int i = t; i < n; i += blockDim.x) {
        const float sum = __bfloat162float(rrow[i]) + __bfloat162float(xrow[i]);
        rrow[i] = __float2bfloat16(sum); // 残差落 bf16（与参考存储边界一致）
    }
    __syncthreads(); // 残差写回全局内存后统一重读（跨线程可见）
    // norm 作用于「舍入后的残差」（python 参考：x = add_bf16(...) 后再 norm）
    float acc2 = 0.f;
    for (int i = t; i < n; i += blockDim.x) {
        const float v = __bfloat162float(rrow[i]);
        acc2 += v * v;
    }
    s[t] = acc2;
    __syncthreads();
    for (unsigned stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if ((unsigned)t < stride) s[t] += s[t + stride];
        __syncthreads();
    }
    const float inv = rsqrtf(s[0] / (float)n + eps);
    for (int i = t; i < n; i += blockDim.x) {
        orow[i] = __float2bfloat16(__bfloat162float(rrow[i]) * inv *
                                   __bfloat162float(w[i]));
    }
}

} // namespace

mc_status_t launch_rmsnorm_rows(const uint16_t* d_x, const uint16_t* d_weight,
                                uint16_t* d_out, uint32_t rows, int32_t n, float eps,
                                cudaStream_t stream) {
    if (d_x == nullptr || d_weight == nullptr || d_out == nullptr || rows == 0 ||
        n <= 0) {
        mc::set_error("launch_rmsnorm_rows: invalid args (rows=%u n=%d)", rows, n);
        return MC_E_INVALID_ARGUMENT;
    }
    constexpr int kThreads = 256;
    MC_NVTX_PUSH("rmsnorm_rows");
    rmsnorm_rows_kernel<<<dim3(rows), dim3(kThreads), kThreads * sizeof(float), stream>>>(
        reinterpret_cast<const bf16*>(d_x), reinterpret_cast<const bf16*>(d_weight),
        reinterpret_cast<bf16*>(d_out), n, eps);
    MC_NVTX_POP();
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        mc::set_error("rmsnorm_rows launch failed: %s", cudaGetErrorString(e));
        return MC_E_CUDA;
    }
    return MC_OK;
}

mc_status_t launch_residual_add_rmsnorm(uint16_t* d_res, const uint16_t* d_x,
                                        const uint16_t* d_w, uint16_t* d_out,
                                        uint32_t rows, int32_t n, float eps,
                                        cudaStream_t stream, bool pdl /* = false */) {
    if (d_res == nullptr || d_x == nullptr || d_w == nullptr || d_out == nullptr ||
        rows == 0 || n <= 0) {
        mc::set_error("launch_residual_add_rmsnorm: invalid args (rows=%u n=%d)", rows,
                      n);
        return MC_E_INVALID_ARGUMENT;
    }
    constexpr int kThreads = 256;
    MC_NVTX_PUSH("residual_add_rmsnorm");
    {
        cudaLaunchAttribute _at_[1];
        _at_[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        _at_[0].val.programmaticStreamSerializationAllowed = 1;
        cudaLaunchConfig_t _cfg_;
        _cfg_.gridDim = dim3(rows);
        _cfg_.blockDim = dim3(kThreads);
        _cfg_.dynamicSmemBytes = kThreads * sizeof(float);
        _cfg_.stream = stream;
        _cfg_.attrs = pdl ? _at_ : nullptr;
        _cfg_.numAttrs = pdl ? 1 : 0;
        cudaLaunchKernelEx(&_cfg_, residual_add_rmsnorm_kernel,
                           reinterpret_cast<bf16*>(d_res),
                           reinterpret_cast<const bf16*>(d_x),
                           reinterpret_cast<const bf16*>(d_w),
                           reinterpret_cast<bf16*>(d_out), n, eps);
    }
    MC_NVTX_POP();
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        mc::set_error("residual_add_rmsnorm launch failed: %s", cudaGetErrorString(e));
        return MC_E_CUDA;
    }
    return MC_OK;
}
