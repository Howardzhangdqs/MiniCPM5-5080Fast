// gemv.cu — P3+：decode 专用 M=1 BF16 GEMV 自研 kernel（计划 §8.2「固定
// shape 下性能不佳的小 GEMM/GEMV」与 §10.2 microbenchmark 的落地）。
//
// 语义与 GemmEngine::gemm_bf16 的 M=1 完全一致：y[n] = Σ_k x[k]·W[n,k]，
// W 行主 [N,K]（输出维在前），fp32 累加、单次 RNE 舍入 bf16（数值规范同
// cuBLASLt 路径：bf16 in → fp32 accumulate → bf16 out）。
//
// 变体 A gemv_rows_kernel（每 warp 一行，全 K）：
//   grid.x = ceil(N/8)，block = 256 threads（8 warp）。warp ↔ 输出行 n；
//   lane 沿 K 以 uint4（8×bf16 = 16B）跨步（每轮 warp 覆盖 32×8=256 元素），
//   主体 4 路展开（v, v+32, v+64, v+96 同时在飞 → 每 lane 4×16B 装载 ILP，
//   延迟隐藏的关键），fp32 fma 累加 → warp butterfly 归约 → lane 0 写 bf16。
//   W 行连续 512B/warp·轮 → 完美合并的纯流式读（带宽型 kernel）。
//   5 个 decode shape 的 N/8 = 320/256/1536/256/16320 blocks，均 ≥ 3
//   blocks/SM（84 SM），occupancy 充分。
//
// 变体 B gemv_rows_splitk_kernel + merge（split-K 两段式，确定性）：
//   阶段 1 grid=(ceil(N/8), S)：每 block 处理 8 行 × K 的第 blockIdx.y 段，
//   partial (fp32) 写 scratch [S, N]；阶段 2 merge：每 thread 一行，按
//   s=0..S-1 固定顺序求和 → bf16。**无原子操作**（fp32 原子加的顺序不
//   确定性会破坏 graph-vs-eager 的 bit 一致性与确定性 decode）。
//   用途：小 N shape 增加并行 block 数（tactic 建期快测定胜负）。
//
// CUDA Graph 兼容：kernel/grid/地址均为建期常量（tactic 在 session_create
// 期决定，见 runtime/gemv_tactic.cpp）；无 host 决策、无分配进 replay。
#include "kernels/sm120/kernels.h"

#include <cuda_bf16.h>

#include "runtime/error.h"

namespace {

using bf16 = __nv_bfloat16;

constexpr uint32_t kWarpsPerBlock = 8;
constexpr uint32_t kBlockThreads = kWarpsPerBlock * 32u; // 256

// uint4（8×bf16）对的 fp32 点积：8 次 fma。
__device__ __forceinline__ float dot8(uint4 w4, uint4 x4) {
    const __nv_bfloat162* wb = reinterpret_cast<const __nv_bfloat162*>(&w4);
    const __nv_bfloat162* xb = reinterpret_cast<const __nv_bfloat162*>(&x4);
    float s = 0.f;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const float2 wf = __bfloat1622float2(wb[j]);
        const float2 xf = __bfloat1622float2(xb[j]);
        s = fmaf(wf.x, xf.x, s);
        s = fmaf(wf.y, xf.y, s);
    }
    return s;
}

// ---- 变体 A：每 warp 一行，全 K ----
__global__ void gemv_rows_kernel(const uint16_t* __restrict__ W, // [N,K]
                                  const uint16_t* __restrict__ x, // [K]
                                  uint16_t* __restrict__ y,       // [N]
                                  uint32_t N, uint32_t K) {
    const uint32_t warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const uint32_t lane = threadIdx.x & 31u;
    const uint4* wv = reinterpret_cast<const uint4*>(W + (size_t)warp * K);
    const uint4* xv = reinterpret_cast<const uint4*>(x);
    const uint32_t nvec = K >> 3; // uint4 总数（launcher 保证 K % 8 == 0）

    // P5 PDL：W 与前驱输出无依赖 —— 前 2 轮 8 路 uint4 预取先于 wait 发射，
    // 与前驱 kernel 的执行/drain 重叠（纯装载调度，运算序不变）。
    uint4 wp[8];
    const bool pre = (warp < N) && (lane + 96u) < nvec;
    const bool pre2 = pre && (lane + 224u) < nvec;
    if (pre) {
#pragma unroll
        for (int u = 0; u < 4; ++u) wp[u] = wv[lane + 32u * u];
        if (pre2) {
#pragma unroll
            for (int u = 4; u < 8; ++u) wp[u] = wv[lane + 32u * u];
        }
    }
    MC_PDL_TRIGGER(); // 入口放行后继（后继消费本 kernel 输出前必 wait）
    MC_PDL_WAIT(); // x 由前驱写出（attention ctx 等）
    if (warp >= N) return;

    float acc = 0.f;
    uint32_t v = lane;
    if (pre) { // 首轮消费预取
        const uint4 x0 = xv[v], x1 = xv[v + 32u], x2 = xv[v + 64u], x3 = xv[v + 96u];
        acc += dot8(wp[0], x0) + dot8(wp[1], x1) + dot8(wp[2], x2) + dot8(wp[3], x3);
        v += 128u;
        if (pre2) {
            const uint4 x4 = xv[v], x5 = xv[v + 32u], x6 = xv[v + 64u], x7 = xv[v + 96u];
            acc += dot8(wp[4], x4) + dot8(wp[5], x5) + dot8(wp[6], x6) + dot8(wp[7], x7);
            v += 128u;
        }
    }
    // 主体：4 路展开（v, v+32, v+64, v+96 同时在飞）。
    for (; v + 96u < nvec; v += 128u) {
        const uint4 w0 = wv[v], w1 = wv[v + 32u], w2 = wv[v + 64u], w3 = wv[v + 96u];
        const uint4 x0 = xv[v], x1 = xv[v + 32u], x2 = xv[v + 64u], x3 = xv[v + 96u];
        acc += dot8(w0, x0) + dot8(w1, x1) + dot8(w2, x2) + dot8(w3, x3);
    }
    // 余量（防御：nvec 非 128 对齐的 K；每 uint4 恰被一个 lane 处理）
    for (; v < nvec; v += 32u) acc += dot8(wv[v], xv[v]);

#pragma unroll
    for (int s = 16; s > 0; s >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, s);
    if (lane == 0)
        y[warp] = __bfloat16_as_ushort(__float2bfloat16(acc));
}

// ---- 变体 B 阶段 1：每 warp 一行 × K 的第 blockIdx.y 段 ----
__global__ void gemv_rows_splitk_kernel(const uint16_t* __restrict__ W, // [N,K]
                                        const uint16_t* __restrict__ x, // [K]
                                        float* __restrict__ partial,    // [S,N]
                                        uint32_t N, uint32_t K,
                                        uint32_t kseg) { // K/S（launcher 保证 %8==0）
    const uint32_t warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    if (warp >= N) return;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t seg = blockIdx.y;
    const uint32_t vec0 = seg * (kseg >> 3);  // 本段起始 uint4（段内编号）
    const uint32_t nseg = kseg >> 3;          // 本段 uint4 数
    const uint4* wv = reinterpret_cast<const uint4*>(W + (size_t)warp * K);
    const uint4* xv = reinterpret_cast<const uint4*>(x);

    float acc = 0.f;
    for (uint32_t v = lane; v < nseg; v += 32u) acc += dot8(wv[vec0 + v], xv[vec0 + v]);
#pragma unroll
    for (int s = 16; s > 0; s >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, s);
    if (lane == 0) partial[(size_t)seg * N + warp] = acc;
}

// ---- 变体 B 阶段 2：按 s 升序固定顺序合并（确定性，无原子）----
__global__ void gemv_splitk_merge_kernel(const float* __restrict__ partial, // [S,N]
                                         uint32_t S, uint32_t N,
                                         uint16_t* __restrict__ y) {
    const uint32_t n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    float acc = 0.f;
    for (uint32_t s = 0; s < S; ++s) acc += partial[(size_t)s * N + n];
    y[n] = __bfloat16_as_ushort(__float2bfloat16(acc));
}

inline bool aligned16(const void* p) { return (reinterpret_cast<uintptr_t>(p) & 15u) == 0u; }

// ---- P5：融合 prologue 变体（decode 专用，BF16 权重路径）----
// 结构与数值约束同 gemv_fp8_prologue_kernel（见 gemv_fp8.cu 注释）：
// 块协作 prologue 逐 bit 复刻 residual_add_rmsnorm_kernel 的运算序；
// normed 落 shared 供 GEMV 读；residual ping-pong 写回（block 0）。
__global__ void gemv_rows_prologue_kernel(
    const uint16_t* __restrict__ W, const uint16_t* __restrict__ res_in,
    const uint16_t* __restrict__ addend, const uint16_t* __restrict__ norm_w,
    uint16_t* __restrict__ res_out, uint16_t* __restrict__ y, uint32_t N,
    uint32_t n, float eps) {
    __shared__ __align__(16) uint16_t sx[2048];
    __shared__ float sred[256];
    const int t = (int)threadIdx.x;

    // ---- W 首轮预取（同 gemv_fp8_prologue_kernel）；PDL：res/addend 由前驱
    //      写出 —— 预取之后 wait ----
    const uint32_t warp_g = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const uint32_t lane_g = threadIdx.x & 31u;
    const uint4* wvg = reinterpret_cast<const uint4*>(W + (size_t)warp_g * n);
    const uint32_t nvec_g = n >> 3;
    uint4 wpre[8];
    const bool pre = (warp_g < N) && (lane_g + 96u) < nvec_g;
    const bool pre2 = pre && (lane_g + 224u) < nvec_g;
    if (pre) {
#pragma unroll
        for (int u = 0; u < 4; ++u) wpre[u] = wvg[lane_g + 32u * u];
        if (pre2) {
#pragma unroll
            for (int u = 4; u < 8; ++u) wpre[u] = wvg[lane_g + 32u * u];
        }
    }
    MC_PDL_TRIGGER(); // 入口放行后继（后继消费本 kernel 输出前必 wait）
    MC_PDL_WAIT();

    uint16_t v8[8];
#pragma unroll
    for (int u = 0; u < 8; ++u) {
        const int i = t + 256 * u;
        if (addend != nullptr) {
            const float sum =
                __bfloat162float(reinterpret_cast<const bf16&>(res_in[i])) +
                __bfloat162float(reinterpret_cast<const bf16&>(addend[i]));
            v8[u] = __bfloat16_as_ushort(__float2bfloat16(sum));
        } else {
            v8[u] = res_in[i];
        }
        if (blockIdx.x == 0) res_out[i] = v8[u];
    }
    float acc2 = 0.f;
#pragma unroll
    for (int u = 0; u < 8; ++u) {
        const float v = __bfloat162float(reinterpret_cast<const bf16&>(v8[u]));
        acc2 += v * v;
    }
    sred[t] = acc2;
    __syncthreads();
#pragma unroll
    for (unsigned stride = 128u; stride > 0; stride >>= 1) {
        if ((unsigned)t < stride) sred[t] += sred[t + stride];
        __syncthreads();
    }
    const float inv = rsqrtf(sred[0] / (float)n + eps);
#pragma unroll
    for (int u = 0; u < 8; ++u) {
        const int i = t + 256 * u;
        sx[i] = __bfloat16_as_ushort(__bfloat162float(
                                         reinterpret_cast<const bf16&>(v8[u])) *
                                     inv *
                                     __bfloat162float(
                                         reinterpret_cast<const bf16&>(norm_w[i])));
    }
    __syncthreads();

    // ---- GEMV 主体（同 gemv_rows_kernel，x 从 shared 读）----
    const uint32_t warp = warp_g;
    if (warp >= N) return;
    const uint32_t lane = lane_g;
    const uint4* wv = wvg;
    const uint4* xv = reinterpret_cast<const uint4*>(sx);
    const uint32_t nvec = nvec_g;

    float acc = 0.f;
    uint32_t v = lane;
    if (pre) { // 首轮：消费预取的 W
        const uint4 x0 = xv[v], x1 = xv[v + 32u], x2 = xv[v + 64u], x3 = xv[v + 96u];
        acc += dot8(wpre[0], x0) + dot8(wpre[1], x1) + dot8(wpre[2], x2) + dot8(wpre[3], x3);
        v += 128u;
        if (pre2) {
            const uint4 x4 = xv[v], x5 = xv[v + 32u], x6 = xv[v + 64u], x7 = xv[v + 96u];
            acc += dot8(wpre[4], x4) + dot8(wpre[5], x5) + dot8(wpre[6], x6) + dot8(wpre[7], x7);
            v += 128u;
        }
    }
    for (; v + 96u < nvec; v += 128u) {
        const uint4 w0 = wv[v], w1 = wv[v + 32u], w2 = wv[v + 64u], w3 = wv[v + 96u];
        const uint4 x0 = xv[v], x1 = xv[v + 32u], x2 = xv[v + 64u], x3 = xv[v + 96u];
        acc += dot8(w0, x0) + dot8(w1, x1) + dot8(w2, x2) + dot8(w3, x3);
    }
    for (; v < nvec; v += 32u) acc += dot8(wv[v], xv[v]);

#pragma unroll
    for (int s = 16; s > 0; s >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, s);
    if (lane == 0) y[warp] = __bfloat16_as_ushort(__float2bfloat16(acc));
}


// ---- P5：融合 swiglu 变体（down 投影专用，BF16 权重路径；K=6144）----
// 结构/数值同 gemv_fp8_swiglu_kernel（见 gemv_fp8.cu 注释）。
constexpr uint32_t kSwigluK = 6144;

__device__ __forceinline__ __nv_bfloat162 swiglu_pair_bf162(__nv_bfloat162 g2,
                                                            __nv_bfloat162 u2) {
    const float2 gf = __bfloat1622float2(g2);
    const float2 uf = __bfloat1622float2(u2);
    const float s0 = gf.x / (1.0f + expf(-gf.x));
    const float s1 = gf.y / (1.0f + expf(-gf.y));
    return __floats2bfloat162_rn(s0 * uf.x, s1 * uf.y);
}

__global__ void gemv_rows_swiglu_kernel(const uint16_t* __restrict__ W,
                                        const uint16_t* __restrict__ gate_up,
                                        uint16_t* __restrict__ y, uint32_t N) {
    __shared__ __align__(16) uint16_t sx[kSwigluK];
    const uint32_t warp_g = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const uint32_t lane_g = threadIdx.x & 31u;
    const uint4* wvg = reinterpret_cast<const uint4*>(W + (size_t)warp_g * kSwigluK);
    constexpr uint32_t nvec_g = kSwigluK >> 3;
    uint4 wpre[8];
    const bool pre = (warp_g < N) && (lane_g + 96u) < nvec_g;
    const bool pre2 = pre && (lane_g + 224u) < nvec_g;
    if (pre) {
#pragma unroll
        for (int u = 0; u < 4; ++u) wpre[u] = wvg[lane_g + 32u * u];
        if (pre2) {
#pragma unroll
            for (int u = 4; u < 8; ++u) wpre[u] = wvg[lane_g + 32u * u];
        }
    }
    MC_PDL_TRIGGER(); // 入口放行后继（后继消费本 kernel 输出前必 wait）
    MC_PDL_WAIT(); // gate_up 输出由前驱写出

    {
        const uint4* gv = reinterpret_cast<const uint4*>(gate_up);
        const uint4* uv = reinterpret_cast<const uint4*>(gate_up + kSwigluK);
        const uint32_t nv = kSwigluK >> 3;
        for (uint32_t i = threadIdx.x; i < nv; i += blockDim.x) {
            const uint4 g4 = gv[i], u4 = uv[i];
            const __nv_bfloat162* g2 = reinterpret_cast<const __nv_bfloat162*>(&g4);
            const __nv_bfloat162* u2 = reinterpret_cast<const __nv_bfloat162*>(&u4);
            uint4 o4;
            __nv_bfloat162* o2 = reinterpret_cast<__nv_bfloat162*>(&o4);
#pragma unroll
            for (int p = 0; p < 4; ++p) o2[p] = swiglu_pair_bf162(g2[p], u2[p]);
            reinterpret_cast<uint4*>(sx)[i] = o4;
        }
    }
    __syncthreads();

    const uint32_t warp = warp_g;
    if (warp >= N) return;
    const uint32_t lane = lane_g;
    const uint4* wv = wvg;
    const uint4* xv = reinterpret_cast<const uint4*>(sx);
    const uint32_t nvec = nvec_g;

    float acc = 0.f;
    uint32_t v = lane;
    if (pre) {
        const uint4 x0 = xv[v], x1 = xv[v + 32u], x2 = xv[v + 64u], x3 = xv[v + 96u];
        acc += dot8(wpre[0], x0) + dot8(wpre[1], x1) + dot8(wpre[2], x2) + dot8(wpre[3], x3);
        v += 128u;
        if (pre2) {
            const uint4 x4 = xv[v], x5 = xv[v + 32u], x6 = xv[v + 64u], x7 = xv[v + 96u];
            acc += dot8(wpre[4], x4) + dot8(wpre[5], x5) + dot8(wpre[6], x6) + dot8(wpre[7], x7);
            v += 128u;
        }
    }
    for (; v + 96u < nvec; v += 128u) {
        const uint4 w0 = wv[v], w1 = wv[v + 32u], w2 = wv[v + 64u], w3 = wv[v + 96u];
        const uint4 x0 = xv[v], x1 = xv[v + 32u], x2 = xv[v + 64u], x3 = xv[v + 96u];
        acc += dot8(w0, x0) + dot8(w1, x1) + dot8(w2, x2) + dot8(w3, x3);
    }
    for (; v < nvec; v += 32u) acc += dot8(wv[v], xv[v]);

#pragma unroll
    for (int s = 16; s > 0; s >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, s);
    if (lane == 0) y[warp] = __bfloat16_as_ushort(__float2bfloat16(acc));
}

} // namespace

mc_status_t launch_gemv_rows_prologue(const uint16_t* d_w,
                                      const uint16_t* d_res_in,
                                      const uint16_t* d_addend,
                                      const uint16_t* d_norm_w,
                                      uint16_t* d_res_out, uint16_t* d_y,
                                      uint32_t N, uint32_t n, float eps,
                                      cudaStream_t stream, bool pdl /* = false */) {
    if (d_w == nullptr || d_res_in == nullptr ||
        d_norm_w == nullptr || d_res_out == nullptr || d_y == nullptr || N == 0 ||
        n != 2048u || !aligned16(d_w)) {
        mc::set_error("launch_gemv_rows_prologue: invalid args (N=%u n=%u)", N, n);
        return MC_E_INVALID_ARGUMENT;
    }
    const dim3 grid((N + kWarpsPerBlock - 1u) / kWarpsPerBlock);
    MC_NVTX_PUSH("gemv_rows_prologue");
    MC_LAUNCH_PDL(gemv_rows_prologue_kernel, grid, dim3(kBlockThreads), stream, pdl, 
        d_w, d_res_in, d_addend, d_norm_w, d_res_out, d_y, N, n, eps);
    MC_NVTX_POP();
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        mc::set_error("gemv_rows_prologue launch failed: %s", cudaGetErrorString(e));
        return MC_E_CUDA;
    }
    return MC_OK;
}


mc_status_t launch_gemv_rows_swiglu(const uint16_t* d_w, const uint16_t* d_gate_up,
                                    uint16_t* d_y, uint32_t N, uint32_t K,
                                    cudaStream_t stream, bool pdl /* = false */) {
    if (d_w == nullptr || d_gate_up == nullptr || d_y == nullptr || N == 0 ||
        K != 6144u || !aligned16(d_w)) {
        mc::set_error("launch_gemv_rows_swiglu: invalid args (N=%u K=%u)", N, K);
        return MC_E_INVALID_ARGUMENT;
    }
    const dim3 grid((N + kWarpsPerBlock - 1u) / kWarpsPerBlock);
    MC_NVTX_PUSH("gemv_rows_swiglu");
    MC_LAUNCH_PDL(gemv_rows_swiglu_kernel, grid, dim3(kBlockThreads), stream, pdl, 
        d_w, d_gate_up, d_y, N);
    MC_NVTX_POP();
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        mc::set_error("gemv_rows_swiglu launch failed: %s", cudaGetErrorString(e));
        return MC_E_CUDA;
    }
    return MC_OK;
}

mc_status_t launch_decode_gemv(const uint16_t* d_w, const uint16_t* d_x,
                               uint16_t* d_y, uint32_t N, uint32_t K, int tactic,
                               uint32_t split_s, float* d_split_partial,
                               cudaStream_t stream, bool pdl /* = false */) {
    if (d_w == nullptr || d_x == nullptr || d_y == nullptr || N == 0 ||
        (K & 7u) != 0 || !aligned16(d_w) || !aligned16(d_x)) {
        mc::set_error("launch_decode_gemv: invalid args (N=%u K=%u tactic=%d)",
                      N, K, tactic);
        return MC_E_INVALID_ARGUMENT;
    }
    if (tactic == mc::kGemvTacticSplitK &&
        (d_split_partial == nullptr || split_s < 2u || (K / split_s) % 8u != 0u ||
         K % split_s != 0u)) {
        mc::set_error("launch_decode_gemv: bad split-K config (S=%u K=%u partial=%p)",
                      split_s, K, (const void*)d_split_partial);
        return MC_E_INVALID_ARGUMENT;
    }

    if (tactic == mc::kGemvTacticRows) {
        MC_NVTX_PUSH("gemv_rows");
        MC_LAUNCH_PDL(gemv_rows_kernel, dim3((N + kWarpsPerBlock - 1u) / kWarpsPerBlock), dim3(kBlockThreads), stream, pdl, d_w, d_x, d_y, N, K);
        MC_NVTX_POP();
    } else if (tactic == mc::kGemvTacticSplitK) {
        const uint32_t kseg = K / split_s;
        MC_NVTX_PUSH("gemv_splitk");
        gemv_rows_splitk_kernel<<<dim3((N + kWarpsPerBlock - 1u) / kWarpsPerBlock,
                                       split_s),
                                  dim3(kBlockThreads), 0, stream>>>(
            d_w, d_x, d_split_partial, N, K, kseg);
        MC_NVTX_POP();
        cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess) {
            mc::set_error("gemv splitk launch failed: %s", cudaGetErrorString(e));
            return MC_E_CUDA;
        }
        MC_NVTX_PUSH("gemv_splitk_merge");
        gemv_splitk_merge_kernel<<<dim3((N + 511u) / 512u), dim3(512), 0, stream>>>(
            d_split_partial, split_s, N, d_y);
        MC_NVTX_POP();
    } else {
        mc::set_error("launch_decode_gemv: unknown tactic %d", tactic);
        return MC_E_INVALID_ARGUMENT;
    }
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        mc::set_error("gemv launch failed: %s", cudaGetErrorString(e));
        return MC_E_CUDA;
    }
    return MC_OK;
}
