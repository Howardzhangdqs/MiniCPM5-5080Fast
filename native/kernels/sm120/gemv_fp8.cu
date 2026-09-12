// gemv_fp8.cu — P4：decode 专用 FP8(E4M3) GEMV kernel（计划 §13/§20-P4）。
//
// 语义：y[n] = scale_row[n] × Σ_k e4m3(w[n,k]) · f32(x[k])。
//   权重 uint8 [N,K] 行主（wpk dtype=2），逐输出行 fp32 scale [N]；
//   x 为 bf16 activation 行；fp32 累加（与 bf16 GEMV 同数值规范），
//   归约完成后乘入行 scale（每行常数 → 数学等价，数值良好），单次 RNE
//   舍入 bf16。
//
// 结构（沿用 gemv_rows 的带宽型形态，计划 §8.2）：
//   grid.x = ceil(N/8)，block 256（8 warp），warp ↔ 输出行；
//   lane 沿 K 以 uint4（16×fp8 = 16B）跨步，每轮 warp 覆盖 32×16=512 元素；
//   主体 2 路展开（v, v+32 同时在飞 → 每 lane 2×16B fp8 + 4×16B x 装载
//   ILP）；K%16 的余量元素走逐元素尾路径（lane j 处理第 j 个尾元素，
//   逐 warp ≤15 个）；warp butterfly 归约后 lane 0 写 bf16。
//
// P5 v2 —— 硬件转换指令替代 shared LUT 解码：v1 每 fp8 元素 1 次 LDS
//（随机 256 项索引 → bank 冲突串行化，实测仅 630-834 GB/s，低于 bf16 字节
// 带宽）。v2 用 cuda_fp8.h 的 __nv_cvt_fp8x2_to_halfraw2（sm_89+ 硬件
// cvt；sm_120 必在）：e4m3→half 精确（e4m3 全部正规/次正规值在 half 的
// 正规范围内），half→f32 精确 → 与 LUT/位转换路径逐 bit 一致（乘加对与
// fma 链的序完全保持，kernel_ops 已知答案锚原样通过）。
//   例外：e4m3 的 S.1111.111 编码（satfinite 量化器不产生，但 kernel_ops
// 全码空间锚会命中）硬件转出 NaN —— uint4 级预检（含 0x7F/0xFF 字节即
// 走慢路径，mc::fp8_e4m3_to_f32 位转换 = ±480）保证语义完整；真实量化
// 权重恒走快路径。
//
// 对齐要求：W 行基址 16B（K%16==0 时行偏移自动满足；launcher 校验
// 基址与 K%8；K 非 16 倍数时行内 uint4 对齐仍成立——行起点按字节计
// 错位 <16 时逐元素尾路径吸收）。本项目 4 个 decode shape 的 K
//（2048/6144）均 %16==0，恒走向量主体。
//
// CUDA Graph 兼容：grid/参数/地址建期固定；无分配、无 host 决策。
#include "kernels/sm120/kernels.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include "runtime/error.h"

namespace {

using bf16 = __nv_bfloat16;

constexpr uint32_t kWarpsPerBlock = 8;
constexpr uint32_t kBlockThreads = kWarpsPerBlock * 32u; // 256

// e4m3×2 → half2（硬件 cvt）→ float2。两条转换均精确（见文件头注释）。
__device__ __forceinline__ float2 fp8x2_to_f2(__nv_fp8x2_storage_t v) {
    const __half2_raw hr = __nv_cvt_fp8x2_to_halfraw2(v, __NV_E4M3);
    return __half22float2(__half2(hr));
}

// 32 位字中是否存在 e4m3 NaN 编码字节（0x7F/0xFF）：
// 剥符号后 +1 进位到 bit7 ⟺ byte == 0x7F（其余 ≤0x7E 无进位）。
__device__ __forceinline__ bool has_nan_code(uint32_t w) {
    return (((w & 0x7F7F7F7Fu) + 0x01010101u) & 0x80808080u) != 0u;
}

// e4m3 的 S.1111.111 编码（satfinite 量化器不产生；kernel_ops 全码空间
// 锚会命中）硬件转出 NaN —— 且 PTX cvt 输出正规范 NaN（符号丢失，
// 实测 half=0x7FFF），符号须从原始字节恢复：值 = ±480（与
// mc::fp8_e4m3_to_f32 位运算语义逐 bit 一致）。
__device__ __forceinline__ float2 fp8x2_to_f2_fixed(__nv_fp8x2_storage_t v) {
    float2 f = fp8x2_to_f2(v);
    const uint32_t b0 = (uint32_t)v & 0xFFu;
    const uint32_t b1 = (uint32_t)v >> 8;
    if ((b0 & 0x7Fu) == 0x7Fu)
        f.x = (b0 & 0x80u) ? -480.f : 480.f;
    if ((b1 & 0x7Fu) == 0x7Fu)
        f.y = (b1 & 0x80u) ? -480.f : 480.f;
    return f;
}

// 16×fp8（uint4 w）× 16×bf16（2×uint4 x）的 fp32 点积（v2 硬件转换）。
// 快路径（真实量化权重恒走）：8× cvt(fp8x2→h2) + 16× fmaf，两条独立
// 累加链（s0/s1）；慢路径（uint4 含 NaN 编码字节：随机数据 ~12%、
// satfinite 量化权重 0%）同硬件转换 + 逐对 ±480 修补。两条路径的乘加对
// 与 fma 链序完全一致（e4m3→half→f32 与 mc::fp8_e4m3_to_f32 位运算对非
// NaN 编码逐 bit 相等）——kernel_ops 全码空间锚 bit-close。
// 分支结构：整 uint4 一次判定（快路径无逐对 predicate/branch 开销，
// ptxas 生成两份干净循环体；慢路径罕见，分叉成本可忽略）。
__device__ __forceinline__ void dot16_fp8_cvt(uint4 w4, uint4 xa, uint4 xb,
                                               float& s0, float& s1) {
    const __nv_fp8x2_storage_t* wp = reinterpret_cast<const __nv_fp8x2_storage_t*>(&w4);
    const __nv_bfloat162* x2a = reinterpret_cast<const __nv_bfloat162*>(&xa);
    const __nv_bfloat162* x2b = reinterpret_cast<const __nv_bfloat162*>(&xb);
    if (has_nan_code(w4.x) || has_nan_code(w4.y) || has_nan_code(w4.z) ||
        has_nan_code(w4.w)) {
#pragma unroll
        for (int p = 0; p < 4; ++p) {
            const float2 wf = fp8x2_to_f2_fixed(wp[p]);
            const float2 xf = __bfloat1622float2(x2a[p]);
            s0 = fmaf(wf.x, xf.x, s0);
            s1 = fmaf(wf.y, xf.y, s1);
        }
#pragma unroll
        for (int p = 0; p < 4; ++p) {
            const float2 wf = fp8x2_to_f2_fixed(wp[4 + p]);
            const float2 xf = __bfloat1622float2(x2b[p]);
            s0 = fmaf(wf.x, xf.x, s0);
            s1 = fmaf(wf.y, xf.y, s1);
        }
        return;
    }
#pragma unroll
    for (int p = 0; p < 4; ++p) {
        const float2 wf = fp8x2_to_f2(wp[p]);
        const float2 xf = __bfloat1622float2(x2a[p]);
        s0 = fmaf(wf.x, xf.x, s0);
        s1 = fmaf(wf.y, xf.y, s1);
    }
#pragma unroll
    for (int p = 0; p < 4; ++p) {
        const float2 wf = fp8x2_to_f2(wp[4 + p]);
        const float2 xf = __bfloat1622float2(x2b[p]);
        s0 = fmaf(wf.x, xf.x, s0);
        s1 = fmaf(wf.y, xf.y, s1);
    }
}

__global__ void gemv_fp8_rows_kernel(const uint8_t* __restrict__ W,  // [N,K]
                                      const float* __restrict__ scale, // [N]
                                      const uint16_t* __restrict__ x,  // [K]
                                      uint16_t* __restrict__ y,        // [N]
                                      uint32_t N, uint32_t K) {
    const uint32_t warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const uint32_t lane = threadIdx.x & 31u;
    const uint8_t* row = W + (size_t)warp * K;
    const uint4* wv = reinterpret_cast<const uint4*>(row);
    const uint4* xv = reinterpret_cast<const uint4*>(x);
    const uint32_t nvec = K >> 4; // uint4(fp8) 总数（每单元 16 元素）

    // P5 PDL：W 预取（前驱无关）→ wait → 消费 x（前驱输出）
    uint4 wpre0, wpre1;
    const bool pre = (warp < N) && (lane + 32u) < nvec;
    if (pre) {
        wpre0 = wv[lane];
        wpre1 = wv[lane + 32u];
    }
    MC_PDL_TRIGGER(); // 入口放行后继（后继消费本 kernel 输出前必 wait）
    MC_PDL_WAIT();
    if (warp >= N) return;

    // 4 条独立累加链（w0 的 s0/s1 + w1 的 s0/s1）
    float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
    uint32_t v = lane;
    if (pre) {
        const uint4 x0a = xv[2 * v], x0b = xv[2 * v + 1u];
        const uint4 x1a = xv[2 * v + 64u], x1b = xv[2 * v + 65u];
        dot16_fp8_cvt(wpre0, x0a, x0b, a0, a1);
        dot16_fp8_cvt(wpre1, x1a, x1b, a2, a3);
        v += 64u;
    }
    // 主体：2 路展开（v, v+32 同时在飞；x 需 2×uint4/每 w-uint4）
    for (; v + 32u < nvec; v += 64u) {
        const uint4 w0 = wv[v], w1 = wv[v + 32u];
        const uint4 x0a = xv[2 * v], x0b = xv[2 * v + 1u];
        const uint4 x1a = xv[2 * v + 64u], x1b = xv[2 * v + 65u];
        dot16_fp8_cvt(w0, x0a, x0b, a0, a1);
        dot16_fp8_cvt(w1, x1a, x1b, a2, a3);
    }
    // 余量（每 uint4 恰被一个 lane 处理；nvec 非 64 对齐时逐单元收尾）
    for (; v < nvec; v += 32u)
        dot16_fp8_cvt(wv[v], xv[2 * v], xv[2 * v + 1u], a0, a1);
    float acc = (a0 + a1) + (a2 + a3);

#pragma unroll
    for (int s = 16; s > 0; s >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, s);
    if (lane == 0)
        y[warp] = __bfloat16_as_ushort(
            __float2bfloat16(acc * scale[warp]));
}

// 标量回退（任意 K；行基址 16B 对齐不成立时的正确性路径——K%16≠0 的行
// 偏移会错位 uint4 寻址。真实 decode shape 的 K 均 %16==0，此路径仅供
// 通用性/测试，逐字节装载）。
__global__ void gemv_fp8_rows_scalar_kernel(const uint8_t* __restrict__ W,
                                            const float* __restrict__ scale,
                                            const uint16_t* __restrict__ x,
                                            uint16_t* __restrict__ y, uint32_t N,
                                            uint32_t K) {
    const uint32_t warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    if (warp >= N) return;
    const uint32_t lane = threadIdx.x & 31u;
    const uint8_t* row = W + (size_t)warp * K;
    float acc = 0.f;
    for (uint32_t k = lane; k < K; k += 32u) {
        const float w = mc::fp8_e4m3_to_f32(row[k]);
        const float xv = __bfloat162float(reinterpret_cast<const bf16&>(x[k]));
        acc = fmaf(w, xv, acc);
    }
#pragma unroll
    for (int s = 16; s > 0; s >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, s);
    if (lane == 0)
        y[warp] = __bfloat16_as_ushort(__float2bfloat16(acc * scale[warp]));
}

inline bool aligned16(const void* p) { return (reinterpret_cast<uintptr_t>(p) & 15u) == 0u; }

// ---- P5：融合 prologue 变体（decode 专用）----
// residual_add_rmsnorm + GEMV 的单 kernel 融合：消除 84 次/step 的独立
// resnorm kernel（实测 6.04µs/次 —— 单 block 小 kernel 是纯延迟）。
// 数值：块协作 prologue 逐 bit 复刻 residual_add_rmsnorm_kernel 的
// 运算序（256 线程、i = t, t+256, … 分组、shared 树归约、
// inv = rsqrtf(s[0]/n + eps)、normed = bf16(f32·inv·f32(w)) 单次 RNE），
// normed 落 shared（不写回全局）→ GEMV 主循环从 shared 读 x ——
// 与「独立 kernel 写全局 normed + GEMV 读回」逐 bit 一致。
// residual 流：ping-pong 双缓冲（res_in → res_out 不同缓冲，块间无读写
// 竞争；block 0 负责写回，全块算得的值相同 → 字节确定）。
// 限制：K 必须 == n（prologue 输出即 x，宽度一致）；decode 的 qkv/gate_up
// K=hidden=2048 恒成立（launcher 校验）。CUDA Graph 兼容（参数建期固定）。
__global__ void gemv_fp8_prologue_kernel(
    const uint8_t* __restrict__ W, const float* __restrict__ scale,
    const uint16_t* __restrict__ res_in, const uint16_t* __restrict__ addend,
    const uint16_t* __restrict__ norm_w, uint16_t* __restrict__ res_out,
    uint16_t* __restrict__ y, uint32_t N, uint32_t n, float eps) {
    __shared__ __align__(16) uint16_t sx[2048]; // normed x（prologue 输出）
    __shared__ float sred[256];
    const int t = (int)threadIdx.x;

    // ---- W 首轮预取：在 prologue 的装载/归约延迟下提前发射 GEMV 的
    //      首两个 W uint4（与 norm 计算无依赖，寄存器驻留跨过 barrier）----
    const uint32_t warp_g = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const uint32_t lane_g = threadIdx.x & 31u;
    const uint4* wvg =
        reinterpret_cast<const uint4*>(W + (size_t)warp_g * n);
    const uint32_t nvec_g = n >> 4;
    uint4 wpre0, wpre1;
    const bool pre = (warp_g < N) && (lane_g + 32u) < nvec_g;
    if (pre) {
        wpre0 = wvg[lane_g];
        wpre1 = wvg[lane_g + 32u];
    }
    MC_PDL_TRIGGER(); // 入口放行后继（后继消费本 kernel 输出前必 wait）
    MC_PDL_WAIT(); // res/addend 由前驱写出（PDL 启动时生效）

    // pass1：v = bf16(f32(res[i]) + f32(add[i]))，i = t, t+256, …
    //（addend == nullptr → 纯 norm 路径：v = res[i]（bf16 恒等），与
    //  rmsnorm_rows_kernel 的数值序逐 bit 一致）
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
        if (blockIdx.x == 0) res_out[i] = v8[u]; // 写回（单 block，值确定）
    }
    // pass2：sumsq（序同 residual_add_rmsnorm_kernel：每线程顺序累加 + 树归约）
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

    // pass3：normed 落 shared（bf16 单次 RNE，乘序 (f32·inv)·f32(w) 同参考）
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

    // ---- GEMV 主体（同 gemv_fp8_rows_kernel，x 从 shared 读）----
    const uint32_t warp = warp_g;
    if (warp >= N) return;
    const uint32_t lane = lane_g;
    const uint4* wv = wvg;
    const uint4* xv = reinterpret_cast<const uint4*>(sx);
    const uint32_t nvec = nvec_g;

    float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
    uint32_t v = lane;
    if (pre) { // 首轮：消费预取的 W（x 此时已在 shared）
        const uint4 x0a = xv[2 * v], x0b = xv[2 * v + 1u];
        const uint4 x1a = xv[2 * v + 64u], x1b = xv[2 * v + 65u];
        dot16_fp8_cvt(wpre0, x0a, x0b, a0, a1);
        dot16_fp8_cvt(wpre1, x1a, x1b, a2, a3);
        v += 64u;
    }
    for (; v + 32u < nvec; v += 64u) {
        const uint4 w0 = wv[v], w1 = wv[v + 32u];
        const uint4 x0a = xv[2 * v], x0b = xv[2 * v + 1u];
        const uint4 x1a = xv[2 * v + 64u], x1b = xv[2 * v + 65u];
        dot16_fp8_cvt(w0, x0a, x0b, a0, a1);
        dot16_fp8_cvt(w1, x1a, x1b, a2, a3);
    }
    for (; v < nvec; v += 32u)
        dot16_fp8_cvt(wv[v], xv[2 * v], xv[2 * v + 1u], a0, a1);
    float acc = (a0 + a1) + (a2 + a3);

#pragma unroll
    for (int s = 16; s > 0; s >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, s);
    if (lane == 0)
        y[warp] = __bfloat16_as_ushort(__float2bfloat16(acc * scale[warp]));
}


// ---- P5：融合 swiglu 变体（down 投影专用，K=inter=6144）----
// 独立 swiglu kernel（1.5µs）+ down GEMV 间的 launch 间隙（~1.3µs）×42 层
// ≈ 118µs/step。融合：块协作把 gate/up 行经 silu 乘积写入 shared（数值与
// swiglu_gated_kernel 逐 bit 一致：g,u = f32(bf16)、silu = g/(1+expf(-g))、
// 单次 RNE 舍入 bf16），GEMV 从 shared 读 x。K 必须 == 6144（launcher 校验）。
// CUDA Graph 兼容（参数/网格建期固定；无分配）。
constexpr uint32_t kSwigluK = 6144;

__device__ __forceinline__ __nv_bfloat162 swiglu_pair_bf162(__nv_bfloat162 g2,
                                                            __nv_bfloat162 u2) {
    const float2 gf = __bfloat1622float2(g2);
    const float2 uf = __bfloat1622float2(u2);
    const float s0 = gf.x / (1.0f + expf(-gf.x));
    const float s1 = gf.y / (1.0f + expf(-gf.y));
    return __floats2bfloat162_rn(s0 * uf.x, s1 * uf.y);
}

__global__ void gemv_fp8_swiglu_kernel(const uint8_t* __restrict__ W,
                                       const float* __restrict__ scale,
                                       const uint16_t* __restrict__ gate_up,
                                       uint16_t* __restrict__ y, uint32_t N) {
    __shared__ __align__(16) uint16_t sx[kSwigluK]; // mlp_mid（silu 乘积）
    // W 首轮预取（与 prologue 变体同式）
    const uint32_t warp_g = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const uint32_t lane_g = threadIdx.x & 31u;
    const uint4* wvg = reinterpret_cast<const uint4*>(W + (size_t)warp_g * kSwigluK);
    constexpr uint32_t nvec_g = kSwigluK >> 4;
    uint4 wpre0, wpre1;
    const bool pre = (warp_g < N) && (lane_g + 32u) < nvec_g;
    if (pre) {
        wpre0 = wvg[lane_g];
        wpre1 = wvg[lane_g + 32u];
    }
    MC_PDL_TRIGGER(); // 入口放行后继（后继消费本 kernel 输出前必 wait）
    MC_PDL_WAIT(); // gate_up 输出由前驱写出

    // 块协作 swiglu：bf162 对处理（i = t*8 起 8 元素连续 → uint4 装载）
    {
        const uint4* gv = reinterpret_cast<const uint4*>(gate_up);
        const uint4* uv = reinterpret_cast<const uint4*>(gate_up + kSwigluK);
        const uint32_t nv = kSwigluK >> 3; // 768 个 uint4
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

    float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
    uint32_t v = lane;
    if (pre) {
        const uint4 x0a = xv[2 * v], x0b = xv[2 * v + 1u];
        const uint4 x1a = xv[2 * v + 64u], x1b = xv[2 * v + 65u];
        dot16_fp8_cvt(wpre0, x0a, x0b, a0, a1);
        dot16_fp8_cvt(wpre1, x1a, x1b, a2, a3);
        v += 64u;
    }
    for (; v + 32u < nvec; v += 64u) {
        const uint4 w0 = wv[v], w1 = wv[v + 32u];
        const uint4 x0a = xv[2 * v], x0b = xv[2 * v + 1u];
        const uint4 x1a = xv[2 * v + 64u], x1b = xv[2 * v + 65u];
        dot16_fp8_cvt(w0, x0a, x0b, a0, a1);
        dot16_fp8_cvt(w1, x1a, x1b, a2, a3);
    }
    for (; v < nvec; v += 32u)
        dot16_fp8_cvt(wv[v], xv[2 * v], xv[2 * v + 1u], a0, a1);
    float acc = (a0 + a1) + (a2 + a3);

#pragma unroll
    for (int s = 16; s > 0; s >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, s);
    if (lane == 0)
        y[warp] = __bfloat16_as_ushort(__float2bfloat16(acc * scale[warp]));
}

// ---- F1 批组 fp8：多行 gemv（M∈[2,8]，批解码合并前向的 kernel 基石）----
// 语义：y[m,n] = scale[n] × Σ_k e4m3(w[n,k]) · f32(x[m,k])；x bf16 [M,K]、
// y bf16 [M,N]。结构沿用 M=1 骨架（warp ↔ 输出行 n、lane 沿 K 以
// uint4=16×fp8 步进、butterfly 归约、scale 归约后乘入、单次 RNE bf16）。
//
// 逐行 bit == solo（锚 1 的构造依据）：
//   本 kernel 的迭代结构与 solo 的 2 路展开主循环同构——每次迭代处理
//   两个 uint4 单元：va = 64·t + lane（→ 链 (a0,a1)）与 vb = va + 32
//   （→ 链 (a2,a3)）。链归属与 solo 严格一致：
//     solo a0/a1 ← v ≡ lane (mod 64)（pre 首项 + main 各轮 + 至多一次
//              tail——solo 的 tail v 恒 ≡ lane mod 64）
//     solo a2/a3 ← v ≡ lane+32 (mod 64)
//   每条链内 fmaf 按 v 升序逐对发射，与 solo 相同；dot16_fp8_m 内 16 对
//   fmaf 与 dot16_fp8_cvt（:83-120）的对序逐对相同（w 解码一次（8×cvt）
//   后按行复用——fmaf 逐次 IEEE 精确，行间交错不改变各链内的舍入序）
//   → 每行输出与 M=1 kernel 逐 bit 一致。
//
// X 驻留（实测迭代后的定稿）：X[M][K] bf16（M=8、K=6144 时 96KB）超 L1
//（128KB 级，且与流式 W 竞争）→ K-tile shared 双缓冲 + XOR swizzle：
//   * tile = TK=1024 元素（= 每迭代 64 单元 = 恰好 2 单元/lane），
//     双缓冲 2×M×TK×2B（M=8 → 32KB，3 block/SM）；
//   * 16B chunk 列 swizzle c → c^(c>>3)：lane 的两条 LDS.128（chunk
//     2u / 2u+1）在 8-lane wavefront 内均落 8 个不同 bank-quad——直存
//     布局下 lane 间 32B 步进会 2-way bank 冲突（实测 LDS 吞吐减半是
//     首版不达带宽的主因之一）；swizzle 为双射，装载侧同样无冲突；
//   * W 直读全局流式（每字节恒用一次，不经 shared）；X 跨块经 L2 复用
//     （总量 ≤96KB ≪ L2 64MB）。
//   性能要点（消融实测）：barrier/装载仅 ~2µs/shape；瓶颈在消费循环的
//   指令压力（bf16→f32 转换与寻址 INT 指令随 M 放大 M 倍）→ 2 单元/
//   lane 迭代把循环体摊到 2 个 W 单元上（链下标静态化亦消除了运行时
//   acc[m][b] 动态索引——否则 acc 会被打入 local memory，ptxas 实测
//   128B 栈帧 + LDL/STL 往返）。
//
// PDL：W 首 2 个 uint4 预取于 TRIGGER/WAIT 之前（前驱无关，:135-142 同
// 式）——恰为迭代 0 的 va/vb；X 的 global 装载全在 WAIT 之后（前驱
// 输出）。launch 参数建期固定（CUDA Graph 兼容，F2 批组直接捕图）。
//
// 模板按 M∈[2,8] 全特化（寄存器压力优先：acc[M][4] 精确 M 份；M=8 →
// 32 累加寄存器 + 16 解码寄存器 + ~12 杂项 ≈ 60-90，0 spill 预算）。
constexpr uint32_t kGemvFp8MTileK = 1024; // K-tile（元素）= 每迭代 2 单元/lane

// 16B chunk 列 swizzle：c → c^(c>>3)。TK=1024 → 128 chunk/行，映射为
// [0,128) 上的双射（y_i = c_i ^ c_{i+3} 可逆）；消费侧两条 LDS.128 的
// 地址组在 8-lane wavefront 内各落 8 个不同 bank-quad（无冲突）。
__device__ __forceinline__ uint32_t gemv_f8m_swz(uint32_t c) {
    return c ^ (c >> 3);
}

// 多行版 dot16：w（16×fp8，uint4）解码一次（8×cvt；含 NaN 编码时整块
// 走 ±480 修补慢路径，判定同 dot16_fp8_cvt），对 M 行各发 16 个 fmaf。
// 每行的 16 对 fmaf 与 solo dot16_fp8_cvt 逐对相同：前 4 对来自 xa（单元
// 元素 0..7 的偶/奇位），后 4 对来自 xb（元素 8..15）；链归属 s0/s1 不变
// → 每行各链的 fmaf 序与 solo 完全一致（fmaf 逐次精确，行间交错不改变
// 链内舍入）——锚 1 逐行 bit == solo 的构造依据。
// 链下标 B 为编译期常量（0→(a0,a1)=v≡lane(mod 64)；2→(a2,a3)=v≡lane+32）：
// 运行时下标会把 acc[m][B] 打入 local memory（ptxas 实测 128B 栈帧 +
// LDL/STL 往返）——故解码与按行 fma 拆成两段，B 走模板参数，全静态索引。
__device__ __forceinline__ void fp8x16_decode(uint4 w4, float2 (&wf)[8]) {
    const __nv_fp8x2_storage_t* wp =
        reinterpret_cast<const __nv_fp8x2_storage_t*>(&w4);
    if (has_nan_code(w4.x) || has_nan_code(w4.y) || has_nan_code(w4.z) ||
        has_nan_code(w4.w)) {
#pragma unroll
        for (int p = 0; p < 8; ++p) wf[p] = fp8x2_to_f2_fixed(wp[p]);
    } else {
#pragma unroll
        for (int p = 0; p < 8; ++p) wf[p] = fp8x2_to_f2(wp[p]);
    }
}

// xr 指向本 lane 本单元在 tile 内的 16 元素窗口（行 0 的 swizzle 列），
// xs1 为第二 chunk 的 swizzle 列地址（swizzle 后两 chunk 不连续），
// rstride 为 tile 行距（元素）。B 见上。
template <uint32_t M, uint32_t B>
__device__ __forceinline__ void dot16_fp8_m(const float2 (&wf)[8],
                                            const uint16_t* __restrict__ xr,
                                            const uint16_t* __restrict__ xs1,
                                            uint32_t rstride,
                                            float (&acc)[M][4]) {
#pragma unroll
    for (uint32_t m = 0; m < M; ++m) {
        const uint4* xa = reinterpret_cast<const uint4*>(xr + (size_t)m * rstride);
        const uint4* xb = reinterpret_cast<const uint4*>(xs1 + (size_t)m * rstride);
        const __nv_bfloat162* x2a = reinterpret_cast<const __nv_bfloat162*>(xa);
        const __nv_bfloat162* x2b = reinterpret_cast<const __nv_bfloat162*>(xb);
        float& s0 = acc[m][B];
        float& s1 = acc[m][B + 1u];
#pragma unroll
        for (int p = 0; p < 4; ++p) {
            const float2 xf = __bfloat1622float2(x2a[p]);
            s0 = fmaf(wf[p].x, xf.x, s0);
            s1 = fmaf(wf[p].y, xf.y, s1);
        }
#pragma unroll
        for (int p = 0; p < 4; ++p) {
            const float2 xf = __bfloat1622float2(x2b[p]);
            s0 = fmaf(wf[4 + p].x, xf.x, s0);
            s1 = fmaf(wf[4 + p].y, xf.y, s1);
        }
    }
}

// X tile 块协作装载：tile t 覆盖 k ∈ [t·TK, min((t+1)·TK, K))，bf16 原样
//（位不换）写入 buf 号缓冲的 swizzle 列。每 tile 共 M×TK/8 个 uint4
//（8×bf16）块，256 线程分摊；界外整块补零（K%16==0 → uint4 块要么整块
// 在界内要么整块出界；补零值不会被消费——越界 unit 在消费侧被 v<nvec
// 跳过，仅保证确定性）。
template <uint32_t M, uint32_t TK>
__device__ __forceinline__ void gemv_fp8_m_load_tile(
    uint16_t (*sx)[M][TK], const uint16_t* __restrict__ x, uint32_t t,
    uint32_t buf, uint32_t K) {
    const uint32_t nchunk = TK >> 3; // 每 tile 行的 uint4 数（TK=1024 → 128）
    const uint32_t k0 = t * TK;
    for (uint32_t i = threadIdx.x; i < M * nchunk; i += blockDim.x) {
        const uint32_t m = i / nchunk, c = i % nchunk;
        const uint32_t k = k0 + 8u * c;
        uint4 v4 = make_uint4(0u, 0u, 0u, 0u);
        if (k + 8u <= K)
            v4 = *reinterpret_cast<const uint4*>(x + (size_t)m * K + k);
        *reinterpret_cast<uint4*>(&sx[buf][m][8u * gemv_f8m_swz(c)]) = v4;
    }
}

// 多行主体（向量路径；K%16==0 且 16B 对齐，launcher 校验）。
// grid.x = ceil(N/8)、block 256（8 warp）、warp ↔ 输出行；每次迭代每
// lane 处理 2 个 uint4 单元（va/vb，恰一个 tile）。越界 warp（grid 上
// 取整）不提前返回——须参与块协作装载与 __syncthreads（trip 数全块
// 一致，无分支死锁），仅跳过计算/写回。
template <uint32_t M>
__global__ void gemv_fp8_rows_m_kernel(const uint8_t* __restrict__ W,  // [N,K]
                                       const float* __restrict__ scale, // [N]
                                       const uint16_t* __restrict__ x,  // [M,K]
                                       uint16_t* __restrict__ y,        // [M,N]
                                       uint32_t N, uint32_t K) {
    constexpr uint32_t TK = kGemvFp8MTileK;
    constexpr uint32_t UPT = TK / 16u / 32u; // 每迭代每 lane 单元数 = 2
    __shared__ __align__(16) uint16_t sx[2][M][TK];
    const uint32_t warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t nvec = K >> 4;                  // uint4(fp8) 单元总数
    const uint32_t ntile = (nvec + (32u * UPT) - 1u) / (32u * UPT);
    const uint4* wv = reinterpret_cast<const uint4*>(W + (size_t)warp * K);

    // P5 PDL：W 预取（前驱无关；恰为迭代 0 的 va/vb 两单元）→ wait →
    // 消费 x（前驱输出；装载在循环内，恒位于 WAIT 之后）
    uint4 wpre0, wpre1;
    const bool pre = (warp < N) && (lane + 32u) < nvec;
    if (pre) {
        wpre0 = wv[lane];
        wpre1 = wv[lane + 32u];
    }
    MC_PDL_TRIGGER(); // 入口放行后继（后继消费本 kernel 输出前必 wait）
    MC_PDL_WAIT();

    const bool active = warp < N; // 越界 warp 仍参与装载/sync（见上）
    float acc[M][4];
#pragma unroll
    for (uint32_t m = 0; m < M; ++m)
        acc[m][0] = acc[m][1] = acc[m][2] = acc[m][3] = 0.f;

    // 双缓冲流水：载 tile t+1（buf cur^1）与消费 tile t（buf cur）并行；
    // 轮末 __syncthreads 同时保证「本轮消费完」（下轮可覆写 cur）与
    // 「下轮 tile 装载完」。首 tile 先载入 buf 0。
    gemv_fp8_m_load_tile<M, TK>(sx, x, 0u, 0u, K);
    __syncthreads();
    for (uint32_t t = 0; t < ntile; ++t) {
        const uint32_t cur = t & 1u;
        if (t + 1u < ntile) gemv_fp8_m_load_tile<M, TK>(sx, x, t + 1u, cur ^ 1u, K);
        if (active) {
            const uint32_t va = 32u * UPT * t + lane; // tile 内单元 0..31
            const uint32_t vb = va + 32u;             // tile 内单元 32..63
            // 本单元的 2 个 16B chunk 的 swizzle 列（tile 内）
            const uint32_t ca = 2u * lane, cb = 2u * (lane + 32u);
            const uint16_t* xr0 = &sx[cur][0][8u * gemv_f8m_swz(ca)];
            const uint16_t* xs0 = &sx[cur][0][8u * gemv_f8m_swz(ca + 1u)];
            const uint16_t* xr1 = &sx[cur][0][8u * gemv_f8m_swz(cb)];
            const uint16_t* xs1 = &sx[cur][0][8u * gemv_f8m_swz(cb + 1u)];
            if (va < nvec) {
                float2 wf[8];
                fp8x16_decode((t == 0u && pre) ? wpre0 : wv[va], wf);
                dot16_fp8_m<M, 0u>(wf, xr0, xs0, TK, acc); // va ≡ lane → (a0,a1)
            }
            if (vb < nvec) {
                float2 wf[8];
                fp8x16_decode((t == 0u && pre) ? wpre1 : wv[vb], wf);
                dot16_fp8_m<M, 2u>(wf, xr1, xs1, TK, acc); // vb ≡ lane+32 → (a2,a3)
            }
        }
        __syncthreads();
    }

    if (!active) return; // warp 一致，butterfly 全 warp 参与
#pragma unroll
    for (uint32_t m = 0; m < M; ++m) {
        // 同 solo :166-172：acc = (a0+a1)+(a2+a3) → butterfly → ×scale → RNE
        float a = (acc[m][0] + acc[m][1]) + (acc[m][2] + acc[m][3]);
#pragma unroll
        for (int s = 16; s > 0; s >>= 1) a += __shfl_xor_sync(0xffffffffu, a, s);
        if (lane == 0)
            y[(size_t)m * N + warp] =
                __bfloat16_as_ushort(__float2bfloat16(a * scale[warp]));
    }
}

// 多行标量回退（任意 K；K%16≠0 或行基址失齐时的正确性路径，语义同 solo
// 的 gemv_fp8_rows_scalar_kernel）：每行独立链，k 升序 fmaf（w 每元素解码
// 一次、M 行复用）→ 每行 butterfly → 写 y[m*N+n]。与 solo 标量路径逐行
// bit 一致（同链序同操作数）。
template <uint32_t M>
__global__ void gemv_fp8_rows_m_scalar_kernel(
    const uint8_t* __restrict__ W, const float* __restrict__ scale,
    const uint16_t* __restrict__ x, uint16_t* __restrict__ y, uint32_t N,
    uint32_t K) {
    const uint32_t warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    if (warp >= N) return;
    const uint32_t lane = threadIdx.x & 31u;
    const uint8_t* row = W + (size_t)warp * K;
    float acc[M];
#pragma unroll
    for (uint32_t m = 0; m < M; ++m) acc[m] = 0.f;
    for (uint32_t k = lane; k < K; k += 32u) {
        const float w = mc::fp8_e4m3_to_f32(row[k]);
#pragma unroll
        for (uint32_t m = 0; m < M; ++m) {
            const float xv = __bfloat162float(
                reinterpret_cast<const bf16&>(x[(size_t)m * K + k]));
            acc[m] = fmaf(w, xv, acc[m]);
        }
    }
#pragma unroll
    for (uint32_t m = 0; m < M; ++m) {
        float a = acc[m];
#pragma unroll
        for (int s = 16; s > 0; s >>= 1) a += __shfl_xor_sync(0xffffffffu, a, s);
        if (lane == 0)
            y[(size_t)m * N + warp] =
                __bfloat16_as_ushort(__float2bfloat16(a * scale[warp]));
    }
}

} // namespace

mc_status_t launch_gemv_fp8_swiglu(const uint8_t* d_w, const float* d_scale,
                                   const uint16_t* d_gate_up, uint16_t* d_y,
                                   uint32_t N, uint32_t K, cudaStream_t stream, bool pdl /* = false */) {
    if (d_w == nullptr || d_scale == nullptr || d_gate_up == nullptr ||
        d_y == nullptr || N == 0 || K != kSwigluK || !aligned16(d_w)) {
        mc::set_error("launch_gemv_fp8_swiglu: invalid args (N=%u K=%u)", N, K);
        return MC_E_INVALID_ARGUMENT;
    }
    const dim3 grid((N + kWarpsPerBlock - 1u) / kWarpsPerBlock);
    MC_NVTX_PUSH("gemv_fp8_swiglu");
    MC_LAUNCH_PDL(gemv_fp8_swiglu_kernel, grid, dim3(kBlockThreads), stream, pdl, 
        d_w, d_scale, d_gate_up, d_y, N);
    MC_NVTX_POP();
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        mc::set_error("gemv_fp8_swiglu launch failed: %s", cudaGetErrorString(e));
        return MC_E_CUDA;
    }
    return MC_OK;
}

mc_status_t launch_gemv_fp8_prologue(const uint8_t* d_w, const float* d_scale,
                                     const uint16_t* d_res_in,
                                     const uint16_t* d_addend,
                                     const uint16_t* d_norm_w,
                                     uint16_t* d_res_out, uint16_t* d_y,
                                     uint32_t N, uint32_t n, float eps,
                                     cudaStream_t stream, bool pdl /* = false */) {
    if (d_w == nullptr || d_scale == nullptr || d_res_in == nullptr ||
        d_norm_w == nullptr || d_res_out == nullptr ||
        d_y == nullptr || N == 0 || n != 2048u || !aligned16(d_w)) {
        mc::set_error("launch_gemv_fp8_prologue: invalid args (N=%u n=%u)", N, n);
        return MC_E_INVALID_ARGUMENT;
    }
    const dim3 grid((N + kWarpsPerBlock - 1u) / kWarpsPerBlock);
    MC_NVTX_PUSH("gemv_fp8_prologue");
    MC_LAUNCH_PDL(gemv_fp8_prologue_kernel, grid, dim3(kBlockThreads), stream, pdl, 
        d_w, d_scale, d_res_in, d_addend, d_norm_w, d_res_out, d_y, N, n, eps);
    MC_NVTX_POP();
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        mc::set_error("gemv_fp8_prologue launch failed: %s", cudaGetErrorString(e));
        return MC_E_CUDA;
    }
    return MC_OK;
}

mc_status_t launch_gemv_fp8(const uint8_t* d_w, const float* d_scale,
                            const uint16_t* d_x, uint16_t* d_y, uint32_t N,
                            uint32_t K, cudaStream_t stream, bool pdl /* = false */) {
    if (d_w == nullptr || d_scale == nullptr || d_x == nullptr || d_y == nullptr ||
        N == 0 || K == 0) {
        mc::set_error("launch_gemv_fp8: invalid args (N=%u K=%u)", N, K);
        return MC_E_INVALID_ARGUMENT;
    }
    const dim3 grid((N + kWarpsPerBlock - 1u) / kWarpsPerBlock);
    if ((K & 15u) == 0u && aligned16(d_w) && aligned16(d_x)) {
        // 向量主体（真实 decode shape：K=2048/6144 均 %16==0 → 行偏移恒对齐）
        MC_NVTX_PUSH("gemv_fp8_rows");
        MC_LAUNCH_PDL(gemv_fp8_rows_kernel, grid, dim3(kBlockThreads), stream, pdl, 
            d_w, d_scale, d_x, d_y, N, K);
        MC_NVTX_POP();
    } else {
        MC_NVTX_PUSH("gemv_fp8_rows_scalar");
        gemv_fp8_rows_scalar_kernel<<<grid, dim3(kBlockThreads), 0, stream>>>(
            d_w, d_scale, d_x, d_y, N, K);
        MC_NVTX_POP();
    }
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        mc::set_error("gemv_fp8 launch failed: %s", cudaGetErrorString(e));
        return MC_E_CUDA;
    }
    return MC_OK;
}

// F1 多行 gemv-fp8：签名与 launch_gemv_fp8 同族仅多 M 维（d_x [M,K] →
// d_y [M,N]）。向量/标量分派同 solo（K%16==0 且 16B 对齐 → 向量主体；
// M∈[2,8] 全模板特化分派）。grid = ceil(N/8)、block 256 建期固定。
mc_status_t launch_gemv_fp8_rows_m(const uint8_t* d_w, const float* d_scale,
                                   const uint16_t* d_x, uint16_t* d_y,
                                   uint32_t M, uint32_t N, uint32_t K,
                                   cudaStream_t stream, bool pdl /* = false */) {
    if (d_w == nullptr || d_scale == nullptr || d_x == nullptr || d_y == nullptr ||
        M < 2u || M > 8u || N == 0 || K == 0) {
        mc::set_error("launch_gemv_fp8_rows_m: invalid args (M=%u N=%u K=%u)", M,
                      N, K);
        return MC_E_INVALID_ARGUMENT;
    }
    const dim3 grid((N + kWarpsPerBlock - 1u) / kWarpsPerBlock);
    MC_NVTX_PUSH("gemv_fp8_rows_m");
#define MC_F8M_CASE_VEC(MV)                                                    \
    case MV##u:                                                                \
        MC_LAUNCH_PDL(gemv_fp8_rows_m_kernel<MV##u>, grid, dim3(kBlockThreads), \
                      stream, pdl, d_w, d_scale, d_x, d_y, N, K);               \
        break
#define MC_F8M_CASE_SCA(MV)                                                    \
    case MV##u:                                                                \
        gemv_fp8_rows_m_scalar_kernel<MV##u><<<grid, dim3(kBlockThreads), 0,    \
            stream>>>(d_w, d_scale, d_x, d_y, N, K);                           \
        break
    if ((K & 15u) == 0u && aligned16(d_w) && aligned16(d_x)) {
        // 向量主体（x 行偏移 2·m·K：K%16==0 → 恒 16B 对齐）
        switch (M) {
            MC_F8M_CASE_VEC(2);
            MC_F8M_CASE_VEC(3);
            MC_F8M_CASE_VEC(4);
            MC_F8M_CASE_VEC(5);
            MC_F8M_CASE_VEC(6);
            MC_F8M_CASE_VEC(7);
            MC_F8M_CASE_VEC(8);
        default:
            MC_NVTX_POP();
            mc::set_error("launch_gemv_fp8_rows_m: unreachable M=%u", M);
            return MC_E_INVALID_ARGUMENT;
        }
    } else {
        // 标量回退（任意 K；同 solo 的 scalar 路径语义/数值）
        switch (M) {
            MC_F8M_CASE_SCA(2);
            MC_F8M_CASE_SCA(3);
            MC_F8M_CASE_SCA(4);
            MC_F8M_CASE_SCA(5);
            MC_F8M_CASE_SCA(6);
            MC_F8M_CASE_SCA(7);
            MC_F8M_CASE_SCA(8);
        default:
            MC_NVTX_POP();
            mc::set_error("launch_gemv_fp8_rows_m: unreachable M=%u", M);
            return MC_E_INVALID_ARGUMENT;
        }
    }
#undef MC_F8M_CASE_VEC
#undef MC_F8M_CASE_SCA
    MC_NVTX_POP();
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        mc::set_error("gemv_fp8_rows_m launch failed: %s", cudaGetErrorString(e));
        return MC_E_CUDA;
    }
    return MC_OK;
}
