// rope_kv.cu — P2 真实现：RoPE(rotate_half 非交错) + KV 页写入（计划 §8.2/§9.3）。
//
// qkv 行布局 [tokens,2560]（wpk layer{i}.qkv 行拼序）：
//   列 0..2047    = q（16 head × 128）
//   列 2048..2303 = k（2 kv head × 128）
//   列 2304..2559 = v（2 kv head × 128）
// grid=(tokens, 20)，block 128 threads（每线程 1 个 dim）：
//   blockIdx.y < 16     → q head：原地 rotate_half
//   blockIdx.y ∈ {16,17} → k head 0/1：rope 后写 KV（kv=0）
//   blockIdx.y ∈ {18,19} → v head 0/1：直通写 KV（kv=1）
// 位置 pos = base_pos + blockIdx.x。
// 数值：fp32 计算（inv_freq = theta^(-pair/64)，pair∈[0,64)）；cos/sin fp32；
// q/k 输出舍入 bf16；KV 存 bf16。
#include "kernels/sm120/kernels.h"

#include <cuda_bf16.h>

#include "runtime/error.h"

namespace {

using bf16 = __nv_bfloat16;

// rotate_half 非交错（HF 语义）：
//   d < 64 : out = x[d]*cos[p] - x[d+64]*sin[p]
//   d >= 64: out = x[d]*cos[d-64] + x[d-64]*sin[d-64]
// P3 可捕获性：base_pos 从固定 device 地址 d_base_pos 运行时读取（§12.2），
// 不作为 kernel 参数 —— decode 步之间只改 device state 内容，launch 参数
// 全部不变（CUDA Graph replay 语义）。
__global__ void rope_kv_kernel(uint16_t* __restrict__ qkv, uint32_t tokens,
                               const uint32_t* __restrict__ d_base_pos,
                               uint32_t layer,
                               const uint32_t* __restrict__ page_table,
                               uint16_t* __restrict__ kv_pool, KvLayout layout,
                               float theta,
                               const float* __restrict__ inv_freq_table) {
    const uint32_t t = blockIdx.x;
    const uint32_t sub = blockIdx.y;
    const int d = threadIdx.x;
    MC_PDL_TRIGGER(); // 入口放行后继（后继消费本 kernel 输出前必 wait）
    MC_PDL_WAIT(); // qkv 行由前驱 GEMV 写出
    if (t >= tokens || d >= 128) return;

    const uint64_t row = (uint64_t)t * 2560u;
    const uint32_t pos = *d_base_pos + t;
    const uint32_t page = page_table[layout.page_index(pos)];
    const uint32_t slot = layout.slot_index(pos);
    const int pair = d < 64 ? d : d - 64;
    // inv_freq 与 python 权威一致：float64 幂再落 fp32（reference_forward.py
    // 的 _INV_FREQ = (theta ** -arange/128, float64).astype(float32)）。
    // P5：优先读建期表（同一 device 表达式预计算 → 逐 bit 一致）；
    // 表缺失（kernel_ops 直调，默认参数 nullptr）时回退逐线程计算。
    const float inv_freq =
        inv_freq_table != nullptr ? inv_freq_table[pair]
                                  : (float)pow((double)theta, -(double)pair / 64.0);
    const float freq = (float)pos * inv_freq;
    const float cs = cosf(freq), sn = sinf(freq);

    if (sub < 16u) {
        // q 原地旋转：d 与 d+64 互读对方的「原始」值。若直接读写全局内存，
        // 跨 warp 的读-写竞争会让后执行的一方读到已旋转的伙伴值（kernel_ops
        // rope prefill 捕获：t=1 head=3 d=64 差异 ~O(1)）。先用 shared 暂存
        // 本头 128 个原始值，同步后统一计算，再写回。
        const uint32_t base = sub * 128u;
        __shared__ uint16_t orig[128];
        orig[d] = qkv[row + base + (uint32_t)d];
        __syncthreads();
        const uint32_t partner = d < 64 ? (uint32_t)d + 64u : (uint32_t)d - 64u;
        const float x = __bfloat162float(reinterpret_cast<const bf16&>(orig[d]));
        const float x2 = __bfloat162float(reinterpret_cast<const bf16&>(orig[partner]));
        const float out = d < 64 ? x * cs - x2 * sn : x * cs + x2 * sn;
        qkv[row + base + (uint32_t)d] =
            __bfloat16_as_ushort(__float2bfloat16(out));
    } else if (sub < 18u) {
        const uint32_t head = sub - 16u;
        const uint32_t base = 2048u + head * 128u;
        const float x = __bfloat162float(reinterpret_cast<const bf16&>(qkv[row + base + (uint32_t)d]));
        float out;
        if (d < 64) {
            const float x2 =
                __bfloat162float(reinterpret_cast<const bf16&>(qkv[row + base + (uint32_t)d + 64u]));
            out = x * cs - x2 * sn;
        } else {
            const float x2 =
                __bfloat162float(reinterpret_cast<const bf16&>(qkv[row + base + (uint32_t)d - 64u]));
            out = x * cs + x2 * sn;
        }
        const uint64_t off = layout.elem_offset(layer, page, slot, 0u, head, (uint32_t)d);
        kv_pool[off] = __bfloat16_as_ushort(__float2bfloat16(out));
    } else {
        const uint32_t head = sub - 18u;
        const uint32_t col = 2304u + head * 128u + (uint32_t)d;
        const uint64_t off = layout.elem_offset(layer, page, slot, 1u, head, (uint32_t)d);
        kv_pool[off] = qkv[row + col];
    }
}

} // namespace

// P5：inv_freq 表初始化（session 建期一次）：与原核内逐线程计算同一
// 表达式（device pow(double) → float）→ 逐 bit 一致；decode 热路径 kernel
// 改读表（消除每线程的 double pow）。表为常量内容、固定地址（graph
// 兼容：建期写入，replay 只读）。
__global__ void rope_inv_freq_init_kernel(float* __restrict__ table, float theta) {
    const int pair = threadIdx.x; // 0..63
    if (pair >= 64) return;
    table[pair] = (float)pow((double)theta, -(double)pair / 64.0);
}

mc_status_t launch_rope_inv_freq_init(float* d_table, double rope_theta,
                                      cudaStream_t stream) {
    if (d_table == nullptr) {
        mc::set_error("launch_rope_inv_freq_init: table is NULL");
        return MC_E_INVALID_ARGUMENT;
    }
    rope_inv_freq_init_kernel<<<dim3(1), dim3(64), 0, stream>>>(d_table,
                                                                (float)rope_theta);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        mc::set_error("rope inv_freq init launch failed: %s", cudaGetErrorString(e));
        return MC_E_CUDA;
    }
    return MC_OK;
}

mc_status_t launch_rope_kv(uint16_t* d_qkv, uint32_t tokens,
                           const uint32_t* d_base_pos, uint32_t layer,
                           const uint32_t* d_page_table,
                           uint16_t* d_kv_pool, KvLayout layout, double rope_theta,
                           cudaStream_t stream,
                           const float* d_inv_freq /* = nullptr，见 kernels.h */, bool pdl /* = false */) {
    if (d_qkv == nullptr || d_base_pos == nullptr || d_page_table == nullptr ||
        d_kv_pool == nullptr || tokens == 0) {
        mc::set_error("launch_rope_kv: invalid args (tokens=%u)", tokens);
        return MC_E_INVALID_ARGUMENT;
    }
    MC_NVTX_PUSH("rope_kv");
    MC_LAUNCH_PDL(rope_kv_kernel, dim3(tokens, 20), dim3(128), stream, pdl,
                  d_qkv, tokens, d_base_pos, layer, d_page_table, d_kv_pool,
                  layout, (float)rope_theta, d_inv_freq);
    MC_NVTX_POP();
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        mc::set_error("rope_kv launch failed: %s", cudaGetErrorString(e));
        return MC_E_CUDA;
    }
    return MC_OK;
}
