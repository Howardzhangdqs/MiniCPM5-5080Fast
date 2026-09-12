// sampling.cu — greedy argmax（真实可用）+ 确定性 mock_next_token +
// P3 decode 状态推进 + P4+（§14.2）GPU 采样管线。
#include "kernels/sm120/kernels.h"

#include <climits>

#include <cuda_bf16.h>

#include "runtime/error.h"

namespace {

using bf16 = __nv_bfloat16;

// ===================== greedy argmax（fp32 logits） =====================
__global__ void greedy_argmax_kernel(const float* __restrict__ logits,
                                     int32_t n,
                                     int32_t* __restrict__ out) {
    extern __shared__ float sval[]; // blockDim.x floats
    __shared__ int32_t sidx[1024];
    const int t = threadIdx.x;
    float best = -INFINITY;
    int32_t best_i = INT_MAX;
    for (int i = t; i < n; i += blockDim.x) {
        const float v = logits[i];
        if (v > best || (v == best && i < best_i)) {
            best = v;
            best_i = i;
        }
    }
    sval[t] = best;
    sidx[t] = best_i;
    __syncthreads();
    for (unsigned stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if ((unsigned)t < stride) {
            if (sval[t + stride] > sval[t] ||
                (sval[t + stride] == sval[t] && sidx[t + stride] < sidx[t])) {
                sval[t] = sval[t + stride];
                sidx[t] = sidx[t + stride];
            }
        }
        __syncthreads();
    }
    if (t == 0) *out = sidx[0];
}

// 确定性 mock forward：token = 100 + ((seq_len * 2654435761u) ^ rng) % 1000
__global__ void mock_next_token_kernel(int32_t* __restrict__ out,
                                       uint32_t seq_len,
                                       uint32_t rng,
                                       int32_t force) {
    const uint32_t h = (seq_len * 2654435761u) ^ rng;
    *out = (force >= 0) ? force : (int32_t)(100u + (h % 1000u));
}

// P3：decode 步状态推进（CUDA Graph 尾节点）：
//   d_seq[*d_seq_len] = *d_next_token;  *d_seq_len += 1;
__global__ void decode_advance_kernel(int32_t* __restrict__ d_seq,
                                      const int32_t* __restrict__ d_next_token,
                                      uint32_t* __restrict__ d_seq_len) {
    MC_PDL_TRIGGER(); // 入口放行后继（后继消费本 kernel 输出前必 wait）
    MC_PDL_WAIT(); // d_next_token 由前驱 argmax 写出

    const uint32_t p = *d_seq_len;
    d_seq[p] = *d_next_token;
    *d_seq_len = p + 1u;
}

// ===================== greedy argmax（bf16 logits，P2） =====================
__global__ void greedy_argmax_bf16_kernel(const uint16_t* __restrict__ logits,
                                           int32_t n, int32_t* __restrict__ out) {
    extern __shared__ float sval[]; // blockDim.x floats
    __shared__ int32_t sidx[1024];
    const int t = threadIdx.x;
    float best = -INFINITY;
    int32_t best_i = INT_MAX;
    for (int i = t; i < n; i += blockDim.x) {
        const float v =
            __bfloat162float(reinterpret_cast<const bf16&>(logits[i]));
        if (v > best || (v == best && i < best_i)) {
            best = v;
            best_i = i;
        }
    }
    sval[t] = best;
    sidx[t] = best_i;
    __syncthreads();
    for (unsigned stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if ((unsigned)t < stride) {
            if (sval[t + stride] > sval[t] ||
                (sval[t + stride] == sval[t] && sidx[t + stride] < sidx[t])) {
                sval[t] = sval[t + stride];
                sidx[t] = sidx[t + stride];
            }
        }
        __syncthreads();
    }
    if (t == 0) *out = sidx[0];
}


// ===================== P5：两段式 greedy argmax（bf16 logits）=====================
// 单 block 版在 vocab=130560 下实测 22.8µs（1024 线程串行 127 元素/线程）。
// 两段式：阶段 1 每 block 扫连续切片出局部 (best, idx)（tie→小下标，与单
// block 版同比较式），阶段 2 单 block 按同序合并 —— 胜者与单 block 版完全
// 一致（切片不相交、比较语义相同）。partial 走固定地址 scratch（graph 兼容；
// 与 split-K GEMV 的 partial 复用同一缓冲，流序保证不重叠）。
__global__ void greedy_bf16_stage1_kernel(const uint16_t* __restrict__ logits,
                                          int32_t n, float* __restrict__ pv,
                                          int32_t* __restrict__ pi) {
    MC_PDL_TRIGGER(); // 入口放行后继（后继消费本 kernel 输出前必 wait）
    MC_PDL_WAIT(); // logits 由前驱 lm_head 写出
    const int slice = blockIdx.x;
    const int t = threadIdx.x;
    const int per = (n + gridDim.x - 1) / gridDim.x;
    const int lo = slice * per;
    const int hi = min(lo + per, n);
    float best = -INFINITY;
    int32_t best_i = INT_MAX;
    for (int i = lo + t; i < hi; i += blockDim.x) {
        const float v =
            __bfloat162float(reinterpret_cast<const bf16&>(logits[i]));
        if (v > best || (v == best && i < best_i)) {
            best = v;
            best_i = i;
        }
    }
    // block 归约（shared 树，tie→小下标）
    __shared__ float sv[512];
    __shared__ int32_t si[512];
    sv[t] = best;
    si[t] = best_i;
    __syncthreads();
    for (unsigned stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if ((unsigned)t < stride) {
            if (sv[t + stride] > sv[t] ||
                (sv[t + stride] == sv[t] && si[t + stride] < si[t])) {
                sv[t] = sv[t + stride];
                si[t] = si[t + stride];
            }
        }
        __syncthreads();
    }
    if (t == 0) {
        pv[slice] = sv[0];
        pi[slice] = si[0];
    }
}

__global__ void greedy_bf16_stage2_kernel(const float* __restrict__ pv,
                                          const int32_t* __restrict__ pi,
                                          int32_t n_slices,
                                          int32_t* __restrict__ out) {
    MC_PDL_TRIGGER(); // 入口放行后继（后继消费本 kernel 输出前必 wait）
    MC_PDL_WAIT(); // pv/pi 由阶段 1 写出
    const int t = threadIdx.x;
    __shared__ float sv[64];
    __shared__ int32_t si[64];
    float best = -INFINITY;
    int32_t best_i = INT_MAX;
    if (t < n_slices) {
        best = pv[t];
        best_i = pi[t];
    }
    sv[t] = best;
    si[t] = best_i;
    __syncthreads();
    for (unsigned stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if ((unsigned)t < stride) {
            if (sv[t + stride] > sv[t] ||
                (sv[t + stride] == sv[t] && si[t + stride] < si[t])) {
                sv[t] = sv[t + stride];
                si[t] = si[t + stride];
            }
        }
        __syncthreads();
    }
    if (t == 0) *out = si[0];
}

// ===================== B1b：批 greedy argmax（bf16 logits，一行一 block）====
// 组集中 logits [n][vocab] → 每 slot 一 block 求 argmax（fp32 比较，
// tie→最小下标 —— 比较式与 greedy_argmax_bf16_kernel 逐条相同）→ 结果写
// device 指针表 d_slot_tok[slot] 指向的成员 d_next_token（int32）。
// 归约：warp 内 butterfly（__shfl_xor 同步原语）+ 跨 warp shared 树。
__global__ void batch_argmax_bf16_kernel(
    const uint16_t* __restrict__ logits,             // [n][vocab]
    int32_t vocab,
    int32_t* const* __restrict__ d_slot_tok) {       // [n] 写目标（成员 d_next_token）
    const int row = blockIdx.x;
    const int t = threadIdx.x;
    const uint16_t* __restrict__ lrow = logits + (int64_t)row * vocab;
    float best = -INFINITY;
    int32_t best_i = INT_MAX;
    for (int i = t; i < vocab; i += blockDim.x) {
        const float v = __bfloat162float(reinterpret_cast<const bf16&>(lrow[i]));
        if (v > best || (v == best && i < best_i)) {
            best = v;
            best_i = i;
        }
    }
    // warp 内归约（lane 0 持胜者）
    for (unsigned off = 16u; off > 0u; off >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffffu, best, off);
        const int32_t oi = __shfl_xor_sync(0xffffffffu, best_i, off);
        if (ov > best || (ov == best && oi < best_i)) {
            best = ov;
            best_i = oi;
        }
    }
    const int nwarps = blockDim.x >> 5;
    const int wid = t >> 5, lane = t & 31;
    __shared__ float wv[32];
    __shared__ int32_t wi[32];
    if (lane == 0) {
        wv[wid] = best;
        wi[wid] = best_i;
    }
    __syncthreads();
    // warp0 跨 warp 归约（shfl 版树，nwarps ≤ 32）
    if (wid == 0) {
        best = (lane < nwarps) ? wv[lane] : -INFINITY;
        best_i = (lane < nwarps) ? wi[lane] : INT_MAX;
        for (unsigned off = 16u; off > 0u; off >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, best, off);
            const int32_t oi = __shfl_xor_sync(0xffffffffu, best_i, off);
            if (ov > best || (ov == best && oi < best_i)) {
                best = ov;
                best_i = oi;
            }
        }
        if (lane == 0) *d_slot_tok[row] = best_i;
    }
}

// ===================== B1b：批 decode 状态推进 =====================
// 每 slot 一线程（单 block）：读各成员 device 侧长度 → d_seq[len] 追加
// 已决策 token → 长度 +1（写成员 &d_state->seq_len）。与 solo eager 路径的
// 「D2D append + host seq_len+=1 + 8B H2D 回写」等价，但全 device 侧
//（批步 n 个成员一次完成；成员 host 镜像由编排层在 sync 后同步）。
__global__ void batch_advance_kernel(
    int32_t* const* __restrict__ d_slot_seq,          // [n] 成员 d_seq（写）
    const int32_t* const* __restrict__ d_slot_tok,    // [n] 成员 d_next_token（读）
    uint32_t* const* __restrict__ d_slot_pos,         // [n] &成员 seq_len（读+写）
    uint32_t n) {
    const uint32_t s = threadIdx.x;
    if (s >= n) return;
    const uint32_t p = *d_slot_pos[s];
    d_slot_seq[s][p] = *d_slot_tok[s];
    *d_slot_pos[s] = p + 1u;
}

// ============ P4+（§14.2）：GPU 采样管线 ============
// 顺序：repetition penalty（两段无原子）→ top-k（直方图预滤 + shared
// bitonic 全排序）→ top-p（前缀截断）→ temperature softmax → device RNG
// 采样。greedy argmax 原样保留：激活规则（temperature ≤0 或 top_k ≤1 →
// greedy）host 侧判定。CUDA Graph 兼容：参数全部从 device 侧固定地址运行
// 时读取，kernel/grid 建期固定。
//
// top-k 的精确抽取（两层直方图 + bitonic）：
//   S1 hist1：vocab 的 fp32 可序键（符号位翻转）高 8 位直方图（原子加，
//      整数和 → 确定性）；
//   S2 thr1（单 block）：自最强 bin 向下累计，找首个 cum ≥ k 的 bin →
//      16 位阈值 thr16；
//   S3 compact：key16 > thr16 的元素原子追加进候选列表（顺序不定，但后续
//      全排序与顺序无关）；key16 == thr16 的成员计入第二层直方图 hist2；
//   S4 thr2：bin 内下一 8 位阈值 thr8；S4b 把 key16==thr16 且 byte2 ≥ thr8
//      者追加进列表。候选总数 ≤ 128 + 255 + 255 ≤ 638（kSampSortCap 截断
//      保险）；全局 top-k ⊆ 候选集（构造保证）；
//   S5 sort+尾巴：shared bitonic 全排序（确定性全序：值降、并列下标升）
//      → 前 k → softmax/top-p/RNG。
// 历史注记：锦标赛/堆路径（单 block 747µs → warp-heap 三级 1.1ms）受
// ~1µs/轮次的隐式代价（分歧收敛 + shared 依赖链）拖累；本方案全程无
// 数据依赖轮次循环，端到端 ~40µs。

__device__ __forceinline__ uint32_t rng_next(uint64_t* st) {
    uint64_t x = *st;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *st = x;
    return (uint32_t)((x * 2685821657736338717ull) >> 32);
}

// fp32 → 单调 u32 键（符号位翻转技巧）
__device__ __forceinline__ uint32_t f32_sort_key(float v) {
    uint32_t u = __float_as_uint(v);
    return (u & 0x80000000u) ? ~u : (u ^ 0x80000000u);
}

__device__ __forceinline__ bool samp_stronger(float va, int32_t ia, float vb,
                                              int32_t ib) {
    return (va > vb) || (va == vb && ia < ib);
}

constexpr uint32_t kSampThreads = 256;
// kSampBlocks/kSampMaxK 取 kernels.h 全局定义；kSampSortCap 见 kernels.h

// ---- kernel A：rep-penalty 位图（幂等写 1，无原子） ----
__global__ void rep_penalty_mark_kernel(const int32_t* __restrict__ d_seq,
                                        const uint32_t* __restrict__ d_seq_len,
                                        uint8_t* __restrict__ bitmap,
                                        uint32_t max_tokens) {
    const uint32_t j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= max_tokens) return;
    if (j >= *d_seq_len) return;
    const int32_t t = d_seq[j];
    if (t >= 0) bitmap[t] = 1;
}

// ---- kernel B：位图命中 logit 应用标准 penalty，bf16 → fp32 ----
__global__ void rep_penalty_apply_kernel(const uint16_t* __restrict__ logits,
                                         const uint8_t* __restrict__ bitmap,
                                         const SamplingConfigDevice* __restrict__ cfg,
                                         float* __restrict__ out, int32_t n) {
    const int32_t i = (int32_t)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= n) return;
    float l = __bfloat162float(reinterpret_cast<const bf16&>(logits[i]));
    const float p = cfg->rep_penalty;
    if (bitmap[i] != 0 && p != 1.0f) {
        l = (l > 0.0f) ? l / p : l * p;
    }
    out[i] = l;
}

// ---- S1：高 8 位直方图（grid 固定按 n；原子加整数和 → 确定性） ----
__global__ void samp_hist1_kernel(const float* __restrict__ logits,
                                  uint32_t* __restrict__ hist, int32_t n) {
    __shared__ uint32_t local[256];
    const uint32_t t = threadIdx.x;
    local[t] = 0;
    __syncthreads();
    for (int32_t i = (int32_t)(blockIdx.x * blockDim.x) + (int32_t)t; i < n;
         i += (int32_t)(gridDim.x * blockDim.x)) {
        atomicAdd(&local[f32_sort_key(logits[i]) >> 24], 1u);
    }
    __syncthreads();
    atomicAdd(&hist[t], local[t]);
}

// ---- S2：16 位阈值（单 block；自最强 bin 向下累计到 ≥ k） ----
__global__ void samp_thr1_kernel(const uint32_t* __restrict__ hist,
                                 const SamplingConfigDevice* __restrict__ cfg,
                                 uint32_t* __restrict__ thr16) {
    __shared__ uint32_t cum[256];
    const uint32_t t = threadIdx.x;
    // 前缀和（单 pass 树形；自高向低的「强序」= bin 值大者强）
    cum[t] = hist[t];
    __syncthreads();
    for (uint32_t off = 1; off < 256; off <<= 1) {
        uint32_t v = 0;
        if (t >= off) v = cum[t - off];
        __syncthreads();
        cum[t] += v;
        __syncthreads();
    }
    // cum[b] = bin ≤ b 的总数；最强序累计 = hist[b] 的 b 递减…… 换算：
    // total_below[b] = cum[b] - hist[b]（严格弱于 b 的数量）
    const uint32_t k = min(cfg->top_k, kSampMaxK);
    // 找最大 bin b 使 (total - cum_below_incl... ) 直接扫描：
    // threshold bin = 最小 b 满足「bin ≥ b 的总数 ≥ k」= hist 的高位后缀和
    // suffix[b] = total - (cum[b] - hist[b])
    const uint32_t total = cum[255];
    __shared__ uint32_t s_thr;
    if (t == 0) s_thr = 0;
    __syncthreads();
    // 每 thread 检查自己的 bin：suffix(b) ≥ k 且 suffix(b+1) < k → b 是阈值 bin
    const uint32_t suffix_b = total - (cum[t] - hist[t]);
    const uint32_t suffix_above = (t == 255) ? 0 : (total - cum[t]);
    if (suffix_b >= k && suffix_above < k) s_thr = t; // 唯一
    __syncthreads();
    if (t == 0) *thr16 = s_thr; // 阈值 bin（key 的最高字节）；bin 之上全取、
                                // bin 内走第二层
}

// ---- S3：compact（key16 > thr16 追加列表；== thr16 计入 hist2） ----
__global__ void samp_compact1_kernel(const float* __restrict__ logits,
                                      const uint32_t* __restrict__ thr16,
                                      uint32_t* __restrict__ count,
                                      float* __restrict__ list_v,
                                      int32_t* __restrict__ list_i,
                                      uint32_t* __restrict__ hist2, int32_t n) {
    for (int32_t i = (int32_t)(blockIdx.x * blockDim.x) + (int32_t)threadIdx.x;
         i < n; i += (int32_t)(gridDim.x * blockDim.x)) {
        const uint32_t key = f32_sort_key(logits[i]);
        if ((key >> 24) > *thr16) {
            const uint32_t slot = atomicAdd(count, 1u);
            if (slot < kSampSortCap) {
                list_v[slot] = logits[i];
                list_i[slot] = i;
            }
        } else if ((key >> 24) == *thr16) {
            atomicAdd(&hist2[(key >> 16) & 0xFFu], 1u);
        }
    }
}

// ---- S4：bin 内第二层阈值 + 追加（单 kernel 两段：先 thr 后扫描） ----
__global__ void samp_thr2_kernel(const uint32_t* __restrict__ hist2,
                                 const SamplingConfigDevice* __restrict__ cfg,
                                 uint32_t* __restrict__ thr8) {
    // 第二层只需在 [0,256) 内找阈值 byte（bin 内成员的 key16 相同，取
    // byte2 使候选总数 ≤ k-1 + 255……与 S2 同算法，规模 256）
    __shared__ uint32_t cum[256];
    const uint32_t t = threadIdx.x;
    cum[t] = hist2[t];
    __syncthreads();
    for (uint32_t off = 1; off < 256; off <<= 1) {
        uint32_t v = 0;
        if (t >= off) v = cum[t - off];
        __syncthreads();
        cum[t] += v;
        __syncthreads();
    }
    const uint32_t k = min(cfg->top_k, kSampMaxK);
    const uint32_t total = cum[255];
    __shared__ uint32_t s_thr;
    if (t == 0) s_thr = 0;
    __syncthreads();
    const uint32_t suffix_b = total - (cum[t] - hist2[t]);
    const uint32_t suffix_above = (t == 255) ? 0 : (total - cum[t]);
    if (suffix_b >= k && suffix_above < k) s_thr = t;
    __syncthreads();
    if (t == 0) *thr8 = s_thr;
}

__global__ void samp_compact2_kernel(const float* __restrict__ logits,
                                     const uint32_t* __restrict__ thr16,
                                     const uint32_t* __restrict__ thr8,
                                     uint32_t* __restrict__ count,
                                     float* __restrict__ list_v,
                                     int32_t* __restrict__ list_i, int32_t n) {
    for (int32_t i = (int32_t)(blockIdx.x * blockDim.x) + (int32_t)threadIdx.x;
         i < n; i += (int32_t)(gridDim.x * blockDim.x)) {
        const uint32_t key = f32_sort_key(logits[i]);
        if ((key >> 24) == *thr16 && ((key >> 16) & 0xFFu) >= *thr8) {
            const uint32_t slot = atomicAdd(count, 1u);
            if (slot < kSampSortCap) {
                list_v[slot] = logits[i];
                list_i[slot] = i;
            }
        }
    }
}

// ---- S5：shared bitonic 全排序（确定性全序）+ softmax/top-p/RNG 尾巴 ----
__global__ void sampling_sort_final_kernel(
    const float* __restrict__ list_v, const int32_t* __restrict__ list_i,
    const uint32_t* __restrict__ count, const SamplingConfigDevice* __restrict__ cfg,
    uint64_t* __restrict__ rng, int32_t* __restrict__ out, int32_t n) {
    __shared__ float sv[kSampSortCap];
    __shared__ int32_t si[kSampSortCap];
    (void)n;
    const uint32_t t = threadIdx.x;
    // 载入（≤ kSampSortCap 实条目 + 哨兵填充；compact 顺序不定但全排序
    // 与输入顺序无关 → 结果确定）
    for (uint32_t i = t; i < kSampSortCap; i += kSampThreads) {
        if (i < *count) {
            sv[i] = list_v[i];
            si[i] = list_i[i];
        } else {
            sv[i] = -INFINITY;
            si[i] = 0x7FFFFFFF; // 哨兵（值最弱 + 下标最大）
        }
    }
    __syncthreads();
    // bitonic：降序（强在前）；比较器 samp_stronger
    for (uint32_t k = 2; k <= kSampSortCap; k <<= 1) {
        for (uint32_t j = k >> 1; j > 0; j >>= 1) {
            for (uint32_t i = t; i < kSampSortCap; i += kSampThreads) {
                const uint32_t l = i ^ j;
                if (l > i) {
                    const bool up = (i & k) != 0; // 该子序列方向
                    const bool a_first = samp_stronger(sv[i], si[i], sv[l], si[l]);
                    // 降序网络：非逆向段保持强在前；up 段翻转
                    if (a_first == up) {
                        const float tv = sv[i];
                        const int32_t ti = si[i];
                        sv[i] = sv[l];
                        si[i] = si[l];
                        sv[l] = tv;
                        si[l] = ti;
                    }
                }
            }
            __syncthreads();
        }
    }
    // 尾巴（thread 0）：前 k → softmax → top-p → RNG
    if (t != 0) return;
    const uint32_t k = min(cfg->top_k, kSampMaxK);
    const uint32_t hsize = min(k, *count);
    const float temp = cfg->temperature; // host 激活规则保证 > 0
    const float mx = sv[0];
    float sum = 0.f;
    for (uint32_t i = 0; i < hsize; ++i) {
        sv[i] = expf((sv[i] - mx) / temp);
        sum += sv[i];
    }
    for (uint32_t i = 0; i < hsize; ++i) sv[i] /= sum;
    const float p = cfg->top_p;
    uint32_t cut = hsize;
    if (p < 1.0f) {
        float cum = 0.f;
        for (uint32_t i = 0; i < hsize; ++i) {
            cum += sv[i];
            if (cum >= p) {
                cut = i + 1;
                break;
            }
        }
    }
    float kept = 0.f;
    for (uint32_t i = 0; i < cut; ++i) kept += sv[i];
    const uint32_t r = rng_next(rng);
    const float u = (float)(r >> 8) * (1.0f / 16777216.0f);
    float acc = 0.f;
    uint32_t pick = cut - 1;
    for (uint32_t i = 0; i < cut; ++i) {
        acc += sv[i] / kept;
        if (u < acc) {
            pick = i;
            break;
        }
    }
    *out = si[pick];
}

} // namespace

// ===================== launchers =====================

mc_status_t launch_sampling_greedy(const float* d_logits, int32_t n,
                                   int32_t* d_out_token, cudaStream_t stream) {
    if (d_logits == nullptr || d_out_token == nullptr || n <= 0) {
        mc::set_error("launch_sampling_greedy: invalid args (logits=%p out=%p n=%d)",
                      (const void*)d_logits, (const void*)d_out_token, n);
        return MC_E_INVALID_ARGUMENT;
    }
    constexpr int kThreads = 1024;
    MC_NVTX_PUSH("sampling_greedy");
    greedy_argmax_kernel<<<dim3(1), dim3(kThreads), kThreads * sizeof(float), stream>>>(
        d_logits, n, d_out_token);
    MC_NVTX_POP();
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        mc::set_error("sampling_greedy launch failed: %s", cudaGetErrorString(e));
        return MC_E_CUDA;
    }
    return MC_OK;
}

mc_status_t launch_mock_next_token(int32_t* d_out, uint32_t seq_len, uint32_t rng,
                                   int32_t force, cudaStream_t stream) {
    if (d_out == nullptr) {
        mc::set_error("launch_mock_next_token: d_out is NULL");
        return MC_E_INVALID_ARGUMENT;
    }
    mock_next_token_kernel<<<dim3(1), dim3(1), 0, stream>>>(d_out, seq_len, rng, force);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        mc::set_error("mock_next_token launch failed: %s", cudaGetErrorString(e));
        return MC_E_CUDA;
    }
    return MC_OK;
}

mc_status_t launch_decode_advance(int32_t* d_seq, const int32_t* d_next_token,
                                  uint32_t* d_seq_len, cudaStream_t stream) {
    if (d_seq == nullptr || d_next_token == nullptr || d_seq_len == nullptr) {
        mc::set_error("launch_decode_advance: invalid args (d_seq=%p d_next=%p "
                      "d_seq_len=%p)",
                      (const void*)d_seq, (const void*)d_next_token,
                      (const void*)d_seq_len);
        return MC_E_INVALID_ARGUMENT;
    }
    MC_NVTX_PUSH("decode_advance");
    decode_advance_kernel<<<dim3(1), dim3(1), 0, stream>>>(d_seq, d_next_token,
                                                           d_seq_len);
    MC_NVTX_POP();
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        mc::set_error("decode_advance launch failed: %s", cudaGetErrorString(e));
        return MC_E_CUDA;
    }
    return MC_OK;
}

// B1b：批 greedy argmax —— 组集中 logits [n][vocab]，一行一 block，
// warp 归约（kernel 见上）；结果经 device 指针表写各成员 d_next_token。
mc_status_t launch_batch_argmax_bf16(const uint16_t* d_logits,
                                     int32_t* const* d_slot_tok,
                                     int32_t vocab, int32_t n,
                                     cudaStream_t stream) {
    if (d_logits == nullptr || d_slot_tok == nullptr || vocab <= 0 || n <= 0) {
        mc::set_error("launch_batch_argmax_bf16: invalid args (logits=%p slot_tok=%p "
                      "vocab=%d n=%d)",
                      (const void*)d_logits, (const void*)d_slot_tok, vocab, n);
        return MC_E_INVALID_ARGUMENT;
    }
    MC_NVTX_PUSH("batch_argmax_bf16");
    batch_argmax_bf16_kernel<<<dim3(n), dim3(512), 0, stream>>>(d_logits, vocab,
                                                                d_slot_tok);
    MC_NVTX_POP();
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        mc::set_error("batch_argmax_bf16 launch failed: %s", cudaGetErrorString(e));
        return MC_E_CUDA;
    }
    return MC_OK;
}

// B1b：批 decode 状态推进 —— 每 slot 一线程（单 block 32 线程）。
mc_status_t launch_batch_advance(int32_t* const* d_slot_seq,
                                 const int32_t* const* d_slot_tok,
                                 uint32_t* const* d_slot_pos, uint32_t n,
                                 cudaStream_t stream) {
    if (d_slot_seq == nullptr || d_slot_tok == nullptr || d_slot_pos == nullptr ||
        n == 0 || n > 32u) {
        mc::set_error("launch_batch_advance: invalid args (seq=%p tok=%p pos=%p n=%u)",
                      (const void*)d_slot_seq, (const void*)d_slot_tok,
                      (const void*)d_slot_pos, n);
        return MC_E_INVALID_ARGUMENT;
    }
    MC_NVTX_PUSH("batch_advance");
    batch_advance_kernel<<<dim3(1), dim3(32), 0, stream>>>(d_slot_seq, d_slot_tok,
                                                           d_slot_pos, n);
    MC_NVTX_POP();
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        mc::set_error("batch_advance launch failed: %s", cudaGetErrorString(e));
        return MC_E_CUDA;
    }
    return MC_OK;
}

mc_status_t launch_sampling_greedy_bf16(const uint16_t* d_logits, int32_t n,
                                        int32_t* d_out_token, cudaStream_t stream,
                                        float* d_stage_v /* = nullptr */,
                                        int32_t* d_stage_i /* = nullptr */,
                                        bool pdl /* = false */) {
    // P5：vocab 级输入且有固定 scratch → 两段式（胜者与单 block 版一致：
    // 切片不相交 + 同一 tie→小下标比较式逐级归约）
    constexpr int32_t kTwoStageMin = 1 << 14;
    if (d_stage_v != nullptr && d_stage_i != nullptr && n >= kTwoStageMin) {
        constexpr int kSlices = 32; // 32 × 512 线程扫 130560 → ~8 元素/线程
        MC_NVTX_PUSH("sampling_greedy_bf16_s1");
        greedy_bf16_stage1_kernel<<<dim3(kSlices), dim3(512), 0, stream>>>(
            d_logits, n, d_stage_v, d_stage_i);
        MC_NVTX_POP();
        cudaError_t e1 = cudaGetLastError();
        if (e1 != cudaSuccess) {
            mc::set_error("sampling_greedy_bf16 stage1 launch failed: %s",
                          cudaGetErrorString(e1));
            return MC_E_CUDA;
        }
        MC_NVTX_PUSH("sampling_greedy_bf16_s2");
        greedy_bf16_stage2_kernel<<<dim3(1), dim3(64), 0, stream>>>(
            d_stage_v, d_stage_i, kSlices, d_out_token);
        MC_NVTX_POP();
        e1 = cudaGetLastError();
        if (e1 != cudaSuccess) {
            mc::set_error("sampling_greedy_bf16 stage2 launch failed: %s",
                          cudaGetErrorString(e1));
            return MC_E_CUDA;
        }
        return MC_OK;
    }
    if (d_logits == nullptr || d_out_token == nullptr || n <= 0) {
        mc::set_error("launch_sampling_greedy_bf16: invalid args (logits=%p out=%p n=%d)",
                      (const void*)d_logits, (const void*)d_out_token, n);
        return MC_E_INVALID_ARGUMENT;
    }
    constexpr int kThreads = 1024;
    MC_NVTX_PUSH("sampling_greedy_bf16");
    greedy_argmax_bf16_kernel<<<dim3(1), dim3(kThreads), kThreads * sizeof(float),
                                stream>>>(d_logits, n, d_out_token);
    MC_NVTX_POP();
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        mc::set_error("sampling_greedy_bf16 launch failed: %s", cudaGetErrorString(e));
        return MC_E_CUDA;
    }
    return MC_OK;
}

// ---- P4+：采样管线 ----
mc_status_t launch_rep_penalty_mark(const int32_t* d_seq,
                                    const uint32_t* d_seq_len, uint8_t* d_bitmap,
                                    uint32_t max_tokens, cudaStream_t stream) {
    if (d_seq == nullptr || d_seq_len == nullptr || d_bitmap == nullptr ||
        max_tokens == 0) {
        mc::set_error("launch_rep_penalty_mark: invalid args");
        return MC_E_INVALID_ARGUMENT;
    }
    const uint32_t grid = (max_tokens + 255u) / 256u; // 固定（建期 max_tokens）
    MC_NVTX_PUSH("rep_penalty_mark");
    rep_penalty_mark_kernel<<<grid, dim3(256), 0, stream>>>(d_seq, d_seq_len,
                                                            d_bitmap, max_tokens);
    MC_NVTX_POP();
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        mc::set_error("rep_penalty_mark launch failed: %s", cudaGetErrorString(e));
        return MC_E_CUDA;
    }
    return MC_OK;
}

mc_status_t launch_rep_penalty_apply(const uint16_t* d_logits_bf16,
                                     const uint8_t* d_bitmap,
                                     const SamplingConfigDevice* d_cfg,
                                     float* d_logits_f32, int32_t n,
                                     cudaStream_t stream) {
    if (d_logits_bf16 == nullptr || d_bitmap == nullptr || d_cfg == nullptr ||
        d_logits_f32 == nullptr || n <= 0) {
        mc::set_error("launch_rep_penalty_apply: invalid args (n=%d)", n);
        return MC_E_INVALID_ARGUMENT;
    }
    MC_NVTX_PUSH("rep_penalty_apply");
    rep_penalty_apply_kernel<<<dim3((n + 255) / 256), dim3(256), 0, stream>>>(
        d_logits_bf16, d_bitmap, d_cfg, d_logits_f32, n);
    MC_NVTX_POP();
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        mc::set_error("rep_penalty_apply launch failed: %s", cudaGetErrorString(e));
        return MC_E_CUDA;
    }
    return MC_OK;
}

mc_status_t launch_sampling_pipeline(const float* d_logits,
                                     const SamplingConfigDevice* d_cfg,
                                     uint64_t* d_rng_state, int32_t* d_out_token,
                                     int32_t n, float* d_partial_v,
                                     int32_t* d_partial_i, cudaStream_t stream) {
    // 采集 scratch 布局（复用 samp_partial 段，见 session.cpp 切分注释）：
    //   [0, 1KB)            hist1[256] u32
    //   [1KB, 2KB)          hist2[256] u32
    //   [2KB, 2KB+4)        count u32
    //   [8B 对齐后]         thr16/thr8 u32 ×2
    //   list_v[cap] float + list_i[cap] int32（d_partial_v/d_partial_i）
    if (d_logits == nullptr || d_cfg == nullptr || d_rng_state == nullptr ||
        d_out_token == nullptr || d_partial_v == nullptr || d_partial_i == nullptr ||
        n <= 0) {
        mc::set_error("launch_sampling_pipeline: invalid args (n=%d)", n);
        return MC_E_INVALID_ARGUMENT;
    }
    // scratch 布局（字节偏移，session.cpp 切分同口径）：
    //   [0,1KB) hist1 | [1KB,2KB) hist2 | [2KB,2KB+4) count |
    //   [2KB+4, +4) thr16 | [+8) thr8 | [4KB, 4KB+cap*4) list_v |
    //   [4KB+cap*4, +cap*4) list_i
    uint8_t* base = reinterpret_cast<uint8_t*>(d_partial_v);
    uint32_t* hist1 = reinterpret_cast<uint32_t*>(base + 0);
    uint32_t* hist2 = reinterpret_cast<uint32_t*>(base + 1024);
    uint32_t* count = reinterpret_cast<uint32_t*>(base + 2048);
    uint32_t* thr16 = reinterpret_cast<uint32_t*>(base + 2052);
    uint32_t* thr8 = reinterpret_cast<uint32_t*>(base + 2056);
    float* list_v = reinterpret_cast<float*>(base + 4096);
    int32_t* list_i = reinterpret_cast<int32_t*>(base + 4096 +
                                                (size_t)kSampSortCap * 4);
    const uint32_t grid = (n + (int32_t)kSampThreads - 1) / kSampThreads;
    (void)cudaMemsetAsync(base, 0, 4096, stream);
    MC_NVTX_PUSH("sampling_hist1");
    samp_hist1_kernel<<<grid, dim3(kSampThreads), 0, stream>>>(d_logits, hist1, n);
    MC_NVTX_POP();
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        mc::set_error("sampling hist1 launch failed: %s", cudaGetErrorString(e));
        return MC_E_CUDA;
    }
    samp_thr1_kernel<<<1, dim3(kSampThreads), 0, stream>>>(hist1, d_cfg, thr16);
    e = cudaGetLastError();
    if (e != cudaSuccess) {
        mc::set_error("sampling thr1 launch failed: %s", cudaGetErrorString(e));
        return MC_E_CUDA;
    }
    MC_NVTX_PUSH("sampling_compact1");
    samp_compact1_kernel<<<grid, dim3(kSampThreads), 0, stream>>>(
        d_logits, thr16, count, list_v, list_i, hist2, n);
    MC_NVTX_POP();
    e = cudaGetLastError();
    if (e != cudaSuccess) {
        mc::set_error("sampling compact1 launch failed: %s", cudaGetErrorString(e));
        return MC_E_CUDA;
    }
    samp_thr2_kernel<<<1, dim3(kSampThreads), 0, stream>>>(hist2, d_cfg, thr8);
    e = cudaGetLastError();
    if (e != cudaSuccess) {
        mc::set_error("sampling thr2 launch failed: %s", cudaGetErrorString(e));
        return MC_E_CUDA;
    }
    MC_NVTX_PUSH("sampling_compact2");
    samp_compact2_kernel<<<grid, dim3(kSampThreads), 0, stream>>>(
        d_logits, thr16, thr8, count, list_v, list_i, n);
    MC_NVTX_POP();
    e = cudaGetLastError();
    if (e != cudaSuccess) {
        mc::set_error("sampling compact2 launch failed: %s", cudaGetErrorString(e));
        return MC_E_CUDA;
    }
    MC_NVTX_PUSH("sampling_sort_final");
    sampling_sort_final_kernel<<<1, dim3(kSampThreads), 0, stream>>>(
        list_v, list_i, count, d_cfg, d_rng_state, d_out_token, n);
    MC_NVTX_POP();
    e = cudaGetLastError();
    if (e != cudaSuccess) {
        mc::set_error("sampling sort-final launch failed: %s", cudaGetErrorString(e));
        return MC_E_CUDA;
    }
    return MC_OK;
}
