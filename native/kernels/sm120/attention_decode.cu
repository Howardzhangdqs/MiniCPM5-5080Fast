// attention_decode.cu — P5-B/P5-C：decode 单 token 的 GQA attention。
//
// 演进：P3 两段式（grid=(16 q head, max_chunks)，每 q head 流读全 KV）→
// P5 cooperative 融合（CHUNK=256，ctx2048 仅 16 活跃 block，实测反慢）→
// P5-B kv-major + CHUNK=64 + ILP 流水（本版默认）。
//
// ============ P5-B：kv-major chunk kernel（GQA K/V 去冗余，默认路径）====
// 动机（三轮实测淘汰记录，RTX 5080 / 84 SM / 100KB smem/SM / 1536 thr/SM）：
//   - 旧 q-major CHUNK=256：K/V 全局流量 8×（每 kv head 被 8 个 q head 的
//     block 重复流读）；ctx2048 32.8µs/层（attentio n~1.5ms/step）。
//   - kv-major CHUNK=256（cooperative 草案）：流量 1× 但 ctx2048 只有
//     16 个活跃 block（2 kvh × 8 chunk）→ 68 SM 闲置，36.8µs 反慢。
//   - q-major CHUNK=64：512 block 铺满 SM → 14.96µs（纯并行度收益）。
//   - kv-major CHUNK=64 单缓冲 stage：barrier 串行装载/计算 → 20.7µs。
//   - kv-major CHUNK=64 + 双缓冲 16-pos stage + 8-pos ILP 批（本版）：
//     ctx2048 10.9µs/层。瓶颈解剖：装载层 8.4MB@8192 ≈ 12.7µs（≈HBM
//     上限），其余为 per-warp 串行 softmax 依赖链 → ILP-8（8 个位置的
//     dot/exp 链并行、每批一次 softmax 状态更新）破链。
//
// 结构（attn_kv_kernel）：
//   grid = (2 kv head, max_chunks=ceil(max_ctx/64))，block = 256 threads
//   = 8 warp；warp w 认领共享该 KV head 的 q head：h_q = kvh*8 + w。
//   K/V 由 block 协作装载 shared —— 每 kv head 的 K/V 全局流量 ÷8
//   （每元素恰被 1 个 block 读取）。双缓冲 16-pos stage：装载 stage t+1
//   与计算 stage t 重叠；stage 内 8-pos ILP 批（批内位置独立 dot 链 →
//   批级 softmax 更新：m'=max(m, batch)；l=l·renew+Σpw；acc 先 renew
//   再串加批内 pw·v —— 数学等价、舍入路径不同，见数值注）。
//   尾部（不足 8 pos 的批）走逐位串行支路（与单流版同锚表达式）。
//
//   ---- smem / occupancy 核算（sm_120 实测属性）----
//   dynamic smem：2(buf) × 2(K,V) × 16 pos × 128 dim × 2B = 16384B
//   （≤48KB 免 opt-in）+ static ~1KB。co-residency：threads 1536/256=6；
//   smem 102400/17KB≈5；regs（~60/thread）65536/(256×60)≈4 → 4 block/SM，
//   32 warp/SM。grid=(2,132)（max_ctx=8448）→ ~1.5 波调度。
//   ctx2048：64 活跃 block；K/V 流量 2×2048×512B = 2.1MB/层（旧 16.8MB）。
//
// ============ P5-C：rope 吸收（phase0，d_inv_freq != nullptr 时）=========
// decode 链原为 rope_kv_kernel（q 原地旋转回写 qkv + k 旋转写池 + v 拷贝
// 写池）→ attention 读 qkv/pool。phase0 把三件事全部搬进 attn_kv_kernel：
//   - q 旋转：lane 的 dim 分配 (d&31)+32r 使 rotate_half 伙伴 (d, d±64)
//     恒落于同 lane 的 (r, r^2) 分量对 → 纯寄存器旋转，零 shuffle、零
//     qkv 回写（q 用后即弃，qkv 行 q 列不再被读）。
//   - k 旋转 + 持久化：limit 所在 chunk 的 block 内 warp0 把 bf16 旋转
//     结果写 KV 池（后续 step 复用）；8 warp 直接用寄存器值算当前位
//     score（与读池逐位同值）。
//   - v 直通 + 持久化：同上。
//   rope kernel 从 decode 链消失（省一次 launch + 20 个小 block 的
//   qkv 读写往返）；prefill（T>1）仍用 rope kernel 原路径。
//   SASS 锚（rope_kv_kernel 实测指令序，cuobjdump -sass）：
//     FMUL(sn,x2) → FFMA(cs,x,∓t) → F2FP.BF16 → [STG]
//   phase0 用 __fmul_rn/__fmaf_rn 显式复刻该收缩选择；cosf/sinf 同
//   libdevice、freq=(float)pos*inv_freq 同式 → 逐位一致（kernel_ops
//   attention_rope_phase0：输出/pool/qkv 三向逐位锚）。
//
// ============ 阶段 2：并行 merge（P8-A1 起两级：merge_a → merge_b）========
// 旧 attn_merge_p（grid=16×1024 thr）在 CHUNK=64 下 n_valid 随 ctx 线性
// 增长：grid 仅 16 block（84 SM 闲置）+ 128K 时 level-1 partials 达
// 17.03MB/层，merge 每步 42 层各一次合计 ~4-5ms。生产路径改为两级：
//   attn_merge_a grid=(16 qh, 32 split)×128 thr：每 block 归并 c ≡ s (mod 32)
//   子序列 → level-2 partial [16][32][130]（266KB，42 层流序复用）；
//   attn_merge_b grid=(16)×128 thr：升序固定串加合并 32 组 → out。
// 两级合计算术与 attn_merge_p 等价（重分组 + 重标定底数），确定性串加。
// 短上下文（limit < chunk_pos，n_valid==1）：chunk kernel 的 direct 支路
// 已写 out（单组恒等退化 out=bf16(acc/l)，与 merge 公式逐位一致），
// merge_a/merge_b 同判定整块早退。attn_merge_p 本体保留（kernel_ops
// 三方对拍 + null-partials2 支路）。
//
// P3 可捕获性：limit 从固定 device 地址 d_base_pos 运行时读取（§12.2），
// launch 参数（grid/block/smem/指针/形状常量）步间不变 → CUDA Graph
// replay 安全；空块（chunk 起点 > limit）立即返回。
//
// 数值注：ILP 批 + 单 warp 升序流 vs 旧「4 子 warp 交错 + 块内合并」
// 均为 fp32 online softmax 的重分组（数学等价、舍入路径不同）。实测
// vs 旧 kernel：2048 输出中 ≤1 个 bf16 值差 1 ulp（随机数据 5 ctx 实验）；
// vs CPU double 参考（kernel_ops）：relL2≤1e-2 容差内，单流版逐位 0 ulp。
// golden 三 case 策略兜底（见 tests/golden_compare.h）。
//
// 回退：MC_ATTN_KV_OFF=1 → 旧 q-major 两段式（attn_chunk_kernel × 旧
// attn_merge_kernel，同一 partials 布局；不支持 phase0 —— forward.cpp
// 在该开关下保留 rope kernel，见 mc_attn_kv_major_enabled）。
#include "kernels/sm120/kernels.h"

#include <cuda_bf16.h>

#include "runtime/error.h"

namespace {

using bf16 = __nv_bfloat16;

constexpr float kScoreScale = 0.08838834764831845f; // 1/sqrt(128)

// blockDim 假设：恰好 4 个满员 warp（__shfl_xor_sync 的全 0xffffffff mask
// 要求 32 lane 全部活跃；launcher 以 kBlockThreads 启动）。
constexpr uint32_t kBlockThreads = 128; // 4 warp × 32 lane（legacy 路径用）
constexpr uint32_t kWarps = kBlockThreads / 32u;
constexpr uint32_t kDimsPerLane = KvLayout::kHeadDim / 32u; // 128/32 = 4
constexpr int kAttnPipe = 4;            // P5：legacy 长序列位置循环流水深度
static_assert(kBlockThreads == 4u * 32u,
              "full-warp shuffle mask requires exactly 4x32 threads");
static_assert(32u * kDimsPerLane == KvLayout::kHeadDim,
              "lane-strided dim ownership must cover head_dim exactly");

// ---- P5-C：phase0 旋转（SASS 锚：FMUL(sn,x2) + FFMA(cs,x,∓t)）----
// rope_kv_kernel 对 `x*cs -/+ x2*sn` 的 ptxas 收缩选择（cuobjdump 实测）：
//   t  = FMUL(sn, x2)          ；先独立舍入
//   out= FFMA(cs, x, -t / +t)  ；d<64 取 -t，d≥64 取 +t
// 之后 F2FP.BF16（RNE）。此处显式 intrinsic 复刻 → 逐位一致。
__device__ __forceinline__ float rope_rot_anchor(float x, float x2, float cs,
                                                 float sn, bool neg) {
    const float t = __fmul_rn(sn, x2);
    return neg ? __fmaf_rn(cs, x, -t) : __fmaf_rn(cs, x, t);
}
// rope 存储边界：fp32 → bf16（RNE）→ 再升 fp32 供点积（与「写显存再读」
// 的旧路径逐位等价）。
__device__ __forceinline__ float rope_round(float v) {
    return __bfloat162float(__float2bfloat16(v));
}

// ---- P9-Step2：cp.async 异步装载（sm_80+，sm_120 可用）----
// LDG→STS 直通依赖（旧 load_stage：每线程仅 2 在途 load，~700cyc DRAM
// 延迟暴露）改为 cp.async.cg 16B：数据不经寄存器直达 shared，配合
// commit_group/wait_group 的双组流水（装载 s+1 ∥ 计算 s），每线程 4 × 16B
// 在途。对齐已核实：全局 page 32KB / slot 1KB / head 256B / v16·16B；
// smem 行 256B / v16·16B；两端恒 16B 对齐（cp.async.cg 16B 的硬要求）。
__device__ __forceinline__ void mc_cp_async16(void* smem_dst, const void* gmem_src) {
    const uint32_t s = (uint32_t)__cvta_generic_to_shared(smem_dst);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s),
                 "l"(gmem_src));
}
__device__ __forceinline__ void mc_cp_async_commit() {
    asm volatile("cp.async.commit_group;\n");
}
template <int N>
__device__ __forceinline__ void mc_cp_async_wait() { // 等到 ≤N 组未完成
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

// ---- v2.1：mbarrier 原语（sm_80+ 硬件同步原语；非 atomics，数据路径
//      确定性不依赖调度）----
__device__ __forceinline__ uint32_t mc_smem_u32(const void* p) {
    return (uint32_t)__cvta_generic_to_shared(p);
}
// 初始化：期待 count 次 arrive 后相位翻转（每线程一次 arrive → 256）。
__device__ __forceinline__ void mc_mbar_init(void* bar, uint32_t count) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;" ::"r"(mc_smem_u32(bar)),
                 "r"(count));
}
// 普通到达（消费完成声明）。sink 目的占位 —— 不读状态字。
__device__ __forceinline__ void mc_mbar_arrive(void* bar) {
    asm volatile("mbarrier.arrive.shared.b64 _, [%0];" ::"r"(mc_smem_u32(bar))
                 : "memory");
}
// cp.async 完成到达：本线程所有「先于本指令发射」的 cp.async 全部落地后
// 触发一次 arrive（部分 stage 无 copy 的线程立即到达 —— 语义合法）。
// 必须用 .noinc 变体：默认（无 .noinc）形式会在发射时把 mbarrier 的
// 期望到达数 +1（deferred arrive 再 -1）—— 与「init count=256 + 每线程
// 恰一次 arrive」的协议叠加会使期望数漂移到 512，相位永不完成 → 全块
// 挂死在 try_wait（首版实现即栽在此：GPU util 0%、host 忙等 sync）。
// .noinc 不增期望数，只做延迟到达 ✓。
__device__ __forceinline__ void mc_cp_async_mbar_arrive(void* bar) {
    asm volatile("cp.async.mbarrier.arrive.noinc.shared.b64 [%0];" ::"r"(
                     mc_smem_u32(bar))
                 : "memory");
}
// 自旋等相位（try_wait.parity 立即返回谓词；硬件翻相即过，无到达偏斜。
// __nanosleep 退避降低自旋对发射端口的挤占）。
__device__ __forceinline__ void mc_mbar_wait_parity(void* bar, uint32_t parity) {
    uint32_t done = 0;
    while (!done) {
        asm volatile(
            "{ .reg .pred p; mbarrier.try_wait.parity.shared.b64 p, [%1], %2; "
            "selp.u32 %0, 1, 0, p; }"
            : "=r"(done)
            : "r"(mc_smem_u32(bar)), "r"(parity)
            : "memory");
        if (!done) __nanosleep(64);
    }
}

// bf16×2 → fp32 解包（P9-Step2 LDS.64 向量化读的伴生）：bits<<16 / &掩码
// 与 __bfloat162float 逐位等价（bf16 即 fp32 高 16 位）。
__device__ __forceinline__ float bf16lo(uint32_t w) {
    return __uint_as_float(w << 16);
}
__device__ __forceinline__ float bf16hi(uint32_t w) {
    return __uint_as_float(w & 0xffff0000u);
}

// ---- P5-B：kv-major chunk kernel 常量（v1/v2 共用）----
constexpr uint32_t kKvBlockThreads = 8u * 32u;  // 256 = 8 warp（warp ↔ q head）
constexpr uint32_t kQPerKv = 8;                  // GQA 16 q / 2 kv
constexpr uint32_t kKvStage = 16;                // 双缓冲 stage（pos 数）
constexpr uint32_t kKvIlp = 8;                   // ILP 位置批大小
constexpr uint32_t kKvSmemBytes = 2u * 2u * kKvStage * 128u * 2u; // 16KB
static_assert(kKvBlockThreads == kQPerKv * 32u, "8 full warps");
static_assert(kKvSmemBytes <= 48u * 1024u, "no opt-in dynamic smem path");

// ---- P9-Step2：kv-major chunk kernel v1（旧实现，保留为对拍基准；生产
//      路径改走下方 v2）。旧线程映射 lane l ↔ dims {l, l+32, l+64, l+96} +
//      LDG→STS 双缓冲。指令账本（ora-1）：K/V 标量 LDS.U16+CVT 占 51%
//     （stride 64B 无法向量化）、accurate expf 37%、装载仅 2 在途 load
//      暴露 ~700cyc DRAM 延迟 → 有效带宽 ~380GB/s。数值与 v2 等价
//     （同为 fp32 online softmax 的重分组，舍入路径不同 → relL2 ≤ 1e-6）。----
__global__ void attn_kv_kernel_v1(
    const uint16_t* __restrict__ q_rows, const uint32_t* __restrict__ d_base_pos,
    uint32_t layer, const uint32_t* __restrict__ page_table,
    uint16_t* __restrict__ kv_pool, /* phase0 持久化写当前位置 K/V */
    KvLayout layout, float* __restrict__ partials, uint32_t max_chunks,
    uint16_t* __restrict__ out /* [2048] 单 token 行 */,
    const float* __restrict__ inv_freq /* null = 无 phase0 */,
    uint32_t chunk_pos /* P8-A2：会话 chunk 位置数（建期档位表选定）*/) {
    const uint32_t kvh = blockIdx.x;    // 0..1
    const uint32_t chunk = blockIdx.y;  // 0..max_chunks-1（grid 建期固定）
    MC_PDL_TRIGGER(); // 入口放行后继（后继消费本 kernel 输出前必 wait）
    MC_PDL_WAIT();    // q/qkv 行由前驱 GEMV 写出（phase0 时含 k/v 列）
    const uint32_t limit = *d_base_pos; // decode 单 token：causal 上界 = 自身位置
    const uint32_t start = chunk * chunk_pos;
    if (start > limit) return; // 空块：超出当前序列（合法，无副作用）
    const uint32_t end = min(start + chunk_pos - 1u, limit);
    const uint32_t tid = threadIdx.x;   // 0..255
    const uint32_t warp = tid >> 5;     // 0..7 ↔ q head
    const uint32_t lane_dim = tid & 31u;
    const uint32_t qh = kvh * kQPerKv + warp;
    const bool ph0 = inv_freq != nullptr;
    const bool direct = limit < chunk_pos; // 单 chunk：直写 out

    // ---- q 预载（dim = lane + 32r，同旧）；phase0：寄存器内旋转 ----
    // 伙伴对：(r, r^2)，pair = r<2 ? lane : lane+32（见文件头注释）。
    float qv[kDimsPerLane];
    float cs0, sn0, cs1, sn1; // pair lane / lane+32 的 cos/sin（phase0 用）
    if (ph0) {
        const float f0 = __fmul_rn((float)limit, inv_freq[lane_dim]);
        const float f1 = __fmul_rn((float)limit, inv_freq[lane_dim + 32u]);
        cs0 = cosf(f0);
        sn0 = sinf(f0);
        cs1 = cosf(f1);
        sn1 = sinf(f1);
        float x[kDimsPerLane];
#pragma unroll
        for (uint32_t r = 0; r < kDimsPerLane; ++r)
            x[r] = __bfloat162float(reinterpret_cast<const bf16&>(
                q_rows[qh * 128u + lane_dim + 32u * r]));
        qv[0] = rope_round(rope_rot_anchor(x[0], x[2], cs0, sn0, true));
        qv[1] = rope_round(rope_rot_anchor(x[1], x[3], cs1, sn1, true));
        qv[2] = rope_round(rope_rot_anchor(x[2], x[0], cs0, sn0, false));
        qv[3] = rope_round(rope_rot_anchor(x[3], x[1], cs1, sn1, false));
    } else {
#pragma unroll
        for (uint32_t r = 0; r < kDimsPerLane; ++r)
            qv[r] = __bfloat162float(reinterpret_cast<const bf16&>(
                q_rows[qh * 128u + lane_dim + 32u * r]));
    }

    // ---- phase0：当前位置 k 旋转 / v 直通（寄存器）+ 持久化写池 ----
    // 每个 warp 自用；warp0 唯一负责把本位置的 K/V 写入池（后续 step 读）。
    float kcur[kDimsPerLane], vcur[kDimsPerLane];
    if (ph0) {
        const uint32_t kcol = 2048u + kvh * 128u; // qkv 行内 k 列基址
        const uint32_t vcol = 2304u + kvh * 128u; // v 列基址
        uint16_t vraw[kDimsPerLane];
        float xk[kDimsPerLane];
#pragma unroll
        for (uint32_t r = 0; r < kDimsPerLane; ++r) {
            xk[r] = __bfloat162float(reinterpret_cast<const bf16&>(
                q_rows[kcol + lane_dim + 32u * r]));
            vraw[r] = q_rows[vcol + lane_dim + 32u * r];
            vcur[r] = __bfloat162float(reinterpret_cast<const bf16&>(vraw[r]));
        }
        kcur[0] = rope_round(rope_rot_anchor(xk[0], xk[2], cs0, sn0, true));
        kcur[1] = rope_round(rope_rot_anchor(xk[1], xk[3], cs1, sn1, true));
        kcur[2] = rope_round(rope_rot_anchor(xk[2], xk[0], cs0, sn0, false));
        kcur[3] = rope_round(rope_rot_anchor(xk[3], xk[1], cs1, sn1, false));
        if (warp == 0u) { // 唯一写者（本 block 即 limit 所在 chunk）
            const uint32_t page = page_table[layout.page_index(limit)];
            const uint32_t slot = layout.slot_index(limit);
#pragma unroll
            for (uint32_t r = 0; r < kDimsPerLane; ++r) {
                const uint32_t e = lane_dim + 32u * r;
                kv_pool[layout.elem_offset(layer, page, slot, /*K*/ 0u, kvh, e)] =
                    __bfloat16_as_ushort(__float2bfloat16(kcur[r]));
                kv_pool[layout.elem_offset(layer, page, slot, /*V*/ 1u, kvh, e)] =
                    vraw[r];
            }
        }
    }

    // ---- 位置主循环：双缓冲 16-pos stage（装载 t+1 ∥ 计算 t）----
    extern __shared__ uint16_t sKV[]; // [2 buf][2 half(K,V)][kKvStage][128]
    uint16_t* buf[2] = {sKV, sKV + 2u * kKvStage * 128u};

    float m = -INFINITY, l = 0.f;
    float acc[kDimsPerLane];
#pragma unroll
    for (uint32_t r = 0; r < kDimsPerLane; ++r) acc[r] = 0.f;

    // 协作装载：每 pos 32 个 uint4（K 16 + V 16），256 线程各 2 个（满
    // stage）。phase0 时 limit 位照装（stale，计算走寄存器支路）。
    auto load_stage = [&](uint32_t s0, int b) {
        const uint32_t sEnd = min(s0 + kKvStage - 1u, end);
        const uint32_t n = sEnd - s0 + 1u;
        for (uint32_t i = tid; i < n * 32u; i += kKvBlockThreads) {
            const uint32_t sp = i >> 5;  // stage 内位置
            const uint32_t w4 = i & 31u; // pos 内第几个 uint4
            const uint32_t pos = s0 + sp;
            const uint32_t page = page_table[layout.page_index(pos)];
            const uint32_t slot = layout.slot_index(pos);
            const uint32_t half = w4 >> 4; // 0=K, 1=V
            const uint32_t v16 = w4 & 15u; // 半边内第几个 uint4（8 dim）
            reinterpret_cast<uint4*>(buf[b] + (half * kKvStage + sp) * 128u)[v16] =
                *reinterpret_cast<const uint4*>(
                    kv_pool +
                    layout.elem_offset(layer, page, slot, half, kvh, v16 * 8u));
        }
    };

    // 计算 stage：8-pos ILP 批（独立 dot 链 → 批级 softmax 更新）+ 串行尾。
    // 当前位（j == limit && ph0）逐成员走寄存器 k/v（与池值逐位同）。
    auto compute_stage = [&](uint32_t s0, int b) {
        const uint32_t sEnd = min(s0 + kKvStage - 1u, end);
        const uint16_t* sK = buf[b];
        const uint16_t* sV = buf[b] + kKvStage * 128u;
        uint32_t j = s0;
        for (; j + kKvIlp <= sEnd + 1u; j += kKvIlp) {
            float dot[kKvIlp];
#pragma unroll
            for (uint32_t q = 0; q < kKvIlp; ++q) dot[q] = 0.f;
#pragma unroll
            for (uint32_t q = 0; q < kKvIlp; ++q) {
                const uint32_t sp = j - s0 + q;
                if (ph0 && (j + q) == limit) {
#pragma unroll
                    for (uint32_t r = 0; r < kDimsPerLane; ++r)
                        dot[q] = fmaf(qv[r], kcur[r], dot[q]);
                } else {
#pragma unroll
                    for (uint32_t r = 0; r < kDimsPerLane; ++r)
                        dot[q] = fmaf(qv[r],
                                      __bfloat162float(reinterpret_cast<const bf16&>(
                                          sK[sp * 128u + lane_dim + 32u * r])),
                                      dot[q]);
                }
            }
#pragma unroll
            for (uint32_t q = 0; q < kKvIlp; ++q)
#pragma unroll
                for (int sw = 16; sw > 0; sw >>= 1)
                    dot[q] += __shfl_xor_sync(0xffffffffu, dot[q], sw);
            float score[kKvIlp];
            float mn = m;
#pragma unroll
            for (uint32_t q = 0; q < kKvIlp; ++q) {
                score[q] = __fmul_rn(dot[q], kScoreScale);
                mn = fmaxf(mn, score[q]);
            }
            // P9-Step1 热路径 fast exp（__expf，max-ulp 2 与 accurate 同级；
            // SASS 由 15-20 条 MUFU.EX2 链缩为 ~4 条 —— 指令账本：accurate
            // expf 曾占每 warp 每批 ~37% 指令）。merge_a/merge_b 的 expf 为
            // 冷路径，保持 accurate 不动。
            const float renew =
                (m == -INFINITY) ? 0.f : __expf(__fadd_rn(m, -mn));
            float ls = __fmul_rn(l, renew);
            float pw[kKvIlp];
#pragma unroll
            for (uint32_t q = 0; q < kKvIlp; ++q) {
                pw[q] = __expf(__fadd_rn(score[q], -mn));
                ls += pw[q];
            }
            l = ls;
#pragma unroll
            for (uint32_t r = 0; r < kDimsPerLane; ++r) {
                float a = __fmul_rn(acc[r], renew);
#pragma unroll
                for (uint32_t q = 0; q < kKvIlp; ++q) {
                    const uint32_t sp = j - s0 + q;
                    const float v =
                        (ph0 && (j + q) == limit)
                            ? vcur[r]
                            : __bfloat162float(reinterpret_cast<const bf16&>(
                                  sV[sp * 128u + lane_dim + 32u * r]));
                    a = fmaf(pw[q], v, a);
                }
                acc[r] = a;
            }
            m = mn;
        }
        // 串行尾（stage 内不足 kKvIlp 的余量；表达式与单流版逐条同锚）
        for (; j <= sEnd; ++j) {
            const uint32_t sp = j - s0;
            float kk[kDimsPerLane], vv[kDimsPerLane];
            if (ph0 && j == limit) {
#pragma unroll
                for (uint32_t r = 0; r < kDimsPerLane; ++r) {
                    kk[r] = kcur[r];
                    vv[r] = vcur[r];
                }
            } else {
#pragma unroll
                for (uint32_t r = 0; r < kDimsPerLane; ++r)
                    kk[r] = __bfloat162float(reinterpret_cast<const bf16&>(
                        sK[sp * 128u + lane_dim + 32u * r]));
#pragma unroll
                for (uint32_t r = 0; r < kDimsPerLane; ++r)
                    vv[r] = __bfloat162float(reinterpret_cast<const bf16&>(
                        sV[sp * 128u + lane_dim + 32u * r]));
            }
            float dot = 0.f;
#pragma unroll
            for (uint32_t r = 0; r < kDimsPerLane; ++r)
                dot = fmaf(qv[r], kk[r], dot);
#pragma unroll
            for (int sw = 16; sw > 0; sw >>= 1)
                dot += __shfl_xor_sync(0xffffffffu, dot, sw);
            const float score = __fmul_rn(dot, kScoreScale);
            const float mn = fmaxf(m, score);
            // P9-Step1 fast exp（串行尾，同批路径）
            const float pw = __expf(__fadd_rn(score, -mn));
            const float renew =
                (m == -INFINITY) ? 0.f : __expf(__fadd_rn(m, -mn));
            l = fmaf(l, renew, pw);
#pragma unroll
            for (uint32_t r = 0; r < kDimsPerLane; ++r)
                acc[r] = fmaf(pw, vv[r], __fmul_rn(acc[r], renew));
            m = mn;
        }
    };

    load_stage(start, 0);
    __syncthreads();
    uint32_t s0 = start;
    for (int b = 0; s0 <= end; b ^= 1, s0 += kKvStage) {
        if (s0 + kKvStage <= end) load_stage(s0 + kKvStage, b ^ 1); // 预取
        compute_stage(s0, b);
        __syncthreads(); // 本 buf 消费完才能被再装载
    }

    // ---- 输出：direct（单 chunk 恒等退化）或 partial 槽 ----
    if (direct) {
        // n_valid==1：merge 公式退化为 out=bf16(acc/l)（sc=e^0=1 精确）；
        // merge_p 对 n_valid==1 全块一致早退 → partial 无需写。
#pragma unroll
        for (uint32_t r = 0; r < kDimsPerLane; ++r)
            out[qh * 128u + lane_dim + 32u * r] =
                __bfloat16_as_ushort(__float2bfloat16(
                    (l > 0.f) ? __fdiv_rn(acc[r], l) : 0.f));
    } else {
        float* pw = partials +
                    ((size_t)qh * max_chunks + chunk) * kAttnPartialStride;
        if (lane_dim == 0u) {
            pw[0] = m;
            pw[1] = l;
        }
#pragma unroll
        for (uint32_t r = 0; r < kDimsPerLane; ++r)
            pw[2 + lane_dim + 32u * r] = acc[r];
    }
}

// ---- P9-Step2：kv-major chunk kernel v2（生产路径）----
// 相对 v1 的三处微结构改动（warp ↔ q head / block 256 / grid=(2,chunks) /
// kKvStage=16 / kKvIlp=8 / partials 契约 / limit 设备读取 / 空块早退全部
// 不变）：
//   1) 线程映射：lane l ∈ [0,32) 拥有连续 4 维 [4l, 4l+4)（旧 {l, l+32,
//      l+64, l+96}）→ K/V smem 读从每-pos 每-lane 4 条标量 LDS.U16
//     （stride 64B）变为 1 条 LDS.64（uint2 = 4×bf16 连续 8B），51% 的
//      标量装载指令被消除；解包 bf16lo/bf16hi（bits<<16 / &掩码，与
//      __bfloat162float 逐位等价）。
//   2) 装载改 cp.async.cg 16B 双组流水：issue(s+1) ∥ compute(s)，
//      每线程 4×16B 在途（旧 LDG→STS 直通仅 2，~700cyc DRAM 延迟暴露）。
//   3) RoPE phase0 蝴蝶路由：新映射下伙伴 (d, d±64) = lane l ↔ l^16 →
//      x2 = __shfl_xor_sync(x, 16)；cs/sn 表 block 协作一次性算入静态
//      shared sCS/sSN[64]（f = __fmul_rn((float)limit, inv_freq[t]) 与
//      v1 的 :157-158 逐位同式 —— 每维 cos/sin 值逐位不变）；旋转算术
//     （rope_rot_anchor 的 FMUL/FFMA 收缩 + rope_round）逐条不动，只改
//      数据来源路由 → phase0 输出/池写逐位同值。表放静态 smem（+512B，
//      不与 cp.async 的 stage 写竞争；occupancy 不变——绑定约束是
//      线程/寄存器而非 smem）。
// 数值：dot 的 lane 内 4 乘积累加顺序随映射改变（v1 {l,l+32,l+64,l+96}
// 升序 vs v2 [4l,4l+4) 升序），butterfly 树不变 → fp32 重分组，kernel_ops
// attention_decode_v2_vs_v1 锚 relL2 ≤ 1e-6 / bf16 ≤1ulp。
// ---- P9-Step2.1：kv-major chunk kernel v2/v2.1 模板（生产路径）----
// v2.1（kV21=true，默认）：把 v2 流水环的两次 __syncthreads（每 stage 逼
// 迫全 8 warp 同进同出 —— 每 SM 每层 ~195 次 stage-event，每次 ~0.25µs
// 暴露，叠加 issue 与访存互相突发）替换为 4 个 mbarrier 的生产者-消费者
// 协议；跨 stage 解锁：快 warp 算完 stage i 直接滚进 i+1（data[i+1] 就绪
// 即可），buf 回收门（free[b]，等最慢消费者）只在再装载路径上、不在计算
// 关键路径。compute_stage 指令序与每 warp 运算顺序一字不动 → 与 v2 逐位
// 一致（cp.async 内容/地址不变，timing 不影响数据）。
// 死锁审计（v2.1）：
//   - 循环边界 s0/end 为全块一致值（空块 warp 早退发生在 init 之前，且
//     整块一致）→ 无 warp 在环内退出，256 arrive 恒可凑齐；
//   - free 相位只由硬件在 256 次到达后推进，等待者（本 buf 下一轮发射）
//     的到达者全部位于 ≤ 当前 stage 的流水位 → 严格有序，无环；
//   - cp.async.mbarrier.arrive 的 all-prior 语义使 data[b] 相位可能保守
//     地多等更早 stage 的 copy —— 只延迟不阻塞（copy 相互独立）；
//   - warp0 池写（phase0）只写全局池，不涉及 mbarrier；
//   - 部分_stage 线程无 copy 时 cp.async.mbarrier.arrive 立即到达 ✓。
// v2（kV21=false，回退）：wait_group + 双 barrier 原流水（MC_ATTN_V21_OFF）。
// kSAnchor（H1 追加）：S 调试锚 —— compute_stage 算出的 score（dot×
// kScoreScale，softmax 前）写入 score_anchor scratch，供 kernel_ops 的
// attention_hmma_qk_anchor 对拍 HMMA QK 段。仅测试实例（<false, true>）编译
// 该路径；生产实例（<*, false>）经 if constexpr 完全消除 → 指令序与既有
// 生产二进制逐条不变。scratch 布局：[(chunk*16 + qh)*chunk_pos + 局部 pos]
// fp32（单 chunk 时即 [16 qh][chunk_pos]；v2 只计算 pos ≤ end，故有效前缀
// [0..end-start] 之外不写）。
template <bool kV21, bool kSAnchor = false>
__global__ void __launch_bounds__(kKvBlockThreads, 4) attn_kv_kernel_x(
    const uint16_t* __restrict__ q_rows, const uint32_t* __restrict__ d_base_pos,
    uint32_t layer, const uint32_t* __restrict__ page_table,
    uint16_t* __restrict__ kv_pool, /* phase0 持久化写当前位置 K/V */
    KvLayout layout, float* __restrict__ partials, uint32_t max_chunks,
    uint16_t* __restrict__ out /* [2048] 单 token 行 */,
    const float* __restrict__ inv_freq /* null = 无 phase0 */,
    uint32_t chunk_pos, /* P8-A2：会话 chunk 位置数（建期档位表选定） */
    float* __restrict__ score_anchor /* H1 S 锚（仅 kSAnchor 实例使用）*/) {
    const uint32_t kvh = blockIdx.x;    // 0..1
    const uint32_t chunk = blockIdx.y;  // 0..max_chunks-1（grid 建期固定）
    MC_PDL_TRIGGER(); // 入口放行后继（后继消费本 kernel 输出前必 wait）
    const uint32_t limit = *d_base_pos; // decode 单 token：causal 上界 = 自身位置
    const uint32_t start = chunk * chunk_pos;
    if (start > limit) return; // 空块：超出当前序列（合法，无副作用）
    const uint32_t end = min(start + chunk_pos - 1u, limit);
    const uint32_t tid = threadIdx.x;   // 0..255
    const uint32_t warp = tid >> 5;     // 0..7 ↔ q head
    const uint32_t lane = tid & 31u;
    const uint32_t d0 = lane * 4u;      // 本 lane 拥有连续 4 维 [d0, d0+4)
    const uint32_t qh = kvh * kQPerKv + warp;
    const bool ph0 = inv_freq != nullptr;
    const bool direct = limit < chunk_pos; // 单 chunk：直写 out

    // ---- v2.1：mbarrier 初始化（4 个：data[2]/free[2]，count=256 即每
    //      线程一次到达；thread0 初始化 + 一次 __syncthreads 使全块可见
    //     —— 每 kernel 摊销一次）。v2 实例化同样携带 32B（无害）。----
    __shared__ uint64_t mBar[4]; // [data0, data1, free0, free1]
    if constexpr (kV21) {
        if (tid == 0u) {
            mc_mbar_init(&mBar[0], kKvBlockThreads);
            mc_mbar_init(&mBar[1], kKvBlockThreads);
            mc_mbar_init(&mBar[2], kKvBlockThreads);
            mc_mbar_init(&mBar[3], kKvBlockThreads);
        }
        __syncthreads();
    }

    // ---- 位置主循环：cp.async 双缓冲 16-pos stage（装载 s+1 ∥ 计算 s）----
    extern __shared__ uint16_t sKV[]; // [2 buf][2 half(K,V)][kKvStage][128]
    uint16_t* buf[2] = {sKV, sKV + 2u * kKvStage * 128u};

    // 协作异步装载：每 pos 32 个 16B（K 16 + V 16），满 stage 时 256 线程
    // 各 2 条。寻址/页表查找与 v1 load_stage 逐条同式；只把 LDG→STS 换为
    // cp.async.cg（16B，绕过寄存器）。phase0 时 limit 位照装（stale，计算
    // 走寄存器支路）。
    auto issue_stage = [&](uint32_t s0, int b) {
        const uint32_t sEnd = min(s0 + kKvStage - 1u, end);
        const uint32_t n = sEnd - s0 + 1u;
        for (uint32_t i = tid; i < n * 32u; i += kKvBlockThreads) {
            const uint32_t sp = i >> 5;  // stage 内位置
            const uint32_t w4 = i & 31u; // pos 内第几个 16B
            const uint32_t pos = s0 + sp;
            const uint32_t page = page_table[layout.page_index(pos)];
            const uint32_t slot = layout.slot_index(pos);
            const uint32_t half = w4 >> 4; // 0=K, 1=V
            const uint32_t v16 = w4 & 15u; // 半边内第几个 16B（8 dim）
            mc_cp_async16(
                reinterpret_cast<uint4*>(buf[b] + (half * kKvStage + sp) * 128u) +
                    v16,
                kv_pool + layout.elem_offset(layer, page, slot, half, kvh,
                                             v16 * 8u));
        }
    };

    // ---- PDL 预取前移（v2.0-P1）：stage0 的 cp.async 在 WAIT 之前发射 ----
    // 前驱（qkv GEMV）的放行等待与 stage0 装载无关，两者重叠 ~整个前驱尾
    // 部。安全性（ora-1 三轮核实）：
    //   - limit 读自 d_base_pos：由 graph 外 H2D / 上一步 decode_advance 所
    //     写，非直接前驱（qkv GEMV）输出 —— PDL wait 只对直接前驱的写排序，
    //     WAIT 前读合法；
    //   - page_table 由 prefill/页增长（graph 外小拷贝）写，同理安全；
    //   - stage0（v2.1 连同 stage1）读的 K/V 池行为上一步及更早所写（每步
    //     流序分隔）；唯一同步内竞态是 limit 位（本 kernel phase0 warp0 自
    //     己在写）——读到 stale/新值/撕裂值均无所谓：计算对 j==limit 恒走
    //     寄存器替换支路，smem 值从不使用（与 issue_stage 注释同一论证）。
    // 完成同步机制按实例化分叉：v2 = commit_group（环内 wait_group 消费）；
    // v2.1 = cp.async.mbarrier.arrive(data[b])（环内 try_wait.parity 消费，
    // stage1 一并前移发射 —— 同一安全性论证）。
    if constexpr (kV21) {
        issue_stage(start, 0);
        mc_cp_async_mbar_arrive(&mBar[0]);
        if (start + kKvStage <= end) { // ≥2 stage：stage1 也前移
            issue_stage(start + kKvStage, 1);
            mc_cp_async_mbar_arrive(&mBar[1]);
        }
    } else {
        issue_stage(start, 0);
        mc_cp_async_commit();
    }

    MC_PDL_WAIT(); // q/qkv 行由前驱 GEMV 写出（phase0 时含 k/v 列）

    // ---- phase0 表：block 协作一次性算 64 个频率的 cos/sin（静态 smem；
    //      值与 v1 逐位同 —— 同式 __fmul_rn + cosf/sinf）----
    __shared__ float sCS[64], sSN[64]; // +512B，见头部注释（不复用 stage 头
                                       // 是为避开 cp.async 覆盖竞争）
    if (ph0) {
        if (tid < 64u) {
            const float f = __fmul_rn((float)limit, inv_freq[tid]);
            sCS[tid] = cosf(f);
            sSN[tid] = sinf(f);
        }
        __syncthreads(); // 表写 → 全 block 可读
    }

    // ---- q 预载（连续 4 维 [d0, d0+4)）；phase0：寄存器内旋转 ----
    // 伙伴 dim = d±64 → lane l↔l^16（同 r 槽位）；neg = (d < 64)。
    float qv[kDimsPerLane];
    if (ph0) {
        float x[kDimsPerLane];
#pragma unroll
        for (uint32_t r = 0; r < kDimsPerLane; ++r)
            x[r] = __bfloat162float(reinterpret_cast<const bf16&>(
                q_rows[qh * 128u + d0 + r]));
#pragma unroll
        for (uint32_t r = 0; r < kDimsPerLane; ++r) {
            const uint32_t e = d0 + r;                    // 本维绝对下标
            const float x2 = __shfl_xor_sync(0xffffffffu, x[r], 16);
            qv[r] = rope_round(
                rope_rot_anchor(x[r], x2, sCS[e & 63u], sSN[e & 63u], e < 64u));
        }
    } else {
#pragma unroll
        for (uint32_t r = 0; r < kDimsPerLane; ++r)
            qv[r] = __bfloat162float(reinterpret_cast<const bf16&>(
                q_rows[qh * 128u + d0 + r]));
    }

    // ---- phase0：当前位置 k 旋转 / v 直通（寄存器）+ 持久化写池 ----
    // 每个 warp 自用；warp0 唯一负责把本位置的 K/V 写入池（后续 step 读）。
    // 写者维度改 [d0, d0+4)（全局布局不变，值逐位同 v1）。
    float kcur[kDimsPerLane], vcur[kDimsPerLane];
    if (ph0) {
        const uint32_t kcol = 2048u + kvh * 128u; // qkv 行内 k 列基址
        const uint32_t vcol = 2304u + kvh * 128u; // v 列基址
        uint16_t vraw[kDimsPerLane];
        float xk[kDimsPerLane];
#pragma unroll
        for (uint32_t r = 0; r < kDimsPerLane; ++r) {
            xk[r] = __bfloat162float(reinterpret_cast<const bf16&>(
                q_rows[kcol + d0 + r]));
            vraw[r] = q_rows[vcol + d0 + r];
            vcur[r] = __bfloat162float(reinterpret_cast<const bf16&>(vraw[r]));
        }
#pragma unroll
        for (uint32_t r = 0; r < kDimsPerLane; ++r) {
            const uint32_t e = d0 + r;
            const float x2 = __shfl_xor_sync(0xffffffffu, xk[r], 16);
            kcur[r] = rope_round(
                rope_rot_anchor(xk[r], x2, sCS[e & 63u], sSN[e & 63u], e < 64u));
        }
        if (warp == 0u) { // 唯一写者（本 block 即 limit 所在 chunk）
            const uint32_t page = page_table[layout.page_index(limit)];
            const uint32_t slot = layout.slot_index(limit);
#pragma unroll
            for (uint32_t r = 0; r < kDimsPerLane; ++r) {
                const uint32_t e = d0 + r;
                kv_pool[layout.elem_offset(layer, page, slot, /*K*/ 0u, kvh, e)] =
                    __bfloat16_as_ushort(__float2bfloat16(kcur[r]));
                kv_pool[layout.elem_offset(layer, page, slot, /*V*/ 1u, kvh, e)] =
                    vraw[r];
            }
        }
    }

    // ---- 位置主循环状态（sKV/buf/issue_stage 已在 WAIT 前声明/定义）----
    float m = -INFINITY, l = 0.f;
    float acc[kDimsPerLane];
#pragma unroll
    for (uint32_t r = 0; r < kDimsPerLane; ++r) acc[r] = 0.f;

    // 计算 stage：8-pos ILP 批 + 串行尾。K/V smem 读向量化：每 pos 每lane
    // 1 条 LDS.64（uint2 = 连续 4 维），解包 bf16lo/bf16hi。当前位
    //（j == limit && ph0）逐成员走寄存器 k/v（与池值逐位同）。
    auto compute_stage = [&](uint32_t s0, int b) {
        const uint32_t sEnd = min(s0 + kKvStage - 1u, end);
        const uint16_t* sK = buf[b];
        const uint16_t* sV = buf[b] + kKvStage * 128u;
        uint32_t j = s0;
        for (; j + kKvIlp <= sEnd + 1u; j += kKvIlp) {
            float dot[kKvIlp];
#pragma unroll
            for (uint32_t q = 0; q < kKvIlp; ++q) dot[q] = 0.f;
#pragma unroll
            for (uint32_t q = 0; q < kKvIlp; ++q) {
                const uint32_t sp = j - s0 + q;
                if (ph0 && (j + q) == limit) {
#pragma unroll
                    for (uint32_t r = 0; r < kDimsPerLane; ++r)
                        dot[q] = fmaf(qv[r], kcur[r], dot[q]);
                } else {
                    const uint2 u =
                        *reinterpret_cast<const uint2*>(sK + sp * 128u + d0);
                    dot[q] = fmaf(qv[0], bf16lo(u.x), dot[q]);
                    dot[q] = fmaf(qv[1], bf16hi(u.x), dot[q]);
                    dot[q] = fmaf(qv[2], bf16lo(u.y), dot[q]);
                    dot[q] = fmaf(qv[3], bf16hi(u.y), dot[q]);
                }
            }
#pragma unroll
            for (uint32_t q = 0; q < kKvIlp; ++q)
#pragma unroll
                for (int sw = 16; sw > 0; sw >>= 1)
                    dot[q] += __shfl_xor_sync(0xffffffffu, dot[q], sw);
            float score[kKvIlp];
            float mn = m;
#pragma unroll
            for (uint32_t q = 0; q < kKvIlp; ++q) {
                score[q] = __fmul_rn(dot[q], kScoreScale);
                mn = fmaxf(mn, score[q]);
            }
            // H1 S 锚（仅测试实例）：butterfly 归约后全 lane 同值，lane0 唯一
            // 写者；写序 q 升序、逐 chunk/head 槽唯一 → 无竞态、确定性。
            if constexpr (kSAnchor) {
                if (lane == 0u)
#pragma unroll
                    for (uint32_t q = 0; q < kKvIlp; ++q)
                        score_anchor[((size_t)chunk * 16u + qh) * chunk_pos +
                                     (j + q - start)] = score[q];
            }
            // P9-Step1 fast exp（与 v1 一致）
            const float renew =
                (m == -INFINITY) ? 0.f : __expf(__fadd_rn(m, -mn));
            float ls = __fmul_rn(l, renew);
            float pw[kKvIlp];
#pragma unroll
            for (uint32_t q = 0; q < kKvIlp; ++q) {
                pw[q] = __expf(__fadd_rn(score[q], -mn));
                ls += pw[q];
            }
            l = ls;
            // V 累加：先统一 renew（per-r 一次 FMUL），再 q 升序 fmaf 链 ——
            // 每 r 的运算序列与 v1 完全同锚（mult → q 升序 FFMA 链）。
#pragma unroll
            for (uint32_t r = 0; r < kDimsPerLane; ++r)
                acc[r] = __fmul_rn(acc[r], renew);
#pragma unroll
            for (uint32_t q = 0; q < kKvIlp; ++q) {
                const uint32_t sp = j - s0 + q;
                float v[kDimsPerLane];
                if (ph0 && (j + q) == limit) {
#pragma unroll
                    for (uint32_t r = 0; r < kDimsPerLane; ++r) v[r] = vcur[r];
                } else {
                    const uint2 uv =
                        *reinterpret_cast<const uint2*>(sV + sp * 128u + d0);
                    v[0] = bf16lo(uv.x);
                    v[1] = bf16hi(uv.x);
                    v[2] = bf16lo(uv.y);
                    v[3] = bf16hi(uv.y);
                }
#pragma unroll
                for (uint32_t r = 0; r < kDimsPerLane; ++r)
                    acc[r] = fmaf(pw[q], v[r], acc[r]);
            }
            m = mn;
        }
        // 串行尾（stage 内不足 kKvIlp 的余量；表达式与批路径同锚）
        for (; j <= sEnd; ++j) {
            const uint32_t sp = j - s0;
            float kk[kDimsPerLane], vv[kDimsPerLane];
            if (ph0 && j == limit) {
#pragma unroll
                for (uint32_t r = 0; r < kDimsPerLane; ++r) {
                    kk[r] = kcur[r];
                    vv[r] = vcur[r];
                }
            } else {
                const uint2 uk =
                    *reinterpret_cast<const uint2*>(sK + sp * 128u + d0);
                kk[0] = bf16lo(uk.x);
                kk[1] = bf16hi(uk.x);
                kk[2] = bf16lo(uk.y);
                kk[3] = bf16hi(uk.y);
                const uint2 uv =
                    *reinterpret_cast<const uint2*>(sV + sp * 128u + d0);
                vv[0] = bf16lo(uv.x);
                vv[1] = bf16hi(uv.x);
                vv[2] = bf16lo(uv.y);
                vv[3] = bf16hi(uv.y);
            }
            float dot = 0.f;
#pragma unroll
            for (uint32_t r = 0; r < kDimsPerLane; ++r)
                dot = fmaf(qv[r], kk[r], dot);
#pragma unroll
            for (int sw = 16; sw > 0; sw >>= 1)
                dot += __shfl_xor_sync(0xffffffffu, dot, sw);
            const float score = __fmul_rn(dot, kScoreScale);
            // H1 S 锚（串行尾，仅测试实例；j ≤ sEnd ≤ end 恒有效，无 mask）
            if constexpr (kSAnchor) {
                if (lane == 0u)
                    score_anchor[((size_t)chunk * 16u + qh) * chunk_pos +
                                 (j - start)] = score;
            }
            const float mn = fmaxf(m, score);
            // P9-Step1 fast exp（串行尾，同批路径）
            const float pw = __expf(__fadd_rn(score, -mn));
            const float renew =
                (m == -INFINITY) ? 0.f : __expf(__fadd_rn(m, -mn));
            l = fmaf(l, renew, pw);
#pragma unroll
            for (uint32_t r = 0; r < kDimsPerLane; ++r)
                acc[r] = fmaf(pw, vv[r], __fmul_rn(acc[r], renew));
            m = mn;
        }
    };

    if constexpr (kV21) {
        // ---- v2.1：mbarrier 无锁步流水（两次 __syncthreads 全部消失）----
        // 每 warp 自由滚动：data[b] 相位就绪即计算（不等兄弟 warp）；
        // compute_stage 一字不动；buf 回收门 free[b] 只挡再装载。
        // parity(i) = ((s0 - start) >> 5) & 1 —— 零持久循环寄存器
        //（b 与 parity 均由 s0 现算；b = ((s0-start)>>4)&1）。
        for (uint32_t s0 = start; s0 <= end; s0 += kKvStage) {
            const int b = (int)((s0 - start) >> 4) & 1;
            const uint32_t par = ((s0 - start) >> 5) & 1u;
            mc_mbar_wait_parity(&mBar[b], par); // stage i 数据就绪
            compute_stage(s0, b);
            mc_mbar_arrive(&mBar[2 + b]); // free[b]：消费完成声明
            if (s0 + 2u * kKvStage <= end) { // stage i+2 回收本 buf
                mc_mbar_wait_parity(&mBar[2 + b], par); // 最慢消费者门
                issue_stage(s0 + 2u * kKvStage, b);
                mc_cp_async_mbar_arrive(&mBar[b]);
            }
        }
    } else {
        // cp.async 双组流水（v2）：迭代 i 先发射 stage i+1（组 G_{i+1}），再等
        // 待 G_i 完成（wait 1）→ 计算 buf_i。组间在途 = 当前 + 下一（每线程
        // 4×16B）。wait_group 语义：等到 ≤N 个最早提交的组未完成 → G_i 落地。
        // G_0（stage0）已在上方 WAIT 前发射 + commit（PDL 预取前移）。
        // 两次 __syncthreads：wait 后（cp.async 写对全 block 可见）与计算后
        //（buf 被再装载前消费完）。
        uint32_t s0 = start;
        for (int b = 0; s0 <= end; b ^= 1, s0 += kKvStage) {
            const bool has_next = (s0 + kKvStage <= end);
            if (has_next) {
                issue_stage(s0 + kKvStage, b ^ 1); // 预取下一 stage
                mc_cp_async_commit();
            }
            if (has_next)
                mc_cp_async_wait<1>();
            else
                mc_cp_async_wait<0>();
            __syncthreads();
            compute_stage(s0, b);
            __syncthreads();
        }
    }

    // ---- 输出：direct（单 chunk 恒等退化）或 partial 槽（写者维度
    //      [d0, d0+4)；布局/语义与 v1 相同）----
    if (direct) {
        // n_valid==1：merge 公式退化为 out=bf16(acc/l)（sc=e^0=1 精确）。
#pragma unroll
        for (uint32_t r = 0; r < kDimsPerLane; ++r)
            out[qh * 128u + d0 + r] =
                __bfloat16_as_ushort(__float2bfloat16(
                    (l > 0.f) ? __fdiv_rn(acc[r], l) : 0.f));
    } else {
        float* pw = partials +
                    ((size_t)qh * max_chunks + chunk) * kAttnPartialStride;
        if (lane == 0u) {
            pw[0] = m;
            pw[1] = l;
        }
#pragma unroll
        for (uint32_t r = 0; r < kDimsPerLane; ++r)
            pw[2 + d0 + r] = acc[r];
    }
}

// ---- H1+H2：HMMA tensor-core 测试 kernel（attn_kv_kernel_h，非生产）----
// 生产路径零改动：本 kernel 仅由 launch_attn_kv_h_test（kernel_ops 的
// attention_hmma_qk_anchor / attention_hmma_vs_v2 对拍）调用，不接入
// forward。骨架照抄 attn_kv_kernel_x 的 v2 实例（kV21=false）：grid=
// (2 kv head, max_chunks)、block=256=8 warp、chunk_pos 档位、16-pos 双缓冲
// stage、cp.async.cg 16B、PDL TRIGGER/WAIT、limit 设备读取、空块早退、
// 协作装载/页表查找 —— 全同 v2。
// H1 范围 = QK 段（mma.m16n8k16 bf16）+ S 跨 warp 归约 + score 调试锚；
// H2 补全主循环 = SOFT + P→A 直通 + PV（mma）+ partials/direct 写回 →
// 输出契约与 v2 完全同构（partials [16][max_chunks][130] / direct out）。
//
// 与 v2 的结构差异：
//  1) K/V stage 拆两个独立数组 [2 buf][16 pos][136] bf16（行距 272B）。
//     272 = 17×16B：行基址恒 16B 对齐（cp.async.cg 16B / ldmatrix 16B 行
//     取数的硬要求）；bank 论证：bank 环 = 32 bank × 4B = 128B，272 mod 128
//     = 16 → 相邻 pos 行旋转 +4 bank，ldmatrix 的同一 8×8 矩阵 8 行 × 16B
//     恰好铺满 32 bank 各一次 → 0 conflict（若行距取 128B，8 行全砸同一
//     4-bank 组 → 8-way conflict）。issue_stage 寻址随之改 sp*136。
//  2) sQT 静态 smem [16 行][136] bf16（同 pitch 论证）：行 0-7 = 本 kvh 的
//     8 个 q head 原始 bf16 直铺（ph0=false，与 v2 的 qv 同源逐位一致），
//     行 8-15 显式清零 pad（mma M=16 需凑满 16 行；pad 行 A 恒 +0 → 其 S
//     行恒 0，构造保证，不参与锚写/写回）。kernel 头 256 线程纯拷贝 pass +
//     一次 __syncthreads（q 读在 PDL WAIT 之后，同 v2）。
//  3) QK 段：A fragment = Q 切片（warp w 负责 dims [16w,16w+16)），kernel
//     头每 warp 一条 ldmatrix.x4 载入、常驻寄存器（象限序 (q00,q10,q01,
//     q11) → 四寄存器的 mma A 布局：R0=(行g,列2c)、R1=(行g+8,列2c)、
//     R2=(行g,列2c+8)、R3=(行g+8,列2c+8)，lane 8-15/16-23 分别给行 8-15
//     （清零 pad）/+16B 列的行址 —— wmma 位编码 dump 锚定，见改动 3 注）。
//     B fragment = K 的 pos 行：smem 行主序 [pos][dim] 恰为 B 的 [n][k]
//     （n=pos 8 个，k=dim 16 个），非 trans 的 ldmatrix.x2 直配（lane
//     0-7/8-15 给 pos 行 × dim 前后 8 的行址；此侧布局与 wmma.load.b
//     一致，实测无误）。每 stage 每 warp 2 条 mma（nt=0/1 两个 8-pos
//     tile），D 从 0 起 → S_partial[M=16 head 行][N=8 pos]。
//  4) RED：sS 静态 smem [8 warp][16 head][16 pos] f32（8KB）。每 stage 一次
//     __syncthreads 后，全 256 线程按 mma D-fragment 归属各归约自己的
//     score：head = lane>>2（本块 8 个真实 head；pad 行 head+8 恒 0，
//     H2 直接零填充 A pad 行、不归约不 SOFT —— 见 H2-1），pos =
//     2*(lane&3)+{0,1}+8*nt 共 4 个；每位置固定 w=0..7 升序 LDS+FADD（固定
//     顺序、无 atomics → 确定性）。
//  5) S 调试锚（H1 语义保留）：score = __fmul_rn(S_sum, kScoreScale)；尾
//     stage 的 pos > end 置 -INF 后由 warp0 写 score_scratch（非 null 时）：
//     [(chunk*16 + qh)*chunk_pos + 局部 pos] fp32（单 chunk 时即 [16 head]
//     [chunk_pos]；有效前缀 [0..end-start]）。锚写 (head,pos) 唯一 → 无竞态。
//
// ---- H2 追加段（主循环闭合）----
// H2-1 SOFT（8 warp 冗余计算，零通信，确定性）：每线程持 head g 的 4 个
//     score（pos 2c+{0,1} 两 tile）；mn = max(自身 4 项) → shfl_xor{1,2}
//     合并 quad（4 lane 共享同 head，覆盖 16 pos）→ 再并入 m_old；renew =
//     (m_old==-INF)?0:__expf(m_old-mn)；pw_i = __expf(score_i-mn)（fp32）；
//     l = l*renew + Σpw（自身 4 项固定序相加后 shfl_xor{1,2} 归约 —— quad
//     内 4 线程值恒同）；m ← mn。m/l 每 head 一份（跨 stage 存活）。
//     pad head（g+8）不 SOFT：其 score 恒 0 的 softmax 无消费者，A pad 行
//     直接零填充（比"算垃圾再丢弃"少一半 RED/SOFT 工作，且无垃圾值依赖）。
// H2-2 P→A 直通（H1 实测寄存器序）：a0 = pack(pw0,pw1)（head g，pos 2c,
//     2c+1）；a2 = pack(pw2,pw3)（pos 8+2c,8+2c+1）；a1/a3 = 0（pad 行）。
//     pack = __floats2bfloat162_rn（RNE，与 v2 的 P fp32 语义差异主源）。
//     masked pos 的 pw = exp(-INF-mn) = 0 → 天然不进 PV。
// H2-3 PV：acc[2 tile][4 reg]（D-fragment：c{0,1} = head g 的 dim
//     {2c,2c+1}+8nt、c{2,3} = pad 行 dim，恒 0），跨 stage 存活；每 stage
//     acc c{0,1} ×= renew（c{2,3} 死数据不缩放）；B = ldmatrix.x2.trans 载
//     V（V smem [pos][dim] 行主序，trans 恰把 pos 行转成 B 的 [n=dim][k=
//     pos] 列序：lane 0-7/8-15 给 pos 行 0-7/8-15 × dim [16w+8nt,+8) 的
//     行址）；每 warp 2 条 mma（D=acc 累加，asm 约束保留 H1 的 "+&f"
//     early-clobber 防御——H1 曾实测 ptxas 把 HMMA 目标压到操作数寄存器）。
// H2-4 写回（chunk 循环后，契约同 v2）：qh = kvh*8 + g；
//     acc：全 8 warp，pw_base[2 + 16w + 8nt + 2c + {0,1}] ← acc[nt] c{0,1}
//     （fp32 4×STG；warp 间 dim 分区无重叠；pad c{2,3} 不写）；
//     m/l：SOFT 后 quad 内 4 线程恒同值 → warp0 的 lane%4==0（lane=4g）
//     线程写 pw_base[0]=m、[1]=l；direct 支路（limit < chunk_pos）仿 v2：
//     out[qh*128 + 16w + 8nt + 2c + {0,1}] = bf16(l>0 ? acc/l : 0)。
//
// 数值（H2 vs v2 的 relL2 预期，判读双峰）：S/锚段同 H1（~1e-7，同数不
// 同序）；SOFT 的 m/l 为 fp32 重分组（~1e-7）；acc 的差异由 P→bf16 量化
// 主导（mma A 操作数仅 bf16，v2 为 fp32 pw）：每 pw 相对误差 ~2^-9 → acc
// 元素级相对差 ~2e-3（随机符号项和之比）→ partials relL2 预期 ~1e-3 量级
// （任务书"1e-3 = 预期舍入"档）；1e-1 档 = 布线 bug。kernel_ops 的
// attention_hmma_vs_v2 以 bf16-P CPU 参考做布线证明（预期 ~1e-6）。
constexpr uint32_t kHmaPitch = 136u; // K/V/Q smem 行距（bf16 元素）= 272B
constexpr uint32_t kHmaSmemBytes = 2u * 2u * kKvStage * kHmaPitch * 2u; // 17408B
static_assert(kHmaSmemBytes <= 48u * 1024u, "no opt-in dynamic smem path");
static_assert(8u * kHmaPitch * 2u == 2176u, "pad 行基址亦 16B 对齐");

// B1a：kernel 体抽为内联函数（kvh/chunk 由调用方解析）—— 单 session 的
// attn_kv_kernel_h 与批化 attn_kv_kernel_hb 共享同一份槽内算术：同一份
// 代码内联进两个 kernel → 槽内指令序逐位同构（验收锚 bitDiff=0 的依据；
// 批化的间接寻址只改变地址来源，不触碰任何浮点表达式/收缩选择）。
__device__ __forceinline__ void attn_kv_h_body(
    const uint16_t* __restrict__ q_rows, const uint32_t* __restrict__ d_base_pos,
    uint32_t layer, const uint32_t* __restrict__ page_table,
    uint16_t* __restrict__ kv_pool, KvLayout layout,
    float* __restrict__ partials, /* [16][max_chunks][130]（H2 写回）*/
    uint32_t max_chunks, uint16_t* __restrict__ out /* [2048]（direct 支路）*/,
    const float* __restrict__ inv_freq /*必须 null：仅非 ph0*/,
    uint32_t chunk_pos,
    float* __restrict__ score_scratch /* 可 null：S 调试锚 [max_chunks][16]
                                          [chunk_pos] fp32 */,
    uint32_t kvh /* 0..1 */, uint32_t chunk /* 0..max_chunks-1 */) {
    const bool ph0 = inv_freq != nullptr; // H3：phase0（rope 吸收）支持
    MC_PDL_TRIGGER();
    const uint32_t limit = *d_base_pos;
    const uint32_t start = chunk * chunk_pos;
    if (start > limit) return; // 空块（全块一致早退，barrier 前返回）
    const uint32_t end = min(start + chunk_pos - 1u, limit);
    const uint32_t tid = threadIdx.x;
    const uint32_t warp = tid >> 5; // 0..7
    const uint32_t lane = tid & 31u;

    // ---- 改动 1：K/V 独立数组 [2 buf][16 pos][136] bf16（动态 smem）----
    extern __shared__ uint16_t sKVH[]; // 16B 对齐（动态 smem 基址；行距 272B）
    uint16_t* const sK = sKVH;                             // [2][16][136]
    uint16_t* const sV = sKVH + 2u * kKvStage * kHmaPitch; // [2][16][136]

    // 协作异步装载（照抄 v2 issue_stage，仅目的地 pitch 128 → 136）：每 pos
    // 32 个 16B（K 16 + V 16），满 stage 时 256 线程各 2 条。
    auto issue_stage = [&](uint32_t s0, int b) {
        const uint32_t sEnd = min(s0 + kKvStage - 1u, end);
        const uint32_t n = sEnd - s0 + 1u;
        for (uint32_t i = tid; i < n * 32u; i += kKvBlockThreads) {
            const uint32_t sp = i >> 5;  // stage 内位置
            const uint32_t w4 = i & 31u; // pos 内第几个 16B
            const uint32_t pos = s0 + sp;
            const uint32_t page = page_table[layout.page_index(pos)];
            const uint32_t slot = layout.slot_index(pos);
            const uint32_t half = w4 >> 4; // 0=K, 1=V
            const uint32_t v16 = w4 & 15u; // 半边内第几个 16B（8 dim）
            uint16_t* const dst =
                (half ? sV : sK) + (b * kKvStage + sp) * kHmaPitch;
            mc_cp_async16(reinterpret_cast<uint4*>(dst) + v16,
                          kv_pool + layout.elem_offset(layer, page, slot, half,
                                                       kvh, v16 * 8u));
        }
    };

    // PDL 预取前移（同 v2）：stage0 在 WAIT 之前发射。
    issue_stage(start, 0);
    mc_cp_async_commit();

    MC_PDL_WAIT(); // q/qkv 行由前驱写出（同 v2 的安全性论证；ph0 时含 k/v 列）

    // ---- H3：phase0 表（block 协作一次性算 64 频率的 cos/sin；表达式与
    //      v2 :600-611 逐位同：__fmul_rn((float)limit, inv_freq[t]) +
    //      cosf/sinf）----
    __shared__ float sCS[64], sSN[64]; // +512B 静态（12544+512=13056B）
    if (ph0) {
        if (tid < 64u) {
            const float f = __fmul_rn((float)limit, inv_freq[tid]);
            sCS[tid] = cosf(f);
            sSN[tid] = sinf(f);
        }
        __syncthreads(); // 表写 → 全 block 可读（rope 装载 pass 消费）
    }

    // ---- 改动 2（H3 扩展）：sQT 装载 pass + 一次 sync ----
    // ph0=false：256 线程纯拷贝（行 8-15 清零 pad，现行为）。
    // ph0=true：t∈0..127 负责实头行 0..7 的 rope —— h=t>>4、16B 块 c=t&15，
    //   取 q_rows[(kvh*8+h)*128 + 8c..+8) 与伙伴块 c^8（(d,d±64) 恒落同线程
    //   的两个块 → 零 shuffle），逐元素 rope_rot_anchor + rope_round（与
    //   v2 :613-628 同锚表达式：x2 取伙伴块同槽位元素，值 = v2 shuffle
    //   交付的伙伴 lane 值 → 逐位一致；sQT 存 bf16 = rope_round 的舍入位）
    //   → STS sQT[h][8c..]。t∈128..255 同时清零 pad 行 8..15。
    __shared__ __align__(16) uint16_t sQT[16 * kHmaPitch];
    __shared__ float sS[kQPerKv][16][16]; // [8 warp][16 head][16 pos] = 8KB
    if (!ph0) {
        for (uint32_t i = tid; i < 16u * 128u; i += kKvBlockThreads) {
            const uint32_t row = i >> 7;  // 0..15
            const uint32_t dim = i & 127u;
            sQT[row * kHmaPitch + dim] =
                (row < 8u) ? q_rows[(kvh * kQPerKv + row) * 128u + dim]
                           : (uint16_t)0u; // pad 行清零（bf16 +0.0）
        }
    } else {
        if (tid < 128u) {
            const uint32_t h = tid >> 4;  // 本块实头 0..7
            const uint32_t c = tid & 15u; // 16B 块号（8 dim）
            const uint32_t qh0 = (kvh * kQPerKv + h) * 128u;
#pragma unroll
            for (uint32_t i = 0; i < 8u; ++i) {
                const uint32_t d = 8u * c + i;
                const uint32_t dp = d ^ 64u; // 伙伴维（同线程块 c^8 内同槽位）
                const float x = __bfloat162float(reinterpret_cast<const bf16&>(
                    q_rows[qh0 + d]));
                const float x2 = __bfloat162float(reinterpret_cast<const bf16&>(
                    q_rows[qh0 + dp]));
                sQT[h * kHmaPitch + d] = __bfloat16_as_ushort(
                    __float2bfloat16(rope_rot_anchor(x, x2, sCS[d & 63u],
                                                     sSN[d & 63u], d < 64u)));
            }
        } else {
            const uint32_t i = tid - 128u; // pad 行 8..15 清零（1024 元素）
            for (uint32_t j = i; j < 8u * 128u; j += 128u)
                sQT[(8u + (j >> 7)) * kHmaPitch + (j & 127u)] = (uint16_t)0u;
        }
    }
    __syncthreads();

    // ---- 改动 3（A fragment）：每 warp 一条 ldmatrix.x4，常驻寄存器 ----
    // 地址（字节）= 行 ((lane&7) + 8*((lane>>3)&1)) × 272 + 列 16*((lane>>4)&1)
    // + 32*w。⚠ 矩阵序 (q00, q10, q01, q11)：mma.m16n8k16 的 A fragment 寄存
    // 器序为 R0=(行g,列2c)、R1=(行g+8,列2c)、R2=(行g,列2c+8)、R3=(行g+8,
    // 列2c+8)（wmma.load.a 位编码 dump 实测锚定，与常见博客的 (q00,q01,q10,
    // q11) 序不同！）→ lane 0-7 给 q00 行址、8-15 给 q10（行 8-15，清零
    // pad）、16-23 给 q01（+16B）、24-31 给 q11。首版按 (q00,q01,q10,q11)
    // 序装载 → R1/R2 内容互换 → relL2≈0.7（对角/δ 输入因两象限同为零而
    // 恰好漏检，随机输入必炸）。
    uint32_t qa[4];
    {
        const uint32_t qAddr =
            mc_smem_u32(sQT) +
            (((lane & 7u) + 8u * ((lane >> 3) & 1u)) * kHmaPitch + 16u * warp) *
                2u +
            ((lane >> 4) & 1u) * 16u;
        asm volatile(
            "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
            : "=r"(qa[0]), "=r"(qa[1]), "=r"(qa[2]), "=r"(qa[3])
            : "r"(qAddr));
    }
    const uint32_t g = lane >> 2;       // mma D 行基（head tile 内 0..7）
    const uint32_t c2 = 2u * (lane & 3u); // mma D 列基（tile 内 0..6）
    const uint32_t qh = kvh * kQPerKv + g; // 本线程负责的真实 q head
    const bool direct = limit < chunk_pos; // 单 chunk：直写 out（同 v2）

    // ---- H2：SOFT / PV 跨 stage 状态（每线程 1 个实头 g + 8 个 acc 元素）----
    float m = -INFINITY, l = 0.f; // head g 的 online softmax 状态
    float acc[2][4];              // PV D-fragment：[nt][{0,1}]=head g dim、
                                   // [{2,3}]=pad 行 dim（恒 0，不写回）
#pragma unroll
    for (int nt = 0; nt < 2; ++nt)
#pragma unroll
        for (int r = 0; r < 4; ++r) acc[nt][r] = 0.f;

    // ---- 位置主循环（v2 双 barrier 流水；QK+STS 替换 compute_stage）----
    uint32_t s0 = start;
    for (int b = 0; s0 <= end; b ^= 1, s0 += kKvStage) {
        const bool has_next = (s0 + kKvStage <= end);
        if (has_next) {
            issue_stage(s0 + kKvStage, b ^ 1); // 预取下一 stage
            mc_cp_async_commit();
        }
        if (has_next)
            mc_cp_async_wait<1>();
        else
            mc_cp_async_wait<0>();
        __syncthreads(); // cp.async 写对全 block 可见

        // ---- H3 附加（正确性关键）：尾 stage 未装载行清零 ----
        // issue_stage 只拷 n = end-s0+1 行；行 [n,16) 是本 buf 残留 —— 本 buf
        // 首次使用时是**未初始化 smem**（生产池/kv 的 stale 与此处无关，拷贝
        // 根本不发生），可含 NaN/Inf。v2 的循环边界天然跳过 masked pos；
        // HMMA 的 mma 对整 16-pos tile 计算，pw=0 × NaN 行 = NaN（mma 微测
        // 实证：0 操作数 × NaN B → NaN D）污染 acc → NaN logits → 采样垃圾
        // token → embedding_gather 越界（生产崩溃链，gpu_layerwise/短测未
        // 爆是因 stale 行恰为有限值）。零清后 0×0=0，确定性成立。
        // 仅尾 stage 付一次清零 + barrier；K 侧 masked 列本就被 score 的
        // -INF mask 覆盖，一并清零只为单一路径。
        if (s0 + kKvStage - 1u > end) {
            const uint32_t nrow = end - s0 + 1u;
            for (uint32_t i = tid; i < (kKvStage - nrow) * 2u * 128u;
                 i += kKvBlockThreads) {
                const uint32_t r = nrow + (i >> 8);
                const uint32_t j = i & 255u;
                (j & 128u ? sV : sK)[(b * kKvStage + r) * kHmaPitch +
                                     (j & 127u)] = (uint16_t)0u;
            }
            __syncthreads(); // 清零 → 全块 ldmatrix 可见
        }

        // ---- H3：limit 位 K/V 的 smem patch（phase0）----
        // v2 对 j==limit 走三处寄存器旁路（:686-689/:731-733/:752-757）；
        // HMMA 消费全走 smem → 改为把 warp0 重算的 kcur/vcur 直接 STS 进
        // 本 stage 的 sK/sV 行（值链与 v2 池写逐位同锚：q_rows 的 k/v 列 +
        // 同式 rope_rot_anchor/shfl_xor(16)，warp0 内同一 lane↔dim 映射）。
        // 时机：本 stage cp.async 已落地（wait+sync 后）→ 写后写无竞态；
        // 补一次 __syncthreads 使 warp0 的 STS 对全块 ldmatrix 可见（仅
        // limit 所在 stage 多付 1 次 barrier）。kcur/vcur 寄存器寿命 confined
        // 在本块 → 不推高主循环寄存器压力（64 regs / 0 spill 红线）。
        if (ph0 && s0 <= limit && limit < s0 + kKvStage) {
            if (warp == 0u) {
                const uint32_t d0 = lane * 4u; // 连续 4 维（同 v2 映射）
                const uint32_t kcol = 2048u + kvh * 128u;
                const uint32_t vcol = 2304u + kvh * 128u;
                float xk[kDimsPerLane];
                uint16_t vraw[kDimsPerLane];
#pragma unroll
                for (uint32_t r = 0; r < kDimsPerLane; ++r) {
                    xk[r] = __bfloat162float(reinterpret_cast<const bf16&>(
                        q_rows[kcol + d0 + r]));
                    vraw[r] = q_rows[vcol + d0 + r];
                }
                uint16_t kcur[kDimsPerLane]; // bf16 位（= rope_round 舍入位）
#pragma unroll
                for (uint32_t r = 0; r < kDimsPerLane; ++r) {
                    const uint32_t e = d0 + r;
                    const float x2 =
                        __shfl_xor_sync(0xffffffffu, xk[r], 16); // 伙伴 d±64
                    kcur[r] = __bfloat16_as_ushort(__float2bfloat16(
                        rope_rot_anchor(xk[r], x2, sCS[e & 63u], sSN[e & 63u],
                                        e < 64u)));
                }
                // 池写（持久化；表达式 = v2 :659-669 逐位同锚）
                {
                    const uint32_t page = page_table[layout.page_index(limit)];
                    const uint32_t slot = layout.slot_index(limit);
#pragma unroll
                    for (uint32_t r = 0; r < kDimsPerLane; ++r) {
                        const uint32_t e = d0 + r;
                        kv_pool[layout.elem_offset(layer, page, slot, /*K*/ 0u,
                                                   kvh, e)] = kcur[r];
                        kv_pool[layout.elem_offset(layer, page, slot, /*V*/ 1u,
                                                   kvh, e)] = vraw[r];
                    }
                }
                // STS patch：sK/sV 的 limit 行（行寻址与 ldmatrix 消费一致：
                // pos 行 = (b*16+sp_lim)*136；QK 读 pos 行、PV 的 trans 半区
                // 选择子式读同样的行 → 单一写者覆盖全 128 dim）
                const uint32_t sp_lim = limit - s0;
                uint16_t* krow = sK + (b * kKvStage + sp_lim) * kHmaPitch;
                uint16_t* vrow = sV + (b * kKvStage + sp_lim) * kHmaPitch;
#pragma unroll
                for (uint32_t r = 0; r < kDimsPerLane; ++r) {
                    krow[d0 + r] = kcur[r];
                    vrow[d0 + r] = vraw[r];
                }
            }
            __syncthreads(); // warp0 patch → 全块 ldmatrix 可见
        }

        // QK：每 warp 2 × (ldmatrix.x2 载 K + mma) → STS sS[warp]
        const uint32_t kBase =
            mc_smem_u32(sK + b * kKvStage * kHmaPitch) + 32u * warp;
#pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
            // K 行主序 [pos][dim] 恰为 B 的 [n][k]：lane 0-7 → pos 行
            // (8nt+0..7) × dim[16w,16w+8)，lane 8-15 → 同 pos 行 × dim+8
            //（x2 只用 lane 0-15 的地址，16-31 计算值被忽略且必在界内）。
            const uint32_t kAddr = kBase +
                                   (8u * nt + (lane & 7u)) * kHmaPitch * 2u +
                                   ((lane >> 3) & 1u) * 16u;
            uint32_t kb0, kb1;
            asm volatile(
                "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
                : "=r"(kb0), "=r"(kb1)
                : "r"(kAddr));
            float d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f; // D 从 0 起
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                : "+&f"(d0), "+&f"(d1), "+&f"(d2), "+&f"(d3)
                : "r"(qa[0]), "r"(qa[1]), "r"(qa[2]), "r"(qa[3]), "r"(kb0),
                  "r"(kb1));
            // D fragment：d0/d1 = D[g][2c+{0,1}]，d2/d3 = D[g+8][…]（pad 行，
            // 恒 0；仍物化写满 16×16 tile，供 RED/调试完整性）。
            sS[warp][g][8u * nt + c2] = d0;
            sS[warp][g][8u * nt + c2 + 1u] = d1;
            sS[warp][g + 8u][8u * nt + c2] = d2;
            sS[warp][g + 8u][8u * nt + c2 + 1u] = d3;
        }
        __syncthreads(); // 全 warp 的 S partial 可见（亦是 RED 与下轮 STS 的栅栏）

        // ---- RED（H2：全 256 线程）+ S 锚（H1 语义，warp0、可选）----
        // 每线程归约 head g 的 4 个 score（pos = 2c+{0,1} + 8nt），w 升序。
        // pad 行（head g+8）恒 0：H2 对其零填充 A、不归约（见 H2-1 注）。
        //（基址 srow 预计算：8 个 w 的 LDS 地址 = srow + w*1024B 立即偏移，
        //  压地址寄存器活跃期。）
        float sc[4];
#pragma unroll
        for (int nt = 0; nt < 2; ++nt)
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                const uint32_t p = 8u * nt + c2 + j;
                const float* srow = &sS[0][g][p];
                float s = srow[0];
#pragma unroll
                for (int w = 1; w < (int)kQPerKv; ++w)
                    s = __fadd_rn(s, srow[w * 256]);
                float score = __fmul_rn(s, kScoreScale);
                if (s0 + p > end) score = -INFINITY; // 尾 stage mask
                sc[2 * nt + j] = score;
            }
        if (score_scratch != nullptr && warp == 0u) { // S 调试锚
            float* const dst =
                score_scratch +
                ((size_t)chunk * 16u + qh) * chunk_pos + (s0 - start);
            dst[c2] = sc[0];
            dst[c2 + 1u] = sc[1];
            dst[8u + c2] = sc[2];
            dst[8u + c2 + 1u] = sc[3];
        }

        // ---- SOFT（H2-1：head g 的 16-pos 批 online softmax，quad 归约）----
        float mn = fmaxf(fmaxf(sc[0], sc[1]), fmaxf(sc[2], sc[3]));
        mn = fmaxf(mn, __shfl_xor_sync(0xffffffffu, mn, 1));
        mn = fmaxf(mn, __shfl_xor_sync(0xffffffffu, mn, 2));
        mn = fmaxf(mn, m); // 并入运行 max（同 v2；max 逐位精确、序无差）
        const float renew =
            (m == -INFINITY) ? 0.f : __expf(__fadd_rn(m, -mn));
        float pwv[4]; // fp32 pw（l 用；PV 用其 bf16 量化副本）
#pragma unroll
        for (int i = 0; i < 4; ++i) pwv[i] = __expf(__fadd_rn(sc[i], -mn));
        float ls = __fmul_rn(l, renew);
        float lsum = (pwv[0] + pwv[1]) + (pwv[2] + pwv[3]); // 自身 4 项固定序
        lsum += __shfl_xor_sync(0xffffffffu, lsum, 1); // quad 内 Σpw（4 lane
        lsum += __shfl_xor_sync(0xffffffffu, lsum, 2); //  覆盖 16 pos，值恒同）
        l = ls + lsum;
        m = mn;

        // ---- P→A 直通（H2-2；寄存器序 = H1 实测的 mma A 布局）----
        // p01 = (head g, pos 2c/2c+1)、p23 = (head g, pos 8+2c/8+2c+1)；
        // A 的 pad 行（R1/R3）零填充（无未初始化读、无垃圾值依赖）。
        const __nv_bfloat162 p01 = __floats2bfloat162_rn(pwv[0], pwv[1]);
        const __nv_bfloat162 p23 = __floats2bfloat162_rn(pwv[2], pwv[3]);
        // acc 先 renew（仅实头分量 c0/c1；c2/c3 为 pad 死数据不缩放）
#pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
            acc[nt][0] = __fmul_rn(acc[nt][0], renew);
            acc[nt][1] = __fmul_rn(acc[nt][1], renew);
        }

        // ---- PV（H2-3）：2 × (ldmatrix.x2.trans 载 V + mma 累加到 acc）----
        // V smem [pos][dim] 行主序：trans 把 pos 行转成 B 的 [k=pos][n=dim]
        // 列片 —— lane 0-7 给 pos 行 0-7、lane 8-15 给 pos 行 8-15（x2 的
        // matrix1，对应 B 的 k=2c+8..9），dim 列 [16w+8nt, +8)。
        const uint32_t vBase =
            mc_smem_u32(sV + b * kKvStage * kHmaPitch) + 32u * warp;
#pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
            const uint32_t vAddr =
                vBase +
                (8u * ((lane >> 3) & 1u) + (lane & 7u)) * kHmaPitch * 2u +
                16u * nt;
            uint32_t vb0, vb1;
            asm volatile(
                "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 "
                "{%0,%1}, [%2];\n"
                : "=r"(vb0), "=r"(vb1)
                : "r"(vAddr));
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                : "+&f"(acc[nt][0]), "+&f"(acc[nt][1]), "+&f"(acc[nt][2]),
                  "+&f"(acc[nt][3])
                : "r"(*reinterpret_cast<const uint32_t*>(&p01)), "r"(0u),
                  "r"(*reinterpret_cast<const uint32_t*>(&p23)), "r"(0u),
                  "r"(vb0), "r"(vb1));
        }
        __syncthreads(); // 本 stage RED 完成 → 下轮 STS sS 可覆盖（v2 同位）
    }

    // ---- 写回（H2-4）：direct 支路或 partials 槽（契约同 v2）----
    float* const pw_base =
        partials + ((size_t)qh * max_chunks + chunk) * kAttnPartialStride;
    if (direct) {
        // n_valid==1：merge 公式退化为 out=bf16(acc/l)（sc=e^0=1 精确）。
        // 与 v2 逐位同式：__fdiv_rn(acc, l)（不预乘 1/l —— 双舍入会引入
        // 系统性 1-ulp 差，破坏 out 对拍 median=0）。
#pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
            out[qh * 128u + 16u * warp + 8u * nt + c2] =
                __bfloat16_as_ushort(__float2bfloat16(
                    (l > 0.f) ? __fdiv_rn(acc[nt][0], l) : 0.f));
            out[qh * 128u + 16u * warp + 8u * nt + c2 + 1u] =
                __bfloat16_as_ushort(__float2bfloat16(
                    (l > 0.f) ? __fdiv_rn(acc[nt][1], l) : 0.f));
        }
    } else {
#pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
            pw_base[2 + 16u * warp + 8u * nt + c2] = acc[nt][0];
            pw_base[2 + 16u * warp + 8u * nt + c2 + 1u] = acc[nt][1];
        }
        if (warp == 0u && (lane & 3u) == 0u) { // quad 内恒同值，lane=4g 单写
            pw_base[0] = m;
            pw_base[1] = l;
        }
    }
}

// 单 session 原 kernel（生产/对拍入口，签名与语义不变）：grid=(2 kvh,
// max_chunks)。体 = attn_kv_h_body。
__global__ void __launch_bounds__(kKvBlockThreads, 4) attn_kv_kernel_h(
    const uint16_t* __restrict__ q_rows, const uint32_t* __restrict__ d_base_pos,
    uint32_t layer, const uint32_t* __restrict__ page_table,
    uint16_t* __restrict__ kv_pool, KvLayout layout,
    float* __restrict__ partials, uint32_t max_chunks,
    uint16_t* __restrict__ out, const float* __restrict__ inv_freq,
    uint32_t chunk_pos, float* __restrict__ score_scratch) {
    attn_kv_h_body(q_rows, d_base_pos, layer, page_table, kv_pool, layout,
                   partials, max_chunks, out, inv_freq, chunk_pos, score_scratch,
                   blockIdx.x, blockIdx.y);
}

// ---- B1a：批化 HMMA kernel（attn_kv_kernel_hb）—— 多 session 合并解码
//      （路径 B）的批 attention 基石。组编排/ABI/图接线是 B1b/B2，本 kernel
//      仅由测试专用 launcher launch_attention_decode_batch 调用，生产路径
//      零接线。----
// grid.x = B×2（x = slot*2 + kvh）、grid.y = max_chunks 不变、block=256
// 结构不变（launch_bounds 与 kernel_h 同款 (256,4)）；槽内算术 =
// attn_kv_h_body —— 与 kernel_h 逐位同构（间接寻址只改地址计算，不改
// 指令序；间接寻址多出的几条地址寄存器若挤爆 64 上限，先缩地址现算，
// 禁降 occupancy）。slot 寻址二元制（ora-3 spec §3c）：
//   - 组集中量（连续布局 + slot stride 索引）：q_rows [B][2560]（qkv 输出
//     在组激活里天然连续）、out [B][2048]、partials [B][16][max_chunks]
//     [130] —— kernel 用 base + slot*stride 计算；
//   - 每 session 私有量（device 指针表 d_slot_*[slot]，表地址建期固定、
//     表内容运行时可变 → graph 友好）：KV 池指针（ph0 时写）、页表指针、
//     limit 的 device 地址（沿 kernel_h 从 device 地址读 limit 的模式）。
// 停车槽（padding 用的 dummy session）：kernel 无感知 —— 组层让对应表项
//   指向 1 页 dummy 池/页表即可（B1b 的事；本 kernel 只管算出任何 slot
//   编码输入的确定性结果）。rope 表 inv_freq 组内共享；ph0 的 limit 位
//   patch/池写均走 slot 解析后的指针，槽内表达式与 kernel_h 逐条相同。
__global__ void __launch_bounds__(kKvBlockThreads, 4) attn_kv_kernel_hb(
    const uint16_t* __restrict__ q_rows /* [B][2560] 组集中 */,
    const uint32_t* const* __restrict__ d_slot_pos /* [B]：limit device 地址 */,
    uint32_t layer, const uint32_t* const* __restrict__ d_slot_pt /* [B] 页表 */,
    uint16_t* const* __restrict__ d_slot_kv /* [B] KV 池（ph0 时写） */,
    KvLayout layout,
    float* __restrict__ partials /* [B][16][max_chunks][130] */,
    uint32_t max_chunks, uint16_t* __restrict__ out /* [B][2048] */,
    const float* __restrict__ inv_freq /* null → 非 ph0（组内共享表） */,
    uint32_t chunk_pos) {
    const uint32_t slot = blockIdx.x >> 1; // x = slot*2 + kvh
    const uint32_t kvh = blockIdx.x & 1u;  // 0..1
    attn_kv_h_body(
        q_rows + (size_t)slot * 2560u, d_slot_pos[slot], layer, d_slot_pt[slot],
        d_slot_kv[slot], layout,
        partials + (size_t)slot * 16u * max_chunks * kAttnPartialStride,
        max_chunks, out + (size_t)slot * 2048u, inv_freq, chunk_pos,
        /*score_scratch=*/nullptr, kvh, blockIdx.y);
}

// ---- 阶段 2（kv 路径）：并行 merge。grid=16，block=1024=8 split × 128 dim ----
// P8-A1 起生产路径改走 attn_merge_a→attn_merge_b（见下）；本 kernel 保留
// 供 kernel_ops 三方对拍（launch_attention_decode_merge 的 null-partials2
// 支路）与 A/B 对照。chunk_pos 参数化（P8-A2：n_valid = limit/chunk_pos+1）。
constexpr uint32_t kMergeSplit = 8;
__global__ void attn_merge_p(const float* __restrict__ partials,
                             const uint32_t* __restrict__ d_base_pos,
                             uint32_t max_chunks, uint32_t chunk_pos,
                             uint16_t* __restrict__ out /* [2048] */) {
    MC_PDL_TRIGGER();
    MC_PDL_WAIT(); // partials 由 attn_kv_kernel 写出
    const uint32_t qh = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    const uint32_t d = tid & 127u;   // 维度
    const uint32_t s = tid >> 7;     // slot 子序列
    const uint32_t limit = *d_base_pos;
    uint32_t n_valid = limit / chunk_pos + 1u;
    if (n_valid > max_chunks) n_valid = max_chunks;
    // 单 chunk：chunk kernel direct 支路已写 out（逐位一致）→ 整块早退
    //（block 内 n_valid 一致 → 无 barrier 越界）。
    if (n_valid == 1u) return;

    __shared__ float sM[kMergeSplit];
    __shared__ float sL[kMergeSplit];
    __shared__ float sA[kMergeSplit][128];

    // pass 1：各 slot 子序列的 max（防御性跳过 l==0，同旧版）
    float mloc = -INFINITY;
    for (uint32_t c = s; c < n_valid; c += kMergeSplit) {
        const float* p =
            partials + ((size_t)qh * max_chunks + c) * kAttnPartialStride;
        if (p[1] != 0.f) mloc = fmaxf(mloc, p[0]);
    }
    if (d == 0u) sM[s] = mloc;
    __syncthreads();
    float M = -INFINITY;
#pragma unroll
    for (uint32_t i = 0; i < kMergeSplit; ++i) M = fmaxf(M, sM[i]);

    // pass 2：各子序列加权和（线程 (s, d) 累 acc[d]）
    float L = 0.f, A = 0.f;
    for (uint32_t c = s; c < n_valid; c += kMergeSplit) {
        const float* p =
            partials + ((size_t)qh * max_chunks + c) * kAttnPartialStride;
        if (p[1] == 0.f) continue;
        const float sc = expf(p[0] - M);
        L += p[1] * sc;
        A += p[2 + d] * sc;
    }
    sA[s][d] = A;
    if (d == 0u) sL[s] = L;
    __syncthreads();
    if (s == 0u) { // 固定树合并（split 升序串加 —— 确定性）
        float Lt = 0.f;
#pragma unroll
        for (uint32_t i = 0; i < kMergeSplit; ++i) Lt += sL[i];
        float At = 0.f;
#pragma unroll
        for (uint32_t i = 0; i < kMergeSplit; ++i) At += sA[i][d];
        out[qh * 128u + d] =
            __bfloat16_as_ushort(__float2bfloat16((Lt > 0.f) ? (At / Lt) : 0.f));
    }
}

// ---- P8-A1 阶段 2：两级 merge（生产路径，替换 attn_merge_p 调用点）----
// 动机：attn_merge_p grid 仅 16 block（84 SM 闲置）+ CHUNK=64 使 level-1
// partials 在 128K ctx 达 17.03MB/层 —— merge 每步 42 层各一次，合计
// ~4-5ms，是 128K decode 差距的主要部分。两级化：
//   merge_a：grid=(16 qh, S2=32)，block 128 thr（线程 d ↔ 维度 d，同
//            attn_merge_p 的 :383 式）。每 block 归并 c ≡ s (mod S2) 的
//            level-1 partial 子序列（pass1 max + pass2 加权和，表达式逐条
//            照抄 attn_merge_p 的 :398-418，不改收缩选择）→ level-2
//            partial {M_s, L_s, acc_s[128]} 写 [16][S2][130]。
//   merge_b：grid=(16)，block 128 thr。读 level-2 partials，升序固定串加
//            合并 S2 组（同 :422-431 固定树式的串加语义 + 跨组 max 重标定）
//            → out[qh*128+d]。
// 确定性：split 下标映射 c ≡ s (mod S2) 固定、两组循环均升序串加、无
// atomics → eager/graph 逐位一致（PDL 边仅提前发射，wait 仍全完成同步）。
// 空块约定：n_valid==1 时 merge_a/merge_b 全块一致早退（chunk kernel 的
// direct 支路已写 out）；s ≥ n_valid 的空 split 写 L=0 partial，merge_b 按
// p[1]==0 跳过（同 :401/:414 的 l==0 防御约定 —— level-2 的 acc 段此时为
// stale，但绝不被读）。
// B1a：merge 体抽为内联函数（qh/s 参数化）—— 单 session 的 attn_merge_a
// 与批化 attn_merge_batch_a 共享同一份归并算术（split 下标映射 / 收缩
// 选择 / 升序串加序逐位同构）。
__device__ __forceinline__ void attn_merge_a_body(
    const float* __restrict__ partials,
    const uint32_t* __restrict__ d_base_pos, uint32_t max_chunks,
    uint32_t chunk_pos,
    float* __restrict__ partials2 /* [16][S2][130] */, uint32_t qh, uint32_t s) {
    MC_PDL_TRIGGER();
    MC_PDL_WAIT(); // partials 由 attn_kv_kernel 写出
    const uint32_t d = threadIdx.x; // 0..127 ↔ 维度 d
    const uint32_t limit = *d_base_pos;
    uint32_t n_valid = limit / chunk_pos + 1u;
    if (n_valid > max_chunks) n_valid = max_chunks;
    if (n_valid == 1u) return; // direct 支路已写 out（与 merge_b 同判定早退）

    float* pw2 = partials2 + ((size_t)qh * kMergeSplit2 + s) * kAttnPartialStride;
    if (s >= n_valid) { // 空 split：L=0 占位（merge_b 跳过；acc 段不写）
        if (d == 0u) {
            pw2[0] = -INFINITY;
            pw2[1] = 0.f;
        }
        return;
    }

    // pass 1：本子序列的 max（防御性跳过 l==0，同 attn_merge_p 式）
    float mloc = -INFINITY;
    for (uint32_t c = s; c < n_valid; c += kMergeSplit2) {
        const float* p =
            partials + ((size_t)qh * max_chunks + c) * kAttnPartialStride;
        if (p[1] != 0.f) mloc = fmaxf(mloc, p[0]);
    }

    // pass 2：本子序列加权和（M = mloc；表达式逐条照抄 attn_merge_p 的
    // :411-418，收缩选择不变 —— 与全局 M 的差别由 merge_b 的重标定吸收）
    float L = 0.f, A = 0.f;
    for (uint32_t c = s; c < n_valid; c += kMergeSplit2) {
        const float* p =
            partials + ((size_t)qh * max_chunks + c) * kAttnPartialStride;
        if (p[1] == 0.f) continue;
        const float sc = expf(p[0] - mloc);
        L += p[1] * sc;
        A += p[2 + d] * sc;
    }
    if (d == 0u) {
        pw2[0] = mloc;
        pw2[1] = L;
    }
    pw2[2 + d] = A;
}

// B1a：merge_b 体同样抽为内联函数（qh 参数化），供批化实例共享。
__device__ __forceinline__ void attn_merge_b_body(
    const float* __restrict__ partials2,
    const uint32_t* __restrict__ d_base_pos, uint32_t max_chunks,
    uint32_t chunk_pos, uint16_t* __restrict__ out /* [2048] */, uint32_t qh) {
    MC_PDL_TRIGGER();
    MC_PDL_WAIT(); // partials2 由 attn_merge_a 写出
    const uint32_t d = threadIdx.x; // 0..127 ↔ 维度 d
    const uint32_t limit = *d_base_pos;
    uint32_t n_valid = limit / chunk_pos + 1u;
    if (n_valid > max_chunks) n_valid = max_chunks;
    // 与 merge_a 同判定早退（level-2 为 stale；direct 支路已写 out）
    if (n_valid == 1u) return;

    // 组间 max（跳过 L==0 的空 split / 空 partial）
    float M = -INFINITY;
#pragma unroll 4
    for (uint32_t i = 0; i < kMergeSplit2; ++i) {
        const float* p =
            partials2 + ((size_t)qh * kMergeSplit2 + i) * kAttnPartialStride;
        if (p[1] != 0.f) M = fmaxf(M, p[0]);
    }
    // 升序固定串加合并 S2 组（重标定 exp(M_i - M)；同 :422-431 树式的
    // 串加语义）
    float L = 0.f, A = 0.f;
#pragma unroll 4
    for (uint32_t i = 0; i < kMergeSplit2; ++i) {
        const float* p =
            partials2 + ((size_t)qh * kMergeSplit2 + i) * kAttnPartialStride;
        if (p[1] == 0.f) continue;
        const float sc = expf(p[0] - M);
        L += p[1] * sc;
        A += p[2 + d] * sc;
    }
    out[qh * 128u + d] =
        __bfloat16_as_ushort(__float2bfloat16((L > 0.f) ? (A / L) : 0.f));
}

// 单 session 原 merge（生产/对拍入口，签名与语义不变）：merge_a grid=
// (16, S2)、merge_b grid=(16)。体 = attn_merge_a_body / attn_merge_b_body。
__global__ void attn_merge_a(const float* __restrict__ partials,
                             const uint32_t* __restrict__ d_base_pos,
                             uint32_t max_chunks, uint32_t chunk_pos,
                             float* __restrict__ partials2) {
    attn_merge_a_body(partials, d_base_pos, max_chunks, chunk_pos, partials2,
                      blockIdx.x, blockIdx.y);
}
__global__ void attn_merge_b(const float* __restrict__ partials2,
                             const uint32_t* __restrict__ d_base_pos,
                             uint32_t max_chunks, uint32_t chunk_pos,
                             uint16_t* __restrict__ out) {
    attn_merge_b_body(partials2, d_base_pos, max_chunks, chunk_pos, out,
                      blockIdx.x);
}

// ---- B1a：批化两级 merge —— 与 attn_merge_a/b 共享体，仅地址/limit 按
//      slot 解析：merge_batch_a grid=(B×16, 32)（x = slot*16 + qh）、
//      merge_batch_b grid=(B×16)；n_valid 读 slot 的 pos（device 指针表），
//      partials/partials2/out 均加 slot 维（组集中连续布局 + stride 索引）。
//      归并算术/串加序与单 session 逐位同构 → 锚 1 的 out 对拍 bitDiff=0。----
__global__ void attn_merge_batch_a(
    const float* __restrict__ partials /* [B][16][max_chunks][130] */,
    const uint32_t* const* __restrict__ d_slot_pos /* [B]：limit device 地址 */,
    uint32_t max_chunks, uint32_t chunk_pos,
    float* __restrict__ partials2 /* [B][16][S2][130] */) {
    const uint32_t slot = blockIdx.x >> 4; // x = slot*16 + qh
    const uint32_t qh = blockIdx.x & 15u;
    attn_merge_a_body(
        partials + (size_t)slot * 16u * max_chunks * kAttnPartialStride,
        d_slot_pos[slot], max_chunks, chunk_pos,
        partials2 + (size_t)slot * 16u * kMergeSplit2 * kAttnPartialStride, qh,
        blockIdx.y);
}
__global__ void attn_merge_batch_b(
    const float* __restrict__ partials2 /* [B][16][S2][130] */,
    const uint32_t* const* __restrict__ d_slot_pos /* [B]：limit device 地址 */,
    uint32_t max_chunks, uint32_t chunk_pos,
    uint16_t* __restrict__ out /* [B][2048] */) {
    const uint32_t slot = blockIdx.x >> 4; // x = slot*16 + qh
    const uint32_t qh = blockIdx.x & 15u;
    attn_merge_b_body(
        partials2 + (size_t)slot * 16u * kMergeSplit2 * kAttnPartialStride,
        d_slot_pos[slot], max_chunks, chunk_pos, out + (size_t)slot * 2048u,
        qh);
}


// ---- 阶段 1（回退路径，MC_ATTN_KV_OFF=1）：q-major chunk ----
// grid=(16 q head, max_chunks)，block 128 threads（4 warp × 32 lane）。
// 每 block 负责本 chunk 的 KV 位置区间 [c*CHUNK, min((c+1)*CHUNK, limit)]：
//   - lane 分摊 head_dim 的 4 个维度 {lane_dim+32r}（32 lane × 4 = 128 维），
//     warp 内 butterfly 归约出完整 128 维点积；
//   - KV 位置流按 j ≡ warp (mod 4) 在 chunk 内分摊，online softmax fp32；
//   - 块内 4 warp 状态合并（shared）→ block partial (m, l, acc[128])
//     fp32 写入 session scratch 固定地址（graph 兼容）。
__global__ void attn_chunk_kernel(const uint16_t* __restrict__ q_rows,
                                  const uint32_t* __restrict__ d_base_pos,
                                  uint32_t layer,
                                  const uint32_t* __restrict__ page_table,
                                  const uint16_t* __restrict__ kv_pool,
                                  KvLayout layout, float* __restrict__ partials,
                                  uint32_t max_chunks,
                                  uint32_t chunk_pos /* P8-A2：与 grid 口径一致 */) {
    const uint32_t qh = blockIdx.x;    // 0..15
    const uint32_t chunk = blockIdx.y; // 0..max_chunks-1（grid 建期固定）
    MC_PDL_TRIGGER(); // 入口放行后继（后继消费本 kernel 输出前必 wait）
    MC_PDL_WAIT(); // q/K/V 由前驱 rope kernel 写出
    const uint32_t limit = *d_base_pos;
    const uint32_t start = chunk * chunk_pos;
    if (start > limit) return;
    const uint32_t end = min(start + chunk_pos - 1u, limit);
    const uint32_t d = threadIdx.x; // 0..127（lane = d&31，warp = d>>5）
    const uint32_t kvh = qh / 8u;    // GQA 16→2
    const uint64_t qrow = 0u;        // 单 token：q 行 = 第 0 行

    const uint32_t lane_dim = d & 31u;
    float qv[kDimsPerLane];
#pragma unroll
    for (uint32_t r = 0; r < kDimsPerLane; ++r)
        qv[r] = __bfloat162float(reinterpret_cast<const bf16&>(
            q_rows[qrow + qh * 128u + lane_dim + 32u * r]));

    const int warp = (int)(d >> 5);
    float m = -INFINITY, l = 0.f;
    float acc[kDimsPerLane];
#pragma unroll
    for (uint32_t r = 0; r < kDimsPerLane; ++r) acc[r] = 0.f;

    // P5：自适应 —— 短序列（本 warp 位置数 < 2 轮）走原串行循环，长序列
    // 走 4 路软件流水（装载与 softmax 链重叠；两体运算/收缩模式逐条对齐）。
    const uint32_t npos_warp = (end >= start) ? (end - start + 1u) : 0u;
    if (npos_warp >= kWarps * 4u * 2u) {
    for (uint32_t j0 = start + (uint32_t)warp; j0 <= end; j0 += kWarps * kAttnPipe) {
        uint32_t pg[kAttnPipe], sl[kAttnPipe];
        float kk[kAttnPipe][kDimsPerLane], vv[kAttnPipe][kDimsPerLane];
#pragma unroll
        for (int p = 0; p < kAttnPipe; ++p) {
            const uint32_t jj = j0 + (uint32_t)p * kWarps;
            if (jj <= end) {
                pg[p] = page_table[layout.page_index(jj)];
                sl[p] = layout.slot_index(jj);
            }
        }
#pragma unroll
        for (int p = 0; p < kAttnPipe; ++p) {
            const uint32_t jj = j0 + (uint32_t)p * kWarps;
            if (jj <= end) {
#pragma unroll
                for (uint32_t r = 0; r < kDimsPerLane; ++r)
                    kk[p][r] = __bfloat162float(reinterpret_cast<const bf16&>(
                        kv_pool[layout.elem_offset(layer, pg[p], sl[p], 0u, kvh,
                                                   lane_dim + 32u * r)]));
#pragma unroll
                for (uint32_t r = 0; r < kDimsPerLane; ++r)
                    vv[p][r] = __bfloat162float(reinterpret_cast<const bf16&>(
                        kv_pool[layout.elem_offset(layer, pg[p], sl[p], 1u, kvh,
                                                   lane_dim + 32u * r)]));
            }
        }
#pragma unroll
        for (int p = 0; p < kAttnPipe; ++p) {
            const uint32_t jj = j0 + (uint32_t)p * kWarps;
            if (jj > end) break;
            float dot = 0.f;
#pragma unroll
            for (uint32_t r = 0; r < kDimsPerLane; ++r)
                dot = fmaf(qv[r], kk[p][r], dot);
#pragma unroll
            for (int sw = 16; sw > 0; sw >>= 1)
                dot += __shfl_xor_sync(0xffffffffu, dot, sw);
            const float score = __fmul_rn(dot, kScoreScale);
            const float mn = fmaxf(m, score);
            const float pw = expf(__fadd_rn(score, -mn));
            const float renew = (m == -INFINITY) ? 0.f : expf(__fadd_rn(m, -mn));
            // 数值锚：dot=FFMA 链、l=FFMA、acc=FMUL(acc·renew)+FFMA(pw·v)
            l = fmaf(l, renew, pw);
#pragma unroll
            for (uint32_t r = 0; r < kDimsPerLane; ++r)
                acc[r] = fmaf(pw, vv[p][r], __fmul_rn(acc[r], renew));
            m = mn;
        }
    }

    } else {
        for (uint32_t j = start + (uint32_t)warp; j <= end; j += kWarps) {
            const uint32_t page = page_table[layout.page_index(j)];
            const uint32_t slot = layout.slot_index(j);
            float dot = 0.f;
#pragma unroll
            for (uint32_t r = 0; r < kDimsPerLane; ++r) {
                const uint64_t koff =
                    layout.elem_offset(layer, page, slot, /*K*/ 0u, kvh,
                                       lane_dim + 32u * r);
                dot += qv[r] *
                       __bfloat162float(reinterpret_cast<const bf16&>(kv_pool[koff]));
            }
#pragma unroll
            for (int sw = 16; sw > 0; sw >>= 1)
                dot += __shfl_xor_sync(0xffffffffu, dot, sw);
            const float score = dot * kScoreScale;
            const float mn = fmaxf(m, score);
            const float p = expf(score - mn);
            const float renew = (m == -INFINITY) ? 0.f : expf(m - mn);
            l = l * renew + p;
#pragma unroll
            for (uint32_t r = 0; r < kDimsPerLane; ++r) {
                const uint64_t voff =
                    layout.elem_offset(layer, page, slot, /*V*/ 1u, kvh,
                                       lane_dim + 32u * r);
                const float v =
                    __bfloat162float(reinterpret_cast<const bf16&>(kv_pool[voff]));
                acc[r] = acc[r] * renew + p * v;
            }
            m = mn;
        }
    }

    // ---- 合并 4 个 warp 的 online softmax 状态（每 warp 覆盖全部 128 维）----
    __shared__ float sM[kWarps];
    __shared__ float sL[kWarps];
    __shared__ float sAcc[kWarps][128];
    const int lane = (int)(d & 31u);
    if (lane == 0) {
        sM[warp] = m;
        sL[warp] = l;
    }
#pragma unroll
    for (uint32_t r = 0; r < kDimsPerLane; ++r)
        sAcc[warp][lane_dim + 32u * r] = acc[r];
    __syncthreads();

    float m_total = -INFINITY;
#pragma unroll
    for (int w = 0; w < (int)kWarps; ++w) m_total = fmaxf(m_total, sM[w]);
    float sc[kWarps], l_total = 0.f;
#pragma unroll
    for (int w = 0; w < (int)kWarps; ++w) {
        sc[w] = (sL[w] == 0.f) ? 0.f : expf(sM[w] - m_total);
        l_total += sL[w] * sc[w];
    }

    // ---- 写 block partial（固定地址 scratch；线程 d 持 dims {lane_dim+32r}）----
    float* out2 = partials + ((size_t)qh * max_chunks + chunk) * kAttnPartialStride;
    if (d == 0) {
        out2[0] = m_total;
        out2[1] = l_total;
    }
#pragma unroll
    for (uint32_t r = 0; r < kDimsPerLane; ++r) {
        const uint32_t e = lane_dim + 32u * r;
        float a_total = 0.f;
#pragma unroll
        for (int w = 0; w < (int)kWarps; ++w) a_total += sAcc[w][e] * sc[w];
        out2[2 + e] = a_total;
    }
}

// ---- 阶段 2（legacy 回退路径）：串行 merge（逻辑与 attn_merge_p 同式）----
// P8-A2：chunk_pos 参数化（n_valid = limit/chunk_pos + 1）。
__global__ void attn_merge_kernel(const float* __restrict__ partials,
                                  const uint32_t* __restrict__ d_base_pos,
                                  uint32_t max_chunks, uint32_t chunk_pos,
                                  uint16_t* __restrict__ out /* [2048] 单 token 行 */) {
    MC_PDL_TRIGGER(); // 入口放行后继（后继消费本 kernel 输出前必 wait）
    MC_PDL_WAIT(); // partials 由阶段 1 写出
    const uint32_t qh = blockIdx.x;
    const int d = threadIdx.x; // 0..127
    const uint32_t limit = *d_base_pos;
    // chunk c 有效 ⟺ c*CHUNK ≤ limit ⟺ c ≤ limit/CHUNK；与阶段 1 同一
    // device 值推导 → 有效前缀内全部 partial 均为本次写入（无 stale 读）。
    uint32_t n_valid = limit / chunk_pos + 1u;
    if (n_valid > max_chunks) n_valid = max_chunks; // 防御（编排不变式保证不触发）

    // pass 1：全局 max（有效 chunk 的 l ≥ 1，防御性跳过 l==0）
    float M = -INFINITY;
#pragma unroll 4
    for (uint32_t c = 0; c < n_valid; ++c) {
        const float* p = partials + ((size_t)qh * max_chunks + c) * kAttnPartialStride;
        if (p[1] != 0.f) M = fmaxf(M, p[0]);
    }

    // pass 2：L 与输出（线程 d 读 p[2+d]，跨线程连续 → 合并访问）
    float L = 0.f, acc = 0.f;
#pragma unroll 4
    for (uint32_t c = 0; c < n_valid; ++c) {
        const float* p = partials + ((size_t)qh * max_chunks + c) * kAttnPartialStride;
        if (p[1] == 0.f) continue;
        const float sc = expf(p[0] - M);
        L += p[1] * sc;
        acc += p[2 + d] * sc;
    }
    out[qh * 128u + (uint32_t)d] =
        __bfloat16_as_ushort(__float2bfloat16((L > 0.f) ? (acc / L) : 0.f));
}


} // namespace

mc_status_t launch_attention_decode(const uint16_t* d_q_rows,
                                    const uint32_t* d_pos, uint32_t layer,
                                     const uint32_t* d_page_table,
                                     const uint16_t* d_kv_pool, KvLayout layout,
                                     float* d_partials, uint32_t max_chunks,
                                     uint16_t* d_out, cudaStream_t stream, bool pdl,
                                     const float* d_inv_freq, uint32_t chunk_pos,
                                     float* d_partials2) {
    if (d_q_rows == nullptr || d_pos == nullptr || d_page_table == nullptr ||
        d_kv_pool == nullptr || d_partials == nullptr || d_out == nullptr ||
        max_chunks == 0 || chunk_pos == 0) {
        mc::set_error("launch_attention_decode: invalid args (max_chunks=%u "
                      "chunk_pos=%u)",
                      max_chunks, chunk_pos);
        return MC_E_INVALID_ARGUMENT;
    }
    // P5-B：默认 kv-major（K/V 流量 ÷8 + CHUNK=64 铺满 SM + ILP-8 破串行
    // 链）；MC_ATTN_KV_OFF=1 回退旧 q-major 两段式（A/B 对照；选择只依赖
    // 进程环境 → graph 安全）。phase0（rope 吸收）仅 kv-major 支持
    //（forward.cpp 保证两判定一致）。
    // P8-A1：d_partials2 != nullptr → merge 走两级 attn_merge_a→attn_merge_b
    //（生产路径，grid=(16,32)+（16）铺开 SM）；nullptr → 旧 attn_merge_p
    //（kernel_ops 三方对拍支路）。P8-A2：chunk_pos 为会话档位（建期恒定
    // → graph 安全）。
    if (mc_attn_kv_major_enabled()) {
        MC_NVTX_PUSH("attention_decode_kv");
        // const_cast：phase0 需写当前位置 K/V 入池（launcher 契约保持只读视角，
        // 写语义等同被吸收的 rope_kv kernel）
        // P9-v2.1：mbarrier 无锁步流水实例（止损判决后默认关 —— MC_ATTN_V21_ON=1
        // 启用；默认与 MC_ATTN_V21_OFF=1 均回 v2 的 wait_group+双 barrier 流水。
        // 选择只依赖进程环境 → graph 安全）。两实例 compute_stage 逐位一致
        //（kernel_ops attention_decode_v21_vs_v2 锚）。
        // H4：路径分发，优先级 HMMA > v2.1 > v2。HMMA 实例（attn_kv_
        // kernel_h，非模板）走 v2 双 barrier 流水语义；launch 参数全静态、
        // limit 设备读取不变、env 进程内恒定 → CUDA Graph 安全（与既有
        // P3 可捕获性论证一致）。PDL：kernel 内 TRIGGER/WAIT 齐备。
        if (mc_attn_hmma_enabled()) {
            MC_LAUNCH_PDL_SMEM(attn_kv_kernel_h, dim3(2, max_chunks),
                               dim3(kKvBlockThreads), kHmaSmemBytes, stream,
                               pdl, d_q_rows, d_pos, layer, d_page_table,
                               const_cast<uint16_t*>(d_kv_pool), layout,
                               d_partials, max_chunks, d_out, d_inv_freq,
                               chunk_pos, /*score_anchor=*/nullptr);
        } else if (mc_attn_v21_enabled()) {
            MC_LAUNCH_PDL_SMEM(attn_kv_kernel_x<true>, dim3(2, max_chunks),
                               dim3(kKvBlockThreads), kKvSmemBytes, stream, pdl,
                               d_q_rows, d_pos, layer, d_page_table,
                               const_cast<uint16_t*>(d_kv_pool), layout,
                               d_partials, max_chunks, d_out, d_inv_freq,
                               chunk_pos, /*score_anchor=*/nullptr);
        } else {
            MC_LAUNCH_PDL_SMEM(attn_kv_kernel_x<false>, dim3(2, max_chunks),
                               dim3(kKvBlockThreads), kKvSmemBytes, stream, pdl,
                               d_q_rows, d_pos, layer, d_page_table,
                               const_cast<uint16_t*>(d_kv_pool), layout,
                               d_partials, max_chunks, d_out, d_inv_freq,
                               chunk_pos, /*score_anchor=*/nullptr);
        }
        MC_NVTX_POP();
        cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess) {
            mc::set_error("attention_decode kv launch failed: %s",
                          cudaGetErrorString(e));
            return MC_E_CUDA;
        }
        if (d_partials2 != nullptr) {
            MC_NVTX_PUSH("attention_decode_merge_a");
            MC_LAUNCH_PDL(attn_merge_a, dim3(16, kMergeSplit2), dim3(128u),
                          stream, pdl, d_partials, d_pos, max_chunks, chunk_pos,
                          d_partials2);
            MC_NVTX_POP();
            e = cudaGetLastError();
            if (e != cudaSuccess) {
                mc::set_error("attention_decode merge_a launch failed: %s",
                              cudaGetErrorString(e));
                return MC_E_CUDA;
            }
            MC_NVTX_PUSH("attention_decode_merge_b");
            MC_LAUNCH_PDL(attn_merge_b, dim3(16), dim3(128u), stream, pdl,
                          d_partials2, d_pos, max_chunks, chunk_pos, d_out);
            MC_NVTX_POP();
            e = cudaGetLastError();
            if (e != cudaSuccess) {
                mc::set_error("attention_decode merge_b launch failed: %s",
                              cudaGetErrorString(e));
                return MC_E_CUDA;
            }
            return MC_OK;
        }
        MC_NVTX_PUSH("attention_decode_merge");
        MC_LAUNCH_PDL(attn_merge_p, dim3(16), dim3(128u * kMergeSplit), stream,
                      pdl, d_partials, d_pos, max_chunks, chunk_pos, d_out);
        MC_NVTX_POP();
        e = cudaGetLastError();
        if (e != cudaSuccess) {
            mc::set_error("attention_decode merge launch failed: %s",
                          cudaGetErrorString(e));
            return MC_E_CUDA;
        }
        return MC_OK;
    }
    // legacy q-major 两段式（MC_ATTN_KV_OFF=1；不支持 phase0）
    MC_NVTX_PUSH("attention_decode_chunk");
    MC_LAUNCH_PDL(attn_chunk_kernel, dim3(16, max_chunks), dim3(kBlockThreads),
                  stream, pdl, d_q_rows, d_pos, layer, d_page_table, d_kv_pool,
                  layout, d_partials, max_chunks, chunk_pos);
    MC_NVTX_POP();
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        mc::set_error("attention_decode chunk launch failed: %s",
                      cudaGetErrorString(e));
        return MC_E_CUDA;
    }
    MC_NVTX_PUSH("attention_decode_merge");
    MC_LAUNCH_PDL(attn_merge_kernel, dim3(16), dim3(kBlockThreads), stream, pdl,
                  d_partials, d_pos, max_chunks, chunk_pos, d_out);
    MC_NVTX_POP();
    e = cudaGetLastError();
    if (e != cudaSuccess) {
        mc::set_error("attention_decode merge launch failed: %s",
                      cudaGetErrorString(e));
        return MC_E_CUDA;
    }
    return MC_OK;
}

// P8-A1 测试直调：仅阶段 2 merge（kernel_ops attention_decode_merge2 的
// 三方对拍入口 —— 构造好的 level-1 partials 直接进 merge，不经 attn_kv
// 重写）。d_partials2 == nullptr → 旧 attn_merge_p；非 null → 生产路径的
// attn_merge_a→attn_merge_b。无 PDL（测试流序即可）。
mc_status_t launch_attention_decode_merge(const float* d_partials,
                                          const uint32_t* d_pos,
                                          uint32_t max_chunks, uint32_t chunk_pos,
                                          float* d_partials2, uint16_t* d_out,
                                          cudaStream_t stream) {
    if (d_partials == nullptr || d_pos == nullptr || d_out == nullptr ||
        max_chunks == 0 || chunk_pos == 0) {
        mc::set_error("launch_attention_decode_merge: invalid args "
                      "(max_chunks=%u chunk_pos=%u)",
                      max_chunks, chunk_pos);
        return MC_E_INVALID_ARGUMENT;
    }
    if (d_partials2 != nullptr) {
        attn_merge_a<<<dim3(16, kMergeSplit2), dim3(128u), 0, stream>>>(
            d_partials, d_pos, max_chunks, chunk_pos, d_partials2);
        cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess) {
            mc::set_error("merge_a launch failed: %s", cudaGetErrorString(e));
            return MC_E_CUDA;
        }
        attn_merge_b<<<dim3(16), dim3(128u), 0, stream>>>(
            d_partials2, d_pos, max_chunks, chunk_pos, d_out);
        e = cudaGetLastError();
        if (e != cudaSuccess) {
            mc::set_error("merge_b launch failed: %s", cudaGetErrorString(e));
            return MC_E_CUDA;
        }
        return MC_OK;
    }
    attn_merge_p<<<dim3(16), dim3(128u * kMergeSplit), 0, stream>>>(
        d_partials, d_pos, max_chunks, chunk_pos, d_out);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        mc::set_error("merge_p launch failed: %s", cudaGetErrorString(e));
        return MC_E_CUDA;
    }
    return MC_OK;
}

// ---- P9-Step2 测试直调：裸 chunk kernel（v1/v2，不含 merge）——
//      kernel_ops attention_decode_v2_vs_v1 的同输入双跑入口（partials
//      fp32 对拍 + direct 支路 out 对拍 + phase0 池写对拍）。删除推迟到
//      最终基准验收后。
mc_status_t launch_attn_kv_v1_test(const uint16_t* d_q_rows,
                                   const uint32_t* d_pos, uint32_t layer,
                                   const uint32_t* d_page_table,
                                   uint16_t* d_kv_pool, KvLayout layout,
                                   float* d_partials, uint32_t max_chunks,
                                   uint16_t* d_out, cudaStream_t stream,
                                   const float* d_inv_freq, uint32_t chunk_pos) {
    attn_kv_kernel_v1<<<dim3(2, max_chunks), dim3(kKvBlockThreads), kKvSmemBytes,
                        stream>>>(d_q_rows, d_pos, layer, d_page_table, d_kv_pool,
                                  layout, d_partials, max_chunks, d_out, d_inv_freq,
                                  chunk_pos);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        mc::set_error("attn_kv_v1 launch failed: %s", cudaGetErrorString(e));
        return MC_E_CUDA;
    }
    return MC_OK;
}

mc_status_t launch_attn_kv_v2_test(const uint16_t* d_q_rows,
                                   const uint32_t* d_pos, uint32_t layer,
                                   const uint32_t* d_page_table,
                                   uint16_t* d_kv_pool, KvLayout layout,
                                   float* d_partials, uint32_t max_chunks,
                                   uint16_t* d_out, cudaStream_t stream,
                                   const float* d_inv_freq, uint32_t chunk_pos,
                                   float* d_score /* H1：非 null → S 锚实例 */) {
    if (d_score != nullptr) {
        attn_kv_kernel_x<false, true><<<dim3(2, max_chunks), dim3(kKvBlockThreads),
                                        kKvSmemBytes, stream>>>(
            d_q_rows, d_pos, layer, d_page_table, d_kv_pool, layout, d_partials,
            max_chunks, d_out, d_inv_freq, chunk_pos, d_score);
    } else {
        attn_kv_kernel_x<false, false><<<dim3(2, max_chunks), dim3(kKvBlockThreads),
                                         kKvSmemBytes, stream>>>(
            d_q_rows, d_pos, layer, d_page_table, d_kv_pool, layout, d_partials,
            max_chunks, d_out, d_inv_freq, chunk_pos, nullptr);
    }
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        mc::set_error("attn_kv_v2 launch failed: %s", cudaGetErrorString(e));
        return MC_E_CUDA;
    }
    return MC_OK;
}

mc_status_t launch_attn_kv_v21_test(const uint16_t* d_q_rows,
                                    const uint32_t* d_pos, uint32_t layer,
                                    const uint32_t* d_page_table,
                                    uint16_t* d_kv_pool, KvLayout layout,
                                    float* d_partials, uint32_t max_chunks,
                                    uint16_t* d_out, cudaStream_t stream,
                                    const float* d_inv_freq, uint32_t chunk_pos) {
    attn_kv_kernel_x<true, false><<<dim3(2, max_chunks), dim3(kKvBlockThreads),
                                    kKvSmemBytes, stream>>>(
        d_q_rows, d_pos, layer, d_page_table, d_kv_pool, layout, d_partials,
        max_chunks, d_out, d_inv_freq, chunk_pos, nullptr);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        mc::set_error("attn_kv_v21 launch failed: %s", cudaGetErrorString(e));
        return MC_E_CUDA;
    }
    return MC_OK;
}

// ---- H1+H2+H3 测试直调：HMMA tensor-core kernel（attn_kv_kernel_h）。非生产
//      路径 —— kernel_ops 的 attention_hmma_qk_anchor / attention_hmma_vs_v2 /
//      attention_rope_phase0(HMMA 臂) 对拍/计时调用。H2 起输出契约与 v2 同构
//      （partials [16][max_chunks][130] + direct 支路 out）；H3 起支持 phase0
//      （d_inv_freq 非 null：sQT rope 装载 + limit 位 smem patch + 池写，与
//      v2 逐位同锚）。chunk_pos 需为 kKvStage 的倍数（尾 stage 锚写/直写的
//      局部 pos 上界证明依赖）；d_score 可 null —— 非 null 时额外写 S 调试锚
//      （[max_chunks][16][chunk_pos]，有效前缀 [0..end-start]，尾 stage 越界
//      pos 写 -INF）。
mc_status_t launch_attn_kv_h_test(const uint16_t* d_q_rows,
                                  const uint32_t* d_pos, uint32_t layer,
                                  const uint32_t* d_page_table,
                                  uint16_t* d_kv_pool, KvLayout layout,
                                  float* d_partials, uint32_t max_chunks,
                                  uint16_t* d_out, cudaStream_t stream,
                                  const float* d_inv_freq, uint32_t chunk_pos,
                                  float* d_score) {
    if (chunk_pos % kKvStage != 0u) {
        mc::set_error("attn_kv_h_test: chunk_pos(%u) 需为 %u 的倍数", chunk_pos,
                      kKvStage);
        return MC_E_INVALID_ARGUMENT;
    }
    attn_kv_kernel_h<<<dim3(2, max_chunks), dim3(kKvBlockThreads), kHmaSmemBytes,
                       stream>>>(d_q_rows, d_pos, layer, d_page_table, d_kv_pool,
                                 layout, d_partials, max_chunks, d_out, d_inv_freq,
                                 chunk_pos, d_score);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        mc::set_error("attn_kv_h launch failed: %s", cudaGetErrorString(e));
        return MC_E_CUDA;
    }
    return MC_OK;
}

// ---- B1a 测试直调：批 decode attention（attn_kv_kernel_hb + 批两级
//      merge attn_merge_batch_a/b）。非生产路径 —— kernel_ops 的
//      attention_batch_vs_single 对拍/计时入口；生产 forward/session 接线
//      是 B1b（本轮零接线）。slot 寻址二元制见 attn_kv_kernel_hb 注：
//      组集中量（q 行/out/partials/partials2）连续布局 + slot stride；
//      每 session 私有量（KV 池/页表/limit 地址）走 device 指针表（表地址
//      建期固定、表内容运行时可变 → graph 友好）。无 PDL（测试流序即可，
//      与 launch_attn_kv_h_test 同口径）。----
mc_status_t launch_attention_decode_batch(
    const uint16_t* d_q_rows,               // [B][2560]（组集中 q 行）
    const uint32_t* const* d_slot_pos,      // [B]：每 slot 的 limit device 地址
    uint32_t B, uint32_t layer,
    const uint32_t* const* d_slot_pt,       // [B]：每 slot 页表 device 地址
    uint16_t* const* d_slot_kv,             // [B]：每 slot KV 池（ph0 时写）
    KvLayout layout,
    float* d_partials,                      // [B][16][max_chunks][130]
    uint32_t max_chunks,
    uint16_t* d_out,                        // [B][2048]
    cudaStream_t stream,
    const float* d_inv_freq,                // null → 非 ph0（组内共享表）
    uint32_t chunk_pos,
    float* d_partials2) {                   // [B][16][kMergeSplit2][130]
    if (d_q_rows == nullptr || d_slot_pos == nullptr || d_slot_pt == nullptr ||
        d_slot_kv == nullptr || d_partials == nullptr || d_out == nullptr ||
        d_partials2 == nullptr || B == 0 || max_chunks == 0 || chunk_pos == 0 ||
        chunk_pos % kKvStage != 0u) {
        mc::set_error("attention_decode_batch: invalid args (B=%u max_chunks=%u "
                      "chunk_pos=%u)",
                      B, max_chunks, chunk_pos);
        return MC_E_INVALID_ARGUMENT;
    }
    attn_kv_kernel_hb<<<dim3(B * 2u, max_chunks), dim3(kKvBlockThreads),
                        kHmaSmemBytes, stream>>>(
        d_q_rows, d_slot_pos, layer, d_slot_pt, d_slot_kv, layout, d_partials,
        max_chunks, d_out, d_inv_freq, chunk_pos);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        mc::set_error("attention_decode_batch kv launch failed: %s",
                      cudaGetErrorString(e));
        return MC_E_CUDA;
    }
    attn_merge_batch_a<<<dim3(B * 16u, kMergeSplit2), dim3(128u), 0, stream>>>(
        d_partials, d_slot_pos, max_chunks, chunk_pos, d_partials2);
    e = cudaGetLastError();
    if (e != cudaSuccess) {
        mc::set_error("attention_decode_batch merge_a launch failed: %s",
                      cudaGetErrorString(e));
        return MC_E_CUDA;
    }
    attn_merge_batch_b<<<dim3(B * 16u), dim3(128u), 0, stream>>>(
        d_partials2, d_slot_pos, max_chunks, chunk_pos, d_out);
    e = cudaGetLastError();
    if (e != cudaSuccess) {
        mc::set_error("attention_decode_batch merge_b launch failed: %s",
                      cudaGetErrorString(e));
        return MC_E_CUDA;
    }
    return MC_OK;
}
