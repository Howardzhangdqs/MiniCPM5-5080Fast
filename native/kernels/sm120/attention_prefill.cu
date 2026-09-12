// attention_prefill.cu — P2 统一流式 GQA causal attention kernel（P3+ 起
// prefill 专用；decode 走 attention_decode.cu 的 KV 分块两段式）。
//
//   grid=(16 q head, tokens)，block 128 threads（= 4 warp × 32 lane）。
//   线程 d（lane d%32）负责 head_dim 的 4 个维度 {lane_dim + 32r}
//  （32 lane × 4 = 128 维全覆盖）：
//     - 点积：每线程累加自己 4 维的 q·k，warp 内 __shfl_xor butterfly 归约
//       （32 lane × 4 维）后每个 lane 都持有完整 128 维点积。
//     - V 累加：同样按 4 维分摊 → 每个 warp 对全部 128 维都有 acc 分量，
//       4 个 warp 的 online softmax 状态（m/l/acc[128]）可以安全合并
//       （历史 bug：线程只持 1 维时 sAcc[w][d] 跨 warp 读到未初始化切片，
//       输出幅度塌缩到 ~1/4 —— 由 kernel_ops attention_decode/prefill 捕获）。
//   KV 位置流按 j ≡ warp (mod 4) 分摊，warp 内 online softmax（fp32）：
//     score = Σ q·k / sqrt(128)；m ← max(m, score)；p = exp(score−m)；
//     l/acc 相应缩放累加。末尾 4 个 warp 状态合并（shared），
//     out[t, qh*128+e] = bf16(acc_e/l)。
//   GQA：q head h → kv head h/8。
// P3 可捕获性：base_pos 从固定 device 地址 d_base_pos 运行时读取（§12.2）。
// 本文件保证「prefill 与 decode 用同一寻址（KvLayout）/同一数值语义」；
// 分块 flash 版（query tile × KV tile）留优化期（计划 §9.3）。
#include "kernels/sm120/kernels.h"

#include <cuda_bf16.h>

#include "runtime/error.h"

namespace {

using bf16 = __nv_bfloat16;

constexpr float kScoreScale = 0.08838834764831845f; // 1/sqrt(128)

// blockDim 假设：恰好 4 个满员 warp（__shfl_xor_sync 的全 0xffffffff mask
// 要求 32 lane 全部活跃；launcher 以 kBlockThreads 启动）。
constexpr uint32_t kBlockThreads = 128; // 4 warp × 32 lane
constexpr uint32_t kWarps = kBlockThreads / 32u;
constexpr uint32_t kDimsPerLane = KvLayout::kHeadDim / 32u; // 128/32 = 4
static_assert(kBlockThreads == 4u * 32u,
              "full-warp shuffle mask requires exactly 4x32 threads");
static_assert(32u * kDimsPerLane == KvLayout::kHeadDim,
              "lane-strided dim ownership must cover head_dim exactly");

__global__ void gqa_attention_kernel(const uint16_t* __restrict__ q_rows,
                                     uint32_t tokens,
                                     const uint32_t* __restrict__ d_base_pos,
                                     uint32_t layer, const uint32_t* __restrict__ page_table,
                                     const uint16_t* __restrict__ kv_pool,
                                     KvLayout layout, uint16_t* __restrict__ out) {
    const uint32_t qh = blockIdx.x; // 0..15
    const uint32_t t = blockIdx.y;  // token
    const uint32_t d = threadIdx.x; // 0..127（lane = d&31，warp = d>>5）
    const uint32_t base_pos = *d_base_pos;
    const uint32_t kvh = qh / 8u;               // GQA 16→2
    const uint32_t limit = base_pos + t;        // causal：j ≤ limit（含自身）
    const uint64_t qrow = (uint64_t)t * 2560u;
    const uint64_t orow = (uint64_t)t * 2048u;

    // 每线程预载 4 个 q 分量：本线程拥有的 dim = lane_dim + 32*r，其中
    // lane_dim = d&31（warp 内 lane 跨步分摊，32 lane × 4 = 128 维全覆盖；
    // 注意不是 d+32r —— d≥32 时那会越过 head_dim 边界）。
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

    for (uint32_t j = (uint32_t)warp; j <= limit; j += kWarps) {
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
        // butterfly：warp 内 32 lane × 4 维 = 128 维完整点积（每 lane 都得到全值）
#pragma unroll
        for (int s = 16; s > 0; s >>= 1) dot += __shfl_xor_sync(0xffffffffu, dot, s);

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
#pragma unroll
    for (uint32_t r = 0; r < kDimsPerLane; ++r) {
        const uint32_t e = lane_dim + 32u * r;
        float a_total = 0.f;
#pragma unroll
        for (int w = 0; w < (int)kWarps; ++w) a_total += sAcc[w][e] * sc[w];
        out[orow + qh * 128u + e] =
            __bfloat16_as_ushort(__float2bfloat16(a_total / l_total));
    }
}

} // namespace

mc_status_t launch_gqa_attention(const uint16_t* d_q_rows, uint32_t tokens,
                                 const uint32_t* d_base_pos, uint32_t layer,
                                 const uint32_t* d_page_table,
                                 const uint16_t* d_kv_pool, KvLayout layout,
                                 uint16_t* d_out, cudaStream_t stream) {
    if (d_q_rows == nullptr || d_base_pos == nullptr || d_page_table == nullptr ||
        d_kv_pool == nullptr || d_out == nullptr || tokens == 0) {
        mc::set_error("launch_gqa_attention: invalid args (tokens=%u)", tokens);
        return MC_E_INVALID_ARGUMENT;
    }
    // 原 seq_len_total/base 容量校验移至编排层（forward.cpp，host 侧 base 值）。
    MC_NVTX_PUSH("gqa_attention");
    gqa_attention_kernel<<<dim3(16, tokens), dim3(kBlockThreads), 0, stream>>>(
        d_q_rows, tokens, d_base_pos, layer, d_page_table, d_kv_pool, layout, d_out);
    MC_NVTX_POP();
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        mc::set_error("gqa_attention launch failed: %s", cudaGetErrorString(e));
        return MC_E_CUDA;
    }
    return MC_OK;
}

mc_status_t launch_attention_prefill(const uint16_t* d_q_rows, uint32_t tokens,
                                      const uint32_t* d_base_pos, uint32_t layer,
                                      const uint32_t* d_page_table,
                                      const uint16_t* d_kv_pool, KvLayout layout,
                                      uint16_t* d_out, cudaStream_t stream) {
    return launch_gqa_attention(d_q_rows, tokens, d_base_pos, layer, d_page_table,
                                d_kv_pool, layout, d_out, stream);
}

// ===========================================================================
// P1 flash-HMMA 化：attn_prefill_flash（P2 起含 cp.async 双组流水，P3 起
// 支持 split-KV partials + 生产接线 launch_attn_prefill_flash；P1 的测试
// 入口 launch_attn_prefill_flash_test 保持 S_kv=1 直写语义不变）。
//
// 结构（架构师定稿）：
//   grid = 一维 2·⌈T/16⌉ kvh-major（idx < ntiles → kvh0 否则 kvh1），
//   block = 256 thr = 8 warp，warp w ↔ q head kvh·8+w（GQA 16→2）。
//   每块负责 16 个 q-token × 该 kvh 的 8 个 head；块内 tile 序
//   tile = idx % ntiles，q 起始 base = tile·16。
//   causal：块扫描 KV 区间 [0, base_pos+base+min(15,T-1-base)]（tile_end）；
//   尾 tile 内 pos > base_pos+base+row_m → S=-INF（row_m = 输出行对应的
//   q-token 行号；mma D-frag 使 thread 持 (m,n) 全知，无需 smem 中转）。
//
// smem（静态 34816B = 34KB）：
//   - Q staging [4 head][16 tok][136] bf16（17.4KB，两批复用：warp0-3 先
//     stage 各自 head，ldmatrix.x4 ×8 k-step 转 A-frag 常驻寄存器后同步，
//     warp4-7 复用同一空间）。行距 136（272B）：行基址恒 16B 对齐
//     （ldmatrix/cp.async.cg 16B 硬要求）+ 相邻行旋转 +4 bank（272 mod
//     128 = 16 → 同一 8×8 矩阵 8 行铺满 32 bank 各一次，0 conflict；同
//     attention_decode.cu kHmaPitch 论证）。
//   - KV 双缓冲 [2 buf][16 pos][136] ×2（K/V 各 8.5KB，共 17.4KB）：
//     P2 起 cp.async.cg 16B 双组流水（issue(i+1) ∥ compute(i)，协议照抄
//     attn_kv_kernel_h 的 v2 双 barrier 环：迭代 i 先 issue tile i+1 进
//     b^1 + commit_group，再 wait_group≤1 → 前一 commit（tile i）落地 →
//     全块 barrier 后计算 buf b，计算后再 barrier（buf 被再装载前消费完）。
//     未装载行（sp ≥ nrows）显式 STS 零清 —— cp.async 无「零填充」语义，
//     与 valid 行的 cp.async 混写同一 stage 是安全的（两者均先于
//     wait+__syncthreads 落地；mma 全 tile 语义下 0×NaN=NaN，H 系列教训）。
//
// 每 warp 状态：acc[16 n-tile][4] f32 = 16tok×128dim 的 C-frag（64 reg/
//   thread）；m[2]/l[2]（行 g 与 g+8，g = lane>>2；quad 内 4 线程冗余持有
//   同两行 → shfl_xor{1,2} 合并、零跨 warp 通信）。
//
// 主循环（16-pos tile，j 从 0 到 tile_end；v2 双 barrier 流水）：
//   1) cp.async 预取 tile i+1 → 缓冲 b^1；wait_group 落地 tile i → 缓冲 b；
//      未装载行显式零清（mma 全 tile 语义下 0×NaN=NaN，H 系列教训：未清
//      smem 残留/未初始化可含 NaN 污染 acc）；
//   2) QK：8 k-step（dim 16 切片）× 2 n-tile = 16 条 mma.m16n8k16：
//      S[16tok][8pos] += Q A-frag(k 切片) × ldmatrix.x2 K[pos][k 切片]；
//   3) mask 尾 tile（pos > limit_row → -INF）；
//   4) SOFT：行向 online softmax（shfl_xor{1,2} quad 合并；公式逐条对齐
//      旧 kernel/attn_ref：score = S·(1/√128) → mn = max(m, 行 max) →
//      pw = exp(score−mn) → l = l·renew + Σpw）；
//   5) PV：A = P（C→A 寄存器直通，fp32→bf16 RNE 打包）× ldmatrix.x2.trans
//      V[pos][dim 切片] → 16 条 mma 累加 acc；acc 先 ×renew。
//
// mma fragment 编排（直接移植 attn_kv_kernel_h 的 H1 实测结论）：
//   - Q A-frag（ldmatrix.x4，矩阵序 (q00,q10,q01,q11)）：R0=(行 g,列 2c)、
//     R1=(行 g+8,列 2c)、R2=(行 g,列 2c+8)、R3=(行 g+8,列 2c+8)。M 维 =
//     q-token 行 —— ldmatrix 的 lane 8-15 给行 8-15（q10 象限）→ R1；
//     首版若按 (q00,q01,q10,q11) 序装载则 R1/R2 互换（H1 实测 relL2≈0.7，
//     对角探针漏检、随机输入必炸）。
//   - P→A 直通同布局：a0=(行 g,pos 2c/2c+1)、a1=(行 g+8,同 pos)、
//     a2=(行 g,pos 8+2c/…)、a3=(行 g+8,…) —— 与 decode H2-2 的差别仅
//     在 R1/R3 不再是清零 pad 而是真实的 g+8 行 score。
//   - K B-frag：smem [pos][dim] 行主序恰为 B 的 [n][k]，非 trans 的
//     ldmatrix.x2 直配；V B-frag：ldmatrix.x2.trans 把 pos 行转成 B 的
//     [n=dim][k=pos] 列序 —— 寻址逐字节照抄 attn_kv_kernel_h。
//
// 数值（预期）：SOFT 的 m/l 为 fp32 重分组（~1e-7）；acc 差异由 P→bf16
//   量化主导（~2^-9/项）→ relL2 vs attn_ref 预期 ~1e-3 量级（判据 1e-2）。
//   确定性：固定归约顺序、无 atomics。
//
// P3 split-KV 小 T 加速：grid = 2·⌈T/16⌉·S_kv（idx 最内层拆 s；S_kv 由
//   host 的 prefill_flash_split_kv(T) 推导，见 kernels.h）。本块只扫自己
//   split 的 KV tile 子区间 [s·q, min((s+1)·q, nkv))（q = ceil(nkv/S_kv)，
//   nkv = 本 q-tile 的 KV tile 数 = last_tile+1，随 tile_end 因 tile 而异、
//   块内一致整数推导）。epilogue 分叉：
//   - S_kv=1：直写 out（P1 契约不变）；
//   - S_kv>1：写 partials[(t·16+qh)·S_kv+s][130] = {m, l, acc[128]} fp32
//     （stride 复用 kAttnPartialStride=130；空段/全 mask 行写 m=-INF、l=0
//     的空态 —— merge 侧 l==0 守卫跳过，无 stale 读），随后独立 kernel
//     prefill_merge_kernel（grid=T·16、block 128，串行合并 S≤8 组，公式
//     逐条复用 attn_merge_kernel）归并写 out。无 atomics、升序串加 →
//     确定性。decode 侧（merge_a/merge_b、attn_kv 全家）一行不动。
// ===========================================================================
namespace {

constexpr uint32_t kPfBlockThreads = 8u * 32u; // 256 = 8 warp
constexpr uint32_t kPfTileRows = 16u;          // q-token tile 行数（mma M 维）
constexpr uint32_t kPfPitch = 136u;            // smem 行距（bf16 元素）= 272B
constexpr uint32_t kPfQBatch = 4u;             // Q staging 一批 head 数（两批）
constexpr uint32_t kPfDimTiles = 16u;          // PV 的 n-tile 数（128 dim/8）
static_assert(kPfBlockThreads == 8u * 32u, "8 full warps");
static_assert(8u * kPfPitch * 2u == 2176u, "smem 行基址 16B 对齐（272B 行距）");
// P2 smem 总量：Q staging 4×16×136 + KV 双缓冲 2buf×(K+V)×16×136 = 34816B
//（34KB；__launch_bounds__(256,2) → 2 block/SM 共 68KB，sm_120 100KB 内）。
static_assert((kPfQBatch + 2u * 2u) * kPfTileRows * kPfPitch * 2u == 34816u,
              "P2 smem 预算（Q staging + KV 双缓冲）");

// generic→shared u32 地址（attention_decode.cu 的 mc_smem_u32 同式；本文件
// 独立副本，匿名 namespace 内部链接不冲突）。
__device__ __forceinline__ uint32_t pf_smem_u32(const void* p) {
    return (uint32_t)__cvta_generic_to_shared(p);
}

// ---- P2：cp.async 异步装载原语（attention_decode.cu 的 mc_cp_async16 同式；
//      独立副本，匿名 namespace 内部链接不冲突）----
// 对齐（cp.async.cg 16B 硬要求）：全局侧 page 32KB / slot 1KB / head 256B /
// v16·16B；shared 侧 272B 行距（17×16B）+ v16·16B —— 两端恒 16B 对齐。
__device__ __forceinline__ void pf_cp_async16(void* smem_dst,
                                              const void* gmem_src) {
    const uint32_t s = (uint32_t)__cvta_generic_to_shared(smem_dst);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s),
                 "l"(gmem_src));
}
__device__ __forceinline__ void pf_cp_async_commit() {
    asm volatile("cp.async.commit_group;\n");
}
template <int N> // 等到 ≤N 个最早提交的组未完成
__device__ __forceinline__ void pf_cp_async_wait() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

__global__ void __launch_bounds__(kPfBlockThreads, 2) attn_prefill_flash(
    const uint16_t* __restrict__ q_rows, uint32_t tokens,
    const uint32_t* __restrict__ d_base_pos, uint32_t layer,
    const uint32_t* page_table, const uint16_t* __restrict__ kv_pool,
    KvLayout layout, uint16_t* __restrict__ out,
    float* __restrict__ partials, uint32_t S) {
    const uint32_t ntiles = (tokens + 15u) / 16u;
    const uint32_t idx = blockIdx.x;
    // P3 split-KV：idx 最内层拆 s（同 tile 的 S 块相邻 → KV 段局部性好）；
    // S=1 时 (kvh, tile) 映射与 P1 逐块一致。
    const uint32_t s = idx % S;
    const uint32_t tile = (idx / S) % ntiles;
    const uint32_t kvh = (idx / S) / ntiles; // kvh-major
    const uint32_t base = tile * 16u;        // q tile 起始（相对行）
    const uint32_t base_pos = *d_base_pos;
    // causal：块扫描到 tile 内最后一个有效行的绝对 limit（尾 tile clamp 到
    // T-1 —— 池中不存在更后面的位置，页表亦只覆盖 base_pos+T-1）。
    const uint32_t tile_end = base_pos + base + min(15u, tokens - 1u - base);
    const uint32_t tid = threadIdx.x;
    const uint32_t warp = tid >> 5;     // 0..7 ↔ head kvh*8+warp
    const uint32_t lane = tid & 31u;
    const uint32_t g = lane >> 2;       // M 维行基（token 行 g / g+8）
    const uint32_t c2 = 2u * (lane & 3u); // N 维列基（tile 内 2c..2c+1）
    const uint32_t qh = kvh * 8u + warp;  // 本 warp 的真实 q head

    // ---- smem：Q staging（两批复用）+ KV 双缓冲（P2：[buf][K/V]）----
    __shared__ __align__(16) uint16_t
        sQ[kPfQBatch * kPfTileRows * kPfPitch]; // 4×16×136×2B = 17408B
    __shared__ __align__(16) uint16_t
        sKV[2u][2u][kPfTileRows * kPfPitch]; // 2buf×(K,V)×16×136×2B = 17408B

    // ---- P3：本 split 的 KV tile 子区间 [jt0, jt1] ----
    // nkv 随 tile_end 因 tile 而异（块内一致整数推导，无 host 预计算依赖）；
    // qsp = ceil(nkv/S) → split s 覆盖 tiles [s·qsp, (s+1)·qsp)；空段
    // （s·qsp > last_tile，尾 q-tile 的尾部 split 可出现）跳过流水，但
    // epilogue 仍写空态 partial（l=0，merge 侧守卫跳过 → 无 stale 读）。
    const uint32_t last_tile = tile_end >> 4;
    const uint32_t nkv = last_tile + 1u;
    const uint32_t qsp = (nkv + S - 1u) / S;
    const uint32_t jt0 = s * qsp;
    const uint32_t jt1 = min(jt0 + qsp - 1u, last_tile);

    // ---- P2：协作异步装载 issue_stage（协议/寻址照抄 attn_kv_kernel_h）----
    // 每 pos K/V 各 256B = 32 个 16B；16 pos × 32 = 512 次 16B，满 stage 时
    // 256 线程各 2 条 cp.async.cg 在途。寻址/页表查找与 P1 的 LDG→STS 逐条
    // 同式，只把 LDG→STS 换成 cp.async（16B，绕过寄存器）。未装载行
    // （sp ≥ nrows，即 pos > tile_end）显式 STS 零清 —— cp.async 无零填充
    // 语义，且双缓冲下该行残留的是两轮前 tile 的真实数据或未初始化 smem
    //（H 系列教训：可能含 NaN → PV 的 0×NaN=NaN 污染 acc）；STS 与
    // cp.async 混写同一 stage 安全：两者均先于 wait_group+__syncthreads
    // 落地可见。
    auto pf_issue_stage = [&](uint32_t j0, int b) {
        const uint32_t nrows = min(16u, tile_end - j0 + 1u);
        for (uint32_t i = tid; i < kPfTileRows * 32u; i += kPfBlockThreads) {
            const uint32_t sp = i >> 5;  // tile 内 pos 行
            const uint32_t w4 = i & 31u;
            const uint32_t half = w4 >> 4; // 0=K, 1=V
            const uint32_t v16 = w4 & 15u; // 半边内第几个 16B（8 dim）
            uint16_t* const dst = sKV[b][half] + sp * kPfPitch + v16 * 8u;
            if (sp < nrows) {
                const uint32_t pos = j0 + sp;
                const uint32_t page = page_table[layout.page_index(pos)];
                const uint32_t slot = layout.slot_index(pos);
                pf_cp_async16(
                    dst, kv_pool + layout.elem_offset(layer, page, slot, half,
                                                      kvh, v16 * 8u));
            } else {
                *reinterpret_cast<uint4*>(dst) = make_uint4(0u, 0u, 0u, 0u);
            }
        }
    };

    // ---- P2 流水 prologue：tile jt0 的 cp.async 在 Q staging 之前发射 ----
    // （KV 装载与 Q staging/LDMatrix 重叠 ~整个 Q 段；sKV 与 sQ 不相交，
    //  无覆盖竞争。G_0 此后由环内 wait_group<1> 消费。）
    if (jt0 <= last_tile) {
        pf_issue_stage(jt0 * 16u, 0);
        pf_cp_async_commit();
    }

    // 本 warp 的 Q staging 槽位（warp&3）：warp0-3 批 0、warp4-7 批 1 复用
    // 同一空间。每 warp 16 行 × 16 个 16B = 256 次 16B 拷贝，lane 8 轮；
    // 尾 tile 无效行（t ≥ tokens）清零 —— mma A 的 +0 → S=0 有限值，
    // 其 softmax 结果为垃圾但 epilogue 按 t<T 拒写，且全程无 NaN。
    uint16_t* const sqh = sQ + (warp & 3u) * kPfTileRows * kPfPitch;
    const uint32_t qcol = qh * 128u; // q_rows 列基（行布局 [tokens,2560]）
    auto stage_q = [&]() {
#pragma unroll
        for (uint32_t u = lane; u < kPfTileRows * 16u; u += 32u) {
            const uint32_t row = u >> 4;   // tile 内 token 行
            const uint32_t v16 = u & 15u;  // 行内 16B 块
            uint4 v = make_uint4(0u, 0u, 0u, 0u);
            const uint32_t t = base + row;
            if (t < tokens)
                v = *reinterpret_cast<const uint4*>(
                    q_rows + (size_t)t * 2560u + qcol + v16 * 8u);
            *reinterpret_cast<uint4*>(sqh + row * kPfPitch + v16 * 8u) = v;
        }
    };

    // Q A-frag：8 个 k-step（dim 16 切片）× ldmatrix.x4，常驻寄存器（32
    // reg/thread）。地址（字节）= 行 ((lane&7)+8·((lane>>3)&1)) × 272 +
    // 列 16·kk + ((lane>>4)&1)·16 —— 矩阵序 (q00,q10,q01,q11)（H1 实测：
    // mma A 的 R1=(行+8, 2c)，lane 8-15 必须给行 8-15 的 q10 象限）。
    uint32_t qa[8][4];
    auto ldmatrix_q = [&]() {
#pragma unroll
        for (uint32_t kk = 0; kk < 8u; ++kk) {
            const uint32_t qAddr =
                pf_smem_u32(sqh) +
                (((lane & 7u) + 8u * ((lane >> 3) & 1u)) * kPfPitch + 16u * kk) *
                    2u +
                ((lane >> 4) & 1u) * 16u;
            asm volatile(
                "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                : "=r"(qa[kk][0]), "=r"(qa[kk][1]), "=r"(qa[kk][2]),
                  "=r"(qa[kk][3])
                : "r"(qAddr));
        }
    };

    // ---- Q staging 两批复用（批 0：warp0-3；同步后批 1 复用同一空间）----
    if (warp < 4u) stage_q();
    __syncthreads(); // 批 0 写入 → warp0-3 ldmatrix 可见
    if (warp < 4u) ldmatrix_q();
    __syncthreads(); // warp0-3 读毕 → 批 1 可覆盖
    if (warp >= 4u) stage_q();
    __syncthreads();
    if (warp >= 4u) ldmatrix_q();
    __syncthreads();

    // ---- online softmax 状态 + PV 累加器（行 g / g+8 两份）----
    float m[2] = {-INFINITY, -INFINITY};
    float l[2] = {0.f, 0.f};
    float acc[kPfDimTiles][4]; // D-frag：c0/c1 = 行 g dim {8nt+2c,+1}、
                               // c2/c3 = 行 g+8 同 dim（全 16 行均真实）
#pragma unroll
    for (uint32_t nt = 0; nt < kPfDimTiles; ++nt)
#pragma unroll
        for (uint32_t r = 0; r < 4; ++r) acc[nt][r] = 0.f;

    // ---- 位置主循环（P2 cp.async 双组流水：issue(i+1) ∥ compute(i)）----
    // 协议照抄 attn_kv_kernel_h 的 v2 双 barrier 环：迭代 i 先发射 tile i+1
    // 进 b^1（组 G_{i+1}）+ commit，再 wait_group：has_next 时 ≤1（G_i 落地
    // 而 G_{i+1} 在途）、尾迭代 ≤0（等全部）。两次 __syncthreads：wait 后
    // （cp.async 写对全 block 可见）与计算后（buf 被再装载前消费完 ——
    // 空段/单 tile 时循环体不进入，prologue 的组由 epilogue 前的最后一次
    // wait+sync 收尾…… 实际尾迭代 has_next=false → wait<0> 全落地）。
    // 确定性：cp.async 只改搬运方式，内容/地址与 P1 逐字节一致。
    for (uint32_t jt = jt0, b = 0; jt <= jt1; ++jt, b ^= 1) {
        const uint32_t j0 = jt * 16u;
        const bool has_next = jt < jt1;
        if (has_next) {
            pf_issue_stage(j0 + 16u, b ^ 1); // 预取下一 tile → 对侧 buf
            pf_cp_async_commit();
        }
        if (has_next)
            pf_cp_async_wait<1>();
        else
            pf_cp_async_wait<0>();
        __syncthreads(); // cp.async 写对全 block 可见

        // ---- QK：S[16tok][16pos]，8 k-step × 2 n-tile = 16 条 mma ----
        // K smem [pos][dim] 行主序恰为 B 的 [n][k]：非 trans ldmatrix.x2，
        // lane 0-7 → pos 行 (8nt+0..7) × dim[16kk,+8)、lane 8-15 → 同 pos
        // 行 × dim+8（x2 只用 lane 0-15 的地址，16-31 被忽略且必在界内）。
        float s[2][4]; // D-frag：{0,1}=行 g 的 pos 8nt+2c+{0,1}、{2,3}=行 g+8
#pragma unroll
        for (uint32_t nt = 0; nt < 2u; ++nt)
#pragma unroll
            for (uint32_t r = 0; r < 4; ++r) s[nt][r] = 0.f;
#pragma unroll
        for (uint32_t kk = 0; kk < 8u; ++kk) {
            const uint32_t kBase =
                pf_smem_u32(sKV[b][0]) + 32u * kk; // dim 16kk 切片
#pragma unroll
            for (uint32_t nt = 0; nt < 2u; ++nt) {
                const uint32_t kAddr =
                    kBase + (8u * nt + (lane & 7u)) * kPfPitch * 2u +
                    ((lane >> 3) & 1u) * 16u;
                uint32_t kb0, kb1;
                asm volatile(
                    "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
                    : "=r"(kb0), "=r"(kb1)
                    : "r"(kAddr));
                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                    : "+&f"(s[nt][0]), "+&f"(s[nt][1]), "+&f"(s[nt][2]),
                      "+&f"(s[nt][3])
                    : "r"(qa[kk][0]), "r"(qa[kk][1]), "r"(qa[kk][2]),
                      "r"(qa[kk][3]), "r"(kb0), "r"(kb1));
            }
        }

        // ---- scale + causal mask + 行向 online softmax（行 g / g+8）----
        // sc[r][2nt+j]：行 base+g+8r 的 pos j0+8nt+2c+j 分数；
        // score = S·(1/√128)（与旧 kernel/attn_ref 同 scale 序）；
        // pos > base_pos+t（未来）→ -INF。
        float sc[2][4];
#pragma unroll
        for (uint32_t r = 0; r < 2u; ++r) {
            const int32_t limit =
                (int32_t)(base_pos + base + g + 8u * r); // 行的 causal limit
#pragma unroll
            for (uint32_t nt = 0; nt < 2u; ++nt)
#pragma unroll
                for (uint32_t j = 0; j < 2u; ++j) {
                    const int32_t pos =
                        (int32_t)(j0 + 8u * nt + c2 + j);
                    float score =
                        __fmul_rn(s[nt][2u * r + j], kScoreScale);
                    if (pos > limit) score = -INFINITY;
                    sc[r][2 * nt + j] = score;
                }
        }
        float renew[2];
        float pw[2][4];
#pragma unroll
        for (uint32_t r = 0; r < 2u; ++r) {
            // 行 max：自身 4 项 → shfl_xor{1,2} 合并 quad（4 lane 覆盖 16
            // pos）→ 并入运行 m（max 逐位精确、序无差）。
            float mn = fmaxf(fmaxf(sc[r][0], sc[r][1]),
                             fmaxf(sc[r][2], sc[r][3]));
            mn = fmaxf(mn, __shfl_xor_sync(0xffffffffu, mn, 1));
            mn = fmaxf(mn, __shfl_xor_sync(0xffffffffu, mn, 2));
            mn = fmaxf(mn, m[r]);
            renew[r] = (m[r] == -INFINITY)
                           ? 0.f
                           : __expf(__fadd_rn(m[r], -mn));
            // 全 mask 守卫：本 stage 行内 16 pos 全 -INF（行 m 亦 -INF）时
            // mn=-INF，exp(-INF-(-INF))=NaN —— 把 pw 的减数换成 0（pw 恒
            // exp(-INF)=0），m 保持 -INF（l/acc 不动）。
            float mnext = mn;
            if (mn == -INFINITY) {
                mn = 0.f;
                mnext = -INFINITY;
            }
#pragma unroll
            for (uint32_t i = 0; i < 4u; ++i)
                pw[r][i] = __expf(__fadd_rn(sc[r][i], -mn));
            float lsum = (pw[r][0] + pw[r][1]) + (pw[r][2] + pw[r][3]);
            lsum += __shfl_xor_sync(0xffffffffu, lsum, 1); // quad 内 Σpw
            lsum += __shfl_xor_sync(0xffffffffu, lsum, 2); //（4 lane 值恒同）
            l[r] = __fmul_rn(l[r], renew[r]) + lsum;
            m[r] = mnext;
        }

        // ---- PV：acc ×= renew → 16 条 mma（P 直通 A + V trans B）----
#pragma unroll
        for (uint32_t nt = 0; nt < kPfDimTiles; ++nt) {
            acc[nt][0] = __fmul_rn(acc[nt][0], renew[0]); // 行 g
            acc[nt][1] = __fmul_rn(acc[nt][1], renew[0]);
            acc[nt][2] = __fmul_rn(acc[nt][2], renew[1]); // 行 g+8
            acc[nt][3] = __fmul_rn(acc[nt][3], renew[1]);
        }
        // C→A 寄存器直通（fp32→bf16 RNE；H1 实测 A 布局）：
        // a0=(行 g, pos 2c/2c+1)、a1=(行 g+8, 同 pos)、a2=(行 g, pos
        // 8+2c/…)、a3=(行 g+8, …)。masked pos 的 pw=exp(-INF)=0 → 天然
        // 不进 PV。
        const __nv_bfloat162 p01 =
            __floats2bfloat162_rn(pw[0][0], pw[0][1]); // 行 g，nt0
        const __nv_bfloat162 p11 =
            __floats2bfloat162_rn(pw[1][0], pw[1][1]); // 行 g+8，nt0
        const __nv_bfloat162 p02 =
            __floats2bfloat162_rn(pw[0][2], pw[0][3]); // 行 g，nt1
        const __nv_bfloat162 p12 =
            __floats2bfloat162_rn(pw[1][2], pw[1][3]); // 行 g+8，nt1
        const uint32_t a0 = *reinterpret_cast<const uint32_t*>(&p01);
        const uint32_t a1 = *reinterpret_cast<const uint32_t*>(&p11);
        const uint32_t a2 = *reinterpret_cast<const uint32_t*>(&p02);
        const uint32_t a3 = *reinterpret_cast<const uint32_t*>(&p12);
        // V smem [pos][dim] 行主序：trans 把 pos 行转成 B 的 [n=dim][k=pos]
        // 列片 —— lane 0-7 给 pos 行 0-7、lane 8-15 给 pos 行 8-15（x2 的
        // matrix1，对应 B 的 k=2c+8..9），dim 列 [8nt,+8)。
#pragma unroll
        for (uint32_t nt = 0; nt < kPfDimTiles; ++nt) {
            const uint32_t vAddr =
                pf_smem_u32(sKV[b][1]) + 16u * nt +
                (8u * ((lane >> 3) & 1u) + (lane & 7u)) * kPfPitch * 2u;
            uint32_t vb0, vb1;
            asm volatile(
                "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
                : "=r"(vb0), "=r"(vb1)
                : "r"(vAddr));
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                : "+&f"(acc[nt][0]), "+&f"(acc[nt][1]), "+&f"(acc[nt][2]),
                  "+&f"(acc[nt][3])
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(vb0), "r"(vb1));
        }
        __syncthreads(); // 本 tile ldmatrix 读毕 → 2 轮后本 buf 可再装载
    }

    // ---- epilogue：S=1 直写（契约同旧 kernel :137-138）/ S>1 写 partials ----
    // 直写：out[t·2048 + qh·128 + d] = bf16(acc/l)；__fdiv_rn 单舍入（与
    // decode direct 支路同式，不预乘 1/l 避免双舍入系统性偏差）。行
    // t ≥ tokens 不写（其 Q 行清零、limit 恒大于 tile_end → 垃圾值但无
    // NaN；l>0 对有效行恒成立 —— pos 0 ≤ limit 必可见，双保险保留判定）。
    // partial（P3）：slot = partials[(t·16+qh)·S+s]，布局 {m, l, acc[128]}
    // fp32 × kAttnPartialStride=130（复用 decode 两段式的 stride 常量）。
    // m/l 由 quad 内 c2==0 的线程写一份（quad 4 线程冗余持有同两行）；
    // acc 由各线程写自己的 {8nt+2c, +1} 两维（4 线程 × 32 维 = 128 全覆盖）。
    // 空段/全 mask 行 l=0 也写（merge 侧 l==0 守卫跳过 → 无 stale 读；
    // 此时 acc 恒 0：renew=0 支路 acc = acc·0 + 0·v）。
    const uint32_t t0 = base + g;
    const uint32_t t1 = base + g + 8u;
    if (S == 1u) {
        if (t0 < tokens && l[0] > 0.f) {
            uint16_t* const dst = out + (size_t)t0 * 2048u + qh * 128u + c2;
#pragma unroll
            for (uint32_t nt = 0; nt < kPfDimTiles; ++nt) {
                dst[8u * nt] = __bfloat16_as_ushort(
                    __float2bfloat16(__fdiv_rn(acc[nt][0], l[0])));
                dst[8u * nt + 1u] = __bfloat16_as_ushort(
                    __float2bfloat16(__fdiv_rn(acc[nt][1], l[0])));
            }
        }
        if (t1 < tokens && l[1] > 0.f) {
            uint16_t* const dst = out + (size_t)t1 * 2048u + qh * 128u + c2;
#pragma unroll
            for (uint32_t nt = 0; nt < kPfDimTiles; ++nt) {
                dst[8u * nt] = __bfloat16_as_ushort(
                    __float2bfloat16(__fdiv_rn(acc[nt][2], l[1])));
                dst[8u * nt + 1u] = __bfloat16_as_ushort(
                    __float2bfloat16(__fdiv_rn(acc[nt][3], l[1])));
            }
        }
    } else if (t0 < tokens) { // 行 base+g 的 partial（行 t ≥ tokens 拒写：
                             // slot 下标 (t·16+qh)·S+s 会越过 T·16·S 区域）
        float* const pw =
            partials + ((size_t)(t0 * 16u + qh) * S + s) * kAttnPartialStride;
        if (c2 == 0u) {
            pw[0] = m[0];
            pw[1] = l[0];
        }
#pragma unroll
        for (uint32_t nt = 0; nt < kPfDimTiles; ++nt) {
            pw[2u + 8u * nt + c2] = acc[nt][0];
            pw[2u + 8u * nt + c2 + 1u] = acc[nt][1];
        }
    }
    if (S > 1u && t1 < tokens) { // 行 base+g+8 同式（独立判定：两行都写）
        float* const pw =
            partials + ((size_t)(t1 * 16u + qh) * S + s) * kAttnPartialStride;
        if (c2 == 0u) {
            pw[0] = m[1];
            pw[1] = l[1];
        }
#pragma unroll
        for (uint32_t nt = 0; nt < kPfDimTiles; ++nt) {
            pw[2u + 8u * nt + c2] = acc[nt][2];
            pw[2u + 8u * nt + c2 + 1u] = acc[nt][3];
        }
    }
}

// ---- P3 prefill_merge：split-KV partials 串行合并（独立 kernel，严禁与
//      decode 的 merge_a/merge_b 共用 —— 本 kernel 只服务 prefill flash）----
// grid = (T·16)（block ↔ (t, qh)，t ≥ tokens 立即返回 —— grid 按 16 对齐
// 铺满时防御尾块）、block = 128（线程 d ↔ 维度 d，跨线程连续读 p[2+d]）。
// 公式逐条复用 attn_merge_kernel（:1695）：pass1 全局 max（l==0 的空态
// split 跳过 —— flash 的空段/全 mask 行写 m=-INF、l=0）；pass2 sc=exp(m−M)、
// L += l·sc、acc += p[2+d]·sc → out = bf16(acc/L)。S ≤ 8 串行升序 →
// 确定性（固定归约顺序，无 atomics）。有效行的 split0 恒覆盖 pos 0 ≤
// limit → l>0 恒有至少一组，L>0 成立。
__global__ void prefill_merge_kernel(const float* __restrict__ partials,
                                     uint32_t S, uint32_t tokens,
                                     uint16_t* __restrict__ out) {
    const uint32_t bh = blockIdx.x; // t·16 + qh
    const uint32_t t = bh >> 4;
    const uint32_t qh = bh & 15u;
    if (t >= tokens) return;
    const int d = (int)threadIdx.x; // 0..127 ↔ head_dim
    const float* const base =
        partials + (size_t)bh * S * kAttnPartialStride;

    float M = -INFINITY; // pass 1：有效 split 的全局 max
#pragma unroll 4
    for (uint32_t s = 0; s < S; ++s) {
        const float* const p = base + (size_t)s * kAttnPartialStride;
        if (p[1] != 0.f) M = fmaxf(M, p[0]);
    }
    float L = 0.f, a = 0.f; // pass 2：L 与输出
#pragma unroll 4
    for (uint32_t s = 0; s < S; ++s) {
        const float* const p = base + (size_t)s * kAttnPartialStride;
        if (p[1] == 0.f) continue;
        const float sc = expf(p[0] - M);
        L += p[1] * sc;
        a += p[2 + d] * sc;
    }
    out[(size_t)t * 2048u + qh * 128u + (uint32_t)d] =
        __bfloat16_as_ushort(
            __float2bfloat16((L > 0.f) ? __fdiv_rn(a, L) : 0.f));
}

} // namespace

// P3 生产 launcher：flash（S_kv>1 时写 partials）→ prefill_merge 归并写
// out。S_kv 由 host 的 prefill_flash_split_kv(tokens)（kernels.h）推导；
// d_partials 需 ≥ tokens·16·S_kv·130 fp32（session arena 的 40MB
// prefill_partials 切片封顶，见 session.cpp compute_sizes）。
mc_status_t launch_attn_prefill_flash(const uint16_t* d_q_rows, uint32_t tokens,
                                      const uint32_t* d_base_pos, uint32_t layer,
                                      const uint32_t* d_page_table,
                                      const uint16_t* d_kv_pool, KvLayout layout,
                                      uint16_t* d_out, float* d_partials,
                                      uint32_t S_kv, cudaStream_t stream) {
    if (d_q_rows == nullptr || d_base_pos == nullptr || d_page_table == nullptr ||
        d_kv_pool == nullptr || d_out == nullptr || tokens == 0 ||
        S_kv == 0u || S_kv > 8u || (S_kv > 1u && d_partials == nullptr)) {
        mc::set_error("launch_attn_prefill_flash: invalid args (tokens=%u "
                      "S_kv=%u partials=%p)",
                      tokens, S_kv, (void*)d_partials);
        return MC_E_INVALID_ARGUMENT;
    }
    MC_NVTX_PUSH("attn_prefill_flash");
    const uint32_t ntiles = (tokens + 15u) / 16u;
    attn_prefill_flash<<<dim3(2u * ntiles * S_kv), dim3(kPfBlockThreads), 0,
                         stream>>>(
        d_q_rows, tokens, d_base_pos, layer, d_page_table, d_kv_pool, layout,
        d_out, (S_kv == 1u) ? nullptr : d_partials, S_kv);
    if (S_kv > 1u) {
        MC_NVTX_POP();
        MC_NVTX_PUSH("prefill_merge");
        prefill_merge_kernel<<<dim3(tokens * 16u), dim3(128u), 0, stream>>>(
            d_partials, S_kv, tokens, d_out);
    }
    MC_NVTX_POP();
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        mc::set_error("attn_prefill_flash launch failed: %s",
                      cudaGetErrorString(e));
        return MC_E_CUDA;
    }
    return MC_OK;
}

// P1 测试专用 launcher（生产 forward 不接；kernel_ops 的 attention_prefill_
// flash 对拍入口；S_kv=1 直写语义，内部转发生产 launcher）。grid =
// 2·⌈T/16⌉（kvh-major 一维）、block 256、静态 smem。
mc_status_t launch_attn_prefill_flash_test(
    const uint16_t* d_q_rows, uint32_t tokens, const uint32_t* d_base_pos,
    uint32_t layer, const uint32_t* d_page_table, const uint16_t* d_kv_pool,
    KvLayout layout, uint16_t* d_out, cudaStream_t stream) {
    return launch_attn_prefill_flash(d_q_rows, tokens, d_base_pos, layer,
                                     d_page_table, d_kv_pool, layout, d_out,
                                     /*d_partials=*/nullptr, /*S_kv=*/1u,
                                     stream);
}
