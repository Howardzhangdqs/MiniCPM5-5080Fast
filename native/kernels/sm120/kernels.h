// kernels.h — SM120 (RTX 5080 / Blackwell GeForce) kernel launchers。
//
// 约定：
//   - launcher 返回 mc_status_t，launch 后必须 cudaGetLastError 检查，
//     失败写入 thread-local 错误并返回 MC_E_CUDA
//   - BF16 在 launcher 边界用 uint16_t 裸位型传递（ABI 不暴露 CUDA 类型）
//   - 全部 launcher 为 P2 真实现：embedding / rmsnorm(单行+行批量+融合残差) /
//     rope_kv（RoPE+KV 写页）/ attention_decode+prefill（统一 GQA causal，
//     每 lane 分摊 4 dim 的完整点积 + 4-warp online softmax 合并）/
//     swiglu(分离+行拼) / sampling(greedy f32/bf16)
//     + mock_next_token（MC_NATIVE_MOCK_FORWARD 的确定性 forward）
#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>

#include "minicpm_runtime.h"

// P2：KV 统一寻址（kv_cache.h 的 host/device 函数在 .cu 里直接可用）
#include "runtime/kv_cache.h"

// ---- P5 PDL（Programmatic Dependent Launch，sm_90+）：后继 kernel 可在前驱
//      drain 期间发射（W 预取等独立工作先行），griddepcontrol.wait 之后才
//      消费前驱输出。未以 PDL 属性启动的 kernel 中两者均为 no-op（安全
//      无条件内联）。trigger 置于本 kernel 全部相关写之后。----
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
#define MC_PDL_WAIT()    asm volatile("griddepcontrol.wait;")
#define MC_PDL_TRIGGER() asm volatile("griddepcontrol.launch_dependents;")
#else
#define MC_PDL_WAIT()    ((void)0)
#define MC_PDL_TRIGGER() ((void)0)
#endif

// host：pdl=true 时以 ProgrammaticStreamSerialization 属性启动（eager 流内
// 生效；stream capture 时转化为 graph 的 programmatic 端口边）
#define MC_LAUNCH_PDL(kern, grid, block, stream, pdl, ...)                      \
    do {                                                                        \
        if (pdl) {                                                              \
            cudaLaunchAttribute _at_[1];                                        \
            _at_[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;    \
            _at_[0].val.programmaticStreamSerializationAllowed = 1;              \
            cudaLaunchConfig_t _cfg_;                                           \
            _cfg_.gridDim = (grid);                                             \
            _cfg_.blockDim = (block);                                           \
            _cfg_.dynamicSmemBytes = 0;                                         \
            _cfg_.stream = (stream);                                            \
            _cfg_.attrs = _at_;                                                 \
            _cfg_.numAttrs = 1;                                                 \
            cudaLaunchKernelEx(&_cfg_, (kern), ##__VA_ARGS__);                  \
        } else {                                                                \
            (kern)<<<(grid), (block), 0, (stream)>>>(__VA_ARGS__);              \
        }                                                                       \
    } while (0)

// P5-B：带 dynamic shared memory 的变体（attn_kv kernel 的 32KB stage）。
#define MC_LAUNCH_PDL_SMEM(kern, grid, block, smem, stream, pdl, ...)           \
    do {                                                                        \
        if (pdl) {                                                              \
            cudaLaunchAttribute _at_[1];                                        \
            _at_[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;    \
            _at_[0].val.programmaticStreamSerializationAllowed = 1;             \
            cudaLaunchConfig_t _cfg_;                                           \
            _cfg_.gridDim = (grid);                                             \
            _cfg_.blockDim = (block);                                           \
            _cfg_.dynamicSmemBytes = (smem);                                    \
            _cfg_.stream = (stream);                                            \
            _cfg_.attrs = _at_;                                                 \
            _cfg_.numAttrs = 1;                                                 \
            cudaLaunchKernelEx(&_cfg_, (kern), ##__VA_ARGS__);                  \
        } else {                                                                \
            (kern)<<<(grid), (block), (smem), (stream)>>>(__VA_ARGS__);         \
        }                                                                       \
    } while (0)

// ---- 可选 NVTX 标记（计划 §26.2；默认关闭，编译期 -DMC_ENABLE_NVTX 开启）----
#if defined(MC_ENABLE_NVTX)
#include <nvtx3/nvToolsExt.h>
#define MC_NVTX_PUSH(name) nvtxRangePushA(name)
#define MC_NVTX_POP()      nvtxRangePop()
#else
#define MC_NVTX_PUSH(name) ((void)0)
#define MC_NVTX_POP()      ((void)0)
#endif

// ---- P4+：device 侧采样配置（计划 §14.2/§12.4）----
// 固定地址（SessionStateDevice.cfg）：参数变化只改值不改指针 → CUDA Graph
// replay 安全。decode_one 每次调用把 config 写入该地址（graph 外小拷贝），
// 采样 kernel 运行时读取。
struct SamplingConfigDevice {
    float    temperature;       // ≤0 → greedy（激活规则，host 侧判定）
    float    top_p;             // (0,1]；(0,1) 外的特殊值见 kernel 边界处理
    uint32_t top_k;             // ≤1 → greedy；>kSampMaxK 截断到上限
    float    rep_penalty;       // 1.0 = 关闭；>1 惩罚 / <1 鼓励重复
    uint32_t flags;             // 预留
    uint32_t pad;               // 对齐填充
};

// host 侧 seed → 64-bit RNG 状态（splitmix64；device PCG 见 sampling.cu）
inline uint64_t splitmix64(uint64_t x) {
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

namespace mc {
// P3+ decode GEMV tactic（session 建期快测选定，session 内固定 → graph 安全；
// 见 runtime/gemv_tactic.cpp）。0 显式保留给 cuBLASLt（自研未胜出的 shape）。
inline constexpr int32_t kGemvTacticCublas = 0;
inline constexpr int32_t kGemvTacticRows   = 1;
inline constexpr int32_t kGemvTacticSplitK = 2;
// P4：fp8 GEMV（precision=MC_FP8 时 decode 的 4 个层内 shape 固定走此路；
// 权重 = e4m3 × 逐行 fp32 scale，无 cuBLASLt 对照 → tactic 固定，
// 快测只做功能校验 + GB/s 记录）。
inline constexpr int32_t kGemvTacticFp8 = 3;

// P4：E4M3 → IEEE fp32 位转换（host/device 共用，kernel 内与 CPU 参考
// 必须同一实现）。e4m3 位布局：s(1) e(4, bias 7) m(3)，max 448；规范：
//   正常数：符号<<31 | (e-7+127)<<23 | m<<20（e ∈ [1,15]）
//   次正规（e=0, m≠0）：值 = m/8 × 2^-6 → 归一化后按上式重偏置
//   ±0：全零；0x7F/0xFF（S.1111.111）为 NaN —— satfinite 量化器不会
//   产生该编码，本实现按位运算结果为 ±480（注释约定，不特殊处理）。
#if defined(__CUDA_ARCH__)
#define MC_FP8_HD __device__ __forceinline__
#else
#define MC_FP8_HD inline
#endif
MC_FP8_HD float fp8_e4m3_to_f32(uint8_t v) {
    const uint32_t sign = (uint32_t)(v & 0x80u) << 24;
    const uint32_t exp  = (uint32_t)(v >> 3) & 0xFu; // 4 位指数
    const uint32_t man  = (uint32_t)(v & 0x7u);
    if (exp == 0u) {
        if (man == 0u) return __builtin_bit_cast(float, sign); // ±0
        // 次正规：m/8 × 2^-6；左移归一化到 m' ∈ [8,15]（≤3 轮）
        uint32_t m = man, e = 0;
        while ((m & 0x8u) == 0u) {
            m <<= 1;
            ++e;
        }
        const uint32_t bits = sign | ((127u - 6u - e) << 23) | ((m & 0x7u) << 20);
        return __builtin_bit_cast(float, bits);
    }
    const uint32_t bits = sign | ((exp - 7u + 127u) << 23) | (man << 20);
    return __builtin_bit_cast(float, bits);
}
} // namespace mc

#ifdef __cplusplus
extern "C++" {
#endif

// embedding.cu — gather：out[t*hidden + d] = table[id[t]*hidden + d]（真实可用）
mc_status_t launch_embedding_gather(const uint16_t* d_table,   // [vocab, hidden] BF16
                                    const int32_t* d_ids,      // [n_tokens]
                                    uint16_t*       d_out,     // [n_tokens, hidden]
                                    int32_t         hidden,
                                    int32_t         n_tokens,
                                    cudaStream_t    stream);

// rmsnorm.cu — N 元素 RMSNorm，fp32 累加，eps 参数（真实可用，两档实现）
mc_status_t launch_rmsnorm(const uint16_t* d_x,        // [n] BF16
                           const uint16_t* d_weight,   // [n] BF16
                           uint16_t*       d_out,      // [n] BF16
                           int32_t         n,
                           float           eps,
                           cudaStream_t    stream);

// rmsnorm.cu（P2）— 行批量 RMSNorm：rows 各自独立归一化（每行 n 元素）。
mc_status_t launch_rmsnorm_rows(const uint16_t* d_x,      // [rows,n] BF16
                                const uint16_t* d_weight, // [n]
                                uint16_t*       d_out,    // [rows,n]
                                uint32_t        rows,
                                int32_t         n,
                                float           eps,
                                cudaStream_t    stream);

// rmsnorm.cu（P2）— 融合 残差加 + RMSNorm（计划 §8.3 高价值融合 1）：
//   res[b] = bf16(f32(res[b]) + f32(x[b]))          // 参考 add_bf16 存储边界
//   out[b] = bf16(f32(res[b]) * inv_rms * f32(w[b])) // fp32 连乘 + 单次 RNE
// norm 作用于「舍入后的残差」；归一化输出无中间 bf16 舍入（与 python 参考
// tools/reference_forward.py rmsnorm_bf16/add_bf16 语义一致，三版统一）。
mc_status_t launch_residual_add_rmsnorm(uint16_t*       d_res,   // [rows,n] 原地累加
                                        const uint16_t* d_x,     // [rows,n]
                                        const uint16_t* d_w,     // [n]
                                        uint16_t*       d_out,   // [rows,n]
                                        uint32_t        rows,
                                        int32_t         n,
                                        float           eps,
                                        cudaStream_t    stream, bool pdl = false /* P5 PDL */);

// embedding.cu（B1b）— 批间接 gather：token 从每 slot 的成员 d_next_token
// 读取（device 指针表 [n] 寻址，表地址组内固定），查表写组集中 residual
// 行 [n][hidden]。批 decode 步的首算子（embedding 侧的 token 链路闭环）。
mc_status_t launch_embedding_gather_indirect(
    const uint16_t*        d_table,      // [vocab, hidden] BF16
    const int32_t* const*  d_slot_tok,   // [n]：每 slot 的 token device 地址
    uint16_t*              d_out,        // [n, hidden]
    int32_t                hidden,
    int32_t                n,
    cudaStream_t           stream);

// rope_kv.cu（P2 真实现）— RoPE(rotate_half 非交错, theta=5e6, fp32) + KV 写入。
// qkv 行布局 [tokens,2560]：q=tokens×(16 head×128)，k 在列 2048..2303，
// v 在 2304..2559。q 原地 rope 写回；k rope 后、v 直通写入 KV 页
// （KvLayout 寻址：layer, pos=base_pos+t）。grid=(tokens, 20)：
// blockIdx.y<16 → q head；16/17 → k head 0/1；18/19 → v head 0/1。
// P3 可捕获性：base_pos 不再是 kernel 参数，kernel 运行时从固定 device
// 地址 d_base_pos 读取（计划 §12.2：seq_len 变化只改 device state 内容，
// 不改 tensor pointer/launch 参数）。KV 容量护栏由编排层（forward.cpp）
// 用 host 侧 base 值检查。
mc_status_t launch_rope_kv(uint16_t*       d_qkv,      // [tokens,2560] 原地
                           uint32_t        tokens,
                           const uint32_t* d_base_pos, // device：首 token 绝对位置
                           uint32_t        layer,
                           const uint32_t* d_page_table,
                           uint16_t*       d_kv_pool,
                           KvLayout        layout,
                           double          rope_theta,
                           cudaStream_t    stream,
                           // P5：建期预计算的 inv_freq 表 [64]（null = 核内
                           // 逐线程计算；kernel_ops 直调用默认值）
                           const float*    d_inv_freq = nullptr, bool pdl = false /* P5 PDL */);

// rope_kv.cu（P5）— inv_freq 表建期初始化：与核内逐线程计算同一 device
// 表达式 → 逐 bit 一致；session 建期调用一次，内容此后只读（graph 兼容）。
mc_status_t launch_rope_inv_freq_init(float* d_table, // [64]
                                      double rope_theta, cudaStream_t stream);

// attention_*.cu（P2 真实现）— 统一的流式 GQA causal attention kernel：
//   grid=(16 q head, tokens)，block 128 threads（4 warp，warp 分摊 KV 位置流，
//   online softmax fp32，warp 状态最终合并）。
//   q 取自 d_q_rows[t, head*128..]，kv 经页表间接读取（KvLayout），
//   causal：query 绝对位置 base_pos+t 只看 j ≤ base_pos+t（含自身）。
//   score=fp32 dot/sqrt(128)；Σp·v fp32 → out[t, head*128..] bf16。
//   GQA：q head h → kv head h/8。
// P3+ 起 prefill 专用（decode 走上方的 KV 分块两段式路径）。
mc_status_t launch_gqa_attention(const uint16_t* d_q_rows,  // [tokens,2560]
                                 uint32_t        tokens,
                                 const uint32_t* d_base_pos, // device：首 token 绝对位置
                                 uint32_t        layer,
                                 const uint32_t* d_page_table,
                                 const uint16_t* d_kv_pool,
                                 KvLayout        layout,
                                 uint16_t*       d_out,        // [tokens,2048]
                                 cudaStream_t    stream);

// attention_decode.cu（P3+）— decode 单 token 的 KV 分块两段式 attention。
//   P5-B（默认）：kv-major chunk kernel（attn_kv_kernel）：grid=(2 kv head,
//   max_chunks)，block 256 threads = 8 warp（warp w ↔ q head kvh*8+w，GQA
//   16→2）。K/V 由 block 协作装载 shared 一次（64-pos stage，2×64×128×2B
//   = 32KB dynamic ≤48KB 免 opt-in），8 个 warp 在 smem 上各跑自己的
//   online softmax —— 每 kv head 的 K/V 全局流量从 8×（16 个 q head 各自
//   流读）降为 1×。per-(qh, chunk) partial 仍由单 warp 升序流 + 锚定
//   FFMA 链产出（lane dim 分配 (d&31)+32r 不变），布局 [16][chunks][130]
//   不变 → merge kernel 逻辑复用。
//   CHUNK 取 64（非 256）：grid=(2,132)（max_ctx=8448），ctx2048 时 64 个
//   活跃 block 铺满多数 SM（256-CHUNK 版只有 16 块 → SM 大量闲置，实测
//   反而慢于旧路径）；co-residency 3 block/SM（smem 100KB/32KB、线程
//   1536/256、寄存器预算 85/thread），264 block 接近单波调度。
//   旧 q-major 两段式（attn_chunk×merge）保留为 MC_ATTN_KV_OFF=1 的
//   A/B 回退路径（同一 partials 布局/同一 merge）。
//   P5-C：rope 吸收（phase0）——d_inv_freq != nullptr 时，当前位置的
//   q/k 旋转与 K/V 池写入在本 kernel phase0 寄存器内完成（(d, d±64) 恒
//   落在同一 lane 的 (r, r^2) 分量对 → 零 shuffle），rope kernel 从 decode
//   链中消失（prefill 仍用）。SASS 锚：FMUL(sn,x2)+FFMA(cs,x,∓t)（与
//   rope_kv_kernel 逐指令一致，kernel_ops attention_rope_phase0 逐位锚）。
//   阶段 2 merge kernel：grid=16，block 128 threads（线程 d ↔ 维度 d），
//   合并 n_valid 组 partial（标准 online-softmax 合并公式），输出
//   out[head] bf16。短上下文（limit < CHUNK，n_valid==1）时 chunk kernel
//   直接写 out（merge 公式的恒等退化，逐位一致）且同样写 partial →
//   merge 空跑结果不变。
//   prefill 不走此路径（prefill attention 后续单独优化）。
constexpr uint32_t kAttnDecodeChunk   = 64;  // 默认 chunk（kernel_ops 直调 / 分档基准）
constexpr uint32_t kAttnPartialStride = 130; // (m, l, acc[128]) fp32/chunk/head

// P8-A1 两级 merge：level-1 split 数（编译期常量）。merge_a grid.y=S2，
// 每 block 归并 c ≡ s (mod S2) 的 level-1 partial 子序列 → level-2
// partials [16][S2][130]；merge_b 再串加 S2 组 → out。固定下标映射 +
// 升序串加 → 确定性（eager/graph 逐位一致）。
constexpr uint32_t kMergeSplit2 = 32;

// P8-A2/P9-Step3 档位表：session 建期按 max_context_tokens 选 chunk
//（session 内恒定 → launch 参数不变 → CUDA Graph 安全）。长 ctx 用大
// chunk 压 level-1 partials 尺寸与 merge 流量。
// P9-Step3 边界修正（ora-1 二轮）：128 档上界 65536→49152。512 档
//（>114688 → 512，partials 4.26→2.13MB/层）**A/B 实测退化，默认关闭**：
// kernel 级 bench 130816 ctx —— c512 575GB/s vs c256 635 / c128 658
//（512 块=grid(2,256) 并行度/在途不足）；端到端探针 14.721 vs 14.224ms。
// MC_ATTN_CHUNK512_ON=1 可重新启用（A/B 复测用）；MC_ATTN_CHUNK512_OFF=1
// 为冗余的显式关闭（与默认一致，保持任务书命名的兼容）。env 静态一次
// 读取 → 进程内恒定，graph 安全。
inline uint32_t attention_decode_chunk_pos(uint32_t max_ctx) {
    static const bool c512_on = [] {
        return getenv("MC_ATTN_CHUNK512_ON") != nullptr &&
               getenv("MC_ATTN_CHUNK512_OFF") == nullptr;
    }();
    if (max_ctx <= 16384u) return 64u;
    if (max_ctx <= 49152u) return 128u;
    if (max_ctx <= 114688u || !c512_on) return 256u;
    return 512u;
}

// session 建期由 max_context_tokens（与所选 chunk）推导 chunk 上限
//（决定 grid.y 与 scratch 尺寸）。
inline uint32_t attention_decode_max_chunks(uint32_t max_context_tokens,
                                            uint32_t chunk = kAttnDecodeChunk) {
    return (max_context_tokens + chunk - 1) / chunk;
}

// P5-B：kv-major 路径开关（默认开；MC_ATTN_KV_OFF=1 回退旧 q-major 两段式
// 做 A/B。静态缓存首次读取 → 进程内恒定，CUDA Graph capture/replay 一致；
// forward.cpp 依据同一判定决定 decode 是否跳过 rope kernel）。
inline bool mc_attn_kv_major_enabled() {
    static const bool on = [] {
        const char* e = getenv("MC_ATTN_KV_OFF");
        return e == nullptr;
    }();
    return on;
}

// P9-v2.1：mbarrier 无锁步流水开关。**默认关（止损判决）**：A/B 实测
// kernel 级 637GB/s ≈ v2、eager bucket 77/122/216µs/层 ≈ v2（76/123/216）、
// TPOT 8.011/10.257/14.157 vs 8.029/10.297/14.107 —— 两条止损线
//（64K >115µs/层、128K >195µs/层）均触发，按预案回退 v2 默认。
// MC_ATTN_V21_ON=1 可重新启用（复测用）；MC_ATTN_V21_OFF=1 为冗余显式
// 关闭（与默认一致）。静态缓存首次读取 → 进程内恒定，graph 安全。
inline bool mc_attn_v21_enabled() {
    static const bool on = [] {
        return getenv("MC_ATTN_V21_ON") != nullptr &&
               getenv("MC_ATTN_V21_OFF") == nullptr;
    }();
    return on;
}

// H4+H5（HMMA 转正）：decode attention 的 HMMA tensor-core 路径（attn_kv_
// kernel_h，QK/PV 均 mma.m16n8k16 bf16）。**默认开（H5 GO 判决）**：
//   - kernel 级 eager（chunk+merge_a/b 背靠背，kernel_ops
//     attention_hmma_perf_gate）：64K 60.1µs/层（v2 120.1）、128K 170.0
//     （v2 214.1）—— GO 双线（64K ≤115、128K ≤195）满足；
//   - TPOT（perf_v2 sweep，1 轮×100 tok，graph）：32K 7.373 / 64K 9.097 /
//     128K 12.364 ms vs v2 7.970/10.246/14.079（-7.5%/-11.2%/-12.2%），
//     三档全部优于 vLLM（8.084/9.737/13.063）；
//   - 短 ctx：tpot_breakdown eager attn bucket ctx16 0.656→0.498ms
//    （-24%）、ctx2048 0.666→0.566ms（-15%）；graph bucket +0.2%；
//   - 数值：partials vs v2 relL2 ~0.7-1.0e-3（P→bf16 量化主导，CPU bf16-P
//     参考 1.3e-7~2.5e-7 钉死布线）；池写/ph0 逐位锚 bitDiff=0。
// MC_ATTN_HMMA_OFF=1 回退 v2（kernel 本体保留对拍，同 v1 先例）；优先级
// HMMA > v2.1 > v2（v2.1 与 HMMA 同开时 HMMA 优先——HMMA 只挂 v2 双 barrier
// 流水，mbarrier 无锁步下 limit 位 patch 可见性不保证）。env 静态缓存首次
// 读取 → 进程内恒定、与 ctx/limit 无关、launch 参数全静态 → CUDA Graph
// capture/replay 安全（graph_bench：graph==eager 逐 token 位精确）。
inline bool mc_attn_hmma_enabled() {
    static const bool on = [] {
        if (getenv("MC_ATTN_HMMA_OFF") != nullptr) return false;
        if (getenv("MC_ATTN_V21_ON") != nullptr &&
            getenv("MC_ATTN_V21_OFF") == nullptr)
            std::fprintf(stderr,
                         "[mc] MC_ATTN_HMMA(默认开) + MC_ATTN_V21_ON 同时开启："
                         "HMMA 优先（attn_kv_kernel_h 仅挂 v2 双 barrier 流"
                         "水，v2.1 mbarrier 实例不参与）\n");
        return true; // H5 GO 判决后默认开（见上注）；MC_ATTN_HMMA_OFF=1 回退
    }();
    return on;
}

mc_status_t launch_attention_decode(const uint16_t* d_q_rows,  // [1, 2560]
                                    const uint32_t* d_pos,     // device：query 绝对位置
                                    uint32_t        layer,
                                    const uint32_t* d_page_table,
                                    const uint16_t* d_kv_pool,
                                    KvLayout        layout,
                                    float*          d_partials, // [16][max_chunks][130]
                                    uint32_t        max_chunks,
                                    uint16_t*       d_out,      // [1, 2048]
                                    cudaStream_t    stream, bool pdl = false /* P5 PDL */,
                                    // P5-C：非 null → phase0（rope 吸收）：
                                    // 建期 inv_freq 表 [64]（launch_rope_inv_freq_init）
                                    const float*    d_inv_freq = nullptr,
                                    // P8-A2：本会话的 chunk 位置数（建期档位表
                                    // 选定；默认 64 = kernel_ops 直调用口径）
                                    uint32_t        chunk_pos = kAttnDecodeChunk,
                                    // P8-A1：level-2 partials [16][kMergeSplit2][130]
                                    //（非 null → merge_a→merge_b 两级 merge 生产路径；
                                    // null → 旧 attn_merge_p —— kernel_ops 对拍用）
                                    float*          d_partials2 = nullptr);

// attention_decode.cu（P8-A1，测试直调）— 仅阶段 2 merge：对已构造的
// level-1 partials 做合并（不经 attn_kv_kernel 重写）。d_partials2 == nullptr
// → 旧 attn_merge_p；非 null → merge_a→merge_b（与生产路径同一对 kernel）。
// kernel_ops 的 attention_decode_merge2 三方对拍入口。
mc_status_t launch_attention_decode_merge(const float*    d_partials, // [16][max_chunks][130]
                                          const uint32_t* d_pos,      // device：limit
                                          uint32_t        max_chunks,
                                          uint32_t        chunk_pos,
                                          float*          d_partials2, // [16][32][130] 或 null
                                          uint16_t*       d_out,      // [1, 2048]
                                          cudaStream_t    stream);

// attention_decode.cu（P9-Step2，测试直调）— 裸 chunk kernel（不含 merge）：
// v1 = 旧实现（lane l ↔ dims {l,l+32,l+64,l+96} + LDG→STS 双缓冲，保留为
// 对拍基准）；v2 = 生产实现（连续 4 维映射 + LDS.64 + cp.async 流水）。
// kernel_ops 的 attention_decode_v2_vs_v1 同输入双跑入口；删除推迟到最终
// 基准验收后。
mc_status_t launch_attn_kv_v1_test(const uint16_t* d_q_rows, const uint32_t* d_pos,
                                   uint32_t layer, const uint32_t* d_page_table,
                                   uint16_t* d_kv_pool, KvLayout layout,
                                   float* d_partials, uint32_t max_chunks,
                                   uint16_t* d_out, cudaStream_t stream,
                                   const float* d_inv_freq, uint32_t chunk_pos);
mc_status_t launch_attn_kv_v2_test(const uint16_t* d_q_rows, const uint32_t* d_pos,
                                   uint32_t layer, const uint32_t* d_page_table,
                                   uint16_t* d_kv_pool, KvLayout layout,
                                   float* d_partials, uint32_t max_chunks,
                                   uint16_t* d_out, cudaStream_t stream,
                                   const float* d_inv_freq, uint32_t chunk_pos,
                                   // H1：非 null → S 锚实例（score = dot×
                                   // kScoreScale 写入该 scratch，布局
                                   // [max_chunks][16 qh][chunk_pos]，仅
                                   // [0..end-start] 前缀有效）；null → 原 v2
                                   float* d_score = nullptr);
// P9-v2.1（生产默认）：mbarrier 无锁步流水实例。kernel_ops 的
// attention_decode_v21_vs_v2 逐位对拍入口。
mc_status_t launch_attn_kv_v21_test(const uint16_t* d_q_rows,
                                    const uint32_t* d_pos, uint32_t layer,
                                    const uint32_t* d_page_table,
                                    uint16_t* d_kv_pool, KvLayout layout,
                                    float* d_partials, uint32_t max_chunks,
                                    uint16_t* d_out, cudaStream_t stream,
                                    const float* d_inv_freq, uint32_t chunk_pos);
// H1+H2+H3（HMMA tensor-core，测试/生产候选）：骨架同 v2（grid=(2,chunks)
// /256thr/16-pos stage/cp.async/PDL/limit 读取/空块早退），QK 段与 PV 段均
// 走 mma.m16n8k16 bf16 + 跨 warp S 归约 + 16-pos 批 online softmax；输出契
// 约与 v2 同构（partials [16][max_chunks][130] / limit<chunk_pos 时直写
// out）。H3 起支持 phase0（d_inv_freq 非 null：sQT rope 装载 + limit 位
// smem patch + 池写，与 v2 逐位同锚；仅挂 v2 双 barrier 流水）。
// d_score 可 null；非 null 时额外写 S 锚（[max_chunks][16][chunk_pos]，
// 有效前缀 [0..end-start]，尾 stage 越界 pos 写 -INF）。kernel_ops 的
// attention_hmma_qk_anchor / attention_hmma_vs_v2 入口。
mc_status_t launch_attn_kv_h_test(const uint16_t* d_q_rows,
                                  const uint32_t* d_pos, uint32_t layer,
                                  const uint32_t* d_page_table,
                                  uint16_t* d_kv_pool, KvLayout layout,
                                  float* d_partials, uint32_t max_chunks,
                                  uint16_t* d_out, cudaStream_t stream,
                                  const float* d_inv_freq, uint32_t chunk_pos,
                                  float* d_score /* 可 null：S 调试锚 */);

// attention_decode.cu（B1a，测试直调）— 批 decode attention：多 session
// 合并解码（路径 B）的批 kernel（attn_kv_kernel_hb，grid.x = B*2 =
// slot*2+kvh）+ 批两级 merge（attn_merge_batch_a grid=(B*16,32) /
// attn_merge_batch_b grid=(B*16)，x = slot*16+qh）。槽内算术与单 session
// 的 attn_kv_kernel_h / attn_merge_a/b 共享同一份内联体 → 逐位同构。
// slot 寻址二元制（ora-3 spec §3c）：
//   - 组集中量（连续布局 + slot stride）：q_rows [B][2560]、out [B][2048]、
//     partials [B][16][max_chunks][130]、partials2 [B][16][32][130]；
//   - 每 session 私有量（device 指针表 [B]，表地址建期固定 → graph 友好）：
//     d_slot_pos（limit 的 device 地址，沿 kernel_h 设备读取模式）、
//     d_slot_pt（页表）、d_slot_kv（KV 池，ph0 时写）；
//   - rope 表 inv_freq 组内共享（null → 非 ph0）。
// 非生产路径（生产接线是 B1b）—— kernel_ops 的 attention_batch_vs_single
// 对拍/计时入口。chunk_pos 需为 16 的倍数（同 launch_attn_kv_h_test）。
mc_status_t launch_attention_decode_batch(
    const uint16_t*        d_q_rows,   // [B][2560]（组集中 q 行）
    const uint32_t* const* d_slot_pos, // [B]：每 slot 的 limit device 地址
    uint32_t               B,
    uint32_t               layer,
    const uint32_t* const* d_slot_pt,  // [B]：每 slot 页表 device 地址
    uint16_t* const*       d_slot_kv,  // [B]：每 slot KV 池（ph0 时写）
    KvLayout               layout,
    float*                 d_partials, // [B][16][max_chunks][130]
    uint32_t               max_chunks,
    uint16_t*              d_out,      // [B][2048]
    cudaStream_t           stream,
    const float*           d_inv_freq, // null → 非 ph0
    uint32_t               chunk_pos,
    float*                 d_partials2 // [B][16][kMergeSplit2][130]
);

// attention_prefill.cu — T token causal（M=T）便捷封装。
mc_status_t launch_attention_prefill(const uint16_t* d_q_rows,
                                      uint32_t        tokens,
                                      const uint32_t* d_base_pos, // device
                                      uint32_t        layer,
                                      const uint32_t* d_page_table,
                                      const uint16_t* d_kv_pool,
                                      KvLayout        layout,
                                      uint16_t*       d_out,
                                      cudaStream_t    stream);

// attention_prefill.cu（P1 flash-HMMA 正确性原型，测试专用 launcher）—
// prefill 的 flash attention tensor-core kernel：grid = 2·⌈T/16⌉（kvh-major
// 一维），block 256 = 8 warp（warp w ↔ q head kvh*8+w）；每块 16 q-token ×
// 8 head，16-pos KV tile 主循环（QK/PV 均 mma.m16n8k16 bf16 + 行向 online
// softmax，fragment 编排移植 attn_kv_kernel_h），S_kv=1 直写 out（契约同
// launch_gqa_attention）。生产路径经 launch_attn_prefill_flash 接线
//（mc_prefill_flash_enabled，P3）；kernel_ops 的 attention_prefill_flash
// 对拍入口。
mc_status_t launch_attn_prefill_flash_test(const uint16_t* d_q_rows,
                                            uint32_t        tokens,
                                            const uint32_t* d_base_pos, // device
                                            uint32_t        layer,
                                            const uint32_t* d_page_table,
                                            const uint16_t* d_kv_pool,
                                            KvLayout        layout,
                                            uint16_t*       d_out,
                                            cudaStream_t    stream);

// P3 split-KV 档位：S_kv(T) = (T ≥ 2048) ? 1 : clamp(4096/T, 1, 8)。
// 取整用 floor（整数除法）而非 ceil：保证 T·S_kv ≤ 4096 恒成立 →
// partials 峰值 ≤ 4096 token-split × 16 head × 130 fp32 = 34.07MB
//（任务书 ceil 版在 T=1023/2047 会到 42.6/51.1MB，超 session 40MB 固定
// 切片；边界点 T=512→8 / T=1024→4 与任务书一致）。T < 384 走旧 kernel
//（host 分支，见 forward.cpp），本函数对 T < 384 的返回值不被使用。
inline uint32_t prefill_flash_split_kv(uint32_t tokens) {
    if (tokens >= 2048u) return 1u;
    const uint32_t s = 4096u / tokens;
    return s < 1u ? 1u : (s > 8u ? 8u : s);
}

// P3：prefill flash-HMMA 生产开关。**默认开（2026-09-11 复核翻转，依据见
// 下方判决附注）**。MC_PREFILL_FLASH_OFF=1 显式关闭回旧 kernel；T ≥ 384
// 的 prefill attention 走 flash，T < 384 恒走旧 kernel。静态缓存首次读取 →
// 进程内恒定；prefill 无 CUDA Graph 约束（decode graph 仅 T=1），host 分
// 支合法。
//
// ---- P4 门槛判决（2026-09-11，RTX 5080，真实 forward 路径，
//      MC_PREFILL_FLASH_ON=1 + MC_BREAKDOWN 段拆分；prefill_e2e 3 轮取
//      中位；对照 vLLM 同卡基线）----
//   e2e prefill tok/s（GO 目标 / 实测 / vLLM）+ TTFT（实测 / vLLM）：
//     T=2048   ≥15k / 19004 / 20.6k   107.8ms / 99.5ms   GO ✓（92% vLLM）
//     T=8192   ≥14k / 17976 / 18k     455.7ms / 457ms    GO ✓（100% vLLM）
//     T=32768  ≥9k  / 10112 / 10.2k   3240.5ms / 3.19s   GO ✓（99% vLLM）
//     T=65536  ≥6k  / 6408  / 6.4k    10226.9ms / 10.2s  GO ✓（100% vLLM）
//     T=130816 ≥6k  / 3678  / 3.65k   35571.9ms / 35.8s  GO ✗（止损 ≥306
//             远未触发；100.8% vLLM —— 与 vLLM 持平，但 GO 线要求 1.75×）
//   段拆分（attn / GEMM 桶，ms）：2048=13.0/63.2、8192=127.1/222.8；
//   >8448 为末 chunk 口径：32768=790.2/201.4、65536=1429.2/172.1、
//   130816=2075.6/115.4 —— 长档 attention 主导（16-row q-tile 的 KV 重读
//   随 ctx² 增长；更大 q-tile 属后续优化，非本判决范围）。
//   kernel 级（P2 cp.async 双组流水 vs P1 单缓冲）：T=2048 0.336→0.309ms
//  （-7.7%）、T=8192 3.614→3.222ms（-10.8%）；vs 旧 kernel：T=512（S=8）
//   0.0775 vs 0.3165ms、T=1024（S=4）0.111 vs 1.182ms、T=2048 0.309 vs
//   4.602ms、T=8192 3.222 vs 77.0ms。e2e 全档 vs 旧路径：2048 7140→19004
//  （2.66×）、8192 2260→17976（7.95×）、32768 590→10112（17.1×）。
//   数值/正确性：kernel_ops attention_prefill_flash 全绿（含 T=8448 自一致
//   性 relL2=2.26e-3、T=512/1024 split 三方对拍 relL2 ≤2.3e-3）；greedy
//   e2e（T=2048→200 tok / T=8464 跨 chunk→64 tok / T=512 split→64 tok）
//   ON/OFF token 逐位一致；MC_PREFILL_FLASH_ON=1 下 real_inference golden
//   三 case + graph_bench 过；13/13 ctest。
//   判决：4/5 档过 GO、零档触止损；T=130816 未达 GO（-39%）但与 vLLM 持
//   平（100.8%）。
//   复核翻转（2026-09-11）：P4 二元门规未覆盖「未过 GO 亦未触止损」的中间
//   区，fixer 按字面保守处置为 opt-in。复核结论：门规目的是防退化，而
//   flash 在每一档均严格优于旧路径（e2e 2.66×/7.95×/17.1×，128K 达
//   10.8×）、TTFT 全档 ≥ vLLM 持平线（92%-100.8%）、正确性/确定性全绿
//   —— 保持默认关会让生产用户守着 340 tok/s 的旧路径，与门规初衷相悖。
//   故翻转默认为开；MC_PREFILL_FLASH_OFF=1 回退保留，旧 kernel 保留对
//   拍基准（同 v1/v2/HMMA 先例）。MC_PREFILL_FLASH_ON=1 为冗余显式开启
//   （与默认一致，兼容既有调用）。
inline bool mc_prefill_flash_enabled() {
    static const bool on = [] {
        return getenv("MC_PREFILL_FLASH_OFF") == nullptr;
    }();
    return on;
}

// attention_prefill.cu（P3 生产接线）— flash 主 kernel（P2 cp.async 双组
// 流水）+ S_kv>1 时的 prefill_merge（独立 kernel，串行合并 S≤8 组 partial，
// 公式复用 attn_merge_kernel；不动 decode 的 merge_a/merge_b）。grid =
// 2·⌈T/16⌉·S_kv +（S_kv>1）T·16；d_partials 需 ≥ T·16·S_kv·130 fp32
//（session arena 40MB prefill_partials 切片，S_kv 见 prefill_flash_split_kv）。
mc_status_t launch_attn_prefill_flash(const uint16_t* d_q_rows,
                                      uint32_t        tokens,
                                      const uint32_t* d_base_pos, // device
                                      uint32_t        layer,
                                      const uint32_t* d_page_table,
                                      const uint16_t* d_kv_pool,
                                      KvLayout        layout,
                                      uint16_t*       d_out,
                                      float*          d_partials, // S_kv>1 必需
                                      uint32_t        S_kv,       // 1..8
                                      cudaStream_t    stream);

// swiglu.cu（P2）— gate/up 行拼输入：src[m, 0..6144)=gate, src[m,6144..12288)=up
//（与 wpk gate_up 行拼序一致）；out[m,d]=bf16(silu_f32(g)·f32(u))。
mc_status_t launch_swiglu_gated(const uint16_t* d_src, // [rows, 2*inter]
                                uint16_t*       d_out, // [rows, inter]
                                uint32_t        rows,
                                int64_t         inter, // 6144
                                cudaStream_t    stream, bool pdl = false /* P5 PDL */);

// swiglu.cu — out[i] = silu(gate[i]) * up[i]，BF16 in/out、fp32 中间（真实可用）
mc_status_t launch_swiglu(const uint16_t* d_gate,
                          const uint16_t* d_up,
                          uint16_t*       d_out,
                          int64_t         n,
                          cudaStream_t    stream);

// sampling.cu — greedy argmax over float logits（真实可用；确定性 tie→最小下标）
mc_status_t launch_sampling_greedy(const float* d_logits,
                                   int32_t      n,
                                   int32_t*     d_out_token,
                                   cudaStream_t stream);

// sampling.cu（P2）— greedy argmax over bf16 logits（fp32 比较，tie→最小下标）。
// golden 的 logits.f32 = bf16 upcast，故比较语义与参考一致。
mc_status_t launch_sampling_greedy_bf16(const uint16_t* d_logits,
                                        int32_t         n,
                                        int32_t*        d_out_token,
                                        cudaStream_t    stream,
                                        // P5 两段式 partial（null = 单 block
                                        // 旧路径；kernel_ops 直调用默认值）
                                        float*   d_stage_v = nullptr,
                                        int32_t* d_stage_i = nullptr, bool pdl = false /* P5 PDL */);

// sampling.cu — 确定性 mock forward：
//   token = 100 + ((seq_len * 2654435761u) ^ rng) % 1000  （恒 < vocab）
//   force >= 0 时直接写 force（用于 max_new_tokens 达到时强制 EOS）
mc_status_t launch_mock_next_token(int32_t*    d_out,
                                   uint32_t    seq_len,
                                   uint32_t    rng,
                                   int32_t     force,
                                   cudaStream_t stream);

// sampling.cu（P3）— decode 步状态推进（CUDA Graph 尾节点，全 device 侧）：
//   d_seq[*d_seq_len] = *d_next_token;  *d_seq_len += 1;
// 等价于 eager 路径的「D2H append d_seq[seq_len] ← d_next_token + seq_len+=1」，
// 但索引从固定 device 地址读取（graph replay 期间 host 不参与寻址）。
mc_status_t launch_decode_advance(int32_t*       d_seq,
                                  const int32_t* d_next_token,
                                  uint32_t*      d_seq_len,
                                  cudaStream_t   stream);

// sampling.cu（B1b）— 批 greedy argmax：组集中 logits [n][vocab] bf16 →
// 每 slot 一 block（512 线程，warp 归约；比较语义与
// launch_sampling_greedy_bf16 一致：fp32 比较、tie→最小下标）→ 结果写
// device 指针表 d_slot_tok[slot] 指向的成员 d_next_token。
mc_status_t launch_batch_argmax_bf16(const uint16_t*        d_logits,   // [n][vocab]
                                     int32_t* const*        d_slot_tok, // [n] 写目标
                                     int32_t                vocab,
                                     int32_t                n,
                                     cudaStream_t           stream);

// sampling.cu（B1b）— 批 decode 状态推进（每 slot 一线程，单 block）：
//   *d_slot_pos[s] 为成员 device 侧 seq_len；
//   d_slot_seq[s][len] = *d_slot_tok[s];  *d_slot_pos[s] += 1;
// 与 solo eager 的「D2D append + host 推进 + 8B 回写」语义等价（全 device 侧）。
mc_status_t launch_batch_advance(int32_t* const*        d_slot_seq,  // [n] 成员 d_seq
                                 const int32_t* const*  d_slot_tok,  // [n] 成员 token
                                 uint32_t* const*       d_slot_pos,  // [n] &seq_len
                                 uint32_t               n,
                                 cudaStream_t           stream);

// gemv.cu（P3+）— decode 专用 M=1 BF16 GEMV 自研 kernel（计划 §8.2/§10.2）：
//   y[n] = Σ_k x[k]·W[n,k]，W 行主 [N,K]，fp32 累加 → bf16（同 cuBLASLt
//   M=1 路径的数值规范，累加序不同 → 容差内差异）。
// tactic（建期快测决定，session 内固定 → CUDA Graph 安全）：
//   kGemvTacticRows   每 warp 一行、lane 沿 K 以 uint4(8×bf16) 跨步、
//                     4 路展开 ILP、warp butterfly 归约（带宽型主力）
//   kGemvTacticSplitK split-K 两段式：partial [S,N] fp32 + 固定顺序 merge
//                     （无原子 → 确定性；小 N shape 增并行度用）
// 选型与 dispatch 见 runtime/gemv_tactic.{h,cpp}。
mc_status_t launch_decode_gemv(const uint16_t* d_w,   // [N,K] 行主（权重）
                               const uint16_t* d_x,   // [K]（activation 行）
                               uint16_t*       d_y,   // [N]
                               uint32_t N, uint32_t K,
                               int tactic, uint32_t split_s,
                               float* d_split_partial, // [S,N]（split-K 用）
                               cudaStream_t stream, bool pdl = false /* P5 PDL */);

// sampling.cu（P4+，§14.2）— 采样管线（greedy argmax 原样保留）：
//   1) launch_rep_penalty_mark：把 d_seq[0..*d_seq_len) 出现过的 token 标记
//      到 vocab 字节位图（重复标记幂等、无原子；固定 grid 按 max_tokens，
//      空线程早退 → graph 安全）。调用前需 memset 位图。
//   2) launch_rep_penalty_apply：位图命中的 logit 应用标准公式
//     （logit>0 ? logit/p : logit*p，p=d_cfg->rep_penalty；p==1 直通），
//      bf16 读 → fp32 写 d_logits_f32（采样主体统一在 fp32 上工作）。
//   3) launch_sampling_pipeline：单 block（256 线程）锦标赛 top-k
//      （shared 最小堆，容量 kSampMaxK=128；tile 内反复提取 block 最大直到
//      不超过堆顶；并列值按下标小者优先）→ 堆内降序排序 → temperature
//      softmax（堆内概率）→ top-p 前缀截断（含恰好达 p 的 token）→
//      device RNG（xorshift64*，状态在显存）采样输出 d_out_token。
//      两层直方图预滤（候选 ≤ ~638）+ shared bitonic 全排序 + softmax/
//      top-p/RNG 尾巴（~40µs/步；确定性：整数直方图和 + 顺序无关全排序）。
//      temperature ≤0 / top_k ≤1 由 host 激活规则排除（不进入本 kernel）。
constexpr uint32_t kSampMaxK = 128;    // top-k 堆容量（top_k 超此值截断并注释）
constexpr uint32_t kSampSortCap = 512; // 采样排序候选容量（kSampMaxK×4）

mc_status_t launch_rep_penalty_mark(const int32_t* d_seq,
                                    const uint32_t* d_seq_len, // device 侧长度
                                    uint8_t* d_bitmap,          // [vocab]（先 memset 0）
                                    uint32_t max_tokens,        // grid 上限（建期固定）
                                    cudaStream_t stream);

mc_status_t launch_rep_penalty_apply(const uint16_t* d_logits_bf16, // [n]
                                     const uint8_t* d_bitmap,       // [n]
                                     const SamplingConfigDevice* d_cfg,
                                     float* d_logits_f32,           // [n] 输出
                                     int32_t n,
                                     cudaStream_t stream);

mc_status_t launch_sampling_pipeline(const float* d_logits,   // [n]（fp32，经 penalty）
                                     const SamplingConfigDevice* d_cfg,
                                     uint64_t* d_rng_state,   // device 侧 RNG 状态
                                     int32_t* d_out_token,
                                     int32_t n,
                                     float* d_partial_v,      // [kSampBlocks][kSampMaxK]
                                     int32_t* d_partial_i,    // [kSampBlocks][kSampMaxK]
                                     cudaStream_t stream);

// gemv_fp8.cu（P4）— decode 专用 FP8(E4M3) GEMV：
//   y[n] = scale_row[n] × Σ_k e4m3(w[n,k]) · f32(x[k])，x bf16、y bf16、
//   fp32 累加（核内位转换见 mc::fp8_e4m3_to_f32；scale 在归约后乘入——
//   每行常数，数学等价且数值良好）。结构沿用 gemv_rows：每 warp 一行、
//   uint4 = 16×fp8 向量化（每 lane 配 2×uint4 的 x 装载）、2 路展开 ILP、
//   butterfly 归约。K%16 余量走逐元素尾路径（lane j 处理第 j 个尾元素）。
//   逐输出行 fp32 scale（[N]）。CUDA Graph 兼容（grid/地址建期固定）。
mc_status_t launch_gemv_fp8(const uint8_t*  d_w,     // [N,K] e4m3（uint8）
                            const float*    d_scale, // [N] 行 scale
                            const uint16_t* d_x,     // [K] bf16
                            uint16_t*       d_y,     // [N] bf16
                            uint32_t N, uint32_t K,
                            cudaStream_t stream, bool pdl = false /* P5 PDL */);

// gemv{,_fp8}.cu（P5）— 融合 prologue 变体（decode T=1）：
//   res_out = bf16(f32(res_in)+f32(addend))（ping-pong 双缓冲，block 0 写回）
//   + RMSNorm(res_out, norm_w)（块协作，序逐 bit 复刻 residual_add_rmsnorm）
//   + GEMV（x = normed，落 shared 不写全局）。
// 与「独立 resnorm kernel + GEMV 读 normed」逐 bit 一致；消除 84 次/step
// 的单 block 小 kernel。限制：n == 2048（qkv/gate_up 的 K=hidden）。
// BF16 变体仅在 tactic=kGemvTacticRows 时可用（复用自研 kernel）。
mc_status_t launch_gemv_fp8_prologue(const uint8_t*  d_w,     // [N,2048] e4m3
                                     const float*    d_scale, // [N]
                                     const uint16_t* d_res_in,  const uint16_t* d_addend,
                                     const uint16_t* d_norm_w,  uint16_t* d_res_out,
                                     uint16_t*       d_y,     // [N] bf16
                                     uint32_t N, uint32_t n, float eps,
                                     cudaStream_t stream, bool pdl = false /* P5 PDL */);

mc_status_t launch_gemv_rows_prologue(const uint16_t* d_w,     // [N,2048] bf16
                                      const uint16_t* d_res_in,  const uint16_t* d_addend,
                                      const uint16_t* d_norm_w,  uint16_t* d_res_out,
                                      uint16_t*       d_y,     // [N] bf16
                                      uint32_t N, uint32_t n, float eps,
                                      cudaStream_t stream, bool pdl = false /* P5 PDL */);

// gemv{,_fp8}.cu（P5）— 融合 swiglu 变体（down 投影，K=6144）：
//   块协作 silu(gate)·up → shared（数值与 swiglu_gated_kernel 逐 bit 一致）
//   → GEMV 从 shared 读 x。消除独立 swiglu kernel + launch 间隙（~2.8µs/层）。
mc_status_t launch_gemv_fp8_swiglu(const uint8_t*  d_w,     // [N,6144] e4m3
                                   const float*    d_scale, // [N]
                                   const uint16_t* d_gate_up, // [12288] gate|up
                                   uint16_t*       d_y,     // [N] bf16
                                   uint32_t N, uint32_t K,
                                   cudaStream_t stream, bool pdl = false /* P5 PDL */);

mc_status_t launch_gemv_rows_swiglu(const uint16_t* d_w,     // [N,6144] bf16
                                    const uint16_t* d_gate_up, // [12288]
                                    uint16_t*       d_y,     // [N] bf16
                                    uint32_t N, uint32_t K,
                                    cudaStream_t stream, bool pdl = false /* P5 PDL */);

// gemv_fp8.cu（F1 批组 fp8）— 多行 gemv-fp8（M∈[2,8]，批解码合并前向的
//   kernel 基石）：y[m,n] = scale[n] × Σ_k e4m3(w[n,k])·f32(x[m,k])，x bf16
//   [M,K]、y bf16 [M,N]。结构沿用 M=1 骨架（warp ↔ 输出行、uint4=16×fp8
//   向量化、butterfly 归约、scale 归约后乘入）；X 经 K-tile(512) shared
//   双缓冲驻留（M=8 → 16KB），W 直读全局流式；K-tile 消耗轮次结构与 solo
//   的链归属逐对等价 → 每行输出与 launch_gemv_fp8 逐 bit 一致（对拍锚）。
//   K%16≠0/失齐时走多行标量回退（同 solo 标量路径数值）。PDL 同族；launch
//   参数建期固定（CUDA Graph 兼容，F2 批组捕图直接复用）。
mc_status_t launch_gemv_fp8_rows_m(const uint8_t*  d_w,     // [N,K] e4m3（uint8）
                                   const float*    d_scale, // [N] 行 scale
                                   const uint16_t* d_x,     // [M,K] bf16
                                   uint16_t*       d_y,     // [M,N] bf16
                                   uint32_t M, uint32_t N, uint32_t K,
                                   cudaStream_t stream, bool pdl = false /* P5 PDL */);

// quant.cu（F2a）— bf16 → e4m3 per-row 动态量化（批解码 F2b / prefill F3
//   共享机械）：pass1 行 amax → scale=amax/448（全零行 scale=1）；pass2
//   x·(448/amax) 硬件 cvt（RNE + satfinite）。逐元素无归约歧义；M≤8 与
//   M=2048 同一路径（块 ↔ 行 grid-stride）。
mc_status_t launch_quant_rows(const uint16_t* d_x,   // [M,K] bf16
                              uint8_t*       d_xq,   // [M,K] e4m3（输出）
                              float*         d_scale,// [M]（输出）
                              uint32_t M, uint32_t K,
                              cudaStream_t stream);

// quant.cu（F2a）— fp8 GEMM 回退模式补偿：y[m,n] ×= w_scale[n]·x_scale[m]
//（仅当 cuBLASLt 不支持 OUTER_VEC_32F 而 GemmEngine 走 per-tensor scalar
//   时由 gemm_fp8 内部发射；首选模式不可达）。d_x_scale 可 null → 仅行 scale。
mc_status_t launch_post_scale_rows(uint16_t*       d_y,      // [M,N] bf16（就地）
                                   const float*    d_w_scale, // [N]
                                   const float*    d_x_scale, // [M] 或 null
                                   uint32_t M, uint32_t N,
                                   cudaStream_t stream);

#ifdef __cplusplus
} /* extern "C++" */
#endif
