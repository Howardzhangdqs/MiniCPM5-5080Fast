// session.cpp — mc_session 生命周期 + mc_prefill / mc_decode_one / mc_get_stats
// ABI 入口（计划 §5.4、§7.2、§10.4、§11.1）。
//
// 分配纪律：session_create 完成全部分配（无 lazy alloc；§5.4 要求 warmup /
// 初始化不进 hot path）；steady-state（session 建成后）零 cudaMalloc /
// cudaFree / cudaHostAlloc；session_reset 复用全部内存只清状态；
// destroy 逆序释放。
//
// 统计口径：total 为生命周期累计（reset 不清零）；since_reset 记录上一轮
// reset 之后的量（reset 时清零）。mc_get_stats 返回 total。
//
// 线程安全：session 非线程安全（一个 session 是独占的可变生成状态对象）；
// model 不可变、可共享。
#include "runtime/internal_state.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>

#include <cuda_runtime.h>

#include "kernels/sm120/kernels.h" // attention_decode_max_chunks / kAttnPartialStride
#include "runtime/decode_executor.h"
#include "runtime/error.h"
#include "runtime/export.h"
#include "runtime/gemv_tactic.h"
#include "runtime/prefill_executor.h"

namespace {

// mock 模式保留的旧 scratch 占位（real 模式用 ActivationScratch，见下）。
constexpr size_t kMockScratchBytes = 4u << 20; // 4 MiB

struct SessionSizes {
    size_t total_bytes = 0;
    uint32_t max_pages = 0;
    size_t kv_pool_bytes = 0;
    size_t act_bytes = 0; // real 模式 activation scratch（bf16 中间量）
    size_t attn_partial_bytes = 0; // P3+ attention 两段式 partials（fp32）
    size_t attn_partial2_bytes = 0; // P8-A1 两级 merge level-2 partials（fp32）
    size_t prefill_partial_bytes = 0; // P3 split-KV prefill flash partials（固定 40MB）
    uint32_t attn_chunk = 0;        // P8-A2 会话 chunk 档位（64/128/256）
    uint32_t attn_max_chunks = 0;
    size_t gemv_partial_bytes = 0; // P3+ GEMV split-K partial（fp32）
    size_t sampling_bytes = 0;     // P4+ 采样位图 + fp32 logits
    size_t gemm_ws_bytes = 0;      // S1 per-session gemm workspace（real 模式 16MB）
};

// P2 activation scratch：prefill T≤max_ctx 全量中间（bf16）。
// 逐项（元素/token）：residual 2048 + normed 2048 + qkv 2560 + ctx 2048 +
// o_out 2048 + gate_up 12288 + mlp_mid 6144 + down_out 2048 = 31232
//（embedding gather 直接写 residual 流起点，无独立 emb 切片——原 32MB/8192
// token 的 emb 缓冲从未被读取，已删；含 emb 的旧值是 33280），
// 另加 logits 1 行 vocab。
constexpr uint64_t kActElemsPerToken = 5ull * 2048 + 2560 + 12288 + 6144; // 31232

SessionSizes compute_sizes(uint32_t max_ctx, bool real_mode) {
    SessionSizes sz;
    sz.max_pages = (max_ctx + PagedKvAllocator::kTokensPerPage - 1) /
                   PagedKvAllocator::kTokensPerPage;
    // kv_pool 按整页容量分配：pages × 32 token × kKvBytesPerToken。
    // 与 KvLayout::total_elems() 一致（= pages×32×512×42 元素 ×2B）；旧的
    // max_ctx × per-token 会在末页不满时少分配——kernel 仍可能寻址末页
    // 全部 32 slot，越界写穿 pool 末端。
    sz.kv_pool_bytes = (size_t)sz.max_pages * PagedKvAllocator::kTokensPerPage *
                       (size_t)mc::cfg::kKvBytesPerToken; // BF16 KV
    // P8 分块 prefill：激活 scratch 改按「单次 forward 的最大 token 数」=
    // min(max_ctx, chunk) 分配（chunk 见 prefill_executor.cpp，env
    // MC_PREFILL_CHUNK 可覆盖）。T > chunk 的 prefill 由 prefill_executor
    // 分块循环处理，单次 forward 的 T 恒 ≤ chunk；max_ctx ≤ chunk 时本式
    // 与旧口径（全量 max_ctx）逐字节相同 → 现有 vram_budget 口径不变。
    const uint32_t act_tokens = max_ctx < mc::prefill_chunk_tokens()
                                    ? max_ctx
                                    : mc::prefill_chunk_tokens();
    sz.act_bytes = real_mode
                       ? (size_t)act_tokens * kActElemsPerToken * 2u /*bf16*/
                             + (size_t)mc::cfg::kVocabSize * 2u   /*logits 行*/
                             + (size_t)mc::cfg::kHiddenSize * 2u  /*P5 residual_alt 行*/
                       : kMockScratchBytes;
    // P3+ attention 两段式 partials：16 q head × max_chunks × 130 fp32
    //（固定地址 scratch；max_chunks 由 max_ctx 建期推导，grid.y 与寻址恒定）。
    // P8-A2：chunk 按档位表选（≤16K→64 / ≤64K→128 / >64K→256）—— 长 ctx
    // 用大 chunk 压 level-1 partials（128K：CHUNK=64 → 2048 chunk 17.03MB
    // /层；CHUNK=256 → 512 chunk 4.26MB/层）。session 内恒定 → graph 安全。
    sz.attn_chunk = attention_decode_chunk_pos(max_ctx); // P9-Step3 档位表
    sz.attn_max_chunks = attention_decode_max_chunks(max_ctx, sz.attn_chunk);
    sz.attn_partial_bytes =
        real_mode ? (size_t)mc::cfg::kNumQueryHeads * sz.attn_max_chunks *
                        kAttnPartialStride * sizeof(float)
                  : 0;
    // P8-A1 两级 merge 的 level-2 partials：[16][kMergeSplit2=32][130] fp32
    //（266KB；42 层流序复用，固定地址 → graph 安全）。
    sz.attn_partial2_bytes =
        real_mode ? (size_t)mc::cfg::kNumQueryHeads * kMergeSplit2 *
                        kAttnPartialStride * sizeof(float)
                  : 0;
    // P3 split-KV prefill flash partials：[T·16·S_kv][130] fp32，S_kv =
    // prefill_flash_split_kv(T)（kernels.h）→ T·S_kv ≤ 4096 恒成立，峰值
    // 4096×16×130×4B = 34.07MB（T=512 S=8 / T=1024 S=4 / T∈(1024,2048) S=2
    // 达峰）。固定 40MB 切片封顶（含对齐 slack；42 层流序复用、固定地址；
    // flash 写 → prefill_merge 读，eager 流序 → 无 graph 约束）。
    sz.prefill_partial_bytes = real_mode ? (40ull << 20) : 0;
    // P3+ GEMV split-K partial（[S,N] fp32；口径见 gemv_tactic.h）。~80KB。
    sz.gemv_partial_bytes =
        real_mode ? (size_t)mc::gemv_partial_floats() * sizeof(float) : 0;
    // P4+ 采样：rep-penalty 位图 [vocab] u8 + penalty 后 fp32 logits [vocab]。
    sz.sampling_bytes =
        real_mode ? (size_t)mc::cfg::kVocabSize * (1u + sizeof(float)) +
                        (4u << 10) /*hist/计数/阈值头*/ +
                        (size_t)kSampSortCap * (sizeof(float) + sizeof(int32_t)) +
                        64u * sizeof(float) /*P5 rope inv_freq 表*/
                  : 0;
    // S1 per-session gemm workspace：real 模式恒 16MB —— 与 model 级
    // workspace（kSharedWorkspaceBytes）同尺寸，cublasLt heuristic 的
    // 过滤条件不变 → 同 algo → 数值逐 bit 不变（gemm.cpp pick_algo）。
    // mock 模式 0（mock forward 不做 gemm）。
    sz.gemm_ws_bytes = real_mode ? (16u << 20) : 0;
    size_t need = 0;
    need += sizeof(SessionStateDevice);
    need += (size_t)max_ctx * sizeof(int32_t);        // d_seq
    need += sizeof(int32_t);                          // d_next_token
    need += (size_t)sz.max_pages * sizeof(uint32_t);  // d_page_table
    need += sz.kv_pool_bytes;                         // paged KV pool
    need += sz.act_bytes;                             // activation scratch
    need += sz.attn_partial_bytes;  // attention partials
    need += sz.attn_partial2_bytes; // P8-A1 两级 merge level-2 partials
    need += sz.prefill_partial_bytes; // P3 split-KV prefill flash partials
    need += sz.gemv_partial_bytes;                    // gemv split-K partial
    need += sz.sampling_bytes;      // 采样位图 + fp32 logits
    need += sz.gemm_ws_bytes;       // S1 per-session gemm workspace
    need += 8 * 256;                                  // bump 对齐 slack
    sz.total_bytes = need;
    return sz;
}

// 逆序释放（与创建顺序相反）。幂等。
void release_session(SessionImpl* s) {
    if (s == nullptr) return;
    if (s->model != nullptr) (void)cudaSetDevice(s->model->device_id);
    if (s->stream != nullptr) (void)cudaStreamSynchronize(s->stream);
    // P5 分解事件（graph 在其之前销毁——事件节点引用需先卸图）
    if (s->bdown.enabled) {
        for (uint32_t i = 0; i < mc::kBdownEvents; ++i)
            if (s->bdown.events[i] != nullptr) (void)cudaEventDestroy(s->bdown.events[i]);
        s->bdown.enabled = false;
    }
    s->graph.destroy();        // P3：decode graph（先于其引用的显存释放）
    s->graph_sample.destroy(); // P4+：采样变体 graph
    if (s->h_next_token != nullptr) (void)cudaFreeHost(s->h_next_token);
    if (s->h_cfg != nullptr) (void)cudaFreeHost(s->h_cfg);
    if (s->h_tokens != nullptr) (void)cudaFreeHost(s->h_tokens);
    s->arena.destroy();
    if (s->stream != nullptr) (void)cudaStreamDestroy(s->stream);
    if (s->model != nullptr) s->model->active_sessions.fetch_sub(1);
    delete s;
}

} // namespace

extern "C" MC_API mc_status_t mc_session_create(mc_model_t model,
                                                mc_session_t* out_session) {
    try {
        mc::clear_error();
        if (out_session != nullptr) *out_session = mc_session{nullptr};
        if (model.impl == nullptr) {
            mc::set_error("mc_session_create: model is NULL");
            return MC_E_INVALID_ARGUMENT;
        }
        if (out_session == nullptr) {
            mc::set_error("mc_session_create: out_session is NULL");
            return MC_E_INVALID_ARGUMENT;
        }
        ModelImpl* modeli = model.impl;

        cudaError_t e = cudaSetDevice(modeli->device_id);
        if (e != cudaSuccess) {
            cudaGetLastError();
            mc::set_error("mc_session_create: cudaSetDevice(%d) failed: %s",
                          modeli->device_id, cudaGetErrorString(e));
            return MC_E_CUDA;
        }

        // ---- session 配额 ----
        const uint32_t n = modeli->active_sessions.fetch_add(1) + 1;
        if (n > modeli->max_sessions) {
            modeli->active_sessions.fetch_sub(1);
            mc::set_error("mc_session_create: session limit reached (max_sessions=%u)",
                          modeli->max_sessions);
            return MC_E_UNSUPPORTED;
        }

        SessionImpl* s = new SessionImpl();
        s->model = modeli;

        // ---- 独占 stream ----
        e = cudaStreamCreate(&s->stream);
        if (e != cudaSuccess) {
            cudaGetLastError();
            mc::set_error("mc_session_create: cudaStreamCreate failed: %s",
                          cudaGetErrorString(e));
            release_session(s);
            return MC_E_CUDA;
        }

        // ---- 一次性大块 device 分配（DeviceArena，唯一 cudaMalloc）----
        const SessionSizes sz = compute_sizes(modeli->max_context_tokens, modeli->has_wpk);
        mc_status_t rs = s->arena.init(sz.total_bytes, "session.arena");
        if (rs != MC_OK) {
            // P7：OOM 报文补 free/need 数字（容量规划可行动；arena 原始报文
            // 只有 need）。sample_free_vram 为纯查询，失败路径允许。
            if (rs == MC_E_OUT_OF_MEMORY) {
                uint64_t free_now = 0;
                if (modeli->sample_free_vram(free_now) == MC_OK)
                    mc::set_error(
                        "mc_session_create: session arena allocation failed: "
                        "need %zu bytes but free VRAM is %llu bytes "
                        "(max_context_tokens=%u; weights are model-shared, "
                        "session arenas are not — reduce max_context_tokens "
                        "or max_sessions)",
                        sz.total_bytes, (unsigned long long)free_now,
                        modeli->max_context_tokens);
            }
            release_session(s);
            return rs;
        }
        s->d_state      = s->arena.alloc<SessionStateDevice>(1);
        s->d_seq        = s->arena.alloc<int32_t>(modeli->max_context_tokens);
        s->d_next_token = s->arena.alloc<int32_t>(1);
        s->d_page_table = s->arena.alloc<uint32_t>(sz.max_pages);
        s->kv_pool      = s->arena.alloc(sz.kv_pool_bytes, 256);
        // scratch 一次 alloc 覆盖整个 carve 范围（act + attn partials +
        // gemv partial + 采样段）：compute_sizes 的 need 已分别计入这些
        // 字节，此处合并分配使 bump 记账与下方 has_wpk 切分的实际写入
        // 范围一致（mock 模式附加段全为 0，大小不变）。
        const size_t scratch_bytes = sz.act_bytes + sz.attn_partial_bytes +
                                     sz.attn_partial2_bytes +
                                     sz.prefill_partial_bytes +
                                     sz.gemv_partial_bytes + sz.sampling_bytes;
        s->scratch      = s->arena.alloc(scratch_bytes, 256);
        // S1 per-session gemm workspace：并入同一 carve 序列（need 已计入，
        // bump 记账一致），独立 256 对齐切片 —— 不并进 scratch 合并块内部，
        // 避免 has_wpk 段切分尾部的对齐推导（arena.alloc 的 align 参数直接
        // 保证 256B）。mock 模式 0 字节：alloc(0) 返回 nullptr，字段保持空。
        s->gemm_ws      = sz.gemm_ws_bytes != 0
                              ? s->arena.alloc(sz.gemm_ws_bytes, 256)
                              : nullptr;
        s->gemm_ws_bytes = sz.gemm_ws_bytes;
        if (s->d_state == nullptr || s->d_seq == nullptr || s->d_next_token == nullptr ||
            s->d_page_table == nullptr || s->kv_pool == nullptr || s->scratch == nullptr ||
            (sz.gemm_ws_bytes != 0 && s->gemm_ws == nullptr)) {
            mc::set_error("mc_session_create: session arena exhausted (need %zu bytes)",
                          sz.total_bytes);
            release_session(s);
            return MC_E_OUT_OF_MEMORY;
        }
        if (modeli->has_wpk) {
            // P2 activation scratch 切分（固定 buffer，steady-state 零分配）。
            // embedding gather 直接写 residual，无 emb 切片。
            // P8：行数与 compute_sizes 同口径 = min(max_context_tokens,
            // prefill chunk)——分块后单次 forward 的 T 恒不超过此值。
            const uint32_t T = modeli->max_context_tokens <
                                       mc::prefill_chunk_tokens()
                                   ? modeli->max_context_tokens
                                   : mc::prefill_chunk_tokens();
            auto& a = s->act;
            uint16_t* p = (uint16_t*)s->scratch;
            a.residual = p;                                  p += (size_t)T * 2048;
            a.normed   = p;                                  p += (size_t)T * 2048;
            a.qkv      = p;                                  p += (size_t)T * 2560;
            a.ctx      = p;                                  p += (size_t)T * 2048;
            a.o_out    = p;                                  p += (size_t)T * 2048;
            a.gate_up  = p;                                  p += (size_t)T * 12288;
            a.mlp_mid  = p;                                  p += (size_t)T * 6144;
            a.down_out = p;                                  p += (size_t)T * 2048;
            a.logits   = p;                                  p += (size_t)mc::cfg::kVocabSize;
            a.residual_alt = p;                              p += (size_t)mc::cfg::kHiddenSize;
            // P3+ attention 两段式 partials（fp32）：前面全部 bf16 切片的字节
            // 数均为 4 的倍数（元素数×2，元素数为偶数）且 arena 基址 256B
            // 对齐 → 此处 float 指针天然 4B 对齐。
            a.attn_partials = reinterpret_cast<float*>(p);
            p += (size_t)mc::cfg::kNumQueryHeads * sz.attn_max_chunks *
                 kAttnPartialStride * 2u /*fp32 → uint16 单位*/;
            s->attn_max_chunks = sz.attn_max_chunks;
            // P8-A1 两级 merge level-2 partials（对齐同上：前面切片均为
            // 4B 倍数）。attn_partials 尾 = 16×chunks×130×4B 恒 4 的倍数。
            a.attn_partials2 = reinterpret_cast<float*>(p);
            p += (size_t)mc::cfg::kNumQueryHeads * kMergeSplit2 *
                 kAttnPartialStride * 2u /*fp32 → uint16 单位*/;
            // P3 split-KV prefill flash partials（[T·16·S_kv][130] fp32，
            // 40MB 固定切片；attn_partials2 尾 16×32×130×4B 为 4B 倍数 →
            // 对齐天然满足）。flash 写 → prefill_merge 读，42 层流序复用。
            a.prefill_partials = reinterpret_cast<float*>(p);
            p += (40ull << 20) / 2u; /*字节 → uint16 单位*/
            s->attn_chunk_pos = sz.attn_chunk;
            a.gemv_partial = reinterpret_cast<float*>(p);
            p += (size_t)mc::gemv_partial_floats() * 2u /*fp32 → uint16 单位*/;
            // P4+ 采样：位图（u8）+ fp32 logits。前置全部切片均为 4B 倍数
            //（gemv_partial 尾 = 8×2560×4B）→ 对齐天然满足。
            a.rep_bitmap = reinterpret_cast<uint8_t*>(p);
            p += (size_t)mc::cfg::kVocabSize / 2u; // u16 单位（vocab 偶数）
            a.logits_f32 = reinterpret_cast<float*>(p);
            p += (size_t)mc::cfg::kVocabSize * 2u; // fp32 → u16 单位
            // 采样 scratch（字节布局见 launch_sampling_pipeline）：
            //   [0,4KB) 头（hist1/hist2/count/thr16/thr8）+ list_v[cap] +
            //   list_i[cap]，共 4096 + cap*8 字节。
            // 回归锚（2026-09 采样持久污染事故）：旧切分只把 p 推进
            // 4096 + cap*4 字节并把 rope 表放在该处 —— 与 launcher 实际
            // 寻址的 list_i 段 [4096+cap*4, 4096+cap*8) 完全重叠，每个
            // 采样步的候选写穿 RoPE inv_freq 表（reset 不重写 → 持久污染）。
            a.samp_partial_v = reinterpret_cast<float*>(p);
            a.samp_partial_i = reinterpret_cast<int32_t*>(
                p + 2048 + (size_t)kSampSortCap * 2u); // u16 单位：4KB 头+list_v
            p += 2048 + (size_t)kSampSortCap * 4u; // u16 单位：4KB+cap*8 字节
            // P5：RoPE inv_freq 表（64 fp32；建期一次写入，热路径只读）
            a.rope_inv_freq = reinterpret_cast<float*>(p);
            p += 64u * 2u; // fp32 → u16 单位
        }
        s->arena.begin_steady_state(); // 此后任何分配都是纪律违规（§7.2）

        // ---- pinned host buffer（建期 cudaHostAlloc，之后只读写内容）----
        e = cudaHostAlloc((void**)&s->h_tokens,
                          (size_t)modeli->max_context_tokens * sizeof(int32_t),
                          cudaHostAllocDefault);
        if (e != cudaSuccess) {
            cudaGetLastError();
            mc::set_error("mc_session_create: cudaHostAlloc(h_tokens, %zu bytes) "
                          "failed: %s",
                          (size_t)modeli->max_context_tokens * sizeof(int32_t),
                          cudaGetErrorString(e));
            release_session(s);
            return (e == cudaErrorMemoryAllocation) ? MC_E_OUT_OF_MEMORY : MC_E_CUDA;
        }
        e = cudaHostAlloc((void**)&s->h_next_token, sizeof(int32_t),
                          cudaHostAllocDefault);
        if (e != cudaSuccess) {
            cudaGetLastError();
            mc::set_error("mc_session_create: cudaHostAlloc(h_next_token) failed: %s",
                          cudaGetErrorString(e));
            release_session(s);
            return (e == cudaErrorMemoryAllocation) ? MC_E_OUT_OF_MEMORY : MC_E_CUDA;
        }
        // P4+：采样配置 pinned 暂存（decode_one 每步 H2D 到 &d_state->cfg）
        e = cudaHostAlloc((void**)&s->h_cfg, sizeof(SamplingConfigDevice),
                          cudaHostAllocDefault);
        if (e != cudaSuccess) {
            cudaGetLastError();
            mc::set_error("mc_session_create: cudaHostAlloc(h_cfg) failed: %s",
                          cudaGetErrorString(e));
            release_session(s);
            return (e == cudaErrorMemoryAllocation) ? MC_E_OUT_OF_MEMORY : MC_E_CUDA;
        }

        // ---- KV page table（容量一次性预留：device buffer + host shadow）----
        rs = s->kv.init(s->d_page_table, sz.max_pages);
        if (rs != MC_OK) {
            release_session(s);
            return rs;
        }

        // ---- 初始清零（建期一次性，reset 不重复）+ 同步 ----
        (void)cudaMemsetAsync(s->d_state, 0, sizeof(SessionStateDevice), s->stream);
        (void)cudaMemsetAsync(s->d_seq, 0,
                              (size_t)modeli->max_context_tokens * sizeof(int32_t),
                              s->stream);
        (void)cudaMemsetAsync(s->d_next_token, 0, sizeof(int32_t), s->stream);
        (void)cudaMemsetAsync(s->d_page_table, 0,
                              (size_t)sz.max_pages * sizeof(uint32_t), s->stream);
        e = cudaStreamSynchronize(s->stream);
        if (e != cudaSuccess) {
            cudaGetLastError();
            mc::set_error("mc_session_create: init sync failed: %s",
                          cudaGetErrorString(e));
            release_session(s);
            return MC_E_CUDA;
        }

        // ---- 状态初始化 ----
        s->seq_len = 0;
        s->rng_seeded = false;
        s->rng_state = 0;
        s->dev_rng_seeded = false;
        s->dev_rng_state = 0;
        s->sampling_step = false;
        if (s->h_cfg != nullptr) memset(s->h_cfg, 0, sizeof(SamplingConfigDevice));
        s->h_state = SessionStateDevice{};
        s->total = SessionStats{};
        s->since_reset = SessionStats{};
        uint64_t free_now = 0;
        rs = modeli->sample_free_vram(free_now);
        if (rs != MC_OK) {
            release_session(s);
            return rs;
        }
        s->min_free_vram = free_now;
        if (modeli->reserve_vram_bytes != 0 && free_now < modeli->reserve_vram_bytes) {
            mc::set_error("mc_session_create: free VRAM %llu bytes is below "
                          "reserve_vram_bytes %llu",
                          (unsigned long long)free_now,
                          (unsigned long long)modeli->reserve_vram_bytes);
            release_session(s);
            return MC_E_OUT_OF_MEMORY;
        }

        fprintf(stderr,
                "[minicpm-native] session_create: arena=%zu bytes (kv_pool=%zu bytes, "
                "%u pages x %u tokens, act_scratch=%zu bytes, attn_chunk=%u "
                "x %u chunks=%zuB + l2=%zuB), pinned=%zu bytes, forward=%s\n",
                sz.total_bytes, sz.kv_pool_bytes, sz.max_pages,
                PagedKvAllocator::kTokensPerPage, sz.act_bytes, sz.attn_chunk,
                sz.attn_max_chunks, sz.attn_partial_bytes, sz.attn_partial2_bytes,
                (size_t)modeli->max_context_tokens * sizeof(int32_t) + sizeof(int32_t),
                modeli->has_wpk ? "REAL" : "MOCK");

        // ---- P5：RoPE inv_freq 表建期初始化（真实模式；内容此后只读）----
        if (modeli->has_wpk) {
            rs = launch_rope_inv_freq_init(s->act.rope_inv_freq,
                                           mc::cfg::kRopeTheta, s->stream);
            if (rs != MC_OK) {
                release_session(s);
                return rs;
            }
            // 回归锚快照：位级记录建期内容（见 internal_state.h 注释；
            // mc_debug_session_rope_intact 的对照源）。
            e = cudaMemcpyAsync(s->rope_snapshot, s->act.rope_inv_freq,
                                sizeof(s->rope_snapshot), cudaMemcpyDeviceToHost,
                                s->stream);
            if (e == cudaSuccess) e = cudaStreamSynchronize(s->stream);
            if (e != cudaSuccess) {
                cudaGetLastError();
                mc::set_error("mc_session_create: rope snapshot D2H failed: %s",
                              cudaGetErrorString(e));
                release_session(s);
                return MC_E_CUDA;
            }
            s->rope_snapshot_valid = true;
        }

        // ---- GEMM 预热（真实模式）：decode 稳态 5 个 M=1 shape 的 plan +
        //      heuristic 在建期完成，首个 decode step 零 heuristic/零 host
        //      分配（§7.2）；已缓存则跳过（多 session 幂等）----
        if (modeli->has_wpk) {
            rs = modeli->gemm.prewarm_decode_shapes(modeli->gemm_ws_bytes);
            if (rs != MC_OK) {
                release_session(s);
                return rs;
            }
        }

        // ---- P3+：decode GEMV tactic 建期快测（§8.2/§10.2）----
        // 5 个 M=1 shape 逐一对比 cuBLASLt / 自研 rows / split-K，赢家
        // 缓存于 model（进程内一次）。失败保底 cuBLASLt（tactic 初值全 0），
        // 不让 session 失败。必须在 graph capture 之前（capture 记录的
        // kernel 即 tactic 赢家，replay 无 host 决策）。
#if !MC_NATIVE_MOCK_FORWARD
        if (modeli->has_wpk) {
            (void)mc::select_decode_gemv_tactics(*modeli, *s);
            mc::clear_error();
        }
#endif

        // ---- P5 perf 分解事件（MC_BREAKDOWN=1；graph capture 之前创建，
        //      capture 期的 cudaEventRecord 成为图中 event record 节点，
        //      replay 后由 mc_debug_breakdown_collect 读相邻事件差）----
        if (modeli->has_wpk && getenv("MC_BREAKDOWN") != nullptr) {
            bool ok = true;
            for (uint32_t i = 0; i < mc::kBdownEvents && ok; ++i)
                ok = cudaEventCreate(&s->bdown.events[i]) == cudaSuccess;
            if (ok) {
                s->bdown.enabled = true;
                s->bdown.n_used = 0;
            } else {
                // 部分创建失败：全部清退，静默禁用（分解是诊断工具，
                // 不让 session 失败）
                for (uint32_t i = 0; i < mc::kBdownEvents; ++i) {
                    if (s->bdown.events[i] != nullptr)
                        (void)cudaEventDestroy(s->bdown.events[i]);
                    s->bdown.events[i] = nullptr;
                }
                cudaGetLastError();
            }
        }

        // ---- P3：decode CUDA Graph capture（§12.3，prewarm 之后）----
        // warmup → Global capture（D2H + 完整 decode step）→ Instantiate →
        // 恢复干净初始状态。失败只记日志并回退 eager：session 不失效
        //（§12.4 Fallback；enable_cuda_graph=1 的 ABI 语义是「尽力而为」）。
#if !MC_NATIVE_MOCK_FORWARD
        if (modeli->enable_cuda_graph && modeli->has_wpk) {
            rs = mc::capture_decode_graph(*s);
            if (rs != MC_OK) {
                fprintf(stderr,
                        "[minicpm-native] cuda graph capture failed (%s) -- decode "
                        "falls back to eager mode\n",
                        mc_last_error());
                mc::clear_error();
                s->graph_mode = false; // graph.available() 已为 false
            }
        }
#endif

        *out_session = mc_session{s}; // handle 包装 impl 指针
        return MC_OK;
    } catch (...) {
        mc::set_error("mc_session_create: internal error (uncaught exception)");
        return MC_E_INTERNAL;
    }
}

extern "C" MC_API mc_status_t mc_session_reset(mc_session_t session) {
    try {
        mc::clear_error();
        if (session.impl == nullptr) {
            mc::set_error("mc_session_reset: session is NULL");
            return MC_E_INVALID_ARGUMENT;
        }
        SessionImpl* s = session.impl;
        (void)cudaSetDevice(s->model->device_id);

        // 复用全部内存，只清状态（§5.4）：
        s->kv.release_all();   // page ownership 清空，显存不释放
        s->seq_len = 0;
        s->rng_seeded = false; // RNG 由下一轮 decode 的 config->seed 重播种
        s->dev_rng_seeded = false; // device PCG 同理（首轮 decode 重播种）
        s->sampling_step = false;
        s->debug_step = 0;     // debug dump 步号回到 step0
        s->since_reset = SessionStats{}; // per-round 统计清零
        // total 保留累计（见文件头统计口径注释）。

        s->h_state = SessionStateDevice{};
        (void)cudaMemsetAsync(s->d_state, 0, sizeof(SessionStateDevice), s->stream);
        s->cfg_dirty = true; // device cfg 已清零：下次 decode 必重写
        cudaError_t e = cudaStreamSynchronize(s->stream);
        if (e != cudaSuccess) {
            cudaGetLastError();
            mc::set_error("mc_session_reset: sync failed: %s", cudaGetErrorString(e));
            return MC_E_CUDA;
        }
        return MC_OK;
    } catch (...) {
        mc::set_error("mc_session_reset: internal error (uncaught exception)");
        return MC_E_INTERNAL;
    }
}

extern "C" MC_API void mc_session_destroy(mc_session_t session) {
    if (session.impl == nullptr) return; // 防御：null 安全 no-op
    release_session(session.impl);       // 逆序释放
}

extern "C" MC_API mc_status_t mc_prefill(mc_session_t session,
                                         const int32_t* token_ids,
                                         uint32_t token_count) {
    try {
        mc::clear_error();
        if (session.impl == nullptr) {
            mc::set_error("mc_prefill: session is NULL");
            return MC_E_INVALID_ARGUMENT;
        }
        cudaError_t e = cudaSetDevice(session.impl->model->device_id);
        if (e != cudaSuccess) {
            cudaGetLastError();
            mc::set_error("mc_prefill: cudaSetDevice failed: %s",
                          cudaGetErrorString(e));
            return MC_E_CUDA;
        }
        return mc::prefill_run(*session.impl, token_ids, token_count);
    } catch (...) {
        mc::set_error("mc_prefill: internal error (uncaught exception)");
        return MC_E_INTERNAL;
    }
}

extern "C" MC_API mc_status_t mc_decode_one(mc_session_t session,
                                            const mc_generation_config_t* config,
                                            int32_t* out_token) {
    try {
        mc::clear_error();
        if (session.impl == nullptr) {
            mc::set_error("mc_decode_one: session is NULL");
            return MC_E_INVALID_ARGUMENT;
        }
        cudaError_t e = cudaSetDevice(session.impl->model->device_id);
        if (e != cudaSuccess) {
            cudaGetLastError();
            mc::set_error("mc_decode_one: cudaSetDevice failed: %s",
                          cudaGetErrorString(e));
            return MC_E_CUDA;
        }
        return mc::decode_one_run(*session.impl, config, out_token);
    } catch (...) {
        mc::set_error("mc_decode_one: internal error (uncaught exception)");
        return MC_E_INTERNAL;
    }
}

extern "C" MC_API mc_status_t mc_get_stats(mc_session_t session,
                                           mc_runtime_stats_t* out_stats) {
    try {
        mc::clear_error();
        if (session.impl == nullptr) {
            mc::set_error("mc_get_stats: session is NULL");
            return MC_E_INVALID_ARGUMENT;
        }
        if (out_stats == nullptr) {
            mc::set_error("mc_get_stats: out_stats is NULL");
            return MC_E_INVALID_ARGUMENT;
        }
        SessionImpl* s = session.impl;

        // cudaMemGetInfo 是查询（非分配），steady-state 允许；顺手刷新峰值。
        uint64_t free_now = 0;
        mc_status_t rs = s->model->sample_free_vram(free_now);
        if (rs != MC_OK) return rs;
        if (free_now < s->min_free_vram) s->min_free_vram = free_now;

        out_stats->prompt_tokens = s->total.prompt_tokens;
        out_stats->generated_tokens = s->total.generated_tokens;
        out_stats->prefill_us = s->total.prefill_us;
        out_stats->decode_us = s->total.decode_us;
        // peak 估算：进程级 total - 观测到的最小 free（多 session 时为上界近似）。
        out_stats->peak_vram_bytes = s->model->total_vram - s->min_free_vram;
        out_stats->cuda_graph_launches = s->graph.launches(); // P3：graph replay 计数
        return MC_OK;
    } catch (...) {
        mc::set_error("mc_get_stats: internal error (uncaught exception)");
        return MC_E_INTERNAL;
    }
}

// ---- P3 additive debug 钩子（不属于冻结 ABI；同 mc_debug_dump_set_steps）----

// 强制开/关该 session 的 graph replay 路径（A/B 基准用；graph 本身仍在
// session_create 期 capture，只是 decode 分发回 eager）。仅在 graph 已
// instantiate 时可开启。
extern "C" MC_API void mc_debug_session_set_graph_enabled(mc_session_t session,
                                                          int enabled) {
    if (session.impl == nullptr) return;
    SessionImpl* s = session.impl;
    s->graph_mode = (enabled != 0) && s->graph.available();
}

// graph 节点数（cudaGraphGetNodes；未 capture 返回 0）。报告/基准用。
extern "C" MC_API size_t mc_debug_session_graph_nodes(mc_session_t session) {
    if (session.impl == nullptr) return 0;
    return session.impl->graph.node_count();
}

// P7：session arena 字节数（建期一次性分配的总量；VRAM 预算表/容量规划用，
// benchmark/results/vram_budget.json 的数据源之一）。
extern "C" MC_API size_t mc_debug_session_arena_bytes(mc_session_t session) {
    if (session.impl == nullptr) return 0;
    return session.impl->arena.capacity();
}

// P8-A2：本会话的 decode attention chunk 档位（建期按 max_context_tokens
// 选定：≤16K→64 / ≤64K→128 / >64K→256）。additive debug 钩子：不属于冻结
// ABI（perf_v2 sweep 记录各档实际 CHUNK 用）。
extern "C" MC_API uint32_t mc_debug_session_attn_chunk(mc_session_t session) {
    if (session.impl == nullptr) return 0;
    return session.impl->attn_chunk_pos;
}

// 回归钩子（2026-09 采样持久污染事故）：RoPE inv_freq 表是否仍与建期快照
// 位级一致（1=完好，0=被写穿；-1=会话无表/无快照，-2=CUDA 错）。调用前置：
// 上一 mc_decode_one/mc_prefill 已返回（其内部已同步 session stream）。
// additive debug 钩子：不属于冻结 ABI。
extern "C" MC_API int mc_debug_session_rope_intact(mc_session_t session) {
    if (session.impl == nullptr) return -1;
    SessionImpl* s = session.impl;
    if (!s->rope_snapshot_valid || s->act.rope_inv_freq == nullptr) return -1;
    (void)cudaSetDevice(s->model->device_id);
    uint32_t cur[64];
    const cudaError_t e = cudaMemcpy(cur, s->act.rope_inv_freq, sizeof(cur),
                                     cudaMemcpyDeviceToHost);
    if (e != cudaSuccess) {
        cudaGetLastError();
        return -2;
    }
    return memcmp(cur, s->rope_snapshot, sizeof(cur)) == 0 ? 1 : 0;
}

// ---- P5 perf 分解钩子（MC_BREAKDOWN=1 的 session；诊断/基准专用）----

// 原始段数（相邻事件对 = 段；事件布局见 internal_state.h）。未启用返回 0。
extern "C" MC_API int mc_debug_breakdown_raw_segments(mc_session_t session) {
    if (session.impl == nullptr || !session.impl->bdown.enabled) return 0;
    return (int)mc::kBdownRawSegs;
}

// 聚合桶数与段名（层数学固定 → 常量表；i 越界返回 nullptr）。
extern "C" MC_API int mc_debug_breakdown_bucket_count() {
    return (int)mc::kBdownSegs;
}

extern "C" MC_API const char* mc_debug_breakdown_bucket_name(int i) {
    if (i < 0 || (uint32_t)i >= mc::kBdownSegs) return nullptr;
    return mc::kBdownSegNames[i];
}

// 读取最近一次 forward 的 297 个原始段时长（ms）到 out[0..n)。
// 返回写入数；0 = 未启用；-1 = 参数不符；-2 = 上次 forward 非完整链
//（分窗加载/截断 —— 事件数不足，调用方应跳过该步）；-3 = CUDA 错。
// 前置：调用前 stream 已同步（decode_one 返回即满足）。
extern "C" MC_API int mc_debug_breakdown_collect(mc_session_t session,
                                                 double* out_ms, int n) {
    if (session.impl == nullptr) return -1;
    SessionImpl* s = session.impl;
    if (!s->bdown.enabled) return 0;
    if (out_ms == nullptr || n != (int)mc::kBdownRawSegs) return -1;
    if (s->bdown.n_used != mc::kBdownEvents) return -2;
    (void)cudaSetDevice(s->model->device_id);
    for (uint32_t k = 0; k < mc::kBdownRawSegs; ++k) {
        float ms = 0.f;
        if (cudaEventElapsedTime(&ms, s->bdown.events[k],
                                 s->bdown.events[k + 1]) != cudaSuccess) {
            cudaGetLastError();
            return -3;
        }
        out_ms[k] = (double)ms;
    }
    return (int)mc::kBdownRawSegs;
}
