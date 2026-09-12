// batch_group.cpp — B1b：批组运行时（多 session 合并解码，路径 B）。
//
// 职责：
//   - BatchGroupImpl：组 stream + 组 arena（一次性 carve，steady-state 零分配）
//     + device 指针表（固定地址；成员私有量按 slot 索引）+ 停车槽（B2 padding
//     用，B1 eager 精确 n 不使用）+ pinned 收割/表暂存缓冲；
//   - add/remove 的成员流↔组流单向事件链（跨流可见性）；
//   - mc_batch_step：eager 精确 n 的批 decode 前向
//     （emb 间接 gather → 42 层 {rmsnorm → qkv GEMM → 批 attention(含 ph0
//     KV append) → o GEMM → resnorm → gate_up GEMM → swiglu → down GEMM →
//     resnorm(next/final)} → lm_head → 批 argmax → 批 advance）+ sync +
//     pinned 收割。发放语义沿用 decode_one 预决策：每步返回各成员「上一步
//     已决策」token。
//
// 指针表两族（地址均建期固定、内容运行时可变 → graph 友好）：
//   - 成员表 d_slot_*[max_slots]：slot ↔ 成员绑定（add 写入 / remove 回指
//     停车槽）。B2 全桶（B=max_slots）图捕获/回放直接消费本表。
//   - 步表 d_step_*[max_slots]：本步 slot_ids[0..n) 的压缩视图 —— 批 kernel
//     的 slot 维是 [0..n) 连续 blockIdx，eager 精确 n 用步表寻址成员私有量
//     （每步 5×n×8B 小 H2D，pinned 暂存源）。
//
// 纪律与不变量：
//   - 生产 batch-1 路径（forward.cpp/decode_executor）零改动；本文件是旁路。
//   - 组操作全部来自调用者单线程；成员在组内期间禁止并发 solo 使用（add 校
//     验忙态 s->batch_group）。
//   - 数值：同 batch 组成下确定（全部 kernel 无 atomics；cuBLASLt 计划按
//     (M,N,K) 缓存、ws 恒 16MB → 同 algo）。
//   - 统计：decode_us 归组累计（B3 再细分成员归属）；成员 seq_len 的 host
//     侧镜像每步同步（solo 调用方读到的 seq_len 恒等于 device 侧值）。
//   - KV append 职责：launch_attention_decode_batch 的 ph0（d_inv_freq 非
//     null）在 kernel 内完成当前位置 K/V 写池（B1a kernel 与单 session
//     attn_kv_kernel_h 共享内联体）；页表 append 由 host 侧
//     kv.ensure_pages_for(seq_len+1)（组流上的 H2D）负责 —— 与 solo eager
//     的分工完全一致。
#include "runtime/internal_state.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <new>
#include <vector>

#include <cuda_runtime.h>

#include "kernels/sm120/kernels.h"
#include "runtime/error.h"
#include "runtime/export.h"
#include "runtime/generated_model_config.h"

namespace {

constexpr uint32_t kMaxBatchSlots = 8; // ABI 上限（mc_batch_options_t）

// ---- MC_BATCH_BREAKDOWN=1 的粗粒度段拆分（事件插桩，默认关闭不进热路径）----
// 段布局（相邻事件对）：[start] emb | 42×{ qkv | attn | o | resnorm2 | gate_up
// | swiglu | down | resnorm(next) } | lm_head | tail(argmax+advance)。
// 桶聚合（collect 钩子输出）：emb / gemm(qkv+o+gate_up+down+lm_head) / attn /
// norm(resnorm×2) / swiglu / tail。
constexpr uint32_t kBdLayerSegs = 8;
constexpr uint32_t kBdEvents = 2 + mc::cfg::kNumLayers * kBdLayerSegs + 2;
constexpr uint32_t kBdBuckets = 6;
inline const char* const kBdNames[kBdBuckets] = {
    "emb", "gemm", "attn", "norm", "swiglu", "tail(argmax+advance)"};

} // namespace

// 全局作用域（与 internal_state.h 的前置声明 + SessionImpl::batch_group 指针
// 类型匹配；辅助函数留在匿名 namespace）。
struct BatchGroupImpl {
    // ---- 建期固定，此后只读 ----
    ModelImpl*   model = nullptr;
    uint32_t     max_slots = 0;
    uint32_t     max_context_tokens = 0;
    uint32_t     attn_chunk = 0;      // 组档位（= attention_decode_chunk_pos）
    uint32_t     attn_max_chunks = 0; // partials 的 grid.y 寻址上限
    KvLayout     layout{};            // 成员同 ctx 档 → 布局恒等
    cudaStream_t stream = nullptr;    // 组独占 stream

    DeviceArena arena; // 组集中量一次性大块

    // 组集中激活（bf16，[max_slots] 行；元素口径同 session.cpp kActElemsPerToken）
    uint16_t* residual = nullptr; // [max][2048]（residual stream；gather 起点）
    uint16_t* normed   = nullptr; // [max][2048]
    uint16_t* qkv      = nullptr; // [max][2560]
    uint16_t* ctx      = nullptr; // [max][2048]
    uint16_t* o_out    = nullptr; // [max][2048]
    uint16_t* gate_up  = nullptr; // [max][12288]
    uint16_t* mlp_mid  = nullptr; // [max][6144]
    uint16_t* down_out = nullptr; // [max][2048]
    uint16_t* logits   = nullptr; // [max][130560]（lm_head 输出，bf16）
    // attention 两段式 partials（fp32；布局同 session，加 slot 维）
    float* partials  = nullptr; // [max][16][max_chunks][130]
    float* partials2 = nullptr; // [max][16][kMergeSplit2][130]
    // 组 gemm workspace：恒 16MB（与 session 级同尺寸 → heuristic 过滤条件
    // 不变 → 同 algo → 数值口径一致）
    void*  gemm_ws = nullptr;
    size_t gemm_ws_bytes = 0;
    // 组自有 RoPE inv_freq 表（内容与 session 表逐位一致；建期一次写入）
    float* rope_inv_freq = nullptr;

    // 成员表（slot ↔ 成员绑定；add 写 / remove 回指停车槽）
    uint32_t* const* d_slot_pos = nullptr; // → &成员 d_state->seq_len（limit）
    const uint32_t** d_slot_pt  = nullptr; // → 成员 d_page_table
    uint16_t* const* d_slot_kv  = nullptr; // → 成员 kv_pool（ph0 时写）
    int32_t* const*  d_slot_tok = nullptr; // → 成员 d_next_token（读/写）
    int32_t* const*  d_slot_seq = nullptr; // → 成员 d_seq（advance 追加）
    // 步表（本步 slot_ids 的 [0..n) 压缩视图；kernel 寻址入口）
    uint32_t* const* d_step_pos = nullptr;
    const uint32_t** d_step_pt  = nullptr;
    uint16_t* const* d_step_kv  = nullptr;
    int32_t* const*  d_step_tok = nullptr;
    int32_t* const*  d_step_seq = nullptr;

    // ---- 停车槽（B2 padding 用；B1 eager 精确 n 不使用）----
    // 未占用槽/被移除槽的成员表项指向这里：ph0 会把 K/V 写进 dummy 池
    //（1 页 × 42 层），limit=0 → 只触及 dummy 页第 0 位，结果无人消费。
    // B2 若对停车槽跑 advance：dummy_seq 有 64 int 余量（接线时复核上限）。
    uint32_t* dummy_pos = nullptr; // 值 0
    int32_t*  dummy_tok = nullptr; // 值 0
    uint32_t* dummy_pt  = nullptr; // [1] = {0}
    uint16_t* dummy_kv  = nullptr; // 1 页 × 42 层
    int32_t*  dummy_seq = nullptr; // [64]

    // pinned（建期 cudaHostAlloc，steady-state 只读写内容）
    int32_t* h_tokens = nullptr; // [max_slots] 收割缓冲
    void**   h_stage  = nullptr; // [5][max_slots] 指针表/步表 H2D 暂存

    // ---- 成员簿记 ----
    SessionImpl* members[kMaxBatchSlots] = {};
    cudaEvent_t  ev_member[kMaxBatchSlots] = {}; // add：成员流 record → 组流 wait
    cudaEvent_t  ev_group = nullptr;             // remove：组流 record → 成员流 wait
                                               //（host 串行 enqueue，可复用单个事件）

    // 段拆分（默认关闭）
    bool        bd_enabled = false;
    uint32_t    bd_used = 0;
    cudaEvent_t bd_ev[kBdEvents] = {};

    // 统计（B1 归组；成员归属细分是 B3）
    uint64_t steps = 0;
    uint64_t decode_us = 0;
};

// 指针表行偏移（write_slot_pointers 的 5 项统一序：pos/pt/kv/tok/seq）
constexpr uint32_t kRowPos = 0, kRowPt = 1, kRowKv = 2, kRowTok = 3, kRowSeq = 4;

namespace {

// 建期把 5 张成员表的第 slot 项指向给定成员（或停车槽）。
// host 栈暂存 → 组流 async H2D（5×8B，目标地址固定 → graph 友好）。
mc_status_t write_slot_pointers(BatchGroupImpl& g, uint32_t slot,
                                const SessionImpl* s) {
    void* p[5];
    if (s != nullptr) {
        p[kRowPos] = (void*)&s->d_state->seq_len;
        p[kRowPt]  = (void*)s->d_page_table;
        p[kRowKv]  = s->kv_pool;
        p[kRowTok] = (void*)s->d_next_token;
        p[kRowSeq] = (void*)s->d_seq;
    } else { // 停车槽
        p[kRowPos] = (void*)g.dummy_pos;
        p[kRowPt]  = (void*)g.dummy_pt;
        p[kRowKv]  = (void*)g.dummy_kv;
        p[kRowTok] = (void*)g.dummy_tok;
        p[kRowSeq] = (void*)g.dummy_seq;
    }
    void** dst[5] = {(void**)&g.d_slot_pos[slot], (void**)&g.d_slot_pt[slot],
                     (void**)&g.d_slot_kv[slot], (void**)&g.d_slot_tok[slot],
                     (void**)&g.d_slot_seq[slot]};
    for (uint32_t r = 0; r < 5; ++r) {
        cudaError_t e = cudaMemcpyAsync(dst[r], &p[r], sizeof(void*),
                                        cudaMemcpyHostToDevice, g.stream);
        if (e != cudaSuccess) {
            cudaGetLastError();
            mc::set_error("batch_group: slot pointer table H2D failed: %s",
                          cudaGetErrorString(e));
            return MC_E_CUDA;
        }
    }
    return MC_OK;
}

// 逆序释放（与创建相反）。幂等。
void release_group(BatchGroupImpl* g) {
    if (g == nullptr) return;
    if (g->model != nullptr) (void)cudaSetDevice(g->model->device_id);
    if (g->stream != nullptr) (void)cudaStreamSynchronize(g->stream);
    // 仍在组的成员：组流 record → 成员流 wait（remove 语义），再摘除。
    for (uint32_t i = 0; i < g->max_slots; ++i) {
        SessionImpl* s = g->members[i];
        if (s == nullptr) continue;
        if (g->ev_group != nullptr) {
            (void)cudaEventRecord(g->ev_group, g->stream);
            (void)cudaStreamWaitEvent(s->stream, g->ev_group, 0);
        }
        s->batch_group = nullptr;
        g->members[i] = nullptr;
    }
    if (g->bd_enabled) {
        for (uint32_t i = 0; i < kBdEvents; ++i)
            if (g->bd_ev[i] != nullptr) (void)cudaEventDestroy(g->bd_ev[i]);
        g->bd_enabled = false;
    }
    for (uint32_t i = 0; i < kMaxBatchSlots; ++i)
        if (g->ev_member[i] != nullptr) (void)cudaEventDestroy(g->ev_member[i]);
    if (g->ev_group != nullptr) (void)cudaEventDestroy(g->ev_group);
    if (g->h_tokens != nullptr) (void)cudaFreeHost(g->h_tokens);
    if (g->h_stage != nullptr) (void)cudaFreeHost(g->h_stage);
    g->arena.destroy();
    if (g->stream != nullptr) (void)cudaStreamDestroy(g->stream);
    delete g;
}

} // namespace

extern "C" MC_API mc_status_t mc_batch_group_create(
    mc_model_t model, const mc_batch_options_t* options,
    mc_batch_group_t* out_group) {
    try {
        mc::clear_error();
        if (out_group != nullptr) *out_group = mc_batch_group{nullptr};
        if (model.impl == nullptr) {
            mc::set_error("mc_batch_group_create: model is NULL");
            return MC_E_INVALID_ARGUMENT;
        }
        if (options == nullptr || out_group == nullptr) {
            mc::set_error("mc_batch_group_create: options/out_group is NULL");
            return MC_E_INVALID_ARGUMENT;
        }
        if (options->struct_size != sizeof(*options)) {
            mc::set_error("mc_batch_group_create: options->struct_size=%u but "
                          "expected %zu",
                          options->struct_size, sizeof(*options));
            return MC_E_INVALID_ARGUMENT;
        }
        if (options->abi_version != MC_ABI_VERSION) {
            mc::set_error("mc_batch_group_create: abi_version=0x%08x expected 0x%08x",
                          options->abi_version, MC_ABI_VERSION);
            return MC_E_INVALID_ARGUMENT;
        }
        for (uint32_t i = 0; i < 4; ++i) {
            if (options->reserved[i] != 0) {
                mc::set_error("mc_batch_group_create: reserved[%u] must be zero", i);
                return MC_E_INVALID_ARGUMENT;
            }
        }
        if (options->max_slots == 0 || options->max_slots > kMaxBatchSlots) {
            mc::set_error("mc_batch_group_create: max_slots=%u out of range [1,%u]",
                          options->max_slots, kMaxBatchSlots);
            return MC_E_UNSUPPORTED;
        }
        ModelImpl* m = model.impl;
        if (!m->has_wpk) {
            mc::set_error("mc_batch_group_create: real weights (model.wpk) required");
            return MC_E_UNSUPPORTED;
        }
        if (!m->weights.complete()) {
            mc::set_error("mc_batch_group_create: partially-loaded weights "
                          "(MC_LOAD_LAYERS_*) cannot serve batch forward");
            return MC_E_UNSUPPORTED;
        }
        if (options->max_context_tokens != m->max_context_tokens) {
            mc::set_error("mc_batch_group_create: max_context_tokens=%u must equal "
                          "the model's %u (ctx/chunk tier mismatch)",
                          options->max_context_tokens, m->max_context_tokens);
            return MC_E_UNSUPPORTED;
        }

        cudaError_t e = cudaSetDevice(m->device_id);
        if (e != cudaSuccess) {
            cudaGetLastError();
            mc::set_error("mc_batch_group_create: cudaSetDevice failed: %s",
                          cudaGetErrorString(e));
            return MC_E_CUDA;
        }

        BatchGroupImpl* g = new BatchGroupImpl();
        g->model = m;
        g->max_slots = options->max_slots;
        g->max_context_tokens = options->max_context_tokens;
        g->attn_chunk = attention_decode_chunk_pos(g->max_context_tokens);
        g->attn_max_chunks =
            attention_decode_max_chunks(g->max_context_tokens, g->attn_chunk);
        g->layout.max_pages = (g->max_context_tokens +
                               PagedKvAllocator::kTokensPerPage - 1) /
                              PagedKvAllocator::kTokensPerPage;

        e = cudaStreamCreate(&g->stream);
        if (e != cudaSuccess) {
            cudaGetLastError();
            mc::set_error("mc_batch_group_create: cudaStreamCreate failed: %s",
                          cudaGetErrorString(e));
            delete g;
            return MC_E_CUDA;
        }

        // ---- 组 arena 一次性 carve（建期；steady-state 零分配）----
        const uint32_t R = g->max_slots; // 激活行数
        const size_t act_bytes =
            (size_t)R * (5ull * 2048 + 2560 + 12288 + 6144) * 2u /*bf16*/
            + (size_t)R * mc::cfg::kVocabSize * 2u /*logits*/;
        const size_t partial_bytes =
            (size_t)R * mc::cfg::kNumQueryHeads * g->attn_max_chunks *
            kAttnPartialStride * sizeof(float);
        const size_t partial2_bytes =
            (size_t)R * mc::cfg::kNumQueryHeads * kMergeSplit2 *
            kAttnPartialStride * sizeof(float);
        // 停车槽 dummy 池：1 页（32 token × 512 元素 × 42 层）bf16
        const size_t dummy_kv_bytes =
            (size_t)KvLayout::kTokensPerPage * KvLayout::kSlotElems * 42u * 2u;
        size_t need = act_bytes + partial_bytes + partial2_bytes;
        need += 16u << 20;                                // gemm_ws（恒 16MB）
        need += 64 * sizeof(float);                       // rope_inv_freq
        need += 10 * (size_t)kMaxBatchSlots * sizeof(void*); // 成员表 + 步表
        need += 4 + 4 + 4;                                // dummy_pos/tok/pt
        need += dummy_kv_bytes;                           // dummy KV 池
        need += 64 * sizeof(int32_t);                     // dummy_seq
        need += 8 * 256;                                  // bump 对齐 slack
        mc_status_t rs = g->arena.init(need, "batch_group.arena");
        if (rs != MC_OK) {
            release_group(g); // stream 已建；arena 未 init 时 destroy 为 no-op
            return rs;
        }
        uint16_t* p = (uint16_t*)g->arena.alloc(act_bytes, 256);
        g->residual = p;  p += (size_t)R * 2048;
        g->normed   = p;  p += (size_t)R * 2048;
        g->qkv      = p;  p += (size_t)R * 2560;
        g->ctx      = p;  p += (size_t)R * 2048;
        g->o_out    = p;  p += (size_t)R * 2048;
        g->gate_up  = p;  p += (size_t)R * 12288;
        g->mlp_mid  = p;  p += (size_t)R * 6144;
        g->down_out = p;  p += (size_t)R * 2048;
        g->logits   = p;  p += (size_t)R * mc::cfg::kVocabSize;
        g->partials = reinterpret_cast<float*>(p);
        p += partial_bytes / 2u; // fp32 → uint16 单位
        g->partials2 = reinterpret_cast<float*>(p);
        p += partial2_bytes / 2u;
        g->rope_inv_freq = reinterpret_cast<float*>(p);
        p += 64u * 2u;
        g->gemm_ws = g->arena.alloc(16u << 20, 256);
        g->gemm_ws_bytes = 16u << 20;
        auto carve_table = [&](void*** out) {
            *out = (void**)g->arena.alloc(kMaxBatchSlots * sizeof(void*), 256);
        };
        carve_table((void***)&g->d_slot_pos);
        carve_table((void***)&g->d_slot_pt);
        carve_table((void***)&g->d_slot_kv);
        carve_table((void***)&g->d_slot_tok);
        carve_table((void***)&g->d_slot_seq);
        carve_table((void***)&g->d_step_pos);
        carve_table((void***)&g->d_step_pt);
        carve_table((void***)&g->d_step_kv);
        carve_table((void***)&g->d_step_tok);
        carve_table((void***)&g->d_step_seq);
        g->dummy_pos = g->arena.alloc<uint32_t>(1);
        g->dummy_tok = g->arena.alloc<int32_t>(1);
        g->dummy_pt = g->arena.alloc<uint32_t>(1);
        g->dummy_kv = static_cast<uint16_t*>(g->arena.alloc(dummy_kv_bytes, 256));
        g->dummy_seq = g->arena.alloc<int32_t>(64);
        if (g->residual == nullptr || g->logits == nullptr || g->partials == nullptr ||
            g->partials2 == nullptr || g->rope_inv_freq == nullptr ||
            g->gemm_ws == nullptr || g->d_slot_pos == nullptr ||
            g->d_slot_pt == nullptr || g->d_slot_kv == nullptr ||
            g->d_slot_tok == nullptr || g->d_slot_seq == nullptr ||
            g->d_step_pos == nullptr || g->d_step_pt == nullptr ||
            g->d_step_kv == nullptr || g->d_step_tok == nullptr ||
            g->d_step_seq == nullptr || g->dummy_pos == nullptr ||
            g->dummy_tok == nullptr || g->dummy_pt == nullptr ||
            g->dummy_kv == nullptr || g->dummy_seq == nullptr) {
            mc::set_error("mc_batch_group_create: arena carve failed (need %zu)",
                          need);
            release_group(g);
            return MC_E_OUT_OF_MEMORY;
        }
        g->arena.begin_steady_state();

        // pinned（收割缓冲 + 表 H2D 暂存）
        e = cudaHostAlloc((void**)&g->h_tokens,
                          (size_t)g->max_slots * sizeof(int32_t),
                          cudaHostAllocDefault);
        if (e == cudaSuccess)
            e = cudaHostAlloc((void**)&g->h_stage,
                              5u * kMaxBatchSlots * sizeof(void*),
                              cudaHostAllocDefault);
        if (e != cudaSuccess) {
            cudaGetLastError();
            mc::set_error("mc_batch_group_create: cudaHostAlloc failed: %s",
                          cudaGetErrorString(e));
            release_group(g);
            return (e == cudaErrorMemoryAllocation) ? MC_E_OUT_OF_MEMORY : MC_E_CUDA;
        }

        // 事件（add/remove 事件链 + 段拆分）
        bool ok = cudaEventCreate(&g->ev_group) == cudaSuccess;
        for (uint32_t i = 0; i < kMaxBatchSlots && ok; ++i)
            ok = cudaEventCreate(&g->ev_member[i]) == cudaSuccess;
        if (!ok) {
            cudaGetLastError();
            mc::set_error("mc_batch_group_create: event creation failed");
            release_group(g);
            return MC_E_CUDA;
        }
        if (getenv("MC_BATCH_BREAKDOWN") != nullptr) {
            bool bok = true;
            for (uint32_t i = 0; i < kBdEvents && bok; ++i)
                bok = cudaEventCreate(&g->bd_ev[i]) == cudaSuccess;
            if (bok) {
                g->bd_enabled = true;
            } else {
                cudaGetLastError(); // 部分创建失败：清退静默禁用（诊断工具）
                for (uint32_t i = 0; i < kBdEvents; ++i) {
                    if (g->bd_ev[i] != nullptr)
                        (void)cudaEventDestroy(g->bd_ev[i]);
                    g->bd_ev[i] = nullptr;
                }
            }
        }

        // ---- 停车槽 + 成员表初始内容：全部槽先指停车 ----
        (void)cudaMemsetAsync(g->dummy_pos, 0, sizeof(uint32_t), g->stream);
        (void)cudaMemsetAsync(g->dummy_tok, 0, sizeof(int32_t), g->stream);
        (void)cudaMemsetAsync(g->dummy_pt, 0, sizeof(uint32_t), g->stream);
        for (uint32_t i = 0; i < g->max_slots; ++i) {
            rs = write_slot_pointers(*g, i, nullptr);
            if (rs != MC_OK) {
                release_group(g);
                return rs;
            }
        }

        // ---- RoPE inv_freq 表（与 session 建期同一 kernel/表达式 → 逐位一致）----
        rs = launch_rope_inv_freq_init(g->rope_inv_freq, mc::cfg::kRopeTheta,
                                       g->stream);
        if (rs != MC_OK) {
            release_group(g);
            return rs;
        }

        // ---- gemm 计划预热：M ∈ {2..max_slots} × 5 shape ----
        // 任务书下限 {2,4,8}；eager 精确 n 可命中任意 M ∈ [2,8]，全档预热
        //（≤35 个计划，远低于缓存上限 256）。用真实权重 + 组激活真实跑一次
        //（输出涂鸦无害 —— 首步 forward 全量覆写），plan+algo 进缓存后
        // steady-state 零 heuristic/零 host 分配。
        {
            const WeightRefs& w = m->weights;
            const uint32_t qkv_n = mc::cfg::kNumQueryHeads * mc::cfg::kHeadDim +
                                   2 * mc::cfg::kNumKvHeads * mc::cfg::kHeadDim;
            for (uint32_t M = 2; M <= g->max_slots; ++M) {
                rs = m->gemm.gemm_bf16(g->stream, g->gemm_ws, g->gemm_ws_bytes, M,
                                       qkv_n, mc::cfg::kHiddenSize, w.qkv[0],
                                       g->normed, g->qkv);
                if (rs == MC_OK)
                    rs = m->gemm.gemm_bf16(g->stream, g->gemm_ws, g->gemm_ws_bytes,
                                           M, mc::cfg::kHiddenSize,
                                           mc::cfg::kHiddenSize, w.o[0], g->ctx,
                                           g->o_out);
                if (rs == MC_OK)
                    rs = m->gemm.gemm_bf16(g->stream, g->gemm_ws, g->gemm_ws_bytes,
                                           M, 2 * mc::cfg::kIntermediateSize,
                                           mc::cfg::kHiddenSize, w.gate_up[0],
                                           g->normed, g->gate_up);
                if (rs == MC_OK)
                    rs = m->gemm.gemm_bf16(g->stream, g->gemm_ws, g->gemm_ws_bytes,
                                           M, mc::cfg::kHiddenSize,
                                           mc::cfg::kIntermediateSize, w.down[0],
                                           g->mlp_mid, g->down_out);
                if (rs == MC_OK)
                    rs = m->gemm.gemm_bf16(g->stream, g->gemm_ws, g->gemm_ws_bytes,
                                           M, mc::cfg::kVocabSize,
                                           mc::cfg::kHiddenSize, w.lm_head,
                                           g->normed, g->logits);
                if (rs != MC_OK) {
                    release_group(g);
                    return rs;
                }
            }
        }

        e = cudaStreamSynchronize(g->stream);
        if (e != cudaSuccess) {
            cudaGetLastError();
            mc::set_error("mc_batch_group_create: init sync failed: %s",
                          cudaGetErrorString(e));
            release_group(g);
            return MC_E_CUDA;
        }

        fprintf(stderr,
                "[minicpm-native] batch_group_create: slots=%u ctx=%u chunk=%u "
                "max_chunks=%u arena=%zu bytes (act=%zu partials=%zu+%zu "
                "gemm_ws=16MiB), breakdown=%s\n",
                g->max_slots, g->max_context_tokens, g->attn_chunk,
                g->attn_max_chunks, need, act_bytes, partial_bytes, partial2_bytes,
                g->bd_enabled ? "on" : "off");
        *out_group = mc_batch_group{g};
        return MC_OK;
    } catch (...) {
        mc::set_error("mc_batch_group_create: internal error (uncaught exception)");
        return MC_E_INTERNAL;
    }
}

extern "C" MC_API mc_status_t mc_batch_group_add(mc_batch_group_t group,
                                                 mc_session_t session,
                                                 uint32_t* slot_id) {
    try {
        mc::clear_error();
        BatchGroupImpl* g = group.impl;
        if (g == nullptr) {
            mc::set_error("mc_batch_group_add: group is NULL");
            return MC_E_INVALID_ARGUMENT;
        }
        SessionImpl* s = session.impl;
        if (s == nullptr || slot_id == nullptr) {
            mc::set_error("mc_batch_group_add: session/slot_id is NULL");
            return MC_E_INVALID_ARGUMENT;
        }
        if (s->model != g->model) {
            mc::set_error("mc_batch_group_add: session belongs to a different model");
            return MC_E_UNSUPPORTED;
        }
        // ctx/chunk 档校验（同模型下恒成立；显式校验以固化 ABI 契约）
        if (s->model->max_context_tokens != g->max_context_tokens ||
            s->attn_chunk_pos != g->attn_chunk ||
            s->kv.layout().max_pages != g->layout.max_pages) {
            mc::set_error("mc_batch_group_add: ctx/chunk tier mismatch");
            return MC_E_UNSUPPORTED;
        }
        if (s->batch_group != nullptr) { // 忙态：已在（任一）组内
            mc::set_error("mc_batch_group_add: session is already in a batch group");
            return MC_E_UNSUPPORTED;
        }
        uint32_t slot = g->max_slots;
        for (uint32_t i = 0; i < g->max_slots; ++i) {
            if (g->members[i] == nullptr) {
                slot = i;
                break;
            }
        }
        if (slot == g->max_slots) {
            mc::set_error("mc_batch_group_add: group is full (max_slots=%u)",
                          g->max_slots);
            return MC_E_UNSUPPORTED;
        }
        (void)cudaSetDevice(g->model->device_id);

        // 成员流→组流单向事件链：成员此前（prefill/solo decode）的 KV 写与
        // 状态推进对组流可见。
        cudaError_t e = cudaEventRecord(g->ev_member[slot], s->stream);
        if (e == cudaSuccess)
            e = cudaStreamWaitEvent(g->stream, g->ev_member[slot], 0);
        if (e != cudaSuccess) {
            cudaGetLastError();
            mc::set_error("mc_batch_group_add: event chain (member->group) failed: %s",
                          cudaGetErrorString(e));
            return MC_E_CUDA;
        }
        // 成员表内容小 H2D（5×8B，表地址固定 → graph 友好）
        mc_status_t rs = write_slot_pointers(*g, slot, s);
        if (rs != MC_OK) return rs;

        g->members[slot] = s;
        s->batch_group = g;
        *slot_id = slot;
        return MC_OK;
    } catch (...) {
        mc::set_error("mc_batch_group_add: internal error (uncaught exception)");
        return MC_E_INTERNAL;
    }
}

extern "C" MC_API mc_status_t mc_batch_group_remove(mc_batch_group_t group,
                                                    uint32_t slot_id) {
    try {
        mc::clear_error();
        BatchGroupImpl* g = group.impl;
        if (g == nullptr) {
            mc::set_error("mc_batch_group_remove: group is NULL");
            return MC_E_INVALID_ARGUMENT;
        }
        if (slot_id >= g->max_slots || g->members[slot_id] == nullptr) {
            mc::set_error("mc_batch_group_remove: slot %u has no member", slot_id);
            return MC_E_INVALID_ARGUMENT;
        }
        (void)cudaSetDevice(g->model->device_id);
        SessionImpl* s = g->members[slot_id];

        // 组流→成员流单向事件链：组的 KV/序列/状态写对此后的 solo 调用可见。
        //（host 串行 enqueue；ev_group 复用安全：每次 wait 捕获其前最近一次
        // record，后续 record 不影响已入队的 wait。）
        cudaError_t e = cudaEventRecord(g->ev_group, g->stream);
        if (e == cudaSuccess)
            e = cudaStreamWaitEvent(s->stream, g->ev_group, 0);
        if (e != cudaSuccess) {
            cudaGetLastError();
            mc::set_error("mc_batch_group_remove: event chain (group->member) "
                          "failed: %s",
                          cudaGetErrorString(e));
            return MC_E_CUDA;
        }
        // 成员表项回指停车槽（防御性：B2 图捕获期若烘焙表读，悬空指针不可接受）
        mc_status_t rs = write_slot_pointers(*g, slot_id, nullptr);
        if (rs != MC_OK) return rs;

        g->members[slot_id] = nullptr;
        s->batch_group = nullptr;
        return MC_OK;
    } catch (...) {
        mc::set_error("mc_batch_group_remove: internal error (uncaught exception)");
        return MC_E_INTERNAL;
    }
}

extern "C" MC_API mc_status_t mc_batch_step(mc_batch_group_t group,
                                            const uint32_t* slot_ids, uint32_t n,
                                            const mc_generation_config_t* cfgs,
                                            int32_t* out_tokens) {
    try {
        mc::clear_error();
#define MC_CHK(call)                                                          \
    do {                                                                      \
        mc_status_t _rs = (call);                                             \
        if (_rs != MC_OK) return _rs;                                         \
    } while (0)
        BatchGroupImpl* g = group.impl;
        if (g == nullptr) {
            mc::set_error("mc_batch_step: group is NULL");
            return MC_E_INVALID_ARGUMENT;
        }
        if (slot_ids == nullptr || cfgs == nullptr || out_tokens == nullptr) {
            mc::set_error("mc_batch_step: slot_ids/cfgs/out_tokens is NULL");
            return MC_E_INVALID_ARGUMENT;
        }
        if (n == 0 || n > g->max_slots) {
            mc::set_error("mc_batch_step: n=%u out of range [1,%u]", n, g->max_slots);
            return MC_E_INVALID_ARGUMENT;
        }
        for (uint32_t i = 0; i < n; ++i) {
            const uint32_t sid = slot_ids[i];
            if (sid >= g->max_slots || g->members[sid] == nullptr) {
                mc::set_error("mc_batch_step: slot_ids[%u]=%u has no member", i, sid);
                return MC_E_INVALID_ARGUMENT;
            }
            for (uint32_t j = 0; j < i; ++j) {
                if (slot_ids[j] == sid) {
                    mc::set_error("mc_batch_step: duplicate slot %u", sid);
                    return MC_E_INVALID_ARGUMENT;
                }
            }
            if (cfgs[i].struct_size != sizeof(cfgs[i])) {
                mc::set_error("mc_batch_step: cfgs[%u].struct_size=%u but expected %zu",
                              i, cfgs[i].struct_size, sizeof(cfgs[i]));
                return MC_E_INVALID_ARGUMENT;
            }
            // B1 仅 greedy（激活规则同 decode_one 的采样判定取反）
            if (cfgs[i].temperature > 0.f && cfgs[i].top_k != 1u) {
                mc::set_error("mc_batch_step: non-greedy cfgs[%u] unsupported in B1 "
                              "(batch sampling is B2)",
                              i);
                return MC_E_UNSUPPORTED;
            }
        }
        (void)cudaSetDevice(g->model->device_id);
        const ModelImpl* m = g->model;
        const WeightRefs& w = m->weights;
        const uint32_t H = mc::cfg::kHiddenSize;
        cudaStream_t st = g->stream;

        // 成员状态校验 + KV 页增长（组流上的 H2D；decode 通常 no-op 快路径）
        for (uint32_t i = 0; i < n; ++i) {
            SessionImpl* s = g->members[slot_ids[i]];
            if (s->seq_len == 0) {
                mc::set_error("mc_batch_step: slot %u sequence is empty (prefill "
                              "first)",
                              slot_ids[i]);
                return MC_E_INVALID_ARGUMENT;
            }
            if (s->seq_len >= g->max_context_tokens) {
                mc::set_error("mc_batch_step: slot %u context is full (seq_len=%u)",
                              slot_ids[i], s->seq_len);
                return MC_E_OUT_OF_MEMORY;
            }
            MC_CHK(s->kv.ensure_pages_for(s->seq_len + 1, st));
        }

        const auto t0 = std::chrono::steady_clock::now();
#define MC_BD_REC()                                                            \
    do {                                                                       \
        if (g->bd_enabled && g->bd_used < kBdEvents)                           \
            (void)cudaEventRecord(g->bd_ev[g->bd_used++], st);                 \
    } while (0)
        g->bd_used = 0;
        MC_BD_REC(); // [0] start

        // ---- 0) 步表压缩视图：slot_ids[0..n) 的成员私有量 → d_step_*（固定
        //      地址、内容小 H2D；pinned 暂存源）----
        {
            void* rows[5] = {(void*)g->d_step_pos, (void*)g->d_step_pt,
                             (void*)g->d_step_kv,   (void*)g->d_step_tok,
                             (void*)g->d_step_seq};
            for (uint32_t r = 0; r < 5; ++r) {
                for (uint32_t i = 0; i < n; ++i) {
                    const SessionImpl* s = g->members[slot_ids[i]];
                    void* v[5] = {(void*)&s->d_state->seq_len,
                                  (void*)s->d_page_table, s->kv_pool,
                                  (void*)s->d_next_token, (void*)s->d_seq};
                    g->h_stage[(size_t)r * kMaxBatchSlots + i] = v[r];
                }
                cudaError_t e = cudaMemcpyAsync(rows[r],
                                                g->h_stage + (size_t)r * kMaxBatchSlots,
                                                (size_t)n * sizeof(void*),
                                                cudaMemcpyHostToDevice, st);
                if (e != cudaSuccess) {
                    cudaGetLastError();
                    mc::set_error("mc_batch_step: step table H2D failed: %s",
                                  cudaGetErrorString(e));
                    return MC_E_CUDA;
                }
            }
        }

        // ---- 1) 发放上一步决策：D2H 旧 token（预决策语义，同 decode_one）----
        for (uint32_t i = 0; i < n; ++i) {
            SessionImpl* s = g->members[slot_ids[i]];
            cudaError_t e = cudaMemcpyAsync(&g->h_tokens[i], s->d_next_token,
                                            sizeof(int32_t), cudaMemcpyDeviceToHost,
                                            st);
            if (e != cudaSuccess) {
                cudaGetLastError();
                mc::set_error("mc_batch_step: token D2H (slot %u) failed: %s",
                              slot_ids[i], cudaGetErrorString(e));
                return MC_E_CUDA;
            }
        }

        // ---- 2) embedding 间接 gather：成员 d_next_token → residual[n][2048] ----
        MC_CHK(launch_embedding_gather_indirect(w.embedding, g->d_step_tok,
                                                g->residual, (int32_t)H, (int32_t)n,
                                                st));
        MC_BD_REC(); // emb

        // ---- 3) 42 层（M=n 全部 cuBLASLt；无融合 prologue —— 那是 T=1 专用）----
        const uint32_t qkv_n = mc::cfg::kNumQueryHeads * mc::cfg::kHeadDim +
                               2 * mc::cfg::kNumKvHeads * mc::cfg::kHeadDim;
        for (uint32_t i = 0; i < mc::cfg::kNumLayers; ++i) {
            if (w.norm1[i] == nullptr || w.qkv[i] == nullptr || w.o[i] == nullptr ||
                w.norm2[i] == nullptr || w.gate_up[i] == nullptr ||
                w.down[i] == nullptr) {
                mc::set_error("mc_batch_step: layer %u weights not loaded", i);
                return MC_E_INTERNAL;
            }
            if (i == 0) {
                MC_CHK(launch_rmsnorm_rows(g->residual, w.norm1[0], g->normed, n,
                                           (int32_t)H, mc::cfg::kRmsEps, st));
            } // i>0：normed 已由上一层尾部 resnorm 产出
            MC_CHK(m->gemm.gemm_bf16(st, g->gemm_ws, g->gemm_ws_bytes, n, qkv_n, H,
                                      w.qkv[i], g->normed, g->qkv));
            MC_BD_REC(); // 层段 1/8：qkv

            // 批 attention（ph0：q/k rope + K/V 写池 + chunk/merge，B1a kernel）
            MC_CHK(launch_attention_decode_batch(
                g->qkv, g->d_step_pos, n, i, g->d_step_pt, g->d_step_kv, g->layout,
                g->partials, g->attn_max_chunks, g->ctx, st, g->rope_inv_freq,
                g->attn_chunk, g->partials2));
            MC_BD_REC(); // 层段 2/8：attn

            MC_CHK(m->gemm.gemm_bf16(st, g->gemm_ws, g->gemm_ws_bytes, n, H, H,
                                      w.o[i], g->ctx, g->o_out));
            MC_BD_REC(); // 层段 3/8：o
            MC_CHK(launch_residual_add_rmsnorm(g->residual, g->o_out, w.norm2[i],
                                               g->normed, n, (int32_t)H,
                                               mc::cfg::kRmsEps, st));
            MC_BD_REC(); // 层段 4/8：resnorm2

            MC_CHK(m->gemm.gemm_bf16(st, g->gemm_ws, g->gemm_ws_bytes, n,
                                     2 * mc::cfg::kIntermediateSize, H,
                                     w.gate_up[i], g->normed, g->gate_up));
            MC_BD_REC(); // 层段 5/8：gate_up
            MC_CHK(launch_swiglu_gated(g->gate_up, g->mlp_mid, n,
                                       mc::cfg::kIntermediateSize, st));
            MC_BD_REC(); // 层段 6/8：swiglu

            MC_CHK(m->gemm.gemm_bf16(st, g->gemm_ws, g->gemm_ws_bytes, n, H,
                                     mc::cfg::kIntermediateSize, w.down[i],
                                     g->mlp_mid, g->down_out));
            MC_BD_REC(); // 层段 7/8：down
            const uint16_t* next_norm_w =
                (i + 1 < mc::cfg::kNumLayers) ? w.norm1[i + 1] : w.final_norm;
            if (next_norm_w == nullptr) {
                mc::set_error("mc_batch_step: final_norm weight is NULL");
                return MC_E_INTERNAL;
            }
            // 残差加 + 下一层 norm1（末层 = final_norm → lm_head 消费 normed）
            MC_CHK(launch_residual_add_rmsnorm(g->residual, g->down_out, next_norm_w,
                                               g->normed, n, (int32_t)H,
                                               mc::cfg::kRmsEps, st));
            MC_BD_REC(); // 层段 8/8：resnorm(next/final)
        }

        // ---- 4) lm_head M=n → logits[n][vocab] ----
        if (w.lm_head == nullptr || w.embedding == nullptr) {
            mc::set_error("mc_batch_step: lm_head/embedding not loaded");
            return MC_E_INTERNAL;
        }
        MC_CHK(m->gemm.gemm_bf16(st, g->gemm_ws, g->gemm_ws_bytes, n,
                                 mc::cfg::kVocabSize, H, w.lm_head, g->normed,
                                 g->logits));
        MC_BD_REC(); // lm_head

        // ---- 5) 批 argmax（写各成员 d_next_token）→ 批 advance（device 侧
        //      d_seq append + seq_len+1；host 镜像在 sync 后同步）----
        MC_CHK(launch_batch_argmax_bf16(g->logits, g->d_step_tok,
                                        (int32_t)mc::cfg::kVocabSize, (int32_t)n,
                                        st));
        MC_CHK(launch_batch_advance(g->d_step_seq, g->d_step_tok, g->d_step_pos, n,
                                    st));
        MC_BD_REC(); // tail(argmax+advance)
#undef MC_BD_REC

        // ---- 6) sync + pinned 收割（同步语义同 decode_one）----
        cudaError_t e = cudaStreamSynchronize(st);
        if (e != cudaSuccess) {
            cudaGetLastError();
            mc::set_error("mc_batch_step: stream sync failed: %s",
                          cudaGetErrorString(e));
            return MC_E_CUDA;
        }
        for (uint32_t i = 0; i < n; ++i) {
            const int32_t tok = g->h_tokens[i];
            if (tok < 0 || (uint32_t)tok >= mc::cfg::kVocabSize) {
                mc::set_error("mc_batch_step: slot %u emitted out-of-range token %d",
                              slot_ids[i], tok);
                return MC_E_INTERNAL;
            }
            out_tokens[i] = tok;
        }

        // ---- 7) host 镜像 + 统计 ----
        // device 侧 seq_len 已由 advance kernel 推进（d_state->next_token 镜像
        // 无 kernel 读者，不回写；下次 solo decode 的 eager/graph 路径均自行
        // 重算并回写 h_state 前 8B，见 decode_executor）。
        for (uint32_t i = 0; i < n; ++i) {
            SessionImpl* s = g->members[slot_ids[i]];
            s->seq_len += 1;
            s->h_state.seq_len = s->seq_len;
            s->h_state.next_token = (uint32_t)out_tokens[i]; // 上一步决策（发放值）
        }
        const auto t1 = std::chrono::steady_clock::now();
        g->decode_us += (uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(
                            t1 - t0)
                            .count();
        g->steps += 1; // decode_us 归组；成员归属细分是 B3
        return MC_OK;
#undef MC_CHK
    } catch (...) {
        mc::set_error("mc_batch_step: internal error (uncaught exception)");
        return MC_E_INTERNAL;
    }
}

extern "C" MC_API void mc_batch_group_destroy(mc_batch_group_t group) {
    if (group.impl == nullptr) return;
    release_group(group.impl);
}

// ---- additive debug 钩子（不属于冻结 ABI；同 mc_debug_* 先例）----

// 组统计：steps / decode_us / 现役成员数。诊断/测试用。
extern "C" MC_API mc_status_t mc_debug_batch_group_stats(mc_batch_group_t group,
                                                         uint64_t* steps,
                                                         uint64_t* decode_us,
                                                         uint32_t* members) {
    BatchGroupImpl* g = group.impl;
    if (g == nullptr || steps == nullptr || decode_us == nullptr ||
        members == nullptr) {
        mc::set_error("mc_debug_batch_group_stats: invalid args");
        return MC_E_INVALID_ARGUMENT;
    }
    uint32_t cnt = 0;
    for (uint32_t i = 0; i < g->max_slots; ++i)
        if (g->members[i] != nullptr) ++cnt;
    *steps = g->steps;
    *decode_us = g->decode_us;
    *members = cnt;
    return MC_OK;
}

// 最近一次 mc_batch_step 的段拆分（ms，桶口径见 kBdNames）。
// 返回桶数；0 = 未启用（MC_BATCH_BREAKDOWN 未设）；-1 = 参数不符；
// -2 = 上次 step 事件数不足（不应发生）；-3 = CUDA 错。
// 前置：上次 mc_batch_step 已返回（内部已 sync）。
extern "C" MC_API int mc_debug_batch_breakdown_collect(mc_batch_group_t group,
                                                       double* out_ms, int n) {
    BatchGroupImpl* g = group.impl;
    if (g == nullptr || out_ms == nullptr || n != (int)kBdBuckets) return -1;
    if (!g->bd_enabled) return 0;
    if (g->bd_used != kBdEvents) return -2;
    (void)cudaSetDevice(g->model->device_id);
    double buckets[kBdBuckets] = {0, 0, 0, 0, 0, 0};
    // 段索引：0=emb；1..336 层段（8 段/层：qkv/attn/o/resnorm2/gate_up/
    // swiglu/down/resnorm）；337=lm_head；338=tail
    auto add = [&](uint32_t seg, uint32_t bucket) {
        float ms = 0.f;
        if (cudaEventElapsedTime(&ms, g->bd_ev[seg], g->bd_ev[seg + 1]) !=
            cudaSuccess) {
            cudaGetLastError();
            return false;
        }
        buckets[bucket] += (double)ms;
        return true;
    };
    // 层内段 → 桶：qkv/o/gate_up/down→gemm(1)；attn→attn(2)；
    // resnorm2/resnorm→norm(3)；swiglu→swiglu(4)
    const uint32_t layer_bucket[kBdLayerSegs] = {1, 2, 1, 3, 1, 4, 1, 3};
    const uint32_t layer_segs = mc::cfg::kNumLayers * kBdLayerSegs;
    if (!add(0, 0)) return -3; // emb
    for (uint32_t s = 0; s < layer_segs; ++s) {
        if (!add(1 + s, layer_bucket[s % kBdLayerSegs])) return -3;
    }
    if (!add(1 + layer_segs, 1)) return -3; // lm_head → gemm
    if (!add(2 + layer_segs, 5)) return -3; // tail
    for (uint32_t i = 0; i < kBdBuckets; ++i) out_ms[i] = buckets[i];
    return (int)kBdBuckets;
}

// 最近一次 mc_batch_step 的某 slot logits（bf16 upcast fp32，[vocab]）。
// 前置：上次 mc_batch_step 已返回（组流已 sync，logits 仍驻留组 arena）。
// 场景 A 的 relL2 对拍数据源（solo 侧用 MC_DEBUG_DUMP 的 step logits）。
extern "C" MC_API mc_status_t mc_debug_batch_last_logits(mc_batch_group_t group,
                                                         uint32_t slot,
                                                         float* out_f32) {
    BatchGroupImpl* g = group.impl;
    if (g == nullptr || out_f32 == nullptr || slot >= g->max_slots) {
        mc::set_error("mc_debug_batch_last_logits: invalid args");
        return MC_E_INVALID_ARGUMENT;
    }
    (void)cudaSetDevice(g->model->device_id);
    const uint16_t* row = g->logits + (size_t)slot * mc::cfg::kVocabSize;
    // bf16 → f32 由 host 侧位扩展（bf16 = f32 高 16 位）
    std::vector<uint16_t> h((size_t)mc::cfg::kVocabSize);
    cudaError_t e = cudaMemcpy(h.data(), row,
                               (size_t)mc::cfg::kVocabSize * sizeof(uint16_t),
                               cudaMemcpyDeviceToHost);
    if (e != cudaSuccess) {
        cudaGetLastError();
        mc::set_error("mc_debug_batch_last_logits: D2H failed: %s",
                      cudaGetErrorString(e));
        return MC_E_CUDA;
    }
    for (uint32_t i = 0; i < mc::cfg::kVocabSize; ++i) {
        const uint32_t bits = (uint32_t)h[i] << 16;
        out_f32[i] = __builtin_bit_cast(float, bits);
    }
    return MC_OK;
}
