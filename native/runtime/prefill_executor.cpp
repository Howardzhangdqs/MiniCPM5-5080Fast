// prefill_executor.cpp — prefill 编排实现（P2：real 模式追加真实 forward）。
// P8：T > chunk 时分块循环调用 forward（128K 上下文支持），见下。
#include "runtime/prefill_executor.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <cuda_runtime.h>

#include "runtime/error.h"
#include "runtime/forward.h"
#include "runtime/generated_model_config.h"
#include "runtime/internal_state.h"

namespace mc {

// ---- 分块常量（P8 长上下文）----
// 单次 forward 的 prefill token 上限：
//   - 下限 8448：保证现有单块路径（graph_bench 的 8412-token prefill 等）
//     恒满足 T ≤ chunk，走与原实现完全一致的单次路径（零行为变化）；
//   - 上限 65535：attention_prefill kernel grid.y = tokens（CUDA grid.y
//     上限 65535，超出即启动失败）。
// 激活 scratch 按 min(max_ctx, chunk) 预分配（session.cpp compute_sizes），
// 分块循环保证每块 T ≤ chunk ≤ scratch 行数，不会写穿切片。
constexpr uint32_t kPrefillChunkTokens = 8448;
constexpr uint32_t kPrefillChunkMax = 65535;

uint32_t prefill_chunk_tokens() {
    // env 一次读取（函数局部 static，进程内缓存）：MC_PREFILL_CHUNK 覆盖
    // 默认值；非法值（非数字 / 超出 [8448,65535]）回退默认并在 stderr 警告。
    // 下限防「缩小 scratch 后单块路径写穿」，上限防 kernel grid.y 溢出。
    static const uint32_t chunk = [] {
        uint32_t v = kPrefillChunkTokens;
        if (const char* env = getenv("MC_PREFILL_CHUNK")) {
            char* end = nullptr;
            const unsigned long parsed = strtoul(env, &end, 10);
            if (end == env || *end != '\0' || parsed < kPrefillChunkTokens ||
                parsed > kPrefillChunkMax) {
                fprintf(stderr,
                        "[minicpm-native] MC_PREFILL_CHUNK=\"%s\" invalid "
                        "(need %u..%u) -- fallback to default %u\n",
                        env, kPrefillChunkTokens, kPrefillChunkMax,
                        kPrefillChunkTokens);
            } else {
                v = (uint32_t)parsed;
            }
        }
        return v;
    }();
    return chunk;
}

mc_status_t prefill_run(SessionImpl& s, const int32_t* token_ids, uint32_t token_count) {
    // ---- 校验：token 范围 [0, vocab)、数量、上下文容量 ----
    if (token_ids == nullptr) {
        mc::set_error("mc_prefill: token_ids is NULL");
        return MC_E_INVALID_ARGUMENT;
    }
    if (token_count == 0) {
        mc::set_error("mc_prefill: token_count must be > 0");
        return MC_E_INVALID_ARGUMENT;
    }
    const uint32_t max_ctx = s.model->max_context_tokens;
    if (token_count > max_ctx - s.seq_len) {
        mc::set_error("mc_prefill: token_count=%u would exceed max_context_tokens=%u "
                      "(seq_len=%u)",
                      token_count, max_ctx, s.seq_len);
        return MC_E_OUT_OF_MEMORY;
    }
    for (uint32_t i = 0; i < token_count; ++i) {
        const int32_t id = token_ids[i];
        if (id < 0 || (uint32_t)id >= cfg::kVocabSize) {
            mc::set_error("mc_prefill: token_ids[%u]=%d out of range [0,%u)", i, id,
                          cfg::kVocabSize);
            return MC_E_INVALID_ARGUMENT;
        }
    }

    const auto t0 = std::chrono::steady_clock::now();
    const uint32_t base = s.seq_len;

    // ---- pinned staging → H2D（token ids 到 device 序列缓冲）----
    std::memcpy(s.h_tokens, token_ids, (size_t)token_count * sizeof(int32_t));
    cudaError_t e = cudaMemcpyAsync(s.d_seq + base, s.h_tokens,
                                    (size_t)token_count * sizeof(int32_t),
                                    cudaMemcpyHostToDevice, s.stream);
    if (e != cudaSuccess) {
        cudaGetLastError();
        mc::set_error("mc_prefill: token H2D copy failed: %s", cudaGetErrorString(e));
        return MC_E_CUDA;
    }

    // ---- KV page 增长（rope 写入前保证页表覆盖）----
    const uint32_t new_len = base + token_count;
    mc_status_t rs = s.kv.ensure_pages_for(new_len, s.stream);
    if (rs != MC_OK) return rs;

    // ---- forward：real 模式跑完整 prefill 并预决策下一 token（= step0）----
    // P8 分块：T > chunk 时循环分块调用现有 forward（一次全量 T 既会撞
    // attention_prefill kernel 的 grid.y=tokens ≤ 65535 上限，激活 scratch
    // 也只按 chunk 预分配）。分块正确性：
    //   - 位置/rope 基址：kernel 运行时读 &d_state->seq_len（与 forward 的
    //     host 侧 s.seq_len 同步推进），每块开始前推进到块首 → 位置跨块
    //     连续（base..base+T-1）；
    //   - KV 可见性：上方 ensure_pages_for(new_len) 已一次性把页表覆盖到
    //     全前缀，前块的 KV 写入对后块 attention 天然可见（增量语义）；
    //   - logits/采样：每块尾部照常算 lm_head + argmax 写 d_next_token，
    //     流序上最后一块的写覆盖全部中间值 → 末 token 决策与单次全量一致。
    // T ≤ chunk 时走单次路径，与原实现逐 op 一致（零行为变化）。
#if !MC_NATIVE_MOCK_FORWARD
    if (s.model->has_wpk) {
        const uint32_t chunk = prefill_chunk_tokens();
        if (token_count <= chunk) {
            rs = run_real_forward(s, s.d_seq + base, token_count,
                                  /*allow_layer_dump=*/true);
            if (rs != MC_OK) return rs;
        } else {
            uint32_t off = 0;
            while (off < token_count) {
                const uint32_t n =
                    (token_count - off > chunk) ? chunk : (token_count - off);
                if (off > 0) {
                    // 前一块已入流；把位置基址推进到本块首。只回写前 8B
                    //（seq_len/next_token），不触碰 device rng/cfg（与下方
                    // 尾部镜像回写同一纪律）。
                    s.seq_len = base + off;
                    s.h_state.seq_len = s.seq_len;
                    e = cudaMemcpyAsync(s.d_state, &s.h_state,
                                        2 * sizeof(uint32_t),
                                        cudaMemcpyHostToDevice, s.stream);
                    if (e != cudaSuccess) {
                        cudaGetLastError();
                        mc::set_error("mc_prefill: chunked d_state H2D update "
                                      "failed: %s",
                                      cudaGetErrorString(e));
                        return MC_E_CUDA;
                    }
                }
                rs = run_real_forward(s, s.d_seq + base + off, n,
                                      /*allow_layer_dump=*/off == 0);
                if (rs != MC_OK) return rs;
                off += n;
            }
        }
    }
#endif

    // ---- seq_len 累加 + device state 镜像（只回写前 8B：不触碰 device 侧
    //      rng/cfg —— 它们由采样链路独立管理）----
    s.seq_len = new_len;
    s.sampling_step = false; // prefill 的末尾预决策恒 greedy argmax（prefill
                             // API 无 config；采样只作用于 decode 步）
    s.h_state.seq_len = s.seq_len;
    e = cudaMemcpyAsync(s.d_state, &s.h_state, 2 * sizeof(uint32_t),
                        cudaMemcpyHostToDevice, s.stream);
    if (e != cudaSuccess) {
        cudaGetLastError();
        mc::set_error("mc_prefill: d_state H2D update failed: %s", cudaGetErrorString(e));
        return MC_E_CUDA;
    }

    e = cudaStreamSynchronize(s.stream);
    if (e != cudaSuccess) {
        cudaGetLastError();
        mc::set_error("mc_prefill: stream sync failed: %s", cudaGetErrorString(e));
        return MC_E_CUDA;
    }

    // ---- 计时与统计（steady_clock，计入 prefill_us）----
    const auto t1 = std::chrono::steady_clock::now();
    const uint64_t us =
        (uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    s.total.prompt_tokens += token_count;
    s.since_reset.prompt_tokens += token_count;
    s.total.prefill_us += us;
    s.since_reset.prefill_us += us;
    return MC_OK;
}

} // namespace mc
