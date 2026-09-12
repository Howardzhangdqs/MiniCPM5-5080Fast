// decode_executor.cpp — decode 编排实现。
//
// 结构按计划 §8.1 单层执行图组织（embedding → 42 层 → final norm + lm_head →
// sampler），便于 P2 逐段填充真实 kernel。MC_NATIVE_MOCK_FORWARD=1 时整条
// 计算链路由一个确定性 mock kernel 代替（kernels/sm120/sampling.cu）。
// device token 链路（§10.4）：token 常驻 d_next_token，下一轮直接从 device
// 读取；host 只经 pinned h_next_token 收 4 字节，绝不把 token 传回 GPU。
//
// P3（§12）：graph_mode 下 decode step = 一次 cudaGraphLaunch。graph 在
// session_create 尾部 capture（capture_decode_graph）：
//   [D2H h_next_token←d_next_token(旧值)] → [embedding → 42 层 → lm_head →
//   argmax(→d_next_token 新值)] → [append d_seq[seq_len]（device 侧索引）] →
//   [seq_len += 1（device 侧）]
// 节点顺序与 eager 路径的提交顺序逐 op 一致（D2H 先于 forward），故
// mc_decode_one 的发放语义 bit 级不变：每次调用返回「上一步已决策」token。
// 流同步留在 graph 外（replay 后 cudaStreamSynchronize）。
#include "runtime/decode_executor.h"

#include <chrono>
#include <cstdio>

#include <cuda_runtime.h>

#include "kernels/sm120/kernels.h"
#include "runtime/debug_dump.h"
#include "runtime/error.h"
#include "runtime/forward.h"
#include "runtime/generated_model_config.h"
#include "runtime/internal_state.h"

namespace mc {

namespace {

inline uint32_t xorshift32(uint32_t x) {
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    return x;
}

// ---- mock forward：确定性 kernel（无权重环境回退，abi_smoke 契约不变）----

mc_status_t run_embedding_from_device_token(SessionImpl& s, cudaStream_t stream) {
    // mock: 不做计算。真实路径见 forward.cpp（embedding gather）。
    (void)s;
    (void)stream;
    return MC_OK;
}

mc_status_t run_decoder_layer(SessionImpl& s, int layer, cudaStream_t stream) {
    (void)s;
    (void)layer;
    (void)stream;
    return MC_OK;
}

mc_status_t run_final_norm_and_lm_head(SessionImpl& s, cudaStream_t stream) {
    (void)s;
    (void)stream;
    return MC_OK;
}

mc_status_t run_gpu_sampler(SessionImpl& s, cudaStream_t stream, int32_t force) {
    return launch_mock_next_token(s.d_next_token, s.seq_len, s.rng_state, force,
                                  stream);
}

// mock forward 一步：kernel + device token 链路 + 状态推进 + 计时。
mc_status_t decode_mock_step(SessionImpl& s, const mc_generation_config_t& config,
                              int32_t* out_token) {
    const auto t0 = std::chrono::steady_clock::now();
    (void)config; // 收尾判定已在 decode_one_run 统一完成（见其注释）

    // ---- forward：mock 单 kernel 代替 §8.1 全链路 ----
    mc_status_t rs = run_embedding_from_device_token(s, s.stream);
    if (rs != MC_OK) return rs;
    for (uint32_t layer = 0; layer < cfg::kNumLayers; ++layer) {
        rs = run_decoder_layer(s, (int)layer, s.stream);
        if (rs != MC_OK) return rs;
    }
    rs = run_final_norm_and_lm_head(s, s.stream);
    if (rs != MC_OK) return rs;
    rs = run_gpu_sampler(s, s.stream, -1); // 写 s.d_next_token（force 由
    if (rs != MC_OK) return rs;            // decode_one_run 的哨兵路径处理）

    // ---- device token 链路（§10.4）：token 追加进 device 序列缓冲 ----
    cudaError_t e = cudaMemcpyAsync(s.d_seq + s.seq_len, s.d_next_token,
                                    sizeof(int32_t), cudaMemcpyDeviceToDevice,
                                    s.stream);
    if (e != cudaSuccess) {
        cudaGetLastError();
        mc::set_error("mc_decode_one: token D2D append failed: %s",
                      cudaGetErrorString(e));
        return MC_E_CUDA;
    }
    // ---- 4 字节 D2H 到 pinned h_next_token → 同步 → 返回 host ----
    e = cudaMemcpyAsync(s.h_next_token, s.d_next_token, sizeof(int32_t),
                        cudaMemcpyDeviceToHost, s.stream);
    if (e != cudaSuccess) {
        cudaGetLastError();
        mc::set_error("mc_decode_one: token D2H copy failed: %s",
                      cudaGetErrorString(e));
        return MC_E_CUDA;
    }
    e = cudaStreamSynchronize(s.stream);
    if (e != cudaSuccess) {
        cudaGetLastError();
        mc::set_error("mc_decode_one: stream sync failed: %s", cudaGetErrorString(e));
        return MC_E_CUDA;
    }
    const int32_t tok = *s.h_next_token;
    if (tok < 0 || (uint32_t)tok >= cfg::kVocabSize) {
        mc::set_error("mc_decode_one: sampler produced out-of-range token %d", tok);
        return MC_E_INTERNAL;
    }

    // ---- 状态推进 + device state 镜像（h_state 是 session 成员，async 源安全）----
    s.seq_len += 1;
    s.rng_state = xorshift32(s.rng_state);
    s.h_state.seq_len = s.seq_len;
    s.h_state.next_token = (uint32_t)tok;
    s.h_state.rng_state = s.rng_state;
    e = cudaMemcpyAsync(s.d_state, &s.h_state, sizeof(SessionStateDevice),
                        cudaMemcpyHostToDevice, s.stream);
    if (e != cudaSuccess) {
        cudaGetLastError();
        mc::set_error("mc_decode_one: d_state H2D update failed: %s",
                      cudaGetErrorString(e));
        return MC_E_CUDA;
    }

    // ---- 计时与统计 ----
    const auto t1 = std::chrono::steady_clock::now();
    const uint64_t us = (uint64_t)std::chrono::duration_cast<
        std::chrono::microseconds>(t1 - t0).count();
    s.total.generated_tokens += 1;
    s.since_reset.generated_tokens += 1;
    s.total.decode_us += us;
    s.since_reset.decode_us += us;

    *out_token = tok;
    return MC_OK;
}

#if !MC_NATIVE_MOCK_FORWARD // ---- P2 真实路径（MC_NATIVE_MOCK_FORWARD=OFF）----

// decode step 的可捕获主体（warmup 与 capture 共用；不含 D2H —— 那是 graph
// 的首节点，由 capture_decode_graph 单独入图）：
//   forward（消费 d_next_token，位置 &d_state->seq_len，argmax 写回）→
//   append + seq_len+1（全 device 侧，固定地址）。
// 前置：host 已 ensure_pages_for(seq_len+1)。
mc_status_t decode_step_body(SessionImpl& s) {
    mc_status_t rs = run_real_forward(s, s.d_next_token, 1, /*allow_layer_dump=*/false);
    if (rs != MC_OK) return rs;
    return launch_decode_advance(s.d_seq, s.d_next_token, &s.d_state->seq_len,
                                 s.stream);
}

// graph 模式 decode 一步（§12.3 hot loop）：host 只做页增长检查 + 一次
// graph launch + 同步 + 读 pinned token。发放语义与 eager 完全一致
//（graph 首节点的 D2H 复制的是「上一步已决策」的旧 token）。
mc_status_t decode_graph_step(SessionImpl& s, const mc_generation_config_t& config,
                              int32_t* out_token) {
    const auto t0 = std::chrono::steady_clock::now();
    (void)config; // 收尾判定已在 decode_one_run 统一完成（见其注释）

    // ---- KV 页增长（host 侧、graph 外）：decode 期间页表内容可增长，kernel
    //      经固定指针运行时读取（§12.4「KV Page 增长：预容量，只改内容」）----
    mc_status_t rs = s.kv.ensure_pages_for(s.seq_len + 1, s.stream);
    if (rs != MC_OK) return rs;

    // ---- graph replay：D2H(旧 token) → forward → 决策（greedy argmax 或
    //      采样管线，按建期捕获的两图之一）→ append → seq_len+1 ----
    rs = (s.sampling_step ? s.graph_sample : s.graph).replay(s.stream);
    if (rs != MC_OK) return rs;
    cudaError_t e = cudaStreamSynchronize(s.stream);
    if (e != cudaSuccess) {
        cudaGetLastError();
        mc::set_error("mc_decode_one: graph replay sync failed: %s",
                      cudaGetErrorString(e));
        return MC_E_CUDA;
    }

    const int32_t emitted = *s.h_next_token; // 旧 token（graph 首节点已 D2H）
    if (emitted < 0 || (uint32_t)emitted >= cfg::kVocabSize) {
        mc::set_error("mc_decode_one: emitted token out of range (%d)", emitted);
        return MC_E_INTERNAL;
    }

    // ---- 状态推进：device 侧 seq_len 已由 graph 尾节点 +1；host 镜像同步 ----
    //（不回写 d_state：device rng 由采样 kernel 推进、cfg 由 decode_one
    // 每步写入 —— 整结构回写会破坏两者，见 internal_state.h 说明）
    s.seq_len += 1;
    s.rng_state = xorshift32(s.rng_state); // mock 链路兼容字段（真实路径未用）
    s.h_state.seq_len = s.seq_len;
    s.h_state.next_token = (uint32_t)emitted;

    const auto t1 = std::chrono::steady_clock::now();
    const uint64_t us = (uint64_t)std::chrono::duration_cast<
        std::chrono::microseconds>(t1 - t0).count();
    s.total.generated_tokens += 1;
    s.since_reset.generated_tokens += 1;
    s.total.decode_us += us;
    s.since_reset.decode_us += us;

    *out_token = emitted;
    return MC_OK;
}

// 真实 decode 一步（预决策/prefetch 语义，对齐 golden 步号约定）：
//   1) 发放上一步决策：D2H d_next_token → *out_token（argmax(step{t-1})，
//      即 golden step{t-1}.token；自然出 eos 时原样返回，不吞掉）
//   2) forward：embedding(d_next_token) → 42 层（position=seq_len，KV 写当前
//      位置）→ final_norm → lm_head → argmax → 新 d_next_token（= step{t}）
//   3) seq_len+1、统计（收尾判定在 decode_one_run 统一完成，见其注释）。
mc_status_t decode_real_step(SessionImpl& s, const mc_generation_config_t& config,
                             int32_t* out_token) {
    const auto t0 = std::chrono::steady_clock::now();
    (void)config;

    // ---- 1) 先把「上一轮已决策」的 token 发放给 host（4 字节 D2H）----
    cudaError_t e = cudaMemcpyAsync(s.h_next_token, s.d_next_token, sizeof(int32_t),
                                    cudaMemcpyDeviceToHost, s.stream);
    if (e != cudaSuccess) {
        cudaGetLastError();
        mc::set_error("mc_decode_one: token D2H copy failed: %s",
                      cudaGetErrorString(e));
        return MC_E_CUDA;
    }

    // ---- KV page 覆盖当前位置（decode 不新增页时为 no-op 快路径）----
    mc_status_t rs = s.kv.ensure_pages_for(s.seq_len + 1, s.stream);
    if (rs != MC_OK) return rs;

    // ---- 2) forward：消费 d_next_token（position=seq_len，KV 写入该位置）----
    rs = run_real_forward(s, s.d_next_token, 1, /*allow_layer_dump=*/false);
    if (rs != MC_OK) return rs;

    // ---- token 追加进 device 序列缓冲（§10.4：GPU 侧闭环）----
    e = cudaMemcpyAsync(s.d_seq + s.seq_len, s.d_next_token, sizeof(int32_t),
                        cudaMemcpyDeviceToDevice, s.stream);
    if (e != cudaSuccess) {
        cudaGetLastError();
        mc::set_error("mc_decode_one: token D2D append failed: %s",
                      cudaGetErrorString(e));
        return MC_E_CUDA;
    }

    e = cudaStreamSynchronize(s.stream);
    if (e != cudaSuccess) {
        cudaGetLastError();
        mc::set_error("mc_decode_one: stream sync failed: %s", cudaGetErrorString(e));
        return MC_E_CUDA;
    }
    const int32_t emitted = *s.h_next_token;
    if (emitted < 0 || (uint32_t)emitted >= cfg::kVocabSize) {
        mc::set_error("mc_decode_one: emitted token out of range (%d)", emitted);
        return MC_E_INTERNAL;
    }

    // ---- 3) 状态推进 + 镜像（只回写前 8B：seq_len/next_token —— device 侧
    //      rng/cfg 由采样 kernel 与 decode_one 的 config 拷贝独立管理，
    //      整结构回写会把 kernel 已推进的 RNG 状态倒回去）----
    s.seq_len += 1;
    s.rng_state = xorshift32(s.rng_state); // mock 链路兼容字段（真实路径未用）
    s.h_state.seq_len = s.seq_len;
    s.h_state.next_token = (uint32_t)emitted;
    e = cudaMemcpyAsync(s.d_state, &s.h_state, 2 * sizeof(uint32_t),
                        cudaMemcpyHostToDevice, s.stream);
    if (e != cudaSuccess) {
        cudaGetLastError();
        mc::set_error("mc_decode_one: d_state H2D update failed: %s",
                      cudaGetErrorString(e));
        return MC_E_CUDA;
    }

    const auto t1 = std::chrono::steady_clock::now();
    const uint64_t us = (uint64_t)std::chrono::duration_cast<
        std::chrono::microseconds>(t1 - t0).count();
    s.total.generated_tokens += 1;
    s.since_reset.generated_tokens += 1;
    s.total.decode_us += us;
    s.since_reset.decode_us += us;

    // 自然 eos（emitted==eos）原样返回，由 host 判停。
    *out_token = emitted;
    return MC_OK;
}

#endif // !MC_NATIVE_MOCK_FORWARD（真实路径）

} // namespace

#if !MC_NATIVE_MOCK_FORWARD
// P3（§12.3）：session_create 尾部调用（外部链接，session.cpp 使用）。流程：
//   1) eager warmup：在 seq_len=0 的初始状态下真实执行一步 decode（cuBLASLt
//      的 5 个 M=1 plan 实际跑一次 + 任何 lazy 初始化完成；此步的写入都是
//      可清理的涂鸦）。
//   2) Global 模式 capture：D2H（旧 token，pinned）→ decode_step_body。
//   3) EndCapture + Instantiate（steady-state 零分配：graph 在建期创建）。
//   4) 恢复干净初始状态（清 d_state/d_next_token/页所有权/host 计数）——
//      真实状态由随后的 prefill 重写；causal KV 保证 warmup 涂鸦不会被读到。
// 任何失败 → 调用方记日志并回退 eager（session 不失效，§12.4 Fallback）。
mc_status_t capture_decode_graph(SessionImpl& s) {
    if (!s.model->has_wpk) {
        mc::set_error("capture_decode_graph: real weights required (mock forward "
                      "is not graph-capturable)");
        return MC_E_GRAPH;
    }

    // warmup/capture 期间禁 dump：dump 的 D2H+sync+fwrite 会破坏 capture，
    // 且 warmup 的涂鸦输出无意义。建期编排调用，非 hot path。
    const int saved_dump_steps = dump::max_steps();
    dump::set_steps(0);

    mc_status_t rs = MC_OK;
    bool capture_begun = false;
    cudaError_t e = cudaSuccess;

    // ---- 1) eager warmup ----
    rs = s.kv.ensure_pages_for(1, s.stream);
    if (rs == MC_OK) rs = decode_step_body(s);
    if (rs == MC_OK) {
        e = cudaStreamSynchronize(s.stream);
        if (e != cudaSuccess) {
            cudaGetLastError();
            mc::set_error("capture_decode_graph: warmup sync failed: %s",
                          cudaGetErrorString(e));
            rs = MC_E_CUDA;
        }
    }

    // ---- 2) capture：D2H（旧 token）→ 完整 decode step ----
    if (rs == MC_OK) {
        rs = s.graph.begin_capture(s.stream);
        capture_begun = (rs == MC_OK);
    }
    if (rs == MC_OK) {
        e = cudaMemcpyAsync(s.h_next_token, s.d_next_token, sizeof(int32_t),
                            cudaMemcpyDeviceToHost, s.stream);
        if (e != cudaSuccess) {
            cudaGetLastError();
            mc::set_error("capture_decode_graph: token D2H copy failed: %s",
                          cudaGetErrorString(e));
            rs = MC_E_CUDA;
        }
    }
    if (rs == MC_OK) {
        s.in_capture = true; // forward 内的调试 sync/dump 禁用
        rs = decode_step_body(s);
        s.in_capture = false;
    }
    if (capture_begun) {
        // 无论成败必须终结 capture 状态；失败时 GraphManager 内部自清理。
        // P5：greedy 图的 PDL graph 边手工修补（链上全部消费 kernel 含 wait；
        // MC_PDL_OFF=1 时跳过 —— eager/graph 的 A/B 开关同源）。
        const bool pdl_edges = (getenv("MC_PDL_OFF") == nullptr);
        const mc_status_t rs_end = s.graph.end_capture(s.stream, pdl_edges);
        if (rs == MC_OK) rs = rs_end;
    }

    // ---- 2b) 采样变体图（§14.2）：同一 step 主体，尾部为采样管线
    //      （位图 memset/mark + penalty apply + 采样 kernel）替代 greedy
    //      argmax。capture 失败只影响采样（运行时回退 eager 采样），
    //      greedy 图不受累。kernel 均已由 warmup 暖过（capture 不执行）。----
    const mc_status_t rs_greedy = rs;
    rs = MC_OK;
    if (rs == MC_OK) {
        rs = s.graph_sample.begin_capture(s.stream);
        capture_begun = (rs == MC_OK);
    }
    if (rs == MC_OK) {
        e = cudaMemcpyAsync(s.h_next_token, s.d_next_token, sizeof(int32_t),
                            cudaMemcpyDeviceToHost, s.stream);
        if (e != cudaSuccess) {
            cudaGetLastError();
            mc::set_error("capture_decode_graph: token D2H copy failed: %s",
                          cudaGetErrorString(e));
            rs = MC_E_CUDA;
        }
    }
    if (rs == MC_OK) {
        s.in_capture = true;
        s.sampling_step = true; // capture 期间的尾部选择（baked 进本图）
        rs = decode_step_body(s);
        s.sampling_step = false;
        s.in_capture = false;
    }
    if (capture_begun) {
        const mc_status_t rs_end = s.graph_sample.end_capture(s.stream);
        if (rs == MC_OK) rs = rs_end;
    }
    if (rs != MC_OK) {
        s.graph_sample.destroy(); // 采样图失败：清理，运行时采样回退 eager
        fprintf(stderr,
                "[minicpm-native] sampling-graph capture failed (%s) -- "
                "sampling falls back to eager\n",
                mc_last_error());
        mc::clear_error();
    }
    rs = rs_greedy; // greedy 图的结果才是 session 的成败

    // ---- 3) 恢复干净初始状态（幂等于 session_create 的初始清零）----
    (void)cudaMemsetAsync(s.d_state, 0, sizeof(SessionStateDevice), s.stream);
    (void)cudaMemsetAsync(s.d_next_token, 0, sizeof(int32_t), s.stream);
    s.kv.release_all(); // 页所有权清空；显存/table 不动（下一次 prefill 重写）
    s.seq_len = 0;
    s.h_state = SessionStateDevice{};
    s.debug_step = 0;
    e = cudaStreamSynchronize(s.stream);

    dump::set_steps(saved_dump_steps);
    if (rs != MC_OK) return rs;
    if (e != cudaSuccess) {
        cudaGetLastError();
        mc::set_error("capture_decode_graph: cleanup sync failed: %s",
                      cudaGetErrorString(e));
        return MC_E_CUDA;
    }

    s.graph_mode = true;
    fprintf(stderr,
            "[minicpm-native] decode cuda graph captured: %zu nodes (replay = 1 "
            "cudaGraphLaunch/step)\n",
            s.graph.node_count());
    return MC_OK;
}
#endif // !MC_NATIVE_MOCK_FORWARD

mc_status_t decode_one_run(SessionImpl& s, const mc_generation_config_t* config,
                           int32_t* out_token) {
    // ---- 校验 ----
    if (config == nullptr) {
        mc::set_error("mc_decode_one: config is NULL");
        return MC_E_INVALID_ARGUMENT;
    }
    if (out_token == nullptr) {
        mc::set_error("mc_decode_one: out_token is NULL");
        return MC_E_INVALID_ARGUMENT;
    }
    if (config->struct_size != sizeof(*config)) {
        mc::set_error("mc_decode_one: config->struct_size=%u but expected %zu",
                      config->struct_size, sizeof(*config));
        return MC_E_INVALID_ARGUMENT;
    }
    if (config->max_new_tokens == 0) {
        mc::set_error("mc_decode_one: max_new_tokens must be > 0");
        return MC_E_INVALID_ARGUMENT;
    }
    if (s.seq_len == 0) {
        mc::set_error("mc_decode_one: sequence is empty (call mc_prefill first)");
        return MC_E_INVALID_ARGUMENT;
    }
    if (s.seq_len >= s.model->max_context_tokens) {
        mc::set_error("mc_decode_one: context is full (seq_len=%u, max=%u)", s.seq_len,
                      s.model->max_context_tokens);
        return MC_E_OUT_OF_MEMORY;
    }

    // ---- 收尾语义（2026-09 对拍修复，OpenAI/vLLM 对齐）：max_new_tokens
    //      是「内容 token 上限」。本轮（自 reset 起）已产出 N 个真实 token
    //      后的调用返回 eos 哨兵：不跑 forward、不推进 d_seq/seq_len、不计
    //      入 generated 统计（幂等）。host 收到哨兵即知达成上限
    //      （finish=length）。自然 eos（forward 的真实决策 == eos id）仍由
    //      下方正常路径原样返回并照常计数 —— 由 host 判停（finish=stop）。
    //      两种 eos 来源互不混淆：哨兵只在 generated == N 时出现，自然
    //      eos 出现时 generated < N（下一调用仍会产出真实 token）。
    //      旧语义（generated+1 >= max 即强制 eos）少发一个内容 token 且
    //      让 host 把截断误标为 stop —— 见 sampling_regression 的 R5。----
    if (s.since_reset.generated_tokens >= config->max_new_tokens) {
        *out_token = cfg::kEosStreamId;
        return MC_OK;
    }

    // RNG：从 config->seed 初始化一次（session 建立后首轮 decode）；
    // reset 后重新播种。host xorshift 仅供 mock 链路；真实采样用 device 侧
    // 64-bit PCG（splitmix64(seed) 播种 → &d_state->rng_state，graph 外
    // 一次性 H2D；之后 kernel 内推进，graph replay 安全）。
    if (!s.rng_seeded) {
        s.rng_state = (uint32_t)config->seed;
        s.rng_seeded = true;
    }
    if (s.model->has_wpk && !s.dev_rng_seeded) {
        s.dev_rng_state = splitmix64(config->seed);
        s.dev_rng_seeded = true;
        cudaError_t e = cudaMemcpyAsync(&s.d_state->rng_state, &s.dev_rng_state,
                            sizeof(uint64_t), cudaMemcpyHostToDevice, s.stream);
        if (e != cudaSuccess) {
            cudaGetLastError();
            mc::set_error("mc_decode_one: device RNG seed H2D failed: %s",
                          cudaGetErrorString(e));
            return MC_E_CUDA;
        }
    }

    // ---- P4+（§14.2/§12.4）：采样配置 → device 固定地址 ----
    // 激活规则（OpenAI 兼容）：temperature ≤ 0 或 top_k == 1 → greedy
    //（逐 bit 不变）；temperature > 0 即采样——top_k 未指定（0）时在 host
    // 侧规范化为 kSampMaxK（核内 top-p 兜底过滤，词表质量损失可忽略），
    // 超上限截断。graph replay 不含本拷贝（参数只改值不改指针）。
    if (s.model->has_wpk) {
        s.h_cfg->temperature = config->temperature;
        s.h_cfg->top_p = config->top_p;
        s.h_cfg->top_k = config->top_k == 0u
            ? kSampMaxK
            : (config->top_k > kSampMaxK ? kSampMaxK : config->top_k);
        s.h_cfg->rep_penalty = config->repetition_penalty;
        s.h_cfg->flags = 0;
        s.h_cfg->pad = 0;
        // P5：内容未变时跳过 H2D（graph 外小拷贝，~2µs/step host 开销；
        // 首次与变化时照常写 → 语义不变）
        const bool changed =
            s.cfg_dirty ||
            memcmp(&s.last_cfg, s.h_cfg, sizeof(SamplingConfigDevice)) != 0;
        if (changed) {
            s.last_cfg = *s.h_cfg;
            s.cfg_dirty = false;
        }
        cudaError_t ce =
            changed
                ? cudaMemcpyAsync(&s.d_state->cfg, s.h_cfg,
                                  sizeof(SamplingConfigDevice),
                                  cudaMemcpyHostToDevice, s.stream)
                : cudaSuccess;
        if (ce != cudaSuccess) {
            cudaGetLastError();
            mc::set_error("mc_decode_one: sampling config H2D failed: %s",
                          cudaGetErrorString(ce));
            return MC_E_CUDA;
        }
        s.sampling_step =
            (config->temperature > 0.f && config->top_k != 1u);
    }

#if MC_NATIVE_MOCK_FORWARD
    // mock 构建：永远走确定性 mock（保留给无权重环境；mock kernel 用 host
    // 参数，device cfg 无关 —— 采样字段在 mock 下不生效）
    return decode_mock_step(s, *config, out_token);
#else
    // 真实构建：有 wpk 走真实 BF16 路径；无 wpk 回退 mock（abi_smoke 契约）。
    // P3：graph capture 成功的 session 走单次 cudaGraphLaunch 的 replay 路径
    //（语义与 eager 逐 op 一致，见 decode_graph_step 注释）。
    // P4+：采样步用采样变体图；其 capture 失败时回退 eager 采样。
    if (!s.model->has_wpk) return decode_mock_step(s, *config, out_token);
    if (s.graph_mode && s.graph.available() &&
        (!s.sampling_step || s.graph_sample.available()))
        return decode_graph_step(s, *config, out_token);
    return decode_real_step(s, *config, out_token);
#endif
}

} // namespace mc
