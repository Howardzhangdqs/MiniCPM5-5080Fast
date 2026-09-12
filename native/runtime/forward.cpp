// forward.cpp — P2 真实 BF16 forward 实现。
//
// 单层执行图（计划 §8.1）：
//   residual ── norm1 ──> qkv GEMM ── rope(q 原地, k/v 入 KV 页) ──
//   attention(causal GQA, 页表间接) ── o GEMM ── 残差加 ── norm2 ──
//   gate_up GEMM ── swiglu ── down GEMM ── 残差加（融合下一层 norm1）
// 末层残差加融合 final_norm，取末行 → lm_head GEMM → bf16 logits →
// greedy argmax → d_next_token。
//
// 残差/归一化编排：residual 缓冲区即 residual stream；每次「残差加 + RMSNorm」
// 用融合 kernel（§8.3 融合 1）原地完成；layer0 的 norm1 用普通行版 RMSNorm。
#include "runtime/forward.h"

#include <cstdio>

#include <cuda_runtime.h>

#include "kernels/sm120/kernels.h"
#include "runtime/debug_dump.h"
#include "runtime/error.h"
#include "runtime/generated_model_config.h"
#include "runtime/gemv_tactic.h"
#include "runtime/internal_state.h"

namespace mc {

namespace {

mc_status_t dump_vec(const char* name, const uint16_t* d_row, cudaStream_t st) {
    char path[1024];
    snprintf(path, sizeof(path), "%s/%s", dump::dir(), name);
    return dump::write_bf16_as_f32(path, d_row, cfg::kHiddenSize, st);
}

} // namespace

mc_status_t run_real_forward(SessionImpl& s, const int32_t* d_ids, uint32_t T,
                              bool allow_layer_dump) {
    const ModelImpl* m = s.model;
    const WeightRefs& w = m->weights;
    cudaStream_t st = s.stream;
    auto& a = s.act;
    const uint32_t base = s.seq_len; // 调用时未累加
    const uint32_t H = cfg::kHiddenSize;
    // 分窗加载窗口（默认全量 [0,42)，行为与单窗版逐指令一致）：
    // layer_min/layer_max 来自 weight_loader 的 MC_LOAD_LAYERS_MIN/MAX。
    const uint32_t lmin = w.layer_min;
    const uint32_t lmax = w.layer_max;

    // dump 开关（一次静态判断，Release 默认关闭不进 hot path）
    const int dump_steps = dump::enabled() ? dump::max_steps() : 0;
    const uint32_t dump_step_index = s.debug_step; // 本 forward 的步号
    const bool step_dump = dump_steps > 0 && (int)dump_step_index < dump_steps;
    const bool layer_dump = step_dump && allow_layer_dump; // 仅 prefill(step0)

#define MC_CHECK(call)                                                        \
    do {                                                                      \
        mc_status_t _rs = (call);                                             \
        if (_rs != MC_OK) return _rs;                                         \
    } while (0)

    // 调试开关 MC_DEBUG_SYNC_OPS=1：每个算子后同步，把 async 故障定位到
    // 首个出错 op（默认关；一次 getenv，不进 Release hot path 语义）。
    // CUDA Graph capture 区间内禁止 stream sync（非法操作），期间强制跳过。
    const bool sync_each = (getenv("MC_DEBUG_SYNC_OPS") != nullptr) && !s.in_capture;
#define MC_DBG_SYNC(label)                                                    \
    do {                                                                       \
        if (sync_each) {                                                       \
            cudaError_t _e = cudaStreamSynchronize(st);                       \
            if (_e != cudaSuccess) {                                           \
                cudaGetLastError();                                           \
                mc::set_error("forward: sync after %s (layer loop) failed: %s",\
                              label, cudaGetErrorString(_e));                 \
                return MC_E_CUDA;                                             \
            }                                                                  \
        }                                                                      \
    } while (0)

    // ---- P5 perf 分解：段边界事件记录（MC_BREAKDOWN=1 的 session）----
    // 段口径（与 mc::kBdownSegNames 对齐）：
    //   [start] embgather [norm1+qkv | rope+kvwrite | attn | o+resnorm2 |
    //   gate_up | swiglu | down+resnorm(next)]×42 [lm_head] [sample]
    // eager：流内事件；graph capture：成为 event record 节点（replay 后
    // mc_debug_breakdown_collect 读相邻事件差）。n_used 溢出保护为丢弃
    //（分窗加载的截断链 collect 会因 n_used != 298 拒绝）。
#define MC_BD_REC()                                                            \
    do {                                                                       \
        if (s.bdown.enabled && s.bdown.n_used < mc::kBdownEvents)              \
            (void)cudaEventRecord(s.bdown.events[s.bdown.n_used++], st);       \
    } while (0)
    s.bdown.n_used = 0;
    MC_BD_REC(); // [0] start（embedding gather 之前）

    // P5 PDL：decode（T=1）kernel 链以 ProgrammaticStreamSerialization 启动
    //（W 预取与前驱 drain 重叠；graph capture 转为 programmatic 端口边）。
    // MC_PDL_OFF=1 A/B 对照。
    const bool pdl_on = T == 1 && (getenv("MC_PDL_OFF") == nullptr);
    auto gemm = [&](uint32_t M, uint32_t N, uint32_t K, const void* A, const void* B,
                    void* C) {
        // P3+：M=1（decode 全部 GEMM + prefill/decode 的 lm_head）且命中
        // decode 固定 shape 时走 tactic 赢家（建期快测选定：自研 GEMV 或
        // cuBLASLt；session 内恒定 → CUDA Graph 安全）。M>1 恒 cuBLASLt。
        if (M == 1) {
            const int si = mc::decode_shape_index(N, K);
            if (si >= 0 && m->gemv_tactic[si] != mc::kGemvTacticCublas) {
                return mc::run_decode_gemv(*m, s.act.gemv_partial, N, K,
                                           (const uint16_t*)A, (const uint16_t*)B,
                                           (uint16_t*)C, st, pdl_on);
            }
        }
        // S1：gemm workspace 用 per-session 切片（s.gemm_ws）—— 多 session
        // 的 stream 并发执行 cuBLASLt 时互不踩踏。大小恒 16MB（与 model 级
        // 相同）→ heuristic 过滤条件不变 → 同 algo → 数值逐 bit 不变。
        return m->gemm.gemm_bf16(st, s.gemm_ws, s.gemm_ws_bytes, M, N, K, A, B, C);
    };

    // P4：层内 GEMM（qkv/o/gate_up/down）。fp8 双拷贝模式下 decode（M=1）
    // 走自研 fp8 GEMV（tactic 固定 kGemvTacticFp8；MC_GEMV_OFF 可强制回
    // bf16 对照）；prefill（M>1）恒 BF16 原权（cuBLASLt）。lm_head 不在此
    // 列（无 fp8 拷贝，恒走上面的 bf16 路径）。
    auto gemm_layer = [&](uint32_t M, uint32_t N, uint32_t K,
                          const uint16_t* w_bf16, const uint8_t* w_fp8,
                          const float* w_scale, const void* B, void* C) {
        if (M == 1 && w_fp8 != nullptr) {
            const int si = mc::decode_shape_index(N, K);
            if (si >= 0 && m->gemv_tactic[si] == mc::kGemvTacticFp8) {
                return launch_gemv_fp8(w_fp8, w_scale, (const uint16_t*)B,
                                       (uint16_t*)C, N, K, st, pdl_on);
            }
        }
        return gemm(M, N, K, w_bf16, B, C);
    };

    // ---- P5 融合 prologue（decode T=1）：residual add + RMSNorm 并入后续
    //      GEMV（qkv 的 norm1 / gate_up 的 norm2），消除 82/84 次独立
    //      resnorm kernel（单 block 小 kernel 纯延迟，实测 6.04µs/次）。
    //      数值逐 bit 等价（见 gemv*_prologue_kernel 注释）；residual 流
    //      ping-pong：res_a（=a.residual，embedding 入口）↔ res_b（alt 行）。
    //      决策为「整链」开关（qkv 与 gate_up 必须同开同关 —— 混合状态会
    //      破坏 ping-pong 不变式）：fp8 路径（全部层内 GEMV 融合）或 bf16
    //      的 qkv+gate_up tactic 均为自研 rows（cuBLASLt 不可融合）。
    //      MC_FUSED_GEMV_OFF=1 关闭（A/B 对照 → 独立 resnorm + 常规 GEMV）。----
    const bool fused_env = (getenv("MC_FUSED_GEMV_OFF") == nullptr);
    uint16_t* res_a = a.residual;
    uint16_t* res_b = a.residual_alt;
    // P5：lm_head 也走融合 prologue（末层 final_norm 并入；fp8 lm_head 段
    // 存在时）。step_dump 需要 a.normed（final_hidden dump）→ dump 步保留
    // 独立 tail（kernel 序按模式一致，dump 关闭时逐 bit 相同）。
    const bool lmhead_fused = fused_env && T == 1 && w.fp8_lm_head != nullptr &&
                              w.lm_head != nullptr && dump_steps == 0;
    const bool layers_fused =
        fused_env && T == 1 &&
        (w.fp8_qkv[lmin] != nullptr ||
         (m->gemv_tactic[0] == mc::kGemvTacticRows &&
          m->gemv_tactic[2] == mc::kGemvTacticRows));
    auto qkv_fused = [&](uint32_t i) {
        return layers_fused && i > lmin; // 首层 norm1 由独立 rmsnorm 产出
    };
    auto gate_up_fused = [&](uint32_t) { return layers_fused; };

    // ---- embedding gather：residual 流起点（layer0 输入）----
    // 分窗：embedding 未加载（种子模式）时 residual 已由测试钩子
    // mc_debug_seed_residual 预置（上一窗输出的全行残差流），跳过 gather。
    if (w.embedding != nullptr) {
        MC_CHECK(launch_embedding_gather(w.embedding, d_ids, a.residual, (int32_t)H, T,
                                          st));
        MC_DBG_SYNC("embedding_gather");
    }
    MC_BD_REC(); // [1] emb（未加载 embedding 的窗口链中为 0时长占位）

    // ---- KV 容量护栏（原 rope/attention launcher 的校验移至此处；host 侧
    //      base 值检查一次。P3：kernel 改从 &s.d_state->seq_len 运行时读取
    //      base —— 两种来源在编排不变式下恒相等：run_real_forward 被调用时
    //      d_state.seq_len == s.seq_len == base（prefill/decode 均在调用后才
    //      推进 seq_len），graph replay 同样成立）----
    if ((uint64_t)base + T >
        (uint64_t)s.kv.layout().max_pages * KvLayout::kTokensPerPage) {
        mc::set_error("forward: positions %u..%u exceed KV capacity (%u pages)", base,
                      base + T - 1, s.kv.layout().max_pages);
        return MC_E_OUT_OF_MEMORY;
    }

    for (uint32_t i = lmin; i < lmax; ++i) {
        // 分窗护栏：窗外层指针为 nullptr，绝不可触碰（release 下也生效）。
        if (w.norm1[i] == nullptr || w.qkv[i] == nullptr || w.o[i] == nullptr ||
            w.norm2[i] == nullptr || w.gate_up[i] == nullptr || w.down[i] == nullptr) {
            mc::set_error("forward: layer %u weights not loaded (window [%u,%u))", i,
                          lmin, lmax);
            return MC_E_INTERNAL;
        }
        const uint16_t* residual_last = a.residual + (size_t)(T - 1) * H;
        if (layer_dump) {
            char name[64];
            snprintf(name, sizeof(name), "layer%u.hidden_in.f32", i);
            MC_CHECK(dump_vec(name, residual_last, st));
        }

        // ---- norm1（窗口首层：普通 RMSNorm；此后由上一轮融合产出）----
        // P5：融合路径下首层 norm1 并入 qkv 的纯 norm prologue 变体。
        if (i == lmin && !layers_fused) {
            MC_CHECK(launch_rmsnorm_rows(a.residual, w.norm1[lmin], a.normed, T,
                                          (int32_t)H, cfg::kRmsEps, st));
        }
        // i>lmin：a.normed 已由第 i-1 层末尾的融合残差+norm1(i) 产出

        // ---- QKV：normed[T,2048] · qkv_w[2560,2048]ᵀ → qkv[T,2560] ----
        // P5：i>lmin 融合「上一层 down 残差 + norm1(i)」prologue（res_b +
        // down_out → res_a）；i==lmin 融合「纯 norm1」变体（addend=nullptr，
        // 与 rmsnorm_rows 数值序逐 bit 一致；res_in=res_a=embedding 残差）。
        {
            const uint32_t qkv_n = cfg::kNumQueryHeads * cfg::kHeadDim +
                                   2 * cfg::kNumKvHeads * cfg::kHeadDim;
            if (layers_fused && i == lmin && w.fp8_qkv[i] != nullptr) {
                MC_CHECK(launch_gemv_fp8_prologue(w.fp8_qkv[i], w.scale_qkv[i],
                                                  res_a, nullptr, w.norm1[i],
                                                  res_b, a.qkv, qkv_n, H,
                                                  cfg::kRmsEps, st, pdl_on));
            } else if (layers_fused && i == lmin) {
                MC_CHECK(launch_gemv_rows_prologue(w.qkv[i], res_a, nullptr,
                                                   w.norm1[i], res_b, a.qkv,
                                                   qkv_n, H, cfg::kRmsEps, st,
                                                   pdl_on));
            } else if (qkv_fused(i) && w.fp8_qkv[i] != nullptr) {
                MC_CHECK(launch_gemv_fp8_prologue(w.fp8_qkv[i], w.scale_qkv[i],
                                                  res_b, a.down_out, w.norm1[i],
                                                  res_a, a.qkv, qkv_n, H,
                                                  cfg::kRmsEps, st, pdl_on));
            } else if (qkv_fused(i)) {
                MC_CHECK(launch_gemv_rows_prologue(w.qkv[i], res_b, a.down_out,
                                                   w.norm1[i], res_a, a.qkv,
                                                   qkv_n, H, cfg::kRmsEps, st,
                                                   pdl_on));
            } else {
                MC_CHECK(gemm_layer(T, qkv_n, H, w.qkv[i], w.fp8_qkv[i],
                                    w.scale_qkv[i], a.normed, a.qkv));
            }
        }
        MC_DBG_SYNC("qkv gemm");
        MC_BD_REC(); // 层段 1/7：norm1+qkv（首层含独立 norm1 kernel；融合
                     // 路径 = prologue kernel，i>lmin 时含上一层残差归一）

        // ---- RoPE(q 原地；k/v 写 KV 页，位置 base..base+T-1) ----
        // P3：base 从 &s.d_state->seq_len（固定 device 地址）运行时读取。
        // P5-C：decode（T=1）默认把 q/k 旋转 + K/V 写池吸收进 attention
        // phase0（寄存器内 (d,d±64) 同 lane，kernel 见 attention_decode.cu；
        // 输出/池/qkv 三向逐位锚 = kernel_ops attention_rope_phase0）→
        // rope kernel 从 decode 链消失。回退条件与 launcher 的路径选择
        // 用同一判定（mc_attn_kv_major_enabled）+ 表可用性：
        //   - MC_ATTN_KV_OFF=1（A/B）→ 保留旧链；
        //   - MC_ROPE_TABLE_OFF=1（表的 A/B）→ phase0 需要建期表，同样保留。
        // prefill（T>1）恒走 rope kernel（多 token 路径不变）。
        const bool attn_phase0 =
            T == 1 && mc_attn_kv_major_enabled() &&
            s.act.rope_inv_freq != nullptr &&
            getenv("MC_ROPE_TABLE_OFF") == nullptr;
        if (!attn_phase0) {
            MC_CHECK(launch_rope_kv(a.qkv, T, &s.d_state->seq_len, i,
                                    s.d_page_table, (uint16_t*)s.kv_pool,
                                    s.kv.layout(), cfg::kRopeTheta, st,
                                    (getenv("MC_ROPE_TABLE_OFF") == nullptr)
                                        ? s.act.rope_inv_freq
                                        : nullptr,
                                    pdl_on));
        }
        MC_DBG_SYNC("rope_kv");
        MC_BD_REC(); // 层段 2/7：rope+kvwrite

        // ---- causal GQA attention（含自身位置）→ ctx[T,2048] ----
        // P3+：decode（T=1）走 KV 分块两段式（P5-B 起 kv-major chunk ×
        // merge，短 kernel 大并行）；prefill（T>1）保持原统一流式 kernel。
        if (T == 1) {
            MC_CHECK(launch_attention_decode(a.qkv, &s.d_state->seq_len, i,
                                             s.d_page_table,
                                             (const uint16_t*)s.kv_pool,
                                             s.kv.layout(), s.act.attn_partials,
                                             s.attn_max_chunks, a.ctx, st, pdl_on,
                                             attn_phase0 ? s.act.rope_inv_freq
                                                         : nullptr,
                                             // P8：会话 chunk 档位 + 两级 merge
                                             //（merge_a→merge_b）生产路径
                                             s.attn_chunk_pos,
                                             s.act.attn_partials2));
        } else if (mc_prefill_flash_enabled() && T >= 384u &&
                   s.act.prefill_partials != nullptr) {
            // P3：prefill flash-HMMA 路径（opt-in，MC_PREFILL_FLASH_ON=1）。
            // host 双路径：T < 384 走旧 kernel（小 T 下 flash 块数不足，
            // split-KV 也救不回 —— 见 kernel_ops/P4 基准）；T ≥ 384 走
            // flash（S_kv=prefill_flash_split_kv：T ≥ 2048 直写、否则
            // split-KV partials → prefill_merge）。host 分支合法：prefill
            // 无 CUDA Graph 约束（decode graph 仅 T=1）；env 静态一次
            // 读取 → 进程内恒定。partials 非空判定为 mock 会话的防御
            //（real 模式 compute_sizes 恒 40MB 切片）。
            MC_CHECK(launch_attn_prefill_flash(
                a.qkv, T, &s.d_state->seq_len, i, s.d_page_table,
                (const uint16_t*)s.kv_pool, s.kv.layout(), a.ctx,
                s.act.prefill_partials, prefill_flash_split_kv(T), st));
        } else {
            MC_CHECK(launch_attention_prefill(a.qkv, T, &s.d_state->seq_len, i,
                                              s.d_page_table,
                                              (const uint16_t*)s.kv_pool,
                                              s.kv.layout(), a.ctx, st));
        }
        MC_DBG_SYNC("attention");
        MC_BD_REC(); // 层段 3/7：attention（decode=chunk+merge）

        // ---- O：ctx[T,2048] · o_w[2048,2048]ᵀ → o_out[T,2048] ----
        MC_CHECK(gemm_layer(T, H, H, w.o[i], w.fp8_o[i], w.scale_o[i], a.ctx,
                            a.o_out));
        MC_DBG_SYNC("o gemm");

        // ---- 残差加 + norm2（融合，原地 residual += o_out）→ normed ----
        // P5：gate_up 走融合 prologue 时跳过（prologue 从 res_a + o_out
        // 直接产出 normed；残差流由 prologue 写入 res_b）。
        if (!gate_up_fused(i)) {
            MC_CHECK(launch_residual_add_rmsnorm(a.residual, a.o_out, w.norm2[i],
                                                 a.normed, T, (int32_t)H,
                                                 cfg::kRmsEps, st, pdl_on));
        }
        MC_DBG_SYNC("residual_add_rmsnorm(norm2)");
        MC_BD_REC(); // 层段 4/7：o+残差（含融合 norm2；融合路径移入 gate_up prologue）
        if (layer_dump) {
            // attn_out = o 投影并入残差后的 residual 末行（python 参考
            // reference_forward.py：x = add_bf16(x, ao) 后收集 attn_out）
            char name[64];
            snprintf(name, sizeof(name), "layer%u.attn_out.f32", i);
            MC_CHECK(dump_vec(name, residual_last, st));
        }

        // ---- gate/up：normed[T,2048] · gate_up_w[12288,2048]ᵀ → [T,12288] ----
        // P5：融合「o 残差 + norm2」prologue（res_a + o_out → res_b）。
        if (gate_up_fused(i) && w.fp8_gate_up[i] != nullptr) {
            MC_CHECK(launch_gemv_fp8_prologue(
                w.fp8_gate_up[i], w.scale_gate_up[i], res_a, a.o_out, w.norm2[i],
                res_b, a.gate_up, 2 * cfg::kIntermediateSize, H, cfg::kRmsEps, st,
                pdl_on));
        } else if (gate_up_fused(i)) {
            MC_CHECK(launch_gemv_rows_prologue(
                w.gate_up[i], res_a, a.o_out, w.norm2[i], res_b, a.gate_up,
                2 * cfg::kIntermediateSize, H, cfg::kRmsEps, st, pdl_on));
        } else {
            MC_CHECK(gemm_layer(T, 2 * cfg::kIntermediateSize, H, w.gate_up[i],
                                w.fp8_gate_up[i], w.scale_gate_up[i], a.normed,
                                a.gate_up));
        }
        MC_DBG_SYNC("gate_up gemm");
        MC_BD_REC(); // 层段 5/7：gate_up（norm2 已融合进段 4）

        // ---- SwiGLU → down（P5：T=1 时融合为单 kernel；数值与独立
        //      swiglu_gated + GEMV 串行逐 bit 一致）----
        // ---- down：mlp_mid[T,6144] · down_w[2048,6144]ᵀ → down_out[T,2048] ----
        // 注：swiglu→down 融合为 opt-in（MC_SWIGLU_FUSE=1）：实测 g/up 行的
        // 25MB L2 重读成本 > 省下的 kernel+间隙（3.605 vs 3.567 ms ctx16），
        // 默认独立 swiglu + down GEMV；kernel 保留供后续 L2 策略实验。
        if (layers_fused && getenv("MC_SWIGLU_FUSE_ON") != nullptr &&
            w.fp8_down[i] != nullptr) {
            MC_CHECK(launch_gemv_fp8_swiglu(w.fp8_down[i], w.scale_down[i],
                                            a.gate_up, a.down_out, H,
                                            cfg::kIntermediateSize, st, pdl_on));
        // 注：BF16 下默认融合（W 流 25MB vs g/u 重读 12.6MB —— 实测净收益
        // ~50µs/step；与 fp8 相反）。MC_SWIGLU_FUSE_OFF=1 对照。
        } else if (layers_fused && getenv("MC_SWIGLU_FUSE_OFF") == nullptr &&
                   m->gemv_tactic[3] == mc::kGemvTacticRows) {
            MC_CHECK(launch_gemv_rows_swiglu(w.down[i], a.gate_up, a.down_out, H,
                                             cfg::kIntermediateSize, st, pdl_on));
        } else {
            // ---- SwiGLU（行拼拆分）→ mlp_mid[T,6144] ----
            MC_CHECK(launch_swiglu_gated(a.gate_up, a.mlp_mid, T,
                                         cfg::kIntermediateSize, st, pdl_on));
            MC_DBG_SYNC("swiglu");
            MC_CHECK(gemm_layer(T, H, cfg::kIntermediateSize, w.down[i],
                                w.fp8_down[i], w.scale_down[i], a.mlp_mid,
                                a.down_out));
        }
        MC_BD_REC(); // 层段 6/7：swiglu（融合路径并入 down kernel）
        MC_DBG_SYNC("down gemm");

        // ---- 残差加 + 下一层 norm1 融合（末层则融合 final_norm）----
        // 截断层（i+1 == lmax < 42）用边界 norm1[lmax]（loader 已加载）；
        // 窗口未到末层时不产生 final_hidden / logits。
        // P5：T=1 且下一层 qkv 走融合 prologue 时跳过本 kernel（prologue
        // 已涵盖）；末层（final_norm → lm_head 消费 a.normed）不可跳过，
        // 在 res_b 上原地收尾（lm_head 不融合）。
        const uint16_t* next_norm_w =
            (i + 1 < cfg::kNumLayers) ? w.norm1[i + 1] : w.final_norm;
        if (next_norm_w == nullptr) {
            mc::set_error("forward: boundary norm weight for layer %u tail is NULL "
                          "(window [%u,%u))",
                          i, lmin, lmax);
            return MC_E_INTERNAL;
        }
        const bool tail_fused_into_next =
            ((i + 1 < cfg::kNumLayers) && qkv_fused(i + 1)) ||
            ((i + 1 == cfg::kNumLayers) && lmhead_fused);
        if (!tail_fused_into_next) {
            uint16_t* res_cur = layers_fused ? res_b : a.residual;
            MC_CHECK(launch_residual_add_rmsnorm(res_cur, a.down_out, next_norm_w,
                                                 a.normed, T, (int32_t)H,
                                                 cfg::kRmsEps, st, pdl_on));
        }
        MC_DBG_SYNC("residual_add_rmsnorm(next norm1/final)");
        MC_BD_REC(); // 层段 7/7：down+残差（融合路径：残差归一移入下层 qkv prologue）
        if (layer_dump) {
            // mlp_out = down 投影并入残差后的 residual 末行（python 参考：
            // x = add_bf16(x, d) 后收集 mlp_out）
            char name[64];
            snprintf(name, sizeof(name), "layer%u.mlp_out.f32", i);
            MC_CHECK(dump_vec(name, residual_last, st));
        }
    }

    // ---- 分窗截断（lmax < 42）：residual 全行 = 本窗输出残差流。
    // 种子链：gpu_layerwise 把它喂给下一窗（层数学只依赖进入的残差流，
    // 窗口链 == 全量 forward 的精确分解）。无 final_norm/lm_head 尾部。
    if (lmax < cfg::kNumLayers) {
        if (step_dump && dump::seed_out()) {
            char path[1024];
            snprintf(path, sizeof(path), "%s/window_seed.bf16", dump::dir());
            MC_CHECK(dump::write_bf16_raw(path, a.residual, (size_t)T * H, st));
        }
        s.debug_step += 1;
        return MC_OK;
    }

    // ---- 尾部（lmax == 42）：final_hidden = final_norm 后的末行 ----
    // 全量/常规窗：循环结束时 a.normed 已是「末层残差 + final_norm」输出；
    // 空层窗（lmin == lmax == 42，纯头部验证）：直接 RMSNorm(residual)。
    if (lmin == lmax) {
        MC_CHECK(launch_rmsnorm_rows(a.residual, w.final_norm, a.normed, T, (int32_t)H,
                                     cfg::kRmsEps, st));
    }
    if (step_dump) {
        char name[64];
        snprintf(name, sizeof(name), "step%u.final_hidden.f32", dump_step_index);
        MC_CHECK(dump_vec(name, a.normed + (size_t)(T - 1) * H, st));
    }

    // ---- lm_head 未加载（首窗截断）：无 logits/argmax，到此为止 ----
    if (w.lm_head == nullptr) {
        s.debug_step += 1;
        return MC_OK;
    }

    // ---- lm_head：末行 normed[1,2048] · lm_head_w[130560,2048]ᵀ → logits bf16 ----
    // P5：fp8 包携带 lm_head 量化段（dtype=2，离线工具链/测试 fixture 产出）
    // 时 decode（M=1）走 gemv_fp8（bf16 原权仍在：prefill M>1 不受影响；
    // MC_FP8_LMHEAD_OFF=1 强制回 bf16 对照）。分支建期解析 → graph 兼容。
    // 融合路径：final_norm 并入 prologue（res_b + down_out；res_out=res_a
    // 为牺牲写 —— 步末无人再读 residual）。
    if (lmhead_fused) {
        MC_CHECK(launch_gemv_fp8_prologue(w.fp8_lm_head, w.scale_lm_head, res_b,
                                          a.down_out, w.final_norm, res_a,
                                          a.logits, cfg::kVocabSize, H,
                                          cfg::kRmsEps, st, pdl_on));
    } else if (T == 1 && w.fp8_lm_head != nullptr) {
        MC_CHECK(launch_gemv_fp8(w.fp8_lm_head, w.scale_lm_head,
                                 a.normed + (size_t)(T - 1) * H, a.logits,
                                 cfg::kVocabSize, H, st));
    } else {
        MC_CHECK(gemm(1, cfg::kVocabSize, H, w.lm_head,
                      a.normed + (size_t)(T - 1) * H, a.logits));
    }
    MC_BD_REC(); // 尾段 1/2：final_norm+lm_head（final_norm 已融合进末层段 7）

    // ---- greedy argmax → d_next_token（device token 链路，§10.4）----
    // P4+（§14.2）：末尾决策 —— 采样步（decode_one 按激活规则置
    // s.sampling_step：temperature>0 且 top_k>1）走 rep-penalty（memset+mark+
    // apply→fp32）+ top-k/top-p/temperature 采样（kernel 内读 device cfg +
    // RNG，graph 兼容）；greedy 步（及 prefill 预决策）走原 argmax，逐 bit 不变。
    if (s.sampling_step) {
        (void)cudaMemsetAsync(s.act.rep_bitmap, 0, (size_t)cfg::kVocabSize, st);
        MC_CHECK(launch_rep_penalty_mark(s.d_seq, &s.d_state->seq_len,
                                         s.act.rep_bitmap,
                                         s.model->max_context_tokens, st));
        MC_CHECK(launch_rep_penalty_apply(a.logits, s.act.rep_bitmap,
                                          &s.d_state->cfg, s.act.logits_f32,
                                          (int32_t)cfg::kVocabSize, st));
        MC_CHECK(launch_sampling_pipeline(s.act.logits_f32, &s.d_state->cfg,
                                          &s.d_state->rng_state, s.d_next_token,
                                          (int32_t)cfg::kVocabSize,
                                          s.act.samp_partial_v, s.act.samp_partial_i,
                                          st));
    } else {
        const bool argmax2 = getenv("MC_ARGMAX2_OFF") == nullptr;
        MC_CHECK(launch_sampling_greedy_bf16(
            a.logits, (int32_t)cfg::kVocabSize, s.d_next_token, st,
            argmax2 ? s.act.gemv_partial : nullptr,
            argmax2 ? reinterpret_cast<int32_t*>(s.act.gemv_partial + 2048)
                    : nullptr,
            pdl_on));
    }
    MC_BD_REC(); // 尾段 2/2：采样（greedy argmax 或采样管线）

    // ---- step dump：logits + token（token = 本步 argmax = 新 d_next_token）----
    if (step_dump) {
        char path[1024];
        snprintf(path, sizeof(path), "%s/step%u.logits.f32", dump::dir(),
                 dump_step_index);
        MC_CHECK(dump::write_bf16_as_f32(path, a.logits, cfg::kVocabSize, st));
        int32_t htok = -1;
        cudaError_t e = cudaMemcpyAsync(&htok, s.d_next_token, sizeof(int32_t),
                                        cudaMemcpyDeviceToHost, st);
        if (e != cudaSuccess || (e = cudaStreamSynchronize(st)) != cudaSuccess) {
            cudaGetLastError();
            mc::set_error("forward: token dump copy failed: %s", cudaGetErrorString(e));
            return MC_E_CUDA;
        }
        snprintf(path, sizeof(path), "%s/step%u.token", dump::dir(), dump_step_index);
        MC_CHECK(dump::write_token(path, htok));
    }

    s.debug_step += 1;
    return MC_OK;
#undef MC_CHECK
#undef MC_DBG_SYNC
#undef MC_BD_REC
}

} // namespace mc
