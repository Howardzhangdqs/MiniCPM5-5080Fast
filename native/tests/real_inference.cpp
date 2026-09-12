// real_inference.cpp — P2 真实 BF16 推理 CTest。
//
// 流程：
//   1) 解析模型目录：MC_MODEL_DIR 环境变量 > 仓库 models/minicpm5-2b >
//      构建目录 wpk_model（make_wpk 产物）。model.wpk 缺失 → FAIL（return 1）：
//      真实构建下静默通过会掩盖 forward 根本没跑。
//   2) golden 目录：MC_GOLDEN_DIR > 仓库 tests/runtime/golden（非空）>
//      构建目录 local_golden（reference_forward 产物，非空）。均缺失 → FAIL
//      （return 1），理由同上。
//   3) golden 用例：读 input_ids.bin → mc_prefill → (S-1)×mc_decode_one；
//      开启 MC_DEBUG_DUMP_DIR 后与 golden 逐文件比对（cos + max abs err，double
//      累加）。比对策略（与 gpu_layerwise 一致，公共定义见
//      tests/golden_compare.h）：
//        - 层张量/final_hidden cos≥0.995；logits cos≥0.99（硬）；
//        - token 逐 step 精确对拍（汇总输出 matched/total）；
//        - 混沌放大例外：cos ∈ [0.99,0.995) 的 hidden 维张量不直接 FAIL，
//          需 硬下限之上 + 每 case ≤2 / 全局 ≤3 + 链恢复 + 端到端达标
//          （定因：跨实现合法舍入差经近平局 softmax 混沌放大，gpu_layerwise
//          case_zh layer8.attn_out cos=0.9909 实测解剖，非语义错误）；
//        - 输入分叉豁免（与混沌例外正交叠加、全局预算合并）：首个 token
//          失配步 golden logits 前两名 gap ≤ max(4×bf16ulp(|top1|), 0.25)
//          且该步 logits cos≥0.99 → 合法翻转，step>t 张量豁免、每 case ≤1
//          次；gap 超限 → FAIL（case_mixed step6 gap=0.125 实测命中）。
//      dump 取证：每 case 独立子目录 dump/{case}/（比对后归档，case 间只清
//      自己的子目录，flat 暂存区逐 case 清空）。
//   4) 压力用例（无 golden，跑通+stats+无 CUDA 错误）：prefill 2048+decode 100
//      （TPOT 速报）、prefill 2048 计时速报、prefill 8000+decode 16 长上下文；
//      压力段前经 mc_debug_dump_set_steps(0) 关闭 dump（避免 4096 步 ×522KB
//      磁盘洪水；env 只读一次，setenv 无效，故走内部开关）。
//   5) steady-state 纪律沿用（decode_us>0）。
#include "minicpm_runtime.h"

// additive debug 钩子（libminicpm_native 导出；不属于冻结 ABI）。
extern "C" void mc_debug_dump_set_steps(int steps);

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "tests/golden_compare.h"

namespace fs = std::filesystem;

#define REQUIRE(cond)                                                          \
    do {                                                                       \
        if (!(cond)) {                                                         \
            fprintf(stderr, "FAIL %s:%d: %s (last_error: %s)\n", __FILE__,     \
                    __LINE__, #cond, mc_last_error());                         \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

// 公共头只前向声明 opaque handle；消费者侧复制等价布局（单指针）。
struct mc_model { void* impl; };
struct mc_session { void* impl; };

namespace {

using golden_cmp::read_i32_file;

int g_failures = 0;
int g_exceptions_total = 0; // 全 case 例外合计：混沌 outlier + 分叉豁免（≤ kOutlierMaxGlobal）

// 目录/文件通用存在性（golden 目录解析用；golden_cmp::file_exists 仅适于文件）。
bool file_exists(const std::string& p) {
    std::error_code ec;
    return fs::exists(p, ec);
}

// 单 case 的策略判定结果（判定规则见文件头 / tests/golden_compare.h）。
struct CaseVerdict {
    bool hard_fail   = false; // 任一 hidden cos<0.99 / logits 不达标 / 非豁免 token 失配 / 文件缺失
    bool tokens_ok   = true;  // 全部受比对 step 的 token 精确一致（豁免步除外）
    int  outliers    = 0;     // cos ∈ [0.99, 0.995) 的 hidden 维张量数
    int  divergences = 0;     // 合法翻转豁免次数（≤ golden_cmp::kDivergenceMaxPerCase）
    bool recovery_ok = true;  // 链恢复
    bool ok() const {
        return !hard_fail && tokens_ok && outliers <= golden_cmp::kOutlierMaxPerCase &&
               divergences <= golden_cmp::kDivergenceMaxPerCase && recovery_ok;
    }
};

// case 开始前清空 dump flat 暂存区中的常规文件（上一 case 比对后已归档移走，
// 此处防御异常残留；各 case 的子目录不受影响）。
void clear_flat_dumps(const std::string& dump_root) {
    std::error_code ec;
    for (const auto& e : fs::directory_iterator(dump_root)) {
        if (e.is_regular_file(ec)) fs::remove(e.path(), ec);
    }
}

// case 比对完成后取证归档：flat 暂存区全部文件移入 dump_root/{case}/
// （每 case 独立子目录，case 间互不清除对方证据；子目录先清自己再归档）。
void archive_case_dumps(const std::string& dump_root, const std::string& cname) {
    std::error_code ec;
    const std::string arc = dump_root + "/" + cname;
    fs::remove_all(arc, ec); // 只清本 case 自己的子目录（历史运行残留）
    fs::create_directories(arc, ec);
    for (const auto& e : fs::directory_iterator(dump_root)) {
        if (e.is_regular_file(ec))
            fs::rename(e.path(), arc + "/" + e.path().filename().string(), ec);
    }
}

// 单个 golden 用例比对（golden 目录 vs dump 目录）。
// 输出：step0 逐层表（gpu_layerwise 风格）+ 逐 decode 步表 + 例外张量清单
// 与判定依据 + 汇总。cos double 累加；maxabs 一并报告。
CaseVerdict compare_case(const std::string& name, const std::string& golden,
                         const std::string& dump, uint32_t steps) {
    (void)steps; // 步数由 dump/golden 文件存在性驱动（逐 step 扫描）
    using namespace golden_cmp;
    CaseVerdict out;
    std::vector<std::string> outlier_detail;

    // ---- 收集并分类 golden 文件，逐项对拍 ----
    std::set<std::string> files; // golden 中全部 step/层文件
    for (const auto& e : fs::directory_iterator(golden)) {
        const std::string fn = e.path().filename().string();
        if (fn == "meta.json" || fn == "input_ids.bin") continue;
        files.insert(fn);
    }

    std::map<int, std::array<TensorVerdict, 3>> layers;   // step0 层张量
    std::map<int, TensorVerdict> fh;                      // step{t}.final_hidden
    std::map<int, TensorVerdict> lg;                      // step{t}.logits
    struct TokRow { int32_t gv = -1, dv = -2; bool gp = false, dp = false; };
    std::map<int, TokRow> tok;                            // step{t}.token
    std::vector<std::pair<std::string, TensorVerdict>> others; // 未分类 .f32（防御）

    for (const auto& fn : files) {
        const std::string gp = golden + "/" + fn;
        const std::string dp = dump + "/" + fn;
        const Classify c = classify(fn);
        switch (c.kind) {
        case Kind::LayerTensor:
            if (c.tensor >= 0)
                layers[c.layer][c.tensor] = compare_f32(gp, dp, kCosLayerTh);
            else
                others.push_back({fn, compare_f32(gp, dp, kCosLayerTh)});
            break;
        case Kind::FinalHidden:
            fh[c.step] = compare_f32(gp, dp, kCosLayerTh);
            break;
        case Kind::Logits:
            lg[c.step] = compare_f32(gp, dp, kCosLogitsTh);
            break;
        case Kind::Token: {
            TokRow r;
            const auto g = read_i32_file(gp);
            const auto d = read_i32_file(dp);
            r.gp = g.size() == 1;
            r.dp = d.size() == 1;
            if (r.gp) r.gv = g[0];
            if (r.dp) r.dv = d[0];
            tok[c.step] = r;
            break;
        }
        default: // OtherF32 / Unknown：按 hidden 维策略（防御，计入预算）
            others.push_back({fn, compare_f32(gp, dp, kCosLayerTh)});
            break;
        }
    }

    // ---- hidden 维张量三态判定（硬 FAIL / outlier / 通过；NaN 安全）----
    auto triage = [&](const std::string& what, const TensorVerdict& v) {
        if (!v.present) {
            out.hard_fail = true;
            printf("  HARD-FAIL %s: dump file missing or size mismatch\n", what.c_str());
        } else if (below_hard(v.cos)) {
            out.hard_fail = true;
            printf("  HARD-FAIL %s cos=%.6f maxabs=%.4g (below hard floor %.3f)\n",
                   what.c_str(), v.cos, v.maxabs, kCosLayerHard);
        } else if (!v.ok) {
            out.outliers++;
            char buf[160];
            snprintf(buf, sizeof(buf), "%s cos=%.6f maxabs=%.4g", what.c_str(), v.cos,
                     v.maxabs);
            outlier_detail.push_back(buf);
        }
    };

    // ---- step0 逐层表（与 gpu_layerwise 同风格）----
    printf("  step0 per-layer cosine (threshold %.3f, hard floor %.3f):\n",
           kCosLayerTh, kCosLayerHard);
    printf("  layer   hidden_in   attn_out    mlp_out\n");
    std::vector<TensorVerdict> chain; // 链 1：layer0..41 三张量 → step0.final_hidden
    for (const auto& [i, arr] : layers) {
        for (int k = 0; k < 3; ++k) triage("layer" + std::to_string(i) + "." +
                                               kTensorNames[k], arr[k]);
        const bool hard = below_hard(arr[0].cos) || below_hard(arr[1].cos) ||
                          below_hard(arr[2].cos);
        const bool below = !arr[0].ok || !arr[1].ok || !arr[2].ok;
        printf("  %-6d  %9.6f  %9.6f  %9.6f%s\n", i, arr[0].cos, arr[1].cos,
               arr[2].cos, row_mark(hard, below));
        for (int k = 0; k < 3; ++k) chain.push_back(arr[k]);
    }

    // ---- 输入分叉豁免判定：首个 token 值失配（此前全部一致）----
    // 文件缺失/尺寸不符（单 i32）仍为硬 FAIL（无例外）；仅「值翻转」可豁免。
    int t_div = -1;
    for (const auto& [t, r] : tok) {
        if (!r.gp || !r.dp) {
            out.hard_fail = true;
            printf("  HARD-FAIL step%d.token: golden/dump file missing or bad size\n",
                   t);
        } else if (r.gv != r.dv && t_div < 0) {
            t_div = t;
        }
    }
    DivergenceInfo div;
    bool div_active = false;
    if (t_div >= 0) {
        div = evaluate_divergence(golden, dump, t_div, tok[t_div].gv, tok[t_div].dv);
        if (!div.evaluated) {
            out.hard_fail = true;
            printf("  divergence @step%d: golden logits unreadable — cannot "
                   "certify flip -> FAIL\n",
                   t_div);
        } else if (div.legitimate) {
            div_active = true;
            out.divergences = 1;
            printf("  divergence @step%d: golden tok %d vs got %d — golden top1 "
                   "id=%d (%.4f) top2 id=%d (%.4f), gap=%.4f <= th %.4f "
                   "(4xbf16ulp(|top1|)=%.4f, floor %.2f), step logits cos=%.6f "
                   "—> LEGITIMATE FLIP (steps>%d exempt, per-case limit %d)\n",
                   t_div, div.golden_tok, div.got_tok, div.top1_id, div.top1,
                   div.top2_id, div.top2, div.gap, div.threshold,
                   golden_cmp::kDivergenceUlpMult * bf16_ulp(std::fabs(div.top1)),
                   golden_cmp::kDivergenceGapFloor, div.logits_cos, t_div,
                   golden_cmp::kDivergenceMaxPerCase);
        } else {
            out.hard_fail = true;
            printf("  divergence @step%d: golden tok %d vs got %d — gap=%.4f vs th "
                   "%.4f, logits cos=%.6f (floor %.2f) —> NOT excusable -> FAIL\n",
                   t_div, div.golden_tok, div.got_tok, div.gap, div.threshold,
                   div.logits_cos, kCosLogitsTh);
        }
    }

    // ---- decode 步表（final_hidden / logits / token）----
    // 豁免步（div_active 且 t > t_div）：张量与 token 均不计失败、不耗预算，
    // cos 照印供取证。t == t_div 的 logits cos 已由合法性判定覆盖（≥0.99）。
    printf("  decode steps:  final_hidden   logits    token\n");
    int token_match = 0, token_total = 0, token_exempt = 0;
    std::vector<TensorVerdict> fh_chain; // 链 2：step1.. 的 final_hidden 按 step 序
    for (int t = 0;; ++t) {
        const bool has_fh = fh.count(t) != 0;
        const bool has_lg = lg.count(t) != 0;
        const bool has_tok = tok.count(t) != 0;
        if (!has_fh && !has_lg && !has_tok) break;
        const bool exempt = div_active && t > t_div;
        char row[192];
        if (has_fh) {
            if (!exempt) {
                triage("step" + std::to_string(t) + ".final_hidden", fh[t]);
                if (t > 0) fh_chain.push_back(fh[t]); // step0 并入层链尾部
            }
            snprintf(row, sizeof(row), "%9.6f", fh[t].cos);
        } else {
            snprintf(row, sizeof(row), "%9s", "-");
        }
        if (has_lg) {
            if (!exempt && (!lg[t].present || !lg[t].ok)) {
                out.hard_fail = true;
                printf("  HARD-FAIL step%d.logits cos=%.6f (threshold %.3f, no "
                       "exception)\n",
                       t, lg[t].cos, kCosLogitsTh);
            }
            snprintf(row + strlen(row), sizeof(row) - strlen(row), "  %9.6f",
                     lg[t].cos);
        } else {
            snprintf(row + strlen(row), sizeof(row) - strlen(row), "  %9s", "-");
        }
        if (has_tok) {
            const auto& r = tok[t];
            if (exempt) {
                token_exempt++;
                snprintf(row + strlen(row), sizeof(row) - strlen(row),
                         "  %6d vs %-6d EXEMPT", r.gv, r.dv);
            } else if (div_active && t == t_div) {
                token_total++; // 受比对（期望翻转，不计 match）
                snprintf(row + strlen(row), sizeof(row) - strlen(row),
                         "  %6d vs %-6d FLIP", r.gv, r.dv);
            } else {
                token_total++;
                const bool ok = r.gv == r.dv;
                if (ok) token_match++;
                else {
                    out.tokens_ok = false; // 非豁免 token 失配 = FAIL
                    out.hard_fail = true;
                }
                snprintf(row + strlen(row), sizeof(row) - strlen(row),
                         "  %6d vs %-6d %s", r.gv, r.dv, ok ? "OK" : "MISMATCH");
            }
        }
        printf("  step%-4d %s%s\n", t, row, exempt ? "  [divergence-exempt]" : "");
        if (t > 100000) break; // 防御
    }

    // step0.final_hidden 接层链尾部；其余未分类张量只计预算（无链序）
    if (fh.count(0) != 0) chain.push_back(fh[0]);
    for (const auto& [nm, v] : others) triage(nm, v);

    // ---- 链恢复 ----
    out.recovery_ok = chain_recovery_ok(chain) && chain_recovery_ok(fh_chain);

    // ---- 例外张量清单与判定依据 ----
    for (const auto& d : outlier_detail)
        printf("  outlier: %s  [chaotic-amplification policy: above hard floor "
                "%.3f; requires per-case<=%d, global<=%d (merged with divergences), "
                "chain recovery, e2e ok (no unexempted token mismatch)]\n",
                d.c_str(), kCosLayerHard, kOutlierMaxPerCase, kOutlierMaxGlobal);

    const bool pass = out.ok();
    if (!pass) g_failures++;
    printf("  case '%s': %s (hard_fail=%d outliers=%d divergence=%d recovery=%s "
           "tokens %d/%d)\n",
           name.c_str(), pass ? "PASS" : "FAIL", (int)out.hard_fail, out.outliers,
           out.divergences, out.recovery_ok ? "OK" : "BROKEN", token_match,
           token_total);
    if (div_active)
        printf("           token flip detail: step%d golden=%d got=%d (top1 id=%d "
               "%.4f / top2 id=%d %.4f, gap=%.4f <= th %.4f); %d later step(s) "
               "exempt\n",
               t_div, div.golden_tok, div.got_tok, div.top1_id, div.top1,
               div.top2_id, div.top2, div.gap, div.threshold, token_exempt);
    return out;
}

mc_model_options_t real_options(uint32_t max_ctx) {
    mc_model_options_t o{};
    o.struct_size = sizeof(o);
    o.abi_version = MC_ABI_VERSION;
    o.device_id = 0;
    o.max_sessions = 1;
    o.max_context_tokens = max_ctx;
    o.precision = MC_BF16;
    o.kv_precision = MC_KV_BF16;
    o.enable_cuda_graph = 0;
    o.reserve_vram_bytes = 0;
    return o;
}

} // namespace

int main() {
    // ---- 模型目录解析 ----
    std::string model_dir;
    if (const char* env = getenv("MC_MODEL_DIR")) {
        model_dir = env;
    } else if (file_exists(std::string(MC_REPO_ROOT) + "/models/minicpm5-2b/model.wpk")) {
        model_dir = std::string(MC_REPO_ROOT) + "/models/minicpm5-2b";
    } else if (file_exists(std::string(MC_BUILD_DIR) + "/wpk_model/model.wpk")) {
        model_dir = std::string(MC_BUILD_DIR) + "/wpk_model";
        fprintf(stderr, "real_inference: using fixture wpk (%s)\n", model_dir.c_str());
    }
    if (model_dir.empty() || !file_exists(model_dir + "/model.wpk")) {
        fprintf(stderr,
                "real_inference: FAIL — model.wpk not found (MC_MODEL_DIR / repo "
                "models/ / build fixture); real forward cannot run\n");
        return 1; // 缺权重 = FAIL（不再静默 SKIP）
    }

    // ---- golden 目录解析 ----
    std::string golden_dir;
    if (const char* env = getenv("MC_GOLDEN_DIR")) {
        golden_dir = env;
    } else {
        const std::string repo_golden = std::string(MC_REPO_ROOT) + "/tests/runtime/golden";
        int cases = 0;
        if (file_exists(repo_golden)) {
            for (const auto& e : fs::directory_iterator(repo_golden))
                if (e.is_directory() &&
                    file_exists(e.path().string() + "/input_ids.bin"))
                    cases++;
        }
        if (cases > 0) {
            golden_dir = repo_golden;
        } else {
            const std::string local = std::string(MC_BUILD_DIR) + "/local_golden";
            int lcases = 0;
            if (file_exists(local)) {
                for (const auto& e : fs::directory_iterator(local))
                    if (e.is_directory() &&
                        file_exists(e.path().string() + "/input_ids.bin"))
                        lcases++;
            }
            if (lcases > 0) {
                golden_dir = local;
                fprintf(stderr, "real_inference: using local reference golden (%s)\n",
                        golden_dir.c_str());
            }
        }
    }

    const std::string dump_dir = std::string(MC_BUILD_DIR) + "/real_inference_dump";
    std::error_code ec;
    fs::remove_all(dump_dir, ec);
    fs::create_directories(dump_dir, ec);
    setenv("MC_DEBUG_DUMP_DIR", dump_dir.c_str(), 1);
    setenv("MC_DEBUG_DUMP_STEPS", "4096", 1); // dump 全部步（对比前即时读走）

    const uint32_t kMaxCtx = 8192;
    mc_model_t m{nullptr};
    mc_model_options_t o = real_options(kMaxCtx);
    REQUIRE(mc_model_load(model_dir.c_str(), &o, &m) == MC_OK);
    REQUIRE(m.impl != nullptr);

    // ================= golden 用例 =================
    if (!golden_dir.empty()) {
        std::vector<std::string> cases;
        for (const auto& e : fs::directory_iterator(golden_dir))
            if (e.is_directory() && file_exists(e.path().string() + "/input_ids.bin"))
                cases.push_back(e.path().filename().string());
        std::sort(cases.begin(), cases.end());
        printf("== real_inference: golden cases (dir=%s, %zu cases) ==\n",
               golden_dir.c_str(), cases.size());

        for (const auto& cname : cases) {
            const std::string cdir = golden_dir + "/" + cname;
            const auto ids = read_i32_file(cdir + "/input_ids.bin");
            REQUIRE(!ids.empty());
            // steps = max step{t}.token 的 t + 1（不解析 meta.json）
            uint32_t steps = 0;
            for (uint32_t t = 0;; ++t) {
                char buf[64];
                snprintf(buf, sizeof(buf), "/step%u.token", t);
                if (!file_exists(cdir + buf)) break;
                steps = t + 1;
            }
            if (steps == 0) {
                printf("  case '%s': no step tokens found, skipped\n", cname.c_str());
                continue;
            }

            // 每 case 独立 session；dump 取证：比对完成后 flat 暂存区归档到
            // dump_root/{case}/（case 间只清自己的子目录，互不覆盖证据）。
            clear_flat_dumps(dump_dir);
            mc_session_t s{nullptr};
            REQUIRE(mc_session_create(m, &s) == MC_OK);
            REQUIRE(mc_prefill(s, ids.data(), (uint32_t)ids.size()) == MC_OK);

            mc_generation_config_t g{};
            g.struct_size = sizeof(g);
            g.max_new_tokens = steps + 8; // 不因 force-eos 截断对比窗口
            g.temperature = 0.f;
            g.top_p = 1.f;
            g.top_k = 0;
            g.repetition_penalty = 1.f;
            g.seed = 1;
            int32_t tok = -1;
            for (uint32_t k = 1; k < steps; ++k) { // step1..step{S-1} 的 forward
                REQUIRE(mc_decode_one(s, &g, &tok) == MC_OK);
            }
            const CaseVerdict v = compare_case(cname, cdir, dump_dir, steps);
            g_exceptions_total += v.outliers + v.divergences;
            archive_case_dumps(dump_dir, cname);
            mc_session_destroy(s);
        }
        // 例外全局预算（混沌 outlier + 分叉豁免合并计数，策略见
        // tests/golden_compare.h）
        if (g_exceptions_total > golden_cmp::kOutlierMaxGlobal) {
            g_failures++;
            printf("  global exception budget (outliers+divergences) exceeded: %d > "
                   "%d\n",
                   g_exceptions_total, golden_cmp::kOutlierMaxGlobal);
        }
    } else {
        fprintf(stderr,
                "real_inference: FAIL — no golden available (repo "
                "tests/runtime/golden and local reference both absent; override "
                "with MC_GOLDEN_DIR); numeric comparison cannot run\n");
        mc_model_destroy(m);
        return 1; // 缺 golden = FAIL（不再静默通过）
    }

    // ================= 性能 + 压力用例 =================
    {
        printf("== real_inference: perf + stress (no golden) ==\n");
        // 压力段前关闭 dump（内部开关；env 已被静态缓存，setenv 无效）
        mc_debug_dump_set_steps(0);
        mc_session_t s{nullptr};
        REQUIRE(mc_session_create(m, &s) == MC_OK);

        std::vector<int32_t> prompt(2048);
        for (uint32_t i = 0; i < prompt.size(); ++i)
            prompt[i] = (int32_t)((i * 2654435761u + 7u) % 130560u);
        REQUIRE(mc_prefill(s, prompt.data(), (uint32_t)prompt.size()) == MC_OK);

        mc_generation_config_t g{};
        g.struct_size = sizeof(g);
        g.max_new_tokens = 200;
        g.seed = 1;
        int32_t tok = -1;
        const int kDecode = 100;
        for (int i = 0; i < kDecode; ++i)
            REQUIRE(mc_decode_one(s, &g, &tok) == MC_OK);
        REQUIRE(tok >= 0 && tok < 130560);

        mc_runtime_stats_t st{};
        REQUIRE(mc_get_stats(s, &st) == MC_OK);
        REQUIRE(st.prompt_tokens == 2048);
        REQUIRE(st.generated_tokens == (uint64_t)kDecode);
        REQUIRE(st.decode_us > 0);
        const double tpot_ms = (double)st.decode_us / 1000.0 / (double)kDecode;
        printf("  perf: prefill(2048) = %.1f ms (%.0f tok/s), decode(%d) TPOT = "
               "%.2f ms (%.0f tok/s)\n",
               (double)st.prefill_us / 1000.0, 2048.0 / ((double)st.prefill_us / 1e6),
               kDecode, tpot_ms, 1000.0 / tpot_ms);

        // 长上下文压力：reset 复用 → prefill 8000 + 16 decode
        REQUIRE(mc_session_reset(s) == MC_OK);
        std::vector<int32_t> long_prompt(8000);
        for (uint32_t i = 0; i < long_prompt.size(); ++i)
            long_prompt[i] = (int32_t)((i * 40503u + 19u) % 130560u);
        REQUIRE(mc_prefill(s, long_prompt.data(), (uint32_t)long_prompt.size()) ==
                MC_OK);
        g.max_new_tokens = 64;
        for (int i = 0; i < 16; ++i) REQUIRE(mc_decode_one(s, &g, &tok) == MC_OK);
        REQUIRE(mc_get_stats(s, &st) == MC_OK);
        REQUIRE(st.prompt_tokens == 2048 + 8000);
        REQUIRE(st.generated_tokens == (uint64_t)(kDecode + 16));
        cudaError_t ce = cudaDeviceSynchronize();
        REQUIRE(ce == cudaSuccess);
        printf("  stress: prefill(8000)+decode(16) OK, cumulative prompt=%llu "
               "generated=%llu peak_vram=%.2f GiB\n",
               (unsigned long long)st.prompt_tokens,
               (unsigned long long)st.generated_tokens,
               (double)st.peak_vram_bytes / (1024.0 * 1024.0 * 1024.0));
        mc_session_destroy(s);
    }

    mc_model_destroy(m);
    if (g_failures > 0) {
        fprintf(stderr, "real_inference: %d case(s) FAILED\n", g_failures);
        return 1;
    }
    fprintf(stderr, "real_inference: all done\n");
    return 0;
}
