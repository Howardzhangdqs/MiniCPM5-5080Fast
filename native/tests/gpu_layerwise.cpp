// gpu_layerwise.cpp — P2 分窗 GPU 逐层验证 CTest（显存受限环境）。
//
// 背景：同机其它进程占用 GPU 显存（~3.4GB 空闲且波动），无法全量加载
// 5.04GB BF16 权重跑 real_inference。本测试利用 Transformer 层因果独立
// 性（layer i 只依赖进入它的残差流），把 42 层切成 ≤N 层的窗口链，逐窗
// 加载真实权重跑 prefill（step0），与 python golden 逐层对拍：
//
//   窗口 0：embedding + layers [0,N)   残差流从 token ids 起步（正常 gather）
//   窗口 k：layers [kN,(k+1)N)         种子 = 上一窗 residual 全行 dump（bf16 位级）
//   末窗  ：layers [S,42) + final_norm + lm_head → final_hidden / logits / token
//
// 种子链的数学依据：上一窗输出的 residual（全部 T 行、bf16 原位）恰是下一
// 窗的精确输入 → 窗口链 == 全量 forward 的逐层等价分解。注意 golden 只提供
// 每层的「末行」张量（2048 floats），任务书原方案 B 用 golden layer{S-1}.
// mlp_out 单行种子种不出 0..T-2 位置的 K/V（末位 attention 会混入全部前序
// 位置的 K/V），数学上不可精确复现多 token prefill；本链式种子保留了全部
// T 行残差，是严格更强的替代（还把覆盖面从 2 窗扩展到全部 42 层 + 头部）。
//
// 验收阈值与「混沌放大例外」策略：与 real_inference 一致，公共定义见
// tests/golden_compare.h（层/hidden cos≥0.995、logits cos≥0.99、token 精确
// 一致；cos∈[0.99,0.995) 例外需预算/链恢复/端到端全部满足）。输入分叉豁免
// 同步：末窗 step0 token 若失配且 golden step0.logits 前两名 gap ≤
// max(4×bf16ulp(|top1|), 0.25)、该步 logits cos≥0.99 → 合法翻转（e2e 的
// token 条件豁免，计入全局合并预算）；gap 超限 → FAIL。本测试仅 prefill，
// 无后续步可豁免，豁免仅作用于端到端 token 条件。
//
// 显存纪律：本进程总占用 ≤ 2.9GB（给室友任务留 headroom）。启动自检 free
// VRAM 决定窗口大小 N（预算：free − ctx/workspace/KV·scratch/embedding 后
// ÷ 95MB/层，且受 2.9GB 纪律约束）；N<4 → SKIP（exit 0 并打印原因）。
// 运行期采样 cudaMemGetInfo + nvidia-smi 进程占用，峰值超 2.9GB 判 FAIL。
#include "minicpm_runtime.h"

// additive debug 钩子（libminicpm_native 导出；不属于冻结 ABI）。
extern "C" mc_status_t mc_debug_seed_residual(mc_session_t session,
                                              const void* host_bf16_rows, uint32_t rows);

#include <cuda_runtime.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
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

using golden_cmp::below_hard;
using golden_cmp::compare_f32;
using golden_cmp::file_exists;
using golden_cmp::kCosLayerHard;
using golden_cmp::kCosLayerTh;
using golden_cmp::kCosLogitsTh;
using golden_cmp::read_i32_file;

// ---- 预算常量（MiB；per 任务书：每层 94.4→95、emb/lm_head 534.5→535、
//      CUDA ctx ~500、workspace 16、KV@512 22 + act scratch@512 32 → 60）----
constexpr double kDisciplineMiB  = 2900.0; // 进程总占用上限
constexpr double kLayerMiB       = 95.0;
constexpr double kEmbMiB         = 535.0;
constexpr double kLmHeadMiB      = 535.0;
constexpr double kWsMiB          = 16.0;
constexpr double kKvScratchMiB   = 60.0;
constexpr double kCtxFallbackMiB = 500.0;
constexpr uint32_t kMaxCtx       = 512;     // KV/scratch 立减
constexpr uint32_t kNumLayers    = 42;
int g_failures = 0;

std::vector<uint16_t> read_u16_file(const std::string& path) {
    FILE* f = fopen(path.c_str(), "rb");
    if (f == nullptr) return {};
    fseek(f, 0, SEEK_END);
    const long n = ftell(f) / 2;
    fseek(f, 0, SEEK_SET);
    std::vector<uint16_t> v((size_t)n);
    if (n > 0 && fread(v.data(), 2, (size_t)n, f) != (size_t)n) {
        fclose(f);
        return {};
    }
    fclose(f);
    return v;
}

// ---- VRAM 采样（cudaMemGetInfo + nvidia-smi 进程占用）----
double g_min_free_mib = 1e9;   // 全程最小 free（估计口径，受室友波动影响）
long g_pid_peak_mib = -1;      // nvidia-smi 的本进程占用峰值（权威口径）

double free_vram_mib() {
    size_t f = 0, t = 0;
    if (cudaMemGetInfo(&f, &t) != cudaSuccess) return -1.0;
    const double mib = (double)f / (1024.0 * 1024.0);
    g_min_free_mib = std::min(g_min_free_mib, mib);
    return mib;
}

long query_pid_gpu_mib() {
    FILE* p = popen(
        "nvidia-smi --query-compute-apps=pid,used_memory "
        "--format=csv,noheader,nounits 2>/dev/null",
        "r");
    if (p == nullptr) return -1;
    char line[256];
    long mine = -1;
    const long self_pid = (long)getpid();
    while (fgets(line, sizeof(line), p) != nullptr) {
        long pid = 0, mem = 0;
        if (sscanf(line, "%ld, %ld", &pid, &mem) == 2 && pid == self_pid) {
            mine = mem;
            break;
        }
    }
    pclose(p);
    return mine;
}

void sample_vram(const char* label) {
    const double f = free_vram_mib();
    const long pid = query_pid_gpu_mib();
    if (pid > g_pid_peak_mib) g_pid_peak_mib = pid;
    printf("  [vram] %-28s free=%.0f MiB%s\n", label, f,
           pid >= 0 ? "" : " (nvidia-smi n/a)");
}

// ---- 窗口定义与 env ----
struct Window {
    uint32_t lo, hi; // 层区间 [lo,hi)
    bool emb;        // 加载 embedding（首窗）
    bool head;       // 加载 final_norm+lm_head（末窗；final_norm 恒加载）
};

void set_window_env(const Window& w) {
    char buf[32];
    snprintf(buf, sizeof(buf), "%u", w.lo);
    setenv("MC_LOAD_LAYERS_MIN", buf, 1);
    snprintf(buf, sizeof(buf), "%u", w.hi);
    setenv("MC_LOAD_LAYERS_MAX", buf, 1);
    if (w.emb) unsetenv("MC_LOAD_SKIP_EMBEDDING");
    else setenv("MC_LOAD_SKIP_EMBEDDING", "1", 1);
    if (w.head) unsetenv("MC_LOAD_SKIP_LM_HEAD");
    else setenv("MC_LOAD_SKIP_LM_HEAD", "1", 1);
}

mc_model_options_t layerwise_options() {
    mc_model_options_t o{};
    o.struct_size = sizeof(o);
    o.abi_version = MC_ABI_VERSION;
    o.device_id = 0;
    o.max_sessions = 1;
    o.max_context_tokens = kMaxCtx;
    o.precision = MC_BF16;
    o.kv_precision = MC_KV_BF16;
    o.enable_cuda_graph = 0;
    o.reserve_vram_bytes = 0;
    return o;
}

// 每 case 的逐层对拍记录（42 层 × 3 张量 + 尾部）。
struct CaseReport {
    std::array<golden_cmp::TensorVerdict, 3> layer[kNumLayers] = {};
    bool layer_checked[kNumLayers] = {};
    golden_cmp::TensorVerdict final_hidden, logits;
    int32_t tok_golden = -1, tok_got = -2;
    bool tok_checked = false;
    golden_cmp::DivergenceInfo div; // step0 token 失配时的近平局判定（可能未评）
};

int skip_exit(const char* reason) {
    printf("gpu_layerwise: SKIP — %s\n", reason);
    fprintf(stderr, "gpu_layerwise: SKIP — %s\n", reason);
    return 0;
}

} // namespace

int main() {
    // ---- 0) 前置：CUDA 设备与 SM120（model_load 硬要求）；缺失 = 环境不具备 → SKIP ----
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0)
        return skip_exit("no CUDA device visible");
    cudaDeviceProp props{};
    if (cudaGetDeviceProperties(&props, 0) != cudaSuccess || props.major != 12 ||
        props.minor != 0)
        return skip_exit("device 0 is not SM120 (requires RTX 5080-class)");

    // ---- 1) 模型目录（real_inference 同序；缺失 = SKIP，权重存在性 FAIL 信号
    //         由 real_inference 承担）----
    std::string model_dir;
    if (const char* env = getenv("MC_MODEL_DIR")) {
        model_dir = env;
    } else if (file_exists(std::string(MC_REPO_ROOT) + "/models/minicpm5-2b/model.wpk")) {
        model_dir = std::string(MC_REPO_ROOT) + "/models/minicpm5-2b";
    } else if (file_exists(std::string(MC_BUILD_DIR) + "/wpk_model/model.wpk")) {
        model_dir = std::string(MC_BUILD_DIR) + "/wpk_model";
    }
    if (model_dir.empty() || !file_exists(model_dir + "/model.wpk"))
        return skip_exit("model.wpk not found (MC_MODEL_DIR / repo models/ / build "
                         "fixture)");

    // ---- 2) golden 目录（3 case：case_zh/case_en/case_mixed）----
    std::string golden_dir;
    if (const char* env = getenv("MC_GOLDEN_DIR")) {
        golden_dir = env;
    } else {
        const std::string repo_golden =
            std::string(MC_REPO_ROOT) + "/tests/runtime/golden";
        if (file_exists(repo_golden + "/case_zh/input_ids.bin")) golden_dir = repo_golden;
    }
    if (golden_dir.empty())
        return skip_exit("golden dir not found (tests/runtime/golden or MC_GOLDEN_DIR)");

    std::vector<std::string> cases;
    for (const char* c : {"case_zh", "case_en", "case_mixed"})
        if (file_exists(golden_dir + "/" + c + "/input_ids.bin")) cases.push_back(c);
    if (cases.size() != 3)
        return skip_exit("golden incomplete: expected case_zh/case_en/case_mixed");

    // ---- 3) dump 目录（静态缓存一次成型：所有 env 必须在任何 forward 前设好）----
    const std::string dump_dir = std::string(MC_BUILD_DIR) + "/gpu_layerwise_dump";
    const std::string seeds_dir = dump_dir + "/seeds";
    std::error_code ec;
    fs::remove_all(dump_dir, ec);
    fs::create_directories(seeds_dir, ec);
    setenv("MC_DEBUG_DUMP_DIR", dump_dir.c_str(), 1);
    setenv("MC_DEBUG_DUMP_STEPS", "1", 1);          // 只需 step0（prefill 层张量）
    setenv("MC_DEBUG_DUMP_WINDOW_SEED", "1", 1);    // 截断窗 residual 全行种子

    // ---- 4) ctx 建立 + free 自检 → 窗口大小 N ----
    REQUIRE(cudaSetDevice(0) == cudaSuccess);
    REQUIRE(cudaFree(0) == cudaSuccess); // 强制建立 context
    sample_vram("after ctx init");
    const double free0 = free_vram_mib();
    const long ctx_mib = query_pid_gpu_mib(); // 权威 ctx 占用（可能 -1）
    const double ctx_budget = ctx_mib > 0 ? (double)ctx_mib + 30.0 : kCtxFallbackMiB;

    // free 侧（任务书公式；free0 已含 ctx，故只扣 ws/kv/emb）与纪律侧（进程
    // 总占用 ≤2900：扣 ctx+ws+kv+emb/lm_head）双约束。
    int n_free = (int)((free0 - (kWsMiB + kKvScratchMiB + kEmbMiB)) / kLayerMiB);
    int n_disc = (int)((kDisciplineMiB - (ctx_budget + kWsMiB + kKvScratchMiB +
                                           kEmbMiB)) /
                       kLayerMiB);
    int N = std::clamp(std::min(n_free, n_disc), 0, (int)kNumLayers);
    printf("gpu_layerwise: free=%.0f MiB, ctx(pid-smi)=%ld MiB → N_free=%d N_disc=%d "
           "→ N=%d layers/window\n",
           free0, ctx_mib, n_free, n_disc, N);
    if (N < 4)
        return skip_exit("free VRAM too low for a meaningful window (need N>=4, "
                         "got budget for fewer)");

    // ---- 5) 窗口 0 试载（OOM → 收缩 N 重试；防室友波动）----
    mc_model_t cur{nullptr};
    mc_model_options_t opts = layerwise_options();
    while (true) {
        set_window_env(Window{0, (uint32_t)N, /*emb=*/true, /*head=*/false});
        mc_status_t st = mc_model_load(model_dir.c_str(), &opts, &cur);
        if (st == MC_OK) break;
        if (st == MC_E_OUT_OF_MEMORY && N > 4) {
            N -= 4;
            printf("gpu_layerwise: model_load OOM, shrinking N to %d and retrying\n",
                   N);
            continue;
        }
        fprintf(stderr, "gpu_layerwise: FAIL — window-0 model_load: %s (N=%d)\n",
                mc_last_error(), N);
        return 1;
    }

    // ---- 6) 窗口链布局（[0,N) [N,2N) ... [S,42)+head）----
    std::vector<Window> windows;
    for (uint32_t lo = 0; lo < kNumLayers; lo += (uint32_t)N)
        windows.push_back(Window{lo, std::min(lo + (uint32_t)N, kNumLayers),
                                 /*emb=*/lo == 0, /*head=*/lo + (uint32_t)N >= kNumLayers});
    // 注：N 不整除 42 时末窗更小；整除时末窗恰 N 层（预算同首窗，ok）。
    printf("gpu_layerwise: window chain (%zu windows, N=%d):", windows.size(), N);
    for (const auto& w : windows)
        printf(" [%u,%u)%s%s", w.lo, w.hi, w.emb ? "+emb" : "",
               w.head ? "+lm_head" : "");
    printf("\n");

    std::vector<CaseReport> reports(cases.size());

    // ---- 7) 逐窗逐 case：load（窗 0 已载）→ seed → prefill → 对拍 ----
    for (size_t wi = 0; wi < windows.size(); ++wi) {
        const Window& w = windows[wi];
        if (wi > 0) { // 窗 0 已在试载阶段就绪
            mc_model_destroy(cur);
            cur = mc_model_t{nullptr};
            set_window_env(w);
            mc_status_t st = mc_model_load(model_dir.c_str(), &opts, &cur);
            if (st != MC_OK) {
                fprintf(stderr,
                        "gpu_layerwise: FAIL — window %zu [%u,%u) model_load: %s\n", wi,
                        w.lo, w.hi, mc_last_error());
                return 1;
            }
        }
        sample_vram(("window " + std::to_string(wi) + " loaded").c_str());
        printf("== window %zu: layers [%u,%u)%s%s ==\n", wi, w.lo, w.hi,
               w.emb ? " +embedding" : "", w.head ? " +lm_head/final_norm" : "");

        for (size_t ci = 0; ci < cases.size(); ++ci) {
            const std::string& cname = cases[ci];
            const std::string cdir = golden_dir + "/" + cname;
            const auto ids = read_i32_file(cdir + "/input_ids.bin");
            REQUIRE(!ids.empty() && ids.size() <= kMaxCtx);
            const uint32_t T = (uint32_t)ids.size();

            // 清理本窗将产出的 dump 文件（防上一 case 残留误判）
            for (uint32_t i = w.lo; i < w.hi; ++i)
                for (int k = 0; k < 3; ++k) {
                    char fn[64];
                    snprintf(fn, sizeof(fn), "layer%u.%s.f32", i, golden_cmp::kTensorNames[k]);
                    fs::remove(dump_dir + "/" + fn, ec);
                }
            fs::remove(dump_dir + "/window_seed.bf16", ec);
            if (w.head) {
                fs::remove(dump_dir + "/step0.final_hidden.f32", ec);
                fs::remove(dump_dir + "/step0.logits.f32", ec);
                fs::remove(dump_dir + "/step0.token", ec);
            }

            mc_session_t s{nullptr};
            REQUIRE(mc_session_create(cur, &s) == MC_OK);

            // 种子（窗 >0）：上一窗 residual 全行（bf16 位级）
            if (wi > 0) {
                char sfn[96];
                snprintf(sfn, sizeof(sfn), "seed_%zu_%s.bf16", wi - 1, cname.c_str());
                const auto seed = read_u16_file(seeds_dir + "/" + sfn);
                REQUIRE(seed.size() == (size_t)T * 2048);
                REQUIRE(mc_debug_seed_residual(s, seed.data(), T) == MC_OK);
            }

            REQUIRE(mc_prefill(s, ids.data(), T) == MC_OK);
            sample_vram(("window " + std::to_string(wi) + " " + cname + " prefill")
                            .c_str());

            // 保存本窗种子（末窗无下一窗，不产种子——loader 无 lm_head 才产）
            if (!w.head) {
                char sfn[96];
                snprintf(sfn, sizeof(sfn), "seed_%zu_%s.bf16", wi, cname.c_str());
                fs::rename(dump_dir + "/window_seed.bf16", seeds_dir + "/" + sfn, ec);
                REQUIRE(!ec);
            }

            // 逐层三张量对拍（本窗层区间）；失败判定延后到汇总（混沌放大
            // 例外策略需要「链恢复 + 端到端」上下文，见文件头注释）。
            CaseReport& rep = reports[ci];
            for (uint32_t i = w.lo; i < w.hi; ++i) {
                for (int k = 0; k < 3; ++k) {
                    char fn[64];
                    snprintf(fn, sizeof(fn), "layer%u.%s.f32", i, golden_cmp::kTensorNames[k]);
                    rep.layer[i][k] =
                        compare_f32(cdir + "/" + fn, dump_dir + "/" + fn, kCosLayerTh);
                    if (rep.layer[i][k].cos < kCosLayerHard)
                        printf("  HARD-FAIL %s layer%u.%s cos=%.6f maxabs=%.4g\n",
                               cname.c_str(), i, golden_cmp::kTensorNames[k], rep.layer[i][k].cos,
                               rep.layer[i][k].maxabs);
                    fs::remove(dump_dir + "/" + fn, ec);
                }
                rep.layer_checked[i] = true;
            }

            // 末窗：final_hidden / logits / greedy token 对拍（lm_head 链路）
            if (w.head) {
                rep.final_hidden = compare_f32(cdir + "/step0.final_hidden.f32",
                                               dump_dir + "/step0.final_hidden.f32",
                                               kCosLayerTh);
                rep.logits = compare_f32(cdir + "/step0.logits.f32",
                                         dump_dir + "/step0.logits.f32",
                                         kCosLogitsTh);
                const auto gtok = read_i32_file(cdir + "/step0.token");
                const auto dtok = read_i32_file(dump_dir + "/step0.token");
                rep.tok_checked = !gtok.empty() && !dtok.empty();
                if (rep.tok_checked) {
                    rep.tok_golden = gtok[0];
                    rep.tok_got = dtok[0];
                    // token 失配 → 近平局合法翻转判定（须在删 dump 前读 logits）
                    if (rep.tok_golden != rep.tok_got)
                        rep.div = golden_cmp::evaluate_divergence(
                            cdir, dump_dir, 0, rep.tok_golden, rep.tok_got);
                }
                fs::remove(dump_dir + "/step0.final_hidden.f32", ec);
                fs::remove(dump_dir + "/step0.logits.f32", ec);
                fs::remove(dump_dir + "/step0.token", ec);
            }
            mc_session_destroy(s);
        }
    }
    mc_model_destroy(cur);
    sample_vram("after teardown");

    // ---- 8) 汇总表 + 逐 case 判定（策略公共定义见 tests/golden_compare.h）----
    printf("\n==== gpu_layerwise per-layer cosine (primary threshold %.3f, hard "
           "floor %.3f) ====\n",
           golden_cmp::kCosLayerTh, golden_cmp::kCosLayerHard);
    int total_outliers = 0;
    for (size_t ci = 0; ci < cases.size(); ++ci) {
        const CaseReport& rep = reports[ci];
        printf("-- %s --\n", cases[ci].c_str());
        printf("  layer   hidden_in   attn_out    mlp_out\n");
        int checked = 0, hard_fail = 0, outliers = 0;
        std::vector<golden_cmp::TensorVerdict> chain; // 层张量 → final_hidden
        for (uint32_t i = 0; i < kNumLayers; ++i) {
            if (!rep.layer_checked[i]) continue;
            checked++;
            const bool any_below = !rep.layer[i][0].ok || !rep.layer[i][1].ok ||
                                   !rep.layer[i][2].ok;
            for (int k = 0; k < 3; ++k) {
                chain.push_back(rep.layer[i][k]);
                if (below_hard(rep.layer[i][k].cos)) hard_fail++;
                else if (!rep.layer[i][k].ok) outliers++;
            }
            printf("  %-6u  %9.6f  %9.6f  %9.6f%s\n", i, rep.layer[i][0].cos,
                   rep.layer[i][1].cos, rep.layer[i][2].cos,
                   golden_cmp::row_mark(
                       below_hard(rep.layer[i][0].cos) ||
                           below_hard(rep.layer[i][1].cos) ||
                           below_hard(rep.layer[i][2].cos),
                       any_below));
        }
        // final_hidden 并入策略链尾部（与 real_inference 口径一致）
        chain.push_back(rep.final_hidden);
        if (rep.final_hidden.present) {
            if (below_hard(rep.final_hidden.cos)) hard_fail++;
            else if (!rep.final_hidden.ok) outliers++;
        } else {
            hard_fail++; // 末窗未产出 final_hidden = 硬失败
        }
        const bool recovery_ok = golden_cmp::chain_recovery_ok(chain);
        // 输入分叉豁免（与 real_inference 同口径）：token 失配且近平局合法翻转
        // → e2e 的 token 条件豁免；否则 tok_ok=false → FAIL。
        const bool flip_legit = rep.div.evaluated && rep.div.legitimate;
        const int divergences = flip_legit ? 1 : 0;
        total_outliers += outliers + divergences; // 全局预算合并（outlier+divergence）
        // min/avg/maxabs（对 3 张量统一汇总）
        double min_cos = 2.0, sum = 0.0, max_abs = 0.0;
        int n_t = 0;
        for (uint32_t i = 0; i < kNumLayers; ++i) {
            if (!rep.layer_checked[i]) continue;
            for (int k = 0; k < 3; ++k) {
                min_cos = std::min(min_cos, rep.layer[i][k].cos);
                sum += rep.layer[i][k].cos;
                max_abs = std::max(max_abs, rep.layer[i][k].maxabs);
                n_t++;
            }
        }
        const bool tok_ok =
            rep.tok_checked && rep.tok_golden == rep.tok_got && rep.tok_golden >= 0;
        const bool e2e_ok = rep.logits.ok && (tok_ok || flip_legit);
        const bool case_ok = hard_fail == 0 &&
                             outliers <= golden_cmp::kOutlierMaxPerCase &&
                             divergences <= golden_cmp::kDivergenceMaxPerCase &&
                             recovery_ok && e2e_ok;
        if (!case_ok) g_failures++;
        printf("  summary: layers=%d tensors=%d min_cos=%.6f avg_cos=%.6f "
               "maxabs=%.4g hard_fail=%d outliers=%d divergence=%d recovery=%s\n",
               checked, n_t, min_cos, sum / std::max(1, n_t), max_abs, hard_fail,
               outliers, divergences, recovery_ok ? "OK" : "BROKEN");
        if (rep.tok_checked)
            printf("  tail: final_hidden cos=%.6f logits cos=%.6f token "
                   "golden=%d got=%d %s\n",
                   rep.final_hidden.cos, rep.logits.cos, rep.tok_golden, rep.tok_got,
                   tok_ok ? "OK"
                          : (flip_legit ? "FLIP (legit, exempt)" : "MISMATCH"));
        if (rep.div.evaluated)
            printf("  divergence @step0: golden=%d got=%d — top1 id=%d (%.4f) "
                   "top2 id=%d (%.4f), gap=%.4f vs th %.4f, logits cos=%.6f —> %s\n",
                   rep.div.golden_tok, rep.div.got_tok, rep.div.top1_id,
                   rep.div.top1, rep.div.top2_id, rep.div.top2, rep.div.gap,
                   rep.div.threshold, rep.div.logits_cos,
                   flip_legit ? "LEGITIMATE FLIP (e2e token exempt)"
                              : "NOT excusable -> FAIL");
        printf("  case verdict: %s\n", case_ok ? "PASS" : "FAIL");
    }
    if (total_outliers > golden_cmp::kOutlierMaxGlobal)
        g_failures++; // 全局例外预算（outlier + divergence 合并；见公共头策略）

    // 首层通过率（窗口 0 的 layer0，直接从 embedding 起步的层）
    {
        int pass0 = 0;
        for (size_t ci = 0; ci < cases.size(); ++ci)
            if (reports[ci].layer_checked[0] && reports[ci].layer[0][0].ok &&
                reports[ci].layer[0][1].ok && reports[ci].layer[0][2].ok)
                pass0++;
        printf("first-layer (layer0) pass rate: %d/%zu cases\n", pass0, cases.size());
    }

    // ---- 9) 显存纪律峰值报告（≤ 2.9GB）----
    printf("\n==== VRAM discipline (cap %.0f MiB) ===\n", kDisciplineMiB);
    const double est_peak = (double)g_pid_peak_mib;
    if (g_pid_peak_mib >= 0) {
        printf("process GPU peak (nvidia-smi pid sampling): %ld MiB — %s\n",
               g_pid_peak_mib, est_peak <= kDisciplineMiB ? "OK" : "EXCEEDED");
        if (est_peak > kDisciplineMiB) g_failures++;
    } else {
        printf("process GPU peak: nvidia-smi unavailable; informational free-delta "
               "only\n");
    }
    printf("min observed free VRAM during test: %.0f MiB (roommate usage may "
           "shift this)\n",
           g_min_free_mib);

    if (g_failures > 0) {
        fprintf(stderr, "gpu_layerwise: %d failure(s)\n", g_failures);
        return 1;
    }
    fprintf(stderr, "gpu_layerwise: all windows PASSED\n");
    return 0;
}
