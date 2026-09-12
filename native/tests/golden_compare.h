// golden_compare.h — golden 对拍公共工具与「混沌放大例外」判定策略。
//
// real_inference 与 gpu_layerwise 共用（header-only，纯 host，无 CUDA 依赖；
// 两测试仍各自独立编译）。策略在两处的语义完全一致，便于交叉核对。
//
// 阈值与例外策略（P2 gpu_layerwise 定因；详见两测试头部注释）：
//   - 层张量 / final_hidden（hidden 维向量）：cos ≥ kCosLayerTh(0.995) 通过；
//   - logits：cos ≥ kCosLogitsTh(0.99)，硬阈值，无例外；
//   - token：精确一致，任何一步不一致 = FAIL（无预算）；
//   - 混沌放大例外：cos ∈ [kCosLayerHard(0.99), kCosLayerTh) 的 hidden 维
//     张量不直接 FAIL，需同时满足：
//       1) 每 case ≤ kOutlierMaxPerCase 个、全部 case 合计 ≤ kOutlierMaxGlobal；
//       2) 链恢复：同一有序链（step0：layer0..41 三张量 → step0.final_hidden；
//          decode 步的 step{t}.final_hidden 按 t 序为第二条链）中，最后一个
//          outlier 之后的全部张量回到主阈值之上；
//       3) 端到端：该 case logits 达标且 greedy token 无非豁免失配
//          （全一致，或仅含一次下述合法翻转）。
//     依据：跨实现合法舍入差（cuBLASLt vs numpy 累加序、CUDA sin/cos/exp
//     ≤2ulp、rsqrtf、归约顺序）产生 1-bf16-ulp 级差异，经近平局 softmax
//     混沌放大（case_zh layer8 head11 top-2 概率 0.2749/0.2593、score gap
//     ≈0.058 实测；fp64 扰动实验证实 golden 自身换合法舍入序同样发生），
//     属阈值标定问题而非语义错误；硬下限 kCosLayerHard 之下仍为 FAIL。
//   - 文件缺失 / 尺寸不符 / cos < 硬下限 = 硬 FAIL，无例外。
//   - 输入分叉豁免（与混沌放大例外正交叠加，全局预算合并）：
//     某 case 首个 greedy token 值失配发生在 step t（此前全部一致）时，读
//     golden step{t}.logits.f32 检查前两名差值：gap ≤ max(4×|top1| 量级的
//     bf16 ulp, 0.25) 且该步自身 logits cos ≥ kCosLogitsTh → 判「合法翻转」
//     （跨实现合法舍入差使近平局贪心选择翻转，GPU 后续输入与 golden 分叉，
//     张量比较无意义）：step>t 的张量豁免（不计失败、不耗预算），每 case
//     至多 kDivergenceMaxPerCase 次；gap 超限或条件不满足 → FAIL（不放宽）。
//     依据：case_mixed step6 golden top1/top2 = 16.125/16.000（差 0.125 = 1
//     个 bf16 ulp；实测 2026-09-08），GPU 与 numpy 参考的合法舍入差足以翻转。
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace golden_cmp {

// ---- 策略参数 ----
constexpr double kCosLayerTh        = 0.995; // 层张量 / final_hidden 主阈值
constexpr double kCosLayerHard      = 0.99;  // 混沌放大硬下限（< 此值必 FAIL）
constexpr double kCosLogitsTh       = 0.99;  // logits（硬阈值）
constexpr int kOutlierMaxPerCase    = 2;     // 每 case 混沌例外预算
constexpr int kOutlierMaxGlobal     = 3;     // 全部 case 合计预算（混沌例外 + 分叉豁免合并计数）

// 输入分叉豁免（见文件头策略注）
constexpr int    kDivergenceMaxPerCase = 1;   // 每 case 至多一次合法翻转
constexpr double kDivergenceGapFloor   = 0.25; // gap 阈值下限（logit 绝对差）
constexpr int    kDivergenceUlpMult    = 4;   // top1 量级 bf16 ulp 的允许倍数

inline constexpr const char* kTensorNames[3] = {"hidden_in", "attn_out", "mlp_out"};

// ---- 文件读取 / 余弦（double 累加）----
inline bool file_exists(const std::string& p) {
    return std::fopen(p.c_str(), "rb") != nullptr;
}

inline std::vector<int32_t> read_i32_file(const std::string& path) {
    FILE* f = fopen(path.c_str(), "rb");
    if (f == nullptr) return {};
    fseek(f, 0, SEEK_END);
    const long n = ftell(f) / 4;
    fseek(f, 0, SEEK_SET);
    std::vector<int32_t> v((size_t)n);
    if (n > 0 && fread(v.data(), 4, (size_t)n, f) != (size_t)n) {
        fclose(f);
        return {};
    }
    fclose(f);
    return v;
}

inline std::vector<float> read_f32_file(const std::string& path) {
    FILE* f = fopen(path.c_str(), "rb");
    if (f == nullptr) return {};
    fseek(f, 0, SEEK_END);
    const long n = ftell(f) / 4;
    fseek(f, 0, SEEK_SET);
    std::vector<float> v((size_t)n);
    if (n > 0 && fread(v.data(), 4, (size_t)n, f) != (size_t)n) {
        fclose(f);
        return {};
    }
    fclose(f);
    return v;
}

inline double cosine(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size() || a.empty()) return -2.0;
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        dot += (double)a[i] * (double)b[i];
        na += (double)a[i] * (double)a[i];
        nb += (double)b[i] * (double)b[i];
    }
    if (na == 0.0 || nb == 0.0) return -2.0;
    return dot / (std::sqrt(na) * std::sqrt(nb));
}

// ---- 单张量对拍结果 ----
struct TensorVerdict {
    double cos    = -2.0;
    double maxabs = 0.0;
    bool   ok     = false;    // cos ≥ 调用方阈值
    bool   present = false;   // 双侧文件存在且尺寸一致
};

inline TensorVerdict compare_f32(const std::string& golden, const std::string& dump,
                                 double th) {
    TensorVerdict v;
    const auto g = read_f32_file(golden);
    const auto d = read_f32_file(dump);
    if (g.empty() || g.size() != d.size()) return v;
    v.present = true;
    v.cos = cosine(g, d);
    for (size_t i = 0; i < g.size(); ++i)
        v.maxabs = std::max(v.maxabs, std::fabs((double)g[i] - (double)d[i]));
    v.ok = v.cos >= th;
    return v;
}

// ---- golden 文件名分类（flat 布局）----
// 注意：sscanf 在尾部字面量（如 ".f32"）不匹配时仍返回已赋值项数，
// 必须用 %n 校验整串消费（"step3.token" 会被 "step%u.%63[^.].f32" 误吞
// 2 项——policy_check 抓获过此 bug）。
inline bool ends_with(const std::string& s, const char* suf) {
    const size_t n = strlen(suf);
    return s.size() >= n && s.compare(s.size() - n, n, suf) == 0;
}

enum class Kind { LayerTensor, FinalHidden, Logits, Token, OtherF32, Unknown };

struct Classify {
    Kind kind = Kind::Unknown;
    int  layer = -1; // LayerTensor
    int  step  = -1; // FinalHidden / Logits / Token
    int  tensor = -1; // LayerTensor: 0=hidden_in 1=attn_out 2=mlp_out, -1=其它名
};

inline Classify classify(const std::string& fn) {
    Classify c;
    unsigned a = 0, b = 0;
    char mid[64] = {0};
    int pos = 0;
    if (ends_with(fn, ".f32")) {
        if (sscanf(fn.c_str(), "layer%u.%63[^.].f32%n", &a, mid, &pos) == 2 &&
            pos == (int)fn.size()) {
            c.kind = Kind::LayerTensor;
            c.layer = (int)a;
            for (int k = 0; k < 3; ++k)
                if (strcmp(mid, kTensorNames[k]) == 0) c.tensor = k;
            return c; // tensor==-1（层下其它 .f32 名）也按 hidden 维策略（链序按层）
        }
        pos = 0;
        if (sscanf(fn.c_str(), "step%u.%63[^.].f32%n", &b, mid, &pos) == 2 &&
            pos == (int)fn.size()) {
            c.step = (int)b;
            if (strcmp(mid, "logits") == 0) c.kind = Kind::Logits;
            else if (strcmp(mid, "final_hidden") == 0) c.kind = Kind::FinalHidden;
            else c.kind = Kind::OtherF32;
            return c;
        }
        c.kind = Kind::OtherF32; // 其它 .f32：按 hidden 维策略（防御）
        return c;
    }
    pos = 0;
    if (ends_with(fn, ".token") &&
        sscanf(fn.c_str(), "step%u.token%n", &b, &pos) == 1 &&
        pos == (int)fn.size()) {
        c.kind = Kind::Token;
        c.step = (int)b;
        return c;
    }
    return c; // Unknown：调用方按防御处理
}

// NaN 安全的硬下限判定：cos 为 NaN/缺失（-2）等任何非法值都视为硬失败
// （NaN 与任何比较均为 false，直接用 `<` 会漏判——policy_check 抓获过）。
inline bool below_hard(double cos) { return !(cos >= kCosLayerHard); }

// ---- 链恢复：同一有序链内，最后一个 outlier（cos ∈ [hard, th)）之后
//      的全部张量必须 ≥ 主阈值。前提：链内无硬 FAIL（调用方先判）。----
inline bool chain_recovery_ok(const std::vector<TensorVerdict>& chain) {
    int last_outlier = -1;
    for (size_t i = 0; i < chain.size(); ++i)
        if (!chain[i].ok && chain[i].cos >= kCosLayerHard) last_outlier = (int)i;
    for (size_t i = (size_t)last_outlier + 1; i < chain.size(); ++i)
        if (!chain[i].ok) return false;
    return true;
}

// ---- 输入分叉豁免：greedy 近平局合法翻转判定（real_inference 与
//      gpu_layerwise 共用；策略见文件头）----
// bf16（8 位尾数、7 位显式）在量级 x>0 处的 ulp = 2^(floor(log2 x) - 7)。
inline double bf16_ulp(double x) {
    if (!(x > 0.0) || !std::isfinite(x)) return 0.0;
    int e = 0;
    std::frexp(x, &e);        // x = m·2^e, m ∈ [0.5,1) → floor(log2 x) = e-1
    return std::ldexp(1.0, e - 1 - 7);
}

struct DivergenceInfo {
    bool    evaluated  = false; // 首个失配存在且 golden step{t}.logits.f32 可读
    bool    legitimate = false; // 近平局合法翻转（gap 达标 + 该步 logits cos 达标）
    int     step       = -1;
    int32_t golden_tok = -1;
    int32_t got_tok    = -2;
    int32_t top1_id = -1, top2_id = -1; // golden logits 前两名（id 即下标）
    double  top1 = 0.0, top2 = 0.0;
    double  gap = -1.0;        // top1 - top2
    double  threshold = -1.0;  // max(kDivergenceUlpMult × bf16_ulp(|top1|), 0.25)
    double  logits_cos = -2.0; // 该步 golden vs dump logits cos
};

// t 为该 case 首个 token 值失配步（调用方保证此前全部一致、双方 token 文件
// 均在且为单 i32）。gap 与该步 logits cos 双达标 → legitimate。
inline DivergenceInfo evaluate_divergence(const std::string& golden_dir,
                                          const std::string& dump_dir, int step,
                                          int32_t golden_tok, int32_t got_tok) {
    DivergenceInfo d;
    d.step = step;
    d.golden_tok = golden_tok;
    d.got_tok = got_tok;
    char fn[64];
    snprintf(fn, sizeof(fn), "step%d.logits.f32", step);
    const auto g = read_f32_file(golden_dir + "/" + fn);
    const auto u = read_f32_file(dump_dir + "/" + fn);
    if (g.empty()) return d; // golden logits 不可读 → 无法认证（调用方按 FAIL）
    if (g.size() == u.size()) d.logits_cos = cosine(g, u);
    int32_t i1 = -1, i2 = -1;
    double v1 = -1e300, v2 = -1e300;
    for (size_t i = 0; i < g.size(); ++i) {
        const double v = g[i];
        if (v > v1) {
            v2 = v1; i2 = i1; v1 = v; i1 = (int32_t)i;
        } else if (v > v2) {
            v2 = v; i2 = (int32_t)i;
        }
    }
    d.evaluated = true;
    d.top1 = v1;
    d.top2 = v2;
    d.top1_id = i1;
    d.top2_id = i2;
    d.gap = v1 - v2;
    d.threshold = std::max(kDivergenceUlpMult * bf16_ulp(std::fabs(v1)),
                           kDivergenceGapFloor);
    d.legitimate = d.gap <= d.threshold && d.logits_cos >= kCosLogitsTh;
    return d;
}

// ---- 表格标记（gpu_layerwise 报告风格）----
inline const char* row_mark(bool any_hard, bool any_outlier) {
    return any_hard ? "  <-- HARD-FAIL" : (any_outlier ? "  <-- outlier (chaotic, see policy)" : "");
}

} // namespace golden_cmp
