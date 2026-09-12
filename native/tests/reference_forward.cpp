// reference_forward.cpp — CPU fp32 参考实现（测试自举验证用）。
//
// 语义与 native GPU 路径的任务书数值规范逐条一致：
//   存储 bf16、计算 fp32；残差加 fp32→bf16；RMSNorm fp32 累加、单次终舍入
//   （x·inv·w → bf16）；RoPE rotate_half/theta 5e6/fp32→bf16；KV bf16；attention
//   score fp32/sqrt(128)、softmax fp32、Σp·v fp32→bf16；GEMM fp32 累加→bf16；
//   SwiGLU silu(g)·u fp32→bf16；greedy argmax fp32 比较、tie→小 id。
// 与 HF 参考的差别：attention score 不做中间 bf16 舍入（任务书规定 fp32）。
//
// 用法：reference_forward <out_golden_dir> [safetensors]
// 产物：<out_golden_dir>/case0/（golden 布局：input_ids.bin、step{t}.*、
//       layer{i}.*，全部 flat 于 case 目录）。
// 权重直接 mmap safetensors（q/k/v、gate/up 用原始分离张量，拼接语义等价）。
#include "safetensors_min.hpp"

#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <functional>
#include <string>
#include <thread>
#include <vector>

#include "runtime/generated_model_config.h"

namespace fs = std::filesystem;
using stmin::bf16_to_f32;
using stmin::f32_to_bf16;
using stmin::SafeTensors;

namespace {

constexpr uint32_t H = mc::cfg::kHiddenSize;         // 2048
constexpr uint32_t QH = mc::cfg::kNumQueryHeads;      // 16
constexpr uint32_t KVH = mc::cfg::kNumKvHeads;        // 2
constexpr uint32_t HD = mc::cfg::kHeadDim;            // 128
constexpr uint32_t INTER = mc::cfg::kIntermediateSize; // 6144
constexpr uint32_t L = mc::cfg::kNumLayers;           // 42
constexpr uint32_t VOCAB = mc::cfg::kVocabSize;
constexpr float EPS = mc::cfg::kRmsEps;
constexpr uint32_t kPromptTokens = 16;
constexpr uint32_t kSteps = 8; // step0..step7

using bf16v = std::vector<uint16_t>;

// 朴素并行：每次并行区段新建线程（工具程序，线程创建开销可忽略）
void parallel_for(uint32_t rows, const std::function<void(uint32_t)>& body) {
    const unsigned hw = std::thread::hardware_concurrency();
    const unsigned nt = hw == 0 ? 8u : (hw > 16u ? 16u : hw);
    std::atomic<uint32_t> next{0};
    std::vector<std::thread> ts;
    for (unsigned t = 0; t < nt; ++t) {
        ts.emplace_back([&] {
            while (true) {
                const uint32_t r = next.fetch_add(1);
                if (r >= rows) break;
                body(r);
            }
        });
    }
    for (auto& th : ts) th.join();
}

// y_bf[rows] = x_bf[in] · W[rows,in]ᵀ（fp32 累加 → bf16）
void gemv(const bf16v& x, const uint16_t* W, uint32_t rows, uint32_t in, bf16v& y) {
    y.resize(rows);
    parallel_for(rows, [&](uint32_t r) {
        const uint16_t* wr = W + (size_t)r * in;
        float acc = 0.f;
        for (uint32_t k = 0; k < in; ++k) acc += bf16_to_f32(x[k]) * bf16_to_f32(wr[k]);
        y[r] = f32_to_bf16(acc);
    });
}

// RMSNorm：fp32 累加；y=x*inv 先落 bf16，再乘 w（与 GPU kernel 一致）
void rmsnorm(const bf16v& x, const uint16_t* w, bf16v& out) {
    out.resize(H);
    // fp32 累加（与 GPU block 归约的顺序差异属于 cos 阈值容忍范围）
    float s32 = 0.f;
    for (uint32_t i = 0; i < H; ++i) {
        const float v = bf16_to_f32(x[i]);
        s32 += v * v;
    }
    const float inv = 1.0f / std::sqrt(s32 / (float)H + EPS);
    for (uint32_t i = 0; i < H; ++i) {
        // 单次终舍入：fp32 连乘 x·inv·w → bf16（与 python 参考 / GPU kernel 一致）
        out[i] = f32_to_bf16(bf16_to_f32(x[i]) * inv * bf16_to_f32(w[i]));
    }
}

void write_f32(const std::string& path, const std::vector<float>& v) {
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) { fprintf(stderr, "ref: cannot write %s\n", path.c_str()); exit(1); }
    fwrite(v.data(), 4, v.size(), f);
    fclose(f);
}
void write_i32(const std::string& path, int32_t v) {
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) { fprintf(stderr, "ref: cannot write %s\n", path.c_str()); exit(1); }
    fwrite(&v, 4, 1, f);
    fclose(f);
}
std::vector<float> upcast(const bf16v& b) {
    std::vector<float> f(b.size());
    for (size_t i = 0; i < b.size(); ++i) f[i] = bf16_to_f32(b[i]);
    return f;
}

struct Ref {
    SafeTensors& st;
    // KV：[layer][pos][2(K,V)][KVH][HD]，bf16
    std::vector<bf16v> kcache{};
    std::vector<bf16v> vcache{};
    uint32_t pos = 0;
    std::string outdir{};
    int dump_step = -1;   // ≥0 时 dump step{t}（logits/final_hidden/token）
    bool dump_layers = false;

    const uint16_t* W(const char* fmt, uint32_t layer) const {
        char n[128];
        snprintf(n, sizeof(n), fmt, layer);
        return st.bf16(n);
    }

    // 单 token forward；返回 logits（bf16）
    bf16v step(int32_t tok) {
        const uint32_t p = pos++;
        bf16v residual(H);
        const uint16_t* erow = st.bf16("model.embed_tokens.weight") + (size_t)tok * H;
        memcpy(residual.data(), erow, H * 2);

        bf16v normed(H);
        bf16v ctx(H), o_out(H), gate(INTER), up(INTER), mid(INTER), down_out(H);
        const uint32_t q_rows = QH * HD, kv_rows = KVH * HD;

        for (uint32_t li = 0; li < L; ++li) {
            if (dump_layers) {
                char path[256];
                snprintf(path, sizeof(path), "%s/layer%u.hidden_in.f32", outdir.c_str(), li);
                write_f32(path, upcast(residual));
            }
            // norm1
            rmsnorm(residual, W("model.layers.%u.input_layernorm.weight", li), normed);
            // qkv（分离 gemv，语义 = 拼接行 GEMM）
            bf16v qo, ko, vo;
            gemv(normed, W("model.layers.%u.self_attn.q_proj.weight", li), q_rows, H, qo);
            gemv(normed, W("model.layers.%u.self_attn.k_proj.weight", li), kv_rows, H, ko);
            gemv(normed, W("model.layers.%u.self_attn.v_proj.weight", li), kv_rows, H, vo);
            // rope：q 原地；k 后入 cache
            rope(qo, p);
            rope(ko, p);
            memcpy(&kcache[li][(size_t)p * kv_rows], ko.data(), kv_rows * 2);
            memcpy(&vcache[li][(size_t)p * kv_rows], vo.data(), kv_rows * 2);
            // attention（GQA，causal j≤p）
            for (uint32_t h = 0; h < QH; ++h) {
                const uint32_t kvh = h / (QH / KVH);
                float maxs = -INFINITY;
                float scores[8192];
                for (uint32_t j = 0; j <= p; ++j) {
                    const uint16_t* kj = &kcache[li][(size_t)j * kv_rows + kvh * HD];
                    float dot = 0.f;
                    for (uint32_t d = 0; d < HD; ++d)
                        dot += bf16_to_f32(qo[h * HD + d]) * bf16_to_f32(kj[d]);
                    const float s = dot * 0.08838834764831845f;
                    scores[j] = s;
                    if (s > maxs) maxs = s;
                }
                float denom = 0.f;
                for (uint32_t j = 0; j <= p; ++j) {
                    scores[j] = std::exp(scores[j] - maxs);
                    denom += scores[j];
                }
                float acc[HD];
                for (uint32_t d = 0; d < HD; ++d) acc[d] = 0.f;
                for (uint32_t j = 0; j <= p; ++j) {
                    const uint16_t* vj = &vcache[li][(size_t)j * kv_rows + kvh * HD];
                    const float w = scores[j];
                    for (uint32_t d = 0; d < HD; ++d) acc[d] += w * bf16_to_f32(vj[d]);
                }
                for (uint32_t d = 0; d < HD; ++d)
                    ctx[h * HD + d] = f32_to_bf16(acc[d] / denom);
            }
            // o 投影
            gemv(ctx, W("model.layers.%u.self_attn.o_proj.weight", li), H, H, o_out);
            // 残差 + norm2
            for (uint32_t i = 0; i < H; ++i)
                residual[i] = f32_to_bf16(bf16_to_f32(residual[i]) + bf16_to_f32(o_out[i]));
            if (dump_layers) {
                // attn_out = o 并入残差后的 residual（python 权威语义：
                // x = add_bf16(x, ao) 之后收集）
                char path[256];
                snprintf(path, sizeof(path), "%s/layer%u.attn_out.f32", outdir.c_str(), li);
                write_f32(path, upcast(residual));
            }
            rmsnorm(residual, W("model.layers.%u.post_attention_layernorm.weight", li),
                    normed);
            // gate/up + swiglu
            gemv(normed, W("model.layers.%u.mlp.gate_proj.weight", li), INTER, H, gate);
            gemv(normed, W("model.layers.%u.mlp.up_proj.weight", li), INTER, H, up);
            for (uint32_t i = 0; i < INTER; ++i) {
                const float g = bf16_to_f32(gate[i]);
                const float silu = g / (1.0f + std::exp(-g));
                mid[i] = f32_to_bf16(silu * bf16_to_f32(up[i]));
            }
            gemv(mid, W("model.layers.%u.mlp.down_proj.weight", li), H, INTER, down_out);
            for (uint32_t i = 0; i < H; ++i)
                residual[i] =
                    f32_to_bf16(bf16_to_f32(residual[i]) + bf16_to_f32(down_out[i]));
            if (dump_layers) {
                // mlp_out = down 并入残差后的 residual（python 权威语义：
                // x = add_bf16(x, d) 之后收集）
                char path[256];
                snprintf(path, sizeof(path), "%s/layer%u.mlp_out.f32", outdir.c_str(), li);
                write_f32(path, upcast(residual));
            }
        }

        // final norm（先算，final_hidden = final_norm 后的输出，python hf[-1]）
        bf16v normed_f(H);
        rmsnorm(residual, st.bf16("model.norm.weight"), normed_f);
        if (dump_step >= 0) {
            char path[256];
            snprintf(path, sizeof(path), "%s/step%d.final_hidden.f32", outdir.c_str(),
                     dump_step);
            write_f32(path, upcast(normed_f));
        }
        // lm_head
        bf16v logits(VOCAB);
        gemv(normed_f, st.bf16("lm_head.weight"), VOCAB, H, logits);
        return logits;
    }

    // RoPE：x 为 [n_head, 128]（q: 16 头 / k: 2 头），逐头 rotate_half。
    // inv_freq 用 float64 幂再落 fp32（对齐 python _INV_FREQ 的
    // astype(np.float32)）；angle = fp32(pos) × fp32(inv_freq)。
    // 历史 bug：单头循环只转了前 128 个元素（q 仅 head0、k 丢 head1），
    // 15/16 个 q 头未旋转 → 逐层发散（compare_dump 抓获）。
    static void rope(bf16v& x, uint32_t pos) {
        const size_t heads = x.size() / HD;
        for (size_t h = 0; h < heads; ++h) {
            uint16_t* hd = x.data() + h * HD;
            for (uint32_t d = 0; d < HD / 2; ++d) {
                const float inv_freq =
                    (float)std::pow(5000000.0, -(double)d / 64.0);
                const float freq = (float)pos * inv_freq;
                const float cs = std::cos(freq), sn = std::sin(freq);
                const float x1 = bf16_to_f32(hd[d]);
                const float x2 = bf16_to_f32(hd[d + 64]);
                hd[d] = f32_to_bf16(x1 * cs - x2 * sn);
                hd[d + 64] = f32_to_bf16(x2 * cs + x1 * sn);
            }
        }
    }
};

int argmax_bf16(const bf16v& logits) {
    float best = -INFINITY;
    int32_t bi = 0;
    for (uint32_t i = 0; i < logits.size(); ++i) {
        const float v = bf16_to_f32(logits[i]);
        if (v > best) { // tie 保持更小 id（首见优先）
            best = v;
            bi = (int32_t)i;
        }
    }
    return bi;
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: reference_forward <out_golden_dir> [safetensors]\n");
        return 1;
    }
    const std::string out_root = argv[1];
    const std::string st_path =
        argc >= 3
            ? argv[2]
            : (fs::path(MC_REPO_ROOT) / "models/minicpm5-2b/model-00000-of-00001.safetensors")
                  .string();

    SafeTensors st;
    if (!st.open(st_path.c_str())) {
        fprintf(stderr,
                "reference_forward: SKIP — safetensors not available at %s\n",
                st_path.c_str());
        return 0; // SKIP + PASS
    }

    fs::create_directories(fs::path(out_root) / "case0");
    Ref ref{st};
    ref.outdir = (fs::path(out_root) / "case0").string();
    const uint32_t max_pos = kPromptTokens + kSteps + 8;
    ref.kcache.resize(L, bf16v((size_t)max_pos * KVH * HD, 0));
    ref.vcache.resize(L, bf16v((size_t)max_pos * KVH * HD, 0));

    // 固定输入（与 golden case0 一致）
    std::vector<int32_t> ids(kPromptTokens);
    for (uint32_t i = 0; i < kPromptTokens; ++i)
        ids[i] = (int32_t)((i * 2654435761u + 12345u) % VOCAB);
    {
        FILE* f = fopen((fs::path(ref.outdir) / "input_ids.bin").string().c_str(), "wb");
        fwrite(ids.data(), 4, ids.size(), f);
        fclose(f);
    }

    bf16v logits;
    for (uint32_t t = 0; t < kPromptTokens; ++t) {
        // step0 = prefill 最后一个位置；层张量仅 step0
        const bool last = (t + 1 == kPromptTokens);
        ref.dump_step = last ? 0 : -1;
        ref.dump_layers = last;
        logits = ref.step(ids[t]);
    }
    int32_t tok = argmax_bf16(logits);
    {
        char path[256];
        snprintf(path, sizeof(path), "%s/step%d.logits.f32", ref.outdir.c_str(), 0);
        write_f32(path, upcast(logits));
        snprintf(path, sizeof(path), "%s/step%d.token", ref.outdir.c_str(), 0);
        write_i32(path, tok);
    }
    for (int s = 1; s < (int)kSteps; ++s) {
        ref.dump_step = s;
        ref.dump_layers = false;
        logits = ref.step(tok);
        tok = argmax_bf16(logits);
        char path[256];
        snprintf(path, sizeof(path), "%s/step%d.logits.f32", ref.outdir.c_str(), s);
        write_f32(path, upcast(logits));
        snprintf(path, sizeof(path), "%s/step%d.token", ref.outdir.c_str(), s);
        write_i32(path, tok);
    }
    fprintf(stderr, "reference_forward: wrote %s (case0: %u prompt, %u steps)\n",
            ref.outdir.c_str(), kPromptTokens, kSteps);
    return 0;
}
