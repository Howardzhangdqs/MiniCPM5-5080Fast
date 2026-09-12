// make_wpk.cpp — 测试自举工具：safetensors → model.wpk v1。
// 写域：仅 native/tests（工具链 lane 的 tools/pack_weights.py 就绪后由其替代；
// 二进制布局按任务书字段规范，132B packed TOC 条目，见 weight_loader.cpp 注释）。
//
// 用法：make_wpk <out_dir> [safetensors_path]
// 产物：<out_dir>/model.wpk（out_dir 自动创建）
#include "safetensors_min.hpp"

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include "runtime/generated_model_config.h"
#include "runtime/weight_loader.h" // fnv1a64

namespace fs = std::filesystem;
using stmin::SafeTensors;

namespace {

#pragma pack(push, 1)
struct WpkHeaderOut {
    uint32_t magic;
    uint16_t abi_version;
    uint16_t target_sm;
    uint8_t model_hash[32];
    uint32_t section_count;
    uint32_t reserved;
    uint64_t toc_offset;
    uint64_t data_start;
};
struct WpkSectionOut {
    uint64_t name_hash;
    uint32_t dtype;
    uint32_t layout;
    uint32_t rank;
    uint32_t dims[8];
    uint64_t data_offset;
    uint64_t data_bytes;
    uint64_t scale_offset;
    uint64_t scale_bytes;
    uint64_t reserved[6];
};
#pragma pack(pop)
static_assert(sizeof(WpkSectionOut) == 132, "field-list packed size");

// 输出段：源指针（文件内 bf16）+ 名字 + dims
struct OutSection {
    std::string name;
    const uint16_t* src;
    uint64_t elems;
    uint32_t rank;
    uint32_t dims[8];
};

bool read_sha256_file(const std::string& path, uint8_t out[32]) {
    FILE* f = fopen(path.c_str(), "rb");
    if (f == nullptr) return false;
    char hex[128] = {0};
    size_t n = fread(hex, 1, 64, f);
    fclose(f);
    if (n < 64) return false;
    for (int i = 0; i < 32; ++i) {
        unsigned v = 0;
        if (sscanf(hex + i * 2, "%2x", &v) != 1) return false;
        out[i] = (uint8_t)v;
    }
    return true;
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: make_wpk <out_dir> [safetensors]\n");
        return 1;
    }
    const std::string out_dir = argv[1];
    const std::string st_path = argc >= 3
                                    ? argv[2]
                                    : (fs::path(MC_REPO_ROOT) / "models/minicpm5-2b" /
                                       "model-00000-of-00001.safetensors")
                                          .string();

    SafeTensors st;
    if (!st.open(st_path.c_str())) {
        fprintf(stderr,
                "make_wpk: SKIP — safetensors not available at %s "
                "(parallel lane will provide model.wpk)\n",
                st_path.c_str());
        return 0; // SKIP + PASS
    }

    using namespace mc::cfg;
    auto W = [&](const char* suffix, uint32_t layer) -> const uint16_t* {
        char n[128];
        snprintf(n, sizeof(n), "model.layers.%u.%s", layer, suffix);
        return st.bf16(n);
    };

    // ---- 组装输出段（顺序固定）----
    std::vector<OutSection> secs;
    auto add1 = [&](const char* name, const uint16_t* p, uint32_t d0) {
        secs.push_back({name, p, d0, 1, {d0, 0, 0, 0, 0, 0, 0, 0}});
    };
    auto add2 = [&](const char* name, const uint16_t* p, uint32_t d0, uint32_t d1) {
        secs.push_back({name, p, (uint64_t)d0 * d1, 2, {d0, d1, 0, 0, 0, 0, 0, 0}});
    };
    // qkv / gate_up 行拼需要临时拼接（q|k|v 与 gate|up 在文件里是分开张量）
    std::vector<std::vector<uint16_t>> stitched; // 持有拼接结果

    add2("embedding", st.bf16("model.embed_tokens.weight"), kVocabSize, kHiddenSize);
    add2("lm_head", st.bf16("lm_head.weight"), kVocabSize, kHiddenSize);
    add1("final_norm", st.bf16("model.norm.weight"), kHiddenSize);

    for (uint32_t i = 0; i < kNumLayers; ++i) {
        char n[64];
        snprintf(n, sizeof(n), "layer%u.norm1", i);
        add1(n, W("input_layernorm.weight", i), kHiddenSize);

        // qkv = [q 2048 | k 512 | v 512 行, 2048]，逐行拼接
        {
            const uint16_t* q = W("self_attn.q_proj.weight", i);
            const uint16_t* k = W("self_attn.k_proj.weight", i);
            const uint16_t* v = W("self_attn.v_proj.weight", i);
            if (q == nullptr || k == nullptr || v == nullptr) {
                fprintf(stderr, "make_wpk: missing qkv tensors at layer %u\n", i);
                return 1;
            }
            stitched.emplace_back();
            auto& buf = stitched.back();
            buf.resize((size_t)(kNumQueryHeads * kHeadDim + 2 * kNumKvHeads * kHeadDim) *
                       kHiddenSize);
            const size_t q_rows = kNumQueryHeads * kHeadDim;
            const size_t kv_rows = kNumKvHeads * kHeadDim;
            memcpy(buf.data(), q, q_rows * kHiddenSize * 2);
            memcpy(buf.data() + q_rows * kHiddenSize, k, kv_rows * kHiddenSize * 2);
            memcpy(buf.data() + (q_rows + kv_rows) * kHiddenSize, v,
                   kv_rows * kHiddenSize * 2);
            snprintf(n, sizeof(n), "layer%u.qkv", i);
            add2(n, buf.data(), (uint32_t)(q_rows + 2 * kv_rows), kHiddenSize);
        }
        snprintf(n, sizeof(n), "layer%u.o", i);
        add2(n, W("self_attn.o_proj.weight", i), kHiddenSize, kHiddenSize);
        snprintf(n, sizeof(n), "layer%u.norm2", i);
        add1(n, W("post_attention_layernorm.weight", i), kHiddenSize);
        // gate_up = [gate 6144 | up 6144 行, 2048]
        {
            const uint16_t* g = W("mlp.gate_proj.weight", i);
            const uint16_t* u = W("mlp.up_proj.weight", i);
            if (g == nullptr || u == nullptr) {
                fprintf(stderr, "make_wpk: missing gate/up tensors at layer %u\n", i);
                return 1;
            }
            stitched.emplace_back();
            auto& buf = stitched.back();
            buf.resize((size_t)2 * kIntermediateSize * kHiddenSize);
            memcpy(buf.data(), g, (size_t)kIntermediateSize * kHiddenSize * 2);
            memcpy(buf.data() + (size_t)kIntermediateSize * kHiddenSize, u,
                   (size_t)kIntermediateSize * kHiddenSize * 2);
            snprintf(n, sizeof(n), "layer%u.gate_up", i);
            add2(n, buf.data(), 2 * kIntermediateSize, kHiddenSize);
        }
        snprintf(n, sizeof(n), "layer%u.down", i);
        add2(n, W("mlp.down_proj.weight", i), kHiddenSize, kIntermediateSize);
    }

    for (const auto& s : secs) {
        if (s.src == nullptr) {
            fprintf(stderr, "make_wpk: missing source tensor for '%s'\n",
                    s.name.c_str());
            return 1;
        }
    }

    fs::create_directories(out_dir);
    const std::string out_path = (fs::path(out_dir) / "model.wpk").string();
    FILE* f = fopen(out_path.c_str(), "wb");
    if (f == nullptr) {
        fprintf(stderr, "make_wpk: cannot create %s\n", out_path.c_str());
        return 1;
    }

    // ---- header + TOC + data ----
    const uint32_t count = (uint32_t)secs.size();
    WpkHeaderOut hdr{};
    hdr.magic = 0x31504B4Du;
    hdr.abi_version = 0x0002;
    hdr.target_sm = 120;
    const std::string sha_path =
        (fs::path(MC_REPO_ROOT) / "models/minicpm5-2b/weights.sha256").string();
    if (!read_sha256_file(sha_path, hdr.model_hash)) {
        fprintf(stderr, "make_wpk: note — weights.sha256 not readable, hash=0\n");
    }
    hdr.section_count = count;
    hdr.reserved = 0;
    hdr.toc_offset = 64;
    // data_start 上取整到 256B（与规范/pack_weights.py 的 WPK_ALIGN 一致；
    // 旧版 64+count*132 未对齐，255 sections 时得 33724 ≠ 规范 33792）
    hdr.data_start = (64 + (uint64_t)count * sizeof(WpkSectionOut) + 255ull) & ~255ull;
    fwrite(&hdr, sizeof(hdr), 1, f);

    uint64_t off = hdr.data_start;
    std::vector<WpkSectionOut> toc(count);
    for (uint32_t i = 0; i < count; ++i) {
        off = (off + 255) & ~255ull; // 256B 对齐
        WpkSectionOut& t = toc[i];
        memset(&t, 0, sizeof(t));
        t.name_hash = fnv1a64(secs[i].name.c_str(), secs[i].name.size());
        t.dtype = 0;
        t.layout = 0;
        t.rank = secs[i].rank;
        memcpy(t.dims, secs[i].dims, sizeof(t.dims));
        t.data_offset = off;
        t.data_bytes = secs[i].elems * 2;
        off += t.data_bytes;
    }
    fwrite(toc.data(), sizeof(WpkSectionOut), count, f);

    // 数据段（按 TOC 偏移写，中间 padding 补零）
    uint64_t cur = hdr.data_start + (uint64_t)count * sizeof(WpkSectionOut);
    std::vector<unsigned char> zeros(256, 0);
    for (uint32_t i = 0; i < count; ++i) {
        while (cur < toc[i].data_offset) {
            const size_t n = (size_t)std::min<uint64_t>(toc[i].data_offset - cur, 256);
            fwrite(zeros.data(), 1, n, f);
            cur += n;
        }
        fwrite(secs[i].src, 1, (size_t)toc[i].data_bytes, f);
        cur += toc[i].data_bytes;
    }
    fclose(f);
    fprintf(stderr, "make_wpk: wrote %s (%u sections, %llu bytes data)\n",
            out_path.c_str(), count, (unsigned long long)(off - hdr.data_start));
    return 0;
}
