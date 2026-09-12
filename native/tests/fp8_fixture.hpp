// fp8_fixture.hpp — P4/P5 测试共用：model.wpk → model_fp8.wpk 现场量化 fixture。
//
// 从 fp8_inference.cpp 提取（逻辑不变），P5 起额外量化 lm_head（第 169 个
// section，dtype=2）——loader（load_wpk_fp8）检测到该段即让 decode 的
// lm_head 走 gemv_fp8（native 侧闭环验证；正式包由工具链产出同格式段）。
// 量化规范与工具链一致：逐行 scale = max|w|/448（全零行 scale=1），
// q = e4m3_rne(w/scale)（satfinite）。
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace fp8fix {

#pragma pack(push, 1)
struct WpkHeaderIo {
    uint32_t magic;
    uint16_t abi_version;
    uint16_t target_sm;
    uint8_t model_hash[32];
    uint32_t section_count;
    uint32_t reserved;
    uint64_t toc_offset;
    uint64_t data_start;
};
struct WpkSectionIo {
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
static_assert(sizeof(WpkHeaderIo) == 64, "wpk header 64B");
static_assert(sizeof(WpkSectionIo) == 132, "wpk toc entry 132B");

inline uint64_t fnv1a64_io(const std::string& s) {
    uint64_t h = 14695981039346656037ull;
    for (char c : s) {
        h ^= (uint64_t)(uint8_t)c;
        h *= 1099511628211ull;
    }
    return h;
}

inline float bf16_to_f32_io(uint16_t h) {
    const uint32_t x = (uint32_t)h << 16;
    float f;
    std::memcpy(&f, &x, 4);
    return f;
}

// host E4M3 RNE 编码（satfinite：超出范围 → ±448；与工具链 lane 规范一致）
inline uint8_t f32_to_e4m3_rne(float f) {
    const uint8_t sign = std::signbit(f) ? 0x80u : 0x00u;
    const float a = std::fabs(f);
    if (!(a <= 463.9f)) return sign | 0x7Eu; // clamp（含 NaN/Inf）
    if (a < std::ldexp(1.0f, -10)) return sign; // < 半个最小次正规 → 0
    int e = 0;
    std::frexp(a, &e);
    e -= 1; // a = frac × 2^e，frac ∈ [1,2)
    if (e < -6) { // 次正规区：quantum = 2^-9
        const float t = std::ldexp(a, 9);
        const float r = std::floor(t);
        const float frac = t - r;
        int m = (int)r;
        if (frac > 0.5f || (frac == 0.5f && (m & 1))) ++m;
        if (m >= 8) return sign | (1u << 3); // 进位到 2^-6 normal
        return sign | (uint8_t)m;
    }
    const float t = std::ldexp(a, -(e - 3)); // ∈ [8,16)
    const float r = std::floor(t);
    const float frac = t - r;
    int m = (int)r;
    if (frac > 0.5f || (frac == 0.5f && (m & 1))) ++m;
    if (m >= 16) {
        m = 8;
        ++e;
    }
    if (e > 8 || (e == 8 && m >= 15)) return sign | 0x7Eu; // 溢出 clamp
    return sign | (uint8_t)((e + 7) << 3) | (uint8_t)(m & 7);
}

// 从 bf16 wpk 生成 fp8 wpk：169 个量化 section（42 层 × {qkv,o,gate_up,down}
// + lm_head，layer-major 序）。逐行 scale = max|w|/448（全零行 scale=1），
// q = e4m3(w/scale)。输出包仅含量化 section（dtype=2）：数据段 256B 对齐起，
// scale 段紧随。
inline bool generate_fp8_fixture(const std::string& src_wpk,
                                 const std::string& dst_path, std::string& err) {
    FILE* in = fopen(src_wpk.c_str(), "rb");
    if (in == nullptr) {
        err = "open src " + src_wpk;
        return false;
    }
    WpkHeaderIo hdr{};
    if (fread(&hdr, sizeof(hdr), 1, in) != 1) {
        fclose(in);
        err = "read header";
        return false;
    }
    std::vector<WpkSectionIo> toc(hdr.section_count);
    if (fseek(in, (long)hdr.toc_offset, SEEK_SET) != 0 ||
        fread(toc.data(), sizeof(WpkSectionIo), hdr.section_count, in) !=
            (size_t)hdr.section_count) {
        fclose(in);
        err = "read toc";
        return false;
    }
    // 168 个层量化 section + lm_head（layer-major：qkv,o,gate_up,down）
    const char* parts[4] = {"qkv", "o", "gate_up", "down"};
    std::vector<const WpkSectionIo*> qs;
    char name[64];
    for (uint32_t i = 0; i < 42; ++i) {
        for (int p = 0; p < 4; ++p) {
            snprintf(name, sizeof(name), "layer%u.%s", i, parts[p]);
            const uint64_t h = fnv1a64_io(name);
            const WpkSectionIo* hit = nullptr;
            for (const auto& t : toc)
                if (t.name_hash == h) {
                    hit = &t;
                    break;
                }
            if (hit == nullptr || hit->dtype != 0) {
                fclose(in);
                err = std::string("missing bf16 section ") + name;
                return false;
            }
            qs.push_back(hit);
        }
    }
    const WpkSectionIo* lm = nullptr;
    {
        const uint64_t h = fnv1a64_io("lm_head");
        for (const auto& t : toc)
            if (t.name_hash == h) {
                lm = &t;
                break;
            }
    }
    if (lm == nullptr || lm->dtype != 0) {
        fclose(in);
        err = "missing bf16 section lm_head";
        return false;
    }
    qs.push_back(lm);
    const size_t nq = qs.size(); // 169

    fs::create_directories(fs::path(dst_path).parent_path());
    FILE* out = fopen(dst_path.c_str(), "wb");
    if (out == nullptr) {
        fclose(in);
        err = "open dst " + dst_path;
        return false;
    }
    // 输出 TOC：数据段从 data_start（256B 对齐）起连续；scale 段紧随。
    WpkHeaderIo ohdr = hdr;
    ohdr.section_count = (uint32_t)nq;
    ohdr.data_start = (64 + nq * sizeof(WpkSectionIo) + 255ull) & ~255ull;
    std::vector<WpkSectionIo> otoc(nq);
    uint64_t cur = ohdr.data_start;
    for (size_t i = 0; i < nq; ++i) {
        const uint64_t dbytes = qs[i]->data_bytes / 2; // bf16 → fp8
        otoc[i] = *qs[i];
        otoc[i].dtype = 2;
        otoc[i].data_offset = cur;
        otoc[i].data_bytes = dbytes;
        cur += dbytes;
    }
    const uint64_t scale0 = (cur + 255ull) & ~255ull;
    {
        uint64_t sc = scale0;
        for (size_t i = 0; i < nq; ++i) {
            const uint64_t sbytes = (uint64_t)qs[i]->dims[0] * 4ull;
            otoc[i].scale_offset = sc;
            otoc[i].scale_bytes = sbytes;
            sc += sbytes;
        }
    }
    fwrite(&ohdr, sizeof(ohdr), 1, out);
    fwrite(otoc.data(), sizeof(WpkSectionIo), nq, out);
    std::vector<uint8_t> zeros(256, 0);
    for (uint64_t p = 64 + nq * sizeof(WpkSectionIo); p < ohdr.data_start;)
        p += fwrite(zeros.data(), 1, (size_t)std::min<uint64_t>(256, ohdr.data_start - p), out);

    // 数据段：逐 section 流式（行缓冲），scale 缓存 host（总 ~3.3MB）
    std::vector<std::vector<float>> scales(nq);
    std::vector<uint16_t> rowb;
    std::vector<uint8_t> qrow;
    for (size_t i = 0; i < nq; ++i) {
        const uint32_t rows = qs[i]->dims[0], cols = qs[i]->dims[1];
        if (fseek(in, (long)qs[i]->data_offset, SEEK_SET) != 0) {
            fclose(in);
            fclose(out);
            std::error_code rm;
            fs::remove(dst_path, rm);
            err = "seek src section";
            return false;
        }
        rowb.resize(cols);
        qrow.resize(cols);
        scales[i].resize(rows);
        for (uint32_t r = 0; r < rows; ++r) {
            if (fread(rowb.data(), 2, cols, in) != cols) {
                fclose(in);
                fclose(out);
                std::error_code rm;
                fs::remove(dst_path, rm);
                err = "read row";
                return false;
            }
            float amax = 0.f;
            for (uint32_t c = 0; c < cols; ++c)
                amax = std::max(amax, std::fabs(bf16_to_f32_io(rowb[c])));
            const float scale = amax > 0.f ? amax / 448.f : 1.f;
            scales[i][r] = scale;
            for (uint32_t c = 0; c < cols; ++c)
                qrow[c] = f32_to_e4m3_rne(bf16_to_f32_io(rowb[c]) / scale);
            if (fwrite(qrow.data(), 1, cols, out) != cols) {
                fclose(in);
                fclose(out);
                std::error_code rm;
                fs::remove(dst_path, rm);
                err = "write qrow";
                return false;
            }
        }
    }
    // scale 段（对齐 + 连续写出）
    while ((uint64_t)ftell(out) < scale0) {
        const long n = (long)std::min<uint64_t>(256, scale0 - ftell(out));
        fwrite(zeros.data(), 1, (size_t)n, out);
    }
    for (size_t i = 0; i < nq; ++i)
        fwrite(scales[i].data(), 4, scales[i].size(), out);
    fclose(in);
    fclose(out);
    err = "";
    return true;
}

} // namespace fp8fix
