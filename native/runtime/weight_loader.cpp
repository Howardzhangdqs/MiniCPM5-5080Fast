// weight_loader.cpp — wpk 解析/上传（P2 真实现）+ model_dir 扫描。
#include "runtime/weight_loader.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <unordered_map>
#include <vector>

#include <cuda_runtime.h>

#include "runtime/error.h"

namespace fs = std::filesystem;

namespace {

// ---------- WPK 二进制结构（小端，按任务书字段序） ----------
#pragma pack(push, 1)
struct WpkHeader {
    uint32_t magic;        // 0x31504B4D
    uint16_t abi_version;  // 0x0002
    uint16_t target_sm;    // 120
    uint8_t  model_hash[32];
    uint32_t section_count;
    uint32_t reserved;
    uint64_t toc_offset;   // == 64
    uint64_t data_start;
};
struct WpkSection {
    uint64_t name_hash;
    uint32_t dtype;        // 0 = BF16
    uint32_t layout;       // 0 = row-major [out,in]
    uint32_t rank;
    uint32_t dims[8];
    uint64_t data_offset;
    uint64_t data_bytes;
    uint64_t scale_offset;
    uint64_t scale_bytes;
    uint64_t reserved[6];
};
#pragma pack(pop)

static_assert(sizeof(WpkHeader) == 64, "wpk header must be 64 bytes");
// 注意：任务书文字写 "TOC 每项 128B"，但字段清单合计 132B（packed）：
// 8+4+4+4+32+8+8+8+8+48=132。以字段清单为准（"逐字段一致，不得擅改"），
// 本实现按 132B packed 解析；与工具链 lane 集成时若其 writer 不同需对齐。
static_assert(sizeof(WpkSection) == 132, "wpk toc entry per field list (packed)");

constexpr uint32_t kWpkMagic = 0x31504B4Du;
constexpr uint16_t kWpkAbi = 0x0002;
constexpr uint16_t kWpkTargetSm = 120;
constexpr size_t kUploadChunk = 64ull << 20; // 64 MiB 流式上传块
// 完整 wpk 的 section 总数（embedding/lm_head/final_norm + 42 层 × 6）。
// 分窗加载只上传子集，但 TOC 必须完整（包完整性校验不变）。
constexpr size_t kFullSectionCount = 3 + (size_t)mc::cfg::kNumLayers * 6;

// 期望 section：name → rank/dims（与 generated_model_config.h 逐项绑定）。
struct ExpectedSection {
    std::string name;
    uint64_t hash;
    uint32_t rank;
    uint32_t dims[8];
    uint64_t elems;
};

// 分窗加载窗口（MC_LOAD_LAYERS_MIN/MAX + SKIP_*，load_wpk 每次调用现读 env，
// 无静态缓存——同进程可按不同窗口多次加载，gpu_layerwise 依赖此语义）。
struct LoadWindow {
    uint32_t min_layer = 0;
    uint32_t max_layer = mc::cfg::kNumLayers;
    bool load_embedding = true;
    bool load_lm_head   = true;
};

uint32_t parse_u32_env(const char* name, bool& present) {
    const char* v = getenv(name);
    present = (v != nullptr && v[0] != '\0');
    if (!present) return 0;
    return (uint32_t)strtoul(v, nullptr, 10);
}

LoadWindow parse_window() {
    LoadWindow w;
    bool present = false;
    uint32_t v = parse_u32_env("MC_LOAD_LAYERS_MIN", present);
    if (present) w.min_layer = v;
    v = parse_u32_env("MC_LOAD_LAYERS_MAX", present);
    if (present) w.max_layer = v;
    if (parse_u32_env("MC_LOAD_SKIP_EMBEDDING", present) != 0 && present)
        w.load_embedding = false;
    if (parse_u32_env("MC_LOAD_SKIP_LM_HEAD", present) != 0 && present)
        w.load_lm_head = false;
    return w;
}

std::vector<ExpectedSection> build_expected(const LoadWindow& win) {
    using namespace mc::cfg;
    std::vector<ExpectedSection> v;
    auto add = [&](std::string n, std::initializer_list<uint32_t> d) {
        ExpectedSection e{};
        e.name = n;
        e.hash = fnv1a64(n.c_str(), n.size());
        e.rank = (uint32_t)d.size();
        uint64_t elems = 1;
        uint32_t i = 0;
        for (uint32_t x : d) {
            e.dims[i++] = x;
            elems *= x;
        }
        e.elems = elems;
        v.push_back(std::move(e));
    };
    if (win.load_embedding) add("embedding", {kVocabSize, kHiddenSize});
    if (win.load_lm_head) add("lm_head", {kVocabSize, kHiddenSize});
    add("final_norm", {kHiddenSize}); // 恒加载：末层融合 / 尾部 rmsnorm 需要
    char buf[64];
    for (uint32_t i = 0; i < kNumLayers; ++i) {
        const bool in_win = (i >= win.min_layer && i < win.max_layer);
        // 截断层（max_layer < kNumLayers）的下一层 norm1：层尾融合
        // residual_add_rmsnorm 的边界权重（kernel 必须读非空指针）。
        const bool boundary_norm1 = (i == win.max_layer && win.max_layer < kNumLayers);
        if (boundary_norm1) {
            snprintf(buf, sizeof(buf), "layer%u.norm1", i);
            add(buf, {kHiddenSize});
            continue;
        }
        if (!in_win) continue;
        snprintf(buf, sizeof(buf), "layer%u.norm1", i);
        add(buf, {kHiddenSize});
        snprintf(buf, sizeof(buf), "layer%u.qkv", i);
        add(buf, {kNumQueryHeads * kHeadDim + 2 * kNumKvHeads * kHeadDim, kHiddenSize});
        snprintf(buf, sizeof(buf), "layer%u.o", i);
        add(buf, {kHiddenSize, kHiddenSize});
        snprintf(buf, sizeof(buf), "layer%u.norm2", i);
        add(buf, {kHiddenSize});
        snprintf(buf, sizeof(buf), "layer%u.gate_up", i);
        add(buf, {2 * kIntermediateSize, kHiddenSize});
        snprintf(buf, sizeof(buf), "layer%u.down", i);
        add(buf, {kHiddenSize, kIntermediateSize});
    }
    return v;
}

struct FileGuard {
    FILE* f;
    ~FileGuard() {
        if (f) fclose(f);
    }
};

} // namespace

mc_status_t WeightLoader::scan(const std::string& model_dir, WeightScanResult& out) {
    out = WeightScanResult{};

    std::error_code ec;
    const fs::path dir(model_dir);
    if (!fs::exists(dir, ec)) {
        mc::set_error("model_dir does not exist: %s", model_dir.c_str());
        return MC_E_IO;
    }
    if (!fs::is_directory(dir, ec)) {
        mc::set_error("model_dir is not a directory: %s", model_dir.c_str());
        return MC_E_IO;
    }

    for (const auto& entry : fs::directory_iterator(dir, ec)) {
        if (ec) break;
        std::error_code sec;
        if (entry.is_regular_file(sec)) {
            out.files.push_back(entry.path().filename().string());
            const std::string fn = entry.path().filename().string();
            if (fn == "model.wpk") {
                out.has_wpk = true;
                out.wpk_bytes = (uint64_t)entry.file_size(sec);
            } else if (fn == "model_fp8.wpk") { // P4
                out.has_fp8_wpk = true;
                out.fp8_wpk_bytes = (uint64_t)entry.file_size(sec);
            }
        }
    }
    if (ec) {
        mc::set_error("failed to scan model_dir %s: %s", model_dir.c_str(),
                      ec.message().c_str());
        return MC_E_IO;
    }

    std::sort(out.files.begin(), out.files.end());
    return MC_OK;
}

mc_status_t WeightLoader::load_wpk(const std::string& model_dir, int device_id,
                                   DeviceArena& weight_arena, WeightRefs& refs,
                                   uint64_t& total_uploaded_bytes) {
    (void)device_id; // 调用方（mc_model_load）已 cudaSetDevice
    total_uploaded_bytes = 0;
    refs = WeightRefs{};
    const std::string path = (fs::path(model_dir) / "model.wpk").string();

    FILE* f = fopen(path.c_str(), "rb");
    if (f == nullptr) {
        mc::set_error("wpk open failed: %s", path.c_str());
        return MC_E_IO;
    }
    FileGuard guard{f};

    // 文件大小（header/toc/data 区间校验用）
    if (fseek(f, 0, SEEK_END) != 0) {
        mc::set_error("wpk seek end failed: %s", path.c_str());
        return MC_E_IO;
    }
    const uint64_t file_size = (uint64_t)ftell(f);
    rewind(f);

    // ---- header ----
    WpkHeader hdr{};
    if (fread(&hdr, sizeof(hdr), 1, f) != 1) {
        mc::set_error("wpk read header failed: %s", path.c_str());
        return MC_E_IO;
    }
    if (hdr.magic != kWpkMagic) {
        mc::set_error("wpk magic mismatch: got 0x%08x expected 0x%08x", hdr.magic,
                      kWpkMagic);
        return MC_E_MODEL_MISMATCH;
    }
    if (hdr.abi_version != kWpkAbi) {
        mc::set_error("wpk abi_version mismatch: got 0x%04x expected 0x%04x",
                      hdr.abi_version, kWpkAbi);
        return MC_E_MODEL_MISMATCH;
    }
    if (hdr.target_sm != kWpkTargetSm) {
        mc::set_error("wpk target_sm mismatch: got %u expected %u", hdr.target_sm,
                      kWpkTargetSm);
        return MC_E_MODEL_MISMATCH;
    }
    if (hdr.toc_offset != 64) {
        mc::set_error("wpk toc_offset must be 64 (got %llu)",
                      (unsigned long long)hdr.toc_offset);
        return MC_E_MODEL_MISMATCH;
    }
    if (hdr.reserved != 0) {
        mc::set_error("wpk reserved must be 0 (got %u)", hdr.reserved);
        return MC_E_MODEL_MISMATCH;
    }
    if (hdr.data_start > file_size) {
        mc::set_error("wpk data_start %llu exceeds file size %llu",
                      (unsigned long long)hdr.data_start, (unsigned long long)file_size);
        return MC_E_MODEL_MISMATCH;
    }
    // TOC 区间必须完整落在 data_start 之前（132B 步长 × count）
    if ((uint64_t)hdr.section_count * sizeof(WpkSection) >
            hdr.data_start - hdr.toc_offset ||
        hdr.section_count == 0) {
        mc::set_error("wpk TOC span %llu..%llu exceeds data_start=%llu (count=%u)",
                      (unsigned long long)hdr.toc_offset,
                      (unsigned long long)(hdr.toc_offset +
                                           (uint64_t)hdr.section_count *
                                               sizeof(WpkSection)),
                      (unsigned long long)hdr.data_start, hdr.section_count);
        return MC_E_MODEL_MISMATCH;
    }
    {
        char hex[17];
        for (int i = 0; i < 8; ++i) snprintf(hex + i * 2, 3, "%02x", hdr.model_hash[i]);
        hex[16] = '\0';
        // 期望 sha256（权重文件级）记录在 generated_model_config.h 头部注释；
        // 与 reference_manifest.json 的一致性由工具链保证，运行时不解析 JSON。
        fprintf(stderr,
                "[minicpm-native] wpk: sections=%u data_start=%llu model_hash[:8]=%s\n",
                hdr.section_count, (unsigned long long)hdr.data_start, hex);
    }

    // ---- TOC ----
    const uint32_t nsec = hdr.section_count;
    std::vector<WpkSection> toc(nsec);
    if (nsec > 0) {
        if (fseek(f, (long)hdr.toc_offset, SEEK_SET) != 0 ||
            fread(toc.data(), sizeof(WpkSection), nsec, f) != nsec) {
            mc::set_error("wpk read TOC failed (%u sections)", nsec);
            return MC_E_IO;
        }
    }

    // ---- 分窗（env 逐次解析；同进程多次加载各自生效）----
    const LoadWindow win = parse_window();
    if (win.min_layer > mc::cfg::kNumLayers || win.max_layer > mc::cfg::kNumLayers ||
        win.min_layer >= win.max_layer) {
        mc::set_error("wpk load window invalid: MC_LOAD_LAYERS_MIN=%u MAX=%u "
                      "(need 0 <= MIN < MAX <= %u)",
                      win.min_layer, win.max_layer, mc::cfg::kNumLayers);
        return MC_E_INVALID_ARGUMENT;
    }

    // ---- 期望表 + 哈希索引 + 逐项校验 ----
    std::vector<ExpectedSection> expected = build_expected(win);
    std::unordered_map<uint64_t, const WpkSection*> by_hash;
    for (const auto& s : toc) {
        if (by_hash.count(s.name_hash)) {
            mc::set_error("wpk TOC has duplicate name_hash 0x%016llx",
                          (unsigned long long)s.name_hash);
            return MC_E_MODEL_MISMATCH;
        }
        by_hash.emplace(s.name_hash, &s);
    }
    // 包完整性：TOC 仍须包含全部 255 section（分窗只影响上传子集）。
    if (by_hash.size() != kFullSectionCount) {
        mc::set_error("wpk section count mismatch: TOC has %zu unique, expected %zu",
                      by_hash.size(), kFullSectionCount);
        return MC_E_MODEL_MISMATCH;
    }

    struct Placement {
        const ExpectedSection* spec;
        const WpkSection* sec;
        uint64_t arena_off; // 紧凑放置偏移（256B 对齐）
    };
    std::vector<Placement> placed;
    placed.reserve(expected.size());
    for (const auto& e : expected) {
        auto it = by_hash.find(e.hash);
        if (it == by_hash.end()) {
            mc::set_error("wpk missing section '%s' (fnv1a=0x%016llx)", e.name.c_str(),
                          (unsigned long long)e.hash);
            return MC_E_MODEL_MISMATCH;
        }
        const WpkSection* s = it->second;
        if (s->dtype != 0 || s->layout != 0) {
            mc::set_error("wpk section '%s': dtype/layout must be BF16/row-major "
                          "(got %u/%u)",
                          e.name.c_str(), s->dtype, s->layout);
            return MC_E_MODEL_MISMATCH;
        }
        if (s->rank != e.rank || memcmp(s->dims, e.dims, sizeof(uint32_t) * e.rank) != 0) {
            mc::set_error("wpk section '%s': dims mismatch (rank %u vs %u)",
                          e.name.c_str(), s->rank, e.rank);
            return MC_E_MODEL_MISMATCH;
        }
        if (s->data_bytes != e.elems * 2ull /*BF16*/) {
            mc::set_error("wpk section '%s': data_bytes=%llu but expected %llu",
                          e.name.c_str(), (unsigned long long)s->data_bytes,
                          (unsigned long long)(e.elems * 2ull));
            return MC_E_MODEL_MISMATCH;
        }
        if (s->data_offset < hdr.data_start) {
            mc::set_error("wpk section '%s': data_offset before data_start",
                          e.name.c_str());
            return MC_E_MODEL_MISMATCH;
        }
        placed.push_back({&e, s, 0});
    }

    // ---- 紧凑放置 + 单次 cudaMalloc + 分段流式上传（64MiB chunk）----
    // 分窗加载的 section 在文件里不连续（被跳过的 section 留下空洞——例如
    // lm_head 的 535MB 夹在 embedding 与 final_norm 之间），按文件 span
    // [min_off,max_end) 分配会把空洞一并 cudaMalloc。改为按加载顺序紧凑
    // 排布（每段 256B 对齐）：arena 大小 == 实际 payload，无空洞。
    uint64_t cursor = 0;
    for (auto& p : placed) {
        cursor = (cursor + 255ull) & ~255ull;
        p.arena_off = cursor;
        cursor += p.sec->data_bytes;
    }
    mc_status_t rs = weight_arena.init((size_t)cursor, "model.weight_arena");
    if (rs != MC_OK) return rs;

    std::vector<unsigned char> chunk(kUploadChunk);
    uint64_t uploaded = 0;
    for (const auto& p : placed) {
        if (fseek(f, (long)p.sec->data_offset, SEEK_SET) != 0) {
            mc::set_error("wpk seek failed for '%s'", p.spec->name.c_str());
            return MC_E_IO;
        }
        uint8_t* dst = (uint8_t*)weight_arena.base_ptr() + p.arena_off;
        uint64_t remain = p.sec->data_bytes;
        while (remain > 0) {
            const size_t n = (size_t)std::min<uint64_t>(remain, kUploadChunk);
            if (fread(chunk.data(), 1, n, f) != n) {
                mc::set_error("wpk read failed for '%s' (%llu bytes remain)",
                              p.spec->name.c_str(), (unsigned long long)remain);
                return MC_E_IO;
            }
            cudaError_t e = cudaMemcpy(dst + (p.sec->data_bytes - remain), chunk.data(),
                                       n, cudaMemcpyHostToDevice);
            if (e != cudaSuccess) {
                cudaGetLastError();
                mc::set_error("wpk upload failed for '%s': %s", p.spec->name.c_str(),
                              cudaGetErrorString(e));
                return MC_E_CUDA;
            }
            remain -= n;
        }
        uploaded += p.sec->data_bytes;
    }
    total_uploaded_bytes = uploaded;

    // ---- 填 WeightRefs（device 指针 + 隐含 dims 已对 constexpr 校验）----
    // name_hash → 紧凑放置偏移（arena 内无空洞，偏移即 Placement 决策）。
    std::unordered_map<uint64_t, uint64_t> off_of;
    for (const auto& p : placed) off_of.emplace(p.spec->hash, p.arena_off);
    auto ptr_of = [&](const char* name) -> const uint16_t* {
        const uint64_t h = fnv1a64(name, strlen(name));
        auto it = off_of.find(h);
        if (it == off_of.end()) return nullptr;
        return (const uint16_t*)((uint8_t*)weight_arena.base_ptr() + it->second);
    };
    char buf[64];
    refs.layer_min     = win.min_layer;
    refs.layer_max     = win.max_layer;
    refs.has_embedding = win.load_embedding;
    refs.has_lm_head   = win.load_lm_head;
    refs.embedding = win.load_embedding ? ptr_of("embedding") : nullptr;
    refs.lm_head   = win.load_lm_head ? ptr_of("lm_head") : nullptr;
    refs.final_norm = ptr_of("final_norm"); // 恒加载
    const bool boundary = (win.max_layer < mc::cfg::kNumLayers);
    for (uint32_t i = 0; i < mc::cfg::kNumLayers; ++i) {
        const bool in_win = (i >= win.min_layer && i < win.max_layer);
        const bool boundary_norm1 = boundary && (i == win.max_layer);
        if (boundary_norm1) {
            snprintf(buf, sizeof(buf), "layer%u.norm1", i);
            refs.norm1[i] = ptr_of(buf); // 截断层融合 kernel 的边界 norm
            continue;
        }
        if (!in_win) continue; // 窗外层指针保持 nullptr（forward 护栏保证不触碰）
        snprintf(buf, sizeof(buf), "layer%u.norm1", i);
        refs.norm1[i] = ptr_of(buf);
        snprintf(buf, sizeof(buf), "layer%u.qkv", i);
        refs.qkv[i] = ptr_of(buf);
        snprintf(buf, sizeof(buf), "layer%u.o", i);
        refs.o[i] = ptr_of(buf);
        snprintf(buf, sizeof(buf), "layer%u.norm2", i);
        refs.norm2[i] = ptr_of(buf);
        snprintf(buf, sizeof(buf), "layer%u.gate_up", i);
        refs.gate_up[i] = ptr_of(buf);
        snprintf(buf, sizeof(buf), "layer%u.down", i);
        refs.down[i] = ptr_of(buf);
    }
    if (!refs.consistent()) { // 前面已逐项校验，此处仅防御（含窗口一致性）
        mc::set_error("wpk internal error: inconsistent WeightRefs for window "
                      "[%u,%u)",
                      refs.layer_min, refs.layer_max);
        return MC_E_INTERNAL;
    }
    if (refs.complete()) {
        fprintf(stderr,
                "[minicpm-native] wpk loaded: %zu sections, %llu bytes uploaded "
                "(arena %llu bytes)\n",
                expected.size(), (unsigned long long)uploaded,
                (unsigned long long)weight_arena.capacity());
    } else {
        fprintf(stderr,
                "[minicpm-native] wpk partial load: layers=[%u,%u) embedding=%d "
                "lm_head=%d final_norm=1 boundary_norm=%d — %zu sections, %llu "
                "bytes uploaded (arena %llu bytes, compact)\n",
                refs.layer_min, refs.layer_max, refs.has_embedding ? 1 : 0,
                refs.has_lm_head ? 1 : 0, boundary ? 1 : 0, expected.size(),
                (unsigned long long)uploaded,
                (unsigned long long)weight_arena.capacity());
    }
    return MC_OK;
}

// ============================ P4：FP8 权重包 ============================
// model_fp8.wpk：Header/TOC 结构与 model.wpk 一致（magic/abi/sm/132B 项）。
// 量化 section：dtype=2，data = uint8 [N,K]（E4M3），scale = fp32 [N]
//（scale_offset/scale_bytes）。本 loader 只消费 168 个量化 section；
// BF16 原权由 load_wpk 提供（双拷贝的另一份），包内其他 section 容忍但忽略。
mc_status_t WeightLoader::load_wpk_fp8(const std::string& model_dir, int device_id,
                                       DeviceArena& fp8_arena, WeightRefs& refs,
                                       uint64_t& total_uploaded_bytes) {
    (void)device_id; // 调用方（mc_model_load）已 cudaSetDevice
    total_uploaded_bytes = 0;
    const std::string path = (fs::path(model_dir) / "model_fp8.wpk").string();

    FILE* f = fopen(path.c_str(), "rb");
    if (f == nullptr) {
        mc::set_error("fp8 wpk open failed: %s (precision=MC_FP8 requires "
                      "model_fp8.wpk with dtype=2 sections + fp32 row scales)",
                      path.c_str());
        return MC_E_MODEL_MISMATCH;
    }
    FileGuard guard{f};

    if (fseek(f, 0, SEEK_END) != 0) {
        mc::set_error("fp8 wpk seek end failed: %s", path.c_str());
        return MC_E_IO;
    }
    const uint64_t file_size = (uint64_t)ftell(f);
    rewind(f);

    WpkHeader hdr{};
    if (fread(&hdr, sizeof(hdr), 1, f) != 1) {
        mc::set_error("fp8 wpk read header failed: %s", path.c_str());
        return MC_E_IO;
    }
    if (hdr.magic != kWpkMagic || hdr.abi_version != kWpkAbi ||
        hdr.target_sm != kWpkTargetSm || hdr.toc_offset != 64 || hdr.reserved != 0 ||
        hdr.data_start > file_size || hdr.section_count == 0 ||
        (uint64_t)hdr.section_count * sizeof(WpkSection) >
            hdr.data_start - hdr.toc_offset) {
        mc::set_error("fp8 wpk header invalid (magic=0x%08x abi=0x%04x sm=%u)",
                      hdr.magic, hdr.abi_version, hdr.target_sm);
        return MC_E_MODEL_MISMATCH;
    }

    const uint32_t nsec = hdr.section_count;
    std::vector<WpkSection> toc(nsec);
    if (fseek(f, (long)hdr.toc_offset, SEEK_SET) != 0 ||
        fread(toc.data(), sizeof(WpkSection), nsec, f) != nsec) {
        mc::set_error("fp8 wpk read TOC failed (%u sections)", nsec);
        return MC_E_IO;
    }
    std::unordered_map<uint64_t, const WpkSection*> by_hash;
    for (const auto& s : toc) {
        if (by_hash.count(s.name_hash)) {
            mc::set_error("fp8 wpk TOC has duplicate name_hash 0x%016llx",
                          (unsigned long long)s.name_hash);
            return MC_E_MODEL_MISMATCH;
        }
        by_hash.emplace(s.name_hash, &s);
    }

    // ---- 期望的 168 个量化 section（42 层 × {qkv,o,gate_up,down}）----
    using namespace mc::cfg;
    struct QuantSpec {
        std::string name;
        uint64_t hash;
        uint32_t rows, cols;
    };
    std::vector<QuantSpec> want;
    want.reserve((size_t)kNumLayers * 4);
    {
        char buf[64];
        const uint32_t qkv_n = kNumQueryHeads * kHeadDim + 2 * kNumKvHeads * kHeadDim;
        for (uint32_t i = 0; i < kNumLayers; ++i) {
            auto add = [&](const char* suffix, uint32_t r, uint32_t c) {
                snprintf(buf, sizeof(buf), "layer%u.%s", i, suffix);
                want.push_back({buf, fnv1a64(buf, strlen(buf)), r, c});
            };
            add("qkv", qkv_n, kHiddenSize);
            add("o", kHiddenSize, kHiddenSize);
            add("gate_up", 2 * kIntermediateSize, kHiddenSize);
            add("down", kHiddenSize, kIntermediateSize);
        }
    }
    // ---- P5：可选 lm_head 量化段（dtype=2，[vocab, hidden] + 行 scale；
    //      工具链/测试 fixture 离线产出。存在即消费；MC_FP8_LMHEAD_OFF=1
    //      跳过 = decode lm_head 回 BF16 的 A/B 对照）----
    bool want_lm_head = false;
    if (getenv("MC_FP8_LMHEAD_OFF") == nullptr) {
        const std::string lname = "lm_head";
        const uint64_t lhash = fnv1a64(lname.c_str(), lname.size());
        auto lit = by_hash.find(lhash);
        if (lit != by_hash.end() && lit->second->dtype == 2) {
            want.push_back({lname, lhash, kVocabSize, kHiddenSize});
            want_lm_head = true;
        }
    }
    const size_t layer_sections = (size_t)kNumLayers * 4;

    // ---- 逐项校验 + 紧凑放置（数据段 16B 对齐：fp8 GEMV 的 uint4 装载；
    //      scale 段 4B 对齐自然满足）----
    struct Placement {
        const QuantSpec* spec;
        const WpkSection* sec;
        uint64_t arena_off;
        uint64_t scale_off;
    };
    std::vector<Placement> placed;
    placed.reserve(want.size());
    // 两段式布局：全部数据段在前（16B 对齐），全部 scale 段紧随（4B）。
    struct Offsets {
        uint64_t data_off = 0;
        uint64_t scale_off = 0;
    };
    std::vector<Offsets> offs(want.size());
    uint64_t cursor = 0;
    for (size_t i = 0; i < want.size(); ++i) {
        auto it = by_hash.find(want[i].hash);
        if (it == by_hash.end()) {
            mc::set_error("fp8 wpk missing quantized section '%s' (fnv1a=0x%016llx)",
                          want[i].name.c_str(), (unsigned long long)want[i].hash);
            return MC_E_MODEL_MISMATCH;
        }
        const WpkSection* s = it->second;
        if (s->dtype != 2 || s->layout != 0) {
            mc::set_error("fp8 wpk section '%s': dtype/layout must be 2/row-major "
                          "(got %u/%u)",
                          want[i].name.c_str(), s->dtype, s->layout);
            return MC_E_MODEL_MISMATCH;
        }
        if (s->rank != 2 || s->dims[0] != want[i].rows || s->dims[1] != want[i].cols) {
            mc::set_error("fp8 wpk section '%s': dims mismatch (got %u×%u, want "
                          "%u×%u)",
                          want[i].name.c_str(), s->dims[0], s->dims[1], want[i].rows,
                          want[i].cols);
            return MC_E_MODEL_MISMATCH;
        }
        const uint64_t dbytes = (uint64_t)want[i].rows * want[i].cols; /*1B/元素*/
        if (s->data_bytes != dbytes || s->scale_bytes != (uint64_t)want[i].rows * 4ull) {
            mc::set_error("fp8 wpk section '%s': data_bytes=%llu (want %llu), "
                          "scale_bytes=%llu (want %llu)",
                          want[i].name.c_str(), (unsigned long long)s->data_bytes,
                          (unsigned long long)dbytes,
                          (unsigned long long)s->scale_bytes,
                          (unsigned long long)((uint64_t)want[i].rows * 4ull));
            return MC_E_MODEL_MISMATCH;
        }
        if (s->data_offset < hdr.data_start || s->scale_offset < hdr.data_start ||
            s->data_offset + s->data_bytes > file_size ||
            s->scale_offset + s->scale_bytes > file_size) {
            mc::set_error("fp8 wpk section '%s': data/scale range out of file",
                          want[i].name.c_str());
            return MC_E_MODEL_MISMATCH;
        }
        // 布局：数据段 16B 对齐依次排布；scale 段随后 4B 对齐。
        cursor = (cursor + 15ull) & ~15ull;
        offs[i].data_off = cursor;
        cursor += dbytes;
    }
    uint64_t scale_cursor = (cursor + 3ull) & ~3ull;
    for (size_t i = 0; i < want.size(); ++i) {
        auto it = by_hash.find(want[i].hash);
        offs[i].scale_off = scale_cursor;
        scale_cursor += (uint64_t)want[i].rows * 4ull;
        placed.push_back({&want[i], it->second, offs[i].data_off, offs[i].scale_off});
    }
    const uint64_t arena_bytes = scale_cursor;

    mc_status_t rs = fp8_arena.init((size_t)arena_bytes, "model.fp8_arena");
    if (rs != MC_OK) return rs; // OOM → MC_E_OUT_OF_MEMORY（显式，不降级）

    // ---- 流式上传（数据 + scale；64MiB chunk）----
    std::vector<unsigned char> chunk(kUploadChunk);
    uint64_t uploaded = 0;
    uint8_t* base = (uint8_t*)fp8_arena.base_ptr();
    for (const auto& p : placed) {
        const uint64_t dbytes = (uint64_t)p.spec->rows * p.spec->cols;
        if (fseek(f, (long)p.sec->data_offset, SEEK_SET) != 0) {
            mc::set_error("fp8 wpk seek failed for '%s'", p.spec->name.c_str());
            return MC_E_IO;
        }
        uint8_t* dst = base + p.arena_off;
        uint64_t remain = dbytes;
        while (remain > 0) {
            const size_t n = (size_t)std::min<uint64_t>(remain, kUploadChunk);
            if (fread(chunk.data(), 1, n, f) != n) {
                mc::set_error("fp8 wpk read failed for '%s'", p.spec->name.c_str());
                return MC_E_IO;
            }
            cudaError_t e = cudaMemcpy(dst + (dbytes - remain), chunk.data(), n,
                                       cudaMemcpyHostToDevice);
            if (e != cudaSuccess) {
                cudaGetLastError();
                mc::set_error("fp8 wpk upload failed for '%s': %s",
                              p.spec->name.c_str(), cudaGetErrorString(e));
                return MC_E_CUDA;
            }
            remain -= n;
        }
        uploaded += dbytes;
        // scale 段（≤ 48KB/section，一次读完）
        if (fseek(f, (long)p.sec->scale_offset, SEEK_SET) != 0 ||
            fread(chunk.data(), 1, (size_t)p.spec->rows * 4u, f) !=
                (size_t)p.spec->rows * 4u) {
            mc::set_error("fp8 wpk read scales failed for '%s'", p.spec->name.c_str());
            return MC_E_IO;
        }
        cudaError_t e = cudaMemcpy(base + p.scale_off, chunk.data(),
                                   (size_t)p.spec->rows * 4u,
                                   cudaMemcpyHostToDevice);
        if (e != cudaSuccess) {
            cudaGetLastError();
            mc::set_error("fp8 wpk upload scales failed for '%s': %s",
                          p.spec->name.c_str(), cudaGetErrorString(e));
            return MC_E_CUDA;
        }
        uploaded += (uint64_t)p.spec->rows * 4u;
    }
    total_uploaded_bytes = uploaded;

    // ---- 填 WeightRefs 的 fp8 指针（want/placed 为 layer-major 序：
    //      每层 qkv,o,gate_up,down 四连 → 直接索引，无字符串解析）----
    for (uint32_t i = 0; i < kNumLayers; ++i) {
        const Placement& pq = placed[(size_t)i * 4 + 0];
        const Placement& po = placed[(size_t)i * 4 + 1];
        const Placement& pg = placed[(size_t)i * 4 + 2];
        const Placement& pd = placed[(size_t)i * 4 + 3];
        refs.fp8_qkv[i]     = base + pq.arena_off;
        refs.scale_qkv[i]   = (const float*)(base + pq.scale_off);
        refs.fp8_o[i]       = base + po.arena_off;
        refs.scale_o[i]     = (const float*)(base + po.scale_off);
        refs.fp8_gate_up[i] = base + pg.arena_off;
        refs.scale_gate_up[i] = (const float*)(base + pg.scale_off);
        refs.fp8_down[i]    = base + pd.arena_off;
        refs.scale_down[i]  = (const float*)(base + pd.scale_off);
    }
    for (uint32_t i = 0; i < kNumLayers; ++i) {
        if (refs.fp8_qkv[i] == nullptr || refs.fp8_o[i] == nullptr ||
            refs.fp8_gate_up[i] == nullptr || refs.fp8_down[i] == nullptr ||
            refs.scale_qkv[i] == nullptr || refs.scale_o[i] == nullptr ||
            refs.scale_gate_up[i] == nullptr || refs.scale_down[i] == nullptr) {
            mc::set_error("fp8 wpk internal error: layer %u pointers incomplete", i);
            return MC_E_INTERNAL;
        }
    }
    // P5：可选 lm_head fp8 段（placed 尾部第 169 项）
    if (want_lm_head) {
        const Placement& pl = placed[layer_sections];
        refs.fp8_lm_head = base + pl.arena_off;
        refs.scale_lm_head = (const float*)(base + pl.scale_off);
        fprintf(stderr,
                "[minicpm-native] fp8 wpk: lm_head fp8 section consumed "
                "(decode lm_head on gemv_fp8; MC_FP8_LMHEAD_OFF=1 to disable)\n");
    }
    refs.has_fp8 = true;
    fprintf(stderr,
            "[minicpm-native] fp8 wpk loaded: %zu quantized sections, %llu bytes "
            "uploaded (fp8_arena %llu bytes)%s\n",
            placed.size(), (unsigned long long)uploaded,
            (unsigned long long)fp8_arena.capacity(),
            want_lm_head ? " [+lm_head fp8]" : "");
    return MC_OK;
}
