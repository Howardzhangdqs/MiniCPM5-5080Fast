// weight_loader.h — wpk 权重包加载（P2 真实现，计划 §16.2）。
//
// WPK v1（小端）：
//   Header 64B：magic u32=0x31504B4D("MKP1")；abi_version u16=0x0002；
//   target_sm u16=120；model_hash u8[32]；section_count u32；reserved u32；
//   toc_offset u64=64；data_start u64。
//   TOC 项 128B：name_hash u64（FNV-1a 64，UTF-8 name；basis
//   14695981039346656037，prime 1099511628211）；dtype u32(0=BF16)；
//   layout u32(0=row-major [out,in])；rank u32；dims u32[8]；data_offset u64；
//   data_bytes u64；scale_offset u64；scale_bytes u64；reserved u64[6]。
// Section 名与形状必须与 generated_model_config.h 一致（逐项校验）。
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "minicpm_runtime.h"
#include "runtime/generated_model_config.h"
#include "runtime/memory_arena.h"

// 解析成功后的 device 权重引用（指针来自 model 级 WeightArena，load 后不可变）。
// 形状已对 constexpr 校验，hot path 直接使用 mc::cfg 常量。
//
// 分窗加载（P2 显存受限验证，gpu_layerwise）：环境变量在 load_wpk 内逐次解析
//（无静态缓存，同进程可多次按不同窗口加载）：
//   MC_LOAD_LAYERS_MIN=S   只加载 layer S..（默认 0）
//   MC_LOAD_LAYERS_MAX=N   只加载 ..layer N-1（默认 kNumLayers；与 MIN 配合）
//   MC_LOAD_SKIP_EMBEDDING=1  跳过 embedding 535MB（种子来自上一窗 residual）
//   MC_LOAD_SKIP_LM_HEAD=1    跳过 lm_head 535MB（截断窗无 logits 尾部）
// 窗口语义：layer_min/layer_max = 已加载层区间 [min,max)；final_norm 恒加载
//（4KB）；layer_max < kNumLayers 时额外加载 norm1[layer_max] 作为截断层的
// 「融合残差+下一层 norm」边界权重（forward 尾部融合 kernel 需要非空指针）。
// 未加载层的指针保持 nullptr——forward 侧有护栏，绝不解引用。
struct WeightRefs {
    const uint16_t* embedding  = nullptr; // [vocab, hidden]（分窗可空）
    const uint16_t* lm_head    = nullptr; // [vocab, hidden]（分窗可空）
    const uint16_t* final_norm = nullptr; // [hidden]（恒加载）
    const uint16_t* norm1[mc::cfg::kNumLayers]    = {}; // [hidden]
    const uint16_t* qkv[mc::cfg::kNumLayers]      = {}; // [2560, 2048] 行拼 q|k|v
    const uint16_t* o[mc::cfg::kNumLayers]        = {}; // [2048, 2048]
    const uint16_t* norm2[mc::cfg::kNumLayers]    = {}; // [hidden]
    const uint16_t* gate_up[mc::cfg::kNumLayers]  = {}; // [12288, 2048] 行拼 gate|up
    const uint16_t* down[mc::cfg::kNumLayers]     = {}; // [2048, 6144]

    // ---- P4 FP8 双拷贝（precision=MC_FP8 时由 load_wpk_fp8 填充）----
    // decode（M=1）用 FP8 权重 + 逐输出行 fp32 scale；prefill（M>1）用上面
    // 的 BF16 原权（同一 WeightRefs 内两套指针并存 = 双拷贝）。embedding/
    // 各 norm 无 FP8 拷贝（BF16 保留）。
    const uint8_t* fp8_qkv[mc::cfg::kNumLayers]     = {}; // [2560, 2048] e4m3
    const uint8_t* fp8_o[mc::cfg::kNumLayers]       = {}; // [2048, 2048]
    const uint8_t* fp8_gate_up[mc::cfg::kNumLayers] = {}; // [12288, 2048]
    const uint8_t* fp8_down[mc::cfg::kNumLayers]    = {}; // [2048, 6144]
    const float* scale_qkv[mc::cfg::kNumLayers]     = {}; // [2560]
    const float* scale_o[mc::cfg::kNumLayers]       = {}; // [2048]
    const float* scale_gate_up[mc::cfg::kNumLayers] = {}; // [12288]
    const float* scale_down[mc::cfg::kNumLayers]    = {}; // [2048]
    // P5：lm_head 的可选 fp8 量化段（包内存在 "lm_head" dtype=2 section 时
    // 填充；decode M=1 走 gemv_fp8，prefill 仍 BF16 原权）。离线量化产物
    //（工具链/测试 fixture）；MC_FP8_LMHEAD_OFF=1 时 loader 跳过消费。
    const uint8_t* fp8_lm_head = nullptr;                   // [vocab, hidden] e4m3
    const float*   scale_lm_head = nullptr;                 // [vocab]
    bool has_fp8 = false; // 上述指针有效（precision=MC_FP8 且加载成功）

    // ---- 分窗窗口（load_wpk 填充；默认全量）----
    uint32_t layer_min     = 0;                   // 首个已加载层
    uint32_t layer_max     = mc::cfg::kNumLayers; // 已加载层上界（不含）
    bool     has_embedding = true;                // false = embedding 指针为空
    bool     has_lm_head   = true;                // false = lm_head 指针为空

    // 指针布局与窗口字段互恰（加载器收尾自检）。
    bool consistent() const {
        if ((embedding != nullptr) != has_embedding) return false;
        if ((lm_head != nullptr) != has_lm_head) return false;
        if (final_norm == nullptr) return false;
        const uint32_t N = mc::cfg::kNumLayers;
        if (layer_min > layer_max || layer_max > N) return false;
        for (uint32_t i = 0; i < N; ++i) {
            const bool in_win  = (i >= layer_min && i < layer_max);
            const bool boundary = (i == layer_max && layer_max < N); // 边界 norm1
            if (in_win || boundary) {
                if (norm1[i] == nullptr) return false;
            } else if (norm1[i] != nullptr) {
                return false;
            }
            if (in_win) {
                if (qkv[i] == nullptr || o[i] == nullptr || norm2[i] == nullptr ||
                    gate_up[i] == nullptr || down[i] == nullptr)
                    return false;
            } else if (qkv[i] != nullptr || o[i] != nullptr || norm2[i] != nullptr ||
                       gate_up[i] != nullptr || down[i] != nullptr) {
                return false;
            }
        }
        return true;
    }

    // 全量加载（无分窗 env）时必须成立。
    bool complete() const {
        return layer_min == 0 && layer_max == mc::cfg::kNumLayers && has_embedding &&
               has_lm_head && consistent();
    }
};

struct WeightScanResult {
    bool        has_wpk    = false; // model_dir/model.wpk 存在
    uint64_t    wpk_bytes  = 0;
    bool        has_fp8_wpk = false; // model_dir/model_fp8.wpk 存在（P4）
    uint64_t    fp8_wpk_bytes = 0;
    std::vector<std::string> files; // 目录内全部常规文件名（排序后）
};

class WeightLoader {
public:
    // 扫描 model_dir。目录不存在/不可读 -> MC_E_IO。
    static mc_status_t scan(const std::string& model_dir, WeightScanResult& out);

    // 解析并上传 model_dir/model.wpk：
    //   - Header/TOC/魔法/版本/SM 校验；dims 逐项校验 vs generated_model_config.h
    //   - FNV-1a 64 名字哈希匹配 section
    //   - WeightArena：单次 cudaMalloc(total_span) + 分段流式上传（64MiB chunk，
    //     host 缓冲复用；load 时序允许大拷贝）
    // 日志打印 model_hash 前 8 字节 hex（期望 sha256 见
    // generated_model_config.h 注释；一致性由工具链保证，运行时不解析 JSON）。
    static mc_status_t load_wpk(const std::string& model_dir, int device_id,
                                DeviceArena& weight_arena, WeightRefs& refs,
                                uint64_t& total_uploaded_bytes);

    // P4：解析并上传 model_dir/model_fp8.wpk（dtype=2，E4M3 × 逐行 scale）。
    // 只消费 168 个量化 section（42 层 × {qkv,o,gate_up,down}）：
    //   uint8 数据 [N,K] → fp8_* 指针；fp32 scales [N]（scale_bytes 校验）
    //   → scale_* 指针；包内其他 section（若工具链附带 BF16 保留段）容忍但
    //   不上传——BF16 原权一律来自 load_wpk（双拷贝的另一份）。
    // fp8_arena 单独分配（caller 已做显存预检）；refs.has_fp8 置位。
    // 缺包/校验失败 → MC_E_MODEL_MISMATCH / MC_E_IO；显存不足 → arena 报
    // MC_E_OUT_OF_MEMORY（不静默降级）。
    static mc_status_t load_wpk_fp8(const std::string& model_dir, int device_id,
                                    DeviceArena& fp8_arena, WeightRefs& refs,
                                    uint64_t& total_uploaded_bytes);
};

// FNV-1a 64（供 loader 与测试工具共用）。
inline constexpr uint64_t fnv1a64(const char* s, size_t len) {
    uint64_t h = 14695981039346656037ull;
    for (size_t i = 0; i < len; ++i) {
        h ^= (uint64_t)(uint8_t)s[i];
        h *= 1099511628211ull;
    }
    return h;
}
