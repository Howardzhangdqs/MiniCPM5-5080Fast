// generated_model_config.h — MiniCPM5-2B 固定结构常量（计划 §10.3）
//
// 来源：models/minicpm5-2b/config.json（已核对，2026-09）。
// 这些值写死，避免 runtime shape dispatch；序列长度不能写死，但
// Page Table / Device State / Workspace 的地址在建期固定（§12.2）。
// P2 由 tools/generate_model_config.py 重新生成时保持本布局。
#pragma once

#include <cstdint>

namespace mc::cfg {

constexpr uint32_t kNumLayers        = 42;      // num_hidden_layers
constexpr uint32_t kHiddenSize       = 2048;    // hidden_size
constexpr uint32_t kIntermediateSize = 6144;    // intermediate_size
constexpr uint32_t kNumQueryHeads    = 16;      // num_attention_heads
constexpr uint32_t kNumKvHeads       = 2;       // num_key_value_heads (GQA)
constexpr uint32_t kHeadDim          = 128;     // head_dim
constexpr uint32_t kVocabSize        = 130560;  // vocab_size
constexpr uint32_t kMaxContextTokens = 131072;  // max_position_embeddings

constexpr float  kRmsEps    = 1e-6f;            // rms_norm_eps
constexpr double kRopeTheta = 5.0e6;            // rope_theta

constexpr int32_t kBosTokenId = 0;
constexpr int32_t kEosTokenIds[2] = {1, 130073}; // eos_token_id
constexpr int32_t kPadTokenId = 1;

// decode 停止信号：generated 达到 max_new_tokens 时返回该 id 让 host 停止。
constexpr int32_t kEosStreamId = 130073;

// KV cache 每 token 每层字节数（BF16 KV：K+V，各 kv_heads*head_dim 元素）。
constexpr uint64_t kKvBytesPerToken =
    (uint64_t)kNumLayers * 2 * kNumKvHeads * kHeadDim * 2u;

static_assert(kHiddenSize == kNumQueryHeads * kHeadDim, "hidden must equal q_heads*head_dim");

} // namespace mc::cfg
