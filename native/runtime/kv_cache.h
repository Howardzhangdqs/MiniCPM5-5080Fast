// kv_cache.h — Paged KV Cache：统一寻址 + PagedKvAllocator（计划 §11.1）。
//
// ===== KV 布局（最终定义，P2/P3 所有 kernel 必须通过 KvLayout 寻址）=====
//   kv_pool 元素类型 bf16（uint16 裸位），一维平铺：
//     [layer 42][page max_pages][slot 32][kv 2(K,V)][kv_head 2][dim 128]
//   元素偏移（bf16 计）：
//     off = (((layer * max_pages + page) * 32 + slot) * 2 + kv) * 256
//           + head * 128 + d
//   页表：d_page_table[pos] = page_id（u32）；pos → page = table[pos/32],
//   slot = pos%32。每层页数 = 总页数（页面按层分区，页 id 在层内唯一）。
//   容量：max_pages = ceil(max_context_tokens / 32)。
#pragma once

#include <cstdint>
#include <vector>

#include <cuda_runtime.h>

#include "minicpm_runtime.h"

#if defined(__CUDA_ARCH__)
#define MC_KV_HD __device__
#else
#define MC_KV_HD inline
#endif

// host/device 共用的寻址函数（任务书 §4：rope_kv 写入与 attention 读取
// 必须同一实现）。所有参数为编译期常量 × 运行期布局参数。
struct KvLayout {
    static constexpr uint32_t kTokensPerPage = 32;
    static constexpr uint32_t kKvOrV = 2;     // K, V
    static constexpr uint32_t kKvHeads = 2;
    static constexpr uint32_t kHeadDim = 128;
    static constexpr uint32_t kSlotElems = kKvOrV * kKvHeads * kHeadDim; // 512
    static constexpr uint32_t kPageElems = kTokensPerPage * kSlotElems;  // 16384

    uint32_t max_pages = 0; // 每层页数

    // bf16 元素偏移
    MC_KV_HD uint64_t elem_offset(uint32_t layer, uint32_t page, uint32_t slot,
                                  uint32_t kv, uint32_t head, uint32_t d) const {
        return (((uint64_t)layer * max_pages + page) * kTokensPerPage + slot) *
                   kSlotElems +
               (uint64_t)kv * kKvHeads * kHeadDim + (uint64_t)head * kHeadDim + d;
    }
    // 由绝对 pos 解析页/槽
    MC_KV_HD void locate(uint32_t pos, uint32_t table_page, uint32_t* page,
                         uint32_t* slot) const {
        *page = table_page;
        *slot = pos % kTokensPerPage;
    }
    MC_KV_HD static uint32_t page_index(uint32_t pos) {
        return pos / kTokensPerPage;
    }
    MC_KV_HD static uint32_t slot_index(uint32_t pos) {
        return pos % kTokensPerPage;
    }
    // pool 总元素数：pages × 32 token × 512 元素/token × 42 层。
    // session.cpp 按 max_pages*kTokensPerPage*kKvBytesPerToken 字节分配
    //（kKvBytesPerToken = 42×2×2×128×2 = 每层每 token 512 bf16），两处
    // 口径一致；42 = cfg::kNumLayers（此处不引 runtime 头，保持 kv_cache
    // 自包含，改动层数时两处都要改）。
    uint64_t total_elems() const {
        return (uint64_t)max_pages * kTokensPerPage * kSlotElems * 42u;
    }
};

class PagedKvAllocator {
public:
    static constexpr uint32_t kTokensPerPage = KvLayout::kTokensPerPage;

    // `d_table` 指向 device 上的 page table（容量 max_pages 个 uint32，
    // 来自 session arena）。host shadow 记录同样的 page id 序列。
    mc_status_t init(uint32_t* d_table, uint32_t max_pages);

    // 确保已拥有覆盖 total_tokens 的页数；必要时从空闲池取页、追加到
    // page table 并异步更新 device 侧。仅 prefill/decode 编排期调用。
    mc_status_t ensure_pages_for(uint32_t total_tokens, cudaStream_t stream);

    // reset 语义：只清 ownership 与页计数，不释放显存、不重新分配、
    // 不触碰 device table（seq_len 归零后 table 内容由下一次 prefill 重写）。
    void release_all();

    uint32_t pages_owned() const     { return pages_owned_; }
    uint32_t max_pages() const       { return max_pages_; }
    const KvLayout& layout() const   { return layout_; }
    uint32_t tokens_capacity() const { return max_pages_ * kTokensPerPage; }

private:
    uint32_t first_free_page() const;
    bool     try_acquire(uint32_t page);

    uint32_t* d_table_      = nullptr; // device page table（arena 提供）
    uint32_t  max_pages_    = 0;
    uint32_t  pages_owned_  = 0;
    // 已同步到 device table 的前缀长度（P3：decode 期间页表通常静态——
    // 仅每 kTokensPerPage 个 token 增长一次；无增长时跳过 H2D 拷贝，
    // steady-state 的 decode 步零多余 stream 提交，对 CUDA Graph replay
    // 尤为重要）。release_all 时归零（强制下一轮 prefill 重写）。
    uint32_t  pages_synced_ = 0;
    KvLayout  layout_{};
    std::vector<uint32_t> shadow_;     // host shadow of d_table_[0..pages_owned_)
    std::vector<uint64_t> ownership_;  // 位图：bit i = page i 被本 session 占用
};
