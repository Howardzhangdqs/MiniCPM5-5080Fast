// kv_cache.cpp — PagedKvAllocator 骨架实现。
#include "runtime/kv_cache.h"

#include <cuda_runtime.h>

#include "runtime/error.h"

mc_status_t PagedKvAllocator::init(uint32_t* d_table, uint32_t max_pages) {
    if (d_table == nullptr || max_pages == 0) {
        mc::set_error("PagedKvAllocator::init: d_table=%p max_pages=%u",
                      (const void*)d_table, max_pages);
        return MC_E_INVALID_ARGUMENT;
    }
    d_table_ = d_table;
    max_pages_ = max_pages;
    pages_owned_ = 0;
    pages_synced_ = 0;
    layout_.max_pages = max_pages; // kernels 经 layout() 读容量做护栏校验
    shadow_.assign(max_pages, 0);
    ownership_.assign((max_pages + 63) / 64, 0);
    return MC_OK;
}

uint32_t PagedKvAllocator::first_free_page() const {
    for (uint32_t w = 0; w < ownership_.size(); ++w) {
        const uint64_t bits = ownership_[w];
        if (bits != ~0ull) {
            for (uint32_t b = 0; b < 64; ++b) {
                const uint32_t page = w * 64 + b;
                if (page >= max_pages_) break;
                if ((bits & (1ull << b)) == 0) return page;
            }
        }
    }
    return max_pages_; // 无空闲页
}

bool PagedKvAllocator::try_acquire(uint32_t page) {
    if (page >= max_pages_) return false;
    const uint64_t bit = 1ull << (page & 63);
    uint64_t& word = ownership_[page >> 6];
    if (word & bit) return false;
    word |= bit;
    return true;
}

mc_status_t PagedKvAllocator::ensure_pages_for(uint32_t total_tokens, cudaStream_t stream) {
    const uint32_t needed = (total_tokens + kTokensPerPage - 1) / kTokensPerPage;
    if (needed > max_pages_) {
        mc::set_error(
            "PagedKvAllocator: KV pages exhausted (need %u pages for %u tokens, "
            "capacity %u pages = %u tokens)",
            needed, total_tokens, max_pages_, tokens_capacity());
        return MC_E_OUT_OF_MEMORY;
    }
    while (pages_owned_ < needed) {
        const uint32_t page = first_free_page();
        if (!try_acquire(page)) { // 不应发生：needed 已做上界检查
            mc::set_error("PagedKvAllocator: ownership bitmap inconsistent");
            return MC_E_INTERNAL;
        }
        shadow_[pages_owned_] = page; // 页在序列中的第 pages_owned_ 个槽位
        ++pages_owned_;
    }
    // decode 稳态（页数不变）下跳过 H2D：device table 内容已覆盖当前
    // pages_owned_ 前缀，重复提交同一份 shadow 只会增加 host/stream 开销
    //（P3：graph replay 步的 host 侧只剩页增长检查 + launch + 同步）。
    // 仅当本调用真的新增了页（pages_owned_ > pages_synced_）才拷贝。
    if (pages_owned_ <= pages_synced_) return MC_OK;
    // host shadow -> device page table（prefill/页增长期小拷贝，pageable 可
    // 接受；steady-state 无分配，只是既有 buffer 的内容更新）。
    cudaError_t e = cudaMemcpyAsync(d_table_, shadow_.data(),
                                    (size_t)pages_owned_ * sizeof(uint32_t),
                                    cudaMemcpyHostToDevice, stream);
    if (e != cudaSuccess) {
        cudaGetLastError();
        mc::set_error("PagedKvAllocator: page table H2D update failed: %s",
                      cudaGetErrorString(e));
        return MC_E_CUDA;
    }
    pages_synced_ = pages_owned_;
    return MC_OK;
}

void PagedKvAllocator::release_all() {
    std::fill(ownership_.begin(), ownership_.end(), 0);
    pages_owned_ = 0;
    pages_synced_ = 0; // 下一轮 prefill 首次 ensure 时重写 device table
    // 显式不释放、不清 device table：见头文件 reset 语义注释（P3 真实
    // KV 数据将由下一次 prefill 覆盖写）。
}
