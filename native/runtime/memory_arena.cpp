// memory_arena.cpp — DeviceArena 实现。
#include "runtime/memory_arena.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <cuda_runtime.h>

#include "runtime/error.h"

mc_status_t DeviceArena::init(size_t bytes, const char* tag) {
    destroy();
    if (bytes == 0) {
        mc::set_error("DeviceArena(%s): init with 0 bytes", tag ? tag : "?");
        return MC_E_INVALID_ARGUMENT;
    }
    void* p = nullptr;
    cudaError_t e = cudaMalloc(&p, bytes);
    if (e != cudaSuccess) {
        // 主动消费掉 sticky error，避免污染后续 cudaGetLastError 检查。
        cudaGetLastError();
        const mc_status_t st =
            (e == cudaErrorMemoryAllocation) ? MC_E_OUT_OF_MEMORY : MC_E_CUDA;
        mc::set_error("DeviceArena(%s): cudaMalloc of %zu bytes failed: %s",
                      tag ? tag : "?", bytes, cudaGetErrorString(e));
        return st;
    }
    base_ = p;
    size_ = bytes;
    offset_ = 0;
    steady_ = false;
    snprintf(tag_, sizeof(tag_), "%s", tag ? tag : "?");
    return MC_OK;
}

void* DeviceArena::alloc(size_t bytes, size_t align) {
    if (steady_) {
        // 计划 §7.2：steady-state 禁止任何新增 device 分配。这是编程错误，
        // 直接终止进程而不是静默失败（Release 下也必须生效）。
        fprintf(stderr,
                "[minicpm-native] FATAL: steady-state allocation attempt in "
                "arena '%s' (%zu bytes, plan §7.2 forbids this)\n",
                tag_, bytes);
        std::abort();
    }
    if (base_ == nullptr || bytes == 0) return nullptr;
    const size_t mask = align - 1;
    const size_t next = (offset_ + mask) & ~mask;
    if (next + bytes < next /* overflow */ || next + bytes > size_) return nullptr;
    offset_ = next + bytes;
    return static_cast<char*>(base_) + next;
}

void DeviceArena::begin_steady_state() { steady_ = true; }

void DeviceArena::destroy() {
    if (base_ != nullptr) {
        cudaError_t e = cudaFree(base_);
        if (e != cudaSuccess) {
            fprintf(stderr, "[minicpm-native] DeviceArena(%s): cudaFree failed: %s\n",
                    tag_, cudaGetErrorString(e));
            cudaGetLastError();
        }
    }
    base_ = nullptr;
    size_ = 0;
    offset_ = 0;
    steady_ = false;
    tag_[0] = '\0';
}

DeviceArena::~DeviceArena() { destroy(); }
