// memory_arena.h — DeviceArena：初始化期一次性 cudaMalloc 大块 + bump 分配。
//
// 显存纪律（计划 §7.2）：steady-state 禁止 cudaMalloc/cudaFree/cudaHostAlloc。
// 因此所有 device 显存的分配只发生在 model_load / session_create 期：
//   - init()   ：一次 cudaMalloc（构造期，唯一一次真实分配）
//   - alloc()  ：纯 host 端 bump 指针推进，零 CUDA 调用
//   - begin_steady_state() 之后任何 alloc() 都是纪律违规，直接 abort
//     （用 abort 而不是 assert：Release -DNDEBUG 下 assert 会被编译掉）
#pragma once

#include <cstddef>
#include <cstdint>

#include "minicpm_runtime.h"

class DeviceArena {
public:
    DeviceArena() = default;
    DeviceArena(const DeviceArena&) = delete;
    DeviceArena& operator=(const DeviceArena&) = delete;

    // 一次性 cudaMalloc `bytes` 字节。tag 仅用于错误信息/日志。
    // cudaErrorMemoryAllocation -> MC_E_OUT_OF_MEMORY，其余 -> MC_E_CUDA。
    mc_status_t init(size_t bytes, const char* tag);

    // bump 分配（默认 256B 对齐）。越界返回 nullptr（由调用方报
    // MC_E_OUT_OF_MEMORY）；steady-state 之后调用则直接 abort。
    void* alloc(size_t bytes, size_t align = 256);

    template <typename T>
    T* alloc(size_t count) {
        constexpr size_t a = alignof(T) > 256 ? alignof(T) : 256;
        return static_cast<T*>(alloc(count * sizeof(T), a));
    }

    // 进入 steady-state：此后 alloc() 一律视为违反 §7.2 纪律。
    void begin_steady_state();

    bool   steady_state() const { return steady_; }
    size_t used() const         { return offset_; }
    size_t capacity() const     { return size_; }
    void*  base_ptr() const     { return base_; } // 权重段偏移寻址用

    // cudaFree 并复位（幂等）。调用前需保证 current device 正确。
    void destroy();

    ~DeviceArena();

private:
    void*  base_   = nullptr;
    size_t size_   = 0;
    size_t offset_ = 0;
    bool   steady_ = false;
    char   tag_[64] = {0};
};
