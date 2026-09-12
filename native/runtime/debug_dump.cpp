// debug_dump.cpp — §17.2 层级 dump 实现（仅 debug；host 缓冲 lazy 一次性分配，
// debug 模式专用，不违反 steady-state 的「每步分配」——缓冲在首次 dump 时
// 分配后复用）。
#include "runtime/debug_dump.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "runtime/error.h"
#include "runtime/export.h"
#include "runtime/internal_state.h"

namespace mc::dump {

namespace {

struct DumpConfig {
    bool on = false;
    int steps = 0;
    bool seed_out = false;
    char dir_[512] = {0};
};

const DumpConfig& config() {
    static DumpConfig cfg = [] {
        DumpConfig c;
        const char* d = getenv("MC_DEBUG_DUMP_DIR");
        const char* s = getenv("MC_DEBUG_DUMP_STEPS");
        if (d != nullptr && d[0] != '\0') {
            c.on = true;
            snprintf(c.dir_, sizeof(c.dir_), "%s", d);
            c.steps = s != nullptr ? atoi(s) : 0;
            const char* w = getenv("MC_DEBUG_DUMP_WINDOW_SEED");
            c.seed_out = (w != nullptr && atoi(w) != 0);
            fprintf(stderr,
                    "[minicpm-native] debug dump enabled: dir=%s steps=%d "
                    "window_seed=%d (debug-only, may slow inference)\n",
                    c.dir_, c.steps, (int)c.seed_out);
        }
        return c;
    }();
    return cfg;
}

// 复用的 host 缓冲（debug 模式首次使用时分配；进程退出时释放，量级 ≤ 1MB 级）
std::vector<uint16_t>& host_bf16_buffer(size_t need) {
    static std::vector<uint16_t> buf;
    if (buf.size() < need) buf.resize(need);
    return buf;
}

} // namespace

bool enabled() { return config().on; }
int max_steps() { return config().on ? config().steps : 0; }
const char* dir() { return config().dir_; }
bool seed_out() { return config().on && config().seed_out; }

void set_steps(int n) {
    // 修改静态缓存的步数上限（编排期专用；n=0 关闭后续步的 dump）。
    const_cast<DumpConfig&>(config()).steps = n;
}

mc_status_t write_bf16_as_f32(const char* path, const uint16_t* d_data, size_t count,
                              cudaStream_t stream) {
    auto& buf = host_bf16_buffer(count);
    cudaError_t e = cudaMemcpyAsync(buf.data(), d_data, count * sizeof(uint16_t),
                                    cudaMemcpyDeviceToHost, stream);
    if (e != cudaSuccess) {
        cudaGetLastError();
        mc::set_error("dump: D2H failed for %s: %s", path, cudaGetErrorString(e));
        return MC_E_CUDA;
    }
    e = cudaStreamSynchronize(stream);
    if (e != cudaSuccess) {
        cudaGetLastError();
        mc::set_error("dump: sync failed for %s: %s", path, cudaGetErrorString(e));
        return MC_E_CUDA;
    }
    std::vector<float> f(count); // debug 路径的小分配可接受（不进 Release hot path）
    for (size_t i = 0; i < count; ++i) {
        const uint32_t u = (uint32_t)buf[i] << 16; // bf16 → fp32 位拼接
        std::memcpy(&f[i], &u, sizeof(float));
    }
    FILE* fp = fopen(path, "wb");
    if (fp == nullptr) {
        mc::set_error("dump: cannot open %s", path);
        return MC_E_IO;
    }
    const size_t n = fwrite(f.data(), sizeof(float), count, fp);
    fclose(fp);
    if (n != count) {
        mc::set_error("dump: short write for %s", path);
        return MC_E_IO;
    }
    return MC_OK;
}

mc_status_t write_token(const char* path, int32_t token) {
    FILE* fp = fopen(path, "wb");
    if (fp == nullptr) {
        mc::set_error("dump: cannot open %s", path);
        return MC_E_IO;
    }
    const size_t n = fwrite(&token, sizeof(int32_t), 1, fp);
    fclose(fp);
    if (n != 1) {
        mc::set_error("dump: short write for %s", path);
        return MC_E_IO;
    }
    return MC_OK;
}

mc_status_t write_bf16_raw(const char* path, const uint16_t* d_data, size_t count,
                           cudaStream_t stream) {
    auto& buf = host_bf16_buffer(count);
    cudaError_t e = cudaMemcpyAsync(buf.data(), d_data, count * sizeof(uint16_t),
                                    cudaMemcpyDeviceToHost, stream);
    if (e != cudaSuccess) {
        cudaGetLastError();
        mc::set_error("dump: D2H failed for %s: %s", path, cudaGetErrorString(e));
        return MC_E_CUDA;
    }
    e = cudaStreamSynchronize(stream);
    if (e != cudaSuccess) {
        cudaGetLastError();
        mc::set_error("dump: sync failed for %s: %s", path, cudaGetErrorString(e));
        return MC_E_CUDA;
    }
    FILE* fp = fopen(path, "wb");
    if (fp == nullptr) {
        mc::set_error("dump: cannot open %s", path);
        return MC_E_IO;
    }
    const size_t n = fwrite(buf.data(), sizeof(uint16_t), count, fp);
    fclose(fp);
    if (n != count) {
        mc::set_error("dump: short write for %s", path);
        return MC_E_IO;
    }
    return MC_OK;
}

} // namespace mc::dump

// C 导出（MC_API default visibility）：编排进程（real_inference 等）经 .so
// 调用的 additive debug 钩子（不属于冻结 ABI 契约）。
extern "C" MC_API void mc_debug_dump_set_steps(int steps) {
    mc::dump::set_steps(steps);
}

// 分窗种子钩子（gpu_layerwise）：host_bf16_rows（rows×hidden bf16，行主序）
// → session residual 流。REAL 模式专用；调用后紧接 mc_prefill（其 embedding
// gather 因 embedding 未加载而跳过，层循环直接消费种子）。
extern "C" MC_API mc_status_t mc_debug_seed_residual(mc_session_t session,
                                                     const void* host_bf16_rows,
                                                     uint32_t rows) {
    try {
        mc::clear_error();
        if (session.impl == nullptr || host_bf16_rows == nullptr) {
            mc::set_error("mc_debug_seed_residual: session/rows pointer is NULL");
            return MC_E_INVALID_ARGUMENT;
        }
        SessionImpl* s = session.impl;
        if (s->model == nullptr || !s->model->has_wpk || s->act.residual == nullptr) {
            mc::set_error("mc_debug_seed_residual: requires a REAL-mode session "
                          "(wpk loaded)");
            return MC_E_UNSUPPORTED;
        }
        if (rows == 0 || rows > s->model->max_context_tokens) {
            mc::set_error("mc_debug_seed_residual: rows=%u out of range (1..%u)", rows,
                          s->model->max_context_tokens);
            return MC_E_INVALID_ARGUMENT;
        }
        cudaError_t e = cudaSetDevice(s->model->device_id);
        if (e != cudaSuccess) {
            cudaGetLastError();
            mc::set_error("mc_debug_seed_residual: cudaSetDevice failed: %s",
                          cudaGetErrorString(e));
            return MC_E_CUDA;
        }
        e = cudaMemcpy(s->act.residual, host_bf16_rows,
                       (size_t)rows * mc::cfg::kHiddenSize * sizeof(uint16_t),
                       cudaMemcpyHostToDevice);
        if (e != cudaSuccess) {
            cudaGetLastError();
            mc::set_error("mc_debug_seed_residual: H2D copy failed: %s",
                          cudaGetErrorString(e));
            return MC_E_CUDA;
        }
        return MC_OK;
    } catch (...) {
        mc::set_error("mc_debug_seed_residual: internal error (uncaught exception)");
        return MC_E_INTERNAL;
    }
}
