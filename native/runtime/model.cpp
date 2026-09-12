// model.cpp — mc_model_load / mc_model_destroy 实现（计划 §5.4、§7.2）。
//
// mc_model_load 依次完成：options/ABI 校验 → device 选择与 SM120 校验 →
// 初始 VRAM 记录 → model_dir 扫描（wpk 存在性 → mock/real 判定）→
// 共享 workspace 一次性分配 → reserve headroom 检查。
// 失败时 *out_model 保持 nullptr。load 成功后 model 不可变，可被多
// session / 多线程共享。
#include "runtime/internal_state.h"

#include <cstdio>
#include <new>

#include <cuda_runtime.h>

#include "runtime/error.h"
#include "runtime/export.h"

namespace {

// mock 阶段的共享 workspace（计划 §7.2 "Shared Runtime Workspace" 段；
// P2 权重上传后由 WeightArena 扩充，纪律不变：load 期一次性分配）。
constexpr size_t kSharedWorkspaceBytes = 16u << 20; // 16 MiB

} // namespace

mc_status_t ModelImpl::sample_free_vram(uint64_t& free_out) const {
    size_t f = 0, t = 0;
    cudaError_t e = cudaMemGetInfo(&f, &t);
    if (e != cudaSuccess) {
        cudaGetLastError();
        mc::set_error("cudaMemGetInfo failed: %s", cudaGetErrorString(e));
        return MC_E_CUDA;
    }
    free_out = (uint64_t)f;
    return MC_OK;
}

extern "C" MC_API mc_status_t mc_model_load(const char* model_dir,
                                            const mc_model_options_t* options,
                                            mc_model_t* out_model) {
    try {
        mc::clear_error();
        if (out_model != nullptr) *out_model = mc_model{nullptr};

        // ---- 1) 参数与 ABI 校验 ----
        if (model_dir == nullptr || model_dir[0] == '\0') {
            mc::set_error("mc_model_load: model_dir must be a non-empty path");
            return MC_E_INVALID_ARGUMENT;
        }
        if (options == nullptr) {
            mc::set_error("mc_model_load: options is NULL");
            return MC_E_INVALID_ARGUMENT;
        }
        if (out_model == nullptr) {
            mc::set_error("mc_model_load: out_model is NULL");
            return MC_E_INVALID_ARGUMENT;
        }
        if (options->struct_size != sizeof(*options)) {
            mc::set_error("mc_model_load: options->struct_size=%u but expected %zu",
                          options->struct_size, sizeof(*options));
            return MC_E_INVALID_ARGUMENT;
        }
        if (options->abi_version != MC_ABI_VERSION) {
            mc::set_error("mc_model_load: options->abi_version=0x%08x but expected 0x%08x",
                          options->abi_version, MC_ABI_VERSION);
            return MC_E_INVALID_ARGUMENT;
        }
        if (options->precision != MC_BF16 && options->precision != MC_FP8) {
            mc::set_error("mc_model_load: only MC_BF16 and MC_FP8 (P4 dual-copy: "
                          "BF16 prefill + FP8 decode) are supported "
                          "(options->precision=%d)",
                          (int)options->precision);
            return MC_E_UNSUPPORTED;
        }
        if (options->kv_precision != MC_KV_BF16) {
            mc::set_error("mc_model_load: only MC_KV_BF16 kv precision is supported in "
                          "this phase (options->kv_precision=%d)",
                          (int)options->kv_precision);
            return MC_E_UNSUPPORTED;
        }
        // P3：enable_cuda_graph=1 已落地（计划 §12/§20-P3）。真实权重
        //（model.wpk）session 在 create 尾部 capture decode graph；capture
        // 失败自动回退 eager（session 不失效）。mock session 恒 eager。
        if (options->device_id < 0) {
            mc::set_error("mc_model_load: device_id must be >= 0 (got %d)",
                          options->device_id);
            return MC_E_INVALID_ARGUMENT;
        }
        if (options->max_sessions == 0) {
            mc::set_error("mc_model_load: max_sessions must be >= 1");
            return MC_E_INVALID_ARGUMENT;
        }
        if (options->max_context_tokens == 0) {
            mc::set_error("mc_model_load: max_context_tokens must be >= 1");
            return MC_E_INVALID_ARGUMENT;
        }
        if (options->max_context_tokens > mc::cfg::kMaxContextTokens) {
            mc::set_error("mc_model_load: max_context_tokens=%u exceeds model limit %u "
                          "(MiniCPM5-2B max_position_embeddings)",
                          options->max_context_tokens, mc::cfg::kMaxContextTokens);
            return MC_E_UNSUPPORTED;
        }

        // ---- 2) device 选择 + compute capability 校验（期望 SM120 == 12.0）----
        int device_count = 0;
        cudaError_t e = cudaGetDeviceCount(&device_count);
        if (e != cudaSuccess) {
            cudaGetLastError();
            mc::set_error("mc_model_load: cudaGetDeviceCount failed: %s",
                          cudaGetErrorString(e));
            return MC_E_CUDA;
        }
        if (options->device_id >= device_count) {
            mc::set_error("mc_model_load: device_id=%d out of range (device count=%d)",
                          options->device_id, device_count);
            return MC_E_INVALID_ARGUMENT;
        }
        e = cudaSetDevice(options->device_id);
        if (e != cudaSuccess) {
            cudaGetLastError();
            mc::set_error("mc_model_load: cudaSetDevice(%d) failed: %s",
                          options->device_id, cudaGetErrorString(e));
            return MC_E_CUDA;
        }
        cudaDeviceProp props{};
        e = cudaGetDeviceProperties(&props, options->device_id);
        if (e != cudaSuccess) {
            cudaGetLastError();
            mc::set_error("mc_model_load: cudaGetDeviceProperties failed: %s",
                          cudaGetErrorString(e));
            return MC_E_CUDA;
        }
        if (props.major != 12 || props.minor != 0) {
            mc::set_error("mc_model_load: device %d ('%s') reports compute capability "
                          "%d.%d, but this runtime requires SM120 (==12.0) for "
                          "RTX 5080 / Blackwell GeForce",
                          options->device_id, props.name, props.major, props.minor);
            return MC_E_MODEL_MISMATCH;
        }

        // ---- 3) 初始 VRAM 记录（不假设标称 16GB 全部可用，计划 §7.2）----
        uint64_t free_b = 0, total_b = 0;
        {
            size_t f = 0, t = 0;
            e = cudaMemGetInfo(&f, &t);
            if (e != cudaSuccess) {
                cudaGetLastError();
                mc::set_error("mc_model_load: cudaMemGetInfo failed: %s",
                              cudaGetErrorString(e));
                return MC_E_CUDA;
            }
            free_b = f;
            total_b = t;
        }

        // ---- 4) model_dir 扫描：model.wpk 存在 → 真实 BF16 权重；缺失 → mock ----
        WeightScanResult scan;
        mc_status_t rs = WeightLoader::scan(model_dir, scan);
        if (rs != MC_OK) return rs;

        // ---- 5) 构造不可变 model ----
        ModelImpl* m = new ModelImpl();
        m->device_id = options->device_id;
        m->max_sessions = options->max_sessions;
        m->max_context_tokens = options->max_context_tokens;
        m->reserve_vram_bytes = options->reserve_vram_bytes;
        m->model_dir = model_dir;
        m->has_wpk = scan.has_wpk;
        m->enable_cuda_graph = (options->enable_cuda_graph != 0);
        m->weight_scan = std::move(scan);
        m->initial_free_vram = free_b;
        m->total_vram = total_b;

        // 共享 workspace（gemm workspace 复用；16MB，BF16 heuristic 以此为上限）
        rs = m->workspace.init(kSharedWorkspaceBytes, "model.shared_workspace");
        if (rs != MC_OK) {
            delete m;
            return rs;
        }
        m->gemm_ws = m->workspace.base_ptr();
        m->gemm_ws_bytes = kSharedWorkspaceBytes;

        // ---- P2：真实权重加载 + cuBLASLt（wpk 缺失则保持 mock，日志明示）----
        if (m->has_wpk) {
            rs = WeightLoader::load_wpk(model_dir, m->device_id, m->weight_arena,
                                        m->weights, m->weight_bytes);
            if (rs != MC_OK) {
                delete m;
                return rs;
            }
            rs = m->gemm.init();
            if (rs != MC_OK) {
                delete m;
                return rs;
            }
        }

        // ---- P4：FP8 双拷贝（precision=MC_FP8）----
        // decode 走 FP8（168 个量化 section + scales），prefill 走 BF16 原权
        //（上面 load_wpk 已上传）。缺包（含 mock 目录无 wpk 的情况）→
        // MC_E_MODEL_MISMATCH（fp8 无 mock 语义，不静默降级）；显存不足 →
        // 显式 MC_E_OUT_OF_MEMORY。
        if (options->precision == MC_FP8) {
            if (!m->has_wpk || !m->weight_scan.has_fp8_wpk) {
                mc::set_error("mc_model_load: precision=MC_FP8 requires "
                              "model_fp8.wpk (dtype=2 sections + fp32 row scales) "
                              "and model.wpk (BF16 originals for prefill) under "
                              "%s — package(s) not found (toolchain lane "
                              "produces model_fp8.wpk; see plan §13)",
                              model_dir);
                delete m;
                return MC_E_MODEL_MISMATCH;
            }
            // 显存预检：fp8 量化段 ~1.98GB + scales ~3MB + 余量。fp8_arena 的
            // cudaMalloc 失败同样显式报 OOM（双保险）。
            {
                uint64_t free_now = 0;
                rs = m->sample_free_vram(free_now);
                if (rs != MC_OK) {
                    delete m;
                    return rs;
                }
                const uint64_t fp8_need =
                    m->weight_scan.fp8_wpk_bytes + (64ull << 20) /*粗略余量*/;
                if (free_now < fp8_need) {
                    mc::set_error("mc_model_load: FP8 dual-copy needs ~%llu bytes for "
                                  "model_fp8.wpk but free VRAM is %llu bytes — "
                                  "MC_E_OUT_OF_MEMORY (no silent fallback)",
                                  (unsigned long long)fp8_need,
                                  (unsigned long long)free_now);
                    delete m;
                    return MC_E_OUT_OF_MEMORY;
                }
            }
            uint64_t fp8_bytes = 0;
            rs = WeightLoader::load_wpk_fp8(model_dir, m->device_id, m->fp8_arena,
                                            m->weights, fp8_bytes);
            if (rs != MC_OK) {
                delete m;
                return rs;
            }
        }
        m->workspace.begin_steady_state(); // 模型期分配到此为止
        m->weight_arena.begin_steady_state();

        uint64_t free_after = 0;
        rs = m->sample_free_vram(free_after);
        if (rs != MC_OK) {
            delete m;
            return rs;
        }
        if (m->reserve_vram_bytes != 0 && free_after < m->reserve_vram_bytes) {
            mc::set_error("mc_model_load: free VRAM %llu bytes is below "
                          "reserve_vram_bytes %llu",
                          (unsigned long long)free_after,
                          (unsigned long long)m->reserve_vram_bytes);
            delete m;
            return MC_E_OUT_OF_MEMORY;
        }

        // ---- 6) 日志（英文，便于检索）----
        if (m->has_wpk) {
            fprintf(stderr,
                    "[minicpm-native] model.wpk loaded (%llu bytes on device) under "
                    "%s; real BF16 forward path enabled\n",
                    (unsigned long long)m->weight_bytes, model_dir);
            if (m->weights.has_fp8) {
                const uint64_t scale_bytes =
                    (uint64_t)mc::cfg::kNumLayers *
                    (mc::cfg::kNumQueryHeads * mc::cfg::kHeadDim +
                     2 * mc::cfg::kNumKvHeads * mc::cfg::kHeadDim +
                     2 * mc::cfg::kHiddenSize + 2 * mc::cfg::kIntermediateSize) *
                    4ull;
                fprintf(stderr,
                        "[minicpm-native] FP8 dual-copy enabled: BF16 originals "
                        "%llu B (prefill) + FP8 quantized+scales %llu B (decode, "
                        "incl. %llu B scales) = %llu B weights total\n",
                        (unsigned long long)m->weight_bytes,
                        (unsigned long long)m->fp8_arena.capacity(),
                        (unsigned long long)scale_bytes,
                        (unsigned long long)(m->weight_bytes +
                                             m->fp8_arena.capacity()));
            }
        } else {
            fprintf(stderr,
                    "[minicpm-native] model.wpk NOT found under %s (%zu files scanned) "
                    "-- mock forward mode (deterministic kernels; P2 loads real "
                    "weights)\n",
                    model_dir, m->weight_scan.files.size());
        }
        fprintf(stderr,
                "[minicpm-native] model_load: device=%d '%s' cc=%d.%d vram_free=%.2fGiB "
                "of %.2fGiB max_ctx=%u precision=%s forward=%s weights=%lluB%s "
                "cuda_graph=%s\n",
                m->device_id, props.name, props.major, props.minor,
                (double)free_after / (1024.0 * 1024.0 * 1024.0),
                (double)total_b / (1024.0 * 1024.0 * 1024.0), m->max_context_tokens,
                options->precision == MC_FP8 ? "FP8(dual-copy)"
                                             : "BF16",
                m->has_wpk ? "REAL" : "MOCK", (unsigned long long)m->weight_bytes,
                m->weights.has_fp8
                    ? (std::string("+") + std::to_string(m->fp8_arena.capacity()) +
                       "B fp8")
                          .c_str()
                    : "",
                m->enable_cuda_graph ? "on" : "off");

        *out_model = mc_model{m}; // handle 包装 impl 指针
        return MC_OK;
    } catch (...) {
        // 绝不让异常跨越 ABI（§5.1）。
        mc::set_error("mc_model_load: internal error (uncaught exception)");
        return MC_E_INTERNAL;
    }
}

extern "C" MC_API void mc_model_destroy(mc_model_t model) {
    ModelImpl* m = model.impl;
    if (m == nullptr) return;
    try {
        cudaSetDevice(m->device_id);
        m->gemm.destroy();          // cuBLASLt handle + 计划缓存
        m->fp8_arena.destroy();     // P4：fp8 量化权重 cudaFree（空则 no-op）
        m->weight_arena.destroy();  // 权重 cudaFree
        m->workspace.destroy();     // 共享 workspace cudaFree
    } catch (...) {
        // destroy 不抛错；泄漏优先于崩溃。
    }
    delete m;
}
