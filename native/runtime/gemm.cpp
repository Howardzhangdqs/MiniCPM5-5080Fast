// gemm.cpp — cuBLASLt BF16 GEMM 封装实现（F2a 起含 FP8 W8A8 变体）。
#include "runtime/gemm.h"

#include <cstdio>
#include <mutex>
#include <unordered_map>

#include "kernels/sm120/kernels.h" // launch_quant_rows / launch_post_scale_rows

#include "runtime/error.h"
#include "runtime/generated_model_config.h"

// F2a 双头兼容：host CUDA 12.8 的 cublasLt.h 无 OUTER_VEC_32F 枚举
//（仅 SCALAR/VEC16/VEC32），venv cu13 头有（值 3）。编译期不分支——
// 统一定义同值枚举，支持性由运行时探测（属性设置/heuristic 失败即回退
// per-tensor），两种头/库组合下行为一致。
#ifndef CUBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F
#define CUBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F ((cublasLtMatmulMatrixScale_t)3)
#endif

namespace mc {

struct GemmPlan {
    cublasLtMatmulDesc_t op = nullptr;
    cublasLtMatrixLayout_t la = nullptr; // A 存储布局 [K,N] col ld=K（权重）
    cublasLtMatrixLayout_t lb = nullptr; // B 存储布局 [K,M] col ld=K（activation）
    cublasLtMatrixLayout_t lc = nullptr; // C 存储布局 [N,M] col ld=N
    cublasLtMatmulAlgo_t algo{};
    bool algo_valid = false; // 首次执行选定后缓存（decode M=1 全命中）
};

// F2a：FP8 W8A8 计划。A/B = e4m3、D = bf16、COMPUTE_32F；scale 走
// A/B_SCALE_POINTER（per-call 设置，见 gemm_fp8_q 注释）。
struct GemmFp8Plan {
    cublasLtMatmulDesc_t op = nullptr;
    cublasLtMatrixLayout_t la = nullptr; // A [K,N] col ld=K，e4m3（权重）
    cublasLtMatrixLayout_t lb = nullptr; // B [K,M] col ld=K，e4m3（激活）
    cublasLtMatrixLayout_t ld = nullptr; // D [N,M] col ld=N，bf16
    cublasLtMatmulAlgo_t algo{};
    bool algo_valid = false;
    bool outer_vec = false; // true：OUTER_VEC_32F；false：per-tensor+post kernel
};

struct GemmEngine::Impl {
    std::mutex mu;
    std::unordered_map<uint64_t, GemmPlan> cache;
    std::unordered_map<uint64_t, GemmFp8Plan> fp8_cache; // F2a：与 bf16 分表
    int fp8_mode = 0;  // 0=未探测 1=outer_vec 2=per-tensor（建期探测缓存）
    float* d_one = nullptr; // per-tensor 模式的 device 常量 1.0f（scale 指针）
    static uint64_t make_key(uint32_t M, uint32_t N, uint32_t K) {
        return ((uint64_t)M << 44) ^ ((uint64_t)N << 22) ^ (uint64_t)K;
    }
};

namespace {

constexpr size_t kMaxCachedPlans = 256; // 超过后 prefill 变 shape 走未缓存路径

void destroy_plan(GemmPlan& p) {
    if (p.op != nullptr) cublasLtMatmulDescDestroy(p.op);
    if (p.la != nullptr) cublasLtMatrixLayoutDestroy(p.la);
    if (p.lb != nullptr) cublasLtMatrixLayoutDestroy(p.lb);
    if (p.lc != nullptr) cublasLtMatrixLayoutDestroy(p.lc);
    p = GemmPlan{};
}

mc_status_t build_plan(uint32_t M, uint32_t N, uint32_t K, GemmPlan& p) {
    cublasStatus_t st;
    st = cublasLtMatmulDescCreate(&p.op, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    if (st != CUBLAS_STATUS_SUCCESS) goto fail;
    {
        const cublasOperation_t ta = CUBLAS_OP_T; // 权重 [K,N] → [N,K]
        const cublasOperation_t tb = CUBLAS_OP_N; // activation [K,M]
        st = cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSA, &ta,
                                            sizeof(ta));
        if (st != CUBLAS_STATUS_SUCCESS) goto fail;
        st = cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSB, &tb,
                                            sizeof(tb));
        if (st != CUBLAS_STATUS_SUCCESS) goto fail;
    }
    st = cublasLtMatrixLayoutCreate(&p.la, CUDA_R_16BF, K, N, K);
    if (st != CUBLAS_STATUS_SUCCESS) goto fail;
    st = cublasLtMatrixLayoutCreate(&p.lb, CUDA_R_16BF, K, M, K);
    if (st != CUBLAS_STATUS_SUCCESS) goto fail;
    st = cublasLtMatrixLayoutCreate(&p.lc, CUDA_R_16BF, N, M, N);
    if (st != CUBLAS_STATUS_SUCCESS) goto fail;
    return MC_OK;
fail:
    mc::set_error("gemm_bf16: cublasLt desc/layout create failed (status=%d, M=%u N=%u "
                  "K=%u)",
                  (int)st, M, N, K);
    destroy_plan(p);
    return MC_E_CUDA;
}

// 查询 heuristic：以 ws_size 为上限过滤，返回可用 algo。
bool pick_algo(cublasLtHandle_t h, const GemmPlan& p, size_t ws_size,
               cublasLtMatmulAlgo_t* out) {
    cublasLtMatmulPreference_t pref = nullptr;
    cublasLtMatmulPreferenceCreate(&pref);
    cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                         &ws_size, sizeof(ws_size));
    cublasLtMatmulHeuristicResult_t heur[4];
    int returned = 0;
    cublasStatus_t st = cublasLtMatmulAlgoGetHeuristic(h, p.op, p.la, p.lb, p.lc,
                                                       p.lc /*C 与 D 同布局*/,
                                                       pref, 4, heur, &returned);
    cublasLtMatmulPreferenceDestroy(pref);
    if (st != CUBLAS_STATUS_SUCCESS || returned == 0) return false;
    *out = heur[0].algo;
    return true;
}

inline mc_status_t run_matmul(cublasLtHandle_t h, const GemmPlan& p, const void* d_a,
                              const void* d_b, void* d_c, void* workspace,
                              size_t ws_size, cudaStream_t stream, uint32_t M,
                              uint32_t N, uint32_t K) {
    const float alpha = 1.0f, beta = 0.0f;
    cublasStatus_t st = cublasLtMatmul(h, p.op, &alpha, d_a, p.la, d_b, p.lb, &beta,
                                       d_c, p.lc, d_c, p.lc, &p.algo, workspace,
                                       ws_size, stream);
    if (st != CUBLAS_STATUS_SUCCESS) {
        mc::set_error("gemm_bf16: cublasLtMatmul failed (M=%u N=%u K=%u, status=%d)", M,
                      N, K, (int)st);
        return MC_E_CUDA;
    }
    return MC_OK;
}

// ---- F2a：FP8 W8A8 ----

void destroy_fp8_plan(GemmFp8Plan& p) {
    if (p.op != nullptr) cublasLtMatmulDescDestroy(p.op);
    if (p.la != nullptr) cublasLtMatrixLayoutDestroy(p.la);
    if (p.lb != nullptr) cublasLtMatrixLayoutDestroy(p.lb);
    if (p.ld != nullptr) cublasLtMatrixLayoutDestroy(p.ld);
    p = GemmFp8Plan{};
}

// 列主映射同 bf16：cublasLtMatmul(opA=T, opB=N, m=N, n=M, k=K)：
//   A := W_col[K,N] ld=K e4m3（= 权重行主 [N,K]），OP_T → [N,K]
//   B := x_col[K,M] ld=K e4m3（= 激活行主 [M,K]），OP_N → [K,M]
//   D := [N,M] col ld=N bf16 ≡ 行主 [M,N]
// outer_vec=true 时设 A/B scale mode = OUTER_VEC_32F（w_scale[n]×x_scale[m]
// 外积缩放核内完成）；false 时 per-tensor scalar（scale 指针 → d_one=1.0，
// 原始 fp8 点积直出，由 post_scale_rows 补偿）。
mc_status_t build_fp8_plan(uint32_t M, uint32_t N, uint32_t K, bool outer_vec,
                           GemmFp8Plan& p) {
    cublasStatus_t st;
    st = cublasLtMatmulDescCreate(&p.op, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    if (st != CUBLAS_STATUS_SUCCESS) goto fail;
    {
        const cublasOperation_t ta = CUBLAS_OP_T;
        const cublasOperation_t tb = CUBLAS_OP_N;
        st = cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSA, &ta,
                                            sizeof(ta));
        if (st != CUBLAS_STATUS_SUCCESS) goto fail;
        st = cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSB, &tb,
                                            sizeof(tb));
        if (st != CUBLAS_STATUS_SUCCESS) goto fail;
        const cublasLtMatmulMatrixScale_t mode = outer_vec
            ? CUBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F
            : CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F;
        st = cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_A_SCALE_MODE,
                                            &mode, sizeof(mode));
        if (st != CUBLAS_STATUS_SUCCESS) goto fail;
        st = cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_B_SCALE_MODE,
                                            &mode, sizeof(mode));
        if (st != CUBLAS_STATUS_SUCCESS) goto fail;
    }
    st = cublasLtMatrixLayoutCreate(&p.la, CUDA_R_8F_E4M3, K, N, K);
    if (st != CUBLAS_STATUS_SUCCESS) goto fail;
    st = cublasLtMatrixLayoutCreate(&p.lb, CUDA_R_8F_E4M3, K, M, K);
    if (st != CUBLAS_STATUS_SUCCESS) goto fail;
    st = cublasLtMatrixLayoutCreate(&p.ld, CUDA_R_16BF, N, M, N);
    if (st != CUBLAS_STATUS_SUCCESS) goto fail;
    p.outer_vec = outer_vec;
    return MC_OK;
fail:
    mc::set_error("gemm_fp8: cublasLt desc/layout create failed (status=%d, mode=%s)",
                  (int)st, outer_vec ? "outer_vec" : "scalar");
    destroy_fp8_plan(p);
    return MC_E_CUDA;
}

// fp8 heuristic（与 bf16 pick_algo 同式，D 布局同时作 C 布局）
bool pick_algo_fp8(cublasLtHandle_t h, const GemmFp8Plan& p, size_t ws_size,
                   cublasLtMatmulAlgo_t* out) {
    cublasLtMatmulPreference_t pref = nullptr;
    cublasLtMatmulPreferenceCreate(&pref);
    cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                         &ws_size, sizeof(ws_size));
    cublasLtMatmulHeuristicResult_t heur[4];
    int returned = 0;
    cublasStatus_t st = cublasLtMatmulAlgoGetHeuristic(h, p.op, p.la, p.lb, p.ld,
                                                       p.ld, pref, 4, heur, &returned);
    cublasLtMatmulPreferenceDestroy(pref);
    if (st != CUBLAS_STATUS_SUCCESS || returned == 0) return false;
    *out = heur[0].algo;
    return true;
}

// FP8 matmul：alpha/beta 必须 nullptr（缩放经 A/B scale 指针，cuBLASLt
// FP8 契约）；scale 指针 per-call 变化 → 在 impl 锁内设置并发射（引擎
// 单 session 串行口径同文件头；多线程并发 fp8 调用需外部串行化）。
mc_status_t run_matmul_fp8(cublasLtHandle_t h, GemmFp8Plan& p, const void* d_a,
                           const void* d_b, void* d_d, const float* d_a_scale,
                           const float* d_b_scale, void* workspace, size_t ws_size,
                           cudaStream_t stream, uint32_t M, uint32_t N, uint32_t K) {
    cublasStatus_t st;
    // FP8 契约（官方 cublasLtFp8Matmul sample 同式）：scale 经 A/B 指针；
    // alpha/beta 传宿主 1.0/0.0（scale 已吸收缩放，alpha 不再折算）
    const float alpha = 1.0f, beta = 0.0f;
    st = cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
                                        &d_a_scale, sizeof(d_a_scale));
    if (st != CUBLAS_STATUS_SUCCESS) goto fail;
    st = cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
                                        &d_b_scale, sizeof(d_b_scale));
    if (st != CUBLAS_STATUS_SUCCESS) goto fail;
    st = cublasLtMatmul(h, p.op, &alpha, d_a, p.la, d_b, p.lb, &beta, d_d, p.ld,
                        d_d, p.ld, &p.algo, workspace, ws_size, stream);
    if (st != CUBLAS_STATUS_SUCCESS) goto fail;
    return MC_OK;
fail:
    mc::set_error("gemm_fp8: cublasLtMatmul failed (M=%u N=%u K=%u, status=%d)", M,
                  N, K, (int)st);
    return MC_E_CUDA;
}

inline bool aligned16(const void* p) {
    return (reinterpret_cast<uintptr_t>(p) & 15u) == 0u;
}

} // namespace

mc_status_t GemmEngine::init() {
    if (handle_ != nullptr) return MC_OK;
    impl_ = new Impl();
    cublasStatus_t st = cublasLtCreate(&handle_);
    if (st != CUBLAS_STATUS_SUCCESS) {
        delete impl_;
        impl_ = nullptr;
        mc::set_error("GemmEngine: cublasLtCreate failed (status=%d)", (int)st);
        return MC_E_CUDA;
    }
    return MC_OK;
}

void GemmEngine::destroy() {
    if (impl_ != nullptr) {
        // F2 根因修复：原实现把 `delete impl_` 放在 lock_guard 作用域内，
        // guard 析构（作用域末尾）会对「已 delete 的 mutex」执行 unlock ——
        // pthread_mutex_unlock 写入已释放堆块（UB），砸毁 glibc fastbin/tcache
        // 元数据，表现为 destroy/退出期 malloc_consolidate abort 或后续
        // malloc 随机 SIGSEGV（kernel_ops gemm case 捕获；去锁二分 100% 复现）。
        // 修法：先在锁内销毁计划缓存，锁出作用域后再 delete impl_。
        {
            std::lock_guard<std::mutex> lk(impl_->mu);
            for (auto& kv : impl_->cache) destroy_plan(kv.second);
            for (auto& kv : impl_->fp8_cache) destroy_fp8_plan(kv.second); // F2a
        } // lock_guard 在此解锁（impl_ 仍有效）
        if (impl_->d_one != nullptr) {
            (void)cudaFree(impl_->d_one); // F2a：per-tensor scale 常量
            impl_->d_one = nullptr;
        }
        delete impl_;
        impl_ = nullptr;
    }
    if (handle_ != nullptr) {
        cublasLtDestroy(handle_);
        handle_ = nullptr;
    }
}

size_t GemmEngine::cache_size() const {
    return impl_ != nullptr ? impl_->cache.size() : 0;
}

mc_status_t GemmEngine::prewarm_decode_shapes(size_t ws_size) {
    if (handle_ == nullptr) {
        mc::set_error("prewarm_decode_shapes: engine not initialized");
        return MC_E_INTERNAL;
    }
    // decode 稳态的 5 个固定 GEMM（M=1；N,K 来自 generated_model_config）：
    //   qkv [1,2560]x[2560,2048]、o [1,2048]x[2048,2048]、
    //   gate_up [1,12288]x[12288,2048]、down [1,2048]x[2048,6144]、
    //   lm_head [1,130560]x[130560,2048]
    using namespace mc::cfg;
    const uint32_t qkv_n = kNumQueryHeads * kHeadDim + 2 * kNumKvHeads * kHeadDim;
    const uint32_t shapes[5][2] = {
        {qkv_n, kHiddenSize},
        {kHiddenSize, kHiddenSize},
        {2 * kIntermediateSize, kHiddenSize},
        {kHiddenSize, kIntermediateSize},
        {kVocabSize, kHiddenSize},
    };
    for (const auto& nk : shapes) {
        const uint64_t key = Impl::make_key(1u, nk[0], nk[1]);
        std::lock_guard<std::mutex> lk(impl_->mu);
        if (impl_->cache.count(key) != 0) continue; // 多 session 幂等
        GemmPlan p;
        if (build_plan(1u, nk[0], nk[1], p) != MC_OK) return MC_E_CUDA;
        if (!pick_algo(handle_, p, ws_size, &p.algo)) {
            mc::set_error("prewarm_decode_shapes: no cublasLt algo for N=%u K=%u "
                          "(ws=%zu)",
                          nk[0], nk[1], ws_size);
            destroy_plan(p);
            return MC_E_CUDA;
        }
        p.algo_valid = true;
        impl_->cache.emplace(key, p);
    }
    return MC_OK;
}

mc_status_t GemmEngine::gemm_bf16(cudaStream_t stream, void* workspace,
                                  size_t ws_size, uint32_t M, uint32_t N, uint32_t K,
                                  const void* d_a, const void* d_b,
                                  void* d_c) const {
    if (handle_ == nullptr || d_a == nullptr || d_b == nullptr || d_c == nullptr ||
        M == 0 || N == 0 || K == 0) {
        mc::set_error("gemm_bf16: invalid args (M=%u N=%u K=%u)", M, N, K);
        return MC_E_INVALID_ARGUMENT;
    }
    if (workspace == nullptr || ws_size == 0) {
        mc::set_error("gemm_bf16: no workspace provided");
        return MC_E_INTERNAL;
    }

    GemmPlan plan; // 描述符按值拷贝（指针），锁外使用；缓存条目本体不迁移
    bool uncached = false;
    {
        std::lock_guard<std::mutex> lk(impl_->mu);
        const uint64_t key = Impl::make_key(M, N, K);
        auto it = impl_->cache.find(key);
        if (it != impl_->cache.end()) {
            plan = it->second;
        } else if (impl_->cache.size() < kMaxCachedPlans) {
            GemmPlan p;
            if (build_plan(M, N, K, p) != MC_OK) return MC_E_CUDA;
            it = impl_->cache.emplace(key, p).first;
            plan = p;
        } else {
            uncached = true;
        }
    }

    if (uncached) {
        // 缓存已满（仅 prefill 变 shape 会到这）：thread_local 计划复用，
        // 每次 heuristic 查询（host 侧分配，prefill 期可接受，任务书 §2）。
        static thread_local GemmPlan tmp{};
        destroy_plan(tmp);
        if (build_plan(M, N, K, tmp) != MC_OK) return MC_E_CUDA;
        if (!pick_algo(handle_, tmp, ws_size, &tmp.algo)) {
            mc::set_error("gemm_bf16: no cublasLt algo for M=%u N=%u K=%u (ws=%zu)", M,
                          N, K, ws_size);
            return MC_E_CUDA;
        }
        return run_matmul(handle_, tmp, d_a, d_b, d_c, workspace, ws_size, stream, M, N,
                          K);
    }

    if (!plan.algo_valid) {
        // 首次执行：选定 algo 并写回缓存（下不为例）。
        if (!pick_algo(handle_, plan, ws_size, &plan.algo)) {
            mc::set_error("gemm_bf16: no cublasLt algo for M=%u N=%u K=%u (ws=%zu)", M,
                          N, K, ws_size);
            return MC_E_CUDA;
        }
        plan.algo_valid = true;
        std::lock_guard<std::mutex> lk(impl_->mu);
        auto it = impl_->cache.find(Impl::make_key(M, N, K));
        if (it != impl_->cache.end()) {
            it->second.algo = plan.algo;
            it->second.algo_valid = true;
        }
    }
    return run_matmul(handle_, plan, d_a, d_b, d_c, workspace, ws_size, stream, M, N,
                      K);
}

// ---- F2a：FP8 W8A8 ----

namespace {

// 建期探测 OUTER_VEC_32F 支持性（调用方持 impl 锁）：以代表性小 shape
// (M=8,N=256,K=256) 建 outer-vec 计划 → heuristic 有解即支持（属性设置或
// heuristic 失败/无 algo → 不支持，回退 per-tensor）。返回 1=outer_vec /
// 2=per-tensor（此时 *d_one_out 分配 device 常量 1.0f，scale 指针用）/
// 0=探测失败。模式选择 std::printf 日志注明。
int probe_fp8_mode(cublasLtHandle_t h, float** d_one_out) {
    bool outer_ok = false;
    GemmFp8Plan p;
    if (build_fp8_plan(8u, 256u, 256u, true, p) == MC_OK) {
        cublasLtMatmulAlgo_t algo{};
        outer_ok = pick_algo_fp8(h, p, 32u << 20, &algo);
        destroy_fp8_plan(p);
    }
    if (!outer_ok) {
        // 回退可用性验证（per-tensor fp8 在本机必支持；不成立则 fp8 不可用）
        GemmFp8Plan q;
        cublasLtMatmulAlgo_t algo{};
        const bool scalar_ok =
            build_fp8_plan(8u, 256u, 256u, false, q) == MC_OK &&
            pick_algo_fp8(h, q, 32u << 20, &algo);
        destroy_fp8_plan(q);
        if (!scalar_ok) return 0;
        const float one = 1.0f;
        if (cudaMalloc((void**)d_one_out, 16u) != cudaSuccess ||
            cudaMemcpy(*d_one_out, &one, sizeof(one), cudaMemcpyHostToDevice) !=
                cudaSuccess) {
            *d_one_out = nullptr;
            return 0;
        }
    }
    std::printf("[GemmEngine] fp8 scale mode probed: %s\n",
                outer_ok ? "OUTER_VEC_32F (w_scale[n] x x_scale[m] fused in "
                          "cublasLt; no post kernel)"
                         : "per-tensor scalar + post_scale_rows fallback "
                           "(OUTER_VEC_32F unsupported by this cublasLt)");
    return outer_ok ? 1 : 2;
}

} // namespace

int GemmEngine::fp8_scale_mode() const {
    return impl_ != nullptr ? impl_->fp8_mode : 0;
}

mc_status_t GemmEngine::gemm_fp8_q(cudaStream_t stream, void* workspace,
                                   size_t ws_size, uint32_t M, uint32_t N,
                                   uint32_t K, const void* d_w_fp8,
                                   const float* d_w_scale, const void* d_x_fp8,
                                   const float* d_x_scale,
                                   uint16_t* d_y_bf16) const {
    if (handle_ == nullptr || d_w_fp8 == nullptr || d_w_scale == nullptr ||
        d_x_fp8 == nullptr || d_x_scale == nullptr || d_y_bf16 == nullptr ||
        M == 0 || N == 0 || K == 0) {
        mc::set_error("gemm_fp8: invalid args (M=%u N=%u K=%u)", M, N, K);
        return MC_E_INVALID_ARGUMENT;
    }
    if ((K & 15u) != 0u || !aligned16(d_w_fp8) || !aligned16(d_x_fp8) ||
        !aligned16(d_y_bf16)) {
        mc::set_error("gemm_fp8: K%%16!=0 or misaligned pointers (K=%u)", K);
        return MC_E_INVALID_ARGUMENT;
    }
    if (workspace == nullptr || ws_size == 0) {
        mc::set_error("gemm_fp8: no workspace provided");
        return MC_E_INTERNAL;
    }

    // 建期探测（一次）
    {
        std::lock_guard<std::mutex> lk(impl_->mu);
        if (impl_->fp8_mode == 0) {
            impl_->fp8_mode = probe_fp8_mode(handle_, &impl_->d_one);
            if (impl_->fp8_mode == 0) {
                mc::set_error("gemm_fp8: scale mode probe failed");
                return MC_E_CUDA;
            }
        }
    }

    // 计划查找/构建（键 (M,N,K)，与 bf16 分表）。outer_vec 计划在该 shape
    // 无 heuristic 解时降级 per-tensor（模式级探测 + 计划级兜底双保险）。
    bool needs_post = false;
    mc_status_t rc = MC_OK;
    {
        std::lock_guard<std::mutex> lk(impl_->mu);
        const uint64_t key = Impl::make_key(M, N, K);
        auto it = impl_->fp8_cache.find(key);
        if (it == impl_->fp8_cache.end()) {
            GemmFp8Plan p;
            bool outer = impl_->fp8_mode == 1;
            if (build_fp8_plan(M, N, K, outer, p) != MC_OK) return MC_E_CUDA;
            cublasLtMatmulAlgo_t algo{};
            if (!pick_algo_fp8(handle_, p, ws_size, &algo)) {
                if (outer) { // 降级重试 per-tensor
                    destroy_fp8_plan(p);
                    if (build_fp8_plan(M, N, K, false, p) != MC_OK) return MC_E_CUDA;
                    if (!pick_algo_fp8(handle_, p, ws_size, &algo)) {
                        destroy_fp8_plan(p);
                        mc::set_error("gemm_fp8: no cublasLt algo (M=%u N=%u K=%u)",
                                      M, N, K);
                        return MC_E_CUDA;
                    }
                } else {
                    destroy_fp8_plan(p);
                    mc::set_error("gemm_fp8: no cublasLt algo (M=%u N=%u K=%u)", M,
                                  N, K);
                    return MC_E_CUDA;
                }
            }
            p.algo = algo;
            p.algo_valid = true;
            it = impl_->fp8_cache.emplace(key, p).first;
        }
        // scale 指针 per-call 设置 + matmul 发射（锁内：desc 是缓存共享对象，
        // 引擎单 session 串行口径见文件头）
        GemmFp8Plan& p = it->second;
        const float* a_scale = p.outer_vec ? d_w_scale : impl_->d_one;
        const float* b_scale = p.outer_vec ? d_x_scale : impl_->d_one;
        rc = run_matmul_fp8(handle_, p, d_w_fp8, d_x_fp8, d_y_bf16, a_scale,
                            b_scale, workspace, ws_size, stream, M, N, K);
        needs_post = !p.outer_vec;
    }
    if (rc != MC_OK) return rc;

    if (needs_post) {
        // 回退模式：GEMM 出原始 fp8 点积（scale=1.0），此处补
        // y[m,n] ×= w_scale[n]·x_scale[m]
        return launch_post_scale_rows(d_y_bf16, d_w_scale, d_x_scale, M, N, stream);
    }
    return MC_OK;
}

mc_status_t GemmEngine::gemm_fp8(cudaStream_t stream, void* workspace,
                                 size_t ws_size, uint32_t M, uint32_t N,
                                 uint32_t K, const void* d_w_fp8,
                                 const float* d_w_scale,
                                 const uint16_t* d_x_bf16, uint16_t* d_y_bf16,
                                 void* d_x_fp8_scratch,
                                 float* d_x_scale_scratch) const {
    if (d_x_bf16 == nullptr || d_x_fp8_scratch == nullptr ||
        d_x_scale_scratch == nullptr || !aligned16(d_x_bf16)) {
        mc::set_error("gemm_fp8: invalid quant buffers");
        return MC_E_INVALID_ARGUMENT;
    }
    // per-token 量化（bf16 → e4m3 + scale[M]）→ 预量化 GEMM 路径
    mc_status_t rc = launch_quant_rows(d_x_bf16, (uint8_t*)d_x_fp8_scratch,
                                       d_x_scale_scratch, M, K, stream);
    if (rc != MC_OK) return rc;
    return gemm_fp8_q(stream, workspace, ws_size, M, N, K, d_w_fp8, d_w_scale,
                      d_x_fp8_scratch, d_x_scale_scratch, d_y_bf16);
}

} // namespace mc
