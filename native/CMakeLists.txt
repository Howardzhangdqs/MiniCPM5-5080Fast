# minicpm_native — MiniCPM5-2B C++/CUDA runtime (plan §24.1)
#
# Usage (plan §24.2):
#   cmake -S native -B build/native -G Ninja \
#     -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=120
#   cmake --build build/native -j
#   ctest --test-dir build/native --output-on-failure

cmake_minimum_required(VERSION 3.28)

# RTX 5080 (Blackwell GeForce, SM120) is the only target; allow explicit override.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 120 CACHE STRING "CUDA architectures (default: SM120)" FORCE)
endif()

project(minicpm_native LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED) # CUDA::cudart + CUDAToolkit_INCLUDE_DIRS

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# The documented default build is Release (plan §24.2).
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type (default Release)" FORCE)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -DNDEBUG")

# Link libcudart dynamically everywhere: two statically-linked cudart copies in
# one process (shared lib + test executable) would fight over the primary
# context and make device allocations non-interoperable.
set(CUDA_RUNTIME_LIBRARY Shared)

option(MC_NATIVE_MOCK_FORWARD
    "Deterministic mock forward path (fallback for weightless environments). OFF = real BF16 forward (P2); wpk-missing sessions still fall back to mock at runtime." OFF)

add_compile_options(
    $<$<COMPILE_LANGUAGE:CXX>:-Wall>
    $<$<COMPILE_LANGUAGE:CXX>:-Wextra>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra>
)

set(MC_RUNTIME_SOURCES
    runtime/error.cpp
    runtime/model.cpp
    runtime/session.cpp
    runtime/memory_arena.cpp
    runtime/kv_cache.cpp
    runtime/weight_loader.cpp
    runtime/gemm.cpp
    runtime/gemv_tactic.cpp
    runtime/forward.cpp
    runtime/debug_dump.cpp
    runtime/prefill_executor.cpp
    runtime/decode_executor.cpp
    runtime/cuda_graph.cpp
    runtime/batch_group.cpp
)

set(MC_KERNEL_SOURCES
    kernels/sm120/embedding.cu
    kernels/sm120/rmsnorm.cu
    kernels/sm120/rope_kv.cu
    kernels/sm120/attention_prefill.cu
    kernels/sm120/attention_decode.cu
    kernels/sm120/gemv.cu
    kernels/sm120/gemv_fp8.cu
    kernels/sm120/quant.cu
    kernels/sm120/swiglu.cu
    kernels/sm120/sampling.cu
)

add_library(minicpm_native SHARED ${MC_RUNTIME_SOURCES} ${MC_KERNEL_SOURCES})

target_include_directories(minicpm_native
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CUDAToolkit_INCLUDE_DIRS})

if(MC_NATIVE_MOCK_FORWARD)
    target_compile_definitions(minicpm_native PRIVATE MC_NATIVE_MOCK_FORWARD=1)
else()
    # P2 默认：真实 BF16 forward（wpk 缺失的 session 在运行时回退 mock）。
    target_compile_definitions(minicpm_native PRIVATE MC_NATIVE_MOCK_FORWARD=0)
endif()

# cuBLASLt（Linear: bf16 in → fp32 累加 → bf16 out）
target_link_libraries(minicpm_native PRIVATE CUDA::cublasLt)

# Hide everything except the public ABI (mc_* functions are marked MC_API).
set_target_properties(minicpm_native PROPERTIES
    VERSION   0.1.0
    SOVERSION 0
    CXX_VISIBILITY_PRESET hidden
    CUDA_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON)

enable_testing()

add_executable(abi_smoke tests/abi_smoke.cpp)
target_compile_features(abi_smoke PRIVATE cxx_std_20)
target_include_directories(abi_smoke PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(abi_smoke PRIVATE minicpm_native CUDA::cudart)

add_test(NAME abi_smoke COMMAND abi_smoke)
set_tests_properties(abi_smoke PROPERTIES
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

# ---- P2 真实推理测试（需要 weights；未就绪则 SKIP+PASS）----
# 模型目录解析顺序：MC_MODEL_DIR 环境变量 > 仓库 models/minicpm5-2b >
# 本构建目录 wpk_model（make_wpk 产物）。golden 目录：MC_GOLDEN_DIR >
# 仓库 tests/runtime/golden。
add_executable(real_inference tests/real_inference.cpp)
target_compile_features(real_inference PRIVATE cxx_std_20)
target_include_directories(real_inference PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR} # tests/golden_compare.h（与 gpu_layerwise 共用的对拍策略头）
    ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(real_inference PRIVATE minicpm_native CUDA::cudart)
target_compile_definitions(real_inference PRIVATE
    MC_REPO_ROOT="${CMAKE_CURRENT_SOURCE_DIR}/.."
    MC_BUILD_DIR="${CMAKE_CURRENT_BINARY_DIR}")
add_test(NAME real_inference COMMAND real_inference)
set_tests_properties(real_inference PROPERTIES
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

# ---- P3 CUDA Graph A/B（golden token 对拍 + ctx16/2048/8192 TPOT 对比 →
#      benchmark/results/graph_ab.json；缺权重/golden 则 FAIL）----
add_executable(graph_bench tests/graph_bench.cpp)
target_compile_features(graph_bench PRIVATE cxx_std_20)
target_include_directories(graph_bench PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR} # tests/golden_compare.h
    ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(graph_bench PRIVATE minicpm_native CUDA::cudart)
target_compile_definitions(graph_bench PRIVATE
    MC_REPO_ROOT="${CMAKE_CURRENT_SOURCE_DIR}/.."
    MC_BUILD_DIR="${CMAKE_CURRENT_BINARY_DIR}")
add_test(NAME graph_bench COMMAND graph_bench)
set_tests_properties(graph_bench PROPERTIES
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

# ---- P3+ 自研 GEMV 微基准（5 个 decode M=1 shape：CPU 参考正确性 +
#      cuBLASLt vs rows vs split-K 计时 → benchmark/results/gemv_ab.json）----
add_executable(gemv_bench
    tests/gemv_bench.cpp
    kernels/sm120/gemv.cu
    kernels/sm120/gemv_fp8.cu
    kernels/sm120/quant.cu
    runtime/gemm.cpp
    runtime/error.cpp)
target_compile_features(gemv_bench PRIVATE cxx_std_20)
target_include_directories(gemv_bench PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(gemv_bench PRIVATE CUDA::cudart CUDA::cublasLt)
target_compile_definitions(gemv_bench PRIVATE
    MC_REPO_ROOT="${CMAKE_CURRENT_SOURCE_DIR}/..")
add_test(NAME gemv_bench COMMAND gemv_bench)
set_tests_properties(gemv_bench PROPERTIES
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

# ---- P4 FP8 集成（fp8 包缺失时醒目 SKIP+PASS；BF16 套件契约不变）----
add_executable(fp8_inference tests/fp8_inference.cpp)
target_compile_features(fp8_inference PRIVATE cxx_std_20)
target_include_directories(fp8_inference PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR} # tests/golden_compare.h
    ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(fp8_inference PRIVATE minicpm_native CUDA::cudart)
target_compile_definitions(fp8_inference PRIVATE
    MC_REPO_ROOT="${CMAKE_CURRENT_SOURCE_DIR}/.."
    MC_BUILD_DIR="${CMAKE_CURRENT_BINARY_DIR}")
add_test(NAME fp8_inference COMMAND fp8_inference)
set_tests_properties(fp8_inference PROPERTIES
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

# ---- P5 TPOT 分段分解（MC_BREAKDOWN 事件插桩；缺模型醒目 SKIP+PASS）----
add_executable(tpot_breakdown tests/tpot_breakdown.cpp)
target_compile_features(tpot_breakdown PRIVATE cxx_std_20)
target_include_directories(tpot_breakdown PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR} # tests/fp8_fixture.hpp
    ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(tpot_breakdown PRIVATE minicpm_native CUDA::cudart)
target_compile_definitions(tpot_breakdown PRIVATE
    MC_REPO_ROOT="${CMAKE_CURRENT_SOURCE_DIR}/.."
    MC_BUILD_DIR="${CMAKE_CURRENT_BINARY_DIR}")
add_test(NAME tpot_breakdown COMMAND tpot_breakdown)
set_tests_properties(tpot_breakdown PROPERTIES
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

# ---- P5 验收基准（空闲 GPU 5×200 中位数 → benchmark/results/perf_v2.json）----
add_executable(perf_v2 tests/perf_v2.cpp)
target_compile_features(perf_v2 PRIVATE cxx_std_20)
target_include_directories(perf_v2 PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(perf_v2 PRIVATE minicpm_native CUDA::cudart)
target_compile_definitions(perf_v2 PRIVATE
    MC_REPO_ROOT="${CMAKE_CURRENT_SOURCE_DIR}/.."
    MC_BUILD_DIR="${CMAKE_CURRENT_BINARY_DIR}")
add_test(NAME perf_v2 COMMAND perf_v2)
set_tests_properties(perf_v2 PROPERTIES
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

# ---- P2 分窗 GPU 逐层验证（显存受限：部分加载 + 窗口链 seed + golden 对拍；
#      free VRAM 不足以开 4 层窗时 SKIP+PASS）----
add_executable(gpu_layerwise tests/gpu_layerwise.cpp)
target_compile_features(gpu_layerwise PRIVATE cxx_std_20)
target_include_directories(gpu_layerwise PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR} # tests/golden_compare.h（与 real_inference 共用的对拍策略头）
    ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(gpu_layerwise PRIVATE minicpm_native CUDA::cudart)
target_compile_definitions(gpu_layerwise PRIVATE
    MC_REPO_ROOT="${CMAKE_CURRENT_SOURCE_DIR}/.."
    MC_BUILD_DIR="${CMAKE_CURRENT_BINARY_DIR}")
add_test(NAME gpu_layerwise COMMAND gpu_layerwise)
set_tests_properties(gpu_layerwise PROPERTIES
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

# ---- CPU 参考实现（golden 自举验证；工具链 lane 的 golden 生成器就绪后
#      由 tests/runtime/golden 替代）----
add_executable(reference_forward tests/reference_forward.cpp)
target_compile_features(reference_forward PRIVATE cxx_std_20)
target_include_directories(reference_forward PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(reference_forward PRIVATE
    MC_REPO_ROOT="${CMAKE_CURRENT_SOURCE_DIR}/..")
add_test(NAME reference_forward COMMAND reference_forward
    ${CMAKE_CURRENT_BINARY_DIR}/local_golden)
set_tests_properties(reference_forward PROPERTIES
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

# ---- wpk 生成器（safetensors → model.wpk，测试自举用；工具链 lane 的
#      pack_weights.py 就绪后由其替代）----
add_executable(make_wpk tests/make_wpk.cpp)
target_compile_features(make_wpk PRIVATE cxx_std_20)
target_include_directories(make_wpk PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
                                            ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(make_wpk PRIVATE MC_REPO_ROOT="${CMAKE_CURRENT_SOURCE_DIR}/..")
add_test(NAME make_wpk COMMAND make_wpk ${CMAKE_CURRENT_BINARY_DIR}/wpk_model)
set_tests_properties(make_wpk PROPERTIES
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

# ---- P7 生产化加固 soak（计划 §20-P7 / §26.4）：长输出 soak / 多 session
#      并发隔离 / OOM 行为 / reset 压测 / VRAM 预算表。需要 model.wpk，
#      缺包则 SKIP+PASS（与 real_inference 同约定）。----
add_executable(soak_test tests/soak_test.cpp)
target_compile_features(soak_test PRIVATE cxx_std_20)
target_include_directories(soak_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(soak_test PRIVATE minicpm_native CUDA::cudart)
target_compile_definitions(soak_test PRIVATE
    MC_REPO_ROOT="${CMAKE_CURRENT_SOURCE_DIR}/.."
    MC_BUILD_DIR="${CMAKE_CURRENT_BINARY_DIR}")
add_test(NAME soak_test COMMAND soak_test)
set_tests_properties(soak_test PROPERTIES
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

# ---- 采样路径持久污染回归（2026-09 事故）：采样→reset→greedy 对拍 /
#      temp→0⁺ 采样≈greedy / top-k 成员性 / 同 seed 确定性 / rope 表完好
#      钩子。需要 model.wpk，缺包则 SKIP+PASS（与 soak_test 同约定）。----
add_executable(sampling_regression tests/sampling_regression.cpp)
target_compile_features(sampling_regression PRIVATE cxx_std_20)
target_include_directories(sampling_regression PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(sampling_regression PRIVATE minicpm_native CUDA::cudart)
target_compile_definitions(sampling_regression PRIVATE
    MC_REPO_ROOT="${CMAKE_CURRENT_SOURCE_DIR}/.."
    MC_BUILD_DIR="${CMAKE_CURRENT_BINARY_DIR}")
add_test(NAME sampling_regression COMMAND sampling_regression)
set_tests_properties(sampling_regression PROPERTIES
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

# ---- B1b 批组端到端（批组运行时 + eager 批前向 + 生命周期/性能门/泄漏；
#      需 model.wpk，缺包则 SKIP+PASS（与 soak_test 同约定）----
add_executable(batch_test tests/batch_test.cpp)
target_compile_features(batch_test PRIVATE cxx_std_20)
target_include_directories(batch_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(batch_test PRIVATE minicpm_native CUDA::cudart)
target_compile_definitions(batch_test PRIVATE
    MC_REPO_ROOT="${CMAKE_CURRENT_SOURCE_DIR}/.."
    MC_BUILD_DIR="${CMAKE_CURRENT_BINARY_DIR}")
add_test(NAME batch_test COMMAND batch_test)
set_tests_properties(batch_test PROPERTIES
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

# ---- P2 单算级 kernel 测试（小合成张量 vs CPU 规范参考；显存 ≤ 64MB）----
# 注意：minicpm_native 把 launcher 符号藏起来了（hidden visibility，只有
# mc_* ABI 导出），无法从 .so 链接 launcher；因此本 target 直接编译同一份
# kernel / runtime 源文件（被测代码与库内逐字节相同）。
add_executable(kernel_ops
    tests/kernel_ops.cpp
    kernels/sm120/embedding.cu
    kernels/sm120/rmsnorm.cu
    kernels/sm120/rope_kv.cu
    kernels/sm120/attention_prefill.cu
    kernels/sm120/attention_decode.cu
    kernels/sm120/gemv.cu
    kernels/sm120/gemv_fp8.cu
    kernels/sm120/quant.cu
    kernels/sm120/swiglu.cu
    kernels/sm120/sampling.cu
    runtime/error.cpp
    runtime/gemm.cpp)
target_compile_features(kernel_ops PRIVATE cxx_std_20)
target_include_directories(kernel_ops PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(kernel_ops PRIVATE CUDA::cudart CUDA::cublasLt)
target_compile_definitions(kernel_ops PRIVATE
    MC_REPO_ROOT="${CMAKE_CURRENT_SOURCE_DIR}/..")
add_test(NAME kernel_ops COMMAND kernel_ops)
set_tests_properties(kernel_ops PROPERTIES
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
