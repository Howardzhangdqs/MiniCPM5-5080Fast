// cuda_graph.cpp — GraphManager 实现：Decode CUDA Graph capture/replay
//（计划 §12.2/§12.3，P3 落地）。
#include "runtime/cuda_graph.h"

#include <cstdio>
#include <cstdlib>
#include <vector>

#include "runtime/error.h"

GraphManager::~GraphManager() { destroy(); }

mc_status_t GraphManager::begin_capture(cudaStream_t stream) {
    if (exec_ != nullptr) {
        mc::set_error("GraphManager: graph already instantiated");
        return MC_E_INTERNAL;
    }
    if (capturing_) {
        mc::set_error("GraphManager: capture already in progress");
        return MC_E_INTERNAL;
    }
    cudaError_t e = cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    if (e != cudaSuccess) {
        cudaGetLastError();
        mc::set_error("GraphManager: cudaStreamBeginCapture failed: %s",
                      cudaGetErrorString(e));
        return MC_E_GRAPH;
    }
    capturing_ = true;
    return MC_OK;
}

mc_status_t GraphManager::end_capture(cudaStream_t stream, bool pdl_edges) {
    if (!capturing_) {
        mc::set_error("GraphManager: end_capture without begin_capture");
        return MC_E_INTERNAL;
    }
    capturing_ = false;

    // 无论后续步骤成败，capture 状态在此终结（stream 恢复正常模式）。
    cudaGraph_t g = nullptr;
    cudaError_t e = cudaStreamEndCapture(stream, &g);
    if (e != cudaSuccess) {
        cudaGetLastError();
        if (g != nullptr) cudaGraphDestroy(g); // 防御：部分实现失败时仍返回图
        mc::set_error("GraphManager: cudaStreamEndCapture failed (capture "
                      "invalidated by an illegal op?): %s",
                      cudaGetErrorString(e));
        return MC_E_GRAPH;
    }

    // ---- P5 PDL：stream capture 不保留 ProgrammaticStreamSerialization
    //      属性（实测 12.8：0 programmatic 边）—— 在 instantiate 前做图手术：
    //      kernel→kernel 边的 to_port 改为 cudaGraphKernelNodePortProgrammatic
    //      （消费侧 kernel 以 griddepcontrol.wait 同步；memcpy/memset 端点
    //      的边保持全完成语义）。失败只损失重叠（回退普通边），不让
    //      session 失败。----
    if (pdl_edges) {
        size_t nn = 0, ne = 0;
        if (cudaGraphGetNodes(g, nullptr, &nn) == cudaSuccess &&
            cudaGraphGetEdges_v2(g, nullptr, nullptr, nullptr, &ne) == cudaSuccess &&
            nn > 0 && ne > 0) {
            std::vector<cudaGraphNode_t> nodes(nn);
            std::vector<cudaGraphNode_t> from(ne), to(ne);
            std::vector<cudaGraphEdgeData> ed(ne);
            std::vector<cudaGraphNode_t> rf, rt;
            std::vector<cudaGraphEdgeData> rd; // 原边（remove 匹配用）
            std::vector<cudaGraphNode_t> af, at;
            std::vector<cudaGraphEdgeData> ad; // 改写后的边（add 用）
            if (cudaGraphGetNodes(g, nodes.data(), &nn) == cudaSuccess &&
                cudaGraphGetEdges_v2(g, from.data(), to.data(), ed.data(),
                                     &ne) == cudaSuccess) {
                // 节点类型表（kernel 判定）
                std::vector<bool> is_kernel(nn, false);
                for (size_t i = 0; i < nn; ++i) {
                    cudaGraphNodeType t = cudaGraphNodeTypeEmpty;
                    if (cudaGraphNodeGetType(nodes[i], &t) == cudaSuccess)
                        is_kernel[i] = (t == cudaGraphNodeTypeKernel);
                }
                std::vector<int> idx(nn, -1);
                for (size_t i = 0; i < nn; ++i) idx[i] = (int)i;
                auto find_node = [&](cudaGraphNode_t n) -> int {
                    for (size_t i = 0; i < nn; ++i)
                        if (nodes[i] == n) return (int)i;
                    return -1;
                };
                for (size_t i = 0; i < ne; ++i) {
                    const int fi = find_node(from[i]);
                    const int ti = find_node(to[i]);
                    if (fi < 0 || ti < 0 || !is_kernel[fi] || !is_kernel[ti])
                        continue; // 端点非 kernel：保持全完成语义
                    // PDL 边：type=Programmatic + from_port=Programmatic
                    //（生产者 trigger 或退出时激活；to_port 必须为 0 ——
                    // 消费侧以 griddepcontrol.wait 做内存可见性同步）
                    cudaGraphEdgeData d = ed[i];
                    rd.push_back(ed[i]);
                    rf.push_back(from[i]);
                    rt.push_back(to[i]);
                    d.type = cudaGraphDependencyTypeProgrammatic;
                    d.from_port = cudaGraphKernelNodePortProgrammatic;
                    d.to_port = 0;
                    ad.push_back(d);
                    af.push_back(from[i]);
                    at.push_back(to[i]);
                }
                if (!af.empty()) {
                    const bool dbg = getenv("MC_GRAPH_DEBUG_EDGES") != nullptr;
                    cudaError_t e1 = cudaGraphRemoveDependencies_v2(
                        g, rf.data(), rt.data(), rd.data(), rf.size());
                    if (dbg)
                        fprintf(stderr, "[minicpm-native] surgery remove(%zu): %s\n",
                                rf.size(), cudaGetErrorName(e1));
                    if (e1 == cudaSuccess) {
                        e1 = cudaGraphAddDependencies_v2(g, af.data(), at.data(),
                                                         ad.data(), af.size());
                        if (dbg)
                            fprintf(stderr, "[minicpm-native] surgery add-prog: %s\n",
                                    cudaGetErrorName(e1));
                        if (e1 != cudaSuccess) {
                            // 加回原边（回退，保证图可用）
                            cudaGetLastError();
                            e1 = cudaGraphAddDependencies_v2(g, rf.data(), rt.data(),
                                                             rd.data(), rf.size());
                            if (dbg)
                                fprintf(stderr,
                                        "[minicpm-native] surgery rollback-add: %s\n",
                                        cudaGetErrorName(e1));
                        }
                    } else {
                        cudaGetLastError();
                    }
                    if (getenv("MC_GRAPH_DEBUG_EDGES") != nullptr)
                        fprintf(stderr,
                                "[minicpm-native] graph PDL surgery: %zu edges "
                                "rewritten (of %zu)\n",
                                af.size(), ne);
                }
            }
        } else {
            cudaGetLastError();
        }
    }

    cudaGraphExec_t exec = nullptr;
    e = cudaGraphInstantiate(&exec, g, 0);
    if (e != cudaSuccess) {
        cudaGetLastError();
        cudaGraphDestroy(g);
        mc::set_error("GraphManager: cudaGraphInstantiate failed: %s",
                      cudaGetErrorString(e));
        return MC_E_GRAPH;
    }
    graph_ = g;     // 保留原 graph 供 node_count 统计（§21 报告用）
    exec_ = exec;

    // P5 PDL 诊断（MC_GRAPH_DEBUG_EDGES=1）：统计边的端口类型 ——
    // cudaGraphKernelNodePortProgrammatic = PDL 边（后继可提前发射）
    if (getenv("MC_GRAPH_DEBUG_EDGES") != nullptr) {
        size_t needed = 0;
        if (cudaGraphGetEdges_v2(g, nullptr, nullptr, nullptr, &needed) ==
            cudaSuccess) {
            std::vector<cudaGraphNode_t> from(needed), to(needed);
            std::vector<cudaGraphEdgeData> ed(needed);
            if (cudaGraphGetEdges_v2(g, from.data(), to.data(), ed.data(),
                                  &needed) == cudaSuccess) {
                size_t prog = 0, lc = 0;
                for (size_t i = 0; i < needed; ++i) {
                    if (ed[i].type == cudaGraphDependencyTypeProgrammatic) ++prog;
                    if (ed[i].from_port == cudaGraphKernelNodePortLaunchCompletion)
                        ++lc;
                }
                fprintf(stderr,
                        "[minicpm-native] graph edges: %zu total, %zu "
                        "programmatic-dep, %zu launch-completion-from\n",
                        needed, prog, lc);
            }
        }
        cudaGetLastError();
    }
    return MC_OK;
}

mc_status_t GraphManager::replay(cudaStream_t stream) {
    if (exec_ == nullptr) {
        mc::set_error("GraphManager: replay without an instantiated graph");
        return MC_E_INTERNAL;
    }
    cudaError_t e = cudaGraphLaunch(exec_, stream);
    if (e != cudaSuccess) {
        cudaGetLastError();
        mc::set_error("GraphManager: cudaGraphLaunch failed: %s",
                      cudaGetErrorString(e));
        return MC_E_CUDA;
    }
    launches_ += 1;
    return MC_OK;
}

size_t GraphManager::node_count() const {
    if (graph_ == nullptr) return 0;
    size_t n = 0;
    if (cudaGraphGetNodes(graph_, nullptr, &n) != cudaSuccess) {
        cudaGetLastError();
        return 0;
    }
    return n;
}

void GraphManager::destroy() {
    if (exec_ != nullptr) {
        (void)cudaGraphExecDestroy(exec_);
        exec_ = nullptr;
    }
    if (graph_ != nullptr) {
        (void)cudaGraphDestroy(graph_);
        graph_ = nullptr;
    }
    capturing_ = false;
    launches_ = 0;
}
