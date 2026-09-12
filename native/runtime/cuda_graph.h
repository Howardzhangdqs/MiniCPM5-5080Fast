// cuda_graph.h — GraphManager：Decode CUDA Graph capture/replay（计划 §12）。
//
// P3 落地（§12.2/§12.3）：decode step 的完整计算链（embedding → 42 层 →
// final norm → lm_head → argmax → append → seq_len+1 → D2H 4B token）在
// session_create 尾部 capture 一次、instantiate 一次，之后每步
// mc_decode_one 只做：ensure_pages（host）→ cudaGraphLaunch → 同步 → 读
// pinned token。
//
// 可捕获性约定（§12.2 固定内容）：d_state / d_next_token / h_next_token /
// d_seq / d_page_table / kv_pool / activation scratch / workspace 的地址在
// session 生存期内固定；seq_len 变化只改 d_state 内容（kernel 从
// &d_state->seq_len 这个固定地址运行时读取），绝不改变任何 tensor
// pointer 或 launch 参数。
//
// 失效条件（§12.4）：本阶段 graph 只覆盖 decode（M=1，shape 恒定）；
// prefill（M 变化）恒走 eager；capture/instantiate 失败 → 记错误并自动
// 回退 eager 模式（session 不失效）。
#pragma once

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

#include "minicpm_runtime.h"

class GraphManager {
public:
    GraphManager() = default;
    GraphManager(const GraphManager&) = delete;
    GraphManager& operator=(const GraphManager&) = delete;
    ~GraphManager(); // 等价 destroy()（防御；正常路径由 session 释放期显式调用）

    static constexpr bool supported() { return true; }

    // 进入 stream capture（cudaStreamCaptureModeGlobal）。调用方随后在
    // stream 上照常 enqueue decode step 的全部工作，最后调 end_capture。
    mc_status_t begin_capture(cudaStream_t stream);

    // 结束 capture 并 instantiate。无论成败都会终结 capture 状态并清理
    // 半成品（stream 恢复正常模式）；成功后 available() 为真。
    // pdl_edges：capture 后把 kernel→kernel 边改写为 programmatic 入端口
    //（PDL；要求消费侧 kernel 内含 griddepcontrol.wait —— greedy decode 链
    // 全部满足；采样变体图不满足，必须 false）
    mc_status_t end_capture(cudaStream_t stream, bool pdl_edges = false);

    // replay：cudaGraphLaunch（hot path 唯一的 CUDA 提交调用）。
    mc_status_t replay(cudaStream_t stream);

    bool available() const { return exec_ != nullptr; }

    // 累计 graph launch 次数（对应 mc_runtime_stats_t.cuda_graph_launches）。
    uint64_t launches() const { return launches_; }

    // graph 节点数（cudaGraphGetNodes 统计；建期一次，供报告）。
    size_t node_count() const;

    // 销毁 exec/graph（幂等）。session 释放期调用。
    void destroy();

private:
    cudaGraph_t graph_ = nullptr; // capture 产物（保留供 node_count 统计）
    cudaGraphExec_t exec_ = nullptr;
    bool capturing_ = false;
    uint64_t launches_ = 0;
};
