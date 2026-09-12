# 实验与实施记录（Experiment Log）

MiniCPM5-2B × RTX 5080 推理系统的完整实验记录：阶段时间线、性能优化轨迹、正确性验证体系、缺陷分析与产物索引。项目现状见 [README](../README.md)（英文版：[English README](../README_EN.md)）。

- 硬件：NVIDIA GeForce RTX 5080 16GB（GeForce Blackwell，SM120 / CC 12.0，带宽 960 GB/s，TDP 360W，PCIe 5.0 x16，机器共享）；宿主机 AMD Ryzen 7 9700X（8C16T）/ 16GB 内存 / 技嘉 B850M / 金士顿 SNV3S1000G 1TB NVMe / Ubuntu 24.04.4 / 内核 7.0
- 软件：CUDA 12.8 / 驱动 595.84 / Rust 1.98 / CMake 3.28 / vLLM 0.29.0（torch 2.13.0+cu130，仅基线对照）
- 模型：openbmb/MiniCPM5-2B @ revision `3497c460`（42L / hidden 2048 / GQA 16q+2kv / head_dim 128 / vocab 130560 / 131K ctx；权重 sha256 见 `models/minicpm5-2b/reference_manifest.json`）
- 周期：2026-09-08 ~ 2026-09-09

---

## 1. 阶段时间线

| 阶段 | 内容 | 关键结果 |
|---|---|---|
| **P0** 模型资产锁定 | 下载 revision `3497c460` 全套文件（含 4.7GB 权重），sha256 记入 `reference_manifest.json`；实测 config 与计划一致 | ✅ |
| **P1** Rust Host | 8 crate workspace：OpenAI API + SSE、官方 chat_template.jinja 逐字执行（minijinja + tojson/str 方法兼容层）、HF tokenizers、stop/tool-call 状态机、调度器、metrics；12 场景模板 golden 快照 | 86 项 Rust 测试全绿；**关键发现：模板自带 bos，服务端必须 `add_special_tokens=false`** |
| **ABI v0.1** | `native/include/minicpm_runtime.h` 定稿（9 函数、struct_size/abi_version、opaque handle 按值 Pimpl）；Rust sys 镜像 + dlopen 封装 | ✅ |
| **P2** 真实 BF16 forward | 工具链（wpk 打包器 + numpy 参考前向 + 比对器）；cuBLASLt + 全层 kernel；`model.wpk`（255 sections，q/k/v 与 gate/up 行融合布局） | 三 golden case 逐 token 对齐；2K/8K 压力通过；server 真实模型首跑（think 块正常） |
| **P3** Graph 与融合 | Decode CUDA Graph（在 session 创建时 capture/instantiate）、attention 分块两段式、自研 bf16 GEMV + 构建期 tactic 选型 | 8K ctx decode **7.9x**；Graph 本身 1.00x（launch 非瓶颈，但 jitter ±2ms→±0.2ms，热点消除后显现 3-4% 收益） |
| **P4** FP8 | E4M3 逐行 scale 量化（168→169 sections，含 lm_head）、fp8 GEMV（硬件 cvt 指令 + shared LUT 演进）、resnorm 融合进 GEMV prologue、lm_head fp8 | TPOT 1.37-1.44x；精度 logits cos avg 0.9986 |
| **P4.2-4.3** BF16 攻坚 | PDL 全链（**发现：stream capture 静默丢弃 PDL 属性 → 需手工修补 graph 边**）、4 级流水线 attention（逐位比对锁定）、kv-major GQA 去冗余（K/V 流量 ÷8）、rope 吸收进 attention phase0（SASS 指令序列比对） | BF16 全 ctx 优化到位；ctx16 = 物理上限 90% |
| **P7** 生产化 | soak/OOM/多 session 隔离/reset 压测/VRAM 预算表（BF16+FP8）；采样启用规则对齐 OpenAI 语义 | 零泄漏零串扰；12 项 ctest 全绿 |
| **基线对照** | vLLM 0.29.0 同机同配置 batch-1 greedy | **native BF16 快 6-11%**，FP8 快 68-86% |

## 2. 性能优化轨迹（BF16 decode TPOT，batch-1，空闲 GPU）

| 阶段 | ctx16 | ctx2048 | ctx8192 | 主要手段 |
|---|---|---|---|---|
| P2 基线 | ~10ms | ~33ms | ~101ms | cuBLASLt + 基础 kernel，无 Graph |
| P3 attention 分块 | ~12.6ms | 14.1ms | 15.0ms | KV 分块两段式（每 q head 多 block） |
| P3 自研 GEMV | ~10.2ms | ~11.5ms | ~11.5ms | warp-per-row + 4 路 uint4 ILP，81-95% 峰值带宽 |
| P4 FP8 首版 | 4.9ms (205 t/s) | 5.8ms | 5.9ms | E4M3 权重，流量 4.5→2.55GB/token |
| P4.1 融合+v2 | 3.57ms (280) | 4.7ms | 4.6ms | resnorm 融合 prologue、fp8 硬件转换、两段 argmax、lm_head fp8 |
| P4.2 BF16 攻坚 | 6.03ms | 6.78ms | 6.96ms | PDL 全链、流水线 attention、深预取 |
| **P4.3 终态** | **5.84ms (171)** | **5.92ms (169)** | **6.52ms (153)** | kv-major GQA（÷8）+ rope 吸收（graph 节点 426→259） |
| **vs P2 累计** | **1.7x** | **5.6x** | **15.5x** | |

**vLLM 同机对照**（vLLM 0.29.0 / FA2+FlashInfer / 3 轮中位）：6.50 / 6.56 / 6.89 ms（154/152/145 t/s）→ native BF16 快 11%/11%/6%，native FP8 快 86%/68%/70%。vLLM 强项：prefill ~20k tok/s（native ~3.5k，未优化项）。API 级（FP8，含 host 全链路）实测 267 t/s。

**greedy 输出充分交叉比对**（crosscheck，2026-09-09）：47 条 prompt（9 类，含固定历史多轮）× {128, 256} = 94 组，`benchmark/scripts/crosscheck.py` 可复跑。修复 max_tokens 边界收尾语义前：全文逐字节一致 18.1%、收尾差异 19.1%、真分歧 62.8%。收尾差异恒为 ours 早 1 token 的边界截断——交叉比对找出的真实缺陷，已按 OpenAI 语义修复（N 个真实内容 token + 越界哨兵 eos 不计数 + finish=length）。修复后复跑（mt=128）：**EXACT 升至 46.8%（22/47）、收尾差异归零**、真分歧 53.2% 集中于 `<think>` 内（去 think 正文一致率 80.8%）。真分歧特征：平均先吻合 ~67 token、同 prompt 跨长度分歧位置逐字符稳定（确定性近并列翻转，非噪声）；vLLM 自身换 batching 也会翻转（3/7 探针）。**判定：两栈数值等价、近并列敏感，无系统性逻辑分歧。**

**分布级相似度**（dist_similarity，2026-09-09）：12 prompt × 32 步，双端逐位置捕获下一 token 分布（native 走 MC_DEBUG_DUMP 全量 logits、vLLM logprobs=32），在 **293 个公共前缀位置**上对比：**JS 散度 median 4.66e-6 / 余弦 mean 0.9998 / top-1 id 0 失配（293/293）**→ 数值等价的直接证据。全部 4 个翻转点均近并列：2 个是 native bf16 逐位平局（top-2 概率十六进制相同，其中 1 个同时互为对方 top-2）、其余 2 个互为对方 top-2；翻转 margin 均值 0.041 vs 前缀内典型 margin 0.794（**仅为典型值的约 1/19**）。翻转位置与文本级 crosscheck 的首分字符位置逐一吻合——文本级 62.8% 分歧率 = 这些近并列位置的**累积** tie-break 噪声，非计算差异。脚本 `benchmark/scripts/dist_similarity.py`，明细 `benchmark/results/dist_similarity.json`。

**硬件物理极限**：960 GB/s × 4.73GB（BF16 权重）→ 理论上限 ~190 t/s；终值 171 t/s，达上限的 90%。FP8 上限 ~377 t/s，实测 286 t/s（76%）。逐 kernel 分解见 `tpot_breakdown.json`。

## 3. 正确性验证（分层验证体系）

自下而上，上一层跑通前不启动下一层：

1. **单算子测试**（`kernel_ops`，14 组）：每个 kernel 与 CPU 规范参考实现交叉比对（多数 bit-exact）；含 bf16 RNE 已知向量、fp8 解码 known-answer 测试（防「kernel 与测试同错」）、GQA 探针、causal 泄漏哨兵、KV 寻址字面量锁定
2. **CPU 双实现交叉验证**：C++ 参考与 python 权威参考同 id 交叉比对（发现 rope 单头 bug）
3. **分窗逐层 GPU 验证**（`gpu_layerwise`）：显存受限时 embedding+前 N 层 / 后段开窗接力交叉比对 42 层（曾以 2.7GB 完成全模型验证）
4. **端到端 golden**（`real_inference`）：三个 case 与 python 参考逐 token 一致 + 逐张量 cos + 近并列豁免策略（见 §5）
5. **压测**（`soak_test`）：泄漏/隔离/OOM/reset 语义

**近并列豁免策略**（golden_compare.h）：贪心解码在参考实现 top1-top2 差 ≤ max(4×bf16 ulp, 0.25) 处允许单次合法翻转（该步 logits cos 仍须 ≥0.99），翻转后各步输入分叉不计失败；层张量另设混沌放大例外（硬下限 0.99、每 case ≤2、全局 ≤3、链恢复）。

## 4. 缺陷记录（8 个严重缺陷，7 个在端到端验收前修复、第 8 个由对话功能测试发现并修复）

| # | 缺陷 | 发现层 | 根因 |
|---|---|---|---|
| 1 | attention 点积只覆盖 32/128 维 | 静态审查 + 单算子测试独立发现 | 每 warp 只归约 32 维且 KV 位置又拆 4 warp，合并的是残缺分数分布（输出幅度系统性 ≈1/4） |
| 2 | GemmEngine 堆损坏（≥2 次调用后崩溃） | 单算子测试（ASAN 静默未检出） | destroy() 持锁 delete，guard 析构时解锁已释放的 mutex，构成 UB |
| 3 | C++ 参考实现 rope 只旋转 head0 | CPU 交叉验证 | 循环只覆盖向量前 128 元素，15/16 q 头未旋转进 attention |
| 4 | GPU rope q 原地旋转跨 warp 读写竞争 | 精度对齐时的回归测试 | 线程 d 的写与另一 warp 线程 d+64 的读竞争（时序敏感，曾侥幸未暴露） |
| 5 | KV layout_.max_pages 未初始化 | 全量首跑 | 容量护栏始终读到 0 页，prefill 必失败 |
| 6 | GEMM 实参颠倒 ×5 处 | 分窗验证 | 权重/激活指针互换（gate_up 越界崩溃 + 数值垃圾） |
| 7 | 权重 arena 空洞多占 535MB | 分窗验证 | 文件 span 含空洞导致 arena 尺寸计算翻倍 |
| 8 | 采样候选列表写穿 RoPE inv_freq 表 | 对话功能测试（server 级） | 采样 scratch 切分算术与 launcher 实际布局不符：`list_i` 区间与 rope 表重叠，每个采样步把候选 token id 覆盖构建期只读的 64 个 fp32 相位值 → 本 session 一切 forward 的 RoPE 全错且 reset 不重写表。症状：采样第 2 token 起退化、greedy 被连带污染、fresh 进程正常。修复：切分算术 + rope 位级快照钩子 + `sampling_regression` 回归测试（采样→reset→greedy 逐 token 一致等 8 项） |

方法论教训：自洽测试（kernel 与测试共享同一误解）发现不了 #1 的位级错误——known-answer 测试 + 双实现交叉 + 外部权威（python）三件套缺一不可。

## 5. 测量规范与统计口径

- **只信同窗口背靠背 A/B**：共享 GPU 上跨窗口漂移 ±20%；正式数据要求空闲 GPU（util <5%）+ 5×200 token 取中位数（`perf_v2` 内置等待重试）
- **DRAM-honest 微基准**：多份权重拷贝轮换（≥2×L2）防止缓存驻留造成假带宽（曾测出 3.3TB/s 的 L2 假象）
- **数值锁定**：kernel 重构可能翻转 ptxas FMA 收缩选择 → golden 漂移；重大重构先做 SASS 指令序列对比（rope 吸收、attention 流水化均走此流程）
- **逐位一致性分级**：graph-vs-eager 必须逐位；kernel 重分组（如 KV 走 smem）允许 ≤1 ulp 差异但需量化证据 + golden 策略兜底

## 6. 产物索引（benchmark/results/）

> 以下产物均为**本地生成文件、不入 git 库**（.gitignore 排除）；生成脚本见各节与 `benchmark/scripts/`。

| 文件 | 内容 |
|---|---|
| `vllm_baseline.json` | vLLM 同机基线 + 与 native 对照（版本/配置/逐轮） |
| `vllm_crosscheck.json` | greedy 输出充分交叉比对明细（47 prompt × 两种长度；注意：已被 mt=128 修复后复跑覆盖） |
| `similarity_metrics.py` / `dist_similarity.py` | 文本级指标（BLEU/ROUGE-L/编辑相似度）与分布级指标（JS/余弦/top-k）脚本 |
| `dist_similarity.json` | 公共前缀分布级对比明细（293 位置 + 4 翻转点取证） |
| `perf_v2.json` / `perf_bf16.json` | 各优化阶段 TPOT 中位数（BF16/FP8 × 三 ctx，util 注记） |
| `tpot_breakdown.json` | decode 逐段分解（GEMV/attention/lm_head/采样…） |
| `graph_ab.json` | CUDA Graph on/off A/B + golden bit 一致性 |
| `attn_before/after.json` | attention 分块重构前后 |
| `gemv_ab.json` / `gemv_e2e.json` | GEMV 微基准（DRAM-honest）与端到端 |
| `fp8_accuracy.json` | FP8 vs BF16 golden 精度报告 |
| `fp8_e2e.json` | FP8 端到端 A/B |
| `vram_budget.json` | VRAM 预算（BF16/FP8 × ctx × session 数） |
| `perf_sweep.json` / `vllm_sweep.json` | **0-128K 七个长度扫描**（本项目双精度多轮均值 / vLLM 对照；生成：perf_v2 env 扫描 + `merge_perf_sweep.py` / `vllm_bench_sweep.py`） |
| `vllm_concurrent.json` / `vllm_fp8.json` | vLLM 并发基线（K=8 431.9 tok/s）/ vLLM FP8 基线（K=1 4.71ms、K=8 666 tok/s） |
| `concurrency_path_a.json` / `concurrency_path_b.json` | 路径 A（时分并发）/ 路径 B（批解码）K 扫描基准 |
| `corun_probe.json` | prefill∥批解码并行同跑探针（收益 11.6%，机制归因） |
| `batch_gemm_probe.json` / `fp8_rows_m_probe.json` / `fp8_gemm_probe.json` | B0 thin-M 探针（no_cliff）/ F1 多行 gemv-fp8（中止）/ F2a cuBLASLt fp8（中止） |

## 7. 已知局限与后续方向（2026-09-12 复核刷新）

1. ~~**prefill 未优化**~~ ✅ 已解决：flash prefill 落地（§11），全部长度与 vLLM 持平（2048 上下文 −8% 为唯一剩余差距，双方同为 cuBLASLt 上限）
2. **FP8 双拷贝**：显存 7.0GB（>BF16 的 4.8GB）；prefill 仍走 BF16 原始权重（fp8 prefill 专项随 §9 中止未实施）
3. ~~**中段 ctx attention block 饥饿**~~ ✅ 已解决：HMMA flash-decoding 重写后该结构不复存在（§10）
4. **采样 top-k 截断 128**：有 top_p 兜底时质量影响可忽略，纯采样语义略偏
5. ~~**scheduler 单 GPU worker**~~ ✅ 已解决：路径 A 活跃集调度 + 路径 B 批解码（§8），`--max-sessions` 为真并发上限
6. NVFP4（P5）、DSpark（P6）未启动，待 FP8 收益进一步明确后按 Go/No-Go 门槛评估
7. **跨请求 prefix caching 缺失**（vLLM 有）：多轮对话每轮全量重 prefill；session 池不复用 KV 前缀——真实多轮场景 TTFT 的最大改进方向（新立项）
8. **FP8 并发批路径**：两次中止存档 §9（CUDA-core 指令吞吐瓶颈 / cublasLt 12.8 缺 outer-vec scale）；实现代码已保留，库升级即可重启
9. **批解码 greedy-only**：采样流走时分 solo 路径（B1 ABI 语义，批采样为 B2+ 范畴）
10. **MC_GEMV_OFF=1 + CUDA Graph 角落缺陷**：强制 cuBLASLt decode 时多 session 图重放存在跨 session 非确定性（PDL 边对无 griddepcontrol.wait 的 cuBLASLt kernel 写可见性无保证）；仅 debug 环境变量可达，默认 gemv-rows 图不受影响（待立项核查）

---

## 8. 路径 B：批 decode（B1b–B4）实施与结论（2026-09-11）

本节取代 §7-5 所述「`--max-sessions>1` 只封顶不并发」的旧局限：S2 起调度器为单 GPU worker 活跃集 + quantum 交错（路径 A，`concurrency_path_a.json`）；B1b–B4 在其上接入 native 批组（路径 B），greedy 流按 quantum 合并解码。

**实施轨迹**：B1b native 侧 `mc_batch_*` 五函数 ABI（eager 精确 n 步，slot_ids 即成员集，无 bucket/padding；8 槽实测 996 tok/s）→ B3 Rust 侧接线（runtime-sys/sys 镜像 + `BatchEngine`/`BatchGroup` trait + 调度器批 quantum：分拣 / sync_slots / 批步失败回退 solo / `MC_BATCH_OFF` 回退开关 / 批指标进 `/v1/stats`）→ B4 正式两组基准与结论存档（本节）。

**B3 关键发现（单流让位）**：批步（cuBLAS M=n）与 solo（自研 gemv-rows）是两条既有数值路径（step1 logits relL2 ≈ 1.1e-2，`batch_test` A-relL2）；solo 首步写入的 KV 与批步数值路径混合后，近 tie 的 greedy argmax 会**确定性翻转**（探针实测：纯批成员 vs 全 solo 参考 100% 一致；单流 solo 首步后入批则自 token 3 起发散，K=2/K=8 同值复现）。native 场景 B（solo→批→solo 生命周期）因 M=1 批步恰好同路径而未暴露。修复：调度器对「本 quantum 唯一 batchable 且从未让过」的流让位一个空时间片（无 GPU 工作、µs 级、每流至多一次）等批伙伴——B=1 只多等一个空时间片，FFI 调用序不变（K=1 TPOT 实证 +0.62%，见下表）。

### 8.1 正确性（前缀一致率）

调度器批路径 vs 全 solo 参考（独立 session 串行 prefill + 64×`decode_one`，同 prompt 同 seed）：K=2 **128/128 = 100.00%**、K=8 **512/512 = 100.00%**（B3 e2e 配置：prompt 256 / tier 512 / greedy，两次运行复现）。e2e 聚合 798.3 / 800.7 tok/s（B3 验收），B4 复测 785.3 tok/s（3 轮中位，见 8.2）。

### 8.2 两组 K 扫描（B4，`benchmark/results/concurrency_path_b.json`）

方法学：真模型 BF16、greedy、max_new_tokens=64（ignore_eos 语义）、3 轮取中位、每轮前 GPU util ≤5% 才起步、每组预热建满 session 池；solo 组 = 路径 A 同配置**同进程同窗口重测**（背靠背 A/B 规范）。

**official 配置**（prompt 2048 / tier cap 2560，与 `vllm_concurrent.json`、`concurrency_path_a.json` 同配置）：

| K | batch 聚合 (tok/s) | batch TPOT p50 (ms) | batch TTFT p50 (ms) | solo 聚合 (tok/s) | 加速比 |
|---|---|---|---|---|---|
| 1 | 137.0 | 5.623 | 112.9 | 136.8 | 1.001 |
| 2 | 189.8 | 7.104 | 226.8 | 137.4 | 1.381 |
| 4 | 279.3 | 9.906 | 283.2 | 138.0 | 2.023 |
| 5 | 311.1 | 10.758 | 338.2 | —（仅 batch 组） | — |
| 6 | 335.2 | 11.680 | 393.3 | — | — |
| 7 | 353.8 | 12.627 | 450.7 | — | — |
| 8 | **370.9** | 13.582 | 500.9 | 138.9 | **2.671** |

**supplementary 配置**（prompt 256 / tier cap 512，B3 验收 e2e 的校准 workload——432 GO 门槛定义所用的配置）：K=8 batch **785.3**（TPOT 8.197ms）vs solo 168.5 → **4.661×**。

批指标（逐轮落盘）：K=8 batched_steps=69、batch_size_mean=7.41（准入 M-ramp 2→8 后稳态满员）、batch_fallbacks=0（全部 K 值下零回退）；稳态交错断言全 OK（K 流按时间片齐头推进，steady_gap 8.1ms ≈ 单批步步长）；K=8 batch 显存 8484→8510MB（批组 arena ≤26MB，solo 组 8484MB）。

### 8.3 结论

| 配置 | K=8 batch 聚合 | GO 门槛 432 | vs vLLM 431.9 | vs 路径 A 139 | 判定 |
|---|---|---|---|---|---|
| B3 e2e（prompt 256 / tier 512，定义门槛的配置） | **785.3**（B3 验收 798.3–800.7） | **1.82×**（B3 时 1.85×） | 1.818× | 5.65×（同配置 solo 168.5） | **GO** |
| official（prompt 2048 / tier 2560，对外对照配置） | **370.9** | 0.86×（未达字面门槛） | 0.859×（K=4 档 279.3/313.9 = 0.890×） | 2.67×（solo 复测 138.9 ≈ 存档 139.0 ✓） | **如实 NO-GO（归因见下）** |

**official 配置差额归因（修正版，2026-09-11 复核）**：初版归因「批 attention 位置扫描」**有误**。`concurrency_path_b.json` 逐轮字段自证：official K=8 的 `steady_max_gap_ms = 8.1ms`（稳态批步，含全部 KV 扫描）与 native `batch_test` C 记录 8.04ms 同一水平——**稳态聚合 ≈ 988 tok/s**；而 `steady_start_ms = 878.8` / `wall_ms = 1380.4`：墙钟 64% 是**串行准入 prefill 爬坡**（8×2048 tok × ~108ms），TPOT p50 13.582ms 因此被爬坡稀释（成员先入者需等后入者 prefill）。验算：8×108ms 爬坡 + 64 步 × 8.1ms 稳态 = 1389ms ≈ 实测 1380ms ✓。K=4 同型（steady_gap 7.5ms / steady_start 451ms）。差额主因是** host 侧准入编排**而非 attention kernel；补偿方案（chunked prefill + 双流重叠）经并行同跑探针评估后中止，见 §8.6。

**单请求零回归（验收项）**：K=1 batch 组 TPOT p50 **5.6226ms** vs native 直连 5.5882ms → **+0.62%（PASS，<2%）**；solo 组交叉对照 5.6330ms——两组 K=1 均为 solo 路由，单流让位只落在 TTFT（112.9ms）不落 TPOT，与 B3 论证一致。

### 8.4 B2（批步 CUDA Graph 捕获）结论：**不立项**

eager 已 1.82× GO 目标（785.3/432；B3 验收时 1.85×），路径 B 的目标是并发级别而非单步极致；图化预计再 +15–20%（P3 经验：launch 非瓶颈、收益来自 jitter 消除与 hotspot 优化后显现的 3–4% 上浮至图场景），且 eager 精确 n 的动态成员集与静态图形状天然冲突（需 bucket+padding，破坏 B1 的精确 n 语义）。留作可选优化（若 official 配置的 attention 扫描项优化后逼近 vLLM，再评估图化尾部收益）。

### 8.5 已知边界（存档）

1. **greedy-only 入批**（B1 ABI 语义）：非 greedy（temperature>0 且 top_k≠1）恒 solo（路径 A 语义）；采样批化是 B2+ 范畴。
2. **近并列跨组成可翻转契约**：批步与 solo 为两条既有数值路径，近 tie 的 greedy argmax 可翻转（与 §2 vLLM 交叉比对的近并列翻转同类：翻转 margin ≈ 典型值的 1/19）。纯批成员 vs 全 solo 参考 100% 一致；**混合路径（solo 首步后入批）确定性发散**——调度器以单流让位规避（首步前让一个空时间片），让位后再独行的流接受混合路径差异（无法无限等伙伴）。
3. **MC_GEMV_OFF + graph 既有角落缺陷**（§8 前已录于 `concurrency_path_a.json` t2b_anchor）：`MC_GEMV_OFF=1` 强制 cuBLASLt decode 时多 session 交错发散（与 workspace 归属无关的 cuBLASLt decode 控制路径独立缺陷，仅 debug 环境变量可达，默认配置稳定 PASS）。
4. **official 配置爬坡占比**（8.3 修正归因）：K=8 距 vLLM 431.9 的 14% 差额主因是串行准入 prefill 爬坡（879/1380ms）；稳态批步 8.1ms ≈ 988 tok/s 已高于 vLLM 稳态估计（~790）。补偿方案见 §8.6 中止记录。
5. TTFT 尾部随 K 抬升（串行 intake prefill，路径 A 已知性质；batch 组反而略优：K=8 batch 500.9ms vs solo 543.1ms——批步减少了总步数）。

产物：`benchmark/results/concurrency_path_b.json`（环境 / 方法学 / 双 workload × 两组 × 逐 K 原始轮次 / verdict）；生成器 `rust/minicpm-scheduler/examples/concurrency_bench.rs`（`cargo run -p minicpm-scheduler --release --example concurrency_bench`）。

### 8.6 并行同跑探针与 B5（chunked 准入 prefill + 双流重叠）中止结论（2026-09-11，`benchmark/results/corun_probe.json`）

**目的**：消除 8.3 修正归因定位的串行爬坡——把准入 prefill 与批解码在双流并行同跑（prefill 成员流 × 批组流，S1 契约下的合法组合）。

**实测（batch_test.cpp 场景 D，env `MC_BATCH_CORUN_PROBE=1`；3 主轮 + 2 补充轮，哨兵全过）**：
- 并行同跑收益（K=8 全并发）：**11.6%**（T_serial 1341.7ms → T_corun 1186.1ms），投影聚合 = 512/1186ms = **432 tok/s**——与 vLLM 431.9 **同一水平而非超越**
- K 无关性：K∈{2,4,8} 收益全部稳定在 10.9–12.1%（prefill↔prefill 并行同跑良好：+22% 吞吐；瓶颈专属 decode↔prefill 互斥）
- 双向拖慢：prefill +533%（103→653ms）、批步 +129%（8.08→18.53ms，强双峰：并行同跑窗口 ~29ms、窗口结束精确恢复 8.07–8.14ms，暂态无持久污染）
- **机制**：「prefill 计算受限 × 批解码带宽受限应互补」在本栈不成立——批步 73% 时间在 cuBLASLt M=8 HMMA GEMM（计算为主），与 prefill 的 tensor core/SM 需求正面争用。vLLM 同场景自身 benefit 也仅 ~18.2%（1−1186/1450 推算）

**结论：中止（11.6% < 15% 线）**——即便细粒度 chunked prefill（推断上限 ~18%），也只到 vLLM 同一水平。**若未来重启，方向是降低批解码的计算占用（如 fp8 批 GEMM 路径），而非更聪明的并行同跑调度。**

## 9. FP8 全链路服务专项与中止（2026-09-12，`benchmark/results/vllm_fp8.json` / `fp8_rows_m_probe.json` / `fp8_gemm_probe.json`）

**目标（公平对比条件：我们 FP8 vs vLLM FP8，official 场景）**：vLLM FP8（dynamic weight-only）实测 K=1 TPOT **4.71ms**、K=8 聚合 **666.1 tok/s**。我们 FP8 solo 已在单请求上胜出（3.21ms，+32%）；缺口在 fp8 批解码（不存在）与 fp8 prefill（现走 BF16 原始权重）。

**F1 多行 gemv-fp8（CUDA-core 支线）——正确性满分、性能中止**：逐行 bit == solo（33/33 配置 bitDiff=0，0 spill）但步级 GEMM 6.4ms vs 2.9ms 线。根因：bf16→f32 转换指令吞吐瓶颈（PRMT+IMAD+SHF 3 指令/对，M=8 时非 FMA 指令 2.4× 于 FMA，发射率 ~1.9/clk）；决定性对照——**bf16 cuBLASLt M=8 仅 6.09ms，权重减半也没跑赢**（tensor core 使 M 近乎免费，M8/M1=1.0-1.17）。按预案升级 tensor-core 支线。

**F2a cuBLASLt fp8 W8A8——引擎无损、门槛中止**：quantizer 与独立 CPU 编码器逐字节一致（~180K 元素 qDiff=0），量化感知 relL2 2.3e-3；引擎大 shape 达 856 GB/s（≥下限）。但步级全路径 eager 4.26ms / CUDA Graph 生产形态 3.82ms，均 >3.0ms 线：thin-M 小 shape 逐调用开销（quant+post 两小核 + thin-M fp8 尾部，~7-8µs/枚）+ **host cublasLt 12.8 缺 `OUTER_VEC_32F`**（被迫 per-tensor + post_scale 补偿路径；venv cu13 头有该枚举，运行时探测自动启用但本机不可达未验证）。连锁算术：step ~6.0ms → official 聚合 ~633 < 666，即使 F3（fp8 prefill，M=2048 预览 171 TFLOPS ≈ e4m3 峰值 90%）与 F4（并发准入）全落地也无法反超——**FP8-batch 线整体中止，F2b/F3/F4 终止**。

**中止后的存量成果**：FP8 单请求 TPOT **3.21 vs 4.71ms（+32%）**保持；W8A8 实现代码（quant.cu + gemm_fp8 引擎 + scale 模式探测）已落地且经基准校验，M=2048 方向 171 TFLOPS——**未来 cublasLt 升级支持 outer-vec scale（或自研 fp8 mma）后可直接重启 F2b-F4**，届时预期步级 GEMM ~2.6ms → 聚合 ~700+。F1 的多行 kernel（逐位基准全绿）一并保留。

## 10. 上下文扩展与 decode 优化专项（2026-09-09 ~ 09-11；时间上先于 §8/§9）

**起点**：0-128K 七个长度扫描（`perf_sweep.json`/`vllm_sweep.json` 的前身）暴露 ≥32K 落后 vLLM（32K −12% / 64K −28% / 128K −55%）。**前置改造**：prefill 分块（8448/块）+ 激活 scratch 缩减——128K 可运行的前提（12288 token 自一致性：分块 vs 单次逐 token 一致）。

**优化链四步（128K BF16 TPOT 轨迹：20.22 → 12.36 ms，−39%）**：
1. **两级 merge + CHUNK 分级**（20.22 → 18.28）：merge_a/merge_b 把 16-block 归约扩到 512-block；chunk 64/128/256 按会话级别。收获 −9.6%，且定位出 merge 仅占缺口 ~2ms（架构师预估 4-5ms 偏高 2×）
2. **attn_kv_v2**（→ 14.11）：lane 连续 4 维映射（LDS.64 向量化）+ cp.async 双缓冲流水 + PDL 预取前移；SIMT 路径的极限。v2.1 mbarrier 无锁步流水：实现完整、逐位 0 ulp、但三个长度性能全平（真正节奏器是块级 free 门与访存系统）——**零收益中止**，opt-in 存档（`MC_ATTN_V21_ON`）
3. **HMMA flash-decoding**（→ 12.36，生产默认）：QK/PV 全部 mma.m16n8k16 bf16、C→A fragment 寄存器直通（FA2 式）、272B 行距（保 cp.async 16B 对齐 + ldmatrix 无 bank conflict）。分五阶段落地（H1 QK 交叉比对 → H2 主循环 → H3 phase0 → H4 接线 → H5 验证通过后转为生产默认，`MC_ATTN_HMMA_OFF` 回退）。关键工程发现：mma A-fragment 寄存器序与常见资料相反（R1=(row+8,2c)，随机数据才能暴露）；未初始化 smem 行 × tensor core 0×NaN=NaN 的生产栈污染链（单测全绿、端到端必炸，尾 stage 显式清零修复）
4. **验证体系**：逐位基准比对（vs SIMT 参考输出 bitDiff=0）、CPU bf16-P 证明 ≤2.5e-7、输出 ulp 直方图（median=0）、greedy 一致性（近并列规范）、graph 节点数核对、13/13→14/14 ctest 全绿

**终局**：BF16 在 7/7 个上下文长度上反超 vLLM（+5.7% ~ +17.9%，`perf_sweep.json` 多轮均值）。512-chunk 级别 A/B 退化 → 默认关（`MC_ATTN_CHUNK512_ON` opt-in）。

## 11. prefill flash-HMMA 化（2026-09-11，P1-P4）

**起点**：旧 prefill attention 为 O(T²) 逐 token 串行 kernel（grid=(16,tokens)），128K ~340 tok/s vs vLLM 3.65k（10× 差距）。**立项洞见**（用户指出）：先修 prefill 还能把基准测试回路从每个长度约 90 分钟压缩到约 10 分钟——测试基础设施的投资回报此前被低估。

**四阶段**：P1 正确性原型（16-token q-tile × HMMA，vs attn_ref relL2 1.7-2.3e-3，kernel 级 13.7-21.3×）→ P2 cp.async 双缓冲流水（kernel −8~11%，输出逐位不变）→ P3 split-KV 小 T 加速（S_kv(T) 级别 + 独立 prefill_merge，session arena +40MB）+ 生产接线 → P4 门槛结论（2048 ≥15k ✓ / 8192 ≥14k ✓ / 32768 ≥9k ✓ / 65536 ≥6k ✓ / 130816 3678 tok/s = 100.8% vLLM 未达 6k GO 线）。

**默认翻转复核**：P4 二元判定规则未覆盖「未过 GO 亦未触发中止」的中间区，fixer 按字面保守 opt-in；复核结论——flash 每个长度严格优于旧路径（2.66×~17.1×）、TTFT 全部长度 ≥ vLLM 92-100.8%、正确性全绿，保守默认让用户停留在 340 tok/s 的旧路径与判定规则初衷相悖 → **翻转默认**（`MC_PREFILL_FLASH_OFF` 回退，结论数字存档 kernels.h）。

**e2e 成果**：2048：7.1k → **19.0k**（2.66×）；8192：2.3k → **18.0k**（7.95×）；32768：590 → **10.1k**（17.1×）；130816：340 → **3.68k**（10.8×，与 vLLM 持平）。128K 更大 q-tile（32/64 行）可推过 6k GO 线，列为后续增量。


