//! `minicpm-scheduler`: single-GPU worker thread + 活跃集（active set）+
//! quantum 交错调度 + session 池（plan §4.5 线程模型；S2 并发推理路径 A）
//! + **B3 批 quantum**（greedy 流经引擎批组合并解码，路径 B）。
//!
//! ```text
//! HTTP/SSE Tokio Workers ──> SchedulerHandle::generate()  (async side)
//!                                  |  std::sync::mpsc (FIFO 等待队列)
//!                                  v
//!                        GPU Worker Thread (std::thread)
//!                            idle: Vec<Box<dyn EngineSession>>   空闲池
//!                            active: VecDeque<ActiveGen>         活跃生成集
//!                                  |
//!                                  v
//!                       solo: session.decode_one   batch: BatchGroup::step
//!                            Native / Mock Runtime
//! ```
//!
//! Design points（S2 起）:
//! * One dedicated OS thread owns the GPU and the session pool; tokenization
//!   and SSE framing stay on the async side, blocking GPU calls never run on
//!   a Tokio worker.
//! * **活跃集 + quantum 交错**：多请求并发推进（每 quantum 轮转各走一步
//!   decode），新请求的 prefill 在准入时同步执行（一步插队），不再排队
//!   等活跃流整体跑完。channel 本身就是 FIFO 等待队列。
//! * **B3 批 quantum**：每 quantum 把活跃流分拣为 batchable（greedy 配置
//!   且无 pending_tok）与 solo 两类；≥2 条 batchable 且引擎支持批
//!   （`Engine::as_batch`，MockEngine 不支持 → 全 solo 构造性零回归）时
//!   经批组合并解码一步（`BatchGroup::step`），token 经 `feed_token` 走
//!   与 solo 完全相同的判停/发放语义；任何批段失败 → 本量子全员回退
//!   solo（`batch_fallbacks_total`，worker 不崩）。`MC_BATCH_OFF=1` env
//!   （进程内静态读一次）或 `SchedulerConfig::batching=false` 永不批——
//!   与 S2 行为逐调用等价（零回归逃生门）。
//! * The async side gets a `tokio::sync::mpsc` receiver of [`GenEvent`]s and
//!   a [`CancelToken`]; the worker checks the cancel flag before every
//!   decode step. All receivers being dropped is also treated as cancel.
//! * Token 事件用 **try_send**（非阻塞）：慢消费者只延迟自己（token 暂存
//!   `pending_tok`，本量子让步），不阻塞整条 worker、不影响别的流。
//! * Sessions are created lazily up to `max_sessions`, `reset()` on take
//!   from and return to the pool (allocations are reused, plan §5.4), and
//!   returned even when a job fails or is cancelled.
//! * Stop **strings** are *not* handled here — the scheduler only knows stop
//!   token ids ([`EOS_IDS`]); string-level truncation lives in the server
//!   (StopState + StreamDecoder over the token stream).

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicI64, Ordering};
use std::sync::mpsc::{Receiver, Sender};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;

use tokio::sync::mpsc::error::TrySendError;

use minicpm_generation::{FinishReason, Usage};
use minicpm_metrics::MetricsRegistry;
use minicpm_runtime::{
    BatchEngine, BatchGroup, BatchGroupOptions, Engine, EngineSession, GenerationConfig,
};

/// 批组容量上限（native ABI：`mc_batch_options_t.max_slots ≤ 8`）。
pub const MAX_BATCH_SLOTS: usize = 8;

/// End-of-sequence token ids the worker stops on (FinishReason::Stop).
///
/// Source: `models/minicpm5-2b/generation_config.json` →
/// `"eos_token_id": [1, 130073]` (`</s>` and `<|im_end|>`).
pub const EOS_IDS: [i32; 2] = [1, 130_073];

/// Events streamed to the async side for one generation job.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GenEvent {
    /// A newly sampled token id (may be an eos id; the consumer decides).
    Token(i32),
    /// Generation finished.
    Done { reason: FinishReason, usage: Usage },
    /// The job failed (engine error); the session is still returned to the pool.
    Failed(String),
}

/// One queued generation job.
#[derive(Debug, Clone)]
pub struct GenerationJob {
    /// Prompt token ids (`[0, vocab_size)`), already template-rendered and
    /// encoded by the caller.
    pub prompt_ids: Vec<i32>,
    /// Sampling / stopping parameters (string stops excluded by design).
    pub config: GenerationConfig,
}

/// Cooperative cancellation handle; the worker polls the flag between tokens.
#[derive(Clone, Debug)]
pub struct CancelToken {
    flag: Arc<AtomicBool>,
}

impl CancelToken {
    pub fn cancel(&self) {
        self.flag.store(true, Ordering::SeqCst);
    }

    pub fn is_cancelled(&self) -> bool {
        self.flag.load(Ordering::SeqCst)
    }
}

struct Job {
    inner: GenerationJob,
    events: tokio::sync::mpsc::Sender<GenEvent>,
    cancel: Arc<AtomicBool>,
    /// 提交时刻（准入超时判定 + worker 侧 TTFT 的排队锚点）。
    enqueued_at: Instant,
}

/// worker 行为配置（S2 引入；后续演进 knobs 挂这里）。
#[derive(Debug, Clone)]
pub struct SchedulerConfig {
    /// session 池容量上限（idle + active 的 session 总数 ≤ 此值），也即
    /// 最大并发活跃流数。
    pub max_sessions: usize,
    /// 每个生成 job 的事件 channel 容量（默认 1024）。调小可压出
    /// try_send `Full` → `pending_tok` 暂存路径（测试用）。
    pub event_channel_capacity: usize,
    /// 等待队列容量上限（S3 准入控制）：提交时 `waiting ≥ queue_limit`
    /// 直接拒绝（[`SubmitError::QueueFull`] → HTTP 429）。默认 32；
    /// **0 = 一律立即拒绝**。
    pub queue_limit: usize,
    /// 排队超时（S3）：超时判定在**准入时**——job 在等待队列里滞留超过
    /// 此时长即发 `Failed("queue timeout")`，不创建 session、不 prefill
    ///（等待期失效的 job 不占 GPU）。默认 120s。
    pub queue_timeout: std::time::Duration,
    /// B3 批 decode 开关：false = 永不批（全 solo，与 S2 行为逐调用等价，
    /// 零回归逃生门）。默认值 = **未设 `MC_BATCH_OFF=1`**（env 进程内静态
    /// 读一次；显式置 false 供测试/运维确定性关批）。
    pub batching: bool,
}

/// `MC_BATCH_OFF` env 是否生效（"1" 或 "true"）——进程内静态读一次
///（OnceLock 缓存；`SchedulerConfig::default` 的 `batching` 取反源）。
fn batch_env_off() -> bool {
    static OFF: OnceLock<bool> = OnceLock::new();
    *OFF.get_or_init(|| {
        let v = std::env::var("MC_BATCH_OFF").unwrap_or_default();
        v == "1" || v == "true"
    })
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            max_sessions: 1,
            event_channel_capacity: 1024,
            queue_limit: 32,
            queue_timeout: std::time::Duration::from_secs(120),
            batching: !batch_env_off(),
        }
    }
}

/// [`SchedulerHandle::try_generate`] 的拒绝原因。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubmitError {
    /// 等待队列已满（`waiting ≥ queue_limit`；`queue_limit == 0` 时恒拒）。
    /// 调用方应回 429 + Retry-After。
    QueueFull,
}

/// Scheduler 与其全部 handle 克隆共享的内核状态。
///
/// 生命周期（S3 有序关停）：最后一个 `Scheduler`/`SchedulerHandle` drop →
/// `SchedulerInner::drop` → 先置空 `tx`（关闭 job channel）→ worker 循环
/// 在 channel 关闭且 active 清空后退出（退出前给活跃 job 发终态事件、
/// retire 全部 session——**session 析构仍发生在 worker 线程上**，先于
/// join 返回，根除「exit() 期间 worker 拆 session 与主线程静态析构竞态」
/// 的退出期 SIGSEGV）→ `join()` 等 worker 结束。
struct SchedulerInner {
    /// `Option` 仅服务于 [`Drop`]：置 None 即关闭 channel（最后一个引用
    /// drop 时触发）；存活期恒 `Some`。
    tx: Option<Sender<Job>>,
    /// 关停标志（Drop 先置 true 再关 channel）：worker 在 active 已满、
    /// 不能 pop channel 的档口用它非破坏性地感知关停。
    closing: Arc<AtomicBool>,
    /// worker 线程 join 句柄（Drop 时 take + join；worker 自身不持有
    /// `Arc<SchedulerInner>`，避免自引用导致永不退出）。
    join: Mutex<Option<std::thread::JoinHandle<()>>>,
    /// 等待队列计数（handle 提交 +1 / worker 准入 −1；与 worker 共享）。
    /// `queue_limit` 守卫在 +1 之前：被拒的提交不占队列位。
    waiting: Arc<AtomicI64>,
    queue_limit: usize,
    max_sessions: usize,
    event_channel_capacity: usize,
}

impl Drop for SchedulerInner {
    fn drop(&mut self) {
        // 1) 先置关停标志（active 满档口的非破坏性检查用），再关闭最后
        //    一个发送端 → worker 侧 recv/try_recv 返回 Disconnected。
        self.closing.store(true, Ordering::SeqCst);
        self.tx = None;
        // 2) 等 worker 有序退出（channel 关闭 → 活跃 job 终态 + session
        //    retire 全在 worker 线程上完成）→ join 返回后本结构才真正释放。
        if let Some(join) = self.join.lock().unwrap().take() {
            let _ = join.join();
        }
    }
}

/// FIFO job submission handle; cloneable and shared across HTTP workers.
#[derive(Clone)]
pub struct SchedulerHandle {
    inner: Arc<SchedulerInner>,
}

impl SchedulerHandle {
    /// Queue a job; returns the event stream and its cancel token.
    ///
    /// 兼容入口（不可失败）：内部走 [`try_generate`](Self::try_generate)；
    /// 若命中 `QueueFull` 仍照常入队（现语义——`std::sync::mpsc::send` 在
    /// worker 存活期不会失败；默认 queue_limit=32 下几乎不触发，新代码
    /// 应使用 try_generate 以获得准入拒绝）。
    pub fn generate(
        &self,
        job: GenerationJob,
    ) -> (tokio::sync::mpsc::Receiver<GenEvent>, CancelToken) {
        if self.admit().is_err() {
            // 超额兜底：waiting 同样 +1 保持 worker 准入 −1 的平衡。
            self.waiting_add_unchecked();
        }
        self.submit_raw(job, self.inner.event_channel_capacity)
    }

    /// Same as [`generate`](Self::generate), but with an explicit event
    /// channel capacity (small capacities exercise the `pending_tok`
    /// slow-consumer path deterministically in tests).
    pub fn generate_with_capacity(
        &self,
        job: GenerationJob,
        capacity: usize,
    ) -> (tokio::sync::mpsc::Receiver<GenEvent>, CancelToken) {
        if self.admit().is_err() {
            self.waiting_add_unchecked();
        }
        self.submit_raw(job, capacity)
    }

    /// 带准入控制的提交：`waiting ≥ queue_limit` → [`SubmitError::QueueFull`]
    ///（不 +1、不占队列位）；否则 +1 后入队，worker 准入时 −1。
    pub fn try_generate(
        &self,
        job: GenerationJob,
    ) -> Result<(tokio::sync::mpsc::Receiver<GenEvent>, CancelToken), SubmitError> {
        self.try_generate_with_capacity(job, self.inner.event_channel_capacity)
    }

    /// [`try_generate`](Self::try_generate) 的显式 channel 容量版。
    pub fn try_generate_with_capacity(
        &self,
        job: GenerationJob,
        capacity: usize,
    ) -> Result<(tokio::sync::mpsc::Receiver<GenEvent>, CancelToken), SubmitError> {
        self.admit()?;
        Ok(self.submit_raw(job, capacity))
    }

    /// 准入计数（CAS 原子的「守卫判定 + waiting +1」）；超限未 +1。
    fn admit(&self) -> Result<(), SubmitError> {
        let limit = self.inner.queue_limit;
        loop {
            let cur = self.inner.waiting.load(Ordering::Acquire);
            // queue_limit == 0：一律立即拒绝（立即 429 语义）。
            if limit == 0 || cur >= limit as i64 {
                return Err(SubmitError::QueueFull);
            }
            if self
                .inner
                .waiting
                .compare_exchange(cur, cur + 1, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                self.mirror_queue_depth();
                return Ok(());
            }
        }
    }

    /// 兼容路径超额兜底的 +1（不经守卫；与 worker 的准入 −1 配对）。
    fn waiting_add_unchecked(&self) {
        self.inner.waiting.fetch_add(1, Ordering::AcqRel);
        self.mirror_queue_depth();
    }

    /// 把权威计数镜像到全局 metrics gauge（/v1/stats 快照源）。
    fn mirror_queue_depth(&self) {
        MetricsRegistry::global().queue_depth.store(
            self.inner.waiting.load(Ordering::Relaxed),
            Ordering::Relaxed,
        );
    }

    fn submit_raw(
        &self,
        job: GenerationJob,
        capacity: usize,
    ) -> (tokio::sync::mpsc::Receiver<GenEvent>, CancelToken) {
        let (tx, rx) = tokio::sync::mpsc::channel(capacity);
        let flag = Arc::new(AtomicBool::new(false));
        // The worker owns the receiving end for the whole process lifetime,
        // so send cannot fail silently in practice; if it somehow does (worker
        // gone), the client just sees the stream end immediately.
        let _ = self
            .inner
            .tx
            .as_ref()
            .expect("tx alive while handle alive")
            .send(Job {
                inner: job,
                events: tx,
                cancel: Arc::clone(&flag),
                enqueued_at: Instant::now(),
            });
        (rx, CancelToken { flag })
    }

    /// 当前等待队列深度（提交了但未被 worker 准入的 job 数）。
    pub fn queue_depth(&self) -> i64 {
        self.inner.waiting.load(Ordering::Acquire)
    }

    /// 池容量上限（idle + active 总数上限；/v1/stats `max_sessions` 用）。
    pub fn max_sessions(&self) -> usize {
        self.inner.max_sessions
    }
}

/// Owns the GPU worker thread's job-channel sender.
///
/// S3 起不再 detached：`Scheduler` 与全部 `SchedulerHandle` 克隆共享
/// [`SchedulerInner`]，最后一个 drop 时关闭 channel 并 **join** worker
///（有序关停，见 `SchedulerInner::drop`）。
pub struct Scheduler {
    inner: Arc<SchedulerInner>,
}

impl Scheduler {
    /// Spawn the single GPU worker thread with a session pool of up to
    /// `max_sessions` lazily created sessions（兼容入口：内部走
    /// [`with_config`](Self::with_config) 默认值）。
    pub fn new(engine: Arc<dyn Engine>, max_sessions: usize) -> anyhow::Result<Self> {
        Self::with_config(
            engine,
            SchedulerConfig {
                max_sessions,
                ..SchedulerConfig::default()
            },
        )
    }

    /// 带 [`SchedulerConfig`] 的完整构造入口。
    pub fn with_config(engine: Arc<dyn Engine>, config: SchedulerConfig) -> anyhow::Result<Self> {
        assert!(config.max_sessions >= 1, "max_sessions must be >= 1");
        assert!(
            config.event_channel_capacity >= 1,
            "event_channel_capacity must be >= 1"
        );
        let (tx, rx) = std::sync::mpsc::channel::<Job>();
        let waiting = Arc::new(AtomicI64::new(0));
        let closing = Arc::new(AtomicBool::new(false));
        let worker_waiting = Arc::clone(&waiting);
        let worker_closing = Arc::clone(&closing);
        let queue_timeout = config.queue_timeout;
        let max_sessions = config.max_sessions;
        let batching = config.batching;
        let join = std::thread::Builder::new()
            .name("gpu-worker".into())
            .spawn(move || {
                worker_loop(
                    engine,
                    max_sessions,
                    queue_timeout,
                    batching,
                    worker_waiting,
                    worker_closing,
                    rx,
                )
            })?;
        Ok(Self {
            inner: Arc::new(SchedulerInner {
                tx: Some(tx),
                closing,
                join: Mutex::new(Some(join)),
                waiting,
                queue_limit: config.queue_limit,
                max_sessions: config.max_sessions,
                event_channel_capacity: config.event_channel_capacity,
            }),
        })
    }

    /// A cloneable submission handle.
    pub fn handle(&self) -> SchedulerHandle {
        SchedulerHandle {
            inner: Arc::clone(&self.inner),
        }
    }
}

// ============================================================================
// worker：活跃集 + quantum 协议
// ============================================================================

/// 活跃生成的阶段。S2 只有 `Decoding`（prefill 在准入期同步一步完成）；
/// enum 形状留给未来 chunked prefill（多量子推进的长 prefill）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Phase {
    Decoding,
}

/// 一条活跃流：job 的全部可变状态 + 独占的 session（活跃期内归此结构
/// 所有，slot 租借语义——retire 时经 reset 归还 idle 池）。
struct ActiveGen {
    session: Box<dyn EngineSession>,
    config: GenerationConfig,
    events: tokio::sync::mpsc::Sender<GenEvent>,
    cancel: Arc<AtomicBool>,
    /// prompt 长度（usage 记账；prefill 完成后 prompt_ids 即弃）。
    prompt_tokens: u64,
    /// 已生成 content token 数（eos 不计）。
    generated: u64,
    /// 上轮 try_send `Full` 暂存的 token：下量子先补送，成功才 decode。
    pending_tok: Option<i32>,
    /// 提交时刻（S3：Job 传入；排队超时判定 + worker 侧 TTFT 的排队锚点）。
    enqueued_at: Instant,
    /// 首 token 时刻（worker 侧 TTFT/TPOT 口径，retire 时上报）。
    first_token_at: Option<Instant>,
    /// B3 批组槽位：入组时分配（`BatchGroup::add`），出组（quantum 边界
    /// 降级 solo / retire）时清空。`None` = 不在组内（solo 语义）。
    slot_id: Option<u32>,
    /// B3 独行让位：本 quantum 唯一的 batchable 流且从未让过 → 不步进、
    /// 回队尾等一量子的潜在批伙伴（见 worker_loop「独行让位」注释）。
    batch_held: bool,
    #[allow(dead_code)] // 未来 chunked prefill 分相位；S2 恒 Decoding
    phase: Phase,
}

/// step() 的三种结果。
enum StepResult {
    /// 推进成功，job 回队尾。
    Continue,
    /// 慢消费者（channel 满）：token 已暂存 `pending_tok`，本量子让步
    ///（不取消——慢消费者只延迟自己），job 回队尾。
    Yield,
    /// job 终局（自然完成 / 取消 / 失败）。
    Done(Finish),
}

/// job 终局分类（事件与指标口径见 retire）。
enum Finish {
    Stop,
    Length,
    Cancelled,
    Failed(String),
}

/// retire 的发送模式。
enum RetireMode {
    /// 常规服务期：终态事件 blocking_send（counted ⇒ delivered）。
    Serving,
    /// 有序关停期：终态事件 try_send best-effort——接收端可能还在但
    /// 可能不读，绝不为关停阻塞 worker。
    Shutdown,
}

/// GPU worker 主循环：活跃集 + quantum 交错调度（B3：quantum 内分拣
/// solo / batch 两段）。
///
/// 每轮（quantum）：
/// 1. 准入：active 空 → 阻塞 `recv`；否则 `active.len() < max_sessions`
///    时 `try_recv` 至多 1 个新 job（容量守卫必须在 try_recv 之前——弹出
///    的 job 必须立即准入；channel 本身就是等待队列，FIFO 语义天然成立）。
/// 2. 分拣：逐条 pop_front——batchable（greedy 配置 && 无 pending_tok &&
///    批开 && 引擎支持批 && 组容量未满）进批段，其余进 solo 段（已持批
///    槽者先出组——ABI：成员在组内禁止 solo 使用）。
///    独行让位：本 quantum 唯一的 batchable 且从未让过 → 不步进、回队尾
///    一量子等批伙伴（避免 solo 首步与批步的数值路径混合破坏批/独跑的
///    greedy 前缀一致性；让位量子无 GPU 工作，B=1 只多等一个空量子）。
/// 3. solo 段（与 S2 逐事件等价）：pop→step→push_back；Done → 终态+retire。
/// 4. 批段（≥2 条 batchable）：decode 前再查取消 → 组惰性创建 →
///    sync_slots（add 对齐成员集）→ `group.step` 一次推进全员 → 逐成员
///    `feed_token`（判停/发放语义与 solo 完全一致）；任何失败 → 本量子
///    全员出组回退 solo step（`batch_fallbacks_total`，worker 不崩）。
/// 5. 组空 → destroy（释放组 arena；档位恒同——单引擎单模型一档，无
///    「档位变了重建」分支）。
///
/// 有序关停（S3）：channel 关闭（最后一个 handle drop）即进入关停——不再
/// 等活跃 job 自然完成，立即逐个发终态事件（try_send best-effort，接收端
/// 可能还在但绝不为关停阻塞）+ retire 全部 session，然后退出循环。session
/// 析构（idle 池 drop）发生在 worker 线程上、先于 join 返回——主线程的
/// exit()/静态析构由此不会与 CUDA teardown 竞态。
///
/// 四条不变量（S2 起 as-designed，B3 修订；任何改动必须保持）：
/// ① 所有 FFI/native 调用永远来自同一 OS 线程（本 worker 线程独占
///    engine、全部 session 与**批组**——B3 的组 create/add/remove/step/
///    destroy 全在 worker 线程上；async 侧只经 channel 交互）。
/// ② 每 session 的 native 调用序恒为 reset→prefill→decode_one*（solo
///    成员）或 reset→prefill→(批步成员)（batch 成员：add→step*→remove，
///    出组后的 solo 调用经 remove 的事件链保持流内严格串行序——native
///    侧确定性承诺的前提）。
/// ③ 每 job 恰 2 次 reset（准入 + retire）——既有测试
///    `consecutive_jobs_reuse_session_with_resets` 的 reset_calls==4（两
///    job）断言必须原样保持。**合法例外（S3）**：排队超时的 job 0 次
///    reset（从未持有过 session，池语义不受影响）；关停期被终态的 job
///    照常走 retire（reset 不缺）。
/// ④（B3 新增）成员在批组内期间绝无 solo 调用/归还/析构——分拣降级与
///    retire 都先 `group.remove`（清忙态）；组内 session 归还池前必已
///    出组（`slot_id` 簿记 + quantum 边界 sync_slots 对齐成员集）。
///
/// 单请求路径的调用序列与旧串行 FIFO 逐调用一致
///（recv→reset→prefill→(decode/send)*）→ 零回归；B3 下单活跃流（B=1）
/// 批段恒 <2 条 → 全 solo，逐调用序列与 S2 完全相同。
#[allow(clippy::too_many_arguments)]
fn worker_loop(
    engine: Arc<dyn Engine>,
    max_sessions: usize,
    queue_timeout: std::time::Duration,
    batching: bool,
    waiting: Arc<AtomicI64>,
    closing: Arc<AtomicBool>,
    rx: Receiver<Job>,
) {
    let metrics = MetricsRegistry::global();
    let mut idle: Vec<Box<dyn EngineSession>> = Vec::new();
    let mut active: VecDeque<ActiveGen> = VecDeque::new();
    // B3 批段状态：引擎批能力（NativeEngine 有 / MockEngine 无）+ 配置开关
    //（MC_BATCH_OFF env 已在 SchedulerConfig::default 折算为 false）；
    // 组惰性创建（首次出现 ≥2 条 batchable 的 quantum）。
    let batch_engine: Option<&dyn BatchEngine> = engine.as_batch();
    let batching_on = batching && batch_engine.is_some();
    let max_batch_slots = max_sessions.min(MAX_BATCH_SLOTS);
    let mut group: Option<Box<dyn BatchGroup>> = None;
    loop {
        // ---- 准入（每量子至多 1 个新 job）----
        if active.is_empty() {
            // 无活跃流：阻塞等待（空闲零轮询；单请求行为与旧 recv 循环一致）。
            // 所有 handle drop → recv Err（Disconnected）→ 关停。
            match rx.recv() {
                Ok(job) => intake(
                    &engine,
                    &mut idle,
                    &mut active,
                    max_sessions,
                    queue_timeout,
                    &waiting,
                    job,
                ),
                Err(_) => break,
            }
        } else if active.len() < max_sessions {
            // 容量守卫在 try_recv 之前：弹出的 job 必须立即准入（idle 有则
            // 复用、无则新建，session 总数不超上限）。
            match rx.try_recv() {
                Ok(job) => intake(
                    &engine,
                    &mut idle,
                    &mut active,
                    max_sessions,
                    queue_timeout,
                    &waiting,
                    job,
                ),
                Err(std::sync::mpsc::TryRecvError::Empty) => {}
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    // channel 已关：立即有序关停——活跃 job 终态 + retire。
                    shutdown_active(&mut active, &mut idle, max_sessions, &mut group);
                    break;
                }
            }
        } else if closing.load(Ordering::SeqCst) {
            // active 已满：不能 pop channel（弹出的 job 必须立即准入），
            // 用关停标志非破坏性地感知 channel 关闭。
            shutdown_active(&mut active, &mut idle, max_sessions, &mut group);
            break;
        }

        // ---- quantum 推进：分拣 → 独行让位 → solo 段 → 批段 ----
        // 分拣（快照长度 n；round-robin 语义：全体各走一步）。
        let n = active.len();
        let mut solo: Vec<ActiveGen> = Vec::with_capacity(n);
        let mut batch: Vec<ActiveGen> = Vec::new();
        for _ in 0..n {
            let mut gen = active
                .pop_front()
                .expect("quantum iterates non-empty active");
            let want_batch = batching_on
                && batch_engine.is_some()
                && gen.pending_tok.is_none() // 慢消费者先补送（solo 路径）
                && gen.config.is_greedy(); // B1 greedy-only（native 同判据）
            if want_batch && batch.len() < max_batch_slots {
                // 入批（batch_held 不在此清：独行让位「每流至多一次」，
                // 由 hold 分支持 true 后终身保持——否则孤立流会反复让位
                // 永不推进）。
                batch.push(gen);
            } else {
                // 出组（已持槽者）——ABI：成员在组内禁止 solo 使用；
                // remove 在两次 batch_step 之间（quantum 边界）调用。
                demote_from_group(&mut gen, &mut group);
                solo.push(gen);
            }
        }

        // 独行让位（B3 数值一致性关键）：本 quantum 唯一的 batchable 流且
        // 从未让过 → 不步进、回队尾一量子，等潜在批伙伴。动机：solo 首步
        // 经 M=1 gemv 数值路径写 KV，其后批步（cuBLAS M=n）在其上推进——
        // 两条既有数值路径的 bf16 舍入差（relL2 ~1e-2）在近 tie 的 greedy
        // argmax 上可翻转，破坏「批 vs 全 solo 参考」的前缀一致性（native
        // batch_test 场景 B 的 M=1 批步掩盖了这一点）。让位量子无任何
        // GPU 工作（µs 级）：B=1 只多等一个空量子，每 session 的 FFI 调用
        // 序不变。**每流至多让一次**（batch_held 终身保持）：仍独行则下量
        // 子起照常 solo——孤立流不会反复让位降速，也让过的流此后再独行
        // 时立即 solo（接受混合路径的数值差异——无法无限等伙伴）。
        if batch.len() == 1 && batching_on && batch_engine.is_some() {
            let mut gen = batch.pop().unwrap();
            if !gen.batch_held {
                gen.batch_held = true;
                active.push_back(gen); // 本量子不步进
            } else {
                // 已让过一次仍独行：solo（B=1 常态路径）。
                demote_from_group(&mut gen, &mut group);
                solo.push(gen);
            }
        }

        // solo 段（与 S2 逐事件等价：FIFO 相对序、step 内取消检查/pending
        // 补送、Continue/Yield 回队尾、Done → retire）。
        for mut gen in solo {
            match step(&mut gen) {
                StepResult::Continue | StepResult::Yield => active.push_back(gen),
                StepResult::Done(finish) => retire(
                    gen,
                    finish,
                    &mut idle,
                    &active,
                    max_sessions,
                    RetireMode::Serving,
                    &mut group,
                ),
            }
        }

        // 批段：decode 前取消检查（step() 第 1 步的批路径等位点）。
        if batch.len() >= 2 {
            let mut i = 0;
            while i < batch.len() {
                if batch[i].cancel.load(Ordering::SeqCst) {
                    let gen = batch.remove(i);
                    retire(
                        gen,
                        Finish::Cancelled,
                        &mut idle,
                        &active,
                        max_sessions,
                        RetireMode::Serving,
                        &mut group,
                    );
                } else {
                    i += 1;
                }
            }
        }

        if batch.len() >= 2 {
            if let Some(be) = batch_engine {
                // ---- 合批尝试：sync_slots（惰性建组 + add 对齐成员集）→ step ----
                let mut tokens: Option<Vec<i32>> = None;
                let mut failed = false;
                if group.is_none() {
                    // 惰性创建（档位=引擎/模型自身档——单引擎单模型一档，所有
                    // session 同 ctx 档，「档位=首成员」的匹配退化为恒真）。
                    match be.create_batch_group(BatchGroupOptions {
                        max_slots: max_batch_slots as u32,
                        max_context_tokens: 0, // 0 = 跟随引擎（模型）档
                    }) {
                        Ok(g) => group = Some(g),
                        Err(_) => failed = true,
                    }
                }
                if !failed {
                    // add 缺槽成员（分拣已保证 ≤ max_batch_slots）。
                    for gen in batch.iter_mut() {
                        if gen.slot_id.is_none() {
                            match group
                                .as_mut()
                                .expect("group created above")
                                .add(gen.session.as_mut())
                            {
                                Ok(slot) => gen.slot_id = Some(slot),
                                Err(_) => {
                                    failed = true;
                                    break;
                                }
                            }
                        }
                    }
                }
                if !failed {
                    let slots: Vec<u32> = batch.iter().map(|g| g.slot_id.unwrap()).collect();
                    let cfgs: Vec<GenerationConfig> = batch.iter().map(|g| g.config).collect();
                    match group
                        .as_mut()
                        .expect("group created above")
                        .step(&slots, &cfgs)
                    {
                        Ok(toks) if toks.len() == batch.len() => tokens = Some(toks),
                        // 长度不符：防御（native 契约恒 n）——按失败处理。
                        Ok(_) => failed = true,
                        Err(_) => failed = true,
                    }
                }

                if failed {
                    // 本量子这批成员全部回退 solo（先全员出组——不变量 ④；
                    // worker 不崩，记指标；下量子重试合批）。
                    metrics
                        .batch_fallbacks_total
                        .fetch_add(1, Ordering::Relaxed);
                    for mut gen in batch {
                        demote_from_group(&mut gen, &mut group);
                        match step(&mut gen) {
                            StepResult::Continue | StepResult::Yield => active.push_back(gen),
                            StepResult::Done(finish) => retire(
                                gen,
                                finish,
                                &mut idle,
                                &active,
                                max_sessions,
                                RetireMode::Serving,
                                &mut group,
                            ),
                        }
                    }
                } else {
                    // 成功批步：一次 FFI 推进全员，逐成员走与 solo 完全相同的
                    // 判停/发放后半段（feed_token）。
                    metrics.batched_steps_total.fetch_add(1, Ordering::Relaxed);
                    metrics
                        .batch_size_sum
                        .fetch_add(batch.len() as u64, Ordering::Relaxed);
                    let toks = tokens.expect("step succeeded above");
                    for (mut gen, tok) in batch.into_iter().zip(toks) {
                        match feed_token(&mut gen, tok) {
                            StepResult::Continue | StepResult::Yield => active.push_back(gen),
                            StepResult::Done(finish) => retire(
                                gen,
                                finish,
                                &mut idle,
                                &active,
                                max_sessions,
                                RetireMode::Serving,
                                &mut group,
                            ),
                        }
                    }
                }
            } else {
                // 引擎不支持批（理论不可达：分拣已查 batch_engine）——solo。
                for mut gen in batch {
                    demote_from_group(&mut gen, &mut group);
                    match step(&mut gen) {
                        StepResult::Continue | StepResult::Yield => active.push_back(gen),
                        StepResult::Done(finish) => retire(
                            gen,
                            finish,
                            &mut idle,
                            &active,
                            max_sessions,
                            RetireMode::Serving,
                            &mut group,
                        ),
                    }
                }
            }
        } else {
            // 批段不足 2 条（含批关闭 / 引擎不支持 / 取消过滤后）：
            // 并入 solo（现协议原样——已持槽者先出组）。
            for mut gen in batch {
                demote_from_group(&mut gen, &mut group);
                match step(&mut gen) {
                    StepResult::Continue | StepResult::Yield => active.push_back(gen),
                    StepResult::Done(finish) => retire(
                        gen,
                        finish,
                        &mut idle,
                        &active,
                        max_sessions,
                        RetireMode::Serving,
                        &mut group,
                    ),
                }
            }
        }

        // 组空 → destroy（释放组 arena VRAM；下次批量子重建）。档位恒同
        //（单引擎单模型），无「档位变了 destroy 重建」分支。
        if group.is_some() && active.iter().all(|g| g.slot_id.is_none()) {
            drop(group.take());
        }
    }
    // worker 线程上析构全部 session（idle 池 drop → native destroy）与批组
    //（组 destroy：仍在组的成员自动经历 remove 语义——关停路径 retire 已
    // 逐个 remove），先于 join 返回——有序关停的核心。
    drop(group);
    drop(idle);
    MetricsRegistry::global()
        .active_sessions
        .store(0, Ordering::Relaxed);
}

/// 出组簿记（不变量 ④）：成员本 quantum 不再走批步（分拣降级 solo / 容量
/// 截断 / 批段失败回退）前必须 `group.remove`——ABI：成员在组内禁止 solo
/// 使用；remove 建立 组流→成员流 事件链（两次 batch_step 之间调用合法）。
fn demote_from_group(gen: &mut ActiveGen, group: &mut Option<Box<dyn BatchGroup>>) {
    if let Some(slot) = gen.slot_id.take() {
        if let Some(g) = group.as_mut() {
            let _ = g.remove(slot);
        }
    }
}

/// 有序关停：活跃 job 逐个发终态事件（best-effort try_send——接收端可能
/// 还在，但绝不因慢消费者阻塞关停）+ retire（session reset 后归还 idle，
/// 随 worker 循环退出在 worker 线程上析构；批组成员先出组——不变量 ④）。
fn shutdown_active(
    active: &mut VecDeque<ActiveGen>,
    idle: &mut Vec<Box<dyn EngineSession>>,
    max_sessions: usize,
    group: &mut Option<Box<dyn BatchGroup>>,
) {
    while let Some(gen) = active.pop_front() {
        retire(
            gen,
            Finish::Failed("scheduler shutting down".into()),
            idle,
            active,
            max_sessions,
            RetireMode::Shutdown,
            group,
        );
    }
}

/// 准入失败善后：该 job 发 `Failed` 事件（worker 不崩，只影响该 job）、
/// 计数；session 照常归还池——归还前再试一次 reset，对齐旧 execute 的
/// 归还期 reset，保持不变量 ③「有 session 的 job 恰 2 次 reset」。
fn fail_intake(
    events: &tokio::sync::mpsc::Sender<GenEvent>,
    session: Option<Box<dyn EngineSession>>,
    idle: &mut Vec<Box<dyn EngineSession>>,
    active: &VecDeque<ActiveGen>,
    max_sessions: usize,
    msg: String,
) {
    let metrics = MetricsRegistry::global();
    metrics.errors_total.fetch_add(1, Ordering::Relaxed);
    metrics.active_streams.fetch_sub(1, Ordering::Relaxed);
    let _ = events.blocking_send(GenEvent::Failed(msg));
    if let Some(mut session) = session {
        let _ = session.reset();
        return_session(session, idle, active, max_sessions);
    }
}

/// session 归还 idle 池；容量守卫（idle + active < max_sessions）外则
/// drop（native 侧 destroy 释放）。
fn return_session(
    session: Box<dyn EngineSession>,
    idle: &mut Vec<Box<dyn EngineSession>>,
    active: &VecDeque<ActiveGen>,
    max_sessions: usize,
) {
    if idle.len() + active.len() < max_sessions {
        idle.push(session);
    }
}

/// 新 job 准入：**等待计数 −1**（提交侧 +1 的配对；本函数无论走成功、
/// 超时还是失败路径都恰好执行一次）→ 排队超时判定 → 取/建 session →
/// reset（第 1 次）→ **同步 prefill（一步插队——新请求的 prefill 不等
/// 活跃流跑完）** → 入活跃集（phase: Decoding）。
/// 任何失败只影响该 job（Failed 事件），worker 不崩。
fn intake(
    engine: &Arc<dyn Engine>,
    idle: &mut Vec<Box<dyn EngineSession>>,
    active: &mut VecDeque<ActiveGen>,
    max_sessions: usize,
    queue_timeout: std::time::Duration,
    waiting: &Arc<AtomicI64>,
    job: Job,
) {
    let metrics = MetricsRegistry::global();
    // 等待计数 −1 + gauge 镜像（提交侧 mirror_queue_depth 的配对）。
    waiting.fetch_sub(1, Ordering::AcqRel);
    metrics
        .queue_depth
        .store(waiting.load(Ordering::Relaxed), Ordering::Relaxed);

    metrics.active_streams.fetch_add(1, Ordering::Relaxed);
    let Job {
        inner,
        events,
        cancel,
        enqueued_at,
    } = job;
    let GenerationJob { prompt_ids, config } = inner;

    // 排队超时判定（在准入时）：等待期失效的 job 不占 GPU——直接 Failed，
    // 不创建 session、不 prefill。reset 次数为 0：从未持有过 session，
    // 是不变量 ③「每 job 恰 2 次 reset」的合法例外（池语义不受影响）。
    if enqueued_at.elapsed() > queue_timeout {
        fail_intake(
            &events,
            None,
            idle,
            active,
            max_sessions,
            "queue timeout".into(),
        );
        return;
    }

    // session：idle 优先复用，否则懒建（总数 ≤ max_sessions 由准入守卫保证）。
    let mut session = match idle.pop() {
        Some(s) => s,
        None => match engine.create_session() {
            Ok(s) => s,
            Err(e) => {
                fail_intake(
                    &events,
                    None,
                    idle,
                    active,
                    max_sessions,
                    format!("create_session: {e:#}"),
                );
                return;
            }
        },
    };

    // 准入 reset（不变量 ③ 的第 1 次）。
    if let Err(e) = session.reset() {
        fail_intake(
            &events,
            Some(session),
            idle,
            active,
            max_sessions,
            format!("session reset: {e:#}"),
        );
        return;
    }

    // 同步 prefill。prefill_us 记账：session 统计是生命周期累计，差分记到
    // 本 job 头上（与旧 run_one 同口径）。
    let prefill_us_before = session.stats().map(|s| s.prefill_us).unwrap_or(0);
    if let Err(e) = session.prefill(&prompt_ids) {
        fail_intake(
            &events,
            Some(session),
            idle,
            active,
            max_sessions,
            format!("prefill: {e:#}"),
        );
        return;
    }
    if let Ok(stats) = session.stats() {
        let delta = stats.prefill_us.saturating_sub(prefill_us_before);
        metrics.prefill_us_total.fetch_add(delta, Ordering::Relaxed);
    }

    active.push_back(ActiveGen {
        session,
        config,
        events,
        cancel,
        prompt_tokens: prompt_ids.len() as u64,
        generated: 0,
        pending_tok: None,
        enqueued_at,
        first_token_at: None,
        slot_id: None,
        batch_held: false,
        phase: Phase::Decoding,
    });
    // 活跃 session gauge（atomic，quantum 内零锁）。
    metrics
        .active_sessions
        .store(active.len() as i64, Ordering::Relaxed);
}

/// 一条活跃流走一步 solo decode（严格对齐旧 run_one 的单步语义，send 改
/// try_send）。B3：批路径不走本函数的 decode 段，产出 token 后经
/// [`feed_token`] 走完全相同的后半段。
///
/// 语义（顺序即优先级）：
/// 1. cancel → Done(Cancelled)
/// 2. 有 pending_tok（上轮 channel 满暂存）：先补送，成功才继续 decode；
///    补送成功后补检长度上限（该 token 已计入 generated）
/// 3. decode_one 失败 → Done(Failed)（session 照常归还）
///
/// 4-7. → feed_token（eos/计数/TTFT/try_send/Length，见其注释）
///
/// Finish 语义（2026-09 crosscheck fix，OpenAI/vLLM-aligned）：
/// `max_new_tokens` 是 **content** token 上限；上限内收到 eos 则 Stop。
/// eos 既不作为 Token 事件流出也不计入 generated/completion_tokens。
fn step(gen: &mut ActiveGen) -> StepResult {
    // 1. 每步 decode 前检查取消。
    if gen.cancel.load(Ordering::SeqCst) {
        return StepResult::Done(Finish::Cancelled);
    }

    // 2. 上轮暂存的 token 先补送（慢消费者路径的第二半）。
    if let Some(t) = gen.pending_tok.take() {
        match gen.events.try_send(GenEvent::Token(t)) {
            Ok(()) => {
                // 补送成功：该 token 已计入 generated，补检长度上限。
                if gen.generated >= gen.config.max_new_tokens as u64 {
                    return StepResult::Done(Finish::Length);
                }
            }
            Err(TrySendError::Full(ev)) => {
                // 我们只 try_send Token，其它变体不可能（else 分支不可达）。
                let GenEvent::Token(t) = ev else {
                    unreachable!("only Token is sent")
                };
                gen.pending_tok = Some(t); // 回存，下量子再试
                return StepResult::Yield;
            }
            Err(TrySendError::Closed(_)) => return StepResult::Done(Finish::Cancelled),
        }
    }

    // 3. decode（不变量 ②：session 的 stream 内串行序——solo 成员）。
    let token = match gen.session.decode_one(&gen.config) {
        Ok(t) => t,
        Err(e) => return StepResult::Done(Finish::Failed(format!("decode_one: {e:#}"))),
    };

    // 4-7. 判停/发放后半段（solo 与批步共用）。
    feed_token(gen, token)
}

/// 一步 decode 产出 token 的判停/发放后半段（B3 从 step 抽出，语义逐条
/// 搬移——solo step() 与批步 `group.step` 的产出走完全相同的路径）：
///
/// 4. EOS → Done(Stop)（eos 不流出、不计入——现语义）
/// 5. generated += 1；首 token 记 first_token_at（TTFT 锚点）
/// 6. events.try_send(Token)：Full → 暂存 pending_tok、本量子让步（不
///    取消——慢消费者只延迟自己；下量子经 solo 路径先补送）；Closed →
///    Done(Cancelled)
/// 7. generated >= max_new_tokens → Done(Length)
fn feed_token(gen: &mut ActiveGen, token: i32) -> StepResult {
    // 4. eos（自然，或 native 预算耗尽哨兵）：Stop；不流出、不计数。
    if EOS_IDS.contains(&token) {
        return StepResult::Done(Finish::Stop);
    }

    // 5. 首个 content token 记 TTFT 锚点。
    gen.generated += 1;
    if gen.first_token_at.is_none() {
        gen.first_token_at = Some(Instant::now());
    }

    // 6. try_send（非阻塞；tokio semaphore 的 Closed 判定优先于 Full，
    //    接收端整体 drop 时恒为 Closed，不会卡在 Full 让步里）。
    match gen.events.try_send(GenEvent::Token(token)) {
        Ok(()) => {}
        Err(TrySendError::Full(ev)) => {
            let GenEvent::Token(token) = ev else {
                unreachable!("only Token is sent")
            };
            gen.pending_tok = Some(token);
            return StepResult::Yield;
        }
        Err(TrySendError::Closed(_)) => return StepResult::Done(Finish::Cancelled),
    }

    // 7. content token 数达上限：Length。
    if gen.generated >= gen.config.max_new_tokens as u64 {
        return StepResult::Done(Finish::Length);
    }
    StepResult::Continue
}

/// job 终局：指标、终态事件、session 归还（retire reset——不变量 ③ 的
/// 第 2 次）。
///
/// B3：批组成员**先出组再归还**（不变量 ④——session 归还池前清忙态；
/// remove 在 quantum 边界语义内：本成员刚完成最后一次批步/solo 步）。
///
/// 事件次序：暂存 token 先冲刷再发终态（counted ⇒ delivered 的旧语义；
/// 只有 Cancelled 可能带暂存到达这里，接收端已 drop 则忽略发送失败）。
///
/// S3 指标上报（worker 侧口径，见 minicpm-metrics 文档注释）：
/// * TTFT = first_token_at − enqueued_at（含排队等待），≥1 token 才记；
/// * TPOT = (done_at − first_token_at)/(generated−1)，generated ≤ 1 记 0；
/// * `recent_requests` 追加一条（Failed 不入列——errors_total 已覆盖）。
#[allow(clippy::too_many_arguments)]
fn retire(
    gen: ActiveGen,
    finish: Finish,
    idle: &mut Vec<Box<dyn EngineSession>>,
    active: &VecDeque<ActiveGen>,
    max_sessions: usize,
    mode: RetireMode,
    group: &mut Option<Box<dyn BatchGroup>>,
) {
    let metrics = MetricsRegistry::global();
    let done_at = Instant::now();
    let ActiveGen {
        mut session,
        events,
        pending_tok,
        prompt_tokens,
        generated,
        enqueued_at,
        first_token_at,
        slot_id,
        ..
    } = gen;

    // 批组成员出组（清忙态）——必须先于 reset/归还（不变量 ④）。
    if let Some(slot) = slot_id {
        if let Some(g) = group.as_mut() {
            let _ = g.remove(slot);
        }
    }
    let usage = Usage {
        prompt_tokens,
        completion_tokens: generated,
    };

    // 暂存 token 冲刷：常规期 blocking（保证 counted⇒delivered），关停期
    // try_send best-effort（不阻塞关停）。
    if let Some(t) = pending_tok {
        match mode {
            RetireMode::Serving => {
                let _ = events.blocking_send(GenEvent::Token(t));
            }
            RetireMode::Shutdown => {
                let _ = events.try_send(GenEvent::Token(t));
            }
        }
    }

    // worker 侧 TTFT/TPOT（含排队等待的完整口径；≥1 token 才有意义）。
    if let Some(ft) = first_token_at {
        if !matches!(finish, Finish::Failed(_)) {
            metrics.observe_ttft(ft.saturating_duration_since(enqueued_at));
            if generated >= 2 {
                let per = done_at.saturating_duration_since(ft) / (generated as u32 - 1);
                metrics.observe_tpot(per);
            }
        }
    }

    // finish_reason 字符串（recent_requests 用；Failed 不入列）。
    let mut recent_reason: Option<&str> = None;

    match finish {
        Finish::Stop | Finish::Length => {
            metrics
                .prompt_tokens_total
                .fetch_add(prompt_tokens, Ordering::Relaxed);
            metrics
                .generated_tokens_total
                .fetch_add(generated, Ordering::Relaxed);
            let reason = match finish {
                Finish::Stop => FinishReason::Stop,
                _ => FinishReason::Length,
            };
            recent_reason = Some(match reason {
                FinishReason::Stop => "stop",
                FinishReason::Length => "length",
                _ => unreachable!("matched Stop|Length above"),
            });
            let ev = GenEvent::Done { reason, usage };
            match mode {
                RetireMode::Serving => {
                    let _ = events.blocking_send(ev);
                }
                RetireMode::Shutdown => {
                    let _ = events.try_send(ev);
                }
            }
        }
        Finish::Cancelled => {
            metrics.cancelled_total.fetch_add(1, Ordering::Relaxed);
            metrics
                .prompt_tokens_total
                .fetch_add(prompt_tokens, Ordering::Relaxed);
            metrics
                .generated_tokens_total
                .fetch_add(generated, Ordering::Relaxed);
            recent_reason = Some("cancelled");
            // Receiver may still be alive (explicit CancelToken): tell it.
            let ev = GenEvent::Done {
                reason: FinishReason::Cancelled,
                usage,
            };
            match mode {
                RetireMode::Serving => {
                    let _ = events.blocking_send(ev);
                }
                RetireMode::Shutdown => {
                    let _ = events.try_send(ev);
                }
            }
        }
        Finish::Failed(err) => {
            metrics.errors_total.fetch_add(1, Ordering::Relaxed);
            // 失败路径不加 token 总量（对齐旧 execute 的 Err 分支口径）。
            let ev = GenEvent::Failed(err);
            match mode {
                RetireMode::Serving => {
                    let _ = events.blocking_send(ev);
                }
                RetireMode::Shutdown => {
                    let _ = events.try_send(ev);
                }
            }
        }
    }

    // recent_requests 明细（每完成请求 1 次锁；Failed 不入列）。
    if let Some(reason) = recent_reason {
        let ttft_us = first_token_at
            .map(|ft| ft.saturating_duration_since(enqueued_at).as_micros() as u64)
            .unwrap_or(0);
        let tpot_us = match first_token_at {
            Some(ft) if generated >= 2 => {
                (done_at.saturating_duration_since(ft) / (generated as u32 - 1)).as_micros() as u64
            }
            _ => 0,
        };
        let ts_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
        metrics.push_recent(minicpm_metrics::RecentReq {
            ttft_us,
            tpot_us,
            prompt_tokens: u32::try_from(prompt_tokens).unwrap_or(u32::MAX),
            completion_tokens: u32::try_from(generated).unwrap_or(u32::MAX),
            finish_reason: reason.to_string(),
            ts_ms,
        });
    }
    metrics.active_streams.fetch_sub(1, Ordering::Relaxed);

    // 归还池：先 reset（池不变量），容量守卫外则 drop（native destroy）。
    let _ = session.reset();
    return_session(session, idle, active, max_sessions);
    // 活跃 session gauge（atomic；retire 后的当前活跃数）。
    metrics
        .active_sessions
        .store(active.len() as i64, Ordering::Relaxed);
}

#[cfg(test)]
// B3 批测试的串行锁守卫刻意跨 await 持有（current_thread runtime 的
// block_on 无 Send 约束，跨 await 持锁正是串行语义本身）。
#[allow(clippy::await_holding_lock)]
mod tests {
    use super::*;
    use minicpm_runtime::mock::MOCK_EOS;
    use minicpm_runtime::MockEngine;
    use std::time::Duration;

    const CANCEL_PACE: Duration = Duration::from_millis(5);

    /// Scripted engine without pacing: full-speed generation.
    fn scripted(script: Vec<i32>) -> Arc<MockEngine> {
        Arc::new(MockEngine::with_script(script))
    }

    async fn collect(mut rx: tokio::sync::mpsc::Receiver<GenEvent>) -> Vec<GenEvent> {
        let mut out = Vec::new();
        while let Some(ev) = rx.recv().await {
            out.push(ev);
        }
        out
    }

    fn cfg(max_new: u32) -> GenerationConfig {
        GenerationConfig {
            max_new_tokens: max_new,
            ..GenerationConfig::default()
        }
    }

    fn tokens_of(events: &[GenEvent]) -> Vec<i32> {
        events
            .iter()
            .filter_map(|e| match e {
                GenEvent::Token(t) => Some(*t),
                _ => None,
            })
            .collect()
    }

    #[tokio::test]
    async fn n_token_stream_then_length() {
        // 50-token script; max_new_tokens=8 must cut it off with Done(Length).
        let script: Vec<i32> = (0..50).map(|i| 100 + i).collect();
        let engine = scripted(script.clone());
        let sched = Scheduler::new(engine.clone(), 2).unwrap();
        let (rx, _cancel) = sched.handle().generate(GenerationJob {
            prompt_ids: vec![0, 5, 9],
            config: cfg(8),
        });
        let events = collect(rx).await;

        let tokens: Vec<i32> = events
            .iter()
            .filter_map(|e| match e {
                GenEvent::Token(t) => Some(*t),
                _ => None,
            })
            .collect();
        assert_eq!(tokens, script[..8].to_vec());
        match events.last() {
            Some(GenEvent::Done {
                reason: FinishReason::Length,
                usage,
            }) => {
                assert_eq!(usage.completion_tokens, 8);
                assert_eq!(usage.prompt_tokens, 3);
            }
            other => panic!("expected Done(Length), got {other:?}"),
        }
    }

    #[tokio::test]
    async fn script_exhaustion_stops_on_eos_uncounted() {
        // Finish semantics (2026-09 fix): the script-exhaustion eos ends the
        // round with Done(Stop); the eos token is NOT streamed and NOT counted
        // in completion_tokens — only the two real content tokens are.
        let engine = scripted(vec![101, 102]);
        let sched = Scheduler::new(engine, 1).unwrap();
        let (rx, _cancel) = sched.handle().generate(GenerationJob {
            prompt_ids: vec![0],
            config: cfg(1000),
        });
        let events = collect(rx).await;

        let tokens: Vec<i32> = events
            .iter()
            .filter_map(|e| match e {
                GenEvent::Token(t) => Some(*t),
                _ => None,
            })
            .collect();
        // Content only — the eos never reaches the stream.
        assert_eq!(tokens, vec![101, 102]);
        match events.last() {
            Some(GenEvent::Done {
                reason: FinishReason::Stop,
                usage,
            }) => {
                assert_eq!(usage.completion_tokens, 2);
            }
            other => panic!("expected Done(Stop), got {other:?}"),
        }
    }

    #[tokio::test]
    async fn eos_at_exact_limit_is_stop_not_length() {
        // eos arriving as the Nth (last budgeted) decode: Stop wins over
        // Length, and the eos is still not counted (N-1 content tokens).
        let mut script = vec![201, 202, 203, 204];
        script.push(MOCK_EOS);
        let engine = scripted(script);
        let sched = Scheduler::new(engine, 1).unwrap();
        let (rx, _cancel) = sched.handle().generate(GenerationJob {
            prompt_ids: vec![0],
            config: cfg(5),
        });
        let events = collect(rx).await;
        let tokens: Vec<i32> = events
            .iter()
            .filter_map(|e| match e {
                GenEvent::Token(t) => Some(*t),
                _ => None,
            })
            .collect();
        assert_eq!(tokens, vec![201, 202, 203, 204]);
        match events.last() {
            Some(GenEvent::Done {
                reason: FinishReason::Stop,
                usage,
            }) => {
                assert_eq!(usage.completion_tokens, 4);
            }
            other => panic!("expected Done(Stop), got {other:?}"),
        }
    }

    #[tokio::test]
    async fn mid_generation_cancel_returns_session() {
        // Paced engine so cancel is observable within ≤2 tokens.
        let script: Vec<i32> = (0..500).map(|i| 100 + i).collect();
        let engine = Arc::new(MockEngine::with_script(script).with_decode_delay(CANCEL_PACE));
        let sched = Scheduler::new(engine.clone(), 1).unwrap();
        let handle = sched.handle();

        let (mut rx, cancel) = handle.generate(GenerationJob {
            prompt_ids: vec![0],
            config: cfg(10_000),
        });
        let mut events = Vec::new();
        let mut seen_first = false;
        while let Some(ev) = rx.recv().await {
            events.push(ev.clone());
            if let GenEvent::Token(_) = ev {
                if !seen_first {
                    seen_first = true;
                    cancel.cancel();
                }
            }
        }
        // ≤2 tokens total after the paced cancel window, then Done(Cancelled).
        let token_count = events
            .iter()
            .filter(|e| matches!(e, GenEvent::Token(_)))
            .count();
        assert!(
            token_count <= 2,
            "cancel must stop within 2 tokens, got {token_count}"
        );
        match events.last() {
            Some(GenEvent::Done {
                reason: FinishReason::Cancelled,
                ..
            }) => {}
            other => panic!("expected Done(Cancelled), got {other:?}"),
        }

        // Session was returned: the next job runs on the same pool.
        let (rx2, _) = handle.generate(GenerationJob {
            prompt_ids: vec![0, 1],
            config: cfg(2),
        });
        let events2 = collect(rx2).await;
        assert!(matches!(
            events2.last(),
            Some(GenEvent::Done {
                reason: FinishReason::Length,
                ..
            })
        ));
        assert_eq!(
            engine.stats().sessions_created,
            1,
            "second job must reuse the pooled session"
        );
    }

    #[tokio::test]
    async fn dropping_receiver_cancels() {
        let script: Vec<i32> = (0..500).map(|i| 100 + i).collect();
        let engine = Arc::new(MockEngine::with_script(script).with_decode_delay(CANCEL_PACE));
        let sched = Scheduler::new(engine.clone(), 1).unwrap();
        let handle = sched.handle();

        let (mut rx, _cancel) = handle.generate(GenerationJob {
            prompt_ids: vec![0],
            config: cfg(10_000),
        });
        // Read exactly one token, then drop the whole receiver.
        let first = tokio::time::timeout(Duration::from_secs(5), rx.recv()).await;
        assert!(
            matches!(first, Ok(Some(GenEvent::Token(_)))),
            "first token must arrive"
        );
        drop(rx);

        // The worker's next try_send fails (Closed) → cancel → session returned.
        // Prove pool health with a follow-up job.
        let (rx2, _) = handle.generate(GenerationJob {
            prompt_ids: vec![0],
            config: cfg(3),
        });
        let events2 = collect(rx2).await;
        assert!(
            matches!(events2.last(), Some(GenEvent::Done { .. })),
            "follow-up job must complete: {events2:?}"
        );
        assert_eq!(
            engine.stats().sessions_created,
            1,
            "cancelled session must be reused, not leaked"
        );
        assert!(
            engine.stats().decode_calls < 500,
            "worker must have stopped early"
        );
    }

    #[tokio::test]
    async fn consecutive_jobs_reuse_session_with_resets() {
        let engine = scripted(vec![201, 202, 203]);
        let sched = Scheduler::new(engine.clone(), 1).unwrap();
        let handle = sched.handle();

        let run = || {
            let (rx, _) = handle.generate(GenerationJob {
                prompt_ids: vec![0, 7],
                config: cfg(1000),
            });
            collect(rx)
        };
        let e1 = run().await;
        let e2 = run().await;

        // Identical scripted output both times (reset replays the script).
        assert_eq!(e1, e2);
        assert!(matches!(
            e1.last(),
            Some(GenEvent::Done {
                reason: FinishReason::Stop,
                ..
            })
        ));

        let st = engine.stats();
        assert_eq!(
            st.sessions_created, 1,
            "session must be created once and pooled"
        );
        // Reset discipline: 2 per job (reset-on-take + reset-on-return) × 2 jobs.
        assert_eq!(
            st.reset_calls, 4,
            "expected 2 resets per job (take + return), got {}",
            st.reset_calls
        );
        assert_eq!(st.prefill_calls, 2);
    }

    #[tokio::test]
    async fn engine_failure_reports_failed() {
        // No script and huge max tokens can't fail on the mock; instead force
        // a create_session failure by draining the pool? The mock never fails,
        // so exercise the Failed path through a failing engine stub.
        struct Failing;
        impl Engine for Failing {
            fn create_session(&self) -> anyhow::Result<Box<dyn EngineSession>> {
                Err(anyhow::anyhow!("boom"))
            }
        }
        let sched = Scheduler::new(Arc::new(Failing), 1).unwrap();
        let (rx, _) = sched.handle().generate(GenerationJob {
            prompt_ids: vec![0],
            config: cfg(4),
        });
        let events = collect(rx).await;
        match events.last() {
            Some(GenEvent::Failed(msg)) => assert!(msg.contains("boom"), "{msg}"),
            other => panic!("expected Failed, got {other:?}"),
        }
    }

    // ========================================================================
    // S2：活跃集 + quantum 交错的并发行为
    // ========================================================================

    /// ① 双 job 交错：两条流的 token 按量子交替推进（无队头阻塞），短流
    ///    在长流远未完成时先拿到全部 token 并 Done。
    #[tokio::test]
    async fn two_jobs_interleave_without_head_of_line_blocking() {
        // 10ms/步：一个 quantum = A 一步 + B 一步 ≈ 20ms，交错 vs 串行的
        // 时间差有两个数量级的区分度。
        const PACE: Duration = Duration::from_millis(10);
        let script: Vec<i32> = (0..500).map(|i| 100 + i).collect();
        let engine = Arc::new(MockEngine::with_script(script.clone()).with_decode_delay(PACE));
        let sched = Scheduler::new(engine, 2).unwrap();
        let handle = sched.handle();

        let (rxa, _ca) = handle.generate(GenerationJob {
            prompt_ids: vec![0],
            config: cfg(25),
        });
        let (rxb, _cb) = handle.generate(GenerationJob {
            prompt_ids: vec![1],
            config: cfg(5),
        });

        // 并发消费两条流，逐事件记录到达时刻。
        async fn logged(mut rx: tokio::sync::mpsc::Receiver<GenEvent>) -> Vec<(GenEvent, Instant)> {
            let mut out = Vec::new();
            while let Some(ev) = rx.recv().await {
                out.push((ev, Instant::now()));
            }
            out
        }
        let (ea, eb) = tokio::join!(logged(rxa), logged(rxb));

        // 内容零回归：两流各拿到自己的完整 token 序列 + Done(Length)。
        assert_eq!(
            tokens_of(&ea.iter().map(|(e, _)| e.clone()).collect::<Vec<_>>()),
            script[..25].to_vec()
        );
        assert_eq!(
            tokens_of(&eb.iter().map(|(e, _)| e.clone()).collect::<Vec<_>>()),
            script[..5].to_vec()
        );
        assert!(matches!(
            ea.last().unwrap().0,
            GenEvent::Done {
                reason: FinishReason::Length,
                ..
            }
        ));
        assert!(matches!(
            eb.last().unwrap().0,
            GenEvent::Done {
                reason: FinishReason::Length,
                ..
            }
        ));

        // 互不等待对方完成：B（5 token）的 Done 早于 A 的第 15 个 token——
        // 旧串行 FIFO 下 B 的任何 token 都要等 A 整体跑完（≥25 步 ≈ 25×PACE）。
        let a_15th_token_at = ea[14].1;
        let b_done_at = eb.last().unwrap().1;
        assert!(
            b_done_at + PACE * 5 < a_15th_token_at,
            "short stream must finish while the long one is still decoding"
        );

        // 交替推进：前 5 个位置两流第 k 个 token 的到达时差 < 4 个 PACE
        //（串行下至少差 (25-5)×PACE = 200ms）。
        for k in 0..5 {
            let d = ea[k]
                .1
                .saturating_duration_since(eb[k].1)
                .max(eb[k].1.saturating_duration_since(ea[k].1));
            assert!(
                d < PACE * 4,
                "streams must advance in lockstep (k={k}, diff={d:?})"
            );
        }
    }

    /// ② mid-quantum 取消：目标流 ≤2 token 即停；另一流不受扰跑满；被
    ///    取消的 session 归还并被后续 job 复用（sessions_created 不涨）。
    #[tokio::test]
    async fn cancel_one_stream_does_not_disturb_the_other() {
        let script: Vec<i32> = (0..500).map(|i| 100 + i).collect();
        let engine =
            Arc::new(MockEngine::with_script(script.clone()).with_decode_delay(CANCEL_PACE));
        let sched = Scheduler::new(engine.clone(), 2).unwrap();
        let handle = sched.handle();

        let (mut rxa, cancel_a) = handle.generate(GenerationJob {
            prompt_ids: vec![0],
            config: cfg(10_000),
        });
        let (rxb, _) = handle.generate(GenerationJob {
            prompt_ids: vec![1],
            config: cfg(20),
        });

        // 消费 A：首个 token 即取消，读完剩余事件（Done(Cancelled)）。
        let mut events_a = Vec::new();
        let mut seen_first = false;
        while let Some(ev) = rxa.recv().await {
            if !seen_first && matches!(ev, GenEvent::Token(_)) {
                seen_first = true;
                cancel_a.cancel();
            }
            events_a.push(ev);
        }
        let a_tokens = events_a
            .iter()
            .filter(|e| matches!(e, GenEvent::Token(_)))
            .count();
        assert!(
            a_tokens <= 2,
            "cancelled stream must stop within 2 tokens, got {a_tokens}"
        );
        assert!(matches!(
            events_a.last(),
            Some(GenEvent::Done {
                reason: FinishReason::Cancelled,
                ..
            })
        ));

        // B 不受扰：完整 20 token + Done(Length)（内容逐 token 对齐 script）。
        let events_b = collect(rxb).await;
        assert_eq!(tokens_of(&events_b), script[..20].to_vec());
        assert!(matches!(
            events_b.last(),
            Some(GenEvent::Done {
                reason: FinishReason::Length,
                ..
            })
        ));

        // A 的 session 已归还：后续 job 复用，sessions_created 保持 2。
        let (rxc, _) = handle.generate(GenerationJob {
            prompt_ids: vec![2],
            config: cfg(3),
        });
        let events_c = collect(rxc).await;
        assert!(matches!(
            events_c.last(),
            Some(GenEvent::Done {
                reason: FinishReason::Length,
                ..
            })
        ));
        assert_eq!(
            engine.stats().sessions_created,
            2,
            "cancelled session must be reused, not leaked"
        );
    }

    /// ③ pending_tok：事件 channel 容量 1 + 慢消费者 → token 暂存、下
    ///    量子送达——不丢失、不取消、有序。
    #[tokio::test]
    async fn full_channel_stashes_token_and_delivers_next_quantum() {
        const PACE: Duration = Duration::from_millis(5);
        let script: Vec<i32> = (0..500).map(|i| 100 + i).collect();
        let engine = Arc::new(MockEngine::with_script(script.clone()).with_decode_delay(PACE));
        let sched = Scheduler::new(engine, 1).unwrap();
        let handle = sched.handle();

        let (rx, _cancel) = handle.generate_with_capacity(
            GenerationJob {
                prompt_ids: vec![0],
                config: cfg(20),
            },
            1, // 容量 1：第一个 token 占满，第二个必走 Full → 暂存
        );

        // 消费者先睡 150ms：worker 期间只能产出 1 个已送达 + 1 个暂存
        // token，其余量子全部让步（慢消费者只延迟自己，绝不取消）。
        tokio::time::sleep(Duration::from_millis(150)).await;
        let events = collect(rx).await;

        let tokens = tokens_of(&events);
        assert_eq!(
            tokens,
            script[..20].to_vec(),
            "stashed tokens must all arrive, in order"
        );
        match events.last() {
            Some(GenEvent::Done {
                reason: FinishReason::Length,
                usage,
            }) => {
                assert_eq!(usage.completion_tokens, 20);
            }
            other => panic!("expected Done(Length), got {other:?}"),
        }
    }

    // ========================================================================
    // S3：准入控制（queue_limit / queue_timeout / waiting 计数）
    // ========================================================================

    fn paced(script_len: usize, pace: Duration) -> Arc<MockEngine> {
        Arc::new(
            MockEngine::with_script((0..script_len).map(|i| 100 + i as i32).collect())
                .with_decode_delay(pace),
        )
    }

    /// ① QueueFull：queue_limit=1 + 单活跃流（max_sessions=1）→ 队列中
    ///    恰 1 个等待 job 时，下一个提交被拒（不占位）；计数最终归零；
    ///    兼容 generate 不受限（超额兜底 +1）。
    #[tokio::test]
    async fn queue_limit_rejects_overflow_submissions() {
        let engine = paced(500, Duration::from_millis(5));
        let sched = Scheduler::with_config(
            engine,
            SchedulerConfig {
                max_sessions: 1,
                queue_limit: 1,
                ..SchedulerConfig::default()
            },
        )
        .unwrap();
        let handle = sched.handle();

        // r1：占住唯一活跃 slot（200 token × 5ms ≈ 1s 长跑）。
        let (rx1, _) = handle
            .try_generate(GenerationJob {
                prompt_ids: vec![0],
                config: cfg(200),
            })
            .unwrap();
        // 等 r1 被 worker 准入（waiting 归零）：有限重试而非死等。
        for _ in 0..200 {
            if handle.queue_depth() == 0 {
                break;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        assert_eq!(
            handle.queue_depth(),
            0,
            "admitted job must decrement waiting"
        );

        // r2：占住唯一等待位（active 满 → 留在 channel）。
        let (rx2, _) = handle
            .try_generate(GenerationJob {
                prompt_ids: vec![1],
                config: cfg(2),
            })
            .unwrap();
        assert_eq!(handle.queue_depth(), 1);
        // r3：queue_limit=1 已满 → QueueFull（不占位：计数仍 1）。
        match handle.try_generate(GenerationJob {
            prompt_ids: vec![2],
            config: cfg(2),
        }) {
            Err(SubmitError::QueueFull) => {}
            other => panic!("expected QueueFull, got {other:?}"),
        }
        assert_eq!(
            handle.queue_depth(),
            1,
            "rejected submission must not take a slot"
        );

        // 兼容入口不受限：generate 照常入队（现语义，超额兜底 +1）。
        let (rx4, _) = handle.generate(GenerationJob {
            prompt_ids: vec![3],
            config: cfg(2),
        });
        assert_eq!(handle.queue_depth(), 2);

        // 全部完成：r1 200 token、r2/r4 各 2 token。
        let e1 = collect(rx1).await;
        let e2 = collect(rx2).await;
        let e4 = collect(rx4).await;
        assert_eq!(tokens_of(&e1).len(), 200);
        assert_eq!(tokens_of(&e2).len(), 2);
        assert_eq!(tokens_of(&e4).len(), 2);
        // waiting 计数准确归零（每次 intake 恰 −1）。
        assert_eq!(handle.queue_depth(), 0, "waiting must drain to 0");
    }

    /// ①b queue_limit=0：idle 提交同样立即拒绝（立即 429 语义）。
    #[tokio::test]
    async fn queue_limit_zero_rejects_immediately() {
        let engine = paced(10, Duration::from_millis(1));
        let sched = Scheduler::with_config(
            engine,
            SchedulerConfig {
                queue_limit: 0,
                ..SchedulerConfig::default()
            },
        )
        .unwrap();
        let handle = sched.handle();
        match handle.try_generate(GenerationJob {
            prompt_ids: vec![0],
            config: cfg(2),
        }) {
            Err(SubmitError::QueueFull) => {}
            other => panic!("queue_limit=0 must reject immediately, got {other:?}"),
        }
    }

    /// ② 排队超时：超时判定在**准入时**——滞留过久的 job 直接 Failed
    ///    （"queue timeout"），不创建 session、不 prefill、0 次 reset。
    #[tokio::test]
    async fn queue_timeout_fails_without_session_or_prefill() {
        let engine = paced(500, Duration::from_millis(5));
        let sched = Scheduler::with_config(
            engine.clone(),
            SchedulerConfig {
                max_sessions: 1,
                queue_timeout: Duration::from_millis(60),
                ..SchedulerConfig::default()
            },
        )
        .unwrap();
        let handle = sched.handle();

        // r1 占住唯一 slot ~1s（200 token × 5ms）≫ 超时 60ms。
        let (rx1, _) = handle.generate(GenerationJob {
            prompt_ids: vec![0],
            config: cfg(200),
        });
        tokio::time::sleep(Duration::from_millis(20)).await; // r1 已 active
                                                             // r2 排队；r1 释放 slot 时 r2 已滞留 ≈1s ≫ 60ms → 准入即 Failed。
        let (rx2, _) = handle.generate(GenerationJob {
            prompt_ids: vec![1],
            config: cfg(5),
        });

        let e1 = collect(rx1).await;
        let e2 = collect(rx2).await;
        assert_eq!(tokens_of(&e1).len(), 200, "r1 unaffected by r2's timeout");
        match e2.as_slice() {
            [GenEvent::Failed(msg)] => assert!(msg.contains("queue timeout"), "{msg}"),
            other => panic!("expected single Failed(queue timeout), got {other:?}"),
        }

        // 超时 job 零 GPU 占用：没建 session、没 prefill；reset 只有 r1 的
        // 2 次（准入 + retire）——不变量 ③ 的合法例外（0 持有 → 0 reset）。
        let st = engine.stats();
        assert_eq!(
            st.sessions_created, 1,
            "timed-out job must not create a session"
        );
        assert_eq!(st.prefill_calls, 1, "timed-out job must not prefill");
        assert_eq!(st.reset_calls, 2, "only the admitted job's 2 resets");
    }

    /// ③ 有序关停：drop 全部 handle 后 join 返回（不卡死）；活跃 job
    ///    收到终态事件（best-effort）且不被等自然完成。
    #[tokio::test]
    async fn orderly_shutdown_terminates_active_jobs() {
        let engine = paced(500, Duration::from_millis(5));
        let sched = Scheduler::with_config(
            engine.clone(),
            SchedulerConfig {
                max_sessions: 1,
                ..SchedulerConfig::default()
            },
        )
        .unwrap();
        let handle = sched.handle();

        // 长 job active（200 × 5ms ≈ 1s），随即整体 drop（handle + sched）。
        let (mut rx, _cancel) = handle.generate(GenerationJob {
            prompt_ids: vec![0],
            config: cfg(200),
        });
        tokio::time::sleep(Duration::from_millis(30)).await; // 确保 active
        drop(handle);
        drop(sched);
        // 到这里 join 已返回（若 worker 不退出则本测试卡死/超时失败）。
        // 活跃 job 收到终态 Failed（接收端在读 → 送达）。
        let mut last = None;
        while let Some(ev) = rx.recv().await {
            last = Some(ev);
        }
        match last {
            Some(GenEvent::Failed(msg)) => assert!(msg.contains("shutting down"), "{msg}"),
            other => panic!("expected Failed(shutting down) terminal event, got {other:?}"),
        }
        assert!(
            engine.stats().decode_calls < 200,
            "shutdown must not wait for natural finish"
        );
    }

    // ========================================================================
    // B3：批 quantum（MockBatchEngine + MockBatchGroup——mock 批组记录
    // add/remove/step 调用序；批步 token 用 900+ 段与 solo 脚本 100+ 段
    // 可区分，断言「哪些 token 来自批」是构造性的）
    // ========================================================================

    use std::sync::Mutex;

    // trait 方法经 dyn 对象调用，无需 trait 导入（保留此处说明）。

    /// 批测试串行锁：B3 断言读取全局 `MetricsRegistry` 的批计数差值，
    /// 并行批测试会互相污染——本组测试串行执行（S2/S3 既有测试不读批
    /// 计数，无需加锁）。
    ///
    /// 守卫刻意跨 await 持有（current_thread runtime 的 block_on 无 Send
    /// 约束，跨 await 持锁正是串行语义本身），allow 见模块头。
    static BATCH_TESTS: Mutex<()> = Mutex::new(());

    /// 批组调用序记录（create / add:slot / remove:slot / step:n:members）。
    #[derive(Clone, Default)]
    struct BatchLog(Arc<Mutex<Vec<String>>>);

    impl BatchLog {
        fn push(&self, e: String) {
            self.0.lock().unwrap().push(e);
        }

        fn snapshot(&self) -> Vec<String> {
            self.0.lock().unwrap().clone()
        }

        fn count_starting_with(&self, prefix: &str) -> usize {
            self.snapshot()
                .iter()
                .filter(|e| e.starts_with(prefix))
                .count()
        }
    }

    /// 测试用批引擎：solo 会话复用 [`MockEngine`]（脚本 + 可 pacing +
    /// 调用计数）；`as_batch` 返回 Some（引擎支持批）；批组可编程失败
    ///（fail_create / fail_step）。
    struct MockBatchEngine {
        mock: MockEngine,
        log: BatchLog,
        fail_create: bool,
        fail_step: bool,
        /// 批步 pacing（ms 级）——放大 quantum 边界，使「提交先于准入」
        /// 的时序在测试里确定性成立。
        step_delay: Option<Duration>,
    }

    impl MockBatchEngine {
        fn new(script: Vec<i32>) -> Self {
            Self {
                mock: MockEngine::with_script(script),
                log: BatchLog::default(),
                fail_create: false,
                fail_step: false,
                step_delay: None,
            }
        }

        fn fail_step(mut self) -> Self {
            self.fail_step = true;
            self
        }

        fn with_pace(mut self, solo: Duration, batch: Duration) -> Self {
            self.mock = MockEngine::with_script((0..500).map(|i| 100 + i).collect())
                .with_decode_delay(solo);
            self.step_delay = Some(batch);
            self
        }

        fn stats(&self) -> minicpm_runtime::MockStats {
            self.mock.stats()
        }
    }

    impl Engine for MockBatchEngine {
        fn create_session(&self) -> anyhow::Result<Box<dyn EngineSession>> {
            self.mock.create_session()
        }

        fn as_batch(&self) -> Option<&dyn minicpm_runtime::BatchEngine> {
            Some(self)
        }
    }

    impl minicpm_runtime::BatchEngine for MockBatchEngine {
        fn create_batch_group(
            &self,
            _opts: minicpm_runtime::BatchGroupOptions,
        ) -> anyhow::Result<Box<dyn minicpm_runtime::BatchGroup>> {
            if self.fail_create {
                anyhow::bail!("mock create_batch_group failure");
            }
            self.log.push("create".into());
            Ok(Box::new(MockBatchGroup {
                log: self.log.clone(),
                fail_step: self.fail_step,
                step_delay: self.step_delay,
                next_slot: 0,
                steps: 0,
            }))
        }
    }

    /// 测试用批组：记录调用序；step 返回 **900 段 token**（900 + slot*100 +
    /// 第几次 step）——与 solo 脚本（100+i）可区分，且 slot/步数可回读。
    struct MockBatchGroup {
        log: BatchLog,
        fail_step: bool,
        step_delay: Option<Duration>,
        next_slot: u32,
        steps: u32,
    }

    impl minicpm_runtime::BatchGroup for MockBatchGroup {
        fn add(&mut self, _session: &mut dyn EngineSession) -> anyhow::Result<u32> {
            let slot = self.next_slot;
            self.next_slot += 1;
            self.log.push(format!("add:{slot}"));
            Ok(slot)
        }

        fn remove(&mut self, slot: u32) -> anyhow::Result<()> {
            self.log.push(format!("remove:{slot}"));
            Ok(())
        }

        fn step(&mut self, slots: &[u32], _cfgs: &[GenerationConfig]) -> anyhow::Result<Vec<i32>> {
            self.steps += 1;
            self.log
                .push(format!("step:{}:{}", self.steps, slots.len()));
            if let Some(d) = self.step_delay {
                std::thread::sleep(d);
            }
            if self.fail_step {
                anyhow::bail!("mock batch step failure");
            }
            Ok(slots
                .iter()
                .map(|&s| 900 + (s as i32) * 100 + self.steps as i32)
                .collect())
        }
    }

    /// greedy 配置（temperature=0 → `is_greedy` 真，可入批）。
    fn gcfg(max_new: u32) -> GenerationConfig {
        GenerationConfig {
            max_new_tokens: max_new,
            temperature: 0.0,
            top_p: 1.0,
            top_k: 0,
            repetition_penalty: 1.0,
            seed: 1,
        }
    }

    /// 采样配置（temperature>0 且 top_k!=1 → 非 greedy，恒 solo）。
    fn scfg(max_new: u32) -> GenerationConfig {
        GenerationConfig {
            max_new_tokens: max_new,
            temperature: 1.0,
            ..gcfg(max_new)
        }
    }

    fn batch_sched(engine: Arc<MockBatchEngine>, max_sessions: usize) -> Scheduler {
        Scheduler::with_config(
            engine,
            SchedulerConfig {
                max_sessions,
                batching: true,
                ..SchedulerConfig::default()
            },
        )
        .unwrap()
    }

    /// metrics 快照（批三键；并行测试下用前后差值断言）。
    fn batch_metrics() -> (u64, u64, u64) {
        let m = MetricsRegistry::global();
        (
            m.batched_steps_total.load(Ordering::Relaxed),
            m.batch_fallbacks_total.load(Ordering::Relaxed),
            m.batch_size_sum.load(Ordering::Relaxed),
        )
    }

    /// ① 双 greedy job → 批步交错推进：一次 mock step 返回两 token，两流
    ///    齐进；调用序 create→add×2→step:n:2（×6）→remove×2 完全确定。
    ///
    /// 时序确定性论证：S（sampling，20ms/步）是时序闸门——GA 在 S 的慢
    /// solo 步期间准入（批成员仅 GA → 独行让位一量子，不步进），GB 保证
    /// 在下一量子准入（闸门 20ms ≫ 提交 µs 级 skew）→ 两流自各自首步起
    /// 全程同批（即「纯批成员」，无混合路径 solo 首 token）。
    #[tokio::test]
    async fn two_greedy_jobs_batch_step_in_lockstep() {
        let _batch_guard = BATCH_TESTS.lock().unwrap();
        let engine = Arc::new(
            MockBatchEngine::new((0..500).map(|i| 100 + i).collect())
                .with_pace(Duration::from_millis(20), Duration::from_millis(5)),
        );
        let log = engine.log.clone();
        let sched = batch_sched(engine, 3);
        let handle = sched.handle();
        let m0 = batch_metrics();

        // S（sampling，20ms/步）作时序闸门：GA 在 S 的慢 solo 步期间准入
        //（批成员仅 GA → **独行让位**一量子不步进），GB 保证在下一量子准
        // 入（闸门的 20ms ≫ 提交的 µs 级 skew）→ 两流自首步起全程同批。
        let (rxs, _) = handle.generate(GenerationJob {
            prompt_ids: vec![9],
            config: scfg(8),
        });
        let (rxa, _) = handle.generate(GenerationJob {
            prompt_ids: vec![0],
            config: gcfg(6),
        });
        let (rxb, _) = handle.generate(GenerationJob {
            prompt_ids: vec![1],
            config: gcfg(6),
        });
        let es = collect(rxs).await;
        let ea = collect(rxa).await;
        let eb = collect(rxb).await;

        // 内容：GA/GB 全部 token 来自批步（独行让位消除混合路径的 solo 首
        // token）；S（sampler）纯脚本。批 token 与 solo token 段位可区分。
        assert_eq!(tokens_of(&ea), vec![901, 902, 903, 904, 905, 906]);
        assert_eq!(tokens_of(&eb), vec![1001, 1002, 1003, 1004, 1005, 1006]);
        assert_eq!(tokens_of(&es), (0..8).map(|i| 100 + i).collect::<Vec<_>>());
        for e in [&ea, &eb, &es] {
            assert!(matches!(
                e.last(),
                Some(GenEvent::Done {
                    reason: FinishReason::Length,
                    ..
                })
            ));
        }

        // 调用序：组创建一次、add 两槽（S 从不入组）、6 次双成员步、两成
        // 员同量子齐满退役（各 6 token）→ remove×2。
        assert_eq!(
            log.snapshot(),
            vec![
                "create", "add:0", "add:1", "step:1:2", "step:2:2", "step:3:2", "step:4:2",
                "step:5:2", "step:6:2", "remove:0", "remove:1",
            ]
        );

        // metrics：6 次批步、批大小和 12、零回退。
        let m1 = batch_metrics();
        assert_eq!(m1.0 - m0.0, 6, "batched_steps delta");
        assert_eq!(m1.1 - m0.1, 0, "no fallbacks");
        assert_eq!(m1.2 - m0.2, 12, "batch_size_sum delta");
    }

    /// ② greedy + sampling 混合：sampler 恒 solo（脚本 token 全 100 段，
    ///    无独行让位——让位只作用于 batchable 流），greedy 两条照常批
    ///    （900 段）；sampler 从不入组（log 只有 2 次 add）。
    #[tokio::test]
    async fn sampling_stays_solo_while_greedy_batch() {
        let _batch_guard = BATCH_TESTS.lock().unwrap();
        let engine = Arc::new(
            MockBatchEngine::new((0..500).map(|i| 100 + i).collect())
                .with_pace(Duration::from_millis(20), Duration::from_millis(5)),
        );
        let log = engine.log.clone();
        let sched = batch_sched(engine.clone(), 3);
        let handle = sched.handle();

        let (rxs, _) = handle.generate(GenerationJob {
            prompt_ids: vec![9],
            config: scfg(8),
        });
        let (rxa, _) = handle.generate(GenerationJob {
            prompt_ids: vec![0],
            config: gcfg(6),
        });
        let (rxb, _) = handle.generate(GenerationJob {
            prompt_ids: vec![1],
            config: gcfg(6),
        });
        let es = collect(rxs).await;
        let ea = collect(rxa).await;
        let eb = collect(rxb).await;

        // sampler 全程 solo：token 全 100 段（与脚本逐 token 一致）。
        assert_eq!(tokens_of(&es), (0..8).map(|i| 100 + i).collect::<Vec<_>>());
        assert!(matches!(
            es.last(),
            Some(GenEvent::Done {
                reason: FinishReason::Length,
                ..
            })
        ));
        // greedy 两流：纯批成员（独行让位后自首步起同批）。
        assert_eq!(tokens_of(&ea), vec![901, 902, 903, 904, 905, 906]);
        assert_eq!(tokens_of(&eb), vec![1001, 1002, 1003, 1004, 1005, 1006]);
        // 组成员只有两条 greedy（sampler 从未 add）。
        assert_eq!(log.count_starting_with("add:"), 2);
        // 无 3 成员步（solo 的 sampler 从不入组；step 记录格式 "step:n:m"）。
        assert!(log.snapshot().iter().all(|e| !e.ends_with(":3")));
        // solo decode 全部来自 sampler（greedy 两流零 solo 步）。
        assert_eq!(engine.stats().decode_calls, 8);
    }

    /// ③ 成员退出（短 job Done）：remove 在其最后一次批步后、长 job 继续
    ///    批推进；A/B/C 全程零 solo decode（engine decode_calls == 16，
    ///    全部来自 sampler 闸门）——批做了三条 greedy 流的全部推进。
    #[tokio::test]
    async fn short_job_leaves_group_long_jobs_continue() {
        let _batch_guard = BATCH_TESTS.lock().unwrap();
        let engine = Arc::new(
            MockBatchEngine::new((0..500).map(|i| 100 + i).collect())
                .with_pace(Duration::from_millis(20), Duration::from_millis(5)),
        );
        let log = engine.log.clone();
        let sched = batch_sched(engine.clone(), 4);
        let handle = sched.handle();

        // S（sampling，20ms/步）时序闸门 + A/B/C 三条 greedy。
        let (rxs, _) = handle.generate(GenerationJob {
            prompt_ids: vec![9],
            config: scfg(16),
        });
        let (rxa, _) = handle.generate(GenerationJob {
            prompt_ids: vec![0],
            config: gcfg(4),
        });
        let (rxb, _) = handle.generate(GenerationJob {
            prompt_ids: vec![1],
            config: gcfg(12),
        });
        let (rxc, _) = handle.generate(GenerationJob {
            prompt_ids: vec![2],
            config: gcfg(11),
        });
        let es = collect(rxs).await;
        let ea = collect(rxa).await;
        let eb = collect(rxb).await;
        let ec = collect(rxc).await;

        // A（4）/B（12）/C（11）全部 token 来自批步（独行让位消除了混合
        // 路径 solo 首 token；C 晚一量子入批故自 step2 起）。
        assert_eq!(tokens_of(&ea), vec![901, 902, 903, 904]);
        assert_eq!(
            tokens_of(&eb),
            (1..=12).map(|k| 1000 + k).collect::<Vec<_>>()
        );
        assert_eq!(
            tokens_of(&ec),
            (2..=12).map(|k| 1100 + k).collect::<Vec<_>>()
        );
        for e in [&ea, &eb, &ec, &es] {
            assert!(matches!(
                e.last(),
                Some(GenEvent::Done {
                    reason: FinishReason::Length,
                    ..
                })
            ));
        }

        // A/B/C 全部推进来自批步：solo decode 只有 sampler 的 16 次。
        assert_eq!(
            engine.stats().decode_calls,
            16,
            "greedy jobs must advance purely via batch steps"
        );

        // 调用序：A 退役的 remove:0 出现在 step:4:3 之后、step:5:2 之前。
        let log = log.snapshot();
        let i_step4 = log.iter().position(|e| e == "step:4:3").unwrap();
        let i_remove0 = log.iter().position(|e| e.as_str() == "remove:0").unwrap();
        let i_step5 = log.iter().position(|e| e == "step:5:2").unwrap();
        assert!(i_step4 < i_remove0 && i_remove0 < i_step5, "log={log:?}");
        assert_eq!(log.iter().filter(|e| e.as_str() == "remove:0").count(), 1);
        assert_eq!(log.iter().filter(|e| e.as_str() == "remove:1").count(), 1);
        assert_eq!(log.iter().filter(|e| e.as_str() == "remove:2").count(), 1);
        // 三成员步至少出现过一次（A+B+C 同量子）。
        assert!(log.iter().any(|e| e == "step:2:3"));
    }

    /// ④ group.step Err → 本量子成员全部回退 solo：不崩、内容零丢失
    ///    （纯脚本 token）、组销毁后下量子重建重试（create ≥ 5）、
    ///    batch_fallbacks_total 计数。
    #[tokio::test]
    async fn batch_step_error_falls_back_to_solo() {
        let _batch_guard = BATCH_TESTS.lock().unwrap();
        let engine = Arc::new(
            MockBatchEngine::new((0..500).map(|i| 100 + i).collect())
                .fail_step()
                .with_pace(Duration::from_millis(20), Duration::from_millis(1)),
        );
        let log = engine.log.clone();
        let sched = batch_sched(engine, 3);
        let handle = sched.handle();
        let m0 = batch_metrics();

        // S（sampling，20ms/步）时序闸门 + 两条 greedy。
        let (rxs, _) = handle.generate(GenerationJob {
            prompt_ids: vec![9],
            config: scfg(8),
        });
        let (rxa, _) = handle.generate(GenerationJob {
            prompt_ids: vec![0],
            config: gcfg(6),
        });
        let (rxb, _) = handle.generate(GenerationJob {
            prompt_ids: vec![1],
            config: gcfg(6),
        });
        let es = collect(rxs).await;
        let ea = collect(rxa).await;
        let eb = collect(rxb).await;

        // 内容零丢失：两流纯 solo 脚本前 6 token。
        assert_eq!(tokens_of(&ea), (0..6).map(|i| 100 + i).collect::<Vec<_>>());
        assert_eq!(tokens_of(&eb), (0..6).map(|i| 100 + i).collect::<Vec<_>>());
        assert_eq!(tokens_of(&es), (0..8).map(|i| 100 + i).collect::<Vec<_>>());
        for e in [&ea, &eb, &es] {
            assert!(matches!(
                e.last(),
                Some(GenEvent::Done {
                    reason: FinishReason::Length,
                    ..
                })
            ));
        }

        // 每个失败量子（GB 入伙后共 6 个）：建组 → add×2 → step 失败 →
        // 全员出组 → 回退 solo（各推进 1 token；GA 首量子独行让位等到了
        // GB → 首个失败量子就双成员）。组每量子空 → destroy → 下量子重建。
        assert_eq!(
            log.count_starting_with("create"),
            6,
            "log={:?}",
            log.snapshot()
        );
        assert_eq!(log.count_starting_with("remove"), 12); // 每失败量子 2 次 × 6
        let m1 = batch_metrics();
        assert_eq!(m1.0 - m0.0, 0, "no successful batch steps");
        assert_eq!(m1.1 - m0.1, 6, "fallbacks counted");
    }

    /// ⑤ MC_BATCH_OFF（`batching=false`）→ 行为与旧版逐事件一致：引擎
    ///    虽支持批，但零批调用（log 空、批 metrics 零增量），两流事件序
    ///    与纯 MockEngine（S2 路径）的参考运行逐字节相等。
    #[tokio::test]
    async fn batch_off_matches_legacy_event_sequences() {
        let _batch_guard = BATCH_TESTS.lock().unwrap();
        let script: Vec<i32> = (0..500).map(|i| 100 + i).collect();
        let mk_job = |pid: i32| GenerationJob {
            prompt_ids: vec![pid],
            config: gcfg(6),
        };

        // 参考：纯 MockEngine（无批能力 → 全 solo，S2 行为）。
        let ref_sched = Scheduler::new(scripted(script.clone()), 2).unwrap();
        let (ra, _) = ref_sched.handle().generate(mk_job(0));
        let (rb, _) = ref_sched.handle().generate(mk_job(1));
        let ref_a = collect(ra).await;
        let ref_b = collect(rb).await;
        drop(ref_sched);

        // 批引擎 + batching=false（MC_BATCH_OFF 的配置化等价物）。
        let engine = Arc::new(MockBatchEngine::new(script));
        let log = engine.log.clone();
        let sched = Scheduler::with_config(
            engine,
            SchedulerConfig {
                max_sessions: 2,
                batching: false,
                ..SchedulerConfig::default()
            },
        )
        .unwrap();
        let handle = sched.handle();
        let m0 = batch_metrics();
        let (ra, _) = handle.generate(mk_job(0));
        let (rb, _) = handle.generate(mk_job(1));
        let ea = collect(ra).await;
        let eb = collect(rb).await;

        // 逐事件一致（含 Done.usage）。
        assert_eq!(ea, ref_a, "stream A must equal legacy events");
        assert_eq!(eb, ref_b, "stream B must equal legacy events");
        // 零批调用。
        assert!(log.snapshot().is_empty(), "no batch calls allowed");
        let m1 = batch_metrics();
        assert_eq!(m1.0 - m0.0, 0);
        assert_eq!(m1.1 - m0.1, 0);
        assert_eq!(m1.2 - m0.2, 0);
    }

    /// ⑥ MockEngine（无 BatchEngine 能力）→ 全 solo：构造性零回归的显式
    ///    证据（既有 14 测试原样绿是整体证明）。
    #[tokio::test]
    async fn plain_mock_engine_stays_solo() {
        let _batch_guard = BATCH_TESTS.lock().unwrap();
        let engine = scripted((0..500).map(|i| 100 + i).collect());
        let sched = Scheduler::with_config(
            engine,
            SchedulerConfig {
                max_sessions: 2,
                batching: true, // 开着也没用：引擎无批能力
                ..SchedulerConfig::default()
            },
        )
        .unwrap();
        let handle = sched.handle();
        let m0 = batch_metrics();
        let (ra, _) = handle.generate(GenerationJob {
            prompt_ids: vec![0],
            config: gcfg(6),
        });
        let (rb, _) = handle.generate(GenerationJob {
            prompt_ids: vec![1],
            config: gcfg(6),
        });
        let ea = collect(ra).await;
        let eb = collect(rb).await;
        assert_eq!(tokens_of(&ea), (0..6).map(|i| 100 + i).collect::<Vec<_>>());
        assert_eq!(tokens_of(&eb), (0..6).map(|i| 100 + i).collect::<Vec<_>>());
        let m1 = batch_metrics();
        assert_eq!(m1.0 - m0.0, 0, "mock engine must never batch");
    }

    /// ⑦ 容量截断：max_sessions=10 > ABI 上限 8 → 组 max_slots=8；第 9 条
    ///    流容量外 solo，前 8 条批推进（每步成员 ≤ 8 且出现过 8 成员步；
    ///    成员资格随退役轮转——第 9 条在槽位释放后可入批）。
    #[tokio::test]
    async fn batch_capacity_caps_at_eight_slots() {
        let _batch_guard = BATCH_TESTS.lock().unwrap();
        let engine = Arc::new(
            MockBatchEngine::new((0..500).map(|i| 100 + i).collect())
                .with_pace(Duration::from_millis(5), Duration::from_millis(5)),
        );
        let log = engine.log.clone();
        let sched = batch_sched(engine.clone(), 10);
        let handle = sched.handle();

        let mut rxs = Vec::new();
        for pid in 0..9 {
            let (rx, _) = handle.generate(GenerationJob {
                prompt_ids: vec![pid],
                config: gcfg(10),
            });
            rxs.push(rx);
        }
        let streams: Vec<Vec<GenEvent>> = futures_collect_all(rxs).await;

        // 全部 9 条：各 10 token + Done(Length)。
        for (i, s) in streams.iter().enumerate() {
            assert_eq!(tokens_of(s).len(), 10, "stream {i}");
            assert!(matches!(
                s.last(),
                Some(GenEvent::Done {
                    reason: FinishReason::Length,
                    ..
                })
            ));
        }
        // 每条流至少经历一次 solo（准入量子批成员 <2——单流首个量子必然
        // 独占）与至少一次批（9 条流里 8 条可批，轮转成员资格）。
        for (i, s) in streams.iter().enumerate() {
            let toks = tokens_of(s);
            assert!(
                toks.iter().any(|&t| t >= 901),
                "stream {i} batch tokens: {toks:?}"
            );
        }
        // 容量不变量：每次批步成员 ≤ 8（ABI 上限），且出现过 8 成员步。
        let steps: Vec<String> = log
            .snapshot()
            .into_iter()
            .filter(|e| e.starts_with("step:"))
            .collect();
        assert!(!steps.is_empty());
        for e in &steps {
            let m: usize = e.rsplit(':').next().unwrap().parse().unwrap();
            assert!(m <= 8, "batch over capacity: {e}");
        }
        assert!(
            steps.iter().any(|e| e.ends_with(":8")),
            "must observe an 8-member batch step: {steps:?}"
        );
        // solo decode 有界：爬坡期（独行让位后仍独行/容量外）的少量步 +
        // g9 容量外的 solo 步——远小于 9 流 × 10 token 的总量。
        assert!(engine.stats().decode_calls <= 20);
    }

    /// MC_BATCH_OFF env → SchedulerConfig::default 的批开关（逃生门布线；
    /// 本测试随进程实际 env 断言两种方向，`MC_BATCH_OFF=1 cargo test` 验
    /// 证关批方向的完整链路）。
    #[test]
    fn batch_env_off_default_wiring() {
        let off = std::env::var("MC_BATCH_OFF").map_or(false, |v| v == "1" || v == "true");
        assert_eq!(
            SchedulerConfig::default().batching,
            !off,
            "default batching must follow MC_BATCH_OFF"
        );
    }

    /// 并发收齐多条流（⑦ 用）。
    async fn futures_collect_all(
        rxs: Vec<tokio::sync::mpsc::Receiver<GenEvent>>,
    ) -> Vec<Vec<GenEvent>> {
        let mut handles = Vec::new();
        for mut rx in rxs {
            handles.push(tokio::spawn(async move {
                let mut out = Vec::new();
                while let Some(ev) = rx.recv().await {
                    out.push(ev);
                }
                out
            }));
        }
        let mut all = Vec::new();
        for h in handles {
            all.push(h.await.unwrap());
        }
        all
    }
}
