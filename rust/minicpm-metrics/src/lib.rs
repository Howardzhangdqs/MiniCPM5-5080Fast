//! Lightweight metrics registry and tracing init (plan §4.2: minicpm-metrics).
//!
//! P1 scope: atomic counters, a small active-stream gauge and running
//! sum/count/max timings (p50/p95 percentiles are a P7 follow-up; the raw
//! data needed for them is not worth the complexity yet). Zero external
//! metric dependencies — snapshots are plain JSON via `serde_json`.
//!
//! 指标口径变更（S3）：**TTFT/TPOT 从 SSE 侧改 worker 侧度量**——TTFT =
//! first_token_at − enqueued_at（含排队等待，调度器 retire 时上报），
//! 此前 SSE 侧只覆盖「首事件送达客户端」的传播段；TPOT 取每请求均值
//!（(done_at − first_token_at)/(generated−1)），而非逐 token 间隔样本。
//! 失败请求（`Failed`）不入 `recent_requests`（`errors_total` 已覆盖）。

use std::collections::VecDeque;
use std::sync::atomic::{AtomicI64, AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::Duration;

/// `recent_requests` 环形缓冲容量上限。
pub const RECENT_REQUESTS_CAP: usize = 64;

/// Monotonic counters and gauges for the host runtime.
#[derive(Debug, Default)]
pub struct MetricsRegistry {
    pub requests_total: AtomicU64,
    pub stream_requests_total: AtomicU64,
    pub cancelled_total: AtomicU64,
    pub errors_total: AtomicU64,
    pub prompt_tokens_total: AtomicU64,
    pub generated_tokens_total: AtomicU64,
    /// Gauge: currently open streams (inc on start, dec on finish/cancel).
    pub active_streams: AtomicI64,
    pub ttft: TimingStats,
    pub tpot: TimingStats,
    pub prefill_us_total: AtomicU64,
    /// Gauge（S3 准入控制）：等待队列深度快照源——scheduler 提交 +1 /
    /// worker 准入 −1 时镜像 store（权威计数在 scheduler 内部的
    /// `Arc<AtomicI64>`，与 handle/worker 共享）。
    pub queue_depth: AtomicI64,
    /// Gauge（S3）：当前活跃 session 数（worker intake/retire 时更新；
    /// 恒 ≤ max_sessions）。
    pub active_sessions: AtomicI64,
    /// B3 批 decode：成功批步次数（一次 `group.step` 推进 n 条流）。
    pub batched_steps_total: AtomicU64,
    /// B3 批 decode：批段失败（create/add/step 任一 Err）→ 本量子全员回退
    /// solo 的事件数（worker 不崩）。
    pub batch_fallbacks_total: AtomicU64,
    /// B3 批 decode：批步成员数累计（均值 = batch_size_sum / batched_steps_total）。
    pub batch_size_sum: AtomicU64,
    /// 最近完成请求的明细环形缓冲（cap = [`RECENT_REQUESTS_CAP`]；每完成
    /// 请求 1 次锁，每 quantum 0 次锁——gauge 全走 atomic）。
    pub recent_requests: Mutex<VecDeque<RecentReq>>,
}

/// 一条最近完成请求的明细（/v1/stats `recent_requests` 用）。
///
/// 口径（S3，worker 侧）：`ttft_us` = first_token_at − enqueued_at（含
/// 排队等待）；`tpot_us` = (done_at − first_token_at)/(completion_tokens−1)，
/// completion ≤ 1 时记 0。`ts_ms` 为完成时刻（Unix epoch 毫秒）。
#[derive(Debug, Clone)]
pub struct RecentReq {
    pub ttft_us: u64,
    pub tpot_us: u64,
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub finish_reason: String,
    /// 完成时刻（Unix epoch 毫秒）。
    pub ts_ms: u64,
}

impl MetricsRegistry {
    pub fn global() -> &'static MetricsRegistry {
        static REG: OnceLock<MetricsRegistry> = OnceLock::new();
        REG.get_or_init(MetricsRegistry::default)
    }

    pub fn observe_ttft(&self, d: Duration) {
        self.ttft.observe(d);
    }

    pub fn observe_tpot(&self, d: Duration) {
        self.tpot.observe(d);
    }

    /// 追加一条最近完成请求明细；超容量 pop 最旧（FIFO 环形，锁内 O(1)）。
    pub fn push_recent(&self, req: RecentReq) {
        let mut q = self.recent_requests.lock().unwrap();
        q.push_back(req);
        while q.len() > RECENT_REQUESTS_CAP {
            q.pop_front();
        }
    }

    /// JSON snapshot for `/v1/stats`.
    pub fn snapshot(&self) -> serde_json::Value {
        serde_json::json!({
            "requests_total": self.requests_total.load(Ordering::Relaxed),
            "stream_requests_total": self.stream_requests_total.load(Ordering::Relaxed),
            "cancelled_total": self.cancelled_total.load(Ordering::Relaxed),
            "errors_total": self.errors_total.load(Ordering::Relaxed),
            "prompt_tokens_total": self.prompt_tokens_total.load(Ordering::Relaxed),
            "generated_tokens_total": self.generated_tokens_total.load(Ordering::Relaxed),
            "active_streams": self.active_streams.load(Ordering::Relaxed),
            "prefill_us_total": self.prefill_us_total.load(Ordering::Relaxed),
            "ttft_us": self.ttft.snapshot(),
            "tpot_us": self.tpot.snapshot(),
        })
    }

    /// `recent_requests` 的 JSON 数组（旧→新；`/v1/stats` 拼装用）。
    pub fn recent_requests_json(&self) -> serde_json::Value {
        let q = self.recent_requests.lock().unwrap();
        serde_json::Value::Array(
            q.iter()
                .map(|r| {
                    serde_json::json!({
                        "ttft_us": r.ttft_us,
                        "tpot_us": r.tpot_us,
                        "prompt_tokens": r.prompt_tokens,
                        "completion_tokens": r.completion_tokens,
                        "finish_reason": r.finish_reason,
                        "ts_ms": r.ts_ms,
                    })
                })
                .collect(),
        )
    }
}

/// Running timing statistics without histograms: count, total, max, mean.
#[derive(Debug, Default)]
pub struct TimingStats {
    count: AtomicU64,
    sum_us: AtomicU64,
    max_us: AtomicU64,
}

impl TimingStats {
    pub fn observe(&self, d: Duration) {
        let us = d.as_micros() as u64;
        self.count.fetch_add(1, Ordering::Relaxed);
        self.sum_us.fetch_add(us, Ordering::Relaxed);
        self.max_us.fetch_max(us, Ordering::Relaxed);
    }

    pub fn snapshot(&self) -> serde_json::Value {
        let count = self.count.load(Ordering::Relaxed);
        let sum = self.sum_us.load(Ordering::Relaxed);
        serde_json::json!({
            "count": count,
            "sum": sum,
            "max": self.max_us.load(Ordering::Relaxed),
            "mean": sum.checked_div(count).unwrap_or(0),
        })
    }
}

/// Initialize tracing with `RUST_LOG`-style filter support.
pub fn init_tracing(default_level: &str) {
    use tracing_subscriber::EnvFilter;
    let filter =
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(default_level));
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(false)
        .init();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counters_and_gauge() {
        let m = MetricsRegistry::default();
        m.requests_total.fetch_add(3, Ordering::Relaxed);
        m.active_streams.fetch_add(2, Ordering::Relaxed);
        m.active_streams.fetch_sub(1, Ordering::Relaxed);
        let s = m.snapshot();
        assert_eq!(s["requests_total"], 3);
        assert_eq!(s["active_streams"], 1);
        assert_eq!(s["ttft_us"]["count"], 0);
    }

    #[test]
    fn timing_stats_math() {
        let t = TimingStats::default();
        t.observe(Duration::from_micros(100));
        t.observe(Duration::from_micros(300));
        let s = t.snapshot();
        assert_eq!(s["count"], 2);
        assert_eq!(s["sum"], 400);
        assert_eq!(s["max"], 300);
        assert_eq!(s["mean"], 200);
    }

    #[test]
    fn recent_requests_ring_cap() {
        let m = MetricsRegistry::default();
        for i in 0..(RECENT_REQUESTS_CAP + 10) {
            m.push_recent(RecentReq {
                ttft_us: i as u64,
                tpot_us: 0,
                prompt_tokens: 1,
                completion_tokens: 1,
                finish_reason: "stop".into(),
                ts_ms: i as u64,
            });
        }
        let q = m.recent_requests.lock().unwrap();
        assert_eq!(q.len(), RECENT_REQUESTS_CAP);
        // 最旧的 10 条被弹出：队首是原第 10 条。
        assert_eq!(q.front().unwrap().ttft_us, 10);
        assert_eq!(q.back().unwrap().ttft_us, RECENT_REQUESTS_CAP as u64 + 9);
        drop(q);
        let arr = m.recent_requests_json();
        assert_eq!(arr.as_array().unwrap().len(), RECENT_REQUESTS_CAP);
        assert_eq!(arr[0]["ttft_us"], 10);
        assert_eq!(arr[0]["finish_reason"], "stop");
    }
}
