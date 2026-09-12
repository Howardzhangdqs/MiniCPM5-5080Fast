//! `minicpm-server`: OpenAI-compatible HTTP + SSE server over the
//! MiniCPM5-2B inference stack (plan §4).
//!
//! Layout:
//! * [`AppState`] / [`build_router`] — application state + routing,
//! * [`PromptBuilder`] — the tokenizer/template seam (chat messages → token
//!   ids, token ids → text); real implementation in [`prompt`]
//!   ([`prompt::TemplatePromptBuilder`]), scripted test double
//!   [`prompt::ScriptedPromptBuilder`],
//! * `chat` — `POST /v1/chat/completions` (streaming SSE + non-streaming),
//!   the streaming-detokenization event loop of plan §4.6,
//! * `anthropic` — `POST /v1/messages` Anthropic Messages API surface
//!   (request translation + Anthropic SSE rendering over the same loop),
//! * `responses` — `POST /v1/responses` OpenAI Responses API surface
//!   (request translation + Responses SSE rendering over the same loop),
//! * `models` / `health` / `stats` — trivial observability endpoints.
//!
//! Thread model (plan §4.5): HTTP/SSE on Tokio workers, prompt building on
//! the request path (cheap), generation on the dedicated GPU worker behind
//! [`minicpm_scheduler::Scheduler`].

use std::sync::atomic::AtomicU64;
use std::sync::Arc;

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use minicpm_metrics::MetricsRegistry;
use minicpm_runtime::GenerationConfig;
use minicpm_scheduler::SchedulerHandle;

pub mod anthropic;
pub mod chat;
pub mod prompt;
pub mod responses;

pub use prompt::{ScriptedPromptBuilder, TemplatePromptBuilder};

/// Tokenizer + chat-template seam used by the HTTP layer.
///
/// `prompt_ids` renders the conversation through the official chat template
/// and encodes it (`add_special_tokens = false` — the template already emits
/// the bos text; `true` would duplicate token id 0, verified against the
/// real tokenizer). `decode_ids` is the streaming decoder's backend and must
/// be deterministic for the same id window.
///
/// Note: `tools` is passed through to the template (tool schemas shape the
/// prompt and the `<function>` output grammar).
pub trait PromptBuilder: Send + Sync {
    fn prompt_ids(
        &self,
        messages: &[serde_json::Value],
        tools: Option<&serde_json::Value>,
        enable_thinking: Option<bool>,
    ) -> anyhow::Result<Vec<i32>>;

    fn decode_ids(&self, ids: &[u32]) -> anyhow::Result<String>;

    fn bos_token(&self) -> &str;
}

/// Shared application state, cloned into every handler.
#[derive(Clone)]
pub struct AppState {
    pub scheduler: SchedulerHandle,
    pub prompt: Arc<dyn PromptBuilder>,
    pub metrics: &'static MetricsRegistry,
    pub model_name: String,
    pub default_config: GenerationConfig,
    /// Monotonic chunk id counter ("chatcmpl-N").
    chunk_counter: Arc<AtomicU64>,
}

impl AppState {
    pub fn new(
        scheduler: SchedulerHandle,
        prompt: Arc<dyn PromptBuilder>,
        model_name: impl Into<String>,
        default_config: GenerationConfig,
    ) -> Self {
        Self {
            scheduler,
            prompt,
            metrics: MetricsRegistry::global(),
            model_name: model_name.into(),
            default_config,
            chunk_counter: Arc::new(AtomicU64::new(0)),
        }
    }

    pub fn next_chunk_id(&self) -> String {
        let n = self
            .chunk_counter
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            + 1;
        format!("chatcmpl-{n}")
    }

    /// Next monotonic id with a protocol-specific prefix ("msg", "resp", …).
    /// Shares the counter with [`Self::next_chunk_id`]; uniqueness is what
    /// matters, not per-prefix numbering.
    pub fn next_id(&self, prefix: &str) -> String {
        let n = self
            .chunk_counter
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            + 1;
        format!("{prefix}-{n}")
    }
}

/// Build the application router.
pub fn build_router(state: AppState) -> Router {
    Router::new()
        .route("/v1/chat/completions", post(chat::chat_completions))
        .route("/v1/messages", post(anthropic::messages))
        .route("/v1/responses", post(responses::responses))
        .route("/v1/models", get(models))
        .route("/health", get(health))
        .route("/v1/stats", get(stats))
        .with_state(state)
}

/// `GET /v1/models` — OpenAI `list` shape.
async fn models(State(state): State<AppState>) -> Response {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    Json(serde_json::json!({
        "object": "list",
        "data": [
            {
                "id": state.model_name,
                "object": "model",
                "created": now,
                "owned_by": "minicpm",
            }
        ]
    }))
    .into_response()
}

/// `GET /health`.
async fn health() -> Response {
    Json(serde_json::json!({ "status": "ok" })).into_response()
}

/// `GET /v1/stats` — metrics snapshot + scheduler 视角（S3；B3 增 batch 键）。
///
/// 现有 metrics 字段不动；新增两键：
/// * `scheduler`：`queue_depth`（等待队列深度，准入计数镜像）、
///   `active_sessions`（活跃 session gauge）、`max_sessions`（池上限）、
///   `batch`（B3 批 decode：成功批步数 / 回退事件数 / 批大小均值）；
/// * `recent_requests`：最近完成请求明细（cap 64，旧→新；Failed 不入列）。
async fn stats(State(state): State<AppState>) -> Response {
    let m = &state.metrics;
    let batched_steps = m
        .batched_steps_total
        .load(std::sync::atomic::Ordering::Relaxed);
    let batch_size_sum = m.batch_size_sum.load(std::sync::atomic::Ordering::Relaxed);
    let mut snap = m.snapshot();
    snap["scheduler"] = serde_json::json!({
        "queue_depth": m.queue_depth.load(std::sync::atomic::Ordering::Relaxed),
        "active_sessions": m.active_sessions.load(std::sync::atomic::Ordering::Relaxed),
        "max_sessions": state.scheduler.max_sessions(),
        "batch": {
            "batched_steps_total": batched_steps,
            "batch_fallbacks_total": m.batch_fallbacks_total.load(std::sync::atomic::Ordering::Relaxed),
            "batch_size_mean": batch_size_sum.checked_div(batched_steps).unwrap_or(0),
        },
    });
    snap["recent_requests"] = m.recent_requests_json();
    Json(snap).into_response()
}

/// Shared JSON error helper.
pub(crate) fn error_response(status: StatusCode, message: impl Into<String>) -> Response {
    (status, Json(serde_json::json!({ "error": { "message": message.into(), "type": "invalid_request_error" } }))).into_response()
}

/// S3 准入控制：等待队列已满（`SubmitError::QueueFull`）→ **HTTP 429 +
/// `Retry-After: 1`**。错误体风格与 [`error_response`] 一致
///（`{"error": {"message", "type"}}`，type=rate_limit_error）。
pub(crate) fn queue_full_response() -> Response {
    (
        StatusCode::TOO_MANY_REQUESTS,
        [("Retry-After", "1")],
        Json(serde_json::json!({
            "error": {
                "message": "queue full: too many waiting requests, retry later",
                "type": "rate_limit_error",
            }
        })),
    )
        .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    use minicpm_runtime::MockEngine;
    use minicpm_scheduler::Scheduler;

    pub(crate) fn test_state(engine: Arc<MockEngine>) -> AppState {
        let scheduler = Scheduler::new(engine, 2).expect("spawn scheduler");
        AppState::new(
            scheduler.handle(),
            Arc::new(ScriptedPromptBuilder),
            "minicpm5-2b",
            GenerationConfig {
                max_new_tokens: 64,
                ..GenerationConfig::default()
            },
        )
    }

    #[test]
    fn chunk_ids_increase() {
        let engine = Arc::new(MockEngine::new());
        let state = test_state(engine);
        assert_eq!(state.next_chunk_id(), "chatcmpl-1");
        assert_eq!(state.next_chunk_id(), "chatcmpl-2");
    }
}
