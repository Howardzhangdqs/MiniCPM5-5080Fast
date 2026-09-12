//! `POST /v1/chat/completions` — OpenAI-compatible chat completions with
//! streaming SSE (plan §4.6) and non-streaming responses.
//!
//! Event loop per request:
//! ```text
//! scheduler Token(id)
//!   ├─ StopState.check_token  ── stop/eos token → finish(Stop), token not emitted
//!   ├─ StreamDecoder.push     ── id → UTF-8 fragment (FFFD tail withheld)
//!   ├─ StopState.push_text    ── stop-string scan; safe_len window
//!   └─ ToolCallParser.push    ── text → content / tool_call deltas → SSE
//! Done / Failed / channel closed
//!   └─ StreamDecoder.finish + StopState.finish + ToolCallParser.finish
//!      → final chunk (finish_reason + usage) → `data: [DONE]`
//! ```
//! String-level stops live here (not in the scheduler): the scheduler only
//! knows stop *token* ids; the server truncates text and cancels the
//! scheduler job early once a stop condition is satisfied.

use std::convert::Infallible;
use std::sync::Arc;

use axum::body::Body;
use axum::extract::State;
use axum::http::{header, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use futures::StreamExt;
use minicpm_generation::tool_call::{ToolCall, ToolCallEvent, ToolCallParser};
use minicpm_generation::{FinishReason, StopState, StreamDecodeError, StreamDecoder, Usage};
use minicpm_metrics::MetricsRegistry;
use minicpm_scheduler::{GenEvent, GenerationJob, SubmitError, EOS_IDS};
use serde::Deserialize;
use tokio_stream::wrappers::ReceiverStream;

use crate::{error_response, queue_full_response, AppState, PromptBuilder};

// ------------------------------------------------------------ request ----

#[derive(Debug, Deserialize)]
pub struct ChatRequest {
    #[serde(default)]
    pub model: Option<String>,
    pub messages: Vec<serde_json::Value>,
    #[serde(default)]
    pub tools: Option<serde_json::Value>,
    #[serde(default)]
    pub temperature: Option<f64>,
    #[serde(default)]
    pub top_p: Option<f64>,
    #[serde(default)]
    pub top_k: Option<u32>,
    #[serde(default)]
    pub repetition_penalty: Option<f64>,
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default)]
    pub max_tokens: Option<u64>,
    /// Newer OpenAI name for `max_tokens`; wins when both are present.
    #[serde(default)]
    pub max_completion_tokens: Option<u64>,
    #[serde(default)]
    pub stream: Option<bool>,
    /// `stop`: string or list of strings (stop-string semantics).
    #[serde(default)]
    pub stop: Option<StopParam>,
    #[serde(default)]
    pub enable_thinking: Option<bool>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum StopParam {
    One(String),
    Many(Vec<String>),
}

impl StopParam {
    fn into_vec(self) -> Vec<String> {
        match self {
            StopParam::One(s) => vec![s],
            StopParam::Many(v) => v,
        }
    }
}

// ------------------------------------------------------------- handler ----

pub async fn chat_completions(
    State(state): State<AppState>,
    Json(req): Json<ChatRequest>,
) -> Response {
    if req.messages.is_empty() {
        return error_response(StatusCode::BAD_REQUEST, "`messages` must not be empty");
    }

    let prompt_ids =
        match state
            .prompt
            .prompt_ids(&req.messages, req.tools.as_ref(), req.enable_thinking)
        {
            Ok(ids) => ids,
            Err(e) => {
                return error_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("prompt build failed: {e:#}"),
                )
            }
        };
    if prompt_ids.is_empty() {
        return error_response(StatusCode::BAD_REQUEST, "prompt encoded to zero tokens");
    }
    let prompt_len = prompt_ids.len() as u64;

    let config = merge_config(&state, &req);
    let stop_strings = req.stop.map(StopParam::into_vec).unwrap_or_default();

    state
        .metrics
        .requests_total
        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    if req.stream.unwrap_or(false) {
        state
            .metrics
            .stream_requests_total
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    // S3 准入控制：等待队列满 → 429 + Retry-After: 1。
    let (rx, cancel) = match state
        .scheduler
        .try_generate(GenerationJob { prompt_ids, config })
    {
        Ok(v) => v,
        Err(SubmitError::QueueFull) => return queue_full_response(),
    };

    let chunk_id = state.next_chunk_id();
    let created = unix_now();
    let model = state.model_name.clone();

    if req.stream.unwrap_or(false) {
        stream_response(
            state,
            rx,
            cancel,
            stop_strings,
            prompt_len,
            chunk_id,
            created,
            model,
        )
        .await
    } else {
        let result = run_generation(
            state.prompt.clone(),
            state.metrics,
            rx,
            stop_strings,
            prompt_len,
            |_| async { true },
        )
        .await;
        // Non-streaming: if we stopped early (stop string), release the worker.
        cancel.cancel();
        match result.error {
            Some(err) => error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("generation failed: {err}"),
            ),
            None => non_stream_json(result, chunk_id, created, &model),
        }
    }
}

pub(crate) fn merge_config(
    state: &AppState,
    req: &ChatRequest,
) -> minicpm_runtime::GenerationConfig {
    let mut c = state.default_config;
    if let Some(t) = req.temperature {
        c.temperature = t as f32;
    }
    if let Some(p) = req.top_p {
        c.top_p = p as f32;
    }
    if let Some(k) = req.top_k {
        c.top_k = k;
    }
    if let Some(rp) = req.repetition_penalty {
        c.repetition_penalty = rp as f32;
    }
    if let Some(s) = req.seed {
        c.seed = s;
    }
    let max = req.max_completion_tokens.or(req.max_tokens);
    if let Some(m) = max {
        c.max_new_tokens = m.min(u32::MAX as u64) as u32;
    }
    c
}

pub(crate) fn unix_now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

// ------------------------------------------------- shared generation core ----

/// Emissions produced by the generation loop; each protocol surface
/// (`chat`, `anthropic`, `responses`) turns them into its own wire frames —
/// OpenAI chunks, Anthropic SSE events, Responses SSE events.
pub(crate) enum Emission {
    Role,
    Content(String),
    ToolStart {
        index: usize,
        id: String,
        name: String,
    },
    ToolArgs {
        index: usize,
        chunk: String,
    },
    Final {
        reason: String,
        usage: Usage,
    },
    Error(String),
}

pub(crate) struct GenResult {
    pub(crate) content: String,
    pub(crate) tool_calls: Vec<ToolCall>,
    pub(crate) finish_reason: FinishReason,
    pub(crate) usage: Usage,
    pub(crate) error: Option<String>,
}

/// The plan §4.6 loop, shared by streaming and non-streaming paths of every
/// protocol surface (chat / anthropic / responses).
///
/// `emit` receives every delta (role / content / tool-call fragments /
/// final / error) and returns `false` to signal "consumer gone" (client
/// disconnected), which stops the loop early.
pub(crate) async fn run_generation<F, Fut>(
    prompt: Arc<dyn PromptBuilder>,
    // S3 起 TTFT/TPOT 由 worker 侧上报，本循环不再使用 metrics；保留参数
    // 以稳定共享签名（chat / anthropic / responses 三处调用点一致）。
    _metrics: &'static MetricsRegistry,
    mut rx: tokio::sync::mpsc::Receiver<GenEvent>,
    stop_strings: Vec<String>,
    prompt_len: u64,
    mut emit: F,
) -> GenResult
where
    F: FnMut(Emission) -> Fut,
    Fut: std::future::Future<Output = bool>,
{
    // TTFT/TPOT 口径已移 worker 侧（S3：scheduler retire 上报，含排队
    // 等待的完整时延）；SSE 侧不再采样，避免与 worker 侧双重计数。

    // Stop tokens: eos ids from the tokenizer config (== scheduler EOS_IDS,
    // see SpecialTokenIds docs: MiniCPM5-2B eos = [1, 130073]).
    let stop_tokens: Vec<i32> = EOS_IDS.to_vec();
    let mut stop_state = StopState::new(stop_tokens, stop_strings);
    let mut decoder = StreamDecoder::new();
    let mut tool_parser = ToolCallParser::new();
    let mut tool_args = ToolArgStream::default();
    let mut emitted = 0usize; // bytes already forwarded through `emit`
    let mut reason: Option<FinishReason> = None;
    let mut failed: Option<String> = None;
    let mut client_gone = false;

    let mut decode = |ids: &[u32]| -> Result<String, StreamDecodeError> {
        prompt
            .decode_ids(ids)
            .map_err(|e| StreamDecodeError(format!("{e:#}")))
    };

    let _ = emit(Emission::Role).await;

    while let Some(event) = rx.recv().await {
        match event {
            GenEvent::Token(id) => {
                // Stop / eos token: never emitted as text.
                if stop_state.check_token(id).is_some() {
                    reason = Some(FinishReason::Stop);
                    break;
                }

                let fragment = match decoder.push(id as u32, &mut decode) {
                    Ok(f) => f,
                    Err(e) => {
                        failed = Some(e.to_string());
                        break;
                    }
                };
                if let Some(text) = fragment {
                    let hit = stop_state.push_text(&text);
                    let safe = stop_state.safe_len();
                    if safe > emitted {
                        let chunk = stop_state.visible_text(safe)[emitted..safe].to_string();
                        emitted = safe;
                        if !forward_tool_text(
                            &mut tool_parser,
                            &mut tool_args,
                            &chunk,
                            &mut emit,
                            &mut client_gone,
                        )
                        .await
                        {
                            break;
                        }
                    }
                    if hit.is_some() {
                        reason = Some(FinishReason::Stop);
                        break;
                    }
                }
            }
            GenEvent::Done { reason: r, .. } => {
                reason = Some(r);
                break;
            }
            GenEvent::Failed(e) => {
                failed = Some(e);
                break;
            }
        }
    }

    // Final flush: release withheld bytes, close pending tool markup.
    if failed.is_none() && !client_gone {
        match decoder.finish(&mut decode) {
            Ok(Some(text)) => {
                stop_state.push_text(&text);
            }
            Ok(None) => {}
            Err(e) => failed = Some(e.to_string()),
        }
        if failed.is_none() {
            let end = stop_state.finish();
            if end > emitted {
                let chunk = stop_state.visible_text(end)[emitted..end].to_string();
                emitted = end;
                let _ = forward_tool_text(
                    &mut tool_parser,
                    &mut tool_args,
                    &chunk,
                    &mut emit,
                    &mut client_gone,
                )
                .await;
            }
        }
        if failed.is_none() && !client_gone {
            // Close any unterminated <function> markup; ParamValue/End
            // events become the final tool_calls argument fragments.
            let mut pending: Vec<Emission> = Vec::new();
            for ev in tool_parser.finish() {
                push_tool_event(&mut tool_args, ev, &mut pending);
            }
            let _ = emit_pending(pending, &mut emit, &mut client_gone).await;
        }
    }

    let finish_reason = if let Some(_e) = &failed {
        FinishReason::Error
    } else {
        reason.unwrap_or(FinishReason::Cancelled)
    };
    let usage = Usage {
        prompt_tokens: prompt_len,
        completion_tokens: decoder.token_count() as u64,
    };
    let content = stop_state.visible_text(emitted).to_string();
    let tool_calls = tool_parser.calls().to_vec();

    if let Some(err) = &failed {
        if !client_gone {
            emit(Emission::Error(err.clone())).await;
        }
    } else if !client_gone {
        emit(Emission::Final {
            reason: finish_reason.as_str().to_string(),
            usage,
        })
        .await;
    }

    GenResult {
        content,
        tool_calls,
        finish_reason,
        usage,
        error: failed,
    }
}

/// Feed a decoded text chunk through the tool-call parser and emit the
/// resulting deltas. Returns false when the consumer is gone.
async fn forward_tool_text<F, Fut>(
    tool_parser: &mut ToolCallParser,
    tool_args: &mut ToolArgStream,
    chunk: &str,
    emit: &mut F,
    client_gone: &mut bool,
) -> bool
where
    F: FnMut(Emission) -> Fut,
    Fut: std::future::Future<Output = bool>,
{
    let events = tool_parser.push(chunk);
    let mut pending: Vec<Emission> = Vec::new();
    for ev in events {
        push_tool_event(tool_args, ev, &mut pending);
    }
    emit_pending(pending, emit, client_gone).await
}

async fn emit_pending<F, Fut>(pending: Vec<Emission>, emit: &mut F, client_gone: &mut bool) -> bool
where
    F: FnMut(Emission) -> Fut,
    Fut: std::future::Future<Output = bool>,
{
    for em in pending {
        if !emit(em).await {
            *client_gone = true;
            return false;
        }
    }
    true
}

/// Map one [`ToolCallEvent`] to emissions, advancing the incremental
/// JSON-arguments builder.
fn push_tool_event(tool_args: &mut ToolArgStream, ev: ToolCallEvent, out: &mut Vec<Emission>) {
    match ev {
        ToolCallEvent::Text(t) => out.push(Emission::Content(t)),
        ToolCallEvent::ToolCallStart { name } => {
            let index = tool_args.next_index;
            tool_args.begin_call(index);
            out.push(Emission::ToolStart {
                index,
                id: format!("call_{index}"),
                name,
            });
        }
        ToolCallEvent::ParamValue { name, value } => {
            for chunk in tool_args.param(&name, &value) {
                out.push(Emission::ToolArgs {
                    index: tool_args.index,
                    chunk,
                });
            }
        }
        ToolCallEvent::ToolCallEnd(_) => {
            for chunk in tool_args.end_call() {
                out.push(Emission::ToolArgs {
                    index: tool_args.index,
                    chunk,
                });
            }
        }
    }
}

/// Incremental builder for OpenAI `tool_calls[].function.arguments` JSON
/// string fragments. Joined fragments of one call form the same JSON object
/// the parser reports in `ToolCall.arguments` (key order: arrival order).
#[derive(Default)]
struct ToolArgStream {
    /// Index of the call currently streaming.
    index: usize,
    /// Index the next `<function>` will use (== number of completed calls).
    next_index: usize,
    open: bool,
    cur_param: Option<String>,
}

impl ToolArgStream {
    fn begin_call(&mut self, index: usize) {
        self.index = index;
        self.next_index = index + 1;
        self.open = false;
        self.cur_param = None;
    }

    fn param(&mut self, name: &str, value: &str) -> Vec<String> {
        let mut out = Vec::new();
        if self.cur_param.as_deref() == Some(name) {
            // continuation chunk of the same param value
            out.push(escape_json_fragment(value));
            return out;
        }
        if self.cur_param.take().is_some() {
            out.push("\"".to_string()); // close previous value string
        }
        out.push(if self.open {
            ",".to_string()
        } else {
            "{".to_string()
        });
        self.open = true;
        out.push(format!("\"{}\":", escape_json_fragment(name)));
        out.push("\"".to_string()); // open value string
        out.push(escape_json_fragment(value));
        self.cur_param = Some(name.to_string());
        out
    }

    fn end_call(&mut self) -> Vec<String> {
        let mut out = Vec::new();
        if self.cur_param.take().is_some() {
            out.push("\"".to_string());
        }
        out.push(if self.open {
            "}".to_string()
        } else {
            "{}".to_string()
        });
        self.open = false;
        out
    }
}

/// JSON-string-escape a fragment (per-character escaping is context-free,
/// so chunk-wise escaping composes to a correctly escaped whole).
fn escape_json_fragment(s: &str) -> String {
    let quoted = serde_json::to_string(s).unwrap_or_else(|_| "\"\"".to_string());
    quoted.trim_matches('"').to_string()
}

// --------------------------------------------------------- SSE rendering ----

#[derive(Clone)]
struct SseCtx {
    id: String,
    created: u64,
    model: String,
}

impl SseCtx {
    fn chunk(
        &self,
        delta: serde_json::Value,
        finish_reason: Option<&str>,
        usage: Option<&Usage>,
    ) -> String {
        let mut v = serde_json::json!({
            "id": self.id,
            "object": "chat.completion.chunk",
            "created": self.created,
            "model": self.model,
            "choices": [
                {
                    "index": 0,
                    "delta": delta,
                    "finish_reason": finish_reason,
                }
            ],
        });
        if let Some(usage) = usage {
            v["usage"] = usage_json(usage);
        }
        format!("data: {v}\n\n")
    }

    fn render(&self, em: &Emission) -> String {
        match em {
            Emission::Role => self.chunk(
                serde_json::json!({ "role": "assistant", "content": "" }),
                None,
                None,
            ),
            Emission::Content(t) => self.chunk(serde_json::json!({ "content": t }), None, None),
            Emission::ToolStart { index, id, name } => self.chunk(
                serde_json::json!({
                    "tool_calls": [
                        {
                            "index": index,
                            "id": id,
                            "type": "function",
                            "function": { "name": name, "arguments": "" },
                        }
                    ]
                }),
                None,
                None,
            ),
            Emission::ToolArgs { index, chunk } => self.chunk(
                serde_json::json!({
                    "tool_calls": [
                        {
                            "index": index,
                            "function": { "arguments": chunk },
                        }
                    ]
                }),
                None,
                None,
            ),
            Emission::Final { reason, usage } => {
                self.chunk(serde_json::json!({}), Some(reason), Some(usage))
            }
            Emission::Error(msg) => format!(
                "data: {}\n\n",
                serde_json::json!({ "error": { "message": msg, "type": "server_error" } })
            ),
        }
    }
}

pub(crate) fn usage_json(u: &Usage) -> serde_json::Value {
    serde_json::json!({
        "prompt_tokens": u.prompt_tokens,
        "completion_tokens": u.completion_tokens,
        "total_tokens": u.total_tokens(),
    })
}

// ----------------------------------------------------------- SSE response ----

async fn stream_response(
    state: AppState,
    rx: tokio::sync::mpsc::Receiver<GenEvent>,
    cancel: minicpm_scheduler::CancelToken,
    stop_strings: Vec<String>,
    prompt_len: u64,
    chunk_id: String,
    created: u64,
    model: String,
) -> Response {
    let (frame_tx, frame_rx) = tokio::sync::mpsc::channel::<String>(64);
    let prompt = state.prompt.clone();
    let metrics = state.metrics;
    let ctx = SseCtx {
        id: chunk_id,
        created,
        model,
    };

    tokio::spawn(async move {
        {
            let frame_tx = frame_tx.clone();
            let ctx = ctx.clone();
            let emit = move |em: Emission| {
                let frame = ctx.render(&em);
                let tx = frame_tx.clone();
                async move {
                    // false when the client disconnected → stop generating.
                    tx.send(frame).await.is_ok()
                }
            };
            let _ = run_generation(prompt, metrics, rx, stop_strings, prompt_len, emit).await;
        }
        // Release the GPU worker promptly if we ended early (stop string,
        // client gone). No-op when the job already finished.
        cancel.cancel();
        let _ = frame_tx.send("data: [DONE]\n\n".to_string()).await;
    });

    let body_stream = ReceiverStream::new(frame_rx)
        .map(|frame| Ok::<_, Infallible>(axum::body::Bytes::from(frame)));
    Response::builder()
        .header(header::CONTENT_TYPE, "text/event-stream")
        .header(header::CACHE_CONTROL, "no-cache")
        .body(Body::from_stream(body_stream))
        .expect("static response parts")
}

// ------------------------------------------------------ non-stream response ----

fn non_stream_json(result: GenResult, chunk_id: String, created: u64, model: &str) -> Response {
    let mut message = serde_json::json!({
        "role": "assistant",
        "content": result.content,
    });
    if !result.tool_calls.is_empty() {
        let calls: Vec<serde_json::Value> = result.tool_calls.iter().map(call_json).collect();
        message["tool_calls"] = serde_json::Value::Array(calls);
    }
    let body = serde_json::json!({
        "id": chunk_id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": result.finish_reason.as_str(),
            }
        ],
        "usage": usage_json(&result.usage),
    });
    (StatusCode::OK, Json(body)).into_response()
}

/// One parsed [`ToolCall`] in OpenAI `tool_calls` JSON shape.
fn call_json(call: &ToolCall) -> serde_json::Value {
    serde_json::json!({
        "id": call.id,
        "type": "function",
        "function": { "name": call.name, "arguments": call.arguments },
    })
}
