//! `POST /v1/messages` — Anthropic Messages API surface.
//!
//! Request translation: Anthropic requests (`system`, block-array message
//! content, `tool_result` / `tool_use` blocks, `input_schema` tools) become
//! the internal OpenAI-shaped messages consumed by [`crate::PromptBuilder`];
//! generation runs on the shared loop ([`crate::chat::run_generation`]).
//!
//! Rendering:
//! ```text
//! non-stream  → {"id","type":"message","content":[text?, tool_use*],
//!                "stop_reason","usage"} JSON
//! stream      → message_start
//!             → content_block_start / content_block_delta / content_block_stop*
//!             → message_delta (stop_reason + final usage) → message_stop
//! ```
//! Every SSE frame carries an `event:` line (Anthropic requirement) and the
//! stream ends with `message_stop` — there is no `data: [DONE]` sentinel.
//! Anthropic error envelopes (`{"type":"error","error":{…}}`) are produced
//! locally; [`crate::error_response`] is only OpenAI-shaped.

use std::convert::Infallible;

use axum::body::Body;
use axum::extract::State;
use axum::http::{header, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use futures::StreamExt;
use minicpm_generation::FinishReason;
use minicpm_scheduler::{GenEvent, GenerationJob, SubmitError};
use serde::Deserialize;
use tokio_stream::wrappers::ReceiverStream;

use crate::chat::{merge_config, run_generation, ChatRequest, Emission, GenResult, StopParam};
use crate::AppState;

// ------------------------------------------------------------- request ----

/// `POST /v1/messages` request body. Unknown fields are ignored (serde
/// default); `messages` is the only required field.
#[derive(Debug, Deserialize)]
pub struct AnthropicRequest {
    #[serde(default)]
    pub model: Option<String>,
    pub messages: Vec<serde_json::Value>,
    /// `system`: string or `[{type:"text",text},…]` block array.
    #[serde(default)]
    pub system: Option<serde_json::Value>,
    #[serde(default)]
    pub max_tokens: Option<u64>,
    #[serde(default)]
    pub stop_sequences: Option<Vec<String>>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub temperature: Option<f64>,
    #[serde(default)]
    pub top_p: Option<f64>,
    #[serde(default)]
    pub top_k: Option<u32>,
    /// `[{name, description?, input_schema}]`.
    #[serde(default)]
    pub tools: Option<Vec<serde_json::Value>>,
    #[serde(default)]
    pub tool_choice: Option<serde_json::Value>,
}

// ---------------------------------------------------- request translation ----

/// Convert an Anthropic `messages` array into the internal OpenAI-shaped
/// messages consumed by [`crate::PromptBuilder`] (pure function; the
/// internal `tool_calls.function.arguments` must be a JSON *object*, not
/// the OpenAI stringified form — the template calls `.items()` on it, see
/// `minicpm_template::ChatMessage`).
fn convert_messages(messages: &[serde_json::Value]) -> Vec<serde_json::Value> {
    let mut out = Vec::with_capacity(messages.len());
    for msg in messages {
        match msg.get("role").and_then(serde_json::Value::as_str) {
            Some("assistant") => convert_assistant_message(msg, &mut out),
            _ => convert_user_message(msg, &mut out), // user (lenient default)
        }
    }
    out
}

/// `system` (string or text-block array) → a leading internal system
/// message; `None` when absent or empty.
fn system_message(system: Option<&serde_json::Value>) -> Option<serde_json::Value> {
    let text = system.map(blocks_text).unwrap_or_default();
    if text.is_empty() {
        None
    } else {
        Some(serde_json::json!({ "role": "system", "content": text }))
    }
}

/// User turn: `tool_result` blocks become `role:"tool"` messages placed
/// *before* the turn's text message; `text` blocks join into one text
/// message; `image` / `document` blocks are dropped.
fn convert_user_message(msg: &serde_json::Value, out: &mut Vec<serde_json::Value>) {
    let role = msg
        .get("role")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("user");
    let content = msg
        .get("content")
        .cloned()
        .unwrap_or(serde_json::Value::Null);
    let mut tool_results: Vec<serde_json::Value> = Vec::new();
    let mut texts: Vec<String> = Vec::new();
    match &content {
        serde_json::Value::String(s) => texts.push(s.clone()),
        serde_json::Value::Array(blocks) => {
            for block in blocks {
                match block.get("type").and_then(serde_json::Value::as_str) {
                    Some("tool_result") => {
                        let tool_use_id = block
                            .get("tool_use_id")
                            .and_then(serde_json::Value::as_str)
                            .unwrap_or("");
                        let content = block.get("content").map(blocks_text).unwrap_or_default();
                        tool_results.push(serde_json::json!({
                            "role": "tool",
                            "tool_call_id": tool_use_id,
                            "content": content,
                        }));
                    }
                    Some("text") => {
                        if let Some(t) = block.get("text").and_then(serde_json::Value::as_str) {
                            texts.push(t.to_string());
                        }
                    }
                    _ => {} // image / document / …: unsupported, dropped
                }
            }
        }
        _ => {}
    }
    let had_tool_results = !tool_results.is_empty();
    out.extend(tool_results);
    let text = texts.join("\n");
    // A pure tool_result turn contributes no extra user message; any other
    // contentless turn keeps an empty user message (do not drop turns).
    if !text.is_empty() || !had_tool_results {
        out.push(serde_json::json!({ "role": role, "content": text }));
    }
}

/// Assistant turn: `text` blocks join into the message text; `tool_use`
/// blocks merge into one `tool_calls` array (arguments as JSON objects);
/// `thinking` / `redacted_thinking` blocks are dropped. A pure tool_use
/// turn gets `content: null`.
fn convert_assistant_message(msg: &serde_json::Value, out: &mut Vec<serde_json::Value>) {
    let content = msg
        .get("content")
        .cloned()
        .unwrap_or(serde_json::Value::Null);
    let mut texts: Vec<String> = Vec::new();
    let mut tool_calls: Vec<serde_json::Value> = Vec::new();
    match &content {
        serde_json::Value::String(s) => texts.push(s.clone()),
        serde_json::Value::Array(blocks) => {
            for block in blocks {
                match block.get("type").and_then(serde_json::Value::as_str) {
                    Some("text") => {
                        if let Some(t) = block.get("text").and_then(serde_json::Value::as_str) {
                            texts.push(t.to_string());
                        }
                    }
                    Some("tool_use") => {
                        let input = block
                            .get("input")
                            .filter(|v| v.is_object())
                            .cloned()
                            .unwrap_or_else(|| serde_json::json!({}));
                        tool_calls.push(serde_json::json!({
                            "id": block.get("id").and_then(serde_json::Value::as_str).unwrap_or(""),
                            "type": "function",
                            "function": {
                                "name": block.get("name").and_then(serde_json::Value::as_str).unwrap_or(""),
                                "arguments": input,
                            },
                        }));
                    }
                    _ => {} // thinking / redacted_thinking: dropped
                }
            }
        }
        _ => {}
    }
    if tool_calls.is_empty() {
        out.push(serde_json::json!({ "role": "assistant", "content": texts.join("\n") }));
    } else {
        let text = if texts.is_empty() {
            serde_json::Value::Null
        } else {
            serde_json::Value::String(texts.join("\n"))
        };
        out.push(serde_json::json!({
            "role": "assistant",
            "content": text,
            "tool_calls": tool_calls,
        }));
    }
}

/// Text of a content value: strings pass through; block arrays contribute
/// their `text` blocks joined by "\n" (also used for `tool_result.content`).
fn blocks_text(content: &serde_json::Value) -> String {
    match content {
        serde_json::Value::String(s) => s.clone(),
        serde_json::Value::Array(blocks) => blocks
            .iter()
            .filter(|b| b.get("type").and_then(serde_json::Value::as_str) == Some("text"))
            .filter_map(|b| b.get("text").and_then(serde_json::Value::as_str))
            .collect::<Vec<_>>()
            .join("\n"),
        _ => String::new(),
    }
}

/// Anthropic tools (`{name, description?, input_schema}`) → the internal
/// OpenAI function-tool shape (`{type:"function", function:{…}}`).
fn convert_tools(tools: &[serde_json::Value]) -> serde_json::Value {
    serde_json::Value::Array(
        tools
            .iter()
            .map(|tool| {
                let mut function = serde_json::Map::new();
                if let Some(name) = tool.get("name") {
                    function.insert("name".to_string(), name.clone());
                }
                if let Some(description) = tool.get("description") {
                    function.insert("description".to_string(), description.clone());
                }
                if let Some(schema) = tool.get("input_schema") {
                    function.insert("parameters".to_string(), schema.clone());
                }
                serde_json::json!({ "type": "function", "function": function })
            })
            .collect(),
    )
}

/// Anthropic `tool_choice` → the OpenAI equivalent. Mapped and unit-tested,
/// but currently has no internal seam: neither [`ChatRequest`] nor
/// [`PromptBuilder::prompt_ids`] carries a tool_choice field (interface gap
/// reported alongside this module).
#[cfg_attr(not(test), allow(dead_code))]
fn convert_tool_choice(choice: &serde_json::Value) -> serde_json::Value {
    match choice.get("type").and_then(serde_json::Value::as_str) {
        Some("any") => serde_json::json!("required"),
        Some("none") => serde_json::json!("none"),
        Some("tool") => serde_json::json!({
            "type": "function",
            "function": { "name": choice.get("name").cloned().unwrap_or(serde_json::Value::Null) },
        }),
        _ => serde_json::json!("auto"), // "auto" (and lenient default)
    }
}

// ------------------------------------------------------------- handler ----

/// `POST /v1/messages` — Anthropic Messages handler.
pub async fn messages(
    State(state): State<AppState>,
    Json(req): Json<AnthropicRequest>,
) -> Response {
    if req.messages.is_empty() {
        return anthropic_error(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "`messages` must not be empty",
        );
    }

    let model = req
        .model
        .clone()
        .unwrap_or_else(|| state.model_name.clone());
    let stop_strings = req.stop_sequences.clone().unwrap_or_default();

    let mut messages = Vec::with_capacity(req.messages.len() + 1);
    if let Some(sys) = system_message(req.system.as_ref()) {
        messages.push(sys);
    }
    messages.extend(convert_messages(&req.messages));

    let tools = req
        .tools
        .as_deref()
        .filter(|t| !t.is_empty())
        .map(convert_tools);
    let chat_req = ChatRequest {
        model: Some(model.clone()),
        messages,
        tools,
        temperature: req.temperature,
        top_p: req.top_p,
        top_k: req.top_k,
        repetition_penalty: None,
        seed: None,
        max_tokens: req.max_tokens,
        max_completion_tokens: None,
        stream: req.stream,
        stop: req.stop_sequences.map(StopParam::Many),
        enable_thinking: None, // thinking parameters are ignored on this surface
    };

    let prompt_ids =
        match state
            .prompt
            .prompt_ids(&chat_req.messages, chat_req.tools.as_ref(), None)
        {
            Ok(ids) => ids,
            Err(e) => {
                return anthropic_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "api_error",
                    format!("prompt build failed: {e:#}"),
                )
            }
        };
    if prompt_ids.is_empty() {
        return anthropic_error(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "prompt encoded to zero tokens",
        );
    }
    let prompt_len = prompt_ids.len() as u64;

    let config = merge_config(&state, &chat_req);

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

    // S3 准入控制：等待队列满 → 429 + Retry-After: 1（Anthropic 错误体）。
    let (rx, cancel) = match state
        .scheduler
        .try_generate(GenerationJob { prompt_ids, config })
    {
        Ok(v) => v,
        Err(SubmitError::QueueFull) => return queue_full_response_anthropic(),
    };

    let id = state.next_id("msg");

    if req.stream.unwrap_or(false) {
        stream_response(state, rx, cancel, stop_strings, prompt_len, id, model).await
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
            Some(err) => anthropic_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "api_error",
                format!("generation failed: {err}"),
            ),
            None => non_stream_json(result, id, &model),
        }
    }
}

/// Anthropic error envelope — the counterpart of [`crate::error_response`]
/// (see file docs).
/// in `{"type":"error","error":{"type","message"}}` shape (`kind`:
/// "invalid_request_error" | "api_error").
fn anthropic_error(status: StatusCode, kind: &str, message: impl Into<String>) -> Response {
    (
        status,
        Json(serde_json::json!({
            "type": "error",
            "error": { "type": kind, "message": message.into() },
        })),
    )
        .into_response()
}

/// S3 准入控制：等待队列已满 → **HTTP 429 + `Retry-After: 1`**（Anthropic
/// 错误体形状，type=rate_limit_error）。
fn queue_full_response_anthropic() -> Response {
    (
        StatusCode::TOO_MANY_REQUESTS,
        [("Retry-After", "1")],
        Json(serde_json::json!({
            "type": "error",
            "error": {
                "type": "rate_limit_error",
                "message": "queue full: too many waiting requests, retry later",
            },
        })),
    )
        .into_response()
}

// --------------------------------------------------------- SSE rendering ----

/// Anthropic SSE encoder state machine: turns shared [`Emission`]s into
/// `event:`-framed Anthropic events
/// (docs.anthropic.com/en/api/streaming).
///
/// Block lifecycle: a text block opens lazily on the first text delta and
/// closes when a tool block starts (text after a tool block opens a *new*
/// text block with a fresh index); tool blocks stay open until Final.
/// Content-block indices come from the encoder's own counter (`next_block`),
/// a different numbering from the tool-call indices carried by
/// [`Emission::ToolStart`] / [`Emission::ToolArgs`]; `tool_blocks` maps
/// between the two.
struct AnthropicSseEncoder {
    id: String,
    model: String,
    prompt_len: u64,
    /// Currently open text block index, if any.
    text_block: Option<usize>,
    /// Next content-block index (shared across text and tool_use blocks).
    next_block: usize,
    /// Tool-call index (Emission numbering) → content-block index.
    tool_blocks: Vec<Option<usize>>,
    /// Any tool block ever opened → final stop_reason "tool_use".
    saw_tool: bool,
    /// Final/Error already rendered; later emissions produce no frames.
    done: bool,
}

impl AnthropicSseEncoder {
    fn new(id: String, model: String, prompt_len: u64) -> Self {
        Self {
            id,
            model,
            prompt_len,
            text_block: None,
            next_block: 0,
            tool_blocks: Vec::new(),
            saw_tool: false,
            done: false,
        }
    }

    /// Render one emission as zero or more concatenated SSE frames.
    fn render(&mut self, em: &Emission) -> String {
        if self.done {
            return String::new();
        }
        match em {
            Emission::Role => sse_frame(
                "message_start",
                serde_json::json!({
                    "type": "message_start",
                    "message": {
                        "id": self.id,
                        "type": "message",
                        "role": "assistant",
                        "model": self.model,
                        "content": [],
                        "stop_reason": serde_json::Value::Null,
                        "stop_sequence": serde_json::Value::Null,
                        "usage": { "input_tokens": self.prompt_len, "output_tokens": 0 },
                    },
                }),
            ),
            Emission::Content(t) => {
                let mut out = String::new();
                let index = match self.text_block {
                    Some(i) => i,
                    None => {
                        let i = self.next_block;
                        self.next_block += 1;
                        self.text_block = Some(i);
                        out.push_str(&sse_frame(
                            "content_block_start",
                            serde_json::json!({
                                "type": "content_block_start",
                                "index": i,
                                "content_block": { "type": "text", "text": "" },
                            }),
                        ));
                        i
                    }
                };
                out.push_str(&sse_frame(
                    "content_block_delta",
                    serde_json::json!({
                        "type": "content_block_delta",
                        "index": index,
                        "delta": { "type": "text_delta", "text": t },
                    }),
                ));
                out
            }
            Emission::ToolStart { index, id, name } => {
                let mut out = String::new();
                if let Some(i) = self.text_block.take() {
                    out.push_str(&content_block_stop(i));
                }
                let block = self.next_block;
                self.next_block += 1;
                if self.tool_blocks.len() <= *index {
                    self.tool_blocks.resize(*index + 1, None);
                }
                self.tool_blocks[*index] = Some(block);
                self.saw_tool = true;
                out.push_str(&sse_frame(
                    "content_block_start",
                    serde_json::json!({
                        "type": "content_block_start",
                        "index": block,
                        "content_block": { "type": "tool_use", "id": id, "name": name, "input": {} },
                    }),
                ));
                out
            }
            Emission::ToolArgs { index, chunk } => {
                match self.tool_blocks.get(*index).and_then(|b| *b) {
                    Some(block) => sse_frame(
                        "content_block_delta",
                        serde_json::json!({
                            "type": "content_block_delta",
                            "index": block,
                            "delta": { "type": "input_json_delta", "partial_json": chunk },
                        }),
                    ),
                    None => String::new(), // args without a ToolStart: drop
                }
            }
            Emission::Final { reason, usage } => {
                let mut out = String::new();
                // Close every open block (tool_use + trailing text) in block
                // order.
                let mut open: Vec<usize> = self.tool_blocks.iter().filter_map(|b| *b).collect();
                if let Some(i) = self.text_block.take() {
                    open.push(i);
                }
                open.sort_unstable();
                for i in open {
                    out.push_str(&content_block_stop(i));
                }
                let stop_reason = if self.saw_tool {
                    "tool_use"
                } else {
                    match reason.as_str() {
                        "length" => "max_tokens",
                        _ => "end_turn", // stop / cancelled
                    }
                };
                out.push_str(&sse_frame(
                    "message_delta",
                    serde_json::json!({
                        "type": "message_delta",
                        "delta": { "stop_reason": stop_reason, "stop_sequence": serde_json::Value::Null },
                        "usage": {
                            "input_tokens": usage.prompt_tokens,
                            "output_tokens": usage.completion_tokens,
                        },
                    }),
                ));
                out.push_str(&sse_frame(
                    "message_stop",
                    serde_json::json!({ "type": "message_stop" }),
                ));
                self.done = true;
                out
            }
            Emission::Error(msg) => {
                self.done = true;
                sse_frame(
                    "error",
                    serde_json::json!({
                        "type": "error",
                        "error": { "type": "api_error", "message": msg },
                    }),
                )
            }
        }
    }
}

/// One Anthropic SSE frame: `event: <type>\ndata: <json>\n\n` (the `event:`
/// line is mandatory in the Anthropic protocol).
fn sse_frame(event: &str, data: serde_json::Value) -> String {
    format!("event: {event}\ndata: {data}\n\n")
}

fn content_block_stop(index: usize) -> String {
    sse_frame(
        "content_block_stop",
        serde_json::json!({ "type": "content_block_stop", "index": index }),
    )
}

// ----------------------------------------------------------- SSE response ----

async fn stream_response(
    state: AppState,
    rx: tokio::sync::mpsc::Receiver<GenEvent>,
    cancel: minicpm_scheduler::CancelToken,
    stop_strings: Vec<String>,
    prompt_len: u64,
    id: String,
    model: String,
) -> Response {
    let (frame_tx, frame_rx) = tokio::sync::mpsc::channel::<String>(64);
    let prompt = state.prompt.clone();
    let metrics = state.metrics;
    let mut encoder = AnthropicSseEncoder::new(id, model, prompt_len);

    tokio::spawn(async move {
        {
            let frame_tx = frame_tx.clone();
            let emit = move |em: Emission| {
                let frame = encoder.render(&em);
                let tx = frame_tx.clone();
                async move {
                    // No-op frames (post-terminal emissions) are skipped;
                    // false when the client disconnected → stop generating.
                    !frame.is_empty() && tx.send(frame).await.is_ok()
                }
            };
            let _ = run_generation(prompt, metrics, rx, stop_strings, prompt_len, emit).await;
        }
        // Release the GPU worker promptly if we ended early (stop string,
        // client gone). No-op when the job already finished. The Anthropic
        // protocol has no `data: [DONE]` sentinel — `message_stop` closes
        // the stream.
        cancel.cancel();
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

fn non_stream_json(result: GenResult, id: String, model: &str) -> Response {
    let mut content: Vec<serde_json::Value> = Vec::new();
    if !result.content.is_empty() {
        content.push(serde_json::json!({ "type": "text", "text": result.content }));
    }
    for call in &result.tool_calls {
        content.push(serde_json::json!({
            "type": "tool_use",
            "id": call.id,
            "name": call.name,
            "input": parse_tool_arguments(&call.arguments),
        }));
    }
    // The content array is never empty: an empty generation reports one
    // empty text block.
    if content.is_empty() {
        content.push(serde_json::json!({ "type": "text", "text": "" }));
    }

    let stop_reason = if !result.tool_calls.is_empty() {
        "tool_use"
    } else {
        match result.finish_reason {
            FinishReason::Length => "max_tokens",
            _ => "end_turn", // Stop | Cancelled (Error returned earlier)
        }
    };

    let body = serde_json::json!({
        "id": id,
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": serde_json::Value::Null,
        "usage": {
            "input_tokens": result.usage.prompt_tokens,
            "output_tokens": result.usage.completion_tokens,
        },
    });
    (StatusCode::OK, Json(body)).into_response()
}

/// Parse a `ToolCall.arguments` JSON-object string into the Anthropic
/// `tool_use.input` value; malformed or non-object strings degrade to `{}`.
fn parse_tool_arguments(args: &str) -> serde_json::Value {
    serde_json::from_str::<serde_json::Value>(args)
        .ok()
        .filter(|v| v.is_object())
        .unwrap_or_else(|| serde_json::json!({}))
}

// ------------------------------------------------------------------ tests ----

#[cfg(test)]
mod tests {
    use super::*;
    use minicpm_generation::Usage;
    use serde_json::json;

    /// Parse rendered SSE text into `(event, data)` frames.
    fn frames(s: &str) -> Vec<(String, serde_json::Value)> {
        s.split("\n\n")
            .filter(|f| !f.is_empty())
            .map(|f| {
                let mut event = String::new();
                let mut data = String::new();
                for line in f.lines() {
                    if let Some(e) = line.strip_prefix("event: ") {
                        event = e.to_string();
                    } else if let Some(d) = line.strip_prefix("data: ") {
                        data = d.to_string();
                    }
                }
                (
                    event,
                    serde_json::from_str(&data).unwrap_or(serde_json::Value::Null),
                )
            })
            .collect()
    }

    // -- system ------------------------------------------------------------

    #[test]
    fn system_string_becomes_leading_system_message() {
        assert_eq!(
            system_message(Some(&json!("Be terse."))),
            Some(json!({ "role": "system", "content": "Be terse." }))
        );
    }

    #[test]
    fn system_block_array_joins_text_blocks() {
        let sys = json!([
            { "type": "text", "text": "You are a cartographer." },
            { "type": "text", "text": "Answer in metric units." },
        ]);
        let m = system_message(Some(&sys)).unwrap();
        assert_eq!(m["role"], "system");
        assert_eq!(
            m["content"],
            "You are a cartographer.\nAnswer in metric units."
        );
    }

    #[test]
    fn missing_or_empty_system_yields_none() {
        assert_eq!(system_message(None), None);
        assert_eq!(system_message(Some(&json!(""))), None);
        assert_eq!(system_message(Some(&json!([]))), None);
    }

    // -- message conversion --------------------------------------------------

    #[test]
    fn string_content_passes_through() {
        let msgs = convert_messages(&[
            json!({ "role": "user", "content": "hi" }),
            json!({ "role": "assistant", "content": "hello" }),
        ]);
        assert_eq!(
            msgs,
            vec![
                json!({ "role": "user", "content": "hi" }),
                json!({ "role": "assistant", "content": "hello" }),
            ]
        );
    }

    #[test]
    fn tool_result_blocks_become_tool_messages_before_user_text() {
        let msgs = convert_messages(&[json!({
            "role": "user",
            "content": [
                { "type": "tool_result", "tool_use_id": "call_0", "content": "22C" },
                { "type": "tool_result", "tool_use_id": "call_1",
                  "content": [ { "type": "text", "text": "sunny" }, { "type": "image", "source": {} } ] },
                { "type": "text", "text": "Summarize." },
                { "type": "image", "source": {} },
            ]
        })]);
        assert_eq!(
            msgs.len(),
            3,
            "two tool messages + one user text message: {msgs:?}"
        );
        assert_eq!(
            msgs[0],
            json!({ "role": "tool", "tool_call_id": "call_0", "content": "22C" })
        );
        assert_eq!(
            msgs[1],
            json!({ "role": "tool", "tool_call_id": "call_1", "content": "sunny" })
        );
        assert_eq!(msgs[2], json!({ "role": "user", "content": "Summarize." }));
    }

    #[test]
    fn pure_tool_result_turn_adds_no_user_message() {
        let msgs = convert_messages(&[json!({
            "role": "user",
            "content": [ { "type": "tool_result", "tool_use_id": "call_0", "content": "22C" } ]
        })]);
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0]["role"], "tool");
    }

    #[test]
    fn tool_use_blocks_become_tool_calls_with_object_arguments() {
        let msgs = convert_messages(&[json!({
            "role": "assistant",
            "content": [
                { "type": "text", "text": "Let me check." },
                { "type": "thinking", "thinking": "hmm" },
                { "type": "tool_use", "id": "call_0", "name": "get_weather",
                  "input": { "city": "Paris", "unit": "C" } },
                { "type": "tool_use", "id": "call_1", "name": "get_time", "input": {} },
            ]
        })]);
        assert_eq!(msgs.len(), 1, "one assistant turn: {msgs:?}");
        let m = &msgs[0];
        assert_eq!(m["role"], "assistant");
        assert_eq!(m["content"], "Let me check.");
        let calls = m["tool_calls"].as_array().unwrap();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0]["id"], "call_0");
        assert_eq!(calls[0]["type"], "function");
        assert_eq!(calls[0]["function"]["name"], "get_weather");
        assert_eq!(
            calls[0]["function"]["arguments"],
            json!({ "city": "Paris", "unit": "C" }),
            "arguments must be a JSON object, not a string"
        );
        assert_eq!(calls[1]["function"]["name"], "get_time");
    }

    #[test]
    fn pure_tool_use_turn_gets_null_content() {
        let msgs = convert_messages(&[json!({
            "role": "assistant",
            "content": [
                { "type": "tool_use", "id": "call_0", "name": "f", "input": { "x": 1 } },
                { "type": "redacted_thinking", "data": "…" },
            ]
        })]);
        assert_eq!(msgs.len(), 1);
        assert!(msgs[0]["content"].is_null());
        assert!(msgs[0]["tool_calls"].is_array());
    }

    #[test]
    fn assistant_thinking_blocks_dropped() {
        let msgs = convert_messages(&[json!({
            "role": "assistant",
            "content": [
                { "type": "thinking", "thinking": "internal" },
                { "type": "redacted_thinking", "data": "…" },
                { "type": "text", "text": "answer" },
            ]
        })]);
        assert_eq!(
            msgs,
            vec![json!({ "role": "assistant", "content": "answer" })]
        );
    }

    // -- tools / tool_choice --------------------------------------------------

    #[test]
    fn tools_map_to_openai_function_schemas() {
        let tools = convert_tools(&[json!({
            "name": "get_weather",
            "description": "weather lookup",
            "input_schema": { "type": "object", "properties": { "city": { "type": "string" } } }
        })]);
        assert_eq!(
            tools,
            json!([
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "weather lookup",
                        "parameters": { "type": "object", "properties": { "city": { "type": "string" } } }
                    }
                }
            ])
        );
    }

    #[test]
    fn tool_choice_mapping() {
        assert_eq!(
            convert_tool_choice(&json!({ "type": "auto" })),
            json!("auto")
        );
        assert_eq!(
            convert_tool_choice(&json!({ "type": "any" })),
            json!("required")
        );
        assert_eq!(
            convert_tool_choice(&json!({ "type": "none" })),
            json!("none")
        );
        assert_eq!(
            convert_tool_choice(&json!({ "type": "tool", "name": "get_weather" })),
            json!({ "type": "function", "function": { "name": "get_weather" } })
        );
    }

    // -- SSE encoder ----------------------------------------------------------

    #[test]
    fn encoder_start_then_text_deltas() {
        let mut enc = AnthropicSseEncoder::new("msg-9".into(), "model-x".into(), 5);

        let start = enc.render(&Emission::Role);
        assert!(start.starts_with("event: message_start\ndata: "));
        let parsed = frames(&start);
        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0].0, "message_start");
        let m = &parsed[0].1["message"];
        assert_eq!(m["id"], "msg-9");
        assert_eq!(m["model"], "model-x");
        assert_eq!(m["role"], "assistant");
        assert_eq!(m["content"], json!([]));
        assert_eq!(m["usage"], json!({ "input_tokens": 5, "output_tokens": 0 }));

        // First content: block start (text, index 0) + text delta.
        let parsed = frames(&enc.render(&Emission::Content("hi".into())));
        assert_eq!(parsed.len(), 2);
        assert_eq!(parsed[0].0, "content_block_start");
        assert_eq!(parsed[0].1["index"], 0);
        assert_eq!(
            parsed[0].1["content_block"],
            json!({ "type": "text", "text": "" })
        );
        assert_eq!(parsed[1].0, "content_block_delta");
        assert_eq!(parsed[1].1["index"], 0);
        assert_eq!(
            parsed[1].1["delta"],
            json!({ "type": "text_delta", "text": "hi" })
        );

        // Same block stays open: delta only.
        let parsed = frames(&enc.render(&Emission::Content("!".into())));
        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0].1["index"], 0);
    }

    #[test]
    fn encoder_mixed_blocks_final_order_and_stop_reason() {
        let mut enc = AnthropicSseEncoder::new("msg-1".into(), "m".into(), 3);
        let mut all = String::new();
        for em in [
            Emission::Role,
            Emission::Content("hi".into()),
            Emission::ToolStart {
                index: 0,
                id: "call_0".into(),
                name: "f".into(),
            },
            Emission::ToolArgs {
                index: 0,
                chunk: "{}".into(),
            },
            Emission::Content("tail".into()),
            Emission::Final {
                reason: "stop".into(),
                usage: Usage {
                    prompt_tokens: 3,
                    completion_tokens: 7,
                },
            },
        ] {
            all.push_str(&enc.render(&em));
        }

        let parsed = frames(&all);
        let names: Vec<&str> = parsed.iter().map(|(e, _)| e.as_str()).collect();
        assert_eq!(
            names,
            vec![
                "message_start",       // Role
                "content_block_start", // text block 0
                "content_block_delta", // text_delta "hi" @0
                "content_block_stop",  // close 0 (tool starts)
                "content_block_start", // tool_use block 1
                "content_block_delta", // input_json_delta @1
                "content_block_start", // NEW text block 2 after the tool
                "content_block_delta", // text_delta "tail" @2
                "content_block_stop",  // Final: close 1 (block order)
                "content_block_stop",  // Final: close 2
                "message_delta",       // stop_reason + final usage
                "message_stop",
            ]
        );

        // Tool block carries id/name/input; json delta targets block 1.
        assert_eq!(parsed[4].1["index"], 1);
        assert_eq!(
            parsed[4].1["content_block"],
            json!({ "type": "tool_use", "id": "call_0", "name": "f", "input": {} })
        );
        assert_eq!(parsed[5].1["index"], 1);
        assert_eq!(
            parsed[5].1["delta"],
            json!({ "type": "input_json_delta", "partial_json": "{}" })
        );
        // New text block after the tool block uses a fresh index.
        assert_eq!(parsed[6].1["index"], 2);
        // Final closes blocks in index order 1, 2.
        assert_eq!(parsed[8].1["index"], 1);
        assert_eq!(parsed[9].1["index"], 2);
        // Tool blocks appeared → tool_use even though reason is "stop".
        assert_eq!(parsed[10].1["delta"]["stop_reason"], "tool_use");
        assert_eq!(
            parsed[10].1["usage"],
            json!({ "input_tokens": 3, "output_tokens": 7 })
        );
        assert_eq!(parsed[11].1, json!({ "type": "message_stop" }));
    }

    #[test]
    fn encoder_maps_length_to_max_tokens() {
        let mut enc = AnthropicSseEncoder::new("msg-1".into(), "m".into(), 2);
        enc.render(&Emission::Role);
        enc.render(&Emission::Content("x".into()));
        let fin = enc.render(&Emission::Final {
            reason: "length".into(),
            usage: Usage {
                prompt_tokens: 2,
                completion_tokens: 5,
            },
        });
        let parsed = frames(&fin);
        assert_eq!(
            parsed.len(),
            3,
            "block stop + message_delta + message_stop: {parsed:?}"
        );
        assert_eq!(parsed[1].1["delta"]["stop_reason"], "max_tokens");
        assert_eq!(parsed[1].1["usage"]["output_tokens"], 5);
    }

    #[test]
    fn encoder_error_frame_is_terminal() {
        let mut enc = AnthropicSseEncoder::new("msg-1".into(), "m".into(), 0);
        enc.render(&Emission::Role);
        let err = enc.render(&Emission::Error("boom".into()));
        assert!(err.starts_with("event: error\ndata: "));
        assert!(err.contains("api_error"));
        assert_eq!(enc.render(&Emission::Content("late".into())), "");
    }

    // -- non-stream rendering --------------------------------------------------

    #[test]
    fn malformed_tool_arguments_degrade_to_empty_object() {
        assert_eq!(parse_tool_arguments("{\"a\":1}"), json!({ "a": 1 }));
        assert_eq!(parse_tool_arguments("not json"), json!({}));
        assert_eq!(
            parse_tool_arguments("[1,2]"),
            json!({}),
            "non-object → {{}}"
        );
    }

    #[tokio::test]
    async fn non_stream_empty_content_becomes_single_empty_text_block() {
        use http_body_util::BodyExt;
        let result = GenResult {
            content: String::new(),
            tool_calls: Vec::new(),
            finish_reason: FinishReason::Stop,
            usage: Usage {
                prompt_tokens: 3,
                completion_tokens: 0,
            },
            error: None,
        };
        let resp = non_stream_json(result, "msg-1".into(), "m");
        let bytes = resp.into_body().collect().await.unwrap().to_bytes();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["content"], json!([{ "type": "text", "text": "" }]));
        assert_eq!(v["stop_reason"], "end_turn");
        assert_eq!(v["usage"], json!({ "input_tokens": 3, "output_tokens": 0 }));
    }
}
