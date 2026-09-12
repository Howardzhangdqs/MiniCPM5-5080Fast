//! `POST /v1/responses` — OpenAI Responses API surface.
//!
//! Translates Responses-API requests into the internal OpenAI-shaped
//! messages consumed by [`crate::PromptBuilder`], runs the shared
//! generation loop ([`crate::chat::run_generation`]) and renders results
//! back in Responses wire format — non-stream JSON and Responses SSE.
//!
//! Request translation is a whitelist: `instructions` / `input` items /
//! `tools` / `tool_choice` / `temperature` / `top_p` /
//! `max_output_tokens` (chat-style `max_tokens` tolerated) / `stream` map
//! onto the internal shapes; everything else (`store`,
//! `previous_response_id`, `text`, `metadata`, `reasoning`, …) is dropped.
//!
//! Event flow per streaming request ([`ResponsesSseEncoder`]):
//! ```text
//! run_generation Emission
//!   ├─ Role       ── response.created → response.in_progress
//!   ├─ Content(t) ── [output_item.added → content_part.added] → output_text.delta
//!   ├─ ToolStart  ── [close open text item] → output_item.added (function_call)
//!   ├─ ToolArgs   ── function_call_arguments.delta
//!   └─ Final      ── [close open items] → response.completed | response.incomplete
//! Emission::Error ── event: error
//! ```
//! The Responses API has no `data: [DONE]` sentinel — the terminal
//! `response.completed` / `response.incomplete` event closes the stream.

use std::convert::Infallible;

use axum::body::Body;
use axum::extract::State;
use axum::http::{header, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use futures::StreamExt;
use minicpm_generation::tool_call::ToolCall;
use minicpm_generation::{FinishReason, Usage};
use minicpm_scheduler::{CancelToken, GenEvent, GenerationJob, SubmitError};
use serde::Deserialize;
use serde_json::{json, Value};
use tokio_stream::wrappers::ReceiverStream;

use crate::chat::{merge_config, run_generation, unix_now, ChatRequest, Emission, GenResult};
use crate::{error_response, queue_full_response, AppState};

// ------------------------------------------------------------ request ----

/// `POST /v1/responses` request body (whitelisted; unknown fields ignored).
#[derive(Debug, Deserialize)]
pub struct ResponsesRequest {
    #[serde(default)]
    pub model: Option<String>,
    /// Top-level system prompt; becomes a leading `system` message.
    #[serde(default)]
    pub instructions: Option<String>,
    /// `input`: a bare string (one user turn) or an array of typed items.
    #[serde(default)]
    pub input: Option<Value>,
    /// Responses-API name for the generation cap.
    #[serde(default)]
    pub max_output_tokens: Option<u64>,
    /// Chat-style spelling of the cap, tolerated for clients that send the
    /// `/v1/chat/completions` name.
    #[serde(default)]
    pub max_tokens: Option<u64>,
    #[serde(default)]
    pub temperature: Option<f64>,
    #[serde(default)]
    pub top_p: Option<f64>,
    #[serde(default)]
    pub stream: Option<bool>,
    /// Flat Responses function tools (`{type, name, description?, parameters?}`).
    #[serde(default)]
    pub tools: Option<Vec<Value>>,
    #[serde(default)]
    pub tool_choice: Option<Value>,
}

impl ResponsesRequest {
    /// The effective generation cap: `max_output_tokens` wins when both
    /// spellings are present.
    fn effective_max_tokens(&self) -> Option<u64> {
        self.max_output_tokens.or(self.max_tokens)
    }
}

// ------------------------------------------------- request translation ----

/// Translate `instructions` + `input` into the internal OpenAI-shaped
/// message list (pure function, unit-testable).
///
/// * `instructions` → a leading `{role: "system"}` message,
/// * a string `input` → one user message,
/// * an array `input` → one message per item (see [`append_input_item`]),
/// * empty pieces contribute nothing — an empty result is the caller's
///   cue to reject the request.
pub fn input_to_messages(instructions: Option<&str>, input: &Value) -> Vec<Value> {
    let mut messages = Vec::new();
    if let Some(sys) = instructions.filter(|s| !s.is_empty()) {
        messages.push(json!({ "role": "system", "content": sys }));
    }
    match input {
        Value::String(text) => {
            if !text.is_empty() {
                messages.push(json!({ "role": "user", "content": text }));
            }
        }
        Value::Array(items) => {
            for item in items {
                append_input_item(&mut messages, item);
            }
        }
        _ => {}
    }
    messages
}

/// Append one Responses `input` array element as internal message(s).
///
/// `"message"` items map to user/assistant/system turns (content parts are
/// flattened to their text); `"function_call"` items fold into one
/// assistant turn's `tool_calls` (adjacent calls share the turn);
/// `"function_call_output"` becomes a `tool` role message; `"reasoning"`
/// and unknown types are dropped.
fn append_input_item(messages: &mut Vec<Value>, item: &Value) {
    // A bare-string element is user text.
    if let Some(text) = item.as_str() {
        if !text.is_empty() {
            messages.push(json!({ "role": "user", "content": text }));
        }
        return;
    }

    match item.get("type").and_then(|t| t.as_str()) {
        Some("message") => {
            let role = match item.get("role").and_then(|r| r.as_str()).unwrap_or("user") {
                "assistant" => "assistant",
                "system" | "developer" => "system",
                _ => "user",
            };
            let content = item.get("content").map(content_text).unwrap_or_default();
            messages.push(json!({ "role": role, "content": content }));
        }
        Some("function_call") => {
            let call_id = item
                .get("call_id")
                .or_else(|| item.get("id"))
                .and_then(|v| v.as_str())
                .unwrap_or_default();
            let name = item
                .get("name")
                .and_then(|v| v.as_str())
                .unwrap_or_default();
            let arguments = item.get("arguments").and_then(|v| v.as_str()).unwrap_or("");
            let call = json!({
                "id": call_id,
                "type": "function",
                // Internal shape: arguments must be a JSON *object* (the chat
                // template calls `.items()` on it), not the wire string form.
                "function": { "name": name, "arguments": parse_arguments_object(arguments) },
            });
            push_tool_call(messages, call);
        }
        Some("function_call_output") => {
            let call_id = item
                .get("call_id")
                .and_then(|v| v.as_str())
                .unwrap_or_default();
            let output = item.get("output").map(content_text).unwrap_or_default();
            messages.push(json!({ "role": "tool", "tool_call_id": call_id, "content": output }));
        }
        // "reasoning" and anything unrecognized: dropped.
        _ => {}
    }
}

/// Append an OpenAI-shape tool call, folding it into the immediately
/// preceding assistant tool-call turn when contiguous, so parallel
/// `function_call` items land in one `tool_calls` array (the standard
/// OpenAI history shape the chat template expects).
fn push_tool_call(messages: &mut Vec<Value>, call: Value) {
    if let Some(last) = messages.last_mut() {
        if last.get("role") == Some(&Value::String("assistant".into()))
            && last.get("tool_calls").is_some_and(|t| t.is_array())
        {
            if let Some(arr) = last["tool_calls"].as_array_mut() {
                arr.push(call);
                return;
            }
        }
    }
    messages.push(json!({ "role": "assistant", "tool_calls": [call] }));
}

/// Plain text of a Responses content slot: a bare string, or the
/// concatenation of the `text` field of typed parts whose `type` is
/// `input_text` / `output_text` / `text`. Other parts (images, …) are
/// skipped.
fn content_text(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Array(parts) => parts
            .iter()
            .filter_map(|p| match p.get("type").and_then(|t| t.as_str()) {
                Some("input_text" | "output_text" | "text") => {
                    p.get("text").and_then(|t| t.as_str()).map(str::to_owned)
                }
                _ => None,
            })
            .collect::<Vec<_>>()
            .join(""),
        _ => String::new(),
    }
}

/// Parse a wire-form `arguments` JSON string into the object form the
/// internal messages use; anything unparseable (or not an object) becomes
/// `{}`.
fn parse_arguments_object(arguments: &str) -> Value {
    match serde_json::from_str::<Value>(arguments) {
        Ok(v) if v.is_object() => v,
        _ => json!({}),
    }
}

/// Translate flat Responses function tools
/// (`{type: "function", name, description?, parameters?}`) into the nested
/// OpenAI chat shape (`{type: "function", function: {…}}`). Non-function
/// (hosted) tools have no internal equivalent and are dropped. Returns
/// `None` when nothing translates so the field stays absent.
pub fn convert_tools(tools: &[Value]) -> Option<Value> {
    let out: Vec<Value> = tools
        .iter()
        .filter_map(|t| {
            if t.get("type").and_then(|v| v.as_str()) != Some("function") {
                return None;
            }
            let name = t.get("name").and_then(|v| v.as_str())?;
            let mut function = serde_json::Map::new();
            function.insert("name".into(), json!(name));
            if let Some(d) = t.get("description") {
                function.insert("description".into(), d.clone());
            }
            if let Some(p) = t.get("parameters") {
                function.insert("parameters".into(), p.clone());
            }
            Some(json!({ "type": "function", "function": Value::Object(function) }))
        })
        .collect();
    (!out.is_empty()).then_some(Value::Array(out))
}

/// Translate Responses `tool_choice` to the OpenAI chat shape:
/// `"auto"|"none"|"required"` pass through; `{type: "function", name}` →
/// `{type: "function", function: {name}}`. Anything else drops to `None`.
pub fn convert_tool_choice(tool_choice: &Value) -> Option<Value> {
    match tool_choice {
        Value::String(s) if matches!(s.as_str(), "auto" | "none" | "required") => {
            Some(tool_choice.clone())
        }
        Value::Object(o) if o.get("type").and_then(|v| v.as_str()) == Some("function") => o
            .get("name")
            .and_then(|v| v.as_str())
            .map(|n| json!({ "type": "function", "function": { "name": n } })),
        _ => None,
    }
}

// ------------------------------------------------------------- handler ----

/// `POST /v1/responses` — OpenAI Responses handler (plan §4.6 loop shared
/// with `chat` via [`crate::chat::run_generation`]).
pub async fn responses(
    State(state): State<AppState>,
    Json(req): Json<ResponsesRequest>,
) -> Response {
    let messages = input_to_messages(
        req.instructions.as_deref(),
        req.input.as_ref().unwrap_or(&Value::Null),
    );
    if messages.is_empty() {
        return error_response(StatusCode::BAD_REQUEST, "`input` must not be empty");
    }
    let tools = req.tools.as_deref().and_then(convert_tools);

    let prompt_ids = match state.prompt.prompt_ids(&messages, tools.as_ref(), None) {
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

    // Only the sampling knobs the Responses API whitelists; `merge_config`
    // reads them off the shared ChatRequest shape.
    let chat_req = ChatRequest {
        model: None,
        messages: Vec::new(),
        tools: None,
        temperature: req.temperature,
        top_p: req.top_p,
        top_k: None,
        repetition_penalty: None,
        seed: None,
        max_tokens: req.effective_max_tokens(),
        max_completion_tokens: None,
        stream: None,
        stop: None,
        enable_thinking: None,
    };
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

    // S3 准入控制：等待队列满 → 429 + Retry-After: 1。
    let (rx, cancel) = match state
        .scheduler
        .try_generate(GenerationJob { prompt_ids, config })
    {
        Ok(v) => v,
        Err(SubmitError::QueueFull) => return queue_full_response(),
    };

    let id = state.next_id("resp");
    let created = unix_now();
    let model = req
        .model
        .clone()
        .unwrap_or_else(|| state.model_name.clone());

    if req.stream.unwrap_or(false) {
        stream_response(state, rx, cancel, prompt_len, id, created, model).await
    } else {
        let result = run_generation(
            state.prompt.clone(),
            state.metrics,
            rx,
            Vec::new(),
            prompt_len,
            |_| async { true },
        )
        .await;
        // Non-streaming: if we stopped early, release the worker.
        cancel.cancel();
        match result.error {
            Some(err) => error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("generation failed: {err}"),
            ),
            None => non_stream_json(&state, &result, id, created, &model),
        }
    }
}

// ------------------------------------------------------ non-stream response ----

/// Render a finished generation as the Responses API response object.
fn non_stream_json(
    state: &AppState,
    result: &GenResult,
    resp_id: String,
    created: u64,
    model: &str,
) -> Response {
    // FinishReason::Error never reaches this point: the caller checks
    // `result.error` first and answers 500.
    let (status, incomplete_details) = match result.finish_reason {
        FinishReason::Length => ("incomplete", json!({ "reason": "max_output_tokens" })),
        _ => ("completed", Value::Null),
    };

    // Output assembly: text (when non-empty) → message item; each tool call
    // → function_call item; both empty → one message item with empty text.
    let mut output: Vec<Value> = Vec::new();
    if !result.content.is_empty() {
        output.push(message_item(state, &result.content));
    }
    for call in &result.tool_calls {
        output.push(function_call_item(state, call));
    }
    if output.is_empty() {
        output.push(message_item(state, ""));
    }

    let body = json!({
        "id": resp_id,
        "object": "response",
        "created_at": created,
        "status": status,
        "incomplete_details": incomplete_details,
        "model": model,
        "output": output,
        "usage": responses_usage_json(&result.usage),
    });
    (StatusCode::OK, Json(body)).into_response()
}

/// A completed assistant `message` output item.
fn message_item(state: &AppState, text: &str) -> Value {
    json!({
        "type": "message",
        "id": state.next_id("msg"),
        "status": "completed",
        "role": "assistant",
        "content": [{ "type": "output_text", "text": text, "annotations": [] }],
    })
}

/// A completed `function_call` output item; `arguments` keeps the parser's
/// JSON-object string form verbatim.
fn function_call_item(state: &AppState, call: &ToolCall) -> Value {
    json!({
        "type": "function_call",
        "id": state.next_id("fc"),
        "call_id": call.id,
        "name": call.name,
        "arguments": call.arguments,
        "status": "completed",
    })
}

/// Usage in the Responses API shape (OpenAI accounting names).
fn responses_usage_json(u: &Usage) -> Value {
    json!({
        "input_tokens": u.prompt_tokens,
        "input_tokens_details": {},
        "output_tokens": u.completion_tokens,
        "output_tokens_details": {},
        "total_tokens": u.total_tokens(),
    })
}

// --------------------------------------------------------- SSE rendering ----

/// One output item tracked by [`ResponsesSseEncoder`], in output order.
enum Item {
    Message {
        id: String,
        output_index: usize,
        text: String,
        closed: bool,
    },
    FunctionCall {
        index: usize,
        id: String,
        call_id: String,
        name: String,
        output_index: usize,
        arguments: String,
        closed: bool,
    },
}

/// A completed item in Responses wire shape (for the terminal
/// `response.completed` / `response.incomplete` payload).
fn item_json(item: &Item) -> Value {
    match item {
        Item::Message { id, text, .. } => json!({
            "type": "message",
            "id": id,
            "status": "completed",
            "role": "assistant",
            "content": [{ "type": "output_text", "text": text, "annotations": [] }],
        }),
        Item::FunctionCall {
            id,
            call_id,
            name,
            arguments,
            ..
        } => json!({
            "type": "function_call",
            "id": id,
            "call_id": call_id,
            "name": name,
            "arguments": arguments,
            "status": "completed",
        }),
    }
}

/// State machine turning [`Emission`]s into Responses-API SSE frames
/// (`event: <type>\ndata: <json>\n\n`).
///
/// Maintains the `sequence_number` counter (one per emitted event, from 0),
/// the `output_index` assignment across items, and the tool-index →
/// output-index mapping. Item ids ("msg-N" / "fc-N") come from `mint_id` —
/// the [`AppState::next_id`] counter in the handler, a plain closure in
/// tests.
struct ResponsesSseEncoder {
    id: String,
    model: String,
    created_at: u64,
    sequence_number: u64,
    mint_id: Box<dyn Fn(&str) -> String + Send>,
    /// Items in output order (== output_index assignment order).
    items: Vec<Item>,
    next_output_index: usize,
}

impl ResponsesSseEncoder {
    fn new(
        id: String,
        model: String,
        created_at: u64,
        mint_id: Box<dyn Fn(&str) -> String + Send>,
    ) -> Self {
        Self {
            id,
            model,
            created_at,
            sequence_number: 0,
            mint_id,
            items: Vec::new(),
            next_output_index: 0,
        }
    }

    /// Build one frame, stamping `type` + `sequence_number` into the data.
    fn event(&mut self, event_type: &str, mut data: Value) -> String {
        let seq = self.sequence_number;
        self.sequence_number += 1;
        if let Value::Object(map) = &mut data {
            map.insert("type".into(), json!(event_type));
            map.insert("sequence_number".into(), json!(seq));
        }
        format!("event: {event_type}\ndata: {data}\n\n")
    }

    /// The bare response object embedded in lifecycle events.
    fn lifecycle_response(&self) -> Value {
        json!({
            "id": self.id,
            "object": "response",
            "created_at": self.created_at,
            "status": "in_progress",
            "model": self.model,
            "output": [],
        })
    }

    /// The full response object carried by the terminal event.
    fn final_response(&self, status: &str, incomplete_details: Value, usage: &Usage) -> Value {
        let output: Vec<Value> = self.items.iter().map(item_json).collect();
        json!({
            "id": self.id,
            "object": "response",
            "created_at": self.created_at,
            "status": status,
            "incomplete_details": incomplete_details,
            "model": self.model,
            "output": output,
            "usage": responses_usage_json(usage),
        })
    }

    /// Position of the still-open message item, if any.
    fn open_message_pos(&self) -> Option<usize> {
        self.items
            .iter()
            .rposition(|it| matches!(it, Item::Message { closed: false, .. }))
    }

    /// Close the open message item (if any): `output_text.done` →
    /// `content_part.done` → `output_item.done`.
    fn close_open_message(&mut self, frames: &mut Vec<String>) {
        let Some(pos) = self.open_message_pos() else {
            return;
        };
        let (item_id, output_index, text) = match &mut self.items[pos] {
            Item::Message {
                id,
                output_index,
                text,
                closed,
            } => {
                *closed = true;
                (id.clone(), *output_index, text.clone())
            }
            _ => unreachable!("open_message_pos only matches message items"),
        };
        frames.push(self.event(
            "response.output_text.done",
            json!({
                "item_id": item_id,
                "output_index": output_index,
                "content_index": 0,
                "text": text,
            }),
        ));
        frames.push(self.event(
            "response.content_part.done",
            json!({
                "item_id": item_id,
                "output_index": output_index,
                "content_index": 0,
                "part": { "type": "output_text", "text": text, "annotations": [] },
            }),
        ));
        frames.push(self.event(
            "response.output_item.done",
            json!({
                "output_index": output_index,
                "item": {
                    "type": "message",
                    "id": item_id,
                    "status": "completed",
                    "role": "assistant",
                    "content": [{ "type": "output_text", "text": text, "annotations": [] }],
                },
            }),
        ));
    }

    /// Close every still-open function_call item:
    /// `function_call_arguments.done` → `output_item.done`.
    fn close_open_tools(&mut self, frames: &mut Vec<String>) {
        let mut to_close: Vec<(String, String, String, usize, String)> = Vec::new();
        for item in self.items.iter_mut() {
            if let Item::FunctionCall {
                id,
                call_id,
                name,
                output_index,
                arguments,
                closed,
                ..
            } = item
            {
                if !*closed {
                    *closed = true;
                    to_close.push((
                        id.clone(),
                        call_id.clone(),
                        name.clone(),
                        *output_index,
                        arguments.clone(),
                    ));
                }
            }
        }
        for (item_id, call_id, name, output_index, arguments) in to_close {
            frames.push(self.event(
                "response.function_call_arguments.done",
                json!({
                    "item_id": item_id,
                    "output_index": output_index,
                    "arguments": arguments,
                }),
            ));
            frames.push(self.event(
                "response.output_item.done",
                json!({
                    "output_index": output_index,
                    "item": {
                        "type": "function_call",
                        "id": item_id,
                        "call_id": call_id,
                        "name": name,
                        "arguments": arguments,
                        "status": "completed",
                    },
                }),
            ));
        }
    }

    /// Translate one emission into the SSE frames to emit (possibly empty).
    fn render(&mut self, em: &Emission) -> Vec<String> {
        match em {
            Emission::Role => vec![
                self.event(
                    "response.created",
                    json!({ "response": self.lifecycle_response() }),
                ),
                self.event(
                    "response.in_progress",
                    json!({ "response": self.lifecycle_response() }),
                ),
            ],
            Emission::Content(t) => {
                // Empty deltas (e.g. the tool parser's finish flush) open no
                // items and emit no frames — a tool-only response then has a
                // tool-only output array, matching the non-streaming shape.
                if t.is_empty() {
                    return Vec::new();
                }
                let mut frames = Vec::new();
                if self.open_message_pos().is_none() {
                    let item_id = (self.mint_id)("msg");
                    let output_index = self.next_output_index;
                    self.next_output_index += 1;
                    self.items.push(Item::Message {
                        id: item_id.clone(),
                        output_index,
                        text: String::new(),
                        closed: false,
                    });
                    frames.push(self.event(
                        "response.output_item.added",
                        json!({
                            "output_index": output_index,
                            "item": {
                                "type": "message",
                                "id": item_id,
                                "status": "in_progress",
                                "role": "assistant",
                                "content": [],
                            },
                        }),
                    ));
                    frames.push(self.event(
                        "response.content_part.added",
                        json!({
                            "item_id": item_id,
                            "output_index": output_index,
                            "content_index": 0,
                            "part": { "type": "output_text", "text": "", "annotations": [] },
                        }),
                    ));
                }
                let pos = self.open_message_pos().expect("message item just opened");
                let (item_id, output_index) = match &mut self.items[pos] {
                    Item::Message {
                        id,
                        output_index,
                        text,
                        ..
                    } => {
                        text.push_str(t);
                        (id.clone(), *output_index)
                    }
                    _ => unreachable!("open_message_pos only matches message items"),
                };
                frames.push(self.event(
                    "response.output_text.delta",
                    json!({
                        "item_id": item_id,
                        "output_index": output_index,
                        "content_index": 0,
                        "delta": t,
                    }),
                ));
                frames
            }
            Emission::ToolStart { index, id, name } => {
                let mut frames = Vec::new();
                // A tool call ends the text section: close the message item
                // before the function_call item opens.
                self.close_open_message(&mut frames);
                let fc_id = (self.mint_id)("fc");
                let output_index = self.next_output_index;
                self.next_output_index += 1;
                self.items.push(Item::FunctionCall {
                    index: *index,
                    id: fc_id.clone(),
                    call_id: id.clone(),
                    name: name.clone(),
                    output_index,
                    arguments: String::new(),
                    closed: false,
                });
                frames.push(self.event(
                    "response.output_item.added",
                    json!({
                        "output_index": output_index,
                        "item": {
                            "type": "function_call",
                            "id": fc_id,
                            "call_id": id,
                            "name": name,
                            "arguments": "",
                            "status": "in_progress",
                        },
                    }),
                ));
                frames
            }
            Emission::ToolArgs { index, chunk } => {
                let mut target: Option<(String, usize)> = None;
                for item in self.items.iter_mut() {
                    if let Item::FunctionCall {
                        index: i,
                        id,
                        output_index,
                        arguments,
                        closed: false,
                        ..
                    } = item
                    {
                        if *i != *index {
                            continue;
                        }
                        arguments.push_str(chunk);
                        target = Some((id.clone(), *output_index));
                        break;
                    }
                }
                match target {
                    Some((item_id, output_index)) => vec![self.event(
                        "response.function_call_arguments.delta",
                        json!({
                            "item_id": item_id,
                            "output_index": output_index,
                            "delta": chunk,
                        }),
                    )],
                    None => Vec::new(),
                }
            }
            Emission::Final { reason, usage } => {
                let mut frames = Vec::new();
                self.close_open_message(&mut frames);
                self.close_open_tools(&mut frames);
                // Empty generation (no text, no tool calls): synthesize an
                // empty message item so the terminal response object matches
                // the non-streaming shape.
                if self.items.is_empty() {
                    self.items.push(Item::Message {
                        id: (self.mint_id)("msg"),
                        output_index: 0,
                        text: String::new(),
                        closed: true,
                    });
                    self.next_output_index = 1;
                }
                let incomplete = reason.as_str() == "length";
                let (status, event_type, incomplete_details) = if incomplete {
                    (
                        "incomplete",
                        "response.incomplete",
                        json!({ "reason": "max_output_tokens" }),
                    )
                } else {
                    ("completed", "response.completed", Value::Null)
                };
                frames.push(self.event(
                    event_type,
                    json!({ "response": self.final_response(status, incomplete_details, usage) }),
                ));
                frames
            }
            Emission::Error(msg) => vec![self.event(
                "error",
                json!({ "code": "server_error", "message": msg, "param": null }),
            )],
        }
    }
}

// ----------------------------------------------------------- SSE response ----

async fn stream_response(
    state: AppState,
    rx: tokio::sync::mpsc::Receiver<GenEvent>,
    cancel: CancelToken,
    prompt_len: u64,
    resp_id: String,
    created: u64,
    model: String,
) -> Response {
    let (frame_tx, frame_rx) = tokio::sync::mpsc::channel::<String>(64);
    let prompt = state.prompt.clone();
    let metrics = state.metrics;

    tokio::spawn(async move {
        {
            let frame_tx = frame_tx.clone();
            let mut encoder = ResponsesSseEncoder::new(
                resp_id,
                model,
                created,
                Box::new(move |p| state.next_id(p)),
            );
            let emit = move |em: Emission| {
                let frames = encoder.render(&em);
                let tx = frame_tx.clone();
                async move {
                    // false when the client disconnected → stop generating.
                    let mut alive = true;
                    for frame in frames {
                        alive = tx.send(frame).await.is_ok();
                        if !alive {
                            break;
                        }
                    }
                    alive
                }
            };
            let _ = run_generation(prompt, metrics, rx, Vec::new(), prompt_len, emit).await;
        }
        // Release the GPU worker promptly if we ended early (client gone).
        // No-op when the job already finished. The Responses API has no
        // `data: [DONE]` sentinel — the terminal response.completed /
        // response.incomplete event closes the stream.
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

// ---------------------------------------------------------------- tests ----

#[cfg(test)]
mod tests {
    use super::*;

    /// Split one SSE frame into `(event, data)`.
    fn split_frame(frame: &str) -> (String, Value) {
        let mut lines = frame.trim_end().split('\n');
        let event = lines
            .next()
            .and_then(|l| l.strip_prefix("event: "))
            .unwrap_or_else(|| panic!("frame without event line: {frame:?}"));
        let data = lines
            .next()
            .and_then(|l| l.strip_prefix("data: "))
            .unwrap_or_else(|| panic!("frame without data line: {frame:?}"));
        (
            event.to_string(),
            serde_json::from_str(data).expect("data is valid JSON"),
        )
    }

    fn test_encoder() -> ResponsesSseEncoder {
        ResponsesSseEncoder::new(
            "resp-1".to_string(),
            "test-model".to_string(),
            1_700_000_000,
            Box::new(|p| format!("{p}-7")),
        )
    }

    // ---- request translation --------------------------------------------

    #[test]
    fn instructions_become_leading_system_message() {
        let msgs = input_to_messages(Some("be terse"), &json!("hi"));
        assert_eq!(
            msgs,
            vec![
                json!({ "role": "system", "content": "be terse" }),
                json!({ "role": "user", "content": "hi" }),
            ]
        );
        // Empty instructions contribute nothing.
        assert_eq!(input_to_messages(Some(""), &json!("hi")).len(), 1);
    }

    #[test]
    fn input_string_and_array_forms() {
        assert_eq!(
            input_to_messages(None, &json!("hi")),
            vec![json!({ "role": "user", "content": "hi" })]
        );
        assert_eq!(
            input_to_messages(
                None,
                &json!(["plain string item", { "type": "message", "role": "user", "content": "typed" }])
            ),
            vec![
                json!({ "role": "user", "content": "plain string item" }),
                json!({ "role": "user", "content": "typed" }),
            ]
        );
        // Empty pieces contribute nothing.
        assert!(input_to_messages(None, &json!("")).is_empty());
        assert!(input_to_messages(None, &json!([])).is_empty());
    }

    #[test]
    fn message_item_content_parts_and_roles() {
        let msgs = input_to_messages(
            None,
            &json!([
                {
                    "type": "message",
                    "role": "developer",
                    "content": [
                        { "type": "input_text", "text": "a" },
                        { "type": "image_url", "image_url": "x" },
                        { "type": "text", "text": "b" },
                    ]
                },
                { "type": "message", "role": "assistant", "content": [{ "type": "output_text", "text": "c" }] },
                { "type": "message", "role": "system", "content": "d" },
                { "type": "message", "content": [{ "type": "input_text", "text": "no role → user" }] },
            ]),
        );
        assert_eq!(msgs[0], json!({ "role": "system", "content": "ab" }));
        assert_eq!(msgs[1], json!({ "role": "assistant", "content": "c" }));
        assert_eq!(msgs[2], json!({ "role": "system", "content": "d" }));
        assert_eq!(
            msgs[3],
            json!({ "role": "user", "content": "no role → user" })
        );
    }

    #[test]
    fn adjacent_function_calls_fold_into_one_assistant_turn() {
        let msgs = input_to_messages(
            None,
            &json!([
                { "type": "message", "role": "user", "content": "weather?" },
                { "type": "function_call", "call_id": "call_1", "name": "get_weather", "arguments": "{\"city\":\"Paris\"}" },
                { "type": "function_call", "id": "call_2", "name": "get_time", "arguments": "not json" },
                { "type": "function_call_output", "call_id": "call_1", "output": "21C" },
            ]),
        );
        assert_eq!(
            msgs.len(),
            3,
            "two function_call items fold into one turn: {msgs:?}"
        );
        assert_eq!(msgs[1]["role"], "assistant");
        let calls = msgs[1]["tool_calls"].as_array().expect("tool_calls array");
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0]["id"], "call_1");
        assert_eq!(calls[0]["function"]["name"], "get_weather");
        // Internal shape: arguments is a JSON *object*, not a string.
        assert_eq!(
            calls[0]["function"]["arguments"],
            json!({ "city": "Paris" })
        );
        assert_eq!(calls[1]["id"], "call_2"); // `id` tolerated when `call_id` is absent
        assert_eq!(calls[1]["function"]["arguments"], json!({})); // unparseable → {}
        assert_eq!(
            msgs[2],
            json!({ "role": "tool", "tool_call_id": "call_1", "content": "21C" })
        );
    }

    #[test]
    fn function_call_after_output_starts_new_turn() {
        let msgs = input_to_messages(
            None,
            &json!([
                { "type": "function_call", "call_id": "c1", "name": "f", "arguments": "{}" },
                { "type": "function_call_output", "call_id": "c1", "output": "ok" },
                { "type": "function_call", "call_id": "c2", "name": "g", "arguments": "{}" },
            ]),
        );
        assert_eq!(msgs.len(), 3);
        assert_eq!(msgs[0]["tool_calls"].as_array().unwrap().len(), 1);
        assert_eq!(msgs[2]["role"], "assistant");
    }

    #[test]
    fn reasoning_and_unknown_items_dropped() {
        let msgs = input_to_messages(
            None,
            &json!([
                { "type": "reasoning", "summary": [], "content": [] },
                { "type": "web_search_call", "status": "completed" },
                { "type": "message", "role": "user", "content": "hi" },
            ]),
        );
        assert_eq!(msgs, vec![json!({ "role": "user", "content": "hi" })]);
    }

    #[test]
    fn tools_flatten_to_nested_chat_shape() {
        let tools = json!([
            { "type": "function", "name": "get_weather", "description": "d", "parameters": { "type": "object" } },
            { "type": "web_search" },
            { "type": "function", "parameters": {} }, // no name → dropped
        ]);
        let converted = convert_tools(tools.as_array().unwrap()).unwrap();
        assert_eq!(
            converted,
            json!([
                { "type": "function", "function": { "name": "get_weather", "description": "d", "parameters": { "type": "object" } } },
            ])
        );
        // Nothing translates → None (field stays absent).
        assert!(convert_tools(&[json!({ "type": "web_search" })]).is_none());
    }

    #[test]
    fn tool_choice_translation() {
        assert_eq!(convert_tool_choice(&json!("auto")), Some(json!("auto")));
        assert_eq!(convert_tool_choice(&json!("none")), Some(json!("none")));
        assert_eq!(
            convert_tool_choice(&json!("required")),
            Some(json!("required"))
        );
        assert_eq!(convert_tool_choice(&json!("bogus")), None);
        assert_eq!(
            convert_tool_choice(&json!({ "type": "function", "name": "get_weather" })),
            Some(json!({ "type": "function", "function": { "name": "get_weather" } }))
        );
        assert_eq!(convert_tool_choice(&json!({ "type": "function" })), None); // no name
    }

    #[test]
    fn max_tokens_field_name_tolerated() {
        let req: ResponsesRequest =
            serde_json::from_value(json!({ "input": "hi", "max_tokens": 9 })).unwrap();
        assert_eq!(req.effective_max_tokens(), Some(9));
        let req: ResponsesRequest = serde_json::from_value(
            json!({ "input": "hi", "max_output_tokens": 5, "max_tokens": 9 }),
        )
        .unwrap();
        assert_eq!(req.effective_max_tokens(), Some(5)); // max_output_tokens wins
    }

    // ---- SSE encoder ------------------------------------------------------

    #[test]
    fn encoder_event_order_for_text() {
        let mut enc = test_encoder();
        let mut all: Vec<(String, Value)> = Vec::new();
        let emissions = vec![
            Emission::Role,
            Emission::Content("Hello".into()),
            Emission::Content(" world".into()),
            Emission::Final {
                reason: "stop".into(),
                usage: Usage {
                    prompt_tokens: 3,
                    completion_tokens: 2,
                },
            },
        ];
        for em in &emissions {
            for frame in enc.render(em) {
                all.push(split_frame(&frame));
            }
        }
        let types: Vec<&str> = all.iter().map(|(e, _)| e.as_str()).collect();
        assert_eq!(
            types,
            vec![
                "response.created",
                "response.in_progress",
                "response.output_item.added",
                "response.content_part.added",
                "response.output_text.delta",
                "response.output_text.delta",
                "response.output_text.done",
                "response.content_part.done",
                "response.output_item.done",
                "response.completed",
            ]
        );
        // sequence_number runs 0..n; data.type matches the event line.
        for (i, (_, data)) in all.iter().enumerate() {
            assert_eq!(
                data["sequence_number"].as_u64(),
                Some(i as u64),
                "event {i}: {data}"
            );
            assert_eq!(data["type"], types[i]);
        }
        let added = &all[2].1;
        assert_eq!(added["item"]["id"], "msg-7");
        assert_eq!(added["item"]["status"], "in_progress");
        let completed = &all.last().unwrap().1;
        assert_eq!(completed["response"]["status"], "completed");
        assert_eq!(
            completed["response"]["output"][0]["content"][0]["text"],
            "Hello world"
        );
        assert_eq!(completed["response"]["usage"]["total_tokens"], 5);
    }

    #[test]
    fn encoder_length_finishes_incomplete() {
        let mut enc = test_encoder();
        enc.render(&Emission::Role);
        enc.render(&Emission::Content("abc".into()));
        let frames = enc.render(&Emission::Final {
            reason: "length".into(),
            usage: Usage::default(),
        });
        let (event, data) = split_frame(frames.last().unwrap());
        assert_eq!(event, "response.incomplete");
        assert_eq!(data["response"]["status"], "incomplete");
        assert_eq!(
            data["response"]["incomplete_details"]["reason"],
            "max_output_tokens"
        );
    }

    #[test]
    fn encoder_tool_call_event_sequence() {
        let mut enc = test_encoder();
        let mut all: Vec<(String, Value)> = Vec::new();
        let emissions = vec![
            Emission::Role,
            Emission::ToolStart {
                index: 0,
                id: "call_0".into(),
                name: "get_weather".into(),
            },
            Emission::ToolArgs {
                index: 0,
                chunk: "{\"city\":".into(),
            },
            Emission::ToolArgs {
                index: 0,
                chunk: "\"Paris\"}".into(),
            },
            Emission::Final {
                reason: "stop".into(),
                usage: Usage {
                    prompt_tokens: 4,
                    completion_tokens: 6,
                },
            },
        ];
        for em in &emissions {
            for frame in enc.render(em) {
                all.push(split_frame(&frame));
            }
        }
        let types: Vec<&str> = all.iter().map(|(e, _)| e.as_str()).collect();
        assert_eq!(
            types,
            vec![
                "response.created",
                "response.in_progress",
                "response.output_item.added",
                "response.function_call_arguments.delta",
                "response.function_call_arguments.delta",
                "response.function_call_arguments.done",
                "response.output_item.done",
                "response.completed",
            ]
        );
        let added = &all[2].1;
        assert_eq!(added["item"]["type"], "function_call");
        assert_eq!(added["item"]["id"], "fc-7");
        assert_eq!(added["item"]["call_id"], "call_0");
        assert_eq!(added["item"]["name"], "get_weather");
        assert_eq!(added["item"]["status"], "in_progress");
        let joined = format!(
            "{}{}",
            all[3].1["delta"].as_str().unwrap(),
            all[4].1["delta"].as_str().unwrap()
        );
        assert_eq!(joined, "{\"city\":\"Paris\"}");
        let done = &all[5].1;
        assert_eq!(done["arguments"].as_str().unwrap(), joined);
        let item_done = &all[6].1;
        assert_eq!(item_done["item"]["status"], "completed");
        assert_eq!(item_done["item"]["arguments"].as_str().unwrap(), joined);
        let completed = &all.last().unwrap().1;
        assert_eq!(completed["response"]["output"][0]["type"], "function_call");
        assert_eq!(completed["response"]["output"][0]["call_id"], "call_0");
        assert_eq!(
            completed["response"]["output"][0]["arguments"]
                .as_str()
                .unwrap(),
            joined
        );
        assert_eq!(completed["response"]["usage"]["output_tokens"], 6);
    }

    #[test]
    fn encoder_tool_start_closes_open_text_item() {
        let mut enc = test_encoder();
        enc.render(&Emission::Role);
        enc.render(&Emission::Content("thinking".into()));
        let frames = enc.render(&Emission::ToolStart {
            index: 0,
            id: "call_0".into(),
            name: "f".into(),
        });
        let types: Vec<String> = frames.iter().map(|f| split_frame(f).0).collect();
        assert_eq!(
            types,
            vec![
                "response.output_text.done",
                "response.content_part.done",
                "response.output_item.done",
                "response.output_item.added",
            ]
        );
    }

    #[test]
    fn encoder_error_frame_shape() {
        let mut enc = test_encoder();
        let frames = enc.render(&Emission::Error("boom".into()));
        assert_eq!(frames.len(), 1);
        let (event, data) = split_frame(&frames[0]);
        assert_eq!(event, "error");
        assert_eq!(data["type"], "error");
        assert_eq!(data["code"], "server_error");
        assert_eq!(data["message"], "boom");
        assert_eq!(data["param"], Value::Null);
    }

    #[test]
    fn encoder_empty_generation_synthesizes_empty_message() {
        let mut enc = test_encoder();
        enc.render(&Emission::Role);
        let frames = enc.render(&Emission::Final {
            reason: "stop".into(),
            usage: Usage::default(),
        });
        let (_, data) = split_frame(frames.last().unwrap());
        let output = data["response"]["output"].as_array().unwrap();
        assert_eq!(output.len(), 1);
        assert_eq!(output[0]["type"], "message");
        assert_eq!(output[0]["content"][0]["text"], "");
    }
}
