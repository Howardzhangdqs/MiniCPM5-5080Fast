//! HTTP integration tests for `POST /v1/messages` (Anthropic Messages API).
//!
//! Same harness as `http.rs`: `ScriptedPromptBuilder` + `MockEngine` +
//! tower oneshot — no model files, no GPU.

use std::sync::Arc;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use axum::Router;
use http_body_util::BodyExt;
use minicpm_runtime::mock::MOCK_EOS;
use minicpm_runtime::{GenerationConfig, MockEngine};
use minicpm_scheduler::Scheduler;
use minicpm_server::prompt::{char_to_id, ScriptedPromptBuilder};
use minicpm_server::{build_router, AppState};
use serde_json::json;
use tower::ServiceExt;

fn app_with(engine: Arc<MockEngine>) -> Router {
    let scheduler = Scheduler::new(engine, 2).expect("scheduler");
    let state = AppState::new(
        scheduler.handle(),
        Arc::new(ScriptedPromptBuilder),
        "minicpm5-2b",
        GenerationConfig {
            max_new_tokens: 64,
            ..GenerationConfig::default()
        },
    );
    build_router(state)
}

/// Script that spells `text`, one token (char) per decode step.
fn text_script(text: &str) -> Vec<i32> {
    text.chars().map(char_to_id).collect()
}

/// `text_script` + a trailing eos token so generation ends naturally
/// (finish "stop" → Anthropic "end_turn", eos neither counted nor leaked).
fn eos_script(text: &str) -> Vec<i32> {
    let mut script = text_script(text);
    script.push(MOCK_EOS);
    script
}

async fn post(app: &Router, body: serde_json::Value) -> (StatusCode, Vec<u8>) {
    let response = app
        .clone()
        .oneshot(
            Request::post("/v1/messages")
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    let status = response.status();
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    (status, bytes.to_vec())
}

fn messages_body(extra: serde_json::Value) -> serde_json::Value {
    let mut body = json!({
        "model": "anthropic-test-model",
        "messages": [{ "role": "user", "content": "hello" }],
    });
    if let (Some(obj), Some(extra)) = (body.as_object_mut(), extra.as_object()) {
        for (k, v) in extra {
            obj.insert(k.clone(), v.clone());
        }
    }
    body
}

/// Parse an Anthropic SSE body into `(event, data)` frames.
fn sse_events(body: &[u8]) -> Vec<(String, serde_json::Value)> {
    String::from_utf8_lossy(body)
        .split("\n\n")
        .filter(|frame| !frame.is_empty())
        .map(|frame| {
            let mut event = String::new();
            let mut data = String::new();
            for line in frame.lines() {
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

// ------------------------------------------------------------------ ① ----

#[tokio::test]
async fn non_stream_basic_shape_and_end_turn() {
    let app = app_with(Arc::new(MockEngine::with_script(eos_script("Bonjour!"))));

    let (status, body) = post(&app, messages_body(json!({ "max_tokens": 64 }))).await;
    assert_eq!(status, StatusCode::OK);

    let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert!(
        v["id"].as_str().unwrap().starts_with("msg-"),
        "id must be msg-N: {v}"
    );
    assert_eq!(v["type"], "message");
    assert_eq!(v["role"], "assistant");
    assert_eq!(
        v["model"], "anthropic-test-model",
        "request model is echoed"
    );
    assert_eq!(v["stop_reason"], "end_turn");
    assert!(v["stop_sequence"].is_null());
    assert_eq!(v["content"].as_array().unwrap().len(), 1);
    assert_eq!(v["content"][0]["type"], "text");
    assert_eq!(v["content"][0]["text"], "Bonjour!");
    assert!(v["usage"]["input_tokens"].as_u64().unwrap() > 0);
    assert_eq!(v["usage"]["output_tokens"], 8, "eos not counted: {v}");
}

// ------------------------------------------------------------------ ② ----

#[tokio::test]
async fn non_stream_max_tokens_reports_max_tokens() {
    let app = app_with(Arc::new(MockEngine::with_script(text_script(
        "Hello, scripted world!",
    ))));

    let (status, body) = post(&app, messages_body(json!({ "max_tokens": 8 }))).await;
    assert_eq!(status, StatusCode::OK);

    let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(v["stop_reason"], "max_tokens");
    assert_eq!(
        v["content"][0]["text"], "Hello, s",
        "8 scripted tokens = 8 chars: {v}"
    );
    assert_eq!(v["usage"]["output_tokens"], 8);
}

// ------------------------------------------------------------------ ③ ----

#[tokio::test]
async fn non_stream_tool_use_from_xml_function_call() {
    let xml = r#"<function name="get_weather"><param name="city">Paris</param></function>"#;
    let app = app_with(Arc::new(MockEngine::with_script(eos_script(xml))));

    let (status, body) = post(&app, messages_body(json!({ "max_tokens": 200 }))).await;
    assert_eq!(status, StatusCode::OK);

    let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
    let blocks = v["content"].as_array().unwrap();
    assert_eq!(
        blocks.len(),
        1,
        "pure tool call → single tool_use block: {v}"
    );
    assert_eq!(blocks[0]["type"], "tool_use");
    assert_eq!(blocks[0]["id"], "call_0");
    assert_eq!(blocks[0]["name"], "get_weather");
    assert_eq!(blocks[0]["input"], json!({ "city": "Paris" }));
    assert_eq!(v["stop_reason"], "tool_use");
}

// ------------------------------------------------------------------ ④ ----

#[tokio::test]
async fn stream_event_sequence_and_usage() {
    let text = "streaming chunk parade!";
    let app = app_with(Arc::new(MockEngine::with_script(eos_script(text))));

    let (status, body) = post(
        &app,
        messages_body(json!({ "stream": true, "max_tokens": 64 })),
    )
    .await;
    assert_eq!(status, StatusCode::OK);

    let raw = String::from_utf8(body.clone()).unwrap();
    assert!(
        !raw.contains("[DONE]"),
        "Anthropic streams have no [DONE] sentinel: {raw:?}"
    );
    assert!(
        !raw.contains("event: ping"),
        "unexpected ping frames: {raw:?}"
    );

    let events = sse_events(&body);
    assert!(
        events.len() >= 5,
        "need start + block start/delta/stop + message_delta + stop: {events:?}"
    );
    assert_eq!(events.first().unwrap().0, "message_start");
    assert_eq!(events.last().unwrap().0, "message_stop");

    let start = &events.first().unwrap().1;
    assert_eq!(start["type"], "message_start");
    assert_eq!(start["message"]["role"], "assistant");
    assert!(start["message"]["id"].as_str().unwrap().starts_with("msg-"));
    assert!(start["message"]["usage"]["input_tokens"].as_u64().unwrap() > 0);
    assert_eq!(start["message"]["usage"]["output_tokens"], 0);
    // Every event payload carries its type field.
    for (_, data) in &events {
        assert!(data.get("type").is_some(), "event data lacks type: {data}");
    }

    let names: Vec<&str> = events.iter().map(|(e, _)| e.as_str()).collect();
    for expected in [
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
        "message_delta",
    ] {
        assert!(
            names.contains(&expected),
            "missing `{expected}` event in {names:?}"
        );
    }

    // Joined text_delta payloads equal the scripted text.
    let joined: String = events
        .iter()
        .filter(|(e, d)| e == "content_block_delta" && d["delta"]["type"] == "text_delta")
        .filter_map(|(_, d)| d["delta"]["text"].as_str().map(str::to_owned))
        .collect();
    assert_eq!(joined, text, "joined text deltas in {names:?}");

    // message_delta carries the final usage and end_turn.
    let delta = events
        .iter()
        .rev()
        .find(|(e, _)| e == "message_delta")
        .unwrap();
    assert_eq!(delta.1["delta"]["stop_reason"], "end_turn");
    assert!(delta.1["usage"]["input_tokens"].as_u64().unwrap() > 0);
    assert_eq!(
        delta.1["usage"]["output_tokens"], 23,
        "len(\"{text}\") tokens"
    );
}

// ------------------------------------------------------------------ ⑤ ----

#[tokio::test]
async fn empty_messages_rejected_with_anthropic_error_envelope() {
    let app = app_with(Arc::new(MockEngine::new()));

    let (status, body) = post(&app, json!({ "model": "x", "messages": [] })).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);

    let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(v["type"], "error");
    assert_eq!(v["error"]["type"], "invalid_request_error");
    assert!(v["error"]["message"].as_str().unwrap().contains("messages"));
}

// ------------------------------------------------------------------ ⑦ ----

#[tokio::test]
async fn stop_sequences_truncate_non_stream() {
    let app = app_with(Arc::new(MockEngine::with_script(text_script(
        "Hello STOP world more",
    ))));

    let (status, body) = post(
        &app,
        messages_body(json!({ "max_tokens": 50, "stop_sequences": ["STOP"] })),
    )
    .await;
    assert_eq!(status, StatusCode::OK);

    let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(
        v["stop_reason"], "end_turn",
        "FinishReason::Stop maps to end_turn"
    );
    assert_eq!(
        v["content"][0]["text"], "Hello ",
        "stop sequence must be withheld: {v}"
    );
    // Tokens decoded until the match completed: H e l l o ' ' S T O P = 10.
    assert_eq!(v["usage"]["output_tokens"], 10);
}

// ------------------------------------------------------------------ ⑥ ----
// Full translation smoke test: system blocks + tools + tool_use /
// tool_result history flow through the prompt builder without error.

#[tokio::test]
async fn system_tools_and_tool_history_translate_into_a_prompt() {
    let app = app_with(Arc::new(MockEngine::with_script(eos_script("ok"))));

    let body = json!({
        "model": "anthropic-test-model",
        "max_tokens": 32,
        "system": [{ "type": "text", "text": "Be terse." }],
        "tools": [{
            "name": "get_weather",
            "description": "weather lookup",
            "input_schema": { "type": "object", "properties": { "city": { "type": "string" } } }
        }],
        "messages": [
            { "role": "user", "content": "weather in Paris?" },
            { "role": "assistant", "content": [
                { "type": "text", "text": "Checking." },
                { "type": "tool_use", "id": "call_0", "name": "get_weather",
                  "input": { "city": "Paris" } }
            ]},
            { "role": "user", "content": [
                { "type": "tool_result", "tool_use_id": "call_0",
                  "content": [{ "type": "text", "text": "22C" }] },
                { "type": "text", "text": "Summarize." }
            ]}
        ]
    });

    let (status, resp) = post(&app, body).await;
    assert_eq!(status, StatusCode::OK);
    let v: serde_json::Value = serde_json::from_slice(&resp).unwrap();
    assert_eq!(v["content"][0]["text"], "ok");
    assert_eq!(v["stop_reason"], "end_turn");
}

// ------------------------------------------------------------------ ⑧ ----
// Streaming tool_use: content_block_start(tool_use) + input_json_delta
// fragments whose concatenation parses to the arguments object.

#[tokio::test]
async fn stream_tool_use_blocks_and_input_json_delta() {
    let xml = r#"<function name="get_weather"><param name="city">Paris</param></function>"#;
    let app = app_with(Arc::new(MockEngine::with_script(eos_script(xml))));

    let (status, body) = post(
        &app,
        messages_body(json!({ "stream": true, "max_tokens": 200 })),
    )
    .await;
    assert_eq!(status, StatusCode::OK);

    let events = sse_events(&body);
    assert_eq!(events.first().unwrap().0, "message_start");
    assert_eq!(events.last().unwrap().0, "message_stop");

    let tool_start = events
        .iter()
        .find(|(e, d)| e == "content_block_start" && d["content_block"]["type"] == "tool_use")
        .expect("tool_use content_block_start");
    assert_eq!(
        tool_start.1["index"], 0,
        "no text block before the tool call"
    );
    assert_eq!(tool_start.1["content_block"]["id"], "call_0");
    assert_eq!(tool_start.1["content_block"]["name"], "get_weather");
    assert_eq!(tool_start.1["content_block"]["input"], json!({}));

    let partials: String = events
        .iter()
        .filter(|(e, d)| e == "content_block_delta" && d["delta"]["type"] == "input_json_delta")
        .filter_map(|(_, d)| d["delta"]["partial_json"].as_str().map(str::to_owned))
        .collect();
    let args: serde_json::Value = serde_json::from_str(&partials)
        .unwrap_or_else(|e| panic!("joined partial_json `{partials}` not JSON: {e}"));
    assert_eq!(args["city"], "Paris");

    let message_delta = events
        .iter()
        .rev()
        .find(|(e, _)| e == "message_delta")
        .unwrap();
    assert_eq!(message_delta.1["delta"]["stop_reason"], "tool_use");

    // The tool block is closed before message_delta.
    let stop_idx = events
        .iter()
        .position(|(e, d)| e == "content_block_stop" && d["index"] == 0)
        .expect("content_block_stop for the tool block");
    let delta_idx = events
        .iter()
        .position(|(e, _)| e == "message_delta")
        .unwrap();
    assert!(
        stop_idx < delta_idx,
        "content_block_stop must precede message_delta"
    );
}
