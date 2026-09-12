//! HTTP integration tests for `POST /v1/responses` (tower oneshot).
//!
//! All tests run against `ScriptedPromptBuilder` + `MockEngine` — no model
//! files, no GPU. Mirrors `tests/http.rs` (same `app_with` / `text_script`
//! / `post` helpers, pointed at the Responses route).

use std::sync::Arc;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use axum::Router;
use http_body_util::BodyExt;
use minicpm_runtime::{GenerationConfig, MockEngine};
use minicpm_scheduler::Scheduler;
use minicpm_server::prompt::{char_to_id, ScriptedPromptBuilder};
use minicpm_server::{build_router, AppState};
use serde_json::{json, Value};
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

async fn post(app: &Router, body: Value) -> (StatusCode, Vec<u8>) {
    let response = app
        .clone()
        .oneshot(
            Request::post("/v1/responses")
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

/// Baseline request body with `extra` merged on top.
fn responses_body(extra: Value) -> Value {
    let mut body = json!({
        "model": "minicpm5-2b",
        "input": "hello",
    });
    if let (Some(obj), Some(extra)) = (body.as_object_mut(), extra.as_object()) {
        for (k, v) in extra {
            obj.insert(k.clone(), v.clone());
        }
    }
    body
}

/// Split an SSE body into `(event, data)` pairs; panics on malformed frames.
fn sse_events(body: &[u8]) -> Vec<(String, Value)> {
    let text = std::str::from_utf8(body).expect("utf-8 SSE body");
    text.split("\n\n")
        .filter(|f| !f.is_empty())
        .map(|frame| {
            let mut lines = frame.split('\n');
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
        })
        .collect()
}

// ------------------------------------------------------------------ ① ----

#[tokio::test]
async fn non_stream_basic_response_shape() {
    let script = text_script("Hello, Responses world!");
    let app = app_with(Arc::new(MockEngine::with_script(script)));

    let (status, body) = post(&app, json!({ "model": "echoed-model", "input": "hello" })).await;
    assert_eq!(status, StatusCode::OK);

    let v: Value = serde_json::from_slice(&body).unwrap();
    assert!(
        v["id"].as_str().unwrap().starts_with("resp-"),
        "id prefix: {v}"
    );
    assert_eq!(v["object"], "response");
    assert_eq!(v["status"], "completed");
    assert_eq!(v["model"], "echoed-model"); // echoed request model
    assert_eq!(v["incomplete_details"], Value::Null);
    assert!(v["created_at"].as_u64().is_some());

    let item = &v["output"][0];
    assert_eq!(item["type"], "message");
    assert_eq!(item["role"], "assistant");
    assert_eq!(item["status"], "completed");
    assert!(
        item["id"].as_str().unwrap().starts_with("msg-"),
        "item: {item}"
    );
    assert_eq!(item["content"][0]["type"], "output_text");
    assert_eq!(item["content"][0]["text"], "Hello, Responses world!");
    assert_eq!(item["content"][0]["annotations"], json!([]));

    let usage = &v["usage"];
    assert_eq!(usage["output_tokens"], 23); // scripted tokens == chars
    assert!(usage["input_tokens"].as_u64().unwrap() > 0);
    assert_eq!(
        usage["total_tokens"],
        usage["input_tokens"].as_u64().unwrap() + 23
    );
    assert!(
        usage.get("input_tokens_details").is_some(),
        "usage: {usage}"
    );
    assert!(
        usage.get("output_tokens_details").is_some(),
        "usage: {usage}"
    );
}

// ------------------------------------------------------------------ ② ----

#[tokio::test]
async fn max_output_tokens_truncation_is_incomplete() {
    let script = text_script("twentytokenabcdefgh");
    let app = app_with(Arc::new(MockEngine::with_script(script)));

    let (status, body) = post(&app, responses_body(json!({ "max_output_tokens": 8 }))).await;
    assert_eq!(status, StatusCode::OK);

    let v: Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(v["status"], "incomplete");
    assert_eq!(v["incomplete_details"]["reason"], "max_output_tokens");
    assert_eq!(v["output"][0]["content"][0]["text"], "twentyto"); // first 8 chars
    assert_eq!(v["usage"]["output_tokens"], 8);
}

#[tokio::test]
async fn max_tokens_field_name_also_truncates() {
    // Chat-style `max_tokens` spelling is tolerated on /v1/responses.
    let script = text_script("twentytokenabcdefgh");
    let app = app_with(Arc::new(MockEngine::with_script(script)));

    let (status, body) = post(&app, responses_body(json!({ "max_tokens": 4 }))).await;
    assert_eq!(status, StatusCode::OK);
    let v: Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(v["status"], "incomplete");
    assert_eq!(v["output"][0]["content"][0]["text"], "twen");
}

// ------------------------------------------------------------------ ③ ----

#[tokio::test]
async fn non_stream_function_call_output_item() {
    let xml = r#"<function name="get_weather"><param name="city">Paris</param></function>"#;
    let app = app_with(Arc::new(MockEngine::with_script(text_script(xml))));

    let (status, body) = post(
        &app,
        responses_body(json!({
            "max_output_tokens": 200,
            "tools": [{
                "type": "function",
                "name": "get_weather",
                "description": "Get the weather",
                "parameters": {
                    "type": "object",
                    "properties": { "city": { "type": "string" } },
                    "required": ["city"],
                },
            }],
        })),
    )
    .await;
    assert_eq!(status, StatusCode::OK);

    let v: Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(v["status"], "completed");
    let item = &v["output"][0]; // no text → function_call comes first
    assert_eq!(item["type"], "function_call");
    assert!(
        item["id"].as_str().unwrap().starts_with("fc-"),
        "item: {item}"
    );
    assert!(
        item["call_id"].as_str().unwrap().starts_with("call_"),
        "item: {item}"
    );
    assert_eq!(item["name"], "get_weather");
    assert_eq!(item["arguments"], "{\"city\":\"Paris\"}"); // string form, verbatim
    assert_eq!(item["status"], "completed");
}

// ------------------------------------------------------------------ ④ ----

#[tokio::test]
async fn stream_event_sequence_and_no_done_sentinel() {
    let script = text_script("streaming responses parade!");
    let app = app_with(Arc::new(MockEngine::with_script(script)));

    let (status, body) = post(&app, responses_body(json!({ "stream": true }))).await;
    assert_eq!(status, StatusCode::OK);

    let text = String::from_utf8(body.clone()).unwrap();
    assert!(
        !text.contains("[DONE]"),
        "Responses SSE has no [DONE]: {text:?}"
    );

    let events = sse_events(&body);
    let types: Vec<&str> = events.iter().map(|(e, _)| e.as_str()).collect();
    assert_eq!(types.first(), Some(&"response.created"));
    assert_eq!(types.get(1), Some(&"response.in_progress"));
    assert_eq!(types.last(), Some(&"response.completed"));

    let pos = |name: &str| types.iter().position(|t| *t == name);
    let added = pos("response.output_item.added").expect("output_item.added");
    let part_added = pos("response.content_part.added").expect("content_part.added");
    let text_done = pos("response.output_text.done").expect("output_text.done");
    let part_done = pos("response.content_part.done").expect("content_part.done");
    let item_done = pos("response.output_item.done").expect("output_item.done");
    assert!(added < part_added, "order: {types:?}");
    assert!(part_added < text_done, "order: {types:?}");
    assert!(text_done < part_done, "order: {types:?}");
    assert!(part_done < item_done, "order: {types:?}");
    assert!(
        item_done < types.len() - 1,
        "completed must be last: {types:?}"
    );

    // Joined deltas == the scripted text.
    let deltas: String = events
        .iter()
        .filter(|(e, _)| e == "response.output_text.delta")
        .map(|(_, d)| d["delta"].as_str().unwrap().to_string())
        .collect();
    assert!(!deltas.is_empty(), "no deltas in {types:?}");
    assert_eq!(deltas, "streaming responses parade!");

    // Every event carries type == event name and a strictly increasing
    // sequence_number (one per frame, from 0).
    for (i, (_, data)) in events.iter().enumerate() {
        assert_eq!(data["type"], types[i], "event {i}: {data}");
        assert_eq!(
            data["sequence_number"].as_u64(),
            Some(i as u64),
            "event {i}: {data}"
        );
    }

    // The terminal event carries the assembled response + usage.
    let completed = events.last().unwrap().1.clone();
    assert_eq!(completed["response"]["status"], "completed");
    assert_eq!(
        completed["response"]["output"][0]["content"][0]["text"],
        "streaming responses parade!"
    );
    assert_eq!(completed["response"]["usage"]["output_tokens"], 27);
}

#[tokio::test]
async fn stream_function_call_event_sequence() {
    let xml = r#"<function name="get_weather"><param name="city">Paris</param></function>"#;
    let app = app_with(Arc::new(MockEngine::with_script(text_script(xml))));

    let (status, body) = post(
        &app,
        responses_body(json!({ "stream": true, "max_output_tokens": 200 })),
    )
    .await;
    assert_eq!(status, StatusCode::OK);

    let events = sse_events(&body);
    let types: Vec<&str> = events.iter().map(|(e, _)| e.as_str()).collect();
    assert_eq!(types.first(), Some(&"response.created"));
    assert_eq!(types.last(), Some(&"response.completed"));

    let added = events
        .iter()
        .find(|(e, _)| e == "response.output_item.added")
        .unwrap();
    assert_eq!(added.1["item"]["type"], "function_call");
    assert_eq!(added.1["item"]["call_id"], "call_0");
    assert_eq!(added.1["item"]["name"], "get_weather");
    assert_eq!(added.1["item"]["status"], "in_progress");

    let arg_deltas: String = events
        .iter()
        .filter(|(e, _)| e == "response.function_call_arguments.delta")
        .map(|(_, d)| d["delta"].as_str().unwrap().to_string())
        .collect();
    let parsed: Value = serde_json::from_str(&arg_deltas)
        .unwrap_or_else(|e| panic!("joined fragments `{arg_deltas}` not JSON: {e}"));
    assert_eq!(parsed["city"], "Paris");

    let args_done = events
        .iter()
        .find(|(e, _)| e == "response.function_call_arguments.done")
        .unwrap();
    assert_eq!(args_done.1["arguments"].as_str().unwrap(), arg_deltas);

    let completed = events.last().unwrap().1.clone();
    let fc = &completed["response"]["output"][0];
    assert_eq!(fc["type"], "function_call");
    assert_eq!(fc["call_id"], "call_0");
    assert_eq!(fc["arguments"].as_str().unwrap(), arg_deltas);
}

// ------------------------------------------------------------------ ⑤ ----

#[tokio::test]
async fn empty_input_rejected() {
    let app = app_with(Arc::new(MockEngine::new()));
    for body in [
        json!({ "model": "minicpm5-2b" }), // input missing
        json!({ "model": "minicpm5-2b", "input": "" }),
        json!({ "model": "minicpm5-2b", "input": [] }),
    ] {
        let (status, body) = post(&app, body).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        let v: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(
            v["error"]["type"], "invalid_request_error",
            "error shape: {v}"
        );
    }
}
