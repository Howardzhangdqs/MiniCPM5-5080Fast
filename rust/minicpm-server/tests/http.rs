//! HTTP integration tests (tower oneshot + one real-TCP cancellation test).
//!
//! All tests run against `ScriptedPromptBuilder` + `MockEngine` — no model
//! files, no GPU.

use std::sync::Arc;
use std::time::Duration;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use axum::Router;
use http_body_util::BodyExt;
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

async fn post(app: &Router, body: serde_json::Value) -> (StatusCode, Vec<u8>) {
    let response = app
        .clone()
        .oneshot(
            Request::post("/v1/chat/completions")
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

async fn get(app: &Router, path: &str) -> (StatusCode, serde_json::Value) {
    let response = app
        .clone()
        .oneshot(Request::get(path).body(Body::empty()).unwrap())
        .await
        .unwrap();
    let status = response.status();
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    (
        status,
        serde_json::from_slice(&bytes).unwrap_or(serde_json::Value::Null),
    )
}

fn chat_body(extra: serde_json::Value) -> serde_json::Value {
    let mut body = json!({
        "model": "minicpm5-2b",
        "messages": [{ "role": "user", "content": "hello" }],
    });
    if let (Some(obj), Some(extra)) = (body.as_object_mut(), extra.as_object()) {
        for (k, v) in extra {
            obj.insert(k.clone(), v.clone());
        }
    }
    body
}

// ------------------------------------------------------------------ ① ----

#[tokio::test]
async fn non_stream_max_tokens_returns_length_and_usage() {
    let script = text_script("Hello, scripted world!");
    let app = app_with(Arc::new(MockEngine::with_script(script)));

    let (status, body) = post(&app, chat_body(json!({ "max_tokens": 8 }))).await;
    assert_eq!(status, StatusCode::OK);

    let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(v["object"], "chat.completion");
    assert_eq!(v["model"], "minicpm5-2b");
    let choice = &v["choices"][0];
    assert_eq!(choice["finish_reason"], "length");
    assert_eq!(choice["message"]["role"], "assistant");
    let content = choice["message"]["content"].as_str().unwrap();
    assert!(!content.is_empty(), "content must be non-empty: {v}");
    assert_eq!(
        content, "Hello, s",
        "8 scripted tokens = 8 chars: {content:?}"
    );
    assert_eq!(v["usage"]["completion_tokens"], 8);
    assert!(v["usage"]["prompt_tokens"].as_u64().unwrap() > 0);
    assert_eq!(
        v["usage"]["total_tokens"],
        v["usage"]["prompt_tokens"].as_u64().unwrap() + 8
    );
}

// ------------------------------------------------------------------ ② ----

#[tokio::test]
async fn stream_emits_deltas_then_finish_then_done() {
    let script = text_script("streaming chunk parade!");
    let app = app_with(Arc::new(MockEngine::with_script(script)));

    let (status, body) = post(&app, chat_body(json!({ "stream": true, "max_tokens": 10 }))).await;
    assert_eq!(status, StatusCode::OK);

    let text = String::from_utf8(body).unwrap();
    // SSE frames: "data: {...}\n\n", terminated by "data: [DONE]\n\n".
    let events: Vec<&str> = text
        .split("\n\n")
        .filter(|s| !s.is_empty())
        .map(|s| s.strip_prefix("data: ").unwrap_or(s))
        .collect();
    assert!(events.len() >= 3, "need role + deltas + final: {events:?}");
    assert_eq!(events.last().unwrap(), &"[DONE]");

    // Second-to-last frame carries finish_reason + usage.
    let final_chunk: serde_json::Value = serde_json::from_str(events[events.len() - 2]).unwrap();
    assert_eq!(final_chunk["object"], "chat.completion.chunk");
    assert_eq!(final_chunk["choices"][0]["finish_reason"], "length");
    assert_eq!(final_chunk["usage"]["completion_tokens"], 10);

    // First frame announces the assistant role; at least one content delta.
    let first: serde_json::Value = serde_json::from_str(events[0]).unwrap();
    assert_eq!(first["choices"][0]["delta"]["role"], "assistant");
    let content_deltas: Vec<String> = events[1..events.len() - 2]
        .iter()
        .filter_map(|e| serde_json::from_str::<serde_json::Value>(e).ok())
        .filter_map(|c| {
            c["choices"][0]["delta"]["content"]
                .as_str()
                .map(str::to_owned)
        })
        .collect();
    assert!(
        !content_deltas.is_empty(),
        "no content deltas in {events:?}"
    );
    let joined: String = content_deltas.concat();
    assert!(joined.starts_with("streaming"), "joined deltas: {joined:?}");
}

// ------------------------------------------------------------------ ③ ----

#[tokio::test]
async fn stop_string_truncates_content() {
    let script = text_script("Hello STOP world more");
    let app = app_with(Arc::new(MockEngine::with_script(script)));

    let (status, body) = post(&app, chat_body(json!({ "max_tokens": 50, "stop": "STOP" }))).await;
    assert_eq!(status, StatusCode::OK);

    let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
    let choice = &v["choices"][0];
    assert_eq!(choice["finish_reason"], "stop");
    assert_eq!(
        choice["message"]["content"], "Hello ",
        "stop string must be withheld"
    );
    // Tokens decoded until the match completed: H e l l o ' ' S T O P = 10.
    assert_eq!(v["usage"]["completion_tokens"], 10);
}

#[tokio::test]
async fn stop_string_array_form_also_works() {
    let script = text_script("alpha<beta>gamma");
    let app = app_with(Arc::new(MockEngine::with_script(script)));

    let (status, body) = post(
        &app,
        chat_body(json!({ "max_tokens": 50, "stop": ["<beta>", "ZZZ"] })),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(v["choices"][0]["finish_reason"], "stop");
    assert_eq!(v["choices"][0]["message"]["content"], "alpha");
}

// ------------------------------------------------------------------ ④ ----

#[tokio::test]
async fn models_health_stats_endpoints() {
    let app = app_with(Arc::new(MockEngine::with_script(text_script("x"))));

    let (status, v) = get(&app, "/v1/models").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(v["object"], "list");
    assert_eq!(v["data"][0]["id"], "minicpm5-2b");
    assert_eq!(v["data"][0]["object"], "model");

    let (status, v) = get(&app, "/health").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(v["status"], "ok");

    let (status, v) = get(&app, "/v1/stats").await;
    assert_eq!(status, StatusCode::OK);
    assert!(v.get("requests_total").is_some(), "stats snapshot: {v}");
    assert!(v.get("ttft_us").is_some());
}

#[tokio::test]
async fn empty_messages_rejected() {
    let app = app_with(Arc::new(MockEngine::new()));
    let (status, _) = post(&app, json!({ "model": "minicpm5-2b", "messages": [] })).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
}

// ------------------------------------------------------------------ ⑤ ----

/// Tool-call end-to-end: the model emits MiniCPM5's XML `<function>` form;
/// the server must turn it into OpenAI `tool_calls` (non-stream) and into
/// streaming `tool_calls` argument fragments (stream).
#[tokio::test]
async fn tool_calls_xml_becomes_openai_tool_calls() {
    let xml = r#"<function name="get_weather"><param name="city">Paris</param></function>"#;
    let app = app_with(Arc::new(MockEngine::with_script(text_script(xml))));

    // Non-streaming.
    let (status, body) = post(&app, chat_body(json!({ "max_tokens": 200 }))).await;
    assert_eq!(status, StatusCode::OK);
    let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
    let calls = &v["choices"][0]["message"]["tool_calls"];
    assert_eq!(calls[0]["id"], "call_0");
    assert_eq!(calls[0]["type"], "function");
    assert_eq!(calls[0]["function"]["name"], "get_weather");
    let args: serde_json::Value =
        serde_json::from_str(calls[0]["function"]["arguments"].as_str().unwrap()).unwrap();
    assert_eq!(args["city"], "Paris");

    // Streaming: tool_calls deltas whose joined arguments form the same JSON.
    let (status, body) = post(
        &app,
        chat_body(json!({ "stream": true, "max_tokens": 200 })),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let text = String::from_utf8(body).unwrap();
    let mut tool_start: Option<serde_json::Value> = None;
    let mut arg_fragments: Vec<String> = Vec::new();
    for ev in text.split("\n\n").filter(|s| !s.is_empty()) {
        let payload = ev.strip_prefix("data: ").unwrap_or(ev);
        if payload == "[DONE]" {
            continue;
        }
        let c: serde_json::Value = match serde_json::from_str(payload) {
            Ok(v) => v,
            Err(_) => continue, // e.g. error frames
        };
        if let Some(tc) = c["choices"][0]["delta"]["tool_calls"][0].as_object() {
            if tc.contains_key("id") {
                tool_start = Some(c["choices"][0]["delta"]["tool_calls"][0].clone());
            }
            if let Some(a) = tc.get("function").and_then(|f| f["arguments"].as_str()) {
                arg_fragments.push(a.to_string());
            }
        }
    }
    let start = tool_start.expect("a tool_call start delta with id+name");
    assert_eq!(start["id"], "call_0");
    assert_eq!(start["function"]["name"], "get_weather");
    let joined: String = arg_fragments.concat();
    let streamed_args: serde_json::Value = serde_json::from_str(&joined)
        .unwrap_or_else(|e| panic!("joined fragments `{joined}` not JSON: {e}"));
    assert_eq!(streamed_args["city"], "Paris");
}

// ------------------------------------------------------------------ ⑦ ----

// Finish-semantics regressions (2026-09 crosscheck fix, OpenAI/vLLM-aligned):
// `max_tokens` caps **content** tokens; reaching the cap ends with
// finish="length"; a natural eos below the cap ends with finish="stop" and
// is neither counted in usage.completion_tokens nor streamed.

#[tokio::test]
async fn max_tokens_boundary_streams_exactly_n_content_tokens() {
    // 20-char script, max_tokens=8 → exactly 8 content chars streamed, then
    // finish="length" with completion_tokens == 8 (the old defect streamed
    // N-1 content tokens and mislabeled the truncation as "stop" when driven
    // by the native runtime).
    let script = text_script("twentytokenabcdefgh");
    let app = app_with(Arc::new(MockEngine::with_script(script)));

    let (status, body) = post(&app, chat_body(json!({ "stream": true, "max_tokens": 8 }))).await;
    assert_eq!(status, StatusCode::OK);

    let text = String::from_utf8(body).unwrap();
    let events: Vec<&str> = text
        .split("\n\n")
        .filter(|s| !s.is_empty())
        .map(|s| s.strip_prefix("data: ").unwrap_or(s))
        .collect();
    assert_eq!(events.last().unwrap(), &"[DONE]");

    let final_chunk: serde_json::Value = serde_json::from_str(events[events.len() - 2]).unwrap();
    assert_eq!(final_chunk["choices"][0]["finish_reason"], "length");
    assert_eq!(final_chunk["usage"]["completion_tokens"], 8);

    let content_deltas: Vec<String> = events[1..events.len() - 2]
        .iter()
        .filter_map(|e| serde_json::from_str::<serde_json::Value>(e).ok())
        .filter_map(|c| {
            c["choices"][0]["delta"]["content"]
                .as_str()
                .map(str::to_owned)
        })
        .collect();
    let joined: String = content_deltas.concat();
    assert_eq!(joined, "twentyto", "exactly the first 8 chars: {joined:?}");
}

#[tokio::test]
async fn natural_eos_stops_uncounted_and_unstreamed() {
    // Script "done" followed by the eos id: finish="stop", content "done",
    // completion_tokens == 4 (eos not counted), and the eos never leaks into
    // the stream (no "<|im_end|>" text anywhere).
    use minicpm_runtime::mock::MOCK_EOS;

    let mut script = text_script("done");
    script.push(MOCK_EOS);
    let app = app_with(Arc::new(MockEngine::with_script(script)));

    // Non-streaming.
    let (status, body) = post(&app, chat_body(json!({ "max_tokens": 64 }))).await;
    assert_eq!(status, StatusCode::OK);
    let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
    let choice = &v["choices"][0];
    assert_eq!(choice["finish_reason"], "stop");
    assert_eq!(choice["message"]["content"], "done");
    assert_eq!(v["usage"]["completion_tokens"], 4);

    // Streaming: joined deltas == "done", no eos text in any frame.
    let (status, body) = post(&app, chat_body(json!({ "stream": true, "max_tokens": 64 }))).await;
    assert_eq!(status, StatusCode::OK);
    let text = String::from_utf8(body).unwrap();
    assert!(
        !text.contains("<|im_end|>"),
        "eos must not be streamed: {text:?}"
    );
    let events: Vec<&str> = text
        .split("\n\n")
        .filter(|s| !s.is_empty())
        .map(|s| s.strip_prefix("data: ").unwrap_or(s))
        .collect();
    let final_chunk: serde_json::Value = serde_json::from_str(events[events.len() - 2]).unwrap();
    assert_eq!(final_chunk["choices"][0]["finish_reason"], "stop");
    assert_eq!(final_chunk["usage"]["completion_tokens"], 4);
    let content_deltas: Vec<String> = events[1..events.len() - 2]
        .iter()
        .filter_map(|e| serde_json::from_str::<serde_json::Value>(e).ok())
        .filter_map(|c| {
            c["choices"][0]["delta"]["content"]
                .as_str()
                .map(str::to_owned)
        })
        .collect();
    assert_eq!(content_deltas.concat(), "done");
}

// ------------------------------------------------------------------ ⑧ ----

/// Real-TCP cancellation: a streaming client that disconnects after the
/// first `data: ` frame must drive `cancelled_total` ≥ 1 within 5s.
#[tokio::test]
async fn real_tcp_client_disconnect_cancels_generation() {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    // Paced mock: 10ms/decode, so a 64-token default run lasts ~640ms and
    // the disconnect at the first frame (~10ms) really is mid-generation.
    let engine = Arc::new(MockEngine::new().with_decode_delay(Duration::from_millis(10)));
    let app = app_with(engine);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });

    // Open a raw HTTP/1.1 streaming request.
    let body =
        r#"{"model":"minicpm5-2b","messages":[{"role":"user","content":"hi"}],"stream":true}"#;
    let request = format!(
        "POST /v1/chat/completions HTTP/1.1\r\nHost: {addr}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    let mut sock = tokio::net::TcpStream::connect(addr).await.unwrap();
    sock.write_all(request.as_bytes()).await.unwrap();

    // Read until the first SSE data frame arrives, then drop the connection.
    let mut buf = [0u8; 4096];
    let mut seen = Vec::new();
    loop {
        let n = tokio::time::timeout(Duration::from_secs(5), sock.read(&mut buf)).await;
        let n = match n {
            Ok(Ok(0)) | Err(_) => break,
            Ok(Ok(n)) => n,
            Ok(Err(e)) => panic!("read error: {e}"),
        };
        seen.extend_from_slice(&buf[..n]);
        if seen.windows(6).any(|w| w == b"data: ") {
            break;
        }
    }
    assert!(
        seen.windows(6).any(|w| w == b"data: "),
        "no SSE frame received: {:?}",
        String::from_utf8_lossy(&seen)
    );
    drop(sock); // ← the disconnect under test

    // Poll /v1/stats (20 × 250ms = 5s) until cancelled_total ≥ 1.
    for attempt in 0..20 {
        let mut s = tokio::net::TcpStream::connect(addr).await.unwrap();
        let get = format!("GET /v1/stats HTTP/1.1\r\nHost: {addr}\r\nConnection: close\r\n\r\n");
        s.write_all(get.as_bytes()).await.unwrap();
        let mut resp = Vec::new();
        let mut rb = [0u8; 4096];
        loop {
            let n = s.read(&mut rb).await.unwrap_or(0);
            if n == 0 {
                break;
            }
            resp.extend_from_slice(&rb[..n]);
        }
        let text = String::from_utf8_lossy(&resp);
        if let Some(pos) = text.find("\"cancelled_total\"") {
            let rest = &text[pos..];
            let value = rest
                .split(':')
                .nth(1)
                .and_then(|s| s.split(',').next())
                .and_then(|s| s.trim().parse::<u64>().ok())
                .unwrap_or(0);
            if value >= 1 {
                return; // pass
            }
        }
        if attempt == 19 {
            panic!("cancelled_total never reached 1 within 5s: {text}");
        }
        tokio::time::sleep(Duration::from_millis(250)).await;
    }
}
