//! S3 集成测试：准入控制（queue_limit → 429 + Retry-After；queue_timeout
//! → Failed）+ /v1/stats 的 scheduler / recent_requests 两键。
//!
//! 全部走 `ScriptedPromptBuilder` + paced `MockEngine`（无模型、无 GPU）。
//! 注意：MetricsRegistry 是进程级全局，测试并行共享——凡对 gauge/明细
//! 的断言只做「存在性 + 本测试贡献的条目存在」，不做独占性断言。

use std::sync::Arc;
use std::time::Duration;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use axum::Router;
use http_body_util::BodyExt;
use minicpm_runtime::{GenerationConfig, MockEngine};
use minicpm_scheduler::{Scheduler, SchedulerConfig, SchedulerHandle};
use minicpm_server::prompt::{char_to_id, ScriptedPromptBuilder};
use minicpm_server::{build_router, AppState};
use serde_json::json;
use tower::ServiceExt;

/// 带自定义 SchedulerConfig 的 app（S3 准入参数可调）。
fn app_with_config(engine: Arc<MockEngine>, config: SchedulerConfig) -> (Router, SchedulerHandle) {
    let scheduler = Scheduler::with_config(engine, config).expect("scheduler");
    let handle = scheduler.handle();
    let state = AppState::new(
        handle.clone(),
        Arc::new(ScriptedPromptBuilder),
        "minicpm5-2b",
        GenerationConfig {
            max_new_tokens: 64,
            ..GenerationConfig::default()
        },
    );
    (build_router(state), handle)
}

fn text_script(text: &str) -> Vec<i32> {
    text.chars().map(char_to_id).collect()
}

fn chat_body(max_tokens: u32) -> serde_json::Value {
    json!({
        "model": "minicpm5-2b",
        "messages": [{ "role": "user", "content": "hello" }],
        "max_tokens": max_tokens,
    })
}

async fn post(app: &Router, body: serde_json::Value) -> (StatusCode, Option<String>, Vec<u8>) {
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
    let retry_after = response
        .headers()
        .get("Retry-After")
        .and_then(|v| v.to_str().ok())
        .map(str::to_string);
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    (status, retry_after, bytes.to_vec())
}

async fn get_stats(app: &Router) -> serde_json::Value {
    let response = app
        .clone()
        .oneshot(Request::get("/v1/stats").body(Body::empty()).unwrap())
        .await
        .unwrap();
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    serde_json::from_slice(&bytes).unwrap()
}

/// 等首个请求真正进入 decode（queue_depth 归零 + engine 已至少 decode
/// 一步——单纯等 queue_depth==0 有 TOCTOU：提交前它本来就是 0）。
async fn wait_decoding(handle: &SchedulerHandle, engine: &MockEngine) {
    for _ in 0..400 {
        if handle.queue_depth() == 0 && engine.stats().decode_calls >= 1 {
            return;
        }
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    panic!("job never started decoding");
}

// ------------------------------------------------------------------ ① ----

/// queue_limit=1：单活跃流（max_sessions=1）+ 1 个排队 → 下一个并发请求
/// 收 429 + Retry-After: 1；被拒请求不占位；排队的照常完成。
#[tokio::test]
async fn queue_full_returns_429_with_retry_after() {
    // 10ms/token：r1（60 token ≈ 600ms）长跑，稳住「1 活跃 + 1 排队」。
    let engine = Arc::new(
        MockEngine::with_script(text_script(&"x".repeat(200)))
            .with_decode_delay(Duration::from_millis(10)),
    );
    let (app, handle) = app_with_config(
        engine.clone(),
        SchedulerConfig {
            max_sessions: 1,
            queue_limit: 1,
            ..SchedulerConfig::default()
        },
    );

    // r1：占唯一活跃 slot。
    let app1 = app.clone();
    let r1 = tokio::spawn(async move { post(&app1, chat_body(60)).await });
    wait_decoding(&handle, &engine).await;

    // r2：占唯一等待位（active 满 → 留在 channel）。
    let app2 = app.clone();
    let r2 = tokio::spawn(async move { post(&app2, chat_body(3)).await });
    // r2 已入队（waiting=1，有限轮询而非裸 sleep）。
    for _ in 0..100 {
        if handle.queue_depth() == 1 {
            break;
        }
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    assert_eq!(
        handle.queue_depth(),
        1,
        "r2 must hold the single queue slot"
    );

    // r3：queue_limit=1 已满 → 429 + Retry-After: 1。
    let (status, retry_after, body) = post(&app, chat_body(3)).await;
    assert_eq!(
        status,
        StatusCode::TOO_MANY_REQUESTS,
        "body: {}",
        String::from_utf8_lossy(&body)
    );
    assert_eq!(retry_after.as_deref(), Some("1"), "Retry-After: 1 required");
    let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(v["error"]["type"], "rate_limit_error");
    assert!(
        v["error"]["message"]
            .as_str()
            .unwrap()
            .contains("queue full"),
        "body: {v}"
    );

    // r1 / r2 照常完成（被拒请求不影响在途请求）。
    let (s1, _, b1) = r1.await.unwrap();
    assert_eq!(s1, StatusCode::OK, "{}", String::from_utf8_lossy(&b1));
    let v1: serde_json::Value = serde_json::from_slice(&b1).unwrap();
    assert_eq!(v1["usage"]["completion_tokens"], 60);
    let (s2, _, b2) = r2.await.unwrap();
    assert_eq!(s2, StatusCode::OK, "{}", String::from_utf8_lossy(&b2));
    let v2: serde_json::Value = serde_json::from_slice(&b2).unwrap();
    assert_eq!(v2["usage"]["completion_tokens"], 3);
}

// ------------------------------------------------------------------ ② ----

/// queue_timeout：排队滞留超过时长的请求在准入时直接 Failed（不建
/// session、不 prefill），客户端收到失败响应。
#[tokio::test]
async fn queue_timeout_fails_stale_request() {
    // r1 60 token × 10ms ≈ 600ms ≫ timeout 100ms → r2 必然过期。
    let engine = Arc::new(
        MockEngine::with_script(text_script(&"y".repeat(200)))
            .with_decode_delay(Duration::from_millis(10)),
    );
    let (app, handle) = app_with_config(
        engine.clone(),
        SchedulerConfig {
            max_sessions: 1,
            queue_timeout: Duration::from_millis(100),
            ..SchedulerConfig::default()
        },
    );

    let app1 = app.clone();
    let r1 = tokio::spawn(async move { post(&app1, chat_body(60)).await });
    wait_decoding(&handle, &engine).await;

    // r2 排队；slot 释放时已滞留 ~600ms > 100ms → Failed("queue timeout")。
    let (s2, _, b2) = post(&app, chat_body(3)).await;
    assert_eq!(
        s2,
        StatusCode::INTERNAL_SERVER_ERROR,
        "stale queued request must fail: {}",
        String::from_utf8_lossy(&b2)
    );
    let v2: serde_json::Value = serde_json::from_slice(&b2).unwrap();
    assert!(
        v2["error"]["message"]
            .as_str()
            .unwrap()
            .contains("queue timeout"),
        "body: {v2}"
    );

    // r1 不受影响。
    let (s1, _, b1) = r1.await.unwrap();
    assert_eq!(s1, StatusCode::OK, "{}", String::from_utf8_lossy(&b1));
}

// ------------------------------------------------------------------ ③ ----

/// /v1/stats：新增 `scheduler`（queue_depth / active_sessions /
/// max_sessions）与 `recent_requests`（worker 侧 TTFT/TPOT 明细）两键，
/// 现有字段不动。
#[tokio::test]
async fn stats_has_scheduler_and_recent_requests() {
    let engine = Arc::new(MockEngine::with_script(text_script(&"z".repeat(64))));
    let (app, _handle) = app_with_config(
        engine,
        SchedulerConfig {
            max_sessions: 2,
            ..SchedulerConfig::default()
        },
    );

    // 一个已知请求：7 个 token、Length 结束（7 是本测试独有指纹，免疫
    // 全局 metrics 的并行测试污染）。
    let (status, _, body) = post(&app, chat_body(7)).await;
    assert_eq!(status, StatusCode::OK, "{}", String::from_utf8_lossy(&body));

    let v = get_stats(&app).await;
    // 现有字段不动。
    for key in [
        "requests_total",
        "errors_total",
        "active_streams",
        "ttft_us",
        "tpot_us",
    ] {
        assert!(v.get(key).is_some(), "existing field {key} must stay: {v}");
    }
    // scheduler 三键。
    let sched = &v["scheduler"];
    assert_eq!(sched["max_sessions"], 2);
    assert!(
        sched["queue_depth"].is_i64(),
        "queue_depth must be numeric: {sched}"
    );
    assert!(
        sched["active_sessions"].is_i64(),
        "active_sessions must be numeric: {sched}"
    );
    // recent_requests：数组 + 本请求的明细（6 键齐全、口径正确）。
    let recent = v["recent_requests"]
        .as_array()
        .expect("recent_requests array");
    let mine = recent
        .iter()
        .find(|r| r["completion_tokens"] == 7 && r["finish_reason"] == "length")
        .unwrap_or_else(|| panic!("our 7-token request must appear: {recent:?}"));
    for key in [
        "ttft_us",
        "tpot_us",
        "prompt_tokens",
        "completion_tokens",
        "finish_reason",
        "ts_ms",
    ] {
        assert!(
            mine.get(key).is_some(),
            "recent entry missing {key}: {mine}"
        );
    }
    assert!(
        mine["ttft_us"].as_u64().unwrap() > 0,
        "worker-side TTFT incl. queue wait: {mine}"
    );
    assert!(mine["prompt_tokens"].as_u64().unwrap() > 0);
}
