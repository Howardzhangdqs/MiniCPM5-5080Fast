//! MiniCPM5-2B inference server entrypoint.
//!
//! ```text
//! minicpm-server [--host 0.0.0.0] [--port 8080] [--max-sessions 4]
//!                [--queue-limit 32] [--queue-timeout-s 120]
//!                [--model-dir models/minicpm5-2b]
//!                [--native-lib build/native/libminicpm_native.so]
//!                [--model-name minicpm5-2b]
//! ```
//!
//! * `--native-lib` given: `dlopen` the CUDA runtime (`NativeEngine`) and
//!   serve real (currently mock-forward) GPU inference.
//! * omitted: serve with the deterministic `MockEngine` (no GPU needed) —
//!   useful for API-level development.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use clap::Parser;
use minicpm_metrics::init_tracing;
use minicpm_runtime::{model_options_for_precision, Engine, MockEngine, NativeEngine, Precision};
use minicpm_scheduler::{Scheduler, SchedulerConfig};
use minicpm_server::build_router;
use minicpm_server::prompt::TemplatePromptBuilder;
use minicpm_server::{AppState, PromptBuilder};

#[derive(Debug, Parser)]
#[command(
    name = "minicpm-server",
    about = "MiniCPM5-2B inference server (OpenAI-compatible API + SSE)"
)]
struct Args {
    /// HTTP bind address.
    #[arg(long, default_value = "0.0.0.0")]
    host: String,

    /// HTTP port.
    #[arg(long, default_value_t = 8080)]
    port: u16,

    /// 真并发上限（活跃 session 数）：调度器活跃集与 native session 配额
    /// 共用此值；等待请求进入排队（见 --queue-limit）。
    #[arg(long, default_value_t = 4)]
    max_sessions: usize,

    /// 等待队列容量上限（S3 准入控制）：提交时等待数 ≥ 此值直接回
    /// 429 + Retry-After；0 = 一律立即 429。
    #[arg(long, default_value_t = 32)]
    queue_limit: usize,

    /// 排队超时秒数：job 在等待队列滞留超过此时长，准入时直接 Failed
    ///（不占 GPU）。0 = 不设超时以外的特殊语义（仍按秒解析）。
    #[arg(long, default_value_t = 120)]
    queue_timeout_s: u64,

    /// Model directory (tokenizer.json, chat_template.jinja, configs).
    #[arg(long, default_value = "models/minicpm5-2b")]
    model_dir: PathBuf,

    /// Native CUDA runtime library; omit to serve with the mock engine.
    #[arg(long)]
    native_lib: Option<PathBuf>,

    /// Weight precision for the native engine: `bf16` (default) or `fp8`
    /// (loads model_fp8.wpk; P4 v1 keeps BF16 prefill weights alongside).
    #[arg(long, default_value = "bf16")]
    precision: String,

    /// Model name reported by /v1/models and in completion payloads.
    #[arg(long, default_value = "minicpm5-2b")]
    model_name: String,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    init_tracing("info");

    let engine: Arc<dyn Engine> = match &args.native_lib {
        Some(lib) => {
            let precision = Precision::parse(&args.precision).ok_or_else(|| {
                anyhow::anyhow!(
                    "unknown --precision '{}': expected 'bf16' or 'fp8'",
                    args.precision
                )
            })?;
            let mut options = model_options_for_precision(precision);
            // Keep options.max_sessions consistent with --max-sessions.
            // ora-3 R2：native 侧 session 配额与 host 侧调度器活跃集上限
            // 必须同值——两处赋值点各留一道 debug_assert 防漂移。
            options.max_sessions = args.max_sessions as u32;
            debug_assert_eq!(options.max_sessions as usize, args.max_sessions);
            tracing::info!(lib = %lib.display(), model_dir = %args.model_dir.display(), ?precision, "loading native engine");
            Arc::new(NativeEngine::load(
                lib,
                args.model_dir.to_string_lossy().as_ref(),
                &options,
            )?)
        }
        None => {
            tracing::warn!(
                "--native-lib not set: serving with MockEngine (deterministic fake tokens)"
            );
            Arc::new(MockEngine::new())
        }
    };

    let prompt = Arc::new(TemplatePromptBuilder::new(&args.model_dir)?);
    tracing::info!(
        bos = prompt.bos_token(),
        "prompt builder ready (official chat template + HF tokenizer)"
    );

    let scheduler = Scheduler::with_config(
        engine,
        SchedulerConfig {
            max_sessions: args.max_sessions,
            queue_limit: args.queue_limit,
            queue_timeout: Duration::from_secs(args.queue_timeout_s),
            ..SchedulerConfig::default()
        },
    )?;
    // ora-3 R2 的第二处：调度器侧上限与 --max-sessions 同值（与上面
    // options.max_sessions 的赋值同步）。
    debug_assert_eq!(scheduler.handle().max_sessions(), args.max_sessions);
    let state = AppState::new(
        scheduler.handle(),
        prompt,
        args.model_name.clone(),
        minicpm_runtime::GenerationConfig::default(),
    );
    let app = build_router(state);

    let addr = format!("{}:{}", args.host, args.port);
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    tracing::info!(%addr, model = %args.model_name, queue_limit = args.queue_limit, queue_timeout_s = args.queue_timeout_s, "listening (POST /v1/chat/completions, GET /v1/models /health /v1/stats)");
    axum::serve(listener, app).await?;
    Ok(())
}
