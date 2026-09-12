//! B3 真机 e2e（ignored）：K ∈ {2, 8} 条 greedy 流经调度器**批路径**与
//! 全 solo 参考的 greedy 前缀一致率 + 聚合吞吐。
//!
//! 默认 `#[ignore]`：需要 cmake 构建的 native 库 + model.wpk + **空闲 GPU**。
//! ```text
//! cmake -S native -B build/native && cmake --build build/native
//! cargo test -p minicpm-scheduler --test batch_e2e -- --ignored --nocapture
//! ```
//!
//! 方法论（native `tests/batch_test.cpp` 场景 A 的 host 调度器镜像）：
//! * 参考：每条流一个独立 session，直接 prefill + 64 次 `decode_one`
//!   （全 solo、串行）——与 native 侧对拍基线同构；
//! * 被测：同一引擎经 `Scheduler`（max_sessions=8、`batching: true`）并发
//!   提交 K 条流——worker 每 quantum 分拣 greedy 流合批，`mc_batch_step`
//!   一次推进全员（B1 eager 精确 n 步）；
//! * 断言：K×64 位置上前缀一致率 ≥99%（bf16 两条数值路径——solo gemv 与
//!   批 cuBLASLt——的固有 1-2 ULP 舍入差经 42 层放大可产生罕见翻转，
//!   native 侧同一门限）；K=8 额外断言聚合吞吐 ≥432 tok/s（B3 GO 门槛；
//!   native 8 槽 eager 实测 996 tok/s，e2e 计入调度/事件开销预期 600+）。
//!
//! 模型口径：ctx=512 档（native 批测试 A 档）、graph off——批路径 B1 恒
//! eager；solo 参考同口径（同精度 bf16 对拍）。

use std::sync::Arc;
use std::time::Instant;

use minicpm_runtime::native::{default_model_options, NativeEngine};
use minicpm_runtime::{Engine, GenerationConfig};
use minicpm_scheduler::{GenEvent, GenerationJob, Scheduler, SchedulerConfig};

const NATIVE_LIB: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../build/native/libminicpm_native.so"
);
const MODEL_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../models/minicpm5-2b");

const STEPS: u32 = 64;
const PROMPT_LEN: usize = 256;

fn greedy(max_new: u32) -> GenerationConfig {
    GenerationConfig {
        max_new_tokens: max_new,
        temperature: 0.0, // greedy → 可入批（B1 greedy-only）
        top_p: 1.0,
        top_k: 0,
        repetition_penalty: 1.0,
        seed: 1,
    }
}

/// 与 native batch_test 同式的确定性 prompt（slot 区分）。
fn make_prompt(n: usize, salt: u32) -> Vec<i32> {
    (0..n)
        .map(|i| {
            (((i as u32)
                .wrapping_mul(2654435761)
                .wrapping_add(7 + salt.wrapping_mul(0x9E37_79B9)))
                % 130_560) as i32
        })
        .collect()
}

fn tokens_of(events: &[GenEvent]) -> Vec<i32> {
    events
        .iter()
        .filter_map(|e| match e {
            GenEvent::Token(t) => Some(*t),
            _ => None,
        })
        .collect()
}

fn engine_ready() -> Option<Arc<NativeEngine>> {
    if !std::path::Path::new(NATIVE_LIB).exists() {
        eprintln!("skip: {NATIVE_LIB} not found (build the native runtime first)");
        return None;
    }
    if !std::path::Path::new(MODEL_DIR).join("model.wpk").exists() {
        eprintln!("skip: {MODEL_DIR}/model.wpk not found");
        return None;
    }
    let mut options = default_model_options();
    options.max_sessions = 16;
    options.max_context_tokens = 512; // native 批测试 A 档（批组 ctx 档一致）
    options.enable_cuda_graph = 0; // 批路径 eager；solo 同口径对拍
    Some(Arc::new(
        NativeEngine::load(NATIVE_LIB, MODEL_DIR, &options).expect("native engine load"),
    ))
}

/// 全 solo 参考：K 条流各用独立 session 串行 prefill + STEPS 次 decode_one。
fn solo_reference(engine: &NativeEngine, k: usize) -> Vec<Vec<i32>> {
    let cfg = greedy(100_000); // 关闭收尾哨兵，步数由本函数控制
    let mut refs = Vec::with_capacity(k);
    for i in 0..k {
        let mut s = engine.create_session().expect("reference session");
        s.prefill(&make_prompt(PROMPT_LEN, i as u32))
            .expect("prefill");
        let mut toks = Vec::with_capacity(STEPS as usize);
        for _ in 0..STEPS {
            toks.push(s.decode_one(&cfg).expect("solo decode"));
        }
        refs.push(toks);
        s.reset().expect("reset");
        drop(s);
    }
    refs
}

/// 调度器并发段：K 条流同时在外，收集全部事件 + 墙钟吞吐。
async fn scheduler_run(engine: Arc<NativeEngine>, k: usize) -> (Vec<Vec<i32>>, f64, u64) {
    let sched = Scheduler::with_config(
        engine,
        SchedulerConfig {
            max_sessions: 8,
            batching: true, // B3 批路径（MC_BATCH_OFF 不设时默认开）
            ..SchedulerConfig::default()
        },
    )
    .expect("scheduler");
    let handle = sched.handle();

    let t0 = Instant::now();
    let mut rxs = Vec::new();
    for i in 0..k {
        let (rx, _) = handle.generate(GenerationJob {
            prompt_ids: make_prompt(PROMPT_LEN, i as u32),
            config: greedy(STEPS),
        });
        rxs.push(rx);
    }
    let mut tasks = Vec::new();
    for mut rx in rxs {
        tasks.push(tokio::spawn(async move {
            let mut out = Vec::new();
            while let Some(ev) = rx.recv().await {
                out.push(ev);
            }
            out
        }));
    }
    let mut streams = Vec::new();
    for t in tasks {
        streams.push(t.await.unwrap());
    }
    let wall = t0.elapsed().as_secs_f64();

    let mut token_total = 0u64;
    let token_streams: Vec<Vec<i32>> = streams
        .iter()
        .map(|s| {
            let toks = tokens_of(s);
            token_total += toks.len() as u64;
            // 终态必须是 Done（Length 为主；eos 早停也合法——前缀对拍覆盖）。
            assert!(
                matches!(s.last(), Some(GenEvent::Done { .. })),
                "stream must finish with Done: {:?}",
                s.last()
            );
            toks
        })
        .collect();
    let tps = token_total as f64 / wall;
    (token_streams, tps, token_total)
}

/// 前缀一致率（K×min(len) 位置；native 场景 A 同口径）。
fn prefix_match_rate(got: &[Vec<i32>], refs: &[Vec<i32>]) -> (f64, usize, usize) {
    let mut total = 0usize;
    let mut matched = 0usize;
    for (g, r) in got.iter().zip(refs) {
        let n = g.len().min(r.len());
        for j in 0..n {
            total += 1;
            if g[j] == r[j] {
                matched += 1;
            }
        }
    }
    (100.0 * matched as f64 / total.max(1) as f64, matched, total)
}

async fn run_k(k: usize, throughput_gate: Option<f64>) {
    let Some(engine) = engine_ready() else { return };

    // 先 solo 参考（串行、每流独立 session）。
    let refs = solo_reference(&engine, k);
    eprintln!(
        "e2e K={k}: solo reference collected ({} streams × {STEPS})",
        refs.len()
    );

    // 被测：调度器批路径（K ≤ 8 全员同批；引擎支持 BatchEngine）。
    let (got, tps, total) = scheduler_run(engine, k).await;

    let (rate, matched, cnt) = prefix_match_rate(&got, &refs);
    eprintln!(
        "e2e K={k}: prefix match {matched}/{cnt} = {rate:.2}% (gate ≥99%) | \
         aggregate {tps:.1} tok/s over {total} tokens (wall incl. prefill)"
    );
    for (i, (g, r)) in got.iter().zip(&refs).enumerate() {
        if let Some(pos) = g.iter().zip(r).position(|(a, b)| a != b) {
            eprintln!(
                "e2e K={k}: stream {i} first diff at {pos}: batch={:?} solo={:?}",
                &g[pos.saturating_sub(2)..(pos + 2).min(g.len())],
                &r[pos.saturating_sub(2)..(pos + 2).min(r.len())],
            );
        }
    }
    assert!(
        rate >= 99.0,
        "prefix match rate {rate:.2}% < 99% (K={k}, {matched}/{cnt})"
    );
    if let Some(gate) = throughput_gate {
        assert!(
            tps >= gate,
            "aggregate throughput {tps:.1} tok/s < gate {gate} tok/s (K={k})"
        );
    }
}

/// K=2：双流 e2e 前缀一致率（吞吐信息性输出，不设门）。
#[tokio::test]
#[ignore = "requires libminicpm_native.so + model.wpk + an idle GPU; run with --ignored"]
async fn batch_e2e_k2_prefix_match() {
    run_k(2, None).await;
}

/// K=8：满槽 e2e 前缀一致率 + 聚合吞吐 GO 门槛（≥432 tok/s；native 8 槽
/// eager 实测 996，e2e 预期 600+）。
#[tokio::test]
#[ignore = "requires libminicpm_native.so + model.wpk + an idle GPU; run with --ignored"]
async fn batch_e2e_k8_prefix_match_and_throughput() {
    run_k(8, Some(432.0)).await;
}
