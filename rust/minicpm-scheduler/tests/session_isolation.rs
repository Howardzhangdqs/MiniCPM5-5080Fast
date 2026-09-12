//! 双 session 交错真机集成测试（P7 生产化加固；native soak_test T2 的
//! host 侧镜像）。
//!
//! 默认 `#[ignore]`：需要 cmake 构建的 native 库 + model.wpk + **空闲 GPU**。
//! ```text
//! cmake -S native -B build/native && cmake --build build/native
//! cargo test -p minicpm-scheduler --test session_isolation -- --ignored --nocapture
//! ```
//!
//! 覆盖两部分：
//! 1. **双 session 直接交错**（minicpm-runtime FFI 层）：两个 native
//!    session 各自 prefill 同一固定 prompt 后交错 decode（A 10 步→B 10 步
//!    × 5 轮），逐 token 与「单 session 顺序跑同样序列」一致 —— native 侧
//!    soak_test T2 核心断言（KV/状态无串扰）的 Rust 镜像。
//! 2. **scheduler 池复用**：同一 Scheduler 交替提交 A/B 两类 job（greedy
//!    确定性），每个 job 的 token 流与首个同类 job 完全一致 —— 验证池的
//!    reset-on-take 隔离（会话复用不泄漏上一 job 的序列状态）。
//!
//! 设计观察（S2 活跃集调度起更新）：scheduler 的 GPU worker 为单线程
//! 「活跃集 + quantum 交错」——多个 job 并发推进（每量子轮转各走一步
//! decode），池在稳态下可持有至多 max_sessions 个 session；本测试逐个
//! await 提交，单请求在外的调用序列与旧串行 FIFO 逐调用一致，断言不变。

use std::sync::Arc;

use minicpm_generation::FinishReason;
use minicpm_runtime::native::{default_model_options, NativeEngine};
use minicpm_runtime::{Engine, GenerationConfig};
use minicpm_scheduler::{GenEvent, GenerationJob, Scheduler};

const NATIVE_LIB: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../build/native/libminicpm_native.so"
);
const MODEL_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../models/minicpm5-2b");

fn greedy(max_new: u32) -> GenerationConfig {
    GenerationConfig {
        max_new_tokens: max_new,
        temperature: 0.0,
        top_p: 1.0,
        top_k: 0,
        repetition_penalty: 1.0,
        seed: 1,
    }
}

/// 与 native soak_test 同式的确定性 prompt（安全 vocab 范围）。
fn make_prompt(n: usize, salt: u32) -> Vec<i32> {
    (0..n)
        .map(|i| (((i as u32).wrapping_mul(2654435761).wrapping_add(7 + salt)) % 130560) as i32)
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

async fn run_job(handle: &minicpm_scheduler::SchedulerHandle, p: Vec<i32>) -> Vec<GenEvent> {
    let (mut rx, _cancel) = handle.generate(GenerationJob {
        prompt_ids: p,
        config: greedy(12),
    });
    let mut out = Vec::new();
    while let Some(ev) = rx.recv().await {
        out.push(ev);
    }
    out
}

#[tokio::test]
#[ignore = "requires libminicpm_native.so + model.wpk + an idle GPU; run with --ignored"]
async fn two_session_interleave_matches_sequential() {
    if !std::path::Path::new(NATIVE_LIB).exists() {
        eprintln!("skip: {NATIVE_LIB} not found (build the native runtime first)");
        return;
    }
    if !std::path::Path::new(MODEL_DIR).join("model.wpk").exists() {
        eprintln!("skip: {MODEL_DIR}/model.wpk not found");
        return;
    }

    let mut options = default_model_options();
    options.max_sessions = 2;
    options.max_context_tokens = 2048;
    options.enable_cuda_graph = 1;
    let engine =
        Arc::new(NativeEngine::load(NATIVE_LIB, MODEL_DIR, &options).expect("native engine load"));

    // ---- 部分 1：双 session 直接交错（FFI 层镜像 native T2）----
    {
        let prompt = make_prompt(256, 3);
        let cfg = greedy(1000);

        // 参照：单 session 顺序 prefill + 50 decode（跑两遍验证自一致；
        // 50 = 交错侧 5 轮 × 10 步）。
        let mut reference = engine.create_session().expect("reference session");
        reference.prefill(&prompt).expect("reference prefill");
        let ref_tokens: Vec<i32> = (0..50)
            .map(|_| reference.decode_one(&cfg).unwrap())
            .collect();
        reference.reset().expect("reference reset");
        reference.prefill(&prompt).expect("reference re-prefill");
        let ref_tokens2: Vec<i32> = (0..50)
            .map(|_| reference.decode_one(&cfg).unwrap())
            .collect();
        match ref_tokens
            .iter()
            .zip(&ref_tokens2)
            .position(|(x, y)| x != y)
        {
            None => println!("part1: reference self-consistent over reset (50 tokens)"),
            Some(i) => {
                let lo = i.saturating_sub(3);
                println!(
                    "part1: reference NOT self-consistent: first diff at {i}: \
                     run1 {:?} run2 {:?}",
                    &ref_tokens[lo..(i + 3).min(50)],
                    &ref_tokens2[lo..(i + 3).min(50)]
                );
            }
        }
        assert_eq!(
            ref_tokens, ref_tokens2,
            "single-session reference is not self-consistent"
        );
        drop(reference);

        // 双 session 交错：A 10 步 → B 10 步 × 5 轮（同为 100 步）。
        let mut a = engine.create_session().expect("session A");
        let mut b = engine.create_session().expect("session B");
        a.prefill(&prompt).expect("prefill A");
        b.prefill(&prompt).expect("prefill B");
        let mut ta = Vec::with_capacity(100);
        let mut tb = Vec::with_capacity(100);
        for _round in 0..5 {
            for _ in 0..10 {
                ta.push(a.decode_one(&cfg).unwrap());
            }
            for _ in 0..10 {
                tb.push(b.decode_one(&cfg).unwrap());
            }
        }
        assert_eq!(ta.len(), 50);
        assert_eq!(tb.len(), 50);
        for (name, got) in [("A", &ta), ("B", &tb)] {
            match got.iter().zip(&ref_tokens).position(|(x, y)| x != y) {
                None => println!("part1: session {name} == sequential (50 tokens)"),
                Some(i) => {
                    let lo = i.saturating_sub(3);
                    println!(
                        "part1: session {name} first diff at {i}: got {:?}.. want {:?}..",
                        &got[lo..(i + 3).min(50)],
                        &ref_tokens[lo..(i + 3).min(50)]
                    );
                    panic!("session {name} tokens diverged from sequential run at {i}");
                }
            }
        }
    }

    // ---- 部分 2：scheduler 池复用（reset-on-take 隔离）----
    {
        let sched = Scheduler::new(engine.clone(), 2).expect("scheduler");
        let handle = sched.handle();

        let prompt_a = make_prompt(192, 11);
        let prompt_b = make_prompt(192, 22);

        // 基准：首个 A / B job 的 12 token。
        let base_a = tokens_of(&run_job(&handle, prompt_a.clone()).await);
        let base_b = tokens_of(&run_job(&handle, prompt_b.clone()).await);
        assert_eq!(base_a.len(), 12);
        assert_eq!(base_b.len(), 12);

        // 交替复用池：A → B → A → B → B → A，逐 job 与基准完全一致。
        for (i, which) in ["a", "b", "a", "b", "b", "a"].iter().enumerate() {
            let events = run_job(
                &handle,
                if *which == "a" {
                    prompt_a.clone()
                } else {
                    prompt_b.clone()
                },
            )
            .await;
            let got = tokens_of(&events);
            let want = if *which == "a" { &base_a } else { &base_b };
            assert_eq!(&got, want, "job {i} ({which}) diverged from its reference");
            match events.last() {
                Some(GenEvent::Done {
                    reason: FinishReason::Length,
                    ..
                })
                | Some(GenEvent::Done {
                    reason: FinishReason::Stop,
                    ..
                }) => {}
                other => panic!("job {i}: expected Done(Length|Stop), got {other:?}"),
            }
        }
        println!("part2: 6 alternating pooled jobs byte-identical to references");
    }
}
