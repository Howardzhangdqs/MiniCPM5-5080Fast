//! dlopen 冒烟测试（可选加分项，默认 `#[ignore]`，不进常规验证）。
//!
//! 需要先构建 native 库与模型目录：
//! ```text
//! cmake -S native -B build/native && cmake --build build/native
//! # 模型资产：/workspace/models/minicpm5-2b（无 model.wpk → mock 权重模式）
//! ```
//! 然后手动运行：
//! ```text
//! cargo test -p minicpm-runtime --test dlopen_smoke -- --ignored --nocapture
//! ```
//!
//! 链路：dlopen `../../build/native/libminicpm_native.so` → load
//! `../../models/minicpm5-2b` → create session → prefill `[0,100,200]` →
//! decode 32 次（与 native/tests/abi_smoke.cpp ① 相同的 mock forward：
//! max_new_tokens 到时返回 eos 130073）→ stats 断言 prompt==3。

use minicpm_runtime::{GenerationConfig, NativeRuntime};

const NATIVE_LIB: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../build/native/libminicpm_native.so"
);
const MODEL_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../models/minicpm5-2b");

#[test]
#[ignore = "requires cmake-built libminicpm_native.so and a GPU; run with --ignored"]
fn dlopen_smoke_load_prefill_decode_stats() {
    if !std::path::Path::new(NATIVE_LIB).exists() {
        eprintln!("skip: {NATIVE_LIB} not found (build the native runtime first)");
        return;
    }
    if !std::path::Path::new(MODEL_DIR).join("config.json").exists() {
        eprintln!("skip: model dir {MODEL_DIR} incomplete");
        return;
    }

    let runtime = NativeRuntime::open(NATIVE_LIB).expect("dlopen libminicpm_native.so");
    let options = minicpm_runtime::native::default_model_options();
    let model = runtime
        .load_model(MODEL_DIR, &options)
        .expect("mc_model_load");

    let mut session = model.create_session().expect("mc_session_create");
    session.prefill(&[0, 100, 200]).expect("mc_prefill");

    let config = GenerationConfig {
        max_new_tokens: 32,
        temperature: 0.8,
        top_p: 0.95,
        top_k: 0,
        repetition_penalty: 1.0,
        seed: 42,
    };
    let mut last = 0;
    for _ in 0..32 {
        let token = session.decode_one(&config).expect("mc_decode_one");
        assert!((0..130_560).contains(&token), "token {token} outside vocab");
        last = token;
    }
    // Native mock forward returns eos (130073) when max_new_tokens is hit.
    assert_eq!(last, 130_073);

    let stats = session.stats().expect("mc_get_stats");
    assert_eq!(stats.prompt_tokens, 3, "prompt tokens must be 3");
    assert!(stats.generated_tokens >= 32);
    assert!(
        stats.prefill_us > 0 || stats.decode_us > 0,
        "timers should have accumulated"
    );

    // Reset keeps the session usable for a second round.
    session.reset().expect("mc_session_reset");
    session
        .prefill(&[0, 5, 10, 15])
        .expect("prefill after reset");
    let t = session.decode_one(&config).expect("decode after reset");
    assert!((0..130_560).contains(&t));
}
