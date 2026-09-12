//! Integration tests against the real MiniCPM5-2B `tokenizer.json`.
//!
//! All tests early-return (with an `eprintln` note) when the model bundle is
//! absent, so the suite still passes in model-less CI environments.

use std::path::PathBuf;

use minicpm_tokenizer::{SpecialTokenIds, TokenizerService};

fn tokenizer_path() -> Option<PathBuf> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models/minicpm5-2b/tokenizer.json");
    if path.exists() {
        Some(path)
    } else {
        eprintln!(
            "SKIPPED: real tokenizer not found at {}, returning early",
            path.display()
        );
        None
    }
}

fn load() -> Option<TokenizerService> {
    tokenizer_path().map(|p| TokenizerService::load(&p).expect("tokenizer must load"))
}

#[test]
fn special_ids_match_minicpm5_expectation() {
    let Some(svc) = load() else { return };
    // MiniCPM5-2B: bos=<s>=0, eos={</s>=1, <|im_end|>=130073}, pad=</s>=1.
    // The second eos id comes from generation_config.json (eos_token_id: [1, 130073]).
    assert_eq!(
        svc.special_tokens(),
        SpecialTokenIds {
            bos: 0,
            eos: vec![1, 130073],
            pad: 1,
        }
    );
}

#[test]
fn encode_decode_roundtrip_multilingual() {
    let Some(svc) = load() else { return };
    let cases = [
        ("Chinese", "你好，世界！今天天气怎么样？"),
        ("English", "The quick brown fox jumps over the lazy dog."),
        ("Mixed", "MiniCPM5 是一个 2B 参数的模型，running on RTX 5080."),
        ("Code", "def fib(n):\n    if n < 2:\n        return n\n    return fib(n-1) + fib(n-2)"),
        ("Emoji", "😀🚀🎉 emoji 也能正确编码！🇨🇳"),
        ("Whitespace", "line1\nline2\r\n\ttabbed   spaces"),
    ];
    for (_name, text) in cases {
        let ids = svc.encode(text, false).expect("encode must succeed");
        assert!(!ids.is_empty(), "encoding produced no tokens for {text:?}");
        let decoded = svc.decode_batch(&ids).expect("decode must succeed");
        // Byte-level BPE roundtrip is exact; the required weaker property
        // (decoded text contains the original as a substring) holds trivially.
        assert_eq!(decoded, *text, "exact roundtrip failed for {text:?}");
    }
}

#[test]
fn add_special_tokens_true_prepends_bos() {
    let Some(svc) = load() else { return };
    let text = "你好，世界！Hello!";
    let without = svc.encode(text, false).expect("encode(false)");
    let with = svc.encode(text, true).expect("encode(true)");

    // Probed actual behavior: MiniCPM5's tokenizer.json ships a
    // TemplateProcessing post processor (`single: [<s>, A]`) which prepends
    // exactly one `<s>` token (id 0) when add_special_tokens=true.
    // => encode(true) == [0] ++ encode(false), for every text.
    assert_eq!(with.len(), without.len() + 1);
    assert_eq!(with[0], 0, "expected prepended <s> id 0");
    assert_eq!(&with[1..], &without[..], "tail after bos must equal encode(false)");

    // Because the chat template emits the bos token text itself
    // (`{{- bos_token }}` at the very top of chat_template.jinja), the server
    // MUST call encode with add_special_tokens=false; otherwise the bos token
    // would be duplicated ([0, 0, ...]).
    let both_bos = svc.encode("<s>", false).expect("encode bos text");
    assert_eq!(both_bos, vec![0], "bos text must encode to the single id 0");
}

#[test]
fn encode_is_deterministic() {
    let Some(svc) = load() else { return };
    let text = "确定性 determinism 测试 😀";
    let a = svc.encode(text, false).expect("first encode");
    let b = svc.encode(text, false).expect("second encode");
    assert_eq!(a, b, "same input must produce identical token ids");
    // Cross-thread determinism: the service is Send+Sync and shareable.
    let svc2 = svc.clone();
    let handle = std::thread::spawn(move || svc2.encode(text, false).expect("thread encode"));
    assert_eq!(handle.join().expect("thread must not panic"), a);
}

#[test]
fn decode_batch_keeps_special_tokens_and_is_deterministic() {
    let Some(svc) = load() else { return };
    // skip_special_tokens=false (streaming contract): stop tokens appear as text.
    assert_eq!(svc.decode_batch(&[]).expect("decode empty"), "");
    assert_eq!(svc.decode_batch(&[0, 1]).expect("decode ids"), "<s></s>");
    assert_eq!(svc.decode_batch(&[130073]).expect("decode im_end"), "<|im_end|>");

    // Deterministic for identical id batches (required by the incremental
    // streaming decoder: same prefix ids => same text).
    let ids: Vec<u32> = svc.encode("<|im_start|>user\nhi<|im_end|>", false).expect("encode");
    let d1 = svc.decode_batch(&ids).expect("decode 1");
    let d2 = svc.decode_batch(&ids).expect("decode 2");
    assert_eq!(d1, d2);
    assert_eq!(d1, "<|im_start|>user\nhi<|im_end|>");
}

#[test]
fn vocab_lookups() {
    let Some(svc) = load() else { return };
    // Probed: 130072 base BPE entries + added tokens => 130560 total ids.
    // 130560 also equals the model's logits dimension (see config.json), and
    // <|im_end|>=130073 / <unk>=130074 live in the added-token tail.
    assert_eq!(svc.vocab_size(), 130560);

    assert_eq!(svc.token_to_id("<s>"), Some(0));
    assert_eq!(svc.token_to_id("</s>"), Some(1));
    assert_eq!(svc.token_to_id("<|im_end|>"), Some(130073));
    assert_eq!(svc.token_to_id("<|no_such_token_xyz|>"), None);

    assert_eq!(svc.id_to_token(0), Some("<s>".to_string()));
    assert_eq!(svc.id_to_token(130073), Some("<|im_end|>".to_string()));
    assert_eq!(svc.id_to_token(u32::MAX), None);
}
