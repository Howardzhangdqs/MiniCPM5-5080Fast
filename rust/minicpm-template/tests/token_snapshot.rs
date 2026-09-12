//! Template → tokenizer id snapshots: the hook point for the Python golden
//! parity pipeline (plan §4.4 step 4 / §17.1: `python_token_ids ==
//! rust_token_ids`).
//!
//! Every golden scenario is rendered and encoded with
//! `add_special_tokens = false` (the template supplies the bos token text, so
//! the tokenizer post-processor must stay off — see minicpm-tokenizer tests).
//! The resulting ids are stored in `tests/fixtures/<name>.ids`, one
//! comma-separated line.
//!
//! Regenerate with:
//! ```text
//! BLESS=1 cargo test -p minicpm-template --test token_snapshot
//! ```

mod common;

use std::fs;
use std::path::PathBuf;

use common::{scenarios, BOS_TOKEN};
use minicpm_template::ChatTemplate;
use minicpm_tokenizer::TokenizerService;

fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

fn bless() -> bool {
    std::env::var("BLESS").ok().as_deref() == Some("1")
}

#[test]
fn token_id_snapshots() {
    let Some(dir) = common::model_dir() else { return };
    let template = ChatTemplate::from_model_dir(&dir).expect("load official template");
    let tokenizer = TokenizerService::load(dir.join("tokenizer.json")).expect("load tokenizer");

    for scenario in scenarios() {
        let rendered = template
            .render(
                &scenario.messages,
                scenario.tools.as_ref(),
                scenario.add_generation_prompt,
                scenario.enable_thinking,
                BOS_TOKEN,
            )
            .unwrap_or_else(|err| panic!("render failed for {}: {err}", scenario.name));

        // add_special_tokens=false: the template already emits "<s>" and the
        // tokenizer's TemplateProcessing post-processor would prepend a second
        // bos token (id 0) otherwise.
        let ids = tokenizer
            .encode(&rendered, false)
            .unwrap_or_else(|err| panic!("encode failed for {}: {err}", scenario.name));

        // The rendered "<s>" prefix must map to exactly one bos token (id 0).
        assert_eq!(ids.first(), Some(&0), "{}: expected bos id 0", scenario.name);
        assert_ne!(ids.get(1), Some(&0), "{}: duplicated bos id", scenario.name);

        let actual = format!(
            "{}\n",
            ids.iter()
                .map(u32::to_string)
                .collect::<Vec<_>>()
                .join(",")
        );
        let path = fixtures_dir().join(format!("{}.ids", scenario.name));
        if bless() {
            fs::create_dir_all(fixtures_dir()).expect("create fixtures dir");
            fs::write(&path, &actual).expect("write ids fixture");
            continue;
        }
        let expected = match fs::read_to_string(&path) {
            Ok(expected) => expected,
            Err(err) => panic!(
                "missing ids fixture {} ({err}); run `BLESS=1 cargo test -p minicpm-template` to generate",
                path.display()
            ),
        };
        assert_eq!(actual, expected, "token id snapshot mismatch for `{}`", scenario.name);
    }
}
