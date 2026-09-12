//! Golden snapshot tests (plan §4.4 / §17.1, P1 set): render the official
//! `chat_template.jinja` for every scenario and byte-compare with
//! `tests/fixtures/<name>.txt`.
//!
//! Regenerate expectations with:
//! ```text
//! BLESS=1 cargo test -p minicpm-template --test golden
//! ```

mod common;

use std::fs;
use std::path::PathBuf;

use common::{scenarios, BOS_TOKEN};
use minicpm_template::{ChatMessage, ChatTemplate};

fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

fn bless() -> bool {
    std::env::var("BLESS").ok().as_deref() == Some("1")
}

/// Write when `BLESS=1`, byte-compare otherwise. The `.txt` file *is* the
/// render output, stored byte-exactly.
fn check_or_bless(name: &str, actual: &str) {
    let path = fixtures_dir().join(format!("{name}.txt"));
    if bless() {
        fs::create_dir_all(fixtures_dir()).expect("create fixtures dir");
        fs::write(&path, actual).expect("write fixture");
        return;
    }
    let expected = match fs::read_to_string(&path) {
        Ok(expected) => expected,
        Err(err) => panic!(
            "missing golden fixture {} ({err}); run `BLESS=1 cargo test -p minicpm-template` to generate",
            path.display()
        ),
    };
    assert_eq!(actual, expected, "golden mismatch for scenario `{name}`");
}

#[test]
fn golden_scenarios_match_fixtures() {
    let Some(dir) = common::model_dir() else { return };
    let template = ChatTemplate::from_model_dir(&dir).expect("load official template");

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
        check_or_bless(scenario.name, &rendered);
    }
}

/// Structural sanity over the goldens (guarded assertions that must hold for
/// *any* correct render of the official template).
#[test]
fn golden_invariants() {
    let Some(dir) = common::model_dir() else { return };
    let template = ChatTemplate::from_model_dir(&dir).expect("load official template");

    let render = |name: &str| -> String {
        let scenario = scenarios()
            .into_iter()
            .find(|s| s.name == name)
            .unwrap_or_else(|| panic!("unknown scenario {name}"));
        template
            .render(
                &scenario.messages,
                scenario.tools.as_ref(),
                scenario.add_generation_prompt,
                scenario.enable_thinking,
                BOS_TOKEN,
            )
            .expect("render")
    };

    // Exactly one bos token at the very start of every render: the template
    // emits `bos_token` as text and the server encodes with
    // add_special_tokens=false, so the bos is never duplicated.
    for scenario in scenarios() {
        let rendered = template
            .render(
                &scenario.messages,
                scenario.tools.as_ref(),
                scenario.add_generation_prompt,
                scenario.enable_thinking,
                BOS_TOKEN,
            )
            .unwrap();
        assert!(
            rendered.starts_with("<s>"),
            "{} lacks leading bos",
            scenario.name
        );
        assert!(
            !rendered["<s>".len()..].contains("<s>"),
            "{} contains a duplicated bos token",
            scenario.name
        );
        assert!(
            rendered.ends_with('\n'),
            "{} must end with a newline (im_end or think block)",
            scenario.name
        );
    }

    // enable_thinking branch differences under add_generation_prompt:
    //   Some(true)  -> open a fresh <think> block
    //   Some(false) -> pre-fill an EMPTY think block (no-think prefill)
    //   None        -> no think scaffolding at all
    let t = render("08a_enable_thinking_true");
    let f = render("08b_enable_thinking_false");
    let n = render("08c_enable_thinking_none");
    assert!(t.ends_with("<|im_start|>assistant\n<think>\n"), "true: {t:?}");
    assert!(
        f.ends_with("<|im_start|>assistant\n<think>\n\n</think>\n\n"),
        "false: {f:?}"
    );
    assert!(n.ends_with("<|im_start|>assistant\n"), "none: {n:?}");
    // The three share the identical conversation prefix before the generation
    // prompt — only the tail differs.
    assert_eq!(&t[..n.len()], &n[..]);
    assert_eq!(&f[..n.len()], &n[..]);

    // History assistant turns with embedded <think>...</think> are re-emitted
    // with the reasoning preserved and the answer after </think>.
    let three = render("03_multiturn_embedded_think");
    assert!(
        three.contains("<|im_start|>assistant\n<think>\n9.11 = 9.11"),
        "{three}"
    );
    assert!(
        three.contains("\n</think>\n\n9.9 更大。<|im_end|>"),
        "{three}"
    );

    // reasoning_content field takes precedence over content-embedded think.
    let four = render("04_reasoning_content");
    assert!(
        four.contains("<|im_start|>assistant\n<think>\n用户让我做自我介绍"),
        "{four}"
    );

    // Tools turn on the `# Tools` header plus the JSON schema (no system).
    let six = render("06_tools_without_system");
    assert!(
        six.contains("<s><|im_start|>system\n# Tools\n\nYou are provided with function signatures"),
        "{six}"
    );

    // Tools + plain system: definitions appended after "\n\n".
    let five = render("05_tools_with_system");
    assert!(
        five.contains("你是一个乐于助人的助手。\n\n# Tools\n"),
        "{five}"
    );
    // Chinese passes through `tojson(ensure_ascii=False)` unescaped.
    assert!(five.contains("获取指定城市的当前天气"), "{five}");
    // NOTE: our tojson override serializes with serde_json semantics — compact
    // separators, insertion-ordered keys (preserve_order), no HTML escaping.
    // Jinja2's Python tojson differs (sorted keys, ", "/"： " separators,
    // \u003c-style escaping) — tracked as a parity risk in the project report.
    assert!(five.contains("\"name\":\"get_current_weather\""), "{five}");

    // Consecutive tool responses are grouped into ONE user block; the closing
    // <|im_end|> is stripped directly onto the last </tool_response> (the
    // template emits it with {{- whitespace control).
    let seven = render("07_tool_response");
    assert_eq!(
        seven.matches("<|im_start|>user\n<tool_response>").count(),
        1,
        "{seven}"
    );
    assert_eq!(
        seven.matches("</tool_response><|im_end|>\n").count(),
        1,
        "{seven}"
    );
    // String tool content passes through verbatim...
    assert!(
        seven.contains("<tool_response>\n{\"temperature_c\": 25, \"condition\": \"晴\"}\n</tool_response>"),
        "{seven}"
    );
    // ... array tool content is JSON-serialized via `message.content | tojson`.
    assert!(
        seven.contains("<tool_response>\n[{\"type\":\"text\",\"text\":\"湿度 40%\"}]\n</tool_response>"),
        "{seven}"
    );
}

/// Documents the (surprising but official) semantics of assistant `tool_calls`
/// turns under Jinja2 scoping rules — verified identical in Jinja2 and
/// minijinja:
///
/// * the `<tool_sep>` content-processing loops (lines 54–117 of the official
///   template) assign `processed_content` INSIDE `{% for %}` bodies, and Jinja
///   scoping keeps those assignments loop-local — so `content` ends up as
///   `content_parts[0]`: everything from the FIRST `<tool_sep>` marker onward
///   is DISCARDED from the rendered prompt,
/// * the `not has_tool_sep` block (has_tool_sep is undefined ⇒ falsy ⇒ the
///   block always runs) is what actually emits the tool XML: exactly once per
///   tool call, `\n`-separated, with the first separator omitted when the
///   (processed) content is empty.
#[test]
fn tool_calls_scoping_behavior_is_pinned() {
    let Some(dir) = common::model_dir() else { return };
    let template = ChatTemplate::from_model_dir(&dir).expect("load official template");

    // Scenario 09: content "我来查一下。<tool_sep>" + two tool calls.
    let nine = scenarios()
        .into_iter()
        .find(|s| s.name == "09_tool_calls_history")
        .expect("scenario 09 exists");
    let rendered = template
        .render(
            &nine.messages,
            nine.tools.as_ref(),
            nine.add_generation_prompt,
            nine.enable_thinking,
            BOS_TOKEN,
        )
        .expect("render");

    // Text after the first <tool_sep> marker is dropped (loop-local set).
    assert!(!rendered.contains("<tool_sep>"), "{rendered}");
    assert!(rendered.contains("我来查一下。\n<function name=\"get_current_weather\">"), "{rendered}");
    // Each tool call's XML appears exactly once...
    assert_eq!(rendered.matches("<function name=\"get_current_weather\">").count(), 1);
    assert_eq!(rendered.matches("<function name=\"get_current_time\">").count(), 1);
    // ...separated by a single newline.
    assert!(
        rendered.contains("</function>\n<function name=\"get_current_time\">"),
        "{rendered}"
    );

    // Scenario 07: EMPTY assistant content + one tool call → no leading '\n'.
    let seven = scenarios()
        .into_iter()
        .find(|s| s.name == "07_tool_response")
        .expect("scenario 07 exists");
    let rendered = template
        .render(
            &seven.messages,
            seven.tools.as_ref(),
            seven.add_generation_prompt,
            seven.enable_thinking,
            BOS_TOKEN,
        )
        .expect("render");
    assert!(
        rendered.contains("<think>\n\n</think>\n\n<function name=\"get_current_weather\">"),
        "{rendered}"
    );
    assert_eq!(
        rendered.matches("<function name=\"get_current_weather\">").count(),
        1
    );
}

/// `<tool_def_sep>` system variant (inline, not a golden fixture — the task
/// pins the fixture to the plain variant): the marker is replaced in-place by
/// the tool definitions instead of being appended after "\n\n".
#[test]
fn tool_def_sep_system_variant_replaces_marker() {
    let Some(dir) = common::model_dir() else { return };
    let template = ChatTemplate::from_model_dir(&dir).expect("load official template");
    let tools = serde_json::json!([{
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "获取指定城市的当前天气",
            "parameters": {
                "type": "object",
                "properties": { "city": { "type": "string" } }
            }
        }
    }]);

    let rendered = template
        .render(
            &[
                ChatMessage::new("system", "你是天气助手。<tool_def_sep>请注意安全。"),
                ChatMessage::new("user", "北京天气？"),
            ],
            Some(&tools),
            true,
            None,
            BOS_TOKEN,
        )
        .expect("render");

    // Marker replaced by the tool block, surrounding system text preserved.
    assert!(!rendered.contains("<tool_def_sep>"), "{rendered}");
    assert!(rendered.contains("你是天气助手。# Tools\n\n"), "{rendered}");
    assert!(rendered.contains("请注意安全。"), "{rendered}");

    // Contrast with the plain variant, which appends after "\n\n".
    let plain = template
        .render(
            &[
                ChatMessage::new("system", "你是天气助手。"),
                ChatMessage::new("user", "北京天气？"),
            ],
            Some(&tools),
            true,
            None,
            BOS_TOKEN,
        )
        .expect("render");
    assert!(plain.contains("你是天气助手。\n\n# Tools"), "{plain}");
    assert_ne!(rendered, plain);
}
