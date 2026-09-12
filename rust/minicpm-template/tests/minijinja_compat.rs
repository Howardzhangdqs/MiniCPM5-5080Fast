//! Per-construct compatibility regression tests: every Jinja2/Python construct
//! used by the official MiniCPM5 `chat_template.jinja` is executed through the
//! same environment configuration `minicpm_template` uses, pinning minijinja's
//! behavior for each of them.

use minicpm_template::{ChatMessage, ChatTemplate};

fn render_tpl(source: &str, tools: Option<&serde_json::Value>) -> String {
    let tpl = ChatTemplate::from_source(source).expect("template compiles");
    // A dummy message satisfies the render() precondition (non-empty
    // conversation); none of the synthetic templates below iterate `messages`.
    let messages = vec![ChatMessage::new("user", "x")];
    tpl.render(&messages, tools, false, None, "<s>").expect("render")
}

#[test]
fn negative_step_slice_reversal() {
    // `messages[::-1]` (used to find the last user query).
    assert_eq!(
        render_tpl(
            "{% for m in tools.msgs[::-1] %}{{ m }};{% endfor %}",
            Some(&serde_json::json!({ "msgs": ["a", "b", "c"] }))
        ),
        "c;b;a;"
    );
    // Bounded slices are supported as well.
    assert_eq!(
        render_tpl(
            "{{ tools.l[1:] | join(',') }}|{{ tools.l[:2] | join(',') }}",
            Some(&serde_json::json!({ "l": [1, 2, 3, 4] }))
        ),
        "2,3,4|1,2"
    );
}

#[test]
fn empty_conversation_is_rejected_not_a_panic() {
    // minijinja 2.24 panics on an EMPTY reversed slice (index out of bounds in
    // range_step_backwards), and the official template slices
    // messages[::-1]; render() therefore rejects empty conversations with a
    // typed error instead of risking a panic.
    let tpl = ChatTemplate::from_source("{% for m in messages[::-1] %}{{ m }}{% endfor %}")
        .expect("compiles");
    let err = tpl
        .render(&[], None, false, None, "<s>")
        .expect_err("must reject empty conversations");
    assert!(err.to_string().contains("empty conversation"), "{err}");
}

#[test]
fn namespace_mutation_across_loop_iterations() {
    assert_eq!(
        render_tpl(
            "{% set ns = namespace(found=false, idx=0) %}\
             {% for m in tools.msgs %}\
               {% if not ns.found and m == 'user' %}{% set ns.found = true %}{% set ns.idx = loop.index0 %}{% endif %}\
             {% endfor %}{{ ns.idx }}",
            Some(&serde_json::json!({ "msgs": ["system", "user", "assistant", "user"] }))
        ),
        "1"
    );
}

#[test]
fn min_filter_over_literal_list() {
    // `[tool_calls_count, tool_sep_count]|min`
    assert_eq!(render_tpl("{{ [3, 1, 2] | min }}", None), "1");
}

#[test]
fn is_tests_string_defined_false_true() {
    // `message.content is string`, `enable_thinking is defined`,
    // `enable_thinking is false/true`.
    assert_eq!(
        render_tpl(
            "{{ tools.s is string }} {{ tools.n is string }} \
             {{ enable_thinking is defined }} \
             {{ tools.f is false }} {{ tools.f is true }}",
            Some(&serde_json::json!({ "s": "x", "n": 5, "f": false }))
        ),
        // minijinja renders booleans Python-style.
        "True False False True False"
    );
}

#[test]
fn undefined_variable_is_falsy_under_boolean_logic() {
    // `message.tool_calls and not has_tool_sep` with has_tool_sep undefined.
    assert_eq!(
        render_tpl("{% if tools.a and not has_tool_sep %}YES{% endif %}", Some(&serde_json::json!({ "a": [1] }))),
        "YES"
    );
}

#[test]
fn block_set_captures_filtered_body() {
    // `{%- set tool_definitions %} ... {%- endset %}` with whitespace control.
    assert_eq!(
        render_tpl("A{%- set x %} {{- 'abc' }} {%- endset %}B{{ x }}", None),
        "ABabc"
    );
}

#[test]
fn loop_variables_and_computed_indexing() {
    // loop.index0 / loop.first / loop.last plus `messages[loop.index0 - 1]`.
    assert_eq!(
        render_tpl(
            "{% for m in tools.msgs %}{{ loop.index0 }}:{{ m }}({% if loop.first %}F{% endif %}{% if loop.last %}L{% endif %})|{% endfor %}",
            Some(&serde_json::json!({ "msgs": ["a", "b"] }))
        ),
        "0:a(F)|1:b(L)|"
    );
}

#[test]
fn range_function() {
    assert_eq!(
        render_tpl("{% for i in range(1, 4) %}{{ i }}{% endfor %}", None),
        "123"
    );
}

#[test]
fn in_operator_on_strings() {
    assert_eq!(
        render_tpl(
            "{{ '<think>' in tools.a }} {{ '<think>' in tools.b }}",
            Some(&serde_json::json!({ "a": "x<think>y", "b": "xy" }))
        ),
        "True False"
    );
}

#[test]
fn python_string_methods() {
    assert_eq!(
        render_tpl(
            "{{ 'a<sep>b<sep>c'.split('<sep>')[1] }}|\
             {{ '  x  '.strip() }}|\
             {{ '\\n\\nx\\n\\n'.strip('\\n') }}|\
             {{ '\\n\\nx'.lstrip('\\n') }}|\
             {{ 'x\\n\\n'.rstrip('\\n') }}|\
             {{ 'ab'.startswith('a') }}|\
             {{ 'ab'.endswith('b') }}|\
             {{ 'a-b-a'.replace('a', 'X') }}",
            None
        ),
        "b|x|x|x|x|True|True|X-b-X"
    );
    // Python split keeps empty fields (unlike Rust's split_whitespace).
    assert_eq!(
        render_tpl("{{ 'a'.split('a') | length }}", None),
        "2"
    );
}

#[test]
fn dict_items_iteration_preserves_insertion_order() {
    // `args_dict.items()` must iterate in the order the JSON keys appear so
    // that `<param name=...>` output is stable and matches Python dicts.
    assert_eq!(
        render_tpl(
            "{% for k, v in tools.d.items() %}{{ k }}={{ v }};{% endfor %}",
            Some(&serde_json::json!({ "d": { "zeta": 1, "alpha": 2, "mid": 3 } }))
        ),
        "zeta=1;alpha=2;mid=3;"
    );
}

#[test]
fn tojson_accepts_ensure_ascii_kwarg() {
    assert_eq!(
        render_tpl(
            "{{ tools.d | tojson(ensure_ascii=False) }}",
            Some(&serde_json::json!({ "d": { "k": "中文" } }))
        ),
        "{\"k\":\"中文\"}"
    );
}

#[test]
fn whitespace_flags_match_transformers_environment() {
    // trim_blocks + lstrip_blocks are enabled like transformers'
    // ImmutableSandboxedEnvironment: the first newline after a block tag and
    // the indentation before it are dropped even without `-%}` markers.
    // (Without the flags this would render "  \nX".)
    assert_eq!(render_tpl("  {% if true %}\nX{% endif %}", None), "X");
}
