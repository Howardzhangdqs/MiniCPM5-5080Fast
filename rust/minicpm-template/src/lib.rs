//! `minicpm-template`: executes the *official* MiniCPM5-2B `chat_template.jinja`
//! verbatim with the `minijinja` engine (plan §4.4). No template logic is
//! re-implemented in Rust; instead, the handful of Python/Jinja2 constructs the
//! official template relies on are supplied through minijinja's sanctioned
//! extension points:
//!
//! * `Environment::add_filter("tojson", ...)` — the built-in `tojson` filter
//!   rejects the `ensure_ascii` keyword argument, so it is overridden with a
//!   `serde_json`-based implementation that accepts (and honors) it.
//! * `Environment::set_unknown_method_callback(...)` — Python string methods
//!   (`.split/.strip/.lstrip/.rstrip/.startswith/.endswith/.replace`) and
//!   `dict.items()` are not built into minijinja; the callback implements them
//!   with Python semantics.
//!
//! The rendering environment mirrors transformers' chat-template environment
//! (`ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
//! keep_trailing_newline=False)`) so whitespace handling matches
//! `apply_chat_template` byte for byte.

use std::path::{Path, PathBuf};

use minijinja::value::{from_args, Kwargs, Value, ValueKind};
use minijinja::{Environment, Error, ErrorKind, State};
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Name under which the chat template is registered in the [`Environment`].
const CHAT_TEMPLATE_NAME: &str = "__minicpm_chat_template__";

/// Errors produced while loading or rendering a [`ChatTemplate`].
#[derive(Debug, Error)]
pub enum TemplateError {
    /// A file could not be read.
    #[error("failed to read `{path}`: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    /// A config file is not valid JSON.
    #[error("failed to parse JSON in `{path}`: {source}")]
    Json {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },

    /// Neither `chat_template.jinja` nor a `chat_template` field exists.
    #[error("no chat template found in `{0}`: expected `chat_template.jinja` or a `chat_template` field in `tokenizer_config.json`")]
    MissingTemplate(PathBuf),

    /// The template failed to compile.
    #[error("failed to compile chat template: {0}")]
    Compile(Error),

    /// The template failed to render.
    #[error("failed to render chat template: {0}")]
    Render(Error),

    /// Message serialization failed (should be impossible for [`ChatMessage`]).
    #[error("failed to serialize chat messages: {0}")]
    Ser(#[source] serde_json::Error),

    /// `render` was called with an empty conversation.
    #[error("cannot render an empty conversation: the official chat template indexes messages[0] and slices messages[::-1]")]
    EmptyMessages,
}

/// One chat message, shaped after the OpenAI-style message objects consumed by
/// the official template.
///
/// `content` is deliberately a raw JSON value: the template passes string
/// contents through unchanged but JSON-serializes non-string contents
/// (`tool` messages support array payloads), so both shapes must survive.
///
/// `tool_calls` follows the `[{id, type, function: {name, arguments}}]` layout;
/// `arguments` must be a JSON *object* (not the OpenAI stringified form) because
/// the template calls `arguments.items()` on it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: serde_json::Value,
    /// Reasoning text for assistant turns (Qwen-style `reasoning_content`).
    /// Absent (not `null`) when unused so `is string` tests behave like Python.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    /// Assistant tool calls. Absent when the turn contains none.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<serde_json::Value>,
}

impl ChatMessage {
    /// Plain text message (system / user / simple assistant / tool).
    pub fn new(role: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            content: serde_json::Value::String(content.into()),
            reasoning_content: None,
            tool_calls: None,
        }
    }

    /// Structured message (e.g. a `tool` message with array content).
    pub fn with_json_content(role: impl Into<String>, content: serde_json::Value) -> Self {
        Self {
            role: role.into(),
            content,
            reasoning_content: None,
            tool_calls: None,
        }
    }

    /// Attach `reasoning_content` (assistant turns).
    pub fn with_reasoning_content(mut self, reasoning: impl Into<String>) -> Self {
        self.reasoning_content = Some(reasoning.into());
        self
    }

    /// Attach `tool_calls` (assistant turns).
    pub fn with_tool_calls(mut self, tool_calls: serde_json::Value) -> Self {
        self.tool_calls = Some(tool_calls);
        self
    }

    /// The content as a plain string, if it is one.
    pub fn content_str(&self) -> Option<&str> {
        self.content.as_str()
    }
}

/// A compiled chat template bound to a minijinja environment.
///
/// Cloning is not needed: rendering takes `&self` and the underlying
/// environment is thread-safe (`Send + Sync`), so one instance can be shared
/// across worker threads.
#[derive(Debug, Clone)]
pub struct ChatTemplate {
    env: Environment<'static>,
    source_path: Option<PathBuf>,
}

// Compile-time guarantee that the template can be shared across threads.
const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<ChatTemplate>();
};

impl ChatTemplate {
    /// Load the template from a model directory.
    ///
    /// Resolution order:
    ///   1. `<dir>/chat_template.jinja` (preferred, the file shipped by
    ///      MiniCPM5-2B),
    ///   2. the `chat_template` string field of `<dir>/tokenizer_config.json`.
    pub fn from_model_dir(dir: impl AsRef<Path>) -> Result<Self, TemplateError> {
        let dir = dir.as_ref();
        let jinja_path = dir.join("chat_template.jinja");
        match std::fs::read_to_string(&jinja_path) {
            Ok(source) if !source.trim().is_empty() => {
                Self::compile(source).map(|t| t.with_source_path(jinja_path))
            }
            _ => {
                let config_path = dir.join("tokenizer_config.json");
                let text = std::fs::read_to_string(&config_path).map_err(|source| {
                    TemplateError::Io {
                        path: config_path.clone(),
                        source,
                    }
                })?;
                let config: serde_json::Value = serde_json::from_str(&text).map_err(|source| {
                    TemplateError::Json {
                        path: config_path.clone(),
                        source,
                    }
                })?;
                match config.get("chat_template") {
                    Some(serde_json::Value::String(source)) => {
                        Self::compile(source.clone()).map(|t| t.with_source_path(config_path))
                    }
                    // Array-valued chat_template fields (template variants) are
                    // not supported; MiniCPM5-2B ships a standalone .jinja file.
                    _ => Err(TemplateError::MissingTemplate(dir.to_path_buf())),
                }
            }
        }
    }

    /// Compile a template directly from source (useful for tests).
    pub fn from_source(source: &str) -> Result<Self, TemplateError> {
        Self::compile(source.to_string())
    }

    fn compile(source: String) -> Result<Self, TemplateError> {
        let mut env = build_environment();
        // The environment owns templates for `'static`, so hand the source
        // over through a loader instead of leaking it.
        env.set_loader(move |name| {
            if name == CHAT_TEMPLATE_NAME {
                Ok(Some(source.clone()))
            } else {
                Ok(None)
            }
        });
        // Compile eagerly (get_template runs the loader) so syntax errors in
        // the official template surface at load time, not at first render.
        env.get_template(CHAT_TEMPLATE_NAME)
            .map_err(TemplateError::Compile)?;
        Ok(Self {
            env,
            source_path: None,
        })
    }

    fn with_source_path(mut self, path: PathBuf) -> Self {
        self.source_path = Some(path);
        self
    }

    /// Where the template source was loaded from, if it came from disk.
    pub fn source_path(&self) -> Option<&Path> {
        self.source_path.as_deref()
    }

    /// Render the prompt for the given conversation.
    ///
    /// Context variable names match the official template exactly:
    /// `messages`, `tools`, `bos_token`, `add_generation_prompt`,
    /// `enable_thinking`. `tools` and `enable_thinking` are only injected when
    /// `Some(..)` — the template probes both with truthiness / `is defined`, and
    /// a `None` `enable_thinking` must stay *undefined* so the template's
    /// `{%- if enable_thinking is defined %}` branch is skipped entirely.
    pub fn render(
        &self,
        messages: &[ChatMessage],
        tools: Option<&serde_json::Value>,
        add_generation_prompt: bool,
        enable_thinking: Option<bool>,
        bos_token: &str,
    ) -> Result<String, TemplateError> {
        // The official template reads messages[0].role and iterates
        // messages[::-1]. Besides being semantically invalid input, an empty
        // list would hit a minijinja 2.24 panic (empty reversed slice indexes
        // out of bounds in range_step_backwards), so reject it up front.
        if messages.is_empty() {
            return Err(TemplateError::EmptyMessages);
        }
        let template = self
            .env
            .get_template(CHAT_TEMPLATE_NAME)
            .map_err(TemplateError::Compile)?;

        let mut context = serde_json::Map::new();
        context.insert(
            "messages".to_string(),
            serde_json::to_value(messages).map_err(TemplateError::Ser)?,
        );
        context.insert(
            "add_generation_prompt".to_string(),
            serde_json::Value::Bool(add_generation_prompt),
        );
        context.insert(
            "bos_token".to_string(),
            serde_json::Value::String(bos_token.to_string()),
        );
        if let Some(tools) = tools {
            context.insert("tools".to_string(), tools.clone());
        }
        if let Some(enable_thinking) = enable_thinking {
            context.insert(
                "enable_thinking".to_string(),
                serde_json::Value::Bool(enable_thinking),
            );
        }

        template
            .render(Value::from_serialize(&context))
            .map_err(TemplateError::Render)
    }
}

/// Build the minijinja environment with all Jinja2/transformers compatibility
/// shims required by the official MiniCPM5 chat template.
fn build_environment() -> Environment<'static> {
    let mut env = Environment::new();

    // Match transformers' chat-template environment
    // (`ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)`,
    // Jinja2's default keep_trailing_newline=False equals minijinja's default).
    // The official template applies `{{-`/`{%-}` whitespace control everywhere,
    // so these flags are no-ops for it today — but they keep the environment
    // byte-compatible with Python for any future template revision.
    env.set_trim_blocks(true);
    env.set_lstrip_blocks(true);

    // `tojson` override: the built-in filter rejects `ensure_ascii`.
    env.add_filter("tojson", tojson_filter);

    // Python str/dict method support for `.split`, `.items()` and friends.
    env.set_unknown_method_callback(python_method_fallback);

    env
}

/// `tojson(ensure_ascii=...)` filter backed by `serde_json`.
///
/// The official template calls `tojson(ensure_ascii=False)`; minijinja's
/// built-in `tojson` rejects that keyword argument outright. This override
/// accepts (and honors) `ensure_ascii`:
///   * `false` / absent-value semantics here: non-ASCII characters are emitted
///     verbatim (`serde_json` never escapes them),
///   * `true`: every non-ASCII code point is escaped as `\uXXXX` (with
///     surrogate pairs above the BMP), like Python's `json.dumps`.
///
/// Note: Jinja2's own `tojson` additionally applies HTML-safe escaping
/// (`<`/`>`/`&`/`'` → `\u003c`/…) for XSS protection inside HTML documents.
/// That behavior is intentionally *not* replicated: for prompt construction the
/// raw characters must survive so tool schemas keep their XML tags literal.
/// This difference is called out in the project report for the Python
/// golden-parity phase.
fn tojson_filter(_state: &State, value: Value, kwargs: Kwargs) -> Result<String, Error> {
    let ensure_ascii: Option<bool> = kwargs.get("ensure_ascii")?;
    kwargs.assert_all_used()?;
    let plain = serde_json::to_string(&value).map_err(|err| {
        Error::new(
            ErrorKind::InvalidOperation,
            format!("tojson failed to serialize value: {err}"),
        )
    })?;
    if ensure_ascii == Some(true) {
        Ok(ascii_escape(&plain))
    } else {
        Ok(plain)
    }
}

/// Escape non-ASCII code points as `\uXXXX` (Python `ensure_ascii=True`).
fn ascii_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        let cp = c as u32;
        if cp <= 0x7f {
            out.push(c);
        } else if cp <= 0xffff {
            out.push_str(&format!("\\u{cp:04x}"));
        } else {
            let v = cp - 0x1_0000;
            out.push_str(&format!(
                "\\u{:04x}\\u{:04x}",
                0xd800 + (v >> 10),
                0xdc00 + (v & 0x3ff)
            ));
        }
    }
    out
}

/// Implements the Python string/dict methods used by the official template.
///
/// Invoked by minijinja whenever a method call on a value fails with
/// `UnknownMethod`. Unsupported names fall through to the original error.
fn python_method_fallback(
    state: &State,
    value: &Value,
    method: &str,
    args: &[Value],
) -> Result<Value, Error> {
    if value.kind() == ValueKind::Map && method == "items" {
        if !args.is_empty() {
            return Err(Error::new(
                ErrorKind::InvalidOperation,
                "dict.items() takes no arguments",
            ));
        }
        return state.apply_filter("items", &[value.clone()]);
    }
    if value.kind() == ValueKind::String {
        let s = value
            .as_str()
            .ok_or_else(|| Error::new(ErrorKind::InvalidOperation, "string value expected"))?;
        return string_method(s, method, args);
    }
    Err(Error::from(ErrorKind::UnknownMethod))
}

/// Python `str` methods with Python semantics.
fn string_method(s: &str, method: &str, args: &[Value]) -> Result<Value, Error> {
    match method {
        // str.split(sep=None): with a separator, empty fields are kept
        // ("a<X>b<X>c".split("<X>") == ["a", "b", "c"]); without one, runs of
        // whitespace are collapsed.
        "split" => {
            let (sep,): (Option<String>,) = from_args(args)?;
            match sep {
                Some(sep) if sep.is_empty() => Err(Error::new(
                    ErrorKind::InvalidOperation,
                    "str.split: empty separator",
                )),
                Some(sep) => Ok(Value::from(
                    s.split(sep.as_str()).map(Value::from).collect::<Vec<_>>(),
                )),
                None => Ok(Value::from(
                    s.split_whitespace().map(Value::from).collect::<Vec<_>>(),
                )),
            }
        }
        // str.strip/lstrip/rstrip(chars=None): the argument is a *set of
        // characters*, not a substring.
        "strip" | "lstrip" | "rstrip" => {
            let (chars,): (Option<String>,) = from_args(args)?;
            let predicate = |c: char| match &chars {
                Some(set) => set.contains(c),
                None => c.is_whitespace(),
            };
            let stripped = match method {
                "strip" => s.trim_matches(predicate),
                "lstrip" => s.trim_start_matches(predicate),
                _ => s.trim_end_matches(predicate),
            };
            Ok(Value::from(stripped))
        }
        "startswith" => {
            let (prefix,): (String,) = from_args(args)?;
            Ok(Value::from(s.starts_with(prefix.as_str())))
        }
        "endswith" => {
            let (suffix,): (String,) = from_args(args)?;
            Ok(Value::from(s.ends_with(suffix.as_str())))
        }
        // str.replace(old, new, count=-1): negative/absent count replaces all.
        "replace" => {
            let (old, new, count): (String, String, Option<i64>) = from_args(args)?;
            if old.is_empty() {
                return Err(Error::new(
                    ErrorKind::InvalidOperation,
                    "str.replace: empty search string",
                ));
            }
            match count {
                Some(n) if n >= 0 => {
                    Ok(Value::from(s.replacen(old.as_str(), new.as_str(), n as usize)))
                }
                _ => Ok(Value::from(s.replace(old.as_str(), new.as_str()))),
            }
        }
        _ => Err(Error::from(ErrorKind::UnknownMethod)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// Minimal smoke test that does not require model files: renders a
    /// hand-written template exercising the compatibility shims.
    #[test]
    fn compatibility_shims_work_end_to_end() {
        let tpl = ChatTemplate::from_source(
            "{{ 'a<sep>b'.split('<sep>')[1] }}|\
             {{ '  x '.strip() }}|\
             {{ '\\n\\nhi\\n'.strip('\\n') }}|\
             {{ 'abc'.startswith('ab') }}|\
             {{ 'abc'.endswith('bc') }}|\
             {{ 'a-b'.replace('-', '+') }}|\
             {% for k, v in tools.obj.items() %}{{ k }}:{{ v }},{% endfor %}|\
             {{ tools.obj | tojson(ensure_ascii=False) }}",
        )
        .expect("template compiles");
        let rendered = tpl
            .render(
                &[ChatMessage::new("user", "hi")],
                Some(&json!({"obj": {"a": 1}})),
                false,
                None,
                "<s>",
            )
            .expect("render");
        // Note: minijinja renders booleans Python-style (`True`/`False`); the
        // official template only uses booleans inside `if` conditions, so this
        // never appears in rendered prompts.
        assert_eq!(rendered, "b|x|hi|True|True|a+b|a:1,|{\"a\":1}");
    }

    #[test]
    fn tojson_escapes_non_ascii_only_when_requested() {
        let tpl = ChatTemplate::from_source(
            "{{ tools.d | tojson(ensure_ascii=False) }};{{ tools.d | tojson(ensure_ascii=True) }}",
        )
        .expect("template compiles");
        let out = tpl
            .render(
                &[ChatMessage::new("user", "x")],
                Some(&json!({"d": {"k": "中文😀"}})),
                false,
                None,
                "",
            )
            .expect("render");
        assert_eq!(
            out,
            "{\"k\":\"中文😀\"};{\"k\":\"\\u4e2d\\u6587\\ud83d\\ude00\"}"
        );
    }

    #[test]
    fn chat_message_serde_roundtrip() {
        let msg = ChatMessage::new("assistant", "answer").with_reasoning_content("why");
        let v = serde_json::to_value(&msg).unwrap();
        // Absent fields must not appear as null (template relies on `is string`).
        assert!(v.get("tool_calls").is_none());
        assert_eq!(v["reasoning_content"], json!("why"));

        let back: ChatMessage = serde_json::from_value(v).unwrap();
        assert_eq!(back, msg);
    }
}
