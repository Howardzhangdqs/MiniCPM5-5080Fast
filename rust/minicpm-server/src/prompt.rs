//! [`PromptBuilder`] implementations.
//!
//! * [`TemplatePromptBuilder`] — the production path: the official
//!   MiniCPM5-2B `chat_template.jinja` executed by `minicpm-template`, plus
//!   the HF tokenizer from `minicpm-tokenizer`. Prompt encoding always uses
//!   `add_special_tokens = false` (the template emits the bos text itself;
//!   `true` would duplicate token id 0 — verified against the real
//!   tokenizer).
//! * [`ScriptedPromptBuilder`] — deterministic test double: bidirectional
//!   char ↔ id mapping into `[100, 1100)` so server tests never touch model
//!   files.

use std::path::Path;

use minicpm_template::{ChatMessage, ChatTemplate};
use minicpm_tokenizer::TokenizerService;

use crate::PromptBuilder;

// ---------------------------------------------------------------- real ----

/// Production prompt builder: official template + real tokenizer.
pub struct TemplatePromptBuilder {
    tokenizer: TokenizerService,
    template: ChatTemplate,
    bos: String,
}

impl TemplatePromptBuilder {
    /// Load tokenizer + chat template from a model directory
    /// (`tokenizer.json`, `chat_template.jinja`, `tokenizer_config.json`,
    /// `generation_config.json`).
    pub fn new(model_dir: impl AsRef<Path>) -> anyhow::Result<Self> {
        let model_dir = model_dir.as_ref();
        let tokenizer = TokenizerService::load(model_dir.join("tokenizer.json")).map_err(|e| {
            anyhow::anyhow!(
                "tokenizer load ({:?}): {e}",
                model_dir.join("tokenizer.json")
            )
        })?;
        let template = ChatTemplate::from_model_dir(model_dir)
            .map_err(|e| anyhow::anyhow!("chat template load: {e}"))?;
        // The template needs the bos *string*; resolve it from the special
        // token id via the vocabulary (TokenizerService exposes ids only).
        let special = tokenizer.special_tokens();
        let bos = tokenizer
            .id_to_token(special.bos)
            .ok_or_else(|| anyhow::anyhow!("bos token id {} not in vocabulary", special.bos))?;
        Ok(Self {
            tokenizer,
            template,
            bos,
        })
    }
}

impl PromptBuilder for TemplatePromptBuilder {
    fn prompt_ids(
        &self,
        messages: &[serde_json::Value],
        tools: Option<&serde_json::Value>,
        enable_thinking: Option<bool>,
    ) -> anyhow::Result<Vec<i32>> {
        let msgs: Vec<ChatMessage> = messages
            .iter()
            .map(|m| {
                serde_json::from_value(m.clone()).map_err(|e| anyhow::anyhow!("bad message: {e}"))
            })
            .collect::<anyhow::Result<_>>()?;
        let text = self
            .template
            .render(&msgs, tools, true, enable_thinking, &self.bos)
            .map_err(|e| anyhow::anyhow!("template render: {e}"))?;
        // add_special_tokens=false: template already emits the bos text.
        let ids = self
            .tokenizer
            .encode(&text, false)
            .map_err(|e| anyhow::anyhow!("encode: {e}"))?;
        Ok(ids.into_iter().map(|id| id as i32).collect())
    }

    fn decode_ids(&self, ids: &[u32]) -> anyhow::Result<String> {
        self.tokenizer
            .decode_batch(ids)
            .map_err(|e| anyhow::anyhow!("decode: {e}"))
    }

    fn bos_token(&self) -> &str {
        &self.bos
    }
}

// --------------------------------------------------------------- scripted ----

/// Deterministic test double with a bidirectional char ↔ id mapping.
///
/// `char_to_id` maps a char's code point into `[100, 1100)`; `id_to_char`
/// inverts it. Exact round trips hold for code points `< 1000` (ASCII and
/// Latin-1), which is all the tests use.
pub struct ScriptedPromptBuilder;

/// Map a char to a fake token id in `[100, 1100)`.
pub fn char_to_id(c: char) -> i32 {
    100 + ((c as u32 % 1000) as i32)
}

/// Inverse of [`char_to_id`] (lossy for code points ≥ 1000).
pub fn id_to_char(id: i32) -> char {
    let cp = (id - 100).rem_euclid(1000) as u32;
    char::from_u32(cp).unwrap_or('\u{FFFD}')
}

impl PromptBuilder for ScriptedPromptBuilder {
    fn prompt_ids(
        &self,
        messages: &[serde_json::Value],
        tools: Option<&serde_json::Value>,
        _enable_thinking: Option<bool>,
    ) -> anyhow::Result<Vec<i32>> {
        if messages.is_empty() {
            anyhow::bail!("empty conversation");
        }
        // A deterministic, injective "render": bos + role/content pairs.
        let mut text = String::from("<s>");
        for m in messages {
            let role = m.get("role").and_then(|v| v.as_str()).unwrap_or("user");
            let content = match m.get("content") {
                Some(serde_json::Value::String(s)) => s.clone(),
                Some(other) => other.to_string(),
                None => String::new(),
            };
            text.push_str(&format!("[{role}] {content}\n"));
        }
        if let Some(tools) = tools {
            text.push_str(&format!(
                "[tools] {}\n",
                serde_json::to_string(tools).unwrap_or_default()
            ));
        }
        Ok(text.chars().map(char_to_id).collect())
    }

    fn decode_ids(&self, ids: &[u32]) -> anyhow::Result<String> {
        Ok(ids.iter().map(|id| id_to_char(*id as i32)).collect())
    }

    fn bos_token(&self) -> &str {
        "<s>"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn scripted_roundtrip_ascii() {
        let b = ScriptedPromptBuilder;
        let text = "Hello, world! 123 <>\"";
        let ids = b
            .prompt_ids(&[json!({"role": "user", "content": text})], None, None)
            .unwrap();
        assert!(!ids.is_empty());
        assert!(ids.iter().all(|id| (100..1100).contains(id)));
        let back = b
            .decode_ids(&(ids.iter().map(|i| *i as u32).collect::<Vec<_>>()))
            .unwrap();
        assert!(
            back.contains(text),
            "decoded `{back}` must contain `{text}`"
        );
    }

    #[test]
    fn scripted_rejects_empty_conversation() {
        assert!(ScriptedPromptBuilder.prompt_ids(&[], None, None).is_err());
    }

    #[test]
    fn scripted_bos_is_angle_s() {
        assert_eq!(ScriptedPromptBuilder.bos_token(), "<s>");
    }
}
