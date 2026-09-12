//! `minicpm-tokenizer`: Hugging Face `tokenizers` wrapper for MiniCPM5-2B.
//!
//! Loads the official `tokenizer.json` (BPE + added tokens, byte-level decoder)
//! without reimplementing any tokenization algorithm (plan §4.3). The service is
//! cheaply cloneable (`Arc` around the `Tokenizer`) and `Send + Sync`, so encode /
//! decode can be called from the tokio blocking pool or dedicated CPU worker
//! threads (plan §4.5).
//!
//! Special-token ids are resolved once at load time:
//!   * `bos_token` / `eos_token` / `pad_token` strings come from the sibling
//!     `tokenizer_config.json` (each field may be a plain string or a
//!     `{"content": "..."}` object — both shapes are accepted),
//!   * the strings are mapped to ids through the tokenizer vocabulary,
//!   * `eos` is additionally unioned with `eos_token_id` from the sibling
//!     `generation_config.json` (integer or array), because MiniCPM5-2B stops on
//!     both `</s>` (id 1) and `<|im_end|>` (id 130073).
//!
//! For MiniCPM5-2B the resolved ids are: `bos = 0`, `eos = [1, 130073]`,
//! `pad = 1`.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use thiserror::Error;
use tokenizers::Tokenizer;

/// Errors produced while loading or using a [`TokenizerService`].
#[derive(Debug, Error)]
pub enum TokenizerError {
    /// `Tokenizer::from_file` failed (corrupt/missing `tokenizer.json`).
    #[error("failed to load tokenizer from `{path}`: {source}")]
    Load {
        path: PathBuf,
        #[source]
        source: tokenizers::Error,
    },

    /// An auxiliary config file could not be read.
    #[error("failed to read `{path}`: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    /// An auxiliary config file is not valid JSON.
    #[error("failed to parse JSON in `{path}`: {source}")]
    Json {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },

    /// A required auxiliary config file is missing.
    #[error("required file `{0}` is missing")]
    MissingFile(PathBuf),

    /// A required field is absent from an auxiliary config file.
    #[error("field `{field}` is missing in `{path}`")]
    MissingField { field: String, path: PathBuf },

    /// A special token declared in the config is not part of the vocabulary.
    #[error("special token `{token}` (from `{path}`) not found in tokenizer vocabulary")]
    UnknownSpecialToken { token: String, path: PathBuf },

    /// `Tokenizer::encode` failed.
    #[error("encode failed: {0}")]
    Encode(#[source] tokenizers::Error),

    /// `Tokenizer::decode` failed.
    #[error("decode failed: {0}")]
    Decode(#[source] tokenizers::Error),
}

/// Resolved special-token ids for the loaded tokenizer.
///
/// For MiniCPM5-2B: `bos = 0`, `eos = [1, 130073]`, `pad = 1`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpecialTokenIds {
    /// Begin-of-sequence token id (`<s>`).
    pub bos: u32,
    /// All acceptable end-of-sequence token ids, sorted ascending.
    /// MiniCPM5-2B stops on both `</s>` (1) and `<|im_end|>` (130073).
    pub eos: Vec<u32>,
    /// Padding token id (`</s>`, id 1 for MiniCPM5-2B).
    pub pad: u32,
}

/// Thread-safe, cloneable tokenizer service backed by the HF `tokenizers` crate.
///
/// The [`Tokenizer`] is shared through an [`Arc`]; cloning the service is cheap
/// and all methods only take `&self`, so a single instance can be shared by the
/// scheduler, the streaming detokenizer and HTTP workers alike.
#[derive(Clone)]
pub struct TokenizerService {
    tokenizer: Arc<Tokenizer>,
    special: SpecialTokenIds,
}

// Compile-time guarantee that the service can cross thread/await boundaries.
const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<TokenizerService>();
};

impl TokenizerService {
    /// Load `tokenizer.json` from `path` and resolve special tokens from the
    /// sibling `tokenizer_config.json` / `generation_config.json`.
    pub fn load(path: impl AsRef<Path>) -> Result<Self, TokenizerError> {
        let path = path.as_ref();
        let tokenizer = Tokenizer::from_file(path).map_err(|source| TokenizerError::Load {
            path: path.to_path_buf(),
            source,
        })?;
        let dir = path.parent().unwrap_or_else(|| Path::new("."));
        let special = Self::resolve_special_token_ids(&tokenizer, dir)?;
        Ok(Self {
            tokenizer: Arc::new(tokenizer),
            special,
        })
    }

    /// Tokenize `text`.
    ///
    /// `add_special_tokens` toggles the tokenizer's post processor. MiniCPM5's
    /// `tokenizer.json` ships a `TemplateProcessing` post processor that prepends
    /// `<s>` (id 0) to every sequence, so `true` yields one extra leading token.
    /// The chat template already emits the bos token text itself, therefore the
    /// server must encode prompts with `add_special_tokens = false` to avoid a
    /// duplicated bos token.
    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>, TokenizerError> {
        let encoding = self
            .tokenizer
            .encode(text, add_special_tokens)
            .map_err(TokenizerError::Encode)?;
        Ok(encoding.get_ids().to_vec())
    }

    /// Decode a contiguous batch of token ids into text.
    ///
    /// `skip_special_tokens = false`: special tokens such as `<s>` / `</s>` /
    /// `<|im_end|>` are materialized in the output so the streaming incremental
    /// decoder can assemble stop strings from the raw text. Decoding is a pure
    /// function of `ids`, hence deterministic across calls and threads.
    pub fn decode_batch(&self, ids: &[u32]) -> Result<String, TokenizerError> {
        self.tokenizer
            .decode(ids, false)
            .map_err(TokenizerError::Decode)
    }

    /// Map a token id to its string form (added tokens included).
    pub fn id_to_token(&self, id: u32) -> Option<String> {
        self.tokenizer.id_to_token(id)
    }

    /// Map a token string to its id (added tokens included).
    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.tokenizer.token_to_id(token)
    }

    /// Vocabulary size including added tokens.
    pub fn vocab_size(&self) -> usize {
        self.tokenizer.get_vocab_size(true)
    }

    /// The special-token ids resolved at load time.
    pub fn special_tokens(&self) -> SpecialTokenIds {
        self.special.clone()
    }

    fn resolve_special_token_ids(
        tokenizer: &Tokenizer,
        dir: &Path,
    ) -> Result<SpecialTokenIds, TokenizerError> {
        let config_path = dir.join("tokenizer_config.json");
        let config = Self::read_json(&config_path)?;

        let bos_token = Self::special_token_string(config.get("bos_token")).ok_or_else(|| {
            TokenizerError::MissingField {
                field: "bos_token".into(),
                path: config_path.clone(),
            }
        })?;
        let eos_token = Self::special_token_string(config.get("eos_token")).ok_or_else(|| {
            TokenizerError::MissingField {
                field: "eos_token".into(),
                path: config_path.clone(),
            }
        })?;

        let bos = tokenizer.token_to_id(&bos_token).ok_or_else(|| {
            TokenizerError::UnknownSpecialToken {
                token: bos_token.clone(),
                path: config_path.clone(),
            }
        })?;
        let eos = tokenizer.token_to_id(&eos_token).ok_or_else(|| {
            TokenizerError::UnknownSpecialToken {
                token: eos_token.clone(),
                path: config_path.clone(),
            }
        })?;

        // Collect every stop-token id: the tokenizer_config eos string plus any
        // ids declared in generation_config.json (int or array). For MiniCPM5-2B
        // this turns [1] into [1, 130073] (`<|im_end|>`).
        let mut eos_ids = vec![eos];
        let gen_path = dir.join("generation_config.json");
        if let Ok(gen) = Self::read_json(&gen_path) {
            match gen.get("eos_token_id") {
                Some(serde_json::Value::Number(n)) => {
                    if let Some(id) = n.as_u64().and_then(|id| u32::try_from(id).ok()) {
                        eos_ids.push(id);
                    }
                }
                Some(serde_json::Value::Array(items)) => {
                    for item in items {
                        if let Some(id) = item.as_u64().and_then(|id| u32::try_from(id).ok()) {
                            eos_ids.push(id);
                        }
                    }
                }
                _ => {}
            }
        }
        eos_ids.sort_unstable();
        eos_ids.dedup();

        // pad_token is optional in general; fall back to eos, then bos.
        let pad = Self::special_token_string(config.get("pad_token"))
            .and_then(|t| tokenizer.token_to_id(&t))
            .unwrap_or(eos);

        Ok(SpecialTokenIds { bos, eos: eos_ids, pad })
    }

    fn read_json(path: &Path) -> Result<serde_json::Value, TokenizerError> {
        let text = std::fs::read_to_string(path).map_err(|source| TokenizerError::Io {
            path: path.to_path_buf(),
            source,
        })?;
        serde_json::from_str(&text).map_err(|source| TokenizerError::Json {
            path: path.to_path_buf(),
            source,
        })
    }

    /// Accept `"token"` or `{"content": "token", ...}` shapes used by
    /// HuggingFace tokenizer configs / special token maps.
    fn special_token_string(value: Option<&serde_json::Value>) -> Option<String> {
        match value {
            Some(serde_json::Value::String(s)) => Some(s.clone()),
            Some(serde_json::Value::Object(map)) => map
                .get("content")?
                .as_str()
                .map(str::to_owned),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn special_token_string_accepts_both_field_shapes() {
        // Plain-string shape (tokenizer_config.json of MiniCPM5-2B).
        assert_eq!(
            TokenizerService::special_token_string(Some(&json!("</s>"))),
            Some("</s>".to_string())
        );
        // Object shape (special_tokens_map.json style).
        assert_eq!(
            TokenizerService::special_token_string(Some(&json!({
                "content": "</s>", "lstrip": false, "special": true
            }))),
            Some("</s>".to_string())
        );
        // Absent / malformed fields yield None instead of panicking.
        assert_eq!(TokenizerService::special_token_string(None), None);
        assert_eq!(
            TokenizerService::special_token_string(Some(&json!({"special": true}))),
            None
        );
        assert_eq!(
            TokenizerService::special_token_string(Some(&json!(123))),
            None
        );
    }
}
