//! Incremental detokenizer for streaming output (plan §4.6).
//!
//! The GPU runtime only returns one `i32` token id per step. Converting ids
//! to text is done by re-decoding the whole id window through a caller
//! supplied callback and diffing against the previously emitted prefix.
//! This stays correct for BPE merge effects across token boundaries without
//! re-implementing any tokenizer logic.
//!
//! If a decode pass ends in `U+FFFD` the replacement character almost
//! certainly stands for an incomplete multi-byte UTF-8 sequence that the
//! next token will complete, so the tail is held back until more tokens
//! arrive (same strategy as HF / vLLM streaming decoders).

use crate::StreamDecodeError;

/// Decode callback: turns a window of token ids into text.
/// Implementations must be deterministic for the same id window.
pub type DecodeFn<'a> = &'a mut dyn FnMut(&[u32]) -> Result<String, StreamDecodeError>;

#[derive(Debug)]
pub struct StreamDecoder {
    ids: Vec<u32>,
    /// Bytes already handed out.
    emitted: usize,
    /// True while the previous decode ended in U+FFFD (tail withheld).
    tail_withheld: bool,
}

impl StreamDecoder {
    pub fn new() -> Self {
        Self { ids: Vec::new(), emitted: 0, tail_withheld: false }
    }

    /// Feed the next generated token id. Returns newly emittable UTF-8 text
    /// (may be empty while bytes are held back).
    pub fn push(&mut self, id: u32, decode: DecodeFn<'_>) -> Result<Option<String>, StreamDecodeError> {
        self.ids.push(id);
        let text = decode(&self.ids)?;
        self.sync_emission(&text);
        self.take_delta(&text)
    }

    /// Flush everything withheld at end of generation (stop/length/eos).
    pub fn finish(&mut self, decode: DecodeFn<'_>) -> Result<Option<String>, StreamDecodeError> {
        let text = decode(&self.ids)?;
        self.tail_withheld = false;
        self.take_delta(&text)
    }

    /// Total tokens fed so far.
    pub fn token_count(&self) -> usize {
        self.ids.len()
    }

    fn sync_emission(&mut self, text: &str) {
        // Withhold a trailing replacement char; it may be an incomplete
        // multi-byte sequence completed by the next token.
        self.tail_withheld = text.ends_with('\u{FFFD}');
    }

    fn take_delta(&mut self, text: &str) -> Result<Option<String>, StreamDecodeError> {
        let mut end = text.len();
        if self.tail_withheld {
            end = end.saturating_sub('\u{FFFD}'.len_utf8());
        }
        while end > self.emitted && !text.is_char_boundary(end) {
            end -= 1;
        }
        if end <= self.emitted {
            return Ok(None);
        }
        let delta = text[self.emitted..end].to_string();
        self.emitted = end;
        Ok(Some(delta))
    }
}

impl Default for StreamDecoder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Tokenizer-like backend: each token maps to bytes; full decode is the
    /// concatenation, lossy-converted to String (like tokenizers::decode).
    struct ByteVocab(Vec<Vec<u8>>);
    impl ByteVocab {
        fn decode(&self, ids: &[u32]) -> Result<String, StreamDecodeError> {
            let bytes: Vec<u8> = ids.iter().filter_map(|i| self.0.get(*i as usize)).flatten().copied().collect();
            Ok(String::from_utf8_lossy(&bytes).into_owned())
        }
    }

    #[test]
    fn plain_ascii_streams_immediately() {
        let vocab = ByteVocab(vec![b"hel".to_vec(), b"lo ".to_vec(), b"world".to_vec()]);
        let mut d = StreamDecoder::new();
        let mut f = |ids: &[u32]| vocab.decode(ids);
        assert_eq!(d.push(0, &mut f).unwrap(), Some("hel".into()));
        assert_eq!(d.push(1, &mut f).unwrap(), Some("lo ".into()));
        assert_eq!(d.push(2, &mut f).unwrap(), Some("world".into()));
    }

    #[test]
    fn incomplete_multibyte_is_held_back() {
        // token 0 = first 2 bytes of "世" (E4 B8 96), token 1 = last byte
        let vocab = ByteVocab(vec![vec![0xE4, 0xB8], vec![0x96]]);
        let mut d = StreamDecoder::new();
        let mut f = |ids: &[u32]| vocab.decode(ids);
        assert_eq!(d.push(0, &mut f).unwrap(), None); // FFFD tail withheld
        assert_eq!(d.push(1, &mut f).unwrap(), Some("世".into()));
    }

    #[test]
    fn finish_flushes_replacement_char() {
        // Token maps to a lone invalid byte; finish() must release it.
        let vocab = ByteVocab(vec![vec![0xFF]]);
        let mut d = StreamDecoder::new();
        let mut f = |ids: &[u32]| vocab.decode(ids);
        assert_eq!(d.push(0, &mut f).unwrap(), None);
        assert_eq!(d.finish(&mut f).unwrap(), Some("\u{FFFD}".into()));
    }

    #[test]
    fn emoji_split_across_tokens() {
        // "🌏" (U+1F30F) = F0 9F 8C 8F split 3+1
        let vocab = ByteVocab(vec![vec![0xF0, 0x9F, 0x8C], vec![0x8F], " ok".as_bytes().to_vec()]);
        let mut d = StreamDecoder::new();
        let mut f = |ids: &[u32]| vocab.decode(ids);
        assert_eq!(d.push(0, &mut f).unwrap(), None);
        assert_eq!(d.push(1, &mut f).unwrap(), Some("🌏".into()));
        assert_eq!(d.push(2, &mut f).unwrap(), Some(" ok".into()));
        assert_eq!(d.token_count(), 3);
    }

    #[test]
    fn decode_error_propagates() {
        let mut d = StreamDecoder::new();
        let mut f = |_ids: &[u32]| Err(StreamDecodeError("boom".into()));
        assert!(d.push(0, &mut f).is_err());
    }
}
