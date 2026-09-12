//! Stop-condition state machine (plan §4.1: Stop Token / Stop String).
//!
//! Semantics follow the OpenAI Chat Completions API:
//! - stop token id: generation stops before the token is emitted.
//! - stop string: generation stops when the string appears in the generated
//!   text; the stop string itself (and any bytes belonging to a potential
//!   future match) is never shown to the client.
//!
//! To guarantee the last point while streaming, [`StopState::safe_len`]
//! withholds the trailing `max_pattern_len - 1` bytes: any stop-string match
//! that may still complete later can overlap already-arrived text by at most
//! that many bytes.

/// What terminated (or should terminate) the generation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StopHit {
    /// A stop / eos token id matched.
    Token(i32),
    /// A stop string matched; `start` is its byte offset in generated text.
    String { start: usize },
}

#[derive(Debug)]
pub struct StopState {
    stop_token_ids: Box<[i32]>,
    stop_strings: Box<[String]>,
    /// Longest stop string, in bytes (UTF-8).
    max_pattern_len: usize,
    /// All stable generated text (before stop-string removal).
    text: String,
    /// Byte range of the matched stop string, once found.
    matched: Option<usize>,
    stopped: bool,
}

impl StopState {
    pub fn new(stop_token_ids: impl IntoIterator<Item = i32>, stop_strings: impl IntoIterator<Item = String>) -> Self {
        let stop_strings: Box<[String]> = stop_strings.into_iter().collect();
        let max_pattern_len = stop_strings.iter().map(|s| s.len()).max().unwrap_or(0);
        Self {
            stop_token_ids: stop_token_ids.into_iter().collect(),
            stop_strings,
            max_pattern_len,
            text: String::new(),
            matched: None,
            stopped: false,
        }
    }

    pub fn is_stopped(&self) -> bool {
        self.stopped
    }

    /// Check a newly generated token id. Returns `Some(StopHit::Token)` when
    /// the token is a stop token; callers must not emit its text.
    pub fn check_token(&mut self, id: i32) -> Option<StopHit> {
        if self.stopped {
            return Some(StopHit::Token(id)); // already stopped; keep reporting
        }
        if self.stop_token_ids.contains(&id) {
            self.stopped = true;
            return Some(StopHit::Token(id));
        }
        None
    }

    /// Feed a decoded text fragment and check for stop-string matches.
    /// Must be called for every fragment, including the final flush.
    pub fn push_text(&mut self, fragment: &str) -> Option<StopHit> {
        if self.stopped || fragment.is_empty() {
            return self.matched.map(|start| StopHit::String { start });
        }
        let search_from = self.text.len().saturating_sub(self.max_pattern_len);
        self.text.push_str(fragment);
        if let Some(start) = self.find_match(search_from) {
            self.matched = Some(start);
            self.stopped = true;
            return Some(StopHit::String { start });
        }
        None
    }

    fn find_match(&self, from: usize) -> Option<usize> {
        let hay = &self.text[from..];
        let mut best: Option<usize> = None;
        for pat in &self.stop_strings {
            if let Some(rel) = hay.find(pat.as_str()) {
                let abs = from + rel;
                if best.is_none_or(|b| abs < b) {
                    best = Some(abs);
                }
            }
        }
        best
    }

    /// Byte length of generated text that is safe to emit to the client now.
    ///
    /// Withholds the trailing `max_pattern_len - 1` bytes until
    /// [`StopState::finish`], and truncates at the matched stop string.
    pub fn safe_len(&self) -> usize {
        if let Some(start) = self.matched {
            return start;
        }
        let holdback = self.max_pattern_len.saturating_sub(1);
        let mut len = self.text.len().saturating_sub(holdback);
        while !self.text.is_char_boundary(len) {
            len -= 1;
        }
        len
    }

    /// Final call: full byte length visible to the client (stop string still
    /// excluded; everything withheld during streaming is now releasable
    /// unless it belongs to the matched stop string).
    pub fn finish(&mut self) -> usize {
        if let Some(start) = self.matched {
            return start;
        }
        // A match could still be pending inside the withheld window only if a
        // stop string occurs there; check once more over the whole text.
        if let Some(start) = self.find_match(0) {
            self.matched = Some(start);
            self.stopped = true;
            return start;
        }
        self.text.len()
    }

    /// Generated text visible to the client for `len` bytes.
    pub fn visible_text(&self, len: usize) -> &str {
        &self.text[..len]
    }

    /// Full generated text (including any stop string). For logging/tests.
    pub fn full_text(&self) -> &str {
        &self.text
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stop_token_hits_once() {
        let mut s = StopState::new([1, 130073], []);
        assert!(s.check_token(42).is_none());
        assert_eq!(s.check_token(130073), Some(StopHit::Token(130073)));
        assert!(s.is_stopped());
        // subsequent tokens keep reporting stopped
        assert!(s.check_token(7).is_some());
    }

    #[test]
    fn stop_string_within_fragment() {
        let mut s = StopState::new([], ["STOP".to_string()]);
        s.push_text("hello ");
        // holdback = len("STOP") - 1 = 3 bytes withheld until finish
        assert_eq!(s.safe_len(), 3);
        assert_eq!(s.visible_text(3), "hel");
        let hit = s.push_text("worldSTOPtail");
        assert_eq!(hit, Some(StopHit::String { start: 11 }));
        assert_eq!(s.finish(), 11);
        assert_eq!(s.visible_text(11), "hello world");
    }

    #[test]
    fn stop_string_across_token_boundary() {
        let mut s = StopState::new([], ["</answer>".to_string()]);
        s.push_text("the answer is </ans");
        // holdback = len("</answer>") - 1 = 8 of 19 bytes withheld
        assert_eq!(s.safe_len(), "the answer ".len());
        let hit = s.push_text("wer> done");
        assert!(hit.is_some());
        let n = s.finish();
        assert_eq!(s.visible_text(n), "the answer is ");
    }

    #[test]
    fn no_stop_strings_emits_everything() {
        let mut s = StopState::new([], Vec::<String>::new());
        s.push_text("你好，世界 🌏");
        assert_eq!(s.safe_len(), "你好，世界 🌏".len());
        assert_eq!(s.finish(), "你好，世界 🌏".len());
    }

    #[test]
    fn earliest_match_wins() {
        let mut s = StopState::new([], ["bbbb".to_string(), "ab".to_string()]);
        let hit = s.push_text("xxabbbbq");
        assert_eq!(hit, Some(StopHit::String { start: 2 }));
        assert_eq!(s.finish(), 2);
    }

    #[test]
    fn withheld_window_never_leaks_partial_stop() {
        // pattern longer than first fragment; match completes much later
        let mut s = StopState::new([], ["<|end_of_tool|>".to_string()]);
        s.push_text("partial <");
        // text (9 bytes) entirely inside the 14-byte holdback window
        assert_eq!(s.safe_len(), 0);
        s.push_text("|end_o");
        // text (15 bytes) releases 1 byte
        assert_eq!(s.safe_len(), 1);
        s.push_text("f_tool|> rest");
        assert!(s.is_stopped());
        assert_eq!(s.finish(), "partial ".len());
    }
}
