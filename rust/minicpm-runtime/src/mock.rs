//! In-process mock backend (no GPU / no native library).
//!
//! Two modes:
//! * [`MockEngine::with_script`] — replays a fixed token script; once the
//!   script is exhausted, `decode_one` returns the `<|im_end|>` eos id
//!   (130073), like the native mock runtime does at `max_new_tokens`.
//! * [`MockEngine::new()` (default)] — a deterministic LCG producing token
//!   ids in `[100, 1100)`, mirroring the native mock forward
//!   (`token = 100 + ((seq_len * 2654435761) ^ rng) % 1000`).
//!
//! A per-decode delay can be attached for pacing cancellation tests.
//! [`MockStats`] is shared between engine and sessions so tests can observe
//! session creation / prefill / decode / reset counts from the outside
//! (the scheduler owns the sessions, tests only hold the engine).

use std::sync::{Arc, Mutex};
use std::time::Duration;

use crate::{Engine, EngineSession, GenerationConfig, RuntimeStats};

/// `<|im_end|>`: the eos id the mock emits when its script is exhausted
/// (matches the native mock runtime and `EOS_IDS` in minicpm-scheduler).
pub const MOCK_EOS: i32 = 130_073;

/// LCG parameters (Numerical Recipes). Seed offset gives a spread sequence.
const LCG_MULT: u32 = 1_664_525;
const LCG_INC: u32 = 1_013_904_223;

/// Observable call counters shared across all sessions of one engine.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct MockStats {
    pub sessions_created: usize,
    pub prefill_calls: usize,
    pub prefill_tokens: usize,
    pub decode_calls: usize,
    pub reset_calls: usize,
}

#[derive(Debug, Default)]
struct Shared {
    stats: MockStats,
}

/// Mock [`Engine`].
pub struct MockEngine {
    script: Option<Vec<i32>>,
    decode_delay: Option<Duration>,
    shared: Arc<Mutex<Shared>>,
}

impl MockEngine {
    /// LCG pseudo-random engine (token ∈ `[100, 1100)`).
    pub fn new() -> Self {
        Self {
            script: None,
            decode_delay: None,
            shared: Arc::new(Mutex::new(Shared::default())),
        }
    }

    /// Scripted engine: replays `script`, then returns [`MOCK_EOS`].
    pub fn with_script(script: Vec<i32>) -> Self {
        Self {
            script: Some(script),
            ..Self::new()
        }
    }

    /// Pace every `decode_one` call with a sleep (test cancellation timing).
    pub fn with_decode_delay(mut self, delay: Duration) -> Self {
        self.decode_delay = Some(delay);
        self
    }

    /// Snapshot of the shared call counters.
    pub fn stats(&self) -> MockStats {
        self.shared.lock().unwrap().stats.clone()
    }
}

impl Default for MockEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl Engine for MockEngine {
    fn create_session(&self) -> anyhow::Result<Box<dyn EngineSession>> {
        self.shared.lock().unwrap().stats.sessions_created += 1;
        Ok(Box::new(MockSession {
            script: self.script.clone(),
            pos: 0,
            lcg: 0x9E37_79B9,
            prompt_len: 0,
            generated: 0,
            decode_delay: self.decode_delay,
            shared: Arc::clone(&self.shared),
        }))
    }
}

/// Mock [`EngineSession`]: records counters, replays a script or runs the LCG.
pub struct MockSession {
    script: Option<Vec<i32>>,
    pos: usize,
    lcg: u32,
    prompt_len: usize,
    generated: u64,
    decode_delay: Option<Duration>,
    shared: Arc<Mutex<Shared>>,
}

impl MockSession {
    /// Tokens generated since the last reset (excludes eos from the script).
    pub fn generated(&self) -> u64 {
        self.generated
    }
}

impl EngineSession for MockSession {
    fn prefill(&mut self, ids: &[i32]) -> anyhow::Result<()> {
        {
            let mut s = self.shared.lock().unwrap();
            s.stats.prefill_calls += 1;
            s.stats.prefill_tokens += ids.len();
        }
        self.prompt_len = ids.len();
        Ok(())
    }

    fn decode_one(&mut self, _config: &GenerationConfig) -> anyhow::Result<i32> {
        if let Some(delay) = self.decode_delay {
            std::thread::sleep(delay);
        }
        {
            let mut s = self.shared.lock().unwrap();
            s.stats.decode_calls += 1;
        }
        self.generated += 1;
        let token = match &self.script {
            Some(script) => {
                if self.pos < script.len() {
                    let t = script[self.pos];
                    self.pos += 1;
                    t
                } else {
                    MOCK_EOS
                }
            }
            None => {
                // Deterministic LCG in [100, 1100).
                self.lcg = self.lcg.wrapping_mul(LCG_MULT).wrapping_add(LCG_INC);
                100 + (self.lcg % 1000) as i32
            }
        };
        Ok(token)
    }

    fn reset(&mut self) -> anyhow::Result<()> {
        {
            let mut s = self.shared.lock().unwrap();
            s.stats.reset_calls += 1;
        }
        // Replay the script from the start on the next generation round,
        // mirroring the native session_reset semantics (fresh sequence state,
        // same allocations/config).
        self.pos = 0;
        self.prompt_len = 0;
        self.generated = 0;
        Ok(())
    }

    fn stats(&mut self) -> anyhow::Result<RuntimeStats> {
        Ok(RuntimeStats {
            prompt_tokens: self.prompt_len as u64,
            generated_tokens: self.generated,
            prefill_us: 0,
            decode_us: 0,
            peak_vram_bytes: 0,
            cuda_graph_launches: 0,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scripted_session_replays_then_eos() {
        let engine = MockEngine::with_script(vec![101, 102, 103]);
        let mut s = engine.create_session().unwrap();
        s.prefill(&[0, 100, 200]).unwrap();
        assert_eq!(s.decode_one(&GenerationConfig::default()).unwrap(), 101);
        assert_eq!(s.decode_one(&GenerationConfig::default()).unwrap(), 102);
        assert_eq!(s.decode_one(&GenerationConfig::default()).unwrap(), 103);
        assert_eq!(
            s.decode_one(&GenerationConfig::default()).unwrap(),
            MOCK_EOS
        );
        assert_eq!(
            s.decode_one(&GenerationConfig::default()).unwrap(),
            MOCK_EOS
        );

        let st = s.stats().unwrap();
        assert_eq!(st.prompt_tokens, 3);
        assert_eq!(st.generated_tokens, 5);

        // Reset replays the script.
        s.reset().unwrap();
        assert_eq!(s.decode_one(&GenerationConfig::default()).unwrap(), 101);
    }

    #[test]
    fn lcg_session_stays_in_range() {
        let engine = MockEngine::new();
        let mut s = engine.create_session().unwrap();
        s.prefill(&[0]).unwrap();
        for _ in 0..100 {
            let t = s.decode_one(&GenerationConfig::default()).unwrap();
            assert!((100..1100).contains(&t), "token {t} out of [100,1100)");
        }
    }

    #[test]
    fn engine_stats_count_calls() {
        let engine = MockEngine::with_script(vec![7]);
        let mut s = engine.create_session().unwrap();
        s.prefill(&[1, 2, 3]).unwrap();
        s.decode_one(&GenerationConfig::default()).unwrap();
        s.reset().unwrap();
        let st = engine.stats();
        assert_eq!(st.sessions_created, 1);
        assert_eq!(st.prefill_calls, 1);
        assert_eq!(st.prefill_tokens, 3);
        assert_eq!(st.decode_calls, 1);
        assert_eq!(st.reset_calls, 1);
    }

    #[test]
    fn mock_engine_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<MockEngine>();
        fn assert_send<T: Send>() {}
        assert_send::<MockSession>();
    }
}
