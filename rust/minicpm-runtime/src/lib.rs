//! `minicpm-runtime`: safe Rust wrapper over the MiniCPM5 native GPU runtime
//! C ABI (plan §6) plus an in-process mock backend.
//!
//! Layering:
//! * [`minicpm_runtime_sys`] keeps the raw `unsafe extern "C"` declarations.
//! * This crate owns RAII, error mapping, lifetime and thread-safety
//!   constraints:
//!   - `Model` is immutable after load and shareable (`Send + Sync`),
//!   - `Session` is an exclusive mutable generation object: `Send` (one
//!     thread at a time) but deliberately **not** `Sync` (plan §5.1/§6).
//! * [`Engine`] / [`EngineSession`] abstract over the native backend and the
//!   [`MockEngine`] used by scheduler/server tests and mock deployments.

use std::fmt;

use minicpm_runtime_sys::McGenerationConfigT;

pub mod mock;
pub mod native;

pub use mock::{MockEngine, MockSession, MockStats};
pub use native::{
    default_model_options, model_options_for_precision, BatchEngine, BatchGroup, BatchGroupOptions,
    Model, NativeBatchGroup, NativeEngine, NativeRuntime, NativeSession, Precision, Session,
};

/// Sampling / stopping parameters for one generation job.
///
/// Mirrors `mc_generation_config_t` (plan §5.3) minus the ABI bookkeeping
/// fields, which [`GenerationConfig::to_native`] fills in.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GenerationConfig {
    pub max_new_tokens: u32,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: u32,
    pub repetition_penalty: f32,
    pub seed: u64,
}

impl GenerationConfig {
    /// Convert to the native ABI struct. `struct_size` is filled from
    /// `size_of::<McGenerationConfigT>()` as required by the contract.
    pub fn to_native(&self) -> McGenerationConfigT {
        McGenerationConfigT {
            struct_size: std::mem::size_of::<McGenerationConfigT>() as u32,
            max_new_tokens: self.max_new_tokens,
            temperature: self.temperature,
            top_p: self.top_p,
            top_k: self.top_k,
            repetition_penalty: self.repetition_penalty,
            seed: self.seed,
        }
    }

    /// greedy 判定（B3 批准入批条件；与 native 激活规则镜像——
    /// `decode_executor.cpp`：「temperature ≤ 0 或 top_k == 1 → greedy」）。
    /// 非 greedy（temperature > 0 且 top_k != 1）在 B1 会被
    /// `mc_batch_step` 拒绝（MC_E_UNSUPPORTED），Rust 侧先行自查避免白调。
    pub fn is_greedy(&self) -> bool {
        self.temperature <= 0.0 || self.top_k == 1
    }
}

impl Default for GenerationConfig {
    /// Defaults follow the model's `generation_config.json`
    /// (`temperature = 1.0`, `top_p = 0.95`); `top_k = 0` (disabled),
    /// `repetition_penalty = 1.0` (neutral), `seed = 0` (native default).
    fn default() -> Self {
        Self {
            max_new_tokens: 512,
            temperature: 1.0,
            top_p: 0.95,
            top_k: 0,
            repetition_penalty: 1.0,
            seed: 0,
        }
    }
}

/// Mirror of `mc_runtime_stats_t` (plain counters).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RuntimeStats {
    pub prompt_tokens: u64,
    pub generated_tokens: u64,
    pub prefill_us: u64,
    pub decode_us: u64,
    pub peak_vram_bytes: u64,
    pub cuda_graph_launches: u64,
}

impl From<minicpm_runtime_sys::McRuntimeStatsT> for RuntimeStats {
    fn from(s: minicpm_runtime_sys::McRuntimeStatsT) -> Self {
        Self {
            prompt_tokens: s.prompt_tokens,
            generated_tokens: s.generated_tokens,
            prefill_us: s.prefill_us,
            decode_us: s.decode_us,
            peak_vram_bytes: s.peak_vram_bytes,
            cuda_graph_launches: s.cuda_graph_launches,
        }
    }
}

/// An `mc_status_t` failure together with the thread-local message read from
/// `mc_last_error()` immediately after the failing call (the message is
/// only valid until the next native call, so it is copied up front).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimeError {
    /// Raw `mc_status_t` value (see `minicpm_runtime_sys` constants).
    pub status: u32,
    /// Best-effort human readable message from `mc_last_error()`.
    pub message: String,
}

impl RuntimeError {
    /// Symbolic name of the status code, mirroring the header enum.
    pub fn status_name(&self) -> &'static str {
        status_name(self.status)
    }
}

impl fmt::Display for RuntimeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "native runtime error {} ({}): {}",
            self.status,
            self.status_name(),
            self.message
        )
    }
}

impl std::error::Error for RuntimeError {}

/// Symbolic names for `mc_status_t` values (header enum mirror).
pub fn status_name(status: u32) -> &'static str {
    use minicpm_runtime_sys as sys;
    match status {
        sys::MC_OK => "MC_OK",
        sys::MC_E_INVALID_ARGUMENT => "MC_E_INVALID_ARGUMENT",
        sys::MC_E_IO => "MC_E_IO",
        sys::MC_E_CUDA => "MC_E_CUDA",
        sys::MC_E_MODEL_MISMATCH => "MC_E_MODEL_MISMATCH",
        sys::MC_E_OUT_OF_MEMORY => "MC_E_OUT_OF_MEMORY",
        sys::MC_E_GRAPH => "MC_E_GRAPH",
        sys::MC_E_UNSUPPORTED => "MC_E_UNSUPPORTED",
        sys::MC_E_INTERNAL => "MC_E_INTERNAL",
        _ => "MC_E_UNKNOWN",
    }
}

/// A loadable model backend: hands out exclusive generation sessions.
///
/// `Send + Sync` so the scheduler can hold one behind an `Arc` and create
/// sessions from its worker thread (plan §4.5).
pub trait Engine: Send + Sync {
    fn create_session(&self) -> anyhow::Result<Box<dyn EngineSession>>;

    /// B3 批 decode 能力协商：支持批的引擎（[`native::NativeEngine`]）覆写
    /// 返回 `Some(self)`；默认 `None`（[`MockEngine`] 不覆写 → 调度器走
    /// 全 solo 回退路径 A，构造性零回归）。
    fn as_batch(&self) -> Option<&dyn native::BatchEngine> {
        None
    }
}

/// One exclusive generation session (plan §6: "a session is an exclusive
/// mutable generation-state object"). All methods take `&mut self`; the
/// implementor guarantees `Send` (single-threaded use, possibly on different
/// threads over its lifetime) and must **not** be `Sync`.
pub trait EngineSession: Send {
    /// Stage prompt tokens (`[0, vocab_size)`) and run prefill.
    fn prefill(&mut self, ids: &[i32]) -> anyhow::Result<()>;

    /// One decode step; returns the sampled token id.
    fn decode_one(&mut self, config: &GenerationConfig) -> anyhow::Result<i32>;

    /// Clear sequence state / KV ownership / RNG, reusing all allocations.
    fn reset(&mut self) -> anyhow::Result<()>;

    /// Lifetime-cumulative counters for this session.
    fn stats(&mut self) -> anyhow::Result<RuntimeStats>;

    /// Downcast hook（B3 批组 FFI 用）：native 批组 add 成员需要拿到底层
    /// [`native::NativeSession`] 的原生句柄；具体后端覆写返回 `Some(self)`，
    /// 默认 `None`（MockSession 等非 native 会话 → add 报「非 native session」）。
    fn as_any_mut(&mut self) -> Option<&mut dyn std::any::Any> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generation_config_to_native_fills_struct_size() {
        let cfg = GenerationConfig {
            max_new_tokens: 32,
            temperature: 0.8,
            top_p: 0.95,
            top_k: 0,
            repetition_penalty: 1.0,
            seed: 42,
        };
        let n = cfg.to_native();
        assert_eq!(
            n.struct_size as usize,
            std::mem::size_of::<McGenerationConfigT>()
        );
        assert_eq!(n.max_new_tokens, 32);
        assert_eq!(n.temperature, 0.8);
        assert_eq!(n.top_p, 0.95);
        assert_eq!(n.top_k, 0);
        assert_eq!(n.repetition_penalty, 1.0);
        assert_eq!(n.seed, 42);
    }

    #[test]
    fn greedy_criterion_mirrors_native() {
        // native 激活规则：temperature ≤ 0 或 top_k == 1 → greedy。
        let g = |t: f32, k: u32| GenerationConfig {
            temperature: t,
            top_k: k,
            ..GenerationConfig::default()
        };
        assert!(
            g(0.0, 0).is_greedy(),
            "temperature=0 → greedy（top_k 无关）"
        );
        assert!(g(-1.0, 0).is_greedy());
        assert!(
            g(1.0, 1).is_greedy(),
            "top_k=1 → greedy（temperature 无关）"
        );
        assert!(!g(1.0, 0).is_greedy(), "temperature>0 且 top_k!=1 → 采样");
        assert!(!g(0.8, 50).is_greedy());
        // 默认配置（temperature=1.0, top_k=0）是采样 → B3 下走 solo。
        assert!(!GenerationConfig::default().is_greedy());
    }

    #[test]
    fn sys_layout_assertions_hold() {
        // Cross-check the mirrored ABI sizes this crate relies on.
        assert_eq!(std::mem::size_of::<minicpm_runtime_sys::McModelT>(), 8);
        assert_eq!(std::mem::size_of::<minicpm_runtime_sys::McSessionT>(), 8);
        assert_eq!(std::mem::size_of::<McGenerationConfigT>(), 32);
        assert_eq!(
            std::mem::size_of::<minicpm_runtime_sys::McModelOptionsT>(),
            40
        );
    }

    #[test]
    fn runtime_error_display() {
        let e = RuntimeError {
            status: 3,
            message: "cuda oops".into(),
        };
        assert_eq!(e.status_name(), "MC_E_CUDA");
        let d = format!("{e}");
        assert!(d.contains("3"), "display shows status code: {d}");
        assert!(d.contains("MC_E_CUDA"), "display shows status name: {d}");
        assert!(d.contains("cuda oops"), "display shows message: {d}");
    }

    #[test]
    fn runtime_stats_mirror() {
        let s = minicpm_runtime_sys::McRuntimeStatsT {
            prompt_tokens: 3,
            generated_tokens: 32,
            prefill_us: 10,
            decode_us: 20,
            peak_vram_bytes: 30,
            cuda_graph_launches: 0,
        };
        let r = RuntimeStats::from(s);
        assert_eq!(
            r,
            RuntimeStats {
                prompt_tokens: 3,
                generated_tokens: 32,
                prefill_us: 10,
                decode_us: 20,
                peak_vram_bytes: 30,
                cuda_graph_launches: 0
            }
        );
    }
}
