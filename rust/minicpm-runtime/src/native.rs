//! Native backend: `dlopen`-based binding to `libminicpm_native.so`
//! (plan §6 FFI safety wrapper).
//!
//! * [`NativeRuntime::open`] loads the shared library and resolves **all**
//!   symbols up front, so a missing entry point fails at startup instead of
//!   mid-generation.
//! * [`Model`] is an immutable shared handle: `Send + Sync`, destroyed in
//!   `Drop` via `mc_model_destroy` (by-value handle).
//! * [`Session`] is an exclusive mutable object: `Send` (worker threads may
//!   move it) but **not** `Sync` — the ABI forbids concurrent calls on one
//!   session (plan §5.1). Destroyed in `Drop` via `mc_session_destroy`.
//! * Every non-OK `mc_status_t` is converted to [`RuntimeError`] by reading
//!   `mc_last_error()` immediately, while the thread-local message is still
//!   the one for the failing call.
//!
//! Safety summary for the `unsafe impl`s: the handles are plain opaque
//! pointers whose native side never stores references back into Rust; the
//! library outlives all handles because `NativeApi` (which owns the
//! `Library`) is held by an `Arc` cloned into every model and session.

use std::ffi::{CStr, CString};
use std::sync::Arc;

use minicpm_runtime_sys as sys;

use crate::{Engine, EngineSession, GenerationConfig, RuntimeError, RuntimeStats};

/// Resolved symbol table + the owning library handle. Auto `Send + Sync`
/// (`Library` and `unsafe extern "C" fn` pointers are).
struct NativeApi {
    #[allow(dead_code)] // keeps the dlopen handle alive; never read directly
    lib: libloading::Library,
    model_load: sys::McModelLoadFn,
    model_destroy: sys::McModelDestroyFn,
    session_create: sys::McSessionCreateFn,
    session_reset: sys::McSessionResetFn,
    session_destroy: sys::McSessionDestroyFn,
    prefill: sys::McPrefillFn,
    decode_one: sys::McDecodeOneFn,
    get_stats: sys::McGetStatsFn,
    last_error: sys::McLastErrorFn,
    batch_group_create: sys::McBatchGroupCreateFn,
    batch_group_add: sys::McBatchGroupAddFn,
    batch_group_remove: sys::McBatchGroupRemoveFn,
    batch_step: sys::McBatchStepFn,
    batch_group_destroy: sys::McBatchGroupDestroyFn,
}

impl NativeApi {
    /// Map a status code to `Result`, reading `mc_last_error()` on failure
    /// immediately (the message is thread-local and short-lived).
    fn check(&self, status: sys::McStatusT) -> Result<(), RuntimeError> {
        if status == sys::MC_OK {
            return Ok(());
        }
        let message = unsafe { (self.last_error)() };
        let message = if message.is_null() {
            String::from("<no error message>")
        } else {
            unsafe { CStr::from_ptr(message) }
                .to_string_lossy()
                .into_owned()
        };
        Err(RuntimeError { status, message })
    }
}

/// A loaded native runtime library (`dlopen`).
pub struct NativeRuntime {
    api: Arc<NativeApi>,
}

impl NativeRuntime {
    /// `dlopen` the native library and resolve all ABI v0.1 symbols.
    pub fn open(path: impl AsRef<std::path::Path>) -> anyhow::Result<Self> {
        let path = path.as_ref();
        unsafe {
            let lib = libloading::Library::new(path)
                .map_err(|e| anyhow::anyhow!("failed to dlopen `{}`: {e}", path.display()))?;
            let api = NativeApi {
                model_load: *lib.get(b"mc_model_load\0")?,
                model_destroy: *lib.get(b"mc_model_destroy\0")?,
                session_create: *lib.get(b"mc_session_create\0")?,
                session_reset: *lib.get(b"mc_session_reset\0")?,
                session_destroy: *lib.get(b"mc_session_destroy\0")?,
                prefill: *lib.get(b"mc_prefill\0")?,
                decode_one: *lib.get(b"mc_decode_one\0")?,
                get_stats: *lib.get(b"mc_get_stats\0")?,
                last_error: *lib.get(b"mc_last_error\0")?,
                batch_group_create: *lib.get(b"mc_batch_group_create\0")?,
                batch_group_add: *lib.get(b"mc_batch_group_add\0")?,
                batch_group_remove: *lib.get(b"mc_batch_group_remove\0")?,
                batch_step: *lib.get(b"mc_batch_step\0")?,
                batch_group_destroy: *lib.get(b"mc_batch_group_destroy\0")?,
                lib,
            };
            Ok(Self { api: Arc::new(api) })
        }
    }

    /// Load a model package. On failure the out-handle stays null and the
    /// `mc_last_error()` message is embedded in the returned error.
    pub fn load_model(
        &self,
        model_dir: &str,
        options: &sys::McModelOptionsT,
    ) -> anyhow::Result<Model> {
        let dir = CString::new(model_dir)
            .map_err(|e| anyhow::anyhow!("model dir contains interior NUL: {e}"))?;
        let mut handle = sys::McModelT::null();
        // SAFETY: valid CString + options pointer; out-handle initialized.
        let status = unsafe { (self.api.model_load)(dir.as_ptr(), options, &mut handle) };
        self.api.check(status).map_err(anyhow::Error::new)?;
        if handle.is_null() {
            return Err(anyhow::anyhow!(
                "mc_model_load returned MC_OK but a null model handle"
            ));
        }
        Ok(Model {
            inner: Arc::new(ModelInner {
                api: self.api.clone(),
                handle,
                // ctx 档随模型定档（mc_batch_group_create 的档位校验源）。
                max_context_tokens: options.max_context_tokens,
            }),
        })
    }
}

/// Convenience defaults matching the native smoke tests: device 0, one
/// session, 8192-token context, BF16 weights + KV, CUDA graphs disabled.
pub fn default_model_options() -> sys::McModelOptionsT {
    sys::McModelOptionsT {
        struct_size: std::mem::size_of::<sys::McModelOptionsT>() as u32,
        abi_version: sys::MC_ABI_VERSION,
        device_id: 0,
        max_sessions: 1,
        max_context_tokens: 8192,
        precision: sys::MC_BF16,
        kv_precision: sys::MC_KV_BF16,
        enable_cuda_graph: 0,
        reserve_vram_bytes: 0,
    }
}

/// Weight precision for [`model_options_for_precision`].
///
/// `Fp8` loads `model_fp8.wpk` when present (P4 v1: BF16 prefill weights +
/// FP8 decode weights, dual copy; the BF16 package stays on disk for
/// one-key rollback).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Precision {
    Bf16,
    Fp8,
}

impl Precision {
    /// Parses a CLI-friendly name (`"bf16"` / `"fp8"`).
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "bf16" => Some(Precision::Bf16),
            "fp8" => Some(Precision::Fp8),
            _ => None,
        }
    }

    fn as_native(self) -> sys::McPrecisionT {
        match self {
            Precision::Bf16 => sys::MC_BF16,
            Precision::Fp8 => sys::MC_FP8,
        }
    }
}

/// [`default_model_options`] with a chosen weight precision.
pub fn model_options_for_precision(precision: Precision) -> sys::McModelOptionsT {
    let mut o = default_model_options();
    o.precision = precision.as_native();
    o
}

/// Shared immutable model handle. Cloning shares ownership; the native model
/// is destroyed when the last clone drops.
struct ModelInner {
    api: Arc<NativeApi>,
    handle: sys::McModelT,
    /// 模型的 ctx 档（load 时的 `max_context_tokens`）——一个模型一档，
    /// 其全部 session 同档（`mc_session_create` 无 ctx 参数），故调度器侧
    /// 的批组「档位匹配」退化为恒真；`mc_batch_group_create` 以此档建组。
    #[allow(dead_code)] // BatchEngine::create_batch_group 读（native 档位口径）
    max_context_tokens: u32,
}

// SAFETY: the native model is immutable after `mc_model_load` and safe to
// share across threads (plan §5.1 "Model 可以只读共享"); destruction happens
// exactly once, in `Drop`, when the last Rust handle is gone.
unsafe impl Send for ModelInner {}
unsafe impl Sync for ModelInner {}

impl Drop for ModelInner {
    fn drop(&mut self) {
        // SAFETY: by-value handle owned exclusively here; dropping moves it.
        unsafe { (self.api.model_destroy)(self.handle) };
    }
}

/// A loaded model: cheaply cloneable shared handle.
#[derive(Clone)]
pub struct Model {
    inner: Arc<ModelInner>,
}

impl Model {
    /// 模型的 ctx 档（load 时的 `max_context_tokens`）。
    pub fn max_context_tokens(&self) -> u32 {
        self.inner.max_context_tokens
    }

    /// Create an exclusive generation session.
    pub fn create_session(&self) -> anyhow::Result<Session> {
        let mut handle = sys::McSessionT::null();
        // SAFETY: model handle stays alive for the call; out param initialized.
        let status = unsafe { (self.inner.api.session_create)(self.inner.handle, &mut handle) };
        self.inner.api.check(status).map_err(anyhow::Error::new)?;
        if handle.is_null() {
            return Err(anyhow::anyhow!(
                "mc_session_create returned MC_OK but a null session handle"
            ));
        }
        Ok(Session {
            api: self.inner.api.clone(),
            model: self.clone(), // keep the model alive at least as long as the session
            handle,
        })
    }
}

/// An exclusive generation session (`Send`, deliberately not `Sync`: raw
/// handle field keeps the auto-trait off).
pub struct Session {
    api: Arc<NativeApi>,
    #[allow(dead_code)] // ownership link, not read
    model: Model,
    handle: sys::McSessionT,
}

// SAFETY: a session is an exclusive mutable object; moving it to another
// thread is fine, concurrent access from several threads is not (and is
// prevented by not implementing `Sync`).
unsafe impl Send for Session {}

impl Session {
    /// Stage prompt token ids and run prefill (pinned staging + H2D happen
    /// natively, plan §5.4).
    pub fn prefill(&mut self, ids: &[i32]) -> Result<(), RuntimeError> {
        // SAFETY: exclusive &mut self; token pointer valid for the call.
        let status = unsafe { (self.api.prefill)(self.handle, ids.as_ptr(), ids.len() as u32) };
        self.api.check(status)
    }

    /// One decode step; returns the sampled token id.
    pub fn decode_one(&mut self, config: &GenerationConfig) -> Result<i32, RuntimeError> {
        let native = config.to_native();
        let mut token: i32 = 0;
        // SAFETY: exclusive &mut self; config/out pointers valid for the call.
        let status = unsafe { (self.api.decode_one)(self.handle, &native, &mut token) };
        self.api.check(status)?;
        Ok(token)
    }

    /// Reset sequence state, reusing all allocations (plan §5.4).
    pub fn reset(&mut self) -> Result<(), RuntimeError> {
        // SAFETY: exclusive &mut self.
        let status = unsafe { (self.api.session_reset)(self.handle) };
        self.api.check(status)
    }

    /// Lifetime-cumulative counters.
    pub fn stats(&mut self) -> Result<RuntimeStats, RuntimeError> {
        let mut out = sys::McRuntimeStatsT::default();
        // SAFETY: exclusive &mut self; out pointer valid for the call.
        let status = unsafe { (self.api.get_stats)(self.handle, &mut out) };
        self.api.check(status)?;
        Ok(RuntimeStats::from(out))
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        // SAFETY: by-value handle owned exclusively here.
        unsafe { (self.api.session_destroy)(self.handle) };
    }
}

/// [`Engine`] implementation over a loaded native model.
pub struct NativeEngine {
    model: Model,
}

impl NativeEngine {
    /// Load a model from a native library path + model directory.
    pub fn load(
        lib_path: impl AsRef<std::path::Path>,
        model_dir: &str,
        options: &sys::McModelOptionsT,
    ) -> anyhow::Result<Self> {
        let runtime = NativeRuntime::open(lib_path)?;
        let model = runtime.load_model(model_dir, options)?;
        Ok(Self { model })
    }

    /// Wrap an already loaded [`Model`].
    pub fn new(model: Model) -> Self {
        Self { model }
    }
}

impl Engine for NativeEngine {
    fn create_session(&self) -> anyhow::Result<Box<dyn EngineSession>> {
        Ok(Box::new(NativeSession {
            session: self.model.create_session()?,
        }))
    }

    fn as_batch(&self) -> Option<&dyn BatchEngine> {
        Some(self)
    }
}

/// [`EngineSession`] implementation over a native [`Session`].
pub struct NativeSession {
    session: Session,
}

impl EngineSession for NativeSession {
    fn prefill(&mut self, ids: &[i32]) -> anyhow::Result<()> {
        self.session.prefill(ids).map_err(anyhow::Error::new)
    }

    fn decode_one(&mut self, config: &GenerationConfig) -> anyhow::Result<i32> {
        self.session.decode_one(config).map_err(anyhow::Error::new)
    }

    fn reset(&mut self) -> anyhow::Result<()> {
        self.session.reset().map_err(anyhow::Error::new)
    }

    fn stats(&mut self) -> anyhow::Result<RuntimeStats> {
        self.session.stats().map_err(anyhow::Error::new)
    }

    fn as_any_mut(&mut self) -> Option<&mut dyn std::any::Any> {
        Some(self)
    }
}

// ============================================================================
// B3 batched decode：BatchEngine / BatchGroup trait + native FFI 包装
// ============================================================================
//
// ABI 契约（header `mc_batch_*` 注释的 Rust 侧镜像）：
// * 全部组调用来自**单线程**（调度器 GPU worker；不变量 ①）；
// * 成员在组内期间禁止 solo 使用（add 时校验忙态；remove 建立 组流→成员流
//   事件链后才能再 solo）——由调度器的 quantum 边界 sync_slots 保证；
// * session 在组内期间不得 destroy（retire 前必须 remove）；
// * `mc_batch_step` B1 仅 greedy（native 返回 MC_E_UNSUPPORTED；Rust 侧先
//   自查避免白调）。

/// 批组创建参数（调度器 → 引擎）。
///
/// `max_context_tokens == 0` 表示「跟随引擎（模型）自身档位」——调度器只
/// 持有一个引擎、一个模型一档（所有 session 同 ctx 档），档位匹配退化为
/// 恒真，因此无需显式传档。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BatchGroupOptions {
    /// 组容量上限，`1..=8`（native ABI 上限；native 侧 clamp）。
    pub max_slots: u32,
    /// 0 = 跟随引擎（模型）档；非 0 且与模型档不一致 → 创建拒绝。
    pub max_context_tokens: u32,
}

/// 支持批 decode 的引擎（[`Engine`] 的扩展能力，B3）。
///
/// [`MockEngine`] **不实现** 本 trait（`Engine::as_batch` 默认 None）——调度器
/// 对无批能力的引擎走全 solo 回退路径 A（构造性零回归）。
pub trait BatchEngine: Engine {
    /// 创建批组（组 stream + 组集中 arena；native `mc_batch_group_create`）。
    fn create_batch_group(&self, opts: BatchGroupOptions) -> anyhow::Result<Box<dyn BatchGroup>>;
}

/// 一个批组：成员 session 以 slot（0..max_slots-1）加入，保留各自 KV/序列
/// 状态；`step` 一次推进全部给定 slot 各一步（eager 精确 n，B1 语义）。
pub trait BatchGroup: Send {
    /// 成员入组（分配空闲 slot；session 忙态校验在 native 侧）。
    fn add(&mut self, session: &mut dyn EngineSession) -> anyhow::Result<u32>;

    /// 成员出组（quantum 边界调用：两次 step 之间）。
    fn remove(&mut self, slot: u32) -> anyhow::Result<()>;

    /// 批 decode 一步：`slots`（不可重复）与 `cfgs` 等长，返回逐成员发放
    /// token（发放语义同 decode_one 的预决策）。仅 greedy cfg。
    fn step(&mut self, slots: &[u32], cfgs: &[GenerationConfig]) -> anyhow::Result<Vec<i32>>;
}

impl BatchEngine for NativeEngine {
    fn create_batch_group(&self, opts: BatchGroupOptions) -> anyhow::Result<Box<dyn BatchGroup>> {
        let tier = self.model.max_context_tokens();
        if opts.max_context_tokens != 0 && opts.max_context_tokens != tier {
            anyhow::bail!(
                "mc_batch_group_create: max_context_tokens={} != model tier {}",
                opts.max_context_tokens,
                tier
            );
        }
        let options = sys::McBatchOptionsT {
            struct_size: std::mem::size_of::<sys::McBatchOptionsT>() as u32,
            abi_version: sys::MC_ABI_VERSION,
            max_slots: opts.max_slots.clamp(1, sys::MC_MAX_BATCH_SLOTS),
            max_context_tokens: tier,
            reserved: [0; 4],
        };
        let mut handle = sys::McBatchGroupT::null();
        // SAFETY: model handle alive for the call; options pointer valid;
        // out-handle initialized.
        let status = unsafe {
            (self.model.inner.api.batch_group_create)(
                self.model.inner.handle,
                &options,
                &mut handle,
            )
        };
        self.model
            .inner
            .api
            .check(status)
            .map_err(anyhow::Error::new)?;
        if handle.is_null() {
            anyhow::bail!("mc_batch_group_create returned MC_OK but a null group handle");
        }
        Ok(Box::new(NativeBatchGroup {
            api: self.model.inner.api.clone(),
            model: self.model.clone(),
            handle,
        }))
    }
}

/// [`BatchGroup`] 的 native FFI 包装。**单线程使用**（调度器 worker；
/// `Send` 是为了随 worker 状态移动，不是并发许可）。
pub struct NativeBatchGroup {
    api: Arc<NativeApi>,
    /// 所有权链接：组析构先于模型析构（session 在组内不得 destroy 的 stronger form）。
    #[allow(dead_code)] // 所有权链接（组与模型同生命周期），不读
    model: Model,
    handle: sys::McBatchGroupT,
}

// SAFETY: 组句柄是独占可变对象；跨线程移动可、并发访问不可（与 Session
// 同口径——不 impl Sync，原始指针字段天然 !Sync）。全部调用来自调度器
// 的单一 GPU worker 线程（不变量 ①）。
unsafe impl Send for NativeBatchGroup {}

impl NativeBatchGroup {
    /// 成员必须是本引擎的 [`NativeSession`]（`as_any_mut` downcast）。
    fn native_session_mut(session: &mut dyn EngineSession) -> anyhow::Result<&mut NativeSession> {
        session
            .as_any_mut()
            .and_then(|a| a.downcast_mut::<NativeSession>())
            .ok_or_else(|| anyhow::anyhow!("mc_batch_group_add: not a native session"))
    }
}

impl BatchGroup for NativeBatchGroup {
    fn add(&mut self, session: &mut dyn EngineSession) -> anyhow::Result<u32> {
        let native = Self::native_session_mut(session)?;
        let mut slot: u32 = 0;
        // SAFETY: 组与 session 句柄均独占有效；out 指针有效。
        let status =
            unsafe { (self.api.batch_group_add)(self.handle, native.session.handle, &mut slot) };
        self.api.check(status).map_err(anyhow::Error::new)?;
        Ok(slot)
    }

    fn remove(&mut self, slot: u32) -> anyhow::Result<()> {
        // SAFETY: 组句柄独占有效。
        let status = unsafe { (self.api.batch_group_remove)(self.handle, slot) };
        self.api.check(status).map_err(anyhow::Error::new)
    }

    fn step(&mut self, slots: &[u32], cfgs: &[GenerationConfig]) -> anyhow::Result<Vec<i32>> {
        anyhow::ensure!(
            !slots.is_empty() && slots.len() == cfgs.len(),
            "mc_batch_step: slots/cfgs must be non-empty and equal length"
        );
        // B1 greedy-only：Rust 侧先自查（native 同判据会拒 MC_E_UNSUPPORTED），
        // 避免白调一次 FFI。
        for (i, cfg) in cfgs.iter().enumerate() {
            anyhow::ensure!(
                cfg.is_greedy(),
                "mc_batch_step: non-greedy cfgs[{i}] unsupported in B1 \
                 (temperature>0 && top_k!=1); batch path is greedy-only"
            );
        }
        let native_cfgs: Vec<_> = cfgs.iter().map(GenerationConfig::to_native).collect();
        let mut out_tokens: Vec<i32> = vec![0; slots.len()];
        // SAFETY: 指针/长度对调用期间有效；slots 不可重复由 native 校验。
        let status = unsafe {
            (self.api.batch_step)(
                self.handle,
                slots.as_ptr(),
                slots.len() as u32,
                native_cfgs.as_ptr(),
                out_tokens.as_mut_ptr(),
            )
        };
        self.api.check(status).map_err(anyhow::Error::new)?;
        Ok(out_tokens)
    }
}

impl Drop for NativeBatchGroup {
    fn drop(&mut self) {
        // SAFETY: by-value handle owned exclusively here；仍在组的成员自动
        // 经历 remove 语义（正常路径下调度器已逐个 remove）。
        unsafe { (self.api.batch_group_destroy)(self.handle) };
    }
}

// Compile-time checks of the thread-safety contract (plan §6).
const _: () = {
    const fn assert_send<T: Send>() {}
    const fn assert_sync<T: Sync>() {}
    assert_send::<NativeRuntime>();
    assert_sync::<NativeRuntime>();
    assert_send::<Model>();
    assert_sync::<Model>();
    assert_send::<Session>();
    assert_send::<NativeBatchGroup>();
    // Session must NOT be Sync: no assert_sync::<Session>() here. The raw
    // pointer field keeps `!Sync` automatic; if that ever changes, the
    // scheduler's single-worker assumption breaks — see tests below.
    // NativeBatchGroup 同理（原始句柄字段保持 !Sync）。
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_options_defaults_are_abi_consistent() {
        let o = default_model_options();
        assert_eq!(
            o.struct_size as usize,
            std::mem::size_of::<sys::McModelOptionsT>()
        );
        assert_eq!(o.abi_version, sys::MC_ABI_VERSION);
    }

    #[test]
    fn open_missing_library_fails_cleanly() {
        let err = match NativeRuntime::open("/nonexistent/libminicpm_native.so") {
            Err(e) => e,
            Ok(_) => panic!("opening a nonexistent library must fail"),
        };
        assert!(
            format!("{err}").contains("dlopen"),
            "unexpected error: {err}"
        );
    }
}
