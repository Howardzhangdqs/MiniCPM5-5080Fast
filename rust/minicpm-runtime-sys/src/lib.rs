//! `minicpm-runtime-sys`: raw C ABI declarations for the MiniCPM5 native GPU
//! runtime (plan §5, header `native/include/minicpm_runtime.h`, ABI v0.1).
//!
//! This crate contains *declarations only* — no business logic, no RAII, no
//! error mapping (plan §6: that is `minicpm-runtime`'s job). It mirrors the
//! header 1:1 so that layout drift is caught by the compile-time and unit
//! assertions at the bottom of this file.
//!
//! Layout notes (x86_64 SysV):
//! * `mc_model_t` / `mc_session_t` are **by-value** 8-byte single-pointer
//!   Pimpl handles (`native/runtime/internal_state.h` defines the C++ side as
//!   `struct mc_model { ModelImpl* impl; }`). Rust therefore models them as
//!   `#[repr(C)]` structs wrapping one `*mut c_void`, never as bare pointers,
//!   and passes them by value across the ABI.
//! * C enums (`mc_status_t`, `mc_precision_t`, ...) are 4-byte ints on every
//!   ABI we target; they are mirrored as `u32` type aliases plus constants.
//! * The 11 function pointer types are what `minicpm-runtime` resolves via
//!   `dlopen` (9 functions from the v0.1 header + the 2 optional async
//!   enqueue/poll entry points reserved by plan §5.5, not yet exported by the
//!   current `libminicpm_native.so`).

use std::ffi::c_void;

/// ABI version negotiated by `mc_model_load` (header: `0x00020000u`).
pub const MC_ABI_VERSION: u32 = 0x0002_0000;

// ---- status codes (mc_status_t) ----

/// C `mc_status_t`: enum → `u32` on the ABIs we target.
pub type McStatusT = u32;

pub const MC_OK: McStatusT = 0;
pub const MC_E_INVALID_ARGUMENT: McStatusT = 1;
pub const MC_E_IO: McStatusT = 2;
pub const MC_E_CUDA: McStatusT = 3;
pub const MC_E_MODEL_MISMATCH: McStatusT = 4;
pub const MC_E_OUT_OF_MEMORY: McStatusT = 5;
pub const MC_E_GRAPH: McStatusT = 6;
pub const MC_E_UNSUPPORTED: McStatusT = 7;
pub const MC_E_INTERNAL: McStatusT = 100;

// ---- precision enums ----

/// C `mc_precision_t`.
pub type McPrecisionT = u32;
pub const MC_BF16: McPrecisionT = 0;
pub const MC_FP16: McPrecisionT = 1;
pub const MC_FP8: McPrecisionT = 2;
pub const MC_NVFP4: McPrecisionT = 3;

/// C `mc_kv_precision_t`.
pub type McKvPrecisionT = u32;
pub const MC_KV_BF16: McKvPrecisionT = 0;
pub const MC_KV_FP8: McKvPrecisionT = 1;

// ---- opaque by-value handles ----

/// Opaque model handle. **By-value** across the ABI: a single `ModelImpl*`
/// Pimpl pointer (`sizeof == 8`). Not a bare Rust pointer — the C prototype
/// is `mc_model_t` (struct, one register) on both sides.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct McModelT {
    _p: *mut c_void,
}

impl McModelT {
    /// A null handle, the value `*out_model` holds before a successful load.
    pub const fn null() -> Self {
        Self {
            _p: std::ptr::null_mut(),
        }
    }

    /// True when the handle carries no implementation pointer.
    pub const fn is_null(&self) -> bool {
        self._p.is_null()
    }
}

/// Opaque session handle (by-value single `SessionImpl*`, `sizeof == 8`).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct McSessionT {
    _p: *mut c_void,
}

impl McSessionT {
    /// A null handle, the value `*out_session` holds before creation.
    pub const fn null() -> Self {
        Self {
            _p: std::ptr::null_mut(),
        }
    }

    /// True when the handle carries no implementation pointer.
    pub const fn is_null(&self) -> bool {
        self._p.is_null()
    }
}

// ---- evolving structs (struct_size versioned) ----

/// `mc_model_options_t` — `sizeof == 40` on x86_64; `struct_size` is checked
/// at call time by the native runtime.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct McModelOptionsT {
    /// Caller must set to `size_of::<McModelOptionsT>()`.
    pub struct_size: u32,
    /// Caller must set to [`MC_ABI_VERSION`].
    pub abi_version: u32,
    pub device_id: i32,
    pub max_sessions: u32,
    pub max_context_tokens: u32,
    pub precision: McPrecisionT,
    pub kv_precision: McKvPrecisionT,
    pub enable_cuda_graph: u32,
    pub reserve_vram_bytes: u64,
}

/// `mc_generation_config_t` — `sizeof == 32` on x86_64.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct McGenerationConfigT {
    /// Caller must set to `size_of::<McGenerationConfigT>()`.
    pub struct_size: u32,
    pub max_new_tokens: u32,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: u32,
    pub repetition_penalty: f32,
    pub seed: u64,
}

/// `mc_runtime_stats_t` — plain POD counters, `sizeof == 48` on x86_64.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct McRuntimeStatsT {
    pub prompt_tokens: u64,
    pub generated_tokens: u64,
    pub prefill_us: u64,
    pub decode_us: u64,
    pub peak_vram_bytes: u64,
    pub cuda_graph_launches: u64,
}

// ---- batched decode（B1b：mc_batch_* 五函数，header ABI v0.2 段）----

/// Opaque batch group handle (by-value single `BatchGroupImpl*`,
/// `sizeof == 8`)——与 [`McModelT`]/[`McSessionT`] 同款 Pimpl 建模。
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct McBatchGroupT {
    _p: *mut c_void,
}

impl McBatchGroupT {
    /// A null handle, the value `*out_group` holds before a successful create.
    pub const fn null() -> Self {
        Self {
            _p: std::ptr::null_mut(),
        }
    }

    /// True when the handle carries no implementation pointer.
    pub const fn is_null(&self) -> bool {
        self._p.is_null()
    }
}

/// ABI 上限（header：`max_slots ≤ 8`；`kMaxBatchSlots`）。
pub const MC_MAX_BATCH_SLOTS: u32 = 8;

/// `mc_batch_options_t` — `sizeof == 32` on x86_64; `struct_size`/
/// `abi_version` required at call time; `reserved` 必须清零。
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct McBatchOptionsT {
    /// Caller must set to `size_of::<McBatchOptionsT>()`.
    pub struct_size: u32,
    /// Caller must set to [`MC_ABI_VERSION`].
    pub abi_version: u32,
    /// 1..=[`MC_MAX_BATCH_SLOTS`]。
    pub max_slots: u32,
    /// 必须与成员 session 的模型档（`mc_model_options_t::max_context_tokens`）
    /// 一致——同 ctx 档 → 同 KV 布局与 attention chunk 档。
    pub max_context_tokens: u32,
    /// 必须清零（自描述 ABI 的兼容扩展先例）。
    pub reserved: [u32; 4],
}

// ---- function pointer types (dlopen) ----
//
// 9 entry points from the v0.1 header + 2 optional async ones reserved by
// plan §5.5 (`mc_decode_enqueue` / `mc_decode_poll`, not exported by the
// current .so).

/// `mc_status_t mc_model_load(const char*, const mc_model_options_t*, mc_model_t*)`
pub type McModelLoadFn = unsafe extern "C" fn(
    model_dir: *const std::ffi::c_char,
    options: *const McModelOptionsT,
    out_model: *mut McModelT,
) -> McStatusT;

/// `void mc_model_destroy(mc_model_t)` — handle by value.
pub type McModelDestroyFn = unsafe extern "C" fn(model: McModelT);

/// `mc_status_t mc_session_create(mc_model_t, mc_session_t*)` — handle by value.
pub type McSessionCreateFn =
    unsafe extern "C" fn(model: McModelT, out_session: *mut McSessionT) -> McStatusT;

/// `mc_status_t mc_session_reset(mc_session_t)` — handle by value.
pub type McSessionResetFn = unsafe extern "C" fn(session: McSessionT) -> McStatusT;

/// `void mc_session_destroy(mc_session_t)` — handle by value.
pub type McSessionDestroyFn = unsafe extern "C" fn(session: McSessionT);

/// `mc_status_t mc_prefill(mc_session_t, const int32_t*, uint32_t)`
pub type McPrefillFn =
    unsafe extern "C" fn(session: McSessionT, token_ids: *const i32, token_count: u32) -> McStatusT;

/// `mc_status_t mc_decode_one(mc_session_t, const mc_generation_config_t*, int32_t*)`
pub type McDecodeOneFn = unsafe extern "C" fn(
    session: McSessionT,
    config: *const McGenerationConfigT,
    out_token: *mut i32,
) -> McStatusT;

/// `mc_status_t mc_get_stats(mc_session_t, mc_runtime_stats_t*)`
pub type McGetStatsFn =
    unsafe extern "C" fn(session: McSessionT, out_stats: *mut McRuntimeStatsT) -> McStatusT;

/// `const char* mc_last_error(void)` — thread-local diagnostics.
pub type McLastErrorFn = unsafe extern "C" fn() -> *const std::ffi::c_char;

/// Reserved by plan §5.5: `mc_status_t mc_decode_enqueue(mc_session_t, const mc_generation_config_t*)`.
/// Not exported by the current native library.
pub type McDecodeEnqueueFn =
    unsafe extern "C" fn(session: McSessionT, config: *const McGenerationConfigT) -> McStatusT;

/// Reserved by plan §5.5: `mc_status_t mc_decode_poll(mc_session_t, uint32_t*, int32_t*)`.
/// Not exported by the current native library.
pub type McDecodePollFn =
    unsafe extern "C" fn(session: McSessionT, ready: *mut u32, out_token: *mut i32) -> McStatusT;

/// `mc_status_t mc_batch_group_create(mc_model_t, const mc_batch_options_t*, mc_batch_group_t*)`
pub type McBatchGroupCreateFn = unsafe extern "C" fn(
    model: McModelT,
    options: *const McBatchOptionsT,
    out_group: *mut McBatchGroupT,
) -> McStatusT;

/// `mc_status_t mc_batch_group_add(mc_batch_group_t, mc_session_t, uint32_t*)`
pub type McBatchGroupAddFn =
    unsafe extern "C" fn(group: McBatchGroupT, session: McSessionT, slot_id: *mut u32) -> McStatusT;

/// `mc_status_t mc_batch_group_remove(mc_batch_group_t, uint32_t)`
///（quantum 边界调用：两次 batch_step 之间）。
pub type McBatchGroupRemoveFn =
    unsafe extern "C" fn(group: McBatchGroupT, slot_id: u32) -> McStatusT;

/// `mc_status_t mc_batch_step(mc_batch_group_t, const uint32_t*, uint32_t, const mc_generation_config_t*, int32_t*)`
pub type McBatchStepFn = unsafe extern "C" fn(
    group: McBatchGroupT,
    slot_ids: *const u32,
    n: u32,
    cfgs: *const McGenerationConfigT,
    out_tokens: *mut i32,
) -> McStatusT;

/// `void mc_batch_group_destroy(mc_batch_group_t)` — 幂等（null 安全 no-op）；
/// 仍在组的成员自动经历 remove 语义后释放组资源。
pub type McBatchGroupDestroyFn = unsafe extern "C" fn(group: McBatchGroupT);

/// Link-mode declarations: for consumers that link `libminicpm_native`
/// directly instead of `dlopen`-ing it. The symbols are only resolved by the
/// linker if a declaration here is actually called.
pub mod linked {
    use super::{
        McBatchGroupT, McBatchOptionsT, McGenerationConfigT, McModelOptionsT, McModelT,
        McRuntimeStatsT, McSessionT, McStatusT,
    };

    unsafe extern "C" {
        pub fn mc_model_load(
            model_dir: *const std::ffi::c_char,
            options: *const McModelOptionsT,
            out_model: *mut McModelT,
        ) -> McStatusT;
        pub fn mc_model_destroy(model: McModelT);
        pub fn mc_session_create(model: McModelT, out_session: *mut McSessionT) -> McStatusT;
        pub fn mc_session_reset(session: McSessionT) -> McStatusT;
        pub fn mc_session_destroy(session: McSessionT);
        pub fn mc_prefill(
            session: McSessionT,
            token_ids: *const i32,
            token_count: u32,
        ) -> McStatusT;
        pub fn mc_decode_one(
            session: McSessionT,
            config: *const McGenerationConfigT,
            out_token: *mut i32,
        ) -> McStatusT;
        pub fn mc_get_stats(session: McSessionT, out_stats: *mut McRuntimeStatsT) -> McStatusT;
        pub fn mc_last_error() -> *const std::ffi::c_char;
        // Plan §5.5 reserved async pair (not present in the current .so).
        pub fn mc_decode_enqueue(
            session: McSessionT,
            config: *const McGenerationConfigT,
        ) -> McStatusT;
        pub fn mc_decode_poll(
            session: McSessionT,
            ready: *mut u32,
            out_token: *mut i32,
        ) -> McStatusT;
        // B1b batched decode 五函数（当前 .so 已导出）。
        pub fn mc_batch_group_create(
            model: McModelT,
            options: *const McBatchOptionsT,
            out_group: *mut McBatchGroupT,
        ) -> McStatusT;
        pub fn mc_batch_group_add(
            group: McBatchGroupT,
            session: McSessionT,
            slot_id: *mut u32,
        ) -> McStatusT;
        pub fn mc_batch_group_remove(group: McBatchGroupT, slot_id: u32) -> McStatusT;
        pub fn mc_batch_step(
            group: McBatchGroupT,
            slot_ids: *const u32,
            n: u32,
            cfgs: *const McGenerationConfigT,
            out_tokens: *mut i32,
        ) -> McStatusT;
        pub fn mc_batch_group_destroy(group: McBatchGroupT);
    }
}

// ---- compile-time layout assertions ----
//
// The native runtime validates `struct_size` at call time, so any accidental
// layout change in these mirrors is an ABI break; fail the build immediately.
const _: () = {
    use std::mem::{align_of, size_of};
    assert!(
        size_of::<McModelOptionsT>() == 40,
        "mc_model_options_t must be 40 bytes"
    );
    assert!(align_of::<McModelOptionsT>() == 8);
    assert!(
        size_of::<McGenerationConfigT>() == 32,
        "mc_generation_config_t must be 32 bytes"
    );
    assert!(align_of::<McGenerationConfigT>() == 8);
    assert!(size_of::<McRuntimeStatsT>() == 48);
    assert!(align_of::<McRuntimeStatsT>() == 8);
    assert!(
        size_of::<McModelT>() == 8,
        "mc_model_t is a by-value single-pointer handle"
    );
    assert!(
        size_of::<McSessionT>() == 8,
        "mc_session_t is a by-value single-pointer handle"
    );
    assert!(align_of::<McModelT>() == 8);
    assert!(align_of::<McSessionT>() == 8);
    assert!(
        size_of::<McBatchGroupT>() == 8,
        "mc_batch_group_t is a by-value single-pointer handle"
    );
    assert!(align_of::<McBatchGroupT>() == 8);
    assert!(
        size_of::<McBatchOptionsT>() == 32,
        "mc_batch_options_t must be 32 bytes"
    );
    assert!(align_of::<McBatchOptionsT>() == 4);
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn handle_sizes_are_pointer_sized() {
        assert_eq!(size_of::<McModelT>(), 8);
        assert_eq!(size_of::<McSessionT>(), 8);
        assert!(McModelT::null().is_null());
        assert!(McSessionT::null().is_null());
    }

    #[test]
    fn struct_sizes_match_header() {
        assert_eq!(size_of::<McModelOptionsT>(), 40);
        assert_eq!(size_of::<McGenerationConfigT>(), 32);
        assert_eq!(size_of::<McRuntimeStatsT>(), 48);
        assert_eq!(size_of::<McBatchOptionsT>(), 32);
        assert_eq!(size_of::<McBatchGroupT>(), 8);
    }

    #[test]
    fn batch_options_shape() {
        let b = McBatchOptionsT {
            struct_size: size_of::<McBatchOptionsT>() as u32,
            abi_version: MC_ABI_VERSION,
            max_slots: 8,
            max_context_tokens: 512,
            reserved: [0; 4],
        };
        assert_eq!(b.struct_size, 32);
        assert_eq!(b.max_slots, MC_MAX_BATCH_SLOTS);
        assert!(McBatchGroupT::null().is_null());
    }

    #[test]
    fn status_constants_match_header() {
        assert_eq!(MC_OK, 0);
        assert_eq!(MC_E_INVALID_ARGUMENT, 1);
        assert_eq!(MC_E_IO, 2);
        assert_eq!(MC_E_CUDA, 3);
        assert_eq!(MC_E_MODEL_MISMATCH, 4);
        assert_eq!(MC_E_OUT_OF_MEMORY, 5);
        assert_eq!(MC_E_GRAPH, 6);
        assert_eq!(MC_E_UNSUPPORTED, 7);
        assert_eq!(MC_E_INTERNAL, 100);
    }

    #[test]
    fn abi_version_matches_header() {
        assert_eq!(MC_ABI_VERSION, 0x0002_0000);
    }

    #[test]
    fn precision_constants_match_header() {
        assert_eq!(MC_BF16, 0);
        assert_eq!(MC_FP16, 1);
        assert_eq!(MC_FP8, 2);
        assert_eq!(MC_NVFP4, 3);
        assert_eq!(MC_KV_BF16, 0);
        assert_eq!(MC_KV_FP8, 1);
    }

    /// `struct_size` is filled at runtime by `minicpm-runtime`; assert the
    /// constants it will report agree with the mirrored layout.
    #[test]
    fn struct_size_fields_agree() {
        let o = McModelOptionsT {
            struct_size: size_of::<McModelOptionsT>() as u32,
            abi_version: MC_ABI_VERSION,
            device_id: 0,
            max_sessions: 1,
            max_context_tokens: 8192,
            precision: MC_BF16,
            kv_precision: MC_KV_BF16,
            enable_cuda_graph: 0,
            reserve_vram_bytes: 0,
        };
        assert_eq!(o.struct_size, 40);
        assert_eq!(o.abi_version, MC_ABI_VERSION);

        let g = McGenerationConfigT {
            struct_size: size_of::<McGenerationConfigT>() as u32,
            max_new_tokens: 32,
            temperature: 0.8,
            top_p: 0.95,
            top_k: 0,
            repetition_penalty: 1.0,
            seed: 42,
        };
        assert_eq!(g.struct_size, 32);
    }
}
