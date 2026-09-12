//! Generation-time host logic: stop conditions, incremental detokenization,
//! tool-call parsing and usage accounting.
//!
//! This crate is tokenizer-agnostic and runtime-agnostic by design: the GPU
//! runtime only ever returns `i32` token ids (plan §4.6 / §10.4), and all
//! byte-level work happens here behind small callbacks.

pub mod stop;
pub mod stream_decode;
pub mod tool_call;
pub mod usage;

pub use stop::{StopHit, StopState};
pub use stream_decode::StreamDecoder;
pub use tool_call::{ToolCall, ToolCallEvent, ToolCallParser};
pub use usage::{FinishReason, Usage};

/// Errors produced while converting token ids to text incrementally.
#[derive(Debug, thiserror::Error)]
#[error("decode backend error: {0}")]
pub struct StreamDecodeError(pub String);
