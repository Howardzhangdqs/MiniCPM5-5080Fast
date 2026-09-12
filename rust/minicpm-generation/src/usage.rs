//! Usage accounting and finish reasons (OpenAI compatible).

/// Why generation ended. Maps to OpenAI `finish_reason`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinishReason {
    /// Natural end (eos / stop token / stop string).
    Stop,
    /// `max_tokens` reached.
    Length,
    /// Client cancelled the request.
    Cancelled,
    /// Runtime or host error.
    Error,
}

impl FinishReason {
    pub fn as_str(&self) -> &'static str {
        match self {
            FinishReason::Stop => "stop",
            FinishReason::Length => "length",
            FinishReason::Cancelled => "stop", // OpenAI has no "cancelled"
            FinishReason::Error => "error",
        }
    }
}

/// Token usage for one request (OpenAI `usage` object).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Usage {
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
}

impl Usage {
    pub fn total_tokens(&self) -> u64 {
        self.prompt_tokens + self.completion_tokens
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn usage_math() {
        let u = Usage { prompt_tokens: 10, completion_tokens: 5 };
        assert_eq!(u.total_tokens(), 15);
    }

    #[test]
    fn finish_reason_strings() {
        assert_eq!(FinishReason::Stop.as_str(), "stop");
        assert_eq!(FinishReason::Length.as_str(), "length");
        assert_eq!(FinishReason::Cancelled.as_str(), "stop");
    }
}
