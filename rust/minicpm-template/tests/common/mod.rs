//! Shared golden-conversation scenarios for template rendering tests.
//!
//! Scenarios follow plan §4.4 / §17.1 (P1 golden set). Each scenario is rendered
//! with the *real* `chat_template.jinja` and byte-compared against
//! `tests/fixtures/*.txt`; the matching `.ids` snapshots hold the tokenized
//! prompt (encode with `add_special_tokens=false`) as the hook point for the
//! later Python `apply_chat_template(..., tokenize=True)` parity pipeline.

#![allow(dead_code)]

use std::path::PathBuf;

use minicpm_template::ChatMessage;
use serde_json::json;

/// The bos token text passed to the template (matches `bos_token` in
/// tokenizer_config.json; encodes to the single id 0).
pub const BOS_TOKEN: &str = "<s>";

#[derive(Clone)]
pub struct Scenario {
    pub name: &'static str,
    pub messages: Vec<ChatMessage>,
    pub tools: Option<serde_json::Value>,
    pub add_generation_prompt: bool,
    pub enable_thinking: Option<bool>,
}

/// Model directory containing the real template + tokenizer; `None` when the
/// model bundle is absent (tests then early-return with a note).
pub fn model_dir() -> Option<PathBuf> {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models/minicpm5-2b");
    if dir.join("chat_template.jinja").exists() {
        Some(dir)
    } else {
        eprintln!("SKIPPED: model bundle not found at {}", dir.display());
        None
    }
}

fn weather_tools() -> serde_json::Value {
    json!([
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "获取指定城市的当前天气",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "城市名称，例如：北京"
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"]
                        }
                    },
                    "required": ["city"]
                }
            }
        }
    ])
}

/// The 8 required scenarios (plus two supplementary tool-call scenarios).
pub fn scenarios() -> Vec<Scenario> {
    vec![
        // ① system + user
        Scenario {
            name: "01_system_user",
            messages: vec![
                ChatMessage::new("system", "You are MiniCPM, created by the MiniCPM Team."),
                ChatMessage::new("user", "你好，请帮我写一首关于秋天的短诗。"),
            ],
            tools: None,
            add_generation_prompt: true,
            enable_thinking: None,
        },
        // ② no system, single user
        Scenario {
            name: "02_single_user_no_system",
            messages: vec![ChatMessage::new(
                "user",
                "What is the capital of France? Answer in one word.",
            )],
            tools: None,
            add_generation_prompt: true,
            enable_thinking: None,
        },
        // ③ multi-turn; history assistant turn embeds <think>...</think> in content
        Scenario {
            name: "03_multiturn_embedded_think",
            messages: vec![
                ChatMessage::new("system", "You are a helpful assistant."),
                ChatMessage::new("user", "9.11 和 9.9 哪个大？"),
                ChatMessage::new(
                    "assistant",
                    "<think>\n9.11 = 9.11，9.9 = 9.90，因为 90 > 11，所以 9.9 更大。\n</think>\n\n9.9 更大。",
                ),
                ChatMessage::new("user", "那 9.11 和 9.11 相比呢？"),
            ],
            tools: None,
            add_generation_prompt: true,
            enable_thinking: None,
        },
        // ④ assistant turn carrying a `reasoning_content` field
        Scenario {
            name: "04_reasoning_content",
            messages: vec![
                ChatMessage::new("system", "You are a helpful assistant."),
                ChatMessage::new("user", "介绍一下你自己。"),
                ChatMessage::new("assistant", "我是 MiniCPM，很高兴认识你！")
                    .with_reasoning_content("用户让我做自我介绍，我应该简洁友好地回答。"),
                ChatMessage::new("user", "你能做什么？"),
            ],
            tools: None,
            add_generation_prompt: true,
            enable_thinking: None,
        },
        // ⑤ tools + system (system content WITHOUT `<tool_def_sep>` — the
        //    fixture variant required by the task; the `<tool_def_sep>` variant
        //    is covered by an inline test in golden.rs)
        Scenario {
            name: "05_tools_with_system",
            messages: vec![
                ChatMessage::new("system", "你是一个乐于助人的助手。"),
                ChatMessage::new("user", "北京现在天气怎么样？"),
            ],
            tools: Some(weather_tools()),
            add_generation_prompt: true,
            enable_thinking: None,
        },
        // ⑥ tools without any system message
        Scenario {
            name: "06_tools_without_system",
            messages: vec![ChatMessage::new("user", "北京现在天气怎么样？")],
            tools: Some(weather_tools()),
            add_generation_prompt: true,
            enable_thinking: None,
        },
        // ⑦ assistant tool_calls followed by `role=tool` tool_response messages
        //    (two consecutive tool messages exercise the single `<|im_start|>user`
        //    block grouping; the second one uses array content, exercising
        //    `message.content | tojson`)
        Scenario {
            name: "07_tool_response",
            messages: vec![
                ChatMessage::new("system", "You are a helpful assistant."),
                ChatMessage::new("user", "北京现在天气怎么样？"),
                ChatMessage::new("assistant", "").with_tool_calls(json!([
                    {
                        "id": "call_0",
                        "type": "function",
                        "function": {
                            "name": "get_current_weather",
                            "arguments": { "city": "北京" }
                        }
                    }
                ])),
                ChatMessage::new("tool", "{\"temperature_c\": 25, \"condition\": \"晴\"}"),
                ChatMessage::with_json_content(
                    "tool",
                    json!([{ "type": "text", "text": "湿度 40%" }]),
                ),
            ],
            tools: Some(weather_tools()),
            add_generation_prompt: true,
            enable_thinking: None,
        },
        // ⑧ add_generation_prompt × enable_thinking = true / false / None
        Scenario {
            name: "08a_enable_thinking_true",
            messages: vec![
                ChatMessage::new("system", "You are a helpful assistant."),
                ChatMessage::new("user", "解释一下重力。"),
            ],
            tools: None,
            add_generation_prompt: true,
            enable_thinking: Some(true),
        },
        Scenario {
            name: "08b_enable_thinking_false",
            messages: vec![
                ChatMessage::new("system", "You are a helpful assistant."),
                ChatMessage::new("user", "解释一下重力。"),
            ],
            tools: None,
            add_generation_prompt: true,
            enable_thinking: Some(false),
        },
        Scenario {
            name: "08c_enable_thinking_none",
            messages: vec![
                ChatMessage::new("system", "You are a helpful assistant."),
                ChatMessage::new("user", "解释一下重力。"),
            ],
            tools: None,
            add_generation_prompt: true,
            enable_thinking: None,
        },
        // ⑨ (supplementary) history assistant turn whose content carries a
        //    `<tool_sep>` marker plus two tool calls — exercises the
        //    min/max split logic and the "extra calls beyond separators" loop
        Scenario {
            name: "09_tool_calls_history",
            messages: vec![
                ChatMessage::new("system", "You are a helpful assistant."),
                ChatMessage::new(
                    "user",
                    "查一下北京明天的天气，顺便告诉我现在几点了。",
                ),
                ChatMessage::new("assistant", "我来查一下。<tool_sep>").with_tool_calls(json!([
                    {
                        "id": "call_0",
                        "type": "function",
                        "function": {
                            "name": "get_current_weather",
                            "arguments": { "city": "北京" }
                        }
                    },
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "get_current_time",
                            "arguments": { "timezone": "Asia/Shanghai" }
                        }
                    }
                ])),
                ChatMessage::new("tool", "{\"temperature_c\": 22, \"condition\": \"多云\"}"),
                ChatMessage::new(
                    "assistant",
                    "北京明天多云，气温约 22 度；现在是下午 3 点。",
                ),
            ],
            tools: Some(weather_tools()),
            add_generation_prompt: true,
            enable_thinking: None,
        },
        // ⑩ (supplementary) tool call arguments containing characters that must
        //    be wrapped in CDATA (`<`, newline) plus a non-string argument value
        Scenario {
            name: "10_tool_calls_cdata",
            messages: vec![
                ChatMessage::new("system", "You are a helpful assistant."),
                ChatMessage::new("user", "帮我检查这段代码：if (a < b) { return; }"),
                ChatMessage::new("assistant", "马上分析。").with_tool_calls(json!([
                    {
                        "id": "call_0",
                        "type": "function",
                        "function": {
                            "name": "analyze_code",
                            "arguments": {
                                "code": "if (a < b) {\n  return;\n}",
                                "severity": 2
                            }
                        }
                    }
                ])),
            ],
            tools: Some(json!([
                {
                    "type": "function",
                    "function": {
                        "name": "analyze_code",
                        "description": "分析一段代码的语法",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "code": { "type": "string" },
                                "severity": { "type": "integer" }
                            },
                            "required": ["code"]
                        }
                    }
                }
            ])),
            add_generation_prompt: true,
            enable_thinking: None,
        },
    ]
}
