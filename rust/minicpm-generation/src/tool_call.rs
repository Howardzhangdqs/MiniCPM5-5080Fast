//! Streaming tool-call text state machine (plan §4.1: Tool Call 文本状态机).
//!
//! MiniCPM5's chat template emits tool calls as XML inside assistant turns:
//!
//! ```text
//! <function name="fn-name"><param name="p1">v1</param>...</function>
//! ```
//!
//! Param values that contain `<`, `&` or newlines are wrapped in CDATA by
//! the template; the parser strips the CDATA wrapper and keeps the raw
//! value. The parser is fed decoded text fragments as they stream out and
//! emits [`ToolCallEvent`]s. OpenAI-style JSON arguments are produced from
//! the parsed params (string values only, which is what the XML form can
//! represent losslessly).
//!
//! Implementation: buffer-scanning state machine. Ambiguous tails (a `<`
//! that might still grow into a tag) are held back until more text arrives;
//! `finish()` flushes any leftover buffer as literal text so no content is
//! ever silently dropped.

use serde_json::{json, Map};

/// A parsed tool call in OpenAI `tool_calls` shape.
#[derive(Debug, Clone, PartialEq)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    /// JSON object string, as expected by `tool_calls[].function.arguments`.
    pub arguments: String,
}

/// Events emitted while parsing assistant text.
#[derive(Debug, Clone, PartialEq)]
pub enum ToolCallEvent {
    /// Ordinary (non-tool-call) text content.
    Text(String),
    /// `<function name="...">` opened. Text events stop until it closes.
    ToolCallStart { name: String },
    /// A `<param name="...">` value chunk. Values arrive verbatim (CDATA
    /// wrappers removed); chunk boundaries are unspecified.
    ParamValue { name: String, value: String },
    /// `</function>` closed; the call is complete.
    ToolCallEnd(ToolCall),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum State {
    /// Ordinary assistant text.
    Text,
    /// Inside `<function ...>`, between params.
    Function,
    /// Inside `<param ...>...</param>`.
    Param,
    /// Inside a param's `<![CDATA[ ... ]]>`.
    Cdata,
}

const FN_OPEN: &str = "<function";
const FN_CLOSE: &str = "</function>";
const PARAM_OPEN: &str = "<param";
const PARAM_CLOSE: &str = "</param>";
const CDATA_OPEN: &str = "<![CDATA[";
const CDATA_CLOSE: &str = "]]>";

#[derive(Debug)]
pub struct ToolCallParser {
    state: State,
    /// Unprocessed text (may hold an ambiguous tail).
    buf: String,
    current_name: String,
    current_params: Vec<(String, String)>,
    param_name: String,
    param_value: String,
    calls: Vec<ToolCall>,
    call_seq: usize,
}

impl ToolCallParser {
    pub fn new() -> Self {
        Self {
            state: State::Text,
            buf: String::new(),
            current_name: String::new(),
            current_params: Vec::new(),
            param_name: String::new(),
            param_value: String::new(),
            calls: Vec::new(),
            call_seq: 0,
        }
    }

    /// Feed a text fragment; returns events in order. Feed everything,
    /// including the final flush (`finish`).
    pub fn push(&mut self, fragment: &str) -> Vec<ToolCallEvent> {
        self.buf.push_str(fragment);
        let mut events = Vec::new();
        loop {
            if !self.step(&mut events) {
                break;
            }
        }
        events
    }

    /// End of generation: flush ambiguous buffers as literal content and
    /// close any unterminated markup best-effort (never drop content).
    pub fn finish(&mut self) -> Vec<ToolCallEvent> {
        let mut events = Vec::new();
        let leftover = std::mem::take(&mut self.buf);
        match self.state {
            State::Text => self.emit_text(&leftover, &mut events),
            State::Function => { /* stray whitespace between tags: ignore */ }
            State::Param | State::Cdata => {
                if !leftover.is_empty() {
                    self.emit_param_value(&leftover, &mut events);
                }
                self.close_param();
                self.state = State::Function;
            }
        }
        if self.state == State::Function {
            self.close_function(&mut events);
        }
        self.state = State::Text;
        events
    }

    /// All tool calls completed so far.
    pub fn calls(&self) -> &[ToolCall] {
        &self.calls
    }

    /// True when at least one complete `<function>` block was parsed.
    pub fn has_calls(&self) -> bool {
        !self.calls.is_empty()
    }

    /// One processing pass; returns false when more input is needed.
    fn step(&mut self, events: &mut Vec<ToolCallEvent>) -> bool {
        match self.state {
            State::Text => self.step_text(events),
            State::Function => self.step_function(events),
            State::Param => self.step_param(events),
            State::Cdata => self.step_cdata(events),
        }
    }

    fn step_text(&mut self, events: &mut Vec<ToolCallEvent>) -> bool {
        match self.buf.find('<') {
            None => {
                if self.buf.is_empty() {
                    return false;
                }
                let text = std::mem::take(&mut self.buf);
                self.emit_text(&text, events);
                true
            }
            Some(i) => {
                if i > 0 {
                    let text = self.buf.drain(..i).collect::<String>();
                    self.emit_text(&text, events);
                }
                // buf starts with '<'
                if self.buf.starts_with(FN_OPEN) {
                    match parse_tag(&self.buf, FN_OPEN, "name") {
                        Some(name) => {
                            self.state = State::Function;
                            self.current_name = name.clone();
                            self.buf.drain(..tag_len(&self.buf)).count();
                            events.push(ToolCallEvent::ToolCallStart { name });
                            return true;
                        }
                        None if self.buf.find('>').is_none() => return false, // wait for '>'
                        None => { /* complete tag without name: literal */ }
                    }
                }
                if viable_prefix(&self.buf, FN_OPEN) {
                    return false; // wait for more input
                }
                // definitively not a function tag: literal '<'
                self.buf.remove(0);
                self.emit_text("<", events);
                true
            }
        }
    }

    fn step_function(&mut self, events: &mut Vec<ToolCallEvent>) -> bool {
        match self.buf.find('<') {
            None => {
                self.buf.clear(); // whitespace between tags
                false
            }
            Some(0) => {
                if starts_with(&self.buf, FN_CLOSE) {
                    self.buf.drain(..FN_CLOSE.len()).count();
                    self.close_function(events);
                    return true;
                }
                if self.buf.starts_with(PARAM_OPEN) {
                    match parse_tag(&self.buf, PARAM_OPEN, "name") {
                        Some(name) => {
                            self.state = State::Param;
                            self.param_name = name;
                            self.param_value.clear();
                            self.buf.drain(..tag_len(&self.buf)).count();
                            return true;
                        }
                        None if self.buf.find('>').is_none() => return false, // wait for '>'
                        None => { /* complete tag without name: skip */ }
                    }
                }
                if viable_prefix(&self.buf, FN_CLOSE) || viable_prefix(&self.buf, PARAM_OPEN) {
                    return false;
                }
                self.buf.remove(0); // stray '<': drop
                true
            }
            Some(i) => {
                self.buf.drain(..i).count(); // whitespace
                true
            }
        }
    }

    fn step_param(&mut self, events: &mut Vec<ToolCallEvent>) -> bool {
        // CDATA opener (only meaningful at value start of a segment)
        if starts_with(&self.buf, CDATA_OPEN) {
            self.state = State::Cdata;
            self.buf.drain(..CDATA_OPEN.len()).count();
            return true;
        }
        if viable_prefix(&self.buf, CDATA_OPEN) {
            return false;
        }
        match self.buf.find('<') {
            None => {
                if self.buf.is_empty() {
                    return false;
                }
                let text = std::mem::take(&mut self.buf);
                self.emit_param_value(&text, events);
                true
            }
            Some(i) => {
                if i > 0 {
                    let text = self.buf.drain(..i).collect::<String>();
                    self.emit_param_value(&text, events);
                }
                if starts_with(&self.buf, PARAM_CLOSE) {
                    self.buf.drain(..PARAM_CLOSE.len()).count();
                    self.close_param();
                    self.state = State::Function;
                    return true;
                }
                if viable_prefix(&self.buf, PARAM_CLOSE) {
                    return false;
                }
                // '<' is literal param data (e.g. "a < b" without CDATA)
                self.buf.remove(0);
                self.emit_param_value("<", events);
                true
            }
        }
    }

    fn step_cdata(&mut self, events: &mut Vec<ToolCallEvent>) -> bool {
        match self.buf.find(CDATA_CLOSE) {
            Some(i) => {
                let text = self.buf.drain(..i).collect::<String>();
                self.emit_param_value(&text, events);
                self.buf.drain(..CDATA_CLOSE.len()).count();
                self.state = State::Param;
                true
            }
            None => {
                // hold back a tail that might grow into "]]>"
                let hold = CDATA_CLOSE.len() - 1;
                if self.buf.len() > hold {
                    let cut = self.buf.len() - hold;
                    let cut = floor_char_boundary(&self.buf, cut);
                    if cut > 0 {
                        let text = self.buf.drain(..cut).collect::<String>();
                        self.emit_param_value(&text, events);
                    }
                }
                false
            }
        }
    }

    fn emit_text(&mut self, text: &str, events: &mut Vec<ToolCallEvent>) {
        events.push(ToolCallEvent::Text(text.to_string()));
    }

    fn emit_param_value(&mut self, text: &str, events: &mut Vec<ToolCallEvent>) {
        self.param_value.push_str(text);
        events.push(ToolCallEvent::ParamValue { name: self.param_name.clone(), value: text.to_string() });
    }

    fn close_param(&mut self) {
        let name = std::mem::take(&mut self.param_name);
        let value = std::mem::take(&mut self.param_value);
        self.current_params.push((name, value));
    }

    fn close_function(&mut self, events: &mut Vec<ToolCallEvent>) {
        let name = std::mem::take(&mut self.current_name);
        let params = std::mem::take(&mut self.current_params);
        let call = finalize_call(&name, params, self.call_seq);
        self.call_seq += 1;
        events.push(ToolCallEvent::ToolCallEnd(call.clone()));
        self.calls.push(call);
        self.state = State::Text;
    }
}

impl Default for ToolCallParser {
    fn default() -> Self {
        Self::new()
    }
}

/// True when `buf` is a (possibly complete) prefix of `lit` after the
/// leading '<' — i.e. more input could still turn it into `lit`.
fn viable_prefix(buf: &str, lit: &str) -> bool {
    if buf.len() >= lit.len() {
        return false;
    }
    lit.starts_with(buf)
}

fn starts_with(buf: &str, lit: &str) -> bool {
    buf.starts_with(lit)
}

/// If `buf` starts with `<tag ... name="value" ...>`, return value and the
/// total tag length is discovered via [`tag_len`]. Returns None when the
/// tag is not fully arrived yet or doesn't match.
fn parse_tag(buf: &str, tag: &str, attr: &str) -> Option<String> {
    if !buf.starts_with(tag) {
        return None;
    }
    let close = buf.find('>')?;
    let body = &buf[tag.len()..close];
    // only well-formed `name="..."` / `name='...'` is accepted
    for part in body.split_whitespace() {
        if let Some(v) = part.strip_prefix(&format!("{attr}=\"")) {
            if let Some(v) = v.strip_suffix('"') {
                return Some(decode_attr(v));
            }
        } else if let Some(v) = part.strip_prefix(&format!("{attr}='")) {
            if let Some(v) = v.strip_suffix('\'') {
                return Some(decode_attr(v));
            }
        }
    }
    None // tag arrived but no name attribute: treat as not-a-call
}

/// Length of the complete tag at the start of `buf` (through '>').
fn tag_len(buf: &str) -> usize {
    buf.find('>').map(|i| i + 1).unwrap_or(buf.len())
}

/// Minimal attribute unescaping.
fn decode_attr(v: &str) -> String {
    v.replace("&lt;", "<").replace("&gt;", ">").replace("&quot;", "\"").replace("&amp;", "&")
}

fn floor_char_boundary(s: &str, mut i: usize) -> usize {
    while i > 0 && !s.is_char_boundary(i) {
        i -= 1;
    }
    i
}

fn finalize_call(name: &str, params: Vec<(String, String)>, seq: usize) -> ToolCall {
    let mut args = Map::new();
    for (k, v) in params {
        args.insert(k, json!(v));
    }
    ToolCall {
        id: format!("call_{seq}"),
        name: name.to_string(),
        arguments: serde_json::Value::Object(args).to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn collect_all(text: &str) -> (Vec<ToolCallEvent>, ToolCallParser) {
        let mut p = ToolCallParser::new();
        let mut events = p.push(text);
        events.extend(p.finish());
        (events, p)
    }

    fn joined_text(events: &[ToolCallEvent]) -> String {
        events
            .iter()
            .filter_map(|e| match e {
                ToolCallEvent::Text(t) => Some(t.as_str()),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn plain_text_passthrough() {
        let (events, _) = collect_all("just a normal answer, 1 < 2 done");
        assert_eq!(joined_text(&events), "just a normal answer, 1 < 2 done");
        assert!(!events.iter().any(|e| matches!(e, ToolCallEvent::ToolCallEnd(_))));
    }

    #[test]
    fn single_tool_call() {
        let text = r#"before <function name="get_weather"><param name="city">北京</param></function> after"#;
        let (events, p) = collect_all(text);
        assert_eq!(p.calls().len(), 1);
        assert_eq!(p.calls()[0].name, "get_weather");
        let args: serde_json::Value = serde_json::from_str(&p.calls()[0].arguments).unwrap();
        assert_eq!(args["city"], "北京");
        assert_eq!(p.calls()[0].id, "call_0");
        assert_eq!(joined_text(&events), "before  after");
        assert!(events.iter().any(|e| matches!(e, ToolCallEvent::ToolCallStart { .. })));
    }

    #[test]
    fn cdata_value_kept_verbatim() {
        let text = "<function name=\"edit\"><param name=\"code\"><![CDATA[int x < 1;\nmultiline]]></param></function>";
        let (_, p) = collect_all(text);
        let args: serde_json::Value = serde_json::from_str(&p.calls()[0].arguments).unwrap();
        assert_eq!(args["code"], "int x < 1;\nmultiline");
    }

    #[test]
    fn multiple_params_and_calls() {
        let text = "<function name=\"f\"><param name=\"a\">1</param><param name=\"b\">2</param></function>\
                    mid\
                    <function name=\"g\"><param name=\"x\">y</param></function>";
        let (events, p) = collect_all(text);
        assert_eq!(p.calls().len(), 2);
        assert_eq!(p.calls()[1].name, "g");
        assert_eq!(p.calls()[1].id, "call_1");
        assert_eq!(joined_text(&events), "mid");
    }

    #[test]
    fn incomplete_call_is_not_lost() {
        let (events, p) = collect_all("partial <function name=\"h\"><param name=\"q\">v");
        assert_eq!(p.calls().len(), 1);
        assert_eq!(p.calls()[0].name, "h");
        let args: serde_json::Value = serde_json::from_str(&p.calls()[0].arguments).unwrap();
        assert_eq!(args["q"], "v");
        assert_eq!(joined_text(&events), "partial ");
    }

    #[test]
    fn streaming_chunks_reassemble() {
        let mut p = ToolCallParser::new();
        let mut all = Vec::new();
        for chunk in ["<func", "tion name=\"f\">", "<param na", "me=\"a\">x</par", "am></function>"] {
            all.extend(p.push(chunk));
        }
        all.extend(p.finish());
        assert_eq!(p.calls().len(), 1);
        assert_eq!(p.calls()[0].name, "f");
        let args: serde_json::Value = serde_json::from_str(&p.calls()[0].arguments).unwrap();
        assert_eq!(args["a"], "x");
    }

    #[test]
    fn literal_angle_bracket_in_param_without_cdata() {
        let text = "<function name=\"cmp\"><param name=\"expr\">a < b</param></function>";
        let (_, p) = collect_all(text);
        let args: serde_json::Value = serde_json::from_str(&p.calls()[0].arguments).unwrap();
        assert_eq!(args["expr"], "a < b");
    }

    #[test]
    fn empty_param_value() {
        let text = "<function name=\"noop\"></function>";
        let (_, p) = collect_all(text);
        assert_eq!(p.calls().len(), 1);
        assert_eq!(p.calls()[0].arguments, "{}");
    }
}
