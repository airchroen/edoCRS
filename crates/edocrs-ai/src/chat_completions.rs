//! OpenAI Chat Completions 协议 —— 本项目**唯一**支持的 wire 协议。
//!
//! 两个方向:
//!   - 请求: [`build_body`] 把 model + messages + tools 塑形成 JSON body;
//!   - 响应: [`parse_chunk`] 把单个 SSE data payload 解析成若干 [`SamplingEvent`]。
//!
//! 为什么只做这一种: openai / openrouter / deepseek / ollama 都兼容它, 一种协议覆盖
//! 全部目标 provider; 多协议抽象 (`Dialect` trait) 只会带来没人用的空壳。
//!
//! 流的典型形态 (开启 `stream_options.include_usage` 时):
//! ```text
//!   data: {"choices":[{"delta":{"role":"assistant","content":""}}]}
//!   data: {"choices":[{"delta":{"content":"Hel"}}]}
//!   data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"read","arguments":""}}]}}]}
//!   data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"path\""}}]}}]}
//!   data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}
//!   data: {"choices":[],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}
//!   data: [DONE]
//! ```

use crate::error::AiError;
use crate::message::Message;
use serde_json::{Value, json};

/// 相对 base_url 的端点路径。
pub const ENDPOINT: &str = "/chat/completions";

/// 流终止哨兵; 由 sampler 在进入 `parse_chunk` 前拦截。
pub const DONE_SENTINEL: &str = "[DONE]";

/// 归一化后的采样事件。上层 (agent 循环) 只认这个枚举, 不接触任何 wire 结构。
///
/// 学习点: `Finish` 与 `Usage` 分成两个事件而不是合成一个 `Done{reason, usage}` ——
///         因为协议里它们本来就在**不同的 chunk** 到达 (usage 在 finish_reason 之后的
///         尾包里)。上层应一直读到流结束, 而不是看到 `Finish` 就 break, 否则会丢掉 usage。
#[derive(Clone, Debug, PartialEq)]
pub enum SamplingEvent {
    /// 可见文本增量。
    TextDelta(String),
    /// 推理增量 (可选兼容字段 `reasoning_content`)。
    ReasoningDelta(String),
    /// 工具调用分片。同一个调用的多个分片共享 `index`; `id`/`name` 通常只在首片出现,
    /// `arguments_fragment` 需要按 index 拼接。
    ToolCallDelta {
        index: usize,
        id: Option<String>,
        name: Option<String>,
        arguments_fragment: String,
    },
    /// 本次回复结束的原因 (`finish_reason`)。
    Finish(StopReason),
    /// token 用量 (尾包)。provider 不支持 `include_usage` 时不会出现。
    Usage(Usage),
}

/// `finish_reason` 的归一化。
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum StopReason {
    /// "stop": 模型自然说完。
    Stop,
    /// "tool_calls" (旧版为 "function_call"): 模型请求调用工具。
    ToolCalls,
    /// "length": 撞到 max_tokens 或上下文上限 —— 压缩阶段用它判断溢出。
    Length,
    /// "content_filter": 被安全策略截断。
    ContentFilter,
    /// 其它未知取值。
    Other,
}

impl StopReason {
    pub fn parse(s: &str) -> Self {
        match s {
            "stop" => StopReason::Stop,
            "tool_calls" | "function_call" => StopReason::ToolCalls,
            "length" => StopReason::Length,
            "content_filter" => StopReason::ContentFilter,
            _ => StopReason::Other,
        }
    }
}

/// token 用量。
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub struct Usage {
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub total_tokens: u64,
}

/// 构造请求 body。
///
/// - `tools` 为空时**不写** `tools` 字段: 部分 provider 会拒绝 `"tools": []`;
/// - `stream_options.include_usage` 让服务端在尾包附带真实 token 计数 (压缩阶段要用)。
pub fn build_body(model_id: &str, messages: &[Message], tools: &[Value]) -> Value {
    let mut body = json!({
        "model": model_id,
        "messages": messages,
        "stream": true,
        "stream_options": {"include_usage": true},
    });
    if !tools.is_empty() {
        // 学习点: `json!` 生成的是 `Value::Object`, 可以像 map 一样用 `[key] = ..` 追加字段。
        body["tools"] = Value::from(tools.to_vec());
    }
    body
}

/// 解析单个 SSE data payload。
///
/// 一个 chunk 可能同时携带多种信息 (例如某些 provider 把最后一段文本和 finish_reason
/// 放在同一个 chunk, 或一次给出多个并行 tool_call 分片), 所以返回 `Vec` 并按
/// 「reasoning → text → tool_calls → finish → usage」的顺序全部产出, 不丢任何一项。
///
/// 学习点: 用 `serde_json::Value` 手动取字段而不是 derive 结构体 —— delta 是稀疏的,
///         各家 provider 还会塞额外字段; 用 Value 按需读取最宽容。
pub fn parse_chunk(payload: &str) -> Result<Vec<SamplingEvent>, AiError> {
    let v: Value = serde_json::from_str(payload)?;

    // 流内错误: OpenRouter 等会在 200 响应里以 `{"error":{"message":..}}` 报错。
    if let Some(err) = v.get("error") {
        let msg = err
            .get("message")
            .and_then(Value::as_str)
            .map(String::from)
            .unwrap_or_else(|| err.to_string());
        return Err(AiError::Provider(msg));
    }

    let mut out = Vec::new();

    if let Some(choice) = v.get("choices").and_then(|c| c.get(0)) {
        if let Some(delta) = choice.get("delta") {
            // 学习点: `Value::as_str` 作为函数指针传给 `and_then` —— 比写闭包
            //         `|x| x.as_str()` 更简洁, 这叫 point-free 风格。
            if let Some(rc) = delta.get("reasoning_content").and_then(Value::as_str) {
                if !rc.is_empty() {
                    out.push(SamplingEvent::ReasoningDelta(rc.to_string()));
                }
            }
            if let Some(text) = delta.get("content").and_then(Value::as_str) {
                if !text.is_empty() {
                    out.push(SamplingEvent::TextDelta(text.to_string()));
                }
            }
            if let Some(calls) = delta.get("tool_calls").and_then(Value::as_array) {
                for (pos, tc) in calls.iter().enumerate() {
                    out.push(parse_tool_call_delta(tc, pos));
                }
            }
        }
        if let Some(reason) = choice.get("finish_reason").and_then(Value::as_str) {
            out.push(SamplingEvent::Finish(StopReason::parse(reason)));
        }
    }

    // usage 可能出现在 choices 为空的尾包, 也可能 (少数 provider) 跟在 finish chunk 里。
    if let Some(u) = v.get("usage").filter(|u| !u.is_null()) {
        let n = |k: &str| u.get(k).and_then(Value::as_u64).unwrap_or(0);
        out.push(SamplingEvent::Usage(Usage {
            prompt_tokens: n("prompt_tokens"),
            completion_tokens: n("completion_tokens"),
            total_tokens: n("total_tokens"),
        }));
    }

    Ok(out)
}

/// 解析一个 tool_call 分片。缺 `index` 时退回它在数组中的位置 (个别 provider 不给 index)。
fn parse_tool_call_delta(tc: &Value, pos: usize) -> SamplingEvent {
    let index = tc.get("index").and_then(Value::as_u64).map_or(pos, |i| i as usize);
    let function = tc.get("function");
    let str_field = |obj: Option<&Value>, k: &str| {
        obj.and_then(|o| o.get(k)).and_then(Value::as_str).map(String::from)
    };
    SamplingEvent::ToolCallDelta {
        index,
        // 空串 id/name 视同缺失: 有的 provider 在后续分片里发 `"id": ""`。
        id: str_field(Some(tc), "id").filter(|s| !s.is_empty()),
        name: str_field(function, "name").filter(|s| !s.is_empty()),
        arguments_fragment: str_field(function, "arguments").unwrap_or_default(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_delta() {
        let evs = parse_chunk(r#"{"choices":[{"delta":{"content":"hello"},"index":0}]}"#).unwrap();
        assert_eq!(evs, vec![SamplingEvent::TextDelta("hello".into())]);
    }

    /// 首包常见 `"content": ""` 只带 role, 不应产出空文本事件。
    #[test]
    fn empty_role_chunk_is_noop() {
        let evs = parse_chunk(r#"{"choices":[{"delta":{"role":"assistant","content":""}}]}"#).unwrap();
        assert!(evs.is_empty());
    }

    #[test]
    fn reasoning_delta() {
        let evs = parse_chunk(r#"{"choices":[{"delta":{"reasoning_content":"先想想"}}]}"#).unwrap();
        assert_eq!(evs, vec![SamplingEvent::ReasoningDelta("先想想".into())]);
    }

    #[test]
    fn tool_call_open() {
        let json = r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"read","arguments":"{\"pa"}}]}}]}"#;
        let evs = parse_chunk(json).unwrap();
        assert_eq!(
            evs,
            vec![SamplingEvent::ToolCallDelta {
                index: 0,
                id: Some("call_1".into()),
                name: Some("read".into()),
                arguments_fragment: "{\"pa".into(),
            }]
        );
    }

    /// 一个 chunk 里多个并行 tool_call 分片都要产出。
    #[test]
    fn multiple_tool_calls_in_one_chunk() {
        let json = r#"{"choices":[{"delta":{"tool_calls":[
            {"index":0,"id":"a","function":{"name":"read","arguments":"{}"}},
            {"index":1,"id":"b","function":{"name":"bash","arguments":"{}"}}]}}]}"#;
        let evs = parse_chunk(json).unwrap();
        assert_eq!(evs.len(), 2);
        assert!(matches!(&evs[1], SamplingEvent::ToolCallDelta { index: 1, .. }));
    }

    /// 文本与 finish_reason 同包: 两者都不能丢 (旧实现会吞掉 finish)。
    #[test]
    fn content_and_finish_in_same_chunk() {
        let evs = parse_chunk(r#"{"choices":[{"delta":{"content":"bye"},"finish_reason":"stop"}]}"#).unwrap();
        assert_eq!(
            evs,
            vec![SamplingEvent::TextDelta("bye".into()), SamplingEvent::Finish(StopReason::Stop)]
        );
    }

    #[test]
    fn finish_reasons_normalize() {
        let f = |r: &str| parse_chunk(&format!(r#"{{"choices":[{{"delta":{{}},"finish_reason":"{r}"}}]}}"#)).unwrap();
        assert_eq!(f("tool_calls"), vec![SamplingEvent::Finish(StopReason::ToolCalls)]);
        assert_eq!(f("length"), vec![SamplingEvent::Finish(StopReason::Length)]);
    }

    #[test]
    fn trailing_usage_chunk() {
        let evs = parse_chunk(
            r#"{"choices":[],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}"#,
        )
        .unwrap();
        assert_eq!(
            evs,
            vec![SamplingEvent::Usage(Usage { prompt_tokens: 10, completion_tokens: 5, total_tokens: 15 })]
        );
    }

    #[test]
    fn in_stream_error_becomes_provider_error() {
        let err = parse_chunk(r#"{"error":{"message":"context too long","code":400}}"#).unwrap_err();
        assert!(matches!(err, AiError::Provider(m) if m.contains("context too long")));
    }

    #[test]
    fn body_omits_empty_tools() {
        let body = build_body("m", &[Message::User { content: "hi".into() }], &[]);
        assert_eq!(body["model"], "m");
        assert_eq!(body["stream"], true);
        assert!(body.get("tools").is_none());
        let body = build_body("m", &[], &[json!({"type":"function"})]);
        assert_eq!(body["tools"].as_array().unwrap().len(), 1);
    }
}
