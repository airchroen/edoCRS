//! chat_completions 方言 —— DeepSeek / OpenAI 的 `/v1/chat/completions`。
//!
//! 这是本子系统唯一真正实现的方言, 承接旧 `api.rs::parse_chunk` 的全部行为, 外加
//! usage 解析 (为压缩子系统预留)。

use super::{Dialect, SamplingEvent, SamplingRequest, StopReason, Usage};
use crate::errors::ApiError;

pub struct ChatCompletionsDialect;

impl Dialect for ChatCompletionsDialect {
    fn endpoint_path(&self) -> &'static str {
        "/v1/chat/completions"
    }

    fn build_body(&self, req: &SamplingRequest) -> Result<serde_json::Value, ApiError> {
        Ok(serde_json::json!({
            "model": req.model.api_id(),
            "messages": req.messages,
            "stream": true,
            // stream_options.include_usage: 让服务端在末 chunk 附带 usage 计数,
            // 供压缩子系统的 TokenLedger 用真实值锚定 (阶段5)。
            "stream_options": {"include_usage": true},
            "tools": req.tools_schema,
        }))
    }

    /// 解析单个 chat_completions SSE payload。
    ///
    /// 学习点: 用 `serde_json::Value` 而非 derive struct, 因为 delta 字段是稀疏的
    ///         (可能只有 content, 或只有 tool_calls, 或末包只有 usage)。用 Value 更灵活。
    ///         返回 `Vec` 是为了统一 Dialect trait 签名 (Anthropic 一个事件可能多个 event);
    ///         chat_completions 一个 payload 恒定映射到 0 或 1 个 SamplingEvent。
    fn parse_chunk(&self, payload: &str) -> Result<Vec<SamplingEvent>, ApiError> {
        let v: serde_json::Value = serde_json::from_str(payload)?;

        // 末包可能只有顶层 usage (choices 为空数组), 先单独看 usage。
        let usage = parse_usage(&v);

        let choice = v.get("choices").and_then(|c| c.get(0));
        let Some(choice) = choice else {
            // 无 choices: 若带 usage 视作一个 Done(Other, usage) 尾包; 否则空。
            if let Some(u) = usage {
                return Ok(vec![SamplingEvent::Done {
                    reason: StopReason::Other,
                    usage: Some(u),
                }]);
            }
            return Ok(vec![]);
        };

        // finish_reason: 若同 chunk 还带 content/tool_calls (尾包), 先出内容再由下一 chunk 收尾;
        // 这里沿用旧逻辑: 只在 delta 无 content/tool_calls 时才发 Done。
        if let Some(reason) = choice.get("finish_reason").and_then(|r| r.as_str()) {
            let delta = choice.get("delta");
            let has_content = delta.and_then(|d| d.get("content")).is_some();
            let has_tools = delta.and_then(|d| d.get("tool_calls")).is_some();
            if !has_content && !has_tools {
                return Ok(vec![SamplingEvent::Done {
                    reason: StopReason::from_chat_completions(reason),
                    usage,
                }]);
            }
        }

        let Some(delta) = choice.get("delta") else {
            return Ok(vec![]);
        };

        if let Some(content) = delta.get("content").and_then(|c| c.as_str()) {
            if !content.is_empty() {
                return Ok(vec![SamplingEvent::TextDelta(content.to_string())]);
            }
        }

        if let Some(rc) = delta.get("reasoning_content").and_then(|c| c.as_str()) {
            if !rc.is_empty() {
                return Ok(vec![SamplingEvent::ReasoningDelta(rc.to_string())]);
            }
        }

        if let Some(tool_calls) = delta.get("tool_calls").and_then(|t| t.as_array()) {
            if let Some(tc) = tool_calls.first() {
                let index = tc.get("index").and_then(|i| i.as_u64()).unwrap_or(0) as usize;
                let id = tc.get("id").and_then(|i| i.as_str()).map(String::from);
                let function = tc.get("function");
                let name = function
                    .and_then(|f| f.get("name"))
                    .and_then(|n| n.as_str())
                    .map(String::from);
                let arguments_fragment = function
                    .and_then(|f| f.get("arguments"))
                    .and_then(|a| a.as_str())
                    .unwrap_or("")
                    .to_string();
                return Ok(vec![SamplingEvent::ToolCallDelta {
                    index,
                    id,
                    name,
                    arguments_fragment,
                }]);
            }
        }

        // 空 delta 无内容: no-op (不再像旧代码那样发空 TextDelta)。
        Ok(vec![])
    }
}

/// 解析顶层 usage 字段 (chat_completions 的 stream_options.include_usage 尾包)。
fn parse_usage(v: &serde_json::Value) -> Option<Usage> {
    let u = v.get("usage")?;
    if u.is_null() {
        return None;
    }
    Some(Usage {
        prompt_tokens: u.get("prompt_tokens").and_then(|x| x.as_u64()).unwrap_or(0) as u32,
        completion_tokens: u
            .get("completion_tokens")
            .and_then(|x| x.as_u64())
            .unwrap_or(0) as u32,
        total_tokens: u.get("total_tokens").and_then(|x| x.as_u64()).unwrap_or(0) as u32,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn d() -> ChatCompletionsDialect {
        ChatCompletionsDialect
    }

    #[test]
    fn parse_text_delta() {
        let json = r#"{"choices":[{"delta":{"content":"hello"},"index":0}]}"#;
        let evs = d().parse_chunk(json).unwrap();
        assert!(matches!(evs.as_slice(), [SamplingEvent::TextDelta(s)] if s == "hello"));
    }

    #[test]
    fn parse_reasoning_delta() {
        let json = r#"{"choices":[{"delta":{"reasoning_content":"先看系统信息"},"index":0}]}"#;
        let evs = d().parse_chunk(json).unwrap();
        assert!(matches!(evs.as_slice(), [SamplingEvent::ReasoningDelta(s)] if s == "先看系统信息"));
    }

    #[test]
    fn parse_tool_call_open() {
        let json = r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"read_file","arguments":"{\"pa"}}]}}]}"#;
        let evs = d().parse_chunk(json).unwrap();
        match evs.as_slice() {
            [SamplingEvent::ToolCallDelta { index, id, name, arguments_fragment }] => {
                assert_eq!(*index, 0);
                assert_eq!(id.as_deref(), Some("call_1"));
                assert_eq!(name.as_deref(), Some("read_file"));
                assert_eq!(arguments_fragment, "{\"pa");
            }
            _ => panic!("wrong: {evs:?}"),
        }
    }

    #[test]
    fn parse_done_event() {
        let json = r#"{"choices":[{"delta":{},"finish_reason":"stop"}]}"#;
        let evs = d().parse_chunk(json).unwrap();
        assert!(matches!(
            evs.as_slice(),
            [SamplingEvent::Done { reason: StopReason::EndTurn, .. }]
        ));
    }

    /// 末包只带 usage (choices 空), 应出一个带 usage 的 Done。
    #[test]
    fn parse_trailing_usage_chunk() {
        let json = r#"{"choices":[],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}"#;
        let evs = d().parse_chunk(json).unwrap();
        match evs.as_slice() {
            [SamplingEvent::Done { usage: Some(u), .. }] => {
                assert_eq!(u.total_tokens, 15);
                assert_eq!(u.prompt_tokens, 10);
            }
            _ => panic!("expected Done+usage: {evs:?}"),
        }
    }

    /// finish_reason=tool_calls 归一化为 ToolUse。
    #[test]
    fn tool_calls_finish_reason_maps_to_tool_use() {
        let json = r#"{"choices":[{"delta":{},"finish_reason":"tool_calls"}]}"#;
        let evs = d().parse_chunk(json).unwrap();
        assert!(matches!(
            evs.as_slice(),
            [SamplingEvent::Done { reason: StopReason::ToolUse, .. }]
        ));
    }
}
