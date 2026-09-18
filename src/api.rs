//! wire 消息类型 (OpenAI/DeepSeek chat_completions 兼容)。
//!
//! 本文件只保留「跨方言复用」的消息 domain 类型: `Message` / `ToolCall` / `FunctionCall`。
//! SSE 切分、方言解析、HTTP client 已在子系统 4 迁到 `sampler/` 三层里 (见 sampler/mod.rs)。
//!
//! ⚠️ 历史: 早期这里还有 `SseSplitter` / `StreamEvent` / `parse_chunk` / `ApiClient`。
//!    三层 sampler 重构后, SSE 切分进了 `sampler/client.rs`, 解析进了
//!    `sampler/dialect/chat_completions.rs`, 归一化事件变成 `sampler::SamplingEvent`。

use serde::{Deserialize, Serialize};

/// OpenAI/DeepSeek 兼容的消息. role 字段由 #[serde(tag = "role")] 自动生成.
///
/// 学习点: #[serde(tag = "role", rename_all = "snake_case")] 把 enum 变体名转换为 role
///         字段值 (User -> "user"). 这就是 internally tagged enum.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "role", rename_all = "snake_case")]
pub enum Message {
    System {
        content: String,
    },
    User {
        content: String,
    },
    Assistant {
        /// 文本输出, 模型只调工具不说话时为 None.
        #[serde(skip_serializing_if = "Option::is_none")]
        content: Option<String>,
        /// DeepSeek thinking mode 返回的推理内容. 如果 assistant 同时产生 tool_calls,
        /// 下一轮请求必须原样带回, 否则 API 会返回 400.
        #[serde(skip_serializing_if = "Option::is_none")]
        reasoning_content: Option<String>,
        /// 工具调用列表. 没有调用时为空 Vec, serde 会跳过该字段不输出.
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        tool_calls: Vec<ToolCall>,
    },
    Tool {
        tool_call_id: String,
        content: String,
    },
}

/// 一次工具调用 (模型生成).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    /// 始终是 "function" (DeepSeek/OpenAI 当前唯一的 type).
    /// 学习点: 我们用 #[serde(rename = "type")] 是因为 Rust 关键字 type 不能做字段名.
    #[serde(rename = "type")]
    pub kind: String,
    pub function: FunctionCall,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FunctionCall {
    pub name: String,
    /// 参数原始 JSON 字符串 (注意: 不是 serde_json::Value, 而是字符串).
    /// 学习点: OpenAI/DeepSeek 协议规定这里是字符串而非对象. 模型有时候输出非法 JSON,
    ///         留给工具自行解析+报错, 比 serde 提前失败要更可控.
    pub arguments: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Message::User 的序列化形态必须严格匹配 OpenAI 协议.
    #[test]
    fn user_message_serializes_with_role() {
        let m = Message::User { content: "hello".into() };
        let v: serde_json::Value = serde_json::to_value(&m).unwrap();
        assert_eq!(v["role"], "user");
        assert_eq!(v["content"], "hello");
    }

    /// Assistant 消息可以同时带 content 和 tool_calls.
    #[test]
    fn assistant_message_with_tool_calls() {
        let m = Message::Assistant {
            content: Some("我来调用工具".into()),
            reasoning_content: None,
            tool_calls: vec![ToolCall {
                id: "call_1".into(),
                kind: "function".into(),
                function: FunctionCall {
                    name: "read_file".into(),
                    arguments: "{\"path\":\"a.txt\"}".into(),
                },
            }],
        };
        let v: serde_json::Value = serde_json::to_value(&m).unwrap();
        assert_eq!(v["role"], "assistant");
        assert_eq!(v["tool_calls"][0]["id"], "call_1");
        assert_eq!(v["tool_calls"][0]["function"]["name"], "read_file");
    }

    /// DeepSeek thinking mode 要求 tool call 后继续请求时回传 assistant.reasoning_content.
    #[test]
    fn assistant_message_preserves_reasoning_content() {
        let m = Message::Assistant {
            content: None,
            reasoning_content: Some("需要查看系统信息".into()),
            tool_calls: vec![ToolCall {
                id: "call_1".into(),
                kind: "function".into(),
                function: FunctionCall {
                    name: "bash".into(),
                    arguments: "{\"command\":\"fastfetch\"}".into(),
                },
            }],
        };
        let v: serde_json::Value = serde_json::to_value(&m).unwrap();
        assert_eq!(v["role"], "assistant");
        assert_eq!(v["reasoning_content"], "需要查看系统信息");
        assert_eq!(v["tool_calls"][0]["id"], "call_1");
    }

    /// Tool 消息携带 tool_call_id.
    #[test]
    fn tool_message_has_tool_call_id() {
        let m = Message::Tool {
            tool_call_id: "call_1".into(),
            content: "file content".into(),
        };
        let v: serde_json::Value = serde_json::to_value(&m).unwrap();
        assert_eq!(v["role"], "tool");
        assert_eq!(v["tool_call_id"], "call_1");
    }
}
