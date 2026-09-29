//! Layer 2: 方言 (Dialect) —— 把各 provider 的 wire 协议归一化为统一 `SamplingEvent`。
//!
//! 设计 (参考 grok-build xai-grok-sampler Layer 2): 上层 agent 只认 `SamplingEvent`,
//! 永远不接触任何一家 provider 的 wire 结构。新增 provider = 新增一个 `Dialect` 实现,
//! `SamplingEvent` 与 agent 都不动 —— 这就是「防腐层」(anti-corruption layer)。
//!
//! 本子系统只真正实现 `ChatCompletionsDialect` (承接 DeepSeek/OpenAI chat 方言);
//! `responses` / `anthropic_messages` 留空壳 + 差异点注释, 等真正接其它 provider 再填。

pub mod chat_completions;

use crate::api::Message;
use crate::config::Model;
use crate::errors::ApiError;

/// 方言无关的统一采样事件。三个 Dialect 适配器都归一化到这个枚举。
///
/// 学习点: 相比旧的 `StreamEvent`, 这里 `Done` 多带了 `usage` —— 为压缩子系统 (阶段5)
///         的 token 计数预留。语义上「方言无关」: 不管底层是 chat_completions 的
///         `finish_reason` 还是 Anthropic 的 `stop_reason`, 都落到统一的 `StopReason`。
#[derive(Clone, Debug)]
pub enum SamplingEvent {
    /// 可见文本增量。
    TextDelta(String),
    /// 推理/思考增量 (DeepSeek reasoning_content / Anthropic thinking 都归一到此)。
    ReasoningDelta(String),
    /// 工具调用分片, 按 index 归并 (沿用 agent 现有归并逻辑)。
    ToolCallDelta {
        index: usize,
        id: Option<String>,
        name: Option<String>,
        arguments_fragment: String,
    },
    /// 流终止。
    Done {
        reason: StopReason,
        usage: Option<Usage>,
    },
}

/// 跨方言归一化的终止原因。
/// - chat_completions: "stop"/"tool_calls"/"length"
/// - anthropic:        "end_turn"/"tool_use"/"max_tokens"
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum StopReason {
    EndTurn,
    ToolUse,
    MaxTokens,
    Other,
}

impl StopReason {
    /// 从 chat_completions 的 finish_reason 字符串归一化。
    pub fn from_chat_completions(s: &str) -> Self {
        match s {
            "stop" => StopReason::EndTurn,
            "tool_calls" => StopReason::ToolUse,
            "length" => StopReason::MaxTokens,
            _ => StopReason::Other,
        }
    }
}

/// token 用量 (方言若不提供则 None)。
#[derive(Copy, Clone, Debug, Default)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

/// 方言无关的采样请求 (Layer 3 → Layer 1/2 传递)。
pub struct SamplingRequest {
    pub model: Model,
    pub messages: Vec<Message>,
    pub tools_schema: Vec<serde_json::Value>,
}

/// 一个 API 方言: 知道如何 (a) 构造该 provider 的请求 body 与 URL 路径;
/// (b) 把该 provider 的单个 SSE data payload 解析成若干 `SamplingEvent`。
///
/// 学习点: 把「请求塑形」与「响应解析」绑在同一 trait, 因为二者是同一套 wire 协议的
///         两个方向。Layer 1 的 `SamplingClient` 只依赖这个 trait, 对具体 provider
///         无感知 (依赖倒置)。
pub trait Dialect: Send + Sync {
    /// 请求相对路径, 如 "/v1/chat/completions" 或 "/v1/messages"。
    fn endpoint_path(&self) -> &'static str;

    /// 把统一请求参数塑形成该方言的 JSON body。
    /// 返回 Result: 未实现的方言返回 `ApiError` 而非 panic —— 让上层把它当作一次
    /// 失败的 turn 优雅处理, 而不是拖垮整个进程。
    fn build_body(&self, req: &SamplingRequest) -> Result<serde_json::Value, ApiError>;

    /// 解析单个 SSE data payload。返回 Vec 是因为 Anthropic 一个事件可能对应
    /// 0 或多个 `SamplingEvent`。终止哨兵 (如 "[DONE]") 由 Layer 1 拦截, 不进这里。
    fn parse_chunk(&self, payload: &str) -> Result<Vec<SamplingEvent>, ApiError>;

    /// 该方言的流终止哨兵 (chat_completions 是 "[DONE]"; Anthropic 无, 返回 None)。
    fn done_sentinel(&self) -> Option<&'static str> {
        Some("[DONE]")
    }
}

/// 按 Model 的 dialect 字段选一个方言适配器。
///
/// 学习点: 这是「工厂」—— 把「配置里的枚举」映射到「trait object 实现」。多方言真正
///         接入时, 这里每加一行就多支持一种 wire 协议。
pub fn for_model(model: &Model) -> Box<dyn Dialect> {
    use crate::config::Dialect as D;
    match model.dialect {
        D::ChatCompletions => Box::new(chat_completions::ChatCompletionsDialect),
        D::Responses => Box::new(ResponsesDialect),
        D::AnthropicMessages => Box::new(AnthropicMessagesDialect),
    }
}

// ── 空壳方言 (本子系统不实现真逻辑, 只固化差异点在注释里) ──────────────────

/// OpenAI responses 方言。**本子系统未实现** —— 见 §差异点。
///
/// 差异点 (相对 chat_completions):
///   - endpoint: `/v1/responses`;
///   - 流结构: `response.output[*]` 事件流, 非 `choices[0].delta`;
///   - 文本增量: `output_text.delta`; 推理: `delta.reasoning`;
///   - 终止: `response.completed` 事件 (无 `[DONE]` 哨兵);
///   - usage: `response.usage`。
pub struct ResponsesDialect;

impl Dialect for ResponsesDialect {
    fn endpoint_path(&self) -> &'static str {
        "/v1/responses"
    }
    fn build_body(&self, _req: &SamplingRequest) -> Result<serde_json::Value, ApiError> {
        Err(ApiError::BadStream(
            "responses 方言尚未实现 (多方言 task 待填)".into(),
        ))
    }
    fn parse_chunk(&self, _payload: &str) -> Result<Vec<SamplingEvent>, ApiError> {
        Err(ApiError::BadStream("responses 方言尚未实现".into()))
    }
    fn done_sentinel(&self) -> Option<&'static str> {
        None // responses 用 response.completed 事件, 无字面哨兵
    }
}

/// Anthropic messages 方言。**本子系统未实现** —— 见 §差异点。
///
/// 差异点 (相对 chat_completions), 归一化的核心难点在 reasoning vs thinking:
///   - endpoint: `/v1/messages`; 鉴权头是 `x-api-key` + `anthropic-version`, 非 Bearer;
///   - system 消息: 顶层 `system` 字段, **不进** messages 数组;
///   - 文本增量: `content_block_delta` 里 `delta.type=="text_delta"` 的 `.text`;
///   - **推理增量**: `delta.type=="thinking_delta"` 的 `.thinking`, 且 thinking block 有
///     独立 `signature` —— 回传时必须保留 (chat_completions 的 reasoning_content 是
///     单一字符串增量, 无 signature 概念, 这是两者归一化最大的落差);
///   - 工具调用: `content_block_start`(type=tool_use, 带 id+name) + `input_json_delta`;
///   - 终止: `message_delta.stop_reason` + `message_stop` 事件;
///   - usage: `message_delta.usage`。
pub struct AnthropicMessagesDialect;

impl Dialect for AnthropicMessagesDialect {
    fn endpoint_path(&self) -> &'static str {
        "/v1/messages"
    }
    fn build_body(&self, _req: &SamplingRequest) -> Result<serde_json::Value, ApiError> {
        Err(ApiError::BadStream(
            "anthropic messages 方言尚未实现 (多方言 task 待填: system 提顶层 + thinking signature)".into(),
        ))
    }
    fn parse_chunk(&self, _payload: &str) -> Result<Vec<SamplingEvent>, ApiError> {
        Err(ApiError::BadStream("anthropic messages 方言尚未实现".into()))
    }
    fn done_sentinel(&self) -> Option<&'static str> {
        None // Anthropic 用 message_stop 事件, 无字面哨兵
    }
}
