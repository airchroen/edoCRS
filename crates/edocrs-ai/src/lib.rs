//! edocrs-ai —— 模型接入层 (对应 pi 的 `pi-ai` 包)。
//!
//! 职责边界: 只关心「怎么和 LLM 说话」——
//!   - provider 注册表 (openai / openrouter / deepseek / ollama + 用户自定义);
//!   - `provider/model-id` 形式的 Model 解析;
//!   - OpenAI Chat Completions 的请求构造、SSE 流式解析、重试与空闲超时。
//!
//! 它**不知道** agent 循环、工具执行、会话文件、TUI 的存在。
//!
//! 学习点: 把「和外部服务通信」单独拆成一个 crate, 依赖方向由编译器强制 ——
//!         上层 (`edocrs-agent`、`edocrs`) 能用它, 它却 `use` 不到上层任何东西。
//!         这比「同一 crate 里靠自觉分模块」可靠得多。
//!
//! 模块一览 (自底向上):
//! ```text
//!   sse               SSE 字节流切分 (协议无关)
//!   chat_completions  请求体构造 + 单个 data payload 解析 (唯一支持的 wire 协议)
//!   provider          provider 注册表: id → base_url + api key 来源
//!   model             `provider/model-id` 标识
//!   sampler           HTTP 发送 + 重试 + 逐块空闲超时, 对外吐 `SamplingEvent` 流
//! ```

pub mod chat_completions;
pub mod error;
pub mod message;
pub mod model;
pub mod provider;
pub mod sampler;
pub mod sse;

// 学习点: 在 crate 根 `pub use` 常用类型 (façade re-export), 调用方写
//         `edocrs_ai::Model` 而不必记住它住在哪个子模块 —— 子模块结构可以自由调整,
//         公共路径保持稳定。
pub use chat_completions::{SamplingEvent, StopReason, Usage};
pub use error::AiError;
pub use message::{FunctionCall, Message, ToolCall};
pub use model::Model;
pub use provider::{ProviderOverride, ProviderRegistry, ProviderSpec};
pub use sampler::{Sampler, SamplerConfig, SamplingRequest, SamplingStream};
