//! 三层 sampler 子系统 (参考 grok-build xai-grok-sampler)。
//!
//! ```text
//!   Layer 3  SamplerHandle  ── 重试 / 取消 / 逐块空闲超时 / 指标   (actor.rs)
//!   Layer 2  Dialect trait  ── chat_completions | responses | anthropic messages
//!                              各方言原始 chunk 归一化为 SamplingEvent  (dialect/)
//!   Layer 1  SamplingClient ── 拼 request / POST / 吐原始 SSE 块流    (client.rs)
//! ```
//!
//! 分层动机: 上层 agent 只依赖 `SamplerHandle::sample` + `SamplingEvent`, 与具体 provider
//! 的 wire 协议、与重试/超时策略都解耦。新增 provider = Layer 2 加一个 Dialect; 调整弹性
//! 策略 = 只动 Layer 3。

pub mod actor;
pub mod client;
pub mod dialect;

pub use actor::{SamplerConfig, SamplerHandle};
pub use client::SamplingClient;
pub use dialect::{for_model, SamplingEvent, SamplingRequest};
// StopReason / Usage 供压缩子系统 (阶段5) 用真实 token 计数, 当前尚未接入。
#[allow(unused_imports)]
pub use dialect::{StopReason, Usage};

use crate::config::Model;
use std::sync::Arc;

/// 便捷装配: 从 Model 选方言, 建 Layer 1 client, 包成 Layer 3 handle。
///
/// 学习点: 这是「组合根」—— 把三层的构造收在一处。main.rs / 测试都走这里, 不各自
///         手搓三层的拼装。
pub fn build(api_key: String, base_url: String, model: &Model, cfg: SamplerConfig) -> SamplerHandle {
    // Box<dyn Dialect> -> Arc<dyn Dialect>: Arc::from 支持从 Box 转 (复用堆分配)。
    let dialect: Arc<dyn dialect::Dialect> = Arc::from(for_model(model));
    let client = SamplingClient::new(api_key, base_url, dialect);
    SamplerHandle::new(client, cfg)
}
