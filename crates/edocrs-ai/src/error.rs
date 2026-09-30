//! 模型接入层的错误类型。
//!
//! 学习点: 每个 crate 维护自己的错误枚举, 上层用 `#[from]` 把它包进自己的顶层错误 ——
//!         这样 `edocrs-ai` 不必知道 `edocrs` 的 `AppError` 长什么样 (依赖方向不反转)。

use std::time::Duration;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum AiError {
    /// 非 2xx 且非 429 的 HTTP 响应。body 原样保留, 便于诊断 (例如上下文溢出的报错文本)。
    #[error("HTTP {status}: {body}")]
    Http { status: u16, body: String },

    /// 连接 / 发送 / 读流时的传输层错误。
    #[error("网络错误: {0}")]
    Network(#[from] reqwest::Error),

    /// 被 provider 限流。单列出来是因为它总是值得重试。
    #[error("被速率限制 (HTTP 429), 请稍后重试")]
    RateLimit,

    /// 流里出现了无法理解的内容。
    #[error("SSE 流解析失败: {0}")]
    BadStream(String),

    /// provider 在流内以 `{"error": {...}}` 报错 (OpenRouter 等会这么做)。
    #[error("provider 报错: {0}")]
    Provider(String),

    /// 两块数据之间等待超过上限。
    #[error("采样空闲超时 ({0:?})")]
    IdleTimeout(Duration),

    #[error("反序列化响应失败: {0}")]
    BadJson(#[from] serde_json::Error),

    /// 模型字符串不是 `provider/model-id` 形式。
    #[error("非法 model {0:?}: 应为 <provider>/<model-id>")]
    BadModel(String),

    /// 注册表里查不到该 provider。
    #[error("未知 provider {0:?}")]
    UnknownProvider(String),

    /// 自定义 provider 缺必需字段等配置问题。
    #[error("provider 配置错误: {0}")]
    BadProvider(String),

    /// 该 provider 需要 key, 但环境变量没设。
    /// 学习点: 只在**真正请求**时才报 —— 未使用的 provider 缺 key 不影响启动。
    #[error("provider {provider:?} 缺少 API key: 请设置环境变量 {env}")]
    MissingApiKey { provider: String, env: String },
}

impl AiError {
    /// 该错误是否属于「瞬时」故障, 值得在连接阶段退避重试。
    ///
    /// 学习点: 只重试网络抖动 / 限流 / 服务端 5xx。4xx (鉴权失败、请求体非法、上下文溢出)
    ///         重试也只会得到同样的结果, 白白浪费时间。
    pub fn is_retryable(&self) -> bool {
        match self {
            AiError::RateLimit | AiError::Network(_) => true,
            AiError::Http { status, .. } => matches!(status, 500 | 502 | 503 | 504),
            _ => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn retryable_classification() {
        assert!(AiError::RateLimit.is_retryable());
        assert!(AiError::Http { status: 503, body: String::new() }.is_retryable());
        assert!(!AiError::Http { status: 400, body: String::new() }.is_retryable());
        assert!(!AiError::BadModel("x".into()).is_retryable());
    }

    #[test]
    fn missing_key_names_the_env_var() {
        let e = AiError::MissingApiKey { provider: "openai".into(), env: "OPENAI_API_KEY".into() };
        assert!(e.to_string().contains("OPENAI_API_KEY"));
    }
}
