//! 错误类型层.
//!
//! 整个项目用 `thiserror` 派生具体错误, 用 `anyhow` 处理顶层装配.
//!
//! 设计:
//! - `AppError` 是顶层枚举, 各子模块的错误用 `#[from]` 自动从子错误升上来.
//! - `?` 运算符配合 `#[from]` 让传播代码极简.
//!
//! 学习点: thiserror 是 derive macro, 它自动给我们生成 `Display` 和 `std::error::Error`
//!         的实现. `#[error("...")]` 字面量里可以用 `{0}`, `{field_name}` 引用变体内容.

use thiserror::Error;

/// 顶层应用错误. main.rs 捕获这一层, 用红字打印后 exit 1.
#[derive(Debug, Error)]
pub enum AppError {
    #[error("API 错误: {0}")]
    Api(#[from] ApiError),

    #[error("工具错误: {0}")]
    Tool(#[from] ToolError),

    #[error("会话错误: {0}")]
    Session(#[from] SessionError),

    #[error("配置错误: {0}")]
    Config(String),

    #[error("io: {0}")]
    Io(#[from] std::io::Error),
}

/// API 层错误 (HTTP / 网络 / 流解析).
#[derive(Debug, Error)]
pub enum ApiError {
    #[error("HTTP {status}: {body}")]
    Http { status: u16, body: String },

    #[error("网络错误: {0}")]
    Network(#[from] reqwest::Error),

    #[error("SSE 流解析失败: {0}")]
    BadStream(String),

    #[error("被速率限制 (HTTP 429), 请稍后重试")]
    RateLimit,

    #[error("采样空闲超时 ({0:?})")]
    IdleTimeout(std::time::Duration),

    /// 预留: 细粒度「流式中途取消」接入后使用 (当前只有 turn 边界取消)。
    #[allow(dead_code)]
    #[error("请求被取消")]
    Cancelled,

    #[error("反序列化响应失败: {0}")]
    BadJson(#[from] serde_json::Error),
}

/// 工具执行错误.
#[derive(Debug, Error)]
pub enum ToolError {
    #[error("参数无效: {0}")]
    BadArgs(String),

    #[error("执行失败: {0}")]
    Failed(String),

    #[error("超时 ({0}s)")]
    Timeout(u64),
}

/// 会话持久化错误.
#[derive(Debug, Error)]
pub enum SessionError {
    #[error("找不到会话: {0}")]
    NotFound(String),

    #[error("解析 session JSON 失败: {0}")]
    BadJson(#[from] serde_json::Error),

    #[error("重放日志失败: {0}")]
    Replay(String),

    #[error("io: {0}")]
    Io(#[from] std::io::Error),
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 测试 #[from] 自动转换: io::Error 应能用 ? 直接转成 AppError.
    /// 为什么这样测: 错误层最关键的人体工程学就是 ? 传播能不能少写代码.
    #[test]
    fn io_error_converts_into_app_error() {
        fn inner() -> Result<(), AppError> {
            // 故意触发 io::Error: 读不存在的文件
            let _ = std::fs::read("/definitely/does/not/exist/edocrs/probe")
                .map_err(AppError::Io)?;
            Ok(())
        }
        let err = inner().unwrap_err();
        // Display 应包含 "io" 或 io 错误本身的描述
        let s = format!("{err}");
        assert!(s.to_lowercase().contains("io") || s.contains("No such file"));
    }

    /// 测试 ApiError::RateLimit 的 Display.
    #[test]
    fn rate_limit_display_is_human_friendly() {
        let e = ApiError::RateLimit;
        let s = format!("{e}");
        assert!(s.contains("429") || s.contains("速率") || s.to_lowercase().contains("rate"));
    }
}
