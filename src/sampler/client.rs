//! Layer 1: SamplingClient —— 纯传输。
//!
//! 职责: 持有 http client + 一个 `Dialect`, 负责发请求、把响应字节流经 `SseSplitter`
//! 切块、逐块交给 `dialect.parse_chunk`, 吐出 `SamplingEvent` 流。**不含**重试/取消/超时
//! 策略 —— 那是 Layer 3 (`SamplerActor`) 的事。
//!
//! `SseSplitter` 从旧 `api.rs` 原样迁入 (SSE 是方言无关的字节切分, 属于传输层)。

use super::dialect::{Dialect, SamplingEvent, SamplingRequest};
use crate::errors::ApiError;
use futures_util::{Stream, StreamExt};
use std::pin::Pin;
use std::sync::Arc;

/// SSE 字节流切分器。喂入任意字节, 按 "\n\n" 边界吐出 data 行 payload (去掉 "data: " 前缀)。
/// 不完整的事件留在内部 buffer 等下一次 feed。
///
/// 学习点: 最小协议解码器。手写 buffer 而不引 tokio_util::codec, 是为了把字节级拼装
///         显式暴露, 便于学习。
pub struct SseSplitter {
    buf: Vec<u8>,
}

impl SseSplitter {
    pub fn new() -> Self {
        Self { buf: Vec::new() }
    }

    /// 喂入一段字节, 返回这一次能完整切出的 data payload 列表。
    pub fn feed(&mut self, chunk: &[u8]) -> Vec<String> {
        self.buf.extend_from_slice(chunk);
        let mut events = Vec::new();
        while let Some(end) = find_double_newline(&self.buf) {
            let event_bytes: Vec<u8> = self.buf.drain(..end + 2).collect();
            let raw = &event_bytes[..event_bytes.len() - 2];
            if let Some(payload) = extract_data(raw) {
                events.push(payload);
            }
        }
        events
    }
}

impl Default for SseSplitter {
    fn default() -> Self {
        Self::new()
    }
}

fn find_double_newline(buf: &[u8]) -> Option<usize> {
    buf.windows(2).position(|w| w == b"\n\n")
}

fn extract_data(line: &[u8]) -> Option<String> {
    let s = std::str::from_utf8(line).ok()?;
    s.lines()
        .find_map(|l| l.strip_prefix("data: ").map(|x| x.to_string()))
}

/// Layer 1 客户端: http + 一个方言 (Arc 共享, 以便移进 stream 闭包)。
///
/// 学习点: dialect 存成 `Arc<dyn Dialect>` 而非 `Box` —— 因为 `stream()` 内部的
///         async_stream 闭包要求 `'static`, 不能借 `&self`。Arc 克隆一份进闭包即可,
///         引用计数在流结束时释放。
pub struct SamplingClient {
    http: reqwest::Client,
    api_key: String,
    base_url: String,
    dialect: Arc<dyn Dialect>,
}

impl SamplingClient {
    pub fn new(api_key: String, base_url: String, dialect: Arc<dyn Dialect>) -> Self {
        let http = reqwest::Client::builder()
            .connect_timeout(std::time::Duration::from_secs(10))
            .timeout(std::time::Duration::from_secs(120))
            .build()
            .expect("reqwest::Client::build() 失败");
        Self {
            http,
            api_key,
            base_url,
            dialect,
        }
    }

    /// 发起一次流式请求, 返回原始 `SamplingEvent` 流 (无重试)。
    pub async fn stream(
        &self,
        req: &SamplingRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<SamplingEvent, ApiError>> + Send>>, ApiError> {
        let body = self.dialect.build_body(req)?;
        let url = format!(
            "{}{}",
            self.base_url.trim_end_matches('/'),
            self.dialect.endpoint_path()
        );

        let resp = match self
            .http
            .post(&url)
            .bearer_auth(&self.api_key)
            .json(&body)
            .send()
            .await
        {
            Ok(r) => r,
            Err(e) => {
                // 连接/发送失败: 归为 Network (可重试)。错误链细节丢进日志级别的诊断。
                // 学习点: reqwest::Error 携带完整 source 链, 但我们的 ApiError::Network
                //         已能 Display 出根因, 无需手工拼链。
                return Err(ApiError::Network(e));
            }
        };

        if resp.status() == reqwest::StatusCode::TOO_MANY_REQUESTS {
            return Err(ApiError::RateLimit);
        }
        if !resp.status().is_success() {
            let status = resp.status().as_u16();
            let body = resp.text().await.unwrap_or_default();
            return Err(ApiError::Http { status, body });
        }

        let dialect = self.dialect.clone();
        let byte_stream = resp.bytes_stream();
        let events = event_stream(byte_stream, dialect);
        Ok(Box::pin(events))
    }
}

/// 把 byte chunk stream 转成 `SamplingEvent` stream, 用给定方言逐块解析。
fn event_stream(
    bytes: impl Stream<Item = Result<bytes::Bytes, reqwest::Error>> + Send + 'static,
    dialect: Arc<dyn Dialect>,
) -> impl Stream<Item = Result<SamplingEvent, ApiError>> + Send {
    async_stream::stream! {
        let mut splitter = SseSplitter::new();
        let sentinel = dialect.done_sentinel();
        futures_util::pin_mut!(bytes);
        while let Some(chunk) = bytes.next().await {
            let chunk = match chunk {
                Ok(c) => c,
                Err(e) => {
                    yield Err(ApiError::Network(e));
                    return;
                }
            };
            for p in splitter.feed(&chunk) {
                if Some(p.as_str()) == sentinel {
                    return;
                }
                match dialect.parse_chunk(&p) {
                    Ok(evs) => {
                        for ev in evs {
                            yield Ok(ev);
                        }
                    }
                    Err(e) => yield Err(e),
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sse_split_single_event() {
        let mut s = SseSplitter::new();
        assert_eq!(s.feed(b"data: hello\n\n"), vec!["hello".to_string()]);
    }

    #[test]
    fn sse_split_across_chunks() {
        let mut s = SseSplitter::new();
        assert!(s.feed(b"data: hel").is_empty());
        assert!(s.feed(b"lo\n").is_empty());
        assert_eq!(s.feed(b"\n"), vec!["hello".to_string()]);
    }

    #[test]
    fn sse_split_done_marker() {
        let mut s = SseSplitter::new();
        assert_eq!(s.feed(b"data: [DONE]\n\n"), vec!["[DONE]".to_string()]);
    }
}
