//! Layer 3: SamplerActor —— 弹性策略层。
//!
//! 职责: 在 Layer 1 的原始流之上加「重试 / 取消 / 逐块空闲超时 / TTFT·ITL 指标」。
//!
//! 设计 (参考 grok-build xai-grok-sampler Layer 3 + actor/mod.rs):
//!   - actor 本身单线程逐条处理命令, 但每个采样请求可 spawn 独立任务并发在途;
//!   - 逐块空闲超时: 用 `tokio::time::timeout` 包住「等下一块」, 每收到一块就重置计时器
//!     (默认 300s, 与 grok-build 同值) —— 防止服务端半死不活挂住我们;
//!   - 重试: 对网络错误 / 429 做指数退避重试。
//!
//! ⚠️ 本项目当前形态: edoCRS 的 SamplerActor 不额外起 actor 线程 (会话已在单线程 actor 上),
//!    而是把弹性策略做成 `SamplerHandle::sample` 的一个包装流。命名沿用 grok-build 的
//!    「Handle/Actor」概念以对齐学习目标, 但实现按本项目体量精简。

use super::client::SamplingClient;
use super::dialect::{SamplingEvent, SamplingRequest};
use crate::errors::ApiError;
use futures_util::{Stream, StreamExt};
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

/// 采样弹性策略配置。
#[derive(Clone, Copy, Debug)]
pub struct SamplerConfig {
    /// 网络错误 / 429 的最大重试次数。
    pub max_retries: u32,
    /// 逐块空闲超时: 两块之间等待上限。
    pub idle_timeout: Duration,
}

impl Default for SamplerConfig {
    fn default() -> Self {
        Self {
            max_retries: 2,
            idle_timeout: Duration::from_secs(300),
        }
    }
}

/// Layer 3 句柄。持有 Layer 1 client + 策略。Clone 廉价 (Arc)。
#[derive(Clone)]
pub struct SamplerHandle {
    client: Arc<SamplingClient>,
    cfg: SamplerConfig,
}

impl SamplerHandle {
    pub fn new(client: SamplingClient, cfg: SamplerConfig) -> Self {
        Self {
            client: Arc::new(client),
            cfg,
        }
    }

    /// 发起一次采样, 返回带弹性策略的事件流。
    ///
    /// 重试: 建立连接阶段的错误 (网络 / 429) 会退避重试 max_retries 次。一旦开始出事件
    /// 就不再重试 (流中途断只能报错, 因为已经有部分输出交给上层了)。
    ///
    /// 空闲超时: 用 `idle_timeout` 包住每次「等下一块」。
    ///
    /// 学习点: 「连接阶段可重试, 流中途不可重试」是流式 API 的常见约束 —— 重试要求幂等,
    ///         而一旦把 token 吐给了上层, 重发会造成重复输出。
    pub async fn sample(
        &self,
        req: SamplingRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<SamplingEvent, ApiError>> + Send>>, ApiError> {
        let mut attempt = 0;
        loop {
            match self.client.stream(&req).await {
                Ok(inner) => {
                    let idle = self.cfg.idle_timeout;
                    return Ok(Box::pin(with_idle_timeout(inner, idle)));
                }
                Err(e) if is_retryable(&e) && attempt < self.cfg.max_retries => {
                    attempt += 1;
                    // 指数退避: 100ms * 2^(attempt-1)。学习项目用短基数, 不真等秒级。
                    let backoff = Duration::from_millis(100 * (1 << (attempt - 1)));
                    tokio::time::sleep(backoff).await;
                }
                Err(e) => return Err(e),
            }
        }
    }
}

/// 连接阶段哪些错误值得重试。
/// 学习点: 只重试「瞬时」错误 (网络抖动 / 限流)。`BadStream` 可能是「方言未实现」这类
///         永久错误, 重试无益, 故不含。
fn is_retryable(e: &ApiError) -> bool {
    matches!(e, ApiError::RateLimit | ApiError::Network(_))
}

/// 给事件流套一层逐块空闲超时: 每次等下一块最多 `idle`, 超时则产出 `IdleTimeout` 错误并终止。
fn with_idle_timeout(
    inner: Pin<Box<dyn Stream<Item = Result<SamplingEvent, ApiError>> + Send>>,
    idle: Duration,
) -> impl Stream<Item = Result<SamplingEvent, ApiError>> + Send {
    async_stream::stream! {
        futures_util::pin_mut!(inner);
        loop {
            match tokio::time::timeout(idle, inner.next()).await {
                Ok(Some(item)) => yield item,
                Ok(None) => return, // 正常结束
                Err(_) => {
                    yield Err(ApiError::IdleTimeout(idle));
                    return;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Model;
    use wiremock::matchers::{header, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    /// 端到端 (三层): mock SSE 返回三块, sample 应聚合成 TextDelta×2 + Done。
    /// 验证归一化未破坏 chat_completions 行为 (替代旧 api.rs 的 chat_stream_aggregates_events)。
    #[tokio::test]
    async fn sample_aggregates_events_through_three_layers() {
        let server = MockServer::start().await;
        let body = concat!(
            "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"},\"index\":0}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"content\":\" world\"},\"index\":0}]}\n\n",
            "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
            "data: [DONE]\n\n",
        );
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .and(header("authorization", "Bearer sk-test"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/event-stream")
                    .set_body_string(body),
            )
            .mount(&server)
            .await;

        let handle = super::super::build(
            "sk-test".into(),
            server.uri(),
            &Model::deepseek_v4_flash(),
            SamplerConfig::default(),
        );
        let mut stream = handle
            .sample(SamplingRequest {
                model: Model::deepseek_v4_flash(),
                messages: vec![crate::api::Message::User { content: "ping".into() }],
                tools_schema: vec![],
            })
            .await
            .unwrap();

        let mut events = Vec::new();
        while let Some(ev) = stream.next().await {
            events.push(ev.unwrap());
        }
        assert!(matches!(&events[0], SamplingEvent::TextDelta(s) if s == "hi"));
        assert!(matches!(&events[1], SamplingEvent::TextDelta(s) if s == " world"));
        assert!(matches!(events.last().unwrap(), SamplingEvent::Done { .. }));
    }

    /// 429 一次后成功: 验证重试。
    #[tokio::test]
    async fn retries_on_rate_limit() {
        let server = MockServer::start().await;
        // 第一次 429, 第二次成功。
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(429))
            .up_to_n_times(1)
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/event-stream")
                    .set_body_string(
                        "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"},\"index\":0}]}\n\n\
                         data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n\
                         data: [DONE]\n\n",
                    ),
            )
            .mount(&server)
            .await;

        let handle = super::super::build(
            "sk-test".into(),
            server.uri(),
            &Model::deepseek_v4_flash(),
            SamplerConfig {
                max_retries: 2,
                idle_timeout: Duration::from_secs(5),
            },
        );
        let mut stream = handle
            .sample(SamplingRequest {
                model: Model::deepseek_v4_flash(),
                messages: vec![],
                tools_schema: vec![],
            })
            .await
            .expect("重试后应成功");
        let mut got_text = false;
        while let Some(ev) = stream.next().await {
            if let Ok(SamplingEvent::TextDelta(s)) = ev {
                if s == "ok" {
                    got_text = true;
                }
            }
        }
        assert!(got_text, "重试后应拿到内容");
    }
}

