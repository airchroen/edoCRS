//! Sampler —— 对外的采样入口: 查 provider → 取 key → POST → 重试 → 逐块空闲超时。
//!
//! ```text
//!   Sampler::sample(req)
//!     ├─ registry.get(req.model.provider)       查 base_url / key 来源
//!     ├─ resolve_api_key                        缺 key 此刻才报错
//!     ├─ POST {base_url}/chat/completions       连接阶段失败 → 指数退避重试
//!     └─ bytes → SseSplitter → parse_chunk      每块之间套空闲超时
//! ```
//!
//! 取消: 调用方 drop 返回的流即可 —— 底层 HTTP 连接随之关闭 (reqwest 在 body 被 drop 时
//! 中止读取)。所以这里不需要 CancellationToken, agent 层用 `select!` 丢弃流就是 abort。

use crate::chat_completions::{self, SamplingEvent};
use crate::error::AiError;
use crate::message::Message;
use crate::model::Model;
use crate::provider::{ProviderRegistry, std_env};
use crate::sse::SseSplitter;
use futures_util::{Stream, StreamExt};
use std::pin::Pin;
use std::time::Duration;

/// 采样事件流。
///
/// 学习点: `Pin<Box<dyn Stream + Send>>` 是「类型擦除的异步流」的标准写法 ——
///   - `dyn Stream`: 不同来源 (真实 HTTP / 测试 mock) 的流统一成一个类型;
///   - `Box`: trait object 大小未知, 必须放堆上;
///   - `Pin`: 异步状态机可能自引用, 被 poll 后不能再移动, Pin 在类型层面保证这一点;
///   - `Send`: 允许流跨线程移动 (即使当前 agent 跑在单线程 LocalSet 上, 也不给将来设限)。
pub type SamplingStream = Pin<Box<dyn Stream<Item = Result<SamplingEvent, AiError>> + Send>>;

/// 一次采样请求。
#[derive(Clone, Debug)]
pub struct SamplingRequest {
    pub model: Model,
    pub messages: Vec<Message>,
    /// 工具的 JSON schema 列表 (`{"type":"function","function":{..}}`), 可为空。
    pub tools: Vec<serde_json::Value>,
}

/// 弹性策略配置。
#[derive(Clone, Copy, Debug)]
pub struct SamplerConfig {
    /// 连接阶段可重试错误的最大重试次数 (不含首次尝试)。
    pub max_retries: u32,
    /// 首次重试前的等待; 之后每次翻倍。
    pub backoff_base: Duration,
    /// 单次等待上限, 防止翻倍到离谱。
    pub backoff_max: Duration,
    /// 逐块空闲超时: 两块数据之间最长等待。
    pub idle_timeout: Duration,
    /// TCP 连接超时。
    pub connect_timeout: Duration,
}

impl Default for SamplerConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            backoff_base: Duration::from_millis(500),
            backoff_max: Duration::from_secs(8),
            // 推理模型「思考」时可能长时间不出 token, 给得宽一些。
            idle_timeout: Duration::from_secs(300),
            connect_timeout: Duration::from_secs(10),
        }
    }
}

impl SamplerConfig {
    /// 第 `attempt` 次重试 (从 1 开始) 前的等待时长: base * 2^(attempt-1), 封顶 max。
    fn backoff(&self, attempt: u32) -> Duration {
        // 学习点: `checked_shl` / `saturating_mul` 防溢出 —— 重试次数被配得很大时,
        //         `1 << 40` 之类会溢出 u32; 饱和运算让它安全地停在最大值。
        let factor = 1u32.checked_shl(attempt.saturating_sub(1)).unwrap_or(u32::MAX);
        self.backoff_base.saturating_mul(factor).min(self.backoff_max)
    }
}

/// 采样器。持有 HTTP 连接池与 provider 注册表; `Clone` 廉价 (reqwest::Client 内部是 Arc)。
#[derive(Clone)]
pub struct Sampler {
    http: reqwest::Client,
    registry: ProviderRegistry,
    cfg: SamplerConfig,
}

impl Sampler {
    pub fn new(registry: ProviderRegistry, cfg: SamplerConfig) -> Self {
        // 学习点: 只设 connect_timeout, **不设**整体 `timeout` —— 整体超时会把一次正常但
        //         很长的流式生成拦腰斩断。卡死检测交给下面的逐块空闲超时。
        let http = reqwest::Client::builder()
            .connect_timeout(cfg.connect_timeout)
            .build()
            .expect("构建 reqwest::Client 失败 (TLS 后端初始化异常)");
        Self { http, registry, cfg }
    }

    pub fn registry(&self) -> &ProviderRegistry {
        &self.registry
    }

    /// 发起一次流式采样。
    ///
    /// 返回 `Err` 表示「连一个事件都没拿到」(未知 provider、缺 key、重试耗尽、4xx...);
    /// 返回 `Ok(stream)` 后, 流中途的错误以 `Err` 项出现, 且之后流结束。
    ///
    /// 学习点: 「连接阶段可重试, 流开始后不重试」—— 重试要求幂等, 而一旦把 token 交给了
    ///         上层, 重发会导致重复输出。流中途断开交给上层决定 (报错或整轮重来)。
    pub async fn sample(&self, req: SamplingRequest) -> Result<SamplingStream, AiError> {
        let provider = self.registry.get(&req.model.provider)?;
        let api_key = provider.resolve_api_key(std_env)?;
        let url = format!("{}{}", provider.base_url, chat_completions::ENDPOINT);
        let body = chat_completions::build_body(&req.model.id, &req.messages, &req.tools);

        let mut attempt = 0;
        let resp = loop {
            match self.send_once(&url, api_key.as_deref(), &body).await {
                Ok(resp) => break resp,
                Err(e) if e.is_retryable() && attempt < self.cfg.max_retries => {
                    attempt += 1;
                    tokio::time::sleep(self.cfg.backoff(attempt)).await;
                }
                Err(e) => return Err(e),
            }
        };

        let events = event_stream(resp.bytes_stream());
        Ok(Box::pin(with_idle_timeout(events, self.cfg.idle_timeout)))
    }

    /// 单次 POST, 把非 2xx 状态码转成对应错误。
    async fn send_once(
        &self,
        url: &str,
        api_key: Option<&str>,
        body: &serde_json::Value,
    ) -> Result<reqwest::Response, AiError> {
        let mut rb = self.http.post(url).json(body);
        // 学习点: builder 模式下「可选地加一个头」—— 先绑定成 mut 变量, 条件满足再重新赋值。
        if let Some(k) = api_key {
            rb = rb.bearer_auth(k);
        }
        let resp = rb.send().await?; // reqwest::Error 经 #[from] 转成 AiError::Network
        let status = resp.status();
        if status == reqwest::StatusCode::TOO_MANY_REQUESTS {
            return Err(AiError::RateLimit);
        }
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            return Err(AiError::Http { status: status.as_u16(), body });
        }
        Ok(resp)
    }
}

/// 字节流 → 事件流: SSE 切分 + 逐 payload 解析。遇到 `[DONE]` 或首个错误即结束。
///
/// 学习点: `async_stream::stream!` 宏让我们用「写循环 + yield」的方式构造 Stream,
///         编译器把它展开成手写 `poll_next` 状态机 —— 比手动实现 Stream trait 直观得多。
fn event_stream(
    bytes: impl Stream<Item = Result<bytes::Bytes, reqwest::Error>> + Send + 'static,
) -> impl Stream<Item = Result<SamplingEvent, AiError>> + Send {
    async_stream::stream! {
        let mut splitter = SseSplitter::new();
        // 学习点: `pin_mut!` 把栈上的流钉住, 才能对它调用需要 `Pin<&mut Self>` 的 `.next()`。
        futures_util::pin_mut!(bytes);
        while let Some(chunk) = bytes.next().await {
            let chunk = match chunk {
                Ok(c) => c,
                Err(e) => {
                    yield Err(AiError::Network(e));
                    return;
                }
            };
            for payload in splitter.feed(&chunk) {
                if payload.trim() == chat_completions::DONE_SENTINEL {
                    return;
                }
                match chat_completions::parse_chunk(&payload) {
                    Ok(evs) => {
                        for ev in evs {
                            yield Ok(ev);
                        }
                    }
                    Err(e) => {
                        yield Err(e);
                        return;
                    }
                }
            }
        }
    }
}

/// 给事件流套逐块空闲超时: 每次「等下一项」最多 `idle`, 超时产出 `IdleTimeout` 并结束。
///
/// 学习点: `tokio::time::timeout(d, fut)` 把任意 future 包成「d 内完成或返回 Elapsed」;
///         每轮循环都新建一个 timeout, 相当于「每收到一块就重置计时器」。
fn with_idle_timeout<S>(inner: S, idle: Duration) -> impl Stream<Item = Result<SamplingEvent, AiError>> + Send
where
    S: Stream<Item = Result<SamplingEvent, AiError>> + Send,
{
    async_stream::stream! {
        futures_util::pin_mut!(inner);
        loop {
            match tokio::time::timeout(idle, inner.next()).await {
                Ok(Some(item)) => yield item,
                Ok(None) => return,
                Err(_elapsed) => {
                    yield Err(AiError::IdleTimeout(idle));
                    return;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chat_completions::{StopReason, Usage};
    use crate::provider::ProviderOverride;
    use wiremock::matchers::{header, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    /// 指向 mock server 的自定义 provider `mock`, key 用字面量 (不碰真实环境变量)。
    fn sampler_for(server: &MockServer, key: Option<&str>) -> Sampler {
        let mut reg = ProviderRegistry::builtin();
        let ov = ProviderOverride {
            base_url: Some(format!("{}/v1", server.uri())),
            api_key: key.map(String::from),
            ..Default::default()
        };
        reg.apply("mock", &ov).unwrap();
        let cfg = SamplerConfig {
            backoff_base: Duration::from_millis(1),
            idle_timeout: Duration::from_secs(5),
            ..Default::default()
        };
        Sampler::new(reg, cfg)
    }

    fn req() -> SamplingRequest {
        SamplingRequest {
            model: Model::new("mock", "m-1"),
            messages: vec![Message::User { content: "ping".into() }],
            tools: vec![],
        }
    }

    fn sse(body: &str) -> ResponseTemplate {
        ResponseTemplate::new(200)
            .insert_header("content-type", "text/event-stream")
            .set_body_string(body.to_string())
    }

    async fn collect(s: SamplingStream) -> Vec<Result<SamplingEvent, AiError>> {
        s.collect().await
    }

    /// 端到端: 文本 ×2 → finish → usage, 且 [DONE] 之后的内容被忽略。
    #[tokio::test]
    async fn streams_text_finish_and_usage() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .and(header("authorization", "Bearer sk-test"))
            .respond_with(sse(concat!(
                "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n",
                "data: {\"choices\":[{\"delta\":{\"content\":\" there\"}}]}\n\n",
                "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
                "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":3,\"completion_tokens\":2,\"total_tokens\":5}}\n\n",
                "data: [DONE]\n\n",
                "data: {\"choices\":[{\"delta\":{\"content\":\"ignored\"}}]}\n\n",
            )))
            .mount(&server)
            .await;

        let evs: Vec<_> = collect(sampler_for(&server, Some("sk-test")).sample(req()).await.unwrap())
            .await
            .into_iter()
            .map(Result::unwrap)
            .collect();
        assert_eq!(
            evs,
            vec![
                SamplingEvent::TextDelta("hi".into()),
                SamplingEvent::TextDelta(" there".into()),
                SamplingEvent::Finish(StopReason::Stop),
                SamplingEvent::Usage(Usage { prompt_tokens: 3, completion_tokens: 2, total_tokens: 5 }),
            ]
        );
    }

    /// 无 key 的 provider 不发 Authorization 头。
    #[tokio::test]
    async fn no_auth_header_without_key() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(|r: &wiremock::Request| {
                // 学习点: wiremock 的 responder 也可以是闭包, 按请求内容动态决定响应。
                if r.headers.contains_key("authorization") {
                    ResponseTemplate::new(400)
                } else {
                    sse("data: [DONE]\n\n")
                }
            })
            .mount(&server)
            .await;
        let s = sampler_for(&server, None).sample(req()).await.unwrap();
        assert!(collect(s).await.is_empty());
    }

    /// 429 与 503 各一次后成功: 验证退避重试。
    #[tokio::test]
    async fn retries_transient_errors() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(429))
            .up_to_n_times(1)
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(503))
            .up_to_n_times(1)
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .respond_with(sse("data: {\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n\ndata: [DONE]\n\n"))
            .mount(&server)
            .await;
        let s = sampler_for(&server, Some("k")).sample(req()).await.expect("重试后应成功");
        let evs = collect(s).await;
        assert!(matches!(&evs[0], Ok(SamplingEvent::TextDelta(t)) if t == "ok"));
    }

    /// 4xx 不重试, 直接带 body 报错。
    #[tokio::test]
    async fn client_error_is_not_retried() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(400).set_body_string("bad request"))
            .expect(1) // 学习点: expect(1) 让 MockServer 在 drop 时断言恰好被调用一次
            .mount(&server)
            .await;
        let err = sampler_for(&server, Some("k")).sample(req()).await.err().unwrap();
        assert!(matches!(err, AiError::Http { status: 400, ref body } if body == "bad request"));
    }

    #[tokio::test]
    async fn unknown_provider_errors_before_network() {
        let server = MockServer::start().await;
        let mut r = req();
        r.model = Model::new("nosuch", "x");
        let err = sampler_for(&server, None).sample(r).await.err().unwrap();
        assert!(matches!(err, AiError::UnknownProvider(_)));
    }

    /// 流迟迟不产出下一项: 空闲超时触发。
    /// 学习点: wiremock 的延迟作用于整个响应 (含响应头), 模拟不了「流中途卡住」,
    ///         所以直接对一个永不就绪的 `pending()` 流测试包装器本身。
    #[tokio::test]
    async fn idle_timeout_fires() {
        let slow = futures_util::stream::pending::<Result<SamplingEvent, AiError>>();
        let items: Vec<_> = with_idle_timeout(slow, Duration::from_millis(20)).collect().await;
        assert!(matches!(items.as_slice(), [Err(AiError::IdleTimeout(_))]));
    }

    #[test]
    fn backoff_doubles_and_caps() {
        let cfg = SamplerConfig {
            backoff_base: Duration::from_millis(100),
            backoff_max: Duration::from_millis(350),
            ..Default::default()
        };
        assert_eq!(cfg.backoff(1), Duration::from_millis(100));
        assert_eq!(cfg.backoff(2), Duration::from_millis(200));
        assert_eq!(cfg.backoff(3), Duration::from_millis(350));
        assert_eq!(cfg.backoff(99), Duration::from_millis(350));
    }
}
