//! 测试公用辅助 (仅测试编译)。
//!
//! 学习点: 整个模块用 `#[cfg(test)]` 在 main.rs 里挂载, 生产二进制里不存在这些代码。

use edocrs_ai::{Model, ProviderOverride, ProviderRegistry, Sampler, SamplerConfig};
use std::time::Duration;

/// 测试用模型: 指向下面 `test_sampler` 注册的 `mock` provider。
pub fn test_model() -> Model {
    Model::new("mock", "test-model")
}

/// 指向 wiremock 服务器的 Sampler。
///
/// mock 服务器约定监听 `/v1/chat/completions`, 所以 base_url = `{uri}/v1`。
/// key 用字面量, 不碰真实环境变量; 退避缩到 1ms 让重试类测试不拖慢。
pub fn test_sampler(server_uri: impl AsRef<str>) -> Sampler {
    let mut reg = ProviderRegistry::builtin();
    reg.apply(
        "mock",
        &ProviderOverride {
            base_url: Some(format!("{}/v1", server_uri.as_ref())),
            api_key: Some("sk-test".into()),
            ..Default::default()
        },
    )
    .expect("注册 mock provider");
    Sampler::new(
        reg,
        SamplerConfig { backoff_base: Duration::from_millis(1), ..Default::default() },
    )
}
