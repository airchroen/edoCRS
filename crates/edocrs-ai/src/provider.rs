//! provider 注册表: provider id → 连接信息 (base_url + API key 来源)。
//!
//! 内置 4 个预设, 全部说 OpenAI Chat Completions 协议:
//!
//! | id         | base_url                       | key 环境变量         |
//! |------------|--------------------------------|----------------------|
//! | openai     | https://api.openai.com/v1      | OPENAI_API_KEY       |
//! | openrouter | https://openrouter.ai/api/v1   | OPENROUTER_API_KEY   |
//! | deepseek   | https://api.deepseek.com       | DEEPSEEK_API_KEY     |
//! | ollama     | http://localhost:11434/v1      | (无需 key)           |
//!
//! settings 里的 `providers{}` 可以覆盖预设的字段, 或追加全新的 provider
//! (新 provider 必须给出 `base_url`)。
//!
//! 设计要点: **不维护模型列表**。模型 id 原样透传给 provider, 合法与否由 provider 判定 ——
//! 新模型发布时无需改代码。

use crate::error::AiError;
use serde::Deserialize;
use std::collections::BTreeMap;

/// API key 从哪里来。
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ApiKeySource {
    /// 不需要 key (本地 ollama)。请求时不发 Authorization 头。
    None,
    /// 运行时从该环境变量读取。
    Env(String),
    /// 直接写在配置里的字面量 (不推荐, 但自建网关等场景方便)。
    Literal(String),
}

/// 一个 provider 的完整连接信息。
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProviderSpec {
    pub id: String,
    /// 不含末尾 `/`; 请求 URL = base_url + "/chat/completions"。
    pub base_url: String,
    pub api_key: ApiKeySource,
    /// 该 provider 下模型的默认上下文窗口 (token)。`None` = 未指定, 由上层用全局默认。
    /// 本 crate 不使用它 (压缩逻辑在上层), 放在这里只是让「provider 的全部设置」住在一处。
    pub context_window: Option<u64>,
}

/// settings 中对单个 provider 的覆盖项。所有字段可选: 覆盖预设时只写想改的字段。
///
/// 学习点: 该类型直接 derive `Deserialize`, 上层 settings 反序列化时可以原样嵌入 ——
///         「配置形状」与「运行时结构」(ProviderSpec) 分离, 合并逻辑集中在 [`ProviderRegistry::apply`]。
///
/// 学习点: `deny_unknown_fields` 让拼错的字段名 (如 `baseUrl`) 直接报错, 而不是被 serde
///         静默忽略后「配置怎么不生效」——配置文件这类人手写的输入, 严格比宽容更友好。
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Eq)]
#[serde(default, deny_unknown_fields)]
pub struct ProviderOverride {
    pub base_url: Option<String>,
    /// 从哪个环境变量读 key。
    pub api_key_env: Option<String>,
    /// key 字面量; 与 `api_key_env` 同时给出时优先。
    pub api_key: Option<String>,
    /// 该 provider 下模型的默认上下文窗口。
    pub context_window: Option<u64>,
}

/// provider 注册表。
///
/// 学习点: 用 `BTreeMap` 而非 `HashMap` —— 键有序, `ids()` 输出稳定 (错误提示、`/model`
///         列表、测试断言都不会因哈希随机化而抖动)。条目只有个位数, 性能差异可忽略。
#[derive(Clone, Debug)]
pub struct ProviderRegistry {
    providers: BTreeMap<String, ProviderSpec>,
}

impl ProviderRegistry {
    /// 仅含 4 个内置预设的注册表。
    pub fn builtin() -> Self {
        // 学习点: 局部辅助闭包把四行重复的结构体字面量压成一行一个, 表格一样易读。
        let spec = |id: &str, url: &str, key: ApiKeySource| {
            (
                id.to_string(),
                ProviderSpec { id: id.into(), base_url: url.into(), api_key: key, context_window: None },
            )
        };
        let env = |name: &str| ApiKeySource::Env(name.into());
        Self {
            providers: BTreeMap::from([
                spec("openai", "https://api.openai.com/v1", env("OPENAI_API_KEY")),
                spec("openrouter", "https://openrouter.ai/api/v1", env("OPENROUTER_API_KEY")),
                spec("deepseek", "https://api.deepseek.com", env("DEEPSEEK_API_KEY")),
                spec("ollama", "http://localhost:11434/v1", ApiKeySource::None),
            ]),
        }
    }

    /// 内置预设 + 一组覆盖 (通常来自 settings 的 `providers{}`)。
    ///
    /// 学习点: 参数写成 `impl IntoIterator<Item = (K, V)>` —— 调用方传 `BTreeMap`、`Vec`、
    ///         数组都行, 比写死 `&BTreeMap` 更通用; `&'a` 引用迭代器也能用 (见下方 Into 约束)。
    pub fn with_overrides<'a>(
        overrides: impl IntoIterator<Item = (&'a String, &'a ProviderOverride)>,
    ) -> Result<Self, AiError> {
        let mut reg = Self::builtin();
        for (id, ov) in overrides {
            reg.apply(id, ov)?;
        }
        Ok(reg)
    }

    /// 应用单个覆盖: 已有 provider 则逐字段覆盖; 否则新建 (必须有 base_url)。
    pub fn apply(&mut self, id: &str, ov: &ProviderOverride) -> Result<(), AiError> {
        if id.is_empty() || id.contains('/') {
            // `/` 是 `provider/model-id` 的分隔符, provider id 里不能有它。
            return Err(AiError::BadProvider(format!("provider id {id:?} 不能为空或含 '/'")));
        }
        // 学习点: `entry` API 一次查找同时处理「存在」与「不存在」两种情况,
        //         避免先 `get` 再 `insert` 的两次哈希/比较。
        use std::collections::btree_map::Entry;
        let spec = match self.providers.entry(id.to_string()) {
            Entry::Occupied(e) => e.into_mut(),
            Entry::Vacant(e) => {
                let Some(url) = &ov.base_url else {
                    return Err(AiError::BadProvider(format!(
                        "自定义 provider {id:?} 必须提供 base_url"
                    )));
                };
                e.insert(ProviderSpec {
                    id: id.to_string(),
                    base_url: url.clone(),
                    // 新 provider 默认无需 key; 下面若给了 key 配置会覆盖。
                    api_key: ApiKeySource::None,
                    context_window: None,
                })
            }
        };
        if let Some(url) = &ov.base_url {
            spec.base_url = url.clone();
        }
        // 字面量优先于环境变量名。
        if let Some(key) = &ov.api_key {
            spec.api_key = ApiKeySource::Literal(key.clone());
        } else if let Some(env) = &ov.api_key_env {
            spec.api_key = ApiKeySource::Env(env.clone());
        }
        if ov.context_window.is_some() {
            spec.context_window = ov.context_window;
        }
        // 统一去掉末尾 `/`, 拼 URL 时就不用再操心双斜杠。
        spec.base_url = spec.base_url.trim_end_matches('/').to_string();
        Ok(())
    }

    /// 按 id 查 provider。
    pub fn get(&self, id: &str) -> Result<&ProviderSpec, AiError> {
        self.providers.get(id).ok_or_else(|| AiError::UnknownProvider(id.to_string()))
    }

    /// 所有 provider id (有序)。
    pub fn ids(&self) -> impl Iterator<Item = &str> {
        // 学习点: 返回 `impl Iterator` 而不是 `Vec<&str>` —— 不分配, 调用方需要时自己 collect。
        self.providers.keys().map(String::as_str)
    }
}

impl ProviderSpec {
    /// 解析出实际的 API key。
    ///
    /// - `Ok(None)`: 该 provider 不需要 key;
    /// - `Err(MissingApiKey)`: 需要但环境变量没设 (或为空)。
    ///
    /// `env` 是「按名字取环境变量」的函数, 生产代码传 [`std_env`], 测试传假实现 ——
    ///
    /// 学习点: 把「读环境变量」抽成参数 (依赖注入) 后, 测试不必调用 `std::env::set_var`
    ///         (Rust 2024 里它是 unsafe, 且多线程测试会互相踩)。
    pub fn resolve_api_key(
        &self,
        env: impl Fn(&str) -> Option<String>,
    ) -> Result<Option<String>, AiError> {
        match &self.api_key {
            ApiKeySource::None => Ok(None),
            ApiKeySource::Literal(k) => Ok(Some(k.clone())),
            ApiKeySource::Env(name) => match env(name) {
                Some(k) if !k.trim().is_empty() => Ok(Some(k)),
                _ => Err(AiError::MissingApiKey { provider: self.id.clone(), env: name.clone() }),
            },
        }
    }
}

/// 生产用的环境变量读取函数 (可直接传给 [`ProviderSpec::resolve_api_key`])。
pub fn std_env(name: &str) -> Option<String> {
    std::env::var(name).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builtin_has_four_presets() {
        let reg = ProviderRegistry::builtin();
        assert_eq!(reg.ids().collect::<Vec<_>>(), ["deepseek", "ollama", "openai", "openrouter"]);
        assert_eq!(reg.get("deepseek").unwrap().base_url, "https://api.deepseek.com");
        assert_eq!(reg.get("ollama").unwrap().api_key, ApiKeySource::None);
        assert!(matches!(reg.get("nope"), Err(AiError::UnknownProvider(_))));
    }

    /// 覆盖预设只改给出的字段, 其余保持。
    #[test]
    fn override_preset_partially() {
        let mut reg = ProviderRegistry::builtin();
        let ov = ProviderOverride { base_url: Some("http://proxy/v1/".into()), ..Default::default() };
        reg.apply("openai", &ov).unwrap();
        let s = reg.get("openai").unwrap();
        assert_eq!(s.base_url, "http://proxy/v1"); // 末尾 / 被去掉
        assert_eq!(s.api_key, ApiKeySource::Env("OPENAI_API_KEY".into()));
    }

    #[test]
    fn custom_provider_requires_base_url() {
        let mut reg = ProviderRegistry::builtin();
        assert!(reg.apply("mine", &ProviderOverride::default()).is_err());
        let ov = ProviderOverride {
            base_url: Some("http://gw".into()),
            api_key_env: Some("MY_KEY".into()),
            ..Default::default()
        };
        reg.apply("mine", &ov).unwrap();
        assert_eq!(reg.get("mine").unwrap().api_key, ApiKeySource::Env("MY_KEY".into()));
        assert!(reg.apply("a/b", &ov).is_err());
    }

    #[test]
    fn context_window_override_is_recorded() {
        let mut reg = ProviderRegistry::builtin();
        assert_eq!(reg.get("ollama").unwrap().context_window, None);
        let ov = ProviderOverride { context_window: Some(32_000), ..Default::default() };
        reg.apply("ollama", &ov).unwrap();
        assert_eq!(reg.get("ollama").unwrap().context_window, Some(32_000));
    }

    #[test]
    fn literal_key_beats_env_name() {
        let mut reg = ProviderRegistry::builtin();
        let ov = ProviderOverride {
            api_key: Some("sk-lit".into()),
            api_key_env: Some("IGNORED".into()),
            ..Default::default()
        };
        reg.apply("deepseek", &ov).unwrap();
        let key = reg.get("deepseek").unwrap().resolve_api_key(|_| None).unwrap();
        assert_eq!(key.as_deref(), Some("sk-lit"));
    }

    #[test]
    fn resolve_key_from_env_or_error() {
        let reg = ProviderRegistry::builtin();
        let ds = reg.get("deepseek").unwrap();
        let fake = |n: &str| (n == "DEEPSEEK_API_KEY").then(|| "sk-env".to_string());
        assert_eq!(ds.resolve_api_key(fake).unwrap().as_deref(), Some("sk-env"));
        // 缺 key 才报错, 且错误里点名环境变量。
        let err = reg.get("openai").unwrap().resolve_api_key(fake).unwrap_err();
        assert!(err.to_string().contains("OPENAI_API_KEY"));
        // 无需 key 的 provider 永远 Ok(None)。
        assert_eq!(reg.get("ollama").unwrap().resolve_api_key(|_| None).unwrap(), None);
    }
}
