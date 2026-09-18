//! 配置层 — Model 矩阵 + Config 加载.
//!
//! 子系统 4 起, `Model` 从「两个 DeepSeek 字面量 enum」升级为
//! **provider + dialect + api_id 的 struct** —— 这是「多方言归一」的地基:
//! 一个 Model 同时携带「哪个厂商 / 哪套 wire 协议 / 请求体里的 model 字段」三要素。
//!
//! ⚠️ 这有意推翻了项目早期的 DeepSeek-only 约束 (见 spec §1.2)。

use crate::errors::AppError;
use serde::{Deserialize, Serialize};

/// LLM 厂商 —— 决定 base_url 默认值、鉴权风格、以及默认方言。
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Provider {
    DeepSeek,
    OpenAI,
    Anthropic,
}

/// API 方言 —— 决定 wire 协议 (sampler 的 Layer 2 选哪个适配器)。
///
/// 学习点: 「厂商」与「方言」是正交的两件事 —— 例如 DeepSeek 和 OpenAI 都说
///         chat_completions 方言, 而 OpenAI 自己还另有 responses 方言。分开建模,
///         新增组合只是矩阵里加一行。
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Dialect {
    ChatCompletions,
    /// OpenAI responses 方言。方言适配器待多方言 task 实现 (见 sampler/dialect)。
    #[allow(dead_code)]
    Responses,
    AnthropicMessages,
}

impl Provider {
    /// 该厂商的默认方言 (裸 "provider/model" 解析时用)。
    fn default_dialect(self) -> Dialect {
        match self {
            Provider::DeepSeek | Provider::OpenAI => Dialect::ChatCompletions,
            Provider::Anthropic => Dialect::AnthropicMessages,
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Provider::DeepSeek => "deepseek",
            Provider::OpenAI => "openai",
            Provider::Anthropic => "anthropic",
        }
    }

    fn parse(s: &str) -> Option<Self> {
        match s {
            "deepseek" => Some(Provider::DeepSeek),
            "openai" => Some(Provider::OpenAI),
            "anthropic" => Some(Provider::Anthropic),
            _ => None,
        }
    }
}

/// 一个模型 = 厂商 + 方言 + 请求体里的 model 字段字符串。
///
/// 学习点: 从 `Copy` enum 变成 `Clone` struct (因为 `api_id: String`)。这意味着
///         之前依赖 Copy 隐式复制的地方现在要显式 `.clone()` 或借 `&Model` ——
///         编译器会逐个点出来, 是理解「Copy vs Clone」语义的好练习。
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Model {
    pub provider: Provider,
    pub dialect: Dialect,
    /// 塞进请求 body 的 model 字段 (例如 "deepseek-v4-flash")。
    pub api_id: String,
}

impl Model {
    /// 内置: DeepSeek V4 flash (默认档, 便宜快)。
    pub fn deepseek_v4_flash() -> Self {
        Self {
            provider: Provider::DeepSeek,
            dialect: Dialect::ChatCompletions,
            api_id: "deepseek-v4-flash".into(),
        }
    }

    /// 内置: DeepSeek V4 pro (更强多步推理)。
    pub fn deepseek_v4_pro() -> Self {
        Self {
            provider: Provider::DeepSeek,
            dialect: Dialect::ChatCompletions,
            api_id: "deepseek-v4-pro".into(),
        }
    }

    /// 返回请求 body 里的 model 字段值。
    pub fn api_id(&self) -> &str {
        &self.api_id
    }

    /// 从配置字符串解析:
    ///   - "deepseek-v4-flash" / "deepseek-v4-pro" → 内置两档;
    ///   - "provider/model-id" (provider ∈ deepseek|openai|anthropic) → 用厂商默认方言;
    ///   - 其它 → 报错。
    ///
    /// 学习点: 保留裸字面量兼容旧配置, 同时用 "provider/id" 语法为多厂商开路。
    pub fn parse(s: &str) -> Result<Self, AppError> {
        match s {
            "deepseek-v4-flash" => return Ok(Self::deepseek_v4_flash()),
            "deepseek-v4-pro" => return Ok(Self::deepseek_v4_pro()),
            _ => {}
        }
        if let Some((prov, id)) = s.split_once('/') {
            if let Some(provider) = Provider::parse(prov) {
                if !id.is_empty() {
                    return Ok(Self {
                        provider,
                        dialect: provider.default_dialect(),
                        api_id: id.to_string(),
                    });
                }
            }
        }
        Err(AppError::Config(format!(
            "非法 model {s:?}. 合法形式: deepseek-v4-flash | deepseek-v4-pro | \
             <provider>/<model-id> (provider ∈ deepseek|openai|anthropic)"
        )))
    }

    /// 渲染回配置字符串 (持久化用)。内置两档回到裸字面量, 其余为 "provider/id"。
    fn render(&self) -> String {
        if self.provider == Provider::DeepSeek && self.dialect == Dialect::ChatCompletions {
            self.api_id.clone()
        } else {
            format!("{}/{}", self.provider.as_str(), self.api_id)
        }
    }
}

impl Default for Model {
    /// 默认 flash: 学习项目高频迭代, 选便宜的更友好.
    fn default() -> Self {
        Self::deepseek_v4_flash()
    }
}

// 学习点: 手写 Serialize/Deserialize 让 Model 在 JSON 里是一个**字符串** (而非 3 字段对象),
//         这样 session 的 updates.jsonl 里 `"model":"deepseek-v4-flash"` 保持可读,
//         且解析走同一份 `parse` 校验逻辑。
impl Serialize for Model {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(&self.render())
    }
}

impl<'de> Deserialize<'de> for Model {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let s = String::deserialize(d)?;
        Model::parse(&s).map_err(serde::de::Error::custom)
    }
}

// ──────────────────────────────────────────────────────────────────────────
//  Config 主体 (Task 4)
// ──────────────────────────────────────────────────────────────────────────

use std::path::PathBuf;

/// 加载完成的运行时配置. 启动一次填一次, 之后只读.
#[derive(Clone, Debug)]
pub struct Config {
    pub api_key: String,
    pub base_url: String,
    pub model: Model,
    pub max_tool_output_bytes: usize,
    pub session_dir: PathBuf,
}

impl Config {
    /// 完整的多源加载: env (含 .env) > config.toml > 默认.
    /// CLI 覆盖由 main.rs 在拿到本对象后再做 (clap 已解析 Cli, 优先级最高).
    ///
    /// 学习点: 我们让 .env 通过 dotenvy 提前注入到 std::env, 这样下游逻辑只需要读 std::env.
    pub fn load() -> Result<Self, AppError> {
        // 优先尝试 cwd/.env, 再尝试 ~/.config/edocrs/.env. 都缺失不报错.
        let _ = dotenvy::dotenv();
        if let Some(home) = dirs::config_dir() {
            let p = home.join("edocrs/.env");
            let _ = dotenvy::from_path(&p);
        }

        // 读 config.toml (可选)
        let toml_cfg = Self::read_config_toml().unwrap_or_default();
        Self::merge(toml_cfg)
    }

    /// 测试专用: 只看 env 变量, 不动 .env / config.toml.
    /// 学习点: 在测试里隔离副作用源.
    #[cfg(test)]
    pub fn load_from_env_only() -> Result<Self, AppError> {
        Self::merge(TomlConfig::default())
    }

    fn merge(toml_cfg: TomlConfig) -> Result<Self, AppError> {
        // env 优先于 toml
        let api_key = std::env::var("DEEPSEEK_API_KEY")
            .ok()
            .or(toml_cfg.api_key)
            .ok_or_else(|| AppError::Config(
                "缺少 api_key. 请通过 --api-key, DEEPSEEK_API_KEY 环境变量, 或 config.toml 提供.".into()
            ))?;

        let base_url = std::env::var("DEEPSEEK_BASE_URL")
            .ok()
            .or(toml_cfg.base_url)
            .unwrap_or_else(|| "https://api.deepseek.com".to_string());

        let model_str = std::env::var("EDOCRS_MODEL")
            .ok()
            .or(toml_cfg.model);
        let model: Model = match model_str.as_deref() {
            None => Model::default(),
            Some(s) => Model::parse(s)?,
        };

        let max_tool_output_bytes = toml_cfg.max_tool_output_bytes.unwrap_or(50_000);

        let session_dir = toml_cfg.session_dir
            .map(expand_tilde)
            .unwrap_or_else(default_session_dir);

        Ok(Self { api_key, base_url, model, max_tool_output_bytes, session_dir })
    }

    fn read_config_toml() -> Result<TomlConfig, AppError> {
        let Some(cfg_dir) = dirs::config_dir() else {
            return Ok(TomlConfig::default());
        };
        let path = cfg_dir.join("edocrs/config.toml");
        if !path.exists() {
            return Ok(TomlConfig::default());
        }
        let text = std::fs::read_to_string(&path)?;
        let parsed: TomlConfig = toml::from_str(&text)
            .map_err(|e| AppError::Config(format!("解析 config.toml 失败: {e}")))?;
        Ok(parsed)
    }
}

/// 内部用: 把 ~/... 这种字面量展开成绝对路径.
fn expand_tilde(s: String) -> PathBuf {
    if let Some(stripped) = s.strip_prefix("~/") {
        if let Some(home) = dirs::home_dir() {
            return home.join(stripped);
        }
    }
    PathBuf::from(s)
}

fn default_session_dir() -> PathBuf {
    if let Some(dir) = dirs::data_local_dir() {
        return dir.join("edocrs/sessions");
    }
    PathBuf::from(".edocrs/sessions")
}

#[derive(Debug, Default, Deserialize)]
struct TomlConfig {
    api_key: Option<String>,
    base_url: Option<String>,
    model: Option<String>,
    max_tool_output_bytes: Option<usize>,
    session_dir: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Model 序列化必须输出可读的字符串字面量 (内置两档回到裸名).
    #[test]
    fn model_serializes_to_kebab_case() {
        let json_flash = serde_json::to_string(&Model::deepseek_v4_flash()).unwrap();
        assert_eq!(json_flash, "\"deepseek-v4-flash\"");
        let json_pro = serde_json::to_string(&Model::deepseek_v4_pro()).unwrap();
        assert_eq!(json_pro, "\"deepseek-v4-pro\"");
    }

    /// Model::parse 接受内置两档 + provider/id 语法, 拒绝纯垃圾串.
    #[test]
    fn model_parse_accepts_builtins_and_provider_syntax() {
        assert_eq!(Model::parse("deepseek-v4-flash").unwrap(), Model::deepseek_v4_flash());
        assert_eq!(Model::parse("deepseek-v4-pro").unwrap(), Model::deepseek_v4_pro());
        // provider/id 语法: OpenAI 走 chat_completions, Anthropic 走 messages。
        let gpt = Model::parse("openai/gpt-4o").unwrap();
        assert_eq!(gpt.provider, Provider::OpenAI);
        assert_eq!(gpt.dialect, Dialect::ChatCompletions);
        assert_eq!(gpt.api_id(), "gpt-4o");
        let claude = Model::parse("anthropic/claude-sonnet-4").unwrap();
        assert_eq!(claude.dialect, Dialect::AnthropicMessages);
        // 垃圾串报错。
        assert!(Model::parse("gpt-4").is_err());
        assert!(Model::parse("bogus/").is_err());
        assert!(Model::parse("nosuchprovider/x").is_err());
    }

    /// serde 往返: 反序列化走 parse, 非法值报错.
    #[test]
    fn model_deserialize_roundtrip_and_rejects_unknown() {
        assert!(serde_json::from_str::<Model>("\"deepseek-v4-flash\"").is_ok());
        assert!(serde_json::from_str::<Model>("\"anthropic/claude-3\"").is_ok());
        assert!(serde_json::from_str::<Model>("\"gpt-4\"").is_err());
        // provider/id 往返: 序列化再反序列化应相等。
        let m = Model::parse("openai/gpt-4o").unwrap();
        let s = serde_json::to_string(&m).unwrap();
        assert_eq!(serde_json::from_str::<Model>(&s).unwrap(), m);
    }

    /// api_id() 返回请求 body 用的原始字面量.
    #[test]
    fn api_id_returns_lowercase_kebab() {
        assert_eq!(Model::deepseek_v4_flash().api_id(), "deepseek-v4-flash");
        assert_eq!(Model::deepseek_v4_pro().api_id(), "deepseek-v4-pro");
    }

    // 学习点: 修改 std::env 在 Rust 2024 是 unsafe (因为多线程读会数据竞争).
    //         我们用一个全局 Mutex 把会动 env 的测试串行化, 避免相互踩.
    use std::sync::Mutex;
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    /// 把测试需要的 env 变量先全部清掉, 形成已知初始状态.
    fn reset_env() {
        for k in ["DEEPSEEK_API_KEY", "DEEPSEEK_BASE_URL", "EDOCRS_MODEL"] {
            // SAFETY: 测试串行执行 (ENV_LOCK), 此时无其他线程在读 env.
            unsafe { std::env::remove_var(k); }
        }
    }

    /// Config::load_from_env_only 应优先 env 变量.
    #[test]
    fn config_load_prefers_env() {
        let _g = ENV_LOCK.lock().unwrap();
        reset_env();
        // SAFETY: 测试串行执行 (见 ENV_LOCK).
        unsafe {
            std::env::set_var("DEEPSEEK_API_KEY", "sk-test-key-from-env");
            std::env::set_var("EDOCRS_MODEL", "deepseek-v4-pro");
        }
        let cfg = Config::load_from_env_only().unwrap();
        assert_eq!(cfg.api_key, "sk-test-key-from-env");
        assert_eq!(cfg.model, Model::deepseek_v4_pro());
        reset_env();
    }

    /// 缺 api_key 应当报 Config 错误.
    #[test]
    fn config_load_fails_without_api_key() {
        let _g = ENV_LOCK.lock().unwrap();
        reset_env();
        let res = Config::load_from_env_only();
        assert!(res.is_err());
        let msg = format!("{}", res.unwrap_err());
        assert!(msg.to_lowercase().contains("api_key") || msg.to_lowercase().contains("api key"));
    }

    /// model 字面量非法应报错.
    #[test]
    fn config_rejects_invalid_model() {
        let _g = ENV_LOCK.lock().unwrap();
        reset_env();
        // SAFETY: 测试串行执行.
        unsafe {
            std::env::set_var("DEEPSEEK_API_KEY", "sk-x");
            std::env::set_var("EDOCRS_MODEL", "gpt-4");
        }
        let res = Config::load_from_env_only();
        assert!(res.is_err());
        reset_env();
    }
}
