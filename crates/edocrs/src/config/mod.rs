//! 配置层: 把多来源的设置合并成一份只读的运行时 [`Config`]。
//!
//! 优先级 (高 → 低):
//! ```text
//!   CLI 参数  >  环境变量 (EDOCRS_MODEL)  >  项目 settings.json  >  全局 settings.json  >  内置默认
//! ```
//! - 全局: `<agent_dir>/settings.json`, agent_dir 默认 `~/.edocrs/agent` (`EDOCRS_AGENT_DIR` 可覆盖);
//! - 项目: `<cwd>/.edocrs/settings.json`, **仅在项目被信任时读取** (项目文件可能来自不可信仓库,
//!   而 settings 能改 base_url / 指定 key —— 不加把关就是把 key 送给别人的服务器)。
//! - 各 provider 的 API key **不在这里读**: sampler 请求时才按需读环境变量, 所以没用到的
//!   provider 缺 key 不会影响启动。
//!
//! 学习点: 「读文件 / 读环境变量」(有副作用) 与「合并 + 校验」(纯计算) 分成两个函数 ——
//!         [`Config::load`] 负责前者, [`Config::resolve`] 是纯函数, 测试只需喂数据。

pub mod settings;

use crate::errors::AppError;
use edocrs_ai::{Model, ProviderRegistry};
pub use settings::{PermissionMode, Settings};
use std::path::{Path, PathBuf};

/// 全局默认上下文窗口 (token)。
pub const DEFAULT_CONTEXT_WINDOW: u64 = 128_000;
pub const DEFAULT_MAX_OUTPUT_BYTES: usize = 50_000;
pub const DEFAULT_BASH_TIMEOUT_SECS: u64 = 30;
pub const DEFAULT_RESERVE_TOKENS: u64 = 16_384;
pub const DEFAULT_KEEP_RECENT_TOKENS: u64 = 20_000;

/// 来自 CLI 的覆盖项 (最高优先级)。
#[derive(Clone, Debug, Default)]
pub struct CliOverrides {
    pub model: Option<String>,
    pub mode: Option<PermissionMode>,
}

/// 解析完成的压缩参数。
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CompactionConfig {
    pub enabled: bool,
    pub reserve_tokens: u64,
    pub keep_recent_tokens: u64,
}

/// 运行时配置。启动时构造一次, 之后只读。
#[derive(Clone, Debug)]
pub struct Config {
    /// 当前默认模型 (运行中可经 `/model` 切换, 切换不回写这里)。
    pub model: Model,
    pub permission_mode: PermissionMode,
    /// 内置预设 + settings 覆盖后的 provider 注册表。
    pub registry: ProviderRegistry,
    pub max_output_bytes: usize,
    pub bash_timeout_secs: u64,
    /// `~/.edocrs/agent` (或 `EDOCRS_AGENT_DIR`)。
    pub agent_dir: PathBuf,
    /// 会话根目录。
    pub session_dir: PathBuf,
    // 下面两项保留合并后的原始设置, 供 `context_window_for` / `compaction_for` 按模型查询。
    settings: Settings,
}

impl Config {
    /// 完整加载: `.env` → 全局 settings → (可选) 项目 settings → 合并校验。
    ///
    /// `project_trusted` 为 false 时完全忽略 `<cwd>/.edocrs/`。
    pub fn load(cwd: &Path, project_trusted: bool, cli: CliOverrides) -> Result<Self, AppError> {
        // 学习点: `.env` 通过 dotenvy 提前注入进程环境, 下游只需读 std::env。
        //         先读 cwd 的 (可能设置 EDOCRS_AGENT_DIR), 再定位 agent_dir 读它自己的。
        let _ = dotenvy::from_path(cwd.join(".env"));
        let agent_dir = agent_dir_from_env();
        let _ = dotenvy::from_path(agent_dir.join(".env"));

        let global = read_layer(&agent_dir.join("settings.json"))?;
        let project = if project_trusted {
            read_layer(&cwd.join(".edocrs/settings.json"))?
        } else {
            Settings::default()
        };
        let env_model = std::env::var("EDOCRS_MODEL").ok().filter(|s| !s.trim().is_empty());
        Self::resolve(global, project, env_model, cli, agent_dir)
    }

    /// 纯函数: 合并各层 + 校验 + 填默认值。
    pub fn resolve(
        global: Settings,
        project: Settings,
        env_model: Option<String>,
        cli: CliOverrides,
        agent_dir: PathBuf,
    ) -> Result<Self, AppError> {
        let settings = global.merge(project);

        // 模型: CLI > 环境变量 > settings。没有内置默认 —— 本项目不维护模型列表,
        // 替用户猜一个「默认模型」只会在 provider 换代后悄悄过期。
        let model_str = cli
            .model
            .or(env_model)
            .or_else(|| settings.default_model.clone())
            .ok_or_else(|| {
                AppError::Config(
                    "未指定模型: 请用 --model、环境变量 EDOCRS_MODEL, 或 settings.json 的 \
                     default_model (形如 deepseek/deepseek-v4-flash)"
                        .into(),
                )
            })?;
        let model = Model::parse(&model_str).map_err(|e| AppError::Config(e.to_string()))?;

        let registry = ProviderRegistry::with_overrides(&settings.providers)
            .map_err(|e| AppError::Config(e.to_string()))?;
        // 提前校验默认模型的 provider 存在 (只查注册表, 不查 key)。
        registry.get(&model.provider).map_err(|e| AppError::Config(e.to_string()))?;

        Ok(Self {
            model,
            permission_mode: cli.mode.or(settings.permission_mode).unwrap_or_default(),
            registry,
            max_output_bytes: settings.tools.max_output_bytes.unwrap_or(DEFAULT_MAX_OUTPUT_BYTES),
            bash_timeout_secs: settings.tools.bash_timeout_secs.unwrap_or(DEFAULT_BASH_TIMEOUT_SECS),
            session_dir: agent_dir.join("sessions"),
            agent_dir,
            settings,
        })
    }

    /// 某模型的上下文窗口: `models[p/id]` > `providers[p]` > 全局 `context_window` > 128000。
    pub fn context_window_for(&self, model: &Model) -> u64 {
        self.settings
            .models
            .get(&model.to_string())
            .and_then(|m| m.context_window)
            .or_else(|| self.registry.get(&model.provider).ok().and_then(|p| p.context_window))
            .or(self.settings.context_window)
            .unwrap_or(DEFAULT_CONTEXT_WINDOW)
    }

    /// 某模型生效的压缩参数: `compaction.model_overrides[p/id]` > `compaction` > 内置默认。
    pub fn compaction_for(&self, model: &Model) -> CompactionConfig {
        let c = &self.settings.compaction;
        let o = c.model_overrides.get(&model.to_string()).copied().unwrap_or_default();
        CompactionConfig {
            enabled: o.enabled.or(c.enabled).unwrap_or(true),
            reserve_tokens: o.reserve_tokens.or(c.reserve_tokens).unwrap_or(DEFAULT_RESERVE_TOKENS),
            keep_recent_tokens: o
                .keep_recent_tokens
                .or(c.keep_recent_tokens)
                .unwrap_or(DEFAULT_KEEP_RECENT_TOKENS),
        }
    }
}

/// agent_dir: `EDOCRS_AGENT_DIR` > `~/.edocrs/agent`。
fn agent_dir_from_env() -> PathBuf {
    if let Some(d) = std::env::var_os("EDOCRS_AGENT_DIR").filter(|d| !d.is_empty()) {
        return PathBuf::from(d);
    }
    // 拿不到 home 时退回相对路径, 至少不会 panic。
    dirs::home_dir().unwrap_or_default().join(".edocrs/agent")
}

/// 读一层 settings.json; 文件不存在 = 空层, 解析失败 = 带文件路径的报错。
fn read_layer(path: &Path) -> Result<Settings, AppError> {
    let text = match std::fs::read_to_string(path) {
        Ok(t) => t,
        // 学习点: 只把 NotFound 当「没配置」; 权限不足等其它 IO 错误应当暴露给用户,
        //         否则会出现「配置明明在却不生效」的怪事。
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Settings::default()),
        Err(e) => return Err(AppError::Config(format!("读取 {} 失败: {e}", path.display()))),
    };
    Settings::parse(&text)
        .map_err(|e| AppError::Config(format!("解析 {} 失败: {e}", path.display())))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(json: &str) -> Settings {
        Settings::parse(json).unwrap()
    }

    fn resolve(global: &str, project: &str, env: Option<&str>, cli: CliOverrides) -> Result<Config, AppError> {
        Config::resolve(s(global), s(project), env.map(String::from), cli, PathBuf::from("/agent"))
    }

    #[test]
    fn model_priority_cli_env_project_global() {
        let g = r#"{"default_model":"openai/g"}"#;
        let p = r#"{"default_model":"openai/p"}"#;
        let cli = |m: &str| CliOverrides { model: Some(m.into()), ..Default::default() };
        assert_eq!(resolve(g, "{}", None, Default::default()).unwrap().model.id, "g");
        assert_eq!(resolve(g, p, None, Default::default()).unwrap().model.id, "p");
        assert_eq!(resolve(g, p, Some("openai/e"), Default::default()).unwrap().model.id, "e");
        assert_eq!(resolve(g, p, Some("openai/e"), cli("openai/c")).unwrap().model.id, "c");
    }

    #[test]
    fn missing_model_is_a_helpful_error() {
        let err = resolve("{}", "{}", None, Default::default()).unwrap_err().to_string();
        assert!(err.contains("--model") && err.contains("default_model"), "{err}");
    }

    #[test]
    fn malformed_or_unknown_provider_model_rejected() {
        assert!(resolve("{}", "{}", Some("gpt-4o"), Default::default()).is_err());
        let err = resolve("{}", "{}", Some("nosuch/x"), Default::default()).unwrap_err().to_string();
        assert!(err.contains("nosuch"), "{err}");
    }

    /// 自定义 provider 出现在 settings 里, 模型就可以引用它。
    #[test]
    fn custom_provider_usable_as_model() {
        let g = r#"{"providers":{"gw":{"base_url":"http://gw/v1"}}}"#;
        let c = resolve(g, "{}", Some("gw/some-model"), Default::default()).unwrap();
        assert_eq!(c.registry.get("gw").unwrap().base_url, "http://gw/v1");
    }

    #[test]
    fn permission_mode_defaults_to_ask_and_cli_wins() {
        let base = resolve(r#"{}"#, "{}", Some("openai/m"), Default::default()).unwrap();
        assert_eq!(base.permission_mode, PermissionMode::Ask);
        let g = r#"{"permission_mode":"auto-edit"}"#;
        assert_eq!(
            resolve(g, "{}", Some("openai/m"), Default::default()).unwrap().permission_mode,
            PermissionMode::AutoEdit
        );
        let cli = CliOverrides { mode: Some(PermissionMode::Ask), ..Default::default() };
        assert_eq!(resolve(g, "{}", Some("openai/m"), cli).unwrap().permission_mode, PermissionMode::Ask);
    }

    #[test]
    fn defaults_and_paths() {
        let c = resolve("{}", "{}", Some("openai/m"), Default::default()).unwrap();
        assert_eq!(c.max_output_bytes, 50_000);
        assert_eq!(c.bash_timeout_secs, 30);
        assert_eq!(c.session_dir, PathBuf::from("/agent/sessions"));
        assert_eq!(c.context_window_for(&c.model), 128_000);
        assert_eq!(
            c.compaction_for(&c.model),
            CompactionConfig { enabled: true, reserve_tokens: 16_384, keep_recent_tokens: 20_000 }
        );
    }

    /// context_window 查找链: models > provider > 全局 > 默认。
    #[test]
    fn context_window_lookup_chain() {
        let g = r#"{
            "context_window": 50000,
            "providers": {"ollama": {"context_window": 32000}},
            "models": {"openai/big": {"context_window": 1000000}}
        }"#;
        let c = resolve(g, "{}", Some("openai/small"), Default::default()).unwrap();
        assert_eq!(c.context_window_for(&Model::new("openai", "big")), 1_000_000);
        assert_eq!(c.context_window_for(&Model::new("ollama", "x")), 32_000);
        assert_eq!(c.context_window_for(&Model::new("openai", "small")), 50_000);
    }

    #[test]
    fn compaction_model_override_wins() {
        let g = r#"{"compaction":{"reserve_tokens":1000,"model_overrides":{"openai/m":{"reserve_tokens":2000,"enabled":false}}}}"#;
        let c = resolve(g, "{}", Some("openai/m"), Default::default()).unwrap();
        let hit = c.compaction_for(&Model::new("openai", "m"));
        assert_eq!((hit.enabled, hit.reserve_tokens), (false, 2000));
        let miss = c.compaction_for(&Model::new("openai", "other"));
        assert_eq!((miss.enabled, miss.reserve_tokens), (true, 1000));
    }

    #[test]
    fn read_layer_missing_file_is_empty_but_bad_json_names_the_file() {
        let dir = tempfile::tempdir().unwrap();
        assert_eq!(read_layer(&dir.path().join("nope.json")).unwrap(), Settings::default());
        let bad = dir.path().join("settings.json");
        std::fs::write(&bad, "{ not json").unwrap();
        let err = read_layer(&bad).unwrap_err().to_string();
        assert!(err.contains("settings.json"), "{err}");
    }
}
