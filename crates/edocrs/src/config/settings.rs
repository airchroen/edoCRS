//! `settings.json` 的「文件形状」—— 一层配置文件反序列化出来的原始数据, 尚未合并、尚无默认值。
//!
//! 文件格式是 JSONC (带 `//` `/* */` 注释与尾逗号的 JSON), 用 `json5` 解析 (JSON5 是 JSONC 的超集)。
//!
//! 设计要点:
//!   - **所有字段都是 `Option` / 空 map**: 「用户没写」与「用户写了默认值」必须能区分,
//!     否则项目层无法表达「我没意见, 沿用全局的」。
//!   - **`deny_unknown_fields`**: 拼错字段名直接报错 (带行列号), 而不是静默不生效。
//!   - 合并规则 (`merge`): 高优先级层的**有值字段**覆盖低优先级层; map 按 key 合并, 同 key 再逐字段合并。

use edocrs_ai::ProviderOverride;
use serde::Deserialize;
use std::collections::BTreeMap;

/// 权限模式 (spec §5)。只有两种, 没有 yolo。
///
/// 学习点: `rename_all = "kebab-case"` 把变体名 `AutoEdit` 映射为 JSON 里的 `"auto-edit"`。
#[derive(Copy, Clone, Debug, Default, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "kebab-case")]
pub enum PermissionMode {
    /// write / edit / bash 都询问 (默认)。
    #[default]
    Ask,
    /// write / edit 自动放行; bash 仍然询问。
    AutoEdit,
}

/// 一层 settings.json。
#[derive(Clone, Debug, Default, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct Settings {
    /// 默认模型, 形如 `deepseek/deepseek-v4-flash`。这里保留字符串, 由 `Config` 统一解析并报错
    /// (这样错误信息能指出是哪一层配置写错的)。
    pub default_model: Option<String>,
    pub permission_mode: Option<PermissionMode>,
    /// 全局默认上下文窗口 (token)。
    pub context_window: Option<u64>,
    /// 追加或覆盖 provider。key = provider id。
    pub providers: BTreeMap<String, ProviderOverride>,
    /// 按模型覆盖。key = `provider/model-id`。
    pub models: BTreeMap<String, ModelSettings>,
    pub compaction: CompactionSettings,
    pub tools: ToolsSettings,
}

/// `models{"p/id": {...}}` 的值。
#[derive(Clone, Debug, Default, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct ModelSettings {
    pub context_window: Option<u64>,
}

/// 压缩相关设置 (阶段 8 使用; 这里先把配置面定下来)。
#[derive(Clone, Debug, Default, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct CompactionSettings {
    pub enabled: Option<bool>,
    /// 给回复预留的 token 数: 上下文超过 `context_window - reserve_tokens` 即触发压缩。
    pub reserve_tokens: Option<u64>,
    /// 压缩后保留的最近上下文 token 数。
    pub keep_recent_tokens: Option<u64>,
    /// 按 `provider/model-id` 覆盖上面三项。
    pub model_overrides: BTreeMap<String, CompactionOverride>,
}

#[derive(Clone, Copy, Debug, Default, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct CompactionOverride {
    pub enabled: Option<bool>,
    pub reserve_tokens: Option<u64>,
    pub keep_recent_tokens: Option<u64>,
}

/// 工具相关设置。
#[derive(Clone, Copy, Debug, Default, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct ToolsSettings {
    /// 单次工具输出在框架层的截断上限 (字节)。
    pub max_output_bytes: Option<usize>,
    /// bash 工具的默认超时 (秒)。
    pub bash_timeout_secs: Option<u64>,
}

impl Settings {
    /// 用 JSONC 文本解析一层配置。空文件 / 纯注释视为「什么都没配」。
    ///
    /// 学习点: json5 对空输入会报 EOF 错误, 而「用户新建了个空 settings.json」是很自然的操作,
    ///         所以先判空。
    pub fn parse(text: &str) -> Result<Self, String> {
        if text.trim().is_empty() {
            return Ok(Self::default());
        }
        json5::from_str(text).map_err(|e| e.to_string())
    }

    /// 把 `over` (高优先级) 叠到 `self` (低优先级) 上, 返回合并结果。
    ///
    /// 学习点: 方法拿 `self` 的所有权 (而不是 `&self`) —— 合并本来就是「消耗两个输入产出一个输出」,
    ///         这样各字段可以直接 move 而不必 clone。
    pub fn merge(self, over: Settings) -> Settings {
        Settings {
            // `Option::or`: over 有值取 over, 否则退回 self。
            default_model: over.default_model.or(self.default_model),
            permission_mode: over.permission_mode.or(self.permission_mode),
            context_window: over.context_window.or(self.context_window),
            providers: merge_maps(self.providers, over.providers, merge_provider),
            models: merge_maps(self.models, over.models, |lo, hi| ModelSettings {
                context_window: hi.context_window.or(lo.context_window),
            }),
            compaction: CompactionSettings {
                enabled: over.compaction.enabled.or(self.compaction.enabled),
                reserve_tokens: over.compaction.reserve_tokens.or(self.compaction.reserve_tokens),
                keep_recent_tokens: over
                    .compaction
                    .keep_recent_tokens
                    .or(self.compaction.keep_recent_tokens),
                model_overrides: merge_maps(
                    self.compaction.model_overrides,
                    over.compaction.model_overrides,
                    |lo, hi| CompactionOverride {
                        enabled: hi.enabled.or(lo.enabled),
                        reserve_tokens: hi.reserve_tokens.or(lo.reserve_tokens),
                        keep_recent_tokens: hi.keep_recent_tokens.or(lo.keep_recent_tokens),
                    },
                ),
            },
            tools: ToolsSettings {
                max_output_bytes: over.tools.max_output_bytes.or(self.tools.max_output_bytes),
                bash_timeout_secs: over.tools.bash_timeout_secs.or(self.tools.bash_timeout_secs),
            },
        }
    }
}

/// 按 key 合并两个 map: 只在一边出现的 key 原样保留, 两边都有的用 `f(低, 高)` 合并。
///
/// 学习点: 泛型 `V` + 闭包参数 `F: FnOnce(V, V) -> V` 让「三种 map 的合并」共用一份骨架。
fn merge_maps<V>(
    mut base: BTreeMap<String, V>,
    over: BTreeMap<String, V>,
    f: impl Fn(V, V) -> V,
) -> BTreeMap<String, V> {
    for (k, hi) in over {
        // `remove` 先把低优先级的值拿出来 (所有权), 合并后再 insert 回去。
        let merged = match base.remove(&k) {
            Some(lo) => f(lo, hi),
            None => hi,
        };
        base.insert(k, merged);
    }
    base
}

fn merge_provider(lo: ProviderOverride, hi: ProviderOverride) -> ProviderOverride {
    // key 的两种写法互斥且字面量优先: 高优先级层只要写了任意一种, 就整体替换低优先级层的 key 设置,
    // 避免「项目层改了 api_key_env, 却被全局层的字面量 api_key 压过去」这种反直觉结果。
    //
    // 学习点: 先把判断结果存进 bool 再 move 字段 —— 在 `if` 条件里借用 `hi.api_key`、
    //         分支里又 move 它, 编译器会因「先 move 后借用」拒绝 (E0382)。
    let hi_sets_key = hi.api_key.is_some() || hi.api_key_env.is_some();
    let (api_key, api_key_env) = if hi_sets_key {
        (hi.api_key, hi.api_key_env)
    } else {
        (lo.api_key, lo.api_key_env)
    };
    ProviderOverride {
        base_url: hi.base_url.or(lo.base_url),
        api_key,
        api_key_env,
        context_window: hi.context_window.or(lo.context_window),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_jsonc_with_comments_and_trailing_commas() {
        let s = Settings::parse(
            r#"{
                // 默认模型
                "default_model": "openai/gpt-4o",
                "permission_mode": "auto-edit", /* 行内注释 */
                "compaction": { "reserve_tokens": 8000, },
            }"#,
        )
        .unwrap();
        assert_eq!(s.default_model.as_deref(), Some("openai/gpt-4o"));
        assert_eq!(s.permission_mode, Some(PermissionMode::AutoEdit));
        assert_eq!(s.compaction.reserve_tokens, Some(8000));
    }

    #[test]
    fn empty_and_comment_only_files_are_ok() {
        assert_eq!(Settings::parse("").unwrap(), Settings::default());
        assert_eq!(Settings::parse("  \n").unwrap(), Settings::default());
        assert_eq!(Settings::parse("{}").unwrap(), Settings::default());
    }

    #[test]
    fn unknown_field_is_rejected() {
        // 拼错的字段名 (驼峰) 必须报错而不是静默忽略。
        assert!(Settings::parse(r#"{"defaultModel": "a/b"}"#).is_err());
        assert!(Settings::parse(r#"{"tools": {"max_output": 1}}"#).is_err());
        assert!(Settings::parse(r#"{"providers": {"x": {"baseUrl": "u"}}}"#).is_err());
    }

    #[test]
    fn bad_permission_mode_is_rejected() {
        // spec: 没有 yolo。
        assert!(Settings::parse(r#"{"permission_mode": "yolo"}"#).is_err());
    }

    #[test]
    fn merge_prefers_higher_layer_field_by_field() {
        let global = Settings::parse(
            r#"{"default_model":"a/x","context_window":1000,
                "tools":{"max_output_bytes":10,"bash_timeout_secs":5},
                "models":{"a/x":{"context_window":111}}}"#,
        )
        .unwrap();
        let project = Settings::parse(
            r#"{"default_model":"b/y","tools":{"max_output_bytes":20},
                "models":{"a/x":{"context_window":222},"b/y":{"context_window":333}}}"#,
        )
        .unwrap();
        let m = global.merge(project);
        assert_eq!(m.default_model.as_deref(), Some("b/y")); // 项目覆盖
        assert_eq!(m.context_window, Some(1000)); // 项目没写, 沿用全局
        assert_eq!(m.tools.max_output_bytes, Some(20)); // 逐字段合并
        assert_eq!(m.tools.bash_timeout_secs, Some(5));
        assert_eq!(m.models["a/x"].context_window, Some(222)); // 同 key 高层覆盖
        assert_eq!(m.models["b/y"].context_window, Some(333)); // 新 key 追加
    }

    /// 项目层改用 api_key_env 时, 不能被全局层的字面量 api_key 压过。
    #[test]
    fn provider_key_setting_is_replaced_as_a_unit() {
        let global = Settings::parse(r#"{"providers":{"p":{"base_url":"u","api_key":"lit"}}}"#).unwrap();
        let project = Settings::parse(r#"{"providers":{"p":{"api_key_env":"MY_ENV"}}}"#).unwrap();
        let p = &global.merge(project).providers["p"];
        assert_eq!(p.base_url.as_deref(), Some("u")); // 未写的字段沿用
        assert_eq!(p.api_key, None);
        assert_eq!(p.api_key_env.as_deref(), Some("MY_ENV"));
    }
}
