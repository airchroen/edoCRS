//! 模型标识: `provider/model-id`。
//!
//! 按**第一个** `/` 切分 —— 因为 OpenRouter 的模型 id 自身就带斜杠:
//! `openrouter/anthropic/claude-sonnet-4` → provider = `openrouter`,
//! id = `anthropic/claude-sonnet-4`。
//!
//! Model 只是一对字符串, 不校验 provider 是否存在 (那要查注册表), 也不校验模型 id
//! 是否合法 (那由 provider 判定)。

use crate::error::AiError;
use serde::{Deserialize, Serialize};
use std::fmt;
use std::str::FromStr;

/// `provider/model-id` 标识。
///
/// 学习点: 两个字段都是 `String`, 所以只能 `Clone` 不能 `Copy` —— 需要复制时显式 `.clone()`,
///         能借用时传 `&Model`。
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct Model {
    /// 注册表里的 provider id, 如 `deepseek`。
    pub provider: String,
    /// 原样塞进请求体 `model` 字段的字符串, 如 `deepseek-v4-flash`。
    pub id: String,
}

impl Model {
    pub fn new(provider: impl Into<String>, id: impl Into<String>) -> Self {
        // 学习点: `impl Into<String>` 让调用方既能传 `&str` 也能传 `String`,
        //         传 String 时不会多一次拷贝。
        Self { provider: provider.into(), id: id.into() }
    }

    /// 解析 `provider/model-id`; 两段都不能为空。
    pub fn parse(s: &str) -> Result<Self, AiError> {
        match s.trim().split_once('/') {
            Some((p, id)) if !p.is_empty() && !id.is_empty() => Ok(Self::new(p, id)),
            _ => Err(AiError::BadModel(s.to_string())),
        }
    }
}

/// `Display` 渲染回 `provider/model-id`, 与 `parse` 互逆。
///
/// 学习点: 实现了 `Display` 就自动获得 `.to_string()` (标准库对所有 `T: Display`
///         有 blanket impl `ToString`)。
impl fmt::Display for Model {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}/{}", self.provider, self.id)
    }
}

/// 实现 `FromStr` 后可以写 `"a/b".parse::<Model>()`, clap 也能直接把参数解析成 Model。
impl FromStr for Model {
    type Err = AiError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::parse(s)
    }
}

// 学习点: 手写 Serialize/Deserialize, 让 Model 在 JSON 里是**一个字符串**
//         (`"deepseek/deepseek-v4-flash"`) 而不是 `{provider, id}` 对象 ——
//         会话文件与 settings 都更可读, 反序列化也复用同一份 `parse` 校验。
impl Serialize for Model {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.collect_str(self) // collect_str 直接用 Display 写出, 省一次中间 String
    }
}

impl<'de> Deserialize<'de> for Model {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let s = String::deserialize(d)?;
        Model::parse(&s).map_err(serde::de::Error::custom)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_splits_on_first_slash() {
        let m = Model::parse("openrouter/anthropic/claude-sonnet-4").unwrap();
        assert_eq!(m.provider, "openrouter");
        assert_eq!(m.id, "anthropic/claude-sonnet-4");
        assert_eq!(m.to_string(), "openrouter/anthropic/claude-sonnet-4");
    }

    #[test]
    fn parse_rejects_malformed() {
        for bad in ["gpt-4o", "/x", "openai/", ""] {
            assert!(Model::parse(bad).is_err(), "{bad:?} 应被拒绝");
        }
    }

    #[test]
    fn serde_as_string_roundtrip() {
        let m = Model::new("deepseek", "deepseek-v4-flash");
        let json = serde_json::to_string(&m).unwrap();
        assert_eq!(json, "\"deepseek/deepseek-v4-flash\"");
        assert_eq!(serde_json::from_str::<Model>(&json).unwrap(), m);
        assert!(serde_json::from_str::<Model>("\"nope\"").is_err());
    }
}
