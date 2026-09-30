//! CLI 参数解析 (clap derive).
//!
//! 学习点: clap 的 derive 模式让我们用普通 struct + 属性宏定义参数,
//!         远比手写 match 更易读. `#[arg(long)]` 自动生成 --xxx 形式.
//!
//! API key / base_url 不再有 CLI 参数: 前者走各 provider 的环境变量 (避免 key 出现在
//! 进程列表 / shell 历史里), 后者在 settings.json 的 `providers` 里配。

use crate::config::PermissionMode;
use clap::{Parser, ValueEnum};
use std::path::PathBuf;

/// `--mode` 的取值。
///
/// 学习点: 单独定义一个 `ValueEnum` 而不是让 `PermissionMode` 直接派生 clap 特征 ——
///         配置层类型不必依赖 clap, CLI 层的取值集合也可以独立演进。
#[derive(Copy, Clone, Debug, ValueEnum)]
pub enum ModeArg {
    Ask,
    AutoEdit,
}

impl From<ModeArg> for PermissionMode {
    fn from(m: ModeArg) -> Self {
        match m {
            ModeArg::Ask => PermissionMode::Ask,
            ModeArg::AutoEdit => PermissionMode::AutoEdit,
        }
    }
}

#[derive(Parser, Debug)]
#[command(name = "edocrs", version, about = "简易 Claude Code 风格 CLI")]
pub struct Cli {
    /// 模型, 形如 `provider/model-id` (如 deepseek/deepseek-v4-flash).
    #[arg(long)]
    pub model: Option<String>,

    /// 权限模式: ask (write/edit/bash 都询问) 或 auto-edit (write/edit 自动放行, bash 仍询问).
    #[arg(long, value_enum)]
    pub mode: Option<ModeArg>,

    /// 继续最近一次会话.
    #[arg(short = 'c', long = "continue")]
    pub continue_last: bool,

    /// 恢复会话: 指定 session id; 不带值时暂同 --continue (会话选择器随 TUI 提供).
    ///
    /// 学习点: `num_args = 0..=1` 让 `--resume` 既可当开关用也可带一个值;
    ///         `Option<Option<String>>`: 外层 = 是否出现该 flag, 内层 = 是否带值。
    #[arg(long, value_name = "ID", num_args = 0..=1)]
    pub resume: Option<Option<String>>,

    /// 工作目录 (默认当前目录). 决定项目 settings / 上下文文件 / 工具的 cwd.
    #[arg(long)]
    pub cwd: Option<PathBuf>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(args: &[&str]) -> Cli {
        Cli::try_parse_from(std::iter::once("edocrs").chain(args.iter().copied())).unwrap()
    }

    #[test]
    fn resume_flag_forms() {
        assert!(parse(&[]).resume.is_none());
        assert_eq!(parse(&["--resume"]).resume, Some(None));
        assert_eq!(parse(&["--resume", "abc"]).resume, Some(Some("abc".into())));
        assert!(parse(&["-c"]).continue_last);
    }

    #[test]
    fn mode_parses_kebab_case() {
        assert!(matches!(parse(&["--mode", "auto-edit"]).mode, Some(ModeArg::AutoEdit)));
        assert!(Cli::try_parse_from(["edocrs", "--mode", "yolo"]).is_err());
    }

    #[test]
    fn removed_flags_are_rejected() {
        assert!(Cli::try_parse_from(["edocrs", "--api-key", "x"]).is_err());
        assert!(Cli::try_parse_from(["edocrs", "--base-url", "x"]).is_err());
        assert!(Cli::try_parse_from(["edocrs", "--yolo"]).is_err());
    }
}
