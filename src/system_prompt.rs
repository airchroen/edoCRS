//! System prompt 构建.
//!
//! 设计参考 Claude Code: system prompt **不写进 conversation history**,
//! 而是每次发请求时即时拼装. 这样的好处:
//!
//!   1. session.json 里只有 user / assistant / tool, 不会被大段 prompt 污染;
//!   2. resume 老会话自动用最新 BASE_PROMPT / AGENTS.md, 改了立即生效, 不需要存档迁移;
//!   3. 多个 session 共享同一份 system, 没有"哪份是真"的问题.
//!
//! 拼接顺序 (复刻 Claude Code 的 layering):
//!
//!   ┌── 1. BASE_PROMPT      内置基础提示 (agent 身份, 行为约束)
//!   ├── 2. Environment 块   运行时上下文 (cwd / date / model)
//!   └── 3. AGENTS.md (可选) cwd 下的项目说明; 不存在则省略
//!
//! 学习点: build() 是纯函数 — 输入相同, 输出确定 (除了 date 字段, 但它本就该跟随时钟).
//!         这种纯度让单元测试很好写: 给个 tempdir 当 cwd, 直接断言输出包含什么.

use crate::config::Model;
use std::path::Path;

/// 内置基础 prompt — 定义 agent 的身份 / 可用工具 / 行为规范.
///
/// 学习点: 这种"硬编码字符串常量"在 Rust 里用 `const &str` 最自然.
///         注意末尾用 \ 续行 + "\" 让多行字符串保持紧凑而无前导空白.
const BASE_PROMPT: &str = "\
你是 edoCRS, 一个用 Rust 编写的简易 agent CLI 助手 (后端 DeepSeek V4).

# 角色
- 协助用户阅读、修改和运行项目代码.
- 可调用以下工具来操作文件系统:
  - read_file: 读取文件
  - write_file: 写入或覆盖文件
  - bash: 执行 shell 命令 (可能需要用户授权)

# 行为准则
- 默认用中文回复.
- 修改文件前先 read_file 确认上下文, 不凭空改.
- 执行有副作用的命令 (rm / 覆盖 / 推送) 前简短说明意图, 然后由权限层向用户确认.
- 回答精简但不省略关键信息.";

/// 构造一份完整的 system prompt 字符串.
///
/// `cwd` 通常是 `std::env::current_dir()` 的结果, 测试里换成 tempdir.
///
/// 学习点: 纯函数 + 显式参数 = 可测试. 我们没有从 `std::env` 直接读 cwd,
///         而是要求调用方传进来, 这样测试可以注入 tempdir, 不污染真实 fs.
pub fn build(model: &Model, cwd: &Path) -> String {
    let mut s = String::with_capacity(1024);
    s.push_str(BASE_PROMPT);

    // ── Environment 块: 把运行时上下文喂给模型 ──
    // 学习点: chrono::Utc::now() 是这里唯一的"非纯"来源, 但只取日期部分,
    //         同一天内多次调用结果一致, 测试不会因为时间漂移而 flaky.
    s.push_str("\n\n# Environment\n");
    s.push_str(&format!("- working directory: {}\n", cwd.display()));
    s.push_str(&format!("- date: {}\n", chrono::Utc::now().format("%Y-%m-%d")));
    s.push_str(&format!("- model: {}\n", model.api_id()));

    // ── 项目级指令 (可选): cwd 下的 AGENTS.md ──
    // 学习点: std::fs::read_to_string 返回 Result, 用 if let Ok 优雅地处理"文件不存在"
    //         这个最常见的失败分支, 而不是 unwrap / panic.
    if let Ok(content) = std::fs::read_to_string(cwd.join("AGENTS.md")) {
        s.push_str("\n\n# Project Instructions (AGENTS.md)\n");
        s.push_str(content.trim());
        s.push('\n');
    }

    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::tempdir;

    /// build 应包含 BASE_PROMPT 的特征字符串 — 用 "edoCRS" 验证 base 段确实被拼进去了.
    #[test]
    fn includes_base_prompt() {
        let dir = tempdir().unwrap();
        let s = build(&Model::deepseek_v4_flash(), dir.path());
        assert!(s.contains("edoCRS"), "缺少 BASE_PROMPT 特征 (期望出现 'edoCRS'): {s}");
    }

    /// Environment 块应包含 cwd 的字面路径 — 这样模型就知道当前在哪个项目目录.
    #[test]
    fn includes_cwd_in_env_block() {
        let dir = tempdir().unwrap();
        let s = build(&Model::deepseek_v4_flash(), dir.path());
        let cwd_str = dir.path().display().to_string();
        assert!(s.contains(&cwd_str), "缺少 cwd ({cwd_str}): {s}");
    }

    /// Environment 块应包含 model.api_id() — 让模型知道自己是哪个档位.
    #[test]
    fn includes_model_id() {
        let dir = tempdir().unwrap();
        let s = build(&Model::deepseek_v4_pro(), dir.path());
        assert!(s.contains("deepseek-v4-pro"), "缺少 model id: {s}");
    }

    /// 当 cwd 下有 AGENTS.md 时, 其内容应被追加进 system prompt.
    /// 用一个独特字符串 "PROJECT-MARKER-XYZ" 当探针, 在输出里找它.
    #[test]
    fn appends_agents_md_when_present() {
        let dir = tempdir().unwrap();
        fs::write(dir.path().join("AGENTS.md"), "PROJECT-MARKER-XYZ").unwrap();
        let s = build(&Model::deepseek_v4_flash(), dir.path());
        assert!(s.contains("PROJECT-MARKER-XYZ"), "AGENTS.md 内容缺失: {s}");
    }

    /// 当 cwd 下没有 AGENTS.md, build 不应 panic, 也不应错误地输出 header 标题
    /// (header 出现意味着模型会把后续空内容误读成"项目无指令", 不如不出现).
    #[test]
    fn skips_agents_md_when_absent() {
        let dir = tempdir().unwrap();
        let s = build(&Model::deepseek_v4_flash(), dir.path());
        assert!(
            !s.contains("Project Instructions"),
            "AGENTS.md 不存在但 header 出现了: {s}"
        );
    }
}
