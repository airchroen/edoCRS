//! UI 辅助层 — crossterm 着色 / 流式打印 / 横幅.
//!
//! 设计目标: 让 agent.rs / repl.rs 不直接接触 crossterm,
//!           集中所有终端样式决策, 后续要改主题或换库都只改这一处.
//!
//! 学习点: crossterm::style::Stylize 是个 extension trait, 给 &str 加了
//!         .with(Color)/.bold() 等方法. 实际产生的 String 已包含 ANSI 转义.

use crossterm::style::{Color, Stylize};
use std::io::{self, Write};

pub fn banner(version: &str, model: &str, session_id: &str) {
    let short_id: String = session_id.chars().take(8).collect();
    println!(
        "{} v{}  ·  model: {}  ·  session: {}",
        "edoCRS".with(Color::Cyan).bold(),
        version,
        model,
        short_id,
    );
    println!("输入 /help 看命令, Ctrl+D 退出.");
}

/// 用户输入提示符 (青色). repl.rs 喂给 rustyline.
pub fn prompt() -> String {
    "▶ ".with(Color::Cyan).to_string()
}

/// 流式打印模型文本片段 (不换行, 即时 flush).
pub fn print_stream(text: &str) {
    print!("{text}");
    let _ = io::stdout().flush();
}

/// 流式结束后补一个换行, 让下一段不挤在一起.
pub fn end_stream() {
    println!();
}

pub fn show_tool_call(name: &str, args_preview: &str) {
    let preview: String = args_preview.chars().take(160).collect();
    println!(
        "{} [{}]  args: {}",
        "⚙".with(Color::Yellow),
        name.with(Color::Yellow),
        preview
    );
}

pub fn show_tool_result(result: &str) {
    let lines: Vec<&str> = result.lines().collect();
    if lines.len() > 10 {
        for l in &lines[..10] {
            println!("  {}", l.with(Color::DarkGrey));
        }
        println!(
            "  {}",
            format!("... ({} more lines)", lines.len() - 10).with(Color::DarkGrey)
        );
    } else {
        for l in &lines {
            println!("  {}", l.with(Color::DarkGrey));
        }
    }
}

pub fn show_error(msg: &str) {
    eprintln!("{} {}", "✗".with(Color::Red).bold(), msg.with(Color::Red));
}

pub fn show_permission_question(tool_name: &str, args_preview: &str) {
    let preview: String = args_preview.chars().take(200).collect();
    println!("{} [tool_call] {}", "?".with(Color::Magenta), tool_name);
    println!("  args: {preview}");
    print!(
        "{} 允许执行? [y]es / [n]o / [a]llow-all-{tool_name}: ",
        "?".with(Color::Magenta)
    );
    let _ = io::stdout().flush();
}

/// 同步读一行 stdin, 解析成权限 Decision (供 REPL 消费端应答 PermissionRequest 用)。
///
/// 学习点: 事件化后, 「读 stdin」这个终端专属动作从 permission.rs 挪到了这里 ——
///         gate 只管决策逻辑, 具体怎么问用户是消费端 (REPL) 的事。未来 TUI 客户端
///         会用自己的输入组件替换这一函数。
pub fn read_permission_answer() -> crate::permission::Decision {
    use crate::permission::Decision;
    let mut buf = String::new();
    if io::stdin().read_line(&mut buf).is_err() {
        return Decision::Deny;
    }
    match buf.trim() {
        "y" | "Y" | "yes" => Decision::Allow,
        "a" | "A" | "all" => Decision::AllowAll,
        _ => Decision::Deny,
    }
}

#[cfg(test)]
mod tests {
    /// 仅 smoke-test: 确保函数不 panic. 颜色码我们不验证.
    #[test]
    fn banner_does_not_panic() {
        super::banner(
            "0.1.0",
            "deepseek-v4-flash",
            "abc12345-aaaa-bbbb-cccc-dddddddddddd",
        );
    }
}
