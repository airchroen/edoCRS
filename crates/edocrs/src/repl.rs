//! REPL 主循环 (客户端形态).
//!
//! Actor 化后, REPL 不再持有 `Agent`/`Session`, 而是持有一个 `SessionHandle` ——
//! 它只是会话 actor 的一个客户端。用户输入分流:
//!   - 以 '/' 开头: slash 命令。
//!   - 否则: `handle.submit(line).await` 交给 actor 跑一个 turn。
//!
//! 学习点: 这是「把逻辑收进 actor, 让 UI 只做视图」的第一步。未来加 TUI / headless
//!         客户端时, 它们与本 REPL 平级, 共享同一个 SessionHandle 语义。
//!
//! 多行输入: 简化方案 —— 行尾 '\\' 续行, 空行结束。

use crate::errors::AppError;
use crate::event::SessionEvent;
use crate::session_actor::{SessionHandle, TurnOutcome};
use crate::ui;
use rustyline::error::ReadlineError;
use rustyline::{DefaultEditor, Result as RlResult};
use std::path::PathBuf;

pub struct Repl {
    pub handle: SessionHandle,
    pub session_dir: PathBuf,
    /// 启动时从工具注册表快照下来的 schema (给 /tools 展示)。
    /// 学习点: 注册表现在被 actor 内的 Agent 独占持有, REPL 拿不到 &Registry;
    ///         工具集在会话期不变, 所以启动时 clone 一份 schema 即可, 无需回问 actor。
    pub tool_schemas: Vec<serde_json::Value>,
}

impl Repl {
    pub async fn run(mut self) -> Result<(), AppError> {
        let mut rl = DefaultEditor::new().map_err(io_err)?;
        loop {
            match self.read_input(&mut rl) {
                Ok(line) if line.trim().is_empty() => continue,
                Ok(line) if line.starts_with('/') => {
                    if self.dispatch_slash(&line).await? {
                        return Ok(()); // /exit / /quit
                    }
                }
                Ok(line) => {
                    // 交给 actor 跑一个 turn, 并在事件循环里渲染进展。
                    self.run_one_turn(line).await;
                }
                Err(ReadlineError::Interrupted) => {
                    // Ctrl+C: 清行回到 prompt。
                    // 说明: rustyline 的同步 readline 阻塞时收不到 Ctrl+C 去中断在跑的 turn;
                    //       真正的「turn 进行中取消」需要异步输入 (留待事件流子系统)。此处
                    //       语义与旧行为一致: 空 prompt 上 Ctrl+C 清行。
                    println!();
                    continue;
                }
                Err(ReadlineError::Eof) => {
                    // Ctrl+D: 退出。actor shutdown 时会存盘。
                    self.handle.shutdown().await;
                    return Ok(());
                }
                Err(e) => {
                    ui::show_error(&format!("readline: {e}"));
                    return Ok(());
                }
            }
        }
    }

    /// 提交一句用户输入, 边收事件边渲染, 直到 turn 终局。
    ///
    /// 学习点: 事件接收 (`events.recv()`) 与终局等待 (`outcome`) 是两条并发的流,
    ///         用 `tokio::select!` 同时驱动。`biased` 让事件优先被处理, 保证渲染顺序
    ///         (先把已到的进展打完, 再看 turn 是否结束)。
    async fn run_one_turn(&mut self, line: String) {
        let Some((mut events, mut outcome)) = self.handle.submit(line).await else {
            ui::show_error("提交失败: 会话 actor 无响应");
            return;
        };

        loop {
            tokio::select! {
                biased;
                maybe_ev = events.recv() => {
                    match maybe_ev {
                        Some(ev) => self.render_event(ev),
                        None => {
                            // 事件通道关闭 (turn 任务结束并 drop 了 sink)。等终局收尾。
                            match (&mut outcome).await {
                                Ok(TurnOutcome::Failed(msg)) => ui::show_error(&msg),
                                _ => {}
                            }
                            break;
                        }
                    }
                }
                res = &mut outcome => {
                    // 终局先到 (少见: 通常事件通道先关)。排空剩余事件再退出。
                    while let Ok(ev) = events.try_recv() {
                        self.render_event(ev);
                    }
                    if let Ok(TurnOutcome::Failed(msg)) = res {
                        ui::show_error(&msg);
                    }
                    break;
                }
            }
        }
    }

    /// 把一个 SessionEvent 渲染到终端。这是「视图」的全部 —— agent 侧只发事件, 不碰 IO。
    fn render_event(&self, ev: SessionEvent) {
        match ev {
            SessionEvent::StreamText(t) => ui::print_stream(&t),
            // 中等档: 思维链暂不显示 (可后续灰显)。
            SessionEvent::ReasoningDelta(_) => {}
            SessionEvent::StreamEnd => ui::end_stream(),
            SessionEvent::ToolStarted { name, args_preview } => {
                ui::show_tool_call(&name, &args_preview);
            }
            SessionEvent::ToolFinished { result, .. } => ui::show_tool_result(&result),
            SessionEvent::PermissionRequest { name, args_preview, reply } => {
                // 同步阻塞读 stdin 应答 —— 这正是我们要的: turn 侧也在 await reply,
                // 单终端客户端阻塞在这里可接受。未来 TUI 才需异步输入组件。
                ui::show_permission_question(&name, &args_preview);
                let decision = ui::read_permission_answer();
                let _ = reply.send(decision);
            }
        }
    }

    /// 读一行 (或多行 \\ 续行) 用户输入。
    /// 学习点: rustyline 给我们历史记录 / 行编辑 / Ctrl+R 反向搜索都自动具备。
    fn read_input(&self, rl: &mut DefaultEditor) -> RlResult<String> {
        let mut acc = String::new();
        loop {
            let prompt = if acc.is_empty() {
                ui::prompt()
            } else {
                "  ".to_string()
            };
            let line = rl.readline(&prompt)?;
            if let Some(stripped) = line.strip_suffix('\\') {
                acc.push_str(stripped);
                acc.push('\n');
                continue;
            }
            if !acc.is_empty() && line.is_empty() {
                break; // 多行模式空行结束
            }
            acc.push_str(&line);
            break;
        }
        let _ = rl.add_history_entry(&acc);
        Ok(acc)
    }

    /// 处理 /xxx, 返回 true 表示请求退出。
    async fn dispatch_slash(&mut self, line: &str) -> Result<bool, AppError> {
        let mut parts = line.trim().splitn(2, ' ');
        let cmd = parts.next().unwrap_or("");
        let arg = parts.next().unwrap_or("").trim();
        match cmd {
            "/help" => {
                println!("可用命令:");
                println!("  /help              本帮助");
                println!("  /exit, /quit       退出 (自动保存)");
                println!("  /clear             清空当前会话, 开新会话");
                println!("  /rewind <n>        回滚到第 n 个回合边界 (文件一起回滚)");
                println!("  /fork              分叉当前会话为一个新会话");
                println!("  /save <name>       给当前会话起别名 (软链接)");
                println!("  /sessions          列出最近 10 个会话");
                println!("  /tools             列出已注册工具的 schema");
                Ok(false)
            }
            "/exit" | "/quit" => {
                self.handle.shutdown().await;
                Ok(true)
            }
            "/clear" => {
                match self.handle.clear().await {
                    Some(id) => println!("已开新会话: {id}"),
                    None => ui::show_error("清空失败: 会话 actor 无响应"),
                }
                Ok(false)
            }
            "/rewind" => {
                let Ok(n) = arg.parse::<u32>() else {
                    ui::show_error("用法: /rewind <回合号, 非负整数>");
                    return Ok(false);
                };
                match self.handle.rewind(n).await {
                    Ok(()) => println!("已回滚到回合 {n} (文件与历史一起回退)"),
                    Err(e) => ui::show_error(&format!("回滚失败: {e}")),
                }
                Ok(false)
            }
            "/fork" => {
                match self.handle.fork().await {
                    Some(id) => println!("已分叉出新会话: {id}"),
                    None => ui::show_error("分叉失败: 会话 actor 无响应"),
                }
                Ok(false)
            }
            "/save" => {
                if arg.is_empty() {
                    ui::show_error("用法: /save <name>");
                    return Ok(false);
                }
                // 取当前会话 id (经 actor 快照)。
                let Some(session) = self.handle.snapshot().await else {
                    ui::show_error("保存失败: 会话 actor 无响应");
                    return Ok(false);
                };
                // 会话现在是目录 `<id>/`; 别名做一个指向该目录的软链接 `<name>`。
                let target = self.session_dir.join(session.id.to_string());
                let link = self.session_dir.join(arg);
                if link.exists() {
                    ui::show_error(&format!("已存在 {arg}, 拒绝覆盖"));
                    return Ok(false);
                }
                #[cfg(unix)]
                std::os::unix::fs::symlink(&target, &link)
                    .map_err(|e| AppError::Config(format!("symlink 失败: {e}")))?;
                #[cfg(not(unix))]
                {
                    // 非 unix: 复制整个会话目录。
                    ui::show_error("非 unix 平台暂不支持 /save 别名 (需软链接)");
                    let _ = &target;
                }
                #[cfg(unix)]
                println!("已保存别名: {arg}");
                Ok(false)
            }
            "/sessions" => {
                let mut entries = vec![];
                if let Ok(mut rd) = tokio::fs::read_dir(&self.session_dir).await {
                    while let Ok(Some(e)) = rd.next_entry().await {
                        let log = e.path().join("updates.jsonl");
                        if log.exists() {
                            let meta = match tokio::fs::metadata(&log).await {
                                Ok(m) => m,
                                Err(_) => continue,
                            };
                            entries.push((meta.modified().ok(), e.path(), log));
                        }
                    }
                }
                entries.sort_by(|a, b| b.0.cmp(&a.0));
                for (_, dir, log) in entries.into_iter().take(10) {
                    let stem = dir.file_name().unwrap().to_string_lossy();
                    let summary = first_user_message_summary(&log).await.unwrap_or_default();
                    println!("  {stem}  {summary}");
                }
                Ok(false)
            }
            "/tools" => {
                for s in &self.tool_schemas {
                    let pretty = serde_json::to_string_pretty(s).unwrap_or_default();
                    println!("{pretty}");
                }
                Ok(false)
            }
            other => {
                ui::show_error(&format!("未知命令 {other}, 试 /help"));
                Ok(false)
            }
        }
    }
}

/// 读 updates.jsonl, 找第一条 user_message 记录的 content 摘要 (≤30 字)。
async fn first_user_message_summary(log_path: &std::path::Path) -> Option<String> {
    let text = tokio::fs::read_to_string(log_path).await.ok()?;
    for line in text.lines() {
        let v: serde_json::Value = match serde_json::from_str(line) {
            Ok(v) => v,
            Err(_) => continue,
        };
        if v.get("kind").and_then(|k| k.as_str()) == Some("user_message") {
            let content = v.get("content")?.as_str()?;
            return Some(content.chars().take(30).collect());
        }
    }
    None
}

fn io_err<E: std::error::Error>(e: E) -> AppError {
    AppError::Config(format!("readline init failed: {e}"))
}
