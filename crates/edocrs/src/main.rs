//! edoCRS 入口.
//! 负责: 解析 CLI -> 加载 config -> 注册工具 -> 装配 Agent + REPL -> 运行.

mod agent;
mod cli;
mod config;
mod errors;
mod event;
mod permission;
mod repl;
mod session;
mod session_actor;
mod system_prompt;
#[cfg(test)]
mod test_support;
mod tools;
mod ui;

use crate::agent::Agent;
use crate::cli::Cli;
use crate::config::{CliOverrides, Config};
use crate::errors::AppError;
use crate::permission::PermissionGate;
use crate::repl::Repl;
use crate::session::Session;
use crate::tools::Registry;
use clap::Parser;
use edocrs_ai::{Sampler, SamplerConfig};
use std::cell::RefCell;

#[tokio::main]
async fn main() {
    if let Err(e) = run().await {
        ui::show_error(&format!("{e}"));
        std::process::exit(1);
    }
}

async fn run() -> Result<(), AppError> {
    let args = Cli::parse();

    // cwd: --cwd 优先。之后整个会话期 working_dir 不变 (即便用户在 REPL 中 `cd` 也不影响)。
    let cwd = match args.cwd {
        Some(p) => p.canonicalize().map_err(|e| AppError::Config(format!("--cwd 无效: {e}")))?,
        None => std::env::current_dir()
            .map_err(|e| AppError::Config(format!("无法获取 cwd: {e}")))?,
    };

    // 分层加载: .env + 全局/项目 settings.json + 环境变量 + CLI。
    // 项目信任 (spec §8) 要到阶段 7 才有 UI, 此前项目层 settings 一律不读 (安全默认)。
    let cfg = Config::load(
        &cwd,
        false,
        CliOverrides { model: args.model, mode: args.mode.map(Into::into) },
    )?;

    // 工具注册表
    let mut registry = Registry::new();
    registry.register(Box::new(tools::read_file::ReadFile {
        max_bytes: cfg.max_output_bytes,
    }));
    registry.register(Box::new(tools::write_file::WriteFile));
    registry.register(Box::new(tools::bash::Bash {
        max_bytes: cfg.max_output_bytes,
        default_timeout_secs: cfg.bash_timeout_secs,
    }));

    // Agent
    let agent = Agent {
        sampler: Sampler::new(cfg.registry.clone(), SamplerConfig::default()),
        registry,
        // 学习点: gate 包在 RefCell 里 —— actor 化后 Agent 只被 `&self` 借用,
        //         权限缓存的可变性下沉到字段级。
        gate: RefCell::new(PermissionGate::new(cfg.permission_mode)),
        working_dir: cwd,
        hunk_tracker: RefCell::new(session::checkpoint::HunkTracker::new()),
    };

    // /tools 展示用的 schema 快照 (工具集会话期不变)。
    let tool_schemas = agent.registry.schemas();

    // 加载 / 新建 Session
    // -c 与 `--resume` (不带 id) 都是「最近一次」; `--resume <id>` 精确恢复。
    let resume_target: Option<String> = match args.resume {
        Some(Some(id)) => Some(id),
        Some(None) => Some("last".into()),
        None if args.continue_last => Some("last".into()),
        None => None,
    };
    let session = match resume_target.as_deref() {
        Some("last") => match Session::most_recent_id(&cfg.session_dir).await? {
            Some(id) => Session::load(&cfg.session_dir, &id)?,
            None => Session::new(cfg.model.clone(), &cfg.session_dir)?,
        },
        Some(id) => Session::load(&cfg.session_dir, id)?,
        None => Session::new(cfg.model.clone(), &cfg.session_dir)?,
    };

    ui::banner(env!("CARGO_PKG_VERSION"), &cfg.model.to_string(), &session.id.to_string());

    // 装配会话 actor + REPL 客户端。
    // 学习点: SessionActor 用 Rc/RefCell (非 Send), 只能跑在单线程 LocalSet 上 ——
    //         `tokio::task::LocalSet::run_until` 在当前线程建一个本地任务作用域,
    //         `spawn_local` 出来的 turn 任务都落在这里面。actor 与 REPL 并发跑,
    //         REPL 结束 (/exit 或 Ctrl+D) 后 run_until 返回。
    let (handle, actor) = session_actor::spawn(agent, session, cfg.session_dir.clone());
    let repl = Repl {
        handle,
        session_dir: cfg.session_dir.clone(),
        tool_schemas,
    };

    let local = tokio::task::LocalSet::new();
    local
        .run_until(async move {
            // actor 主循环作为一个本地任务常驻; REPL 在前台驱动。
            let actor_task = tokio::task::spawn_local(actor.run());
            let result = repl.run().await;
            // REPL 退出后 actor 也应已收到 Shutdown; 等它收尾。
            let _ = actor_task.await;
            result
        })
        .await
}
