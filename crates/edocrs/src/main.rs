//! edoCRS 入口.
//! 负责: 解析 CLI -> 加载 config -> 注册工具 -> 装配 Agent + REPL -> 运行.

mod agent;
mod api;
mod cli;
mod config;
mod errors;
mod event;
mod permission;
mod repl;
mod sampler;
mod session;
mod session_actor;
mod system_prompt;
mod tools;
mod ui;

use crate::agent::Agent;
use crate::cli::Cli;
use crate::config::Config;
use crate::errors::AppError;
use crate::permission::PermissionGate;
use crate::repl::Repl;
use crate::session::Session;
use crate::tools::Registry;
use clap::Parser;
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

    // 多源 config 加载 (.env + config.toml + 默认), CLI flag 后续覆盖.
    let mut cfg = Config::load()?;

    if let Some(k) = args.api_key {
        cfg.api_key = k;
    }
    if let Some(u) = args.base_url {
        cfg.base_url = u;
    }
    if let Some(m) = args.model {
        // 复用 Model::parse 的合法性校验 (支持内置两档 + provider/id 语法).
        cfg.model = config::Model::parse(&m)?;
    }

    // 工具注册表
    let mut registry = Registry::new();
    registry.register(Box::new(tools::read_file::ReadFile {
        max_bytes: cfg.max_tool_output_bytes,
    }));
    registry.register(Box::new(tools::write_file::WriteFile));
    registry.register(Box::new(tools::bash::Bash {
        max_bytes: cfg.max_tool_output_bytes,
    }));

    // Agent
    let agent = Agent {
        sampler: sampler::build(
            cfg.api_key.clone(),
            cfg.base_url.clone(),
            &cfg.model,
            sampler::SamplerConfig::default(),
        ),
        registry,
        // 学习点: gate 现在包在 RefCell 里 —— actor 化后 Agent 只被 `&self` 借用,
        //         权限缓存的可变性下沉到字段级。
        gate: RefCell::new(PermissionGate::new(args.yolo)),
        // 学习点: 在 main 里一次拿到 cwd, 之后整个会话期 working_dir 不变 — 即便用户
        //         在 REPL 中 `cd` 也不影响 (REPL 没暴露 cd, 这里更多是为了行为可预测).
        working_dir: std::env::current_dir()
            .map_err(|e| AppError::Config(format!("无法获取 cwd: {e}")))?,
        hunk_tracker: RefCell::new(session::checkpoint::HunkTracker::new()),
    };

    // /tools 展示用的 schema 快照 (工具集会话期不变)。
    let tool_schemas = agent.registry.schemas();

    // 加载 / 新建 Session
    let session = match args.resume.as_deref() {
        Some("last") => match Session::most_recent_id(&cfg.session_dir).await? {
            Some(id) => Session::load(&cfg.session_dir, &id)?,
            None => Session::new(cfg.model.clone(), &cfg.session_dir)?,
        },
        Some(id) => Session::load(&cfg.session_dir, id)?,
        None => Session::new(cfg.model.clone(), &cfg.session_dir)?,
    };

    ui::banner(env!("CARGO_PKG_VERSION"), cfg.model.api_id(), &session.id.to_string());

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
