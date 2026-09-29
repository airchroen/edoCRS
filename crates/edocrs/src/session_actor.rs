//! SessionActor —— 单线程会话 actor (跑在 tokio LocalSet 上).
//!
//! 设计动机 (参考 grok-build 的 SessionActor): 旧的 `Repl` 把 `Agent` + `Session`
//! 用 `&mut self` 独占串行持有, 一旦 `run_turn().await` 就把整个线程占住 —— 无法在
//! turn 进行中接收 Ctrl+C 取消、无法被非终端客户端复用。
//!
//! 解决办法是把「会话状态 + agent loop」搬进一个 actor:
//!   - 状态字段用 `Rc<RefCell<...>>` (单线程内部可变), 不用 `Arc<Mutex>` —— 因为
//!     整个 actor 跑在单线程 `LocalSet` 上, 无跨线程需求, 也就无需 `Send`/锁。
//!   - actor 主循环从 `mpsc` channel 逐条收 `SessionCommand`。
//!   - 每个 `SubmitUserInput` 命令 `spawn_local` 一个独立任务跑 turn, 主循环立刻回到
//!     `recv()` —— 于是 turn 运行期间仍能收到 `Cancel` 命令。
//!
//! ⚠️ 学习点: `Rc`/`RefCell` 不是 `Send`, 所以 turn 任务只能 `spawn_local` (投到当前
//!            线程的 LocalSet), 不能 `tokio::spawn` (要求 `Send`)。这正是我们要单线程
//!            runtime 的原因。

use crate::agent::Agent;
use crate::event::EventSink;
use crate::session::Session;
use std::cell::RefCell;
use std::path::PathBuf;
use std::rc::Rc;
use tokio::sync::{mpsc, oneshot};
use tokio_util::sync::CancellationToken;

/// 外部对 actor 下达的命令. 需要回执的命令自带一个 oneshot::Sender.
///
/// 学习点: 「命令 + oneshot 回执」是 actor 模式里实现「请求-应答」的惯用法 ——
///         调用方 send 命令后 await 回执接收端, 就得到了看似同步的 RPC 体验。
pub enum SessionCommand {
    /// 提交一条用户输入, 跑一个完整 turn. turn 终局经 `done` 回执。
    /// `sink` 由客户端提供: turn 内的流式文本 / 工具进展 / 权限询问都发到它,
    /// 客户端在自己那侧消费对应的接收端并渲染。
    SubmitUserInput {
        text: String,
        sink: EventSink,
        done: oneshot::Sender<TurnOutcome>,
    },
    /// 取消当前正在进行的 turn (若空闲则 no-op)。
    /// 说明: 目前 REPL 的同步 readline 阻塞时无法触发它 —— 真正的「turn 进行中取消」
    ///       要等事件流子系统引入异步输入。命令与主循环处理已就位, 先接线待用。
    #[allow(dead_code)]
    Cancel,
    /// 取一份会话快照 (给 slash 命令 / 存盘用)。
    Snapshot { reply: oneshot::Sender<Session> },
    /// 清空当前会话: 换一个全新空会话。回执新会话 id (失败为 None)。
    Clear { reply: oneshot::Sender<Option<uuid::Uuid>> },
    /// 回滚到某回合边界 (prompt_index)。回执 Ok/Err 描述。
    Rewind {
        to_prompt_index: u32,
        reply: oneshot::Sender<Result<(), String>>,
    },
    /// 分叉当前会话为一个新会话。回执新会话 id (失败为 None)。
    Fork { reply: oneshot::Sender<Option<uuid::Uuid>> },
    /// 请求 actor 退出主循环。
    Shutdown { reply: oneshot::Sender<()> },
}

/// 一个 turn 的终局。
#[derive(Debug)]
pub enum TurnOutcome {
    Completed,
    Cancelled,
    /// turn 内发生错误, 携带人类可读描述 (已格式化, 供 REPL 直接渲染)。
    Failed(String),
}

/// Actor 内部状态。
///
/// 学习点: 字段用 `RefCell` 而非 `&mut` —— 因为 turn 任务被 `spawn_local` 后, 主循环
///         与 turn 任务并发访问同一状态。单线程执行把它们的实际访问序列化, `RefCell`
///         在运行期做借用检查 (借用不跨 await 就不会 panic)。
pub struct SessionActor {
    agent: Rc<Agent>,
    session: Rc<RefCell<Session>>,
    session_dir: PathBuf,
    /// 当前 turn 的取消令牌; None 表示空闲。
    current_turn: RefCell<Option<CancellationToken>>,
    rx: mpsc::Receiver<SessionCommand>,
}

impl SessionActor {
    /// 消费自身跑主循环, 直到 channel 关闭或收到 Shutdown。
    ///
    /// 必须在 `LocalSet` 内 `spawn_local` 或直接 `.await`。
    pub async fn run(mut self) {
        while let Some(cmd) = self.rx.recv().await {
            match cmd {
                SessionCommand::SubmitUserInput { text, sink, done } => {
                    self.spawn_turn(text, sink, done);
                }
                SessionCommand::Cancel => {
                    // borrow 即取即放: 只读一下取消令牌并触发。
                    if let Some(tok) = self.current_turn.borrow().as_ref() {
                        tok.cancel();
                    }
                }
                SessionCommand::Snapshot { reply } => {
                    let _ = reply.send(self.session.borrow().clone());
                }
                SessionCommand::Clear { reply } => {
                    // 旧会话已实时落盘 (append-only), 无需再存。直接换一个新会话。
                    // borrow 即取即放: 先读旧 model, 再 borrow_mut 换新。
                    let model = self.session.borrow().model.clone();
                    match Session::new(model, &self.session_dir) {
                        Ok(fresh) => {
                            let new_id = fresh.id;
                            *self.session.borrow_mut() = fresh;
                            let _ = reply.send(Some(new_id));
                        }
                        Err(_) => {
                            let _ = reply.send(None);
                        }
                    }
                }
                SessionCommand::Rewind {
                    to_prompt_index,
                    reply,
                } => {
                    // borrow 即取即放: rewind 是同步的 (文件 IO + 重放), 不跨 await。
                    let res = self
                        .session
                        .borrow_mut()
                        .rewind(to_prompt_index)
                        .map_err(|e| e.to_string());
                    let _ = reply.send(res);
                }
                SessionCommand::Fork { reply } => {
                    // fork 出的新会话独立演进; 这里不切换当前会话, 只落盘一份副本并回 id。
                    let forked = self.session.borrow().fork(&self.session_dir);
                    let _ = reply.send(forked.ok().map(|s| s.id));
                }
                SessionCommand::Shutdown { reply } => {
                    let _ = reply.send(());
                    break;
                }
            }
        }
    }

    /// 派生一个独立本地任务跑 turn; 主循环不 await 它, 于是仍能收后续命令 (含 Cancel)。
    ///
    /// 学习点: `tokio::select!` 在「turn future」与「取消令牌」之间竞速 —— 本子系统先做
    ///         **turn 边界取消** (令牌触发时整个 run_turn future 被 drop, 停在下一个
    ///         await 点)。更细粒度的「流式中途取消」留作后续。
    fn spawn_turn(&self, text: String, sink: EventSink, done: oneshot::Sender<TurnOutcome>) {
        let token = CancellationToken::new();
        *self.current_turn.borrow_mut() = Some(token.clone());

        let agent = self.agent.clone();
        let session = self.session.clone();

        tokio::task::spawn_local(async move {
            let outcome = tokio::select! {
                r = agent.run_turn(&session, text, &sink) => match r {
                    Ok(()) => TurnOutcome::Completed,
                    Err(e) => TurnOutcome::Failed(e.to_string()),
                },
                _ = token.cancelled() => TurnOutcome::Cancelled,
            };
            // 无需存盘: 会话在 turn 过程中已由 push_* 实时 append 落盘。
            let _ = done.send(outcome);
        });
    }
}

/// 给客户端 (REPL / 未来多客户端) 用的句柄。Clone 廉价 (内含一个 mpsc Sender)。
#[derive(Clone)]
pub struct SessionHandle {
    tx: mpsc::Sender<SessionCommand>,
}

impl SessionHandle {
    /// 提交输入, 返回 (事件接收端, turn 终局的 oneshot 接收端)。
    ///
    /// 学习点: 不再是「submit().await 直接给结果」—— 因为 turn 期间会源源不断发事件,
    ///         调用方需要边收事件边渲染, 同时等待终局。所以这里把两条流都交出去,
    ///         由调用方用 `tokio::select!` / 循环同时驱动 (见 repl.rs)。
    pub async fn submit(
        &self,
        text: String,
    ) -> Option<(
        mpsc::UnboundedReceiver<crate::event::SessionEvent>,
        oneshot::Receiver<TurnOutcome>,
    )> {
        let (sink, events) = EventSink::new();
        let (done, outcome) = oneshot::channel();
        self.tx
            .send(SessionCommand::SubmitUserInput { text, sink, done })
            .await
            .ok()?;
        Some((events, outcome))
    }

    /// 取消当前 turn (fire-and-forget)。
    /// 说明: 见 `SessionCommand::Cancel` —— 待事件流子系统接入异步输入后启用。
    #[allow(dead_code)]
    pub async fn cancel(&self) {
        let _ = self.tx.send(SessionCommand::Cancel).await;
    }

    /// 取会话快照 (给 slash 命令用)。
    pub async fn snapshot(&self) -> Option<Session> {
        let (reply, rx) = oneshot::channel();
        self.tx.send(SessionCommand::Snapshot { reply }).await.ok()?;
        rx.await.ok()
    }

    /// 清空会话, 返回新会话 id。
    pub async fn clear(&self) -> Option<uuid::Uuid> {
        let (reply, rx) = oneshot::channel();
        self.tx.send(SessionCommand::Clear { reply }).await.ok()?;
        rx.await.ok().flatten()
    }

    /// 回滚到某回合边界。返回 Ok(()) 或错误描述。
    pub async fn rewind(&self, to_prompt_index: u32) -> Result<(), String> {
        let (reply, rx) = oneshot::channel();
        self.tx
            .send(SessionCommand::Rewind {
                to_prompt_index,
                reply,
            })
            .await
            .map_err(|_| "会话 actor 无响应".to_string())?;
        rx.await.map_err(|_| "会话 actor 掉线".to_string())?
    }

    /// 分叉当前会话, 返回新会话 id。
    pub async fn fork(&self) -> Option<uuid::Uuid> {
        let (reply, rx) = oneshot::channel();
        self.tx.send(SessionCommand::Fork { reply }).await.ok()?;
        rx.await.ok().flatten()
    }

    /// 请求 actor 退出并等它确认。
    pub async fn shutdown(&self) {
        let (reply, rx) = oneshot::channel();
        if self.tx.send(SessionCommand::Shutdown { reply }).await.is_ok() {
            let _ = rx.await;
        }
    }
}

/// 装配: 建 channel, 返回 (handle, actor)。actor 由调用方在 LocalSet 内 spawn_local / await。
///
/// 学习点: 把 `Agent`/`Session` 在这里一次性 move 进 `Rc`, 之后主循环与 turn 任务共享
///         同一份, 谁都不再独占。
pub fn spawn(agent: Agent, session: Session, session_dir: PathBuf) -> (SessionHandle, SessionActor) {
    let (tx, rx) = mpsc::channel(32);
    let actor = SessionActor {
        agent: Rc::new(agent),
        session: Rc::new(RefCell::new(session)),
        session_dir,
        current_turn: RefCell::new(None),
        rx,
    };
    (SessionHandle { tx }, actor)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::Message;
    use crate::config::Model;
    use crate::permission::PermissionGate;
    use crate::tools::Registry;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    /// 构造一个指向 mock server 的 Agent (无工具)。
    fn test_agent(uri: String) -> Agent {
        Agent {
            sampler: crate::sampler::build(
                "sk-test".into(),
                uri,
                &Model::deepseek_v4_flash(),
                crate::sampler::SamplerConfig::default(),
            ),
            registry: Registry::new(),
            gate: RefCell::new(PermissionGate::new(false)),
            working_dir: std::env::temp_dir(),
            hunk_tracker: RefCell::new(crate::session::checkpoint::HunkTracker::new()),
        }
    }

    /// actor 端到端: 提交一句话, 模型回 "hi" 后 stop, 应得到 Completed 且会话里多两条消息。
    ///
    /// 学习点: actor 只能跑在 LocalSet 上 (Rc/RefCell 非 Send)。测试里用
    ///         `LocalSet::run_until` 建单线程作用域, 在其中 spawn_local actor 主循环,
    ///         再用 handle 驱动 —— 这与 main.rs 的装配方式一致。
    #[tokio::test]
    async fn actor_runs_a_turn_to_completion() {
        let server = MockServer::start().await;
        let body = concat!(
            "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"},\"index\":0}]}\n\n",
            "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
            "data: [DONE]\n\n",
        );
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/event-stream")
                    .set_body_string(body),
            )
            .mount(&server)
            .await;

        let dir = tempfile::tempdir().unwrap();
        let agent = test_agent(server.uri());
        let session = Session::new(Model::deepseek_v4_flash(), tempfile::tempdir().unwrap().path()).unwrap();
        let (handle, actor) = spawn(agent, session, dir.path().to_path_buf());

        let local = tokio::task::LocalSet::new();
        local
            .run_until(async move {
                let actor_task = tokio::task::spawn_local(actor.run());

                let (mut events, outcome) = handle.submit("ping".into()).await.expect("submit");
                // 排空事件流 (不渲染, 只是驱动 turn 前进)。
                while events.recv().await.is_some() {}
                let outcome = outcome.await.expect("outcome");
                assert!(matches!(outcome, TurnOutcome::Completed), "got {outcome:?}");

                // 快照应含 user + assistant 两条。
                let snap = handle.snapshot().await.expect("snapshot");
                assert_eq!(snap.messages.len(), 2);
                assert!(matches!(&snap.messages[0], Message::User { content } if content == "ping"));

                handle.shutdown().await;
                let _ = actor_task.await;
            })
            .await;
    }

    /// /clear: 清空后应得到一个 id 不同的空会话。
    #[tokio::test]
    async fn actor_clear_starts_fresh_session() {
        let server = MockServer::start().await;
        let body = concat!(
            "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"},\"index\":0}]}\n\n",
            "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
            "data: [DONE]\n\n",
        );
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/event-stream")
                    .set_body_string(body),
            )
            .mount(&server)
            .await;

        let dir = tempfile::tempdir().unwrap();
        let agent = test_agent(server.uri());
        let session = Session::new(Model::deepseek_v4_flash(), tempfile::tempdir().unwrap().path()).unwrap();
        let old_id = session.id;
        let (handle, actor) = spawn(agent, session, dir.path().to_path_buf());

        let local = tokio::task::LocalSet::new();
        local
            .run_until(async move {
                let actor_task = tokio::task::spawn_local(actor.run());

                let (mut events, outcome) = handle.submit("ping".into()).await.expect("submit");
                while events.recv().await.is_some() {}
                let _ = outcome.await;
                let new_id = handle.clear().await.expect("clear");
                assert_ne!(new_id, old_id, "clear 应换新 id");

                let snap = handle.snapshot().await.expect("snapshot");
                assert!(snap.messages.is_empty(), "新会话应为空");

                handle.shutdown().await;
                let _ = actor_task.await;
            })
            .await;
    }
}

