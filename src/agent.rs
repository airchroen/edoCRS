//! Agent loop 编排.
//!
//! 单次 turn 的简化伪码:
//! ```text
//!   loop:
//!       chat_stream(history) -> events
//!       accumulate text + tool_calls
//!       push Assistant message
//!       if tool_calls is empty: return
//!       for tc in tool_calls:
//!           ask permission, execute, push Tool message
//! ```
//!
//! 本任务 (Task 17) 实现「空 tool_calls 单轮」happy path;
//! Task 18 / 19 增加工具执行 / 权限 / 错误恢复.

use crate::api::{FunctionCall, Message, ToolCall};
use crate::errors::AppError;
use crate::event::{EventSink, SessionEvent};
use crate::permission::PermissionGate;
use crate::sampler::{SamplerHandle, SamplingEvent, SamplingRequest};
use crate::session::checkpoint::HunkTracker;
use crate::session::Session;
use crate::tools::Registry;
use futures_util::StreamExt;
use std::cell::RefCell;
use std::rc::Rc;

pub struct Agent {
    pub sampler: SamplerHandle,
    pub registry: Registry,
    /// 权限门控.
    /// 学习点: actor 化后 run_turn 只借 `&self`, 但 gate.check 需要 `&mut self`
    ///         (维护 always_allow 缓存). 用 RefCell 把可变性下沉到字段级 —— 这是
    ///         「单线程内部可变」的惯用手法, 运行期借用检查, 无锁无 Send 要求.
    pub gate: RefCell<PermissionGate>,
    /// 工作目录 — 决定 system prompt 中 cwd 字段和 AGENTS.md 的查找位置.
    /// 学习点: 显式持有而非每次 std::env::current_dir(), 测试可注入 tempdir 隔离.
    pub working_dir: std::path::PathBuf,
    /// 本回合文件变更追踪器 (checkpoint 用)。RefCell 承载可变性 (同 gate)。
    pub hunk_tracker: RefCell<HunkTracker>,
}

impl Agent {
    /// 跑一次完整 turn: 用户输入 -> 若干次 chat (中间可能调用工具) -> 直到模型不再要求工具.
    ///
    /// 学习点: actor 化后签名从 `&mut self, &mut Session` 变成 `&self, &Rc<RefCell<Session>>`.
    ///         为什么? SessionActor 会 `spawn_local` 这个 future, 同时主循环仍持有 agent/session
    ///         的 Rc 克隆. 独占 `&mut` 在这种共享结构下无法通过借用检查, 改用 `Rc<RefCell>`
    ///         把「谁能改」的判定推迟到运行期.
    ///
    /// ⚠️ 关键约束: RefCell 的 borrow()/borrow_mut() 不能跨越 `.await` 点存活, 否则
    ///            同一线程内并发任务再借时会 panic. 下面所有 borrow 都「即取即放」:
    ///            要么在 await 前 clone 出数据, 要么在同步小块里 push 完立即 drop.
    pub async fn run_turn(
        &self,
        session: &Rc<RefCell<Session>>,
        user_input: String,
        sink: &EventSink,
    ) -> Result<(), AppError> {
        // borrow 即取即放: 追加用户消息 (同时落盘 + 更新内存)。
        session.borrow_mut().push_user(user_input)?;

        loop {
            // 学习点: 每个 turn 重新构建 system prompt — AGENTS.md 改动 / 日期推移 / cwd 信息
            //         能立即生效, 而 session.messages 保持纯净 (不进 System), session.json
            //         可读且老会话 resume 时自动用最新 system 内容.
            //
            // borrow 即取即放: 在这个同步块里克隆出 model + messages, 块结束 borrow 释放,
            // 后续 await chat_stream 时不再持有任何 session 借用.
            let (model, req_messages) = {
                let s = session.borrow();
                let system_msg = Message::System {
                    content: crate::system_prompt::build(&s.model, &self.working_dir),
                };
                let mut req = Vec::with_capacity(s.messages.len() + 1);
                req.push(system_msg);
                req.extend(s.messages.iter().cloned());
                (s.model.clone(), req)
            };

            let mut stream = self
                .sampler
                .sample(SamplingRequest {
                    model,
                    messages: req_messages,
                    tools_schema: self.registry.schemas(),
                })
                .await?;

            let mut text_buf = String::new();
            let mut reasoning_buf = String::new();
            // index -> (id, name, args buffer). 流式 tool_call 分片到达, 按 index 归并.
            let mut tc_buf: std::collections::HashMap<usize, (String, String, String)> =
                std::collections::HashMap::new();

            while let Some(ev) = stream.next().await {
                match ev? {
                    SamplingEvent::TextDelta(text) => {
                        sink.emit(SessionEvent::StreamText(text.clone()));
                        text_buf.push_str(&text);
                    }
                    SamplingEvent::ReasoningDelta(text) => {
                        sink.emit(SessionEvent::ReasoningDelta(text.clone()));
                        reasoning_buf.push_str(&text);
                    }
                    SamplingEvent::ToolCallDelta { index, id, name, arguments_fragment } => {
                        let entry = tc_buf
                            .entry(index)
                            .or_insert_with(|| (String::new(), String::new(), String::new()));
                        if let Some(i) = id { entry.0 = i; }
                        if let Some(n) = name { entry.1 = n; }
                        entry.2.push_str(&arguments_fragment);
                    }
                    SamplingEvent::Done { .. } => break,
                }
            }
            sink.emit(SessionEvent::StreamEnd);

            // 把累积的 tc_buf 转成 ToolCall 数组 (按 index 排序保证顺序确定).
            let mut indices: Vec<usize> = tc_buf.keys().copied().collect();
            indices.sort();
            let mut tool_calls: Vec<ToolCall> = Vec::new();
            for i in indices {
                let (id, name, args) = tc_buf.remove(&i).unwrap();
                tool_calls.push(ToolCall {
                    id,
                    kind: "function".into(),
                    function: FunctionCall { name, arguments: args },
                });
            }

            // borrow 即取即放: 追加 assistant 消息 (落盘 + 内存)。
            session.borrow_mut().push_assistant(
                if text_buf.is_empty() { None } else { Some(text_buf) },
                if reasoning_buf.is_empty() {
                    None
                } else {
                    Some(reasoning_buf)
                },
                tool_calls.clone(),
            )?;

            if tool_calls.is_empty() {
                // 回合结束前: 若本回合有文件改动, 封存一个 checkpoint (记录三域回滚点)。
                // 学习点: seal 在「模型不再要求工具」时做一次, 对应一个完整回合的净变更。
                self.seal_checkpoint(session)?;
                return Ok(());
            }

            // 顺序执行工具调用. execute_tool_call 内部有 await, 所以这里绝不持有 session 借用:
            // 每个结果拿到后, 在同步小块里 push 进 history.
            for tc in tool_calls {
                let result = self.execute_tool_call(&tc, sink).await;
                sink.emit(SessionEvent::ToolFinished {
                    name: tc.function.name.clone(),
                    result: result.clone(),
                });
                session.borrow_mut().push_tool_result(tc.id, result)?;
            }
        }
    }

    /// 回合末封存 checkpoint: 若 HunkTracker 有待封存变更, seal 出 Checkpoint 并 push。
    ///
    /// 学习点: borrow 即取即放 —— 先在同步小块里 seal (拿到 Checkpoint 值), 再 borrow_mut
    ///         session push。两个 borrow 不重叠, 也不跨 await。
    fn seal_checkpoint(&self, session: &Rc<RefCell<Session>>) -> Result<(), AppError> {
        let has_pending = self.hunk_tracker.borrow().has_pending();
        if !has_pending {
            return Ok(());
        }
        let prompt_index = session.borrow().prompt_index;
        let git_head = git_head(&self.working_dir);
        let cp = self.hunk_tracker.borrow_mut().seal(prompt_index, git_head);
        session.borrow_mut().push_checkpoint(cp)?;
        Ok(())
    }

    /// 真正的工具执行分支.
    ///
    /// 顺序: 显示 banner -> 查注册表 -> 必要时询问权限 -> 执行 -> 返回字符串.
    /// 错误不向上抛, 转字符串回填给模型 (`"error: ..."` 或 `"user denied"`).
    ///
    /// 学习点: 签名 `&self` (不再 `&mut self`) —— gate 的可变性由 `RefCell` 承载.
    async fn execute_tool_call(&self, tc: &ToolCall, sink: &EventSink) -> String {
        sink.emit(SessionEvent::ToolStarted {
            name: tc.function.name.clone(),
            args_preview: tc.function.arguments.chars().take(160).collect(),
        });

        let requires_permission = match self.registry.get(&tc.function.name) {
            Some(t) => t.requires_permission(),
            None => return format!("error: 未知工具 {}", tc.function.name),
        };

        if requires_permission {
            // check 现在是 async (等消费端答复)。`borrow_mut()` 产生的 RefMut 临时值活到
            // 整条语句末尾, 因而横跨 `.await`。这在本项目是**安全**的: turn 由 actor 串行
            // 提交, 一个 turn 内工具调用也顺序执行, 期间没有其它任务借用 self.gate ——
            // 所以「借用跨 await」不会触发 RefCell 的运行期 panic。
            let decision = self
                .gate
                .borrow_mut()
                .check(&tc.function.name, &tc.function.arguments, sink)
                .await;
            if matches!(decision, crate::permission::Decision::Deny) {
                return "user denied".to_string();
            }
        }

        let tool = match self.registry.get(&tc.function.name) {
            Some(t) => t,
            None => return format!("error: 未知工具 {}", tc.function.name),
        };

        // 写前留快照: 若工具会写某文件, 在执行前记录它的「回合前」内容 (checkpoint 域1)。
        if let Some(path) = tool.writes_path(&tc.function.arguments) {
            self.hunk_tracker.borrow_mut().on_before_write(&path);
        }

        match tool.execute(&tc.function.arguments).await {
            Ok(out) => out,
            Err(e) => format!("error: {e}"),
        }
    }
}

/// 读工作目录当前的 git HEAD (40 字符 sha)。非 git 仓 / git 不可用则 None。
///
/// 学习点: 用 shell git 而非 gix 依赖 —— 学习项目取「显式暴露机制」且依赖最小。
///         `git rev-parse HEAD` 是拿当前提交 sha 的标准命令。
fn git_head(dir: &std::path::Path) -> Option<String> {
    let out = std::process::Command::new("git")
        .arg("-C")
        .arg(dir)
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let sha = String::from_utf8(out.stdout).ok()?.trim().to_string();
    if sha.is_empty() {
        None
    } else {
        Some(sha)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Model;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    /// 收集一个 turn 发出的所有事件 (变体判别串), 用于断言事件序列。
    /// 学习点: 事件是 agent 对外契约的一部分, 值得像断言消息历史那样断言事件流。
    async fn collect_events(
        mut rx: tokio::sync::mpsc::UnboundedReceiver<SessionEvent>,
    ) -> Vec<&'static str> {
        let mut seq = Vec::new();
        while let Some(ev) = rx.recv().await {
            seq.push(match ev {
                SessionEvent::StreamText(_) => "text",
                SessionEvent::ReasoningDelta(_) => "reasoning",
                SessionEvent::StreamEnd => "stream_end",
                SessionEvent::ToolStarted { .. } => "tool_started",
                SessionEvent::ToolFinished { .. } => "tool_finished",
                SessionEvent::PermissionRequest { .. } => "permission",
            });
        }
        seq
    }

    /// 无工具的单轮应发出: text -> stream_end (至少一次 text, 一个 stream_end)。
    #[tokio::test]
    async fn emits_text_then_stream_end() {
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

        let agent = Agent {
            sampler: crate::sampler::build("sk-test".into(), server.uri(), &Model::deepseek_v4_flash(), crate::sampler::SamplerConfig::default()),
            registry: Registry::new(),
            gate: RefCell::new(PermissionGate::new(false)),
            working_dir: std::env::temp_dir(),
            hunk_tracker: RefCell::new(crate::session::checkpoint::HunkTracker::new()),
        };
        let session = Rc::new(RefCell::new(Session::new(Model::deepseek_v4_flash(), tempfile::tempdir().unwrap().path()).unwrap()));
        let (sink, rx) = EventSink::new();
        // 先跑完 turn (sink 全程存活), 再 drop sink 让通道关闭, 最后 drain rx。
        agent.run_turn(&session, "ping".into(), &sink).await.unwrap();
        drop(sink);
        let seq = collect_events(rx).await;
        assert!(seq.contains(&"text"), "应含 text 事件: {seq:?}");
        assert_eq!(seq.last(), Some(&"stream_end"), "末事件应是 stream_end: {seq:?}");
    }

    use crate::permission::PermissionGate;

    /// mock 单轮: 模型只说 "hi" 然后 stop, 不调用工具.
    /// 期望: history 长度 = 2 (user + assistant), assistant.content = "hi".
    #[tokio::test]
    async fn run_turn_no_tool_calls() {
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

        let agent = Agent {
            sampler: crate::sampler::build("sk-test".into(), server.uri(), &Model::deepseek_v4_flash(), crate::sampler::SamplerConfig::default()),
            registry: Registry::new(),
            gate: RefCell::new(PermissionGate::new(false)),
            working_dir: std::env::temp_dir(),
            hunk_tracker: RefCell::new(crate::session::checkpoint::HunkTracker::new()),
        };
        let session = Rc::new(RefCell::new(Session::new(Model::deepseek_v4_flash(), tempfile::tempdir().unwrap().path()).unwrap()));
        agent.run_turn(&session, "ping".into(), &EventSink::new().0).await.unwrap();

        assert_eq!(session.borrow().messages.len(), 2);
        match &session.borrow().messages[1] {
            Message::Assistant { content, tool_calls, .. } => {
                assert_eq!(content.as_deref(), Some("hi"));
                assert!(tool_calls.is_empty());
            }
            _ => panic!("expected assistant"),
        }
    }

    // ── Tool execution (Task 18) ─────────────────────────────────────────

    use crate::tools::read_file::ReadFile;

    /// 模型: 第一次返回 read_file tool_call, 第二次返回 "done".
    /// 期望 history: user, assistant(tool_call), tool, assistant("done").
    ///
    /// 学习点: wiremock 默认按声明顺序匹配 + up_to_n_times 控制次数,
    ///         正好够模拟「先 A 再 B」的场景.
    #[tokio::test]
    async fn run_turn_executes_tool_then_completes() {
        // 先准备一个临时文件给 read_file 读
        let mut tf = tempfile::NamedTempFile::new().unwrap();
        use std::io::Write;
        write!(tf, "FILE-CONTENT").unwrap();
        let file_path = tf.path().to_str().unwrap().to_string();
        // JSON 字符串嵌套需要把 path 中的反斜杠转义 (Windows 才有的问题)
        let path_escaped = file_path.replace('\\', "\\\\");

        let server = MockServer::start().await;
        let first = format!(
            "data: {{\"choices\":[{{\"delta\":{{\"tool_calls\":[{{\"index\":0,\"id\":\"call_1\",\"type\":\"function\",\"function\":{{\"name\":\"read_file\",\"arguments\":\"{{\\\"path\\\":\\\"{}\\\"}}\"}}}}]}},\"index\":0}}]}}\n\n\
             data: {{\"choices\":[{{\"delta\":{{}},\"finish_reason\":\"tool_calls\"}}]}}\n\n\
             data: [DONE]\n\n",
            path_escaped,
        );
        let second = "data: {\"choices\":[{\"delta\":{\"content\":\"done\"},\"index\":0}]}\n\n\
                      data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n\
                      data: [DONE]\n\n";

        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/event-stream")
                    .set_body_string(first.clone()),
            )
            .up_to_n_times(1)
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/event-stream")
                    .set_body_string(second),
            )
            .mount(&server)
            .await;

        let mut registry = Registry::new();
        registry.register(Box::new(ReadFile { max_bytes: 10_000 }));

        let agent = Agent {
            sampler: crate::sampler::build("sk-test".into(), server.uri(), &Model::deepseek_v4_flash(), crate::sampler::SamplerConfig::default()),
            registry,
            gate: RefCell::new(PermissionGate::new(true)),
            working_dir: std::env::temp_dir(),
            hunk_tracker: RefCell::new(crate::session::checkpoint::HunkTracker::new()),
        };
        let session = Rc::new(RefCell::new(Session::new(Model::deepseek_v4_flash(), tempfile::tempdir().unwrap().path()).unwrap()));
        agent.run_turn(&session, "请读这个文件".into(), &EventSink::new().0).await.unwrap();

        assert_eq!(session.borrow().messages.len(), 4);
        assert!(matches!(&session.borrow().messages[2], Message::Tool { content, .. } if content.contains("FILE-CONTENT")));
        assert!(matches!(&session.borrow().messages[3], Message::Assistant { content, .. } if content.as_deref() == Some("done")));
    }

    /// DeepSeek thinking mode: assistant 返回 reasoning_content + tool_calls 后,
    /// 下一次带 tool 结果继续请求时必须回传该 reasoning_content.
    #[tokio::test]
    async fn run_turn_preserves_reasoning_content_after_tool_call() {
        let server = MockServer::start().await;
        let first = "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"需要执行命令\"},\"index\":0}]}\n\n\
                     data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"type\":\"function\",\"function\":{\"name\":\"bash\",\"arguments\":\"{\\\"command\\\":\\\"echo x\\\"}\"}}]},\"index\":0}]}\n\n\
                     data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n\
                     data: [DONE]\n\n";
        let second = "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"},\"index\":0}]}\n\n\
                      data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n\
                      data: [DONE]\n\n";

        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/event-stream")
                    .set_body_string(first),
            )
            .up_to_n_times(1)
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/event-stream")
                    .set_body_string(second),
            )
            .mount(&server)
            .await;

        let mut registry = Registry::new();
        registry.register(Box::new(crate::tools::bash::Bash { max_bytes: 1000 }));

        let agent = Agent {
            sampler: crate::sampler::build("sk-test".into(), server.uri(), &Model::deepseek_v4_flash(), crate::sampler::SamplerConfig::default()),
            registry,
            gate: RefCell::new(PermissionGate::new(true)),
            working_dir: std::env::temp_dir(),
            hunk_tracker: RefCell::new(crate::session::checkpoint::HunkTracker::new()),
        };
        let session = Rc::new(RefCell::new(Session::new(Model::deepseek_v4_flash(), tempfile::tempdir().unwrap().path()).unwrap()));
        agent.run_turn(&session, "跑 echo x".into(), &EventSink::new().0).await.unwrap();

        let received = server.received_requests().await.expect("无法读取请求");
        assert_eq!(received.len(), 2, "工具调用应产生两次请求");
        let req: serde_json::Value = serde_json::from_slice(&received[1].body)
            .expect("请求体应是合法 JSON");
        let messages = req["messages"].as_array().expect("messages 应是数组");
        let assistant_with_tool = messages
            .iter()
            .find(|m| m["role"] == "assistant" && m.get("tool_calls").is_some())
            .expect("第二次请求应包含带 tool_calls 的 assistant 历史");
        assert_eq!(
            assistant_with_tool["reasoning_content"],
            "需要执行命令",
            "第二次请求必须回传 reasoning_content: {messages:?}"
        );
    }

    // ── Permission denial + Tool error (Task 19) ─────────────────────────

    /// 模型要求 bash, 用户拒绝 -> agent 把 "user denied" 回填给模型,
    /// 第二次 API 调用收到 "ok" 收尾.
    #[tokio::test]
    async fn permission_denial_feeds_back_user_denied() {
        let server = MockServer::start().await;
        let first = "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"type\":\"function\",\"function\":{\"name\":\"bash\",\"arguments\":\"{\\\"command\\\":\\\"echo x\\\"}\"}}]},\"index\":0}]}\n\n\
                     data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n\
                     data: [DONE]\n\n";
        let second = "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"},\"index\":0}]}\n\n\
                      data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n\
                      data: [DONE]\n\n";
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/event-stream")
                    .set_body_string(first),
            )
            .up_to_n_times(1)
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/event-stream")
                    .set_body_string(second),
            )
            .mount(&server)
            .await;

        let mut registry = Registry::new();
        registry.register(Box::new(crate::tools::bash::Bash { max_bytes: 1000 }));

        let agent = Agent {
            sampler: crate::sampler::build("sk-test".into(), server.uri(), &Model::deepseek_v4_flash(), crate::sampler::SamplerConfig::default()),
            registry,
            gate: RefCell::new(PermissionGate::with_test_answers(vec![crate::permission::Decision::Deny])),
            working_dir: std::env::temp_dir(),
            hunk_tracker: RefCell::new(crate::session::checkpoint::HunkTracker::new()),
        };
        let session = Rc::new(RefCell::new(Session::new(Model::deepseek_v4_flash(), tempfile::tempdir().unwrap().path()).unwrap()));
        agent.run_turn(&session, "跑 echo x".into(), &EventSink::new().0).await.unwrap();

        let tool_msg = session
            .borrow()
            .messages
            .iter()
            .find_map(|m| match m {
                Message::Tool { content, .. } => Some(content.clone()),
                _ => None,
            })
            .expect("missing tool message");
        assert_eq!(tool_msg, "user denied");
    }

    /// 工具执行失败 (read_file 不存在路径) -> 回填 "error: ...", agent 不 panic.
    #[tokio::test]
    async fn tool_error_feeds_back_error_string() {
        let server = MockServer::start().await;
        let first = "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"type\":\"function\",\"function\":{\"name\":\"read_file\",\"arguments\":\"{\\\"path\\\":\\\"/no/such/file/edocrs/probe\\\"}\"}}]},\"index\":0}]}\n\n\
                     data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n\
                     data: [DONE]\n\n";
        let second = "data: {\"choices\":[{\"delta\":{\"content\":\"sad\"},\"index\":0}]}\n\n\
                      data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n\
                      data: [DONE]\n\n";
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/event-stream")
                    .set_body_string(first),
            )
            .up_to_n_times(1)
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/event-stream")
                    .set_body_string(second),
            )
            .mount(&server)
            .await;

        let mut registry = Registry::new();
        registry.register(Box::new(ReadFile { max_bytes: 1000 }));

        let agent = Agent {
            sampler: crate::sampler::build("sk-test".into(), server.uri(), &Model::deepseek_v4_flash(), crate::sampler::SamplerConfig::default()),
            registry,
            gate: RefCell::new(PermissionGate::new(true)),
            working_dir: std::env::temp_dir(),
            hunk_tracker: RefCell::new(crate::session::checkpoint::HunkTracker::new()),
        };
        let session = Rc::new(RefCell::new(Session::new(Model::deepseek_v4_flash(), tempfile::tempdir().unwrap().path()).unwrap()));
        agent.run_turn(&session, "读不存在的文件".into(), &EventSink::new().0).await.unwrap();

        let tool_msg = session
            .borrow()
            .messages
            .iter()
            .find_map(|m| match m {
                Message::Tool { content, .. } => Some(content.clone()),
                _ => None,
            })
            .expect("missing tool message");
        assert!(tool_msg.starts_with("error:"), "expected error prefix, got: {tool_msg}");
    }

    // ── System prompt 注入 (本次新增) ────────────────────────────────────

    /// run_turn 应在 chat_stream 的请求体里, 把 role=system 作为 messages 数组首条插入,
    /// 且 session.messages 保持纯净 (不应被 System 污染).
    ///
    /// 学习点: wiremock 0.6 的 `server.received_requests().await` 返回所有命中的请求,
    ///         可以拿 raw body 反序列化做断言. 这是验证"请求长什么样"的标准手法.
    #[tokio::test]
    async fn run_turn_prepends_system_message_to_request_body() {
        let server = MockServer::start().await;
        let body = "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"},\"index\":0}]}\n\n\
                    data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n\
                    data: [DONE]\n\n";
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/event-stream")
                    .set_body_string(body),
            )
            .mount(&server)
            .await;

        let agent = Agent {
            sampler: crate::sampler::build("sk-test".into(), server.uri(), &Model::deepseek_v4_flash(), crate::sampler::SamplerConfig::default()),
            registry: Registry::new(),
            gate: RefCell::new(PermissionGate::new(false)),
            working_dir: std::env::temp_dir(),
            hunk_tracker: RefCell::new(crate::session::checkpoint::HunkTracker::new()),
        };
        let session = Rc::new(RefCell::new(Session::new(Model::deepseek_v4_flash(), tempfile::tempdir().unwrap().path()).unwrap()));
        agent.run_turn(&session, "ping".into(), &EventSink::new().0).await.unwrap();

        // 1) 检查实际发出的请求体: messages[0].role == "system"
        let received = server.received_requests().await.expect("无法读取请求");
        assert_eq!(received.len(), 1, "应只发 1 次请求 (无工具调用)");
        let req: serde_json::Value = serde_json::from_slice(&received[0].body)
            .expect("请求体应是合法 JSON");
        let messages = req
            .get("messages")
            .and_then(|m| m.as_array())
            .expect("messages 字段缺失");
        assert!(messages.len() >= 2, "至少含 system + user 两条: {messages:?}");
        assert_eq!(messages[0]["role"], "system", "首条不是 system: {messages:?}");
        let sys_content = messages[0]["content"].as_str().expect("content 应是字符串");
        assert!(
            sys_content.contains("edoCRS"),
            "system content 缺 BASE 特征: {sys_content}"
        );

        // 2) session.messages 不应被 System 污染 — 只该有 user + assistant
        assert_eq!(session.borrow().messages.len(), 2);
        assert!(
            !session
                .borrow()
                .messages
                .iter()
                .any(|m| matches!(m, Message::System { .. })),
            "session 不应保存 System: {:?}",
            session.borrow().messages
        );
    }
}
