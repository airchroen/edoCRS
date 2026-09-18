//! SessionEvent —— 会话 turn 对外的单向进展事件流 (ACP 风)。
//!
//! 设计动机 (参考 grok-build 的 ACP 语义): 旧的 `run_turn` 把渲染硬编码进 agent loop
//! (`ui::print_stream` 等直接打 stdout), 权限询问同步阻塞读 stdin。这让 agent 无法被
//! 非终端客户端复用, 也让「谁决定怎么渲染」与「发生了什么」纠缠在一起。
//!
//! 改法: turn 只管**发生了什么** —— 把每个进展 `emit` 成 `SessionEvent` 打进 channel;
//! **怎么渲染**是消费端 (REPL / 未来 TUI) 的事。二者用 channel 解耦。
//!
//! 权限是双向的一环: turn 发 `PermissionRequest{reply}`, 消费端拿到用户答复后经 `reply`
//! (oneshot) 回传 —— 事件里塞一个 oneshot::Sender 就实现了「请求-应答」。

use crate::permission::Decision;
use tokio::sync::{mpsc, oneshot};

/// turn 在执行中对外广播的事件。
///
/// 学习点: 变体都用 `String` 拥有数据, 不借 agent 内部缓冲的引用 —— 事件要跨 channel
///         到别的 await 点, 借用活不了那么久。
#[derive(Debug)]
pub enum SessionEvent {
    /// 模型正文增量 (原 `ui::print_stream`)。
    StreamText(String),
    /// 思维链增量 (DeepSeek reasoning_content; 消费端可选择灰显或折叠)。
    ReasoningDelta(String),
    /// 一段模型正文流结束 (给消费端补换行的时机)。
    StreamEnd,
    /// 某工具即将执行 (原 `ui::show_tool_call`)。
    ToolStarted { name: String, args_preview: String },
    /// 工具执行完毕, 携带回填给模型的字符串结果 (原 `ui::show_tool_result`)。
    ToolFinished { name: String, result: String },
    /// 需要用户授权。`reply` 是一次性回执通道: 消费端拿到 Decision 后 send 回来。
    ///
    /// 学习点: turn 侧 `await` reply 的接收端, 消费端 send 后 turn 才继续 —— 这把
    ///         「阻塞等用户」从 agent 内部挪到了事件边界上, agent 自己不碰 stdin。
    PermissionRequest {
        name: String,
        args_preview: String,
        reply: oneshot::Sender<Decision>,
    },
}

/// turn 发事件用的句柄。内部包一个 mpsc::Sender。
///
/// 学习点: 用 newtype 而非到处传裸 Sender —— 给发送方法起业务名, 并把「发送失败
///         (接收端已挂)」统一吞掉 (消费端没了, turn 也没必要继续渲染)。
#[derive(Clone)]
pub struct EventSink(mpsc::UnboundedSender<SessionEvent>);

impl EventSink {
    /// 建一对 (sink, 接收端)。用 unbounded: turn 发事件不该被消费端背压阻塞
    /// (渲染慢不能拖慢模型流)。
    pub fn new() -> (Self, mpsc::UnboundedReceiver<SessionEvent>) {
        let (tx, rx) = mpsc::unbounded_channel();
        (Self(tx), rx)
    }

    /// 发一个事件; 接收端挂了就静默丢弃。
    pub fn emit(&self, ev: SessionEvent) {
        let _ = self.0.send(ev);
    }

    /// 发一条权限请求并 await 应答。返回 None 表示消费端未应答 (通道断) —— 按拒绝处理。
    pub async fn request_permission(&self, name: &str, args: &str) -> Option<Decision> {
        let (reply, rx) = oneshot::channel();
        let preview: String = args.chars().take(200).collect();
        self.emit(SessionEvent::PermissionRequest {
            name: name.to_string(),
            args_preview: preview,
            reply,
        });
        rx.await.ok()
    }
}
