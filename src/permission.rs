//! 权限询问门控.
//!
//! 设计 (见 spec §5):
//! - read_file 不询问 (Tool::requires_permission 返回 false, agent.rs 不会调到这里).
//! - write_file/bash 默认询问.
//! - 用户可以选 'a' 在本会话内对该工具名「全允许」.
//! - --yolo flag 全程跳过询问.
//!
//! 事件化 (子系统 2): `check` 不再自己读 stdin, 而是经 `EventSink` 发一条
//! `PermissionRequest` 事件, await 消费端 (REPL) 经 oneshot 回传的 Decision。
//! 这样 gate 与终端 IO 解耦, 未来 TUI 客户端也能复用同一条路径。

use crate::event::EventSink;
use std::collections::HashSet;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Decision {
    Allow,
    AllowAll,
    Deny,
}

pub struct PermissionGate {
    always_allow: HashSet<String>,
    yolo: bool,
    /// 用于测试的预录答案队列. 生产代码请用默认 (空 Vec).
    test_answers: Vec<Decision>,
}

impl PermissionGate {
    pub fn new(yolo: bool) -> Self {
        Self {
            always_allow: HashSet::new(),
            yolo,
            test_answers: Vec::new(),
        }
    }

    /// 测试构造: 预录答案队列, 不去询问消费端.
    /// 学习点: #[cfg(test)] 使这个方法只在测试编译里存在, 生产二进制零体积.
    #[cfg(test)]
    pub fn with_test_answers(answers: Vec<Decision>) -> Self {
        Self {
            always_allow: HashSet::new(),
            yolo: false,
            test_answers: answers,
        }
    }

    /// 检查是否允许执行某工具.
    /// - 如果之前选过 AllowAll, 直接放行;
    /// - 如果是 yolo 模式, 直接放行;
    /// - 否则经 sink 发 PermissionRequest 事件, await 消费端答复 (生产)
    ///   或弹出预录答案 (测试).
    ///
    /// 学习点: 签名变 async + 收 `&EventSink`。快路径 (yolo/always_allow/test) 不发事件,
    ///         也就不 await —— async fn 里不含 await 点时会立即完成, 零开销。
    pub async fn check(
        &mut self,
        tool_name: &str,
        args_preview: &str,
        sink: &EventSink,
    ) -> Decision {
        if self.yolo || self.always_allow.contains(tool_name) {
            return Decision::Allow;
        }
        let d = if !self.test_answers.is_empty() {
            self.test_answers.remove(0)
        } else {
            // 通道断 (消费端没了) 按拒绝处理 —— 安全默认。
            sink.request_permission(tool_name, args_preview)
                .await
                .unwrap_or(Decision::Deny)
        };
        if d == Decision::AllowAll {
            self.always_allow.insert(tool_name.to_string());
        }
        d
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 造一个「无人接收」的 sink: 快路径测试不会真的发事件, 所以接收端可丢弃。
    fn dummy_sink() -> EventSink {
        EventSink::new().0
    }

    /// yolo 永远 Allow, 不动 always_allow 集合.
    #[tokio::test]
    async fn yolo_always_allows() {
        let mut g = PermissionGate::new(true);
        assert_eq!(g.check("bash", "ls", &dummy_sink()).await, Decision::Allow);
    }

    /// allow-all 应该按工具名缓存.
    /// 为什么这样测: 这是 "用户体验 vs 安全" 取舍点的关键不变式.
    #[tokio::test]
    async fn allow_all_caches_per_tool_name() {
        let mut g = PermissionGate::with_test_answers(vec![Decision::AllowAll]);
        let sink = dummy_sink();
        assert_eq!(g.check("bash", "ls -la", &sink).await, Decision::AllowAll);
        // 第二次不再询问, 直接 Allow (注意是 Allow 而非 AllowAll, 因为缓存命中走快路径)
        assert_eq!(g.check("bash", "rm -rf /tmp/x", &sink).await, Decision::Allow);
    }

    /// allow-all 不能跨工具泄漏.
    #[tokio::test]
    async fn allow_all_does_not_leak_across_tools() {
        let mut g = PermissionGate::with_test_answers(vec![Decision::AllowAll, Decision::Deny]);
        let sink = dummy_sink();
        g.check("bash", "ls", &sink).await;
        // write_file 不在 allow 列表, 询问拿到 Deny
        assert_eq!(g.check("write_file", "/tmp/x", &sink).await, Decision::Deny);
    }

    #[tokio::test]
    async fn deny_returns_deny() {
        let mut g = PermissionGate::with_test_answers(vec![Decision::Deny]);
        assert_eq!(g.check("bash", "rm -rf /", &dummy_sink()).await, Decision::Deny);
    }
}
