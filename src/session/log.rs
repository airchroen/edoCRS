//! updates.jsonl 重放日志 —— 记录类型 + 追加写 + 重放。
//!
//! 设计 (参考 grok-build 的 updates.jsonl · 事件溯源):
//! 旧的持久化是「整文件覆盖写 session.json」。改成 append-only 日志后:
//!   - 每个内部事件 (用户消息 / assistant 回复 / 工具结果 / 检查点 / rewind) 是一行 JSON;
//!   - 内存里的 `messages` 是这些记录「折叠」出来的**投影**, 磁盘只存事件;
//!   - resume / rewind / fork 全部统一为「重放这条日志」。
//!
//! 学习点: 这就是 event sourcing (事件溯源) 的最小实现。相比覆盖写, 它天然可审计
//!         (完整历史都在)、可增量 (不重写整个文件)、可派生多种视图 (投影)。
//!
//! ⚠️ 追加用**同步** std::fs (O_APPEND)。为什么不用 async? 会话跑在单线程 actor 上,
//!    每条记录就一行 JSON, 阻塞时间可忽略; 而 async 追加会逼着 agent 的「borrow 后同步
//!    push」小块拆成跨 await 的结构, 反而更易踩 RefCell 借用跨 await 的坑。grok-build
//!    体量大用 async 日志, 我们学习项目取简单可靠的同步追加。

use crate::api::{Message, ToolCall};
use crate::config::Model;
use crate::errors::SessionError;
use crate::session::checkpoint::Checkpoint;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::io::Write;
use std::path::Path;
use uuid::Uuid;

/// updates.jsonl 里的一条记录 = 一个内部事件。每行一个 JSON 对象。
///
/// 学习点: internally-tagged enum (`tag = "kind"`) 让每行 JSON 带一个 "kind" 字段
///         区分变体, 既可读又可被 serde 精确还原。
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum UpdateRecord {
    /// 会话头, 仅文件首行出现一次。存不可变元信息。
    Meta {
        id: Uuid,
        created_at: DateTime<Utc>,
        model: Model,
    },
    /// 一条用户消息。重放时: 先 `prompt_index = 记录里的值`, 再 push。
    /// 这就是「回合边界」—— 数这类记录即可恢复 turn 数。
    UserMessage { prompt_index: u32, content: String },
    /// 一条 assistant 消息 (可能带 reasoning + tool_calls)。
    AssistantMessage {
        content: Option<String>,
        reasoning_content: Option<String>,
        tool_calls: Vec<ToolCall>,
    },
    /// 一条工具结果。
    ToolResult {
        tool_call_id: String,
        content: String,
    },
    /// 检查点: 在某个 prompt_index 处打包三域回滚点 (见 checkpoint.rs)。
    Checkpoint(Checkpoint),
    /// rewind 墓碑: 逻辑上「截断到 to_prompt_index」。重放遇到它就丢弃其后的消息记录。
    /// 学习点: 追加日志不物理删除, 用 tombstone 记录「撤销」—— 保留完整审计轨迹。
    Rewind { to_prompt_index: u32 },
}

/// 把单条记录序列化成一行 JSON + '\n', 同步 O_APPEND 追加。不覆盖不 seek。
///
/// 学习点: `OpenOptions::append(true)` 让每次写都原子地落到文件尾 (OS 保证单次 write
///         不与其它 append 交错), 无需自己 seek。
pub fn append(path: &Path, rec: &UpdateRecord) -> Result<(), SessionError> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(SessionError::Io)?;
    }
    let mut line = serde_json::to_vec(rec)?;
    line.push(b'\n');
    let mut f = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(SessionError::Io)?;
    f.write_all(&line).map_err(SessionError::Io)?;
    Ok(())
}

/// 读全部记录 (逐行反序列化)。空行跳过; 坏行报错 (日志损坏应显式暴露)。
pub fn read_all(path: &Path) -> Result<Vec<UpdateRecord>, SessionError> {
    let text = std::fs::read_to_string(path).map_err(SessionError::Io)?;
    let mut out = Vec::new();
    for (i, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let rec: UpdateRecord = serde_json::from_str(line)
            .map_err(|e| SessionError::Replay(format!("第 {} 行损坏: {e}", i + 1)))?;
        out.push(rec);
    }
    Ok(out)
}

/// 重放的产物: 折叠后的会话投影。
pub struct ReplayState {
    pub id: Uuid,
    pub created_at: DateTime<Utc>,
    pub model: Model,
    pub messages: Vec<Message>,
    pub prompt_index: u32,
    pub checkpoints: Vec<Checkpoint>,
}

/// 把记录序列折叠成会话投影。
///
/// 两趟算法:
///   1. 先扫一遍找出最终有效的 prompt_index 上界 (处理 Rewind 截断)。多个 Rewind 取最小。
///   2. 再折叠: 属于「被撤销回合」(prompt_index > 上界) 的消息记录一律跳过。
///
/// 学习点: 为什么两趟? 因为 Rewind 记录出现在日志**后面**, 但它要作废前面的记录。
///         一趟折叠时还不知道后面会不会 rewind, 所以先扫描确定截断点, 再折叠。
pub fn replay(records: &[UpdateRecord]) -> Result<ReplayState, SessionError> {
    // ── 第 1 趟: 求有效 prompt_index 上界 ──
    // 无 Rewind 时上界 = u32::MAX (不截断)。
    let mut cutoff = u32::MAX;
    for rec in records {
        if let UpdateRecord::Rewind { to_prompt_index } = rec {
            cutoff = cutoff.min(*to_prompt_index);
        }
    }

    // ── 第 2 趟: 折叠 ──
    let mut id = Uuid::nil();
    let mut created_at = Utc::now();
    let mut model = Model::default();
    let mut messages = Vec::new();
    let mut prompt_index = 0u32;
    let mut checkpoints = Vec::new();
    // 当前记录归属的回合序号 (由最近一条 UserMessage 决定)。用于判断是否被 cutoff 撤销。
    let mut cur_turn = 0u32;
    let mut seen_meta = false;

    for rec in records {
        match rec {
            UpdateRecord::Meta {
                id: mid,
                created_at: mc,
                model: mm,
            } => {
                id = *mid;
                created_at = *mc;
                model = mm.clone();
                seen_meta = true;
            }
            UpdateRecord::UserMessage {
                prompt_index: pi,
                content,
            } => {
                cur_turn = *pi;
                if cur_turn > cutoff {
                    continue; // 被 rewind 撤销的回合, 跳过
                }
                prompt_index = cur_turn;
                messages.push(Message::User {
                    content: content.clone(),
                });
            }
            UpdateRecord::AssistantMessage {
                content,
                reasoning_content,
                tool_calls,
            } => {
                if cur_turn > cutoff {
                    continue;
                }
                messages.push(Message::Assistant {
                    content: content.clone(),
                    reasoning_content: reasoning_content.clone(),
                    tool_calls: tool_calls.clone(),
                });
            }
            UpdateRecord::ToolResult {
                tool_call_id,
                content,
            } => {
                if cur_turn > cutoff {
                    continue;
                }
                messages.push(Message::Tool {
                    tool_call_id: tool_call_id.clone(),
                    content: content.clone(),
                });
            }
            UpdateRecord::Checkpoint(cp) => {
                // 被撤销回合之后的 checkpoint 也丢弃 (rewind 会一起清)。
                if cp.prompt_index > cutoff {
                    continue;
                }
                checkpoints.push(cp.clone());
            }
            UpdateRecord::Rewind { .. } => {
                // 第 1 趟已处理; 折叠阶段跳过。
            }
        }
    }

    if !seen_meta {
        return Err(SessionError::Replay("日志缺少 Meta 首行".into()));
    }

    Ok(ReplayState {
        id,
        created_at,
        model,
        messages,
        prompt_index,
        checkpoints,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn meta() -> UpdateRecord {
        UpdateRecord::Meta {
            id: Uuid::new_v4(),
            created_at: Utc::now(),
            model: Model::deepseek_v4_flash(),
        }
    }

    /// append 三条记录后 read_all + replay, 应还原成 2 条消息、prompt_index=1。
    #[test]
    fn append_then_replay_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("s/updates.jsonl");
        append(&path, &meta()).unwrap();
        append(
            &path,
            &UpdateRecord::UserMessage {
                prompt_index: 1,
                content: "hi".into(),
            },
        )
        .unwrap();
        append(
            &path,
            &UpdateRecord::AssistantMessage {
                content: Some("hello".into()),
                reasoning_content: None,
                tool_calls: vec![],
            },
        )
        .unwrap();

        let recs = read_all(&path).unwrap();
        let state = replay(&recs).unwrap();
        assert_eq!(state.messages.len(), 2);
        assert_eq!(state.prompt_index, 1);
        assert!(matches!(&state.messages[0], Message::User { content } if content == "hi"));
    }

    /// Rewind 应作废其后 (prompt_index > cutoff) 的消息。
    #[test]
    fn rewind_truncates_later_turns() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("s/updates.jsonl");
        append(&path, &meta()).unwrap();
        for pi in 1..=3 {
            append(
                &path,
                &UpdateRecord::UserMessage {
                    prompt_index: pi,
                    content: format!("turn {pi}"),
                },
            )
            .unwrap();
            append(
                &path,
                &UpdateRecord::AssistantMessage {
                    content: Some(format!("reply {pi}")),
                    reasoning_content: None,
                    tool_calls: vec![],
                },
            )
            .unwrap();
        }
        // rewind 回到回合 1: 回合 2、3 被撤销。
        append(&path, &UpdateRecord::Rewind { to_prompt_index: 1 }).unwrap();

        let state = replay(&read_all(&path).unwrap()).unwrap();
        // 只剩回合 1 的 user + assistant = 2 条。
        assert_eq!(state.messages.len(), 2);
        assert_eq!(state.prompt_index, 1);
    }

    /// 缺 Meta 的日志应报 Replay 错误。
    #[test]
    fn missing_meta_errors() {
        let recs = vec![UpdateRecord::UserMessage {
            prompt_index: 1,
            content: "x".into(),
        }];
        assert!(matches!(replay(&recs), Err(SessionError::Replay(_))));
    }
}
