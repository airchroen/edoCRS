//! 会话层 —— 内存投影 + JSONL 重放日志 + resume/rewind/fork。
//!
//! 目录结构:
//!   - `session/mod.rs`   (本文件): `Session` 结构 + new/load/append 接口 + rewind/fork;
//!   - `session/log.rs`   : `UpdateRecord` 记录类型 + append/read_all/replay;
//!   - `session/checkpoint.rs`: `Checkpoint` 三域 + `HunkTracker` + 文件回滚。
//!
//! 与旧版 (整文件覆盖 session.json) 的关键差异:
//!   - 磁盘形态: `<dir>/<id>/updates.jsonl` 追加日志 (不再是 `<dir>/<id>.json`);
//!   - `messages` 由重放产生, 不再整体序列化;
//!   - 每次改动 messages 时同步 append 一条记录 (实时落盘, 不再回合末覆盖);
//!   - 新增 `prompt_index` (回合序号) 与 `checkpoints` (回滚点)。

pub mod checkpoint;
pub mod log;

use crate::api::{Message, ToolCall};
use crate::config::Model;
use crate::errors::SessionError;
use checkpoint::Checkpoint;
use chrono::{DateTime, Utc};
use log::UpdateRecord;
use std::path::{Path, PathBuf};
use uuid::Uuid;

/// 内存中的会话投影。
///
/// 学习点: `messages` 仍是 `Vec<Message>` (agent loop 直接读它), 但它现在是「日志折叠
///         出来的投影」, 不再是持久化的真身。真身是 `log_path` 指向的 updates.jsonl。
#[derive(Clone, Debug)]
pub struct Session {
    pub id: Uuid,
    pub created_at: DateTime<Utc>,
    pub model: Model,
    pub messages: Vec<Message>,
    /// 当前回合序号 (数 UserMessage 记录得到)。下一条用户输入的 prompt_index = 此值 + 1。
    pub prompt_index: u32,
    /// 已打的检查点 (rewind 用)。
    pub checkpoints: Vec<Checkpoint>,
    /// updates.jsonl 路径 = `<dir>/<id>/updates.jsonl`。
    log_path: PathBuf,
}

impl Session {
    /// 会话目录 = `<dir>/<id>/`, 日志文件 = 目录下 `updates.jsonl`。
    fn log_path_for(dir: &Path, id: &Uuid) -> PathBuf {
        dir.join(id.to_string()).join("updates.jsonl")
    }

    /// 创建一个全新的空会话, 写入 Meta 首行。
    ///
    /// 学习点: 签名从 `new(model)` 变成 `new(model, dir)` —— 因为会话一诞生就要有日志落点,
    ///         Meta 必须立即落盘, 这样即便用户一句话没说就退出, resume 也能认得这个会话。
    pub fn new(model: Model, dir: &Path) -> Result<Self, SessionError> {
        let id = Uuid::new_v4();
        let created_at = Utc::now();
        let log_path = Self::log_path_for(dir, &id);
        log::append(
            &log_path,
            &UpdateRecord::Meta {
                id,
                created_at,
                model: model.clone(),
            },
        )?;
        Ok(Self {
            id,
            created_at,
            model,
            messages: Vec::new(),
            prompt_index: 0,
            checkpoints: Vec::new(),
            log_path,
        })
    }

    /// 从 `<dir>/<id>/updates.jsonl` 读回 (重放)。
    pub fn load(dir: &Path, id: &str) -> Result<Self, SessionError> {
        let uuid = Uuid::parse_str(id).map_err(|_| SessionError::NotFound(id.to_string()))?;
        let log_path = Self::log_path_for(dir, &uuid);
        if !log_path.exists() {
            return Err(SessionError::NotFound(id.to_string()));
        }
        let recs = log::read_all(&log_path)?;
        let state = log::replay(&recs)?;
        Ok(Self {
            id: state.id,
            created_at: state.created_at,
            model: state.model,
            messages: state.messages,
            prompt_index: state.prompt_index,
            checkpoints: state.checkpoints,
            log_path,
        })
    }

    /// 追加一条用户消息: prompt_index += 1, 写记录 + 更新内存。
    pub fn push_user(&mut self, content: String) -> Result<(), SessionError> {
        self.prompt_index += 1;
        log::append(
            &self.log_path,
            &UpdateRecord::UserMessage {
                prompt_index: self.prompt_index,
                content: content.clone(),
            },
        )?;
        self.messages.push(Message::User { content });
        Ok(())
    }

    /// 追加一条 assistant 消息。
    pub fn push_assistant(
        &mut self,
        content: Option<String>,
        reasoning_content: Option<String>,
        tool_calls: Vec<ToolCall>,
    ) -> Result<(), SessionError> {
        log::append(
            &self.log_path,
            &UpdateRecord::AssistantMessage {
                content: content.clone(),
                reasoning_content: reasoning_content.clone(),
                tool_calls: tool_calls.clone(),
            },
        )?;
        self.messages.push(Message::Assistant {
            content,
            reasoning_content,
            tool_calls,
        });
        Ok(())
    }

    /// 追加一条工具结果。
    pub fn push_tool_result(
        &mut self,
        tool_call_id: String,
        content: String,
    ) -> Result<(), SessionError> {
        log::append(
            &self.log_path,
            &UpdateRecord::ToolResult {
                tool_call_id: tool_call_id.clone(),
                content: content.clone(),
            },
        )?;
        self.messages.push(Message::Tool {
            tool_call_id,
            content,
        });
        Ok(())
    }

    /// 追加一个检查点 (回合末封存文件三域时调用)。
    pub fn push_checkpoint(&mut self, cp: Checkpoint) -> Result<(), SessionError> {
        log::append(&self.log_path, &UpdateRecord::Checkpoint(cp.clone()))?;
        self.checkpoints.push(cp);
        Ok(())
    }

    /// 回滚到 to_prompt_index: 文件三域回滚 + 内存截断 + 写 Rewind 墓碑。
    ///
    /// 学习点: rewind 是「重放 + 撤销」的组合 —— 先把 to 之后回合的 checkpoint 文件域恢复,
    ///         再截断内存 messages, 最后 append 一条 Rewind 让下次重放也认这个截断。
    pub fn rewind(&mut self, to_prompt_index: u32) -> Result<(), SessionError> {
        // 收集要撤销的 checkpoint (prompt_index > to)。
        let to_undo: Vec<Checkpoint> = self
            .checkpoints
            .iter()
            .filter(|cp| cp.prompt_index > to_prompt_index)
            .cloned()
            .collect();
        // 域1 文件回滚 (git 域对齐留作后续 P6, 此处先做文件权威域)。
        checkpoint::restore_files(&to_undo)?;

        // 写 Rewind 墓碑。
        log::append(
            &self.log_path,
            &UpdateRecord::Rewind { to_prompt_index },
        )?;

        // 内存截断: 重放当前日志 (含刚写的 Rewind) 得到截断后的投影。
        let recs = log::read_all(&self.log_path)?;
        let state = log::replay(&recs)?;
        self.messages = state.messages;
        self.prompt_index = state.prompt_index;
        self.checkpoints = state.checkpoints;
        Ok(())
    }

    /// fork: 把本会话日志复制到一个新 id 目录, 返回新会话 (独立演进)。
    pub fn fork(&self, dir: &Path) -> Result<Self, SessionError> {
        let new_id = Uuid::new_v4();
        let new_path = Self::log_path_for(dir, &new_id);
        if let Some(parent) = new_path.parent() {
            std::fs::create_dir_all(parent).map_err(SessionError::Io)?;
        }
        // 复制日志内容, 但把 Meta 的 id 换成新 id (其余记录原样重放)。
        let recs = log::read_all(&self.log_path)?;
        for rec in &recs {
            let rewritten = match rec {
                UpdateRecord::Meta {
                    created_at, model, ..
                } => UpdateRecord::Meta {
                    id: new_id,
                    created_at: *created_at,
                    model: model.clone(),
                },
                other => other.clone(),
            };
            log::append(&new_path, &rewritten)?;
        }
        Self::load(dir, &new_id.to_string())
    }

    /// 找到 `dir` 下 mtime 最新的会话 (子目录含 updates.jsonl), 返回其 id。
    /// 学习点: 这就是 `--resume last` 的实现机制。目录化后扫的是 `<id>/updates.jsonl`。
    pub async fn most_recent_id(dir: &Path) -> Result<Option<String>, SessionError> {
        if !dir.exists() {
            return Ok(None);
        }
        let mut latest: Option<(std::time::SystemTime, String)> = None;
        let mut rd = tokio::fs::read_dir(dir).await.map_err(SessionError::Io)?;
        while let Some(entry) = rd.next_entry().await.map_err(SessionError::Io)? {
            let path = entry.path();
            if !path.is_dir() {
                continue;
            }
            let log = path.join("updates.jsonl");
            if !log.exists() {
                continue;
            }
            let meta = tokio::fs::metadata(&log).await.map_err(SessionError::Io)?;
            let mtime = meta.modified().map_err(SessionError::Io)?;
            let id = path.file_name().unwrap().to_string_lossy().to_string();
            match &latest {
                None => latest = Some((mtime, id)),
                Some((t, _)) if &mtime > t => latest = Some((mtime, id)),
                _ => {}
            }
        }
        Ok(latest.map(|(_, id)| id))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// new 写 Meta, 之后 load 应重放出同一个会话 (空 messages)。
    #[test]
    fn new_then_load_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let s = Session::new(Model::deepseek_v4_flash(), dir.path()).unwrap();
        let back = Session::load(dir.path(), &s.id.to_string()).unwrap();
        assert_eq!(back.id, s.id);
        assert_eq!(back.model, Model::deepseek_v4_flash());
        assert!(back.messages.is_empty());
    }

    /// push_user/assistant 后 load, 应还原消息与 prompt_index。
    #[test]
    fn push_then_reload_preserves_messages() {
        let dir = tempfile::tempdir().unwrap();
        let mut s = Session::new(Model::deepseek_v4_flash(), dir.path()).unwrap();
        s.push_user("hi".into()).unwrap();
        s.push_assistant(Some("hello".into()), None, vec![]).unwrap();

        let back = Session::load(dir.path(), &s.id.to_string()).unwrap();
        assert_eq!(back.messages.len(), 2);
        assert_eq!(back.prompt_index, 1);
        assert!(matches!(&back.messages[0], Message::User { content } if content == "hi"));
    }

    #[test]
    fn load_missing_returns_not_found() {
        let dir = tempfile::tempdir().unwrap();
        let res = Session::load(dir.path(), &Uuid::new_v4().to_string());
        assert!(matches!(res, Err(SessionError::NotFound(_))));
    }

    /// 跑两轮后 prompt_index == 2。
    #[test]
    fn prompt_index_counts_turns() {
        let dir = tempfile::tempdir().unwrap();
        let mut s = Session::new(Model::deepseek_v4_flash(), dir.path()).unwrap();
        s.push_user("t1".into()).unwrap();
        s.push_assistant(Some("r1".into()), None, vec![]).unwrap();
        s.push_user("t2".into()).unwrap();
        s.push_assistant(Some("r2".into()), None, vec![]).unwrap();
        assert_eq!(s.prompt_index, 2);
        let back = Session::load(dir.path(), &s.id.to_string()).unwrap();
        assert_eq!(back.prompt_index, 2);
        assert_eq!(back.messages.len(), 4);
    }

    /// rewind: 写文件 → checkpoint → 再改 → rewind, 文件还原、messages 截断。
    #[test]
    fn rewind_restores_files_and_truncates() {
        let dir = tempfile::tempdir().unwrap();
        let work = dir.path().join("work.txt");
        std::fs::write(&work, "v1\n").unwrap();

        let mut s = Session::new(Model::deepseek_v4_flash(), dir.path()).unwrap();
        // 回合 1
        s.push_user("改文件".into()).unwrap();
        let mut ht = checkpoint::HunkTracker::new();
        ht.on_before_write(&work);
        std::fs::write(&work, "v2\n").unwrap();
        s.push_checkpoint(ht.seal(1, None)).unwrap();
        s.push_assistant(Some("改好了".into()), None, vec![]).unwrap();
        // 回合 2
        s.push_user("再改".into()).unwrap();

        // rewind 到回合 0 (撤销回合 1 及之后)。
        s.rewind(0).unwrap();
        assert_eq!(std::fs::read_to_string(&work).unwrap(), "v1\n", "文件应还原");
        assert!(s.messages.is_empty(), "messages 应截断到回合 0");
        assert_eq!(s.prompt_index, 0);
    }

    /// fork 得到独立 id 但相同 messages。
    #[test]
    fn fork_copies_history_with_new_id() {
        let dir = tempfile::tempdir().unwrap();
        let mut s = Session::new(Model::deepseek_v4_flash(), dir.path()).unwrap();
        s.push_user("hi".into()).unwrap();
        s.push_assistant(Some("hello".into()), None, vec![]).unwrap();

        let forked = s.fork(dir.path()).unwrap();
        assert_ne!(forked.id, s.id);
        assert_eq!(forked.messages.len(), 2);
        assert!(matches!(&forked.messages[0], Message::User { content } if content == "hi"));
    }

    #[tokio::test]
    async fn most_recent_id_picks_latest_mtime() {
        let dir = tempfile::tempdir().unwrap();
        let s1 = Session::new(Model::deepseek_v4_flash(), dir.path()).unwrap();
        // 拉开 mtime。
        tokio::time::sleep(std::time::Duration::from_millis(1100)).await;
        let s2 = Session::new(Model::deepseek_v4_flash(), dir.path()).unwrap();
        let id = Session::most_recent_id(dir.path()).await.unwrap().unwrap();
        assert_eq!(id, s2.id.to_string());
        assert_ne!(id, s1.id.to_string());
    }
}
