//! 检查点 (Checkpoint) —— 三域回滚点 + HunkTracker。
//!
//! 设计 (参考 grok-build session/checkpoint.rs + xai-hunk-tracker):
//! 每个回合 (一条 UserMessage) 开始时打一个 checkpoint, 绑定 prompt_index。它记录本回合
//! **可能被工具改动的三个域**的「回合前状态」, rewind 时三域一起回滚:
//!   - 域1 文件快照: 本回合被写过的文件的「前一版」内容 (增量, 只存触碰过的文件);
//!   - 域2 hunk 增量: AI 编辑产生的 unified diff (展示 / 审计用, 文件域才是回滚权威);
//!   - 域3 git 锚点: 回合开始时的 HEAD (可选, 非 git 仓为 None)。
//!
//! 学习点: 为什么文件域是权威、hunk 域只展示? 因为回滚要「把内容恢复成回合前」, 直接
//!         写回快照最可靠; diff 只是给人看「改了啥」。git 域做 HEAD/index 对齐, 不替代
//!         文件恢复。三域各司其职。

use crate::errors::SessionError;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::path::{Path, PathBuf};

/// 一个回合起点的三域快照。
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Checkpoint {
    pub prompt_index: u32,
    /// 域1: 本回合触碰过的文件的回合前内容。
    pub file_snapshots: Vec<FileSnapshot>,
    /// 域2: AI 编辑的 unified diff。
    pub hunks: Vec<HunkPatch>,
    /// 域3: 回合开始时的 git HEAD (可选)。
    pub git_head: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FileSnapshot {
    pub path: PathBuf,
    /// 回合开始前的内容; None 表示文件当时不存在 (回滚 = 删除它)。
    /// 学习点: 用 Option 区分「文件本来有内容」和「文件本来不存在」两种回滚目标 ——
    ///         后者回滚要删掉工具新建的文件, 而非写空。
    pub before: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HunkPatch {
    pub path: PathBuf,
    pub unified_diff: String,
}

/// 本回合内累积文件变更, 回合末封进 Checkpoint。
///
/// 学习点: grok-build 里 hunk-tracker 是个独立 actor; 我们学习项目降级为普通 struct,
///         由 Agent 持有 (RefCell 包裹以配合 `&self`)。功能等价, 少一层并发复杂度。
#[derive(Default)]
pub struct HunkTracker {
    /// 本回合首次触碰某文件时记的 before 快照。用 path 去重。
    pending: Vec<FileSnapshot>,
    seen: HashSet<PathBuf>,
}

impl HunkTracker {
    pub fn new() -> Self {
        Self::default()
    }

    /// 工具即将写 path 前调用: 若本回合首次触碰, 记录 before 快照 (读当前内容)。
    ///
    /// 学习点: 幂等 —— 同一回合内同一文件被写多次, 只记第一次的 before (那才是「回合前」)。
    pub fn on_before_write(&mut self, path: &Path) {
        let canon = path.to_path_buf();
        if self.seen.contains(&canon) {
            return;
        }
        self.seen.insert(canon.clone());
        // 读当前内容; 读不到 (文件不存在 / 二进制) 记 None。
        let before = std::fs::read_to_string(&canon).ok();
        self.pending.push(FileSnapshot {
            path: canon,
            before,
        });
    }

    /// 回合末: 计算每个 pending 文件的 unified diff, 打包成 Checkpoint, 并清空自身。
    pub fn seal(&mut self, prompt_index: u32, git_head: Option<String>) -> Checkpoint {
        let mut hunks = Vec::new();
        for snap in &self.pending {
            let before = snap.before.clone().unwrap_or_default();
            let after = std::fs::read_to_string(&snap.path).unwrap_or_default();
            if before != after {
                let diff = similar::TextDiff::from_lines(&before, &after)
                    .unified_diff()
                    .header(
                        &format!("a/{}", snap.path.display()),
                        &format!("b/{}", snap.path.display()),
                    )
                    .to_string();
                hunks.push(HunkPatch {
                    path: snap.path.clone(),
                    unified_diff: diff,
                });
            }
        }
        let cp = Checkpoint {
            prompt_index,
            file_snapshots: std::mem::take(&mut self.pending),
            hunks,
            git_head,
        };
        self.seen.clear();
        cp
    }

    /// 本回合是否有待封存的变更 (决定要不要写 Checkpoint 记录)。
    pub fn has_pending(&self) -> bool {
        !self.pending.is_empty()
    }
}

/// 回滚一组 checkpoint 的文件域 (域1)。逆序恢复: 后改的先还原。
///
/// 学习点: 逆序是因为同一文件可能跨多个回合被改, 逆序保证最终落到最早的「回合前」内容。
///         而每个 checkpoint 内 before=None 表示「原本不存在」→ 删除。
pub fn restore_files(checkpoints: &[Checkpoint]) -> Result<(), SessionError> {
    for cp in checkpoints.iter().rev() {
        for snap in &cp.file_snapshots {
            match &snap.before {
                Some(content) => {
                    if let Some(parent) = snap.path.parent() {
                        if !parent.as_os_str().is_empty() {
                            std::fs::create_dir_all(parent).map_err(SessionError::Io)?;
                        }
                    }
                    std::fs::write(&snap.path, content).map_err(SessionError::Io)?;
                }
                None => {
                    // 原本不存在: 删掉工具新建的文件 (不存在则忽略)。
                    let _ = std::fs::remove_file(&snap.path);
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seal_captures_before_and_diff() {
        let dir = tempfile::tempdir().unwrap();
        let f = dir.path().join("a.txt");
        std::fs::write(&f, "line1\n").unwrap();

        let mut ht = HunkTracker::new();
        ht.on_before_write(&f);
        // 模拟工具写入。
        std::fs::write(&f, "line1\nline2\n").unwrap();
        let cp = ht.seal(1, None);

        assert_eq!(cp.file_snapshots.len(), 1);
        assert_eq!(cp.file_snapshots[0].before.as_deref(), Some("line1\n"));
        assert_eq!(cp.hunks.len(), 1);
        assert!(cp.hunks[0].unified_diff.contains("line2"));
    }

    #[test]
    fn restore_reverts_file_content() {
        let dir = tempfile::tempdir().unwrap();
        let f = dir.path().join("a.txt");
        std::fs::write(&f, "orig\n").unwrap();

        let mut ht = HunkTracker::new();
        ht.on_before_write(&f);
        std::fs::write(&f, "changed\n").unwrap();
        let cp = ht.seal(1, None);

        restore_files(&[cp]).unwrap();
        assert_eq!(std::fs::read_to_string(&f).unwrap(), "orig\n");
    }

    #[test]
    fn restore_deletes_newly_created_file() {
        let dir = tempfile::tempdir().unwrap();
        let f = dir.path().join("new.txt");
        // 文件回合前不存在。
        let mut ht = HunkTracker::new();
        ht.on_before_write(&f);
        std::fs::write(&f, "created by tool\n").unwrap();
        let cp = ht.seal(1, None);
        assert_eq!(cp.file_snapshots[0].before, None);

        restore_files(&[cp]).unwrap();
        assert!(!f.exists(), "回滚应删除工具新建的文件");
    }

    /// 幂等: 同一回合内同一文件写两次, 只记第一次的 before。
    #[test]
    fn on_before_write_is_idempotent_per_turn() {
        let dir = tempfile::tempdir().unwrap();
        let f = dir.path().join("a.txt");
        std::fs::write(&f, "v0\n").unwrap();
        let mut ht = HunkTracker::new();
        ht.on_before_write(&f);
        std::fs::write(&f, "v1\n").unwrap();
        ht.on_before_write(&f); // 第二次触碰, 应被忽略
        std::fs::write(&f, "v2\n").unwrap();
        let cp = ht.seal(1, None);
        assert_eq!(cp.file_snapshots.len(), 1);
        assert_eq!(cp.file_snapshots[0].before.as_deref(), Some("v0\n"));
    }
}
