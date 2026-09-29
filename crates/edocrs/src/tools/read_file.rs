//! read_file 工具: 读普通文本文件 (二进制返回错误).
//!
//! 设计:
//! - 不询问权限 (读是只读副作用, 风险低).
//! - 输出截断到 max_bytes 防止 context 爆炸.
//! - 检测 NUL 字节作为二进制信号, 拒绝读取.

use crate::errors::ToolError;
use crate::tools::Tool;
use async_trait::async_trait;
use serde::Deserialize;

/// read_file 工具实例 — 仅持有截断阈值, 其他无状态.
pub struct ReadFile {
    pub max_bytes: usize,
}

#[derive(Deserialize)]
struct Args {
    path: String,
}

#[async_trait]
impl Tool for ReadFile {
    fn name(&self) -> &'static str { "read_file" }

    fn schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "读取本地文本文件. 二进制文件会失败. 输出长度受 max_tool_output_bytes 截断.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": { "type": "string", "description": "绝对或相对路径" }
                    },
                    "required": ["path"]
                }
            }
        })
    }

    fn requires_permission(&self) -> bool { false }

    async fn execute(&self, arguments: &str) -> Result<String, ToolError> {
        let args: Args = serde_json::from_str(arguments)
            .map_err(|e| ToolError::BadArgs(e.to_string()))?;
        let bytes = tokio::fs::read(&args.path)
            .await
            .map_err(|e| ToolError::Failed(format!("读文件失败 {}: {}", args.path, e)))?;
        // 检测是否二进制: 简化策略 — 如果包含 NUL 字节则视作二进制.
        // 学习点: 真实的二进制检测要复杂得多 (chardet/file 命令), 学习项目用最朴素的够了.
        if bytes.contains(&0u8) {
            return Err(ToolError::Failed(format!(
                "{} 看起来是二进制, 拒绝读取", args.path
            )));
        }
        let mut text = String::from_utf8(bytes)
            .map_err(|e| ToolError::Failed(format!("非 UTF-8: {e}")))?;
        if text.len() > self.max_bytes {
            let kept = self.max_bytes;
            let extra = text.len() - kept;
            text.truncate(kept);
            text.push_str(&format!("\n... [截断, 还有 {extra} 字节未显示]\n"));
        }
        Ok(text)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[tokio::test]
    async fn reads_existing_file() {
        let mut f = tempfile::NamedTempFile::new().unwrap();
        writeln!(f, "hello").unwrap();
        let path = f.path().to_str().unwrap().to_string();
        let tool = ReadFile { max_bytes: 1000 };
        let args = serde_json::json!({ "path": path }).to_string();
        let out = tool.execute(&args).await.unwrap();
        assert!(out.contains("hello"));
    }

    #[tokio::test]
    async fn missing_file_returns_error() {
        let tool = ReadFile { max_bytes: 1000 };
        let args = r#"{"path":"/nonexistent/edocrs/probe"}"#;
        let res = tool.execute(args).await;
        assert!(res.is_err());
    }

    /// 文件超过 max_bytes 应该被截断且附 [截断] 标记.
    #[tokio::test]
    async fn truncates_long_file() {
        let mut f = tempfile::NamedTempFile::new().unwrap();
        for _ in 0..1000 {
            writeln!(f, "0123456789").unwrap();
        }
        let tool = ReadFile { max_bytes: 100 };
        let args = serde_json::json!({ "path": f.path() }).to_string();
        let out = tool.execute(&args).await.unwrap();
        assert!(out.contains("[截断"));
    }
}
