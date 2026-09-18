//! write_file 工具: 写文本到指定路径.
//!
//! 设计:
//! - 询问权限 (写是不可逆副作用).
//! - 父目录不存在则报错, 不自动创建 (防误操作).

use crate::errors::ToolError;
use crate::tools::Tool;
use async_trait::async_trait;
use serde::Deserialize;

pub struct WriteFile;

#[derive(Deserialize)]
struct Args {
    path: String,
    content: String,
}

#[async_trait]
impl Tool for WriteFile {
    fn name(&self) -> &'static str { "write_file" }

    fn schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "function",
            "function": {
                "name": "write_file",
                "description": "把 content 写到 path. 父目录必须存在 (本工具不创建目录).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "content": {"type": "string"}
                    },
                    "required": ["path", "content"]
                }
            }
        })
    }

    fn requires_permission(&self) -> bool { true }

    /// write_file 会写 args.path —— 报给 checkpoint 用于回合前快照。
    fn writes_path(&self, arguments: &str) -> Option<std::path::PathBuf> {
        let args: Args = serde_json::from_str(arguments).ok()?;
        Some(std::path::PathBuf::from(args.path))
    }

    async fn execute(&self, arguments: &str) -> Result<String, ToolError> {
        let args: Args = serde_json::from_str(arguments)
            .map_err(|e| ToolError::BadArgs(e.to_string()))?;
        let path = std::path::Path::new(&args.path);
        if let Some(parent) = path.parent() {
            // 学习点: parent.as_os_str().is_empty() 排除 path 是单一文件名的情况
            //         (例如 "x.txt", parent 是 ""), 此时不需要检查存在性.
            if !parent.as_os_str().is_empty() && !parent.exists() {
                return Err(ToolError::Failed(format!(
                    "父目录不存在: {}", parent.display()
                )));
            }
        }
        let n = args.content.len();
        tokio::fs::write(path, &args.content)
            .await
            .map_err(|e| ToolError::Failed(format!("写文件失败: {e}")))?;
        Ok(format!("wrote {n} bytes to {}", args.path))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn writes_to_existing_dir() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("hello.txt");
        let args = serde_json::json!({
            "path": path.to_str().unwrap(),
            "content": "hi"
        }).to_string();
        let tool = WriteFile;
        let out = tool.execute(&args).await.unwrap();
        assert!(out.contains("wrote 2 bytes"));
        let read_back = std::fs::read_to_string(&path).unwrap();
        assert_eq!(read_back, "hi");
    }

    #[tokio::test]
    async fn fails_when_parent_missing() {
        let args = r#"{"path":"/nonexistent/edocrs/x.txt", "content":"hi"}"#;
        let tool = WriteFile;
        let res = tool.execute(args).await;
        assert!(res.is_err());
    }
}
