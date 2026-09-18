//! bash 工具: 用 bash -c 执行命令, 默认 30s 超时, stdout+stderr 合并.

use crate::errors::ToolError;
use crate::tools::Tool;
use async_trait::async_trait;
use serde::Deserialize;
use std::time::Duration;
use tokio::process::Command;

pub struct Bash {
    pub max_bytes: usize,
}

#[derive(Deserialize)]
struct Args {
    command: String,
    #[serde(default = "default_timeout")]
    timeout_secs: u64,
}

fn default_timeout() -> u64 { 30 }

#[async_trait]
impl Tool for Bash {
    fn name(&self) -> &'static str { "bash" }

    fn schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "function",
            "function": {
                "name": "bash",
                "description": "用 bash -c 执行命令. stdout 与 stderr 合并返回. 默认 30s 超时.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string"},
                        "timeout_secs": {"type": "integer", "default": 30}
                    },
                    "required": ["command"]
                }
            }
        })
    }

    fn requires_permission(&self) -> bool { true }

    async fn execute(&self, arguments: &str) -> Result<String, ToolError> {
        let args: Args = serde_json::from_str(arguments)
            .map_err(|e| ToolError::BadArgs(e.to_string()))?;

        // 学习点: tokio::process 是 std::process 的异步版.
        //         我们用 -c 把整条命令交给 bash 解释 (支持管道 / 重定向).
        let mut cmd = Command::new("bash");
        cmd.arg("-c").arg(&args.command);
        cmd.stdout(std::process::Stdio::piped());
        cmd.stderr(std::process::Stdio::piped());

        let fut = cmd.output();
        // 学习点: tokio::time::timeout 把任意 future 包成"在 X 时间内完成或超时".
        //         超时后 tokio 会 drop 内部 future, 但子进程不一定立刻死掉
        //         (操作系统侧可能继续跑直到自己 exit). 学习项目接受这个限制.
        let output = tokio::time::timeout(Duration::from_secs(args.timeout_secs), fut)
            .await
            .map_err(|_| ToolError::Timeout(args.timeout_secs))?
            .map_err(|e| ToolError::Failed(format!("启动失败: {e}")))?;

        let mut combined = String::new();
        combined.push_str(&String::from_utf8_lossy(&output.stdout));
        combined.push_str(&String::from_utf8_lossy(&output.stderr));
        if combined.len() > self.max_bytes {
            let kept = self.max_bytes;
            let extra = combined.len() - kept;
            combined.truncate(kept);
            combined.push_str(&format!("\n... [截断, 还有 {extra} 字节未显示]\n"));
        }
        if !output.status.success() {
            combined.push_str(&format!("\n[exit code: {}]", output.status.code().unwrap_or(-1)));
        }
        Ok(combined)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn executes_simple_command() {
        let tool = Bash { max_bytes: 10_000 };
        let args = r#"{"command":"echo hello world"}"#;
        let out = tool.execute(args).await.unwrap();
        assert!(out.contains("hello world"));
    }

    #[tokio::test]
    async fn captures_stderr() {
        let tool = Bash { max_bytes: 10_000 };
        // 学习点: 1>&2 把 stdout 重定向到 stderr, 所以 to-err 会出现在 stderr.
        let args = r#"{"command":"echo to-err 1>&2; echo to-out"}"#;
        let out = tool.execute(args).await.unwrap();
        assert!(out.contains("to-err"));
        assert!(out.contains("to-out"));
    }

    #[tokio::test]
    async fn enforces_timeout() {
        let tool = Bash { max_bytes: 10_000 };
        let args = r#"{"command":"sleep 5", "timeout_secs":1}"#;
        let res = tool.execute(args).await;
        assert!(matches!(res, Err(ToolError::Timeout(1))));
    }
}
