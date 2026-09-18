//! 工具系统 — trait 定义 + registry.
//!
//! 学习点: trait + Box<dyn Trait> = 动态派发 (dynamic dispatch).
//!         每个工具一个 struct, 都实现 Tool, 用 HashMap<&str, Box<dyn Tool>> 注册.
//!
//! 子模块 (后续 task 加入):
//!   - read_file: 读文本文件
//!   - write_file: 写文本文件
//!   - bash: 执行 shell 命令

pub mod bash;
pub mod read_file;
pub mod write_file;

use crate::errors::ToolError;
use async_trait::async_trait;
use std::collections::HashMap;

/// 工具抽象.
///
/// `Send + Sync` 是为了能在 tokio::spawn 后被跨线程使用.
///
/// 学习点: #[async_trait] 是 async-trait 板条箱给我们的: Rust 当前还不能在原生 trait
///         里直接写 async fn, 必须用这个宏包装一下.
#[async_trait]
pub trait Tool: Send + Sync {
    fn name(&self) -> &'static str;

    /// 给模型看的 JSON Schema (OpenAI function-calling 格式 — type=function 包装).
    fn schema(&self) -> serde_json::Value;

    /// 是否需要在执行前询问用户.
    fn requires_permission(&self) -> bool;

    /// 真正执行. arguments 是 raw JSON 字符串, 自行 parse.
    async fn execute(&self, arguments: &str) -> Result<String, ToolError>;

    /// 若本工具会写某个文件, 返回该路径 (供 checkpoint 的 HunkTracker 在写前留快照)。
    /// 默认 None (只读工具 / 无文件副作用)。write_file override 为目标 path。
    ///
    /// 学习点: 给 trait 加带默认实现的方法是「开放扩展」的惯用法 —— 已有工具不受影响,
    ///         只有关心「写前快照」的工具才 override。
    fn writes_path(&self, _arguments: &str) -> Option<std::path::PathBuf> {
        None
    }
}

/// 工具注册表.
pub struct Registry {
    tools: HashMap<&'static str, Box<dyn Tool>>,
}

impl Registry {
    pub fn new() -> Self {
        Self { tools: HashMap::new() }
    }

    pub fn register(&mut self, tool: Box<dyn Tool>) {
        self.tools.insert(tool.name(), tool);
    }

    pub fn get(&self, name: &str) -> Option<&dyn Tool> {
        // 学习点: HashMap<_, Box<dyn Tool>>::get 返回 Option<&Box<dyn Tool>>.
        //         我们用 |b| &**b 解一层 Box, 再借成 &dyn Tool.
        self.tools.get(name).map(|b| &**b)
    }

    pub fn schemas(&self) -> Vec<serde_json::Value> {
        self.tools.values().map(|t| t.schema()).collect()
    }
}

impl Default for Registry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 用于测试的 Echo 工具: 直接把 arguments 字符串原样返回.
    struct Echo;

    #[async_trait]
    impl Tool for Echo {
        fn name(&self) -> &'static str { "echo" }
        fn schema(&self) -> serde_json::Value {
            serde_json::json!({"type":"function","function":{"name":"echo"}})
        }
        fn requires_permission(&self) -> bool { false }
        async fn execute(&self, args: &str) -> Result<String, ToolError> {
            Ok(args.to_string())
        }
    }

    /// 注册和查找应找回同一个工具.
    /// 为什么这样测: dispatch 路径 (Registry::get -> Tool::execute) 是 agent loop 的核心.
    #[tokio::test]
    async fn register_and_dispatch() {
        let mut r = Registry::new();
        r.register(Box::new(Echo));
        let t = r.get("echo").expect("missing");
        let out = t.execute("hi").await.unwrap();
        assert_eq!(out, "hi");
    }

    /// schemas 返回所有已注册工具的 schema.
    #[test]
    fn schemas_collects_all() {
        let mut r = Registry::new();
        r.register(Box::new(Echo));
        let s = r.schemas();
        assert_eq!(s.len(), 1);
        assert_eq!(s[0]["function"]["name"], "echo");
    }
}
