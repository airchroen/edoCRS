//! edocrs-ai —— 模型接入层 (对应 pi 的 `pi-ai` 包)。
//!
//! 职责边界: 只关心「怎么和 LLM 说话」——
//!   - provider 注册表 (openai / openrouter / deepseek / ollama + 用户自定义);
//!   - `provider/model-id` 形式的 Model 解析;
//!   - OpenAI Chat Completions 的请求构造、SSE 流式解析、重试与空闲超时。
//!
//! 它**不知道** agent 循环、工具执行、会话文件、TUI 的存在。
//!
//! 学习点: 把「和外部服务通信」单独拆成一个 crate, 依赖方向由编译器强制 ——
//!         上层 (`edocrs-agent`、`edocrs`) 能用它, 它却 `use` 不到上层任何东西。
//!         这比「同一 crate 里靠自觉分模块」可靠得多。
//!
//! 阶段 0: 仅占位; 阶段 1 把 sampler / api 迁入并改造。
