//! edocrs-agent —— agent 核心 (对应 pi 的 `pi-agent-core` 包)。
//!
//! 职责边界: agent 循环本身 ——
//!   - `AgentEvent` 分层事件 (agent / turn / message / tool_exec);
//!   - `Tool` trait (可取消、可流式进度、content + details 双输出);
//!   - `AgentHooks` (prepare_request / before_tool_call / after_tool_call / finish_turn);
//!   - steer / follow-up 队列与 abort 语义;
//!   - 并行 / 串行工具执行。
//!
//! 它**不持有会话树**: 上层经回调 (类似 pi 的 `transformContext` / `convertToLlm`)
//! 供给本轮要发给模型的消息, 新产生的消息则经事件交回上层落盘。
//!
//! 学习点: 依赖 `edocrs-ai` 但不依赖 `edocrs` —— 所以这里的代码天然无法触碰文件系统上的
//!         会话格式或 TUI, 可以用一个 mock 采样流单独测试整个循环。
//!
//! 阶段 0: 仅占位; 阶段 2 实现。
