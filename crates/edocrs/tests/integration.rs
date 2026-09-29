//! 端到端集成测试 — 当前为占位.
//!
//! 实际的端到端覆盖在 `src/agent.rs` 的单元测试里 (wiremock 模拟 SSE 服务器,
//! 跑「user 输入 -> 模型调用工具 -> 工具结果回填 -> 模型收尾」全链路).
//!
//! 学习扩展: Rust 的 tests/ 目录是独立 crate, 只能链接到本 crate 的 lib target.
//! 当前项目是纯 binary (Cargo.toml 里 [[bin]], 没有 [lib]), 所以这里要写真实的
//! integration 必须先把 src/main.rs 拆成 src/lib.rs (导出 run / Agent / Repl 等)
//! + src/main.rs 调用 lib::run. 这是后续 (范围外) 可选练习.

#[test]
fn placeholder_compiles() {
    // 占位: 让 cargo test --test integration 至少跑得起来.
    assert!(true);
}
