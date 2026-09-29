//! CLI 参数解析 (clap derive).
//!
//! 学习点: clap 的 derive 模式让我们用普通 struct + 属性宏定义参数,
//!         远比手写 match 更易读. `#[arg(long)]` 自动生成 --xxx 形式.

use clap::Parser;

#[derive(Parser, Debug)]
#[command(name = "edocrs", version, about = "简易 Claude Code 风格 CLI (DeepSeek V4)")]
pub struct Cli {
    /// DeepSeek API key. 也可走 DEEPSEEK_API_KEY env 或 .env.
    #[arg(long)]
    pub api_key: Option<String>,

    /// 自定义 base url (运维/测试用; 默认 https://api.deepseek.com).
    #[arg(long)]
    pub base_url: Option<String>,

    /// 模型: deepseek-v4-flash 或 deepseek-v4-pro.
    #[arg(long)]
    pub model: Option<String>,

    /// 恢复会话: 指定 session id 或 'last'.
    #[arg(long)]
    pub resume: Option<String>,

    /// 危险: 跳过所有权限询问. 仅在你 100% 信任脚本时使用.
    #[arg(long, default_value_t = false)]
    pub yolo: bool,
}
