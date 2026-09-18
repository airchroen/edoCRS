# edoCRS

A demo Claude Code-style agent CLI in Rust, backed by DeepSeek V4. Chat with streaming output, three tools (`read_file` / `write_file` / `bash`), and session persistence.

## Quick Start

```bash
cp .env.example .env   # fill in DEEPSEEK_API_KEY
$EDITOR .env

cargo build --release  # binary only lands in release
./target/release/edocrs

# or with flags
./target/release/edocrs --model deepseek-v4-pro --resume last
```

## Configuration

Precedence: CLI flag > process env > `.env` > `~/.config/edocrs/config.toml` > built-in defaults.

Valid models: `deepseek-v4-flash` (default, cheap) or `deepseek-v4-pro` (stronger multi-step reasoning). Anything else fails at startup.

## Usage

Slash commands: `/help` `/exit` `/quit` `/clear` `/save <name>` `/sessions` `/tools`

## Tests

```bash
cargo test
```

All network calls are mocked with wiremock — no real DeepSeek access or API key required.

## License

This project is licensed under the [GPL-3.0](LICENSE) license.

---

MCP, hooks, full-screen TUI, parallel tool calls, and other providers are intentionally out of scope.
