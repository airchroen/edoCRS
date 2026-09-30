# edoCRS

A demo Claude Code-style agent CLI in Rust. Chat with streaming output, three tools (`read_file` / `write_file` / `bash`), and session persistence. Talks to any OpenAI Chat Completions compatible provider (OpenAI, OpenRouter, DeepSeek, Ollama built in).

## Quick Start

```bash
cp .env.example .env   # fill in the API key of the provider you use
$EDITOR .env

cargo build --release  # binary only lands in release
./target/release/edocrs --model deepseek/deepseek-v4-flash

# continue the latest session / resume a specific one
./target/release/edocrs -c
./target/release/edocrs --resume <session-id>
```

## Configuration

Settings live in `~/.edocrs/agent/settings.json` (JSONC: comments and trailing commas allowed); see `settings.example.jsonc` for every field. The directory can be moved with `EDOCRS_AGENT_DIR`.

Precedence: CLI flag > `EDOCRS_MODEL` env > project settings > global settings > built-in defaults. (Project settings are not read yet; they arrive together with project trust.)

Models are written `provider/model-id` (split on the first `/`, so `openrouter/anthropic/claude-sonnet-4` works). There is no built-in default model and no model list: pass `--model`, set `EDOCRS_MODEL`, or set `default_model` in settings. API keys come from each provider's environment variable (`OPENAI_API_KEY`, `OPENROUTER_API_KEY`, `DEEPSEEK_API_KEY`) and are only read when that provider is actually used.

Permission modes (`--mode` / `permission_mode`): `ask` (default) asks before write/edit/bash; `auto-edit` auto-approves file writes. `bash` always asks.

## Usage

Slash commands: `/help` `/exit` `/quit` `/clear` `/save <name>` `/sessions` `/tools`

## Tests

```bash
cargo test
```

All network calls are mocked with wiremock — no real provider access or API key required.

## License

This project is licensed under the [GPL-3.0](LICENSE) license.

---

MCP, hooks, full-screen TUI, parallel tool calls, and other providers are intentionally out of scope.
