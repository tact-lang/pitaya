# Plugins

Pitaya supports multiple agent tools via runner plugins. The two built‑ins are:

- Claude Code (`--plugin claude-code`)
- Codex CLI (`--plugin codex`) — works with OpenAI‑compatible providers like OpenRouter

## Models

Pitaya passes model identifiers through to the plugin; there is no models.yaml mapping. Use the model names your provider expects.

Examples

```bash
# Claude Code
pitaya "task" --plugin claude-code --model sonnet

# Codex CLI with an OpenAI‑compatible model ID
pitaya "task" --plugin codex --model "openai/gpt-5"
```

## Docker Images

Plugins choose a default Docker image. Override globally per run with `--docker-image`.

```bash
pitaya "task" --docker-image ghcr.io/you/agents:latest
```

## Authentication

You can pass credentials via CLI or env vars.

Claude Code

```bash
# Subscription (OAuth)
export CLAUDE_CODE_OAUTH_TOKEN=...
# API key
export ANTHROPIC_API_KEY=...
# Optional base URL (proxies)
export ANTHROPIC_BASE_URL=...
```

Codex CLI / OpenAI‑compatible

```bash
export OPENAI_API_KEY=...
# Optional custom endpoint
export OPENAI_BASE_URL=...
```

OpenRouter example

```bash
pitaya "Write the funniest and most original joke possible" \
  --plugin codex \
  --model "openai/gpt-5" \
  --api-key "$OPENROUTER_API_KEY" \
  --base-url https://openrouter.ai/api/v1
```

## CLI Passthrough

Forward raw flags to the underlying agent CLI for advanced use.

```bash
# Repeatable single tokens
pitaya "task" --plugin codex \
  --cli-arg -c \
  --cli-arg 'feature_flag=true' \
  --cli-arg --no-color

# One quoted list (preserves order)
pitaya "task" --plugin codex \
  --cli-args '-c model="gpt-4o-mini" --dry-run -v'
```

## Network & Limits

- Network egress: default `online`; set per task via `-S network_egress=offline` or in config.
- Limits come from `runner` config (`cpu_limit`, `memory_limit`, `timeout`).

## Security

- Strategies and plugins run inside containers with repo mounts. Pitaya imports commits into branches and avoids destructive operations by default.
- Loading strategies from files/modules executes code. Only load trusted strategies.
