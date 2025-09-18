# 🎯 Pitaya

<div align="center">

Orchestrate AI coding agents (Claude Code, Codex CLI) with pluggable strategies and a clean TUI.

[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-required-blue.svg)](https://www.docker.com/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

</div>

Note: Pitaya is in public beta. Interfaces and defaults may evolve.

## What It Does

- Runs multiple agents in parallel and helps you pick the best result
- Uses per‑task branches in your repo to keep alternatives safe and reviewable
- Displays a clean, adaptive TUI with live progress, costs, and tokens
- Supports custom Python strategies for multi‑stage workflows

Built‑in strategies: simple, scoring, best‑of‑n, iterative, bug‑finding, doc‑review

## Demo

Watch a short video demo of the TUI and orchestration:

https://github.com/user-attachments/assets/3c426890-4c5a-4765-b594-adfd09d3810b

## Install

```bash
pip install pitaya
# or
pipx install pitaya
# or
uv tool install pitaya
```

## Build the Docker image

Pitaya runs agents inside Docker. You need a Docker image that contains the agent CLI(s) you plan to use:

- `claude` (for `--plugin claude-code`)
- `codex` (for `--plugin codex`)

You can build a ready‑to‑use image from this repository’s [Dockerfile](./Dockerfile):

```bash
git clone https://github.com/tact-lang/pitaya
cd pitaya
docker build -t pitaya-agents:latest .
```

Prefer a custom image? Use any base you like as long as it includes the required CLI(s) above. You can point Pitaya to a different image via `--docker-image` or set `runner.docker_image` in `pitaya.yaml`.

Authenticate:

- Claude Code: set `CLAUDE_CODE_OAUTH_TOKEN` (subscription) or `ANTHROPIC_API_KEY`
- Codex CLI: export the API key for your provider (`OPENAI_API_KEY`, `OPENROUTER_API_KEY`, `GROQ_API_KEY`, etc.). Optional base URLs (such as `OPENROUTER_BASE_URL`) are detected automatically.

## Quickstart

From the root of a git repo:

```bash
pitaya "Create a HELLO.txt file with 'Hello from Pitaya' text in it and commit it"
```

More examples:

```bash
# Parallel candidates with scoring (pick the best)
pitaya "Write the funniest and most original joke possible" --strategy best-of-n -S n=5

# Iterative refine (generate → review → refine)
pitaya "Write the funniest and most original joke possible" --strategy iterative -S iterations=3

# Headless JSON output (pure NDJSON)
pitaya "task" --json
```

OpenRouter (Codex plugin) example:

```bash
pitaya "Write the funniest and most original joke possible" \
  --plugin codex \
  --model "openai/gpt-5" \
  --api-key "$OPENROUTER_API_KEY" \
  --base-url https://openrouter.ai/api/v1
```

## Documentation

- Start here: [docs/index.md](docs/index.md)
- Quickstart: [docs/quickstart.md](docs/quickstart.md)
- CLI: [docs/cli.md](docs/cli.md)
- TUI: [docs/tui.md](docs/tui.md)
- Strategies: [docs/strategies.md](docs/strategies.md)
- Custom Strategies: [docs/custom-strategies.md](docs/custom-strategies.md)
- Configuration: [docs/configuration.md](docs/configuration.md)
- Plugins: [docs/plugins.md](docs/plugins.md)

## Configuration (peek)

Optional `pitaya.yaml` to set defaults:

```yaml
model: sonnet
plugin_name: claude-code
orchestration:
  max_parallel_instances: auto
  branch_namespace: hierarchical
```

CLI overrides config; `-S key=value` only affects the selected strategy.

## Results & Logs

- Logs: `logs/<run_id>/events.jsonl` and JSONL component logs (orchestration.jsonl, runner.jsonl, tui.jsonl, other.jsonl)
- Results: `results/<run_id>/` (summary.json, metadata.json, branches.txt, instance_metrics.csv, instances/<id>.json)
- Branches: `pitaya/<strategy>/<run_id>/k<short8>` (hierarchical namespace)
- Resume: `pitaya --resume <run_id>`

## Contributing

Issues and PRs are welcome. This project is evolving—feedback on UX, strategies, and plugin support is especially helpful.

Dev quickstart:

```bash
git clone https://github.com/tact-lang/pitaya
cd pitaya
python -m venv .venv && source .venv/bin/activate
pip install -U pip && pip install -e .

# Optional dev tools
pip install -U ruff black mypy pytest pytest-asyncio
```

## Changelog

- See [CHANGELOG.md](CHANGELOG.md) for release notes and version history
- GitHub Releases: https://github.com/tact-lang/pitaya/releases

## License

MIT License — see [LICENSE](LICENSE).
