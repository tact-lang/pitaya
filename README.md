# 🎯 Pitaya

<div align="center">

**Orchestrate AI coding agents (Claude Code, Codex CLI, and more)**

[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-required-blue.svg)](https://www.docker.com/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

</div>

> Note: Pitaya is in public beta. Interfaces and defaults may change between releases.

## Overview

Pitaya is an orchestrator for AI coding agents such as Claude Code and Codex CLI. It runs multiple agents in parallel, compares their results, and helps you pick the best outcome. Each agent works in an isolated Docker container with its own git branch, so you can explore alternative solution paths safely and quickly. You can also define arbitrary custom strategies to build your own multi-stage workflows.

- Parallel strategies (simple, best-of-n, iterative, bug-finding, doc-review)
- Clean TUI with live progress, costs, and artifacts
- Orchestrates Claude Code, Codex CLI, and others via plugins
- Define arbitrary custom strategies (Python) for complex flows
- Strict “agent commits only” mode; artifact-first, no destructive merges
- Resumable runs with detailed logs and events

## Quick Start

Prerequisites

- Docker Desktop or Docker Engine running
- Python 3.13
- Git repository (the tool operates inside your repo)

Install (choose one)

- From PyPI (recommended):

  ```bash
  pip install pitaya
  # or
  pipx install pitaya
  # or (uv as a tool)
  uv tool install pitaya
  ```

  Upgrade:

  ```bash
  pip install -U pitaya
  # or
  pipx upgrade pitaya
  # or
  uv tool upgrade pitaya
  ```

- From a local clone (editable dev install):

  ```bash
  python -m venv .venv && source .venv/bin/activate
  pip install -U pip
  pip install -e .
  ```

Authenticate

- Anthropic (Claude Code): set either `CLAUDE_CODE_OAUTH_TOKEN` or `ANTHROPIC_API_KEY`
- OpenAI (Codex plugin): set `OPENAI_API_KEY` (and optionally `OPENAI_BASE_URL`)

Hello world

```bash
pitaya "implement authentication" --strategy simple
```

Best-of-N with scoring

```bash
pitaya "fix bug in user.py" --strategy best-of-n -S n=5
```

Doc review (N reviewers per page, immediate validators, composer)

```bash
pitaya "Review docs" --strategy doc-review -S pages_file=pages.yml -S reviewers_per_page=2
```

Headless JSON output

```bash
pitaya "add tests" --no-tui --output json
```

Override Docker image

```bash
pitaya "task" --plugin codex --docker-image ghcr.io/me/codex-cli:mytag
```

Agent CLI passthrough

Forward raw arguments directly to the underlying agent CLI (Codex or Claude). This is useful for advanced flags or experimental options exposed by the tools.

```bash
# Single tokens (repeatable)
pitaya "fix bug" --plugin codex \
  --cli-arg -c \
  --cli-arg 'feature_flag=true' \
  --cli-arg --no-color

# Quoted list (recommended for ordered sequences)
pitaya "refactor module" --plugin codex \
  --cli-args '-c model="gpt-4o-mini" --dry-run -v'

# Mix-and-match
pitaya "task" --plugin claude-code \
  --cli-arg --verbose \
  --cli-args '--max-turns 8'
```

Notes

- Passthrough args are inserted before the final prompt position.
- `--cli-args` uses POSIX shlex splitting; quote values that contain spaces.
- If both forms are provided, Pitaya combines all `--cli-arg` tokens followed by tokens from `--cli-args`. For strict ordering, prefer a single `--cli-args` string.

Resume a run

```bash
pitaya --resume run_20250114_123456
```

## CLI Essentials

The CLI is designed to be discoverable and production-ready. Run `pitaya --help` to see grouped options and examples.

Highlights

- Strategy: `--strategy <name>` (use `-S key=value` for strategy params)
- Model: `--model <name>`
- Plugin: `--plugin <claude-code|codex>`
- Parallel runs: `--runs <N>`
- Scheduling: `--randomize-queue` (dequeue tasks in random order)
- TUI controls: `--no-tui`, `--output <streaming|json|quiet>`
- Maintenance: `--list-runs`, `--show-run <id>`, `--prune`, `--clean-containers <id>`
- Docker image override: `--docker-image <repo/name:tag>`

TUI viewer (offline or live)

```bash
pitaya-tui --run-id run_20250114_123456
# or
pitaya-tui --events-file logs/run_20250114_123456/events.jsonl --output streaming
```

## Strategies

- simple: one agent, one branch
- best-of-n: spawn N agents, score and pick the highest-rated branch
- iterative: loop with propose → review → refine (configurable iterations)
- bug-finding: search for issues across the repo and propose fixes
- doc-review: reviewers per page → validators per reviewer → compose final report
  - Pass pages via `-S pages_file=...`; set `-S reviewers_per_page=<n>` (default 1)
  - Reviewer reports: `reports/doc-review/raw/REPORT_{slug}__r{n}.md`
  - Final report: `reports/doc-review/REPORT.md`

Pass strategy options with `-S key=value`. Example: `-S n=5 -S scorer_prompt="evaluate correctness"`.

## Configuration

You can run everything from the CLI, or add an optional `pitaya.yaml` to set defaults:

```yaml
model: sonnet
plugin_name: claude-code

runner:
  timeout: 3600
  cpu_limit: 2
  memory_limit: 4g
  network_egress: online  # online|offline|proxy
  docker_image: ghcr.io/me/codex-cli:mytag  # optional global override

orchestration:
  max_parallel_instances: auto
  branch_namespace: hierarchical
  snapshot_interval: 30
  randomize_queue_order: false  # set true to dequeue in random order

strategies:
  best-of-n:
    n: 5
  doc-review:
    reviewers_per_page: 2
```

CLI overrides config; `-S` only affects the selected strategy.

## Models

Plugins accept model identifiers as provided. Claude Code commonly uses `sonnet`, `haiku`, or `opus`; OpenAI‑compatible providers accept their own model IDs. No `models.yaml` mapping is used.

## Docker & Plugins

- Unified agent image: `pitaya-agents:latest` (includes Claude Code and Codex CLIs)
- Plugins default to `pitaya-agents:latest`; override per run with `--docker-image <repo/name:tag>`
- Full isolation per instance: dedicated container, workspace mount, and session volume

### OpenAI‑Compatible Providers (Codex)

- Pass `--api-key` and `--base-url` to target any OpenAI‑compatible endpoint. Pitaya configures the Codex CLI provider under the hood.
- Example (OpenRouter):

  ```bash
  pitaya "hello" \
    --plugin codex \
    --model "openai/gpt-5" \
    --api-key "$OPENROUTER_API_KEY" \
    --base-url https://openrouter.ai/api/v1
  ```

- This works with any OpenAI‑compatible API (self‑hosted or proxy). The API key is provided to the container via `OPENAI_API_KEY`.

### Anthropic (Claude Code)

- Use `--oauth-token` (subscription) or `--api-key` (API mode). Optional `--base-url` is respected as `ANTHROPIC_BASE_URL`.
- Env alternatives: `CLAUDE_CODE_OAUTH_TOKEN`, `ANTHROPIC_API_KEY`, and `ANTHROPIC_BASE_URL`.

## Custom Strategies

- Use built-ins (e.g., `simple`, `best-of-n`, `iterative`) or point to custom ones via:
  - File path: `pitaya "task" --strategy ./examples/custom_simple.py -S model=sonnet`
  - File + class: `pitaya "task" --strategy ./examples/propose_refine.py:ProposeRefineStrategy`
  - Module + class: `pitaya "task" --strategy examples.fanout_two:FanOutTwoStrategy`
  - Module (single strategy exported): `pitaya "task" --strategy examples.custom_simple`
- Define a class subclassing `Strategy` (see `src/orchestration/strategies/base.py`).
  - Minimal boilerplate: set `NAME = "your-name"`, implement `execute(self, prompt, base_branch, ctx)`.
  - Use `self.logger` for logging, no logging imports needed.

Examples

- `examples/custom_simple.py` — one durable task with a friendly greeting.
- `examples/fanout_two.py` — runs N tasks in parallel (default 2).
- `examples/propose_refine.py` — two-stage propose → refine flow (stage 2 bases on stage 1 branch).

Security note: loading strategies from modules/files executes Python code. Only load trusted code.

## Logs & Artifacts

- Logs: `logs/<run_id>/events.jsonl` and structured component logs
- Results: `results/<run_id>/...`
- Branches: `pitaya/<strategy>/<run_id>/k<short8>` (hierarchical namespace)
- Resuming: `--resume <run_id>` picks up from the last consistent snapshot

## Troubleshooting

- Cannot connect to Docker: start Docker Desktop / system service; run `docker info`
- Missing credentials: set `CLAUDE_CODE_OAUTH_TOKEN` or `ANTHROPIC_API_KEY` (Claude), or `OPENAI_API_KEY` (Codex)
- Model not recognized by the agent: pass a valid model ID for your provider
- Slow or flaky network: use `--parallel conservative` or `--max-parallel <n>`
- Clean stale state: `pitaya --prune` and `pitaya --clean-containers <run_id>`

## Contributing

Issues and PRs are welcome. This project is evolving—feedback on UX, strategies, and plugin support is especially helpful.

Local dev quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -e .

# Optional dev tools
pip install -U ruff black mypy pytest pytest-asyncio

# Lint/format (optional)
ruff check .
black .
```

## Changelog

- See [CHANGELOG.md](CHANGELOG.md) for release notes and version history.
- GitHub Releases: https://github.com/tact-lang/agent-orchestrator/releases

## License

MIT License — see [LICENSE](LICENSE).
