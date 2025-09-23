# CLI Reference

Use Pitaya from the terminal to run strategies, control output, and manage runs.

## Basic Usage

```bash
pitaya "your prompt here"
```

- Quotes recommended for multi‑word prompts.
- Default strategy is `simple` (one agent, one branch).

## Strategies

- Select a strategy:

  ```bash
  pitaya "task" --strategy best-of-n
  ```

- Pass strategy params (repeat `-S KEY=VALUE`):

  ```bash
  pitaya "task" --strategy best-of-n -S n=5 -S scorer_model=opus
  ```

  Notes for list values:

  - `-S include_branches=feature/a,release/1.2.0` (CSV)
  - `-S include_branches='["feature/a","release/1.2.0"]'` (JSON list)

- Multiple parallel executions of the selected strategy:

  ```bash
  pitaya "task" --strategy best-of-n --runs 3
  ```

- Custom strategy spec (file/module):

  ```bash
  # File path (discover Strategy subclass or STRATEGY)
  pitaya "task" --strategy ./examples/propose_refine.py

  # File with explicit class
  pitaya "task" --strategy ./examples/propose_refine.py:ProposeRefineStrategy

  # Module with class
  pitaya "task" --strategy examples.fanout_two:FanOutTwoStrategy

  # Module (single Strategy exported)
  pitaya "task" --strategy examples.custom_simple
  ```

Built‑ins: simple, scoring, best-of-n, iterative, bug-finding, doc-review

## Model & Plugin

- Choose model and plugin:

  ```bash
  pitaya "task" --model sonnet --plugin claude-code
  pitaya "task" --model gpt-5 --plugin codex
  ```

- Override Docker image:

  ```bash
  pitaya "task" --docker-image ghcr.io/you/agents:latest
  ```

- Agent CLI passthrough (advanced):

  ```bash
  # Repeatable single tokens
  pitaya "task" --plugin codex \
    --cli-arg -c \
    --cli-arg 'feature_flag=true' \
    --cli-arg --no-color

  # One quoted list (preserves order)
  pitaya "task" --plugin codex \
    --cli-args '-c model="gpt-5" --dry-run -v'
  ```

- OpenRouter (Codex plugin) example:

  ```bash
  export OPENROUTER_API_KEY=...
  export OPENROUTER_BASE_URL=https://openrouter.ai/api/v1

  pitaya "Write the funniest and most original joke possible" \
    --plugin codex \
    --model "openai/gpt-5"
  ```

## Repository

- Work in a different repo or base branch:

  ```bash
  pitaya "task" --repo /path/to/repo --base-branch main
  ```

- Enforce clean working tree (fail if dirty):

  ```bash
  pitaya "task" --require-clean-wt
  ```

- Allow overwriting protected refs (dangerous):

  ```bash
  pitaya "task" --allow-overwrite-protected-refs
  ```

## Display & Output

- TUI is default in interactive terminals. In CI/non‑TTY, Pitaya defaults to headless.

- Disable TUI and pick an output:

  ```bash
  pitaya "task" --no-tui --output streaming   # human text
  pitaya "task" --no-tui --output json        # structured NDJSON (one JSON per line)
  pitaya "task" --no-tui --output quiet       # minimal
  ```

- Convenience alias:

  ```bash
  pitaya "task" --json   # implies --no-tui --output json
  # Note: JSON mode writes only NDJSON to stdout (no human summary)
  ```

- TUI density:

  ```bash
  pitaya "task" --display auto|detailed|compact|dense
  ```

- Streaming output tweaks:

  ```bash
  pitaya "task" --no-emoji                 # strip emoji
  pitaya "task" --show-ids full            # show full IDs in logs
  pitaya "task" --verbose                  # more container/import detail
  ```

## Execution & Limits

```bash
pitaya "task" --max-parallel 6               # concurrent instances
pitaya "task" --max-startup-parallel 4       # concurrent startup preps
pitaya "task" --timeout 3600                 # per-instance seconds
pitaya "task" --randomize-queue              # dequeue queued tasks randomly
pitaya "task" --force-commit                  # force a commit if changes exist
```

## Config, State & Logs

- Config file (optional): `pitaya.yaml` or pass a path.

  ```bash
  pitaya "task" --config ./pitaya.yaml
  ```
  If you pass `--config` with a missing/unreadable file, Pitaya errors out with a clear message (no silent fallback).

- Precedence: CLI > env > .env > project config > defaults.
- Directories:
  - State: `--state-dir ./pitaya_state` (default)
  - Logs: `--logs-dir ./logs` (default)

## Maintenance

```bash
pitaya --list-runs                      # list previous runs
pitaya --show-run run_20250114_123456   # show run details
pitaya --resume run_20250114_123456     # resume an interrupted run
```

### Resume overrides (advanced)

By default, resume uses the saved effective configuration from the original run to preserve durable keys and behavior. You can override some values safely on resume (e.g., timeouts, parallelism, Docker image). Unsafe overrides (e.g., model, plugin, network egress) are ignored unless you opt in.

```bash
# Allow unsafe overrides on resume
pitaya --resume <run_id> \
  --override-config \
  --model sonnet-extended \
  --plugin claude-code \
  --max-parallel 2

# If you change model/plugin and want to avoid disturbing durable task keys,
# you can append a suffix to newly scheduled keys on resume:
pitaya --resume <run_id> \
  --override-config \
  --resume-key-policy suffix
```

Policies:
- `strict` (default): keep durable keys identical; unsafe overrides are applied only if they won’t change keys. If they would, they are ignored and a warning is printed.
- `suffix`: when unsafe overrides are applied, new tasks get a `/r<xxxx>` suffix to their durable keys to avoid collisions with prior work.

## Diagnostics & Utilities

- Doctor (Docker, repo, disk, auth checks):

  ```bash
  pitaya doctor
  ```

- Print effective config (with source and redaction):

  ```bash
  pitaya config print
  pitaya config print --json
  pitaya config print --redact false   # allow unredacted when PITAYA_ALLOW_UNREDACTED=1
  ```

  Redaction patterns from config (`logging.redaction.custom_patterns`) are applied to events and logs.

- Agree to confirmations automatically:

  ```bash
  pitaya "task" --yes
  ```

## Exit Codes

- 0: Success
- 1: Error (validation, environment, or runtime failure)
- 2: Interrupted by user (Ctrl+C)
- 3: Completed with failures (some instances failed)

Ctrl+C behavior: one press initiates graceful shutdown (containers stopped, state snapshotted, resume hint printed). A second press forces exit.

## Examples

- One‑liner (default strategy):

  ```bash
  pitaya "Create a HELLO.txt file with 'Hello from Pitaya' text in it and commit it"
  ```

- Best‑of‑N:

  ```bash
  pitaya "Write the funniest and most original joke possible" --strategy best-of-n -S n=5
  ```

- Iterative:

  ```bash
  pitaya "Write the funniest and most original joke possible" --strategy iterative -S iterations=3
  ```

- JSON output (headless):

  ```bash
  pitaya "task" --json
  ```

- Custom strategy from module:

  ```bash
  pitaya "task" --strategy examples.fanout_two:FanOutTwoStrategy
  ```
