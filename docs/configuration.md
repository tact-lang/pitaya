# Configuration

You can run Pitaya entirely from the CLI, or add an optional `pitaya.yaml` to set defaults. CLI overrides config; `-S` only affects the selected strategy.

## Precedence

- Highest → Lowest: CLI > environment variables > .env > project config (pitaya.yaml) > built‑in defaults.

## pitaya.yaml

Example:

```yaml
model: sonnet
plugin_name: claude-code

runner:
  timeout: 3600            # seconds per instance
  cpu_limit: 2             # cores
  memory_limit: 4g         # gigabytes (number or "Ng")
  network_egress: online   # online|offline|proxy
  docker_image: ghcr.io/you/agents:latest   # optional override
  force_commit: false

orchestration:
  max_parallel_instances: auto   # or an integer
  max_parallel_startup: auto     # auto -> min(10, max_parallel_instances)
  branch_namespace: hierarchical # pitaya/<strategy>/<run_id>/k<short8>
  snapshot_interval: 30          # seconds
  event_buffer_size: 10000
  randomize_queue_order: false

strategies:
  best-of-n:
    n: 5
    scorer_model: opus
  iterative:
    iterations: 3
```

## CLI overrides

- `--model` and `--plugin` override `model`/`plugin_name`.
- `-S key=value` only applies to the selected `--strategy`.
- `--docker-image`, `--max-parallel`, etc., override corresponding config entries.

## Environment variables

- Claude Code:
  - `CLAUDE_CODE_OAUTH_TOKEN`
  - `ANTHROPIC_API_KEY`
  - `ANTHROPIC_BASE_URL`
- Codex CLI / OpenAI‑compatible:
  - `OPENAI_API_KEY`
  - `OPENAI_BASE_URL`

`.env` files are loaded as a separate layer (useful in development). Secrets are redacted by default in `pitaya config print` unless `PITAYA_ALLOW_UNREDACTED=1` and `--redact false` are used.

## Import policy

Runner import behavior (how commits are imported into your repo):

- `import_policy`: `auto` (default), `never`, `always`
- `import_conflict_policy`: `fail` (default), `overwrite`, `suffix`
- `skip_empty_import`: true/false (default true)

Most strategies leave these at defaults. To force an import even when there are no changes, set `import_policy: always`. To avoid any branch import for a task, set `import_policy: never`.

## Directories

- State: `pitaya_state/` (can be changed with `--state-dir`)
- Logs: `logs/<run_id>/` (change with `--logs-dir`)
- Results: `results/<run_id>/` (summary.json, branches.txt, instance_metrics.csv)

## Tips

- Prefer configuration for persistent team defaults; use CLI for ad‑hoc overrides.
- For complex strategy parameters (lists, nested structures), use `pitaya.yaml` under the `strategies:` section instead of many `-S` flags.
