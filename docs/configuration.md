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

logging:
  # JSONL component logs under logs/<run_id>/
  max_file_size: 10485760   # 10MB per component file before rotation
  retention_days: 7         # cleanup old run dirs (component logs)
  redaction:
    # Additional regexes applied to event payloads/logs (case-insensitive where needed)
    custom_patterns:
      - "(?i)authorization: Bearer [A-Za-z0-9._-]+"
      - "sk-[A-Za-z0-9]{20,}"
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
  - Auto-detected API keys: `OPENAI_API_KEY`, `OPENROUTER_API_KEY`, `GROQ_API_KEY`, `AZURE_OPENAI_API_KEY`, etc.
  - Optional base URLs: `OPENAI_BASE_URL`, `OPENROUTER_BASE_URL`, …
  - Advanced overrides: `CODEX_ENV_KEY`, `CODEX_BASE_URL`, `CODEX_MODEL_PROVIDER`

`.env` files are loaded as a separate layer (useful in development). Loading is best-effort: if `python-dotenv` is not installed, Pitaya skips this layer without raising. Secrets are redacted by default in `pitaya config print` unless `PITAYA_ALLOW_UNREDACTED=1` and `--redact false` are used.

## Import policy

Runner import behavior (how commits are imported into your repo):

- `import_policy`: `auto` (default), `never`, `always`
- `import_conflict_policy`: `fail` (default), `overwrite`, `suffix`
- `skip_empty_import`: true/false (default true)

Most strategies leave these at defaults. To force an import even when there are no changes, set `import_policy: always`. To avoid any branch import for a task, set `import_policy: never`.

## Directories

- State: `pitaya_state/` (can be changed with `--state-dir`)
- Logs: `logs/<run_id>/` (change with `--logs-dir`)
- Results: `results/<run_id>/` (summary.json, metadata.json, branches.txt, instance_metrics.csv, instances/<id>.json)

Structured logs

- Pitaya writes JSON Lines component logs in `logs/<run_id>/` (orchestration.jsonl, runner.jsonl, tui.jsonl, other.jsonl).
- Old log directories are cleaned up periodically; component files rotate by size.
- Custom redaction patterns from `logging.redaction.custom_patterns` are applied to both logs and emitted events.

## Effective config persistence and resume

On a fresh run, Pitaya writes the fully merged configuration (CLI + env + .env + file + defaults):

- Unredacted copy to `pitaya_state/<run_id>/config.json` (for fidelity on resume). The `pitaya_state/` directory is git‑ignored by default.
- Redacted copy to `logs/<run_id>/config.json` (tokens/API keys masked) for auditability alongside logs.

On `--resume <run_id>`, Pitaya loads the saved effective config by default. This preserves durable keys and behavior. CLI overrides on resume are applied as follows:

- Safe overrides (applied by default): timeouts, Docker image, force_commit, parallelism settings, queue randomization, and auth secrets/base URL.
- Unsafe overrides (require `--override-config`): `model`, `plugin_name`, and `runner.network_egress`.

When unsafe overrides are applied, durable keys might change. To avoid collisions with previously scheduled tasks, you can choose a policy with `--resume-key-policy`:

- `strict` (default): keep keys unchanged; if an override would change them, it is ignored.
- `suffix`: append a short resume suffix to newly generated keys (e.g., `/r1a2b`).

## Tips

- Prefer configuration for persistent team defaults; use CLI for ad‑hoc overrides.
- For complex strategy parameters (lists, nested structures), use `pitaya.yaml` under the `strategies:` section instead of many `-S` flags.
- Missing credentials fail fast with a clear error. For `--mode api` set an API key; for subscription mode set an OAuth token. See Plugins for details.
