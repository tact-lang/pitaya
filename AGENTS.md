# Repository Guidelines

## Project Structure & Modules
- `src/` — main source code
  - `orchestration/` (strategies, event bus, state, CLI control)
  - `instance_runner/` (Docker, git isolation, plugins)
  - `tui/` (Rich-based dashboard)
  - `utils/`, `shared/`, `exceptions.py`
- `docs/` — specs (see `SPECIFICATION.md`)
- `logs/`, `results/`, `pitaya_state/` — runtime artifacts (git-ignored)

## Build, Test, and Development
- Python: 3.13 is required.
- Install deps (uv recommended):
  - `uv sync` (installs project and dev deps)
- Run CLI:
  - `uv run pitaya "fix bug" --strategy best-of-n -S n=3`
  - `uv run pitaya-tui --run-id <run_id>`
  - Without uv: `python -m src.cli "prompt"` or `python -m src.tui.cli --run-id <run_id>`
- Build Docker image used by the runner: `docker build -t claude-code .`

## Coding Style & Naming
- Python (PEP 8, 4‑space indents), type hints required on public APIs.
- Lint/format: use `ruff` and `black` (dev deps). Example:
  - `uv run ruff check src`
  - `uv run black src`
- Deterministic names:
  - Containers: `pitaya_{run_id}_s{strategy_index}_k<hash>`
  - Branches: `pitaya/<strategy>/{run_id}/k<hash>`

## Testing
- Frameworks: `pytest`, `pytest-asyncio` (see `pyproject.toml`).
- Place tests under `tests/`, name files `test_*.py`.
- Run: `uv run pytest -q` (or `pytest -q`).
- Prefer unit tests around orchestration decisions, event emission, and git import logic. Avoid Docker-in-Docker in unit tests.

## Commit & Pull Requests
- Commit messages: imperative mood, short scope prefix when helpful (e.g., `runner:` / `tui:`). Example: `orchestration: emit strategy.rand events on resume`.
- PRs must include:
  - What changed and why (link issues if applicable)
  - How you tested (commands, screenshots of TUI if relevant)
  - Risk/rollback notes

## Security & Configuration Tips
- Do NOT commit secrets. The runner reads `CLAUDE_CODE_OAUTH_TOKEN` or `ANTRHOPIC_API_KEY` (and optional `ANTHROPIC_BASE_URL`).
- Public events are sanitized, but avoid logging raw prompts/headers.
- Offline runs: use `network_egress=offline` (runner sets `network_mode=none`).

## Architecture Overview (quick)
- Three layers: Instance Runner (Docker+git), Orchestration (durable strategies, events), TUI (read-only UI). Events are canonical JSONL with byte offsets for replay.
