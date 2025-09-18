# Quickstart

Get Pitaya running in a few minutes: install, authenticate, run a strategy, and see results.

## Prerequisites

- Docker Desktop or Docker Engine running
- Python 3.13
- A git repository (run Pitaya inside your repo)

## Install

```bash
# Recommended
pip install pitaya
# or
pipx install pitaya
# or (uv as a tool)
uv tool install pitaya
```

## Build the Docker image

Pitaya executes agents inside Docker. You need an image that contains the agent CLI(s) you will run:

- `claude` (for `--plugin claude-code`)
- `codex` (for `--plugin codex`)

Use this repository’s [Dockerfile](../Dockerfile) to build a ready image:

```bash
git clone https://github.com/tact-lang/pitaya
cd pitaya
docker build -t pitaya-agents:latest .
```

You can also bring your own base image; just ensure it includes the required CLI(s). Point Pitaya at a custom image with `--docker-image` or set `runner.docker_image` in `pitaya.yaml`.

Dev install from source:

```bash
# Clone the repo
git clone https://github.com/tact-lang/pitaya
cd pitaya

# Editable install (virtualenv)
python -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -e .
```

Run from the repo without installing (uv or Python):

```bash
git clone https://github.com/tact-lang/pitaya
cd pitaya

# Using uv
uv run pitaya --version

# Or directly via Python module
python -m src.cli --version
```

## Authenticate

Choose a plugin and set credentials. By default Pitaya uses the Claude Code plugin.

- Claude Code (subscription or API):

  ```bash
  # One of these is required
  export CLAUDE_CODE_OAUTH_TOKEN=…   # subscription
  # or
  export ANTHROPIC_API_KEY=…         # API
  # optional
  export ANTHROPIC_BASE_URL=…        # proxy/base URL if needed
  ```

- Codex CLI (OpenAI‑compatible):

  ```bash
  export OPENAI_API_KEY=…          # or OPENROUTER_API_KEY, GROQ_API_KEY, etc.
  # optional
  export OPENAI_BASE_URL=…         # provider-base overrides are auto-detected
  ```

You can also pass `--api-key`, `--oauth-token`, and `--base-url` via CLI.

## Your First Run

From the root of a git repo, run a single agent. The TUI launches by default.

```bash
pitaya "Create a HELLO.txt file with 'Hello from Pitaya' text in it and commit it"
```

Using uv from the repo clone:

```bash
uv run pitaya "Create a HELLO.txt file with 'Hello from Pitaya' text in it and commit it"
```

Tips

  - Press Ctrl+C to stop the run. Pitaya shuts down gracefully (containers stopped, state saved) and prints a resume hint with the run ID.
- After completion, results are written under `results/<run_id>/` and logs under `logs/<run_id>/`.
- Branches are created only if the agent commits changes.

## Explore More

Best‑of‑N (parallel candidates with scoring):

```bash
pitaya "Write the funniest and most original joke possible" --strategy best-of-n -S n=5
```

Iterative refine (generate → review → refine):

```bash
pitaya "Write the funniest and most original joke possible" --strategy iterative -S iterations=3
```

## Resume

- Resume an interrupted run:

  ```bash
  pitaya --resume run_20250114_123456
  ```

## Where to Find Results

- Logs: `logs/<run_id>/events.jsonl` and JSONL component logs (orchestration.jsonl, runner.jsonl, tui.jsonl, other.jsonl)
- Results: `results/<run_id>/` (summary.json, metadata.json, branches.txt, metrics, instances/<id>.json)
- Branches: `pitaya/<strategy>/<run_id>/k<short8>` (hierarchical namespace)

List branches:

```bash
git branch -a | rg '^\s*pitaya/'
```
