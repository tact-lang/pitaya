# Pitaya Documentation

Pitaya orchestrates AI coding agents (e.g., Claude Code and Codex CLI) with production‑ready strategies and a clean TUI. It runs agents in parallel, isolates them in containers, and imports results into per‑task git branches, so you can compare approaches safely.

- Parallel strategies: simple, best‑of‑n, iterative, bug‑finding, doc‑review
- Clean TUI: detailed/compact/dense modes, costs/tokens, progress
- Custom strategies: load via file or module spec (path.py[:Class], package.module[:Class])
- Safe by default: isolated containers, artifact‑first, branch‑per‑task

Start with Quickstart, then dive into CLI, TUI, and Strategies.

## Start Here

- Quickstart: getting set up fast → [quickstart.md](quickstart.md)

## Usage

- CLI reference and examples → [cli.md](cli.md)
- TUI modes (detailed, compact, dense), offline viewer → [tui.md](tui.md)

## Strategies

- Built‑in strategies and parameters → [strategies.md](strategies.md)
- Write your own (Strategy subclass, module/file loading) → [custom-strategies.md](custom-strategies.md)

## Configuration & Plugins

- pitaya.yaml, env/precedence, orchestration/runner options → [configuration.md](configuration.md)
- Plugins (Claude Code, Codex CLI), docker image overrides, CLI passthrough → [plugins.md](plugins.md)

 
