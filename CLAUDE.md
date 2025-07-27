# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Orchestrator is a Python tool for parallel execution of AI coding agents (primarily Claude Code) to explore multiple solution paths simultaneously. It runs instances in isolated Docker containers with separate git branches.

## Development Commands

```bash
# Install dependencies (using uv package manager)
uv sync

# Run the orchestrator
uv run orchestrator [strategy] [prompt]

# Run type checking
uv run mypy src

# Run linter
uv run ruff check src

# Format code
uv run black src

# Run tests
uv run pytest

# Run a specific test
uv run pytest tests/test_file.py::test_name
```

## Architecture

The codebase follows a strict three-layer architecture with unidirectional event flow:

1. **Instance Runner** (`src/instance_runner/`): Executes AI instances in Docker containers
   - `runner.py`: Core instance execution logic
   - `docker_manager.py`: Container lifecycle management
   - `git_operations.py`: Git branch isolation
   - `plugins/`: AI tool plugins (Claude Code implementation)

2. **Orchestration** (`src/orchestration/`): Coordinates multiple instances via strategies
   - `orchestrator.py`: Main coordination logic
   - `strategies/`: Strategy pattern implementations (Simple, BestOfN, Scoring, Iterative)
   - `event_bus.py`: Event communication system

3. **TUI** (`src/tui/`): Terminal UI for real-time monitoring
   - Components communicate via file-based events (`events.jsonl`)
   - No direct dependencies between layers

## Key Patterns

- **Event-Driven**: All communication flows through events (Instance → Orchestration → TUI)
- **Strategy Pattern**: Extensible strategy system in `src/orchestration/strategies/`
- **Plugin Architecture**: AI tools are plugins in `src/instance_runner/plugins/`
- **Async Throughout**: Uses Python asyncio for concurrent operations

## Configuration

Hierarchical config system (CLI > env > .env > config.yaml > defaults):
- `src/config.py`: Central configuration management
- Supports OAuth tokens and API keys
- Platform-aware (Linux, macOS, Windows/WSL)

## Important Files

- `docs/SPECIFICATION.md`: Comprehensive 1800+ line specification
- `pyproject.toml`: Project configuration and dependencies
- `Dockerfile`: Container definition for Claude Code instances

## Python Version

Always use `python3` (specifically Python 3.13+), not `python`.

## Testing Approach

The project uses pytest for testing. Tests should follow the existing patterns and maintain isolation between layers.