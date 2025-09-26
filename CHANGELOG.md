# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.0] - 2025-09-26

### Added

- Persist per-instance metadata to results and logs artifacts. ([#79](https://github.com/tact-lang/pitaya/pull/79))
- CLI flags: `--override-config` and `--resume-key-policy {strict|suffix}` to control how resume applies overrides and durable key behavior. ([#81](https://github.com/tact-lang/pitaya/pull/81))
- Persist the effective run configuration to `pitaya_state/<run_id>/config.json` and a redacted copy to `logs/<run_id>/config.json` to improve resume fidelity. ([#81](https://github.com/tact-lang/pitaya/pull/81))
- Strategy: `pr-review` — N reviewers, validator per reviewer, and a composer; CI‑friendly with JSON trailer parsing and fail gating.
- Workspace: Optional `--include-branches` (CSV/JSON) or config `runner.include_branches` to materialize extra read‑only branches in the isolated workspace for all strategies. Also supported per-task via `workspace_include_branches` metadata. ([#95](https://github.com/tact-lang/pitaya/pull/95))

### Changed

- `.env` loading is now best-effort; if `python-dotenv` is missing Pitaya silently skips that layer. ([#82](https://github.com/tact-lang/pitaya/pull/82))
- `TaskFailed` now groups metadata under `failure` (with `key`, `error_type`, `message`, `result`). Update custom strategies to inspect `exc.failure`. ([#83](https://github.com/tact-lang/pitaya/pull/83))
- Codex plugin now requires an API key, auto-detects provider-specific env vars (OpenAI/OpenRouter/Groq/etc.), and emits matching `model_provider` wiring automatically. ([#89](https://github.com/tact-lang/pitaya/pull/89))
- Preflight: disallow remote‑qualified base names (e.g., `origin/main`) to match runner’s strict workspace rules. ([#95](https://github.com/tact-lang/pitaya/pull/95))

### Fixed

- TUI: coordinated error handling and graceful teardown; surface friendly errors instead of plain crashes. ([#80](https://github.com/tact-lang/pitaya/pull/80))
- Errors: propagate agent `final_message`/metrics on failures and show `error_type` + per‑instance log hint in the summary. ([#80](https://github.com/tact-lang/pitaya/pull/80))
- Token accounting: record streaming usage for Claude/Codex, feed it into the TUI so tokens/costs update live, and prevent completion-time double counting. ([#87](https://github.com/tact-lang/pitaya/pull/87))

## [0.2.0] - 2025-08-29

### Added

- Custom strategies loader (file.py[:Class], module[:Class]) ([#10](https://github.com/tact-lang/pitaya/pull/10)).
- CLI flag: `--randomize-queue` ([#50](https://github.com/tact-lang/pitaya/pull/50)).
- CLI flag: `--max-startup-parallel` ([#47](https://github.com/tact-lang/pitaya/pull/47)).
- CLI flag: `--force-commit` ([#34](https://github.com/tact-lang/pitaya/pull/34)).
- CLI flags: `--cli-arg`, `--cli-args` ([#19](https://github.com/tact-lang/pitaya/pull/19)).
- CLI flag: `--base-url` (OpenAI‑compatible providers) ([#15](https://github.com/tact-lang/pitaya/pull/15)).
- Canonical events enriched (strategy.rand; task.scheduled includes branch/base) ([#10](https://github.com/tact-lang/pitaya/pull/10)).
- Log rotation and old log cleanup ([#9](https://github.com/tact-lang/pitaya/pull/9)).
- Commands: `pitaya doctor`, `pitaya config print` ([#65](https://github.com/tact-lang/pitaya/pull/65)).

### Changed

- TUI redesign (modes, header/footer, throttling, compact branch tail) ([#66](https://github.com/tact-lang/pitaya/pull/66)).
- CLI refinements (help, headless defaults, validation, show-run) ([#65](https://github.com/tact-lang/pitaya/pull/65)).
- Strategies hardened (Scoring, Best‑of‑N, Iterative; doc‑review two‑phase) ([#64](https://github.com/tact-lang/pitaya/pull/64), [#36](https://github.com/tact-lang/pitaya/pull/36)).
- Cleanup lifecycle simplified (immediate container cleanup; prune/cleanup removed) ([#51](https://github.com/tact-lang/pitaya/pull/51)).
- Dockerfiles combined ([#7](https://github.com/tact-lang/pitaya/pull/7), [813220d](https://github.com/tact-lang/pitaya/commit/813220de0a7748e85725fda13219214605613160), [c77cc16](https://github.com/tact-lang/pitaya/commit/c77cc16903e1eb43fb9be47e4ba7587143b974da)).
- Resume behavior adjusted (runner handles continuation prompts) ([#51](https://github.com/tact-lang/pitaya/pull/51)).
- Structured logging and TUI event mapping improved ([#9](https://github.com/tact-lang/pitaya/pull/9)).
- Branch namespace default hierarchical with deterministic `k<short8>` ([#35](https://github.com/tact-lang/pitaya/pull/35)).

### Removed

- Disk space preflight check ([#46](https://github.com/tact-lang/pitaya/pull/46)).
- `models.yaml` mapping ([#38](https://github.com/tact-lang/pitaya/pull/38)).
- Debug mode and auto parallelism cap calculation ([#41](https://github.com/tact-lang/pitaya/pull/41)).
- CLI flags: `--display-details`, `--allow-global-session-volume`, `--ci-artifacts` ([#65](https://github.com/tact-lang/pitaya/pull/65)).

### Fixed

- Avoid git collisions under high parallelism ([#42](https://github.com/tact-lang/pitaya/pull/42)).
- Resume reliability with new Claude Code API ([#6](https://github.com/tact-lang/pitaya/pull/6), [9714f00](https://github.com/tact-lang/pitaya/commit/9714f006027cc10d62ad588483f456ffde51881a), [27ffe75](https://github.com/tact-lang/pitaya/commit/27ffe7535b6a8ee044e98f8b2a75aed086a080b4)).

### Documentation

- New docs (Quickstart, CLI, TUI, Strategies, Custom, Config, Plugins) ([#67](https://github.com/tact-lang/pitaya/pull/67)).
- README rewritten to link docs and new repo name ([#67](https://github.com/tact-lang/pitaya/pull/67)).

## [0.1.0] - 2025-08-20

### Added

- Initial release of Pitaya.

[Unreleased]: https://github.com/tact-lang/pitaya/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/tact-lang/pitaya/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/tact-lang/pitaya/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/tact-lang/pitaya/releases/tag/v0.1.0
