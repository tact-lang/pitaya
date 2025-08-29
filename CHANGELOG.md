# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/tact-lang/pitaya/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/tact-lang/pitaya/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/tact-lang/pitaya/releases/tag/v0.1.0
