# TUI

Pitaya launches an interactive TUI by default in terminals. It shows a live header, dashboard, and footer with adaptive density based on how many instances are running.

## Overview

- Header: Run ID, Strategy(params), Base branch, Started/Runtime, task counts, latest error (if any).
- Dashboard: Adapts to instance count (detailed ≤5, compact 6–30, dense >30).
- Footer: Events processed, tokens (in/out/total), cost, burn ($/h), runtime, and paths to logs/results.

## Display Modes

- Detailed (≤5 instances)
  - Per‑instance cards with status, activity, last tool, duration, error, and full branch name.
  - Highest clarity; minimal truncation.
- Compact (6–30 instances)
  - One row per instance, grouped by strategy. Columns: Status | Instance | Model | Activity | Dur | Branch.
  - No‑wrap columns; instance shows 8‑char ID; model in its own column.
  - Branch display prefers the full branch if it fits; otherwise shows only the unique tail with a leading slash (e.g., `/k…`). Tail is computed per‑strategy by trimming the common prefix across its instances.
- Dense (>30 instances)
  - Per‑strategy summaries only. Top row: progress bar with finished/total and percent.
  - Second row: colored counts (run/que/done/fail/int) plus tokens and cost.

Use `--display detailed|compact|dense` to force a mode; otherwise auto‑selection is based on instance count.

## Colors & Status

- Status colors: queued (dim), running (yellow), completed (green), failed (red), interrupted (magenta).
- Branches display in bright blue. Header highlights Strategy (green), Base (blue), Model (magenta), and Run ID (bright cyan).

## Errors & Stability

- The header shows the latest error as a red `ERR:` line if something goes wrong during rendering or state updates.
- Rendering is throttled to reduce flicker; the TUI repaints on new events, selection changes, or once per second for the runtime tick.

## Headless Output

- In non‑TTY (e.g., CI) or when `--no-tui` is set, Pitaya streams concise logs (`--output streaming`) or emits JSON (`--output json`, or `--json`).

## Offline Viewer (optional)

Inspect a previous run without the orchestrator running:

```bash
pitaya-tui --run-id run_20250114_123456
# or from an events file
pitaya-tui --events-file logs/run_20250114_123456/events.jsonl --output streaming
```

Filters:
- `--instance-id <id>`
- `--event-types task.started task.completed …`

## Tips

- Cancel with Ctrl+C; you’ll see a resume hint printed with the run ID.
- Logs live at `logs/<run_id>/`; results at `results/<run_id>/`.
- In compact mode, long branches shorten to their unique suffix with a leading `/` so the `k<short8>` tail is visible.
