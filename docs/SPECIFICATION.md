# 1. Introduction

## 1.1 Purpose and Goals

The Orchestrator enables parallel execution of AI coding agents to explore multiple solution paths simultaneously. When using AI for software development, outcomes vary significantly between runs - even with identical prompts. This tool leverages that variability as a strength, running multiple instances in parallel and implementing selection strategies to identify the best results.

Key goals:

- **Scale AI coding horizontally** - Run N instances of Claude Code (or similar tools) in parallel
- **Implement complex workflows** - Chain instances for review, scoring, and iterative refinement
- **Abstract infrastructure complexity** - Handle containerization, git operations, and state management transparently
- **Enable rapid experimentation** - Define sophisticated multi-stage strategies in ~50 lines of code
- **Maintain solution traceability** - Every task has a branch plan, and by default a branch is created whenever the agent produced commits. You can force a branch even with no changes via `import_policy="always"`.

The tool transforms single-threaded AI coding into a parallel, strategy-driven process where you can run approaches like "generate 5 implementations, have AI review each one, pick the best scoring result, then iterate on it with feedback" - all automated.

## 1.2 Core Concepts

**Instance** - A single isolated execution of an AI coding agent (Claude Code) with a specific prompt. Each instance runs in its own Docker container with a clean git workspace, producing a new branch with its solution.

**Strategy** - A composable execution pattern that coordinates multiple durable tasks via `ctx.run`. Strategies schedule tasks, wait for results, and make decisions (e.g., best‑of‑n selection, iterative refinement).

**RunnerResult** - The runner-level output from a completed task containing: branch artifact info, token usage, cost, execution time, and final message. Strategy-specific scoring/selection metadata lives in orchestration, not in runner results.

**Orchestration Run** - A complete execution of the orchestrator with a chosen strategy. Identified by a unique run ID, it tracks all scheduled tasks, aggregate metrics, and final results.

**Event Stream** - Real-time runner events (internal) and canonical task/strategy events written by orchestration. The TUI consumes the public `task.*` and `strategy.*` events for monitoring.

**Branch-per-Task** - By default (`import_policy="auto"` with `skip_empty_import=true`), a task creates a branch named `{strategy}_{run_id}_k{short8(key)}` only if changes exist, enabling easy comparison and cherry-picking. With `import_policy="always"`, an empty branch may be created pointing at the base.

Where `short8(x)` means the first 8 hex characters of `sha256(x)`.

Default: branch only if there are commits. To always create a branch pointing to the base, set `import_policy="always"`.

Terminology: “Task” is the orchestration unit scheduled via `ctx.run`; each task is executed by a runner instance identified by `instance_id`. The TUI surfaces tasks; runner logs surface instances. Container labels carry `instance_id` for debugging, while public events and UI use task keys and `k<8hex>` short hashes.

## 1.3 Example Workflows

### Simple Parallel Exploration

```bash
orchestrator "implement user authentication with OAuth2" --strategy simple --runs 5
```

Runs 5 parallel strategies (each strategy schedules one task). The `--runs` parameter controls orchestrator-level parallelism - how many strategy executions run simultaneously. Creates 5 branches with different implementations, displays real-time progress in TUI.

### Best-of-N (inline scoring)

```python
import json

# N-way generation with inline scoring and one repair attempt
async def best_of_n(prompt: str, base_branch: str, ctx):
    # Stage 1: Generate candidates
    gens = [
        ctx.run({"prompt": f"{prompt}", "base_branch": base_branch}, key=ctx.key("gen", i))
        for i in range(5)
    ]
    ok, failed = await ctx.wait_all(gens, tolerate_failures=True)

    # Stage 2: Score each successful candidate
    def try_parse(msg):
        try:
            data = json.loads(msg)
            return data if isinstance(data.get("score"), (int, float)) else None
        except Exception:
            return None

    scored = []
    for r in ok:
        bb = r["artifact"]["branch_final"] if r["artifact"]["has_changes"] else base_branch
        review1 = await ctx.wait(
            ctx.run(
                {
                    "prompt": f"Return ONLY JSON {{score:0..10,rationale:string}} reviewing this result: {r['final_message']}",
                    "base_branch": bb,
                    "import_policy": "never",
                },
                key=ctx.key("score", r["instance_id"], "attempt-1")
            )
        )
        data = try_parse(review1["final_message"])  # try initial parse
        if data is None:
            review2 = await ctx.wait(
                ctx.run(
                    {"prompt": "Return ONLY JSON {score:0..10,rationale:string} reviewing this result again:\n"
                                f"{r['final_message']}",
                     "base_branch": bb, "import_policy": "never"},
                    key=ctx.key("score", r["instance_id"], "attempt-2")
                )
            )
            data = try_parse(review2["final_message"])  # repair attempt
        if data is None:
            continue
        scored.append((data["score"], r))

    # Select max score among valid ones
    if not scored:
        raise ctx.errors.NoViableCandidates()
    return max(scored, key=lambda t: t[0])[1]
```

### Plan → Score → Implement

```python
import json

async def plan_score_implement(prompt: str, base_branch: str, ctx):
    # Stage 1: plans
    approaches = ["performance", "simplicity", "extensibility"]
    plan_handles = [
        ctx.run({
            "prompt": f"Create a detailed plan for: {prompt} (focus: {a})",
            "base_branch": base_branch,
            "import_policy": "never"
        }, key=ctx.key("plan", a))
        for a in approaches
    ]
    plan_results = await ctx.wait_all(plan_handles)

    # Stage 2: score plans (with one repair attempt)
    def parse_json(msg):
        try:
            d = json.loads(msg)
            return d if isinstance(d.get("score"), (int, float)) else None
        except Exception:
            return None
    scored = []
    for p in plan_results:
        s1 = await ctx.wait(ctx.run({
            "prompt": "Return ONLY JSON {score:0..10,rationale:string} rating this plan:\n"
                      f"{p['final_message']}\nBe strict.",
            "base_branch": base_branch,
            "import_policy": "never"
        }, key=ctx.key("score-plan", p["instance_id"], "attempt-1")))
        data = parse_json(s1["final_message"]) 
        if not data:
            s2 = await ctx.wait(ctx.run({
                "prompt": "Your previous response did not match the schema. Return ONLY JSON {score:0..10,rationale:string} rating this plan again:\n"
                          f"{p['final_message']}",
                "base_branch": base_branch,
                "import_policy": "never"
            }, key=ctx.key("score-plan", p["instance_id"], "attempt-2")))
            data = parse_json(s2["final_message"]) 
        if data:
            scored.append((data["score"], p))

    if not scored:
        raise ctx.errors.NoViableCandidates()
    best_score, best_plan = max(scored, key=lambda t: t[0])

    # Stage 3: implement with session/branch resume
    impl_base = best_plan["artifact"]["branch_final"] if best_plan["artifact"]["has_changes"] else best_plan["artifact"]["base"]
    return await ctx.wait(ctx.run({
        "prompt": f"Implement this plan with all details:\n{best_plan['final_message']}",
        "base_branch": impl_base,
        "resume_session_id": best_plan["session_id"]
    }, key=ctx.key("implement", "plan", best_plan["instance_id"])))
```

### Complex Multi-Stage Pipeline

Use the durable `ctx.run` primitive to compose multi-stage flows. See the examples above for Best‑of‑N and Plan → Score → Implement patterns.

### Resumable Long-Running Exploration

```bash
# Start massive exploration - runs 20 parallel strategies
orchestrator "build complete e-commerce backend" --strategy best-of-n --runs 20 -S n=5

# Ctrl+C after some strategies complete
# Resume from saved state - continues remaining strategies
orchestrator --resume run_20250114_123456
```

Each workflow produces branches in your repository, detailed logs, and a final summary with costs, tokens used, and execution times for informed decision-making.

# 2. System Architecture

## 2.1 Three-Layer Design

The orchestrator implements strict separation of concerns through three independent layers, each with a single, well-defined responsibility:

**Instance Runner** - Executes individual AI coding instances in isolation. Knows nothing about strategies or UI. Provides a simple async function: give it a prompt and git repo, get back a result with a branch name. Handles all Docker lifecycle, git operations, and AI tool process management internally. Accepts orchestration-provided names and configurations (including run_id) as opaque identifiers without understanding their semantic purpose.

**Orchestration** - Coordinates multiple durable tasks according to strategies. Owns the event bus, manages parallel execution, and tracks global state. Depends only on Instance Runner's public API. Strategies schedule durable tasks with `ctx.run` and make decisions based on results. Provides branch names and container names to tasks.

**TUI (Interface)** - Displays real-time progress and results. Subscribes to orchestration events for real-time updates and periodically polls state for reconciliation. Has zero knowledge of how tasks run or how strategies work - just visualizes events and state. Can be replaced with a web UI or CLI-only output without touching other layers.

This architecture emerged from painful lessons: previous attempts failed when components were tightly coupled. Now, you can completely rewrite the TUI without touching instance execution, or swap Claude Code for another AI tool by only modifying Instance Runner.

## 2.2 Event-Driven Communication

Components communicate through a simple event system that lives within the Orchestration layer:

```python
# Instance Runner emits events via callback
await run_instance(
    prompt="implement feature",
    event_callback=lambda e: orchestrator.ingest_runner_event(e)
)

# TUI subscribes to events it cares about
orchestrator.subscribe("task.started", update_display)
orchestrator.subscribe("task.completed", show_result)
```

Events flow unidirectionally upward:

- Instance Runner emits fine-grained events (git operations, tool usage, token updates)
- Orchestration writes strategy-level events (strategy.started, strategy.completed) and maps runner events to the public task lifecycle.
- TUI consumes events to update displays

Note: Orchestration may ingest runner events for live state and diagnostics, but ONLY canonical `task.*` and `strategy.*` events are written to `events.jsonl`. Runner events are routed to `runner.log` and are not part of the public file contract.

The event system uses append-only file storage (`events.jsonl`) with monotonic byte offsets for position tracking. A single writer task computes the `start_offset` (byte position before writing the line), appends the JSON line, and flushes according to the configured flush policy (interval-based batching by default, or per-event with `--safe-fsync=per_event`). The event records this `start_offset`. Components can request events from a specific `start_offset` for recovery after disconnection.

**Offset & Writer Semantics**

- Offsets are byte positions (not line counts). Readers MUST open in binary mode and advance offsets by the exact number of bytes consumed.
- Readers MUST align to newline boundaries: if an offset lands mid-line, scan forward to the next `\n` before parsing.
- Readers MUST tolerate a truncated final line (e.g., during a crash) by skipping it until a terminating newline appears.
- Writers MUST be single-process per run (single-writer model), emit UTF-8 JSON per line, and flush according to the configured flush policy (interval-based batching by default; per-event when `--safe-fsync=per_event`). The event carries the `start_offset` (byte position before the record was written).

**Event Envelope**

- Public events in `events.jsonl`: `task.scheduled`, `task.started`, `task.completed`, `task.failed`, `task.interrupted`, and `strategy.started`/`strategy.completed`. See the “Minimum payload fields” block at the end of this document for the canonical schema and required payload fields.
- Envelope fields are authoritative for `ts`, `run_id`, and `strategy_execution_id`. Payloads MUST NOT duplicate these.
- Each event includes `{id, type, ts, run_id, strategy_execution_id, key?, start_offset, payload}`.
  - For `task.*` events, `key` is REQUIRED.
  - For `strategy.*` events, `key` is ABSENT.
  - `id` MUST be a UUIDv4 string; `ts` MUST be RFC3339/ISO‑8601 UTC with milliseconds.
- `task.scheduled.payload` includes a stable `task_fingerprint_hash` used for audit/deduplication. Fingerprint = SHA‑256 of the JCS‑normalized (RFC 8785) JSON over the exact inputs that define execution semantics:
  - Object to hash (normative schema):
    ```json
    {
      "schema_version": "1",
      "prompt": "...",
      "base_branch": "...",
      "model": "...",
      "import_policy": "auto|never|always",
      "import_conflict_policy": "fail|overwrite|suffix",
      "skip_empty_import": true,
      "session_group_key": "...",
      "resume_session_id": "...",
      "plugin_name": "...",
      "system_prompt": "...",
      "append_system_prompt": "...",
      "runner": {
        "container_limits": {"cpus": 2, "memory": "4g"},
        "network_egress": "online|offline|proxy",
        "max_turns": null
      }
    }
    ```
  - Remove non‑semantic fields: drop `metadata` entirely
  - Normalize defaults before hashing (see §6.1.1) and include `schema_version` to avoid spurious conflicts across releases
  - Compute the SHA‑256 over the UTF‑8 bytes of the JCS‑encoded JSON
  The engine MUST reject scheduling a task with the same key if the fingerprint differs from previously recorded history for that key (`KeyConflictDifferentFingerprint`). On resume, orchestration MUST reuse the previously stored normalized input when computing/validating fingerprints to avoid drift from default changes.
- Runner-specific events are not written to `events.jsonl` and instead go to runner logs.

Components follow a strict communication pattern: downward direct calls (Orchestration → Runner) are allowed for control flow, while upward communication (Runner → Orchestration, any → TUI) happens primarily through events, with periodic state polling for reconciliation. The `subscribe()` function sets up file watching on `events.jsonl`, enabling capabilities like multiple processes monitoring the same run by tailing the event file or replaying events for debugging.

## 2.3 Data Flow

The system follows a clear request/response + event stream pattern:

**Downward Flow (Requests)**

1. User provides prompt and strategy selection via CLI
2. TUI passes request to Orchestration
3. Orchestration interprets the strategy and schedules durable tasks via the runner API
4. Runner executes Claude Code in containers

**Upward Flow (Results + Events)**

1. Claude Code outputs structured logs in JSON format
2. Runner parses logs, emits runner-level events, collects results
3. Orchestration aggregates results, makes strategy decisions, emits higher-level events
4. TUI receives events and displays progress

**State Queries**

- TUI can query Orchestration for current state snapshots
- Orchestration maintains authoritative state derived from events
- No component stores UI state - everything derives from event stream

This unidirectional flow prevents circular dependencies and makes the system predictable. Each layer exposes a narrow API to the layer above it, maintaining clear boundaries and separation of concerns.

## 2.4 Technology Choices

**Python 3.11+** - Minimum 3.11, tested on 3.13. Modern async performance and improved error messages. Type hints enable clear interfaces between components. Rich ecosystem for required functionality.

**asyncio** - Natural fit for I/O-bound operations (Docker commands, git operations, API calls). Enables high concurrency for managing hundreds of container operations and git commands without thread complexity. Built-in primitives for coordination (locks, queues, events).

**docker-py 7.1.0** - Python library for Docker container management. Provides programmatic control over container lifecycle, resource limits, and cleanup. The Docker daemon version is less critical as we use standard features compatible with any recent version.

**Local Docker image** - The runner uses a local image named `claude-code`. Build it from this repo: `docker build -t claude-code .`.

**uv** - Lightning-fast Python package and project manager. Replaces pip, pip-tools, pipenv, poetry, and virtualenv with a single tool. Near-instant dependency resolution and project setup. Written in Rust for performance. Recommended: latest stable uv; dependency versions are pinned via `uv.lock` for reproducibility.

**Git** - Natural version control for code outputs. Branch-per-task model enables easy comparison. Local operations are fast and reliable. Universal developer familiarity.

**Rich 14.0.0** - Modern terminal UI capabilities with responsive layouts. Live updating without flicker. Built-in tables, progress bars, and syntax highlighting.

**Structured JSON output** - Claude Code outputs structured JSON logs that enable real-time parsing of agent actions. Provides detailed metrics and session IDs for resume capability.

**No database** - Event sourcing with file-based persistence. Reduces operational complexity. State rebuilds from event log. Git itself acts as the "database" for code outputs.

**No message queue** - Simple file-based event system suffices for single-machine operation. Append-only `events.jsonl` with offset tracking enables multi-process monitoring. Direct callback pattern reduces latency.

These choices optimize for developer experience and operational simplicity. Pinning explicit versions avoids drift, and uv simplifies Python project management compared to traditional tooling.

# 3. Instance Runner Component

## 3.1 Overview and Responsibilities

The Instance Runner is the atomic unit of execution - it runs exactly one AI coding instance in complete isolation and returns the result. This is the only component that directly interfaces with Docker, Git, and Claude Code. Its responsibility is singular: given a prompt and repository, reliably produce a git branch containing the AI's solution.

Core responsibilities:

- Spawn and manage Docker containers with proper resource limits
- Create isolated git clones and manage branch operations
- Execute Claude Code with correct parameters and parse its output
- Stream events upward via callbacks without knowing what consumes them
- Manage container lifecycle for resumability
- Track metrics (tokens, cost, duration) from execution

The Runner knows nothing about strategies, parallel execution, or UI. It accepts configuration from orchestration (branch names, container names) without understanding their purpose or conventions. This isolation enables swapping Claude Code for other AI tools or changing container runtimes without touching the rest of the system.

## 3.2 Public API

The Runner exposes a single async function `run_instance()` that accepts:

- **prompt**: The instruction for Claude Code
- **repo_path**: Path to host repository (not the isolated clone)
- **base_branch**: Starting point for the new branch (default: "main")
- **branch_name**: Target branch name for import (provided by orchestration)
- **run_id**: Run identifier for correlation (provided by orchestration)
- **strategy_execution_id**: Strategy execution identifier (provided by orchestration)
- **instance_id**: Stable identifier provided by orchestration (normative: `sha256(JCS({"run_id": ..., "strategy_execution_id": ..., "key": ...})).hex()[:16]`); runner treats as opaque and it is stable within a run across retries/resumes.
- **task_key**: Durable key for this task (for labeling/reattach; opaque to runner)
- **container_name**: Full container name including run_id (provided by orchestration)
- **model**: Claude model to use (default: "sonnet")
  Accepted examples: `sonnet`, `haiku`, `opus`. Orchestration maps these friendly names to the tool’s concrete model IDs and validates them at run start; unknown names MUST fail fast in orchestration before any tasks are scheduled.
  Runner assumes model names are pre‑validated by orchestration, and it MUST defensively validate against its configured allowed models; unknown values are a fast failure.
- **session_id**: Resume a previous Claude Code session
  Mapping: the strategy/task input field `resume_session_id` maps directly to this runner parameter `session_id`.
  
Task vs instance: The orchestration schedules tasks; the runner executes each task as an instance. `instance_id` uniquely identifies the runner instance executing a task and is surfaced in container labels and runner logs.
- **session_group_key**: Optional key to group tasks that should share the same persistent session volume; defaults to the task key if not provided
- **event_callback**: Function to receive real-time events
- **timeout_seconds**: Maximum execution time (default: 3600)
- **container_limits**: CPU/memory restrictions
- **auth_config**: Authentication configuration (OAuth token or API key)
- **reuse_container**: Reuse existing container if name matches (default: True)
- **finalize**: Stop container after completion (default: True)
- **retry_config**: Retry configuration with pattern-based error matching
- **docker_image**: Custom Docker image to use (default: from config)
- **plugin_name**: AI tool plugin to use (default: "claude-code")
- **system_prompt**: System prompt to prepend to Claude's instructions
- **append_system_prompt**: Additional system prompt to append
- **import_policy**: "auto" | "never" | "always" (default: "auto")
- **import_conflict_policy**: "fail" | "overwrite" | "suffix" (default: "fail")
- **skip_empty_import**: Whether to skip import when no changes (default: True)

Returns `RunnerResult` containing:

- Branch name where solution was imported to host repository
- Execution metrics: `metrics.cost_usd`, `metrics.tokens_in`, `metrics.tokens_out`, `metrics.duration_s`
- Token breakdown: `tokens_total = tokens_in + tokens_out`
- Session ID for resume capability
- Final message from Claude
- Status (success/failed/timeout/interrupted)
- Container and branch details
- Commit statistics (count, lines added/deleted)
- Timestamps (started_at, completed_at)
- Number of retry attempts
- Path to detailed logs
- Workspace path (until cleanup)
- Error information if failed
- has_changes: Boolean indicating if any code changes were made
- Convenience properties: `cost` (maps to `metrics.cost_usd`), `tokens_total` (maps to `metrics.tokens_in + metrics.tokens_out`)

The API is designed for both simple usage (`await run_instance("fix bug", repo)`) and advanced control when needed. The runner remains agnostic about why the container is named a certain way or how branches are named - it simply uses what orchestration provides. This separation keeps the runner generic and testable.

## 3.3 Execution Pipeline

The runner follows six strict phases ensuring consistent execution and proper cleanup:

1. **Validation** - Verify Docker daemon, repository exists, validate provided container_name, check disk space
2. **Workspace Preparation** - Create isolated git clone in temporary directory BEFORE container creation. This ensures the workspace is ready when the container starts
3. **Container Creation** - If `reuse_container=True` and container exists, reuse it. Otherwise, start new container with provided name and resource limits, mount the pre-prepared workspace. Container runs with `sleep infinity` for persistence
4. **Claude Code Execution** - Execute via `docker exec` with structured JSON output, parse events
5. **Result Collection** - Extract metrics, and (if `import_policy` is `auto` or `always`) import branch from workspace to host repository as the final step. Before any import, the runner MUST perform an idempotent pre-check (compare workspace HEAD to existing branch HEAD) and treat exact matches as already imported to avoid duplicates across the crash window. Orchestration emits `task.completed` only after the runner finishes and import (per policy) succeeds; for `auto` with zero changes, import is skipped and the task completes with `artifact.has_changes=false`.
6. **Cleanup Decision** - For successful instances: delete workspace immediately. If `finalize=True`, stop the container (retained for 2h). If `finalize=False`, leave the container running (see persistence semantics below). For failed instances: keep both workspace and container for 24h debugging

Each phase emits structured events for observability. The pipeline is designed for resumability - if execution fails at phase 4, a retry can resume from that point using the session_id without repeating setup.

Key architectural boundaries:

- Container naming comes from orchestration (runner doesn't generate IDs)
- Git isolation happens through separate clones, not shared mounts
- Each instance works in complete isolation

The runner executes these phases mechanically without understanding the broader orchestration context. It doesn't know why containers are named a certain way - it just follows the contract.

## 3.4 Docker Management

Containers follow a careful lifecycle designed to provide complete isolation while preserving Claude Code sessions for resumability:

**Container Creation**: Uses the container name provided by orchestration, ensuring consistent naming across the system. The runner treats run_id and other identifiers as opaque values provided by orchestration. Each container represents one isolated AI coding instance.

**Pre-Container Setup**: Before starting any container, the Instance Runner prepares an isolated workspace:

```bash
# Create temporary directory for this task (key-derived)
# Note: Orchestration computes short hashes cross-platform (Python), not via shell utilities.
# Provided environment vars:
#   TASK_KEY              # durable task key (fully qualified)
#   SESSION_GROUP_KEY     # optional session group key; defaults to TASK_KEY
#   KHASH                 # short_hash(TASK_KEY) = first 8 hex of SHA-256
#   GHASH                 # per §3.4 GHASH (run/global scope)

WORKSPACE_DIR="/tmp/orchestrator/${run_id}/k_${KHASH}"
mkdir -p "$WORKSPACE_DIR"

# Clone the specific branch in isolation (handled by git operations)
# ... git clone happens here ...
```

**Volume Mounts**: Three mounts required for proper operation:

```
/tmp/orchestrator/<run-id>/k_<hash(key)> → /workspace
orc_home_<run_id>_g{GHASH} → /home/node
tmpfs → /tmp
```

Platform-specific mount handling ensures compatibility. The named volume (pattern: `orc_home_{run_id}_g{GHASH}`) provides persistent storage for Claude Code's session data, ensuring sessions survive container restarts and enabling cross-task continuity when the same session group key is used. Note: The implementation uses `/home/node` due to the base Docker image using the `node` user.

`GHASH` (normative):
  • Run scope (default): `GHASH = short8( sha256( JCS({"session_group_key": EFFECTIVE_SGK}) ) )`, where `EFFECTIVE_SGK = session_group_key if provided else durable task key`.
  • Global scope: `GHASH = short8( sha256( JCS({"session_group_key": "...", "plugin": "...", "model": "..."}) ) )`.
When `runner.session_volume_scope="global"`, the named volume omits the `run_id` prefix and uses only `GHASH`. Global scope is advanced and requires `--allow-global-session-volume`.

**Container Execution**: Containers run with these security constraints:

```bash
# Compute mount flags for workspace; apply SELinux and read-only for reviews
mount_flags=""
if [[ "$(uname)" == "Linux" ]] && [[ -e /sys/fs/selinux ]]; then
  mount_flags=":z"
fi
if [ "${import_policy:-auto}" = "never" ]; then
  if [[ "$mount_flags" == ":z" ]]; then
    mount_flags=":ro,z"
  else
    mount_flags=":ro"
  fi
fi

docker run \
  -v "$WORKSPACE_DIR:/workspace${mount_flags}" \
  --mount "type=volume,source=orc_home_${run_id}_g${GHASH},target=/home/node" \
  --tmpfs /tmp:rw,size=256m \       # Writable temporary space
  --read-only \                     # Entire filesystem read-only except mounts
  --name "$container_name" \
  --cpus="2" \
  --memory="4g" \
  --label "orchestrator=true" \
  --label "run_id=${run_id}" \
  --label "strategy_execution_id=${strategy_execution_id}" \
  --label "strategy_index=${sidx}" \
  --label "task_key=${TASK_KEY}" \
  --label "session_group_key=${SESSION_GROUP_KEY:-$TASK_KEY}" \
  --label "orchestrator.last_active_ts=$(date -u +%Y-%m-%dT%H:%M:%S.%3NZ)" \
  --label "instance_id=${instance_id}" \
  claude-code \
  sleep infinity                    # Keep container running for exec
```

The `--read-only` flag locks down the entire container filesystem except for explicitly mounted volumes. Containers have access to exactly three locations: `/workspace` (isolated git clone), `/home/node` (session state), and `/tmp` (temporary files). The `sleep infinity` command keeps the container running, allowing subsequent `docker exec` commands to run Claude Code. Mount flags include `:z` on SELinux systems and `:ro` when `import_policy="never"`.

When a task declares `import_policy="never"`, the runner SHOULD mount the `/workspace` path as read-only when `runner.review_workspace_mode=ro`. Default is `runner.review_workspace_mode=ro`. If `reuse_container=True` would prevent a read-only remount, the runner MUST start a fresh container (ignoring `reuse_container`) to honor read-only when `review_workspace_mode=ro`. Review/scoring tasks MAY always write to `/tmp` and `/home/node`. When `review_workspace_mode=ro`, attempted writes to `/workspace` MUST error fast with a clear message. Plugins MAY override scratch locations to `/tmp` to enable strict RO mode.

**Container Reuse**: When `reuse_container=True`:

- If container exists and is running: Execute Claude Code in existing container
- If container exists but stopped: Start container, then execute  
- If container doesn't exist: Create new container as above

**Runner Decision Tree (Normative)**:

1. If `import_policy="never"` and `review_workspace_mode="ro"`, **force** RO `/workspace`.
2. If a container exists but can't be remounted RO, **replace** it (same name, same session volume) and log `runner.container.replaced`.

Priority order when constraints conflict:

1. Safety — enforce read-only `/workspace` for `import_policy="never"`
2. Session continuity — reuse the same session volume via `session_group_key`
3. Container reuse — reuse an existing container if it does not violate (1)

If (1) and (3) conflict, the runner MUST start a fresh container attached to the same session volume to satisfy read-only.

Container replacement protocol: To honor mount mode changes (e.g., switching to read‑only), the runner MAY stop and remove the existing container and re-create a new one with the same deterministic `container_name`, attached to the same session volume. This replacement MUST be recorded via a runner-internal event (e.g., `runner.container.replaced`) and does not alter public events. Retention timers apply to the latest active container; replaced containers are not retained.

**Container Persistence**: Containers are kept after execution to enable Claude Code session resume:

- Failed/timeout instances: Retained for 24 hours (configurable)
- Successful instances with `finalize=True`: Container is stopped (not removed) after successful completion and retained for 2 hours (configurable)
- Successful instances with `finalize=False`: Container continues running until explicitly stopped by the operator or via an orchestration cleanup command; there is no implicit auto‑stop

This persistence is crucial because Claude Code maintains session state inside the container. If an instance times out after partial work, the session can be resumed by using the same container with the preserved session. Operators can stop non‑finalized containers using explicit cleanup commands (e.g., `orchestrator --stop-nonfinalized` or similar tooling); the orchestrator does not auto‑stop them by default.

**Session Groups**: To maintain session continuity across multiple tasks (e.g., plan → implement → refine), strategies may supply a `session_group_key` in `ctx.run`. Containers and the `/home/node` named volume are keyed by this group (default = the task key). Passing the same `session_group_key` for related tasks ensures they share the same session volume; passing a different key isolates sessions. Volume scope is configurable via `runner.session_volume_scope = "run" | "global"` (default: `"run"`). In `"global"` scope, the volume name omits the `run_id` so sessions can be reused across runs, but this requires explicit `--allow-global-session-volume` to prevent cross-contamination between different model/tool configs; in `"run"` scope, volumes are run‑scoped.

**Resource Limits**: Each container gets configurable CPU/memory limits with sensible defaults:

- CPUs: 2 (prevents CPU starvation of parallel instances)
- Memory: 4GB (sufficient for most coding tasks)
- No disk quotas (workspace size limited by temporary directory)

These limits prevent runaway instances from affecting system stability while providing enough resources for complex coding tasks.

**Post-Container Cleanup**: After the container completes and branches are successfully imported:

```bash
# For successful instances - immediate workspace cleanup
if [ "$status" = "success" ]; then
    rm -rf "$WORKSPACE_DIR"  # Remove isolated clone immediately
    # Respect finalize: stop container only when finalize=true
    if [ "$finalize" = "true" ]; then
        docker stop "$container_name"  # Stop but retain container for 2h
    fi
fi
# Failed instances keep both workspace and container for 24h debugging
```

**Orphan Cleanup**: Removed. Orchestration owns scanning and cleanup of orphan containers by run namespace (see §4.11). The Runner only cleans up its current container and workspace on task failure.

**Image Structure**: The custom Docker image includes:

- Claude Code CLI with all dependencies
- Git and essential development tools
- Python, Node.js, and common runtimes
- Non-root user configuration per Claude Code requirements
- Minimal size while supporting typical development scenarios

**Environment Configuration**: Each container receives base configuration plus authentication:

```yaml
environment:
  - GIT_AUTHOR_NAME="AI Agent"
  - GIT_AUTHOR_EMAIL="agent@orchestrator.local"
  - GIT_COMMITTER_NAME="AI Agent"
  - GIT_COMMITTER_EMAIL="agent@orchestrator.local"
  - PYTHONUNBUFFERED="1"
  # Plus authentication based on auth_config:
  # For subscription mode:
  - CLAUDE_CODE_OAUTH_TOKEN="${oauth_token}"
  # For API mode:
  - ANTHROPIC_API_KEY="${api_key}"
  - ANTHROPIC_BASE_URL="${base_url}"  # if provided
```

Session resumption is handled via Claude Code's `--resume` CLI flag with the session_id.

**Container Status Tracking**: The implementation may use an additional status tracking mechanism via `/tmp/orchestrator_status/` on the host filesystem. Container labels include `task_status` which may be updated throughout the lifecycle, but the authoritative status lives in orchestration state and public events.

The Docker management layer remains agnostic about orchestration strategies or git workflows. It simply provides isolated execution environments with proper resource controls and session preservation. The true isolation comes from each container working with its own complete git repository, with no possibility of cross-contamination between instances or with the host system.

## 3.5 Git Operations

Git operations use a carefully designed isolation strategy that achieves perfect separation between parallel instances while maintaining efficiency. This approach emerged from the fundamental requirement that each AI agent must work in complete isolation, seeing only the branch it needs to modify, without any possibility of interfering with other instances or accessing the host repository.

**The Isolation Strategy**: Before starting each container, the Instance Runner creates a completely isolated git repository on the host:

Note: Shell snippets in this section are illustrative. The reference implementation performs these steps in Python for cross‑platform portability (Linux/macOS/Windows/WSL2), avoiding dependencies on non‑portable shell tools.

```bash
# On the host, before container starts
git clone --branch <base-branch> --single-branch --no-hardlinks \
          /path/to/host/repo /tmp/orchestrator/${run_id}/k_${KHASH}
cd /tmp/orchestrator/${run_id}/k_${KHASH}
git remote remove origin        # Complete disconnection from source
# Store base branch reference for later
echo "<base-branch>" > .git/BASE_BRANCH
git rev-parse HEAD > .git/BASE_COMMIT
```

Let's break down why each flag matters:

- `--branch <base-branch> --single-branch` ensures the clone contains ONLY the target branch. No other refs exist in `.git/refs/heads/`. The agent literally cannot see any other branches because they don't exist in its universe
- `--no-hardlinks` forces Git to physically copy all object files instead of using filesystem hardlinks. This prevents any inode-level crosstalk between repositories - critical for true isolation
- `git remote remove origin` completes the isolation. With no remote configured, the agent cannot push anywhere even if it tried
- Storing the base branch reference ensures we maintain traceability of where changes originated

**Isolation Modes**:

- `full_copy` (default): `--no-hardlinks --single-branch`, no remotes; maximizes isolation.
- `shared_objects`: partial clone/alternates for speed; reduces isolation and should be used only when disk IO is a bottleneck. **Warning**: Reduced isolation using Git alternates/partial clone. Requires explicit flag or env to enable.

**Agent Commits**: The AI agent works directly on the base branch in its isolated clone. It doesn't create new branches - it simply commits its changes to the only branch it can see. This simplifies the agent's task and ensures predictable behavior.

**Container Workspace**: Each container receives its isolated clone as a volume mount:

```bash
# Compute mount flags for workspace; apply SELinux and read-only for reviews
mount_flags=""
if [[ "$(uname)" == "Linux" ]] && [[ -e /sys/fs/selinux ]]; then
  mount_flags=":z"
fi
if [ "${import_policy:-auto}" = "never" ]; then
  if [[ "$mount_flags" == ":z" ]]; then
    mount_flags=":ro,z"
  else
    mount_flags=":ro"
  fi
fi

docker run \
  -v /tmp/orchestrator/${run_id}/k_${KHASH}:/workspace${mount_flags} \
  --read-only \
  --name ${container_name} \
  claude-code
```

The SELinux flag is only applied on Linux systems where it's needed. The `--read-only` flag locks down everything except the mounted volumes. The container can only modify its isolated git repository.

Note: This minimal example omits the `/home/node` named volume and `/tmp` tmpfs shown in 3.4; those mounts are required in the full runner configuration.

**Branch Import After Completion**: The magic happens after the container exits, as the final step of run_instance(). Instead of complex push coordination, we use Git's ability to fetch from local filesystem paths:

```bash
# After container exits, back on the host
cd /path/to/host/repo

# Acquire per-repo import lock to serialize ref/pack updates
# (Cross-platform, process-internal lock implemented by the runner
# using fcntl/msvcrt/portalocker; do not rely on shell flock.)

# Idempotent import pre-check (handles crash window)
# (Illustrative shell; reference implementation performs these checks in Python under the process-level lock.)
if git show-ref --verify --quiet "refs/heads/${branch_name}"; then
  WS_HEAD=$(git --git-dir="/tmp/orchestrator/${run_id}/k_${KHASH}/.git" rev-parse HEAD)
  BR_HEAD=$(git rev-parse "refs/heads/${branch_name}")
  if [ "$WS_HEAD" = "$BR_HEAD" ]; then
    echo "Branch already matches workspace HEAD; treating as completed."
    # Return success to orchestration; it will emit task.completed
    exit 0
  fi
fi

## Provenance-based suffix idempotency (see §4.7)
if [ "${import_conflict_policy:-fail}" = "suffix" ] && ! git show-ref --verify --quiet "refs/heads/${branch_name}"; then
  WS_HEAD=$(git --git-dir="/tmp/orchestrator/${run_id}/k_${KHASH}/.git" rev-parse HEAD)
  CANDIDATE=$(git for-each-ref --format='%(refname:short) %(objectname)' refs/heads \
    | awk -v base="${branch_name}" -v head="$WS_HEAD" 'match($1, "^" base "(_[0-9]+)?$") && $2==head {print $1; exit}')
  if [ -n "$CANDIDATE" ]; then
    if git notes --ref=orchestrator show "$WS_HEAD" 2>/dev/null | grep -q "task_key=${TASK_KEY}"; then
      echo "Found existing suffixed branch $CANDIDATE with matching HEAD + provenance; treating as completed."
      exit 0
    fi
  fi
fi

# Determine target branch honoring import_conflict_policy
TARGET_BRANCH="${branch_name}"
if git show-ref --verify --quiet "refs/heads/${TARGET_BRANCH}"; then
  case "${import_conflict_policy:-fail}" in
    fail)
      echo "Error: Branch ${TARGET_BRANCH} already exists" >&2; exit 1 ;;
    overwrite)
      : # proceed; fetch will move ref (forced update)
      ;;
    suffix)
      i=2
      while git show-ref --verify --quiet "refs/heads/${TARGET_BRANCH}_${i}"; do i=$((i+1)); done
      TARGET_BRANCH="${TARGET_BRANCH}_${i}"
      ;;
  esac
fi

# Skip empty imports if enabled (no commits relative to base)
BASE_BRANCH=$(cat /tmp/orchestrator/${run_id}/k_${KHASH}/.git/BASE_BRANCH)
if [ "${import_policy:-auto}" = "never" ]; then
  echo "Import policy is 'never'; skipping branch import"
  exit 0
fi

if [ "${skip_empty_import:-true}" = "true" ] && [ "${import_policy:-auto}" != "always" ]; then
  BASE_COMMIT=$(cat /tmp/orchestrator/${run_id}/k_${KHASH}/.git/BASE_COMMIT)
  COUNT=$(git --git-dir="/tmp/orchestrator/${run_id}/k_${KHASH}/.git" rev-list --count "${BASE_COMMIT}..HEAD" || echo 0)
  if [ "$COUNT" = "0" ]; then
    echo "No changes; skipping branch creation for ${TARGET_BRANCH}"
    # Caller records artifact.has_changes=false and planned branch name
    exit 0
  fi
fi

# Import from the base branch to new (or suffixed) branch name
if [ "${import_conflict_policy:-fail}" = "overwrite" ]; then
  git fetch /tmp/orchestrator/${run_id}/k_${KHASH} +HEAD:${TARGET_BRANCH}
else
  git fetch /tmp/orchestrator/${run_id}/k_${KHASH} HEAD:${TARGET_BRANCH}
fi
```

This import operation:

- Copies all new commits and their associated blobs/trees
- Creates the branch atomically in the host repository
- Requires no network operations or remote configuration
- Fails cleanly if the branch already exists (according to `import_conflict_policy`)
- Preserves the connection to the original base branch
- For `import_policy="always"`, handles the "no changes" case by creating a branch pointing to the base.
- For `import_policy="auto"` with `skip_empty_import=true` (default), no branch is created when there are no commits.

**Runner-internal event emission and cleanup**: The Runner MUST NOT write to `events.jsonl`. Only Orchestration emits public `task.*`/`strategy.*` events after `run_instance()` returns. The instance runner must emit the final runner-internal completion event BEFORE cleaning up the workspace:

```bash
# Emit runner-specific completion event with workspace path (to runner.log, not events.jsonl)
emit_runner_event("runner.instance.completed", {
    "workspace_path": "/tmp/orchestrator/${run_id}/k_${KHASH}",
    "branch_imported": TARGET_BRANCH
})

# Then clean up after successful import
rm -rf /tmp/orchestrator/${run_id}/k_${KHASH}
```

This ensures the runner log contains the workspace path for audit trails before it's deleted. Public events in `events.jsonl` do not include host workspace paths to avoid leaking host paths; if a strategy needs provenance, prefer sidecar metadata (results directory) or git notes rather than adding host paths to public events. On success, the workspace path is only valid until the completion event is emitted; callers must not rely on it afterward.

**Performance Characteristics**: Testing with 50 parallel instances showed:

- Clone operations complete in 0.5-2 seconds depending on repository size
- Each isolated clone uses full disk space (no sharing), but this is negligible with modern storage
- No workspace cross-contamination; branch import is serialized and collision-safe
- Branch import via fetch takes milliseconds to low seconds depending on changeset size
- No performance degradation with high parallelism

**Why This Approach Is Superior**:

- **True isolation**: Each agent gets a complete, disconnected repository. It cannot see or affect anything outside its container
- **No coordination needed**: Since instances never push, there's no need for locks or semaphores
- **Atomic branch creation**: The fetch operation either succeeds completely or fails cleanly
- **Offline operation**: Everything works without network access since we're using filesystem operations
- **Debugging friendly**: Each instance's work is preserved in its temporary directory until explicitly cleaned up
- **Branch lineage preserved**: Original base branch reference maintained throughout

**Alternative Bundle Approach**: For environments requiring even stronger isolation, branches can be exported as bundle files:

```bash
# Inside container before exit
git bundle create /workspace/result.bundle HEAD

# On host after container exits
git fetch /tmp/orchestrator/${run_id}/k_${KHASH}/result.bundle HEAD:${branch_name}
```

Bundles are self-contained files containing exactly the commits and objects reachable from the specified branch. They provide tamper-evidence and can be archived for audit purposes.

The beauty of this solution is its conceptual simplicity. By preparing isolated environments before container start and importing results after completion, we achieve perfect isolation without any complex coordination. Each instance truly runs in its own universe, and the Instance Runner simply collects the results afterward.

## 3.6 Claude Code Integration

Integration focuses on reliable execution and comprehensive event extraction:

**Process Management**: Claude Code runs via `docker exec` within the persistent container. The `--print` mode with `--output-format stream-json` provides non-interactive execution with structured JSONL (JSON Lines) output for parsing.

**Event Stream Parsing**: Real-time parsing of Claude's JSONL output where each line is a JSON object:

```json
// Initial system message
{"type":"system","subtype":"init","session_id":"...","tools":[...],"model":"..."}

// Assistant messages with tool usage
{"type":"assistant","message":{"content":[{"type":"tool_use","name":"Write","input":{...}}]}}

// User messages with tool results
{"type":"user","message":{"content":[{"type":"tool_result","content":"..."}]}}

// Final result with metrics
{"type":"result","subtype":"success","total_cost_usd":0.42,"usage":{"input_tokens":1200,"output_tokens":900}}
```

Key events extracted:

- Tool usage (Write, Edit, Bash commands, etc.) - emitted as `runner.tool_use` events
- Token consumption from usage field - tracked in real-time
- Cost accumulation from total_cost_usd - aggregated across the session
- Error states from is_error flag
- Session ID for resume capability
- Tool results and output - emitted as `runner.tool_result` events
- Phase transitions and workspace events

**Session Handling**: Each execution gets a unique session_id from Claude Code (e.g., `"2280667e-25e1-46ac-b7f4-722d7e486c9c"`). This ID enables resuming interrupted work via the `--resume <session_id>` CLI flag. Sessions persist in the named volume mounted at `/home/node`, surviving container restarts.

**Authentication**: Based on auth_config, the appropriate environment variables are set:

- Subscription mode: `CLAUDE_CODE_OAUTH_TOKEN`
- API mode: `ANTHROPIC_API_KEY` and optionally `ANTHROPIC_BASE_URL`

**Prompt Engineering**: System prompts and append-system-prompt options enable customizing Claude's behavior per task. Model selection allows choosing between speed (Sonnet) and capability (Opus). The default model is configurable via `ORCHESTRATOR_DEFAULT_MODEL` (default: `sonnet`).

## 3.7 Error Handling

Errors are categorized for appropriate recovery strategies:

**Container Errors**: Docker daemon issues, resource exhaustion, network problems. Generally retriable with new container.

**Git Errors**: Clone failures, missing branches, fetch issues. Usually require configuration fixes, not retriable.

**Claude Code Errors**: API failures, timeout, cost limits. Retriable using session resume to continue from last checkpoint.

**Timeout Handling**: Tool-level turn/time limits as supported by the selected plugin (optional `runner.max_turns` config; default: unset). Container-level timeout (default 1 hour) as a fallback. Timed-out instances keep containers for resume.

**Retry Strategy**: Automatic exponential backoff for transient failures during execution. Session-based resume for Claude Code failures. Maximum retry attempts configurable (default: 3). Retries handle temporary issues like network blips or API rate limits. After exhausting retries, the instance is marked as failed. This is distinct from orchestration-level resume after interruption, where already-failed instances remain failed.

**Retryable Error Patterns**: Specific error strings trigger automatic retry:

- Docker: "connection refused", "no such host", "timeout", "daemon", "Cannot connect"
- Claude: "rate limit", "API error", "connection reset", "overloaded_error"
- General: "ECONNREFUSED", "ETIMEDOUT", "ENETUNREACH"
- System: OSError, IOError

The retry configuration supports pattern-based matching with tuples of (pattern, error_type) for fine-grained control.

## 3.8 Runner Plugin Interface

Abstract interface enables supporting multiple AI coding tools:

**Core Methods**:

- `validate_environment()`: Check tool is available and configured
- `prepare_environment()`: Return environment variables including authentication
- `build_command()`: Construct tool-specific command with all parameters
- `execute()`: Run tool and parse its output format
- `parse_events()`: Transform tool output into common event schema
- `handle_error()`: Process tool-specific errors and determine retry strategy

**Capability Flags**: Each plugin declares its supported features:

```python
class PluginCapabilities:
    supports_resume: bool = False
    supports_cost: bool = False
    supports_streaming_events: bool = True
    supports_token_counts: bool = True
    supports_streaming: bool = True
    supports_cost_limits: bool = False
    auth_methods: List[str] = ["oauth", "api_key"]
```

Semantics:

- `supports_streaming`: tool streams assistant content incrementally (partial responses)
- `supports_streaming_events`: tool emits structured step events while running (distinct from content streaming)

Orchestration checks required plugin capabilities before scheduling tasks and fails fast with clear errors if unsupported features are needed. This capability gating happens once at run start, not per task. If a strategy requires session resume and the selected plugin has `supports_resume=False`, the run MUST fail at startup with a clear message.

**Authentication Handling**: Each plugin manages its own authentication approach:

```python
# Plugin receives auth config and decides how to use it
auth_config = {"oauth_token": "...", "api_key": "...", "base_url": "..."}
env = await plugin.prepare_environment(auth_config)
```

This keeps tool-specific auth logic encapsulated within the appropriate plugin.

**Implementation Strategy**: The runner controls the full container lifecycle and mounts. Plugins provide capabilities, authentication environment, command construction, event parsing, and error handling. Container creation and teardown remain in the runner (plugins MUST NOT create containers).

**Selection**: Runners are selected via configuration. Allows users to choose their preferred AI tool while keeping same orchestration features.

**Graceful Degradation**: When plugins lack certain capabilities:

- If `supports_cost` is False but `supports_token_counts` is True: estimate costs using configured rates
- If neither cost nor token support: display "N/A" for these metrics
- If `supports_resume` is False: strategies requiring resume capability fail with clear message

# 4. Orchestration Component

## 4.1 Overview and Responsibilities

The Orchestration layer coordinates multiple tasks according to strategies. While Instance Runner handles "how to run one instance", Orchestration handles "how to run many tasks intelligently". This separation enables complex workflows like "generate 5 solutions and pick the best" without complicating the runner logic.

Core responsibilities:

- Execute strategies that schedule and coordinate tasks
- Manage parallel execution with resource limits
- Own the event bus for component communication
- Track current state of all tasks and strategies
- Generate container names from durable task keys for grouping (`orchestrator_{run_id}_s{sidx}_k{short8(key)}`), where `sidx` is the 1-based index of the strategy execution within the run
- Generate deterministic branch names from durable task keys (provided to the runner)
- Handle task failures according to strategy logic
- Support resumption of interrupted runs
- Clean up orphaned containers on startup
- Export comprehensive results including metrics and branches
- Validate disk space before execution

The orchestration depends only on Instance Runner's public API. It exposes clean interfaces for running strategies and querying state, hiding all complexity of parallel coordination.

Key architectural decisions:

- Container naming scheme (`orchestrator_{run_id}_s{sidx}_k{short8(key)}`) is orchestration's responsibility (derived from the durable task key)
- Branch naming pattern is deterministic from the durable task key (orchestration provides the final name)
- Strategies work with high-level abstractions, unaware of these coordination details

This layer transforms the simple "run one instance" capability into powerful multi-instance workflows while keeping both runners and strategies clean.

## 4.2 Public API

The Orchestration component provides five essential interfaces:

**Strategy Execution**: The `run_strategy()` function accepts a strategy name, prompt, base branch, and strategy-specific configuration. Returns a list of task results representing the final outputs. This simple interface hides all complexity - whether the strategy runs 1 or 100 tasks internally.

**State Management**: The `get_current_state()` function returns a complete snapshot of the system: running tasks, active strategies, aggregate metrics. This enables the TUI to display current status without tracking individual events. State includes derived data like "5 of 10 tasks complete" for progress displays.

**Event System**: The `subscribe()` function sets up file watching on `events.jsonl` to receive real-time events. Orchestration writes the canonical public events (`task.scheduled|started|completed|failed|interrupted` and `strategy.started|completed`) — see the “Minimum payload fields” block at the end of this document for the canonical schema. Runner-specific events are kept in runner logs and are not part of the public file contract. The runner NEVER writes public `task.*`/`strategy.*` events; orchestration emits them, and `task.completed` is written only after `run_instance()` returns. This decoupling means TUI never directly calls Instance Runner. Even in-process components use file watching for consistency.

**Event History**: The `get_events_since()` function retrieves historical events from `events.jsonl` starting at a given `start_offset` (byte position before the line) or timestamp. Accepts run_id, offset/timestamp, optional event type filters, and result limit. The events file is never rotated during an active run; rotation/archival may occur only after the run completes.

**Run Resumption**: The `resume_run()` function restores an interrupted run from saved state. It loads the state snapshot, replays events, verifies container existence, checks plugin capabilities, and continues execution from where it left off. Supports both session-based resume and fresh restart options.

These APIs follow a key principle: make simple things simple, complex things possible. Running a basic strategy is one function call, but power users can subscribe to detailed events for monitoring.

## 4.3 Strategy System

Durable strategies reduce orchestration to a single primitive while guaranteeing seamless resume and no duplicate work. Strategies are deterministic async Python that schedule durable tasks and wait for their results using a tiny surface area:

**API Surface**:

- `handle = ctx.run(task: dict, *, key: str, policy: RetryPolicy|None=None)` — schedules a durable task and returns a Handle immediately.
- `result = await ctx.wait(handle)` / `results = await ctx.wait_all(list[Handle])` — await completion.
- `ctx.parallel([ ... ])` — sugar for schedule-many + wait_all.
- Deterministic utilities: `ctx.key(*parts) -> str` (stable idempotency keys), `ctx.now()`, `ctx.rand()` (recorded + replayed), `await ctx.sleep(seconds)` (checkpointed). `ctx.rand()` values MUST be recorded per strategy execution (e.g., strategy-scoped events/state) so replay is deterministic without re-seeding.

Handle contract:

```python
class Handle:
    key: str            # fully-qualified durable key (run_id/strategy_execution_id/..)
    scheduled_at: float # monotonic timestamp (seconds)
    # Opaque; not awaitable directly
```

Waiting contract:

- `await ctx.wait(handle)` → returns `TaskResult` or raises `TaskFailed(key, error_type, message)`
- `await ctx.wait_all(handles, tolerate_failures=False)` →
  - default: returns `list[TaskResult]` or raises `AggregateTaskFailed([key...])`
  - if `tolerate_failures=True`: returns `(successes: list[TaskResult], failures: list[TaskError])`

Retry policy: The `policy` parameter defaults to `None` (no orchestration-level retry). Orchestration-level `policy` applies only to scheduling/reattach failures (e.g., queuing, container start/attach). Once the runner starts executing a task, runner-level `retry_config` handles transient execution errors (API/rate-limit/network) with session-resume.

Exceptions (thrown by orchestration utilities):

- `TaskFailed(key, error_type, message)` — raised by `wait()` when a single task fails.
- `AggregateTaskFailed([key...])` — raised by `wait_all()` when one or more tasks fail and `tolerate_failures=False`.
- `NoViableCandidates()` — recommended to signal selection failure in strategies (e.g., Best‑of‑N scoring yields zero valid candidates).
- `KeyConflictDifferentFingerprint(key)` — scheduling error: same key with a different canonical fingerprint.

`ctx.key(...)` always expands to an internal fully-qualified key namespaced by the current run and strategy execution: `run_id/strategy_execution_id/<joined parts>`. Authors only supply the `<joined parts>`; namespacing prevents collisions across parallel runs. Normative: `instance_id = sha256(JCS({"run_id": ..., "strategy_execution_id": ..., "key": ...})).hex()[:16]` — stable within a run across retries/resumes. UI **short form** is `inst-<first5>` for display only; the full 16‑hex `instance_id` MUST appear in labels and events.

**Task Input Schema** (single provider under the hood); `TaskResult` returned to strategies maps directly from `RunnerResult` fields exposed by the runner:

```json
{
  "prompt": "string",
  "base_branch": "string",
  "model": "string (optional)",
  "import_policy": "auto|never|always (optional, default: auto)",
  "import_conflict_policy": "fail|overwrite|suffix (optional)",
  "skip_empty_import": true,
  "session_group_key": "string (optional)",
  "resume_session_id": "string (optional)",
  "metadata": { "any": "json" }  // optional, stored for traceability
}

Field mapping to Runner API: `resume_session_id` (strategy input) maps to Runner `session_id`.
```

**Result Shape (TaskResult)** returned by `wait`/`wait_all` (fields stable unless marked optional):

```json
{
  "artifact": {
    "type": "branch",
    "branch_planned": "<name>",
    "branch_final": "<name|null>",
    "base": "<base>",
    "commit": "<sha>",
    "has_changes": true
  },
  "final_message": "string",
  "metrics": {"tokens_in":0,"tokens_out":0,"cost_usd":0.0,"duration_s":0.0},
  "session_id": "string",
  "instance_id": "string",
  "status": "success|failed|timeout|interrupted"
}
```

Artifact field semantics:

- `artifact.commit` is the tip commit of the imported branch and is the primary stable identifier of the artifact. For `import_policy="always"` with no changes, it equals the base commit. For `import_policy="never"` or skip-empty cases, `artifact.has_changes=false`, `artifact.branch_final=null`, and `artifact.commit` equals the base commit for provenance.
- `artifact.branch_planned` is the deterministic name planned from the durable key; `artifact.branch_final` is the actual created branch (may differ when `import_conflict_policy="suffix"`).

`instance_id` is stable across runner-level retries, container replacements, and resumes within a run. Normative derivation: `instance_id = sha256(JCS({"run_id": ..., "strategy_execution_id": ..., "key": ...})).hex()[:16]`. `TaskResult` is the strategy-facing view of the runner's `RunnerResult`.

There are no other orchestration operations. Generation, review/scoring, and refinement are all regular `ctx.run` tasks with distinct keys.

`wait(handle)` raises on task failure. `wait_all(handles)` raises an aggregated error by default; callers may opt into partial tolerance via `await ctx.wait_all(handles, tolerate_failures=True)` which returns `(successes, failures)`.

**Durability Semantics**:

- Exactly-once by key: each `ctx.run(..., key=...)` is idempotent; on resume, completed keys return recorded results.
- Checkpointed at task boundaries with a single writer fsyncing `events.jsonl`; each event records its `start_offset`.
- Replay on resume: strategy code re-executes deterministically; completed runs are short-circuited; pending keys are resumed or reattached by key.

Key reuse contract:

- Same key + same fingerprint → idempotent: return the recorded result.
- Same key + different fingerprint → hard error `KeyConflictDifferentFingerprint`.

## 4.4 Built-in Strategies

The system provides reference patterns implemented purely with the durable `ctx.run` primitive:

**Single Task**: Schedule exactly one `ctx.run` and await its result. Baseline for tasks where a single attempt is sufficient.

**Best-of-N (inline scoring)**: Generate N candidates with `ctx.run`. For each candidate, run a reviewer `ctx.run` that returns JSON `{score:0..10, rationale:string}`. If the JSON is invalid, do one repair attempt with a stricter “return ONLY JSON” prompt. Exclude candidates that still fail schema validation. Pick the max score among valid results and return that candidate’s result.

**Iterative Refinement**: Generate an initial solution, run a scoring/review `ctx.run` for feedback, then schedule an improvement `ctx.run` using the prior `session_id`. Repeat for a fixed number of iterations (default 3).

These patterns are expressed directly in strategy code using keys like `gen/<i>`, `score/<instance_id>/attempt-1`, and `improve/iter-<n>`. There are no special strategy types; everything reduces to idempotent `ctx.run` calls plus deterministic selection logic.

## 4.5 Parallel Execution Engine

Managing concurrent tasks requires careful resource control:

**Resource Pool**: Maximum parallel tasks configurable (default adaptive). The default limit is `max(2, min(20, floor(host_cpu / runner.container_cpu)))`. This prevents host oversubscription while allowing parallelism. When at capacity, new task requests queue until a slot opens. A warning MUST be logged if the configured value oversubscribes the host (i.e., `max_parallel_tasks * runner.container_cpu > host_cpu`).

**FIFO Scheduling**: Simple first-in-first-out queue ensures fairness. No complex priority systems - strategies that submit first get resources first. This predictability makes debugging easier.

**Task Scheduling**: When launching tasks, the engine:

 - Generates container names using pattern `orchestrator_{run_id}_s{sidx}_k{short8(key)}` (derived from the durable task key) and computes them before scheduling; container_name MUST NOT change thereafter
- Uses deterministic branch names derived from the durable task key (orchestration provides the final name)
- Persists `strategy_index` label on containers with the 1-based `sidx` value for easier grouping and debugging
- Tracks task-to-container mapping for monitoring and cleanup
- Manages multiple background executor tasks (configurable count) for true parallel execution
- Validates disk space before starting tasks (20GB minimum)

**Execution Tracking**: Each task tracked from schedule to completion. Strategies can wait for specific tasks or groups. Essential for patterns like "wait for all 5 to complete, then pick best".

**Resource Cleanup**: When tasks complete, their slots immediately return to the pool. Failed tasks don't block resources. Ensures maximum utilization without manual intervention.

The engine is intentionally simple. We resisted adding complex features like priority queues or dynamic scaling because FIFO + fixed pool size handles real workloads well. The engine handles the mechanical aspects of parallel execution while strategies provide the intelligence about what to run and when.

## 4.6 State Management

State tracking serves two purposes: enabling UI displays and crash recovery:

**In-Memory State**: Primary state lives in memory as simple data structures:

- Running tasks with their status and progress
- Active strategies with their configuration
- Completed results awaiting processing
- Aggregate metrics (total cost, duration, token usage)

**Event Sourcing**: Every state change emits an event to `events.jsonl`. The current state can be reconstructed by replaying events from this file. This pattern enables resumability - after a crash, replay events to restore state.

**State Persistence Implementation**: State is persisted through two mechanisms:

- `events.jsonl`: Append-only log containing every state change with monotonic byte offsets. A single writer computes the `start_offset`, writes the record, flushes, and fsyncs.
- `state.json`: Periodic snapshot (every 30 seconds) containing full state and the `last_event_start_offset` (the byte position of the last applied event's start). Snapshots MUST persist, per instance: `state`, `started_at`, `completed_at`, `interrupted_at`, `branch_name`, `container_name`, and `session_id` to enable resumability. Written atomically using temp file + rename

Recovery process:

1. Load `state.json` if it exists (includes `last_event_start_offset`)
2. Read `events.jsonl` from the saved `last_event_start_offset`
3. Apply events to reconstruct current state
4. Handle cases where state directory structure changed (e.g., run_ prefix removal)

**Event Bus Features**: The event system includes:

- Ring buffer implementation (default 10,000 events) to prevent memory issues
- File watching for external processes via the public `subscribe()` API
- Event filtering by type, timestamp, and run_id
- Monotonic offset tracking for reliable position management (using `start_offset` semantics)

**Atomic Snapshots**: The `get_current_state()` function returns consistent snapshots. No partial updates or race conditions. UI can poll this API for smooth updates without event subscription complexity.

This approach balances simplicity with reliability. Memory is fast for normal operation, while file-based persistence enables recovery when needed. The combination of event log and periodic snapshots ensures quick recovery without replaying the entire event history.

**Idempotent Recovery**

- `state.json` MUST include `last_event_start_offset` (byte position of the last applied event's start).
- Recovery MUST replay only events after `last_event_start_offset`.
- Event application MUST be idempotent: re-applying a transition already reflected in the snapshot MUST NOT double count aggregate metrics (e.g., completed/failed counters).
- Aggregate counters MUST be derived from guarded state transitions (increment only when moving from a non-terminal to a terminal state).
- Internal runner events may include `session_id` updates used for in-memory tracking. These are not part of the public `events.jsonl` contract. The public event set remains `task.*` and `strategy.*` only.

## 4.7 Git Synchronization

Parallel tasks work in complete isolation, requiring no coordination during execution. The synchronization happens after containers complete through a simple branch import process:

**Branch Import Process**: After each container exits, the Instance Runner performs the git import as the final atomic step of `run_instance()`. This uses git's filesystem fetch capability to import the isolated workspace into the host repository. The runner MUST serialize imports per repository using a cross‑platform, process‑internal file lock located under the actual git directory (resolve with `git rev-parse --git-dir`), e.g., `<git-dir>/.orc_import.lock` (fcntl on POSIX, msvcrt on Windows, or a cross‑platform library), keyed by the canonical realpath of the repository to avoid aliasing. Do not rely on shell `flock`. The import respects the per‑task `import_policy` ("auto" | "never" | "always"). The import is atomic — either all commits transfer successfully or none do. Collision policy is enforced while holding the import lock.

**Branch Naming (normative)**: `branch_name = f"{strategy_name}_{run_id}_k{short8(durable_key)}"` where `short8(x) = sha256(x).hex()[:8]`. Using `run_id` (not a timestamp) guarantees stability across resumes of the same run. Runner never invents names; it receives the final branch name from orchestration and creates that branch during import.

**Clean Separation**: The synchronization mechanism maintains architectural boundaries:

- Orchestration owns branch naming policy
- Instance Runner performs the mechanical git fetch operation
- Isolated workspaces eliminate any possibility of conflicts
- Neither component needs to understand the other's logic

**Import Operation**: The actual branch import preserves the original base branch reference and MUST perform an idempotent pre-check to avoid duplicate imports across crashes:

```bash
# Performed by Instance Runner as final step
cd /path/to/host/repo
BASE_BRANCH=$(cat /tmp/orchestrator/${run_id}/k_${KHASH}/.git/BASE_BRANCH)

# Idempotent import pre-check (handles crash window)
# (Illustrative shell; reference implementation performs these checks in Python under the process-level lock.)
if git show-ref --verify --quiet "refs/heads/${branch_name}"; then
  WS_HEAD=$(git --git-dir="/tmp/orchestrator/${run_id}/k_${KHASH}/.git" rev-parse HEAD)
  BR_HEAD=$(git rev-parse "refs/heads/${branch_name}")
  if [ "$WS_HEAD" = "$BR_HEAD" ]; then
    echo "Branch already matches workspace HEAD; treating as completed."
    # Return success to orchestration; it will emit task.completed
    exit 0
  fi
fi

if [ "${import_conflict_policy:-fail}" = "overwrite" ]; then
  git fetch /tmp/orchestrator/${run_id}/k_${KHASH} +HEAD:${branch_name}
else
  git fetch /tmp/orchestrator/${run_id}/k_${KHASH} HEAD:${branch_name}
fi
```

This operation runs under the per-repo import lock to serialize updates to refs and packs. Unique branch names avoid cross-branch conflicts; the lock prevents ref/pack races.

The idempotent pre-check ("does the branch already equal the workspace HEAD?") **and** the subsequent `git fetch` MUST both occur while holding the same per-repo import lock.

**Import Policies & Edge Cases**:

- Empty changes (auto): with `skip_empty_import=true` (default) and `import_policy="auto"`, if there are no commits relative to the base (detected via `BASE_COMMIT..HEAD`), no branch is created. The task result sets `artifact.has_changes=false`, `artifact.branch_final=null`, and records the planned branch name in `artifact.branch_planned` for reporting.
- Empty changes (always): with `import_policy="always"`, a branch is created even when there are zero commits; it points at the base commit.
- Never import: with `import_policy="never"`, no branch is created regardless of changes. The result MUST set `artifact.has_changes=false`, `artifact.branch_final=null`, and `artifact.branch_planned` MUST carry the planned branch name for traceability. Strategies MUST pick a real base for subsequent tasks (e.g., the candidate’s imported branch if `has_changes=true`, else the original `base_branch`).
- Name collisions: controlled by `import_conflict_policy = "fail" | "overwrite" | "suffix"` (default `"fail"`). For `"suffix"`, `_2`, `_3`, … are appended atomically while holding the import lock. For `"overwrite"`, a forced ref update is used.
- Failures: if git fetch fails (disk space/corruption), the instance is marked failed with details. The isolated workspace is preserved for debugging.

**Suffix idempotency with provenance**: if policy is `"suffix"` and the base `branch_name` does not exist, the runner MUST treat any existing branch whose name matches `^${branch_name}(_[0-9]+)?$` and whose HEAD equals the workspace HEAD as "already imported" only if the provenance marker matches the same `task_key`. 

After import, attach a provenance marker to the commit using git notes:

```bash
git notes --ref=orchestrator add -m 'task_key=<fully_qualified_key>; run_id=<run_id>' <WS_HEAD>
```

Idempotency rule: treat as already imported only if both (a) HEADs match and (b) the provenance note on that commit contains the same `task_key`.

Determinism note: `import_conflict_policy="suffix"` breaks strict determinism of branch names across resumes and is intended for manual/experimental runs. For reproducibility, the recommended default is `"fail"`.

**No Remote Operations**: By design, containers never push to any remote. All git operations are local to the host machine. If remote synchronization is needed, it happens as a separate post-processing step after all tasks complete, completely decoupled from instance execution.

**Workspace Cleanup**: After successful import, the temporary workspace is immediately deleted to free disk space. Failed instances retain their workspaces according to the retention policy, allowing post-mortem analysis of what the AI agent attempted.

Workspaces are not used for resume; they exist solely for debugging and import. Resume relies on persistent session volumes and container state.

**Performance Characteristics**: The import approach scales perfectly:

- No coordination needed between parallel tasks
- Import operations take milliseconds to low seconds depending on changeset size
- No network latency since everything is filesystem-based
- Default limit of 20 parallel tasks ensures system stability while maintaining high throughput

**Audit Trail**: Each import operation is logged with:

- Source workspace path
- Target branch name
- Commit count and size
- Success/failure status
- Timestamp and duration

This provides complete visibility into what each AI agent produced and when it was integrated.

The git synchronization is minimal by design. By preparing isolated environments beforehand and importing results afterward under a simple file lock, we eliminate complex coordination. Each instance truly runs in its own universe; the orchestrator collects results deterministically.

**Provenance**: Do not mutate the imported branch tip after import. Prefer one of the following:

- Write provenance as a sidecar file in `results/<run_id>/strategy_output/` (recommended), or
- Attach provenance via git notes or a lightweight tag referencing the imported commit.

If provenance must live in-repo history, write it into the workspace before import so it becomes part of the agent’s own commit(s). Avoid adding commits after import to preserve the “import HEAD” auditability guarantee.

## 4.8 (Removed)

The prior HTTP server and REST endpoints are removed. Monitoring relies solely on local `events.jsonl` and `state.json`. The public event set is canonicalized to `task.*` and `strategy.*` only.

## 4.9 Resume Capabilities

Resumption produces identical final outcomes with no duplicate work using durable task keys:

**Seamless Resume Algorithm**:

1. Load last `state.json` snapshot (if any).
2. Replay `events.jsonl` from the saved `last_event_start_offset` to rebuild: completed tasks map (`key → result`), pending tasks (`scheduled` without `completed/failed`), and strategy completion state.
3. Reattach pending tasks by key:
   - Find the container via labels for `run_id/strategy_execution_id/key`.
   - If running → reattach streams.
   - If stopped but a session-group volume exists (see 3.4 Session Groups) → start container and continue using `resume_session_id`.
   - Else → start fresh with the same key.
4. Re-enter the strategy function deterministically: completed `ctx.run` calls return recorded results immediately; execution advances to the first unfinished run.
5. Continue to completion. Container and branch names derive from the durable key, preventing duplicates.

**Crash Inference**: On resume, any task that is `RUNNING` in `state.json` and has no terminal event at or after `last_event_start_offset` MUST be marked `INTERRUPTED` before resuming.

**State Recovery**: The system handles various failure scenarios:

- Graceful shutdown: Full state preserved, seamless resume
- Crash during execution: State recovered from last snapshot + event replay
- Partial container failure: Failed instances marked, others continue

## 4.10 Results Export

After run completion, the orchestrator exports comprehensive results:

**Directory Structure**:

```
results/
  run_20250114_123456/
    summary.json       # Overview with metrics and branch list
    branches.txt       # Simple list of created branches
    metrics.csv        # Time-series metrics data
    strategy_output/   # Strategy-specific data
      best_branch.txt  # For BestOfN strategy
      scores.json      # Scoring details
```

**Summary Format**: The `summary.json` includes:

- Run metadata (ID, strategy, configuration)
- Complete list of created branches with status
- Aggregate metrics (total cost, duration, tokens)
- Task-level details and outcomes
- Strategy-specific results

**Metrics Export**: Time-series data in CSV format:

- Timestamp, task count, cost accumulation
- Token usage over time
- Task state transitions
- Useful for cost analysis and optimization

## 4.11 Resource Management

**Startup Validation**:

- Check 20GB free disk space before starting
- Verify Docker daemon accessibility
- Clean up orphaned containers from previous runs (owned by orchestration)
- Validate git repository and base branch

**Container Cleanup**: On startup, orchestration owns orphan cleanup by run namespace:

1. Lists all containers matching naming pattern
2. Checks against known run IDs
3. Removes orphaned containers older than retention period (based on `orchestrator.last_active_ts` label; falls back to Docker CreatedAt if absent)
4. Logs cleanup actions for audit trail

Runner cleanup is limited to its own container on task failure; it MUST NOT scan for or delete unrelated containers to avoid races with orchestration.

**Parallel Shutdown**: On Ctrl+C or error:

1. Set shutdown flag to prevent new tasks
2. Cancel all pending task operations
3. Stop all running containers in parallel
4. Save final state snapshot
5. Clean up temporary resources

The parallel container stopping significantly improves shutdown speed when many instances are running.

# 5. TUI Component

## 5.1 Overview and Responsibilities

The TUI is how users experience the orchestrator. While the underlying system manages complex parallel executions, the TUI's job is to make that complexity understandable at a glance. Built with Rich, it provides a real-time dashboard during execution and clean CLI output for both interactive and non-interactive use.

Core responsibilities:

 - Display real-time progress of all tasks and strategies
 - Adapt visualization based on scale (5 vs 500 tasks)
- Provide CLI interface for launching runs
- Show aggregate metrics and individual task details
- Stream important events to console in non-TUI mode
- Present final results and branch information

The TUI knows nothing about how tasks run or how strategies work - it simply subscribes to events and queries state. This separation means you could replace it entirely with a web UI without touching orchestration logic.

**Multi-UI Support**: Multiple UIs can monitor the same run by tailing the shared `events.jsonl` file and periodically reading `state.json` snapshots. No HTTP server is provided.

## 5.2 Display Architecture

The TUI uses a hybrid approach combining event streams with periodic state polling for both responsiveness and reliability:

**Event Handling**: The TUI watches `events.jsonl` for new events using file notification APIs (inotify/kqueue). As new events are appended to the file, the TUI reads and processes them, updating its internal state immediately. However, events don't trigger renders directly. Whether 1 event or 100 events arrive, they simply update the internal data structures. This prevents display flooding when hundreds of tasks generate rapid events.

**Fixed Render Loop**: A separate render loop runs at exactly 10Hz (every 100ms), reading the current internal state and updating the display. This decoupling is crucial - event processing and display rendering operate independently. Users see smooth, consistent updates regardless of event volume.

**State Reconciliation**: Every 3 seconds, the TUI reads the latest `state.json` snapshot to reconcile any drift. This isn't the primary update mechanism—just a safety net catching missed events or recovering from disconnections.

**Buffer Management**: Events update a simple in-memory representation:

- Task map keyed by ID with current status
- Strategy progress counters
- Aggregate metrics accumulation
- Last-updated timestamps for staleness detection

**Why This Architecture**: Pure event-driven displays suffer from flooding and missed events. Pure polling feels sluggish and wastes resources. This hybrid leverages events for responsiveness while maintaining predictable render performance and reliability. The 10Hz update rate is fast enough to feel real-time but sustainable even with hundreds of tasks.

The separation of concerns is clean: events flow from file watching, polling ensures correctness, rendering happens on its own schedule. This pattern emerged from experience - it handles both the 5-instance demo and the 500-instance production run equally well.

## 5.3 Layout Design

The dashboard uses a three-zone layout maximizing information density:

**Header Zone**: Single-line header showing run-level information:

- Run ID with emoji indicator
- Strategy name and configuration
- Model being used
- Task counts (running/completed/failed)
- Total runtime

**Dashboard Zone**: Dynamic grid adapting to task count:

- Each strategy gets a bordered section
- Tasks within strategies shown as cards
- Card size varies based on total count (detailed → compact → minimal)
- Visual grouping makes strategy boundaries clear

**Footer Zone**: Fixed 2-line footer with aggregate metrics:

- Total runtime, accumulated cost, token usage
- Task counts (running/completed/failed)
- Critical warnings or errors

The layout prioritizes current activity. Completed tasks fade visually while active ones draw attention. This focus on "what needs attention now" helps users manage large runs.

## 5.4 Adaptive Display

The display intelligently adapts to task count, showing maximum useful information without clutter:

**Detailed Mode (1-10 tasks)**:

- Large cards with full Claude Code activity feed
- Live progress bars based on task completion
- Token usage breakdown (input/output/total)
- Real-time cost accumulation
- Full error messages if failed

**Compact Mode (11-50 tasks)**:

- Medium cards with current status line only
- Simplified progress indicator
- Cost and runtime prominently displayed
- Status shown as emoji (🔄 running, ✅ complete, ❌ failed)

**Dense Mode (50+ tasks)**:

- Minimal cards in tight grid
- Just ID, status emoji, cost, and runtime
- Color coding for quick status scanning
- Strategy-level progress more prominent
- Summary stats matter more than individuals

The adaptation is automatic but overridable. Users can force detail level if monitoring specific tasks closely. This flexibility handles both overview monitoring and detailed debugging.

## 5.5 CLI Interface

The CLI provides the primary interface for launching and controlling runs:

**Command Structure**: Clean, intuitive commands following Unix conventions:

```bash
orchestrator "your prompt" --strategy best-of-n --runs 3 -S n=5
orchestrator --resume <run-id>
orchestrator --list-runs
orchestrator --clean-containers
```

**Output Modes**:

- Default: Full TUI dashboard
- `--no-tui`: Stream events to console with clean formatting
- `--json`: Machine-readable output for scripting

**Cleanup Commands**:

- `--clean-containers`: Removes stopped or orphaned containers labeled `orchestrator=true` older than the configured retention. Running containers are not removed. Supports `--dry-run` to list targets without deleting and `--force` to bypass prompts.
- `--quiet`: Only show final results

**Strategy Configuration**:

- `--runs`: Number of parallel strategy executions within a single run ID (one run folder; N executions). An alias `--executions` MAY be provided for clarity.
- `-S key=value`: Set strategy-specific parameters (repeatable)
- `--config`: Load configuration from YAML or JSON file

Examples:

```bash
# Single execution with parameters
orchestrator "fix bug" --strategy best-of-n -S n=3

# Multiple parallel executions
orchestrator "refactor" --strategy best-of-n --runs 5 -S n=3

# Using config file
orchestrator "build feature" --config prod-strategy.yaml

# Override config values
orchestrator "optimize" --config base.yaml --runs 10 -S timeout=600

# Force branch creation even with no changes
orchestrator "analyze code" -S import_policy=always
```

**Escaping and Quoting**:

- Use single quotes for prompts containing spaces or special characters
- For prompts with single quotes, use double quotes or backslash escaping
- For complex multi-line prompts, use `--config` file instead
- Strategy parameters (`-S`) follow standard shell parsing rules

Examples:

```bash
# Simple prompt
orchestrator "implement hello world"

# Prompt with quotes
orchestrator 'implement "Hello, World!" program'

# Complex prompt with special characters
orchestrator "implement function that returns \"it's working!\""

# Multi-line or very complex prompts - use config file
orchestrator --config complex-prompt.yaml
```

**Non-TUI Output**: When TUI is disabled, events stream to console with:

- Colored prefixes for event types ([STARTED], [COMPLETED], [FAILED])
- Instance IDs for correlation
- Key information only - no verbose logging
- Clean formatting without timestamp clutter

**Configuration**: CLI arguments take precedence over all other config sources. Complex strategies can be configured via `--config` with YAML or JSON files for repeatability. Environment variables provide defaults without cluttering commands.

The CLI design philosophy: make common tasks simple, provide power when needed. Running basic strategies requires minimal flags while complex workflows remain possible.

## 5.6 Standalone TUI Operation

The TUI can run independently of the orchestrator for monitoring:

**Standalone Mode**: Launch with `python -m src.tui` to monitor existing runs:

```bash
# Monitor a specific run
python -m src.tui --run-id run_20250114_123456
```

**Use Cases**:

- Monitor runs from different terminals
- Debug event processing separately
- Test display layouts without running instances
- Multiple team members viewing same run

## 5.7 Implementation Notes

**Missing Features from Specification**:

- Task-based progress tracking (shows time-based progress instead)
- Lines of code changed display (not implemented)
- Global progress bar in header (uses counts instead)

These were omitted as they required deeper Claude Code integration than available through the event stream.

# 6. Cross-Cutting Concerns

## 6.1 Configuration System

Configuration follows a strict precedence hierarchy, making defaults sensible while allowing complete control when needed:

**Precedence Order** (highest to lowest):

1. CLI arguments - Direct control for one-off runs
2. Environment variables - Deployment-specific settings
3. `.env` file - Local development convenience
4. `orchestrator.yaml` - Shared team configuration
5. Built-in defaults - Sensible out-of-box experience

**Authentication Modes**: The orchestrator supports two Claude Code authentication methods:

- **Subscription mode** (default): Uses `CLAUDE_CODE_OAUTH_TOKEN` for users with Claude Pro subscriptions. Significantly cheaper for heavy usage.
- **API mode**: Uses `ANTHROPIC_API_KEY` for usage-based billing or custom LLMs.

Mode selection follows this logic:

1. If `--mode api` specified, use API key
2. If OAuth token present, use subscription mode
3. If only API key present, use API mode
4. Otherwise, error with clear message

**Custom LLM Support**: When using API mode, `ANTHROPIC_BASE_URL` enables routing to custom endpoints. This allows local LLMs, proxies, or alternative providers that implement Claude's API. The Instance Runner passes this through to Claude Code without modification.

**Configuration Namespacing**: Each component reads its own config section:

- `runner.*` - Instance Runner settings
- `orchestration.*` - Parallelism limits, timeouts
- `tui.*` - Display preferences
- `logging.*` - Output paths, verbosity

This separation ensures components remain independent. Adding TUI color schemes doesn't affect runner configuration.

**Environment Variables**: The system supports extensive environment variables (27 beyond core auth):

```bash
# Core paths
ORCHESTRATOR_STATE_DIR=/custom/state/path
ORCHESTRATOR_LOGS_DIR=/custom/logs/path

# Debug settings
ORCHESTRATOR_DEBUG=true

# Strategy parameters
ORCHESTRATOR_STRATEGY__BEST_OF_N__N=5

# Runner settings
ORCHESTRATOR_RUNNER__ISOLATION_MODE=full_copy  # shared_objects requires explicit env to enable

# TUI settings
ORCHESTRATOR_TUI__REFRESH_RATE=100
ORCHESTRATOR_TUI__FORCE_DISPLAY_MODE=detailed

# Model configuration
ORCHESTRATOR_DEFAULT_MODEL=sonnet
```

**Configuration Merging**: Complex deep merge with recursive dictionary merging for nested configurations. Arrays are replaced, not merged.

### 6.1.1 Default Values Catalog (normative)

Defaults used by orchestration and runner. Fingerprinting canonicalization treats missing fields as these explicit defaults, and **removes keys with `null` values prior to hashing** (i.e., `null` ≡ missing).

- model: `sonnet`
- import_policy: `auto`
- import_conflict_policy: `fail`
- skip_empty_import: `true`
- max_parallel_tasks: `"auto"`  # auto => max(2, min(20, floor(host_cpu / runner.container_cpu)))
- runner.max_turns: unset (tool decides)
- runner.session_volume_scope: `run`
- runner.container_timeout_seconds: `3600`
- runner.container_cpu: `2`
- runner.container_memory: `4g`
- runner.reuse_container: `true`
- runner.finalize: `true`
 - runner.review_workspace_mode: `ro`  # was rw

Events configuration:
- events.flush_policy.mode: `"interval"`
- events.flush_policy.interval_ms: `50`
- events.flush_policy.max_batch: `256`
- events.max_final_message_bytes: `65536`

All defaults are validated and logged at run start. Fingerprint computation (RFC 8785 JCS) must apply these defaults before hashing.

## 6.2 Logging System

Logging serves both debugging and audit purposes without cluttering the user experience:

**Structured JSON Logs**: All components emit structured logs with consistent fields:

- `timestamp` - ISO 8601 with milliseconds
- `component` - Origin (runner/orchestration/tui)
- `level` - Standard levels (debug/info/warn/error)
- `run_id` - Correlation across entire run
- `instance_id` - When applicable
- `message` - Human-readable description

**Message Sanitization**: All `message` fields in public events MUST pass through the sanitizer (see §6.3).
- Additional context fields as relevant

**File Organization**: Logs are separated by component and run:

```
logs/
  run_20250114_123456/
    orchestration.jsonl
    runner.jsonl
    tui.jsonl
    events.jsonl  # Complete event stream
```

**Console Output**: Clean, purposeful console output:

- TUI mode: Only errors and warnings
- No-TUI mode: Key events with colored prefixes
- Debug mode: Full log stream to stderr
- Never mix logs with TUI display

**Rotation and Cleanup**: Simple time-based retention - logs older than 7 days are deleted on startup. No complex rotation schemes. Users who need permanent logs can configure longer retention or disable cleanup.

The logging philosophy: capture everything to files for debugging, show users only what matters.

**Advanced Logging Features**:

- Component-specific log file routing (runner.log, orchestration.log, tui.log)
- JSON formatter with serialization fallback for complex objects
- Dynamic logger configuration based on debug mode
- Headless mode console output integration
- File size-based rotation with configurable backup count (applies to component logs only; never to `events.jsonl`)
- `events.jsonl` is never rotated during an active run; rotation/archival may occur only after the run completes
- Async rotation tasks to prevent blocking
  
Cost estimation configuration:

- When a plugin cannot report cost but can report tokens, costs are estimated from `pricing.*` config keys (USD):
  - `pricing.anthropic.sonnet.input_per_1k_usd`
  - `pricing.anthropic.sonnet.output_per_1k_usd`
  - (similarly for other models)
- Orchestration performs the estimate and reports in results and TUI.


## 6.3 Security and Isolation

Security focuses on protecting the host system and preventing instance interference:

**Container Isolation**: Each instance runs in a Docker container with:

- No privileged access
- Three volume mounts only: isolated workspace, /home/node persistent volume, and /tmp tmpfs
- Entire container filesystem read-only except the mounted volumes
- Container-specific network namespace with outbound internet access (configurable via `runner.network_egress = online|offline|proxy`; default `online`)
- Resource limits preventing system exhaustion
- Non-root user execution

**API Key Handling**: Authentication tokens are:

- Never logged or displayed
- Passed via environment variables to containers
- Validated on startup with clear errors
- Support for both OAuth tokens (subscription) and API keys
- Secrets MUST be redacted from logs and error traces; environment dumps are forbidden in runner events

**Centralized Redaction**: A sanitizer with SECRET_PATTERNS is used by Orchestration before writing any public event:

```python
SECRET_PATTERNS = [
    r'(?i)(api|token|oauth|secret)[-_ ]?(key|token)\s*[:=]\s*[\w\-]{8,}',
    r'sk-[A-Za-z0-9]{20,}',
]
def sanitize_public(obj):
    def _scrub(s):
        for pat in SECRET_PATTERNS:
            s = re.sub(pat, '[REDACTED]', s)
        return s
    # walk dict/str recursively and scrub strings
```

**File System Boundaries**: Instances cannot:

- Access the original host repository or any host paths except their isolated workspace at `/tmp/orchestrator/<run>/k_<short_hash(key)>`
- See or modify the original repository
- Read other instances' working directories
- Escape their container boundaries
- Access any system paths outside their designated mounts

**Git Safety**: The isolated clone approach ensures:

- Host repository never exposed to containers
- Each task works in a completely separate git repository
- No shared git objects or metadata between instances
- Physical file copying with `--no-hardlinks` prevents inode-level interference
- Branch imports happen after container exits, eliminating concurrent access

These measures provide reasonable security without enterprise complexity. We're protecting against accidents and basic isolation violations, not nation-state attackers. The complete workspace isolation is the key defense - containers literally cannot access the host repository or other instances' work.

## 6.4 Resource Management

Resource management keeps things simple and functional:

**Parallel Task Limit**: Single configuration value `max_parallel_tasks` (default adaptive: `max(2, min(20, floor(host_cpu / runner.container_cpu)))`). Prevents system overload and API rate limit issues. No complex scheduling or priorities - just a simple pool. (`max_parallel_instances` is supported as a deprecated alias and MUST log a one-time deprecation warning if used.) A warning MUST be logged if the configured value oversubscribes the host.

**Container Resources**: Fixed defaults that work well:

- 2 CPUs per container
- 4GB memory per container
- No disk quotas (rely on system capacity)

Override only if needed via `container_cpu` and `container_memory` settings. Most users never touch these.

**Cost Tracking**: Simple accumulation from Claude Code's reported costs. No budgets or limits for now - users can monitor costs in real-time and cancel if needed. Complex budget enforcement can come later if demanded.

**Timeouts**: Two-level timeout system:

- Tool-level turn/time limits as supported by the plugin (optional `runner.max_turns`; default unset)
- Container-level timeout (default 1 hour) as a fallback safety net

This dual approach catches both API-level runaway usage and container-level hangs. Both are configurable but defaults work for most cases.

The philosophy: provide essential limits that prevent system failure, avoid complex resource management that users won't use.

## 6.5 Data Persistence

Persistence focuses on essential data for resumability and results:

**Event Log**: Append-only `events.jsonl` with a single-writer model:

- One writer task computes `start_offset`, appends, and flushes according to the configured flush policy (buffers events and writes+fsyncs every `interval_ms` or `max_batch`, whichever comes first). The writer MUST acquire an OS-level exclusive lock on `logs/<run_id>/events.jsonl.lock` for the duration of the run using the stale lock detection algorithm below; if an active lock is already held, startup MUST fail fast to prevent multiple writers. Crash semantics: state snapshots carry `last_event_start_offset`; on crash, replay up to the last fsynced batch.

**Stale Lock Detection**: The lockfile format includes `{ pid, hostname, started_at_iso }` metadata. Acquisition algorithm:

```python
path = logs/run_id/events.jsonl.lock
if exists(path):
    meta = json.load(open(path))
    if not process_is_alive(meta["pid"], meta["hostname"]):
        # stale; replace
        os.remove(path)
    else:
        fail("Another writer is active")
# now acquire OS lock (fcntl/msvcrt), write metadata, fsync
```

The `process_is_alive` function uses `os.kill(pid, 0)` on POSIX (Linux/macOS) and `OpenProcess` on Windows.

**Filesystem Requirements**: The logs/state directories MUST be on a local filesystem; network filesystems are unsupported.

- Enables state reconstruction after crashes and provides an audit trail.
- Simple UTF-8 JSONL for easy parsing; partial final lines are ignored by readers until newline.
- Event envelope fields: `{id, type, ts, run_id, strategy_execution_id, key?, start_offset, payload}`.
- Task lifecycle events: `task.scheduled`, `task.started`, `task.completed`, `task.failed`, `task.interrupted`, plus `strategy.started/strategy.completed`. See the “Minimum payload fields” block at the end of this document for the canonical schema.
 - Readers MUST resume from a durable offset (`last_event_start_offset`) rather than attempting to consume an unterminated final line. The writer-only lock and snapshot offset guarantee a safe resume point.

**State Snapshots**: Periodic `state.json` dumps of current state (primary snapshot is authoritative; the logs copy is best‑effort only):

- Every 30 seconds during execution.
- Contains full state, `last_event_start_offset`, and task index (completed/pending).
- Includes per-task: `state`, `started_at`, `completed_at`, `interrupted_at`, `branch_name`, `container_name`, `session_id`, `session_group_key`.
- Speeds up recovery by replaying only events after the `last_event_start_offset`.

**File Organization**: Data is organized per-run (primary state dir + optional duplicate under logs):

```
orchestrator_state/
  run_20250114_123456/
    state.json         # Primary state snapshot with last_event_start_offset

logs/
  run_20250114_123456/
    events.jsonl       # Events for this run only
    state.json         # Optional duplicate snapshot
    orchestration.log  # Component logs
    runner.log
    tui.log

results/
  run_20250114_123456/
    summary.json       # Overview with metrics
    branches.txt       # List of created branches
    strategy_output/   # Strategy-specific data
```

**Event Ordering**: Critical events MUST be durably written before cleanup operations:

- Runner completion log entries include workspace paths; public `task.completed` events do not carry paths.
- Writer flushes and fsyncs before workspace deletion.
- Ensures audit trail remains complete for reconstruction.

**Crash Tolerance & Sessions**: Readers tolerate truncated last lines, resuming from the `last_event_start_offset`. Claude Code sessions are preserved via container+volume naming. On resume, orchestration reattaches by durable task key without duplicating containers or branches.

This approach balances durability with simplicity. No databases to manage, no complex schemas to migrate. Just files that are easy to inspect, backup, and understand.

## 6.6 Platform Support

The orchestrator includes extensive platform-specific support:

**Windows Subsystem for Linux (WSL)**: Automatic detection and path conversion:

- Detects WSL via /proc/version inspection
- Converts Windows paths to WSL format (C:\\Users → /mnt/c/Users)
- Provides WSL-specific Docker setup recommendations
- Handles Docker Desktop vs native Docker in WSL

**Docker Path Normalization**: Platform-aware path handling:

- Windows: Converts to Docker Desktop format (/c/Users/...)
- Unix/WSL: Uses standard paths
- Automatic detection of Docker environment

**SELinux Support**: Container security on systems with SELinux:

- Detects SELinux via kernel checks
- Applies `:z` volume mount flags when needed
- Ensures containers can access mounted volumes

**Platform Recommendations**: On startup, provides platform-specific advice:

- WSL users: Docker backend configuration tips
- Windows users: WSL2 installation suggestions
- All platforms: Temp directory accessibility checks

**Cross-Platform Temp Directories**:

- Windows: Uses %TEMP% or C:\\Temp
- Unix: Uses /tmp
- Validates write permissions on startup

# 7. Operations

## 7.1 Starting a Run

Starting a run is designed to be frictionless while catching common errors early:

**Basic Invocation**: The simplest case requires just a prompt:

```bash
orchestrator "implement user authentication"
```

This uses defaults: SimpleStrategy, main branch, Sonnet model. Perfect for quick tasks.

**Model Selection**: Choose specific models for different tasks:

```bash
orchestrator "complex task" --model opus
orchestrator "standard task"  # Uses default: sonnet
```

**Strategy Selection**: Real power comes from choosing strategies:

```bash
orchestrator "refactor database layer" --strategy best-of-n -S n=5
```

Strategy-specific parameters are passed with `-S` flags. For multiple parameters:

```bash
orchestrator "build API" --strategy best-of-n --runs 3 -S n=5
```

Parameters are validated immediately with clear errors for missing required options.

**Pre-flight Checks**: Before scheduling any tasks, the orchestrator verifies:

- Docker daemon is running and accessible
- Repository exists and the requested `base_branch` exists and is cleanly check‑outable (warn on dirty working tree by default, which is safe to proceed since imports touch refs, not your working tree; use `--require-clean-wt` to enforce cleanliness)
- Authentication is configured (OAuth token or API key)
- Required tools are available in containers
- Sufficient disk space for logs and results

These checks take milliseconds but prevent frustrating failures mid-run. Better to fail fast with helpful errors.

**Run ID Generation**: Each run gets a unique ID: `run_20250114_123456`. This ID correlates everything - logs, results, branches, resume capability. Timestamp-based for natural sorting and human readability.

**Configuration Loading**: Settings are resolved in precedence order and logged at startup. Users see exactly what configuration is active. No guessing about which config file was loaded or which environment variable took precedence.

The startup sequence is deliberately verbose in non-TUI mode, showing each initialization step. This transparency helps debugging configuration issues and builds confidence the run is properly set up.

## 7.2 Monitoring Progress

Monitoring is designed to answer the key question: "Is everything going as expected?"

**TUI Dashboard**: The default monitoring experience shows:

- Real-time task status with visual grouping by strategy
- Live cost accumulation (critical for budget awareness)
- Progress indicators based on Claude's task completion
- Clear failure indication with error snippets

The dashboard adapts to run scale automatically. Users don't configure view modes - it just works whether running 3 tasks or 300.

**Non-TUI Monitoring**: When TUI is disabled, progress streams to console (showing both task hash and instance id):

Identifier formats:

- `k<8hex>` = `short_hash(key)` = first 8 hex chars of SHA‑256 of the fully-qualified durable key
- `inst-<5hex>` = first 5 hex chars of the stable `instance_id`
- Log prefix format MUST be: `<k8>/<inst5>: <message>`

```
k7aa12cc/inst-7a3f2: Started → feat_auth_1
k7aa12cc/inst-7a3f2: Implementing authentication module...
k7aa12cc/inst-7a3f2: Created auth/jwt.py with JWT handler
k7aa12cc/inst-7a3f2: Completed ✓ 3m 24s • $0.42 • 2.5k tokens

k19bb44a/inst-8b4c1: Started → reviewing feat_auth_1  
k19bb44a/inst-8b4c1: Analyzing implementation quality...
k19bb44a/inst-8b4c1: Strategy output → score=8.5, feedback="Good error handling"
k19bb44a/inst-8b4c1: Completed ✓ 45s • $0.12 • 0.8k tokens

k2f88e01/inst-9d5e3: Failed ✗ Timeout after 60m • $2.41
```

Clean prefixes, instance correlation, key metrics. No timestamp spam or debug noise.

**Remote Monitoring**: The event log enables remote monitoring:

```
tail -f logs/run_20250114_123456/events.jsonl | jq
```

Teams can build custom dashboards or alerts from this stream. The orchestrator doesn't prescribe monitoring solutions.

**Health Indicators**: Key metrics visible at all times:

- Queue depth (waiting tasks)
- Active task count vs limit
- Total cost accumulation rate
- Average task duration

These indicators reveal system health without overwhelming detail. Rising queue depth might indicate too aggressive parallelism. Unusual cost rates prompt investigation.

## 7.3 Handling Failures

Failure handling follows a simple principle: assume transient issues and retry with session resume before declaring defeat.

**Instance-Level Retries**: Every instance failure triggers automatic retry:

- Default 3 attempts before marking as permanently failed
- Each retry resumes from the last Claude Code session checkpoint
- Exponential backoff between attempts (10s, 60s, 360s)
- Container preserved across retries for session continuity

This approach handles the reality of AI coding: API timeouts, rate limits, temporary network issues. Most failures are transient. Session resume means retry doesn't waste work already done - if Claude wrote half the solution before failing, retry continues from there.

**Failure Categories**: After exhausting retries, failures are marked with cause:

- Timeout: Exceeded instance time limit
- API Error: Persistent API failures (rate limit, auth)
- Container Error: Docker issues, out of memory
- Git Error: import/fetch failures (e.g., ref lock, pack corruption, insufficient disk)

The Instance Runner reports failure type, but doesn't interpret it. That's the strategy's job.

**Cancellation vs Failure**

- Async cancellations caused by orchestrator-initiated shutdown (e.g., Ctrl+C) are treated as interruption, not failure, and MUST NOT be retried.
- `INTERRUPTED` is a first-class terminal state recorded for tasks halted by graceful shutdown or inferred after a crash.
- Strategies MUST NOT treat `INTERRUPTED` as `FAILED`.

**Strategy-Level Handling**: Strategies decide how to respond to failures:

- Best-of-N patterns: Continue with successful candidates; exclude failed or unscorable ones.
- Inline scoring patterns: If a candidate can't be scored after one repair attempt, exclude it rather than coercing a score.
- Custom patterns: Implement domain-specific recovery and selection logic.

This separation is crucial. Infrastructure (Runner) handles retries for transient issues. Business logic (Strategy) handles semantic failures. A network blip gets retried automatically. A fundamental task impossibility gets surfaced to strategy.

**Partial Success Scenarios**: Reality is messy - some instances succeed, others fail:

- 47 of 50 succeed: Results show both successes and failures
- User decides if 94% success rate is acceptable
- Failed instances preserved for debugging
- Clear accounting of what worked and what didn't

No hidden failures or averaged-away problems. Users see exactly what happened and make informed decisions.

**Container Preservation**: Failed instance containers are kept longer than successful ones:

- Failed: 24 hour retention for debugging
- Successful: 2 hour retention
- Contains full Claude session, logs, partial work

This enables post-mortem analysis. Why did instance 3 fail after retries? Check its container, review Claude's session, understand the issue.

The philosophy: be optimistic about recovery, transparent about failures, and pragmatic about partial success.

## 7.4 Resume and Recovery

Resume capability preserves work across interruptions while keeping control explicit:

**Graceful Interruption**: Ctrl+C triggers immediate shutdown:

1. Stop all running containers (preserving their state), regardless of `finalize` settings
2. Save current orchestration state snapshot
3. Set all RUNNING tasks to `INTERRUPTED` and persist `interrupted_at`
4. Display resume command
5. Exit quickly (typically under 10 seconds for 20 containers)

Users see: `Run interrupted. Resume with: orchestrator --resume run_20250114_123456`

The key insight: stopping containers preserves their state. When resumed, instances continue from their last checkpoint. No work is lost, but shutdown is responsive. Users pressing Ctrl+C want immediate response, not graceful completion.

Note: A future `--respect-finalize-on-shutdown` switch may preserve containers that would otherwise keep running; default behavior is to stop all.

**Container Stop vs Kill**: Containers are stopped, not killed:

- `docker stop` sends SIGTERM, allowing clean shutdown
- 10-second grace period before SIGKILL
- Containers stopped in parallel to minimize total time
- Session remains resumable after stop

This balance ensures quick shutdown while maximizing resumability.

**Interrupted State Representation**

- The system MUST persist which tasks were interrupted by recording state=`INTERRUPTED` and an `interrupted_at` timestamp in `state.json`.
- Graceful shutdown MUST NOT mark tasks `FAILED` solely due to cancellation.

**Crash vs Graceful Interruption**

- In a hard crash or power loss, interruption markers may not be written. On resume, any instance that is `RUNNING` in `state.json` and has no terminal state event at or after `last_event_start_offset` MUST be treated as "interrupted by crash" and handled under the Interrupted rules.

**Explicit Resume**: Recovery requires deliberate action:

```bash
orchestrator --resume run_20250114_123456
```

No automatic detection of previous runs. No "found interrupted run, continue?" prompts. If users want to resume, they explicitly say so. This prevents confusion and unexpected behavior.

**Resume State Rules**:

- **Completed instances**: No action, remain completed

- **Failed instances**: No action, remain failed

- **Interrupted instances** (including those inferred after a crash):
  - If container exists AND plugin supports resume: Resume from saved session
  - If container exists BUT plugin doesn't support resume: Mark as `cannot_resume`
  - If container missing BUT workspace exists: Mark as `container_missing`
  - If both container and workspace missing: Mark as `artifacts_missing`

- **Queued instances**: Start normally

Override rule (normative): On resume, orchestration MUST override any `resume_session_id` present in a task input with the most recent persisted `session_id` from `state.json` before calling the runner. This prevents races with later session updates.

Diagnostic note: The isolated workspace is never used for resume; only the container/session state matters. Any references to `workspace` presence in statuses (e.g., `artifacts_missing`) are diagnostic for post‑mortem only and MUST NOT trigger resume logic from the workspace.

**Immediate Finalization on Resume**

- If after restoration and verification there are no runnable tasks (all are terminal or cannot be resumed), the orchestrator SHOULD immediately finalize the run and exit. Completion is inferred when all top‑level strategies emit `strategy.completed`.

Notes on Sessions:

- Each instance's `session_id` MUST be persisted in `state.json`. Resume MUST rely on this persisted `session_id` rather than event replay to determine resumability. If event replay does occur and includes newer `session_id` updates, those MUST overwrite the in-memory value before any resume checks.

Special cases:

- Plugin without session support: Strategy can choose to restart fresh or fail
- Corrupted session: Detected by plugin, treated as failed with `session_corrupted` error
- User can pass `--resume-fresh` to restart interrupted instances without sessions

**State Reconstruction**: Resume follows a simple sequence:

1. Load run snapshot (instance states, strategy progress) from `state.json`
2. Read events from saved `last_event_start_offset` in `events.jsonl`
3. Verify containers still exist for interrupted instances
4. Restart stopped containers and resume sessions where possible
5. Continue strategy execution from saved point

Missing containers or workspaces are handled according to the rules above. No complex recovery heuristics.

**Run Management Commands**: Simple tools for run inspection:

```bash
orchestrator --list-runs              # Show all runs with status
orchestrator --show-run <run-id>      # Display run details
orchestrator --cleanup-run <run-id>   # Remove containers and state
```

These commands provide visibility without magic. Users control their runs explicitly.

The philosophy: preserve work through session continuity, but require explicit decisions about recovery. No surprising automatic behaviors.

## 7.5 Snapshot Atomicity

All snapshots MUST be written atomically using a temp-file-then-rename pattern to avoid partial/corrupt files:

- Primary snapshot: `orchestrator_state/<run_id>/state.json`
- Duplicate snapshot (for convenience and multi-UI): `logs/<run_id>/state.json`

Both MUST be written via a temporary file (e.g., `state.json.tmp`) and then `rename()`d to the final path so readers never observe partial JSON. Readers SHOULD prefer the primary snapshot but MAY fall back to the duplicate if the primary is missing or corrupt.

## 7.6 Results and Reporting

Results presentation focuses on actionability - what branches were created and how to evaluate them:

**Summary Display**: At run completion, users see:

```
═══ Run Complete: run_20250114_123456 ═══

Strategy: best-of-n
  n: 2
  
Runs: 3

Results by strategy:

Strategy #1 (best-of-n):
  ✓ bestofn_20250114_123456_k5d3a9f2  3m 12s • $0.38 • 2.3k tokens
    strategy_output (example): score=7.5, complexity=medium, test_coverage=85%
  ✓ bestofn_20250114_123456_k7aa12cc  3m 45s • $0.41 • 2.5k tokens  
    strategy_output (example): score=8.5, complexity=low, test_coverage=92%
  → Selected: bestofn_20250114_123456_k7aa12cc

Strategy #2 (best-of-n):
  ✓ bestofn_20250114_123456_k19bb44a  4m 23s • $0.52 • 3.1k tokens
    strategy_output (example): score=9.0, complexity=high, test_coverage=78%
  ✗ bestofn_20250114_123456_k2f88e01  Failed: timeout after 3 retries
  → Selected: bestofn_20250114_123456_k19bb44a

Strategy #3 (best-of-n):  
  ✓ bestofn_20250114_123456_k8c0d3aa  3m 56s • $0.44 • 2.7k tokens
    strategy_output (example): score=8.0, complexity=medium, test_coverage=88%
  ✓ bestofn_20250114_123456_k0a9f3de  3m 33s • $0.39 • 2.4k tokens
    strategy_output (example): score=7.0, complexity=low, test_coverage=90%
  → Selected: bestofn_20250114_123456_k8c0d3aa

Summary:
  Total Duration: 8m 45s
  Total Cost: $2.14
  Success Rate: 5/6 tasks (83%)
  
Final branches (3):
  bestofn_20250114_123456_k7aa12cc
  bestofn_20250114_123456_k19bb44a
  bestofn_20250114_123456_k8c0d3aa

Full results: ./results/run_20250114_123456/
```

**Results Directory**: Structured output for further analysis:

- `summary.json` - Machine-readable version of above
- `branches.txt` - Simple list for scripting
- `metrics.csv` - Task-level metrics for analysis
- `strategy/` - Strategy-specific outputs (scores, feedback)

**Branch Comparison**: Practical next steps are suggested:

```
To compare implementations:
  git diff main..bestofn_20250114_123456_k7aa12cc
  git checkout bestofn_20250114_123456_k8c0d3aa
  
To merge selected solution:
  git merge bestofn_20250114_123456_k8c0d3aa
```

**Export Formats**: Results can be exported for different audiences:

- JSON for tooling integration
- Markdown for documentation
- CSV for spreadsheet analysis

No complex reporting engine - just structured data users can process with familiar tools.

The results philosophy: make it clear what was produced and how to use it. The orchestrator's job ends at branch creation - users decide what to merge.

## 7.7 Platform Support

**Primary Platform**: Linux is the primary development and deployment platform with full feature support.

**macOS Support**: Fully supported via Docker Desktop. The SELinux `:z` flag is automatically excluded on macOS. Path length limitations are generally not an issue. Requires Docker Desktop installation.

**Windows Support**:

- Recommended: Run via WSL2 (Windows Subsystem for Linux) for best compatibility
- Alternative: Native Docker Desktop works but has limitations:
  - Path length restrictions (260 character limit)
  - Different path separators may cause issues in some edge cases
  - Performance is typically slower than WSL2
  
WSL2 note: Place the repository inside the WSL filesystem (e.g., `~/repo`), not under `/mnt/c/...`, to avoid slow bind mounts. Bind-mounting NTFS paths into Linux containers can significantly degrade performance.

**Platform Detection**: The system automatically detects the platform and adjusts behavior:

- SELinux flags only applied on Linux systems with SELinux enabled
- Path handling adjusted for Windows when necessary
- File watching uses appropriate APIs (inotify on Linux, kqueue on macOS, polling on Windows)

**Minimum Requirements**:

- Docker 20.10 or newer (Docker Desktop on macOS/Windows)
- Git 2.25 or newer
- Python 3.11+ (tested on 3.13)
- 8GB RAM minimum, 16GB recommended for parallel execution
- 20GB free disk space for Docker images and temporary workspaces

## 7.8 Extended Features

The implementation includes several features beyond the core specification:

**Authentication Flexibility**:

- Fallback authentication from environment variables
- Support for both OAuth tokens (subscription) and API keys
- Custom LLM endpoints via ANTHROPIC_BASE_URL

**Advanced Container Management**:

- Container status tracking via /tmp/orchestrator_status/
- Different retention policies for successful (2h) vs failed (24h) containers
- Orphaned container cleanup on startup
- SELinux volume mount flag application

**Rich Event System**:

- 20+ runner/internal event types including workspace lifecycle and Claude-specific events (not part of the public `events.jsonl` stream)
- Event filtering by run_id, instance_id, and type
- Ring buffer implementation to prevent memory issues
- Monotonic offset tracking for reliable position management

**Comprehensive Error Handling**:

- Pattern-based retry with specific error string matching
- Additional error types: OSError, IOError
- Sophisticated retry configuration beyond basic exponential backoff
- Cancellations (`asyncio.CancelledError`) are terminal and reported as `INTERRUPTED`; they are not retried.
- Error categories with detailed failure reasons

**Performance Optimizations**:

- Multiple background executor tasks for true parallel execution
- Shared git clones for fast instance startup
- Streaming JSON parsing for large Claude responses
- Fixed 10Hz render loop for consistent UI performance

**Operational Excellence**:

- Structured JSON logging with component routing
- File size-based log rotation with backup count
- Rich CLI with table displays and tree views
- Platform-specific optimizations and recommendations
- Debug mode with verbose logging to stderr
RetryPolicy schema (orchestration-level):

```json
{
  "max_attempts": 3,
  "backoff": {"type": "exponential", "base_s": 10, "factor": 6, "max_s": 360},
  "retry_on": ["docker", "api", "network"]
}
```

This augments runner retries; orchestration retries at the task level only for schedule/attach failures.


These extensions emerged from real-world usage and make the orchestrator production-ready while maintaining the clean architecture described in this specification.
Minimum payload fields per event type (MUST be present, alongside envelope fields `run_id`, `strategy_execution_id`, and `start_offset`):

- `task.scheduled`: `{ key, instance_id, container_name, model, task_fingerprint_hash }`
- `task.started`: `{ key, instance_id, container_name, model }`
- `task.completed`: `{ key, instance_id, artifact, metrics, final_message, final_message_truncated, final_message_path }`  # success only
  `final_message` MAY be truncated per `events.max_final_message_bytes`. When truncated, `final_message_truncated=true` and `final_message_path` MUST point to the full blob under logs; otherwise `final_message_truncated=false` and `final_message_path` MAY be empty.
- `task.failed`: `{ key, instance_id, error_type, message }`
- `task.interrupted`: `{ key, instance_id }`
- `strategy.started`: `{ name, params }`
- `strategy.completed`: `{ status }` where `status ∈ {"success","failed","canceled"}`
