# 1. Introduction

## 1.1 Purpose and Goals

The Orchestrator enables parallel execution of AI coding agents to explore multiple solution paths simultaneously. When using AI for software development, outcomes vary significantly between runs - even with identical prompts. This tool leverages that variability as a strength, running multiple instances in parallel and implementing selection strategies to identify the best results.

Key goals:

- **Scale AI coding horizontally** - Run N instances of Claude Code (or similar tools) in parallel
- **Implement complex workflows** - Chain instances for review, scoring, and iterative refinement
- **Abstract infrastructure complexity** - Handle containerization, git operations, and state management transparently
- **Enable rapid experimentation** - Define sophisticated multi-stage strategies in ~50 lines of code
- **Maintain solution traceability** - Every instance produces a git branch for review and selection

The tool transforms single-threaded AI coding into a parallel, strategy-driven process where you can run approaches like "generate 5 implementations, have AI review each one, pick the best scoring result, then iterate on it with feedback" - all automated.

## 1.2 Core Concepts

**Instance** - A single isolated execution of an AI coding agent (Claude Code) with a specific prompt. Each instance runs in its own Docker container with a clean git workspace, producing a new branch with its solution.

**Strategy** - A composable execution pattern that coordinates multiple instances. Strategies can spawn instances, wait for results, and make decisions based on outputs. Examples: `BestOfNStrategy`, `ScoringStrategy`, `IterativeStrategy`.

**InstanceResult** - The output from a completed instance containing: branch name, token usage, cost, execution time, final message, and strategy-specific metadata (e.g., review scores).

**Orchestration Run** - A complete execution of the orchestrator with a chosen strategy. Identified by a unique run ID, it tracks all spawned instances, aggregate metrics, and final results.

**Event Stream** - Real-time events emitted by instances and the orchestration layer. Events drive the TUI display and enable monitoring of parallel executions.

**Branch-per-Instance** - Each instance commits to an isolated branch (e.g., `bestofn_20250723_141523_2_3`), enabling easy comparison and cherry-picking of solutions.

## 1.3 Example Workflows

### Simple Parallel Exploration

```bash
orchestrator "implement user authentication with OAuth2" --strategy simple --runs 5
```

Runs 5 parallel strategies (each strategy spawns one instance). The `--runs` parameter controls orchestrator-level parallelism - how many strategy executions run simultaneously. Creates 5 branches with different implementations, displays real-time progress in TUI.

### Best-of-N Strategy

```bash
orchestrator "implement caching layer" --strategy best-of-n --runs 3 -S n=5
```

Runs 3 parallel best-of-n strategies. Each strategy internally spawns 5 instances (specified by `-S n=5`), reviews them, and selects the best one. With `--runs 3`, you get 3 final "best" solutions, each selected from its own pool of candidates.

### Scoring Strategy (Building Block)

```bash
orchestrator "refactor auth module" --strategy scoring
```

Runs one scoring strategy that: generates an implementation → spawns a reviewer instance → attaches review score and feedback to the result. The scoring strategy is primarily useful as a component within custom strategies rather than standalone use.

### Complex Multi-Stage Pipeline

```python
# Custom strategy that separates planning from execution
# Goal: First explore different high-level approaches, then deeply implement the best one
class PlanAndExecuteStrategy(Strategy):
    async def execute(self, prompt, base_branch, ctx):
        # Stage 1: Generate diverse high-level plans using different approaches
        plans = await ctx.parallel([
            ctx.spawn_instance(
                f"Create a detailed plan: {prompt} - focusing on {approach}", 
                base_branch
            )
            for approach in ["performance", "simplicity", "extensibility"]
        ])
        
        # Stage 2: Score each plan to find the best approach
        scored_plans = await ctx.parallel([
            ctx.spawn_instance(
                "Rate this plan's feasibility and quality", 
                plan.branch_name,
                metadata={"strategy": "scoring"}
            )
            for plan in plans
        ])
        
        # Stage 3: Take the best plan and implement it with iterative refinement
        best_plan = max(scored_plans, key=lambda x: x.metadata['score'])
        return await ctx.spawn_instance(
            f"Implement this plan with all details:\n{best_plan.final_message}",
            best_plan.branch_name,
            metadata={"strategy": "iterative", "iterations": 3}
        )
```

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

**Orchestration** - Coordinates multiple instances according to strategies. Owns the event bus, manages parallel execution, and tracks global state. Depends only on Instance Runner's public API. Strategies are just Python classes that spawn instances and make decisions based on results. Provides branch names and container names to instances.

**TUI (Interface)** - Displays real-time progress and results. Subscribes to orchestration events for real-time updates and periodically polls state for reconciliation. Has zero knowledge of how instances run or how strategies work - just visualizes events and state. Can be replaced with a web UI or CLI-only output without touching other layers.

This architecture emerged from painful lessons: previous attempts failed when components were tightly coupled. Now, you can completely rewrite the TUI without touching instance execution, or swap Claude Code for another AI tool by only modifying Instance Runner.

## 2.2 Event-Driven Communication

Components communicate through a simple event system that lives within the Orchestration layer:

```python
# Instance Runner emits events via callback
await run_instance(
    prompt="implement feature",
    event_callback=lambda e: orchestrator.emit_event(e)
)

# TUI subscribes to events it cares about
orchestrator.subscribe("instance.status_changed", update_display)
orchestrator.subscribe("instance.completed", show_result)
```

Events flow unidirectionally upward:

- Instance Runner emits fine-grained events (git operations, tool usage, token updates)
- Orchestration adds strategy-level events (strategy started, instance spawned, selection made)
- TUI consumes events to update displays

The event system uses append-only file storage (`events.jsonl`) with monotonic byte offsets for position tracking. Each event is written synchronously with its offset, preventing data loss and enabling reliable event replay. Components can request events from a specific offset for recovery after disconnection.

**Offset Semantics**

- Offsets are byte positions (not line counts). Readers MUST open in binary mode and advance offsets by the exact number of bytes consumed.
- Readers MUST align to newline boundaries: if an offset lands mid-line, scan forward to the next `\n` before parsing.
- Readers MUST tolerate a truncated final line (e.g., during a crash) by skipping it until a terminating newline appears.
- Writers SHOULD emit UTF-8 JSON per line and flush synchronously to maintain monotonicity.

Components follow a strict communication pattern: downward direct calls (Orchestration → Runner) are allowed for control flow, while upward communication (Runner → Orchestration, any → TUI) happens primarily through events, with periodic state polling for reconciliation. The `subscribe()` function sets up file watching on `events.jsonl`, enabling capabilities like multiple processes monitoring the same run by tailing the event file or replaying events for debugging.

## 2.3 Data Flow

The system follows a clear request/response + event stream pattern:

**Downward Flow (Requests)**

1. User provides prompt and strategy selection via CLI
2. TUI passes request to Orchestration
3. Orchestration interprets strategy, spawns instances via Runner
4. Runner executes Claude Code in containers

**Upward Flow (Results + Events)**

1. Claude Code outputs structured logs in JSON format
2. Runner parses logs, emits events, collects results
3. Orchestration aggregates results, makes strategy decisions, emits higher-level events
4. TUI receives events and displays progress

**State Queries**

- TUI can query Orchestration for current state snapshots
- Orchestration maintains authoritative state derived from events
- No component stores UI state - everything derives from event stream

This unidirectional flow prevents circular dependencies and makes the system predictable. Each layer exposes a narrow API to the layer above it, maintaining clear boundaries and separation of concerns.

## 2.4 Technology Choices

**Python 3.13** - Latest stable release with improved async performance and better error messages. Type hints enable clear interfaces between components. Rich ecosystem for all required functionality. Modern Python for modern tooling.

**asyncio** - Natural fit for I/O-bound operations (Docker commands, git operations, API calls). Enables high concurrency for managing hundreds of container operations and git commands without thread complexity. Built-in primitives for coordination (locks, queues, events).

**docker-py 7.1.0** - Python library for Docker container management. Provides programmatic control over container lifecycle, resource limits, and cleanup. The Docker daemon version is less critical as we use standard features compatible with any recent version.

**uv** - Lightning-fast Python package and project manager. Replaces pip, pip-tools, pipenv, poetry, and virtualenv with a single tool. Near-instant dependency resolution and project setup. Written in Rust for performance.

**Git** - Natural version control for code outputs. Branch-per-instance model enables easy comparison. Local operations are fast and reliable. Universal developer familiarity.

**Rich 14.0.0** - Modern terminal UI capabilities with responsive layouts. Live updating without flicker. Built-in tables, progress bars, and syntax highlighting. Latest version includes improved performance for large displays.

**Structured JSON output** - Claude Code outputs structured JSON logs that enable real-time parsing of agent actions. Provides detailed metrics and session IDs for resume capability.

**No database** - Event sourcing with file-based persistence. Reduces operational complexity. State rebuilds from event log. Git itself acts as the "database" for code outputs.

**No message queue** - Simple file-based event system suffices for single-machine operation. Append-only `events.jsonl` with offset tracking enables multi-process monitoring. Direct callback pattern reduces latency.

These choices optimize for developer experience and operational simplicity. Using latest stable versions ensures access to performance improvements and security updates, while uv dramatically simplifies Python project management compared to traditional tooling.

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
- **instance_id**: Unique identifier (auto-generated if not provided)
- **container_name**: Full container name including run_id (provided by orchestration)
- **model**: Claude model to use (default: "sonnet")
- **session_id**: Resume a previous Claude Code session
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

Returns `InstanceResult` containing:

- Branch name where solution was imported to host repository
- Execution metrics (cost, tokens, duration)
- Token breakdown (input/output/total)
- Session ID for resume capability
- Final message from Claude
- Status (success/failed/timeout/canceled)
- Container and branch details
- Commit statistics (count, lines added/deleted)
- Timestamps (started_at, completed_at)
- Number of retry attempts
- Path to detailed logs
- Workspace path (until cleanup)
- Error information if failed
- has_changes: Boolean indicating if any code changes were made
- Strategy-specific metadata (Dict[str, Any])
- Convenience properties: cost (returns metrics.cost), tokens (returns metrics.total_tokens)

The API is designed for both simple usage (`await run_instance("fix bug", repo)`) and advanced control when needed. The runner remains agnostic about why the container is named a certain way or how branches are named - it simply uses what orchestration provides. This separation keeps the runner generic and testable.

## 3.3 Execution Pipeline

The runner follows six strict phases ensuring consistent execution and proper cleanup:

1. **Validation** - Verify Docker daemon, repository exists, validate provided container_name, check disk space
2. **Workspace Preparation** - Create isolated git clone in temporary directory BEFORE container creation. This ensures the workspace is ready when the container starts
3. **Container Creation** - If `reuse_container=True` and container exists, reuse it. Otherwise, start new container with provided name and resource limits, mount the pre-prepared workspace. Container runs with `sleep infinity` for persistence
4. **Claude Code Execution** - Execute via `docker exec` with structured JSON output, parse events
5. **Result Collection** - Extract metrics, import branch from workspace to host repository as the final atomic step. Instance is only marked complete after successful import
6. **Cleanup Decision** - For successful instances: delete workspace immediately, stop container (retained for 2h). For failed instances: keep both workspace and container for 24h debugging

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
# Create temporary directory for this instance
WORKSPACE_DIR="/tmp/orchestrator/${run_id}/i_${sidx}_${iidx}"
mkdir -p "$WORKSPACE_DIR"

# Clone the specific branch in isolation (handled by git operations)
# ... git clone happens here ...
```

**Volume Mounts**: Three mounts required for proper operation:

```
/tmp/orchestrator/<run-id>/i_<sidx>_<iidx> → /workspace
Named volume → /home/node
tmpfs → /tmp
```

Platform-specific mount handling ensures compatibility. The named volume (pattern: `orc_home_{run_id}_s{sidx}_i{iidx}`) provides persistent storage for Claude Code's session data, ensuring sessions survive container restarts. Note: The implementation uses `/home/node` due to the base Docker image using the `node` user.

**Container Execution**: Containers run with these security constraints:

```bash
# Detect platform and conditionally add SELinux flag
selinux_flag=""
if [[ "$(uname)" == "Linux" ]] && [[ -e /sys/fs/selinux ]]; then
    selinux_flag=":z"
fi

docker run \
  -v "$WORKSPACE_DIR:/workspace${selinux_flag}" \
  --mount "type=volume,source=orc_home_${run_id}_s${sidx}_i${iidx},target=/home/claude" \
  --tmpfs /tmp:rw,size=256m \       # Writable temporary space
  --read-only \                     # Entire filesystem read-only except mounts
  --name "$container_name" \
  --cpus="2" \
  --memory="4g" \
  --label "orchestrator=true" \
  --label "run_id=${run_id}" \
  --label "strategy_exec=${sidx}" \
  --label "instance_index=${iidx}" \
  claude-code:latest \
  sleep infinity                    # Keep container running for exec
```

The `--read-only` flag locks down the entire container filesystem except for explicitly mounted volumes. Containers have access to exactly three locations: `/workspace` (isolated git clone), `/home/claude` (session state), and `/tmp` (temporary files). The `sleep infinity` command keeps the container running, allowing subsequent `docker exec` commands to run Claude Code.

**Container Reuse**: When `reuse_container=True`:

- If container exists and is running: Execute Claude Code in existing container
- If container exists but stopped: Start container, then execute
- If container doesn't exist: Create new container as above

**Container Persistence**: Containers are kept after execution to enable Claude Code session resume:

- Failed/timeout instances: Retained for 24 hours (configurable)
- Successful instances: Retained for 2 hours (configurable)
- When `finalize=True`: Container is stopped (not removed) after successful completion
- When `finalize=False`: Container continues running

This persistence is crucial because Claude Code maintains session state inside the container. If an instance times out after partial work, the session can be resumed by using the same container with the preserved session.

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
    docker stop "$container_name"  # Stop but retain container for 2h
fi
# Failed instances keep both workspace and container for 24h debugging
```

**Orphan Cleanup**: On component startup, the Instance Runner:

1. Scans for containers with label `orchestrator=true` older than retention period
2. Checks if their workspace directories still exist
3. Removes both orphaned containers and their associated volumes
4. Logs cleanup actions for audit purposes

This handles cases where the orchestrator crashed without proper cleanup.

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

**Container Status Tracking**: The implementation uses an additional status tracking mechanism via `/tmp/orchestrator_status/` on the host filesystem. Container labels include `instance_status` which is updated throughout the lifecycle.

The Docker management layer remains agnostic about orchestration strategies or git workflows. It simply provides isolated execution environments with proper resource controls and session preservation. The true isolation comes from each container working with its own complete git repository, with no possibility of cross-contamination between instances or with the host system.

## 3.5 Git Operations

Git operations use a carefully designed isolation strategy that achieves perfect separation between parallel instances while maintaining efficiency. This approach emerged from the fundamental requirement that each AI agent must work in complete isolation, seeing only the branch it needs to modify, without any possibility of interfering with other instances or accessing the host repository.

**The Isolation Strategy**: Before starting each container, the Instance Runner creates a completely isolated git repository on the host:

```bash
# On the host, before container starts
git clone --branch <base-branch> --single-branch --no-hardlinks \
          /path/to/host/repo /tmp/orchestrator/${run_id}/i_${sidx}_${iidx}
cd /tmp/orchestrator/${run_id}/i_${sidx}_${iidx}
git remote remove origin        # Complete disconnection from source
# Store base branch reference for later
echo "<base-branch>" > .git/BASE_BRANCH
```

Let's break down why each flag matters:

- `--branch <base-branch> --single-branch` ensures the clone contains ONLY the target branch. No other refs exist in `.git/refs/heads/`. The agent literally cannot see any other branches because they don't exist in its universe
- `--no-hardlinks` forces Git to physically copy all object files instead of using filesystem hardlinks. This prevents any inode-level crosstalk between repositories - critical for true isolation
- `git remote remove origin` completes the isolation. With no remote configured, the agent cannot push anywhere even if it tried
- Storing the base branch reference ensures we maintain traceability of where changes originated

**Agent Commits**: The AI agent works directly on the base branch in its isolated clone. It doesn't create new branches - it simply commits its changes to the only branch it can see. This simplifies the agent's task and ensures predictable behavior.

**Container Workspace**: Each container receives its isolated clone as a volume mount:

```bash
# Detect platform and conditionally add SELinux flag
selinux_flag=""
if [[ "$(uname)" == "Linux" ]] && [[ -e /sys/fs/selinux ]]; then
    selinux_flag=":z"
fi

docker run \
  -v /tmp/orchestrator/${run_id}/i_${sidx}_${iidx}:/workspace${selinux_flag} \
  --read-only \
  --name ${container_name} \
  claude-code:latest
```

The SELinux flag is only applied on Linux systems where it's needed. The `--read-only` flag locks down everything except the mounted volumes. The container can only modify its isolated git repository.

**Branch Import After Completion**: The magic happens after the container exits, as the final step of run_instance(). Instead of complex push coordination, we use Git's ability to fetch from local filesystem paths:

```bash
# After container exits, back on the host
cd /path/to/host/repo
# Check if target branch already exists
if git show-ref --verify --quiet refs/heads/${branch_name}; then
    echo "Error: Branch ${branch_name} already exists"
    exit 1  # Or handle with --force-import flag if provided
fi
# Read the original base branch
BASE_BRANCH=$(cat /tmp/orchestrator/${run_id}/i_${sidx}_${iidx}/.git/BASE_BRANCH)
# Import from the base branch to new branch name
git fetch /tmp/orchestrator/${run_id}/i_${sidx}_${iidx} ${BASE_BRANCH}:${branch_name}
```

This import operation:

- Copies all new commits and their associated blobs/trees
- Creates the branch atomically in the host repository
- Requires no network operations or remote configuration
- Fails cleanly if the branch already exists (unless --force-import is used)
- Preserves the connection to the original base branch
- Handles cases where no commits were made by creating a branch pointing to the base branch

**Event Emission and Cleanup**: The instance runner must emit the final completion event BEFORE cleaning up the workspace:

```bash
# Emit completion event with workspace path
emit_event("instance.completed", {
    "workspace_path": "/tmp/orchestrator/${run_id}/i_${sidx}_${iidx}",
    "branch_imported": branch_name
})

# Then clean up after successful import
rm -rf /tmp/orchestrator/${run_id}/i_${sidx}_${iidx}
```

This ensures the event log contains the workspace path for audit trails before it's deleted.

**Performance Characteristics**: Testing with 50 parallel instances showed:

- Clone operations complete in 0.5-2 seconds depending on repository size
- Each isolated clone uses full disk space (no sharing), but this is negligible with modern storage
- Zero conflicts possible - each instance works in complete isolation
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
git fetch /tmp/orchestrator/${run_id}/i_${sidx}_${iidx}/result.bundle HEAD:${branch_name}
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

- Tool usage (Write, Edit, Bash commands, etc.) - emitted as `instance.claude_tool_use` events
- Token consumption from usage field - tracked in real-time
- Cost accumulation from total_cost_usd - aggregated across the session
- Error states from is_error flag
- Session ID for resume capability
- Tool results and output - emitted as `instance.claude_tool_result` events
- Phase transitions and workspace events

**Session Handling**: Each execution gets a unique session_id from Claude Code (e.g., `"2280667e-25e1-46ac-b7f4-722d7e486c9c"`). This ID enables resuming interrupted work via the `--resume <session_id>` CLI flag. Sessions persist in the named volume mounted at `/home/claude`, surviving container restarts.

**Authentication**: Based on auth_config, the appropriate environment variables are set:

- Subscription mode: `CLAUDE_CODE_OAUTH_TOKEN`
- API mode: `ANTHROPIC_API_KEY` and optionally `ANTHROPIC_BASE_URL`

**Prompt Engineering**: System prompts and append-system-prompt options enable customizing Claude's behavior per instance. Model selection allows choosing between speed (Sonnet) and capability (Opus).

## 3.7 Error Handling

Errors are categorized for appropriate recovery strategies:

**Container Errors**: Docker daemon issues, resource exhaustion, network problems. Generally retriable with new container.

**Git Errors**: Clone failures, missing branches, fetch issues. Usually require configuration fixes, not retriable.

**Claude Code Errors**: API failures, timeout, cost limits. Retriable using session resume to continue from last checkpoint.

**Timeout Handling**: Enforced at Claude Code level via max_turns. Container-level timeout as fallback. Timed-out instances keep containers for resume.

**Retry Strategy**: Automatic exponential backoff for transient failures during execution. Session-based resume for Claude Code failures. Maximum retry attempts configurable (default: 3). Retries handle temporary issues like network blips or API rate limits. After exhausting retries, the instance is marked as failed. This is distinct from orchestration-level resume after interruption, where already-failed instances remain failed.

**Retryable Error Patterns**: Specific error strings trigger automatic retry:

- Docker: "connection refused", "no such host", "timeout", "daemon", "Cannot connect"
- Claude: "rate limit", "API error", "connection reset", "overloaded_error"
- General: "ECONNREFUSED", "ETIMEDOUT", "ENETUNREACH"
- System: OSError, IOError, asyncio.CancelledError (handled with appropriate recovery)

The retry configuration supports pattern-based matching with tuples of (pattern, error_type) for fine-grained control.

## 3.8 Runner Plugin Interface

Abstract interface enables supporting multiple AI coding tools:

**Core Methods**:

- `validate_environment()`: Check tool is available and configured
- `prepare_container()`: Create container with tool-specific setup
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

Strategies check required capabilities before execution and fail fast with clear errors if unsupported features are needed.

**Authentication Handling**: Each plugin manages its own authentication approach:

```python
# Plugin receives auth config and decides how to use it
auth_config = {"oauth_token": "...", "api_key": "...", "base_url": "..."}
env = await plugin.prepare_environment(auth_config)
```

This keeps tool-specific auth logic encapsulated within the appropriate plugin.

**Implementation Strategy**: Each runner translates its tool's specific output into common events and results. ClaudeCodeRunner handles structured JSON format. Future GeminiRunner would parse Gemini's output.

**Selection**: Runners are selected via configuration. Allows users to choose their preferred AI tool while keeping same orchestration features.

**Graceful Degradation**: When plugins lack certain capabilities:

- If `supports_cost` is False but `supports_token_counts` is True: estimate costs using configured rates
- If neither cost nor token support: display "N/A" for these metrics
- If `supports_resume` is False: strategies requiring resume capability fail with clear message

# 4. Orchestration Component

## 4.1 Overview and Responsibilities

The Orchestration layer coordinates multiple instances according to strategies. While Instance Runner handles "how to run one instance", Orchestration handles "how to run many instances intelligently". This separation enables complex workflows like "generate 5 solutions and pick the best" without complicating the runner logic.

Core responsibilities:

- Execute strategies that spawn and coordinate instances
- Manage parallel execution with resource limits
- Own the event bus for component communication
- Track current state of all instances and strategies
- Generate container names with run_id for proper grouping
- Generate unique branch names for each instance
- Handle instance failures according to strategy logic
- Provide HTTP server for multi-UI support
- Support resumption of interrupted runs
- Clean up orphaned containers on startup
- Export comprehensive results including metrics and branches
- Validate disk space before execution

The orchestration depends only on Instance Runner's public API. It exposes clean interfaces for running strategies and querying state, hiding all complexity of parallel coordination.

Key architectural decisions:

- Container naming scheme (`orchestrator_{run_timestamp}_s{sidx}_i{iidx}`) is orchestration's responsibility (extracts timestamp from run_id)
- Branch naming pattern (`strategy_timestamp_sidx_iidx`) ensures uniqueness
- Strategies work with high-level abstractions, unaware of these coordination details

This layer transforms the simple "run one instance" capability into powerful multi-instance workflows while keeping both runners and strategies clean.

## 4.2 Public API

The Orchestration component provides six essential interfaces:

**Strategy Execution**: The `run_strategy()` function accepts a strategy name, prompt, base branch, and strategy-specific configuration. Returns a list of `InstanceResult` objects representing the final outputs. This simple interface hides all complexity - whether the strategy runs 1 or 100 instances internally.

**State Management**: The `get_current_state()` function returns a complete snapshot of the system: running instances, active strategies, aggregate metrics. This enables the TUI to display current status without tracking individual events. State includes derived data like "5 of 10 instances complete" for progress displays.

**Event System**: The `subscribe()` function sets up file watching on `events.jsonl` to receive real-time events. Orchestration appends all events to this file, including forwarded Instance Runner events and its own strategy-level events. This decoupling means TUI never directly calls Instance Runner. Even in-process components use file watching for consistency.

**Event History**: The `get_events_since()` function retrieves historical events from `events.jsonl` starting at a given byte offset or timestamp. Accepts run_id, offset/timestamp, optional event type filters, and result limit. If offset is before the earliest available event, returns from the beginning of the file. This enables reliable recovery after disconnection without missing events.

**Run Resumption**: The `resume_run()` function restores an interrupted run from saved state. It loads the state snapshot, replays events, verifies container existence, checks plugin capabilities, and continues execution from where it left off. Supports both session-based resume and fresh restart options.

**HTTP Server**: The `start_http_server()` function launches an optional HTTP server providing REST API endpoints for external monitoring. Endpoints include `/state` for current state, `/events` for event streaming, and `/health` for liveness checks.

These APIs follow a key principle: make simple things simple, complex things possible. Running a basic strategy is one function call, but power users can subscribe to detailed events for monitoring.

## 4.3 Strategy System

Strategies encapsulate patterns for coordinating instances. Think of them as "recipes" for using AI coding tools effectively:

**Base Strategy**: All strategies inherit from an abstract base class with a single required method: `execute()`. This method signature is:

```python
async def execute(self, prompt: str, base_branch: str, ctx: StrategyContext) -> List[InstanceResult]
```

The `StrategyContext` provides access to instance spawning, parallel execution helpers, and the event bus.

**Instance Spawning**: Strategies create instances through the context's `spawn_instance()` method:

```python
handle = await ctx.spawn_instance(
    prompt="implement feature", 
    base_branch="main",
    model="opus",
    metadata={"phase": "implementation"}
)
result = await handle.result()  # Wait for completion
```

This allows orchestration to manage resources, assign branch names, and emit events. Strategies focus on logic, not infrastructure. The context also provides:

- `instance_counter`: Automatic instance index tracking within the strategy
- `emit_event()`: Method to emit strategy-specific events
- Access to orchestration configuration and run metadata

**Parallel Execution**: The context provides a `parallel()` helper for concurrent instance spawning:

```python
results = await ctx.parallel([
    ctx.spawn_instance(f"implement with approach {i}", base_branch)
    for i in range(5)
])
```

**Failure Handling**: Each `InstanceResult` includes a status (success/failed). Strategies decide how to handle failures - BestOfNStrategy might skip failed instances, while ScoringStrategy fails entirely if it can't score. This flexibility enables robust workflows.

**Composition**: Strategies can use other strategies as building blocks. For example, a custom strategy might use ScoringStrategy internally to evaluate results. This composability enables complex workflows from simple parts.

**Configuration**: Each strategy defines its own parameters. BestOfNStrategy needs `n` (how many to generate) and optionally `scorer_prompt`. Strategies validate their configuration and provide clear errors for missing parameters.

## 4.4 Built-in Strategies

The system includes four proven strategies covering common use cases:

**SimpleStrategy**: Executes exactly one instance. No parallelism, no complexity. Exists as a baseline and for tasks where you want precisely one attempt. If the instance fails, the strategy fails.

**ScoringStrategy**: Two-phase pattern - first generates a solution, then runs a reviewer instance to evaluate it. Attaches score and feedback to result metadata. Fails if either phase fails, since scoring without a result is meaningless. Primarily useful as a component in other strategies.

**BestOfNStrategy**: The workhorse strategy - generates N solutions in parallel, scores each one, returns the best. Default N is 5. Gracefully handles partial failures by selecting from successful instances. If all instances fail, the strategy fails. This pattern leverages AI's variability as a strength. The strategy:

- Uses ScoringStrategy internally as a component for evaluation
- Handles invalid scores (non-numeric) by treating them as 0
- Attaches rich metadata including scores, scorer branch, and selection status
- Supports custom scorer prompts via configuration

**IterativeStrategy**: Refinement loop - generates initial solution, runs scoring instance for feedback, then improves the solution. Each iteration continues in the same Claude session for context. Configurable iteration count (default 3). Powerful for complex tasks that benefit from revision.

These strategies emerged from real usage patterns. They're not theoretical - they solve actual problems developers face when using AI coding tools.

## 4.5 Parallel Execution Engine

Managing concurrent instances requires careful resource control:

**Resource Pool**: Maximum parallel instances configurable (default 20). This limit prevents system overload and API rate limit issues. When at capacity, new instance requests queue until a slot opens.

**FIFO Scheduling**: Simple first-in-first-out queue ensures fairness. No complex priority systems - strategies that submit first get resources first. This predictability makes debugging easier.

**Instance Spawning**: When launching instances, the engine:

- Generates container names using pattern `orchestrator_{run_timestamp}_s{sidx}_i{iidx}` (extracts timestamp from run_id)
- Generates unique branch names using pattern `{strategy}_{timestamp}_{sidx}_{iidx}` (uses current timestamp)
- Tracks instance-to-container mapping for monitoring and cleanup
- Manages multiple background executor tasks (configurable count) for true parallel execution
- Validates disk space before starting instances (20GB minimum)

**Execution Tracking**: Each instance tracked from spawn to completion. Strategies can wait for specific instances or groups. Essential for patterns like "wait for all 5 to complete, then pick best".

**Resource Cleanup**: When instances complete, their slots immediately return to the pool. Failed instances don't block resources. Ensures maximum utilization without manual intervention.

The engine is intentionally simple. We resisted adding complex features like priority queues or dynamic scaling because FIFO + fixed pool size handles real workloads well. The engine handles the mechanical aspects of parallel execution while strategies provide the intelligence about what to run and when.

## 4.6 State Management

State tracking serves two purposes: enabling UI displays and crash recovery:

**In-Memory State**: Primary state lives in memory as simple data structures:

- Running instances with their status and progress
- Active strategies with their configuration
- Completed results awaiting processing
- Aggregate metrics (total cost, duration, token usage)

**Event Sourcing**: Every state change emits an event to `events.jsonl`. The current state can be reconstructed by replaying events from this file. This pattern enables resumability - after a crash, replay events to restore state.

**State Persistence Implementation**: State is persisted through two mechanisms:

- `events.jsonl`: Append-only log containing every state change with monotonic byte offsets. Each event is written synchronously with flush to ensure durability
- `state.json`: Periodic snapshot (every 30 seconds) containing full state and the last applied event offset. Snapshots MUST persist, per instance: `state`, `started_at`, `completed_at`, `interrupted_at`, `branch_name`, `container_name`, and `session_id` to enable resumability. Written atomically using temp file + rename

Recovery process:

1. Load `state.json` if it exists (includes last applied event offset)
2. Read `events.jsonl` from the saved offset
3. Apply events to reconstruct current state
4. Handle cases where state directory structure changed (e.g., run_ prefix removal)

**Event Bus Features**: The event system includes:

- Ring buffer implementation (default 10,000 events) to prevent memory issues
- File watching support for external processes via `subscribe_to_file()`
- Event filtering by type, timestamp, and run_id
- Monotonic offset tracking for reliable position management

**Atomic Snapshots**: The `get_current_state()` function returns consistent snapshots. No partial updates or race conditions. UI can poll this API for smooth updates without event subscription complexity.

This approach balances simplicity with reliability. Memory is fast for normal operation, while file-based persistence enables recovery when needed. The combination of event log and periodic snapshots ensures quick recovery without replaying the entire event history.

**Idempotent Recovery**

- `state.json` MUST include `last_event_offset` (byte position of the last applied event).
- Recovery MUST replay only events after `last_event_offset`.
- Event application MUST be idempotent: re-applying a transition already reflected in the snapshot MUST NOT double count aggregate metrics (e.g., completed/failed counters).
- Aggregate counters MUST be derived from guarded state transitions (increment only when moving from a non-terminal to a terminal state).
- `state.instance_updated` events MAY include `session_id`. Applying such events MUST update the instance's `session_id` in memory for subsequent snapshots and resume logic.

## 4.7 Git Synchronization

Parallel instances work in complete isolation, requiring no coordination during execution. The synchronization happens after containers complete through a simple branch import process:

**Branch Import Process**: After each container exits, the Instance Runner performs the git import as the final atomic step of `run_instance()`. This uses git's filesystem fetch capability to import the isolated workspace into the host repository. The import happens sequentially per instance but requires no global locking since each creates a unique branch. The import is atomic - either all commits transfer successfully or none do.

**Branch Naming**: Orchestration generates branch names like `bestofn_20250114_123456_2_3`. Pattern includes strategy name, run_id timestamp, strategy execution index, and instance index. This ensures uniqueness, aids debugging, and groups related branches. The full name is passed to Instance Runner which creates the corresponding branch in the host repository during import.

**Clean Separation**: The synchronization mechanism maintains architectural boundaries:

- Orchestration owns branch naming policy
- Instance Runner performs the mechanical git fetch operation
- Isolated workspaces eliminate any possibility of conflicts
- Neither component needs to understand the other's logic

**Import Operation**: The actual branch import preserves the original base branch reference:

```bash
# Performed by Instance Runner as final step
cd /path/to/host/repo
BASE_BRANCH=$(cat /tmp/orchestrator/${run_id}/i_${sidx}_${iidx}/.git/BASE_BRANCH)
git fetch /tmp/orchestrator/${run_id}/i_${sidx}_${iidx} ${BASE_BRANCH}:${branch_name}
```

This operation cannot conflict with other imports because each targets a unique branch name. Multiple imports can even run simultaneously without issue.

**Import Failures**: If git fetch fails (typically due to disk space or corrupted objects), the instance is marked as failed with clear error details. The isolated workspace is preserved for debugging. No retry logic needed - the workspace remains available for manual inspection or re-import attempts. This simplicity makes failures rare and easy to diagnose.

**No Remote Operations**: By design, containers never push to any remote. All git operations are local to the host machine. If remote synchronization is needed, it happens as a separate post-processing step after all instances complete, completely decoupled from instance execution.

**Workspace Cleanup**: After successful import, the temporary workspace is immediately deleted to free disk space. Failed instances retain their workspaces according to the retention policy, allowing post-mortem analysis of what the AI agent attempted.

**Performance Characteristics**: The import approach scales perfectly:

- No coordination needed between parallel instances
- Import operations take milliseconds to low seconds depending on changeset size
- No network latency since everything is filesystem-based
- Default limit of 20 parallel instances ensures system stability while maintaining high throughput

**Audit Trail**: Each import operation is logged with:

- Source workspace path
- Target branch name
- Commit count and size
- Success/failure status
- Timestamp and duration

This provides complete visibility into what each AI agent produced and when it was integrated.

The git synchronization is minimal by design. By preparing isolated environments beforehand and importing results afterward, we eliminate coordination complexity entirely. This approach handles parallel instances without locks, semaphores, or conflict resolution - each instance truly runs in its own universe and the orchestrator simply collects the results.

## 4.8 HTTP Server and Multi-UI Support

The orchestrator can run an HTTP server providing REST endpoints for external monitoring:

**Server Configuration**: The server starts on port 8080 by default (configurable via `--http-port`). It runs in a separate thread from the main orchestrator, providing read-only access to state and events. Multiple clients can connect simultaneously without affecting orchestration performance.

**API Endpoints**:

- `GET /state` - Returns current state JSON with all instance details
- `GET /events?since=<offset>&limit=<n>` - Streams events from the specified offset
- `GET /health` - Simple liveness check returning 200 OK

**Use Cases**:

- Web dashboards showing real-time progress
- External monitoring systems tracking runs
- Multiple TUI instances viewing the same run
- Integration with CI/CD pipelines

**Security**: The server is read-only by design. No endpoints allow state modification. Authentication is not implemented - the server should only be exposed on trusted networks.

## 4.9 Resume Capabilities

The orchestrator supports resuming interrupted runs with full state recovery:

**Resume Process**:

1. Load saved state from `state.json` including last event offset
2. Replay events from the saved offset to catch up
3. Check which containers still exist using Docker API
4. Verify plugin supports resumption (not all do)
5. Continue execution from where it left off

**Resume Options**:

- `--resume <run-id>`: Continue existing Claude sessions if possible
- `--resume-fresh <run-id>`: Start fresh instances but maintain run state

**Limitations**:

- Only works if Claude Code plugin supports session resumption
- Containers must still exist (not cleaned up)
- Maximum resume window depends on container retention policy

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
- Instance-level details and outcomes
- Strategy-specific results

**Metrics Export**: Time-series data in CSV format:

- Timestamp, instance count, cost accumulation
- Token usage over time
- Instance state transitions
- Useful for cost analysis and optimization

## 4.11 Resource Management

**Startup Validation**:

- Check 20GB free disk space before starting
- Verify Docker daemon accessibility
- Clean up orphaned containers from previous runs
- Validate git repository and base branch

**Container Cleanup**: On startup, the orchestrator:

1. Lists all containers matching naming pattern
2. Checks against known run IDs
3. Removes orphaned containers older than retention period
4. Logs cleanup actions for audit trail

**Parallel Shutdown**: On Ctrl+C or error:

1. Set shutdown flag to prevent new instances
2. Cancel all pending instance tasks
3. Stop all running containers in parallel
4. Save final state snapshot
5. Clean up temporary resources

The parallel container stopping significantly improves shutdown speed when many instances are running.

# 5. TUI Component

## 5.1 Overview and Responsibilities

The TUI is how users experience the orchestrator. While the underlying system manages complex parallel executions, the TUI's job is to make that complexity understandable at a glance. Built with Rich, it provides a real-time dashboard during execution and clean CLI output for both interactive and non-interactive use.

Core responsibilities:

- Display real-time progress of all instances and strategies
- Adapt visualization based on scale (5 vs 500 instances)
- Provide CLI interface for launching runs
- Show aggregate metrics and individual instance details
- Stream important events to console in non-TUI mode
- Present final results and branch information

The TUI knows nothing about how instances run or how strategies work - it simply subscribes to events and queries state. This separation means you could replace it entirely with a web UI without touching orchestration logic.

**Multi-UI Support**: Multiple UIs can monitor the same run by tailing the shared `events.jsonl` file. An optional HTTP server provides read-only endpoints for web clients:

- `GET /state` - Returns current state.json
- `GET /events?since=<offset>&limit=<n>` - Streams events from file

The HTTP server starts with `--http-port <port>` flag and enables web dashboards or external monitoring tools to observe runs in progress.

## 5.2 Display Architecture

The TUI uses a hybrid approach combining event streams with periodic state polling for both responsiveness and reliability:

**Event Handling**: The TUI watches `events.jsonl` for new events using file notification APIs (inotify/kqueue). As new events are appended to the file, the TUI reads and processes them, updating its internal state immediately. However, events don't trigger renders directly. Whether 1 event or 100 events arrive, they simply update the internal data structures. This prevents display flooding when hundreds of instances generate rapid events.

**Fixed Render Loop**: A separate render loop runs at exactly 10Hz (every 100ms), reading the current internal state and updating the display. This decoupling is crucial - event processing and display rendering operate independently. Users see smooth, consistent updates regardless of event volume.

**State Reconciliation**: Every 3 seconds, the TUI polls orchestration's `get_current_state()` to reconcile any drift. This isn't the primary update mechanism - just a safety net catching missed events or recovering from disconnections. The infrequent polling avoids unnecessary load while ensuring long-running displays stay accurate.

**Buffer Management**: Events update a simple in-memory representation:

- Instance map keyed by ID with current status
- Strategy progress counters
- Aggregate metrics accumulation
- Last-updated timestamps for staleness detection

**Why This Architecture**: Pure event-driven displays suffer from flooding and missed events. Pure polling feels sluggish and wastes resources. This hybrid leverages events for responsiveness while maintaining predictable render performance and reliability. The 10Hz update rate is fast enough to feel real-time but sustainable even with hundreds of instances.

The separation of concerns is clean: events flow from file watching, polling ensures correctness, rendering happens on its own schedule. This pattern emerged from experience - it handles both the 5-instance demo and the 500-instance production run equally well.

## 5.3 Layout Design

The dashboard uses a three-zone layout maximizing information density:

**Header Zone**: Single-line header showing run-level information:

- Run ID with emoji indicator
- Strategy name and configuration
- Model being used
- Instance counts (running/completed/failed)
- Total runtime

**Dashboard Zone**: Dynamic grid adapting to instance count:

- Each strategy gets a bordered section
- Instances within strategies shown as cards
- Card size varies based on total count (detailed → compact → minimal)
- Visual grouping makes strategy boundaries clear

**Footer Zone**: Fixed 2-line footer with aggregate metrics:

- Total runtime, accumulated cost, token usage
- Instance counts (running/completed/failed)
- Critical warnings or errors

The layout prioritizes current activity. Completed instances fade visually while active ones draw attention. This focus on "what needs attention now" helps users manage large runs.

## 5.4 Adaptive Display

The display intelligently adapts to instance count, showing maximum useful information without clutter:

**Detailed Mode (1-10 instances)**:

- Large cards with full Claude Code activity feed
- Live progress bars based on task completion
- Token usage breakdown (input/output/total)
- Real-time cost accumulation
- Lines of code changed, commits made
- Full error messages if failed

**Compact Mode (11-50 instances)**:

- Medium cards with current status line only
- Simplified progress indicator
- Cost and runtime prominently displayed
- Status shown as emoji (🔄 running, ✅ complete, ❌ failed)

**Dense Mode (50+ instances)**:

- Minimal cards in tight grid
- Just ID, status emoji, cost, and runtime
- Color coding for quick status scanning
- Strategy-level progress more prominent
- Summary stats matter more than individuals

The adaptation is automatic but overridable. Users can force detail level if monitoring specific instances closely. This flexibility handles both overview monitoring and detailed debugging.

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
- `--quiet`: Only show final results

**Strategy Configuration**:

- `--runs`: Number of parallel strategy executions
- `-S key=value`: Set strategy-specific parameters (repeatable)
- `--config`: Load configuration from YAML or JSON file

Examples:

```bash
# Single execution with parameters
orchestrator "fix bug" --strategy best-of-n -S n=3 -S scorer_prompt="evaluate correctness"

# Multiple parallel executions
orchestrator "refactor" --strategy best-of-n --runs 5 -S n=3

# Using config file
orchestrator "build feature" --config prod-strategy.yaml

# Override config values
orchestrator "optimize" --config base.yaml --runs 10 -S timeout=600
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

# Connect to remote orchestrator
python -m src.tui --connect localhost:8080
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

# HTTP server
ORCHESTRATOR_HTTP_PORT=8080

# Debug settings
ORCHESTRATOR_DEBUG=true

# Strategy parameters
ORCHESTRATOR_STRATEGY__BEST_OF_N__N=5
ORCHESTRATOR_STRATEGY__BEST_OF_N__SCORER_PROMPT="..."

# TUI settings
ORCHESTRATOR_TUI__REFRESH_RATE=100
ORCHESTRATOR_TUI__FORCE_DISPLAY_MODE=detailed

# Model configuration
ORCHESTRATOR_DEFAULT_MODEL=sonnet
```

**Configuration Merging**: Complex deep merge with recursive dictionary merging for nested configurations. Arrays are replaced, not merged.

## 6.2 Logging System

Logging serves both debugging and audit purposes without cluttering the user experience:

**Structured JSON Logs**: All components emit structured logs with consistent fields:

- `timestamp` - ISO 8601 with milliseconds
- `component` - Origin (runner/orchestration/tui)
- `level` - Standard levels (debug/info/warn/error)
- `run_id` - Correlation across entire run
- `instance_id` - When applicable
- `message` - Human-readable description
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
- File size-based rotation with configurable backup count
- Async rotation tasks to prevent blocking

## 6.3 Security and Isolation

Security focuses on protecting the host system and preventing instance interference:

**Container Isolation**: Each instance runs in a Docker container with:

- No privileged access
- Three volume mounts only: isolated workspace, /home/node persistent volume, and /tmp tmpfs
- Entire container filesystem read-only except the mounted volumes
- Container-specific network namespace with outbound internet access
- Resource limits preventing system exhaustion
- Non-root user execution

**API Key Handling**: Authentication tokens are:

- Never logged or displayed
- Passed via environment variables to containers
- Validated on startup with clear errors
- Support for both OAuth tokens (subscription) and API keys

**File System Boundaries**: Instances cannot:

- Access the original host repository or any host paths except their isolated workspace at `/tmp/orchestrator/<run>/<instance>`
- See or modify the original repository
- Read other instances' working directories
- Escape their container boundaries
- Access any system paths outside their designated mounts

**Git Safety**: The isolated clone approach ensures:

- Host repository never exposed to containers
- Each instance works in completely separate git repository
- No shared git objects or metadata between instances
- Physical file copying with `--no-hardlinks` prevents inode-level interference
- Branch imports happen after container exits, eliminating concurrent access

These measures provide reasonable security without enterprise complexity. We're protecting against accidents and basic isolation violations, not nation-state attackers. The complete workspace isolation is the key defense - containers literally cannot access the host repository or other instances' work.

## 6.4 Resource Management

Resource management keeps things simple and functional:

**Parallel Instance Limit**: Single configuration value `max_parallel_instances` (default 20). Prevents system overload and API rate limit issues. No complex scheduling or priorities - just a simple pool.

**Container Resources**: Fixed defaults that work well:

- 2 CPUs per container
- 4GB memory per container
- No disk quotas (rely on system capacity)

Override only if needed via `container_cpu` and `container_memory` settings. Most users never touch these.

**Cost Tracking**: Simple accumulation from Claude Code's reported costs. No budgets or limits for now - users can monitor costs in real-time and cancel if needed. Complex budget enforcement can come later if demanded.

**Timeouts**: Two-level timeout system:

- Claude Code's internal max_turns limit (prevents excessive API calls)
- Container-level timeout (default 1 hour) as a fallback safety net

This dual approach catches both API-level runaway usage and container-level hangs. Both are configurable but defaults work for most cases.

The philosophy: provide essential limits that prevent system failure, avoid complex resource management that users won't use.

## 6.5 Data Persistence

Persistence focuses on essential data for resumability and results:

**Event Log**: Append-only file with all events:

- Enables state reconstruction after crashes
- Provides audit trail of execution
- Simple JSONL format for easy parsing
- Contains workspace paths before deletion for audit
- No database, no complexity

**State Snapshots**: Periodic JSON dumps of current state:

- Every 30 seconds during execution
- Contains full state and last applied event offset
- Speeds up recovery (replay fewer events from saved offset)
- Includes all instance states and metrics
- Overwritten each time (only need latest)

**File Organization**: Data is organized per-run:

```
logs/
  run_20250114_123456/
    events.jsonl       # Events for this run only
    state.json         # Final state snapshot with event offset
    orchestration.log  # Component logs
    runner.log
    tui.log

results/
  run_20250114_123456/
    summary.json       # Overview with metrics
    branches.txt       # List of created branches
    strategy_output/   # Strategy-specific data
```

**Event Ordering**: Critical events must be emitted before cleanup operations:

- Instance completion events include workspace paths
- Events are flushed to disk before workspace deletion
- Ensures audit trail remains complete for reconstruction

**Session Preservation**: Claude Code sessions are preserved through container naming. If orchestrator crashes, containers remain for resume. Cleanup only happens for old orphaned containers.

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
orchestrator "build API" --strategy best-of-n --runs 3 -S n=5 -S scorer_prompt="evaluate performance"
```

Parameters are validated immediately with clear errors for missing required options.

**Pre-flight Checks**: Before spawning any instances, the orchestrator verifies:

- Docker daemon is running and accessible
- Repository exists and has requested base branch
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

- Real-time instance status with visual grouping by strategy
- Live cost accumulation (critical for budget awareness)
- Progress indicators based on Claude's task completion
- Clear failure indication with error snippets

The dashboard adapts to run scale automatically. Users don't configure view modes - it just works whether running 3 instances or 300.

**Non-TUI Monitoring**: When TUI is disabled, progress streams to console:

```
7a3f2: Started → feat_auth_1
7a3f2: Implementing authentication module...
7a3f2: Created auth/jwt.py with JWT handler
7a3f2: Completed ✓ 3m 24s • $0.42 • 2.5k tokens

8b4c1: Started → reviewing feat_auth_1  
8b4c1: Analyzing implementation quality...
8b4c1: Metadata → score=8.5, feedback="Good error handling"
8b4c1: Completed ✓ 45s • $0.12 • 0.8k tokens

9d5e3: Failed ✗ Timeout after 60m • $2.41
```

Clean prefixes, instance correlation, key metrics. No timestamp spam or debug noise.

**Remote Monitoring**: The event log enables remote monitoring:

```
tail -f logs/run_20250114_123456/events.jsonl | jq
```

Teams can build custom dashboards or alerts from this stream. The orchestrator doesn't prescribe monitoring solutions.

**Health Indicators**: Key metrics visible at all times:

- Queue depth (waiting instances)
- Active instance count vs limit
- Total cost accumulation rate
- Average instance duration

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
- Git Error: Push conflicts, repository issues

The Instance Runner reports failure type, but doesn't interpret it. That's the strategy's job.

**Cancellation vs Failure**

- Async cancellations caused by orchestrator-initiated shutdown (e.g., Ctrl+C) are treated as interruption, not failure.
- On graceful shutdown, instances SHOULD be recorded as `INTERRUPTED` (or equivalently flagged as interrupted) rather than `FAILED`.
- Strategies MUST NOT treat shutdown-induced interruptions as semantic task failures.

**Strategy-Level Handling**: Strategies receive failed results and decide responses:

- BestOfNStrategy: Continues with successful instances, ignores failures
- ScoringStrategy: Fails entirely - can't score nonexistent code
- Custom strategies: Implement domain-specific recovery

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

1. Stop all running containers (preserving their state)
2. Save current orchestration state snapshot
3. Record which instances were interrupted
4. Display resume command
5. Exit quickly (typically under 10 seconds for 20 containers)

Users see: `Run interrupted. Resume with: orchestrator --resume run_20250114_123456`

The key insight: stopping containers preserves their state. When resumed, instances continue from their last checkpoint. No work is lost, but shutdown is responsive. Users pressing Ctrl+C want immediate response, not graceful completion.

**Container Stop vs Kill**: Containers are stopped, not killed:

- `docker stop` sends SIGTERM, allowing clean shutdown
- 10-second grace period before SIGKILL
- Containers stopped in parallel to minimize total time
- Session remains resumable after stop

This balance ensures quick shutdown while maximizing resumability.

**Interrupted State Representation**

- The system MUST persist which instances were interrupted. Two equivalent approaches are permitted:
  - Add `INTERRUPTED` to the instance state model, or
  - Persist an `interrupted_at` timestamp and/or an explicit list of interrupted instance IDs in `state.json` while leaving instance state as `RUNNING`.
- In both approaches, graceful shutdown MUST NOT mark instances `FAILED` solely due to cancellation.

**Crash vs Graceful Interruption**

- In a hard crash or power loss, interruption markers may not be written. On resume, any instance that is `RUNNING` in `state.json` and has no terminal state event at or after `last_event_offset` MUST be treated as "interrupted by crash" and handled under the Interrupted rules.

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

**Immediate Finalization on Resume**

- If after restoration and verification there are no runnable instances (all are terminal or cannot be resumed), the orchestrator SHOULD immediately finalize the run, emit `run.completed` with a summary (`resumed`, counts for `cannot_resume`, etc.), and exit.

Notes on Sessions:

- Each instance's `session_id` MUST be persisted in `state.json`. Resume MUST rely on this persisted `session_id` rather than event replay to determine resumability. If event replay does occur and includes newer `session_id` updates, those MUST overwrite the in-memory value before any resume checks.

Special cases:

- Plugin without session support: Strategy can choose to restart fresh or fail
- Corrupted session: Detected by plugin, treated as failed with `session_corrupted` error
- User can pass `--resume-fresh` to restart interrupted instances without sessions

**State Reconstruction**: Resume follows a simple sequence:

1. Load run snapshot (instance states, strategy progress) from `state.json`
2. Read events from saved offset in `events.jsonl`
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

## 7.8 Snapshot Atomicity

All snapshots MUST be written atomically using a temp-file-then-rename pattern to avoid partial/corrupt files:

- Primary snapshot: `orchestrator_state/<run_id>/state.json`
- Duplicate snapshot (for convenience and multi-UI): `logs/<run_id>/state.json`

Both MUST be written via a temporary file (e.g., `state.json.tmp`) and then `rename()`d to the final path so readers never observe partial JSON. Readers SHOULD prefer the primary snapshot but MAY fall back to the duplicate if the primary is missing.

## 7.5 Results and Reporting

Results presentation focuses on actionability - what branches were created and how to evaluate them:

**Summary Display**: At run completion, users see:

```
═══ Run Complete: run_20250114_123456 ═══

Strategy: best-of-n
  n: 2
  scorer_prompt: "evaluate code quality, architecture, and completeness"
  
Runs: 3

Results by strategy:

Strategy #1 (best-of-n):
  ✓ bestofn_20250114_123456_1_1  3m 12s • $0.38 • 2.3k tokens
    metadata: score=7.5, complexity=medium, test_coverage=85%
  ✓ bestofn_20250114_123456_1_2  3m 45s • $0.41 • 2.5k tokens  
    metadata: score=8.5, complexity=low, test_coverage=92%
  → Selected: bestofn_20250114_123456_1_2

Strategy #2 (best-of-n):
  ✓ bestofn_20250114_123456_2_1  4m 23s • $0.52 • 3.1k tokens
    metadata: score=9.0, complexity=high, test_coverage=78%
  ✗ bestofn_20250114_123456_2_2  Failed: timeout after 3 retries
  → Selected: bestofn_20250114_123456_2_1

Strategy #3 (best-of-n):  
  ✓ bestofn_20250114_123456_3_1  3m 56s • $0.44 • 2.7k tokens
    metadata: score=8.0, complexity=medium, test_coverage=88%
  ✓ bestofn_20250114_123456_3_2  3m 33s • $0.39 • 2.4k tokens
    metadata: score=7.0, complexity=low, test_coverage=90%
  → Selected: bestofn_20250114_123456_3_1

Summary:
  Total Duration: 8m 45s
  Total Cost: $2.14
  Success Rate: 5/6 instances (83%)
  
Final branches (3):
  bestofn_20250114_123456_1_2
  bestofn_20250114_123456_2_1  
  bestofn_20250114_123456_3_1

Full results: ./results/run_20250114_123456/
```

**Results Directory**: Structured output for further analysis:

- `summary.json` - Machine-readable version of above
- `branches.txt` - Simple list for scripting
- `metrics.csv` - Instance-level metrics for analysis
- `strategy/` - Strategy-specific outputs (scores, feedback)

**Branch Comparison**: Practical next steps are suggested:

```
To compare implementations:
  git diff main..bestofn_20250114_123456_1_2
  git checkout bestofn_20250114_123456_3_1
  
To merge selected solution:
  git merge bestofn_20250114_123456_2_1
```

**Export Formats**: Results can be exported for different audiences:

- JSON for tooling integration
- Markdown for documentation
- CSV for spreadsheet analysis

No complex reporting engine - just structured data users can process with familiar tools.

The results philosophy: make it clear what was produced and how to use it. The orchestrator's job ends at branch creation - users decide what to merge.

## 7.6 Platform Support

**Primary Platform**: Linux is the primary development and deployment platform with full feature support.

**macOS Support**: Fully supported via Docker Desktop. The SELinux `:z` flag is automatically excluded on macOS. Path length limitations are generally not an issue. Requires Docker Desktop installation.

**Windows Support**:

- Recommended: Run via WSL2 (Windows Subsystem for Linux) for best compatibility
- Alternative: Native Docker Desktop works but has limitations:
  - Path length restrictions (260 character limit)
  - Different path separators may cause issues in some edge cases
  - Performance is typically slower than WSL2

**Platform Detection**: The system automatically detects the platform and adjusts behavior:

- SELinux flags only applied on Linux systems with SELinux enabled
- Path handling adjusted for Windows when necessary
- File watching uses appropriate APIs (inotify on Linux, kqueue on macOS, polling on Windows)

**Minimum Requirements**:

- Docker 20.10 or newer (Docker Desktop on macOS/Windows)
- Git 2.25 or newer
- Python 3.11 or newer (3.13 recommended)
- 8GB RAM minimum, 16GB recommended for parallel execution
- 20GB free disk space for Docker images and temporary workspaces

## 7.7 Extended Features

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

- 20+ event types including workspace lifecycle and Claude-specific events
- Event filtering by run_id, instance_id, and type
- Ring buffer implementation to prevent memory issues
- Monotonic offset tracking for reliable position management

**Comprehensive Error Handling**:

- Pattern-based retry with specific error string matching
- Additional error types: OSError, IOError, asyncio.CancelledError
- Sophisticated retry configuration beyond basic exponential backoff
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

These extensions emerged from real-world usage and make the orchestrator production-ready while maintaining the clean architecture described in this specification.
