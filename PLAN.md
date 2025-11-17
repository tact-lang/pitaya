# Pitaya → Codex 0.58.0 Plan

Pitaya is still early‑stage, so we don’t need backwards compatibility with older Codex CLIs. The goal is for Pitaya’s Codex integration to look **as if it was designed for 0.58.0 from day one**:

- Clean, single‑path integration with `codex exec` in JSON mode.
- No legacy parsing, no feature flags, no conditional behavior for older versions.
- CLI flags, sandboxing, and auth that match Codex 0.58.0 docs and internal types.

---

## 1. Target architecture (post‑migration)

This section defines the **end state** we want. The implementation steps in section 3 describe how to get there.

### 1.1 How Pitaya should invoke Codex 0.58.0

Relevant code:

- `src/instance_runner/plugins/codex.py`
- `Dockerfile`

Desired behavior:

- **Invocation shape**
  - Base command (fixed):
    - `["codex", "exec", "--json", "-C", "/workspace"]`
  - Sandbox / approvals (fixed for Pitaya):
    - `["--skip-git-repo-check", "--sandbox", "workspace-write", "--ask-for-approval", "never"]`
  - Model selection:
    - `-m <model>` when Pitaya specifies a model.
    - Optional `-c` config overrides to define `model_provider` and point at custom endpoints (e.g. OpenRouter).
  - Prompt:
    - Single positional `PROMPT` argument.
    - No stdin prompts in the Pitaya integration (we always pass a prompt string).
  - Resume:
    - No Codex‑level resume wire‑up from Pitaya initially.
    - Pitaya “resume runs” remains based on its own run IDs, logs, and state.

- **Auth / configuration**
  - Primary credential: `CODEX_API_KEY` inside the container.
  - Backwards compatibility with the wider ecosystem via `OPENAI_API_KEY`:
    - Pitaya maps `AuthConfig.api_key` → both `CODEX_API_KEY` **and** `OPENAI_API_KEY`.
  - Base URL:
    - Pitaya maps `AuthConfig.base_url` → `OPENAI_BASE_URL`.
    - Advanced users can override via `CODEX_BASE_URL` (environment on the host).

- **Sandbox posture**
  - Pitaya assumes Docker is the outer sandbox, but Codex 0.58.0’s sandbox is still valuable for tool policy and approvals.
  - We choose:
    - `--sandbox workspace-write` (Codex may edit files under `/workspace`, matching Pitaya’s isolated clone).
    - `--ask-for-approval never` (no interactive prompts inside the container).

### 1.2 How Pitaya should parse Codex 0.58.0 JSONL

Relevant code:

- `src/instance_runner/codex_parser.py`
- `CodexPlugin.parse_events` / `extract_result`

Desired behavior:

- Expect **only** Codex 0.58.0 `--json` output:
  - JSON Lines, 1 event per line.
  - Event schema defined by `codex-rs/exec/src/exec_events.rs`.
- Supported top‑level `type` values:
  - `thread.started`
  - `turn.started`
  - `turn.completed`
  - `turn.failed`
  - `item.started`
  - `item.updated`
  - `item.completed`
  - `error`
- Supported `item.details.type` values:
  - `agent_message`
  - `reasoning`
  - `command_execution`
  - `file_change`
  - `mcp_tool_call`
  - `web_search`
  - `todo_list`
  - `error`

Internal mapping:

- `thread.started`
  - Capture `thread_id` as our `session_id`.
  - Emit an internal “system/init” event for observability.
- `turn.started`
  - Optional “turn_started” event (used mainly for metrics and TUI).
- `turn.completed`
  - Emit an internal `"turn_complete"` event with metrics from `usage`:
    - `input_tokens = usage.input_tokens + usage.cached_input_tokens`
    - `output_tokens = usage.output_tokens`
    - `total_tokens = input_tokens + output_tokens`
  - Maintain running maxima so multiple turns aggregate sensibly.
- `turn.failed`
  - Emit an internal `"error"` event with the embedded `ThreadErrorEvent`.
  - Record `last_error` for `extract_result()`.
- `item.*`:
  - `agent_message`:
    - For `item.completed`, emit `"assistant"` events with `content = text`.
    - Update `last_message` (used as `final_message`).
  - `reasoning`:
    - Emit `"assistant"` events with content prefixed, e.g. `[reasoning] …` (visible but non‑critical).
  - `command_execution`:
    - `item.started` → `"tool_use"` with `tool="bash"`, `command`.
    - `item.completed` → `"tool_result"` with `success` (from status), `exit_code`, and truncated `aggregated_output`.
  - `file_change`:
    - `item.completed` → `"tool_result"` representing a batch edit (tool `"Edit"`, list of changed files).
  - `mcp_tool_call`:
    - Map to `"tool_use"` / `"tool_result"` for a generic `"mcp"` tool.
    - Include server/tool identifiers and either `result` or `error`.
  - `web_search`:
    - Optional `"tool_use"` for `"web_search"` (can start as internal/no‑op if we don’t expose external search).
  - `todo_list`:
    - Optionally surface as “plan” events later; for now we can ignore them at the orchestrator level while leaving room to extend.
  - `error`:
    - Emit `"error"` events; they do not abort runs by themselves unless coupled with `turn.failed`.
- `error` (top level):
  - Emit an internal `"error"` event and set `last_error`.

Summary extraction:

- `session_id`:
  - From `thread.started.thread_id` (last seen).
- `final_message`:
  - Last `agent_message` text from `item.completed`.
- `metrics`:
  - `input_tokens`, `output_tokens`, `total_tokens` from cumulative `turn.completed` usage.
- `error`:
  - If any `turn.failed` or top‑level `error` was seen, use its message.

The parser does **not** need to support older Codex event formats; we assume Codex 0.58.0+ is always used.

---

## 2. Container and packaging

Target state:

- `Dockerfile` installs **only** the current Codex version:
  - `npm install -g @openai/codex@0.58.0`
- The image contains:
  - `codex` binary.
  - `codex-linux-sandbox` (from the npm package) in PATH.
  - System tools required by Codex for sandboxing and execution (bash, git, etc.).
- Runtime user:
  - `node` user owns `/workspace`.
  - Directories are not world‑writable (avoids Codex sandbox warnings).

Checks to enforce:

- `codex --version` prints `0.58.0`.
- `codex exec --help` shows `--json`, `--sandbox`, `--ask-for-approval`, `--output-schema`, etc.
- `codex exec --json -C /workspace "echo test"` runs non‑interactively and emits valid ThreadEvent JSONL.

---

## 3. Implementation steps

This is an ordered list of concrete changes to make Pitaya match the target architecture above.

### 3.1 Update Docker image

1. In `Dockerfile`:
   - Change `@openai/codex@0.42.0` → `@openai/codex@0.58.0`.
   - Keep Claude Code as‑is.
2. Rebuild `pitaya-agents:latest`.
3. Manually validate Codex inside the container:
   - `codex --version`
   - `codex exec --json -C /workspace "echo ok"` (with `CODEX_API_KEY` set).

### 3.2 Update Codex plugin CLI command

Edit `src/instance_runner/plugins/codex.py`:

1. **Base command & sandbox flags**
   - Keep `_BASE_COMMAND = ["codex", "exec", "--json", "-C", "/workspace"]`.
   - Replace `_SANDBOX_FLAGS` with:
     - `["--skip-git-repo-check", "--sandbox", "workspace-write", "--ask-for-approval", "never"]`.
2. **Remove legacy resume flag**
   - Delete the `--resume` handling in `build_command`.
   - If a `session_id` is provided today, ignore it in the Codex plugin (we can reintroduce Codex‑level resume later using the `resume` subcommand).
3. **Refine system prompt handling**
   - Remove hard‑coded `--system-prompt` unless we confirm Codex 0.58.0 supports it.
   - If we want to influence “developer instructions” or similar, pass them via `-c` config overrides or `agent_cli_args` instead of undocumented flags.
4. **Keep provider mapping**
   - Keep `-c model_provider=…` and `model_providers.<label>={...}` overrides.
   - Keep the provider env selection logic (`CODEX_ENV_KEY`, `OPENAI_API_KEY`, OPENROUTER, etc.).

### 3.3 Update Codex environment preparation

Edit `_collect_codex_env` in `src/instance_runner/plugins/codex.py`:

1. For `auth_config.api_key`:
   - Set both `CODEX_API_KEY` and `OPENAI_API_KEY` if not already present.
2. For `auth_config.base_url`:
   - Keep setting `OPENAI_BASE_URL`.
3. Preserve host‑defined provider envs and overrides (OPENROUTER, AZURE, etc.).

### 3.4 Rewrite `CodexOutputParser` for 0.58.0 ThreadEvents

Edit `src/instance_runner/codex_parser.py`:

1. **Simplify assumptions**
   - Assume every JSON line is either:
     - A valid ThreadEvent (preferred), or
     - Malformed/noisy, which we ignore.
   - We no longer parse older `{"msg": {"type": ...}}` internal events; the code can still be robust to an outer `{"id": ..., "msg": {...}}` envelope if it appears, but the “type” vocabulary is 100% ThreadEvents.
2. **Event routing**
   - Inspect `obj["type"]` to distinguish:
     - `thread.started`, `turn.*`, `item.*`, `error`.
   - For `item.*`, inspect `item["details"]["type"]` to map to one of the internal tool/assistant event types.
3. **Metrics**
   - Drop token parsing from ad‑hoc `token_count` events.
   - Use only `turn.completed.usage` for `tokens_in` / `tokens_out` / `total_tokens`.
4. **Final summary**
   - Store:
     - `session_id` from `thread.started`.
     - `last_message` from the last `agent_message` item.
     - Metrics from `turn.completed`.
     - `last_error` from `turn.failed` or top‑level `error`.
   - `get_summary()` should return this state directly; no legacy fields.
5. **Internal event mapping**
   - Map thread/turn/item events into the internal event bus shapes used by the instance runner (`assistant`, `tool_use`, `tool_result`, `turn_complete`, `error`), as described in section 1.2.

### 3.5 Auth and error handling

1. In `CodexPlugin.validate_environment`:
   - Update the error message to mention `CODEX_API_KEY` explicitly.
2. In `CodexPlugin.handle_error`:
   - Keep the error classification logic (`timeout`, `auth`, `network`, `docker`, `codex`).
   - Optionally add detection for “insufficient_quota” / rate‑limit messages if needed, but this is not required for a clean 0.58.0 integration.

### 3.6 Strategy / docs updates

1. **Docs**
   - In `README.md` and `docs/plugins.md`:
     - Update references to Codex to note that Pitaya targets Codex 0.58.0+.
     - Show example commands using:
       - `--plugin codex`
       - `--model gpt-5-codex` or `gpt-5-codex-mini`.
   - Mention that Pitaya runs Codex in non‑interactive JSON mode via `codex exec --json`.
2. **No strategy‑level hacks**
   - Strategies treat Codex like any other agent plugin; no Codex‑specific branches.
   - All Codex‑specific behavior is isolated to `CodexPlugin` and `CodexOutputParser`.

---

## 4. Testing

We only need to test against Codex 0.58.0; older versions are not supported.

### 4.1 Unit tests for `CodexOutputParser`

Add or extend tests under `tests/` to cover:

- A successful run:
  - Sample JSONL:
    - `thread.started`
    - `turn.started`
    - `item.completed` with `reasoning`
    - `item.started` + `item.completed` with `command_execution`
    - `item.completed` with `agent_message`
    - `turn.completed` with `usage`
  - Assertions:
    - `session_id` is set.
    - `final_message` matches the last agent message.
    - `metrics` match `usage`.
- A failed run:
  - JSONL includes `turn.failed` and/or `error`.
  - `extract_result()` raises `AgentError` with the right message.

### 4.2 Manual E2E runs

After updating the image and plugin:

- Run a simple read‑only task:
  - `pitaya "List all Python files in this repo" --plugin codex --strategy simple`.
- Run a write task:
  - `pitaya "Create a HELLO_CODEX.txt file and commit it" --plugin codex --strategy simple`.
- Inspect:
  - `logs/<run_id>/` to ensure events look reasonable.
  - `results/<run_id>/summary.json` to confirm metrics and final message.

---

## 5. Non‑goals

To keep this upgrade focused and clean:

- We do **not** support Codex < 0.58.0.
- We do **not** add compatibility layers for legacy Codex `--json` formats.
- We do **not** wire Codex’s own resume mechanism (`codex exec resume`) yet; Pitaya resume behavior continues to be based on its own run IDs and logs.
- We do **not** attempt to expose every new Codex feature (ghost commits, cloud tasks, etc.) through Pitaya in this iteration.

---

## 3. Detailed migration steps

### 3.1 Container / packaging changes

**Goal:** Run Codex 0.58.0 in the Pitaya Docker image, with the right runtime/tooling for `codex exec --json`.

Steps:

1. **Bump Codex CLI version in `Dockerfile`:**
   - Change `@openai/codex@0.42.0` → `@openai/codex@0.58.0`.
   - Keep the rest of the image minimal: git, curl, jq, Python 3, etc.
2. **Ensure codex-linux-sandbox availability:**
   - Codex 0.58.0 expects `codex-linux-sandbox` in PATH for unified exec / sandboxing.
   - The Rust build ships `linux-sandbox` as part of the toolchain, but npm cask packaging bundles prebuilt binaries.
   - Verify that the npm package includes `codex-linux-sandbox` and that `npm i -g @openai/codex` puts it in PATH inside the container.
   - If necessary, add a `ln -s` or `cp` step to place `codex-linux-sandbox` in a stable location.
3. **Verify runtime user & permissions:**
   - Confirm `node` user has execute permissions for `codex` and `codex-linux-sandbox`.
   - Ensure `/workspace` is owned by `node` and not world‑writable (Codex lint warns on world‑writable dirs).
4. **Smoke‑test Codex inside the image (manual step):**
   - `docker run --rm -it pitaya-agents:latest codex --version`
   - `docker run --rm -it pitaya-agents:latest codex exec --help`
   - `docker run --rm -it pitaya-agents:latest codex exec --json -C /workspace "echo hello"` (with `CODEX_API_KEY` set), verifying JSONL events and no interactive prompts.

Risk: Low–medium. A broken image breaks Codex runs entirely but is reversible by reverting the Dockerfile change.

### 3.2 Codex plugin CLI adaptation

**Goal:** Make `CodexPlugin.build_command` align with Codex 0.58.0’s CLI and best practices.

Current base command:

- `_BASE_COMMAND = ["codex", "exec", "--json", "-C", "/workspace"]`
- `_SANDBOX_FLAGS = ["--skip-git-repo-check", "--dangerously-bypass-approvals-and-sandbox"]`

Planned changes:

1. **Keep `--json` and base CLI stable:**
   - Codex 0.58.0 still uses `--json` for JSONL mode, so `_BASE_COMMAND` stays conceptually the same.
   - Confirm we always supply a prompt (or use stdin) to avoid the “No prompt provided” early exit.
2. **Revisit sandbox and approvals flags:**
   - Today we run with `--dangerously-bypass-approvals-and-sandbox`, which disables internal sandboxing and confirmations.
   - Codex 0.58.0’s documentation suggests using:
     - `--sandbox workspace-write` + `--ask-for-approval never` for “externally sandboxed” environments, or
     - `--full-auto` for safe auto‑mode with internal sandboxing.
   - Proposed Pitaya default:
     - Replace `--dangerously-bypass-approvals-and-sandbox` with a safer equivalent:
       - `--sandbox workspace-write`
       - `--ask-for-approval never`
     - Continue to set `--skip-git-repo-check` because Pitaya already ensures a clean clone.
   - Provide a configuration escape hatch:
     - Add a Pitaya config option `runner.codex_dangerous_bypass: bool` that, when true, re‑enables `--dangerously-bypass-approvals-and-sandbox` for advanced users who prefer Codex’s legacy non‑sandboxed behavior.
3. **Keep model & provider overrides:**
   - Codex still supports `--model` and `-c` overrides; the plugin’s provider mapping logic remains valid.
   - We will only revisit this after we expose new model families (section 3.5).
4. **Update resume semantics:**
   - Codex 0.58.0 uses `codex exec resume` as a subcommand, not a `--resume` flag.
   - Pitaya currently appends `["--resume", session_id]`; this may be ignored or rejected by newer builds.
   - Proposed approach:
     - For now, **disable Codex resume** from Pitaya by:
       - Removing the `--resume` flag from `build_command` and leaving Codex resume unsupported (Pitaya still supports its own “resume run” semantics via logs/results).
     - If we need Codex resume:
       - Introduce a mode where `build_command` changes the shape of the invocation to:
         - `["codex", "exec", "--json", "-C", "/workspace", "resume", session_id, prompt]`
       - This is more invasive because it changes positional arguments; we can defer it until we have a concrete use case.
5. **System prompt and extra args:**
   - Codex’s public CLI docs do not expose a `--system-prompt` flag.
   - For compatibility, keep passing `system_prompt` through `agent_cli_args` rather than a hard‑coded flag (or remove `--system-prompt` entirely if it is not recognized by 0.58.0).

Concrete changes to implement:

- In `src/instance_runner/plugins/codex.py`:
  - Adjust `_SANDBOX_FLAGS` to:
    - Default: `["--skip-git-repo-check", "--sandbox", "workspace-write", "--ask-for-approval", "never"]`.
    - Optional toggle to use `--dangerously-bypass-approvals-and-sandbox` via config.
  - Remove `--resume` from `build_command` for now (log a debug message when `session_id` is provided and is ignored).

### 3.3 Codex JSONL parser upgrade

**Goal:** Parse Codex 0.58.0’s `ThreadEvent` JSONL events directly, while remaining tolerant to older formats.

Current parser expectations:

- `type` is a simple string (`"agent_message"`, `"tokens"`, etc.).
- Usage events are separate from completion events.
- Final result is inferred from the last `task_complete`/`shutdown_complete` or last assistant message.

Codex 0.58.0 JSONL (`exec_events.rs`):

- Top‑level `ThreadEvent` with `type`:
  - `thread.started`
  - `turn.started`
  - `turn.completed`
  - `turn.failed`
  - `item.started`
  - `item.updated`
  - `item.completed`
  - `error`
- Token usage lives only in `turn.completed.usage` (`input_tokens`, `cached_input_tokens`, `output_tokens`).
- Assistant text is in `item.completed.item.details.AgentMessage.text`.
- Commands, MCP tools, web search, TODO lists are all typed via `item.*` events.

Planned parser changes:

1. **Preserve legacy tolerance, but prefer new schema:**
   - Keep the existing “legacy” code path for JSON objects that are not clearly `ThreadEvent`‑shaped.
   - Add an early check: if `type` matches `thread.started`, `turn.*`, `item.*`, or `error` and the shape matches `exec_events.rs`, handle via a new path.
2. **Introduce a new parsing path for `ThreadEvent`:**
   - For every input line:
     - Attempt to interpret it as a **ThreadEvent**:
       - `type` exactly matches one of the known strings.
       - `item` (if present) contains `details.type`.
     - If successful, map to Pitaya events:
       - `thread.started` → internal “system/init” event; capture `thread_id` as `session_id`.
       - `turn.started` → internal no‑op or “turn_started” event (optional).
       - `turn.completed` → internal `"turn_complete"` event with tokens:
         - `input_tokens` = `usage.input_tokens + usage.cached_input_tokens`
         - `output_tokens` = `usage.output_tokens`
         - `total_tokens` = sum or tracked max.
       - `turn.failed` → internal `"error"` event, storing `error.message`.
       - `item.*`:
         - `agent_message` → `"assistant"` event; update `last_message`.
         - `reasoning` → internal `"assistant"` or `"tool_use"` (we can log as assistant with a `[reasoning]` prefix to keep it visible but non‑blocking).
         - `command_execution`:
           - `item.started` → `"tool_use"` with `tool="bash"`, `command=<command>`.
           - `item.completed` → `"tool_result"` with `success`, `exit_code`, `aggregated_output` (truncated).
         - `file_change`:
           - Map to `"tool_use"` / `"tool_result"` for “Edit” tool as today.
         - `mcp_tool_call`:
           - Map to `"tool_use"` / `"tool_result"` for a generic `"mcp"` tool; record `server`, `tool`, and key fields of `result` or `error`.
         - `web_search`:
           - Optional `"tool_use"` for a `"web_search"` tool (for future use; can start as no‑op).
         - `todo_list`:
           - Optional mapping to an internal “plan” event if we choose to surface it; initially can be ignored to keep behavior stable.
       - `error` → internal `"error"` event, as we already do for `error` / `stream_error` in the legacy path.
3. **Update metrics handling:**
   - Instead of relying on ad‑hoc `token_count` events, rely primarily on `turn.completed.usage`.
   - Keep the legacy `token_count` path as a fallback if we ever see older event schemas.
   - Ensure `get_summary()` uses whichever of:
     - The last `turn.completed` usage.
     - Legacy `token_count` totals.
     - Or consistent fallback (0 tokens) if nothing is available.
4. **Final message selection:**
   - Prefer the last `AgentMessage` item from `item.completed` as `final_message`.
   - If none exists, fall back to:
     - Any `last_agent_message` captured in `turn.completed` (if we extend the mapping to look at that field), or
     - The legacy `last_message` computed from legacy events.
5. **Error propagation:**
   - If we see a `turn.failed` or top‑level `error` event, record `last_error` with its message.
   - `extract_result()` already raises `AgentError` when `error` is non‑empty; this behavior stays.
6. **Non‑JSON lines / metadata:**
   - Event streams may contain non‑JSON lines (tests, debug logs).
   - Keep the “ignore non‑JSON” behavior for robustness; the JSONL output mode spec states stdout is JSONL only, but being tolerant costs little.

Testing plan for the parser:

- Add dedicated tests under `tests/` to cover Codex 0.58.0 JSONL:
  - Feed sample `ThreadEvent` sequences (from `docs/exec.md` and from codex‑rs tests) into `CodexOutputParser`.
  - Assert:
    - Correct detection of `session_id`.
    - Correct accumulation of token usage.
    - Correct mapping of agent messages to `"assistant"` events and `final_message`.
    - Correct mapping of `command_execution` to `tool_use`/`tool_result`.
    - Proper error propagation for `turn.failed` and `error`.

### 3.4 Auth, providers, and configuration

Codex 0.58.0 introduces a richer auth stack:

- Uses an auth manager abstraction with keyring support and CLI auth storage modes.
- For `codex exec`, `CODEX_API_KEY` is supported as a direct override.

Pitaya’s needs are simpler:

- Run non‑interactive in a container with pre‑injected API key.

Migration actions:

1. **Environment mapping:**
   - Continue to map Pitaya’s `AuthConfig.api_key` to `OPENAI_API_KEY` (as today).
   - Optionally, also export:
     - `CODEX_API_KEY` when `AuthConfig.api_key` is set, to align with Codex docs and avoid surprises if future versions prefer it.
2. **Base URL handling:**
   - Keep mapping `AuthConfig.base_url` → `OPENAI_BASE_URL`.
   - Allow `CODEX_BASE_URL` from Pitaya configuration as an override (already supported via `CODEX_BASE_URL` env).
3. **Auth error messaging:**
   - Ensure `validate_environment()`’s error message mentions Codex 0.58.0 semantics:
     - e.g., “Missing Codex API key (CODEX_API_KEY or OPENAI_API_KEY).”

No behavioral breaking change is strictly required here; this is mostly tightening language and supporting `CODEX_API_KEY` explicitly.

### 3.5 Model and reasoning options

Codex 0.58.0 adds:

- gpt‑5‑codex‑mini.
- gpt‑5.1 family, including reasoning level controls.
- Explicit model reasoning effort and summaries.

Pitaya currently:

- Treats `model` as an opaque string forwarded to Codex.
- Has its own default model alias (`sonnet` for Claude; `openai/gpt-5` for Codex via example).

Migration steps:

1. **Document supported Codex models for Pitaya:**
   - In `README.md` or `docs/plugins.md`, add a section listing example Codex models:
     - `gpt-5-codex`
     - `gpt-5-codex-mini`
     - `gpt-5.1-*` variants as they’re exposed via Codex CLI.
2. **Strategy‑level model configuration:**
   - No code changes strictly necessary: Pitaya already allows passing `--model` to Codex plugin.
   - Optional enhancement: allow specifying Codex‑specific reasoning effort via additional `-c` overrides if needed (e.g., `model_reasoning_effort`).
3. **Avoid hidden defaults:**
   - Do not silently switch Pitaya Codex runs to gpt‑5.1; keep the existing defaults in Pitaya and let users opt into new models.

### 3.6 Testing & rollout plan

**Goal:** Ensure behavior is stable and regressions are easy to detect.

Concrete test cases:

1. **Unit tests for parser (new):**
   - Add a `tests/test_codex_parser_thread_events.py` that:
     - Feeds one full turn with:
       - `thread.started`
       - multiple `item.*` (reasoning, command_execution start/end, agent_message)
       - `turn.completed`
     - Asserts:
       - Correct `session_id`.
       - `tokens_in` / `tokens_out` from `usage` are applied.
       - The last agent message is the final message.
   - Add a `turn.failed` scenario and ensure `AgentError` is raised.
2. **Integration tests (optional but recommended):**
   - If practical in CI:
     - Build a throwaway Codex 0.58.0 image locally and run:
       - `run_instance` with `plugin_name="codex"` on a tiny test repo.
     - Assert:
       - `InstanceResult.status` is success.
       - `metrics.tokens.*` are populated.
       - `branch_name` is set when changes are made (for write scenarios).
   - If Docker is not available in CI, we can at least pipe captured JSONL logs from a local run into the parser tests.
3. **Manual end‑to‑end smoke tests:**
   - After building `pitaya-agents:latest` with Codex 0.58.0:
     - Run a read‑only query:
       - `pitaya "List all files in this repo" --plugin codex --strategy simple`.
     - Run a write scenario:
       - `pitaya "Create a HELLO_Codex.txt file and commit it" --plugin codex`.
     - Run a longer multi‑turn strategy (e.g., `best-of-n`) and monitor:
       - JSON logs under `logs/<run_id>/`.
       - Branches created under `pitaya/<strategy>/<run_id>/k<hash>`.
4. **Rollout guardrails:**
   - Add a configuration flag `runner.codex_enable_thread_events_parser`:
     - When `false`, use legacy parsing only (for debugging).
     - When `true` (default), enable the new ThreadEvent parsing path.
   - This allows quick rollback in case of unexpected shape changes in Codex JSONL.

---

## 4. Implementation order and checkpoints

To keep the migration safe and incremental:

1. **Phase 1 – Container bump (minimal risk):**
   - Update Dockerfile to use `@openai/codex@0.58.0`.
   - Rebuild `pitaya-agents:latest`.
   - Manual: verify `codex exec --json` inside the container and ensure no new interactive prompts appear for simple tasks.
2. **Phase 2 – Plugin CLI flags:**
   - Adjust `_SANDBOX_FLAGS` as per 3.2.
   - Remove `--resume` from `build_command` and log when a `session_id` is ignored.
   - Add a config toggle for `dangerously-bypass-approvals-and-sandbox` if needed.
   - Manual: run a few `pitaya --plugin codex` tasks end‑to‑end.
3. **Phase 3 – Parser upgrade + tests:**
   - Implement the ThreadEvent parsing path in `CodexOutputParser`.
   - Add unit tests covering `thread.started`, `turn.completed`, `turn.failed`, `item.*`.
   - Add a config flag gating the new parser (default on).
4. **Phase 4 – Docs & configs:**
   - Update `docs/plugins.md` (and/or README) with:
     - New Codex version.
     - Supported models and example commands.
     - Notes about `--sandbox`/`--ask-for-approval` behavior and how Pitaya uses them.
   - Update any mention of 0.42.0 in docs/scripts.
5. **Phase 5 – Optional resume support:**
   - If required, design a small, targeted change to support:
     - `codex exec resume <SESSION_ID>` mapping from Pitaya’s `session_id`.
   - Adjust `build_command` to support a separate “resume mode” command shape.
   - Add tests or docs clarifying how “resume” interacts with Pitaya run IDs.

Each phase can be rolled back independently by reverting a small set of changes.

---

## 5. Non‑goals for this migration

To keep the migration focused and low‑risk, we will **not**:

- Change Pitaya’s default models or strategy behavior (beyond what Codex itself does internally).
- Integrate new Codex features like ghost commits, cloud tasks, or MCP server management into Pitaya orchestration in this pass.
- Depend on Codex’s `history.jsonl` or internal rollout paths for run tracking (Pitaya keeps its own run state and logs).

These can be considered in future work once the basic 0.58.0 upgrade is stable.
