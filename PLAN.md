# Pitaya → Codex 0.58.0 Migration Plan

This document describes how to safely upgrade Pitaya from `@openai/codex@0.42.0` to `@openai/codex@0.58.0`, based on:

- The current Pitaya Codex integration (`src/instance_runner/plugins/codex.py`, `src/instance_runner/codex_parser.py`, `Dockerfile`).
- Codex 0.58.0 sources under `~/Coding/codex` (notably `codex-rs/exec`, `codex-rs/cli`, and `docs/exec.md`).
- Codex changelog entries from 0.44.0–0.58.0 (`tmp`).

The goal is to keep the migration low‑risk and observable, with clear fallbacks.

---

## 1. Current Pitaya–Codex coupling

### 1.1 How Pitaya invokes Codex

Relevant code:

- `src/instance_runner/plugins/codex.py`
- `Dockerfile`

Key behaviors:

- Pitaya runs Codex via:
  - Base command: `["codex", "exec", "--json", "-C", "/workspace"]`
  - Sandbox/control flags: `["--skip-git-repo-check", "--dangerously-bypass-approvals-and-sandbox"]`
  - Optional model: `-m <model>`
  - Optional config overrides: `-c model_provider=…`, `-c model_providers.<label>={…}`, `-c model="<model>"`.
  - Optional `--resume <session_id>` for session reuse.
  - Optional `--system-prompt <prompt>`.
  - Optional extra CLI args injected before the prompt (`agent_cli_args`).
- Auth:
  - Pitaya maps `AuthConfig.api_key`/`base_url` into `OPENAI_API_KEY`/`OPENAI_BASE_URL`.
  - It also passes through any host‑set provider envs (OPENROUTER_API_KEY, AZURE_*_API_KEY, etc.) and supports overrides via `CODEX_ENV_KEY`, `CODEX_BASE_URL`, `CODEX_MODEL_PROVIDER`.
- Docker:
  - The `pitaya-agents:latest` image is built from `Dockerfile`.
  - It currently installs `@openai/codex@0.42.0` globally with npm, uses `node` user, and sets `WORKDIR /workspace`.

### 1.2 How Pitaya parses Codex output

Relevant code:

- `src/instance_runner/codex_parser.py`
- `CodexPlugin.parse_events` / `extract_result`

Key expectations:

- Pitaya expects **JSONL** on stdout (one JSON object per line), matching `codex exec --json` semantics.
- It supports both:
  - Raw objects (`{"type": "...", ...}`), and
  - Wrapped objects of the form `{"id": "...", "msg": { "type": "...", ... }}` (older Codex exec format).
- It recognizes a loose set of `type` / `event` values:
  - Session / config: `session_configured`, `session_started`, `session`.
  - Usage: `token_count`, `tokens`, `usage` (with nested `info.total_token_usage` etc.).
  - Assistant output: `agent_message`, `assistant_message`, `assistant`, `message`.
  - Command execution: `exec_command_start`, `exec_command_begin`, `tool_start`, `exec_command_end`, `tool_end`.
  - File edits: `patch_apply_start/end`, `file_edit_start/end`, `write_start/end`.
  - Turn completion: `task_complete`, `complete`, `shutdown_complete`.
  - Errors: `error`, `stream_error`.
  - Any unknown `type` is mapped to a generic `assistant` event with the raw JSON snippet.
- It accumulates a simple summary:
  - `session_id`
  - `final_message`
  - `metrics.{input_tokens, output_tokens, total_tokens}`

### 1.3 What Codex 0.58.0 actually does

From the 0.58.0 sources (`codex-rs/exec` and docs):

- CLI:
  - `codex exec` is implemented in `codex-rs/exec`.
  - Non‑interactive CLI struct: `codex-rs/exec/src/cli.rs` (`Cli`).
  - `--json` is a boolean flag: in that mode, stdout must be valid JSONL and any other human‑oriented output goes to stderr.
  - Default sandbox behavior and approvals are controlled by `--sandbox` / `--full-auto` / `--dangerously-bypass-approvals-and-sandbox` and approval policy in config.
  - `--skip-git-repo-check` is still supported and disables the “must be in a git repo” constraint.
  - `-C/--cd` sets working directory; `-m/--model` still selects the model.
  - `codex exec resume` is implemented as a subcommand, not a `--resume` flag.
- JSONL format:
  - In `--json` mode, Codex emits **typed ThreadEvents** (see `codex-rs/exec/src/exec_events.rs`) with `type` strings such as:
    - `thread.started`
    - `turn.started`, `turn.completed`, `turn.failed`
    - `item.started`, `item.updated`, `item.completed`
    - `error`
  - Each event includes a nested `item` with `details.type` for domain objects:
    - `agent_message`, `reasoning`, `command_execution`, `file_change`, `mcp_tool_call`, `web_search`, `todo_list`, `error`.
  - Token usage is surfaced via `turn.completed` with an embedded `usage` structure (`input_tokens`, `cached_input_tokens`, `output_tokens`).
  - CODEx converts internal events (e.g., `ExecCommandBeginEvent`, `AgentMessageEvent`, `TaskCompleteEvent`) into these JSONL events via `EventProcessorWithJsonOutput` (`codex-rs/exec/src/event_processor_with_jsonl_output.rs`).
- Docs (`docs/exec.md`) match this:
  - Describe `--json` semantics as JSONL with `thread.started`, `turn.*`, `item.*`, `error`.
  - Emphasize default output mode vs. `--json`.
  - Confirm `codex exec resume <SESSION_ID>` and `codex exec resume --last` are the supported resume forms.

Conclusion: Pitaya’s parser currently speaks an **older internal event vocabulary**; Codex 0.58.0 speaks the **ThreadEvent JSONL schema**. The plugin’s CLI flags (`--json`, `-C`, `-m`, `--skip-git-repo-check`, `--dangerously-bypass-approvals-and-sandbox`) are all still valid, but session resume and JSON format semantics have evolved.

---

## 2. High‑level migration strategy

We want to:

1. Upgrade the container to Codex 0.58.0 (binary + node wrapper) without changing Pitaya behavior yet.
2. Adapt the Codex plugin to:
   - Use Codex 0.58.0’s `--json` ThreadEvent format explicitly.
   - Map those events cleanly into Pitaya’s internal event bus (assistant/tool/result/turn metrics).
   - Support resume via `codex exec resume` semantics instead of the old `--resume` flag.
3. Preserve a conservative sandbox posture (Codex sandbox still enabled; Pitaya’s Docker remains “outer sandbox”).
4. Make model family changes explicit (support gpt‑5‑codex, gpt‑5‑codex‑mini, gpt‑5.1 family) but **not** silently change Pitaya’s defaults unless configured.
5. Roll out behind configuration and add regression tests to prevent format drift in the future.

Implementation will be staged to ensure we always have a working fallback.

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

