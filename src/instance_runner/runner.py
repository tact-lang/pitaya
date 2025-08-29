"""
Instance Runner - Executes a single AI coding instance in isolation.

This module provides the main entry point for running instances. It knows nothing
about strategies or UI — it takes a prompt and returns a result with a branch name.
"""

import asyncio
import logging
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from . import (
    AgentError,
    DockerError,
    GitError,
    TimeoutError,
    ValidationError,
)
from .docker_manager import DockerManager
from .git_operations import GitOperations
from .plugin_interface import RunnerPlugin
from .plugins import AVAILABLE_PLUGINS
from ..shared import AuthConfig, ContainerLimits, InstanceResult, RetryConfig

logger = logging.getLogger(__name__)


def _is_retryable_error(
    error_str: str, error_type: str, retry_config: RetryConfig
) -> bool:
    """
    Check if an error is retryable based on pattern matching.

    Args:
        error_str: The error message string
        error_type: The type of error (docker, claude, timeout, etc)
        retry_config: Retry configuration with error patterns

    Returns:
        True if the error matches a retryable pattern
    """
    error_lower = error_str.lower()

    # Always retry timeouts
    if error_type == "timeout":
        return True

    # Check Docker patterns
    if error_type == "docker":
        for pattern in retry_config.docker_error_patterns:
            if pattern.lower() in error_lower:
                return True

    # Check agent patterns
    if error_type in ("agent",):
        for pattern in retry_config.agent_error_patterns:
            if pattern.lower() in error_lower:
                return True

    # Check general patterns for any error type
    for pattern in retry_config.general_error_patterns:
        if pattern.lower() in error_lower:
            return True

    return False


async def run_instance(
    prompt: str,
    repo_path: Path,
    base_branch: str = "main",
    branch_name: Optional[str] = None,
    run_id: Optional[str] = None,
    strategy_execution_id: Optional[str] = None,
    instance_id: Optional[str] = None,
    task_key: Optional[str] = None,
    container_name: Optional[str] = None,
    model: str = "sonnet",
    session_id: Optional[str] = None,
    operator_resume: bool = False,
    session_group_key: Optional[str] = None,
    event_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    startup_semaphore: Optional[asyncio.Semaphore] = None,
    timeout_seconds: int = 3600,
    container_limits: Optional[ContainerLimits] = None,
    auth_config: Optional[AuthConfig] = None,
    reuse_container: bool = True,
    finalize: bool = True,
    docker_image: Optional[str] = None,
    retry_config: Optional[RetryConfig] = None,
    plugin_name: str = "claude-code",
    system_prompt: Optional[str] = None,
    append_system_prompt: Optional[str] = None,
    # Import controls per spec
    import_policy: str = "auto",  # auto|never|always
    import_conflict_policy: str = "fail",  # fail|overwrite|suffix
    skip_empty_import: bool = True,
    network_egress: Optional[str] = None,
    max_turns: Optional[int] = None,
    allow_overwrite_protected_refs: bool = False,
    allow_global_session_volume: bool = False,
    agent_cli_args: Optional[list[str]] = None,
    force_commit: bool = False,
) -> InstanceResult:
    """
    Execute a single AI coding instance in Docker.

    This is the main entry point for the Instance Runner component. It follows
    a 6-phase execution pipeline to reliably run the agent tool in isolation.

    Args:
        prompt: The instruction for the agent tool
        repo_path: Path to host repository (not the isolated clone)
        base_branch: Starting point for the new branch (default: "main")
        branch_name: Target branch name for import (provided by orchestration)
        run_id: Run identifier for correlation (provided by orchestration)
        strategy_execution_id: Strategy execution identifier (provided by orchestration)
        instance_id: Unique identifier (auto-generated if not provided)
        container_name: Full container name including run_id (provided by orchestration)
        model: Model to use (default: "sonnet")
        session_id: Resume a previous agent session
        event_callback: Function to receive real-time events
        timeout_seconds: Maximum execution time (default: 3600)
        container_limits: CPU/memory restrictions
        auth_config: Authentication configuration (OAuth token or API key)
        reuse_container: Reuse existing container if name matches (default: True)
        finalize: Stop container after completion (default: True)
        docker_image: Docker image override (uses plugin default if None)
        retry_config: Retry configuration for transient failures
        plugin_name: Name of the agent tool plugin to use (default: "claude-code")
        system_prompt: System prompt for the AI tool
        append_system_prompt: Additional system prompt to append
        import_policy: "auto|never|always"
        import_conflict_policy: "fail|overwrite|suffix"
        skip_empty_import: Whether to skip import when no changes

    Returns:
        InstanceResult with branch name, metrics, and status
    """
    logger.info(f"run_instance called with prompt: {prompt[:50]}...")
    instance_id = instance_id or str(uuid.uuid4())
    container_limits = container_limits or ContainerLimits()
    retry_config = retry_config or RetryConfig()
    logger.info(f"Instance ID: {instance_id}")
    if not session_group_key:
        session_group_key = task_key or instance_id

    # Get plugin
    if plugin_name not in AVAILABLE_PLUGINS:
        raise ValidationError(
            f"Unknown plugin: {plugin_name}. Available: {list(AVAILABLE_PLUGINS.keys())}"
        )

    plugin_class = AVAILABLE_PLUGINS[plugin_name]
    plugin = plugin_class()

    # Use plugin's default image if not specified
    if docker_image is None:
        docker_image = plugin.docker_image
    # Model IDs are passed through as provided; no models.yaml enforcement.
    # Plugins can interpret model identifiers as they see fit.
    resolved_model_id = model

    # Validate plugin environment
    # Convert AuthConfig to dict for plugin
    auth_dict = None
    if auth_config:
        auth_dict = {
            "oauth_token": auth_config.oauth_token,
            "api_key": auth_config.api_key,
            "base_url": auth_config.base_url,
        }

    is_valid, error_msg = await plugin.validate_environment(auth_dict)
    if not is_valid:
        raise ValidationError(f"Plugin {plugin_name} validation failed: {error_msg}")

    # Retry loop with exponential backoff
    attempt = 0
    delay = retry_config.initial_delay_seconds
    last_result = None
    # Ensure optional force_import flag defaults to False (used in internal attempt API)
    force_import = False

    while attempt < retry_config.max_attempts:
        attempt += 1

        # Use session from previous attempt if retrying
        current_session_id = (
            last_result.session_id
            if last_result and last_result.session_id
            else session_id
        )

        result = await _run_instance_attempt(
            prompt=prompt,
            repo_path=repo_path,
            base_branch=base_branch,
            branch_name=branch_name,
            run_id=run_id,
            strategy_execution_id=strategy_execution_id,
            instance_id=instance_id,
            container_name=container_name,
            model=model,
            resolved_model_id=resolved_model_id,
            session_id=current_session_id,
            operator_resume=operator_resume,
            session_group_key=session_group_key,
            event_callback=event_callback,
            startup_semaphore=startup_semaphore,
            timeout_seconds=timeout_seconds,
            container_limits=container_limits,
            auth_config=auth_config,
            reuse_container=reuse_container,
            finalize=finalize,
            docker_image=docker_image,
            plugin=plugin,
            attempt_number=attempt,
            total_attempts=retry_config.max_attempts,
            system_prompt=system_prompt,
            append_system_prompt=append_system_prompt,
            retry_config=retry_config,
            force_import=force_import,
            import_policy=import_policy,
            import_conflict_policy=import_conflict_policy,
            skip_empty_import=skip_empty_import,
            task_key=task_key,
            network_egress=network_egress,
            max_turns=max_turns,
            allow_overwrite_protected_refs=allow_overwrite_protected_refs,
            allow_global_session_volume=allow_global_session_volume,
            agent_cli_args=agent_cli_args,
            force_commit=force_commit,
        )

        # Success or non-retryable error
        if result.success:
            return result

        # Check if error is retryable using pattern matching
        is_retryable = _is_retryable_error(
            result.error or "", result.error_type or "", retry_config
        )

        # Special-case: immediate Docker exec exit on resume can be a transient session issue.
        # If operator_resume and Docker error is a plain non-zero exit, fall back to a fresh run
        # by clearing session_id and disabling reuse_container for the next attempt.
        try:
            _err = (result.error or "").lower()
            if (
                not is_retryable
                and operator_resume
                and (result.error_type or "").lower() == "docker"
                and ("exited with code" in _err or "exit code" in _err)
            ):
                is_retryable = True
                # Emit a retry hint event for diagnostics
                if event_callback:
                    event_callback(
                        {
                            "type": "instance.retrying",
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "instance_id": instance_id,
                            "data": {
                                "reason": "resume_exec_exit",
                                "action": "fresh_container",
                            },
                        }
                    )
                # Force next attempt to use a fresh container and drop resume session
                reuse_container = False
                session_id = None
                # Use a short backoff for this specific case
                delay = min(delay, 1.0)
        except Exception:
            pass

        if not is_retryable:
            return result

        # Save for potential next attempt
        last_result = result

        # Don't retry if this was the last attempt
        if attempt >= retry_config.max_attempts:
            return result

        # Log retry
        logger.info(
            f"Retrying instance {instance_id} after {result.error_type} error (attempt {attempt + 1}/{retry_config.max_attempts})"
        )

        # Wait with exponential backoff
        await asyncio.sleep(delay)
        delay = min(
            delay * retry_config.exponential_base, retry_config.max_delay_seconds
        )

    # Should never reach here, but just in case
    return last_result


async def _run_instance_attempt(
    prompt: str,
    repo_path: Path,
    base_branch: str,
    branch_name: str,
    run_id: Optional[str],
    strategy_execution_id: Optional[str],
    instance_id: str,
    container_name: str,
    model: str,
    resolved_model_id: Optional[str],
    session_id: Optional[str],
    operator_resume: bool,
    session_group_key: Optional[str],
    event_callback: Optional[Callable[[Dict[str, Any]], None]],
    startup_semaphore: Optional[asyncio.Semaphore],
    timeout_seconds: int,
    container_limits: ContainerLimits,
    auth_config: Optional[AuthConfig],
    reuse_container: bool,
    finalize: bool,
    docker_image: str,
    plugin: RunnerPlugin,
    attempt_number: int,
    total_attempts: int,
    system_prompt: Optional[str],
    append_system_prompt: Optional[str],
    retry_config: RetryConfig,
    force_import: bool,
    import_policy: str,
    import_conflict_policy: str,
    skip_empty_import: bool,
    task_key: Optional[str] = None,
    network_egress: Optional[str] = None,
    max_turns: Optional[int] = None,
    allow_overwrite_protected_refs: bool = False,
    allow_global_session_volume: bool = False,
    agent_cli_args: Optional[list[str]] = None,
    force_commit: bool = False,
) -> InstanceResult:
    """Single attempt at running an instance (internal helper for retry logic)."""
    start_time = time.time()
    started_at = datetime.now(timezone.utc).isoformat()
    workspace_dir = None
    container = None
    agent_session_id = None  # Track session for error recovery

    # Set log path based on run_id and instance_id
    if run_id and instance_id:
        log_path = f"./logs/{run_id}/instance_{instance_id}.log"
    else:
        log_path = None

    # Helper to emit events
    def emit_event(event_type: str, data: Dict[str, Any]) -> None:
        """Emit an event (generic agent namespace)."""
        if not event_callback:
            return
        event_callback(
            {
                "type": event_type,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "instance_id": instance_id,
                "data": data,
            }
        )

    docker_manager: Optional[DockerManager] = None
    try:
        # Phase 1: Validation
        logger.info(f"Starting instance {instance_id} with prompt: {prompt[:100]}...")
        emit_event(
            "instance.started",
            {
                "prompt": prompt,
                "model": model,
                "attempt": attempt_number,
                "total_attempts": total_attempts,
                "is_retry": attempt_number > 1,
                "session_id": session_id,
                "log_path": log_path,
            },
        )

        # Validate inputs
        if not repo_path.exists():
            raise ValidationError(f"Repository path does not exist: {repo_path}")

        if not repo_path.is_dir():
            raise ValidationError(f"Repository path is not a directory: {repo_path}")

        if not (repo_path / ".git").exists():
            raise ValidationError(f"Not a git repository: {repo_path}")

        if not container_name:
            raise ValidationError("Container name must be provided by orchestration")

        if not branch_name:
            raise ValidationError("Branch name must be provided by orchestration")

        # Disk space check removed: proceed without enforcing a minimum.

        # Initialize components
        docker_manager = DockerManager()
        git_ops = GitOperations()

        try:
            # Initialize Docker manager
            await docker_manager.initialize()

            # Validate Docker daemon
            logger.info("Validating Docker environment...")
            if not await docker_manager.validate_environment():
                raise DockerError("Docker daemon is not accessible")
            logger.info("Docker validation successful")

            emit_event("instance.phase_completed", {"phase": "validation"})

            # Phase 2: Workspace Preparation (happens BEFORE container)
            logger.info(f"Preparing isolated workspace for branch {base_branch}")
            if startup_semaphore is not None:
                # Signal that we are waiting on startup slot
                try:
                    emit_event(
                        "instance.startup_waiting",
                        {"base_branch": base_branch},
                    )
                except Exception:
                    pass
                async with startup_semaphore:
                    emit_event(
                        "instance.workspace_preparing", {"base_branch": base_branch}
                    )
                    logger.info(
                        "Calling git_ops.prepare_workspace (startup slot acquired)..."
                    )
                    # Reuse workspace only when resuming an existing session
                    _reuse_ws = bool(operator_resume and session_id)
                    workspace_dir = await git_ops.prepare_workspace(
                        repo_path=repo_path,
                        base_branch=base_branch,
                        instance_id=instance_id,
                        run_id=run_id,
                        strategy_execution_id=strategy_execution_id,
                        container_name=container_name,
                        reuse_if_exists=_reuse_ws,
                    )
            else:
                emit_event("instance.workspace_preparing", {"base_branch": base_branch})
                logger.info("Calling git_ops.prepare_workspace...")
                # Reuse workspace only when resuming an existing session
                _reuse_ws = bool(operator_resume and session_id)
                workspace_dir = await git_ops.prepare_workspace(
                    repo_path=repo_path,
                    base_branch=base_branch,
                    instance_id=instance_id,
                    run_id=run_id,
                    strategy_execution_id=strategy_execution_id,
                    container_name=container_name,
                    reuse_if_exists=_reuse_ws,
                )
            logger.info(f"Workspace prepared at: {workspace_dir}")

            emit_event(
                "instance.workspace_prepared", {"workspace_dir": str(workspace_dir)}
            )
            emit_event("instance.phase_completed", {"phase": "workspace_preparation"})

            # Phase 3: Container Creation
            logger.info(f"Creating container {container_name}")
            emit_event(
                "instance.container_creating",
                {
                    "container_name": container_name,
                    "workspace_dir": str(workspace_dir),
                    "image": docker_image,
                },
            )

            # Prepare auth env before container creation via plugin
            emit_event(
                "instance.container_env_preparing", {"container_name": container_name}
            )
            from dataclasses import asdict

            env_vars = await plugin.prepare_environment(
                None, asdict(auth_config) if auth_config else None
            )
            try:
                emit_event(
                    "instance.container_env_prepared",
                    {
                        "env_vars_count": len(env_vars or {}),
                        "container_name": container_name,
                    },
                )
            except Exception:
                emit_event(
                    "instance.container_env_prepared",
                    {"env_vars_count": 0, "container_name": container_name},
                )

            # Create base container with plugin/environment hooks, with a hard timeout guard
            try:
                # Record timing for create_container to diagnose stalls
                _cc_start = time.time()
                emit_event(
                    "instance.container_create_call",
                    {"container_name": container_name, "image": docker_image},
                )
                logger.debug(
                    "Awaiting DockerManager.create_container (image=%s, cpus=%s, mem_gb=%s, swap_gb=%s, reuse=%s, ws=%s)",
                    docker_image,
                    container_limits.cpu_count,
                    container_limits.memory_gb,
                    container_limits.memory_swap_gb,
                    reuse_container,
                    str(workspace_dir),
                )
                container = await asyncio.wait_for(
                    docker_manager.create_container(
                        container_name=container_name,
                        workspace_dir=workspace_dir,  # Now using workspace instead of repo
                        cpu_count=container_limits.cpu_count,
                        memory_gb=container_limits.memory_gb,
                        memory_swap_gb=container_limits.memory_swap_gb,
                        run_id=run_id,
                        strategy_execution_id=strategy_execution_id,
                        instance_id=instance_id,
                        session_id=session_id,  # Plugin will handle this
                        session_group_key=session_group_key,
                        import_policy=import_policy,
                        image=docker_image,
                        auth_config=auth_config,
                        reuse_container=reuse_container,
                        extra_env=env_vars,
                        plugin=plugin,
                        network_egress=(network_egress or "online"),
                        event_callback=event_callback,
                        task_key=task_key,
                        plugin_name=getattr(plugin, "name", "claude-code"),
                        resolved_model_id=resolved_model_id,
                        allow_global_session_volume=allow_global_session_volume,
                    ),
                    timeout=60,
                )
                logger.info(
                    "Container created: name=%s id=%s (%.2fs)",
                    container_name,
                    (
                        getattr(container, "id", "unknown")[:12]
                        if container
                        else "unknown"
                    ),
                    time.time() - _cc_start,
                )
            except asyncio.TimeoutError:
                emit_event(
                    "instance.container_create_timeout",
                    {"container_name": container_name, "timeout_s": 60},
                )
                raise DockerError(
                    "Container creation phase exceeded 60s. Docker may be unresponsive; try restarting Docker Desktop."
                )
            finally:
                try:
                    if "_cc_start" in locals():
                        logger.debug(
                            "create_container finished in %.2fs (including error path)",
                            time.time() - _cc_start,
                        )
                except Exception:
                    pass

            # 'instance.container_created' is emitted by DockerManager via event_callback
            # to avoid duplicate events. Do not emit it again here.

            # Note: Environment variables were already set pre-creation via plugin

            emit_event("instance.phase_completed", {"phase": "container_creation"})

            # Optional tool verification inside container per spec operations
            # Verify only git (mandatory for import). The AI CLI is validated implicitly during execute.
            try:
                await docker_manager.verify_container_tools(container, ["git"])
            except DockerError:
                # Surface failure; git is required
                raise

            # Start heartbeat inside container for last_active tracking
            try:
                await docker_manager.start_heartbeat(container)
            except Exception:
                pass

            # Phase 4: AI Tool Execution
            logger.info(
                f"Executing {plugin.name} with model {(resolved_model_id or model)} (alias={model})"
            )
            emit_event(
                "instance.agent_starting",
                {
                    "model": model,
                    "model_id": (resolved_model_id or model),
                    "session_id": session_id,
                    "operator_resume": bool(operator_resume),
                },
            )

            # Execute via plugin interface
            # Provide provider hints to Codex when custom base URL is configured
            codex_provider_kwargs = {}
            try:
                if getattr(plugin, "name", "") == "codex":
                    purl = None
                    if auth_config and getattr(auth_config, "base_url", None):
                        purl = auth_config.base_url
                    if not purl:
                        purl = (env_vars or {}).get("OPENAI_BASE_URL")
                    if purl:
                        codex_provider_kwargs["provider_base_url"] = purl
                        codex_provider_kwargs["provider_env_key"] = "OPENAI_API_KEY"
            except Exception:
                pass

            # Choose per-attempt prompt: use continuation only when resuming with a session
            _attempt_prompt = "Continue" if (operator_resume and session_id) else prompt
            result_data = await plugin.execute(
                docker_manager=docker_manager,
                container=container,
                prompt=_attempt_prompt,
                model=(resolved_model_id or model),
                session_id=session_id,
                timeout_seconds=timeout_seconds,
                event_callback=lambda event: emit_event(
                    f"instance.agent_{event['type']}", event
                ),
                system_prompt=system_prompt,
                append_system_prompt=append_system_prompt,
                operator_resume=operator_resume,
                max_turns=max_turns,
                stream_log_path=log_path,
                **codex_provider_kwargs,
                agent_cli_args=(agent_cli_args or []),
            )

            agent_session_id = result_data.get("session_id")
            final_message = result_data.get("final_message", "")
            metrics = result_data.get("metrics", {})

            emit_event(
                "instance.agent_completed",
                {
                    "session_id": agent_session_id,
                    "metrics": metrics,
                },
            )
            # Emit generic phase name
            emit_event("instance.phase_completed", {"phase": "agent_execution"})

            # Phase 5: Result Collection (includes branch import)
            logger.info("Collecting results and importing branch")
            emit_event("instance.result_collection_started", {})

            # Optional: force a commit if there are uncommitted changes
            if force_commit and workspace_dir:
                try:
                    # Check for uncommitted changes
                    status_cmd = [
                        "git",
                        "-C",
                        str(workspace_dir),
                        "status",
                        "--porcelain",
                    ]
                    rc, out = await git_ops._run_command(status_cmd)  # type: ignore[attr-defined]
                    if rc == 0 and (out or "").strip():
                        # Stage all and commit
                        await git_ops._run_command(["git", "-C", str(workspace_dir), "add", "-A"])  # type: ignore[attr-defined]
                        msg = f"pitaya: force commit (run_id={run_id or ''} iid={instance_id})"
                        commit_cmd = [
                            "git",
                            "-C",
                            str(workspace_dir),
                            "commit",
                            "-m",
                            msg,
                        ]
                        rc2, out2 = await git_ops._run_command(commit_cmd)  # type: ignore[attr-defined]
                        if rc2 != 0:
                            logger.debug(
                                f"force-commit skipped (git commit failed): {out2}"
                            )
                        else:
                            logger.info("force-commit created a commit in workspace")
                except Exception as e:
                    logger.debug(f"force-commit error ignored: {e}")

            # Import work from workspace to host repository according to policy
            import_info = await git_ops.import_branch(
                repo_path=repo_path,
                workspace_dir=workspace_dir,
                branch_name=branch_name,
                import_policy=import_policy,
                import_conflict_policy=import_conflict_policy,
                skip_empty_import=skip_empty_import,
                task_key=task_key,
                run_id=run_id,
                strategy_execution_id=strategy_execution_id,
                allow_overwrite_protected_refs=allow_overwrite_protected_refs,
            )

            has_changes = str(import_info.get("has_changes", "false")).lower() == "true"
            final_branch = (
                import_info.get("target_branch") or branch_name
                if has_changes
                else import_info.get("target_branch")
            )

            if has_changes:
                emit_event("instance.branch_imported", {"branch_name": final_branch})
            else:
                if import_info.get("target_branch"):
                    logger.info(
                        "No new commits; created branch pointing to base branch"
                    )
                else:
                    logger.info("No changes; skipped branch import per policy")
                emit_event(
                    "instance.no_changes",
                    {"branch_name": import_info.get("target_branch")},
                )

            emit_event("instance.phase_completed", {"phase": "result_collection"})

            # Phase 6: Cleanup Decision
            # For successful instances, clean up workspace and remove container immediately
            logger.info("Instance completed successfully")

            # Calculate duration before any cleanup
            duration = time.time() - start_time

            # Emit completion event BEFORE cleanup (as per spec)
            # Note: Do not include host workspace paths in public events
            emit_event(
                "instance.completed",
                {
                    "success": True,
                    "branch_name": final_branch,
                    "duration_seconds": duration,
                    "metrics": metrics,
                },
            )

            # Emit runner-internal completion (persisted by event bus to runner.jsonl) BEFORE cleanup
            try:
                emit_event(
                    "runner.instance.completed",
                    {
                        "instance_id": instance_id,
                        "workspace_path": str(workspace_dir) if workspace_dir else None,
                        "branch_imported": final_branch,
                        "duration_seconds": duration,
                    },
                )
            except Exception:
                pass

            # Now clean up workspace for successful instances
            if workspace_dir:
                await git_ops.cleanup_workspace(workspace_dir)
                emit_event(
                    "instance.workspace_cleaned", {"workspace_dir": str(workspace_dir)}
                )

            # Remove container immediately
            if container:
                if finalize:
                    try:
                        try:
                            await docker_manager.stop_heartbeat(container)
                        except Exception:
                            pass
                        # Inform UI it was stopped, then remove
                        try:
                            emit_event(
                                "instance.container_stopped",
                                {"container_name": container_name},
                            )
                        except Exception:
                            pass
                        await docker_manager.cleanup_container(
                            container, remove_home_volume=True
                        )
                    except Exception as e:
                        logger.warning(
                            f"Failed to cleanup container {container_name}: {e}"
                        )

            emit_event("instance.phase_completed", {"phase": "cleanup_decision"})

            completed_at = datetime.now(timezone.utc).isoformat()

            # Get commit statistics if changes were made
            commit_statistics = None
            if has_changes:
                try:
                    # Get commit stats from the imported branch
                    stats_cmd = [
                        "git",
                        "-C",
                        str(repo_path),
                        "diff",
                        "--stat",
                        f"{base_branch}..{branch_name}",
                    ]
                    _, stats_output = await git_ops._run_command(stats_cmd)

                    # Parse stats to get files changed, insertions, deletions
                    import re

                    stats_match = re.search(
                        r"(\d+) files? changed(?:, (\d+) insertions?\(\+\))?(?:, (\d+) deletions?\(\-\))?",
                        stats_output,
                    )
                    if stats_match:
                        commit_statistics = {
                            "files_changed": int(stats_match.group(1) or 0),
                            "insertions": int(stats_match.group(2) or 0),
                            "deletions": int(stats_match.group(3) or 0),
                        }

                    # Get commit count
                    count_cmd = [
                        "git",
                        "-C",
                        str(repo_path),
                        "rev-list",
                        "--count",
                        f"{base_branch}..{branch_name}",
                    ]
                    _, count_output = await git_ops._run_command(count_cmd)
                    if commit_statistics and count_output.strip().isdigit():
                        commit_statistics["commit_count"] = int(count_output.strip())
                except (GitError, OSError) as e:
                    logger.warning(f"Failed to get commit statistics: {e}")

            return InstanceResult(
                success=True,
                branch_name=final_branch,
                has_changes=has_changes,
                final_message=final_message,
                session_id=agent_session_id,
                container_name=container_name,
                metrics=metrics,
                duration_seconds=duration,
                commit_statistics=commit_statistics,
                started_at=started_at,
                completed_at=completed_at,
                retry_attempts=attempt_number - 1,
                log_path=log_path,
                workspace_path=(
                    str(workspace_dir) if workspace_dir and not has_changes else None
                ),
                status="success",
                commit=import_info.get("commit"),
                duplicate_of_branch=import_info.get("duplicate_of_branch"),
                dedupe_reason=import_info.get("dedupe_reason"),
            )

        except TimeoutError as e:
            logger.error(f"Instance {instance_id} timed out: {e}")
            emit_event(
                "instance.failed",
                {
                    "error": str(e),
                    "error_type": "timeout",
                    "attempt": attempt_number,
                    "total_attempts": total_attempts,
                    "will_retry": attempt_number
                    < total_attempts,  # Timeouts are always retryable
                },
            )

            # Keep workspace for debugging on timeout
            completed_at = datetime.now(timezone.utc).isoformat()
            # Remove container on timeout (treat as failure for cleanup policy)
            if container:
                try:
                    try:
                        await docker_manager.stop_heartbeat(container)
                    except Exception:
                        pass
                    await docker_manager.cleanup_container(
                        container, remove_home_volume=False
                    )
                except Exception:
                    pass
            # Always clean up workspace; do not retain for debugging
            try:
                if workspace_dir:
                    await git_ops.cleanup_workspace(workspace_dir)
                    emit_event(
                        "instance.workspace_cleaned",
                        {"workspace_dir": str(workspace_dir)},
                    )
            except Exception:
                pass

            return InstanceResult(
                success=False,
                error=str(e),
                error_type="timeout",
                session_id=agent_session_id,
                container_name=container_name,
                duration_seconds=time.time() - start_time,
                started_at=started_at,
                completed_at=completed_at,
                retry_attempts=attempt_number - 1,
                log_path=log_path,
                workspace_path=None,
                status="timeout",
            )

        except (DockerError, GitError, AgentError, ValidationError) as e:
            logger.error(f"Instance {instance_id} failed: {type(e).__name__}: {e}")
            emit_event(
                "instance.failed",
                {
                    "error": str(e),
                    "error_type": type(e).__name__.lower().replace("error", ""),
                    "attempt": attempt_number,
                    "total_attempts": total_attempts,
                    "will_retry": attempt_number < total_attempts
                    and _is_retryable_error(
                        str(e),
                        type(e).__name__.lower().replace("error", ""),
                        retry_config,
                    ),
                },
            )

            # Update container status and remove container immediately on failure
            if container:
                try:
                    await docker_manager.update_container_status(container, "failed")
                except Exception:
                    pass
                try:
                    try:
                        await docker_manager.stop_heartbeat(container)
                    except Exception:
                        pass
                    await docker_manager.cleanup_container(
                        container, remove_home_volume=False
                    )
                except Exception:
                    pass

            # Always clean up workspace; do not retain for debugging
            try:
                if workspace_dir:
                    await git_ops.cleanup_workspace(workspace_dir)
                    emit_event(
                        "instance.workspace_cleaned",
                        {"workspace_dir": str(workspace_dir)},
                    )
            except Exception:
                pass

            completed_at = datetime.now(timezone.utc).isoformat()
            return InstanceResult(
                success=False,
                error=str(e),
                error_type=type(e).__name__.lower().replace("error", ""),
                session_id=agent_session_id,
                container_name=container_name,
                duration_seconds=time.time() - start_time,
                started_at=started_at,
                completed_at=completed_at,
                retry_attempts=attempt_number - 1,
                log_path=log_path,
                workspace_path=None,
                status="failed",
            )

        except asyncio.CancelledError:
            # Treat orchestrator-initiated cancellation as interruption, not failure
            logger.info(f"Instance {instance_id} canceled; recording interruption")
            emit_event(
                "instance.canceled",
                {
                    "error": "canceled",
                    "error_type": "canceled",
                    "attempt": attempt_number,
                    "total_attempts": total_attempts,
                    "will_retry": False,
                },
            )

            # Preserve container and session volume for resume; stop heartbeat only
            if container:
                try:
                    await docker_manager.stop_heartbeat(container)
                except Exception:
                    pass

            completed_at = datetime.now(timezone.utc).isoformat()
            return InstanceResult(
                success=False,
                error="canceled",
                error_type="canceled",
                session_id=agent_session_id,
                container_name=container_name,
                duration_seconds=time.time() - start_time,
                started_at=started_at,
                completed_at=completed_at,
                retry_attempts=attempt_number - 1,
                log_path=log_path,
                workspace_path=str(workspace_dir) if workspace_dir else None,
                status="canceled",
            )
        except (OSError, IOError) as e:
            logger.exception(f"System error in instance {instance_id}")
            emit_event(
                "instance.failed",
                {
                    "error": str(e),
                    "error_type": "system",
                    "attempt": attempt_number,
                    "total_attempts": total_attempts,
                    "will_retry": False,  # Never retry system errors
                },
            )

            # Remove container immediately on unexpected failure
            if container:
                try:
                    try:
                        await docker_manager.stop_heartbeat(container)
                    except Exception:
                        pass
                    await docker_manager.cleanup_container(
                        container, remove_home_volume=True
                    )
                except Exception:
                    pass

            # Always clean up workspace; do not retain for debugging
            try:
                if workspace_dir:
                    await git_ops.cleanup_workspace(workspace_dir)
                    emit_event(
                        "instance.workspace_cleaned",
                        {"workspace_dir": str(workspace_dir)},
                    )
            except Exception:
                pass

            completed_at = datetime.now(timezone.utc).isoformat()
            return InstanceResult(
                success=False,
                error=f"Unexpected error: {str(e)}",
                error_type="unexpected",
                session_id=agent_session_id,
                container_name=container_name,
                duration_seconds=time.time() - start_time,
                started_at=started_at,
                completed_at=completed_at,
                retry_attempts=attempt_number - 1,
                log_path=log_path,
                workspace_path=None,
                status="failed",
            )

    finally:
        # Always close the Docker client to prevent connection leaks
        try:
            if docker_manager:
                docker_manager.close()
        except Exception:
            pass
