"""Agent execution phase for instance attempts."""

from __future__ import annotations

from typing import Any, Dict, Optional

from .docker_manager import DockerManager
from .plugin_interface import RunnerPlugin


async def execute_agent_phase(
    docker_manager: DockerManager,
    plugin: RunnerPlugin,
    container,
    prompt: str,
    model: str,
    resolved_model_id: Optional[str],
    session_id: Optional[str],
    system_prompt: Optional[str],
    append_system_prompt: Optional[str],
    agent_cli_args: Optional[list[str]],
    max_turns: Optional[int],
    timeout_seconds: int,
    event_callback,
    run_id: Optional[str],
    instance_id: str,
) -> Dict[str, Any]:
    command = await plugin.build_command(
        prompt,
        model=resolved_model_id or model,
        session_id=session_id,
        system_prompt=system_prompt,
        append_system_prompt=append_system_prompt,
        agent_cli_args=agent_cli_args,
        max_turns=max_turns,
    )

    if event_callback:
        event_callback(
            "instance.agent_starting",
            {
                "container_name": getattr(container, "name", None),
                "command": command,
                "model": model,
                "resolved_model_id": resolved_model_id,
                "system_prompt_provided": bool(system_prompt),
                "append_system_prompt_provided": bool(append_system_prompt),
                "max_turns": max_turns,
            },
        )

    stream_log_path = None
    if run_id and instance_id:
        stream_log_path = f"./logs/{run_id}/runner_{instance_id}.log"

    return await docker_manager.execute_command(
        container,
        command,
        plugin,
        event_callback=event_callback,
        timeout_seconds=timeout_seconds,
        max_turns=max_turns,
        stream_log_path=stream_log_path,
    )


__all__ = ["execute_agent_phase"]
