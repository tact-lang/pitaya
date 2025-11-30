"""Run commands inside containers and stream/parse output."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from .. import DockerError, TimeoutError
from .exec_logging import close_raw_log, open_stream_log, write_exec_end
from .exec_prep import ensure_codex_home


async def execute_command(
    manager,
    *,
    container,
    command: List[str],
    plugin: Any,
    event_callback: Optional[Callable[[Dict[str, Any]], None]],
    timeout_seconds: int,
    max_turns: Optional[int],
    stream_log_path: Optional[str],
) -> Dict[str, Any]:
    raw_f = None
    loop = asyncio.get_event_loop()
    try:
        cmd_str = " ".join(f'"{arg}"' if " " in arg else arg for arg in command)
        manager.logger.info(  # noqa: SLF001
            f"Executing {getattr(plugin, 'name', 'tool')}: {cmd_str}"
        )

        await ensure_codex_home(loop, container, plugin)

        exec_instance = await _create_exec_instance(loop, container, command)
        raw_f = open_stream_log(
            stream_log_path, container, command, plugin, exec_instance
        )
        output_stream = await _start_exec_stream(loop, container, exec_instance)

        parser_state: Dict[str, Any] = {}
        start_time = asyncio.get_event_loop().time()
        raw_lines: List[str] = []

        await asyncio.wait_for(
            _parse_exec_stream(
                async_iter=manager._async_iter,  # noqa: SLF001
                output_stream=output_stream,
                plugin=plugin,
                parser_state=parser_state,
                event_callback=event_callback,
                timeout_seconds=timeout_seconds,
                max_turns=max_turns,
                raw_lines=raw_lines,
                raw_f=raw_f,
                start_time=start_time,
            ),
            timeout=timeout_seconds,
        )

        exec_info = await _inspect_exec(loop, container, exec_instance)
        if exec_info["ExitCode"] != 0:
            tail = "\n".join(raw_lines[-10:]) if raw_lines else ""
            msg = f"Command exited with code {exec_info['ExitCode']}"
            if tail:
                msg += f"\nLast output:\n{tail}"
            write_exec_end(raw_f, exec_info, fallback="nonzero")
            raise DockerError(msg)

        result_data = await plugin.extract_result(parser_state)
        manager.logger.info("Command execution completed successfully")  # noqa: SLF001
        write_exec_end(raw_f, exec_info)
        return result_data

    except TimeoutError:
        raise
    except Exception:
        raise
    finally:
        close_raw_log(raw_f)


async def _create_exec_instance(loop, container, command: List[str]) -> Dict[str, Any]:
    return await loop.run_in_executor(
        None,
        lambda: container.client.api.exec_create(
            container.id,
            command,
            stdout=True,
            stderr=True,
            tty=False,
            workdir="/workspace",
        ),
    )


async def _start_exec_stream(loop, container, exec_instance: Dict[str, Any]):
    return await loop.run_in_executor(
        None, lambda: container.client.api.exec_start(exec_instance["Id"], stream=True)
    )


async def _parse_exec_stream(
    *,
    async_iter,
    output_stream,
    plugin: Any,
    parser_state: Dict[str, Any],
    event_callback: Optional[Callable[[Dict[str, Any]], None]],
    timeout_seconds: int,
    max_turns: Optional[int],
    raw_lines: List[str],
    raw_f,
    start_time: float,
) -> None:
    turns_seen = 0
    last_activity = asyncio.get_event_loop().time()

    async def idle_watcher() -> None:
        while True:
            await asyncio.sleep(5)
            now = asyncio.get_event_loop().time()
            idle = now - last_activity
            if idle >= 10:
                try:
                    if event_callback:
                        event_callback(
                            {
                                "type": "instance.progress",
                                "data": {
                                    "phase": "model_wait",
                                    "idle_seconds": int(idle),
                                },
                            }
                        )
                except Exception:
                    pass

    watcher = asyncio.create_task(idle_watcher())
    async for chunk in async_iter(output_stream):
        if asyncio.get_event_loop().time() - start_time > timeout_seconds:
            raise TimeoutError(f"Agent execution exceeded {timeout_seconds}s timeout")

        if isinstance(chunk, bytes):
            raw_chunk = chunk
            try:
                if raw_f is not None:
                    raw_f.write(raw_chunk.decode("utf-8", errors="replace"))
                    raw_f.flush()
            except Exception:
                pass
            chunk = raw_chunk.decode("utf-8", errors="replace")
        else:
            try:
                if raw_f is not None:
                    raw_f.write(str(chunk))
                    raw_f.flush()
            except Exception:
                pass

        for line in chunk.strip().split("\n"):
            if not line:
                continue

            try:
                parsed = await plugin.parse_events(line, parser_state)
                if parsed:
                    if event_callback:
                        event_callback(parsed)
                    await asyncio.sleep(0)
                    last_activity = asyncio.get_event_loop().time()

                    if max_turns is not None and isinstance(max_turns, int):
                        try:
                            if str(parsed.get("type", "")).lower() == "turn_complete":
                                turns_seen += 1
                                if turns_seen >= max_turns:
                                    return
                        except Exception:
                            pass
                else:
                    _emit_log_event(event_callback, line, raw_lines)
            except json.JSONDecodeError:
                _emit_log_event(event_callback, line, raw_lines)
                last_activity = asyncio.get_event_loop().time()

    try:
        watcher.cancel()
    except Exception:
        pass


def _emit_log_event(event_callback, line: str, raw_lines: List[str]) -> None:
    try:
        msg = line[:1000]
        raw_lines.append(msg)
        if len(raw_lines) > 200:
            raw_lines.pop(0)
        if event_callback:
            event_callback(
                {
                    "type": "log",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "stream": "stdout",
                    "message": msg,
                }
            )
    except Exception:
        pass


async def _inspect_exec(
    loop, container, exec_instance: Dict[str, Any]
) -> Dict[str, Any]:
    return await loop.run_in_executor(
        None, lambda: container.client.api.exec_inspect(exec_instance["Id"])
    )
