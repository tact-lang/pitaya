"""Command execution helpers for DockerManager."""

import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import docker

from .docker_async import async_iter
from ..exceptions import DockerError, TimeoutError

logger = logging.getLogger(__name__)


async def execute_command(
    container: Any,
    command: list[str],
    plugin: Any,
    event_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    timeout_seconds: int = 3600,
    max_turns: Optional[int] = None,
    stream_log_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute AI tool command in container and parse output."""
    try:
        cmd_str = " ".join(f'"{arg}"' if " " in arg else arg for arg in command)
        try:
            _pname = getattr(plugin, "name", "tool")
        except Exception:
            _pname = "tool"
        logger.info(f"Executing {_pname}: {cmd_str}")

        try:
            if str(getattr(plugin, "name", "")) == "codex":
                _mk = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: container.client.api.exec_create(
                        container.id,
                        "sh -lc 'mkdir -p /home/node/.codex'",
                        stdout=False,
                        stderr=False,
                        tty=False,
                        workdir="/workspace",
                    ),
                )
                try:
                    await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: container.client.api.exec_start(_mk["Id"], detach=True),
                    )
                except Exception:
                    pass
        except Exception:
            pass

        loop = asyncio.get_event_loop()
        exec_instance = await loop.run_in_executor(
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

        raw_f = None
        if stream_log_path:
            try:
                p = Path(stream_log_path)
                p.parent.mkdir(parents=True, exist_ok=True)
                raw_f = p.open("a", encoding="utf-8", errors="replace")
                ts = datetime.now(timezone.utc).isoformat()
                try:
                    cname = getattr(container, "name", None) or "unknown"
                    cid = (getattr(container, "id", None) or "unknown")[:12]
                except Exception:
                    cname = "unknown"
                    cid = "unknown"
                header = (
                    f"=== EXEC START {ts} ===\n"
                    f"container_id={cid} name={cname}\n"
                    f"workdir=/workspace\n"
                    f"command={' '.join(command)}\n"
                    f"plugin={getattr(plugin, 'name', 'tool')}\n"
                )
                raw_f.write(header)
                raw_f.write(f"exec_id={exec_instance.get('Id')}\n")
                raw_f.flush()
            except Exception:
                raw_f = None

        output_stream = await loop.run_in_executor(
            None,
            lambda: container.client.api.exec_start(exec_instance["Id"], stream=True),
        )

        parser_state: Dict[str, Any] = {}
        start_time = asyncio.get_event_loop().time()
        raw_lines: list[str] = []

        async def parse_stream():
            turns_seen = 0
            last_activity = asyncio.get_event_loop().time()

            async def idle_watcher():
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
                    raise TimeoutError(
                        f"Agent execution exceeded {timeout_seconds}s timeout"
                    )

                raw_chunk = chunk
                if isinstance(chunk, bytes):
                    try:
                        if raw_f is not None:
                            raw_f.write(raw_chunk.decode("utf-8", errors="replace"))
                            raw_f.flush()
                    except Exception:
                        pass
                    chunk = chunk.decode("utf-8", errors="replace")
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
                                    if (
                                        str(parsed.get("type", "")).lower()
                                        == "turn_complete"
                                    ):
                                        turns_seen += 1
                                        if turns_seen >= max_turns:
                                            return
                                except Exception:
                                    pass
                        else:
                            try:
                                msg = line[:1000]
                                raw_lines.append(msg)
                                if len(raw_lines) > 200:
                                    raw_lines.pop(0)
                                if event_callback:
                                    event_callback(
                                        {
                                            "type": "log",
                                            "timestamp": datetime.now(
                                                timezone.utc
                                            ).isoformat(),
                                            "stream": "stdout",
                                            "message": msg,
                                        }
                                    )
                            except Exception:
                                pass

                    except json.JSONDecodeError:
                        try:
                            msg = line[:1000]
                            raw_lines.append(msg)
                            if len(raw_lines) > 200:
                                raw_lines.pop(0)
                            if event_callback:
                                event_callback(
                                    {
                                        "type": "log",
                                        "timestamp": datetime.now(
                                            timezone.utc
                                        ).isoformat(),
                                        "stream": "stdout",
                                        "message": msg,
                                    }
                                )
                            last_activity = asyncio.get_event_loop().time()
                        except Exception:
                            pass
            try:
                watcher.cancel()
            except Exception:
                pass

        try:
            await asyncio.wait_for(parse_stream(), timeout=timeout_seconds)
        except asyncio.TimeoutError:
            raise TimeoutError(f"Agent execution exceeded {timeout_seconds}s timeout")

        exec_info = await loop.run_in_executor(
            None, lambda: container.client.api.exec_inspect(exec_instance["Id"])
        )

        if exec_info["ExitCode"] != 0:
            tail = "\n".join(raw_lines[-10:]) if raw_lines else ""
            msg = f"Command exited with code {exec_info['ExitCode']}"
            if tail:
                msg += f"\nLast output:\n{tail}"
            try:
                if raw_f is not None:
                    ts = datetime.now(timezone.utc).isoformat()
                    raw_f.write(
                        f"\n=== EXEC END {ts} exit={exec_info.get('ExitCode', 'nonzero')} ===\n"
                    )
                    raw_f.flush()
            except Exception:
                pass
            raise DockerError(msg)

        result_data = await plugin.extract_result(parser_state)

        logger.info("Command execution completed successfully")
        try:
            if raw_f is not None:
                ts = datetime.now(timezone.utc).isoformat()
                raw_f.write(
                    f"\n=== EXEC END {ts} exit={exec_info.get('ExitCode', 0)} ===\n"
                )
                raw_f.flush()
        except Exception:
            pass
        return result_data

    except TimeoutError:
        raise
    except (docker.errors.APIError, docker.errors.DockerException, OSError) as e:
        raise DockerError(f"Failed to execute agent tool: {e}")
    finally:
        try:
            if raw_f is not None:
                raw_f.close()
        except Exception:
            pass


__all__ = ["execute_command"]
