"""Structured logging utilities for Pitaya."""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

__all__ = ["JSONFormatter", "get_component_logger", "setup_structured_logging"]

_COMPONENT_TO_MODULES: Dict[str, Tuple[str, ...]] = {
    "orchestration.jsonl": ("src.orchestration",),
    "runner.jsonl": ("src.instance_runner",),
    "tui.jsonl": ("src.tui",),
}
_NOISY_THIRD_PARTY_LOGGERS = (
    "urllib3",
    "docker",
    "docker.utils",
    "docker.auth",
    "docker.api",
)


class JSONFormatter(logging.Formatter):
    """Format log records as JSON Lines."""

    def format(
        self, record: logging.LogRecord
    ) -> str:  # noqa: D401 - short override doc
        entry = {
            "timestamp": _to_iso_millis(datetime.now(timezone.utc)),
            "level": record.levelname,
            "component": record.name,
            "message": record.getMessage(),
        }

        if record.exc_info:
            entry["exception"] = self.formatException(record.exc_info)

        for key, value in record.__dict__.items():
            if key in _LOG_RECORD_IGNORED_FIELDS:
                continue
            entry[key] = _json_safe(value)

        return json.dumps(entry)


_LOG_RECORD_IGNORED_FIELDS = {
    "name",
    "msg",
    "args",
    "created",
    "filename",
    "funcName",
    "levelname",
    "levelno",
    "lineno",
    "module",
    "msecs",
    "pathname",
    "process",
    "processName",
    "relativeCreated",
    "thread",
    "threadName",
    "exc_info",
    "exc_text",
    "stack_info",
    "getMessage",
}


def setup_structured_logging(
    logs_dir: Path,
    run_id: str,
    debug: bool = False,
    quiet: bool = False,
    no_tui: bool = False,
) -> None:
    """Configure JSONL logging for the current run."""
    run_logs_dir = _prepare_run_directory(logs_dir, run_id)
    console_level = _determine_console_level(debug=debug, quiet=quiet)
    file_formatter = JSONFormatter()

    component_handlers = _build_component_handlers(run_logs_dir, file_formatter)
    console_handlers = _build_console_handlers(no_tui, quiet, debug, console_level)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(logging.DEBUG)

    _configure_existing_loggers(component_handlers, console_handlers)
    _configure_parent_loggers(component_handlers, console_handlers)
    _configure_root_logger(root_logger, run_logs_dir, file_formatter, console_handlers)
    _limit_third_party_noise()


def get_component_logger(component: str) -> logging.Logger:
    """Return logger name-spaced under ``src.`` when missing the prefix."""
    if not component.startswith("src."):
        component = f"src.{component}"
    return logging.getLogger(component)


def _prepare_run_directory(logs_dir: Path, run_id: str) -> Path:
    run_logs_dir = logs_dir / run_id
    run_logs_dir.mkdir(parents=True, exist_ok=True)
    return run_logs_dir


def _determine_console_level(*, debug: bool, quiet: bool) -> int:
    if quiet:
        return logging.ERROR
    if debug:
        return logging.DEBUG
    return logging.INFO


def _build_component_handlers(
    run_logs_dir: Path, formatter: logging.Formatter
) -> Dict[str, logging.Handler]:
    handlers: Dict[str, logging.Handler] = {}
    for filename, modules in _COMPONENT_TO_MODULES.items():
        handler = logging.FileHandler(run_logs_dir / filename, mode="a")
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(formatter)
        for module_prefix in modules:
            handlers[module_prefix] = handler
    return handlers


def _build_console_handlers(
    no_tui: bool,
    quiet: bool,
    debug: bool,
    console_level: int,
) -> List[logging.Handler]:
    if not no_tui or quiet:
        return []

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(
        logging.Formatter("%(levelname)s %(name)s: %(message)s")
    )

    handlers: List[logging.Handler] = [console_handler]
    if debug:
        debug_handler = logging.StreamHandler(sys.stderr)
        debug_handler.setLevel(logging.DEBUG)
        debug_handler.setFormatter(
            logging.Formatter("%(levelname)s %(name)s: %(message)s")
        )
        handlers.append(debug_handler)
    return handlers


def _configure_existing_loggers(
    component_handlers: Dict[str, logging.Handler],
    console_handlers: Iterable[logging.Handler],
) -> None:
    for logger_name in list(logging.Logger.manager.loggerDict.keys()):
        if not logger_name.startswith("src."):
            continue

        logger = logging.getLogger(logger_name)
        logger.handlers.clear()
        logger.setLevel(logging.DEBUG)
        logger.propagate = False

        handler = _handler_for_logger(logger_name, component_handlers)
        if handler is not None:
            logger.addHandler(handler)

        for console_handler in console_handlers:
            logger.addHandler(console_handler)


def _configure_parent_loggers(
    component_handlers: Dict[str, logging.Handler],
    console_handlers: Iterable[logging.Handler],
) -> None:
    for module_prefix, handler in component_handlers.items():
        parent_logger = logging.getLogger(module_prefix)
        parent_logger.handlers.clear()
        parent_logger.setLevel(logging.DEBUG)
        parent_logger.propagate = False
        parent_logger.addHandler(handler)
        for console_handler in console_handlers:
            parent_logger.addHandler(console_handler)


def _configure_root_logger(
    root_logger: logging.Logger,
    run_logs_dir: Path,
    formatter: logging.Formatter,
    console_handlers: Iterable[logging.Handler],
) -> None:
    catch_all = logging.FileHandler(run_logs_dir / "other.jsonl", mode="a")
    catch_all.setLevel(logging.DEBUG)
    catch_all.setFormatter(formatter)
    root_logger.addHandler(catch_all)
    for console_handler in console_handlers:
        root_logger.addHandler(console_handler)


def _handler_for_logger(
    logger_name: str, component_handlers: Dict[str, logging.Handler]
) -> Optional[logging.Handler]:
    for module_prefix, handler in component_handlers.items():
        if logger_name.startswith(module_prefix):
            return handler
    return None


def _limit_third_party_noise() -> None:
    for name in _NOISY_THIRD_PARTY_LOGGERS:
        noisy_logger = logging.getLogger(name)
        noisy_logger.setLevel(logging.WARNING)
        noisy_logger.propagate = False


def _to_iso_millis(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _json_safe(value: object) -> object:
    try:
        json.dumps(value)
    except (TypeError, ValueError):
        return str(value)
    return value
