"""
Structured logging setup for the orchestrator.

Implements JSON Lines logging with component separation as per spec section 6.1.
Each component gets its own log file with structured JSON output.
"""

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path


class JSONFormatter(logging.Formatter):
    """Format log records as JSON Lines."""

    def format(self, record: logging.LogRecord) -> str:
        """Convert log record to JSON string."""
        # Build base log entry
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "component": record.name,
            "message": record.getMessage(),
        }

        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        # Add any extra fields
        for key, value in record.__dict__.items():
            if key not in [
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
            ]:
                try:
                    # Ensure value is JSON serializable
                    json.dumps(value)
                    log_entry[key] = value
                except (TypeError, ValueError):
                    log_entry[key] = str(value)

        return json.dumps(log_entry)


def setup_structured_logging(
    logs_dir: Path,
    run_id: str,
    debug: bool = False,
    quiet: bool = False,
    no_tui: bool = False,
) -> None:
    """
    Setup structured logging for all components.

    Per spec section 6.1:
    - Separate log files per component
    - JSON Lines format
    - Proper log levels
    - Optional console output in headless mode

    Args:
        logs_dir: Base logs directory
        run_id: Current run ID
        debug: Enable debug logging
        quiet: Quiet mode (errors only)
        no_tui: Running without TUI (enable console output)
    """
    # Create run-specific log directory
    run_logs_dir = logs_dir / run_id
    run_logs_dir.mkdir(parents=True, exist_ok=True)

    # Determine log level
    if quiet:
        log_level = logging.ERROR
    elif debug:
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO

    # Component mapping - which modules log to which files
    component_mapping = {
        "orchestration.jsonl": ["src.orchestration"],
        "runner.jsonl": ["src.instance_runner"],
        "tui.jsonl": ["src.tui"],
    }

    # Clear any existing handlers
    root_logger = logging.getLogger()
    root_logger.handlers = []

    # Set root logger level
    root_logger.setLevel(log_level)

    # Create handlers for each component
    handlers = {}
    json_formatter = JSONFormatter()

    for log_file, modules in component_mapping.items():
        file_path = run_logs_dir / log_file
        handler = logging.FileHandler(file_path, mode="a")
        handler.setFormatter(json_formatter)
        handler.setLevel(log_level)
        handlers[log_file] = (handler, modules)

    # Add console handler for headless mode
    console_handler = None
    if no_tui and not quiet:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(json_formatter)
        console_handler.setLevel(log_level)

    # Configure loggers for each module
    # First configure existing loggers
    for logger_name in list(logging.Logger.manager.loggerDict.keys()):
        if logger_name.startswith("src."):
            logger = logging.getLogger(logger_name)
            logger.handlers = []
            logger.setLevel(log_level)
            logger.propagate = False

            # Find the appropriate handler for this logger
            for log_file, (handler, modules) in handlers.items():
                for module_prefix in modules:
                    if logger_name.startswith(module_prefix):
                        logger.addHandler(handler)
                        break

            # Add console handler if in headless mode
            if console_handler:
                logger.addHandler(console_handler)

    # Also configure future loggers by setting up parent loggers
    for log_file, (handler, modules) in handlers.items():
        for module_prefix in modules:
            parent_logger = logging.getLogger(module_prefix)
            parent_logger.handlers = []
            parent_logger.setLevel(log_level)
            parent_logger.propagate = False
            parent_logger.addHandler(handler)

            if console_handler:
                parent_logger.addHandler(console_handler)

    # Configure root logger to catch anything else
    catch_all_path = run_logs_dir / "other.jsonl"
    catch_all_handler = logging.FileHandler(catch_all_path, mode="a")
    catch_all_handler.setFormatter(json_formatter)
    catch_all_handler.setLevel(log_level)
    root_logger.addHandler(catch_all_handler)

    if console_handler:
        root_logger.addHandler(console_handler)


def get_component_logger(component: str) -> logging.Logger:
    """
    Get a logger for a specific component.

    Args:
        component: Component name (e.g., "instance_runner", "orchestration", "tui")

    Returns:
        Logger instance for the component
    """
    # Ensure component name starts with src prefix
    if not component.startswith("src."):
        component = f"src.{component}"

    return logging.getLogger(component)
