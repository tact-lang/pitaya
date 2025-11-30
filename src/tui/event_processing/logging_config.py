"""Logging utilities shared across event processing modules."""

import logging

# Preserve the original logger name for consistent output
LOGGER_NAME = "src.tui.event_handler"
logger = logging.getLogger(LOGGER_NAME)

__all__ = ["logger", "LOGGER_NAME"]
