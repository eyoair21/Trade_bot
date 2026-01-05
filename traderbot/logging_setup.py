"""Structured logging setup for TraderBot.

Provides JSON-ish line formatting with module-level child loggers.
"""

import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any

from traderbot.config import get_config


class JSONFormatter(logging.Formatter):
    """JSON-ish line formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record as a JSON-ish line."""
        log_data: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Add extra fields if present
        if hasattr(record, "extra_fields"):
            log_data.update(record.extra_fields)

        return json.dumps(log_data)


class SimpleFormatter(logging.Formatter):
    """Simple formatter for human-readable output."""

    def __init__(self) -> None:
        super().__init__(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )


_initialized = False


def setup_logging(level: str | None = None, format_type: str | None = None) -> None:
    """Initialize the root logger with structured formatting.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
               Defaults to config value.
        format_type: Format type ('json' or 'simple').
                     Defaults to config value.
    """
    global _initialized
    if _initialized:
        return

    config = get_config()
    level = level or config.logging.level
    format_type = format_type or config.logging.format

    # Get the root logger for traderbot
    root_logger = logging.getLogger("traderbot")
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Remove existing handlers
    root_logger.handlers.clear()

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Set formatter based on format type
    if format_type == "json":
        console_handler.setFormatter(JSONFormatter())
    else:
        console_handler.setFormatter(SimpleFormatter())

    root_logger.addHandler(console_handler)

    # Allow propagation for caplog capture in tests
    root_logger.propagate = True

    _initialized = True


def get_logger(name: str) -> logging.Logger:
    """Get a child logger for a module.

    Args:
        name: Module name (will be prefixed with 'traderbot.')

    Returns:
        Logger instance for the module.
    """
    # Ensure logging is set up
    setup_logging()

    # Return child logger
    if name.startswith("traderbot."):
        return logging.getLogger(name)
    return logging.getLogger(f"traderbot.{name}")


def reset_logging() -> None:
    """Reset logging state (useful for testing)."""
    global _initialized
    _initialized = False
    logger = logging.getLogger("traderbot")
    logger.handlers.clear()
