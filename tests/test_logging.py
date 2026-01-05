"""Tests for logging setup module."""

import json
import logging

from traderbot.logging_setup import (
    JSONFormatter,
    SimpleFormatter,
    get_logger,
    reset_logging,
    setup_logging,
)


class TestJSONFormatter:
    """Tests for JSON formatter."""

    def test_format_basic_message(self) -> None:
        """Test formatting a basic log message."""
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        output = formatter.format(record)
        data = json.loads(output)

        assert "timestamp" in data
        assert data["level"] == "INFO"
        assert data["logger"] == "test"
        assert data["message"] == "Test message"

    def test_format_with_args(self) -> None:
        """Test formatting with message arguments."""
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.WARNING,
            pathname="test.py",
            lineno=1,
            msg="Value is %d",
            args=(42,),
            exc_info=None,
        )

        output = formatter.format(record)
        data = json.loads(output)

        assert data["message"] == "Value is 42"


class TestSimpleFormatter:
    """Tests for simple formatter."""

    def test_format_includes_level(self) -> None:
        """Test simple formatter includes level."""
        formatter = SimpleFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        output = formatter.format(record)

        assert "INFO" in output
        assert "test" in output
        assert "Test message" in output


class TestLoggingSetup:
    """Tests for logging setup functions."""

    def test_setup_logging_creates_handler(self) -> None:
        """Test setup_logging adds handler to logger."""
        reset_logging()
        setup_logging(level="DEBUG")

        logger = logging.getLogger("traderbot")
        assert len(logger.handlers) > 0

    def test_setup_logging_only_once(self) -> None:
        """Test setup_logging only runs once."""
        reset_logging()
        setup_logging()
        handler_count = len(logging.getLogger("traderbot").handlers)

        setup_logging()  # Second call should do nothing
        assert len(logging.getLogger("traderbot").handlers) == handler_count

    def test_get_logger_returns_child(self) -> None:
        """Test get_logger returns child logger."""
        reset_logging()
        logger = get_logger("test.module")

        assert logger.name == "traderbot.test.module"

    def test_get_logger_handles_prefixed_name(self) -> None:
        """Test get_logger handles already-prefixed names."""
        reset_logging()
        logger = get_logger("traderbot.test.module")

        assert logger.name == "traderbot.test.module"

    def test_reset_logging(self) -> None:
        """Test reset_logging clears handlers."""
        setup_logging()
        reset_logging()

        # After reset, handlers should be cleared
        logger = logging.getLogger("traderbot")
        assert len(logger.handlers) == 0
