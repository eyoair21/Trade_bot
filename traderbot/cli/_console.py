"""Console output helpers for CLI commands.

Provides encoding-safe output with emoji/ASCII fallback for cross-platform
compatibility (Windows cmd.exe, PowerShell, Unix terminals).
"""

import contextlib
import os
import sys


def _can_encode(s: str) -> bool:
    """Check if string can be encoded with current stdout encoding.

    Args:
        s: String to check.

    Returns:
        True if encodable, False otherwise.
    """
    try:
        encoding = sys.stdout.encoding or "utf-8"
        s.encode(encoding, errors="strict")
        return True
    except Exception:
        return False


def _fmt(s: str, ascii_fallback: str, use_emoji: bool | None = None) -> str:
    """Format string with emoji fallback to ASCII.

    Args:
        s: String with emoji.
        ascii_fallback: ASCII fallback string.
        use_emoji: Whether to attempt emoji. If None, checks TRADERBOT_NO_EMOJI
            environment variable (set to any value to disable emoji).

    Returns:
        Formatted string (emoji if possible, ASCII otherwise).
    """
    # Check environment variable if use_emoji not explicitly set
    if use_emoji is None:
        use_emoji = not os.environ.get("TRADERBOT_NO_EMOJI")

    if not use_emoji:
        return ascii_fallback
    if _can_encode(s):
        return s
    return ascii_fallback


def configure_windows_console() -> None:
    """Configure Windows console for UTF-8 output (best-effort).

    On Windows, attempts to reconfigure stdout to use UTF-8 encoding.
    Silently ignores any errors (e.g., if stdout doesn't support reconfigure).
    """
    if os.name == "nt":
        with contextlib.suppress(Exception):
            sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
