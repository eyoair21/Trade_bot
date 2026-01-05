"""Tests for console encoding fallback in regress CLI."""

import os
import types

import pytest

from traderbot.cli._console import _can_encode, _fmt


class TestFmtFunction:
    """Tests for _fmt emoji fallback function."""

    def test_fmt_falls_back_on_cp1252(self, monkeypatch):
        """Test that _fmt falls back to ASCII when stdout can't encode emoji."""
        fake_stdout = types.SimpleNamespace(encoding="cp1252", write=lambda s: None)
        monkeypatch.setattr("sys.stdout", fake_stdout)

        check = "✅ ok"
        ascii_fb = "[OK] ok"
        out = _fmt(check, ascii_fb, use_emoji=True)
        assert out == ascii_fb  # cp1252 can't encode ✅

    def test_fmt_uses_emoji_with_utf8(self, monkeypatch):
        """Test that _fmt uses emoji when stdout supports UTF-8."""
        fake_stdout = types.SimpleNamespace(encoding="utf-8", write=lambda s: None)
        monkeypatch.setattr("sys.stdout", fake_stdout)

        check = "✅ ok"
        ascii_fb = "[OK] ok"
        out = _fmt(check, ascii_fb, use_emoji=True)
        assert out == check

    def test_fmt_respects_use_emoji_false(self, monkeypatch):
        """Test that _fmt returns ASCII when use_emoji=False."""
        fake_stdout = types.SimpleNamespace(encoding="utf-8", write=lambda s: None)
        monkeypatch.setattr("sys.stdout", fake_stdout)

        check = "✅ ok"
        ascii_fb = "[OK] ok"
        out = _fmt(check, ascii_fb, use_emoji=False)
        assert out == ascii_fb

    def test_fmt_handles_none_encoding(self, monkeypatch):
        """Test that _fmt handles None stdout encoding gracefully."""
        fake_stdout = types.SimpleNamespace(encoding=None, write=lambda s: None)
        monkeypatch.setattr("sys.stdout", fake_stdout)

        check = "✅ ok"
        ascii_fb = "[OK] ok"
        # Should default to utf-8 when encoding is None
        out = _fmt(check, ascii_fb, use_emoji=True)
        assert out == check

    def test_fmt_respects_env_var(self, monkeypatch):
        """Test that _fmt respects TRADERBOT_NO_EMOJI env var."""
        fake_stdout = types.SimpleNamespace(encoding="utf-8", write=lambda s: None)
        monkeypatch.setattr("sys.stdout", fake_stdout)

        check = "✅ ok"
        ascii_fb = "[OK] ok"

        # With env var unset (or removed), use_emoji=None should allow emoji
        monkeypatch.delenv("TRADERBOT_NO_EMOJI", raising=False)
        out = _fmt(check, ascii_fb, use_emoji=None)
        assert out == check

        # With env var set, use_emoji=None should disable emoji
        monkeypatch.setenv("TRADERBOT_NO_EMOJI", "1")
        out = _fmt(check, ascii_fb, use_emoji=None)
        assert out == ascii_fb


class TestCanEncode:
    """Tests for _can_encode helper function."""

    def test_can_encode_ascii(self, monkeypatch):
        """Test that ASCII strings are always encodable."""
        fake_stdout = types.SimpleNamespace(encoding="ascii", write=lambda s: None)
        monkeypatch.setattr("sys.stdout", fake_stdout)

        assert _can_encode("hello world")
        assert not _can_encode("✅ emoji")

    def test_can_encode_utf8(self, monkeypatch):
        """Test that UTF-8 can encode emoji."""
        fake_stdout = types.SimpleNamespace(encoding="utf-8", write=lambda s: None)
        monkeypatch.setattr("sys.stdout", fake_stdout)

        assert _can_encode("hello world")
        assert _can_encode("✅ emoji")
        assert _can_encode("❌ cross")
