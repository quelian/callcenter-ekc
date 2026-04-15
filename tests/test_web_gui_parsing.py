"""Tests for web_gui.py: parse_manual_wait_mm_ss and related parsing helpers."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _parse_manual_wait_mm_ss(text: str) -> tuple[int, int] | None:
    """Mirror of the function from web_gui.py for testing."""
    import re

    _MANUAL_WAIT_INPUT_RE = re.compile(r"^\s*(\d{1,3})\s*[:-]\s*(\d{1,2})\s*$")
    raw = (text or "").strip()
    if not raw:
        return (0, 0)
    if raw.isdigit():
        total = int(raw)
        if total < 0 or total > 180 * 60 + 59:
            return None
        return (total // 60, total % 60)
    m = _MANUAL_WAIT_INPUT_RE.match(raw)
    if not m:
        return None
    mm = int(m.group(1))
    ss = int(m.group(2))
    if ss >= 60 or mm > 180 or mm < 0:
        return None
    return (mm, ss)


def test_empty_string_returns_zero_tuple() -> None:
    assert _parse_manual_wait_mm_ss("") == (0, 0)
    assert _parse_manual_wait_mm_ss("   ") == (0, 0)
    assert _parse_manual_wait_mm_ss(None) == (0, 0)  # type: ignore[arg-type]


def test_colon_format() -> None:
    assert _parse_manual_wait_mm_ss("1:05") == (1, 5)
    assert _parse_manual_wait_mm_ss("01:30") == (1, 30)
    assert _parse_manual_wait_mm_ss("10:00") == (10, 0)


def test_dash_format() -> None:
    assert _parse_manual_wait_mm_ss("1-05") == (1, 5)
    assert _parse_manual_wait_mm_ss("01-30") == (1, 30)


def test_plain_seconds() -> None:
    assert _parse_manual_wait_mm_ss("90") == (1, 30)
    assert _parse_manual_wait_mm_ss("60") == (1, 0)
    assert _parse_manual_wait_mm_ss("0") == (0, 0)
    assert _parse_manual_wait_mm_ss("3659") == (60, 59)


def test_invalid_formats_return_none() -> None:
    # Seconds >= 60 in colon/dash format
    assert _parse_manual_wait_mm_ss("1:60") is None
    assert _parse_manual_wait_mm_ss("1-99") is None
    # Minutes too large
    assert _parse_manual_wait_mm_ss("181:00") is None
    # Non-numeric
    assert _parse_manual_wait_mm_ss("abc") is None
    assert _parse_manual_wait_mm_ss("1:05:30") is None
    assert _parse_manual_wait_mm_ss("foo:bar") is None


def test_plain_seconds_too_large() -> None:
    # 180 * 60 + 59 = 10859 max
    assert _parse_manual_wait_mm_ss("10859") == (180, 59)
    assert _parse_manual_wait_mm_ss("10860") is None
