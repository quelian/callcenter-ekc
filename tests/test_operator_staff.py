"""Tests for operator_staff module: name canonicalization and leniency."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from operator_staff import (
    OPERATOR_CANONICAL_NAMES,
    OPERATORS_LENIENCY_FOCUS,
    canonicalize_operator_name,
    operator_in_leniency_focus,
)


def _is_in_staff(name: str) -> bool:
    return canonicalize_operator_name(name) is not None


def test_known_operators_are_recognized() -> None:
    for name in OPERATOR_CANONICAL_NAMES:
        assert _is_in_staff(name)


def test_unknown_operator_is_not_in_staff() -> None:
    assert not _is_in_staff("НеизвестныйЧеловек")
    assert not _is_in_staff("")


def test_leniency_operators_are_normalized() -> None:
    """All leniency-focus operators should resolve to a canonical name."""
    for raw in OPERATORS_LENIENCY_FOCUS:
        canonical = canonicalize_operator_name(raw)
        assert canonical is not None
        assert canonical in OPERATOR_CANONICAL_NAMES


def test_leniency_focus_after_canonicalization() -> None:
    """Known nicknames should trigger leniency after canonicalization."""
    assert operator_in_leniency_focus("Артем")
    assert operator_in_leniency_focus("Даша")
    assert operator_in_leniency_focus("Егор")


def test_non_leniency_operators() -> None:
    """Operators not in leniency focus should not trigger it."""
    # "Иван" is not in the leniency list by default
    assert not operator_in_leniency_focus("Иван")


def test_case_insensitive_canonicalization() -> None:
    assert canonicalize_operator_name("егор") == "Егор"
    assert canonicalize_operator_name("ЕГОР") == "Егор"


def test_whitespace_tolerant_canonicalization() -> None:
    assert canonicalize_operator_name("  Артем  ") == "Артем"


def test_operator_aliases_map_variants() -> None:
    """Various nickname forms should resolve correctly."""
    assert canonicalize_operator_name("тёма") == "Артем"
    assert canonicalize_operator_name("юля") == "Юлия"
    assert canonicalize_operator_name("настя") == "Анастасия"
    assert canonicalize_operator_name("рома") == "Роман"
    assert canonicalize_operator_name("ваня") == "Иван"
