from __future__ import annotations

from operator_staff import canonicalize_operator_name, operator_in_leniency_focus
from sheets_export import needs_review, parse_call_filename_metadata
from transcription import parse_call_filename_wait_skip


def test_parse_call_filename_wait_skip_standard() -> None:
    parsed = parse_call_filename_wait_skip("12-02-2026_7 (902) 556-73-27_01-12.mp3")
    assert parsed == (72.0, "01-12")


def test_parse_call_filename_wait_skip_rejects_bad_seconds() -> None:
    assert parse_call_filename_wait_skip("12-02-2026_79990000000_01-99.mp3") is None


def test_parse_call_filename_metadata_extracts_parts() -> None:
    meta = parse_call_filename_metadata("12-02-2026_79990000000_01-12.mp3")
    assert meta == {
        "date": "12-02-2026",
        "phone": "79990000000",
        "wait_display": "01:12",
    }


def test_operator_aliases_are_normalized() -> None:
    assert canonicalize_operator_name("оператор тёма") == "Артем"
    assert operator_in_leniency_focus("тёма")


def test_needs_review_for_low_score_and_critical_phrase() -> None:
    assert needs_review(5, ["Низкая оценка"]) == "Да"
    assert needs_review(8, ["Оператор грубил в ответе"]) == "Да"
    assert needs_review(8, ["Не представился"]) == "Нет"
