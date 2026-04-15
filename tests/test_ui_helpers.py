"""Tests for UI helper modules: progress helpers and batch helpers."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ui_progress_helpers import (
    TranscriptionProgressState,
    format_eta,
    make_steps_html,
    update_transcription_progress_from_log,
)
from ui_batch_helpers import format_uploaded_size, result_to_row_dict
from transcription import QualityEvaluation, TranscriptionResult


# ── progress helpers ──────────────────────────────────────────────────────

def test_format_eta_minutes_and_seconds() -> None:
    assert format_eta(125) == "2 мин 5 сек"
    assert format_eta(60) == "1 мин 0 сек"


def test_format_eta_seconds_only() -> None:
    assert format_eta(30) == "30 сек"
    assert format_eta(0) == "0 сек"


def test_format_eta_negative_returns_zero() -> None:
    assert format_eta(-5) == "0 сек"


def test_make_steps_html_all_idle() -> None:
    html = make_steps_html(-1, 0)
    assert "steps-bar" in html
    assert "step-dot idle" in html


def test_make_steps_html_first_done() -> None:
    html = make_steps_html(1, 1)
    assert "step-dot done" in html
    assert "step-dot active" in html


def test_make_steps_html_all_done() -> None:
    html = make_steps_html(-1, 5)
    assert "step-dot done" in html
    assert "idle" not in html
    assert "active" not in html


def test_progress_state_default_values() -> None:
    state = TranscriptionProgressState()
    assert state.current_step == 0
    assert state.done_steps == 0
    assert state.current_progress == 0


def test_update_progress_from_model_loading_log() -> None:
    state = TranscriptionProgressState()
    text, update = update_transcription_progress_from_log(
        "[этап: asr-модель] загрузка модели...", state
    )
    assert text is not None
    assert update is True
    assert state.current_progress == 12


def test_update_progress_from_evaluation_log() -> None:
    state = TranscriptionProgressState()
    text, update = update_transcription_progress_from_log(
        "[evaluator] оцениваю скрипт...", state
    )
    assert text is not None
    assert update is True
    assert state.current_step == 4


def test_update_progress_unknown_returns_none() -> None:
    state = TranscriptionProgressState()
    text, _ = update_transcription_progress_from_log("some random log line", state)
    assert text is None


# ── batch helpers ────────────────────────────────────────────────────────

def test_format_uploaded_size() -> None:
    assert format_uploaded_size(500) == "500 Б"
    assert format_uploaded_size(1024) == "1.0 КБ"
    assert format_uploaded_size(1024 * 1024) == "1.0 МБ"
    assert format_uploaded_size(1_500_000) == "1.4 МБ"


def test_result_to_row_dict_contains_expected_keys() -> None:
    result = TranscriptionResult(
        text="текст",
        role_text="Оператор: привет",
        tone_summary="Нейтральный",
    )
    evaluation = QualityEvaluation(
        total_score=7,
        max_score=10,
        operator_name="Егор",
        operator_in_staff=True,
        applicant_name="Анна",
        script_score=7,
        speech_score=6,
        consultation_score=8,
        engagement_score=7,
        script_details=["Поздоровался: Да"],
        positives=["Вежливо"],
        negatives=[],
    )
    row = result_to_row_dict("12-02-2026_79990000000_01-30.mp3", result, evaluation)
    assert row["original_filename"] == "12-02-2026_79990000000_01-30.mp3"
    assert row["operator_name"] == "Егор"
    assert row["total_score"] == 7
    assert row["review_flag"] in ("Да", "Нет")
