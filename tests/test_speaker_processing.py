from __future__ import annotations

from speaker_voice_roles import (
    _build_speaker_spans_from_windows,
    _remap_segments_by_speaker_spans,
)
from transcription import should_attempt_intro_recovery


def test_build_speaker_spans_from_windows_tracks_turn_change() -> None:
    spans = _build_speaker_spans_from_windows(
        [0.0, 0.8, 1.6, 2.4, 3.2],
        [0, 0, 1, 1, 1],
        2.0,
    )

    assert len(spans) == 2
    assert spans[0][2] == 0
    assert spans[1][2] == 1
    assert spans[0][0] <= 0.0 and spans[0][1] >= 1.5
    assert 2.0 <= spans[1][0] <= 2.5
    assert spans[1][1] >= 5.0


def test_remap_segments_by_speaker_spans_splits_mixed_segment() -> None:
    out = _remap_segments_by_speaker_spans(
        [
            (
                0.0,
                4.0,
                "здравствуйте меня зовут анна чем могу помочь подскажите пожалуйста общежитие",
            )
        ],
        [
            (0.0, 1.9, "SPK_0"),
            (1.9, 4.0, "SPK_1"),
        ],
        min_split_sec=0.6,
        min_split_share=0.20,
        min_words_to_split=4,
    )

    assert len(out) == 2
    assert out[0][3] == "SPK_0"
    assert out[1][3] == "SPK_1"
    assert "здравствуйте" in out[0][2]
    assert "общежитие" in out[1][2]


def test_remap_segments_by_speaker_spans_keeps_dominant_speaker_when_minor_overlap() -> None:
    out = _remap_segments_by_speaker_spans(
        [
            (
                0.0,
                3.5,
                "подскажите пожалуйста как оформить справку для общежития",
            )
        ],
        [
            (0.0, 3.0, "SPK_0"),
            (3.0, 3.5, "SPK_1"),
        ],
    )

    assert len(out) == 1
    assert out[0][3] == "SPK_0"


def test_should_attempt_intro_recovery_when_start_is_late() -> None:
    assert should_attempt_intro_recovery(
        [(3.4, 5.1, "здравствуйте чем могу помочь")],
        audio_duration=48.0,
        duration_after_vad=15.0,
    )


def test_should_not_attempt_intro_recovery_when_intro_is_present() -> None:
    assert not should_attempt_intro_recovery(
        [
            (0.2, 2.4, "здравствуйте меня зовут анна"),
            (2.5, 6.0, "чем могу помочь подскажите пожалуйста"),
        ],
        audio_duration=42.0,
        duration_after_vad=19.0,
    )
