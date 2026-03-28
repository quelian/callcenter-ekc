from __future__ import annotations

from transcription import (
    PreparedAudioDiagnostics,
    should_retry_ultima_full_decode,
    should_use_fast_ultima_decode,
)


def test_should_use_fast_ultima_decode_for_clean_dense_audio() -> None:
    diag = PreparedAudioDiagnostics(
        duration_seconds=74.0,
        active_rms=0.026,
        noise_rms=0.0038,
        speech_ratio=0.41,
        peak=0.73,
        compressed_source=True,
    )

    assert should_use_fast_ultima_decode(diag)


def test_should_not_use_fast_ultima_decode_for_noisy_sparse_audio() -> None:
    diag = PreparedAudioDiagnostics(
        duration_seconds=96.0,
        active_rms=0.0105,
        noise_rms=0.0048,
        speech_ratio=0.11,
        peak=0.99,
        compressed_source=True,
    )

    assert not should_use_fast_ultima_decode(diag)


def test_should_retry_ultima_full_decode_when_fast_result_is_too_sparse() -> None:
    retry, reason = should_retry_ultima_full_decode(
        text="здравствуйте чем могу помочь",
        pieces=[(0.0, 1.2, "здравствуйте"), (8.0, 9.0, "чем могу помочь")],
        duration_after_vad=42.0,
        segment_count=2,
        hallucination_count=0,
        intro_recovery_added=1,
    )

    assert retry
    assert reason in {
        "empty_or_too_short",
        "too_sparse_for_vad_span",
        "intro_still_sparse",
        "too_few_segments",
    }


def test_should_keep_fast_ultima_result_when_density_is_normal() -> None:
    retry, reason = should_retry_ultima_full_decode(
        text=(
            "здравствуйте единый контактный центр двфу меня зовут егор чем могу помочь "
            "подскажите пожалуйста номер заявления я сейчас уточню информацию"
        ),
        pieces=[
            (0.0, 2.0, "здравствуйте единый контактный центр двфу"),
            (2.0, 4.5, "меня зовут егор чем могу помочь"),
            (5.0, 8.5, "подскажите пожалуйста номер заявления"),
            (9.0, 12.0, "я сейчас уточню информацию"),
        ],
        duration_after_vad=12.0,
        segment_count=4,
        hallucination_count=0,
        intro_recovery_added=0,
    )

    assert not retry
    assert reason == ""
