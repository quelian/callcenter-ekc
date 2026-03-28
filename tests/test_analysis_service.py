from __future__ import annotations

from analysis_service import AnalysisArtifacts, build_sheets_export_payload
from transcription import QualityEvaluation, TranscriptionResult


def test_build_sheets_export_payload_contains_expected_fields() -> None:
    analysis = AnalysisArtifacts(
        logs=["ok"],
        result=TranscriptionResult(
            text="текст",
            role_text="Оператор: здравствуйте",
            tone_summary="Нейтральный",
        ),
        evaluation=QualityEvaluation(
            total_score=8,
            max_score=10,
            operator_name="Егор",
            operator_in_staff=True,
            applicant_name="Анна",
            script_score=8,
            speech_score=7,
            consultation_score=9,
            engagement_score=8,
            script_details=["Поздоровался"],
            positives=["Вежливо начал разговор"],
            negatives=[],
        ),
        report="report",
    )

    payload = build_sheets_export_payload(
        analysis,
        original_filename="12-02-2026_79990000000_01-12.mp3",
        filename_meta_override={"date": "12-02-2026"},
    )

    assert payload["original_filename"] == "12-02-2026_79990000000_01-12.mp3"
    assert payload["operator_name"] == "Егор"
    assert payload["applicant_name"] == "Анна"
    assert payload["tone_summary"] == "Нейтральный"
    assert payload["filename_meta_override"] == {"date": "12-02-2026"}
