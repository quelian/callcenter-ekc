from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from transcription import (
    CallQualityEvaluator,
    QualityEvaluation,
    Transcriber,
    TranscriptionResult,
    build_report,
)

if TYPE_CHECKING:
    from llm_cloud_eval import YandexCloudConfig


@dataclass(frozen=True)
class TranscriberSettings:
    model_name: str
    compute_type: str
    asr_profile: str
    heavy_diarization: bool
    heavy_diarization_timeout_seconds: int
    enable_post_edit: bool
    post_edit_timeout_seconds: int
    enable_llm_post_edit: bool = False
    llm_backend: str = "off"
    llm_embedded_model_id: str | None = None
    llm_post_edit_timeout_seconds: float = 120.0
    llm_post_edit_base_url: str | None = None
    llm_post_edit_model: str | None = None
    llm_post_edit_api_key: str | None = None
    llm_yandex_api_key: str | None = None
    llm_yandex_folder_id: str | None = None
    llm_yandex_model: str = "yandexgpt-lite"
    language: str = "ru"
    model_load_timeout_seconds: int = 600
    total_time_budget_seconds: int = 300


@dataclass(frozen=True)
class AnalysisRequest:
    audio_path: str | Path
    operator_name: str | None = None
    original_basename: str | None = None
    skip_first_seconds: float | None = None
    cloud_eval_cfg: "YandexCloudConfig | None" = None
    cloud_eval_extra_instructions: str | None = None


@dataclass
class AnalysisArtifacts:
    logs: list[str]
    result: TranscriptionResult
    evaluation: QualityEvaluation
    report: str

    @property
    def logs_text(self) -> str:
        return "\n".join(self.logs)


class CollectingLogger:
    def __init__(self, on_log: Callable[[str], None] | None = None) -> None:
        self.lines: list[str] = []
        self.on_log = on_log

    def __call__(self, message: str) -> None:
        self.lines.append(message)
        if self.on_log is not None:
            self.on_log(message)


def create_transcriber(settings: TranscriberSettings) -> Transcriber:
    return Transcriber(
        model_name=settings.model_name,
        compute_type=settings.compute_type,
        skip_first_seconds=None,
        language=settings.language,
        model_load_timeout_seconds=settings.model_load_timeout_seconds,
        asr_profile=settings.asr_profile,
        heavy_diarization=settings.heavy_diarization,
        heavy_diarization_timeout_seconds=settings.heavy_diarization_timeout_seconds,
        enable_post_edit=settings.enable_post_edit,
        post_edit_timeout_seconds=settings.post_edit_timeout_seconds,
        total_time_budget_seconds=settings.total_time_budget_seconds,
        enable_llm_post_edit=settings.enable_llm_post_edit,
        llm_backend=settings.llm_backend,
        llm_embedded_model_id=settings.llm_embedded_model_id,
        llm_post_edit_timeout_seconds=settings.llm_post_edit_timeout_seconds,
        llm_post_edit_base_url=settings.llm_post_edit_base_url,
        llm_post_edit_model=settings.llm_post_edit_model,
        llm_post_edit_api_key=settings.llm_post_edit_api_key,
        llm_yandex_api_key=settings.llm_yandex_api_key,
        llm_yandex_folder_id=settings.llm_yandex_folder_id,
        llm_yandex_model=settings.llm_yandex_model,
    )


@contextmanager
def _override_transcriber_runtime(
    transcriber: Transcriber,
    *,
    skip_first_seconds: float | None,
    logger: Callable[[str], None],
):
    old_skip = transcriber.skip_first_seconds
    old_log = transcriber._log
    transcriber.skip_first_seconds = skip_first_seconds
    transcriber._log = logger  # type: ignore[method-assign]
    try:
        yield
    finally:
        transcriber.skip_first_seconds = old_skip
        transcriber._log = old_log  # type: ignore[method-assign]


def analyze_call(
    request: AnalysisRequest,
    *,
    transcriber: Transcriber | None = None,
    transcriber_settings: TranscriberSettings | None = None,
    on_log: Callable[[str], None] | None = None,
) -> AnalysisArtifacts:
    runtime_transcriber = transcriber
    if runtime_transcriber is None:
        if transcriber_settings is None:
            raise ValueError("transcriber or transcriber_settings must be provided")
        runtime_transcriber = create_transcriber(transcriber_settings)

    logger = CollectingLogger(on_log=on_log)
    with _override_transcriber_runtime(
        runtime_transcriber,
        skip_first_seconds=request.skip_first_seconds,
        logger=logger,
    ):
        result = runtime_transcriber.transcribe(
            request.audio_path,
            original_basename=request.original_basename,
        )

    logger("[evaluator] Оцениваю качество консультации...")
    if request.cloud_eval_cfg is not None:
        from llm_cloud_eval import run_cloud_evaluation

        evaluation, cloud_status = run_cloud_evaluation(
            result.text,
            result.role_text,
            request.cloud_eval_cfg,
            forced_operator_name=request.operator_name,
            log=logger,
            extra_eval_instructions=request.cloud_eval_extra_instructions,
        )
        logger(f"[evaluator] Оценка готова ({cloud_status}).")
    else:
        evaluator = CallQualityEvaluator()
        evaluation = evaluator.evaluate(
            result.text,
            result.role_text,
            forced_operator_name=request.operator_name,
        )
        logger("[evaluator] Оценка готова.")

    report = build_report(result, evaluation)
    return AnalysisArtifacts(
        logs=logger.lines,
        result=result,
        evaluation=evaluation,
        report=report,
    )


def build_sheets_export_payload(
    analysis: AnalysisArtifacts,
    *,
    original_filename: str,
    filename_meta_override: dict[str, str] | None = None,
) -> dict[str, object]:
    return {
        "original_filename": original_filename,
        "total_score": analysis.evaluation.total_score,
        "max_score": analysis.evaluation.max_score,
        "operator_name": analysis.evaluation.operator_name,
        "applicant_name": analysis.evaluation.applicant_name or "",
        "script_score": analysis.evaluation.script_score,
        "speech_score": analysis.evaluation.speech_score,
        "consultation_score": analysis.evaluation.consultation_score,
        "engagement_score": analysis.evaluation.engagement_score,
        "tone_summary": analysis.result.tone_summary,
        "script_details": analysis.evaluation.script_details,
        "positives": analysis.evaluation.positives,
        "negatives": analysis.evaluation.negatives,
        "filename_meta_override": filename_meta_override,
    }
