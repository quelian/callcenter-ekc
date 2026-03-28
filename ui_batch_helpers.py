from __future__ import annotations

import streamlit as st

from transcription import QualityEvaluation, TranscriptionResult


def result_to_row_dict(
    filename: str,
    result: TranscriptionResult,
    evaluation: QualityEvaluation,
) -> dict:
    """Конвертирует результат одного файла в dict для sheets_export / pandas."""
    from sheets_export import needs_review, parse_call_filename_metadata

    meta = parse_call_filename_metadata(filename)
    return {
        "original_filename": filename,
        "date": meta["date"] or "—",
        "operator_name": evaluation.operator_name or "—",
        "phone": meta["phone"] or "—",
        "applicant_name": evaluation.applicant_name or "—",
        "wait_display": meta["wait_display"] or "—",
        "total_score": evaluation.total_score,
        "max_score": evaluation.max_score,
        "script_score": evaluation.script_score,
        "speech_score": evaluation.speech_score,
        "consultation_score": evaluation.consultation_score,
        "engagement_score": evaluation.engagement_score,
        "tone_summary": result.tone_summary or "—",
        "script_details": evaluation.script_details,
        "positives": evaluation.positives,
        "negatives": evaluation.negatives,
        "review_flag": needs_review(evaluation.total_score, evaluation.negatives),
    }


def render_live_batch_table(
    placeholder,
    collected: list[dict],
    errors: dict[str, str],
    sheets_log: dict[str, str],
    sheets_configured: bool,
) -> None:
    """Обновляет живую таблицу результатов прямо во время обработки."""
    if not collected and not errors:
        return
    try:
        import pandas as pd

        rows = []
        for idx, row_data in enumerate(collected):
            row = {
                "№": idx + 1,
                "Файл": row_data["original_filename"],
                "Дата": row_data["date"],
                "Оператор": row_data["operator_name"],
                "Заявитель": row_data["applicant_name"],
                "Итог": row_data["total_score"],
                "Прослушать": row_data["review_flag"],
            }
            if sheets_configured:
                row["Таблица"] = sheets_log.get(row_data["original_filename"], "⏳")
            rows.append(row)

        for fname, emsg in errors.items():
            row = {
                "№": "—",
                "Файл": fname,
                "Дата": "—",
                "Оператор": "—",
                "Заявитель": "—",
                "Итог": "❌",
                "Прослушать": emsg[:40],
            }
            if sheets_configured:
                row["Таблица"] = "—"
            rows.append(row)

        df = pd.DataFrame(rows)

        def _color_score(val):
            try:
                score = int(val)
                if score >= 8:
                    return "background-color: #dcfce7; color: #166534"
                if score >= 5:
                    return "background-color: #fef9c3; color: #854d0e"
                return "background-color: #fee2e2; color: #991b1b"
            except Exception:
                return ""

        def _color_review(val):
            if val == "Да":
                return "background-color: #fee2e2; color: #991b1b; font-weight:600"
            if val == "Нет":
                return "background-color: #dcfce7; color: #166534; font-weight:600"
            return ""

        style = df.style.map(_color_score, subset=["Итог"]).map(
            _color_review,
            subset=["Прослушать"],
        )
        placeholder.dataframe(style, width="stretch", hide_index=True)
    except ImportError:
        lines = [f"{row['№']}. {row['original_filename']} — {row['total_score']}/10" for row in collected]
        placeholder.text("\n".join(lines))


def format_uploaded_size(num_bytes: int) -> str:
    if num_bytes >= 1024 * 1024:
        return f"{num_bytes / (1024 * 1024):.1f} МБ"
    if num_bytes >= 1024:
        return f"{num_bytes / 1024:.1f} КБ"
    return f"{num_bytes} Б"
