"""
Постобработка оценок для операторов в зоне повышенного внимания (Егор, Артем, Даша).

- Формальные критерии и чек-лист не меняются на уровне фактов (script_details).
- После расчёта: +1 к каждому из четырёх критериев (потолок 10), пересчёт итога.
- Смягчение списка negatives / усиление акцента на positives (без фантазий в баллах).
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Callable

from operator_staff import operator_in_leniency_focus


def apply_focus_operator_adjustments(
    evaluation: Any,
    log: Callable[[str], None] | None = None,
) -> QualityEvaluation:
    if not operator_in_leniency_focus(evaluation.operator_name):
        return evaluation

    _log = log or (lambda _m: None)
    _log(
        "[eval] Оператор в зоне повышенного внимания (Егор/Артем/Даша): "
        "+1 к script/speech/consultation/engagement (макс 10), смягчение формулировок."
    )

    script = min(10, evaluation.script_score + 1)
    speech = min(10, evaluation.speech_score + 1)
    consultation = min(10, evaluation.consultation_score + 1)
    engagement = min(10, evaluation.engagement_score + 1)
    total_score = round(
        0.30 * script
        + 0.20 * speech
        + 0.30 * consultation
        + 0.20 * engagement
    )

    positives = list(evaluation.positives)
    negatives = [n for n in evaluation.negatives if (n or "").strip()]
    # Меньше замечаний, без «прострела» статистики: оставляем самое важное
    if len(negatives) > 2:
        negatives = negatives[:2]

    mentor_line = (
        "Контекст развития: сотрудник на этапе усиленной поддержки; "
        "критерии те же, акцент на конструктивном разборе и сильных сторонах."
    )
    if not any(mentor_line[:30] in (p or "") for p in positives):
        positives.insert(0, mentor_line)
    if len(positives) < 3:
        positives.append(
            "Соблюдён деловой тон и работа по стандарту в условиях живого диалога с заявителем."
        )

    return replace(
        evaluation,
        script_score=script,
        speech_score=speech,
        consultation_score=consultation,
        engagement_score=engagement,
        total_score=total_score,
        positives=positives[:10],
        negatives=negatives[:3],
    )
