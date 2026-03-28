from __future__ import annotations

from applicant_name_utils import normalize_applicant_name_candidate
from llm_cloud_eval import apply_deterministic_speech_findings_merge
from transcription import CallQualityEvaluator, QualityEvaluation, TranscriptionResult, build_report


def test_call_quality_evaluator_reports_detailed_speech_issues() -> None:
    evaluator = CallQualityEvaluator()
    role_text = "\n".join(
        [
            "Оператор: Здравствуйте, меня зовут Егор, чем могу помочь?",
            "Заявитель: Меня зовут Анна.",
            "Оператор: Анна, ну смотрите, маленько подождите минуточку, девушка, я не знаю, в общем уточню.",
            "Заявитель: Хорошо.",
            "Оператор: Спасибо за обращение, до свидания.",
        ]
    )

    evaluation = evaluator.evaluate(
        transcript=role_text.replace("\n", " "),
        role_transcript=role_text,
        forced_operator_name="Егор",
    )

    assert evaluation.applicant_name == "Анна"
    assert evaluation.speech_score <= 2
    assert any(item.startswith("Слова-паразиты:") for item in evaluation.negatives)
    assert any(item.startswith("Уменьшительно-ласкательные формы:") for item in evaluation.negatives)
    assert any(item.startswith("Категоричные/беспомощные формулировки:") for item in evaluation.negatives)
    assert any(item.startswith("Некорректные обращения вместо имени:") for item in evaluation.negatives)
    assert any(item.startswith("Обращение по имени") for item in evaluation.negatives)


def test_build_report_keeps_all_negative_findings() -> None:
    evaluation = QualityEvaluation(
        total_score=4,
        max_score=10,
        operator_name="Егор",
        operator_in_staff=True,
        applicant_name="Анна",
        script_score=3,
        speech_score=2,
        consultation_score=5,
        engagement_score=4,
        script_details=[
            "Приветствие + предложение помочь: да",
            "Представился по имени: да",
            "Познакомился с заявителем: нет",
            "Называл заявителя по имени ≥2 раз: нет (Анна) [оп.1/всего2]",
            "Предложил дополнительную помощь: нет",
            "Поблагодарил за обращение: да",
            "Персонализированное прощание: нет",
        ],
        positives=["Оператор вежливо начал разговор."],
        negatives=[
            "Слова-паразиты: \"ну\"×2 — уберите лишние вводные слова.",
            "Уменьшительно-ласкательные формы: \"минуточку\" — используйте нейтральные формы.",
            "Некорректные обращения вместо имени: \"девушка\" — обращайтесь по имени.",
            "Обращение по имени не выполнено: имя заявителя найдено, но оператор не обращался по имени ≥2 раз.",
        ],
    )
    result = TranscriptionResult(
        text="Тестовый текст",
        role_text="Оператор: тест\nЗаявитель: тест",
    )

    report = build_report(result, evaluation)

    assert report.count("- ") >= 12
    assert "Некорректные обращения вместо имени" in report
    assert "Обращение по имени не выполнено" in report


def test_cloud_eval_postprocess_adds_deterministic_speech_findings() -> None:
    evaluation = QualityEvaluation(
        total_score=8,
        max_score=10,
        operator_name="Егор",
        operator_in_staff=True,
        applicant_name="Анна",
        script_score=7,
        speech_score=9,
        consultation_score=8,
        engagement_score=8,
        script_details=[
            "Приветствие + предложение помочь: да",
            "Представился по имени: да",
            "Познакомился с заявителем: да",
            "Называл заявителя по имени ≥2 раз: да (Анна) [оп.2/всего3]",
            "Предложил дополнительную помощь: нет",
            "Поблагодарил за обращение: да",
            "Персонализированное прощание: нет",
        ],
        positives=["Оператор дал консультацию."],
        negatives=[],
    )
    role_text = "\n".join(
        [
            "Оператор: Здравствуйте, меня зовут Егор, чем могу помочь?",
            "Заявитель: Меня зовут Анна.",
            "Оператор: Девушка, ну смотрите, я не знаю, в общем сейчас уточню.",
            "Заявитель: Хорошо.",
        ]
    )
    logs: list[str] = []

    apply_deterministic_speech_findings_merge(
        evaluation,
        flat_text=role_text.replace("\n", " "),
        role_text=role_text,
        forced_operator_name="Егор",
        log=logs.append,
    )

    assert evaluation.speech_score < 9
    assert any(item.startswith("Слова-паразиты:") for item in evaluation.negatives)
    assert any(item.startswith("Категоричные/беспомощные формулировки:") for item in evaluation.negatives)
    assert any("детермин. речевых замечаний" in line for line in logs)


def test_applicant_name_normalizer_rejects_obvious_non_names() -> None:
    assert normalize_applicant_name_candidate("Воскресенье") is None
    assert normalize_applicant_name_candidate("Такое") is None
    assert normalize_applicant_name_candidate("Добрый день") is None
    assert normalize_applicant_name_candidate("Анна") == "Анна"
    assert normalize_applicant_name_candidate("мирослава строфская") == "Мирослава Строфская"


def test_call_quality_evaluator_skips_non_name_applicant_reply() -> None:
    evaluator = CallQualityEvaluator()
    role_text = "\n".join(
        [
            "Оператор: Как вас зовут?",
            "Заявитель: Воскресенье.",
            "Оператор: Чем могу помочь?",
        ]
    )

    evaluation = evaluator.evaluate(
        transcript=role_text.replace("\n", " "),
        role_transcript=role_text,
        forced_operator_name="Егор",
    )

    assert evaluation.applicant_name is None


def test_cloud_eval_postprocess_keeps_final_applicant_name_consistent_in_negatives() -> None:
    evaluation = QualityEvaluation(
        total_score=8,
        max_score=10,
        operator_name="Егор",
        operator_in_staff=True,
        applicant_name="Анна",
        script_score=7,
        speech_score=9,
        consultation_score=8,
        engagement_score=8,
        script_details=[
            "Приветствие + предложение помочь: да",
            "Представился по имени: да",
            "Познакомился с заявителем: да",
            "Называл заявителя по имени ≥2 раз: нет (Анна)",
            "Предложил дополнительную помощь: нет",
            "Поблагодарил за обращение: да",
            "Персонализированное прощание: нет",
        ],
        positives=["Оператор дал консультацию."],
        negatives=[],
    )
    role_text = "\n".join(
        [
            "Оператор: Здравствуйте, меня зовут Егор, чем могу помочь?",
            "Заявитель: Меня зовут Наталья.",
            "Оператор: Девушка, ну смотрите, я не знаю, в общем сейчас уточню.",
            "Заявитель: Хорошо.",
        ]
    )

    apply_deterministic_speech_findings_merge(
        evaluation,
        flat_text=role_text.replace("\n", " "),
        role_text=role_text,
        forced_operator_name="Егор",
        log=lambda _line: None,
    )

    joined = "\n".join(evaluation.negatives)
    assert "«Анна»" in joined
    assert "«Наталья»" not in joined
