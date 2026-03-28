from __future__ import annotations

from dataclasses import dataclass

_STEPS = [
    ("Загрузка модели", "🧠"),
    ("Распознавание речи", "🎙️"),
    ("Разделение ролей", "👥"),
    ("AI-обработка", "✨"),
    ("Оценка качества", "📊"),
]


def make_steps_html(active_idx: int, done_up_to: int) -> str:
    parts = []
    for i, (label, _icon) in enumerate(_STEPS):
        if i < done_up_to:
            dot_cls, lbl_cls, symbol = "done", "done", "✓"
        elif i == active_idx:
            dot_cls, lbl_cls, symbol = "active", "active", str(i + 1)
        else:
            dot_cls, lbl_cls, symbol = "idle", "idle", str(i + 1)
        parts.append(
            f'<div class="step">'
            f'<div class="step-dot {dot_cls}">{symbol}</div>'
            f'<div class="step-label {lbl_cls}">{label}</div>'
            f"</div>"
        )
    return '<div class="steps-bar">' + "".join(parts) + "</div>"


def format_eta(seconds: float) -> str:
    sec = max(0, int(seconds))
    minutes, rem = divmod(sec, 60)
    if minutes > 0:
        return f"{minutes} мин {rem} сек"
    return f"{rem} сек"


@dataclass
class TranscriptionProgressState:
    """Прогресс одного файла (шаги + полоса), синхронно с логами транскрибера."""

    current_step: int = 0
    done_steps: int = 0
    current_progress: int = 0


def update_transcription_progress_from_log(
    message: str,
    state: TranscriptionProgressState,
) -> tuple[str | None, bool]:
    """
    Обновляет state по строке лога.
    Возвращает (текст для st.progress или None, обновлять ли полосу шагов).

    Логи транскрибера: ``[transcriber] [этап: имя] сообщение`` (см. transcription.Transcriber._stage).
    """
    m = (message or "").lower()

    if "[этап: начало]" in m:
        state.current_progress = max(state.current_progress, 4)
        return "Открываю файл и запускаю пайплайн...", False
    if "[этап: asr-параметры]" in m:
        state.current_progress = max(state.current_progress, 6)
        return "Параметры распознавания...", False
    if "[этап: asr-модель]" in m and "загрузка" in m:
        state.current_step, state.done_steps, state.current_progress = 0, 0, 12
        return "Загрузка модели Whisper (может быть долго при первом запуске)...", True
    if "[этап: asr-модель]" in m and ("готова" in m or "уже в памяти" in m):
        state.current_step, state.done_steps, state.current_progress = 1, 1, 28
        return "Модель готова", True
    if "[этап: пропуск ожидания]" in m or "[этап: аудио]" in m:
        state.current_progress = max(state.current_progress, 18)
        return "Подготовка аудио (обрезка, нормализация)...", False
    if "[этап: whisper]" in m and "запуск" in m:
        state.current_step, state.done_steps, state.current_progress = 1, 1, 38
        return "Whisper: признаки и VAD...", True
    if "[этап: whisper]" in m and "декодирование сегментов" in m:
        state.current_step, state.done_steps, state.current_progress = 1, 1, 52
        return "Распознавание речи (декодирование)...", True
    if "asr прогресс:" in m:
        state.current_progress = min(84, max(state.current_progress, 52) + 3)
        return "Распознавание речи...", False
    if "распознавание завершено" in m:
        state.current_step, state.done_steps, state.current_progress = 1, 2, 86
        return "Распознавание завершено", True
    if "[этап: роли]" in m and "черновик" in m:
        state.current_step, state.done_steps = 2, 2
        state.current_progress = max(state.current_progress, 87)
        return "Определяю роли (черновик по тексту)...", True
    if ("[этап: роли/" in m or "[этап: роли/ecapa" in m) and "ошибка" not in m:
        state.current_step, state.done_steps = 2, 2
        state.current_progress = max(state.current_progress, 88)
        return "Разделение ролей (голос / диаризация)...", True
    if "[этап: роли+голос]" in m and "совмещение" in m:
        state.current_step, state.done_steps = 2, 2
        state.current_progress = max(state.current_progress, 89)
        return "Совмещение голоса и текста...", True
    if "[этап: роли+голос]" in m and "блок завершён" in m:
        state.current_step, state.done_steps, state.current_progress = 3, 3, 91
        return "Роли определены", True
    if "[этап: llm]" in m or (
        "[этап: llm-постредактор]" in m and "выключен" not in m and "не сработал" not in m
    ):
        state.current_step = 3
        state.current_progress = max(state.current_progress, 92)
        return "AI: пост-редактура транскрипта...", True
    if "llm пост-редактор (" in m:
        state.current_step = 3
        state.current_progress = max(state.current_progress, 93)
        return "AI: пост-редактура...", False
    if "[этап: локальный пост]" in m and "выключен" not in m and "пропуск" not in m:
        state.current_step = 3
        state.current_progress = max(state.current_progress, 93)
        return "Локальная пунктуация и роли...", True
    if "[этап: сборка отчёта]" in m:
        state.current_progress = max(state.current_progress, 94)
        return "Сборка расшифровки по ролям...", False
    if "текст собран" in m:
        state.current_progress = max(state.current_progress, 95)
        return "Текст и роли собраны", False
    if "сформирована расшифровка по ролям" in m:
        state.current_step, state.done_steps = 3, 3
        state.current_progress = max(state.current_progress, 96)
        return "Транскрипт по ролям готов", True
    if "[этап: тональность]" in m and "анализ" in m:
        state.current_progress = max(state.current_progress, 97)
        return "Анализ тональности...", False
    if "[этап: готово]" in m:
        state.current_progress = max(state.current_progress, 98)
        return "Завершение распознавания...", False
    if "[evaluator] оцениваю" in m:
        state.current_step, state.done_steps, state.current_progress = 4, 4, 96
        return "Оцениваю качество консультации...", True
    if "[evaluator] оценка готова" in m:
        state.current_step, state.done_steps, state.current_progress = 4, 5, 99
        return "Формирую отчёт...", True
    return None, False
