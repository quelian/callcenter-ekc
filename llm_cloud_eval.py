"""
Облачный оценщик качества звонков через Yandex AI Studio (OpenAI-compatible API).

LLM определяет имена оператора/заявителя и выставляет баллы по 4 критериям.
При ошибке — автоматический fallback на эвристический оценщик.

Переменные окружения:
  YANDEX_API_KEY               — API-ключ Yandex Cloud (обязательно)
  YANDEX_FOLDER_ID             — Folder ID (обязательно; виден в консоли Yandex Cloud)
  YANDEX_CLOUD_MODEL           — имя модели (по умолчанию: yandexgpt-lite)
  YANDEX_CLOUD_TIMEOUT_SECONDS — таймаут запроса (по умолчанию: 60)
"""
from __future__ import annotations

import json
import os
import re
import ssl
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Callable

from operator_staff import (
    canonicalize_operator_name,
    operator_in_leniency_focus,
    operators_prompt_sentence,
)

_BASE_URL = "https://llm.api.cloud.yandex.net/v1"
_DEFAULT_MODEL = "yandexgpt-lite"


@dataclass(frozen=True)
class YandexCloudConfig:
    api_key: str
    folder_id: str
    model: str = _DEFAULT_MODEL
    timeout_seconds: float = 60.0


def load_cloud_config_from_env() -> YandexCloudConfig | None:
    key = (os.environ.get("YANDEX_API_KEY") or "").strip()
    folder = (os.environ.get("YANDEX_FOLDER_ID") or "").strip()
    if not key or not folder:
        return None
    model = (os.environ.get("YANDEX_CLOUD_MODEL") or _DEFAULT_MODEL).strip()
    try:
        timeout = float(os.environ.get("YANDEX_CLOUD_TIMEOUT_SECONDS") or "60")
    except ValueError:
        timeout = 60.0
    timeout = max(15.0, min(timeout, 300.0))
    return YandexCloudConfig(api_key=key, folder_id=folder, model=model, timeout_seconds=timeout)


_SYSTEM_PROMPT = f"""\
Ты — эксперт QA-оценки звонков Единого Контактного Центра ДВФУ (Дальневосточный федеральный университет, Владивосток).
Тебе дают расшифровку телефонного разговора оператора с заявителем.
Оценивай ТОЛЬКО реплики оператора — не снижай баллы за слова и поведение заявителя.

═══════════════════════════════════════════════════════════
ЗАДАЧА 1 — Определи имена
═══════════════════════════════════════════════════════════
{operators_prompt_sentence()}

• operator_name: кто принял звонок. Ищи «меня зовут …», «… слушаю», «это …», имя в приветствии.
  Сопоставь с штатным списком выше (допускаются только эти имена). Если имя не сказано, неясно или не из списка → "Не определено".
  В operator_name НЕЛЬЗЯ писать имён вне штатного списка — это почти всегда заявитель.
• applicant_name: кто позвонил. Ищи ответ на «как вас зовут», «меня зовут», самопредставление.
  Любое другое личное имя в разговоре (не из списка операторов), по контексту относящееся к клиенту, — это заявитель; укажи в applicant_name.
  Если не найдено → null.

═══════════════════════════════════════════════════════════
ЗАДАЧА 2 — Оцени по 4 критериям (каждый 1–10)
═══════════════════════════════════════════════════════════

──────────────────────────────────────────────────────────
script_score — Соблюдение скрипта звонка
──────────────────────────────────────────────────────────
Проверь 7 обязательных пунктов (каждый ≈1.4 балла):

1. ПРИВЕТСТВИЕ + ПРЕДЛОЖЕНИЕ ПОМОЧЬ
   ВСЕГДА «да» — все операторы ЕКЦ 100% выполняют этот пункт. Ставь «да» автоматически.

2. ПРЕДСТАВЛЕНИЕ ПО ИМЕНИ
   ВСЕГДА «да» — все операторы ЕКЦ 100% называют своё имя. Ставь «да» автоматически.

3. ЗНАКОМСТВО С ЗАЯВИТЕЛЕМ
   Спросил имя заявителя («как вас зовут?», «как к вам обращаться?», «представьтесь, пожалуйста») ИЛИ
   заявитель представился сам И оператор обращался к нему по имени ≥2 раз.

4. ОБРАЩЕНИЕ ПО ИМЕНИ В ХОДЕ РАЗГОВОРА
   Просматривай **весь транскрипт целиком** — реплики и Оператора, и Заявителя.
   «да», если выполняется ХОТЯ БЫ ОДНО из условий:
   а) имя заявителя (из поля applicant_name) упоминается в диалоге **≥3 раз суммарно** — в любых репликах, неважно кто говорит;
   б) любое имя, которое НЕ является именем оператора, встречается в диалоге **≥3 раз суммарно** — в любых репликах.
   Считай все упоминания по всему тексту разговора от начала до конца.

5. ПРЕДЛОЖЕНИЕ ДОПОЛНИТЕЛЬНОЙ ПОМОЩИ В КОНЦЕ
   («чем ещё могу помочь?», «остались ли вопросы?», «могу ли ещё чем-то быть полезен(-на)?»,
   «все ли я для вас сделал(-а)?»)

6. БЛАГОДАРНОСТЬ ЗА ОБРАЩЕНИЕ
   («спасибо за обращение», «благодарю за звонок», «рады вашему обращению»)

7. ПЕРСОНАЛИЗИРОВАННОЕ ПРОЩАНИЕ
   Попрощался с доброй фразой, подходящей к контексту разговора, желательно по имени заявителя.
   («всего доброго», «хорошего дня», «был(-а) рад(-а) помочь», «удачного дня», «приятного заселения» и т.п.)
   Прощание должно быть тёплым и индивидуальным, а не формальным «до свидания».

Шкала script_score:
  9–10: 6–7 пунктов выполнены
  7–8:  4–5 пунктов выполнены
  5–6:  2–3 пункта выполнены
  3–4:  1 пункт выполнен
  1–2:  ни одного пункта

──────────────────────────────────────────────────────────
speech_score — Чистота и культура речи оператора
──────────────────────────────────────────────────────────
Начало: 10 баллов. Снижай ТОЛЬКО за нарушения в репликах ОПЕРАТОРА.

СЛОВА-ПАРАЗИТЫ — «-1» за каждое уникальное слово/оборот, обнаруженное в разговоре:
  ну, эм, ааа, как бы, типа, вот, так сказать, в общем, короче, кстати, просто,
  походу, маленько, грубо говоря, на самом деле, это самое,
  «смотрите» / «послушайте» в качестве вводного слова (без реальной просьбы посмотреть/послушать)

УМЕНЬШИТЕЛЬНО-ЛАСКАТЕЛЬНЫЕ — «-1» за каждое употреблённое слово (они неуместны в деловой речи):
  минуточку, трубочка, звоночек, заявочка, договорчик, документик, справочка, денежки,
  счётик, актик, и любые другие слова с уменьшительными суффиксами в деловом контексте

ЗАПРЕЩЁННЫЕ ФРАЗЫ — «-2» за каждую (документ ЕКЦ: «Фразы которые следует избегать»):
  «я не знаю» / «не знаю» → должно быть: «на текущий момент отсутствует информация, уточню»
  «нет» в начале предложения как форма отказа → должна быть конкретная альтернатива
  «невозможно» как отказ без предложения альтернативы → ищи и предлагай варианты решения
  «мы не можем» / «не можем это сделать» → должно быть: «для решения вопроса необходимо…»
  «не могу» как прямой отказ без предложения решения → всегда предлагай альтернативу
  «вам нужно» / «вы должны» (директивный тон) → должно быть: «для решения вопроса необходимо…»
  «вы не поняли» / «вы неправильно поняли» → должно быть: «давайте ещё раз пройдёмся по шагам»
  «ваша проблема» → должно быть: «мы обязательно поможем вам в решении вашего вопроса»
  «вы не правы» → должна быть корректная альтернатива
  «вас беспокоит» при ответе на входящий → некорректная формулировка
  «к сожалению» → проявление беспомощности; ищи решение, а не сочувствуй отказу
  «не могу подсказать» / «это не ко мне» / «это не моя зона ответственности» → всегда помогай найти решение
  Нечёткие сроки без конкретики: «недели через две», «через какое-то время», «скоро», «потом»
    → должны быть конкретные сроки: «от 3 до 5 рабочих дней», «в течение 15 минут» и т.п.

Минимум speech_score: 0 баллов.

──────────────────────────────────────────────────────────
consultation_score — Качество и полнота консультации
──────────────────────────────────────────────────────────
Оценивай по совокупности критериев «Золотых правил ЕКЦ»:

Правило активности: оператор удерживает инициативу разговора, предоставляет полную информацию,
  не ждёт когда заявитель «вытянет» данные вопросами, не молчит и не уходит в долгие паузы без объяснения.

Правило правильной информации: не использует «нет», «невозможно», «не могу» как форму отказа —
  всегда ищет и предлагает ≥2 варианта решения. При невозможности решить «здесь и сейчас» —
  чётко объясняет: когда, где и при каких условиях заявитель получит ответ.

Правило удержания (hold): если уходит на удержание —
  а) объясняет конкретную причину («уточню информацию по вашему вопросу»);
  б) просит оставаться на линии не более 2 минут;
  в) если нужно дольше — предлагает перезвонить («вам будет удобно, если я перезвоню в течение 15 минут?»).
  Бросить заявителя на «тишину» без объяснения — грубое нарушение.

Шкала consultation_score:
  9–10: полный ответ + конкретные шаги/документы/сроки + предложил ≥2 варианта решения +
        резюмировал ключевые моменты («итак, вам необходимо…») + убедился что вопрос закрыт
        («я ответил(-а) на ваш вопрос?») + удерживал инициативу; при удержании соблюл всю процедуру
  7–8:  хороший ответ, конкретные шаги есть, но без резюме ИЛИ без проверки закрытия вопроса
  5–6:  ответил частично: без конкретных сроков или документов, заявитель переспрашивал
  3–4:  перенаправил без объяснения ИЛИ использовал «нет/невозможно/не могу» без альтернативы ИЛИ
        дал общие слова без конкретных шагов
  1–2:  сказал «не знаю» без уточнения, отказался помочь, бросил на удержание без предупреждения,
        не предложил ни одного варианта решения

──────────────────────────────────────────────────────────
engagement_score — Вовлечённость, эмоциональность и культура общения
──────────────────────────────────────────────────────────
Оценивай по совокупности «Золотых правил ЕКЦ» в части поведения оператора:

Правило внимательности: не перебивает заявителя, не переспрашивает уже озвученное,
  запоминает детали обращения. Переспрашивает только если заявитель плохо сформулировал вопрос.

Правило заинтересованности: задаёт уточняющие вопросы («правильно ли я понял(-а)…»,
  «что именно происходит?»), использует одобряющие фразы («да», «верно», «я вас понял(-а)»),
  управляет разговором с помощью вопросов, демонстрирует желание помочь.

Правило доброжелательности: оператор всегда эмоционально «на ступень выше» собеседника —
  при нейтральном заявителе — доброжелателен;
  при агрессивном / раздражённом — сохраняет нейтрально-вежливый тон;
  использует слова вежливости, благодарность («благодарю вас за ожидание»),
  НЕ повышает голос, НЕ выражает раздражения.

Правило корректности: сохраняет спокойствие в любой ситуации, не критикует клиента,
  не делает ему замечаний, не переходит на личный стиль («не грубите мне»),
  не выражает недовольства интонацией или словами.

Правило комфорта: говорит на понятном языке — без аббревиатур, без сложных профессиональных
  терминов без объяснения, подстраивается под темп и словарный запас заявителя.

При конфликтном / недовольном заявителе — 4 обязательных шага:
  1. Выслушать без перебивания
  2. Выразить сочувствие («я понимаю, это действительно неприятно», «мне очень жаль, что так сложилось»)
  3. Уточнить детали проблемы
  4. Предложить конкретное решение или эскалацию («давайте вместе выберем удобное для вас решение»,
     «мы сделаем всё, что в наших силах для решения вашего вопроса»)

Шкала engagement_score:
  9–10: соблюдены все правила: внимателен, заинтересован, доброжелателен, корректен, говорит понятно;
        при конфликте — прошёл все 4 шага; задавал уточняющие вопросы, использовал одобряющие фразы
  7–8:  большинство правил соблюдено, уточняющие вопросы есть, но не всегда; изредка пассивен
  5–6:  минимальные уточняющие вопросы; в основном пассивно отвечал; не всегда доброжелателен
  3–4:  не задавал вопросов, молчал в паузах, перебивал; использовал термины без объяснения;
        говорил свысока или отстранённо; при конфликте не прошёл ни одного шага
  1–2:  повысил голос, критиковал клиента, делал замечания, проявлял безразличие или грубость

═══════════════════════════════════════════════════════════
ЗАДАЧА 3 — Сформируй итоговый вывод
═══════════════════════════════════════════════════════════
• script_details — ровно 7 строк «Пункт: да» или «Пункт: нет» (точно в том же порядке и формулировках)
• positives — 2–4 конкретные сильные стороны с примером-цитатой из диалога оператора
• negatives — 2–4 замечания в формате: «[Нарушение]: "[цитата из диалога]" — [рекомендация]»
  Если нарушений нет — верни пустой массив [].
  Цитируй только реплики оператора, не заявителя.

═══════════════════════════════════════════════════════════
ФОРМАТ ОТВЕТА — только валидный JSON, без markdown, без комментариев
═══════════════════════════════════════════════════════════
{{
  "operator_name": "Егор или Не определено (только имя из штатного списка)",
  "applicant_name": "Имя или null",
  "script_score": 8,
  "speech_score": 9,
  "consultation_score": 7,
  "engagement_score": 6,
  "script_details": [
    "Приветствие + предложение помочь: да",
    "Представился по имени: да",
    "Познакомился с заявителем: нет",
    "Называл заявителя по имени ≥2 раз: нет",
    "Предложил дополнительную помощь: да",
    "Поблагодарил за обращение: да",
    "Персонализированное прощание: да"
  ],
  "positives": ["Оператор активно управлял разговором: \\"Правильно ли я понял, что вас интересует...\\""],
  "negatives": ["Запрещённая фраза: \\"Я не знау\\" — следует говорить \\"На текущий момент мне необходимо уточнить информацию\\""]
}}"""


def _leniency_extra_system_for_forced(forced_operator_name: str | None) -> str:
    """
    Если в UI выбран оператор из зоны повышенного внимания — доп. инструкции к LLM
    (пограничные случаи, акцент на positives). Постобработка +1 — в evaluation_leniency.
    """
    if not operator_in_leniency_focus(forced_operator_name):
        return ""
    return (
        "\n\n---\nДОПОЛНЕНИЕ К ЭТОМУ ЗВОНКУ:\n"
        "Оператор — из группы усиленной поддержки и развития (Егор, Артем или Даша).\n"
        "Формальные критерии и чек-лист те же, без ослабления требований.\n"
        "Если по тексту допустимы две близкие оценки — выбери более высокую.\n"
        "positives: 3–4 содержательных пункта с короткими цитатами из реплик оператора.\n"
        "negatives: только при явных нарушениях стандарта; по возможности 0–2 пункта, "
        "формулировки уважительные; не дроби мелкие огрехи.\n"
        "Не заполняй везде 10 баллов без оснований по шкалам критериев выше.\n---\n"
    )


def _operator_intro_snippet(role_text: str | None, max_lines: int = 8, max_chars: int = 600) -> str:
    """
    Первые реплики оператора из role_text (до max_lines строк, до max_chars символов).
    Используется как «наводка» на имя оператора — оно почти всегда в самом начале.
    """
    rt = role_text or ""
    lines = [l for l in rt.splitlines() if l.strip().startswith("Оператор:")]
    snippet = "\n".join(lines[:max_lines])
    return snippet[:max_chars].strip()


def _user_prompt(flat_text: str | None, role_text: str | None) -> str:
    # Приоритет — транскрипт по ролям (до 9000 символов, весь звонок).
    # Сплошной текст — fallback, если диаризация не сработала.
    role_trunc = (role_text or "").strip()[:9000]
    if len(role_trunc) >= 200:
        intro = _operator_intro_snippet(role_text)
        intro_block = (
            f"\n\n[НАЧАЛО ЗВОНКА — первые реплики оператора для определения имени]:\n{intro}"
            if intro
            else ""
        )
        return (
            f"ТРАНСКРИПТ ПО РОЛЯМ:{intro_block}\n\n"
            f"{role_trunc}\n\n"
            "Оцени звонок и верни только JSON."
        )
    # Fallback: диаризация не сработала, используем сплошной текст
    flat_trunc = (flat_text or "").strip()[:9000]
    # Выделяем начало — первые 300 символов, где обычно звучит имя оператора
    intro_flat = flat_trunc[:300].strip()
    intro_block = (
        f"\n[НАЧАЛО ЗАПИСИ — первые слова для определения имени оператора]:\n{intro_flat}\n\n"
        if intro_flat
        else ""
    )
    return (
        f"ТЕКСТ ЗВОНКА (роли не определены):{intro_block}"
        f"{flat_trunc}\n\n"
        "Оцени звонок и верни только JSON."
    )


def _extract_json(content: str | None) -> dict:
    content = (content or "").strip()
    # Убираем markdown-обёртки если модель их добавила
    m = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", content)
    if m:
        content = m.group(1)
    # Берём первый {...}
    start = content.find("{")
    end = content.rfind("}")
    if start != -1 and end > start:
        content = content[start : end + 1]
    return json.loads(content)


def _safe_score(data: dict, key: str, default: int = 5) -> int:
    try:
        return max(0, min(10, int(data.get(key, default))))
    except (TypeError, ValueError):
        return default


def _safe_strlist(data: dict, key: str) -> list[str]:
    v = data.get(key) or []
    if not isinstance(v, list):
        return []
    return [str(s) for s in v if s]


# Слова, которые модель иногда кладёт в поле имени, но это не ФИО заявителя
_APPLICANT_NAME_BLOCKLIST = frozenset(
    {
        "оператор",
        "заявитель",
        "клиент",
        "абонент",
        "слушаю",
        "здравствуйте",
        "добрый",
        "день",
        "колл",
        "центр",
        "двфу",
        "университет",
    }
)


def _format_applicant_name_candidate(raw: str) -> str | None:
    """Если строка похожа на имя человека (не оператор из штата) — нормализует для applicant_name."""
    s = (raw or "").strip()
    if not s:
        return None
    low = s.lower()
    if low in _APPLICANT_NAME_BLOCKLIST:
        return None
    parts = re.findall(r"[А-Яа-яЁё]+(?:-[А-Яа-яЁё]+)?", s)
    if not parts or len(parts) > 3:
        return None
    for p in parts:
        pl = p.lower()
        if pl in _APPLICANT_NAME_BLOCKLIST:
            return None
        if len(p) < 2 or len(p) > 24:
            return None
    titled: list[str] = []
    for p in parts:
        if "-" in p:
            titled.append(
                "-".join(
                    (seg[0].upper() + seg[1:].lower()) if len(seg) > 1 else seg.upper()
                    for seg in p.split("-")
                    if seg
                )
            )
        else:
            titled.append(p[0].upper() + p[1:].lower() if len(p) > 1 else p.upper())
    return " ".join(titled)


def _maybe_promote_nonstaff_operator_name_to_applicant(
    op_raw: str,
    resolved_operator: str,
    applicant_name: str | None,
    forced_operator_name: str | None,
) -> str | None:
    """
    Если модель положила в operator_name имя не из штата — с большой вероятностью это заявитель.
    Подставляем в applicant_name, только если заявитель ещё пуст и оператор не задан вручную.
    """
    if forced_operator_name is not None and str(forced_operator_name).strip():
        return None
    if applicant_name and str(applicant_name).strip():
        return None
    s = (op_raw or "").strip()
    if not s:
        return None
    low = s.lower()
    if low in {
        "не определено",
        "не определен",
        "неизвестно",
        "unknown",
        "none",
        "null",
        "n/a",
        "-",
        "—",
    }:
        return None
    if canonicalize_operator_name(s):
        return None
    if resolved_operator != "Не определено":
        return None
    return _format_applicant_name_candidate(s)


def _resolve_operator(name_from_llm: str, forced: str | None) -> tuple[str, bool]:
    """Возвращает (отображаемое_имя, in_staff). Только штатный список — иначе «Не определено»."""
    if forced is not None and str(forced).strip():
        c = canonicalize_operator_name(str(forced).strip())
        if c:
            return c, True
        return "Не определено", False
    c = canonicalize_operator_name(name_from_llm)
    if c:
        return c, True
    return "Не определено", False


def _checklist_line_is_yes(line: str | None) -> bool:
    """Как в web_gui.render_checklist: после последнего «:» значение начинается с «да»."""
    if not line:
        return False
    _, _, val = line.rpartition(":")
    return (val or "").strip().lower().startswith("да")


def _script_score_from_seven_checklist(lines: list[str]) -> int | None:
    """Та же формула, что в CallQualityEvaluator.evaluate: 7 пунктов × (10/7), max 10."""
    if len(lines) != 7:
        return None
    n_yes = sum(1 for ln in lines if _checklist_line_is_yes(ln))
    return min(10, round(n_yes * (10 / 7)))


def _recalculate_total_score(evaluation) -> None:
    """Пересчитывает total_score после изменения script_score."""
    evaluation.total_score = round(
        0.30 * evaluation.script_score
        + 0.20 * evaluation.speech_score
        + 0.30 * evaluation.consultation_score
        + 0.20 * evaluation.engagement_score
    )


def _find_name_mentions_checklist_index(script_details: list[str]) -> int | None:
    """Индекс строки «Называл заявителя по имени ≥2 раз» (как в промпте LLM)."""
    for i, line in enumerate(script_details):
        if line and "Называл заявителя по имени" in line:
            return i
    return None


def _resolve_applicant_name_for_merge(
    evaluation,
    flat_text: str,
    role_text: str,
) -> str | None:
    """Имя заявителя: сначала из ответа LLM, иначе эвристики CallQualityEvaluator."""
    from transcription import CallQualityEvaluator

    ev = CallQualityEvaluator()
    if evaluation.applicant_name and str(evaluation.applicant_name).strip():
        return str(evaluation.applicant_name).strip()

    applicant_text_raw = ev._extract_role_text(role_text, "Заявитель")
    name = ev._find_applicant_name_from_dialog(role_text)
    if not name and (applicant_text_raw or "").strip():
        name = ev._extract_name_from_applicant_reply(applicant_text_raw)
    if not name:
        name = ev._find_applicant_name(applicant_text_raw or flat_text)
    return name


def apply_deterministic_name_mentions_merge(
    evaluation,
    flat_text: str,
    role_text: str,
    log: Callable[[str], None],
) -> None:
    """
    После облачной оценки: пункт «Называл заявителя по имени ≥2 раз»
    выставляется по подсчёту в репликах оператора (как в локальном оценщике).

    Пересчитываются script_score и total_score.
    """
    from transcription import CallQualityEvaluator

    details = evaluation.script_details
    if len(details) != 7:
        log(
            f"[cloud_eval] Пропуск детермин. имени: script_details не из 7 строк ({len(details)})."
        )
        return

    idx = _find_name_mentions_checklist_index(details)
    if idx is None:
        log("[cloud_eval] Пропуск детермин. имени: нет строки про обращение по имени.")
        return

    ev = CallQualityEvaluator()
    operator_lower = ev._extract_role_text(role_text, "Оператор").lower()
    applicant_lower = ev._extract_role_text(role_text, "Заявитель").lower()

    applicant_name = _resolve_applicant_name_for_merge(evaluation, flat_text, role_text)
    if applicant_name and not (evaluation.applicant_name or "").strip():
        evaluation.applicant_name = applicant_name

    old_line = details[idx] or ""
    _, _, old_val = old_line.rpartition(":")
    old_yes = _checklist_line_is_yes(old_line)

    if not applicant_name:
        log("[cloud_eval] Детермин. имя: заявитель не определён — строка не меняется.")
        return

    uses_ok, op_hits, dialog_hits = ev.applicant_name_checklist_ok(
        operator_lower, applicant_lower, applicant_name
    )
    new_line = (
        f"Называл заявителя по имени ≥2 раз: {'да' if uses_ok else 'нет'}"
        + (f" ({applicant_name})" if applicant_name else "")
        + (
            f" [оп.{op_hits}/всего{dialog_hits}]"
            if applicant_name and (op_hits or dialog_hits)
            else ""
        )
    )

    new_yes = _checklist_line_is_yes(new_line)
    if new_yes != old_yes:
        log(
            f"[cloud_eval] Пункт «имя ≥2 раз»: LLM={(old_val or '').strip()!r} → "
            f"по тексту (оп.{op_hits}, по диалогу {dialog_hits}) → «{'да' if uses_ok else 'нет'}»."
        )

    details[idx] = new_line
    evaluation.script_details = details

    new_script = _script_score_from_seven_checklist(details)
    if new_script is not None:
        evaluation.script_score = new_script
        _recalculate_total_score(evaluation)


def _build_evaluation(data: dict, forced_operator_name: str | None):
    from transcription import QualityEvaluation  # локальный импорт — нет circular dependency

    op_raw = str(data.get("operator_name") or "").strip() or "Не определено"
    operator_name, operator_in_staff = _resolve_operator(op_raw, forced_operator_name)

    app = data.get("applicant_name")
    applicant_name: str | None = str(app).strip() if app and str(app).strip() not in {"null", "None", ""} else None

    promoted = _maybe_promote_nonstaff_operator_name_to_applicant(
        op_raw, operator_name, applicant_name, forced_operator_name
    )
    if promoted:
        applicant_name = promoted

    script_score = _safe_score(data, "script_score")
    speech_score = _safe_score(data, "speech_score")
    consultation_score = _safe_score(data, "consultation_score")
    engagement_score = _safe_score(data, "engagement_score")

    total_score = round(
        0.30 * script_score
        + 0.20 * speech_score
        + 0.30 * consultation_score
        + 0.20 * engagement_score
    )

    script_details = _safe_strlist(data, "script_details")
    if not script_details:
        script_details = ["Детали не извлечены моделью"]

    positives = _safe_strlist(data, "positives")
    if not positives:
        positives = ["Сильные стороны определены LLM — детали в ответе модели."]

    negatives = _safe_strlist(data, "negatives")

    return QualityEvaluation(
        total_score=total_score,
        max_score=10,
        operator_name=operator_name,
        operator_in_staff=operator_in_staff,
        applicant_name=applicant_name,
        script_score=script_score,
        speech_score=speech_score,
        consultation_score=consultation_score,
        engagement_score=engagement_score,
        script_details=script_details,
        positives=positives,
        negatives=negatives,
    )


def _make_ssl_context() -> ssl.SSLContext:
    """SSL-контекст с сертификатами certifi (решает проблему macOS с CA-бандлом)."""
    try:
        import certifi  # noqa: PLC0415
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def _call_yandex_messages(
    system_prompt: str,
    user_content: str,
    cfg: YandexCloudConfig,
    max_tokens: int = 1200,
    temperature: float = 0.1,
) -> str:
    """Универсальный HTTP-запрос к Yandex AI Studio; возвращает текст ответа модели."""
    model_uri = f"gpt://{cfg.folder_id}/{cfg.model}"
    url = f"{_BASE_URL}/chat/completions"
    payload = {
        "model": model_uri,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    body_bytes = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body_bytes,
        method="POST",
        headers={
            "Content-Type": "application/json; charset=utf-8",
            "Authorization": f"Api-Key {cfg.api_key}",
            "x-folder-id": cfg.folder_id,
            "x-data-logging-enabled": "false",
        },
    )
    timeout = max(15.0, min(float(cfg.timeout_seconds), 300.0))
    with urllib.request.urlopen(req, timeout=timeout, context=_make_ssl_context()) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
    body = json.loads(raw)
    choices = body.get("choices") or []
    if not choices:
        raise ValueError("yandex_response:empty_choices")
    msg = choices[0].get("message") or {}
    raw_content = msg.get("content")
    # Редко API возвращает content: null
    if raw_content is None:
        return ""
    if isinstance(raw_content, list):
        parts: list[str] = []
        for p in raw_content:
            if isinstance(p, dict):
                if p.get("type") == "text" and p.get("text"):
                    parts.append(str(p["text"]))
                elif "text" in p and p["text"]:
                    parts.append(str(p["text"]))
            elif isinstance(p, str):
                parts.append(p)
        return "".join(parts)
    return str(raw_content)


def _call_yandex_api(
    flat_text: str | None,
    role_text: str | None,
    cfg: YandexCloudConfig,
    *,
    extra_system: str = "",
) -> str:
    """Запрос для режима облачной оценки (использует eval-промпт).

    При невалидном JSON делает одну повторную попытку с просьбой исправить.
    """
    system_prompt = _SYSTEM_PROMPT + (extra_system or "")
    user_content = _user_prompt(flat_text, role_text)
    content = _call_yandex_messages(
        system_prompt=system_prompt,
        user_content=user_content,
        cfg=cfg,
        max_tokens=2000,
    )
    # Validate JSON — if broken, retry once with a correction request
    try:
        if not (content or "").strip():
            raise ValueError("yandex_response:empty_content")
        _extract_json(content)
        return content
    except (json.JSONDecodeError, ValueError):
        retry_content = _call_yandex_messages(
            system_prompt=system_prompt,
            user_content=(
                f"{user_content}\n\n"
                "ВАЖНО: твой предыдущий ответ содержал невалидный JSON. "
                "Верни ТОЛЬКО валидный JSON без пояснений, без markdown, без комментариев."
            ),
            cfg=cfg,
            max_tokens=2000,
        )
        if not (retry_content or "").strip():
            raise ValueError("yandex_response:empty_content_after_retry")
        return retry_content


def run_yandex_post_edit(
    flat_text: str,
    role_text: str,
    cfg: YandexCloudConfig,
) -> tuple[str | None, str | None, str]:
    """
    Пост-редактирование транскрипта через Yandex AI Studio.
    Возвращает (исправленный_текст, транскрипт_по_ролям, примечание).
    """
    from llm_post_edit import _SYSTEM_PROMPT as _PE_SYS, _parse_llm_blocks, _user_prompt as _pe_user, _validate_output

    user_content = _pe_user(flat_text, role_text)
    try:
        content = _call_yandex_messages(
            system_prompt=_PE_SYS,
            user_content=user_content,
            cfg=cfg,
            max_tokens=16_384,
            temperature=0.08,
        )
    except urllib.error.HTTPError as exc:
        try:
            detail = exc.read().decode("utf-8", errors="replace")[:300]
        except Exception:
            detail = str(exc)
        return None, None, f"http_{exc.code}:{detail}"
    except Exception as exc:
        return None, None, f"request_error:{type(exc).__name__}:{exc}"

    full, roles = _parse_llm_blocks(content)
    if not full or not roles:
        return None, None, "parse_blocks_failed"

    ok, reason = _validate_output(flat_text, full, roles)
    if not ok:
        return None, None, f"validation:{reason}"

    return full, roles, "ok"


def run_yandex_post_edit_threaded(
    flat_text: str,
    role_text: str,
    cfg: YandexCloudConfig,
    wall_timeout: float,
) -> tuple[str | None, str | None, str]:
    """Запускает run_yandex_post_edit в отдельном потоке с wall-timeout."""
    import threading

    wall = max(15.0, float(wall_timeout))
    box: dict[str, object] = {}

    def worker() -> None:
        try:
            f, r, note = run_yandex_post_edit(flat_text, role_text, cfg)
            box["full"] = f
            box["roles"] = r
            box["note"] = note
        except Exception as exc:
            box["note"] = f"worker:{type(exc).__name__}:{exc}"

    th = threading.Thread(target=worker, daemon=True)
    th.start()
    th.join(timeout=wall)
    if th.is_alive():
        return None, None, "timeout"
    return (
        box.get("full") if isinstance(box.get("full"), str) else None,
        box.get("roles") if isinstance(box.get("roles"), str) else None,
        str(box.get("note", "unknown")),
    )


def _ask_operator_name_only(
    role_text: str,
    flat_text: str,
    cfg: YandexCloudConfig,
    _log: Callable[[str], None],
) -> str | None:
    """
    Отдельный лёгкий запрос через тот же API, что и основная оценка (/chat/completions).
    Первые реплики оператора + начало сплошного текста при пустых ролях.
    """
    from operator_staff import OPERATOR_CANONICAL_NAMES, canonicalize_operator_name

    op_lines = [
        l[len("Оператор:"):].strip()
        for l in (role_text or "").splitlines()
        if l.strip().startswith("Оператор:")
    ][:6]
    snippet = "\n".join(op_lines).strip() or (flat_text or "").strip()[:600]
    if not snippet:
        _log("[cloud_eval] name-only: нет текста для запроса.")
        return None

    names_list = ", ".join(OPERATOR_CANONICAL_NAMES)
    system = (
        "Ты извлекаешь имя оператора колл-центра. "
        "Ответь ровно одним словом: каноническое имя из списка пользователя или слово Нет. "
        "Без кавычек, без точки, без пояснений."
    )
    user = (
        f"Возможные операторы (только эти имена): {names_list}.\n"
        "Уменьшительные: Маша→Мария, Настя→Анастасия, Юля→Юлия, Света→Светлана, "
        "Рома→Роман, Ваня→Иван, Дарья/Даша→Даша, Тёма/Артём→Артем, Аля→Алина, Эля→Эльвира.\n\n"
        f"Фрагмент разговора:\n{snippet}\n\n"
        "Как зовут оператора? Одно слово из списка или Нет."
    )

    try:
        raw = _call_yandex_messages(
            system_prompt=system,
            user_content=user,
            cfg=cfg,
            max_tokens=24,
            temperature=0.0,
        )
    except Exception as exc:
        _log(f"[cloud_eval] name-only API ошибка: {type(exc).__name__}: {exc}")
        return None

    answer = (raw or "").strip()
    # Иногда модель возвращает «Имя: Юлия» или короткую фразу — берём последнее слово-имя
    for part in reversed(re.split(r"[\s.,:;!?\n]+", answer)):
        part = part.strip()
        if not part:
            continue
        c = canonicalize_operator_name(part)
        if c:
            return c
    c = canonicalize_operator_name(answer)
    if c:
        return c
    low = answer.lower()
    if low in ("нет", "no", "none"):
        _log("[cloud_eval] name-only: модель ответила «нет».")
        return None
    _log(f"[cloud_eval] name-only: не распознан ответ «{answer[:80]}».")
    return None


def run_cloud_evaluation(
    flat_text: str,
    role_text: str,
    cfg: YandexCloudConfig,
    forced_operator_name: str | None = None,
    log: Callable[[str], None] | None = None,
):
    """
    Оценивает качество звонка через Yandex AI Studio.
    Возвращает (QualityEvaluation, статус_строкой).
    При любой ошибке — fallback на встроенную эвристику.
    """
    from transcription import CallQualityEvaluator  # избегаем circular import на уровне модуля

    _log = log or (lambda m: print(f"[cloud_eval] {m}", flush=True))
    model_uri = f"gpt://{cfg.folder_id}/{cfg.model}"
    _log(f"[cloud_eval] Облачная оценка: модель {model_uri}, таймаут={cfg.timeout_seconds:.0f}s...")

    try:
        extra_sys = _leniency_extra_system_for_forced(forced_operator_name)
        content = _call_yandex_api(flat_text, role_text, cfg, extra_system=extra_sys)
        data = _extract_json(content)
        evaluation = _build_evaluation(data, forced_operator_name)

        # Если основной LLM не нашёл оператора — второй запрос к API (без локального поиска по тексту)
        if evaluation.operator_name in ("Не определено", "", None):
            _log("[cloud_eval] Имя не найдено основным LLM — уточняющий API-запрос для имени…")
            name_only = _ask_operator_name_only(role_text, flat_text, cfg, _log)
            if name_only:
                _log(f"[cloud_eval] Уточняющий запрос вернул: «{name_only}».")
                evaluation.operator_name = name_only
                evaluation.operator_in_staff = True

        # Объективный пункт чек-листа по тексту (модель часто ошибается)
        apply_deterministic_name_mentions_merge(evaluation, flat_text, role_text, _log)

        from evaluation_leniency import apply_focus_operator_adjustments

        evaluation = apply_focus_operator_adjustments(evaluation, _log)

        _log("[cloud_eval] Облачная оценка завершена успешно.")
        return evaluation, "cloud_ok"

    except urllib.error.HTTPError as exc:
        try:
            detail = exc.read().decode("utf-8", errors="replace")[:300]
        except Exception:
            detail = str(exc)
        err = f"http_{exc.code}:{detail}"
    except json.JSONDecodeError as exc:
        err = f"json_parse_error:{exc}"
    except (KeyError, IndexError, TypeError) as exc:
        err = f"response_shape:{type(exc).__name__}:{exc}"
    except Exception as exc:
        err = f"{type(exc).__name__}:{exc}"

    _log(f"[cloud_eval] Ошибка облачной оценки ({err}) — fallback на эвристику.")
    evaluator = CallQualityEvaluator()
    fallback = evaluator.evaluate(flat_text, role_text, forced_operator_name=forced_operator_name)
    return fallback, f"cloud_fallback:{err}"
