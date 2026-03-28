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
from functools import lru_cache
from pathlib import Path
from typing import Callable

from applicant_name_utils import (
    choose_authoritative_applicant_name,
    normalize_applicant_name_candidate,
)
from app_config import load_app_config
from operator_staff import (
    canonicalize_operator_name,
    operator_in_leniency_focus,
    operators_prompt_sentence,
)
from results_paths import project_root
from role_misattribution_fix import fix_operator_thanks_mislabeled_as_applicant

_BASE_URL = "https://llm.api.cloud.yandex.net/v1"
_DEFAULT_MODEL = "yandexgpt-lite"


@dataclass(frozen=True)
class YandexCloudConfig:
    api_key: str
    folder_id: str
    model: str = _DEFAULT_MODEL
    timeout_seconds: float = 60.0


def load_cloud_config_from_env() -> YandexCloudConfig | None:
    return load_app_config().yandex_cloud.to_runtime_config()


@lru_cache(maxsize=1)
def _load_ekc_dialog_standards_excerpt(max_chars: int = 3200) -> str:
    path = project_root() / "Стандарты диалогов ЕКЦ"
    if not Path(path).exists():
        return ""
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""

    keywords = (
        "слова-паразиты",
        "уменьшительно-ласкательные",
        "слишком официальные фразы",
        "обращайтесь к ним по имени",
        "обращайтесь по имени",
        "не грубите мне",
        "не могу",
        "невозможно",
        "информация отсутствует",
        "к сожалению",
    )
    chunks: list[str] = []
    seen: set[str] = set()
    for paragraph in re.split(r"\n\s*\n+", text):
        compact = re.sub(r"\s+", " ", paragraph).strip()
        if not compact:
            continue
        lowered = compact.lower()
        if not any(keyword in lowered for keyword in keywords):
            continue
        if compact in seen:
            continue
        seen.add(compact)
        chunks.append(f"• {compact}")
        if len("\n".join(chunks)) >= max_chars:
            break

    if not chunks:
        return ""
    excerpt = "\n".join(chunks)
    if len(excerpt) > max_chars:
        excerpt = excerpt[:max_chars].rstrip() + "…"
    return (
        "\n\nДОПОЛНИТЕЛЬНЫЕ ТРЕБОВАНИЯ ИЗ ЛОКАЛЬНОГО ФАЙЛА "
        "«Стандарты диалогов ЕКЦ»:\n"
        f"{excerpt}\n"
        "Если эти требования противоречат твоей общей привычке оценивания, выбирай требования ЕКЦ.\n"
    )


_SYSTEM_PROMPT = f"""\
Ты — строгий эксперт QA-оценки звонков Единого Контактного Центра ДВФУ (Дальневосточный федеральный университет, Владивосток).
Тебе дают расшифровку телефонного разговора оператора с заявителем.
Оценивай ТОЛЬКО реплики оператора — не снижай баллы за слова и поведение заявителя.

╔═══════════════════════════════════════════════════════════╗
║  КАЛИБРОВКА ОЦЕНОК — ПРОЧИТАЙ ПЕРЕД ВЫСТАВЛЕНИЕМ БАЛЛОВ  ║
╚═══════════════════════════════════════════════════════════╝
Ты — строгий, но справедливый эксперт. НЕ ЗАВЫШАЙ оценки.
Средний, «нормальный» звонок оператора — это 5 баллов, не 7 и не 8.
Высокие оценки нужно ЗАСЛУЖИТЬ конкретными действиями.

  1–2: грубое нарушение стандартов, критические ошибки
  3–4: заметные нарушения, неполная работа, ощутимые пробелы
  5–6: ОБЫЧНЫЙ адекватный звонок: оператор справился с задачей,
       но без активности, инициативы и «сверхусилий» — это НОРМА, не «хорошо»
  7–8: ХОРОШО: оператор проявил инициативу, предложил альтернативы,
       задавал уточняющие вопросы, резюмировал, обращался по имени
  9–10: ОТЛИЧНО — РЕДКОСТЬ: ВСЕ стандарты соблюдены безупречно,
        оператор активен, внимателен, предложил ≥2 варианта, резюмировал,
        убедился в закрытии вопроса, тепло попрощался по имени

Если оператор просто ответил на вопрос и попрощался — это 5, не 8.
Оценка 9–10 по любому критерию должна быть ИСКЛЮЧЕНИЕМ, а не нормой.
{_load_ekc_dialog_standards_excerpt()}

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
   «да» только если в начале разговора реально есть приветствие И предложение помочь.
   Если приветствие или предложение помощи не найдены в тексте — ставь «нет».

2. ПРЕДСТАВЛЕНИЕ ПО ИМЕНИ
   «да» только если оператор действительно представился по имени в транскрипте.
   Не ставь «да» автоматически по предположению.

3. ЗНАКОМСТВО С ЗАЯВИТЕЛЕМ
   Спросил имя заявителя («как вас зовут?», «как к вам обращаться?», «представьтесь, пожалуйста») ИЛИ
   заявитель представился сам И оператор далее обращался к нему по имени ≥2 раз.
   Если оператор НЕ спросил имя и НЕ использовал имя заявителя — «нет».

4. ОБРАЩЕНИЕ ПО ИМЕНИ В ХОДЕ РАЗГОВОРА
   Просматривай **весь транскрипт целиком** — реплики и Оператора, и Заявителя.
   «да», если выполняется ХОТЯ БЫ ОДНО из условий:
   а) имя заявителя (из поля applicant_name) упоминается в диалоге **≥3 раз суммарно** — в любых репликах;
   б) любое имя, которое НЕ является именем оператора, встречается в диалоге **≥3 раз суммарно**.
   Считай все упоминания по всему тексту. Если имя не звучит или <3 раз — «нет».

5. ПРЕДЛОЖЕНИЕ ДОПОЛНИТЕЛЬНОЙ ПОМОЩИ В КОНЦЕ
   Только в **фазе закрытия** звонка: «чем ещё могу помочь?», «остались ли вопросы?»,
   «могу ли ещё чем-то быть полезен(-на)?», «все ли я для вас сделал(-а)?» и т.п.
   Не требуй этой фразы в середине консультации. Если фразы нет в закрытии — «нет».

6. БЛАГОДАРНОСТЬ ЗА ОБРАЩЕНИЕ
   Только в **фазе закрытия** звонка: «спасибо за обращение», «благодарю за звонок» и аналоги.
   **Фаза закрытия** — последние реплики оператора после завершения консультации или прощания заявителя.
   Ставь «нет», если в закрытии нет благодарности. Если запись обрублена — не наказывай.

7. ПЕРСОНАЛИЗИРОВАННОЕ ПРОЩАНИЕ
   Только в **фазе закрытия**: тёплое прощание по контексту, желательно по имени заявителя
   («всего доброго, [Имя]!», «хорошего дня», «был(-а) рад(-а) помочь», «приятного заселения» и т.п.).
   Стандартное «до свидания» без имени и без тёплой фразы — «нет».
   Если запись оборвана и финала нет — не наказывай.

Шкала script_score (жёсткая):
  10:   все 7 пунктов выполнены
  8–9:  5–6 пунктов выполнены
  6–7:  4 пункта выполнены
  4–5:  2–3 пункта выполнены
  2–3:  1 пункт выполнен
  1:    ни одного пункта

──────────────────────────────────────────────────────────
speech_score — Чистота и культура речи оператора
──────────────────────────────────────────────────────────
Начало: 10 баллов. Снижай ТОЛЬКО за нарушения в репликах ОПЕРАТОРА.

СЛОВА-ПАРАЗИТЫ — «−1» за каждое уникальное слово/оборот:
  ну, эм, ааа, как бы, типа, вот, так сказать, в общем, короче, кстати, просто,
  походу, маленько, грубо говоря, на самом деле, это самое,
  «смотрите» / «послушайте» в качестве вводного слова (без реальной просьбы)

УМЕНЬШИТЕЛЬНО-ЛАСКАТЕЛЬНЫЕ — «−1» за каждое (неуместны в деловой речи):
  минуточку, трубочка, звоночек, заявочка, договорчик, документик, справочка, денежки,
  счётик, актик и любые слова с уменьшительными суффиксами в деловом контексте

ЗАПРЕЩЁННЫЕ ФРАЗЫ (по стандартам ЕКЦ) — «−2» за каждую:
  «я не знаю» / «не знаю» → должно быть: «на текущий момент отсутствует информация, мне потребуется время для уточнения»
  «нет» в начале предложения как форма отказа → конкретная альтернатива
  «невозможно» / «мы не можем» / «не можем это сделать» → «для решения вопроса вам необходимо…»
  «не могу» как прямой отказ → предложи альтернативу
  «вам нужно» / «вы должны» (директивный тон) → «для решения вашего вопроса необходимо…»
  «вы не поняли» / «вы неправильно поняли» → «давайте ещё раз пройдёмся по пунктам»
  «ваша проблема» → «мы поможем в решении вашего вопроса» (слово «проблема» заменяем на «вопрос»)
  «вы не правы» → корректная альтернатива
  «вас беспокоит» → некорректная формулировка
  «к сожалению» → вместо сочувствия отказу — предлагай решение
  «не могу подсказать» / «это не ко мне» / «не моя зона ответственности» → помоги найти решение
  «передадим ваше обращение директору центра коммуникаций» → «зафиксируем обращение и передадим руководству для решения»
  Нечёткие сроки: «недели через две», «скоро», «потом» → конкретные сроки

РЕЧЕВЫЕ ПОВТОРЫ — «−1» если оператор многократно использует одно и то же слово/фразу
  (однообразная, «конвейерная» речь без синонимов).

ОБРАЩЕНИЕ ПО ИМЕНИ И НЕКОРРЕКТНЫЕ ОБРАЩЕНИЯ:
  если имя заявителя известно, но оператор не обращается по имени — это отдельное замечание;
  если вместо имени используются «девушка», «женщина», «мужчина», «молодой человек», «клиент», «абонент» —
  это некорректное обращение и отдельное снижение.

Минимум speech_score: 0 баллов.

──────────────────────────────────────────────────────────
consultation_score — Качество и полнота консультации
──────────────────────────────────────────────────────────
Оценивай по совокупности критериев «Золотых правил ЕКЦ». Помни: обычный ответ = 5, не 7.

▸ Правило активности: оператор удерживает инициативу, предоставляет полную информацию,
  не ждёт пока заявитель сам «вытянет» данные. Не молчит, не уходит в паузы без объяснения.
  Ошибки: отвечает слишком кратко, «молчит в трубку», теряется, не знает что сказать.

▸ Правило правильной информации: не использует «нет», «невозможно», «не могу» как отказ.
  Ищет и предлагает ≥2 варианта. При невозможности решить «здесь и сейчас» — чётко объясняет
  когда, где и при каких условиях заявитель получит ответ. Предлагает альтернативу:
  «запишите номер телефона данного отдела, а также адрес электронной почты».

▸ Правило удержания (hold): если уходит на удержание —
  а) объясняет причину («уточню информацию по вашему вопросу, оставайтесь на линии»);
  б) не более 1 минуты (по стандарту); если дольше — предлагает перезвонить;
  в) не бросает на «тишину» без объяснения — это ГРУБОЕ нарушение (−3 балла сразу);
  г) при повторном удержании: «мне потребуется ещё некоторое время».

▸ Резюме и проверка: завершая консультацию, оператор должен:
  — резюмировать ключевые моменты («итак, вам необходимо…»);
  — убедиться что вопрос закрыт («я ответил(-а) на ваш вопрос?», «чем ещё могу помочь?»).
  Без этого — максимум 7 баллов.

Шкала consultation_score (жёсткая):
  10:   полный ответ, конкретные шаги/документы/сроки, ≥2 варианта решения,
        резюме, проверка закрытия, инициатива, идеальная процедура удержания
  8–9:  полный ответ с конкретикой, но нет резюме ИЛИ нет проверки закрытия
  6–7:  ответил по существу, но: только 1 вариант решения, ИЛИ не задал уточняющих вопросов,
        ИЛИ заявитель переспрашивал по 2+ раза, ИЛИ нечёткие сроки
  5:    стандартный ответ без инициативы: сказал что знал, не предложил альтернатив,
        не проверил понимание — ТИПИЧНЫЙ звонок
  3–4:  перенаправил без объяснения ИЛИ использовал «нет/невозможно» без альтернативы ИЛИ
        общие слова без конкретики ИЛИ бросил на удержание >1 мин без предупреждения
  1–2:  «не знаю» без уточнения, отказ помочь, заявитель оставлен без ответа

──────────────────────────────────────────────────────────
engagement_score — Вовлечённость, эмоциональность и культура общения
──────────────────────────────────────────────────────────
Оценивай по «Золотым правилам ЕКЦ». Помни: механический ответ без эмоций = 5, не 7.

▸ Правило внимательности: не перебивает, не переспрашивает уже озвученное,
  запоминает детали. Переспрашивает только если заявитель нечётко сформулировал.
  Ошибки: перебивает, быстро говорит, невнимательно слушает, задаёт повторные вопросы.

▸ Правило заинтересованности: задаёт уточняющие вопросы («правильно ли я вас понял(-а)…»,
  «что именно происходит?»), использует одобряющие фразы («да», «верно», «я вас понял(-а)»),
  управляет разговором вопросами, предлагает альтернативы.
  Ошибки: не задаёт вопросов, сразу перенаправляет в другое подразделение без уточнения,
  категорично отказывает, не проявляет желания помочь.

▸ Правило доброжелательности: оператор ВСЕГДА эмоционально «на ступень выше» собеседника:
  — собеседник доброжелателен → оператор ОЧЕНЬ доброжелателен;
  — собеседник нейтрален → оператор доброжелателен;
  — собеседник агрессивен/раздражён → оператор КАК МИНИМУМ нейтрально-вежлив.
  Использует слова вежливости, благодарит за ожидание.
  Ошибки: слова-раздражители, не использует слова вежливости, повышает голос.

▸ Правило корректности: спокойствие в любой ситуации, не критикует клиента,
  не делает замечаний, не переходит на личный стиль, не выражает недовольства.
  НЕДОПУСТИМО: раздражённые интонации, «не грубите мне», разговор «свысока», смех/заигрывания.

▸ Правило комфорта и понятной коммуникации: без аббревиатур, сложных терминов,
  жаргонизмов. Подстраивается под темп и словарный запас заявителя.

▸ При конфликтном / недовольном заявителе — 4 ОБЯЗАТЕЛЬНЫХ шага:
  1. Выслушать без перебивания
  2. Выразить сочувствие: «я понимаю, это неприятно», «мне очень жаль, что так сложилось»
  3. Уточнить детали
  4. Предложить конкретное решение или эскалацию
  Пропуск хотя бы одного шага при конфликтном заявителе — снижение на 2+ балла.

Шкала engagement_score (жёсткая):
  10:   все правила безупречно: внимателен, заинтересован, доброжелателен, корректен,
        понятен; при конфликте — все 4 шага; ≥3 уточняющих вопроса; одобряющие фразы;
        обращение по имени; эмоционально «выше» собеседника
  8–9:  все правила в целом соблюдены, уточняющие вопросы есть, но не более 1–2;
        одобряющие фразы есть; при конфликте — минимум 3 из 4 шагов
  6–7:  уточняющие вопросы есть, но оператор в основном пассивен:
        отвечает на вопросы, но сам не проявляет инициативу; вежлив, но без тепла
  5:    ТИПИЧНЫЙ звонок: оператор вежлив, но механичен; нет уточняющих вопросов;
        нет одобряющих фраз; просто ответил и попрощался
  3–4:  не задавал вопросов, молчал в паузах, перебивал, использовал термины без объяснения;
        свысока или отстранённо; при конфликте — не прошёл ни одного из 4 шагов
  1–2:  повысил голос, критиковал, замечания, безразличие, грубость

═══════════════════════════════════════════════════════════
ЗАДАЧА 3 — Сформируй итоговый вывод
═══════════════════════════════════════════════════════════
• script_details — ровно 7 строк «Пункт: да» или «Пункт: нет» (точно в том же порядке и формулировках).
  НЕ СТАВЬ «да» если не нашёл конкретной фразы/действия в тексте.
• positives — 2–4 конкретные сильные стороны с примером-цитатой из диалога оператора.
  Если ничего особенного — можешь указать 1–2 пункта. НЕ ВЫДУМЫВАЙ позитивы.
• negatives — 2–5 замечаний в формате: «[Нарушение]: "[цитата из диалога]" — [рекомендация]».
  Будь критичен: ищи РЕАЛЬНЫЕ недоработки, а не формальные придирки.
  Если нарушений нет — верни пустой массив [].
  Цитируй только реплики оператора, не заявителя.
  Для пунктов 5–7 (фаза закрытия) — цитируй только конец разговора.
  Если нашёл слова-паразиты, уменьшительные формы, слова-раздражители, слишком официальные фразы,
  некорректные обращения или отсутствие обращения по имени — ОБЯЗАТЕЛЬНО добавь их в negatives отдельными пунктами.

ВАЖНО: перед выставлением баллов спроси себя:
  «Этот звонок действительно ОТЛИЧНЫЙ (9–10), или оператор просто сделал минимум?»
  Если оператор просто ответил на вопрос — это 5, не 8.

═══════════════════════════════════════════════════════════
ФОРМАТ ОТВЕТА — только валидный JSON, без markdown, без комментариев
═══════════════════════════════════════════════════════════
{{
  "operator_name": "Егор или Не определено (только имя из штатного списка)",
  "applicant_name": "Имя или null",
  "script_score": 5,
  "speech_score": 8,
  "consultation_score": 5,
  "engagement_score": 5,
  "script_details": [
    "Приветствие + предложение помочь: да",
    "Представился по имени: да",
    "Познакомился с заявителем: нет",
    "Называл заявителя по имени ≥2 раз: нет",
    "Предложил дополнительную помощь: нет",
    "Поблагодарил за обращение: да",
    "Персонализированное прощание: нет"
  ],
  "positives": ["Оператор предоставил корректную информацию по вопросу заявителя"],
  "negatives": ["Не спросил имя заявителя и не обращался по имени — рекомендуется уточнять имя в начале разговора", "Не предложил дополнительную помощь перед прощанием — рекомендуется спрашивать \\"Чем ещё могу помочь?\\""]
}}"""


def _format_user_eval_instructions(raw: str | None, *, max_chars: int = 2500) -> str:
    """Блок для system prompt: доп. указания оператора (пакет / ручной ввод)."""
    s = (raw or "").strip()
    if not s:
        return ""
    if len(s) > max_chars:
        s = s[:max_chars].rstrip() + "\n…(текст обрезан по лимиту)"
    return (
        "\n\n---\nДОПОЛНИТЕЛЬНЫЕ УКАЗАНИЯ ОТ ПОЛЬЗОВАТЕЛЯ:\n"
        f"{s}\n\n"
        "Учитывай их при оценке по всем критериям и при формировании positives/negatives. "
        "Не ослабляй обязательные правила ЕКЦ и формат ответа (только валидный JSON).\n---\n"
    )


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
    lines = [line for line in rt.splitlines() if line.strip().startswith("Оператор:")]
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


def _format_applicant_name_candidate(raw: str) -> str | None:
    """Если строка похожа на имя человека (не оператор из штата) — нормализует для applicant_name."""
    return normalize_applicant_name_candidate(raw)


def _extract_applicant_name_from_any_json_or_text(raw: str) -> str | None:
    """Пытается извлечь candidate applicant_name из JSON или простого текста ответа LLM."""
    s = (raw or "").strip()
    if not s:
        return None
    try:
        data = _extract_json(s)
        v = data.get("applicant_name")
        if v is None:
            return None
        return _format_applicant_name_candidate(str(v))
    except Exception:
        pass
    cleaned = re.sub(r"(?i)^applicant_name\s*[:=]\s*", "", s).strip()
    if cleaned.lower() in {"null", "none", "нет", "не определено", "неизвестно"}:
        return None
    return _format_applicant_name_candidate(cleaned)


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
    flat_text: str,
    role_text: str,
) -> str | None:
    """Имя заявителя только по тексту диалога, без доверия к текущему полю evaluation."""
    from transcription import CallQualityEvaluator

    ev = CallQualityEvaluator()
    applicant_text_raw = ev._extract_role_text(role_text, "Заявитель")
    name = ev._find_applicant_name_from_dialog(role_text)
    if not name and (applicant_text_raw or "").strip():
        name = ev._extract_name_from_applicant_reply(applicant_text_raw)
    if not name:
        name = ev._find_applicant_name(applicant_text_raw or flat_text)
    return normalize_applicant_name_candidate(name or "")


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

    applicant_name = choose_authoritative_applicant_name(
        evaluation.applicant_name,
        _resolve_applicant_name_for_merge(flat_text, role_text),
    )
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


def _dedupe_keep_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        token = (item or "").strip()
        if not token or token in seen:
            continue
        seen.add(token)
        out.append(token)
    return out


def apply_deterministic_speech_findings_merge(
    evaluation,
    flat_text: str,
    role_text: str,
    forced_operator_name: str | None,
    log: Callable[[str], None],
) -> None:
    """
    После облачной оценки: дотягиваем объективные речевые замечания из локального evaluator.
    Это нужно, чтобы LLM не пропускал слова-паразиты, некорректные обращения и отсутствие имени.
    """
    from transcription import CallQualityEvaluator

    heuristic = CallQualityEvaluator().evaluate(
        flat_text,
        role_text,
        forced_operator_name=forced_operator_name,
        forced_applicant_name=choose_authoritative_applicant_name(
            evaluation.applicant_name,
            _resolve_applicant_name_for_merge(flat_text, role_text),
        ),
    )
    detailed_prefixes = (
        "Слова-паразиты:",
        "Уменьшительно-ласкательные формы:",
        "Категоричные/беспомощные формулировки:",
        "Слова-раздражители и некорректные фразы:",
        "Слишком официальные формулировки:",
        "Некорректные обращения вместо имени:",
        "Обращение по имени:",
        "Обращение по имени не выполнено:",
        "Речевые повторы:",
    )
    additions = [
        item
        for item in heuristic.negatives
        if any(item.startswith(prefix) for prefix in detailed_prefixes)
    ]
    if additions:
        before_count = len(evaluation.negatives)
        evaluation.negatives = _dedupe_keep_order(list(evaluation.negatives) + additions)
        added_count = len(evaluation.negatives) - before_count
        if added_count > 0:
            log(f"[cloud_eval] Добавлено детермин. речевых замечаний: {added_count}.")

    if heuristic.speech_score < evaluation.speech_score:
        log(
            f"[cloud_eval] speech_score снижен по детермин. правилам: "
            f"{evaluation.speech_score} → {heuristic.speech_score}."
        )
        evaluation.speech_score = heuristic.speech_score
        _recalculate_total_score(evaluation)


def _build_evaluation(data: dict, forced_operator_name: str | None):
    from transcription import QualityEvaluation  # локальный импорт — нет circular dependency

    op_raw = str(data.get("operator_name") or "").strip() or "Не определено"
    operator_name, operator_in_staff = _resolve_operator(op_raw, forced_operator_name)

    app = data.get("applicant_name")
    applicant_name: str | None = None
    if app and str(app).strip() not in {"null", "None", ""}:
        applicant_name = _format_applicant_name_candidate(str(app))

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

    roles, pe_fix = fix_operator_thanks_mislabeled_as_applicant(roles)
    fix_sfx = ":autofix_operator_thanks" if pe_fix else ""

    ok, reason = _validate_output(flat_text, full, roles)
    if not ok:
        return None, None, f"validation:{reason}"

    return full, roles, f"ok{fix_sfx}"


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
        line[len("Оператор:"):].strip()
        for line in (role_text or "").splitlines()
        if line.strip().startswith("Оператор:")
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


def _ask_applicant_name_only(
    role_text: str,
    flat_text: str,
    cfg: YandexCloudConfig,
    _log: Callable[[str], None],
    current_candidate: str | None = None,
) -> str | None:
    """
    Отдельный API-запрос на имя заявителя + строгая валидация.
    Возвращает только валидное человеческое имя или None.
    """
    role_lines = [line for line in (role_text or "").splitlines() if line.strip()]
    role_snippet = "\n".join(role_lines[:24]).strip()
    flat_snippet = (flat_text or "").strip()[:1200]
    snippet = role_snippet or flat_snippet
    if not snippet:
        _log("[cloud_eval] applicant-name-only: нет текста для запроса.")
        return None

    system = (
        "Ты извлекаешь ИМЯ ЗАЯВИТЕЛЯ из телефонного диалога. "
        "Верни ТОЛЬКО JSON формата {\"applicant_name\": \"Имя\"} или {\"applicant_name\": null}. "
        "В applicant_name допускаются только личные имена/имя+фамилия человека. "
        "Запрещено возвращать обычные слова, дни недели, местоимения, вводные слова или фразы "
        "(например: «заявление», «понимаю», «вопрос», «спасибо», «Воскресенье», «Такое», «Добрый день»)."
    )
    if current_candidate:
        user = (
            "Проверь гипотезу по имени заявителя и при необходимости исправь её. "
            "Если гипотеза неверна или в диалоге имя не названо явно — верни null.\n\n"
            f"Текущая гипотеза: {current_candidate}\n\n"
            f"Фрагмент:\n{snippet}\n\n"
            "Ответ: только JSON."
        )
    else:
        user = (
            "Выдели имя заявителя из фрагмента разговора. "
            "Если имя не названо явно или есть сомнения — верни null.\n\n"
            f"Фрагмент:\n{snippet}\n\n"
            "Ответ: только JSON."
        )
    try:
        raw = _call_yandex_messages(
            system_prompt=system,
            user_content=user,
            cfg=cfg,
            max_tokens=80,
            temperature=0.0,
        )
    except Exception as exc:
        _log(f"[cloud_eval] applicant-name-only API ошибка: {type(exc).__name__}: {exc}")
        return None

    candidate = _extract_applicant_name_from_any_json_or_text(raw)
    if candidate:
        return candidate
    _log("[cloud_eval] applicant-name-only: валидное имя не найдено.")
    return None


def run_cloud_evaluation(
    flat_text: str,
    role_text: str,
    cfg: YandexCloudConfig,
    forced_operator_name: str | None = None,
    log: Callable[[str], None] | None = None,
    *,
    extra_eval_instructions: str | None = None,
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
        extra_sys = (
            _leniency_extra_system_for_forced(forced_operator_name)
            + _format_user_eval_instructions(extra_eval_instructions)
        )
        if (extra_eval_instructions or "").strip():
            _log("[cloud_eval] Учитываются дополнительные указания пользователя для оценки.")
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

        current_applicant = normalize_applicant_name_candidate(str(evaluation.applicant_name or ""))
        if current_applicant:
            _log("[cloud_eval] Дополнительная нейровалидация applicant_name…")
        else:
            _log("[cloud_eval] Имя заявителя пустое/невалидное — уточняющий API-запрос…")
        applicant_only = _ask_applicant_name_only(
            role_text,
            flat_text,
            cfg,
            _log,
            current_candidate=current_applicant,
        )
        heuristic_applicant = _resolve_applicant_name_for_merge(flat_text, role_text)
        if applicant_only:
            resolved_applicant = applicant_only
        elif current_applicant and heuristic_applicant == current_applicant:
            resolved_applicant = current_applicant
        else:
            resolved_applicant = heuristic_applicant
        if applicant_only and applicant_only != current_applicant:
            _log(f"[cloud_eval] Нейровалидация applicant_name уточнила имя: «{applicant_only}».")
        elif current_applicant and current_applicant != resolved_applicant:
            _log(
                f"[cloud_eval] applicant_name «{current_applicant}» заменено на "
                f"«{resolved_applicant or 'null'}» после валидации."
            )
        evaluation.applicant_name = resolved_applicant

        # Объективный пункт чек-листа по тексту (модель часто ошибается)
        apply_deterministic_name_mentions_merge(evaluation, flat_text, role_text, _log)
        apply_deterministic_speech_findings_merge(
            evaluation,
            flat_text,
            role_text,
            forced_operator_name,
            _log,
        )

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
