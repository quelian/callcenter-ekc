"""
LLM-постредактор транскрипта (локально: Ollama, LM Studio, vLLM и т.д.).

Ожидается OpenAI-совместимый endpoint ``/chat/completions``.
Пример Ollama: ``http://127.0.0.1:11434/v1``, модель ``llama3.2`` / ``qwen2.5`` и т.п.

Переменные окружения (опционально, можно переопределить из Transcriber):
  LLM_POST_EDIT_BASE_URL
  LLM_POST_EDIT_MODEL
  LLM_POST_EDIT_API_KEY  — если пусто, заголовок Authorization не шлём
  LLM_POST_EDIT_TIMEOUT_SECONDS
"""

from __future__ import annotations

import json
import os
import re
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass, replace

from role_misattribution_fix import (
    applicant_still_has_operator_thanks,
    fix_operator_thanks_mislabeled_as_applicant,
)


@dataclass(frozen=True)
class LLMPostEditConfig:
    base_url: str
    model: str
    api_key: str | None
    timeout_seconds: float


def load_llm_config_from_env() -> LLMPostEditConfig | None:
    base = (os.environ.get("LLM_POST_EDIT_BASE_URL") or "").strip().rstrip("/")
    if not base:
        return None
    model = (os.environ.get("LLM_POST_EDIT_MODEL") or "").strip()
    if not model:
        return None
    key = (os.environ.get("LLM_POST_EDIT_API_KEY") or "").strip() or None
    try:
        timeout = float(os.environ.get("LLM_POST_EDIT_TIMEOUT_SECONDS") or "120")
    except ValueError:
        timeout = 120.0
    timeout = max(15.0, min(timeout, 600.0))
    return LLMPostEditConfig(base_url=base, model=model, api_key=key, timeout_seconds=timeout)


def _token_set(text: str) -> set[str]:
    return set(re.findall(r"[а-яёa-z0-9]+", (text or "").lower()))


def _validate_output(original_flat: str, new_flat: str, new_roles: str) -> tuple[bool, str]:
    o = (original_flat or "").strip()
    nf = (new_flat or "").strip()
    nr = (new_roles or "").strip()
    if len(nf) < 20:
        return False, "full_too_short"
    if len(nr) < 20:
        return False, "roles_too_short"
    if "Оператор:" not in nr and "Заявитель:" not in nr:
        return False, "missing_role_prefixes"
    if applicant_still_has_operator_thanks(nr):
        return False, "applicant_has_operator_closing_formula"
    ts_o = _token_set(o)
    ts_n = _token_set(nf)
    if not ts_o:
        return True, "ok"
    overlap = len(ts_o & ts_n) / max(1, len(ts_o))
    if overlap < 0.28:
        return False, f"token_overlap_{overlap:.2f}"
    wo = len(o.split())
    wn = len(nf.split())
    if wo > 0:
        ratio = wn / max(1, wo)
        if ratio < 0.45 or ratio > 1.75:
            return False, f"word_ratio_{ratio:.2f}"
    return True, "ok"


_SYSTEM_PROMPT = """Ты ведущий редактор расшифровки телефонного разговора: единый контактный центр / приёмная ДВФУ, русский язык.

## Задача (две части)
1) **Текст:** исправить типичные ошибки ASR (пропуск/лишние буквы, слитые слова, «как слышно»), расставить пунктуацию и заглавные по нормам русского, сохранить факты, даты, числа и названия как в оригинале (можно исправить очевидную опечатку распознавания).
2) **Роли:** заново определить, **кто сказал каждую реплику по смыслу и роли в разговоре**. Черновик по ролям и метки спикеров в ASR часто ошибаются — **не копируй их слепо**. Прочитай весь диалог и для каждой фразы спроси себя: это реплика **сотрудника линии** или **звонящего с личной потребностью**?

## Распределение ролей по смыслу (главный приоритет!!! ЭТО САМОЕ ВАЖНОЕ ПРАВИЛО!!!)

**Оператор (смысл):** представитель организации; **отвечает за линию**; снимает звонок первым; **информирует и направляет** звонящего по правилам вуза; даёт **алгоритм действий**, сроки, адреса, куда обратиться, что принести; **уточняет** данные и перезванивает; задаёт служебные вопросы («как вас зовут», «как к вам обращаться»); в закрытии — **официальная благодарность за обращение**, «чем ещё могу помочь», прощание от имени линии. Типично много обращений **«вы / вам / ваш / ваши»** к собеседнику и формулировок **«необходимо», «можете обратиться», «уточню», «оставайтесь на линии»**.

**Заявитель (смысл):** **клиент**; звонит **со своей ситуацией**; формулирует **свою потребность** («мне нужно», «я хотел узнать», «у меня вопрос», «как мне оформить»); **спрашивает** о документах, сроках, порядке, льготах, общежитии; отвечает на вопросы оператора **о себе** (имя, ситуация); короткие реплики **«да», «угу», «понятно», «спасибо»** чаще заявитель **после** объяснения оператора. Типично **«я / мне / мой / у меня»** про своё дело (не путать с «я уточню» у оператора).

**Как решать спорные места (обязательно применяй):**
- **Функция реплики важнее длины:** длинный монолог с инструкциями и требованиями к звонящему — почти всегда **Оператор**; монолог с описанием своей проблемы и просьбой помочь — **Заявитель**.
- **Вопрос — не всегда заявитель:** «Как вас зовут?» / «Уточните дату» — **Оператор**; «А до какого числа подача?» / «Мне нужна справка, как её получить?» — **Заявитель**.
- **Смена говорящего:** после вопроса оператора о имени следующая реплика с именем или фамилией — **Заявитель**; после вопроса заявителя о правилах — развёрнутый ответ — **Оператор**.
- **Не путай канцелярское «спасибо за обращение»** (только оператор) с **коротким «спасибо»** клиента.
- Если одна и та же фраза по смыслу явно не может быть от заявителя (например, закрытие линии) — перенеси под **Оператор**, даже если в черновике стоит иначе.

## Жёсткие правила (нарушать нельзя)
- Первый говорящий в звонке — **всегда Оператор** (входящая линия).
- Официальное закрытие с благодарностью **за обращение** говорит **только оператор**. **Никогда** после «Заявитель:»: «Спасибо вам за обращение», «Благодарим вас за обращение», «Спасибо за ваше обращение» / «Спасибо за обращение» в смысле завершения звонка линии. Заявитель: «спасибо», «спасибо большое», «всего доброго» — **без** шаблона «… за обращение».
- Не придумывай факты, цифры и целые реплики, которых не было в исходнике.
- Не меняй **порядок** смысловых реплик; можно склеить только явный обрыв **одного** говорящего в соседних строках черновика.

## Формат вывода (строго)
- В блоке ролей — **только** строки «Оператор:» или «Заявитель:», каждая новая реплика с новой строки.
- Без комментариев, примечаний редактора, markdown и лишних меток."""


def _user_prompt(flat_text: str, role_text: str) -> str:
    return f"""Ниже сырой вывод распознавания и черновик по ролям. Приведи оба фрагмента к финальному виду.

── СПЛОШНОЙ ТЕКСТ (ASR) ──
{flat_text.strip()}

── ЧЕРНОВИК ПО РОЛЯМ ──
{role_text.strip()}

**Роли:** пройди диалог от начала до конца и для **каждой** реплики реши по смыслу: это голос **сотрудника ЕКЦ (Оператор)** или **звонящего (Заявитель)**? Сверяйся со **сплошным текстом** и с черновиком; если черновик противоречит смыслу — **исправь метку** в <<<ROLES>>>.

Чек-лист перед выводом <<<ROLES>>>:
1) Первая реплика — **Оператор**.
2) Инструкции «вам нужно / обратитесь / принесите / уточню / перезвоню» — **Оператор**; запросы «мне нужно / как подать / подскажите по документам» — **Заявитель** (если это не служебный вопрос оператора).
3) «Спасибо/благодарим … **за обращение**» (официальное) — только **Оператор**; короткое «спасибо» клиента — **Заявитель**.
4) Нет «рваного» чередования без причины: после длинного ответа оператора короткое «да/понятно» — обычно **Заявитель**.
5) Смысл и факты совпадают с исходником; исправлен только текст ASR и **ошибочные** метки ролей.

Верни СТРОГО два блока подряд (без текста до <<<FULL>>> и после последней реплики в <<<ROLES>>>):

<<<FULL>>>
… исправленный связный текст звонка (абзацы по смыслу) …

<<<ROLES>>>
Оператор: …
Заявитель: …
Оператор: …
(и так далее; только эти две метки, каждая реплика с новой строки)
"""


def _parse_llm_blocks(content: str) -> tuple[str | None, str | None]:
    if not content:
        return None, None
    m = re.search(
        r"<<<FULL>>>\s*(.*?)\s*<<<ROLES>>>\s*(.*)",
        content,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if not m:
        return None, None
    full = m.group(1).strip()
    roles = m.group(2).strip()
    # убрать возможный хвост «пояснений»
    roles = re.split(r"\n\s*(?:Примечание|Note|P\.S\.)", roles, maxsplit=1, flags=re.IGNORECASE)[0].strip()
    return full or None, roles or None


def _chat_completions_url(base_url: str) -> str:
    return f"{base_url}/chat/completions"


def _post_json(url: str, payload: dict, api_key: str | None, timeout: float) -> dict:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        method="POST",
        headers={
            "Content-Type": "application/json; charset=utf-8",
            **({"Authorization": f"Bearer {api_key}"} if api_key else {}),
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
    return json.loads(raw)


def run_llm_post_edit(
    flat_text: str,
    role_transcript: str,
    config: LLMPostEditConfig,
) -> tuple[str | None, str | None, str]:
    """
    Синхронный вызов. Для таймаута оборачивайте в поток на стороне вызывающего.
    Возвращает (новый_сплошной, новый_по_ролям, примечание_диагностики).
    """
    url = _chat_completions_url(config.base_url)
    payload = {
        "model": config.model,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": _user_prompt(flat_text, role_transcript)},
        ],
        "temperature": 0.08,
        "max_tokens": 24_576,
    }
    try:
        body = _post_json(url, payload, config.api_key, config.timeout_seconds)
    except urllib.error.HTTPError as exc:
        try:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
        except Exception:
            detail = str(exc)
        return None, None, f"http_{exc.code}:{detail}"
    except Exception as exc:
        return None, None, f"request_error:{type(exc).__name__}:{exc}"

    try:
        content = body["choices"][0]["message"]["content"]
        if not isinstance(content, str):
            return None, None, "bad_response_shape"
    except (KeyError, IndexError, TypeError):
        return None, None, "no_choices"

    full, roles = _parse_llm_blocks(content)
    if not full or not roles:
        return None, None, "parse_blocks_failed"

    roles, closing_fixed = fix_operator_thanks_mislabeled_as_applicant(roles)
    fix_note = ":autofix_operator_thanks" if closing_fixed else ""

    ok, reason = _validate_output(flat_text, full, roles)
    if not ok:
        return None, None, f"validation:{reason}"

    return full, roles, f"ok{fix_note}"


def run_llm_post_edit_threaded(
    flat_text: str,
    role_transcript: str,
    config: LLMPostEditConfig,
    wall_timeout: float,
) -> tuple[str | None, str | None, str]:
    """Вызов в отдельном потоке с жёстким wall-timeout."""
    wall = max(15.0, float(wall_timeout))
    cfg = replace(config, timeout_seconds=min(config.timeout_seconds, wall))
    box: dict[str, object] = {}

    def worker() -> None:
        try:
            f, r, note = run_llm_post_edit(flat_text, role_transcript, cfg)
            box["full"] = f
            box["roles"] = r
            box["note"] = note
        except Exception as exc:
            box["note"] = f"worker:{type(exc).__name__}:{exc}"

    th = threading.Thread(target=worker, daemon=True)
    th.start()
    th.join(timeout=max(5.0, wall_timeout))
    if th.is_alive():
        return None, None, "timeout"
    return (
        box.get("full") if isinstance(box.get("full"), str) else None,
        box.get("roles") if isinstance(box.get("roles"), str) else None,
        str(box.get("note", "unknown")),
    )
