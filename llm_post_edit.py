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


_SYSTEM_PROMPT = """Ты редактор расшифровки телефонного разговора колл-центра ДВФУ (русский язык).
Задача: привести текст к читаемому виду и ТОЧНО определить роли спикеров.

РОЛИ:
- «Оператор:» — сотрудник ДВФУ, принял входящий звонок.
  Признаки: здравствуйте / меня зовут / слушаю вас / добрый день,
  вам необходимо / нужно, обратитесь / позвоните / принесите / предоставьте,
  уточню / одну секунду / подождите, спасибо за обращение,
  говорит «вам» / «вас» / «ваши» обращаясь к звонящему, даёт инструкции и объяснения.
- «Заявитель:» — человек, который позвонил.
  Признаки: я хочу узнать / подскажите / у меня вопрос / скажите пожалуйста,
  мне нужно / меня интересует / можно ли,
  говорит «я» / «мне» / «меня» / «мой» о себе и своей ситуации, задаёт вопросы.

Правила:
- Первый говорящий ВСЕГДА оператор — он принимает входящий звонок.
- Используй контекст ВСЕГО разговора для исправления ошибок определения ролей.
- Сохраняй смысл и факты; не выдумывай то, чего не было в исходнике.
- Исправь ошибки распознавания речи, расставь знаки препинания, заглавные буквы по правилам русского.
- Только метки «Оператор:» и «Заявитель:» в начале каждой реплики (новая строка перед каждой меткой).
- Не добавляй комментарии редактора и никаких других меток."""


def _user_prompt(flat_text: str, role_text: str) -> str:
    return f"""Ниже автоматическая расшифровка. Исправь и отформатируй.

СПЛОШНОЙ ТЕКСТ (как распозналось):
{flat_text.strip()}

ТРАНСКРИПТ ПО РОЛЯМ (черновик):
{role_text.strip()}

Верни СТРОГО два блока подряд, без текста до или после них:

<<<FULL>>>
(здесь исправленный сплошной текст одним или несколькими абзацами)

<<<ROLES>>>
(здесь только строки вида)
Оператор: ...
Заявитель: ...
(чередуй роли по смыслу; каждая новая реплика с новой строки с меткой Оператор: или Заявитель:)
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
        "temperature": 0.15,
        "max_tokens": 16_384,
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

    ok, reason = _validate_output(flat_text, full, roles)
    if not ok:
        return None, None, f"validation:{reason}"

    return full, roles, "ok"


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
