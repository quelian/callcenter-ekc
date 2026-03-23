"""
Экспорт строк результатов QA в Google Таблицу (через сервисный аккаунт).

Колонки (шапка таблицы):
  A  — №                         (порядковый номер)
  B  — Дата                      (из имени файла DD-MM-YYYY)
  C  — Оператор                  (имя из QA-оценки)
  D  — Телефон                   (из имени файла)
  E  — Заявитель                 (имя из QA-оценки)
  F  — Ожидание MM:SS            (из имени файла)
  G  — Итог /10                  (total_score)
  H  — Скрипт /10               (script_score)
  I  — Речь /10                  (speech_score)
  J  — Консультация /10         (consultation_score)
  K  — Вовлечённость /10        (engagement_score)
  L  — Тон                       (tone_summary)
  M  — Чек-лист                  (script_details, строки через \n)
  N  — Плюсы                     (positives через \n)
  O  — Минусы                    (negatives через \n)
  P  — Требует прослушивания     (Да / Нет — авто по оценке и нарушениям)

Настройка:
  1. Google Cloud: сервисный аккаунт, JSON-ключ.
  2. Таблицу «Поделиться» с email сервисного аккаунта (редактор).
  3. В .env: GOOGLE_SHEETS_CREDENTIALS_JSON=/abs/path/to/key.json
     опционально GOOGLE_SHEETS_SPREADSHEET_ID
     опционально GOOGLE_SHEETS_WORKSHEET — имя листа (по умолчанию первый)
     GOOGLE_SHEETS_HAS_HEADER=true — если в 1-й строке заголовки
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

DEFAULT_SPREADSHEET_ID = "171ix3FDr7kWzMZMykOwYTd5f6jZ7j3tjJEKzF7W4_1M"

_DEFAULT_HAS_HEADER = True

# Кеш открытого листа: меньше авторизаций при серии append (инвалидируется при смене .env / ошибке).
_worksheet_cache_fp: str | None = None
_worksheet_cache_ws: Any | None = None


def invalidate_worksheet_cache() -> None:
    """Сбросить кеш листа (смена таблицы/листа/ключа, ошибка API)."""
    global _worksheet_cache_fp, _worksheet_cache_ws
    _worksheet_cache_fp = None
    _worksheet_cache_ws = None


def _worksheet_fingerprint() -> str:
    path = _credentials_path()
    p = str(path.resolve()) if path is not None else ""
    sid = (os.environ.get("GOOGLE_SHEETS_SPREADSHEET_ID") or DEFAULT_SPREADSHEET_ID).strip()
    wname = (os.environ.get("GOOGLE_SHEETS_WORKSHEET") or "").strip()
    return f"{p}|{sid}|{wname}"

# Ключевые слова в «Минусах», которые означают критическое нарушение
_CRITICAL_NEGATIVE_PATTERNS = re.compile(
    r"запрещ[её]н|конфликт|хамств|груб|оскорб|невозможно|не могу помочь|"
    r"недовол|претензи|жалоб|кричал|агресси",
    re.IGNORECASE,
)


def _project_root() -> Path:
    return Path(__file__).resolve().parent


def _resolve_json_key_path(raw: str) -> Path | None:
    raw = raw.strip().strip('"').strip("'")
    if not raw:
        return None
    candidates: list[Path] = []
    p = Path(raw).expanduser()
    candidates.append(p)
    if not p.is_absolute():
        candidates.append(Path.cwd() / raw)
        candidates.append(_project_root() / raw)
    for c in candidates:
        try:
            if c.is_file():
                return c.resolve()
        except OSError:
            continue
    return None


CALL_FILENAME_REQUIREMENTS_RU = """\
**Шаблон имени файла:** `ДД-ММ-ГГГГ_телефон_ММ-СС.расширение`

**Пример:** `12-02-2026_7 (902) 556-73-27_01-12.mp3`

**Правила:**
- Ровно **два** символа подчёркивания `_`, разделяющие **три** части в имени без расширения.
- **Первая часть** — дата: `ДД-ММ-ГГГГ` (день-месяц-год, цифры и дефисы).
- **Вторая часть** — номер телефона или произвольная метка **без** символа `_`.
- **Третья часть** — ожидание: `ММ-СС` (минуты и секунды), секунды `00–59`, минуты `00–180`.
"""


def is_standard_call_filename(filename: str) -> bool:
    """True, если имя файла успешно разбирается как стандарт записи колл-центра."""
    meta = parse_call_filename_metadata(filename)
    return bool(meta.get("date") and meta.get("phone") and meta.get("wait_display"))


def parse_call_filename_metadata(filename: str) -> dict[str, str]:
    """
    Имя: ``ДД-ММ-ГГГГ_телефон_MM-SS.ext`` (последний блок — ожидание).

    Пример: ``12-02-2026_7 (902) 556-73-27_01-12.mp3``
    """
    stem = Path(filename).stem.strip()
    out: dict[str, str] = {"date": "", "phone": "", "wait_display": ""}
    if stem.count("_") < 2:
        return out
    date_part, phone_part, wait_token = stem.rsplit("_", 2)
    date_part = date_part.strip()
    wait_token = wait_token.strip()
    if not re.match(r"^\d{2}-\d{2}-\d{4}$", date_part):
        return out
    m = re.match(r"^(\d{2})-(\d{2})$", wait_token)
    if not m:
        return out
    mm, ss = int(m.group(1)), int(m.group(2))
    if ss >= 60 or mm < 0 or mm > 180:
        return out
    out["date"] = date_part
    out["phone"] = phone_part.strip()
    out["wait_display"] = f"{mm:02d}:{ss:02d}"
    return out


def needs_review(total_score: int, negatives: list[str]) -> str:
    """
    Возвращает «Да» если звонок требует ручного прослушивания:
      - итоговая оценка < 6, ИЛИ
      - в минусах есть критические нарушения (запрещённые фразы, конфликт и т.п.)
    Иначе — «Нет».
    """
    if total_score < 6:
        return "Да"
    combined = " ".join(negatives)
    if _CRITICAL_NEGATIVE_PATTERNS.search(combined):
        return "Да"
    return "Нет"


def _service_account_email(creds_path: Path) -> str | None:
    try:
        data = json.loads(creds_path.read_text(encoding="utf-8"))
        email = data.get("client_email")
        return str(email).strip() if email else None
    except (OSError, json.JSONDecodeError, TypeError):
        return None


def _credentials_path() -> Path | None:
    raw = (os.environ.get("GOOGLE_SHEETS_CREDENTIALS_JSON") or "").strip()
    if not raw:
        raw = (os.environ.get("GOOGLE_APPLICATION_CREDENTIALS") or "").strip()
    if raw:
        resolved = _resolve_json_key_path(raw)
        if resolved is not None:
            return resolved
    fallback = _project_root() / "keygoogle.json"
    if fallback.is_file():
        return fallback.resolve()
    return None


def _open_worksheet_uncached():
    import gspread
    from gspread.exceptions import APIError, SpreadsheetNotFound
    from google.oauth2.service_account import Credentials

    path = _credentials_path()
    if path is None:
        return None, "JSON ключ не найден (см. GOOGLE_SHEETS_CREDENTIALS_JSON или keygoogle.json в проекте)"

    sid = (os.environ.get("GOOGLE_SHEETS_SPREADSHEET_ID") or DEFAULT_SPREADSHEET_ID).strip()
    sa_email = _service_account_email(path)
    email_hint = f" Добавьте в доступ таблицы email из ключа: `{sa_email}` (роль «Редактор»)." if sa_email else ""

    scopes = ["https://www.googleapis.com/auth/spreadsheets"]
    creds = Credentials.from_service_account_file(str(path), scopes=scopes)
    gc = gspread.authorize(creds)
    try:
        sh = gc.open_by_key(sid)
    except PermissionError as exc:
        cause = getattr(exc, "__cause__", None)
        api_detail = f" {cause}" if cause is not None else ""
        return None, (
            f"Доступ запрещён (HTTP 403) к таблице `{sid}`.{email_hint} "
            "В Google Cloud Console должен быть включён API «Google Sheets»."
            f"{api_detail}"
        )
    except SpreadsheetNotFound as exc:
        return None, (
            f"Таблица не найдена или нет доступа (ID `{sid}`). Проверьте ID в ссылке docs.google.com/spreadsheets/d/… "
            f"{email_hint} Детали: {exc}"
        )
    except APIError as exc:
        return None, f"Ошибка Google API при открытии `{sid}`: {exc}.{email_hint}"
    except Exception as exc:
        cause = getattr(exc, "__cause__", None)
        tail = f" (причина: {cause})" if cause is not None else ""
        return None, f"Не удалось открыть таблицу по ID `{sid}`: {type(exc).__name__}: {exc!r}.{tail}"

    wname = (os.environ.get("GOOGLE_SHEETS_WORKSHEET") or "").strip()
    if wname:
        try:
            ws = sh.worksheet(wname)
        except gspread.WorksheetNotFound:
            titles = [w.title for w in sh.worksheets()]
            return None, f"Лист «{wname}» не найден. Задайте GOOGLE_SHEETS_WORKSHEET или оставьте пустым. Листы: {titles}"
    else:
        ws = sh.sheet1
    return ws, None


def _open_worksheet():
    """Открыть лист с кешированием при неизменных пути ключа, ID таблицы и имени листа."""
    global _worksheet_cache_fp, _worksheet_cache_ws
    fp = _worksheet_fingerprint()
    if _worksheet_cache_ws is not None and _worksheet_cache_fp == fp:
        return _worksheet_cache_ws, None
    ws, err = _open_worksheet_uncached()
    if err is None and ws is not None:
        _worksheet_cache_fp = fp
        _worksheet_cache_ws = ws
    else:
        invalidate_worksheet_cache()
    return ws, err


def _next_serial(ws, has_header: bool) -> int:
    all_values = ws.get_all_values()
    if not all_values:
        return 1
    if has_header and len(all_values) == 1:
        return 1
    if has_header:
        return len(all_values)
    return len(all_values) + 1


def _build_row(
    serial: int,
    original_filename: str,
    total_score: int,
    max_score: int = 10,
    operator_name: str = "",
    applicant_name: str = "",
    script_score: int = 0,
    speech_score: int = 0,
    consultation_score: int = 0,
    engagement_score: int = 0,
    tone_summary: str = "",
    script_details: list[str] | None = None,
    positives: list[str] | None = None,
    negatives: list[str] | None = None,
    filename_meta_override: dict[str, str] | None = None,
) -> list:
    """Строит список значений для одной строки таблицы (16 колонок)."""
    meta = (
        {
            "date": (filename_meta_override or {}).get("date", ""),
            "phone": (filename_meta_override or {}).get("phone", ""),
            "wait_display": (filename_meta_override or {}).get("wait_display", ""),
        }
        if filename_meta_override
        else parse_call_filename_metadata(original_filename)
    )

    score_cell: int | str = int(total_score) if max_score == 10 else f"{total_score}/{max_score}"
    op_name = (operator_name or "").strip() or "—"
    app_name = (applicant_name or "").strip() or "—"
    tone = (tone_summary or "").strip() or "—"

    checklist_text = "\n".join(script_details or []) or "—"
    positives_text = "\n".join(positives or []) or "—"
    negatives_text = "\n".join(negatives or []) or "—"
    review_flag = needs_review(total_score, negatives or [])

    return [
        serial,                        # A — №
        meta["date"],                  # B — Дата
        op_name,                       # C — Оператор
        meta["phone"],                 # D — Телефон
        app_name,                      # E — Заявитель
        meta["wait_display"],          # F — Ожидание
        score_cell,                    # G — Итог
        script_score,                  # H — Скрипт
        speech_score,                  # I — Речь
        consultation_score,            # J — Консультация
        engagement_score,              # K — Вовлечённость
        tone,                          # L — Тон
        checklist_text,                # M — Чек-лист
        positives_text,                # N — Плюсы
        negatives_text,                # O — Минусы
        review_flag,                   # P — Требует прослушивания
    ]


def append_analysis_row(
    *,
    original_filename: str,
    total_score: int,
    max_score: int = 10,
    operator_name: str = "",
    applicant_name: str = "",
    script_score: int = 0,
    speech_score: int = 0,
    consultation_score: int = 0,
    engagement_score: int = 0,
    tone_summary: str = "",
    script_details: list[str] | None = None,
    positives: list[str] | None = None,
    negatives: list[str] | None = None,
    filename_meta_override: dict[str, str] | None = None,
    # deprecated — kept for compatibility, no longer stored
    report_text: str = "",
) -> tuple[bool, str]:
    """
    Добавляет одну строку (16 колонок) в таблицу.
    Возвращает (успех, сообщение).
    """
    if os.environ.get("GOOGLE_SHEETS_DISABLE", "").strip().lower() in ("1", "true", "yes"):
        return False, "Экспорт отключён (GOOGLE_SHEETS_DISABLE)"

    path = _credentials_path()
    if path is None:
        return False, "Пропуск экспорта: не найден JSON ключ сервисного аккаунта"

    last_exc: Exception | None = None
    for attempt in range(2):
        try:
            ws, err = _open_worksheet()
            if ws is None:
                return False, err or "не удалось открыть таблицу"

            has_header = os.environ.get("GOOGLE_SHEETS_HAS_HEADER", str(_DEFAULT_HAS_HEADER)).lower() in (
                "1", "true", "yes",
            )
            serial = _next_serial(ws, has_header)
            row = _build_row(
                serial=serial,
                original_filename=original_filename,
                total_score=total_score,
                max_score=max_score,
                operator_name=operator_name,
                applicant_name=applicant_name,
                script_score=script_score,
                speech_score=speech_score,
                consultation_score=consultation_score,
                engagement_score=engagement_score,
                tone_summary=tone_summary,
                script_details=script_details,
                positives=positives,
                negatives=negatives,
                filename_meta_override=filename_meta_override,
            )
            ws.append_row(row, value_input_option="USER_ENTERED")
            url = (
                f"https://docs.google.com/spreadsheets/d/"
                f"{(os.environ.get('GOOGLE_SHEETS_SPREADSHEET_ID') or DEFAULT_SPREADSHEET_ID).strip()}/edit"
            )
            return True, f"Строка {serial} добавлена. Откройте таблицу: {url}"
        except Exception as exc:
            last_exc = exc
            invalidate_worksheet_cache()
            if attempt == 0:
                continue
            hint = ""
            err_l = str(exc).lower()
            if "403" in str(exc) or "permission" in err_l:
                hint = (
                    " Проверьте: 1) в Google Cloud включён API «Google Sheets»; "
                    "2) таблица расшарена на email сервисного аккаунта из JSON (`client_email`) с правом «Редактор»."
                )
            return False, f"Ошибка Google Sheets: {type(exc).__name__}: {exc}.{hint}"
    return False, f"Ошибка Google Sheets: {type(last_exc).__name__}: {last_exc!r}."


def append_batch_rows(
    rows_data: list[dict],
) -> tuple[bool, str]:
    """
    Пакетное добавление нескольких строк за один API-запрос.

    Каждый элемент rows_data — словарь с ключами:
      original_filename, total_score, max_score, operator_name, applicant_name,
      script_score, speech_score, consultation_score, engagement_score,
      tone_summary, script_details, positives, negatives

    Возвращает (успех, сообщение).
    """
    if not rows_data:
        return False, "Нет данных для экспорта."

    if os.environ.get("GOOGLE_SHEETS_DISABLE", "").strip().lower() in ("1", "true", "yes"):
        return False, "Экспорт отключён (GOOGLE_SHEETS_DISABLE)"

    path = _credentials_path()
    if path is None:
        return False, "Пропуск экспорта: не найден JSON ключ сервисного аккаунта"

    last_exc: Exception | None = None
    for attempt in range(2):
        try:
            ws, err = _open_worksheet()
            if ws is None:
                return False, err or "не удалось открыть таблицу"

            has_header = os.environ.get("GOOGLE_SHEETS_HAS_HEADER", str(_DEFAULT_HAS_HEADER)).lower() in (
                "1", "true", "yes",
            )
            start_serial = _next_serial(ws, has_header)

            built_rows = []
            for i, d in enumerate(rows_data):
                built_rows.append(_build_row(
                    serial=start_serial + i,
                    original_filename=d.get("original_filename", ""),
                    total_score=d.get("total_score", 0),
                    max_score=d.get("max_score", 10),
                    operator_name=d.get("operator_name", ""),
                    applicant_name=d.get("applicant_name", ""),
                    script_score=d.get("script_score", 0),
                    speech_score=d.get("speech_score", 0),
                    consultation_score=d.get("consultation_score", 0),
                    engagement_score=d.get("engagement_score", 0),
                    tone_summary=d.get("tone_summary", ""),
                    script_details=d.get("script_details"),
                    positives=d.get("positives"),
                    negatives=d.get("negatives"),
                ))

            ws.append_rows(built_rows, value_input_option="USER_ENTERED")
            url = (
                f"https://docs.google.com/spreadsheets/d/"
                f"{(os.environ.get('GOOGLE_SHEETS_SPREADSHEET_ID') or DEFAULT_SPREADSHEET_ID).strip()}/edit"
            )
            return True, (
                f"Добавлено {len(built_rows)} строк (№{start_serial}–{start_serial + len(built_rows) - 1}). "
                f"Откройте таблицу: {url}"
            )
        except Exception as exc:
            last_exc = exc
            invalidate_worksheet_cache()
            if attempt == 0:
                continue
            hint = ""
            err_l = str(exc).lower()
            if "403" in str(exc) or "permission" in err_l:
                hint = (
                    " Проверьте: 1) в Google Cloud включён API «Google Sheets»; "
                    "2) таблица расшарена на email сервисного аккаунта из JSON (`client_email`) с правом «Редактор»."
                )
            return False, f"Ошибка Google Sheets: {type(exc).__name__}: {exc}.{hint}"
    return False, f"Ошибка Google Sheets: {type(last_exc).__name__}: {last_exc!r}."


def is_sheets_configured() -> bool:
    return _credentials_path() is not None


def sheets_config_hint() -> str:
    cp = _credentials_path()
    if cp is not None:
        return f"Ключ сервисного аккаунта: `{cp}`"
    raw = (os.environ.get("GOOGLE_SHEETS_CREDENTIALS_JSON") or "").strip() or (
        os.environ.get("GOOGLE_APPLICATION_CREDENTIALS") or ""
    ).strip()
    if raw and _resolve_json_key_path(raw) is None:
        return (
            f"В .env указан путь `{raw}`, но файл не найден. cwd=`{Path.cwd()}`, проект=`{_project_root()}`."
        )
    return (
        "Нет JSON-ключа: добавьте `keygoogle.json` в папку проекта или строку в `.env`: "
        "`GOOGLE_SHEETS_CREDENTIALS_JSON=keygoogle.json`"
    )


# Заголовок таблицы (для удобства — можно вставить в первую строку вручную)
SHEETS_HEADER = [
    "№", "Дата", "Оператор", "Телефон", "Заявитель", "Ожидание",
    "Итог /10", "Скрипт /10", "Речь /10", "Консультация /10", "Вовлечённость /10",
    "Тон", "Чек-лист", "Плюсы", "Минусы", "Требует прослушивания",
]
