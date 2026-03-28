from __future__ import annotations

import re

_NULL_MARKERS = frozenset(
    {
        "",
        "null",
        "none",
        "n/a",
        "unknown",
        "не определено",
        "не определен",
        "неизвестно",
        "нет",
        "-",
        "—",
    }
)

_APPLICANT_NAME_PREFIX_PATTERNS = (
    r"^(?:меня\s+зовут|зовут\s+меня)\s+",
    r"^(?:это|имя)\s+",
    r"^(?:заявителя\s+зовут|заявитель\s+)\s+",
    r"^(?:applicant_name\s*[:=])\s*",
)

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
        "доброе",
        "утро",
        "добрыйдень",
        "вечер",
        "алло",
        "колл",
        "центр",
        "двфу",
        "университет",
        "заявление",
        "понимаю",
        "подскажите",
        "спасибо",
        "пожалуйста",
        "вопрос",
        "проблема",
        "обращение",
        "документы",
        "справка",
        "заявка",
        "да",
        "нет",
        "ну",
        "вот",
        "это",
        "вам",
        "вас",
        "меня",
        "мне",
        "нас",
        "здесь",
        "там",
        "ага",
        "угу",
        "хорошо",
        "ладно",
        "понятно",
        "извините",
        "простите",
        "конечно",
        "такое",
        "такой",
        "такая",
        "такие",
        "такого",
        "такому",
        "такую",
        "этот",
        "эта",
        "эти",
        "этого",
        "этому",
        "эту",
        "понедельник",
        "вторник",
        "среда",
        "четверг",
        "пятница",
        "суббота",
        "воскресенье",
        "сегодня",
        "завтра",
        "вчера",
    }
)

_IMPLAUSIBLE_SINGLE_TOKEN_SUFFIXES = (
    "ое",
    "ее",
    "ие",
    "ье",
    "ого",
    "ему",
    "ому",
    "ыми",
    "ими",
)

APPLICANT_NAME_NOISE_WORDS = _APPLICANT_NAME_BLOCKLIST


def clean_name_token(raw: str) -> str:
    s = (raw or "").strip()
    if not s:
        return ""
    s = re.sub(r"^[\s\.…,:;!?«»\"'(\[\{-]+|[\s\.…,:;!?»\"')\]\}-]+$", "", s)
    return re.sub(r"\s+", " ", s).strip()


def title_cyrillic_name(token: str) -> str:
    t = clean_name_token(token)
    if not t:
        return ""
    if "-" in t:
        return "-".join(
            (part[0].upper() + part[1:].lower()) if len(part) > 1 else part.upper()
            for part in t.split("-")
            if part
        )
    return t[0].upper() + t[1:].lower() if len(t) > 1 else t.upper()


def normalize_applicant_name_candidate(raw: str, *, max_words: int = 3) -> str | None:
    """Возвращает только правдоподобное имя заявителя или ``None``."""
    s = clean_name_token(raw)
    if not s:
        return None

    low = s.lower()
    if low in _NULL_MARKERS or low in _APPLICANT_NAME_BLOCKLIST:
        return None

    for pattern in _APPLICANT_NAME_PREFIX_PATTERNS:
        low = re.sub(pattern, "", low, count=1, flags=re.IGNORECASE).strip()
    s = clean_name_token(low)
    if not s or s in _NULL_MARKERS or s in _APPLICANT_NAME_BLOCKLIST:
        return None

    parts = re.findall(r"[А-Яа-яЁё]+(?:-[А-Яа-яЁё]+)?", s)
    if not parts or len(parts) > max_words:
        return None

    for part in parts:
        token = part.lower()
        if token in _APPLICANT_NAME_BLOCKLIST:
            return None
        if len(part) < 2 or len(part) > 24:
            return None
        if len(parts) == 1 and token.endswith(_IMPLAUSIBLE_SINGLE_TOKEN_SUFFIXES):
            return None

    titled = [title_cyrillic_name(part) for part in parts]
    if not all(titled):
        return None
    return " ".join(titled)


def choose_authoritative_applicant_name(*candidates: str | None) -> str | None:
    for candidate in candidates:
        normalized = normalize_applicant_name_candidate(candidate or "")
        if normalized:
            return normalized
    return None
