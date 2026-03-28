"""
Исправление ошибок диаризации: официальные формулы благодарности **за обращение** / за звонок
говорит только оператор ЕКЦ. Если фраза оказалась под меткой «Заявитель:» — переносим на «Оператор:».
"""

from __future__ import annotations

import re

# Согласовано с типичными формулировками чек-листа и llm_post_edit / transcription._THANK_PATTERNS.
_OPERATOR_ONLY_THANKS_RE = re.compile(
    r"спасибо\s+вам\s+за\s+обращени[ея]\b"
    r"|благодарим\s+вас\s+за\s+обращени[ея]\b"
    r"|спасибо\s+за\s+(?:ваше\s+|ваш\s+)?обращени[ея]\b"
    r"|благодар\w+\s+за\s+(?:ваш\s+)?(?:обращени[ея]|звонок)\b"
    r"|благодар\w+\s+(?:вас\s+|вам\s+)?за\s+звонок\b"
    r"|спасибо\s+за\s+(?:ваш\s+)?звонок\b"
    r"|рады\s+вашему\s+обращени[юя]\b",
    re.IGNORECASE,
)

_NEGATION_START = (
    "нет",
    "не ",
    "ни ",
    "без ",
    "никакого",
    "ничего",
    "отказ",
    "откажусь",
)


def _body_starts_with_negation(body: str) -> bool:
    bl = (body or "").strip().lower()
    return any(bl.startswith(p) for p in _NEGATION_START)


def _iter_role_blocks(role_text: str) -> list[tuple[str, str]]:
    if not (role_text or "").strip():
        return []
    blocks: list[tuple[str, str]] = []
    current_role: str | None = None
    chunks: list[str] = []
    header = re.compile(r"^(Оператор|Заявитель):\s*(.*)$")
    for line in role_text.splitlines():
        raw = line.strip()
        if not raw:
            continue
        m = header.match(raw)
        if m:
            if current_role is not None:
                blocks.append((current_role, " ".join(chunks).strip()))
            current_role = m.group(1)
            first = m.group(2).strip()
            chunks = [first] if first else []
        elif current_role is not None:
            chunks.append(raw)
    if current_role is not None:
        blocks.append((current_role, " ".join(chunks).strip()))
    return blocks


def applicant_block_contains_operator_thanks(body: str) -> bool:
    if _body_starts_with_negation(body):
        return False
    return bool(_OPERATOR_ONLY_THANKS_RE.search(body or ""))


def applicant_still_has_operator_thanks(role_text: str) -> bool:
    """True, если после правок под «Заявитель:» всё ещё есть официальная формула оператора."""
    for role, body in _iter_role_blocks(role_text or ""):
        if role == "Заявитель" and applicant_block_contains_operator_thanks(body):
            return True
    # Фолбэк: построчно (черновик без переносов внутри блока)
    for raw in (role_text or "").splitlines():
        line = raw.strip()
        low = line.lower()
        if not low.startswith("заявитель:"):
            continue
        body = line.split(":", 1)[1] if ":" in line else ""
        if applicant_block_contains_operator_thanks(body):
            return True
    return False


def _fix_line_by_line(role_text: str) -> tuple[str, bool]:
    lines = (role_text or "").splitlines()
    out: list[str] = []
    changed = False
    for raw in lines:
        line = raw.strip()
        low = line.lower()
        if low.startswith("заявитель:"):
            body = line.split(":", 1)[1] if ":" in line else ""
            if applicant_block_contains_operator_thanks(body):
                line = "Оператор:" + line.split(":", 1)[1]
                changed = True
        out.append(line)
    return "\n".join(out), changed


def fix_operator_thanks_mislabeled_as_applicant(role_text: str) -> tuple[str, bool]:
    """
    Возвращает (исправленный_текст, был_ли_перенос).
    Сохраняет исходную строку, если изменений нет.
    """
    rt = role_text or ""
    blocks = _iter_role_blocks(rt)
    if not blocks:
        if "Заявитель:" in rt or "заявитель:" in rt.lower():
            return _fix_line_by_line(rt)
        return rt, False

    changed = False
    new_blocks: list[tuple[str, str]] = []
    for role, body in blocks:
        if role == "Заявитель" and applicant_block_contains_operator_thanks(body):
            new_blocks.append(("Оператор", body))
            changed = True
        else:
            new_blocks.append((role, body))
    if not changed:
        return rt, False
    return "\n".join(f"{r}: {b}" for r, b in new_blocks), True
