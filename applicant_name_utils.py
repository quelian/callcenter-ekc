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
        "девушка",
        "женщина",
        "мужчина",
        "человек",
        "молодой",
        "гражданин",
        "гражданка",
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
        "приятно",
        "интересует",
        "интересно",
        "подходит",
        "устраивает",
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
    "ый",
    "ий",
    "ой",
    "ая",
    "ое",
    "ее",
    "ые",
    "ие",
    "ье",
    "ую",
    "юю",
    "ого",
    "ему",
    "ому",
    "ыми",
    "ими",
)

_FIRST_NAME_ALLOWED_ENDINGS = (
    "а",
    "я",
    "ия",
    "ья",
    "ей",
    "ий",
    "ай",
    "им",
    "ем",
    "ам",
    "ан",
    "ян",
    "ен",
    "он",
    "ор",
    "ир",
    "ил",
    "ис",
    "ас",
    "ав",
    "эл",
)

_SURNAME_SUFFIXES = (
    "ов",
    "ова",
    "ев",
    "ева",
    "ёв",
    "ёва",
    "ин",
    "ина",
    "ын",
    "ына",
    "ский",
    "ская",
    "цкий",
    "цкая",
    "енко",
    "ченко",
    "ко",
    "ук",
    "юк",
    "як",
    "ич",
    "вич",
    "ович",
    "евич",
    "ича",
    "ян",
    "янц",
    "дзе",
    "швили",
)

_PATRONYMIC_SUFFIXES = (
    "ович",
    "евич",
    "ич",
    "овна",
    "евна",
    "ична",
    "инична",
)

_COMMON_RUSSIAN_FIRST_NAMES = frozenset(
    {
        "авдей",
        "агафья",
        "агата",
        "агния",
        "адам",
        "аделина",
        "айдар",
        "аксинья",
        "алевтина",
        "александр",
        "александра",
        "алексей",
        "алина",
        "алиса",
        "алла",
        "альбина",
        "альфия",
        "амина",
        "анагель",
        "анастасия",
        "анатолий",
        "андрей",
        "анжелика",
        "анжела",
        "аниса",
        "анна",
        "антон",
        "антонина",
        "арам",
        "арина",
        "аркадий",
        "арсен",
        "артем",
        "артём",
        "артур",
        "архип",
        "асель",
        "аскар",
        "белла",
        "богдан",
        "борис",
        "вадим",
        "валентин",
        "валентина",
        "валерия",
        "варвара",
        "василий",
        "василиса",
        "вера",
        "вероника",
        "викентий",
        "виктор",
        "виктория",
        "виолетта",
        "виталий",
        "владимир",
        "владислав",
        "владлена",
        "галина",
        "гелена",
        "геннадий",
        "георгий",
        "герман",
        "глеб",
        "григорий",
        "даниил",
        "данил",
        "даниэль",
        "дарина",
        "дарья",
        "денис",
        "диана",
        "диляра",
        "дмитрий",
        "ева",
        "евгений",
        "евгения",
        "егор",
        "екатерина",
        "елена",
        "елизавета",
        "елисей",
        "елена",
        "жанна",
        "залина",
        "зара",
        "зинаида",
        "злата",
        "иван",
        "игнат",
        "игорь",
        "изабелла",
        "ильдар",
        "илья",
        "инесса",
        "инна",
        "ирина",
        "карина",
        "каролина",
        "кира",
        "кирилл",
        "клавдия",
        "константин",
        "кристина",
        "ксения",
        "лаврентий",
        "лада",
        "лариса",
        "лев",
        "леонид",
        "лиана",
        "лидия",
        "лилит",
        "лилия",
        "любовь",
        "людмила",
        "майя",
        "макар",
        "максим",
        "малик",
        "маргарита",
        "марина",
        "мария",
        "марк",
        "марта",
        "матвей",
        "мелания",
        "михаил",
        "мирон",
        "мирослава",
        "надежда",
        "назим",
        "назар",
        "наталия",
        "наталья",
        "нестор",
        "ника",
        "никита",
        "николай",
        "нил",
        "нинель",
        "нина",
        "оксана",
        "октябрина",
        "олеся",
        "ольга",
        "павел",
        "полина",
        "пётр",
        "петр",
        "рада",
        "радмир",
        "разиль",
        "раиса",
        "рамиль",
        "регина",
        "ренат",
        "римма",
        "родион",
        "роман",
        "ростислав",
        "руслан",
        "савелий",
        "светлана",
        "семен",
        "семён",
        "серафима",
        "сергей",
        "снежана",
        "софия",
        "софья",
        "станислав",
        "степан",
        "таисия",
        "тамара",
        "татьяна",
        "тимофей",
        "ульяна",
        "федор",
        "фёдор",
        "христина",
        "эвелина",
        "эдуард",
        "элина",
        "элла",
        "эльвира",
        "эмиль",
        "эмилия",
        "эмма",
        "юлия",
        "юрий",
        "яков",
        "яна",
        "ярослав",
    }
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


def _looks_like_patronymic(token: str) -> bool:
    low = token.lower()
    return len(low) >= 5 and low.endswith(_PATRONYMIC_SUFFIXES)


def _looks_like_surname(token: str) -> bool:
    low = token.lower()
    return len(low) >= 4 and low.endswith(_SURNAME_SUFFIXES)


def _looks_like_first_name(token: str, *, explicit_context: bool) -> bool:
    low = token.lower()
    if low in _COMMON_RUSSIAN_FIRST_NAMES:
        return True
    if not explicit_context:
        return False
    if len(low) < 3 or len(low) > 15:
        return False
    if low in _APPLICANT_NAME_BLOCKLIST:
        return False
    if low.endswith(_IMPLAUSIBLE_SINGLE_TOKEN_SUFFIXES):
        return False
    if _looks_like_surname(low) or _looks_like_patronymic(low):
        return False
    return low.endswith(_FIRST_NAME_ALLOWED_ENDINGS)


def _is_plausible_person_name_parts(parts: list[str], *, explicit_context: bool) -> bool:
    lowered = [part.lower() for part in parts]
    if len(lowered) == 1:
        return _looks_like_first_name(lowered[0], explicit_context=explicit_context)
    if len(lowered) == 2:
        first, second = lowered
        if not _looks_like_first_name(first, explicit_context=explicit_context):
            return False
        return (
            _looks_like_surname(second)
            or _looks_like_patronymic(second)
            or _looks_like_first_name(second, explicit_context=False)
        )
    if len(lowered) == 3:
        first, second, third = lowered
        if not _looks_like_first_name(first, explicit_context=explicit_context):
            return False
        second_ok = _looks_like_patronymic(second) or _looks_like_first_name(
            second,
            explicit_context=False,
        )
        third_ok = _looks_like_surname(third) or _looks_like_patronymic(third)
        return second_ok and third_ok
    return False


def normalize_applicant_name_candidate(
    raw: str,
    *,
    max_words: int = 3,
    explicit_context: bool = False,
) -> str | None:
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
    if not _is_plausible_person_name_parts(parts, explicit_context=explicit_context):
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
