from __future__ import annotations

import argparse
import atexit
import inspect
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout
from dataclasses import dataclass
from pathlib import Path

# Один поток: не копим фоновые punctuate после wall-timeout; следующая задача ждёт слот.
_POST_EDIT_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="punct_pe")


def _shutdown_post_edit_executor() -> None:
    try:
        _POST_EDIT_EXECUTOR.shutdown(wait=False)
    except Exception:
        pass


atexit.register(_shutdown_post_edit_executor)

from operator_staff import OPERATOR_ALIASES


@dataclass
class TranscriptionResult:
    text: str
    language: str | None = None
    role_text: str = ""
    tone_summary: str = "Не определен"
    role_attribution: str = ""
    role_confidence: str = "n/a"
    role_diagnostics: str = ""
    asr_profile: str = "medium_ru"
    asr_model: str = "unknown"


@dataclass
class QualityEvaluation:
    total_score: int
    max_score: int
    operator_name: str
    operator_in_staff: bool
    applicant_name: str | None
    script_score: int
    speech_score: int
    consultation_score: int
    engagement_score: int
    script_details: list[str]
    positives: list[str]
    negatives: list[str]


# Имя файла: `дата_телефон_MM-SS.ext` — последний блок после `_` это ожидание: минуты-секунды.
_FILENAME_WAIT_TAIL_RE = re.compile(r"^(\d{2})-(\d{2})$")


def parse_call_filename_wait_skip(filename: str) -> tuple[float, str] | None:
    """
    Извлекает длительность ожидания из конца имени файла.

    Пример: ``12-02-2026_7 (902) 556-73-27_01-12.mp3`` → хвост ``01-12`` = 1 мин 12 с → 72.0 с.

    Возвращает (секунды_пропуска, как_в_имени) или None, если шаблон не распознан.
    """
    stem = Path(filename).stem.strip()
    if "_" not in stem:
        return None
    tail = stem.rsplit("_", 1)[-1].strip()
    m = _FILENAME_WAIT_TAIL_RE.match(tail)
    if not m:
        return None
    minutes = int(m.group(1))
    seconds = int(m.group(2))
    if seconds >= 60 or minutes < 0 or minutes > 180:
        return None
    total = float(minutes * 60 + seconds)
    return total, tail


_punct_deepmulti_compat_applied = False


def _apply_deepmultilingual_transformers_compat() -> None:
    """
    deepmultilingualpunctuation вызывает pipeline(..., grouped_entities=False).
    В актуальных transformers вместо этого — aggregation_strategy=\"none\".
    """
    global _punct_deepmulti_compat_applied
    if _punct_deepmulti_compat_applied:
        return
    try:
        import deepmultilingualpunctuation.punctuationmodel as pmp
    except ImportError:
        return

    import torch
    from transformers import pipeline

    def _patched_punctuation_init(self, model: str = "oliverguhr/fullstop-punctuation-multilang-large") -> None:
        if torch.cuda.is_available():
            try:
                self.pipe = pipeline("ner", model, aggregation_strategy="none", device=0)
            except TypeError:
                self.pipe = pipeline("ner", model, grouped_entities=False, device=0)
        else:
            try:
                self.pipe = pipeline("ner", model, aggregation_strategy="none")
            except TypeError:
                self.pipe = pipeline("ner", model, grouped_entities=False)

    pmp.PunctuationModel.__init__ = _patched_punctuation_init  # type: ignore[method-assign]
    _punct_deepmulti_compat_applied = True


class Transcriber:
    _punct_model = None

    def __init__(
        self,
        model_name: str = "medium",
        compute_type: str = "int8",
        skip_first_seconds: float | None = None,
        language: str = "ru",
        model_load_timeout_seconds: int = 600,
        asr_profile: str = "medium_ru",
        heavy_diarization: bool = False,
        heavy_diarization_timeout_seconds: int = 240,
        enable_post_edit: bool = False,
        post_edit_timeout_seconds: int = 60,
        total_time_budget_seconds: int = 300,
        enable_llm_post_edit: bool = False,
        llm_backend: str = "off",
        llm_embedded_model_id: str | None = None,
        llm_post_edit_timeout_seconds: float = 120.0,
        llm_post_edit_base_url: str | None = None,
        llm_post_edit_model: str | None = None,
        llm_post_edit_api_key: str | None = None,
        llm_yandex_api_key: str | None = None,
        llm_yandex_folder_id: str | None = None,
        llm_yandex_model: str = "yandexgpt-lite",
    ) -> None:
        self.model_name = self._resolve_model_name(model_name)
        self.compute_type = compute_type
        # None = только пропуск из имени файла (_MM-SS ожидание) или с начала; иначе ручной clip
        self.skip_first_seconds = skip_first_seconds
        self.language = "ru"
        self.model_load_timeout_seconds = model_load_timeout_seconds
        self.asr_profile = asr_profile.strip().lower()
        self.heavy_diarization = heavy_diarization
        self.heavy_diarization_timeout_seconds = heavy_diarization_timeout_seconds
        self.enable_post_edit = enable_post_edit
        self.post_edit_timeout_seconds = post_edit_timeout_seconds
        self.total_time_budget_seconds = total_time_budget_seconds
        self.enable_llm_post_edit = enable_llm_post_edit
        self.llm_backend = (llm_backend or "off").strip().lower()
        self.llm_embedded_model_id = llm_embedded_model_id
        self.llm_post_edit_timeout_seconds = float(llm_post_edit_timeout_seconds)
        self.llm_post_edit_base_url = llm_post_edit_base_url
        self.llm_post_edit_model = llm_post_edit_model
        self.llm_post_edit_api_key = llm_post_edit_api_key
        self.llm_yandex_api_key = llm_yandex_api_key
        self.llm_yandex_folder_id = llm_yandex_folder_id
        self.llm_yandex_model = llm_yandex_model or "yandexgpt-lite"
        self._model = None
        self._asr_runtime_device = ""
        self._asr_runtime_compute = ""
        # Large-v3 долго качается и грузится; «Максимум» + тяжёлая диаризация — дольше общий бюджет
        if self.asr_profile == "ideal_ru":
            self.model_load_timeout_seconds = max(int(self.model_load_timeout_seconds), 1200)
            self.total_time_budget_seconds = max(int(self.total_time_budget_seconds), 1200)

    def _iter_whisper_load_attempts(self, effective_model: str) -> list[tuple[str, str]]:
        """
        Пары (device, compute_type) для faster-whisper/CTranslate2.
        float16 на CPU (macOS без CUDA) обычно не работает — не предлагаем его на cpu.
        """
        req = (self.compute_type or "default").strip().lower()
        cuda = False
        try:
            import ctranslate2 as ct2

            cuda = ct2.get_cuda_device_count() > 0
        except Exception:
            pass
        attempts: list[tuple[str, str]] = []

        def _add(dev: str, ctype: str) -> None:
            t = (dev, ctype)
            if t not in attempts:
                attempts.append(t)

        if cuda and req in {
            "float16",
            "float32",
            "int8",
            "int8_float16",
            "int8_float32",
            "bfloat16",
        }:
            _add("cuda", req)
        if cuda:
            for ctype in ("float16", "int8_float16", "float32", "int8"):
                _add("cuda", ctype)
        large = effective_model == "large-v3" or self.asr_profile == "ideal_ru"
        if large:
            # int8_float32 — лучший баланс качества/скорости на CPU для large-v3
            for ctype in ("int8_float32", "float32", "default", "int8"):
                _add("cpu", ctype)
        else:
            for ctype in ("int8", "int8_float32", "default"):
                _add("cpu", ctype)
        return attempts

    @staticmethod
    def _resolve_model_name(model_name: str) -> str:
        normalized = model_name.strip().lower()
        if normalized in {"best-ru", "best_ru", "best"}:
            return "large-v3"
        return model_name

    def _resolve_asr_model(self) -> str:
        if self.asr_profile == "ideal_ru":
            return "large-v3"
        if self.asr_profile == "medium_ru":
            return "medium"
        return "medium"

    def _asr_decode_preset(self) -> dict[str, object]:
        if self.asr_profile == "ideal_ru":
            # Выше medium: шире луч, выше patience; ниже порог «плохого» сегмента → чаще retry с temperature
            return {"beam_size": 16, "best_of": 14, "patience": 1.55}
        return {"beam_size": 8, "best_of": 6, "patience": 1.15}

    def _log(self, message: str) -> None:
        print(f"[transcriber] {message}", flush=True)

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", text) if p.strip()]
        return parts if parts else ([text.strip()] if text.strip() else [])

    @staticmethod
    def _split_sentences_ru(text: str) -> list[str]:
        """Дробление на предложения для пост-редактора (в т.ч. многоточие)."""
        cleaned = re.sub(r"\s+", " ", (text or "").strip())
        if not cleaned:
            return []
        parts = re.split(r"(?<=[\.\!\?…])\s+", cleaned)
        out = [p.strip() for p in parts if p.strip()]
        return out if out else [cleaned]

    @staticmethod
    def _merge_adjacent_role_pieces(
        pieces: list[tuple[float, float, str]],
        roles: list[str],
    ) -> tuple[list[tuple[float, float, str]], list[str]]:
        """Склеивает подряд идущие сегменты одной роли — лучший контекст для пунктуации и ролей."""
        if not pieces or not roles or len(pieces) != len(roles):
            return pieces, roles
        mp: list[tuple[float, float, str]] = []
        mr: list[str] = []
        cs, ce, ct = pieces[0][0], pieces[0][1], pieces[0][2].strip()
        cr = roles[0]
        for i in range(1, len(pieces)):
            s, e, raw_t = pieces[i]
            t = raw_t.strip()
            r = roles[i]
            if r == cr:
                ct = f"{ct} {t}".strip() if ct else t
                ce = e
            else:
                mp.append((cs, ce, ct))
                mr.append(cr)
                cs, ce, ct = s, e, t
                cr = r
        mp.append((cs, ce, ct))
        mr.append(cr)
        return mp, mr

    @staticmethod
    def _normalize_sentence_caps_ru(text: str) -> str:
        """Заглавная буква в начале и после . ! ? …"""
        t = (text or "").strip()
        if not t:
            return text
        out = t[0].upper() + t[1:] if len(t) > 1 else t.upper()
        out = re.sub(
            r"(?<=[\.\!\?…])(\s+)([а-яё])",
            lambda m: m.group(1) + m.group(2).upper(),
            out,
        )
        return out

    @staticmethod
    def _role_scores(sentence: str) -> tuple[int, int]:
        normalized = sentence.lower()
        operator_patterns = (
            r"\bздравствуйте\b",
            r"\bменя\s+зовут\b",
            r"\bчем\s+я\s+могу\s+помочь\b",
            r"\bчем\s+я\s+могу\s+быть\s+вам\s+полезен\b",
            r"\bмогу\s+подсказать\b",
            r"\bвам\s+необходимо\b",
            r"\bвам\s+нужно\b",
            r"\bобратитесь\b",
            r"\bзапишите\b",
            r"\bпродиктую\b",
            r"\bномер\s+телефона\b",
            r"\bконтакт",
            r"\bсотрудники\s+работают\b",
            r"\bдепартамент\b",
            r"\bграфик\s+работы\b",
            r"\bприемная\s+комиссия\b",
            r"\bуточню\b",
            r"\bпо\s+вашему\s+вопросу\b",
            r"\bя\s+вас\s+понял\b",
            r"\bкак\s+я\s+могу\s+к\s+вам\s+обращаться\b",
            r"\bкак\s+вас\s+зовут\b",
            r"\bспасибо\s+вам\s+за\s+обращение\b",
            r"\bдо\s+свидания\b",
            r"\bхорошего\s+дня\b",
            r"\bдобрый\s+день\b",
            r"\bдоброе\s+утро\b",
            r"\bдобрый\s+вечер\b",
            r"\bслушаю\s+вас\b",
            r"\bслушаю\b",
            r"\bодну\s+секунду\b",
            r"\bсейчас\s+уточню\b",
            r"\bя\s+уточн",
            r"\bпо\s+данному\s+вопросу\b",
            r"\bвы\s+можете\s+обратиться\b",
            r"\bвам\s+следует\b",
            r"\bнеобходимо\s+предоставить\b",
            r"\bнужно\s+предоставить\b",
            r"\bпринести\s+документы\b",
            r"\bподать\s+документы\b",
            r"\bзаполнить\s+заявление\b",
            r"\bзапишите,\s*пожалуйста\b",
            r"\bпродиктуйте\b",
            r"\bваши\s+данные\b",
            r"\bваш\s+вопрос\b",
            r"\bпо\s+расписанию\b",
            r"\bв\s+рабочее\s+время\b",
            r"\bхорошего\s+вам\s+дня\b",
            r"\bвсего\s+доброго\b",
            r"\bбудьте\s+здоровы\b",
        )
        applicant_patterns = (
            r"\bкуда\s+мне\s+обратиться\b",
            r"\bу\s+меня\b",
            r"\bмне\s+нужно\b",
            r"\bя\s+хочу\b",
            r"\bможно\s+ли\b",
            r"\bкак\s+оформить\b",
            r"\bчто\s+для\s+этого\b",
            r"\bкакие\s+документ",
            r"\bспасибо\b",
            r"\bподскажите,\s*пожалуйста\b",
            r"\bу\s+меня\s+дочь\b",
            r"\bкакие\s+льготы\b",
            r"\bя\s+звоню\b",
            r"\bхочу\s+узнать\b",
            r"\bинтересует\b",
            r"\bмог\s+бы\s+я\b",
            r"\bмогу\s+ли\s+я\b",
            r"\bскажите,\s*пожалуйста\b",
            r"\bа\s+можно\b",
        )
        short_backchannels = (
            "ага",
            "угу",
            "да",
            "нет",
            "понял",
            "поняла",
            "хорошо",
            "ясно",
            "спасибо",
            "понятно",
            "ладно",
            "окей",
            "ок",
            "конечно",
            "всё",
            "всё понял",
            "всё поняла",
        )

        operator_score = sum(2 for p in operator_patterns if re.search(p, normalized))
        applicant_score = sum(2 for p in applicant_patterns if re.search(p, normalized))

        # ------------------------------------------------------------------
        # Pronoun-based scoring — very strong signal in formal Russian dialogue.
        # Operators address the applicant with formal «вы»: вам/вас/ваш/ваши.
        # Applicants talk about themselves: мне/меня/мой/моя/мои/у меня.
        # «Я» is weaker: operators also say «я уточню», «я могу подсказать».
        # ------------------------------------------------------------------
        op_pronouns = len(re.findall(r"\b(вам|вас|ваш(?:и|а|е|его|ей)?)\b", normalized))
        app_pronouns = len(re.findall(
            r"\b(мне|меня|мой|моя|мои|моего|моей)\b", normalized
        ))
        first_person_ya = len(re.findall(r"(?<![а-яё])я(?![а-яё])", normalized))

        operator_score += min(3, op_pronouns)
        applicant_score += min(3, app_pronouns)
        # «я» contributes 1 per every 2 occurrences, starting from first
        applicant_score += min(2, first_person_ya // 2 + (1 if first_person_ya >= 1 else 0))

        # ------------------------------------------------------------------
        # High-confidence operator phrases (+2 extra bonus on top of pattern score).
        # These phrases are virtually never said by the applicant.
        # ------------------------------------------------------------------
        if re.search(
            r"\bменя\s+зовут\b"
            r"|\bслушаю\s+вас\b"
            r"|\bспасибо\s+вам\s+за\s+обращение\b"
            r"|\bчем\s+(?:я\s+могу|могу)\s+(?:вам\s+)?помочь\b"
            r"|\bвам\s+необходимо\b"
            r"|\bвы\s+можете\s+обратиться\b",
            normalized,
        ):
            operator_score += 2

        # ------------------------------------------------------------------
        # Imperative verbs directed at the applicant — always operator instructions.
        # ------------------------------------------------------------------
        if re.search(
            r"\b(позвоните|обратитесь|принесите|предоставьте|заполните"
            r"|уточните|назовите|продиктуйте|приходите|свяжитесь"
            r"|отправьте|напишите|сообщите|подтвердите)\b",
            normalized,
        ):
            operator_score += 3

        # ------------------------------------------------------------------
        # General imperative / directive bonus (existing, keep it).
        # ------------------------------------------------------------------
        if re.search(r"\b(запишите|позвоните|принесите|оформите)\b", normalized):
            operator_score += 2

        # ------------------------------------------------------------------
        # Question-mark scoring — context-dependent.
        # Operators also ask questions ("Как вас зовут?"), so "?" is not a
        # universal applicant signal. Only give full bonus when no operator
        # patterns matched, otherwise a weak +1.
        # ------------------------------------------------------------------
        if "?" in sentence:
            if operator_score == 0:
                applicant_score += 2   # strong: no operator evidence → applicant question
            else:
                applicant_score += 1   # weak: may be operator's clarifying question

        # Interrogative word at the START of a phrase → weak applicant signal.
        # Exclude operator name-questions like "Как вас зовут?".
        if re.match(r"^\s*(как|где|когда|зачем|почему|куда|откуда|сколько)\b", normalized):
            if not re.search(r"\bкак\s+(?:вас|я\s+могу)\b", normalized):
                applicant_score += 1

        if re.search(r"\b(мне|у\s+меня|подскажите\s+пожалуйста)\b", normalized):
            applicant_score += 1
        if normalized.strip() in short_backchannels:
            applicant_score += 1

        return operator_score, applicant_score

    @staticmethod
    def _is_operator_anchor(sentence: str) -> bool:
        normalized = sentence.lower()
        anchor_patterns = (
            r"\bздравствуйте\b.*\bменя\s+зовут\b",
            r"\bчем\s+я\s+могу\s+быть\s+вам\s+полезен\b",
            r"\bчем\s+я\s+могу\s+вам\s+помочь\b",
            r"\bкак\s+я\s+могу\s+к\s+вам\s+обращаться\b",
            r"\bподскажите,\s*пожалуйста,\s*как\s+вас\s+зовут\b",
            r"\bподскажите,\s*пожалуйста,\s*как\s+я\s+могу\s+к\s+вам\s+обращаться\b",
            r"\bспасибо\s+вам\s+за\s+обращение\b",
            r"\bдвфу\b.*\bздравствуйте\b",
            r"\bздравствуйте\b.*\bдвфу\b",
            r"\bдобрый\s+день\b.*\bменя\s+зовут\b",
            r"\bалл[оо]?\b.*\bменя\s+зовут\b",
            r"\bменя\s+зовут\b",
            r"\bслушаю\s+вас\b",
            r"\bчем\s+могу\s+помочь\b",
        )
        return any(re.search(pattern, normalized) for pattern in anchor_patterns)

    @staticmethod
    def _is_operator_name_question(sentence: str) -> bool:
        normalized = sentence.lower()
        patterns = (
            r"\bкак\s+я\s+могу\s+к\s+вам\s+обращаться\b",
            r"\bкак\s+вас\s+зовут\b",
            r"\bподскажите,\s*пожалуйста,\s*как\s+вас\s+зовут\b",
            r"\bподскажите,\s*пожалуйста,\s*как\s+я\s+могу\s+к\s+вам\s+обращаться\b",
            r"\bкак\s+вас\s+могу\s+называть\b",
            r"\bваше\s+имя\b",
            r"\bпредставьтесь,?\s*пожалуйста\b",
        )
        return any(re.search(pattern, normalized) for pattern in patterns)

    @staticmethod
    def _is_operator_opening(sentence: str) -> bool:
        normalized = sentence.lower()
        patterns = (
            r"\bздравствуйте\b.*\bменя\s+зовут\b",
            r"\bчем\s+я\s+могу\s+быть\s+вам\s+полезен\b",
            r"\bчем\s+я\s+могу\s+вам\s+помочь\b",
            r"\bдвфу\b.*\bздравствуйте\b",
            r"\bздравствуйте\b.*\bдвфу\b",
            r"\bдобрый\s+день\b.*\bменя\s+зовут\b",
            r"\bалл[оо]?\b.*\bменя\s+зовут\b",
            r"\bменя\s+зовут\b",
            r"\bслушаю\s+вас\b",
            r"\bчем\s+могу\s+помочь\b",
        )
        return any(re.search(pattern, normalized) for pattern in patterns)

    @staticmethod
    def _detect_role(
        sentence: str,
        previous_role: str | None,
        next_sentence: str | None = None,
    ) -> str:
        operator_score, applicant_score = Transcriber._role_scores(sentence)

        if operator_score > applicant_score:
            return "Оператор"
        if applicant_score > operator_score:
            return "Заявитель"

        # Tie-breakers.
        normalized = sentence.lower()
        if "?" in sentence:
            return "Заявитель"
        if previous_role is None:
            return "Оператор"

        # Prefer natural alternation for short ambiguous replies.
        if len(normalized.split()) <= 3:
            return "Заявитель" if previous_role == "Оператор" else "Оператор"

        # If next phrase looks like a question, current often belongs to operator.
        if next_sentence and "?" in next_sentence:
            return "Оператор"
        return previous_role

    @staticmethod
    def _smooth_roles(chunks: list[str], roles: list[str]) -> list[str]:
        if len(roles) < 3:
            return roles
        result = roles[:]
        # Remove isolated role flips: A B A -> A A A
        for i in range(1, len(result) - 1):
            if result[i - 1] == result[i + 1] and result[i] != result[i - 1]:
                if len(chunks[i].split()) <= 8:
                    result[i] = result[i - 1]

        # Question-answer pairing: after a question, a short neutral answer likely other speaker.
        for i in range(0, len(result) - 1):
            current = chunks[i]
            nxt = chunks[i + 1]
            if "?" in current and len(nxt.split()) <= 8 and "?" not in nxt:
                if result[i] == result[i + 1]:
                    result[i + 1] = "Заявитель" if result[i] == "Оператор" else "Оператор"
        return result

    @staticmethod
    def _apply_operator_anchors(chunks: list[str], roles: list[str]) -> list[str]:
        """Force operator role for known call-center script anchors.

        After a confirmed anchor, the operator typically continues for 1-2 more
        short segments (finishing the greeting/closing script) before handing
        over to the applicant. Propagate forward to 2 segments instead of 1.
        """
        result = roles[:]
        for i, chunk in enumerate(chunks):
            if not Transcriber._is_operator_anchor(chunk):
                continue
            result[i] = "Оператор"
            # Short preceding fragment usually belongs to the same operator turn.
            if i - 1 >= 0 and len(chunks[i - 1].split()) <= 6:
                result[i - 1] = "Оператор"
            # First following segment: propagate unconditionally if short.
            if i + 1 < len(chunks) and len(chunks[i + 1].split()) <= 12:
                result[i + 1] = "Оператор"
            # Second following segment: propagate only when text scoring agrees.
            if i + 2 < len(chunks) and len(chunks[i + 2].split()) <= 10:
                op_s, app_s = Transcriber._role_scores(chunks[i + 2])
                if op_s >= app_s:
                    result[i + 2] = "Оператор"
        return result

    @staticmethod
    def _apply_dialog_flow_rules(chunks: list[str], roles: list[str]) -> list[str]:
        """Apply strict conversation-flow rules tailored for call-center dialogues."""
        if not roles:
            return roles
        result = roles[:]
        n = len(chunks)

        # First strong opening should belong to operator.
        for i in range(min(4, n)):
            if Transcriber._is_operator_opening(chunks[i]):
                result[i] = "Оператор"
                if i + 1 < n and not Transcriber._is_operator_anchor(chunks[i + 1]):
                    result[i + 1] = "Заявитель"
                break

        for i in range(n):
            current = chunks[i]
            # Name question from operator -> next meaningful response should be applicant.
            if Transcriber._is_operator_name_question(current):
                result[i] = "Оператор"
                if i + 1 < n and not Transcriber._is_operator_anchor(chunks[i + 1]):
                    result[i + 1] = "Заявитель"

            # Applicant question should generally be followed by operator answer.
            if result[i] == "Заявитель" and "?" in current:
                if i + 1 < n and not Transcriber._is_operator_anchor(chunks[i + 1]):
                    result[i + 1] = "Оператор"

            # Operator clarifying question usually expects applicant short response.
            if result[i] == "Оператор" and "?" in current:
                if i + 1 < n and len(chunks[i + 1].split()) <= 12:
                    if not Transcriber._is_operator_anchor(chunks[i + 1]):
                        result[i + 1] = "Заявитель"

            # Closing phrase belongs to operator; prior short confirmation likely applicant.
            if re.search(r"\bспасибо\s+вам\s+за\s+обращение\b", current.lower()):
                result[i] = "Оператор"
                if i - 1 >= 0 and len(chunks[i - 1].split()) <= 8:
                    result[i - 1] = "Заявитель"

        return result

    def _operator_speaker_likelihood_score(self, texts: list[str]) -> float:
        joined = " ".join(texts).strip()
        if not joined:
            return -100.0
        bonus = 0.0
        if (
            self._is_operator_opening(joined)
            or self._is_operator_anchor(joined)
            or self._is_operator_name_question(joined)
        ):
            bonus = 12.0
        operator_part, applicant_part = self._role_scores(joined)
        return float(operator_part - applicant_part + bonus)

    def _build_role_transcript_from_voice_clusters(
        self,
        pieces: list[tuple[float, float, str, str]],
        role_map: dict[str, str],
    ) -> str:
        lines: list[str] = []
        current_role: str | None = None
        current_parts: list[str] = []
        prev_heuristic: str | None = None
        for _abs_s, _abs_e, text, cluster_id in pieces:
            role = role_map.get(cluster_id)
            if role is None:
                role = self._detect_role(text, prev_heuristic, next_sentence=None)
            prev_heuristic = role
            if current_role is None:
                current_role = role
                current_parts = [text]
            elif role == current_role:
                current_parts.append(text)
            else:
                lines.append(f"{current_role}: {' '.join(current_parts)}")
                current_role = role
                current_parts = [text]
        if current_parts and current_role:
            lines.append(f"{current_role}: {' '.join(current_parts)}")
        return "\n".join(lines)

    def _refine_voice_role_sequence(
        self, pieces: list[tuple[float, float, str, str]], role_map: dict[str, str]
    ) -> list[str]:
        text_chunks = [p[2] for p in pieces]
        raw_roles = [role_map.get(p[3], "Заявитель") for p in pieces]
        refined = self._smooth_roles(text_chunks, raw_roles)
        refined = self._apply_dialog_flow_rules(text_chunks, refined)
        refined = self._apply_operator_anchors(text_chunks, refined)
        refined = self._smooth_roles(text_chunks, refined)
        return refined

    def _apply_operator_prologue_prior(
        self, pieces: list[tuple[float, float, str, str]], roles: list[str], first_n: int = 12
    ) -> list[str]:
        result = roles[:]
        n = min(first_n, len(result))
        t0 = float(pieces[0][0]) if pieces else 0.0
        for i in range(n):
            s, _e, txt, _cid = pieces[i]
            # Первые ~55 с именно диалога (после clip по ожиданию из имени файла).
            if (s - t0) <= 55.0 and (
                self._is_operator_opening(txt)
                or self._is_operator_anchor(txt)
                or self._is_operator_name_question(txt)
            ):
                result[i] = "Оператор"
                if i + 1 < len(result) and len(pieces[i + 1][2].split()) <= 12:
                    result[i + 1] = "Заявитель"
        return result

    @staticmethod
    def _roles_to_transcript(chunks: list[str], roles: list[str]) -> str:
        if not chunks or not roles or len(chunks) != len(roles):
            return ""
        lines: list[str] = []
        cur_role = roles[0]
        cur_parts = [chunks[0]]
        for i in range(1, len(chunks)):
            if roles[i] == cur_role:
                cur_parts.append(chunks[i])
            else:
                lines.append(f"{cur_role}: {' '.join(cur_parts)}")
                cur_role = roles[i]
                cur_parts = [chunks[i]]
        lines.append(f"{cur_role}: {' '.join(cur_parts)}")
        return "\n".join(lines)

    @staticmethod
    def _overlap_seconds(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
        return max(0.0, min(a_end, b_end) - max(a_start, b_start))

    def _assign_cluster_by_overlap(
        self,
        segment_start: float,
        segment_end: float,
        diarization_turns: list[tuple[float, float, str]],
    ) -> str:
        if not diarization_turns:
            return "SPK_0"
        best_overlap = 0.0
        best_label = diarization_turns[0][2]
        center = 0.5 * (segment_start + segment_end)
        best_distance = float("inf")
        for s, e, label in diarization_turns:
            ov = self._overlap_seconds(segment_start, segment_end, s, e)
            if ov > best_overlap:
                best_overlap = ov
                best_label = label
            distance = abs(center - (0.5 * (s + e)))
            if distance < best_distance:
                best_distance = distance
                nearest_label = label
        if best_overlap <= 1e-6:
            return nearest_label
        return best_label

    @staticmethod
    def _postprocess_ru_text(text: str) -> str:
        cleaned = re.sub(r"\s+", " ", text).strip()
        cleaned = re.sub(r"\s+([,.;!?])", r"\1", cleaned)
        # Нормализация частых доменных написаний.
        replacements = {
            "двф у": "ДВФУ",
            "д в ф у": "ДВФУ",
            "дальневосточного федерального университета": "Дальневосточного федерального университета",
            "кол центр": "колл-центр",
            "колл центр": "колл-центр",
        }
        lowered = cleaned.lower()
        for src, dst in replacements.items():
            lowered = lowered.replace(src, dst.lower())
        # Восстанавливаем базовый регистр для начала предложений.
        lowered = lowered[:1].upper() + lowered[1:] if lowered else lowered
        lowered = lowered.replace("двфу", "ДВФУ")
        return lowered

    @classmethod
    def _get_local_punctuation_model(cls):
        if cls._punct_model is not None:
            return cls._punct_model
        _apply_deepmultilingual_transformers_compat()
        from deepmultilingualpunctuation import PunctuationModel

        cls._punct_model = PunctuationModel(model="kredor/punctuate-all")
        return cls._punct_model

    def _resolve_llm_post_edit_config(self):
        """OpenAI-compatible API (Ollama ``/v1``, LM Studio, …). Только для ``llm_backend=remote``."""
        if not self.enable_llm_post_edit or self.llm_backend != "remote":
            return None
        from llm_post_edit import LLMPostEditConfig

        base = (
            (self.llm_post_edit_base_url or "").strip()
            or (os.environ.get("LLM_POST_EDIT_BASE_URL") or "").strip()
        ).rstrip("/")
        if not base:
            return None
        model = (self.llm_post_edit_model or "").strip() or (
            os.environ.get("LLM_POST_EDIT_MODEL") or ""
        ).strip()
        if not model:
            return None
        api_key = self.llm_post_edit_api_key
        if api_key is None:
            raw = (os.environ.get("LLM_POST_EDIT_API_KEY") or "").strip()
            api_key = raw or None
        to = max(15.0, min(float(self.llm_post_edit_timeout_seconds), 600.0))
        return LLMPostEditConfig(base_url=base, model=model, api_key=api_key, timeout_seconds=to)

    @staticmethod
    def _token_set(text: str) -> set[str]:
        return set(re.findall(r"[а-яёa-z0-9]+", (text or "").lower()))

    def _punctuate_long_text(self, model: object, chunk: str, max_batch_chars: int = 520) -> str:
        """
        Пунктуация с батчингом по предложениям: длинные сегменты не ломаем одним вызовом модели.
        """
        chunk_pp = self._postprocess_ru_text(chunk)
        if not chunk_pp:
            return chunk_pp
        if len(chunk_pp) <= max_batch_chars:
            out = getattr(model, "restore_punctuation")(chunk_pp)
            return str(out).strip()
        sents = self._split_sentences_ru(chunk_pp)
        if len(sents) <= 1:
            out = getattr(model, "restore_punctuation")(chunk_pp)
            return str(out).strip()
        batches: list[str] = []
        buf: list[str] = []
        nchars = 0
        rp = getattr(model, "restore_punctuation")
        for s in sents:
            add = len(s) + (1 if buf else 0)
            if buf and nchars + add > max_batch_chars:
                batches.append(rp(" ".join(buf)).strip())
                buf = [s]
                nchars = len(s)
            else:
                buf.append(s)
                nchars += add
        if buf:
            batches.append(rp(" ".join(buf)).strip())
        return self._postprocess_ru_text(" ".join(batches))

    def _role_logic_refiner(
        self,
        pieces: list[tuple[float, float, str]],
        roles: list[str],
        confidence: str,
    ) -> tuple[list[str], bool]:
        if not pieces or not roles or len(pieces) != len(roles):
            return roles, False
        # При высокой уверенности диаризации голос уже хорош — эвристики могут только портить.
        if confidence not in {"low", "medium"}:
            return roles, False
        text_chunks = [p[2] for p in pieces]
        refined = self._apply_dialog_flow_rules(text_chunks, roles[:])
        refined = self._apply_operator_anchors(text_chunks, refined)
        refined = self._smooth_roles(text_chunks, refined)
        piece4 = [(s, e, txt, "SPK_0") for s, e, txt in pieces]
        refined = self._apply_operator_prologue_prior(piece4, refined)
        changed = refined != roles
        return refined, changed

    def _post_edit_ru_dialogue(
        self,
        raw_text: str,
        pieces: list[tuple[float, float, str]],
        roles: list[str],
        role_confidence: str,
        elapsed_before_post_edit: float,
    ) -> tuple[str, str, list[str], str, list[tuple[float, float, str]] | None]:
        if not self.enable_post_edit:
            return raw_text, "off", roles, "", None
        if self.asr_profile not in {"ideal_ru", "medium_ru"}:
            return raw_text, "off_profile", roles, "", None
        if not pieces:
            return raw_text, "off_no_pieces", roles, "", None

        # degrade depth near time budget
        budget_left = max(0.0, self.total_time_budget_seconds - elapsed_before_post_edit)
        timeout = int(max(10, min(self.post_edit_timeout_seconds, budget_left)))
        if timeout <= 10:
            return raw_text, "fallback_budget", roles, "timeout_budget", None

        box: dict[str, object] = {}
        err: dict[str, Exception] = {}

        def worker() -> None:
            try:
                model = self._get_local_punctuation_model()
                local_chunks = [p[2] for p in pieces]
                punctuated: list[str] = []
                punct_limit = len(local_chunks)
                if budget_left < 32:
                    punct_limit = max(6, len(local_chunks) // 2)
                rp = getattr(model, "restore_punctuation")
                for i, ch in enumerate(local_chunks):
                    ch = (ch or "").strip()
                    if not ch:
                        punctuated.append(ch)
                        continue
                    if i >= punct_limit:
                        punctuated.append(ch)
                        continue
                    if len(ch) > 800:
                        punctuated.append(self._punctuate_long_text(model, ch, max_batch_chars=500))
                    else:
                        punctuated.append(str(rp(ch)).strip())
                merged = self._postprocess_ru_text(" ".join(punctuated))
                base = self._normalize_sentence_caps_ru(self._postprocess_ru_text(raw_text))
                edited = self._normalize_sentence_caps_ru(merged)
                tok_src = self._token_set(base)
                tok_new = self._token_set(edited)
                overlap = (len(tok_src & tok_new) / max(1, len(tok_src))) if tok_src else 1.0
                ratio = (len(edited.split()) / max(1, len(base.split()))) if base else 1.0
                if overlap < 0.52 or ratio < 0.72 or ratio > 1.38:
                    box["text"] = base
                    box["reason"] = "anti_drift_revert"
                    texts_out = list(local_chunks)
                else:
                    box["text"] = edited
                    box["reason"] = "ok"
                    texts_out = punctuated
                work_pieces = [(pieces[i][0], pieces[i][1], texts_out[i]) for i in range(len(pieces))]
                new_roles, changed = self._role_logic_refiner(
                    pieces=work_pieces,
                    roles=roles,
                    confidence=role_confidence,
                )
                box["roles"] = new_roles
                box["roles_changed"] = changed
                box["out_pieces"] = [(s, e, t) for s, e, t in work_pieces]
            except Exception as exc:
                err["error"] = exc

        fut = _POST_EDIT_EXECUTOR.submit(worker)
        try:
            fut.result(timeout=float(timeout))
        except FuturesTimeout:
            return raw_text, "fallback_timeout", roles, "post_edit_timeout", None
        if "error" in err:
            return raw_text, "fallback_error", roles, str(err["error"]), None

        final_text = str(box.get("text", raw_text))
        final_roles = list(box.get("roles", roles))
        out_list = box.get("out_pieces")
        updated: list[tuple[float, float, str]] | None
        if isinstance(out_list, list):
            updated = out_list
        else:
            updated = None
        status = "on"
        if box.get("reason") == "anti_drift_revert":
            status = "fallback_anti_drift"
        return final_text, status, final_roles, str(box.get("reason", "")), updated

    def _pick_operator_cluster(self, clustered: list[tuple[float, float, str, str]]) -> str | None:
        by_cluster_text: dict[str, list[str]] = {}
        by_cluster_seconds: dict[str, float] = {}
        for s, e, txt, cid in clustered:
            by_cluster_text.setdefault(cid, []).append(txt)
            by_cluster_seconds[cid] = by_cluster_seconds.get(cid, 0.0) + max(0.0, e - s)
        if len(by_cluster_text) < 2:
            return None

        # Правило «первый говорящий = оператор»:
        # В колл-центре после clip_timestamps первым всегда говорит оператор.
        # Проверяем, что текст первых сегментов этого кластера не противоречит роли оператора
        # (т.е. оператор-score >= заявитель-score), и если так — возвращаем сразу.
        first_cid = clustered[0][3]
        first_cluster_texts = by_cluster_text.get(first_cid, [])
        first_op_s, first_app_s = 0, 0
        for t in first_cluster_texts[:4]:
            o, a = self._role_scores(t)
            first_op_s += o
            first_app_s += a
        if first_op_s >= first_app_s:
            return first_cid

        # Приоритет операторского приветствия в начале звонка.
        # Use relative time from the first segment so the bonus works even when
        # clip_timestamps shifts all absolute timestamps forward (e.g. to 72s+).
        _seg_t0 = clustered[0][0] if clustered else 0.0
        early = [x for x in clustered if x[0] <= _seg_t0 + 35.0]
        early_bonus: dict[str, float] = {}
        for _s, _e, txt, cid in early:
            if self._is_operator_anchor(txt) or self._is_operator_opening(txt):
                early_bonus[cid] = early_bonus.get(cid, 0.0) + 8.0
            if self._is_operator_name_question(txt):
                early_bonus[cid] = early_bonus.get(cid, 0.0) + 4.0

        scored: list[tuple[float, str]] = []
        for cid, texts in by_cluster_text.items():
            score = self._operator_speaker_likelihood_score(texts)
            score += min(4.0, by_cluster_seconds.get(cid, 0.0) / 60.0)  # tiny duration prior
            score += early_bonus.get(cid, 0.0)
            scored.append((score, cid))
        scored.sort(reverse=True)
        return scored[0][1] if scored else None

    def _role_confidence_label(
        self,
        cluster_scores: dict[str, float],
        diagnostics: dict[str, float | int | str],
    ) -> str:
        vals = sorted(cluster_scores.values(), reverse=True)
        margin = (vals[0] - vals[1]) if len(vals) >= 2 else 0.0
        quality = float(diagnostics.get("cluster_quality", 0.0) or 0.0)
        if margin >= 4.0 and quality >= 2.0:
            return "high"
        if margin >= 1.5 and quality >= 1.2:
            return "medium"
        return "low"

    @staticmethod
    def _decode_roles_globally(chunks: list[str]) -> list[str]:
        """Global 2-state Viterbi decoding with linguistically-grounded transitions."""
        if not chunks:
            return []
        states = ("Оператор", "Заявитель")
        score_by_chunk = [Transcriber._role_scores(chunk) for chunk in chunks]

        # Precompute anchor flags so we can use them in transition logic.
        _anchor_prev = [Transcriber._is_operator_anchor(c) for c in chunks]

        dp: list[dict[str, float]] = []
        prev_state: list[dict[str, str | None]] = []

        # Seed: in a call centre the first speaker is always the operator.
        _seed_op_bias = 1.0
        first_op, first_app = score_by_chunk[0]
        dp.append({
            "Оператор": float(first_op) + _seed_op_bias,
            "Заявитель": float(first_app),
        })
        prev_state.append({"Оператор": None, "Заявитель": None})

        for i in range(1, len(chunks)):
            op_score, app_score = score_by_chunk[i]
            cur_scores = {"Оператор": float(op_score), "Заявитель": float(app_score)}
            chunk = chunks[i]
            chunk_prev = chunks[i - 1]
            chunk_words = len(chunk.split())
            cur_prev: dict[str, str | None] = {}
            cur_dp: dict[str, float] = {}
            for state in states:
                best_val = -10**9
                best_prev: str | None = None
                for ps in states:
                    transition = 0.0

                    # Base turn-taking preference (dialogue alternates).
                    if ps != state:
                        transition += 0.5  # was 0.4

                    # Long explanatory chunk → same speaker likely continues.
                    # Reduced threshold from >20 to >15 words, bonus from 0.2 to 0.4.
                    if ps == state and chunk_words > 15:
                        transition += 0.4

                    # After a question in previous chunk → very strong switch pressure.
                    if "?" in chunk_prev and ps != state:
                        transition += 1.0  # was 0.6

                    # Very short chunk (1-3 words) = backchannel («да», «угу», «понятно»).
                    # Backchannels do NOT represent a speaker turn change — reward STAYING.
                    # Old code had +0.3 for SWITCHING on short chunks — that was wrong.
                    if chunk_words <= 3:
                        if ps == state:
                            transition += 0.4  # backchannel stays with same speaker
                        # (no switching bonus for short chunks — removed)

                    # After a confirmed operator anchor the applicant is expected next.
                    # This prevents runs of operator-labeled segments after e.g. "меня зовут".
                    if _anchor_prev[i - 1] and state == "Заявитель" and ps == "Оператор":
                        transition += 0.8  # push toward applicant after anchor

                    candidate = dp[i - 1][ps] + transition + cur_scores[state]
                    if candidate > best_val:
                        best_val = candidate
                        best_prev = ps
                cur_dp[state] = best_val
                cur_prev[state] = best_prev
            dp.append(cur_dp)
            prev_state.append(cur_prev)

        last_state = max(states, key=lambda s: dp[-1][s])
        decoded = [last_state]
        for i in range(len(chunks) - 1, 0, -1):
            last_state = prev_state[i][last_state] or "Оператор"
            decoded.append(last_state)
        decoded.reverse()
        return decoded

    @staticmethod
    def _split_segment_at_silences(
        segment: object,
        min_silence_gap: float = 0.35,
    ) -> list[tuple[float, float, str]]:
        """Split a faster-whisper segment at internal silence gaps using word timestamps.

        Returns list of (start, end, text) sub-segments. Falls back to the
        original single segment when word timestamps are unavailable or empty.
        A gap of ≥ min_silence_gap seconds between consecutive words almost always
        marks a speaker turn boundary in call-center dialogue.
        """
        words = getattr(segment, "words", None)
        seg_text = (getattr(segment, "text", "") or "").strip()
        seg_start = float(getattr(segment, "start", 0.0))
        seg_end = float(getattr(segment, "end", 0.0))

        if not words:
            return [(seg_start, seg_end, seg_text)] if seg_text else []

        # Group consecutive words; start a new group when silence >= min_silence_gap.
        groups: list[list] = [[words[0]]]
        for word in words[1:]:
            gap = float(getattr(word, "start", 0.0)) - float(getattr(groups[-1][-1], "end", 0.0))
            if gap >= min_silence_gap:
                groups.append([word])
            else:
                groups[-1].append(word)

        result: list[tuple[float, float, str]] = []
        for group in groups:
            text = " ".join((getattr(w, "word", "") or "").strip() for w in group).strip()
            if text:
                result.append((
                    float(getattr(group[0], "start", seg_start)),
                    float(getattr(group[-1], "end", seg_end)),
                    text,
                ))
        return result if result else [(seg_start, seg_end, seg_text)]

    def _build_role_transcript(self, text: str, chunks: list[str] | None = None) -> str:
        source_chunks = chunks if chunks is not None else self._split_sentences(text)
        prepared = [chunk.strip() for chunk in source_chunks if chunk.strip()]
        if not prepared:
            return ""

        roles = self._decode_roles_globally(prepared)
        roles = self._apply_operator_anchors(prepared, roles)
        roles = self._apply_dialog_flow_rules(prepared, roles)
        roles = self._smooth_roles(prepared, roles)
        roles = self._apply_dialog_flow_rules(prepared, roles)
        roles = self._apply_operator_anchors(prepared, roles)

        # Merge neighboring chunks with same role for cleaner dialogue.
        merged_lines: list[str] = []
        current_role = roles[0]
        current_parts = [prepared[0]]
        for i in range(1, len(prepared)):
            if roles[i] == current_role:
                current_parts.append(prepared[i])
            else:
                merged_lines.append(f"{current_role}: {' '.join(current_parts)}")
                current_role = roles[i]
                current_parts = [prepared[i]]
        merged_lines.append(f"{current_role}: {' '.join(current_parts)}")
        return "\n".join(merged_lines)

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        effective_model = self._resolve_asr_model()
        self._log(
            f"Загружаю ASR модель '{effective_model}' (profile={self.asr_profile}, "
            f"запрошенный compute_type={self.compute_type})..."
        )
        self._log(
            "На первом запуске large-v3 может скачиваться 10–30+ минут. "
            "Если сеть медленная, подождите или проверьте доступ к Hugging Face."
        )
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError(
                "Не найден faster-whisper. Установите зависимости: pip install -r requirements.txt"
            ) from exc

        attempts = self._iter_whisper_load_attempts(effective_model)
        self._log(
            "Порядок загрузки CTranslate2: "
            + ", ".join(f"{d}/{c}" for d, c in attempts[:8])
            + ("…" if len(attempts) > 8 else "")
        )

        last_error: Exception | None = None
        for device, ctype in attempts:
            model_box: dict[str, object] = {}
            error_box: dict[str, Exception] = {}

            def _load_model(dev: str = device, ct: str = ctype) -> None:
                try:
                    model_box["model"] = WhisperModel(
                        effective_model,
                        device=dev,
                        compute_type=ct,
                    )
                except Exception as exc:  # pragma: no cover - safety net
                    error_box["error"] = exc

            thread = threading.Thread(target=_load_model, daemon=True)
            thread.start()
            thread.join(timeout=self.model_load_timeout_seconds)
            if thread.is_alive():
                self._log(
                    "[asr] Достигнут лимит ожидания загрузки; жду завершения потока "
                    "без запуска второй параллельной загрузки…"
                )
                thread.join()
            if "model" in model_box:
                self._model = model_box["model"]
                self._log(f"Модель загружена: device={device}, compute_type={ctype}")
                self._asr_runtime_device = device
                self._asr_runtime_compute = ctype
                break
            if "error" in error_box:
                err = error_box["error"]
                last_error = err
                self._log(f"[asr] Не удалось {device}/{ctype}: {err}")
                continue
            last_error = RuntimeError(
                f"Загрузка {device}/{ctype} завершилась без модели и без ошибки в логе."
            )
            self._log(f"[asr] {last_error}")
            continue
        else:
            hint = (
                "Превышен таймаут или все варианты compute_type отвергнуты. "
                "Проверьте интернет (скачивание large-v3), место на диске и версию ctranslate2. "
                "Временно выберите «Стандарт» (medium / int8)."
            )
            if last_error is not None:
                raise RuntimeError(f"{hint} Последняя ошибка: {last_error}") from last_error
            raise RuntimeError(hint)

    def _joint_voice_role_decode(
        self,
        audio_path: str,
        voice_items: list[tuple[float, float, str]],
        asr_elapsed: float,
    ) -> tuple[str, str, str, str, list[tuple[float, float, str]], list[str]]:
        role_text = self._build_role_transcript(
            " ".join(x[2] for x in voice_items),
            chunks=[x[2] for x in voice_items],
        )
        base_pieces = [(s, e, t) for s, e, t in voice_items]
        base_roles = self._decode_roles_globally([x[2] for x in base_pieces]) if base_pieces else []
        role_attribution = "Текстовый fallback"
        role_confidence = "low"
        role_diagnostics = f"role_mode=fallback; asr_seconds={asr_elapsed:.2f}"
        diarization_started = time.perf_counter()
        used_heavy = False
        try:
            from speaker_voice_roles import cluster_voice_segments

            clustered: list[tuple[float, float, str, str]] = []
            cluster_reason = "fallback text"
            cluster_diag: dict[str, object] = {}
            if self.heavy_diarization:
                self._log("Joint mode: heavy diarization по голосу...")
                try:
                    from speaker_diarization_heavy import run_heavy_diarization

                    turns, heavy_reason, heavy_diag = run_heavy_diarization(
                        audio_path=audio_path,
                        target_speakers=2,
                        max_seconds=self.heavy_diarization_timeout_seconds,
                    )
                    if turns:
                        clustered = []
                        for s, e, txt in voice_items:
                            cid = self._assign_cluster_by_overlap(s, e, turns)
                            clustered.append((s, e, txt, cid))
                        cluster_reason = heavy_reason
                        cluster_diag = heavy_diag
                        used_heavy = True
                except Exception as exc:
                    self._log(f"Heavy diarization ошибка ({exc}), перехожу на light.")
            if not clustered:
                clustered, cluster_reason, cluster_diag = cluster_voice_segments(
                    audio_path=audio_path,
                    segments=voice_items,
                    target_speakers=2,
                )
            if clustered:
                by_cluster: dict[str, list[str]] = {}
                for _s, _e, txt, cid in clustered:
                    by_cluster.setdefault(cid, []).append(txt)
                operator_cluster = self._pick_operator_cluster(clustered) or ""
                role_map = {
                    cid: ("Оператор" if cid == operator_cluster else "Заявитель")
                    for cid in by_cluster.keys()
                }
                refined_roles = self._refine_voice_role_sequence(clustered, role_map)
                refined_roles = self._apply_operator_prologue_prior(clustered, refined_roles)
                cluster_scores = {
                    cid: self._operator_speaker_likelihood_score(texts)
                    for cid, texts in by_cluster.items()
                }
                role_confidence = self._role_confidence_label(cluster_scores, cluster_diag)
                # Cluster consistency: when voice quality is medium or high, enforce
                # that each acoustic cluster carries one dominant role (majority vote).
                # This prevents text heuristics from "breaking up" a correct cluster by
                # flipping a few individual segment labels against the cluster's evidence.
                if role_confidence in {"high", "medium"} and clustered:
                    _cluster_votes: dict[str, dict[str, int]] = {}
                    for (_s, _e, _txt, _cid), _role in zip(clustered, refined_roles):
                        _cluster_votes.setdefault(_cid, {"Оператор": 0, "Заявитель": 0})
                        _cluster_votes[_cid][_role] += 1
                    _cluster_dominant = {
                        _cid: max(_votes, key=lambda k: _votes[k])
                        for _cid, _votes in _cluster_votes.items()
                    }
                    refined_roles = [
                        "Оператор"  # explicit operator anchor always wins
                        if self._is_operator_anchor(_txt) or self._is_operator_opening(_txt)
                        else _cluster_dominant.get(_cid, _role)
                        for (_s, _e, _txt, _cid), _role in zip(clustered, refined_roles)
                    ]
                if role_confidence == "low":
                    # Joint fallback: text-подсказка при низкой уверенности голоса.
                    # Text wins only when it is DECISIVE (|op_score - app_score| >= 2).
                    # When both scores are 0-0 or tied, voice prediction is kept to
                    # avoid replacing a correct voice label with an arbitrary text guess.
                    text_chunks = [x[2] for x in clustered]
                    text_roles = self._decode_roles_globally(text_chunks)
                    _text_scores = [self._role_scores(c) for c in text_chunks]
                    refined_roles = [
                        "Оператор"
                        if self._is_operator_anchor(ch) or self._is_operator_opening(ch)
                        else (
                            tr                           # text wins: it is decisive
                            if abs(ts[0] - ts[1]) >= 2 and tr != vr
                            else vr                      # voice wins: text is ambiguous
                        )
                        for vr, tr, ch, ts in zip(
                            refined_roles, text_roles, text_chunks, _text_scores
                        )
                    ]
                    refined_roles = self._smooth_roles(text_chunks, refined_roles)
                role_text = self._roles_to_transcript([x[2] for x in clustered], refined_roles)
                base_pieces = [(x[0], x[1], x[2]) for x in clustered]
                base_roles = refined_roles[:]
                role_attribution = f"Joint voice-role decode ({'heavy' if used_heavy else 'light'})"
                diarization_elapsed = time.perf_counter() - diarization_started
                role_diagnostics = (
                    f"role_mode=joint; diarization_path={'heavy_diarization' if used_heavy else 'fallback_light'}; "
                    f"split_mode={cluster_diag.get('split_mode', 'native')}; reason={cluster_reason}; "
                    f"quality={cluster_diag.get('cluster_quality', 0.0):.3f}; "
                    f"asr_seconds={asr_elapsed:.2f}; diarization_seconds={diarization_elapsed:.2f}"
                )
        except Exception as exc:
            role_diagnostics = f"role_mode=fallback; exception={exc}; asr_seconds={asr_elapsed:.2f}"
        return role_text, role_attribution, role_confidence, role_diagnostics, base_pieces, base_roles

    def transcribe(
        self,
        audio_path: str | Path,
        *,
        original_basename: str | None = None,
    ) -> TranscriptionResult:
        self._ensure_model()
        path = str(audio_path)
        self._log(f"Начинаю обработку файла: {path}")
        info_language: str | None = None
        transcribe_kwargs = {
            "vad_filter": True,
            # Tuned VAD: lower onset threshold + shorter min-speech so the first
            # utterance is captured even when speech starts immediately after music.
            "vad_parameters": {
                "threshold": 0.35,           # onset (default 0.5) — catch quieter starts
                "min_speech_duration_ms": 150,  # shorter min (default 250) — catch brief phrases
                "speech_pad_ms": 400,        # keep default padding around each speech chunk
            },
            "beam_size": 7,
            "best_of": 5,
            "temperature": 0.0,
            "language": "ru",
            "task": "transcribe",
            "condition_on_previous_text": False,
            "initial_prompt": "ДВФУ, деканат, зачётная книжка, расписание, приёмная комиссия, общежитие",
        }
        transcribe_kwargs.update(self._asr_decode_preset())
        if self.asr_profile == "ideal_ru":
            # Меньше повторов/галлюцинаций, жёстче отсев «плохих» окон → чаще повтор с другой temperature
            transcribe_kwargs.update(
                {
                    "compression_ratio_threshold": 2.15,
                    "log_prob_threshold": -0.85,
                    "no_speech_threshold": 0.58,
                    "repetition_penalty": 1.07,
                    "no_repeat_ngram_size": 3,
                }
            )
        # Enable word-level timestamps for sub-segment splitting at silence gaps.
        # Checked at runtime so the code works with older faster-whisper versions too.
        _wt_sig = inspect.signature(self._model.transcribe)
        _word_ts_enabled = "word_timestamps" in _wt_sig.parameters
        if _word_ts_enabled:
            transcribe_kwargs["word_timestamps"] = True
        self._log(
            "ASR пресет: "
            f"profile={self.asr_profile}, beam={transcribe_kwargs['beam_size']}, "
            f"best_of={transcribe_kwargs['best_of']}, patience={transcribe_kwargs.get('patience', 'n/a')}, "
            f"word_timestamps={'on' if _word_ts_enabled else 'off (not supported)'}"
        )
        clip_applied = False
        skip_sec = 0.0
        file_skip_extra = ""
        if self.skip_first_seconds is not None:
            skip_sec = max(0.0, float(self.skip_first_seconds))
            self._log(f"Ручной пропуск начала (--skip-first-seconds): {skip_sec:.1f} с")
            file_skip_extra = f"file_skip_source=manual; file_skip_seconds={skip_sec:.2f}"
        else:
            # Streamlit сохраняет загрузку во временный файл (tmpXXX.mp3) — имя для парсинга передавайте отдельно.
            name_for_skip = (original_basename or Path(path).name).strip()
            name_for_skip = Path(name_for_skip).name
            parsed = parse_call_filename_wait_skip(name_for_skip)
            if parsed is not None:
                skip_sec, wait_token = parsed
                mm = int(skip_sec // 60)
                ss = int(skip_sec % 60)
                self._log(
                    f"Пропуск ожидания по имени файла: «{wait_token}» → {skip_sec:.0f} с "
                    f"({mm:02d}:{ss:02d} от начала записи)"
                )
                file_skip_extra = (
                    f"file_skip_source=filename; file_wait_token={wait_token}; "
                    f"file_skip_seconds={skip_sec:.2f}"
                )
            else:
                file_skip_extra = "file_skip_source=none"

        if skip_sec > 0:
            signature = inspect.signature(self._model.transcribe)
            if "clip_timestamps" in signature.parameters:
                transcribe_kwargs["clip_timestamps"] = f"{skip_sec}"
                clip_applied = True
                self._log(f"Clip timestamps: {skip_sec:.1f}s")
            else:
                self._log("Внимание: версия faster-whisper без clip_timestamps, начало не пропускаю.")
        # faster-whisper reports segment timestamps absolute from the file start
        # even when clip_timestamps is set, so no additional offset is needed.
        time_offset = 0.0
        asr_started = time.perf_counter()
        self._log("Распознаю речь...")
        segments, info = self._model.transcribe(path, **transcribe_kwargs)
        asr_elapsed = time.perf_counter() - asr_started

        # Hallucination filter: discard segments whose words overlap >50% with initial_prompt.
        # Whisper sometimes repeats the prompt verbatim when audio is silent or clipped.
        _raw_prompt = str(transcribe_kwargs.get("initial_prompt") or "")
        _stop_words = {"и", "в", "по", "на", "с", "к", "у", "из", "за", "не", "это", "что", "как"}
        _prompt_words = set(re.sub(r"[^\w\s]", "", _raw_prompt.lower()).split()) - _stop_words

        # Phrases spoken by PBX auto-informer at the very end of a call.
        # These are never part of the human conversation and must be removed.
        _IVR_TAIL_RE = re.compile(
            r"оцените\s+работу\s+нашего\s+сотрудника"
            r"|по\s+шкале\s+от\s+1\s+до\s+5"
            r"|после\s+сигнала"
            r"|приняли\s+участие\s+в\s+(?:нашем\s+)?вопросе",
            re.IGNORECASE,
        )

        parts: list[str] = []
        chunks: list[str] = []
        whisper_pieces_rel: list[tuple[float, float, str]] = []
        segment_count = 0
        hallucination_count = 0
        ivr_tail_hit = False  # once IVR phrase seen, discard everything after
        for segment in segments:
            segment_count += 1
            sub_segs = self._split_segment_at_silences(segment, min_silence_gap=0.35)
            for seg_start, seg_end, piece in sub_segs:
                if not piece:
                    continue
                # Once PBX auto-informer phrase is detected, stop collecting.
                if _IVR_TAIL_RE.search(piece):
                    ivr_tail_hit = True
                    self._log(f"[asr] Отброшен хвост автоинформатора: «{piece[:80]}»")
                if ivr_tail_hit:
                    continue
                if len(_prompt_words) >= 3:
                    seg_words = set(re.sub(r"[^\w\s]", "", piece.lower()).split())
                    overlap = len(seg_words & _prompt_words) / len(seg_words) if seg_words else 0.0
                    if overlap > 0.85 and len(seg_words) >= 5:
                        hallucination_count += 1
                        self._log(f"[asr] Отброшен сегмент-галлюцинация ({overlap:.0%}): «{piece[:80]}»")
                        continue
                parts.append(piece)
                chunks.extend(self._split_sentences(piece))
                whisper_pieces_rel.append((seg_start, seg_end, piece))
            if segment_count % 10 == 0:
                self._log(
                    f"Обработано сегментов: {segment_count}, текущий таймкод: {segment.end:.1f}с"
                )
        self._log(
            f"Распознавание завершено. Сегментов: {segment_count}"
            + (", хвост автоинформатора обрезан" if ivr_tail_hit else "")
            + (f", отброшено галлюцинаций: {hallucination_count}" if hallucination_count else "")
        )
        text = " ".join(parts).strip()
        info_language = getattr(info, "language", None)
        whisper_pieces_rel = [(s + time_offset, e + time_offset, t) for s, e, t in whisper_pieces_rel]

        text = self._postprocess_ru_text(text)
        total_started = time.perf_counter()
        (
            role_text,
            role_attribution,
            role_confidence,
            role_diagnostics,
            role_pieces,
            role_labels,
        ) = self._joint_voice_role_decode(
            audio_path=path,
            voice_items=whisper_pieces_rel,
            asr_elapsed=asr_elapsed,
        )
        text_work = text
        role_text_work = role_text
        llm_elapsed = 0.0
        llm_diag = "disabled"
        nf: str | None = None
        nr: str | None = None
        llm_note = ""
        llm_cfg = self._resolve_llm_post_edit_config()
        if self.enable_llm_post_edit and self.llm_backend == "yandex" and self.llm_yandex_api_key and self.llm_yandex_folder_id:
            from llm_cloud_eval import YandexCloudConfig, run_yandex_post_edit_threaded

            ycfg = YandexCloudConfig(
                api_key=self.llm_yandex_api_key,
                folder_id=self.llm_yandex_folder_id,
                model=self.llm_yandex_model,
                timeout_seconds=self.llm_post_edit_timeout_seconds,
            )
            self._log(
                f"LLM пост-редактор (Yandex AI Studio, gpt://{self.llm_yandex_folder_id}/{self.llm_yandex_model}), "
                f"таймаут≤{self.llm_post_edit_timeout_seconds:.0f}s..."
            )
            t_llm = time.perf_counter()
            nf, nr, llm_note = run_yandex_post_edit_threaded(
                text_work,
                role_text_work,
                cfg=ycfg,
                wall_timeout=float(self.llm_post_edit_timeout_seconds),
            )
            llm_elapsed = time.perf_counter() - t_llm
            if nf and nr:
                text_work = self._postprocess_ru_text(nf)
                role_text_work = nr.strip()
                llm_diag = f"ok:{llm_note}"
                self._log(f"LLM пост-редактор готов за {llm_elapsed:.2f}s ({llm_diag})")
            else:
                llm_diag = f"fallback:{llm_note}"
                self._log(f"LLM пост-редактор не применён ({llm_diag}), {llm_elapsed:.2f}s")
        elif self.enable_llm_post_edit and self.llm_backend == "embedded":
            from llm_embedded_deepseek import run_embedded_llm_post_edit_threaded

            self._log(
                f"LLM пост-редактор (встроенная DeepSeek / Hugging Face), "
                f"таймаут≤{self.llm_post_edit_timeout_seconds:.0f}s..."
            )
            t_llm = time.perf_counter()
            nf, nr, llm_note = run_embedded_llm_post_edit_threaded(
                text_work,
                role_text_work,
                model_id=(self.llm_embedded_model_id or "").strip() or None,
                wall_timeout=float(self.llm_post_edit_timeout_seconds),
                log=None,
            )
            llm_elapsed = time.perf_counter() - t_llm
            if nf and nr:
                text_work = self._postprocess_ru_text(nf)
                role_text_work = nr.strip()
                llm_diag = f"ok:{llm_note}"
                self._log(f"LLM пост-редактор готов за {llm_elapsed:.2f}s ({llm_diag})")
            else:
                llm_diag = f"fallback:{llm_note}"
                self._log(f"LLM пост-редактор не применён ({llm_diag}), {llm_elapsed:.2f}s")
        elif llm_cfg is not None:
            from llm_post_edit import run_llm_post_edit_threaded

            self._log(
                f"LLM пост-редактор (HTTP): {llm_cfg.base_url}, модель={llm_cfg.model}, "
                f"таймаут≤{self.llm_post_edit_timeout_seconds:.0f}s..."
            )
            t_llm = time.perf_counter()
            nf, nr, llm_note = run_llm_post_edit_threaded(
                text_work,
                role_text_work,
                llm_cfg,
                wall_timeout=float(self.llm_post_edit_timeout_seconds),
            )
            llm_elapsed = time.perf_counter() - t_llm
            if nf and nr:
                text_work = self._postprocess_ru_text(nf)
                role_text_work = nr.strip()
                llm_diag = f"ok:{llm_note}"
                self._log(f"LLM пост-редактор готов за {llm_elapsed:.2f}s ({llm_diag})")
            else:
                llm_diag = f"fallback:{llm_note}"
                self._log(f"LLM пост-редактор не применён ({llm_diag}), {llm_elapsed:.2f}s")
        elif self.enable_llm_post_edit and self.llm_backend == "remote":
            llm_diag = "remote_missing_url_or_model"
            self._log(
                "LLM remote: задайте LLM_POST_EDIT_BASE_URL и LLM_POST_EDIT_MODEL в .env "
                "или передайте URL/модель в Transcriber."
            )

        skip_local_post = nf is not None and nr is not None

        post_started = time.perf_counter()
        post_elapsed = 0.0
        post_status = "off"
        post_reason = ""
        role_refine_applied = "no"
        refined_roles = role_labels
        post_pieces: list[tuple[float, float, str]] | None = None

        if skip_local_post:
            post_status = "skipped_after_llm"
            post_reason = llm_diag
        elif self.enable_post_edit:
            self._log("Запускаю локальный пост-редактор текста/ролей...")
            post_text, post_status, refined_roles, post_reason, post_pieces = self._post_edit_ru_dialogue(
                raw_text=text_work,
                pieces=role_pieces,
                roles=role_labels,
                role_confidence=role_confidence,
                elapsed_before_post_edit=(asr_elapsed + (time.perf_counter() - total_started)),
            )
            text_work = post_text
            post_elapsed = time.perf_counter() - post_started
            self._log(
                f"Локальный пост-редактор: mode={post_status}, reason={post_reason or 'ok'}, "
                f"time={post_elapsed:.2f}s"
            )
        else:
            post_reason = "disabled"

        pieces_for_roles = post_pieces if post_pieces is not None else role_pieces
        if not skip_local_post and refined_roles and pieces_for_roles and len(refined_roles) == len(pieces_for_roles):
            new_role_text = self._roles_to_transcript([p[2] for p in pieces_for_roles], refined_roles)
            if new_role_text:
                role_refine_applied = "yes" if new_role_text != role_text_work else "no"
                role_text_work = new_role_text

        text = text_work
        role_text = role_text_work

        role_diagnostics = (
            f"{role_diagnostics}; llm_post_edit={llm_diag}; llm_seconds={llm_elapsed:.2f}; "
            f"post_edit={post_status}; role_refine_applied={role_refine_applied}; "
            f"fallback_reason={post_reason or 'n/a'}; post_edit_seconds={post_elapsed:.2f}"
        )
        if file_skip_extra:
            role_diagnostics = f"{role_diagnostics}; {file_skip_extra}"

        self._log("Текст собран.")
        self._log("Сформирована расшифровка по ролям.")
        tone_summary = analyze_tone(audio_path)
        self._log(f"Оценка тона: {tone_summary}")
        return TranscriptionResult(
            text=text,
            language=info_language,
            role_text=role_text,
            tone_summary=tone_summary,
            role_attribution=role_attribution,
            role_confidence=role_confidence,
            role_diagnostics=role_diagnostics,
            asr_profile=self.asr_profile,
            asr_model=(
                f"{self._resolve_asr_model()}"
                f" ({self._asr_runtime_device}/{self._asr_runtime_compute})"
                if self._asr_runtime_device
                else self._resolve_asr_model()
            ),
        )


class CallQualityEvaluator:
    """Экспертная эвристическая оценка качества консультации."""

    _OPERATOR_ALIASES = OPERATOR_ALIASES

    _GREETING_PATTERNS = (r"\bздравств", r"\bдобрый\s+(день|вечер|утро)")
    _INTRO_PATTERNS = (
        r"\bменя\s+зовут\b",
        r"\bоператор\b",
        r"\bспециалист\b",
        r"\bколл[- ]?центр\b",
        r"\bдвфу\b",
    )
    _ASK_APPLICANT_NAME_PATTERNS = (
        r"\bкак\s+вас\s+зовут\b",
        r"\bкак\s+к\s+вам\s+обращаться\b",
        r"\bпредставьтесь\b",
        r"\bподскажите\s+ваше\s+имя\b",
    )
    _GOODBYE_PATTERNS = (
        r"\bдо\s+свидания\b",
        r"\bхорошего\s+дня\b",
        r"\bвсего\s+доброго\b",
        r"\bобращайтесь\b",
        r"\bрад.?\s+был.?\s+помочь\b",
        r"\bбыл.?\s+рад.?\s+помочь\b",
        r"\bдо\s+встречи\b",
        r"\bхорошего\s+вечера\b",
        r"\bприятного\s+дня\b",
    )
    _OFFER_HELP_PATTERNS = (
        r"\bчем\s+ещё\s+могу\s+помочь\b",
        r"\bчем\s+еще\s+могу\s+помочь\b",
        r"\bостались\s+ли\s+(?:у\s+вас\s+)?вопросы\b",
        r"\bесть\s+ли\s+(?:ещё|еще)\s+вопросы\b",
        r"\bмогу\s+(?:ли\s+)?(?:ещё|еще|чем-то)\s+быть\s+полезн",
        r"\bещё\s+(?:чем-то\s+)?могу\s+помочь\b",
        r"\bеще\s+(?:чем-то\s+)?могу\s+помочь\b",
        r"\bмогу\s+помочь\s+чем.нибудь",
    )
    _THANK_PATTERNS = (
        r"\bспасибо\s+за\s+(?:ваш\s+)?обращение\b",
        r"\bблагодар\w+\s+за\s+(?:ваш\s+)?(?:обращение|звонок)\b",
        r"\bблагодар\w+\s+(?:вас\s+)?за\s+звонок\b",
        r"\bспасибо\s+за\s+(?:ваш\s+)?звонок\b",
        r"\bрады\s+вашему\s+обращению\b",
    )
    _FILLER_PATTERNS = (
        r"\bээ+\b",
        r"\bэм+\b",
        r"\bну\b",
        r"\bкак\s+бы\b",
        r"\bтипа\b",
        r"\bв\s+общем\b",
        r"\bкороче\b",
        r"\bтак\s+сказать\b",
        r"\bгрубо\s+говоря\b",
        r"\bна\s+самом\s+деле\b",
        r"\bэто\s+самое\b",
        r"\bпоходу\b",
        r"\bмаленько\b",
    )
    _DIMINUTIVE_PATTERNS = (
        r"\bминуточку\b",
        r"\bтрубочку\b",
        r"\bзвоночек\b",
        r"\bдоговорчик\b",
        r"\bзаявочка\b",
        r"\bдокументик\b",
        r"\bсправочка\b",
        r"\bденежки\b",
    )
    _NEGATIVE_PHRASES = (
        r"\bк\s+сожалению\b",
        r"\bне\s+знаю\b",
        r"\bне\s+могу\s+подсказать\b",
        r"\bэто\s+не\s+ко\s+мне\b",
        r"\bэто\s+не\s+моя\s+зона\b",
        r"(?:^|\.\s+|\n)нет\b",
        r"\bвам\s+нужно\b",
        r"\bвы\s+должны\b",
        r"\bвы\s+не\s+поняли\b",
        r"\bвы\s+неправильно\s+поняли\b",
        r"\bваша\s+проблема\b",
        r"\bвы\s+не\s+правы\b",
        r"\bвас\s+беспокоит\b",
    )
    _CONSULTATION_PATTERNS = (
        r"\bнеобходимо\b",
        r"\bнужно\b",
        r"\bпорядок\b",
        r"\bдокумент",
        r"\bсрок",
        r"\bшаг\b",
        r"\bподать\b",
        r"\bзаявлен",
        r"\bличный\s+кабинет\b",
    )
    _RESOLUTION_PATTERNS = (
        r"\bрешили\b",
        r"\bвопрос\s+решен\b",
        r"\bпонятно\b",
        r"\bспасибо\b",
        r"\bблагодар",
    )
    _ENGAGEMENT_PATTERNS = (
        r"\bуточн",
        r"\bправильно\s+ли\s+я\s+понял\b",
        r"\bдавайте\s+разбер",
        r"\bподскажите,\s*пожалуйста\b",
        r"\bостались\s+ли\s+вопросы\b",
        r"\bчем\s+еще\s+могу\s+помочь\b",
    )
    _COMMON_NAMES = (
        "алексей",
        "александр",
        "андрей",
        "анна",
        "артем",
        "артём",
        "валерия",
        "виктория",
        "виктор",
        "владимир",
        "дарья",
        "денис",
        "дмитрий",
        "евгений",
        "елена",
        "екатерина",
        "иван",
        "игорь",
        "ирина",
        "кирилл",
        "константин",
        "ксения",
        "мария",
        "максим",
        "михаил",
        "надежда",
        "наталья",
        "никита",
        "оксана",
        "ольга",
        "павел",
        "полина",
        "роман",
        "светлана",
        "сергей",
        "софья",
        "татьяна",
        "юлия",
        "яна",
    )

    # Слова, которые не считаем именем в коротком ответе заявителя.
    _APPLICANT_NAME_STOPWORDS = frozenset(
        {
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
            "спасибо",
            "пожалуйста",
            "здравствуйте",
            "добрый",
            "день",
            "алло",
            "слушаю",
            "хорошо",
            "ладно",
            "понятно",
            "извините",
            "простите",
            "конечно",
            "я",
        }
    )

    def _count_pattern_hits(self, text: str, patterns: tuple[str, ...]) -> int:
        return sum(1 for pattern in patterns if re.search(pattern, text))

    @staticmethod
    def _clean_name_token(raw: str) -> str:
        s = raw.strip()
        s = re.sub(r"^[\s\.…,:;!?«»\"'(\[\{]+|[\s\.…,:;!?»\"')\]\}]+$", "", s)
        return s.strip()

    @staticmethod
    def _title_cyrillic_name(token: str) -> str:
        t = CallQualityEvaluator._clean_name_token(token)
        if not t:
            return t
        if "-" in t:
            return "-".join(
                (p[0].upper() + p[1:].lower()) if len(p) > 1 else p.upper()
                for p in t.split("-")
                if p
            )
        return t[0].upper() + t[1:].lower() if len(t) > 1 else t.upper()

    def _normalize_operator_name(self, name: str) -> str | None:
        name = self._clean_name_token(name)
        if not name:
            return None
        key = name.lower()
        if key in self._OPERATOR_ALIASES:
            return self._OPERATOR_ALIASES[key]
        for part in re.split(r"[\s\-.]+", key):
            if part in self._OPERATOR_ALIASES:
                return self._OPERATOR_ALIASES[part]
        return None

    def _find_operator_name(self, transcript: str, forced_operator_name: str | None) -> tuple[str, bool]:
        """
        Имя оператора в эвристическом режиме — только из ручного forced_operator_name.
        Автоопределение по тексту отключено (для «Авто» используется облачный API в web_gui).
        """
        if forced_operator_name:
            normalized = self._normalize_operator_name(forced_operator_name)
            if normalized:
                return normalized, True
            return "Не определено", False
        return "Не определено", False

    @staticmethod
    def _extract_role_text(role_transcript: str | None, role: str) -> str:
        prefix = f"{role}:"
        rt = role_transcript or ""
        lines = [
            line[len(prefix):].strip()
            for line in rt.splitlines()
            if line.startswith(prefix)
        ]
        return " ".join(lines).strip()

    @staticmethod
    def _iter_role_blocks(role_transcript: str) -> list[tuple[str, str]]:
        """Разбор «Оператор: …» / «Заявитель: …» с переносами строк внутри блока."""
        if not role_transcript or not role_transcript.strip():
            return []
        blocks: list[tuple[str, str]] = []
        current_role: str | None = None
        chunks: list[str] = []
        header = re.compile(r"^(Оператор|Заявитель):\s*(.*)$")
        for line in role_transcript.splitlines():
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

    def _extract_name_from_applicant_reply(self, text: str) -> str | None:
        """Имя из ответа заявителя после вопроса «как вас зовут» и т.п."""
        raw = text.strip()
        if not raw:
            return None
        low = raw.lower()

        structured = (
            r"меня\s+зовут[\s,:-]+([а-яё-]{2,30})\b",
            r"зовут\s+меня[\s,:-]+([а-яё-]{2,30})\b",
            r"\bэто\s+([а-яё-]{2,25})\b",
            r"\bимя\s+([а-яё-]{2,25})\b",
            r"\bя\s+([а-яё-]{2,25})\s*[,.]",
        )
        for pat in structured:
            m = re.search(pat, low)
            if m:
                tok = self._clean_name_token(m.group(1))
                if (
                    len(tok) >= 2
                    and tok not in self._APPLICANT_NAME_STOPWORDS
                    and re.fullmatch(r"[а-яё-]+", tok, re.IGNORECASE)
                ):
                    return self._title_cyrillic_name(tok)

        compact = re.sub(r"\s+", " ", low).strip()
        if len(compact) <= 45 and compact.count(" ") <= 4:
            words = re.findall(r"[а-яё]{2,}", compact)
            words = [w for w in words if w not in self._APPLICANT_NAME_STOPWORDS and len(w) >= 3]
            if len(words) == 1:
                return self._title_cyrillic_name(words[0])
            # «Мирослава Строфская.» — оператор обычно называет по имени; сохраняем оба слова для отчёта
            if len(words) == 2:
                return (
                    f"{self._title_cyrillic_name(words[0])} "
                    f"{self._title_cyrillic_name(words[1])}"
                )
        return None

    def _find_applicant_name_from_dialog(self, role_transcript: str) -> str | None:
        """
        1) Оператор спрашивает имя → следующая реплика заявителя (в т.ч. короткое «Наталья»).
        2) Любая реплика заявителя с «меня зовут …» / явной самопрезентацией.
        """
        blocks = self._iter_role_blocks(role_transcript)
        for i, (role, text) in enumerate(blocks):
            if role != "Оператор":
                continue
            low = text.lower()
            asked = any(re.search(p, low) for p in self._ASK_APPLICANT_NAME_PATTERNS)
            if not asked:
                continue
            for j in range(i + 1, len(blocks)):
                r2, t2 = blocks[j]
                if r2 == "Оператор":
                    break
                if r2 == "Заявитель":
                    name = self._extract_name_from_applicant_reply(t2)
                    if name:
                        return name
        for role, text in blocks:
            if role != "Заявитель":
                continue
            low = text.lower()
            if not re.search(
                r"меня\s+зовут|зовут\s+меня|это\s+[а-яё]{2,}|имя\s+[а-яё]{2,}", low
            ):
                continue
            name = self._extract_name_from_applicant_reply(text)
            if name:
                return name
        return None

    @staticmethod
    def _count_applicant_name_token_in_text(text_lower: str, applicant_name: str | None) -> int:
        """
        Сколько раз в тексте произносится **первое слово** имени заявителя (обычно имя, не фамилия).

        Для ответа «Имя Фамилия» в речи чаще только **имя** — ищем в первую очередь
        первое слово, а не целую строку (иначе «мирослава строфская» не матчит «Мирослава,»).
        Границы — по-кириллически, без ``\\b`` (на некоторых сборках оно хуже для «имя,»).
        """
        if not applicant_name or not (text_lower or "").strip():
            return 0
        parts = [p for p in re.split(r"\s+", applicant_name.strip().lower()) if p]
        if not parts:
            return 0
        primary = parts[0]
        if len(primary) < 2:
            return 0
        # Не буква кириллицы слева/справа (запятая, пробел, начало строки — ок)
        pat = rf"(?<![а-яё]){re.escape(primary)}(?![а-яё])"
        return len(re.findall(pat, text_lower))

    @classmethod
    def applicant_name_checklist_ok(
        cls,
        operator_text_lower: str,
        applicant_text_lower: str,
        applicant_name: str | None,
    ) -> tuple[bool, int, int]:
        """
        Пункт «Называл заявителя по имени ≥2 раз» (по смыслу для ЕКЦ):

        - **Да**, если оператор назвал имя заявителя **≥ 2 раза**, **или**
        - **Да**, если имя (первое слово из ``applicant_name``) прозвучало **≥ 3 раза**
          **во всём диалоге** (реплики оператора и заявителя вместе).

        Возвращает ``(ok, hits_operator, hits_full_dialog)``.
        """
        op_hits = cls._count_applicant_name_token_in_text(operator_text_lower, applicant_name)
        dialog = f"{operator_text_lower} {applicant_text_lower}".strip()
        all_hits = cls._count_applicant_name_token_in_text(dialog, applicant_name)
        ok = op_hits >= 2 or all_hits >= 3
        return ok, op_hits, all_hits

    def _find_applicant_name(self, transcript: str) -> str | None:
        """Эвристики по тексту (частота / фраза «меня зовут»). Полный role_transcript — в evaluate."""
        if not (transcript or "").strip():
            return None
        text = transcript.lower()

        for name in self._COMMON_NAMES:
            hits = len(re.findall(rf"\b{re.escape(name)}\b", text))
            if hits >= 2:
                return name[0].upper() + name[1:] if len(name) > 1 else name.upper()
            if hits == 1 and len(name) >= 4:
                # Редкие имена чаще уникальны в реплике заявителя
                if re.search(rf"(меня\s+зовут|зовут\s+меня|это\s+){re.escape(name)}\b", text):
                    return name[0].upper() + name[1:] if len(name) > 1 else name.upper()
        return None

    def evaluate(
        self,
        transcript: str,
        role_transcript: str,
        forced_operator_name: str | None = None,
    ) -> QualityEvaluation:
        operator_text_raw = self._extract_role_text(role_transcript, "Оператор")
        applicant_text_raw = self._extract_role_text(role_transcript, "Заявитель")
        operator_text = operator_text_raw.lower()
        applicant_text = applicant_text_raw.lower()
        dialog_text = f"{operator_text} {applicant_text}".strip()

        operator_name, operator_in_staff = self._find_operator_name(
            operator_text_raw or transcript,
            forced_operator_name,
        )
        applicant_name = self._find_applicant_name_from_dialog(role_transcript)
        if not applicant_name and (applicant_text_raw or "").strip():
            applicant_name = self._extract_name_from_applicant_reply(applicant_text_raw)
        applicant_name = applicant_name or self._find_applicant_name(applicant_text_raw or transcript)

        has_greeting = self._count_pattern_hits(operator_text, self._GREETING_PATTERNS) > 0
        has_intro = self._count_pattern_hits(operator_text, self._INTRO_PATTERNS) > 0
        has_goodbye = self._count_pattern_hits(operator_text, self._GOODBYE_PATTERNS) > 0
        has_offer_help = self._count_pattern_hits(operator_text, self._OFFER_HELP_PATTERNS) > 0
        has_thank = self._count_pattern_hits(operator_text, self._THANK_PATTERNS) > 0
        # Упоминания имени: для чек-листа — оператор ≥2 ИЛИ по всему диалогу ≥3;
        # для «познакомился» — классически: спросил имя ИЛИ (известно имя и оператор ≥2).
        uses_name_checklist, op_name_hits, dialog_name_hits = self.applicant_name_checklist_ok(
            operator_text, applicant_text, applicant_name
        )
        operator_asked_name = self._count_pattern_hits(operator_text, self._ASK_APPLICANT_NAME_PATTERNS) > 0
        has_acquaintance = operator_asked_name or (
            applicant_name is not None and op_name_hits >= 2
        )

        # 7 script points, ~1.43 pts each → max 10
        script_score = round(
            (
                (1 if has_greeting else 0)
                + (1 if has_intro else 0)
                + (1 if has_acquaintance else 0)
                + (1 if uses_name_checklist else 0)
                + (1 if has_offer_help else 0)
                + (1 if has_thank else 0)
                + (1 if has_goodbye else 0)
            )
            * (10 / 7)
        )
        script_score = min(script_score, 10)

        filler_hits = len(re.findall("|".join(self._FILLER_PATTERNS), operator_text))
        diminutive_hits = len(re.findall("|".join(self._DIMINUTIVE_PATTERNS), operator_text))
        negative_hits = len(re.findall("|".join(self._NEGATIVE_PHRASES), operator_text))
        penalties = filler_hits + (2 * diminutive_hits) + (2 * negative_hits)
        speech_score = max(0, 10 - penalties)

        consultation_hits = self._count_pattern_hits(operator_text, self._CONSULTATION_PATTERNS)
        resolution_hits = self._count_pattern_hits(applicant_text or dialog_text, self._RESOLUTION_PATTERNS)
        consultation_score = min(10, consultation_hits + (2 * resolution_hits))
        operator_questions = len(re.findall(r"\?", operator_text_raw))
        engagement_hits = self._count_pattern_hits(operator_text, self._ENGAGEMENT_PATTERNS)
        engagement_score = min(10, engagement_hits * 2 + min(4, operator_questions))

        total_score = round(
            (0.30 * script_score)
            + (0.20 * speech_score)
            + (0.30 * consultation_score)
            + (0.20 * engagement_score)
        )
        max_score = 10

        if has_acquaintance and not operator_asked_name:
            acquaintance_label = "да (заявитель представился сам)"
        elif has_acquaintance:
            acquaintance_label = "да"
        else:
            acquaintance_label = "нет"
        script_details = [
            f"Приветствие + предложение помочь: {'да' if has_greeting else 'нет'}",
            f"Представился по имени: {'да' if has_intro else 'нет'}",
            f"Познакомился с заявителем: {acquaintance_label}",
            (
                f"Называл заявителя по имени ≥2 раз: {'да' if uses_name_checklist else 'нет'}"
                + (f" ({applicant_name})" if applicant_name else "")
                + (
                    f" [оп.{op_name_hits}/всего{dialog_name_hits}]"
                    if applicant_name and (op_name_hits or dialog_name_hits)
                    else ""
                )
            ),
            f"Предложил дополнительную помощь: {'да' if has_offer_help else 'нет'}",
            f"Поблагодарил за обращение: {'да' if has_thank else 'нет'}",
            f"Персонализированное прощание: {'да' if has_goodbye else 'нет'}",
        ]

        positives: list[str] = []
        negatives: list[str] = []

        if script_score >= 7:
            positives.append("Скрипт звонка в основном соблюден.")
        else:
            negatives.append("Нарушения по скрипту: проверьте приветствие/представление/завершение.")

        if speech_score >= 8:
            positives.append("Речь оператора чистая, без заметных паразитов и запрещенных формулировок.")
        else:
            negatives.append(
                "Есть речевые риски: слова-паразиты/уменьшительные формы/фразы 'к сожалению', 'не знаю'."
            )

        if consultation_score >= 7:
            positives.append("Консультация предметная, вероятно вопрос заявителя закрыт.")
        else:
            negatives.append("Консультация недостаточно четкая: не хватает шагов, сроков или фиксации результата.")
        if engagement_score >= 7:
            positives.append("Оператор проявляет вовлеченность и ведет диалог уточняющими вопросами.")
        else:
            negatives.append("Низкая вовлеченность: добавьте уточняющие вопросы и активное сопровождение заявителя.")
        if operator_name == "Не определено":
            negatives.append(
                "Имя оператора не извлечено из транскрипта (ожидаются фразы вроде «меня зовут …» "
                "или имя из штатного списка в репликах оператора; можно выбрать оператора вручную в интерфейсе)."
            )
        elif not operator_in_staff:
            negatives.append("Имя оператора не из штатного списка (показано как в транскрипте).")

        if not positives:
            positives.append("Сильные стороны слабо выражены в текущем транскрипте.")

        ev = QualityEvaluation(
            total_score=total_score,
            max_score=max_score,
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
        from evaluation_leniency import apply_focus_operator_adjustments

        return apply_focus_operator_adjustments(ev)


def analyze_tone(audio_path: str | Path) -> str:
    """Approximate tone from signal features (energy/pitch variation)."""
    try:
        import librosa  # type: ignore
        import numpy as np
    except ImportError:
        return "Не определен (установите librosa для анализа тона)"

    try:
        y, sr = librosa.load(str(audio_path), sr=16000, mono=True)
        if y.size == 0:
            return "Не определен (пустой аудиосигнал)"

        # Signal energy and variability.
        rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=256)[0]
        mean_rms = float(np.mean(rms))
        std_rms = float(np.std(rms))

        # Pitch statistics as rough emotional proxy.
        f0, _, _ = librosa.pyin(
            y,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"),
        )
        voiced = f0[~np.isnan(f0)]
        pitch_var = float(np.std(voiced)) if voiced.size else 0.0

        agitation = (std_rms * 4.0) + (pitch_var / 120.0) + (mean_rms * 2.0)
        if agitation < 0.8:
            return "Спокойный"
        if agitation < 1.8:
            return "Нейтральный"
        return "Напряженный/эмоциональный"
    except Exception:
        return "Не определен (ошибка анализа аудиосигнала)"


def build_report(result: TranscriptionResult, evaluation: QualityEvaluation) -> str:
    lines = [
        "",
        "=== Оценка консультации (ДВФУ) ===",
        f"Оператор: {evaluation.operator_name}",
        f"Оператор из штатного списка: {'да' if evaluation.operator_in_staff else 'нет'}",
        f"Заявитель: {evaluation.applicant_name or 'Не определено'}",
        f"Язык распознавания: {result.language or 'не определен'}",
        f"ASR профиль: {result.asr_profile}",
        f"ASR модель: {result.asr_model}",
        f"Роли в транскрипте: {result.role_attribution or 'не указано'}",
        f"Уверенность ролей: {result.role_confidence}",
        f"Тон голоса (оценка): {result.tone_summary}",
        f"Итоговая экспертная оценка: {evaluation.total_score}/{evaluation.max_score}",
        "",
        "Критерий 1. Соблюдение скрипта: "
        f"{evaluation.script_score}/10",
        "Критерий 2. Чистота речи оператора: "
        f"{evaluation.speech_score}/10",
        "Критерий 3. Качество консультации и решение вопроса: "
        f"{evaluation.consultation_score}/10",
        "Критерий 4. Вовлеченность оператора: "
        f"{evaluation.engagement_score}/10",
        "",
        "Проверка скрипта:",
    ]
    lines.extend(f"- {item}" for item in evaluation.script_details)
    lines.append("")
    lines.append("Плюсы (кратко):")
    lines.extend(f"- {item}" for item in evaluation.positives[:3])
    lines.append("")
    lines.append("Минусы (кратко):")
    if evaluation.negatives:
        lines.extend(f"- {item}" for item in evaluation.negatives[:3])
    else:
        lines.append("- Явных минусов по текущим критериям не выявлено.")
    lines.append("")
    if result.role_diagnostics:
        lines.append("Диагностика ролей:")
        lines.append(result.role_diagnostics)
        lines.append("")
    lines.append("Транскрипт по ролям:")
    lines.append(result.role_text if result.role_text else "<пусто>")
    lines.append("")
    lines.append("Сплошной транскрипт:")
    lines.append(result.text if result.text else "<пусто>")
    return "\n".join(lines)


def _default_report_path(audio_path: str | Path) -> Path:
    from results_paths import ensure_results_dir

    source = Path(audio_path)
    stem = source.stem if source.stem else "report"
    return ensure_results_dir() / f"{stem}_report.txt"


def main() -> None:
    try:
        from dotenv import load_dotenv

        _root = Path(__file__).resolve().parent
        load_dotenv(_root / ".env")
        load_dotenv()
    except ImportError:
        pass

    parser = argparse.ArgumentParser(
        description="Транскрибация и базовая оценка качества разговоров колл-центра ДВФУ."
    )
    parser.add_argument("audio", nargs="?", help="Путь к аудиофайлу (wav/mp3/m4a и т.п.)")
    parser.add_argument(
        "--model",
        default="best-ru",
        help="Whisper model name (medium|large-v3|best-ru)",
    )
    parser.add_argument("--compute-type", default="int8", help="Whisper compute_type")
    parser.add_argument(
        "--asr-profile",
        default="medium_ru",
        choices=["medium_ru", "ideal_ru"],
        help="Профиль ASR-декодинга: medium_ru|ideal_ru",
    )
    parser.add_argument(
        "--heavy-diarization",
        action="store_true",
        help="Включить heavy diarization как основной путь с fallback на light clustering",
    )
    parser.add_argument(
        "--heavy-diarization-timeout-seconds",
        type=int,
        default=240,
        help="Таймаут heavy diarization в секундах",
    )
    parser.add_argument(
        "--llm-backend",
        choices=["off", "embedded", "remote"],
        default="off",
        help=(
            "LLM пост-редактор: off | embedded (DeepSeek с Hugging Face, без Ollama) | "
            "remote (нужны LLM_POST_EDIT_BASE_URL и LLM_POST_EDIT_MODEL)"
        ),
    )
    parser.add_argument(
        "--llm-embedded-model",
        default=None,
        help="Репозиторий Hugging Face для встроенной модели (иначе LLM_EMBEDDED_MODEL_ID или дефолт DeepSeek-R1 1.5B)",
    )
    parser.add_argument(
        "--llm-post-edit",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--llm-post-edit-timeout-seconds",
        type=float,
        default=120.0,
        help="Таймаут LLM пост-редактора (сек)",
    )
    parser.add_argument(
        "--enable-post-edit",
        action="store_true",
        help="Включить локальный пост-редактор пунктуации (medium_ru и ideal_ru)",
    )
    parser.add_argument(
        "--post-edit-timeout-seconds",
        type=int,
        default=60,
        help="Таймаут локального пост-редактора в секундах",
    )
    parser.add_argument(
        "--total-time-budget-seconds",
        type=int,
        default=300,
        help="Общий тайм-бюджет обработки звонка в секундах",
    )
    parser.add_argument(
        "--skip-first-seconds",
        type=float,
        default=None,
        help=(
            "Пропустить N секунд с начала (clip_timestamps). "
            "Если не задано — из имени файла берётся ожидание как последний фрагмент _MM-SS (минуты-секунды)."
        ),
    )
    parser.add_argument("--language", default="ru", help="Язык распознавания (фиксирован ru)")
    parser.add_argument(
        "--model-load-timeout-seconds",
        type=int,
        default=600,
        help="Таймаут загрузки модели в секундах",
    )
    parser.add_argument(
        "--operator-name",
        default=None,
        help="Имя оператора (если не указать, будет автоопределение по транскрипту)",
    )
    parser.add_argument(
        "--warmup-only",
        action="store_true",
        help="Только предварительно скачать/загрузить модель, без анализа аудио",
    )
    parser.add_argument(
        "--report-path",
        default=None,
        help="Куда сохранить итоговый отчет (.txt). По умолчанию рядом с запуском.",
    )
    parser.add_argument(
        "--no-google-sheets",
        action="store_true",
        help="Не отправлять строку в Google Таблицу (если настроен GOOGLE_SHEETS_CREDENTIALS_JSON)",
    )
    args = parser.parse_args()
    llm_backend = args.llm_backend
    if getattr(args, "llm_post_edit", False) and llm_backend == "off":
        llm_backend = "remote"
    enable_llm = llm_backend != "off"
    active_backend = llm_backend if enable_llm else "embedded"

    transcriber = Transcriber(
        model_name=args.model,
        compute_type=args.compute_type,
        skip_first_seconds=args.skip_first_seconds,
        language=args.language,
        model_load_timeout_seconds=args.model_load_timeout_seconds,
        asr_profile=args.asr_profile,
        heavy_diarization=args.heavy_diarization,
        heavy_diarization_timeout_seconds=args.heavy_diarization_timeout_seconds,
        enable_post_edit=args.enable_post_edit,
        post_edit_timeout_seconds=args.post_edit_timeout_seconds,
        total_time_budget_seconds=args.total_time_budget_seconds,
        enable_llm_post_edit=enable_llm,
        llm_backend=active_backend,
        llm_embedded_model_id=args.llm_embedded_model,
        llm_post_edit_timeout_seconds=args.llm_post_edit_timeout_seconds,
    )
    if args.warmup_only:
        print("[transcriber] Режим warmup: только загрузка модели...", flush=True)
        transcriber._ensure_model()
        print("[transcriber] Warmup завершен. Модель готова к работе.", flush=True)
        return

    if not args.audio:
        raise SystemExit("Укажите путь к аудио или используйте --warmup-only")

    result = transcriber.transcribe(args.audio)

    print("[evaluator] Оцениваю качество консультации...", flush=True)
    evaluator = CallQualityEvaluator()
    evaluation = evaluator.evaluate(
        result.text,
        result.role_text,
        forced_operator_name=args.operator_name,
    )
    print("[evaluator] Оценка готова.", flush=True)

    report = build_report(result, evaluation)
    print(report)
    output_path = Path(args.report_path) if args.report_path else _default_report_path(args.audio)
    output_path.write_text(report, encoding="utf-8")
    print(f"[evaluator] Отчет сохранен: {output_path}", flush=True)

    if not args.no_google_sheets:
        try:
            from sheets_export import append_analysis_row

            fn = Path(args.audio).name
            ok, msg = append_analysis_row(
                original_filename=fn,
                report_text=report,
                total_score=evaluation.total_score,
                max_score=evaluation.max_score,
                operator_name=evaluation.operator_name,
            )
            tag = "[sheets] OK" if ok else "[sheets] skip/err"
            print(f"{tag}: {msg}", flush=True)
        except Exception as exc:
            print(f"[sheets] Ошибка: {exc}", flush=True)


if __name__ == "__main__":
    main()
