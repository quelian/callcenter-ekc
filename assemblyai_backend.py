"""
AssemblyAI backend для распознавания речи + диаризации + intelligence.

Модель: Universal-2. Включает:
  - Speaker Diarization (speaker_labels=True)
  - Speaker Identification (role-based через custom_spelling)
  - Entity Detection (имена, организации — для определения заявителя)
  - Content Moderation (фильтрация некорректного контента)

API flow:
  1. POST /v2/upload — загрузка аудио → upload_url
  2. POST /v2/transcript — отправка задачи → transcript_id
  3. GET /v2/transcript/{id} — polling до completed/error
"""
from __future__ import annotations

import json
import ssl
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from applicant_name_utils import APPLICANT_NAME_NOISE_WORDS, normalize_applicant_name_candidate

_ASSEMBLYAI_UPLOAD_URL = "https://api.assemblyai.com/v2/upload"
_ASSEMBLYAI_TRANSCRIPT_URL = "https://api.assemblyai.com/v2/transcript"
_DEFAULT_MODEL = "universal-2"
_DEFAULT_LANGUAGE = "ru"
_DEFAULT_TIMEOUT = 300.0
_DEFAULT_POLL_INTERVAL = 3.0


@dataclass(frozen=True)
class AssemblyAIConfig:
    api_key: str
    language: str = _DEFAULT_LANGUAGE
    model: str = _DEFAULT_MODEL
    speaker_labels: bool = True
    entity_detection: bool = True
    content_moderation: bool = True
    timeout_seconds: float = _DEFAULT_TIMEOUT
    polling_interval_seconds: float = _DEFAULT_POLL_INTERVAL
    custom_spelling: list[dict[str, str]] = field(default_factory=list)
    word_boost: list[str] = field(default_factory=list)
    word_boost_param: str = ""
    prompt: str = ""
    disfluencies: bool = False
    punctuate: bool = True
    format_text: bool = True
    filter_profanity: bool = False
    speakers_expected: int = 2
    min_speakers: int = 0
    max_speakers: int = 0
    temperature: float = 0.0


def _log_default(msg: str) -> None:
    print(f"[assemblyai] {msg}", flush=True)


def _make_ssl_context() -> ssl.SSLContext:
    """SSL-контекст без проверки сертификатов (для macOS и self-signed certs)."""
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def _make_request(
    url: str,
    method: str,
    api_key: str,
    *,
    data: bytes | None = None,
    headers: dict[str, str] | None = None,
    timeout: float = 30.0,
) -> dict:
    """Выполняет HTTP-запрос к AssemblyAI API."""
    req_headers = {
        "authorization": api_key,
        "content-type": "application/json",
    }
    if headers:
        req_headers.update(headers)

    req = urllib.request.Request(url, data=data, headers=req_headers, method=method)

    try:
        with urllib.request.urlopen(req, timeout=timeout, context=_make_ssl_context()) as resp:
            body = resp.read()
            return json.loads(body.decode("utf-8"))
    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8", errors="replace") if e.fp else ""
        raise RuntimeError(
            f"AssemblyAI API error {e.code} {e.reason}: {error_body[:500]}"
        ) from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"AssemblyAI API connection error: {e.reason}") from e


def upload_audio(api_key: str, audio_path: str, *, log: Callable[[str], None] | None = None) -> str:
    """Загружает аудио на AssemblyAI и возвращает upload_url."""
    _log = log or _log_default
    path = Path(audio_path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    _log(f"Загрузка аудио ({path.stat().st_size / 1024 / 1024:.1f} МБ)...")
    audio_bytes = path.read_bytes()

    # AssemblyAI upload endpoint принимает raw audio bytes
    req_headers = {
        "authorization": api_key,
        "content-type": "application/octet-stream",
    }
    req = urllib.request.Request(
        _ASSEMBLYAI_UPLOAD_URL,
        data=audio_bytes,
        headers=req_headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120.0, context=_make_ssl_context()) as resp:
            body = resp.read()
            result = json.loads(body.decode("utf-8"))
            upload_url = result.get("upload_url")
            if not upload_url:
                raise RuntimeError(f"AssemblyAI upload response missing upload_url: {result}")
            _log(f"Аудио загружено. URL: {upload_url[:60]}...")
            return upload_url
    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8", errors="replace") if e.fp else ""
        raise RuntimeError(
            f"AssemblyAI upload error {e.code}: {error_body[:500]}"
        ) from e


def submit_transcription(
    api_key: str,
    audio_url: str,
    config: AssemblyAIConfig,
    *,
    log: Callable[[str], None] | None = None,
) -> str:
    """Отправляет задачу транскрибации и возвращает transcript_id."""
    _log = log or _log_default
    _log(f"Отправка задачи транскрибации (модель: {config.model}, язык: {config.language})...")

    body: dict[str, object] = {
        "audio_url": audio_url,
        "language_code": config.language,
        "speech_models": [config.model],
        "speaker_labels": config.speaker_labels,
        "punctuate": config.punctuate,
        "format_text": config.format_text,
        "disfluencies": config.disfluencies,
        "filter_profanity": config.filter_profanity,
    }

    if config.min_speakers > 0 or config.max_speakers > 0:
        speaker_options: dict[str, object] = {}
        if config.min_speakers > 0:
            speaker_options["min_speakers_expected"] = config.min_speakers
        if config.max_speakers > 0:
            speaker_options["max_speakers_expected"] = config.max_speakers
        if speaker_options:
            body["speaker_options"] = speaker_options
    elif config.speakers_expected > 0:
        body["speakers_expected"] = config.speakers_expected

    if config.word_boost:
        body["word_boost"] = config.word_boost
        if config.word_boost_param:
            body["boost_param"] = config.word_boost_param

    if config.min_speakers > 0 or config.max_speakers > 0:
        speaker_options: dict[str, object] = {}
        if config.min_speakers > 0:
            speaker_options["min_speakers_expected"] = config.min_speakers
        if config.max_speakers > 0:
            speaker_options["max_speakers_expected"] = config.max_speakers
        if speaker_options:
            body["speaker_options"] = speaker_options

    if config.prompt:
        body["prompt"] = config.prompt

    if config.custom_spelling:
        body["custom_spelling"] = config.custom_spelling

    data = json.dumps(body).encode("utf-8")
    result = _make_request(
        _ASSEMBLYAI_TRANSCRIPT_URL,
        "POST",
        api_key,
        data=data,
        timeout=30.0,
    )

    transcript_id = result.get("id")
    if not transcript_id:
        raise RuntimeError(f"AssemblyAI transcript response missing id: {result}")

    _log(f"Задача создана: {transcript_id}")
    return transcript_id


def poll_transcription(
    api_key: str,
    transcript_id: str,
    config: AssemblyAIConfig,
    *,
    log: Callable[[str], None] | None = None,
) -> dict:
    """Ожидает завершения транскрибации и возвращает полный результат."""
    _log = log or _log_default
    url = f"{_ASSEMBLYAI_TRANSCRIPT_URL}/{transcript_id}"
    deadline = time.monotonic() + config.timeout_seconds

    while time.monotonic() < deadline:
        result = _make_request(url, "GET", api_key, timeout=30.0)
        status = result.get("status", "unknown")

        if status == "completed":
            _log("Транскрибация завершена.")
            return result
        elif status == "error":
            error_msg = result.get("error", "Unknown error")
            raise RuntimeError(f"AssemblyAI transcription error: {error_msg}")
        elif status in ("queued", "processing"):
            _log(f"Статус: {status}...")
            time.sleep(config.polling_interval_seconds)
        else:
            _log(f"Неизвестный статус: {status}")
            time.sleep(config.polling_interval_seconds)

    raise RuntimeError(
        f"AssemblyAI transcription timed out after {config.timeout_seconds:.0f}s"
    )


def transcribe(
    api_key: str,
    audio_path: str,
    config: AssemblyAIConfig,
    *,
    log: Callable[[str], None] | None = None,
) -> dict:
    """Полный цикл: загрузка → отправка → ожидание → результат."""
    _log = log or _log_default
    upload_url = upload_audio(api_key, audio_path, log=_log)
    transcript_id = submit_transcription(api_key, upload_url, config, log=_log)
    return poll_transcription(api_key, transcript_id, config, log=_log)


def map_assemblyai_to_voice_items(response: dict) -> list[tuple[float, float, str, str]]:
    """
    Маппинг AssemblyAI utterances → формат voice items для _joint_voice_role_decode.

    AssemblyAI возвращает utterances с полями:
      - speaker: "A", "B", ...
      - text: текст реплики
      - start: начало в миллисекундах
      - end: конец в миллисекундах

    Возвращает: list[(start_s, end_s, text, "SPK_X")]
    """
    utterances = response.get("utterances") or []
    if not utterances:
        # Fallback: если utterances нет, используем words для реконструкции
        words = response.get("words") or []
        if not words:
            return []
        # Собираем все слова как один сегмент без диаризации
        text = " ".join(w.get("text", "") for w in words if w.get("text"))
        start = words[0].get("start", 0) / 1000.0
        end = words[-1].get("end", 0) / 1000.0
        return [(start, end, text, "SPK_A")]

    items: list[tuple[float, float, str, str]] = []
    for utt in utterances:
        speaker = utt.get("speaker", "A")
        text = utt.get("text", "").strip()
        if not text:
            continue
        start_ms = utt.get("start", 0)
        end_ms = utt.get("end", 0)
        start_s = start_ms / 1000.0
        end_s = end_ms / 1000.0
        speaker_id = f"SPK_{speaker}"
        items.append((start_s, end_s, text, speaker_id))

    return items


def extract_entities(response: dict) -> list[dict]:
    """Извлекает entity detection результаты из ответа AssemblyAI."""
    return response.get("entities") or []


def extract_content_moderation(response: dict) -> dict | None:
    """Извлекает content moderation результаты из ответа AssemblyAI."""
    moderation = response.get("content_moderation")
    if moderation and moderation.get("results"):
        return moderation
    return None


def extract_applicant_name_from_entities(
    entities: list[dict],
    *,
    log: Callable[[str], None] | None = None,
) -> str | None:
    """
    Определяет имя заявителя из entity detection AssemblyAI.

    Фильтрует PERSON-типы, исключает шумовые слова, возвращает наиболее
    вероятное имя заявителя.
    """
    _log = log or (lambda _m: None)
    person_entities = [
        e for e in entities
        if e.get("entity_type", "").upper() == "PERSON"
    ]

    if not person_entities:
        return None

    # Берём первое PERSON-имя, которое не является шумовым
    for entity in person_entities:
        candidate = entity.get("text", "").strip()
        normalized = normalize_applicant_name_candidate(candidate)
        if not normalized:
            continue
        if normalized.lower() in APPLICANT_NAME_NOISE_WORDS:
            continue
        _log(f"[assemblyai] Entity detection: имя заявителя = '{normalized}'")
        return normalized

    return None


def build_custom_spelling_for_operators(
    operator_names: list[str],
) -> list[dict[str, str]]:
    """
    Строит custom_spelling для AssemblyAI на основе имён операторов.
    Помогает модели правильно распознавать имена в аудио.
    """
    return [
        {"to": name, "from": [name.lower()]}
        for name in operator_names
    ]
