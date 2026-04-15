"""
Nexara backend для распознавания речи + диаризации.

API: POST https://api.nexara.ru/api/v1/audio/transcriptions
Auth: Bearer <API_KEY>
Diarization: task=diarize, diarization_setting=telephonic
"""
from __future__ import annotations

import json
import os
import ssl
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

_DEFAULT_LANGUAGE = "ru"
_DEFAULT_TIMEOUT = 300.0
_NEXARA_API_URL = "https://api.nexara.ru/api/v1/audio/transcriptions"


@dataclass(frozen=True)
class NexaraConfig:
    api_key: str
    language: str = _DEFAULT_LANGUAGE
    timeout_seconds: float = _DEFAULT_TIMEOUT
    num_speakers: int = 2
    diarization_setting: str = "telephonic"
    diarization_enabled: bool = True
    max_retries: int = 2  # Retry count for transient failures


def _make_ssl_context() -> ssl.SSLContext:
    """SSL-контекст без проверки сертификатов."""
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def _log_default(msg: str) -> None:
    print(f"[nexara] {msg}", flush=True)


def transcribe(
    api_key: str,
    audio_path: str,
    config: NexaraConfig,
    *,
    log: Callable[[str], None] | None = None,
) -> dict:
    """
    Синхронная транскрибация через Nexara API.
    Возвращает dict с полями: text, segments, language, duration.
    """
    _log = log or _log_default
    path = Path(audio_path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    if not path.is_file():
        raise ValueError(f"Path is not a file: {audio_path}")

    file_size_mb = path.stat().st_size / 1024 / 1024
    max_upload_mb = float(os.environ.get("NEXARA_MAX_UPLOAD_MB", "256"))
    if file_size_mb > max_upload_mb:
        raise ValueError(
            f"Audio file too large: {file_size_mb:.1f} МБ > {max_upload_mb:.0f} МБ limit. "
            f"Override with NEXARA_MAX_UPLOAD_MB env var."
        )
    _log(f"Отправка аудио ({file_size_mb:.1f} МБ) в Nexara...")

    # Multipart form-data request
    boundary = "----NexaraBoundary7d4a6c8e9f0b2a1"
    content_type = f"multipart/form-data; boundary={boundary}"

    body_parts: list[bytes] = []

    # Add form fields
    fields = {
        "model": "nexara-ru",
        "response_format": "verbose_json",
        "language": config.language,
    }
    if config.diarization_enabled:
        fields["task"] = "diarize"
        fields["num_speakers"] = str(config.num_speakers)
        fields["diarization_setting"] = config.diarization_setting
    for key, value in fields.items():
        body_parts.append(f"--{boundary}\r\n".encode())
        body_parts.append(
            f'Content-Disposition: form-data; name="{key}"\r\n\r\n'.encode()
        )
        body_parts.append(f"{value}\r\n".encode())

    # Add file
    body_parts.append(f"--{boundary}\r\n".encode())
    body_parts.append(
        f'Content-Disposition: form-data; name="file"; filename="{path.name}"\r\n'.encode()
    )
    # Infer Content-Type from file extension
    ext = path.suffix.lower()
    content_type_map = {".wav": "audio/wav", ".mp3": "audio/mpeg", ".m4a": "audio/mp4", ".ogg": "audio/ogg", ".flac": "audio/flac"}
    audio_content_type = content_type_map.get(ext, "audio/mpeg")
    body_parts.append(f"Content-Type: {audio_content_type}\r\n\r\n".encode())
    body_parts.append(path.read_bytes())
    body_parts.append(b"\r\n")
    body_parts.append(f"--{boundary}--\r\n".encode())

    body = b"".join(body_parts)

    timeout = max(30.0, min(float(config.timeout_seconds), 600.0))
    max_attempts = 1 + max(0, int(config.max_retries))

    last_exception = None
    for attempt in range(1, max_attempts + 1):
        if attempt > 1:
            backoff = min(2 ** attempt, 30)
            _log(f"Nexara: повторная попытка {attempt}/{max_attempts} через {backoff}с...")
            time.sleep(backoff)

        req = urllib.request.Request(
            _NEXARA_API_URL,
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": content_type,
            },
        )

        _log(f"Ожидание ответа (таймаут≤{timeout:.0f}с, попытка {attempt}/{max_attempts})...")
        t0 = time.perf_counter()

        try:
            with urllib.request.urlopen(req, timeout=timeout, context=_make_ssl_context()) as resp:
                status = resp.getcode()
                if status not in (200, 201, 202):
                    raise RuntimeError(f"Nexara API returned HTTP {status}")
                raw = resp.read().decode("utf-8", errors="replace")
                result = json.loads(raw)
                elapsed = time.perf_counter() - t0
                _log(f"Nexara ответ получен за {elapsed:.1f}с (попытка {attempt})")
                return result
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8", errors="replace") if e.fp else ""
            err_msg = f"Nexara API error {e.code} {e.reason}: {error_body[:500]}"
            last_exception = RuntimeError(err_msg)
            # Don't retry auth errors or client errors
            if e.code == 401 or e.code == 403 or (400 <= e.code < 500 and e.code != 429):
                raise RuntimeError(err_msg) from e
            _log(f"Nexara: переходная ошибка {e.code}, буду повторять: {error_body[:200]}")
        except urllib.error.URLError as e:
            err_msg = f"Nexara API connection error: {e.reason}"
            last_exception = RuntimeError(err_msg)
            _log(f"Nexara: ошибка подключения — {e.reason}")

    raise last_exception or RuntimeError("Nexara API: все попытки исчерпаны")


def validate_nexara_response(response: dict) -> tuple[str | None, str | None]:
    """Validate Nexara API response. Returns (error_type, error_message) or (None, None)."""
    if not isinstance(response, dict):
        return "invalid_response", "Response is not a JSON object"
    # Check for error field
    if "error" in response:
        return "api_error", str(response["error"])
    # Need at least one of: segments or text
    has_segments = bool(response.get("segments"))
    has_text = bool(response.get("text", "").strip())
    if not has_segments and not has_text:
        return "empty_result", "No segments or text in response"
    return None, None


def map_nexara_to_voice_items(response: dict) -> list[tuple[float, float, str, str]]:
    """
    Маппинг Nexara segments → voice items.

    Nexara возвращает:
      segments: [{"speaker": "speaker_0", "start": 0.0, "end": 5.0, "text": "..."}]

    Возвращает: list[(start_s, end_s, text, "SPK_X")]
    """
    segments = response.get("segments") or []
    if not isinstance(segments, list):
        segments = []
    if not segments:
        # Fallback: use text
        flat_text = (response.get("text") or "").strip()
        if flat_text:
            duration = float(response.get("duration", 0) or 0)
            return [(0.0, duration, flat_text, "SPK_A")]
        return []

    items: list[tuple[float, float, str, str]] = []
    for seg in segments:
        if not isinstance(seg, dict):
            continue
        speaker = seg.get("speaker")
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        try:
            start = float(seg.get("start", 0) or 0)
            end = float(seg.get("end", 0) or 0)
        except (TypeError, ValueError):
            continue
        if end <= start:
            continue
        if speaker is None:
            # No diarization: placeholder label, caller will remap via local diarization
            speaker_id = "SPK_0"
        else:
            # Normalize speaker ID: speaker_0 -> SPK_A, speaker_1 -> SPK_B
            try:
                idx = int(str(speaker).split("_")[-1])
                speaker_id = f"SPK_{chr(ord('A') + idx)}"
            except (ValueError, IndexError):
                speaker_id = f"SPK_{speaker}"
        items.append((start, end, text, speaker_id))

    return items
