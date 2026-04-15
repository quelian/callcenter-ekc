"""Tests for nexara_backend module: config, mapping, and validation."""
from __future__ import annotations

import json
import ssl
import urllib.error
from pathlib import Path

import pytest

import nexara_backend as nex


# ── NexaraConfig defaults ─────────────────────────────────────────────────


def test_nexara_config_defaults() -> None:
    cfg = nex.NexaraConfig(api_key="test")
    assert cfg.api_key == "test"
    assert cfg.language == "ru"
    assert cfg.timeout_seconds == 300.0
    assert cfg.num_speakers == 2
    assert cfg.diarization_setting == "telephonic"
    assert cfg.diarization_enabled is True
    assert cfg.max_retries == 2


# ── map_nexara_to_voice_items ────────────────────────────────────────────


def test_map_nexara_segments_to_voice_items() -> None:
    response = {
        "segments": [
            {"speaker": "speaker_0", "start": 0.0, "end": 3.5, "text": "Здравствуйте"},
            {"speaker": "speaker_1", "start": 4.0, "end": 7.0, "text": "Добрый день"},
        ]
    }
    items = nex.map_nexara_to_voice_items(response)
    assert len(items) == 2
    assert items[0] == (0.0, 3.5, "Здравствуйте", "SPK_A")
    assert items[1] == (4.0, 7.0, "Добрый день", "SPK_B")


def test_map_nexara_empty_segments_uses_text() -> None:
    response = {"segments": [], "text": "Привет мир", "duration": 5.0}
    items = nex.map_nexara_to_voice_items(response)
    assert len(items) == 1
    assert items[0] == (0.0, 5.0, "Привет мир", "SPK_A")


def test_map_nexara_no_segments_no_text() -> None:
    response = {"segments": [], "text": ""}
    items = nex.map_nexara_to_voice_items(response)
    assert items == []


def test_map_nexara_skips_empty_text_segments() -> None:
    response = {
        "segments": [
            {"speaker": "speaker_0", "start": 0.0, "end": 1.0, "text": "  "},
            {"speaker": "speaker_0", "start": 2.0, "end": 5.0, "text": "Hello"},
        ]
    }
    items = nex.map_nexara_to_voice_items(response)
    assert len(items) == 1
    assert items[0][2] == "Hello"


def test_map_nexara_skips_invalid_segments() -> None:
    response = {
        "segments": [
            "not_a_dict",
            {"speaker": "speaker_0", "start": 1.0, "end": 0.5, "text": "bad"},
            {"speaker": "speaker_0", "start": 0.0, "end": 2.0, "text": "ok"},
        ]
    }
    items = nex.map_nexara_to_voice_items(response)
    assert len(items) == 1
    assert items[0][2] == "ok"


def test_map_nexara_no_speaker_defaults_spk_0() -> None:
    response = {
        "segments": [
            {"speaker": None, "start": 0.0, "end": 2.0, "text": "No speaker"}
        ]
    }
    items = nex.map_nexara_to_voice_items(response)
    assert items[0][3] == "SPK_0"


def test_map_nexara_segments_not_list() -> None:
    response = {"segments": "not_a_list", "text": "fallback"}
    items = nex.map_nexara_to_voice_items(response)
    # segments is not a list, so it gets normalized to [] and uses text fallback
    assert len(items) == 1
    assert items[0][2] == "fallback"


# ── validate_nexara_response ─────────────────────────────────────────────


def test_validate_nexara_valid_response() -> None:
    response = {
        "text": "Hello",
        "segments": [{"speaker": "A", "start": 0.0, "end": 1.0, "text": "Hello"}]
    }
    err_type, err_msg = nex.validate_nexara_response(response)
    assert err_type is None
    assert err_msg is None


def test_validate_nexara_error_field() -> None:
    response = {"error": "Invalid API key"}
    err_type, err_msg = nex.validate_nexara_response(response)
    assert err_type == "api_error"
    assert "Invalid API key" in err_msg


def test_validate_nexara_not_dict() -> None:
    err_type, err_msg = nex.validate_nexara_response("not_json")
    assert err_type == "invalid_response"


def test_validate_nexara_empty_result() -> None:
    response = {"segments": [], "text": ""}
    err_type, err_msg = nex.validate_nexara_response(response)
    assert err_type == "empty_result"


# ── Transcribe retry logic (mocked) ──────────────────────────────────────


class _DummyResponse:
    def __init__(self, status_code: int, payload: bytes) -> None:
        self._status = status_code
        self._raw = payload

    def __enter__(self) -> "_DummyResponse":
        return self

    def __exit__(self, *args) -> bool:
        return False

    def getcode(self) -> int:
        return self._status

    def read(self) -> bytes:
        return self._raw


def test_transcribe_retries_on_transient_error(tmp_path: Path) -> None:
    audio_file = tmp_path / "test.wav"
    audio_file.write_bytes(b"\x00" * 100)

    attempts: list[int] = []
    sleeps: list[float] = []

    def fake_urlopen(req, timeout, context):  # noqa: ANN001
        del req, timeout, context
        attempts.append(1)
        if len(attempts) < 3:
            raise urllib.error.URLError(ConnectionError("connection refused"))
        return _DummyResponse(200, json.dumps({
            "text": "hello",
            "segments": [{"speaker": "speaker_0", "start": 0.0, "end": 1.0, "text": "hello"}],
        }).encode())

    cfg = nex.NexaraConfig(api_key="k", timeout_seconds=30.0, max_retries=3)

    import unittest.mock

    with unittest.mock.patch.object(nex.urllib.request, "urlopen", fake_urlopen):
        with unittest.mock.patch.object(nex.time, "sleep", lambda s: sleeps.append(s)):
            result = nex.transcribe("k", str(audio_file), cfg)

    assert len(attempts) == 3
    assert len(sleeps) == 2  # 2 retries = 2 sleeps
    assert "text" in result


def test_transcribe_does_not_retry_auth_error(tmp_path: Path) -> None:
    audio_file = tmp_path / "test.wav"
    audio_file.write_bytes(b"\x00" * 100)

    attempts: list[int] = []

    def fake_urlopen(req, timeout, context):  # noqa: ANN001
        del req, timeout, context
        attempts.append(1)
        raise urllib.error.HTTPError(
            url="https://api.nexara.ru", code=401, msg="Unauthorized", hdrs=None, fp=None
        )

    cfg = nex.NexaraConfig(api_key="k", timeout_seconds=30.0, max_retries=3)

    import unittest.mock

    with unittest.mock.patch.object(nex.urllib.request, "urlopen", fake_urlopen):
        with pytest.raises(RuntimeError, match="401"):
            nex.transcribe("k", str(audio_file), cfg)

    assert len(attempts) == 1  # No retries for auth error
