from __future__ import annotations

import json
import urllib.error

import pytest

import assemblyai_backend as aai


class _DummyResponse:
    def __init__(self, payload: dict) -> None:
        self._raw = json.dumps(payload).encode("utf-8")

    def __enter__(self) -> _DummyResponse:
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def read(self) -> bytes:
        return self._raw


# ── map_assemblyai_to_voice_items ─────────────────────────────────────────────


def test_map_utterances_to_voice_items() -> None:
    response = {
        "utterances": [
            {"speaker": "A", "text": "Здравствуйте", "start": 0, "end": 1500},
            {"speaker": "B", "text": "Добрый день", "start": 2000, "end": 3500},
            {"speaker": "A", "text": "Чем могу помочь?", "start": 4000, "end": 5500},
        ]
    }
    items = aai.map_assemblyai_to_voice_items(response)
    assert len(items) == 3
    assert items[0] == (0.0, 1.5, "Здравствуйте", "SPK_A")
    assert items[1] == (2.0, 3.5, "Добрый день", "SPK_B")
    assert items[2] == (4.0, 5.5, "Чем могу помочь?", "SPK_A")


def test_map_utterances_skips_empty_text() -> None:
    response = {
        "utterances": [
            {"speaker": "A", "text": "", "start": 0, "end": 1000},
            {"speaker": "B", "text": "Привет", "start": 2000, "end": 3000},
            {"speaker": "A", "text": "   ", "start": 4000, "end": 5000},
        ]
    }
    items = aai.map_assemblyai_to_voice_items(response)
    assert len(items) == 1
    assert items[0] == (2.0, 3.0, "Привет", "SPK_B")


def test_map_utterances_empty_list() -> None:
    response = {"utterances": []}
    items = aai.map_assemblyai_to_voice_items(response)
    assert items == []


def test_map_fallback_to_words_when_no_utterances() -> None:
    response = {
        "words": [
            {"text": "hello", "start": 0, "end": 500},
            {"text": "world", "start": 600, "end": 1200},
        ]
    }
    items = aai.map_assemblyai_to_voice_items(response)
    assert len(items) == 1
    start, end, text, speaker = items[0]
    assert start == 0.0
    assert end == 1.2
    assert text == "hello world"
    assert speaker == "SPK_A"


def test_map_returns_empty_when_no_utterances_and_no_words() -> None:
    response = {}
    items = aai.map_assemblyai_to_voice_items(response)
    assert items == []


# ── extract_entities ──────────────────────────────────────────────────────────


def test_extract_entities() -> None:
    response = {
        "entities": [
            {"entity_type": "PERSON", "text": "Иван Петров"},
            {"entity_type": "ORGANIZATION", "text": "Сбербанк"},
        ]
    }
    entities = aai.extract_entities(response)
    assert len(entities) == 2
    assert entities[0]["entity_type"] == "PERSON"


def test_extract_entities_empty() -> None:
    assert aai.extract_entities({}) == []


# ── extract_content_moderation ────────────────────────────────────────────────


def test_extract_content_moderation_with_results() -> None:
    response = {
        "content_moderation": {
            "results": [
                {"type": "profanity", "text": "...", "start": 1000, "end": 2000}
            ]
        }
    }
    moderation = aai.extract_content_moderation(response)
    assert moderation is not None
    assert "results" in moderation


def test_extract_content_moderation_no_results() -> None:
    response = {"content_moderation": {}}
    assert aai.extract_content_moderation(response) is None


def test_extract_content_moderation_missing() -> None:
    assert aai.extract_content_moderation({}) is None


# ── extract_applicant_name_from_entities ──────────────────────────────────────


def test_applicant_name_from_person_entity() -> None:
    entities = [
        {"entity_type": "PERSON", "text": "Алексей Смирнов"},
        {"entity_type": "PERSON", "text": "Оператор"},
    ]
    name = aai.extract_applicant_name_from_entities(entities)
    assert name is not None
    assert "алексей" in name.lower()


def test_applicant_name_skips_noise_words() -> None:
    entities = [
        {"entity_type": "PERSON", "text": "Оператор"},
        {"entity_type": "PERSON", "text": "Мария Иванова"},
    ]
    name = aai.extract_applicant_name_from_entities(entities)
    assert name is not None
    assert "оператор" not in name.lower()
    assert "мария" in name.lower()


def test_applicant_name_no_person_entities() -> None:
    entities = [
        {"entity_type": "ORGANIZATION", "text": "МФЦ"},
    ]
    assert aai.extract_applicant_name_from_entities(entities) is None


def test_applicant_name_empty_entities() -> None:
    entities = [
        {"entity_type": "PERSON", "text": ""},
        {"entity_type": "PERSON", "text": "   "},
    ]
    assert aai.extract_applicant_name_from_entities(entities) is None


# ── build_custom_spelling_for_operators ───────────────────────────────────────


def test_build_custom_spelling() -> None:
    names = ["Егор", "Даша"]
    spelling = aai.build_custom_spelling_for_operators(names)
    assert len(spelling) == 2
    assert spelling[0] == {"to": "Егор", "from": ["егор"]}
    assert spelling[1] == {"to": "Даша", "from": ["даша"]}


def test_build_custom_spelling_empty() -> None:
    assert aai.build_custom_spelling_for_operators([]) == []


# ── AssemblyAIConfig defaults ─────────────────────────────────────────────────


def test_config_defaults() -> None:
    cfg = aai.AssemblyAIConfig(api_key="test-key")
    assert cfg.language == "ru"
    assert cfg.model == "universal-2"
    assert cfg.speaker_labels is True
    assert cfg.entity_detection is True
    assert cfg.content_moderation is True
    assert cfg.timeout_seconds == 300.0
    assert cfg.polling_interval_seconds == 3.0
    assert cfg.custom_spelling == []


# ── HTTP error handling ───────────────────────────────────────────────────────


def test_make_request_http_error(monkeypatch) -> None:
    def fake_urlopen(req, timeout, context):
        raise urllib.error.HTTPError(
            url="https://api.assemblyai.com/v2/upload",
            code=401,
            msg="Unauthorized",
            hdrs=None,
            fp=None,
        )

    monkeypatch.setattr(aai.urllib.request, "urlopen", fake_urlopen)

    with pytest.raises(RuntimeError, match="401"):
        aai._make_request("https://api.assemblyai.com/v2/upload", "POST", "bad-key")


def test_make_request_connection_error(monkeypatch) -> None:
    def fake_urlopen(req, timeout, context):
        raise urllib.error.URLError("Connection refused")

    monkeypatch.setattr(aai.urllib.request, "urlopen", fake_urlopen)

    with pytest.raises(RuntimeError, match="Connection refused"):
        aai._make_request("https://api.assemblyai.com/v2/upload", "POST", "key")


# ── poll_transcription status handling ────────────────────────────────────────


def test_poll_completes_immediately(monkeypatch) -> None:
    calls: list[int] = []
    sleeps: list[float] = []

    def fake_urlopen(req, timeout, context):
        del req, timeout, context
        calls.append(1)
        return _DummyResponse({
            "id": "test-123",
            "status": "completed",
            "utterances": [
                {"speaker": "A", "text": "Hello", "start": 0, "end": 1000}
            ],
        })

    monkeypatch.setattr(aai.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(aai.time, "sleep", lambda s: sleeps.append(s))

    cfg = aai.AssemblyAIConfig(api_key="key")
    result = aai.poll_transcription("key", "test-123", cfg)
    assert result["status"] == "completed"
    assert len(calls) == 1
    assert sleeps == []


def test_poll_retries_then_completes(monkeypatch) -> None:
    attempts: list[int] = []
    sleeps: list[float] = []

    def fake_urlopen(req, timeout, context):
        del req, timeout, context
        attempts.append(1)
        if len(attempts) <= 2:
            status = "queued" if len(attempts) == 1 else "processing"
        else:
            status = "completed"
        return _DummyResponse({"id": "test-123", "status": status, "utterances": []})

    monkeypatch.setattr(aai.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(aai.time, "sleep", lambda s: sleeps.append(s))

    cfg = aai.AssemblyAIConfig(api_key="key", polling_interval_seconds=0.1)
    result = aai.poll_transcription("key", "test-123", cfg)
    assert result["status"] == "completed"
    assert len(attempts) == 3
    assert len(sleeps) == 2


def test_poll_error_status(monkeypatch) -> None:
    def fake_urlopen(req, timeout, context):
        del req, timeout, context
        return _DummyResponse({
            "id": "test-123",
            "status": "error",
            "error": "Audio file too short",
        })

    monkeypatch.setattr(aai.urllib.request, "urlopen", fake_urlopen)

    cfg = aai.AssemblyAIConfig(api_key="key")
    with pytest.raises(RuntimeError, match="Audio file too short"):
        aai.poll_transcription("key", "test-123", cfg)
