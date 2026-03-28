from __future__ import annotations

import json
import urllib.error

import pytest

import llm_cloud_eval as cloud


class _DummyResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self._raw = json.dumps(payload).encode("utf-8")

    def __enter__(self) -> _DummyResponse:
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def read(self) -> bytes:
        return self._raw


def test_call_yandex_messages_retries_temporary_timeout(monkeypatch) -> None:
    calls: list[float] = []
    sleeps: list[float] = []
    logs: list[str] = []

    def fake_urlopen(req, timeout, context):  # noqa: ANN001
        del req, context
        calls.append(float(timeout))
        if len(calls) == 1:
            raise urllib.error.URLError(TimeoutError("The handshake operation timed out"))
        return _DummyResponse({"choices": [{"message": {"content": "ok"}}]})

    monkeypatch.setattr(cloud.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(cloud.time, "sleep", lambda seconds: sleeps.append(float(seconds)))

    result = cloud._call_yandex_messages(
        system_prompt="sys",
        user_content="user",
        cfg=cloud.YandexCloudConfig(api_key="token", folder_id="folder", timeout_seconds=60.0),
        log=logs.append,
    )

    assert result == "ok"
    assert calls == [60.0, 75.0]
    assert sleeps == [1.0]
    assert any("Повтор 2/3" in line for line in logs)


def test_call_yandex_messages_does_not_retry_bad_request(monkeypatch) -> None:
    attempts = 0

    def fake_urlopen(req, timeout, context):  # noqa: ANN001
        del req, timeout, context
        nonlocal attempts
        attempts += 1
        raise urllib.error.HTTPError(
            url="https://example.test",
            code=400,
            msg="Bad Request",
            hdrs=None,
            fp=None,
        )

    monkeypatch.setattr(cloud.urllib.request, "urlopen", fake_urlopen)

    with pytest.raises(urllib.error.HTTPError):
        cloud._call_yandex_messages(
            system_prompt="sys",
            user_content="user",
            cfg=cloud.YandexCloudConfig(api_key="token", folder_id="folder", timeout_seconds=60.0),
        )

    assert attempts == 1
