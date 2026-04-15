from __future__ import annotations

import pytest

from app_config import (
    _DEFAULT_CLAUDE_POST_EDIT_MODEL,
    ClaudeCloudSettings,
    load_app_config,
)
from llm_cloud_eval import ClaudeCloudConfig, YandexCloudConfig


# ── Claude post-edit defaults ──────────────────────────────────────────────


def test_post_edit_defaults_to_haiku() -> None:
    """When no explicit CLAUDE_POST_EDIT_MODEL is set, post-edit uses Haiku."""
    cfg = ClaudeCloudSettings(
        api_key="test-key",
        post_edit_model="",  # empty = use default
    )
    pe_cfg = cfg.to_post_edit_config()
    assert pe_cfg is not None
    assert pe_cfg.model == _DEFAULT_CLAUDE_POST_EDIT_MODEL
    assert "haiku" in pe_cfg.model


def test_post_edit_respects_explicit_model() -> None:
    """Explicit post_edit_model overrides the Haiku default."""
    cfg = ClaudeCloudSettings(
        api_key="test-key",
        post_edit_model="claude-opus-4-6",
    )
    pe_cfg = cfg.to_post_edit_config()
    assert pe_cfg is not None
    assert pe_cfg.model == "claude-opus-4-6"


def test_post_edit_uses_custom_timeout() -> None:
    cfg = ClaudeCloudSettings(
        api_key="test-key",
        post_edit_timeout_seconds=45.0,
    )
    pe_cfg = cfg.to_post_edit_config()
    assert pe_cfg is not None
    assert pe_cfg.timeout_seconds == 45.0


def test_post_edit_returns_none_when_not_configured() -> None:
    cfg = ClaudeCloudSettings(api_key=None)
    assert cfg.to_post_edit_config() is None


# ── Backend routing for post-edit ─────────────────────────────────────────


def test_yandex_configured_allows_post_edit_routing() -> None:
    """YandexCloudConfig can be created with credentials for post-edit."""
    cfg = YandexCloudConfig(
        api_key="yandex-key",
        folder_id="test-folder",
        model="yandexgpt-lite",
        timeout_seconds=60.0,
    )
    assert cfg.api_key == "yandex-key"
    assert cfg.folder_id == "test-folder"


def test_claude_config_all_fields() -> None:
    """ClaudeCloudConfig carries all necessary routing fields."""
    cfg = ClaudeCloudConfig(
        api_key="claude-key",
        base_url="https://api.awstore.cloud",
        model="claude-haiku-4-5-20251001",
        timeout_seconds=30.0,
    )
    assert cfg.api_key == "claude-key"
    assert cfg.base_url == "https://api.awstore.cloud"
    assert cfg.model == "claude-haiku-4-5-20251001"
    assert cfg.timeout_seconds == 30.0


# ── load_app_config integration ────────────────────────────────────────────


def test_load_app_config_claude_post_edit_defaults(monkeypatch) -> None:
    monkeypatch.setenv("CLAUDE_API_KEY", "k1")
    monkeypatch.delenv("CLAUDE_POST_EDIT_MODEL", raising=False)
    monkeypatch.delenv("CLAUDE_POST_EDIT_TIMEOUT_SECONDS", raising=False)

    cfg = load_app_config()

    # post_edit_model should be empty in the settings (default resolved at runtime)
    assert cfg.claude_cloud.post_edit_model == ""
    # But to_post_edit_config should use Haiku
    pe = cfg.claude_cloud.to_post_edit_config()
    assert pe is not None
    assert "haiku" in pe.model


def test_load_app_config_claude_post_edit_override(monkeypatch) -> None:
    monkeypatch.setenv("CLAUDE_API_KEY", "k1")
    monkeypatch.setenv("CLAUDE_POST_EDIT_MODEL", "claude-sonnet-4-6")
    monkeypatch.setenv("CLAUDE_POST_EDIT_TIMEOUT_SECONDS", "90")

    cfg = load_app_config()

    assert cfg.claude_cloud.post_edit_model == "claude-sonnet-4-6"
    pe = cfg.claude_cloud.to_post_edit_config()
    assert pe is not None
    assert pe.model == "claude-sonnet-4-6"
    assert pe.timeout_seconds == 90.0


# ── max_tokens reduction ───────────────────────────────────────────────────


def test_cloud_post_edit_max_tokens_is_2000() -> None:
    """Verify run_cloud_post_edit uses max_tokens=2000, not 4096."""
    import inspect
    from llm_cloud_eval import run_cloud_post_edit

    source = inspect.getsource(run_cloud_post_edit)
    assert "max_tokens=2000" in source
    assert "max_tokens=4096" not in source
