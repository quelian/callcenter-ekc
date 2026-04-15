from __future__ import annotations

from app_config import load_app_config


def test_load_app_config_has_no_admin_default(monkeypatch) -> None:
    monkeypatch.delenv("ADMIN_PANEL_PASSWORD", raising=False)
    monkeypatch.delenv("GOOGLE_SHEETS_CREDENTIALS_JSON", raising=False)
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
    monkeypatch.delenv("GOOGLE_SHEETS_SPREADSHEET_ID", raising=False)

    cfg = load_app_config()

    assert cfg.security.admin_panel_password is None
    assert not cfg.security.admin_panel_enabled
    assert not cfg.google_sheets.configured


def test_load_app_config_clamps_yandex_timeout(monkeypatch) -> None:
    monkeypatch.setenv("YANDEX_API_KEY", "token")
    monkeypatch.setenv("YANDEX_FOLDER_ID", "folder")
    monkeypatch.setenv("YANDEX_CLOUD_TIMEOUT_SECONDS", "999")

    cfg = load_app_config()

    assert cfg.yandex_cloud.configured
    assert cfg.yandex_cloud.timeout_seconds == 300.0


def test_load_app_config_keeps_custom_yandex_model(monkeypatch) -> None:
    monkeypatch.setenv("YANDEX_API_KEY", "token")
    monkeypatch.setenv("YANDEX_FOLDER_ID", "folder")
    monkeypatch.setenv("YANDEX_CLOUD_MODEL", "aliceai-llm/latest")

    cfg = load_app_config()

    assert cfg.yandex_cloud.configured
    assert cfg.yandex_cloud.model == "aliceai-llm/latest"


def test_load_app_config_yandex_not_configured_without_key(monkeypatch) -> None:
    monkeypatch.delenv("YANDEX_API_KEY", raising=False)
    monkeypatch.delenv("YANDEX_FOLDER_ID", raising=False)

    cfg = load_app_config()
    assert not cfg.yandex_cloud.configured


def test_load_app_config_claude_configured_with_key(monkeypatch) -> None:
    monkeypatch.setenv("CLAUDE_API_KEY", "sk-test123")

    cfg = load_app_config()
    assert cfg.claude_cloud.configured


def test_load_app_config_claude_not_configured_without_key(monkeypatch) -> None:
    monkeypatch.delenv("CLAUDE_API_KEY", raising=False)

    cfg = load_app_config()
    assert not cfg.claude_cloud.configured


def test_load_app_config_assemblyai_configured(monkeypatch) -> None:
    monkeypatch.setenv("ASSEMBLYAI_API_KEY", "key")

    cfg = load_app_config()
    assert cfg.assemblyai.configured


def test_load_app_config_nexara_configured(monkeypatch) -> None:
    monkeypatch.setenv("NEXARA_API_KEY", "key")

    cfg = load_app_config()
    assert cfg.nexara.configured


def test_load_app_config_admin_enabled_when_password_set(monkeypatch) -> None:
    monkeypatch.setenv("ADMIN_PANEL_PASSWORD", "secret")

    cfg = load_app_config()
    assert cfg.security.admin_panel_enabled
