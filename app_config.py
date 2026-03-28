from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from results_paths import project_root

if TYPE_CHECKING:
    from llm_cloud_eval import YandexCloudConfig

_DEFAULT_YANDEX_MODEL = "yandexgpt-lite"


def load_app_environment(root: Path | None = None) -> None:
    """Load .env from the project root when python-dotenv is available."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return

    env_path = (root or project_root()) / ".env"
    load_dotenv(env_path)
    load_dotenv()


def _env_str(key: str) -> str:
    return (os.environ.get(key) or "").strip()


def _env_bool(key: str, default: bool = False) -> bool:
    raw = _env_str(key).lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def _env_float(key: str, default: float, *, minimum: float | None = None, maximum: float | None = None) -> float:
    try:
        value = float(_env_str(key) or default)
    except ValueError:
        value = default
    if minimum is not None:
        value = max(minimum, value)
    if maximum is not None:
        value = min(maximum, value)
    return value


@dataclass(frozen=True)
class SecuritySettings:
    admin_panel_password: str | None

    @property
    def admin_panel_enabled(self) -> bool:
        return bool(self.admin_panel_password)


@dataclass(frozen=True)
class GoogleSheetsSettings:
    credentials_json: str | None
    application_credentials: str | None
    spreadsheet_id: str | None
    worksheet: str | None
    has_header: bool = True
    disabled: bool = False

    @property
    def credentials_hint(self) -> str | None:
        return self.credentials_json or self.application_credentials

    @property
    def configured(self) -> bool:
        return bool(self.credentials_hint and self.spreadsheet_id and not self.disabled)

    @property
    def spreadsheet_url(self) -> str | None:
        if not self.spreadsheet_id:
            return None
        return f"https://docs.google.com/spreadsheets/d/{self.spreadsheet_id}/edit"


@dataclass(frozen=True)
class YandexCloudSettings:
    api_key: str | None
    folder_id: str | None
    model: str = _DEFAULT_YANDEX_MODEL
    timeout_seconds: float = 60.0

    @property
    def configured(self) -> bool:
        return bool(self.api_key and self.folder_id)

    def to_runtime_config(self) -> "YandexCloudConfig | None":
        if not self.configured:
            return None
        from llm_cloud_eval import YandexCloudConfig

        return YandexCloudConfig(
            api_key=self.api_key or "",
            folder_id=self.folder_id or "",
            model=self.model,
            timeout_seconds=self.timeout_seconds,
        )


@dataclass(frozen=True)
class AppConfig:
    security: SecuritySettings
    google_sheets: GoogleSheetsSettings
    yandex_cloud: YandexCloudSettings


def load_app_config() -> AppConfig:
    return AppConfig(
        security=SecuritySettings(
            admin_panel_password=_env_str("ADMIN_PANEL_PASSWORD") or None,
        ),
        google_sheets=GoogleSheetsSettings(
            credentials_json=_env_str("GOOGLE_SHEETS_CREDENTIALS_JSON") or None,
            application_credentials=_env_str("GOOGLE_APPLICATION_CREDENTIALS") or None,
            spreadsheet_id=_env_str("GOOGLE_SHEETS_SPREADSHEET_ID") or None,
            worksheet=_env_str("GOOGLE_SHEETS_WORKSHEET") or None,
            has_header=_env_bool("GOOGLE_SHEETS_HAS_HEADER", default=True),
            disabled=_env_bool("GOOGLE_SHEETS_DISABLE", default=False),
        ),
        yandex_cloud=YandexCloudSettings(
            api_key=_env_str("YANDEX_API_KEY") or None,
            folder_id=_env_str("YANDEX_FOLDER_ID") or None,
            model=_env_str("YANDEX_CLOUD_MODEL") or _DEFAULT_YANDEX_MODEL,
            timeout_seconds=_env_float(
                "YANDEX_CLOUD_TIMEOUT_SECONDS",
                60.0,
                minimum=15.0,
                maximum=300.0,
            ),
        ),
    )
