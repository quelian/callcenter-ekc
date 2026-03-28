from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Mapping

from results_paths import project_root

_GOOGLE_SHEET_URL_RE = re.compile(
    r"docs\.google\.com/spreadsheets/d/([a-zA-Z0-9_-]+)",
    re.IGNORECASE,
)


def env_file_path(root: Path | None = None) -> Path:
    return (root or project_root()) / ".env"


def dotenv_escape_value(val: str) -> str:
    """Value for a KEY=... line in .env (without the key name)."""
    val = val.replace("\n", " ").strip()
    if re.search(r'[\s#"\'=]', val) or val != val.strip():
        escaped = val.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    return val


def merge_env_file(updates: Mapping[str, str], *, path: Path | None = None) -> None:
    """Update or append keys in .env while preserving unrelated lines and comments."""
    target = path or env_file_path()
    keys_written: set[str] = set()
    out_lines: list[str] = []
    if target.exists():
        for line in target.read_text(encoding="utf-8").splitlines():
            raw = line
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                out_lines.append(raw)
                continue
            if "=" in stripped:
                key = stripped.split("=", 1)[0].strip()
                if key in updates:
                    out_lines.append(f"{key}={dotenv_escape_value(updates[key])}")
                    keys_written.add(key)
                    continue
            out_lines.append(raw)
    for key, value in updates.items():
        if key not in keys_written:
            out_lines.append(f"{key}={dotenv_escape_value(value)}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    for key, value in updates.items():
        os.environ[key] = value


def normalize_google_spreadsheet_id(text: str) -> str:
    """Extract spreadsheet ID from a Google Sheets URL or return the raw ID."""
    candidate = (text or "").strip()
    if not candidate:
        return ""
    match = _GOOGLE_SHEET_URL_RE.search(candidate)
    if match:
        return match.group(1)
    return candidate
