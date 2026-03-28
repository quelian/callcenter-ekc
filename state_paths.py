from __future__ import annotations

from pathlib import Path

from results_paths import project_root

APP_STATE_DIR_NAME = ".callqa_state"


def ensure_app_state_dir() -> Path:
    path = project_root() / APP_STATE_DIR_NAME
    path.mkdir(parents=True, exist_ok=True)
    return path
