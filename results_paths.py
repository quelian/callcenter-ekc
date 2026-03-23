"""
Каталог для сохранения отчётов по диалогам (txt и т.п.).

Папка создаётся автоматически в корне проекта.
"""

from __future__ import annotations

from pathlib import Path

RESULTS_DIR_NAME = "результаты"


def project_root() -> Path:
    return Path(__file__).resolve().parent


def ensure_results_dir() -> Path:
    """Папка ``результаты`` в корне репозитория; создаётся при необходимости."""
    d = project_root() / RESULTS_DIR_NAME
    d.mkdir(parents=True, exist_ok=True)
    return d
