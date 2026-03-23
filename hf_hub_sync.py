"""Синхронизация и принудительный онлайн-режим Hugging Face Hub.

Важно: в ``huggingface_hub.constants`` флаг ``HF_HUB_OFFLINE`` выставляется **один раз при импорте**
модуля и **не перечитывается** из ``os.environ``. Изменение только переменных окружения недостаточно —
нужно патчить ``huggingface_hub.constants.HF_HUB_OFFLINE`` вручную.
"""

from __future__ import annotations

import os
import threading
from contextlib import contextmanager

# Прогрев ECAPA и загрузка faster-whisper с HF не должны одновременно трогать офлайн-флаг.
HF_HUB_DOWNLOAD_LOCK = threading.Lock()


@contextmanager
def force_huggingface_hub_online():
    """Временно разрешить HTTP к huggingface.co (скачивание моделей faster-whisper и т.д.)."""
    prev_hf = os.environ.get("HF_HUB_OFFLINE")
    prev_tf = os.environ.get("TRANSFORMERS_OFFLINE")
    prev_const: bool | None = None
    try:
        os.environ.pop("HF_HUB_OFFLINE", None)
        os.environ.pop("TRANSFORMERS_OFFLINE", None)
        os.environ["HF_HUB_OFFLINE"] = "0"
        try:
            import huggingface_hub.constants as hfc

            prev_const = hfc.HF_HUB_OFFLINE
            hfc.HF_HUB_OFFLINE = False
        except Exception:
            pass
        yield
    finally:
        if prev_const is not None:
            try:
                import huggingface_hub.constants as hfc

                hfc.HF_HUB_OFFLINE = prev_const
            except Exception:
                pass
        if prev_hf is None:
            os.environ.pop("HF_HUB_OFFLINE", None)
        else:
            os.environ["HF_HUB_OFFLINE"] = prev_hf
        if prev_tf is None:
            os.environ.pop("TRANSFORMERS_OFFLINE", None)
        else:
            os.environ["TRANSFORMERS_OFFLINE"] = prev_tf


def snapshot_huggingface_hub_offline_flag() -> bool | None:
    """Текущее значение ``constants.HF_HUB_OFFLINE`` или None, если модуль недоступен."""
    try:
        import huggingface_hub.constants as hfc

        return bool(hfc.HF_HUB_OFFLINE)
    except Exception:
        return None


def set_huggingface_hub_offline_flag(value: bool) -> None:
    try:
        import huggingface_hub.constants as hfc

        hfc.HF_HUB_OFFLINE = bool(value)
    except Exception:
        pass


def recompute_huggingface_hub_offline_from_env() -> None:
    """Синхронизировать ``constants.HF_HUB_OFFLINE`` с текущими переменными окружения."""
    try:
        import huggingface_hub.constants as hfc
        from huggingface_hub.constants import _is_true

        hfc.HF_HUB_OFFLINE = _is_true(
            os.environ.get("HF_HUB_OFFLINE") or os.environ.get("TRANSFORMERS_OFFLINE")
        )
    except Exception:
        pass
