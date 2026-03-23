#!/usr/bin/env python3
"""
Патч file_uploader в установленном Streamlit (RU + 5 файлов на страницу).

Запуск из корня проекта:
  python scripts/patch_streamlit_file_uploader_ru.py

Логика совпадает с автопатчем при `streamlit run web_gui.py` (см. streamlit_file_uploader_patch.py).
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from streamlit_file_uploader_patch import apply_streamlit_file_uploader_localization_patch  # noqa: E402


def main() -> int:
    n = apply_streamlit_file_uploader_localization_patch()
    if n:
        print(f"Обновлено файлов: {n}")
    else:
        print("Изменений не потребовалось (уже применено или чанк не найден).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
