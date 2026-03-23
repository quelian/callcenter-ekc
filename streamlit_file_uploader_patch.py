"""
Патчит фронтенд Streamlit в **текущем** интерпретаторе: виджет file_uploader — по 5 файлов
на страницу и русские подписи. Вызывается при старте web_gui.py.

Ищет любой чанк `index.*.js`, где есть разметка загрузчика (stFileUploaderPagination и т.д.).
"""
from __future__ import annotations

from pathlib import Path


def _patch_one_file(text: str) -> str | None:
    """Возвращает новый текст, если были замены; иначе None."""
    if "stFileUploaderPagination" not in text:
        return None
    new = text
    if "pageSize:3" in new:
        new = new.replace("pageSize:3", "pageSize:5")
    pairs = [
        ("F=()=>`Limit ${oe(o,P.Byte,0)} per file`", "F=()=>`Не более ${oe(o,P.Byte,0)} на файл`"),
        ('["Drag and drop ",w()," here"]', '["Перетащите ",w()," сюда"]'),
        ('w=()=>i?"directories":e?"files":"file"', 'w=()=>i?"папки":e?"файлы":"файл"'),
        ('children:c?"Browse directories":"Browse files"', 'children:c?"Выбрать папки":"Выбрать файлы"'),
        ("`Showing page ${e} of ${t}`", "`Страница ${e} из ${t}`"),
    ]
    for old, rep in pairs:
        if old in new:
            new = new.replace(old, rep, 1)
    return new if new != text else None


def apply_streamlit_file_uploader_localization_patch() -> int:
    """
    Проходит по `streamlit/static/static/js/index.*.js` и применяет замены.
    Возвращает число изменённых файлов.
    """
    try:
        import streamlit as st_mod

        js_dir = Path(st_mod.__file__).resolve().parent / "static" / "static" / "js"
    except Exception:
        return 0
    if not js_dir.is_dir():
        return 0
    changed = 0
    for fpath in sorted(js_dir.glob("index.*.js")):
        try:
            text = fpath.read_text(encoding="utf-8")
        except OSError:
            continue
        updated = _patch_one_file(text)
        if updated is not None:
            try:
                fpath.write_text(updated, encoding="utf-8")
                changed += 1
            except OSError:
                pass
    return changed
