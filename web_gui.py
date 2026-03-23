from __future__ import annotations

# До streamlit и любого импорта speechbrain: torchaudio 2.2+ и фильтр шумных варнингов SB.
from speechbrain_compat import apply_speechbrain_environment

apply_speechbrain_environment()

import io
import os
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
import tempfile
import threading
import time
from typing import Callable

import streamlit as st

_APP_DIR = Path(__file__).resolve().parent

# Версия UI — дублируйте при релизе в CHANGELOG.md
APP_VERSION_LABEL = "Beta 1.0"
APP_VERSION_DATE = "23.03.2026"
try:
    from dotenv import load_dotenv

    load_dotenv(_APP_DIR / ".env")
    load_dotenv()
except ImportError:
    pass

from operator_staff import OPERATOR_CANONICAL_NAMES
from results_paths import ensure_results_dir
from streamlit_file_uploader_patch import apply_streamlit_file_uploader_localization_patch
from transcription import (
    CallQualityEvaluator,
    QualityEvaluation,
    Transcriber,
    TranscriptionResult,
    build_report,
    resolve_ultima_whisper_model,
)

_MANUAL_WAIT_INPUT_RE = re.compile(
    r"^\s*(\d{1,3})\s*[:-]\s*(\d{1,2})\s*$"
)

# Пароль админ-панели (ключи Yandex + Google Таблица). Переопределение: ADMIN_PANEL_PASSWORD в .env
_DEFAULT_ADMIN_PANEL_PASSWORD = "9632"

_GOOGLE_SHEET_URL_RE = re.compile(
    r"docs\.google\.com/spreadsheets/d/([a-zA-Z0-9_-]+)",
    re.IGNORECASE,
)


def _admin_panel_password() -> str:
    return (os.environ.get("ADMIN_PANEL_PASSWORD") or _DEFAULT_ADMIN_PANEL_PASSWORD).strip()


def _env_file_path() -> Path:
    return _APP_DIR / ".env"


def _dotenv_escape_value(val: str) -> str:
    """Значение для строки KEY=... в .env (без имени ключа)."""
    val = val.replace("\n", " ").strip()
    if re.search(r'[\s#"\'=]', val) or val != val.strip():
        escaped = val.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    return val


def merge_env_file(updates: dict[str, str]) -> None:
    """Обновляет или добавляет ключи в .env; прочие строки и комментарии сохраняет."""
    path = _env_file_path()
    keys_written: set[str] = set()
    out_lines: list[str] = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            raw = line
            s = line.strip()
            if not s or s.startswith("#"):
                out_lines.append(raw)
                continue
            if "=" in s:
                k = s.split("=", 1)[0].strip()
                if k in updates:
                    out_lines.append(f"{k}={_dotenv_escape_value(updates[k])}")
                    keys_written.add(k)
                    continue
            out_lines.append(raw)
    for k, v in updates.items():
        if k not in keys_written:
            out_lines.append(f"{k}={_dotenv_escape_value(v)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    for k, v in updates.items():
        os.environ[k] = v


def normalize_google_spreadsheet_id(text: str) -> str:
    """ID из полной ссылки Google Sheets или сырой ID."""
    t = (text or "").strip()
    if not t:
        return ""
    m = _GOOGLE_SHEET_URL_RE.search(t)
    if m:
        return m.group(1)
    return t


def render_admin_panel_sidebar() -> None:
    """Ключ Yandex, Folder ID и таблица Google — только после ввода пароля; сохранение в .env."""
    st.divider()
    with st.expander("🔐 Админ-панель", expanded=False):
        st.caption(
            "Ключ API Яндекса и ссылка/ID Google Таблицы можно менять **только здесь** "
            "(после входа по паролю). Изменения записываются в файл `.env`."
        )

        if not st.session_state.get("admin_panel_unlocked"):
            with st.form("admin_login_form", clear_on_submit=False):
                pwd_in = st.text_input("Пароль администратора", type="password", key="admin_pw_field")
                login = st.form_submit_button("Войти")
            if login:
                if (pwd_in or "").strip() == _admin_panel_password():
                    st.session_state["admin_panel_unlocked"] = True
                    st.rerun()
                else:
                    st.error("Неверный пароль")
            return

        from sheets_export import DEFAULT_SPREADSHEET_ID

        yf_cur = (os.environ.get("YANDEX_FOLDER_ID") or "").strip()
        gs_env = (os.environ.get("GOOGLE_SHEETS_SPREADSHEET_ID") or "").strip()
        sheet_display = gs_env or DEFAULT_SPREADSHEET_ID

        with st.form("admin_save_form"):
            st.markdown("**Yandex AI Studio**")
            new_api = st.text_input(
                "API Key",
                value="",
                type="password",
                placeholder="Оставьте пустым, чтобы не менять текущий ключ",
                help="Новое значение сохранится в `.env` (переменная YANDEX_API_KEY).",
                key="admin_yandex_api",
            )
            new_folder = st.text_input(
                "Folder ID",
                value=yf_cur,
                help="YANDEX_FOLDER_ID в консоли Yandex Cloud.",
                key="admin_yandex_folder",
            )
            st.markdown("**Google Таблица**")
            new_sheet = st.text_input(
                "Ссылка или ID таблицы",
                value=sheet_display,
                help=(
                    "Вставьте полную ссылку вида "
                    "https://docs.google.com/spreadsheets/d/…/edit "
                    f"или только ID. Если в `.env` не задано — сейчас используется ID по умолчанию: `{DEFAULT_SPREADSHEET_ID}`."
                ),
                key="admin_sheet_url",
            )
            save = st.form_submit_button("Сохранить в .env")
        if save:
            updates: dict[str, str] = {}
            api_stripped = (new_api or "").strip()
            if api_stripped:
                updates["YANDEX_API_KEY"] = api_stripped
            folder_stripped = (new_folder or "").strip()
            if folder_stripped:
                updates["YANDEX_FOLDER_ID"] = folder_stripped
            sid = normalize_google_spreadsheet_id(new_sheet)
            if sid:
                updates["GOOGLE_SHEETS_SPREADSHEET_ID"] = sid
            if not updates:
                st.warning("Нечего сохранить: укажите новый API Key и/или проверьте Folder ID и таблицу.")
            else:
                try:
                    merge_env_file(updates)
                    _upd_keys = set(updates)
                    if _upd_keys & {
                        "GOOGLE_SHEETS_SPREADSHEET_ID",
                        "GOOGLE_SHEETS_WORKSHEET",
                        "GOOGLE_SHEETS_CREDENTIALS_JSON",
                        "GOOGLE_APPLICATION_CREDENTIALS",
                    }:
                        try:
                            from sheets_export import invalidate_worksheet_cache

                            invalidate_worksheet_cache()
                        except Exception:
                            pass
                    if _upd_keys & {"YANDEX_API_KEY", "YANDEX_FOLDER_ID"}:
                        st.cache_resource.clear()
                    st.success("Сохранено. Переменные обновлены для текущего сеанса.")
                    st.rerun()
                except OSError as exc:
                    st.error(f"Не удалось записать `.env`: {exc}")

        if st.button("Выйти из админ-панели", key="admin_logout_btn"):
            st.session_state["admin_panel_unlocked"] = False
            st.rerun()


def parse_manual_wait_mm_ss(text: str) -> tuple[int, int] | None:
    """
    Ручной ввод ожидания до ответа оператора.

    Допустимо: пусто → ``0:00``; ``ММ:СС`` или ``ММ-СС`` (как в имени файла);
    одно число — секунды (например ``90`` → 1:30).
    Минуты 0–180, секунды 0–59 (после разбора).
    """
    raw = (text or "").strip()
    if not raw:
        return (0, 0)
    if raw.isdigit():
        total = int(raw)
        if total < 0 or total > 180 * 60 + 59:
            return None
        return (total // 60, total % 60)
    m = _MANUAL_WAIT_INPUT_RE.match(raw)
    if not m:
        return None
    mm = int(m.group(1))
    ss = int(m.group(2))
    if ss >= 60 or mm > 180 or mm < 0:
        return None
    return (mm, ss)


# ── CSS ───────────────────────────────────────────────────────────────────────
_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

/* Запрещаем браузеру (Comet, Chrome Auto Dark Mode) инвертировать тему */
:root {
    color-scheme: light only !important;
}

html, body {
    color-scheme: light only !important;
    background-color: #ffffff !important;
    color: #111827 !important;
}

html, body, [class*="css"], * {
    font-family: 'Inter', -apple-system, sans-serif;
}

/* Принудительный светлый фон — все варианты селекторов для Streamlit 1.40–1.55 */
.stApp,
[data-testid="stApp"],
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
[data-testid="stMainBlockContainer"],
.main,
.main .block-container {
    background-color: #ffffff !important;
    color: #111827 !important;
}

/* Переопределяем CSS-переменные темы Streamlit */
:root, [data-theme="dark"], [data-theme="light"] {
    --background-color: #ffffff !important;
    --secondary-background-color: #f8fafc !important;
    --text-color: #111827 !important;
    --font: 'Inter', sans-serif !important;
}

/* Убрать стандартную шапку и меню Streamlit (Streamlit ≥1.40) */
#MainMenu                          { display: none !important; }
footer                             { display: none !important; }
[data-testid="stHeader"]           { display: none !important; }
[data-testid="stToolbar"]          { display: none !important; }
[data-testid="stDecoration"]       { display: none !important; }
[data-testid="stAppDeployButton"]  { display: none !important; }
.stAppDeployButton                 { display: none !important; }
header[data-testid="stHeader"]     { display: none !important; }

.block-container {
    padding-top: 2rem !important;
    padding-bottom: 3rem !important;
    max-width: 1140px !important;
}

/* ── Шапка ── */
.qa-header {
    display: flex; align-items: center; gap: 14px;
    padding: 0 0 20px 0; border-bottom: 1px solid #f0f0f0; margin-bottom: 24px;
}
.qa-header-icon {
    background: linear-gradient(135deg, #2563eb, #1d4ed8);
    border-radius: 14px; width: 48px; height: 48px;
    display: flex; align-items: center; justify-content: center;
    font-size: 24px; box-shadow: 0 4px 12px rgba(37,99,235,.25);
    flex-shrink: 0;
}
.qa-header-title { font-size: 22px; font-weight: 800; color: #111; line-height: 1.2; }
.qa-header-sub   { font-size: 13px; color: #9ca3af; margin-top: 2px; }

/* ── Сайдбар — принудительно видимый ── */
section[data-testid="stSidebar"],
section[data-testid="stSidebar"] > div:first-child,
section[data-testid="stSidebar"] > div {
    background: #f8fafc !important;
    border-right: 1px solid #e9ecef !important;
    color: #111827 !important;
    transform: translateX(0) !important;
    visibility: visible !important;
    opacity: 1 !important;
    display: block !important;
    pointer-events: auto !important;
    min-width: 244px !important;
    width: 244px !important;
}
/* Гарантируем что кнопка раскрытия сайдбара не перекрывает его */
[data-testid="stExpandSidebarButton"],
[data-testid="stSidebarCollapsedControl"] {
    display: none !important;
}
.sidebar-logo {
    display: flex; align-items: center; gap: 10px;
    padding: 4px 0 16px 0; border-bottom: 1px solid #e5e7eb; margin-bottom: 16px;
}
.sidebar-logo-icon {
    background: linear-gradient(135deg, #2563eb, #1d4ed8);
    border-radius: 10px; width: 36px; height: 36px;
    display: flex; align-items: center; justify-content: center; font-size: 18px;
}
.sidebar-logo-text { font-size: 14px; font-weight: 700; color: #111; }
.sidebar-logo-sub  { font-size: 11px; color: #9ca3af; }
.sidebar-logo-version { font-size: 10px; color: #cbd5e1; margin-top: 2px; }

.ai-connected {
    background: #f0fdf4; border: 1px solid #bbf7d0;
    border-radius: 10px; padding: 10px 14px;
    font-size: 13px; color: #166534; margin-bottom: 12px;
    display: flex; align-items: center; gap: 8px;
}
.ai-disconnected {
    background: #fff7ed; border: 1px solid #fed7aa;
    border-radius: 10px; padding: 10px 14px;
    font-size: 13px; color: #9a3412; margin-bottom: 12px;
    display: flex; align-items: center; gap: 8px;
}

/* ── Шаговый прогресс ── */
.steps-bar {
    display: flex; align-items: center;
    background: #f9fafb; border: 1px solid #f0f0f0;
    border-radius: 14px; padding: 14px 20px; margin: 16px 0; gap: 0;
}
.step {
    display: flex; align-items: center; gap: 8px;
    flex: 1; justify-content: center; position: relative;
}
.step:not(:last-child)::after {
    content: ''; position: absolute; right: 0;
    width: 1px; height: 28px; background: #e5e7eb;
}
.step-dot {
    width: 28px; height: 28px; border-radius: 9999px;
    display: flex; align-items: center; justify-content: center;
    font-size: 12px; font-weight: 700; flex-shrink: 0;
}
.step-dot.done  { background: #dcfce7; color: #16a34a; }
.step-dot.active{ background: #dbeafe; color: #2563eb; animation: pulse 1.5s infinite; }
.step-dot.idle  { background: #f3f4f6; color: #9ca3af; }
.step-label { font-size: 12px; font-weight: 500; }
.step-label.done  { color: #16a34a; }
.step-label.active{ color: #2563eb; }
.step-label.idle  { color: #9ca3af; }

@keyframes pulse {
    0%,100%{ box-shadow: 0 0 0 0 rgba(37,99,235,.3); }
    50%    { box-shadow: 0 0 0 6px rgba(37,99,235,.0); }
}

/* ── Карточки счётчиков ── */
.score-card {
    background: #fff; border: 1px solid #f0f0f0; border-radius: 14px;
    padding: 16px; text-align: center;
    box-shadow: 0 1px 4px rgba(0,0,0,.05);
    transition: box-shadow .2s;
}
.score-card:hover { box-shadow: 0 4px 16px rgba(0,0,0,.09); }
.score-title { font-size: 12px; color: #6b7280; font-weight: 500;
               text-transform: uppercase; letter-spacing: .05em; margin-bottom: 8px; }
.score-value { font-size: 36px; font-weight: 900; line-height: 1; }
.score-sub   { font-size: 13px; color: #9ca3af; margin-top: 2px; }
.score-bar   { height: 5px; border-radius: 9999px; background: #f3f4f6;
               margin-top: 10px; overflow: hidden; }
.score-fill  { height: 100%; border-radius: 9999px; transition: width .4s; }
.score-badge { display:inline-block; font-size:11px; font-weight:600;
               border-radius:6px; padding:2px 8px; margin-top:8px; }

/* ── Summary ── */
.summary-card {
    background: #fff; border: 1px solid #f0f0f0; border-radius: 18px;
    padding: 24px 28px; margin-bottom: 20px;
    box-shadow: 0 2px 8px rgba(0,0,0,.06);
    display: flex; align-items: center; justify-content: space-between;
    flex-wrap: wrap; gap: 20px;
}
.summary-person-label {
    font-size: 11px; color: #9ca3af; text-transform: uppercase;
    letter-spacing: .06em; margin-bottom: 4px;
}
.summary-person-name {
    font-size: 20px; font-weight: 700; color: #111;
}
.summary-score-wrap { text-align: right; }
.summary-score-label {
    font-size: 11px; color: #9ca3af; text-transform: uppercase;
    letter-spacing: .06em; margin-bottom: 4px;
}
.summary-score-val {
    font-size: 54px; font-weight: 900; line-height: 1;
}
.summary-score-denom { font-size: 22px; color: #d1d5db; }

/* ── Чеклист ── */
.checklist-item {
    display: flex; align-items: center; gap: 10px;
    padding: 10px 14px; border-radius: 10px; margin-bottom: 6px;
    font-size: 14px; font-weight: 500;
}
.checklist-item.ok  { background: #f0fdf4; color: #166534; }
.checklist-item.fail{ background: #fef2f2; color: #991b1b; }
.checklist-icon { font-size: 16px; flex-shrink: 0; }

/* ── Плюсы / минусы ── */
.feedback-item {
    border-radius: 12px; padding: 12px 16px; margin-bottom: 8px;
    font-size: 14px; line-height: 1.5;
}
.feedback-item.pos { background: #f0fdf4; border-left: 4px solid #22c55e; color: #166534; }
.feedback-item.neg { background: #fff7ed; border-left: 4px solid #f59e0b; color: #92400e; }

/* ── Транскрипт ── */
.transcript-wrap {
    background: #fafafa; border: 1px solid #f0f0f0; border-radius: 14px;
    padding: 16px; max-height: 480px; overflow-y: auto; font-size: 14px;
}
.tx-line { display: flex; gap: 10px; margin-bottom: 10px; }
.tx-badge {
    flex-shrink: 0; font-size: 10px; font-weight: 700; border-radius: 6px;
    padding: 2px 8px; height: fit-content; margin-top: 2px; letter-spacing: .04em;
    white-space: nowrap;
}
.tx-badge.op  { background: #dbeafe; color: #1d4ed8; }
.tx-badge.ap  { background: #f3f4f6; color: #374151; }
.tx-text { color: #374151; line-height: 1.6; }

/* ── Verdict ── */
.verdict-ok  {
    background: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 12px;
    padding: 14px 18px; color: #166534; font-size: 15px; font-weight: 600;
    display: flex; align-items: center; gap: 10px; margin-bottom: 20px;
}
.verdict-bad {
    background: #fef2f2; border: 1px solid #fecaca; border-radius: 12px;
    padding: 14px 18px; color: #991b1b; font-size: 15px; font-weight: 600;
    display: flex; align-items: center; gap: 10px; margin-bottom: 20px;
}
.verdict-reasons { font-size: 13px; font-weight: 400; color: #b91c1c; margin-top: 4px; }

/* ── Секции ── */
.section-title {
    font-size: 16px; font-weight: 700; color: #111;
    margin: 24px 0 12px 0; display: flex; align-items: center; gap: 8px;
}

/* ── Пакетный режим — прогресс-файл ── */
.batch-file-card {
    background: #f9fafb; border: 1px solid #e5e7eb; border-radius: 10px;
    padding: 10px 14px; margin-bottom: 8px; font-size: 13px; color: #374151;
    display: flex; align-items: center; gap: 10px;
}
.batch-file-card.ok  { border-left: 4px solid #22c55e; }
.batch-file-card.err { border-left: 4px solid #ef4444; }
.batch-review-yes {
    background: #fef2f2; color: #991b1b; font-weight: 600;
    border-radius: 5px; padding: 2px 8px; font-size: 12px;
}
.batch-review-no {
    background: #f0fdf4; color: #166534; font-weight: 600;
    border-radius: 5px; padding: 2px 8px; font-size: 12px;
}
</style>
"""


# ── Helpers ──────────────────────────────────────────────────────────────────

def _score_color(score: int) -> str:
    if score >= 8:
        return "#22c55e"
    if score >= 5:
        return "#f59e0b"
    return "#ef4444"


def _score_label(score: int) -> str:
    if score >= 8:
        return "Хорошо"
    if score >= 5:
        return "Удовл."
    return "Слабо"


def _score_badge_style(score: int) -> str:
    if score >= 8:
        return "background:#dcfce7;color:#16a34a;"
    if score >= 5:
        return "background:#fef9c3;color:#b45309;"
    return "background:#fee2e2;color:#b91c1c;"


def render_header() -> None:
    st.markdown("""
<div class="qa-header">
  <div class="qa-header-icon">📞</div>
  <div>
    <div class="qa-header-title">Анализ звонков</div>
    <div class="qa-header-sub">ДВФУ · Единый контактный центр</div>
  </div>
</div>""", unsafe_allow_html=True)


def render_score_card(col, title: str, score: int, icon: str) -> None:
    color = _score_color(score)
    label = _score_label(score)
    badge_style = _score_badge_style(score)
    pct = score * 10
    with col:
        st.markdown(f"""
<div class="score-card">
  <div class="score-title">{icon} {title}</div>
  <div class="score-value" style="color:{color};">{score}</div>
  <div class="score-sub">/10</div>
  <div class="score-bar">
    <div class="score-fill" style="width:{pct}%; background:{color};"></div>
  </div>
  <div><span class="score-badge" style="{badge_style}">{label}</span></div>
</div>""", unsafe_allow_html=True)


def render_summary(evaluation: QualityEvaluation) -> None:
    color = _score_color(evaluation.total_score)
    op = evaluation.operator_name or "Не определено"
    app = evaluation.applicant_name or "Не определено"
    op_badge = "" if evaluation.operator_in_staff else " ⚠️"
    st.markdown(f"""
<div class="summary-card">
  <div>
    <div class="summary-person-label">Оператор</div>
    <div class="summary-person-name">{op}{op_badge}</div>
  </div>
  <div>
    <div class="summary-person-label">Заявитель</div>
    <div class="summary-person-name">{app}</div>
  </div>
  <div class="summary-score-wrap">
    <div class="summary-score-label">Итоговая оценка</div>
    <div class="summary-score-val" style="color:{color};">
      {evaluation.total_score}<span class="summary-score-denom">/10</span>
    </div>
  </div>
</div>""", unsafe_allow_html=True)


def render_gauge(score: int) -> None:
    try:
        import plotly.graph_objects as go

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score,
            number={"font": {"size": 52, "family": "Inter"}, "suffix": "/10"},
            gauge={
                "axis": {"range": [0, 10], "tickwidth": 1, "tickcolor": "#e5e7eb",
                         "tickfont": {"size": 11}},
                "bar": {"color": _score_color(score), "thickness": 0.28},
                "bgcolor": "#f9fafb",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 5],  "color": "#fee2e2"},
                    {"range": [5, 8],  "color": "#fef9c3"},
                    {"range": [8, 10], "color": "#dcfce7"},
                ],
                "threshold": {
                    "line": {"color": _score_color(score), "width": 3},
                    "thickness": 0.75,
                    "value": score,
                },
            },
        ))
        fig.update_layout(
            height=220, margin={"t": 20, "b": 10, "l": 20, "r": 20},
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font={"family": "Inter"},
        )
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    except ImportError:
        # fallback если plotly не установлен
        st.metric("Итоговая оценка", f"{score}/10")


def render_verdict(evaluation: QualityEvaluation, tone_summary: str) -> None:
    needs_review, reasons = requires_manual_review(evaluation, tone_summary)
    if needs_review:
        reasons_html = f'<div class="verdict-reasons">Причины: {", ".join(reasons)}</div>'
        st.markdown(f"""
<div class="verdict-bad">
  <span style="font-size:22px;">⚠️</span>
  <div><div>Требует ручной проверки</div>{reasons_html}</div>
</div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
<div class="verdict-ok">
  <span style="font-size:22px;">✅</span>
  <div>Автопроверка пройдена — критических нарушений не выявлено</div>
</div>""", unsafe_allow_html=True)


def render_checklist(script_details: list[str]) -> None:
    items_html = []
    for item in script_details:
        label, _, val = item.rpartition(":")
        ok = val.strip().lower().startswith("да")
        css = "ok" if ok else "fail"
        icon = "✅" if ok else "❌"
        # Убираем лишнее в конце (уточнение в скобках оставляем)
        items_html.append(
            f'<div class="checklist-item {css}">'
            f'<span class="checklist-icon">{icon}</span>{label.strip()}'
            f"</div>"
        )
    st.markdown("\n".join(items_html), unsafe_allow_html=True)


def render_feedback(positives: list[str], negatives: list[str]) -> None:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="section-title">💪 Сильные стороны</div>', unsafe_allow_html=True)
        if positives:
            for item in positives[:5]:
                st.markdown(f'<div class="feedback-item pos">✦ {item}</div>',
                            unsafe_allow_html=True)
        else:
            st.markdown('<div class="feedback-item pos">Нарушений не выявлено</div>',
                        unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="section-title">⚠️ Замечания</div>', unsafe_allow_html=True)
        if negatives:
            for item in negatives[:5]:
                st.markdown(f'<div class="feedback-item neg">→ {item}</div>',
                            unsafe_allow_html=True)
        else:
            st.markdown('<div class="feedback-item neg" style="background:#f0fdf4;'
                        'border-color:#22c55e;color:#166534;">Замечаний нет</div>',
                        unsafe_allow_html=True)


def render_transcript(role_text: str) -> None:
    if not role_text or role_text.strip() == "—":
        st.info("Транскрипт не сформирован.")
        return
    lines_html: list[str] = []
    for raw_line in role_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("Оператор:"):
            text = line[len("Оператор:"):].strip()
            lines_html.append(
                '<div class="tx-line">'
                '<span class="tx-badge op">Оператор</span>'
                f'<span class="tx-text">{_esc(text)}</span>'
                "</div>"
            )
        elif line.startswith("Заявитель:"):
            text = line[len("Заявитель:"):].strip()
            lines_html.append(
                '<div class="tx-line">'
                '<span class="tx-badge ap">Заявитель</span>'
                f'<span class="tx-text">{_esc(text)}</span>'
                "</div>"
            )
        else:
            lines_html.append(
                f'<div class="tx-line"><span class="tx-text" style="color:#9ca3af;">'
                f"{_esc(line)}</span></div>"
            )
    st.markdown(
        '<div class="transcript-wrap">' + "\n".join(lines_html) + "</div>",
        unsafe_allow_html=True,
    )


def _esc(text: str) -> str:
    return (
        text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
    )


# ── Steps ─────────────────────────────────────────────────────────────────────

_STEPS = [
    ("Загрузка модели", "🧠"),
    ("Распознавание речи", "🎙️"),
    ("Разделение ролей", "👥"),
    ("AI-обработка", "✨"),
    ("Оценка качества", "📊"),
]


def _make_steps_html(active_idx: int, done_up_to: int) -> str:
    parts = []
    for i, (label, _icon) in enumerate(_STEPS):
        if i < done_up_to:
            dot_cls, lbl_cls, symbol = "done", "done", "✓"
        elif i == active_idx:
            dot_cls, lbl_cls, symbol = "active", "active", str(i + 1)
        else:
            dot_cls, lbl_cls, symbol = "idle", "idle", str(i + 1)
        parts.append(
            f'<div class="step">'
            f'<div class="step-dot {dot_cls}">{symbol}</div>'
            f'<div class="step-label {lbl_cls}">{label}</div>'
            f"</div>"
        )
    return '<div class="steps-bar">' + "".join(parts) + "</div>"


# ── Misc helpers ──────────────────────────────────────────────────────────────

def requires_manual_review(
    evaluation: QualityEvaluation, tone_summary: str
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if not evaluation.operator_in_staff:
        reasons.append("оператор не найден в штатном списке")
    if evaluation.total_score <= 4:
        reasons.append("низкий итоговый балл")
    if evaluation.consultation_score <= 4:
        reasons.append("слабая консультация")
    if evaluation.script_score <= 4:
        reasons.append("серьёзные нарушения скрипта")
    if "напряж" in tone_summary.lower():
        reasons.append("напряжённый тон разговора")
    return bool(reasons), reasons


def _format_eta(seconds: float) -> str:
    sec = max(0, int(seconds))
    minutes, rem = divmod(sec, 60)
    if minutes > 0:
        return f"{minutes} мин {rem} сек"
    return f"{rem} сек"


@dataclass
class TranscriptionProgressState:
    """Прогресс одного файла (шаги + полоса), синхронно с логами транскрибера."""

    current_step: int = 0
    done_steps: int = 0
    current_progress: int = 0


def _update_transcription_progress_from_log(
    message: str, state: TranscriptionProgressState
) -> tuple[str | None, bool]:
    """
    Обновляет state по строке лога.
    Возвращает (текст для st.progress или None, обновлять ли полосу шагов).

    Логи транскрибера: ``[transcriber] [этап: имя] сообщение`` (см. transcription.Transcriber._stage).
    """
    m = (message or "").lower()

    # Старт пайплайна — даём ненулевой %, чтобы таймер ETA мог экстраполировать
    if "[этап: начало]" in m:
        state.current_progress = max(state.current_progress, 4)
        return "Открываю файл и запускаю пайплайн...", False
    if "[этап: asr-параметры]" in m:
        state.current_progress = max(state.current_progress, 6)
        return "Параметры распознавания...", False
    if "[этап: asr-модель]" in m and "загрузка" in m:
        state.current_step, state.done_steps, state.current_progress = 0, 0, 12
        return "Загрузка модели Whisper (может быть долго при первом запуске)...", True
    if "[этап: asr-модель]" in m and ("готова" in m or "уже в памяти" in m):
        state.current_step, state.done_steps, state.current_progress = 1, 1, 28
        return "Модель готова", True
    if "[этап: пропуск ожидания]" in m or "[этап: аудио]" in m:
        state.current_progress = max(state.current_progress, 18)
        return "Подготовка аудио (обрезка, нормализация)...", False
    if "[этап: whisper]" in m and "запуск" in m:
        state.current_step, state.done_steps, state.current_progress = 1, 1, 38
        return "Whisper: признаки и VAD...", True
    if "[этап: whisper]" in m and "декодирование сегментов" in m:
        state.current_step, state.done_steps, state.current_progress = 1, 1, 52
        return "Распознавание речи (декодирование)...", True
    if "asr прогресс:" in m:
        state.current_progress = min(84, max(state.current_progress, 52) + 3)
        return "Распознавание речи...", False
    if "распознавание завершено" in m:
        state.current_step, state.done_steps, state.current_progress = 1, 2, 86
        return "Распознавание завершено", True
    if "[этап: роли]" in m and "черновик" in m:
        state.current_step, state.done_steps = 2, 2
        state.current_progress = max(state.current_progress, 87)
        return "Определяю роли (черновик по тексту)...", True
    if (
        ("[этап: роли/" in m or "[этап: роли/ecapa" in m)
        and "ошибка" not in m
    ):
        state.current_step, state.done_steps = 2, 2
        state.current_progress = max(state.current_progress, 88)
        return "Разделение ролей (голос / диаризация)...", True
    if "[этап: роли+голос]" in m and "совмещение" in m:
        state.current_step, state.done_steps = 2, 2
        state.current_progress = max(state.current_progress, 89)
        return "Совмещение голоса и текста...", True
    if "[этап: роли+голос]" in m and "блок завершён" in m:
        state.current_step, state.done_steps, state.current_progress = 3, 3, 91
        return "Роли определены", True
    if "[этап: llm]" in m or (
        "[этап: llm-постредактор]" in m and "выключен" not in m and "не сработал" not in m
    ):
        state.current_step = 3
        state.current_progress = max(state.current_progress, 92)
        return "AI: пост-редактура транскрипта...", True
    if "llm пост-редактор (" in m:
        state.current_step = 3
        state.current_progress = max(state.current_progress, 93)
        return "AI: пост-редактура...", False
    if "[этап: локальный пост]" in m and "выключен" not in m and "пропуск" not in m:
        state.current_step = 3
        state.current_progress = max(state.current_progress, 93)
        return "Локальная пунктуация и роли...", True
    if "[этап: сборка отчёта]" in m:
        state.current_progress = max(state.current_progress, 94)
        return "Сборка расшифровки по ролям...", False
    if "текст собран" in m:
        state.current_progress = max(state.current_progress, 95)
        return "Текст и роли собраны", False
    if "сформирована расшифровка по ролям" in m:
        state.current_step, state.done_steps = 3, 3
        state.current_progress = max(state.current_progress, 96)
        return "Транскрипт по ролям готов", True
    if "[этап: тональность]" in m and "анализ" in m:
        state.current_progress = max(state.current_progress, 97)
        return "Анализ тональности...", False
    if "[этап: готово]" in m:
        state.current_progress = max(state.current_progress, 98)
        return "Завершение распознавания...", False
    if "[evaluator] оцениваю" in m:
        state.current_step, state.done_steps, state.current_progress = 4, 4, 96
        return "Оцениваю качество консультации...", True
    if "[evaluator] оценка готова" in m:
        state.current_step, state.done_steps, state.current_progress = 4, 5, 99
        return "Формирую отчёт...", True
    return None, False


class StreamlitLogger:
    def __init__(self, on_log: Callable[[str], None] | None = None) -> None:
        self.lines: list[str] = []
        self.on_log = on_log

    def __call__(self, message: str) -> None:
        self.lines.append(message)
        if self.on_log is not None:
            self.on_log(message)


class UserStopRequested(Exception):
    pass


# ── Core logic ────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False, scope="session")
def get_cached_transcriber_for_gui(
    model_name: str,
    compute_type: str,
    asr_profile: str,
    heavy_diarization: bool,
    heavy_diarization_timeout_seconds: int,
    enable_post_edit: bool,
    post_edit_timeout_seconds: int,
    enable_llm_post_edit: bool,
    llm_yandex_api_key: str,
    llm_yandex_folder_id: str,
    llm_yandex_model: str,
    llm_yandex_timeout: float,
) -> Transcriber:
    """
    Один экземпляр Transcriber + Whisper на комбинацию настроек сайдбара.
    skip_first_seconds выставляется перед каждым transcribe() (не входит в ключ кеша).
    """
    return Transcriber(
        model_name=model_name,
        compute_type=compute_type,
        skip_first_seconds=None,
        language="ru",
        asr_profile=asr_profile,
        heavy_diarization=heavy_diarization,
        heavy_diarization_timeout_seconds=heavy_diarization_timeout_seconds,
        enable_post_edit=enable_post_edit,
        post_edit_timeout_seconds=post_edit_timeout_seconds,
        enable_llm_post_edit=enable_llm_post_edit,
        llm_backend="yandex" if enable_llm_post_edit else "off",
        llm_yandex_api_key=llm_yandex_api_key or None,
        llm_yandex_folder_id=llm_yandex_folder_id or None,
        llm_yandex_model=llm_yandex_model,
        llm_post_edit_timeout_seconds=float(llm_yandex_timeout),
    )


def run_analysis(
    audio_path: str,
    model_name: str,
    compute_type: str,
    asr_profile: str,
    heavy_diarization: bool,
    heavy_diarization_timeout_seconds: int,
    enable_post_edit: bool,
    post_edit_timeout_seconds: int,
    operator_name: str | None,
    original_basename: str | None = None,
    skip_first_seconds: float | None = None,
    on_log: Callable[[str], None] | None = None,
    *,
    enable_llm_post_edit: bool = False,
    llm_yandex_api_key: str | None = None,
    llm_yandex_folder_id: str | None = None,
    llm_yandex_model: str = "yandexgpt-lite",
    llm_yandex_timeout: float = 60.0,
    cloud_eval_cfg=None,
) -> tuple[str, TranscriptionResult, QualityEvaluation, str]:
    logger = StreamlitLogger(on_log=on_log)
    transcriber = get_cached_transcriber_for_gui(
        model_name,
        compute_type,
        asr_profile,
        heavy_diarization,
        heavy_diarization_timeout_seconds,
        enable_post_edit,
        post_edit_timeout_seconds,
        enable_llm_post_edit,
        llm_yandex_api_key or "",
        llm_yandex_folder_id or "",
        llm_yandex_model,
        float(llm_yandex_timeout),
    )
    transcriber.skip_first_seconds = skip_first_seconds
    transcriber._log = logger  # type: ignore[method-assign]

    result = transcriber.transcribe(audio_path, original_basename=original_basename)
    logger("[evaluator] Оцениваю качество консультации...")
    if cloud_eval_cfg is not None:
        from llm_cloud_eval import run_cloud_evaluation

        evaluation, cloud_status = run_cloud_evaluation(
            result.text,
            result.role_text,
            cloud_eval_cfg,
            forced_operator_name=operator_name,
            log=logger,
        )
        logger(f"[evaluator] Оценка готова ({cloud_status}).")
    else:
        evaluator = CallQualityEvaluator()
        evaluation = evaluator.evaluate(
            result.text,
            result.role_text,
            forced_operator_name=operator_name,
        )
        logger("[evaluator] Оценка готова.")
    report = build_report(result, evaluation)
    return "\n".join(logger.lines), result, evaluation, report


# ── Sidebar ───────────────────────────────────────────────────────────────────

def render_sidebar() -> tuple[bool, object | None]:
    """Рисует сайдбар и возвращает (use_yandex, cloud_eval_cfg)."""
    with st.sidebar:
        st.markdown(f"""
<div class="sidebar-logo">
  <div class="sidebar-logo-icon">📞</div>
  <div>
    <div class="sidebar-logo-text">Анализ звонков</div>
    <div class="sidebar-logo-sub">ДВФУ · ЕКЦ</div>
    <div class="sidebar-logo-version">{APP_VERSION_LABEL} · {APP_VERSION_DATE}</div>
  </div>
</div>""", unsafe_allow_html=True)

        env_api_key = os.environ.get("YANDEX_API_KEY", "").strip()
        env_folder_id = os.environ.get("YANDEX_FOLDER_ID", "").strip()
        env_keys_set = bool(env_api_key and env_folder_id)

        st.markdown("**Яндекс AI Studio**")
        use_yandex = st.toggle(
            "Включить AI-анализ",
            value=env_keys_set,
            help="Включает облачную оценку и исправление транскрипта через Яндекс AI.",
        )

        cloud_eval_cfg = None
        yandex_api_key = env_api_key
        yandex_folder_id = env_folder_id
        _YANDEX_MODEL_ORDER = ("yandexgpt-lite", "yandexgpt/latest")
        _YANDEX_MODEL_LABELS = {
            "yandexgpt-lite": "YandexGPT Lite (по умолчанию)",
            "yandexgpt/latest": "YandexGPT Pro",
        }
        # При выключенном AI значение не используется для запросов
        yandex_model_key = "yandexgpt-lite"
        yandex_timeout = int(float(os.environ.get("YANDEX_CLOUD_TIMEOUT_SECONDS", "60")))

        if use_yandex:
            if env_keys_set:
                st.markdown(
                    '<div class="ai-connected">🟢 Подключено — ключи заданы (см. Админ-панель для смены)</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div class="ai-disconnected">🟡 Ключи не заданы — укажите API Key и Folder ID в «Админ-панели» ниже</div>',
                    unsafe_allow_html=True,
                )

            # Lite / Pro
            if "yandex_cloud_model_lp" not in st.session_state:
                st.session_state["yandex_cloud_model_lp"] = "yandexgpt-lite"
            yandex_model_key = st.radio(
                "Модель",
                options=list(_YANDEX_MODEL_ORDER),
                format_func=lambda k: _YANDEX_MODEL_LABELS[k],
                horizontal=True,
                key="yandex_cloud_model_lp",
                help="Тот же API-ключ Yandex AI Studio. Lite — быстрее; Pro — точнее.",
            )

            with st.expander("Дополнительно", expanded=False):
                yandex_timeout = st.slider(
                    "Таймаут запроса (сек)",
                    min_value=15, max_value=300, value=yandex_timeout, step=5,
                    help="Для длинных звонков при необходимости увеличьте (до 120 с и выше).",
                )

            if yandex_api_key.strip() and yandex_folder_id.strip():
                from llm_cloud_eval import YandexCloudConfig

                cloud_eval_cfg = YandexCloudConfig(
                    api_key=yandex_api_key.strip(),
                    folder_id=yandex_folder_id.strip(),
                    model=yandex_model_key,
                    timeout_seconds=float(yandex_timeout),
                )
            else:
                st.warning("Укажите API Key и Folder ID в разделе «Админ-панель» ниже.")
        else:
            st.markdown(
                '<div class="ai-disconnected">⚪ AI отключён — только локальная эвристика</div>',
                unsafe_allow_html=True,
            )

        st.divider()
        st.markdown("**Распознавание речи**")
        quality = st.radio(
            "Качество",
            # Первый пункт — значение по умолчанию (Ultima)
            options=["ultima", "max", "standard"],
            format_func=lambda x: {
                "max": "🎯 Максимум (~12–22 мин)",
                "standard": "⚡ Стандарт (~1–3 мин)",
                "ultima": "🔮 Ultima (~7–12 мин, RU Large-v3)",
            }[x],
            horizontal=True,
            help=(
                "Стандарт: Whisper Medium, int8; быстро, достаточно для большинства звонков. "
                "Максимум: Systran Large-v3 + beam 9/best_of 5 + 3 температуры; максимальное качество. "
                "Ultima: fine-tuned whisper-large-v3-russian; beam 9/best_of 4/patience 1.1 + 3 температуры; "
                "спектральный денойз (SNR↑ для тихой/шумной речи); "
                "сверхчувствительный VAD pad=2000ms (не режет первые слова); "
                "no_speech_threshold=0.82, log_prob=-1.2 (декодирует даже тихие вступления); "
                "плотная диаризация: окна 2.4с/шаг 0.6с, взвешенный vote. На Mac без GPU — int8_float32."
            ),
        )
        if quality == "standard":
            model_name, compute_type, asr_profile, heavy_mode = "medium", "int8", "medium_ru", False
        elif quality == "ultima":
            model_name, compute_type, asr_profile, heavy_mode = (
                resolve_ultima_whisper_model(),
                "int8_float32",
                "ultima_ru",
                True,
            )
        else:
            model_name, compute_type, asr_profile, heavy_mode = "large-v3", "int8_float32", "ideal_ru", True

        with st.expander("Тонкая настройка", expanded=False):
            heavy_timeout = st.slider(
                "Таймаут диаризации (сек)",
                min_value=90, max_value=300, value=150, step=10,
            )
            enable_post_edit = st.checkbox(
                "Локальный пунктуатор",
                value=False,
                help="Расставляет знаки препинания без интернета. Обычно не нужен с Яндекс AI.",
            )

        render_admin_panel_sidebar()

    return (
        use_yandex, cloud_eval_cfg,
        model_name, compute_type, asr_profile, heavy_mode,
        heavy_timeout, enable_post_edit,
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    # Патчим JS Streamlit в этом venv (5 файлов на страницу + RU), иначе браузер может тянуть старый чанк
    apply_streamlit_file_uploader_localization_patch()

    _warmup_models_once()

    st.set_page_config(
        page_title="Анализ звонков · ДВФУ",
        page_icon="📞",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={},
    )
    st.markdown(_CSS, unsafe_allow_html=True)

    # Сбрасываем сохранённое состояние сайдбара из localStorage —
    # иначе Comet/Chrome может прочитать stSidebarCollapsed=true и схлопнуть его
    st.markdown("""
<script>
(function(){
  try {
    Object.keys(localStorage).forEach(function(k){
      if(k.startsWith('stSidebarCollapsed')) localStorage.removeItem(k);
    });
  } catch(e){}
})();
</script>""", unsafe_allow_html=True)

    render_header()

    (
        use_yandex, cloud_eval_cfg,
        model_name, compute_type, asr_profile, heavy_mode,
        heavy_timeout, enable_post_edit,
    ) = render_sidebar()

    cloud_mode_active = use_yandex and cloud_eval_cfg is not None
    enable_llm_post_edit = cloud_mode_active

    def parse_diag_pairs(diag: str) -> dict[str, str]:
        out: dict[str, str] = {}
        for part in diag.split(";"):
            token = part.strip()
            if not token or "=" not in token:
                continue
            k, v = token.split("=", 1)
            out[k.strip()] = v.strip()
        return out

    _operator_select_labels: dict[str, str] = {
        "Егор": "Егор",
        "Артем": "Артем",
        "Даша": "Даша",
        "Иван": "Иван (Ваня)",
        "Роман": "Роман (Рома)",
        "Юлия": "Юлия (Юля)",
        "Анастасия": "Анастасия (Настя)",
        "Эльвира": "Эльвира",
        "Алина": "Алина",
        "Мария": "Мария",
        "Светлана": "Светлана",
    }
    operator_options = ["Автоопределение"] + [
        _operator_select_labels[n] for n in OPERATOR_CANONICAL_NAMES
    ]
    operator_option_to_name: dict[str, str | None] = {"Автоопределение": None}
    for n in OPERATOR_CANONICAL_NAMES:
        operator_option_to_name[_operator_select_labels[n]] = n

    # ── Табы ────────────────────────────────────────────────────────────────
    tab_single, tab_batch = st.tabs(["📄 Один файл", "📁 Пакетная обработка"])

    with tab_batch:
        render_batch_tab(
            model_name=model_name,
            compute_type=compute_type,
            asr_profile=asr_profile,
            heavy_mode=heavy_mode,
            heavy_timeout=heavy_timeout,
            enable_post_edit=enable_post_edit,
            cloud_eval_cfg=cloud_eval_cfg,
            cloud_mode_active=cloud_mode_active,
        )

    with tab_single:
        _render_single_tab(
            operator_options=operator_options,
            operator_option_to_name=operator_option_to_name,
            parse_diag_pairs=parse_diag_pairs,
            model_name=model_name,
            compute_type=compute_type,
            asr_profile=asr_profile,
            heavy_mode=heavy_mode,
            heavy_timeout=heavy_timeout,
            enable_post_edit=enable_post_edit,
            cloud_mode_active=cloud_mode_active,
            cloud_eval_cfg=cloud_eval_cfg,
            enable_llm_post_edit=enable_llm_post_edit,
        )
# ── Single file tab ───────────────────────────────────────────────────────────

def _render_single_tab(
    operator_options: list[str],
    operator_option_to_name: dict[str, str | None],
    parse_diag_pairs,
    model_name: str,
    compute_type: str,
    asr_profile: str,
    heavy_mode: bool,
    heavy_timeout: int,
    enable_post_edit: bool,
    cloud_mode_active: bool,
    cloud_eval_cfg,
    enable_llm_post_edit: bool,
) -> None:
    col_file, col_op = st.columns([3, 1])
    with col_file:
        uploaded_file = st.file_uploader(
            "Аудиофайл звонка",
            type=["wav", "mp3", "m4a", "ogg", "flac"],
            help=(
                "WAV, MP3, M4A, OGG, FLAC. "
                "Стандартное имя: суффикс `_01-30` — расшифровка с 1:30. "
                "Если имя не по стандарту — укажите ожидание ниже, оно же обрежет начало аудио."
            ),
        )
    with col_op:
        selected_operator = st.selectbox(
            "Оператор",
            options=operator_options,
            index=0,
            help=(
                "Вручную — имя фиксируется сразу. «Автоопределение» — имя оператора только через Yandex API "
                "(облачная оценка должна быть включена в боковой панели)."
            ),
        )

    operator_name = operator_option_to_name[selected_operator]

    from sheets_export import (
        CALL_FILENAME_REQUIREMENTS_RU,
        append_analysis_row,
        is_sheets_configured,
        is_standard_call_filename,
    )

    # Ручные метаданные, если имя файла не по стандарту
    filename_meta_override: dict[str, str] | None = None
    _manual_date: date | None = None
    _manual_phone: str = ""
    _manual_wait_str: str = ""
    _parsed_wait: tuple[int, int] | None = None
    if uploaded_file is not None and not is_standard_call_filename(uploaded_file.name):
        st.warning(
            "Имя файла **не соответствует стандарту**. Укажите вручную **дату**, **телефон** и **время ожидания** "
            "— они будут использованы в таблице и отчёте."
        )
        with st.expander("Требования к имени файла", expanded=False):
            st.markdown(CALL_FILENAME_REQUIREMENTS_RU)
        c1, c2 = st.columns(2)
        with c1:
            _manual_date = st.date_input(
                "Дата звонка",
                value=date.today(),
                key="single_call_manual_date",
            )
        with c2:
            _manual_phone = st.text_input(
                "Номер телефона",
                key="single_call_manual_phone",
                placeholder="Например: 7 (902) 123-45-67",
            )
        # Одно компактное поле ожидания (узкая колонка)
        _wc, _ = st.columns([1, 2.2])
        with _wc:
            _manual_wait_str = st.text_input(
                "Ожидание до ответа",
                key="single_wait_mmss",
                placeholder="1:05",
                max_chars=12,
                help=(
                    "Формат **ММ:СС** или **ММ-СС** (как в шаблоне имени). "
                    "Можно ввести только секунды числом (например **90** = 1:30). "
                    "Пусто = без ожидания. Используется для обрезки записи и колонки «Ожидание»."
                ),
            )

    if "stop_requested" not in st.session_state:
        st.session_state["stop_requested"] = False

    btn_col1, btn_col2 = st.columns([5, 1])
    with btn_col1:
        start_clicked = st.button(
            "🚀  Запустить анализ", type="primary", width="stretch",
        )
    with btn_col2:
        stop_clicked = st.button("⏹ Стоп", type="secondary", width="stretch")

    if stop_clicked:
        st.session_state["stop_requested"] = True
        st.warning("Остановка запрошена — анализ прервётся на ближайшем шаге.")

    if not start_clicked:
        return

    if uploaded_file is None:
        st.error("Выберите аудиофайл.")
        return

    if operator_name is None and cloud_eval_cfg is None:
        st.error(
            "Автоопределение оператора выполняется только через Yandex API (включите облачную оценку в боковой панели). "
            "Или выберите оператора вручную в списке «Оператор»."
        )
        return

    if uploaded_file is not None and not is_standard_call_filename(uploaded_file.name):
        phone_t = (_manual_phone or "").strip()
        if not phone_t:
            st.error("Укажите **номер телефона** вручную (имя файла не по стандарту).")
            return
        _parsed_wait = parse_manual_wait_mm_ss(_manual_wait_str)
        if _parsed_wait is None:
            st.error(
                "Неверный формат **ожидания**. Примеры: `1:05`, `01-30`, или только секунды — `90`. "
                "Пустое поле = 0:00."
            )
            return
        _wm, _ws = _parsed_wait
        filename_meta_override = {
            "date": _manual_date.strftime("%d-%m-%Y"),
            "phone": phone_t,
            "wait_display": f"{_wm:02d}:{_ws:02d}",
        }

    st.session_state["stop_requested"] = False

    # ── Прогресс ────────────────────────────────────────────────────────────
    steps_placeholder = st.empty()
    steps_placeholder.markdown(_make_steps_html(-1, 0), unsafe_allow_html=True)

    progress = st.progress(0, text="Подготовка...")
    eta_placeholder = st.empty()

    with st.expander("Лог выполнения", expanded=False):
        live_logs_placeholder = st.empty()

    ui_logs: list[str] = []
    progress_state = TranscriptionProgressState()
    started_at = time.perf_counter()

    def update_live_log(message: str) -> None:
        if st.session_state.get("stop_requested", False):
            raise UserStopRequested("Анализ остановлен пользователем.")
        ui_logs.append(message)
        live_logs_placeholder.code("\n".join(ui_logs[-20:]) or "")
        bar_text, update_steps = _update_transcription_progress_from_log(message, progress_state)
        if bar_text is not None:
            if update_steps:
                steps_placeholder.markdown(
                    _make_steps_html(progress_state.current_step, progress_state.done_steps),
                    unsafe_allow_html=True,
                )
            progress.progress(progress_state.current_progress, text=bar_text)

    # ── Таймер ETA ──────────────────────────────────────────────────────────
    _stop_timer = threading.Event()
    try:
        from streamlit.runtime.scriptrunner_utils.script_run_context import (
            add_script_run_ctx, get_script_run_ctx,
        )
        _st_ctx = get_script_run_ctx()
    except Exception:
        _st_ctx = None
        add_script_run_ctx = None  # type: ignore[assignment]

    def _eta_timer_loop() -> None:
        if _st_ctx is not None and add_script_run_ctx is not None:
            try:
                add_script_run_ctx(threading.current_thread(), _st_ctx)
            except Exception:
                pass
        while not _stop_timer.wait(1.0):
            elapsed = time.perf_counter() - started_at
            pct = progress_state.current_progress
            if pct <= 0:
                eta_placeholder.caption(
                    f"⏱ Прошло: **{_format_eta(elapsed)}** · "
                    "ожидание первых этапов лога (долгая загрузка модели — норма)…"
                )
            elif pct < 6:
                # При 1–5% линейная экстраполяция даёт огромный «остаток»
                eta_placeholder.caption(
                    f"⏱ Прошло: **{_format_eta(elapsed)}** · "
                    "осталось: **уточняется…** (ранняя стадия)"
                )
            else:
                remaining = elapsed * (100 - pct) / max(pct, 1)
                remaining = min(remaining, 3_600.0)  # не показываем > 1 ч от одной экстраполяции
                eta_placeholder.caption(
                    f"⏱ Прошло: **{_format_eta(elapsed)}** · "
                    f"осталось примерно: **{_format_eta(remaining)}**"
                )

    _timer_thread = threading.Thread(target=_eta_timer_loop, daemon=True, name="eta_timer")
    _timer_thread.start()

    try:
        with st.spinner("Анализирую звонок..."):
            try:
                suffix = Path(uploaded_file.name).suffix or ".wav"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = Path(tmp.name)

                # Пропуск «ожидания» в начале записи: из имени (стандарт) или из ручных полей
                skip_from_manual: float | None = None
                if not is_standard_call_filename(uploaded_file.name):
                    assert _parsed_wait is not None
                    _skip_total = _parsed_wait[0] * 60 + _parsed_wait[1]
                    if _skip_total > 0:
                        skip_from_manual = float(_skip_total)

                logs, result, evaluation, report = run_analysis(
                    audio_path=str(tmp_path),
                    model_name=model_name,
                    compute_type=compute_type,
                    asr_profile=asr_profile,
                    heavy_diarization=heavy_mode,
                    heavy_diarization_timeout_seconds=heavy_timeout,
                    enable_post_edit=enable_post_edit,
                    post_edit_timeout_seconds=60,
                    operator_name=operator_name,
                    original_basename=uploaded_file.name,
                    skip_first_seconds=skip_from_manual,
                    on_log=update_live_log,
                    enable_llm_post_edit=enable_llm_post_edit,
                    llm_yandex_api_key=cloud_eval_cfg.api_key if cloud_eval_cfg else None,
                    llm_yandex_folder_id=cloud_eval_cfg.folder_id if cloud_eval_cfg else None,
                    llm_yandex_model=cloud_eval_cfg.model if cloud_eval_cfg else "yandexgpt-lite",
                    llm_yandex_timeout=cloud_eval_cfg.timeout_seconds if cloud_eval_cfg else 60.0,
                    cloud_eval_cfg=cloud_eval_cfg,
                )
            except UserStopRequested as exc:
                progress.progress(100, text="Остановлено")
                eta_placeholder.empty()
                st.warning(str(exc))
                return
            except Exception as exc:
                progress.progress(100, text="Ошибка")
                eta_placeholder.empty()
                st.error(f"Произошла ошибка: {exc}")
                return
    finally:
        _stop_timer.set()
        _timer_thread.join(timeout=2.0)

    # Сохранить отчёт
    output_path = ensure_results_dir() / f"{Path(uploaded_file.name).stem}_report.txt"
    output_path.write_text(report, encoding="utf-8")

    progress.progress(100, text="Готово!")
    eta_placeholder.empty()
    steps_placeholder.markdown(_make_steps_html(-1, 5), unsafe_allow_html=True)
    progress.empty()

    # Экспорт в Google Таблицу (полный набор колонок)
    try:
        if is_sheets_configured():
            ok, msg = append_analysis_row(
                original_filename=uploaded_file.name,
                total_score=evaluation.total_score,
                max_score=evaluation.max_score,
                operator_name=evaluation.operator_name,
                applicant_name=evaluation.applicant_name or "",
                script_score=evaluation.script_score,
                speech_score=evaluation.speech_score,
                consultation_score=evaluation.consultation_score,
                engagement_score=evaluation.engagement_score,
                tone_summary=result.tone_summary,
                script_details=evaluation.script_details,
                positives=evaluation.positives,
                negatives=evaluation.negatives,
                filename_meta_override=filename_meta_override,
            )
            if ok:
                st.success(f"Google Таблица: {msg}")
            else:
                st.error(f"Google Таблица — строка не добавлена: {msg}")
    except Exception as exc:
        st.error(f"Google Таблица — ошибка: {type(exc).__name__}: {exc}")

    # ── Результаты ──────────────────────────────────────────────────────────
    st.divider()

    # 1. Summary + Gauge
    sc1, sc2 = st.columns([2, 1])
    with sc1:
        render_summary(evaluation)
        render_verdict(evaluation, result.tone_summary)
    with sc2:
        render_gauge(evaluation.total_score)

    # 2. Четыре балла
    st.markdown('<div class="section-title">📊 Детализация оценки</div>',
                unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    render_score_card(c1, "Скрипт звонка",  evaluation.script_score,       "📋")
    render_score_card(c2, "Чистота речи",   evaluation.speech_score,       "🗣️")
    render_score_card(c3, "Консультация",   evaluation.consultation_score, "💡")
    render_score_card(c4, "Вовлечённость",  evaluation.engagement_score,   "🤝")

    # 3. Чеклист скрипта + Плюсы/Минусы
    chk_col, fb_col = st.columns([1, 2])
    with chk_col:
        st.markdown('<div class="section-title">✔ Проверка скрипта</div>',
                    unsafe_allow_html=True)
        render_checklist(evaluation.script_details)
    with fb_col:
        render_feedback(evaluation.positives, evaluation.negatives)

    # 4. Транскрипт
    st.markdown('<div class="section-title">💬 Транскрипт по ролям</div>',
                unsafe_allow_html=True)
    render_transcript(result.role_text or "—")

    # 5. Полный отчёт + скачать
    st.markdown('<div class="section-title">📄 Полный отчёт</div>',
                unsafe_allow_html=True)
    colr1, colr2 = st.columns([4, 1])
    with colr1:
        with st.expander("Показать текстовый отчёт", expanded=False):
            st.text(report)
    with colr2:
        st.download_button(
            label="⬇ Скачать отчёт",
            data=report.encode("utf-8"),
            file_name=f"{Path(uploaded_file.name).stem}_report.txt",
            mime="text/plain",
            width="stretch",
        )

    # 6. Техническая диагностика
    diag_map = parse_diag_pairs(result.role_diagnostics or "")
    with st.expander("🔧 Техническая диагностика", expanded=False):
        d1, d2 = st.columns(2)
        with d1:
            if result.role_attribution:
                st.caption(f"**Роли:** {result.role_attribution}")
            if result.role_confidence:
                st.caption(f"**Уверенность:** {result.role_confidence}")
            st.caption(f"**ASR:** профиль {result.asr_profile}, модель {result.asr_model}")
            if result.language:
                st.caption(f"**Язык:** {result.language}")
            if result.tone_summary and result.tone_summary != "Не определен":
                st.caption(f"**Тон:** {result.tone_summary}")
        with d2:
            asr_s  = diag_map.get("asr_seconds")
            diar_s = diag_map.get("diarization_seconds")
            merge_s = diag_map.get("merge_seconds")
            pe_s   = diag_map.get("post_edit_seconds")
            if any([asr_s, diar_s, merge_s]):
                st.caption(
                    f"**Время:** ASR {asr_s or '?'}s · Диаризация {diar_s or '?'}s"
                    f" · Слияние {merge_s or '?'}s · PostEdit {pe_s or '?'}s"
                )
            fss  = diag_map.get("file_skip_source")
            fsecs = diag_map.get("file_skip_seconds")
            if fss and fss != "none":
                st.caption(f"**Пропуск:** {fss} → {fsecs}s")
        if result.role_diagnostics:
            st.code(result.role_diagnostics, language=None)


# ── Batch processing ──────────────────────────────────────────────────────────

def _result_to_row_dict(
    filename: str,
    result: TranscriptionResult,
    evaluation: QualityEvaluation,
) -> dict:
    """Конвертирует результат одного файла в dict для sheets_export / pandas."""
    from sheets_export import parse_call_filename_metadata, needs_review
    meta = parse_call_filename_metadata(filename)
    return {
        "original_filename": filename,
        "date": meta["date"] or "—",
        "operator_name": evaluation.operator_name or "—",
        "phone": meta["phone"] or "—",
        "applicant_name": evaluation.applicant_name or "—",
        "wait_display": meta["wait_display"] or "—",
        "total_score": evaluation.total_score,
        "max_score": evaluation.max_score,
        "script_score": evaluation.script_score,
        "speech_score": evaluation.speech_score,
        "consultation_score": evaluation.consultation_score,
        "engagement_score": evaluation.engagement_score,
        "tone_summary": result.tone_summary or "—",
        "script_details": evaluation.script_details,
        "positives": evaluation.positives,
        "negatives": evaluation.negatives,
        "review_flag": needs_review(evaluation.total_score, evaluation.negatives),
    }


def _render_live_batch_table(
    placeholder,
    collected: list[dict],
    errors: dict[str, str],
    sheets_log: dict[str, str],
    sheets_configured: bool,
) -> None:
    """Обновляет живую таблицу результатов прямо во время обработки."""
    if not collected and not errors:
        return
    try:
        import pandas as pd

        rows = []
        for i, r in enumerate(collected):
            row = {
                "№": i + 1,
                "Файл": r["original_filename"],
                "Дата": r["date"],
                "Оператор": r["operator_name"],
                "Заявитель": r["applicant_name"],
                "Итог": r["total_score"],
                "Прослушать": r["review_flag"],
            }
            if sheets_configured:
                row["Таблица"] = sheets_log.get(r["original_filename"], "⏳")
            rows.append(row)

        for fname, emsg in errors.items():
            row = {
                "№": "—",
                "Файл": fname,
                "Дата": "—",
                "Оператор": "—",
                "Заявитель": "—",
                "Итог": "❌",
                "Прослушать": emsg[:40],
            }
            if sheets_configured:
                row["Таблица"] = "—"
            rows.append(row)

        df = pd.DataFrame(rows)

        def _color_score(val):
            try:
                v = int(val)
                if v >= 8:
                    return "background-color: #dcfce7; color: #166534"
                if v >= 5:
                    return "background-color: #fef9c3; color: #854d0e"
                return "background-color: #fee2e2; color: #991b1b"
            except Exception:
                return ""

        def _color_review(val):
            if val == "Да":
                return "background-color: #fee2e2; color: #991b1b; font-weight:600"
            if val == "Нет":
                return "background-color: #dcfce7; color: #166534; font-weight:600"
            return ""

        style = df.style.applymap(_color_score, subset=["Итог"]).applymap(
            _color_review, subset=["Прослушать"]
        )
        placeholder.dataframe(style, width="stretch", hide_index=True)
    except ImportError:
        lines = [f"{r['№']}. {r['original_filename']} — {r['total_score']}/10" for r in collected]
        placeholder.text("\n".join(lines))


def _format_uploaded_size(num_bytes: int) -> str:
    if num_bytes >= 1024 * 1024:
        return f"{num_bytes / (1024 * 1024):.1f} МБ"
    if num_bytes >= 1024:
        return f"{num_bytes / 1024:.1f} КБ"
    return f"{num_bytes} Б"


def render_batch_tab(
    model_name: str,
    compute_type: str,
    asr_profile: str,
    heavy_mode: bool,
    heavy_timeout: int,
    enable_post_edit: bool,
    cloud_eval_cfg,
    cloud_mode_active: bool,
) -> None:
    """Рендерит вкладку пакетной обработки."""
    st.markdown(
        '<div class="section-title">📁 Пакетная обработка звонков</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Загрузите несколько аудиофайлов — они будут обработаны по очереди. "
        "Укажите оператора: все записи в пакете считаются звонками одного оператора (автоопределения нет)."
    )

    uploaded_files = st.file_uploader(
        "Аудиофайлы звонков",
        type=["wav", "mp3", "m4a", "ogg", "flac"],
        accept_multiple_files=True,
        help=(
            "WAV, MP3, M4A, OGG, FLAC. "
            "Имена файлов вида ДД-ММ-ГГГГ_телефон_MM-SS.mp3 будут разобраны автоматически."
        ),
        key="batch_uploader",
    )

    if not uploaded_files:
        st.info("Выберите один или несколько аудиофайлов для начала.")
        return

    from sheets_export import (
        CALL_FILENAME_REQUIREMENTS_RU,
        append_analysis_row,
        is_sheets_configured,
        is_standard_call_filename,
    )

    valid_batch_files = [f for f in uploaded_files if is_standard_call_filename(f.name)]
    invalid_batch_files = [f for f in uploaded_files if not is_standard_call_filename(f.name)]

    for bad in invalid_batch_files:
        st.error(
            f"Файл **не принят** к пакетной обработке: `{bad.name}` — имя не соответствует требованиям."
        )
    if invalid_batch_files:
        with st.expander("Требования к имени файла (пакетная обработка)", expanded=True):
            st.markdown(CALL_FILENAME_REQUIREMENTS_RU)

    if not valid_batch_files:
        st.warning(
            "Нет файлов с корректными именами. Переименуйте записи по шаблону выше или загрузите другие файлы."
        )
        return

    st.markdown(
        f"**К обработке: {len(valid_batch_files)}** из {len(uploaded_files)} "
        f"(отклонено по имени: {len(invalid_batch_files)})."
    )
    with st.expander(
        f"📋 Файлы в пакете ({len(valid_batch_files)})",
        expanded=True,
    ):
        for i, uf in enumerate(valid_batch_files, 1):
            st.markdown(f"{i}. `{uf.name}` — {_format_uploaded_size(uf.size)}")

    _BATCH_OP_PLACEHOLDER = "— Выберите оператора —"
    batch_operator_pick = st.selectbox(
        "Оператор для всего пакета (обязательно)",
        options=[_BATCH_OP_PLACEHOLDER] + list(OPERATOR_CANONICAL_NAMES),
        index=0,
        help="Все файлы в пакете будут проанализированы как звонки этого оператора. Автоопределение в пакете отключено.",
        key="batch_operator_select",
    )
    batch_operator_resolved: str | None = (
        None if batch_operator_pick == _BATCH_OP_PLACEHOLDER else batch_operator_pick
    )

    if "batch_results" not in st.session_state:
        st.session_state["batch_results"] = []
    if "batch_errors" not in st.session_state:
        st.session_state["batch_errors"] = {}

    btn_col1, btn_col2 = st.columns([4, 1])
    _can_start_batch = bool(valid_batch_files) and batch_operator_resolved is not None
    with btn_col1:
        start_batch = st.button(
            "🚀 Запустить пакетный анализ",
            type="primary",
            width="stretch",
            key="batch_start",
            disabled=not _can_start_batch,
        )
    if valid_batch_files and batch_operator_resolved is None:
        st.warning("Выберите оператора из списка, чтобы запустить пакетный анализ.")
    with btn_col2:
        if st.button("🗑 Очистить", width="stretch", key="batch_clear"):
            st.session_state["batch_results"] = []
            st.session_state["batch_errors"] = {}
            st.rerun()

    sheets_configured = is_sheets_configured()

    if start_batch:
        st.session_state["batch_results"] = []
        st.session_state["batch_errors"] = {}
        st.session_state["batch_sheets_log"] = {}

        batch_timer_placeholder = st.empty()
        steps_placeholder = st.empty()
        steps_placeholder.markdown(_make_steps_html(-1, 0), unsafe_allow_html=True)
        file_progress = st.progress(0, text="Текущий файл: ожидание…")
        progress_bar = st.progress(0, text="Пакет: подготовка…")
        status_text = st.empty()
        live_table_placeholder = st.empty()
        with st.expander("Лог текущего файла", expanded=True):
            batch_live_logs_placeholder = st.empty()

        total = len(valid_batch_files)
        collected: list[dict] = []
        errors: dict[str, str] = {}
        sheets_log: dict[str, str] = {}

        # ── Оценка длительности аудио по размеру файла ───────────────────────
        # Эмпирический коэффициент из замеров: 979 с аудио → 1061 с обработки
        _PROC_PER_AUDIO_S: float = 1.084  # секунд обработки на секунду аудио

        def _est_audio_s(uf) -> float:
            """Грубая оценка длительности аудио по размеру файла и расширению."""
            ext = Path(uf.name).suffix.lower()
            # байт в секунду для типичных форматов записей звонков
            bps = {
                ".mp3":  8_000,   # 64 кбит/с — стандарт АТС/CRM
                ".wav":  16_000,  # 8 кГц 16 бит моно (телефон G.711)
                ".ogg":  8_000,
                ".m4a":  8_000,
                ".aac":  8_000,
                ".opus": 6_000,
                ".flac": 32_000,
            }.get(ext, 8_000)
            return max(1.0, uf.size / bps)

        est_audio: list[float] = [_est_audio_s(f) for f in valid_batch_files]

        batch_ui: dict = {
            "batch_start": time.perf_counter(),
            "file_idx": 0,
            "file_start": time.perf_counter(),
            "current_progress": 0,
            # wall-clock секунды на каждый завершённый файл
            "completed_durations": [],
            # фактически отработанные секунды аудио (est) — для калибровки ratio
            "completed_audio_est": [],
            "total": len(valid_batch_files),
            # предрасчитанные оценки аудио для каждого файла
            "est_audio": est_audio,
            # текущий калиброванный коэффициент (обновляется после каждого файла)
            "proc_ratio": _PROC_PER_AUDIO_S,
        }
        stop_batch_timer = threading.Event()
        try:
            from streamlit.runtime.scriptrunner_utils.script_run_context import (
                add_script_run_ctx,
                get_script_run_ctx,
            )

            _batch_st_ctx = get_script_run_ctx()
        except Exception:
            _batch_st_ctx = None
            add_script_run_ctx = None  # type: ignore[assignment]

        def _batch_total_timer_loop() -> None:
            if _batch_st_ctx is not None and add_script_run_ctx is not None:
                try:
                    add_script_run_ctx(threading.current_thread(), _batch_st_ctx)
                except Exception:
                    pass
            while not stop_batch_timer.wait(1.0):
                now = time.perf_counter()
                elapsed = now - batch_ui["batch_start"]
                idx = batch_ui["file_idx"]
                n = batch_ui["total"]
                pct = batch_ui["current_progress"]
                file_start = batch_ui["file_start"]
                elapsed_file = now - file_start
                comp = batch_ui["completed_durations"]
                comp_audio = batch_ui["completed_audio_est"]
                ratio = batch_ui["proc_ratio"]
                all_est = batch_ui["est_audio"]

                # ── Оставшееся время для текущего файла ──────────────────────
                cur_audio_est = all_est[idx] if idx < len(all_est) else 60.0
                if pct > 3:
                    # линейная экстраполяция по уже прошедшему времени файла
                    rem_cur = elapsed_file * (100 - pct) / pct
                elif comp:
                    # есть данные по завершённым файлам — берём среднее
                    rem_cur = sum(comp) / len(comp) - elapsed_file
                    rem_cur = max(0.0, rem_cur)
                else:
                    # только начали — используем калиброванный прогноз по аудио
                    rem_cur = max(0.0, cur_audio_est * ratio - elapsed_file)

                # ── Оставшееся время для ещё не начатых файлов ───────────────
                files_after = max(0, n - idx - 1)
                if files_after > 0:
                    # суммируем оценки аудио для оставшихся файлов × ratio
                    future_audio = sum(all_est[idx + 1 : idx + 1 + files_after])
                    rem_rest = future_audio * ratio
                else:
                    rem_rest = 0.0

                rem_total = rem_cur + rem_rest

                # Надпись «уточняется» только в самом начале первого файла
                if not comp and pct < 5 and idx == 0:
                    rest_str = "уточняется…"
                else:
                    rest_str = _format_eta(rem_total)

                batch_timer_placeholder.caption(
                    f"⏱ **Весь пакет:** прошло **{_format_eta(elapsed)}** · "
                    f"осталось примерно **{rest_str}**"
                )

        _batch_timer_thread = threading.Thread(
            target=_batch_total_timer_loop, daemon=True, name="batch_eta_timer"
        )
        _batch_timer_thread.start()

        try:
            for idx, ufile in enumerate(valid_batch_files):
                batch_ui["file_idx"] = idx
                batch_ui["file_start"] = time.perf_counter()
                batch_ui["current_progress"] = 0
                progress_state = TranscriptionProgressState()
                file_ui_logs: list[str] = []

                status_text.markdown(
                    f"⏳ **Файл {idx + 1}/{total}** — `{ufile.name}`"
                )
                progress_bar.progress(
                    idx / total, text=f"Пакет: файл {idx + 1} из {total}"
                )
                file_progress.progress(0, text=f"«{ufile.name}» — старт…")
                steps_placeholder.markdown(_make_steps_html(-1, 0), unsafe_allow_html=True)
                batch_live_logs_placeholder.empty()

                def _make_batch_file_logger(
                    logs_ref: list[str],
                    logs_ph,
                    st_local: TranscriptionProgressState,
                ) -> Callable[[str], None]:
                    def _cb(message: str) -> None:
                        logs_ref.append(message)
                        logs_ph.code("\n".join(logs_ref[-25:]) or "")
                        bar_text, update_steps = _update_transcription_progress_from_log(
                            message, st_local
                        )
                        batch_ui["current_progress"] = st_local.current_progress
                        if bar_text is not None:
                            if update_steps:
                                steps_placeholder.markdown(
                                    _make_steps_html(
                                        st_local.current_step, st_local.done_steps
                                    ),
                                    unsafe_allow_html=True,
                                )
                            file_progress.progress(
                                st_local.current_progress, text=bar_text
                            )

                    return _cb

                update_batch_file_log = _make_batch_file_logger(
                    file_ui_logs, batch_live_logs_placeholder, progress_state
                )

                try:
                    suffix = Path(ufile.name).suffix or ".wav"
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                        tmp.write(ufile.getvalue())
                        tmp_path = Path(tmp.name)

                    _, result, evaluation, _ = run_analysis(
                        audio_path=str(tmp_path),
                        model_name=model_name,
                        compute_type=compute_type,
                        asr_profile=asr_profile,
                        heavy_diarization=heavy_mode,
                        heavy_diarization_timeout_seconds=heavy_timeout,
                        enable_post_edit=enable_post_edit,
                        post_edit_timeout_seconds=60,
                        operator_name=batch_operator_resolved,
                        original_basename=ufile.name,
                        on_log=update_batch_file_log,
                        enable_llm_post_edit=cloud_mode_active,
                        llm_yandex_api_key=cloud_eval_cfg.api_key if cloud_eval_cfg else None,
                        llm_yandex_folder_id=cloud_eval_cfg.folder_id if cloud_eval_cfg else None,
                        llm_yandex_model=cloud_eval_cfg.model if cloud_eval_cfg else "yandexgpt-lite",
                        llm_yandex_timeout=cloud_eval_cfg.timeout_seconds if cloud_eval_cfg else 60.0,
                        cloud_eval_cfg=cloud_eval_cfg,
                    )
                    _file_wall = time.perf_counter() - batch_ui["file_start"]
                    batch_ui["completed_durations"].append(_file_wall)
                    # Обновляем калиброванный коэффициент proc_ratio
                    _caudio = batch_ui["est_audio"][idx] if idx < len(batch_ui["est_audio"]) else 60.0
                    batch_ui["completed_audio_est"].append(_caudio)
                    _total_proc = sum(batch_ui["completed_durations"])
                    _total_audio = sum(batch_ui["completed_audio_est"])
                    if _total_audio > 0:
                        batch_ui["proc_ratio"] = _total_proc / _total_audio
                    file_progress.progress(100, text=f"«{ufile.name}» — готово")
                    steps_placeholder.markdown(
                        _make_steps_html(-1, 5), unsafe_allow_html=True
                    )

                    row = _result_to_row_dict(ufile.name, result, evaluation)
                    collected.append(row)

                    try:
                        from transcription import build_report

                        rpt = build_report(result, evaluation)
                        out = ensure_results_dir() / f"{Path(ufile.name).stem}_report.txt"
                        out.write_text(rpt, encoding="utf-8")
                    except Exception:
                        pass

                    if sheets_configured:
                        try:
                            ok, msg = append_analysis_row(
                                original_filename=ufile.name,
                                total_score=evaluation.total_score,
                                max_score=evaluation.max_score,
                                operator_name=evaluation.operator_name,
                                applicant_name=evaluation.applicant_name or "",
                                script_score=evaluation.script_score,
                                speech_score=evaluation.speech_score,
                                consultation_score=evaluation.consultation_score,
                                engagement_score=evaluation.engagement_score,
                                tone_summary=result.tone_summary,
                                script_details=evaluation.script_details,
                                positives=evaluation.positives,
                                negatives=evaluation.negatives,
                            )
                            sheets_log[ufile.name] = (
                                "✅ Добавлено" if ok else f"⚠️ {msg[:60]}"
                            )
                        except Exception as exc:
                            sheets_log[ufile.name] = f"❌ {str(exc)[:60]}"

                except Exception as exc:
                    errors[ufile.name] = str(exc)

                progress_bar.progress(
                    (idx + 1) / total, text=f"Пакет: обработано {idx + 1} из {total}"
                )
                _render_live_batch_table(
                    live_table_placeholder,
                    collected,
                    errors,
                    sheets_log,
                    sheets_configured,
                )

        finally:
            stop_batch_timer.set()
            _batch_timer_thread.join(timeout=2.0)

        st.session_state["batch_results"] = collected
        st.session_state["batch_errors"] = errors
        st.session_state["batch_sheets_log"] = sheets_log

        total_elapsed = time.perf_counter() - batch_ui["batch_start"]
        batch_timer_placeholder.caption(
            f"⏱ **Пакет завершён.** Всего времени: **{_format_eta(total_elapsed)}**"
        )
        progress_bar.progress(1.0, text="Пакет: готово!")
        ok_cnt = len(collected)
        err_cnt = len(errors)
        sheets_ok_cnt = sum(1 for v in sheets_log.values() if v.startswith("✅"))
        summary = f"**Обработано: {ok_cnt}/{total}**"
        if err_cnt:
            summary += f" · Ошибок: {err_cnt}"
        if sheets_configured:
            summary += f" · В Google Таблицу: {sheets_ok_cnt}/{ok_cnt}"
        summary += f" · ⏱ {_format_eta(total_elapsed)}"
        status_text.markdown(summary)

    # ── Результаты ──────────────────────────────────────────────────────────
    collected = st.session_state.get("batch_results", [])
    errors = st.session_state.get("batch_errors", {})

    if errors:
        with st.expander(f"⚠️ Ошибки обработки ({len(errors)})", expanded=True):
            for fname, emsg in errors.items():
                st.error(f"**{fname}**: {emsg}")

    if not collected:
        return

    sheets_log: dict[str, str] = st.session_state.get("batch_sheets_log", {})

    st.divider()
    st.markdown(
        f'<div class="section-title">📊 Результаты — {len(collected)} звонков</div>',
        unsafe_allow_html=True,
    )

    # Итоговая таблица
    try:
        import pandas as pd

        df = pd.DataFrame([
            {
                "№": i + 1,
                "Дата": r["date"],
                "Оператор": r["operator_name"],
                "Телефон": r["phone"],
                "Заявитель": r["applicant_name"],
                "Ожидание": r["wait_display"],
                "Итог": r["total_score"],
                "Скрипт": r["script_score"],
                "Речь": r["speech_score"],
                "Консультация": r["consultation_score"],
                "Вовлечённость": r["engagement_score"],
                "Тон": r["tone_summary"],
                "Прослушать": r["review_flag"],
                "Таблица": sheets_log.get(r["original_filename"], "—") if sheets_configured else "—",
                "Файл": r["original_filename"],
            }
            for i, r in enumerate(collected)
        ])

        # Цветовое форматирование
        def color_score(val):
            try:
                v = int(val)
            except (TypeError, ValueError):
                return ""
            if v >= 8:
                return "background-color: #dcfce7; color: #166534"
            if v >= 5:
                return "background-color: #fef9c3; color: #854d0e"
            return "background-color: #fee2e2; color: #991b1b"

        def color_review(val):
            if val == "Да":
                return "background-color: #fee2e2; color: #991b1b; font-weight: 600"
            return "background-color: #dcfce7; color: #166534; font-weight: 600"

        score_cols = ["Итог", "Скрипт", "Речь", "Консультация", "Вовлечённость"]
        styled = (
            df.style
            .applymap(color_score, subset=score_cols)
            .applymap(color_review, subset=["Прослушать"])
        )
        st.dataframe(styled, width="stretch", hide_index=True)

        # Статус Google Таблицы
        if sheets_configured:
            sheets_ok_cnt = sum(1 for v in sheets_log.values() if v.startswith("✅"))
            if sheets_ok_cnt == len(collected):
                st.success(f"✅ Все {sheets_ok_cnt} записей добавлены в Google Таблицу автоматически.")
            else:
                failed = [f for f, v in sheets_log.items() if not v.startswith("✅")]
                st.warning(
                    f"Google Таблица: добавлено {sheets_ok_cnt} из {len(collected)}. "
                    f"Ошибки: {', '.join(failed)}"
                )
        else:
            st.caption("Google Таблица не настроена. Добавьте `keygoogle.json` в папку проекта.")

        # CSV
        csv_buf = io.StringIO()
        df.drop(columns=["Таблица"], errors="ignore").to_csv(csv_buf, index=False, encoding="utf-8-sig")
        st.download_button(
            label="⬇ Скачать CSV",
            data=csv_buf.getvalue().encode("utf-8-sig"),
            file_name="qa_batch_results.csv",
            mime="text/csv",
            width="stretch",
            key="batch_csv",
        )

    except ImportError:
        st.warning("Установите pandas для отображения таблицы: `pip install pandas`")
        for i, r in enumerate(collected):
            flag_html = (
                '<span class="batch-review-yes">Да — прослушать</span>'
                if r["review_flag"] == "Да"
                else '<span class="batch-review-no">Нет</span>'
            )
            st.markdown(
                f'<div class="batch-file-card ok">'
                f'<b>{i+1}.</b> {r["original_filename"]} &nbsp;|&nbsp; '
                f'Оценка: <b>{r["total_score"]}/10</b> &nbsp;|&nbsp; '
                f'{flag_html}'
                f'</div>',
                unsafe_allow_html=True,
            )


# ── Warmup ────────────────────────────────────────────────────────────────────

def _warmup_models_once() -> None:
    """Прогрев ECAPA-TDNN в фоне при первом запуске страницы."""
    if st.session_state.get("_models_warmed_up"):
        return
    st.session_state["_models_warmed_up"] = True
    try:
        from speaker_voice_roles import warmup_ecapa_model_background
        warmup_ecapa_model_background()
    except Exception:
        pass


if __name__ == "__main__":
    main()
