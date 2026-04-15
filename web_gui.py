from __future__ import annotations

import os
import ssl
import sys

# Глобально отключаем проверку SSL-сертификатов — нужно для api.awstore.cloud
# и других внешних API с self-signed сертификатами.
# Патчим ДО любого импорта, который может создать SSL-контекст.

def _make_insecure_context():
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx

ssl._create_default_https_context = _make_insecure_context
ssl.create_default_context = _make_insecure_context

# macOS: при спавне воркеров Streamlit/lib в stderr сыпется
# «MallocStackLogging: can't turn off...» — задаём режим malloc до нативных импортов.
if sys.platform == "darwin":
    # Явно «выкл.»; иначе у потомков иногда остаётся несогласованное состояние и libsystem_malloc шумит в stderr.
    os.environ["MallocStackLogging"] = "0"

# До streamlit и любого импорта speechbrain: torchaudio 2.2+ и фильтр шумных варнингов SB.
from speechbrain_compat import apply_speechbrain_environment

apply_speechbrain_environment()

import io
import re
from datetime import date
from pathlib import Path
import tempfile
import threading
import time
from typing import Callable

import streamlit as st

from analysis_service import (
    AnalysisRequest,
    TranscriberSettings,
    analyze_call,
    build_sheets_export_payload,
    create_transcriber,
)
from app_config import load_app_environment
from batch_state import (
    BatchFileJob,
    BatchResumeState,
    batch_resume_files_dir,
    batch_resume_state_path,
    clear_batch_resume_state,
    load_batch_resume_state,
    save_batch_resume_state,
)
from ui_batch_helpers import (
    format_uploaded_size,
    render_live_batch_table,
    result_to_row_dict,
)
from ui_progress_helpers import (
    TranscriptionProgressState,
    format_eta,
    make_steps_html,
    update_transcription_progress_from_log,
)
from ui_result_components import (
    render_checklist,
    render_feedback,
    render_gauge,
    render_header,
    render_score_card,
    render_summary,
    render_transcript,
    render_verdict,
)
from ui_sidebar import render_sidebar

# Версия UI — дублируйте при релизе в CHANGELOG.md
APP_VERSION_LABEL = "Beta 1.1"
APP_VERSION_DATE = "24.03.2026"
load_app_environment()

from operator_staff import OPERATOR_CANONICAL_NAMES
from results_paths import ensure_results_dir
from streamlit_file_uploader_patch import apply_streamlit_file_uploader_localization_patch
from transcription import Transcriber

_MANUAL_WAIT_INPUT_RE = re.compile(
    r"^\s*(\d{1,3})\s*[:-]\s*(\d{1,2})\s*$"
)

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

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #d1d5db; border-radius: 9999px; }
::-webkit-scrollbar-thumb:hover { background: #9ca3af; }

/* ── Шапка ── */
.qa-header {
    display: flex; align-items: center; gap: 16px;
    padding: 0 0 24px 0; border-bottom: 2px solid #e5e7eb; margin-bottom: 28px;
}
.qa-header-icon {
    background: linear-gradient(135deg, #2563eb, #1e40af);
    border-radius: 16px; width: 52px; height: 52px;
    display: flex; align-items: center; justify-content: center;
    font-size: 26px; box-shadow: 0 6px 16px rgba(37,99,235,.3);
    flex-shrink: 0;
}
.qa-header-title { font-size: 24px; font-weight: 800; color: #0f172a; line-height: 1.2; }
.qa-header-sub   { font-size: 13px; color: #64748b; margin-top: 3px; font-weight: 500; }

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
    display: flex; align-items: center; gap: 12px;
    padding: 0 0 16px 0; border-bottom: 1px solid #e2e8f0; margin-bottom: 20px;
}
.sidebar-logo-icon {
    background: linear-gradient(135deg, #2563eb, #1e40af);
    border-radius: 12px; width: 40px; height: 40px;
    display: flex; align-items: center; justify-content: center; font-size: 20px;
    box-shadow: 0 2px 8px rgba(37,99,235,.2);
}
.sidebar-logo-text { font-size: 15px; font-weight: 700; color: #0f172a; }
.sidebar-logo-sub  { font-size: 12px; color: #64748b; font-weight: 500; }
.sidebar-logo-version { font-size: 11px; color: #94a3b8; margin-top: 2px; }

.ai-connected {
    background: linear-gradient(135deg, #f0fdf4, #ecfdf5); border: 1px solid #bbf7d0;
    border-radius: 8px; padding: 8px 12px;
    font-size: 12px; color: #15803d; margin-bottom: 14px;
    display: flex; align-items: center; gap: 6px; font-weight: 500;
}
.ai-disconnected {
    background: linear-gradient(135deg, #fff7ed, #fffbeb); border: 1px solid #fed7aa;
    border-radius: 8px; padding: 8px 12px;
    font-size: 12px; color: #c2410c; margin-bottom: 14px;
    display: flex; align-items: center; gap: 6px; font-weight: 500;
}

/* ── Шаговый прогресс ── */
.steps-bar {
    display: flex; align-items: center;
    background: #f8fafc; border: 1px solid #e2e8f0;
    border-radius: 16px; padding: 16px 24px; margin: 18px 0; gap: 0;
    box-shadow: 0 1px 3px rgba(0,0,0,.03);
}
.step {
    display: flex; align-items: center; gap: 10px;
    flex: 1; justify-content: center; position: relative;
}
.step:not(:last-child)::after {
    content: ''; position: absolute; right: 0;
    width: 1px; height: 32px; background: #e2e8f0;
}
.step-dot {
    width: 32px; height: 32px; border-radius: 9999px;
    display: flex; align-items: center; justify-content: center;
    font-size: 13px; font-weight: 700; flex-shrink: 0;
}
.step-dot.done  { background: #dcfce7; color: #16a34a; }
.step-dot.active{ background: #dbeafe; color: #2563eb; animation: pulse 1.5s infinite; }
.step-dot.idle  { background: #f1f5f9; color: #94a3b8; }
.step-label { font-size: 12px; font-weight: 500; }
.step-label.done  { color: #16a34a; }
.step-label.active{ color: #2563eb; }
.step-label.idle  { color: #94a3b8; }

@keyframes pulse {
    0%,100%{ box-shadow: 0 0 0 0 rgba(37,99,235,.25); }
    50%    { box-shadow: 0 0 0 8px rgba(37,99,235,.0); }
}

/* ── Карточки счётчиков ── */
.score-card {
    background: #fff; border: 1px solid #e5e7eb; border-radius: 16px;
    padding: 20px 16px 16px; text-align: center;
    box-shadow: 0 1px 3px rgba(0,0,0,.04), 0 1px 2px rgba(0,0,0,.06);
    transition: transform .15s ease, box-shadow .2s ease;
}
.score-card:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(0,0,0,.08); }
.score-title { font-size: 11px; color: #64748b; font-weight: 600;
               text-transform: uppercase; letter-spacing: .06em; margin-bottom: 10px; }
.score-value { font-size: 40px; font-weight: 900; line-height: 1; }
.score-sub   { font-size: 13px; color: #94a3b8; margin-top: 2px; }
.score-bar   { height: 6px; border-radius: 9999px; background: #f1f5f9;
               margin-top: 12px; overflow: hidden; }
.score-fill  { height: 100%; border-radius: 9999px; transition: width .5s ease; }
.score-badge { display:inline-block; font-size:11px; font-weight:600;
               border-radius:6px; padding:3px 10px; margin-top:10px; }

/* ── Summary ── */
.summary-card {
    background: #fff; border: 1px solid #e5e7eb; border-radius: 20px;
    padding: 24px 32px; margin-bottom: 24px;
    box-shadow: 0 2px 12px rgba(0,0,0,.04), 0 1px 3px rgba(0,0,0,.06);
    display: flex; align-items: center; justify-content: space-between;
    flex-wrap: wrap; gap: 20px;
}
.summary-person-label {
    font-size: 11px; color: #64748b; text-transform: uppercase;
    letter-spacing: .06em; margin-bottom: 6px; font-weight: 600;
}
.summary-person-name {
    font-size: 20px; font-weight: 700; color: #0f172a;
}
.summary-score-wrap { text-align: right; }
.summary-score-label {
    font-size: 11px; color: #64748b; text-transform: uppercase;
    letter-spacing: .06em; margin-bottom: 6px; font-weight: 600;
}
.summary-score-val {
    font-size: 56px; font-weight: 900; line-height: 1;
}
.summary-score-denom { font-size: 24px; color: #cbd5e1; }

/* ── Чеклист ── */
.checklist-item {
    display: flex; align-items: center; gap: 10px;
    padding: 12px 14px; border-radius: 10px; margin-bottom: 6px;
    font-size: 14px; font-weight: 500;
    transition: transform .1s ease;
}
.checklist-item:hover { transform: translateX(2px); }
.checklist-item.ok  { background: #f0fdf4; color: #166534; }
.checklist-item.fail{ background: #fef2f2; color: #991b1b; }
.checklist-icon { font-size: 16px; flex-shrink: 0; }

/* ── Плюсы / минусы ── */
.feedback-item {
    border-radius: 12px; padding: 12px 16px; margin-bottom: 8px;
    font-size: 14px; line-height: 1.6;
}
.feedback-item.pos { background: #f0fdf4; border-left: 4px solid #22c55e; color: #166534; }
.feedback-item.neg { background: #fff7ed; border-left: 4px solid #f59e0b; color: #92400e; }

/* ── Транскрипт ── */
.transcript-wrap {
    background: #fafafa; border: 1px solid #e5e7eb; border-radius: 14px;
    padding: 16px; max-height: 480px; overflow-y: auto; font-size: 14px;
}
.tx-line { display: flex; gap: 12px; margin-bottom: 10px; }
.tx-badge {
    flex-shrink: 0; font-size: 10px; font-weight: 700; border-radius: 6px;
    padding: 3px 10px; height: fit-content; margin-top: 2px; letter-spacing: .04em;
    white-space: nowrap;
}
.tx-badge.op  { background: #dbeafe; color: #1d4ed8; }
.tx-badge.ap  { background: #f1f5f9; color: #475569; }
.tx-text { color: #334155; line-height: 1.65; }

/* ── Verdict ── */
.verdict-ok  {
    background: linear-gradient(135deg, #f0fdf4, #ecfdf5); border: 1px solid #bbf7d0; border-radius: 12px;
    padding: 16px 20px; color: #15803d; font-size: 15px; font-weight: 600;
    display: flex; align-items: center; gap: 10px; margin-bottom: 20px;
    box-shadow: 0 1px 3px rgba(0,0,0,.04);
}
.verdict-bad {
    background: linear-gradient(135deg, #fef2f2, #fff1f2); border: 1px solid #fecaca; border-radius: 12px;
    padding: 16px 20px; color: #b91c1c; font-size: 15px; font-weight: 600;
    display: flex; align-items: center; gap: 10px; margin-bottom: 20px;
    box-shadow: 0 1px 3px rgba(0,0,0,.04);
}
.verdict-reasons { font-size: 13px; font-weight: 400; color: #b91c1c; margin-top: 4px; }

/* ── Секции ── */
.section-title {
    font-size: 17px; font-weight: 700; color: #0f172a;
    margin: 28px 0 14px 0; display: flex; align-items: center; gap: 8px;
}

/* ── Пакетный режим — прогресс-файл ── */
.batch-file-card {
    background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 10px;
    padding: 12px 14px; margin-bottom: 8px; font-size: 13px; color: #475569;
    display: flex; align-items: center; gap: 10px;
    transition: box-shadow .15s ease;
}
.batch-file-card:hover { box-shadow: 0 2px 8px rgba(0,0,0,.06); }
.batch-file-card.ok  { border-left: 4px solid #22c55e; }
.batch-file-card.err { border-left: 4px solid #ef4444; }
.batch-review-yes {
    background: #fef2f2; color: #991b1b; font-weight: 600;
    border-radius: 6px; padding: 3px 8px; font-size: 12px;
}
.batch-review-no {
    background: #f0fdf4; color: #166534; font-weight: 600;
    border-radius: 6px; padding: 3px 8px; font-size: 12px;
}

/* ── Streamlit element refinements ── */
/* Expander bodies with subtle border */
.streamlit-expander-header { border-radius: 8px !important; }
[data-baseweb="Disclosure"] [data-baseweb="Panel"] {
    border-radius: 0 0 10px 10px !important;
}

/* Buttons: slightly more prominent primary */
.stButton button[kind="primary"] {
    border-radius: 10px !important;
    font-weight: 600 !important;
    transition: transform .1s ease, box-shadow .15s ease !important;
}
.stButton button[kind="primary"]:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(37,99,235,.25) !important;
}
.stButton button[kind="secondary"] {
    border-radius: 10px !important;
    font-weight: 500 !important;
}

/* File uploader: cleaner look */
[data-testid="stFileUploader"] {
    border-radius: 12px !important;
}

/* SelectBox and text inputs: slightly rounder */
input[type="text"], [data-baseweb="Select"] {
    border-radius: 8px !important;
}

/* Tab styling */
[data-baseweb="Tab"] {
    font-weight: 500 !important;
    border-radius: 8px 8px 0 0 !important;
}
</style>
"""


class UserStopRequested(Exception):
    pass


# ── Core logic ────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False, scope="session")
def get_cached_transcriber_for_gui(settings: TranscriberSettings) -> Transcriber:
    """
    Один экземпляр Transcriber + Whisper на комбинацию настроек сайдбара.
    skip_first_seconds выставляется перед каждым transcribe() (не входит в ключ кеша).
    """
    return create_transcriber(settings)


def _build_gui_transcriber_settings(
    model_name: str,
    compute_type: str,
    asr_profile: str,
    heavy_diarization: bool,
    heavy_diarization_timeout_seconds: int,
    enable_post_edit: bool,
    post_edit_timeout_seconds: int,
    enable_llm_post_edit: bool,
    llm_backend: str,
    llm_yandex_api_key: str | None,
    llm_yandex_folder_id: str | None,
    llm_yandex_model: str,
    llm_yandex_timeout: float,
    llm_claude_api_key: str | None = None,
    llm_claude_base_url: str | None = None,
    llm_claude_model: str | None = None,
    asr_backend: str = "whisper",
) -> TranscriberSettings:
    # For post-edit: prefer Yandex (no forced-thinking delay) when both are configured.
    # The awstore proxy forces thinking mode on all Claude models regardless of the
    # "thinking: disabled" parameter, causing 60–95 s delays. Yandex has no such issue.
    effective_llm_backend = llm_backend
    if enable_llm_post_edit and llm_backend == "claude" and llm_yandex_api_key and llm_yandex_folder_id:
        effective_llm_backend = "yandex"

    return TranscriberSettings(
        model_name=model_name,
        compute_type=compute_type,
        asr_profile=asr_profile,
        heavy_diarization=heavy_diarization,
        heavy_diarization_timeout_seconds=heavy_diarization_timeout_seconds,
        enable_post_edit=enable_post_edit,
        post_edit_timeout_seconds=post_edit_timeout_seconds,
        enable_llm_post_edit=enable_llm_post_edit,
        llm_backend=effective_llm_backend if enable_llm_post_edit else "off",
        llm_yandex_api_key=llm_yandex_api_key,
        llm_yandex_folder_id=llm_yandex_folder_id,
        llm_yandex_model=llm_yandex_model,
        llm_post_edit_timeout_seconds=float(llm_yandex_timeout),
        llm_claude_api_key=llm_claude_api_key,
        llm_claude_base_url=llm_claude_base_url,
        llm_claude_model=llm_claude_model,
        asr_backend=asr_backend,
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

    sidebar_state = render_sidebar(APP_VERSION_LABEL, APP_VERSION_DATE)

    cloud_mode_active = (
        sidebar_state.cloud_backend != "off" and sidebar_state.cloud_eval_cfg is not None
    )
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
            model_name=sidebar_state.model_name,
            compute_type=sidebar_state.compute_type,
            asr_profile=sidebar_state.asr_profile,
            heavy_mode=sidebar_state.heavy_mode,
            heavy_timeout=sidebar_state.heavy_timeout,
            enable_post_edit=sidebar_state.enable_post_edit,
            cloud_eval_cfg=sidebar_state.cloud_eval_cfg,
            cloud_mode_active=cloud_mode_active,
            sidebar_state=sidebar_state,
        )

    with tab_single:
        _render_single_tab(
            operator_options=operator_options,
            operator_option_to_name=operator_option_to_name,
            parse_diag_pairs=parse_diag_pairs,
            model_name=sidebar_state.model_name,
            compute_type=sidebar_state.compute_type,
            asr_profile=sidebar_state.asr_profile,
            heavy_mode=sidebar_state.heavy_mode,
            heavy_timeout=sidebar_state.heavy_timeout,
            enable_post_edit=sidebar_state.enable_post_edit,
            cloud_mode_active=cloud_mode_active,
            cloud_eval_cfg=sidebar_state.cloud_eval_cfg,
            enable_llm_post_edit=enable_llm_post_edit,
            sidebar_state=sidebar_state,
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
    sidebar_state,
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
    steps_placeholder.markdown(make_steps_html(-1, 0), unsafe_allow_html=True)

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
        bar_text, update_steps = update_transcription_progress_from_log(message, progress_state)
        if bar_text is not None:
            if update_steps:
                steps_placeholder.markdown(
                    make_steps_html(progress_state.current_step, progress_state.done_steps),
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
                    f"⏱ Прошло: **{format_eta(elapsed)}** · "
                    "ожидание первых этапов лога (долгая загрузка модели — норма)…"
                )
            elif pct < 6:
                # При 1–5% линейная экстраполяция даёт огромный «остаток»
                eta_placeholder.caption(
                    f"⏱ Прошло: **{format_eta(elapsed)}** · "
                    "осталось: **уточняется…** (ранняя стадия)"
                )
            else:
                remaining = elapsed * (100 - pct) / max(pct, 1)
                remaining = min(remaining, 3_600.0)  # не показываем > 1 ч от одной экстраполяции
                eta_placeholder.caption(
                    f"⏱ Прошло: **{format_eta(elapsed)}** · "
                    f"осталось примерно: **{format_eta(remaining)}**"
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

                from llm_cloud_eval import YandexCloudConfig

                _is_yandex = isinstance(cloud_eval_cfg, YandexCloudConfig)
                # Always fetch Yandex credentials from app_config so post-edit can route
                # to Yandex (fast) even when Claude is the primary evaluation backend.
                _app_cfg = load_app_config()
                _yandex_cfg = _app_cfg.yandex_cloud
                transcriber_settings = _build_gui_transcriber_settings(
                    model_name=model_name,
                    compute_type=compute_type,
                    asr_profile=asr_profile,
                    heavy_diarization=heavy_mode,
                    heavy_diarization_timeout_seconds=heavy_timeout,
                    enable_post_edit=enable_post_edit,
                    post_edit_timeout_seconds=60,
                    enable_llm_post_edit=enable_llm_post_edit,
                    llm_backend=sidebar_state.cloud_backend,
                    llm_yandex_api_key=_yandex_cfg.api_key if _yandex_cfg.configured else None,
                    llm_yandex_folder_id=_yandex_cfg.folder_id if _yandex_cfg.configured else None,
                    llm_yandex_model=_yandex_cfg.model,
                    llm_yandex_timeout=_yandex_cfg.timeout_seconds,
                    asr_backend=sidebar_state.asr_backend,
                )
                analysis = analyze_call(
                    AnalysisRequest(
                        audio_path=str(tmp_path),
                        operator_name=operator_name,
                        original_basename=uploaded_file.name,
                        skip_first_seconds=skip_from_manual,
                        cloud_eval_cfg=cloud_eval_cfg,
                    ),
                    transcriber=get_cached_transcriber_for_gui(transcriber_settings),
                    on_log=update_live_log,
                )
                result = analysis.result
                evaluation = analysis.evaluation
                report = analysis.report
            except UserStopRequested as exc:
                progress.progress(100, text="Остановлено")
                eta_placeholder.empty()
                st.warning(str(exc))
                return
            except Exception as exc:
                progress.progress(100, text="Ошибка")
                eta_placeholder.empty()
                import traceback
                st.error(f"Произошла ошибка: {exc}")
                st.code(traceback.format_exc())
                return
    finally:
        _stop_timer.set()
        _timer_thread.join(timeout=2.0)

    # Сохранить отчёт
    output_path = ensure_results_dir() / f"{Path(uploaded_file.name).stem}_report.txt"
    output_path.write_text(report, encoding="utf-8")

    progress.progress(100, text="Готово!")
    eta_placeholder.empty()
    steps_placeholder.markdown(make_steps_html(-1, 5), unsafe_allow_html=True)
    progress.empty()

    # Экспорт в Google Таблицу (полный набор колонок)
    try:
        if is_sheets_configured():
            ok, msg = append_analysis_row(
                **build_sheets_export_payload(
                    analysis,
                    original_filename=uploaded_file.name,
                    filename_meta_override=filename_meta_override,
                )
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

def render_batch_tab(
    model_name: str,
    compute_type: str,
    asr_profile: str,
    heavy_mode: bool,
    heavy_timeout: int,
    enable_post_edit: bool,
    cloud_eval_cfg,
    cloud_mode_active: bool,
    sidebar_state,
) -> None:
    """Рендерит вкладку пакетной обработки."""
    st.markdown(
        '<div class="section-title">📁 Пакетная обработка звонков</div>',
        unsafe_allow_html=True,
    )
    st.info(
        "Загрузите несколько аудиофайлов с именами в формате **ДД-ММ-ГГГГ_телефон_MM-СС**. "
        "Все файлы обрабатываются последовательно с одинаковыми настройками."
    )

    from sheets_export import (
        CALL_FILENAME_REQUIREMENTS_RU,
        append_analysis_row,
        is_sheets_configured,
        is_standard_call_filename,
        sheets_config_hint,
    )

    resume_state = load_batch_resume_state()
    resume_files = list(resume_state.files) if resume_state is not None else []
    resume_next_idx = resume_state.next_idx if resume_state is not None else 0
    resume_total = len(resume_files)
    resume_available = bool(resume_total > 0 and 0 <= resume_next_idx < resume_total)

    resume_batch = False
    _AUTO_RESUME_WINDOW_S = 300
    _AUTO_RESUME_COUNTDOWN_S = 15
    if resume_available:
        # Авто-возобновление: если пакет прервался недавно (< 5 мин), запускаем обратный отсчёт.
        try:
            _batch_mtime = batch_resume_state_path().stat().st_mtime
            _recently_active = (time.time() - _batch_mtime) < _AUTO_RESUME_WINDOW_S
        except OSError:
            _recently_active = False

        _ar_cancelled = st.session_state.get("batch_auto_resume_cancelled", False)

        if _recently_active and not _ar_cancelled:
            if "batch_auto_resume_deadline" not in st.session_state:
                st.session_state["batch_auto_resume_deadline"] = time.time() + _AUTO_RESUME_COUNTDOWN_S

            _ar_deadline = st.session_state["batch_auto_resume_deadline"]
            _ar_remaining = max(0, int(_ar_deadline - time.time()))

            if _ar_remaining <= 0:
                st.session_state.pop("batch_auto_resume_deadline", None)
                st.session_state.pop("batch_auto_resume_cancelled", None)
                resume_batch = True
            else:
                st.warning(
                    f"⏱ Незавершённый пакет ({resume_next_idx}/{resume_total} обработано). "
                    f"Автоматическое продолжение через **{_ar_remaining}** сек…"
                )
                _ar_c1, _ar_c2 = st.columns([5, 2])
                with _ar_c1:
                    st.progress(
                        1.0 - (_ar_remaining / _AUTO_RESUME_COUNTDOWN_S),
                        text=f"{_ar_remaining} сек до автовозобновления",
                    )
                with _ar_c2:
                    if st.button("❌ Отменить автовозобновление", key="batch_auto_resume_cancel_btn"):
                        st.session_state["batch_auto_resume_cancelled"] = True
                        st.session_state.pop("batch_auto_resume_deadline", None)
                        st.rerun()
                time.sleep(1)
                st.rerun()

        if not resume_batch:
            st.info(
                f"Найден незавершённый пакет: {resume_next_idx}/{resume_total} файлов уже обработано. "
                "Можно продолжить с места остановки."
            )
            r1, r2 = st.columns([4, 1])
            with r1:
                resume_batch = st.button(
                    "▶ Продолжить незавершённый пакет",
                    type="secondary",
                    width="stretch",
                    key="batch_resume",
                )
            with r2:
                if st.button("🗑 Сбросить", width="stretch", key="batch_resume_clear"):
                    clear_batch_resume_state()
                    st.session_state.pop("batch_auto_resume_deadline", None)
                    st.session_state.pop("batch_auto_resume_cancelled", None)
                    st.success("Незавершённый пакет удалён.")
                    st.rerun()

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

    if not uploaded_files and not resume_batch:
        st.info("Загрузите файлы с именами в формате ДД-ММ-ГГГГ_телефон_MM-СС — они будут обработаны последовательно.")
        return

    valid_batch_files = [f for f in (uploaded_files or []) if is_standard_call_filename(f.name)]
    invalid_batch_files = [f for f in (uploaded_files or []) if not is_standard_call_filename(f.name)]

    for bad in invalid_batch_files:
        st.error(
            f"Файл **не принят** к пакетной обработке: `{bad.name}` — имя не соответствует требованиям."
        )
    if invalid_batch_files:
        with st.expander("Требования к имени файла (пакетная обработка)", expanded=True):
            st.markdown(CALL_FILENAME_REQUIREMENTS_RU)

    if not valid_batch_files and not resume_batch:
        st.warning(
            "Нет файлов с корректными именами. Переименуйте записи по шаблону выше или загрузите другие файлы."
        )
        return

    if valid_batch_files:
        st.markdown(
            f"**К обработке: {len(valid_batch_files)}** из {len(uploaded_files)} "
            f"(отклонено по имени: {len(invalid_batch_files)})."
        )
        with st.expander(
            f"📋 Файлы в пакете ({len(valid_batch_files)})",
            expanded=True,
        ):
            for i, uf in enumerate(valid_batch_files, 1):
                st.markdown(f"{i}. `{uf.name}` — {format_uploaded_size(uf.size)}")

    _BATCH_OP_PLACEHOLDER = "— Выберите оператора —"
    if resume_batch and resume_state is not None:
        _saved_op = resume_state.operator_name.strip()
        batch_operator_resolved: str | None = _saved_op or None
        st.caption(f"Режим возобновления: оператор пакета — `{batch_operator_resolved or 'Не указан'}`")
    else:
        batch_operator_pick = st.selectbox(
            "Оператор для всего пакета (обязательно)",
            options=[_BATCH_OP_PLACEHOLDER] + list(OPERATOR_CANONICAL_NAMES),
            index=0,
            help="Все файлы в пакете будут проанализированы как звонки этого оператора. Автоопределение в пакете отключено.",
            key="batch_operator_select",
        )
        batch_operator_resolved = (
            None if batch_operator_pick == _BATCH_OP_PLACEHOLDER else batch_operator_pick
        )

    if "batch_results" not in st.session_state:
        st.session_state["batch_results"] = []
    if "batch_errors" not in st.session_state:
        st.session_state["batch_errors"] = {}

    if valid_batch_files and batch_operator_resolved is None and not resume_batch:
        st.warning("Выберите оператора из списка, чтобы запустить пакетный анализ.")

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
    with btn_col2:
        if st.button("🗑 Очистить", width="stretch", key="batch_clear"):
            st.session_state["batch_results"] = []
            st.session_state["batch_errors"] = {}
            st.rerun()

    sheets_configured = is_sheets_configured()

    if cloud_mode_active and cloud_eval_cfg is not None:
        st.text_area(
            "Доп. комментарии для нейросети (оценка звонка)",
            height=100,
            key="batch_eval_extra_instructions",
            placeholder=(
                "Например: пакет по льготам — в positives отмечай, если оператор перечислил документы; "
                "строже к срокам ответа…"
            ),
            help=(
                "Текст добавляется к системному промпту **облачной оценки** для **каждого** файла "
                "в пакете. Не меняет распознавание речи. При «Продолжить незавершённый пакет» "
                "используются инструкции, сохранённые при старте пакета."
            ),
        )

    if start_batch or resume_batch:
        st.session_state.pop("batch_auto_resume_deadline", None)
        st.session_state.pop("batch_auto_resume_cancelled", None)
        if start_batch:
            st.session_state["batch_results"] = []
            st.session_state["batch_errors"] = {}
            st.session_state["batch_sheets_log"] = {}

        # Список файлов для обработки:
        # - новый запуск: сохраняем upload-байты на диск и пишем checkpoint
        # - resume: читаем сохранённое состояние и продолжаем с next_idx
        file_jobs: list[BatchFileJob] = []
        start_idx = 0
        collected: list[dict] = []
        errors: dict[str, str] = {}
        sheets_log: dict[str, str] = {}
        batch_timings: list[dict] = []

        if resume_batch and resume_state is not None:
            raw_jobs = resume_state.files
            if not raw_jobs:
                st.error("Незавершённый пакет повреждён: список файлов пуст.")
                clear_batch_resume_state()
                return
            file_jobs = list(raw_jobs)
            start_idx = resume_state.next_idx
            collected = list(resume_state.collected)
            errors = dict(resume_state.errors)
            sheets_log = dict(resume_state.sheets_log)
            batch_timings = list(resume_state.timings) if hasattr(resume_state, "timings") else []
            if start_idx >= len(file_jobs):
                st.info("Незавершённых файлов не осталось.")
                clear_batch_resume_state()
                return
            # Берём сохранённого оператора, чтобы не зависеть от текущего выбора в UI.
            _saved_op = resume_state.operator_name.strip()
            if _saved_op:
                batch_operator_resolved = _saved_op
            batch_cloud_eval_extra = resume_state.cloud_eval_extra.strip()
        else:
            # Новый запуск: сохраняем все загруженные файлы в temp-хранилище для восстановления после сброса.
            _resume_dir = batch_resume_files_dir()
            _resume_dir.mkdir(parents=True, exist_ok=True)
            for old in _resume_dir.glob("*"):
                try:
                    old.unlink()
                except Exception:
                    pass
            for idx_f, uf in enumerate(valid_batch_files, 1):
                safe_name = re.sub(r"[^\w.\-]+", "_", uf.name, flags=re.UNICODE)
                suffix = Path(uf.name).suffix or ".wav"
                fp = _resume_dir / f"{idx_f:04d}_{safe_name}{'' if safe_name.endswith(suffix) else suffix}"
                fp.write_bytes(uf.getvalue())
                file_jobs.append(
                    BatchFileJob(
                        name=uf.name,
                        path=str(fp),
                        size=int(getattr(uf, "size", 0) or 0),
                    )
                )
            batch_cloud_eval_extra = (
                (st.session_state.get("batch_eval_extra_instructions") or "").strip()
                if cloud_mode_active and cloud_eval_cfg is not None
                else ""
            )
            save_batch_resume_state(
                BatchResumeState(
                    created_at=time.time(),
                    next_idx=0,
                    operator_name=batch_operator_resolved or "",
                    cloud_eval_extra=batch_cloud_eval_extra,
                    files=file_jobs,
                    collected=[],
                    errors={},
                    sheets_log={},
                )
            )

        batch_timer_placeholder = st.empty()
        steps_placeholder = st.empty()
        steps_placeholder.markdown(make_steps_html(-1, 0), unsafe_allow_html=True)
        file_progress = st.progress(0, text="Текущий файл: ожидание…")
        progress_bar = st.progress(0, text="Пакет: подготовка…")
        status_text = st.empty()
        live_table_placeholder = st.empty()
        with st.expander("Лог текущего файла", expanded=True):
            batch_live_logs_placeholder = st.empty()

        total = len(file_jobs)
        # Always fetch Yandex credentials from app_config so post-edit can route
        # to Yandex (fast) even when Claude is the primary evaluation backend.
        _app_cfg = load_app_config()
        _yandex_cfg = _app_cfg.yandex_cloud
        transcriber_settings = _build_gui_transcriber_settings(
            model_name=model_name,
            compute_type=compute_type,
            asr_profile=asr_profile,
            heavy_diarization=heavy_mode,
            heavy_diarization_timeout_seconds=heavy_timeout,
            enable_post_edit=enable_post_edit,
            post_edit_timeout_seconds=60,
            enable_llm_post_edit=cloud_mode_active,
            llm_backend=sidebar_state.cloud_backend,
            llm_yandex_api_key=_yandex_cfg.api_key if _yandex_cfg.configured else None,
            llm_yandex_folder_id=_yandex_cfg.folder_id if _yandex_cfg.configured else None,
            llm_yandex_model=_yandex_cfg.model,
            llm_yandex_timeout=_yandex_cfg.timeout_seconds,
            asr_backend=sidebar_state.asr_backend,
        )
        cached_transcriber = get_cached_transcriber_for_gui(transcriber_settings)

        if batch_cloud_eval_extra and cloud_eval_cfg is not None:
            st.caption("📝 Для оценки в пакете учитываются **ваши доп. комментарии** (см. поле выше / чекпоинт пакета).")

        # ── Оценка длительности аудио по размеру файла ───────────────────────
        # Эмпирический коэффициент из замеров: 979 с аудио → 1061 с обработки
        _PROC_PER_AUDIO_S: float = 1.084  # секунд обработки на секунду аудио

        def _est_audio_s(job: BatchFileJob) -> float:
            """Грубая оценка длительности аудио по размеру файла и расширению."""
            ext = Path(job.name).suffix.lower()
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
            size = int(job.size or 0)
            return max(1.0, size / bps)

        est_audio: list[float] = [_est_audio_s(f) for f in file_jobs]

        batch_ui: dict = {
            "batch_start": time.perf_counter(),
            "file_idx": 0,
            "file_start": time.perf_counter(),
            "current_progress": 0,
            # wall-clock секунды на каждый завершённый файл
            "completed_durations": [],
            # фактически отработанные секунды аудио (est) — для калибровки ratio
            "completed_audio_est": [],
            "total": len(file_jobs),
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
                    rest_str = format_eta(rem_total)

                batch_timer_placeholder.caption(
                    f"⏱ **Весь пакет:** прошло **{format_eta(elapsed)}** · "
                    f"осталось примерно **{rest_str}**"
                )

        _batch_timer_thread = threading.Thread(
            target=_batch_total_timer_loop, daemon=True, name="batch_eta_timer"
        )
        _batch_timer_thread.start()

        try:
            for idx in range(start_idx, len(file_jobs)):
                job = file_jobs[idx]
                file_name = job.name or f"file_{idx+1}"
                file_path = job.path
                batch_ui["file_idx"] = idx
                batch_ui["file_start"] = time.perf_counter()
                batch_ui["current_progress"] = 0
                progress_state = TranscriptionProgressState()
                file_ui_logs: list[str] = []

                status_text.markdown(
                    f"⏳ **Файл {idx + 1}/{total}** — `{file_name}`"
                )
                progress_bar.progress(
                    idx / total, text=f"Пакет: файл {idx + 1} из {total}"
                )
                file_progress.progress(0, text=f"«{file_name}» — старт…")
                steps_placeholder.markdown(make_steps_html(-1, 0), unsafe_allow_html=True)
                batch_live_logs_placeholder.empty()

                def _make_batch_file_logger(
                    logs_ref: list[str],
                    logs_ph,
                    st_local: TranscriptionProgressState,
                ) -> Callable[[str], None]:
                    def _cb(message: str) -> None:
                        logs_ref.append(message)
                        logs_ph.code("\n".join(logs_ref[-25:]) or "")
                        bar_text, update_steps = update_transcription_progress_from_log(
                            message, st_local
                        )
                        batch_ui["current_progress"] = st_local.current_progress
                        if bar_text is not None:
                            if update_steps:
                                steps_placeholder.markdown(
                                    make_steps_html(
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
                    if not file_path or not Path(file_path).exists():
                        raise FileNotFoundError(f"Временный файл пакета не найден: {file_name}")
                    tmp_path = Path(file_path)

                    analysis = analyze_call(
                        AnalysisRequest(
                            audio_path=str(tmp_path),
                            operator_name=batch_operator_resolved,
                            original_basename=file_name,
                            cloud_eval_cfg=cloud_eval_cfg,
                            cloud_eval_extra_instructions=(
                                batch_cloud_eval_extra if cloud_eval_cfg is not None else None
                            ),
                        ),
                        transcriber=cached_transcriber,
                        on_log=update_batch_file_log,
                    )
                    result = analysis.result
                    evaluation = analysis.evaluation
                    _file_wall = time.perf_counter() - batch_ui["file_start"]
                    batch_ui["completed_durations"].append(_file_wall)
                    # Обновляем калиброванный коэффициент proc_ratio
                    _caudio = batch_ui["est_audio"][idx] if idx < len(batch_ui["est_audio"]) else 60.0
                    batch_ui["completed_audio_est"].append(_caudio)
                    _total_proc = sum(batch_ui["completed_durations"])
                    _total_audio = sum(batch_ui["completed_audio_est"])
                    if _total_audio > 0:
                        batch_ui["proc_ratio"] = _total_proc / _total_audio
                    file_progress.progress(100, text=f"«{file_name}» — готово")
                    steps_placeholder.markdown(
                        make_steps_html(-1, 5), unsafe_allow_html=True
                    )

                    row = result_to_row_dict(file_name, result, evaluation)
                    collected.append(row)

                    try:
                        out = ensure_results_dir() / f"{Path(file_name).stem}_report.txt"
                        out.write_text(analysis.report, encoding="utf-8")
                    except Exception:
                        pass

                    if sheets_configured:
                        try:
                            ok, msg = append_analysis_row(
                                **build_sheets_export_payload(
                                    analysis,
                                    original_filename=file_name,
                                )
                            )
                            sheets_log[file_name] = (
                                "✅ Добавлено" if ok else f"⚠️ {msg[:60]}"
                            )
                        except Exception as exc:
                            sheets_log[file_name] = f"❌ {str(exc)[:60]}"

                except Exception as exc:
                    errors[file_name] = str(exc)

                # Per-file timing tracking
                _file_wall = time.perf_counter() - batch_ui["file_start"]
                timing_record = {
                    "name": file_name,
                    "duration_seconds": round(_file_wall, 1),
                    "status": "success" if file_name not in errors else "error",
                }
                if file_name not in errors and "row" in dir():
                    try:
                        timing_record["score"] = row.get("total_score", 0)
                    except Exception:
                        pass
                batch_timings.append(timing_record)

                # Чекпоинт после каждого файла: при сбросе продолжаем с места.
                save_batch_resume_state(
                    BatchResumeState(
                        created_at=time.time(),
                        next_idx=idx + 1,
                        operator_name=batch_operator_resolved or "",
                        cloud_eval_extra=batch_cloud_eval_extra,
                        files=file_jobs,
                        collected=collected,
                        errors=errors,
                        sheets_log=sheets_log,
                        timings=batch_timings,
                        batch_start_time=batch_ui["batch_start"],
                    )
                )

                progress_bar.progress(
                    (idx + 1) / total, text=f"Пакет: обработано {idx + 1} из {total}"
                )
                render_live_batch_table(
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
        # Пакет завершён — удаляем checkpoint и временные файлы резюма.
        if len(collected) + len(errors) >= total:
            clear_batch_resume_state()

        total_elapsed = time.perf_counter() - batch_ui["batch_start"]
        batch_timer_placeholder.caption(
            f"⏱ **Пакет завершён.** Всего времени: **{format_eta(total_elapsed)}**"
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
        summary += f" · ⏱ {format_eta(total_elapsed)}"
        status_text.markdown(summary)

    # ── Результаты ──────────────────────────────────────────────────────────
    collected = st.session_state.get("batch_results", [])
    errors = st.session_state.get("batch_errors", {})

    if errors:
        with st.expander(f"⚠️ Ошибки обработки ({len(errors)})", expanded=True):
            for fname, emsg in errors.items():
                st.error(
                    f"**{fname}**\n\n`{emsg}`"
                )

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
        styled = df.style.map(color_score, subset=score_cols).map(
            color_review, subset=["Прослушать"]
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
            st.caption(f"Google Таблица не настроена. {sheets_config_hint()}")

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
