from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import streamlit as st

from app_config import load_app_config
from env_store import merge_env_file, normalize_google_spreadsheet_id
from transcription import resolve_ultima_whisper_model

if TYPE_CHECKING:
    from llm_cloud_eval import CloudEvalConfig

_YANDEX_MODEL_ORDER = (
    "yandexgpt-lite",
    "yandexgpt/latest",
    "aliceai-llm/latest",
)

_YANDEX_MODEL_LABELS = {
    "yandexgpt-lite": "YandexGPT Lite (рекомендуем)",
    "yandexgpt/latest": "YandexGPT Pro",
    "aliceai-llm/latest": "Alice AI LLM",
}


@dataclass(frozen=True)
class SidebarState:
    cloud_backend: str  # "yandex" | "claude" | "off"
    cloud_eval_cfg: "CloudEvalConfig | None"
    model_name: str
    compute_type: str
    asr_profile: str
    heavy_mode: bool
    heavy_timeout: int
    enable_post_edit: bool
    asr_backend: str = "whisper"  # "whisper" | "assemblyai" | "nexara"


def render_admin_panel_sidebar() -> None:
    """Ключ Yandex, Folder ID и таблица Google — только после ввода пароля; сохранение в .env."""
    app_config = load_app_config()
    st.divider()
    with st.expander("⚙️ Админ-панель", expanded=False):
        st.caption(
            "Ключи API и ссылка/ID Google Таблицы настраиваются здесь "
            "(после входа по паролю). Изменения записываются в файл `.env`."
        )

        if not app_config.security.admin_panel_enabled:
            st.warning(
                "Админ-панель отключена: задайте `ADMIN_PANEL_PASSWORD` в `.env`, затем перезапустите приложение."
            )
            st.code("ADMIN_PANEL_PASSWORD=ваш_новый_пароль")
            st.session_state["admin_panel_unlocked"] = False
            return

        if not st.session_state.get("admin_panel_unlocked"):
            with st.form("admin_login_form", clear_on_submit=False):
                pwd_in = st.text_input("Пароль администратора", type="password", key="admin_pw_field", placeholder="Введите пароль")
                login = st.form_submit_button("Войти", type="primary")
            if login:
                if (pwd_in or "").strip() == (app_config.security.admin_panel_password or ""):
                    st.session_state["admin_panel_unlocked"] = True
                    st.rerun()
                else:
                    st.error("Неверный пароль")
            return

        yf_cur = app_config.yandex_cloud.folder_id or ""
        sheet_display = app_config.google_sheets.spreadsheet_id or ""

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
            st.markdown("**Claude (awstore)**")
            claude_configured = app_config.claude_cloud.configured
            if claude_configured:
                st.markdown(
                    '<div class="ai-connected">🟢 Claude подключён</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div class="ai-disconnected">⚪ Claude: ключ не задан</div>',
                    unsafe_allow_html=True,
                )
            new_claude_api = st.text_input(
                "Claude API Key",
                value="",
                type="password",
                placeholder="Оставьте пустым, чтобы не менять текущий ключ",
                help="Ключ API для Claude (api.awstore.cloud).",
                key="admin_claude_api",
            )
            new_claude_base = st.text_input(
                "Claude Base URL",
                value=app_config.claude_cloud.base_url,
                help="Базовый URL OpenAI-совместимого API Claude.",
                key="admin_claude_base",
            )
            st.markdown("**AssemblyAI**")
            assemblyai_configured = app_config.assemblyai.configured
            if assemblyai_configured:
                st.markdown(
                    '<div class="ai-connected">🟢 AssemblyAI подключён</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div class="ai-disconnected">⚪ AssemblyAI: ключ не задан</div>',
                    unsafe_allow_html=True,
                )
            new_assemblyai_api = st.text_input(
                "AssemblyAI API Key",
                value="",
                type="password",
                placeholder="Оставьте пустым, чтобы не менять текущий ключ",
                help="Ключ API AssemblyAI (https://www.assemblyai.com/app/account).",
                key="admin_assemblyai_api",
            )
            nexara_configured = app_config.nexara.configured
            if nexara_configured:
                st.markdown(
                    '<div class="ai-connected">🟢 Nexara подключён</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div class="ai-disconnected">⚪ Nexara: ключ не задан</div>',
                    unsafe_allow_html=True,
                )
            new_nexara_api = st.text_input(
                "Nexara API Key",
                value="",
                type="password",
                placeholder="Оставьте пустым, чтобы не менять текущий ключ",
                help="Ключ API Nexara (https://docs.nexara.ru/guides).",
                key="admin_nexara_api",
            )
            st.markdown("**Google Таблица**")
            new_sheet = st.text_input(
                "Ссылка или ID таблицы",
                value=sheet_display,
                help=(
                    "Вставьте полную ссылку вида "
                    "https://docs.google.com/spreadsheets/d/…/edit "
                    "или только ID. Если поле пустое, экспорт в Google Sheets не считается полностью настроенным."
                ),
                key="admin_sheet_url",
            )
            save = st.form_submit_button("Сохранить", type="primary")
        if save:
            updates: dict[str, str] = {}
            api_stripped = (new_api or "").strip()
            if api_stripped:
                updates["YANDEX_API_KEY"] = api_stripped
            folder_stripped = (new_folder or "").strip()
            if folder_stripped:
                updates["YANDEX_FOLDER_ID"] = folder_stripped
            claude_api_stripped = (new_claude_api or "").strip()
            if claude_api_stripped:
                updates["CLAUDE_API_KEY"] = claude_api_stripped
            claude_base_stripped = (new_claude_base or "").strip()
            if claude_base_stripped:
                updates["CLAUDE_BASE_URL"] = claude_base_stripped
            assemblyai_stripped = (new_assemblyai_api or "").strip()
            if assemblyai_stripped:
                updates["ASSEMBLYAI_API_KEY"] = assemblyai_stripped
            nexara_stripped = (new_nexara_api or "").strip()
            if nexara_stripped:
                updates["NEXARA_API_KEY"] = nexara_stripped
            sid = normalize_google_spreadsheet_id(new_sheet)
            if sid:
                updates["GOOGLE_SHEETS_SPREADSHEET_ID"] = sid
            if not updates:
                st.warning("Нечего сохранить: укажите новый API Key и/или проверьте Folder ID и таблицу.")
            else:
                try:
                    merge_env_file(updates)
                    updated_keys = set(updates)
                    if updated_keys & {
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
                    if updated_keys & {"YANDEX_API_KEY", "YANDEX_FOLDER_ID"}:
                        st.cache_resource.clear()
                    if updated_keys & {"CLAUDE_API_KEY", "CLAUDE_BASE_URL"}:
                        st.cache_resource.clear()
                    if "ASSEMBLYAI_API_KEY" in updated_keys:
                        st.cache_resource.clear()
                    if "NEXARA_API_KEY" in updated_keys:
                        st.cache_resource.clear()
                    st.success("Сохранено. Переменные обновлены для текущего сеанса.")
                    st.rerun()
                except OSError as exc:
                    st.error(f"Не удалось записать `.env`: {exc}")

        if st.button("Выйти из админ-панели", key="admin_logout_btn"):
            st.session_state["admin_panel_unlocked"] = False
            st.rerun()


def _resolve_transcription_profile(
    quality: str,
) -> tuple[str, str, str, bool]:
    if quality == "standard":
        return "medium", "int8", "medium_ru", False
    if quality == "ultima":
        return resolve_ultima_whisper_model(), "int8_float32", "ultima_ru", True
    return "large-v3", "int8_float32", "ideal_ru", True


def render_sidebar(app_version_label: str, app_version_date: str) -> SidebarState:
    """Рисует сайдбар и возвращает выбранные настройки анализа."""
    app_config = load_app_config()
    with st.sidebar:
        st.markdown(
            f"""
<div class="sidebar-logo">
  <div class="sidebar-logo-icon">📞</div>
  <div>
    <div class="sidebar-logo-text">Анализ звонков</div>
    <div class="sidebar-logo-sub">ДВФУ · ЕКЦ</div>
    <div class="sidebar-logo-version">{app_version_label} · {app_version_date}</div>
  </div>
</div>""",
            unsafe_allow_html=True,
        )

        # Determine which backends are configured
        yandex_configured = app_config.yandex_cloud.configured
        claude_configured = app_config.claude_cloud.configured

        # Default backend: prefer Claude if configured, then Yandex, then off
        _default_backend = "off"
        if yandex_configured or claude_configured:
            if claude_configured:
                _default_backend = "claude"
            elif yandex_configured:
                _default_backend = "yandex"

        st.markdown("**🤖 AI-анализ**")
        cloud_backend = st.radio(
            "Бэкенд",
            options=["yandex", "claude", "off"],
            format_func=lambda v: {
                "yandex": "Yandex AI Studio",
                "claude": "Claude (awstore)",
                "off": "Отключить",
            }[v],
            horizontal=True,
            index=["yandex", "claude", "off"].index(_default_backend),
            help="Yandex AI Studio или Claude через api.awstore.cloud для оценки качества звонков.",
        )

        cloud_eval_cfg: "CloudEvalConfig | None" = None

        if cloud_backend == "yandex":
            env_api_key = app_config.yandex_cloud.api_key or ""
            env_folder_id = app_config.yandex_cloud.folder_id or ""
            yandex_timeout = int(app_config.yandex_cloud.timeout_seconds)

            if yandex_configured:
                st.markdown(
                    '<div class="ai-connected">🟢 Yandex AI Studio подключён</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div class="ai-disconnected">⚠️ Ключи не заданы — откройте Админ-панель ниже</div>',
                    unsafe_allow_html=True,
                )

            if "yandex_cloud_model_lp" not in st.session_state:
                st.session_state["yandex_cloud_model_lp"] = app_config.yandex_cloud.model
            if st.session_state["yandex_cloud_model_lp"] not in _YANDEX_MODEL_ORDER:
                st.session_state["yandex_cloud_model_lp"] = "yandexgpt-lite"
            yandex_model_key = st.radio(
                "Модель",
                options=list(_YANDEX_MODEL_ORDER),
                format_func=lambda key: _YANDEX_MODEL_LABELS[key],
                horizontal=True,
                key="yandex_cloud_model_lp",
                help=(
                    "Тот же API-ключ Yandex AI Studio. "
                    "Lite — лучший баланс цены и качества для оценки русскоязычных звонков; "
                    "Pro — для самых сложных кейсов, когда важнее максимум точности; "
                    "Alice AI LLM — отдельная модель Yandex для экспериментов и сравнения качества."
                ),
            )

            with st.expander("⚡ Дополнительно", expanded=False):
                yandex_timeout = st.slider(
                    "Таймаут запроса (сек)",
                    min_value=15,
                    max_value=300,
                    value=yandex_timeout,
                    step=5,
                    help="Для длинных звонков при необходимости увеличьте (до 120 с и выше).",
                )

            if env_api_key.strip() and env_folder_id.strip():
                from llm_cloud_eval import YandexCloudConfig

                cloud_eval_cfg = YandexCloudConfig(
                    api_key=env_api_key.strip(),
                    folder_id=env_folder_id.strip(),
                    model=yandex_model_key,
                    timeout_seconds=float(yandex_timeout),
                )
            else:
                st.warning("Задайте API Key и Folder ID в Админ-панели ниже.")

        elif cloud_backend == "claude":
            claude_api_key = app_config.claude_cloud.api_key or ""
            claude_base_url = app_config.claude_cloud.base_url
            claude_model = app_config.claude_cloud.model
            claude_timeout = int(app_config.claude_cloud.timeout_seconds)

            if claude_configured:
                st.markdown(
                    '<div class="ai-connected">🟢 Claude подключён</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div class="ai-disconnected">🟡 Ключ не задан — укажите CLAUDE_API_KEY в «Админ-панели» ниже</div>',
                    unsafe_allow_html=True,
                )

            with st.expander("⚡ Дополнительно", expanded=False):
                claude_timeout = st.slider(
                    "Таймаут запроса (сек)",
                    min_value=15,
                    max_value=300,
                    value=claude_timeout,
                    step=5,
                    help="Для длинных звонков при необходимости увеличьте.",
                )

            if claude_api_key.strip():
                from llm_cloud_eval import ClaudeCloudConfig

                cloud_eval_cfg = ClaudeCloudConfig(
                    api_key=claude_api_key.strip(),
                    base_url=claude_base_url,
                    model=claude_model,
                    timeout_seconds=float(claude_timeout),
                )
            else:
                st.warning("Задайте API Key в Админ-панели ниже.")

        else:
            st.markdown(
                '<div class="ai-disconnected">⚪ AI отключён — только локальная эвристика</div>',
                unsafe_allow_html=True,
            )

        st.divider()
        st.markdown("**🎙️ Распознавание речи**")

        # ASR backend selector
        asr_backend = st.radio(
            "Бэкенд ASR",
            options=["whisper", "assemblyai", "nexara"],
            format_func=lambda v: {
                "whisper": "Whisper (локальный)",
                "assemblyai": "AssemblyAI (облачный)",
                "nexara": "Nexara (облачный)",
            }[v],
            horizontal=True,
            help=(
                "Whisper: локальная модель, полная настройка VAD и hotwords. "
                "AssemblyAI: облачное распознавание с автоматической диаризацией; "
                "не требует загрузки моделей, но нужен API-ключ. "
                "Nexara: облачное распознавание с диаризацией; экспериментальная модель."
            ),
        )

        if asr_backend == "assemblyai":
            if app_config.assemblyai.configured:
                st.markdown(
                    '<div class="ai-connected">🟢 AssemblyAI подключён</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div class="ai-disconnected">⚪ AssemblyAI: ключ не задан (укажите ASSEMBLYAI_API_KEY в Админ-панели)</div>',
                    unsafe_allow_html=True,
                )

        if asr_backend == "nexara":
            if app_config.nexara.configured:
                st.markdown(
                    '<div class="ai-connected">🟢 Nexara подключён</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div class="ai-disconnected">⚪ Nexara: ключ не задан (укажите NEXARA_API_KEY в Админ-панели)</div>',
                    unsafe_allow_html=True,
                )

        quality = st.radio(
            "Качество",
            options=["ultima", "max", "standard"],
            format_func=lambda value: {
                "max": "🎯 Максимум (~12–22 мин)",
                "standard": "⚡ Стандарт (~1–3 мин)",
                "ultima": "🔮 Ultima (~7–12 мин, RU Large-v3)",
            }[value],
            horizontal=True,
            disabled=(asr_backend in ("assemblyai", "nexara")),
            help=(
                "Стандарт: Whisper Medium, int8; быстро, достаточно для большинства звонков. "
                "Максимум: Systran Large-v3 + beam 9/best_of 5 + 3 температуры; максимальное качество. "
                "Ultima: fine-tuned whisper-large-v3-russian; локальный auto fast-path для чистых звонков "
                "с авто-откатом на полный decode при сомнительном результате; "
                "спектральный денойз включается адаптивно (SNR↑ для тихой/шумной речи); "
                "сверхчувствительный VAD pad=2000ms (не режет первые слова); "
                "no_speech_threshold=0.82, log_prob=-1.2 (декодирует даже тихие вступления); "
                "плотная диаризация: окна 2.4с/шаг 0.6с, взвешенный vote. На Mac без GPU — int8_float32."
            ),
        )
        model_name, compute_type, asr_profile, heavy_mode = _resolve_transcription_profile(quality)

        with st.expander("⚡ Тонкая настройка", expanded=False):
            heavy_timeout = st.slider(
                "Таймаут диаризации (сек)",
                min_value=90,
                max_value=300,
                value=150,
                step=10,
                disabled=(asr_backend in ("assemblyai", "nexara")),
            )
            enable_post_edit = st.checkbox(
                "Локальный пунктуатор",
                value=False,
                help="Расставляет знаки препинания без интернета. Обычно не нужен с Яндекс AI.",
            )

        render_admin_panel_sidebar()

    return SidebarState(
        cloud_backend=cloud_backend,
        cloud_eval_cfg=cloud_eval_cfg,
        model_name=model_name,
        compute_type=compute_type,
        asr_profile=asr_profile,
        heavy_mode=heavy_mode,
        heavy_timeout=heavy_timeout,
        enable_post_edit=enable_post_edit,
        asr_backend=asr_backend,
    )
