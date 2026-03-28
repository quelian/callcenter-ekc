from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import streamlit as st

from app_config import load_app_config
from env_store import merge_env_file, normalize_google_spreadsheet_id
from transcription import resolve_ultima_whisper_model

if TYPE_CHECKING:
    from llm_cloud_eval import YandexCloudConfig


@dataclass(frozen=True)
class SidebarState:
    use_yandex: bool
    cloud_eval_cfg: YandexCloudConfig | None
    model_name: str
    compute_type: str
    asr_profile: str
    heavy_mode: bool
    heavy_timeout: int
    enable_post_edit: bool


def render_admin_panel_sidebar() -> None:
    """Ключ Yandex, Folder ID и таблица Google — только после ввода пароля; сохранение в .env."""
    app_config = load_app_config()
    st.divider()
    with st.expander("🔐 Админ-панель", expanded=False):
        st.caption(
            "Ключ API Яндекса и ссылка/ID Google Таблицы можно менять **только здесь** "
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
                pwd_in = st.text_input("Пароль администратора", type="password", key="admin_pw_field")
                login = st.form_submit_button("Войти")
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

        env_api_key = app_config.yandex_cloud.api_key or ""
        env_folder_id = app_config.yandex_cloud.folder_id or ""
        env_keys_set = bool(env_api_key and env_folder_id)

        st.markdown("**Яндекс AI Studio**")
        use_yandex = st.toggle(
            "Включить AI-анализ",
            value=env_keys_set,
            help="Включает облачную оценку и исправление транскрипта через Яндекс AI.",
        )

        cloud_eval_cfg: YandexCloudConfig | None = None
        yandex_api_key = env_api_key
        yandex_folder_id = env_folder_id
        yandex_model_order = ("yandexgpt-lite", "yandexgpt/latest")
        yandex_model_labels = {
            "yandexgpt-lite": "YandexGPT Lite (по умолчанию)",
            "yandexgpt/latest": "YandexGPT Pro",
        }
        # При выключенном AI значение не используется для запросов.
        yandex_timeout = int(app_config.yandex_cloud.timeout_seconds)

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

            if "yandex_cloud_model_lp" not in st.session_state:
                st.session_state["yandex_cloud_model_lp"] = "yandexgpt-lite"
            yandex_model_key = st.radio(
                "Модель",
                options=list(yandex_model_order),
                format_func=lambda key: yandex_model_labels[key],
                horizontal=True,
                key="yandex_cloud_model_lp",
                help="Тот же API-ключ Yandex AI Studio. Lite — быстрее; Pro — точнее.",
            )

            with st.expander("Дополнительно", expanded=False):
                yandex_timeout = st.slider(
                    "Таймаут запроса (сек)",
                    min_value=15,
                    max_value=300,
                    value=yandex_timeout,
                    step=5,
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
            options=["ultima", "max", "standard"],
            format_func=lambda value: {
                "max": "🎯 Максимум (~12–22 мин)",
                "standard": "⚡ Стандарт (~1–3 мин)",
                "ultima": "🔮 Ultima (~7–12 мин, RU Large-v3)",
            }[value],
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
        model_name, compute_type, asr_profile, heavy_mode = _resolve_transcription_profile(quality)

        with st.expander("Тонкая настройка", expanded=False):
            heavy_timeout = st.slider(
                "Таймаут диаризации (сек)",
                min_value=90,
                max_value=300,
                value=150,
                step=10,
            )
            enable_post_edit = st.checkbox(
                "Локальный пунктуатор",
                value=False,
                help="Расставляет знаки препинания без интернета. Обычно не нужен с Яндекс AI.",
            )

        render_admin_panel_sidebar()

    return SidebarState(
        use_yandex=use_yandex,
        cloud_eval_cfg=cloud_eval_cfg,
        model_name=model_name,
        compute_type=compute_type,
        asr_profile=asr_profile,
        heavy_mode=heavy_mode,
        heavy_timeout=heavy_timeout,
        enable_post_edit=enable_post_edit,
    )
