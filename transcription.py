from __future__ import annotations

import argparse
import atexit
import inspect
import os
import sys
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable

if sys.platform == "darwin":
    os.environ["MallocStackLogging"] = "0"

# Один поток: не копим фоновые punctuate после wall-timeout; следующая задача ждёт слот.
_POST_EDIT_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="punct_pe")


def _shutdown_post_edit_executor() -> None:
    try:
        _POST_EDIT_EXECUTOR.shutdown(wait=False)
    except Exception:
        pass


atexit.register(_shutdown_post_edit_executor)

# Те же фильтры шумных варнингов, что в Streamlit (torch STFT, SpeechBrain и т.д.).
try:
    from speechbrain_compat import suppress_noisy_third_party_warnings

    suppress_noisy_third_party_warnings()
except Exception:
    pass

from app_config import load_app_environment
from hf_hub_sync import HF_HUB_DOWNLOAD_LOCK, force_huggingface_hub_online
from operator_staff import OPERATOR_ALIASES
from audio_io_safe import is_probably_compressed_audio, load_audio_mono_16k_safe
from role_misattribution_fix import fix_operator_thanks_mislabeled_as_applicant
from role_semantic_refine import _semantic_refine_enabled, try_semantic_role_refine

if TYPE_CHECKING:
    import numpy as np


@dataclass
class TranscriptionResult:
    text: str
    language: str | None = None
    role_text: str = ""
    tone_summary: str = "Не определен"
    role_attribution: str = ""
    role_confidence: str = "n/a"
    role_diagnostics: str = ""
    asr_profile: str = "medium_ru"
    asr_model: str = "unknown"


@dataclass
class QualityEvaluation:
    total_score: int
    max_score: int
    operator_name: str
    operator_in_staff: bool
    applicant_name: str | None
    script_score: int
    speech_score: int
    consultation_score: int
    engagement_score: int
    script_details: list[str]
    positives: list[str]
    negatives: list[str]


# Имя файла: `дата_телефон_MM-SS.ext` — последний блок после `_` это ожидание: минуты-секунды.
_FILENAME_WAIT_TAIL_RE = re.compile(r"^(\d{2})-(\d{2})$")


def parse_call_filename_wait_skip(filename: str) -> tuple[float, str] | None:
    """
    Извлекает длительность ожидания из конца имени файла.

    Пример: ``12-02-2026_7 (902) 556-73-27_01-12.mp3`` → хвост ``01-12`` = 1 мин 12 с → 72.0 с.

    Возвращает (секунды_пропуска, как_в_имени) или None, если шаблон не распознан.
    """
    stem = Path(filename).stem.strip()
    if "_" not in stem:
        return None
    tail = stem.rsplit("_", 1)[-1].strip()
    m = _FILENAME_WAIT_TAIL_RE.match(tail)
    if not m:
        return None
    minutes = int(m.group(1))
    seconds = int(m.group(2))
    if seconds >= 60 or minutes < 0 or minutes > 180:
        return None
    total = float(minutes * 60 + seconds)
    return total, tail


def should_attempt_intro_recovery(
    pieces: list[tuple[float, float, str]],
    *,
    audio_duration: float,
    duration_after_vad: float,
) -> bool:
    """Нужно ли делать rescue-pass для начала звонка.

    Сигналы проблемы:
    - первое распознанное слово появляется заметно позже старта;
    - в первых секундах почти нет слов;
    - VAD отрезал слишком большую долю файла.
    """
    full_dur = max(0.0, float(audio_duration))
    vad_dur = max(0.0, float(duration_after_vad))
    if full_dur < 16.0:
        return False
    if not pieces:
        return True

    first_start = max(0.0, float(pieces[0][0]))
    early_words = 0
    early_speech = 0.0
    for start, end, text in pieces:
        if float(start) >= 12.0:
            continue
        early_words += len(str(text).split())
        early_speech += max(0.0, min(float(end), 12.0) - max(0.0, float(start)))

    if first_start >= 2.8:
        return True
    if first_start >= 1.8 and early_words <= 3:
        return True
    if early_words <= 2 and early_speech < 1.4 and full_dur >= 22.0:
        return True
    if vad_dur < full_dur * 0.16 and full_dur >= 25.0:
        return True
    return False


# ideal_ru (Максимум) и ultima_ru (Ultima): hotwords для смещения декодера на русскую лексику ЕКЦ и телефонного общения.
_IDEAL_RU_RU_HOTWORDS = (
    "ДВФУ Владивосток университет деканат институт факультет кафедра студент аспирант абитуриент "
    "зачётная книжка расписание экзамен сессия общежитие оплата стипендия льготы документы справка "
    "заявление договор приёмная комиссия зачисление перевод куратор военная кафедра "
    "здравствуйте добрый день добрый вечер алло слушаю слышу вас чем помочь подскажите пожалуйста "
    "уточню перезвоню спасибо благодарю за обращение заявитель оператор линия ожидание"
)

# Доп. якоря только для ultima_ru (не раздуваем hotwords у ideal_ru).
_ULTIMA_RU_HOTWORDS_EXTRA = (
    "госуслуги личный кабинет заочное очное отделение академический отпуск отчисление "
    "восстановление заселение общежитие проживание пропуск деканат приём документов "
    "номер заявления статус обращения очередь соединяю перевожу на специалиста"
)

# «Ultima»: CTranslate2-сборка fine-tuned whisper-large-v3-russian (см. Hugging Face).
# Переопределение: CALLQA_WHISPER_MODEL_ULTIMA=org/name
DEFAULT_ULTIMA_WHISPER_REPO = "bzikst/faster-whisper-large-v3-russian"

# Профили large: общий чувствительный VAD + hotwords. ultima_ru — отдельный decode/температуры/денойз под RU FT.
_PREMIUM_ASR_PROFILES = frozenset({"ideal_ru", "ultima_ru"})


def resolve_ultima_whisper_model() -> str:
    raw = os.environ.get("CALLQA_WHISPER_MODEL_ULTIMA", DEFAULT_ULTIMA_WHISPER_REPO).strip()
    return raw or DEFAULT_ULTIMA_WHISPER_REPO


_punct_deepmulti_compat_applied = False


def _apply_deepmultilingual_transformers_compat() -> None:
    """
    deepmultilingualpunctuation вызывает pipeline(..., grouped_entities=False).
    В актуальных transformers вместо этого — aggregation_strategy=\"none\".
    """
    global _punct_deepmulti_compat_applied
    if _punct_deepmulti_compat_applied:
        return
    try:
        import deepmultilingualpunctuation.punctuationmodel as pmp
    except ImportError:
        return

    import torch
    from transformers import pipeline

    def _patched_punctuation_init(self, model: str = "oliverguhr/fullstop-punctuation-multilang-large") -> None:
        if torch.cuda.is_available():
            try:
                self.pipe = pipeline("ner", model, aggregation_strategy="none", device=0)
            except TypeError:
                self.pipe = pipeline("ner", model, grouped_entities=False, device=0)
        else:
            try:
                self.pipe = pipeline("ner", model, aggregation_strategy="none")
            except TypeError:
                self.pipe = pipeline("ner", model, grouped_entities=False)

    pmp.PunctuationModel.__init__ = _patched_punctuation_init  # type: ignore[method-assign]
    _punct_deepmulti_compat_applied = True


def _frame_rms_curve(y: "object", *, sample_rate: int, frame_ms: float = 40.0, hop_ms: float = 20.0) -> tuple[object, int, int]:
    """RMS по коротким окнам для оценки уровня речи (не голой тишины)."""
    import numpy as np

    sig = np.asarray(y, dtype=np.float32).reshape(-1)
    if sig.size == 0:
        return np.array([], dtype=np.float64), 0, 0
    wl = max(256, int(float(sample_rate) * frame_ms / 1000.0))
    hop = max(64, int(float(sample_rate) * hop_ms / 1000.0))
    n_fr = 1 + (sig.size - wl) // hop
    if n_fr < 1:
        r = float(np.sqrt(np.mean(sig * sig) + 1e-18))
        return np.array([r], dtype=np.float64), wl, hop
    out = np.empty(n_fr, dtype=np.float64)
    for i in range(n_fr):
        sl = sig[i * hop : i * hop + wl]
        out[i] = float(np.sqrt(np.mean(sl * sl) + 1e-18))
    return out, wl, hop


def _estimate_active_speech_rms(fr_rms: "object") -> float:
    """Уровень «активной» речи: медиана громких кадров относительно шумового подпора."""
    import numpy as np

    fr = np.asarray(fr_rms, dtype=np.float64)
    if fr.size == 0:
        return 1e-8
    if fr.size < 4:
        return float(np.median(fr) + 1e-12)
    p15 = float(np.percentile(fr, 15))
    p50 = float(np.percentile(fr, 50))
    p85 = float(np.percentile(fr, 85))
    # Порог: выше типичного «фона» в тишине линии, но не только самые всплески.
    thresh = max(p15 * 3.0, p50 * 0.55, 1e-6)
    loud = fr[fr >= thresh]
    if loud.size >= max(3, fr.size // 8):
        return float(np.median(loud) + 1e-12)
    return float(np.median(fr) + 1e-12)


def _apply_cosine_fades_edges(
    y: "object",
    *,
    sample_rate: int,
    fade_in_ms: float = 5.0,
    fade_out_ms: float = 5.0,
) -> object:
    """Косинусные фейды по краям. fade_in_ms≤0 — не трогаем начало (важно после pre-roll: не глушить первый слог)."""
    import numpy as np

    sig = np.asarray(y, dtype=np.float32).reshape(-1)
    if sig.size < 32:
        return sig
    out = sig.copy()
    if fade_in_ms > 0:
        n_in = int(float(sample_rate) * fade_in_ms / 1000.0)
        n_in = max(8, min(n_in, sig.size // 8))
        w_in = (0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, n_in, dtype=np.float32))).astype(np.float32)
        out[:n_in] *= w_in
    if fade_out_ms > 0:
        n_out = int(float(sample_rate) * fade_out_ms / 1000.0)
        n_out = max(8, min(n_out, sig.size // 8))
        w_out = (0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, n_out, dtype=np.float32))).astype(np.float32)
        out[-n_out:] *= w_out[::-1]
    return out


def _normalize_audio_for_asr(
    y: object,
    *,
    sample_rate: int,
    log: Callable[[str], None] | None = None,
    target_active_rms: float = 0.11,
    max_gain_linear: float = 45.0,
    peak_limit: float = 0.989,
    fade_in_ms: float = 5.0,
    fade_out_ms: float = 5.0,
) -> object:
    """Нормализация **уже после обрезки ожидания**: DC, уровень по речевым кадрам, пик-лимит, фейды.

    Тишина и ожидание в расчёте уровня почти не участвуют — не раздуваем шум до целевого RMS всего файла.
    """
    import numpy as np

    arr = np.asarray(y, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return arr
    # Постоянная составляющая после обрезки
    arr = (arr - float(np.mean(arr))).astype(np.float32)
    full_rms = float(np.sqrt(np.mean(np.square(arr)) + 1e-18))
    fr_curve, _, _ = _frame_rms_curve(arr, sample_rate=sample_rate)
    active_rms = _estimate_active_speech_rms(fr_curve)
    gain = min(target_active_rms / active_rms, max_gain_linear)
    out = (arr * float(gain)).astype(np.float32)
    peak_b = float(np.max(np.abs(out)) + 1e-18)
    if peak_b > peak_limit:
        out = (out * (peak_limit / peak_b)).astype(np.float32)
    peak_a = float(np.max(np.abs(out)) + 1e-18)
    out = np.asarray(
        _apply_cosine_fades_edges(
            out, sample_rate=sample_rate, fade_in_ms=fade_in_ms, fade_out_ms=fade_out_ms
        ),
        dtype=np.float32,
    )
    if log is not None:
        log(
            f"Нормализация (хвост после ожидания): полный RMS {full_rms:.5f} → активный~{active_rms:.5f}, "
            f"усиление ×{gain:.2f}, пик {peak_b:.3f}→{peak_a:.3f} (лимит {peak_limit:.3f})."
        )
    return out


def _spectral_denoise_audio(
    y: "np.ndarray",
    sr: int,
    *,
    noise_percentile: float = 12.0,
    over_subtraction: float = 2.2,
    spectral_floor: float = 0.04,
    log: "Callable[[str], None] | None" = None,
) -> "np.ndarray":
    """Спектральное шумоподавление (spectral subtraction + Wiener floor).

    Алгоритм:
    1. STFT → мощностный спектр.
    2. Оценка шума: медиана спектра по кадрам с наименьшей энергией (нижний квантиль).
    3. Вычитание шума с «полом» (spectral_floor) — речь не подавляется ниже floor×signal.
    4. iSTFT → восстановленный сигнал.

    Улучшает распознавание тихой/шумной речи: Whisper получает сигнал с лучшим SNR.
    """
    import numpy as np

    try:
        from scipy.signal import stft, istft
    except Exception:
        return y

    arr = np.asarray(y, dtype=np.float32)
    if arr.size < 1600:
        return arr

    nperseg = 512
    noverlap = 384  # 75% перекрытие — точнее огибающая шума

    try:
        _, _, Zxx = stft(arr, fs=sr, nperseg=nperseg, noverlap=noverlap)
        power = np.abs(Zxx) ** 2

        frame_energies = np.mean(power, axis=0)
        energy_threshold = np.percentile(frame_energies, noise_percentile)
        noise_mask = frame_energies <= energy_threshold
        if noise_mask.sum() < 3:
            return arr
        noise_psd = np.mean(power[:, noise_mask], axis=1, keepdims=True)

        # Wiener-like gain: вычитаем шум с floor
        clean_power = np.maximum(
            power - over_subtraction * noise_psd,
            spectral_floor * power,
        )
        gain = np.sqrt(clean_power / (power + 1e-12))
        Zxx_clean = Zxx * gain

        _, y_clean = istft(Zxx_clean, fs=sr, nperseg=nperseg, noverlap=noverlap)
        y_clean = y_clean.astype(np.float32)

        # Выравниваем длину (stft/istft могут добавить/убрать несколько сэмплов)
        if len(y_clean) > len(arr):
            y_clean = y_clean[: len(arr)]
        elif len(y_clean) < len(arr):
            y_clean = np.pad(y_clean, (0, len(arr) - len(y_clean)), mode="constant")

        if log is not None:
            before_snr = float(np.mean(power))
            after_snr = float(np.mean(clean_power))
            log(
                f"Спектральный денойз: SNR до={before_snr:.4e}, после={after_snr:.4e} "
                f"(подавление ×{over_subtraction}, floor={spectral_floor})"
            )
        return y_clean
    except Exception as exc:
        if log is not None:
            log(f"Спектральный денойз пропущен ({exc}), используем исходный сигнал.")
        return arr


def _prepare_asr_normalized_wav(
    source_path: str,
    skip_seconds: float,
    transcribe_kwargs: dict[str, object],
    log: Callable[[str], None],
    *,
    target_active_rms: float = 0.11,
    denoise: bool = False,
    denoise_kwargs: dict[str, object] | None = None,
) -> tuple[str, float, dict[str, object], str | None]:
    """16 kHz mono WAV для Whisper: обрезка ожидания + опциональный **pre-roll** (см. docs/ASR_START_ANALYSIS.md).

    При ``skip_seconds > 0`` грузим с ``offset = skip - pre_roll`` (хвост ожидания перед точкой разреза).
    Таймкоды Whisper отсчитываются от начала WAV → ``time_offset = load_offset`` (не ``skip``).

    ``clip_timestamps`` из kwargs удаляется.
    """
    import tempfile

    try:
        import numpy as np
        import soundfile as sf
    except Exception as exc:
        log(f"Подготовка ASR WAV пропущена (нет numpy/soundfile: {exc}).")
        return source_path, 0.0, transcribe_kwargs, None

    off = max(0.0, float(skip_seconds))
    try:
        pr_env = os.environ.get("CALLQA_ASR_PRE_ROLL_SECONDS", "0.25").strip() or "0.25"
        pre_roll = max(0.0, min(float(pr_env), 2.0, off))
    except ValueError:
        pre_roll = min(0.25, off)
    load_offset = max(0.0, off - pre_roll) if off > 0 else 0.0
    try:
        y, sr = load_audio_mono_16k_safe(
            source_path,
            offset=load_offset,
            duration=None,
            log=log,
        )
    except Exception as exc:
        log(f"Подготовка ASR: не удалось загрузить аудио ({exc}), исходный файл.")
        if is_probably_compressed_audio(source_path):
            raise RuntimeError(
                "Не удалось декодировать сжатое аудио для ASR (нужен рабочий ffmpeg). "
                f"Исходная ошибка: {exc}"
            ) from exc
        return source_path, 0.0, transcribe_kwargs, None

    y = np.asarray(y, dtype=np.float32)
    if y.size == 0:
        log(
            "Подготовка ASR: после обрезки начала файл пуст (offset ≥ длины?) — исходный файл без обрезки."
        )
        if is_probably_compressed_audio(source_path):
            raise RuntimeError(
                "После обрезки по ожиданию сжатое аудио пустое — уменьшите время ожидания или проверьте файл."
            )
        return source_path, 0.0, transcribe_kwargs, None

    if off > 0 and pre_roll > 0:
        log(
            f"ASR pre-roll: в WAV добавлено ~{pre_roll:.2f} с аудио *до* точки ожидания "
            f"(загрузка с {load_offset:.2f} с, не с {off:.2f}) — стабильнее VAD/первое окно Whisper. "
            f"time_offset={load_offset:.2f} с. Отключить: CALLQA_ASR_PRE_ROLL_SECONDS=0"
        )

    # Спектральный денойз (только когда запрошен, напр. Ultima): улучшает SNR для тихой/шумной речи.
    if denoise:
        dk = denoise_kwargs or {}
        y = _spectral_denoise_audio(y, int(sr), log=log, **dk)

    after_trim_rms = float(np.sqrt(np.mean(np.square(y))) + 1e-12)
    # После pre-roll начало — не жёсткий ноль: не глушим вход косинусом (иначе тихий «алло»).
    fi = 0.0 if off > 0 else 5.0
    y_n = np.asarray(
        _normalize_audio_for_asr(
            y, sample_rate=int(sr), log=log,
            target_active_rms=target_active_rms,
            fade_in_ms=fi, fade_out_ms=5.0,
        ),
        dtype=np.float32,
    )
    after_norm_rms = float(np.sqrt(np.mean(np.square(y_n))) + 1e-12)

    fd, tmp = tempfile.mkstemp(suffix=".wav", prefix="callqa_asr_norm_")
    os.close(fd)
    try:
        sf.write(tmp, y_n, int(sr))
    except Exception as exc:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        log(f"Подготовка ASR: запись временного WAV не удалась ({exc}).")
        return source_path, 0.0, transcribe_kwargs, None

    kw = dict(transcribe_kwargs)
    kw.pop("clip_timestamps", None)
    if off > 0:
        log(
            f"Для ASR: ожидание по метке {off:.1f} с; фактическая загрузка с {load_offset:.2f} с, "
            f"длина WAV {len(y_n) / float(sr):.1f} с (RMS до/после норм. ~{after_trim_rms:.4f} / ~{after_norm_rms:.4f})."
        )
    else:
        log(
            f"Аудио для ASR без обрезки ожидания: {len(y_n) / float(sr):.1f} с "
            f"(RMS до/после норм. ~{after_trim_rms:.4f} / ~{after_norm_rms:.4f})."
        )
    return tmp, load_offset, kw, tmp


class Transcriber:
    _punct_model = None

    def __init__(
        self,
        model_name: str = "medium",
        compute_type: str = "int8",
        skip_first_seconds: float | None = None,
        language: str = "ru",
        model_load_timeout_seconds: int = 600,
        asr_profile: str = "medium_ru",
        heavy_diarization: bool = False,
        heavy_diarization_timeout_seconds: int = 240,
        enable_post_edit: bool = False,
        post_edit_timeout_seconds: int = 60,
        total_time_budget_seconds: int = 300,
        enable_llm_post_edit: bool = False,
        llm_backend: str = "off",
        llm_embedded_model_id: str | None = None,
        llm_post_edit_timeout_seconds: float = 120.0,
        llm_post_edit_base_url: str | None = None,
        llm_post_edit_model: str | None = None,
        llm_post_edit_api_key: str | None = None,
        llm_yandex_api_key: str | None = None,
        llm_yandex_folder_id: str | None = None,
        llm_yandex_model: str = "yandexgpt-lite",
    ) -> None:
        self.model_name = self._resolve_model_name(model_name)
        self.compute_type = compute_type
        # None = только пропуск из имени файла (_MM-SS ожидание) или с начала; иначе ручной clip
        self.skip_first_seconds = skip_first_seconds
        self.language = "ru"
        self.model_load_timeout_seconds = model_load_timeout_seconds
        self.asr_profile = asr_profile.strip().lower()
        self.heavy_diarization = heavy_diarization
        self.heavy_diarization_timeout_seconds = heavy_diarization_timeout_seconds
        self.enable_post_edit = enable_post_edit
        self.post_edit_timeout_seconds = post_edit_timeout_seconds
        self.total_time_budget_seconds = total_time_budget_seconds
        self.enable_llm_post_edit = enable_llm_post_edit
        self.llm_backend = (llm_backend or "off").strip().lower()
        self.llm_embedded_model_id = llm_embedded_model_id
        self.llm_post_edit_timeout_seconds = float(llm_post_edit_timeout_seconds)
        self.llm_post_edit_base_url = llm_post_edit_base_url
        self.llm_post_edit_model = llm_post_edit_model
        self.llm_post_edit_api_key = llm_post_edit_api_key
        self.llm_yandex_api_key = llm_yandex_api_key
        self.llm_yandex_folder_id = llm_yandex_folder_id
        self.llm_yandex_model = llm_yandex_model or "yandexgpt-lite"
        self._model = None
        self._asr_runtime_device = ""
        self._asr_runtime_compute = ""
        # Large / HF-модели долго качаются; «Максимум» / «Ultima» + тяжёлая диаризация — дольше общий бюджет
        if self.asr_profile in _PREMIUM_ASR_PROFILES:
            self.model_load_timeout_seconds = max(int(self.model_load_timeout_seconds), 1200)
            self.total_time_budget_seconds = max(int(self.total_time_budget_seconds), 1200)

    def _iter_whisper_load_attempts(self, effective_model: str) -> list[tuple[str, str]]:
        """
        Пары (device, compute_type) для faster-whisper/CTranslate2.
        float16 на CPU (macOS без CUDA) обычно не работает — не предлагаем его на cpu.
        """
        req = (self.compute_type or "default").strip().lower()
        cuda = False
        try:
            import ctranslate2 as ct2

            cuda = ct2.get_cuda_device_count() > 0
        except Exception:
            pass
        attempts: list[tuple[str, str]] = []

        def _add(dev: str, ctype: str) -> None:
            t = (dev, ctype)
            if t not in attempts:
                attempts.append(t)

        if cuda and req in {
            "float16",
            "float32",
            "int8",
            "int8_float16",
            "int8_float32",
            "bfloat16",
        }:
            _add("cuda", req)
        if cuda:
            for ctype in ("float16", "int8_float16", "float32", "int8"):
                _add("cuda", ctype)
        large = (
            self.asr_profile in _PREMIUM_ASR_PROFILES
            or effective_model == "large-v3"
            or "/" in effective_model
        )
        if large:
            # int8_float32 — лучший баланс качества/скорости на CPU для large-v3
            for ctype in ("int8_float32", "float32", "default", "int8"):
                _add("cpu", ctype)
        else:
            for ctype in ("int8", "int8_float32", "default"):
                _add("cpu", ctype)
        return attempts

    @staticmethod
    def _resolve_model_name(model_name: str) -> str:
        normalized = model_name.strip().lower()
        if normalized in {"best-ru", "best_ru", "best"}:
            return "large-v3"
        return model_name

    def _resolve_asr_model(self) -> str:
        if self.asr_profile == "ideal_ru":
            return "large-v3"
        if self.asr_profile == "ultima_ru":
            return resolve_ultima_whisper_model()
        if self.asr_profile == "medium_ru":
            return "medium"
        return "medium"

    def _asr_decode_preset(self) -> dict[str, object]:
        if self.asr_profile == "ultima_ru":
            # RU fine-tuned: чуть глубже поиск, чем раньше — модель устойчивее к beam>9, чем «сырой» large-v3.
            return {"beam_size": 10, "best_of": 5, "patience": 1.18}
        if self.asr_profile == "ideal_ru":
            # Systran large-v3: нейтральная модель, нужен широкий поиск.
            return {"beam_size": 9, "best_of": 5, "patience": 1.12}
        # medium_ru: лёгкий пресет для скорости.
        return {"beam_size": 5, "best_of": 3, "patience": 0.9}

    def _log(self, message: str) -> None:
        print(f"[transcriber] {message}", flush=True)

    def _stage(self, name: str, message: str) -> None:
        """Помечает этап пайплайна распознавания в логах."""
        self._log(f"[этап: {name}] {message}")

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", text) if p.strip()]
        return parts if parts else ([text.strip()] if text.strip() else [])

    @staticmethod
    def _split_sentences_ru(text: str) -> list[str]:
        """Дробление на предложения для пост-редактора (в т.ч. многоточие)."""
        cleaned = re.sub(r"\s+", " ", (text or "").strip())
        if not cleaned:
            return []
        parts = re.split(r"(?<=[\.\!\?…])\s+", cleaned)
        out = [p.strip() for p in parts if p.strip()]
        return out if out else [cleaned]

    @staticmethod
    def _merge_adjacent_role_pieces(
        pieces: list[tuple[float, float, str]],
        roles: list[str],
    ) -> tuple[list[tuple[float, float, str]], list[str]]:
        """Склеивает подряд идущие сегменты одной роли — лучший контекст для пунктуации и ролей."""
        if not pieces or not roles or len(pieces) != len(roles):
            return pieces, roles
        mp: list[tuple[float, float, str]] = []
        mr: list[str] = []
        cs, ce, ct = pieces[0][0], pieces[0][1], pieces[0][2].strip()
        cr = roles[0]
        for i in range(1, len(pieces)):
            s, e, raw_t = pieces[i]
            t = raw_t.strip()
            r = roles[i]
            if r == cr:
                ct = f"{ct} {t}".strip() if ct else t
                ce = e
            else:
                mp.append((cs, ce, ct))
                mr.append(cr)
                cs, ce, ct = s, e, t
                cr = r
        mp.append((cs, ce, ct))
        mr.append(cr)
        return mp, mr

    @staticmethod
    def _normalize_sentence_caps_ru(text: str) -> str:
        """Заглавная буква в начале и после . ! ? …"""
        t = (text or "").strip()
        if not t:
            return text
        out = t[0].upper() + t[1:] if len(t) > 1 else t.upper()
        out = re.sub(
            r"(?<=[\.\!\?…])(\s+)([а-яё])",
            lambda m: m.group(1) + m.group(2).upper(),
            out,
        )
        return out

    @staticmethod
    def _role_scores(sentence: str) -> tuple[int, int]:
        normalized = sentence.lower()
        operator_patterns = (
            r"\bздравствуйте\b",
            r"\bменя\s+зовут\b",
            r"\bчем\s+я\s+могу\s+помочь\b",
            r"\bчем\s+я\s+могу\s+быть\s+вам\s+полезен\b",
            r"\bмогу\s+подсказать\b",
            r"\bвам\s+необходимо\b",
            r"\bвам\s+нужно\b",
            r"\bобратитесь\b",
            r"\bзапишите\b",
            r"\bпродиктую\b",
            r"\bномер\s+телефона\b",
            r"\bконтакт",
            r"\bсотрудники\s+работают\b",
            r"\bдепартамент\b",
            r"\bграфик\s+работы\b",
            r"\bприемная\s+комиссия\b",
            r"\bуточню\b",
            r"\bпо\s+вашему\s+вопросу\b",
            r"\bя\s+вас\s+понял\b",
            r"\bкак\s+я\s+могу\s+к\s+вам\s+обращаться\b",
            r"\bкак\s+вас\s+зовут\b",
            r"\bспасибо\s+вам\s+за\s+обращение\b",
            r"\bдо\s+свидания\b",
            r"\bхорошего\s+дня\b",
            r"\bдобрый\s+день\b",
            r"\bдоброе\s+утро\b",
            r"\bдобрый\s+вечер\b",
            r"\bслушаю\s+вас\b",
            r"\bслушаю\b",
            r"\bодну\s+секунду\b",
            r"\bсейчас\s+уточню\b",
            r"\bя\s+уточн",
            r"\bпо\s+данному\s+вопросу\b",
            r"\bвы\s+можете\s+обратиться\b",
            r"\bвам\s+следует\b",
            r"\bнеобходимо\s+предоставить\b",
            r"\bнужно\s+предоставить\b",
            r"\bпринести\s+документы\b",
            r"\bподать\s+документы\b",
            r"\bзаполнить\s+заявление\b",
            r"\bзапишите,\s*пожалуйста\b",
            r"\bпродиктуйте\b",
            r"\bваши\s+данные\b",
            r"\bваш\s+вопрос\b",
            r"\bпо\s+расписанию\b",
            r"\bв\s+рабочее\s+время\b",
            r"\bхорошего\s+вам\s+дня\b",
            r"\bвсего\s+доброго\b",
            r"\bбудьте\s+здоровы\b",
        )
        applicant_patterns = (
            r"\bкуда\s+мне\s+обратиться\b",
            r"\bу\s+меня\b",
            r"\bмне\s+нужно\b",
            r"\bя\s+хочу\b",
            r"\bможно\s+ли\b",
            r"\bкак\s+оформить\b",
            r"\bчто\s+для\s+этого\b",
            r"\bкакие\s+документ",
            r"\bспасибо\b",
            r"\bподскажите,\s*пожалуйста\b",
            r"\bу\s+меня\s+дочь\b",
            r"\bкакие\s+льготы\b",
            r"\bя\s+звоню\b",
            r"\bхочу\s+узнать\b",
            r"\bинтересует\b",
            r"\bмог\s+бы\s+я\b",
            r"\bмогу\s+ли\s+я\b",
            r"\bскажите,\s*пожалуйста\b",
            r"\bа\s+можно\b",
        )
        short_backchannels = (
            "ага",
            "угу",
            "да",
            "нет",
            "понял",
            "поняла",
            "хорошо",
            "ясно",
            "спасибо",
            "понятно",
            "ладно",
            "окей",
            "ок",
            "конечно",
            "всё",
            "всё понял",
            "всё поняла",
        )

        operator_score = sum(2 for p in operator_patterns if re.search(p, normalized))
        applicant_score = sum(2 for p in applicant_patterns if re.search(p, normalized))

        # ------------------------------------------------------------------
        # Pronoun-based scoring — very strong signal in formal Russian dialogue.
        # Operators address the applicant with formal «вы»: вам/вас/ваш/ваши.
        # Applicants talk about themselves: мне/меня/мой/моя/мои/у меня.
        # «Я» is weaker: operators also say «я уточню», «я могу подсказать».
        # ------------------------------------------------------------------
        op_pronouns = len(re.findall(r"\b(вам|вас|ваш(?:и|а|е|его|ей)?)\b", normalized))
        app_pronouns = len(re.findall(
            r"\b(мне|меня|мой|моя|мои|моего|моей)\b", normalized
        ))
        first_person_ya = len(re.findall(r"(?<![а-яё])я(?![а-яё])", normalized))

        operator_score += min(3, op_pronouns)
        applicant_score += min(3, app_pronouns)
        # «я» contributes 1 per every 2 occurrences, starting from first
        applicant_score += min(2, first_person_ya // 2 + (1 if first_person_ya >= 1 else 0))

        # ------------------------------------------------------------------
        # High-confidence operator phrases (+2 extra bonus on top of pattern score).
        # These phrases are virtually never said by the applicant.
        # ------------------------------------------------------------------
        if re.search(
            r"\bменя\s+зовут\b"
            r"|\bслушаю\s+вас\b"
            r"|\bспасибо\s+вам\s+за\s+обращение\b"
            r"|\bблагодарим\s+вас\s+за\s+обращение\b"
            r"|\bспасибо\s+за\s+(?:ваше\s+)?обращение\b"
            r"|\bчем\s+(?:я\s+могу|могу)\s+(?:вам\s+)?помочь\b"
            r"|\bвам\s+необходимо\b"
            r"|\bвы\s+можете\s+обратиться\b",
            normalized,
        ):
            operator_score += 2

        # ------------------------------------------------------------------
        # Imperative verbs directed at the applicant — always operator instructions.
        # ------------------------------------------------------------------
        if re.search(
            r"\b(позвоните|обратитесь|принесите|предоставьте|заполните"
            r"|уточните|назовите|продиктуйте|приходите|свяжитесь"
            r"|отправьте|напишите|сообщите|подтвердите)\b",
            normalized,
        ):
            operator_score += 3

        # ------------------------------------------------------------------
        # General imperative / directive bonus (existing, keep it).
        # ------------------------------------------------------------------
        if re.search(r"\b(запишите|позвоните|принесите|оформите)\b", normalized):
            operator_score += 2

        # ------------------------------------------------------------------
        # Question-mark scoring — context-dependent.
        # Operators also ask questions ("Как вас зовут?"), so "?" is not a
        # universal applicant signal. Only give full bonus when no operator
        # patterns matched, otherwise a weak +1.
        # ------------------------------------------------------------------
        if "?" in sentence:
            if operator_score == 0:
                applicant_score += 2   # strong: no operator evidence → applicant question
            else:
                applicant_score += 1   # weak: may be operator's clarifying question

        # Interrogative word at the START of a phrase → weak applicant signal.
        # Exclude operator name-questions like "Как вас зовут?".
        if re.match(r"^\s*(как|где|когда|зачем|почему|куда|откуда|сколько)\b", normalized):
            if not re.search(r"\bкак\s+(?:вас|я\s+могу)\b", normalized):
                applicant_score += 1

        if re.search(r"\b(мне|у\s+меня|подскажите\s+пожалуйста)\b", normalized):
            applicant_score += 1
        if normalized.strip() in short_backchannels:
            applicant_score += 1

        return operator_score, applicant_score

    @staticmethod
    def _is_operator_anchor(sentence: str) -> bool:
        normalized = sentence.lower()
        anchor_patterns = (
            r"\bздравствуйте\b.*\bменя\s+зовут\b",
            r"\bчем\s+я\s+могу\s+быть\s+вам\s+полезен\b",
            r"\bчем\s+я\s+могу\s+вам\s+помочь\b",
            r"\bкак\s+я\s+могу\s+к\s+вам\s+обращаться\b",
            r"\bподскажите,\s*пожалуйста,\s*как\s+вас\s+зовут\b",
            r"\bподскажите,\s*пожалуйста,\s*как\s+я\s+могу\s+к\s+вам\s+обращаться\b",
            r"\bспасибо\s+вам\s+за\s+обращение\b",
            r"\bблагодарим\s+вас\s+за\s+обращение\b",
            r"\bспасибо\s+за\s+(?:ваше\s+)?обращение\b",
            r"\bдвфу\b.*\bздравствуйте\b",
            r"\bздравствуйте\b.*\bдвфу\b",
            r"\bдобрый\s+день\b.*\bменя\s+зовут\b",
            r"\bалл[оо]?\b.*\bменя\s+зовут\b",
            r"\bменя\s+зовут\b",
            r"\bслушаю\s+вас\b",
            r"\bчем\s+могу\s+помочь\b",
        )
        return any(re.search(pattern, normalized) for pattern in anchor_patterns)

    @staticmethod
    def _is_operator_name_question(sentence: str) -> bool:
        normalized = sentence.lower()
        patterns = (
            r"\bкак\s+я\s+могу\s+к\s+вам\s+обращаться\b",
            r"\bкак\s+вас\s+зовут\b",
            r"\bподскажите,\s*пожалуйста,\s*как\s+вас\s+зовут\b",
            r"\bподскажите,\s*пожалуйста,\s*как\s+я\s+могу\s+к\s+вам\s+обращаться\b",
            r"\bкак\s+вас\s+могу\s+называть\b",
            r"\bваше\s+имя\b",
            r"\bпредставьтесь,?\s*пожалуйста\b",
        )
        return any(re.search(pattern, normalized) for pattern in patterns)

    @staticmethod
    def _is_operator_opening(sentence: str) -> bool:
        normalized = sentence.lower()
        patterns = (
            r"\bздравствуйте\b.*\bменя\s+зовут\b",
            r"\bчем\s+я\s+могу\s+быть\s+вам\s+полезен\b",
            r"\bчем\s+я\s+могу\s+вам\s+помочь\b",
            r"\bдвфу\b.*\bздравствуйте\b",
            r"\bздравствуйте\b.*\bдвфу\b",
            r"\bдобрый\s+день\b.*\bменя\s+зовут\b",
            r"\bалл[оо]?\b.*\bменя\s+зовут\b",
            r"\bменя\s+зовут\b",
            r"\bслушаю\s+вас\b",
            r"\bчем\s+могу\s+помочь\b",
        )
        return any(re.search(pattern, normalized) for pattern in patterns)

    @staticmethod
    def _detect_role(
        sentence: str,
        previous_role: str | None,
        next_sentence: str | None = None,
    ) -> str:
        operator_score, applicant_score = Transcriber._role_scores(sentence)

        if operator_score > applicant_score:
            return "Оператор"
        if applicant_score > operator_score:
            return "Заявитель"

        # Tie-breakers.
        normalized = sentence.lower()
        if "?" in sentence:
            return "Заявитель"
        if previous_role is None:
            return "Оператор"

        # Prefer natural alternation for short ambiguous replies.
        if len(normalized.split()) <= 3:
            return "Заявитель" if previous_role == "Оператор" else "Оператор"

        # If next phrase looks like a question, current often belongs to operator.
        if next_sentence and "?" in next_sentence:
            return "Оператор"
        return previous_role

    @staticmethod
    def _smooth_roles(chunks: list[str], roles: list[str]) -> list[str]:
        if len(roles) < 3:
            return roles
        result = roles[:]
        # Remove isolated role flips: A B A -> A A A
        for i in range(1, len(result) - 1):
            if result[i - 1] == result[i + 1] and result[i] != result[i - 1]:
                if len(chunks[i].split()) <= 8:
                    result[i] = result[i - 1]

        # Question-answer pairing: after a question, a short neutral answer likely other speaker.
        for i in range(0, len(result) - 1):
            current = chunks[i]
            nxt = chunks[i + 1]
            if "?" in current and len(nxt.split()) <= 8 and "?" not in nxt:
                if result[i] == result[i + 1]:
                    result[i + 1] = "Заявитель" if result[i] == "Оператор" else "Оператор"
        return result

    @staticmethod
    def _apply_operator_anchors(chunks: list[str], roles: list[str]) -> list[str]:
        """Force operator role for known call-center script anchors.

        After a confirmed anchor, the operator typically continues for 1-2 more
        short segments (finishing the greeting/closing script) before handing
        over to the applicant. Propagate forward to 2 segments instead of 1.
        """
        result = roles[:]
        for i, chunk in enumerate(chunks):
            if not Transcriber._is_operator_anchor(chunk):
                continue
            result[i] = "Оператор"
            # Short preceding fragment usually belongs to the same operator turn.
            if i - 1 >= 0 and len(chunks[i - 1].split()) <= 6:
                result[i - 1] = "Оператор"
            # First following segment: propagate unconditionally if short.
            if i + 1 < len(chunks) and len(chunks[i + 1].split()) <= 12:
                result[i + 1] = "Оператор"
            # Second following segment: propagate only when text scoring agrees.
            if i + 2 < len(chunks) and len(chunks[i + 2].split()) <= 10:
                op_s, app_s = Transcriber._role_scores(chunks[i + 2])
                if op_s >= app_s:
                    result[i + 2] = "Оператор"
        return result

    @staticmethod
    def _apply_dialog_flow_rules(chunks: list[str], roles: list[str]) -> list[str]:
        """Apply strict conversation-flow rules tailored for call-center dialogues."""
        if not roles:
            return roles
        result = roles[:]
        n = len(chunks)

        # First strong opening should belong to operator.
        for i in range(min(4, n)):
            if Transcriber._is_operator_opening(chunks[i]):
                result[i] = "Оператор"
                if i + 1 < n and not Transcriber._is_operator_anchor(chunks[i + 1]):
                    result[i + 1] = "Заявитель"
                break

        for i in range(n):
            current = chunks[i]
            # Name question from operator -> next meaningful response should be applicant.
            if Transcriber._is_operator_name_question(current):
                result[i] = "Оператор"
                if i + 1 < n and not Transcriber._is_operator_anchor(chunks[i + 1]):
                    result[i + 1] = "Заявитель"

            # Applicant question should generally be followed by operator answer.
            if result[i] == "Заявитель" and "?" in current:
                if i + 1 < n and not Transcriber._is_operator_anchor(chunks[i + 1]):
                    result[i + 1] = "Оператор"

            # Operator clarifying question usually expects applicant short response.
            if result[i] == "Оператор" and "?" in current:
                if i + 1 < n and len(chunks[i + 1].split()) <= 12:
                    if not Transcriber._is_operator_anchor(chunks[i + 1]):
                        result[i + 1] = "Заявитель"

            # Closing phrase belongs to operator; prior short confirmation likely applicant.
            # См. docs/CALL_CENTER_ROLE_ANCHORS.md — «за обращение» в официальной благодарности только у линии.
            if re.search(
                r"\bспасибо\s+вам\s+за\s+обращение\b"
                r"|\bблагодарим\s+вас\s+за\s+обращение\b"
                r"|\bспасибо\s+за\s+(?:ваше\s+)?обращение\b",
                current.lower(),
            ):
                result[i] = "Оператор"
                if i - 1 >= 0 and len(chunks[i - 1].split()) <= 8:
                    result[i - 1] = "Заявитель"

        return result

    def _operator_speaker_likelihood_score(self, texts: list[str]) -> float:
        joined = " ".join(texts).strip()
        if not joined:
            return -100.0
        bonus = 0.0
        if (
            self._is_operator_opening(joined)
            or self._is_operator_anchor(joined)
            or self._is_operator_name_question(joined)
        ):
            bonus = 12.0
        operator_part, applicant_part = self._role_scores(joined)
        return float(operator_part - applicant_part + bonus)

    def _build_role_transcript_from_voice_clusters(
        self,
        pieces: list[tuple[float, float, str, str]],
        role_map: dict[str, str],
    ) -> str:
        lines: list[str] = []
        current_role: str | None = None
        current_parts: list[str] = []
        prev_heuristic: str | None = None
        for _abs_s, _abs_e, text, cluster_id in pieces:
            role = role_map.get(cluster_id)
            if role is None:
                role = self._detect_role(text, prev_heuristic, next_sentence=None)
            prev_heuristic = role
            if current_role is None:
                current_role = role
                current_parts = [text]
            elif role == current_role:
                current_parts.append(text)
            else:
                lines.append(f"{current_role}: {' '.join(current_parts)}")
                current_role = role
                current_parts = [text]
        if current_parts and current_role:
            lines.append(f"{current_role}: {' '.join(current_parts)}")
        return "\n".join(lines)

    def _refine_voice_role_sequence(
        self, pieces: list[tuple[float, float, str, str]], role_map: dict[str, str]
    ) -> list[str]:
        text_chunks = [p[2] for p in pieces]
        raw_roles = [role_map.get(p[3], "Заявитель") for p in pieces]
        refined = self._smooth_roles(text_chunks, raw_roles)
        refined = self._apply_dialog_flow_rules(text_chunks, refined)
        refined = self._apply_operator_anchors(text_chunks, refined)
        refined = self._smooth_roles(text_chunks, refined)
        return refined

    def _apply_operator_prologue_prior(
        self, pieces: list[tuple[float, float, str, str]], roles: list[str], first_n: int = 12
    ) -> list[str]:
        result = roles[:]
        n = min(first_n, len(result))
        t0 = float(pieces[0][0]) if pieces else 0.0
        for i in range(n):
            s, _e, txt, _cid = pieces[i]
            # Первые ~55 с именно диалога (после clip по ожиданию из имени файла).
            if (s - t0) <= 55.0 and (
                self._is_operator_opening(txt)
                or self._is_operator_anchor(txt)
                or self._is_operator_name_question(txt)
            ):
                result[i] = "Оператор"
                if i + 1 < len(result) and len(pieces[i + 1][2].split()) <= 12:
                    result[i + 1] = "Заявитель"
        return result

    @staticmethod
    def _roles_to_transcript(chunks: list[str], roles: list[str]) -> str:
        if not chunks or not roles or len(chunks) != len(roles):
            return ""
        lines: list[str] = []
        cur_role = roles[0]
        cur_parts = [chunks[0]]
        for i in range(1, len(chunks)):
            if roles[i] == cur_role:
                cur_parts.append(chunks[i])
            else:
                lines.append(f"{cur_role}: {' '.join(cur_parts)}")
                cur_role = roles[i]
                cur_parts = [chunks[i]]
        lines.append(f"{cur_role}: {' '.join(cur_parts)}")
        return "\n".join(lines)

    @staticmethod
    def _overlap_seconds(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
        return max(0.0, min(a_end, b_end) - max(a_start, b_start))

    def _assign_cluster_by_overlap(
        self,
        segment_start: float,
        segment_end: float,
        diarization_turns: list[tuple[float, float, str]],
    ) -> str:
        if not diarization_turns:
            return "SPK_0"
        best_overlap = 0.0
        best_label = diarization_turns[0][2]
        center = 0.5 * (segment_start + segment_end)
        best_distance = float("inf")
        for s, e, label in diarization_turns:
            ov = self._overlap_seconds(segment_start, segment_end, s, e)
            if ov > best_overlap:
                best_overlap = ov
                best_label = label
            distance = abs(center - (0.5 * (s + e)))
            if distance < best_distance:
                best_distance = distance
                nearest_label = label
        if best_overlap <= 1e-6:
            return nearest_label
        return best_label

    @staticmethod
    def _postprocess_ru_text(text: str) -> str:
        cleaned = re.sub(r"\s+", " ", text).strip()
        cleaned = re.sub(r"\s+([,.;!?])", r"\1", cleaned)
        # Нормализация частых доменных написаний.
        replacements = {
            "двф у": "ДВФУ",
            "д в ф у": "ДВФУ",
            "дальневосточного федерального университета": "Дальневосточного федерального университета",
            "кол центр": "колл-центр",
            "колл центр": "колл-центр",
        }
        lowered = cleaned.lower()
        for src, dst in replacements.items():
            lowered = lowered.replace(src, dst.lower())
        # Восстанавливаем базовый регистр для начала предложений.
        lowered = lowered[:1].upper() + lowered[1:] if lowered else lowered
        lowered = lowered.replace("двфу", "ДВФУ")
        return lowered

    @classmethod
    def _get_local_punctuation_model(cls):
        if cls._punct_model is not None:
            return cls._punct_model
        _apply_deepmultilingual_transformers_compat()
        from deepmultilingualpunctuation import PunctuationModel

        cls._punct_model = PunctuationModel(model="kredor/punctuate-all")
        return cls._punct_model

    def _resolve_llm_post_edit_config(self):
        """OpenAI-compatible API (Ollama ``/v1``, LM Studio, …). Только для ``llm_backend=remote``."""
        if not self.enable_llm_post_edit or self.llm_backend != "remote":
            return None
        from llm_post_edit import LLMPostEditConfig

        base = (
            (self.llm_post_edit_base_url or "").strip()
            or (os.environ.get("LLM_POST_EDIT_BASE_URL") or "").strip()
        ).rstrip("/")
        if not base:
            return None
        model = (self.llm_post_edit_model or "").strip() or (
            os.environ.get("LLM_POST_EDIT_MODEL") or ""
        ).strip()
        if not model:
            return None
        api_key = self.llm_post_edit_api_key
        if api_key is None:
            raw = (os.environ.get("LLM_POST_EDIT_API_KEY") or "").strip()
            api_key = raw or None
        to = max(15.0, min(float(self.llm_post_edit_timeout_seconds), 600.0))
        return LLMPostEditConfig(base_url=base, model=model, api_key=api_key, timeout_seconds=to)

    @staticmethod
    def _token_set(text: str) -> set[str]:
        return set(re.findall(r"[а-яёa-z0-9]+", (text or "").lower()))

    def _punctuate_long_text(self, model: object, chunk: str, max_batch_chars: int = 520) -> str:
        """
        Пунктуация с батчингом по предложениям: длинные сегменты не ломаем одним вызовом модели.
        """
        chunk_pp = self._postprocess_ru_text(chunk)
        if not chunk_pp:
            return chunk_pp
        if len(chunk_pp) <= max_batch_chars:
            out = getattr(model, "restore_punctuation")(chunk_pp)
            return str(out).strip()
        sents = self._split_sentences_ru(chunk_pp)
        if len(sents) <= 1:
            out = getattr(model, "restore_punctuation")(chunk_pp)
            return str(out).strip()
        batches: list[str] = []
        buf: list[str] = []
        nchars = 0
        rp = getattr(model, "restore_punctuation")
        for s in sents:
            add = len(s) + (1 if buf else 0)
            if buf and nchars + add > max_batch_chars:
                batches.append(rp(" ".join(buf)).strip())
                buf = [s]
                nchars = len(s)
            else:
                buf.append(s)
                nchars += add
        if buf:
            batches.append(rp(" ".join(buf)).strip())
        return self._postprocess_ru_text(" ".join(batches))

    def _role_logic_refiner(
        self,
        pieces: list[tuple[float, float, str]],
        roles: list[str],
        confidence: str,
    ) -> tuple[list[str], bool]:
        if not pieces or not roles or len(pieces) != len(roles):
            return roles, False
        # При высокой уверенности диаризации голос уже хорош — эвристики могут только портить.
        if confidence not in {"low", "medium"}:
            return roles, False
        text_chunks = [p[2] for p in pieces]
        refined = self._apply_dialog_flow_rules(text_chunks, roles[:])
        refined = self._apply_operator_anchors(text_chunks, refined)
        refined = self._smooth_roles(text_chunks, refined)
        piece4 = [(s, e, txt, "SPK_0") for s, e, txt in pieces]
        refined = self._apply_operator_prologue_prior(piece4, refined)
        changed = refined != roles
        return refined, changed

    def _post_edit_ru_dialogue(
        self,
        raw_text: str,
        pieces: list[tuple[float, float, str]],
        roles: list[str],
        role_confidence: str,
        elapsed_before_post_edit: float,
    ) -> tuple[str, str, list[str], str, list[tuple[float, float, str]] | None]:
        if not self.enable_post_edit:
            return raw_text, "off", roles, "", None
        if self.asr_profile not in {"ideal_ru", "ultima_ru", "medium_ru"}:
            return raw_text, "off_profile", roles, "", None
        if not pieces:
            return raw_text, "off_no_pieces", roles, "", None

        # degrade depth near time budget
        budget_left = max(0.0, self.total_time_budget_seconds - elapsed_before_post_edit)
        timeout = int(max(10, min(self.post_edit_timeout_seconds, budget_left)))
        if timeout <= 10:
            return raw_text, "fallback_budget", roles, "timeout_budget", None

        box: dict[str, object] = {}
        err: dict[str, Exception] = {}

        def worker() -> None:
            try:
                model = self._get_local_punctuation_model()
                local_chunks = [p[2] for p in pieces]
                punctuated: list[str] = []
                punct_limit = len(local_chunks)
                if budget_left < 32:
                    punct_limit = max(6, len(local_chunks) // 2)
                rp = getattr(model, "restore_punctuation")
                for i, ch in enumerate(local_chunks):
                    ch = (ch or "").strip()
                    if not ch:
                        punctuated.append(ch)
                        continue
                    if i >= punct_limit:
                        punctuated.append(ch)
                        continue
                    if len(ch) > 800:
                        punctuated.append(self._punctuate_long_text(model, ch, max_batch_chars=500))
                    else:
                        punctuated.append(str(rp(ch)).strip())
                merged = self._postprocess_ru_text(" ".join(punctuated))
                base = self._normalize_sentence_caps_ru(self._postprocess_ru_text(raw_text))
                edited = self._normalize_sentence_caps_ru(merged)
                tok_src = self._token_set(base)
                tok_new = self._token_set(edited)
                overlap = (len(tok_src & tok_new) / max(1, len(tok_src))) if tok_src else 1.0
                ratio = (len(edited.split()) / max(1, len(base.split()))) if base else 1.0
                if overlap < 0.52 or ratio < 0.72 or ratio > 1.38:
                    box["text"] = base
                    box["reason"] = "anti_drift_revert"
                    texts_out = list(local_chunks)
                else:
                    box["text"] = edited
                    box["reason"] = "ok"
                    texts_out = punctuated
                work_pieces = [(pieces[i][0], pieces[i][1], texts_out[i]) for i in range(len(pieces))]
                new_roles, changed = self._role_logic_refiner(
                    pieces=work_pieces,
                    roles=roles,
                    confidence=role_confidence,
                )
                box["roles"] = new_roles
                box["roles_changed"] = changed
                box["out_pieces"] = [(s, e, t) for s, e, t in work_pieces]
            except Exception as exc:
                err["error"] = exc

        fut = _POST_EDIT_EXECUTOR.submit(worker)
        try:
            fut.result(timeout=float(timeout))
        except FuturesTimeout:
            return raw_text, "fallback_timeout", roles, "post_edit_timeout", None
        if "error" in err:
            return raw_text, "fallback_error", roles, str(err["error"]), None

        final_text = str(box.get("text", raw_text))
        final_roles = list(box.get("roles", roles))
        out_list = box.get("out_pieces")
        updated: list[tuple[float, float, str]] | None
        if isinstance(out_list, list):
            updated = out_list
        else:
            updated = None
        status = "on"
        if box.get("reason") == "anti_drift_revert":
            status = "fallback_anti_drift"
        return final_text, status, final_roles, str(box.get("reason", "")), updated

    def _pick_operator_cluster(self, clustered: list[tuple[float, float, str, str]]) -> str | None:
        by_cluster_text: dict[str, list[str]] = {}
        by_cluster_seconds: dict[str, float] = {}
        for s, e, txt, cid in clustered:
            by_cluster_text.setdefault(cid, []).append(txt)
            by_cluster_seconds[cid] = by_cluster_seconds.get(cid, 0.0) + max(0.0, e - s)
        if len(by_cluster_text) < 2:
            return None

        # Правило «первый говорящий = оператор»:
        # В колл-центре после clip_timestamps первым всегда говорит оператор.
        # Проверяем, что текст первых сегментов этого кластера не противоречит роли оператора
        # (т.е. оператор-score >= заявитель-score), и если так — возвращаем сразу.
        first_cid = clustered[0][3]
        first_cluster_texts = by_cluster_text.get(first_cid, [])
        first_op_s, first_app_s = 0, 0
        for t in first_cluster_texts[:4]:
            o, a = self._role_scores(t)
            first_op_s += o
            first_app_s += a
        if first_op_s >= first_app_s:
            return first_cid

        # Приоритет операторского приветствия в начале звонка.
        # Use relative time from the first segment so the bonus works even when
        # clip_timestamps shifts all absolute timestamps forward (e.g. to 72s+).
        _seg_t0 = clustered[0][0] if clustered else 0.0
        early = [x for x in clustered if x[0] <= _seg_t0 + 35.0]
        early_bonus: dict[str, float] = {}
        for _s, _e, txt, cid in early:
            if self._is_operator_anchor(txt) or self._is_operator_opening(txt):
                early_bonus[cid] = early_bonus.get(cid, 0.0) + 8.0
            if self._is_operator_name_question(txt):
                early_bonus[cid] = early_bonus.get(cid, 0.0) + 4.0

        scored: list[tuple[float, str]] = []
        for cid, texts in by_cluster_text.items():
            score = self._operator_speaker_likelihood_score(texts)
            score += min(4.0, by_cluster_seconds.get(cid, 0.0) / 60.0)  # tiny duration prior
            score += early_bonus.get(cid, 0.0)
            scored.append((score, cid))
        scored.sort(reverse=True)
        return scored[0][1] if scored else None

    def _role_confidence_label(
        self,
        cluster_scores: dict[str, float],
        diagnostics: dict[str, float | int | str],
    ) -> str:
        vals = sorted(cluster_scores.values(), reverse=True)
        margin = (vals[0] - vals[1]) if len(vals) >= 2 else 0.0
        quality = float(diagnostics.get("cluster_quality", 0.0) or 0.0)
        if margin >= 4.0 and quality >= 2.0:
            return "high"
        if margin >= 1.5 and quality >= 1.2:
            return "medium"
        return "low"

    def _apply_semantic_role_refine(
        self,
        chunks: list[str],
        roles: list[str],
        voice_confidence: str = "low",
    ) -> list[str]:
        """Локальная RuBERT-Tiny2: уточнение ролей по смыслу (бесплатно, без облака)."""
        if not _semantic_refine_enabled() or not chunks or not roles:
            return roles
        self._stage(
            "роли/семантика",
            "RuBERT-Tiny2: уточнение Оператор/Заявитель по смыслу реплик (локально)",
        )
        return try_semantic_role_refine(
            chunks,
            roles,
            protect_fn=lambda t: (
                self._is_operator_anchor(t)
                or self._is_operator_opening(t)
                or self._is_operator_name_question(t)
            ),
            log=self._log,
            voice_confidence=voice_confidence,
        )

    @staticmethod
    def _decode_roles_globally(chunks: list[str]) -> list[str]:
        """Global 2-state Viterbi decoding with linguistically-grounded transitions."""
        if not chunks:
            return []
        states = ("Оператор", "Заявитель")
        score_by_chunk = [Transcriber._role_scores(chunk) for chunk in chunks]

        # Precompute anchor flags so we can use them in transition logic.
        _anchor_prev = [Transcriber._is_operator_anchor(c) for c in chunks]

        dp: list[dict[str, float]] = []
        prev_state: list[dict[str, str | None]] = []

        # Seed: in a call centre the first speaker is always the operator.
        _seed_op_bias = 1.0
        first_op, first_app = score_by_chunk[0]
        dp.append({
            "Оператор": float(first_op) + _seed_op_bias,
            "Заявитель": float(first_app),
        })
        prev_state.append({"Оператор": None, "Заявитель": None})

        for i in range(1, len(chunks)):
            op_score, app_score = score_by_chunk[i]
            cur_scores = {"Оператор": float(op_score), "Заявитель": float(app_score)}
            chunk = chunks[i]
            chunk_prev = chunks[i - 1]
            chunk_words = len(chunk.split())
            cur_prev: dict[str, str | None] = {}
            cur_dp: dict[str, float] = {}
            for state in states:
                best_val = -10**9
                best_prev: str | None = None
                for ps in states:
                    transition = 0.0

                    # Base turn-taking preference (dialogue alternates).
                    if ps != state:
                        transition += 0.5  # was 0.4

                    # Long explanatory chunk → same speaker likely continues.
                    # Reduced threshold from >20 to >15 words, bonus from 0.2 to 0.4.
                    if ps == state and chunk_words > 15:
                        transition += 0.4

                    # After a question in previous chunk → very strong switch pressure.
                    if "?" in chunk_prev and ps != state:
                        transition += 1.0  # was 0.6

                    # Very short chunk (1-3 words) = backchannel («да», «угу», «понятно»).
                    # Backchannels do NOT represent a speaker turn change — reward STAYING.
                    # Old code had +0.3 for SWITCHING on short chunks — that was wrong.
                    if chunk_words <= 3:
                        if ps == state:
                            transition += 0.4  # backchannel stays with same speaker
                        # (no switching bonus for short chunks — removed)

                    # After a confirmed operator anchor the applicant is expected next.
                    # This prevents runs of operator-labeled segments after e.g. "меня зовут".
                    if _anchor_prev[i - 1] and state == "Заявитель" and ps == "Оператор":
                        transition += 0.8  # push toward applicant after anchor

                    candidate = dp[i - 1][ps] + transition + cur_scores[state]
                    if candidate > best_val:
                        best_val = candidate
                        best_prev = ps
                cur_dp[state] = best_val
                cur_prev[state] = best_prev
            dp.append(cur_dp)
            prev_state.append(cur_prev)

        last_state = max(states, key=lambda s: dp[-1][s])
        decoded = [last_state]
        for i in range(len(chunks) - 1, 0, -1):
            last_state = prev_state[i][last_state] or "Оператор"
            decoded.append(last_state)
        decoded.reverse()
        return decoded

    @staticmethod
    def _split_segment_at_silences(
        segment: object,
        min_silence_gap: float = 0.35,
    ) -> list[tuple[float, float, str]]:
        """Split a faster-whisper segment at internal silence gaps using word timestamps.

        Returns list of (start, end, text) sub-segments. Falls back to the
        original single segment when word timestamps are unavailable or empty.
        A gap of ≥ min_silence_gap seconds between consecutive words almost always
        marks a speaker turn boundary in call-center dialogue.
        """
        words = getattr(segment, "words", None)
        seg_text = (getattr(segment, "text", "") or "").strip()
        seg_start = float(getattr(segment, "start", 0.0))
        seg_end = float(getattr(segment, "end", 0.0))

        if not words:
            return [(seg_start, seg_end, seg_text)] if seg_text else []

        # Group consecutive words; start a new group when silence >= min_silence_gap.
        groups: list[list] = [[words[0]]]
        for word in words[1:]:
            gap = float(getattr(word, "start", 0.0)) - float(getattr(groups[-1][-1], "end", 0.0))
            if gap >= min_silence_gap:
                groups.append([word])
            else:
                groups[-1].append(word)

        result: list[tuple[float, float, str]] = []
        for group in groups:
            text = " ".join((getattr(w, "word", "") or "").strip() for w in group).strip()
            if text:
                result.append((
                    float(getattr(group[0], "start", seg_start)),
                    float(getattr(group[-1], "end", seg_end)),
                    text,
                ))
        return result if result else [(seg_start, seg_end, seg_text)]

    def _build_role_transcript(self, text: str, chunks: list[str] | None = None) -> str:
        source_chunks = chunks if chunks is not None else self._split_sentences(text)
        prepared = [chunk.strip() for chunk in source_chunks if chunk.strip()]
        if not prepared:
            return ""

        roles = self._decode_roles_globally(prepared)
        roles = self._apply_operator_anchors(prepared, roles)
        roles = self._apply_dialog_flow_rules(prepared, roles)
        roles = self._smooth_roles(prepared, roles)
        roles = self._apply_dialog_flow_rules(prepared, roles)
        roles = self._apply_operator_anchors(prepared, roles)
        roles = self._smooth_roles(prepared, roles)
        roles = self._apply_semantic_role_refine(prepared, roles, voice_confidence="low")

        # Merge neighboring chunks with same role for cleaner dialogue.
        merged_lines: list[str] = []
        current_role = roles[0]
        current_parts = [prepared[0]]
        for i in range(1, len(prepared)):
            if roles[i] == current_role:
                current_parts.append(prepared[i])
            else:
                merged_lines.append(f"{current_role}: {' '.join(current_parts)}")
                current_role = roles[i]
                current_parts = [prepared[i]]
        merged_lines.append(f"{current_role}: {' '.join(current_parts)}")
        return "\n".join(merged_lines)

    def _ensure_model(self) -> None:
        if self._model is not None:
            self._stage(
                "ASR-модель",
                f"уже в памяти ({self._resolve_asr_model()}, {self._asr_runtime_device}/{self._asr_runtime_compute})",
            )
            return
        effective_model = self._resolve_asr_model()
        self._stage(
            "ASR-модель",
            f"загрузка «{effective_model}» (profile={self.asr_profile}, compute_type={self.compute_type})…",
        )
        self._log(
            "На первом запуске крупная ASR-модель (large-v3 или репозиторий с Hugging Face) "
            "может скачиваться 10–30+ минут. Если сеть медленная, подождите."
        )
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError(
                "Не найден faster-whisper. Установите зависимости: pip install -r requirements.txt"
            ) from exc

        attempts = self._iter_whisper_load_attempts(effective_model)
        self._log(
            "Порядок загрузки CTranslate2: "
            + ", ".join(f"{d}/{c}" for d, c in attempts[:8])
            + ("…" if len(attempts) > 8 else "")
        )

        last_error: Exception | None = None
        for device, ctype in attempts:
            model_box: dict[str, object] = {}
            error_box: dict[str, Exception] = {}

            def _load_model(dev: str = device, ct: str = ctype) -> None:
                # huggingface_hub: HF_HUB_OFFLINE в constants задаётся при импорте и не следует за env.
                # Прогрев ECAPA мог оставить constants.HF_HUB_OFFLINE=True навсегда — без патча сеть «мёртвая».
                try:
                    with HF_HUB_DOWNLOAD_LOCK:
                        with force_huggingface_hub_online():
                            model_box["model"] = WhisperModel(
                                effective_model,
                                device=dev,
                                compute_type=ct,
                            )
                except Exception as exc:  # pragma: no cover - safety net
                    error_box["error"] = exc

            thread = threading.Thread(target=_load_model, daemon=True)
            thread.start()
            thread.join(timeout=self.model_load_timeout_seconds)
            if thread.is_alive():
                self._log(
                    "[asr] Достигнут лимит ожидания загрузки; жду завершения потока "
                    "без запуска второй параллельной загрузки…"
                )
                thread.join()
            if "model" in model_box:
                self._model = model_box["model"]
                self._stage("ASR-модель", f"готова: device={device}, compute_type={ctype}")
                self._asr_runtime_device = device
                self._asr_runtime_compute = ctype
                break
            if "error" in error_box:
                err = error_box["error"]
                last_error = err
                self._log(f"[asr] Не удалось {device}/{ctype}: {err}")
                continue
            last_error = RuntimeError(
                f"Загрузка {device}/{ctype} завершилась без модели и без ошибки в логе."
            )
            self._log(f"[asr] {last_error}")
            continue
        else:
            hint = (
                "Превышен таймаут или все варианты compute_type отвергнуты. "
                "Проверьте интернет (скачивание модели с Hugging Face), место на диске и версию ctranslate2. "
                "Временно выберите «Стандарт» (medium / int8)."
            )
            if last_error is not None:
                err_txt = str(last_error).lower()
                if "hf_hub_offline" in err_txt or "outgoing traffic" in err_txt:
                    hint += (
                        " Если в сообщении про HF_HUB_OFFLINE: уберите из .env строку HF_HUB_OFFLINE=1 "
                        "или задайте HF_HUB_OFFLINE=0 на время первой загрузки модели с Hugging Face."
                    )
                raise RuntimeError(f"{hint} Последняя ошибка: {last_error}") from last_error
            raise RuntimeError(hint)

    def _joint_voice_role_decode(
        self,
        audio_path: str,
        voice_items: list[tuple[float, float, str]],
        asr_elapsed: float,
    ) -> tuple[str, str, str, str, list[tuple[float, float, str]], list[str]]:
        self._stage("роли", "черновик: спикеры по тексту (до уточнения голосом)")
        role_text = self._build_role_transcript(
            " ".join(x[2] for x in voice_items),
            chunks=[x[2] for x in voice_items],
        )
        base_pieces = [(s, e, t) for s, e, t in voice_items]
        base_roles = self._decode_roles_globally([x[2] for x in base_pieces]) if base_pieces else []
        role_attribution = "Текстовый fallback"
        role_confidence = "low"
        role_diagnostics = f"role_mode=fallback; asr_seconds={asr_elapsed:.2f}"
        diarization_started = time.perf_counter()
        used_heavy = False
        try:
            from speaker_voice_roles import (
                _remap_segments_by_speaker_spans,
                cluster_voice_segments,
            )

            clustered: list[tuple[float, float, str, str]] = []
            cluster_reason = "fallback text"
            cluster_diag: dict[str, object] = {}
            if self.heavy_diarization:
                self._stage("роли/heavy", "тяжёлая диаризация по голосу…")
                self._log("Joint mode: heavy diarization по голосу...")
                try:
                    from speaker_diarization_heavy import run_heavy_diarization

                    turns, heavy_reason, heavy_diag = run_heavy_diarization(
                        audio_path=audio_path,
                        target_speakers=2,
                        max_seconds=self.heavy_diarization_timeout_seconds,
                    )
                    if turns:
                        clustered = _remap_segments_by_speaker_spans(
                            voice_items,
                            turns,
                            min_split_sec=0.85,
                            min_split_share=0.22,
                            min_words_to_split=6,
                        )
                        if not clustered:
                            clustered = []
                            for s, e, txt in voice_items:
                                cid = self._assign_cluster_by_overlap(s, e, turns)
                                clustered.append((s, e, txt, cid))
                        cluster_reason = heavy_reason
                        cluster_diag = heavy_diag
                        used_heavy = True
                except Exception as exc:
                    self._log(f"Heavy diarization ошибка ({exc}), перехожу на light.")
            if not clustered:
                self._stage(
                    "роли/ECAPA",
                    f"кластеризация голоса (лёгкий режим), фрагментов от ASR={len(voice_items)}",
                )
                clustered, cluster_reason, cluster_diag = cluster_voice_segments(
                    audio_path=audio_path,
                    segments=voice_items,
                    target_speakers=2,
                )
            if not clustered:
                self._stage("роли/ECAPA", "кластеры не получены — остаётся текстовая модель спикеров")
            if clustered:
                by_cluster: dict[str, list[str]] = {}
                for _s, _e, txt, cid in clustered:
                    by_cluster.setdefault(cid, []).append(txt)
                operator_cluster = self._pick_operator_cluster(clustered) or ""
                role_map = {
                    cid: ("Оператор" if cid == operator_cluster else "Заявитель")
                    for cid in by_cluster.keys()
                }
                refined_roles = self._refine_voice_role_sequence(clustered, role_map)
                refined_roles = self._apply_operator_prologue_prior(clustered, refined_roles)
                cluster_scores = {
                    cid: self._operator_speaker_likelihood_score(texts)
                    for cid, texts in by_cluster.items()
                }
                role_confidence = self._role_confidence_label(cluster_scores, cluster_diag)
                # Cluster consistency: when voice quality is medium or high, enforce
                # that each acoustic cluster carries one dominant role (majority vote).
                # This prevents text heuristics from "breaking up" a correct cluster by
                # flipping a few individual segment labels against the cluster's evidence.
                if role_confidence in {"high", "medium"} and clustered:
                    _cluster_votes: dict[str, dict[str, int]] = {}
                    for (_s, _e, _txt, _cid), _role in zip(clustered, refined_roles):
                        _cluster_votes.setdefault(_cid, {"Оператор": 0, "Заявитель": 0})
                        _cluster_votes[_cid][_role] += 1
                    _cluster_dominant = {
                        _cid: max(_votes, key=lambda k: _votes[k])
                        for _cid, _votes in _cluster_votes.items()
                    }
                    refined_roles = [
                        "Оператор"  # explicit operator anchor always wins
                        if self._is_operator_anchor(_txt) or self._is_operator_opening(_txt)
                        else _cluster_dominant.get(_cid, _role)
                        for (_s, _e, _txt, _cid), _role in zip(clustered, refined_roles)
                    ]
                if role_confidence == "low":
                    # Joint fallback: text-подсказка при низкой уверенности голоса.
                    # Text wins only when it is DECISIVE (|op_score - app_score| >= 2).
                    # When both scores are 0-0 or tied, voice prediction is kept to
                    # avoid replacing a correct voice label with an arbitrary text guess.
                    text_chunks = [x[2] for x in clustered]
                    text_roles = self._decode_roles_globally(text_chunks)
                    _text_scores = [self._role_scores(c) for c in text_chunks]
                    refined_roles = [
                        "Оператор"
                        if self._is_operator_anchor(ch) or self._is_operator_opening(ch)
                        else (
                            tr                           # text wins: it is decisive
                            if abs(ts[0] - ts[1]) >= 2 and tr != vr
                            else vr                      # voice wins: text is ambiguous
                        )
                        for vr, tr, ch, ts in zip(
                            refined_roles, text_roles, text_chunks, _text_scores
                        )
                    ]
                    refined_roles = self._smooth_roles(text_chunks, refined_roles)
                refined_roles = self._apply_semantic_role_refine(
                    [x[2] for x in clustered],
                    refined_roles,
                    voice_confidence=role_confidence,
                )
                refined_roles = self._smooth_roles([x[2] for x in clustered], refined_roles)
                role_text = self._roles_to_transcript([x[2] for x in clustered], refined_roles)
                base_pieces = [(x[0], x[1], x[2]) for x in clustered]
                base_roles = refined_roles[:]
                role_attribution = f"Joint voice-role decode ({'heavy' if used_heavy else 'light'})"
                diarization_elapsed = time.perf_counter() - diarization_started
                _sem_note = (
                    "; semantic_role_refine=rubert_tiny2"
                    if _semantic_refine_enabled()
                    else ""
                )
                _dv = cluster_diag.get("dual_voice_encoder")
                _dual_note = f"; dual_voice_encoder={_dv}" if _dv is not None else ""
                role_diagnostics = (
                    f"role_mode=joint; diarization_path={'heavy_diarization' if used_heavy else 'fallback_light'}; "
                    f"split_mode={cluster_diag.get('split_mode', 'native')}; reason={cluster_reason}; "
                    f"quality={cluster_diag.get('cluster_quality', 0.0):.3f}; "
                    f"asr_seconds={asr_elapsed:.2f}; diarization_seconds={diarization_elapsed:.2f}"
                    f"{_sem_note}{_dual_note}"
                )
                self._stage(
                    "роли",
                    f"голос+текст за {diarization_elapsed:.2f} с: "
                    f"{'heavy' if used_heavy else 'light'}, уверенность={role_confidence}",
                )
        except Exception as exc:
            self._stage("роли/ошибка", f"{exc!s} — текстовый fallback")
            role_diagnostics = f"role_mode=fallback; exception={exc}; asr_seconds={asr_elapsed:.2f}"
        return role_text, role_attribution, role_confidence, role_diagnostics, base_pieces, base_roles

    def transcribe(
        self,
        audio_path: str | Path,
        *,
        original_basename: str | None = None,
    ) -> TranscriptionResult:
        self._ensure_model()
        path = str(audio_path)
        self._stage("начало", f"файл «{Path(path).name}»")
        self._log(f"Начинаю обработку файла: {path}")
        info_language: str | None = None
        # VAD: чувствительнее к тихому началу звонка; большой speech_pad — не отрезать первые слова до порога Silero.
        _vad_base = {
            "threshold": 0.26,
            "neg_threshold": 0.11,
            "min_speech_duration_ms": 120,
            "min_silence_duration_ms": 2000,
            "speech_pad_ms": 600,
        }
        # ideal_ru: чуть чувствительнее base (тихое «алло»), но без экстремальных порогов —
        # иначе в поток попадает шум/музыка ожидания → large-v3 даёт больше мусора, чем medium.
        _vad_ideal = {
            "threshold": 0.17,
            "neg_threshold": 0.085,
            "min_speech_duration_ms": 100,
            "min_silence_duration_ms": 2000,
            "speech_pad_ms": 900,
        }
        # ultima_ru: максимальная чувствительность — ловим тихое «алло»/«здравствуйте» в самом начале.
        # Очень низкий порог + большой pad гарантируют, что начало диалога не срезается Silero.
        # speech_pad_ms=2200: чуть шире, чем 2000 — слоги у границ VAD реже попадают в обрезку.
        _vad_ultima = {
            "threshold": 0.12,
            "neg_threshold": 0.06,
            "min_speech_duration_ms": 80,
            "min_silence_duration_ms": 2000,
            "speech_pad_ms": 2200,
        }
        if self.asr_profile == "ultima_ru":
            _vad_for_profile = _vad_ultima
        elif self.asr_profile == "ideal_ru":
            _vad_for_profile = _vad_ideal
        else:
            _vad_for_profile = _vad_base
        _initial_asr_prompt = (
            "Разговор на русском языке. ДВФУ, единый контактный центр Владивостока. "
            "Приёмная комиссия, общежитие, зачётная книжка, расписание, справка, документы, студент, "
            "стипендия, оплата, зачисление, перевод, куратор, военная кафедра. "
            "Здравствуйте, добрый день, добрый вечер, алло, слушаю вас, слышу вас, чем могу помочь, "
            "подскажите пожалуйста, уточню информацию, перезвоню, спасибо за обращение."
        )
        if self.asr_profile == "ultima_ru":
            _initial_asr_prompt += (
                " Телефонная линия, шум и компрессия; речь может быть тихой или с обрывами слогов."
            )
        transcribe_kwargs = {
            "vad_filter": True,
            "vad_parameters": _vad_for_profile,
            "beam_size": 7,
            "best_of": 5,
            # medium_ru: одна температура (быстрее). premium-профили ниже — цепочка fallback.
            "temperature": 0.0,
            "language": "ru",
            "task": "transcribe",
            "condition_on_previous_text": False,
            # Русский контекст ЕКЦ + разговорные клише линии (якорь для large-v3).
            "initial_prompt": _initial_asr_prompt,
        }
        transcribe_kwargs.update(self._asr_decode_preset())
        if os.environ.get("CALLQA_ASR_NO_VAD", "").strip().lower() in ("1", "true", "yes"):
            transcribe_kwargs["vad_filter"] = False
            self._stage(
                "ASR",
                "VAD Silero отключён (CALLQA_ASR_NO_VAD=1) — медленнее, но начало разговора не режется VAD",
            )
        if self.asr_profile == "ideal_ru":
            # 3-шаговая цепочка: Systran large-v3 без RU fine-tune — нужны резервные проходы, но не все 6.
            transcribe_kwargs.update(
                {
                    "temperature": (0.0, 0.4, 0.8),
                    "compression_ratio_threshold": 2.4,
                    "log_prob_threshold": -1.0,
                    "no_speech_threshold": 0.62,
                }
            )
        elif self.asr_profile == "ultima_ru":
            # 4-шаговая цепочка: fine-tuned RU переживает лишний проход лучше, чем «сырой» large-v3.
            # Чуть мягче отсев по log_prob/compression — меньше пустых дыр на тихой речи и шумной линии.
            transcribe_kwargs.update(
                {
                    "temperature": (0.0, 0.25, 0.5, 0.72),
                    "compression_ratio_threshold": 2.62,
                    "log_prob_threshold": -1.32,
                    "no_speech_threshold": 0.78,
                }
            )
        else:
            # medium_ru: первое окно Whisper (до 30 с) при дефолтном no_speech может «прыгнуть» вперёд.
            transcribe_kwargs.update(
                {
                    "no_speech_threshold": 0.56,
                    "log_prob_threshold": -1.42,
                }
            )
        # Enable word-level timestamps for sub-segment splitting at silence gaps.
        # Checked at runtime so the code works with older faster-whisper versions too.
        _wt_sig = inspect.signature(self._model.transcribe)
        _word_ts_enabled = "word_timestamps" in _wt_sig.parameters
        if _word_ts_enabled:
            # ideal_ru и ultima_ru: вкл. по умолчанию — сегменты режутся по паузам → точнее дiaризация.
            # medium_ru: выкл. (модель слабее, экономия важнее). Переопределение: CALLQA_ASR_WORD_TIMESTAMPS=0|1
            _wts_default = "0" if self.asr_profile == "medium_ru" else "1"
            _wts_env = os.environ.get("CALLQA_ASR_WORD_TIMESTAMPS", _wts_default).strip().lower()
            if _wts_env in ("0", "false", "no", "off"):
                transcribe_kwargs["word_timestamps"] = False
            else:
                transcribe_kwargs["word_timestamps"] = True

        if self.asr_profile in _PREMIUM_ASR_PROFILES:
            _tw_sig = inspect.signature(self._model.transcribe)
            _twp = _tw_sig.parameters
            if "hotwords" in _twp:
                transcribe_kwargs["hotwords"] = _IDEAL_RU_RU_HOTWORDS
                if self.asr_profile == "ultima_ru":
                    transcribe_kwargs["hotwords"] = (
                        f"{transcribe_kwargs['hotwords']} {_ULTIMA_RU_HOTWORDS_EXTRA}"
                    )
            # Кавычки «» для слияния с токенами при word_timestamps.
            if "prepend_punctuations" in _twp:
                transcribe_kwargs["prepend_punctuations"] = "\"'\u201c\u00bf([{-\u00ab"
            if "append_punctuations" in _twp:
                transcribe_kwargs["append_punctuations"] = "\"'.。,，!！?？:：\u201d)]}、\u00bb"

        _wts_log = (
            "off (API не поддерживает)"
            if not _word_ts_enabled
            else (
                "on"
                if transcribe_kwargs.get("word_timestamps")
                else "off (ускорение, CALLQA_ASR_WORD_TIMESTAMPS=0)"
            )
        )
        self._stage(
            "ASR-параметры",
            f"profile={self.asr_profile}, beam={transcribe_kwargs['beam_size']}, "
            f"best_of={transcribe_kwargs['best_of']}, patience={transcribe_kwargs.get('patience', 'n/a')}, "
            f"word_timestamps={_wts_log}"
            + (", ru_hotwords=on" if transcribe_kwargs.get("hotwords") else ""),
        )
        clip_applied = False
        skip_sec = 0.0
        file_skip_extra = ""
        if self.skip_first_seconds is not None:
            skip_sec = max(0.0, float(self.skip_first_seconds))
            self._log(f"Ручной пропуск начала (--skip-first-seconds): {skip_sec:.1f} с")
            file_skip_extra = f"file_skip_source=manual; file_skip_seconds={skip_sec:.2f}"
        else:
            # Streamlit сохраняет загрузку во временный файл (tmpXXX.mp3) — имя для парсинга передавайте отдельно.
            name_for_skip = (original_basename or Path(path).name).strip()
            name_for_skip = Path(name_for_skip).name
            parsed = parse_call_filename_wait_skip(name_for_skip)
            if parsed is not None:
                tot_wait, wait_token = parsed
                disable_fn = os.environ.get("CALLQA_DISABLE_FILENAME_SKIP", "").strip().lower() in (
                    "1",
                    "true",
                    "yes",
                )
                if disable_fn:
                    self._log(
                        "[asr] Пропуск начала по хвосту имени отключён (CALLQA_DISABLE_FILENAME_SKIP)."
                    )
                    file_skip_extra = "file_skip_source=filename_disabled_by_env"
                else:
                    skip_sec = tot_wait
                    mm = int(skip_sec // 60)
                    ss = int(skip_sec % 60)
                    self._log(
                        f"Пропуск ожидания по имени файла: «{wait_token}» → {skip_sec:.0f} с "
                        f"({mm:02d}:{ss:02d} от начала записи)"
                    )
                    file_skip_extra = (
                        f"file_skip_source=filename; file_wait_token={wait_token}; "
                        f"file_skip_seconds={skip_sec:.2f}"
                    )
            else:
                file_skip_extra = "file_skip_source=none"

        if skip_sec > 0:
            self._stage("пропуск ожидания", f"{skip_sec:.1f} с от начала будут отрезаны до распознавания")
        else:
            self._stage("пропуск ожидания", "не задан (имя файла без _MM-SS и нет --skip-first-seconds)")

        self._stage(
            "аудио",
            "подготовка временного WAV: обрезка начала (если есть) + нормализация громкости, 16 kHz mono",
        )
        # Пропуск ожидания: по возможности физическая обрезка во временном WAV (см. _prepare_asr_normalized_wav),
        # а не clip_timestamps. clip — только запасной вариант, если WAV не создать.
        # Ultima: агрессивнее нормализуем + спектральный денойз (параметры под телефон/ЕКЦ).
        _norm_target = 0.135 if self.asr_profile == "ultima_ru" else 0.11
        _do_denoise = self.asr_profile == "ultima_ru"
        _ultima_denoise_kw: dict[str, object] | None = (
            {"noise_percentile": 10.0, "over_subtraction": 2.28, "spectral_floor": 0.038}
            if _do_denoise
            else None
        )
        asr_path, time_offset, transcribe_kwargs, _asr_tmp = _prepare_asr_normalized_wav(
            path, skip_sec, transcribe_kwargs, self._log,
            target_active_rms=_norm_target,
            denoise=_do_denoise,
            denoise_kwargs=_ultima_denoise_kw,
        )
        if skip_sec > 0 and _asr_tmp is None and asr_path == path:
            signature = inspect.signature(self._model.transcribe)
            if "clip_timestamps" in signature.parameters:
                transcribe_kwargs["clip_timestamps"] = f"{skip_sec}"
                clip_applied = True
                time_offset = 0.0
                self._log(
                    f"Запасной режим: начало {skip_sec:.1f} с только через clip_timestamps "
                    f"(временный WAV с обрезкой не создан)."
                )
            else:
                self._log(
                    "Внимание: пропуск начала недоступен — нет clip_timestamps и не удалось собрать WAV."
                )
        asr_started = time.perf_counter()
        self._stage(
            "Whisper",
            "запуск модели: чтение подготовленного WAV, VAD (Silero), признаки — на длинных файлах несколько минут",
        )
        try:
            segments, info = self._model.transcribe(asr_path, **transcribe_kwargs)
        finally:
            if _asr_tmp:
                try:
                    os.unlink(_asr_tmp)
                except OSError:
                    pass
        prep_elapsed = time.perf_counter() - asr_started
        _dur_full = float(getattr(info, "duration", 0.0) or 0.0)
        _dur_vad = float(getattr(info, "duration_after_vad", _dur_full) or _dur_full)
        self._stage(
            "Whisper",
            f"признаки за {prep_elapsed:.1f} с (аудио {_dur_full:.1f} с, после VAD ~{_dur_vad:.1f} с речи); "
            f"дальше в этом же этапе — потоковое декодирование и сбор текста (логи «ASR прогресс»); "
            f"отсев IVR и галлюцинаций промпта — по мере поступления сегментов",
        )
        decode_started = time.perf_counter()
        _last_progress_log = decode_started
        # Hallucination filter: discard segments whose words overlap >50% with initial_prompt.
        # Whisper sometimes repeats the prompt verbatim when audio is silent or clipped.
        _raw_prompt = str(transcribe_kwargs.get("initial_prompt") or "")
        # Слова из initial_prompt, которые совпадают с нормальной речью оператора — не считаем для анти-галлюцинации
        _stop_words = {
            "и", "в", "по", "на", "с", "к", "у", "из", "за", "не", "это", "что", "как", "языке",
            "двфу", "единый", "контактный", "центр", "владивостока", "здравствуйте", "добрый", "день",
            "вечер", "алло", "слушаю", "вас", "слышу", "чем", "могу", "вам", "помочь", "подскажите",
            "пожалуйста", "уточню", "информацию", "перезвоню", "спасибо", "обращение", "разговор",
            "русском", "приёмная", "комиссия", "общежитие", "зачётная", "книжка", "расписание",
            "справка", "документы", "студент", "стипендия", "оплата", "зачисление", "перевод",
            "куратор", "военная", "кафедра", "университет", "деканат", "институт", "факультет",
            # Частые слова из initial_prompt, иначе приветствие оператора ошибочно режется как «копия промпта».
            "ваш", "ваше", "вашего", "оператор", "специалист", "линия", "звонок", "телефон",
            "абитуриент", "заявитель",
        }
        _prompt_words = set(re.sub(r"[^\w\s]", "", _raw_prompt.lower()).split()) - _stop_words
        # Первые секунды после обрезки ожидания не фильтруем по пересечению с промптом (там реальное приветствие).
        _hallucination_overlap_ignore_sec = 32.0

        # Phrases spoken by PBX auto-informer at the very end of a call.
        # These are never part of the human conversation and must be removed.
        # NB: regex detects IVR; trimming logic below keeps real speech before it.
        _IVR_TAIL_RE = re.compile(
            r"(?:пожалуйста\s*,?\s*)?оцените\s+работу"
            r"|по\s+шкале\s+от\s+1\s+до\s+5"
            r"|после\s+(?:звукового\s+)?сигнала"
            r"|приняли\s+участие\s+в\s+(?:нашем\s+)?опросе",
            re.IGNORECASE,
        )

        _min_gap = 0.25 if self.asr_profile == "ultima_ru" else 0.35

        def _collect_asr_pieces(
            segments_iter,
            *,
            progress_prefix: str,
            emit_progress: bool,
        ) -> tuple[list[str], list[str], list[tuple[float, float, str]], int, int]:
            local_parts: list[str] = []
            local_chunks: list[str] = []
            local_pieces: list[tuple[float, float, str]] = []
            local_segment_count = 0
            local_hallucination_count = 0
            local_last_log = decode_started

            for segment in segments_iter:
                local_segment_count += 1
                _now = time.perf_counter()
                if emit_progress and (
                    local_segment_count == 1 or (_now - local_last_log) >= 6.0
                ):
                    self._log(
                        f"{progress_prefix}: сегмент #{local_segment_count}, "
                        f"распознано примерно до {segment.end:.1f} с записи, "
                        f"декодирование идёт {(_now - decode_started):.1f} с"
                    )
                    local_last_log = _now

                sub_segs = self._split_segment_at_silences(segment, min_silence_gap=_min_gap)
                for seg_start, seg_end, piece in sub_segs:
                    if not piece:
                        continue
                    _ivr_m = _IVR_TAIL_RE.search(piece)
                    if _ivr_m:
                        _before_ivr = piece[:_ivr_m.start()].rstrip()
                        _before_ivr = re.sub(
                            r"\s*пожалуйста\s*,?\s*$", "", _before_ivr, flags=re.IGNORECASE
                        ).rstrip()
                        if _before_ivr and len(_before_ivr.split()) >= 2:
                            piece = _before_ivr
                            self._log(
                                f"[asr] Обрезан хвост автоинформатора, сохранена речь: «{piece[:80]}»"
                            )
                        else:
                            self._log(
                                f"[asr] Отброшен сегмент автоинформатора: «{piece[:80]}»"
                            )
                            continue
                    if (
                        len(_prompt_words) >= 3
                        and float(seg_start) >= _hallucination_overlap_ignore_sec
                    ):
                        seg_words = set(re.sub(r"[^\w\s]", "", piece.lower()).split())
                        overlap = len(seg_words & _prompt_words) / len(seg_words) if seg_words else 0.0
                        if overlap > 0.85 and len(seg_words) >= 5:
                            local_hallucination_count += 1
                            self._log(
                                f"[asr] Отброшен сегмент-галлюцинация ({overlap:.0%}): «{piece[:80]}»"
                            )
                            continue
                    local_parts.append(piece)
                    local_chunks.extend(self._split_sentences(piece))
                    local_pieces.append((seg_start, seg_end, piece))

            return (
                local_parts,
                local_chunks,
                local_pieces,
                local_segment_count,
                local_hallucination_count,
            )

        (
            parts,
            chunks,
            whisper_pieces_rel,
            segment_count,
            hallucination_count,
        ) = _collect_asr_pieces(
            segments,
            progress_prefix="ASR прогресс",
            emit_progress=True,
        )

        intro_recovery_added = 0
        intro_recovery_hallucinations = 0
        try:
            _transcribe_sig = inspect.signature(self._model.transcribe)
            _clip_supported = "clip_timestamps" in _transcribe_sig.parameters
        except Exception:
            _clip_supported = False

        if _clip_supported and should_attempt_intro_recovery(
            whisper_pieces_rel,
            audio_duration=_dur_full,
            duration_after_vad=_dur_vad,
        ):
            intro_limit_sec = 32.0 if self.asr_profile in _PREMIUM_ASR_PROFILES else 28.0
            rescue_kwargs = dict(transcribe_kwargs)
            rescue_kwargs["clip_timestamps"] = f"0,{intro_limit_sec:.1f}"
            rescue_kwargs["vad_filter"] = False
            rescue_kwargs["condition_on_previous_text"] = False
            rescue_kwargs["temperature"] = 0.0
            rescue_kwargs["no_speech_threshold"] = max(
                float(rescue_kwargs.get("no_speech_threshold", 0.0) or 0.0),
                0.86 if self.asr_profile == "ultima_ru" else 0.78,
            )
            rescue_kwargs["log_prob_threshold"] = min(
                float(rescue_kwargs.get("log_prob_threshold", -1.0) or -1.0),
                -1.8,
            )
            rescue_kwargs["initial_prompt"] = (
                f"{_initial_asr_prompt} "
                "Начало звонка: приветствие оператора, ДВФУ, меня зовут, слушаю вас, "
                "чем могу помочь, как к вам обращаться."
            )
            self._log(
                f"[asr] Основной проход выглядит обрезанным в начале; запускаю intro rescue "
                f"до {intro_limit_sec:.0f} с без VAD."
            )
            try:
                rescue_segments, _ = self._model.transcribe(asr_path, **rescue_kwargs)
                (
                    _rescue_parts,
                    _rescue_chunks,
                    rescue_pieces_rel,
                    _rescue_segment_count,
                    intro_recovery_hallucinations,
                ) = _collect_asr_pieces(
                    rescue_segments,
                    progress_prefix="ASR intro",
                    emit_progress=False,
                )
                del _rescue_parts, _rescue_chunks, _rescue_segment_count

                first_main_start = whisper_pieces_rel[0][0] if whisper_pieces_rel else intro_limit_sec
                main_head_texts = [piece_text for _s, _e, piece_text in whisper_pieces_rel[:3]]
                rescue_keep: list[tuple[float, float, str]] = []
                for rescue_start, rescue_end, rescue_text in rescue_pieces_rel:
                    if whisper_pieces_rel and rescue_end > (first_main_start + 0.35):
                        continue
                    rescue_tokens = self._token_set(rescue_text)
                    if not rescue_tokens:
                        continue
                    duplicate = False
                    for main_text in main_head_texts:
                        main_tokens = self._token_set(main_text)
                        overlap = len(rescue_tokens & main_tokens) / max(
                            1,
                            min(len(rescue_tokens), len(main_tokens)),
                        )
                        if overlap >= 0.82:
                            duplicate = True
                            break
                    if not duplicate:
                        rescue_keep.append((rescue_start, rescue_end, rescue_text))

                if rescue_keep:
                    whisper_pieces_rel = rescue_keep + whisper_pieces_rel
                    parts = [piece_text for _s, _e, piece_text in whisper_pieces_rel]
                    chunks = []
                    for _s, _e, piece_text in whisper_pieces_rel:
                        chunks.extend(self._split_sentences(piece_text))
                    intro_recovery_added = len(rescue_keep)
                    self._log(
                        f"[asr] Intro rescue добавил {intro_recovery_added} ранних фрагм. "
                        f"до первого основного сегмента."
                    )
                else:
                    self._log("[asr] Intro rescue не дал новых уникальных ранних фрагментов.")
            except Exception as exc:
                self._log(f"[asr] Intro rescue пропущен ({type(exc).__name__}: {exc})")

        decode_elapsed = time.perf_counter() - decode_started
        asr_elapsed = time.perf_counter() - asr_started
        text = " ".join(parts).strip()
        total_hallucinations = hallucination_count + intro_recovery_hallucinations
        self._stage(
            "текст",
            f"распознавание завершено: сегментов Whisper={segment_count}, символов ≈{len(text)}, "
            f"фрагментов с таймкодами={len(whisper_pieces_rel)}; "
            f"декод {decode_elapsed:.1f} с, всего ASR {asr_elapsed:.1f} с"
            + (f"; intro_rescue=+{intro_recovery_added}" if intro_recovery_added else "")
            + (
                f"; галлюцинаций отброшено: {total_hallucinations}"
                if total_hallucinations
                else ""
            ),
        )
        info_language = getattr(info, "language", None)
        whisper_pieces_rel = [(s + time_offset, e + time_offset, t) for s, e, t in whisper_pieces_rel]

        text = self._postprocess_ru_text(text)
        self._stage("текст", "постобработка русского (нормализация текста) завершена")
        total_started = time.perf_counter()
        self._stage(
            "роли+голос",
            f"совмещение ECAPA/кластеров с {len(whisper_pieces_rel)} фрагментами ASR (исходное аудио для голоса)",
        )
        (
            role_text,
            role_attribution,
            role_confidence,
            role_diagnostics,
            role_pieces,
            role_labels,
        ) = self._joint_voice_role_decode(
            audio_path=path,
            voice_items=whisper_pieces_rel,
            asr_elapsed=asr_elapsed,
        )
        joint_elapsed = time.perf_counter() - total_started
        self._stage(
            "роли+голос",
            f"блок завершён за {joint_elapsed:.2f} с — {role_attribution}; уверенность={role_confidence}",
        )
        text_work = text
        role_text_work = role_text
        llm_elapsed = 0.0
        llm_diag = "disabled"
        nf: str | None = None
        nr: str | None = None
        llm_note = ""
        llm_cfg = self._resolve_llm_post_edit_config()
        if not self.enable_llm_post_edit:
            self._stage("LLM-постредактор", "выключен в настройках")
        if self.enable_llm_post_edit and self.llm_backend == "yandex" and self.llm_yandex_api_key and self.llm_yandex_folder_id:
            from llm_cloud_eval import YandexCloudConfig, run_yandex_post_edit_threaded

            self._stage(
                "LLM",
                f"Yandex AI Studio, модель gpt://{self.llm_yandex_folder_id}/{self.llm_yandex_model}",
            )
            ycfg = YandexCloudConfig(
                api_key=self.llm_yandex_api_key,
                folder_id=self.llm_yandex_folder_id,
                model=self.llm_yandex_model,
                timeout_seconds=self.llm_post_edit_timeout_seconds,
            )
            self._log(
                f"LLM пост-редактор (Yandex AI Studio, gpt://{self.llm_yandex_folder_id}/{self.llm_yandex_model}), "
                f"таймаут≤{self.llm_post_edit_timeout_seconds:.0f}s..."
            )
            t_llm = time.perf_counter()
            nf, nr, llm_note = run_yandex_post_edit_threaded(
                text_work,
                role_text_work,
                cfg=ycfg,
                wall_timeout=float(self.llm_post_edit_timeout_seconds),
            )
            llm_elapsed = time.perf_counter() - t_llm
            if nf and nr:
                text_work = self._postprocess_ru_text(nf)
                role_text_work = nr.strip()
                llm_diag = f"ok:{llm_note}"
                self._log(f"LLM пост-редактор готов за {llm_elapsed:.2f}s ({llm_diag})")
            else:
                llm_diag = f"fallback:{llm_note}"
                self._log(f"LLM пост-редактор не применён ({llm_diag}), {llm_elapsed:.2f}s")
        elif self.enable_llm_post_edit and self.llm_backend == "embedded":
            from llm_embedded_deepseek import run_embedded_llm_post_edit_threaded

            self._stage("LLM", "встроенная DeepSeek / Hugging Face")
            self._log(
                f"LLM пост-редактор (встроенная DeepSeek / Hugging Face), "
                f"таймаут≤{self.llm_post_edit_timeout_seconds:.0f}s..."
            )
            t_llm = time.perf_counter()
            nf, nr, llm_note = run_embedded_llm_post_edit_threaded(
                text_work,
                role_text_work,
                model_id=(self.llm_embedded_model_id or "").strip() or None,
                wall_timeout=float(self.llm_post_edit_timeout_seconds),
                log=None,
            )
            llm_elapsed = time.perf_counter() - t_llm
            if nf and nr:
                text_work = self._postprocess_ru_text(nf)
                role_text_work = nr.strip()
                llm_diag = f"ok:{llm_note}"
                self._log(f"LLM пост-редактор готов за {llm_elapsed:.2f}s ({llm_diag})")
            else:
                llm_diag = f"fallback:{llm_note}"
                self._log(f"LLM пост-редактор не применён ({llm_diag}), {llm_elapsed:.2f}s")
        elif llm_cfg is not None:
            from llm_post_edit import run_llm_post_edit_threaded

            self._stage("LLM", f"HTTP API {llm_cfg.base_url}, модель={llm_cfg.model}")
            self._log(
                f"LLM пост-редактор (HTTP): {llm_cfg.base_url}, модель={llm_cfg.model}, "
                f"таймаут≤{self.llm_post_edit_timeout_seconds:.0f}s..."
            )
            t_llm = time.perf_counter()
            nf, nr, llm_note = run_llm_post_edit_threaded(
                text_work,
                role_text_work,
                llm_cfg,
                wall_timeout=float(self.llm_post_edit_timeout_seconds),
            )
            llm_elapsed = time.perf_counter() - t_llm
            if nf and nr:
                text_work = self._postprocess_ru_text(nf)
                role_text_work = nr.strip()
                llm_diag = f"ok:{llm_note}"
                self._log(f"LLM пост-редактор готов за {llm_elapsed:.2f}s ({llm_diag})")
            else:
                llm_diag = f"fallback:{llm_note}"
                self._log(f"LLM пост-редактор не применён ({llm_diag}), {llm_elapsed:.2f}s")
        elif self.enable_llm_post_edit and self.llm_backend == "remote":
            llm_diag = "remote_missing_url_or_model"
            self._stage("LLM", "remote: нет URL/модели — задайте LLM_POST_EDIT_* в .env")
            self._log(
                "LLM remote: задайте LLM_POST_EDIT_BASE_URL и LLM_POST_EDIT_MODEL в .env "
                "или передайте URL/модель в Transcriber."
            )
        elif self.enable_llm_post_edit:
            self._stage(
                "LLM-постредактор",
                f"включён, но сценарий не сработал (backend={self.llm_backend!r}) — см. ключи Yandex/URL",
            )

        skip_local_post = nf is not None and nr is not None

        post_started = time.perf_counter()
        post_elapsed = 0.0
        post_status = "off"
        post_reason = ""
        role_refine_applied = "no"
        refined_roles = role_labels
        post_pieces: list[tuple[float, float, str]] | None = None

        if skip_local_post:
            post_status = "skipped_after_llm"
            post_reason = llm_diag
            self._stage("локальный пост", "пропуск — текст и роли уже обработаны LLM")
        elif self.enable_post_edit:
            self._stage("локальный пост", "пунктуация и уточнение ролей (локальная модель)")
            self._log("Запускаю локальный пост-редактор текста/ролей...")
            post_text, post_status, refined_roles, post_reason, post_pieces = self._post_edit_ru_dialogue(
                raw_text=text_work,
                pieces=role_pieces,
                roles=role_labels,
                role_confidence=role_confidence,
                elapsed_before_post_edit=(asr_elapsed + (time.perf_counter() - total_started)),
            )
            text_work = post_text
            post_elapsed = time.perf_counter() - post_started
            self._log(
                f"Локальный пост-редактор: mode={post_status}, reason={post_reason or 'ok'}, "
                f"time={post_elapsed:.2f}s"
            )
        else:
            post_reason = "disabled"
            self._stage("локальный пост", "выключен (enable_post_edit=false)")

        self._stage("сборка отчёта", "финальная склейка расшифровки по ролям после пост-обработки")
        pieces_for_roles = post_pieces if post_pieces is not None else role_pieces
        if not skip_local_post and refined_roles and pieces_for_roles and len(refined_roles) == len(pieces_for_roles):
            new_role_text = self._roles_to_transcript([p[2] for p in pieces_for_roles], refined_roles)
            if new_role_text:
                role_refine_applied = "yes" if new_role_text != role_text_work else "no"
                role_text_work = new_role_text

        role_text_work, _thanks_role_fix = fix_operator_thanks_mislabeled_as_applicant(role_text_work)
        if _thanks_role_fix:
            self._log(
                "[роли] Формула «спасибо/благодарность за обращение» была под «Заявитель» — "
                "перенесена на «Оператор» (ошибка распределения ролей)."
            )

        text = text_work
        role_text = role_text_work

        role_diagnostics = (
            f"{role_diagnostics}; llm_post_edit={llm_diag}; llm_seconds={llm_elapsed:.2f}; "
            f"post_edit={post_status}; role_refine_applied={role_refine_applied}; "
            f"fallback_reason={post_reason or 'n/a'}; post_edit_seconds={post_elapsed:.2f}"
        )
        if file_skip_extra:
            role_diagnostics = f"{role_diagnostics}; {file_skip_extra}"

        self._log("Текст собран.")
        self._log("Сформирована расшифровка по ролям.")
        self._stage("тональность", "анализ тона по исходному аудио (librosa)")
        tone_summary = analyze_tone(audio_path)
        self._stage("тональность", f"результат: {tone_summary}")
        self._log(f"Оценка тона: {tone_summary}")
        self._stage(
            "готово",
            f"итог: язык={info_language or '—'}, символов текста={len(text)}, ASR={self._resolve_asr_model()}",
        )
        return TranscriptionResult(
            text=text,
            language=info_language,
            role_text=role_text,
            tone_summary=tone_summary,
            role_attribution=role_attribution,
            role_confidence=role_confidence,
            role_diagnostics=role_diagnostics,
            asr_profile=self.asr_profile,
            asr_model=(
                f"{self._resolve_asr_model()}"
                f" ({self._asr_runtime_device}/{self._asr_runtime_compute})"
                if self._asr_runtime_device
                else self._resolve_asr_model()
            ),
        )


class CallQualityEvaluator:
    """Экспертная эвристическая оценка качества консультации."""

    _OPERATOR_ALIASES = OPERATOR_ALIASES

    _GREETING_PATTERNS = (r"\bздравств", r"\bдобрый\s+(день|вечер|утро)")
    _INTRO_PATTERNS = (
        r"\bменя\s+зовут\b",
        r"\bоператор\b",
        r"\bспециалист\b",
        r"\bколл[- ]?центр\b",
        r"\bдвфу\b",
    )
    _ASK_APPLICANT_NAME_PATTERNS = (
        r"\bкак\s+вас\s+зовут\b",
        r"\bкак\s+к\s+вам\s+обращаться\b",
        r"\bпредставьтесь\b",
        r"\bподскажите\s+ваше\s+имя\b",
    )
    _GOODBYE_PATTERNS = (
        r"\bдо\s+свидания\b",
        r"\bхорошего\s+дня\b",
        r"\bвсего\s+доброго\b",
        r"\bобращайтесь\b",
        r"\bрад.?\s+был.?\s+помочь\b",
        r"\bбыл.?\s+рад.?\s+помочь\b",
        r"\bдо\s+встречи\b",
        r"\bхорошего\s+вечера\b",
        r"\bприятного\s+дня\b",
    )
    _OFFER_HELP_PATTERNS = (
        r"\bчем\s+ещё\s+могу\s+помочь\b",
        r"\bчем\s+еще\s+могу\s+помочь\b",
        r"\bостались\s+ли\s+(?:у\s+вас\s+)?вопросы\b",
        r"\bесть\s+ли\s+(?:ещё|еще)\s+вопросы\b",
        r"\bмогу\s+(?:ли\s+)?(?:ещё|еще|чем-то)\s+быть\s+полезн",
        r"\bещё\s+(?:чем-то\s+)?могу\s+помочь\b",
        r"\bеще\s+(?:чем-то\s+)?могу\s+помочь\b",
        r"\bмогу\s+помочь\s+чем.нибудь",
    )
    _THANK_PATTERNS = (
        r"\bспасибо\s+за\s+(?:ваш\s+)?обращение\b",
        r"\bблагодар\w+\s+за\s+(?:ваш\s+)?(?:обращение|звонок)\b",
        r"\bблагодар\w+\s+(?:вас\s+)?за\s+звонок\b",
        r"\bспасибо\s+за\s+(?:ваш\s+)?звонок\b",
        r"\bрады\s+вашему\s+обращению\b",
    )
    _FILLER_PATTERNS = (
        r"\bээ+\b",
        r"\bэм+\b",
        r"\bну\b",
        r"\bкак\s+бы\b",
        r"\bтипа\b",
        r"\bв\s+общем\b",
        r"\bкороче\b",
        r"\bтак\s+сказать\b",
        r"\bгрубо\s+говоря\b",
        r"\bна\s+самом\s+деле\b",
        r"\bэто\s+самое\b",
        r"\bпоходу\b",
        r"\bмаленько\b",
    )
    _DIMINUTIVE_PATTERNS = (
        r"\bминуточку\b",
        r"\bтрубочку\b",
        r"\bзвоночек\b",
        r"\bдоговорчик\b",
        r"\bзаявочка\b",
        r"\bдокументик\b",
        r"\bсправочка\b",
        r"\bденежки\b",
    )
    _NEGATIVE_PHRASES = (
        r"\bк\s+сожалению\b",
        r"\bне\s+знаю\b",
        r"\bне\s+могу\s+подсказать\b",
        r"\bэто\s+не\s+ко\s+мне\b",
        r"\bэто\s+не\s+моя\s+зона\b",
        r"(?:^|\.\s+|\n)нет\b",
        r"\bвам\s+нужно\b",
        r"\bвы\s+должны\b",
        r"\bвы\s+не\s+поняли\b",
        r"\bвы\s+неправильно\s+поняли\b",
        r"\bваша\s+проблема\b",
        r"\bвы\s+не\s+правы\b",
        r"\bвас\s+беспокоит\b",
    )
    _CONSULTATION_PATTERNS = (
        r"\bнеобходимо\b",
        r"\bнужно\b",
        r"\bпорядок\b",
        r"\bдокумент",
        r"\bсрок",
        r"\bшаг\b",
        r"\bподать\b",
        r"\bзаявлен",
        r"\bличный\s+кабинет\b",
    )
    _RESOLUTION_PATTERNS = (
        r"\bрешили\b",
        r"\bвопрос\s+решен\b",
        r"\bпонятно\b",
        r"\bспасибо\b",
        r"\bблагодар",
    )
    _ENGAGEMENT_PATTERNS = (
        r"\bуточн",
        r"\bправильно\s+ли\s+я\s+понял\b",
        r"\bдавайте\s+разбер",
        r"\bподскажите,\s*пожалуйста\b",
        r"\bостались\s+ли\s+вопросы\b",
        r"\bчем\s+еще\s+могу\s+помочь\b",
    )
    _COMMON_NAMES = (
        "алексей",
        "александр",
        "андрей",
        "анна",
        "артем",
        "артём",
        "валерия",
        "виктория",
        "виктор",
        "владимир",
        "дарья",
        "денис",
        "дмитрий",
        "евгений",
        "елена",
        "екатерина",
        "иван",
        "игорь",
        "ирина",
        "кирилл",
        "константин",
        "ксения",
        "мария",
        "максим",
        "михаил",
        "надежда",
        "наталья",
        "никита",
        "оксана",
        "ольга",
        "павел",
        "полина",
        "роман",
        "светлана",
        "сергей",
        "софья",
        "татьяна",
        "юлия",
        "яна",
    )

    # Слова, которые не считаем именем в коротком ответе заявителя.
    _APPLICANT_NAME_STOPWORDS = frozenset(
        {
            "да",
            "нет",
            "ну",
            "вот",
            "это",
            "вам",
            "вас",
            "меня",
            "мне",
            "нас",
            "здесь",
            "там",
            "ага",
            "угу",
            "спасибо",
            "пожалуйста",
            "здравствуйте",
            "добрый",
            "день",
            "алло",
            "слушаю",
            "хорошо",
            "ладно",
            "понятно",
            "извините",
            "простите",
            "конечно",
            "я",
        }
    )

    def _count_pattern_hits(self, text: str, patterns: tuple[str, ...]) -> int:
        return sum(1 for pattern in patterns if re.search(pattern, text))

    @staticmethod
    def _clean_name_token(raw: str) -> str:
        s = raw.strip()
        s = re.sub(r"^[\s\.…,:;!?«»\"'(\[\{]+|[\s\.…,:;!?»\"')\]\}]+$", "", s)
        return s.strip()

    @staticmethod
    def _title_cyrillic_name(token: str) -> str:
        t = CallQualityEvaluator._clean_name_token(token)
        if not t:
            return t
        if "-" in t:
            return "-".join(
                (p[0].upper() + p[1:].lower()) if len(p) > 1 else p.upper()
                for p in t.split("-")
                if p
            )
        return t[0].upper() + t[1:].lower() if len(t) > 1 else t.upper()

    def _normalize_operator_name(self, name: str) -> str | None:
        name = self._clean_name_token(name)
        if not name:
            return None
        key = name.lower()
        if key in self._OPERATOR_ALIASES:
            return self._OPERATOR_ALIASES[key]
        for part in re.split(r"[\s\-.]+", key):
            if part in self._OPERATOR_ALIASES:
                return self._OPERATOR_ALIASES[part]
        return None

    def _find_operator_name(self, transcript: str, forced_operator_name: str | None) -> tuple[str, bool]:
        """
        Имя оператора в эвристическом режиме — только из ручного forced_operator_name.
        Автоопределение по тексту отключено (для «Авто» используется облачный API в web_gui).
        """
        if forced_operator_name:
            normalized = self._normalize_operator_name(forced_operator_name)
            if normalized:
                return normalized, True
            return "Не определено", False
        return "Не определено", False

    @staticmethod
    def _extract_role_text(role_transcript: str | None, role: str) -> str:
        prefix = f"{role}:"
        rt = role_transcript or ""
        lines = [
            line[len(prefix):].strip()
            for line in rt.splitlines()
            if line.startswith(prefix)
        ]
        return " ".join(lines).strip()

    @staticmethod
    def _iter_role_blocks(role_transcript: str) -> list[tuple[str, str]]:
        """Разбор «Оператор: …» / «Заявитель: …» с переносами строк внутри блока."""
        if not role_transcript or not role_transcript.strip():
            return []
        blocks: list[tuple[str, str]] = []
        current_role: str | None = None
        chunks: list[str] = []
        header = re.compile(r"^(Оператор|Заявитель):\s*(.*)$")
        for line in role_transcript.splitlines():
            raw = line.strip()
            if not raw:
                continue
            m = header.match(raw)
            if m:
                if current_role is not None:
                    blocks.append((current_role, " ".join(chunks).strip()))
                current_role = m.group(1)
                first = m.group(2).strip()
                chunks = [first] if first else []
            elif current_role is not None:
                chunks.append(raw)
        if current_role is not None:
            blocks.append((current_role, " ".join(chunks).strip()))
        return blocks

    @staticmethod
    def _operator_script_closing_text(
        operator_text_raw: str,
        role_transcript: str | None,
        *,
        max_chars: int = 600,
        max_blocks: int = 4,
    ) -> str:
        """Последние реплики оператора — только для пунктов скрипта «в конце» (помощь / благодарность / прощание)."""
        blocks = CallQualityEvaluator._iter_role_blocks(role_transcript or "")
        op_only = [t for r, t in blocks if r == "Оператор" and (t or "").strip()]
        if op_only:
            tail = " ".join(op_only[-max_blocks:]).strip()
            if len(tail) > max_chars:
                tail = tail[-max_chars:]
            return tail.lower()
        ot = (operator_text_raw or "").strip()
        if not ot:
            return ""
        if len(ot) > max_chars:
            return ot[-max_chars:].lower()
        return ot.lower()

    def _extract_name_from_applicant_reply(self, text: str) -> str | None:
        """Имя из ответа заявителя после вопроса «как вас зовут» и т.п."""
        raw = text.strip()
        if not raw:
            return None
        low = raw.lower()

        structured = (
            r"меня\s+зовут[\s,:-]+([а-яё-]{2,30})\b",
            r"зовут\s+меня[\s,:-]+([а-яё-]{2,30})\b",
            r"\bэто\s+([а-яё-]{2,25})\b",
            r"\bимя\s+([а-яё-]{2,25})\b",
            r"\bя\s+([а-яё-]{2,25})\s*[,.]",
        )
        for pat in structured:
            m = re.search(pat, low)
            if m:
                tok = self._clean_name_token(m.group(1))
                if (
                    len(tok) >= 2
                    and tok not in self._APPLICANT_NAME_STOPWORDS
                    and re.fullmatch(r"[а-яё-]+", tok, re.IGNORECASE)
                ):
                    return self._title_cyrillic_name(tok)

        compact = re.sub(r"\s+", " ", low).strip()
        if len(compact) <= 45 and compact.count(" ") <= 4:
            words = re.findall(r"[а-яё]{2,}", compact)
            words = [w for w in words if w not in self._APPLICANT_NAME_STOPWORDS and len(w) >= 3]
            if len(words) == 1:
                return self._title_cyrillic_name(words[0])
            # «Мирослава Строфская.» — оператор обычно называет по имени; сохраняем оба слова для отчёта
            if len(words) == 2:
                return (
                    f"{self._title_cyrillic_name(words[0])} "
                    f"{self._title_cyrillic_name(words[1])}"
                )
        return None

    def _find_applicant_name_from_dialog(self, role_transcript: str) -> str | None:
        """
        1) Оператор спрашивает имя → следующая реплика заявителя (в т.ч. короткое «Наталья»).
        2) Любая реплика заявителя с «меня зовут …» / явной самопрезентацией.
        """
        blocks = self._iter_role_blocks(role_transcript)
        for i, (role, text) in enumerate(blocks):
            if role != "Оператор":
                continue
            low = text.lower()
            asked = any(re.search(p, low) for p in self._ASK_APPLICANT_NAME_PATTERNS)
            if not asked:
                continue
            for j in range(i + 1, len(blocks)):
                r2, t2 = blocks[j]
                if r2 == "Оператор":
                    break
                if r2 == "Заявитель":
                    name = self._extract_name_from_applicant_reply(t2)
                    if name:
                        return name
        for role, text in blocks:
            if role != "Заявитель":
                continue
            low = text.lower()
            if not re.search(
                r"меня\s+зовут|зовут\s+меня|это\s+[а-яё]{2,}|имя\s+[а-яё]{2,}", low
            ):
                continue
            name = self._extract_name_from_applicant_reply(text)
            if name:
                return name
        return None

    @staticmethod
    def _count_applicant_name_token_in_text(text_lower: str, applicant_name: str | None) -> int:
        """
        Сколько раз в тексте произносится **первое слово** имени заявителя (обычно имя, не фамилия).

        Для ответа «Имя Фамилия» в речи чаще только **имя** — ищем в первую очередь
        первое слово, а не целую строку (иначе «мирослава строфская» не матчит «Мирослава,»).
        Границы — по-кириллически, без ``\\b`` (на некоторых сборках оно хуже для «имя,»).
        """
        if not applicant_name or not (text_lower or "").strip():
            return 0
        parts = [p for p in re.split(r"\s+", applicant_name.strip().lower()) if p]
        if not parts:
            return 0
        primary = parts[0]
        if len(primary) < 2:
            return 0
        # Не буква кириллицы слева/справа (запятая, пробел, начало строки — ок)
        pat = rf"(?<![а-яё]){re.escape(primary)}(?![а-яё])"
        return len(re.findall(pat, text_lower))

    @classmethod
    def applicant_name_checklist_ok(
        cls,
        operator_text_lower: str,
        applicant_text_lower: str,
        applicant_name: str | None,
    ) -> tuple[bool, int, int]:
        """
        Пункт «Называл заявителя по имени ≥2 раз» (по смыслу для ЕКЦ):

        - **Да**, если оператор назвал имя заявителя **≥ 2 раза**, **или**
        - **Да**, если имя (первое слово из ``applicant_name``) прозвучало **≥ 3 раза**
          **во всём диалоге** (реплики оператора и заявителя вместе).

        Возвращает ``(ok, hits_operator, hits_full_dialog)``.
        """
        op_hits = cls._count_applicant_name_token_in_text(operator_text_lower, applicant_name)
        dialog = f"{operator_text_lower} {applicant_text_lower}".strip()
        all_hits = cls._count_applicant_name_token_in_text(dialog, applicant_name)
        ok = op_hits >= 2 or all_hits >= 3
        return ok, op_hits, all_hits

    def _find_applicant_name(self, transcript: str) -> str | None:
        """Эвристики по тексту (частота / фраза «меня зовут»). Полный role_transcript — в evaluate."""
        if not (transcript or "").strip():
            return None
        text = transcript.lower()

        for name in self._COMMON_NAMES:
            hits = len(re.findall(rf"\b{re.escape(name)}\b", text))
            if hits >= 2:
                return name[0].upper() + name[1:] if len(name) > 1 else name.upper()
            if hits == 1 and len(name) >= 4:
                # Редкие имена чаще уникальны в реплике заявителя
                if re.search(rf"(меня\s+зовут|зовут\s+меня|это\s+){re.escape(name)}\b", text):
                    return name[0].upper() + name[1:] if len(name) > 1 else name.upper()
        return None

    def evaluate(
        self,
        transcript: str,
        role_transcript: str,
        forced_operator_name: str | None = None,
    ) -> QualityEvaluation:
        operator_text_raw = self._extract_role_text(role_transcript, "Оператор")
        applicant_text_raw = self._extract_role_text(role_transcript, "Заявитель")
        operator_text = operator_text_raw.lower()
        applicant_text = applicant_text_raw.lower()
        dialog_text = f"{operator_text} {applicant_text}".strip()

        operator_name, operator_in_staff = self._find_operator_name(
            operator_text_raw or transcript,
            forced_operator_name,
        )
        applicant_name = self._find_applicant_name_from_dialog(role_transcript)
        if not applicant_name and (applicant_text_raw or "").strip():
            applicant_name = self._extract_name_from_applicant_reply(applicant_text_raw)
        applicant_name = applicant_name or self._find_applicant_name(applicant_text_raw or transcript)

        has_greeting = self._count_pattern_hits(operator_text, self._GREETING_PATTERNS) > 0
        has_intro = self._count_pattern_hits(operator_text, self._INTRO_PATTERNS) > 0
        closing_op = self._operator_script_closing_text(operator_text_raw, role_transcript)
        has_goodbye = self._count_pattern_hits(closing_op, self._GOODBYE_PATTERNS) > 0
        has_offer_help = self._count_pattern_hits(closing_op, self._OFFER_HELP_PATTERNS) > 0
        has_thank = self._count_pattern_hits(closing_op, self._THANK_PATTERNS) > 0
        # Упоминания имени: для чек-листа — оператор ≥2 ИЛИ по всему диалогу ≥3;
        # для «познакомился» — классически: спросил имя ИЛИ (известно имя и оператор ≥2).
        uses_name_checklist, op_name_hits, dialog_name_hits = self.applicant_name_checklist_ok(
            operator_text, applicant_text, applicant_name
        )
        operator_asked_name = self._count_pattern_hits(operator_text, self._ASK_APPLICANT_NAME_PATTERNS) > 0
        has_acquaintance = operator_asked_name or (
            applicant_name is not None and op_name_hits >= 2
        )

        # 7 script points, ~1.43 pts each → max 10
        script_score = round(
            (
                (1 if has_greeting else 0)
                + (1 if has_intro else 0)
                + (1 if has_acquaintance else 0)
                + (1 if uses_name_checklist else 0)
                + (1 if has_offer_help else 0)
                + (1 if has_thank else 0)
                + (1 if has_goodbye else 0)
            )
            * (10 / 7)
        )
        script_score = min(script_score, 10)

        filler_hits = len(re.findall("|".join(self._FILLER_PATTERNS), operator_text))
        diminutive_hits = len(re.findall("|".join(self._DIMINUTIVE_PATTERNS), operator_text))
        negative_hits = len(re.findall("|".join(self._NEGATIVE_PHRASES), operator_text))
        penalties = filler_hits + (2 * diminutive_hits) + (2 * negative_hits)
        speech_score = max(0, 10 - penalties)

        consultation_hits = self._count_pattern_hits(operator_text, self._CONSULTATION_PATTERNS)
        resolution_hits = self._count_pattern_hits(applicant_text or dialog_text, self._RESOLUTION_PATTERNS)
        consultation_score = min(10, consultation_hits + (2 * resolution_hits))
        operator_questions = len(re.findall(r"\?", operator_text_raw))
        engagement_hits = self._count_pattern_hits(operator_text, self._ENGAGEMENT_PATTERNS)
        engagement_score = min(10, engagement_hits * 2 + min(4, operator_questions))

        total_score = round(
            (0.30 * script_score)
            + (0.20 * speech_score)
            + (0.30 * consultation_score)
            + (0.20 * engagement_score)
        )
        max_score = 10

        if has_acquaintance and not operator_asked_name:
            acquaintance_label = "да (заявитель представился сам)"
        elif has_acquaintance:
            acquaintance_label = "да"
        else:
            acquaintance_label = "нет"
        script_details = [
            f"Приветствие + предложение помочь: {'да' if has_greeting else 'нет'}",
            f"Представился по имени: {'да' if has_intro else 'нет'}",
            f"Познакомился с заявителем: {acquaintance_label}",
            (
                f"Называл заявителя по имени ≥2 раз: {'да' if uses_name_checklist else 'нет'}"
                + (f" ({applicant_name})" if applicant_name else "")
                + (
                    f" [оп.{op_name_hits}/всего{dialog_name_hits}]"
                    if applicant_name and (op_name_hits or dialog_name_hits)
                    else ""
                )
            ),
            f"Предложил дополнительную помощь: {'да' if has_offer_help else 'нет'}",
            f"Поблагодарил за обращение: {'да' if has_thank else 'нет'}",
            f"Персонализированное прощание: {'да' if has_goodbye else 'нет'}",
        ]

        positives: list[str] = []
        negatives: list[str] = []

        if script_score >= 7:
            positives.append("Скрипт звонка в основном соблюден.")
        else:
            negatives.append("Нарушения по скрипту: проверьте приветствие/представление/завершение.")

        if speech_score >= 8:
            positives.append("Речь оператора чистая, без заметных паразитов и запрещенных формулировок.")
        else:
            negatives.append(
                "Есть речевые риски: слова-паразиты/уменьшительные формы/фразы 'к сожалению', 'не знаю'."
            )

        if consultation_score >= 7:
            positives.append("Консультация предметная, вероятно вопрос заявителя закрыт.")
        else:
            negatives.append("Консультация недостаточно четкая: не хватает шагов, сроков или фиксации результата.")
        if engagement_score >= 7:
            positives.append("Оператор проявляет вовлеченность и ведет диалог уточняющими вопросами.")
        else:
            negatives.append("Низкая вовлеченность: добавьте уточняющие вопросы и активное сопровождение заявителя.")
        if operator_name == "Не определено":
            negatives.append(
                "Имя оператора не извлечено из транскрипта (ожидаются фразы вроде «меня зовут …» "
                "или имя из штатного списка в репликах оператора; можно выбрать оператора вручную в интерфейсе)."
            )
        elif not operator_in_staff:
            negatives.append("Имя оператора не из штатного списка (показано как в транскрипте).")

        if not positives:
            positives.append("Сильные стороны слабо выражены в текущем транскрипте.")

        ev = QualityEvaluation(
            total_score=total_score,
            max_score=max_score,
            operator_name=operator_name,
            operator_in_staff=operator_in_staff,
            applicant_name=applicant_name,
            script_score=script_score,
            speech_score=speech_score,
            consultation_score=consultation_score,
            engagement_score=engagement_score,
            script_details=script_details,
            positives=positives,
            negatives=negatives,
        )
        from evaluation_leniency import apply_focus_operator_adjustments

        return apply_focus_operator_adjustments(ev)


def analyze_tone(audio_path: str | Path) -> str:
    """Approximate tone from signal features (energy/pitch variation)."""
    try:
        import librosa  # type: ignore
        import numpy as np
    except ImportError:
        return "Не определен (установите librosa для анализа тона)"

    try:
        y, sr = load_audio_mono_16k_safe(str(audio_path))
        if y.size == 0:
            return "Не определен (пустой аудиосигнал)"

        # Signal energy and variability.
        rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=256)[0]
        mean_rms = float(np.mean(rms))
        std_rms = float(np.std(rms))

        # Pitch statistics as rough emotional proxy.
        f0, _, _ = librosa.pyin(
            y,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"),
        )
        voiced = f0[~np.isnan(f0)]
        pitch_var = float(np.std(voiced)) if voiced.size else 0.0

        agitation = (std_rms * 4.0) + (pitch_var / 120.0) + (mean_rms * 2.0)
        if agitation < 0.8:
            return "Спокойный"
        if agitation < 1.8:
            return "Нейтральный"
        return "Напряженный/эмоциональный"
    except Exception:
        return "Не определен (ошибка анализа аудиосигнала)"


def build_report(result: TranscriptionResult, evaluation: QualityEvaluation) -> str:
    lines = [
        "",
        "=== Оценка консультации (ДВФУ) ===",
        f"Оператор: {evaluation.operator_name}",
        f"Оператор из штатного списка: {'да' if evaluation.operator_in_staff else 'нет'}",
        f"Заявитель: {evaluation.applicant_name or 'Не определено'}",
        f"Язык распознавания: {result.language or 'не определен'}",
        f"ASR профиль: {result.asr_profile}",
        f"ASR модель: {result.asr_model}",
        f"Роли в транскрипте: {result.role_attribution or 'не указано'}",
        f"Уверенность ролей: {result.role_confidence}",
        f"Тон голоса (оценка): {result.tone_summary}",
        f"Итоговая экспертная оценка: {evaluation.total_score}/{evaluation.max_score}",
        "",
        "Критерий 1. Соблюдение скрипта: "
        f"{evaluation.script_score}/10",
        "Критерий 2. Чистота речи оператора: "
        f"{evaluation.speech_score}/10",
        "Критерий 3. Качество консультации и решение вопроса: "
        f"{evaluation.consultation_score}/10",
        "Критерий 4. Вовлеченность оператора: "
        f"{evaluation.engagement_score}/10",
        "",
        "Проверка скрипта:",
    ]
    lines.extend(f"- {item}" for item in evaluation.script_details)
    lines.append("")
    lines.append("Плюсы (кратко):")
    lines.extend(f"- {item}" for item in evaluation.positives[:3])
    lines.append("")
    lines.append("Минусы (кратко):")
    if evaluation.negatives:
        lines.extend(f"- {item}" for item in evaluation.negatives[:3])
    else:
        lines.append("- Явных минусов по текущим критериям не выявлено.")
    lines.append("")
    if result.role_diagnostics:
        lines.append("Диагностика ролей:")
        lines.append(result.role_diagnostics)
        lines.append("")
    lines.append("Транскрипт по ролям:")
    lines.append(result.role_text if result.role_text else "<пусто>")
    lines.append("")
    lines.append("Сплошной транскрипт:")
    lines.append(result.text if result.text else "<пусто>")
    return "\n".join(lines)


def _default_report_path(audio_path: str | Path) -> Path:
    from results_paths import ensure_results_dir

    source = Path(audio_path)
    stem = source.stem if source.stem else "report"
    return ensure_results_dir() / f"{stem}_report.txt"


def main() -> None:
    from analysis_service import (
        AnalysisRequest,
        TranscriberSettings,
        analyze_call,
        build_sheets_export_payload,
        create_transcriber,
    )

    load_app_environment()

    parser = argparse.ArgumentParser(
        description="Транскрибация и базовая оценка качества разговоров колл-центра ДВФУ."
    )
    parser.add_argument("audio", nargs="?", help="Путь к аудиофайлу (wav/mp3/m4a и т.п.)")
    parser.add_argument(
        "--model",
        default="best-ru",
        help="Whisper model name (medium|large-v3|best-ru)",
    )
    parser.add_argument("--compute-type", default="int8", help="Whisper compute_type")
    parser.add_argument(
        "--asr-profile",
        default="medium_ru",
        choices=["medium_ru", "ideal_ru", "ultima_ru"],
        help="Профиль ASR-декодинга: medium_ru|ideal_ru|ultima_ru (Ultima = whisper-large-v3-russian, HF)",
    )
    parser.add_argument(
        "--heavy-diarization",
        action="store_true",
        help="Включить heavy diarization как основной путь с fallback на light clustering",
    )
    parser.add_argument(
        "--heavy-diarization-timeout-seconds",
        type=int,
        default=240,
        help="Таймаут heavy diarization в секундах",
    )
    parser.add_argument(
        "--llm-backend",
        choices=["off", "embedded", "remote"],
        default="off",
        help=(
            "LLM пост-редактор: off | embedded (DeepSeek с Hugging Face, без Ollama) | "
            "remote (нужны LLM_POST_EDIT_BASE_URL и LLM_POST_EDIT_MODEL)"
        ),
    )
    parser.add_argument(
        "--llm-embedded-model",
        default=None,
        help="Репозиторий Hugging Face для встроенной модели (иначе LLM_EMBEDDED_MODEL_ID или дефолт DeepSeek-R1 1.5B)",
    )
    parser.add_argument(
        "--llm-post-edit",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--llm-post-edit-timeout-seconds",
        type=float,
        default=120.0,
        help="Таймаут LLM пост-редактора (сек)",
    )
    parser.add_argument(
        "--enable-post-edit",
        action="store_true",
        help="Включить локальный пост-редактор пунктуации (medium_ru, ideal_ru, ultima_ru)",
    )
    parser.add_argument(
        "--post-edit-timeout-seconds",
        type=int,
        default=60,
        help="Таймаут локального пост-редактора в секундах",
    )
    parser.add_argument(
        "--total-time-budget-seconds",
        type=int,
        default=300,
        help="Общий тайм-бюджет обработки звонка в секундах",
    )
    parser.add_argument(
        "--skip-first-seconds",
        type=float,
        default=None,
        help=(
            "Пропустить N секунд с начала (clip_timestamps). "
            "Если не задано — из имени файла берётся ожидание как последний фрагмент _MM-SS (минуты-секунды)."
        ),
    )
    parser.add_argument("--language", default="ru", help="Язык распознавания (фиксирован ru)")
    parser.add_argument(
        "--model-load-timeout-seconds",
        type=int,
        default=600,
        help="Таймаут загрузки модели в секундах",
    )
    parser.add_argument(
        "--operator-name",
        default=None,
        help="Имя оператора (если не указать, будет автоопределение по транскрипту)",
    )
    parser.add_argument(
        "--warmup-only",
        action="store_true",
        help="Только предварительно скачать/загрузить модель, без анализа аудио",
    )
    parser.add_argument(
        "--report-path",
        default=None,
        help="Куда сохранить итоговый отчет (.txt). По умолчанию рядом с запуском.",
    )
    parser.add_argument(
        "--no-google-sheets",
        action="store_true",
        help="Не отправлять строку в Google Таблицу (если настроен GOOGLE_SHEETS_CREDENTIALS_JSON)",
    )
    parser.add_argument(
        "--cloud-eval",
        action="store_true",
        help="Использовать облачную оценку Yandex (нужны YANDEX_API_KEY и YANDEX_FOLDER_ID).",
    )
    parser.add_argument(
        "--cloud-eval-extra-instructions",
        default=None,
        help="Дополнительные указания для облачной оценки Yandex.",
    )
    args = parser.parse_args()
    llm_backend = args.llm_backend
    if getattr(args, "llm_post_edit", False) and llm_backend == "off":
        llm_backend = "remote"
    enable_llm = llm_backend != "off"
    active_backend = llm_backend if enable_llm else "embedded"
    transcriber_settings = TranscriberSettings(
        model_name=args.model,
        compute_type=args.compute_type,
        language=args.language,
        model_load_timeout_seconds=args.model_load_timeout_seconds,
        asr_profile=args.asr_profile,
        heavy_diarization=args.heavy_diarization,
        heavy_diarization_timeout_seconds=args.heavy_diarization_timeout_seconds,
        enable_post_edit=args.enable_post_edit,
        post_edit_timeout_seconds=args.post_edit_timeout_seconds,
        total_time_budget_seconds=args.total_time_budget_seconds,
        enable_llm_post_edit=enable_llm,
        llm_backend=active_backend,
        llm_embedded_model_id=args.llm_embedded_model,
        llm_post_edit_timeout_seconds=args.llm_post_edit_timeout_seconds,
    )
    if args.warmup_only:
        print("[transcriber] Режим warmup: только загрузка модели...", flush=True)
        transcriber = create_transcriber(transcriber_settings)
        transcriber._ensure_model()
        print("[transcriber] Warmup завершен. Модель готова к работе.", flush=True)
        return

    if not args.audio:
        raise SystemExit("Укажите путь к аудио или используйте --warmup-only")

    cloud_eval_cfg = None
    if args.cloud_eval:
        from llm_cloud_eval import load_cloud_config_from_env

        cloud_eval_cfg = load_cloud_config_from_env()
        if cloud_eval_cfg is None:
            raise SystemExit(
                "Для --cloud-eval задайте YANDEX_API_KEY и YANDEX_FOLDER_ID в .env или окружении."
            )

    analysis = analyze_call(
        AnalysisRequest(
            audio_path=args.audio,
            operator_name=args.operator_name,
            original_basename=Path(args.audio).name,
            skip_first_seconds=args.skip_first_seconds,
            cloud_eval_cfg=cloud_eval_cfg,
            cloud_eval_extra_instructions=args.cloud_eval_extra_instructions,
        ),
        transcriber=create_transcriber(transcriber_settings),
        on_log=lambda message: print(message, flush=True),
    )

    report = analysis.report
    print(report)
    output_path = Path(args.report_path) if args.report_path else _default_report_path(args.audio)
    output_path.write_text(report, encoding="utf-8")
    print(f"[evaluator] Отчет сохранен: {output_path}", flush=True)

    if not args.no_google_sheets:
        try:
            from sheets_export import append_analysis_row

            fn = Path(args.audio).name
            ok, msg = append_analysis_row(
                **build_sheets_export_payload(
                    analysis,
                    original_filename=fn,
                )
            )
            tag = "[sheets] OK" if ok else "[sheets] skip/err"
            print(f"{tag}: {msg}", flush=True)
        except Exception as exc:
            print(f"[sheets] Ошибка: {exc}", flush=True)


if __name__ == "__main__":
    main()
