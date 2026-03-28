from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar

from audio_io_safe import load_audio_mono_16k_safe

if TYPE_CHECKING:
    import numpy as np

LabelT = TypeVar("LabelT")


def _normalize_rows(x):
    import numpy as np

    norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-8
    return x / norms


def _forced_bipartition_labels(embs):
    """
    Принудительный бинарный split:
    1) time-aware max-contrast split
    2) fallback: median split по первой principal оси
    """
    import numpy as np

    n = len(embs)
    if n < 2:
        return None
    min_side = max(1, int(0.15 * n))

    # 1) Time-aware split по максимуму межгрупповой дистанции.
    best_idx = None
    best_score = -1.0
    for cut in range(min_side, n - min_side + 1):
        left = embs[:cut]
        right = embs[cut:]
        lc = np.mean(left, axis=0)
        rc = np.mean(right, axis=0)
        lc = lc / (np.linalg.norm(lc) + 1e-8)
        rc = rc / (np.linalg.norm(rc) + 1e-8)
        inter = 1.0 - float(np.dot(lc, rc))
        if inter > best_score:
            best_score = inter
            best_idx = cut
    if best_idx is not None:
        labels = np.zeros(n, dtype=int)
        labels[best_idx:] = 1
        if len(set(labels.tolist())) == 2:
            return labels

    # 2) PCA-like fallback: порог по медиане проекции.
    centered = embs - np.mean(embs, axis=0, keepdims=True)
    try:
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        axis = vt[0]
        proj = centered @ axis
        thr = float(np.median(proj))
        labels = (proj >= thr).astype(int)
    except Exception:
        # last fallback: trivial split by index
        labels = np.zeros(n, dtype=int)
        labels[n // 2 :] = 1
    if len(set(labels.tolist())) < 2:
        labels = np.zeros(n, dtype=int)
        labels[n // 2 :] = 1
    return labels


def _merge_adjacent_labeled_spans(
    spans: list[tuple[float, float, LabelT]],
    *,
    max_gap_sec: float = 0.12,
) -> list[tuple[float, float, LabelT]]:
    if not spans:
        return []
    ordered = sorted(spans, key=lambda item: (item[0], item[1]))
    merged: list[tuple[float, float, LabelT]] = []
    for start, end, label in ordered:
        if end - start <= 1e-6:
            continue
        if not merged:
            merged.append((start, end, label))
            continue
        prev_start, prev_end, prev_label = merged[-1]
        if label == prev_label and start <= prev_end + max_gap_sec:
            merged[-1] = (prev_start, max(prev_end, end), prev_label)
        else:
            merged.append((start, end, label))
    return merged


def _smooth_short_label_islands(
    spans: list[tuple[float, float, LabelT]],
    *,
    min_island_sec: float = 0.45,
    passes: int = 2,
    max_gap_sec: float = 0.12,
) -> list[tuple[float, float, LabelT]]:
    if len(spans) < 3:
        return spans
    current = list(spans)
    for _ in range(max(1, passes)):
        changed = False
        updated = current[:]
        for i in range(1, len(current) - 1):
            left = current[i - 1]
            mid = current[i]
            right = current[i + 1]
            mid_dur = mid[1] - mid[0]
            if mid_dur >= min_island_sec:
                continue
            if left[2] == right[2] and mid[2] != left[2]:
                updated[i] = (mid[0], mid[1], left[2])
                changed = True
        current = _merge_adjacent_labeled_spans(updated, max_gap_sec=max_gap_sec)
        if not changed:
            break
    return current


def _build_speaker_spans_from_windows(
    window_times: list[float],
    labels,
    window_sec: float,
    *,
    max_gap_sec: float = 0.12,
    min_span_sec: float = 0.45,
) -> list[tuple[float, float, int]]:
    import numpy as np

    label_arr = np.asarray(labels, dtype=np.int64).reshape(-1)
    if not window_times or len(window_times) != len(label_arr):
        return []
    if window_sec <= 0:
        return []

    breakpoints: list[float] = []
    for start in window_times:
        breakpoints.append(float(start))
        breakpoints.append(float(start) + float(window_sec))
    breakpoints = sorted(set(breakpoints))
    if len(breakpoints) < 2:
        return []

    atomic: list[tuple[float, float, int]] = []
    for idx in range(len(breakpoints) - 1):
        span_start = breakpoints[idx]
        span_end = breakpoints[idx + 1]
        if span_end - span_start <= 0.03:
            continue
        votes: dict[int, float] = {}
        for win_start, label in zip(window_times, label_arr.tolist()):
            win_end = float(win_start) + float(window_sec)
            overlap = max(0.0, min(span_end, win_end) - max(span_start, float(win_start)))
            if overlap > 0.0:
                votes[int(label)] = votes.get(int(label), 0.0) + overlap
        if not votes:
            continue
        chosen = max(votes, key=lambda key: (votes[key], -key))
        atomic.append((span_start, span_end, int(chosen)))

    merged = _merge_adjacent_labeled_spans(atomic, max_gap_sec=max_gap_sec)
    merged = _smooth_short_label_islands(
        merged,
        min_island_sec=min_span_sec,
        max_gap_sec=max_gap_sec,
    )
    return [
        (start, end, int(label))
        for start, end, label in merged
        if (end - start) >= 0.08
    ]


def _nearest_speaker_label(
    mid: float,
    speaker_spans: list[tuple[float, float, LabelT]],
) -> LabelT:
    best_label = speaker_spans[0][2]
    best_distance = math.inf
    for span_start, span_end, label in speaker_spans:
        distance = abs(mid - ((span_start + span_end) * 0.5))
        if distance < best_distance:
            best_distance = distance
            best_label = label
    return best_label


def _split_text_by_duration(
    text: str,
    durations: list[float],
) -> list[str] | None:
    words = [word for word in str(text).split() if word]
    if not words or len(durations) < 2:
        return None
    if len(words) < len(durations) * 2:
        return None

    total_words = len(words)
    total_duration = max(1e-6, sum(max(0.0, value) for value in durations))
    parts: list[str] = []
    prev_cut = 0
    cumulative = 0.0
    for idx, duration in enumerate(durations[:-1]):
        cumulative += max(0.0, duration)
        target_cut = int(round(total_words * (cumulative / total_duration)))
        min_cut = prev_cut + 1
        max_cut = total_words - (len(durations) - idx - 1)
        cut = max(min_cut, min(target_cut, max_cut))
        parts.append(" ".join(words[prev_cut:cut]).strip())
        prev_cut = cut
    parts.append(" ".join(words[prev_cut:]).strip())
    if any(not part for part in parts):
        return None
    return parts


def _remap_segments_by_speaker_spans(
    segments: list[tuple[float, float, str]],
    speaker_spans: list[tuple[float, float, LabelT]],
    *,
    min_split_sec: float = 0.85,
    min_split_share: float = 0.22,
    min_words_to_split: int = 6,
) -> list[tuple[float, float, str, LabelT]]:
    if not segments or not speaker_spans:
        return []

    out: list[tuple[float, float, str, LabelT]] = []
    for seg_start, seg_end, text in segments:
        seg_s = float(seg_start)
        seg_e = max(seg_s, float(seg_end))
        seg_mid = (seg_s + seg_e) * 0.5
        seg_duration = max(1e-6, seg_e - seg_s)

        overlaps: list[tuple[float, float, LabelT]] = []
        for span_start, span_end, label in speaker_spans:
            ov_start = max(seg_s, float(span_start))
            ov_end = min(seg_e, float(span_end))
            if ov_end - ov_start >= 0.05:
                overlaps.append((ov_start, ov_end, label))

        if not overlaps:
            out.append((seg_s, seg_e, text, _nearest_speaker_label(seg_mid, speaker_spans)))
            continue

        merged = _merge_adjacent_labeled_spans(overlaps, max_gap_sec=0.05)
        dominant_by_label: dict[LabelT, float] = {}
        for ov_start, ov_end, label in merged:
            dominant_by_label[label] = dominant_by_label.get(label, 0.0) + (ov_end - ov_start)
        dominant_label = max(dominant_by_label, key=dominant_by_label.get)

        if len(merged) == 1:
            out.append((seg_s, seg_e, text, dominant_label))
            continue

        long_groups = [
            group
            for group in merged
            if (group[1] - group[0]) >= min_split_sec
            and ((group[1] - group[0]) / seg_duration) >= min_split_share
        ]
        can_split = (
            len({group[2] for group in merged}) >= 2
            and len(merged) <= 3
            and len(long_groups) >= 2
            and len(str(text).split()) >= min_words_to_split
            and seg_duration >= max(1.7, min_split_sec * 1.8)
        )
        if not can_split:
            out.append((seg_s, seg_e, text, dominant_label))
            continue

        durations = [group[1] - group[0] for group in merged]
        text_parts = _split_text_by_duration(text, durations)
        if not text_parts:
            out.append((seg_s, seg_e, text, dominant_label))
            continue

        for (piece_start, piece_end, label), piece_text in zip(merged, text_parts):
            cleaned = piece_text.strip()
            if not cleaned:
                continue
            out.append((piece_start, piece_end, cleaned, label))

    return out


def _patch_torchaudio_compat() -> None:
    """Делегирует в общий модуль: непустой list_audio_backends (soundfile), иначе ложный варнинг SB."""
    try:
        from speechbrain_compat import apply_torchaudio_speechbrain_compat

        apply_torchaudio_speechbrain_compat()
    except Exception:
        pass


# Module-level cache so the model is loaded only once per process.
_ECAPA_MODEL_CACHE: object | None = None
_ECAPA_SAVEDIR = "pretrained_models/spkrec-ecapa-voxceleb"
_ECAPA_SOURCE = "speechbrain/spkrec-ecapa-voxceleb"


def _best_torch_device() -> str:
    """Возвращает наилучшее доступное устройство: mps > cpu."""
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def _load_ecapa_model():
    """Load (or return cached) SpeechBrain ECAPA-TDNN speaker encoder.

    Forces offline mode so HuggingFace Hub network checks are skipped entirely —
    the model is already on disk and we never want to hang waiting for a remote
    version check.  `HF_HUB_OFFLINE` is restored after loading.
    Uses MPS (Apple Silicon GPU) if available for faster inference.

    Lock: не пересекаемся с загрузкой faster-whisper (Ultima / HF), иначе тот поток
    видит HF_HUB_OFFLINE=1 и падает с «outgoing traffic has been disabled».
    """
    global _ECAPA_MODEL_CACHE
    if _ECAPA_MODEL_CACHE is not None:
        return _ECAPA_MODEL_CACHE

    import os

    from hf_hub_sync import (
        HF_HUB_DOWNLOAD_LOCK,
        recompute_huggingface_hub_offline_from_env,
        set_huggingface_hub_offline_flag,
        snapshot_huggingface_hub_offline_flag,
    )

    with HF_HUB_DOWNLOAD_LOCK:
        _prev_offline = os.environ.get("HF_HUB_OFFLINE")
        _prev_hf_const = snapshot_huggingface_hub_offline_flag()
        os.environ["HF_HUB_OFFLINE"] = "1"
        set_huggingface_hub_offline_flag(True)
        try:
            _patch_torchaudio_compat()
            from speechbrain.inference.speaker import EncoderClassifier

            device = _best_torch_device()
            model = EncoderClassifier.from_hparams(
                source=_ECAPA_SOURCE,
                savedir=_ECAPA_SAVEDIR,
                run_opts={"device": device},
            )
            _ECAPA_MODEL_CACHE = model
        finally:
            if _prev_offline is None:
                os.environ.pop("HF_HUB_OFFLINE", None)
            else:
                os.environ["HF_HUB_OFFLINE"] = _prev_offline
            if _prev_hf_const is not None:
                set_huggingface_hub_offline_flag(_prev_hf_const)
            else:
                recompute_huggingface_hub_offline_from_env()
    return _ECAPA_MODEL_CACHE


def warmup_ecapa_model_background() -> None:
    """Загружает ECAPA-TDNN в фоновом потоке при старте приложения.

    Вызывается один раз из web_gui.py при инициализации, чтобы к моменту
    первой загрузки файла модель уже была в памяти.
    """
    import threading

    def _worker() -> None:
        try:
            _load_ecapa_model()
        except Exception:
            pass  # тихий прогрев — ошибки не критичны

    t = threading.Thread(target=_worker, daemon=True, name="ecapa-warmup")
    t.start()


def _embed_windows_ecapa(
    model: object,
    windows: "list[np.ndarray]",
    target_len: int,
    batch_size: int = 64,
) -> "np.ndarray":
    """Batch-embed a list of audio windows using ECAPA-TDNN.

    All windows are zero-padded / truncated to *target_len* samples so they
    can be stacked into a single tensor batch.  Tensors are moved to the
    model's device (MPS / CPU) automatically.
    """
    import numpy as np
    import torch

    # Determine which device the model lives on.
    try:
        _dev = next(iter(model.parameters())).device  # type: ignore[union-attr]
    except Exception:
        _dev = torch.device("cpu")

    padded: list[np.ndarray] = []
    for w in windows:
        if len(w) >= target_len:
            padded.append(w[:target_len])
        else:
            padded.append(np.pad(w, (0, target_len - len(w)), mode="constant"))

    enc = getattr(model, "encode_batch")
    rows: list[np.ndarray] = []
    for start in range(0, len(padded), batch_size):
        chunk = padded[start : start + batch_size]
        batch_t = torch.tensor(np.stack(chunk), dtype=torch.float32).to(_dev)
        lens_t = torch.ones(len(chunk), device=_dev)
        with torch.no_grad():
            out = enc(batch_t, lens_t).squeeze(1).detach().cpu().numpy()
        rows.append(out)
    return np.vstack(rows)


def cluster_voice_segments(
    audio_path: str | Path,
    segments: list[tuple[float, float, str]],
    target_speakers: int = 2,
) -> tuple[list[tuple[float, float, str, str]] | None, str, dict[str, float | int | str]]:
    """
    Голосовая кластеризация через VAD-регионы + SpeechBrain ECAPA-TDNN:
    - VAD (librosa.effects.split) выявляет участки речи, разделённые паузами
    - каждый VAD-регион нарезается на 1.2с окна → эмбеддинги почти не содержат «смешанных» спикеров
    - ECAPA-TDNN (192-dim, обучен на VoxCeleb2) — качественнее resemblyzer d-vector
    - батчевый инференс — быстрее, чем посегментно
    - кластеризация в 2 группы
    - маппинг обратно к Whisper-сегментам через временное перекрытие
    Возвращает (start, end, text, cluster_id).
    """
    diagnostics: dict[str, float | int | str] = {
        "input_segments": len(segments),
        "valid_segments": 0,
        "window_count": 0,
        "inter_cluster_distance": 0.0,
        "intra_cluster_distance": 0.0,
        "cluster_quality": 0.0,
    }

    try:
        import numpy as np
        import librosa
        from sklearn.cluster import AgglomerativeClustering
    except Exception as exc:
        diagnostics["reason"] = f"import_error: {exc}"
        return None, f"не удалось импортировать зависимости: {exc}", diagnostics

    # SpeechBrain ECAPA-TDNN — загружаем один раз, кэшируем.
    try:
        ecapa_model = _load_ecapa_model()
    except Exception as exc:
        diagnostics["reason"] = f"ecapa_load_error: {exc}"
        return None, f"не удалось загрузить ECAPA-TDNN: {exc}", diagnostics

    path = Path(audio_path)
    if not path.exists():
        diagnostics["reason"] = "audio_not_found"
        return None, f"аудиофайл не найден: {path}", diagnostics
    if len(segments) < 2:
        diagnostics["reason"] = "too_few_segments"
        return None, "слишком мало сегментов для голосовой кластеризации", diagnostics

    # Determine the audio range actually covered by the segments, and add a
    # small safety margin.  Loading only this slice instead of the whole file
    # can save 30-90 seconds for recordings where the first few minutes are
    # informer music / silence that was already skipped by ASR.
    seg_start_sec = max(0.0, min(s for s, e, *_ in segments) - 1.0)
    seg_end_sec = max(s for s, e, *_ in segments) + 30.0  # generous tail margin
    load_offset = seg_start_sec
    load_duration = seg_end_sec - seg_start_sec

    try:
        wav, sr = load_audio_mono_16k_safe(
            str(path),
            offset=load_offset,
            duration=load_duration,
        )
    except Exception as exc:
        diagnostics["reason"] = f"audio_read_error: {exc}"
        return None, f"не удалось прочитать аудио: {exc}", diagnostics
    if wav.size == 0:
        diagnostics["reason"] = "empty_audio"
        return None, "пустой аудиосигнал", diagnostics

    # All segment timestamps in the rest of this function must be shifted by
    # the load offset so that they correctly index into the trimmed waveform.
    _ts_shift = load_offset

    def _rms_energy(x: "np.ndarray") -> float:
        return float(np.sqrt(np.mean(np.square(x))) + 1e-9)

    # ------------------------------------------------------------------
    # 1. VAD-регионы по всему файлу (librosa.effects.split)
    # Смена спикера почти всегда совпадает с паузой, поэтому каждый
    # VAD-регион с высокой вероятностью принадлежит одному спикеру.
    # top_db=28 (было 35) — не срезаем тихую речь: шёпот, «алло», начало фразы.
    # ------------------------------------------------------------------
    try:
        raw_vad = librosa.effects.split(wav, top_db=28)
    except Exception as exc:
        diagnostics["reason"] = f"vad_error: {exc}"
        return None, f"ошибка VAD: {exc}", diagnostics

    vad_regions: list[tuple[float, float]] = []
    for vs_samp, ve_samp in raw_vad.tolist():
        ss = float(vs_samp) / sr
        ee = float(ve_samp) / sr
        # 0.25 с вместо 0.35 — ловим очень короткие реплики («да», «нет», «ага»).
        if ee - ss < 0.25:
            continue
        # Склеить короткие паузы < 200ms — одна реплика часто содержит паузы внутри.
        if vad_regions and ss - vad_regions[-1][1] < 0.20:
            vad_regions[-1] = (vad_regions[-1][0], ee)
        else:
            vad_regions.append((ss, ee))

    if not vad_regions:
        diagnostics["reason"] = "no_vad_regions"
        return None, "не найдено голосовых регионов (VAD)", diagnostics

    # ------------------------------------------------------------------
    # 2. Нарезать VAD-регионы на окна для эмбеддингов.
    #    ECAPA-TDNN принимает сырую нормализованную волну (float32, 16kHz).
    #
    #    Параметры подобраны для баланса точность/скорость:
    #    - window_sec=2.0  — достаточный контекст для стабильного d-vector у ECAPA
    #    - hop_sec=1.5     — 25% перекрытие (было 2.0 = без перекрытия; 1.0 = слишком много окон).
    #                        Перекрывающиеся окна дают лучшее покрытие смен спикера.
    #    - MAX_WINDOWS=240 — разумный лимит для длинных записей
    # ------------------------------------------------------------------
    window_sec = 2.0
    hop_sec = 1.5
    win_samp = int(window_sec * 16000)
    hop_samp = int(hop_sec * 16000)
    MAX_WINDOWS = 240

    # Оцениваем порог энергии по всем регионам.
    energy_samples: list[float] = []
    for vs, ve in vad_regions:
        i0, i1 = int(vs * sr), min(int(ve * sr), len(wav))
        chunk = wav[i0:i1]
        if chunk.size >= int(0.20 * sr):
            energy_samples.append(_rms_energy(chunk))
    base_thr = float(np.median(energy_samples)) if energy_samples else 0.0
    energy_thr = max(0.002, base_thr * 0.35)

    # (start_sec_of_window, audio_array)
    vad_window_times: list[float] = []
    vad_window_audio: list["np.ndarray"] = []

    for vs, ve in vad_regions:
        i0, i1 = int(vs * sr), min(int(ve * sr), len(wav))
        region = wav[i0:i1]
        if region.size < int(0.25 * sr):
            continue
        # Normalize to [-1, 1] — ECAPA-TDNN expects float32 waveform.
        peak = float(np.max(np.abs(region)) + 1e-8)
        proc = (region / peak).astype("float32")
        if proc.size < int(0.20 * 16000):
            continue

        if proc.size <= win_samp:
            if _rms_energy(proc) >= energy_thr:
                vad_window_times.append(vs)
                vad_window_audio.append(proc)
        else:
            for j in range(0, proc.size - win_samp + 1, hop_samp):
                piece = proc[j : j + win_samp]
                if _rms_energy(piece) >= energy_thr:
                    t = vs + j / 16000.0
                    vad_window_times.append(t)
                    vad_window_audio.append(piece)

    # Равномерная подвыборка если окон слишком много.
    if len(vad_window_times) > MAX_WINDOWS:
        step = len(vad_window_times) / MAX_WINDOWS
        indices = [int(i * step) for i in range(MAX_WINDOWS)]
        vad_window_times = [vad_window_times[i] for i in indices]
        vad_window_audio = [vad_window_audio[i] for i in indices]

    # Shift VAD window times from local (trimmed wav) coordinates to
    # absolute file coordinates so they match the segment timestamps.
    vad_window_times = [t + _ts_shift for t in vad_window_times]

    diagnostics["window_count"] = len(vad_window_times)

    if len(vad_window_times) < 2:
        diagnostics["reason"] = "too_few_vad_windows"
        return None, "недостаточно голосовых окон после VAD", diagnostics

    # ------------------------------------------------------------------
    # 3. Строим эмбеддинги через ECAPA-TDNN (батчевый инференс) и кластеризуем.
    #    ECAPA-TDNN (192-dim, VoxCeleb2) значительно точнее resemblyzer d-vector.
    # ------------------------------------------------------------------
    try:
        embs = _embed_windows_ecapa(ecapa_model, vad_window_audio, win_samp, batch_size=64)
        embs = _normalize_rows(embs)
    except Exception as exc:
        diagnostics["reason"] = f"embedding_error: {exc}"
        return None, f"не удалось построить голосовые эмбеддинги: {exc}", diagnostics

    n_clusters = min(target_speakers, len(embs))
    if n_clusters < 2:
        diagnostics["reason"] = "too_few_embeddings"
        return None, "недостаточно данных для 2 кластеров", diagnostics

    forced_mode = False
    try:
        model = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric="cosine",
            linkage="average",
        )
        labels = model.fit_predict(embs)
    except Exception as exc:
        diagnostics["reason"] = f"clustering_error: {exc}; forced_split"
        labels = _forced_bipartition_labels(embs)
        forced_mode = True
        if labels is None:
            return None, f"ошибка кластеризации: {exc}", diagnostics

    cluster_ids = sorted(set(int(x) for x in labels))
    if len(cluster_ids) < 2:
        labels = _forced_bipartition_labels(embs)
        forced_mode = True
        if labels is None:
            diagnostics["reason"] = "single_cluster"
            return None, "кластеризация вернула только одного спикера", diagnostics
        cluster_ids = sorted(set(int(x) for x in labels))

    # ------------------------------------------------------------------
    # 4. Диагностика качества кластеров.
    # ------------------------------------------------------------------
    def _mean_cosine_distance(a: "np.ndarray", b: "np.ndarray") -> float:
        dists = 1.0 - np.sum(a * b, axis=1)
        return float(np.mean(dists))

    intra_vals: list[float] = []
    cluster_centroids: dict[int, "np.ndarray"] = {}
    for cid in cluster_ids:
        cemb = embs[labels == cid]
        center = np.mean(cemb, axis=0)
        center = center / (np.linalg.norm(center) + 1e-8)
        cluster_centroids[cid] = center
        intra_vals.append(
            _mean_cosine_distance(
                cemb, np.repeat(center[None, :], len(cemb), axis=0)
            )
        )

    inter_vals: list[float] = []
    for i, c1 in enumerate(cluster_ids):
        for c2 in cluster_ids[i + 1 :]:
            inter_vals.append(
                1.0 - float(np.dot(cluster_centroids[c1], cluster_centroids[c2]))
            )

    intra = float(np.mean(intra_vals)) if intra_vals else 0.0
    inter = float(np.mean(inter_vals)) if inter_vals else 0.0
    quality = inter / (intra + 1e-6)
    diagnostics["inter_cluster_distance"] = inter
    diagnostics["intra_cluster_distance"] = intra
    diagnostics["cluster_quality"] = quality
    # Повышенный порог (1.3 вместо 1.05): если кластеры плохо разделены, лучше использовать
    # time-aware forced split, который учитывает временну̀ю структуру диалога.
    if quality < 1.3:
        labels = _forced_bipartition_labels(embs)
        forced_mode = True
        if labels is not None:
            cluster_ids = sorted(set(int(x) for x in labels))

    # Двойной голосовой энкодер: Resemblyzer (d-vector) + согласование с ECAPA на тех же окнах.
    from speaker_voice_dual_encoder import fuse_ecapa_resemblyzer_from_windows

    labels = fuse_ecapa_resemblyzer_from_windows(vad_window_audio, embs, labels, diagnostics)

    # ------------------------------------------------------------------
    # 5. Строим сглаженные speaker-spans из перекрывающихся VAD-окон и
    #    ремапим Whisper-сегменты на них. Если один Whisper-фрагмент
    #    перекрывает смену спикера, режем его на 2-3 части пропорционально
    #    длительности голоса разных кластеров.
    # ------------------------------------------------------------------
    speaker_spans_raw = _build_speaker_spans_from_windows(
        vad_window_times,
        labels,
        window_sec,
        max_gap_sec=0.18,
        min_span_sec=0.45,
    )
    diagnostics["speaker_span_count"] = len(speaker_spans_raw)
    speaker_spans = [
        (start, end, f"SPK_{int(label)}")
        for start, end, label in speaker_spans_raw
    ]
    out = _remap_segments_by_speaker_spans(
        segments,
        speaker_spans,
        min_split_sec=0.85,
        min_split_share=0.22,
        min_words_to_split=6,
    )
    diagnostics["split_segments"] = max(0, len(out) - len(segments))
    if not out:
        diagnostics["reason"] = "speaker_span_remap_failed"
        return None, "не удалось сопоставить ASR-сегменты со спикерами", diagnostics

    # ------------------------------------------------------------------
    # 6. Сглаживание одиночных перескоков A-B-A на коротких репликах.
    # Два прохода: первый — очевидные одиночные флипы;
    # второй — убирает «реликты» после первого прохода.
    # Порог увеличен до 1.5 с (было 0.8 с) — убираем больше ошибочных смен.
    # ------------------------------------------------------------------
    smoothed = out[:]
    backup_two_cluster = out[:]
    for _pass in range(2):
        for i in range(1, len(smoothed) - 1):
            prev_c = smoothed[i - 1][3]
            cur_c = smoothed[i][3]
            next_c = smoothed[i + 1][3]
            cur_len = smoothed[i][1] - smoothed[i][0]
            if prev_c == next_c and cur_c != prev_c and cur_len < 1.5:
                smoothed[i] = (smoothed[i][0], smoothed[i][1], smoothed[i][2], prev_c)

    if len({x[3] for x in smoothed}) < 2:
        if len({x[3] for x in backup_two_cluster}) >= 2:
            smoothed = backup_two_cluster
            forced_mode = True
        else:
            if len(smoothed) >= 2:
                repaired: list[tuple[float, float, str, str]] = []
                cut = max(1, len(smoothed) // 2)
                for i, (s, e, t, _cid) in enumerate(smoothed):
                    repaired.append((s, e, t, "SPK_0" if i < cut else "SPK_1"))
                smoothed = repaired
                forced_mode = True
                diagnostics["reason"] = "hard_forced_index_split"
            else:
                diagnostics["reason"] = "single_cluster_after_smoothing"
                return None, "кластеризация вернула только одного спикера", diagnostics

    diagnostics["reason"] = "ok_forced_split" if forced_mode else "ok_native"
    diagnostics["split_mode"] = "forced" if forced_mode else "native"
    _dual = diagnostics.get("dual_voice_encoder")
    _dual_suffix = ""
    if _dual == "ok":
        _n = int(diagnostics.get("dual_voice_flips_to_resemb") or 0)
        _dual_suffix = f"; двойной голос: Resemblyzer, споров снято={_n}"
    elif _dual and _dual != "disabled":
        _dual_suffix = f"; двойной голос: {_dual}"
    return (
        smoothed,
        f"успех (ECAPA-TDNN{_dual_suffix}): VAD-регионов {len(vad_regions)}, окон {len(vad_window_times)}, "
        f"итоговых {len(smoothed)}, кластеров {len({x[3] for x in smoothed})}",
        diagnostics,
    )
