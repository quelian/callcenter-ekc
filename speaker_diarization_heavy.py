from __future__ import annotations

import time
from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np
from audio_io_safe import load_audio_mono_16k_safe


@dataclass
class _Chunk:
    start: float
    end: float
    parent_idx: int
    duration: float  # для взвешенного голосования


def _forced_bipartition_labels(vectors: np.ndarray) -> np.ndarray:
    if len(vectors) < 2:
        return np.zeros((len(vectors),), dtype=np.int32)
    centered = vectors - vectors.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    direction = vt[0]
    projection = centered @ direction
    median = float(np.median(projection))
    labels = (projection >= median).astype(np.int32)
    if len(set(labels.tolist())) < 2:
        order = np.argsort(projection)
        labels = np.zeros((len(vectors),), dtype=np.int32)
        labels[order[len(order) // 2 :]] = 1
    return labels


def _normalize(v: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(v))
    if norm <= 1e-9:
        return v
    return v / norm


def _temporal_smooth_labels(
    labels: np.ndarray,
    chunks: list[_Chunk],
    *,
    min_island_sec: float = 0.5,
) -> np.ndarray:
    """Убираем короткие «острова» — сегменты, где один кластер окружён другим.

    Алгоритм: проходим по временно̀й последовательности; если сегмент <min_island_sec
    полностью окружён противоположным кластером — перекрашиваем его в кластер соседей.
    Делаем 2 прохода (возможны каскадные правки).
    """
    n = len(labels)
    if n < 3:
        return labels
    out = labels.copy()
    for _ in range(2):
        for i in range(1, n - 1):
            dur = chunks[i].duration
            if dur >= min_island_sec:
                continue
            if out[i - 1] == out[i + 1] and out[i] != out[i - 1]:
                out[i] = out[i - 1]
    return out


def run_heavy_diarization(
    audio_path: str,
    target_speakers: int = 2,
    max_seconds: int = 240,
) -> tuple[list[tuple[float, float, str]], str, dict[str, object]]:
    diagnostics: dict[str, object] = {
        "mode": "heavy",
        "split_mode": "native",
        "window_count": 0,
        "valid_regions": 0,
        "cluster_quality": 0.0,
        "elapsed_seconds": 0.0,
    }
    started = time.perf_counter()
    if target_speakers != 2:
        return [], "heavy diarization supports only 2 speakers", diagnostics
    try:
        import librosa
        from sklearn.cluster import AgglomerativeClustering
    except Exception as exc:
        return [], f"не удалось импортировать heavy зависимости: {exc}", diagnostics

    # Используем кешированную ECAPA-модель (MPS на M1, иначе CPU) вместо загрузки новой.
    try:
        from speaker_voice_roles import (
            _build_speaker_spans_from_windows,
            _embed_windows_ecapa,
            _load_ecapa_model,
            _normalize_rows,
        )
        model = _load_ecapa_model()
    except Exception as exc:
        return [], f"не удалось загрузить heavy speaker model: {exc}", diagnostics

    try:
        wav, sr = load_audio_mono_16k_safe(audio_path)
    except Exception as exc:
        return [], f"не удалось прочитать аудио: {exc}", diagnostics
    if wav.size == 0:
        return [], "пустое аудио", diagnostics

    # top_db=28: не срезаем тихие реплики (шёпот, «алло» в начале).
    # top_db=24 для heavy режима — ещё чувствительнее, чем лёгкий.
    split = librosa.effects.split(wav, top_db=24)
    regions: list[tuple[float, float]] = []
    for s, e in split.tolist():
        ss = float(s) / 16000.0
        ee = float(e) / 16000.0
        # 0.25 с — ловим очень короткие реплики (вопросы, «да»/«нет»)
        if ee - ss >= 0.25:
            regions.append((ss, ee))
    if not regions:
        return [], "не найдено голосовых регионов", diagnostics

    # Склеиваем паузы ≤ 0.15 с — один говорящий часто делает микропаузы.
    merged: list[tuple[float, float]] = []
    for s, e in regions:
        if not merged or s - merged[-1][1] > 0.15:
            merged.append((s, e))
        else:
            merged[-1] = (merged[-1][0], e)
    diagnostics["valid_regions"] = len(merged)

    # Нарезаем на окна 2.4 с с шагом 0.8 с (было 1.0 с) — плотнее покрываем смены спикера, но не перегружаем.
    WIN_SEC = 2.4
    STEP_SEC = 0.8
    MIN_WIN_SEC = 0.6  # минимальный размер окна

    chunks: list[_Chunk] = []
    for idx, (s, e) in enumerate(merged):
        cur = s
        while cur < e:
            win_end = min(e, cur + WIN_SEC)
            dur = win_end - cur
            if dur >= MIN_WIN_SEC:
                chunks.append(_Chunk(start=cur, end=win_end, parent_idx=idx, duration=dur))
            cur += STEP_SEC
    if not chunks:
        return [], "не удалось сформировать окна для embedding", diagnostics
    diagnostics["window_count"] = len(chunks)

    if time.perf_counter() - started > max_seconds:
        diagnostics["elapsed_seconds"] = round(time.perf_counter() - started, 3)
        return [], f"таймаут heavy diarization ({max_seconds}s) до embedding", diagnostics

    # Батчевый инференс через _embed_windows_ecapa (MPS/CPU, нормализация волны).
    win_samp = int(WIN_SEC * 16000)
    windows: list[np.ndarray] = []
    valid_chunks: list[_Chunk] = []
    for ch in chunks:
        seg = wav[int(ch.start * 16000) : int(ch.end * 16000)]
        if seg.size < int(0.25 * 16000):
            continue
        peak = float(np.max(np.abs(seg)) + 1e-8)
        windows.append((seg / peak).astype(np.float32))
        valid_chunks.append(ch)

    if not windows:
        diagnostics["elapsed_seconds"] = round(time.perf_counter() - started, 3)
        return [], "нет валидных окон после фильтрации", diagnostics

    try:
        embs = _embed_windows_ecapa(model, windows, win_samp, batch_size=64)
        embs = _normalize_rows(embs)
    except Exception as exc:
        diagnostics["elapsed_seconds"] = round(time.perf_counter() - started, 3)
        return [], f"ошибка embedding: {exc}", diagnostics

    chunks = valid_chunks
    if len(embs) < 4:
        diagnostics["elapsed_seconds"] = round(time.perf_counter() - started, 3)
        return [], "слишком мало валидных голосовых окон", diagnostics

    try:
        labels = AgglomerativeClustering(n_clusters=2, metric="cosine", linkage="average").fit_predict(embs)
    except Exception:
        labels = _forced_bipartition_labels(embs)
        diagnostics["split_mode"] = "forced"
    if len(set(labels.tolist())) < 2:
        labels = _forced_bipartition_labels(embs)
        diagnostics["split_mode"] = "forced"

    from speaker_voice_dual_encoder import fuse_ecapa_resemblyzer_from_windows

    labels = fuse_ecapa_resemblyzer_from_windows(windows, embs, labels, diagnostics)
    labels = np.asarray(labels, dtype=np.int32)

    # Качество кластеров: inter / intra cosine distance.
    clusters_embs: dict[int, list[np.ndarray]] = {}
    for i, lbl in enumerate(labels.tolist()):
        clusters_embs.setdefault(int(lbl), []).append(embs[i])
    c_ids = list(clusters_embs.keys())
    if len(c_ids) >= 2:
        centroids = {cid: np.mean(vs, axis=0) for cid, vs in clusters_embs.items()}
        for cid in centroids:
            n = float(np.linalg.norm(centroids[cid]))
            centroids[cid] = centroids[cid] / (n + 1e-8)
        inter = 1.0 - float(np.dot(centroids[c_ids[0]], centroids[c_ids[1]]))
        intra_vals = []
        for cid, vecs in clusters_embs.items():
            for v in vecs:
                intra_vals.append(1.0 - float(np.dot(v, centroids[cid])))
        intra = float(np.mean(intra_vals)) if intra_vals else 1.0
        diagnostics["cluster_quality"] = inter / (intra + 1e-6)

    # Временно̀е сглаживание: убираем короткие «острова» не того кластера.
    labels = _temporal_smooth_labels(np.asarray(labels, dtype=np.int32), chunks, min_island_sec=0.5)

    window_starts = [ch.start for ch in chunks]
    speaker_spans = _build_speaker_spans_from_windows(
        window_starts,
        labels,
        WIN_SEC,
        max_gap_sec=0.18,
        min_span_sec=0.50,
    )
    diagnostics["speaker_span_count"] = len(speaker_spans)
    turns: list[tuple[float, float, str]] = [
        (start, end, f"SPK_{int(label)}")
        for start, end, label in speaker_spans
    ]

    if not turns:
        # Fallback: взвешенное голосование по исходным merged-регионам.
        votes_w: dict[int, dict[int, float]] = defaultdict(lambda: defaultdict(float))
        for ch, label in zip(chunks, labels.tolist()):
            votes_w[ch.parent_idx][int(label)] += ch.duration
        for idx, (s, e) in enumerate(merged):
            if idx not in votes_w:
                continue
            label = max(votes_w[idx], key=lambda k: votes_w[idx][k])
            turns.append((s, e, f"SPK_{label}"))
    if len(turns) < 2:
        diagnostics["elapsed_seconds"] = round(time.perf_counter() - started, 3)
        return [], "heavy diarization вернула слишком мало сегментов", diagnostics

    diagnostics["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    return turns, "heavy diarization completed", diagnostics
