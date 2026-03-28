"""
Второй локальный голосовой энкодер (Resemblyzer d-vector) + слияние с ECAPA.

ECAPA-TDNN и Resemblyzer обучены по-разному; на телефонии один иногда ошибается там,
где другой стабилен. Этап бесплатный, веса Resemblyzer идут в пакете resemblyzer.

Отключить: CALLQA_VOICE_DUAL_ENCODER=0
"""

from __future__ import annotations

import os
import threading
from typing import Any

import numpy as np

_lock = threading.Lock()
_encoder_cache: object | None = None


def _dual_encoder_enabled() -> bool:
    v = (os.environ.get("CALLQA_VOICE_DUAL_ENCODER") or "1").strip().lower()
    return v not in ("0", "false", "no", "off")


def _pick_resemblyzer_device() -> str:
    """Resemblyzer LSTM стабильнее на CUDA/CPU; MPS часто даёт сбои."""
    try:
        import torch
    except Exception:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _get_resemblyzer_encoder():
    global _encoder_cache
    with _lock:
        if _encoder_cache is not None:
            return _encoder_cache
        import torch
        from resemblyzer import VoiceEncoder

        dev = _pick_resemblyzer_device()
        enc = VoiceEncoder(device=torch.device(dev), verbose=False)
        _encoder_cache = enc
        return enc


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-8
    return x / norms


def _centroids(embs: np.ndarray, labels: np.ndarray, n_c: int = 2) -> np.ndarray:
    out = np.zeros((n_c, embs.shape[1]), dtype=np.float32)
    for c in range(n_c):
        mask = labels == c
        if np.any(mask):
            mu = np.mean(embs[mask], axis=0)
            out[c] = mu / (np.linalg.norm(mu) + 1e-8)
    return out


def _margin_to_own_cluster(emb: np.ndarray, lab: int, centroids: np.ndarray) -> float:
    """Положительное значение — ближе к «своему» центроиду, чем к чужому (косинусная дистанция)."""
    own = int(lab) % 2
    other = 1 - own
    e = emb / (np.linalg.norm(emb) + 1e-8)
    d_own = 1.0 - float(np.dot(e, centroids[own]))
    d_oth = 1.0 - float(np.dot(e, centroids[other]))
    return float(d_oth - d_own)


def _align_resemb_to_ecapa_labels(le: np.ndarray, lr: np.ndarray) -> tuple[np.ndarray, dict[int, int]]:
    """Сопоставляем id кластеров Resemblyzer к нумерации ECAPA (0/1) по максимуму совпадений."""
    le = le.astype(np.int64)
    lr = lr.astype(np.int64)
    counts = np.zeros((2, 2), dtype=np.int64)
    for a, b in zip(le, lr):
        aa, bb = int(a) % 2, int(b) % 2
        counts[aa, bb] += 1
    from scipy.optimize import linear_sum_assignment

    cost = -counts.astype(np.float64)
    row_ind, col_ind = linear_sum_assignment(cost)
    # row_ind[k] = ecapa id, col_ind[k] = resemb id
    r_to_e: dict[int, int] = {int(col_ind[k]): int(row_ind[k]) for k in range(len(row_ind))}
    lr_aligned = np.array([r_to_e.get(int(x) % 2, int(x) % 2) for x in lr], dtype=np.int64)
    return lr_aligned, r_to_e


def embed_windows_resemblyzer(windows: list[np.ndarray]) -> np.ndarray | None:
    """Один эмбеддинг на окно (16 kHz float32), как у ECAPA."""
    if not windows:
        return None
    try:
        from resemblyzer.audio import preprocess_wav
    except Exception:
        return None
    enc = _get_resemblyzer_encoder()
    rows: list[np.ndarray] = []
    for w in windows:
        arr = np.asarray(w, dtype=np.float32).reshape(-1)
        if arr.size < 4000:
            arr = np.pad(arr, (0, 4000 - arr.size), mode="constant")
        try:
            pw = preprocess_wav(arr, source_sr=16000)
            emb = enc.embed_utterance(pw)
            rows.append(np.asarray(emb, dtype=np.float32))
        except Exception:
            return None
    return np.stack(rows, axis=0)


def fuse_ecapa_resemblyzer_from_windows(
    vad_window_audio: list[np.ndarray],
    embs_ecapa: np.ndarray,
    labels_ecapa: np.ndarray,
    diagnostics: dict[str, Any],
) -> np.ndarray:
    """Полный путь: эмбеддинги Resemblyzer с тех же окон, кластеризация, слияние с ECAPA."""
    if not _dual_encoder_enabled():
        diagnostics["dual_voice_encoder"] = "disabled"
        return labels_ecapa

    le = np.asarray(labels_ecapa, dtype=np.int64).reshape(-1)
    ee = _normalize_rows(np.asarray(embs_ecapa, dtype=np.float32))
    if len(vad_window_audio) != len(le) or len(le) < 2:
        diagnostics["dual_voice_encoder"] = "skipped_mismatch"
        return labels_ecapa

    try:
        embs_r = embed_windows_resemblyzer(vad_window_audio)
        if embs_r is None or len(embs_r) != len(le):
            diagnostics["dual_voice_encoder"] = "resemb_embed_failed"
            return labels_ecapa
        embs_r = _normalize_rows(embs_r)
    except Exception as exc:
        diagnostics["dual_voice_encoder"] = f"error:{type(exc).__name__}"
        return labels_ecapa

    try:
        from sklearn.cluster import AgglomerativeClustering

        lr = AgglomerativeClustering(n_clusters=2, metric="cosine", linkage="average").fit_predict(
            embs_r
        )
        lr = np.asarray(lr, dtype=np.int64)
    except Exception as exc:
        diagnostics["dual_voice_encoder"] = f"cluster_error:{type(exc).__name__}"
        return labels_ecapa

    lr_aligned, mapping = _align_resemb_to_ecapa_labels(le, lr)
    cent_e = _centroids(ee, le)
    cent_r = _centroids(embs_r, lr)

    fused = le.copy()
    disagreements = 0
    flips_from_e = 0
    for i in range(len(le)):
        if int(le[i]) == int(lr_aligned[i]):
            continue
        disagreements += 1
        me = _margin_to_own_cluster(ee[i], int(le[i]), cent_e)
        mr = _margin_to_own_cluster(embs_r[i], int(lr[i]), cent_r)
        if mr > me + 0.02:
            fused[i] = int(lr_aligned[i])
            flips_from_e += 1

    diagnostics["dual_voice_encoder"] = "ok"
    diagnostics["dual_voice_disagreements"] = disagreements
    diagnostics["dual_voice_flips_to_resemb"] = flips_from_e
    diagnostics["dual_voice_cluster_map"] = str(mapping)
    return fused
