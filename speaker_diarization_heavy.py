from __future__ import annotations

import time
from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np


@dataclass
class _Chunk:
    start: float
    end: float
    parent_idx: int


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
        "elapsed_seconds": 0.0,
    }
    started = time.perf_counter()
    if target_speakers != 2:
        return [], "heavy diarization supports only 2 speakers", diagnostics
    try:
        import librosa
        import torch
        from sklearn.cluster import AgglomerativeClustering
        # SpeechBrain 1.x calls torchaudio.list_audio_backends() which was removed in
        # torchaudio 2.2+.  Patch it with a stub before importing speechbrain.
        try:
            import torchaudio as _ta
            if not hasattr(_ta, "list_audio_backends"):
                _ta.list_audio_backends = lambda: []
        except Exception:
            pass
        from speechbrain.inference.speaker import EncoderClassifier
    except Exception as exc:
        return [], f"не удалось импортировать heavy зависимости: {exc}", diagnostics

    try:
        wav, sr = librosa.load(audio_path, sr=16000, mono=True)
    except Exception as exc:
        return [], f"не удалось прочитать аудио: {exc}", diagnostics
    if wav.size == 0:
        return [], "пустое аудио", diagnostics

    split = librosa.effects.split(wav, top_db=35)
    regions: list[tuple[float, float]] = []
    for s, e in split.tolist():
        ss = float(s) / 16000.0
        ee = float(e) / 16000.0
        if ee - ss >= 0.40:
            regions.append((ss, ee))
    if not regions:
        return [], "не найдено голосовых регионов", diagnostics

    merged: list[tuple[float, float]] = []
    for s, e in regions:
        if not merged or s - merged[-1][1] > 0.22:
            merged.append((s, e))
        else:
            merged[-1] = (merged[-1][0], e)
    diagnostics["valid_regions"] = len(merged)

    chunks: list[_Chunk] = []
    for idx, (s, e) in enumerate(merged):
        cur = s
        while cur < e:
            win_end = min(e, cur + 2.4)
            if win_end - cur >= 0.8:
                chunks.append(_Chunk(start=cur, end=win_end, parent_idx=idx))
            cur += 1.0
    if not chunks:
        return [], "не удалось сформировать окна для embedding", diagnostics
    diagnostics["window_count"] = len(chunks)

    try:
        model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="pretrained_models/spkrec-ecapa-voxceleb",
            run_opts={"device": "cpu"},
        )
    except Exception as exc:
        return [], f"не удалось загрузить heavy speaker model: {exc}", diagnostics

    emb_rows: list[np.ndarray] = []
    for ch in chunks:
        if time.perf_counter() - started > max_seconds:
            diagnostics["elapsed_seconds"] = round(time.perf_counter() - started, 3)
            return [], f"таймаут heavy diarization ({max_seconds}s)", diagnostics
        seg = wav[int(ch.start * 16000) : int(ch.end * 16000)]
        if seg.size < 1600:
            continue
        tens = torch.tensor(seg, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            emb = model.encode_batch(tens).squeeze().detach().cpu().numpy()
        emb_rows.append(_normalize(emb))

    if len(emb_rows) < 4:
        diagnostics["elapsed_seconds"] = round(time.perf_counter() - started, 3)
        return [], "слишком мало валидных голосовых окон", diagnostics

    emb = np.vstack(emb_rows)
    try:
        labels = AgglomerativeClustering(n_clusters=2, linkage="average").fit_predict(emb)
    except Exception:
        labels = _forced_bipartition_labels(emb)
        diagnostics["split_mode"] = "forced"
    if len(set(labels.tolist())) < 2:
        labels = _forced_bipartition_labels(emb)
        diagnostics["split_mode"] = "forced"

    votes: dict[int, list[int]] = defaultdict(list)
    for ch, label in zip(chunks, labels.tolist()):
        votes[ch.parent_idx].append(int(label))

    turns: list[tuple[float, float, str]] = []
    for idx, (s, e) in enumerate(merged):
        if idx not in votes:
            continue
        label = Counter(votes[idx]).most_common(1)[0][0]
        turns.append((s, e, f"SPK_{label}"))
    if len(turns) < 2:
        diagnostics["elapsed_seconds"] = round(time.perf_counter() - started, 3)
        return [], "heavy diarization вернула слишком мало сегментов", diagnostics

    diagnostics["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    return turns, "heavy diarization completed", diagnostics
