from __future__ import annotations

from pathlib import Path


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


def _patch_torchaudio_compat() -> None:
    """Monkeypatch torchaudio for SpeechBrain 1.x compatibility.

    torchaudio 2.2+ removed ``list_audio_backends()``; SpeechBrain 1.0.x still
    calls it unconditionally at import time.  Adding a stub prevents the
    AttributeError that otherwise blocks the entire SpeechBrain import.
    """
    try:
        import torchaudio as _ta
        if not hasattr(_ta, "list_audio_backends"):
            _ta.list_audio_backends = lambda: []
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
    """
    global _ECAPA_MODEL_CACHE
    if _ECAPA_MODEL_CACHE is not None:
        return _ECAPA_MODEL_CACHE

    import os
    _prev_offline = os.environ.get("HF_HUB_OFFLINE")
    os.environ["HF_HUB_OFFLINE"] = "1"
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
        wav, sr = librosa.load(
            str(path),
            sr=16000,
            mono=True,
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
    # ------------------------------------------------------------------
    try:
        raw_vad = librosa.effects.split(wav, top_db=35)
    except Exception as exc:
        diagnostics["reason"] = f"vad_error: {exc}"
        return None, f"ошибка VAD: {exc}", diagnostics

    vad_regions: list[tuple[float, float]] = []
    for vs_samp, ve_samp in raw_vad.tolist():
        ss = float(vs_samp) / sr
        ee = float(ve_samp) / sr
        if ee - ss < 0.35:
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
    #    Параметры подобраны для баланса скорость/качество:
    #    - window_sec=2.0  — больший контекст = стабильнее d-vector у ECAPA
    #    - hop_sec=2.0     — без перекрытия, уменьшает число окон вдвое
    #    - MAX_WINDOWS=180 — жёсткий лимит; при превышении берём равномерную
    #                        подвыборку чтобы не затянуть вычисления
    # ------------------------------------------------------------------
    window_sec = 2.0
    hop_sec = 2.0
    win_samp = int(window_sec * 16000)
    hop_samp = int(hop_sec * 16000)
    MAX_WINDOWS = 180

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
        if region.size < int(0.35 * sr):
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
    if quality < 1.05:
        labels = _forced_bipartition_labels(embs)
        forced_mode = True
        if labels is not None:
            cluster_ids = sorted(set(int(x) for x in labels))

    # ------------------------------------------------------------------
    # 5. Маппинг VAD-окон → Whisper-сегменты через временное перекрытие.
    #
    # Для каждого Whisper-сегмента находим все VAD-окна, перекрывающиеся
    # с ним, и берём majority-vote их меток. Это принципиально лучше, чем
    # старый подход (majority-vote по окнам одного Whisper-сегмента):
    # теперь несколько Whisper-сегментов могут получить метку от одних и
    # тех же коротких VAD-окон вместо усреднённых «смешанных» эмбеддингов.
    # ------------------------------------------------------------------
    # Для _nearest_time_label: отсортированный список (mid, label) по VAD-окнам.
    labeled_times: list[tuple[float, int]] = sorted(
        ((vad_window_times[i] + window_sec * 0.5, int(labels[i]))
         for i in range(len(vad_window_times))),
        key=lambda x: x[0],
    )

    def _nearest_time_label(mid: float) -> int:
        best_lab = labeled_times[0][1]
        best_dist = abs(labeled_times[0][0] - mid)
        for t, lab in labeled_times[1:]:
            d = abs(t - mid)
            if d < best_dist:
                best_dist = d
                best_lab = lab
        return best_lab

    out: list[tuple[float, float, str, str]] = []
    for seg_start, seg_end, txt in segments:
        seg_s = max(0.0, float(seg_start))
        seg_e = max(seg_s, float(seg_end))
        mid = (seg_s + seg_e) * 0.5

        # Majority-vote по VAD-окнам с перекрытием ≥ 0.1с.
        votes: dict[int, int] = {}
        for win_t, lab in zip(vad_window_times, labels):
            win_end = win_t + window_sec
            overlap = max(0.0, min(seg_e, win_end) - max(seg_s, win_t))
            if overlap >= 0.1:
                votes[int(lab)] = votes.get(int(lab), 0) + 1

        if votes:
            assigned = max(votes, key=lambda k: votes[k])
        else:
            assigned = _nearest_time_label(mid)

        out.append((seg_s, seg_e, txt, f"SPK_{assigned}"))

    # ------------------------------------------------------------------
    # 6. Сглаживание одиночных перескоков A-B-A на коротких репликах.
    # ------------------------------------------------------------------
    smoothed = out[:]
    backup_two_cluster = out[:]
    for i in range(1, len(smoothed) - 1):
        prev_c = smoothed[i - 1][3]
        cur_c = smoothed[i][3]
        next_c = smoothed[i + 1][3]
        cur_len = smoothed[i][1] - smoothed[i][0]
        if prev_c == next_c and cur_c != prev_c and cur_len < 0.8:
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
    return (
        smoothed,
        f"успех (ECAPA-TDNN): VAD-регионов {len(vad_regions)}, окон {len(vad_window_times)}, итоговых {len(smoothed)}, кластеров {len({x[3] for x in smoothed})}",
        diagnostics,
    )
