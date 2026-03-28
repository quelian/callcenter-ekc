from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Callable


_COMPRESSED_EXTS = {
    ".mp3",
    ".m4a",
    ".aac",
    ".ogg",
    ".opus",
    ".wma",
    ".mp4",
    ".webm",
    ".mpga",
}


def _ffmpeg_executable() -> str | None:
    """Путь к ffmpeg: PATH, FFMPEG_BINARY, типичные каталоги Homebrew (GUI-запуск часто без них в PATH)."""
    for key in ("FFMPEG_BINARY", "IMAGEIO_FFMPEG_EXE"):
        raw = (os.environ.get(key) or "").strip()
        if raw and Path(raw).is_file():
            return raw
    w = shutil.which("ffmpeg")
    if w:
        return w
    for p in ("/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg"):
        if Path(p).is_file():
            return p
    return None


def _sniff_needs_ffmpeg_decode(path: str) -> bool:
    """True, если по заголовку файл похож на сжатый поток (MP4/OGG/MP3…), а не WAV/FLAC.

    Нужен, когда расширение врёт (например .wav при реальном MP3) — иначе soundfile падает
    и срабатывает librosa.load → libmpg123 в процессе Python и спам в консоль.
    """
    try:
        with open(path, "rb") as f:
            b = f.read(8192)
    except OSError:
        return False
    if len(b) < 12:
        return False
    if b.startswith(b"RIFF") and b[8:12] == b"WAVE":
        return False
    if b.startswith(b"fLaC"):
        return False
    if b.startswith(b"ID3"):
        return True
    if b[0] == 0xFF and b[1] in (0xFB, 0xF3, 0xFA, 0xF2):
        return True
    if b.startswith(b"OggS"):
        return True
    if b[4:8] == b"ftyp":
        return True
    if b[0] == 0xFF and b[1] in (0xF1, 0xF9):
        return True
    if b.startswith(b"\x1a\x45\xdf\xa3"):
        return True
    return False


def _resample_if_needed(y, src_sr: int, target_sr: int):
    if src_sr == target_sr:
        return y
    import math

    from scipy.signal import resample_poly

    g = math.gcd(src_sr, target_sr)
    up = target_sr // g
    down = src_sr // g
    return resample_poly(y, up, down).astype("float32")


def _read_with_soundfile(path: str, *, offset: float = 0.0, duration: float | None = None):
    import soundfile as sf

    with sf.SoundFile(path) as f:
        sr = int(f.samplerate)
        start = max(0, int(float(offset) * sr))
        f.seek(start)
        frames = -1 if duration is None else max(0, int(float(duration) * sr))
        y = f.read(frames=frames, dtype="float32", always_2d=False)
    return y, sr


def _load_via_ffmpeg(
    path: str,
    *,
    ffmpeg_bin: str,
    target_sr: int,
    offset: float = 0.0,
    duration: float | None = None,
    robust: bool = False,
) -> tuple[object, int]:
    """Декодирование через внешний ffmpeg (отдельный процесс — не валит Python при битых MP3).

    При успехе stderr/stdout глушим: часть сборок ffmpeg пишет в libmpg123 напрямую в fd 2,
    и это пробивалось в консоль Streamlit даже при capture_output в других конфигурациях.
    """
    import soundfile as sf

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        out_wav = tmp.name
    try:
        cmd: list[str] = [
            ffmpeg_bin,
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "fatal",
        ]
        if robust:
            cmd += ["-fflags", "+discardcorrupt", "-err_detect", "ignore_err"]
        if offset and offset > 0:
            cmd += ["-ss", f"{float(offset):.3f}"]
        cmd += ["-i", path]
        if duration is not None:
            cmd += ["-t", f"{max(0.0, float(duration)):.3f}"]
        cmd += ["-ac", "1", "-ar", str(int(target_sr)), "-f", "wav", out_wav, "-y"]
        r = subprocess.run(
            cmd,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if r.returncode != 0:
            r2 = subprocess.run(cmd, check=False, capture_output=True, text=True)
            err = (r2.stderr or r2.stdout or "").strip()[:500]
            raise subprocess.CalledProcessError(
                r2.returncode, cmd, output=r2.stdout, stderr=err
            )
        y, sr = sf.read(out_wav, dtype="float32", always_2d=False)
        return y, int(sr)
    finally:
        try:
            Path(out_wav).unlink(missing_ok=True)
        except Exception:
            pass


def _decode_with_ffmpeg_required(
    p: str,
    *,
    offset: float,
    duration: float | None,
    log: Callable[[str], None] | None,
) -> tuple[object, int]:
    ff = _ffmpeg_executable()
    if not ff:
        msg = (
            "Нужна утилита ffmpeg для сжатого аудио: установите (macOS: brew install ffmpeg) "
            "или задайте FFMPEG_BINARY=/полный/путь/к/ffmpeg."
        )
        if log is not None:
            log(msg)
        raise RuntimeError(msg)
    try:
        return _load_via_ffmpeg(
            p,
            ffmpeg_bin=ff,
            target_sr=16000,
            offset=offset,
            duration=duration,
            robust=False,
        )
    except subprocess.CalledProcessError as exc:
        if log is not None:
            log(
                f"FFmpeg (1-я попытка) для {Path(p).name}: {exc}; повтор с discardcorrupt…"
            )
        return _load_via_ffmpeg(
            p,
            ffmpeg_bin=ff,
            target_sr=16000,
            offset=offset,
            duration=duration,
            robust=True,
        )


def is_probably_compressed_audio(path: str) -> bool:
    """Расширение или сигнатура файла указывают на контейнер, который нельзя отдавать в Whisper как сырой путь."""
    p = str(path)
    if Path(p).suffix.lower() in _COMPRESSED_EXTS:
        return True
    return _sniff_needs_ffmpeg_decode(p)


def load_audio_mono_16k_safe(
    path: str,
    *,
    offset: float = 0.0,
    duration: float | None = None,
    log: Callable[[str], None] | None = None,
):
    """Безопасная загрузка для ASR/диаризации.

    Сжатые форматы — **только через ffmpeg** (никакого librosa/soundfile по исходному MP3:
    иначе в процесс тянется libmpg123 и возможен segfault / спам в консоль).

    WAV/FLAC — soundfile, при необходимости ресемпл до 16 kHz mono.
    """
    import numpy as np

    p = str(path)
    ext = Path(p).suffix.lower()
    sniff_ffmpeg = _sniff_needs_ffmpeg_decode(p)

    if ext in _COMPRESSED_EXTS or sniff_ffmpeg:
        y, sr = _decode_with_ffmpeg_required(
            p, offset=offset, duration=duration, log=log
        )
        return np.asarray(y, dtype=np.float32).reshape(-1), int(sr)

    try:
        y, sr = _read_with_soundfile(p, offset=offset, duration=duration)
        y = np.asarray(y, dtype=np.float32)
        if y.ndim > 1:
            y = y.mean(axis=1)
        y = _resample_if_needed(y.reshape(-1), int(sr), 16000)
        return y.astype(np.float32), 16000
    except Exception:
        pass

    import librosa

    y, sr = librosa.load(p, sr=16000, mono=True, offset=float(offset), duration=duration)
    return np.asarray(y, dtype=np.float32).reshape(-1), int(sr)
