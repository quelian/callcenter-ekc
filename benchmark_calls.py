from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

from transcription import Transcriber


def _audio_files(root: Path) -> list[Path]:
    exts = {".wav", ".mp3", ".m4a", ".ogg", ".flac"}
    return sorted([p for p in root.rglob("*") if p.suffix.lower() in exts])


def _diag_map(diag: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for part in (diag or "").split(";"):
        token = part.strip()
        if "=" not in token:
            continue
        k, v = token.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch benchmark for RU heavy pipeline")
    parser.add_argument("calls_dir", help="Папка с 5-10 звонками")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--compute-type", default="int8")
    parser.add_argument("--output", default="benchmark_results.csv")
    parser.add_argument("--post-edit", action="store_true", help="Включить локальный пост-редактор")
    parser.add_argument("--post-edit-timeout-seconds", type=int, default=60)
    args = parser.parse_args()

    calls = _audio_files(Path(args.calls_dir))[: args.limit]
    if not calls:
        raise SystemExit("В папке не найдены аудио-файлы.")

    transcriber = Transcriber(
        model_name="large-v3",
        compute_type=args.compute_type,
        asr_profile="ideal_ru",
        heavy_diarization=True,
        heavy_diarization_timeout_seconds=240,
        enable_post_edit=args.post_edit,
        post_edit_timeout_seconds=args.post_edit_timeout_seconds,
        total_time_budget_seconds=300,
    )
    rows: list[dict[str, str]] = []
    for idx, audio in enumerate(calls, start=1):
        started = time.perf_counter()
        result = transcriber.transcribe(audio)
        elapsed = time.perf_counter() - started
        diag = _diag_map(result.role_diagnostics or "")
        rows.append(
            {
                "file": str(audio),
                "elapsed_total_seconds": f"{elapsed:.2f}",
                "role_confidence": result.role_confidence,
                "diarization_path": diag.get("diarization_path", ""),
                "asr_seconds": diag.get("asr_seconds", ""),
                "diarization_seconds": diag.get("diarization_seconds", ""),
                "merge_seconds": diag.get("merge_seconds", ""),
                "post_edit": diag.get("post_edit", ""),
                "post_edit_seconds": diag.get("post_edit_seconds", ""),
                "role_refine_applied": diag.get("role_refine_applied", ""),
            }
        )
        print(f"[{idx}/{len(calls)}] {audio.name}: {elapsed:.1f}s")

    out = Path(args.output)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved benchmark: {out}")


if __name__ == "__main__":
    main()
