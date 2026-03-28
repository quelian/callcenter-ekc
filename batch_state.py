from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from state_paths import ensure_app_state_dir

STATE_VERSION = 1


@dataclass(frozen=True)
class BatchFileJob:
    name: str
    path: str
    size: int = 0


@dataclass
class BatchResumeState:
    version: int = STATE_VERSION
    created_at: float = 0.0
    next_idx: int = 0
    operator_name: str = ""
    cloud_eval_extra: str = ""
    files: list[BatchFileJob] = field(default_factory=list)
    collected: list[dict[str, Any]] = field(default_factory=list)
    errors: dict[str, str] = field(default_factory=dict)
    sheets_log: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "created_at": self.created_at,
            "next_idx": self.next_idx,
            "operator_name": self.operator_name,
            "cloud_eval_extra": self.cloud_eval_extra,
            "files": [
                {"name": job.name, "path": job.path, "size": int(job.size)}
                for job in self.files
            ],
            "collected": self.collected,
            "errors": self.errors,
            "sheets_log": self.sheets_log,
        }


def batch_resume_state_path() -> Path:
    return ensure_app_state_dir() / "batch_resume_state.json"


def batch_resume_files_dir() -> Path:
    path = ensure_app_state_dir() / "batch_resume_files"
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_batch_resume_state() -> BatchResumeState | None:
    path = batch_resume_state_path()
    if not path.exists():
        return None
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return None
    if not isinstance(obj, dict):
        return None
    if int(obj.get("version", STATE_VERSION) or STATE_VERSION) != STATE_VERSION:
        return None
    raw_files = obj.get("files", [])
    files: list[BatchFileJob] = []
    if isinstance(raw_files, list):
        for item in raw_files:
            if not isinstance(item, dict):
                continue
            files.append(
                BatchFileJob(
                    name=str(item.get("name") or ""),
                    path=str(item.get("path") or ""),
                    size=int(item.get("size") or 0),
                )
            )
    collected = obj.get("collected", [])
    errors = obj.get("errors", {})
    sheets_log = obj.get("sheets_log", {})
    return BatchResumeState(
        version=STATE_VERSION,
        created_at=float(obj.get("created_at") or 0.0),
        next_idx=int(obj.get("next_idx") or 0),
        operator_name=str(obj.get("operator_name") or ""),
        cloud_eval_extra=str(obj.get("cloud_eval_extra") or ""),
        files=files,
        collected=list(collected) if isinstance(collected, list) else [],
        errors=dict(errors) if isinstance(errors, dict) else {},
        sheets_log=dict(sheets_log) if isinstance(sheets_log, dict) else {},
    )


def save_batch_resume_state(state: BatchResumeState) -> None:
    path = batch_resume_state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(
        json.dumps(state.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    tmp.replace(path)


def clear_batch_resume_state() -> None:
    state = load_batch_resume_state()
    if state is not None:
        for job in state.files:
            try:
                file_path = Path(job.path)
                if file_path.exists():
                    file_path.unlink()
            except OSError:
                pass
    try:
        batch_resume_state_path().unlink(missing_ok=True)
    except OSError:
        pass
    resume_dir = batch_resume_files_dir()
    for path in resume_dir.glob("*"):
        try:
            path.unlink()
        except OSError:
            pass
