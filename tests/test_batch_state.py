from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from batch_state import (
    BatchFileJob,
    BatchResumeState,
    clear_batch_resume_state,
    load_batch_resume_state,
    save_batch_resume_state,
)


def _patch_state_dir(monkeypatch, tmp_path) -> None:
    """Redirect all batch state paths to a temp directory."""
    import state_paths
    import results_paths

    state_dir = tmp_path / ".callqa_state"
    state_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(results_paths, "project_root", lambda: tmp_path)
    monkeypatch.setattr(state_paths, "ensure_app_state_dir", lambda: state_dir)


def test_save_and_load_roundtrip(monkeypatch, tmp_path) -> None:
    """Saving a checkpoint and loading it back restores the same data."""
    _patch_state_dir(monkeypatch, tmp_path)

    state = BatchResumeState(
        created_at=1700000000.0,
        next_idx=3,
        operator_name="Егор",
        cloud_eval_extra="",
        files=[
            BatchFileJob(name="call1.wav", path="/tmp/1.wav", size=1000),
            BatchFileJob(name="call2.wav", path="/tmp/2.wav", size=2000),
        ],
        collected=[
            {"original_filename": "call1.wav", "total_score": 8},
            {"original_filename": "call2.wav", "total_score": 7},
        ],
        errors={"call3.wav": "timeout"},
        sheets_log={"call1.wav": "ok", "call2.wav": "ok"},
    )

    save_batch_resume_state(state)
    loaded = load_batch_resume_state()

    assert loaded is not None
    assert loaded.next_idx == 3
    assert loaded.operator_name == "Егор"
    assert len(loaded.files) == 2
    assert loaded.files[0].name == "call1.wav"
    assert len(loaded.collected) == 2
    assert loaded.errors == {"call3.wav": "timeout"}


def test_load_returns_none_when_no_checkpoint(monkeypatch, tmp_path) -> None:
    _patch_state_dir(monkeypatch, tmp_path)
    # Ensure no stale state file exists
    clear_batch_resume_state()
    assert load_batch_resume_state() is None


def test_clear_removes_checkpoint(monkeypatch, tmp_path) -> None:
    _patch_state_dir(monkeypatch, tmp_path)

    save_batch_resume_state(
        BatchResumeState(
            created_at=1700000000.0,
            next_idx=0,
            operator_name="",
            cloud_eval_extra="",
            files=[],
            collected=[],
            errors={},
            sheets_log={},
        )
    )
    assert load_batch_resume_state() is not None
    clear_batch_resume_state()
    assert load_batch_resume_state() is None


def test_batch_file_job_serialization() -> None:
    job = BatchFileJob(name="test.wav", path="/a/b.wav", size=42_000)
    assert job.name == "test.wav"
    assert job.size == 42_000


def test_state_to_dict_preserves_fields() -> None:
    state = BatchResumeState(
        created_at=1.0,
        next_idx=5,
        operator_name="Артем",
        cloud_eval_extra="extra info",
        files=[BatchFileJob(name="call.wav", path="/call.wav", size=100)],
        collected=[{"x": 1}],
        errors={"e": "err"},
        sheets_log={"call.wav": "ok"},
        timings=[{"dur": 42.0}],
        batch_start_time=100.0,
    )
    d = state.to_dict()
    assert d["version"] == 2
    assert d["next_idx"] == 5
    assert d["operator_name"] == "Артем"
    assert d["cloud_eval_extra"] == "extra info"
    assert d["files"] == [{"name": "call.wav", "path": "/call.wav", "size": 100}]
    assert d["timings"] == [{"dur": 42.0}]
    assert d["batch_start_time"] == 100.0
