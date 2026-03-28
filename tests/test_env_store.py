from __future__ import annotations

import os

from env_store import merge_env_file, normalize_google_spreadsheet_id


def test_normalize_google_spreadsheet_id_extracts_from_url() -> None:
    url = "https://docs.google.com/spreadsheets/d/abc123XYZ_-/edit#gid=0"
    assert normalize_google_spreadsheet_id(url) == "abc123XYZ_-"


def test_merge_env_file_updates_existing_and_preserves_comments(tmp_path) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text("# comment\nFOO=old\nBAR=keep\n", encoding="utf-8")

    merge_env_file({"FOO": "new value", "BAZ": "123"}, path=env_path)

    text = env_path.read_text(encoding="utf-8")
    assert "# comment" in text
    assert 'FOO="new value"' in text
    assert "BAR=keep" in text
    assert "BAZ=123" in text
    assert os.environ["FOO"] == "new value"
