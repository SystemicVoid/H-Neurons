from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from artifact_io import (  # noqa: E402
    JsonlFormatError,
    alpha_jsonl_filename,
    alpha_jsonl_path,
    file_sha256,
    jsonl_rows_sha256,
    read_jsonl_recovery,
    read_jsonl_strict,
    row_id,
    row_sha256,
    write_canonical_json,
    write_jsonl_rows,
)


def test_strict_jsonl_reports_malformed_rows_and_missing_files(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    path.write_text('{"id": "ok"}\nnot-json\n', encoding="utf-8")

    with pytest.raises(JsonlFormatError, match=r"rows\.jsonl:2: invalid JSONL row"):
        read_jsonl_strict(path)

    assert read_jsonl_strict(tmp_path / "missing.jsonl", missing_ok=True) == []
    with pytest.raises(FileNotFoundError):
        read_jsonl_strict(tmp_path / "missing.jsonl")


def test_recovery_jsonl_recovers_multiline_rows(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    path.write_text(
        '{\n  "id": "a",\n  "text": "line\\nbreak"\n}\n{"id": "b"}\n',
        encoding="utf-8",
    )

    assert read_jsonl_recovery(path) == [
        {"id": "a", "text": "line\nbreak"},
        {"id": "b"},
    ]


def test_recovery_jsonl_reports_unrecoverable_tail(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text('{"id": "ok"}\n{"id": ', encoding="utf-8")

    with pytest.raises(JsonlFormatError, match=r"bad\.jsonl:2: unrecoverable"):
        read_jsonl_recovery(path)


def test_jsonl_writer_uses_trailing_newlines_and_stable_order(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"

    write_jsonl_rows(path, [{"b": 1, "a": 2}])

    assert path.read_text(encoding="utf-8") == '{"a": 2, "b": 1}\n'
    assert read_jsonl_strict(path) == [{"a": 2, "b": 1}]

    empty_path = tmp_path / "empty.jsonl"
    write_jsonl_rows(empty_path, [])
    assert empty_path.read_text(encoding="utf-8") == ""


def test_canonical_json_and_hashes_are_stable(tmp_path: Path) -> None:
    path = tmp_path / "payload.json"
    payload = {"b": 1, "a": {"z": 2}}

    write_canonical_json(path, payload)

    assert (
        path.read_text(encoding="utf-8")
        == json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    assert file_sha256(path) == hashlib.sha256(path.read_bytes()).hexdigest()
    assert row_sha256({"b": 1, "a": 2}) == row_sha256({"a": 2, "b": 1})
    assert (
        jsonl_rows_sha256([{"b": 1, "a": 2}])
        == hashlib.sha256(b'{"a":2,"b":1}\n').hexdigest()
    )


def test_alpha_paths_and_row_ids_cover_edge_cases(tmp_path: Path) -> None:
    assert alpha_jsonl_filename(1) == "alpha_1.0.jsonl"
    assert alpha_jsonl_filename(1.5) == "alpha_1.5.jsonl"
    assert alpha_jsonl_filename(0.05) == "alpha_0.05.jsonl"
    assert alpha_jsonl_path(tmp_path, 0.05) == tmp_path / "alpha_0.05.jsonl"
    assert row_id({"id": 123}) == "123"
    with pytest.raises(KeyError, match="identity field"):
        row_id({"sample_id": "x"})
