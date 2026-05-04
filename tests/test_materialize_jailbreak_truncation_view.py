from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import materialize_jailbreak_truncation_view as truncation_view
from materialize_jailbreak_truncation_view import (
    main,
    materialize_alpha_file,
    truncate_record,
)


class FakeTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[str]:
        assert add_special_tokens is False
        return text.split()

    def decode(self, token_ids: list[str], skip_special_tokens: bool = False) -> str:
        assert skip_special_tokens is False
        return " ".join(token_ids)


def test_truncate_record_replaces_response_and_removes_evaluator_fields() -> None:
    record = {
        "id": "jbb_harmful_1_t0",
        "alpha": 3.0,
        "goal": "bad goal",
        "response": "one two three four",
        "judge": "HARMFUL",
        "compliance": True,
        "csv2": {"harmful_binary": "yes"},
    }

    out = truncate_record(
        record,
        tokenizer=FakeTokenizer(),
        token_limit=2,
        model_key="fake_model",
        model_path="fake/path",
    )

    assert out["response"] == "one two"
    assert "judge" not in out
    assert "compliance" not in out
    assert "csv2" not in out
    assert out["response_truncation"]["source_response_model_tokens"] == 4
    assert out["response_truncation"]["truncated_response_model_tokens"] == 2
    assert out["response_truncation"]["truncated"] is True


def test_materialize_alpha_file_writes_jsonl(tmp_path) -> None:
    input_path = tmp_path / "in" / "alpha_0.0.jsonl"
    output_path = tmp_path / "out" / "alpha_0.0.jsonl"
    input_path.parent.mkdir()
    input_path.write_text(
        json.dumps(
            {
                "id": "jbb_harmful_2_t1",
                "alpha": 0.0,
                "response": "a b c",
                "judge": "SAFE",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    n_rows = materialize_alpha_file(
        input_path,
        output_path,
        tokenizer=FakeTokenizer(),
        token_limit=2,
        model_key=None,
        model_path="fake/path",
    )

    assert n_rows == 1
    rows = [json.loads(line) for line in output_path.read_text().splitlines()]
    assert rows[0]["id"] == "jbb_harmful_2_t1"
    assert rows[0]["response"] == "a b"
    assert rows[0]["response_truncation"]["token_limit"] == 2


def test_materialize_alpha_file_preserves_compatible_existing_judgments(
    tmp_path,
) -> None:
    input_path = tmp_path / "in" / "alpha_0.0.jsonl"
    output_path = tmp_path / "out" / "alpha_0.0.jsonl"
    input_path.parent.mkdir()
    output_path.parent.mkdir()
    record = {
        "id": "jbb_harmful_4_t0",
        "alpha": 0.0,
        "response": "a b c",
        "judge": "FULL_RESPONSE_SAFE",
        "compliance": False,
        "csv2": {"harmful_binary": "no"},
    }
    input_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    existing = truncate_record(
        record,
        tokenizer=FakeTokenizer(),
        token_limit=2,
        model_key=None,
        model_path="fake/path",
    )
    existing["judge"] = "TRUNCATED_HARMFUL"
    existing["compliance"] = True
    existing["csv2"] = {"harmful_binary": "yes"}
    output_path.write_text(json.dumps(existing) + "\n", encoding="utf-8")

    n_rows = materialize_alpha_file(
        input_path,
        output_path,
        tokenizer=FakeTokenizer(),
        token_limit=2,
        model_key=None,
        model_path="fake/path",
    )

    assert n_rows == 1
    rows = [json.loads(line) for line in output_path.read_text().splitlines()]
    assert rows[0]["response"] == "a b"
    assert rows[0]["judge"] == "TRUNCATED_HARMFUL"
    assert rows[0]["compliance"] is True
    assert rows[0]["csv2"] == {"harmful_binary": "yes"}


def test_materialize_alpha_file_drops_existing_judgments_when_source_changes(
    tmp_path,
) -> None:
    input_path = tmp_path / "in" / "alpha_0.0.jsonl"
    output_path = tmp_path / "out" / "alpha_0.0.jsonl"
    input_path.parent.mkdir()
    output_path.parent.mkdir()
    original = {
        "id": "jbb_harmful_5_t0",
        "alpha": 0.0,
        "response": "a b c",
    }
    existing = truncate_record(
        original,
        tokenizer=FakeTokenizer(),
        token_limit=2,
        model_key=None,
        model_path="fake/path",
    )
    existing["judge"] = "TRUNCATED_HARMFUL"
    existing["compliance"] = True
    output_path.write_text(json.dumps(existing) + "\n", encoding="utf-8")

    changed = {
        "id": "jbb_harmful_5_t0",
        "alpha": 0.0,
        "response": "a b changed",
    }
    input_path.write_text(json.dumps(changed) + "\n", encoding="utf-8")

    n_rows = materialize_alpha_file(
        input_path,
        output_path,
        tokenizer=FakeTokenizer(),
        token_limit=2,
        model_key=None,
        model_path="fake/path",
    )

    assert n_rows == 1
    rows = [json.loads(line) for line in output_path.read_text().splitlines()]
    assert rows[0]["response"] == "a b"
    assert "judge" not in rows[0]
    assert "compliance" not in rows[0]
    assert (
        rows[0]["response_truncation"]["source_response_sha256"]
        != existing["response_truncation"]["source_response_sha256"]
    )


def test_main_records_provenance_and_active_run_lock(monkeypatch, tmp_path) -> None:
    subprocess.run(
        ["git", "init"],
        cwd=tmp_path,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    monkeypatch.chdir(tmp_path)

    input_dir = tmp_path / "input"
    output_dir = tmp_path / "data" / "truncated"
    input_dir.mkdir()
    (input_dir / "alpha_0.0.jsonl").write_text(
        json.dumps(
            {
                "id": "jbb_harmful_3_t0",
                "alpha": 0.0,
                "response": "alpha beta gamma",
                "judge": "HARMFUL",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        truncation_view.AutoTokenizer,
        "from_pretrained",
        lambda _model_path, **_kwargs: FakeTokenizer(),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "scripts/materialize_jailbreak_truncation_view.py",
            "--input_dir",
            str(input_dir),
            "--output_dir",
            str(output_dir),
            "--alphas",
            "0.0",
            "--model_path",
            "fake/path",
            "--response-token-limit",
            "2",
        ],
    )
    monkeypatch.setattr(
        sys, "orig_argv", ["uv", "run", "python", *sys.argv], raising=False
    )

    active_lock_seen = False
    original_materialize_alpha_file = truncation_view.materialize_alpha_file

    def materialize_with_lock_assertion(*args, **kwargs) -> int:
        nonlocal active_lock_seen
        lock_paths = list((tmp_path / ".git" / "h-neurons-active-runs").glob("*.json"))
        assert len(lock_paths) == 1
        lock_payload = json.loads(lock_paths[0].read_text(encoding="utf-8"))
        protected_paths = {
            (item["kind"], item["path"]) for item in lock_payload["protected_paths"]
        }
        assert ("directory", str(output_dir.resolve())) in protected_paths
        active_lock_seen = True
        return original_materialize_alpha_file(*args, **kwargs)

    monkeypatch.setattr(
        truncation_view,
        "materialize_alpha_file",
        materialize_with_lock_assertion,
    )

    main()

    assert active_lock_seen
    assert not list((tmp_path / ".git" / "h-neurons-active-runs").glob("*.json"))
    provenance_paths = list(
        output_dir.glob("materialize_jailbreak_truncation_view.provenance.*.json")
    )
    assert len(provenance_paths) == 1
    provenance = json.loads(provenance_paths[0].read_text(encoding="utf-8"))
    assert provenance["status"] == "completed"
    assert provenance["n_records"] == 1
    assert provenance["manifest_path"] == str(
        output_dir / "truncation_view_manifest.json"
    )
    assert str(output_dir.resolve()) in provenance["output_targets"]
