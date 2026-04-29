from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_answer_tokens(path: Path, *, true_count: int, false_count: int) -> None:
    rows = []
    for index in range(true_count):
        rows.append({f"t{index}": {"judge": "true"}})
    for index in range(false_count):
        rows.append({f"f{index}": {"judge": "false"}})
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )


def _run_sample_balanced_ids(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "scripts/sample_balanced_ids.py", *args],
        cwd=REPO_ROOT,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def test_sample_balanced_ids_strict_rejects_short_split(tmp_path: Path) -> None:
    input_path = tmp_path / "answer_tokens.jsonl"
    output_path = tmp_path / "split.json"
    _write_answer_tokens(input_path, true_count=5, false_count=2)

    result = _run_sample_balanced_ids(
        "--input_path",
        str(input_path),
        "--output_path",
        str(output_path),
        "--num_samples",
        "3",
        "--strict",
    )

    assert result.returncode != 0
    assert "Only 2 samples per class available" in result.stderr
    assert not output_path.exists()


def test_sample_balanced_ids_strict_honors_exclusions(tmp_path: Path) -> None:
    input_path = tmp_path / "answer_tokens.jsonl"
    first_path = tmp_path / "train.json"
    second_path = tmp_path / "dev.json"
    _write_answer_tokens(input_path, true_count=5, false_count=5)

    first = _run_sample_balanced_ids(
        "--input_path",
        str(input_path),
        "--output_path",
        str(first_path),
        "--num_samples",
        "3",
        "--seed",
        "1",
        "--strict",
    )
    second = _run_sample_balanced_ids(
        "--input_path",
        str(input_path),
        "--output_path",
        str(second_path),
        "--num_samples",
        "2",
        "--seed",
        "2",
        "--exclude_path",
        str(first_path),
        "--strict",
    )

    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr
    first_ids = json.loads(first_path.read_text(encoding="utf-8"))
    second_ids = json.loads(second_path.read_text(encoding="utf-8"))
    assert len(first_ids["t"]) == len(first_ids["f"]) == 3
    assert len(second_ids["t"]) == len(second_ids["f"]) == 2
    assert set(first_ids["t"] + first_ids["f"]).isdisjoint(
        second_ids["t"] + second_ids["f"]
    )
