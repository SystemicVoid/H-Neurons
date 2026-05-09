from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any, Mapping

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from benchmark_sweep import (  # noqa: E402
    AlphaSweepContext,
    SweepRecord,
    run_alpha_sweep,
    summarize_faitheval_alpha_file,
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_alpha_sweep_preserves_resume_writing_and_faitheval_summary(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "sweep"
    output_dir.mkdir()
    (output_dir / "alpha_0.0.jsonl").write_text(
        json.dumps(
            {
                "id": "seen",
                "alpha": 0.0,
                "compliance": True,
                "chosen": "B",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    generated: list[tuple[str, float]] = []
    alpha_values: list[float] = []

    def generate_record(
        sample: Mapping[str, Any],
        context: AlphaSweepContext,
    ) -> SweepRecord:
        generated.append((sample["id"], context.alpha))
        chosen = sample["chosen"]
        return SweepRecord(
            {
                "id": sample["id"],
                "alpha": context.alpha,
                "compliance": chosen == "B",
                "chosen": chosen,
                "parse_failure": chosen is None,
            }
        )

    result = run_alpha_sweep(
        benchmark="faitheval",
        samples=[
            {"id": "seen", "chosen": "B"},
            {"id": "new", "chosen": None},
        ],
        alphas=[0.0, 1.0],
        output_dir=str(output_dir),
        set_alpha=alpha_values.append,
        generate_record=generate_record,
        summarize_alpha=summarize_faitheval_alpha_file,
    )

    assert alpha_values == [0.0, 1.0]
    assert generated == [("new", 0.0), ("seen", 1.0), ("new", 1.0)]
    assert result.output_paths == {
        "0.0": str(output_dir / "alpha_0.0.jsonl"),
        "1.0": str(output_dir / "alpha_1.0.jsonl"),
    }
    assert result.results["0.0"]["n_total"] == 2
    assert result.results["0.0"]["n_compliant"] == 1
    assert result.results["0.0"]["parse_failures"] == 1
    assert result.results["0.0"]["parse_failure"]["count"] == 1
    assert result.results["1.0"]["compliance_rate"] == 0.5

    alpha_0_records = _read_jsonl(output_dir / "alpha_0.0.jsonl")
    assert [record["id"] for record in alpha_0_records] == ["seen", "new"]


def test_alpha_sweep_respects_zero_max_samples(tmp_path: Path) -> None:
    output_dir = tmp_path / "sweep"

    def generate_record(
        sample: Mapping[str, Any],
        context: AlphaSweepContext,
    ) -> SweepRecord:
        raise AssertionError(f"unexpected generated record for {sample} at {context}")

    result = run_alpha_sweep(
        benchmark="faitheval",
        samples=[{"id": "skip"}],
        alphas=[0.0],
        output_dir=str(output_dir),
        max_samples=0,
        generate_record=generate_record,
        summarize_alpha=summarize_faitheval_alpha_file,
    )

    assert result.results["0.0"]["n_total"] == 0
    assert result.output_paths["0.0"] == str(output_dir / "alpha_0.0.jsonl")
