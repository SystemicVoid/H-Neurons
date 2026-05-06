from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from analyze_available_jailbreak_evaluators import compute_mistral_anchor_summary


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def _write_complete_alpha_set(tmp_path: Path) -> dict[str, Path]:
    dirs = {
        "binary_full": tmp_path / "full",
        "binary_256": tmp_path / "trunc",
        "csv3_full": tmp_path / "csv3",
        "strongreject_full": tmp_path / "sr",
    }
    _write_jsonl(
        dirs["binary_full"] / "alpha_0.0.jsonl",
        [
            {"id": "rec-a", "alpha": 0.0, "judge": "HARMFUL"},
            {"id": "rec-b", "alpha": 0.0, "judge": "SAFE"},
            {"id": "rec-c", "alpha": 0.0, "judge": "HARMFUL"},
        ],
    )
    _write_jsonl(
        dirs["binary_256"] / "alpha_0.0.jsonl",
        [
            {"id": "rec-a", "alpha": 0.0, "judge": "HARMFUL"},
            {"id": "rec-b", "alpha": 0.0, "judge": "HARMFUL"},
            {"id": "rec-c", "alpha": 0.0, "judge": "SAFE"},
        ],
    )
    _write_jsonl(
        dirs["csv3_full"] / "alpha_0.0.jsonl",
        [
            {"id": "rec-a", "alpha": 0.0, "csv2": {"harmful_binary": "yes"}},
            {"id": "rec-b", "alpha": 0.0, "csv2": {"harmful_binary": "no"}},
            {"id": "rec-c", "alpha": 0.0, "csv2": {"harmful_binary": "no"}},
        ],
    )
    _write_jsonl(
        dirs["strongreject_full"] / "alpha_0.0.jsonl",
        [
            {
                "id": "rec-a",
                "alpha": 0.0,
                "strongreject": {"harmful_binary": "yes"},
            },
            {
                "id": "rec-b",
                "alpha": 0.0,
                "strongreject": {"harmful_binary": "no"},
            },
            {
                "id": "rec-c",
                "alpha": 0.0,
                "strongreject": {"harmful_binary": "yes"},
            },
        ],
    )
    return dirs


def test_mistral_anchor_summary_reports_pairwise_discordance(tmp_path: Path) -> None:
    dirs = _write_complete_alpha_set(tmp_path)

    summary = compute_mistral_anchor_summary(
        binary_full_dir=dirs["binary_full"],
        binary_256_dir=dirs["binary_256"],
        csv3_full_dir=dirs["csv3_full"],
        strongreject_full_dir=dirs["strongreject_full"],
        alphas=[0.0],
    )

    assert summary["n_records"] == 3
    binary_pair = summary["pairs"]["binary_256_vs_binary_full"]["combined"]
    assert binary_pair["left_yes_right_yes"] == 1
    assert binary_pair["left_yes_right_no"] == 1
    assert binary_pair["left_no_right_yes"] == 1
    assert binary_pair["left_no_right_no"] == 0
    assert binary_pair["disagreement"]["rate"] == pytest.approx(2 / 3)
    assert binary_pair["cohen_kappa"] == pytest.approx(-0.5)
    assert binary_pair["mcnemar_exact_two_sided_p"] == pytest.approx(1.0)

    csv3_pair = summary["pairs"]["binary_full_vs_csv3_full"]["combined"]
    assert csv3_pair["left_yes_right_no"] == 1
    assert csv3_pair["agreement"]["rate"] == pytest.approx(2 / 3)

    sr_pair = summary["pairs"]["csv3_full_vs_strongreject_full"]["combined"]
    assert sr_pair["left_no_right_yes"] == 1
    assert sr_pair["agreement"]["rate"] == pytest.approx(2 / 3)
    assert (
        summary["diagnostics"]["csv3_vs_strongreject_full"]["combined"]["n_paired"] == 3
    )


def test_mistral_anchor_summary_fails_on_id_mismatch(tmp_path: Path) -> None:
    dirs = _write_complete_alpha_set(tmp_path)
    _write_jsonl(
        dirs["binary_256"] / "alpha_0.0.jsonl",
        [
            {"id": "rec-a", "alpha": 0.0, "judge": "HARMFUL"},
            {"id": "rec-b", "alpha": 0.0, "judge": "HARMFUL"},
            {"id": "rec-other", "alpha": 0.0, "judge": "SAFE"},
        ],
    )

    with pytest.raises(ValueError, match="keys do not match"):
        compute_mistral_anchor_summary(
            binary_full_dir=dirs["binary_full"],
            binary_256_dir=dirs["binary_256"],
            csv3_full_dir=dirs["csv3_full"],
            strongreject_full_dir=dirs["strongreject_full"],
            alphas=[0.0],
        )
