"""Tests for D7 alpha locking and paired CSV2 reporting helpers."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import lock_d7_alpha
import report_d7_csv2


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _csv2_row(
    sample_id: str,
    *,
    harmful_binary: str,
    c: float = 0.0,
    s: float = 0.0,
    v: float | None = 0.0,
    harmful_payload_share: float = 0.0,
    response: str = "ok",
) -> dict:
    return {
        "id": sample_id,
        "response": response,
        "csv2": {
            "harmful_binary": harmful_binary,
            "C": c,
            "S": s,
            "V": v,
            "harmful_payload_share": harmful_payload_share,
        },
    }


class TestD7AlphaLock:
    def test_lock_alpha_prefers_lower_alpha_on_delta_tie(self, tmp_path, monkeypatch):
        csv2_dir = tmp_path / "csv2"
        _write_jsonl(
            csv2_dir / "alpha_0.0.jsonl",
            [
                _csv2_row("s1", harmful_binary="yes"),
                _csv2_row("s2", harmful_binary="yes"),
                _csv2_row("s3", harmful_binary="no"),
                _csv2_row("s4", harmful_binary="no"),
            ],
        )
        _write_jsonl(
            csv2_dir / "alpha_1.0.jsonl",
            [
                _csv2_row("s1", harmful_binary="yes"),
                _csv2_row("s2", harmful_binary="no"),
                _csv2_row("s3", harmful_binary="no"),
                _csv2_row("s4", harmful_binary="no"),
            ],
        )
        _write_jsonl(
            csv2_dir / "alpha_2.0.jsonl",
            [
                _csv2_row("s1", harmful_binary="no"),
                _csv2_row("s2", harmful_binary="yes"),
                _csv2_row("s3", harmful_binary="no"),
                _csv2_row("s4", harmful_binary="no"),
            ],
        )

        output_path = tmp_path / "lock.json"
        monkeypatch.setattr(
            lock_d7_alpha,
            "parse_args",
            lambda: argparse.Namespace(
                csv2_dir=csv2_dir,
                output_path=output_path,
                baseline_alpha=0.0,
                candidate_alphas=[1.0, 2.0],
            ),
        )

        lock_d7_alpha.main()

        payload = json.loads(output_path.read_text(encoding="utf-8"))
        assert payload["selected_alpha"] == 1.0
        assert len(payload["results"]) == 2

    def test_lock_alpha_rejects_prompt_id_mismatch(self, tmp_path, monkeypatch):
        csv2_dir = tmp_path / "csv2"
        _write_jsonl(
            csv2_dir / "alpha_0.0.jsonl",
            [
                _csv2_row("s1", harmful_binary="yes"),
                _csv2_row("s2", harmful_binary="no"),
            ],
        )
        _write_jsonl(
            csv2_dir / "alpha_1.0.jsonl",
            [_csv2_row("s1", harmful_binary="no")],
        )
        output_path = tmp_path / "lock.json"
        monkeypatch.setattr(
            lock_d7_alpha,
            "parse_args",
            lambda: argparse.Namespace(
                csv2_dir=csv2_dir,
                output_path=output_path,
                baseline_alpha=0.0,
                candidate_alphas=[1.0],
            ),
        )

        with pytest.raises(ValueError, match="sample-ID parity failed"):
            lock_d7_alpha.main()


class TestD7PairedReport:
    def test_report_paired_deltas_and_prompt_parity(self, tmp_path, monkeypatch):
        baseline_dir = tmp_path / "baseline"
        condition_dir = tmp_path / "condition"
        _write_jsonl(
            baseline_dir / "alpha_1.0.jsonl",
            [
                _csv2_row(
                    "s1",
                    harmful_binary="yes",
                    c=0.6,
                    s=0.4,
                    v=0.2,
                    harmful_payload_share=0.3,
                ),
                _csv2_row(
                    "s2",
                    harmful_binary="no",
                    c=0.2,
                    s=0.1,
                    v=0.0,
                    harmful_payload_share=0.1,
                ),
            ],
        )
        _write_jsonl(
            condition_dir / "alpha_2.0.jsonl",
            [
                _csv2_row(
                    "s1",
                    harmful_binary="no",
                    c=0.3,
                    s=0.2,
                    v=0.1,
                    harmful_payload_share=0.1,
                ),
                _csv2_row(
                    "s2",
                    harmful_binary="no",
                    c=0.1,
                    s=0.1,
                    v=0.0,
                    harmful_payload_share=0.05,
                ),
            ],
        )

        output_path = tmp_path / "report.json"
        monkeypatch.setattr(
            report_d7_csv2,
            "parse_args",
            lambda: argparse.Namespace(
                baseline_dir=baseline_dir,
                baseline_alpha=1.0,
                condition=[f"candidate:{condition_dir}:2.0"],
                output_path=output_path,
            ),
        )

        report_d7_csv2.main()
        payload = json.loads(output_path.read_text(encoding="utf-8"))
        paired = payload["paired_vs_baseline"]["candidate"]
        assert paired["n"] == 2
        assert paired["csv2_yes"]["estimate_pp"] < 0.0
        assert paired["C"]["estimate"] == pytest.approx(-0.2)
        assert paired["harmful_payload_share"]["estimate"] == pytest.approx(-0.125)

    def test_report_rejects_prompt_id_mismatch(self):
        baseline = {"s1": {"id": "s1"}, "s2": {"id": "s2"}}
        candidate = {"s1": {"id": "s1"}}
        with pytest.raises(ValueError, match="prompt-ID parity failed"):
            report_d7_csv2._require_parity(
                baseline,
                candidate,
                context="candidate",
            )

    def test_report_rejects_duplicate_sample_ids(self):
        rows = [
            _csv2_row("s1", harmful_binary="yes"),
            _csv2_row("s1", harmful_binary="no"),
        ]
        with pytest.raises(ValueError, match="duplicate sample id"):
            report_d7_csv2._rows_by_id(rows, context="baseline")
