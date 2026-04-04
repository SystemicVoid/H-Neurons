"""Tests for E2 selector overrides and lock robustness utilities."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import extract_truthfulness_iti as extract_iti
import lock_config
import report_e2_canonical
import report_simpleqa_shortlist_pilot
import run_calibration_sweep


def _toy_sweep_results() -> list[dict]:
    return [
        {"k": 8, "alpha": 8.0, "mc1": 0.3000, "mc2": 0.4200, "n_mc1": 81, "n_mc2": 81},
        {"k": 12, "alpha": 8.0, "mc1": 0.3050, "mc2": 0.4300, "n_mc1": 81, "n_mc2": 81},
        {"k": 12, "alpha": 6.0, "mc1": 0.3050, "mc2": 0.4300, "n_mc1": 81, "n_mc2": 81},
        {"k": 16, "alpha": 6.0, "mc1": 0.3050, "mc2": 0.4350, "n_mc1": 81, "n_mc2": 81},
    ]


class TestSelectorPolicy:
    def test_triviaqa_defaults_to_auroc_all_positions(self):
        rank_primary, summaries, position_policy = extract_iti.resolve_selector_policy(
            family="iti_triviaqa_transfer",
            ranking_metric_override=None,
            position_policy_override=None,
        )
        assert rank_primary == "auroc"
        assert summaries == extract_iti.POSITION_SUMMARIES
        assert position_policy == "all_answer_positions"

    def test_source_isolated_override_uses_paper_faithful_selectors(self):
        rank_primary, summaries, position_policy = extract_iti.resolve_selector_policy(
            family="iti_triviaqa_transfer",
            ranking_metric_override="val_accuracy",
            position_policy_override="last_answer_token",
        )
        assert rank_primary == "val_accuracy"
        assert summaries == ("last_answer_token",)
        assert position_policy == "last_answer_token"


class TestResolutionAwareSelection:
    def test_tolerance_floor_uses_resolution_or_1_5pp(self):
        resolved = run_calibration_sweep._resolve_tolerance_pp(
            requested_tolerance_pp=0.5,
            n_calibration_samples=81,
        )
        assert resolved["mc1_resolution_pp"] == 100 / 81
        assert resolved["tolerance_floor_pp"] == 1.5
        assert resolved["tolerance_pp_applied"] == 1.5
        assert resolved["tolerance_raised_to_resolution_floor"] is True

    def test_tie_break_rule_is_mc2_then_alpha_then_k(self):
        selected = run_calibration_sweep.select_locked_config(
            _toy_sweep_results(), tolerance_pp=1.5
        )
        assert selected["k"] == 16
        assert selected["alpha"] == 6.0
        assert selected["mc2"] == 0.4350

    def test_selection_diagnostics_include_shortlist_and_trace(self):
        diag = run_calibration_sweep.compute_selection_diagnostics(
            _toy_sweep_results(),
            tolerance_pp_requested=0.5,
            tolerance_pp_applied=1.5,
            n_calibration_samples=81,
        )
        assert diag["n_calibration_samples"] == 81
        assert diag["mc1_resolution_pp"] == 100 / 81
        assert diag["tolerance_pp_applied"] == 1.5
        assert len(diag["shortlist"]) >= 1
        assert len(diag["tie_break_path"]) == 3
        assert diag["tie_break_path"][0]["step"] == "max_mc2"
        assert diag["tie_break_path"][1]["step"] == "min_alpha"
        assert diag["tie_break_path"][2]["step"] == "min_k"


class TestPilotPoisonGate:
    def test_pilot_gate_filters_poisoned_candidates(self):
        shortlist = [
            {"k": 8, "alpha": 8.0, "mc1": 0.31, "mc2": 0.43},
            {"k": 12, "alpha": 8.0, "mc1": 0.31, "mc2": 0.44},
        ]
        pilot_map = {
            lock_config._candidate_key(8, 8.0): {
                "k": 8,
                "alpha": 8.0,
                "attempt_delta_pp": -12.0,
                "precision_delta_pp": -0.1,
                "not_attempted_delta_n": 20,
            },
            lock_config._candidate_key(12, 8.0): {
                "k": 12,
                "alpha": 8.0,
                "attempt_delta_pp": -2.0,
                "precision_delta_pp": +0.2,
                "not_attempted_delta_n": 3,
            },
        }
        survivors, diagnostics = lock_config._apply_pilot_poison_gate(
            shortlist,
            pilot_map=pilot_map,
            attempt_threshold_pp=-10.0,
            precision_threshold_pp=0.0,
            not_attempted_threshold_n=15,
        )
        assert len(survivors) == 1
        assert survivors[0]["k"] == 12
        rejected = [row for row in diagnostics if row["k"] == 8][0]
        assert rejected["rejected"] is True
        assert "attempt_and_precision_gate" in rejected["rejection_reasons"]
        assert "not_attempted_spike_gate" in rejected["rejection_reasons"]

    def test_pilot_precision_ci_is_zero_when_no_attempts(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        experiment_dir = tmp_path / "pilot"
        experiment_dir.mkdir()
        baseline_path = experiment_dir / "alpha_0.0.jsonl"
        candidate_path = experiment_dir / "alpha_8.0.jsonl"
        rows = [
            {"id": "q1", "simpleqa_grade": "NOT_ATTEMPTED"},
            {"id": "q2", "simpleqa_grade": "NOT_ATTEMPTED"},
            {"id": "q3", "simpleqa_grade": "NOT_ATTEMPTED"},
        ]
        baseline_path.write_text(
            "\n".join(json.dumps(row) for row in rows) + "\n",
            encoding="utf-8",
        )
        candidate_path.write_text(
            "\n".join(json.dumps(row) for row in rows) + "\n",
            encoding="utf-8",
        )

        # Keep test runtime low while still exercising paired bootstrap path.
        monkeypatch.setattr(
            report_simpleqa_shortlist_pilot, "DEFAULT_BOOTSTRAP_RESAMPLES", 200
        )
        summary = report_simpleqa_shortlist_pilot._summary_for_candidate(
            k=8,
            alpha=8.0,
            experiment_dir=experiment_dir,
            baseline_alpha=0.0,
            seed=42,
            attempt_gate_pp=-10.0,
            precision_gate_pp=0.0,
            not_attempted_gate_n=15,
        )

        assert summary["precision"]["baseline"] == 0.0
        assert summary["precision"]["candidate"] == 0.0
        assert summary["precision"]["baseline_ci"]["lower"] == 0.0
        assert summary["precision"]["baseline_ci"]["upper"] == 0.0
        assert summary["precision"]["candidate_ci"]["lower"] == 0.0
        assert summary["precision"]["candidate_ci"]["upper"] == 0.0


class TestPairedIdParity:
    def test_shortlist_pilot_rejects_mismatched_ids(self):
        baseline_map = {"q1": "CORRECT", "q2": "INCORRECT"}
        candidate_map = {"q1": "CORRECT"}
        with pytest.raises(ValueError, match="paired sample IDs must match exactly"):
            report_simpleqa_shortlist_pilot._require_identical_sample_ids(
                baseline_map,
                candidate_map,
                context="pilot",
            )

    def test_canonical_mc_paired_rejects_mismatched_ids(self):
        baseline_map: dict[str, float | bool] = {"f0:q1": True, "f0:q2": False}
        compare_map: dict[str, float | bool] = {"f0:q1": True}
        with pytest.raises(ValueError, match="paired sample IDs must match exactly"):
            report_e2_canonical._paired_delta_from_maps(
                baseline_map,
                compare_map,
                variant="mc1",
                seed=42,
            )

    def test_canonical_simpleqa_paired_rejects_mismatched_ids(self):
        baseline_map = {"q1": "CORRECT", "q2": "NOT_ATTEMPTED"}
        compare_map = {"q1": "CORRECT"}
        with pytest.raises(ValueError, match="paired sample IDs must match exactly"):
            report_e2_canonical._paired_simpleqa_delta(
                baseline_map,
                compare_map,
                seed=42,
            )
