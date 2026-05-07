"""Tests for E2 selector overrides and lock robustness utilities."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import extract_truthfulness_iti as extract_iti
import lock_config
import report_e2_canonical
import report_simpleqa_shortlist_pilot
import report_iti_2fold
import run_calibration_sweep


REPO_ROOT = Path(__file__).resolve().parent.parent


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

    def test_calibration_sweep_passes_registered_mistral_tokenizer_kwargs(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        tokenizer_calls: list[tuple[str, dict]] = []
        model_calls: list[tuple[str, dict]] = []

        class FakeModel:
            def eval(self):
                return None

            def parameters(self):
                import torch

                yield torch.zeros(1)

        class FakeScaler:
            def __init__(self, *args, **kwargs):
                self.alpha = 0.0

            def remove(self):
                return None

        def fake_tokenizer_from_pretrained(model_path, **kwargs):
            tokenizer_calls.append((model_path, kwargs))
            return object()

        def fake_model_from_pretrained(model_path, **kwargs):
            model_calls.append((model_path, kwargs))
            return FakeModel()

        def fake_load_truthfulqa_mc(variant, csv_path="data/benchmarks/TruthfulQA.csv"):
            return [{"id": f"{variant}_sample"}]

        def fake_score_mc_samples(model, tokenizer, scaler, samples, alpha):
            if samples and samples[0]["id"] == "mc1_sample":
                return [{"id": "mc1_sample", "mc1_correct": True}]
            return [{"id": "mc2_sample", "truthful_mass": 0.75}]

        artifact_path = tmp_path / "iti_heads.pt"
        artifact_path.write_bytes(b"fake")
        mc1_manifest = tmp_path / "mc1.json"
        mc2_manifest = tmp_path / "mc2.json"
        mc1_manifest.write_text(json.dumps(["mc1_sample"]), encoding="utf-8")
        mc2_manifest.write_text(json.dumps(["mc2_sample"]), encoding="utf-8")
        output_dir = tmp_path / "sweep"

        monkeypatch.setattr(
            run_calibration_sweep.AutoTokenizer,
            "from_pretrained",
            fake_tokenizer_from_pretrained,
        )
        monkeypatch.setattr(
            run_calibration_sweep.AutoModelForCausalLM,
            "from_pretrained",
            fake_model_from_pretrained,
        )
        monkeypatch.setattr(
            run_calibration_sweep,
            "load_iti_artifact",
            lambda _: {
                "family": "iti_truthfulqa_paperfaithful",
                "ranked_heads": [{"layer": 0, "head": 0}],
            },
        )
        monkeypatch.setattr(run_calibration_sweep, "ITIHeadScaler", FakeScaler)
        monkeypatch.setattr(
            run_calibration_sweep,
            "load_truthfulqa_mc",
            fake_load_truthfulqa_mc,
        )
        monkeypatch.setattr(
            run_calibration_sweep,
            "score_mc_samples",
            fake_score_mc_samples,
        )
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "run_calibration_sweep.py",
                "--model_key",
                "mistral24b",
                "--artifact_path",
                str(artifact_path),
                "--cal_val_mc1_manifest",
                str(mc1_manifest),
                "--cal_val_mc2_manifest",
                str(mc2_manifest),
                "--k_values",
                "1",
                "--alpha_values",
                "0.0",
                "--output_dir",
                str(output_dir),
            ],
        )

        run_calibration_sweep.main()

        assert tokenizer_calls == [
            (
                "mistralai/Mistral-Small-24B-Instruct-2501",
                {"fix_mistral_regex": True},
            )
        ]
        assert model_calls[0][0] == "mistralai/Mistral-Small-24B-Instruct-2501"
        locked = json.loads((output_dir / "locked_iti_config.json").read_text())
        assert locked["registered_model"]["key"] == "mistral_small_24b_instruct_2501"
        assert locked["tokenizer_kwargs"] == {"fix_mistral_regex": True}


class TestTruthfulQAMCGate:
    @staticmethod
    def _write_alpha(path: Path, rows: list[dict]) -> None:
        path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )

    def test_mc1_positive_ci_gate_passes_for_uniform_paired_improvement(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        fold_dir = tmp_path / "mc1"
        fold_dir.mkdir()
        baseline_rows = [
            {"id": f"q{i}", "compliance": False, "metric_value": 0.0} for i in range(6)
        ]
        locked_rows = [
            {"id": f"q{i}", "compliance": True, "metric_value": 1.0} for i in range(6)
        ]
        self._write_alpha(fold_dir / "alpha_0.0.jsonl", baseline_rows)
        self._write_alpha(fold_dir / "alpha_8.0.jsonl", locked_rows)
        monkeypatch.setattr(
            report_iti_2fold,
            "paired_bootstrap_binary_rate_difference",
            lambda baseline, locked: {
                "estimate_pp": 100.0,
                "ci_pp": {"lower": 100.0, "upper": 100.0},
            },
        )

        fold_report, baseline_arr, locked_arr = report_iti_2fold.compute_fold_report(
            str(fold_dir),
            locked_alpha=8.0,
            variant="mc1",
            fold_idx=0,
        )
        pooled = report_iti_2fold.compute_pooled_report(
            "mc1",
            8.0,
            [(baseline_arr, locked_arr)],
        )
        full_report = {
            "locked_alpha": 8.0,
            "locked_k": 12,
            "variant": "mc1",
            "folds": [fold_report],
            "pooled": pooled,
        }

        gate = report_iti_2fold.build_gate_decision(
            full_report,
            "mc1_positive_ci",
        )

        assert gate is not None
        assert gate["passed"] is True

    def test_sample_manifest_binding_accepts_exact_pooled_ids(self, tmp_path: Path):
        manifest = tmp_path / "manifest.json"
        manifest.write_text(json.dumps(["q1", "q2"]), encoding="utf-8")

        binding = report_iti_2fold.validate_sample_manifest_binding(
            [["q2"], ["q1"]],
            manifest,
        )

        assert binding["path"] == str(manifest)
        assert binding["n_ids"] == 2
        assert binding["validated"] is True

    def test_sample_manifest_binding_rejects_duplicate_pooled_ids(self, tmp_path: Path):
        manifest = tmp_path / "manifest.json"
        manifest.write_text(json.dumps(["q1", "q2"]), encoding="utf-8")

        with pytest.raises(
            ValueError,
            match="duplicate sample IDs across fold directories",
        ):
            report_iti_2fold.validate_sample_manifest_binding(
                [["q1"], ["q1"]],
                manifest,
            )

    def test_truthfulqa_report_rejects_manifest_mismatch_before_gate(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        fold_dir = tmp_path / "mc1"
        fold_dir.mkdir()
        manifest = tmp_path / "manifest.json"
        output_path = tmp_path / "report.json"
        manifest.write_text(json.dumps(["q1", "q2"]), encoding="utf-8")
        self._write_alpha(
            fold_dir / "alpha_0.0.jsonl",
            [{"id": "q1", "compliance": False, "metric_value": 0.0}],
        )
        self._write_alpha(
            fold_dir / "alpha_8.0.jsonl",
            [{"id": "q1", "compliance": True, "metric_value": 1.0}],
        )
        monkeypatch.setattr(
            report_iti_2fold,
            "paired_bootstrap_binary_rate_difference",
            lambda baseline, locked: {
                "estimate_pp": 100.0,
                "ci_pp": {"lower": 100.0, "upper": 100.0},
            },
        )
        monkeypatch.setattr(
            report_iti_2fold,
            "parse_args",
            lambda: argparse.Namespace(
                fold0_dir=None,
                fold1_dir=None,
                fold_dirs=[str(fold_dir)],
                locked_alpha=8.0,
                locked_k=12,
                variant="mc1",
                output_dir=str(tmp_path),
                output_prefix="iti_2fold",
                output_path=str(output_path),
                gate_mode="mc1_positive_ci",
                sample_manifest=str(manifest),
            ),
        )

        with pytest.raises(
            ValueError,
            match="sample IDs must exactly match --sample_manifest",
        ):
            report_iti_2fold.main()

        assert not output_path.exists()

    def test_truthfulqa_report_rejects_mismatched_paired_ids(self, tmp_path: Path):
        fold_dir = tmp_path / "mc1"
        fold_dir.mkdir()
        self._write_alpha(
            fold_dir / "alpha_0.0.jsonl",
            [{"id": "q1", "compliance": False, "metric_value": 0.0}],
        )
        self._write_alpha(
            fold_dir / "alpha_8.0.jsonl",
            [{"id": "q2", "compliance": True, "metric_value": 1.0}],
        )

        with pytest.raises(ValueError, match="paired sample IDs must match exactly"):
            report_iti_2fold.compute_fold_report(
                str(fold_dir),
                locked_alpha=8.0,
                variant="mc1",
                fold_idx=0,
            )

    def test_truthfulqa_report_rejects_empty_paired_ids(self, tmp_path: Path):
        fold_dir = tmp_path / "mc1"
        fold_dir.mkdir()
        self._write_alpha(fold_dir / "alpha_0.0.jsonl", [])
        self._write_alpha(fold_dir / "alpha_8.0.jsonl", [])

        with pytest.raises(ValueError, match="no paired sample IDs"):
            report_iti_2fold.compute_fold_report(
                str(fold_dir),
                locked_alpha=8.0,
                variant="mc1",
                fold_idx=0,
            )

    def test_truthfulqa_report_rejects_duplicate_paired_ids(self, tmp_path: Path):
        fold_dir = tmp_path / "mc1"
        fold_dir.mkdir()
        self._write_alpha(
            fold_dir / "alpha_0.0.jsonl",
            [
                {"id": "q1", "compliance": False, "metric_value": 0.0},
                {"id": "q1", "compliance": True, "metric_value": 1.0},
            ],
        )
        self._write_alpha(
            fold_dir / "alpha_8.0.jsonl",
            [{"id": "q1", "compliance": True, "metric_value": 1.0}],
        )

        with pytest.raises(ValueError, match="Duplicate sample IDs in baseline"):
            report_iti_2fold.compute_fold_report(
                str(fold_dir),
                locked_alpha=8.0,
                variant="mc1",
                fold_idx=0,
            )

    def test_anchor2_wrapper_binds_mc_gate_report_to_locked_sample_set(self):
        script = (
            REPO_ROOT / "scripts/infra/mistral24b_anchor2_iti_bridge.sh"
        ).read_text()

        assert '--sample_manifest "${TRUTHFULQA_MC1_MANIFEST}"' in script
        assert '--sample_manifest "${TRUTHFULQA_MC2_MANIFEST}"' in script
        assert (
            '"${MC1_REPORT_PATH}" "${LOCKED_K_VALUE}" "${LOCKED_ALPHA_VALUE}" '
            '"${TRUTHFULQA_MC1_MANIFEST}"'
        ) in script


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
