"""Tests for scripts.lib.pipeline — incident-driven guard functions.

Each test traces to the invariant it protects. If a test here fails,
a GPU pipeline would silently skip incomplete data or lose work.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess

import numpy as np
import pytest

from scripts.lib.pipeline import (
    _count_lines,
    active_run_registry_dir,
    build_active_run_lock,
    check_stage_complete,
    check_live_output_track_state,
    check_sentinel,
    check_staged_paths_against_live_runs,
    evaluate_gpu_hardware_snapshot,
    current_process_start_identity,
    log_run,
    main,
    manifest_count,
    path_intersects_live_run,
    validate_intervention_run_contract,
    validate_sample_manifest_locks,
)
from scripts.validate_mistral24b_cp23 import (
    validate_classifier_activation_inputs,
    validate_cp23_artifacts,
)
from scripts.utils import fingerprint_ids, format_alpha_label


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def manifest_file(tmp_path: Path) -> Path:
    """Create a 5-entry manifest."""
    p = tmp_path / "manifest.json"
    p.write_text(json.dumps(["id_0", "id_1", "id_2", "id_3", "id_4"]))
    return p


@pytest.fixture()
def stage_dir(tmp_path: Path) -> Path:
    d = tmp_path / "experiment"
    d.mkdir()
    return d


def _write_alpha(stage_dir: Path, alpha: float, n_lines: int) -> Path:
    """Write a fake alpha JSONL with n_lines records."""
    f = stage_dir / f"alpha_{format_alpha_label(alpha)}.jsonl"
    f.write_text(
        "".join(f'{{"id": "id_{i}", "alpha": {alpha}}}\n' for i in range(n_lines))
    )
    return f


def _current_start_identity_or_skip() -> str:
    identity = current_process_start_identity()
    if identity is None:
        pytest.skip("/proc start identity unavailable on this platform")
    return identity


def _write_active_run_lock(
    registry_dir: Path,
    protected_path: Path,
    *,
    kind: str = "file",
    run_id: str = "test-live-run",
    pid: int | None = None,
    start_identity: str | None = None,
    hostname: str | None = None,
) -> Path:
    registry_dir.mkdir(parents=True, exist_ok=True)
    process_id = os.getpid() if pid is None else pid
    identity = (
        _current_start_identity_or_skip() if start_identity is None else start_identity
    )
    lock = {
        "schema_version": "active_run_lock/v1",
        "run_id": run_id,
        "hostname": hostname or socket.gethostname(),
        "pid": process_id,
        "process_start_identity": {"start_ticks": identity},
        "protected_paths": [
            {"path": str(protected_path.resolve()), "kind": kind},
        ],
    }
    path = registry_dir / f"{run_id}.json"
    path.write_text(json.dumps(lock))
    return path


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


# ---------------------------------------------------------------------------
# _count_lines
# ---------------------------------------------------------------------------


class TestCountLines:
    def test_normal_file(self, tmp_path: Path) -> None:
        f = tmp_path / "data.jsonl"
        f.write_text("line1\nline2\nline3\n")
        assert _count_lines(f) == 3

    def test_no_trailing_newline(self, tmp_path: Path) -> None:
        f = tmp_path / "data.jsonl"
        f.write_text("line1\nline2")
        # wc -l counts newline-terminated lines; we count all lines
        assert _count_lines(f) == 2

    def test_empty_file(self, tmp_path: Path) -> None:
        f = tmp_path / "data.jsonl"
        f.write_text("")
        assert _count_lines(f) == 0


# ---------------------------------------------------------------------------
# manifest_count
# ---------------------------------------------------------------------------


class TestManifestCount:
    def test_basic(self, manifest_file: Path) -> None:
        assert manifest_count(manifest_file) == 5

    def test_sample_manifest_wrapper(self, tmp_path: Path) -> None:
        p = tmp_path / "sample_manifest.json"
        p.write_text(
            json.dumps({"schema_version": "sample_manifest/v2", "ids": ["a", "b"]})
        )
        assert manifest_count(p) == 2

    def test_empty_list(self, tmp_path: Path) -> None:
        p = tmp_path / "empty.json"
        p.write_text("[]")
        assert manifest_count(p) == 0

    def test_non_list_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.json"
        p.write_text('{"not": "a list"}')
        with pytest.raises(ValueError, match="neither a JSON list nor an object"):
            manifest_count(p)


class TestMistralManifestLocks:
    def test_lock_schema_and_source_hashes(self) -> None:
        expected_names = [
            "faitheval_iti_calibration_ids_seed42_n20_mistral24b.lock.json",
            "faitheval_seed42_n200_mistral24b.lock.json",
            "jbb_d7_full_harmful500_seed42_mistral24b.lock.json",
            "simpleqa_verified_control200_seed42_mistral24b.lock.json",
            "simpleqa_verified_pilot100_disjoint_seed42_mistral24b.lock.json",
            "triviaqa_bridge_test500_seed42_mistral24b.lock.json",
            "truthfulqa_paper_heldout_mc1_ids_seed42_mistral24b.lock.json",
            "truthfulqa_paper_heldout_mc2_ids_seed42_mistral24b.lock.json",
        ]
        lock_paths = sorted(Path("data/manifests").glob("*_mistral24b.lock.json"))

        assert [path.name for path in lock_paths] == expected_names
        assert validate_sample_manifest_locks(lock_paths) == []
        for path in lock_paths:
            payload = json.loads(path.read_text(encoding="utf-8"))
            ids = payload["ids"]
            source_path = Path(payload["source_path"])

            assert payload["schema_version"] == "sample_manifest/v2"
            assert payload["n_ids"] == len(ids)
            assert len(ids) == len(set(ids))
            assert payload["fingerprint"] == fingerprint_ids(ids)
            assert source_path.exists()
            assert (
                payload["source_sha256"]
                == hashlib.sha256(source_path.read_bytes()).hexdigest()
            )
            source_payload = json.loads(source_path.read_text(encoding="utf-8"))
            if isinstance(source_payload, list):
                source_ids = [str(item) for item in source_payload]
            elif isinstance(source_payload, dict):
                source_ids = [str(item) for item in source_payload["ids"]]
            else:
                raise AssertionError(f"Unsupported source payload in {source_path}")
            assert ids == source_ids

    def test_lock_validation_rejects_source_mismatch(self, tmp_path: Path) -> None:
        source = tmp_path / "source.json"
        source.write_text(json.dumps(["a", "b"]), encoding="utf-8")
        lock = tmp_path / "bad.lock.json"
        lock.write_text(
            json.dumps(
                {
                    "schema_version": "sample_manifest/v2",
                    "benchmark": "unit",
                    "split_name": "bad",
                    "ids": ["a"],
                    "n_ids": 1,
                    "fingerprint": fingerprint_ids(["a"]),
                    "source_path": str(source),
                    "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                    "lock_context": "mistral24b_claim_stage",
                }
            ),
            encoding="utf-8",
        )

        errors = validate_sample_manifest_locks([lock])

        assert any("ids do not match source manifest" in error for error in errors)


def _write_cp23_split(path: Path, false_ids: list[str], true_ids: list[str]) -> None:
    path.write_text(json.dumps({"f": false_ids, "t": true_ids}), encoding="utf-8")


def _write_activation_summary(
    root: Path,
    split_name: str,
    locations: list[str],
    count: int,
) -> None:
    split_root = root / split_name
    split_root.mkdir(parents=True)
    per_location = {
        location: {
            "final_completed_count": count,
            "missing_region_count": 0,
        }
        for location in locations
    }
    (split_root / "summary.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "triage_status": "ready_for_downstream",
                "per_location": per_location,
            }
        ),
        encoding="utf-8",
    )
    (split_root / "metadata.json").write_text(
        json.dumps({"schema_version": 1}),
        encoding="utf-8",
    )


def _write_activation_sample(
    root: Path,
    split_name: str,
    location: str,
    qid: str,
    shape: tuple[int, int] = (2, 3),
) -> None:
    location_root = root / split_name / location
    location_root.mkdir(parents=True, exist_ok=True)
    np.save(location_root / f"act_{qid}.npy", np.ones(shape, dtype=np.float32))


def _interval_metric(value: float) -> dict:
    return {
        "estimate": value,
        "ci": {
            "lower": value,
            "upper": value,
            "level": 0.95,
            "method": "unit",
        },
    }


def _evaluation_payload() -> dict:
    point = {
        "accuracy": 0.75,
        "f1": 0.76,
        "auroc": 0.84,
        "precision": 0.7,
        "recall": 0.8,
    }
    return {
        "metrics": {
            name: _interval_metric(point[name]) for name in ("accuracy", "f1", "auroc")
        },
        "point_metrics": point,
        "n_examples": 2,
        "n_positive": 1,
        "n_negative": 1,
        "confusion_matrix": {"tp": 1, "tn": 1, "fp": 0, "fn": 0},
        "bootstrap": {"n_resamples": 10},
    }


def _write_cp23_metrics(
    pipeline_dir: Path,
    *,
    classifier_path: str = "models/mistral24b_classifier_canonical.pkl",
) -> None:
    dev_metrics = {
        "model_key": "mistral_small_24b_instruct_2501",
        "model_path": "mistralai/Mistral-Small-24B-Instruct-2501",
        "total_ffn_neurons": 6,
        "selection_split": "test",
        "best": {
            "C": 1.0,
            "selected_h_neurons": 3,
            "test_metrics": {
                "accuracy": 0.75,
                "f1": 0.76,
                "auroc": 0.84,
            },
        },
        "evaluation": _evaluation_payload(),
    }
    test_metrics = {
        "model_key": "mistral_small_24b_instruct_2501",
        "model_path": "mistralai/Mistral-Small-24B-Instruct-2501",
        "total_ffn_neurons": 6,
        "loaded_model_path": classifier_path,
        "selection_split": "pretrained_load",
        "selected_h_neurons": 3,
        "evaluation": _evaluation_payload(),
    }
    (pipeline_dir / "classifier_canonical_dev_metrics.json").write_text(
        json.dumps(dev_metrics), encoding="utf-8"
    )
    (pipeline_dir / "classifier_canonical_test_metrics.json").write_text(
        json.dumps(test_metrics), encoding="utf-8"
    )
    (
        pipeline_dir / "classifier_canonical_dev_metrics.json.provenance.1.json"
    ).write_text(
        json.dumps({"args": {"save_model": classifier_path}, "status": "completed"}),
        encoding="utf-8",
    )
    (
        pipeline_dir / "classifier_canonical_test_metrics.json.provenance.1.json"
    ).write_text(
        json.dumps({"args": {"load_model": classifier_path}, "status": "completed"}),
        encoding="utf-8",
    )


class TestMistralCp23Validator:
    def test_accepts_current_nested_metric_schema_and_count_selected_neurons(
        self, tmp_path: Path
    ) -> None:
        pipeline_dir = tmp_path / "pipeline"
        activation_root = pipeline_dir / "activations_llm_canonical"
        pipeline_dir.mkdir()
        _write_cp23_split(
            pipeline_dir / "train_qids_llm.json", ["f1", "f2"], ["t1", "t2"]
        )
        _write_cp23_split(pipeline_dir / "dev_qids_llm.json", ["f3"], ["t3"])
        _write_cp23_split(pipeline_dir / "test_qids_llm.json", ["f4"], ["t4"])
        _write_activation_summary(
            activation_root,
            "train",
            ["answer_tokens", "all_except_answer_tokens"],
            4,
        )
        _write_activation_summary(activation_root, "dev", ["answer_tokens"], 2)
        _write_activation_summary(activation_root, "test", ["answer_tokens"], 2)
        _write_activation_sample(activation_root, "train", "answer_tokens", "f1")
        _write_activation_sample(
            activation_root, "train", "all_except_answer_tokens", "f1"
        )
        _write_activation_sample(activation_root, "dev", "answer_tokens", "f3")
        _write_activation_sample(activation_root, "test", "answer_tokens", "f4")
        _write_cp23_metrics(pipeline_dir)

        result = validate_cp23_artifacts(
            pipeline_dir=pipeline_dir,
            activation_root=activation_root,
            train_per_class=2,
            dev_per_class=1,
            test_per_class=1,
            layers=2,
            ffn_neurons=3,
            total_ffn_neurons=6,
        )

        assert result["accepted"] is True
        assert result["details"]["classifier_metrics"]["dev_selected_h_neurons"] == 3
        assert result["checks"]["dev_evaluation_metrics_auroc_estimate"] is True

    def test_rejects_mismatched_loaded_classifier_path(self, tmp_path: Path) -> None:
        pipeline_dir = tmp_path / "pipeline"
        activation_root = pipeline_dir / "activations_llm_canonical"
        pipeline_dir.mkdir()
        _write_cp23_split(
            pipeline_dir / "train_qids_llm.json", ["f1", "f2"], ["t1", "t2"]
        )
        _write_cp23_split(pipeline_dir / "dev_qids_llm.json", ["f3"], ["t3"])
        _write_cp23_split(pipeline_dir / "test_qids_llm.json", ["f4"], ["t4"])
        _write_activation_summary(
            activation_root,
            "train",
            ["answer_tokens", "all_except_answer_tokens"],
            4,
        )
        _write_activation_summary(activation_root, "dev", ["answer_tokens"], 2)
        _write_activation_summary(activation_root, "test", ["answer_tokens"], 2)
        _write_activation_sample(activation_root, "train", "answer_tokens", "f1")
        _write_activation_sample(
            activation_root, "train", "all_except_answer_tokens", "f1"
        )
        _write_activation_sample(activation_root, "dev", "answer_tokens", "f3")
        _write_activation_sample(activation_root, "test", "answer_tokens", "f4")
        _write_cp23_metrics(pipeline_dir, classifier_path="models/other.pkl")

        result = validate_cp23_artifacts(
            pipeline_dir=pipeline_dir,
            activation_root=activation_root,
            train_per_class=2,
            dev_per_class=1,
            test_per_class=1,
            layers=2,
            ffn_neurons=3,
            total_ffn_neurons=6,
        )

        assert result["accepted"] is False
        assert result["checks"]["test_loaded_model_path"] is False


class TestMistralClassifierActivationInputValidator:
    def test_accepts_complete_train_dev_activation_inputs(self, tmp_path: Path) -> None:
        pipeline_dir = tmp_path / "pipeline"
        activation_root = pipeline_dir / "activations_llm_canonical"
        pipeline_dir.mkdir()
        train_ids = pipeline_dir / "train_qids_llm.json"
        dev_ids = pipeline_dir / "dev_qids_llm.json"
        _write_cp23_split(train_ids, ["f1", "f2"], ["t1", "t2"])
        _write_cp23_split(dev_ids, ["f3"], ["t3"])
        _write_activation_summary(
            activation_root,
            "train",
            ["answer_tokens", "all_except_answer_tokens"],
            4,
        )
        _write_activation_summary(activation_root, "dev", ["answer_tokens"], 2)
        for qid in ["f1", "f2", "t1", "t2"]:
            _write_activation_sample(activation_root, "train", "answer_tokens", qid)
            _write_activation_sample(
                activation_root, "train", "all_except_answer_tokens", qid
            )
        for qid in ["f3", "t3"]:
            _write_activation_sample(activation_root, "dev", "answer_tokens", qid)

        result = validate_classifier_activation_inputs(
            train_ids_path=train_ids,
            dev_ids_path=dev_ids,
            train_answer_dir=activation_root / "train" / "answer_tokens",
            train_other_dir=activation_root / "train" / "all_except_answer_tokens",
            dev_answer_dir=activation_root / "dev" / "answer_tokens",
            train_per_class=2,
            dev_per_class=1,
            layers=2,
            ffn_neurons=3,
        )

        assert result["accepted"] is True
        assert (
            result["checks"]["train_answer_tokens_all_expected_files_present"] is True
        )
        assert (
            result["details"]["activation_inputs"]["dev_answer_tokens"][
                "present_expected_count"
            ]
            == 2
        )

    def test_rejects_missing_activation_file_even_when_summary_claims_complete(
        self, tmp_path: Path
    ) -> None:
        pipeline_dir = tmp_path / "pipeline"
        activation_root = pipeline_dir / "activations_llm_canonical"
        pipeline_dir.mkdir()
        train_ids = pipeline_dir / "train_qids_llm.json"
        dev_ids = pipeline_dir / "dev_qids_llm.json"
        _write_cp23_split(train_ids, ["f1"], ["t1"])
        _write_cp23_split(dev_ids, ["f2"], ["t2"])
        _write_activation_summary(
            activation_root,
            "train",
            ["answer_tokens", "all_except_answer_tokens"],
            2,
        )
        _write_activation_summary(activation_root, "dev", ["answer_tokens"], 2)
        for qid in ["f1", "t1"]:
            _write_activation_sample(activation_root, "train", "answer_tokens", qid)
            _write_activation_sample(
                activation_root, "train", "all_except_answer_tokens", qid
            )
        _write_activation_sample(activation_root, "dev", "answer_tokens", "f2")

        result = validate_classifier_activation_inputs(
            train_ids_path=train_ids,
            dev_ids_path=dev_ids,
            train_answer_dir=activation_root / "train" / "answer_tokens",
            train_other_dir=activation_root / "train" / "all_except_answer_tokens",
            dev_answer_dir=activation_root / "dev" / "answer_tokens",
            train_per_class=1,
            dev_per_class=1,
            layers=2,
            ffn_neurons=3,
        )

        assert result["accepted"] is False
        assert result["checks"]["dev_answer_tokens_all_expected_files_present"] is False
        assert result["details"]["activation_inputs"]["dev_answer_tokens"][
            "missing_ids_preview"
        ] == ["t2"]


# ---------------------------------------------------------------------------
# check_stage_complete — the core guard
# ---------------------------------------------------------------------------


class TestCheckStageComplete:
    """Incident: 2026-04-07 — alpha_8.0.jsonl had 64/100 lines after power-off.
    Old guard only checked file existence, so it skipped re-generation."""

    def test_all_complete(self, stage_dir: Path, manifest_file: Path) -> None:
        for alpha in [0.0, 1.0]:
            _write_alpha(stage_dir, alpha, 5)
        assert check_stage_complete(stage_dir, manifest_file, [0.0, 1.0]) is True

    def test_missing_file(self, stage_dir: Path, manifest_file: Path) -> None:
        _write_alpha(stage_dir, 0.0, 5)
        # alpha_1.0 is missing
        assert check_stage_complete(stage_dir, manifest_file, [0.0, 1.0]) is False

    def test_incomplete_file(self, stage_dir: Path, manifest_file: Path) -> None:
        """THE bug this library was created to catch."""
        _write_alpha(stage_dir, 0.0, 5)
        _write_alpha(stage_dir, 1.0, 3)  # only 3/5
        assert check_stage_complete(stage_dir, manifest_file, [0.0, 1.0]) is False

    def test_over_count_fails(self, stage_dir: Path, manifest_file: Path) -> None:
        """More records than expected can hide stale duplicate output rows."""
        _write_alpha(stage_dir, 0.0, 7)
        assert check_stage_complete(stage_dir, manifest_file, [0.0]) is False

    def test_duplicate_id_fails(self, stage_dir: Path, manifest_file: Path) -> None:
        f = _write_alpha(stage_dir, 0.0, 5)
        f.write_text(
            "".join(
                [
                    '{"id": "id_0", "alpha": 0.0}\n',
                    '{"id": "id_0", "alpha": 0.0}\n',
                    '{"id": "id_2", "alpha": 0.0}\n',
                    '{"id": "id_3", "alpha": 0.0}\n',
                    '{"id": "id_4", "alpha": 0.0}\n',
                ]
            )
        )
        assert check_stage_complete(stage_dir, manifest_file, [0.0]) is False

    def test_unexpected_id_fails(self, stage_dir: Path, manifest_file: Path) -> None:
        f = _write_alpha(stage_dir, 0.0, 5)
        f.write_text(
            "".join(
                [
                    '{"id": "id_0", "alpha": 0.0}\n',
                    '{"id": "id_1", "alpha": 0.0}\n',
                    '{"id": "id_2", "alpha": 0.0}\n',
                    '{"id": "id_3", "alpha": 0.0}\n',
                    '{"id": "other", "alpha": 0.0}\n',
                ]
            )
        )
        assert check_stage_complete(stage_dir, manifest_file, [0.0]) is False

    def test_missing_record_id_fails(
        self, stage_dir: Path, manifest_file: Path
    ) -> None:
        f = stage_dir / "alpha_0.0.jsonl"
        f.write_text('{"alpha": 0.0}\n')
        assert check_stage_complete(stage_dir, manifest_file, [0.0]) is False

    def test_empty_alpha_list(self, stage_dir: Path, manifest_file: Path) -> None:
        """No alphas to check = vacuously complete."""
        assert check_stage_complete(stage_dir, manifest_file, []) is True

    def test_diagnostics_on_stderr(
        self, stage_dir: Path, manifest_file: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _write_alpha(stage_dir, 0.0, 5)
        _write_alpha(stage_dir, 1.0, 2)
        check_stage_complete(stage_dir, manifest_file, [0.0, 1.0])
        captured = capsys.readouterr()
        assert "incomplete" in captured.err
        assert "2/5" in captured.err

    def test_missing_file_diagnostics(
        self, stage_dir: Path, manifest_file: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        check_stage_complete(stage_dir, manifest_file, [0.0])
        captured = capsys.readouterr()
        assert "missing" in captured.err

    def test_manifest_id_prefix_matches_namespaced_outputs(
        self, tmp_path: Path
    ) -> None:
        manifest = tmp_path / "triviaqa_manifest.json"
        manifest.write_text(json.dumps(["bb_2174", "tc_42"]))
        stage_dir = tmp_path / "experiment"
        stage_dir.mkdir()
        (stage_dir / "alpha_0.0.jsonl").write_text(
            '{"id": "tqa_bridge_bb_2174", "alpha": 0.0}\n'
            '{"id": "tqa_bridge_tc_42", "alpha": 0.0}\n'
        )

        assert (
            check_stage_complete(
                stage_dir,
                manifest,
                [0.0],
                manifest_id_prefix="tqa_bridge_",
            )
            is True
        )
        assert check_stage_complete(stage_dir, manifest, [0.0]) is False


# ---------------------------------------------------------------------------
# validate_intervention_run_contract
# ---------------------------------------------------------------------------


def test_intervention_contract_validation_accepts_matching_contract(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "experiment"
    output_dir.mkdir()
    classifier = tmp_path / "classifier.pkl"
    classifier.write_bytes(b"classifier-v1")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(["id_0", "id_1"]))
    (output_dir / "intervention_run_config.json").write_text(
        json.dumps(
            {
                "schema_version": "intervention_run_config/v1",
                "benchmark": "faitheval",
                "model": {"key": "mistral", "path": "mistral/path"},
                "classifier": {
                    "content_sha256": hashlib.sha256(
                        classifier.read_bytes()
                    ).hexdigest()
                },
                "sample_selection": {
                    "sample_manifest": {
                        "content_sha256": hashlib.sha256(
                            manifest.read_bytes()
                        ).hexdigest()
                    }
                },
                "schedule": {"alphas": [0.0, 1.0]},
                "benchmark_config": {"prompt_style": "standard"},
            }
        )
    )

    assert (
        validate_intervention_run_contract(
            output_dir,
            benchmark="faitheval",
            model_key="mistral",
            model_path="mistral/path",
            classifier_path=classifier,
            sample_manifest=manifest,
            alphas=[0.0, 1.0],
            benchmark_config={"prompt_style": "standard"},
        )
        == []
    )
    assert (
        main(
            [
                "check-intervention-contract",
                "--output-dir",
                str(output_dir),
                "--benchmark",
                "faitheval",
                "--model-key",
                "mistral",
                "--model-path",
                "mistral/path",
                "--classifier-path",
                str(classifier),
                "--sample-manifest",
                str(manifest),
                "--alphas",
                "0.0",
                "1.0",
                "--benchmark-config",
                "prompt_style=standard",
            ]
        )
        == 0
    )


def test_intervention_contract_validation_rejects_stale_classifier(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "experiment"
    output_dir.mkdir()
    classifier = tmp_path / "classifier.pkl"
    classifier.write_bytes(b"classifier-v2")
    (output_dir / "intervention_run_config.json").write_text(
        json.dumps(
            {
                "schema_version": "intervention_run_config/v1",
                "benchmark": "faitheval",
                "model": {"key": "mistral", "path": "mistral/path"},
                "classifier": {"content_sha256": "old-hash"},
                "sample_selection": {"sample_manifest": None},
                "schedule": {"alphas": [0.0]},
                "benchmark_config": {"prompt_style": "standard"},
            }
        )
    )

    errors = validate_intervention_run_contract(
        output_dir,
        benchmark="faitheval",
        model_key="mistral",
        model_path="mistral/path",
        classifier_path=classifier,
        alphas=[0.0],
        benchmark_config={"prompt_style": "standard"},
    )

    assert any("classifier.content_sha256 mismatch" in error for error in errors)


# ---------------------------------------------------------------------------
# log_run
# ---------------------------------------------------------------------------


class TestLogRun:
    def test_creates_file(self, tmp_path: Path) -> None:
        notes = tmp_path / "notes" / "runs_to_analyse.md"
        log_run(
            "data/foo/experiment", "jailbreak + causal + alphas 0-8", notes_file=notes
        )
        content = notes.read_text()
        assert "data/foo/experiment" in content
        assert "jailbreak + causal + alphas 0-8" in content
        assert "awaiting analysis" in content

    def test_appends_to_existing(self, tmp_path: Path) -> None:
        notes = tmp_path / "runs.md"
        notes.write_text("# Existing\n")
        log_run("run1", "first run", notes_file=notes)
        log_run("run2", "second run", notes_file=notes)
        content = notes.read_text()
        assert "# Existing" in content
        assert "run1" in content
        assert "run2" in content

    def test_includes_iso_timestamp(self, tmp_path: Path) -> None:
        notes = tmp_path / "runs.md"
        log_run("run1", "desc", notes_file=notes)
        content = notes.read_text()
        # ISO timestamp has T separator and +00:00 or Z
        assert "T" in content


# ---------------------------------------------------------------------------
# check_sentinel
# ---------------------------------------------------------------------------


class TestCheckSentinel:
    def test_missing_returns_none(self, tmp_path: Path) -> None:
        assert check_sentinel(tmp_path, "stop_after_pre_canary") is None

    def test_present_without_reason_returns_empty_string(self, tmp_path: Path) -> None:
        sentinel = tmp_path / "stop_after_pre_canary"
        sentinel.write_text("")
        assert check_sentinel(tmp_path, "stop_after_pre_canary") == ""

    def test_present_with_reason_returns_trimmed_text(self, tmp_path: Path) -> None:
        sentinel = tmp_path / "stop_after_pre_canary"
        sentinel.write_text("pause after canary\n")
        assert check_sentinel(tmp_path, "stop_after_pre_canary") == "pause after canary"


class TestGpuHardwareGuard:
    def test_accepts_h100_80gb_with_bf16(self) -> None:
        errors = evaluate_gpu_hardware_snapshot(
            {
                "available": True,
                "bf16_supported": True,
                "devices": [
                    {
                        "name": "NVIDIA H100 80GB HBM3",
                        "total_memory_bytes": 80 * 1024**3,
                    }
                ],
            },
            min_memory_gib=75,
            name_pattern="H100|A100",
        )

        assert errors == []

    def test_rejects_local_16gb_cuda_gpu(self) -> None:
        errors = evaluate_gpu_hardware_snapshot(
            {
                "available": True,
                "bf16_supported": True,
                "devices": [
                    {
                        "name": "NVIDIA GeForce RTX 5060 Ti",
                        "total_memory_bytes": 16 * 1024**3,
                    }
                ],
            },
            min_memory_gib=75,
            name_pattern="H100|A100",
        )

        assert errors == ["no CUDA device matches /H100|A100/ with >= 75 GiB"]

    def test_rejects_missing_bf16(self) -> None:
        errors = evaluate_gpu_hardware_snapshot(
            {
                "available": True,
                "bf16_supported": False,
                "devices": [
                    {
                        "name": "NVIDIA H100 80GB HBM3",
                        "total_memory_bytes": 80 * 1024**3,
                    }
                ],
            },
            min_memory_gib=75,
            name_pattern="H100|A100",
        )

        assert errors == ["CUDA BF16 support is not reported"]


# ---------------------------------------------------------------------------
# active-run Git guard
# ---------------------------------------------------------------------------


class TestActiveRunGitGuard:
    """Incident: 2026-04-24 — a commit touched JSONL files during a live run."""

    def test_registry_dir_resolves_git_common_dir_from_subdirectory(
        self, tmp_path: Path
    ) -> None:
        repo = tmp_path / "repo"
        repo.mkdir()
        _git(repo, "init")
        subdir = repo / "scripts"
        subdir.mkdir()

        registry = active_run_registry_dir(subdir)

        assert registry == (repo / ".git" / "h-neurons-active-runs").resolve()

    def test_directory_targets_are_protected_even_when_outputs_are_files(
        self, tmp_path: Path
    ) -> None:
        output_dir = tmp_path / "data" / "run"
        lock = build_active_run_lock(
            run_id="file-output-dir-run",
            provenance_path=output_dir / "run.provenance.json",
            output_targets=[output_dir / "results.json"],
            directory_targets=[output_dir],
            started_at_utc="2026-04-24T00:00:00+00:00",
            script="run_intervention.py",
            argv=["run_intervention.py"],
            cwd=tmp_path,
        )

        protected_paths = {
            (Path(item["path"]), item["kind"]) for item in lock["protected_paths"]
        }

        assert (output_dir.resolve(), "directory") in protected_paths
        assert path_intersects_live_run(
            "data/run/answer_span_scores.jsonl",
            lock,
            repo_root=tmp_path,
        )

    def test_pre_commit_guard_runs_without_filename_selection(self) -> None:
        config = (
            Path(__file__).resolve().parents[1] / ".pre-commit-config.yaml"
        ).read_text()
        hook_start = config.index("      - id: active-run-git-guard")
        next_hook = config.index("\n      - id:", hook_start + 1)
        hook_block = config[hook_start:next_hook]

        assert "pass_filenames: false" in hook_block
        assert "always_run: true" in hook_block

    def test_live_lock_blocks_staged_file_target(self, tmp_path: Path) -> None:
        registry = tmp_path / "registry"
        protected = tmp_path / "data" / "run" / "utility_scores.jsonl"
        _write_active_run_lock(registry, protected, kind="file")

        violations = check_staged_paths_against_live_runs(
            ["data/run/utility_scores.jsonl"],
            registry_dir=registry,
            repo_root=tmp_path,
        )

        assert len(violations) == 1
        assert violations[0].staged_path == "data/run/utility_scores.jsonl"
        assert path_intersects_live_run(
            "data/run/utility_scores.jsonl",
            json.loads((registry / "test-live-run.json").read_text()),
            repo_root=tmp_path,
        )

    def test_live_lock_blocks_staged_path_below_directory(self, tmp_path: Path) -> None:
        registry = tmp_path / "registry"
        protected_dir = tmp_path / "data" / "run"
        _write_active_run_lock(registry, protected_dir, kind="directory")

        violations = check_staged_paths_against_live_runs(
            ["data/run/answer_span_scores.jsonl"],
            registry_dir=registry,
            repo_root=tmp_path,
        )

        assert len(violations) == 1
        assert violations[0].protected_kind == "directory"

    def test_unrelated_staged_path_passes(self, tmp_path: Path) -> None:
        registry = tmp_path / "registry"
        _write_active_run_lock(
            registry,
            tmp_path / "data" / "run",
            kind="directory",
        )

        violations = check_staged_paths_against_live_runs(
            ["notes/research-log.md"],
            registry_dir=registry,
            repo_root=tmp_path,
        )

        assert violations == []

    def test_stale_missing_pid_passes(self, tmp_path: Path) -> None:
        registry = tmp_path / "registry"
        _write_active_run_lock(
            registry,
            tmp_path / "data" / "run",
            kind="directory",
            pid=999_999_999,
            start_identity="1",
        )

        violations = check_staged_paths_against_live_runs(
            ["data/run/out.jsonl"],
            registry_dir=registry,
            repo_root=tmp_path,
        )

        assert violations == []

    def test_reused_pid_with_mismatched_start_identity_passes(
        self, tmp_path: Path
    ) -> None:
        registry = tmp_path / "registry"
        _write_active_run_lock(
            registry,
            tmp_path / "data" / "run",
            kind="directory",
            start_identity="not-the-current-process-start",
        )

        violations = check_staged_paths_against_live_runs(
            ["data/run/out.jsonl"],
            registry_dir=registry,
            repo_root=tmp_path,
        )

        assert violations == []

    def test_malformed_lock_warns_but_passes(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        registry = tmp_path / "registry"
        registry.mkdir()
        (registry / "bad.json").write_text("{not json")

        violations = check_staged_paths_against_live_runs(
            ["data/run/out.jsonl"],
            registry_dir=registry,
            repo_root=tmp_path,
        )

        captured = capsys.readouterr()
        assert violations == []
        assert "ignoring malformed lock" in captured.err

    def test_cli_blocks_staged_path_owned_by_live_run(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        repo = tmp_path / "repo"
        repo.mkdir()
        _git(repo, "init")
        registry = repo / ".git" / "h-neurons-active-runs"
        protected_dir = repo / "data" / "run"
        protected_dir.mkdir(parents=True)
        _write_active_run_lock(registry, protected_dir, kind="directory")
        staged_file = protected_dir / "out.jsonl"
        staged_file.write_text("{}\n")
        _git(repo, "add", "data/run/out.jsonl")
        monkeypatch.chdir(repo)

        rc = main(["check-active-run-git-guard"])

        captured = capsys.readouterr()
        assert rc == 1
        assert "Active run Git guard blocked this commit" in captured.err
        assert "data/run/out.jsonl" in captured.err


class TestLiveOutputTrackState:
    def test_tracked_output_target_detection_blocks_tracked_file(
        self, tmp_path: Path
    ) -> None:
        repo = tmp_path / "repo"
        repo.mkdir()
        _git(repo, "init")
        tracked_file = repo / "data" / "run" / "results.json"
        tracked_file.parent.mkdir(parents=True)
        tracked_file.write_text("{}\n")
        _git(repo, "add", "data/run/results.json")

        violations = check_live_output_track_state(
            [repo / "data" / "run"],
            repo_root=repo,
        )

        assert len(violations) == 1
        assert violations[0].tracked_path == "data/run/results.json"

    def test_tracked_output_target_detection_passes_for_untracked_file(
        self, tmp_path: Path
    ) -> None:
        repo = tmp_path / "repo"
        repo.mkdir()
        _git(repo, "init")
        output_file = repo / "data" / "run" / "new_results.json"

        violations = check_live_output_track_state([output_file], repo_root=repo)

        assert violations == []

    def test_repo_relative_target_uses_explicit_repo_root_from_other_cwd(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        repo = tmp_path / "repo"
        repo.mkdir()
        _git(repo, "init")
        tracked_file = repo / "data" / "run" / "results.json"
        tracked_file.parent.mkdir(parents=True)
        tracked_file.write_text("{}\n")
        _git(repo, "add", "data/run/results.json")
        outside = tmp_path / "outside"
        outside.mkdir()
        monkeypatch.chdir(outside)

        violations = check_live_output_track_state(["data/run"], repo_root=repo)

        assert len(violations) == 1
        assert violations[0].output_target == tracked_file.parent.resolve()
        assert violations[0].tracked_path == "data/run/results.json"

    def test_repo_relative_target_uses_detected_repo_root_from_subdirectory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        repo = tmp_path / "repo"
        repo.mkdir()
        _git(repo, "init")
        tracked_file = repo / "data" / "run" / "results.json"
        tracked_file.parent.mkdir(parents=True)
        tracked_file.write_text("{}\n")
        _git(repo, "add", "data/run/results.json")
        subdir = repo / "scripts"
        subdir.mkdir()
        monkeypatch.chdir(subdir)

        violations = check_live_output_track_state(["data/run"])

        assert len(violations) == 1
        assert violations[0].output_target == tracked_file.parent.resolve()
        assert violations[0].tracked_path == "data/run/results.json"


# ---------------------------------------------------------------------------
# CLI integration
# ---------------------------------------------------------------------------


class TestCli:
    def test_check_stage_complete_exit_0(
        self, stage_dir: Path, manifest_file: Path
    ) -> None:
        _write_alpha(stage_dir, 0.0, 5)
        rc = main(
            [
                "check-stage",
                "--output-dir",
                str(stage_dir),
                "--manifest",
                str(manifest_file),
                "--alphas",
                "0.0",
            ]
        )
        assert rc == 0

    def test_check_stage_incomplete_exit_1(
        self, stage_dir: Path, manifest_file: Path
    ) -> None:
        _write_alpha(stage_dir, 0.0, 2)
        rc = main(
            [
                "check-stage",
                "--output-dir",
                str(stage_dir),
                "--manifest",
                str(manifest_file),
                "--alphas",
                "0.0",
            ]
        )
        assert rc == 1

    def test_manifest_count(
        self, manifest_file: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        rc = main(["manifest-count", str(manifest_file)])
        assert rc == 0
        captured = capsys.readouterr()
        assert "5" in captured.out

    def test_log_run(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        (tmp_path / "notes").mkdir()
        rc = main(
            [
                "log-run",
                "--run-dir",
                "data/foo",
                "--description",
                "test run",
            ]
        )
        assert rc == 0
        assert "data/foo" in (tmp_path / "notes" / "runs_to_analyse.md").read_text()

    def test_check_sentinel_exit_0_and_prints_reason(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        (tmp_path / "stop_after_pre_canary").write_text("human review gate\n")
        rc = main(
            [
                "check-sentinel",
                "--dir",
                str(tmp_path),
                "--name",
                "stop_after_pre_canary",
            ]
        )
        assert rc == 0
        captured = capsys.readouterr()
        assert "human review gate" in captured.out

    def test_check_sentinel_absent_exit_1(self, tmp_path: Path) -> None:
        rc = main(
            [
                "check-sentinel",
                "--dir",
                str(tmp_path),
                "--name",
                "stop_after_pre_canary",
            ]
        )
        assert rc == 1
