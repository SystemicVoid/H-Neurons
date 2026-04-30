from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import warnings

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from classifier import (  # noqa: E402
    file_sha256,
    fit_model,
    main as classifier_main,
    parse_args,
    persist_candidate_model,
    summarize_candidate,
)


def _set_argv(monkeypatch: pytest.MonkeyPatch, *extra: str) -> None:
    monkeypatch.setattr(sys, "argv", ["classifier.py", *extra])


class _TinyModelDimensions:
    total_ffn_neurons = 3

    def to_dict(self) -> dict[str, int]:
        return {"total_ffn_neurons": self.total_ffn_neurons}


def _write_activation(root: Path, qid: str, values: list[float]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    np.save(root / f"act_{qid}.npy", np.array(values))


def test_save_model_falls_back_to_registry_default_when_model_key_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_argv(monkeypatch, "--model_key", "mistral24b")
    args = parse_args()
    assert args.save_model == "models/mistral24b_classifier.pkl"


def test_save_model_legacy_default_when_no_registry_match(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_argv(
        monkeypatch,
        "--model_path",
        "/workspace/models/some-unregistered-model",
    )
    args = parse_args()
    assert args.save_model == "models/detector.pkl"


def test_save_model_legacy_default_when_no_model_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("HNEURONS_MODEL_KEY", raising=False)
    monkeypatch.delenv("HNEURONS_MODEL_PATH", raising=False)
    _set_argv(monkeypatch)

    args = parse_args()

    assert args.save_model == "models/detector.pkl"


def test_save_model_explicit_override_wins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_argv(
        monkeypatch,
        "--model_key",
        "mistral24b",
        "--save_model",
        "custom/path.pkl",
    )
    args = parse_args()
    assert args.save_model == "custom/path.pkl"


def test_load_model_keeps_save_model_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_argv(
        monkeypatch,
        "--model_key",
        "mistral24b",
        "--load_model",
        "models/mistral24b_classifier.pkl",
    )
    args = parse_args()
    assert args.save_model is None
    assert args.candidate_models_dir is None


def test_candidate_models_dir_rejected_with_load_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _set_argv(
        monkeypatch,
        "--model_key",
        "mistral24b",
        "--load_model",
        "models/mistral24b_classifier.pkl",
        "--candidate_models_dir",
        str(tmp_path / "candidates"),
    )

    with pytest.raises(ValueError, match="candidate_models_dir"):
        classifier_main()


def test_save_model_uses_registry_default_for_gemma3_4b(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_argv(monkeypatch, "--model_key", "gemma3_4b_it")
    args = parse_args()
    assert args.save_model == "models/gemma3_4b_classifier.pkl"


def test_l1_liblinear_uses_l1_ratio_without_penalty_deprecation() -> None:
    args = argparse.Namespace(penalty="l1", solver="liblinear")
    X = np.array(
        [
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.1, 0.0, 0.8],
            [0.9, 1.0, 0.1],
        ]
    )
    y = np.array([0, 0, 1, 1, 0, 1])

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = fit_model(args, X, y, c_value=1.0, verbose=0)

    messages = [str(warning.message) for warning in caught]
    assert model.get_params()["penalty"] == "deprecated"
    assert model.get_params()["l1_ratio"] == 1.0
    assert not any(
        "penalty" in message and "deprecated" in message for message in messages
    )
    assert not any("Inconsistent values" in message for message in messages)


def test_candidate_model_export_records_path_and_hash(tmp_path: Path) -> None:
    args = argparse.Namespace(penalty="l1", solver="liblinear")
    X = np.array(
        [
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.1, 0.0, 0.8],
            [0.9, 1.0, 0.1],
        ]
    )
    y = np.array([0, 0, 1, 1, 0, 1])
    model = fit_model(args, X, y, c_value=0.05, verbose=0)

    metadata = persist_candidate_model(model, tmp_path, 0.05)
    candidate_path = Path(metadata["candidate_model_path"])

    assert candidate_path == tmp_path / "classifier_C_0_05.pkl"
    assert candidate_path.exists()
    assert metadata["candidate_model_sha256"] == file_sha256(candidate_path)

    candidate = {
        "C": 0.05,
        "model": model,
        "train_metrics": {"accuracy": 1.0},
        "test_metrics": {"accuracy": 0.5},
        **metadata,
    }
    summary = summarize_candidate(
        candidate,
        total_neurons=3,
        selection_metric="accuracy",
        prefer_test=True,
    )
    assert summary["candidate_model_path"] == str(candidate_path)
    assert summary["candidate_model_sha256"] == metadata["candidate_model_sha256"]


def test_main_c_sweep_records_candidate_checkpoints_and_provenance(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "classifier.get_model_dimensions",
        lambda _model_path, _model_key=None: _TinyModelDimensions(),
    )
    monkeypatch.setattr(
        "classifier.model_metadata",
        lambda _model_key, _model_path: {"name": "tiny-test-model"},
    )

    train_ids = tmp_path / "train_qids.json"
    test_ids = tmp_path / "test_qids.json"
    train_ids.write_text(json.dumps({"f": ["f0", "f1"], "t": ["t0", "t1"]}))
    test_ids.write_text(json.dumps({"f": ["tf0", "tf1"], "t": ["tt0", "tt1"]}))

    train_ans = tmp_path / "train" / "answer_tokens"
    train_other = tmp_path / "train" / "all_except_answer_tokens"
    test_acts = tmp_path / "dev" / "answer_tokens"
    for qid in ["f0", "f1"]:
        _write_activation(train_ans, qid, [1.0, 1.0, 0.0])
        _write_activation(train_other, qid, [0.0, 0.0, 1.0])
    for qid in ["t0", "t1"]:
        _write_activation(train_ans, qid, [0.0, 0.0, 1.0])
        _write_activation(train_other, qid, [0.0, 1.0, 0.0])
    for qid in ["tf0", "tf1"]:
        _write_activation(test_acts, qid, [1.0, 1.0, 0.0])
    for qid in ["tt0", "tt1"]:
        _write_activation(test_acts, qid, [0.0, 0.0, 1.0])

    metrics_path = tmp_path / "metrics" / "classifier_c_sweep_metrics.json"
    candidate_models_dir = tmp_path / "candidate_models"
    saved_model_path = tmp_path / "models" / "selected.pkl"
    _set_argv(
        monkeypatch,
        "--train_ids",
        str(train_ids),
        "--train_ans_acts",
        str(train_ans),
        "--train_other_acts",
        str(train_other),
        "--test_ids",
        str(test_ids),
        "--test_acts",
        str(test_acts),
        "--train_mode",
        "3-vs-1",
        "--solver",
        "liblinear",
        "--selection_metric",
        "accuracy",
        "--c_values",
        "0.05",
        "0.1",
        "--candidate_models_dir",
        str(candidate_models_dir),
        "--save_model",
        str(saved_model_path),
        "--metrics_out",
        str(metrics_path),
    )

    classifier_main()

    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert len(payload["candidates"]) == 2
    for candidate in payload["candidates"]:
        checkpoint = Path(candidate["candidate_model_path"])
        assert checkpoint.parent == candidate_models_dir
        assert checkpoint.exists()
        assert candidate["candidate_model_sha256"] == file_sha256(checkpoint)

    provenance_paths = sorted(
        metrics_path.parent.glob("classifier_c_sweep_metrics.json.provenance.*.json")
    )
    assert len(provenance_paths) == 1
    provenance = json.loads(provenance_paths[0].read_text(encoding="utf-8"))
    assert provenance["status"] == "completed"
    assert str(candidate_models_dir.resolve()) in provenance["output_targets"]


def test_candidate_model_metadata_absent_by_default() -> None:
    model = argparse.Namespace(coef_=[np.array([1.0, -1.0, 0.0])])
    candidate = {
        "C": 1.0,
        "model": model,
        "train_metrics": {"accuracy": 0.5},
        "test_metrics": None,
    }
    summary = summarize_candidate(
        candidate,
        total_neurons=3,
        selection_metric="accuracy",
        prefer_test=False,
    )
    assert "candidate_model_path" not in summary
    assert "candidate_model_sha256" not in summary
