from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from select_intervention_aware_c import select_intervention_aware_c  # noqa: E402


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _model_bytes(c_value: float) -> bytes:
    return f"model for C={c_value:g}".encode("utf-8")


def _model_hash(c_value: float) -> str:
    return hashlib.sha256(_model_bytes(c_value)).hexdigest()


def _write_model(tmp_path: Path, c_value: float) -> tuple[Path, str]:
    path = tmp_path / f"candidate_{c_value:g}.pkl"
    path.write_bytes(_model_bytes(c_value))
    return path, _sha256(path)


def _write_classifier_metrics(
    tmp_path: Path,
    candidates: list[dict[str, float | int]],
) -> Path:
    rows = []
    for candidate in candidates:
        model_path, model_hash = _write_model(tmp_path, float(candidate["C"]))
        rows.append(
            {
                "C": float(candidate["C"]),
                "selected_h_neurons": int(candidate["selected_h_neurons"]),
                "train_metrics": {"accuracy": 1.0},
                "test_metrics": {"accuracy": float(candidate["dev_accuracy"])},
                "candidate_model_path": str(model_path),
                "candidate_model_sha256": model_hash,
            }
        )
    path = tmp_path / "classifier_metrics.json"
    path.write_text(json.dumps({"candidates": rows}), encoding="utf-8")
    return path


def _write_triviaqa_summary(
    tmp_path: Path,
    c_value: float,
    accuracy: float,
    *,
    n_total: int = 10,
    classifier_hash: str | None = None,
) -> Path:
    run_dir = tmp_path / f"triviaqa_C_{c_value:g}"
    run_dir.mkdir()
    n_correct = round(accuracy * n_total)
    summary = {
        "benchmark": "triviaqa_bridge",
        "intervention_run_config": {
            "classifier": {
                "content_sha256": classifier_hash or _model_hash(c_value),
            }
        },
        "results": {
            "0.0": {
                "n_total": n_total,
                "deterministic_accuracy_rate": accuracy,
                "n_deterministic_correct": n_correct,
                "deterministic_accuracy": {
                    "estimate": accuracy,
                    "n_deterministic_correct": n_correct,
                    "n_total": n_total,
                },
            }
        },
    }
    path = run_dir / "results.20260430_000000.json"
    path.write_text(json.dumps(summary), encoding="utf-8")
    return run_dir


def test_selector_computes_coupled_score_and_copies_winner(tmp_path: Path) -> None:
    metrics_path = _write_classifier_metrics(
        tmp_path,
        [
            {"C": 0.05, "dev_accuracy": 0.70, "selected_h_neurons": 3},
            {"C": 0.1, "dev_accuracy": 0.65, "selected_h_neurons": 4},
        ],
    )
    triviaqa_results = {
        0.05: _write_triviaqa_summary(tmp_path, 0.05, 0.20),
        0.1: _write_triviaqa_summary(tmp_path, 0.1, 0.30),
    }

    summary = select_intervention_aware_c(
        classifier_metrics_path=metrics_path,
        triviaqa_results=triviaqa_results,
        output_summary_path=tmp_path / "selection_summary.json",
        selected_model_out=tmp_path / "selected.pkl",
    )

    selected = summary["selected"]
    assert selected["C"] == 0.1
    assert selected["heldout_dev_accuracy"] == pytest.approx(0.65)
    assert selected["triviaqa_alpha0_deterministic_accuracy_rate"] == pytest.approx(
        0.30
    )
    assert selected["selection_score"] == pytest.approx(0.95)
    assert (
        Path(selected["selected_model_path"]).read_bytes()
        == Path(selected["candidate_model_path"]).read_bytes()
    )


def test_selector_tie_break_order_is_deterministic(tmp_path: Path) -> None:
    metrics_path = _write_classifier_metrics(
        tmp_path,
        [
            {"C": 0.1, "dev_accuracy": 0.60, "selected_h_neurons": 5},
            {"C": 0.2, "dev_accuracy": 0.50, "selected_h_neurons": 7},
            {"C": 0.25, "dev_accuracy": 0.50, "selected_h_neurons": 4},
            {"C": 0.3, "dev_accuracy": 0.50, "selected_h_neurons": 4},
        ],
    )
    triviaqa_results = {
        0.1: _write_triviaqa_summary(tmp_path, 0.1, 0.30),
        0.2: _write_triviaqa_summary(tmp_path, 0.2, 0.40),
        0.25: _write_triviaqa_summary(tmp_path, 0.25, 0.40),
        0.3: _write_triviaqa_summary(tmp_path, 0.3, 0.40),
    }

    summary = select_intervention_aware_c(
        classifier_metrics_path=metrics_path,
        triviaqa_results=triviaqa_results,
        output_summary_path=tmp_path / "selection_summary.json",
        selected_model_out=tmp_path / "selected.pkl",
    )

    assert summary["selected"]["C"] == 0.25


def test_selector_refuses_missing_candidate_metrics(tmp_path: Path) -> None:
    model_path, model_hash = _write_model(tmp_path, 0.1)
    metrics_path = tmp_path / "classifier_metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "candidates": [
                    {
                        "C": 0.1,
                        "selected_h_neurons": 3,
                        "test_metrics": {},
                        "candidate_model_path": str(model_path),
                        "candidate_model_sha256": model_hash,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    triviaqa_results = {0.1: _write_triviaqa_summary(tmp_path, 0.1, 0.3)}

    with pytest.raises(ValueError, match="missing held-out dev accuracy"):
        select_intervention_aware_c(
            classifier_metrics_path=metrics_path,
            triviaqa_results=triviaqa_results,
            output_summary_path=tmp_path / "selection_summary.json",
            selected_model_out=tmp_path / "selected.pkl",
        )


def test_selector_refuses_missing_per_c_results(tmp_path: Path) -> None:
    metrics_path = _write_classifier_metrics(
        tmp_path,
        [{"C": 0.1, "dev_accuracy": 0.6, "selected_h_neurons": 3}],
    )

    with pytest.raises(ValueError, match="Missing per-C TriviaQA results"):
        select_intervention_aware_c(
            classifier_metrics_path=metrics_path,
            triviaqa_results={},
            output_summary_path=tmp_path / "selection_summary.json",
            selected_model_out=tmp_path / "selected.pkl",
        )


def test_selector_refuses_zero_row_triviaqa_outputs(tmp_path: Path) -> None:
    metrics_path = _write_classifier_metrics(
        tmp_path,
        [{"C": 0.1, "dev_accuracy": 0.6, "selected_h_neurons": 3}],
    )
    triviaqa_results = {0.1: _write_triviaqa_summary(tmp_path, 0.1, 0.0, n_total=0)}

    with pytest.raises(ValueError, match="zero-row TriviaQA output"):
        select_intervention_aware_c(
            classifier_metrics_path=metrics_path,
            triviaqa_results=triviaqa_results,
            output_summary_path=tmp_path / "selection_summary.json",
            selected_model_out=tmp_path / "selected.pkl",
        )


def test_selector_refuses_stale_triviaqa_classifier_hash(tmp_path: Path) -> None:
    metrics_path = _write_classifier_metrics(
        tmp_path,
        [{"C": 0.1, "dev_accuracy": 0.6, "selected_h_neurons": 3}],
    )
    triviaqa_results = {
        0.1: _write_triviaqa_summary(
            tmp_path,
            0.1,
            0.3,
            classifier_hash="old-classifier-hash",
        )
    }

    with pytest.raises(ValueError, match="classifier hash mismatch"):
        select_intervention_aware_c(
            classifier_metrics_path=metrics_path,
            triviaqa_results=triviaqa_results,
            output_summary_path=tmp_path / "selection_summary.json",
            selected_model_out=tmp_path / "selected.pkl",
        )
