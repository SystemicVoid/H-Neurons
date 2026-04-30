from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from pathlib import Path
from typing import Any, Mapping


SCHEMA_VERSION = "intervention_aware_c_selection/v1"
SELECTION_SCORE_FORMULA = (
    "heldout_dev_accuracy + triviaqa_alpha0_deterministic_accuracy_rate"
)
TIE_BREAKERS = [
    "higher selection_score",
    "higher triviaqa_alpha0_deterministic_accuracy_rate",
    "higher heldout_dev_accuracy",
    "fewer selected_h_neurons",
    "lower C",
]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json_object(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing JSON file: {path}")
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def resolve_recorded_path(raw_path: str, *, base_dir: Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute() or path.exists():
        return path
    return base_dir / path


def resolve_triviaqa_summary_path(path: Path) -> Path:
    if path.is_file():
        return path
    if not path.exists():
        raise FileNotFoundError(f"Missing per-C TriviaQA result path: {path}")
    if not path.is_dir():
        raise ValueError(f"TriviaQA result path is not a file or directory: {path}")

    canonical = path / "results.json"
    if canonical.exists():
        return canonical

    candidates = sorted(path.glob("results.*.json"))
    if not candidates:
        raise FileNotFoundError(f"Missing per-C TriviaQA summary under: {path}")
    return candidates[-1]


def parse_triviaqa_result_spec(raw: str) -> tuple[float, Path]:
    separator = "=" if "=" in raw else ":"
    if separator not in raw:
        raise ValueError(
            f"--triviaqa_result entries must have the form C=PATH, got {raw!r}"
        )
    c_value_raw, path_raw = raw.split(separator, 1)
    if not c_value_raw or not path_raw:
        raise ValueError(
            f"--triviaqa_result entries must have the form C=PATH, got {raw!r}"
        )
    return float(c_value_raw), Path(path_raw)


def parse_triviaqa_result_specs(values: list[str]) -> dict[float, Path]:
    results: dict[float, Path] = {}
    for raw in values:
        c_value, path = parse_triviaqa_result_spec(raw)
        if any(math.isclose(c_value, existing, abs_tol=1e-12) for existing in results):
            raise ValueError(f"Duplicate --triviaqa_result for C={c_value:g}")
        results[c_value] = path
    return results


def lookup_triviaqa_path(c_value: float, results: Mapping[float, Path]) -> Path:
    for result_c, path in results.items():
        if math.isclose(float(result_c), c_value, abs_tol=1e-12):
            return path
    raise ValueError(f"Missing per-C TriviaQA results for C={c_value:g}")


def find_alpha_result(
    summary: dict[str, Any],
    *,
    alpha: float = 0.0,
    path: Path,
) -> dict[str, Any]:
    results = summary.get("results")
    if not isinstance(results, dict):
        raise ValueError(f"{path} missing results object")

    for raw_alpha, result in results.items():
        try:
            result_alpha = float(raw_alpha)
        except (TypeError, ValueError):
            continue
        if math.isclose(result_alpha, alpha, abs_tol=1e-12):
            if not isinstance(result, dict):
                raise ValueError(f"{path} results[{raw_alpha!r}] must be an object")
            return result
    raise ValueError(f"{path} missing alpha={alpha:g} TriviaQA results")


def validate_triviaqa_classifier_hash(
    summary: dict[str, Any],
    *,
    summary_path: Path,
    expected_classifier_sha256: str,
    c_value: float,
) -> None:
    if summary.get("benchmark") != "triviaqa_bridge":
        raise ValueError(f"{summary_path} is not a triviaqa_bridge summary")

    run_config = summary.get("intervention_run_config")
    if not isinstance(run_config, dict):
        raise ValueError(
            f"{summary_path} missing intervention_run_config for C={c_value:g}"
        )

    classifier = run_config.get("classifier")
    if not isinstance(classifier, dict):
        raise ValueError(
            f"{summary_path} missing intervention_run_config.classifier "
            f"for C={c_value:g}"
        )
    observed_hash = classifier.get("content_sha256")
    if not isinstance(observed_hash, str) or not observed_hash:
        raise ValueError(
            f"{summary_path} missing classifier content_sha256 for C={c_value:g}"
        )
    if observed_hash != expected_classifier_sha256:
        raise ValueError(
            f"{summary_path} classifier hash mismatch for C={c_value:g}: "
            f"expected {expected_classifier_sha256}, observed {observed_hash}"
        )


def extract_triviaqa_alpha0_accuracy(
    summary_path: Path,
    *,
    expected_classifier_sha256: str,
    c_value: float,
) -> dict[str, Any]:
    summary = load_json_object(summary_path)
    validate_triviaqa_classifier_hash(
        summary,
        summary_path=summary_path,
        expected_classifier_sha256=expected_classifier_sha256,
        c_value=c_value,
    )
    alpha_result = find_alpha_result(summary, alpha=0.0, path=summary_path)

    det_summary = alpha_result.get("deterministic_accuracy")
    if isinstance(det_summary, dict) and "estimate" in det_summary:
        accuracy = float(det_summary["estimate"])
        n_total = int(det_summary.get("n_total", alpha_result.get("n_total", 0)))
        n_correct = det_summary.get(
            "n_deterministic_correct",
            alpha_result.get("n_deterministic_correct"),
        )
    elif "deterministic_accuracy_rate" in alpha_result:
        accuracy = float(alpha_result["deterministic_accuracy_rate"])
        n_total = int(alpha_result.get("n_total", 0))
        n_correct = alpha_result.get("n_deterministic_correct")
    else:
        raise ValueError(
            f"{summary_path} missing alpha=0.0 deterministic TriviaQA accuracy"
        )

    if n_total <= 0:
        raise ValueError(f"zero-row TriviaQA output in {summary_path}")

    return {
        "triviaqa_summary_path": str(summary_path),
        "triviaqa_alpha0_deterministic_accuracy_rate": accuracy,
        "triviaqa_n_total": n_total,
        "triviaqa_n_deterministic_correct": None
        if n_correct is None
        else int(n_correct),
    }


def _candidate_label(candidate: dict[str, Any]) -> str:
    c_value = candidate.get("C", "<missing>")
    return f"C={c_value}"


def extract_dev_accuracy(candidate: dict[str, Any]) -> float:
    metrics = candidate.get("test_metrics")
    if not isinstance(metrics, dict) or "accuracy" not in metrics:
        raise ValueError(
            f"Candidate {_candidate_label(candidate)} missing held-out dev "
            "accuracy at test_metrics.accuracy"
        )
    return float(metrics["accuracy"])


def extract_selected_h_neurons(candidate: dict[str, Any]) -> int:
    if "selected_h_neurons" not in candidate:
        raise ValueError(
            f"Candidate {_candidate_label(candidate)} missing selected_h_neurons"
        )
    return int(candidate["selected_h_neurons"])


def extract_candidate_model_metadata(
    candidate: dict[str, Any],
    *,
    metrics_base_dir: Path,
) -> tuple[Path, str]:
    raw_path = candidate.get("candidate_model_path")
    expected_hash = candidate.get("candidate_model_sha256")
    if not isinstance(raw_path, str) or not isinstance(expected_hash, str):
        raise ValueError(
            f"Candidate {_candidate_label(candidate)} missing candidate model "
            "path/hash; rerun scripts/classifier.py with --candidate_models_dir"
        )
    model_path = resolve_recorded_path(raw_path, base_dir=metrics_base_dir)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Candidate {_candidate_label(candidate)} model file missing: {model_path}"
        )
    observed_hash = file_sha256(model_path)
    if observed_hash != expected_hash:
        raise ValueError(
            f"Candidate {_candidate_label(candidate)} model hash mismatch: "
            f"metrics={expected_hash}, observed={observed_hash}"
        )
    return model_path, expected_hash


def build_candidate_scores(
    classifier_metrics_path: Path,
    triviaqa_results: Mapping[float, Path],
) -> list[dict[str, Any]]:
    metrics = load_json_object(classifier_metrics_path)
    raw_candidates = metrics.get("candidates")
    if not isinstance(raw_candidates, list) or not raw_candidates:
        raise ValueError(f"{classifier_metrics_path} missing non-empty candidates list")

    rows: list[dict[str, Any]] = []
    for candidate in raw_candidates:
        if not isinstance(candidate, dict):
            raise ValueError(f"{classifier_metrics_path} candidates must be objects")
        if "C" not in candidate:
            raise ValueError(f"{classifier_metrics_path} candidate missing C")
        c_value = float(candidate["C"])
        dev_accuracy = extract_dev_accuracy(candidate)
        selected_h_neurons = extract_selected_h_neurons(candidate)
        model_path, model_hash = extract_candidate_model_metadata(
            candidate,
            metrics_base_dir=classifier_metrics_path.parent,
        )

        triviaqa_path = lookup_triviaqa_path(c_value, triviaqa_results)
        triviaqa_summary_path = resolve_triviaqa_summary_path(triviaqa_path)
        triviaqa = extract_triviaqa_alpha0_accuracy(
            triviaqa_summary_path,
            expected_classifier_sha256=model_hash,
            c_value=c_value,
        )
        triviaqa_accuracy = float(
            triviaqa["triviaqa_alpha0_deterministic_accuracy_rate"]
        )

        row = {
            "C": c_value,
            "heldout_dev_accuracy": dev_accuracy,
            "triviaqa_alpha0_deterministic_accuracy_rate": triviaqa_accuracy,
            "selection_score": dev_accuracy + triviaqa_accuracy,
            "selected_h_neurons": selected_h_neurons,
            "candidate_model_path": str(model_path),
            "candidate_model_sha256": model_hash,
            **triviaqa,
        }
        rows.append(row)
    return rows


def choose_winner(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("No candidates to select from")
    return max(
        rows,
        key=lambda row: (
            float(row["selection_score"]),
            float(row["triviaqa_alpha0_deterministic_accuracy_rate"]),
            float(row["heldout_dev_accuracy"]),
            -int(row["selected_h_neurons"]),
            -float(row["C"]),
        ),
    )


def select_intervention_aware_c(
    *,
    classifier_metrics_path: Path,
    triviaqa_results: Mapping[float, Path],
    output_summary_path: Path,
    selected_model_out: Path,
) -> dict[str, Any]:
    rows = build_candidate_scores(classifier_metrics_path, triviaqa_results)
    winner = choose_winner(rows)

    source_model = Path(str(winner["candidate_model_path"]))
    selected_model_out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_model, selected_model_out)
    selected_model_hash = file_sha256(selected_model_out)

    selected = {
        **winner,
        "selected_model_path": str(selected_model_out),
        "selected_model_sha256": selected_model_hash,
    }
    summary = {
        "schema_version": SCHEMA_VERSION,
        "classifier_metrics_path": str(classifier_metrics_path),
        "selection_score_formula": SELECTION_SCORE_FORMULA,
        "tie_breakers": TIE_BREAKERS,
        "selected": selected,
        "candidates": sorted(rows, key=lambda row: float(row["C"])),
    }

    output_summary_path.parent.mkdir(parents=True, exist_ok=True)
    with output_summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")

    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select C using held-out classifier accuracy plus TriviaQA suppression accuracy."
    )
    parser.add_argument("--classifier_metrics", type=Path, required=True)
    parser.add_argument(
        "--triviaqa_result",
        action="append",
        required=True,
        help="Per-C TriviaQA summary or output directory, formatted as C=PATH.",
    )
    parser.add_argument("--output_summary", type=Path, required=True)
    parser.add_argument("--selected_model_out", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    summary = select_intervention_aware_c(
        classifier_metrics_path=args.classifier_metrics,
        triviaqa_results=parse_triviaqa_result_specs(args.triviaqa_result),
        output_summary_path=args.output_summary,
        selected_model_out=args.selected_model_out,
    )
    selected = summary["selected"]
    print(
        "Selected intervention-aware C="
        f"{selected['C']:g} with score={selected['selection_score']:.6f}; "
        f"copied model to {selected['selected_model_path']}"
    )


if __name__ == "__main__":
    main()
