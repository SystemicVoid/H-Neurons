"""Validate Mistral 24B CP2/CP3 split, activation, and classifier artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

from utils import json_dumps


SCHEMA_VERSION = "mistral24b_cp23_validation/v1"
ACTIVATION_INPUT_SCHEMA_VERSION = "mistral24b_classifier_activation_inputs/v1"
DEFAULT_MODEL_KEY = "mistral_small_24b_instruct_2501"
DEFAULT_MODEL_PATH = "mistralai/Mistral-Small-24B-Instruct-2501"
DEFAULT_CLASSIFIER_PATH = "models/mistral24b_classifier_canonical.pkl"
METRIC_NAMES = ("accuracy", "f1", "auroc")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the CP2/CP3 Mistral 24B artifacts before using the saved "
            "classifier for downstream GPU stages."
        )
    )
    parser.add_argument(
        "--pipeline-dir",
        type=Path,
        default=Path("data/mistral24b/pipeline"),
        help="Directory containing split JSONs and classifier metrics.",
    )
    parser.add_argument(
        "--activation-root",
        type=Path,
        default=Path("data/mistral24b/pipeline/activations_llm_canonical"),
        help="Root directory containing train/dev/test activation summaries.",
    )
    parser.add_argument("--model-key", default=DEFAULT_MODEL_KEY)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--classifier-path", default=DEFAULT_CLASSIFIER_PATH)
    parser.add_argument("--train-per-class", type=int, default=360)
    parser.add_argument("--dev-per-class", type=int, default=100)
    parser.add_argument("--test-per-class", type=int, default=100)
    parser.add_argument("--layers", type=int, default=40)
    parser.add_argument("--ffn-neurons", type=int, default=32768)
    parser.add_argument("--total-ffn-neurons", type=int, default=1_310_720)
    parser.add_argument(
        "--activation-inputs-only",
        action="store_true",
        help=(
            "Validate the exact train/dev activation inputs consumed by "
            "scripts/classifier.py without requiring CP3 classifier metrics."
        ),
    )
    parser.add_argument(
        "--train-ids-path",
        type=Path,
        help="Train split JSON used by scripts/classifier.py.",
    )
    parser.add_argument(
        "--dev-ids-path",
        type=Path,
        help="Dev/test split JSON used by scripts/classifier.py.",
    )
    parser.add_argument(
        "--train-answer-dir",
        type=Path,
        help="Train answer-token activation directory.",
    )
    parser.add_argument(
        "--train-other-dir",
        type=Path,
        help="Train all-except-answer-token activation directory.",
    )
    parser.add_argument(
        "--dev-answer-dir",
        type=Path,
        help="Dev answer-token activation directory.",
    )
    args = parser.parse_args(argv)
    if args.activation_inputs_only:
        required = {
            "--train-ids-path": args.train_ids_path,
            "--dev-ids-path": args.dev_ids_path,
            "--train-answer-dir": args.train_answer_dir,
            "--train-other-dir": args.train_other_dir,
            "--dev-answer-dir": args.dev_answer_dir,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            parser.error(
                "--activation-inputs-only requires " + ", ".join(sorted(missing))
            )
    return args


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _json_exists(path: Path) -> bool:
    return path.is_file()


def _path_matches(value: Any, expected: str) -> bool:
    if not isinstance(value, str) or not value:
        return False
    normalized = Path(value).as_posix()
    expected_normalized = Path(expected).as_posix()
    return normalized == expected_normalized or normalized.endswith(
        f"/{expected_normalized}"
    )


def _provenance_paths(path: Path) -> list[Path]:
    return sorted(path.parent.glob(f"{path.name}.provenance.*.json"))


def _provenance_value(provenance_paths: list[Path], key: str) -> Any:
    for path in provenance_paths:
        payload = _read_json(path)
        args = payload.get("args")
        if isinstance(args, dict) and key in args:
            return args[key]
    return None


def _split_ids(payload: dict[str, Any], key: str) -> list[str]:
    value = payload.get(key)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"split key {key!r} must be a list of string ids")
    return value


def _selected_count(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, list):
        return len(value)
    if isinstance(value, dict):
        for key in ("count", "n", "selected_h_neurons"):
            count = value.get(key)
            if isinstance(count, int) and not isinstance(count, bool):
                return count
    return None


def _metric_estimate(metrics: dict[str, Any], name: str) -> float | None:
    value = metrics.get(name)
    if isinstance(value, dict):
        estimate = value.get("estimate")
        if isinstance(estimate, int | float) and not isinstance(estimate, bool):
            return float(estimate)
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    return None


def _evaluation_metric_checks(payload: dict[str, Any]) -> dict[str, bool]:
    evaluation = payload.get("evaluation")
    if not isinstance(evaluation, dict):
        return {f"evaluation_{name}": False for name in METRIC_NAMES}
    interval_metrics = evaluation.get("metrics")
    point_metrics = evaluation.get("point_metrics")
    if not isinstance(interval_metrics, dict):
        interval_metrics = {}
    if not isinstance(point_metrics, dict):
        point_metrics = {}
    checks: dict[str, bool] = {}
    for name in METRIC_NAMES:
        checks[f"evaluation_metrics_{name}_estimate"] = (
            _metric_estimate(interval_metrics, name) is not None
        )
        checks[f"evaluation_point_metrics_{name}"] = (
            _metric_estimate(point_metrics, name) is not None
        )
    return checks


def _best_dev_metric_checks(payload: dict[str, Any]) -> dict[str, bool]:
    best = payload.get("best")
    test_metrics = best.get("test_metrics") if isinstance(best, dict) else None
    if not isinstance(test_metrics, dict):
        test_metrics = {}
    return {
        f"best_test_metrics_{name}": _metric_estimate(test_metrics, name) is not None
        for name in METRIC_NAMES
    }


def _activation_count(summary: dict[str, Any], location: str) -> int | None:
    per_location = summary.get("per_location")
    if not isinstance(per_location, dict):
        return None
    location_summary = per_location.get(location)
    if not isinstance(location_summary, dict):
        return None
    value = location_summary.get("final_completed_count")
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return None


def _activation_missing_count(summary: dict[str, Any], location: str) -> int | None:
    per_location = summary.get("per_location")
    if not isinstance(per_location, dict):
        return None
    location_summary = per_location.get(location)
    if not isinstance(location_summary, dict):
        return None
    value = location_summary.get("missing_region_count")
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return None


def _sample_activation_path(
    activation_root: Path,
    split_name: str,
    location: str,
    split_payload: dict[str, Any],
) -> Path:
    for label in ("f", "t"):
        ids = _split_ids(split_payload, label)
        if ids:
            return activation_root / split_name / location / f"act_{ids[0]}.npy"
    raise ValueError(f"{split_name} split contains no ids")


def _validate_splits(
    pipeline_dir: Path,
    *,
    train_per_class: int,
    dev_per_class: int,
    test_per_class: int,
) -> tuple[dict[str, bool], dict[str, Any], dict[str, dict[str, Any]]]:
    expected = {
        "train": train_per_class,
        "dev": dev_per_class,
        "test": test_per_class,
    }
    payloads: dict[str, dict[str, Any]] = {}
    checks: dict[str, bool] = {}
    details: dict[str, Any] = {"split_counts": {}}
    all_split_sets: dict[str, set[str]] = {}

    for split_name, per_class in expected.items():
        payload = _read_json(pipeline_dir / f"{split_name}_qids_llm.json")
        payloads[split_name] = payload
        false_ids = _split_ids(payload, "f")
        true_ids = _split_ids(payload, "t")
        combined = [*false_ids, *true_ids]
        details["split_counts"][split_name] = {
            "f": len(false_ids),
            "t": len(true_ids),
            "total": len(combined),
        }
        checks[f"{split_name}_false_count"] = len(false_ids) == per_class
        checks[f"{split_name}_true_count"] = len(true_ids) == per_class
        checks[f"{split_name}_unique_ids"] = len(combined) == len(set(combined))
        all_split_sets[split_name] = set(combined)

    pairs = (("train", "dev"), ("train", "test"), ("dev", "test"))
    for left, right in pairs:
        overlap = all_split_sets[left] & all_split_sets[right]
        checks[f"{left}_{right}_disjoint"] = not overlap
        details[f"{left}_{right}_overlap_count"] = len(overlap)

    return checks, details, payloads


def _validate_activations(
    activation_root: Path,
    split_payloads: dict[str, dict[str, Any]],
    *,
    layers: int,
    ffn_neurons: int,
) -> tuple[dict[str, bool], dict[str, Any]]:
    split_totals = {
        split_name: len(_split_ids(payload, "f")) + len(_split_ids(payload, "t"))
        for split_name, payload in split_payloads.items()
    }
    expected_counts = {
        ("train", "answer_tokens"): split_totals["train"],
        ("train", "all_except_answer_tokens"): split_totals["train"],
        ("dev", "answer_tokens"): split_totals["dev"],
        ("test", "answer_tokens"): split_totals["test"],
    }
    checks: dict[str, bool] = {}
    details: dict[str, Any] = {"activation_counts": {}, "activation_samples": {}}
    expected_shape = (layers, ffn_neurons)
    expected_width = layers * ffn_neurons

    for split_name, location in expected_counts:
        summary = _read_json(activation_root / split_name / "summary.json")
        check_prefix = f"{split_name}_{location}"
        count = _activation_count(summary, location)
        missing = _activation_missing_count(summary, location)
        details["activation_counts"][check_prefix] = {
            "final_completed_count": count,
            "missing_region_count": missing,
        }
        checks[f"{check_prefix}_summary_completed"] = (
            summary.get("status") == "completed"
        )
        checks[f"{check_prefix}_summary_ready"] = (
            summary.get("triage_status") == "ready_for_downstream"
        )
        checks[f"{check_prefix}_count"] = (
            count == expected_counts[(split_name, location)]
        )
        checks[f"{check_prefix}_missing_zero"] = missing == 0

        sample_path = _sample_activation_path(
            activation_root, split_name, location, split_payloads[split_name]
        )
        sample_exists = sample_path.is_file()
        checks[f"{check_prefix}_sample_exists"] = sample_exists
        if sample_exists:
            sample = np.load(sample_path, mmap_mode="r")
            sample_shape = tuple(int(dim) for dim in sample.shape)
            sample_width = int(np.prod(sample_shape))
        else:
            sample_shape = ()
            sample_width = 0
        details["activation_samples"][check_prefix] = {
            "path": sample_path.as_posix(),
            "shape": list(sample_shape),
            "flattened_width": sample_width,
        }
        checks[f"{check_prefix}_sample_shape"] = sample_shape == expected_shape
        checks[f"{check_prefix}_flattened_width"] = sample_width == expected_width

    return checks, details


def _split_ids_for_classifier_input(
    payload: dict[str, Any],
) -> tuple[list[str], list[str], list[str]]:
    false_ids = _split_ids(payload, "f")
    true_ids = _split_ids(payload, "t")
    return false_ids, true_ids, [*false_ids, *true_ids]


def _validate_classifier_input_splits(
    *,
    train_ids_path: Path,
    dev_ids_path: Path,
    train_per_class: int,
    dev_per_class: int,
) -> tuple[dict[str, bool], dict[str, Any], dict[str, dict[str, Any]]]:
    checks: dict[str, bool] = {}
    details: dict[str, Any] = {"split_counts": {}}
    payloads: dict[str, dict[str, Any]] = {}
    split_specs = {
        "train": (train_ids_path, train_per_class),
        "dev": (dev_ids_path, dev_per_class),
    }
    split_sets: dict[str, set[str]] = {}

    for split_name, (path, per_class) in split_specs.items():
        payload = _read_json(path)
        payloads[split_name] = payload
        false_ids, true_ids, combined = _split_ids_for_classifier_input(payload)
        details["split_counts"][split_name] = {
            "f": len(false_ids),
            "t": len(true_ids),
            "total": len(combined),
            "path": path.as_posix(),
        }
        checks[f"{split_name}_false_count"] = len(false_ids) == per_class
        checks[f"{split_name}_true_count"] = len(true_ids) == per_class
        checks[f"{split_name}_unique_ids"] = len(combined) == len(set(combined))
        split_sets[split_name] = set(combined)

    overlap = split_sets["train"] & split_sets["dev"]
    checks["train_dev_disjoint"] = not overlap
    details["train_dev_overlap_count"] = len(overlap)
    return checks, details, payloads


def _activation_location_file_checks(
    *,
    split_name: str,
    location: str,
    activation_dir: Path,
    expected_ids: list[str],
    expected_shape: tuple[int, int],
) -> tuple[dict[str, bool], dict[str, Any]]:
    check_prefix = f"{split_name}_{location}"
    checks: dict[str, bool] = {}
    details: dict[str, Any] = {}
    split_root = activation_dir.parent
    summary_path = split_root / "summary.json"
    metadata_path = split_root / "metadata.json"

    checks[f"{check_prefix}_dir_exists"] = activation_dir.is_dir()
    checks[f"{check_prefix}_metadata_exists"] = metadata_path.is_file()
    checks[f"{check_prefix}_summary_exists"] = summary_path.is_file()

    summary = _read_json(summary_path) if summary_path.is_file() else {}
    count = _activation_count(summary, location)
    missing = _activation_missing_count(summary, location)
    checks[f"{check_prefix}_summary_completed"] = summary.get("status") == "completed"
    checks[f"{check_prefix}_summary_ready"] = (
        summary.get("triage_status") == "ready_for_downstream"
    )
    checks[f"{check_prefix}_summary_count"] = count == len(expected_ids)
    checks[f"{check_prefix}_summary_missing_zero"] = missing == 0

    existing_ids: set[str] = set()
    if activation_dir.is_dir():
        existing_ids = {
            path.stem.removeprefix("act_") for path in activation_dir.glob("act_*.npy")
        }
    expected_set = set(expected_ids)
    missing_ids = sorted(expected_set - existing_ids)
    unexpected_ids = sorted(existing_ids - expected_set)
    present_expected_count = len(expected_set & existing_ids)

    shape_errors: list[str] = []
    for qid in expected_ids:
        path = activation_dir / f"act_{qid}.npy"
        if not path.is_file():
            continue
        try:
            activation = np.load(path, mmap_mode="r")
            shape = tuple(int(dim) for dim in activation.shape)
        except Exception as exc:  # noqa: BLE001
            shape_errors.append(f"{path}: could not load header: {exc}")
            continue
        if shape != expected_shape:
            shape_errors.append(
                f"{path}: shape {list(shape)} != expected {list(expected_shape)}"
            )

    checks[f"{check_prefix}_all_expected_files_present"] = not missing_ids
    checks[f"{check_prefix}_expected_file_count"] = present_expected_count == len(
        expected_ids
    )
    checks[f"{check_prefix}_all_shapes"] = not shape_errors
    details.update(
        {
            "dir": activation_dir.as_posix(),
            "metadata_path": metadata_path.as_posix(),
            "summary_path": summary_path.as_posix(),
            "expected_count": len(expected_ids),
            "present_expected_count": present_expected_count,
            "actual_act_file_count": len(existing_ids),
            "missing_ids_preview": missing_ids[:10],
            "missing_id_count": len(missing_ids),
            "unexpected_ids_preview": unexpected_ids[:10],
            "unexpected_id_count": len(unexpected_ids),
            "summary_final_completed_count": count,
            "summary_missing_region_count": missing,
            "shape_errors_preview": shape_errors[:10],
            "shape_error_count": len(shape_errors),
        }
    )
    return checks, details


def validate_classifier_activation_inputs(
    *,
    train_ids_path: Path,
    dev_ids_path: Path,
    train_answer_dir: Path,
    train_other_dir: Path,
    dev_answer_dir: Path,
    train_per_class: int = 360,
    dev_per_class: int = 100,
    layers: int = 40,
    ffn_neurons: int = 32768,
) -> dict[str, Any]:
    checks: dict[str, bool] = {}
    details: dict[str, Any] = {"activation_inputs": {}}

    split_checks, split_details, split_payloads = _validate_classifier_input_splits(
        train_ids_path=train_ids_path,
        dev_ids_path=dev_ids_path,
        train_per_class=train_per_class,
        dev_per_class=dev_per_class,
    )
    checks.update(split_checks)
    details.update(split_details)

    _, _, train_ids = _split_ids_for_classifier_input(split_payloads["train"])
    _, _, dev_ids = _split_ids_for_classifier_input(split_payloads["dev"])
    expected_shape = (layers, ffn_neurons)
    locations = (
        ("train", "answer_tokens", train_answer_dir, train_ids),
        ("train", "all_except_answer_tokens", train_other_dir, train_ids),
        ("dev", "answer_tokens", dev_answer_dir, dev_ids),
    )
    for split_name, location, activation_dir, expected_ids in locations:
        location_checks, location_details = _activation_location_file_checks(
            split_name=split_name,
            location=location,
            activation_dir=activation_dir,
            expected_ids=expected_ids,
            expected_shape=expected_shape,
        )
        checks.update(location_checks)
        details["activation_inputs"][f"{split_name}_{location}"] = location_details

    errors = _collect_errors(checks)
    accepted = not errors
    return {
        "schema_version": ACTIVATION_INPUT_SCHEMA_VERSION,
        "status": "ok" if accepted else "needs_review",
        "accepted": accepted,
        "checks": checks,
        "details": details,
        "errors": errors,
    }


def _validate_metrics(
    pipeline_dir: Path,
    *,
    model_key: str,
    model_path: str,
    classifier_path: str,
    total_ffn_neurons: int,
) -> tuple[dict[str, bool], dict[str, Any]]:
    checks: dict[str, bool] = {}
    details: dict[str, Any] = {"classifier_metrics": {}}
    metric_files = {
        "dev": pipeline_dir / "classifier_canonical_dev_metrics.json",
        "test": pipeline_dir / "classifier_canonical_test_metrics.json",
    }
    payloads: dict[str, dict[str, Any]] = {}
    provenances: dict[str, list[Path]] = {}

    for split_name, path in metric_files.items():
        checks[f"{split_name}_metrics_exists"] = _json_exists(path)
        payload = _read_json(path) if path.is_file() else {}
        payloads[split_name] = payload
        provenance_paths = _provenance_paths(path)
        provenances[split_name] = provenance_paths
        checks[f"{split_name}_metrics_provenance_exists"] = bool(provenance_paths)
        checks[f"{split_name}_model_key"] = payload.get("model_key") == model_key
        checks[f"{split_name}_model_path"] = payload.get("model_path") == model_path
        checks[f"{split_name}_total_ffn_neurons"] = (
            payload.get("total_ffn_neurons") == total_ffn_neurons
        )
        checks.update(
            {
                f"{split_name}_{key}": passed
                for key, passed in _evaluation_metric_checks(payload).items()
            }
        )

    dev_payload = payloads["dev"]
    test_payload = payloads["test"]
    dev_best = dev_payload.get("best")
    dev_selected = (
        _selected_count(dev_best.get("selected_h_neurons"))
        if isinstance(dev_best, dict)
        else None
    )
    test_selected = _selected_count(test_payload.get("selected_h_neurons"))
    checks["dev_best_selected_h_neurons_count"] = dev_selected is not None
    checks["test_selected_h_neurons_count"] = test_selected is not None
    checks.update(
        {
            f"dev_{key}": passed
            for key, passed in _best_dev_metric_checks(dev_payload).items()
        }
    )

    dev_saved_model_path = dev_payload.get("saved_model_path") or _provenance_value(
        provenances["dev"], "save_model"
    )
    checks["dev_selected_model_path"] = _path_matches(
        dev_saved_model_path, classifier_path
    )
    checks["test_loaded_model_path"] = _path_matches(
        test_payload.get("loaded_model_path"), classifier_path
    )

    details["classifier_metrics"] = {
        "dev_selected_h_neurons": dev_selected,
        "test_selected_h_neurons": test_selected,
        "dev_model_path_source": dev_saved_model_path,
        "test_loaded_model_path": test_payload.get("loaded_model_path"),
        "dev_provenance_count": len(provenances["dev"]),
        "test_provenance_count": len(provenances["test"]),
    }
    return checks, details


def _collect_errors(checks: dict[str, bool]) -> list[str]:
    return [f"{name} failed" for name, passed in sorted(checks.items()) if not passed]


def validate_cp23_artifacts(
    *,
    pipeline_dir: Path,
    activation_root: Path,
    model_key: str = DEFAULT_MODEL_KEY,
    model_path: str = DEFAULT_MODEL_PATH,
    classifier_path: str = DEFAULT_CLASSIFIER_PATH,
    train_per_class: int = 360,
    dev_per_class: int = 100,
    test_per_class: int = 100,
    layers: int = 40,
    ffn_neurons: int = 32768,
    total_ffn_neurons: int = 1_310_720,
) -> dict[str, Any]:
    checks: dict[str, bool] = {}
    details: dict[str, Any] = {}

    split_checks, split_details, split_payloads = _validate_splits(
        pipeline_dir,
        train_per_class=train_per_class,
        dev_per_class=dev_per_class,
        test_per_class=test_per_class,
    )
    checks.update(split_checks)
    details.update(split_details)

    activation_checks, activation_details = _validate_activations(
        activation_root,
        split_payloads,
        layers=layers,
        ffn_neurons=ffn_neurons,
    )
    checks.update(activation_checks)
    details.update(activation_details)

    metric_checks, metric_details = _validate_metrics(
        pipeline_dir,
        model_key=model_key,
        model_path=model_path,
        classifier_path=classifier_path,
        total_ffn_neurons=total_ffn_neurons,
    )
    checks.update(metric_checks)
    details.update(metric_details)

    errors = _collect_errors(checks)
    accepted = not errors
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "ok" if accepted else "needs_review",
        "accepted": accepted,
        "checks": checks,
        "details": details,
        "errors": errors,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.activation_inputs_only:
        result = validate_classifier_activation_inputs(
            train_ids_path=args.train_ids_path,
            dev_ids_path=args.dev_ids_path,
            train_answer_dir=args.train_answer_dir,
            train_other_dir=args.train_other_dir,
            dev_answer_dir=args.dev_answer_dir,
            train_per_class=args.train_per_class,
            dev_per_class=args.dev_per_class,
            layers=args.layers,
            ffn_neurons=args.ffn_neurons,
        )
    else:
        result = validate_cp23_artifacts(
            pipeline_dir=args.pipeline_dir,
            activation_root=args.activation_root,
            model_key=args.model_key,
            model_path=args.model_path,
            classifier_path=args.classifier_path,
            train_per_class=args.train_per_class,
            dev_per_class=args.dev_per_class,
            test_per_class=args.test_per_class,
            layers=args.layers,
            ffn_neurons=args.ffn_neurons,
            total_ffn_neurons=args.total_ffn_neurons,
        )
    print(json_dumps(result))
    return 0 if result["accepted"] else 1


if __name__ == "__main__":
    sys.exit(main())
