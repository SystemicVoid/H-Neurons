#!/usr/bin/env python3
"""Build the current-state D7 full-500 summary."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np

from evaluate_csv2 import normalize_csv2_payload
from report_d7_csv2 import _paired_bootstrap_mean_delta, _rows_by_id
from uncertainty import (
    DEFAULT_BOOTSTRAP_RESAMPLES,
    DEFAULT_BOOTSTRAP_SEED,
    paired_bootstrap_binary_rate_difference,
    percentile_interval,
    wilson_interval,
)

RUN_ROOT = Path("data/gemma3_4b/intervention/jailbreak_d7/full500_canonical")


@dataclass(frozen=True)
class ConditionSpec:
    key: str
    label: str
    alpha: float
    experiment_path: Path
    csv2_path: Path


BASE_CURRENT_CONDITIONS = [
    ConditionSpec(
        key="baseline",
        label="baseline_noop",
        alpha=1.0,
        experiment_path=RUN_ROOT / "baseline_noop/experiment/alpha_1.0.jsonl",
        csv2_path=RUN_ROOT / "baseline_noop/csv2_v3_evaluation/alpha_1.0.jsonl",
    ),
    ConditionSpec(
        key="l1",
        label="l1_neuron",
        alpha=3.0,
        experiment_path=RUN_ROOT / "l1_neuron/experiment/alpha_3.0.jsonl",
        csv2_path=RUN_ROOT / "l1_neuron/csv2_v3_evaluation/alpha_3.0.jsonl",
    ),
    ConditionSpec(
        key="causal",
        label="causal_locked",
        alpha=4.0,
        experiment_path=RUN_ROOT / "causal_locked/experiment/alpha_4.0.jsonl",
        csv2_path=RUN_ROOT / "causal_locked/csv2_v3_evaluation/alpha_4.0.jsonl",
    ),
    ConditionSpec(
        key="random_layer_seed1",
        label="causal_random_head_layer_matched/seed_1",
        alpha=4.0,
        experiment_path=(
            RUN_ROOT
            / "causal_random_head_layer_matched/seed_1/experiment/alpha_4.0.jsonl"
        ),
        csv2_path=(
            RUN_ROOT
            / "causal_random_head_layer_matched/seed_1/csv2_evaluation/alpha_4.0.jsonl"
        ),
    ),
    ConditionSpec(
        key="probe",
        label="probe_locked",
        alpha=1.0,
        experiment_path=RUN_ROOT / "probe_locked/experiment/alpha_1.0.jsonl",
        csv2_path=RUN_ROOT / "probe_locked/csv2_evaluation/alpha_1.0.jsonl",
    ),
]

OPTIONAL_CURRENT_CONDITIONS = [
    ConditionSpec(
        key="random_layer_seed2",
        label="causal_random_head_layer_matched/seed_2",
        alpha=4.0,
        experiment_path=(
            RUN_ROOT
            / "causal_random_head_layer_matched/seed_2/experiment/alpha_4.0.jsonl"
        ),
        csv2_path=(
            RUN_ROOT
            / "causal_random_head_layer_matched/seed_2/csv2_evaluation/alpha_4.0.jsonl"
        ),
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output_path",
        type=Path,
        default=RUN_ROOT / "d7_full500_current_state_summary.json",
    )
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _rate_summary(values: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(values, dtype=bool)
    n = int(arr.size)
    n_yes = int(arr.sum())
    return {
        "estimate": n_yes / n if n else 0.0,
        "estimate_pct": 100.0 * n_yes / n if n else 0.0,
        "n_yes": n_yes,
        "n": n,
        "ci": wilson_interval(n_yes, n).to_dict() if n else None,
    }


def _bootstrap_mean_summary(values: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError("mean summary expects a one-dimensional array")
    n = len(arr)
    if n == 0:
        return {
            "estimate": 0.0,
            "ci": percentile_interval(
                np.array([0.0]),
                method="bootstrap_percentile_mean",
            ).to_dict(),
            "bootstrap": {
                "n_resamples": int(DEFAULT_BOOTSTRAP_RESAMPLES),
                "seed": int(DEFAULT_BOOTSTRAP_SEED),
                "resampling": "iid_rows",
                "interval": "percentile",
            },
        }

    rng = np.random.default_rng(DEFAULT_BOOTSTRAP_SEED)
    samples = np.empty(DEFAULT_BOOTSTRAP_RESAMPLES, dtype=np.float64)
    for sample_idx in range(DEFAULT_BOOTSTRAP_RESAMPLES):
        indices = rng.choice(n, size=n, replace=True)
        samples[sample_idx] = float(arr[indices].mean())

    return {
        "estimate": float(arr.mean()),
        "ci": percentile_interval(
            samples,
            method="bootstrap_percentile_mean",
        ).to_dict(),
        "bootstrap": {
            "n_resamples": int(DEFAULT_BOOTSTRAP_RESAMPLES),
            "seed": int(DEFAULT_BOOTSTRAP_SEED),
            "resampling": "iid_rows",
            "interval": "percentile",
        },
    }


def _paired_binary_summary(
    baseline: np.ndarray,
    comparison: np.ndarray,
) -> dict[str, Any]:
    if baseline.shape != comparison.shape:
        raise ValueError("paired binary summaries require matching shapes")
    summary = paired_bootstrap_binary_rate_difference(
        baseline.tolist(),
        comparison.tolist(),
    )
    baseline_true = baseline.astype(bool)
    comparison_true = comparison.astype(bool)
    harmful_to_not = int((baseline_true & ~comparison_true).sum())
    not_to_harmful = int((~baseline_true & comparison_true).sum())
    return {
        **summary,
        "n": int(len(baseline_true)),
        "transitions": {
            "harmful_to_not_harmful": harmful_to_not,
            "not_harmful_to_harmful": not_to_harmful,
        },
    }


def _schema_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        schema_version = row.get("csv2", {}).get("schema_version")
        key = (
            "legacy_unversioned"
            if schema_version in (None, "")
            else str(schema_version)
        )
        counts[key] += 1
    return dict(counts)


def _condition_status(error_count: int) -> str:
    if error_count:
        return "complete_scored_error_bearing"
    return "complete_scored"


def _condition_has_complete_csv2_annotations(spec: ConditionSpec) -> bool:
    """Return True when the optional condition is fully ready for summarying.

    `evaluate_csv2.py` creates the output JSONL by copying the experiment rows
    before any `csv2` payloads have been written. Treat that intermediate file
    as unavailable so live summary/export regeneration stays stable.
    """
    if not spec.experiment_path.exists() or not spec.csv2_path.exists():
        return False

    try:
        experiment_rows = _rows_by_id(
            _load_jsonl(spec.experiment_path),
            context=f"{spec.key} experiment availability",
        )
        csv2_rows = _rows_by_id(
            _load_jsonl(spec.csv2_path),
            context=f"{spec.key} csv2 availability",
        )
    except (json.JSONDecodeError, OSError, KeyError, TypeError, ValueError):
        return False

    if set(experiment_rows) != set(csv2_rows):
        return False

    return all(isinstance(row.get("csv2"), dict) for row in csv2_rows.values())


def _available_current_conditions() -> list[ConditionSpec]:
    available = list(BASE_CURRENT_CONDITIONS)
    for spec in OPTIONAL_CURRENT_CONDITIONS:
        if _condition_has_complete_csv2_annotations(spec):
            available.append(spec)
    return available


def _collect_condition_data(
    spec: ConditionSpec,
) -> tuple[dict[str, Any], dict[str, Any]]:
    experiment_rows = _rows_by_id(
        _load_jsonl(spec.experiment_path),
        context=f"{spec.key} experiment",
    )
    csv2_raw_rows = _load_jsonl(spec.csv2_path)
    csv2_rows = _rows_by_id(csv2_raw_rows, context=f"{spec.key} csv2")

    exp_ids = set(experiment_rows)
    csv2_ids = set(csv2_rows)
    if exp_ids != csv2_ids:
        raise ValueError(
            f"{spec.key}: experiment/csv2 prompt-ID parity failed "
            f"(missing_in_csv2={len(exp_ids - csv2_ids)}, "
            f"missing_in_experiment={len(csv2_ids - exp_ids)})"
        )

    ordered_ids = sorted(exp_ids)
    strict_harm = []
    binary_harm = []
    token_cap = []
    c_values = []
    s_values = []
    payload_values = []
    primary_outcomes: Counter[str] = Counter()
    raw_harmful_binary: Counter[str] = Counter()
    normalized_harmful_binary: Counter[str] = Counter()
    error_types: Counter[str] = Counter()
    error_rows: list[str] = []
    borderline_to: Counter[str] = Counter()

    for sample_id in ordered_ids:
        exp_row = experiment_rows[sample_id]
        raw_csv2 = csv2_rows[sample_id]["csv2"]
        normalized = normalize_csv2_payload(raw_csv2)

        raw_label = raw_csv2.get("harmful_binary")
        raw_harmful_binary["null" if raw_label is None else str(raw_label)] += 1
        normalized_harmful_binary[str(normalized.get("harmful_binary"))] += 1
        if raw_label == "borderline":
            borderline_to[str(normalized.get("harmful_binary"))] += 1

        strict_harm.append(normalized.get("harmful_binary") == "yes")
        binary_harm.append(str(exp_row.get("judge", "")).upper() == "HARMFUL")
        token_cap.append(bool(exp_row.get("timings", {}).get("hit_token_cap", False)))
        c_values.append(float(normalized.get("C", 0)))
        s_values.append(float(normalized.get("S", 0)))
        payload_values.append(float(normalized.get("harmful_payload_share") or 0.0))
        primary_outcomes[str(normalized.get("primary_outcome"))] += 1

        if normalized.get("error"):
            error_types[str(normalized["error"])] += 1
            error_rows.append(sample_id)

    strict_arr = np.asarray(strict_harm, dtype=bool)
    binary_arr = np.asarray(binary_harm, dtype=bool)
    token_cap_arr = np.asarray(token_cap, dtype=bool)
    c_arr = np.asarray(c_values, dtype=np.float64)
    s_arr = np.asarray(s_values, dtype=np.float64)
    payload_arr = np.asarray(payload_values, dtype=np.float64)
    clean_mask = np.array(
        [sample_id not in error_rows for sample_id in ordered_ids],
        dtype=bool,
    )

    metadata = {
        "status": _condition_status(len(error_rows)),
        "label": spec.label,
        "alpha": spec.alpha,
        "experiment_path": str(spec.experiment_path),
        "csv2_path": str(spec.csv2_path),
        "experiment_row_count": len(experiment_rows),
        "csv2_row_count": len(csv2_rows),
        "prompt_id_parity_with_experiment": True,
        "csv2_schema_versions": _schema_counts(csv2_raw_rows),
        "raw_harmful_binary_counts": dict(raw_harmful_binary),
        "normalized_harmful_binary_counts": dict(normalized_harmful_binary),
        "borderline_reclassification": dict(borderline_to),
        "primary_outcome_counts": dict(primary_outcomes),
        "csv2_errors": {
            "count": len(error_rows),
            "types": dict(error_types),
            "sample_ids": error_rows[:12],
        },
        "strict_harmfulness_normalized": _rate_summary(strict_arr),
        "strict_harmfulness_normalized_clean_rows": _rate_summary(
            strict_arr[clean_mask]
        ),
        "binary_harmful": _rate_summary(binary_arr),
        "token_cap": {
            "count": int(token_cap_arr.sum()),
            "n": int(token_cap_arr.size),
            "share": float(token_cap_arr.mean()),
            "share_pct": float(token_cap_arr.mean() * 100.0),
        },
        "C": _bootstrap_mean_summary(c_arr),
        "S": _bootstrap_mean_summary(s_arr),
        "harmful_payload_share": _bootstrap_mean_summary(payload_arr),
    }

    arrays = {
        "ordered_ids": ordered_ids,
        "strict_harmfulness_normalized": strict_arr,
        "binary_harmful": binary_arr,
        "token_cap": token_cap_arr,
        "C": c_arr,
        "S": s_arr,
        "harmful_payload_share": payload_arr,
        "clean_mask": clean_mask,
    }
    return metadata, arrays


def _make_direct_comparison(
    left_arrays: dict[str, Any],
    right_arrays: dict[str, Any],
) -> dict[str, Any]:
    clean_mask = left_arrays["clean_mask"] & right_arrays["clean_mask"]
    return {
        "strict_harmfulness_normalized": _paired_binary_summary(
            left_arrays["strict_harmfulness_normalized"],
            right_arrays["strict_harmfulness_normalized"],
        ),
        "strict_harmfulness_normalized_clean_rows": _paired_binary_summary(
            left_arrays["strict_harmfulness_normalized"][clean_mask],
            right_arrays["strict_harmfulness_normalized"][clean_mask],
        ),
        "binary_harmful": _paired_binary_summary(
            left_arrays["binary_harmful"],
            right_arrays["binary_harmful"],
        ),
        "harmful_payload_share": _paired_bootstrap_mean_delta(
            left_arrays["harmful_payload_share"],
            right_arrays["harmful_payload_share"],
        ),
        "C": _paired_bootstrap_mean_delta(
            left_arrays["C"],
            right_arrays["C"],
        ),
        "S": _paired_bootstrap_mean_delta(
            left_arrays["S"],
            right_arrays["S"],
        ),
    }


def _random_layer_family_summary(
    condition_metadata: dict[str, Any],
) -> dict[str, Any] | None:
    seed_keys = [
        key
        for key in ("random_layer_seed1", "random_layer_seed2")
        if key in condition_metadata
    ]
    if not seed_keys:
        return None

    strict_by_seed = {
        key: condition_metadata[key]["strict_harmfulness_normalized"]["estimate_pct"]
        for key in seed_keys
    }
    binary_by_seed = {
        key: condition_metadata[key]["binary_harmful"]["estimate_pct"]
        for key in seed_keys
    }
    payload_by_seed = {
        key: condition_metadata[key]["harmful_payload_share"]["estimate"]
        for key in seed_keys
    }
    return {
        "status": (
            "available_multi_seed"
            if len(seed_keys) >= 2
            else "available_single_seed_only"
        ),
        "n_seeds": len(seed_keys),
        "seed_keys": seed_keys,
        "strict_harmfulness_normalized_estimate_pct_by_seed": strict_by_seed,
        "binary_harmful_estimate_pct_by_seed": binary_by_seed,
        "harmful_payload_share_estimate_by_seed": payload_by_seed,
        "strict_harmfulness_normalized_mean_estimate_pct": float(
            np.mean(list(strict_by_seed.values()))
        ),
        "binary_harmful_mean_estimate_pct": float(
            np.mean(list(binary_by_seed.values()))
        ),
        "harmful_payload_share_mean_estimate": float(
            np.mean(list(payload_by_seed.values()))
        ),
    }


def _current_panel_summary() -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    condition_metadata: dict[str, Any] = {}
    arrays_by_condition: dict[str, dict[str, Any]] = {}
    current_conditions = _available_current_conditions()
    for spec in current_conditions:
        metadata, arrays = _collect_condition_data(spec)
        condition_metadata[spec.key] = metadata
        arrays_by_condition[spec.key] = arrays

    baseline_arrays = arrays_by_condition["baseline"]
    ordered_ids = baseline_arrays["ordered_ids"]
    for key, arrays in arrays_by_condition.items():
        if arrays["ordered_ids"] != ordered_ids:
            raise ValueError(f"{key}: prompt order mismatch in current panel")

    paired_vs_baseline: dict[str, Any] = {}
    for key in condition_metadata:
        if key == "baseline":
            continue
        comparison_arrays = arrays_by_condition[key]
        clean_mask = baseline_arrays["clean_mask"] & comparison_arrays["clean_mask"]
        paired_vs_baseline[key] = {
            "strict_harmfulness_normalized": _paired_binary_summary(
                baseline_arrays["strict_harmfulness_normalized"],
                comparison_arrays["strict_harmfulness_normalized"],
            ),
            "strict_harmfulness_normalized_clean_rows": _paired_binary_summary(
                baseline_arrays["strict_harmfulness_normalized"][clean_mask],
                comparison_arrays["strict_harmfulness_normalized"][clean_mask],
            ),
            "binary_harmful": _paired_binary_summary(
                baseline_arrays["binary_harmful"],
                comparison_arrays["binary_harmful"],
            ),
            "harmful_payload_share": _paired_bootstrap_mean_delta(
                baseline_arrays["harmful_payload_share"],
                comparison_arrays["harmful_payload_share"],
            ),
            "C": _paired_bootstrap_mean_delta(
                baseline_arrays["C"],
                comparison_arrays["C"],
            ),
            "S": _paired_bootstrap_mean_delta(
                baseline_arrays["S"],
                comparison_arrays["S"],
            ),
        }

    direct_comparisons: dict[str, Any] = {
        "random_layer_seed1_vs_causal": _make_direct_comparison(
            arrays_by_condition["causal"],
            arrays_by_condition["random_layer_seed1"],
        ),
        "probe_vs_causal": _make_direct_comparison(
            arrays_by_condition["causal"],
            arrays_by_condition["probe"],
        ),
        "probe_vs_random_layer_seed1": _make_direct_comparison(
            arrays_by_condition["random_layer_seed1"],
            arrays_by_condition["probe"],
        ),
    }
    if "random_layer_seed2" in arrays_by_condition:
        direct_comparisons["random_layer_seed2_vs_causal"] = _make_direct_comparison(
            arrays_by_condition["causal"],
            arrays_by_condition["random_layer_seed2"],
        )
        direct_comparisons["probe_vs_random_layer_seed2"] = _make_direct_comparison(
            arrays_by_condition["random_layer_seed2"],
            arrays_by_condition["probe"],
        )
        direct_comparisons["random_layer_seed2_vs_random_layer_seed1"] = (
            _make_direct_comparison(
                arrays_by_condition["random_layer_seed1"],
                arrays_by_condition["random_layer_seed2"],
            )
        )

    return {
        "description": (
            "Current D7 current-state panel built from the stored full-500 artifacts "
            "using CSV2 v3-native outputs for all six compared conditions. The "
            "baseline, L1, and causal branches were rescored from raw generations "
            "into fresh v3 outputs, and the probe/random branches were repaired so "
            "the live panel no longer depends on mixed-ruler reconciliation. After "
            "two repair passes, a small documented residual set of explicit CSV2 "
            "errors remains; the main live caveat is still causal token-cap and "
            "quality debt rather than ruler debt."
        ),
        "conditions": condition_metadata,
        "random_layer_matched_family": _random_layer_family_summary(condition_metadata),
        "paired_vs_baseline": paired_vs_baseline,
        "direct_comparisons": direct_comparisons,
    }, arrays_by_condition


def _historical_panel_summary() -> dict[str, Any]:
    report_json = _load_json(RUN_ROOT / "d7_csv2_report.json")
    conditions = {
        condition["name"]: condition for condition in report_json["conditions"]
    }
    return {
        "description": (
            "Historical April 8 legacy-ruler panel. This is the stored structured "
            "summary used by the April 8 report and remains provenance only."
        ),
        "source_files": [
            str(RUN_ROOT / "d7_csv2_report.json"),
            "notes/act3-reports/2026-04-08-d7-full500-audit.md",
        ],
        "baseline": report_json["baseline"],
        "conditions": conditions,
        "paired_vs_baseline": report_json["paired_vs_baseline"],
    }


def _ruler_drift_summary(
    historical_panel: dict[str, Any],
    current_panel: dict[str, Any],
) -> dict[str, Any]:
    drift: dict[str, Any] = {}
    for key in ("baseline", "l1", "causal"):
        historical_condition = historical_panel["conditions"][key]
        current_condition = current_panel["conditions"][key]
        drift[key] = {
            "historical_csv2_yes_rate_pct": historical_condition["csv2_yes"]["estimate"]
            * 100.0,
            "current_normalized_rate_pct": current_condition[
                "strict_harmfulness_normalized"
            ]["estimate_pct"],
            "delta_pct_points": current_condition["strict_harmfulness_normalized"][
                "estimate_pct"
            ]
            - historical_condition["csv2_yes"]["estimate"] * 100.0,
            "raw_harmful_binary_counts": current_condition["raw_harmful_binary_counts"],
            "normalized_harmful_binary_counts": current_condition[
                "normalized_harmful_binary_counts"
            ],
            "borderline_reclassification": current_condition[
                "borderline_reclassification"
            ],
        }
    return drift


def _artifact_status(current_panel: dict[str, Any]) -> dict[str, Any]:
    random_seed2 = current_panel["conditions"].get("random_layer_seed2")
    return {
        "baseline_noop": current_panel["conditions"]["baseline"],
        "l1_neuron": current_panel["conditions"]["l1"],
        "causal_locked": current_panel["conditions"]["causal"],
        "causal_random_head_layer_matched": {
            "seed_1": current_panel["conditions"]["random_layer_seed1"],
            "seed_2": random_seed2
            if random_seed2 is not None
            else {
                "status": "absent",
                "label": "causal_random_head_layer_matched/seed_2",
                "alpha": 4.0,
                "experiment_path": None,
                "csv2_path": None,
                "experiment_row_count": 0,
                "csv2_row_count": 0,
            },
        },
        "probe_locked": current_panel["conditions"]["probe"],
    }


def build_summary() -> dict[str, Any]:
    historical_panel = _historical_panel_summary()
    current_panel, _ = _current_panel_summary()
    current_condition_sources = [
        str(spec.experiment_path) for spec in _available_current_conditions()
    ] + [str(spec.csv2_path) for spec in _available_current_conditions()]
    return {
        "schema_version": 1,
        "generated_at": date.today().isoformat(),
        "generated_by": "scripts/build_d7_control_and_ruler_summary.py",
        "benchmark": "jailbreak_d7_full500",
        "model": "google/gemma-3-4b-it",
        "summary_focus": "current_state_canonical",
        "data_root": str(RUN_ROOT),
        "source_files": [
            str(RUN_ROOT / "d7_csv2_report.json"),
            *current_condition_sources,
            "notes/act3-reports/2026-04-08-d7-full500-audit.md",
            "notes/act3-reports/2026-04-14-d7-control-and-ruler-audit.md",
            "notes/act3-reports/2026-04-16-d7-full500-two-seed-current-state-audit.md",
        ],
        "artifact_status": _artifact_status(current_panel),
        "historical_panel": historical_panel,
        "current_panel": current_panel,
        "ruler_drift": _ruler_drift_summary(historical_panel, current_panel),
    }


def main() -> None:
    args = parse_args()
    summary = build_summary()
    args.output_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Saved D7 current-state summary to {args.output_path}")


if __name__ == "__main__":
    main()
