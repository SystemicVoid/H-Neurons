#!/usr/bin/env python3
"""Analyze SIMID run outputs with paired item-level estimates."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np

from run_simid import OPEN_GRADING_METHOD, assert_noop_equivalence, load_jsonl
from uncertainty import (
    DEFAULT_BOOTSTRAP_RESAMPLES,
    DEFAULT_BOOTSTRAP_SEED,
    paired_bootstrap_continuous_mean_difference_raw,
    percentile_interval,
)
from utils import (
    finish_run_provenance,
    provenance_error_message,
    provenance_status_for_exception,
    start_run_provenance,
)


MetricFn = Callable[[dict[str, Any]], float | bool]
RowGroup = list[dict[str, Any]]
Panel = dict[float, dict[str, RowGroup]]
StratumKey = tuple[str, str, tuple[tuple[str, str], ...]]
PRIMARY_MC_RATE_METRIC = "mc_letter_likelihood_correct"
PRIMARY_MC_MARGIN_METRIC = "mc_full_margin"
TRUTHFULQA_LEAKAGE_METADATA_FIELDS = (
    "truthfulqa_artifact_split",
    "truthfulqa_seen_in_iti_fit",
    "truthfulqa_leakage_policy",
)
LOCKED_MANIFEST_METADATA_FIELDS = (
    "base_sample_id",
    "dataset",
    "mc_endpoint",
    *TRUTHFULQA_LEAKAGE_METADATA_FIELDS,
)


BOOL_METRICS: dict[str, MetricFn] = {
    "mc_letter_likelihood_correct": lambda row: bool(
        row["mc_letter_likelihood"]["chosen_is_gold"]
    ),
    "mc_generated_letter_correct": lambda row: bool(
        row["mc_generated_letter"]["chosen_is_gold"]
    ),
    "mc_likelihood_full_correct": lambda row: bool(
        row["mc_likelihood"]["full"]["chosen_is_gold"]
    ),
    "mc_likelihood_avg_correct": lambda row: bool(
        row["mc_likelihood"]["avg"]["chosen_is_gold"]
    ),
    "open_correct": lambda row: bool(row["open_grade"]["correct"]),
    "open_attempted": lambda row: bool(row["open_grade"]["attempted"]),
}
CONTINUOUS_METRICS: dict[str, MetricFn] = {
    "mc_full_margin": lambda row: float(row["mc_letter_likelihood"]["full"]["margin"]),
    "mc_avg_margin": lambda row: float(row["mc_letter_likelihood"]["avg"]["margin"]),
    "mc_option_text_full_margin": lambda row: float(
        row["mc_likelihood"]["full"]["margin"]
    ),
    "mc_option_text_avg_margin": lambda row: float(
        row["mc_likelihood"]["avg"]["margin"]
    ),
    "open_first_token_margin": lambda row: float(
        row["open_margins"]["first_token"]["margin"]
    ),
    "open_first3_margin": lambda row: float(row["open_margins"]["first3"]["margin"]),
    "open_full_margin": lambda row: float(row["open_margins"]["full"]["margin"]),
    "open_avg_margin": lambda row: float(row["open_margins"]["avg"]["margin"]),
}


def parse_alpha_label(path: Path) -> float:
    stem = path.stem
    if not stem.startswith("alpha_"):
        raise ValueError(f"Expected alpha_*.jsonl file, got {path}")
    return float(stem.removeprefix("alpha_"))


def load_locked_manifest_metadata(run_dir: Path) -> dict[str, dict[str, Any]]:
    path = run_dir / "manifest.locked.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("rows", []) if isinstance(payload, dict) else []
    metadata: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or "sample_id" not in row:
            continue
        metadata[str(row["sample_id"])] = {
            field: row.get(field) for field in LOCKED_MANIFEST_METADATA_FIELDS
        }
    return metadata


def load_run_rows(run_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    manifest_metadata = load_locked_manifest_metadata(run_dir)
    for path in sorted(run_dir.glob("*/alpha_*.jsonl")):
        alpha = parse_alpha_label(path)
        condition = path.parent.name
        for row in load_jsonl(path):
            row = dict(row)
            metadata = manifest_metadata.get(str(row.get("sample_id")))
            if metadata:
                for key, value in metadata.items():
                    if value is not None:
                        row.setdefault(key, value)
            row.setdefault("condition", condition)
            row.setdefault("alpha", alpha)
            rows.append(row)
    if not rows:
        raise ValueError(f"No SIMID alpha JSONL files found under {run_dir}")
    return rows


def index_rows(rows: list[dict[str, Any]]) -> dict[str, Panel]:
    indexed: dict[str, Panel] = defaultdict(lambda: defaultdict(dict))
    exact_keys: set[tuple[str, float, str]] = set()
    for row in rows:
        condition = str(row["condition"])
        alpha = float(row["alpha"])
        sample_id = str(row["sample_id"])
        base_sample_id = str(row.get("base_sample_id") or sample_id)
        exact_key = (condition, alpha, sample_id)
        if exact_key in exact_keys:
            raise ValueError(
                f"Duplicate SIMID row for condition={condition}, "
                f"alpha={alpha}, sample_id={sample_id}"
            )
        exact_keys.add(exact_key)
        bucket = indexed[condition][alpha]
        bucket.setdefault(base_sample_id, []).append(row)
    return {condition: dict(panel) for condition, panel in indexed.items()}


def replicate_key(row: dict[str, Any]) -> str:
    if "option_order_replicate" in row:
        return f"ord:{int(row['option_order_replicate'])}"
    return f"sample:{row['sample_id']}"


def replicate_key_set(rows: RowGroup) -> set[str]:
    return {replicate_key(row) for row in rows}


def row_mc_endpoint(row: dict[str, Any]) -> str:
    return str(row.get("mc_endpoint") or "unknown")


def dataset_endpoint_for_group(rows: RowGroup) -> tuple[str, str]:
    datasets = {str(row.get("dataset") or "unknown") for row in rows}
    endpoints = {row_mc_endpoint(row) for row in rows}
    if len(datasets) != 1 or len(endpoints) != 1:
        raise ValueError(
            "SIMID base-item replicate group mixes datasets or MC endpoints: "
            f"datasets={sorted(datasets)}, mc_endpoints={sorted(endpoints)}"
        )
    return next(iter(datasets)), next(iter(endpoints))


def dataset_endpoint_key(dataset: str, mc_endpoint: str) -> str:
    return f"{dataset}::{mc_endpoint}"


def render_stratum_metadata_value(value: Any) -> str:
    if value is None:
        return "unknown"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def truthfulqa_leakage_metadata_for_group(
    rows: RowGroup,
    *,
    dataset: str,
) -> tuple[tuple[str, str], ...]:
    if dataset != "truthfulqa":
        return ()
    has_leakage_metadata = any(
        any(field in row for field in TRUTHFULQA_LEAKAGE_METADATA_FIELDS)
        for row in rows
    )
    if not has_leakage_metadata:
        return ()
    metadata: list[tuple[str, str]] = []
    for field in TRUTHFULQA_LEAKAGE_METADATA_FIELDS:
        values = {render_stratum_metadata_value(row.get(field)) for row in rows}
        if len(values) != 1:
            raise ValueError(
                "SIMID TruthfulQA replicate group mixes leakage metadata for "
                f"{field}: {sorted(values)}"
            )
        metadata.append((field, next(iter(values))))
    return tuple(metadata)


def analysis_stratum_for_group(rows: RowGroup) -> StratumKey:
    dataset, mc_endpoint = dataset_endpoint_for_group(rows)
    return (
        dataset,
        mc_endpoint,
        truthfulqa_leakage_metadata_for_group(rows, dataset=dataset),
    )


def analysis_stratum_key(
    dataset: str,
    mc_endpoint: str,
    leakage_metadata: tuple[tuple[str, str], ...],
) -> str:
    key = dataset_endpoint_key(dataset, mc_endpoint)
    if not leakage_metadata:
        return key
    suffix = "::".join(f"{field}={value}" for field, value in leakage_metadata)
    return f"{key}::{suffix}"


def flatten_groups(groups_by_id: dict[str, RowGroup]) -> list[dict[str, Any]]:
    return [row for rows in groups_by_id.values() for row in rows]


def group_metric_value(rows: RowGroup, metric: MetricFn) -> float:
    if not rows:
        raise ValueError("SIMID row group cannot be empty")
    return float(np.mean([float(metric(row)) for row in rows]))


def bootstrap_mean_summary(
    values: list[float] | np.ndarray,
    *,
    n_resamples: int,
    seed: int,
    scale: float = 1.0,
    estimate_key: str = "estimate",
    ci_key: str = "ci",
) -> dict[str, Any]:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError("values must be one-dimensional")
    if len(arr) == 0:
        raise ValueError("values cannot be empty")
    rng = np.random.default_rng(seed)
    samples = np.empty(n_resamples, dtype=float)
    for sample_idx in range(n_resamples):
        indices = rng.choice(len(arr), size=len(arr), replace=True)
        samples[sample_idx] = float(arr[indices].mean() * scale)
    interval = percentile_interval(
        samples,
        method="bootstrap_percentile_item_grouped",
    )
    return {
        estimate_key: float(arr.mean() * scale),
        "sum": float(arr.sum()),
        "n": int(len(arr)),
        ci_key: interval.to_dict(),
        "bootstrap": {
            "n_resamples": int(n_resamples),
            "seed": int(seed),
            "confidence": float(interval.level),
            "resampling": "base_sample_id",
            "interval": "percentile",
        },
    }


def paired_rate_delta(
    sample_ids: list[str],
    baseline_rows: dict[str, RowGroup],
    comparison_rows: dict[str, RowGroup],
    metric: MetricFn,
    *,
    n_resamples: int,
    seed: int,
) -> dict[str, Any]:
    baseline = np.array(
        [
            group_metric_value(baseline_rows[sample_id], metric)
            for sample_id in sample_ids
        ],
        dtype=float,
    )
    comparison = np.array(
        [
            group_metric_value(comparison_rows[sample_id], metric)
            for sample_id in sample_ids
        ],
        dtype=float,
    )
    rng = np.random.default_rng(seed)
    samples = np.empty(n_resamples, dtype=float)
    for sample_idx in range(n_resamples):
        indices = rng.choice(len(sample_ids), size=len(sample_ids), replace=True)
        samples[sample_idx] = (
            float(comparison[indices].mean()) - float(baseline[indices].mean())
        ) * 100.0
    interval = percentile_interval(
        samples,
        method="bootstrap_percentile_paired_base_sample_id",
    )
    return {
        "estimate_pp": float((comparison.mean() - baseline.mean()) * 100.0),
        "ci_pp": interval.to_dict(),
        "bootstrap": {
            "n_resamples": int(n_resamples),
            "seed": int(seed),
            "confidence": float(interval.level),
            "resampling": "paired_by_base_sample_id",
            "interval": "percentile",
        },
    }


def require_paired_panel(
    indexed: dict[str, Panel],
    *,
    condition: str,
    alphas: list[float],
) -> tuple[list[str], Panel]:
    if condition not in indexed:
        raise ValueError(f"Missing SIMID condition: {condition}")
    panel = indexed[condition]
    missing_alphas = [alpha for alpha in alphas if alpha not in panel]
    if missing_alphas:
        raise ValueError(
            f"Condition {condition} is missing alpha files: {missing_alphas}"
        )
    sample_sets = {alpha: set(panel[alpha]) for alpha in alphas}
    reference_alpha = alphas[0]
    reference = sample_sets[reference_alpha]
    for alpha, sample_ids in sample_sets.items():
        if sample_ids != reference:
            raise ValueError(
                f"Unpaired SIMID rows for condition={condition}: alpha={alpha} "
                f"has {len(sample_ids)} samples, alpha={reference_alpha} has "
                f"{len(reference)}. Missing vs reference: "
                f"{sorted(reference - sample_ids)[:5]}; extra: "
                f"{sorted(sample_ids - reference)[:5]}"
            )
        for sample_id in reference:
            reference_replicates = replicate_key_set(panel[reference_alpha][sample_id])
            alpha_replicates = replicate_key_set(panel[alpha][sample_id])
            if alpha_replicates != reference_replicates:
                raise ValueError(
                    f"Unpaired SIMID replicate rows for condition={condition}, "
                    f"sample_id={sample_id}, alpha={alpha}: "
                    f"missing vs alpha={reference_alpha}: "
                    f"{sorted(reference_replicates - alpha_replicates)[:5]}; "
                    f"extra: {sorted(alpha_replicates - reference_replicates)[:5]}"
                )
    if not reference:
        raise ValueError(f"Condition {condition} has no paired samples")
    return sorted(reference), panel


def rate_at_alpha(
    sample_ids: list[str],
    rows_by_id: dict[str, RowGroup],
    metric: MetricFn,
    *,
    n_resamples: int,
    seed: int,
) -> dict[str, Any]:
    values = [
        group_metric_value(rows_by_id[sample_id], metric) for sample_id in sample_ids
    ]
    return bootstrap_mean_summary(values, n_resamples=n_resamples, seed=seed)


def paired_bool_delta(
    sample_ids: list[str],
    baseline_rows: dict[str, RowGroup],
    comparison_rows: dict[str, RowGroup],
    metric: MetricFn,
    *,
    n_resamples: int,
    seed: int,
) -> dict[str, Any]:
    return paired_rate_delta(
        sample_ids,
        baseline_rows,
        comparison_rows,
        metric,
        n_resamples=n_resamples,
        seed=seed,
    )


def paired_continuous_delta(
    sample_ids: list[str],
    baseline_rows: dict[str, RowGroup],
    comparison_rows: dict[str, RowGroup],
    metric: MetricFn,
    *,
    n_resamples: int,
    seed: int,
) -> dict[str, Any]:
    baseline = np.array(
        [
            group_metric_value(baseline_rows[sample_id], metric)
            for sample_id in sample_ids
        ],
        dtype=float,
    )
    comparison = np.array(
        [
            group_metric_value(comparison_rows[sample_id], metric)
            for sample_id in sample_ids
        ],
        dtype=float,
    )
    return paired_bootstrap_continuous_mean_difference_raw(
        baseline,
        comparison,
        n_resamples=n_resamples,
        seed=seed,
    )


def summarize_sample_ids(
    panel: Panel,
    *,
    sample_ids: list[str],
    alphas: list[float],
    baseline_alpha: float,
    n_resamples: int,
    seed: int,
) -> dict[str, Any]:
    baseline_rows = panel[baseline_alpha]
    rates: dict[str, Any] = {}
    deltas: dict[str, Any] = {}
    continuous_deltas: dict[str, Any] = {}
    interactions: dict[str, Any] = {}
    flips: dict[str, Any] = {}

    for alpha in alphas:
        rows_by_id = panel[alpha]
        rates[str(alpha)] = {
            name: rate_at_alpha(
                sample_ids,
                rows_by_id,
                metric,
                n_resamples=n_resamples,
                seed=seed,
            )
            for name, metric in BOOL_METRICS.items()
        }
        interactions[str(alpha)] = interaction_counts(
            sample_ids,
            rows_by_id,
            n_resamples=n_resamples,
            seed=seed,
        )
        if alpha == baseline_alpha:
            continue
        comparison_rows = rows_by_id
        deltas[str(alpha)] = {
            name: paired_bool_delta(
                sample_ids,
                baseline_rows,
                comparison_rows,
                metric,
                n_resamples=n_resamples,
                seed=seed,
            )
            for name, metric in BOOL_METRICS.items()
        }
        continuous_deltas[str(alpha)] = {
            name: paired_continuous_delta(
                sample_ids,
                baseline_rows,
                comparison_rows,
                metric,
                n_resamples=n_resamples,
                seed=seed,
            )
            for name, metric in CONTINUOUS_METRICS.items()
        }
        flips[str(alpha)] = flip_table(sample_ids, baseline_rows, comparison_rows)

    return {
        "n_paired_items": len(sample_ids),
        "pairing_unit": "base_sample_id",
        "n_rows_at_baseline": sum(
            len(baseline_rows[sample_id]) for sample_id in sample_ids
        ),
        "alphas": alphas,
        "baseline_alpha": baseline_alpha,
        "rates": rates,
        "paired_deltas_vs_baseline": deltas,
        "paired_margin_deltas_vs_baseline": continuous_deltas,
        "mc_open_interactions": interactions,
        "flip_tables_vs_baseline": flips,
    }


def analysis_stratum_sample_ids(
    sample_ids: list[str],
    panel: Panel,
    *,
    alphas: list[float],
    baseline_alpha: float,
) -> dict[StratumKey, list[str]]:
    strata: dict[StratumKey, list[str]] = defaultdict(list)
    for sample_id in sample_ids:
        reference = analysis_stratum_for_group(panel[baseline_alpha][sample_id])
        for alpha in alphas:
            alpha_value = analysis_stratum_for_group(panel[alpha][sample_id])
            if alpha_value != reference:
                raise ValueError(
                    f"SIMID sample_id={sample_id} changes analysis stratum "
                    f"between alpha={baseline_alpha} and alpha={alpha}: "
                    f"{reference} vs {alpha_value}"
                )
        strata[reference].append(sample_id)
    return {key: sorted(ids) for key, ids in strata.items()}


def summarize_condition(
    indexed: dict[str, Panel],
    *,
    condition: str,
    alphas: list[float],
    baseline_alpha: float,
    n_resamples: int,
    seed: int,
) -> dict[str, Any]:
    sample_ids, panel = require_paired_panel(
        indexed, condition=condition, alphas=alphas
    )
    summary = summarize_sample_ids(
        panel,
        sample_ids=sample_ids,
        alphas=alphas,
        baseline_alpha=baseline_alpha,
        n_resamples=n_resamples,
        seed=seed,
    )
    strata: dict[str, Any] = {}
    for (
        dataset,
        mc_endpoint,
        leakage_metadata,
    ), stratum_ids in analysis_stratum_sample_ids(
        sample_ids,
        panel,
        alphas=alphas,
        baseline_alpha=baseline_alpha,
    ).items():
        stratum_summary = summarize_sample_ids(
            panel,
            sample_ids=stratum_ids,
            alphas=alphas,
            baseline_alpha=baseline_alpha,
            n_resamples=n_resamples,
            seed=seed,
        )
        stratum_summary["dataset"] = dataset
        stratum_summary["mc_endpoint"] = mc_endpoint
        if leakage_metadata:
            stratum_summary["truthfulqa_leakage_metadata"] = dict(leakage_metadata)
        strata[analysis_stratum_key(dataset, mc_endpoint, leakage_metadata)] = (
            stratum_summary
        )

    return {
        "condition": condition,
        "summary_scope": "pooled_across_dataset_and_mc_endpoint",
        **summary,
        "dataset_endpoint_summaries": strata,
    }


def interaction_counts(
    sample_ids: list[str],
    rows_by_id: dict[str, RowGroup],
    *,
    n_resamples: int,
    seed: int,
) -> dict[str, Any]:
    counts: defaultdict[str, float] = defaultdict(float)
    agreement_values: list[float] = []
    for sample_id in sample_ids:
        rows = rows_by_id[sample_id]
        per_group: Counter[str] = Counter()
        for row in rows:
            mc = bool(BOOL_METRICS[PRIMARY_MC_RATE_METRIC](row))
            open_correct = bool(row["open_grade"]["correct"])
            key = f"mc_{'R' if mc else 'W'}__open_{'R' if open_correct else 'W'}"
            per_group[key] += 1
        for key, count in per_group.items():
            counts[key] += float(count) / len(rows)
        agreement_values.append(
            float(per_group["mc_R__open_R"] + per_group["mc_W__open_W"]) / len(rows)
        )
    return {
        "counts": {key: round(value, 6) for key, value in sorted(counts.items())},
        "agreement_rate": bootstrap_mean_summary(
            agreement_values,
            n_resamples=n_resamples,
            seed=seed,
        ),
    }


def flip_table(
    sample_ids: list[str],
    baseline_rows: dict[str, RowGroup],
    comparison_rows: dict[str, RowGroup],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    counts: defaultdict[str, float] = defaultdict(float)
    for sample_id in sample_ids:
        paired_rows = pair_replicate_groups(
            baseline_rows[sample_id],
            comparison_rows[sample_id],
            sample_id=sample_id,
        )
        per_group: Counter[str] = Counter()
        for replicate, base, comp in paired_rows:
            base_mc = bool(BOOL_METRICS[PRIMARY_MC_RATE_METRIC](base))
            comp_mc = bool(BOOL_METRICS[PRIMARY_MC_RATE_METRIC](comp))
            base_open = bool(base["open_grade"]["correct"])
            comp_open = bool(comp["open_grade"]["correct"])
            if not base_mc and comp_mc and base_open and not comp_open:
                failure_type = str(comp["open_grade"].get("failure_type"))
                if not bool(comp["open_grade"].get("attempted")):
                    bucket = "mc_W_to_R__open_R_to_not_attempted"
                elif failure_type == "wrong_entity":
                    bucket = "mc_W_to_R__open_R_to_wrong_entity"
                else:
                    bucket = "mc_W_to_R__open_R_to_alias_or_other"
                per_group[bucket] += 1
                rows.append(
                    {
                        "base_sample_id": sample_id,
                        "sample_id": comp["sample_id"],
                        "option_order_replicate": replicate,
                        "dataset": comp.get("dataset"),
                        "bucket": bucket,
                        "question": comp.get("question"),
                        "baseline_open_response": base["open_generation"]["response"],
                        "comparison_open_response": comp["open_generation"]["response"],
                        "comparison_open_grade": comp["open_grade"],
                    }
                )
        for bucket, count in per_group.items():
            counts[bucket] += float(count) / len(paired_rows)
    return {
        "counts": {key: round(value, 6) for key, value in sorted(counts.items())},
        "rows": rows,
        "count_unit": "base_sample_id_mean_over_option_order_replicates",
    }


def pair_replicate_groups(
    baseline: RowGroup,
    comparison: RowGroup,
    *,
    sample_id: str,
) -> list[tuple[str, dict[str, Any], dict[str, Any]]]:
    baseline_by_key = {replicate_key(row): row for row in baseline}
    comparison_by_key = {replicate_key(row): row for row in comparison}
    if set(baseline_by_key) != set(comparison_by_key):
        raise ValueError(
            f"Unpaired SIMID replicate rows for sample_id={sample_id}: "
            f"missing vs baseline={sorted(set(baseline_by_key) - set(comparison_by_key))[:5]}; "
            f"extra={sorted(set(comparison_by_key) - set(baseline_by_key))[:5]}"
        )
    return [
        (key, baseline_by_key[key], comparison_by_key[key])
        for key in sorted(baseline_by_key)
    ]


def per_item_slopes(
    sample_ids: list[str],
    panel: Panel,
    alphas: list[float],
    metric: MetricFn,
) -> np.ndarray:
    x = np.array(alphas, dtype=float)
    x_centered = x - x.mean()
    ss_x = float((x_centered**2).sum())
    if ss_x == 0.0:
        raise ValueError("Need at least two distinct alphas for slope summaries")
    slopes = np.empty(len(sample_ids), dtype=float)
    for idx, sample_id in enumerate(sample_ids):
        y = np.array(
            [group_metric_value(panel[alpha][sample_id], metric) for alpha in alphas],
            dtype=float,
        )
        slopes[idx] = float((x_centered * (y - y.mean())).sum() / ss_x)
    return slopes


def require_selected_control_replicate_pairing(
    *,
    selected_panel: Panel,
    control_panel: Panel,
    sample_ids: list[str],
    alphas: list[float],
    control: str,
) -> None:
    for alpha in alphas:
        for sample_id in sample_ids:
            selected_replicates = replicate_key_set(selected_panel[alpha][sample_id])
            control_replicates = replicate_key_set(control_panel[alpha][sample_id])
            if selected_replicates != control_replicates:
                raise ValueError(
                    f"Selected/control replicate sets differ for control={control}, "
                    f"sample_id={sample_id}, alpha={alpha}; "
                    f"missing vs selected="
                    f"{sorted(selected_replicates - control_replicates)[:5]}; "
                    f"extra={sorted(control_replicates - selected_replicates)[:5]}"
                )


def selected_minus_control_slope_summaries(
    indexed: dict[str, Panel],
    *,
    selected_condition: str,
    control_conditions: list[str],
    alphas: list[float],
    n_resamples: int,
    seed: int,
) -> dict[str, Any]:
    selected_ids, selected_panel = require_paired_panel(
        indexed, condition=selected_condition, alphas=alphas
    )
    summaries: dict[str, Any] = {}
    for control in control_conditions:
        control_ids, control_panel = require_paired_panel(
            indexed, condition=control, alphas=alphas
        )
        if set(control_ids) != set(selected_ids):
            raise ValueError(
                f"Selected/control sample sets differ for control={control}; "
                "SIMID slope summaries require item-level pairing"
            )
        sample_ids = sorted(selected_ids)
        require_selected_control_replicate_pairing(
            selected_panel=selected_panel,
            control_panel=control_panel,
            sample_ids=sample_ids,
            alphas=alphas,
            control=control,
        )
        metric_summaries: dict[str, Any] = {}
        for metric_name in ("mc_full_margin", "open_first3_margin", "open_full_margin"):
            metric = CONTINUOUS_METRICS[metric_name]
            selected_slopes = per_item_slopes(
                sample_ids, selected_panel, alphas, metric
            )
            control_slopes = per_item_slopes(sample_ids, control_panel, alphas, metric)
            metric_summaries[metric_name] = (
                paired_bootstrap_continuous_mean_difference_raw(
                    control_slopes,
                    selected_slopes,
                    n_resamples=n_resamples,
                    seed=seed,
                )
            )
        summaries[control] = metric_summaries
    return summaries


def _audit_queue_append(
    queue: list[dict[str, Any]],
    seen: set[tuple[str, str, float, str]],
    row: dict[str, Any],
    *,
    reason: str,
    extra: dict[str, Any] | None = None,
) -> None:
    key = (
        str(row["sample_id"]),
        str(row["condition"]),
        float(row["alpha"]),
        reason,
    )
    if key in seen:
        return
    seen.add(key)
    item = {
        "sample_id": row["sample_id"],
        "condition": row["condition"],
        "alpha": row["alpha"],
        "dataset": row.get("dataset"),
        "mc_endpoint": row.get("mc_endpoint"),
        "question": row.get("question"),
        "response": row["open_generation"]["response"],
        "gold_aliases": row.get("gold_aliases"),
        "open_grade": row.get("open_grade"),
        "mc_letter_likelihood": {
            "chosen_letter": row.get("mc_letter_likelihood", {}).get("chosen_letter"),
            "chosen_index": row.get("mc_letter_likelihood", {}).get("chosen_index"),
            "chosen_is_gold": row.get("mc_letter_likelihood", {}).get("chosen_is_gold"),
        },
        "reason": reason,
    }
    if extra:
        item.update(extra)
    queue.append(item)


def build_alias_audit_queue(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    queue: list[dict[str, Any]] = []
    seen: set[tuple[str, str, float, str]] = set()
    for row in rows:
        grade = row["open_grade"]
        mc_correct = bool(BOOL_METRICS[PRIMARY_MC_RATE_METRIC](row))
        open_correct = bool(grade["correct"])
        attempted = bool(grade["attempted"])

        if mc_correct != open_correct:
            _audit_queue_append(
                queue,
                seen,
                row,
                reason="mc_open_disagreement_requires_adjudication",
            )

        audit_reason: str | None = None
        if bool(grade["correct"]):
            if row.get("dataset") != "triviaqa_bridge":
                audit_reason = (
                    "non_bridge_deterministic_alias_correct_requires_adjudication"
                )
            elif grade.get("match_tier") != "exact":
                audit_reason = "non_exact_deterministic_alias_correct_requires_audit"
            if audit_reason is not None:
                queue.append(
                    {
                        "sample_id": row["sample_id"],
                        "condition": row["condition"],
                        "alpha": row["alpha"],
                        "dataset": row.get("dataset"),
                        "mc_endpoint": row.get("mc_endpoint"),
                        "question": row.get("question"),
                        "response": row["open_generation"]["response"],
                        "gold_aliases": row.get("gold_aliases"),
                        "open_grade": row.get("open_grade"),
                        "reason": audit_reason,
                    }
                )
            continue
        if row.get("dataset") != "triviaqa_bridge":
            _audit_queue_append(
                queue,
                seen,
                row,
                reason=(
                    "non_bridge_not_attempted_requires_adjudication"
                    if not attempted
                    else "non_bridge_deterministic_alias_incorrect_requires_adjudication"
                ),
            )
        margin = float(row["open_margins"]["full"]["margin"])
        if margin <= 0.0:
            continue
        _audit_queue_append(
            queue,
            seen,
            row,
            reason="open_incorrect_but_gold_margin_positive",
            extra={"open_full_margin": margin},
        )
    return queue


def option_order_stability_gate(groups: dict[str, RowGroup]) -> dict[str, Any]:
    replicate_values: defaultdict[str, list[float]] = defaultdict(list)
    for rows in groups.values():
        for row in rows:
            replicate_values[replicate_key(row)].append(
                float(BOOL_METRICS[PRIMARY_MC_RATE_METRIC](row))
            )
    rates = {
        key: float(np.mean(values)) for key, values in sorted(replicate_values.items())
    }
    if len(rates) < 2:
        return {
            "passed": False,
            "reason": "requires at least two option-order replicates",
            "replicate_rates": rates,
        }
    spread = max(rates.values()) - min(rates.values())
    n_base_items = len(groups)
    return {
        "passed": spread <= 0.25,
        "replicate_rates": rates,
        "max_rate_spread": spread,
        "n_base_items": n_base_items,
        "threshold": 0.25,
        "rate_unit": PRIMARY_MC_RATE_METRIC,
    }


def gold_position_balance_gate(groups: dict[str, RowGroup]) -> dict[str, Any]:
    counts: Counter[int] = Counter()
    n_options_values: set[int] = set()
    n_rows = 0
    for rows in groups.values():
        for row in rows:
            gold_indices = [int(idx) for idx in row.get("gold_option_indices", [])]
            if len(gold_indices) != 1:
                continue
            counts[gold_indices[0]] += 1
            n_options_values.add(len(row.get("mc_options", [])))
            n_rows += 1
    if not counts or len(n_options_values) != 1:
        return {
            "passed": False,
            "reason": "requires single-gold rows with a fixed option count",
            "counts": dict(sorted(counts.items())),
        }
    n_options = n_options_values.pop()
    expected = n_rows / n_options if n_options else 0.0
    max_abs_deviation = (
        max(abs(counts.get(idx, 0) - expected) for idx in range(n_options))
        if expected
        else 0.0
    )
    max_abs_deviation_share = max_abs_deviation / n_rows if n_rows else 0.0
    return {
        "passed": max_abs_deviation_share <= 0.20 or n_rows < 30,
        "counts": {str(idx): counts.get(idx, 0) for idx in range(n_options)},
        "n_rows": n_rows,
        "n_options": n_options,
        "max_abs_deviation_share": max_abs_deviation_share,
        "threshold": 0.20,
        "small_n_leniency": n_rows < 30,
    }


def option_length_balance_gate(groups: dict[str, RowGroup]) -> dict[str, Any]:
    diffs: list[float] = []
    for rows in groups.values():
        for row in rows:
            options = [str(option) for option in row.get("mc_options", [])]
            gold_indices = {int(idx) for idx in row.get("gold_option_indices", [])}
            if not options or not gold_indices:
                continue
            gold_lens = [
                len(options[idx].split()) for idx in gold_indices if idx < len(options)
            ]
            distractor_lens = [
                len(option.split())
                for idx, option in enumerate(options)
                if idx not in gold_indices
            ]
            if not gold_lens or not distractor_lens:
                continue
            diffs.append(float(np.mean(gold_lens) - np.mean(distractor_lens)))
    if not diffs:
        return {"passed": False, "reason": "no length-comparable MC rows"}
    mean_diff = float(np.mean(diffs))
    return {
        "passed": abs(mean_diff) <= 0.75,
        "mean_gold_minus_distractor_word_length": mean_diff,
        "n_rows": len(diffs),
        "threshold_abs_words": 0.75,
    }


def open_margin_alignment_gate(groups: dict[str, RowGroup]) -> dict[str, Any]:
    correct_margins: list[float] = []
    incorrect_margins: list[float] = []
    for rows in groups.values():
        for row in rows:
            margin = float(row["open_margins"]["full"]["margin"])
            if bool(row["open_grade"]["correct"]):
                correct_margins.append(margin)
            else:
                incorrect_margins.append(margin)
    if not correct_margins or not incorrect_margins:
        return {
            "passed": False,
            "reason": "requires both open-correct and open-incorrect rows",
            "n_correct": len(correct_margins),
            "n_incorrect": len(incorrect_margins),
        }
    correct_mean = float(np.mean(correct_margins))
    incorrect_mean = float(np.mean(incorrect_margins))
    return {
        "passed": correct_mean > incorrect_mean,
        "correct_mean_open_full_margin": correct_mean,
        "incorrect_mean_open_full_margin": incorrect_mean,
        "n_correct": len(correct_margins),
        "n_incorrect": len(incorrect_margins),
    }


def run_phase0_gates(
    indexed: dict[str, Panel],
    *,
    noop_tolerance: float,
) -> dict[str, Any]:
    gates: dict[str, Any] = {}
    if "unhooked" in indexed and "selected" in indexed:
        unhooked = indexed["unhooked"].get(0.0)
        selected = indexed["selected"].get(0.0)
        if unhooked is None or selected is None:
            gates["noop_equivalence"] = {
                "passed": False,
                "reason": "missing unhooked or selected alpha=0.0",
            }
        else:
            try:
                assert_noop_equivalence(
                    flatten_groups(unhooked),
                    flatten_groups(selected),
                    tolerance=noop_tolerance,
                )
            except AssertionError as exc:
                gates["noop_equivalence"] = {"passed": False, "reason": str(exc)}
            else:
                gates["noop_equivalence"] = {"passed": True}

    if "selected" in indexed and 0.0 in indexed["selected"]:
        groups = indexed["selected"][0.0]
        bridge_groups = [
            group
            for group in groups.values()
            if any(row.get("dataset") == "triviaqa_bridge" for row in group)
        ]
        if bridge_groups:
            correctness = [
                group_metric_value(group, BOOL_METRICS[PRIMARY_MC_RATE_METRIC])
                for group in bridge_groups
            ]
            n_options = max(
                len(row.get("mc_options", []))
                for group in bridge_groups
                for row in group
            )
            random_rate = 1.0 / n_options if n_options else 0.0
            observed = float(np.mean(correctness))
            gates["bridge_synthetic_mc_sanity"] = {
                "passed": observed > random_rate + 0.05 and observed < 0.98,
                "observed_rate": observed,
                "random_rate": random_rate,
                "n_base_items": len(bridge_groups),
                "replicate_policy": "mean_within_base_sample_id",
            }
            bridge_groups_by_id = {
                sample_id: group
                for sample_id, group in groups.items()
                if any(row.get("dataset") == "triviaqa_bridge" for row in group)
            }
            gates["bridge_option_order_stability"] = option_order_stability_gate(
                bridge_groups_by_id
            )
            gates["bridge_gold_position_balance"] = gold_position_balance_gate(
                bridge_groups_by_id
            )
            gates["bridge_option_length_balance"] = option_length_balance_gate(
                bridge_groups_by_id
            )
            gates["bridge_open_margin_alignment"] = open_margin_alignment_gate(
                bridge_groups_by_id
            )
    return gates


def format_ci(result: dict[str, Any], *, unit: str = "") -> str:
    if "estimate_pp" in result:
        ci = result["ci_pp"]
        return (
            f"{result['estimate_pp']:.2f}{unit} [{ci['lower']:.2f}, {ci['upper']:.2f}]"
        )
    ci = result["ci"]
    return f"{result['estimate']:.4f}{unit} [{ci['lower']:.4f}, {ci['upper']:.4f}]"


def write_report(results: dict[str, Any], path: Path) -> None:
    lines = [
        "# SIMID Report",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "All primary estimates use base-item pairing and bootstrap 95% CIs. "
        "The primary MC endpoint is the lettered forced-choice likelihood prompt.",
        "Open correctness is deterministic alias-grader correctness "
        "(not adjudicated human/judge correctness). Claimable open-ended results "
        "require adjudication or an explicit audit.",
        "",
    ]
    for condition, summary in results["conditions"].items():
        lines.append(f"## {condition}")
        lines.append(
            "Pooled aggregate scope: all datasets and MC endpoints in the selected "
            "condition panel."
        )
        lines.append(f"Paired items: {summary['n_paired_items']}")
        baseline = str(summary["baseline_alpha"])
        baseline_open = summary["rates"][baseline]["open_correct"]
        baseline_mc = summary["rates"][baseline][PRIMARY_MC_RATE_METRIC]
        lines.append(
            "Baseline rates: "
            f"lettered MC={format_ci(baseline_mc)}, "
            f"open={format_ci(baseline_open)}"
        )
        for alpha, deltas in summary["paired_deltas_vs_baseline"].items():
            mc = deltas[PRIMARY_MC_RATE_METRIC]
            open_delta = deltas["open_correct"]
            attempted = deltas["open_attempted"]
            lines.append(
                f"- alpha {alpha}: MC delta {format_ci(mc, unit=' pp')}; "
                f"open delta {format_ci(open_delta, unit=' pp')}; "
                f"attempt delta {format_ci(attempted, unit=' pp')}"
            )
        strata = summary.get("dataset_endpoint_summaries", {})
        if strata:
            lines.append("")
            lines.append(
                "Dataset / MC endpoint strata (TruthfulQA leakage split when available):"
            )
            for key in sorted(strata):
                stratum = strata[key]
                stratum_baseline = str(stratum["baseline_alpha"])
                stratum_open = stratum["rates"][stratum_baseline]["open_correct"]
                stratum_mc = stratum["rates"][stratum_baseline][PRIMARY_MC_RATE_METRIC]
                leakage_metadata = stratum.get("truthfulqa_leakage_metadata", {})
                leakage_label = ""
                if leakage_metadata:
                    rendered = ", ".join(
                        f"{field.removeprefix('truthfulqa_')}={value}"
                        for field, value in leakage_metadata.items()
                    )
                    leakage_label = f" ({rendered})"
                lines.append(
                    f"- {stratum['dataset']} / {stratum['mc_endpoint']}"
                    f"{leakage_label}: "
                    f"n={stratum['n_paired_items']}; "
                    f"lettered MC={format_ci(stratum_mc)}; "
                    f"open={format_ci(stratum_open)}"
                )
                for alpha, deltas in stratum["paired_deltas_vs_baseline"].items():
                    mc = deltas[PRIMARY_MC_RATE_METRIC]
                    open_delta = deltas["open_correct"]
                    lines.append(
                        f"  - alpha {alpha}: MC delta {format_ci(mc, unit=' pp')}; "
                        f"open delta {format_ci(open_delta, unit=' pp')}"
                    )
        lines.append("")
    if results.get("selected_minus_control_slopes"):
        lines.append("## Selected Minus Control Slopes")
        for control, metrics in results["selected_minus_control_slopes"].items():
            lines.append(f"- {control}:")
            for metric_name, result in metrics.items():
                lines.append(f"  - {metric_name}: {format_ci(result)}")
        lines.append("")
    if results.get("phase0_gates"):
        lines.append("## Phase 0 Gates")
        for name, gate in results["phase0_gates"].items():
            lines.append(f"- {name}: {'PASS' if gate.get('passed') else 'FAIL'}")
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--baseline-alpha", type=float, default=0.0)
    parser.add_argument("--conditions", nargs="+", default=None)
    parser.add_argument("--alphas", type=float, nargs="+", default=None)
    parser.add_argument("--n-resamples", type=int, default=DEFAULT_BOOTSTRAP_RESAMPLES)
    parser.add_argument("--seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    parser.add_argument("--phase0-gates", action="store_true")
    parser.add_argument("--noop-tolerance", type=float, default=1e-5)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--report-md", default=None)
    parser.add_argument("--allow-overwrite", action="store_true")
    return parser.parse_args(argv)


def refuse_analysis_output_overwrite(
    paths: list[Path], *, allow_overwrite: bool
) -> None:
    if allow_overwrite:
        return
    existing = [path for path in dict.fromkeys(paths) if path.exists()]
    if existing:
        rendered = ", ".join(str(path) for path in existing)
        raise FileExistsError(
            f"Refusing to overwrite existing SIMID analysis outputs: {rendered}. "
            "Pass --allow-overwrite to replace them explicitly."
        )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    run_dir = Path(args.run_dir)
    output_json = (
        Path(args.output_json) if args.output_json else run_dir / "results.json"
    )
    report_md = Path(args.report_md) if args.report_md else run_dir / "report.md"
    alias_queue_path = run_dir / "alias_audit_queue.jsonl"
    refuse_analysis_output_overwrite(
        [output_json, report_md, alias_queue_path],
        allow_overwrite=args.allow_overwrite,
    )
    provenance = start_run_provenance(
        args,
        primary_target=output_json,
        output_targets=[output_json, report_md, alias_queue_path],
    )
    status = "completed"
    extra: dict[str, Any] = {}
    try:
        rows = load_run_rows(run_dir)
        indexed = index_rows(rows)
        conditions = args.conditions or sorted(indexed)
        alphas = args.alphas or sorted(
            {alpha for condition in conditions for alpha in indexed[condition]}
        )
        if args.baseline_alpha not in alphas:
            raise ValueError(
                f"baseline alpha {args.baseline_alpha} is not in alpha grid {alphas}"
            )

        condition_summaries = {
            condition: summarize_condition(
                indexed,
                condition=condition,
                alphas=alphas,
                baseline_alpha=args.baseline_alpha,
                n_resamples=args.n_resamples,
                seed=args.seed,
            )
            for condition in conditions
        }
        controls = [
            condition
            for condition in conditions
            if condition not in {"selected", "unhooked"}
        ]
        slope_summaries = {}
        if "selected" in conditions and controls and len(alphas) >= 2:
            slope_summaries = selected_minus_control_slope_summaries(
                indexed,
                selected_condition="selected",
                control_conditions=controls,
                alphas=alphas,
                n_resamples=args.n_resamples,
                seed=args.seed,
            )
        alias_queue = build_alias_audit_queue(rows)
        with alias_queue_path.open("w", encoding="utf-8") as handle:
            for row in alias_queue:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

        results = {
            "schema_version": "simid_analysis/v1",
            "run_dir": str(run_dir),
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "baseline_alpha": args.baseline_alpha,
            "alphas": alphas,
            "conditions": condition_summaries,
            "selected_minus_control_slopes": slope_summaries,
            "open_grading": OPEN_GRADING_METHOD,
            "alias_audit_queue": {
                "path": str(alias_queue_path),
                "n": len(alias_queue),
            },
            "bootstrap": {
                "n_resamples": args.n_resamples,
                "seed": args.seed,
                "confidence": 0.95,
            },
        }
        if args.phase0_gates:
            results["phase0_gates"] = run_phase0_gates(
                indexed,
                noop_tolerance=args.noop_tolerance,
            )
        output_json.write_text(
            json.dumps(results, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        write_report(results, report_md)
        extra["n_rows"] = len(rows)
        extra["n_alias_audit_rows"] = len(alias_queue)
        print(f"Wrote SIMID analysis to {output_json} and {report_md}")
    except BaseException as exc:
        status = provenance_status_for_exception(exc)
        extra["error"] = provenance_error_message(exc)
        raise
    finally:
        finish_run_provenance(provenance, status, extra)


if __name__ == "__main__":
    main()
