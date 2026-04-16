#!/usr/bin/env python3
"""Export site-facing JSON artifacts from committed experiment outputs."""

from __future__ import annotations

import argparse
from collections import defaultdict
import copy
import hashlib
import json
import re
from datetime import date
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from scipy.stats import spearmanr

try:
    from uncertainty import (
        DEFAULT_BOOTSTRAP_RESAMPLES,
        DEFAULT_BOOTSTRAP_SEED,
        DEFAULT_CONFIDENCE,
        build_rate_summary,
        paired_bootstrap_curve_effects,
        percentile_interval,
    )
except ModuleNotFoundError:
    from scripts.uncertainty import (
        DEFAULT_BOOTSTRAP_RESAMPLES,
        DEFAULT_BOOTSTRAP_SEED,
        DEFAULT_CONFIDENCE,
        build_rate_summary,
        paired_bootstrap_curve_effects,
        percentile_interval,
    )


ALPHAS = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
GEMMA_3_4B_LAYER_COUNT = 34
CLASSIFIER_STRUCTURE_SUMMARY_PATH = Path(
    "data/gemma3_4b/pipeline/classifier_structure_summary.json"
)
TOP_NEURON_ARTIFACT_SUMMARY_PATH = Path(
    "data/gemma3_4b/pipeline/neuron_4288_summary.json"
)
TOP_NEURON_ARTIFACT_TEST_SLUGS = (
    "single_neuron_auc",
    "distribution_separation",
    "c_sweep_stability",
    "largest_contribution_share",
    "ablation_accuracy_drop",
    "max_top10_correlation",
)
FALSEQA_NEGATIVE_CONTROL_STATUS = "available"
JAILBREAK_DECODING_TEMPERATURE = 0.7
JAILBREAK_GENERATION_LABEL = f"stochastic (T={JAILBREAK_DECODING_TEMPERATURE:.1f})"
JAILBREAK_HOLDOUT_CONTAMINATED_IDS = frozenset(
    {
        "jbb_harmful_62_t2",
        "jbb_harmful_97_t0",
        "jbb_harmful_14_t3",
        "jbb_harmful_91_t4",
        "jbb_harmful_36_t4",
        "jbb_harmful_34_t0",
        "jbb_harmful_19_t0",
        "jbb_harmful_3_t2",
    }
)
JAILBREAK_HOLDOUT_ALPHAS = (0.0, 1.5, 3.0)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def load_text(path: Path) -> str:
    return path.read_text()


def require_existing_path(path: Path, label: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")
    return path


def find_results_json(experiment_dir: Path) -> Path:
    """Return the results JSON for an experiment directory.

    Prefers the legacy ``results.json`` name for backwards compatibility with
    committed experiment directories.  Falls back to the newest timestamped
    ``results.YYYYMMDD_HHMMSS.json`` produced by the current naming scheme.
    Raises ``FileNotFoundError`` if neither is found.
    """
    legacy = experiment_dir / "results.json"
    if legacy.exists():
        return legacy
    candidates = sorted(experiment_dir.glob("results.*.json"))
    if candidates:
        return candidates[-1]
    raise FileNotFoundError(
        f"No results JSON found in {experiment_dir}. "
        "Expected results.json or results.YYYYMMDD_HHMMSS.json."
    )


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def normalize_report_text(text: str) -> str:
    return text.replace("−", "-").replace("–", "-").replace("—", "-")


def require_report_match(text: str, pattern: str, label: str) -> re.Match[str]:
    match = re.search(pattern, text, flags=re.MULTILINE | re.DOTALL)
    if match is None:
        raise ValueError(f"Could not parse {label} from report text.")
    return match


def strip_markdown_formatting(text: str) -> str:
    return re.sub(r"[*`]+", "", text).strip()


def normalize_markdown_label(text: str) -> str:
    cleaned = strip_markdown_formatting(text).lower().replace("_", " ")
    return re.sub(r"\s+", " ", cleaned).strip()


def parse_markdown_pipe_rows(text: str) -> list[list[str]]:
    rows: list[list[str]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if cells and all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells):
            continue
        rows.append(cells)
    return rows


def parse_pp_per_alpha(text: str, label: str) -> float:
    match = re.search(
        r"([+-]?\d+(?:\.\d+)?)\s*pp/alpha",
        strip_markdown_formatting(text),
    )
    if match is None:
        raise ValueError(f"Could not parse {label} estimate from {text!r}")
    return float(match.group(1))


def parse_ci_bounds(text: str, label: str) -> tuple[float, float]:
    match = re.search(
        r"\[([+-]?\d+(?:\.\d+)?),\s*([+-]?\d+(?:\.\d+)?)\]",
        strip_markdown_formatting(text),
    )
    if match is None:
        raise ValueError(f"Could not parse {label} CI from {text!r}")
    return float(match.group(1)), float(match.group(2))


def extract_jailbreak_slope_statistics_from_report(
    paired_report: str,
) -> dict[str, dict[str, Any]]:
    stats: dict[tuple[str, str], dict[str, Any]] = {}
    for cells in parse_markdown_pipe_rows(paired_report):
        if len(cells) != 4:
            continue
        condition = normalize_markdown_label(cells[0])
        metric = normalize_markdown_label(cells[1])
        if condition not in {"h-neuron v2", "h-neuron v3"}:
            continue
        if metric not in {"harmful slope", "substantive compliance slope"}:
            continue
        lower, upper = parse_ci_bounds(cells[3], f"{condition} {metric}")
        stats[(condition, metric)] = {
            "estimate_pp_per_alpha": parse_pp_per_alpha(
                cells[2], f"{condition} {metric}"
            ),
            "ci": (lower, upper),
        }

    required_pairs = {
        ("h-neuron v2", "harmful slope"): "v2_yes_slope_pp_per_alpha",
        ("h-neuron v3", "harmful slope"): "v3_yes_slope_pp_per_alpha",
        (
            "h-neuron v3",
            "substantive compliance slope",
        ): "v3_substantive_slope_pp_per_alpha",
    }
    if all(pair in stats for pair in required_pairs):
        return {output_key: stats[pair] for pair, output_key in required_pairs.items()}

    legacy_patterns = {
        "v2_yes_slope_pp_per_alpha": (
            r"\|\s*harmful_binary slope \(H-neuron\)\s*\|\s*\*{0,2}v2\*{0,2}\s*\|"
            r"\s*\*{0,2}([+-]?\d+\.\d+)\s*pp/alpha\*{0,2}\s*\|"
            r"\s*\*{0,2}\[([+-]?\d+\.\d+),\s*([+-]?\d+\.\d+)\]\*{0,2}",
            "legacy jailbreak v2 binary slope",
        ),
        "v3_yes_slope_pp_per_alpha": (
            r"\|\s*harmful_binary slope \(H-neuron\)\s*\|\s*\*{0,2}v3\*{0,2}\s*\|"
            r"\s*\*{0,2}([+-]?\d+\.\d+)\s*pp/alpha\*{0,2}\s*\|"
            r"\s*\*{0,2}\[([+-]?\d+\.\d+),\s*([+-]?\d+\.\d+)\]\*{0,2}",
            "legacy jailbreak v3 binary slope",
        ),
        "v3_substantive_slope_pp_per_alpha": (
            r"\|\s*\*{0,2}substantive[_ ]compliance slope \(H-neuron\)\*{0,2}\s*\|"
            r"\s*\*{0,2}v3\*{0,2}\s*\|"
            r"\s*\*{0,2}([+-]?\d+\.\d+)\s*pp/alpha\*{0,2}\s*\|"
            r"\s*\*{0,2}\[([+-]?\d+\.\d+),\s*([+-]?\d+\.\d+)\]\*{0,2}",
            "legacy jailbreak v3 substantive slope",
        ),
    }
    fallback_stats: dict[str, dict[str, Any]] = {}
    for key, (pattern, label) in legacy_patterns.items():
        match = require_report_match(paired_report, pattern, label)
        fallback_stats[key] = {
            "estimate_pp_per_alpha": float(match.group(1)),
            "ci": (float(match.group(2)), float(match.group(3))),
        }
    return fallback_stats


def load_jailbreak_paired_evaluator_statistics(
    paired_report: str,
    available_evaluator_summary: dict[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    full_set_stats: Any = None
    if isinstance(available_evaluator_summary, dict):
        full_set_stats = available_evaluator_summary.get("h_neuron_sweep", {}).get(
            "statistics"
        )
    required_keys = {
        "v2_yes_slope_pp_per_alpha",
        "v3_yes_slope_pp_per_alpha",
        "v3_substantive_slope_pp_per_alpha",
    }
    if isinstance(full_set_stats, dict) and required_keys.issubset(full_set_stats):
        return {
            key: copy.deepcopy(full_set_stats[key]) for key in sorted(required_keys)
        }
    return extract_jailbreak_slope_statistics_from_report(paired_report)


def pp_summary(estimate: float, lower: float, upper: float) -> dict[str, Any]:
    return {
        "estimate": estimate,
        "ci": {
            "lower": lower,
            "upper": upper,
            "level": 0.95,
            "method": "report_percentile_ci",
        },
    }


def add_percent_point_aliases(metric: dict[str, Any]) -> dict[str, Any]:
    metric_copy = copy.deepcopy(metric)
    if "estimate_pp" in metric_copy and "estimate" not in metric_copy:
        metric_copy["estimate"] = metric_copy["estimate_pp"]
    if "ci_pp" in metric_copy and "ci" not in metric_copy:
        metric_copy["ci"] = copy.deepcopy(metric_copy["ci_pp"])
    return metric_copy


def add_delta_aliases(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        metric_name: (
            add_percent_point_aliases(metric_value)
            if isinstance(metric_value, dict)
            else copy.deepcopy(metric_value)
        )
        for metric_name, metric_value in metrics.items()
    }


def add_legacy_d7_condition_aliases(condition: dict[str, Any]) -> dict[str, Any]:
    """Preserve the historical D7 site contract during the v3 migration."""
    condition_copy = copy.deepcopy(condition)
    if (
        "csv2_yes" not in condition_copy
        and "strict_harmfulness_normalized" in condition_copy
    ):
        condition_copy["csv2_yes"] = copy.deepcopy(
            condition_copy["strict_harmfulness_normalized"]
        )
    return condition_copy


def add_legacy_d7_comparison_aliases(metrics: dict[str, Any]) -> dict[str, Any]:
    """Expose both legacy `csv2_yes` and current normalized comparison keys."""
    metrics_copy = add_delta_aliases(metrics)
    if (
        "csv2_yes" not in metrics_copy
        and "strict_harmfulness_normalized" in metrics_copy
    ):
        metrics_copy["csv2_yes"] = copy.deepcopy(
            metrics_copy["strict_harmfulness_normalized"]
        )
    return metrics_copy


def index_conditions_by_name(
    conditions: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    return {condition["name"]: condition for condition in conditions}


def invert_interval(interval: dict[str, Any]) -> dict[str, Any]:
    interval_copy = copy.deepcopy(interval)
    lower = float(interval_copy["lower"])
    upper = float(interval_copy["upper"])
    interval_copy["lower"] = -upper
    interval_copy["upper"] = -lower
    return interval_copy


def invert_transition_counts(metric: dict[str, Any]) -> dict[str, Any]:
    transitions = metric.get("transitions")
    if not isinstance(transitions, dict):
        return {}
    return {
        "harmful_to_not_harmful": int(transitions.get("not_harmful_to_harmful", 0)),
        "not_harmful_to_harmful": int(transitions.get("harmful_to_not_harmful", 0)),
    }


def invert_delta_metric(metric: dict[str, Any]) -> dict[str, Any]:
    metric_copy = copy.deepcopy(metric)
    if "estimate_pp" in metric_copy:
        metric_copy["estimate_pp"] = -float(metric_copy["estimate_pp"])
    elif "estimate" in metric_copy:
        metric_copy["estimate"] = -float(metric_copy["estimate"])

    if "ci_pp" in metric_copy:
        metric_copy["ci_pp"] = invert_interval(metric_copy["ci_pp"])
    elif "ci" in metric_copy:
        metric_copy["ci"] = invert_interval(metric_copy["ci"])

    inverted_transitions = invert_transition_counts(metric_copy)
    if inverted_transitions:
        metric_copy["transitions"] = inverted_transitions

    return add_percent_point_aliases(metric_copy)


def invert_delta_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        metric_name: (
            invert_delta_metric(metric_value)
            if isinstance(metric_value, dict)
            else copy.deepcopy(metric_value)
        )
        for metric_name, metric_value in metrics.items()
    }


def alpha_key(alpha: float) -> str:
    return f"{alpha:.1f}"


def sorted_result_alphas(results: dict[str, Any]) -> list[float]:
    return sorted(float(alpha) for alpha in results)


def as_pct(rate: float) -> float:
    return round(rate * 100, 1)


def count_jsonl_rows(path: Path) -> int:
    with path.open() as handle:
        return sum(1 for line in handle if line.strip())


def load_rows_by_key(
    base_dir: Path,
    alphas: tuple[float, ...],
) -> dict[tuple[str, float], dict[str, Any]]:
    rows_by_key: dict[tuple[str, float], dict[str, Any]] = {}
    for alpha in alphas:
        for row in load_jsonl(base_dir / f"alpha_{alpha:.1f}.jsonl"):
            rows_by_key[(row["id"], float(row["alpha"]))] = row
    return rows_by_key


def gold_label_to_bool(label: str) -> bool:
    if label == "HARMFUL":
        return True
    if label == "SAFE":
        return False
    raise ValueError(f"Unsupported gold label: {label}")


def build_cluster_bootstrap_accuracy_summary(
    records: list[dict[str, Any]],
    prediction_key: str,
    *,
    n_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    confidence: float = DEFAULT_CONFIDENCE,
) -> dict[str, Any]:
    grouped_correctness: dict[str, list[float]] = defaultdict(list)
    correct_count = 0
    total = 0

    for record in records:
        correct = float(record[prediction_key] == record["gold"])
        grouped_correctness[record["id"]].append(correct)
        correct_count += int(correct)
        total += 1

    if total == 0:
        raise ValueError("Cannot build holdout accuracy summary from zero records.")

    cluster_ids = np.array(sorted(grouped_correctness), dtype=object)
    rng = np.random.default_rng(seed)
    samples = np.empty(n_resamples, dtype=float)

    for sample_idx in range(n_resamples):
        sampled_ids = rng.choice(cluster_ids, size=len(cluster_ids), replace=True)
        sample_correctness = [
            value
            for cluster_id in sampled_ids
            for value in grouped_correctness[str(cluster_id)]
        ]
        samples[sample_idx] = float(np.mean(sample_correctness))

    interval = percentile_interval(
        samples,
        confidence,
        method="bootstrap_percentile_cluster_by_prompt_id",
    )
    return {
        "estimate": correct_count / total,
        "count": correct_count,
        "n": total,
        "ci": interval.to_dict(),
    }


def build_jailbreak_holdout_summary(repo_root: Path) -> dict[str, Any]:
    gold_rows = load_jsonl(
        repo_root / "tests/gold_labels/jailbreak_cross_alpha_gold.jsonl"
    )
    csv2_v3_rows = load_rows_by_key(
        repo_root / "data/judge_validation/csv2_v3",
        JAILBREAK_HOLDOUT_ALPHAS,
    )
    strongreject_rows = load_rows_by_key(
        repo_root / "data/judge_validation/strongreject",
        JAILBREAK_HOLDOUT_ALPHAS,
    )

    joined_rows: list[dict[str, Any]] = []
    for gold_row in gold_rows:
        human_label = gold_row["human_label"]
        if human_label not in {"HARMFUL", "SAFE"}:
            continue

        key = (gold_row["id"], float(gold_row["alpha"]))
        csv2_row = csv2_v3_rows.get(key)
        strongreject_row = strongreject_rows.get(key)
        if csv2_row is None or strongreject_row is None:
            raise KeyError(f"Missing holdout evaluator row for {key!r}")

        joined_rows.append(
            {
                "id": gold_row["id"],
                "alpha": float(gold_row["alpha"]),
                "gold": gold_label_to_bool(human_label),
                "csv2v3": csv2_row["csv2"]["harmful_binary"] == "yes",
                "strongreject_4o": (
                    strongreject_row["strongreject"]["harmful_binary"] == "yes"
                ),
            }
        )

    holdout_rows = [
        row
        for row in joined_rows
        if row["id"] not in JAILBREAK_HOLDOUT_CONTAMINATED_IDS
    ]

    v3_summary = build_cluster_bootstrap_accuracy_summary(holdout_rows, "csv2v3")
    strongreject_summary = build_cluster_bootstrap_accuracy_summary(
        holdout_rows, "strongreject_4o"
    )
    v3_errors = {
        (row["id"], row["alpha"])
        for row in holdout_rows
        if row["csv2v3"] != row["gold"]
    }
    strongreject_errors = {
        (row["id"], row["alpha"])
        for row in holdout_rows
        if row["strongreject_4o"] != row["gold"]
    }
    discordant_correctness_count = sum(
        (row["csv2v3"] == row["gold"]) != (row["strongreject_4o"] == row["gold"])
        for row in holdout_rows
    )

    return {
        "status": "tie_with_v3_on_holdout",
        "source_files": [
            "tests/gold_labels/jailbreak_cross_alpha_gold.jsonl",
            "data/judge_validation/csv2_v3/alpha_0.0.jsonl",
            "data/judge_validation/csv2_v3/alpha_1.5.jsonl",
            "data/judge_validation/csv2_v3/alpha_3.0.jsonl",
            "data/judge_validation/strongreject/alpha_0.0.jsonl",
            "data/judge_validation/strongreject/alpha_1.5.jsonl",
            "data/judge_validation/strongreject/alpha_3.0.jsonl",
        ],
        "holdout_n_records": len(holdout_rows),
        "holdout_n_prompt_ids": len({row["id"] for row in holdout_rows}),
        "bootstrap": {
            "n_resamples": DEFAULT_BOOTSTRAP_RESAMPLES,
            "seed": DEFAULT_BOOTSTRAP_SEED,
            "confidence": DEFAULT_CONFIDENCE,
            "resampling": "cluster_by_prompt_id",
            "interval": "percentile",
        },
        "v3_accuracy": v3_summary,
        "strongreject_4o_accuracy": strongreject_summary,
        "v3_accuracy_pct": as_pct(v3_summary["estimate"]),
        "strongreject_4o_accuracy_pct": as_pct(strongreject_summary["estimate"]),
        "discordant_correctness_count": discordant_correctness_count,
        "error_sets_match": v3_errors == strongreject_errors,
    }


def with_pct(summary: dict[str, Any], estimate_key: str = "estimate") -> dict[str, Any]:
    payload = dict(summary)
    payload["pct"] = as_pct(payload[estimate_key])
    return payload


def explicit_answer_agreement_summary(
    samples: list[dict[str, Any]],
) -> dict[str, Any] | None:
    agreement_flags = [
        sample["answer_agrees_with_model_alpha0"]
        for sample in samples
        if sample.get("answer_agrees_with_model_alpha0") is not None
    ]
    if not agreement_flags:
        return None
    return build_rate_summary(
        int(sum(agreement_flags)),
        len(agreement_flags),
        total_key="n_total",
    )


def compact_llm_enrichment(llm: dict[str, Any]) -> dict[str, Any]:
    payload = dict(llm)
    agreement = payload.get(
        "verification_agreement"
    ) or explicit_answer_agreement_summary(payload.get("samples", []))
    if agreement:
        payload["verification_agreement"] = with_pct(agreement)
    return payload


def compliance_summary_from_record(record: dict[str, Any]) -> dict[str, Any]:
    return record.get("compliance") or build_rate_summary(
        record["n_compliant"],
        record["n_total"],
        count_key="n_compliant",
        total_key="n_total",
    )


def parse_failure_summary_from_record(record: dict[str, Any]) -> dict[str, Any]:
    if "parse_failure" in record:
        return record["parse_failure"]
    return build_rate_summary(
        record["parse_failures"],
        record["n_total"],
        count_key="count",
        total_key="n_total",
    )


def build_rate_points(
    results: dict[str, Any], alphas: list[float] | None = None
) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    for alpha in alphas or ALPHAS:
        key = alpha_key(alpha)
        if key not in results:
            continue
        record = results[key]
        compliance = compliance_summary_from_record(record)
        points.append(
            {
                "alpha": alpha,
                "n_total": record["n_total"],
                "n_compliant": record["n_compliant"],
                "compliance_rate": record["compliance_rate"],
                "compliance_pct": as_pct(record["compliance_rate"]),
                "ci": compliance["ci"],
            }
        )
    return points


def format_p_value(p_value: float, digits: int = 3) -> str:
    threshold = 10**-digits
    if p_value < threshold:
        return f"p<{threshold:.{digits}f}"
    return f"p={p_value:.{digits}f}"


def build_monotonicity_summary(points: list[dict[str, Any]]) -> dict[str, Any]:
    alphas = np.array([point["alpha"] for point in points], dtype=float)
    rates = np.array([point["compliance_rate"] for point in points], dtype=float)
    rho, p_value = spearmanr(alphas, rates)
    is_monotonic = all(rates[idx] <= rates[idx + 1] for idx in range(len(rates) - 1))
    is_strictly_increasing = all(
        rates[idx] < rates[idx + 1] for idx in range(len(rates) - 1)
    )
    if is_strictly_increasing:
        description = (
            f"Strictly increasing across the {len(points)} exported α values "
            f"(Spearman ρ={rho:.3f}, {format_p_value(float(p_value))})."
        )
    elif is_monotonic:
        description = (
            f"Non-decreasing across the {len(points)} exported α values "
            f"(Spearman ρ={rho:.3f}, {format_p_value(float(p_value))})."
        )
    else:
        description = (
            f"Not monotonic across the {len(points)} exported α values "
            f"(Spearman ρ={rho:.3f}, {format_p_value(float(p_value))})."
        )
    return {
        "spearman_rho": float(rho),
        "spearman_p": float(p_value),
        "is_significant": bool(p_value < 0.05),
        "is_monotonic": is_monotonic,
        "is_strictly_increasing": is_strictly_increasing,
        "description": description,
    }


def build_standard_format_points(
    base_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    parse_failure_points: list[dict[str, Any]] = []
    parseable_subset_points: list[dict[str, Any]] = []

    for alpha in ALPHAS:
        rows = load_jsonl(base_dir / f"alpha_{alpha:.1f}.jsonl")
        n_total = len(rows)
        parse_failures = sum(row["chosen"] is None for row in rows)
        parseable_rows = [row for row in rows if row["chosen"] is not None]
        parseable_n = len(parseable_rows)
        compliant_parseable = sum(bool(row["compliance"]) for row in parseable_rows)
        parseable_rate = compliant_parseable / parseable_n if parseable_n else 0.0
        parse_failure_summary = build_rate_summary(
            parse_failures,
            n_total,
            count_key="count",
            total_key="n_total",
        )
        parseable_summary = build_rate_summary(
            compliant_parseable,
            parseable_n,
            count_key="n_compliant",
            total_key="parseable_n",
        )

        parse_failure_points.append(
            {
                "alpha": alpha,
                "n_total": n_total,
                "count": parse_failures,
                "rate": parse_failures / n_total,
                "pct": as_pct(parse_failures / n_total),
                "ci": parse_failure_summary["ci"],
            }
        )
        parseable_subset_points.append(
            {
                "alpha": alpha,
                "n_total": n_total,
                "parseable_n": parseable_n,
                "n_compliant": compliant_parseable,
                "compliance_rate": parseable_rate,
                "compliance_pct": as_pct(parseable_rate),
                "ci": parseable_summary["ci"],
            }
        )

    return parse_failure_points, parseable_subset_points


def build_anti_population(base_dir: Path) -> dict[str, Any]:
    rows_by_alpha = {
        alpha: {
            row["id"]: row for row in load_jsonl(base_dir / f"alpha_{alpha:.1f}.jsonl")
        }
        for alpha in ALPHAS
    }
    reference_ids = set(rows_by_alpha[ALPHAS[0]])
    for alpha in ALPHAS[1:]:
        ids = set(rows_by_alpha[alpha])
        if ids != reference_ids:
            raise ValueError(
                f"Mismatched anti-compliance sample IDs at alpha={alpha:.1f}"
            )

    trajectories = {
        sample_id: [
            bool(rows_by_alpha[alpha][sample_id]["compliance"]) for alpha in ALPHAS
        ]
        for sample_id in sorted(reference_ids)
    }
    always_compliant_ids = [
        sample_id for sample_id, values in trajectories.items() if all(values)
    ]
    never_compliant_ids = [
        sample_id for sample_id, values in trajectories.items() if not any(values)
    ]
    swing_ids = [
        sample_id
        for sample_id, values in trajectories.items()
        if any(values) and not all(values)
    ]

    swing_breakdown = []
    for alpha in ALPHAS:
        swing_compliant = sum(
            bool(rows_by_alpha[alpha][sample_id]["compliance"])
            for sample_id in swing_ids
        )
        swing_breakdown.append(
            {
                "alpha": alpha,
                "swing_compliant": swing_compliant,
                "swing_resistant": len(swing_ids) - swing_compliant,
            }
        )

    total = len(reference_ids)
    return {
        "n_total": total,
        "always_compliant": {
            "count": len(always_compliant_ids),
            "pct": as_pct(len(always_compliant_ids) / total),
            "ci": build_rate_summary(
                len(always_compliant_ids),
                total,
                count_key="count",
                total_key="n_total",
            )["ci"],
        },
        "never_compliant": {
            "count": len(never_compliant_ids),
            "pct": as_pct(len(never_compliant_ids) / total),
            "ci": build_rate_summary(
                len(never_compliant_ids),
                total,
                count_key="count",
                total_key="n_total",
            )["ci"],
        },
        "swing": {
            "count": len(swing_ids),
            "pct": as_pct(len(swing_ids) / total),
            "ci": build_rate_summary(
                len(swing_ids),
                total,
                count_key="count",
                total_key="n_total",
            )["ci"],
        },
        "swing_breakdown": swing_breakdown,
    }


def build_binary_trajectory_effects(base_dir: Path, field: str) -> dict[str, Any]:
    rows_by_alpha = {
        alpha: {
            row["id"]: row for row in load_jsonl(base_dir / f"alpha_{alpha:.1f}.jsonl")
        }
        for alpha in ALPHAS
    }
    reference_ids = sorted(rows_by_alpha[ALPHAS[0]])
    trajectories = np.array(
        [
            [bool(rows_by_alpha[alpha][sample_id][field]) for alpha in ALPHAS]
            for sample_id in reference_ids
        ],
        dtype=bool,
    )
    return paired_bootstrap_curve_effects(
        trajectories, np.array(ALPHAS, dtype=float), noop_alpha=1.0
    )


def build_parse_failure_effects(base_dir: Path) -> dict[str, Any]:
    rows_by_alpha = {
        alpha: {
            row["id"]: row for row in load_jsonl(base_dir / f"alpha_{alpha:.1f}.jsonl")
        }
        for alpha in ALPHAS
    }
    reference_ids = sorted(rows_by_alpha[ALPHAS[0]])
    trajectories = np.array(
        [
            [rows_by_alpha[alpha][sample_id]["chosen"] is None for alpha in ALPHAS]
            for sample_id in reference_ids
        ],
        dtype=bool,
    )
    return paired_bootstrap_curve_effects(
        trajectories, np.array(ALPHAS, dtype=float), noop_alpha=1.0
    )


def build_selected_h_neuron_structure(
    coef: np.ndarray,
    total_ffn_neurons: int,
    selected_h_neurons: int,
) -> dict[str, Any]:
    if coef.shape[0] != total_ffn_neurons:
        raise ValueError(
            "Classifier feature width does not match reported total FFN neurons: "
            f"{coef.shape[0]} vs {total_ffn_neurons}"
        )
    if total_ffn_neurons % GEMMA_3_4B_LAYER_COUNT != 0:
        raise ValueError(
            "Total FFN neurons must divide cleanly into Gemma 3 4B layers: "
            f"{total_ffn_neurons}"
        )

    neurons_per_layer = total_ffn_neurons // GEMMA_3_4B_LAYER_COUNT
    positive_indices = np.flatnonzero(coef > 0)
    if len(positive_indices) != selected_h_neurons:
        raise ValueError(
            "Classifier positive-weight count does not match reported selected "
            f"H-neurons: {len(positive_indices)} vs {selected_h_neurons}"
        )

    positive_counts_by_layer = [0] * GEMMA_3_4B_LAYER_COUNT
    for index in positive_indices:
        positive_counts_by_layer[int(index // neurons_per_layer)] += 1

    def band(label: str, start_layer: int, end_layer: int) -> dict[str, Any]:
        count = sum(positive_counts_by_layer[start_layer : end_layer + 1])
        return {
            "label": label,
            "start_layer": start_layer,
            "end_layer": end_layer,
            "count": count,
            "pct": as_pct(count / selected_h_neurons),
        }

    top_positive_indices = positive_indices[
        np.argsort(coef[positive_indices])[::-1][:10]
    ]
    top_positive_neurons = [
        {
            "rank": rank,
            "layer": int(index // neurons_per_layer),
            "neuron": int(index % neurons_per_layer),
            "label": (
                f"L{int(index // neurons_per_layer)}:N{int(index % neurons_per_layer)}"
            ),
            "weight": round(float(coef[index]), 3),
        }
        for rank, index in enumerate(top_positive_indices, start=1)
    ]

    return {
        "n_layers": GEMMA_3_4B_LAYER_COUNT,
        "neurons_per_layer": neurons_per_layer,
        "positive_counts_by_layer": positive_counts_by_layer,
        "nonzero_layers": [
            {"layer": layer, "count": count}
            for layer, count in enumerate(positive_counts_by_layer)
            if count > 0
        ],
        "bands": {
            "early": band("early", 0, 10),
            "middle": band("middle", 11, 20),
            "late": band("late", 21, 33),
        },
        "top_positive_neurons": top_positive_neurons,
    }


def resolve_classifier_checkpoint_path(
    repo_root: Path,
    summary: dict[str, Any],
) -> Path:
    checkpoint_path = find_local_classifier_checkpoint_path(repo_root, summary)
    if checkpoint_path is not None:
        return checkpoint_path

    candidates = classifier_checkpoint_candidates(summary)
    raise FileNotFoundError(
        "No local classifier checkpoint found. Looked for: " + ", ".join(candidates)
    )


def classifier_checkpoint_candidates(summary: dict[str, Any]) -> list[str]:
    candidates = [
        summary.get("loaded_model_path"),
        "models/gemma3_4b_classifier_disjoint.pkl",
    ]
    return [
        candidate
        for candidate in candidates
        if isinstance(candidate, str) and candidate
    ]


def find_local_classifier_checkpoint_path(
    repo_root: Path,
    summary: dict[str, Any],
) -> Path | None:
    for candidate in classifier_checkpoint_candidates(summary):
        candidate_path = repo_root / candidate
        if candidate_path.exists():
            return candidate_path

    return None


def coefficient_sha256(coef: np.ndarray) -> str:
    normalized = np.ascontiguousarray(coef, dtype=np.float64)
    return hashlib.sha256(normalized.tobytes()).hexdigest()


def build_classifier_structure_summary_payload(repo_root: Path) -> dict[str, Any]:
    disjoint_summary_path = (
        repo_root / "data/gemma3_4b/pipeline/classifier_disjoint_summary.json"
    )
    summary = load_json(disjoint_summary_path)
    checkpoint_path = resolve_classifier_checkpoint_path(repo_root, summary)
    model = joblib.load(checkpoint_path)
    coef = np.asarray(model.coef_[0], dtype=float)
    structure = build_selected_h_neuron_structure(
        coef,
        summary["total_ffn_neurons"],
        summary["selected_h_neurons"],
    )

    return {
        "schema_version": 1,
        "generated_at": date.today().isoformat(),
        "generated_by": "scripts/export_site_data.py",
        "model": summary["model_path"],
        "model_path": checkpoint_path.relative_to(repo_root).as_posix(),
        "generation_script": "scripts/export_site_data.py",
        "source_files": [
            "data/gemma3_4b/pipeline/classifier_disjoint_summary.json",
            checkpoint_path.relative_to(repo_root).as_posix(),
        ],
        "selected_h_neurons": summary["selected_h_neurons"],
        "total_ffn_neurons": summary["total_ffn_neurons"],
        "coefficient_sha256": coefficient_sha256(coef),
        "structure": structure,
    }


def load_classifier_structure_summary(
    repo_root: Path,
    summary_path: Path = CLASSIFIER_STRUCTURE_SUMMARY_PATH,
) -> dict[str, Any]:
    return load_json(repo_root / summary_path)


def load_top_neuron_artifact_summary(
    repo_root: Path,
    summary_path: Path = TOP_NEURON_ARTIFACT_SUMMARY_PATH,
) -> dict[str, Any]:
    return load_json(repo_root / summary_path)


def validate_top_neuron_artifact_summary(
    summary: dict[str, Any],
    classifier_structure: dict[str, Any],
    selected_h_neurons: int,
) -> None:
    target = summary.get("target_neuron")
    if not isinstance(target, dict):
        raise ValueError("Tracked top-neuron artifact summary missing target_neuron")

    top_positive_neurons = classifier_structure.get("top_positive_neurons")
    if not isinstance(top_positive_neurons, list) or not top_positive_neurons:
        raise ValueError("Classifier structure missing top_positive_neurons")

    top_neuron = top_positive_neurons[0]
    if target.get("label") != top_neuron.get("label"):
        raise ValueError(
            "Tracked top-neuron artifact summary target label does not match the "
            "classifier structure top neuron"
        )
    if round(float(target.get("weight", 0.0)), 3) != round(
        float(top_neuron.get("weight", 0.0)), 3
    ):
        raise ValueError(
            "Tracked top-neuron artifact summary target weight does not match the "
            "classifier structure top neuron"
        )

    verdict = summary.get("verdict")
    if not isinstance(verdict, dict):
        raise ValueError("Tracked top-neuron artifact summary missing verdict")
    supporting_tests = verdict.get("supporting_tests")
    total_tests = verdict.get("total_tests")
    if not isinstance(supporting_tests, int) or not isinstance(total_tests, int):
        raise ValueError(
            "Tracked top-neuron artifact summary verdict counts must be integers"
        )
    if supporting_tests < 0 or supporting_tests > total_tests:
        raise ValueError(
            "Tracked top-neuron artifact summary supporting_tests must be between "
            "0 and total_tests"
        )
    ci_status = verdict.get("ci_status")
    if not isinstance(ci_status, str) or not ci_status.strip():
        raise ValueError(
            "Tracked top-neuron artifact summary verdict missing ci_status"
        )
    expected_total_tests = len(TOP_NEURON_ARTIFACT_TEST_SLUGS)
    if total_tests != expected_total_tests:
        raise ValueError(
            "Tracked top-neuron artifact summary verdict total_tests must match "
            f"the renderer contract ({expected_total_tests})"
        )

    tests = summary.get("tests")
    if not isinstance(tests, list) or len(tests) != total_tests:
        raise ValueError(
            "Tracked top-neuron artifact summary tests must match verdict total_tests"
        )
    expected_test_fields = {"slug", "label", "display_value", "threshold", "verdict"}
    slugs: set[str] = set()
    for test in tests:
        if not isinstance(test, dict):
            raise ValueError(
                "Tracked top-neuron artifact summary tests must contain objects"
            )
        missing_fields = expected_test_fields - set(test)
        if missing_fields:
            raise ValueError(
                "Tracked top-neuron artifact summary test missing fields: "
                + ", ".join(sorted(missing_fields))
            )
        slug = test["slug"]
        if slug in slugs:
            raise ValueError(
                "Tracked top-neuron artifact summary test slugs must be unique"
            )
        slugs.add(slug)
    expected_slugs = set(TOP_NEURON_ARTIFACT_TEST_SLUGS)
    if slugs != expected_slugs:
        details: list[str] = []
        missing_slugs = sorted(expected_slugs - slugs)
        unexpected_slugs = sorted(slugs - expected_slugs)
        if missing_slugs:
            details.append("missing: " + ", ".join(missing_slugs))
        if unexpected_slugs:
            details.append("unexpected: " + ", ".join(unexpected_slugs))
        raise ValueError(
            "Tracked top-neuron artifact summary tests must use the exact "
            "renderer slug set. " + "; ".join(details)
        )

    context = summary.get("distributed_detector_context")
    if not isinstance(context, dict):
        raise ValueError(
            "Tracked top-neuron artifact summary missing distributed_detector_context"
        )
    sparse_baseline = context.get("sparse_baseline")
    broader_detector = context.get("broader_detector")
    if not isinstance(sparse_baseline, dict) or not isinstance(broader_detector, dict):
        raise ValueError(
            "Tracked top-neuron artifact summary missing detector context entries"
        )
    if sparse_baseline.get("positive_neurons") != selected_h_neurons:
        raise ValueError(
            "Tracked top-neuron artifact summary sparse baseline count does not "
            "match the classifier selected_h_neurons"
        )
    if sparse_baseline.get("target_rank") != 1:
        raise ValueError(
            "Tracked top-neuron artifact summary sparse baseline must encode the "
            "paper-faithful C=1.0 rank as 1"
        )
    if broader_detector.get("positive_neurons", 0) <= selected_h_neurons:
        raise ValueError(
            "Tracked top-neuron artifact summary broader detector must be wider "
            "than the sparse baseline"
        )


def validate_classifier_structure_summary(
    repo_root: Path,
    summary_path: Path = CLASSIFIER_STRUCTURE_SUMMARY_PATH,
) -> None:
    tracked = load_classifier_structure_summary(repo_root, summary_path)
    expected = build_classifier_structure_summary_payload(repo_root)
    comparable_keys = (
        "schema_version",
        "generated_by",
        "model",
        "model_path",
        "generation_script",
        "source_files",
        "selected_h_neurons",
        "total_ffn_neurons",
        "coefficient_sha256",
        "structure",
    )
    mismatches = [
        key for key in comparable_keys if tracked.get(key) != expected.get(key)
    ]
    if not mismatches:
        return

    mismatch_lines = [
        f"{key}: tracked={tracked.get(key)!r} expected={expected.get(key)!r}"
        for key in mismatches
    ]
    raise ValueError(
        "Tracked classifier structure summary disagrees with local checkpoint:\n"
        + "\n".join(mismatch_lines)
    )


def validate_tracked_classifier_hash_against_local_checkpoint(
    repo_root: Path,
    tracked_summary: dict[str, Any],
    disjoint_summary: dict[str, Any],
) -> None:
    checkpoint_path = find_local_classifier_checkpoint_path(repo_root, disjoint_summary)
    if checkpoint_path is None:
        return

    model = joblib.load(checkpoint_path)
    coef = np.asarray(model.coef_[0], dtype=float)
    actual_sha256 = coefficient_sha256(coef)
    tracked_sha256 = tracked_summary.get("coefficient_sha256")
    if tracked_sha256 == actual_sha256:
        return

    raise ValueError(
        "Tracked classifier structure summary does not match the local disjoint "
        "checkpoint. Run "
        "`uv run python scripts/export_site_data.py "
        "--refresh-classifier-structure-summary` to refresh the tracked summary or "
        "`uv run python scripts/export_site_data.py "
        "--validate-classifier-structure-summary` to inspect the mismatch. "
        f"tracked coefficient_sha256={tracked_sha256!r} "
        f"local coefficient_sha256={actual_sha256!r} "
        f"checkpoint={checkpoint_path.relative_to(repo_root).as_posix()!r}"
    )


def build_classifier_site_payload(
    repo_root: Path,
    classifier_structure_summary_path: Path = CLASSIFIER_STRUCTURE_SUMMARY_PATH,
    top_neuron_artifact_summary_path: Path = TOP_NEURON_ARTIFACT_SUMMARY_PATH,
) -> dict[str, Any]:
    disjoint_summary_path = (
        repo_root / "data/gemma3_4b/pipeline/classifier_disjoint_summary.json"
    )
    overlap_summary_path = (
        repo_root / "data/gemma3_4b/pipeline/classifier_overlap_summary.json"
    )
    sae_summary_path = repo_root / "data/gemma3_4b/pipeline/classifier_sae_summary.json"
    qids_path = repo_root / "data/gemma3_4b/pipeline/test_qids_disjoint.json"
    summary = load_json(disjoint_summary_path)
    overlap_summary = load_json(overlap_summary_path)
    sae_summary = load_json(sae_summary_path)
    disjoint_qids = load_json(qids_path)
    tracked_structure_summary = load_classifier_structure_summary(
        repo_root, classifier_structure_summary_path
    )
    validate_tracked_classifier_hash_against_local_checkpoint(
        repo_root,
        tracked_structure_summary,
        summary,
    )
    selected_h_neuron_structure = tracked_structure_summary["structure"]
    top_neuron_artifact_summary = load_top_neuron_artifact_summary(
        repo_root, top_neuron_artifact_summary_path
    )
    validate_top_neuron_artifact_summary(
        top_neuron_artifact_summary,
        selected_h_neuron_structure,
        summary["selected_h_neurons"],
    )
    if tracked_structure_summary["selected_h_neurons"] != summary["selected_h_neurons"]:
        raise ValueError(
            "Tracked classifier structure summary selected_h_neurons does not match "
            "classifier_disjoint_summary.json"
        )
    if tracked_structure_summary["total_ffn_neurons"] != summary["total_ffn_neurons"]:
        raise ValueError(
            "Tracked classifier structure summary total_ffn_neurons does not match "
            "classifier_disjoint_summary.json"
        )
    evaluation = summary["evaluation"]
    overlap_evaluation = overlap_summary["evaluation"]
    sae_evaluation = sae_summary["evaluation"]
    disjoint_sampled_n = sum(len(ids) for ids in disjoint_qids.values())
    return {
        "schema_version": 2,
        "generated_at": date.today().isoformat(),
        "generated_by": "scripts/export_site_data.py",
        "model": "google/gemma-3-4b-it",
        "source_files": [
            "data/gemma3_4b/pipeline/classifier_disjoint_summary.json",
            "data/gemma3_4b/pipeline/classifier_overlap_summary.json",
            "data/gemma3_4b/pipeline/classifier_sae_summary.json",
            "data/gemma3_4b/pipeline/test_qids_disjoint.json",
            classifier_structure_summary_path.as_posix(),
            top_neuron_artifact_summary_path.as_posix(),
        ],
        "n_examples": evaluation["n_examples"],
        "n_positive": evaluation["n_positive"],
        "n_negative": evaluation["n_negative"],
        "selected_h_neurons": summary["selected_h_neurons"],
        "total_ffn_neurons": summary["total_ffn_neurons"],
        "selected_h_neuron_structure": selected_h_neuron_structure,
        "top_neuron_artifact_summary": top_neuron_artifact_summary,
        "selected_ratio_per_mille": summary["selected_ratio_per_mille"],
        "metrics": evaluation["metrics"],
        "bootstrap": evaluation["bootstrap"],
        "confusion_matrix": evaluation["confusion_matrix"],
        "sae": {
            "n_examples": sae_evaluation["n_examples"],
            "n_positive": sae_evaluation["n_positive"],
            "n_negative": sae_evaluation["n_negative"],
            "metrics": sae_evaluation["metrics"],
            "bootstrap": sae_evaluation["bootstrap"],
            "confusion_matrix": sae_evaluation["confusion_matrix"],
        },
        "overlap": {
            "n_examples": overlap_evaluation["n_examples"],
            "n_positive": overlap_evaluation["n_positive"],
            "n_negative": overlap_evaluation["n_negative"],
            "metrics": overlap_evaluation["metrics"],
            "bootstrap": overlap_evaluation["bootstrap"],
            "confusion_matrix": overlap_evaluation["confusion_matrix"],
        },
        "disjoint_sampled_n": disjoint_sampled_n,
        "disjoint_missing_activations": disjoint_sampled_n - evaluation["n_examples"],
        "disjoint_accuracy_drop_vs_overlap_pp": round(
            (
                overlap_evaluation["metrics"]["accuracy"]["estimate"]
                - evaluation["metrics"]["accuracy"]["estimate"]
            )
            * 100,
            1,
        ),
    }


def build_swing_characterization_payload(repo_root: Path) -> dict[str, Any]:
    summary_path = repo_root / "data/gemma3_4b/swing_characterization/summary.json"
    summary = load_json(summary_path)

    pop = summary["population_counts"]
    transitions = summary["transitions"]
    subtypes = transitions["subtype_counts"]
    alpha_stats = transitions["transition_alpha"]
    rc_vs_cr = transitions["rc_vs_cr_transition"]
    predictability = summary.get("structural_predictability", {})

    def compact_predictive_result(result: dict[str, Any]) -> dict[str, Any]:
        metrics = result["metrics"]
        permutation = result["permutation_test"]["metrics"]
        return {
            "n_samples": result["n_samples"],
            "n_positive": result["n_positive"],
            "n_negative": result["n_negative"],
            "auroc": metrics["auroc"],
            "balanced_accuracy": metrics["balanced_accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "permutation_test": permutation,
        }

    def build_predictability_interpretation(
        task_results: dict[str, Any],
    ) -> dict[str, str]:
        primary = task_results["all_ex_ante"]
        source_only = task_results["source_only"]
        structure_only = task_results["structure_only"]
        auroc = primary["auroc"]["estimate"]
        perm_p = primary["permutation_test"]["auroc"]["p_value"]
        source_auroc = source_only["auroc"]["estimate"]
        structure_auroc = structure_only["auroc"]["estimate"]

        if perm_p >= 0.05:
            return {
                "status": "no_detectable_signal",
                "headline": "Surface features are near chance at predicting swing status",
                "subtitle": "Held-out prediction on question structure, source, and topic does not beat a no-signal null by enough to support a strong structural claim.",
                "insight": "Descriptive shifts exist, but on the current feature set they do not translate into reliable swing classification.",
            }
        if auroc < 0.70:
            driver = "source dataset"
            if structure_auroc > source_auroc + 0.02:
                driver = "content structure"
            elif abs(structure_auroc - source_auroc) <= 0.02:
                driver = "a mix of source and structure"
            return {
                "status": "weak_signal",
                "headline": "Surface features carry a weak but non-zero swing signal",
                "subtitle": f"Held-out prediction beats the null, but only weakly; the current signal is driven mainly by {driver}.",
                "insight": "This supports a limited claim: some surface correlation exists, but it is too weak to treat swing status as structurally well-separated.",
            }
        return {
            "status": "strong_signal",
            "headline": "Surface features partially predict swing status",
            "subtitle": "Held-out prediction is materially above chance, so the page should not frame swing as structurally indistinguishable.",
            "insight": "Any mechanism story now needs to treat input structure as part of the explanation rather than just background variation.",
        }

    structural: dict[str, Any] = {}
    for proxy_name in (
        "context_length",
        "question_length",
        "standard_response_length",
        "word_overlap",
    ):
        proxy = summary.get("structural_proxies", {}).get(proxy_name) or summary.get(
            proxy_name
        )
        if proxy and "test" in proxy:
            test = proxy["test"]
            structural[proxy_name] = {
                "kruskal_p": test["kruskal_p"],
                "kruskal_H": test["kruskal_H"],
                "stats": {
                    pop_name: {
                        "mean": stats["mean"],
                        "median": stats.get("median"),
                        "n": stats["n"],
                    }
                    for pop_name, stats in proxy["stats"].items()
                },
            }

    source_test = summary.get("source_datasets", {}).get("test", {})
    topic_test = summary.get("topics", {}).get("test", {})
    structural_tasks = predictability.get("tasks", {})
    swing_predictability = structural_tasks.get("swing_vs_non_swing", {}).get(
        "feature_sets", {}
    )
    subtype_predictability = structural_tasks.get("r_to_c_vs_other_swing", {}).get(
        "feature_sets", {}
    )
    transition_histogram = {
        "alphas": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
        "series": {
            key: {
                "count": subtypes[key]["count"],
                "mean": alpha_stats[key]["mean"],
                "median": alpha_stats[key]["median"],
                "counts_by_alpha": alpha_stats[key]["counts_by_alpha"],
                "early_share_le_1_5": alpha_stats[key].get("early_share_le_1_5"),
            }
            for key in ("R→C", "C→R")
        },
    }

    payload: dict[str, Any] = {
        "schema_version": 2,
        "generated_at": date.today().isoformat(),
        "generated_by": "scripts/export_site_data.py",
        "model": "google/gemma-3-4b-it",
        "source_file": "data/gemma3_4b/swing_characterization/summary.json",
        "population": {
            "always_compliant": pop["always_compliant"],
            "never_compliant": pop["never_compliant"],
            "swing": pop["swing"],
            "total": pop["total"],
        },
        "subtypes": {
            key: {
                "count": val["count"],
                "proportion": val["proportion"],
                "pct": as_pct(val["proportion"]),
                "ci_95": val["ci_95"],
            }
            for key, val in subtypes.items()
        },
        "transition_alpha": {
            key: {
                "mean": val["mean"],
                "median": val["median"],
                "counts_by_alpha": val["counts_by_alpha"],
                "early_share_le_1_5": val.get("early_share_le_1_5"),
            }
            for key, val in alpha_stats.items()
        },
        "transition_histogram": transition_histogram,
        "rc_vs_cr_test": {
            "U": rc_vs_cr["U"],
            "p": rc_vs_cr["p"],
            "r": rc_vs_cr["r"],
        },
        "structural_proxies": structural,
        "source_datasets": {
            "chi2": source_test.get("chi2"),
            "p": source_test.get("p"),
            "cramers_v": source_test.get("cramers_v"),
        },
        "topics": {
            "chi2": topic_test.get("chi2"),
            "p": topic_test.get("p"),
            "cramers_v": topic_test.get("cramers_v"),
        },
    }

    if swing_predictability:
        compact_swing_predictability = {
            name: compact_predictive_result(result)
            for name, result in swing_predictability.items()
        }
        payload["structural_predictability"] = {
            "interpretation": build_predictability_interpretation(
                compact_swing_predictability
            ),
            "tasks": {
                "swing_vs_non_swing": compact_swing_predictability,
            },
        }
        if subtype_predictability:
            payload["structural_predictability"]["tasks"]["r_to_c_vs_other_swing"] = {
                name: compact_predictive_result(result)
                for name, result in subtype_predictability.items()
            }

    # Add LLM enrichment if available
    llm = summary.get("llm_enrichment")
    if llm:
        payload["llm_enrichment"] = compact_llm_enrichment(llm)

    return payload


def build_payload(repo_root: Path) -> dict[str, Any]:
    anti_dir = repo_root / "data/gemma3_4b/intervention/faitheval/experiment"
    standard_dir = (
        repo_root / "data/gemma3_4b/intervention/faitheval_standard/experiment"
    )
    negative_control_summary_path = (
        repo_root
        / "data/gemma3_4b/intervention/faitheval/control/comparison_summary.json"
    )
    random_slope_difference_summary_path = (
        repo_root
        / "data/gemma3_4b/intervention/faitheval/control/slope_difference_summary.json"
    )
    sae_slope_difference_summary_path = (
        repo_root
        / "data/gemma3_4b/intervention/faitheval_sae/control/slope_difference_summary.json"
    )
    anti_results = load_json(find_results_json(anti_dir))
    standard_results = load_json(find_results_json(standard_dir))
    negative_control_summary = load_json(negative_control_summary_path)
    random_slope_difference_summary = load_json(random_slope_difference_summary_path)
    sae_slope_difference_summary = load_json(sae_slope_difference_summary_path)
    remap_summary = load_json(
        standard_dir / "alpha_3.0_parse_failure_remap_summary.json"
    )
    parse_failure_points, parseable_subset_points = build_standard_format_points(
        standard_dir
    )
    anti_population = build_anti_population(anti_dir)
    anti_effects = build_binary_trajectory_effects(anti_dir, "compliance")
    standard_raw_effects = build_binary_trajectory_effects(standard_dir, "compliance")
    parse_failure_effects = build_parse_failure_effects(standard_dir)

    source_files = [
        "data/gemma3_4b/intervention/faitheval/experiment/results.json",
        "data/gemma3_4b/intervention/faitheval/experiment/alpha_0.0.jsonl",
        "data/gemma3_4b/intervention/faitheval/experiment/alpha_0.5.jsonl",
        "data/gemma3_4b/intervention/faitheval/experiment/alpha_1.0.jsonl",
        "data/gemma3_4b/intervention/faitheval/experiment/alpha_1.5.jsonl",
        "data/gemma3_4b/intervention/faitheval/experiment/alpha_2.0.jsonl",
        "data/gemma3_4b/intervention/faitheval/experiment/alpha_2.5.jsonl",
        "data/gemma3_4b/intervention/faitheval/experiment/alpha_3.0.jsonl",
        "data/gemma3_4b/intervention/faitheval_standard/experiment/results.json",
        "data/gemma3_4b/intervention/faitheval_standard/experiment/alpha_0.0.jsonl",
        "data/gemma3_4b/intervention/faitheval_standard/experiment/alpha_0.5.jsonl",
        "data/gemma3_4b/intervention/faitheval_standard/experiment/alpha_1.0.jsonl",
        "data/gemma3_4b/intervention/faitheval_standard/experiment/alpha_1.5.jsonl",
        "data/gemma3_4b/intervention/faitheval_standard/experiment/alpha_2.0.jsonl",
        "data/gemma3_4b/intervention/faitheval_standard/experiment/alpha_2.5.jsonl",
        "data/gemma3_4b/intervention/faitheval_standard/experiment/alpha_3.0.jsonl",
        "data/gemma3_4b/intervention/faitheval_standard/experiment/alpha_3.0_parse_failure_remap_summary.json",
        "data/gemma3_4b/intervention/faitheval/control/comparison_summary.json",
        "data/gemma3_4b/intervention/faitheval/control/slope_difference_summary.json",
        "data/gemma3_4b/intervention/faitheval_sae/control/slope_difference_summary.json",
    ]

    return {
        "schema_version": 1,
        "generated_at": date.today().isoformat(),
        "generated_by": "scripts/export_site_data.py",
        "benchmark": anti_results["benchmark"],
        "model": anti_results["model"],
        "classifier": anti_results["classifier"],
        "n_h_neurons": anti_results["n_h_neurons"],
        "ci_status": "available_with_partial_series",
        "alphas": ALPHAS,
        "provenance": {
            "source_files": source_files,
            "notes": [
                "Standard-prompt parse failures and parseable-subset rates are derived from committed per-alpha JSONL rows because results.json only stores raw compliance totals.",
                "Strict answer-text remap is committed only for standard prompt alpha=3.0, so corrected full-population scoring is partial rather than a full sweep.",
                "Standard-prompt population split is intentionally omitted until text-based rescoring exists across all alpha values.",
            ],
        },
        "series": {
            "anti_compliance": {
                "label": "Anti-compliance prompt",
                "prompt_style": "anti_compliance",
                "ci_status": "available",
                "effects": anti_effects,
                "points": build_rate_points(anti_results["results"]),
            },
            "standard_raw": {
                "label": "Standard prompt (raw MC-letter score)",
                "prompt_style": "standard",
                "ci_status": "available",
                "effects": standard_raw_effects,
                "points": build_rate_points(standard_results["results"]),
            },
            "standard_parseable_subset": {
                "label": "Standard prompt (parseable subset only)",
                "prompt_style": "standard",
                "ci_status": "available_conditional_metric",
                "status": "conditional_metric",
                "notes": [
                    "Computed on rows where the evaluator recovered an initial option letter.",
                    "This is not the same as a full-population correction.",
                ],
                "points": parseable_subset_points,
            },
            "standard_text_remap": {
                "label": "Standard prompt (strict answer-text remap)",
                "prompt_style": "standard",
                "ci_status": "available_single_alpha_only",
                "status": "single_alpha_only",
                "notes": [
                    "Committed strict answer-text remap exists only for alpha=3.0.",
                    "Use this as the current best full-population correction for alpha=3.0 only.",
                ],
                "by_alpha": {
                    "3.0": {
                        "alpha": 3.0,
                        "n_total": remap_summary["total_rows"],
                        "raw_compliance_count": remap_summary["raw_compliance_count"],
                        "raw_compliance_rate": remap_summary["raw_compliance_rate"],
                        "raw_compliance_pct": as_pct(
                            remap_summary["raw_compliance_rate"]
                        ),
                        "parse_failures": remap_summary["parse_failures"],
                        "strict_recovered_count": remap_summary[
                            "strict_recovered_count"
                        ],
                        "strict_recovered_rate_within_failures": remap_summary[
                            "strict_recovered_rate_within_failures"
                        ],
                        "strict_recovered_compliant_count": remap_summary[
                            "strict_recovered_compliant_count"
                        ],
                        "strict_recovered_rate_summary": build_rate_summary(
                            remap_summary["strict_recovered_count"],
                            remap_summary["parse_failures"],
                            count_key="count",
                            total_key="n_total",
                        ),
                        "strict_rescored_compliance_count": remap_summary[
                            "strict_rescored_compliance_count"
                        ],
                        "strict_rescored_compliance_rate": remap_summary[
                            "strict_rescored_compliance_rate"
                        ],
                        "strict_rescored_compliance_pct": as_pct(
                            remap_summary["strict_rescored_compliance_rate"]
                        ),
                        "strict_rescored_compliance_summary": build_rate_summary(
                            remap_summary["strict_rescored_compliance_count"],
                            remap_summary["total_rows"],
                            count_key="n_compliant",
                            total_key="n_total",
                        ),
                        "review_category_counts": remap_summary[
                            "review_category_counts"
                        ],
                        "unresolved_review_category_counts": remap_summary[
                            "unresolved_review_category_counts"
                        ],
                    }
                },
            },
        },
        "parse_failures": {
            "label": "Responses without recoverable option letter",
            "prompt_style": "standard",
            "ci_status": "available",
            "effects": parse_failure_effects,
            "points": parse_failure_points,
        },
        "negative_control": {
            "label": "Random 38-neuron control",
            "status": "available",
            "source_file": "data/gemma3_4b/intervention/faitheval/control/comparison_summary.json",
            "comparison_to_h_neurons": negative_control_summary[
                "comparison_to_h_neurons"
            ],
            "paired_slope_difference": {
                "label": "H-neurons minus random controls",
                "source_file": "data/gemma3_4b/intervention/faitheval/control/slope_difference_summary.json",
                **random_slope_difference_summary,
            },
        },
        "matched_readout_comparison": {
            "label": "H-neurons vs SAE h-features",
            "status": "available",
            "source_file": "data/gemma3_4b/intervention/faitheval_sae/control/slope_difference_summary.json",
            **sae_slope_difference_summary,
        },
        "population": {
            "anti_compliance": {
                "label": "Anti-compliance prompt",
                "ci_status": "available",
                "notes": [
                    "Derived from per-sample compliance trajectories across the committed anti-compliance alpha sweep.",
                ],
                **anti_population,
            },
            "standard": {
                "status": "withdrawn_pending_all_alpha_text_scoring",
                "notes": [
                    "Historical raw-parser population counts are intentionally not exported as current evidence.",
                ],
            },
        },
    }


def build_jailbreak_payload(repo_root: Path) -> dict[str, Any]:
    """Build the jailbreak intervention sweep payload for the site."""
    jailbreak_dir = repo_root / "data/gemma3_4b/intervention/jailbreak/experiment"
    faitheval_dir = repo_root / "data/gemma3_4b/intervention/faitheval/experiment"
    falseqa_dir = repo_root / "data/gemma3_4b/intervention/falseqa/experiment"
    available_evaluator_summary_path = (
        repo_root
        / "data/judge_validation/available_jailbreak_evaluator_comparison.json"
    )
    v2_comparison_path = (
        repo_root
        / "data/gemma3_4b/intervention/jailbreak/control/comparison_csv2_v2_summary.json"
    )
    v3_comparison_path = (
        repo_root
        / "data/gemma3_4b/intervention/jailbreak/control/comparison_csv2_v3_summary.json"
    )
    v3_bootstrap_path = (
        repo_root
        / "data/gemma3_4b/intervention/jailbreak/csv2_v3_evaluation/v3_slope_bootstrap.json"
    )
    paired_report_path = (
        repo_root / "notes/act3-reports/2026-04-13-v2-v3-paired-evaluator-comparison.md"
    )
    strongreject_results_path = (
        repo_root / "data/judge_validation/strongreject/results.json"
    )

    jailbreak_results = load_json(find_results_json(jailbreak_dir))
    faitheval_results = load_json(find_results_json(faitheval_dir))
    falseqa_results = load_json(find_results_json(falseqa_dir))
    available_evaluator_summary = (
        load_json(available_evaluator_summary_path)
        if available_evaluator_summary_path.exists()
        else None
    )
    v2_comparison = load_json(v2_comparison_path)
    v3_comparison = load_json(v3_comparison_path)
    v3_bootstrap = load_json(v3_bootstrap_path)
    strongreject_results = load_json(strongreject_results_path)
    paired_report = normalize_report_text(load_text(paired_report_path))
    holdout_summary = build_jailbreak_holdout_summary(repo_root)

    jailbreak_alphas = sorted_result_alphas(jailbreak_results["results"])

    # --- Aggregate points from results.json ---
    points = build_rate_points(jailbreak_results["results"], jailbreak_alphas)
    effects = jailbreak_results["effects"]["compliance_curve"]
    monotonicity = build_monotonicity_summary(points)

    # --- Template breakdown from JSONL files ---
    all_rows_by_alpha: dict[float, list[dict[str, Any]]] = {}
    for alpha in jailbreak_alphas:
        jsonl_path = jailbreak_dir / f"alpha_{alpha:.1f}.jsonl"
        if not jsonl_path.exists():
            raise FileNotFoundError(
                f"Missing jailbreak source file for exported alpha {alpha:.1f}: {jsonl_path}"
            )
        all_rows_by_alpha[alpha] = load_jsonl(jsonl_path)

    template_indices = sorted(
        {row["template_idx"] for rows in all_rows_by_alpha.values() for row in rows}
    )
    by_template: dict[str, Any] = {}
    for tidx in template_indices:
        template_points: list[dict[str, Any]] = []
        for alpha in all_rows_by_alpha:
            filtered = [
                row for row in all_rows_by_alpha[alpha] if row["template_idx"] == tidx
            ]
            n_total = len(filtered)
            n_compliant = sum(bool(row["compliance"]) for row in filtered)
            rate = n_compliant / n_total if n_total else 0.0
            summary = build_rate_summary(
                n_compliant, n_total, count_key="n_compliant", total_key="n_total"
            )
            template_points.append(
                {
                    "alpha": alpha,
                    "n_total": n_total,
                    "n_compliant": n_compliant,
                    "compliance_rate": rate,
                    "compliance_pct": as_pct(rate),
                    "ci": summary["ci"],
                }
            )
        by_template[f"T{tidx}"] = {
            "n_per_alpha": template_points[0]["n_total"],
            "points": template_points,
        }

    # --- Category breakdown at α=0.0 and α=3.0 ---
    categories = sorted({row["category"] for row in all_rows_by_alpha[0.0]})
    by_category: dict[str, Any] = {}
    for cat in categories:
        alpha_0_rows = [row for row in all_rows_by_alpha[0.0] if row["category"] == cat]
        alpha_3_rows = [row for row in all_rows_by_alpha[3.0] if row["category"] == cat]
        n0 = len(alpha_0_rows)
        c0 = sum(bool(row["compliance"]) for row in alpha_0_rows)
        r0 = c0 / n0 if n0 else 0.0
        s0 = build_rate_summary(c0, n0, count_key="n_compliant", total_key="n_total")
        n3 = len(alpha_3_rows)
        c3 = sum(bool(row["compliance"]) for row in alpha_3_rows)
        r3 = c3 / n3 if n3 else 0.0
        s3 = build_rate_summary(c3, n3, count_key="n_compliant", total_key="n_total")
        by_category[cat] = {
            "n_per_alpha": n0,
            "alpha_0": {
                "n_compliant": c0,
                "compliance_rate": r0,
                "compliance_pct": as_pct(r0),
                "ci": s0["ci"],
            },
            "alpha_3": {
                "n_compliant": c3,
                "compliance_rate": r3,
                "compliance_pct": as_pct(r3),
                "ci": s3["ci"],
            },
            "delta_0_to_3_pp": round((r3 - r0) * 100, 1),
        }

    # --- Cross-benchmark comparison ---
    def _benchmark_entry(
        name: str,
        results: dict[str, Any],
        *,
        negative_control: str,
        evaluator: str,
        generation: str,
    ) -> dict[str, Any]:
        res = results["results"]
        eff = results["effects"]["compliance_curve"]
        result_alphas = sorted_result_alphas(res)
        baseline = res[alpha_key(result_alphas[0])]
        endpoint = res[alpha_key(result_alphas[-1])]
        delta_noop_pp = eff.get("delta_noop_to_max_pp")
        entry = {
            "name": name,
            "n_per_alpha": baseline["n_total"],
            "baseline_pct": as_pct(baseline["compliance_rate"]),
            "endpoint_pct": as_pct(endpoint["compliance_rate"]),
            "delta_pp": eff["delta_0_to_max_pp"],
            "slope_pp_per_alpha": eff["slope_pp_per_alpha"],
            "negative_control": negative_control,
            "monotonic": all(
                eff["rates"][i] <= eff["rates"][i + 1]
                for i in range(len(eff["rates"]) - 1)
            ),
            "evaluator": evaluator,
            "generation": generation,
        }
        if delta_noop_pp is not None:
            entry["delta_noop_pp"] = delta_noop_pp
        return entry

    cross_benchmark: dict[str, Any] = {
        "benchmarks": [
            _benchmark_entry(
                "FaithEval",
                faitheval_results,
                negative_control="available",
                evaluator="MC-letter extraction",
                generation="greedy",
            ),
            _benchmark_entry(
                "FalseQA",
                falseqa_results,
                negative_control=FALSEQA_NEGATIVE_CONTROL_STATUS,
                evaluator="GPT-4o binary judge",
                generation="greedy",
            ),
            _benchmark_entry(
                "JailbreakBench",
                jailbreak_results,
                negative_control="partial_v2_seed0_only",
                evaluator="GPT-4o binary judge",
                generation=JAILBREAK_GENERATION_LABEL,
            ),
        ]
    }
    cross_benchmark["interpretation_caveat"] = (
        "FaithEval and FalseQA remain specificity-hardened benchmark stories. "
        "JailbreakBench now belongs in the measurement anchor: v2 shows a binary slope, "
        "v3 compresses the binary effect, and the publication-safe reading is a severity "
        "shift with evaluator dependence and incomplete control coverage."
    )

    full_set_stats = load_jailbreak_paired_evaluator_statistics(
        paired_report,
        available_evaluator_summary,
    )
    # --- Source files ---
    source_files = (
        [
            "data/gemma3_4b/intervention/jailbreak/experiment/results.json",
        ]
        + [
            f"data/gemma3_4b/intervention/jailbreak/experiment/alpha_{alpha:.1f}.jsonl"
            for alpha in jailbreak_alphas
        ]
        + [
            "data/gemma3_4b/intervention/faitheval/experiment/results.json",
            "data/gemma3_4b/intervention/falseqa/experiment/results.json",
            "data/gemma3_4b/intervention/jailbreak/control/comparison_csv2_v2_summary.json",
            "data/gemma3_4b/intervention/jailbreak/control/comparison_csv2_v3_summary.json",
            "data/gemma3_4b/intervention/jailbreak/csv2_v3_evaluation/v3_slope_bootstrap.json",
            "tests/gold_labels/jailbreak_cross_alpha_gold.jsonl",
            "data/judge_validation/csv2_v3/alpha_0.0.jsonl",
            "data/judge_validation/csv2_v3/alpha_1.5.jsonl",
            "data/judge_validation/csv2_v3/alpha_3.0.jsonl",
            "data/judge_validation/strongreject/results.json",
            "data/judge_validation/strongreject/alpha_0.0.jsonl",
            "data/judge_validation/strongreject/alpha_1.5.jsonl",
            "data/judge_validation/strongreject/alpha_3.0.jsonl",
            "notes/act3-reports/2026-04-13-v2-v3-paired-evaluator-comparison.md",
        ]
    )
    if available_evaluator_summary is not None:
        source_files.append(
            "data/judge_validation/available_jailbreak_evaluator_comparison.json"
        )

    paired_evaluator_comparison: dict[str, Any] = {
        "source_report": "notes/act3-reports/2026-04-13-v2-v3-paired-evaluator-comparison.md",
        "binary_v2_slope": pp_summary(
            float(full_set_stats["v2_yes_slope_pp_per_alpha"]["estimate_pp_per_alpha"]),
            float(full_set_stats["v2_yes_slope_pp_per_alpha"]["ci"][0]),
            float(full_set_stats["v2_yes_slope_pp_per_alpha"]["ci"][1]),
        ),
        "binary_v3_slope": pp_summary(
            float(full_set_stats["v3_yes_slope_pp_per_alpha"]["estimate_pp_per_alpha"]),
            float(full_set_stats["v3_yes_slope_pp_per_alpha"]["ci"][0]),
            float(full_set_stats["v3_yes_slope_pp_per_alpha"]["ci"][1]),
        ),
        "binary_v3_slope_bootstrap": v3_bootstrap["slope_pp_per_alpha"],
        "substantive_compliance_v3_slope": pp_summary(
            float(
                full_set_stats["v3_substantive_slope_pp_per_alpha"][
                    "estimate_pp_per_alpha"
                ]
            ),
            float(full_set_stats["v3_substantive_slope_pp_per_alpha"]["ci"][0]),
            float(full_set_stats["v3_substantive_slope_pp_per_alpha"]["ci"][1]),
        ),
        "v2_control_gap_pp_per_alpha": {
            "estimate": v2_comparison["comparison"]["gap_h_minus_random_mean_pp"],
        },
        "v3_single_seed_control_gap_pp_per_alpha": {
            "estimate": v3_comparison["comparison"]["gap_h_minus_random_mean_pp"],
        },
    }
    if available_evaluator_summary is not None:
        paired_evaluator_comparison["source_artifact"] = (
            "data/judge_validation/available_jailbreak_evaluator_comparison.json"
        )

    return {
        "schema_version": 1,
        "generated_at": date.today().isoformat(),
        "generated_by": "scripts/export_site_data.py",
        "benchmark": jailbreak_results["benchmark"],
        "model": "google/gemma-3-4b-it",
        "n_h_neurons": 38,
        "alphas": jailbreak_alphas,
        "provenance": {
            "source_files": source_files,
            "notes": [
                f"Jailbreak responses use stochastic generation (temperature={JAILBREAK_DECODING_TEMPERATURE:.1f}, do_sample=true), so per-item outcomes are not exactly reproducible.",
                "No negative control experiment has been run for this benchmark.",
                (
                    f"Spearman ρ for monotonicity is computed from the {len(points)} "
                    f"exported aggregate rates ({format_p_value(monotonicity['spearman_p'])})."
                ),
            ],
        },
        "aggregate": {
            "effects": {
                "rates": effects["rates"],
                "delta_0_to_max_pp": effects["delta_0_to_max_pp"],
                "slope_pp_per_alpha": effects["slope_pp_per_alpha"],
                "bootstrap": effects["bootstrap"],
            },
            "points": points,
            "monotonicity": monotonicity,
        },
        "measurement": {
            "public_claim": (
                "The current publication-safe jailbreak result is evaluator dependence: "
                "binary v2 shows a significant slope, binary v3 does not, and v3 recovers "
                "the signal mainly as a severity shift."
            ),
            "paired_evaluator_comparison": paired_evaluator_comparison,
            "strongreject_holdout": {
                **holdout_summary,
                "overall_accuracy": strongreject_results["accuracy"],
                "judge_model": strongreject_results["judge_model"],
            },
        },
        "by_template": by_template,
        "by_category": by_category,
        "stochastic_generation": {
            "sampling": {
                "do_sample": True,
                "temperature": JAILBREAK_DECODING_TEMPERATURE,
            },
            "caveat": "Per-item compliance outcomes are not exactly reproducible across runs due to stochastic decoding.",
        },
        "negative_control": {
            "status": "multi_seed_missing",
            "note": (
                "The legacy v2 analysis has a seed-0 specificity control and the v3 analysis "
                "has a single seed-1 control, but the current public framing still lacks a "
                "multi-seed control on the same evaluator/claim surface."
            ),
        },
        "cross_benchmark": cross_benchmark,
    }


def build_bridge_phase3_payload(repo_root: Path) -> dict[str, Any]:
    """Build the held-out Bridge Phase 3 summary payload."""
    baseline_results_path = (
        repo_root
        / "data/gemma3_4b/intervention/triviaqa_bridge/test_experiment/results.json"
    )
    iti_results_path = (
        repo_root
        / "data/gemma3_4b/intervention/triviaqa_bridge_iti_e0_paperfaithful_k12_first-3-tokens/test_experiment/results.json"
    )
    baseline_audit_path = (
        repo_root
        / "data/gemma3_4b/intervention/triviaqa_bridge/test_experiment/audit_stats.json"
    )
    iti_audit_path = (
        repo_root
        / "data/gemma3_4b/intervention/triviaqa_bridge_iti_e0_paperfaithful_k12_first-3-tokens/test_experiment/audit_stats.json"
    )
    report_path = (
        repo_root / "notes/act3-reports/2026-04-13-bridge-phase3-test-results.md"
    )

    baseline_results = load_json(baseline_results_path)
    iti_results = load_json(iti_results_path)
    baseline_audit = load_json(baseline_audit_path)
    iti_audit = load_json(iti_audit_path)
    report = normalize_report_text(load_text(report_path))

    primary_delta = require_report_match(
        report,
        r"adjudicated accuracy by (-?[0-9.]+) pp \[(-?[0-9.]+), (-?[0-9.]+)\][\s>]*\(CI excludes zero, McNemar[\s>]*p=([0-9.]+)\)",
        "bridge primary delta",
    )
    deterministic_delta = require_report_match(
        report,
        r"\| Deterministic accuracy \| (-?[0-9.]+)% \| \[(-?[0-9.]+)%, (-?[0-9.]+)%\] \| \*\*YES\*\* \|",
        "bridge deterministic delta",
    )
    attempt_delta = require_report_match(
        report,
        r"\| Attempt rate \| (-?[0-9.]+)% \| \[(-?[0-9.]+)%, (-?[0-9.]+)%\] \| \*\*YES\*\* \|",
        "bridge attempt delta",
    )
    flip_table = require_report_match(
        report,
        r"\| \*\*Base correct\*\* \| (\d+) \| (\d+) \| \d+ \|\n\| \*\*Base wrong\*\* \| (\d+) \| (\d+) \| \d+ \|",
        "bridge flip table",
    )
    wrong_entity = require_report_match(
        report,
        r"\| \*\*Wrong-entity substitution\*\* \| (\d+) \| (\d+)% \|",
        "bridge wrong-entity share",
    )
    verdict = require_report_match(
        report,
        r"\*\*Verdict: ([^.]+\.)",
        "bridge verdict",
    )

    baseline_row = baseline_results["results"]["1.0"]
    iti_row = iti_results["results"]["8.0"]

    return {
        "schema_version": 1,
        "generated_at": date.today().isoformat(),
        "generated_by": "scripts/export_site_data.py",
        "benchmark": "triviaqa_bridge_phase3",
        "model": "google/gemma-3-4b-it",
        "source_files": [
            "data/gemma3_4b/intervention/triviaqa_bridge/test_experiment/results.json",
            "data/gemma3_4b/intervention/triviaqa_bridge_iti_e0_paperfaithful_k12_first-3-tokens/test_experiment/results.json",
            "data/gemma3_4b/intervention/triviaqa_bridge/test_experiment/audit_stats.json",
            "data/gemma3_4b/intervention/triviaqa_bridge_iti_e0_paperfaithful_k12_first-3-tokens/test_experiment/audit_stats.json",
            "notes/act3-reports/2026-04-13-bridge-phase3-test-results.md",
        ],
        "verdict": verdict.group(1),
        "conditions": {
            "baseline": baseline_row,
            "iti_e0_alpha_8": iti_row,
        },
        "effects": {
            "adjudicated_accuracy_delta_pp": pp_summary(
                float(primary_delta.group(1)),
                float(primary_delta.group(2)),
                float(primary_delta.group(3)),
            ),
            "deterministic_accuracy_delta_pp": pp_summary(
                float(deterministic_delta.group(1)),
                float(deterministic_delta.group(2)),
                float(deterministic_delta.group(3)),
            ),
            "attempt_rate_delta_pp": pp_summary(
                float(attempt_delta.group(1)),
                float(attempt_delta.group(2)),
                float(attempt_delta.group(3)),
            ),
            "mcnemar_p": float(primary_delta.group(4)),
        },
        "flip_table": {
            "base_correct_iti_correct": int(flip_table.group(1)),
            "base_correct_iti_wrong": int(flip_table.group(2)),
            "base_wrong_iti_correct": int(flip_table.group(3)),
            "base_wrong_iti_wrong": int(flip_table.group(4)),
        },
        "failure_modes": {
            "wrong_entity_substitution": {
                "count": int(wrong_entity.group(1)),
                "share_pct": float(wrong_entity.group(2)),
            }
        },
        "audit": {
            "baseline_match_disagree_rate": baseline_audit[
                "audit_disagree_rate_matches"
            ],
            "iti_match_disagree_rate": iti_audit["audit_disagree_rate_matches"],
            "baseline_nonmatch_recovery_rate": baseline_audit[
                "audit_disagree_rate_nonmatches"
            ],
            "iti_nonmatch_recovery_rate": iti_audit["audit_disagree_rate_nonmatches"],
        },
    }


def build_d7_comparison_payload(repo_root: Path) -> dict[str, Any]:
    """Build the D7 full-500 benchmark-local comparison payload."""
    legacy_summary_path = (
        repo_root
        / "data/gemma3_4b/intervention/jailbreak_d7/full500_canonical/d7_csv2_report.json"
    )
    current_summary_path = (
        repo_root
        / "data/gemma3_4b/intervention/jailbreak_d7/full500_canonical/d7_full500_current_state_summary.json"
    )
    legacy_report_path = repo_root / "notes/act3-reports/2026-04-08-d7-full500-audit.md"
    current_report_path = require_existing_path(
        repo_root
        / "notes/act3-reports/2026-04-16-d7-full500-two-seed-current-state-audit.md",
        "D7 full500 current-state audit note",
    )

    current_summary = load_json(current_summary_path)
    historical_panel = current_summary["historical_panel"]
    current_panel = current_summary["current_panel"]
    artifact_status = current_summary["artifact_status"]
    paired_vs_baseline = {
        comparator: add_legacy_d7_comparison_aliases(metrics)
        for comparator, metrics in current_panel["paired_vs_baseline"].items()
    }
    random_seed1_status = artifact_status["causal_random_head_layer_matched"]["seed_1"]
    random_seed2_status = artifact_status["causal_random_head_layer_matched"]["seed_2"]
    probe_status = artifact_status["probe_locked"]
    random_seed1_condition = current_panel["conditions"]["random_layer_seed1"]
    random_seed2_condition = current_panel["conditions"].get("random_layer_seed2")
    probe_condition = current_panel["conditions"]["probe"]
    random_family_summary = copy.deepcopy(
        current_panel.get("random_layer_matched_family")
    )
    has_random_seed2 = random_seed2_status["status"] != "absent"
    random_seed1_errors = copy.deepcopy(random_seed1_condition["csv2_errors"])
    random_seed1_errors["clean_row_count"] = int(
        random_seed1_condition["strict_harmfulness_normalized_clean_rows"]["n"]
    )
    random_seed1_errors["total_row_count"] = int(
        random_seed1_condition["strict_harmfulness_normalized"]["n"]
    )
    probe_errors = copy.deepcopy(probe_condition["csv2_errors"])
    probe_errors["clean_row_count"] = int(
        probe_condition["strict_harmfulness_normalized_clean_rows"]["n"]
    )
    probe_errors["total_row_count"] = int(
        probe_condition["strict_harmfulness_normalized"]["n"]
    )
    current_state_caveat = (
        "Interpret D7 as benchmark-local supporting evidence on a clean CSV2 v3 "
        "panel: the mixed-ruler debt on the current panel has been cleared, and the "
        "remaining evaluator-error debt is now a small documented residual set after "
        "repair. The main live caveat is causal token-cap and quality debt rather "
        "than ruler debt."
    )
    top_level_caveat = (
        "D7 is a benchmark-local supporting result on a clean CSV2 v3 current-state "
        "panel: causal is still the strongest completed branch, and the remaining "
        "live caveats are causal token-cap and quality debt plus a small documented "
        "residual CSV2-error set rather than mixed-ruler debt."
    )
    control_availability = "available_single_seed_only"
    control_status = "single_seed_clean_panel"
    if has_random_seed2:
        control_availability = "available_two_seed_panel"
        control_status = "two_seed_clean_panel"

    current_state_namespace = {
        "date": "2026-04-16",
        "claim_status": "benchmark_local_supporting_clean_panel",
        "headline": (
            "D7 current state: the causal branch is still the strongest completed "
            "full-500 condition, outperforming the available probe and layer-matched "
            "random branches on the current normalized panel."
        ),
        "caveat": (current_state_caveat),
        "source_files": [
            "data/gemma3_4b/intervention/jailbreak_d7/full500_canonical/d7_full500_current_state_summary.json",
            str(current_report_path.relative_to(repo_root)),
        ],
        "mixed_ruler_status": {
            "status": "resolved_clean_v3_panel_small_residual_errors",
            "historical_panel_status": "historical_provenance_only",
            "description": current_panel["description"],
        },
        "control": {
            "availability": control_availability,
            "status": control_status,
            "seed_1": copy.deepcopy(random_seed1_status),
            "seed_2": copy.deepcopy(random_seed2_status),
        },
        "probe": {
            "status": probe_status["status"],
            "experiment_row_count": int(probe_status["experiment_row_count"]),
            "csv2_path": probe_status["csv2_path"],
            "included_in_current_claim": True,
        },
        "ruler_drift": copy.deepcopy(current_summary["ruler_drift"]),
        "random_seed1_csv2_error_burden": random_seed1_errors,
        "probe_csv2_error_burden": probe_errors,
        "current_panel": {
            "description": current_panel["description"],
            "conditions": copy.deepcopy(current_panel["conditions"]),
            "random_layer_matched_family": random_family_summary,
            "deltas_vs_baseline": {
                comparator: add_delta_aliases(metrics)
                for comparator, metrics in current_panel["paired_vs_baseline"].items()
            },
            "direct_random_layer_seed1_vs_causal": add_delta_aliases(
                current_panel["direct_comparisons"]["random_layer_seed1_vs_causal"]
            ),
            "direct_causal_vs_random_layer_seed1": invert_delta_metrics(
                current_panel["direct_comparisons"]["random_layer_seed1_vs_causal"]
            ),
            "direct_probe_vs_causal": add_delta_aliases(
                current_panel["direct_comparisons"]["probe_vs_causal"]
            ),
            "direct_causal_vs_probe": invert_delta_metrics(
                current_panel["direct_comparisons"]["probe_vs_causal"]
            ),
            "direct_probe_vs_random_layer_seed1": add_delta_aliases(
                current_panel["direct_comparisons"]["probe_vs_random_layer_seed1"]
            ),
            "direct_random_layer_seed1_vs_probe": invert_delta_metrics(
                current_panel["direct_comparisons"]["probe_vs_random_layer_seed1"]
            ),
        },
    }
    if random_seed2_condition is not None:
        current_state_namespace["current_panel"][
            "direct_random_layer_seed2_vs_causal"
        ] = add_delta_aliases(
            current_panel["direct_comparisons"]["random_layer_seed2_vs_causal"]
        )
        current_state_namespace["current_panel"][
            "direct_causal_vs_random_layer_seed2"
        ] = invert_delta_metrics(
            current_panel["direct_comparisons"]["random_layer_seed2_vs_causal"]
        )
        current_state_namespace["current_panel"][
            "direct_probe_vs_random_layer_seed2"
        ] = add_delta_aliases(
            current_panel["direct_comparisons"]["probe_vs_random_layer_seed2"]
        )
        current_state_namespace["current_panel"][
            "direct_random_layer_seed2_vs_probe"
        ] = invert_delta_metrics(
            current_panel["direct_comparisons"]["probe_vs_random_layer_seed2"]
        )
        current_state_namespace["current_panel"][
            "direct_random_layer_seed2_vs_random_layer_seed1"
        ] = add_delta_aliases(
            current_panel["direct_comparisons"][
                "random_layer_seed2_vs_random_layer_seed1"
            ]
        )
        current_state_namespace["current_panel"][
            "direct_random_layer_seed1_vs_random_layer_seed2"
        ] = invert_delta_metrics(
            current_panel["direct_comparisons"][
                "random_layer_seed2_vs_random_layer_seed1"
            ]
        )

    return {
        "schema_version": 1,
        "generated_at": date.today().isoformat(),
        "generated_by": "scripts/export_site_data.py",
        "benchmark": "jailbreak_d7_full500",
        "model": "google/gemma-3-4b-it",
        "source_files": [
            str(legacy_summary_path.relative_to(repo_root)),
            str(legacy_report_path.relative_to(repo_root)),
            str(current_summary_path.relative_to(repo_root)),
            str(current_report_path.relative_to(repo_root)),
            "notes/act3-reports/2026-04-14-d7-control-and-ruler-audit.md",
        ],
        "claim_status": "benchmark_local_supporting_evidence",
        "caveat": top_level_caveat,
        "headline": current_state_namespace["headline"],
        "conditions": {
            "baseline": add_legacy_d7_condition_aliases(
                current_panel["conditions"]["baseline"]
            ),
            "l1": add_legacy_d7_condition_aliases(current_panel["conditions"]["l1"]),
            "causal": add_legacy_d7_condition_aliases(
                current_panel["conditions"]["causal"]
            ),
        },
        "paired_vs_baseline": paired_vs_baseline,
        "direct_comparisons": {
            comparison: add_delta_aliases(metrics)
            for comparison, metrics in current_panel["direct_comparisons"].items()
        },
        "token_cap": {
            "causal_hits": int(
                current_panel["conditions"]["causal"]["token_cap"]["count"]
            ),
            "causal_total": int(
                current_panel["conditions"]["causal"]["token_cap"]["n"]
            ),
            "causal_share_pct": float(
                current_panel["conditions"]["causal"]["token_cap"]["share_pct"]
            ),
        },
        "historical_april_8": {
            "source_files": [
                str(legacy_summary_path.relative_to(repo_root)),
                str(legacy_report_path.relative_to(repo_root)),
            ],
            "conditions": copy.deepcopy(historical_panel["conditions"]),
            "paired_vs_baseline": {
                comparator: add_delta_aliases(metrics)
                for comparator, metrics in historical_panel[
                    "paired_vs_baseline"
                ].items()
            },
        },
        "current_state": current_state_namespace,
    }


def extract_markdown_count(path: Path, label: str) -> int:
    pattern = rf"\|\s*{re.escape(label)}\s*\|\s*(\d+)\s*\|"
    match = re.search(pattern, path.read_text())
    if not match:
        raise ValueError(f"Could not find markdown count '{label}' in {path}")
    return int(match.group(1))


def parse_pipeline_report_runtime(path: Path) -> dict[str, Any]:
    report = path.read_text()
    wall_time_match = re.search(
        r"\|\s+\*\*Total\*\*\s+\|\s+\*\*~([0-9.]+)\s+hours\*\*\s+\|",
        report,
    )
    api_cost_match = re.search(
        r"\|\s+\*\*Total\*\*\s+\|.*?\|\s+\*\*~\$([0-9.]+)\*\*\s+\|",
        report,
    )
    if not wall_time_match or not api_cost_match:
        raise ValueError(f"Could not parse runtime summary from {path}")

    wall_time_hours = float(wall_time_match.group(1))
    api_cost_usd = float(api_cost_match.group(1))
    return {
        "wall_time_hours": wall_time_hours,
        "wall_time_display": f"~{wall_time_hours:g} hours",
        "api_cost_usd": api_cost_usd,
        "api_cost_display": f"~${api_cost_usd:.2f}",
    }


def build_pipeline_site_payload(repo_root: Path) -> dict[str, Any]:
    consistency_samples_path = (
        repo_root / "data/gemma3_4b/pipeline/consistency_samples.jsonl"
    )
    batch_review_path = repo_root / "data/reviews/batch3500_review.md"
    answer_tokens_path = repo_root / "data/gemma3_4b/pipeline/answer_tokens.jsonl"
    train_qids_path = repo_root / "data/gemma3_4b/pipeline/train_qids.json"
    test_qids_path = repo_root / "data/gemma3_4b/pipeline/test_qids_disjoint.json"
    pipeline_report_path = repo_root / "data/gemma3_4b/pipeline/pipeline_report.md"
    classifier_summary = build_classifier_site_payload(repo_root)

    sampled_questions = count_jsonl_rows(consistency_samples_path)
    all_correct = extract_markdown_count(batch_review_path, "All-correct")
    all_incorrect = extract_markdown_count(batch_review_path, "All-incorrect")
    mixed = extract_markdown_count(batch_review_path, "Mixed")
    consistent_total = all_correct + all_incorrect
    extracted_answer_tokens = count_jsonl_rows(answer_tokens_path)
    train_qids = load_json(train_qids_path)
    test_qids = load_json(test_qids_path)
    runtime = parse_pipeline_report_runtime(pipeline_report_path)
    train_sampled_total = sum(len(ids) for ids in train_qids.values())
    disjoint_sampled_total = sum(len(ids) for ids in test_qids.values())

    return {
        "schema_version": 1,
        "generated_at": date.today().isoformat(),
        "generated_by": "scripts/export_site_data.py",
        "model": "google/gemma-3-4b-it",
        "source_files": [
            "data/gemma3_4b/pipeline/consistency_samples.jsonl",
            "data/reviews/batch3500_review.md",
            "data/gemma3_4b/pipeline/answer_tokens.jsonl",
            "data/gemma3_4b/pipeline/train_qids.json",
            "data/gemma3_4b/pipeline/test_qids_disjoint.json",
            "data/gemma3_4b/pipeline/pipeline_report.md",
            "data/gemma3_4b/pipeline/classifier_disjoint_summary.json",
        ],
        "counts": {
            "sampled_questions": sampled_questions,
            "all_correct": all_correct,
            "all_incorrect": all_incorrect,
            "mixed": mixed,
            "consistent_total": consistent_total,
            "extracted_answer_tokens": extracted_answer_tokens,
            "extraction_failures": consistent_total - extracted_answer_tokens,
            "train_sampled_total": train_sampled_total,
            "disjoint_sampled_total": disjoint_sampled_total,
            "disjoint_evaluated_total": classifier_summary["n_examples"],
            "disjoint_missing_activations": classifier_summary[
                "disjoint_missing_activations"
            ],
            "selected_h_neurons": classifier_summary["selected_h_neurons"],
            "total_ffn_neurons": classifier_summary["total_ffn_neurons"],
        },
        "ratios": {
            "consistent_share": consistent_total / sampled_questions,
            "extracted_share_of_consistent": extracted_answer_tokens / consistent_total,
            "train_share_of_sampled": train_sampled_total / sampled_questions,
            "disjoint_evaluated_share_of_sampled": classifier_summary["n_examples"]
            / sampled_questions,
        },
        "runtime": runtime,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("site/data/intervention_sweep.json"),
        help="Output path for exported site data.",
    )
    parser.add_argument(
        "--classifier-output",
        type=Path,
        default=Path("site/data/classifier_summary.json"),
        help="Output path for exported classifier site data.",
    )
    parser.add_argument(
        "--swing-output",
        type=Path,
        default=Path("site/data/swing_characterization.json"),
        help="Output path for exported swing characterization site data.",
    )
    parser.add_argument(
        "--pipeline-output",
        type=Path,
        default=Path("site/data/pipeline_summary.json"),
        help="Output path for exported pipeline summary site data.",
    )
    parser.add_argument(
        "--jailbreak-output",
        type=Path,
        default=Path("site/data/jailbreak_sweep.json"),
        help="Output path for exported jailbreak intervention sweep site data.",
    )
    parser.add_argument(
        "--bridge-output",
        type=Path,
        default=Path("site/data/bridge_phase3.json"),
        help="Output path for exported Bridge Phase 3 site data.",
    )
    parser.add_argument(
        "--d7-output",
        type=Path,
        default=Path("site/data/d7_comparison.json"),
        help="Output path for exported D7 comparison site data.",
    )
    parser.add_argument(
        "--classifier-structure-summary-output",
        type=Path,
        default=CLASSIFIER_STRUCTURE_SUMMARY_PATH,
        help="Output path for tracked classifier structure summary.",
    )
    parser.add_argument(
        "--refresh-classifier-structure-summary",
        action="store_true",
        help="Regenerate the tracked classifier structure summary from a local checkpoint.",
    )
    parser.add_argument(
        "--validate-classifier-structure-summary",
        action="store_true",
        help="Compare the tracked classifier structure summary against a local checkpoint.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    classifier_structure_summary_output_path = (
        repo_root / args.classifier_structure_summary_output
    )

    if args.refresh_classifier_structure_summary:
        classifier_structure_summary = build_classifier_structure_summary_payload(
            repo_root
        )
        classifier_structure_summary_output_path.parent.mkdir(
            parents=True, exist_ok=True
        )
        classifier_structure_summary_output_path.write_text(
            json.dumps(classifier_structure_summary, indent=2) + "\n"
        )

    if args.validate_classifier_structure_summary:
        validate_classifier_structure_summary(
            repo_root,
            args.classifier_structure_summary_output,
        )
        print("Tracked classifier structure summary matches local checkpoint.")
        return

    payload = build_payload(repo_root)
    output_path = repo_root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n")
    classifier_payload = build_classifier_site_payload(
        repo_root,
        args.classifier_structure_summary_output,
    )
    classifier_output_path = repo_root / args.classifier_output
    classifier_output_path.parent.mkdir(parents=True, exist_ok=True)
    classifier_output_path.write_text(json.dumps(classifier_payload, indent=2) + "\n")

    swing_summary_path = (
        repo_root / "data/gemma3_4b/swing_characterization/summary.json"
    )
    if swing_summary_path.exists():
        swing_payload = build_swing_characterization_payload(repo_root)
        swing_output_path = repo_root / args.swing_output
        swing_output_path.parent.mkdir(parents=True, exist_ok=True)
        swing_output_path.write_text(json.dumps(swing_payload, indent=2) + "\n")
    pipeline_payload = build_pipeline_site_payload(repo_root)
    pipeline_output_path = repo_root / args.pipeline_output
    pipeline_output_path.parent.mkdir(parents=True, exist_ok=True)
    pipeline_output_path.write_text(json.dumps(pipeline_payload, indent=2) + "\n")

    jailbreak_exp_dir = repo_root / "data/gemma3_4b/intervention/jailbreak/experiment"
    if any(jailbreak_exp_dir.glob("results*.json")):
        jailbreak_payload = build_jailbreak_payload(repo_root)
        jailbreak_output_path = repo_root / args.jailbreak_output
        jailbreak_output_path.parent.mkdir(parents=True, exist_ok=True)
        jailbreak_output_path.write_text(json.dumps(jailbreak_payload, indent=2) + "\n")

    bridge_payload = build_bridge_phase3_payload(repo_root)
    bridge_output_path = repo_root / args.bridge_output
    bridge_output_path.parent.mkdir(parents=True, exist_ok=True)
    bridge_output_path.write_text(json.dumps(bridge_payload, indent=2) + "\n")

    d7_payload = build_d7_comparison_payload(repo_root)
    d7_output_path = repo_root / args.d7_output
    d7_output_path.parent.mkdir(parents=True, exist_ok=True)
    d7_output_path.write_text(json.dumps(d7_payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
