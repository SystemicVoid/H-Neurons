"""Report the held-out FaithEval SAE utility-selector diagnostic bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from uncertainty import (
    DEFAULT_BOOTSTRAP_RESAMPLES,
    DEFAULT_BOOTSTRAP_SEED,
    build_rate_summary,
    paired_bootstrap_binary_rate_difference,
    paired_bootstrap_continuous_mean_difference_raw,
    percentile_interval,
)
from utils import (
    finish_run_provenance,
    fingerprint_ids,
    format_alpha_label,
    provenance_error_message,
    provenance_status_for_exception,
    start_run_provenance,
)


DEFAULT_N_RANDOM_SEEDS = 10
PATH_DRIFT_FAMILY = "matched_zero_dead"
BENCHMARK_TO_SECTION = {
    "faitheval": "heldout_compliance",
    "faitheval_anti_compliance_margin": "heldout_anti_compliance_margin",
}


def matched_random_seed_families(n_random_seeds: int) -> list[str]:
    if n_random_seeds < 0:
        raise ValueError("n_random_seeds must be non-negative")
    return [f"matched_random_seed_{seed}" for seed in range(n_random_seeds)]


def build_main_family_order(
    n_random_seeds: int,
    *,
    include_path_drift: bool = True,
) -> tuple[str, ...]:
    families = [
        "noop",
        "readout_selected",
        "utility_selected",
        *matched_random_seed_families(n_random_seeds),
    ]
    if include_path_drift:
        families.append(PATH_DRIFT_FAMILY)
    return tuple(families)


def _random_seed_index(family: str) -> int:
    return int(family.rsplit("_", 1)[-1])


def _alpha_by_family(family_order: list[str]) -> dict[str, float]:
    return {family: (1.0 if family == "noop" else 0.0) for family in family_order}


def _pairwise_delta_baselines(
    random_seed_families: list[str],
) -> tuple[tuple[str, str], ...]:
    return (
        ("readout_selected", "utility_minus_readout"),
        ("noop", "utility_minus_noop"),
        *((family, f"utility_minus_{family}") for family in random_seed_families),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--selector_dir",
        type=Path,
        default=Path(
            "data/gemma3_4b/intervention/faitheval_sae_utility_selector/selector"
        ),
    )
    parser.add_argument(
        "--heldout_root",
        type=Path,
        default=Path(
            "data/gemma3_4b/intervention/faitheval_sae_utility_selector/heldout"
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data/gemma3_4b/intervention/faitheval_sae_utility_selector"),
    )
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load_manifest_ids(path: Path) -> list[str]:
    payload = _load_json(path)
    ids = payload.get("ids")
    if not isinstance(ids, list) or not ids:
        raise ValueError(f"Sample manifest {path} is missing a non-empty ids list")
    return [str(sample_id) for sample_id in ids]


def _alpha_path(root: Path, alpha: float) -> Path:
    return root / f"alpha_{format_alpha_label(alpha)}.jsonl"


def _rows_by_id(
    rows: list[dict[str, Any]], *, context: str
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        sample_id = str(row["id"])
        if sample_id in out:
            raise ValueError(f"{context}: duplicate sample id {sample_id!r}")
        out[sample_id] = row
    return out


def _ordered_compliance_array(
    rows: list[dict[str, Any]],
    *,
    expected_ids: list[str],
    benchmark: str,
    family: str,
) -> np.ndarray:
    context = f"{benchmark}/{family}"
    rows_by_id = _rows_by_id(rows, context=context)
    expected_set = set(expected_ids)
    actual_set = set(rows_by_id)
    if actual_set != expected_set:
        missing = sorted(expected_set - actual_set)
        extra = sorted(actual_set - expected_set)
        problems: list[str] = []
        if missing:
            problems.append(f"missing ids={missing[:5]!r}")
        if extra:
            problems.append(f"unexpected ids={extra[:5]!r}")
        details = ", ".join(problems) if problems else "id mismatch"
        raise ValueError(
            "Held-out sample-ID parity failed for "
            f"{context}: expected selector/test_manifest.json ids ({details})"
        )
    return np.asarray(
        [bool(rows_by_id[sample_id].get("compliance")) for sample_id in expected_ids],
        dtype=bool,
    )


def _ordered_metric_array(
    rows: list[dict[str, Any]],
    *,
    expected_ids: list[str],
    benchmark: str,
    family: str,
    metric_name: str,
) -> np.ndarray:
    context = f"{benchmark}/{family}"
    rows_by_id = _rows_by_id(rows, context=context)
    expected_set = set(expected_ids)
    actual_set = set(rows_by_id)
    if actual_set != expected_set:
        missing = sorted(expected_set - actual_set)
        extra = sorted(actual_set - expected_set)
        problems: list[str] = []
        if missing:
            problems.append(f"missing ids={missing[:5]!r}")
        if extra:
            problems.append(f"unexpected ids={extra[:5]!r}")
        details = ", ".join(problems) if problems else "id mismatch"
        raise ValueError(
            "Held-out sample-ID parity failed for "
            f"{context}: expected selector/test_manifest.json ids ({details})"
        )

    values: list[float] = []
    for sample_id in expected_ids:
        row = rows_by_id[sample_id]
        row_metric_name = str(row.get("metric_name") or "")
        if row_metric_name != metric_name:
            raise ValueError(
                f"{context}: expected metric_name={metric_name!r}, "
                f"got {row_metric_name!r} for sample {sample_id!r}"
            )
        if "metric_value" not in row:
            raise ValueError(f"{context}: sample {sample_id!r} is missing metric_value")
        values.append(float(row["metric_value"]))
    return np.asarray(values, dtype=float)


def _bootstrap_mean_summary(values: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError("Mean summary expects a one-dimensional array")
    if len(arr) == 0:
        return {
            "n": 0,
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
    samples = np.empty(DEFAULT_BOOTSTRAP_RESAMPLES, dtype=float)
    for sample_idx in range(DEFAULT_BOOTSTRAP_RESAMPLES):
        indices = rng.choice(len(arr), size=len(arr), replace=True)
        samples[sample_idx] = float(arr[indices].mean())

    return {
        "n": int(len(arr)),
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


def _infer_random_seed_families(
    selector_summary: dict[str, Any],
    heldout_rows_by_benchmark: dict[str, dict[str, list[dict[str, Any]]]] | None = None,
) -> list[str]:
    if heldout_rows_by_benchmark:
        shared: set[str] | None = None
        for benchmark_rows in heldout_rows_by_benchmark.values():
            benchmark_families = {
                family
                for family in benchmark_rows
                if family.startswith("matched_random_seed_")
            }
            shared = (
                benchmark_families if shared is None else shared & benchmark_families
            )
        if shared:
            return sorted(shared, key=_random_seed_index)

    controls = selector_summary.get("matched_random_controls")
    if isinstance(controls, dict):
        seed_families = controls.get("seed_families")
        if isinstance(seed_families, dict) and seed_families:
            return sorted(
                [str(family) for family in seed_families],
                key=_random_seed_index,
            )
        n_random_seeds = controls.get("n_random_seeds")
        if isinstance(n_random_seeds, int):
            return matched_random_seed_families(n_random_seeds)

    return matched_random_seed_families(DEFAULT_N_RANDOM_SEEDS)


def _include_path_drift_family(
    selector_summary: dict[str, Any],
    heldout_rows_by_benchmark: dict[str, dict[str, list[dict[str, Any]]]] | None = None,
) -> bool:
    if heldout_rows_by_benchmark:
        if all(
            PATH_DRIFT_FAMILY in benchmark_rows
            for benchmark_rows in heldout_rows_by_benchmark.values()
        ):
            return True
    return isinstance(selector_summary.get("path_drift_control"), dict)


def _bundle_config(
    selector_summary: dict[str, Any],
    heldout_rows_by_benchmark: dict[str, dict[str, list[dict[str, Any]]]] | None = None,
) -> dict[str, Any]:
    random_seed_families = _infer_random_seed_families(
        selector_summary,
        heldout_rows_by_benchmark,
    )
    family_order = list(
        build_main_family_order(
            len(random_seed_families),
            include_path_drift=_include_path_drift_family(
                selector_summary,
                heldout_rows_by_benchmark,
            ),
        )
    )
    if random_seed_families:
        family_order = [
            "noop",
            "readout_selected",
            "utility_selected",
            *random_seed_families,
            *([PATH_DRIFT_FAMILY] if PATH_DRIFT_FAMILY in family_order else []),
        ]
    alpha_by_family = _alpha_by_family(family_order)
    return {
        "family_order": family_order,
        "alpha_by_family": alpha_by_family,
        "random_seed_families": random_seed_families,
        "pairwise_delta_baselines": _pairwise_delta_baselines(random_seed_families),
    }


def _seed_mean_summary(
    point_estimates: np.ndarray,
    per_item_differences: np.ndarray,
    *,
    scale: float,
) -> dict[str, Any]:
    estimates = np.asarray(point_estimates, dtype=float)
    per_item = np.asarray(per_item_differences, dtype=float)
    if estimates.ndim != 1:
        raise ValueError("point_estimates must be one-dimensional")
    if per_item.ndim != 2 or per_item.shape[0] != len(estimates):
        raise ValueError("per_item_differences must be 2D with one row per seed")

    rng = np.random.default_rng(DEFAULT_BOOTSTRAP_SEED)

    naive_samples = np.empty(DEFAULT_BOOTSTRAP_RESAMPLES, dtype=float)
    for sample_idx in range(DEFAULT_BOOTSTRAP_RESAMPLES):
        indices = rng.choice(len(estimates), size=len(estimates), replace=True)
        naive_samples[sample_idx] = float(estimates[indices].mean())

    avg_per_item = per_item.mean(axis=0) * scale
    nested_samples = np.empty(DEFAULT_BOOTSTRAP_RESAMPLES, dtype=float)
    for sample_idx in range(DEFAULT_BOOTSTRAP_RESAMPLES):
        indices = rng.choice(len(avg_per_item), size=len(avg_per_item), replace=True)
        nested_samples[sample_idx] = float(avg_per_item[indices].mean())

    return {
        "n_seeds": int(len(estimates)),
        "estimate": float(estimates.mean()) if len(estimates) else 0.0,
        "seed_sd": (float(np.std(estimates, ddof=1)) if len(estimates) > 1 else 0.0),
        "seed_range": {
            "min": float(estimates.min()) if len(estimates) else 0.0,
            "max": float(estimates.max()) if len(estimates) else 0.0,
        },
        "naive_seed_bootstrap": {
            "ci": percentile_interval(
                naive_samples,
                method="bootstrap_percentile_seed_mean",
            ).to_dict(),
            "bootstrap": {
                "n_resamples": int(DEFAULT_BOOTSTRAP_RESAMPLES),
                "seed": int(DEFAULT_BOOTSTRAP_SEED),
                "resampling": "iid_seed_estimates",
                "interval": "percentile",
            },
        },
        "nested_bootstrap": {
            "ci": percentile_interval(
                nested_samples,
                method="bootstrap_percentile_nested_paired_seed_mean",
            ).to_dict(),
            "bootstrap": {
                "n_resamples": int(DEFAULT_BOOTSTRAP_RESAMPLES),
                "seed": int(DEFAULT_BOOTSTRAP_SEED),
                "resampling": (
                    "paired_by_sample_id_joint_across_utility_and_all_random_seeds"
                ),
                "interval": "percentile",
            },
        },
        "primary_ci_method": "nested_bootstrap",
        "headline_supporting_ci_method": "naive_seed_bootstrap",
    }


def _across_seed_summary(
    *,
    random_seed_families: list[str],
    delta_by_key: dict[str, Any],
    utility_values: np.ndarray,
    baseline_by_family: dict[str, np.ndarray],
    scale: float,
) -> dict[str, Any]:
    per_seed: list[dict[str, Any]] = []
    point_estimates: list[float] = []
    per_item_differences: list[np.ndarray] = []
    for family in random_seed_families:
        delta_key = f"utility_minus_{family}"
        paired_delta = delta_by_key[delta_key]
        per_seed.append(
            {
                "seed": _random_seed_index(family),
                "family": family,
                "delta_key": delta_key,
                "paired_delta": paired_delta,
            }
        )
        if "estimate_pp" in paired_delta:
            point_estimates.append(float(paired_delta["estimate_pp"]))
        else:
            point_estimates.append(float(paired_delta["estimate"]))
        per_item_differences.append(
            np.asarray(utility_values, dtype=float)
            - np.asarray(baseline_by_family[family], dtype=float)
        )

    return {
        "seed_family_order": list(random_seed_families),
        "per_seed": per_seed,
        "seed_mean_summary": _seed_mean_summary(
            np.asarray(point_estimates, dtype=float),
            np.asarray(per_item_differences, dtype=float),
            scale=scale,
        ),
    }


def _build_measurement_section(
    *,
    benchmark: str,
    rows_by_family: dict[str, list[dict[str, Any]]],
    ordered_ids: list[str],
    config: dict[str, Any],
) -> dict[str, Any]:
    compliance_by_family: dict[str, np.ndarray] = {}
    families: dict[str, Any] = {}

    for family in config["family_order"]:
        compliance = _ordered_compliance_array(
            rows_by_family[family],
            expected_ids=ordered_ids,
            benchmark=benchmark,
            family=family,
        )
        compliance_by_family[family] = compliance
        families[family] = build_rate_summary(
            int(compliance.sum()),
            int(len(compliance)),
            count_key="n_compliant",
            total_key="n_total",
        )

    paired_deltas_pp: dict[str, Any] = {}
    utility = compliance_by_family["utility_selected"]
    for baseline_family, delta_key in config["pairwise_delta_baselines"]:
        paired_deltas_pp[delta_key] = paired_bootstrap_binary_rate_difference(
            compliance_by_family[baseline_family],
            utility,
        )

    return {
        "benchmark": benchmark,
        "prompt_style": "anti_compliance",
        "metric_name": "compliance",
        "delta_units": "percentage_points",
        "family_order": list(config["family_order"]),
        "alpha_by_family": dict(config["alpha_by_family"]),
        "families": families,
        "paired_deltas_pp": paired_deltas_pp,
        "across_seeds": _across_seed_summary(
            random_seed_families=config["random_seed_families"],
            delta_by_key=paired_deltas_pp,
            utility_values=utility.astype(float),
            baseline_by_family={
                family: compliance_by_family[family]
                for family in config["random_seed_families"]
            },
            scale=100.0,
        ),
    }


def _build_margin_measurement_section(
    *,
    benchmark: str,
    rows_by_family: dict[str, list[dict[str, Any]]],
    ordered_ids: list[str],
    config: dict[str, Any],
) -> dict[str, Any]:
    metric_by_family: dict[str, np.ndarray] = {}
    families: dict[str, Any] = {}

    for family in config["family_order"]:
        values = _ordered_metric_array(
            rows_by_family[family],
            expected_ids=ordered_ids,
            benchmark=benchmark,
            family=family,
            metric_name="misleading_minus_preferred_logprob_margin",
        )
        metric_by_family[family] = values
        families[family] = _bootstrap_mean_summary(values)

    paired_deltas: dict[str, Any] = {}
    utility = metric_by_family["utility_selected"]
    for baseline_family, delta_key in config["pairwise_delta_baselines"]:
        paired_deltas[delta_key] = paired_bootstrap_continuous_mean_difference_raw(
            metric_by_family[baseline_family],
            utility,
        )

    section = {
        "benchmark": benchmark,
        "prompt_style": "anti_compliance",
        "metric_name": "misleading_minus_preferred_logprob_margin",
        "margin_definition": "logp(counterfactual_key) - logp(preferred_key)",
        "delta_units": "logprob_margin",
        "family_order": list(config["family_order"]),
        "alpha_by_family": dict(config["alpha_by_family"]),
        "families": families,
        "paired_deltas": paired_deltas,
        "across_seeds": _across_seed_summary(
            random_seed_families=config["random_seed_families"],
            delta_by_key=paired_deltas,
            utility_values=utility,
            baseline_by_family={
                family: metric_by_family[family]
                for family in config["random_seed_families"]
            },
            scale=1.0,
        ),
    }
    if PATH_DRIFT_FAMILY in metric_by_family:
        section["path_drift"] = {
            "family": PATH_DRIFT_FAMILY,
            "baseline_family": "noop",
            "paired_delta": paired_bootstrap_continuous_mean_difference_raw(
                metric_by_family["noop"],
                metric_by_family[PATH_DRIFT_FAMILY],
            ),
            "dead_family_mean": families[PATH_DRIFT_FAMILY],
            "noop_mean": families["noop"],
        }
    return section


def _selector_calibration_gap(selector_summary: dict[str, Any]) -> Any:
    utility_family = selector_summary.get("families", {}).get("utility_selected", {})
    if "calibration_to_heldout_gap" in utility_family:
        return utility_family["calibration_to_heldout_gap"]
    return selector_summary.get("calibration_to_heldout_gap")


def _selector_design(selector_summary: dict[str, Any]) -> dict[str, Any]:
    design = selector_summary.get("selector_design")
    if isinstance(design, dict):
        return design
    return {}


def build_heldout_summary(
    *,
    selector_summary: dict[str, Any],
    ordered_ids: list[str],
    heldout_rows_by_benchmark: dict[str, dict[str, list[dict[str, Any]]]],
) -> dict[str, Any]:
    config = _bundle_config(selector_summary, heldout_rows_by_benchmark)
    selector_design = _selector_design(selector_summary)
    utility_family = selector_summary["families"]["utility_selected"]
    readout_family = selector_summary["families"]["readout_selected"]
    overlap = selector_summary["family_overlap"]["utility_selected_vs_readout_selected"]
    candidate_pool = selector_summary.get("candidate_pool", {})
    return {
        "schema_version": "faitheval_sae_utility_selector_report/v5",
        "n_samples": len(ordered_ids),
        "sample_ids_fingerprint": fingerprint_ids(ordered_ids),
        "heldout_compliance": _build_measurement_section(
            benchmark="faitheval",
            rows_by_family=heldout_rows_by_benchmark["faitheval"],
            ordered_ids=ordered_ids,
            config=config,
        ),
        "heldout_anti_compliance_margin": _build_margin_measurement_section(
            benchmark="faitheval_anti_compliance_margin",
            rows_by_family=heldout_rows_by_benchmark[
                "faitheval_anti_compliance_margin"
            ],
            ordered_ids=ordered_ids,
            config=config,
        ),
        "selector_design": selector_design,
        "selector_diagnostics": {
            "candidate_pool_n": candidate_pool.get("n_features"),
            "candidate_pool_layer_histogram": candidate_pool.get("layer_histogram"),
            "candidate_pool_weight_sign_counts": candidate_pool.get(
                "weight_sign_counts"
            ),
            "selected_k": utility_family["k"],
            "utility_layer_histogram": utility_family["layer_histogram"],
            "readout_layer_histogram": readout_family["layer_histogram"],
            "utility_weight_sign_counts": utility_family.get("weight_sign_counts"),
            "readout_weight_sign_counts": readout_family.get("weight_sign_counts"),
            "overlap_with_readout": overlap,
            "outside_old_shortlist_count": utility_family["outside_old_shortlist"][
                "count"
            ],
            "outside_old_shortlist_fraction": utility_family["outside_old_shortlist"][
                "fraction"
            ],
            "calibration_to_heldout_gap": _selector_calibration_gap(selector_summary),
            "layer_coverage_note": selector_design.get("layer_coverage_note"),
            "target_families": selector_design.get("target_families"),
        },
    }


def _format_rate(stats: dict[str, Any]) -> str:
    return f"{stats['estimate']:.4f} ({stats['n_compliant']}/{stats['n_total']})"


def _format_mean(stats: dict[str, Any]) -> str:
    return (
        f"{stats['estimate']:+.4f} "
        f"[{stats['ci']['lower']:+.4f}, {stats['ci']['upper']:+.4f}]"
    )


def _format_pair_delta(delta: dict[str, Any]) -> str:
    if "estimate_pp" in delta:
        return (
            f"{delta['estimate_pp']:+.3f} pp "
            f"[{delta['ci_pp']['lower']:+.3f}, {delta['ci_pp']['upper']:+.3f}]"
        )
    return (
        f"{delta['estimate']:+.4f} "
        f"[{delta['ci']['lower']:+.4f}, {delta['ci']['upper']:+.4f}]"
    )


def _format_across_seed_line(section: dict[str, Any]) -> str:
    seed_summary = section["across_seeds"]["seed_mean_summary"]
    if section["metric_name"] == "compliance":
        fmt = (
            "range={min:+.3f}..{max:+.3f} pp; "
            "seed_mean={estimate:+.3f} pp "
            "[{lower:+.3f}, {upper:+.3f}] nested primary; "
            "naive seed bootstrap "
            "[{naive_lower:+.3f}, {naive_upper:+.3f}]"
        )
    else:
        fmt = (
            "range={min:+.4f}..{max:+.4f}; "
            "seed_mean={estimate:+.4f} "
            "[{lower:+.4f}, {upper:+.4f}] nested primary; "
            "naive seed bootstrap "
            "[{naive_lower:+.4f}, {naive_upper:+.4f}]"
        )
    return fmt.format(
        min=seed_summary["seed_range"]["min"],
        max=seed_summary["seed_range"]["max"],
        estimate=seed_summary["estimate"],
        lower=seed_summary["nested_bootstrap"]["ci"]["lower"],
        upper=seed_summary["nested_bootstrap"]["ci"]["upper"],
        naive_lower=seed_summary["naive_seed_bootstrap"]["ci"]["lower"],
        naive_upper=seed_summary["naive_seed_bootstrap"]["ci"]["upper"],
    )


def _format_family_summary(
    families: dict[str, Any],
    *,
    formatter,
) -> str:
    ordered_families = ["noop", "readout_selected", "utility_selected"]
    if PATH_DRIFT_FAMILY in families:
        ordered_families.append(PATH_DRIFT_FAMILY)
    return "; ".join(
        f"{family}={formatter(families[family])}" for family in ordered_families
    )


def build_audit_note(summary: dict[str, Any]) -> str:
    diagnostics = summary["selector_diagnostics"]
    target_families = diagnostics.get("target_families") or {}
    compliance = summary["heldout_compliance"]
    margin = summary["heldout_anti_compliance_margin"]
    lines = [
        "# FaithEval SAE Utility Selector Audit",
        "",
        (
            "- Held-out diagnostic bundle on a single locked test manifest "
            f"(n={summary['n_samples']})."
        ),
        (
            "- FaithEval families: "
            f"{_format_family_summary(compliance['families'], formatter=_format_rate)}"
        ),
        (
            "- FaithEval paired deltas: "
            f"utility_minus_readout={_format_pair_delta(compliance['paired_deltas_pp']['utility_minus_readout'])}; "
            f"utility_minus_noop={_format_pair_delta(compliance['paired_deltas_pp']['utility_minus_noop'])}"
        ),
        (
            "- FaithEval random-null across seeds: "
            f"{_format_across_seed_line(compliance)}"
        ),
        (
            "- FaithEval anti-compliance margin families: "
            f"{_format_family_summary(margin['families'], formatter=_format_mean)}"
        ),
        (
            "- FaithEval anti-compliance margin paired deltas: "
            f"utility_minus_readout={_format_pair_delta(margin['paired_deltas']['utility_minus_readout'])}; "
            f"utility_minus_noop={_format_pair_delta(margin['paired_deltas']['utility_minus_noop'])}"
        ),
        (
            "- FaithEval anti-compliance margin random-null across seeds: "
            f"{_format_across_seed_line(margin)}"
        ),
    ]
    path_drift = margin.get("path_drift")
    if isinstance(path_drift, dict):
        lines.append(
            "- Path drift (matched_zero_dead - noop): "
            f"{_format_pair_delta(path_drift['paired_delta'])}; "
            f"matched_zero_dead_mean={_format_mean(path_drift['dead_family_mean'])}; "
            f"noop_mean={_format_mean(path_drift['noop_mean'])}"
        )
    lines.extend(
        [
            (
                "- Candidate pool scope: "
                f"{diagnostics.get('candidate_pool_n', 'unknown')} non-zero "
                "probe-support SAE features from the existing extraction scope; "
                "layer histogram="
                f"{json.dumps(diagnostics.get('candidate_pool_layer_histogram'), sort_keys=True)}"
            ),
            (
                "- Candidate-pool sign counts: "
                f"{json.dumps(diagnostics.get('candidate_pool_weight_sign_counts'), sort_keys=True)}"
            ),
            (
                "- Utility-selected sign counts: "
                f"{json.dumps(diagnostics.get('utility_weight_sign_counts'), sort_keys=True)}"
            ),
            (
                "- Readout-selected sign counts: "
                f"{json.dumps(diagnostics.get('readout_weight_sign_counts'), sort_keys=True)}"
            ),
            (
                "- Outside old |w|>1e-3 shortlist: "
                f"{diagnostics['outside_old_shortlist_count']} / {diagnostics['selected_k']} "
                f"({diagnostics['outside_old_shortlist_fraction']:.4f})"
            ),
            (
                "- Utility vs readout overlap: "
                f"{json.dumps(diagnostics['overlap_with_readout'], sort_keys=True)}"
            ),
            (
                "- Utility layer histogram: "
                f"{json.dumps(diagnostics['utility_layer_histogram'], sort_keys=True)}"
            ),
            (
                "- Readout layer histogram: "
                f"{json.dumps(diagnostics['readout_layer_histogram'], sort_keys=True)}"
            ),
            (
                "- Target-family definitions: "
                f"{json.dumps(target_families, sort_keys=True)}"
            ),
            (
                "- Layer-coverage status: "
                f"{diagnostics.get('layer_coverage_note', 'not recorded')}"
            ),
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir / "report"
    summary_path = output_dir / "heldout_summary.json"
    audit_path = output_dir / "audit_note.md"
    provenance_handle = start_run_provenance(
        args,
        primary_target=output_dir,
        output_targets=[output_dir, summary_path, audit_path],
        primary_target_is_dir=True,
    )
    provenance_status = "completed"
    provenance_extra: dict[str, Any] = {}
    try:
        selector_summary = _load_json(args.selector_dir / "selector_summary.json")
        ordered_ids = _load_manifest_ids(args.selector_dir / "test_manifest.json")
        config = _bundle_config(selector_summary)

        heldout_rows_by_benchmark: dict[str, dict[str, list[dict[str, Any]]]] = {}
        for benchmark in BENCHMARK_TO_SECTION:
            heldout_rows_by_benchmark[benchmark] = {}
            for family in config["family_order"]:
                run_dir = args.heldout_root / benchmark / family / "experiment"
                alpha = config["alpha_by_family"][family]
                heldout_rows_by_benchmark[benchmark][family] = _load_jsonl(
                    _alpha_path(run_dir, alpha)
                )

        summary = build_heldout_summary(
            selector_summary=selector_summary,
            ordered_ids=ordered_ids,
            heldout_rows_by_benchmark=heldout_rows_by_benchmark,
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        audit_path.write_text(build_audit_note(summary) + "\n", encoding="utf-8")
        provenance_extra["n_samples"] = summary["n_samples"]
        provenance_extra["benchmarks"] = list(BENCHMARK_TO_SECTION)
        provenance_extra["families"] = list(config["family_order"])
    except BaseException as exc:
        provenance_status = provenance_status_for_exception(exc)
        provenance_extra["error"] = provenance_error_message(exc)
        raise
    finally:
        finish_run_provenance(provenance_handle, provenance_status, provenance_extra)


if __name__ == "__main__":
    main()
