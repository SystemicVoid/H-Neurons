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


FAMILY_ORDER = (
    "noop",
    "readout_selected",
    "utility_selected",
    "matched_random_seed_0",
    "matched_random_seed_1",
    "matched_random_seed_2",
)
BENCHMARK_TO_SECTION = {
    "faitheval": "heldout_compliance",
    "faitheval_anti_compliance_margin": "heldout_anti_compliance_margin",
}
ALPHA_BY_FAMILY = {
    "noop": 1.0,
    "readout_selected": 0.0,
    "utility_selected": 0.0,
    "matched_random_seed_0": 0.0,
    "matched_random_seed_1": 0.0,
    "matched_random_seed_2": 0.0,
}
PAIRWISE_DELTA_BASELINES = (
    ("readout_selected", "utility_minus_readout"),
    ("noop", "utility_minus_noop"),
    ("matched_random_seed_0", "utility_minus_matched_random_seed_0"),
    ("matched_random_seed_1", "utility_minus_matched_random_seed_1"),
    ("matched_random_seed_2", "utility_minus_matched_random_seed_2"),
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


def _build_measurement_section(
    *,
    benchmark: str,
    rows_by_family: dict[str, list[dict[str, Any]]],
    ordered_ids: list[str],
) -> dict[str, Any]:
    compliance_by_family: dict[str, np.ndarray] = {}
    families: dict[str, Any] = {}

    for family in FAMILY_ORDER:
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
    for baseline_family, delta_key in PAIRWISE_DELTA_BASELINES:
        paired_deltas_pp[delta_key] = paired_bootstrap_binary_rate_difference(
            compliance_by_family[baseline_family],
            utility,
        )

    return {
        "benchmark": benchmark,
        "prompt_style": "anti_compliance",
        "metric_name": "compliance",
        "delta_units": "percentage_points",
        "family_order": list(FAMILY_ORDER),
        "alpha_by_family": {
            family: float(ALPHA_BY_FAMILY[family]) for family in FAMILY_ORDER
        },
        "families": families,
        "paired_deltas_pp": paired_deltas_pp,
    }


def _build_margin_measurement_section(
    *,
    benchmark: str,
    rows_by_family: dict[str, list[dict[str, Any]]],
    ordered_ids: list[str],
) -> dict[str, Any]:
    metric_by_family: dict[str, np.ndarray] = {}
    families: dict[str, Any] = {}

    for family in FAMILY_ORDER:
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
    for baseline_family, delta_key in PAIRWISE_DELTA_BASELINES:
        paired_deltas[delta_key] = paired_bootstrap_continuous_mean_difference_raw(
            metric_by_family[baseline_family],
            utility,
        )

    return {
        "benchmark": benchmark,
        "prompt_style": "anti_compliance",
        "metric_name": "misleading_minus_preferred_logprob_margin",
        "margin_definition": "logp(counterfactual_key) - logp(preferred_key)",
        "delta_units": "logprob_margin",
        "family_order": list(FAMILY_ORDER),
        "alpha_by_family": {
            family: float(ALPHA_BY_FAMILY[family]) for family in FAMILY_ORDER
        },
        "families": families,
        "paired_deltas": paired_deltas,
    }


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
    selector_design = _selector_design(selector_summary)
    utility_family = selector_summary["families"]["utility_selected"]
    readout_family = selector_summary["families"]["readout_selected"]
    overlap = selector_summary["family_overlap"]["utility_selected_vs_readout_selected"]
    candidate_pool = selector_summary.get("candidate_pool", {})
    return {
        "schema_version": "faitheval_sae_utility_selector_report/v4",
        "n_samples": len(ordered_ids),
        "sample_ids_fingerprint": fingerprint_ids(ordered_ids),
        "heldout_compliance": _build_measurement_section(
            benchmark="faitheval",
            rows_by_family=heldout_rows_by_benchmark["faitheval"],
            ordered_ids=ordered_ids,
        ),
        "heldout_anti_compliance_margin": _build_margin_measurement_section(
            benchmark="faitheval_anti_compliance_margin",
            rows_by_family=heldout_rows_by_benchmark[
                "faitheval_anti_compliance_margin"
            ],
            ordered_ids=ordered_ids,
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


def _format_family_rates(section: dict[str, Any]) -> str:
    parts = []
    for family in FAMILY_ORDER:
        stats = section["families"][family]
        parts.append(
            f"{family}={stats['estimate']:.4f} "
            f"({stats['n_compliant']}/{stats['n_total']})"
        )
    return "; ".join(parts)


def _format_delta_triplet(section: dict[str, Any]) -> str:
    keys = (
        "utility_minus_readout",
        "utility_minus_noop",
        "utility_minus_matched_random_seed_0",
        "utility_minus_matched_random_seed_1",
        "utility_minus_matched_random_seed_2",
    )
    parts = []
    for key in keys:
        delta = section["paired_deltas_pp"][key]
        parts.append(
            f"{key}={delta['estimate_pp']:+.3f} pp "
            f"[{delta['ci_pp']['lower']:+.3f}, {delta['ci_pp']['upper']:+.3f}]"
        )
    return "; ".join(parts)


def _format_family_means(section: dict[str, Any]) -> str:
    parts = []
    for family in FAMILY_ORDER:
        stats = section["families"][family]
        parts.append(
            f"{family}={stats['estimate']:+.4f} "
            f"[{stats['ci']['lower']:+.4f}, {stats['ci']['upper']:+.4f}]"
        )
    return "; ".join(parts)


def _format_raw_delta_triplet(section: dict[str, Any]) -> str:
    keys = (
        "utility_minus_readout",
        "utility_minus_noop",
        "utility_minus_matched_random_seed_0",
        "utility_minus_matched_random_seed_1",
        "utility_minus_matched_random_seed_2",
    )
    parts = []
    for key in keys:
        delta = section["paired_deltas"][key]
        parts.append(
            f"{key}={delta['estimate']:+.4f} "
            f"[{delta['ci']['lower']:+.4f}, {delta['ci']['upper']:+.4f}]"
        )
    return "; ".join(parts)


def build_audit_note(summary: dict[str, Any]) -> str:
    diagnostics = summary["selector_diagnostics"]
    target_families = diagnostics.get("target_families") or {}
    return "\n".join(
        [
            "# FaithEval SAE Utility Selector Audit",
            "",
            (
                "- Held-out diagnostic bundle on a single locked test manifest "
                f"(n={summary['n_samples']})."
            ),
            (
                "- FaithEval families: "
                f"{_format_family_rates(summary['heldout_compliance'])}"
            ),
            (
                "- FaithEval paired deltas: "
                f"{_format_delta_triplet(summary['heldout_compliance'])}"
            ),
            (
                "- FaithEval anti-compliance margin families: "
                f"{_format_family_means(summary['heldout_anti_compliance_margin'])}"
            ),
            (
                "- FaithEval anti-compliance margin paired deltas: "
                f"{_format_raw_delta_triplet(summary['heldout_anti_compliance_margin'])}"
            ),
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

        heldout_rows_by_benchmark: dict[str, dict[str, list[dict[str, Any]]]] = {}
        for benchmark in BENCHMARK_TO_SECTION:
            heldout_rows_by_benchmark[benchmark] = {}
            for family in FAMILY_ORDER:
                run_dir = args.heldout_root / benchmark / family / "experiment"
                alpha = ALPHA_BY_FAMILY[family]
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
        provenance_extra["families"] = list(FAMILY_ORDER)
    except BaseException as exc:
        provenance_status = provenance_status_for_exception(exc)
        provenance_extra["error"] = provenance_error_message(exc)
        raise
    finally:
        finish_run_provenance(provenance_handle, provenance_status, provenance_extra)


if __name__ == "__main__":
    main()
