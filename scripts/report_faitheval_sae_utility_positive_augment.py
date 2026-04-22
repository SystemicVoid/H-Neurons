"""Report the FaithEval SAE utility-positive augment bundle.

Augment slice for the L2/L3 reviewer critique: k=154 strictly-positive-utility
SAE features + 3 layer-matched zero-weight seeds at α=0.0. The noop (α=1.0)
reference is the same held-out bundle produced by Phase 1
(faitheval_sae_utility_selector.sh), so no duplicate noop runs are needed.

Outputs:
  - {output_dir}/augment_heldout_summary.json
  - {output_dir}/augment_audit_note.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from report_faitheval_sae_utility_selector import (
    _alpha_path,
    _bootstrap_mean_summary,
    _load_jsonl,
    _ordered_compliance_array,
    _ordered_metric_array,
)
from uncertainty import (
    build_rate_summary,
    paired_bootstrap_binary_rate_difference,
    paired_bootstrap_continuous_mean_difference_raw,
)
from utils import (
    finish_run_provenance,
    fingerprint_ids,
    provenance_error_message,
    provenance_status_for_exception,
    start_run_provenance,
)


FAMILY_ORDER: tuple[str, ...] = (
    "noop",
    "utility_positive_selected",
    "matched_random_positive_seed_0",
    "matched_random_positive_seed_1",
    "matched_random_positive_seed_2",
)
BENCHMARK_TO_SECTION = {
    "faitheval": "heldout_compliance",
    "faitheval_anti_compliance_margin": "heldout_anti_compliance_margin",
}
ALPHA_BY_FAMILY: dict[str, float] = {
    "noop": 1.0,
    "utility_positive_selected": 0.0,
    "matched_random_positive_seed_0": 0.0,
    "matched_random_positive_seed_1": 0.0,
    "matched_random_positive_seed_2": 0.0,
}
PAIRWISE_DELTA_BASELINES: tuple[tuple[str, str], ...] = (
    ("noop", "utility_positive_minus_noop"),
    (
        "matched_random_positive_seed_0",
        "utility_positive_minus_matched_random_positive_seed_0",
    ),
    (
        "matched_random_positive_seed_1",
        "utility_positive_minus_matched_random_positive_seed_1",
    ),
    (
        "matched_random_positive_seed_2",
        "utility_positive_minus_matched_random_positive_seed_2",
    ),
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
        default=Path(
            "data/gemma3_4b/intervention/faitheval_sae_utility_selector/report_augment"
        ),
    )
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _load_manifest_ids(path: Path) -> list[str]:
    payload = _load_json(path)
    ids = payload.get("ids")
    if not isinstance(ids, list) or not ids:
        raise ValueError(f"Sample manifest {path} is missing a non-empty ids list")
    return [str(sample_id) for sample_id in ids]


def _build_compliance_section(
    *,
    benchmark: str,
    rows_by_family: dict[str, list[dict[str, Any]]],
    ordered_ids: list[str],
) -> dict[str, Any]:
    compliance_by_family: dict[str, Any] = {}
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
    utility_positive = compliance_by_family["utility_positive_selected"]
    for baseline_family, delta_key in PAIRWISE_DELTA_BASELINES:
        paired_deltas_pp[delta_key] = paired_bootstrap_binary_rate_difference(
            compliance_by_family[baseline_family],
            utility_positive,
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


def _build_margin_section(
    *,
    benchmark: str,
    rows_by_family: dict[str, list[dict[str, Any]]],
    ordered_ids: list[str],
) -> dict[str, Any]:
    metric_by_family: dict[str, Any] = {}
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
    utility_positive = metric_by_family["utility_positive_selected"]
    for baseline_family, delta_key in PAIRWISE_DELTA_BASELINES:
        paired_deltas[delta_key] = paired_bootstrap_continuous_mean_difference_raw(
            metric_by_family[baseline_family],
            utility_positive,
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


def build_augment_summary(
    *,
    selector_summary: dict[str, Any],
    augment_summary: dict[str, Any],
    ordered_ids: list[str],
    heldout_rows_by_benchmark: dict[str, dict[str, list[dict[str, Any]]]],
) -> dict[str, Any]:
    utility_positive_diag = augment_summary["family_diagnostics"][
        "utility_positive_selected"
    ]
    candidate_pool = selector_summary.get("candidate_pool", {})
    main_utility_family = selector_summary["families"]["utility_selected"]
    return {
        "schema_version": "faitheval_sae_utility_positive_augment_report/v1",
        "n_samples": len(ordered_ids),
        "sample_ids_fingerprint": fingerprint_ids(ordered_ids),
        "heldout_compliance": _build_compliance_section(
            benchmark="faitheval",
            rows_by_family=heldout_rows_by_benchmark["faitheval"],
            ordered_ids=ordered_ids,
        ),
        "heldout_anti_compliance_margin": _build_margin_section(
            benchmark="faitheval_anti_compliance_margin",
            rows_by_family=heldout_rows_by_benchmark[
                "faitheval_anti_compliance_margin"
            ],
            ordered_ids=ordered_ids,
        ),
        "augment_design": augment_summary.get("augment_design", {}),
        "augment_diagnostics": {
            "utility_positive_k": utility_positive_diag["k"],
            "utility_positive_layer_histogram": utility_positive_diag[
                "layer_histogram"
            ],
            "utility_positive_weight_sign_counts": utility_positive_diag[
                "weight_sign_counts"
            ],
            "utility_positive_mean_selector_score": utility_positive_diag[
                "mean_selector_score"
            ],
            "utility_positive_min_selector_score": utility_positive_diag[
                "min_selector_score"
            ],
            "main_utility_k": main_utility_family["k"],
            "main_utility_layer_histogram": main_utility_family["layer_histogram"],
            "main_utility_weight_sign_counts": main_utility_family.get(
                "weight_sign_counts"
            ),
            "candidate_pool_layer_histogram": candidate_pool.get("layer_histogram"),
            "candidate_pool_n": candidate_pool.get("n_features"),
        },
    }


def _format_family_rates(section: dict[str, Any]) -> str:
    parts: list[str] = []
    for family in FAMILY_ORDER:
        stats = section["families"][family]
        parts.append(
            f"{family}={stats['estimate']:.4f} "
            f"({stats['n_compliant']}/{stats['n_total']})"
        )
    return "; ".join(parts)


def _format_family_means(section: dict[str, Any]) -> str:
    parts: list[str] = []
    for family in FAMILY_ORDER:
        stats = section["families"][family]
        parts.append(
            f"{family}={stats['estimate']:+.4f} "
            f"[{stats['ci']['lower']:+.4f}, {stats['ci']['upper']:+.4f}]"
        )
    return "; ".join(parts)


def _format_delta_triplet_pp(section: dict[str, Any]) -> str:
    parts: list[str] = []
    for _baseline, delta_key in PAIRWISE_DELTA_BASELINES:
        delta = section["paired_deltas_pp"][delta_key]
        parts.append(
            f"{delta_key}={delta['estimate_pp']:+.3f} pp "
            f"[{delta['ci_pp']['lower']:+.3f}, {delta['ci_pp']['upper']:+.3f}]"
        )
    return "; ".join(parts)


def _format_delta_triplet_raw(section: dict[str, Any]) -> str:
    parts: list[str] = []
    for _baseline, delta_key in PAIRWISE_DELTA_BASELINES:
        delta = section["paired_deltas"][delta_key]
        parts.append(
            f"{delta_key}={delta['estimate']:+.4f} "
            f"[{delta['ci']['lower']:+.4f}, {delta['ci']['upper']:+.4f}]"
        )
    return "; ".join(parts)


def build_audit_note(summary: dict[str, Any]) -> str:
    diagnostics = summary["augment_diagnostics"]
    design = summary["augment_design"]
    return "\n".join(
        [
            "# FaithEval SAE Utility-Positive Augment Audit",
            "",
            "- Augment slice: "
            f"k={diagnostics['utility_positive_k']} strictly-positive-utility "
            "SAE features + 3 layer-matched zero-weight seeds at α=0.0.",
            "- Reference noop (α=1.0) reused from Phase 1 selector bundle.",
            (
                "- Review loophole targeted: "
                f"{design.get('review_loophole_targeted', 'n/a')}"
            ),
            (
                "- FaithEval compliance families: "
                f"{_format_family_rates(summary['heldout_compliance'])}"
            ),
            (
                "- FaithEval compliance paired deltas: "
                f"{_format_delta_triplet_pp(summary['heldout_compliance'])}"
            ),
            (
                "- Anti-compliance margin families: "
                f"{_format_family_means(summary['heldout_anti_compliance_margin'])}"
            ),
            (
                "- Anti-compliance margin paired deltas: "
                f"{_format_delta_triplet_raw(summary['heldout_anti_compliance_margin'])}"
            ),
            (
                "- Utility-positive layer histogram: "
                f"{json.dumps(diagnostics['utility_positive_layer_histogram'], sort_keys=True)}"
            ),
            (
                "- Main utility-selected (k=266) layer histogram: "
                f"{json.dumps(diagnostics['main_utility_layer_histogram'], sort_keys=True)}"
            ),
            (
                "- Candidate pool layer histogram: "
                f"{json.dumps(diagnostics['candidate_pool_layer_histogram'], sort_keys=True)}"
            ),
            (
                "- Utility-positive weight-sign counts: "
                f"{json.dumps(diagnostics['utility_positive_weight_sign_counts'], sort_keys=True)}"
            ),
            (
                "- Utility-positive selector-score range: min="
                f"{diagnostics['utility_positive_min_selector_score']:.6g}, "
                "mean="
                f"{diagnostics['utility_positive_mean_selector_score']:.6g}"
            ),
        ]
    )


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    summary_path = output_dir / "augment_heldout_summary.json"
    audit_path = output_dir / "augment_audit_note.md"
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
        augment_summary = _load_json(
            args.selector_dir / "utility_positive_summary.json"
        )
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

        summary = build_augment_summary(
            selector_summary=selector_summary,
            augment_summary=augment_summary,
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
