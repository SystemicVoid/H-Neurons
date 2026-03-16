#!/usr/bin/env python3
"""Export site-facing JSON artifacts from committed experiment outputs."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from typing import Any


ALPHAS = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def alpha_key(alpha: float) -> str:
    return f"{alpha:.1f}"


def as_pct(rate: float) -> float:
    return round(rate * 100, 1)


def build_rate_points(results: dict[str, Any]) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    for alpha in ALPHAS:
        record = results[alpha_key(alpha)]
        points.append(
            {
                "alpha": alpha,
                "n_total": record["n_total"],
                "n_compliant": record["n_compliant"],
                "compliance_rate": record["compliance_rate"],
                "compliance_pct": as_pct(record["compliance_rate"]),
            }
        )
    return points


def build_standard_format_points(base_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    parse_failure_points: list[dict[str, Any]] = []
    parseable_subset_points: list[dict[str, Any]] = []

    for alpha in ALPHAS:
        rows = load_jsonl(base_dir / f"alpha_{alpha:.1f}.jsonl")
        n_total = len(rows)
        parse_failures = sum(row["chosen"] is None for row in rows)
        parseable_rows = [row for row in rows if row["chosen"] is not None]
        parseable_n = len(parseable_rows)
        compliant_parseable = sum(bool(row["compliance"]) for row in parseable_rows)
        parseable_rate = compliant_parseable / parseable_n

        parse_failure_points.append(
            {
                "alpha": alpha,
                "n_total": n_total,
                "count": parse_failures,
                "rate": parse_failures / n_total,
                "pct": as_pct(parse_failures / n_total),
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
            }
        )

    return parse_failure_points, parseable_subset_points


def build_anti_population(base_dir: Path) -> dict[str, Any]:
    rows_by_alpha = {
        alpha: {row["id"]: row for row in load_jsonl(base_dir / f"alpha_{alpha:.1f}.jsonl")}
        for alpha in ALPHAS
    }
    reference_ids = set(rows_by_alpha[ALPHAS[0]])
    for alpha in ALPHAS[1:]:
        ids = set(rows_by_alpha[alpha])
        if ids != reference_ids:
            raise ValueError(f"Mismatched anti-compliance sample IDs at alpha={alpha:.1f}")

    trajectories = {
        sample_id: [bool(rows_by_alpha[alpha][sample_id]["compliance"]) for alpha in ALPHAS]
        for sample_id in sorted(reference_ids)
    }
    always_compliant_ids = [sample_id for sample_id, values in trajectories.items() if all(values)]
    never_compliant_ids = [sample_id for sample_id, values in trajectories.items() if not any(values)]
    swing_ids = [sample_id for sample_id, values in trajectories.items() if any(values) and not all(values)]

    swing_breakdown = []
    for alpha in ALPHAS:
        swing_compliant = sum(bool(rows_by_alpha[alpha][sample_id]["compliance"]) for sample_id in swing_ids)
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
        },
        "never_compliant": {
            "count": len(never_compliant_ids),
            "pct": as_pct(len(never_compliant_ids) / total),
        },
        "swing": {
            "count": len(swing_ids),
            "pct": as_pct(len(swing_ids) / total),
        },
        "swing_breakdown": swing_breakdown,
    }


def build_payload(repo_root: Path) -> dict[str, Any]:
    anti_dir = repo_root / "data/gemma3_4b/intervention/faitheval"
    standard_dir = repo_root / "data/gemma3_4b/intervention/faitheval_standard"
    anti_results = load_json(anti_dir / "results.json")
    standard_results = load_json(standard_dir / "results.json")
    remap_summary = load_json(standard_dir / "alpha_3.0_parse_failure_remap_summary.json")
    parse_failure_points, parseable_subset_points = build_standard_format_points(standard_dir)
    anti_population = build_anti_population(anti_dir)

    source_files = [
        "data/gemma3_4b/intervention/faitheval/results.json",
        "data/gemma3_4b/intervention/faitheval/alpha_0.0.jsonl",
        "data/gemma3_4b/intervention/faitheval/alpha_0.5.jsonl",
        "data/gemma3_4b/intervention/faitheval/alpha_1.0.jsonl",
        "data/gemma3_4b/intervention/faitheval/alpha_1.5.jsonl",
        "data/gemma3_4b/intervention/faitheval/alpha_2.0.jsonl",
        "data/gemma3_4b/intervention/faitheval/alpha_2.5.jsonl",
        "data/gemma3_4b/intervention/faitheval/alpha_3.0.jsonl",
        "data/gemma3_4b/intervention/faitheval_standard/results.json",
        "data/gemma3_4b/intervention/faitheval_standard/alpha_0.0.jsonl",
        "data/gemma3_4b/intervention/faitheval_standard/alpha_0.5.jsonl",
        "data/gemma3_4b/intervention/faitheval_standard/alpha_1.0.jsonl",
        "data/gemma3_4b/intervention/faitheval_standard/alpha_1.5.jsonl",
        "data/gemma3_4b/intervention/faitheval_standard/alpha_2.0.jsonl",
        "data/gemma3_4b/intervention/faitheval_standard/alpha_2.5.jsonl",
        "data/gemma3_4b/intervention/faitheval_standard/alpha_3.0.jsonl",
        "data/gemma3_4b/intervention/faitheval_standard/alpha_3.0_parse_failure_remap_summary.json",
    ]

    return {
        "schema_version": 1,
        "generated_at": date.today().isoformat(),
        "generated_by": "scripts/export_site_data.py",
        "benchmark": anti_results["benchmark"],
        "model": anti_results["model"],
        "classifier": anti_results["classifier"],
        "n_h_neurons": anti_results["n_h_neurons"],
        "ci_status": "no_ci_yet",
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
                "ci_status": "no_ci_yet",
                "points": build_rate_points(anti_results["results"]),
            },
            "standard_raw": {
                "label": "Standard prompt (raw MC-letter score)",
                "prompt_style": "standard",
                "ci_status": "no_ci_yet",
                "points": build_rate_points(standard_results["results"]),
            },
            "standard_parseable_subset": {
                "label": "Standard prompt (parseable subset only)",
                "prompt_style": "standard",
                "ci_status": "no_ci_yet",
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
                "ci_status": "no_ci_yet",
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
                        "raw_compliance_pct": as_pct(remap_summary["raw_compliance_rate"]),
                        "parse_failures": remap_summary["parse_failures"],
                        "strict_recovered_count": remap_summary["strict_recovered_count"],
                        "strict_recovered_rate_within_failures": remap_summary["strict_recovered_rate_within_failures"],
                        "strict_recovered_compliant_count": remap_summary["strict_recovered_compliant_count"],
                        "strict_rescored_compliance_count": remap_summary["strict_rescored_compliance_count"],
                        "strict_rescored_compliance_rate": remap_summary["strict_rescored_compliance_rate"],
                        "strict_rescored_compliance_pct": as_pct(remap_summary["strict_rescored_compliance_rate"]),
                        "review_category_counts": remap_summary["review_category_counts"],
                        "unresolved_review_category_counts": remap_summary["unresolved_review_category_counts"],
                    }
                },
            },
        },
        "parse_failures": {
            "label": "Responses without recoverable option letter",
            "prompt_style": "standard",
            "ci_status": "no_ci_yet",
            "points": parse_failure_points,
        },
        "population": {
            "anti_compliance": {
                "label": "Anti-compliance prompt",
                "ci_status": "no_ci_yet",
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("site/data/intervention_sweep.json"),
        help="Output path for exported site data.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    payload = build_payload(repo_root)
    output_path = repo_root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
