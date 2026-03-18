#!/usr/bin/env python3
"""Export site-facing JSON artifacts from committed experiment outputs."""

from __future__ import annotations

import argparse
import json
import re
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np

from uncertainty import build_rate_summary, paired_bootstrap_curve_effects


ALPHAS = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def alpha_key(alpha: float) -> str:
    return f"{alpha:.1f}"


def as_pct(rate: float) -> float:
    return round(rate * 100, 1)


def count_jsonl_rows(path: Path) -> int:
    with path.open() as handle:
        return sum(1 for line in handle if line.strip())


def with_pct(summary: dict[str, Any], estimate_key: str = "estimate") -> dict[str, Any]:
    payload = dict(summary)
    payload["pct"] = as_pct(payload[estimate_key])
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


def build_rate_points(results: dict[str, Any]) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    for alpha in ALPHAS:
        record = results[alpha_key(alpha)]
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
    return paired_bootstrap_curve_effects(trajectories, np.array(ALPHAS, dtype=float))


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
    return paired_bootstrap_curve_effects(trajectories, np.array(ALPHAS, dtype=float))


def build_classifier_site_payload(repo_root: Path) -> dict[str, Any]:
    disjoint_summary_path = (
        repo_root / "data/gemma3_4b/pipeline/classifier_disjoint_summary.json"
    )
    overlap_summary_path = (
        repo_root / "data/gemma3_4b/pipeline/classifier_overlap_summary.json"
    )
    qids_path = repo_root / "data/gemma3_4b/pipeline/test_qids_disjoint.json"
    summary = load_json(disjoint_summary_path)
    overlap_summary = load_json(overlap_summary_path)
    disjoint_qids = load_json(qids_path)
    evaluation = summary["evaluation"]
    overlap_evaluation = overlap_summary["evaluation"]
    disjoint_sampled_n = sum(len(ids) for ids in disjoint_qids.values())
    return {
        "schema_version": 1,
        "generated_at": date.today().isoformat(),
        "generated_by": "scripts/export_site_data.py",
        "model": "google/gemma-3-4b-it",
        "source_files": [
            "data/gemma3_4b/pipeline/classifier_disjoint_summary.json",
            "data/gemma3_4b/pipeline/classifier_overlap_summary.json",
            "data/gemma3_4b/pipeline/test_qids_disjoint.json",
        ],
        "n_examples": evaluation["n_examples"],
        "n_positive": evaluation["n_positive"],
        "n_negative": evaluation["n_negative"],
        "selected_h_neurons": summary["selected_h_neurons"],
        "total_ffn_neurons": summary["total_ffn_neurons"],
        "selected_ratio_per_mille": summary["selected_ratio_per_mille"],
        "metrics": evaluation["metrics"],
        "bootstrap": evaluation["bootstrap"],
        "confusion_matrix": evaluation["confusion_matrix"],
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

    payload: dict[str, Any] = {
        "schema_version": 1,
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
            }
            for key, val in alpha_stats.items()
        },
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

    # Add LLM enrichment if available
    llm = summary.get("llm_enrichment")
    if llm:
        payload["llm_enrichment"] = llm

    return payload


def build_payload(repo_root: Path) -> dict[str, Any]:
    anti_dir = repo_root / "data/gemma3_4b/intervention/faitheval"
    standard_dir = repo_root / "data/gemma3_4b/intervention/faitheval_standard"
    negative_control_summary_path = (
        repo_root
        / "data/gemma3_4b/intervention/negative_control/comparison_summary.json"
    )
    anti_results = load_json(anti_dir / "results.json")
    standard_results = load_json(standard_dir / "results.json")
    negative_control_summary = load_json(negative_control_summary_path)
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
        "data/gemma3_4b/intervention/negative_control/comparison_summary.json",
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
            "source_file": "data/gemma3_4b/intervention/negative_control/comparison_summary.json",
            "comparison_to_h_neurons": negative_control_summary[
                "comparison_to_h_neurons"
            ],
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
    batch_review_path = repo_root / "data/batch3500_review.md"
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
            "data/batch3500_review.md",
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
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    payload = build_payload(repo_root)
    output_path = repo_root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n")
    classifier_payload = build_classifier_site_payload(repo_root)
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


if __name__ == "__main__":
    main()
