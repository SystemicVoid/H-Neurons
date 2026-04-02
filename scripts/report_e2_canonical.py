#!/usr/bin/env python3
"""Build canonical E2 report artifacts (JSON + Markdown)."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.stats import spearmanr

from uncertainty import (
    DEFAULT_BOOTSTRAP_RESAMPLES,
    paired_bootstrap_binary_rate_difference,
    paired_bootstrap_continuous_mean_difference,
    percentile_interval,
    wilson_interval,
)
from utils import format_alpha_label

VALID_GRADES = {"CORRECT", "INCORRECT", "NOT_ATTEMPTED"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--e2_name", type=str, default="E2-A TriviaQA Source-Isolated")
    p.add_argument("--e2_locked_config", type=Path, required=True)
    p.add_argument("--e2_pilot_report", type=Path, default=None)
    p.add_argument("--e2_mc1_fold0_dir", type=Path, required=True)
    p.add_argument("--e2_mc1_fold1_dir", type=Path, required=True)
    p.add_argument("--e2_mc2_fold0_dir", type=Path, required=True)
    p.add_argument("--e2_mc2_fold1_dir", type=Path, required=True)
    p.add_argument("--e2_simpleqa_dir", type=Path, required=True)
    p.add_argument("--e2_artifact", type=Path, required=True)

    p.add_argument("--paper_locked_config", type=Path, required=True)
    p.add_argument("--paper_mc1_fold0_dir", type=Path, required=True)
    p.add_argument("--paper_mc1_fold1_dir", type=Path, required=True)
    p.add_argument("--paper_mc2_fold0_dir", type=Path, required=True)
    p.add_argument("--paper_mc2_fold1_dir", type=Path, required=True)
    p.add_argument("--paper_simpleqa_dir", type=Path, required=True)
    p.add_argument("--paper_artifact", type=Path, required=True)

    p.add_argument("--e1_locked_config", type=Path, required=True)
    p.add_argument("--e1_mc1_fold0_dir", type=Path, required=True)
    p.add_argument("--e1_mc1_fold1_dir", type=Path, required=True)
    p.add_argument("--e1_mc2_fold0_dir", type=Path, required=True)
    p.add_argument("--e1_mc2_fold1_dir", type=Path, required=True)
    p.add_argument("--e1_simpleqa_dir", type=Path, required=True)
    p.add_argument("--e1_artifact", type=Path, required=True)

    p.add_argument("--materiality_pp", type=float, default=1.5)
    p.add_argument("--output_json", type=Path, required=True)
    p.add_argument("--output_md", type=Path, required=True)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _safe_wilson(successes: int, total: int) -> dict[str, Any]:
    if total <= 0:
        return {"lower": 0.0, "upper": 0.0, "level": 0.95, "method": "wilson"}
    return wilson_interval(successes, total).to_dict()


def _bootstrap_mean_ci(values: np.ndarray, *, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    n = len(values)
    means = np.empty(DEFAULT_BOOTSTRAP_RESAMPLES, dtype=float)
    for i in range(DEFAULT_BOOTSTRAP_RESAMPLES):
        means[i] = values[rng.choice(n, size=n, replace=True)].mean()
    ci = percentile_interval(means, 0.95, method="bootstrap_percentile")
    return {"estimate": float(values.mean()), "ci": ci.to_dict()}


def _read_mc_fold_map(
    fold_dir: Path, alpha: float, variant: str
) -> dict[str, float | bool]:
    label = format_alpha_label(alpha)
    rows = _load_jsonl(fold_dir / f"alpha_{label}.jsonl")
    out: dict[str, float | bool] = {}
    for row in rows:
        key = str(row["id"])
        if variant == "mc1":
            out[key] = bool(row["compliance"])
        else:
            out[key] = float(row["metric_value"])
    return out


def _merge_fold_maps(
    fold0: dict[str, float | bool], fold1: dict[str, float | bool]
) -> dict[str, float | bool]:
    merged: dict[str, float | bool] = {}
    for key, value in fold0.items():
        merged[f"f0:{key}"] = value
    for key, value in fold1.items():
        merged[f"f1:{key}"] = value
    return merged


def _preview_ids(ids: list[str], limit: int = 5) -> list[str]:
    return ids[:limit]


def _require_identical_sample_ids(
    baseline_map: dict[str, Any],
    compare_map: dict[str, Any],
    *,
    context: str,
) -> list[str]:
    baseline_ids = set(baseline_map)
    compare_ids = set(compare_map)
    if baseline_ids != compare_ids:
        missing_in_compare = sorted(baseline_ids - compare_ids)
        missing_in_baseline = sorted(compare_ids - baseline_ids)
        raise ValueError(
            f"{context}: paired sample IDs must match exactly "
            f"(missing_in_compare={len(missing_in_compare)} "
            f"sample={_preview_ids(missing_in_compare)}; "
            f"missing_in_baseline={len(missing_in_baseline)} "
            f"sample={_preview_ids(missing_in_baseline)})."
        )
    if not baseline_ids:
        raise ValueError(f"{context}: no sample IDs found for paired comparison")
    return sorted(baseline_ids)


def _paired_delta_from_maps(
    baseline_map: dict[str, float | bool],
    compare_map: dict[str, float | bool],
    *,
    variant: str,
    seed: int,
) -> dict[str, Any]:
    common_ids = _require_identical_sample_ids(
        baseline_map,
        compare_map,
        context=f"{variant} paired comparison",
    )

    if variant == "mc1":
        baseline = np.array([bool(baseline_map[sid]) for sid in common_ids], dtype=bool)
        compare = np.array([bool(compare_map[sid]) for sid in common_ids], dtype=bool)
        return {
            "n": len(common_ids),
            "baseline_rate": float(baseline.mean()),
            "baseline_ci": _safe_wilson(int(baseline.sum()), len(baseline)),
            "compare_rate": float(compare.mean()),
            "compare_ci": _safe_wilson(int(compare.sum()), len(compare)),
            "delta": paired_bootstrap_binary_rate_difference(
                baseline, compare, seed=seed
            ),
        }

    baseline_c = np.array([float(baseline_map[sid]) for sid in common_ids], dtype=float)
    compare_c = np.array([float(compare_map[sid]) for sid in common_ids], dtype=float)
    return {
        "n": len(common_ids),
        "baseline_rate": float(baseline_c.mean()),
        "baseline_ci": _bootstrap_mean_ci(baseline_c, seed=seed)["ci"],
        "compare_rate": float(compare_c.mean()),
        "compare_ci": _bootstrap_mean_ci(compare_c, seed=seed)["ci"],
        "delta": paired_bootstrap_continuous_mean_difference(
            baseline_c, compare_c, seed=seed
        ),
    }


def _read_simpleqa_grade_map(experiment_dir: Path, alpha: float) -> dict[str, str]:
    label = format_alpha_label(alpha)
    rows = _load_jsonl(experiment_dir / f"alpha_{label}.jsonl")
    out: dict[str, str] = {}
    for row in rows:
        grade = str(row.get("simpleqa_grade", "")).upper()
        if grade not in VALID_GRADES:
            raise ValueError(
                f"Missing/invalid simpleqa_grade for id={row.get('id')} in {experiment_dir}"
            )
        out[str(row["id"])] = grade
    return out


def _counts_from_grade_map(grade_map: dict[str, str]) -> dict[str, int]:
    return {
        "CORRECT": sum(g == "CORRECT" for g in grade_map.values()),
        "INCORRECT": sum(g == "INCORRECT" for g in grade_map.values()),
        "NOT_ATTEMPTED": sum(g == "NOT_ATTEMPTED" for g in grade_map.values()),
    }


def _precision(correct: int, attempted: int) -> float:
    return (correct / attempted) if attempted > 0 else 0.0


def _simpleqa_summary(grade_map: dict[str, str]) -> dict[str, Any]:
    counts = _counts_from_grade_map(grade_map)
    n_total = len(grade_map)
    attempted = counts["CORRECT"] + counts["INCORRECT"]
    compliance = counts["CORRECT"] / n_total if n_total else 0.0
    attempt_rate = attempted / n_total if n_total else 0.0
    precision = _precision(counts["CORRECT"], attempted)
    return {
        "n": int(n_total),
        "counts": counts,
        "compliance": {
            "rate": float(compliance),
            "ci": _safe_wilson(counts["CORRECT"], n_total),
        },
        "attempt_rate": {
            "rate": float(attempt_rate),
            "ci": _safe_wilson(attempted, n_total),
        },
        "precision": {
            "rate": float(precision),
            "ci": _safe_wilson(counts["CORRECT"], attempted),
        },
    }


def _bootstrap_precision_delta(
    baseline: list[str], compare: list[str], *, seed: int
) -> dict[str, Any]:
    base = np.array(baseline, dtype=object)
    comp = np.array(compare, dtype=object)
    if base.shape != comp.shape:
        raise ValueError("Precision delta arrays must match")

    def _prec(arr: np.ndarray) -> float:
        attempted = int(np.sum(arr != "NOT_ATTEMPTED"))
        correct = int(np.sum(arr == "CORRECT"))
        return _precision(correct, attempted)

    rng = np.random.default_rng(seed)
    n = len(base)
    samples = np.empty(DEFAULT_BOOTSTRAP_RESAMPLES, dtype=float)
    for idx in range(DEFAULT_BOOTSTRAP_RESAMPLES):
        sample_idx = rng.choice(n, size=n, replace=True)
        samples[idx] = (_prec(comp[sample_idx]) - _prec(base[sample_idx])) * 100.0

    ci = percentile_interval(
        samples, 0.95, method="bootstrap_percentile_paired_conditional_precision"
    ).to_dict()
    return {
        "estimate_pp": float((_prec(comp) - _prec(base)) * 100.0),
        "ci_pp": ci,
        "bootstrap": {
            "n_resamples": int(DEFAULT_BOOTSTRAP_RESAMPLES),
            "seed": int(seed),
            "resampling": "paired_by_sample_id",
        },
    }


def _paired_simpleqa_delta(
    baseline_map: dict[str, str], compare_map: dict[str, str], *, seed: int
) -> dict[str, Any]:
    common_ids = _require_identical_sample_ids(
        baseline_map,
        compare_map,
        context="SimpleQA paired comparison",
    )

    baseline = [baseline_map[sid] for sid in common_ids]
    compare = [compare_map[sid] for sid in common_ids]

    baseline_compliance = np.array([g == "CORRECT" for g in baseline], dtype=bool)
    compare_compliance = np.array([g == "CORRECT" for g in compare], dtype=bool)
    baseline_attempt = np.array([g != "NOT_ATTEMPTED" for g in baseline], dtype=bool)
    compare_attempt = np.array([g != "NOT_ATTEMPTED" for g in compare], dtype=bool)

    transitions: dict[str, int] = {}
    for before, after in zip(baseline, compare, strict=False):
        key = f"{before}->{after}"
        transitions[key] = transitions.get(key, 0) + 1

    return {
        "n": len(common_ids),
        "compliance_delta": paired_bootstrap_binary_rate_difference(
            baseline_compliance, compare_compliance, seed=seed
        ),
        "attempt_delta": paired_bootstrap_binary_rate_difference(
            baseline_attempt, compare_attempt, seed=seed
        ),
        "precision_delta": _bootstrap_precision_delta(baseline, compare, seed=seed),
        "transitions": transitions,
        "required_transition_counts": {
            "CORRECT->INCORRECT": transitions.get("CORRECT->INCORRECT", 0),
            "INCORRECT->CORRECT": transitions.get("INCORRECT->CORRECT", 0),
            "NOT_ATTEMPTED->INCORRECT": transitions.get("NOT_ATTEMPTED->INCORRECT", 0),
            "NOT_ATTEMPTED->CORRECT": transitions.get("NOT_ATTEMPTED->CORRECT", 0),
        },
    }


def _head_key(entry: dict[str, Any]) -> tuple[int, int, str]:
    return (
        int(entry["layer"]),
        int(entry["head"]),
        str(entry.get("position_summary", "")),
    )


def _direction_cosine(a: list[float], b: list[float]) -> float:
    va = np.asarray(a, dtype=float)
    vb = np.asarray(b, dtype=float)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(va, vb) / denom)


def _head_diagnostics(
    e2_artifact: Path, e2_k: int, ref_artifact: Path, ref_k: int
) -> dict[str, Any]:
    e2 = torch.load(e2_artifact, map_location="cpu")
    ref = torch.load(ref_artifact, map_location="cpu")

    e2_ranked = list(e2.get("ranked_heads", []))
    ref_ranked = list(ref.get("ranked_heads", []))
    e2_selected = e2_ranked[:e2_k]
    ref_selected = ref_ranked[:ref_k]

    e2_set = {_head_key(row) for row in e2_selected}
    ref_set = {_head_key(row) for row in ref_selected}
    intersection = sorted(e2_set & ref_set)
    union = e2_set | ref_set

    e2_rank_map = {_head_key(row): idx + 1 for idx, row in enumerate(e2_ranked)}
    ref_rank_map = {_head_key(row): idx + 1 for idx, row in enumerate(ref_ranked)}
    shared_rank_keys = sorted(set(e2_rank_map) & set(ref_rank_map))
    if len(shared_rank_keys) >= 2:
        e2_ranks = np.array([e2_rank_map[key] for key in shared_rank_keys], dtype=float)
        ref_ranks = np.array(
            [ref_rank_map[key] for key in shared_rank_keys], dtype=float
        )
        corr, pvalue = spearmanr(e2_ranks, ref_ranks)
        rank_agreement = {
            "n_shared_ranked_heads": len(shared_rank_keys),
            "spearman_rho": float(corr),
            "spearman_pvalue": float(pvalue),
        }
    else:
        rank_agreement = {
            "n_shared_ranked_heads": len(shared_rank_keys),
            "spearman_rho": None,
            "spearman_pvalue": None,
        }

    ref_by_key = {_head_key(row): row for row in ref_selected}
    e2_by_key = {_head_key(row): row for row in e2_selected}
    direction_cosines: list[float] = []
    for key in intersection:
        e2_dir = e2_by_key[key].get("direction")
        ref_dir = ref_by_key[key].get("direction")
        if isinstance(e2_dir, list) and isinstance(ref_dir, list):
            direction_cosines.append(_direction_cosine(e2_dir, ref_dir))

    return {
        "selected_overlap": {
            "n_e2_selected": len(e2_set),
            "n_ref_selected": len(ref_set),
            "intersection": len(intersection),
            "union": len(union),
            "jaccard": float(len(intersection) / len(union)) if union else 0.0,
            "overlap_keys": [list(key) for key in intersection],
        },
        "rank_agreement": rank_agreement,
        "direction_similarity": {
            "n_shared_selected_with_direction": len(direction_cosines),
            "mean_cosine": float(np.mean(direction_cosines))
            if direction_cosines
            else None,
            "median_cosine": float(np.median(direction_cosines))
            if direction_cosines
            else None,
            "min_cosine": float(np.min(direction_cosines))
            if direction_cosines
            else None,
            "max_cosine": float(np.max(direction_cosines))
            if direction_cosines
            else None,
        },
    }


def _classify_outcome(
    *,
    e2_vs_paper_mc1_delta_pp: float,
    e2_vs_paper_mc1_ci_lower_pp: float,
    e2_vs_paper_compliance_delta_pp: float,
    e2_vs_paper_compliance_ci_lower_pp: float,
    e2_vs_paper_attempt_delta_pp: float,
    e2_vs_paper_attempt_ci_lower_pp: float,
    e2_vs_paper_precision_delta_pp: float,
    e2_vs_paper_precision_ci_lower_pp: float,
    materiality_pp: float,
) -> str:
    mc1_not_materially_worse = e2_vs_paper_mc1_ci_lower_pp >= -materiality_pp
    compliance_improves = e2_vs_paper_compliance_ci_lower_pp > 0.0
    attempt_improves = e2_vs_paper_attempt_ci_lower_pp > 0.0
    precision_not_materially_worse = (
        e2_vs_paper_precision_ci_lower_pp >= -materiality_pp
    )

    if (
        mc1_not_materially_worse
        and compliance_improves
        and attempt_improves
        and precision_not_materially_worse
    ):
        return "clean_improvement"
    if e2_vs_paper_compliance_delta_pp > 0.0 and e2_vs_paper_mc1_delta_pp < 0.0:
        return "gentler_but_weaker_tradeoff"
    if e2_vs_paper_mc1_delta_pp > 0.0 and e2_vs_paper_compliance_delta_pp <= 0.0:
        return "discriminative_only_improvement"
    if (
        abs(e2_vs_paper_mc1_delta_pp) < materiality_pp
        and abs(e2_vs_paper_compliance_delta_pp) < materiality_pp
    ):
        return "no_meaningful_change"
    return "obvious_regression"


def main() -> None:
    args = parse_args()
    e2_lock = _load_json(args.e2_locked_config)
    paper_lock = _load_json(args.paper_locked_config)
    e1_lock = _load_json(args.e1_locked_config)

    e2_alpha = float(e2_lock["alpha_locked"])
    e2_k = int(e2_lock["K_locked"])
    paper_alpha = float(paper_lock["alpha_locked"])
    paper_k = int(paper_lock["K_locked"])
    e1_alpha = float(e1_lock["alpha_locked"])
    e1_k = int(e1_lock["K_locked"])

    e2_mc1_base = _merge_fold_maps(
        _read_mc_fold_map(args.e2_mc1_fold0_dir, 0.0, "mc1"),
        _read_mc_fold_map(args.e2_mc1_fold1_dir, 0.0, "mc1"),
    )
    e2_mc1_lock = _merge_fold_maps(
        _read_mc_fold_map(args.e2_mc1_fold0_dir, e2_alpha, "mc1"),
        _read_mc_fold_map(args.e2_mc1_fold1_dir, e2_alpha, "mc1"),
    )
    e2_mc2_base = _merge_fold_maps(
        _read_mc_fold_map(args.e2_mc2_fold0_dir, 0.0, "mc2"),
        _read_mc_fold_map(args.e2_mc2_fold1_dir, 0.0, "mc2"),
    )
    e2_mc2_lock = _merge_fold_maps(
        _read_mc_fold_map(args.e2_mc2_fold0_dir, e2_alpha, "mc2"),
        _read_mc_fold_map(args.e2_mc2_fold1_dir, e2_alpha, "mc2"),
    )

    e2_mc1_within = _paired_delta_from_maps(
        e2_mc1_base, e2_mc1_lock, variant="mc1", seed=int(args.seed)
    )
    e2_mc2_within = _paired_delta_from_maps(
        e2_mc2_base, e2_mc2_lock, variant="mc2", seed=int(args.seed)
    )

    paper_mc1_lock = _merge_fold_maps(
        _read_mc_fold_map(args.paper_mc1_fold0_dir, paper_alpha, "mc1"),
        _read_mc_fold_map(args.paper_mc1_fold1_dir, paper_alpha, "mc1"),
    )
    paper_mc2_lock = _merge_fold_maps(
        _read_mc_fold_map(args.paper_mc2_fold0_dir, paper_alpha, "mc2"),
        _read_mc_fold_map(args.paper_mc2_fold1_dir, paper_alpha, "mc2"),
    )
    e1_mc1_lock = _merge_fold_maps(
        _read_mc_fold_map(args.e1_mc1_fold0_dir, e1_alpha, "mc1"),
        _read_mc_fold_map(args.e1_mc1_fold1_dir, e1_alpha, "mc1"),
    )
    e1_mc2_lock = _merge_fold_maps(
        _read_mc_fold_map(args.e1_mc2_fold0_dir, e1_alpha, "mc2"),
        _read_mc_fold_map(args.e1_mc2_fold1_dir, e1_alpha, "mc2"),
    )

    e2_vs_paper_mc1 = _paired_delta_from_maps(
        paper_mc1_lock, e2_mc1_lock, variant="mc1", seed=int(args.seed)
    )
    e2_vs_paper_mc2 = _paired_delta_from_maps(
        paper_mc2_lock, e2_mc2_lock, variant="mc2", seed=int(args.seed)
    )
    e2_vs_e1_mc1 = _paired_delta_from_maps(
        e1_mc1_lock, e2_mc1_lock, variant="mc1", seed=int(args.seed)
    )
    e2_vs_e1_mc2 = _paired_delta_from_maps(
        e1_mc2_lock, e2_mc2_lock, variant="mc2", seed=int(args.seed)
    )

    e2_simpleqa_base = _read_simpleqa_grade_map(args.e2_simpleqa_dir, 0.0)
    e2_simpleqa_lock = _read_simpleqa_grade_map(args.e2_simpleqa_dir, e2_alpha)
    paper_simpleqa_lock = _read_simpleqa_grade_map(args.paper_simpleqa_dir, paper_alpha)
    e1_simpleqa_lock = _read_simpleqa_grade_map(args.e1_simpleqa_dir, e1_alpha)

    e2_simpleqa_base_summary = _simpleqa_summary(e2_simpleqa_base)
    e2_simpleqa_lock_summary = _simpleqa_summary(e2_simpleqa_lock)
    e2_simpleqa_within = _paired_simpleqa_delta(
        e2_simpleqa_base, e2_simpleqa_lock, seed=int(args.seed)
    )
    e2_vs_paper_simpleqa = _paired_simpleqa_delta(
        paper_simpleqa_lock, e2_simpleqa_lock, seed=int(args.seed)
    )
    e2_vs_e1_simpleqa = _paired_simpleqa_delta(
        e1_simpleqa_lock, e2_simpleqa_lock, seed=int(args.seed)
    )

    head_vs_paper = _head_diagnostics(
        args.e2_artifact, e2_k, args.paper_artifact, paper_k
    )
    head_vs_e1 = _head_diagnostics(args.e2_artifact, e2_k, args.e1_artifact, e1_k)
    pilot_report = _load_json(args.e2_pilot_report) if args.e2_pilot_report else None

    classification = _classify_outcome(
        e2_vs_paper_mc1_delta_pp=float(e2_vs_paper_mc1["delta"]["estimate_pp"]),
        e2_vs_paper_mc1_ci_lower_pp=float(e2_vs_paper_mc1["delta"]["ci_pp"]["lower"]),
        e2_vs_paper_compliance_delta_pp=float(
            e2_vs_paper_simpleqa["compliance_delta"]["estimate_pp"]
        ),
        e2_vs_paper_compliance_ci_lower_pp=float(
            e2_vs_paper_simpleqa["compliance_delta"]["ci_pp"]["lower"]
        ),
        e2_vs_paper_attempt_delta_pp=float(
            e2_vs_paper_simpleqa["attempt_delta"]["estimate_pp"]
        ),
        e2_vs_paper_attempt_ci_lower_pp=float(
            e2_vs_paper_simpleqa["attempt_delta"]["ci_pp"]["lower"]
        ),
        e2_vs_paper_precision_delta_pp=float(
            e2_vs_paper_simpleqa["precision_delta"]["estimate_pp"]
        ),
        e2_vs_paper_precision_ci_lower_pp=float(
            e2_vs_paper_simpleqa["precision_delta"]["ci_pp"]["lower"]
        ),
        materiality_pp=float(args.materiality_pp),
    )

    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "e2_name": args.e2_name,
        "config": {
            "e2": {
                "lock_config": str(args.e2_locked_config),
                "alpha_locked": e2_alpha,
                "k_locked": e2_k,
                "artifact": str(args.e2_artifact),
            },
            "paper": {
                "lock_config": str(args.paper_locked_config),
                "alpha_locked": paper_alpha,
                "k_locked": paper_k,
                "artifact": str(args.paper_artifact),
            },
            "e1": {
                "lock_config": str(args.e1_locked_config),
                "alpha_locked": e1_alpha,
                "k_locked": e1_k,
                "artifact": str(args.e1_artifact),
            },
            "materiality_pp": float(args.materiality_pp),
        },
        "pilot": pilot_report,
        "truthfulqa": {
            "within_e2": {"mc1": e2_mc1_within, "mc2": e2_mc2_within},
            "e2_vs_paper": {"mc1": e2_vs_paper_mc1, "mc2": e2_vs_paper_mc2},
            "e2_vs_e1": {"mc1": e2_vs_e1_mc1, "mc2": e2_vs_e1_mc2},
        },
        "simpleqa": {
            "within_e2": {
                "baseline": e2_simpleqa_base_summary,
                "locked": e2_simpleqa_lock_summary,
                "paired": e2_simpleqa_within,
            },
            "e2_vs_paper": e2_vs_paper_simpleqa,
            "e2_vs_e1": e2_vs_e1_simpleqa,
        },
        "diagnostics": {
            "required_transition_counts": e2_simpleqa_within[
                "required_transition_counts"
            ],
            "head_overlap_vs_paper": head_vs_paper,
            "head_overlap_vs_e1": head_vs_e1,
        },
        "classification": classification,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")

    md = [
        f"# {args.e2_name} Audit — {datetime.now().date().isoformat()}",
        "",
        "## Source Hierarchy",
        "",
        f"- Canonical E2 lock: `{args.e2_locked_config}`",
        f"- Consolidated machine-readable report: `{args.output_json}`",
    ]
    if args.e2_pilot_report:
        md.append(f"- Shortlist generation pilot report: `{args.e2_pilot_report}`")
    md.extend(
        [
            "",
            "## 1. Data (Observations Only)",
            "",
            "### 1.1 TruthfulQA",
            (
                f"- E2 vs α=0 MC1 delta: {e2_mc1_within['delta']['estimate_pp']:+.2f} pp "
                f"(95% CI [{e2_mc1_within['delta']['ci_pp']['lower']:+.2f}, "
                f"{e2_mc1_within['delta']['ci_pp']['upper']:+.2f}])."
            ),
            (
                f"- E2 vs α=0 MC2 delta: {e2_mc2_within['delta']['estimate_pp']:+.2f} pp "
                f"(95% CI [{e2_mc2_within['delta']['ci_pp']['lower']:+.2f}, "
                f"{e2_mc2_within['delta']['ci_pp']['upper']:+.2f}])."
            ),
            (
                f"- E2 vs paper MC1 delta: {e2_vs_paper_mc1['delta']['estimate_pp']:+.2f} pp "
                f"(95% CI [{e2_vs_paper_mc1['delta']['ci_pp']['lower']:+.2f}, "
                f"{e2_vs_paper_mc1['delta']['ci_pp']['upper']:+.2f}])."
            ),
            (
                f"- E2 vs paper MC2 delta: {e2_vs_paper_mc2['delta']['estimate_pp']:+.2f} pp "
                f"(95% CI [{e2_vs_paper_mc2['delta']['ci_pp']['lower']:+.2f}, "
                f"{e2_vs_paper_mc2['delta']['ci_pp']['upper']:+.2f}])."
            ),
            "",
            "### 1.2 SimpleQA 200-ID",
            (
                f"- E2 vs α=0 compliance delta: "
                f"{e2_simpleqa_within['compliance_delta']['estimate_pp']:+.2f} pp "
                f"(95% CI [{e2_simpleqa_within['compliance_delta']['ci_pp']['lower']:+.2f}, "
                f"{e2_simpleqa_within['compliance_delta']['ci_pp']['upper']:+.2f}])."
            ),
            (
                f"- E2 vs α=0 attempt delta: {e2_simpleqa_within['attempt_delta']['estimate_pp']:+.2f} pp "
                f"(95% CI [{e2_simpleqa_within['attempt_delta']['ci_pp']['lower']:+.2f}, "
                f"{e2_simpleqa_within['attempt_delta']['ci_pp']['upper']:+.2f}])."
            ),
            (
                f"- E2 vs α=0 precision delta: "
                f"{e2_simpleqa_within['precision_delta']['estimate_pp']:+.2f} pp "
                f"(95% CI [{e2_simpleqa_within['precision_delta']['ci_pp']['lower']:+.2f}, "
                f"{e2_simpleqa_within['precision_delta']['ci_pp']['upper']:+.2f}])."
            ),
            (
                f"- E2 vs paper compliance delta: "
                f"{e2_vs_paper_simpleqa['compliance_delta']['estimate_pp']:+.2f} pp "
                f"(95% CI [{e2_vs_paper_simpleqa['compliance_delta']['ci_pp']['lower']:+.2f}, "
                f"{e2_vs_paper_simpleqa['compliance_delta']['ci_pp']['upper']:+.2f}])."
            ),
            (
                f"- E2 vs paper attempt delta: "
                f"{e2_vs_paper_simpleqa['attempt_delta']['estimate_pp']:+.2f} pp "
                f"(95% CI [{e2_vs_paper_simpleqa['attempt_delta']['ci_pp']['lower']:+.2f}, "
                f"{e2_vs_paper_simpleqa['attempt_delta']['ci_pp']['upper']:+.2f}])."
            ),
            (
                f"- E2 vs paper precision delta: "
                f"{e2_vs_paper_simpleqa['precision_delta']['estimate_pp']:+.2f} pp "
                f"(95% CI [{e2_vs_paper_simpleqa['precision_delta']['ci_pp']['lower']:+.2f}, "
                f"{e2_vs_paper_simpleqa['precision_delta']['ci_pp']['upper']:+.2f}])."
            ),
            "",
            "### 1.3 Required Transition Diagnostics",
        ]
    )
    for key, value in e2_simpleqa_within["required_transition_counts"].items():
        md.append(f"- {key}: {value}")
    md.extend(
        [
            "",
            "## 2. Interpretation (Inference From Data)",
            "",
            f"- Outcome class under pre-registered logic: `{classification}`.",
            "- Head overlap and direction similarity are treated as descriptive only.",
            "",
            "## 3. Uncertainty Register",
            "",
            "- SimpleQA precision is conditional on attempts and typically noisier than attempt-rate effects.",
            "- All paired deltas require identical sample ID sets; report generation fails fast on mismatches.",
            "",
            "## 4. Decision / Next Steps",
            "",
            f"- Recommended decision status: `{classification}`.",
        ]
    )
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"Saved JSON report: {args.output_json}")
    print(f"Saved Markdown report: {args.output_md}")


if __name__ == "__main__":
    main()
