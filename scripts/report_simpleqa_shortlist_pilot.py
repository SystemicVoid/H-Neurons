#!/usr/bin/env python3
"""Summarize SimpleQA shortlist pilot candidates for lock gating.

Expected input is one or more candidate specs in the form:
``K:ALPHA:/abs/path/to/experiment``
where the directory contains judged ``alpha_0.0.jsonl`` and
``alpha_<ALPHA>.jsonl`` files with ``simpleqa_grade``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from uncertainty import DEFAULT_BOOTSTRAP_RESAMPLES, wilson_interval
from utils import format_alpha_label

VALID_GRADES = {"CORRECT", "INCORRECT", "NOT_ATTEMPTED"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate",
        action="append",
        required=True,
        help="Candidate spec as K:ALPHA:EXPERIMENT_DIR (repeatable).",
    )
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument("--baseline_alpha", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--attempt_gate_pp", type=float, default=-10.0)
    parser.add_argument("--precision_gate_pp", type=float, default=0.0)
    parser.add_argument("--not_attempted_gate_n", type=int, default=15)
    return parser.parse_args()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _grade_map(rows: list[dict[str, Any]]) -> dict[str, str]:
    output: dict[str, str] = {}
    for row in rows:
        grade = str(row.get("simpleqa_grade", "")).upper()
        if grade not in VALID_GRADES:
            raise ValueError(
                f"Invalid or missing simpleqa_grade for id={row.get('id')}"
            )
        output[str(row["id"])] = grade
    return output


def _counts(grades: dict[str, str]) -> dict[str, int]:
    return {
        "CORRECT": sum(g == "CORRECT" for g in grades.values()),
        "INCORRECT": sum(g == "INCORRECT" for g in grades.values()),
        "NOT_ATTEMPTED": sum(g == "NOT_ATTEMPTED" for g in grades.values()),
    }


def _precision(correct: int, attempted: int) -> float:
    return (correct / attempted) if attempted > 0 else 0.0


def _preview_ids(ids: list[str], limit: int = 5) -> list[str]:
    return ids[:limit]


def _require_identical_sample_ids(
    baseline_map: dict[str, str],
    candidate_map: dict[str, str],
    *,
    context: str,
) -> list[str]:
    baseline_ids = set(baseline_map)
    candidate_ids = set(candidate_map)
    if baseline_ids != candidate_ids:
        missing_in_candidate = sorted(baseline_ids - candidate_ids)
        missing_in_baseline = sorted(candidate_ids - baseline_ids)
        raise ValueError(
            f"{context}: paired sample IDs must match exactly "
            f"(missing_in_candidate={len(missing_in_candidate)} "
            f"sample={_preview_ids(missing_in_candidate)}; "
            f"missing_in_baseline={len(missing_in_baseline)} "
            f"sample={_preview_ids(missing_in_baseline)})."
        )
    if not baseline_ids:
        raise ValueError(f"{context}: no sample IDs found for paired comparison")
    return sorted(baseline_ids)


def _bootstrap_precision_delta_pp(
    baseline_grades: list[str],
    candidate_grades: list[str],
    *,
    n_resamples: int,
    seed: int,
) -> dict[str, Any]:
    baseline_arr = np.array(baseline_grades, dtype=object)
    candidate_arr = np.array(candidate_grades, dtype=object)
    if baseline_arr.shape != candidate_arr.shape:
        raise ValueError("Precision bootstrap arrays must have matching shapes")

    n = len(baseline_arr)
    rng = np.random.default_rng(seed)
    samples = np.empty(n_resamples, dtype=float)

    def _compute_precision(arr: np.ndarray) -> float:
        attempted = np.sum(arr != "NOT_ATTEMPTED")
        correct = np.sum(arr == "CORRECT")
        return _precision(int(correct), int(attempted))

    for idx in range(n_resamples):
        sample_idx = rng.choice(n, size=n, replace=True)
        b = baseline_arr[sample_idx]
        c = candidate_arr[sample_idx]
        samples[idx] = (_compute_precision(c) - _compute_precision(b)) * 100.0

    lo, hi = np.quantile(samples, [0.025, 0.975])
    point = (
        _compute_precision(candidate_arr) - _compute_precision(baseline_arr)
    ) * 100.0
    return {
        "estimate_pp": float(point),
        "ci_pp": {
            "lower": float(lo),
            "upper": float(hi),
            "level": 0.95,
            "method": "bootstrap_percentile_paired_conditional_precision",
        },
        "bootstrap": {
            "n_resamples": int(n_resamples),
            "seed": int(seed),
            "resampling": "paired_by_sample_id",
        },
    }


def _parse_candidate_spec(raw: str) -> tuple[int, float, Path]:
    try:
        k_raw, alpha_raw, dir_raw = raw.split(":", 2)
    except ValueError as exc:
        raise ValueError("Candidate format must be K:ALPHA:EXPERIMENT_DIR") from exc
    return int(k_raw), float(alpha_raw), Path(dir_raw)


def _summary_for_candidate(
    *,
    k: int,
    alpha: float,
    experiment_dir: Path,
    baseline_alpha: float,
    seed: int,
    attempt_gate_pp: float,
    precision_gate_pp: float,
    not_attempted_gate_n: int,
) -> dict[str, Any]:
    baseline_label = format_alpha_label(baseline_alpha)
    candidate_label = format_alpha_label(alpha)
    baseline_path = experiment_dir / f"alpha_{baseline_label}.jsonl"
    candidate_path = experiment_dir / f"alpha_{candidate_label}.jsonl"

    if not baseline_path.exists():
        raise FileNotFoundError(f"Missing baseline file: {baseline_path}")
    if not candidate_path.exists():
        raise FileNotFoundError(f"Missing candidate file: {candidate_path}")

    baseline_rows = _load_jsonl(baseline_path)
    candidate_rows = _load_jsonl(candidate_path)
    baseline_map = _grade_map(baseline_rows)
    candidate_map = _grade_map(candidate_rows)

    common_ids = _require_identical_sample_ids(
        baseline_map,
        candidate_map,
        context=f"{experiment_dir}",
    )

    baseline_grades = [baseline_map[sid] for sid in common_ids]
    candidate_grades = [candidate_map[sid] for sid in common_ids]

    baseline_counts = _counts({sid: baseline_map[sid] for sid in common_ids})
    candidate_counts = _counts({sid: candidate_map[sid] for sid in common_ids})

    n_total = len(common_ids)
    baseline_attempted = baseline_counts["CORRECT"] + baseline_counts["INCORRECT"]
    candidate_attempted = candidate_counts["CORRECT"] + candidate_counts["INCORRECT"]

    baseline_attempt_rate = baseline_attempted / n_total
    candidate_attempt_rate = candidate_attempted / n_total
    attempt_delta_pp = (candidate_attempt_rate - baseline_attempt_rate) * 100.0

    baseline_precision = _precision(baseline_counts["CORRECT"], baseline_attempted)
    candidate_precision = _precision(candidate_counts["CORRECT"], candidate_attempted)
    precision_delta_pp = (candidate_precision - baseline_precision) * 100.0

    not_attempted_delta_n = (
        candidate_counts["NOT_ATTEMPTED"] - baseline_counts["NOT_ATTEMPTED"]
    )

    transitions: dict[str, int] = {}
    for before, after in zip(baseline_grades, candidate_grades, strict=False):
        key = f"{before}->{after}"
        transitions[key] = transitions.get(key, 0) + 1

    reject = False
    reject_reasons: list[str] = []
    if attempt_delta_pp <= attempt_gate_pp and precision_delta_pp <= precision_gate_pp:
        reject = True
        reject_reasons.append("attempt_and_precision_gate")
    if not_attempted_delta_n >= not_attempted_gate_n:
        reject = True
        reject_reasons.append("not_attempted_spike_gate")

    return {
        "k": int(k),
        "alpha": float(alpha),
        "experiment_dir": str(experiment_dir),
        "n_samples": int(n_total),
        "baseline_counts": baseline_counts,
        "candidate_counts": candidate_counts,
        "attempt_rate": {
            "baseline": float(baseline_attempt_rate),
            "candidate": float(candidate_attempt_rate),
            "delta_pp": float(attempt_delta_pp),
        },
        "precision": {
            "baseline": float(baseline_precision),
            "candidate": float(candidate_precision),
            "delta_pp": float(precision_delta_pp),
            "delta_bootstrap": _bootstrap_precision_delta_pp(
                baseline_grades,
                candidate_grades,
                n_resamples=DEFAULT_BOOTSTRAP_RESAMPLES,
                seed=seed,
            ),
            "baseline_ci": wilson_interval(
                baseline_counts["CORRECT"], max(baseline_attempted, 1)
            ).to_dict(),
            "candidate_ci": wilson_interval(
                candidate_counts["CORRECT"], max(candidate_attempted, 1)
            ).to_dict(),
        },
        "not_attempted_delta_n": int(not_attempted_delta_n),
        "attempt_delta_pp": float(attempt_delta_pp),
        "precision_delta_pp": float(precision_delta_pp),
        "transitions": transitions,
        "poison_gate_reject": bool(reject),
        "poison_gate_reasons": reject_reasons,
    }


def main() -> None:
    args = parse_args()
    rows = []
    for raw in args.candidate:
        k, alpha, experiment_dir = _parse_candidate_spec(raw)
        rows.append(
            _summary_for_candidate(
                k=k,
                alpha=alpha,
                experiment_dir=experiment_dir,
                baseline_alpha=float(args.baseline_alpha),
                seed=int(args.seed),
                attempt_gate_pp=float(args.attempt_gate_pp),
                precision_gate_pp=float(args.precision_gate_pp),
                not_attempted_gate_n=int(args.not_attempted_gate_n),
            )
        )

    payload = {
        "baseline_alpha": float(args.baseline_alpha),
        "poison_gate": {
            "attempt_delta_pp_lte": float(args.attempt_gate_pp),
            "precision_delta_pp_lte": float(args.precision_gate_pp),
            "not_attempted_delta_n_gte": int(args.not_attempted_gate_n),
            "logic": "reject_if((attempt<=A and precision<=P) or not_attempted_delta>=N)",
        },
        "candidates": rows,
        "n_candidates": len(rows),
        "n_rejected": sum(row["poison_gate_reject"] for row in rows),
    }

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Saved pilot report: {args.output_path}")


if __name__ == "__main__":
    main()
