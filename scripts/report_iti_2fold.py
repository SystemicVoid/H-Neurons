#!/usr/bin/env python3
"""Aggregate ITI TruthfulQA results with paired bootstrap CIs.

Reads per-fold JSONL outputs (alpha_0.0.jsonl and alpha_{locked}.jsonl)
from one or more fold/held-out directories, computes per-split and pooled
MC1/MC2 deltas using paired bootstrap, and writes a structured report.

Usage::

    uv run python scripts/report_iti_2fold.py \\
        --fold0_dir data/gemma3_4b/intervention/.../fold0/experiment \\
        --fold1_dir data/gemma3_4b/intervention/.../fold1/experiment \\
        --locked_alpha 4.0 \\
        --locked_k 48 \\
        --variant mc1 \\
        --output_dir notes/act3-reports
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from uncertainty import (
    paired_bootstrap_binary_rate_difference,
    paired_bootstrap_continuous_mean_difference,
    percentile_interval,
    wilson_interval,
)
from utils import format_alpha_label


# ---------------------------------------------------------------------------
# Load fold data
# ---------------------------------------------------------------------------


def load_fold_records(fold_dir: str, alpha_label: str) -> list[dict[str, Any]]:
    """Load JSONL records for a given alpha label from a fold directory."""
    path = Path(fold_dir) / f"alpha_{alpha_label}.jsonl"
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def extract_mc1_correctness(records: list[dict[str, Any]]) -> dict[str, bool]:
    """Extract per-sample binary correctness for MC1 (top choice truthful)."""
    return {r["id"]: bool(r["compliance"]) for r in records}


def extract_mc2_truthful_mass(records: list[dict[str, Any]]) -> dict[str, float]:
    """Extract per-sample truthful mass for MC2 (continuous 0–1)."""
    return {r["id"]: float(r["metric_value"]) for r in records}


def assert_unique_sample_ids(
    records: list[dict[str, Any]],
    *,
    label: str,
    fold_idx: int,
    variant: str,
) -> None:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for row in records:
        sample_id = str(row["id"])
        if sample_id in seen:
            duplicates.add(sample_id)
        else:
            seen.add(sample_id)
    if duplicates:
        raise ValueError(
            f"Duplicate sample IDs in {label} alpha file "
            f"(fold={fold_idx}, variant={variant}): {sorted(duplicates)[:5]}"
        )


def mcnemar_p_value(
    baseline_correct: np.ndarray, intervened_correct: np.ndarray
) -> float:
    """Compute McNemar's test p-value for paired binary outcomes."""
    from scipy.stats import binomtest

    # Discordant pairs
    b = int(np.sum(baseline_correct & ~intervened_correct))  # correct→wrong
    c = int(np.sum(~baseline_correct & intervened_correct))  # wrong→correct
    n_discordant = b + c
    if n_discordant == 0:
        return 1.0
    # Exact binomial test (two-sided)
    result = binomtest(min(b, c), n_discordant, 0.5, alternative="two-sided")
    return float(result.pvalue)


def _bootstrap_mean_ci(
    values: np.ndarray,
    *,
    confidence: float = 0.95,
    n_resamples: int = 10_000,
    seed: int = 42,
) -> dict[str, Any]:
    """Bootstrap CI on the mean of a continuous array."""
    rng = np.random.default_rng(seed)
    n = len(values)
    means = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        means[i] = values[rng.choice(n, size=n, replace=True)].mean()
    ci = percentile_interval(means, confidence, method="bootstrap_percentile")
    return {
        "estimate": float(values.mean()),
        "ci": ci.to_dict(),
    }


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def compute_fold_report(
    fold_dir: str,
    locked_alpha: float,
    variant: str,
    fold_idx: int,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    """Compute single-fold baseline vs intervened comparison.

    Returns (report_dict, baseline_values, intervened_values) where values
    are aligned numpy arrays (bool for mc1, float for mc2).
    """
    baseline_label = format_alpha_label(0.0)
    locked_label = format_alpha_label(locked_alpha)

    baseline_records = load_fold_records(fold_dir, baseline_label)
    locked_records = load_fold_records(fold_dir, locked_label)
    assert_unique_sample_ids(
        baseline_records, label="baseline", fold_idx=fold_idx, variant=variant
    )
    assert_unique_sample_ids(
        locked_records, label="intervened", fold_idx=fold_idx, variant=variant
    )

    if variant == "mc1":
        baseline_map = extract_mc1_correctness(baseline_records)
        locked_map = extract_mc1_correctness(locked_records)
    else:
        baseline_map = extract_mc2_truthful_mass(baseline_records)
        locked_map = extract_mc2_truthful_mass(locked_records)

    # Align by sample ID. Claim-bearing reports must fail instead of silently
    # dropping mismatched rows from an incomplete alpha file.
    baseline_ids = set(baseline_map)
    locked_ids = set(locked_map)
    if baseline_ids != locked_ids:
        missing_from_locked = sorted(baseline_ids - locked_ids)
        missing_from_baseline = sorted(locked_ids - baseline_ids)
        raise ValueError(
            "paired sample IDs must match exactly for TruthfulQA ITI report "
            f"(fold={fold_idx}, variant={variant}); "
            f"missing_from_locked={missing_from_locked[:5]}, "
            f"missing_from_baseline={missing_from_baseline[:5]}"
        )
    if not baseline_ids:
        raise ValueError(
            "TruthfulQA ITI report has no paired sample IDs "
            f"(fold={fold_idx}, variant={variant})"
        )
    common_ids = sorted(baseline_ids)

    if variant == "mc1":
        baseline_arr = np.array([baseline_map[sid] for sid in common_ids], dtype=bool)
        locked_arr = np.array([locked_map[sid] for sid in common_ids], dtype=bool)

        baseline_rate = float(baseline_arr.mean())
        locked_rate = float(locked_arr.mean())
        delta = paired_bootstrap_binary_rate_difference(baseline_arr, locked_arr)
        baseline_ci = wilson_interval(int(baseline_arr.sum()), len(baseline_arr))
        locked_ci = wilson_interval(int(locked_arr.sum()), len(locked_arr))

        report: dict[str, Any] = {
            "fold": fold_idx,
            "variant": variant,
            "n_samples": len(common_ids),
            "baseline": {
                "alpha": 0.0,
                "rate": round(baseline_rate, 6),
                "ci": baseline_ci.to_dict(),
            },
            "intervened": {
                "alpha": locked_alpha,
                "rate": round(locked_rate, 6),
                "ci": locked_ci.to_dict(),
            },
            "delta_pp": delta,
            "mcnemar_p": mcnemar_p_value(baseline_arr, locked_arr),
        }
    else:
        baseline_arr = np.array([baseline_map[sid] for sid in common_ids], dtype=float)
        locked_arr = np.array([locked_map[sid] for sid in common_ids], dtype=float)

        baseline_mean = float(baseline_arr.mean())
        locked_mean = float(locked_arr.mean())
        delta = paired_bootstrap_continuous_mean_difference(baseline_arr, locked_arr)
        baseline_ci = _bootstrap_mean_ci(baseline_arr)
        locked_ci = _bootstrap_mean_ci(locked_arr)

        report = {
            "fold": fold_idx,
            "variant": variant,
            "n_samples": len(common_ids),
            "baseline": {
                "alpha": 0.0,
                "rate": round(baseline_mean, 6),
                "ci": baseline_ci["ci"],
            },
            "intervened": {
                "alpha": locked_alpha,
                "rate": round(locked_mean, 6),
                "ci": locked_ci["ci"],
            },
            "delta_pp": delta,
        }

    return report, baseline_arr, locked_arr


def compute_pooled_report(
    variant: str,
    locked_alpha: float,
    fold_arrays: list[tuple[np.ndarray, np.ndarray]],
) -> dict[str, Any]:
    """Pool samples across folds for aggregate statistics."""
    baseline_arr = np.concatenate([a[0] for a in fold_arrays])
    locked_arr = np.concatenate([a[1] for a in fold_arrays])

    if variant == "mc1":
        delta = paired_bootstrap_binary_rate_difference(baseline_arr, locked_arr)
        baseline_ci = wilson_interval(int(baseline_arr.sum()), len(baseline_arr))
        locked_ci = wilson_interval(int(locked_arr.sum()), len(locked_arr))

        report: dict[str, Any] = {
            "variant": variant,
            "n_samples_total": len(baseline_arr),
            "n_folds": len(fold_arrays),
            "baseline": {
                "alpha": 0.0,
                "rate": round(float(baseline_arr.mean()), 6),
                "ci": baseline_ci.to_dict(),
            },
            "intervened": {
                "alpha": locked_alpha,
                "rate": round(float(locked_arr.mean()), 6),
                "ci": locked_ci.to_dict(),
            },
            "delta_pp": delta,
            "mcnemar_p": mcnemar_p_value(baseline_arr, locked_arr),
        }
    else:
        delta = paired_bootstrap_continuous_mean_difference(baseline_arr, locked_arr)
        baseline_ci = _bootstrap_mean_ci(baseline_arr)
        locked_ci = _bootstrap_mean_ci(locked_arr)

        report = {
            "variant": variant,
            "n_samples_total": len(baseline_arr),
            "n_folds": len(fold_arrays),
            "baseline": {
                "alpha": 0.0,
                "rate": round(float(baseline_arr.mean()), 6),
                "ci": baseline_ci["ci"],
            },
            "intervened": {
                "alpha": locked_alpha,
                "rate": round(float(locked_arr.mean()), 6),
                "ci": locked_ci["ci"],
            },
            "delta_pp": delta,
        }

    return report


def build_gate_decision(
    report: dict[str, Any], gate_mode: str
) -> dict[str, Any] | None:
    if gate_mode == "none":
        return None
    if gate_mode != "mc1_positive_ci":
        raise ValueError(f"Unsupported gate_mode: {gate_mode!r}")
    if report["variant"] != "mc1":
        raise ValueError("--gate_mode mc1_positive_ci is only valid with --variant mc1")

    pooled = report["pooled"]
    delta_pp = pooled["delta_pp"]
    estimate_pp = float(delta_pp["estimate_pp"])
    lower_pp = float(delta_pp["ci_pp"]["lower"])
    passed = estimate_pp > 0.0 and lower_pp > 0.0
    return {
        "mode": "mc1_positive_ci",
        "passed": passed,
        "criterion": "pooled MC1 delta estimate > 0 and paired-bootstrap 95% CI lower bound > 0",
        "estimate_pp": estimate_pp,
        "ci_lower_pp": lower_pp,
        "ci_upper_pp": float(delta_pp["ci_pp"]["upper"]),
        "n_samples_total": int(pooled["n_samples_total"]),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fold0_dir", type=str, default=None)
    p.add_argument("--fold1_dir", type=str, default=None)
    p.add_argument(
        "--fold_dirs",
        type=str,
        nargs="+",
        default=None,
        help=(
            "One or more directories containing alpha_*.jsonl. If omitted, "
            "--fold0_dir and --fold1_dir preserve the original 2-fold interface."
        ),
    )
    p.add_argument("--locked_alpha", type=float, required=True)
    p.add_argument("--locked_k", type=int, required=True)
    p.add_argument("--variant", type=str, default="mc1", choices=["mc1", "mc2"])
    p.add_argument("--output_dir", type=str, default="notes/act3-reports")
    p.add_argument(
        "--output_prefix",
        type=str,
        default="iti_2fold",
        help="Prefix for the default report filename.",
    )
    p.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Explicit JSON report path. Overrides --output_dir/--output_prefix.",
    )
    p.add_argument(
        "--gate_mode",
        type=str,
        default="none",
        choices=["none", "mc1_positive_ci"],
        help="Optional strict launch gate to embed in the report.",
    )
    p.add_argument(
        "--sample_manifest",
        type=str,
        default=None,
        help="Optional sample manifest path to bind the report to a locked sample set.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    if args.output_path is None:
        out.mkdir(parents=True, exist_ok=True)

    if args.fold_dirs is not None:
        fold_dirs = args.fold_dirs
    elif args.fold0_dir is not None and args.fold1_dir is not None:
        fold_dirs = [args.fold0_dir, args.fold1_dir]
    else:
        raise ValueError("Provide --fold_dirs or both --fold0_dir and --fold1_dir")

    fold_reports = []
    fold_arrays = []
    for fold_idx, fold_dir in enumerate(fold_dirs):
        report, baseline_arr, locked_arr = compute_fold_report(
            fold_dir, args.locked_alpha, args.variant, fold_idx
        )
        fold_reports.append(report)
        fold_arrays.append((baseline_arr, locked_arr))
        b = report["baseline"]
        i = report["intervened"]
        d = report["delta_pp"]
        print(
            f"Fold {fold_idx}: {args.variant} baseline={b['rate']:.3f}, "
            f"intervened={i['rate']:.3f}, Δ={d['estimate_pp']:+.1f}pp "
            f"[{d['ci_pp']['lower']:+.1f}, {d['ci_pp']['upper']:+.1f}]"
        )

    pooled = compute_pooled_report(args.variant, args.locked_alpha, fold_arrays)
    b = pooled["baseline"]
    i = pooled["intervened"]
    d = pooled["delta_pp"]
    print(
        f"\nPooled: {args.variant} baseline={b['rate']:.3f}, "
        f"intervened={i['rate']:.3f}, Δ={d['estimate_pp']:+.1f}pp "
        f"[{d['ci_pp']['lower']:+.1f}, {d['ci_pp']['upper']:+.1f}]"
    )
    if "mcnemar_p" in pooled:
        print(f"  McNemar p={pooled['mcnemar_p']:.4f}")

    # Write JSON report
    full_report = {
        "locked_alpha": args.locked_alpha,
        "locked_k": args.locked_k,
        "variant": args.variant,
        "sample_manifest": args.sample_manifest,
        "fold_dirs": fold_dirs,
        "folds": fold_reports,
        "pooled": pooled,
    }
    gate = build_gate_decision(full_report, args.gate_mode)
    if gate is not None:
        full_report["gate"] = gate
        print(
            f"  Gate {gate['mode']}: "
            f"{'PASS' if gate['passed'] else 'FAIL'} "
            f"(estimate={gate['estimate_pp']:+.1f}pp, "
            f"lower={gate['ci_lower_pp']:+.1f}pp)"
        )

    report_path = (
        Path(args.output_path)
        if args.output_path
        else out / f"{args.output_prefix}_{args.variant}_report.json"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(full_report, f, indent=2)
        f.write("\n")
    print(f"\n→ {report_path}")


if __name__ == "__main__":
    main()
