#!/usr/bin/env python3
"""Review calibration sweep results and summarise K×α landscape.

Prints a formatted table, diagnostic summary, and plateau map to help
a human decide whether to accept the auto-suggested locked config or
override it.

Usage::

    uv run python scripts/review_sweep.py \\
        --sweep_path data/contrastive/truthfulness/iti_truthfulqa_paperfaithful_calibration/sweep_results.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from run_calibration_sweep import select_locked_config


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

MARKER = ">>>"


def _fmt_pct(v: float) -> str:
    """Format a proportion as a percentage string with 2 decimals."""
    return f"{v * 100:.2f}%"


# ---------------------------------------------------------------------------
# Table
# ---------------------------------------------------------------------------


def print_table(
    results: list[dict[str, Any]],
    locked: dict[str, Any],
) -> None:
    """Print a formatted K×α sweep table, marking the auto-suggested pick."""
    header = f"{'':>3}  {'K':>4}  {'α':>6}  {'MC1':>8}  {'MC2':>8}"
    sep = "-" * len(header)
    print("\n" + "=" * len(header))
    print(header)
    print(sep)
    for r in results:
        is_pick = r["k"] == locked["k"] and r["alpha"] == locked["alpha"]
        mark = MARKER if is_pick else "   "
        print(
            f"{mark}  {r['k']:>4}  {r['alpha']:>6.1f}"
            f"  {_fmt_pct(r['mc1']):>8}  {_fmt_pct(r['mc2']):>8}"
        )
    print("=" * len(header))
    print(
        f"\n  Auto-suggested lock: K={locked['k']}, α={locked['alpha']}"
        f"  (MC1={_fmt_pct(locked['mc1'])}, MC2={_fmt_pct(locked['mc2'])})"
    )


# ---------------------------------------------------------------------------
# Diagnostic summary
# ---------------------------------------------------------------------------


def print_diagnostics(
    results: list[dict[str, Any]],
    tolerance_pp: float = 0.5,
) -> None:
    """Print diagnostic summary: best MC1, spike/plateau, correlation, α warning."""
    best_mc1 = max(r["mc1"] for r in results)
    candidates = [r for r in results if (best_mc1 - r["mc1"]) * 100 <= tolerance_pp]

    print("\n── Diagnostic Summary ──")

    # Best MC1 and which combos achieve it
    combo_strs = [f"K={r['k']},α={r['alpha']}" for r in candidates]
    print(f"  Best MC1: {_fmt_pct(best_mc1)}")
    print(
        f"  Combos within {tolerance_pp}pp: {len(candidates)}  "
        f"[{', '.join(combo_strs)}]"
    )

    # Spike vs plateau
    if len(candidates) <= 2:
        shape = "SPIKE (1–2 combos) — fragile optimum, consider robustness"
    else:
        shape = "PLATEAU (3+ combos) — robust optimum"
    print(f"  Optimum shape: {shape}")

    # MC1-MC2 correlation (Pearson across all combos)
    mc1_vals = [r["mc1"] for r in results]
    mc2_vals = [r["mc2"] for r in results]
    if len(set(mc1_vals)) > 1 and len(set(mc2_vals)) > 1:
        mean1 = sum(mc1_vals) / len(mc1_vals)
        mean2 = sum(mc2_vals) / len(mc2_vals)
        cov = sum((a - mean1) * (b - mean2) for a, b in zip(mc1_vals, mc2_vals))
        std1 = (sum((a - mean1) ** 2 for a in mc1_vals)) ** 0.5
        std2 = (sum((b - mean2) ** 2 for b in mc2_vals)) ** 0.5
        corr = cov / (std1 * std2) if std1 > 0 and std2 > 0 else 0.0
        if corr > 0.7:
            direction_note = "strong agreement"
        elif corr > 0.3:
            direction_note = "moderate agreement"
        elif corr > -0.3:
            direction_note = "weak / no relationship"
        else:
            direction_note = "DIVERGENT — MC1 and MC2 disagree on direction"
        print(f"  MC1–MC2 correlation: r={corr:.3f} ({direction_note})")
    else:
        print("  MC1–MC2 correlation: insufficient variation to compute")

    # Max α warning
    max_alpha = max(r["alpha"] for r in candidates)
    if max_alpha > 10.0:
        print(f"  ⚠  Maximum α in candidate set: {max_alpha} (> 10.0 — high!)")
    else:
        print(f"  Max α in candidate set: {max_alpha}")


# ---------------------------------------------------------------------------
# Plateau map
# ---------------------------------------------------------------------------


def print_plateau_map(
    results: list[dict[str, Any]],
    tolerance_pp: float = 1.0,
) -> None:
    """For each K, show the α range where MC1 is within tolerance_pp of global best."""
    best_mc1 = max(r["mc1"] for r in results)

    # Group by K
    by_k: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for r in results:
        by_k[r["k"]].append(r)

    print(f"\n── Plateau Map (MC1 within {tolerance_pp}pp of {_fmt_pct(best_mc1)}) ──")
    print(f"  {'K':>4}  {'α range':>16}  {'# in range':>10}  {'best MC1':>10}")

    for k in sorted(by_k):
        rows = by_k[k]
        in_range = [r for r in rows if (best_mc1 - r["mc1"]) * 100 <= tolerance_pp]
        if in_range:
            alphas = sorted(r["alpha"] for r in in_range)
            alpha_range = f"{alphas[0]:.1f}–{alphas[-1]:.1f}"
            local_best = max(r["mc1"] for r in in_range)
            print(
                f"  {k:>4}  {alpha_range:>16}  {len(in_range):>10}"
                f"  {_fmt_pct(local_best):>10}"
            )
        else:
            print(f"  {k:>4}  {'—':>16}  {0:>10}  {'—':>10}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Review calibration sweep results before locking K×α."
    )
    p.add_argument(
        "--sweep_path",
        type=str,
        required=True,
        help="Path to sweep_results.json produced by run_calibration_sweep.py",
    )
    p.add_argument(
        "--tolerance_pp",
        type=float,
        default=0.5,
        help="Tolerance (in pp) for the selection rule (default: 0.5)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    sweep_path = Path(args.sweep_path)

    if not sweep_path.exists():
        print(f"ERROR: {sweep_path} not found", file=sys.stderr)
        sys.exit(1)

    with open(sweep_path) as f:
        sweep_data = json.load(f)

    results: list[dict[str, Any]] = sweep_data["results"]
    if not results:
        print("ERROR: sweep_results.json contains no results", file=sys.stderr)
        sys.exit(1)

    # Metadata
    print(f"Sweep: {sweep_path}")
    print(f"  Artifact: {sweep_data.get('artifact_path', '?')}")
    print(
        f"  Grid: {len(sweep_data.get('k_values', []))} K"
        f" × {len(sweep_data.get('alpha_values', []))} α"
        f" = {len(results)} combos"
    )
    print(
        f"  Samples: {sweep_data.get('n_mc1_samples', '?')} MC1,"
        f" {sweep_data.get('n_mc2_samples', '?')} MC2"
    )

    locked = select_locked_config(results, tolerance_pp=args.tolerance_pp)

    print_table(results, locked)
    print_diagnostics(results, tolerance_pp=args.tolerance_pp)
    print_plateau_map(results)


if __name__ == "__main__":
    main()
