#!/usr/bin/env python3
"""Holdout evaluator comparison: split on calibration contamination boundary.

Loads the 4-way joined data, removes the 24 records (8 prompt IDs × 3 alphas)
used during CSV2 v3 calibration, and recomputes metrics on the true holdout.
Adds McNemar paired tests and prompt-clustered bootstrap CIs.
"""

from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

INPUT_PATH = ROOT / "data/judge_validation/4way_joined.json"
OUTPUT_PATH = ROOT / "data/judge_validation/holdout_comparison.json"

# ── Contamination boundary ────────────────────────────────────────────
# Source: scripts/csv2_v3_smoke_hardcases.py lines 27-36
# These 8 prompt IDs (× 3 alphas = 24 rows) were used to calibrate the
# CSV2 v3 4th few-shot example.
CONTAMINATED_IDS = frozenset(
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

EVALUATORS = ("csv2v3", "sr", "binary", "csv2v2")
EVAL_NAMES: dict[str, str] = {
    "csv2v3": "CSV2 v3",
    "sr": "StrongREJECT",
    "binary": "Binary Judge",
    "csv2v2": "CSV2 v2",
}

BOOTSTRAP_ITERATIONS = 10_000
BOOTSTRAP_SEED = 42
CONFIDENCE = 0.95


# ── Helpers ───────────────────────────────────────────────────────────


def confusion_counts(records: list[dict], evaluator: str) -> dict[str, int]:
    """TP/TN/FP/FN for one evaluator against gold."""
    tp = tn = fp = fn = 0
    for r in records:
        v = r[evaluator]
        if v is None:
            continue
        g = r["gold"]
        if g and v:
            tp += 1
        elif g and not v:
            fn += 1
        elif not g and v:
            fp += 1
        else:
            tn += 1
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def metrics_from_counts(c: dict[str, int]) -> dict[str, float]:
    """Accuracy, precision, recall, F1 from confusion counts."""
    total = c["tp"] + c["tn"] + c["fp"] + c["fn"]
    accuracy = (c["tp"] + c["tn"]) / total if total else 0.0
    precision = c["tp"] / (c["tp"] + c["fp"]) if (c["tp"] + c["fp"]) else 0.0
    recall = c["tp"] / (c["tp"] + c["fn"]) if (c["tp"] + c["fn"]) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def split_by_contamination(
    records: list[dict],
) -> tuple[list[dict], list[dict]]:
    """Partition into dev (contaminated) and holdout splits."""
    dev = [r for r in records if r["id"] in CONTAMINATED_IDS]
    holdout = [r for r in records if r["id"] not in CONTAMINATED_IDS]
    assert len(dev) == 24, f"Expected 24 dev records, got {len(dev)}"
    assert len(holdout) == 50, f"Expected 50 holdout records, got {len(holdout)}"
    return dev, holdout


def _binom_cdf_le(k: int, n: int, p: float = 0.5) -> float:
    """P(X <= k) for X ~ Binomial(n, p). Exact, via math.comb."""
    return sum(math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k + 1))


def mcnemar_exact_p(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value.

    b = eval_a correct & eval_b wrong
    c = eval_a wrong & eval_b correct
    Under H0: min(b,c) ~ Binomial(b+c, 0.5), two-sided.
    """
    n = b + c
    if n == 0:
        return 1.0
    tail = _binom_cdf_le(min(b, c), n, 0.5)
    return min(2.0 * tail, 1.0)


def mcnemar_discordant(
    records: list[dict], eval_a: str, eval_b: str
) -> tuple[int, int]:
    """Count discordant pairs: (a_right_b_wrong, a_wrong_b_right)."""
    b = c = 0
    for r in records:
        va, vb = r[eval_a], r[eval_b]
        if va is None or vb is None:
            continue
        a_correct = va == r["gold"]
        b_correct = vb == r["gold"]
        if a_correct and not b_correct:
            b += 1
        elif not a_correct and b_correct:
            c += 1
    return b, c


def cluster_bootstrap_ci(
    records: list[dict],
    evaluators: tuple[str, ...],
    n_iter: int,
    seed: int,
    confidence: float,
) -> dict[str, dict[str, dict[str, float]]]:
    """Prompt-clustered bootstrap CIs for accuracy and F1.

    Resamples prompt IDs (not rows) to respect within-prompt dependence.
    """
    # Group by prompt ID
    clusters: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        clusters[r["id"]].append(r)
    prompt_ids = sorted(clusters.keys())
    n_clusters = len(prompt_ids)

    rng = random.Random(seed)
    alpha = 1.0 - confidence
    lo_pct = alpha / 2
    hi_pct = 1.0 - alpha / 2

    # Collect bootstrap samples
    samples: dict[str, dict[str, list[float]]] = {
        e: {"accuracy": [], "f1": []} for e in evaluators
    }

    for _ in range(n_iter):
        # Resample prompt IDs with replacement
        sampled_ids = rng.choices(prompt_ids, k=n_clusters)
        # Expand to rows
        boot_records = []
        for pid in sampled_ids:
            boot_records.extend(clusters[pid])

        for e in evaluators:
            c = confusion_counts(boot_records, e)
            m = metrics_from_counts(c)
            samples[e]["accuracy"].append(m["accuracy"])
            samples[e]["f1"].append(m["f1"])

    # Compute point estimates and percentile CIs
    result: dict[str, dict[str, dict[str, float]]] = {}
    for e in evaluators:
        result[e] = {}
        for metric in ("accuracy", "f1"):
            vals = sorted(samples[e][metric])
            lo_idx = max(0, int(math.floor(lo_pct * n_iter)) - 1)
            hi_idx = min(n_iter - 1, int(math.ceil(hi_pct * n_iter)) - 1)
            # Point estimate from actual data
            c = confusion_counts(records, e)
            m = metrics_from_counts(c)
            result[e][metric] = {
                "point": m[metric],
                "ci_lower": vals[lo_idx],
                "ci_upper": vals[hi_idx],
            }

    return result


def cross_alpha_breakdown(
    records: list[dict], evaluators: tuple[str, ...]
) -> dict[str, dict]:
    """Per-alpha accuracy breakdown."""
    result: dict[str, dict] = {}
    for alpha in (0.0, 1.5, 3.0):
        subset = [r for r in records if r["alpha"] == alpha]
        if not subset:
            continue
        alpha_key = str(alpha)
        entry: dict = {
            "n_records": len(subset),
            "n_harmful": sum(1 for r in subset if r["gold"]),
            "n_safe": sum(1 for r in subset if not r["gold"]),
            "evaluators": {},
        }
        for e in evaluators:
            c = confusion_counts(subset, e)
            m = metrics_from_counts(c)
            entry["evaluators"][e] = {**c, **m}
        result[alpha_key] = entry
    return result


def compute_split_metrics(records: list[dict], evaluators: tuple[str, ...]) -> dict:
    """Full metrics for one split."""
    split_result: dict = {
        "n_records": len(records),
        "n_harmful": sum(1 for r in records if r["gold"]),
        "n_safe": sum(1 for r in records if not r["gold"]),
        "evaluators": {},
    }
    for e in evaluators:
        c = confusion_counts(records, e)
        m = metrics_from_counts(c)
        split_result["evaluators"][e] = {**c, **m}
    return split_result


# ── Main ──────────────────────────────────────────────────────────────


def main() -> None:
    # Load joined data
    with open(INPUT_PATH) as f:
        joined: list[dict] = json.load(f)
    print(f"Loaded {len(joined)} records from {INPUT_PATH.relative_to(ROOT)}")

    # Split
    dev, holdout = split_by_contamination(joined)
    n_holdout_ids = len({r["id"] for r in holdout})
    print(
        f"Split: dev={len(dev)} (8 prompt IDs), "
        f"holdout={len(holdout)} ({n_holdout_ids} prompt IDs)"
    )

    # Per-split metrics
    dev_metrics = compute_split_metrics(dev, EVALUATORS)
    holdout_metrics = compute_split_metrics(holdout, EVALUATORS)

    # McNemar tests on holdout
    mcnemar_results: dict[str, dict] = {}
    for ea, eb in combinations(EVALUATORS, 2):
        b, c = mcnemar_discordant(holdout, ea, eb)
        p = mcnemar_exact_p(b, c)
        more_accurate = ea if b > c else (eb if c > b else "tie")
        mcnemar_results[f"{ea}_vs_{eb}"] = {
            "eval_a": ea,
            "eval_b": eb,
            "b_a_right_b_wrong": b,
            "c_a_wrong_b_right": c,
            "n_discordant": b + c,
            "p_value": p,
            "more_accurate": more_accurate,
        }

    # Cluster bootstrap on holdout
    print(f"Running {BOOTSTRAP_ITERATIONS:,} cluster-bootstrap iterations...")
    bootstrap_results = cluster_bootstrap_ci(
        holdout, EVALUATORS, BOOTSTRAP_ITERATIONS, BOOTSTRAP_SEED, CONFIDENCE
    )

    # Cross-alpha breakdown on holdout
    cross_alpha = cross_alpha_breakdown(holdout, EVALUATORS)

    # Dev vs holdout delta
    delta: dict[str, dict[str, float]] = {}
    for e in EVALUATORS:
        d_acc = dev_metrics["evaluators"][e]["accuracy"]
        h_acc = holdout_metrics["evaluators"][e]["accuracy"]
        d_f1 = dev_metrics["evaluators"][e]["f1"]
        h_f1 = holdout_metrics["evaluators"][e]["f1"]
        delta[e] = {
            "dev_accuracy": d_acc,
            "holdout_accuracy": h_acc,
            "accuracy_delta": h_acc - d_acc,
            "dev_f1": d_f1,
            "holdout_f1": h_f1,
            "f1_delta": h_f1 - d_f1,
        }

    # Assemble output
    output = {
        "metadata": {
            "input_path": str(INPUT_PATH.relative_to(ROOT)),
            "contaminated_ids": sorted(CONTAMINATED_IDS),
            "n_contaminated_ids": len(CONTAMINATED_IDS),
            "n_holdout_ids": n_holdout_ids,
            "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "confidence_level": CONFIDENCE,
        },
        "splits": {
            "dev": dev_metrics,
            "holdout": holdout_metrics,
        },
        "mcnemar_tests": mcnemar_results,
        "bootstrap_cis": {
            "method": "cluster_bootstrap_by_prompt_id",
            "n_clusters": n_holdout_ids,
            "n_iterations": BOOTSTRAP_ITERATIONS,
            "confidence": CONFIDENCE,
            "evaluators": bootstrap_results,
        },
        "cross_alpha_holdout": cross_alpha,
        "dev_vs_holdout_delta": delta,
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Output written to {OUTPUT_PATH.relative_to(ROOT)}")

    # ── Print summary ─────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("HOLDOUT EVALUATOR COMPARISON")
    print(
        f"({holdout_metrics['n_records']} records, {n_holdout_ids} prompt clusters, "
        f"{len(CONTAMINATED_IDS)} contaminated IDs excluded)"
    )
    print("=" * 90)

    # Holdout metrics table
    print(
        f"\n{'':18s} {'Acc':>7s} {'Prec':>7s} {'Rec':>7s} {'F1':>7s}   "
        f"{'95% CI acc':>16s}  {'95% CI F1':>16s}"
    )
    print("-" * 90)
    for e in EVALUATORS:
        m = holdout_metrics["evaluators"][e]
        ci = bootstrap_results[e]
        print(
            f"{EVAL_NAMES[e]:18s} {m['accuracy']:7.3f} {m['precision']:7.3f} "
            f"{m['recall']:7.3f} {m['f1']:7.3f}   "
            f"[{ci['accuracy']['ci_lower']:.3f}, {ci['accuracy']['ci_upper']:.3f}]  "
            f"[{ci['f1']['ci_lower']:.3f}, {ci['f1']['ci_upper']:.3f}]"
        )

    # McNemar table
    print(f"\n{'McNemar pairwise tests (exact, two-sided) — holdout only':}")
    print("-" * 70)
    for key, res in mcnemar_results.items():
        a_name = EVAL_NAMES[res["eval_a"]]
        b_name = EVAL_NAMES[res["eval_b"]]
        print(
            f"  {a_name} vs {b_name}: "
            f"b={res['b_a_right_b_wrong']}, c={res['c_a_wrong_b_right']}, "
            f"n={res['n_discordant']}, p={res['p_value']:.3f}"
        )

    # Dev vs holdout
    print(f"\n{'Dev vs Holdout (contamination check)':}")
    print("-" * 70)
    for e in EVALUATORS:
        d = delta[e]
        sign = "+" if d["accuracy_delta"] >= 0 else ""
        print(
            f"  {EVAL_NAMES[e]:18s} dev={d['dev_accuracy']:.3f} → "
            f"holdout={d['holdout_accuracy']:.3f}  "
            f"(Δ={sign}{d['accuracy_delta']:.3f})"
        )

    # Cross-alpha
    print(f"\n{'Cross-alpha accuracy (holdout only)':}")
    print("-" * 70)
    print(f"  {'':18s}", end="")
    for alpha_key in ("0.0", "1.5", "3.0"):
        print(f" {'α=' + alpha_key:>10s}", end="")
    print()
    for e in EVALUATORS:
        print(f"  {EVAL_NAMES[e]:18s}", end="")
        for alpha_key in ("0.0", "1.5", "3.0"):
            if alpha_key in cross_alpha:
                acc = cross_alpha[alpha_key]["evaluators"][e]["accuracy"]
                n = cross_alpha[alpha_key]["n_records"]
                print(f" {acc:>6.3f}({n:>2d})", end="")
            else:
                print(f" {'—':>10s}", end="")
        print()


if __name__ == "__main__":
    main()
