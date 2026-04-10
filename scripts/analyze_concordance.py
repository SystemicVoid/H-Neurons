"""Three-judge concordance analysis for jailbreak intervention data.

Joins binary judge, CSV2 v2, and CSV2 v3 annotations on the same responses
and computes pairwise agreement, alpha-stratified concordance, structural
correlation on shared axes, and intervention-delta sign stability.

Run AFTER all three judge layers have been applied to both H-neuron and
control data.

Usage:
    uv run python scripts/analyze_concordance.py

    uv run python scripts/analyze_concordance.py \\
        --h_neuron_binary_dir data/gemma3_4b/intervention/jailbreak/experiment \\
        --h_neuron_csv2_v2_dir data/gemma3_4b/intervention/jailbreak/csv2_evaluation \\
        --h_neuron_csv2_v3_dir data/gemma3_4b/intervention/jailbreak/csv2_v3_evaluation \\
        --control_base data/gemma3_4b/intervention/jailbreak/control \\
        --control_seeds 0 1 2 \\
        --alphas 0.0 1.0 1.5 3.0 \\
        --gold_labels tests/gold_labels/jailbreak_cross_alpha_gold.jsonl \\
        --output_dir data/gemma3_4b/intervention/jailbreak/control/concordance
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as sp_stats

from evaluate_csv2 import normalize_csv2_payload
from uncertainty import build_rate_summary

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_H_BINARY = Path("data/gemma3_4b/intervention/jailbreak/experiment")
DEFAULT_H_V2 = Path("data/gemma3_4b/intervention/jailbreak/csv2_evaluation")
DEFAULT_H_V3 = Path("data/gemma3_4b/intervention/jailbreak/csv2_v3_evaluation")
DEFAULT_CONTROL_BASE = Path("data/gemma3_4b/intervention/jailbreak/control")
DEFAULT_GOLD = Path("tests/gold_labels/jailbreak_cross_alpha_gold.jsonl")
DEFAULT_OUTPUT = Path("data/gemma3_4b/intervention/jailbreak/control/concordance")
DEFAULT_ALPHAS = [0.0, 1.0, 1.5, 3.0]
DEFAULT_SEEDS = [0, 1, 2]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def load_alpha(directory: Path, alpha: float) -> list[dict]:
    path = directory / f"alpha_{alpha:.1f}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Missing: {path}")
    return load_jsonl(path)


def _safe_csv2(rec: dict) -> dict | None:
    """Extract and normalize csv2 payload, returning None on failure."""
    raw = rec.get("csv2")
    if not isinstance(raw, dict) or not raw or raw.get("error"):
        return None
    return normalize_csv2_payload(raw)


def _raw_csv2(rec: dict) -> dict | None:
    """Extract raw csv2 payload without normalization."""
    raw = rec.get("csv2")
    if not isinstance(raw, dict) or not raw or raw.get("error"):
        return None
    return raw


# ---------------------------------------------------------------------------
# Joining
# ---------------------------------------------------------------------------


def join_condition(
    binary_dir: Path,
    v2_dir: Path,
    v3_dir: Path,
    alphas: list[float],
    condition_name: str,
) -> list[dict]:
    """Join three annotation layers on (id, alpha) for one condition."""
    joined: list[dict] = []
    for alpha in alphas:
        try:
            binary_recs = load_alpha(binary_dir, alpha)
        except FileNotFoundError:
            print(f"  WARN: {condition_name} binary alpha={alpha:.1f} not found")
            continue
        try:
            v2_recs = load_alpha(v2_dir, alpha)
        except FileNotFoundError:
            print(f"  WARN: {condition_name} v2 alpha={alpha:.1f} not found")
            continue
        try:
            v3_recs = load_alpha(v3_dir, alpha)
        except FileNotFoundError:
            print(f"  WARN: {condition_name} v3 alpha={alpha:.1f} not found")
            continue

        v2_by_id = {r["id"]: r for r in v2_recs}
        v3_by_id = {r["id"]: r for r in v3_recs}

        n_joined = 0
        for rec in binary_recs:
            rid = rec["id"]
            v2_rec = v2_by_id.get(rid)
            v3_rec = v3_by_id.get(rid)
            if v2_rec is None or v3_rec is None:
                continue

            v2_csv2 = _safe_csv2(v2_rec)
            v3_csv2 = _safe_csv2(v3_rec)
            v2_raw = _raw_csv2(v2_rec)
            if v2_csv2 is None or v3_csv2 is None:
                continue

            # Binary judge label → boolean harmful
            binary_judge = rec.get("judge", "").upper()
            binary_harmful = binary_judge == "HARMFUL"

            # V2 harmful_binary (raw, before normalization — preserves borderline)
            v2_harmful_raw = (
                str(v2_raw.get("harmful_binary", "")).lower() if v2_raw else ""
            )
            v2_harmful_binary = v2_harmful_raw == "yes"
            v2_borderline = v2_harmful_raw == "borderline"

            # V3 harmful_binary (native, 2-way)
            v3_harmful_binary = str(v3_csv2.get("harmful_binary", "")).lower() == "yes"

            joined.append(
                {
                    "id": rid,
                    "alpha": alpha,
                    "condition": condition_name,
                    "category": rec.get("category", ""),
                    # Binary layer
                    "binary_harmful": binary_harmful,
                    "binary_judge_raw": binary_judge,
                    # V2 layer
                    "v2_harmful": v2_harmful_binary,
                    "v2_borderline": v2_borderline,
                    "v2_harmful_raw": v2_harmful_raw,
                    "v2_C": v2_csv2.get("C"),
                    "v2_S": v2_csv2.get("S"),
                    "v2_V": v2_csv2.get("V"),
                    # V3 layer
                    "v3_harmful": v3_harmful_binary,
                    "v3_primary_outcome": v3_csv2.get("primary_outcome"),
                    "v3_intent_match": v3_csv2.get("intent_match"),
                    "v3_C": v3_csv2.get("C"),
                    "v3_S": v3_csv2.get("S"),
                    "v3_V": v3_csv2.get("V"),
                }
            )
            n_joined += 1

        print(
            f"  {condition_name} alpha={alpha:.1f}: {n_joined} three-way joined records"
        )
    return joined


# ---------------------------------------------------------------------------
# Agreement metrics
# ---------------------------------------------------------------------------


def cohens_kappa(y1: np.ndarray, y2: np.ndarray) -> float:
    """Compute Cohen's kappa for two binary label arrays."""
    n = len(y1)
    if n == 0:
        return 0.0
    observed = float(np.mean(y1 == y2))
    p1 = float(np.mean(y1))
    p2 = float(np.mean(y2))
    expected = p1 * p2 + (1 - p1) * (1 - p2)
    if abs(expected - 1.0) < 1e-9:
        return 1.0
    return (observed - expected) / (1.0 - expected)


def confusion_matrix_2x2(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, int]:
    """Compute 2x2 confusion matrix counts."""
    tp = int(np.sum(y_true & y_pred))
    tn = int(np.sum(~y_true & ~y_pred))
    fp = int(np.sum(~y_true & y_pred))
    fn = int(np.sum(y_true & ~y_pred))
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def pairwise_agreement(
    records: list[dict],
    key_a: str,
    key_b: str,
    label_a: str,
    label_b: str,
) -> dict[str, Any]:
    """Compute agreement between two binary judge labels."""
    y_a = np.array([bool(r[key_a]) for r in records])
    y_b = np.array([bool(r[key_b]) for r in records])
    n = len(y_a)
    agree = int(np.sum(y_a == y_b))
    return {
        "pair": f"{label_a}_vs_{label_b}",
        "n": n,
        "agreement_rate": build_rate_summary(agree, n),
        "kappa": round(cohens_kappa(y_a, y_b), 4),
        "confusion": confusion_matrix_2x2(y_a, y_b),
        f"{label_a}_positive_rate": round(float(np.mean(y_a)), 4),
        f"{label_b}_positive_rate": round(float(np.mean(y_b)), 4),
    }


def alpha_stratified_agreement(
    records: list[dict],
    alphas: list[float],
    key_a: str,
    key_b: str,
    label_a: str,
    label_b: str,
) -> dict[str, Any]:
    """Compute agreement metrics stratified by alpha."""
    by_alpha: dict[float, list[dict]] = defaultdict(list)
    for r in records:
        by_alpha[r["alpha"]].append(r)

    per_alpha = {}
    for a in alphas:
        recs = by_alpha.get(a, [])
        if not recs:
            continue
        per_alpha[str(a)] = pairwise_agreement(recs, key_a, key_b, label_a, label_b)

    return {
        "pair": f"{label_a}_vs_{label_b}",
        "per_alpha": per_alpha,
    }


# ---------------------------------------------------------------------------
# Structural correlation (v2 vs v3 ordinal axes)
# ---------------------------------------------------------------------------


def structural_correlation(records: list[dict]) -> dict[str, Any]:
    """Compute correlation between v2 and v3 ordinal axes (C, S, V)."""
    results: dict[str, Any] = {}
    for axis in ("C", "S", "V"):
        v2_key = f"v2_{axis}"
        v3_key = f"v3_{axis}"
        pairs = [
            (r[v2_key], r[v3_key])
            for r in records
            if r[v2_key] is not None and r[v3_key] is not None
        ]
        if len(pairs) < 3:
            results[axis] = {"n": len(pairs), "note": "insufficient data"}
            continue
        v2_vals = np.array([p[0] for p in pairs], dtype=float)
        v3_vals = np.array([p[1] for p in pairs], dtype=float)
        pearson_r, pearson_p = sp_stats.pearsonr(v2_vals, v3_vals)
        spearman_r, spearman_p = sp_stats.spearmanr(v2_vals, v3_vals)
        mad = float(np.mean(np.abs(v2_vals - v3_vals)))
        exact_match = float(np.mean(v2_vals == v3_vals))
        results[axis] = {
            "n": len(pairs),
            "pearson_r": round(float(pearson_r), 4),
            "pearson_p": float(pearson_p),
            "spearman_r": round(float(spearman_r), 4),
            "spearman_p": float(spearman_p),
            "mean_abs_diff": round(mad, 4),
            "exact_match_rate": round(exact_match, 4),
        }
    return results


# ---------------------------------------------------------------------------
# Intervention delta sign stability
# ---------------------------------------------------------------------------


def compute_harmful_rates(
    records: list[dict],
    alphas: list[float],
    key: str,
) -> dict[float, dict]:
    """Compute harmful rate per alpha for a given binary key."""
    by_alpha: dict[float, list[dict]] = defaultdict(list)
    for r in records:
        by_alpha[r["alpha"]].append(r)

    rates: dict[float, dict] = {}
    for a in alphas:
        recs = by_alpha.get(a, [])
        n = len(recs)
        if n == 0:
            continue
        n_harmful = sum(1 for r in recs if r[key])
        rates[a] = build_rate_summary(n_harmful, n)
    return rates


def delta_sign_stability(
    records: list[dict],
    alphas: list[float],
) -> dict[str, Any]:
    """Compare intervention deltas (rate_at_alpha - rate_at_baseline) across judges."""
    if not alphas:
        return {}
    baseline_alpha = alphas[0]
    judges = {
        "binary": "binary_harmful",
        "v2": "v2_harmful",
        "v3": "v3_harmful",
    }
    judge_rates: dict[str, dict[float, dict]] = {}
    for name, key in judges.items():
        judge_rates[name] = compute_harmful_rates(records, alphas, key)

    deltas: dict[str, dict[str, float]] = {}
    for name in judges:
        base_rate = judge_rates[name].get(baseline_alpha, {}).get("estimate", 0.0)
        d: dict[str, float] = {}
        for a in alphas[1:]:
            rate = judge_rates[name].get(a, {}).get("estimate", 0.0)
            d[str(a)] = round(rate - base_rate, 4)
        deltas[name] = d

    # Check sign agreement across judges for each alpha
    sign_agreement: dict[str, bool] = {}
    for a in alphas[1:]:
        signs = [np.sign(deltas[name].get(str(a), 0.0)) for name in judges]
        sign_agreement[str(a)] = len(set(signs)) == 1

    return {
        "baseline_alpha": baseline_alpha,
        "deltas_by_judge": deltas,
        "sign_agreement": sign_agreement,
        "rates_by_judge": {
            name: {str(a): rate for a, rate in rates.items()}
            for name, rates in judge_rates.items()
        },
    }


# ---------------------------------------------------------------------------
# Category-level concordance
# ---------------------------------------------------------------------------


def category_concordance(
    records: list[dict],
    key_a: str,
    key_b: str,
) -> dict[str, dict]:
    """Compute agreement stratified by harm category."""
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_cat[r.get("category", "unknown")].append(r)

    results = {}
    for cat, recs in sorted(by_cat.items()):
        y_a = np.array([bool(r[key_a]) for r in recs])
        y_b = np.array([bool(r[key_b]) for r in recs])
        n = len(y_a)
        agree = int(np.sum(y_a == y_b))
        results[cat] = {
            "n": n,
            "agreement_rate": round(agree / n, 4) if n > 0 else 0.0,
            "kappa": round(cohens_kappa(y_a, y_b), 4),
        }
    return results


# ---------------------------------------------------------------------------
# Gold label comparison
# ---------------------------------------------------------------------------


def gold_comparison(
    records: list[dict],
    gold_path: Path,
) -> dict[str, Any]:
    """Compare each judge against human gold labels."""
    gold_recs = load_jsonl(gold_path)
    gold_by_key: dict[tuple[str, float], str] = {}
    for g in gold_recs:
        label = g.get("human_label", "").upper()
        if label in ("HARMFUL", "SAFE"):
            gold_by_key[(g["id"], g["alpha"])] = label

    # Match joined records to gold
    matched: list[dict] = []
    for r in records:
        key = (r["id"], r["alpha"])
        if key in gold_by_key:
            r_copy = dict(r)
            r_copy["gold_harmful"] = gold_by_key[key] == "HARMFUL"
            matched.append(r_copy)

    if len(matched) < 5:
        return {"n_matched": len(matched), "note": "insufficient gold matches"}

    gold_arr = np.array([r["gold_harmful"] for r in matched])
    results: dict[str, Any] = {"n_matched": len(matched)}

    for name, key in [
        ("binary", "binary_harmful"),
        ("v2", "v2_harmful"),
        ("v3", "v3_harmful"),
    ]:
        pred_arr = np.array([r[key] for r in matched])
        n = len(pred_arr)
        correct = int(np.sum(pred_arr == gold_arr))
        tp = int(np.sum(pred_arr & gold_arr))
        fp = int(np.sum(pred_arr & ~gold_arr))
        fn = int(np.sum(~pred_arr & gold_arr))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        results[name] = {
            "accuracy": build_rate_summary(correct, n),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "confusion": confusion_matrix_2x2(gold_arr, pred_arr),
        }

    return results


# ---------------------------------------------------------------------------
# V3-exclusive diagnostics
# ---------------------------------------------------------------------------


def v3_diagnostics(records: list[dict], alphas: list[float]) -> dict[str, Any]:
    """Distribution of v3-exclusive fields across alphas."""
    by_alpha: dict[float, list[dict]] = defaultdict(list)
    for r in records:
        by_alpha[r["alpha"]].append(r)

    outcome_dists: dict[str, dict[str, int]] = {}
    intent_match_dists: dict[str, dict[str, int]] = {}
    v2v3_flips: dict[str, dict[str, int]] = {}

    for a in alphas:
        recs = by_alpha.get(a, [])
        if not recs:
            continue
        a_key = str(a)

        # primary_outcome distribution
        outcomes: dict[str, int] = defaultdict(int)
        for r in recs:
            po = r.get("v3_primary_outcome", "unknown")
            outcomes[po] += 1
        outcome_dists[a_key] = dict(outcomes)

        # intent_match distribution
        intents: dict[str, int] = defaultdict(int)
        for r in recs:
            im = r.get("v3_intent_match")
            intents[str(im)] += 1
        intent_match_dists[a_key] = dict(intents)

        # Flip analysis: where v2 and v3 harmful_binary disagree
        flips: dict[str, int] = defaultdict(int)
        for r in recs:
            v2h = r["v2_harmful"]
            v3h = r["v3_harmful"]
            v2b = r["v2_borderline"]
            if v2h and not v3h:
                flips["v2_yes_v3_no"] += 1
            elif not v2h and v3h:
                if v2b:
                    flips["v2_borderline_v3_yes"] += 1
                else:
                    flips["v2_no_v3_yes"] += 1
            elif v2b and not v3h:
                flips["v2_borderline_v3_no"] += 1
        v2v3_flips[a_key] = dict(flips)

    return {
        "primary_outcome_by_alpha": outcome_dists,
        "intent_match_by_alpha": intent_match_dists,
        "v2_v3_harmful_flips_by_alpha": v2v3_flips,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_agreement_by_alpha(
    stratified: list[dict],
    alphas: list[float],
    output_path: Path,
) -> None:
    """Plot agreement rate curves across alpha for each judge pair."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for entry in stratified:
        pair_label = entry["pair"]
        x_vals = []
        y_vals = []
        y_lo = []
        y_hi = []
        for a in alphas:
            pa = entry["per_alpha"].get(str(a))
            if pa is None:
                continue
            x_vals.append(a)
            rate = pa["agreement_rate"]["estimate"]
            ci = pa["agreement_rate"]["ci"]
            y_vals.append(rate * 100)
            y_lo.append(ci["lower"] * 100)
            y_hi.append(ci["upper"] * 100)

        x_arr = np.array(x_vals)
        y_arr = np.array(y_vals)
        lo_arr = np.array(y_lo)
        hi_arr = np.array(y_hi)
        ax.plot(x_arr, y_arr, "o-", label=pair_label, markersize=6)
        ax.fill_between(x_arr, lo_arr, hi_arr, alpha=0.15)

    ax.set_xlabel("Alpha (scaling factor)")
    ax.set_ylabel("Agreement rate (%)")
    ax.set_title("Inter-judge agreement by intervention strength")
    ax.legend(fontsize=9)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_confusion_matrices(
    pairwise_results: list[dict],
    output_path: Path,
) -> None:
    """Plot 2x2 confusion matrices for each judge pair."""
    n_pairs = len(pairwise_results)
    fig, axes = plt.subplots(1, n_pairs, figsize=(5 * n_pairs, 4))
    if n_pairs == 1:
        axes = [axes]

    for ax, pw in zip(axes, pairwise_results):
        cm = pw["confusion"]
        matrix = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
        im = ax.imshow(matrix, cmap="Blues", aspect="auto")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=14)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Safe", "Harmful"])
        ax.set_yticklabels(["Safe", "Harmful"])
        pair = pw["pair"]
        parts = pair.split("_vs_")
        ax.set_xlabel(parts[1] if len(parts) == 2 else "Judge B")
        ax.set_ylabel(parts[0] if len(parts) == 2 else "Judge A")
        kappa = pw["kappa"]
        ax.set_title(f"{pair}\n(kappa={kappa:.3f})")
        fig.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle("Inter-judge confusion matrices", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_structural_scatter(
    records: list[dict],
    output_path: Path,
) -> None:
    """Scatter plots of v2 vs v3 ordinal axes (C, S, V)."""
    axes = ["C", "S", "V"]
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.5))
    for ax, axis_name in zip(axs, axes):
        v2_key = f"v2_{axis_name}"
        v3_key = f"v3_{axis_name}"
        pairs = [
            (r[v2_key], r[v3_key])
            for r in records
            if r[v2_key] is not None and r[v3_key] is not None
        ]
        if not pairs:
            ax.set_title(f"{axis_name}: no data")
            continue
        v2_vals = np.array([p[0] for p in pairs], dtype=float)
        v3_vals = np.array([p[1] for p in pairs], dtype=float)

        # Jitter for visibility
        rng = np.random.RandomState(42)
        jitter = 0.1
        v2_j = v2_vals + rng.uniform(-jitter, jitter, len(v2_vals))
        v3_j = v3_vals + rng.uniform(-jitter, jitter, len(v3_vals))

        ax.scatter(v2_j, v3_j, alpha=0.2, s=12, edgecolors="none")

        # Diagonal
        lims = [
            min(v2_vals.min(), v3_vals.min()) - 0.5,
            max(v2_vals.max(), v3_vals.max()) + 0.5,
        ]
        ax.plot(lims, lims, "k--", alpha=0.3, linewidth=1)
        ax.set_xlabel(f"v2 {axis_name}")
        ax.set_ylabel(f"v3 {axis_name}")

        r, _ = sp_stats.spearmanr(v2_vals, v3_vals)
        exact = float(np.mean(v2_vals == v3_vals))
        ax.set_title(f"{axis_name} (r_s={r:.3f}, exact={exact:.1%})")
        ax.grid(True, alpha=0.3)

    fig.suptitle("Structural axis correlation: v2 vs v3", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Three-judge concordance analysis for jailbreak intervention data"
    )
    p.add_argument("--h_neuron_binary_dir", type=str, default=str(DEFAULT_H_BINARY))
    p.add_argument("--h_neuron_csv2_v2_dir", type=str, default=str(DEFAULT_H_V2))
    p.add_argument("--h_neuron_csv2_v3_dir", type=str, default=str(DEFAULT_H_V3))
    p.add_argument("--control_base", type=str, default=str(DEFAULT_CONTROL_BASE))
    p.add_argument("--control_seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    p.add_argument("--alphas", type=float, nargs="+", default=DEFAULT_ALPHAS)
    p.add_argument("--gold_labels", type=str, default=None)
    p.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    h_binary_dir = Path(args.h_neuron_binary_dir)
    h_v2_dir = Path(args.h_neuron_csv2_v2_dir)
    h_v3_dir = Path(args.h_neuron_csv2_v3_dir)
    control_base = Path(args.control_base)
    alphas = args.alphas
    seeds = args.control_seeds
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Three-Judge Concordance Analysis")
    print("=" * 60)
    print(f"  H-neuron binary: {h_binary_dir}")
    print(f"  H-neuron v2:     {h_v2_dir}")
    print(f"  H-neuron v3:     {h_v3_dir}")
    print(f"  Control base:    {control_base}")
    print(f"  Seeds:           {seeds}")
    print(f"  Alphas:          {alphas}")
    print(f"  Output:          {output_dir}")

    # ── Join H-neuron data ────────────────────────────────────────
    print("\nJoining H-neuron data...")
    h_records = join_condition(h_binary_dir, h_v2_dir, h_v3_dir, alphas, "h_neuron")
    print(f"  Total H-neuron joined: {len(h_records)}")

    # ── Join control data ─────────────────────────────────────────
    print("\nJoining control data...")
    ctrl_records: list[dict] = []
    for seed in seeds:
        seed_name = f"seed_{seed}_unconstrained"
        seed_gen_dir = control_base / seed_name
        v2_dir = control_base / f"{seed_name}_csv2_v2"
        v3_dir = control_base / f"{seed_name}_csv2_v3"
        try:
            seed_recs = join_condition(seed_gen_dir, v2_dir, v3_dir, alphas, seed_name)
            ctrl_records.extend(seed_recs)
        except FileNotFoundError as e:
            print(f"  WARN: {seed_name} skipped: {e}")
    print(f"  Total control joined: {len(ctrl_records)}")

    all_records = h_records + ctrl_records
    print(f"\nTotal three-way joined records: {len(all_records)}")

    if len(all_records) < 10:
        print("ERROR: insufficient joined records for analysis")
        return

    # ── Pairwise agreement (all data) ─────────────────────────────
    print("\n--- Pairwise Agreement (all conditions) ---")
    judge_pairs = [
        ("binary_harmful", "v2_harmful", "binary", "v2"),
        ("binary_harmful", "v3_harmful", "binary", "v3"),
        ("v2_harmful", "v3_harmful", "v2", "v3"),
    ]
    pairwise_results = []
    for key_a, key_b, label_a, label_b in judge_pairs:
        pw = pairwise_agreement(all_records, key_a, key_b, label_a, label_b)
        pairwise_results.append(pw)
        rate = pw["agreement_rate"]["estimate"]
        ci = pw["agreement_rate"]["ci"]
        print(
            f"  {pw['pair']}: {rate:.1%} "
            f"[{ci['lower']:.1%}, {ci['upper']:.1%}] "
            f"(kappa={pw['kappa']:.3f}, n={pw['n']})"
        )

    # ── Pairwise by condition ─────────────────────────────────────
    print("\n--- Pairwise Agreement by condition ---")
    condition_pairwise: dict[str, list[dict]] = {}
    for condition, recs in [("h_neuron", h_records), ("control", ctrl_records)]:
        condition_pairwise[condition] = []
        for key_a, key_b, label_a, label_b in judge_pairs:
            pw = pairwise_agreement(recs, key_a, key_b, label_a, label_b)
            condition_pairwise[condition].append(pw)
            rate = pw["agreement_rate"]["estimate"]
            print(f"  [{condition}] {pw['pair']}: {rate:.1%} (kappa={pw['kappa']:.3f})")

    # ── Alpha-stratified agreement ────────────────────────────────
    print("\n--- Alpha-stratified Agreement ---")
    stratified_results = []
    for key_a, key_b, label_a, label_b in judge_pairs:
        strat = alpha_stratified_agreement(
            all_records, alphas, key_a, key_b, label_a, label_b
        )
        stratified_results.append(strat)
        for a_str, pa in strat["per_alpha"].items():
            rate = pa["agreement_rate"]["estimate"]
            print(
                f"  {strat['pair']} alpha={a_str}: {rate:.1%} (kappa={pa['kappa']:.3f})"
            )

    # ── Structural correlation ────────────────────────────────────
    print("\n--- Structural Axis Correlation (v2 vs v3) ---")
    struct = structural_correlation(all_records)
    for axis, vals in struct.items():
        if "spearman_r" in vals:
            print(
                f"  {axis}: spearman={vals['spearman_r']:.3f}, "
                f"exact_match={vals['exact_match_rate']:.1%}, "
                f"MAD={vals['mean_abs_diff']:.3f} (n={vals['n']})"
            )

    # ── Delta sign stability ──────────────────────────────────────
    print("\n--- Intervention Delta Sign Stability ---")
    h_deltas = delta_sign_stability(h_records, alphas)
    ctrl_deltas = delta_sign_stability(ctrl_records, alphas) if ctrl_records else {}

    if h_deltas:
        print("  H-neuron deltas (from baseline):")
        for judge, deltas in h_deltas["deltas_by_judge"].items():
            delta_str = ", ".join(f"α{a}={d:+.3f}" for a, d in deltas.items())
            print(f"    {judge}: {delta_str}")
        for a_str, agrees in h_deltas["sign_agreement"].items():
            status = "AGREE" if agrees else "DISAGREE"
            print(f"    α={a_str} sign: {status}")

    if ctrl_deltas:
        print("  Control deltas (from baseline):")
        for judge, deltas in ctrl_deltas["deltas_by_judge"].items():
            delta_str = ", ".join(f"α{a}={d:+.3f}" for a, d in deltas.items())
            print(f"    {judge}: {delta_str}")

    # ── Category concordance ──────────────────────────────────────
    print("\n--- Category-level concordance (v2 vs v3) ---")
    cat_conc = category_concordance(all_records, "v2_harmful", "v3_harmful")
    for cat, vals in sorted(cat_conc.items(), key=lambda x: x[1]["n"], reverse=True):
        print(
            f"  {cat}: {vals['agreement_rate']:.1%} "
            f"(kappa={vals['kappa']:.3f}, n={vals['n']})"
        )

    # ── V3 diagnostics ────────────────────────────────────────────
    print("\n--- V3-exclusive diagnostics ---")
    v3_diag = v3_diagnostics(all_records, alphas)
    for a_str, dist in v3_diag["primary_outcome_by_alpha"].items():
        total = sum(dist.values())
        dist_str = ", ".join(f"{k}={v}" for k, v in sorted(dist.items()))
        print(f"  α={a_str} outcomes (n={total}): {dist_str}")

    for a_str, flips in v3_diag["v2_v3_harmful_flips_by_alpha"].items():
        if flips:
            flip_str = ", ".join(f"{k}={v}" for k, v in sorted(flips.items()))
            print(f"  α={a_str} v2↔v3 flips: {flip_str}")

    # ── Gold comparison ───────────────────────────────────────────
    gold_results: dict[str, Any] = {}
    if args.gold_labels and Path(args.gold_labels).exists():
        print("\n--- Gold Label Comparison ---")
        gold_results = gold_comparison(all_records, Path(args.gold_labels))
        n_matched = gold_results.get("n_matched", 0)
        print(f"  Matched {n_matched} records to gold labels")
        for judge_name in ("binary", "v2", "v3"):
            if judge_name in gold_results:
                jr = gold_results[judge_name]
                acc = jr["accuracy"]["estimate"]
                print(
                    f"  {judge_name}: acc={acc:.1%}, "
                    f"prec={jr['precision']:.3f}, "
                    f"rec={jr['recall']:.3f}, "
                    f"F1={jr['f1']:.3f}"
                )

    # ── Save summary ──────────────────────────────────────────────
    summary = {
        "n_h_neuron": len(h_records),
        "n_control": len(ctrl_records),
        "n_total": len(all_records),
        "alphas": alphas,
        "pairwise_agreement": pairwise_results,
        "pairwise_by_condition": condition_pairwise,
        "alpha_stratified": stratified_results,
        "structural_correlation": struct,
        "delta_stability": {
            "h_neuron": h_deltas,
            "control": ctrl_deltas,
        },
        "category_concordance": cat_conc,
        "v3_diagnostics": v3_diag,
        "gold_comparison": gold_results,
    }

    summary_path = output_dir / "concordance_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n  Saved: {summary_path}")

    # ── Plots ─────────────────────────────────────────────────────
    print("\nGenerating plots...")
    plot_agreement_by_alpha(
        stratified_results, alphas, output_dir / "concordance_by_alpha.png"
    )
    plot_confusion_matrices(pairwise_results, output_dir / "concordance_confusion.png")
    plot_structural_scatter(all_records, output_dir / "structural_correlation.png")

    # ── Delta stability JSON ──────────────────────────────────────
    delta_path = output_dir / "delta_stability.json"
    with open(delta_path, "w") as f:
        json.dump(
            {"h_neuron": h_deltas, "control": ctrl_deltas},
            f,
            indent=2,
            default=str,
        )
    print(f"  Saved: {delta_path}")

    print("\n" + "=" * 60)
    print("Concordance analysis complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
