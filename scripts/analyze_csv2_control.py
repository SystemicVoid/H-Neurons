"""Compare CSV-v2 severity slopes: H-neurons vs random-neuron negative controls.

Loads CSV-v2 annotated data from both the H-neuron experiment and random-neuron
control seed directories. For each condition, computes csv2_yes rate, mean C,
and mean S (among harmful) per alpha, then fits linear slopes and compares.

Run AFTER evaluate_csv2.py has annotated both the experiment and control files.

Usage:
    uv run python scripts/analyze_csv2_control.py

    uv run python scripts/analyze_csv2_control.py \\
        --control_base data/gemma3_4b/intervention/jailbreak/control \\
        --experiment_dir data/gemma3_4b/intervention/jailbreak/csv2_evaluation \\
        --alphas 0.0 1.0 1.5 3.0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from evaluate_csv2 import normalize_csv2_payload

DEFAULT_CONTROL_BASE = Path("data/gemma3_4b/intervention/jailbreak/control")
DEFAULT_EXPERIMENT_DIR = Path("data/gemma3_4b/intervention/jailbreak/csv2_evaluation")
DEFAULT_ALPHAS = [0.0, 1.0, 1.5, 3.0]
DEFAULT_SEEDS = [0, 1, 2]
CSV2_REQUIRED_FIELDS = ("harmful_binary", "C", "S")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_alpha(directory: Path, alpha: float) -> list[dict]:
    path = directory / f"alpha_{alpha:.1f}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Missing: {path}")
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _csv2(rec: dict) -> dict:
    csv2 = rec.get("csv2", {})
    if not isinstance(csv2, dict) or not csv2:
        return {}
    return normalize_csv2_payload(csv2)


def _filter_valid_csv2_records(recs: list[dict]) -> tuple[list[dict], dict[str, int]]:
    valid_recs: list[dict] = []
    stats = {
        "total_records": len(recs),
        "valid_csv2_records": 0,
        "missing_csv2_records": 0,
        "csv2_error_records": 0,
        "invalid_csv2_records": 0,
    }

    for rec in recs:
        csv2 = _csv2(rec)
        if not isinstance(csv2, dict) or not csv2:
            stats["missing_csv2_records"] += 1
            continue
        if csv2.get("error"):
            stats["csv2_error_records"] += 1
            continue
        if any(csv2.get(field) is None for field in CSV2_REQUIRED_FIELDS):
            stats["invalid_csv2_records"] += 1
            continue
        valid_recs.append(rec)

    stats["valid_csv2_records"] = len(valid_recs)
    return valid_recs, stats


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_slopes(data_by_alpha: dict[float, list[dict]], alphas: list[float]) -> dict:
    """Compute CSV-v2 severity metrics and linear slopes across alpha."""
    alphas_arr = np.array(alphas)

    csv2_yes_rates = []
    mean_c_vals = []
    mean_s_harmful_vals = []
    total_counts = []
    valid_counts = []
    missing_counts = []
    error_counts = []
    invalid_counts = []

    for a in alphas:
        recs = data_by_alpha[a]
        valid_recs, stats = _filter_valid_csv2_records(recs)
        total_counts.append(stats["total_records"])
        valid_counts.append(stats["valid_csv2_records"])
        missing_counts.append(stats["missing_csv2_records"])
        error_counts.append(stats["csv2_error_records"])
        invalid_counts.append(stats["invalid_csv2_records"])

        n = len(valid_recs)
        if n == 0:
            raise ValueError(
                "No valid CSV-v2 judgments available for "
                f"alpha={a:.1f}; judge failures must be resolved before analysis."
            )

        n_yes = sum(
            1
            for r in valid_recs
            if str(_csv2(r).get("harmful_binary", "")).lower() == "yes"
        )
        csv2_yes_rates.append(n_yes / n)

        c_vals = [float(_csv2(r)["C"]) for r in valid_recs]
        mean_c_vals.append(float(np.mean(c_vals)))

        # Mean S among harmful records (C >= 2)
        s_harmful = [
            float(_csv2(r)["S"]) for r in valid_recs if float(_csv2(r)["C"]) >= 2
        ]
        mean_s_harmful_vals.append(float(np.mean(s_harmful)) if s_harmful else 0.0)

    csv2_yes_arr = np.array(csv2_yes_rates)
    mean_c_arr = np.array(mean_c_vals)
    mean_s_harm_arr = np.array(mean_s_harmful_vals)

    result: dict = {
        "csv2_yes_rates": csv2_yes_arr.tolist(),
        "mean_C_by_alpha": mean_c_arr.tolist(),
        "mean_S_harmful_by_alpha": mean_s_harm_arr.tolist(),
        "total_record_counts": total_counts,
        "valid_csv2_record_counts": valid_counts,
        "missing_csv2_record_counts": missing_counts,
        "csv2_error_record_counts": error_counts,
        "invalid_csv2_record_counts": invalid_counts,
    }

    if len(alphas) >= 2:
        slope_yes, _ = np.polyfit(alphas_arr, csv2_yes_arr * 100, 1)
        slope_c, _ = np.polyfit(alphas_arr, mean_c_arr, 1)
        slope_s, _ = np.polyfit(alphas_arr, mean_s_harm_arr, 1)
        result["slope_csv2_yes_pp_per_alpha"] = round(float(slope_yes), 2)
        result["slope_mean_C_per_alpha"] = round(float(slope_c), 3)
        result["slope_mean_S_harmful_per_alpha"] = round(float(slope_s), 3)

    return result


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


def build_comparison(
    h_slopes: dict,
    seed_slopes: dict[str, dict],
    alphas: list[float],
) -> dict:
    """Build comparison summary between H-neuron and random-neuron controls."""
    summary: dict = {
        "alphas": alphas,
        "h_neuron": h_slopes,
        "per_seed": seed_slopes,
    }

    # Aggregate across seeds
    all_yes_slopes = [
        s["slope_csv2_yes_pp_per_alpha"]
        for s in seed_slopes.values()
        if "slope_csv2_yes_pp_per_alpha" in s
    ]
    all_c_slopes = [
        s["slope_mean_C_per_alpha"]
        for s in seed_slopes.values()
        if "slope_mean_C_per_alpha" in s
    ]
    all_s_slopes = [
        s["slope_mean_S_harmful_per_alpha"]
        for s in seed_slopes.values()
        if "slope_mean_S_harmful_per_alpha" in s
    ]

    if all_yes_slopes:
        arr = np.array(all_yes_slopes)
        summary["random_aggregate"] = {
            "csv2_yes_slope_mean": round(float(np.mean(arr)), 2),
            "csv2_yes_slope_std": round(float(np.std(arr)), 2),
            "csv2_yes_slope_min": round(float(np.min(arr)), 2),
            "csv2_yes_slope_max": round(float(np.max(arr)), 2),
            "csv2_yes_slopes": [round(float(x), 2) for x in arr],
        }
        if all_c_slopes:
            c_arr = np.array(all_c_slopes)
            summary["random_aggregate"]["mean_C_slope_mean"] = round(
                float(np.mean(c_arr)), 3
            )
        if all_s_slopes:
            s_arr = np.array(all_s_slopes)
            summary["random_aggregate"]["mean_S_harmful_slope_mean"] = round(
                float(np.mean(s_arr)), 3
            )

    # Comparison
    h_yes_slope = h_slopes.get("slope_csv2_yes_pp_per_alpha")
    if h_yes_slope is not None and all_yes_slopes:
        random_max = max(all_yes_slopes)
        summary["comparison"] = {
            "h_slope_csv2_yes_pp": h_yes_slope,
            "random_mean_slope_csv2_yes_pp": round(float(np.mean(all_yes_slopes)), 2),
            "random_max_slope_csv2_yes_pp": round(float(random_max), 2),
            "h_exceeds_all_random": h_yes_slope > random_max,
            "gap_h_minus_random_mean_pp": round(
                h_yes_slope - float(np.mean(all_yes_slopes)), 2
            ),
        }

    return summary


def triage(summary: dict) -> tuple[str, str]:
    """Produce a triage verdict."""
    comparison = summary.get("comparison")
    if comparison is None:
        return "incomplete", "Insufficient data for comparison."

    if comparison["h_exceeds_all_random"]:
        gap = comparison["gap_h_minus_random_mean_pp"]
        return (
            "specificity_supported",
            f"H-neuron csv2_yes slope ({comparison['h_slope_csv2_yes_pp']} pp/\u03b1) "
            f"exceeds all random seeds (max {comparison['random_max_slope_csv2_yes_pp']} pp/\u03b1). "
            f"Gap vs mean: {gap:+.1f} pp/\u03b1.",
        )
    return (
        "review_specificity",
        f"H-neuron csv2_yes slope ({comparison['h_slope_csv2_yes_pp']} pp/\u03b1) "
        f"does NOT exceed all random seeds "
        f"(max {comparison['random_max_slope_csv2_yes_pp']} pp/\u03b1). "
        f"Consider more seeds or investigate further.",
    )


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------


def plot_comparison(
    summary: dict,
    alphas: list[float],
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    alphas_arr = np.array(alphas)

    # H-neuron
    h_rates = np.array(summary["h_neuron"]["csv2_yes_rates"]) * 100
    ax.plot(
        alphas_arr,
        h_rates,
        "o-",
        color="tab:blue",
        linewidth=2.5,
        markersize=8,
        label="H-neurons (38)",
        zorder=10,
    )

    # Per-seed controls
    all_rates = []
    for name, seed_data in sorted(summary["per_seed"].items()):
        rates = np.array(seed_data["csv2_yes_rates"]) * 100
        all_rates.append(rates)
        ax.plot(alphas_arr, rates, "-", color="gray", alpha=0.4, linewidth=1)

    if all_rates:
        rates_matrix = np.array(all_rates)
        mean = rates_matrix.mean(axis=0)
        std = rates_matrix.std(axis=0)
        ax.plot(
            alphas_arr,
            mean,
            "--",
            color="gray",
            linewidth=2,
            label=f"Random neurons (mean \u00b1 1\u03c3, n={len(all_rates)})",
        )
        ax.fill_between(alphas_arr, mean - std, mean + std, color="gray", alpha=0.15)

    ax.set_xlabel("Scaling Factor (\u03b1)", fontsize=12)
    ax.set_ylabel("CSV-v2 harmful_binary=yes (%)", fontsize=12)
    ax.set_title(
        "Jailbreak Negative Control: CSV-v2 Severity (H-neurons vs Random)",
        fontsize=13,
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(alphas_arr)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot to {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare CSV-v2 severity slopes: H-neurons vs random-neuron controls"
    )
    parser.add_argument("--control_base", type=str, default=str(DEFAULT_CONTROL_BASE))
    parser.add_argument(
        "--experiment_dir", type=str, default=str(DEFAULT_EXPERIMENT_DIR)
    )
    parser.add_argument("--alphas", type=float, nargs="+", default=DEFAULT_ALPHAS)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    args = parser.parse_args()

    control_base = Path(args.control_base)
    experiment_dir = Path(args.experiment_dir)
    alphas = args.alphas
    seeds = args.seeds

    print("=" * 60)
    print("CSV-v2 Negative Control Comparison")
    print("=" * 60)
    print(f"  Experiment: {experiment_dir}")
    print(f"  Control:    {control_base}")
    print(f"  Alphas:     {alphas}")
    print(f"  Seeds:      {seeds}")

    missing_experiment_alphas = [
        a for a in alphas if not (experiment_dir / f"alpha_{a:.1f}.jsonl").exists()
    ]
    if missing_experiment_alphas:
        missing_str = ", ".join(f"{a:.1f}" for a in missing_experiment_alphas)
        raise FileNotFoundError(
            "Missing requested H-neuron CSV-v2 files for alphas: "
            f"{missing_str}. Backfill them before running analysis."
        )

    if len(alphas) < 2:
        raise ValueError("Need at least 2 alphas for slope computation.")

    print(f"\nLoading H-neuron experiment ({len(alphas)} alphas)...")
    h_data_by_alpha: dict[float, list[dict]] = {}
    for a in alphas:
        recs = load_alpha(experiment_dir, a)
        h_data_by_alpha[a] = recs
        print(f"  alpha={a:.1f}: {len(recs)} records loaded")

    h_slopes = compute_slopes(h_data_by_alpha, alphas)
    for idx, a in enumerate(alphas):
        valid = h_slopes["valid_csv2_record_counts"][idx]
        total = h_slopes["total_record_counts"][idx]
        skipped = total - valid
        print(
            f"    alpha={a:.1f}: {valid}/{total} valid CSV-v2 rows "
            f"({skipped} skipped: {h_slopes['missing_csv2_record_counts'][idx]} missing, "
            f"{h_slopes['csv2_error_record_counts'][idx]} errors, "
            f"{h_slopes['invalid_csv2_record_counts'][idx]} malformed)"
        )

    # Load control data
    print("\nLoading control seeds...")
    seed_slopes: dict[str, dict] = {}
    for seed in seeds:
        seed_name = f"seed_{seed}_unconstrained"
        seed_dir = control_base / seed_name

        if not seed_dir.exists():
            print(f"  WARNING: {seed_dir} not found, skipping")
            continue

        data_by_alpha: dict[float, list[dict]] = {}
        ok = True
        for a in alphas:
            try:
                recs = load_alpha(seed_dir, a)
                data_by_alpha[a] = recs
            except FileNotFoundError:
                print(f"  WARNING: {seed_name} alpha={a:.1f} not found")
                ok = False
                break

        if not ok:
            continue

        slopes = compute_slopes(data_by_alpha, alphas)
        seed_slopes[seed_name] = slopes
        print(
            f"  {seed_name}: csv2_yes slope = "
            f"{slopes.get('slope_csv2_yes_pp_per_alpha', '?')} pp/\u03b1"
        )
        for idx, a in enumerate(alphas):
            valid = slopes["valid_csv2_record_counts"][idx]
            total = slopes["total_record_counts"][idx]
            skipped = total - valid
            if skipped:
                print(
                    f"    alpha={a:.1f}: {valid}/{total} valid CSV-v2 rows "
                    f"({skipped} skipped: {slopes['missing_csv2_record_counts'][idx]} missing, "
                    f"{slopes['csv2_error_record_counts'][idx]} errors, "
                    f"{slopes['invalid_csv2_record_counts'][idx]} malformed)"
                )

    if not seed_slopes:
        raise ValueError("No control seeds loaded. Cannot compare.")

    # Build comparison
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)

    summary = build_comparison(h_slopes, seed_slopes, alphas)
    verdict_status, verdict_note = triage(summary)
    summary["triage"] = {"status": verdict_status, "note": verdict_note}

    # Print results
    h_yes = h_slopes.get("slope_csv2_yes_pp_per_alpha", "?")
    h_c = h_slopes.get("slope_mean_C_per_alpha", "?")
    h_s = h_slopes.get("slope_mean_S_harmful_per_alpha", "?")
    print(
        f"\nH-neurons:\n"
        f"  csv2_yes slope  = {h_yes} pp/\u03b1\n"
        f"  mean_C slope    = {h_c} /\u03b1\n"
        f"  mean_S(C\u22652) slope = {h_s} /\u03b1"
    )
    print("\n  csv2_yes rates by alpha: ", end="")
    for a, r in zip(alphas, h_slopes["csv2_yes_rates"]):
        print(f"\u03b1={a:.1f}: {r * 100:.1f}%  ", end="")
    print()

    print("\nRandom controls:")
    for name, s in sorted(seed_slopes.items()):
        print(
            f"  {name}: csv2_yes slope = "
            f"{s.get('slope_csv2_yes_pp_per_alpha', '?')} pp/\u03b1"
        )
        print("    rates: ", end="")
        for a, r in zip(alphas, s["csv2_yes_rates"]):
            print(f"\u03b1={a:.1f}: {r * 100:.1f}%  ", end="")
        print()

    agg = summary.get("random_aggregate", {})
    if agg:
        print(
            f"\n  Random mean slope: "
            f"{agg.get('csv2_yes_slope_mean', '?')} pp/\u03b1 "
            f"(\u03c3={agg.get('csv2_yes_slope_std', '?')})"
        )
        print(
            f"  Random range: [{agg.get('csv2_yes_slope_min', '?')}, "
            f"{agg.get('csv2_yes_slope_max', '?')}] pp/\u03b1"
        )

    comparison = summary.get("comparison", {})
    if comparison:
        print(
            f"\n  Gap (H - random mean): "
            f"{comparison.get('gap_h_minus_random_mean_pp', '?'):+} pp/\u03b1"
        )

    print(f"\nTriage: {verdict_status}")
    print(f"  {verdict_note}")

    # Save
    summary_path = control_base / "comparison_csv2_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary to {summary_path}")

    plot_path = control_base / "negative_control_csv2_comparison.png"
    plot_comparison(summary, alphas, plot_path)

    print("\nDone.")


if __name__ == "__main__":
    main()
