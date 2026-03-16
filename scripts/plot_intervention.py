"""
Plot compliance rate vs α for intervention experiments (Figure 3 equivalent).

Usage:
    uv run python scripts/plot_intervention.py \
        --input_dir data/intervention \
        --output data/intervention/figure3_compliance.png
"""

import os
import json
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


BENCHMARK_LABELS = {
    "faitheval": "Compliance with\nmisleading context (FaithEval)",
    "faitheval_standard": "FaithEval\n(standard prompt)",
    "falseqa": "Compliance with\ninvalid premises (FalseQA)",
    "sycophancy_triviaqa": "Compliance with\nskeptical attitudes (Sycophancy)",
    "jailbreak": "Compliance with\nharmful instructions (Jailbreak)",
}

COLORS = {
    "faitheval": "#1D9E75",      # teal
    "faitheval_standard": "#0E6B4F",  # dark teal
    "falseqa": "#7F77DD",        # purple
    "sycophancy_triviaqa": "#D85A30",  # coral
    "jailbreak": "#E24B4A",      # red
}


def load_benchmark_results(input_dir):
    """Load results.json from each benchmark subdirectory.

    Uses directory name as the key so variants like faitheval/ and
    faitheval_standard/ remain separate entries.
    """
    all_results = {}
    for name in os.listdir(input_dir):
        results_path = os.path.join(input_dir, name, "results.json")
        if os.path.isfile(results_path):
            with open(results_path) as f:
                data = json.load(f)
            all_results[name] = data["results"]
    return all_results


def plot_compliance(all_results, output_path):
    """Create multi-panel figure of compliance rate vs α."""
    benchmarks = [b for b in BENCHMARK_LABELS if b in all_results]
    n = len(benchmarks)
    if n == 0:
        print("No results found!")
        return

    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False)
    axes = axes[0]

    for i, benchmark in enumerate(benchmarks):
        ax = axes[i]
        results = all_results[benchmark]

        alphas = sorted(float(a) for a in results.keys())
        rates = [results[str(a) if str(a) in results else f"{a:.1f}"]["compliance_rate"] * 100
                 for a in alphas]

        color = COLORS.get(benchmark, "#378ADD")
        ax.plot(alphas, rates, "o-", color=color, linewidth=2, markersize=6)

        # Highlight α=1.0 (baseline)
        if 1.0 in alphas:
            idx = alphas.index(1.0)
            ax.axvline(x=1.0, color="gray", linestyle="--", alpha=0.4, linewidth=0.8)
            ax.plot(1.0, rates[idx], "s", color=color, markersize=10, zorder=5)

        ax.set_xlabel("Scaling factor (α)", fontsize=12)
        if i == 0:
            ax.set_ylabel("Compliance rate (%)", fontsize=12)
        ax.set_title(BENCHMARK_LABELS.get(benchmark, benchmark), fontsize=11)
        ax.set_xlim(-0.2, 3.2)
        ax.grid(True, alpha=0.2)
        ax.tick_params(labelsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved figure to {output_path}")
    plt.close()


def parse_args():
    p = argparse.ArgumentParser(description="Plot intervention results")
    p.add_argument("--input_dir", type=str, default="data/intervention")
    p.add_argument("--output", type=str, default="data/intervention/figure3_compliance.png")
    return p.parse_args()


def main():
    args = parse_args()
    all_results = load_benchmark_results(args.input_dir)
    print(f"Found benchmarks: {list(all_results.keys())}")
    plot_compliance(all_results, args.output)


if __name__ == "__main__":
    main()
