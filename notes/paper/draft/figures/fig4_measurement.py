"""
Figure 4: How Measurement Choices Changed the Scientific Conclusion.

Three-panel figure for "Detection Is Not Enough" showing that the choice
of evaluation instrument and neuron selection both determine whether
the H-neuron intervention is detected.

Panel A: Binary vs graded evaluation on same data
  Binary judge: Delta +3.0 pp, CI includes zero -> NOT significant
  Graded (CSV-v2): slope +2.30 pp/alpha [+0.99, +3.58] -> significant
  Random control for both: flat

Panel B: Evaluator accuracy comparison (dev vs holdout)
  4 evaluators, 2 conditions (dev n=74, holdout n=50)

Panel C: Specificity control (H-neurons vs random)
  csv2_yes rate vs alpha with CIs, annotated slope difference and p-value

Data sources:
  - notes/act3-reports/2026-04-12-seed0-jailbreak-control-audit.md
  - notes/act3-reports/2026-04-12-4way-evaluator-comparison.md
  - notes/act3-reports/2026-04-12-4way-evaluator-holdout-validation.md

Usage:
    uv run python notes/paper/draft/figures/fig4_measurement.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[4]  # repo root
OUTPUT = ROOT / "notes/paper/draft/figures/fig4_measurement.png"

# ---------------------------------------------------------------------------
# Color palette -- consistent with Figure 1 & 2
# ---------------------------------------------------------------------------
TITLE_COLOR = "#1E3044"
SUBTITLE_COLOR = "#5A6E7F"
BG_COLOR = "#FFFFFF"

C_HNEURON = "#3E6A8A"  # dark steel blue -- primary
C_RANDOM = "#8899A6"  # medium gray-blue -- random control
C_HNEURON_FILL = "#DAEAF6"
C_RANDOM_FILL = "#E8ECF0"

# Evaluator colors
C_CSV2V3 = "#3E6A8A"  # dark steel blue
C_BINARY = "#BF4E38"  # muted terracotta
C_STRONGREJECT = "#6B8E5A"  # sage green
C_CSV2V2 = "#C49A3C"  # gold/amber

C_NONSIG = "#C0392B"  # red accent for "not significant"
C_SIG = "#27AE60"  # green accent for "significant"

# ---------------------------------------------------------------------------
# Font settings
# ---------------------------------------------------------------------------
FONT_FAMILY = "DejaVu Sans"
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": [FONT_FAMILY, "Helvetica", "Arial"],
        "text.color": TITLE_COLOR,
        "axes.labelcolor": TITLE_COLOR,
        "xtick.color": SUBTITLE_COLOR,
        "ytick.color": SUBTITLE_COLOR,
    }
)

# --- Panel B: Evaluator accuracy (from 4-way reports) ---
EVALUATOR_NAMES = ["CSV2 v3", "Binary", "StrongREJECT", "CSV2 v2"]
DEV_ACCURACY = [86.5, 77.0, 74.3, 73.0]  # n=74
HOLDOUT_ACCURACY = [96.0, 90.0, 94.0, 92.0]  # n=50

HN_GRADED_CI = (0.99, 3.58)
CTRL_GRADED_CI = (-1.42, 0.47)
SLOPE_DIFF_CI = (1.17, 4.42)
PERM_P = 0.013


def load_json(rel_path: str) -> dict:
    """Load a JSON file relative to repo root."""
    with open(ROOT / rel_path) as f:
        return json.load(f)


def wilson_interval(
    n_success: int, n_total: int, z: float = 1.96
) -> tuple[float, float]:
    """Wilson score interval in percentage points."""
    if n_total == 0:
        return 0.0, 0.0

    phat = n_success / n_total
    denom = 1 + z**2 / n_total
    center = (phat + z**2 / (2 * n_total)) / denom
    margin = z * np.sqrt((phat * (1 - phat) + z**2 / (4 * n_total)) / n_total) / denom
    return (center - margin) * 100, (center + margin) * 100


def load_measurement_data() -> dict:
    """Load the canonical JSON-backed data for panels A and C."""
    exp = load_json("data/gemma3_4b/intervention/jailbreak/experiment/results.json")
    ctrl = load_json(
        "data/gemma3_4b/intervention/jailbreak/control/seed_0_unconstrained/results.json"
    )
    csv2 = load_json(
        "data/gemma3_4b/intervention/jailbreak/control/comparison_csv2_summary.json"
    )

    hn_binary_delta = (
        exp["results"]["3.0"]["compliance"]["estimate"]
        - exp["results"]["0.0"]["compliance"]["estimate"]
    ) * 100
    ctrl_binary_delta = (
        ctrl["results"]["3.0"]["compliance"]["estimate"]
        - ctrl["results"]["0.0"]["compliance"]["estimate"]
    ) * 100

    alphas = np.array(csv2["alphas"])
    hn_counts = [
        round(rate * total)
        for rate, total in zip(
            csv2["h_neuron"]["csv2_yes_rates"],
            csv2["h_neuron"]["total_record_counts"],
        )
    ]
    ctrl_counts = [
        round(rate * total)
        for rate, total in zip(
            csv2["per_seed"]["seed_0_unconstrained"]["csv2_yes_rates"],
            csv2["per_seed"]["seed_0_unconstrained"]["total_record_counts"],
        )
    ]

    hn_ci = [
        wilson_interval(n_yes, n_total)
        for n_yes, n_total in zip(hn_counts, csv2["h_neuron"]["total_record_counts"])
    ]
    ctrl_ci = [
        wilson_interval(n_yes, n_total)
        for n_yes, n_total in zip(
            ctrl_counts, csv2["per_seed"]["seed_0_unconstrained"]["total_record_counts"]
        )
    ]

    return {
        "hn_binary_delta": hn_binary_delta,
        "ctrl_binary_delta": ctrl_binary_delta,
        "hn_graded_slope": csv2["h_neuron"]["slope_csv2_yes_pp_per_alpha"],
        "ctrl_graded_slope": csv2["per_seed"]["seed_0_unconstrained"][
            "slope_csv2_yes_pp_per_alpha"
        ],
        "alphas": alphas,
        "hn_rates": np.array(csv2["h_neuron"]["csv2_yes_rates"]) * 100,
        "ctrl_rates": np.array(
            csv2["per_seed"]["seed_0_unconstrained"]["csv2_yes_rates"]
        )
        * 100,
        "hn_ci_lo": np.array([lo for lo, _ in hn_ci]),
        "hn_ci_hi": np.array([hi for _, hi in hn_ci]),
        "ctrl_ci_lo": np.array([lo for lo, _ in ctrl_ci]),
        "ctrl_ci_hi": np.array([hi for _, hi in ctrl_ci]),
        "slope_diff": csv2["comparison"]["gap_h_minus_random_mean_pp"],
    }


# ---------------------------------------------------------------------------
# Panel A: Conclusion reversal -- binary vs graded evaluation
# ---------------------------------------------------------------------------
def draw_panel_a(ax: plt.Axes, data: dict) -> None:
    """Show how switching from binary to graded evaluation reverses the
    statistical conclusion about H-neuron effects."""

    categories = ["H-neurons", "Random\nneurons"]
    x = np.array([0, 1])
    bar_width = 0.32

    # Binary judge (delta pp)
    binary_vals = [data["hn_binary_delta"], data["ctrl_binary_delta"]]

    # Graded (CSV-v2 slope pp/alpha)
    graded_vals = [data["hn_graded_slope"], data["ctrl_graded_slope"]]
    graded_errs_lo = [
        data["hn_graded_slope"] - HN_GRADED_CI[0],
        data["ctrl_graded_slope"] - CTRL_GRADED_CI[0],
    ]
    graded_errs_hi = [
        HN_GRADED_CI[1] - data["hn_graded_slope"],
        CTRL_GRADED_CI[1] - data["ctrl_graded_slope"],
    ]

    # Binary bars
    ax.bar(
        x - bar_width / 2,
        binary_vals,
        bar_width,
        color=C_RANDOM_FILL,
        edgecolor=C_RANDOM,
        linewidth=1.5,
        label="Binary judge (Delta pp)",
        zorder=3,
    )

    # Graded bars
    ax.bar(
        x + bar_width / 2,
        graded_vals,
        bar_width,
        yerr=[graded_errs_lo, graded_errs_hi],
        color=C_HNEURON_FILL,
        edgecolor=C_HNEURON,
        linewidth=1.5,
        capsize=4,
        error_kw={"linewidth": 1.3, "color": SUBTITLE_COLOR},
        label="Graded CSV-v2 (slope pp/$\\alpha$)",
        zorder=3,
    )

    # Zero line
    ax.axhline(0, color=SUBTITLE_COLOR, linewidth=0.8, linestyle="-", zorder=1)

    # Significance annotations
    # Binary H-neuron: CI includes zero -> NOT significant
    ax.annotate(
        "CI includes 0\nn.s.",
        xy=(0 - bar_width / 2, data["hn_binary_delta"]),
        xytext=(0 - bar_width / 2 - 0.15, 11.5),
        fontsize=7.5,
        ha="center",
        va="bottom",
        color=C_NONSIG,
        fontweight="bold",
        arrowprops=dict(arrowstyle="-", color=C_NONSIG, linewidth=0.7),
    )

    # Graded H-neuron: CI excludes zero -> significant
    ax.annotate(
        "CI excludes 0\np < 0.05",
        xy=(0 + bar_width / 2, HN_GRADED_CI[1]),
        xytext=(0 + bar_width / 2 + 0.15, 11.5),
        fontsize=7.5,
        ha="center",
        va="bottom",
        color=C_SIG,
        fontweight="bold",
        arrowprops=dict(arrowstyle="-", color=C_SIG, linewidth=0.7),
    )

    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=9)
    ax.set_ylabel("Effect size (pp or pp/$\\alpha$)", fontsize=9, fontweight="bold")
    ax.set_ylim(-10, 15)
    ax.set_title(
        "A. Conclusion reversal: binary vs graded",
        fontsize=10.5,
        fontweight="bold",
        color=TITLE_COLOR,
        loc="left",
        pad=8,
    )

    ax.legend(
        fontsize=7.5,
        loc="lower right",
        framealpha=0.9,
        edgecolor="#CCCCCC",
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(axis="y", alpha=0.15, zorder=0)


# ---------------------------------------------------------------------------
# Panel B: Evaluator accuracy comparison (dev vs holdout)
# ---------------------------------------------------------------------------
def draw_panel_b(ax: plt.Axes) -> None:
    """Grouped bar chart: 4 evaluators x 2 conditions (dev/holdout)."""

    x = np.arange(len(EVALUATOR_NAMES))
    bar_width = 0.32

    eval_colors = [C_CSV2V3, C_BINARY, C_STRONGREJECT, C_CSV2V2]
    eval_fills = ["#DAEAF6", "#FDF1ED", "#E3EDDF", "#FDF5E0"]

    # Dev bars
    bars_dev = ax.bar(
        x - bar_width / 2,
        DEV_ACCURACY,
        bar_width,
        color=[f + "99" for f in eval_fills],  # slightly transparent
        edgecolor=eval_colors,
        linewidth=1.3,
        label="Dev (n=74)",
        zorder=3,
        hatch="//",
    )

    # Holdout bars
    bars_ho = ax.bar(
        x + bar_width / 2,
        HOLDOUT_ACCURACY,
        bar_width,
        color=eval_fills,
        edgecolor=eval_colors,
        linewidth=1.3,
        label="Holdout (n=50)",
        zorder=3,
    )

    # Value labels
    for bar, val in zip(bars_dev, DEV_ACCURACY):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{val:.1f}",
            ha="center",
            va="bottom",
            fontsize=7,
            color=SUBTITLE_COLOR,
            fontweight="bold",
        )
    for bar, val in zip(bars_ho, HOLDOUT_ACCURACY):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{val:.1f}",
            ha="center",
            va="bottom",
            fontsize=7,
            color=TITLE_COLOR,
            fontweight="bold",
        )

    # Gap compression annotation
    # Dev gap (v3 - SR): 86.5 - 74.3 = 12.2pp
    # Holdout gap (v3 - SR): 96.0 - 94.0 = 2.0pp
    bracket_y = 98.5
    # Bracket between CSV2 v3 (holdout) and StrongREJECT (holdout) positions
    x_v3_ho = 0 + bar_width / 2
    x_sr_ho = 2 + bar_width / 2
    ax.annotate(
        "",
        xy=(x_v3_ho, bracket_y),
        xytext=(x_sr_ho, bracket_y),
        arrowprops=dict(
            arrowstyle="<->",
            color=SUBTITLE_COLOR,
            linewidth=1.0,
        ),
    )
    ax.text(
        (x_v3_ho + x_sr_ho) / 2,
        bracket_y + 0.6,
        "Gap: 12.2 pp (dev) $\\rightarrow$ 2.0 pp (holdout)",
        ha="center",
        va="bottom",
        fontsize=7,
        color=SUBTITLE_COLOR,
        fontstyle="italic",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(EVALUATOR_NAMES, fontsize=8, rotation=0)
    ax.set_ylabel("Accuracy (%)", fontsize=9, fontweight="bold")
    ax.set_ylim(65, 106)
    ax.set_title(
        "B. Evaluator accuracy: dev vs holdout",
        fontsize=10.5,
        fontweight="bold",
        color=TITLE_COLOR,
        loc="left",
        pad=8,
    )

    ax.legend(
        fontsize=7.5,
        loc="lower left",
        framealpha=0.9,
        edgecolor="#CCCCCC",
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(axis="y", alpha=0.15, zorder=0)


# ---------------------------------------------------------------------------
# Panel C: Specificity control -- H-neurons vs random
# ---------------------------------------------------------------------------
def draw_panel_c(ax: plt.Axes, data: dict) -> None:
    """Line plot of csv2_yes rate vs alpha with CIs for H-neurons and
    random control, annotated with slope difference and p-value."""

    # H-neuron curve
    ax.fill_between(
        data["alphas"],
        data["hn_ci_lo"],
        data["hn_ci_hi"],
        alpha=0.18,
        color=C_HNEURON,
        zorder=2,
    )
    ax.plot(
        data["alphas"],
        data["hn_rates"],
        color=C_HNEURON,
        linewidth=2.2,
        marker="o",
        markersize=6,
        label=f"H-neurons (38, seed-0): slope = +{data['hn_graded_slope']:.2f} pp/$\\alpha$",
        zorder=4,
    )

    # Random control curve
    ax.fill_between(
        data["alphas"],
        data["ctrl_ci_lo"],
        data["ctrl_ci_hi"],
        alpha=0.18,
        color=C_RANDOM,
        zorder=2,
    )
    ax.plot(
        data["alphas"],
        data["ctrl_rates"],
        color=C_RANDOM,
        linewidth=2.2,
        marker="s",
        markersize=6,
        label=f"Random neurons (38, seed-0): slope = {data['ctrl_graded_slope']:.2f} pp/$\\alpha$",
        zorder=4,
    )

    # Fit lines (OLS slopes for visual reference)
    hn_fit = np.polyfit(data["alphas"], data["hn_rates"], 1)
    ctrl_fit = np.polyfit(data["alphas"], data["ctrl_rates"], 1)
    alpha_range = np.linspace(-0.1, 3.2, 100)
    ax.plot(
        alpha_range,
        np.polyval(hn_fit, alpha_range),
        color=C_HNEURON,
        linewidth=1.0,
        linestyle="--",
        alpha=0.5,
        zorder=3,
    )
    ax.plot(
        alpha_range,
        np.polyval(ctrl_fit, alpha_range),
        color=C_RANDOM,
        linewidth=1.0,
        linestyle="--",
        alpha=0.5,
        zorder=3,
    )

    # Annotation box: slope difference and p-value
    annotation_text = (
        f"Slope difference: +{data['slope_diff']:.2f} pp/$\\alpha$\n"
        f"95% CI: [{SLOPE_DIFF_CI[0]:.2f}, {SLOPE_DIFF_CI[1]:.2f}]\n"
        f"Permutation $p$ = {PERM_P:.3f}"
    )
    ax.annotate(
        annotation_text,
        xy=(2.2, 24.5),
        fontsize=7.5,
        color=TITLE_COLOR,
        bbox=dict(
            boxstyle="round,pad=0.4",
            facecolor="#F7F9FB",
            edgecolor=C_HNEURON,
            linewidth=1.0,
            alpha=0.9,
        ),
        ha="left",
        va="center",
        zorder=5,
    )

    ax.set_xlabel("Scaling factor ($\\alpha$)", fontsize=9, fontweight="bold")
    ax.set_ylabel("csv2_yes rate (%)", fontsize=9, fontweight="bold")
    ax.set_xlim(-0.2, 3.4)
    ax.set_ylim(12, 34)
    ax.set_xticks(data["alphas"])
    ax.set_title(
        "C. Specificity: H-neurons vs random control",
        fontsize=10.5,
        fontweight="bold",
        color=TITLE_COLOR,
        loc="left",
        pad=8,
    )

    ax.legend(
        fontsize=7.5,
        loc="upper left",
        framealpha=0.9,
        edgecolor="#CCCCCC",
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=8)
    ax.grid(alpha=0.15, zorder=0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    data = load_measurement_data()
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))
    fig.set_facecolor(BG_COLOR)

    draw_panel_a(axes[0], data)
    draw_panel_b(axes[1])
    draw_panel_c(axes[2], data)

    fig.suptitle(
        "Figure 4: How Measurement Choices Changed the Scientific Conclusion",
        fontsize=12,
        fontweight="bold",
        color=TITLE_COLOR,
        y=1.02,
    )

    plt.tight_layout()
    fig.savefig(
        OUTPUT,
        dpi=300,
        bbox_inches="tight",
        facecolor=BG_COLOR,
        pad_inches=0.15,
    )
    print(f"Saved: {OUTPUT}")
    print(f"  Size: {OUTPUT.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
