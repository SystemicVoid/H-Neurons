"""
Figure 4: Measurement Choices Changed the Scientific Conclusion.

The figure is aligned to the current paper claim boundary:
- binary and graded results are shown in separate panels
- holdout evaluator accuracy reflects the post-SR-4o tie
- specificity is shown with explicit uncertainty

Usage:
    uv run python paper/draft/figures/fig4_measurement.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / "paper/draft/figures/fig4_measurement.png"

TITLE_COLOR = "#1E3044"
SUBTITLE_COLOR = "#5A6E7F"
BG_COLOR = "#FFFFFF"

C_H = "#3E6A8A"
C_H_FILL = "#DAEAF6"
C_R = "#8899A6"
C_R_FILL = "#E8ECF0"
C_V3 = "#3E6A8A"
C_SR = "#6B8E5A"
C_V2 = "#C49A3C"
C_BIN = "#BF4E38"

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
        "text.color": TITLE_COLOR,
        "axes.labelcolor": TITLE_COLOR,
        "xtick.color": SUBTITLE_COLOR,
        "ytick.color": SUBTITLE_COLOR,
    }
)


def load_json(rel_path: str) -> dict:
    with open(ROOT / rel_path, encoding="utf-8") as f:
        return json.load(f)


def wilson_interval(
    n_success: int, n_total: int, z: float = 1.96
) -> tuple[float, float]:
    phat = n_success / n_total
    denom = 1 + z**2 / n_total
    center = (phat + z**2 / (2 * n_total)) / denom
    margin = z * np.sqrt((phat * (1 - phat) + z**2 / (4 * n_total)) / n_total) / denom
    return (center - margin) * 100, (center + margin) * 100


def load_data() -> dict:
    exp = load_json("data/gemma3_4b/intervention/jailbreak/experiment/results.json")
    ctrl = load_json(
        "data/gemma3_4b/intervention/jailbreak/control/seed_0_unconstrained/results.json"
    )
    csv2 = load_json(
        "data/gemma3_4b/intervention/jailbreak/control/comparison_csv2_v2_summary.json"
    )

    binary_alphas = np.array([0.0, 3.0])
    binary_h_rates = (
        np.array(
            [
                exp["results"]["0.0"]["compliance"]["estimate"],
                exp["results"]["3.0"]["compliance"]["estimate"],
            ]
        )
        * 100
    )
    binary_r_rates = (
        np.array(
            [
                ctrl["results"]["0.0"]["compliance"]["estimate"],
                ctrl["results"]["3.0"]["compliance"]["estimate"],
            ]
        )
        * 100
    )
    binary_h_ci = np.array(
        [
            [
                exp["results"]["0.0"]["compliance"]["ci"]["lower"] * 100,
                exp["results"]["0.0"]["compliance"]["ci"]["upper"] * 100,
            ],
            [
                exp["results"]["3.0"]["compliance"]["ci"]["lower"] * 100,
                exp["results"]["3.0"]["compliance"]["ci"]["upper"] * 100,
            ],
        ]
    )
    binary_r_ci = np.array(
        [
            [
                ctrl["results"]["0.0"]["compliance"]["ci"]["lower"] * 100,
                ctrl["results"]["0.0"]["compliance"]["ci"]["upper"] * 100,
            ],
            [
                ctrl["results"]["3.0"]["compliance"]["ci"]["lower"] * 100,
                ctrl["results"]["3.0"]["compliance"]["ci"]["upper"] * 100,
            ],
        ]
    )

    graded_alphas = np.array(csv2["alphas"])
    h_counts = [
        round(rate * total)
        for rate, total in zip(
            csv2["h_neuron"]["csv2_yes_rates"],
            csv2["h_neuron"]["total_record_counts"],
        )
    ]
    r_counts = [
        round(rate * total)
        for rate, total in zip(
            csv2["per_seed"]["seed_0_unconstrained"]["csv2_yes_rates"],
            csv2["per_seed"]["seed_0_unconstrained"]["total_record_counts"],
        )
    ]
    h_ci = [
        wilson_interval(count, total)
        for count, total in zip(h_counts, csv2["h_neuron"]["total_record_counts"])
    ]
    r_ci = [
        wilson_interval(count, total)
        for count, total in zip(
            r_counts, csv2["per_seed"]["seed_0_unconstrained"]["total_record_counts"]
        )
    ]

    evaluator_counts = {
        "CSV2 v3": 48,
        "StrongREJECT\n(SR-4o)": 48,
        "CSV2 v2": 46,
        "Binary judge": 45,
    }
    evaluator_colors = {
        "CSV2 v3": (C_V3, C_H_FILL),
        "StrongREJECT\n(SR-4o)": (C_SR, "#E3EDDF"),
        "CSV2 v2": (C_V2, "#FDF5E0"),
        "Binary judge": (C_BIN, "#FDF1ED"),
    }

    evaluator_data = []
    for name, correct in evaluator_counts.items():
        lo, hi = wilson_interval(correct, 50)
        color, fill = evaluator_colors[name]
        evaluator_data.append(
            {
                "name": name,
                "accuracy": correct / 50 * 100,
                "ci_lo": lo,
                "ci_hi": hi,
                "color": color,
                "fill": fill,
            }
        )

    return {
        "binary_alphas": binary_alphas,
        "binary_h_rates": binary_h_rates,
        "binary_r_rates": binary_r_rates,
        "binary_h_ci": binary_h_ci,
        "binary_r_ci": binary_r_ci,
        "graded_alphas": graded_alphas,
        "graded_h_rates": np.array(csv2["h_neuron"]["csv2_yes_rates"]) * 100,
        "graded_r_rates": np.array(
            csv2["per_seed"]["seed_0_unconstrained"]["csv2_yes_rates"]
        )
        * 100,
        "graded_h_ci_lo": np.array([lo for lo, _ in h_ci]),
        "graded_h_ci_hi": np.array([hi for _, hi in h_ci]),
        "graded_r_ci_lo": np.array([lo for lo, _ in r_ci]),
        "graded_r_ci_hi": np.array([hi for _, hi in r_ci]),
        "h_slope": csv2["h_neuron"]["slope_csv2_yes_pp_per_alpha"],
        "r_slope": csv2["per_seed"]["seed_0_unconstrained"][
            "slope_csv2_yes_pp_per_alpha"
        ],
        "slope_diff": csv2["comparison"]["gap_h_minus_random_mean_pp"],
        "evaluators": evaluator_data,
    }


def draw_binary_panel(ax: plt.Axes, data: dict) -> None:
    x = data["binary_alphas"]
    h_lo = data["binary_h_rates"] - data["binary_h_ci"][:, 0]
    h_hi = data["binary_h_ci"][:, 1] - data["binary_h_rates"]
    r_lo = data["binary_r_rates"] - data["binary_r_ci"][:, 0]
    r_hi = data["binary_r_ci"][:, 1] - data["binary_r_rates"]

    ax.errorbar(
        x,
        data["binary_h_rates"],
        yerr=[h_lo, h_hi],
        color=C_H,
        marker="o",
        linewidth=2.0,
        capsize=4,
        label="H-neurons",
    )
    ax.errorbar(
        x,
        data["binary_r_rates"],
        yerr=[r_lo, r_hi],
        color=C_R,
        marker="s",
        linewidth=1.8,
        linestyle="--",
        capsize=4,
        label="Random control",
    )

    ax.set_title(
        "A. Binary scoring", fontsize=11, fontweight="bold", loc="left", pad=10
    )
    ax.set_xlabel("Scaling factor (α)", fontsize=10, fontweight="bold")
    ax.set_ylabel("Binary harmfulness rate (%)", fontsize=10, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xlim(-0.2, 3.2)
    ax.set_ylim(20, 38)
    ax.text(
        0.03,
        0.97,
        "Endpoint shift on binary surface\nH-neurons: +3.0 pp\nrandom: -2.0 pp",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.4,
        bbox=dict(boxstyle="round,pad=0.28", facecolor="white", edgecolor="#D4DCE3"),
    )
    ax.text(
        0.98,
        0.03,
        "Binary evaluation returns a weak, non-decisive picture.",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7,
        color=SUBTITLE_COLOR,
        fontstyle="italic",
    )
    ax.legend(fontsize=7.5, loc="upper left", framealpha=0.9, edgecolor="#CCCCCC")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.15)


def draw_graded_panel(ax: plt.Axes, data: dict) -> None:
    x = data["graded_alphas"]
    ax.fill_between(
        x, data["graded_h_ci_lo"], data["graded_h_ci_hi"], color=C_H, alpha=0.18
    )
    ax.plot(
        x,
        data["graded_h_rates"],
        color=C_H,
        marker="o",
        linewidth=2.2,
        label=f"H-neurons: slope {data['h_slope']:+.2f} pp/α",
    )
    ax.fill_between(
        x, data["graded_r_ci_lo"], data["graded_r_ci_hi"], color=C_R, alpha=0.18
    )
    ax.plot(
        x,
        data["graded_r_rates"],
        color=C_R,
        marker="s",
        linestyle="--",
        linewidth=2.0,
        label=f"Random control: slope {data['r_slope']:+.2f} pp/α",
    )
    ax.text(
        0.03,
        0.97,
        f"Slope difference = {data['slope_diff']:+.2f} pp/α\n95% CI [+1.17, +4.42]\nPermutation p = 0.013",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.3,
        bbox=dict(boxstyle="round,pad=0.28", facecolor="white", edgecolor="#D4DCE3"),
    )
    ax.set_title(
        "B. Graded scoring", fontsize=11, fontweight="bold", loc="left", pad=10
    )
    ax.set_xlabel("Scaling factor (α)", fontsize=10, fontweight="bold")
    ax.set_ylabel("Strict harmfulness rate (%)", fontsize=10, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xlim(-0.1, 3.1)
    ax.set_ylim(16, 30)
    ax.legend(fontsize=7.4, loc="upper left", framealpha=0.9, edgecolor="#CCCCCC")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.15)


def draw_evaluator_panel(ax: plt.Axes, data: dict) -> None:
    x = np.arange(len(data["evaluators"]))
    vals = [entry["accuracy"] for entry in data["evaluators"]]
    lo = [entry["accuracy"] - entry["ci_lo"] for entry in data["evaluators"]]
    hi = [entry["ci_hi"] - entry["accuracy"] for entry in data["evaluators"]]
    ax.bar(
        x,
        vals,
        yerr=[lo, hi],
        color=[entry["fill"] for entry in data["evaluators"]],
        edgecolor=[entry["color"] for entry in data["evaluators"]],
        linewidth=1.5,
        capsize=4,
        error_kw={"linewidth": 1.2, "color": SUBTITLE_COLOR},
        zorder=3,
    )
    for xi, val in zip(x, vals):
        ax.text(
            xi,
            val + 0.8,
            f"{val:.1f}",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
        )
    ax.text(
        0.03,
        0.97,
        "Post-upgrade holdout result\nCSV-v3 and SR-4o tie at 96.0%\nReason to keep v3: richer outcome taxonomy",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.3,
        bbox=dict(boxstyle="round,pad=0.28", facecolor="white", edgecolor="#D4DCE3"),
    )
    ax.set_title(
        "C. Holdout evaluator accuracy",
        fontsize=11,
        fontweight="bold",
        loc="left",
        pad=10,
    )
    ax.set_ylabel("Accuracy (%)", fontsize=10, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([entry["name"] for entry in data["evaluators"]], fontsize=8)
    ax.set_ylim(80, 102)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.15, zorder=0)


def draw_specificity_panel(ax: plt.Axes, data: dict) -> None:
    x = data["graded_alphas"]
    h_fit = np.polyfit(x, data["graded_h_rates"], 1)
    r_fit = np.polyfit(x, data["graded_r_rates"], 1)
    fit_x = np.linspace(0.0, 3.0, 100)

    ax.plot(
        x,
        data["graded_h_rates"],
        color=C_H,
        marker="o",
        linewidth=2.2,
        label="H-neurons",
    )
    ax.plot(
        x,
        data["graded_r_rates"],
        color=C_R,
        marker="s",
        linestyle="--",
        linewidth=2.0,
        label="Random control",
    )
    ax.plot(
        fit_x,
        np.polyval(h_fit, fit_x),
        color=C_H,
        linestyle=":",
        linewidth=1.2,
        alpha=0.7,
    )
    ax.plot(
        fit_x,
        np.polyval(r_fit, fit_x),
        color=C_R,
        linestyle=":",
        linewidth=1.2,
        alpha=0.7,
    )
    ax.text(
        0.03,
        0.97,
        "Specificity contrast\nsame graded surface as Panel B\nseed-0 result only",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.3,
        bbox=dict(boxstyle="round,pad=0.28", facecolor="white", edgecolor="#D4DCE3"),
    )
    ax.set_title(
        "D. Seed-0 specificity contrast",
        fontsize=11,
        fontweight="bold",
        loc="left",
        pad=10,
    )
    ax.set_xlabel("Scaling factor (α)", fontsize=10, fontweight="bold")
    ax.set_ylabel("Strict harmfulness rate (%)", fontsize=10, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xlim(-0.1, 3.1)
    ax.set_ylim(16, 30)
    ax.legend(fontsize=7.4, loc="upper left", framealpha=0.9, edgecolor="#CCCCCC")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.15)


def main() -> None:
    data = load_data()
    fig = plt.figure(figsize=(13.4, 9.4), dpi=300)
    fig.set_facecolor(BG_COLOR)
    grid = fig.add_gridspec(2, 2, hspace=0.34, wspace=0.24)

    draw_binary_panel(fig.add_subplot(grid[0, 0]), data)
    draw_graded_panel(fig.add_subplot(grid[0, 1]), data)
    draw_evaluator_panel(fig.add_subplot(grid[1, 0]), data)
    draw_specificity_panel(fig.add_subplot(grid[1, 1]), data)

    fig.suptitle(
        "Figure 4: Measurement Choices Changed the Scientific Conclusion",
        fontsize=13.5,
        fontweight="bold",
        color=TITLE_COLOR,
        y=0.98,
    )
    fig.subplots_adjust(top=0.92, bottom=0.07, left=0.07, right=0.98)
    fig.savefig(
        OUTPUT,
        dpi=300,
        bbox_inches="tight",
        facecolor=BG_COLOR,
        pad_inches=0.18,
    )
    plt.close(fig)
    print(f"Saved: {OUTPUT}")


if __name__ == "__main__":
    main()
