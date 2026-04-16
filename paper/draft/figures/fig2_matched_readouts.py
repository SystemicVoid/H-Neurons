"""
Figure 2: Matched Readouts, Divergent Control.

This main-text figure carries only the FaithEval anchor comparison. Supporting
jailbreak selector detail is preserved in the appendix rather than sharing equal
visual weight here.

Usage:
    uv run python paper/draft/figures/fig2_matched_readouts.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / "paper/draft/figures/fig2_matched_readouts.png"

TITLE_COLOR = "#1E3044"
SUBTITLE_COLOR = "#5A6E7F"
BG_COLOR = "#FFFFFF"

C_HNEURON = "#3E6A8A"
C_SAE = "#BF4E38"
C_RANDOM = "#8899A6"
C_SUPPORT = "#7A8794"

C_HNEURON_FILL = "#DAEAF6"
C_SAE_FILL = "#FDF1ED"
C_RANDOM_FILL = "#E8ECF0"

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


def load_data() -> dict:
    neuron_cls = load_json("data/gemma3_4b/pipeline/classifier_disjoint_summary.json")
    sae_cls = load_json("data/gemma3_4b/pipeline/classifier_sae_summary.json")
    fe_neuron_ctrl = load_json(
        "data/gemma3_4b/intervention/faitheval/control/comparison_summary.json"
    )
    fe_sae = load_json(
        "data/gemma3_4b/intervention/faitheval_sae/experiment/results.json"
    )
    fe_sae_ctrl = load_json(
        "data/gemma3_4b/intervention/faitheval_sae/control/comparison_summary.json"
    )
    fe_sae_slope_diff = load_json(
        "data/gemma3_4b/intervention/faitheval_sae/control/slope_difference_summary.json"
    )
    alphas = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

    sae_results = fe_sae["results"]
    sae_ci = [sae_results[str(alpha)]["compliance"]["ci"] for alpha in alphas]

    return {
        "auroc_h": neuron_cls["evaluation"]["metrics"]["auroc"]["estimate"],
        "auroc_sae": sae_cls["best"]["test_metrics"]["auroc"],
        "alphas": np.array(alphas),
        "h_rates": np.array(fe_neuron_ctrl["h_neuron_baseline"]["compliance_rates"]),
        "h_ci_lo": np.array(
            [
                entry["ci"]["lower"]
                for entry in fe_neuron_ctrl["h_neuron_baseline"][
                    "compliance_ci_by_alpha"
                ]
            ]
        ),
        "h_ci_hi": np.array(
            [
                entry["ci"]["upper"]
                for entry in fe_neuron_ctrl["h_neuron_baseline"][
                    "compliance_ci_by_alpha"
                ]
            ]
        ),
        "sae_rates": np.array(fe_sae["effects"]["compliance_curve"]["rates"]),
        "sae_ci_lo": np.array([entry["lower"] for entry in sae_ci]),
        "sae_ci_hi": np.array([entry["upper"] for entry in sae_ci]),
        "rand_rates": np.array(
            fe_sae_ctrl["random_sae_features"]["mean_compliance_rates"]
        ),
        "rand_std": np.array(
            fe_sae_ctrl["random_sae_features"]["std_compliance_rates"]
        ),
        "slope_diff": fe_sae_slope_diff["slope_difference_pp_per_alpha"]["estimate"],
        "slope_diff_ci": fe_sae_slope_diff["slope_difference_pp_per_alpha"]["ci"],
    }


def draw_panel_a(ax: plt.Axes, data: dict) -> None:
    x = np.array([0.0, 1.0])
    bars = ax.bar(
        x,
        [data["auroc_h"], data["auroc_sae"]],
        color=[C_HNEURON_FILL, C_SAE_FILL],
        edgecolor=[C_HNEURON, C_SAE],
        linewidth=1.8,
        width=0.52,
        zorder=3,
    )
    for bar, val in zip(bars, [data["auroc_h"], data["auroc_sae"]]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.007,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            color=TITLE_COLOR,
        )

    ax.plot([0, 1], [0.818, 0.818], color=SUBTITLE_COLOR, linewidth=0.8)
    ax.plot([0, 0], [0.818, 0.823], color=SUBTITLE_COLOR, linewidth=0.8)
    ax.plot([1, 1], [0.818, 0.823], color=SUBTITLE_COLOR, linewidth=0.8)
    ax.text(
        0.5,
        0.81,
        r"$\Delta$ = 0.005",
        ha="center",
        va="top",
        fontsize=8,
        color=SUBTITLE_COLOR,
        fontstyle="italic",
    )

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["H-neurons\n(38)", "SAE features\n(266)"], fontsize=8)
    ax.set_ylabel("Detection AUROC", fontsize=10, fontweight="bold")
    ax.set_ylim(0.70, 0.88)
    ax.set_title(
        "A. FaithEval matched detection quality",
        fontsize=11,
        fontweight="bold",
        loc="left",
        pad=10,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.2, zorder=0)


def draw_panel_b(ax: plt.Axes, data: dict) -> None:
    alphas = data["alphas"]
    ax.fill_between(
        alphas, data["h_ci_lo"], data["h_ci_hi"], color=C_HNEURON, alpha=0.15
    )
    ax.plot(
        alphas,
        data["h_rates"],
        color=C_HNEURON,
        linewidth=2.2,
        marker="o",
        markersize=5,
        label="H-neurons",
        zorder=4,
    )

    ax.fill_between(
        alphas, data["sae_ci_lo"], data["sae_ci_hi"], color=C_SAE, alpha=0.10
    )
    ax.plot(
        alphas,
        data["sae_rates"],
        color=C_SAE,
        linewidth=2.0,
        marker="s",
        markersize=5,
        label="SAE features",
        zorder=4,
    )

    ax.fill_between(
        alphas,
        data["rand_rates"] - data["rand_std"],
        data["rand_rates"] + data["rand_std"],
        color=C_RANDOM,
        alpha=0.16,
    )
    ax.plot(
        alphas,
        data["rand_rates"],
        color=C_RANDOM,
        linewidth=1.5,
        linestyle="--",
        marker="^",
        markersize=4,
        label="Random SAE features",
        zorder=3,
    )

    slope_ci = data["slope_diff_ci"]
    ax.text(
        0.03,
        0.97,
        "FaithEval anchor\n"
        + r"$\Delta$"
        + f" slope = {data['slope_diff']:+.2f} pp/$\\alpha$\n"
        + f"95% CI [{slope_ci['lower']:+.2f}, {slope_ci['upper']:+.2f}]",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.3,
        bbox=dict(
            boxstyle="round,pad=0.30",
            facecolor="white",
            edgecolor="#D4DCE3",
            alpha=0.95,
        ),
    )

    ax.set_xlabel("Scaling factor (α)", fontsize=10, fontweight="bold")
    ax.set_ylabel("Compliance rate", fontsize=10, fontweight="bold")
    ax.set_title(
        "B. FaithEval steering divergence",
        fontsize=11,
        fontweight="bold",
        loc="left",
        pad=10,
    )
    ax.set_xlim(-0.15, 3.15)
    ax.set_ylim(0.62, 0.79)
    ax.set_xticks(alphas)
    ax.legend(fontsize=7.2, loc="lower right", framealpha=0.90, edgecolor="#CCCCCC")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.2, zorder=0)


def main() -> None:
    data = load_data()
    fig = plt.figure(figsize=(12.6, 4.8), dpi=300)
    fig.set_facecolor(BG_COLOR)
    grid = fig.add_gridspec(
        1,
        2,
        width_ratios=[0.85, 1.35],
        hspace=0.0,
        wspace=0.26,
    )

    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])

    draw_panel_a(ax_a, data)
    draw_panel_b(ax_b, data)

    fig.suptitle(
        "Figure 2: FaithEval Matched Readouts, Divergent Control",
        fontsize=14,
        fontweight="bold",
        color=TITLE_COLOR,
        y=0.985,
    )
    fig.subplots_adjust(top=0.86, bottom=0.16, left=0.06, right=0.985)
    fig.savefig(
        OUTPUT,
        dpi=300,
        bbox_inches="tight",
        facecolor=BG_COLOR,
        pad_inches=0.2,
    )
    plt.close(fig)
    print(f"Saved: {OUTPUT}")


if __name__ == "__main__":
    main()
