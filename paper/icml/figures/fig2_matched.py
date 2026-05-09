"""
Figure 2 (workshop): Matched Readouts, Divergent Control.

Adapted from the archived long-draft Figure 2 script for ICML
two-column format (figure* full-width, 6.75in, PDF output).

Usage:
    uv run python paper/icml/figures/fig2_matched.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / "paper/icml/figures/fig2_matched.pdf"

TITLE_COLOR = "#1E3044"
SUBTITLE_COLOR = "#5A6E7F"
BG_COLOR = "#FFFFFF"

C_HNEURON = "#3E6A8A"
C_SAE = "#BF4E38"
C_RANDOM = "#8899A6"

C_HNEURON_FILL = "#DAEAF6"
C_SAE_FILL = "#FDF1ED"
C_RANDOM_FILL = "#E8ECF0"

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times", "Times New Roman"],
        "text.usetex": True,
        "text.color": TITLE_COLOR,
        "axes.labelcolor": TITLE_COLOR,
        "xtick.color": SUBTITLE_COLOR,
        "ytick.color": SUBTITLE_COLOR,
        "font.size": 7,
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
    fe_sae_delta_ctrl = load_json(
        "data/gemma3_4b/intervention/faitheval_sae_delta/control/comparison_summary.json"
    )
    alphas = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    sae_results = fe_sae["results"]
    sae_ci = [sae_results[str(alpha)]["compliance"]["ci"] for alpha in alphas]
    h_auroc = neuron_cls["evaluation"]["metrics"]["auroc"]
    sae_auroc = sae_cls["evaluation"]["metrics"]["auroc"]

    return {
        "auroc_h": h_auroc["estimate"],
        "auroc_h_ci": h_auroc["ci"],
        "auroc_sae": sae_auroc["estimate"],
        "auroc_sae_ci": sae_auroc["ci"],
        "alphas": np.array(alphas),
        "h_rates": np.array(fe_neuron_ctrl["h_neuron_baseline"]["compliance_rates"]),
        "h_ci_lo": np.array(
            [
                e["ci"]["lower"]
                for e in fe_neuron_ctrl["h_neuron_baseline"]["compliance_ci_by_alpha"]
            ]
        ),
        "h_ci_hi": np.array(
            [
                e["ci"]["upper"]
                for e in fe_neuron_ctrl["h_neuron_baseline"]["compliance_ci_by_alpha"]
            ]
        ),
        "sae_rates": np.array(fe_sae["effects"]["compliance_curve"]["rates"]),
        "sae_ci_lo": np.array([e["lower"] for e in sae_ci]),
        "sae_ci_hi": np.array([e["upper"] for e in sae_ci]),
        "rand_rates": np.array(
            fe_sae_ctrl["random_sae_features"]["mean_compliance_rates"]
        ),
        "rand_std": np.array(
            fe_sae_ctrl["random_sae_features"]["std_compliance_rates"]
        ),
        "delta_h_slope": fe_sae_delta_ctrl["h_sae_features"]["slope_per_alpha"],
        "delta_rand_slope": fe_sae_delta_ctrl["random_sae_features"][
            "mean_slope_per_alpha"
        ],
        "slope_diff": fe_sae_slope_diff["slope_difference_pp_per_alpha"]["estimate"],
        "slope_diff_ci": fe_sae_slope_diff["slope_difference_pp_per_alpha"]["ci"],
    }


def draw_panel_a(ax: plt.Axes, data: dict) -> None:
    x = np.array([0.0, 1.0])
    values = [data["auroc_h"], data["auroc_sae"]]
    ci_bounds = [data["auroc_h_ci"], data["auroc_sae_ci"]]
    err_lo = [val - ci["lower"] for val, ci in zip(values, ci_bounds, strict=True)]
    err_hi = [ci["upper"] - val for val, ci in zip(values, ci_bounds, strict=True)]
    bars = ax.bar(
        x,
        values,
        color=[C_HNEURON_FILL, C_SAE_FILL],
        edgecolor=[C_HNEURON, C_SAE],
        linewidth=1.2,
        width=0.50,
        yerr=[err_lo, err_hi],
        capsize=3,
        error_kw={"linewidth": 0.9, "color": SUBTITLE_COLOR},
        zorder=3,
    )
    for bar, val, ci in zip(bars, values, ci_bounds, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            ci["upper"] + 0.004,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=7,
            fontweight="bold",
            color=TITLE_COLOR,
        )
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["H-neurons\n(38)", "SAE features\n(266)"], fontsize=6)
    ax.set_ylabel("Detection AUROC", fontsize=7, fontweight="bold")
    ax.set_ylim(0.70, 0.885)
    ax.set_title(
        "(a) Matched detection quality",
        fontsize=7.5,
        fontweight="bold",
        loc="left",
        pad=5,
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
        linewidth=1.5,
        marker="o",
        markersize=3.5,
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
        linewidth=1.3,
        marker="s",
        markersize=3.5,
        label="SAE H-feat. (full)",
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
        linewidth=1.0,
        linestyle="--",
        marker="^",
        markersize=3,
        label="SAE rand. (full)",
        zorder=3,
    )

    slope_ci = data["slope_diff_ci"]
    ax.text(
        0.03,
        0.97,
        r"$\Delta$"
        + f" slope = {data['slope_diff']:+.2f} pp/"
        + r"$\alpha$"
        + f"\n95\\% CI [{slope_ci['lower']:+.2f}, {slope_ci['upper']:+.2f}]",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=5.5,
        clip_on=False,
        bbox=dict(
            boxstyle="round,pad=0.30",
            facecolor="white",
            edgecolor="#D4DCE3",
            alpha=0.95,
        ),
    )

    ax.set_xlabel(r"Scaling factor ($\alpha$)", fontsize=7, fontweight="bold")
    ax.set_ylabel("Compliance rate", fontsize=7, fontweight="bold")
    ax.set_title(
        "(b) Steering divergence on FaithEval",
        fontsize=7.5,
        fontweight="bold",
        loc="left",
        pad=5,
    )
    ax.set_xlim(-0.15, 3.15)
    ax.set_ylim(0.62, 0.79)
    ax.set_xticks(alphas)
    ax.legend(fontsize=5.2, loc="lower right", framealpha=0.90, edgecolor="#CCCCCC")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.2, zorder=0)


def main() -> None:
    data = load_data()
    fig = plt.figure(figsize=(6.75, 2.6), dpi=300)
    fig.set_facecolor(BG_COLOR)
    grid = fig.add_gridspec(1, 2, width_ratios=[0.85, 1.35], wspace=0.30)
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    draw_panel_a(ax_a, data)
    draw_panel_b(ax_b, data)
    fig.subplots_adjust(bottom=0.24)
    fig.text(
        0.54,
        0.035,
        "SAE curves show the full-replacement path; non-monotone movement is shared "
        "by target and random SAE.\n"
        + r"Feature-specific delta-only slopes: H "
        + f"{data['delta_h_slope']:+.2f}, random "
        + f"{data['delta_rand_slope']:+.2f} pp/"
        + r"$\alpha$ (audit summary; slope CIs not claimed).",
        ha="center",
        va="bottom",
        fontsize=6,
        color=SUBTITLE_COLOR,
        linespacing=1.15,
    )
    fig.savefig(
        OUTPUT, dpi=300, bbox_inches="tight", facecolor=BG_COLOR, pad_inches=0.08
    )
    plt.close(fig)
    print(f"Saved: {OUTPUT}")


if __name__ == "__main__":
    main()
