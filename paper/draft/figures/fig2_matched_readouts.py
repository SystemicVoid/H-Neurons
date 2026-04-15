"""
Figure 2: Matched Readouts, Divergent Control.

FaithEval is the anchor result. The jailbreak comparator appears only as
supporting evidence, and the probe summary is shown as a distribution rather
than a headline-matching bar.

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
    probe_meta = load_json(
        "data/contrastive/refusal/iti_refusal_probe_d7/extraction_metadata.json"
    )
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
    d7 = load_json(
        "data/gemma3_4b/intervention/jailbreak_d7/full500_canonical/d7_full500_current_state_summary.json"
    )

    probe_aurocs = [entry["auroc"] for entry in probe_meta["selected_head_manifest"]]
    alphas = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

    sae_results = fe_sae["results"]
    sae_ci = [sae_results[str(alpha)]["compliance"]["ci"] for alpha in alphas]

    return {
        "auroc_h": neuron_cls["evaluation"]["metrics"]["auroc"]["estimate"],
        "auroc_sae": sae_cls["best"]["test_metrics"]["auroc"],
        "probe_summary": {
            "n": len(probe_aurocs),
            "median": float(np.median(probe_aurocs)),
            "min": float(np.min(probe_aurocs)),
            "max": float(np.max(probe_aurocs)),
        },
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
        "d7_conditions": d7["current_panel"]["conditions"],
        "d7_direct": d7["current_panel"]["direct_comparisons"],
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

    probe = data["probe_summary"]
    probe_x = 1.95
    probe_y = probe["median"]
    probe_err = np.array([[probe_y - probe["min"]], [probe["max"] - probe_y]])
    ax.errorbar(
        probe_x,
        probe_y,
        yerr=probe_err,
        fmt="D",
        markersize=5.5,
        markerfacecolor="white",
        markeredgecolor=C_SUPPORT,
        markeredgewidth=1.4,
        capsize=4,
        linewidth=1.2,
        color=C_SUPPORT,
        zorder=4,
    )
    ax.text(
        probe_x,
        probe["max"] + 0.006,
        "Jailbreak probe-head AUROC\nsupporting context",
        ha="center",
        va="bottom",
        fontsize=7,
        color=C_SUPPORT,
    )
    ax.text(
        probe_x,
        probe["min"] - 0.013,
        f"median {probe_y:.3f}\nrange {probe['min']:.2f}-{probe['max']:.2f}",
        ha="center",
        va="top",
        fontsize=6.8,
        color=C_SUPPORT,
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

    ax.set_xticks([0, 1, probe_x])
    ax.set_xticklabels(
        ["H-neurons\n(38)", "SAE features\n(266)", f"Probe heads\n(n={probe['n']})"],
        fontsize=8,
    )
    ax.set_ylabel("Detection AUROC", fontsize=10, fontweight="bold")
    ax.set_ylim(0.70, 1.06)
    ax.set_title(
        "A. FaithEval anchor, supporting jailbreak AUROC context",
        fontsize=11,
        fontweight="bold",
        loc="left",
        pad=10,
    )
    ax.text(
        0.99,
        0.03,
        "Probe shown as a distribution, not a headline bar.",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7,
        color=SUBTITLE_COLOR,
        fontstyle="italic",
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


def draw_panel_c(ax: plt.Axes, data: dict) -> None:
    order = ["baseline", "random_layer_seed1", "probe", "causal"]
    labels = [
        "Baseline\n(no-op)",
        "Matched random\nheads",
        "Probe-selected\nheads",
        "Gradient-selected\nheads",
    ]
    conditions = data["d7_conditions"]
    rates = [
        conditions[name]["strict_harmfulness_normalized"]["estimate_pct"]
        for name in order
    ]
    lo = [
        conditions[name]["strict_harmfulness_normalized"]["ci"]["lower"] * 100
        for name in order
    ]
    hi = [
        conditions[name]["strict_harmfulness_normalized"]["ci"]["upper"] * 100
        for name in order
    ]
    x = np.arange(len(order))
    ax.bar(
        x,
        rates,
        yerr=[
            [rate - lower for rate, lower in zip(rates, lo)],
            [upper - rate for rate, upper in zip(rates, hi)],
        ],
        color=[C_RANDOM_FILL, "#F6EEE8", C_SAE_FILL, C_HNEURON_FILL],
        edgecolor=[SUBTITLE_COLOR, "#8B6B5D", C_SAE, C_HNEURON],
        linewidth=1.6,
        width=0.50,
        capsize=4,
        error_kw={"linewidth": 1.2, "color": SUBTITLE_COLOR},
        zorder=3,
    )
    for xi, rate in zip(x, rates):
        ax.text(
            xi,
            rate + 2.0,
            f"{rate:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    probe_vs_causal = data["d7_direct"]["probe_vs_causal"][
        "strict_harmfulness_normalized"
    ]
    rand_vs_causal = data["d7_direct"]["random_layer_seed1_vs_causal"][
        "strict_harmfulness_normalized"
    ]
    ax.text(
        0.02,
        0.98,
        "Supporting comparator only\n"
        + f"causal vs probe = {-probe_vs_causal['estimate_pp']:+.1f} pp "
        + f"[{-probe_vs_causal['ci_pp']['upper']:+.1f}, {-probe_vs_causal['ci_pp']['lower']:+.1f}]\n"
        + f"causal vs random = {-rand_vs_causal['estimate_pp']:+.1f} pp "
        + f"[{-rand_vs_causal['ci_pp']['upper']:+.1f}, {-rand_vs_causal['ci_pp']['lower']:+.1f}]",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.0,
        bbox=dict(
            boxstyle="round,pad=0.30",
            facecolor="#FAFBFC",
            edgecolor="#D4DCE3",
            alpha=0.95,
        ),
    )

    ax.set_ylabel(
        "Strict harmfulness rate (%)\n(full-500 panel with differing ruler histories)",
        fontsize=9.5,
        fontweight="bold",
    )
    ax.set_title(
        "C. Jailbreak selector comparator (supporting)",
        fontsize=11,
        fontweight="bold",
        color=SUBTITLE_COLOR,
        loc="left",
        pad=10,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.2)
    ax.set_ylim(0, 60)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.2, zorder=0)


def main() -> None:
    data = load_data()
    fig = plt.figure(figsize=(14.8, 7.9), dpi=300)
    fig.set_facecolor(BG_COLOR)
    grid = fig.add_gridspec(
        2,
        3,
        height_ratios=[1.0, 0.78],
        width_ratios=[0.95, 1.05, 1.05],
        hspace=0.38,
        wspace=0.32,
    )

    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1:])
    ax_c = fig.add_subplot(grid[1, :])

    draw_panel_a(ax_a, data)
    draw_panel_b(ax_b, data)
    draw_panel_c(ax_c, data)

    fig.suptitle(
        "Figure 2: Matched Readouts, Divergent Control",
        fontsize=14,
        fontweight="bold",
        color=TITLE_COLOR,
        y=0.985,
    )
    fig.subplots_adjust(top=0.90, bottom=0.08, left=0.055, right=0.985)
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
