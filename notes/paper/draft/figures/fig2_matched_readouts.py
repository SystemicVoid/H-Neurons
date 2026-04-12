"""
Figure 2: Matched Readouts, Divergent Control.

Flagship empirical figure for "Detection Is Not Enough" showing that
representation methods with comparable detection AUROC produce sharply
different intervention outcomes.

Panel A: Detection quality (AUROC) for three methods.
Panel B: FaithEval compliance dose-response for H-neurons vs SAE features
         vs random neuron control.
Panel C: Jailbreak csv2_yes rates (full-500 D7 confirmatory run) for
         causal head intervention vs L1-neuron comparator vs shared baseline.

Data sources (all real, loaded from repo JSON):
  - data/gemma3_4b/pipeline/classifier_disjoint_summary.json
  - data/gemma3_4b/pipeline/classifier_sae_summary.json
  - data/gemma3_4b/intervention/faitheval/experiment/results.json
  - data/gemma3_4b/intervention/faitheval/control/comparison_summary.json
  - data/gemma3_4b/intervention/faitheval_sae/experiment/results.json
  - data/gemma3_4b/intervention/faitheval_sae/control/comparison_summary.json
  - data/gemma3_4b/intervention/jailbreak_d7/full500_canonical/d7_csv2_report.json
  - data/contrastive/refusal/iti_refusal_probe_d7/extraction_metadata.json

Usage:
    uv run python notes/paper/draft/figures/fig2_matched_readouts.py
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
OUTPUT = ROOT / "notes/paper/draft/figures/fig2_matched_readouts.png"

# ---------------------------------------------------------------------------
# Color palette -- consistent with Figure 1 (muted blues / terracotta)
# ---------------------------------------------------------------------------
TITLE_COLOR = "#1E3044"
SUBTITLE_COLOR = "#5A6E7F"
BG_COLOR = "#FFFFFF"

# Data-series colors
C_HNEURON = "#3E6A8A"  # dark steel blue -- primary
C_SAE = "#BF4E38"  # muted terracotta -- SAE
C_RANDOM = "#8899A6"  # medium gray-blue -- random control
C_CAUSAL = "#3E6A8A"  # dark steel blue -- causal intervention
C_L1 = "#BF4E38"  # terracotta -- L1 comparator (worsens)
C_BASELINE = "#5A6E7F"  # mid-gray -- baseline

C_HNEURON_FILL = "#DAEAF6"
C_SAE_FILL = "#FDF1ED"
C_RANDOM_FILL = "#E8ECF0"

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


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_json(rel_path: str) -> dict:
    """Load a JSON file relative to repo root."""
    with open(ROOT / rel_path) as f:
        return json.load(f)


def load_all_data() -> dict:
    """Load and return all required data in a structured dict."""
    data: dict = {}

    # ------------------------------------------------------------------
    # Panel A: AUROC values
    # ------------------------------------------------------------------
    neuron_cls = load_json("data/gemma3_4b/pipeline/classifier_disjoint_summary.json")
    sae_cls = load_json("data/gemma3_4b/pipeline/classifier_sae_summary.json")
    probe_meta = load_json(
        "data/contrastive/refusal/iti_refusal_probe_d7/extraction_metadata.json"
    )

    data["auroc_hneuron"] = neuron_cls["evaluation"]["metrics"]["auroc"]["estimate"]
    data["auroc_hneuron_ci"] = neuron_cls["evaluation"]["metrics"]["auroc"]["ci"]
    data["auroc_sae"] = sae_cls["best"]["test_metrics"]["auroc"]

    # Probe heads: top-20 selected heads with per-head AUROC
    head_manifest = probe_meta["selected_head_manifest"]
    data["auroc_probe_heads"] = [h["auroc"] for h in head_manifest]
    data["auroc_probe_best"] = max(data["auroc_probe_heads"])

    # ------------------------------------------------------------------
    # Panel B: FaithEval compliance curves
    # ------------------------------------------------------------------
    fe_neuron_ctrl = load_json(
        "data/gemma3_4b/intervention/faitheval/control/comparison_summary.json"
    )
    fe_sae = load_json(
        "data/gemma3_4b/intervention/faitheval_sae/experiment/results.json"
    )

    alphas_fe = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

    data["fe_alphas"] = alphas_fe
    data["fe_hn_rates"] = fe_neuron_ctrl["h_neuron_baseline"]["compliance_rates"]
    data["fe_hn_ci"] = fe_neuron_ctrl["h_neuron_baseline"]["compliance_ci_by_alpha"]

    data["fe_sae_rates"] = fe_sae["effects"]["compliance_curve"]["rates"]
    data["fe_sae_ci"] = list(fe_sae["results"].values())

    data["fe_random_rates"] = fe_neuron_ctrl["unconstrained_random"][
        "mean_compliance_rates"
    ]
    data["fe_random_std"] = fe_neuron_ctrl["unconstrained_random"][
        "std_compliance_rates"
    ]

    # ------------------------------------------------------------------
    # Panel C: Jailbreak D7 full-500
    # ------------------------------------------------------------------
    d7 = load_json(
        "data/gemma3_4b/intervention/jailbreak_d7/full500_canonical/d7_csv2_report.json"
    )
    data["d7_conditions"] = d7["conditions"]
    data["d7_paired"] = d7["paired_vs_baseline"]

    return data


# ---------------------------------------------------------------------------
# Panel A: Detection AUROC comparison
# ---------------------------------------------------------------------------
def draw_panel_a(ax: plt.Axes, data: dict) -> None:
    """Grouped bar chart of detection AUROC for three methods."""
    methods = [
        "H-neurons\n(38 neurons)",
        "SAE features\n(266 features)",
        "Probe heads\n(D7, top-20)",
    ]
    aurocs = [
        data["auroc_hneuron"],
        data["auroc_sae"],
        data["auroc_probe_best"],
    ]
    fill_colors = [C_HNEURON_FILL, C_SAE_FILL, C_RANDOM_FILL]
    edge_colors = [C_HNEURON, C_SAE, C_BASELINE]

    bars = ax.bar(
        methods,
        aurocs,
        color=fill_colors,
        edgecolor=edge_colors,
        linewidth=1.8,
        width=0.55,
        zorder=3,
    )

    # Value labels on bars
    for bar, val in zip(bars, aurocs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.007,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=10.5,
            fontweight="bold",
            color=TITLE_COLOR,
        )

    ax.set_ylim(0.70, 1.06)
    ax.set_ylabel("Detection AUROC", fontsize=10, fontweight="bold")
    ax.set_title(
        "A. Matched detection quality",
        fontsize=11,
        fontweight="bold",
        color=TITLE_COLOR,
        loc="left",
        pad=10,
    )

    # Bracket + delta annotation between H-neurons and SAE
    bracket_y = 0.818
    ax.plot([0, 1], [bracket_y, bracket_y], color=SUBTITLE_COLOR, linewidth=0.8)
    ax.plot([0, 0], [bracket_y, bracket_y + 0.005], color=SUBTITLE_COLOR, linewidth=0.8)
    ax.plot([1, 1], [bracket_y, bracket_y + 0.005], color=SUBTITLE_COLOR, linewidth=0.8)
    ax.text(
        0.5,
        bracket_y - 0.008,
        r"$\Delta$ = 0.005",
        ha="center",
        va="top",
        fontsize=8,
        color=SUBTITLE_COLOR,
        fontstyle="italic",
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", labelsize=8)
    ax.tick_params(axis="y", labelsize=8.5)
    ax.grid(axis="y", alpha=0.2, zorder=0)


# ---------------------------------------------------------------------------
# Panel B: FaithEval dose-response
# ---------------------------------------------------------------------------
def draw_panel_b(ax: plt.Axes, data: dict) -> None:
    """Line plot of compliance rate vs alpha for H-neurons, SAE, random."""
    alphas = np.array(data["fe_alphas"])

    # --- H-neuron curve with Wilson CIs ---
    hn_rates = np.array(data["fe_hn_rates"])
    hn_lo = np.array([c["ci"]["lower"] for c in data["fe_hn_ci"]])
    hn_hi = np.array([c["ci"]["upper"] for c in data["fe_hn_ci"]])

    ax.fill_between(alphas, hn_lo, hn_hi, alpha=0.15, color=C_HNEURON, zorder=2)
    ax.plot(
        alphas,
        hn_rates,
        color=C_HNEURON,
        linewidth=2.2,
        marker="o",
        markersize=5,
        label="H-neuron scaling (n=38)",
        zorder=4,
    )

    # --- SAE feature curve with Wilson CIs ---
    # Note: alpha=1.0 is the true no-op (bypass SAE encode/decode cycle).
    # Other alphas pass through the lossy SAE, inflating compliance ~8-9pp.
    # We plot the full curve but mark alpha=1.0 distinctly.
    sae_rates = np.array(data["fe_sae_rates"])
    sae_ci_data = data["fe_sae_ci"]
    sae_lo = np.array([c["compliance"]["ci"]["lower"] for c in sae_ci_data])
    sae_hi = np.array([c["compliance"]["ci"]["upper"] for c in sae_ci_data])

    ax.fill_between(alphas, sae_lo, sae_hi, alpha=0.10, color=C_SAE, zorder=2)
    ax.plot(
        alphas,
        sae_rates,
        color=C_SAE,
        linewidth=2.2,
        marker="s",
        markersize=5,
        label="SAE feature scaling (n=266)",
        zorder=4,
    )

    # Mark alpha=1.0 SAE point (true no-op, before lossy cycle)
    ax.plot(
        1.0,
        sae_rates[2],
        marker="s",
        markersize=7,
        markerfacecolor="white",
        markeredgecolor=C_SAE,
        markeredgewidth=2,
        zorder=5,
    )
    ax.annotate(
        "no-op\n(bypass SAE)",
        xy=(1.0, sae_rates[2] + 0.004),
        xytext=(1.55, 0.635),
        fontsize=6.5,
        color=C_SAE,
        fontstyle="italic",
        ha="center",
        arrowprops=dict(
            arrowstyle="->",
            color=C_SAE,
            linewidth=0.8,
            connectionstyle="arc3,rad=-0.2",
        ),
    )

    # --- Random neuron control (mean +/- 1 SD across 5 seeds) ---
    rand_rates = np.array(data["fe_random_rates"])
    rand_std = np.array(data["fe_random_std"])

    ax.fill_between(
        alphas,
        rand_rates - rand_std,
        rand_rates + rand_std,
        alpha=0.18,
        color=C_RANDOM,
        zorder=1,
    )
    ax.plot(
        alphas,
        rand_rates,
        color=C_RANDOM,
        linewidth=1.5,
        marker="^",
        markersize=4,
        linestyle="--",
        label="Random neurons (5 seeds)",
        zorder=3,
    )

    # --- Slope annotations ---
    ax.annotate(
        "+2.1 pp/\u03b1\n(monotonic)",
        xy=(2.5, 0.695),
        xytext=(0.65, 0.715),
        fontsize=8,
        fontweight="bold",
        color=C_HNEURON,
        arrowprops=dict(arrowstyle="->", color=C_HNEURON, linewidth=1.0),
    )

    ax.annotate(
        "slope \u2248 0 pp/\u03b1",
        xy=(2.75, 0.73),
        xytext=(2.9, 0.775),
        fontsize=7.5,
        fontweight="bold",
        color=C_SAE,
        ha="center",
        arrowprops=dict(arrowstyle="->", color=C_SAE, linewidth=0.8),
    )

    ax.set_xlabel("Scaling factor (\u03b1)", fontsize=10, fontweight="bold")
    ax.set_ylabel("Compliance rate", fontsize=10, fontweight="bold")
    ax.set_title(
        "B. FaithEval: divergent steering",
        fontsize=11,
        fontweight="bold",
        color=TITLE_COLOR,
        loc="left",
        pad=10,
    )

    ax.set_xlim(-0.15, 3.15)
    ax.set_ylim(0.62, 0.79)
    ax.set_xticks(alphas)

    ax.legend(
        fontsize=7.5,
        loc="lower right",
        framealpha=0.90,
        edgecolor="#CCCCCC",
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=8.5)
    ax.grid(alpha=0.2, zorder=0)


# ---------------------------------------------------------------------------
# Panel C: Jailbreak D7 -- full-500 confirmatory results
# ---------------------------------------------------------------------------
def draw_panel_c(ax: plt.Axes, data: dict) -> None:
    """Bar chart of csv2_yes rates for baseline / L1 / causal (D7 full-500)."""
    conditions = data["d7_conditions"]
    paired = data["d7_paired"]

    # Extract csv2_yes rates (as percentages)
    names_raw = [c["name"] for c in conditions]
    rates = [c["csv2_yes"]["estimate"] * 100 for c in conditions]
    ci_lo = [c["csv2_yes"]["ci"]["lower"] * 100 for c in conditions]
    ci_hi = [c["csv2_yes"]["ci"]["upper"] * 100 for c in conditions]
    errors_lo = [r - lo for r, lo in zip(rates, ci_lo)]
    errors_hi = [hi - r for r, hi in zip(rates, ci_hi)]

    # Paper-facing names
    display_names = {
        "baseline": "Baseline\n(no-op)",
        "l1": "L1-neuron\nscaling (\u03b1=3)",
        "causal": "Causal-head\nITI (\u03b1=4)",
    }
    labels = [display_names.get(n, n) for n in names_raw]
    fill_colors = [C_RANDOM_FILL, C_SAE_FILL, C_HNEURON_FILL]
    edge_colors = [C_BASELINE, C_L1, C_CAUSAL]

    x = np.arange(len(labels))
    bars = ax.bar(
        x,
        rates,
        yerr=[errors_lo, errors_hi],
        color=fill_colors,
        edgecolor=edge_colors,
        linewidth=1.8,
        width=0.50,
        capsize=5,
        error_kw=dict(linewidth=1.3, capthick=1.3, color=SUBTITLE_COLOR),
        zorder=3,
    )

    # Value labels above error bars
    for i, (bar, rate) in enumerate(zip(bars, rates)):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + errors_hi[i] + 0.8,
            f"{rate:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10.5,
            fontweight="bold",
            color=TITLE_COLOR,
        )

    # Paired effect annotations with CI
    # L1 vs baseline: +4.0pp (worsens)
    l1_delta = paired["l1"]["csv2_yes"]["estimate_pp"]
    l1_ci = paired["l1"]["csv2_yes"]["ci_pp"]
    ax.text(
        1,
        rates[1] + errors_hi[1] + 5.0,
        f"+{l1_delta:.1f} pp [{l1_ci['lower']:+.1f}, {l1_ci['upper']:+.1f}]",
        ha="center",
        va="bottom",
        fontsize=7.5,
        color=C_L1,
        fontweight="bold",
        bbox=dict(
            boxstyle="round,pad=0.25",
            facecolor=C_SAE_FILL,
            edgecolor=C_L1,
            linewidth=0.6,
            alpha=0.9,
        ),
    )

    # Causal vs baseline: -9.0pp (improves)
    causal_delta = paired["causal"]["csv2_yes"]["estimate_pp"]
    causal_ci = paired["causal"]["csv2_yes"]["ci_pp"]
    ax.text(
        2,
        rates[2] + errors_hi[2] + 5.0,
        f"{causal_delta:.1f} pp [{causal_ci['lower']:.1f}, {causal_ci['upper']:.1f}]",
        ha="center",
        va="bottom",
        fontsize=7.5,
        color=C_CAUSAL,
        fontweight="bold",
        bbox=dict(
            boxstyle="round,pad=0.25",
            facecolor=C_HNEURON_FILL,
            edgecolor=C_CAUSAL,
            linewidth=0.6,
            alpha=0.9,
        ),
    )

    # Reference line at baseline
    ax.axhline(
        rates[0], color=C_BASELINE, linewidth=0.8, linestyle=":", alpha=0.5, zorder=1
    )

    ax.set_ylabel(
        "Strict harmfulness rate (%)\n(csv2_yes, n=500)",
        fontsize=9.5,
        fontweight="bold",
    )
    ax.set_title(
        "C. Jailbreak: opposite outcomes",
        fontsize=11,
        fontweight="bold",
        color=TITLE_COLOR,
        loc="left",
        pad=10,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)

    ax.set_ylim(0, 45)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", labelsize=8.5)
    ax.tick_params(axis="y", labelsize=8.5)
    ax.grid(axis="y", alpha=0.2, zorder=0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    """Assemble and save the three-panel figure."""
    data = load_all_data()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5), dpi=300)

    draw_panel_a(axes[0], data)
    draw_panel_b(axes[1], data)
    draw_panel_c(axes[2], data)

    fig.suptitle(
        "Figure 2: Matched Readouts, Divergent Control",
        fontsize=14,
        fontweight="bold",
        color=TITLE_COLOR,
        y=1.02,
    )

    fig.tight_layout(w_pad=3.5)
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
