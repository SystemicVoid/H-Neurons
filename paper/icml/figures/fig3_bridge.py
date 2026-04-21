"""
Figure 3 (workshop): Surface-Local Control and Bridge Failure Modes.

Simplified 2-panel version of paper/draft/figures/fig3_bridge_failure.py.
Panel A: ITI effect across task types (bar chart).
Panel B: Right-to-wrong flip taxonomy (horizontal bar).
Panel C (substitution examples) becomes a LaTeX table in the paper.

ICML two-column format (figure* full-width, 6.75in, PDF output).

Usage:
    uv run python paper/icml/figures/fig3_bridge.py
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
OUTPUT = ROOT / "paper/icml/figures/fig3_bridge.pdf"
IRR_SUMMARY = ROOT / "data/judge_validation/bridge_irr/bridge_irr_summary.json"

TITLE_COLOR = "#1E3044"
SUBTITLE_COLOR = "#5A6E7F"
BG_COLOR = "#FFFFFF"

C_POSITIVE = "#3E6A8A"
C_NEGATIVE = "#BF4E38"
C_POSITIVE_FILL = "#DAEAF6"
C_NEGATIVE_FILL = "#FDF1ED"

C_SUBSTITUTION = "#BF4E38"
C_EVASION = "#8899A6"
C_VERBOSITY = "#D4915E"
C_OTHER = "#B8C5D0"

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

PANEL_A_DATA = [
    {
        "label": "TruthfulQA MC1\n(answer selection)",
        "delta": +6.3,
        "ci_lo": +3.7,
        "ci_hi": +8.9,
        "direction": "positive",
    },
    {
        "label": "SimpleQA\n(generation)",
        "delta": -1.8,
        "ci_lo": -3.1,
        "ci_hi": -0.6,
        "direction": "negative",
    },
    {
        "label": "TriviaQA bridge\n(generation, test)",
        "delta": -5.8,
        "ci_lo": -8.8,
        "ci_hi": -3.0,
        "direction": "negative",
    },
]

DEFAULT_FLIP_TAXONOMY = [
    ("Wrong-entity\nsubstitution", 30, C_SUBSTITUTION),
    ("Evasion /\nfactual denial", 8, C_EVASION),
    ("Verbosity /\ndilution", 3, C_VERBOSITY),
    ("Formal\nrefusal", 2, C_OTHER),
]


def load_flip_taxonomy() -> list[tuple[str, int, str]]:
    if not IRR_SUMMARY.exists():
        return DEFAULT_FLIP_TAXONOMY
    payload = json.loads(IRR_SUMMARY.read_text(encoding="utf-8"))
    if payload.get("status") != "adjudicated":
        return DEFAULT_FLIP_TAXONOMY
    categories = payload["direction_summaries"]["right_to_wrong"]["categories"]
    return [
        (
            "Wrong-entity\nsubstitution",
            int(categories["wrong_entity_substitution"]["count"]),
            C_SUBSTITUTION,
        ),
        (
            "Evasion /\nfactual denial",
            int(categories["evasion_or_factual_denial"]["count"]),
            C_EVASION,
        ),
        (
            "Verbosity /\ndilution",
            int(categories["answer_dilution"]["count"]),
            C_VERBOSITY,
        ),
        (
            "Formal\nrefusal",
            int(categories["formal_refusal"]["count"]),
            C_OTHER,
        ),
    ]


def draw_panel_a(ax: plt.Axes) -> None:
    n = len(PANEL_A_DATA)
    x = np.arange(n)
    deltas = [float(d["delta"]) for d in PANEL_A_DATA]
    ci_los = [float(d["ci_lo"]) for d in PANEL_A_DATA]
    ci_his = [float(d["ci_hi"]) for d in PANEL_A_DATA]
    err_lo = [d - lo for d, lo in zip(deltas, ci_los)]
    err_hi = [hi - d for d, hi in zip(deltas, ci_his)]

    fill_colors = [
        C_POSITIVE_FILL if d["direction"] == "positive" else C_NEGATIVE_FILL
        for d in PANEL_A_DATA
    ]
    edge_colors = [
        C_POSITIVE if d["direction"] == "positive" else C_NEGATIVE for d in PANEL_A_DATA
    ]

    bars = ax.bar(
        x,
        deltas,
        color=fill_colors,
        edgecolor=edge_colors,
        linewidth=1.2,
        width=0.50,
        zorder=3,
        yerr=[err_lo, err_hi],
        capsize=4,
        error_kw=dict(linewidth=1.0, capthick=1.0, color=SUBTITLE_COLOR),
    )

    for i, (bar, d) in enumerate(zip(bars, PANEL_A_DATA)):
        delta = float(d["delta"])
        ci_lo_val = float(d["ci_lo"])
        ci_hi_val = float(d["ci_hi"])
        ci_text = f"[{ci_lo_val:+.1f}, {ci_hi_val:+.1f}]"
        sign = "+" if delta > 0 else ""
        if delta >= 0:
            y_pos = delta + err_hi[i] + 0.5
            va = "bottom"
        else:
            y_pos = delta - err_lo[i] - 0.5
            va = "top"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y_pos,
            f"{sign}{delta:.1f} pp\n{ci_text}",
            ha="center",
            va=va,
            fontsize=6,
            fontweight="bold",
            color=edge_colors[i],
            linespacing=1.15,
        )

    labels = [str(d["label"]) for d in PANEL_A_DATA]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=5.5)
    ax.axhline(0, color=SUBTITLE_COLOR, linewidth=0.6, linestyle="-", zorder=1)
    ax.set_ylabel("ITI effect (pp)", fontsize=7, fontweight="bold")
    ax.set_title(
        "(a) ITI: surface-local gain, generation harm",
        fontsize=7.5,
        fontweight="bold",
        color=TITLE_COLOR,
        loc="left",
        pad=5,
    )
    ax.set_ylim(-18, 14)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="y", labelsize=6)
    ax.grid(axis="y", alpha=0.2, zorder=0)


def draw_panel_b(ax: plt.Axes) -> None:
    flip_taxonomy = load_flip_taxonomy()
    labels = [t[0] for t in flip_taxonomy]
    counts = [t[1] for t in flip_taxonomy]
    colors = [t[2] for t in flip_taxonomy]

    y = np.arange(len(labels))
    bars = ax.barh(
        y,
        counts,
        color=colors,
        edgecolor=[TITLE_COLOR] * len(labels),
        linewidth=0.6,
        height=0.50,
        zorder=3,
    )

    total = sum(counts)
    for bar, count in zip(bars, counts):
        pct = count / total * 100
        ax.text(
            bar.get_width() + 0.15,
            bar.get_y() + bar.get_height() / 2,
            f"{count}  ({pct:.0f}%)",
            ha="left",
            va="center",
            fontsize=6.5,
            fontweight="bold",
            color=TITLE_COLOR,
        )

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=6.5)
    ax.invert_yaxis()
    ax.set_xlabel("Count (n=43 flips)", fontsize=7, fontweight="bold")
    ax.set_title(
        r"(b) R$\rightarrow$W flip taxonomy (bridge test, n=500)",
        fontsize=7.5,
        fontweight="bold",
        color=TITLE_COLOR,
        loc="left",
        pad=5,
    )
    ax.set_xlim(0, 38.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", labelsize=6)
    ax.grid(axis="x", alpha=0.2, zorder=0)


def main() -> None:
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(6.75, 2.3), dpi=300)
    fig.set_facecolor(BG_COLOR)
    fig.subplots_adjust(wspace=0.40)
    draw_panel_a(ax_a)
    draw_panel_b(ax_b)
    fig.savefig(
        OUTPUT, dpi=300, bbox_inches="tight", facecolor=BG_COLOR, pad_inches=0.04
    )
    plt.close(fig)
    print(f"Saved: {OUTPUT}")


if __name__ == "__main__":
    main()
