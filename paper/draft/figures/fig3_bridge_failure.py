"""
Figure 3: Surface-Local Control and Bridge Failure Modes.

Shows that ITI's MC truthfulness gain does not transfer to free-form
generation: wrong-entity substitution is the dominant failure mode.

Panel A: Grouped bar chart of ITI effect across task types (MC vs generation).
Panel B: Taxonomy of right-to-wrong flips at E0 alpha=8 on TriviaQA bridge test set.
Panel C: Text table of substitution examples showing entity-swap pattern.

Data sources (from audited reports):
  - notes/act3-reports/2026-04-01-priority-reruns-audit.md (ITI MC1 & SimpleQA)
  - notes/act3-reports/2026-04-13-bridge-phase3-test-results.md (bridge test-set results)
  - notes/act3-reports/2026-04-04-bridge-phase2-dev-results.md (E1 dev-only comparison)

Usage:
    uv run python paper/draft/figures/fig3_bridge_failure.py
"""

from __future__ import annotations

from pathlib import Path
from textwrap import fill

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[3]  # repo root
OUTPUT = ROOT / "paper/draft/figures/fig3_bridge_failure.png"

# ---------------------------------------------------------------------------
# Color palette -- consistent with Figures 1 & 2
# ---------------------------------------------------------------------------
TITLE_COLOR = "#1E3044"  # near-black slate
SUBTITLE_COLOR = "#5A6E7F"  # mid-gray blue
BG_COLOR = "#FFFFFF"

# Semantic colors for positive/negative effects
C_POSITIVE = "#3E6A8A"  # dark steel blue (improvement)
C_NEGATIVE = "#BF4E38"  # muted terracotta (harm)
C_NEUTRAL = "#8899A6"  # medium gray-blue
C_POSITIVE_FILL = "#DAEAF6"  # light steel blue fill
C_NEGATIVE_FILL = "#FDF1ED"  # light terracotta fill

# Pie/bar category colors (Panel B)
C_SUBSTITUTION = "#BF4E38"  # terracotta -- dominant failure
C_VERBOSITY = "#D4915E"  # warm amber
C_EVASION = "#8899A6"  # gray-blue
C_OTHER = "#B8C5D0"  # light gray

# Table colors (Panel C)
C_TABLE_HEADER_BG = "#DAEAF6"
C_TABLE_ROW_ALT = "#F7FAFC"
C_TABLE_CORRECT = "#3E6A8A"
C_TABLE_WRONG = "#BF4E38"
C_TABLE_BORDER = "#8899A6"

# ---------------------------------------------------------------------------
# Font settings -- match fig1/fig2
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
# Hard-coded data from audited reports
# ---------------------------------------------------------------------------

# Panel A: ITI effect deltas and CIs
PANEL_A_DATA = [
    {
        "label": "TruthfulQA MC1\n(surface-local)",
        "delta": +6.3,
        "ci_lo": +3.7,
        "ci_hi": +8.9,
        "direction": "positive",
    },
    {
        "label": "SimpleQA\ncorrect-answer rate",
        "delta": -1.8,
        "ci_lo": -3.1,
        "ci_hi": -0.6,
        "direction": "negative",
    },
    {
        "label": "TriviaQA bridge\n(E0 \u03b1=8, test)",
        "delta": -5.8,
        "ci_lo": -8.8,
        "ci_hi": -3.0,
        "direction": "negative",
    },
]

# Panel B: Flip taxonomy from Phase 3 test report section 5.1
# (43 right-to-wrong flips at E0 alpha=8.0 on held-out test set)
FLIP_TAXONOMY = [
    ("Wrong-entity\nsubstitution", 30, C_SUBSTITUTION),
    ("Evasion /\nfactual denial", 8, C_EVASION),
    ("Verbosity /\ndilution", 3, C_VERBOSITY),
    ("Formal\nrefusal", 2, C_OTHER),
]

# Panel C: Substitution examples from Phase 3 test report section 5.2
SUBSTITUTION_EXAMPLES = [
    (
        "Danny Boyle 1996 film?",
        '"Trainspotting"',
        '"Slumdog Millionaire" (same director)',
    ),
    (
        "Third musician in 1959 crash?",
        '"Ritchie Valens"',
        '"J.P. Richardson" (same crash)',
    ),
    (
        "Family Guy spin-off character?",
        '"Cleveland Brown"',
        '"Peter Griffin" (same show)',
    ),
    (
        "DC comic introducing Superman?",
        '"Action Comics"',
        '"Detective Comics" (same publisher)',
    ),
]


# ---------------------------------------------------------------------------
# Panel A: ITI effect across task types
# ---------------------------------------------------------------------------
def draw_panel_a(ax: plt.Axes) -> None:
    """Grouped bar chart: ITI MC improvement vs generation harm."""
    n = len(PANEL_A_DATA)
    x = np.arange(n)
    deltas = [float(d["delta"]) for d in PANEL_A_DATA]
    ci_los = [float(d["ci_lo"]) for d in PANEL_A_DATA]
    ci_his = [float(d["ci_hi"]) for d in PANEL_A_DATA]

    # Error bars: distance from point estimate to CI bounds
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
        linewidth=1.8,
        width=0.55,
        zorder=3,
        yerr=[err_lo, err_hi],
        capsize=6,
        error_kw=dict(linewidth=1.5, capthick=1.5, color=SUBTITLE_COLOR),
    )

    # Value + CI labels
    for i, (bar, d) in enumerate(zip(bars, PANEL_A_DATA)):
        delta = float(d["delta"])
        ci_lo_val = float(d["ci_lo"])
        ci_hi_val = float(d["ci_hi"])
        ci_text = f"[{ci_lo_val:+.1f}, {ci_hi_val:+.1f}]"
        sign = "+" if delta > 0 else ""

        # Position label above positive bars, below negative bars
        if delta >= 0:
            y_pos = delta + err_hi[i] + 0.7
            va = "bottom"
        else:
            y_pos = delta - err_lo[i] - 0.7
            va = "top"

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y_pos,
            f"{sign}{delta:.1f} pp\n{ci_text}",
            ha="center",
            va=va,
            fontsize=9,
            fontweight="bold",
            color=edge_colors[i],
            linespacing=1.2,
        )

    labels = [str(d["label"]) for d in PANEL_A_DATA]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)

    ax.axhline(0, color=SUBTITLE_COLOR, linewidth=0.8, linestyle="-", zorder=1)

    ax.set_ylabel("ITI effect (pp)", fontsize=10, fontweight="bold")
    ax.set_title(
        "A. ITI: surface-local gain, generation harm",
        fontsize=11,
        fontweight="bold",
        color=TITLE_COLOR,
        loc="left",
        pad=10,
    )

    ax.set_ylim(-22, 14)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="y", labelsize=8.5)
    ax.grid(axis="y", alpha=0.2, zorder=0)


# ---------------------------------------------------------------------------
# Panel B: Bridge flip taxonomy (horizontal bar)
# ---------------------------------------------------------------------------
def draw_panel_b(ax: plt.Axes) -> None:
    """Horizontal bar chart of right-to-wrong flip categories."""
    labels = [t[0] for t in FLIP_TAXONOMY]
    counts = [t[1] for t in FLIP_TAXONOMY]
    colors = [t[2] for t in FLIP_TAXONOMY]

    y = np.arange(len(labels))
    bars = ax.barh(
        y,
        counts,
        color=colors,
        edgecolor=[TITLE_COLOR] * len(labels),
        linewidth=0.8,
        height=0.55,
        zorder=3,
    )

    # Count and percentage labels
    total = sum(counts)
    for i, (bar, count) in enumerate(zip(bars, counts)):
        pct = count / total * 100
        ax.text(
            bar.get_width() + 0.15,
            bar.get_y() + bar.get_height() / 2,
            f"{count}  ({pct:.0f}%)",
            ha="left",
            va="center",
            fontsize=9.5,
            fontweight="bold",
            color=TITLE_COLOR,
        )

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9.5)
    ax.invert_yaxis()

    ax.set_xlabel("Count (n=43 flips)", fontsize=9.5, fontweight="bold")
    ax.set_title(
        "B. Right-to-wrong flip taxonomy\n     (E0 \u03b1=8, bridge test set, n=500)",
        fontsize=11,
        fontweight="bold",
        color=TITLE_COLOR,
        loc="left",
        pad=10,
    )

    ax.set_xlim(0, 38.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", labelsize=8.5)
    ax.grid(axis="x", alpha=0.2, zorder=0)


# ---------------------------------------------------------------------------
# Panel C: Substitution examples (text table)
# ---------------------------------------------------------------------------
def draw_panel_c(ax: plt.Axes) -> None:
    """Render wrapped substitution examples without table-cell clipping."""
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.set_title(
        "C. Wrong-entity substitution: swap, not refusal",
        fontsize=11,
        fontweight="bold",
        color=TITLE_COLOR,
        loc="left",
        pad=10,
    )

    col_labels = ["Question", "Baseline (correct)", "ITI \u03b1=8 (wrong)"]
    wrap_widths = [31, 22, 35]
    wrapped_rows = [
        tuple(
            fill(
                text,
                width=width,
                break_long_words=False,
                break_on_hyphens=False,
            )
            for text, width in zip(row, wrap_widths, strict=True)
        )
        for row in SUBSTITUTION_EXAMPLES
    ]

    left = 0.02
    top = 0.92
    table_width = 0.96
    table_height = 0.84
    col_widths = [0.27, 0.22, 0.51]
    col_x = [left]
    for width in col_widths[:-1]:
        col_x.append(col_x[-1] + width * table_width)

    header_units = 1.3
    row_units = []
    for row in wrapped_rows:
        max_lines = max(text.count("\n") + 1 for text in row)
        row_units.append(max_lines + 0.75)

    total_units = header_units + sum(row_units)
    header_height = table_height * (header_units / total_units)
    row_heights = [table_height * (units / total_units) for units in row_units]
    text_padding = 0.018

    # Header row
    y_top = top
    y_bottom = y_top - header_height
    for x, width, label in zip(col_x, col_widths, col_labels, strict=True):
        rect = Rectangle(
            (x, y_bottom),
            width * table_width,
            header_height,
            facecolor=C_TABLE_HEADER_BG,
            edgecolor=C_TABLE_BORDER,
            linewidth=1.0,
            transform=ax.transAxes,
            clip_on=False,
        )
        ax.add_patch(rect)
        ax.text(
            x + text_padding,
            y_bottom + header_height / 2,
            label,
            ha="left",
            va="center",
            fontsize=9.0,
            fontweight="bold",
            color=TITLE_COLOR,
            transform=ax.transAxes,
        )

    # Data rows
    y_top = y_bottom
    for i, (row, row_height) in enumerate(zip(wrapped_rows, row_heights, strict=True)):
        y_bottom = y_top - row_height
        row_bg = BG_COLOR if i % 2 == 0 else C_TABLE_ROW_ALT
        row_colors = [TITLE_COLOR, C_TABLE_CORRECT, C_TABLE_WRONG]

        for x, width, text, color in zip(
            col_x, col_widths, row, row_colors, strict=True
        ):
            rect = Rectangle(
                (x, y_bottom),
                width * table_width,
                row_height,
                facecolor=row_bg,
                edgecolor=C_TABLE_BORDER,
                linewidth=1.0,
                transform=ax.transAxes,
                clip_on=False,
            )
            ax.add_patch(rect)
            ax.text(
                x + text_padding,
                y_bottom + row_height / 2,
                text,
                ha="left",
                va="center",
                fontsize=8.5,
                color=color,
                linespacing=1.2,
                transform=ax.transAxes,
            )

        y_top = y_bottom


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    """Assemble and save the three-panel figure."""
    fig = plt.figure(figsize=(12.4, 6.6), dpi=300)

    # Layout: panels A and B on the top row, Panel C spans the full bottom row.
    gs = gridspec.GridSpec(
        2,
        2,
        figure=fig,
        width_ratios=[1.0, 1.35],
        height_ratios=[0.86, 1.14],
        hspace=0.38,
        wspace=0.35,
    )

    ax_a = fig.add_subplot(gs[0, 0])  # Panel A top-left
    ax_b = fig.add_subplot(gs[0, 1])  # Panel B top-right
    ax_c = fig.add_subplot(gs[1, :])  # Panel C full-width bottom row

    draw_panel_a(ax_a)
    draw_panel_b(ax_b)
    draw_panel_c(ax_c)

    fig.suptitle(
        "Figure 3: Surface-Local Control and Bridge Failure Modes",
        fontsize=13,
        fontweight="bold",
        color=TITLE_COLOR,
        y=1.01,
    )

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
