"""
Figure 1: Four-Stage Interpretability Scaffold.

Conceptual pipeline diagram for "Detection Is Not Enough" showing the four
stages (Measurement, Localization, Control, Externality) and the anchor case
studies that demonstrate where transitions between stages break.

Usage:
    uv run python notes/paper/draft/figures/fig1_four_stage_scaffold.py
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# ---------------------------------------------------------------------------
# Layout constants
# ---------------------------------------------------------------------------
FIG_W, FIG_H = 12, 4.5
DPI = 300
OUTPUT = "notes/paper/draft/figures/fig1_four_stage_scaffold.png"

# Stage definitions (label, subtitle)
STAGES = [
    ("Measurement", "Can we trust\nthe evaluation?"),
    ("Localization", "Where does the\nfeature live?"),
    ("Control", "Can we steer\nbehavior?"),
    ("Externality", "Does it\ntransfer?"),
]

# Anchor case-study text placed below each transition arrow.
# Each string is pre-wrapped for compact rendering.
ANCHORS = [
    "Anchor 3: Truncation, binary vs\ngraded, evaluator dependence",
    (
        "Anchor 1: SAE vs H-neurons matched\n"
        "AUROC, divergent steering; probe\n"
        "AUROC 1.0, null intervention"
    ),
    (
        "Anchor 2: ITI MC +6.3 pp vs bridge\n"
        "\u22127 pp; confident wrong-entity\n"
        "substitution"
    ),
]

# ---------------------------------------------------------------------------
# Color palette — muted blues/grays with warm accent for break points
# ---------------------------------------------------------------------------
BOX_FACE = "#DAEAF6"  # light steel blue
BOX_EDGE = "#3E6A8A"  # dark steel blue
ARROW_COLOR = "#8899A6"  # medium gray-blue
BREAK_COLOR = "#BF4E38"  # muted terracotta accent for break annotations
BREAK_BG = "#FDF1ED"  # very light terracotta tint
TITLE_COLOR = "#1E3044"  # near-black slate
SUBTITLE_COLOR = "#5A6E7F"  # mid-gray blue
BG_COLOR = "#FFFFFF"

# ---------------------------------------------------------------------------
# Font settings
# ---------------------------------------------------------------------------
FONT_FAMILY = "DejaVu Sans"
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": [FONT_FAMILY, "Helvetica", "Arial"],
        "text.color": TITLE_COLOR,
    }
)


def draw_figure():
    """Build and save the four-stage scaffold diagram."""
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), dpi=DPI)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("auto")
    ax.axis("off")
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    # -------------------------------------------------------------------
    # Geometry (axes-fraction coordinates)
    # -------------------------------------------------------------------
    n = len(STAGES)
    box_w = 0.16
    box_h = 0.28

    x_margin = 0.08
    usable = 1.0 - 2 * x_margin
    spacing = usable / (n - 1)
    x_centers = [x_margin + i * spacing for i in range(n)]

    y_center = 0.64  # shifted up to make room for anchor annotations

    # -------------------------------------------------------------------
    # Draw boxes + labels
    # -------------------------------------------------------------------
    for i, (label, subtitle) in enumerate(STAGES):
        xc = x_centers[i]
        x0 = xc - box_w / 2
        y0 = y_center - box_h / 2

        box = FancyBboxPatch(
            (x0, y0),
            box_w,
            box_h,
            boxstyle="round,pad=0.018",
            facecolor=BOX_FACE,
            edgecolor=BOX_EDGE,
            linewidth=2.0,
            zorder=3,
        )
        ax.add_patch(box)

        # Stage number + title
        ax.text(
            xc,
            y_center + 0.045,
            f"{i + 1}. {label}",
            ha="center",
            va="center",
            fontsize=11.5,
            fontweight="bold",
            color=TITLE_COLOR,
            zorder=4,
        )

        # Guiding question
        ax.text(
            xc,
            y_center - 0.065,
            subtitle,
            ha="center",
            va="center",
            fontsize=8.0,
            fontstyle="italic",
            color=SUBTITLE_COLOR,
            zorder=4,
            linespacing=1.25,
        )

    # -------------------------------------------------------------------
    # Draw arrows, break marks, and anchor annotations
    # -------------------------------------------------------------------
    for i in range(n - 1):
        x_start = x_centers[i] + box_w / 2 + 0.010
        x_end = x_centers[i + 1] - box_w / 2 - 0.010
        y_arrow = y_center

        # Main connecting arrow
        arrow = FancyArrowPatch(
            (x_start, y_arrow),
            (x_end, y_arrow),
            arrowstyle="->,head_length=7,head_width=4.5",
            color=ARROW_COLOR,
            linewidth=2.5,
            mutation_scale=1,
            zorder=2,
        )
        ax.add_patch(arrow)

        # "Break" lightning-bolt / X indicator at midpoint
        x_mid = (x_start + x_end) / 2
        ms = 0.014  # marker half-size
        ax.plot(
            [x_mid - ms, x_mid + ms],
            [y_arrow + ms, y_arrow - ms],
            color=BREAK_COLOR,
            linewidth=2.4,
            solid_capstyle="round",
            zorder=6,
        )
        ax.plot(
            [x_mid - ms, x_mid + ms],
            [y_arrow - ms, y_arrow + ms],
            color=BREAK_COLOR,
            linewidth=2.4,
            solid_capstyle="round",
            zorder=6,
        )

        # Thin dashed drop-line from midpoint to annotation
        y_box_bottom = y_center - box_h / 2
        y_anchor_top = y_box_bottom - 0.06
        ax.plot(
            [x_mid, x_mid],
            [y_arrow - ms - 0.005, y_anchor_top + 0.01],
            color=BREAK_COLOR,
            linewidth=0.7,
            linestyle="--",
            alpha=0.5,
            zorder=1,
        )

        # Anchor annotation box
        ax.text(
            x_mid,
            y_anchor_top,
            ANCHORS[i],
            ha="center",
            va="top",
            fontsize=6.5,
            fontstyle="italic",
            color=BREAK_COLOR,
            zorder=4,
            linespacing=1.3,
            bbox=dict(
                boxstyle="round,pad=0.35",
                facecolor=BREAK_BG,
                edgecolor=BREAK_COLOR,
                linewidth=0.7,
                alpha=0.9,
            ),
        )

    # -------------------------------------------------------------------
    # Figure title
    # -------------------------------------------------------------------
    ax.text(
        0.5,
        0.965,
        "The Four-Stage Interpretability Scaffold",
        ha="center",
        va="top",
        fontsize=14,
        fontweight="bold",
        color=TITLE_COLOR,
        zorder=4,
    )

    # -------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------
    fig.savefig(
        OUTPUT,
        dpi=DPI,
        bbox_inches="tight",
        facecolor=BG_COLOR,
        pad_inches=0.15,
    )
    plt.close(fig)
    print(f"Saved: {OUTPUT}")


if __name__ == "__main__":
    draw_figure()
