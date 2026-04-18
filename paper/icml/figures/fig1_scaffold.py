"""
Figure 1 (workshop): Four-Stage Interpretability Scaffold.

Adapted from paper/draft/figures/fig1_four_stage_scaffold.py for ICML
two-column format (figure* full-width, 6.75in x 1.6in, PDF output).

Usage:
    uv run python paper/icml/figures/fig1_scaffold.py
"""

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

FIG_W, FIG_H = 6.75, 1.6
DPI = 300
OUTPUT = "paper/icml/figures/fig1_scaffold.pdf"

STAGES = [
    ("Measurement", "Can we trust\nthe evaluation?"),
    ("Localization", "Where is the\nfeature?"),
    ("Control", "Can we steer\nit?"),
    ("Externality", "Does it\ntransfer?"),
]

ANCHORS = [
    r"Meas.$\to$Verdict" + "\ntruncation, grading,\n" + r"evaluator ($\S$6)",
    r"Loc.$\to$Control" + "\nSAE vs H-neurons\n" + r"steering ($\S$4)",
    r"Ctrl.$\to$Externality" + "\nITI gain vs bridge\n" + r"harm ($\S$5)",
]

BOX_FACE = "#DAEAF6"
BOX_EDGE = "#3E6A8A"
ARROW_COLOR = "#8899A6"
BREAK_COLOR = "#BF4E38"
BREAK_BG = "#FDF1ED"
TITLE_COLOR = "#1E3044"
SUBTITLE_COLOR = "#5A6E7F"
BG_COLOR = "#FFFFFF"

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times", "Times New Roman"],
        "text.usetex": True,
        "text.color": TITLE_COLOR,
        "font.size": 7,
    }
)


def draw_figure():
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), dpi=DPI)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("auto")
    ax.axis("off")
    fig.patch.set_facecolor(BG_COLOR)

    n = len(STAGES)
    box_w = 0.15
    box_h = 0.38
    x_margin = 0.08
    usable = 1.0 - 2 * x_margin
    spacing = usable / (n - 1)
    x_centers = [x_margin + i * spacing for i in range(n)]
    y_center = 0.62

    for i, (label, subtitle) in enumerate(STAGES):
        xc = x_centers[i]
        box = FancyBboxPatch(
            (xc - box_w / 2, y_center - box_h / 2),
            box_w,
            box_h,
            boxstyle="round,pad=0.015",
            facecolor=BOX_FACE,
            edgecolor=BOX_EDGE,
            linewidth=1.2,
            zorder=3,
        )
        ax.add_patch(box)
        ax.text(
            xc,
            y_center + 0.055,
            label,
            ha="center",
            va="center",
            fontsize=7.5,
            fontweight="bold",
            color=TITLE_COLOR,
            zorder=4,
        )
        ax.text(
            xc,
            y_center - 0.075,
            subtitle,
            ha="center",
            va="center",
            fontsize=5.5,
            fontstyle="italic",
            color=SUBTITLE_COLOR,
            zorder=4,
            linespacing=1.2,
        )

    for i in range(n - 1):
        x_start = x_centers[i] + box_w / 2 + 0.008
        x_end = x_centers[i + 1] - box_w / 2 - 0.008
        y_arrow = y_center
        arrow = FancyArrowPatch(
            (x_start, y_arrow),
            (x_end, y_arrow),
            arrowstyle="->,head_length=5,head_width=3",
            color=ARROW_COLOR,
            linewidth=1.5,
            mutation_scale=1,
            zorder=2,
        )
        ax.add_patch(arrow)

        x_mid = (x_start + x_end) / 2
        ms = 0.012
        ax.plot(
            [x_mid - ms, x_mid + ms],
            [y_arrow + ms, y_arrow - ms],
            color=BREAK_COLOR,
            linewidth=1.5,
            solid_capstyle="round",
            zorder=6,
        )
        ax.plot(
            [x_mid - ms, x_mid + ms],
            [y_arrow - ms, y_arrow + ms],
            color=BREAK_COLOR,
            linewidth=1.5,
            solid_capstyle="round",
            zorder=6,
        )

        y_box_bottom = y_center - box_h / 2
        y_anchor_top = y_box_bottom - 0.06
        ax.plot(
            [x_mid, x_mid],
            [y_arrow - ms - 0.005, y_anchor_top + 0.01],
            color=BREAK_COLOR,
            linewidth=0.5,
            linestyle="--",
            alpha=0.5,
            zorder=1,
        )
        ax.text(
            x_mid,
            y_anchor_top,
            ANCHORS[i],
            ha="center",
            va="top",
            fontsize=5.0,
            fontstyle="italic",
            color=BREAK_COLOR,
            zorder=4,
            linespacing=1.2,
            bbox=dict(
                boxstyle="round,pad=0.2",
                facecolor=BREAK_BG,
                edgecolor=BREAK_COLOR,
                linewidth=0.5,
                alpha=0.9,
            ),
        )

    fig.savefig(
        OUTPUT, dpi=DPI, bbox_inches="tight", facecolor=BG_COLOR, pad_inches=0.02
    )
    plt.close(fig)
    print(f"Saved: {OUTPUT}")


if __name__ == "__main__":
    draw_figure()
