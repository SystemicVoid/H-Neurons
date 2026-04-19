"""
Figure 1 (draft): Four-Stage Interpretability Scaffold.

Two-row layout: (top) four stage cards + three plain arrows; (bottom) three
audit-gate callouts, each tethered to its anchor. Box-anchored gate =
within-stage break (Measurement -> Verdict); arrow-anchored gates =
inter-stage breaks. Mirrors the ICML version at a larger canvas with
sans-serif body for the draft.

Usage:
    uv run python paper/draft/figures/fig1_four_stage_scaffold.py
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

FIG_W, FIG_H = 12, 5.2
DPI = 300
OUTPUT = "paper/draft/figures/fig1_four_stage_scaffold.png"

# Draft paper section numbering (distinct from ICML's §3/§4/§5).
SEC_LOCALIZATION = "4"
SEC_EXTERNALITY = "5"
SEC_MEASUREMENT = "6"

STAGES = [
    ("Measurement", "Can we trust\nthe evaluation?"),
    ("Localization", "Where is the\nfeature?"),
    ("Control", "Can we steer\nit?"),
    ("Externality", "Does it\ntransfer?"),
]

# Gate = (title, body, anchor_kind, anchor_index, section)
# "box" -> tether lands on stage[anchor_index] bottom edge (within-stage).
# "arrow" -> tether lands on arrow[anchor_index] midpoint (inter-stage).
GATES = [
    (
        "Measurement -> Verdict",
        "truncation, grading, evaluator",
        "box",
        0,
        SEC_MEASUREMENT,
    ),
    (
        "Localization -> Control",
        "SAE vs H-neurons, steering",
        "arrow",
        1,
        SEC_LOCALIZATION,
    ),
    (
        "Control -> Externality",
        "ITI gain vs bridge harm",
        "arrow",
        2,
        SEC_EXTERNALITY,
    ),
]

BOX_FACE = "#EAF2F8"
BOX_EDGE = "#2D4A60"
ARROW_COLOR = "#98A2AB"
BREAK_COLOR = "#A0402E"
BREAK_BG = "#FBEDE6"
BREAK_EDGE = "#D9B5AB"
TITLE_COLOR = "#14202E"
SUBTITLE_COLOR = "#3E5060"
BG_COLOR = "#FFFFFF"

FONT_FAMILY = "DejaVu Sans"
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": [FONT_FAMILY, "Helvetica", "Arial"],
        "text.color": TITLE_COLOR,
    }
)


def draw_figure():
    """Build and save the two-row scaffold diagram."""
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), dpi=DPI)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("auto")
    ax.axis("off")
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    n = len(STAGES)
    box_w = 0.155
    box_h = 0.24
    x_margin = 0.105
    usable = 1.0 - 2 * x_margin
    spacing = usable / (n - 1)
    x_centers = [x_margin + i * spacing for i in range(n)]
    y_stage = 0.76

    for i, (label, subtitle) in enumerate(STAGES):
        xc = x_centers[i]
        box = FancyBboxPatch(
            (xc - box_w / 2, y_stage - box_h / 2),
            box_w,
            box_h,
            boxstyle="round,pad=0.012,rounding_size=0.020",
            facecolor=BOX_FACE,
            edgecolor=BOX_EDGE,
            linewidth=1.4,
            zorder=3,
        )
        ax.add_patch(box)

        ax.text(
            xc,
            y_stage + 0.052,
            f"{i + 1}. {label}",
            ha="center",
            va="center",
            fontsize=13.5,
            fontweight="bold",
            color=TITLE_COLOR,
            zorder=4,
        )
        ax.text(
            xc,
            y_stage - 0.044,
            subtitle,
            ha="center",
            va="center",
            fontsize=10.0,
            color=SUBTITLE_COLOR,
            zorder=4,
            linespacing=1.3,
        )

    arrow_mid_x = []
    for i in range(n - 1):
        x_start = x_centers[i] + box_w / 2 + 0.008
        x_end = x_centers[i + 1] - box_w / 2 - 0.008
        arrow = FancyArrowPatch(
            (x_start, y_stage),
            (x_end, y_stage),
            arrowstyle="->,head_length=7,head_width=4.5",
            color=ARROW_COLOR,
            linewidth=1.8,
            mutation_scale=1,
            zorder=2,
        )
        ax.add_patch(arrow)
        arrow_mid_x.append((x_start + x_end) / 2)

    gate_box_w = 0.215
    gate_box_h = 0.25
    gate_y = 0.24

    y_box_bottom = y_stage - box_h / 2
    anchor_points = []
    gate_x_centers = []
    for _title, _body, anchor_kind, anchor_idx, _section in GATES:
        if anchor_kind == "box":
            anchor_points.append((x_centers[anchor_idx], y_box_bottom))
            half = gate_box_w / 2
            gate_x_centers.append(max(x_centers[anchor_idx], x_margin + half))
        else:
            anchor_points.append((arrow_mid_x[anchor_idx], y_stage))
            gate_x_centers.append(arrow_mid_x[anchor_idx])

    gate_top = gate_y + gate_box_h / 2

    for i, (title, body, _anchor_kind, _anchor_idx, section) in enumerate(GATES):
        xc = gate_x_centers[i]
        gate_box = FancyBboxPatch(
            (xc - gate_box_w / 2, gate_y - gate_box_h / 2),
            gate_box_w,
            gate_box_h,
            boxstyle="round,pad=0.012,rounding_size=0.020",
            facecolor=BREAK_BG,
            edgecolor=BREAK_EDGE,
            linewidth=0.9,
            zorder=3,
        )
        ax.add_patch(gate_box)

        anchor_x, anchor_y = anchor_points[i]
        ax.plot(
            [xc, anchor_x],
            [gate_top + 0.004, anchor_y - 0.010],
            color=BREAK_COLOR,
            linewidth=0.9,
            linestyle=(0, (3.6, 2.4)),
            alpha=0.55,
            zorder=1,
        )
        ax.plot(
            [anchor_x],
            [anchor_y],
            marker="o",
            markersize=5.5,
            markeredgecolor=BREAK_COLOR,
            markerfacecolor=BREAK_COLOR,
            zorder=5,
        )

        ax.text(
            xc,
            gate_y + 0.055,
            f"{title}  (\u00a7{section})",
            ha="center",
            va="center",
            fontsize=11.0,
            fontweight="bold",
            color=BREAK_COLOR,
            zorder=4,
        )
        ax.text(
            xc,
            gate_y - 0.048,
            body,
            ha="center",
            va="center",
            fontsize=10.0,
            color=BREAK_COLOR,
            zorder=4,
            linespacing=1.3,
        )

    ax.text(
        0.5,
        0.965,
        "The Four-Stage Interpretability Scaffold",
        ha="center",
        va="top",
        fontsize=15.5,
        fontweight="bold",
        color=TITLE_COLOR,
        zorder=4,
    )

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
