"""
Figure 1 (ICML): Four-Stage Interpretability Scaffold.

Two-row layout: (top) four stage cards connected by plain arrows; (bottom)
three audit-gate callouts, each tethered up to its anchor. The tether's
landing encodes the semantics: a gate tethered to a stage box is a
within-stage break; a gate tethered to an arrow midpoint is an inter-stage
break. This disambiguates "Measurement -> Verdict" (within-stage, no
missing fifth stage) from the two genuinely inter-stage transitions.

Usage:
    uv run python paper/icml/figures/fig1_scaffold.py
"""

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

FIG_W, FIG_H = 6.75, 2.05
DPI = 300
OUTPUT = "paper/icml/figures/fig1_scaffold.pdf"

# ICML paper section numbering (Related Work is §2, shifting domains by -1
# relative to the draft).
SEC_LOCALIZATION = "3"
SEC_EXTERNALITY = "4"
SEC_MEASUREMENT = "5"

STAGES = [
    ("Measurement", "Can we trust\nthe evaluation?"),
    ("Localization", "Where is the\nfeature?"),
    ("Control", "Can we steer\nit?"),
    ("Externality", "Does it\ntransfer?"),
]

# Gate = (title, body, anchor_kind, anchor_index, section)
# anchor_kind == "box": tether lands on bottom edge of stage[anchor_index]
# anchor_kind == "arrow": tether lands on midpoint of arrow[anchor_index]
# The first entry is deliberately "box" to encode within-stage semantics.
GATES = [
    (
        r"Meas.$\to$Verdict",
        "truncation, grading,\nevaluator",
        "box",
        0,
        SEC_MEASUREMENT,
    ),
    (
        r"Loc.$\to$Control",
        "SAE vs H-neurons,\nsteering",
        "arrow",
        1,
        SEC_LOCALIZATION,
    ),
    (
        r"Ctrl.$\to$Externality",
        "ITI gain vs bridge\nharm",
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

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times", "Times New Roman"],
        "text.usetex": True,
        "text.color": TITLE_COLOR,
        "font.size": 8,
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
    box_w = 0.165
    box_h = 0.26
    x_margin = 0.105
    usable = 1.0 - 2 * x_margin
    spacing = usable / (n - 1)
    x_centers = [x_margin + i * spacing for i in range(n)]
    y_stage = 0.77

    for i, (label, subtitle) in enumerate(STAGES):
        xc = x_centers[i]
        box = FancyBboxPatch(
            (xc - box_w / 2, y_stage - box_h / 2),
            box_w,
            box_h,
            boxstyle="round,pad=0.010,rounding_size=0.018",
            facecolor=BOX_FACE,
            edgecolor=BOX_EDGE,
            linewidth=0.9,
            zorder=3,
        )
        ax.add_patch(box)
        ax.text(
            xc,
            y_stage + 0.058,
            rf"\textbf{{{i + 1}.\ {label}}}",
            ha="center",
            va="center",
            fontsize=8.0,
            color=TITLE_COLOR,
            zorder=4,
        )
        ax.text(
            xc,
            y_stage - 0.050,
            subtitle,
            ha="center",
            va="center",
            fontsize=6.5,
            color=SUBTITLE_COLOR,
            zorder=4,
            linespacing=1.25,
        )

    arrow_mid_x = []
    for i in range(n - 1):
        x_start = x_centers[i] + box_w / 2 + 0.008
        x_end = x_centers[i + 1] - box_w / 2 - 0.008
        arrow = FancyArrowPatch(
            (x_start, y_stage),
            (x_end, y_stage),
            arrowstyle="->,head_length=5,head_width=3",
            color=ARROW_COLOR,
            linewidth=1.1,
            mutation_scale=1,
            zorder=2,
        )
        ax.add_patch(arrow)
        arrow_mid_x.append((x_start + x_end) / 2)

    gate_box_w = 0.225
    gate_box_h = 0.27
    gate_y = 0.235

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
            boxstyle="round,pad=0.010,rounding_size=0.018",
            facecolor=BREAK_BG,
            edgecolor=BREAK_EDGE,
            linewidth=0.5,
            zorder=3,
        )
        ax.add_patch(gate_box)

        anchor_x, anchor_y = anchor_points[i]
        ax.plot(
            [xc, anchor_x],
            [gate_top + 0.002, anchor_y - 0.006],
            color=BREAK_COLOR,
            linewidth=0.55,
            linestyle=(0, (2.4, 1.8)),
            alpha=0.55,
            zorder=1,
        )
        ax.plot(
            [anchor_x],
            [anchor_y],
            marker="o",
            markersize=2.2,
            markeredgecolor=BREAK_COLOR,
            markerfacecolor=BREAK_COLOR,
            zorder=5,
        )

        ax.text(
            xc,
            gate_y + 0.060,
            rf"\textbf{{{title}}}\ \ ($\S${section})",
            ha="center",
            va="center",
            fontsize=7.0,
            color=BREAK_COLOR,
            zorder=4,
        )
        ax.text(
            xc,
            gate_y - 0.055,
            body,
            ha="center",
            va="center",
            fontsize=6.2,
            color=BREAK_COLOR,
            zorder=4,
            linespacing=1.3,
        )

    fig.savefig(
        OUTPUT,
        dpi=DPI,
        bbox_inches="tight",
        facecolor=BG_COLOR,
        pad_inches=0.04,
    )
    plt.close(fig)
    print(f"Saved: {OUTPUT}")


if __name__ == "__main__":
    draw_figure()
