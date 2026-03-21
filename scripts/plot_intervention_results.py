"""Plot SAE vs neuron intervention results from local JSON files.

Usage:
    uv run python scripts/plot_intervention_results.py
"""

from __future__ import annotations

import json
from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots

DATA = Path("data/gemma3_4b/intervention")
ALPHAS = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]


def load_rates(path: Path) -> list[float]:
    with open(path) as f:
        d = json.load(f)
    return [d["results"][str(a)]["compliance_rate"] * 100 for a in ALPHAS]


def load_ci(path: Path) -> tuple[list[float], list[float]]:
    with open(path) as f:
        d = json.load(f)
    lo = [d["results"][str(a)]["compliance"]["ci"]["lower"] * 100 for a in ALPHAS]
    hi = [d["results"][str(a)]["compliance"]["ci"]["upper"] * 100 for a in ALPHAS]
    return lo, hi


def load_parse_failures(path: Path) -> list[float]:
    with open(path) as f:
        d = json.load(f)
    return [
        d["results"][str(a)]["parse_failures"] / d["results"][str(a)]["n_total"] * 100
        for a in ALPHAS
    ]


def load_comparison(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def main() -> None:
    # --- Load data ---
    neuron_rates = load_rates(DATA / "faitheval/experiment/results.json")
    sae_rates = load_rates(DATA / "faitheval_sae/experiment/results.json")
    sae_lo, sae_hi = load_ci(DATA / "faitheval_sae/experiment/results.json")
    standard_rates = load_rates(DATA / "faitheval_standard/experiment/results.json")
    sae_parse = load_parse_failures(DATA / "faitheval_sae/experiment/results.json")

    comp = load_comparison(DATA / "faitheval_sae/control/comparison_summary.json")
    rand_mean = [r * 100 for r in comp["random_sae_features"]["mean_compliance_rates"]]
    rand_std = [s * 100 for s in comp["random_sae_features"]["std_compliance_rates"]]
    rand_lo = [m - s for m, s in zip(rand_mean, rand_std)]
    rand_hi = [m + s for m, s in zip(rand_mean, rand_std)]

    alpha_labels = [str(a) for a in ALPHAS]

    # --- Build figure ---
    fig = make_subplots(
        rows=2,
        cols=1,
        row_heights=[0.72, 0.28],
        vertical_spacing=0.10,
        subplot_titles=(
            "FaithEval: compliance rate by intervention strength",
            "SAE h-feature parse failure rate",
        ),
    )

    # -- Panel 1: compliance curves --

    # SAE h-features CI band
    fig.add_trace(
        go.Scatter(
            x=alpha_labels,
            y=sae_hi,
            mode="lines",
            line={"width": 0},
            showlegend=False,
            hoverinfo="skip",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=alpha_labels,
            y=sae_lo,
            mode="lines",
            line={"width": 0},
            fill="tonexty",
            fillcolor="rgba(127,119,221,0.15)",
            showlegend=False,
            hoverinfo="skip",
        ),
        row=1,
        col=1,
    )

    # Random SAE ± std band
    fig.add_trace(
        go.Scatter(
            x=alpha_labels,
            y=rand_hi,
            mode="lines",
            line={"width": 0},
            showlegend=False,
            hoverinfo="skip",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=alpha_labels,
            y=rand_lo,
            mode="lines",
            line={"width": 0},
            fill="tonexty",
            fillcolor="rgba(216,90,48,0.12)",
            showlegend=False,
            hoverinfo="skip",
        ),
        row=1,
        col=1,
    )

    # H-neurons (MLP)
    fig.add_trace(
        go.Scatter(
            x=alpha_labels,
            y=neuron_rates,
            mode="lines+markers",
            name="H-neurons (MLP) — slope 2.09 pp/α",
            line={"color": "#1D9E75", "width": 2.5},
            marker={"size": 7},
            hovertemplate="%{y:.1f}%<extra>H-neurons</extra>",
        ),
        row=1,
        col=1,
    )

    # SAE h-features
    fig.add_trace(
        go.Scatter(
            x=alpha_labels,
            y=sae_rates,
            mode="lines+markers",
            name="SAE h-features (266) — slope 0.16 pp/α",
            line={"color": "#7F77DD", "width": 2.5},
            marker={"size": 7},
            hovertemplate="%{y:.1f}%<extra>SAE h-features</extra>",
        ),
        row=1,
        col=1,
    )

    # Random SAE features (mean)
    fig.add_trace(
        go.Scatter(
            x=alpha_labels,
            y=rand_mean,
            mode="lines+markers",
            name="Random SAE features (mean, 3 seeds) — slope 0.59 pp/α",
            line={"color": "#D85A30", "width": 1.5, "dash": "dash"},
            marker={"size": 5},
            hovertemplate="%{y:.1f}%<extra>Random SAE</extra>",
        ),
        row=1,
        col=1,
    )

    # Standard (no classifier)
    fig.add_trace(
        go.Scatter(
            x=alpha_labels,
            y=standard_rates,
            mode="lines+markers",
            name="Standard (no classifier)",
            line={"color": "#888780", "width": 1.5, "dash": "dot"},
            marker={"size": 5},
            hovertemplate="%{y:.1f}%<extra>Standard</extra>",
        ),
        row=1,
        col=1,
    )

    # -- Panel 2: parse failures --
    fig.add_trace(
        go.Bar(
            x=alpha_labels,
            y=sae_parse,
            name="Parse failures",
            marker_color="#BA7517",
            hovertemplate="%{y:.1f}%<extra>Parse failures</extra>",
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    # --- Layout ---
    fig.update_layout(
        template="plotly_dark",
        height=700,
        width=900,
        margin={"t": 60, "b": 60, "l": 70, "r": 30},
        legend={
            "orientation": "h",
            "yanchor": "top",
            "y": -0.08,
            "xanchor": "center",
            "x": 0.5,
            "font": {"size": 11},
        },
        font={"family": "Inter, system-ui, sans-serif"},
    )

    fig.update_xaxes(
        title_text="Scaling factor (α)",
        row=1,
        col=1,
        title_font_size=13,
    )
    fig.update_yaxes(
        title_text="Compliance rate (%)",
        row=1,
        col=1,
        range=[58, 82],
        title_font_size=13,
    )

    fig.update_xaxes(
        title_text="Scaling factor (α)",
        row=2,
        col=1,
        title_font_size=13,
    )
    fig.update_yaxes(
        title_text="Parse failures (%)",
        row=2,
        col=1,
        range=[0, 4],
        title_font_size=13,
    )

    fig.show()


if __name__ == "__main__":
    main()
