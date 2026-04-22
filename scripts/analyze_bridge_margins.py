"""Aggregate per-case bridge-margin records into cohort summaries.

Reads ``margins.jsonl`` produced by ``score_bridge_margins.py`` and
computes for each cohort (A/B/C/D) a bootstrap-CI summary of the gold vs.
wrong-entity log-likelihood margin shift under the ITI intervention, plus
between-cohort comparisons (A vs. B, A vs. D).

**Confirmatory primary window: first-3-token log-probability.** This was
locked before the full run; the post-run analysis and the sealed
precommit decision tree both live at
``paper/icml/reports/2026-04-21-bridge-margin-report.md`` (Appendix A
preserves the precommit verbatim). Do not change the endpoint post-hoc
to chase the numbers.

**Sensitivity (locked secondary):** full-continuation joint log-probability.

**Mandatory diagnostic:** tokenwise decomposition at positions 0, 1, 2
(the three positions inside the ITI ``first_3_tokens`` scope) so the
reader can see where inside the manipulated prefix the shift lives.

**Exploratory sensitivity:** tokens-2-3 window (positions 1 and 2,
0-indexed). Still fully inside the hooked window, drops the very first
token which may index answer-frame initiation rather than entity
commitment.

**Exploratory subgroup:** within cohort A, split by the sign of the
baseline first-3 delta (``delta_base > 0`` vs. ``<= 0``). Distinguishes
clean ITI-induced reversal from amplification of an already shaky
baseline preference.

Outputs:

  * ``results.json`` — per-cohort + between-cohort summaries.
  * ``margin_shift_by_cohort.png`` and ``.pdf`` — strip plot.
  * stdout — human-readable text report.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from uncertainty import percentile_interval  # noqa: E402
from utils import format_path_for_metadata, sanitize_run_config  # noqa: E402

COHORTS = (
    "A_rw_substitution",
    "B_rw_nonsubstitution",
    "C_wr_rescue",
    "D_random_control",
)
COHORT_LABELS = {
    "A_rw_substitution": "A. R→W substitution",
    "B_rw_nonsubstitution": "B. R→W non-substitution",
    "C_wr_rescue": "C. W→R rescue",
    "D_random_control": "D. Random wrong-entity control",
}
DEFAULT_N_RESAMPLES = 10_000
DEFAULT_SEED = 42


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def git_head_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def bootstrap_mean_ci(
    values: np.ndarray,
    *,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    seed: int = DEFAULT_SEED,
    confidence: float = 0.95,
) -> dict[str, Any]:
    values = np.asarray(values, dtype=float)
    n = values.size
    if n == 0:
        return {
            "estimate": 0.0,
            "n": 0,
            "ci": {
                "lower": 0.0,
                "upper": 0.0,
                "level": confidence,
                "method": "bootstrap_percentile_within_cohort",
            },
        }
    rng = np.random.default_rng(seed)
    samples = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        idx = rng.choice(n, size=n, replace=True)
        samples[i] = values[idx].mean()
    interval = percentile_interval(
        samples, confidence, method="bootstrap_percentile_within_cohort"
    )
    return {
        "estimate": float(values.mean()),
        "n": int(n),
        "ci": interval.to_dict(),
        "bootstrap": {
            "n_resamples": int(n_resamples),
            "seed": int(seed),
            "confidence": float(confidence),
        },
    }


def unpaired_bootstrap_mean_difference(
    a: np.ndarray,
    b: np.ndarray,
    *,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    seed: int = DEFAULT_SEED,
    permutation_resamples: int = DEFAULT_N_RESAMPLES,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """CI on ``mean(a) - mean(b)`` via independent resampling of each group,
    plus a one-sided permutation test for ``mean(a) < mean(b)``."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size == 0 or b.size == 0:
        return {
            "estimate": 0.0,
            "n_a": int(a.size),
            "n_b": int(b.size),
            "ci": {
                "lower": 0.0,
                "upper": 0.0,
                "level": confidence,
                "method": "bootstrap_percentile_unpaired",
            },
        }
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        ia = rng.choice(a.size, size=a.size, replace=True)
        ib = rng.choice(b.size, size=b.size, replace=True)
        diffs[i] = a[ia].mean() - b[ib].mean()
    ci = percentile_interval(diffs, confidence, method="bootstrap_percentile_unpaired")

    # One-sided permutation test: H1: mean(a) < mean(b), i.e. diff < observed.
    observed = float(a.mean() - b.mean())
    combined = np.concatenate([a, b])
    perm_rng = np.random.default_rng(seed + 1)
    n_extreme = 0
    n_a = a.size
    for _ in range(permutation_resamples):
        perm = perm_rng.permutation(combined)
        perm_diff = perm[:n_a].mean() - perm[n_a:].mean()
        if perm_diff <= observed:
            n_extreme += 1
    p_value = (n_extreme + 1) / (permutation_resamples + 1)

    return {
        "estimate": observed,
        "n_a": int(a.size),
        "n_b": int(b.size),
        "ci": ci.to_dict(),
        "bootstrap": {
            "n_resamples": int(n_resamples),
            "seed": int(seed),
            "confidence": float(confidence),
            "resampling": "independent_groups",
        },
        "permutation_test": {
            "p_value": float(p_value),
            "n_extreme": int(n_extreme),
            "n_permutations": int(permutation_resamples),
            "alternative": "one_sided_less",
            "null": "equal_means",
        },
    }


def summarize_cohort(
    records: list[dict[str, Any]],
    *,
    window: str,
    metric: str,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    values = np.array([float(r[window][metric]) for r in records], dtype=float)
    return bootstrap_mean_ci(values, n_resamples=n_resamples, seed=seed)


def per_position_shift(record: dict[str, Any], position: int) -> float | None:
    """Shift at a single 0-indexed continuation-token position.

    ``(ell_iti_gold - ell_iti_wrong) - (ell_base_gold - ell_base_wrong)``
    at one position. Returns ``None`` if the record lacks per-token data
    or if either target is shorter than ``position + 1`` tokens (e.g. a
    2-token gold alias has no position-2 entry).
    """
    try:
        bg = record["base_gold"]["per_token_logprobs"]
        bw = record["base_wrong"]["per_token_logprobs"]
        ig = record["iti_gold"]["per_token_logprobs"]
        iw = record["iti_wrong"]["per_token_logprobs"]
    except KeyError:
        return None
    if not (
        len(bg) > position
        and len(bw) > position
        and len(ig) > position
        and len(iw) > position
    ):
        return None
    return (float(ig[position]) - float(iw[position])) - (
        float(bg[position]) - float(bw[position])
    )


def tokens_2_3_shift(record: dict[str, Any]) -> float | None:
    """Sum of shifts at positions 1 and 2 (0-indexed) — the 2nd + 3rd
    continuation tokens. Both remain inside the ITI ``first_3_tokens``
    hook scope. Returns ``None`` if either target is shorter than 3 tokens.
    """
    s1 = per_position_shift(record, 1)
    s2 = per_position_shift(record, 2)
    if s1 is None or s2 is None:
        return None
    return s1 + s2


def tokenwise_cohort_summary(
    records: list[dict[str, Any]],
    *,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Bootstrap CI for per-position shifts (positions 0, 1, 2) and for
    the exploratory tokens-2-3 window."""
    summary: dict[str, Any] = {}
    for position in (0, 1, 2):
        values = np.array(
            [
                v
                for v in (per_position_shift(r, position) for r in records)
                if v is not None
            ],
            dtype=float,
        )
        summary[f"position_{position}"] = (
            bootstrap_mean_ci(values, n_resamples=n_resamples, seed=seed)
            if values.size
            else None
        )
    pair_values = np.array(
        [v for v in (tokens_2_3_shift(r) for r in records) if v is not None],
        dtype=float,
    )
    summary["tokens_2_3"] = (
        bootstrap_mean_ci(pair_values, n_resamples=n_resamples, seed=seed)
        if pair_values.size
        else None
    )
    return summary


def within_a_baseline_sign_split(
    by_cohort: dict[str, list[dict[str, Any]]],
    *,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Split cohort A by the sign of ``first3.delta_base``.

    ``A_pos`` = baseline already prefers gold (clean ITI-induced reversal).
    ``A_neg`` = baseline already ambivalent or wrong-leaning (ITI
    amplifies a shaky baseline).

    Both subgroups are reported with first-3 shift-nats and the
    tokens-2-3 exploratory window, each with bootstrap mean + 95% CI.
    """
    a_rows = by_cohort.get("A_rw_substitution", [])
    a_pos = [r for r in a_rows if float(r["first3"]["delta_base"]) > 0]
    a_neg = [r for r in a_rows if float(r["first3"]["delta_base"]) <= 0]
    out: dict[str, Any] = {}
    for name, rows in (
        ("A_pos_baseline_prefers_gold", a_pos),
        ("A_neg_baseline_ambivalent", a_neg),
    ):
        entry: dict[str, Any] = {"n": len(rows), "label": name}
        if not rows:
            entry["first3_shift_nats"] = None
            entry["tokens_2_3_shift"] = None
            out[name] = entry
            continue
        first3 = np.array([float(r["first3"]["shift_nats"]) for r in rows])
        entry["first3_shift_nats"] = bootstrap_mean_ci(
            first3, n_resamples=n_resamples, seed=seed
        )
        pair = np.array(
            [v for v in (tokens_2_3_shift(r) for r in rows) if v is not None],
            dtype=float,
        )
        entry["tokens_2_3_shift"] = (
            bootstrap_mean_ci(pair, n_resamples=n_resamples, seed=seed)
            if pair.size
            else None
        )
        out[name] = entry
    return out


def collect_by_cohort(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        out[str(r["cohort"])].append(r)
    return dict(out)


def cohort_table(
    by_cohort: dict[str, list[dict[str, Any]]],
    *,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for name in COHORTS:
        rows = by_cohort.get(name, [])
        cohort_summary: dict[str, Any] = {
            "n": len(rows),
            "label": COHORT_LABELS[name],
        }
        for window in ("first3", "full"):
            cohort_summary[window] = {
                "shift_nats": summarize_cohort(
                    rows,
                    window=window,
                    metric="shift_nats",
                    n_resamples=n_resamples,
                    seed=seed,
                )
                if rows
                else None,
                "shift_per_token": summarize_cohort(
                    rows,
                    window=window,
                    metric="shift_per_token",
                    n_resamples=n_resamples,
                    seed=seed,
                )
                if rows
                else None,
                "delta_base_mean": float(
                    np.mean([r[window]["delta_base"] for r in rows])
                )
                if rows
                else None,
                "delta_iti_mean": float(np.mean([r[window]["delta_iti"] for r in rows]))
                if rows
                else None,
            }
        cohort_summary["tokenwise"] = tokenwise_cohort_summary(
            rows, n_resamples=n_resamples, seed=seed
        )
        summary[name] = cohort_summary
    return summary


def between_cohort_tests(
    by_cohort: dict[str, list[dict[str, Any]]],
    *,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for cohort_b in ("B_rw_nonsubstitution", "D_random_control"):
        pair_key = f"A_vs_{cohort_b.split('_')[0]}"
        a_rows = by_cohort.get("A_rw_substitution", [])
        b_rows = by_cohort.get(cohort_b, [])
        if not a_rows or not b_rows:
            continue
        pair_result: dict[str, Any] = {
            "group_a": "A_rw_substitution",
            "group_b": cohort_b,
            "n_a": len(a_rows),
            "n_b": len(b_rows),
        }
        for window in ("first3", "full"):
            for metric in ("shift_nats", "shift_per_token"):
                a_vals = np.array(
                    [float(r[window][metric]) for r in a_rows], dtype=float
                )
                b_vals = np.array(
                    [float(r[window][metric]) for r in b_rows], dtype=float
                )
                pair_result.setdefault(window, {})[metric] = (
                    unpaired_bootstrap_mean_difference(
                        a_vals,
                        b_vals,
                        n_resamples=n_resamples,
                        seed=seed,
                        permutation_resamples=n_resamples,
                    )
                )
        # Exploratory tokens-2-3 window (positions 1+2, 0-indexed): both
        # still inside the ITI hook scope, excludes the first token which
        # may index answer-frame initiation rather than entity commitment.
        a_23 = np.array(
            [v for v in (tokens_2_3_shift(r) for r in a_rows) if v is not None],
            dtype=float,
        )
        b_23 = np.array(
            [v for v in (tokens_2_3_shift(r) for r in b_rows) if v is not None],
            dtype=float,
        )
        if a_23.size and b_23.size:
            pair_result["tokens_2_3"] = {
                "shift_nats": unpaired_bootstrap_mean_difference(
                    a_23,
                    b_23,
                    n_resamples=n_resamples,
                    seed=seed,
                    permutation_resamples=n_resamples,
                )
            }
        results[pair_key] = pair_result
    return results


def sign_check_baseline(by_cohort: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """Baseline delta should be positive on R→W cohorts (gold preferred).

    Returns a dict of per-cohort baseline delta means, so the reviewer can
    confirm the sign convention held before trusting the shift interpretation.
    """
    check: dict[str, Any] = {}
    for name in ("A_rw_substitution", "B_rw_nonsubstitution"):
        rows = by_cohort.get(name, [])
        if not rows:
            continue
        deltas = np.array([float(r["first3"]["delta_base"]) for r in rows])
        check[name] = {
            "n": len(rows),
            "first3_delta_base_mean": float(deltas.mean()),
            "first3_delta_base_positive_fraction": float((deltas > 0).mean()),
        }
    return check


def _draw_cohort_strip(
    ax: Any,
    *,
    present_cohorts: list[str],
    values_by_cohort: dict[str, np.ndarray],
    ci_by_cohort: dict[str, dict[str, Any] | None],
    rng: np.random.Generator,
    x_label: str,
) -> None:
    for y_pos, cohort_name in enumerate(present_cohorts):
        values = values_by_cohort.get(cohort_name, np.array([], dtype=float))
        if values.size:
            jitter = rng.uniform(-0.12, 0.12, size=values.size)
            ax.scatter(
                values,
                np.full_like(values, y_pos, dtype=float) + jitter,
                s=28,
                alpha=0.45,
                color="#444",
            )
        ci = ci_by_cohort.get(cohort_name)
        if ci is not None:
            est = ci["estimate"]
            lo = ci["ci"]["lower"]
            hi = ci["ci"]["upper"]
            ax.errorbar(
                est,
                y_pos,
                xerr=[[est - lo], [hi - est]],
                fmt="o",
                color="#c03020",
                capsize=5,
                markersize=8,
                elinewidth=1.8,
                zorder=5,
            )
            ax.text(
                hi + 0.05,
                y_pos,
                f" mean={est:+.2f} [{lo:+.2f}, {hi:+.2f}], n={ci['n']}",
                va="center",
                fontsize=8,
            )
    ax.axvline(0.0, color="#888", linestyle="--", linewidth=0.8)
    ax.set_yticks(range(len(present_cohorts)))
    ax.set_xlabel(x_label)
    ax.grid(axis="x", alpha=0.25)


def make_strip_plot(
    by_cohort: dict[str, list[dict[str, Any]]],
    summary: dict[str, dict[str, Any]],
    *,
    window: str,
    metric: str,
    title: str,
    out_path_png: Path,
    out_path_pdf: Path,
    seed: int = DEFAULT_SEED,
) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    present_cohorts = [name for name in COHORTS if by_cohort.get(name)]
    values_by_cohort = {
        name: np.array([float(r[window][metric]) for r in by_cohort[name]], dtype=float)
        for name in present_cohorts
    }
    ci_by_cohort: dict[str, dict[str, Any] | None] = {
        name: summary[name][window][metric] for name in present_cohorts
    }
    _draw_cohort_strip(
        ax,
        present_cohorts=present_cohorts,
        values_by_cohort=values_by_cohort,
        ci_by_cohort=ci_by_cohort,
        rng=np.random.default_rng(seed),
        x_label=f"{metric} ({window} window, nats)",
    )
    ax.set_yticklabels([COHORT_LABELS[c] for c in present_cohorts])
    ax.invert_yaxis()
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path_png, dpi=160)
    fig.savefig(out_path_pdf)
    plt.close(fig)


def make_position_panels_plot(
    by_cohort: dict[str, list[dict[str, Any]]],
    summary: dict[str, dict[str, Any]],
    *,
    title: str,
    out_path_png: Path,
    out_path_pdf: Path,
    seed: int = DEFAULT_SEED,
) -> None:
    """Two-panel figure separating position-0 (answer-frame) from the
    tokens-2-3 content window. Reveals that B's larger aggregate shift is
    concentrated but not exclusive at position 0 — the finding §8 calls out.
    """
    present_cohorts = [name for name in COHORTS if by_cohort.get(name)]
    fig, axes = plt.subplots(
        1, 2, figsize=(11.5, 4.2), sharey=True, constrained_layout=True
    )
    panel_specs = (
        (
            axes[0],
            "position_0",
            "position 0 (first continuation token)",
            lambda r: per_position_shift(r, 0),
        ),
        (
            axes[1],
            "tokens_2_3",
            "tokens 2–3 (positions 1 + 2 summed per case)",
            tokens_2_3_shift,
        ),
    )
    rng = np.random.default_rng(seed)
    for ax, summary_key, panel_title, per_case in panel_specs:
        values_by_cohort = {
            name: np.array(
                [v for v in (per_case(r) for r in by_cohort[name]) if v is not None],
                dtype=float,
            )
            for name in present_cohorts
        }
        ci_by_cohort: dict[str, dict[str, Any] | None] = {
            name: summary[name]["tokenwise"].get(summary_key)
            for name in present_cohorts
        }
        _draw_cohort_strip(
            ax,
            present_cohorts=present_cohorts,
            values_by_cohort=values_by_cohort,
            ci_by_cohort=ci_by_cohort,
            rng=rng,
            x_label="shift_nats",
        )
        ax.set_title(panel_title)
    axes[0].set_yticklabels([COHORT_LABELS[c] for c in present_cohorts])
    axes[0].invert_yaxis()
    fig.suptitle(title)
    fig.savefig(out_path_png, dpi=160)
    fig.savefig(out_path_pdf)
    plt.close(fig)


def _fmt_ci(ci: dict[str, Any] | None) -> str:
    if ci is None:
        return "n/a"
    return f"{ci['estimate']:+.3f} [{ci['ci']['lower']:+.3f}, {ci['ci']['upper']:+.3f}]"


def format_text_report(
    summary: dict[str, dict[str, Any]],
    pairs: dict[str, dict[str, Any]],
    sign_check: dict[str, Any],
    within_a: dict[str, Any],
) -> str:
    lines: list[str] = []
    lines.append("=" * 72)
    lines.append("Bridge logprob-margin mechanism check — summary")
    lines.append("=" * 72)
    lines.append("")
    lines.append("CONFIRMATORY primary: first-3-token shift_nats (see precommit).")
    lines.append("")

    lines.append(
        "Cohort | n | shift_nats (first3) [95% CI] | shift_per_token (first3) [95% CI]"
    )
    lines.append("-" * 72)
    for name in COHORTS:
        row = summary.get(name)
        if row is None or row["n"] == 0:
            lines.append(f"{name:30s} | empty")
            continue
        nats = row["first3"]["shift_nats"]
        perk = row["first3"]["shift_per_token"]
        lines.append(f"{name:30s} | {row['n']:3d} | {_fmt_ci(nats)} | {_fmt_ci(perk)}")
    lines.append("")

    lines.append("Baseline sign check (expect delta_base > 0 on R→W cohorts):")
    for name, check in sign_check.items():
        lines.append(
            f"  {name}: n={check['n']}  "
            f"delta_base_mean(first3)={check['first3_delta_base_mean']:+.3f}  "
            f"fraction>0={check['first3_delta_base_positive_fraction']:.2f}"
        )
    lines.append("")

    lines.append(
        "Between-cohort (one-sided H1: mean(A) < mean(B/D), i.e. A more negative):"
    )
    for pair_name, pair in pairs.items():
        diff = pair["first3"]["shift_nats"]
        lines.append(
            f"  {pair_name} first3 (n_a={pair['n_a']}, n_b={pair['n_b']}): "
            f"mean(A)-mean(B) = {_fmt_ci(diff)}  "
            f"p={diff['permutation_test']['p_value']:.4f}"
        )
        if "tokens_2_3" in pair:
            t23 = pair["tokens_2_3"]["shift_nats"]
            lines.append(
                f"  {pair_name} tokens-2-3: "
                f"mean(A)-mean(B) = {_fmt_ci(t23)}  "
                f"p={t23['permutation_test']['p_value']:.4f}"
            )
    lines.append("")

    lines.append("DIAGNOSTIC: tokenwise decomposition (positions 0,1,2; 0-indexed):")
    for name in COHORTS:
        row = summary.get(name)
        if row is None or row["n"] == 0:
            continue
        tok = row.get("tokenwise", {})
        p0 = tok.get("position_0")
        p1 = tok.get("position_1")
        p2 = tok.get("position_2")
        t23 = tok.get("tokens_2_3")
        lines.append(
            f"  {name:30s} | n={row['n']:3d} | "
            f"pos0={_fmt_ci(p0)}  pos1={_fmt_ci(p1)}  pos2={_fmt_ci(p2)}  "
            f"tok2_3={_fmt_ci(t23)}"
        )
    lines.append("")

    lines.append(
        "EXPLORATORY within-A split by baseline sign (A_pos = baseline prefers "
        "gold, A_neg = baseline ambivalent/wrong-leaning):"
    )
    for name, entry in within_a.items():
        if entry["n"] == 0:
            lines.append(f"  {name}: n=0")
            continue
        lines.append(
            f"  {name}: n={entry['n']:2d}  "
            f"first3_shift_nats={_fmt_ci(entry['first3_shift_nats'])}  "
            f"tok2_3={_fmt_ci(entry.get('tokens_2_3_shift'))}"
        )
    lines.append("")

    lines.append("SENSITIVITY (full-continuation window):")
    for name in COHORTS:
        row = summary.get(name)
        if row is None or row["n"] == 0:
            continue
        nats = row["full"]["shift_nats"]
        lines.append(
            f"  {name:30s} | n={row['n']:3d} | shift_nats(full)={_fmt_ci(nats)}"
        )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "data/gemma3_4b/analysis/bridge_margins/test/margins.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data/gemma3_4b/analysis/bridge_margins/test",
    )
    parser.add_argument("--n-resamples", type=int, default=DEFAULT_N_RESAMPLES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.input.exists():
        print(f"Input not found: {args.input}", file=sys.stderr)
        return 1

    records = load_jsonl(args.input)
    if not records:
        print("No records in input file", file=sys.stderr)
        return 1

    by_cohort = collect_by_cohort(records)
    summary = cohort_table(by_cohort, n_resamples=args.n_resamples, seed=args.seed)
    pairs = between_cohort_tests(
        by_cohort, n_resamples=args.n_resamples, seed=args.seed
    )
    sign_check = sign_check_baseline(by_cohort)
    within_a = within_a_baseline_sign_split(
        by_cohort, n_resamples=args.n_resamples, seed=args.seed
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    png_path_f3 = args.output_dir / "margin_shift_by_cohort_first3.png"
    pdf_path_f3 = args.output_dir / "margin_shift_by_cohort_first3.pdf"
    make_strip_plot(
        by_cohort,
        summary,
        window="first3",
        metric="shift_nats",
        title="ITI margin shift on bridge discordant cases (first-3 tokens, nats)",
        out_path_png=png_path_f3,
        out_path_pdf=pdf_path_f3,
        seed=args.seed,
    )
    png_path_full = args.output_dir / "margin_shift_by_cohort_full.png"
    pdf_path_full = args.output_dir / "margin_shift_by_cohort_full.pdf"
    make_strip_plot(
        by_cohort,
        summary,
        window="full",
        metric="shift_nats",
        title="ITI margin shift on bridge discordant cases (full continuation, nats)",
        out_path_png=png_path_full,
        out_path_pdf=pdf_path_full,
        seed=args.seed,
    )
    png_path_pos = args.output_dir / "margin_shift_by_position_panels.png"
    pdf_path_pos = args.output_dir / "margin_shift_by_position_panels.pdf"
    make_position_panels_plot(
        by_cohort,
        summary,
        title=(
            "ITI margin shift decomposed: answer-frame token vs. "
            "content tokens (bridge discordant cases, nats)"
        ),
        out_path_png=png_path_pos,
        out_path_pdf=pdf_path_pos,
        seed=args.seed,
    )

    results = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_head_commit(),
        "n_records": len(records),
        "cohort_sizes": {name: len(by_cohort.get(name, [])) for name in COHORTS},
        "cohorts": summary,
        "between_cohort": pairs,
        "baseline_sign_check": sign_check,
        "within_A_baseline_sign": within_a,
        "bootstrap": {
            "n_resamples": int(args.n_resamples),
            "seed": int(args.seed),
            "confidence": 0.95,
        },
        "figures": {
            "first3": {
                "png": format_path_for_metadata(png_path_f3, root=ROOT),
                "pdf": format_path_for_metadata(pdf_path_f3, root=ROOT),
            },
            "full": {
                "png": format_path_for_metadata(png_path_full, root=ROOT),
                "pdf": format_path_for_metadata(pdf_path_full, root=ROOT),
            },
            "position_panels": {
                "png": format_path_for_metadata(png_path_pos, root=ROOT),
                "pdf": format_path_for_metadata(pdf_path_pos, root=ROOT),
            },
        },
    }
    results_path = args.output_dir / "results.json"
    results_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    report = format_text_report(summary, pairs, sign_check, within_a)
    print(report)

    prov_path = (
        args.output_dir
        / f"analyze_bridge_margins.provenance.{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    )
    prov_path.write_text(
        json.dumps(
            {
                "script": "scripts/analyze_bridge_margins.py",
                "timestamp_utc": datetime.now(timezone.utc).isoformat(
                    timespec="seconds"
                ),
                "git_commit": git_head_commit(),
                "args": sanitize_run_config(vars(args)),
                "input_path": format_path_for_metadata(args.input, root=ROOT),
                "results_path": format_path_for_metadata(results_path, root=ROOT),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
