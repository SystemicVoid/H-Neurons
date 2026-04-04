#!/usr/bin/env python3
"""Build E2-B diagnostic report artifacts (JSON + Markdown).

Compares E2-B (TriviaQA family-default selectors: AUROC + all_answer_positions)
against E2-A (paper-faithful overrides), E0 (paper-faithful TruthfulQA), and
E1 (modernized TruthfulQA).  Includes two frozen comparator sidecars:
  - Sidecar A: E2-B artifact @ K=40, alpha=6.0 (E2-A lock portability)
  - Sidecar B: E2-B artifact @ K=16, alpha=6.0 (compact-K rival)
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from report_e2_canonical import (
    _head_diagnostics,
    _load_json,
    _merge_fold_maps,
    _paired_delta_from_maps,
    _paired_simpleqa_delta,
    _read_mc_fold_map,
    _read_simpleqa_grade_map,
    _simpleqa_summary,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)

    # E2-B (this run)
    p.add_argument("--e2b_name", type=str, default="E2-B TriviaQA Family-Default")
    p.add_argument("--e2b_locked_config", type=Path, required=True)
    p.add_argument("--e2b_pilot_report", type=Path, default=None)
    p.add_argument("--e2b_mc1_fold0_dir", type=Path, required=True)
    p.add_argument("--e2b_mc1_fold1_dir", type=Path, required=True)
    p.add_argument("--e2b_mc2_fold0_dir", type=Path, required=True)
    p.add_argument("--e2b_mc2_fold1_dir", type=Path, required=True)
    p.add_argument("--e2b_simpleqa_dir", type=Path, required=True)
    p.add_argument("--e2b_artifact", type=Path, required=True)
    p.add_argument("--e2b_extraction_metadata", type=Path, required=True)

    # Sidecar A: K=40, alpha=6.0
    p.add_argument("--e2b_k40_mc1_fold0_dir", type=Path, required=True)
    p.add_argument("--e2b_k40_mc1_fold1_dir", type=Path, required=True)
    p.add_argument("--e2b_k40_mc2_fold0_dir", type=Path, required=True)
    p.add_argument("--e2b_k40_mc2_fold1_dir", type=Path, required=True)

    # Sidecar B: K=16, alpha=6.0
    p.add_argument("--e2b_k16_mc1_fold0_dir", type=Path, required=True)
    p.add_argument("--e2b_k16_mc1_fold1_dir", type=Path, required=True)
    p.add_argument("--e2b_k16_mc2_fold0_dir", type=Path, required=True)
    p.add_argument("--e2b_k16_mc2_fold1_dir", type=Path, required=True)

    # E2-A (canonical source-isolated, paper-faithful overrides)
    p.add_argument("--e2a_locked_config", type=Path, required=True)
    p.add_argument("--e2a_mc1_fold0_dir", type=Path, required=True)
    p.add_argument("--e2a_mc1_fold1_dir", type=Path, required=True)
    p.add_argument("--e2a_mc2_fold0_dir", type=Path, required=True)
    p.add_argument("--e2a_mc2_fold1_dir", type=Path, required=True)
    p.add_argument("--e2a_simpleqa_dir", type=Path, required=True)
    p.add_argument("--e2a_artifact", type=Path, required=True)
    p.add_argument("--e2a_extraction_metadata", type=Path, required=True)

    # E0 (paper-faithful TruthfulQA)
    p.add_argument("--paper_locked_config", type=Path, required=True)
    p.add_argument("--paper_mc1_fold0_dir", type=Path, required=True)
    p.add_argument("--paper_mc1_fold1_dir", type=Path, required=True)
    p.add_argument("--paper_mc2_fold0_dir", type=Path, required=True)
    p.add_argument("--paper_mc2_fold1_dir", type=Path, required=True)
    p.add_argument("--paper_simpleqa_dir", type=Path, required=True)
    p.add_argument("--paper_artifact", type=Path, required=True)

    # E1 (modernized TruthfulQA)
    p.add_argument("--e1_locked_config", type=Path, required=True)
    p.add_argument("--e1_mc1_fold0_dir", type=Path, required=True)
    p.add_argument("--e1_mc1_fold1_dir", type=Path, required=True)
    p.add_argument("--e1_mc2_fold0_dir", type=Path, required=True)
    p.add_argument("--e1_mc2_fold1_dir", type=Path, required=True)
    p.add_argument("--e1_simpleqa_dir", type=Path, required=True)
    p.add_argument("--e1_artifact", type=Path, required=True)

    # March 30 sanity check
    p.add_argument("--march30_extraction_metadata", type=Path, default=None)
    p.add_argument("--noncanonical_artifact_mismatch", action="store_true")
    p.add_argument("--artifact_sha_cal", type=str, default=None)
    p.add_argument("--artifact_sha_fold0", type=str, default=None)
    p.add_argument("--artifact_sha_fold1", type=str, default=None)
    p.add_argument("--artifact_sha_prod", type=str, default=None)

    p.add_argument("--materiality_pp", type=float, default=1.5)
    p.add_argument("--output_json", type=Path, required=True)
    p.add_argument("--output_md", type=Path, required=True)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Extraction metadata diff
# ---------------------------------------------------------------------------


def _top_heads_summary(metadata: dict[str, Any], k: int) -> list[dict[str, Any]]:
    manifest = metadata.get("selected_head_manifest", [])
    out = []
    for entry in manifest[:k]:
        out.append(
            {
                "layer": entry["layer"],
                "head": entry["head"],
                "position_summary": entry.get("position_summary", ""),
                "auroc": entry.get("auroc"),
                "balanced_accuracy": entry.get("balanced_accuracy"),
            }
        )
    return out


def _probe_quality_stats(metadata: dict[str, Any], k: int) -> dict[str, Any]:
    manifest = metadata.get("selected_head_manifest", [])[:k]
    if not manifest:
        return {}
    aurocs = [e["auroc"] for e in manifest if e.get("auroc") is not None]
    val_accs = [
        e.get("val_accuracy", e.get("balanced_accuracy"))
        for e in manifest
        if e.get("val_accuracy") is not None or e.get("balanced_accuracy") is not None
    ]
    return {
        "k": k,
        "n_heads": len(manifest),
        "mean_auroc": sum(aurocs) / len(aurocs) if aurocs else None,
        "max_auroc": max(aurocs) if aurocs else None,
        "min_auroc": min(aurocs) if aurocs else None,
        "mean_val_accuracy": sum(val_accs) / len(val_accs) if val_accs else None,
        "max_val_accuracy": max(val_accs) if val_accs else None,
    }


def _extraction_metadata_diff(
    e2b_meta: dict[str, Any],
    e2a_meta: dict[str, Any],
) -> dict[str, Any]:
    fields = [
        "head_ranking_metric",
        "position_policy",
        "position_summaries",
    ]
    diff: dict[str, dict[str, Any]] = {}
    for field in fields:
        e2b_val = e2b_meta.get(field)
        e2a_val = e2a_meta.get(field)
        diff[field] = {"e2b": e2b_val, "e2a": e2a_val, "changed": e2b_val != e2a_val}

    e2b_overrides = e2b_meta.get("selector_overrides", {})
    e2a_overrides = e2a_meta.get("selector_overrides", {})
    diff["selector_overrides"] = {
        "e2b": e2b_overrides,
        "e2a": e2a_overrides,
        "changed": e2b_overrides != e2a_overrides,
    }

    return diff


def _march30_sanity_check(
    e2b_meta: dict[str, Any],
    march30_meta: dict[str, Any],
) -> dict[str, Any]:
    e2b_top5 = _top_heads_summary(e2b_meta, 5)
    m30_ranking = march30_meta.get("ranking", {})
    m30_top5_raw = m30_ranking.get("top5", [])
    m30_top5 = [
        {
            "layer": e["layer"],
            "head": e["head"],
            "position_summary": e.get("position_summary", ""),
            "auroc": e.get("auroc"),
        }
        for e in m30_top5_raw[:5]
    ]

    matches = 0
    for e2b_h, m30_h in zip(e2b_top5, m30_top5, strict=False):
        if (
            e2b_h["layer"] == m30_h["layer"]
            and e2b_h["head"] == m30_h["head"]
            and e2b_h["position_summary"] == m30_h["position_summary"]
        ):
            matches += 1

    return {
        "e2b_top5": e2b_top5,
        "march30_top5": m30_top5,
        "positional_matches": matches,
        "top5_identical": matches == min(len(e2b_top5), len(m30_top5), 5),
    }


# ---------------------------------------------------------------------------
# Diagnostic classification
# ---------------------------------------------------------------------------

SIDECAR_ALPHA = 6.0


def _ci_includes_zero(lower: float, upper: float) -> bool:
    return lower <= 0.0 <= upper


def _format_sha(sha: str | None) -> str:
    if not sha:
        return "missing"
    return sha[:12]


def _artifact_mismatch_pairs(sha_map: dict[str, str | None]) -> list[str]:
    labels = sorted(sha_map)
    out: list[str] = []
    for i, left in enumerate(labels):
        for right in labels[i + 1 :]:
            left_sha = sha_map[left]
            right_sha = sha_map[right]
            if left_sha is None or right_sha is None:
                continue
            if left_sha != right_sha:
                out.append(f"{left}!={right}")
    return out


def _classify_diagnostic(
    *,
    e2b_within_mc1_delta_pp: float,
    e2b_within_mc1_ci_lower_pp: float,
    e2b_within_mc1_ci_upper_pp: float,
    e2b_vs_e2a_mc1_delta_pp: float,
    e2b_vs_e2a_mc1_ci_lower_pp: float,
    e2b_vs_e2a_mc1_ci_upper_pp: float,
    k40_vs_e2a_mc1_delta_pp: float,
    k40_vs_e2a_mc1_ci_lower_pp: float,
    k40_vs_e2a_mc1_ci_upper_pp: float,
    k16_vs_e2a_mc1_delta_pp: float,
    k16_vs_e2a_mc1_ci_lower_pp: float,
    k16_vs_e2a_mc1_ci_upper_pp: float,
    k16_vs_k40_mc1_delta_pp: float,
    k16_vs_k40_mc1_ci_lower_pp: float,
    materiality_pp: float,
) -> str:
    if e2b_within_mc1_ci_lower_pp > 0.0 and e2b_vs_e2a_mc1_ci_lower_pp > 0.0:
        return "selector_mismatch_confirmed"

    # Compact-K anomaly gets precedence over generic suggestive/null labels.
    if k16_vs_k40_mc1_ci_lower_pp > 0.0 or k16_vs_k40_mc1_delta_pp > materiality_pp:
        return "compact_k_anomaly"

    within_suggestive = e2b_within_mc1_delta_pp > materiality_pp and _ci_includes_zero(
        e2b_within_mc1_ci_lower_pp, e2b_within_mc1_ci_upper_pp
    )
    vs_e2a_suggestive = e2b_vs_e2a_mc1_delta_pp > materiality_pp and _ci_includes_zero(
        e2b_vs_e2a_mc1_ci_lower_pp, e2b_vs_e2a_mc1_ci_upper_pp
    )
    if within_suggestive or vs_e2a_suggestive:
        return "suggestive_but_underpowered"

    strict_wrong_source = (
        _ci_includes_zero(e2b_within_mc1_ci_lower_pp, e2b_within_mc1_ci_upper_pp)
        and _ci_includes_zero(k40_vs_e2a_mc1_ci_lower_pp, k40_vs_e2a_mc1_ci_upper_pp)
        and _ci_includes_zero(k16_vs_e2a_mc1_ci_lower_pp, k16_vs_e2a_mc1_ci_upper_pp)
        and abs(e2b_within_mc1_delta_pp) < materiality_pp
        and abs(k40_vs_e2a_mc1_delta_pp) < materiality_pp
        and abs(k16_vs_e2a_mc1_delta_pp) < materiality_pp
    )
    if strict_wrong_source:
        return "wrong_source_still_likely"

    return "ambiguous"


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------


def _fmt_delta(d: dict[str, Any], *, variant: str = "mc1") -> str:
    """Format a paired delta dict as 'X.XX pp (95% CI [X.XX, X.XX])'."""
    est = d["delta"]["estimate_pp"]
    lo = d["delta"]["ci_pp"]["lower"]
    hi = d["delta"]["ci_pp"]["upper"]
    return f"{est:+.2f} pp (95% CI [{lo:+.2f}, {hi:+.2f}])"


def _fmt_simpleqa_delta(d: dict[str, Any], metric: str) -> str:
    sub = d[f"{metric}_delta"]
    if "estimate_pp" in sub:
        est = sub["estimate_pp"]
        lo = sub["ci_pp"]["lower"]
        hi = sub["ci_pp"]["upper"]
    else:
        est = sub["estimate_pp"]
        lo = sub["ci_pp"]["lower"]
        hi = sub["ci_pp"]["upper"]
    return f"{est:+.2f} pp (95% CI [{lo:+.2f}, {hi:+.2f}])"


def _render_markdown(
    *,
    args: argparse.Namespace,
    output: dict[str, Any],
) -> str:
    md: list[str] = []
    artifact_integrity = output.get("artifact_integrity", {})
    noncanonical = bool(artifact_integrity.get("noncanonical_artifact_mismatch", False))
    md.append(f"# {args.e2b_name} -- {datetime.now().date().isoformat()}")
    md.append("")
    if noncanonical:
        md.append(
            "> **NON-CANONICAL: artifact SHA mismatch override was used (`--allow-nonidentical-artifacts`).**"
        )
        md.append(">")
        md.append(
            "> Source-isolation identity assumption is broken; classification is forced to `ambiguous`."
        )
        sha_map = artifact_integrity.get("sha256", {})
        md.append(
            f"> SHAs (prefixes): CAL={_format_sha(sha_map.get('cal'))}, "
            f"FOLD0={_format_sha(sha_map.get('fold0'))}, "
            f"FOLD1={_format_sha(sha_map.get('fold1'))}, "
            f"PROD={_format_sha(sha_map.get('prod'))}"
        )
        mismatches = artifact_integrity.get("mismatched_pairs", [])
        if mismatches:
            md.append(f"> Mismatches: {', '.join(mismatches)}")
        md.append("")
    md.append("> **Status: diagnostic report for E2-B selector sensitivity test.**")
    md.append(">")
    md.append(
        "> E2-A (val_accuracy + last_answer_token) was near-inert. "
        "E2-B tests whether AUROC + all_answer_positions unlocks a signal "
        "in the same TriviaQA data."
    )
    md.append("")

    # Section 1: Extraction metadata diff
    diff = output["extraction_metadata_diff"]
    md.append("## 1. Extraction Metadata Diff (E2-B vs E2-A)")
    md.append("")
    md.append("| Property | E2-A | E2-B | Changed? |")
    md.append("|---|---|---|---|")
    for field in ["head_ranking_metric", "position_policy", "position_summaries"]:
        d = diff[field]
        e2a_val = d["e2a"]
        e2b_val = d["e2b"]
        changed = "YES" if d["changed"] else "no"
        md.append(f"| {field} | `{e2a_val}` | `{e2b_val}` | {changed} |")
    overrides = diff["selector_overrides"]
    md.append(
        f"| selector_overrides | `{overrides['e2a']}` | `{overrides['e2b']}` "
        f"| {'YES' if overrides['changed'] else 'no'} |"
    )
    md.append("")

    # Section 2: Probe quality
    pq = output["probe_quality"]
    md.append("## 2. Probe Quality Under Family-Default Selectors")
    md.append("")
    md.append("| Metric | E2-B | E2-A | E0 (paper) |")
    md.append("|---|---|---|---|")
    for metric in ["mean_auroc", "max_auroc", "min_auroc", "mean_val_accuracy"]:
        e2b_v = pq["e2b"].get(metric)
        e2a_v = pq["e2a"].get(metric)
        paper_v = pq.get("paper", {}).get(metric)

        def _fmt_val(v: float | None) -> str:
            return f"{v:.4f}" if v is not None else "—"

        md.append(
            f"| {metric} | {_fmt_val(e2b_v)} | {_fmt_val(e2a_v)} | {_fmt_val(paper_v)} |"
        )
    md.append("")

    # Top-5 heads
    md.append("**E2-B top-5 heads:**")
    md.append("")
    md.append("| Rank | Layer | Head | Position | AUROC |")
    md.append("|---|---|---|---|---|")
    for i, h in enumerate(pq.get("e2b_top5", [])[:5]):
        md.append(
            f"| {i + 1} | {h['layer']} | {h['head']} | {h['position_summary']} "
            f"| {h.get('auroc', 0):.4f} |"
        )
    md.append("")

    # March 30 sanity check
    sanity = output.get("march30_sanity_check")
    if sanity:
        md.append(
            "**March 30 artifact sanity check:** "
            f"{'PASS' if sanity['top5_identical'] else 'DIVERGENT'} "
            f"({sanity['positional_matches']}/5 positional matches)"
        )
        md.append("")

    # Section 3: Calibration sweep
    cal = output.get("calibration_sweep")
    if cal:
        md.append("## 3. Calibration Sweep Summary")
        md.append("")
        md.append(f"- Best MC1: {cal.get('best_mc1', '—')}")
        md.append(
            f"- Locked config: K={cal.get('locked_k', '—')}, "
            f"alpha={cal.get('locked_alpha', '—')}"
        )
        if cal.get("shortlist"):
            md.append(f"- Shortlist ({len(cal['shortlist'])} candidates):")
            for row in cal["shortlist"]:
                md.append(
                    f"  - K={row['k']}, alpha={row['alpha']}, "
                    f"MC1={row['mc1']:.4f}, MC2={row['mc2']:.4f}"
                )
        md.append("")

    # Section 4: TruthfulQA
    tq = output["truthfulqa"]
    md.append("## 4. Held-Out TruthfulQA (Pooled n=655)")
    md.append("")
    md.append("### 4.1 Within E2-B (optimized lock vs alpha=0)")
    md.append(f"- MC1: {_fmt_delta(tq['within_e2b']['mc1'])}")
    md.append(f"- MC2: {_fmt_delta(tq['within_e2b']['mc2'])}")
    md.append("")

    md.append("### 4.2 E2-B Optimized vs E2-A (KEY DIAGNOSTIC)")
    md.append(f"- MC1: {_fmt_delta(tq['e2b_vs_e2a']['mc1'])}")
    md.append(f"- MC2: {_fmt_delta(tq['e2b_vs_e2a']['mc2'])}")
    md.append("")

    md.append(
        "### 4.3 Sidecar A: E2-B @ K=40 alpha=6.0 vs E2-A (artifact-only isolation)"
    )
    md.append(f"- MC1: {_fmt_delta(tq['k40_vs_e2a']['mc1'])}")
    md.append(f"- MC2: {_fmt_delta(tq['k40_vs_e2a']['mc2'])}")
    md.append("")

    md.append(
        "### 4.4 Sidecar B: E2-B @ K=16 alpha=6.0 vs E2-B @ K=40 alpha=6.0 (compact vs broad)"
    )
    md.append(f"- MC1: {_fmt_delta(tq['k16_vs_k40']['mc1'])}")
    md.append(f"- MC2: {_fmt_delta(tq['k16_vs_k40']['mc2'])}")
    md.append("")

    md.append("### 4.5 E2-B @ K=16 alpha=6.0 vs E2-A (combined effect)")
    md.append(f"- MC1: {_fmt_delta(tq['k16_vs_e2a']['mc1'])}")
    md.append(f"- MC2: {_fmt_delta(tq['k16_vs_e2a']['mc2'])}")
    md.append("")

    md.append("### 4.6 E2-B Optimized vs E0 (paper-faithful TruthfulQA)")
    md.append(f"- MC1: {_fmt_delta(tq['e2b_vs_paper']['mc1'])}")
    md.append(f"- MC2: {_fmt_delta(tq['e2b_vs_paper']['mc2'])}")
    md.append("")

    md.append("### 4.7 E2-B Optimized vs E1 (modernized TruthfulQA)")
    md.append(f"- MC1: {_fmt_delta(tq['e2b_vs_e1']['mc1'])}")
    md.append(f"- MC2: {_fmt_delta(tq['e2b_vs_e1']['mc2'])}")
    md.append("")

    # Section 5: SimpleQA
    sq = output["simpleqa"]
    md.append("## 5. SimpleQA-200 (first_3_tokens)")
    md.append("")
    md.append("### 5.1 Within E2-B")
    md.append(
        f"- Compliance: {_fmt_simpleqa_delta(sq['within_e2b']['paired'], 'compliance')}"
    )
    md.append(
        f"- Attempt: {_fmt_simpleqa_delta(sq['within_e2b']['paired'], 'attempt')}"
    )
    md.append(
        f"- Precision: {sq['within_e2b']['paired']['precision_delta']['estimate_pp']:+.2f} pp "
        f"(95% CI [{sq['within_e2b']['paired']['precision_delta']['ci_pp']['lower']:+.2f}, "
        f"{sq['within_e2b']['paired']['precision_delta']['ci_pp']['upper']:+.2f}])"
    )
    md.append("")

    md.append("### 5.2 E2-B vs E2-A")
    md.append(f"- Compliance: {_fmt_simpleqa_delta(sq['e2b_vs_e2a'], 'compliance')}")
    md.append(f"- Attempt: {_fmt_simpleqa_delta(sq['e2b_vs_e2a'], 'attempt')}")
    md.append(
        f"- Precision: {sq['e2b_vs_e2a']['precision_delta']['estimate_pp']:+.2f} pp "
        f"(95% CI [{sq['e2b_vs_e2a']['precision_delta']['ci_pp']['lower']:+.2f}, "
        f"{sq['e2b_vs_e2a']['precision_delta']['ci_pp']['upper']:+.2f}])"
    )
    md.append("")

    md.append("### 5.3 E2-B vs E0 (paper-faithful)")
    md.append(f"- Compliance: {_fmt_simpleqa_delta(sq['e2b_vs_paper'], 'compliance')}")
    md.append(f"- Attempt: {_fmt_simpleqa_delta(sq['e2b_vs_paper'], 'attempt')}")
    md.append("")

    md.append("### 5.4 E2-B vs E1 (modernized)")
    md.append(f"- Compliance: {_fmt_simpleqa_delta(sq['e2b_vs_e1'], 'compliance')}")
    md.append(f"- Attempt: {_fmt_simpleqa_delta(sq['e2b_vs_e1'], 'attempt')}")
    md.append("")

    # Section 6: Head overlap
    diag = output["diagnostics"]
    md.append("## 6. Head Overlap Diagnostics")
    md.append("")
    for label, key in [
        ("E2-B vs E2-A (same data, different selectors)", "head_overlap_vs_e2a"),
        ("E2-B vs E0 (different data, different selectors)", "head_overlap_vs_paper"),
        ("E2-B vs E1 (different data, different selectors)", "head_overlap_vs_e1"),
    ]:
        hd = diag[key]
        overlap = hd["selected_overlap"]
        rank = hd["rank_agreement"]
        dirs = hd["direction_similarity"]
        md.append(f"### {label}")
        md.append(
            f"- Jaccard: {overlap['jaccard']:.3f} "
            f"(intersection={overlap['intersection']}, union={overlap['union']})"
        )
        rho = rank.get("spearman_rho")
        if rho is not None:
            md.append(
                f"- Spearman rho (full rank): {rho:.3f} "
                f"(n={rank['n_shared_ranked_heads']}, p={rank['spearman_pvalue']:.4f})"
            )
        cos = dirs.get("mean_cosine")
        if cos is not None:
            md.append(
                f"- Direction cosines (shared selected): "
                f"mean={cos:.3f}, median={dirs['median_cosine']:.3f}, "
                f"range=[{dirs['min_cosine']:.3f}, {dirs['max_cosine']:.3f}]"
            )
        md.append("")

    # Section 7: Classification
    md.append("## 7. Diagnostic Classification")
    md.append("")
    md.append(f"**Classification: `{output['classification']}`**")
    override_reason = output.get("classification_override_reason")
    if override_reason:
        md.append(
            f"**Override reason:** `{override_reason}` (raw class: `{output.get('classification_raw')}`)"
        )
    md.append("")

    labels = {
        "selector_mismatch_confirmed": (
            "Family-default selectors (AUROC + all_answer_positions) recover a signal "
            "that paper-faithful overrides missed. E2-A's null was a selector problem."
        ),
        "wrong_source_still_likely": (
            "Both selector policies produce null results. The issue is likely TriviaQA "
            "source quality or model-level direction weakness, not selector choice."
        ),
        "compact_k_anomaly": (
            "K=16 outperforms K=40 on held-out data, suggesting a brittle large-K "
            "calibration tie-break in E2-A. The signal may be concentrated in fewer heads."
        ),
        "suggestive_but_underpowered": (
            "E2-B shows a positive MC1 point estimate but the CI includes zero. "
            "Directionally consistent with selector sensitivity but not conclusive."
        ),
        "ambiguous": "Mixed signals; no clean interpretation available.",
    }
    md.append(labels.get(output["classification"], "Unknown classification."))
    md.append("")

    # Section 8: Decision
    md.append("## 8. Decision")
    md.append("")
    if output["classification"] == "selector_mismatch_confirmed":
        md.append(
            "The selector override was the dominant failure mode for E2-A. "
            "TriviaQA-sourced directions under family-default selectors show real MC gain. "
            "Consider whether the gain rivals E0 before investing further."
        )
    elif output["classification"] == "wrong_source_still_likely":
        md.append(
            "Both selector policies failed. Accept the three-variant evidence: "
            "TriviaQA transfer under mass-mean ITI does not engage TruthfulQA-relevant "
            "representations in this model. Shift priority to D5 (externality audit) "
            "or D7 (causal head selection)."
        )
    else:
        md.append(
            "Result is not cleanly interpretable. Review the sidecar comparisons "
            "and head overlap diagnostics before deciding next steps."
        )
    md.append("")

    return "\n".join(md) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    # Load locked configs
    e2b_lock = _load_json(args.e2b_locked_config)
    e2a_lock = _load_json(args.e2a_locked_config)
    paper_lock = _load_json(args.paper_locked_config)
    e1_lock = _load_json(args.e1_locked_config)

    e2b_alpha = float(e2b_lock["alpha_locked"])
    e2b_k = int(e2b_lock["K_locked"])
    e2a_alpha = float(e2a_lock["alpha_locked"])
    e2a_k = int(e2a_lock["K_locked"])
    paper_alpha = float(paper_lock["alpha_locked"])
    paper_k = int(paper_lock["K_locked"])
    e1_alpha = float(e1_lock["alpha_locked"])
    e1_k = int(e1_lock["K_locked"])

    seed = int(args.seed)

    # ------------------------------------------------------------------
    # Extraction metadata diff
    # ------------------------------------------------------------------
    e2b_meta = _load_json(args.e2b_extraction_metadata)
    e2a_meta = _load_json(args.e2a_extraction_metadata)
    meta_diff = _extraction_metadata_diff(e2b_meta, e2a_meta)

    # Probe quality comparison
    e2b_pq = _probe_quality_stats(e2b_meta, e2b_k)
    e2b_top5 = _top_heads_summary(e2b_meta, 5)
    e2a_pq = _probe_quality_stats(e2a_meta, e2a_k)

    # Try to get E0 probe quality from its extraction metadata
    paper_meta_path = args.paper_artifact.parent / "extraction_metadata.json"
    paper_pq: dict[str, Any] = {}
    if paper_meta_path.exists():
        paper_meta_data = _load_json(paper_meta_path)
        paper_pq = _probe_quality_stats(paper_meta_data, paper_k)

    # March 30 sanity check
    march30_check = None
    if args.march30_extraction_metadata and args.march30_extraction_metadata.exists():
        march30_meta = _load_json(args.march30_extraction_metadata)
        march30_check = _march30_sanity_check(e2b_meta, march30_meta)

    # ------------------------------------------------------------------
    # E2-B within: optimized lock vs baseline
    # ------------------------------------------------------------------
    e2b_mc1_base = _merge_fold_maps(
        _read_mc_fold_map(args.e2b_mc1_fold0_dir, 0.0, "mc1"),
        _read_mc_fold_map(args.e2b_mc1_fold1_dir, 0.0, "mc1"),
    )
    e2b_mc1_lock = _merge_fold_maps(
        _read_mc_fold_map(args.e2b_mc1_fold0_dir, e2b_alpha, "mc1"),
        _read_mc_fold_map(args.e2b_mc1_fold1_dir, e2b_alpha, "mc1"),
    )
    e2b_mc2_base = _merge_fold_maps(
        _read_mc_fold_map(args.e2b_mc2_fold0_dir, 0.0, "mc2"),
        _read_mc_fold_map(args.e2b_mc2_fold1_dir, 0.0, "mc2"),
    )
    e2b_mc2_lock = _merge_fold_maps(
        _read_mc_fold_map(args.e2b_mc2_fold0_dir, e2b_alpha, "mc2"),
        _read_mc_fold_map(args.e2b_mc2_fold1_dir, e2b_alpha, "mc2"),
    )

    e2b_mc1_within = _paired_delta_from_maps(
        e2b_mc1_base, e2b_mc1_lock, variant="mc1", seed=seed
    )
    e2b_mc2_within = _paired_delta_from_maps(
        e2b_mc2_base, e2b_mc2_lock, variant="mc2", seed=seed
    )

    # ------------------------------------------------------------------
    # E2-A locked maps
    # ------------------------------------------------------------------
    e2a_mc1_lock = _merge_fold_maps(
        _read_mc_fold_map(args.e2a_mc1_fold0_dir, e2a_alpha, "mc1"),
        _read_mc_fold_map(args.e2a_mc1_fold1_dir, e2a_alpha, "mc1"),
    )
    e2a_mc2_lock = _merge_fold_maps(
        _read_mc_fold_map(args.e2a_mc2_fold0_dir, e2a_alpha, "mc2"),
        _read_mc_fold_map(args.e2a_mc2_fold1_dir, e2a_alpha, "mc2"),
    )

    # E2-B optimized vs E2-A
    e2b_vs_e2a_mc1 = _paired_delta_from_maps(
        e2a_mc1_lock, e2b_mc1_lock, variant="mc1", seed=seed
    )
    e2b_vs_e2a_mc2 = _paired_delta_from_maps(
        e2a_mc2_lock, e2b_mc2_lock, variant="mc2", seed=seed
    )

    # ------------------------------------------------------------------
    # Sidecar A: K=40, alpha=6.0
    # ------------------------------------------------------------------
    k40_mc1_lock = _merge_fold_maps(
        _read_mc_fold_map(args.e2b_k40_mc1_fold0_dir, SIDECAR_ALPHA, "mc1"),
        _read_mc_fold_map(args.e2b_k40_mc1_fold1_dir, SIDECAR_ALPHA, "mc1"),
    )
    k40_mc2_lock = _merge_fold_maps(
        _read_mc_fold_map(args.e2b_k40_mc2_fold0_dir, SIDECAR_ALPHA, "mc2"),
        _read_mc_fold_map(args.e2b_k40_mc2_fold1_dir, SIDECAR_ALPHA, "mc2"),
    )

    k40_vs_e2a_mc1 = _paired_delta_from_maps(
        e2a_mc1_lock, k40_mc1_lock, variant="mc1", seed=seed
    )
    k40_vs_e2a_mc2 = _paired_delta_from_maps(
        e2a_mc2_lock, k40_mc2_lock, variant="mc2", seed=seed
    )

    # ------------------------------------------------------------------
    # Sidecar B: K=16, alpha=6.0
    # ------------------------------------------------------------------
    k16_mc1_lock = _merge_fold_maps(
        _read_mc_fold_map(args.e2b_k16_mc1_fold0_dir, SIDECAR_ALPHA, "mc1"),
        _read_mc_fold_map(args.e2b_k16_mc1_fold1_dir, SIDECAR_ALPHA, "mc1"),
    )
    k16_mc2_lock = _merge_fold_maps(
        _read_mc_fold_map(args.e2b_k16_mc2_fold0_dir, SIDECAR_ALPHA, "mc2"),
        _read_mc_fold_map(args.e2b_k16_mc2_fold1_dir, SIDECAR_ALPHA, "mc2"),
    )

    k16_vs_k40_mc1 = _paired_delta_from_maps(
        k40_mc1_lock, k16_mc1_lock, variant="mc1", seed=seed
    )
    k16_vs_k40_mc2 = _paired_delta_from_maps(
        k40_mc2_lock, k16_mc2_lock, variant="mc2", seed=seed
    )

    k16_vs_e2a_mc1 = _paired_delta_from_maps(
        e2a_mc1_lock, k16_mc1_lock, variant="mc1", seed=seed
    )
    k16_vs_e2a_mc2 = _paired_delta_from_maps(
        e2a_mc2_lock, k16_mc2_lock, variant="mc2", seed=seed
    )

    # ------------------------------------------------------------------
    # E0 and E1 comparisons
    # ------------------------------------------------------------------
    paper_mc1_lock = _merge_fold_maps(
        _read_mc_fold_map(args.paper_mc1_fold0_dir, paper_alpha, "mc1"),
        _read_mc_fold_map(args.paper_mc1_fold1_dir, paper_alpha, "mc1"),
    )
    paper_mc2_lock = _merge_fold_maps(
        _read_mc_fold_map(args.paper_mc2_fold0_dir, paper_alpha, "mc2"),
        _read_mc_fold_map(args.paper_mc2_fold1_dir, paper_alpha, "mc2"),
    )
    e1_mc1_lock = _merge_fold_maps(
        _read_mc_fold_map(args.e1_mc1_fold0_dir, e1_alpha, "mc1"),
        _read_mc_fold_map(args.e1_mc1_fold1_dir, e1_alpha, "mc1"),
    )
    e1_mc2_lock = _merge_fold_maps(
        _read_mc_fold_map(args.e1_mc2_fold0_dir, e1_alpha, "mc2"),
        _read_mc_fold_map(args.e1_mc2_fold1_dir, e1_alpha, "mc2"),
    )

    e2b_vs_paper_mc1 = _paired_delta_from_maps(
        paper_mc1_lock, e2b_mc1_lock, variant="mc1", seed=seed
    )
    e2b_vs_paper_mc2 = _paired_delta_from_maps(
        paper_mc2_lock, e2b_mc2_lock, variant="mc2", seed=seed
    )
    e2b_vs_e1_mc1 = _paired_delta_from_maps(
        e1_mc1_lock, e2b_mc1_lock, variant="mc1", seed=seed
    )
    e2b_vs_e1_mc2 = _paired_delta_from_maps(
        e1_mc2_lock, e2b_mc2_lock, variant="mc2", seed=seed
    )

    # ------------------------------------------------------------------
    # SimpleQA
    # ------------------------------------------------------------------
    e2b_sq_base = _read_simpleqa_grade_map(args.e2b_simpleqa_dir, 0.0)
    e2b_sq_lock = _read_simpleqa_grade_map(args.e2b_simpleqa_dir, e2b_alpha)
    e2a_sq_lock = _read_simpleqa_grade_map(args.e2a_simpleqa_dir, e2a_alpha)
    paper_sq_lock = _read_simpleqa_grade_map(args.paper_simpleqa_dir, paper_alpha)
    e1_sq_lock = _read_simpleqa_grade_map(args.e1_simpleqa_dir, e1_alpha)

    e2b_sq_within = _paired_simpleqa_delta(e2b_sq_base, e2b_sq_lock, seed=seed)
    e2b_vs_e2a_sq = _paired_simpleqa_delta(e2a_sq_lock, e2b_sq_lock, seed=seed)
    e2b_vs_paper_sq = _paired_simpleqa_delta(paper_sq_lock, e2b_sq_lock, seed=seed)
    e2b_vs_e1_sq = _paired_simpleqa_delta(e1_sq_lock, e2b_sq_lock, seed=seed)

    # ------------------------------------------------------------------
    # Head diagnostics
    # ------------------------------------------------------------------
    head_vs_e2a = _head_diagnostics(args.e2b_artifact, e2b_k, args.e2a_artifact, e2a_k)
    head_vs_paper = _head_diagnostics(
        args.e2b_artifact, e2b_k, args.paper_artifact, paper_k
    )
    head_vs_e1 = _head_diagnostics(args.e2b_artifact, e2b_k, args.e1_artifact, e1_k)

    pilot_report = _load_json(args.e2b_pilot_report) if args.e2b_pilot_report else None

    # ------------------------------------------------------------------
    # Calibration sweep summary (from locked config)
    # ------------------------------------------------------------------
    cal_sweep: dict[str, Any] = {
        "best_mc1": e2b_lock.get("selection_diagnostics", {}).get("best_mc1"),
        "locked_k": e2b_k,
        "locked_alpha": e2b_alpha,
        "shortlist": e2b_lock.get("selection_diagnostics", {}).get("shortlist"),
    }

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------
    classification_raw = _classify_diagnostic(
        e2b_within_mc1_delta_pp=float(e2b_mc1_within["delta"]["estimate_pp"]),
        e2b_within_mc1_ci_lower_pp=float(e2b_mc1_within["delta"]["ci_pp"]["lower"]),
        e2b_within_mc1_ci_upper_pp=float(e2b_mc1_within["delta"]["ci_pp"]["upper"]),
        e2b_vs_e2a_mc1_delta_pp=float(e2b_vs_e2a_mc1["delta"]["estimate_pp"]),
        e2b_vs_e2a_mc1_ci_lower_pp=float(e2b_vs_e2a_mc1["delta"]["ci_pp"]["lower"]),
        e2b_vs_e2a_mc1_ci_upper_pp=float(e2b_vs_e2a_mc1["delta"]["ci_pp"]["upper"]),
        k40_vs_e2a_mc1_delta_pp=float(k40_vs_e2a_mc1["delta"]["estimate_pp"]),
        k40_vs_e2a_mc1_ci_lower_pp=float(k40_vs_e2a_mc1["delta"]["ci_pp"]["lower"]),
        k40_vs_e2a_mc1_ci_upper_pp=float(k40_vs_e2a_mc1["delta"]["ci_pp"]["upper"]),
        k16_vs_e2a_mc1_delta_pp=float(k16_vs_e2a_mc1["delta"]["estimate_pp"]),
        k16_vs_e2a_mc1_ci_lower_pp=float(k16_vs_e2a_mc1["delta"]["ci_pp"]["lower"]),
        k16_vs_e2a_mc1_ci_upper_pp=float(k16_vs_e2a_mc1["delta"]["ci_pp"]["upper"]),
        k16_vs_k40_mc1_delta_pp=float(k16_vs_k40_mc1["delta"]["estimate_pp"]),
        k16_vs_k40_mc1_ci_lower_pp=float(k16_vs_k40_mc1["delta"]["ci_pp"]["lower"]),
        materiality_pp=float(args.materiality_pp),
    )
    classification = classification_raw
    classification_override_reason: str | None = None
    if args.noncanonical_artifact_mismatch:
        classification = "ambiguous"
        classification_override_reason = "artifact_identity_mismatch"

    artifact_sha = {
        "cal": args.artifact_sha_cal,
        "fold0": args.artifact_sha_fold0,
        "fold1": args.artifact_sha_fold1,
        "prod": args.artifact_sha_prod,
    }
    artifact_integrity = {
        "noncanonical_artifact_mismatch": bool(args.noncanonical_artifact_mismatch),
        "sha256": artifact_sha,
        "mismatched_pairs": _artifact_mismatch_pairs(artifact_sha),
    }

    # ------------------------------------------------------------------
    # Assemble output
    # ------------------------------------------------------------------
    output: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "e2b_name": args.e2b_name,
        "config": {
            "e2b": {
                "lock_config": str(args.e2b_locked_config),
                "alpha_locked": e2b_alpha,
                "k_locked": e2b_k,
                "artifact": str(args.e2b_artifact),
            },
            "e2a": {
                "lock_config": str(args.e2a_locked_config),
                "alpha_locked": e2a_alpha,
                "k_locked": e2a_k,
                "artifact": str(args.e2a_artifact),
            },
            "paper": {
                "lock_config": str(args.paper_locked_config),
                "alpha_locked": paper_alpha,
                "k_locked": paper_k,
                "artifact": str(args.paper_artifact),
            },
            "e1": {
                "lock_config": str(args.e1_locked_config),
                "alpha_locked": e1_alpha,
                "k_locked": e1_k,
                "artifact": str(args.e1_artifact),
            },
            "sidecar_alpha": SIDECAR_ALPHA,
            "sidecar_k_values": [40, 16],
            "materiality_pp": float(args.materiality_pp),
        },
        "extraction_metadata_diff": meta_diff,
        "probe_quality": {
            "e2b": e2b_pq,
            "e2b_top5": e2b_top5,
            "e2a": e2a_pq,
            "paper": paper_pq,
        },
        "march30_sanity_check": march30_check,
        "calibration_sweep": cal_sweep,
        "pilot": pilot_report,
        "truthfulqa": {
            "within_e2b": {"mc1": e2b_mc1_within, "mc2": e2b_mc2_within},
            "e2b_vs_e2a": {"mc1": e2b_vs_e2a_mc1, "mc2": e2b_vs_e2a_mc2},
            "k40_vs_e2a": {"mc1": k40_vs_e2a_mc1, "mc2": k40_vs_e2a_mc2},
            "k16_vs_k40": {"mc1": k16_vs_k40_mc1, "mc2": k16_vs_k40_mc2},
            "k16_vs_e2a": {"mc1": k16_vs_e2a_mc1, "mc2": k16_vs_e2a_mc2},
            "e2b_vs_paper": {"mc1": e2b_vs_paper_mc1, "mc2": e2b_vs_paper_mc2},
            "e2b_vs_e1": {"mc1": e2b_vs_e1_mc1, "mc2": e2b_vs_e1_mc2},
        },
        "simpleqa": {
            "within_e2b": {
                "baseline": _simpleqa_summary(e2b_sq_base),
                "locked": _simpleqa_summary(e2b_sq_lock),
                "paired": e2b_sq_within,
            },
            "e2b_vs_e2a": e2b_vs_e2a_sq,
            "e2b_vs_paper": e2b_vs_paper_sq,
            "e2b_vs_e1": e2b_vs_e1_sq,
        },
        "diagnostics": {
            "head_overlap_vs_e2a": head_vs_e2a,
            "head_overlap_vs_paper": head_vs_paper,
            "head_overlap_vs_e1": head_vs_e1,
        },
        "artifact_integrity": artifact_integrity,
        "classification_raw": classification_raw,
        "classification_override_reason": classification_override_reason,
        "classification": classification,
    }

    # Write JSON
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")

    # Write Markdown
    md_text = _render_markdown(args=args, output=output)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(md_text, encoding="utf-8")

    print(f"Saved JSON report: {args.output_json}")
    print(f"Saved Markdown report: {args.output_md}")
    print(f"Classification: {classification}")


if __name__ == "__main__":
    main()
