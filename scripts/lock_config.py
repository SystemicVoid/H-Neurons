#!/usr/bin/env python3
"""Lock ITI hyperparameters into the pipeline state file.

Reads ``sweep_results.json`` from the state directory, applies the
resolution-aware shortlist rule (optionally with pilot poisoning gate),
and writes both ``locked_iti_config.json`` and ``pipeline_state.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from run_calibration_sweep import (
    MIN_TOLERANCE_PP,
    compute_selection_diagnostics,
    infer_position_policy,
    infer_ranking_metric,
    select_locked_config,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Lock ITI hyperparameters into pipeline state."
    )
    p.add_argument(
        "--state_dir",
        type=str,
        required=True,
        help="Directory containing sweep_results.json",
    )
    p.add_argument("--K", type=int, default=None, help="Override K value")
    p.add_argument("--alpha", type=float, default=None, help="Override alpha value")
    p.add_argument(
        "--reason",
        type=str,
        default=None,
        help="Required justification when using --K / --alpha override",
    )
    p.add_argument(
        "--tolerance_pp",
        type=float,
        default=None,
        help=(
            "Optional MC1 shortlist tolerance in pp. Applied tolerance is "
            "max(requested, 100/n_cal, 1.5). If omitted, reuses sweep tolerance."
        ),
    )
    p.add_argument(
        "--pilot_report",
        type=str,
        default=None,
        help="Optional shortlist pilot report JSON used for generation-poison gating.",
    )
    p.add_argument(
        "--pilot_poison_attempt_delta_pp",
        type=float,
        default=-10.0,
        help="Reject if attempt delta <= this and precision delta <= threshold.",
    )
    p.add_argument(
        "--pilot_poison_precision_delta_pp",
        type=float,
        default=0.0,
        help="Used with attempt threshold for poison rejection.",
    )
    p.add_argument(
        "--pilot_poison_not_attempted_delta_n",
        type=int,
        default=15,
        help="Reject if NOT_ATTEMPTED count increase >= this value.",
    )
    return p.parse_args()


def _find_result(
    results: list[dict[str, Any]], k: int, alpha: float
) -> dict[str, Any] | None:
    """Find the sweep result matching (K, alpha), or None."""
    for r in results:
        if int(r["k"]) == int(k) and float(r["alpha"]) == float(alpha):
            return r
    return None


def _infer_artifact_family(
    sweep_data: dict[str, Any], state_dir: Path, artifact_path: str | None
) -> str | None:
    """Infer family if absent from sweep metadata."""
    family = sweep_data.get("artifact_family")
    if family:
        return str(family)

    candidates = [str(state_dir), artifact_path or ""]
    for value in candidates:
        if "paperfaithful" in value:
            return "iti_truthfulqa_paperfaithful"
        if "modernized" in value:
            return "iti_truthfulqa_modernized"
        if "triviaqa_transfer" in value or "iti_triviaqa" in value:
            return "iti_triviaqa_transfer"
        if "context_grounded" in value:
            return "iti_context_grounded"
    return None


def _candidate_key(k: int, alpha: float) -> tuple[int, str]:
    return int(k), f"{float(alpha):.12g}"


def _resolve_tolerance(
    *,
    requested_tolerance_pp: float | None,
    sweep_data: dict[str, Any],
    n_calibration_samples: int,
) -> tuple[float | None, float]:
    existing_diag = sweep_data.get("selection_diagnostics", {})
    if (
        requested_tolerance_pp is None
        and existing_diag.get("tolerance_pp_applied") is not None
    ):
        return (
            existing_diag.get("tolerance_pp_requested"),
            float(existing_diag["tolerance_pp_applied"]),
        )

    if n_calibration_samples <= 0:
        raise ValueError("n_calibration_samples must be positive")
    resolution_floor_pp = max(100.0 / n_calibration_samples, MIN_TOLERANCE_PP)
    if requested_tolerance_pp is None:
        return (None, resolution_floor_pp)
    return (
        requested_tolerance_pp,
        max(float(requested_tolerance_pp), resolution_floor_pp),
    )


def _load_pilot_map(pilot_report_path: str) -> dict[tuple[int, str], dict[str, Any]]:
    pilot_data = json.loads(Path(pilot_report_path).read_text(encoding="utf-8"))
    rows = pilot_data.get("candidates", [])
    pilot_map: dict[tuple[int, str], dict[str, Any]] = {}
    for row in rows:
        key = _candidate_key(int(row["k"]), float(row["alpha"]))
        pilot_map[key] = row
    return pilot_map


def _apply_pilot_poison_gate(
    shortlist: list[dict[str, Any]],
    *,
    pilot_map: dict[tuple[int, str], dict[str, Any]],
    attempt_threshold_pp: float,
    precision_threshold_pp: float,
    not_attempted_threshold_n: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    survivors: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []

    for candidate in shortlist:
        key = _candidate_key(int(candidate["k"]), float(candidate["alpha"]))
        pilot_row = pilot_map.get(key)
        if pilot_row is None:
            raise ValueError(
                "Pilot report missing shortlist candidate "
                f"(K={candidate['k']}, alpha={candidate['alpha']})."
            )

        attempt_delta = float(pilot_row["attempt_delta_pp"])
        precision_delta = float(pilot_row["precision_delta_pp"])
        not_attempted_delta = int(pilot_row["not_attempted_delta_n"])

        reasons: list[str] = []
        if (
            attempt_delta <= attempt_threshold_pp
            and precision_delta <= precision_threshold_pp
        ):
            reasons.append("attempt_and_precision_gate")
        if not_attempted_delta >= not_attempted_threshold_n:
            reasons.append("not_attempted_spike_gate")

        rejected = len(reasons) > 0
        row = {
            "k": int(candidate["k"]),
            "alpha": float(candidate["alpha"]),
            "attempt_delta_pp": attempt_delta,
            "precision_delta_pp": precision_delta,
            "not_attempted_delta_n": not_attempted_delta,
            "rejected": rejected,
            "rejection_reasons": reasons,
        }
        diagnostics.append(row)
        if not rejected:
            survivors.append(candidate)

    return survivors, diagnostics


def _select_from_shortlist(shortlist: list[dict[str, Any]]) -> dict[str, Any]:
    if not shortlist:
        raise ValueError("Shortlist is empty")
    shortlist_sorted = sorted(
        shortlist,
        key=lambda row: (-float(row["mc2"]), float(row["alpha"]), int(row["k"])),
    )
    return shortlist_sorted[0]


def main() -> None:
    args = parse_args()
    state_dir = Path(args.state_dir)

    # --- Validate override args ---
    has_k = args.K is not None
    has_alpha = args.alpha is not None
    if has_k != has_alpha:
        print(
            "ERROR: --K and --alpha must both be provided (or neither).",
            file=sys.stderr,
        )
        sys.exit(1)
    if (has_k or has_alpha) and not args.reason:
        print(
            "ERROR: --reason is required when using --K / --alpha override.",
            file=sys.stderr,
        )
        sys.exit(1)

    # --- Load sweep results ---
    sweep_path = state_dir / "sweep_results.json"
    if not sweep_path.exists():
        print(f"ERROR: {sweep_path} not found.", file=sys.stderr)
        sys.exit(1)

    sweep_data = json.loads(sweep_path.read_text(encoding="utf-8"))
    results: list[dict[str, Any]] = sweep_data["results"]
    if not results:
        print("ERROR: sweep_results.json has no results.", file=sys.stderr)
        sys.exit(1)

    n_calibration_samples = int(
        sweep_data.get("n_mc1_samples")
        or next((int(r["n_mc1"]) for r in results if int(r.get("n_mc1", 0)) > 0), 0)
    )
    if n_calibration_samples <= 0:
        print("ERROR: could not infer calibration sample size.", file=sys.stderr)
        sys.exit(1)

    tolerance_pp_requested, tolerance_pp_applied = _resolve_tolerance(
        requested_tolerance_pp=args.tolerance_pp,
        sweep_data=sweep_data,
        n_calibration_samples=n_calibration_samples,
    )

    # --- Auto-suggestion (always computed for the state record) ---
    auto_pick = select_locked_config(results, tolerance_pp=tolerance_pp_applied)
    auto_suggestion = {
        "K": int(auto_pick["k"]),
        "alpha": float(auto_pick["alpha"]),
        "mc1": float(auto_pick["mc1"]),
        "mc2": float(auto_pick["mc2"]),
    }

    selection_diagnostics = compute_selection_diagnostics(
        results,
        tolerance_pp_requested=tolerance_pp_requested,
        tolerance_pp_applied=tolerance_pp_applied,
        n_calibration_samples=n_calibration_samples,
    )

    shortlist = [
        _find_result(results, int(row["k"]), float(row["alpha"]))
        for row in selection_diagnostics.get("shortlist", [])
    ]
    shortlist = [row for row in shortlist if row is not None]

    # --- Optional pilot gating ---
    pilot_gate_diag: dict[str, Any] | None = None
    gated_shortlist = shortlist
    if args.pilot_report:
        pilot_map = _load_pilot_map(args.pilot_report)
        gated_shortlist, candidate_gate_rows = _apply_pilot_poison_gate(
            shortlist,
            pilot_map=pilot_map,
            attempt_threshold_pp=float(args.pilot_poison_attempt_delta_pp),
            precision_threshold_pp=float(args.pilot_poison_precision_delta_pp),
            not_attempted_threshold_n=int(args.pilot_poison_not_attempted_delta_n),
        )
        pilot_gate_diag = {
            "pilot_report": args.pilot_report,
            "rule": {
                "attempt_delta_pp_lte": float(args.pilot_poison_attempt_delta_pp),
                "precision_delta_pp_lte": float(args.pilot_poison_precision_delta_pp),
                "not_attempted_delta_n_gte": int(
                    args.pilot_poison_not_attempted_delta_n
                ),
                "logic": "reject_if((attempt<=A and precision<=P) or not_attempted_delta>=N)",
            },
            "candidate_gate": candidate_gate_rows,
            "n_shortlist": len(shortlist),
            "n_survivors": len(gated_shortlist),
        }

    # --- Determine locked config ---
    human_override: dict[str, Any] | None = None
    lock_status = "locked"

    if has_k:
        match = _find_result(results, args.K, args.alpha)
        if match is None:
            swept_combos = sorted({(int(r["k"]), float(r["alpha"])) for r in results})
            print(
                f"ERROR: (K={args.K}, α={args.alpha}) not found in sweep results.\n"
                f"Available combos: {swept_combos}",
                file=sys.stderr,
            )
            sys.exit(1)
        locked_result = match
        human_override = {
            "K": int(args.K),
            "alpha": float(args.alpha),
            "reason": args.reason,
        }
    else:
        if not gated_shortlist:
            lock_status = "generation_poisoned_shortlist"
            locked_result = None
        else:
            locked_result = _select_from_shortlist(gated_shortlist)

    locked_entry = (
        {
            "K": int(locked_result["k"]),
            "alpha": float(locked_result["alpha"]),
            "mc1": float(locked_result["mc1"]),
            "mc2": float(locked_result["mc2"]),
        }
        if locked_result is not None
        else None
    )

    artifact_path = sweep_data.get("artifact_path")
    artifact_family = _infer_artifact_family(sweep_data, state_dir, artifact_path)
    extraction_metadata = sweep_data.get("extraction_metadata", {})
    ranking_metric = sweep_data.get("ranking_metric") or infer_ranking_metric(
        artifact_family,
        extraction_metadata,
    )
    position_policy = sweep_data.get("position_policy") or infer_position_policy(
        artifact_family,
        extraction_metadata,
    )
    direction_type = sweep_data.get("direction_type", "mass_mean")
    source_dataset = sweep_data.get("source_dataset")
    answer_token_policy = sweep_data.get(
        "answer_token_policy", "raw_prompt_answer_span"
    )

    # --- Write locked_iti_config.json ---
    locked_config: dict[str, Any] = {
        "model": sweep_data.get("model_path"),
        "artifact_family": artifact_family,
        "source_dataset": source_dataset,
        "direction_type": direction_type,
        "ranking_metric": ranking_metric,
        "position_policy": position_policy,
        "answer_token_policy": answer_token_policy,
        "selection_rule": "mc1_within_tolerance_then_mc2_then_min_alpha_then_min_k",
        "selection_diagnostics": selection_diagnostics,
        "lock_status": lock_status,
        "seed": 42,
        "calibration_artifact_path": artifact_path,
        "artifact_fingerprint": sweep_data.get("artifact_fingerprint"),
        "family_fingerprint": sweep_data.get("family_fingerprint"),
    }
    if locked_result is not None:
        locked_config.update(
            {
                "K_locked": int(locked_result["k"]),
                "alpha_locked": float(locked_result["alpha"]),
                "calibration_mc1": float(locked_result["mc1"]),
                "calibration_mc2": float(locked_result["mc2"]),
                "locked_candidate_summary": {
                    "k": int(locked_result["k"]),
                    "alpha": float(locked_result["alpha"]),
                    "mc1": float(locked_result["mc1"]),
                    "mc2": float(locked_result["mc2"]),
                },
            }
        )

    if pilot_gate_diag is not None:
        locked_config["pilot_gate"] = pilot_gate_diag
    if human_override:
        locked_config["human_override"] = human_override

    locked_path = state_dir / "locked_iti_config.json"
    locked_path.write_text(json.dumps(locked_config, indent=2) + "\n", encoding="utf-8")

    # --- Write pipeline_state.json ---
    now_iso = datetime.now(timezone.utc).isoformat()
    pipeline_state = {
        "seed": 42,
        "gate_1_sweep": {
            "completed_at": now_iso,
            "sweep_results_path": "sweep_results.json",
            "auto_suggestion": auto_suggestion,
            "locked": locked_entry,
            "human_override": human_override,
            "selection_diagnostics": selection_diagnostics,
            "lock_status": lock_status,
        },
    }
    if pilot_gate_diag is not None:
        pipeline_state["gate_1_sweep"]["pilot_gate"] = pilot_gate_diag

    state_path = state_dir / "pipeline_state.json"
    state_path.write_text(json.dumps(pipeline_state, indent=2) + "\n", encoding="utf-8")

    # --- Print confirmation ---
    if lock_status == "generation_poisoned_shortlist" and not human_override:
        print("\n✗ Lock aborted: generation-poisoned shortlist")
        print(f"  shortlist size={len(shortlist)}, survivors=0")
        print(f"  → {locked_path}")
        print(f"  → {state_path}")
        sys.exit(2)

    mode = "HUMAN OVERRIDE" if human_override else "AUTO-SELECTED"
    print(f"\n✓ Locked config ({mode}):")
    if locked_entry is not None:
        print(f"  K={locked_entry['K']}, α={locked_entry['alpha']}")
        print(f"  MC1={locked_entry['mc1']:.4f}, MC2={locked_entry['mc2']:.4f}")
    if human_override:
        print(
            f"  Auto-suggestion was: K={auto_suggestion['K']}, α={auto_suggestion['alpha']}"
        )
        print(f"  Reason: {human_override['reason']}")
    print(f"  → {locked_path}")
    print(f"  → {state_path}")


if __name__ == "__main__":
    main()
