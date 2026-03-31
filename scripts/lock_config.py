#!/usr/bin/env python3
"""Lock ITI hyperparameters into the pipeline state file.

Reads ``sweep_results.json`` from the state directory, applies the
auto-selection rule (or a human override), and writes both
``locked_iti_config.json`` and ``pipeline_state.json``.

Usage::

    # Accept auto-suggestion
    uv run python scripts/lock_config.py \\
        --state_dir data/contrastive/truthfulness/iti_truthfulqa_paperfaithful_calibration

    # Human override
    uv run python scripts/lock_config.py \\
        --state_dir data/contrastive/truthfulness/iti_truthfulqa_paperfaithful_calibration \\
        --K 24 --alpha 6.0 --reason "plateau center, robust across K=16-32"
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from run_calibration_sweep import select_locked_config


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
        default=0.5,
        help="MC1 tolerance for auto-selection (default: 0.5pp)",
    )
    return p.parse_args()


def _find_result(
    results: list[dict[str, Any]], k: int, alpha: float
) -> dict[str, Any] | None:
    """Find the sweep result matching (K, alpha), or None."""
    for r in results:
        if r["k"] == k and r["alpha"] == alpha:
            return r
    return None


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

    with open(sweep_path) as f:
        sweep_data = json.load(f)

    results: list[dict[str, Any]] = sweep_data["results"]

    # --- Auto-suggestion (always computed for the state record) ---
    auto_pick = select_locked_config(results, tolerance_pp=args.tolerance_pp)
    auto_suggestion = {
        "K": auto_pick["k"],
        "alpha": auto_pick["alpha"],
        "mc1": auto_pick["mc1"],
        "mc2": auto_pick["mc2"],
    }

    # --- Determine locked config ---
    human_override: dict[str, Any] | None = None

    if has_k:
        match = _find_result(results, args.K, args.alpha)
        if match is None:
            swept_combos = sorted({(r["k"], r["alpha"]) for r in results})
            print(
                f"ERROR: (K={args.K}, α={args.alpha}) not found in sweep results.\n"
                f"Available combos: {swept_combos}",
                file=sys.stderr,
            )
            sys.exit(1)
        locked_result = match
        human_override = {
            "K": args.K,
            "alpha": args.alpha,
            "reason": args.reason,
        }
    else:
        locked_result = auto_pick

    locked_entry = {
        "K": locked_result["k"],
        "alpha": locked_result["alpha"],
        "mc1": locked_result["mc1"],
        "mc2": locked_result["mc2"],
    }

    # --- Write locked_iti_config.json ---
    locked_config = {
        "model": sweep_data.get("model_path"),
        "direction_type": "mass_mean",
        "ranking_metric": "val_accuracy",
        "K_locked": locked_result["k"],
        "alpha_locked": locked_result["alpha"],
        "selection_rule": (
            f"mc1_primary_mc2_tiebreak_{args.tolerance_pp}pp_"
            "then_smaller_alpha_then_smaller_k"
        ),
        "calibration_mc1": locked_result["mc1"],
        "calibration_mc2": locked_result["mc2"],
        "seed": 42,
        "calibration_artifact_path": sweep_data.get("artifact_path"),
        "artifact_fingerprint": sweep_data.get("artifact_fingerprint"),
    }
    if human_override:
        locked_config["human_override"] = human_override

    locked_path = state_dir / "locked_iti_config.json"
    with open(locked_path, "w") as f:
        json.dump(locked_config, f, indent=2)
        f.write("\n")

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
        },
    }

    state_path = state_dir / "pipeline_state.json"
    with open(state_path, "w") as f:
        json.dump(pipeline_state, f, indent=2)
        f.write("\n")

    # --- Print confirmation ---
    mode = "HUMAN OVERRIDE" if human_override else "AUTO-SELECTED"
    print(f"\n✓ Locked config ({mode}):")
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
