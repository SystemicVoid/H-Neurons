"""Patch results.json files to add delta_noop_to_max_pp (no-op-to-max effect size).

One-shot migration: reads raw JSONL trajectories, calls the updated
paired_bootstrap_curve_effects with noop_alpha=1.0, sanity-checks against
existing effects, and patches results.json in-place.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from scripts.uncertainty import paired_bootstrap_curve_effects

NOOP_ALPHA = 1.0
N_RESAMPLES = 10_000
SEED = 42

EXPERIMENT_DIRS = [
    Path("data/gemma3_4b/intervention/faitheval/experiment"),
    Path("data/gemma3_4b/intervention/falseqa/experiment"),
    Path("data/gemma3_4b/intervention/jailbreak/experiment_256tok_legacy"),
]


def load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def patch_experiment(experiment_dir: Path) -> None:
    results_path = experiment_dir / "results.json"
    with open(results_path) as f:
        results = json.load(f)

    # Discover alphas from JSONL files
    alpha_files = sorted(experiment_dir.glob("alpha_*.jsonl"))
    alphas = [float(p.stem.replace("alpha_", "")) for p in alpha_files]

    # Load rows by alpha
    rows_by_alpha: dict[float, list[dict]] = {}
    for alpha, path in zip(alphas, alpha_files):
        rows_by_alpha[alpha] = load_jsonl(path)

    # Build paired trajectories (same logic as aggregate_results in run_intervention.py)
    reference_ids = {rec["id"] for rec in rows_by_alpha[alphas[0]]}
    for alpha in alphas[1:]:
        current_ids = {rec["id"] for rec in rows_by_alpha[alpha]}
        if current_ids != reference_ids:
            raise ValueError(
                f"{experiment_dir}: sample IDs mismatch between "
                f"alpha={alphas[0]} and alpha={alpha}"
            )

    rows_indexed: dict[float, dict[str, dict]] = {
        alpha: {rec["id"]: rec for rec in recs} for alpha, recs in rows_by_alpha.items()
    }

    trajectories = np.array(
        [
            [
                bool(rows_indexed[alpha][sample_id].get("compliance", False))
                for alpha in alphas
            ]
            for sample_id in sorted(reference_ids)
        ],
        dtype=bool,
    )

    # Compute effects with noop_alpha
    effects = paired_bootstrap_curve_effects(
        trajectories,
        np.array(alphas, dtype=float),
        noop_alpha=NOOP_ALPHA,
        n_resamples=N_RESAMPLES,
        seed=SEED,
    )

    # Sanity check: existing values must match (not assert — survives python -O)
    existing = results["effects"]["compliance_curve"]
    if not math.isclose(
        existing["delta_0_to_max_pp"]["estimate"],
        effects["delta_0_to_max_pp"]["estimate"],
        abs_tol=1e-6,
    ):
        raise ValueError(
            f"{experiment_dir}: delta_0_to_max mismatch: "
            f"existing={existing['delta_0_to_max_pp']['estimate']}, "
            f"recomputed={effects['delta_0_to_max_pp']['estimate']}"
        )
    if not math.isclose(
        existing["slope_pp_per_alpha"]["estimate"],
        effects["slope_pp_per_alpha"]["estimate"],
        abs_tol=1e-6,
    ):
        raise ValueError(
            f"{experiment_dir}: slope mismatch: "
            f"existing={existing['slope_pp_per_alpha']['estimate']}, "
            f"recomputed={effects['slope_pp_per_alpha']['estimate']}"
        )

    # Patch: add delta_noop_to_max_pp
    results["effects"]["compliance_curve"]["delta_noop_to_max_pp"] = effects[
        "delta_noop_to_max_pp"
    ]

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
        f.write("\n")

    noop_est = effects["delta_noop_to_max_pp"]["estimate"]
    noop_ci = effects["delta_noop_to_max_pp"]["ci"]
    print(
        f"  {experiment_dir.parent.name}: "
        f"delta_noop_to_max = {noop_est:+.1f} pp "
        f"[{noop_ci['lower']:.1f}, {noop_ci['upper']:.1f}]"
    )


def main() -> None:
    print("Patching results.json files with delta_noop_to_max_pp (noop_alpha=1.0)...")
    for experiment_dir in EXPERIMENT_DIRS:
        if not experiment_dir.exists():
            print(f"  SKIP (not found): {experiment_dir}")
            continue
        patch_experiment(experiment_dir)
    print("Done.")


if __name__ == "__main__":
    main()
