"""Compute paired bootstrap slope differences for FaithEval intervention comparisons.

Produces interaction-style statistical tests for:
  1. Neuron vs SAE (anchor result, §4.2)
  2. Neuron vs each random-neuron control seed (specificity, §4.2)

Outputs structured JSON with slope differences, CIs, and permutation p-values.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    from uncertainty import (
        DEFAULT_BOOTSTRAP_RESAMPLES,
        DEFAULT_BOOTSTRAP_SEED,
        paired_bootstrap_slope_difference,
    )
except ModuleNotFoundError:
    from scripts.uncertainty import (
        DEFAULT_BOOTSTRAP_RESAMPLES,
        DEFAULT_BOOTSTRAP_SEED,
        paired_bootstrap_slope_difference,
    )

PROJECT_ROOT = Path(__file__).resolve().parent.parent

NEURON_DIR = PROJECT_ROOT / "data/gemma3_4b/intervention/faitheval/experiment"
SAE_DIR = PROJECT_ROOT / "data/gemma3_4b/intervention/faitheval_sae/experiment"
CONTROL_BASE = PROJECT_ROOT / "data/gemma3_4b/intervention/faitheval/control"

ALPHAS = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

CONTROL_SEEDS = [
    "seed_0_unconstrained",
    "seed_1_unconstrained",
    "seed_2_unconstrained",
    "seed_3_unconstrained",
    "seed_4_unconstrained",
    "seed_0_layer_matched",
    "seed_1_layer_matched",
    "seed_2_layer_matched",
]


def load_trajectories(
    data_dir: Path, alphas: list[float]
) -> tuple[list[str], np.ndarray]:
    """Load per-item compliance across alphas, returning (item_ids, trajectories).

    Returns trajectories as a bool array of shape (n_items, n_alphas).
    """
    items_by_alpha: list[dict[str, bool]] = []
    for alpha in alphas:
        path = data_dir / f"alpha_{alpha:.1f}.jsonl"
        by_id: dict[str, bool] = {}
        with open(path) as f:
            for line in f:
                rec = json.loads(line)
                by_id[rec["id"]] = bool(rec["compliance"])
        items_by_alpha.append(by_id)

    # Use first alpha's ID order as canonical
    item_ids = list(items_by_alpha[0].keys())
    n_items = len(item_ids)

    # Verify all alphas have the same items
    for i, by_id in enumerate(items_by_alpha):
        if set(by_id.keys()) != set(item_ids):
            raise ValueError(
                f"Item ID mismatch at alpha={alphas[i]}: "
                f"expected {n_items} items, got {len(by_id)}"
            )

    trajectories = np.empty((n_items, len(alphas)), dtype=bool)
    for col, by_id in enumerate(items_by_alpha):
        for row, item_id in enumerate(item_ids):
            trajectories[row, col] = by_id[item_id]

    return item_ids, trajectories


def align_trajectories(
    reference_ids: list[str],
    other_ids: list[str],
    other_traj: np.ndarray,
) -> np.ndarray | None:
    """Reorder *other_traj* rows to match *reference_ids* ordering.

    Returns the reordered array, or None if alignment fails (partial overlap).
    """
    if other_ids == reference_ids:
        return other_traj
    other_lookup = {oid: idx for idx, oid in enumerate(other_ids)}
    common = [i for i in reference_ids if i in other_lookup]
    if len(common) != len(reference_ids):
        return None
    reorder = [other_lookup[i] for i in reference_ids]
    return other_traj[reorder]


def compute_neuron_vs_sae(
    n_resamples: int,
    seed: int,
    permutation_resamples: int,
) -> dict:
    """Compute neuron-minus-SAE slope difference on FaithEval."""
    neuron_ids, neuron_traj = load_trajectories(NEURON_DIR, ALPHAS)
    sae_ids, sae_traj = load_trajectories(SAE_DIR, ALPHAS)

    aligned = align_trajectories(neuron_ids, sae_ids, sae_traj)
    if aligned is None:
        raise ValueError("Neuron and SAE item IDs do not fully overlap")
    sae_traj = aligned

    alphas_arr = np.array(ALPHAS, dtype=float)
    result = paired_bootstrap_slope_difference(
        neuron_traj,
        sae_traj,
        alphas_arr,
        n_resamples=n_resamples,
        seed=seed,
        permutation_resamples=permutation_resamples,
    )
    result["comparison"] = "neuron_minus_sae"
    result["n_items"] = len(neuron_ids)
    result["alphas"] = ALPHAS
    return result


def compute_neuron_vs_random(
    n_resamples: int,
    seed: int,
    permutation_resamples: int,
) -> dict:
    """Compute neuron-minus-random slope difference for each control seed."""
    neuron_ids, neuron_traj = load_trajectories(NEURON_DIR, ALPHAS)
    alphas_arr = np.array(ALPHAS, dtype=float)

    per_seed = {}
    missing_dirs: list[str] = []
    misaligned_ids: list[str] = []
    for seed_name in CONTROL_SEEDS:
        seed_dir = CONTROL_BASE / seed_name
        if not seed_dir.exists():
            missing_dirs.append(seed_name)
            continue

        ctrl_ids, ctrl_traj = load_trajectories(seed_dir, ALPHAS)
        aligned = align_trajectories(neuron_ids, ctrl_ids, ctrl_traj)
        if aligned is None:
            misaligned_ids.append(seed_name)
            continue
        ctrl_traj = aligned

        result = paired_bootstrap_slope_difference(
            neuron_traj,
            ctrl_traj,
            alphas_arr,
            n_resamples=n_resamples,
            seed=seed,
            permutation_resamples=permutation_resamples,
        )
        result["comparison"] = f"neuron_minus_{seed_name}"
        result["n_items"] = len(neuron_ids)
        per_seed[seed_name] = result

    problems: list[str] = []
    if missing_dirs:
        problems.append(
            f"missing control directories: {', '.join(sorted(missing_dirs))}"
        )
    if misaligned_ids:
        problems.append(
            "control trajectories without full item overlap: "
            f"{', '.join(sorted(misaligned_ids))}"
        )
    if problems:
        raise ValueError(
            "Cannot compute FaithEval control slope differences because "
            + "; ".join(problems)
            + "."
        )

    # Aggregate across seeds
    diffs = [s["slope_difference_pp_per_alpha"]["estimate"] for s in per_seed.values()]
    summary = {
        "per_seed": per_seed,
        "aggregate": {
            "mean_slope_difference": float(np.mean(diffs)),
            "min_slope_difference": float(np.min(diffs)),
            "max_slope_difference": float(np.max(diffs)),
            "n_seeds": len(diffs),
        },
        "alphas": ALPHAS,
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n-resamples",
        type=int,
        default=DEFAULT_BOOTSTRAP_RESAMPLES,
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    parser.add_argument(
        "--permutation-resamples",
        type=int,
        default=50_000,
    )
    args = parser.parse_args()

    print("=== Neuron vs SAE slope difference ===")
    neuron_sae = compute_neuron_vs_sae(
        args.n_resamples, args.seed, args.permutation_resamples
    )
    diff = neuron_sae["slope_difference_pp_per_alpha"]
    print(f"  Neuron slope: {neuron_sae['slope_a_pp_per_alpha']:+.2f} pp/α")
    print(f"  SAE slope:    {neuron_sae['slope_b_pp_per_alpha']:+.2f} pp/α")
    print(
        f"  Difference:   {diff['estimate']:+.2f} pp/α "
        f"[{diff['ci']['lower']:+.2f}, {diff['ci']['upper']:+.2f}]"
    )
    if "permutation_test" in neuron_sae:
        pt = neuron_sae["permutation_test"]
        print(
            f"  Permutation p = {pt['p_value']:.4f} "
            f"({pt['n_extreme']}/{pt['n_permutations']})"
        )

    out_sae = (
        PROJECT_ROOT
        / "data/gemma3_4b/intervention/faitheval_sae/control/slope_difference_summary.json"
    )
    out_sae.parent.mkdir(parents=True, exist_ok=True)
    with open(out_sae, "w") as f:
        json.dump(neuron_sae, f, indent=2)
    print(f"  Saved: {out_sae}")

    print("\n=== Neuron vs random-neuron slope differences ===")
    neuron_random = compute_neuron_vs_random(
        args.n_resamples, args.seed, args.permutation_resamples
    )
    agg = neuron_random["aggregate"]
    print(
        f"  Mean diff: {agg['mean_slope_difference']:+.2f} pp/α "
        f"(range: {agg['min_slope_difference']:+.2f} to {agg['max_slope_difference']:+.2f}, "
        f"n={agg['n_seeds']} seeds)"
    )
    for name, s in neuron_random["per_seed"].items():
        d = s["slope_difference_pp_per_alpha"]
        p_str = ""
        if "permutation_test" in s:
            p_str = f", p={s['permutation_test']['p_value']:.4f}"
        print(
            f"    {name}: {d['estimate']:+.2f} "
            f"[{d['ci']['lower']:+.2f}, {d['ci']['upper']:+.2f}]{p_str}"
        )

    out_ctrl = (
        PROJECT_ROOT
        / "data/gemma3_4b/intervention/faitheval/control/slope_difference_summary.json"
    )
    with open(out_ctrl, "w") as f:
        json.dump(neuron_random, f, indent=2)
    print(f"  Saved: {out_ctrl}")


if __name__ == "__main__":
    main()
