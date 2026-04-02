#!/usr/bin/env python3
"""Sweep K × α on TruthfulQA calibration-val to lock ITI hyperparameters.

Loads the model once and iterates over all (K, α) combinations, scoring
MC1 and MC2 on the calibration validation set.  Writes a sweep table and
a ``locked_iti_config.json`` using the selection rule:

1. Best MC1
2. Among ties within tolerance: best MC2
3. Smaller α
4. Smaller K

Usage::

    uv run python scripts/run_calibration_sweep.py \\
        --artifact_path data/contrastive/truthfulness/iti_truthfulqa_paperfaithful_calibration/iti_heads.pt \\
        --cal_val_mc1_manifest data/manifests/truthfulqa_cal_val_mc1_seed42.json \\
        --cal_val_mc2_manifest data/manifests/truthfulqa_cal_val_mc2_seed42.json \\
        --k_values 8 12 16 24 32 40 \\
        --alpha_values 0.0 0.5 1.0 2.0 4.0 6.0 8.0 12.0 16.0 \\
        --output_dir data/contrastive/truthfulness/iti_truthfulqa_paperfaithful_calibration
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from intervene_iti import ITIHeadScaler, load_iti_artifact
from run_intervention import (
    _reset_scaler_sample_stats,
    load_truthfulqa_mc,
    score_continuation_decode_only,
)
from utils import (
    finish_run_provenance,
    fingerprint_ids,
    start_run_provenance,
)

DEFAULT_MODEL = os.environ.get("HNEURONS_MODEL_PATH", "google/gemma-3-4b-it")
MIN_TOLERANCE_PP = 1.5


# ---------------------------------------------------------------------------
# MC scoring helpers
# ---------------------------------------------------------------------------


def _truthfulqa_prompt(sample: dict[str, Any]) -> str:
    return f"Question: {sample['question']}\nAnswer:"


def score_mc_samples(
    model: torch.nn.Module,
    tokenizer: Any,
    scaler: ITIHeadScaler,
    samples: list[dict[str, Any]],
    alpha: float,
) -> list[dict[str, Any]]:
    """Score all MC samples at a given alpha.  Returns per-sample records."""
    scaler.alpha = alpha
    records = []
    for sample in samples:
        _reset_scaler_sample_stats(scaler)
        prompt = _truthfulqa_prompt(sample)
        score_values = []
        for choice in sample["choices"]:
            scores = score_continuation_decode_only(
                model, tokenizer, prompt, f" {choice}", scaler=scaler
            )
            score_values.append(scores["log_likelihood"])

        scores_arr = np.array(score_values, dtype=np.float64)
        best_idx = int(np.argmax(scores_arr))
        truthful_indices = [
            i for i, label in enumerate(sample["labels"]) if int(label) == 1
        ]
        mc1_correct = best_idx in truthful_indices

        # MC2: truthful mass
        totals = np.exp(scores_arr)
        denom = float(np.sum(totals)) if totals.size else 1.0
        truthful_mass = (
            float(np.sum(totals[truthful_indices]) / denom) if denom > 0 else 0.0
        )

        records.append(
            {
                "id": sample["id"],
                "mc1_correct": mc1_correct,
                "truthful_mass": truthful_mass,
            }
        )
    return records


# ---------------------------------------------------------------------------
# Selection rule
# ---------------------------------------------------------------------------


def _resolve_tolerance_pp(
    *,
    requested_tolerance_pp: float | None,
    n_calibration_samples: int,
) -> dict[str, Any]:
    if n_calibration_samples <= 0:
        raise ValueError("n_calibration_samples must be positive")
    mc1_resolution_pp = 100.0 / n_calibration_samples
    tolerance_floor_pp = max(mc1_resolution_pp, MIN_TOLERANCE_PP)
    if requested_tolerance_pp is None:
        applied_tolerance_pp = tolerance_floor_pp
        raised_to_floor = False
    else:
        applied_tolerance_pp = max(float(requested_tolerance_pp), tolerance_floor_pp)
        raised_to_floor = bool(requested_tolerance_pp < tolerance_floor_pp)
    return {
        "mc1_resolution_pp": float(mc1_resolution_pp),
        "tolerance_floor_pp": float(tolerance_floor_pp),
        "tolerance_pp_requested": (
            float(requested_tolerance_pp)
            if requested_tolerance_pp is not None
            else None
        ),
        "tolerance_pp_applied": float(applied_tolerance_pp),
        "tolerance_raised_to_resolution_floor": raised_to_floor,
    }


def _candidate_brief(entry: dict[str, Any], *, best_mc1: float) -> dict[str, Any]:
    return {
        "k": int(entry["k"]),
        "alpha": float(entry["alpha"]),
        "mc1": float(entry["mc1"]),
        "mc2": float(entry["mc2"]),
        "n_mc1": int(entry["n_mc1"]),
        "n_mc2": int(entry["n_mc2"]),
        "mc1_gap_pp_from_best": round((best_mc1 - float(entry["mc1"])) * 100.0, 6),
    }


def _shortlist_candidates(
    sweep_results: list[dict[str, Any]],
    *,
    tolerance_pp: float,
) -> tuple[float, list[dict[str, Any]]]:
    if not sweep_results:
        raise ValueError("No sweep results to select from")
    best_mc1 = max(float(r["mc1"]) for r in sweep_results)
    shortlist = [
        r for r in sweep_results if (best_mc1 - float(r["mc1"])) * 100.0 <= tolerance_pp
    ]
    shortlist.sort(
        key=lambda r: (-float(r["mc1"]), -float(r["mc2"]), r["alpha"], r["k"])
    )
    return best_mc1, shortlist


def _select_with_trace(
    shortlist: list[dict[str, Any]],
    *,
    best_mc1: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not shortlist:
        raise ValueError("Shortlist is empty")

    current = list(shortlist)
    tie_break_path: list[dict[str, Any]] = []
    eps = 1e-12

    best_mc2 = max(float(r["mc2"]) for r in current)
    current = [r for r in current if abs(float(r["mc2"]) - best_mc2) <= eps]
    tie_break_path.append(
        {
            "step": "max_mc2",
            "criterion": "maximize_mc2",
            "selected_value": round(best_mc2, 6),
            "n_survivors": len(current),
            "survivors": [_candidate_brief(r, best_mc1=best_mc1) for r in current],
        }
    )

    min_alpha = min(float(r["alpha"]) for r in current)
    current = [r for r in current if abs(float(r["alpha"]) - min_alpha) <= eps]
    tie_break_path.append(
        {
            "step": "min_alpha",
            "criterion": "minimize_alpha",
            "selected_value": round(min_alpha, 6),
            "n_survivors": len(current),
            "survivors": [_candidate_brief(r, best_mc1=best_mc1) for r in current],
        }
    )

    min_k = min(int(r["k"]) for r in current)
    current = [r for r in current if int(r["k"]) == min_k]
    tie_break_path.append(
        {
            "step": "min_k",
            "criterion": "minimize_k",
            "selected_value": int(min_k),
            "n_survivors": len(current),
            "survivors": [_candidate_brief(r, best_mc1=best_mc1) for r in current],
        }
    )

    current.sort(key=lambda r: (int(r["k"]), float(r["alpha"])))
    return current[0], tie_break_path


def select_locked_config(
    sweep_results: list[dict[str, Any]],
    tolerance_pp: float,
) -> dict[str, Any]:
    """Apply the locking rule and return the locked candidate."""
    best_mc1, shortlist = _shortlist_candidates(
        sweep_results, tolerance_pp=float(tolerance_pp)
    )
    selected, _ = _select_with_trace(shortlist, best_mc1=best_mc1)
    return selected


def infer_ranking_metric(
    family: str | None, extraction_metadata: dict[str, Any]
) -> str:
    """Infer the primary head-ranking metric used for the artifact family."""
    rank_from_meta = extraction_metadata.get(
        "ranking_metric", extraction_metadata.get("head_ranking_metric")
    )
    if rank_from_meta:
        return str(rank_from_meta)
    if family == "iti_truthfulqa_paperfaithful":
        return "val_accuracy"
    return "auroc"


def infer_position_policy(
    family: str | None,
    extraction_metadata: dict[str, Any],
) -> str:
    value = extraction_metadata.get("position_policy")
    if value:
        return str(value)
    if family == "iti_truthfulqa_paperfaithful":
        return "last_answer_token"
    return "all_answer_positions"


def compute_selection_diagnostics(
    sweep_results: list[dict[str, Any]],
    *,
    tolerance_pp_requested: float | None,
    tolerance_pp_applied: float,
    n_calibration_samples: int,
) -> dict[str, Any]:
    """Compute audit metadata for lock selection behavior."""
    if not sweep_results:
        return {
            "tolerance_pp_requested": tolerance_pp_requested,
            "tolerance_pp_applied": tolerance_pp_applied,
            "n_candidates_within_tolerance_pp": 0,
            "tie_break_triggered": False,
        }

    best_mc1, shortlist = _shortlist_candidates(
        sweep_results, tolerance_pp=float(tolerance_pp_applied)
    )
    selected, tie_break_path = _select_with_trace(shortlist, best_mc1=best_mc1)
    tolerance_resolution = _resolve_tolerance_pp(
        requested_tolerance_pp=tolerance_pp_requested,
        n_calibration_samples=n_calibration_samples,
    )

    diagnostics: dict[str, Any] = {
        "selection_rule": "mc1_within_tolerance_then_mc2_then_min_alpha_then_min_k",
        "tolerance_pp_requested": tolerance_pp_requested,
        "tolerance_pp_applied": tolerance_pp_applied,
        "tolerance_floor_rule": "max(100/n_cal, 1.5)",
        "mc1_resolution_pp": tolerance_resolution["mc1_resolution_pp"],
        "n_calibration_samples": int(n_calibration_samples),
        "best_mc1": round(best_mc1, 6),
        "n_candidates_within_tolerance_pp": len(shortlist),
        "tie_break_triggered": len(shortlist) > 1,
        "shortlist": [_candidate_brief(r, best_mc1=best_mc1) for r in shortlist],
        "tie_break_path": tie_break_path,
        "locked_candidate_summary": _candidate_brief(selected, best_mc1=best_mc1),
        "tolerance_raised_to_resolution_floor": tolerance_resolution[
            "tolerance_raised_to_resolution_floor"
        ],
    }
    return diagnostics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--artifact_path", type=str, required=True)
    p.add_argument("--cal_val_mc1_manifest", type=str, required=True)
    p.add_argument("--cal_val_mc2_manifest", type=str, required=True)
    p.add_argument("--k_values", type=int, nargs="+", default=[8, 12, 16, 24, 32, 40])
    p.add_argument(
        "--alpha_values",
        type=float,
        nargs="+",
        default=[0.0, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 16.0],
    )
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--model_path", type=str, default=DEFAULT_MODEL)
    p.add_argument("--device_map", type=str, default="cuda:0")
    p.add_argument(
        "--tolerance_pp",
        type=float,
        default=None,
        help=(
            "Optional MC1 shortlist tolerance in pp. Applied tolerance is always "
            "max(requested, 100/n_cal, 1.5)."
        ),
    )
    p.add_argument(
        "--truthfulqa_csv",
        type=str,
        default="data/benchmarks/TruthfulQA.csv",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    prov = start_run_provenance(
        args,
        primary_target=str(out / "locked_iti_config.json"),
        output_targets=[
            str(out / "locked_iti_config.json"),
            str(out / "sweep_results.json"),
        ],
    )

    # Load manifests (sample ID filter lists)
    with open(args.cal_val_mc1_manifest) as f:
        mc1_filter_ids = set(json.load(f))
    with open(args.cal_val_mc2_manifest) as f:
        mc2_filter_ids = set(json.load(f))

    # Load samples
    all_mc1 = load_truthfulqa_mc("mc1", csv_path=args.truthfulqa_csv)
    all_mc2 = load_truthfulqa_mc("mc2", csv_path=args.truthfulqa_csv)
    mc1_samples = [s for s in all_mc1 if s["id"] in mc1_filter_ids]
    mc2_samples = [s for s in all_mc2 if s["id"] in mc2_filter_ids]
    print(f"Cal-val: {len(mc1_samples)} MC1 samples, {len(mc2_samples)} MC2 samples")

    # Load model
    print(f"Loading model: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=torch.bfloat16, device_map=args.device_map
    )
    model.eval()
    device = next(model.parameters()).device

    # Load artifact
    artifact = load_iti_artifact(args.artifact_path)
    max_k = max(args.k_values)
    n_ranked = len(artifact.get("ranked_heads", []))
    if max_k > n_ranked:
        print(
            f"WARNING: max K={max_k} exceeds ranked heads ({n_ranked}). "
            f"Clamping K values."
        )
        args.k_values = [k for k in args.k_values if k <= n_ranked]

    artifact_family = artifact.get("family")

    extraction_metadata: dict[str, Any] = {}
    extraction_meta_path = Path(args.artifact_path).with_name(
        "extraction_metadata.json"
    )
    if extraction_meta_path.exists():
        extraction_metadata = json.loads(
            extraction_meta_path.read_text(encoding="utf-8")
        )
        meta_family = extraction_metadata.get(
            "artifact_family"
        ) or extraction_metadata.get("family")
        if meta_family and artifact_family and str(meta_family) != str(artifact_family):
            raise ValueError(
                "Artifact family mismatch between iti_heads.pt and extraction metadata: "
                f"{artifact_family!r} vs {meta_family!r}"
            )

    ranking_metric = infer_ranking_metric(artifact_family, extraction_metadata)
    position_policy = infer_position_policy(artifact_family, extraction_metadata)
    answer_token_policy = extraction_metadata.get(
        "answer_token_policy", "raw_prompt_answer_span"
    )
    source_dataset = extraction_metadata.get("source_dataset")
    direction_type = extraction_metadata.get(
        "direction_type", artifact.get("steering_direction_type", "mass_mean")
    )

    # Artifact fingerprint
    with open(args.artifact_path, "rb") as f:
        artifact_fp = hashlib.sha256(f.read()).hexdigest()[:16]

    # Sweep
    sweep_results: list[dict[str, Any]] = []
    total_combos = len(args.k_values) * len(args.alpha_values)

    # Resume from partial progress
    progress_path = out / "sweep_progress.jsonl"
    completed_combos: set[tuple[int, float]] = set()
    if progress_path.exists():
        with open(progress_path) as f:
            for line in f:
                rec = json.loads(line)
                completed_combos.add((rec["k"], rec["alpha"]))
                sweep_results.append(rec)
        print(f"Resumed {len(completed_combos)} combos from {progress_path}")

    print(
        f"Sweeping {len(args.k_values)} K × {len(args.alpha_values)} α = {total_combos} combos"
    )
    combo_idx = 0
    scaler = None

    for k in args.k_values:
        # Remove previous hooks before building scaler for this K
        if scaler is not None:
            scaler.remove()
        scaler = ITIHeadScaler(
            model,
            artifact,
            device,
            family=artifact.get("family"),
            k=k,
            selection_strategy="ranked",
        )

        for alpha in args.alpha_values:
            combo_idx += 1
            label = f"[{combo_idx}/{total_combos}] K={k}, α={alpha}"

            if (k, alpha) in completed_combos:
                print(f"  {label}: skipped (resumed)")
                continue

            # Score MC1
            mc1_records = score_mc_samples(model, tokenizer, scaler, mc1_samples, alpha)
            mc1_acc = sum(r["mc1_correct"] for r in mc1_records) / max(
                len(mc1_records), 1
            )

            # Score MC2
            mc2_records = score_mc_samples(model, tokenizer, scaler, mc2_samples, alpha)
            mc2_mass = sum(r["truthful_mass"] for r in mc2_records) / max(
                len(mc2_records), 1
            )

            result = {
                "k": k,
                "alpha": alpha,
                "mc1": round(mc1_acc, 6),
                "mc2": round(mc2_mass, 6),
                "n_mc1": len(mc1_records),
                "n_mc2": len(mc2_records),
            }
            sweep_results.append(result)
            with open(progress_path, "a") as pf:
                pf.write(json.dumps(result) + "\n")
            print(f"  {label}: MC1={mc1_acc:.3f}, MC2={mc2_mass:.3f}")

    # Clean up final scaler hooks
    if scaler is not None:
        scaler.remove()

    # Select locked config
    tolerance_resolution = _resolve_tolerance_pp(
        requested_tolerance_pp=args.tolerance_pp,
        n_calibration_samples=len(mc1_samples),
    )
    if tolerance_resolution["tolerance_raised_to_resolution_floor"]:
        floor_pp = float(tolerance_resolution["tolerance_floor_pp"])
        print(
            "NOTE: requested tolerance_pp was below the resolution-aware floor; "
            f"raised to {floor_pp:.3f}pp."
        )

    selection_diagnostics = compute_selection_diagnostics(
        sweep_results,
        tolerance_pp_requested=args.tolerance_pp,
        tolerance_pp_applied=float(tolerance_resolution["tolerance_pp_applied"]),
        n_calibration_samples=len(mc1_samples),
    )
    locked = select_locked_config(
        sweep_results, tolerance_pp=float(tolerance_resolution["tolerance_pp_applied"])
    )

    # Write sweep table
    sweep_path = out / "sweep_results.json"
    with open(sweep_path, "w") as f:
        json.dump(
            {
                "model_path": args.model_path,
                "artifact_family": artifact_family,
                "source_dataset": source_dataset,
                "direction_type": direction_type,
                "ranking_metric": ranking_metric,
                "position_policy": position_policy,
                "answer_token_policy": answer_token_policy,
                "k_values": args.k_values,
                "alpha_values": args.alpha_values,
                "results": sweep_results,
                "selection_diagnostics": selection_diagnostics,
                "locked_candidate": {
                    "k": locked["k"],
                    "alpha": locked["alpha"],
                    "mc1": locked["mc1"],
                    "mc2": locked["mc2"],
                },
                "artifact_path": args.artifact_path,
                "artifact_fingerprint": artifact_fp,
                "extraction_metadata": extraction_metadata,
                "family_fingerprint": {
                    "artifact_family": artifact_family,
                    "source_dataset": source_dataset,
                    "ranking_metric": ranking_metric,
                    "position_policy": position_policy,
                    "answer_token_policy": answer_token_policy,
                    "direction_type": direction_type,
                },
                "n_mc1_samples": len(mc1_samples),
                "n_mc2_samples": len(mc2_samples),
            },
            f,
            indent=2,
        )
        f.write("\n")
    print(f"\nSweep table → {sweep_path}")

    # Print table
    print("\n" + "=" * 70)
    print(f"{'K':>4}  {'α':>6}  {'MC1':>8}  {'MC2':>8}")
    print("-" * 70)
    for r in sweep_results:
        print(f"{r['k']:>4}  {r['alpha']:>6.1f}  {r['mc1']:>8.4f}  {r['mc2']:>8.4f}")
    print("=" * 70)

    locked_config = {
        "model": args.model_path,
        "dataset_fingerprint": fingerprint_ids(
            [s["id"] for s in mc1_samples + mc2_samples]
        ),
        "artifact_family": artifact_family,
        "source_dataset": source_dataset,
        "direction_type": direction_type,
        "ranking_metric": ranking_metric,
        "position_policy": position_policy,
        "answer_token_policy": answer_token_policy,
        "K_locked": locked["k"],
        "alpha_locked": locked["alpha"],
        "selection_rule": "mc1_within_tolerance_then_mc2_then_min_alpha_then_min_k",
        "selection_diagnostics": selection_diagnostics,
        "calibration_mc1": locked["mc1"],
        "calibration_mc2": locked["mc2"],
        "seed": 42,
        "calibration_artifact_path": args.artifact_path,
        "artifact_fingerprint": artifact_fp,
        "family_fingerprint": {
            "artifact_family": artifact_family,
            "source_dataset": source_dataset,
            "ranking_metric": ranking_metric,
            "position_policy": position_policy,
            "answer_token_policy": answer_token_policy,
            "direction_type": direction_type,
        },
    }

    locked_path = out / "locked_iti_config.json"
    with open(locked_path, "w") as f:
        json.dump(locked_config, f, indent=2)
        f.write("\n")

    print(f"\nLocked config: K={locked['k']}, α={locked['alpha']}")
    print(f"  MC1={locked['mc1']:.4f}, MC2={locked['mc2']:.4f}")
    print(f"  → {locked_path}")

    finish_run_provenance(
        prov, status="completed", extra={"locked_config": locked_config}
    )
    print("Done.")


if __name__ == "__main__":
    main()
