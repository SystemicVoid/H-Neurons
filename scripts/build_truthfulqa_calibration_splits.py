#!/usr/bin/env python3
"""Build calibration, final 2-fold CV, and production split manifests.

This extends ``build_truthfulqa_splits.py`` with a 3-way partition that
separates hyperparameter search from final evaluation:

1. **Calibration** (10%+10%): cal_train + cal_val — used to lock K and α.
2. **Final 2-fold CV** from the remaining 80%: produces the honest TruthfulQA
   report with locked hyperparameters.
3. **Production fold**: all 817 questions as dev (4:1 train/val) for building
   the strongest downstream artifact.

Usage::

    uv run python scripts/build_truthfulqa_calibration_splits.py --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any

from build_truthfulqa_splits import (
    build_canonical_manifest,
    build_mc_manifests,
)
from utils import fingerprint_ids


# ---------------------------------------------------------------------------
# Split builders
# ---------------------------------------------------------------------------


def build_calibration_split(
    questions: list[dict[str, Any]],
    seed: int,
    cal_frac: float = 0.10,
) -> dict[str, Any]:
    """Partition questions into calibration (cal_frac × 2) and remainder.

    Returns a dict with ``cal_train``, ``cal_val``, ``remainder`` stable IDs
    and counts.  The partition is deterministic for a given seed.
    """
    stable_ids = sorted(q["stable_id"] for q in questions)
    rng = random.Random(seed)
    shuffled = list(stable_ids)
    rng.shuffle(shuffled)

    n_cal = int(len(shuffled) * cal_frac)
    cal_train = sorted(shuffled[:n_cal])
    cal_val = sorted(shuffled[n_cal : 2 * n_cal])
    remainder = sorted(shuffled[2 * n_cal :])

    # Verify disjointness and coverage
    all_ids = set(stable_ids)
    assert set(cal_train).isdisjoint(set(cal_val)), "cal_train ∩ cal_val"
    assert set(cal_train).isdisjoint(set(remainder)), "cal_train ∩ remainder"
    assert set(cal_val).isdisjoint(set(remainder)), "cal_val ∩ remainder"
    assert set(cal_train) | set(cal_val) | set(remainder) == all_ids, "coverage"

    return {
        "version": 1,
        "seed": seed,
        "cal_frac": cal_frac,
        "cal_train": cal_train,
        "cal_val": cal_val,
        "remainder": remainder,
        "counts": {
            "cal_train": len(cal_train),
            "cal_val": len(cal_val),
            "remainder": len(remainder),
            "total": len(questions),
        },
    }


def build_calibration_fold(
    cal_split: dict[str, Any],
    canonical_manifest_path: str,
    canonical_fingerprint: str,
) -> dict[str, Any]:
    """Build a pseudo-fold file for extraction from the calibration partition.

    Formatted identically to the standard fold files so that
    ``extract_truthfulness_iti.py`` can consume it unchanged.
    """
    return {
        "version": 1,
        "seed": cal_split["seed"],
        "fold": "calibration",
        "dev": {
            "train": cal_split["cal_train"],
            "val": cal_split["cal_val"],
        },
        "test": cal_split["remainder"],
        "counts": {
            "train": cal_split["counts"]["cal_train"],
            "val": cal_split["counts"]["cal_val"],
            "test": cal_split["counts"]["remainder"],
        },
        "canonical_manifest": canonical_manifest_path,
        "canonical_fingerprint": canonical_fingerprint,
    }


def build_final_folds(
    remainder_ids: list[str],
    seed: int,
) -> list[dict[str, Any]]:
    """Build 2-fold CV splits from the remainder questions only.

    Same logic as ``build_truthfulqa_splits.build_folds`` but operates on
    a subset of questions.  Fold symmetry and coverage assertions apply to
    the remainder, not the full 817.
    """
    rng = random.Random(seed)
    shuffled = list(sorted(remainder_ids))
    rng.shuffle(shuffled)

    mid = len(shuffled) // 2
    fold0_dev_ids = shuffled[:mid]
    fold1_dev_ids = shuffled[mid:]

    folds = []
    for fold_idx, (dev_ids, test_ids) in enumerate(
        [(fold0_dev_ids, fold1_dev_ids), (fold1_dev_ids, fold0_dev_ids)]
    ):
        n_val = len(dev_ids) // 5
        fold_rng = random.Random(seed + fold_idx)
        dev_shuffled = list(dev_ids)
        fold_rng.shuffle(dev_shuffled)
        val_ids = sorted(dev_shuffled[:n_val])
        train_ids = sorted(dev_shuffled[n_val:])

        folds.append(
            {
                "version": 1,
                "seed": seed,
                "fold": fold_idx,
                "dev": {"train": train_ids, "val": val_ids},
                "test": sorted(test_ids),
                "counts": {
                    "train": len(train_ids),
                    "val": len(val_ids),
                    "test": len(test_ids),
                },
            }
        )

    # Verify fold symmetry
    assert set(folds[0]["test"]) == set(folds[1]["dev"]["train"]) | set(
        folds[1]["dev"]["val"]
    ), "fold0.test != fold1.dev"
    assert set(folds[1]["test"]) == set(folds[0]["dev"]["train"]) | set(
        folds[0]["dev"]["val"]
    ), "fold1.test != fold0.dev"

    # Verify no overlap
    for f in folds:
        dev_all = set(f["dev"]["train"]) | set(f["dev"]["val"])
        test_set = set(f["test"])
        assert dev_all.isdisjoint(test_set), f"Fold {f['fold']}: dev ∩ test overlap"
        assert set(f["dev"]["train"]).isdisjoint(set(f["dev"]["val"])), (
            f"Fold {f['fold']}: train/val overlap"
        )

    # Verify full coverage of remainder
    remainder_set = set(remainder_ids)
    for f in folds:
        fold_all = set(f["dev"]["train"]) | set(f["dev"]["val"]) | set(f["test"])
        assert fold_all == remainder_set, (
            f"Fold {f['fold']}: covers {len(fold_all)}/{len(remainder_set)} remainder Qs"
        )

    return folds


def build_production_fold(
    questions: list[dict[str, Any]],
    seed: int,
    canonical_manifest_path: str,
    canonical_fingerprint: str,
) -> dict[str, Any]:
    """Build a fold file using all 817 questions as dev (4:1 train/val).

    The test set is empty.  This produces the strongest possible artifact
    for downstream evals where TruthfulQA overlap is not a concern.
    """
    stable_ids = sorted(q["stable_id"] for q in questions)
    n_val = len(stable_ids) // 5
    rng = random.Random(seed)
    shuffled = list(stable_ids)
    rng.shuffle(shuffled)
    val_ids = sorted(shuffled[:n_val])
    train_ids = sorted(shuffled[n_val:])

    assert set(train_ids).isdisjoint(set(val_ids)), "production train/val overlap"
    assert len(train_ids) + len(val_ids) == len(questions), "production coverage"

    return {
        "version": 1,
        "seed": seed,
        "fold": "production",
        "dev": {"train": train_ids, "val": val_ids},
        "test": [],
        "counts": {
            "train": len(train_ids),
            "val": len(val_ids),
            "test": 0,
        },
        "canonical_manifest": canonical_manifest_path,
        "canonical_fingerprint": canonical_fingerprint,
    }


_TRUTHFULQA_MC_ID_RE = re.compile(r"^truthfulqa_(mc[12])_(\d+)$")


def _load_sample_manifest_ids(path: str | Path) -> list[str]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [str(item) for item in payload]
    if isinstance(payload, dict):
        for key in ("ids", "sample_ids"):
            ids = payload.get(key)
            if isinstance(ids, list):
                return [str(item) for item in ids]
    raise ValueError(f"{path} must be a JSON list or an object with an 'ids' list")


def stable_ids_from_truthfulqa_mc_manifest(
    questions: list[dict[str, Any]],
    manifest_path: str | Path,
    *,
    expected_variant: str,
) -> list[str]:
    """Map TruthfulQA MC sample IDs back to canonical stable question IDs."""
    csv_idx_to_stable_id = {int(q["csv_idx"]): str(q["stable_id"]) for q in questions}
    stable_ids: list[str] = []
    seen: set[str] = set()
    for sample_id in _load_sample_manifest_ids(manifest_path):
        match = _TRUTHFULQA_MC_ID_RE.match(sample_id)
        if not match:
            raise ValueError(f"Malformed TruthfulQA MC sample ID: {sample_id!r}")
        variant, csv_idx_text = match.groups()
        if variant != expected_variant:
            raise ValueError(
                f"{manifest_path} contains {variant!r} ID {sample_id!r}; "
                f"expected {expected_variant!r}"
            )
        csv_idx = int(csv_idx_text)
        try:
            stable_id = csv_idx_to_stable_id[csv_idx]
        except KeyError as exc:
            raise ValueError(
                f"{manifest_path} references CSV index {csv_idx}, "
                "which is absent from the canonical TruthfulQA manifest"
            ) from exc
        if stable_id in seen:
            raise ValueError(
                f"Duplicate TruthfulQA question in {manifest_path}: {stable_id}"
            )
        seen.add(stable_id)
        stable_ids.append(stable_id)
    return sorted(stable_ids)


def build_anchor2_heldout_fold(
    questions: list[dict[str, Any]],
    *,
    heldout_stable_ids: list[str],
    seed: int,
    canonical_manifest_path: str,
    canonical_fingerprint: str,
    fold_name: str = "anchor2_mistral24b",
) -> dict[str, Any]:
    """Build the direct Anchor 2 fit fold excluding locked held-out MC IDs."""
    all_ids = sorted(str(q["stable_id"]) for q in questions)
    all_id_set = set(all_ids)
    heldout_set = set(heldout_stable_ids)
    unknown = heldout_set - all_id_set
    if unknown:
        raise ValueError(
            "Anchor2 held-out IDs are absent from the canonical TruthfulQA manifest: "
            f"{sorted(unknown)[:5]}"
        )

    dev_ids = sorted(all_id_set - heldout_set)
    rng = random.Random(seed)
    dev_shuffled = list(dev_ids)
    rng.shuffle(dev_shuffled)
    n_val = len(dev_shuffled) // 5
    val_ids = sorted(dev_shuffled[:n_val])
    train_ids = sorted(dev_shuffled[n_val:])

    assert set(train_ids).isdisjoint(val_ids), "anchor2 train/val overlap"
    assert (set(train_ids) | set(val_ids)).isdisjoint(heldout_set), (
        "anchor2 dev/test overlap"
    )
    assert set(train_ids) | set(val_ids) | heldout_set == all_id_set, (
        "anchor2 fold coverage"
    )

    return {
        "version": 1,
        "seed": seed,
        "fold": fold_name,
        "protocol": "anchor2_locked_heldout_exclusion",
        "dev": {
            "train": train_ids,
            "val": val_ids,
        },
        "test": sorted(heldout_set),
        "counts": {
            "train": len(train_ids),
            "val": len(val_ids),
            "test": len(heldout_set),
        },
        "canonical_manifest": canonical_manifest_path,
        "canonical_fingerprint": canonical_fingerprint,
        "heldout_fingerprint": fingerprint_ids(sorted(heldout_set)),
    }


def build_mc_manifests_for_stable_ids(
    questions: list[dict[str, Any]],
    stable_ids: list[str],
    *,
    seed: int,
    prefix: str,
) -> dict[str, list[str]]:
    sid_to_csv_idx = {str(q["stable_id"]): int(q["csv_idx"]) for q in questions}
    csv_indices = sorted(sid_to_csv_idx[sid] for sid in stable_ids)
    manifests: dict[str, list[str]] = {}
    for variant in ("mc1", "mc2"):
        key = f"{prefix}_cal_val_{variant}_seed{seed}"
        manifests[key] = [f"truthfulqa_{variant}_{csv_idx}" for csv_idx in csv_indices]
    return manifests


def build_cal_val_mc_manifests(
    questions: list[dict[str, Any]],
    cal_val_ids: list[str],
    seed: int,
) -> dict[str, list[str]]:
    """Build MC eval manifests for the calibration val set."""
    sid_to_csv_idx = {q["stable_id"]: q["csv_idx"] for q in questions}
    csv_indices = sorted(sid_to_csv_idx[sid] for sid in cal_val_ids)
    manifests: dict[str, list[str]] = {}
    for variant in ("mc1", "mc2"):
        key = f"truthfulqa_cal_val_{variant}_seed{seed}"
        manifests[key] = [f"truthfulqa_{variant}_{csv_idx}" for csv_idx in csv_indices]
    return manifests


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--csv-path",
        type=str,
        default="data/benchmarks/TruthfulQA.csv",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default="data/manifests",
    )
    p.add_argument(
        "--cal-frac",
        type=float,
        default=0.10,
        help="Fraction of questions for each of cal_train and cal_val (default: 0.10)",
    )
    p.add_argument(
        "--anchor2-heldout-mc1-manifest",
        type=str,
        default=None,
        help=(
            "Optional locked TruthfulQA MC1 manifest. When provided with the MC2 "
            "manifest, writes an Anchor 2 fold that excludes exactly those "
            "held-out questions from extraction/sweep data."
        ),
    )
    p.add_argument(
        "--anchor2-heldout-mc2-manifest",
        type=str,
        default=None,
        help="Optional locked TruthfulQA MC2 manifest paired with --anchor2-heldout-mc1-manifest.",
    )
    p.add_argument(
        "--anchor2-prefix",
        type=str,
        default="truthfulqa_anchor2_mistral24b",
        help="Filename prefix for Anchor 2 fold and internal cal-val manifests.",
    )
    p.add_argument(
        "--only-anchor2",
        action="store_true",
        help="Write only the Anchor 2 fold/manifests and leave existing generic files untouched.",
    )
    return p.parse_args()


def _write_json(path: Path, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # --- Load or reuse canonical manifest ---
    canonical_path = out / "truthfulqa_canonical_questions.json"
    if canonical_path.exists():
        print(f"Loading existing canonical manifest from {canonical_path}")
        with open(canonical_path) as f:
            manifest = json.load(f)
        questions = manifest["questions"]
        canonical_fp = manifest["fingerprint"]
    else:
        print("Building canonical question manifest...")
        questions = build_canonical_manifest(args.csv_path)
        canonical_fp = fingerprint_ids([q["stable_id"] for q in questions])
        manifest = {
            "version": 1,
            "source_csv": args.csv_path,
            "n_questions": len(questions),
            "fingerprint": canonical_fp,
            "questions": questions,
        }
        _write_json(canonical_path, manifest)
        print(f"  Wrote {len(questions)} questions → {canonical_path}")

    if bool(args.anchor2_heldout_mc1_manifest) != bool(
        args.anchor2_heldout_mc2_manifest
    ):
        raise ValueError(
            "--anchor2-heldout-mc1-manifest and --anchor2-heldout-mc2-manifest "
            "must be provided together"
        )

    if args.anchor2_heldout_mc1_manifest and args.anchor2_heldout_mc2_manifest:
        print("Building Anchor 2 locked-heldout fold...")
        mc1_heldout = stable_ids_from_truthfulqa_mc_manifest(
            questions,
            args.anchor2_heldout_mc1_manifest,
            expected_variant="mc1",
        )
        mc2_heldout = stable_ids_from_truthfulqa_mc_manifest(
            questions,
            args.anchor2_heldout_mc2_manifest,
            expected_variant="mc2",
        )
        if mc1_heldout != mc2_heldout:
            raise ValueError(
                "Anchor 2 MC1/MC2 held-out manifests refer to different "
                "TruthfulQA question sets"
            )
        anchor2_fold = build_anchor2_heldout_fold(
            questions,
            heldout_stable_ids=mc1_heldout,
            seed=args.seed,
            canonical_manifest_path=str(canonical_path),
            canonical_fingerprint=canonical_fp,
            fold_name=args.anchor2_prefix,
        )
        anchor2_fold_path = out / f"{args.anchor2_prefix}_fold_seed{args.seed}.json"
        _write_json(anchor2_fold_path, anchor2_fold)
        print(
            f"  Anchor2 fold: train={anchor2_fold['counts']['train']}, "
            f"val={anchor2_fold['counts']['val']}, "
            f"test={anchor2_fold['counts']['test']} → {anchor2_fold_path}"
        )
        anchor2_mc = build_mc_manifests_for_stable_ids(
            questions,
            anchor2_fold["dev"]["val"],
            seed=args.seed,
            prefix=args.anchor2_prefix,
        )
        for name, ids in anchor2_mc.items():
            path = out / f"{name}.json"
            _write_json(path, ids)
            print(f"  {len(ids)} Anchor2 internal cal-val IDs → {path}")

    if args.only_anchor2:
        print("Done.")
        return

    # --- Calibration split ---
    print(f"Building calibration split (cal_frac={args.cal_frac})...")
    cal_split = build_calibration_split(questions, args.seed, args.cal_frac)
    cal_split_path = out / f"truthfulqa_calibration_split_seed{args.seed}.json"
    _write_json(cal_split_path, cal_split)
    c = cal_split["counts"]
    print(
        f"  cal_train={c['cal_train']}, cal_val={c['cal_val']}, "
        f"remainder={c['remainder']} → {cal_split_path}"
    )

    # --- Calibration pseudo-fold (for extraction) ---
    cal_fold = build_calibration_fold(cal_split, str(canonical_path), canonical_fp)
    cal_fold_path = out / f"truthfulqa_cal_fold_seed{args.seed}.json"
    _write_json(cal_fold_path, cal_fold)
    print(f"  Calibration fold → {cal_fold_path}")

    # --- Cal-val MC manifests ---
    cal_mc = build_cal_val_mc_manifests(questions, cal_split["cal_val"], args.seed)
    for name, ids in cal_mc.items():
        path = out / f"{name}.json"
        _write_json(path, ids)
        print(f"  {len(ids)} cal-val IDs → {path}")

    # --- Final 2-fold CV from remainder ---
    print("Building final 2-fold CV from remainder...")
    final_folds = build_final_folds(cal_split["remainder"], args.seed)
    for fold in final_folds:
        fold["canonical_manifest"] = str(canonical_path)
        fold["canonical_fingerprint"] = canonical_fp
        fold["calibration_excluded"] = {
            "cal_train_fingerprint": fingerprint_ids(cal_split["cal_train"]),
            "cal_val_fingerprint": fingerprint_ids(cal_split["cal_val"]),
            "n_excluded": c["cal_train"] + c["cal_val"],
        }
        fold_path = out / f"truthfulqa_final_fold{fold['fold']}_seed{args.seed}.json"
        _write_json(fold_path, fold)
        fc = fold["counts"]
        print(
            f"  Final fold {fold['fold']}: train={fc['train']}, val={fc['val']}, "
            f"test={fc['test']} → {fold_path}"
        )

    # --- Final fold MC manifests ---
    final_mc = build_mc_manifests(questions, final_folds)
    # Rename keys to include "final_" prefix
    for fold in final_folds:
        fold_idx = fold["fold"]
        for variant in ("mc1", "mc2"):
            old_key = f"truthfulqa_fold{fold_idx}_heldout_{variant}_seed{args.seed}"
            new_key = (
                f"truthfulqa_final_fold{fold_idx}_heldout_{variant}_seed{args.seed}"
            )
            if old_key in final_mc:
                ids = final_mc.pop(old_key)
                path = out / f"{new_key}.json"
                _write_json(path, ids)
                print(f"  {len(ids)} final fold {fold_idx} {variant} IDs → {path}")

    # --- Production fold (all 817 as dev) ---
    print("Building production fold (all 817 questions)...")
    prod_fold = build_production_fold(
        questions, args.seed, str(canonical_path), canonical_fp
    )
    prod_path = out / f"truthfulqa_production_fold_seed{args.seed}.json"
    _write_json(prod_path, prod_fold)
    pc = prod_fold["counts"]
    print(f"  Production: train={pc['train']}, val={pc['val']} → {prod_path}")

    # --- Cross-partition verification ---
    cal_all = set(cal_split["cal_train"]) | set(cal_split["cal_val"])
    remainder_set = set(cal_split["remainder"])
    for fold in final_folds:
        fold_test = set(fold["test"])
        assert fold_test.issubset(remainder_set), (
            f"Final fold {fold['fold']} test leaks into calibration"
        )
        assert fold_test.isdisjoint(cal_all), (
            f"Final fold {fold['fold']} test overlaps with calibration"
        )

    print("Done. All partitions verified.")


if __name__ == "__main__":
    main()
