#!/usr/bin/env python3
"""Build canonical TruthfulQA question manifest and 2-fold CV split files.

This script is the single source of truth for TruthfulQA question identity
and train/val/test assignment.  All data comes from the official CSV file
(``data/benchmarks/TruthfulQA.csv``); the HuggingFace dataset is not used.

It produces:

1. A **canonical manifest** mapping each of the 817 questions to a stable ID
   (hash of normalized question text) and the CSV row index.

2. **Two fold files** implementing the paper's 2-fold CV protocol (Li et al.,
   arXiv 2306.03341v6, Appendix C):
   - Each fold splits questions 50/50 into development and test.
   - Development is further split 4:1 into train and validation.
   - fold0.test == fold1.dev and vice versa.

3. **MC evaluation manifests** for each fold × variant (mc1, mc2), containing
   the ``truthfulqa_{variant}_{csv_idx}`` IDs expected by
   ``run_intervention.py``'s ``load_truthfulqa_mc()``.

Usage::

    uv run python scripts/build_truthfulqa_splits.py --seed 42
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import unicodedata
from pathlib import Path
from typing import Any

import pandas as pd

from utils import fingerprint_ids


# ---------------------------------------------------------------------------
# Stable question identity
# ---------------------------------------------------------------------------


def normalize_question(text: str) -> str:
    """Normalize question text for stable hashing.

    Lighter than ``utils.normalize_answer`` (which strips articles), this
    only applies unicode NFC, lowercasing, whitespace collapse, and strip.
    """
    text = unicodedata.normalize("NFC", text)
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def stable_question_id(question_text: str) -> str:
    """Return a deterministic ID for a question based on its normalized text.

    Format: ``tqa_{sha256_prefix_12hex}`` — 48 bits of collision resistance,
    more than sufficient for 817 items.
    """
    normalized = normalize_question(question_text)
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:12]
    return f"tqa_{digest}"


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
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def build_canonical_manifest(
    csv_path: str,
) -> list[dict[str, Any]]:
    """Build canonical question list with stable IDs from the official CSV.

    Returns a list of dicts ordered by CSV row, each with keys:
    ``stable_id``, ``csv_idx``, ``question_text``.
    The CSV (``data/benchmarks/TruthfulQA.csv``) is the sole authoritative
    source; no HuggingFace dataset is loaded.
    """
    csv_df = pd.read_csv(csv_path, encoding="utf-8-sig")
    assert len(csv_df) == 817, f"Expected 817 CSV rows, got {len(csv_df)}"

    questions: list[dict[str, Any]] = []
    seen_ids: dict[str, str] = {}  # stable_id → norm text (collision check)

    for csv_idx, q_text in enumerate(csv_df["Question"]):
        norm = normalize_question(str(q_text))
        sid = stable_question_id(norm)
        if sid in seen_ids:
            raise ValueError(
                f"Hash collision: {sid!r} for {norm!r} and {seen_ids[sid]!r}"
            )
        seen_ids[sid] = norm
        questions.append(
            {
                "stable_id": sid,
                "csv_idx": csv_idx,
                "question_text": str(q_text),
            }
        )

    assert len(questions) == 817
    return questions


def build_folds(
    questions: list[dict[str, Any]],
    seed: int,
) -> list[dict[str, Any]]:
    """Build 2-fold CV split files from canonical questions.

    Returns a list of two fold dicts.
    """
    stable_ids = sorted(q["stable_id"] for q in questions)
    rng = random.Random(seed)
    shuffled = list(stable_ids)
    rng.shuffle(shuffled)

    mid = len(shuffled) // 2
    fold0_dev_ids = shuffled[:mid]
    fold1_dev_ids = shuffled[mid:]

    folds = []
    for fold_idx, (dev_ids, test_ids) in enumerate(
        [(fold0_dev_ids, fold1_dev_ids), (fold1_dev_ids, fold0_dev_ids)]
    ):
        # Split dev 4:1 into train/val
        n_val = len(dev_ids) // 5
        # Use a fold-specific RNG so val selection is deterministic per fold
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
        assert dev_all.isdisjoint(test_set), (
            f"Fold {f['fold']}: {len(dev_all & test_set)} questions in both dev and test"
        )
        assert set(f["dev"]["train"]).isdisjoint(set(f["dev"]["val"])), (
            f"Fold {f['fold']}: train/val overlap"
        )

    # Verify full coverage
    all_ids = set(q["stable_id"] for q in questions)
    for f in folds:
        fold_all = set(f["dev"]["train"]) | set(f["dev"]["val"]) | set(f["test"])
        assert fold_all == all_ids, (
            f"Fold {f['fold']}: covers {len(fold_all)}/{len(all_ids)} questions"
        )

    return folds


def build_mc_manifests(
    questions: list[dict[str, Any]],
    folds: list[dict[str, Any]],
) -> dict[str, list[str]]:
    """Build MC eval manifests mapping test-fold stable IDs to CSV-based IDs.

    Returns a dict of ``{filename_suffix: [id_list]}`` for each fold × variant.
    """
    sid_to_csv_idx = {q["stable_id"]: q["csv_idx"] for q in questions}
    manifests: dict[str, list[str]] = {}

    for fold in folds:
        fold_idx = fold["fold"]
        test_csv_indices = sorted(sid_to_csv_idx[sid] for sid in fold["test"])
        for variant in ("mc1", "mc2"):
            key = f"truthfulqa_fold{fold_idx}_heldout_{variant}_seed{fold['seed']}"
            manifests[key] = [
                f"truthfulqa_{variant}_{csv_idx}" for csv_idx in test_csv_indices
            ]

    return manifests


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # --- Canonical manifest ---
    print("Building canonical question manifest...")
    questions = build_canonical_manifest(args.csv_path)
    fp = fingerprint_ids([q["stable_id"] for q in questions])
    manifest = {
        "version": 1,
        "source_csv": args.csv_path,
        "n_questions": len(questions),
        "fingerprint": fp,
        "questions": questions,
    }
    manifest_path = out / "truthfulqa_canonical_questions.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"  Wrote {len(questions)} questions → {manifest_path}")

    # --- 2-fold splits ---
    print("Building 2-fold CV splits...")
    folds = build_folds(questions, args.seed)
    for fold in folds:
        fold["canonical_manifest"] = str(manifest_path)
        fold["canonical_fingerprint"] = fp
        fold_path = out / f"truthfulqa_fold{fold['fold']}_seed{args.seed}.json"
        with open(fold_path, "w", encoding="utf-8") as f:
            json.dump(fold, f, indent=2, ensure_ascii=False)
            f.write("\n")
        c = fold["counts"]
        print(
            f"  Fold {fold['fold']}: train={c['train']}, val={c['val']}, "
            f"test={c['test']} → {fold_path}"
        )

    # --- MC eval manifests ---
    print("Building MC eval manifests...")
    mc_manifests = build_mc_manifests(questions, folds)
    for name, ids in mc_manifests.items():
        path = out / f"{name}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(ids, f, indent=2)
            f.write("\n")
        print(f"  {len(ids)} IDs → {path}")

    print("Done.")


if __name__ == "__main__":
    main()
