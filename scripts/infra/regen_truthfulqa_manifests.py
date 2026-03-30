#!/usr/bin/env python3
"""Regenerate TruthfulQA held-out eval manifests from the local CSV.

The manifests record which sample IDs belong to the 20% val split that was
held out during ITI artifact extraction (build_truthfulqa_paper_examples,
seed=42).  They must be regenerated whenever load_truthfulqa_mc() changes
its ID scheme (e.g. after switching from HuggingFace to CSV indices).

Reproduction procedure
----------------------
1. Reproduce the val split used during extraction: shuffle range(817) with
   Random(seed) on the HF generation dataset — same logic as
   build_truthfulqa_paper_examples() in extract_truthfulness_iti.py.
2. Collect the question texts for those val indices.
3. Look each question up in TruthfulQA.csv to get the CSV row index
   (the stable ID key used by load_truthfulqa_mc).
4. Write truthfulqa_{variant}_{csv_idx} lists to the manifest files.

Usage
-----
    uv run python scripts/infra/regen_truthfulqa_manifests.py
    uv run python scripts/infra/regen_truthfulqa_manifests.py --seed 42 --val-fraction 0.2
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import pandas as pd
from datasets import load_dataset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-fraction", type=float, default=0.2)
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


def main() -> None:
    args = parse_args()

    # Reproduce the exact val split used during ITI extraction
    ds = load_dataset("truthful_qa", "generation", split="validation")
    rng = random.Random(args.seed)
    question_indices = list(range(len(ds)))
    rng.shuffle(question_indices)
    n_val = int(len(question_indices) * args.val_fraction)
    val_questions = {ds[i]["question"].strip() for i in question_indices[:n_val]}
    print(
        f"Val split: {len(val_questions)} questions (seed={args.seed}, "
        f"val_fraction={args.val_fraction})"
    )

    # Map val question texts → CSV row indices
    csv_df = pd.read_csv(args.csv_path, encoding="utf-8-sig")
    csv_q_to_idx = {q.strip(): i for i, q in enumerate(csv_df["Question"])}

    missing = val_questions - set(csv_q_to_idx)
    if missing:
        raise ValueError(
            f"{len(missing)} val questions not found in CSV: {sorted(missing)[:3]}"
        )

    val_csv_indices = sorted(csv_q_to_idx[q] for q in val_questions)
    print(f"All {len(val_csv_indices)} val questions resolved to CSV indices.")

    # Write manifests
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for variant in ("mc1", "mc2"):
        ids = [f"truthfulqa_{variant}_{i}" for i in val_csv_indices]
        path = out / f"truthfulqa_paper_heldout_{variant}_ids_seed{args.seed}.json"
        with open(path, "w") as f:
            json.dump(ids, f, indent=2)
            f.write("\n")
        print(f"Wrote {len(ids)} IDs → {path}")


if __name__ == "__main__":
    main()
