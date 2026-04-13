#!/usr/bin/env python3
"""Build a deterministic BioASQ question-id manifest."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import pandas as pd

from utils import (
    finish_run_provenance,
    fingerprint_ids,
    json_dumps,
    provenance_error_message,
    provenance_status_for_exception,
    start_run_provenance,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument(
        "--data_path",
        type=Path,
        default=Path("data/benchmarks/bioasq13b_factoid.parquet"),
    )
    parser.add_argument("--output_dir", type=Path, default=Path("data/manifests"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / f"bioasq_factoid_n{args.n}_seed{args.seed}.json"
    metadata_path = (
        args.output_dir / f"bioasq_factoid_n{args.n}_seed{args.seed}_metadata.json"
    )
    provenance_handle = start_run_provenance(
        args,
        primary_target=str(args.output_dir),
        output_targets=[str(manifest_path), str(metadata_path)],
        primary_target_is_dir=True,
    )
    provenance_status = "completed"
    provenance_extra: dict[str, object] = {}
    try:
        df = pd.read_parquet(args.data_path)
        question_ids = sorted(str(qid) for qid in df["question_id"].tolist())
        if args.n <= 0:
            raise ValueError("--n must be positive")
        if args.n > len(question_ids):
            raise ValueError(
                f"Requested n={args.n} exceeds available BioASQ rows={len(question_ids)}"
            )

        rng = random.Random(args.seed)
        selected = sorted(rng.sample(question_ids, k=args.n))
        manifest_path.write_text(
            f"{json.dumps(selected, indent=2)}\n", encoding="utf-8"
        )
        metadata_path.write_text(
            json_dumps(
                {
                    "seed": args.seed,
                    "n": args.n,
                    "data_path": str(args.data_path),
                    "n_available": len(question_ids),
                    "fingerprint": fingerprint_ids(selected),
                }
            ),
            encoding="utf-8",
        )
        provenance_extra["record_count"] = len(selected)
        print(f"Wrote BioASQ manifest:  {manifest_path}")
        print(f"Wrote BioASQ metadata:  {metadata_path}")
    except BaseException as exc:
        provenance_status = provenance_status_for_exception(exc)
        provenance_extra["error"] = provenance_error_message(exc)
        raise
    finally:
        finish_run_provenance(provenance_handle, provenance_status, provenance_extra)


if __name__ == "__main__":
    main()
