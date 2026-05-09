#!/usr/bin/env python3
"""Validate and merge SIMID open independent-rater label files."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

from review_package import (
    expected_batch_label_files,
    expect,
    merged_label_rows,
    validate_batch_file,
    write_jsonl,
)


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PACKAGE_DIR = (
    ROOT / "data/gemma3_4b/intervention/"
    "simid_iti_truthfulqa-paperfaithful_k12_first-3-tokens/"
    "mvp_20260427_calibration/human_review_package"
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate SIMID open blind batch labels and optionally merge."
    )
    parser.add_argument("--package-dir", type=Path, default=DEFAULT_PACKAGE_DIR)
    parser.add_argument(
        "--label-file",
        type=Path,
        help="Validate one batch label file. Defaults to all expected batches.",
    )
    parser.add_argument(
        "--batch-dir",
        type=Path,
        help="Batch directory for --label-file. Defaults to label file parent.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Write merged calibration-ID labels after validating all batches.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing --output file.",
    )
    return parser.parse_args(argv)


def validate_one(args: argparse.Namespace) -> int:
    label_file = args.label_file.resolve()
    batch_dir = (
        args.batch_dir.resolve()
        if args.batch_dir is not None
        else label_file.parent.resolve()
    )
    rows = validate_batch_file(label_file, batch_dir)
    print(f"Validated {len(rows)} rows: {label_file}")
    return 0


def validate_all(args: argparse.Namespace) -> int:
    package_dir = args.package_dir.resolve()
    all_rows: list[dict[str, Any]] = []
    for label_file, batch_dir in expected_batch_label_files(package_dir):
        expect(label_file.exists(), f"Missing batch label file: {label_file}")
        all_rows.extend(validate_batch_file(label_file, batch_dir))

    print(f"Validated {len(all_rows)} blind batch labels.")
    if args.output is not None:
        output = args.output.resolve()
        if output.exists() and not args.overwrite:
            raise FileExistsError(f"{output} already exists; pass --overwrite")
        merged = merged_label_rows(package_dir=package_dir, batch_rows=all_rows)
        output.parent.mkdir(parents=True, exist_ok=True)
        write_jsonl(output, merged)
        print(f"Wrote {len(merged)} merged labels: {output}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.label_file is not None:
        return validate_one(args)
    return validate_all(args)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
