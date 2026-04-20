#!/usr/bin/env python3
"""Prepare the bridge inter-rater reliability coding queues."""

from __future__ import annotations

import argparse
from pathlib import Path

from bridge_irr import (
    DEFAULT_DEV_BASELINE,
    DEFAULT_DEV_COMPARISON,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_TEST_BASELINE,
    DEFAULT_TEST_COMPARISON,
    build_adjudication_rule_metadata,
    build_blinded_case_artifacts,
    build_pending_status_payload,
    ensure_progress_files_compatible,
    empty_progress_template,
    extract_discordant_cases,
    write_json,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev_baseline", type=Path, default=DEFAULT_DEV_BASELINE)
    parser.add_argument("--dev_comparison", type=Path, default=DEFAULT_DEV_COMPARISON)
    parser.add_argument("--test_baseline", type=Path, default=DEFAULT_TEST_BASELINE)
    parser.add_argument("--test_comparison", type=Path, default=DEFAULT_TEST_COMPARISON)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--adjudication_rule_path",
        type=Path,
        help="Frozen adjudication rule document committed before test labeling starts.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    adjudication_rule_path = args.adjudication_rule_path or (
        args.output_dir / "adjudication_rule.md"
    )

    dev_cases = extract_discordant_cases(
        args.dev_baseline,
        args.dev_comparison,
        comparison_name="comparison",
    )
    test_cases = extract_discordant_cases(
        args.test_baseline,
        args.test_comparison,
        comparison_name="comparison",
    )

    dev_queue, _ = build_blinded_case_artifacts(
        dev_cases,
        split="dev",
        seed=args.seed,
    )
    test_queue, test_key = build_blinded_case_artifacts(
        test_cases,
        split="test",
        seed=args.seed,
    )

    ensure_progress_files_compatible(
        expected_case_ids={row["case_id"] for row in test_queue},
        rater_a_path=args.output_dir / "rater_a_progress.jsonl",
        rater_b_path=args.output_dir / "rater_b_progress.jsonl",
        adjudication_path=args.output_dir / "adjudication_progress.jsonl",
    )

    write_jsonl(args.output_dir / "dev_calibration_queue.jsonl", dev_queue)
    write_jsonl(args.output_dir / "test_queue_blinded.jsonl", test_queue)
    write_jsonl(args.output_dir / "test_queue_key.jsonl", test_key)
    empty_progress_template(args.output_dir / "rater_a_progress.jsonl")
    empty_progress_template(args.output_dir / "rater_b_progress.jsonl")
    empty_progress_template(args.output_dir / "adjudication_progress.jsonl")

    status_payload = build_pending_status_payload(
        dev_cases=dev_cases,
        test_cases=test_cases,
        output_dir=args.output_dir,
        adjudication_rule=build_adjudication_rule_metadata(adjudication_rule_path),
    )
    write_json(args.output_dir / "bridge_irr_status.json", status_payload)

    print(
        "Prepared bridge IRR queues: "
        f"dev={status_payload['counts']['dev']['discordant_total']} "
        f"test={status_payload['counts']['test']['discordant_total']}"
    )


if __name__ == "__main__":
    main()
