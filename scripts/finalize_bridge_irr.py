#!/usr/bin/env python3
"""Finalize bridge inter-rater reliability labels into a summary artifact."""

from __future__ import annotations

import argparse
from pathlib import Path

from bridge_irr import (
    DEFAULT_OUTPUT_DIR,
    build_rater_b_summary_provenance,
    finalize_bridge_irr,
    load_adjudication_progress,
    load_json,
    load_jsonl,
    load_label_progress,
    write_json,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--queue_path", type=Path)
    parser.add_argument("--key_path", type=Path)
    parser.add_argument("--rater_a_path", type=Path)
    parser.add_argument("--rater_b_path", type=Path)
    parser.add_argument("--adjudication_path", type=Path)
    parser.add_argument("--status_path", type=Path)
    parser.add_argument("--rater_b_provenance_path", type=Path)
    parser.add_argument("--summary_path", type=Path)
    parser.add_argument("--adjudicated_rows_path", type=Path)
    parser.add_argument("--disagreements_path", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    queue_path = args.queue_path or (args.output_dir / "test_queue_blinded.jsonl")
    key_path = args.key_path or (args.output_dir / "test_queue_key.jsonl")
    rater_a_path = args.rater_a_path or (args.output_dir / "rater_a_progress.jsonl")
    rater_b_path = args.rater_b_path or (args.output_dir / "rater_b_progress.jsonl")
    adjudication_path = args.adjudication_path or (
        args.output_dir / "adjudication_progress.jsonl"
    )
    status_path = args.status_path or (args.output_dir / "bridge_irr_status.json")
    rater_b_provenance_path = args.rater_b_provenance_path or (
        args.output_dir / "rater_b_provenance.json"
    )
    summary_path = args.summary_path or (args.output_dir / "bridge_irr_summary.json")
    adjudicated_rows_path = args.adjudicated_rows_path or (
        args.output_dir / "adjudicated_labels.jsonl"
    )
    disagreements_path = args.disagreements_path or (
        args.output_dir / "bridge_irr_disagreements.jsonl"
    )

    queue_rows = load_jsonl(queue_path)
    key_rows = load_jsonl(key_path)
    rater_a_rows = load_label_progress(rater_a_path)
    rater_b_rows = load_label_progress(rater_b_path)
    adjudication_rows = load_adjudication_progress(adjudication_path)
    status_payload = load_json(status_path)
    adjudication_rule = status_payload.get("adjudication_rule")
    if not isinstance(adjudication_rule, dict):
        raise ValueError(
            f"{status_path} is missing frozen adjudication_rule provenance"
        )
    rater_b_provenance = build_rater_b_summary_provenance(
        rater_b_rows,
        provenance_payload=(
            load_json(rater_b_provenance_path)
            if rater_b_provenance_path.exists()
            else None
        ),
    )

    summary = finalize_bridge_irr(
        queue_rows=queue_rows,
        key_rows=key_rows,
        rater_a_rows=rater_a_rows,
        rater_b_rows=rater_b_rows,
        adjudication_rows=adjudication_rows,
        adjudication_rule=adjudication_rule,
        rater_b_provenance=rater_b_provenance,
    )
    write_json(summary_path, summary)
    write_jsonl(adjudicated_rows_path, summary["adjudicated_rows"])
    write_jsonl(disagreements_path, summary["disagreements"])
    print(
        "Finalized bridge IRR summary: "
        f"agreement={summary['irr']['raw_agreement']['estimate_pct']:.1f}% "
        f"kappa={summary['irr']['cohen_kappa']:.4f} "
        f"ac1={summary['irr']['gwet_ac1']:.4f}"
    )


if __name__ == "__main__":
    main()
