#!/usr/bin/env python3
"""Finalize SIMID open-correctness calibration labels."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from simid_open_calibration import (
    OPEN_CORRECTNESS_CALIBRATION_SCHEMA_VERSION,
    finalize_open_calibration,
    format_optional_metric,
    input_file_metadata,
    load_jsonl,
    write_json,
)
from utils import (
    finish_run_provenance,
    provenance_error_message,
    provenance_status_for_exception,
    start_run_provenance,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue-path", type=Path, required=True)
    parser.add_argument("--secondary-rater-path", type=Path, required=True)
    parser.add_argument("--adjudication-path", type=Path, required=True)
    parser.add_argument("--rule-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--primary-rater-name", default="simid_open_primary_judge")
    parser.add_argument("--secondary-rater-name", default="simid_open_secondary_rater")
    parser.add_argument("--secondary-model", default="gpt-5.5")
    parser.add_argument("--adjudicator-model", default="gpt-5.5")
    parser.add_argument("--allow-overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.output.exists() and not args.allow_overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing SIMID open calibration summary: "
            f"{args.output}. Pass --allow-overwrite to replace it explicitly."
        )
    input_metadata = {
        "queue": input_file_metadata(args.queue_path),
        "secondary_rater": input_file_metadata(args.secondary_rater_path),
        "adjudication": (
            input_file_metadata(args.adjudication_path)
            if args.adjudication_path.exists()
            else {"path": str(args.adjudication_path), "content_sha256": None}
        ),
        "rule": input_file_metadata(args.rule_path),
    }
    provenance = start_run_provenance(
        args,
        primary_target=args.output,
        output_targets=[args.output],
        extra={
            "simid_schema": OPEN_CORRECTNESS_CALIBRATION_SCHEMA_VERSION,
            "inputs": input_metadata,
        },
    )
    status = "completed"
    extra: dict[str, Any] = {}
    try:
        summary = finalize_open_calibration(
            queue_rows=load_jsonl(args.queue_path),
            secondary_rows=load_jsonl(args.secondary_rater_path),
            adjudication_rows=load_jsonl(args.adjudication_path, required=False),
            rule_path=args.rule_path,
            primary_rater_name=args.primary_rater_name,
            secondary_rater_name=args.secondary_rater_name,
            secondary_model=args.secondary_model,
            adjudicator_model=args.adjudicator_model,
        )
        summary["inputs"] = input_metadata
        write_json(args.output, summary)
        extra = {
            "n_cases": summary["n_cases"],
            "claimability": summary["claimability"],
            "queue_case_ids_sha256": summary["queue_case_ids_sha256"],
        }
        print(
            "Finalized SIMID open calibration: "
            f"agreement={summary['irr']['raw_agreement']['estimate']:.1%} "
            f"kappa={format_optional_metric(summary['irr']['cohen_kappa'])} "
            f"AC1={format_optional_metric(summary['irr']['gwet_ac1'])}"
        )
    except BaseException as exc:
        status = provenance_status_for_exception(exc)
        extra["error"] = provenance_error_message(exc)
        raise
    finally:
        finish_run_provenance(provenance, status, extra)


if __name__ == "__main__":
    main()
