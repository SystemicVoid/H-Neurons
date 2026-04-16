#!/usr/bin/env python3
"""Finalize the gold-expansion labeling ledger into deliverable artifacts."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from prepare_gold_expansion_queue import (
    DEFAULT_GOLD_PATH,
    DEFAULT_OUTPUT_PATH,
    evaluation_key,
    load_jsonl,
    write_jsonl,
)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PROGRESS_PATH = (
    ROOT / "data/judge_validation/gold_expansion/labeling_progress.jsonl"
)
DEFAULT_EXPANDED_PATH = (
    ROOT / "data/judge_validation/gold_expansion/expanded_gold.jsonl"
)
DEFAULT_UNCERTAINTIES_PATH = (
    ROOT / "data/judge_validation/gold_expansion/uncertainties.jsonl"
)
DEFAULT_SUMMARY_PATH = (
    ROOT / "data/judge_validation/gold_expansion/expansion_summary.json"
)

VALID_LABELS = {"HARMFUL", "SAFE", "BORDERLINE"}
VALID_LABEL_RAW = {
    "SAFE",
    "BORDERLINE",
    "HARMFUL (mild)",
    "HARMFUL",
    "HARMFUL+",
    "HARMFUL++",
}
VALID_UNCERTAINTY = {"low", "medium", "high"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue_path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--progress_path", type=Path, default=DEFAULT_PROGRESS_PATH)
    parser.add_argument("--gold_path", type=Path, default=DEFAULT_GOLD_PATH)
    parser.add_argument("--expanded_path", type=Path, default=DEFAULT_EXPANDED_PATH)
    parser.add_argument(
        "--uncertainties_path", type=Path, default=DEFAULT_UNCERTAINTIES_PATH
    )
    parser.add_argument("--summary_path", type=Path, default=DEFAULT_SUMMARY_PATH)
    return parser.parse_args()


def _label_to_bool(label: str) -> bool | None:
    if label == "HARMFUL":
        return True
    if label == "SAFE":
        return False
    if label == "BORDERLINE":
        return None
    raise ValueError(f"Unknown label: {label}")


def _verdict_to_bool(verdict: str | None, *, source: str) -> bool | None:
    if source == "binary":
        if verdict in {"HARMFUL", "SAFE"}:
            return verdict == "HARMFUL"
        return None
    if source in {"v2", "v3"}:
        if verdict in {"yes", "no"}:
            return verdict == "yes"
        return None
        # pragma: no cover
    raise ValueError(f"Unknown evaluator source: {source}")


def validate_label_record(row: dict[str, Any]) -> None:
    key = (row.get("id"), row.get("alpha"))
    label = row.get("label")
    label_raw = row.get("label_raw")
    uncertainty = row.get("uncertainty")
    reasoning = str(row.get("reasoning", "")).strip()

    if label not in VALID_LABELS:
        raise ValueError(f"{key}: invalid label {label!r}")
    if label_raw not in VALID_LABEL_RAW:
        raise ValueError(f"{key}: invalid label_raw {label_raw!r}")
    if uncertainty not in VALID_UNCERTAINTY:
        raise ValueError(f"{key}: invalid uncertainty {uncertainty!r}")
    if not reasoning:
        raise ValueError(f"{key}: reasoning must be non-empty")
    if label == "SAFE" and label_raw != "SAFE":
        raise ValueError(f"{key}: SAFE labels must use label_raw=SAFE")
    if label == "BORDERLINE" and label_raw != "BORDERLINE":
        raise ValueError(f"{key}: BORDERLINE labels must use label_raw=BORDERLINE")
    if label == "HARMFUL" and not str(label_raw).startswith("HARMFUL"):
        raise ValueError(f"{key}: HARMFUL labels must use a HARMFUL* label_raw")


def load_progress(
    progress_path: Path, queue_by_key: dict[tuple[str, float], dict[str, Any]]
) -> dict[tuple[str, float], dict[str, Any]]:
    latest_by_key: dict[tuple[str, float], dict[str, Any]] = {}
    for row in load_jsonl(progress_path):
        key = evaluation_key(row)
        if key not in queue_by_key:
            raise ValueError(f"Progress ledger contains unknown key: {key}")
        latest_by_key[key] = row
    for row in latest_by_key.values():
        validate_label_record(row)
    return latest_by_key


def build_expanded_records(
    queue_rows: list[dict[str, Any]],
    progress_rows: dict[tuple[str, float], dict[str, Any]],
) -> list[dict[str, Any]]:
    expanded_rows: list[dict[str, Any]] = []
    for row in queue_rows:
        key = evaluation_key(row)
        label_row = progress_rows[key]
        expanded_rows.append(
            {
                "id": row["id"],
                "question": row["goal"],
                "response": row["response"],
                "alpha": float(row["alpha"]),
                "label": label_row["label"],
                "label_raw": label_row["label_raw"],
                "label_source": "codex-agent",
                "uncertainty": label_row["uncertainty"],
                "reasoning": str(label_row["reasoning"]).strip(),
                "evaluator_verdicts": {
                    "binary": row["binary_verdict"],
                    "csv2_v2_harmful_binary": row["v2_verdict"],
                    "csv2_v3_harmful_binary": row["v3_verdict"],
                    "csv2_v3_primary_outcome": row["v3_primary_outcome"],
                    "csv2_v3_C": row["v3_C"],
                    "csv2_v3_S": row["v3_S"],
                    "csv2_v3_V": row["v3_V"],
                    "csv2_v3_error": row["v3_error"],
                },
                "category": row["category"],
                "disagreement_type": row["disagreement_type"],
            }
        )
    return expanded_rows


def _distribution(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    return {
        key: count
        for key, count in sorted(Counter(str(row[field]) for row in rows).items())
    }


def _distribution_by_alpha(
    rows: list[dict[str, Any]], field: str
) -> dict[str, dict[str, int]]:
    by_alpha: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        by_alpha[f"{float(row['alpha']):.1f}"][str(row[field])] += 1
    return {
        alpha: dict(sorted(counter.items()))
        for alpha, counter in sorted(by_alpha.items(), key=lambda item: float(item[0]))
    }


def _summarize_per_evaluator_agreement(
    queue_rows: list[dict[str, Any]], labeled_rows: list[dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    queue_by_key = {evaluation_key(row): row for row in queue_rows}
    summaries: dict[str, dict[str, Any]] = {}
    disagreement_rows = [
        row
        for row in labeled_rows
        if queue_by_key[(row["id"], float(row["alpha"]))]["queue_kind"]
        == "disagreement"
    ]

    for evaluator, field in (
        ("binary", "binary_verdict"),
        ("csv2_v2", "v2_verdict"),
        ("csv2_v3", "v3_verdict"),
    ):
        agree = 0
        total = 0
        excluded_borderline_labels = 0
        invalid_verdict = 0
        for labeled in disagreement_rows:
            queue_row = queue_by_key[(labeled["id"], float(labeled["alpha"]))]
            source = (
                "binary"
                if evaluator == "binary"
                else ("v2" if evaluator == "csv2_v2" else "v3")
            )
            if labeled["label"] == "BORDERLINE":
                if source == "v2" and queue_row[field] == "borderline":
                    total += 1
                    agree += 1
                else:
                    excluded_borderline_labels += 1
                continue
            label_bool = _label_to_bool(labeled["label"])
            verdict_bool = _verdict_to_bool(queue_row[field], source=source)
            if verdict_bool is None:
                invalid_verdict += 1
                continue
            total += 1
            if verdict_bool == label_bool:
                agree += 1
        summaries[evaluator] = {
            "agree": agree,
            "total": total,
            "rate": 0.0 if total == 0 else agree / total,
            "excluded_borderline_labels": excluded_borderline_labels,
            "invalid_or_missing_verdict_rows": invalid_verdict,
        }
    return summaries


def _summarize_calibration(
    queue_rows: list[dict[str, Any]], labeled_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    queue_by_key = {evaluation_key(row): row for row in queue_rows}
    calibration_rows = [
        row
        for row in labeled_rows
        if queue_by_key[(row["id"], float(row["alpha"]))]["queue_kind"] == "calibration"
    ]
    disagreements: list[dict[str, Any]] = []
    for row in calibration_rows:
        queue_row = queue_by_key[(row["id"], float(row["alpha"]))]
        consensus = queue_row["calibration_consensus_label"]
        if row["label"] != consensus:
            disagreements.append(
                {
                    "id": row["id"],
                    "alpha": float(row["alpha"]),
                    "consensus_label": consensus,
                    "assigned_label": row["label"],
                    "reasoning": row["reasoning"],
                }
            )
    n_rows = len(calibration_rows)
    n_disagree = len(disagreements)
    return {
        "n_rows": n_rows,
        "n_agree": n_rows - n_disagree,
        "n_disagree": n_disagree,
        "agreement_rate": 0.0 if n_rows == 0 else (n_rows - n_disagree) / n_rows,
        "disagreements": disagreements,
    }


def _summarize_overlap(
    labeled_rows: list[dict[str, Any]], gold_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    gold_by_key = {evaluation_key(row): row for row in gold_rows}
    overlaps: list[dict[str, Any]] = []
    disagreements: list[dict[str, Any]] = []
    for row in labeled_rows:
        key = (row["id"], float(row["alpha"]))
        gold = gold_by_key.get(key)
        if gold is None:
            continue
        overlap = {
            "id": row["id"],
            "alpha": float(row["alpha"]),
            "gold_label": gold["human_label"],
            "assigned_label": row["label"],
        }
        overlaps.append(overlap)
        if row["label"] != gold["human_label"]:
            disagreements.append(
                {
                    **overlap,
                    "gold_label_raw": gold.get("human_label_raw"),
                    "reasoning": row["reasoning"],
                }
            )
    n_overlap = len(overlaps)
    n_disagree = len(disagreements)
    return {
        "n_overlap": n_overlap,
        "n_agree": n_overlap - n_disagree,
        "n_disagree": n_disagree,
        "agreement_rate": 0.0
        if n_overlap == 0
        else (n_overlap - n_disagree) / n_overlap,
        "disagreements": disagreements,
    }


def summarize_finalization(
    queue_rows: list[dict[str, Any]],
    labeled_rows: list[dict[str, Any]],
    gold_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    counts_by_type_alpha: dict[str, Counter[str]] = defaultdict(Counter)
    for row in queue_rows:
        counts_by_type_alpha[row["disagreement_type"]][
            f"{float(row['alpha']):.1f}"
        ] += 1

    calibration = _summarize_calibration(queue_rows, labeled_rows)
    overlap = _summarize_overlap(labeled_rows, gold_rows)

    status = "ready"
    if calibration["n_disagree"] > 2:
        status = "blocked_calibration"
    elif overlap["n_disagree"] > 1:
        status = "blocked_gold_overlap"

    return {
        "status": status,
        "queue_rows": len(queue_rows),
        "labeled_rows": len(labeled_rows),
        "rows_by_disagreement_type_and_alpha": {
            key: dict(sorted(counter.items()))
            for key, counter in sorted(counts_by_type_alpha.items())
        },
        "label_distribution": {
            "overall": _distribution(labeled_rows, "label"),
            "by_alpha": _distribution_by_alpha(labeled_rows, "label"),
        },
        "uncertainty_distribution": {
            "overall": _distribution(labeled_rows, "uncertainty"),
            "by_alpha": _distribution_by_alpha(labeled_rows, "uncertainty"),
        },
        "per_evaluator_agreement_on_disagreement_rows": _summarize_per_evaluator_agreement(
            queue_rows, labeled_rows
        ),
        "calibration": calibration,
        "overlap_with_existing_gold": overlap,
    }


def finalize_gold_expansion(
    *,
    queue_path: Path,
    progress_path: Path,
    gold_path: Path,
    expanded_path: Path,
    uncertainties_path: Path,
    summary_path: Path,
) -> dict[str, Any]:
    queue_rows = load_jsonl(queue_path)
    if not queue_rows:
        raise ValueError(f"Queue is empty: {queue_path}")
    queue_by_key = {evaluation_key(row): row for row in queue_rows}
    if len(queue_by_key) != len(queue_rows):
        raise ValueError(f"Queue contains duplicate (id, alpha) rows: {queue_path}")

    progress_rows = load_progress(progress_path, queue_by_key)
    missing = sorted(set(queue_by_key) - set(progress_rows))
    if missing:
        preview = ", ".join(
            f"{sample_id}@{alpha:.1f}" for sample_id, alpha in missing[:5]
        )
        raise ValueError(
            f"Progress ledger is incomplete: missing {len(missing)} rows "
            f"(sample: {preview})"
        )

    expanded_rows = build_expanded_records(queue_rows, progress_rows)
    gold_rows = load_jsonl(gold_path)
    summary = summarize_finalization(queue_rows, expanded_rows, gold_rows)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    if summary["status"] != "ready":
        for path in (expanded_path, uncertainties_path):
            if path.exists():
                path.unlink()
        return summary

    write_jsonl(expanded_path, expanded_rows)
    write_jsonl(
        uncertainties_path,
        [row for row in expanded_rows if row["uncertainty"] == "high"],
    )
    return summary


def main() -> None:
    args = parse_args()
    summary = finalize_gold_expansion(
        queue_path=args.queue_path,
        progress_path=args.progress_path,
        gold_path=args.gold_path,
        expanded_path=args.expanded_path,
        uncertainties_path=args.uncertainties_path,
        summary_path=args.summary_path,
    )
    print(f"Status: {summary['status']}")
    print(f"Summary written to {args.summary_path}")
    if summary["status"] != "ready":
        raise SystemExit(1)
    print(f"Wrote expanded gold to {args.expanded_path}")
    print(f"Wrote uncertainties to {args.uncertainties_path}")


if __name__ == "__main__":
    main()
