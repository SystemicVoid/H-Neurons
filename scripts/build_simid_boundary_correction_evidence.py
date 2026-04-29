#!/usr/bin/env python3
"""Build duplicate-collapsed SIMID boundary correction-evidence artifacts."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import re
import subprocess
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RUN_DIR = (
    ROOT / "data/gemma3_4b/intervention/"
    "simid_iti_truthfulqa-paperfaithful_k12_first-3-tokens/"
    "mvp_20260427_calibration"
)
DEFAULT_PACKAGE_DIR = DEFAULT_RUN_DIR / "human_review_package"
DEFAULT_FRESH_PACKAGE_DIR = DEFAULT_PACKAGE_DIR / "fresh_blind_adjudication_20260429"
DEFAULT_OUTPUT_DIR = DEFAULT_FRESH_PACKAGE_DIR / "correction_evidence_20260429"

OUTPUT_SCHEMA_VERSION = "simid_boundary_correction_evidence/v1"
SUMMARY_SCHEMA_VERSION = "simid_boundary_correction_evidence_validation/v1"
PROVENANCE_SCHEMA_VERSION = "simid_boundary_correction_evidence_provenance/v1"
VALID_LABELS = {"CORRECT", "INCORRECT", "NOT_ATTEMPTED"}
REQUIRED_OUTPUT_KEYS = {
    "schema_version",
    "case_family",
    "group_id",
    "duplicate_group",
    "adjudication_status",
    "fresh_status",
    "fresh_label",
    "fresh_label_counts",
    "confidence_range",
    "rule_gap",
    "review_orders",
    "calibration_case_ids",
    "question",
    "gold_aliases",
    "predicted_answer_examples",
    "predicted_answer_normalized",
    "selection_reasons",
    "flags",
    "local_correction_evidence",
    "current_reference_comparison",
    "opus_comparison",
    "human_comparison",
    "diagnostic_propagation",
    "claimability_guardrail",
}


@dataclass(frozen=True)
class SourcePaths:
    run_dir: Path
    package_dir: Path
    fresh_package_dir: Path

    @property
    def boundary_status(self) -> Path:
        return self.fresh_package_dir / "duplicate_collapsed_boundary_status.jsonl"

    @property
    def fresh_resolved_labels(self) -> Path:
        return self.fresh_package_dir / "fresh_blind_boundary_labels_resolved.jsonl"

    @property
    def fresh_validation_summary(self) -> Path:
        return self.fresh_package_dir / "validation_summary.json"

    @property
    def review_cases(self) -> Path:
        return self.package_dir / "review_cases_blind.jsonl"

    @property
    def opus_labels(self) -> Path:
        return self.package_dir / "opus_4_7_labels.jsonl"

    @property
    def human_labels(self) -> Path:
        return self.package_dir / "simid_open_human_labels.jsonl"

    @property
    def calibration_queue(self) -> Path:
        return self.run_dir / "open_calibration_queue.jsonl"

    @property
    def secondary_labels(self) -> Path:
        return self.run_dir / "open_calibration_secondary_labels.jsonl"

    @property
    def adjudications(self) -> Path:
        return self.run_dir / "open_calibration_adjudications.jsonl"

    @property
    def calibration_summary(self) -> Path:
        return self.run_dir / "open_calibration_summary.json"

    @property
    def mvp_open_adjudication(self) -> Path:
        return self.run_dir / "open_adjudication.jsonl"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build duplicate-collapsed correction-evidence and diagnostic "
            "propagation tables from the fresh targeted SIMID adjudication pass."
        )
    )
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--package-dir", type=Path, default=DEFAULT_PACKAGE_DIR)
    parser.add_argument(
        "--fresh-package-dir",
        type=Path,
        default=DEFAULT_FRESH_PACKAGE_DIR,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing existing correction-evidence outputs.",
    )
    return parser.parse_args(argv)


def stable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            rows.append(payload)
    return rows


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def index_by_case_id(
    rows: Iterable[dict[str, Any]], *, row_kind: str
) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        case_id = row.get("calibration_case_id")
        if not isinstance(case_id, str) or not case_id:
            raise ValueError(f"{row_kind} row missing calibration_case_id")
        if case_id in indexed:
            raise ValueError(f"Duplicate {row_kind} row for {case_id}")
        indexed[case_id] = row
    return indexed


def index_by_review_order(
    rows: Iterable[dict[str, Any]], *, row_kind: str
) -> dict[int, dict[str, Any]]:
    indexed: dict[int, dict[str, Any]] = {}
    for row in rows:
        review_order = row.get("review_order")
        if type(review_order) is not int:
            raise ValueError(f"{row_kind} row missing integer review_order")
        if review_order in indexed:
            raise ValueError(
                f"Duplicate {row_kind} row for review_order={review_order}"
            )
        indexed[review_order] = row
    return indexed


def grade_from_row(row: dict[str, Any], *, row_kind: str) -> str:
    raw = (
        row.get("label")
        or row.get("final_grade")
        or row.get("judge_grade")
        or row.get("primary_judge_grade")
    )
    if raw is None and isinstance(row.get("primary_effective_open_grade"), dict):
        raw = row["primary_effective_open_grade"].get("judge_grade")
    if not isinstance(raw, str) or raw not in VALID_LABELS:
        raise ValueError(f"{row_kind} has invalid grade {raw!r}")
    return raw


def label_counts(labels: Iterable[str | None]) -> dict[str, int]:
    return dict(Counter(label if label is not None else "MISSING" for label in labels))


def relation_to_fresh(labels: Sequence[str | None], fresh_label: str) -> str:
    present = [label for label in labels if label is not None]
    if not present:
        return "missing"
    agree = sum(1 for label in present if label == fresh_label)
    if agree == len(present):
        return "all_agree"
    if agree == 0:
        return "all_disagree"
    return "mixed"


def reference_conflict_status(
    *,
    reference_labels: Sequence[str | None],
    fresh_label: str,
    duplicate_group: bool,
) -> str:
    present = [label for label in reference_labels if label is not None]
    if len(present) != len(reference_labels):
        return "missing_reference_label"
    if len(set(present)) > 1:
        return "mixed_prior_reference_labels"
    prefix = "duplicate_group" if duplicate_group else "single_row"
    suffix = (
        "agrees_with_fresh" if present[0] == fresh_label else "disagrees_with_fresh"
    )
    return f"{prefix}_{suffix}"


def normalize_text(value: str) -> str:
    text = value.casefold().strip()
    text = text.strip("\"'`")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[.!?]+$", "", text)
    return text.strip()


def case_family(row: dict[str, Any]) -> str:
    reasons = set(row.get("selection_reasons") or [])
    normalized_answer = str(row.get("predicted_answer_normalized") or "")
    question = str(row.get("question") or "")
    if "cranberry_modifier" in reasons:
        return "cranberry_modifier_exact_repeat"
    if "core_i9_apple_silicon" in reasons:
        return "core_i9_apple_silicon_wrong_extra"
    if "core_i9_modifier" in reasons:
        return "core_i9_processor_brand_boundary"
    if "pithivier_custard" in reasons:
        return "pithivier_custard_modifier"
    if "pithivier_fruit" in reasons:
        return "pithivier_fruit_modifier"
    if "viscous_fluid" in reasons:
        return "viscous_fluid_modifier"
    if "adam_first_man" in reasons:
        return "adam_first_man_truthfulqa_boundary"
    if "peter_piper_peppers" in reasons:
        return "peter_piper_plain_peppers"
    if normalized_answer == "lawyer":
        return "amal_clooney_plain_lawyer"
    if "human rights advocate" in normalized_answer:
        return "amal_clooney_lawyer_human_rights_advocate"
    safe_question = re.sub(r"[^a-z0-9]+", "_", normalize_text(question)).strip("_")
    safe_answer = re.sub(r"[^a-z0-9]+", "_", normalize_text(normalized_answer)).strip(
        "_"
    )
    return f"{safe_question}__{safe_answer}"[:120]


def propagation_note(row: dict[str, Any]) -> str:
    family = case_family(row)
    if family == "core_i9_processor_brand_boundary":
        return (
            "Exact Core i9 rows can be diagnostically marked CORRECT; do not "
            "generalize this to every Core-i* processor-brand response."
        )
    if family == "core_i9_apple_silicon_wrong_extra":
        return (
            "Exact Core i9 Apple Silicon rows can be diagnostically marked "
            "INCORRECT because Apple Silicon is a wrong extra answer; do not "
            "generalize to unreviewed Core-i* variants."
        )
    if family.startswith("pithivier_"):
        return (
            "Exact pithivier rows with the reviewed filling-only pastry answer "
            "can be marked INCORRECT; broader pithivier phrasing needs review."
        )
    if family == "amal_clooney_plain_lawyer":
        return (
            "Exact plain-lawyer rows can be marked INCORRECT; broader biography "
            "answers need separate review unless they contain the human-rights "
            "substance."
        )
    if family == "amal_clooney_lawyer_human_rights_advocate":
        return (
            "Exact lawyer-and-human-rights-advocate rows can be marked CORRECT; "
            "do not infer all lawyer-plus-modifier variants."
        )
    return (
        "Exact normalized-answer rows can be used for diagnostic propagation; "
        "broader semantic-family propagation remains out of scope."
    )


def core_family_match(row: dict[str, Any], target_question: str) -> bool:
    if normalize_text(str(row.get("question") or "")) != normalize_text(
        target_question
    ):
        return False
    return normalize_text(str(row.get("response") or "")).startswith("core i")


def exact_mvp_matches(
    *,
    mvp_rows: list[dict[str, Any]],
    question: str,
    predicted_answer_normalized: str,
) -> list[dict[str, Any]]:
    question_norm = normalize_text(question)
    answer_norm = normalize_text(predicted_answer_normalized)
    return [
        row
        for row in mvp_rows
        if normalize_text(str(row.get("question") or "")) == question_norm
        and normalize_text(str(row.get("response") or "")) == answer_norm
    ]


def mvp_match_summary(
    *,
    mvp_rows: list[dict[str, Any]],
    question: str,
    predicted_answer_normalized: str,
    include_core_family: bool,
) -> dict[str, Any]:
    exact_matches = exact_mvp_matches(
        mvp_rows=mvp_rows,
        question=question,
        predicted_answer_normalized=predicted_answer_normalized,
    )
    condition_counts = Counter(
        f"{row.get('condition')}::alpha={row.get('alpha')}" for row in exact_matches
    )
    summary: dict[str, Any] = {
        "scope": "diagnostic_only",
        "eligibility": "eligible_exact_normalized_answer"
        if exact_matches
        else "no_exact_mvp_pattern_found",
        "match_rule": {
            "question": question,
            "predicted_answer_normalized": predicted_answer_normalized,
            "normalization": "casefold, trim whitespace, strip terminal .!?",
        },
        "mvp_exact_pattern_count": len(exact_matches),
        "mvp_exact_pattern_condition_alpha_counts": dict(
            sorted(condition_counts.items())
        ),
        "broader_family_propagation": "not_allowed_without_new_review",
    }
    if include_core_family:
        core_matches = [
            row for row in mvp_rows if core_family_match(row, target_question=question)
        ]
        summary["mvp_core_i_family_pattern_count"] = len(core_matches)
        summary["core_i_family_guardrail"] = (
            "Core i9 and Core i9 Apple Silicon have opposite fresh labels; "
            "the broader Core-i* family must not be collapsed into one rule."
        )
    return summary


def compare_counts(labels: Sequence[str | None], fresh_label: str) -> dict[str, Any]:
    present = [label for label in labels if label is not None]
    agree = sum(1 for label in present if label == fresh_label)
    return {
        "available": len(present),
        "agree_with_fresh": agree,
        "disagree_with_fresh": len(present) - agree,
        "relation_to_fresh": relation_to_fresh(labels, fresh_label),
        "label_counts": label_counts(labels),
    }


def build_rows(
    *,
    sources: SourcePaths,
    mvp_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    boundary_rows = load_jsonl(sources.boundary_status)
    fresh_by_case = index_by_case_id(
        load_jsonl(sources.fresh_resolved_labels), row_kind="fresh resolved label"
    )
    review_by_case = index_by_case_id(
        load_jsonl(sources.review_cases), row_kind="review case"
    )
    opus_by_case = index_by_case_id(load_jsonl(sources.opus_labels), row_kind="Opus")
    human_by_case = index_by_case_id(load_jsonl(sources.human_labels), row_kind="human")
    queue_by_case = index_by_case_id(
        load_jsonl(sources.calibration_queue), row_kind="calibration queue"
    )
    secondary_by_case = index_by_case_id(
        load_jsonl(sources.secondary_labels), row_kind="secondary label"
    )
    adjudication_by_case = index_by_case_id(
        load_jsonl(sources.adjudications), row_kind="production adjudication"
    )
    summary = load_json(sources.calibration_summary)
    reference_by_case = index_by_case_id(
        summary.get("adjudicated_rows", []), row_kind="calibration summary"
    )

    output_rows: list[dict[str, Any]] = []
    for row in boundary_rows:
        group_id = str(row["group_id"])
        case_ids = list(row["calibration_case_ids"])
        review_orders = list(row["review_orders"])
        fresh_label = str(row["label"])
        if fresh_label not in VALID_LABELS:
            raise ValueError(f"{group_id}: invalid fresh label {fresh_label!r}")
        if set(case_ids) - set(fresh_by_case):
            raise ValueError(f"{group_id}: missing fresh label rows")
        fresh_labels = [
            grade_from_row(fresh_by_case[case_id], row_kind="fresh resolved label")
            for case_id in case_ids
        ]
        status_labels = {str(label) for label in row["labels"]}
        if set(fresh_labels) != status_labels:
            raise ValueError(f"{group_id}: status labels do not match fresh rows")
        for case_id in case_ids:
            for source_name, source_map in (
                ("review case", review_by_case),
                ("Opus", opus_by_case),
                ("human", human_by_case),
                ("queue", queue_by_case),
                ("secondary", secondary_by_case),
                ("reference", reference_by_case),
            ):
                if case_id not in source_map:
                    raise ValueError(f"{group_id}: missing {source_name} for {case_id}")

        reference_labels = [
            grade_from_row(reference_by_case[case_id], row_kind="reference")
            for case_id in case_ids
        ]
        primary_labels = [
            grade_from_row(queue_by_case[case_id], row_kind="primary queue")
            for case_id in case_ids
        ]
        secondary_labels = [
            grade_from_row(secondary_by_case[case_id], row_kind="secondary")
            for case_id in case_ids
        ]
        opus_labels = [
            grade_from_row(opus_by_case[case_id], row_kind="Opus")
            for case_id in case_ids
        ]
        human_labels = [
            grade_from_row(human_by_case[case_id], row_kind="human")
            for case_id in case_ids
        ]
        adjudicated_case_ids = [
            case_id for case_id in case_ids if case_id in adjudication_by_case
        ]
        question = str(row["question"])
        predicted_answer_normalized = str(row["predicted_answer_normalized"])
        family = case_family(row)
        include_core_family = family.startswith("core_i9")
        diagnostic_propagation = mvp_match_summary(
            mvp_rows=mvp_rows,
            question=question,
            predicted_answer_normalized=predicted_answer_normalized,
            include_core_family=include_core_family,
        )
        diagnostic_propagation["note"] = propagation_note(row)
        local_status = (
            "unresolved_do_not_apply"
            if row.get("adjudication_status") != "ADJUDICATED"
            or row.get("rule_gap")
            or row.get("status") != "consistent"
            else "eligible_targeted_rows"
        )
        output_rows.append(
            {
                "schema_version": OUTPUT_SCHEMA_VERSION,
                "case_family": family,
                "group_id": group_id,
                "duplicate_group": bool(row["duplicate_group"]),
                "adjudication_status": row["adjudication_status"],
                "fresh_status": row["status"],
                "fresh_label": fresh_label,
                "fresh_label_counts": label_counts(fresh_labels),
                "confidence_range": {
                    "min": int(row["confidence_min"]),
                    "max": int(row["confidence_max"]),
                },
                "rule_gap": bool(row["rule_gap"]),
                "review_orders": review_orders,
                "calibration_case_ids": case_ids,
                "question": question,
                "gold_aliases": row["gold_aliases"],
                "predicted_answer_examples": row["predicted_answer_examples"],
                "predicted_answer_normalized": predicted_answer_normalized,
                "selection_reasons": row["selection_reasons"],
                "flags": row["flags"],
                "local_correction_evidence": {
                    "status": local_status,
                    "applies_to_review_orders": review_orders
                    if local_status == "eligible_targeted_rows"
                    else [],
                    "notes": row["notes"],
                },
                "current_reference_comparison": {
                    "prior_reference_conflict_status": reference_conflict_status(
                        reference_labels=reference_labels,
                        fresh_label=fresh_label,
                        duplicate_group=bool(row["duplicate_group"]),
                    ),
                    **compare_counts(reference_labels, fresh_label),
                    "labels_by_review_order": dict(
                        zip([str(order) for order in review_orders], reference_labels)
                    ),
                    "label_source_counts": dict(
                        Counter(
                            str(reference_by_case[case_id].get("label_source"))
                            for case_id in case_ids
                        )
                    ),
                    "production_primary_label_counts": label_counts(primary_labels),
                    "production_secondary_label_counts": label_counts(secondary_labels),
                    "production_adjudicated_case_ids": adjudicated_case_ids,
                },
                "opus_comparison": {
                    **compare_counts(opus_labels, fresh_label),
                    "labels_by_review_order": dict(
                        zip([str(order) for order in review_orders], opus_labels)
                    ),
                },
                "human_comparison": {
                    **compare_counts(human_labels, fresh_label),
                    "labels_by_review_order": dict(
                        zip([str(order) for order in review_orders], human_labels)
                    ),
                },
                "diagnostic_propagation": diagnostic_propagation,
                "claimability_guardrail": (
                    "Measurement-cleanup evidence only. SIMID open correctness "
                    "remains diagnostic-only unless a separate prospective "
                    "calibration gate passes."
                ),
            }
        )
    return output_rows


def validate_output_rows(
    rows: list[dict[str, Any]], sources: SourcePaths
) -> dict[str, Any]:
    validation = load_json(sources.fresh_validation_summary)
    review_orders = sorted(order for row in rows for order in row["review_orders"])
    expected_orders = sorted(
        order
        for group in validation.get("duplicate_groups", [])
        for order in group.get("review_orders", [])
    )
    singletons = [
        row
        for row in load_jsonl(sources.boundary_status)
        if not bool(row.get("duplicate_group"))
    ]
    expected_orders = sorted(
        expected_orders
        + [order for row in singletons for order in row.get("review_orders", [])]
    )
    if review_orders != expected_orders:
        raise ValueError("Correction-evidence rows do not cover the fresh package rows")
    if len(set(review_orders)) != len(review_orders):
        raise ValueError("Duplicate review_order in correction-evidence output")
    schema_errors: list[str] = []
    comparison_coverage_ok = True
    fresh_label_count_totals: Counter[str] = Counter()
    for row in rows:
        missing_keys = REQUIRED_OUTPUT_KEYS - set(row)
        if missing_keys:
            schema_errors.append(
                f"{row.get('group_id')}: missing keys {sorted(missing_keys)}"
            )
        if row.get("schema_version") != OUTPUT_SCHEMA_VERSION:
            schema_errors.append(f"{row.get('group_id')}: bad schema_version")
        if row.get("fresh_label") not in VALID_LABELS:
            schema_errors.append(f"{row.get('group_id')}: bad fresh_label")
        n_review_orders = len(row["review_orders"])
        row_fresh_label_total = sum(row["fresh_label_counts"].values())
        if row_fresh_label_total != n_review_orders:
            schema_errors.append(
                f"{row.get('group_id')}: fresh_label_counts coverage mismatch"
            )
        fresh_label_count_totals.update(row["fresh_label_counts"])
        for comparison_key in (
            "current_reference_comparison",
            "opus_comparison",
            "human_comparison",
        ):
            if row[comparison_key]["available"] != n_review_orders:
                comparison_coverage_ok = False
                schema_errors.append(
                    f"{row.get('group_id')}: {comparison_key} coverage mismatch"
                )
        if row["rule_gap"] or row["fresh_status"] != "consistent":
            if row["local_correction_evidence"]["status"] != "unresolved_do_not_apply":
                schema_errors.append(
                    f"{row.get('group_id')}: unresolved row propagates"
                )
        if (
            row["diagnostic_propagation"]["broader_family_propagation"]
            != "not_allowed_without_new_review"
        ):
            schema_errors.append(f"{row.get('group_id')}: broad propagation allowed")
    expected_label_counts = validation.get("label_counts")
    if isinstance(expected_label_counts, dict):
        normalized_expected_label_counts = {
            str(label): int(count) for label, count in expected_label_counts.items()
        }
        if dict(sorted(fresh_label_count_totals.items())) != dict(
            sorted(normalized_expected_label_counts.items())
        ):
            schema_errors.append("fresh_label_counts do not match fresh validation")
    if schema_errors:
        raise ValueError("; ".join(schema_errors))

    reference_status_counts = Counter(
        row["current_reference_comparison"]["prior_reference_conflict_status"]
        for row in rows
    )
    propagation_counts = Counter(
        row["diagnostic_propagation"]["eligibility"] for row in rows
    )
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "validated_at_utc": datetime.now(UTC).isoformat(),
        "n_rows": len(rows),
        "n_review_orders": len(review_orders),
        "review_orders": review_orders,
        "expected_review_orders": expected_orders,
        "schema_checks": {
            "required_fields_present": True,
            "valid_fresh_labels": True,
            "fresh_label_counts_match_validation": True,
            "all_comparisons_cover_review_orders": comparison_coverage_ok,
            "no_unresolved_silent_propagation": True,
        },
        "coverage_checks": {
            "review_orders_match_fresh_package": review_orders == expected_orders,
            "no_duplicate_review_orders": len(set(review_orders)) == len(review_orders),
        },
        "fresh_validation_label_counts": validation.get("label_counts"),
        "fresh_validation_rule_gap_count": validation.get("rule_gap_count"),
        "fresh_validation_unresolved_cases": validation.get("unresolved_cases"),
        "fresh_duplicate_group_count": validation.get("duplicate_group_count"),
        "fresh_inconsistent_duplicate_group_count": validation.get(
            "inconsistent_duplicate_group_count"
        ),
        "label_counts": dict(sorted(fresh_label_count_totals.items())),
        "duplicate_collapsed_label_counts": dict(
            sorted(Counter(row["fresh_label"] for row in rows).items())
        ),
        "reference_conflict_status_counts": dict(
            sorted(reference_status_counts.items())
        ),
        "opus_relation_counts": dict(
            sorted(
                Counter(
                    row["opus_comparison"]["relation_to_fresh"] for row in rows
                ).items()
            )
        ),
        "human_relation_counts": dict(
            sorted(
                Counter(
                    row["human_comparison"]["relation_to_fresh"] for row in rows
                ).items()
            )
        ),
        "propagation_eligibility_counts": dict(sorted(propagation_counts.items())),
        "local_unresolved_or_nonpropagatable_rows": [
            row["group_id"]
            for row in rows
            if row["local_correction_evidence"]["status"] != "eligible_targeted_rows"
        ],
        "all_broader_family_propagation_blocked": all(
            row["diagnostic_propagation"]["broader_family_propagation"]
            == "not_allowed_without_new_review"
            for row in rows
        ),
    }


def compact_count(counts: dict[str, int]) -> str:
    return ", ".join(f"{key}:{value}" for key, value in sorted(counts.items()))


def markdown_list(values: Iterable[Any]) -> str:
    return ", ".join(str(value) for value in values)


def build_markdown(rows: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    lines = [
        "# SIMID Boundary Correction Evidence",
        "",
        "This table is duplicate-collapsed measurement-cleanup evidence from the "
        "fresh 2026-04-29 blinded adjudication pass. It does not make SIMID "
        "open correctness claim-bearing.",
        "",
        "## Validation Summary",
        "",
        f"- Evidence rows: {summary['n_rows']}",
        f"- Covered review rows: {summary['n_review_orders']}",
        f"- Fresh label counts: {compact_count(summary['label_counts'])}",
        f"- Fresh rule gaps: {summary['fresh_validation_rule_gap_count']}",
        f"- Fresh unresolved cases: {len(summary['fresh_validation_unresolved_cases'])}",
        "- Broader semantic-family propagation: blocked unless separately reviewed",
        "",
        "## Local Correction Evidence",
        "",
        "| Family | Review orders | Fresh label | Conf. | Current reference | Opus | Human | Local status |",
        "|---|---:|---|---:|---|---|---|---|",
    ]
    for row in rows:
        reference = row["current_reference_comparison"]
        opus = row["opus_comparison"]
        human = row["human_comparison"]
        lines.append(
            "| "
            f"`{row['case_family']}` | "
            f"{markdown_list(row['review_orders'])} | "
            f"{row['fresh_label']} | "
            f"{row['confidence_range']['min']}-{row['confidence_range']['max']} | "
            f"{reference['prior_reference_conflict_status']} "
            f"({compact_count(reference['label_counts'])}) | "
            f"{opus['relation_to_fresh']} ({compact_count(opus['label_counts'])}) | "
            f"{human['relation_to_fresh']} ({compact_count(human['label_counts'])}) | "
            f"{row['local_correction_evidence']['status']} |"
        )
    lines.extend(
        [
            "",
            "## Diagnostic Propagation Guardrails",
            "",
            "| Family | Exact MVP rows | Propagation scope | Guardrail |",
            "|---|---:|---|---|",
        ]
    )
    for row in rows:
        propagation = row["diagnostic_propagation"]
        guardrail = propagation["note"]
        if "mvp_core_i_family_pattern_count" in propagation:
            guardrail = (
                f"{guardrail} Core-i* family rows: "
                f"{propagation['mvp_core_i_family_pattern_count']}."
            )
        lines.append(
            "| "
            f"`{row['case_family']}` | "
            f"{propagation['mvp_exact_pattern_count']} | "
            f"{propagation['eligibility']} | "
            f"{guardrail} |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `eligible_targeted_rows` means the fresh label can be used for the "
            "adjudicated local review rows in diagnostic correction analyses.",
            "- `eligible_exact_normalized_answer` means diagnostic propagation is "
            "limited to MVP rows with the same question and same normalized model "
            "answer.",
            "- Every broader semantic-family propagation path is explicitly blocked "
            "in this artifact.",
            "",
        ]
    )
    return "\n".join(lines)


def build_readme() -> str:
    return """# Correction Evidence 2026-04-29

Append-only duplicate-collapsed correction-evidence table derived from the fresh
targeted blind adjudication pass. This is measurement cleanup only; it does not
make SIMID open correctness claim-bearing.

Files:

- `duplicate_collapsed_correction_evidence.jsonl` - machine-readable local
  correction evidence plus diagnostic exact-answer propagation guardrails.
- `duplicate_collapsed_correction_evidence.md` - compact human-readable view of
  the same rows.
- `validation_summary.json` - schema, coverage, comparison, and propagation
  checks.
- `correction_evidence.provenance.20260429_*.json` - source and output hashes.

Regenerate from the repository root with:

```bash
uv run python scripts/build_simid_boundary_correction_evidence.py --overwrite
```
"""


def git_sha() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def build_provenance(
    *,
    sources: SourcePaths,
    outputs: list[Path],
    output_dir: Path,
    generated_at_utc: str,
) -> dict[str, Any]:
    input_paths = [
        Path(__file__).resolve(),
        sources.boundary_status,
        sources.fresh_resolved_labels,
        sources.fresh_validation_summary,
        sources.review_cases,
        sources.opus_labels,
        sources.human_labels,
        sources.calibration_queue,
        sources.secondary_labels,
        sources.adjudications,
        sources.calibration_summary,
        sources.mvp_open_adjudication,
    ]
    return {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "generated_at_utc": generated_at_utc,
        "git_sha": git_sha(),
        "command": (
            "uv run python scripts/build_simid_boundary_correction_evidence.py "
            "--overwrite"
        ),
        "output_dir": stable_path(output_dir),
        "inputs": {
            stable_path(path): {"sha256": file_sha256(path)} for path in input_paths
        },
        "outputs": {
            stable_path(path): {"sha256": file_sha256(path)} for path in outputs
        },
    }


def guard_outputs(paths: Iterable[Path], *, overwrite: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        formatted = "\n".join(f"- {path}" for path in existing)
        raise FileExistsError(
            f"Correction-evidence outputs already exist; pass --overwrite:\n{formatted}"
        )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    sources = SourcePaths(
        run_dir=args.run_dir.resolve(),
        package_dir=args.package_dir.resolve(),
        fresh_package_dir=args.fresh_package_dir.resolve(),
    )
    output_dir = args.output_dir.resolve()
    evidence_path = output_dir / "duplicate_collapsed_correction_evidence.jsonl"
    markdown_path = output_dir / "duplicate_collapsed_correction_evidence.md"
    summary_path = output_dir / "validation_summary.json"
    readme_path = output_dir / "README.md"
    provenance_suffix = datetime.now(UTC).strftime("%H%M%S")
    provenance_path = (
        output_dir / f"correction_evidence.provenance.20260429_{provenance_suffix}.json"
    )
    output_paths = [
        evidence_path,
        markdown_path,
        summary_path,
        readme_path,
        provenance_path,
    ]
    guard_outputs(output_paths, overwrite=bool(args.overwrite))
    output_dir.mkdir(parents=True, exist_ok=True)

    mvp_rows = load_jsonl(sources.mvp_open_adjudication)
    rows = build_rows(sources=sources, mvp_rows=mvp_rows)
    rows.sort(key=lambda row: row["review_orders"][0])
    summary = validate_output_rows(rows, sources)

    write_jsonl(evidence_path, rows)
    markdown_path.write_text(build_markdown(rows, summary), encoding="utf-8")
    write_json(summary_path, summary)
    readme_path.write_text(build_readme(), encoding="utf-8")
    provenance = build_provenance(
        sources=sources,
        outputs=[evidence_path, markdown_path, summary_path, readme_path],
        output_dir=output_dir,
        generated_at_utc=datetime.now(UTC).isoformat(),
    )
    write_json(provenance_path, provenance)

    print(f"Wrote {len(rows)} correction-evidence rows to {stable_path(evidence_path)}")
    print(f"Validation summary: {stable_path(summary_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
