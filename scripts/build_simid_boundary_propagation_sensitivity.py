#!/usr/bin/env python3
"""Build SIMID exact-answer diagnostic propagation sensitivity artifacts."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
from typing import Any

from analyze_simid import (
    DEFAULT_BOOTSTRAP_RESAMPLES,
    DEFAULT_BOOTSTRAP_SEED,
    SIMID_OPEN_FINAL_GRADES,
    attach_open_adjudications,
    effective_open_grade,
    effective_open_attempted,
    effective_open_correct,
    index_rows,
    load_open_adjudications,
    load_run_rows,
    paired_bool_delta,
    rate_at_alpha,
    validate_complete_run_outputs,
)
from analyze_simid import (
    analysis_stratum_key,
    analysis_stratum_sample_ids,
    require_paired_panel,
    row_mc_endpoint,
)
from build_simid_boundary_correction_evidence import normalize_text
from utils import (
    finish_run_provenance,
    provenance_error_message,
    provenance_status_for_exception,
    start_run_provenance,
)


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RUN_DIR = (
    ROOT / "data/gemma3_4b/intervention/"
    "simid_iti_truthfulqa-paperfaithful_k12_first-3-tokens/"
    "mvp_20260427_calibration"
)
DEFAULT_FRESH_PACKAGE_DIR = (
    DEFAULT_RUN_DIR / "human_review_package" / "fresh_blind_adjudication_20260429"
)
DEFAULT_EVIDENCE_JSONL = (
    DEFAULT_FRESH_PACKAGE_DIR
    / "correction_evidence_20260429"
    / "duplicate_collapsed_correction_evidence.jsonl"
)
DEFAULT_OUTPUT_DIR = DEFAULT_FRESH_PACKAGE_DIR / "diagnostic_exact_propagation_20260429"

ROW_CORRECTION_SCHEMA_VERSION = "simid_diagnostic_exact_propagation_row/v1"
SUMMARY_SCHEMA_VERSION = "simid_diagnostic_exact_propagation_summary/v1"
NORMALIZATION_DESCRIPTION = "casefold, trim whitespace, strip terminal .!?"
VALID_LABELS = set(SIMID_OPEN_FINAL_GRADES)
EXACT_APPLIED_STATUSES = {"exact_label_confirmed", "exact_label_replaced"}
MetricFn = Callable[[dict[str, Any]], float | bool]


@dataclass(frozen=True)
class EvidenceRule:
    case_family: str
    group_id: str
    question: str
    question_normalized: str
    predicted_answer_normalized: str
    answer_normalized: str
    fresh_label: str
    review_orders: tuple[int, ...]
    diagnostic_eligibility: str
    local_status: str
    fresh_status: str
    adjudication_status: str
    rule_gap: bool
    mvp_exact_pattern_count: int
    broader_family_propagation: str
    note: str
    core_i_family_pattern_count: int | None

    @property
    def exact_key(self) -> tuple[str, str]:
        return (self.question_normalized, self.answer_normalized)

    @property
    def propagatable(self) -> bool:
        return (
            self.diagnostic_eligibility == "eligible_exact_normalized_answer"
            and self.local_status == "eligible_targeted_rows"
            and self.fresh_status == "consistent"
            and self.adjudication_status == "ADJUDICATED"
            and not self.rule_gap
            and self.broader_family_propagation == "not_allowed_without_new_review"
        )


@dataclass(frozen=True)
class EvidenceIndex:
    by_exact_key: dict[tuple[str, str], EvidenceRule]
    by_question: dict[str, list[EvidenceRule]]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--evidence-jsonl", type=Path, default=DEFAULT_EVIDENCE_JSONL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--baseline-alpha", type=float, default=0.0)
    parser.add_argument("--alphas", type=float, nargs="+", default=None)
    parser.add_argument("--conditions", nargs="+", default=None)
    parser.add_argument("--n-resamples", type=int, default=DEFAULT_BOOTSTRAP_RESAMPLES)
    parser.add_argument("--seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    parser.add_argument("--tolerance", type=float, default=1e-9)
    parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow replacing the non-provenance diagnostic output files.",
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


def require_str(value: Any, *, field: str, row_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{row_name}: invalid {field}")
    return value


def require_int(value: Any, *, field: str, row_name: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{row_name}: invalid {field}")
    return value


def label_to_grade(label: str, *, source: str) -> dict[str, Any]:
    if label not in VALID_LABELS:
        raise ValueError(f"Invalid SIMID open label: {label!r}")
    return {
        "label": label,
        "correct": label == "CORRECT",
        "attempted": label in {"CORRECT", "INCORRECT"},
        "source": source,
    }


def label_from_effective_grade(grade: dict[str, Any]) -> str:
    raw = grade.get("judge_grade")
    if isinstance(raw, str) and raw in VALID_LABELS:
        return raw
    if not bool(grade.get("attempted")):
        return "NOT_ATTEMPTED"
    return "CORRECT" if bool(grade.get("correct")) else "INCORRECT"


def row_response(row: dict[str, Any]) -> str:
    open_generation = row.get("open_generation")
    if (
        isinstance(open_generation, dict)
        and open_generation.get("response") is not None
    ):
        return str(open_generation["response"])
    if row.get("response") is not None:
        return str(row["response"])
    return ""


def row_identity(row: dict[str, Any]) -> dict[str, Any]:
    identity: dict[str, Any] = {
        "sample_id": row.get("sample_id"),
        "base_sample_id": row.get("base_sample_id") or row.get("sample_id"),
        "condition": row.get("condition"),
        "alpha": row.get("alpha"),
        "dataset": row.get("dataset"),
        "mc_endpoint": row_mc_endpoint(row),
    }
    if "option_order_replicate" in row:
        identity["option_order_replicate"] = row.get("option_order_replicate")
    return identity


def evidence_rule_from_row(row: dict[str, Any], *, row_name: str) -> EvidenceRule:
    diagnostic = row.get("diagnostic_propagation")
    if not isinstance(diagnostic, dict):
        raise ValueError(f"{row_name}: missing diagnostic_propagation")
    match_rule = diagnostic.get("match_rule")
    if not isinstance(match_rule, dict):
        raise ValueError(f"{row_name}: missing diagnostic match_rule")
    normalization = require_str(
        match_rule.get("normalization"),
        field="diagnostic_propagation.match_rule.normalization",
        row_name=row_name,
    )
    if normalization != NORMALIZATION_DESCRIPTION:
        raise ValueError(
            f"{row_name}: unsupported evidence normalization {normalization!r}"
        )

    local = row.get("local_correction_evidence")
    if not isinstance(local, dict):
        raise ValueError(f"{row_name}: missing local_correction_evidence")
    review_orders_raw = row.get("review_orders")
    if not isinstance(review_orders_raw, list) or not review_orders_raw:
        raise ValueError(f"{row_name}: missing review_orders")
    review_orders = tuple(
        require_int(order, field="review_orders", row_name=row_name)
        for order in review_orders_raw
    )
    fresh_label = require_str(
        row.get("fresh_label"), field="fresh_label", row_name=row_name
    )
    if fresh_label not in VALID_LABELS:
        raise ValueError(f"{row_name}: invalid fresh_label {fresh_label!r}")
    question = require_str(row.get("question"), field="question", row_name=row_name)
    predicted_answer_normalized = require_str(
        row.get("predicted_answer_normalized"),
        field="predicted_answer_normalized",
        row_name=row_name,
    )
    exact_count = diagnostic.get("mvp_exact_pattern_count")
    if type(exact_count) is not int or exact_count < 0:
        raise ValueError(f"{row_name}: invalid mvp_exact_pattern_count")

    core_count_raw = diagnostic.get("mvp_core_i_family_pattern_count")
    core_count = int(core_count_raw) if type(core_count_raw) is int else None
    return EvidenceRule(
        case_family=require_str(
            row.get("case_family"), field="case_family", row_name=row_name
        ),
        group_id=require_str(row.get("group_id"), field="group_id", row_name=row_name),
        question=question,
        question_normalized=normalize_text(question),
        predicted_answer_normalized=predicted_answer_normalized,
        answer_normalized=normalize_text(predicted_answer_normalized),
        fresh_label=fresh_label,
        review_orders=review_orders,
        diagnostic_eligibility=require_str(
            diagnostic.get("eligibility"),
            field="diagnostic_propagation.eligibility",
            row_name=row_name,
        ),
        local_status=require_str(
            local.get("status"),
            field="local_correction_evidence.status",
            row_name=row_name,
        ),
        fresh_status=require_str(
            row.get("fresh_status"), field="fresh_status", row_name=row_name
        ),
        adjudication_status=require_str(
            row.get("adjudication_status"),
            field="adjudication_status",
            row_name=row_name,
        ),
        rule_gap=bool(row.get("rule_gap")),
        mvp_exact_pattern_count=exact_count,
        broader_family_propagation=require_str(
            diagnostic.get("broader_family_propagation"),
            field="diagnostic_propagation.broader_family_propagation",
            row_name=row_name,
        ),
        note=require_str(
            diagnostic.get("note"),
            field="diagnostic_propagation.note",
            row_name=row_name,
        ),
        core_i_family_pattern_count=core_count,
    )


def load_evidence_rules(path: Path) -> list[EvidenceRule]:
    rules = [
        evidence_rule_from_row(row, row_name=f"{path}:{index}")
        for index, row in enumerate(load_jsonl(path), start=1)
    ]
    if not rules:
        raise ValueError(f"No evidence rows found in {path}")
    build_evidence_index(rules)
    return rules


def build_evidence_index(rules: Iterable[EvidenceRule]) -> EvidenceIndex:
    by_exact_key: dict[tuple[str, str], EvidenceRule] = {}
    by_question: dict[str, list[EvidenceRule]] = defaultdict(list)
    for rule in rules:
        previous = by_exact_key.get(rule.exact_key)
        if previous is not None:
            raise ValueError(
                "Duplicate diagnostic propagation evidence for exact normalized "
                f"key {rule.exact_key!r}: {previous.group_id} and {rule.group_id}"
            )
        by_exact_key[rule.exact_key] = rule
        by_question[rule.question_normalized].append(rule)
    return EvidenceIndex(
        by_exact_key=by_exact_key,
        by_question={
            key: sorted(value, key=lambda rule: rule.group_id)
            for key, value in by_question.items()
        },
    )


def is_core_i_question_rule(rule: EvidenceRule) -> bool:
    return rule.core_i_family_pattern_count is not None or rule.case_family.startswith(
        "core_i9"
    )


def blocked_status_for_nonexact_row(
    *,
    rules_for_question: list[EvidenceRule],
    response_normalized: str,
) -> str:
    if any(is_core_i_question_rule(rule) for rule in rules_for_question) and (
        response_normalized.startswith("core i")
    ):
        return "blocked_core_i_family"
    return "blocked_same_question_nonexact"


def row_correction_for_rule(
    *,
    row: dict[str, Any],
    rule: EvidenceRule | None,
    exact_key: tuple[str, str],
    evidence_index: EvidenceIndex,
) -> dict[str, Any]:
    question_normalized, response_normalized = exact_key
    original_effective_grade = effective_open_grade(row)
    original_label = label_from_effective_grade(original_effective_grade)
    corrected_label = original_label
    status: str
    evidence_rule = rule
    notes: list[str] = []

    if evidence_rule is not None:
        if evidence_rule.propagatable:
            corrected_label = evidence_rule.fresh_label
            status = (
                "exact_label_confirmed"
                if corrected_label == original_label
                else "exact_label_replaced"
            )
            notes.append(evidence_rule.note)
        else:
            status = "evidence_nonpropagatable_not_applied"
            notes.append(
                "Exact evidence row is not eligible for diagnostic propagation; "
                "label left unchanged."
            )
    else:
        rules_for_question = evidence_index.by_question.get(question_normalized, [])
        if rules_for_question:
            status = blocked_status_for_nonexact_row(
                rules_for_question=rules_for_question,
                response_normalized=response_normalized,
            )
            notes.append(
                "Same question as correction evidence, but the normalized model "
                "answer is not an exact reviewed answer; no propagation applied."
            )
        else:
            status = "no_reviewed_question_evidence"
            notes.append("No correction-evidence row covers this question.")

    original_grade = label_to_grade(
        original_label, source=str(original_effective_grade.get("source"))
    )
    corrected_grade = label_to_grade(
        corrected_label,
        source=(
            "diagnostic_exact_propagation"
            if status in EXACT_APPLIED_STATUSES
            else original_grade["source"]
        ),
    )
    label_changed = original_label != corrected_label
    correction = {
        "schema_version": ROW_CORRECTION_SCHEMA_VERSION,
        **row_identity(row),
        "question": row.get("question"),
        "response": row_response(row),
        "question_normalized": question_normalized,
        "response_normalized": response_normalized,
        "original_open_label": original_label,
        "corrected_diagnostic_open_label": corrected_label,
        "original_open_grade": original_grade,
        "corrected_diagnostic_open_grade": corrected_grade,
        "label_changed": label_changed,
        "propagation_status": status,
        "evidence_family": evidence_rule.case_family if evidence_rule else None,
        "evidence_group_id": evidence_rule.group_id if evidence_rule else None,
        "evidence_fresh_label": evidence_rule.fresh_label if evidence_rule else None,
        "evidence_review_orders": list(evidence_rule.review_orders)
        if evidence_rule
        else [],
        "propagation_notes": notes,
    }
    if status not in EXACT_APPLIED_STATUSES and label_changed:
        raise ValueError(
            "Blocked or unresolved propagation row changed label: "
            f"{row_identity(row)!r}"
        )
    return correction


def build_row_corrections(
    rows: list[dict[str, Any]],
    rules: list[EvidenceRule],
) -> list[dict[str, Any]]:
    evidence_index = build_evidence_index(rules)
    corrections: list[dict[str, Any]] = []
    for row in rows:
        question = str(row.get("question") or "")
        response = row_response(row)
        exact_key = (normalize_text(question), normalize_text(response))
        rule = evidence_index.by_exact_key.get(exact_key)
        correction = row_correction_for_rule(
            row=row,
            rule=rule,
            exact_key=exact_key,
            evidence_index=evidence_index,
        )
        row["diagnostic_exact_open_grade"] = correction[
            "corrected_diagnostic_open_grade"
        ]
        corrections.append(correction)
    return corrections


def diagnostic_exact_open_correct(row: dict[str, Any]) -> bool:
    grade = row.get("diagnostic_exact_open_grade")
    if not isinstance(grade, dict):
        raise ValueError("row missing diagnostic_exact_open_grade")
    return bool(grade["correct"])


def diagnostic_exact_open_attempted(row: dict[str, Any]) -> bool:
    grade = row.get("diagnostic_exact_open_grade")
    if not isinstance(grade, dict):
        raise ValueError("row missing diagnostic_exact_open_grade")
    return bool(grade["attempted"])


OPEN_METRICS: dict[str, MetricFn] = {
    "adjudicated_open_correct": effective_open_correct,
    "adjudicated_open_attempted": effective_open_attempted,
    "diagnostic_exact_open_correct": diagnostic_exact_open_correct,
    "diagnostic_exact_open_attempted": diagnostic_exact_open_attempted,
}


def summarize_sample_ids_for_open_metrics(
    panel: dict[float, dict[str, list[dict[str, Any]]]],
    *,
    sample_ids: list[str],
    alphas: list[float],
    baseline_alpha: float,
    n_resamples: int,
    seed: int,
) -> dict[str, Any]:
    baseline_rows = panel[baseline_alpha]
    rates: dict[str, Any] = {}
    deltas: dict[str, Any] = {}
    for alpha in alphas:
        rows_by_id = panel[alpha]
        rates[str(alpha)] = {
            name: rate_at_alpha(
                sample_ids,
                rows_by_id,
                metric,
                n_resamples=n_resamples,
                seed=seed,
            )
            for name, metric in OPEN_METRICS.items()
        }
        if alpha == baseline_alpha:
            continue
        deltas[str(alpha)] = {
            name: paired_bool_delta(
                sample_ids,
                baseline_rows,
                rows_by_id,
                metric,
                n_resamples=n_resamples,
                seed=seed,
            )
            for name, metric in OPEN_METRICS.items()
        }
    return {
        "n_paired_items": len(sample_ids),
        "pairing_unit": "base_sample_id",
        "n_rows_at_baseline": sum(
            len(baseline_rows[sample_id]) for sample_id in sample_ids
        ),
        "alphas": alphas,
        "baseline_alpha": baseline_alpha,
        "rates": rates,
        "paired_deltas_vs_baseline": deltas,
    }


def summarize_condition_for_open_metrics(
    indexed: dict[str, dict[float, dict[str, list[dict[str, Any]]]]],
    *,
    condition: str,
    alphas: list[float],
    baseline_alpha: float,
    n_resamples: int,
    seed: int,
) -> dict[str, Any]:
    sample_ids, panel = require_paired_panel(
        indexed,
        condition=condition,
        alphas=alphas,
    )
    summary = summarize_sample_ids_for_open_metrics(
        panel,
        sample_ids=sample_ids,
        alphas=alphas,
        baseline_alpha=baseline_alpha,
        n_resamples=n_resamples,
        seed=seed,
    )
    strata: dict[str, Any] = {}
    for (
        dataset,
        mc_endpoint,
        leakage_metadata,
    ), stratum_ids in analysis_stratum_sample_ids(
        sample_ids,
        panel,
        alphas=alphas,
        baseline_alpha=baseline_alpha,
    ).items():
        stratum_summary = summarize_sample_ids_for_open_metrics(
            panel,
            sample_ids=stratum_ids,
            alphas=alphas,
            baseline_alpha=baseline_alpha,
            n_resamples=n_resamples,
            seed=seed,
        )
        stratum_summary["dataset"] = dataset
        stratum_summary["mc_endpoint"] = mc_endpoint
        if leakage_metadata:
            stratum_summary["truthfulqa_leakage_metadata"] = dict(leakage_metadata)
        strata[analysis_stratum_key(dataset, mc_endpoint, leakage_metadata)] = (
            stratum_summary
        )
    return {
        "condition": condition,
        "summary_scope": "pooled_across_dataset_and_mc_endpoint",
        **summary,
        "dataset_endpoint_summaries": strata,
    }


def find_dataset_scope(
    condition_summary: dict[str, Any],
    *,
    dataset: str,
) -> tuple[str, dict[str, Any]]:
    strata = condition_summary.get("dataset_endpoint_summaries")
    if not isinstance(strata, dict):
        raise ValueError("condition summary missing dataset_endpoint_summaries")
    matches = [
        (key, value)
        for key, value in strata.items()
        if isinstance(value, dict) and value.get("dataset") == dataset
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one {dataset} stratum, found {len(matches)}")
    return matches[0]


def metric_number(result: dict[str, Any], key: str) -> float:
    value = result.get(key)
    if not isinstance(value, int | float):
        raise ValueError(f"Expected numeric {key} in metric result")
    return float(value)


def compare_metric_result(
    *,
    observed: dict[str, Any],
    expected: dict[str, Any],
    estimate_key: str,
    ci_key: str,
    tolerance: float,
    cell_name: str,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    fields = [estimate_key]
    if estimate_key == "estimate":
        fields.extend(["sum", "n"])
    for field in fields:
        observed_value = metric_number(observed, field)
        expected_value = metric_number(expected, field)
        checks.append(
            {
                "field": field,
                "observed": observed_value,
                "expected": expected_value,
                "abs_diff": abs(observed_value - expected_value),
                "passed": abs(observed_value - expected_value) <= tolerance,
            }
        )
    observed_ci = observed.get(ci_key)
    expected_ci = expected.get(ci_key)
    if not isinstance(observed_ci, dict) or not isinstance(expected_ci, dict):
        raise ValueError(f"{cell_name}: missing CI payload")
    for field in ("lower", "upper"):
        observed_value = metric_number(observed_ci, field)
        expected_value = metric_number(expected_ci, field)
        checks.append(
            {
                "field": f"{ci_key}.{field}",
                "observed": observed_value,
                "expected": expected_value,
                "abs_diff": abs(observed_value - expected_value),
                "passed": abs(observed_value - expected_value) <= tolerance,
            }
        )
    passed = all(bool(check["passed"]) for check in checks)
    if not passed:
        raise ValueError(f"Uncorrected recomputation failed for {cell_name}: {checks}")
    return {"cell": cell_name, "passed": True, "checks": checks}


def selected_alpha8_reproduction_checks(
    *,
    generated_conditions: dict[str, Any],
    existing_results: dict[str, Any],
    tolerance: float,
) -> list[dict[str, Any]]:
    selected = generated_conditions["selected"]
    existing_selected = existing_results["conditions"]["selected"]
    generated_scopes: dict[str, dict[str, Any]] = {
        "pooled": selected,
        "truthfulqa": find_dataset_scope(selected, dataset="truthfulqa")[1],
        "triviaqa_bridge": find_dataset_scope(selected, dataset="triviaqa_bridge")[1],
    }
    existing_scopes: dict[str, dict[str, Any]] = {
        "pooled": existing_selected,
        "truthfulqa": find_dataset_scope(existing_selected, dataset="truthfulqa")[1],
        "triviaqa_bridge": find_dataset_scope(
            existing_selected, dataset="triviaqa_bridge"
        )[1],
    }
    checks: list[dict[str, Any]] = []
    for scope_name in ("pooled", "truthfulqa", "triviaqa_bridge"):
        generated_scope = generated_scopes[scope_name]
        existing_scope = existing_scopes[scope_name]
        checks.append(
            compare_metric_result(
                observed=generated_scope["rates"]["8.0"]["adjudicated_open_correct"],
                expected=existing_scope["rates"]["8.0"]["adjudicated_open_correct"],
                estimate_key="estimate",
                ci_key="ci",
                tolerance=tolerance,
                cell_name=f"selected/{scope_name}/alpha8/rate",
            )
        )
        checks.append(
            compare_metric_result(
                observed=generated_scope["paired_deltas_vs_baseline"]["8.0"][
                    "adjudicated_open_correct"
                ],
                expected=existing_scope["paired_deltas_vs_baseline"]["8.0"][
                    "adjudicated_open_correct"
                ],
                estimate_key="estimate_pp",
                ci_key="ci_pp",
                tolerance=tolerance,
                cell_name=f"selected/{scope_name}/alpha8/delta",
            )
        )
    return checks


def selected_alpha8_cell_table(
    conditions: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    selected = conditions["selected"]
    scopes: dict[str, dict[str, Any]] = {
        "pooled": selected,
        "truthfulqa": find_dataset_scope(selected, dataset="truthfulqa")[1],
        "triviaqa_bridge": find_dataset_scope(selected, dataset="triviaqa_bridge")[1],
    }
    cells: dict[str, dict[str, Any]] = {}
    for scope_name, scope in scopes.items():
        primary_rate = scope["rates"]["8.0"]["adjudicated_open_correct"]
        diagnostic_rate = scope["rates"]["8.0"]["diagnostic_exact_open_correct"]
        primary_delta = scope["paired_deltas_vs_baseline"]["8.0"][
            "adjudicated_open_correct"
        ]
        diagnostic_delta = scope["paired_deltas_vs_baseline"]["8.0"][
            "diagnostic_exact_open_correct"
        ]
        cells[scope_name] = {
            "primary_rate": primary_rate,
            "diagnostic_exact_rate": diagnostic_rate,
            "rate_shift_pp": (
                metric_number(diagnostic_rate, "estimate")
                - metric_number(primary_rate, "estimate")
            )
            * 100.0,
            "primary_delta": primary_delta,
            "diagnostic_exact_delta": diagnostic_delta,
            "delta_shift_pp": metric_number(
                diagnostic_delta,
                "estimate_pp",
            )
            - metric_number(primary_delta, "estimate_pp"),
        }
    return cells


def propagation_validation(
    *,
    corrections: list[dict[str, Any]],
    rules: list[EvidenceRule],
) -> dict[str, Any]:
    expected_exact_counts = {
        rule.case_family: rule.mvp_exact_pattern_count
        for rule in rules
        if rule.propagatable
    }
    exact_counts: Counter[str] = Counter()
    changed_counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    label_transition_counts: Counter[str] = Counter()
    blocked_or_unresolved_changed: list[dict[str, Any]] = []
    for row in corrections:
        status = str(row["propagation_status"])
        status_counts[status] += 1
        family = row.get("evidence_family")
        if status in EXACT_APPLIED_STATUSES:
            if not isinstance(family, str):
                raise ValueError("Exact applied row missing evidence_family")
            exact_counts[family] += 1
        elif row.get("label_changed"):
            blocked_or_unresolved_changed.append(row_identity(row))
        if row.get("label_changed"):
            if isinstance(family, str):
                changed_counts[family] += 1
            label_transition_counts[
                f"{row['original_open_label']}->{row['corrected_diagnostic_open_label']}"
            ] += 1

    mismatched_families = {
        family: {
            "expected": expected,
            "observed": exact_counts.get(family, 0),
        }
        for family, expected in expected_exact_counts.items()
        if exact_counts.get(family, 0) != expected
    }
    missing_families = sorted(set(expected_exact_counts) - set(exact_counts))
    extra_families = sorted(set(exact_counts) - set(expected_exact_counts))
    if blocked_or_unresolved_changed:
        raise ValueError("Blocked/unresolved rows changed labels")
    if mismatched_families or missing_families or extra_families:
        raise ValueError(
            "Exact propagation counts do not match correction evidence: "
            f"mismatched={mismatched_families}, missing={missing_families}, "
            f"extra={extra_families}"
        )

    core_rules = [rule for rule in rules if is_core_i_question_rule(rule)]
    core_boundary_separated = (
        len({rule.answer_normalized for rule in core_rules}) == len(core_rules)
        and len({rule.fresh_label for rule in core_rules}) > 1
    )
    if core_rules and not core_boundary_separated:
        raise ValueError("Core-i boundary evidence is not separated")

    return {
        "schema_version": "simid_diagnostic_exact_propagation_validation/v1",
        "all_evidence_families_represented": set(exact_counts)
        == set(expected_exact_counts),
        "exact_counts_match_evidence_mvp_exact_pattern_count": True,
        "no_blocked_or_unresolved_row_changed_label": True,
        "core_i9_and_core_i9_apple_silicon_separated": core_boundary_separated,
        "n_rows": len(corrections),
        "n_exact_applied_rows": sum(exact_counts.values()),
        "n_label_changed_rows": sum(changed_counts.values()),
        "status_counts": dict(sorted(status_counts.items())),
        "exact_applied_counts_by_family": dict(sorted(exact_counts.items())),
        "expected_exact_counts_by_family": dict(sorted(expected_exact_counts.items())),
        "label_changed_counts_by_family": dict(sorted(changed_counts.items())),
        "label_transition_counts": dict(sorted(label_transition_counts.items())),
        "blocked_core_i_family_rows": status_counts.get("blocked_core_i_family", 0),
        "blocked_same_question_nonexact_rows": status_counts.get(
            "blocked_same_question_nonexact", 0
        ),
        "nonpropagatable_exact_rows": status_counts.get(
            "evidence_nonpropagatable_not_applied", 0
        ),
    }


def build_conditions_summary(
    *,
    rows: list[dict[str, Any]],
    conditions: list[str] | None,
    alphas: list[float] | None,
    baseline_alpha: float,
    n_resamples: int,
    seed: int,
) -> tuple[dict[str, Any], list[str], list[float]]:
    indexed = index_rows(rows)
    selected_conditions = conditions or sorted(indexed)
    selected_alphas = alphas or sorted(
        {alpha for condition in selected_conditions for alpha in indexed[condition]}
    )
    if baseline_alpha not in selected_alphas:
        raise ValueError(
            f"baseline alpha {baseline_alpha} is not in alpha grid {selected_alphas}"
        )
    condition_summaries = {
        condition: summarize_condition_for_open_metrics(
            indexed,
            condition=condition,
            alphas=selected_alphas,
            baseline_alpha=baseline_alpha,
            n_resamples=n_resamples,
            seed=seed,
        )
        for condition in selected_conditions
    }
    return condition_summaries, selected_conditions, selected_alphas


def format_rate(result: dict[str, Any]) -> str:
    ci = result["ci"]
    return (
        f"{float(result['estimate']) * 100.0:.1f}% "
        f"[{float(ci['lower']) * 100.0:.1f}, {float(ci['upper']) * 100.0:.1f}]"
    )


def format_delta(result: dict[str, Any]) -> str:
    ci = result["ci_pp"]
    return (
        f"{float(result['estimate_pp']):+.1f} pp "
        f"[{float(ci['lower']):+.1f}, {float(ci['upper']):+.1f}]"
    )


def build_markdown(summary: dict[str, Any]) -> str:
    validation = summary["validation"]
    cells = summary["selected_alpha8_cells"]
    lines = [
        "# SIMID Diagnostic Exact-Propagation Sensitivity",
        "",
        "This is a diagnostic measurement-cleanup sensitivity analysis. It applies "
        "fresh labels only when a full MVP row has the same question and the same "
        "normalized model response as an eligible correction-evidence row. It does "
        "not mutate the historical SIMID outputs and does not make open "
        "correctness claim-bearing.",
        "",
        "## Coverage",
        "",
        f"- MVP rows evaluated: {validation['n_rows']}",
        f"- Exact evidence rows applied: {validation['n_exact_applied_rows']}",
        f"- Rows whose diagnostic label changed: {validation['n_label_changed_rows']}",
        f"- Broader Core-i* rows blocked: {validation['blocked_core_i_family_rows']}",
        f"- Same-question nonexact rows blocked: {validation['blocked_same_question_nonexact_rows']}",
        "",
        "## Family Counts",
        "",
        "| Evidence family | Exact rows | Changed rows |",
        "|---|---:|---:|",
    ]
    exact_counts = validation["exact_applied_counts_by_family"]
    changed_counts = validation["label_changed_counts_by_family"]
    for family in sorted(exact_counts):
        lines.append(
            f"| `{family}` | {exact_counts[family]} | {changed_counts.get(family, 0)} |"
        )
    lines.extend(
        [
            "",
            "## Selected Alpha=8 Open Correctness",
            "",
            "| Scope | Primary rate | Diagnostic exact rate | Rate shift | Primary delta | Diagnostic exact delta | Delta shift |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    labels = {
        "pooled": "Pooled",
        "truthfulqa": "TruthfulQA",
        "triviaqa_bridge": "Bridge",
    }
    for key in ("pooled", "truthfulqa", "triviaqa_bridge"):
        cell = cells[key]
        lines.append(
            "| "
            f"{labels[key]} | "
            f"{format_rate(cell['primary_rate'])} | "
            f"{format_rate(cell['diagnostic_exact_rate'])} | "
            f"{float(cell['rate_shift_pp']):+.1f} pp | "
            f"{format_delta(cell['primary_delta'])} | "
            f"{format_delta(cell['diagnostic_exact_delta'])} | "
            f"{float(cell['delta_shift_pp']):+.1f} pp |"
        )
    lines.extend(
        [
            "",
            "## Validation",
            "",
            "- Uncorrected recomputation reproduces the selected pooled, TruthfulQA, "
            "and Bridge alpha=8 rate and delta cells from `results_adjudicated.json`.",
            "- Exact propagation counts equal the evidence table's "
            "`mvp_exact_pattern_count` by family.",
            "- Blocked and nonpropagatable rows leave labels unchanged.",
            "- Core i9 and Core i9 Apple Silicon remain separate exact-answer rules.",
            "",
        ]
    )
    return "\n".join(lines)


def build_readme() -> str:
    return """# Diagnostic Exact Propagation 2026-04-29

Append-only SIMID measurement-cleanup sensitivity outputs. These artifacts apply
fresh labels only to full MVP rows with the same question and the same normalized
model response as an eligible duplicate-collapsed correction-evidence row.

Files:

- `row_corrections.jsonl` - one row per 4,800-row MVP open response, including
  original label, diagnostic exact-propagation label, evidence family, status,
  and notes.
- `diagnostic_sensitivity_summary.json` - paired base-item open-correctness
  rates and deltas under primary and diagnostic exact-propagation labels, plus
  coverage and validation checks.
- `diagnostic_sensitivity.md` - compact human-readable sensitivity report.
- `diagnostic_exact_propagation.provenance.*.json` - command, inputs, outputs,
  active-run guard metadata, and hashes.

Regenerate from the repository root with:

```bash
uv run python scripts/build_simid_boundary_propagation_sensitivity.py --overwrite
```
"""


def guard_outputs(paths: Iterable[Path], *, overwrite: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        formatted = "\n".join(f"- {path}" for path in existing)
        raise FileExistsError(
            "Diagnostic propagation outputs already exist; pass --overwrite:\n"
            f"{formatted}"
        )


def output_hashes(paths: Iterable[Path]) -> dict[str, dict[str, str]]:
    return {stable_path(path): {"sha256": file_sha256(path)} for path in paths}


def input_hashes(paths: Iterable[Path]) -> dict[str, dict[str, str]]:
    return {stable_path(path): {"sha256": file_sha256(path)} for path in paths}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_dir = args.run_dir.resolve()
    evidence_jsonl = args.evidence_jsonl.resolve()
    output_dir = args.output_dir.resolve()
    row_corrections_path = output_dir / "row_corrections.jsonl"
    summary_path = output_dir / "diagnostic_sensitivity_summary.json"
    markdown_path = output_dir / "diagnostic_sensitivity.md"
    readme_path = output_dir / "README.md"
    output_paths = [row_corrections_path, summary_path, markdown_path, readme_path]

    guard_outputs(output_paths, overwrite=bool(args.overwrite))
    output_dir.mkdir(parents=True, exist_ok=True)
    provenance = start_run_provenance(
        args,
        primary_target=output_dir / "diagnostic_exact_propagation",
        output_targets=output_paths,
    )
    status = "completed"
    extra: dict[str, Any] = {}
    try:
        validate_complete_run_outputs(run_dir)
        rows = load_run_rows(run_dir)
        open_adjudication_path = run_dir / "open_adjudication.jsonl"
        adjudications = load_open_adjudications(open_adjudication_path)
        n_attached = attach_open_adjudications(rows, adjudications)
        if n_attached != len(rows):
            raise ValueError(
                f"Loaded {n_attached} adjudications for {len(rows)} SIMID rows"
            )

        rules = load_evidence_rules(evidence_jsonl)
        corrections = build_row_corrections(rows, rules)
        validation = propagation_validation(corrections=corrections, rules=rules)

        conditions, selected_conditions, selected_alphas = build_conditions_summary(
            rows=rows,
            conditions=args.conditions,
            alphas=args.alphas,
            baseline_alpha=float(args.baseline_alpha),
            n_resamples=int(args.n_resamples),
            seed=int(args.seed),
        )
        existing_results_path = run_dir / "results_adjudicated.json"
        existing_results = load_json(existing_results_path)
        reproduction_checks = selected_alpha8_reproduction_checks(
            generated_conditions=conditions,
            existing_results=existing_results,
            tolerance=float(args.tolerance),
        )
        summary = {
            "schema_version": SUMMARY_SCHEMA_VERSION,
            "generated_at_utc": datetime.now(UTC).isoformat(),
            "run_dir": stable_path(run_dir),
            "evidence_jsonl": stable_path(evidence_jsonl),
            "open_adjudication_jsonl": stable_path(open_adjudication_path),
            "results_adjudicated_json": stable_path(existing_results_path),
            "matching_rule": {
                "scope": "same question plus same normalized model response only",
                "normalization": NORMALIZATION_DESCRIPTION,
                "broader_family_propagation": "blocked_without_new_review",
            },
            "claimability_guardrail": (
                "Diagnostic sensitivity only. Historical MVP outputs are preserved, "
                "and open correctness remains non-claim-bearing until a separate "
                "prospective calibration gate passes."
            ),
            "bootstrap": {
                "n_resamples": int(args.n_resamples),
                "seed": int(args.seed),
                "confidence": 0.95,
                "resampling": "paired_by_base_sample_id",
            },
            "baseline_alpha": float(args.baseline_alpha),
            "alphas": selected_alphas,
            "conditions_included": selected_conditions,
            "validation": validation,
            "uncorrected_reproduction_checks": reproduction_checks,
            "selected_alpha8_cells": selected_alpha8_cell_table(conditions),
            "conditions": conditions,
        }

        write_jsonl(row_corrections_path, corrections)
        write_json(summary_path, summary)
        markdown_path.write_text(build_markdown(summary), encoding="utf-8")
        readme_path.write_text(build_readme(), encoding="utf-8")

        input_paths = [
            Path(__file__).resolve(),
            evidence_jsonl,
            open_adjudication_path,
            existing_results_path,
            run_dir / "manifest.locked.json",
            run_dir / "run_config.json",
        ]
        extra = {
            "n_rows": len(rows),
            "n_open_adjudications_loaded": n_attached,
            "n_evidence_rows": len(rules),
            "n_exact_applied_rows": validation["n_exact_applied_rows"],
            "n_label_changed_rows": validation["n_label_changed_rows"],
            "inputs": input_hashes(input_paths),
            "outputs": output_hashes(output_paths),
        }
        print(
            f"Wrote {len(corrections)} row corrections to {stable_path(row_corrections_path)}"
        )
        print(f"Diagnostic summary: {stable_path(summary_path)}")
    except BaseException as exc:
        status = provenance_status_for_exception(exc)
        extra["error"] = provenance_error_message(exc)
        raise
    finally:
        finish_run_provenance(provenance, status, extra)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
