#!/usr/bin/env python3
"""Summarize available jailbreak evaluator comparisons from local artifacts only."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from uncertainty import paired_bootstrap_curve_effects
from utils import format_alpha_label

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = (
    ROOT / "data/judge_validation/available_jailbreak_evaluator_comparison.json"
)

CONTAMINATED_IDS = frozenset(
    {
        "jbb_harmful_62_t2",
        "jbb_harmful_97_t0",
        "jbb_harmful_14_t3",
        "jbb_harmful_91_t4",
        "jbb_harmful_36_t4",
        "jbb_harmful_34_t0",
        "jbb_harmful_19_t0",
        "jbb_harmful_3_t2",
    }
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def repo_relative(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def rate_summary(yes: int, total: int) -> dict[str, Any]:
    return {
        "yes": yes,
        "total": total,
        "rate": 0.0 if total == 0 else yes / total,
    }


def confusion(records: list[dict[str, Any]], field: str) -> dict[str, int]:
    tp = tn = fp = fn = 0
    for record in records:
        pred = record[field]
        gold = record["gold"]
        if gold and pred:
            tp += 1
        elif gold and not pred:
            fn += 1
        elif not gold and pred:
            fp += 1
        else:
            tn += 1
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def metrics_from_confusion(c: dict[str, int]) -> dict[str, float]:
    total = c["tp"] + c["tn"] + c["fp"] + c["fn"]
    precision = c["tp"] / (c["tp"] + c["fp"]) if (c["tp"] + c["fp"]) else 0.0
    recall = c["tp"] / (c["tp"] + c["fn"]) if (c["tp"] + c["fn"]) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "accuracy": 0.0 if total == 0 else (c["tp"] + c["tn"]) / total,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def summarize_binary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    harmful = sum(row.get("judge") == "HARMFUL" for row in rows)
    return rate_summary(harmful, len(rows))


def summarize_v2(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid_rows = [row for row in rows if "harmful_binary" in row.get("csv2", {})]
    yes = sum(row["csv2"]["harmful_binary"] == "yes" for row in valid_rows)
    borderline = sum(
        row["csv2"]["harmful_binary"] == "borderline" for row in valid_rows
    )
    summary = rate_summary(yes, len(valid_rows))
    summary["valid"] = len(valid_rows)
    summary["errors"] = len(rows) - len(valid_rows)
    summary["borderline"] = borderline
    summary["borderline_rate"] = 0.0 if not valid_rows else borderline / len(valid_rows)
    return summary


def summarize_v3(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid_rows = [row for row in rows if not row.get("csv2", {}).get("error")]
    yes = sum(row["csv2"]["harmful_binary"] == "yes" for row in valid_rows)
    substantive = sum(
        row["csv2"].get("primary_outcome") == "substantive_compliance"
        for row in valid_rows
    )
    summary = rate_summary(yes, len(valid_rows))
    summary["valid"] = len(valid_rows)
    summary["errors"] = len(rows) - len(valid_rows)
    summary["substantive_compliance"] = {
        "yes": substantive,
        "total": len(valid_rows),
        "rate": 0.0 if not valid_rows else substantive / len(valid_rows),
    }
    return summary


def _mistral_key(
    row: dict[str, Any], alpha: float, source_name: str
) -> tuple[str, str]:
    record_id = row.get("id")
    if not isinstance(record_id, str) or not record_id:
        raise ValueError(f"{source_name}: row missing non-empty string id")

    expected_alpha = format_alpha_label(alpha)
    row_alpha = format_alpha_label(float(row.get("alpha", alpha)))
    if row_alpha != expected_alpha:
        raise ValueError(
            f"{source_name}: row id={record_id!r} has alpha {row_alpha}, "
            f"expected {expected_alpha}"
        )
    return record_id, expected_alpha


def _load_mistral_keyed_rows(
    directory: Path, alphas: list[float], source_name: str
) -> dict[tuple[str, str], dict[str, Any]]:
    keyed: dict[tuple[str, str], dict[str, Any]] = {}
    for alpha in alphas:
        alpha_label = format_alpha_label(alpha)
        path = directory / f"alpha_{alpha_label}.jsonl"
        rows = load_jsonl(path)
        for row in rows:
            key = _mistral_key(row, alpha, source_name)
            if key in keyed:
                raise ValueError(f"{source_name}: duplicate key {key!r}")
            keyed[key] = row
    return keyed


def _require_matching_mistral_keys(
    expected: set[tuple[str, str]],
    actual: set[tuple[str, str]],
    source_name: str,
) -> None:
    if expected == actual:
        return
    missing = sorted(expected - actual)[:5]
    extra = sorted(actual - expected)[:5]
    raise ValueError(
        f"{source_name}: keys do not match binary_full "
        f"(missing={missing}, extra={extra})"
    )


def _binary_label(row: dict[str, Any]) -> bool | None:
    judge = row.get("judge")
    if judge == "HARMFUL":
        return True
    if judge == "SAFE":
        return False
    return None


def _csv3_label(row: dict[str, Any]) -> bool | None:
    payload = row.get("csv2")
    if not isinstance(payload, dict) or payload.get("error"):
        return None
    harmful_binary = payload.get("harmful_binary")
    if harmful_binary == "yes":
        return True
    if harmful_binary == "no":
        return False
    return None


def _strongreject_label(row: dict[str, Any]) -> bool | None:
    payload = row.get("strongreject")
    if not isinstance(payload, dict) or payload.get("error"):
        return None
    harmful_binary = payload.get("harmful_binary")
    if harmful_binary == "yes":
        return True
    if harmful_binary == "no":
        return False
    return None


def _safe_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def _mean(values: list[float]) -> float | None:
    return None if not values else sum(values) / len(values)


def _rankdata_average(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i + 1
        while j < len(indexed) and indexed[j][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[indexed[k][0]] = avg_rank
        i = j
    return ranks


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    dx = [x - mean_x for x in xs]
    dy = [y - mean_y for y in ys]
    denom_x = sum(x * x for x in dx)
    denom_y = sum(y * y for y in dy)
    if denom_x == 0.0 or denom_y == 0.0:
        return None
    return sum(x * y for x, y in zip(dx, dy, strict=True)) / math.sqrt(
        denom_x * denom_y
    )


def _spearman(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    return _pearson(_rankdata_average(xs), _rankdata_average(ys))


def _numeric_association(
    rows: list[dict[str, Any]], left_field: str, right_field: str
) -> dict[str, Any]:
    pairs: list[tuple[float, float]] = []
    for row in rows:
        left = _safe_number(row.get(left_field))
        right = _safe_number(row.get(right_field))
        if left is not None and right is not None:
            pairs.append((left, right))
    xs = [left for left, _ in pairs]
    ys = [right for _, right in pairs]
    return {
        "n": len(pairs),
        "pearson": _pearson(xs, ys),
        "spearman": _spearman(xs, ys),
    }


def _cohen_kappa_from_counts(
    left_yes_right_yes: int,
    left_yes_right_no: int,
    left_no_right_yes: int,
    left_no_right_no: int,
) -> float | None:
    n = left_yes_right_yes + left_yes_right_no + left_no_right_yes + left_no_right_no
    if n == 0:
        return None
    observed = (left_yes_right_yes + left_no_right_no) / n
    left_yes = (left_yes_right_yes + left_yes_right_no) / n
    right_yes = (left_yes_right_yes + left_no_right_yes) / n
    expected = left_yes * right_yes + (1.0 - left_yes) * (1.0 - right_yes)
    if expected == 1.0:
        return 1.0 if observed == 1.0 else None
    return (observed - expected) / (1.0 - expected)


def _mcnemar_exact_two_sided(left_yes_right_no: int, left_no_right_yes: int) -> float:
    n = left_yes_right_no + left_no_right_yes
    if n == 0:
        return 1.0
    smaller = min(left_yes_right_no, left_no_right_yes)
    tail = sum(math.comb(n, k) for k in range(smaller + 1)) * (0.5**n)
    return min(1.0, 2.0 * tail)


def _summarize_mistral_evaluator(
    joined: list[dict[str, Any]], field: str
) -> dict[str, Any]:
    labels = [row[field] for row in joined]
    valid = [label for label in labels if label is not None]
    harmful = sum(label is True for label in valid)
    summary = rate_summary(harmful, len(valid))
    summary["valid"] = len(valid)
    summary["errors"] = len(labels) - len(valid)
    return summary


def _summarize_mistral_pair(
    joined: list[dict[str, Any]], left_field: str, right_field: str
) -> dict[str, Any]:
    left_valid = [row for row in joined if row[left_field] is not None]
    right_valid = [row for row in joined if row[right_field] is not None]
    paired = [
        row
        for row in joined
        if row[left_field] is not None and row[right_field] is not None
    ]
    left_yes_right_yes = sum(
        row[left_field] is True and row[right_field] is True for row in paired
    )
    left_yes_right_no = sum(
        row[left_field] is True and row[right_field] is False for row in paired
    )
    left_no_right_yes = sum(
        row[left_field] is False and row[right_field] is True for row in paired
    )
    left_no_right_no = sum(
        row[left_field] is False and row[right_field] is False for row in paired
    )
    agreements = left_yes_right_yes + left_no_right_no
    disagreements = left_yes_right_no + left_no_right_yes
    left_harmful = sum(row[left_field] is True for row in left_valid)
    right_harmful = sum(row[right_field] is True for row in right_valid)
    left_rate = 0.0 if not left_valid else left_harmful / len(left_valid)
    right_rate = 0.0 if not right_valid else right_harmful / len(right_valid)
    return {
        "n_records": len(joined),
        "n_paired_evaluable": len(paired),
        "left_valid": len(left_valid),
        "right_valid": len(right_valid),
        "left_errors": len(joined) - len(left_valid),
        "right_errors": len(joined) - len(right_valid),
        "agreement": rate_summary(agreements, len(paired)),
        "disagreement": rate_summary(disagreements, len(paired)),
        "left_harmful": rate_summary(left_harmful, len(left_valid)),
        "right_harmful": rate_summary(right_harmful, len(right_valid)),
        "left_minus_right_harmful_rate_pp": (left_rate - right_rate) * 100.0,
        "cohen_kappa": _cohen_kappa_from_counts(
            left_yes_right_yes,
            left_yes_right_no,
            left_no_right_yes,
            left_no_right_no,
        ),
        "mcnemar_exact_two_sided_p": _mcnemar_exact_two_sided(
            left_yes_right_no,
            left_no_right_yes,
        ),
        "left_yes_right_yes": left_yes_right_yes,
        "left_yes_right_no": left_yes_right_no,
        "left_no_right_yes": left_no_right_yes,
        "left_no_right_no": left_no_right_no,
    }


def _summarize_mistral_pair_by_alpha(
    joined: list[dict[str, Any]], left_field: str, right_field: str
) -> dict[str, Any]:
    by_alpha: dict[str, list[dict[str, Any]]] = {}
    for row in joined:
        by_alpha.setdefault(row["alpha_label"], []).append(row)
    return {
        "combined": _summarize_mistral_pair(joined, left_field, right_field),
        "per_alpha": {
            alpha: _summarize_mistral_pair(rows, left_field, right_field)
            for alpha, rows in sorted(by_alpha.items())
        },
    }


def _summarize_csv3_diagnostics(joined: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in joined if row["csv3_full"] is not None]
    harmful = [row for row in valid if row["csv3_full"] is True]
    outcomes: dict[str, int] = {}
    for row in valid:
        outcome = row.get("csv3_primary_outcome")
        if isinstance(outcome, str):
            outcomes[outcome] = outcomes.get(outcome, 0) + 1
    c_values = [
        value for row in valid if (value := _safe_number(row.get("csv3_C"))) is not None
    ]
    s_values = [
        value for row in valid if (value := _safe_number(row.get("csv3_S"))) is not None
    ]
    v_values = [
        value for row in valid if (value := _safe_number(row.get("csv3_V"))) is not None
    ]
    high_specificity = sum(
        bool((value := _safe_number(row.get("csv3_S"))) is not None and value >= 3)
        for row in valid
    )
    return {
        "n_valid": len(valid),
        "harmful": rate_summary(len(harmful), len(valid)),
        "primary_outcome_counts": dict(sorted(outcomes.items())),
        "substantive_compliance": rate_summary(
            outcomes.get("substantive_compliance", 0), len(valid)
        ),
        "partial_or_substantive_compliance": rate_summary(
            outcomes.get("partial_compliance", 0)
            + outcomes.get("substantive_compliance", 0),
            len(valid),
        ),
        "high_specificity_s_ge_3": rate_summary(high_specificity, len(valid)),
        "mean_C": _mean(c_values),
        "mean_S": _mean(s_values),
        "mean_V_when_present": _mean(v_values),
    }


def _summarize_strongreject_diagnostics(joined: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in joined if row["strongreject_full"] is not None]
    harmful = [row for row in valid if row["strongreject_full"] is True]
    refused = sum(row.get("strongreject_refused") == 1 for row in valid)
    scores = [
        value
        for row in valid
        if (value := _safe_number(row.get("strongreject_score"))) is not None
    ]
    convincingness = [
        value
        for row in valid
        if (value := _safe_number(row.get("strongreject_convincingness"))) is not None
    ]
    specificity = [
        value
        for row in valid
        if (value := _safe_number(row.get("strongreject_specificity"))) is not None
    ]
    return {
        "n_valid": len(valid),
        "harmful": rate_summary(len(harmful), len(valid)),
        "refused": rate_summary(refused, len(valid)),
        "mean_score": _mean(scores),
        "mean_convincingness": _mean(convincingness),
        "mean_specificity": _mean(specificity),
    }


def _summarize_csv3_strongreject_association(
    joined: list[dict[str, Any]],
) -> dict[str, Any]:
    paired = [
        row
        for row in joined
        if row["csv3_full"] is not None and row["strongreject_full"] is not None
    ]
    score_by_outcome: dict[str, list[float]] = {}
    for row in paired:
        outcome = row.get("csv3_primary_outcome")
        score = _safe_number(row.get("strongreject_score"))
        if isinstance(outcome, str) and score is not None:
            score_by_outcome.setdefault(outcome, []).append(score)
    return {
        "n_paired": len(paired),
        "associations_with_strongreject_score": {
            "csv3_harmful_binary": _numeric_association(
                paired, "csv3_harmful_numeric", "strongreject_score"
            ),
            "csv3_C": _numeric_association(paired, "csv3_C", "strongreject_score"),
            "csv3_S": _numeric_association(paired, "csv3_S", "strongreject_score"),
            "csv3_V": _numeric_association(paired, "csv3_V", "strongreject_score"),
            "csv3_substantive_compliance": _numeric_association(
                paired, "csv3_substantive_numeric", "strongreject_score"
            ),
        },
        "mean_strongreject_score_by_csv3_outcome": {
            outcome: {"n": len(scores), "mean_score": _mean(scores)}
            for outcome, scores in sorted(score_by_outcome.items())
        },
    }


def _binary_curve_effects(
    joined: list[dict[str, Any]], field: str, alphas: list[float]
) -> dict[str, Any]:
    alpha_labels = [format_alpha_label(alpha) for alpha in alphas]
    by_id: dict[str, dict[str, dict[str, Any]]] = {}
    for row in joined:
        by_id.setdefault(str(row["id"]), {})[str(row["alpha_label"])] = row

    trajectories: list[list[bool]] = []
    dropped_incomplete = 0
    for rows_by_alpha in by_id.values():
        trajectory: list[bool] = []
        complete = True
        for alpha_label in alpha_labels:
            row = rows_by_alpha.get(alpha_label)
            label = None if row is None else row.get(field)
            if label is None:
                complete = False
                break
            trajectory.append(bool(label))
        if complete:
            trajectories.append(trajectory)
        else:
            dropped_incomplete += 1

    if not trajectories:
        return {
            "n_complete_ids": 0,
            "dropped_incomplete_ids": dropped_incomplete,
            "effects": None,
        }

    if len(alphas) < 2:
        matrix = np.asarray(trajectories, dtype=bool)
        return {
            "n_complete_ids": len(trajectories),
            "dropped_incomplete_ids": dropped_incomplete,
            "effects": {"rates": matrix.mean(axis=0).tolist()},
        }

    effects = paired_bootstrap_curve_effects(
        np.asarray(trajectories, dtype=bool),
        np.asarray(alphas, dtype=float),
    )
    return {
        "n_complete_ids": len(trajectories),
        "dropped_incomplete_ids": dropped_incomplete,
        "effects": effects,
    }


def compute_mistral_anchor_summary(
    *,
    binary_full_dir: Path,
    binary_256_dir: Path,
    csv3_full_dir: Path,
    strongreject_full_dir: Path,
    alphas: list[float],
) -> dict[str, Any]:
    """Summarize Mistral anchor evaluator agreement from local artifacts."""
    binary_full = _load_mistral_keyed_rows(binary_full_dir, alphas, "binary_full")
    sources = {
        "binary_256": _load_mistral_keyed_rows(binary_256_dir, alphas, "binary_256"),
        "csv3_full": _load_mistral_keyed_rows(csv3_full_dir, alphas, "csv3_full"),
        "strongreject_full": _load_mistral_keyed_rows(
            strongreject_full_dir, alphas, "strongreject_full"
        ),
    }
    expected_keys = set(binary_full)
    for source_name, rows in sources.items():
        _require_matching_mistral_keys(expected_keys, set(rows), source_name)

    joined: list[dict[str, Any]] = []
    for record_id, alpha_label in sorted(
        expected_keys, key=lambda key: (float(key[1]), key[0])
    ):
        key = (record_id, alpha_label)
        csv3_row = sources["csv3_full"][key]
        strongreject_row = sources["strongreject_full"][key]
        csv3_payload = csv3_row.get("csv2", {})
        strongreject_payload = strongreject_row.get("strongreject", {})
        csv3_label = _csv3_label(csv3_row)
        strongreject_label = _strongreject_label(strongreject_row)
        joined.append(
            {
                "id": record_id,
                "alpha": float(alpha_label),
                "alpha_label": alpha_label,
                "binary_full": _binary_label(binary_full[key]),
                "binary_256": _binary_label(sources["binary_256"][key]),
                "csv3_full": csv3_label,
                "csv3_harmful_numeric": (
                    None if csv3_label is None else float(csv3_label)
                ),
                "csv3_primary_outcome": (
                    csv3_payload.get("primary_outcome")
                    if isinstance(csv3_payload, dict)
                    else None
                ),
                "csv3_substantive_numeric": (
                    1.0
                    if isinstance(csv3_payload, dict)
                    and csv3_payload.get("primary_outcome") == "substantive_compliance"
                    else 0.0
                    if csv3_label is not None
                    else None
                ),
                "csv3_C": (
                    csv3_payload.get("C") if isinstance(csv3_payload, dict) else None
                ),
                "csv3_S": (
                    csv3_payload.get("S") if isinstance(csv3_payload, dict) else None
                ),
                "csv3_V": (
                    csv3_payload.get("V") if isinstance(csv3_payload, dict) else None
                ),
                "strongreject_full": strongreject_label,
                "strongreject_score": (
                    strongreject_payload.get("score")
                    if isinstance(strongreject_payload, dict)
                    else None
                ),
                "strongreject_refused": (
                    strongreject_payload.get("refused")
                    if isinstance(strongreject_payload, dict)
                    else None
                ),
                "strongreject_convincingness": (
                    strongreject_payload.get("convincingness")
                    if isinstance(strongreject_payload, dict)
                    else None
                ),
                "strongreject_specificity": (
                    strongreject_payload.get("specificity")
                    if isinstance(strongreject_payload, dict)
                    else None
                ),
            }
        )

    joined_by_alpha: dict[str, list[dict[str, Any]]] = {}
    for row in joined:
        joined_by_alpha.setdefault(row["alpha_label"], []).append(row)

    return {
        "mode": "mistral_anchor_jailbreak_evaluator_summary",
        "n_records": len(joined),
        "alphas": [format_alpha_label(alpha) for alpha in alphas],
        "sources": {
            "binary_full": repo_relative(binary_full_dir),
            "binary_256": repo_relative(binary_256_dir),
            "csv3_full": repo_relative(csv3_full_dir),
            "strongreject_full": repo_relative(strongreject_full_dir),
        },
        "evaluators": {
            field: _summarize_mistral_evaluator(joined, field)
            for field in (
                "binary_full",
                "binary_256",
                "csv3_full",
                "strongreject_full",
            )
        },
        "curve_effects": {
            field: _binary_curve_effects(joined, field, alphas)
            for field in (
                "binary_full",
                "binary_256",
                "csv3_full",
                "strongreject_full",
            )
        },
        "diagnostics": {
            "csv3_full": {
                "combined": _summarize_csv3_diagnostics(joined),
                "per_alpha": {
                    alpha: _summarize_csv3_diagnostics(rows)
                    for alpha, rows in sorted(joined_by_alpha.items())
                },
            },
            "strongreject_full": {
                "combined": _summarize_strongreject_diagnostics(joined),
                "per_alpha": {
                    alpha: _summarize_strongreject_diagnostics(rows)
                    for alpha, rows in sorted(joined_by_alpha.items())
                },
            },
            "csv3_vs_strongreject_full": {
                "combined": _summarize_csv3_strongreject_association(joined),
                "per_alpha": {
                    alpha: _summarize_csv3_strongreject_association(rows)
                    for alpha, rows in sorted(joined_by_alpha.items())
                },
            },
        },
        "pairs": {
            "binary_256_vs_binary_full": _summarize_mistral_pair_by_alpha(
                joined, "binary_256", "binary_full"
            ),
            "binary_full_vs_csv3_full": _summarize_mistral_pair_by_alpha(
                joined, "binary_full", "csv3_full"
            ),
            "binary_full_vs_strongreject_full": _summarize_mistral_pair_by_alpha(
                joined, "binary_full", "strongreject_full"
            ),
            "csv3_full_vs_strongreject_full": _summarize_mistral_pair_by_alpha(
                joined, "csv3_full", "strongreject_full"
            ),
        },
    }


def v2_v3_transition(
    v2_rows: list[dict[str, Any]], v3_rows: list[dict[str, Any]]
) -> dict[str, int]:
    v3_by_id = {row["id"]: row for row in v3_rows}
    counts = {
        "yes_to_yes": 0,
        "yes_to_no": 0,
        "borderline_to_yes": 0,
        "borderline_to_no": 0,
        "no_to_yes": 0,
        "no_to_no": 0,
        "v2_errors": 0,
        "v3_errors": 0,
    }
    for row in v2_rows:
        row_v3 = v3_by_id[row["id"]]
        payload_v2 = row.get("csv2", {})
        payload_v3 = row_v3.get("csv2", {})
        if "harmful_binary" not in payload_v2:
            counts["v2_errors"] += 1
            continue
        if payload_v3.get("error"):
            counts["v3_errors"] += 1
            continue
        source = payload_v2["harmful_binary"]
        target = payload_v3["harmful_binary"]
        if source == "borderline":
            source = "borderline"
        key = f"{source}_to_{target}"
        counts[key] += 1
    return counts


def compute_gold_summary() -> dict[str, Any]:
    gold_rows = [
        row
        for row in load_jsonl(
            ROOT / "tests/gold_labels/jailbreak_cross_alpha_gold.jsonl"
        )
        if row["human_label"] != "BORDERLINE"
    ]
    gold_by_key = {(row["id"], row["alpha"]): row for row in gold_rows}

    v3_by_key = {}
    for alpha in ("0.0", "1.5", "3.0"):
        for row in load_jsonl(
            ROOT / "data/judge_validation/csv2_v3" / f"alpha_{alpha}.jsonl"
        ):
            v3_by_key[(row["id"], row["alpha"])] = row

    v2_by_key = {}
    for alpha in ("0.0", "1.5", "3.0"):
        for row in load_jsonl(
            ROOT
            / "data/gemma3_4b/intervention/jailbreak/csv2_evaluation"
            / f"alpha_{alpha}.jsonl"
        ):
            key = (row["id"], row["alpha"])
            if key in gold_by_key:
                v2_by_key[key] = row

    binary_by_key = {}
    for alpha in ("0.0", "1.5", "3.0"):
        for row in load_jsonl(
            ROOT
            / "data/gemma3_4b/intervention/jailbreak/experiment"
            / f"alpha_{alpha}.jsonl"
        ):
            key = (row["id"], row["alpha"])
            if key in gold_by_key:
                binary_by_key[key] = row

    sr_by_key = {}
    for row in load_jsonl(
        ROOT / "data/judge_validation/strongreject/strongreject_gold_results.jsonl"
    ):
        sr_by_key[(row["id"], row["alpha"])] = row

    joined: list[dict[str, Any]] = []
    for key, gold in sorted(gold_by_key.items()):
        joined.append(
            {
                "id": gold["id"],
                "alpha": gold["alpha"],
                "gold": gold["human_label"] == "HARMFUL",
                "split": "dev" if gold["id"] in CONTAMINATED_IDS else "holdout",
                "v3": v3_by_key[key]["csv2"]["harmful_binary"] == "yes",
                "v2": v2_by_key[key]["csv2"]["harmful_binary"] == "yes",
                "binary": binary_by_key[key].get("judge") == "HARMFUL",
                "sr4o": sr_by_key[key]["strongreject"]["harmful_binary"] == "yes",
            }
        )

    splits = {
        "combined": joined,
        "dev": [row for row in joined if row["split"] == "dev"],
        "holdout": [row for row in joined if row["split"] == "holdout"],
    }
    out: dict[str, Any] = {"n_total": len(joined), "splits": {}}
    for split_name, records in splits.items():
        split_summary: dict[str, Any] = {
            "n_records": len(records),
            "n_harmful": sum(record["gold"] for record in records),
            "n_safe": sum(not record["gold"] for record in records),
            "evaluators": {},
        }
        for field in ("v3", "sr4o", "v2", "binary"):
            c = confusion(records, field)
            split_summary["evaluators"][field] = {**c, **metrics_from_confusion(c)}
        out["splits"][split_name] = split_summary

    holdout = splits["holdout"]
    discordant = sum(row["v3"] != row["sr4o"] for row in holdout)
    out["holdout_v3_vs_sr4o"] = {"discordant_records": discordant}
    return out


def compute_h_sweep_summary() -> dict[str, Any]:
    base = ROOT / "data/gemma3_4b/intervention/jailbreak"
    out: dict[str, Any] = {"alphas": {}}
    for alpha in ("0.0", "1.0", "1.5", "3.0"):
        binary_rows = load_jsonl(base / "experiment" / f"alpha_{alpha}.jsonl")
        v2_rows = load_jsonl(base / "csv2_evaluation" / f"alpha_{alpha}.jsonl")
        v3_rows = load_jsonl(base / "csv2_v3_evaluation" / f"alpha_{alpha}.jsonl")
        out["alphas"][alpha] = {
            "binary": summarize_binary(binary_rows),
            "v2": summarize_v2(v2_rows),
            "v3": summarize_v3(v3_rows),
            "v2_to_v3_transition": v2_v3_transition(v2_rows, v3_rows),
        }
    return out


def compute_control_summary() -> dict[str, Any]:
    seed0 = (
        ROOT
        / "data/gemma3_4b/intervention/jailbreak/control/seed_0_unconstrained/csv2_evaluation"
    )
    seed1 = (
        ROOT
        / "data/gemma3_4b/intervention/jailbreak/control/seed_1_unconstrained_csv2_v3"
    )
    out: dict[str, Any] = {"seed0_v2": {}, "seed1_v3": {}}
    for alpha in ("0.0", "1.0", "1.5", "3.0"):
        out["seed0_v2"][alpha] = summarize_v2(
            load_jsonl(seed0 / f"alpha_{alpha}.jsonl")
        )
        out["seed1_v3"][alpha] = summarize_v3(
            load_jsonl(seed1 / f"alpha_{alpha}.jsonl")
        )
    return out


def compute_d7_summary() -> dict[str, Any]:
    root = ROOT / "data/gemma3_4b/intervention/jailbreak_d7/full500_canonical"
    surfaces = {
        "baseline_noop": ("1.0", True),
        "l1_neuron": ("3.0", True),
        "causal_locked": ("4.0", True),
        "probe_locked": ("1.0", False),
        "causal_random_head_layer_matched/seed_1": ("4.0", False),
        "causal_random_head_layer_matched/seed_2": ("4.0", False),
    }
    out: dict[str, Any] = {}
    for surface, (alpha, has_v3) in surfaces.items():
        surface_root = root / surface
        binary_rows = load_jsonl(surface_root / "experiment" / f"alpha_{alpha}.jsonl")
        v2_rows = load_jsonl(surface_root / "csv2_evaluation" / f"alpha_{alpha}.jsonl")
        summary: dict[str, Any] = {
            "alpha": float(alpha),
            "binary": summarize_binary(binary_rows),
            "v2": summarize_v2(v2_rows),
            "has_v3": has_v3,
        }
        if has_v3:
            v3_rows = load_jsonl(
                surface_root / "csv2_v3_evaluation" / f"alpha_{alpha}.jsonl"
            )
            summary["v3"] = summarize_v3(v3_rows)
            summary["v2_to_v3_transition"] = v2_v3_transition(v2_rows, v3_rows)
        out[surface] = summary
    return out


def compute_coverage_matrix() -> dict[str, Any]:
    root = ROOT / "data/gemma3_4b/intervention"
    coverage = {
        "gold_subset": {
            "binary": True,
            "v2": True,
            "v3": True,
            "strongreject": True,
        },
        "h_neuron_sweep": {
            "binary": True,
            "v2": True,
            "v3": True,
            "strongreject": False,
        },
        "seed0_control": {
            "binary": True,
            "v2": True,
            "v3": False,
            "strongreject": False,
        },
        "seed1_control": {
            "binary": False,
            "v2": False,
            "v3": True,
            "strongreject": False,
        },
        "d7_baseline_l1_causal": {
            "binary": True,
            "v2": True,
            "v3": True,
            "strongreject": False,
        },
        "d7_probe_randoms": {
            "binary": True,
            "v2": True,
            "v3": False,
            "strongreject": False,
        },
        "available_paths_checked": {
            "jailbreak_root": repo_relative(root / "jailbreak"),
            "d7_root": repo_relative(root / "jailbreak_d7/full500_canonical"),
        },
    }
    return coverage


def compute_available_summary() -> dict[str, Any]:
    return {
        "coverage_matrix": compute_coverage_matrix(),
        "gold_comparison": compute_gold_summary(),
        "h_neuron_sweep": compute_h_sweep_summary(),
        "controls": compute_control_summary(),
        "d7": compute_d7_summary(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize available jailbreak evaluator comparisons."
    )
    parser.add_argument(
        "--mode",
        choices=["available", "mistral-anchor"],
        default="available",
    )
    parser.add_argument("--output-path", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--binary-full-dir", type=Path, default=None)
    parser.add_argument("--binary-256-dir", type=Path, default=None)
    parser.add_argument("--csv3-full-dir", type=Path, default=None)
    parser.add_argument("--strongreject-full-dir", type=Path, default=None)
    parser.add_argument("--alphas", type=float, nargs="+", default=None)
    return parser.parse_args()


def _required_path(value: Path | None, arg_name: str) -> Path:
    if value is None:
        raise ValueError(f"{arg_name} is required for --mode mistral-anchor")
    return value


def main() -> None:
    args = parse_args()
    if args.mode == "mistral-anchor":
        if not args.alphas:
            raise ValueError("--alphas is required for --mode mistral-anchor")
        summary = compute_mistral_anchor_summary(
            binary_full_dir=_required_path(args.binary_full_dir, "--binary-full-dir"),
            binary_256_dir=_required_path(args.binary_256_dir, "--binary-256-dir"),
            csv3_full_dir=_required_path(args.csv3_full_dir, "--csv3-full-dir"),
            strongreject_full_dir=_required_path(
                args.strongreject_full_dir, "--strongreject-full-dir"
            ),
            alphas=args.alphas,
        )
    else:
        summary = compute_available_summary()

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
