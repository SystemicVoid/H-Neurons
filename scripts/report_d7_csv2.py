#!/usr/bin/env python3
"""Paired D7 CSV-v2 report with uncertainty estimates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from evaluate_csv2 import normalize_csv2_payload
from uncertainty import (
    DEFAULT_BOOTSTRAP_RESAMPLES,
    DEFAULT_BOOTSTRAP_SEED,
    paired_bootstrap_binary_rate_difference,
    percentile_interval,
    wilson_interval,
)
from utils import format_alpha_label


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline_dir", type=Path, required=True)
    parser.add_argument("--baseline_alpha", type=float, default=0.0)
    parser.add_argument(
        "--condition",
        action="append",
        default=[],
        help="Condition spec name:/path/to/csv2_dir:alpha (repeatable).",
    )
    parser.add_argument("--output_path", type=Path, required=True)
    return parser.parse_args()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _alpha_file(root: Path, alpha: float) -> Path:
    return root / f"alpha_{format_alpha_label(alpha)}.jsonl"


def _require_csv2(rows: list[dict[str, Any]], *, context: str) -> None:
    missing = [str(row.get("id")) for row in rows if "csv2" not in row]
    if missing:
        raise ValueError(
            f"{context}: missing csv2 annotations for {len(missing)} rows "
            f"(sample={missing[:5]})"
        )


def _require_parity(
    baseline: dict[str, dict[str, Any]],
    candidate: dict[str, dict[str, Any]],
    *,
    context: str,
) -> list[str]:
    baseline_ids = set(baseline)
    candidate_ids = set(candidate)
    if baseline_ids != candidate_ids:
        missing_candidate = sorted(baseline_ids - candidate_ids)
        missing_baseline = sorted(candidate_ids - baseline_ids)
        raise ValueError(
            f"{context}: prompt-ID parity failed "
            f"(missing_in_candidate={len(missing_candidate)} sample={missing_candidate[:5]}, "
            f"missing_in_baseline={len(missing_baseline)} sample={missing_baseline[:5]})"
        )
    return sorted(baseline_ids)


def _rows_by_id(
    rows: list[dict[str, Any]], *, context: str
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        sample_id = str(row["id"])
        if sample_id in out:
            raise ValueError(f"{context}: duplicate sample id {sample_id!r}")
        out[sample_id] = row
    return out


def _condition_arrays(
    rows: dict[str, dict[str, Any]], ordered_ids: list[str]
) -> dict[str, np.ndarray]:
    csv2_yes: list[bool] = []
    substantive_compliance: list[bool] = []
    intent_match: list[float] = []
    c_values: list[float] = []
    s_values: list[float] = []
    v_values: list[float] = []
    payload_share: list[float] = []
    pivot_position: list[float] = []
    empty_response: list[bool] = []
    v_null_count = 0
    pivot_null_count = 0

    for sample_id in ordered_ids:
        rec = rows[sample_id]
        raw_csv2 = rec.get("csv2", {})
        if not isinstance(raw_csv2, dict):
            raise ValueError(f"Record {sample_id} has invalid csv2 payload")
        csv2 = normalize_csv2_payload(raw_csv2)
        if csv2.get("error"):
            raise ValueError(
                f"Record {sample_id} has csv2 error={csv2.get('error')!r}; "
                "repair csv2 annotations before D7 reporting"
            )

        csv2_yes.append(csv2.get("harmful_binary") == "yes")
        substantive_compliance.append(
            csv2.get("primary_outcome") == "substantive_compliance"
        )
        intent_match.append(float(csv2.get("intent_match", 0)))
        c_values.append(float(csv2.get("C", 0)))
        s_values.append(float(csv2.get("S", 0)))
        v_raw = csv2.get("V")
        if v_raw is None:
            v_null_count += 1
            v_values.append(0.0)
        else:
            v_values.append(float(v_raw))
        payload_share.append(float(csv2.get("harmful_payload_share", 0.0) or 0.0))
        pivot_raw = csv2.get("pivot_position")
        if pivot_raw is None:
            pivot_null_count += 1
            pivot_position.append(float("nan"))
        else:
            pivot_position.append(float(pivot_raw))
        response_text = str(rec.get("response", ""))
        empty_response.append(len(response_text.strip()) == 0)

    return {
        "csv2_yes": np.array(csv2_yes, dtype=bool),
        "substantive_compliance": np.array(substantive_compliance, dtype=bool),
        "intent_match": np.array(intent_match, dtype=np.float64),
        "C": np.array(c_values, dtype=np.float64),
        "S": np.array(s_values, dtype=np.float64),
        "V": np.array(v_values, dtype=np.float64),
        "harmful_payload_share": np.array(payload_share, dtype=np.float64),
        "pivot_position": np.array(pivot_position, dtype=np.float64),
        "empty_response": np.array(empty_response, dtype=bool),
        "v_null_count": np.array([v_null_count], dtype=np.int64),
        "pivot_null_count": np.array([pivot_null_count], dtype=np.int64),
    }


def _bootstrap_mean_summary(values: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError("Mean summary expects a one-dimensional array")
    n = len(arr)
    if n == 0:
        return {
            "estimate": 0.0,
            "ci": percentile_interval(
                np.array([0.0]),
                method="bootstrap_percentile_mean",
            ).to_dict(),
        }

    rng = np.random.default_rng(DEFAULT_BOOTSTRAP_SEED)
    samples = np.empty(DEFAULT_BOOTSTRAP_RESAMPLES, dtype=np.float64)
    for sample_idx in range(DEFAULT_BOOTSTRAP_RESAMPLES):
        indices = rng.choice(n, size=n, replace=True)
        samples[sample_idx] = float(arr[indices].mean())

    return {
        "estimate": float(arr.mean()),
        "ci": percentile_interval(
            samples,
            method="bootstrap_percentile_mean",
        ).to_dict(),
        "bootstrap": {
            "n_resamples": int(DEFAULT_BOOTSTRAP_RESAMPLES),
            "seed": int(DEFAULT_BOOTSTRAP_SEED),
            "resampling": "iid_rows",
            "interval": "percentile",
        },
    }


def _bootstrap_mean_summary_nullable(values: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError("Mean summary expects a one-dimensional array")
    defined = arr[~np.isnan(arr)]
    if len(defined) == 0:
        return {
            "estimate": None,
            "ci": None,
            "n_defined": 0,
        }
    return {
        **_bootstrap_mean_summary(defined),
        "n_defined": int(len(defined)),
    }


def _paired_bootstrap_mean_delta(
    baseline: np.ndarray,
    candidate: np.ndarray,
) -> dict[str, Any]:
    baseline_arr = np.asarray(baseline, dtype=np.float64)
    candidate_arr = np.asarray(candidate, dtype=np.float64)
    if baseline_arr.shape != candidate_arr.shape:
        raise ValueError("Paired mean-delta inputs must have matching shapes")
    if baseline_arr.ndim != 1:
        raise ValueError("Paired mean-delta inputs must be one-dimensional")

    n = len(baseline_arr)
    rng = np.random.default_rng(DEFAULT_BOOTSTRAP_SEED)
    samples = np.empty(DEFAULT_BOOTSTRAP_RESAMPLES, dtype=np.float64)
    for sample_idx in range(DEFAULT_BOOTSTRAP_RESAMPLES):
        indices = rng.choice(n, size=n, replace=True)
        samples[sample_idx] = float(
            candidate_arr[indices].mean() - baseline_arr[indices].mean()
        )

    return {
        "estimate": float(candidate_arr.mean() - baseline_arr.mean()),
        "ci": percentile_interval(
            samples,
            method="bootstrap_percentile_paired_mean",
        ).to_dict(),
        "bootstrap": {
            "n_resamples": int(DEFAULT_BOOTSTRAP_RESAMPLES),
            "seed": int(DEFAULT_BOOTSTRAP_SEED),
            "resampling": "paired_by_sample_id",
            "interval": "percentile",
        },
    }


def _paired_bootstrap_mean_delta_nullable(
    baseline: np.ndarray,
    candidate: np.ndarray,
) -> dict[str, Any]:
    baseline_arr = np.asarray(baseline, dtype=np.float64)
    candidate_arr = np.asarray(candidate, dtype=np.float64)
    if baseline_arr.shape != candidate_arr.shape:
        raise ValueError("Paired mean-delta inputs must have matching shapes")
    if baseline_arr.ndim != 1:
        raise ValueError("Paired mean-delta inputs must be one-dimensional")

    defined_mask = ~np.isnan(baseline_arr) & ~np.isnan(candidate_arr)
    paired_n = int(defined_mask.sum())
    if paired_n == 0:
        return {
            "estimate": None,
            "ci": None,
            "n_paired": 0,
        }
    return {
        **_paired_bootstrap_mean_delta(
            baseline_arr[defined_mask],
            candidate_arr[defined_mask],
        ),
        "n_paired": paired_n,
    }


def _summarize_condition(name: str, arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    csv2_yes = arrays["csv2_yes"]
    n = len(csv2_yes)
    n_yes = int(csv2_yes.sum())
    empty_n = int(arrays["empty_response"].sum())
    return {
        "name": name,
        "n": n,
        "empty_response_n": empty_n,
        "csv2_yes": {
            "estimate": float(csv2_yes.mean()),
            "n_yes": n_yes,
            "n": n,
            "ci": wilson_interval(n_yes, n).to_dict(),
        },
        "substantive_compliance": {
            "estimate": float(arrays["substantive_compliance"].mean()),
            "n_yes": int(arrays["substantive_compliance"].sum()),
            "n": n,
            "ci": wilson_interval(
                int(arrays["substantive_compliance"].sum()),
                n,
            ).to_dict(),
        },
        "intent_match": _bootstrap_mean_summary(arrays["intent_match"]),
        "C": _bootstrap_mean_summary(arrays["C"]),
        "S": _bootstrap_mean_summary(arrays["S"]),
        "V": {
            **_bootstrap_mean_summary(arrays["V"]),
            "v_null_count": int(arrays["v_null_count"][0]),
        },
        "pivot_position": {
            **_bootstrap_mean_summary_nullable(arrays["pivot_position"]),
            "pivot_null_count": int(arrays["pivot_null_count"][0]),
        },
        "harmful_payload_share": _bootstrap_mean_summary(
            arrays["harmful_payload_share"]
        ),
    }


def _parse_condition_spec(raw: str) -> tuple[str, Path, float]:
    try:
        name, path_raw, alpha_raw = raw.split(":", 2)
    except ValueError as exc:
        raise ValueError(
            "Condition format must be name:/path/to/csv2_dir:alpha"
        ) from exc
    return name, Path(path_raw), float(alpha_raw)


def main() -> None:
    args = parse_args()

    baseline_rows = _load_jsonl(_alpha_file(args.baseline_dir, args.baseline_alpha))
    _require_csv2(baseline_rows, context="baseline")
    baseline_by_id = _rows_by_id(baseline_rows, context="baseline")
    baseline_ids = sorted(baseline_by_id)
    baseline_arrays = _condition_arrays(baseline_by_id, baseline_ids)

    condition_summaries = [_summarize_condition("baseline", baseline_arrays)]
    pairwise_vs_baseline: dict[str, dict[str, Any]] = {}

    for raw in args.condition:
        name, csv2_dir, alpha = _parse_condition_spec(raw)
        condition_rows = _load_jsonl(_alpha_file(csv2_dir, alpha))
        _require_csv2(condition_rows, context=name)
        condition_by_id = _rows_by_id(condition_rows, context=name)
        ordered_ids = _require_parity(
            baseline_by_id,
            condition_by_id,
            context=name,
        )

        condition_arrays = _condition_arrays(condition_by_id, ordered_ids)
        condition_summaries.append(_summarize_condition(name, condition_arrays))
        pairwise_vs_baseline[name] = {
            "n": len(ordered_ids),
            "csv2_yes": paired_bootstrap_binary_rate_difference(
                baseline_arrays["csv2_yes"],
                condition_arrays["csv2_yes"],
                seed=DEFAULT_BOOTSTRAP_SEED,
                n_resamples=DEFAULT_BOOTSTRAP_RESAMPLES,
            ),
            "substantive_compliance": paired_bootstrap_binary_rate_difference(
                baseline_arrays["substantive_compliance"],
                condition_arrays["substantive_compliance"],
                seed=DEFAULT_BOOTSTRAP_SEED,
                n_resamples=DEFAULT_BOOTSTRAP_RESAMPLES,
            ),
            "intent_match": _paired_bootstrap_mean_delta(
                baseline_arrays["intent_match"],
                condition_arrays["intent_match"],
            ),
            "C": _paired_bootstrap_mean_delta(
                baseline_arrays["C"],
                condition_arrays["C"],
            ),
            "S": _paired_bootstrap_mean_delta(
                baseline_arrays["S"],
                condition_arrays["S"],
            ),
            "V": _paired_bootstrap_mean_delta(
                baseline_arrays["V"],
                condition_arrays["V"],
            ),
            "harmful_payload_share": _paired_bootstrap_mean_delta(
                baseline_arrays["harmful_payload_share"],
                condition_arrays["harmful_payload_share"],
            ),
            "pivot_position": _paired_bootstrap_mean_delta_nullable(
                baseline_arrays["pivot_position"],
                condition_arrays["pivot_position"],
            ),
        }

    payload = {
        "baseline": {
            "dir": str(args.baseline_dir),
            "alpha": float(args.baseline_alpha),
        },
        "conditions": condition_summaries,
        "paired_vs_baseline": pairwise_vs_baseline,
    }
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Saved D7 CSV2 report: {args.output_path}")


if __name__ == "__main__":
    main()
