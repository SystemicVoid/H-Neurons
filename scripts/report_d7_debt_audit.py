#!/usr/bin/env python3
"""Summarize the minimal D7 debt audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

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
    parser.add_argument("--bioasq_baseline_dir", type=Path, required=True)
    parser.add_argument("--bioasq_baseline_alpha", type=float, required=True)
    parser.add_argument("--bioasq_causal_dir", type=Path, required=True)
    parser.add_argument("--bioasq_causal_alpha", type=float, required=True)
    parser.add_argument("--benign_baseline_dir", type=Path, required=True)
    parser.add_argument("--benign_baseline_alpha", type=float, required=True)
    parser.add_argument("--benign_causal_dir", type=Path, required=True)
    parser.add_argument("--benign_causal_alpha", type=float, required=True)
    parser.add_argument("--harmful_baseline_dir", type=Path, required=True)
    parser.add_argument("--harmful_baseline_alpha", type=float, required=True)
    parser.add_argument("--harmful_causal_dir", type=Path, required=True)
    parser.add_argument("--harmful_causal_alpha", type=float, required=True)
    parser.add_argument("--harmful_baseline_csv2_dir", type=Path, required=True)
    parser.add_argument("--harmful_causal_csv2_dir", type=Path, required=True)
    parser.add_argument("--harmful_random_dir", type=Path, default=None)
    parser.add_argument("--harmful_random_alpha", type=float, default=None)
    parser.add_argument("--harmful_random_csv2_dir", type=Path, default=None)
    parser.add_argument("--output_path", type=Path, required=True)
    return parser.parse_args()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _alpha_file(root: Path, alpha: float) -> Path:
    return root / f"alpha_{format_alpha_label(alpha)}.jsonl"


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


def _require_parity(
    baseline: dict[str, dict[str, Any]],
    comparison: dict[str, dict[str, Any]],
    *,
    context: str,
) -> list[str]:
    baseline_ids = set(baseline)
    comparison_ids = set(comparison)
    if baseline_ids != comparison_ids:
        raise ValueError(f"{context}: prompt-ID parity failed")
    return sorted(baseline_ids)


def _binary_rate_summary(values: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(values, dtype=bool)
    count = int(arr.sum())
    n = int(arr.size)
    return {
        "count": count,
        "n": n,
        "rate": float(arr.mean()) if n else 0.0,
        "ci": wilson_interval(count, n).to_dict() if n else None,
    }


def _mean_summary(values: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError("Expected one-dimensional array")
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()) if arr.size else 0.0,
    }


def _paired_binary_summary(
    baseline: np.ndarray, comparison: np.ndarray
) -> dict[str, Any]:
    return {
        "baseline": _binary_rate_summary(baseline),
        "comparison": _binary_rate_summary(comparison),
        "delta": paired_bootstrap_binary_rate_difference(baseline, comparison),
    }


def _paired_mean_summary(baseline: ArrayLike, comparison: ArrayLike) -> dict[str, Any]:
    baseline_arr = np.asarray(baseline, dtype=float)
    comparison_arr = np.asarray(comparison, dtype=float)
    if baseline_arr.shape != comparison_arr.shape:
        raise ValueError("baseline and comparison must have matching shapes")
    if baseline_arr.ndim != 1:
        raise ValueError("baseline and comparison must be one-dimensional")

    n_items = len(baseline_arr)
    rng = np.random.default_rng(DEFAULT_BOOTSTRAP_SEED)
    samples = np.empty(DEFAULT_BOOTSTRAP_RESAMPLES, dtype=float)
    for sample_idx in range(DEFAULT_BOOTSTRAP_RESAMPLES):
        indices = rng.choice(n_items, size=n_items, replace=True)
        samples[sample_idx] = (
            comparison_arr[indices].mean() - baseline_arr[indices].mean()
        )

    interval = percentile_interval(
        samples,
        method="bootstrap_percentile_paired_continuous",
    )
    return {
        "baseline": _mean_summary(baseline_arr),
        "comparison": _mean_summary(comparison_arr),
        "delta": {
            "estimate": float(comparison_arr.mean() - baseline_arr.mean()),
            "ci": interval.to_dict(),
            "bootstrap": {
                "n_resamples": int(DEFAULT_BOOTSTRAP_RESAMPLES),
                "seed": int(DEFAULT_BOOTSTRAP_SEED),
                "confidence": interval.level,
                "resampling": "paired_by_sample_id",
                "interval": "percentile",
            },
        },
    }


def _bool_array(
    rows: dict[str, dict[str, Any]], ordered_ids: list[str], field: str
) -> np.ndarray:
    return np.array([bool(rows[sample_id].get(field)) for sample_id in ordered_ids])


def _response_chars_array(
    rows: dict[str, dict[str, Any]], ordered_ids: list[str]
) -> np.ndarray:
    return np.array(
        [
            float(len(str(rows[sample_id].get("response", ""))))
            for sample_id in ordered_ids
        ]
    )


def _hit_token_cap_array(
    rows: dict[str, dict[str, Any]], ordered_ids: list[str]
) -> np.ndarray:
    return np.array(
        [
            bool(rows[sample_id].get("timings", {}).get("hit_token_cap", False))
            for sample_id in ordered_ids
        ]
    )


def _csv2_payload_share_array(
    rows: dict[str, dict[str, Any]], ordered_ids: list[str]
) -> np.ndarray:
    payloads = []
    for sample_id in ordered_ids:
        raw_csv2 = rows[sample_id].get("csv2")
        if not isinstance(raw_csv2, dict):
            raise ValueError(f"Record {sample_id} has invalid csv2 payload")
        csv2 = normalize_csv2_payload(raw_csv2)
        if csv2.get("error"):
            raise ValueError(
                f"Record {sample_id} has csv2 error={csv2.get('error')!r}; "
                "repair csv2 annotations before D7 debt audit reporting"
            )
        payloads.append(float(csv2.get("harmful_payload_share", 0.0) or 0.0))
    return np.array(payloads, dtype=float)


def _load_condition_rows(root: Path, alpha: float) -> dict[str, dict[str, Any]]:
    return _rows_by_id(_load_jsonl(_alpha_file(root, alpha)), context=str(root))


def _summarize_bioasq(args: argparse.Namespace) -> dict[str, Any]:
    baseline = _load_condition_rows(
        args.bioasq_baseline_dir, args.bioasq_baseline_alpha
    )
    causal = _load_condition_rows(args.bioasq_causal_dir, args.bioasq_causal_alpha)
    ordered_ids = _require_parity(baseline, causal, context="bioasq")
    return {
        "accuracy": _paired_binary_summary(
            _bool_array(baseline, ordered_ids, "compliance"),
            _bool_array(causal, ordered_ids, "compliance"),
        )
    }


def _summarize_benign(args: argparse.Namespace) -> dict[str, Any]:
    baseline = _load_condition_rows(
        args.benign_baseline_dir, args.benign_baseline_alpha
    )
    causal = _load_condition_rows(args.benign_causal_dir, args.benign_causal_alpha)
    ordered_ids = _require_parity(baseline, causal, context="benign_jbb")
    return {
        "helpful": _paired_binary_summary(
            _bool_array(baseline, ordered_ids, "benign_helpful"),
            _bool_array(causal, ordered_ids, "benign_helpful"),
        ),
        "over_refusal": _paired_binary_summary(
            _bool_array(baseline, ordered_ids, "over_refusal"),
            _bool_array(causal, ordered_ids, "over_refusal"),
        ),
        "degenerate": _paired_binary_summary(
            _bool_array(baseline, ordered_ids, "degenerate_response"),
            _bool_array(causal, ordered_ids, "degenerate_response"),
        ),
    }


def _structure_panel(
    baseline_rows: dict[str, dict[str, Any]],
    comparison_rows: dict[str, dict[str, Any]],
    baseline_csv2_rows: dict[str, dict[str, Any]],
    comparison_csv2_rows: dict[str, dict[str, Any]],
    *,
    context: str,
) -> dict[str, Any]:
    ordered_ids = _require_parity(baseline_rows, comparison_rows, context=context)
    _require_parity(
        baseline_rows, baseline_csv2_rows, context=f"{context}_baseline_csv2"
    )
    _require_parity(
        comparison_rows, comparison_csv2_rows, context=f"{context}_comparison_csv2"
    )
    _require_parity(baseline_csv2_rows, comparison_csv2_rows, context=f"{context}_csv2")
    return {
        "hit_token_cap": _paired_binary_summary(
            _hit_token_cap_array(baseline_rows, ordered_ids),
            _hit_token_cap_array(comparison_rows, ordered_ids),
        ),
        "empty_response": _paired_binary_summary(
            _bool_array(baseline_rows, ordered_ids, "empty_response"),
            _bool_array(comparison_rows, ordered_ids, "empty_response"),
        ),
        "response_chars": _paired_mean_summary(
            _response_chars_array(baseline_rows, ordered_ids),
            _response_chars_array(comparison_rows, ordered_ids),
        ),
        "harmful_payload_share": _paired_mean_summary(
            _csv2_payload_share_array(baseline_csv2_rows, ordered_ids),
            _csv2_payload_share_array(comparison_csv2_rows, ordered_ids),
        ),
    }


def _inject_empty_response_flags(rows: dict[str, dict[str, Any]]) -> None:
    for row in rows.values():
        row["empty_response"] = len(str(row.get("response", "")).strip()) == 0


def _summarize_harmful_structure(args: argparse.Namespace) -> dict[str, Any]:
    baseline_rows = _load_condition_rows(
        args.harmful_baseline_dir, args.harmful_baseline_alpha
    )
    causal_rows = _load_condition_rows(
        args.harmful_causal_dir, args.harmful_causal_alpha
    )
    baseline_csv2_rows = _load_condition_rows(
        args.harmful_baseline_csv2_dir, args.harmful_baseline_alpha
    )
    causal_csv2_rows = _load_condition_rows(
        args.harmful_causal_csv2_dir, args.harmful_causal_alpha
    )
    _inject_empty_response_flags(baseline_rows)
    _inject_empty_response_flags(causal_rows)
    summary = {
        "causal_vs_baseline": _structure_panel(
            baseline_rows,
            causal_rows,
            baseline_csv2_rows,
            causal_csv2_rows,
            context="harmful_causal",
        )
    }

    if (
        args.harmful_random_dir is not None
        and args.harmful_random_alpha is not None
        and args.harmful_random_csv2_dir is not None
    ):
        random_rows = _load_condition_rows(
            args.harmful_random_dir, args.harmful_random_alpha
        )
        random_csv2_rows = _load_condition_rows(
            args.harmful_random_csv2_dir, args.harmful_random_alpha
        )
        _inject_empty_response_flags(random_rows)
        summary["random_vs_baseline"] = _structure_panel(
            baseline_rows,
            random_rows,
            baseline_csv2_rows,
            random_csv2_rows,
            context="harmful_random",
        )

    return summary


def main() -> None:
    args = parse_args()
    summary = {
        "bootstrap": {
            "n_resamples": int(DEFAULT_BOOTSTRAP_RESAMPLES),
            "seed": int(DEFAULT_BOOTSTRAP_SEED),
        },
        "bioasq": _summarize_bioasq(args),
        "benign_jbb": _summarize_benign(args),
        "harmful_structure": _summarize_harmful_structure(args),
    }
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved D7 debt audit report: {args.output_path}")


if __name__ == "__main__":
    main()
