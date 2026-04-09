#!/usr/bin/env python3
"""Lock D7 pilot alpha by paired decrease in csv2_yes."""

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
    wilson_interval,
)
from utils import format_alpha_label


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv2_dir", type=Path, required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument("--baseline_alpha", type=float, default=0.0)
    parser.add_argument(
        "--candidate_alphas",
        type=float,
        nargs="+",
        default=[1.0, 2.0, 4.0, 8.0],
    )
    return parser.parse_args()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _alpha_path(csv2_dir: Path, alpha: float) -> Path:
    return csv2_dir / f"alpha_{format_alpha_label(alpha)}.jsonl"


def _normalized_csv2(rec: dict[str, Any]) -> dict[str, Any]:
    csv2 = rec.get("csv2", {})
    if not isinstance(csv2, dict):
        sample_id = rec.get("id", "?")
        raise ValueError(f"Record {sample_id} has invalid csv2 payload")
    normalized = normalize_csv2_payload(csv2)
    if normalized.get("error"):
        sample_id = rec.get("id", "?")
        raise ValueError(
            f"Record {sample_id} has csv2 error={normalized.get('error')!r}; "
            "repair csv2 annotations before D7 alpha locking"
        )
    return normalized


def _csv2_yes(rec: dict[str, Any]) -> bool:
    return _normalized_csv2(rec).get("harmful_binary") == "yes"


def _build_bool_by_id(rows: list[dict[str, Any]]) -> dict[str, bool]:
    out: dict[str, bool] = {}
    for row in rows:
        sample_id = str(row["id"])
        if sample_id in out:
            raise ValueError(f"Duplicate sample id in alpha file: {sample_id}")
        out[sample_id] = _csv2_yes(row)
    return out


def _require_parity(
    baseline: dict[str, bool],
    candidate: dict[str, bool],
    *,
    context: str,
) -> list[str]:
    baseline_ids = set(baseline)
    candidate_ids = set(candidate)
    if baseline_ids != candidate_ids:
        missing_candidate = sorted(baseline_ids - candidate_ids)
        missing_baseline = sorted(candidate_ids - baseline_ids)
        raise ValueError(
            f"{context}: sample-ID parity failed "
            f"(missing_in_candidate={len(missing_candidate)} sample={missing_candidate[:5]}, "
            f"missing_in_baseline={len(missing_baseline)} sample={missing_baseline[:5]})"
        )
    return sorted(baseline_ids)


def main() -> None:
    args = parse_args()
    baseline_path = _alpha_path(args.csv2_dir, args.baseline_alpha)
    if not baseline_path.exists():
        raise FileNotFoundError(f"Missing baseline alpha file: {baseline_path}")

    baseline_map = _build_bool_by_id(_load_jsonl(baseline_path))
    results: list[dict[str, Any]] = []

    for alpha in args.candidate_alphas:
        candidate_path = _alpha_path(args.csv2_dir, alpha)
        if not candidate_path.exists():
            raise FileNotFoundError(f"Missing candidate alpha file: {candidate_path}")
        candidate_map = _build_bool_by_id(_load_jsonl(candidate_path))
        ordered_ids = _require_parity(
            baseline_map,
            candidate_map,
            context=f"alpha={format_alpha_label(alpha)}",
        )

        baseline_vec = np.array([baseline_map[sid] for sid in ordered_ids], dtype=bool)
        candidate_vec = np.array(
            [candidate_map[sid] for sid in ordered_ids], dtype=bool
        )

        baseline_yes = int(baseline_vec.sum())
        candidate_yes = int(candidate_vec.sum())
        paired_delta = paired_bootstrap_binary_rate_difference(
            baseline_vec,
            candidate_vec,
            seed=DEFAULT_BOOTSTRAP_SEED,
            n_resamples=DEFAULT_BOOTSTRAP_RESAMPLES,
        )

        results.append(
            {
                "alpha": float(alpha),
                "n": len(ordered_ids),
                "baseline": {
                    "csv2_yes": baseline_yes,
                    "csv2_yes_rate": float(baseline_vec.mean()),
                    "ci": wilson_interval(
                        baseline_yes,
                        len(ordered_ids),
                    ).to_dict(),
                },
                "candidate": {
                    "csv2_yes": candidate_yes,
                    "csv2_yes_rate": float(candidate_vec.mean()),
                    "ci": wilson_interval(
                        candidate_yes,
                        len(ordered_ids),
                    ).to_dict(),
                },
                "paired_delta_csv2_yes_pp": paired_delta,
            }
        )

    results.sort(key=lambda item: item["alpha"])
    best = min(
        results,
        key=lambda item: (
            float(item["paired_delta_csv2_yes_pp"]["estimate_pp"]),
            float(item["alpha"]),
        ),
    )

    payload = {
        "rule": "largest paired decrease in csv2_yes (min delta_pp), tie-break lower alpha",
        "baseline_alpha": float(args.baseline_alpha),
        "candidate_alphas": [float(alpha) for alpha in args.candidate_alphas],
        "selected_alpha": float(best["alpha"]),
        "results": results,
    }
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Saved D7 lock decision to {args.output_path}")


if __name__ == "__main__":
    main()
