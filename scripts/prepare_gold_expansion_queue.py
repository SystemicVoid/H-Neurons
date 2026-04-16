#!/usr/bin/env python3
"""Prepare the gold-expansion labeling queue for jailbreak evaluator disputes."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from utils import format_alpha_label

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BINARY_DIR = ROOT / "data/gemma3_4b/intervention/jailbreak/experiment"
DEFAULT_V2_DIR = ROOT / "data/gemma3_4b/intervention/jailbreak/csv2_evaluation"
DEFAULT_V3_DIR = ROOT / "data/gemma3_4b/intervention/jailbreak/csv2_v3_evaluation"
DEFAULT_GOLD_PATH = ROOT / "tests/gold_labels/jailbreak_cross_alpha_gold.jsonl"
DEFAULT_OUTPUT_PATH = ROOT / "data/judge_validation/gold_expansion/labeling_queue.jsonl"
ALPHAS = (0.0, 1.0, 1.5, 3.0)

DISAGREEMENT_PRIORITIES = {
    "v3_parse_failed": 0,
    "two_vs_one_binary_dissents": 1,
    "two_vs_one_v2_dissents": 1,
    "two_vs_one_v3_dissents": 1,
    "v2_borderline": 2,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary_dir", type=Path, default=DEFAULT_BINARY_DIR)
    parser.add_argument("--v2_dir", type=Path, default=DEFAULT_V2_DIR)
    parser.add_argument("--v3_dir", type=Path, default=DEFAULT_V3_DIR)
    parser.add_argument("--gold_path", type=Path, default=DEFAULT_GOLD_PATH)
    parser.add_argument("--output_path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--calibration_per_label", type=int, default=15)
    parser.add_argument("--alphas", type=float, nargs="+", default=list(ALPHAS))
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def evaluation_key(row: dict[str, Any]) -> tuple[str, float]:
    return str(row["id"]), float(row["alpha"])


def _load_multi_alpha_dir(
    directory: Path, alphas: tuple[float, ...]
) -> dict[tuple[str, float], dict[str, Any]]:
    rows_by_key: dict[tuple[str, float], dict[str, Any]] = {}
    for alpha in alphas:
        path = directory / f"alpha_{format_alpha_label(alpha)}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"Missing alpha file: {path}")
        for row in load_jsonl(path):
            key = evaluation_key(row)
            if key in rows_by_key:
                raise ValueError(f"Duplicate (id, alpha) in {path}: {key}")
            rows_by_key[key] = row
    return rows_by_key


def _require_same_keysets(
    binary_rows: dict[tuple[str, float], dict[str, Any]],
    v2_rows: dict[tuple[str, float], dict[str, Any]],
    v3_rows: dict[tuple[str, float], dict[str, Any]],
) -> list[tuple[str, float]]:
    keysets = {
        "binary": set(binary_rows),
        "csv2_v2": set(v2_rows),
        "csv2_v3": set(v3_rows),
    }
    if len({frozenset(keys) for keys in keysets.values()}) != 1:
        raise ValueError(
            "Evaluator key parity failed: "
            + ", ".join(f"{name}={len(keys)}" for name, keys in keysets.items())
        )
    return sorted(keysets["binary"], key=lambda item: (item[1], item[0]))


def _binary_verdict(row: dict[str, Any]) -> str:
    verdict = str(row.get("judge", "")).strip()
    if verdict not in {"HARMFUL", "SAFE"}:
        raise ValueError(
            f"Unexpected binary verdict for {evaluation_key(row)}: {verdict}"
        )
    return verdict


def _v2_verdict(row: dict[str, Any]) -> str:
    verdict = str(row.get("csv2", {}).get("harmful_binary", "")).strip()
    if verdict not in {"yes", "no", "borderline"}:
        raise ValueError(f"Unexpected v2 verdict for {evaluation_key(row)}: {verdict}")
    return verdict


def _v3_verdict(row: dict[str, Any]) -> str | None:
    csv2 = row.get("csv2", {})
    if csv2.get("error") == "parse_failed":
        return None
    verdict = str(csv2.get("harmful_binary", "")).strip()
    if verdict not in {"yes", "no"}:
        raise ValueError(f"Unexpected v3 verdict for {evaluation_key(row)}: {verdict}")
    return verdict


def _validate_shared_fields(
    binary_row: dict[str, Any],
    v2_row: dict[str, Any],
    v3_row: dict[str, Any],
) -> None:
    for field in ("id", "alpha", "goal", "category", "response"):
        values = [row.get(field) for row in (binary_row, v2_row, v3_row)]
        if values[0] != values[1] or values[0] != values[2]:
            key = evaluation_key(binary_row)
            raise ValueError(f"Mismatched {field!r} across evaluators for {key}")


def classify_joined_record(
    binary_row: dict[str, Any],
    v2_row: dict[str, Any],
    v3_row: dict[str, Any],
) -> dict[str, Any]:
    _validate_shared_fields(binary_row, v2_row, v3_row)
    binary_verdict = _binary_verdict(binary_row)
    v2_verdict = _v2_verdict(v2_row)
    v3_verdict = _v3_verdict(v3_row)
    v3_csv2 = v3_row.get("csv2", {})

    if v3_csv2.get("error") == "parse_failed":
        disagreement_type = "v3_parse_failed"
    elif v2_verdict == "borderline":
        disagreement_type = "v2_borderline"
    else:
        binary_harmful = binary_verdict == "HARMFUL"
        v2_harmful = v2_verdict == "yes"
        v3_harmful = v3_verdict == "yes"
        if binary_harmful and v2_harmful and v3_harmful:
            disagreement_type = "all_agree_harmful"
        elif not binary_harmful and not v2_harmful and not v3_harmful:
            disagreement_type = "all_agree_safe"
        elif v2_harmful == v3_harmful != binary_harmful:
            disagreement_type = "two_vs_one_binary_dissents"
        elif binary_harmful == v3_harmful != v2_harmful:
            disagreement_type = "two_vs_one_v2_dissents"
        elif binary_harmful == v2_harmful != v3_harmful:
            disagreement_type = "two_vs_one_v3_dissents"
        else:
            raise ValueError(
                f"Unclassifiable evaluator pattern for {evaluation_key(binary_row)}"
            )

    return {
        "id": str(binary_row["id"]),
        "alpha": float(binary_row["alpha"]),
        "goal": str(binary_row["goal"]),
        "category": str(binary_row["category"]),
        "response": str(binary_row["response"]),
        "binary_verdict": binary_verdict,
        "v2_verdict": v2_verdict,
        "v2_reasoning": v2_row.get("csv2", {}).get("reasoning"),
        "v3_verdict": v3_verdict,
        "v3_primary_outcome": None
        if v3_verdict is None
        else v3_csv2.get("primary_outcome"),
        "v3_C": None if v3_verdict is None else v3_csv2.get("C"),
        "v3_S": None if v3_verdict is None else v3_csv2.get("S"),
        "v3_V": None if v3_verdict is None else v3_csv2.get("V"),
        "v3_reasoning": None if v3_verdict is None else v3_csv2.get("reasoning"),
        "v3_harmful_spans": None
        if v3_verdict is None
        else v3_csv2.get("harmful_spans"),
        "disagreement_type": disagreement_type,
        "queue_kind": "disagreement",
        "calibration_consensus_label": None,
        "v3_error": v3_csv2.get("error"),
    }


def apportion_alpha_samples(
    counts_by_alpha: dict[float, int], total_requested: int
) -> dict[float, int]:
    total_available = sum(counts_by_alpha.values())
    if total_requested < 0:
        raise ValueError("total_requested must be non-negative")
    if total_requested > total_available:
        raise ValueError(
            f"Requested {total_requested} calibration rows but only "
            f"{total_available} are available"
        )
    if total_requested == 0 or total_available == 0:
        return {alpha: 0 for alpha in sorted(counts_by_alpha)}

    quotas = {
        alpha: (total_requested * count / total_available)
        for alpha, count in counts_by_alpha.items()
    }
    allocations = {
        alpha: min(int(quotas[alpha]), counts_by_alpha[alpha])
        for alpha in counts_by_alpha
    }
    assigned = sum(allocations.values())
    while assigned < total_requested:
        candidates = [
            alpha
            for alpha, available in counts_by_alpha.items()
            if allocations[alpha] < available
        ]
        alpha = max(
            candidates, key=lambda item: (quotas[item] - allocations[item], -item)
        )
        allocations[alpha] += 1
        assigned += 1
    return {alpha: allocations[alpha] for alpha in sorted(allocations)}


def sample_calibration_pool(
    rows: list[dict[str, Any]],
    *,
    consensus_label: str,
    total_requested: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    rows_by_alpha: dict[float, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_alpha[float(row["alpha"])].append(row)
    allocations = apportion_alpha_samples(
        {alpha: len(alpha_rows) for alpha, alpha_rows in rows_by_alpha.items()},
        total_requested,
    )
    selected: list[dict[str, Any]] = []
    for alpha in sorted(rows_by_alpha):
        alpha_rows = sorted(rows_by_alpha[alpha], key=lambda row: str(row["id"]))
        sample_size = allocations.get(alpha, 0)
        if sample_size == 0:
            continue
        sampled = rng.sample(alpha_rows, sample_size)
        for row in sorted(sampled, key=lambda item: str(item["id"])):
            copied = dict(row)
            copied["queue_kind"] = "calibration"
            copied["calibration_consensus_label"] = consensus_label
            selected.append(copied)
    return selected


def build_labeling_queue(
    *,
    binary_dir: Path,
    v2_dir: Path,
    v3_dir: Path,
    gold_path: Path,
    alphas: tuple[float, ...] = ALPHAS,
    seed: int = 42,
    calibration_per_label: int = 15,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    binary_rows = _load_multi_alpha_dir(binary_dir, alphas)
    v2_rows = _load_multi_alpha_dir(v2_dir, alphas)
    v3_rows = _load_multi_alpha_dir(v3_dir, alphas)
    joined_keys = _require_same_keysets(binary_rows, v2_rows, v3_rows)
    gold_keys = {evaluation_key(row) for row in load_jsonl(gold_path)}

    agreement_pools: dict[str, list[dict[str, Any]]] = {
        "all_agree_harmful": [],
        "all_agree_safe": [],
    }
    disagreement_rows: list[dict[str, Any]] = []
    counts_by_type_alpha: dict[str, Counter[float]] = defaultdict(Counter)

    for key in joined_keys:
        if key in gold_keys:
            continue
        classified = classify_joined_record(
            binary_rows[key], v2_rows[key], v3_rows[key]
        )
        counts_by_type_alpha[classified["disagreement_type"]][classified["alpha"]] += 1
        if classified["disagreement_type"] in agreement_pools:
            agreement_pools[classified["disagreement_type"]].append(classified)
        else:
            disagreement_rows.append(classified)

    rng = random.Random(seed)
    calibration_rows = sample_calibration_pool(
        agreement_pools["all_agree_harmful"],
        consensus_label="HARMFUL",
        total_requested=calibration_per_label,
        rng=rng,
    )
    calibration_rows.extend(
        sample_calibration_pool(
            agreement_pools["all_agree_safe"],
            consensus_label="SAFE",
            total_requested=calibration_per_label,
            rng=rng,
        )
    )
    calibration_rows.sort(
        key=lambda row: (
            row["alpha"],
            str(row["id"]),
            str(row["calibration_consensus_label"]),
        )
    )
    disagreement_rows.sort(
        key=lambda row: (
            DISAGREEMENT_PRIORITIES[row["disagreement_type"]],
            row["alpha"],
            str(row["id"]),
        )
    )
    queue_rows = calibration_rows + disagreement_rows

    summary = {
        "joined_rows": len(joined_keys),
        "gold_excluded_rows": len(gold_keys & set(joined_keys)),
        "non_gold_rows": len(joined_keys) - len(gold_keys & set(joined_keys)),
        "queue_rows": len(queue_rows),
        "calibration_rows": len(calibration_rows),
        "disagreement_rows": len(disagreement_rows),
        "counts_by_disagreement_type_alpha": {
            kind: {
                format_alpha_label(alpha): count
                for alpha, count in sorted(alpha_counts.items())
            }
            for kind, alpha_counts in sorted(counts_by_type_alpha.items())
        },
    }
    return queue_rows, summary


def _print_summary(summary: dict[str, Any]) -> None:
    print(
        "Prepared labeling queue: "
        f"joined={summary['joined_rows']} "
        f"non_gold={summary['non_gold_rows']} "
        f"queue={summary['queue_rows']} "
        f"calibration={summary['calibration_rows']} "
        f"disagreements={summary['disagreement_rows']}"
    )
    print("Counts by disagreement_type x alpha:")
    for kind, alpha_counts in summary["counts_by_disagreement_type_alpha"].items():
        parts = ", ".join(
            f"alpha_{alpha}={count}" for alpha, count in alpha_counts.items()
        )
        print(f"  {kind}: {parts}")


def main() -> None:
    args = parse_args()
    queue_rows, summary = build_labeling_queue(
        binary_dir=args.binary_dir,
        v2_dir=args.v2_dir,
        v3_dir=args.v3_dir,
        gold_path=args.gold_path,
        alphas=tuple(float(alpha) for alpha in args.alphas),
        seed=int(args.seed),
        calibration_per_label=int(args.calibration_per_label),
    )
    write_jsonl(args.output_path, queue_rows)
    print(f"Wrote {len(queue_rows)} rows to {args.output_path}")
    _print_summary(summary)


if __name__ == "__main__":
    main()
