"""Post-evaluation analysis of CSV2 jailbreak responses."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from evaluate_csv2 import CATEGORY_TO_FAMILY, normalize_csv2_payload

EXPERIMENT_DIR = Path("data/gemma3_4b/intervention/jailbreak/csv2_evaluation")
ALPHAS = [0.0, 1.5, 3.0]
PRIMARY_OUTCOME_ORDER = [
    "refusal",
    "deflection",
    "benign_answer",
    "partial_compliance",
    "substantive_compliance",
]


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return 0.0, 0.0, 0.0
    p_hat = k / n
    denom = 1 + z**2 / n
    centre = (p_hat + z**2 / (2 * n)) / denom
    spread = z * np.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * n)) / n) / denom
    return p_hat, max(0.0, centre - spread), min(1.0, centre + spread)


def _fmt_ci(k: int, n: int) -> str:
    rate, lo, hi = wilson_ci(k, n)
    return f"{rate:.1%} [{lo:.1%}, {hi:.1%}] ({k}/{n})"


def _mean(values: Sequence[float | int]) -> float:
    if not values:
        return 0.0
    return float(np.asarray(values, dtype=float).mean())


def _float_csv2_field(row: dict, key: str) -> float | None:
    value = _csv2(row).get(key)
    if value is None:
        return None
    return float(value)


def load_alpha(experiment_dir: Path, alpha: float) -> list[dict]:
    path = experiment_dir / f"alpha_{alpha:.1f}.jsonl"
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _csv2(rec: dict) -> dict:
    csv2 = rec.get("csv2", {})
    if not isinstance(csv2, dict) or not csv2:
        return {}
    return normalize_csv2_payload(csv2)


def _valid_rows(rows: list[dict]) -> list[dict]:
    return [row for row in rows if _csv2(row) and not _csv2(row).get("error")]


def _harmful_rows(rows: list[dict]) -> list[dict]:
    return [
        row for row in _valid_rows(rows) if _csv2(row).get("harmful_binary") == "yes"
    ]


def data_summary(
    data_by_alpha: dict[float, list[dict]], alphas: Sequence[float]
) -> None:
    print("=" * 72)
    print("1. DATA SUMMARY & VALIDATION")
    print("=" * 72)

    for alpha in alphas:
        rows = data_by_alpha[alpha]
        annotated = sum(1 for row in rows if "csv2" in row)
        valid = len(_valid_rows(rows))
        errors = sum(1 for row in rows if _csv2(row).get("error"))
        span_errors = sum(int(_csv2(row).get("span_errors", 0)) for row in rows)
        corrections = sum(int(_csv2(row).get("span_corrections", 0)) for row in rows)
        schemas = Counter(
            _csv2(row).get("schema_version", "?") for row in rows if _csv2(row)
        )
        print(
            f"  alpha={alpha:.1f}: {len(rows)} records, {annotated} annotated, "
            f"{valid} valid, {errors} errors"
        )
        print(
            f"    schemas={dict(schemas)} | span_errors={span_errors} | "
            f"span_corrections={corrections}"
        )


def outcome_summary(
    data_by_alpha: dict[float, list[dict]],
    alphas: Sequence[float],
) -> None:
    print("\n" + "=" * 72)
    print("2. PRIMARY OUTCOME & HARMFUL BINARY")
    print("=" * 72)

    header = (
        "  "
        + f"{'Alpha':<8}"
        + "".join(f"{outcome:<24}" for outcome in PRIMARY_OUTCOME_ORDER)
    )
    print("\nPrimary outcome distribution:")
    print(header)
    print("  " + "-" * len(header))
    for alpha in alphas:
        rows = _valid_rows(data_by_alpha[alpha])
        n = len(rows) or 1
        counts = Counter(_csv2(row).get("primary_outcome") for row in rows)
        print(
            "  "
            + f"{alpha:<8.1f}"
            + "".join(
                f"{counts.get(outcome, 0):>3} ({counts.get(outcome, 0) / n:>4.0%})      "
                for outcome in PRIMARY_OUTCOME_ORDER
            )
        )

    print("\nDerived harmful_binary distribution:")
    print(f"  {'Alpha':<8} {'yes':<18} {'no':<18}")
    print("  " + "-" * 44)
    for alpha in alphas:
        rows = _valid_rows(data_by_alpha[alpha])
        n = len(rows) or 1
        counts = Counter(_csv2(row).get("harmful_binary") for row in rows)
        print(
            f"  {alpha:<8.1f} "
            f"{counts.get('yes', 0):>3} ({counts.get('yes', 0) / n:.0%})      "
            f"{counts.get('no', 0):>3} ({counts.get('no', 0) / n:.0%})"
        )

    print("\nDerived harmful_binary='yes' vs existing judge HARMFUL:")
    print(
        f"  {'Alpha':<8} {'csv2_yes':<14} {'judge_HARMFUL':<16} "
        f"{'agree':<14} {'csv2↑judge↓':<14} {'csv2↓judge↑':<14}"
    )
    print("  " + "-" * 70)
    for alpha in alphas:
        rows = _valid_rows(data_by_alpha[alpha])
        n = len(rows) or 1
        csv2_yes = sum(1 for row in rows if _csv2(row).get("harmful_binary") == "yes")
        judge_h = sum(1 for row in rows if row.get("compliance"))
        both = sum(
            1
            for row in rows
            if _csv2(row).get("harmful_binary") == "yes" and row.get("compliance")
        )
        csv2_only = csv2_yes - both
        judge_only = judge_h - both
        agree = both + (n - csv2_yes - judge_h + both)
        print(
            f"  {alpha:<8.1f} {csv2_yes:>3} ({csv2_yes / n:.0%})    "
            f"{judge_h:>3} ({judge_h / n:.0%})      "
            f"{agree:>3} ({agree / n:.0%})    "
            f"{csv2_only:>3}            {judge_only:>3}"
        )


def intent_and_axes(
    data_by_alpha: dict[float, list[dict]],
    alphas: Sequence[float],
) -> None:
    print("\n" + "=" * 72)
    print("3. INTENT MATCH, COMMITMENT, AND SPECIFICITY")
    print("=" * 72)

    print("\nIntent match:")
    print(f"  {'Alpha':<8} {'0':<14} {'1':<14} {'2':<14}")
    print("  " + "-" * 52)
    for alpha in alphas:
        rows = _valid_rows(data_by_alpha[alpha])
        n = len(rows) or 1
        counts = Counter(_csv2(row).get("intent_match", 0) for row in rows)
        print(
            f"  {alpha:<8.1f}"
            f" {counts.get(0, 0):>3} ({counts.get(0, 0) / n:>4.0%})"
            f"    {counts.get(1, 0):>3} ({counts.get(1, 0) / n:>4.0%})"
            f"    {counts.get(2, 0):>3} ({counts.get(2, 0) / n:>4.0%})"
        )

    print("\nCommitment (C) and specificity (S):")
    print(f"  {'Alpha':<8} {'Mean C':<10} {'Mean S':<10} {'Mean S | harmful':<18}")
    print("  " + "-" * 52)
    for alpha in alphas:
        rows = _valid_rows(data_by_alpha[alpha])
        harmful = _harmful_rows(data_by_alpha[alpha])
        c_vals = [float(_csv2(row).get("C", 0)) for row in rows]
        s_vals = [float(_csv2(row).get("S", 0)) for row in rows]
        s_harmful = [float(_csv2(row).get("S", 0)) for row in harmful]
        print(
            f"  {alpha:<8.1f} {_mean(c_vals):<10.2f} "
            f"{_mean(s_vals):<10.2f} "
            f"{_mean(s_harmful):<18.2f}"
        )


def validity_and_wrappers(
    data_by_alpha: dict[float, list[dict]],
    alphas: Sequence[float],
) -> None:
    print("\n" + "=" * 72)
    print("4. VALIDITY AND WRAPPER TAGS")
    print("=" * 72)

    print("\nValidity (V) among rows with S >= 1:")
    for alpha in alphas:
        rows = [
            row
            for row in _valid_rows(data_by_alpha[alpha])
            if _csv2(row).get("S", 0) >= 1
        ]
        counts = Counter(_csv2(row).get("V") for row in rows)
        values = [
            float(_csv2(row).get("V", 0))
            for row in rows
            if _csv2(row).get("V") is not None
        ]
        print(
            f"  alpha={alpha:.1f} (n={len(rows)}): "
            + " | ".join(f"V{v}={counts.get(v, 0)}" for v in range(4))
            + f" | null={counts.get(None, 0)}"
            + (f" | mean={_mean(values):.2f}" if values else "")
        )

    print("\nWrapper tag prevalence:")
    print(f"  {'Alpha':<8} {'R':<14} {'P':<14} {'F':<14} {'T':<14}")
    print("  " + "-" * 68)
    for alpha in alphas:
        rows = _valid_rows(data_by_alpha[alpha])
        n = len(rows) or 1
        print(f"  {alpha:<8.1f}", end="")
        for tag in ("R", "P", "F", "T"):
            k = sum(1 for row in rows if _csv2(row).get(tag))
            print(f" {k:>3} ({k / n:>4.0%})    ", end="")
        print()


def localization_summary(
    data_by_alpha: dict[float, list[dict]],
    alphas: Sequence[float],
) -> None:
    print("\n" + "=" * 72)
    print("5. LOCALIZATION METRICS")
    print("=" * 72)

    for alpha in alphas:
        harmful = _harmful_rows(data_by_alpha[alpha])
        pivot_values = [
            pivot_value
            for row in harmful
            if (pivot_value := _float_csv2_field(row, "pivot_position")) is not None
        ]
        payload_values = [
            float(_csv2(row).get("harmful_payload_share", 0.0))
            for row in harmful
            if _csv2(row).get("harmful_payload_share") is not None
        ]
        if not harmful:
            print(f"  alpha={alpha:.1f}: no harmful rows")
            continue
        pivot_msg = "n/a"
        if pivot_values:
            pivot_arr = np.asarray(pivot_values, dtype=float)
            pivot_msg = (
                f"mean pivot={pivot_arr.mean():.3f}, median={np.median(pivot_arr):.3f}"
            )
        payload_arr = (
            np.asarray(payload_values, dtype=float)
            if payload_values
            else np.asarray([0.0])
        )
        print(
            f"  alpha={alpha:.1f} (n={len(harmful)}): "
            f"{pivot_msg}, mean payload={payload_arr.mean():.3f}, "
            f"p75 payload={np.percentile(payload_arr, 75):.3f}"
        )


def task_family_breakdown(
    data_by_alpha: dict[float, list[dict]],
    alphas: Sequence[float],
) -> None:
    print("\n" + "=" * 72)
    print("6. TASK FAMILY BREAKDOWN")
    print("=" * 72)

    families = sorted(set(CATEGORY_TO_FAMILY.values()))
    print(f"  {'Family':<14}", end="")
    for alpha in alphas:
        print(f" {'alpha=' + f'{alpha:.1f}':<22}", end="")
    print()
    print("  " + "-" * (14 + 22 * len(alphas)))

    for family in families:
        categories = {
            category
            for category, mapped in CATEGORY_TO_FAMILY.items()
            if mapped == family
        }
        print(f"  {family:<14}", end="")
        for alpha in alphas:
            rows = [
                row
                for row in _valid_rows(data_by_alpha[alpha])
                if row.get("category") in categories
            ]
            n = len(rows)
            harmful = sum(
                1 for row in rows if _csv2(row).get("harmful_binary") == "yes"
            )
            print(f" {_fmt_ci(harmful, n) if n else 'n/a':<22}", end="")
        print()


def overall_summary(
    data_by_alpha: dict[float, list[dict]],
    alphas: Sequence[float],
) -> None:
    print("\n" + "=" * 72)
    print("7. OVERALL SUMMARY")
    print("=" * 72)
    print(
        "\n  | Alpha | Harmful yes | Mean intent | Mean C | Mean S | Mean V (S>=1) | "
        "Mean pivot | Mean payload |"
    )
    print(
        "  |-------|-------------|-------------|--------|--------|---------------|"
        "------------|--------------|"
    )

    for alpha in alphas:
        rows = _valid_rows(data_by_alpha[alpha])
        n = len(rows) or 1
        harmful = _harmful_rows(data_by_alpha[alpha])
        harmful_yes = sum(
            1 for row in rows if _csv2(row).get("harmful_binary") == "yes"
        )
        intent_vals = [float(_csv2(row).get("intent_match", 0)) for row in rows]
        c_vals = [float(_csv2(row).get("C", 0)) for row in rows]
        s_vals = [float(_csv2(row).get("S", 0)) for row in rows]
        v_vals = [
            float(_csv2(row).get("V", 0))
            for row in rows
            if _csv2(row).get("S", 0) >= 1 and _csv2(row).get("V") is not None
        ]
        pivot_vals = [
            pivot_value
            for row in harmful
            if (pivot_value := _float_csv2_field(row, "pivot_position")) is not None
        ]
        payload_vals = [
            float(_csv2(row).get("harmful_payload_share", 0.0))
            for row in harmful
            if _csv2(row).get("harmful_payload_share") is not None
        ]
        print(
            f"  | {alpha:.1f}   | {_fmt_ci(harmful_yes, n):<11} | "
            f"{_mean(intent_vals):>11.2f} | "
            f"{_mean(c_vals):>6.2f} | "
            f"{_mean(s_vals):>6.2f} | "
            f"{_mean(v_vals):>13.2f} | "
            f"{_mean(pivot_vals):>10.3f} | "
            f"{_mean(payload_vals):>12.3f} |"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze CSV2 annotated jailbreak responses"
    )
    parser.add_argument("--experiment_dir", type=str, default=str(EXPERIMENT_DIR))
    parser.add_argument("--alphas", type=float, nargs="+", default=ALPHAS)
    args = parser.parse_args()

    experiment_dir = Path(args.experiment_dir)
    alphas = args.alphas

    print(f"CSV2 Analysis: {experiment_dir}")
    print(f"Alphas: {alphas}\n")

    data_by_alpha: dict[float, list[dict]] = {}
    for alpha in alphas:
        rows = load_alpha(experiment_dir, alpha)
        data_by_alpha[alpha] = rows
        print(f"  Loaded alpha={alpha:.1f}: {len(rows)} records")

    unannotated = sum(
        1 for alpha in alphas for row in data_by_alpha[alpha] if "csv2" not in row
    )
    if unannotated:
        print(
            f"\n  WARNING: {unannotated} records not annotated. Run evaluate_csv2.py first."
        )
        sys.exit(1)

    data_summary(data_by_alpha, alphas)
    outcome_summary(data_by_alpha, alphas)
    intent_and_axes(data_by_alpha, alphas)
    validity_and_wrappers(data_by_alpha, alphas)
    localization_summary(data_by_alpha, alphas)
    task_family_breakdown(data_by_alpha, alphas)
    overall_summary(data_by_alpha, alphas)


if __name__ == "__main__":
    main()
