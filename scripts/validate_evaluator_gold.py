"""Validate LLM judge against human gold-label audit sets.

Re-runs the GPT-4o judge on curated cases with human-verified ground truth
and reports agreement. Exit code 1 if agreement falls below threshold.

Usage:
    uv run python scripts/validate_evaluator_gold.py --benchmark falseqa
    uv run python scripts/validate_evaluator_gold.py --benchmark jailbreak
    uv run python scripts/validate_evaluator_gold.py --benchmark all
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from evaluate_intervention import judge_falseqa, judge_jailbreak

load_dotenv()

GOLD_DIR = Path(__file__).resolve().parent.parent / "tests" / "gold_labels"

BENCHMARK_CONFIG: dict[str, dict] = {
    "falseqa": {
        "fixture": "falseqa_candidates.jsonl",
        "judge_fn": "falseqa",
        "label_field": "human_label",
        "valid_labels": {"ACCEPTED", "REFUSED"},
    },
    "jailbreak": {
        "fixture": "jailbreak_candidates.jsonl",
        "judge_fn": "jailbreak",
        "label_field": "human_label",
        "valid_labels": {"HARMFUL", "SAFE"},
    },
}


def load_gold(fixture_name: str) -> list[dict]:
    path = GOLD_DIR / fixture_name
    if not path.exists():
        print(f"ERROR: Gold-label fixture not found: {path}")
        sys.exit(1)
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def validate_benchmark(
    benchmark: str,
    client: OpenAI,
    judge_model: str,
    threshold: float,
) -> bool:
    config = BENCHMARK_CONFIG[benchmark]
    records = load_gold(config["fixture"])

    labeled = [r for r in records if r.get(config["label_field"]) is not None]
    if not labeled:
        print(f"\n[{benchmark}] FAIL: No human-labeled cases found.")
        print(f"  ({len(records)} candidates await human review)")
        return False

    print(f"\n[{benchmark}] Validating {len(labeled)} gold-labeled cases...")

    agree = 0
    disagree = 0
    errors = 0
    disagreements: list[dict] = []

    for rec in labeled:
        human = rec[config["label_field"]]
        if human not in config["valid_labels"]:
            print(f"  WARNING: invalid human_label {human!r} for {rec['id']}")
            continue

        if config["judge_fn"] == "falseqa":
            verdict = judge_falseqa(
                client, judge_model, rec["question"], rec["response"]
            )
        elif config["judge_fn"] == "jailbreak":
            verdict = judge_jailbreak(
                client, judge_model, rec["question"], rec["response"]
            )
        else:
            print(f"  ERROR: unknown judge_fn {config['judge_fn']!r}")
            errors += 1
            continue

        if verdict in ("ERROR", "UNKNOWN"):
            errors += 1
            disagree += 1
            print(f"  ERROR judging {rec['id']}: got {verdict}")
            disagreements.append({"id": rec["id"], "human": human, "judge": verdict})
            continue

        if verdict == human:
            agree += 1
        else:
            disagree += 1
            disagreements.append({"id": rec["id"], "human": human, "judge": verdict})

    total = agree + disagree
    if total == 0:
        print("  No valid comparisons — all errored.")
        return False

    rate = agree / total
    print(f"  Agreement: {agree}/{total} ({rate:.1%})")
    if errors:
        print(f"  Errors (counted as failures): {errors}")

    if disagreements:
        print("  Disagreements:")
        for d in disagreements:
            print(f"    {d['id']}: human={d['human']}, judge={d['judge']}")

    passed = rate >= threshold
    if passed:
        print(f"  PASS (>= {threshold:.0%} threshold)")
    else:
        print(f"  FAIL (< {threshold:.0%} threshold)")
    return passed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--benchmark",
        choices=["falseqa", "jailbreak", "all"],
        default="all",
    )
    p.add_argument("--judge_model", default="gpt-4o")
    p.add_argument(
        "--threshold",
        type=float,
        default=0.80,
        help="Minimum agreement rate to pass (default: 0.80)",
    )
    p.add_argument("--api_key", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    import os

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OpenAI API key required. Set OPENAI_API_KEY or --api_key")
        sys.exit(1)

    client = OpenAI(api_key=api_key)

    benchmarks = (
        list(BENCHMARK_CONFIG.keys()) if args.benchmark == "all" else [args.benchmark]
    )

    all_passed = True
    for bench in benchmarks:
        passed = validate_benchmark(bench, client, args.judge_model, args.threshold)
        if not passed:
            all_passed = False

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
