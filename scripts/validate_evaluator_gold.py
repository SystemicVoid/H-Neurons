"""Validate LLM judge against human gold-label audit sets.

Re-runs the GPT-4o judge on curated cases with human-verified ground truth
and reports agreement. Exit code 1 if agreement falls below threshold.

Supports --api_mode batch|fast (default: batch).

Usage:
    uv run python scripts/validate_evaluator_gold.py --benchmark falseqa
    uv run python scripts/validate_evaluator_gold.py --benchmark jailbreak
    uv run python scripts/validate_evaluator_gold.py --benchmark jailbreak-cross-alpha --compare --output_dir results/judge_validation/
    uv run python scripts/validate_evaluator_gold.py --benchmark all
    uv run python scripts/validate_evaluator_gold.py --benchmark all --api_mode fast
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from evaluate_intervention import (
    FALSEQA_JUDGE_KWARGS,
    JAILBREAK_JUDGE_KWARGS,
    build_falseqa_judge_messages,
    build_jailbreak_judge_messages,
    judge_falseqa,
    judge_jailbreak,
    parse_falseqa_verdict,
    parse_jailbreak_verdict,
)

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
    "jailbreak-cross-alpha": {
        "fixture": "jailbreak_cross_alpha_gold.jsonl",
        "judge_fn": "jailbreak",
        "label_field": "human_label",
        "valid_labels": {"HARMFUL", "SAFE"},
        "skip_labels": {"BORDERLINE"},
        "group_by": "alpha",
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
    output_dir: Path | None = None,
) -> tuple[bool, list[dict]]:
    config = BENCHMARK_CONFIG[benchmark]
    records = load_gold(config["fixture"])
    skip_labels = config.get("skip_labels", set())
    group_by = config.get("group_by")

    labeled = [r for r in records if r.get(config["label_field"]) is not None]
    if not labeled:
        print(f"\n[{benchmark}] FAIL: No human-labeled cases found.")
        print(f"  ({len(records)} candidates await human review)")
        return False, []

    # Filter out skip labels
    skipped_count = 0
    evaluable = []
    for r in labeled:
        human = r[config["label_field"]]
        if human in skip_labels:
            skipped_count += 1
        elif human not in config["valid_labels"]:
            print(f"  WARNING: invalid human_label {human!r} for {r['id']}")
        else:
            evaluable.append(r)

    print(f"\n[{benchmark}] Validating {len(evaluable)} gold-labeled cases...")
    if skipped_count:
        print(f"  Skipped {skipped_count} records with labels in {skip_labels}")

    # Prepare output file
    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / f"judge_validation_{benchmark}_{ts}.jsonl"

    agree = 0
    disagree = 0
    errors = 0
    disagreements: list[dict] = []
    results: list[dict] = []

    for rec in evaluable:
        human = rec[config["label_field"]]

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

        result = {
            "id": rec["id"],
            "human_label": human,
            "judge_verdict": verdict,
            "agree": False,
            "skipped": False,
        }
        if group_by and group_by in rec:
            result[group_by] = rec[group_by]

        if verdict in ("ERROR", "UNKNOWN"):
            errors += 1
            disagree += 1
            print(f"  ERROR judging {rec['id']}: got {verdict}")
            disagreements.append({"id": rec["id"], "human": human, "judge": verdict})
        elif verdict == human:
            agree += 1
            result["agree"] = True
        else:
            disagree += 1
            disagreements.append({"id": rec["id"], "human": human, "judge": verdict})

        results.append(result)

        # Per-record write (reopen pattern per CLAUDE.md)
        if out_path:
            with open(out_path, "a") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")

    total = agree + disagree
    if total == 0:
        print("  No valid comparisons — all errored.")
        return False, results

    rate = agree / total
    print(f"  Agreement: {agree}/{total} ({rate:.1%})")
    if errors:
        print(f"  Errors (counted as failures): {errors}")

    if disagreements:
        print("  Disagreements:")
        for d in disagreements:
            print(f"    {d['id']}: human={d['human']}, judge={d['judge']}")

    # Per-group breakdown
    if group_by:
        groups: dict[str, list[dict]] = {}
        for r in results:
            key = r.get(group_by, "unknown")
            groups.setdefault(key, []).append(r)
        print(f"\n  Per-{group_by} breakdown:")
        for key in sorted(groups.keys()):
            grp = groups[key]
            g_agree = sum(1 for r in grp if r["agree"])
            g_total = len(grp)
            g_rate = g_agree / g_total if g_total else 0
            print(f"    {group_by}={key}: {g_agree}/{g_total} ({g_rate:.1%})")

    passed = rate >= threshold
    if passed:
        print(f"  PASS (>= {threshold:.0%} threshold)")
    else:
        print(f"  FAIL (< {threshold:.0%} threshold)")

    if out_path:
        print(f"  Results written to {out_path}")

    return passed, results


def validate_benchmark_batch(
    benchmark: str,
    client: OpenAI,
    judge_model: str,
    threshold: float,
    output_dir: Path | None = None,
) -> tuple[bool, list[dict]]:
    """Batch API version of validate_benchmark.

    Produces identical output fields (id, human_label, judge_verdict, agree)
    so results are interchangeable with the sync path.
    """
    from openai_batch import build_chat_request, parse_chat_content, resume_or_submit

    config = BENCHMARK_CONFIG[benchmark]
    records = load_gold(config["fixture"])
    skip_labels = config.get("skip_labels", set())
    group_by = config.get("group_by")

    labeled = [r for r in records if r.get(config["label_field"]) is not None]
    if not labeled:
        print(f"\n[{benchmark}] FAIL: No human-labeled cases found.")
        return False, []

    skipped_count = 0
    evaluable = []
    for r in labeled:
        human = r[config["label_field"]]
        if human in skip_labels:
            skipped_count += 1
        elif human not in config["valid_labels"]:
            print(f"  WARNING: invalid human_label {human!r} for {r['id']}")
        else:
            evaluable.append(r)

    print(f"\n[{benchmark}] Validating {len(evaluable)} gold-labeled cases (batch)...")
    if skipped_count:
        print(f"  Skipped {skipped_count} records with labels in {skip_labels}")

    if not evaluable:
        return False, []

    # Build batch requests
    batch_requests = []
    for idx, rec in enumerate(evaluable):
        custom_id = f"{benchmark}_{idx}"
        if config["judge_fn"] == "falseqa":
            messages = build_falseqa_judge_messages(rec["question"], rec["response"])
            kwargs = FALSEQA_JUDGE_KWARGS
        elif config["judge_fn"] == "jailbreak":
            messages = build_jailbreak_judge_messages(rec["question"], rec["response"])
            kwargs = JAILBREAK_JUDGE_KWARGS
        else:
            continue
        batch_requests.append(
            build_chat_request(custom_id, judge_model, messages, **kwargs)
        )

    state_dir = output_dir or Path("/tmp")
    state_path = state_dir / f".validate_{benchmark}_batch_state.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    batch_results = resume_or_submit(
        client,
        batch_requests,
        state_path,
        metadata={"benchmark": benchmark, "script": "validate_evaluator_gold"},
    )

    # Process results
    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / f"judge_validation_{benchmark}_{ts}.jsonl"

    agree = 0
    disagree = 0
    errors = 0
    disagreements: list[dict] = []
    results: list[dict] = []

    for idx, rec in enumerate(evaluable):
        human = rec[config["label_field"]]
        custom_id = f"{benchmark}_{idx}"
        entry = batch_results.get(custom_id)
        content = parse_chat_content(entry) if entry else None

        if content is None:
            verdict = "ERROR"
        elif config["judge_fn"] == "falseqa":
            verdict = parse_falseqa_verdict(content)
        else:
            verdict = parse_jailbreak_verdict(content)

        result = {
            "id": rec["id"],
            "human_label": human,
            "judge_verdict": verdict,
            "agree": False,
            "skipped": False,
        }
        if group_by and group_by in rec:
            result[group_by] = rec[group_by]

        if verdict in ("ERROR", "UNKNOWN"):
            errors += 1
            disagree += 1
            disagreements.append({"id": rec["id"], "human": human, "judge": verdict})
        elif verdict == human:
            agree += 1
            result["agree"] = True
        else:
            disagree += 1
            disagreements.append({"id": rec["id"], "human": human, "judge": verdict})

        results.append(result)

        if out_path:
            with open(out_path, "a") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")

    total = agree + disagree
    if total == 0:
        print("  No valid comparisons — all errored.")
        return False, results

    rate = agree / total
    print(f"  Agreement: {agree}/{total} ({rate:.1%})")
    if errors:
        print(f"  Errors (counted as failures): {errors}")

    if disagreements:
        print("  Disagreements:")
        for d in disagreements:
            print(f"    {d['id']}: human={d['human']}, judge={d['judge']}")

    if group_by:
        groups: dict[str, list[dict]] = {}
        for r in results:
            key = r.get(group_by, "unknown")
            groups.setdefault(key, []).append(r)
        print(f"\n  Per-{group_by} breakdown:")
        for key in sorted(groups.keys()):
            grp = groups[key]
            g_agree = sum(1 for r in grp if r["agree"])
            g_total = len(grp)
            g_rate = g_agree / g_total if g_total else 0
            print(f"    {group_by}={key}: {g_agree}/{g_total} ({g_rate:.1%})")

    passed = rate >= threshold
    if passed:
        print(f"  PASS (>= {threshold:.0%} threshold)")
    else:
        print(f"  FAIL (< {threshold:.0%} threshold)")

    if out_path:
        print(f"  Results written to {out_path}")

    return passed, results


def write_comparison(
    results: list[dict],
    fixture_records: list[dict],
    output_dir: Path,
) -> None:
    """Write comparison table: old truncated judge vs new full-length judge vs human.

    Only meaningful for α=0.0 records that have judge_label_at_creation.
    """
    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = output_dir / f"judge_comparison_{ts}.md"

    # Build lookup: (id, alpha) -> result
    result_by_key = {(r["id"], r.get("alpha")): r for r in results}

    lines = [
        "# Judge Comparison: Truncated (old) vs Full-Length (new) vs Human",
        "",
        f"Generated: {ts}",
        "",
        "## α=0.0 Records (old truncated judge available)",
        "",
        "| ID | Human | Judge (new, full-length) | Judge (old, truncated) | New agrees | Old agrees |",
        "|---|---|---|---|---|---|",
    ]

    new_agree_count = 0
    old_agree_count = 0
    total = 0

    for fix_rec in fixture_records:
        if fix_rec["alpha"] != 0.0:
            continue
        sample_id = fix_rec["id"]
        old_judge = fix_rec.get("judge_label_at_creation")
        if old_judge is None:
            continue

        result = result_by_key.get((sample_id, 0.0))
        if result is None:
            continue

        human = result["human_label"]
        new_judge = result["judge_verdict"]
        new_ok = "yes" if new_judge == human else "**NO**"
        old_ok = "yes" if old_judge == human else "**NO**"

        if new_judge == human:
            new_agree_count += 1
        if old_judge == human:
            old_agree_count += 1
        total += 1

        lines.append(
            f"| `{sample_id}` | {human} | {new_judge} | {old_judge} | {new_ok} | {old_ok} |"
        )

    lines.append("")
    if total > 0:
        lines.append(
            f"**New (full-length) agreement:** {new_agree_count}/{total} ({new_agree_count / total:.1%})"
        )
        lines.append(
            f"**Old (truncated) agreement:** {old_agree_count}/{total} ({old_agree_count / total:.1%})"
        )
    lines.append("")

    # Also include per-alpha summary from all results
    lines.append("## All Alphas — Per-Alpha Agreement Summary")
    lines.append("")
    lines.append("| Alpha | Agree | Total | Rate |")
    lines.append("|---|---|---|---|")

    alpha_groups: dict[float, list[dict]] = {}
    for r in results:
        a = r.get("alpha", 0.0)
        alpha_groups.setdefault(a, []).append(r)
    for alpha in sorted(alpha_groups.keys()):
        grp = alpha_groups[alpha]
        a = sum(1 for r in grp if r["agree"])
        t = len(grp)
        lines.append(f"| {alpha} | {a} | {t} | {a / t:.1%} |")

    lines.append("")

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    print(f"\n  Comparison written to {out_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--benchmark",
        choices=["falseqa", "jailbreak", "jailbreak-cross-alpha", "all"],
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
    p.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Write per-record results to timestamped JSONL in this directory",
    )
    p.add_argument(
        "--compare",
        action="store_true",
        help="Print comparison table: old truncated judge vs new full-length judge vs human",
    )
    p.add_argument(
        "--api-mode",
        type=str,
        choices=["batch", "fast"],
        default="batch",
        help="API mode: 'batch' for 50%% cheaper async Batch API (default), "
        "'fast' for synchronous per-request calls",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OpenAI API key required. Set OPENAI_API_KEY or --api_key")
        sys.exit(1)

    client = OpenAI(api_key=api_key)

    benchmarks = (
        list(BENCHMARK_CONFIG.keys()) if args.benchmark == "all" else [args.benchmark]
    )

    validate_fn = (
        validate_benchmark_batch if args.api_mode == "batch" else validate_benchmark
    )
    print(f"API mode: {args.api_mode}")

    all_passed = True
    for bench in benchmarks:
        passed, results = validate_fn(
            bench, client, args.judge_model, args.threshold, args.output_dir
        )
        if not passed:
            all_passed = False

        # Comparison mode for cross-alpha
        if args.compare and bench == "jailbreak-cross-alpha" and args.output_dir:
            fixture_records = load_gold(BENCHMARK_CONFIG[bench]["fixture"])
            write_comparison(results, fixture_records, args.output_dir)

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
