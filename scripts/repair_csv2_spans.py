"""One-shot repair: re-evaluate explicit CSV2 error rows in an eval directory.

Identifies records whose existing ``csv2`` payload contains an explicit
``error`` field, strips the failed annotation, re-runs the judge, and writes
the repaired records back in place.

Usage:
    uv run python scripts/repair_csv2_spans.py
    uv run python scripts/repair_csv2_spans.py --dry_run
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from evaluate_csv2 import (
    annotate_record,
    judge_csv2_sync,
)

load_dotenv()

EVAL_DIR = Path("data/gemma3_4b/intervention/jailbreak/csv2_evaluation")
ALPHAS = [0.0, 1.5, 3.0]


def find_error_records(
    records: list[dict],
) -> list[int]:
    """Return indices of records with an explicit CSV2 error payload."""
    broken: list[int] = []
    for idx, rec in enumerate(records):
        csv2 = rec.get("csv2", {})
        if isinstance(csv2, dict) and csv2.get("error"):
            broken.append(idx)
    return broken


def load_alpha(eval_dir: Path, alpha: float) -> list[dict]:
    path = eval_dir / f"alpha_{alpha:.1f}.jsonl"
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_alpha(eval_dir: Path, alpha: float, records: list[dict]) -> None:
    path = eval_dir / f"alpha_{alpha:.1f}.jsonl"
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def attempt_repair(
    rec: dict,
    *,
    client: OpenAI,
    judge_model: str,
) -> tuple[str, dict]:
    """Repair one record without discarding the existing payload on failure."""
    old_csv2 = rec["csv2"]
    goal = rec.get("goal", rec.get("question", ""))
    category = rec.get("category", "")
    raw = judge_csv2_sync(client, judge_model, goal, rec["response"], category)

    if raw is None:
        return "api_failed", old_csv2

    repaired_rec = dict(rec)
    repaired_rec.pop("csv2", None)
    annotate_record(repaired_rec, raw)
    new_csv2 = repaired_rec["csv2"]
    if new_csv2.get("error"):
        return "retry_error", new_csv2

    rec["csv2"] = new_csv2
    return "repaired", new_csv2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval_dir", type=str, default=str(EVAL_DIR))
    parser.add_argument("--alphas", type=float, nargs="+", default=ALPHAS)
    parser.add_argument("--judge_model", type=str, default="gpt-4o")
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="List affected records without calling the API",
    )
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key and not args.dry_run:
        raise ValueError("OPENAI_API_KEY required (or use --dry_run)")

    total_repaired = 0

    for alpha in args.alphas:
        records = load_alpha(eval_dir, alpha)
        broken_indices = find_error_records(records)

        if not broken_indices:
            print(f"alpha={alpha:.1f}: 0 explicit error rows - skipping")
            continue

        print(f"alpha={alpha:.1f}: {len(broken_indices)} explicit error rows")
        for idx in broken_indices:
            rec = records[idx]
            old_csv2 = rec["csv2"]
            print(
                f"  [{idx}] {rec['id']}: "
                f"error={old_csv2.get('error')} "
                f"C={old_csv2.get('C')} S={old_csv2.get('S')} "
                f"spans={len(old_csv2.get('harmful_spans', []))}"
            )

        if args.dry_run:
            continue

        client = OpenAI(api_key=api_key)
        repaired = 0

        for idx in broken_indices:
            rec = records[idx]
            old_csv2 = rec["csv2"]
            old_c = old_csv2.get("C")
            old_s = old_csv2.get("S")
            old_hb = old_csv2.get("harmful_binary")

            status, result_csv2 = attempt_repair(
                rec,
                client=client,
                judge_model=args.judge_model,
            )

            if status == "api_failed":
                print(f"  [{idx}] {rec['id']}: API FAILED")
                continue

            if status == "retry_error":
                print(
                    f"  [{idx}] {rec['id']}: "
                    f"C={old_c}->{result_csv2.get('C')} "
                    f"S={old_s}->{result_csv2.get('S')} "
                    f"hb={old_hb}->{result_csv2.get('harmful_binary')} "
                    f"error={result_csv2.get('error')} "
                    "(kept original payload)"
                )
                continue

            new_csv2 = result_csv2
            valid_spans = sum(
                1 for s in new_csv2.get("harmful_spans", []) if s.get("valid")
            )
            print(
                f"  [{idx}] {rec['id']}: "
                f"C={old_c}->{new_csv2['C']} "
                f"S={old_s}->{new_csv2['S']} "
                f"hb={old_hb}->{new_csv2['harmful_binary']} "
                f"valid_spans={valid_spans} "
                f"payload_share={new_csv2.get('harmful_payload_share', 0):.3f} "
                f"pivot={new_csv2.get('pivot_position')}"
            )
            repaired += 1

        write_alpha(eval_dir, alpha, records)
        print(
            f"  -> Wrote {repaired} repaired records to {eval_dir}/alpha_{alpha:.1f}.jsonl"
        )
        total_repaired += repaired

    print(f"\nTotal repaired: {total_repaired}")


if __name__ == "__main__":
    main()
