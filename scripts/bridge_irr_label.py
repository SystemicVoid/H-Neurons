#!/usr/bin/env python3
"""Interactive Rater A labeler and adjudicator for bridge IRR.

Rater A (human first author) labels blinded bridge discordant cases against the
``bridge_incorrect_response_v1`` rubric. Same tool also adjudicates A-vs-B
test-split disagreements per the frozen adjudication rule.

Resume-safe: skips cases already present in the target progress file.
Appends one JSONL row per case (reopen-per-record; see scripts/CLAUDE.md).

Usage:
    # Dev calibration (13 cases, informal):
    uv run python scripts/bridge_irr_label.py --split dev

    # Test split (57 cases, counts toward reported IRR):
    uv run python scripts/bridge_irr_label.py --split test

    # Adjudicate A-vs-B disagreements (after both raters finish test):
    uv run python scripts/bridge_irr_label.py --split test --mode adjudicate
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from bridge_irr import (
    CONFIDENCE_LEVELS,
    DEFAULT_OUTPUT_DIR,
    LABELS,
    RUBRIC_VERSION,
    load_adjudication_progress,
    load_jsonl,
    load_label_progress,
    validate_adjudication_row,
    validate_label_row,
)

LABEL_HELP = {
    "wrong_entity_substitution": "one specific entity named as the answer (R2)",
    "evasion_or_factual_denial": "denies existence/knowability, no entity (R4)",
    "answer_dilution": "multiple candidates or tangential padding (R3/R6)",
    "formal_refusal": "explicit refusal, safety/policy framing (R1)",
}

RULE_CARD = """\
DECISION RULES (first match wins):
  R1 formal refusal: explicit refusal language, no answer attempt
       -> formal_refusal
  R2 commit: exactly one specific entity as the answer (hedging OK)
       -> wrong_entity_substitution
  R3 multi-candidate: >=2 plausible alternatives, none picked
       -> answer_dilution
  R4 denial: denies existence/knowability/fit, no entity named
       -> evasion_or_factual_denial
  R5 commit + caveats/alternatives: primary answer + alts
       -> wrong_entity_substitution (R2 wins over R3/R4)
  R6 tangential context: subject info, no commit or denial
       -> answer_dilution

TIEBREAKERS:
  R2 vs R3: primary-position entity -> R2; parallel entities -> R3
  R3 vs R4: candidates offered -> R3; denies fit without candidates -> R4
  R2 vs R4: any specific entity proposed (hedged OK) -> R2
"""


def split_paths(output_dir: Path, split: str) -> dict[str, Path]:
    if split == "dev":
        return {
            "queue": output_dir / "dev_calibration_queue.jsonl",
            "rater_a": output_dir / "rater_a_dev_progress.jsonl",
            "rater_b": output_dir / "rater_b_dev_progress.jsonl",
            "adjudication": output_dir / "adjudication_dev_progress.jsonl",
        }
    return {
        "queue": output_dir / "test_queue_blinded.jsonl",
        "rater_a": output_dir / "rater_a_progress.jsonl",
        "rater_b": output_dir / "rater_b_progress.jsonl",
        "adjudication": output_dir / "adjudication_progress.jsonl",
    }


def append_jsonl_record(path: Path, row: dict[str, Any]) -> None:
    """Append one JSONL row, reopening per record."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def prompt_label(prompt_text: str) -> str | None:
    menu = "\n".join(
        f"  [{i + 1}] {label:<28}  {LABEL_HELP[label]}"
        for i, label in enumerate(LABELS)
    )
    while True:
        try:
            raw = input(
                f"\n{prompt_text}\n{menu}\n  [q] quit and save progress\n> "
            ).strip()
        except EOFError:
            return None
        if raw.lower() == "q":
            return None
        if raw.isdigit() and 1 <= int(raw) <= len(LABELS):
            return LABELS[int(raw) - 1]
        if raw in LABELS:
            return raw
        print(f"  invalid. pick 1-{len(LABELS)} or label name or 'q'.")


def prompt_confidence() -> str | None:
    menu = "  [1] low   [2] medium   [3] high"
    while True:
        try:
            raw = input(f"confidence?\n{menu}\n> ").strip()
        except EOFError:
            return None
        if raw.lower() == "q":
            return None
        if raw.isdigit() and 1 <= int(raw) <= len(CONFIDENCE_LEVELS):
            return CONFIDENCE_LEVELS[int(raw) - 1]
        if raw in CONFIDENCE_LEVELS:
            return raw
        print("  invalid. pick 1-3 or level name or 'q'.")


def prompt_notes(prompt_text: str = "notes (cite rule + phrase)") -> str:
    try:
        return input(f"{prompt_text}: ").strip()
    except EOFError:
        return ""


def prompt_yes_no(prompt_text: str, default: bool = False) -> bool:
    suffix = " [y/N]" if not default else " [Y/n]"
    while True:
        try:
            raw = input(f"{prompt_text}{suffix}: ").strip().lower()
        except EOFError:
            return default
        if raw == "":
            return default
        if raw in ("y", "yes"):
            return True
        if raw in ("n", "no"):
            return False
        print("  answer y or n.")


def render_case(case: dict[str, Any], header: str) -> None:
    print("\n" + "=" * 72)
    print(header)
    print("=" * 72)
    print(f"case_id: {case['case_id']}")
    print(f"question: {case['question']}")
    aliases = case.get("gold_aliases", [])
    if aliases:
        print(f"gold_aliases: {aliases}")
    print(f"\nincorrect_response:\n  {case['incorrect_response']}")
    print(f"\npaired_correct_response:\n  {case['paired_correct_response']}")


def render_rater_row(name: str, row: dict[str, Any]) -> None:
    label = row.get("label", "")
    conf = row.get("confidence", "")
    notes = row.get("notes", "")
    print(f"\n{name}: label={label}  confidence={conf}")
    if notes:
        print(f"  notes: {notes}")


def run_label_mode(
    *,
    queue: list[dict[str, Any]],
    progress_path: Path,
) -> int:
    existing = load_label_progress(progress_path)
    pending = [row for row in queue if row["case_id"] not in existing]
    n_total = len(queue)
    n_done = len(existing)
    n_pending = len(pending)
    print(
        f"Rater A labeling: {n_done}/{n_total} done, {n_pending} pending. "
        f"Writing to {progress_path}"
    )
    print(RULE_CARD)

    labeled_this_session = 0
    for idx, case in enumerate(pending, start=1):
        render_case(case, f"Case {n_done + idx}/{n_total} (session {idx}/{n_pending})")
        label = prompt_label("label?")
        if label is None:
            break
        confidence = prompt_confidence()
        if confidence is None:
            break
        notes = prompt_notes()
        row = {
            "case_id": case["case_id"],
            "label": label,
            "confidence": confidence,
            "notes": notes,
            "rater": "rater_a_human",
            "rubric_version": RUBRIC_VERSION,
        }
        validate_label_row(row)
        append_jsonl_record(progress_path, row)
        labeled_this_session += 1
        print(f"  -> saved {label} ({confidence})")

    remaining = n_pending - labeled_this_session
    print(
        f"\nSession complete. Labeled {labeled_this_session} new cases. "
        f"{remaining} pending, {n_done + labeled_this_session}/{n_total} total."
    )
    return 0


def run_adjudicate_mode(
    *,
    queue: list[dict[str, Any]],
    rater_a_path: Path,
    rater_b_path: Path,
    adjudication_path: Path,
) -> int:
    rater_a = load_label_progress(rater_a_path)
    rater_b = load_label_progress(rater_b_path)
    existing = load_adjudication_progress(adjudication_path)

    missing_a = [row["case_id"] for row in queue if row["case_id"] not in rater_a]
    missing_b = [row["case_id"] for row in queue if row["case_id"] not in rater_b]
    if missing_a or missing_b:
        print(
            "Both raters must finish labeling before adjudication. "
            f"missing_a={len(missing_a)} missing_b={len(missing_b)}",
            file=sys.stderr,
        )
        return 2

    disagreements = [
        row
        for row in queue
        if rater_a[row["case_id"]]["label"] != rater_b[row["case_id"]]["label"]
    ]
    pending = [row for row in disagreements if row["case_id"] not in existing]
    n_total = len(disagreements)
    n_done = len(existing)
    n_pending = len(pending)
    print(
        f"Adjudication: {n_done}/{n_total} disagreements resolved, "
        f"{n_pending} pending. Writing to {adjudication_path}"
    )
    print(RULE_CARD)

    resolved_this_session = 0
    for idx, case in enumerate(pending, start=1):
        case_id = case["case_id"]
        render_case(
            case, f"Disagreement {n_done + idx}/{n_total} (session {idx}/{n_pending})"
        )
        render_rater_row("Rater A (human)", rater_a[case_id])
        render_rater_row("Rater B (LLM)", rater_b[case_id])
        label = prompt_label("adjudicated label?")
        if label is None:
            break
        rule_gap = prompt_yes_no(
            "rule_gap? (set if rule R1-R6 did not cleanly cover this case)",
            default=False,
        )
        notes = prompt_notes("notes (cite rule + phrase; see §5 of rule doc)")
        if not notes:
            print("  notes required per §5 of adjudication_rule.md. Skipping.")
            continue
        row = {
            "case_id": case_id,
            "label": label,
            "notes": notes,
            "rule_gap": rule_gap,
        }
        validate_adjudication_row(row)
        append_jsonl_record(adjudication_path, row)
        resolved_this_session += 1
        gap_marker = " (rule_gap)" if rule_gap else ""
        print(f"  -> adjudicated {label}{gap_marker}")

    remaining = n_pending - resolved_this_session
    print(
        f"\nSession complete. Resolved {resolved_this_session} new disagreements. "
        f"{remaining} pending, "
        f"{n_done + resolved_this_session}/{n_total} total."
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--split", choices=("dev", "test"), required=True)
    parser.add_argument("--mode", choices=("label", "adjudicate"), default="label")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = split_paths(args.output_dir, args.split)
    if not paths["queue"].exists():
        print(f"Queue not found: {paths['queue']}", file=sys.stderr)
        return 2
    queue = load_jsonl(paths["queue"])

    if args.mode == "label":
        return run_label_mode(queue=queue, progress_path=paths["rater_a"])
    if args.mode == "adjudicate":
        if args.split != "test":
            print(
                "Adjudication is only defined for the test split per §5 of "
                "adjudication_rule.md.",
                file=sys.stderr,
            )
            return 2
        return run_adjudicate_mode(
            queue=queue,
            rater_a_path=paths["rater_a"],
            rater_b_path=paths["rater_b"],
            adjudication_path=paths["adjudication"],
        )
    return 2


if __name__ == "__main__":
    sys.exit(main())
