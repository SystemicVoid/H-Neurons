#!/usr/bin/env python3
"""LLM rater B for bridge inter-rater reliability.

Labels blinded bridge discordant cases against the
``bridge_incorrect_response_v1`` rubric using GPT-4o via the OpenAI Batch API
(crash-safe, 50% cheaper than sync). Rater B is a sensitivity check, not a
strong-form second human rater; it runs zero-shot on the rubric, without
few-shot exemplars from the dev calibration queue, so it remains independent
of Rater A.

Usage:
    # Step 1: calibrate on 13 dev cases (separate progress file; not counted
    # toward reported IRR):
    uv run python scripts/bridge_irr_rater_b.py --split dev

    # Step 2: after dev inspection and any prompt revision, freeze adjudication
    # rule and run on 57 test cases:
    uv run python scripts/bridge_irr_rater_b.py --split test

    # Pin to a different snapshot (default: gpt-4o-2024-11-20):
    uv run python scripts/bridge_irr_rater_b.py --split test \\
        --model gpt-4o-2024-11-20

Per scripts/CLAUDE.md:
- Batch mode only, never fast.
- Run ``scripts/infra/check_openai_batch_limits_via_codex.sh`` before submitting.
- JSONL writes reopen file per record.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openai import OpenAI

from bridge_irr import (
    CONFIDENCE_LEVELS,
    DEFAULT_OUTPUT_DIR,
    LABELS,
    RUBRIC_VERSION,
    load_jsonl,
    load_label_progress,
)
from openai_batch import (
    build_chat_request,
    parse_chat_content,
    resume_or_submit,
)

DEFAULT_MODEL = "gpt-4o-2024-11-20"

SYSTEM_PROMPT = """\
You are labeling a model's incorrect response to a TriviaQA trivia question using a 4-category failure taxonomy. You are Rater B in an inter-rater reliability study; your labels are combined with a human rater's labels to measure coding agreement.

CATEGORIES:
- wrong_entity_substitution: the response names exactly one specific wrong entity as the answer. Hedging words ("probably", "I think", "most likely") do not disqualify this label if one entity is committed.
- evasion_or_factual_denial: the response denies the answer's existence, knowability, or fit, and does not name a specific entity as the answer.
- answer_dilution: the response lists two or more plausible candidates without committing to one, OR pads the answer slot with tangential context that provides no specific answer.
- formal_refusal: explicit refusal to answer, typically with safety or policy framing.

DECISION RULES (apply the first whose precondition fits):
R1. Explicit refusal language ("I can't", "I won't", "I'm not able to") and no answer attempt -> formal_refusal.
R2. Response commits to exactly one specific entity as the answer, even with hedging -> wrong_entity_substitution. Operational test: a reader completes "The model said the answer is ___" with one entity.
R3. Response lists two or more plausible candidates without picking one -> answer_dilution.
R4. Response denies the answer's existence, knowability, or unique fit and names no entity -> evasion_or_factual_denial.
R5. Commitment plus caveats or alternatives (one primary answer, others as alternatives) -> R2 wins, label wrong_entity_substitution.
R6. Tangential context about the subject without commitment or explicit denial -> answer_dilution.

TIEBREAKERS:
- R2 vs R3: if one entity is in primary answer position, apply R2; if entities are parallel with no primary, apply R3.
- R3 vs R4: prefer R3 when candidates are offered; prefer R4 when the response denies fit without offering candidates.
- R2 vs R4: prefer R2 if any specific entity is proposed, even hedged.

INSTRUCTIONS:
- Label ONLY the `incorrect_response` field.
- `gold_aliases` and `paired_correct_response` are disambiguation aids only (e.g., to distinguish substitution from dilution). Never use them to guess the label.
- In `notes`, cite the specific rule (R1-R6) and quote the phrase in `incorrect_response` that triggered it.
- `confidence` = high when the rule fit is unambiguous; medium when two rules compete but one clearly wins via tiebreakers; low when the case is genuinely borderline.

Output a single JSON object conforming to the schema. No prose outside the JSON.
"""

RESPONSE_JSON_SCHEMA: dict[str, Any] = {
    "name": "bridge_failure_label",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["label", "confidence", "notes"],
        "properties": {
            "label": {
                "type": "string",
                "enum": list(LABELS),
            },
            "confidence": {
                "type": "string",
                "enum": list(CONFIDENCE_LEVELS),
            },
            "notes": {
                "type": "string",
                "minLength": 1,
                "maxLength": 500,
            },
        },
    },
}


def build_user_content(queue_row: dict[str, Any]) -> str:
    """Serialize one queue row into a deterministic JSON payload."""
    payload = {
        "question": queue_row["question"],
        "gold_aliases": queue_row["gold_aliases"],
        "incorrect_response": queue_row["incorrect_response"],
        "paired_correct_response": queue_row["paired_correct_response"],
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2)


def build_requests(
    queue_rows: list[dict[str, Any]],
    *,
    model: str,
    existing_case_ids: set[str],
) -> list[dict[str, Any]]:
    """Build batch requests for queue rows not already labeled."""
    requests: list[dict[str, Any]] = []
    for row in queue_rows:
        case_id = str(row["case_id"])
        if case_id in existing_case_ids:
            continue
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_content(row)},
        ]
        requests.append(
            build_chat_request(
                custom_id=case_id,
                model=model,
                messages=messages,
                temperature=0,
                response_format={
                    "type": "json_schema",
                    "json_schema": RESPONSE_JSON_SCHEMA,
                },
            )
        )
    return requests


def validate_label_payload(payload: dict[str, Any], *, case_id: str) -> None:
    label = payload.get("label")
    confidence = payload.get("confidence")
    notes = payload.get("notes")
    if label not in LABELS:
        raise ValueError(f"{case_id}: invalid label {label!r}")
    if confidence not in CONFIDENCE_LEVELS:
        raise ValueError(f"{case_id}: invalid confidence {confidence!r}")
    if not isinstance(notes, str) or not notes.strip():
        raise ValueError(f"{case_id}: notes must be a non-empty string")


def append_label_row(path: Path, row: dict[str, Any]) -> None:
    """Append a labeled row. Reopens file per record per the JSONL rule."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def git_commit_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def prompt_hash() -> str:
    return hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest()[:16]


def write_provenance(
    path: Path,
    *,
    split: str,
    model: str,
    n_cases: int,
    n_newly_labeled: int,
    n_failures: int,
) -> None:
    provenance = {
        "rater": "rater_b_llm",
        "rater_kind": "llm_judge",
        "model": model,
        "rubric_version": RUBRIC_VERSION,
        "prompt_hash": prompt_hash(),
        "split": split,
        "n_cases": n_cases,
        "n_newly_labeled": n_newly_labeled,
        "n_failures": n_failures,
        "git_commit": git_commit_hash(),
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "decoding": {
            "temperature": 0,
            "response_format": "json_schema (strict)",
        },
        "batch_mode": True,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(provenance, indent=2) + "\n")


def split_paths(output_dir: Path, split: str) -> dict[str, Path]:
    if split == "dev":
        return {
            "queue": output_dir / "dev_calibration_queue.jsonl",
            "progress": output_dir / "rater_b_dev_progress.jsonl",
            "state": output_dir / ".rater_b_dev_batch_state.json",
            "provenance": output_dir / "rater_b_dev_provenance.json",
        }
    return {
        "queue": output_dir / "test_queue_blinded.jsonl",
        "progress": output_dir / "rater_b_progress.jsonl",
        "state": output_dir / ".rater_b_batch_state.json",
        "provenance": output_dir / "rater_b_provenance.json",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--split",
        choices=("dev", "test"),
        default="test",
        help="dev = calibration queue, test = paper IRR queue",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Pinned snapshot for reproducibility (default: %(default)s)",
    )
    parser.add_argument("--poll_interval", type=int, default=30)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = split_paths(args.output_dir, args.split)

    if not paths["queue"].exists():
        print(f"Queue not found: {paths['queue']}", file=sys.stderr)
        return 2

    queue_rows = load_jsonl(paths["queue"])
    existing = load_label_progress(paths["progress"])
    existing_case_ids = set(existing.keys())

    requests = build_requests(
        queue_rows, model=args.model, existing_case_ids=existing_case_ids
    )

    n_total = len(queue_rows)
    n_done = len(existing_case_ids)
    n_pending = len(requests)
    print(
        f"Bridge IRR rater B ({args.split}): "
        f"{n_done}/{n_total} done, {n_pending} pending "
        f"[model={args.model}, prompt_hash={prompt_hash()}]"
    )
    if n_pending == 0:
        print("Nothing to do.")
        return 0

    client = OpenAI()
    t0 = time.time()
    results = resume_or_submit(
        client,
        requests,
        state_path=paths["state"],
        metadata={
            "task": "bridge_irr_rater_b",
            "split": args.split,
            "rubric_version": RUBRIC_VERSION,
            "prompt_hash": prompt_hash(),
        },
        poll_interval=args.poll_interval,
    )
    elapsed = time.time() - t0
    print(f"Batch returned {len(results)} entries in {elapsed:.0f}s")

    newly_labeled = 0
    failures: list[tuple[str, str]] = []
    for request in requests:
        case_id = str(request["custom_id"])
        entry = results.get(case_id)
        if entry is None:
            failures.append((case_id, "no batch result returned"))
            continue
        content = parse_chat_content(entry)
        if content is None:
            response = entry.get("response") or {}
            body = response.get("body")
            failures.append((case_id, f"non-200 or malformed: {body!r}"))
            continue
        try:
            payload = json.loads(content)
        except json.JSONDecodeError as exc:
            failures.append((case_id, f"invalid JSON: {exc}"))
            continue
        try:
            validate_label_payload(payload, case_id=case_id)
        except ValueError as exc:
            failures.append((case_id, str(exc)))
            continue

        append_label_row(
            paths["progress"],
            {
                "case_id": case_id,
                "label": payload["label"],
                "confidence": payload["confidence"],
                "notes": payload["notes"].strip(),
                "rater": "rater_b_llm",
                "model": args.model,
                "prompt_hash": prompt_hash(),
                "rubric_version": RUBRIC_VERSION,
            },
        )
        newly_labeled += 1

    write_provenance(
        paths["provenance"],
        split=args.split,
        model=args.model,
        n_cases=n_total,
        n_newly_labeled=newly_labeled,
        n_failures=len(failures),
    )

    if failures:
        print(f"{len(failures)} case(s) failed:", file=sys.stderr)
        for case_id, reason in failures:
            print(f"  {case_id}: {reason}", file=sys.stderr)
        return 1

    print(
        f"Labeled {newly_labeled} new cases. "
        f"Progress: {paths['progress']} (total rows: {n_done + newly_labeled})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
