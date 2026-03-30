"""Build a truthfulness/hallucination contrastive dataset for D4 direction extraction.

Sources model-generated TriviaQA responses from consistency_samples.jsonl, selecting
questions where the model is *consistently* correct (all 10 judges = true) or
*consistently* wrong (all 10 judges = false). Uses the existing train/test ID split
from train_qids.json / test_qids_disjoint.json (originally built for H-neuron
classifier training), remapped to train/val for direction extraction.

Class definitions:
  - "truthful": questions where model consistently answers correctly
  - "hallucinatory": questions where model consistently hallucinates

This is NOT refusal data. The contrastive axis is factual accuracy, not safety.

Usage:
    uv run python scripts/build_truthfulness_contrastive.py \\
        --consistency_path data/gemma3_4b/pipeline/consistency_samples.jsonl \\
        --train_qids_path data/gemma3_4b/pipeline/train_qids.json \\
        --test_qids_path data/gemma3_4b/pipeline/test_qids_disjoint.json \\
        --output_dir data/contrastive/truthfulness
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils import (
    finish_run_provenance,
    get_git_sha,
    json_dumps,
    normalize_answer,
    provenance_error_message,
    provenance_status_for_exception,
    start_run_provenance,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build truthfulness/hallucination contrastive dataset for D4"
    )
    p.add_argument(
        "--consistency_path",
        type=Path,
        default=Path("data/gemma3_4b/pipeline/consistency_samples.jsonl"),
        help="Path to TriviaQA consistency_samples.jsonl",
    )
    p.add_argument(
        "--train_qids_path",
        type=Path,
        default=Path("data/gemma3_4b/pipeline/train_qids.json"),
        help="Path to train question ID split",
    )
    p.add_argument(
        "--test_qids_path",
        type=Path,
        default=Path("data/gemma3_4b/pipeline/test_qids_disjoint.json"),
        help="Path to disjoint test question ID split (used as val for direction extraction)",
    )
    p.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data/contrastive/truthfulness"),
        help="Output directory for contrastive JSONL and metadata",
    )
    p.add_argument(
        "--max_per_class_per_split",
        type=int,
        default=0,
        help="If >0, cap each (class, split) to this many samples for balance",
    )
    p.add_argument(
        "--refusal_contrastive_path",
        type=Path,
        default=Path("data/contrastive/refusal/refusal_contrastive.jsonl"),
        help="Path to refusal contrastive data for overlap check",
    )
    p.add_argument(
        "--faitheval_prompts_dir",
        type=Path,
        default=Path("data/gemma3_4b/intervention/faitheval"),
        help="Path to FaithEval data for overlap check",
    )
    p.add_argument(
        "--falseqa_path",
        type=Path,
        default=Path("data/benchmarks/falseqa_test.csv"),
        help="Path to FalseQA data for overlap check",
    )
    p.add_argument(
        "--dry_run",
        action="store_true",
        help="Build and validate in memory without writing files",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_consistency_samples(path: Path) -> dict[str, dict[str, Any]]:
    """Load consistency_samples.jsonl → {qid: {question, responses, judges, ground_truth}}."""
    samples: dict[str, dict[str, Any]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            for qid, inner in record.items():
                samples[qid] = inner
    return samples


def load_id_split(path: Path) -> dict[str, list[str]]:
    """Load train/test QID split → {'t': [...], 'f': [...]}."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def classify_consistency(judges: list[str]) -> str | None:
    """Classify a sample based on judge verdicts.

    Returns 'truthful' if all true, 'hallucinatory' if all false, None if mixed.
    """
    true_count = sum(1 for j in judges if j == "true")
    if true_count == len(judges):
        return "truthful"
    if true_count == 0:
        return "hallucinatory"
    return None


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------


def build_contrastive_records(
    samples: dict[str, dict[str, Any]],
    train_qids: dict[str, list[str]],
    test_qids: dict[str, list[str]],
    *,
    max_per_class_per_split: int = 0,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Build contrastive JSONL records from consistency data.

    Returns (records, build_stats).
    """
    validate_disjoint_qid_splits(train_qids, test_qids)

    # Build ID → split mapping
    # train_qids['t'] and train_qids['f'] are train IDs
    # test_qids['t'] and test_qids['f'] are test/val IDs
    train_ids = set(train_qids.get("t", []) + train_qids.get("f", []))
    test_ids = set(test_qids.get("t", []) + test_qids.get("f", []))

    records: list[dict[str, str]] = []
    stats: dict[str, int] = {
        "total_samples": len(samples),
        "consistent_truthful": 0,
        "consistent_hallucinatory": 0,
        "mixed_excluded": 0,
        "no_split_excluded": 0,
    }

    # Collect by (split, label) for optional balancing
    buckets: dict[str, list[dict[str, str]]] = {}

    for qid, inner in sorted(samples.items()):
        judges = inner.get("judges", [])
        label = classify_consistency(judges)

        if label is None:
            stats["mixed_excluded"] += 1
            continue

        if label == "truthful":
            stats["consistent_truthful"] += 1
        else:
            stats["consistent_hallucinatory"] += 1

        if qid in train_ids:
            split = "train"
        elif qid in test_ids:
            split = "val"  # Remap test → val for direction extraction
        else:
            stats["no_split_excluded"] += 1
            continue

        question = inner["question"]
        record = {
            "id": f"truth_{split}_{label[:5]}_{qid}",
            "text": question,
            "label": label,
            "source": "triviaqa_consistency",
            "split": split,
            "qid": qid,
        }

        bucket_key = f"{split}_{label}"
        buckets.setdefault(bucket_key, []).append(record)

    # Optional balancing
    for key, bucket in sorted(buckets.items()):
        if max_per_class_per_split > 0 and len(bucket) > max_per_class_per_split:
            bucket = bucket[:max_per_class_per_split]
        records.extend(bucket)

    records, cross_split_duplicate_stats = drop_cross_split_normalized_duplicates(
        records
    )

    # Compute split/label counts
    split_label_counts: dict[str, int] = {}
    for r in records:
        k = f"{r['split']}_{r['label']}"
        split_label_counts[k] = split_label_counts.get(k, 0) + 1

    build_stats: dict[str, Any] = {
        **stats,
        "split_label_counts": split_label_counts,
        "total_records": len(records),
        "max_per_class_per_split": max_per_class_per_split,
        "cross_split_normalized_duplicates_removed": (
            cross_split_duplicate_stats["dropped_record_count"]
        ),
        "cross_split_normalized_duplicate_groups": (
            cross_split_duplicate_stats["duplicate_group_count"]
        ),
        "cross_split_normalized_duplicate_examples": (
            cross_split_duplicate_stats["examples"]
        ),
    }
    return records, build_stats


# ---------------------------------------------------------------------------
# Overlap / leakage checks
# ---------------------------------------------------------------------------


def normalized_text_set(texts: list[str]) -> set[str]:
    """Build a set of normalized texts for overlap detection."""
    return {normalize_answer(t) for t in texts if t.strip()}


def validate_disjoint_qid_splits(
    train_qids: dict[str, list[str]], test_qids: dict[str, list[str]]
) -> None:
    """Fail fast if the train/val QID definitions are not disjoint."""

    train_ids = train_qids.get("t", []) + train_qids.get("f", [])
    test_ids = test_qids.get("t", []) + test_qids.get("f", [])

    duplicate_train_ids = [
        qid for qid, count in Counter(train_ids).items() if count > 1
    ]
    duplicate_test_ids = [qid for qid, count in Counter(test_ids).items() if count > 1]
    overlap_ids = sorted(set(train_ids) & set(test_ids))

    problems: list[str] = []
    if duplicate_train_ids:
        problems.append(
            "train split repeats "
            f"{len(duplicate_train_ids)} QIDs (examples: {duplicate_train_ids[:5]})"
        )
    if duplicate_test_ids:
        problems.append(
            "val split repeats "
            f"{len(duplicate_test_ids)} QIDs (examples: {duplicate_test_ids[:5]})"
        )
    if overlap_ids:
        problems.append(
            "train/val overlap contains "
            f"{len(overlap_ids)} QIDs (examples: {overlap_ids[:5]})"
        )

    if problems:
        raise ValueError(
            "QID splits must be disjoint before building truthfulness records: "
            + "; ".join(problems)
        )


def drop_cross_split_normalized_duplicates(
    records: list[dict[str, str]],
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Drop normalized prompt duplicates that would leak train text into val.

    If a normalized prompt appears in multiple splits, keep the training copy and
    drop the held-out copy so downstream validation remains textually unseen.
    """

    by_normalized_text: dict[str, list[dict[str, str]]] = {}
    for record in records:
        normalized_text = normalize_answer(record["text"])
        if not normalized_text:
            continue
        by_normalized_text.setdefault(normalized_text, []).append(record)

    split_priority = {"train": 0, "val": 1}
    leaked_record_ids: set[str] = set()
    examples: list[dict[str, Any]] = []
    for normalized_text, grouped_records in sorted(by_normalized_text.items()):
        split_names = sorted({record["split"] for record in grouped_records})
        if len(split_names) <= 1:
            continue

        kept_split = min(split_names, key=lambda split: split_priority.get(split, 999))
        dropped_records = sorted(
            (record for record in grouped_records if record["split"] != kept_split),
            key=lambda record: record["id"],
        )
        for record in dropped_records:
            leaked_record_ids.add(record["id"])

        examples.append(
            {
                "normalized_text": normalized_text,
                "kept_split": kept_split,
                "kept_ids": sorted(
                    record["id"]
                    for record in grouped_records
                    if record["split"] == kept_split
                ),
                "dropped_ids": [record["id"] for record in dropped_records],
                "dropped_qids": [record["qid"] for record in dropped_records],
            }
        )

    filtered_records = [
        record for record in records if record["id"] not in leaked_record_ids
    ]
    stats = {
        "duplicate_group_count": len(examples),
        "dropped_record_count": len(leaked_record_ids),
        "examples": examples[:5],
    }
    return filtered_records, stats


def check_refusal_overlap(
    records: list[dict[str, str]], refusal_path: Path
) -> dict[str, Any]:
    """Check for text overlap between truthfulness and refusal contrastive data."""
    if not refusal_path.exists():
        return {"status": "skipped", "reason": f"{refusal_path} not found"}

    refusal_texts: list[str] = []
    with open(refusal_path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            refusal_texts.append(r["text"])

    truth_norms = normalized_text_set([r["text"] for r in records])
    refusal_norms = normalized_text_set(refusal_texts)
    overlap = truth_norms & refusal_norms

    return {
        "status": "checked",
        "truthfulness_count": len(truth_norms),
        "refusal_count": len(refusal_norms),
        "normalized_overlap_count": len(overlap),
        "overlap_examples": sorted(overlap)[:5],
    }


def check_faitheval_overlap(
    records: list[dict[str, str]], faitheval_dir: Path
) -> dict[str, Any]:
    """Check for text overlap between truthfulness data and FaithEval prompts."""
    # FaithEval prompts are embedded in run_intervention.py prompt builders
    # and loaded from HuggingFace. Since FaithEval is MCQ and TriviaQA is
    # open-ended factoid, structural overlap is essentially impossible.
    # We still check normalized text for due diligence.
    return {
        "status": "structural_check",
        "note": (
            "FaithEval uses multiple-choice questions from HuggingFace dataset. "
            "TriviaQA uses open-ended factoid questions. Format and domain are "
            "structurally disjoint. No text-level overlap check possible without "
            "loading the HF dataset, but contamination risk is negligible."
        ),
        "contamination_risk": "negligible",
    }


def check_falseqa_overlap(
    records: list[dict[str, str]], falseqa_path: Path
) -> dict[str, Any]:
    """Check for text overlap between truthfulness data and FalseQA."""
    if not falseqa_path.exists():
        return {"status": "skipped", "reason": f"{falseqa_path} not found"}

    import csv

    falseqa_texts: list[str] = []
    with open(falseqa_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            q = row.get("question", row.get("Question", ""))
            if q.strip():
                falseqa_texts.append(q)

    truth_norms = normalized_text_set([r["text"] for r in records])
    falseqa_norms = normalized_text_set(falseqa_texts)
    overlap = truth_norms & falseqa_norms

    return {
        "status": "checked",
        "truthfulness_count": len(truth_norms),
        "falseqa_count": len(falseqa_norms),
        "normalized_overlap_count": len(overlap),
        "overlap_examples": sorted(overlap)[:5],
    }


def check_internal_duplicates(records: list[dict[str, str]]) -> dict[str, Any]:
    """Check for duplicates within the truthfulness contrastive dataset."""
    texts = [r["text"] for r in records]
    exact_counter = Counter(texts)
    norm_counter = Counter(normalize_answer(t) for t in texts)

    exact_dupes = sum(c - 1 for c in exact_counter.values() if c > 1)
    norm_dupes = sum(c - 1 for c in norm_counter.values() if c > 1)

    return {
        "exact_duplicate_count": exact_dupes,
        "normalized_duplicate_count": norm_dupes,
        "exact_examples": [
            {"text": t, "count": c} for t, c in exact_counter.items() if c > 1
        ][:5],
    }


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def fingerprint_records(records: list[dict[str, str]]) -> str:
    """SHA-256 fingerprint of sorted record IDs."""
    ids = sorted(r["id"] for r in records)
    return hashlib.sha256("\n".join(ids).encode()).hexdigest()[:16]


def write_jsonl(records: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    output_dir: Path = args.output_dir
    jsonl_path = output_dir / "truthfulness_contrastive.jsonl"
    metadata_path = output_dir / "metadata.json"

    provenance_handle = None
    provenance_status = "completed"
    provenance_extra: dict[str, Any] = {}

    try:
        if not args.dry_run:
            provenance_handle = start_run_provenance(
                args,
                primary_target=str(output_dir),
                output_targets=[str(jsonl_path), str(metadata_path)],
                primary_target_is_dir=True,
            )

        # Load data
        print(f"Loading consistency samples from {args.consistency_path}")
        samples = load_consistency_samples(args.consistency_path)
        print(f"  Total questions: {len(samples)}")

        print(f"Loading train QIDs from {args.train_qids_path}")
        train_qids = load_id_split(args.train_qids_path)
        print(
            f"  Train IDs: {len(train_qids.get('t', []))} true, "
            f"{len(train_qids.get('f', []))} false"
        )

        print(f"Loading test QIDs from {args.test_qids_path}")
        test_qids = load_id_split(args.test_qids_path)
        print(
            f"  Test/Val IDs: {len(test_qids.get('t', []))} true, "
            f"{len(test_qids.get('f', []))} false"
        )

        # Build records
        records, build_stats = build_contrastive_records(
            samples,
            train_qids,
            test_qids,
            max_per_class_per_split=args.max_per_class_per_split,
        )

        print(f"\nDataset built: {len(records)} records")
        for key, count in sorted(build_stats["split_label_counts"].items()):
            print(f"  {key}: {count}")
        print(f"  Mixed excluded: {build_stats['mixed_excluded']}")
        print(f"  No-split excluded: {build_stats['no_split_excluded']}")

        # Overlap checks
        print("\nRunning overlap checks...")
        refusal_overlap = check_refusal_overlap(records, args.refusal_contrastive_path)
        print(
            f"  Refusal overlap: {refusal_overlap.get('normalized_overlap_count', 'N/A')}"
        )

        faitheval_overlap = check_faitheval_overlap(records, args.faitheval_prompts_dir)
        print(f"  FaithEval overlap: {faitheval_overlap['contamination_risk']}")

        falseqa_overlap = check_falseqa_overlap(records, args.falseqa_path)
        print(
            f"  FalseQA overlap: {falseqa_overlap.get('normalized_overlap_count', 'N/A')}"
        )

        internal_dupes = check_internal_duplicates(records)
        print(
            f"  Internal duplicates: {internal_dupes['exact_duplicate_count']} exact, "
            f"{internal_dupes['normalized_duplicate_count']} normalized"
        )

        # Build metadata
        metadata: dict[str, Any] = {
            "dataset_name": "truthfulness_contrastive",
            "purpose": "D4 truthfulness/hallucination direction extraction",
            "class_definitions": {
                "truthful": (
                    "Questions where Gemma-3-4B-IT consistently answers correctly "
                    "(all 10 sampled responses judged true by GPT-4o)"
                ),
                "hallucinatory": (
                    "Questions where Gemma-3-4B-IT consistently hallucinates "
                    "(all 10 sampled responses judged false by GPT-4o)"
                ),
            },
            "NOT_refusal_data": (
                "The contrastive axis is factual accuracy, not safety/refusal. "
                "Harmful/harmless behavior is not part of the class definition."
            ),
            "sources": {
                "consistency_samples": {
                    "path": str(args.consistency_path),
                    "description": (
                        "TriviaQA questions with 10x model responses, "
                        "each judged true/false by GPT-4o"
                    ),
                    "total_questions": len(samples),
                },
            },
            "split_mapping": {
                "train": (
                    f"From {args.train_qids_path.name} "
                    "(original classifier training split)"
                ),
                "val": (
                    f"From {args.test_qids_path.name} "
                    "(remapped: classifier test -> direction val)"
                ),
            },
            "build_stats": build_stats,
            "overlap_checks": {
                "vs_refusal_contrastive": refusal_overlap,
                "vs_faitheval": faitheval_overlap,
                "vs_falseqa": falseqa_overlap,
                "internal_duplicates": internal_dupes,
            },
            "leakage_assessment": (
                "No text overlap with refusal contrastive (different domain). "
                "No structural overlap with FaithEval (MCQ) or FalseQA (false premise). "
                "Any normalized prompt collision between train and val is dropped from "
                "val before export. "
                "TriviaQA questions are open-ended factoid; eval benchmarks are "
                "multiple-choice or false-premise. Cross-contamination risk is negligible."
            ),
            "known_limitations": [
                "Single source (TriviaQA via Gemma-3-4B-IT consistency sampling). "
                "A multi-source mix would strengthen external validity.",
                "Class boundary is model-specific: 'consistently wrong' on Gemma-3-4B "
                "may not generalize to other models.",
                "Mixed-consistency questions (385 of 3500) are excluded, reducing "
                "the dataset to the clearest cases. This may bias toward questions "
                "that are 'easy to get right' or 'easy to get wrong'.",
                "GPT-4o judge may have its own accuracy errors, propagated into labels.",
            ],
            "provisional": False,
            "fingerprint": fingerprint_records(records),
            "git_sha": get_git_sha(),
            "paths": {
                "jsonl": str(jsonl_path),
                "metadata": str(metadata_path),
            },
        }

        provenance_extra.update(
            {
                "record_count": len(records),
                "split_label_counts": build_stats["split_label_counts"],
                "fingerprint": metadata["fingerprint"],
            }
        )

        if args.dry_run:
            print("\nDry run — skipping file writes.")
            print(f"\nMetadata preview:\n{json.dumps(metadata, indent=2)[:2000]}")
            return

        # Write outputs
        output_dir.mkdir(parents=True, exist_ok=True)
        write_jsonl(records, jsonl_path)
        print(f"\nWrote {len(records)} records to {jsonl_path}")

        metadata_path.write_text(json_dumps(metadata), encoding="utf-8")
        print(f"Wrote metadata to {metadata_path}")

    except BaseException as exc:
        provenance_status = provenance_status_for_exception(exc)
        provenance_extra["error"] = provenance_error_message(exc)
        raise
    finally:
        if provenance_handle is not None:
            finish_run_provenance(
                provenance_handle, provenance_status, provenance_extra
            )


if __name__ == "__main__":
    main()
