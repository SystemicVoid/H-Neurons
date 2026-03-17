import argparse
import json
import re
import string
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from datasets import Dataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert BioASQ Task B JSON into the parquet schema expected by collect_responses.py."
    )
    parser.add_argument(
        "--input_json",
        type=Path,
        required=True,
        help="Path to BioASQ training JSON (for example training13b.json).",
    )
    parser.add_argument(
        "--output_parquet",
        type=Path,
        required=True,
        help="Output parquet path for collect_responses.py.",
    )
    parser.add_argument(
        "--summary_out",
        type=Path,
        default=None,
        help="Optional JSON summary path. Defaults to <output_parquet>.summary.json.",
    )
    parser.add_argument(
        "--question_types",
        nargs="+",
        default=["factoid"],
        choices=["factoid", "yesno", "list", "summary"],
        help="BioASQ question types to retain. Factoid is the cleanest match to TriviaQA-style QA.",
    )
    parser.add_argument(
        "--min_question_year",
        type=int,
        default=None,
        help="Minimum question creation year inferred from the ObjectId-style question id.",
    )
    parser.add_argument(
        "--max_question_year",
        type=int,
        default=None,
        help="Maximum question creation year inferred from the ObjectId-style question id.",
    )
    parser.add_argument(
        "--max_alias_words",
        type=int,
        default=None,
        help="Drop questions whose longest alias exceeds this many whitespace-separated words.",
    )
    parser.add_argument(
        "--exclude_sentence_like",
        action="store_true",
        help="Drop questions with aliases that look like full-sentence answers.",
    )
    return parser.parse_args()


def normalize_answer(text: str) -> str:
    """Copy of collect_responses.normalize_answer to keep dataset prep lightweight."""

    def remove_articles(value: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", value)

    def white_space_fix(value: str) -> str:
        return " ".join(value.split())

    def handle_punc(value: str) -> str:
        exclude = set(string.punctuation + "‘’´`")
        return "".join(ch if ch not in exclude else " " for ch in value)

    if not text:
        return ""
    lowered = str(text).lower().replace("_", " ")
    return white_space_fix(remove_articles(handle_punc(lowered))).strip()


def parse_question_datetime(question_id: str) -> datetime | None:
    """BioASQ ids are ObjectId-like 24-char hex strings; their first 8 chars encode a timestamp."""
    if len(question_id) != 24:
        return None
    try:
        return datetime.fromtimestamp(int(question_id[:8], 16), tz=timezone.utc)
    except ValueError:
        return None


def dedupe_preserve_order(values: list[str]) -> list[str]:
    seen = set()
    output = []
    for value in values:
        if value not in seen:
            seen.add(value)
            output.append(value)
    return output


def flatten_exact_answer(exact_answer) -> list[str]:
    if exact_answer is None:
        return []
    if isinstance(exact_answer, str):
        return [exact_answer]

    flattened = []
    for item in exact_answer:
        if isinstance(item, list):
            flattened.extend(str(value) for value in item if value is not None)
        elif item is not None:
            flattened.append(str(item))
    return flattened


def is_sentence_like(alias: str) -> bool:
    words = alias.split()
    return alias.strip().endswith(".") or len(words) >= 12


def load_questions(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload["questions"]


def question_passes_filters(question: dict, args) -> tuple[bool, dict]:
    question_id = question["id"]
    question_dt = parse_question_datetime(question_id)
    question_year = question_dt.year if question_dt else None
    aliases = dedupe_preserve_order(flatten_exact_answer(question.get("exact_answer")))

    metadata = {
        "question_year": question_year,
        "aliases": aliases,
    }

    if question.get("type") not in args.question_types:
        return False, metadata
    if not aliases:
        return False, metadata
    if args.min_question_year is not None and question_year is not None:
        if question_year < args.min_question_year:
            return False, metadata
    if args.max_question_year is not None and question_year is not None:
        if question_year > args.max_question_year:
            return False, metadata
    if args.max_alias_words is not None:
        if max(len(alias.split()) for alias in aliases) > args.max_alias_words:
            return False, metadata
    if args.exclude_sentence_like and any(is_sentence_like(alias) for alias in aliases):
        return False, metadata
    return True, metadata


def build_record(question: dict, question_year: int | None, aliases: list[str]) -> dict:
    normalized_aliases = dedupe_preserve_order(
        [
            normalized
            for normalized in (normalize_answer(alias) for alias in aliases)
            if normalized
        ]
    )
    return {
        "question_id": question["id"],
        "question": question["body"],
        "answer": {
            "aliases": aliases,
            "normalized_aliases": normalized_aliases,
        },
        "bioasq_type": question["type"],
        "question_year": question_year,
        "n_documents": len(question.get("documents", [])),
        "n_snippets": len(question.get("snippets", [])),
    }


def summarize(records: list[dict], source_questions: list[dict], args) -> dict:
    type_counts = Counter(record["bioasq_type"] for record in records)
    year_counts = Counter(record["question_year"] for record in records)
    alias_count_hist = Counter(len(record["answer"]["aliases"]) for record in records)
    max_alias_words_hist = Counter(
        max(len(alias.split()) for alias in record["answer"]["aliases"])
        for record in records
    )

    return {
        "source_path": str(args.input_json),
        "output_path": str(args.output_parquet),
        "source_question_count": len(source_questions),
        "selected_question_count": len(records),
        "selected_types": args.question_types,
        "min_question_year": args.min_question_year,
        "max_question_year": args.max_question_year,
        "max_alias_words": args.max_alias_words,
        "exclude_sentence_like": args.exclude_sentence_like,
        "type_counts": dict(sorted(type_counts.items())),
        "year_counts": {
            str(year): count for year, count in sorted(year_counts.items())
        },
        "alias_count_histogram": dict(sorted(alias_count_hist.items())),
        "max_alias_word_histogram": dict(sorted(max_alias_words_hist.items())),
    }


def main():
    args = parse_args()
    summary_out = args.summary_out or args.output_parquet.with_suffix(".summary.json")

    questions = load_questions(args.input_json)
    records = []

    for question in questions:
        passed, metadata = question_passes_filters(question, args)
        if not passed:
            continue
        records.append(
            build_record(question, metadata["question_year"], metadata["aliases"])
        )

    args.output_parquet.parent.mkdir(parents=True, exist_ok=True)
    summary_out.parent.mkdir(parents=True, exist_ok=True)

    dataset = Dataset.from_list(records)
    dataset.to_parquet(str(args.output_parquet))

    summary = summarize(records, questions, args)
    with summary_out.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    print(
        f"Wrote {len(records)} questions to {args.output_parquet} "
        f"and summary to {summary_out}"
    )


if __name__ == "__main__":
    main()
