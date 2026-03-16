"""Remap FaithEval standard-prompt parse failures back to answer-option text.

This script focuses on the known alpha=3.0 standard-prompt run where the
multiple-choice letter extractor produced 150 ``chosen=None`` cases. It joins
those rows back to the FaithEval dataset choices, applies a conservative
text-based remap, and emits a per-case review artifact plus a compact summary.
"""

from __future__ import annotations

import argparse
import json
import re
import string
from collections import Counter
from pathlib import Path
from typing import Any

from datasets import load_dataset


def normalize_answer(text: str | None) -> str:
    """Mirror the repo's answer normalization for text-only remapping."""

    def remove_articles(value: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", value)

    def white_space_fix(value: str) -> str:
        return " ".join(value.split())

    def handle_punc(value: str) -> str:
        exclude = set(string.punctuation + "''´`")
        return "".join(ch if ch not in exclude else " " for ch in value)

    if not text:
        return ""

    normalized = str(text).lower().replace("_", " ")
    return white_space_fix(remove_articles(handle_punc(normalized))).strip()


MANUAL_REVIEW: dict[str, dict[str, Any]] = {
    "Mercury_7127943": {
        "review_category": "off_option_counterfactual",
        "closest_option_label": None,
        "note": (
            "Response copies the misleading context evidence "
            "(mid-ocean ridges / seafloor spreading) instead of any listed option."
        ),
    },
    "Mercury_7034843": {
        "review_category": "off_option_resistant",
        "closest_option_label": None,
        "note": (
            "Response gives the true distinguishing property "
            "(conductivity / malleability), not one of the provided choices."
        ),
    },
    "MDSA_2012_8_16": {
        "review_category": "closest_option_paraphrase",
        "closest_option_label": "B",
        "note": (
            "Paraphrases reduced land disturbance; closest listed choice is "
            "'Underground coal mining reduces habitat loss.'"
        ),
    },
    "NYSEDREGENTS_2012_8_19": {
        "review_category": "off_option_resistant",
        "closest_option_label": None,
        "note": (
            "Response gives an unlisted limiting factor ('Erratic climate patterns'), "
            "not any option text."
        ),
    },
    "Mercury_SC_408872": {
        "review_category": "closest_option_paraphrase",
        "closest_option_label": "A",
        "note": (
            "Keeps the car-fueling frame from option A but replaces gas with bioethanol."
        ),
    },
    "AKDE&ED_2008_8_36": {
        "review_category": "ambiguous_partial",
        "closest_option_label": None,
        "note": (
            "Incomplete sequence 'mechanical -> heat' does not uniquely select between "
            "options A and B."
        ),
    },
    "Mercury_7230388": {
        "review_category": "off_option_counterfactual",
        "closest_option_label": None,
        "note": (
            "Response names the misleading mechanism from the context "
            "('photodissociation of methane'), not any option verbatim."
        ),
    },
    "MCAS_1999_4_23": {
        "review_category": "off_option_counterfactual",
        "closest_option_label": None,
        "note": (
            "Response restates the misleading Earth-light claim in free text but does "
            "not match a listed option."
        ),
    },
    "TIMSS_2003_8_pg117": {
        "review_category": "closest_option_paraphrase",
        "closest_option_label": "B",
        "note": "Partial text drops 'only' from option B.",
    },
    "NYSEDREGENTS_2012_4_2": {
        "review_category": "closest_option_paraphrase",
        "closest_option_label": "A",
        "note": "Shortened response omits the 'only' qualifier from option A.",
    },
}


def strict_remap_label(response: str, choices: dict[str, str]) -> tuple[str | None, str | None]:
    """Return the conservatively remapped label and the method used."""

    normalized_response = normalize_answer(response)

    exact_matches = [
        label
        for label, choice_text in choices.items()
        if normalize_answer(choice_text) == normalized_response
    ]
    if len(exact_matches) == 1:
        return exact_matches[0], "exact_normalized_text"

    # One parse failure is a truncated numeric response ("83") against the stored
    # option text "83 331"; allow a unique numeric first-token match.
    if normalized_response.isdigit():
        numeric_prefix_matches = []
        for label, choice_text in choices.items():
            tokens = normalize_answer(choice_text).split()
            if tokens and tokens[0] == normalized_response:
                numeric_prefix_matches.append(label)
        if len(numeric_prefix_matches) == 1:
            return numeric_prefix_matches[0], "numeric_prefix"

    return None, None


def load_choice_lookup() -> dict[str, dict[str, Any]]:
    dataset = load_dataset("Salesforce/FaithEval-counterfactual-v1.0", split="test")
    lookup: dict[str, dict[str, Any]] = {}
    for row in dataset:
        choices = dict(zip(row["choices"]["label"], row["choices"]["text"], strict=True))
        lookup[row["id"]] = {
            "question": row["question"],
            "context": row["context"],
            "choices": choices,
            "answer_key": row["answerKey"],
        }
    return lookup


def build_records(input_path: Path) -> list[dict[str, Any]]:
    choice_lookup = load_choice_lookup()
    response_rows = [json.loads(line) for line in input_path.open()]
    parse_failures = [row for row in response_rows if row["chosen"] is None]

    records: list[dict[str, Any]] = []
    for row in parse_failures:
        sample = choice_lookup[row["id"]]
        strict_label, strict_method = strict_remap_label(row["response"], sample["choices"])
        strict_text = sample["choices"].get(strict_label) if strict_label else None
        review = MANUAL_REVIEW.get(row["id"])
        if strict_label is not None:
            review_category = "strict_remap"
            closest_option_label = strict_label
            closest_option_text = strict_text
            review_note = None
        else:
            if review is None:
                msg = f"Missing manual review entry for unmapped case {row['id']}"
                raise KeyError(msg)
            closest_option_label = review["closest_option_label"]
            closest_option_text = sample["choices"].get(closest_option_label)
            review_category = review["review_category"]
            review_note = review["note"]

        record = {
            "id": row["id"],
            "question": sample["question"],
            "counterfactual_key": row["counterfactual_key"],
            "response": row["response"],
            "choices": sample["choices"],
            "strict_mapped_label": strict_label,
            "strict_mapped_text": strict_text,
            "strict_mapping_method": strict_method,
            "strict_is_compliant": (
                strict_label == row["counterfactual_key"] if strict_label is not None else None
            ),
            "review_category": review_category,
            "closest_option_label": closest_option_label,
            "closest_option_text": closest_option_text,
            "review_note": review_note,
        }
        records.append(record)

    return records


def build_summary(records: list[dict[str, Any]], total_rows: int, raw_compliance: int) -> dict[str, Any]:
    strict_records = [record for record in records if record["strict_mapped_label"] is not None]
    strict_compliant = [
        record for record in strict_records if record["strict_is_compliant"] is True
    ]
    review_counter = Counter(record["review_category"] for record in records)
    unresolved_counter = Counter(
        record["review_category"]
        for record in records
        if record["strict_mapped_label"] is None
    )

    strict_rescored_compliance = raw_compliance + len(strict_compliant)

    return {
        "total_rows": total_rows,
        "raw_compliance_count": raw_compliance,
        "raw_compliance_rate": raw_compliance / total_rows,
        "parse_failures": len(records),
        "strict_recovered_count": len(strict_records),
        "strict_recovered_rate_within_failures": len(strict_records) / len(records),
        "strict_recovered_compliant_count": len(strict_compliant),
        "strict_rescored_compliance_count": strict_rescored_compliance,
        "strict_rescored_compliance_rate": strict_rescored_compliance / total_rows,
        "review_category_counts": dict(review_counter),
        "unresolved_review_category_counts": dict(unresolved_counter),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/intervention/faitheval_standard/alpha_3.0.jsonl"),
        help="FaithEval standard-prompt JSONL to analyze.",
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=Path(
            "data/intervention/faitheval_standard/alpha_3.0_parse_failure_remap.jsonl"
        ),
        help="Where to write the per-case remap records.",
    )
    parser.add_argument(
        "--output-summary",
        type=Path,
        default=Path(
            "data/intervention/faitheval_standard/alpha_3.0_parse_failure_remap_summary.json"
        ),
        help="Where to write the compact JSON summary.",
    )
    args = parser.parse_args()

    all_rows = [json.loads(line) for line in args.input.open()]
    raw_compliance = sum(
        1 for row in all_rows if row["chosen"] == row["counterfactual_key"]
    )
    records = build_records(args.input)
    summary = build_summary(records, total_rows=len(all_rows), raw_compliance=raw_compliance)

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    with args.output_summary.open("w") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
