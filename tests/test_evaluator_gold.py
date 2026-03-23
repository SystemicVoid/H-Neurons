"""Gold-label regression tests for deterministic evaluators.

Validates extract_mc_answer and strict_remap_label against hand-verified
cases from tests/gold_labels/faitheval_standard.jsonl.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from remap_faitheval_standard_parse_failures import strict_remap_label
from utils import extract_mc_answer

GOLD_PATH = Path(__file__).resolve().parent / "gold_labels" / "faitheval_standard.jsonl"

VALID_LETTERS = ["A", "B", "C", "D"]


def _load_gold() -> list[dict]:
    return [json.loads(line) for line in GOLD_PATH.read_text().splitlines() if line]


def _gold_by_category(category: str) -> list[dict]:
    return [r for r in _load_gold() if r["category"] == category]


# ---------------------------------------------------------------------------
# extract_mc_answer: should succeed on parser_success, fail on parse failures
# ---------------------------------------------------------------------------


class TestExtractMcAnswerGold:
    @pytest.mark.parametrize(
        "case",
        _gold_by_category("parser_success"),
        ids=[c["id"] for c in _gold_by_category("parser_success")],
    )
    def test_parser_success(self, case):
        result = extract_mc_answer(case["response"], VALID_LETTERS)
        assert result == case["expected_mc_extract"], (
            f"{case['id']}: expected {case['expected_mc_extract']!r}, got {result!r}"
        )

    @pytest.mark.parametrize(
        "case",
        _gold_by_category("strict_remap"),
        ids=[c["id"] for c in _gold_by_category("strict_remap")],
    )
    def test_parse_failure_on_remap_cases(self, case):
        valid = list(case["choices"].keys()) if case["choices"] else VALID_LETTERS
        result = extract_mc_answer(case["response"], valid)
        assert result is None, (
            f"{case['id']}: expected parse failure (None), got {result!r}"
        )

    @pytest.mark.parametrize(
        "case",
        _gold_by_category("manual_review"),
        ids=[c["id"] for c in _gold_by_category("manual_review")],
    )
    def test_parse_failure_on_manual_review_cases(self, case):
        valid = list(case["choices"].keys()) if case["choices"] else VALID_LETTERS
        result = extract_mc_answer(case["response"], valid)
        assert result is None, (
            f"{case['id']}: expected parse failure (None), got {result!r}"
        )


# ---------------------------------------------------------------------------
# strict_remap_label: text-based recovery for parse failures
# ---------------------------------------------------------------------------


class TestStrictRemapGold:
    @pytest.mark.parametrize(
        "case",
        _gold_by_category("strict_remap"),
        ids=[c["id"] for c in _gold_by_category("strict_remap")],
    )
    def test_remap_recovers_label(self, case):
        label, method = strict_remap_label(case["response"], case["choices"])
        assert label == case["expected_remap_label"], (
            f"{case['id']}: expected label {case['expected_remap_label']!r}, "
            f"got {label!r}"
        )
        assert method == case["expected_remap_method"], (
            f"{case['id']}: expected method {case['expected_remap_method']!r}, "
            f"got {method!r}"
        )

    @pytest.mark.parametrize(
        "case",
        _gold_by_category("strict_remap"),
        ids=[c["id"] for c in _gold_by_category("strict_remap")],
    )
    def test_remap_compliance(self, case):
        label, _ = strict_remap_label(case["response"], case["choices"])
        is_compliant = label == case["counterfactual_key"] if label else None
        assert is_compliant == case["expected_compliance"], (
            f"{case['id']}: expected compliance {case['expected_compliance']!r}, "
            f"got {is_compliant!r}"
        )

    @pytest.mark.parametrize(
        "case",
        _gold_by_category("manual_review"),
        ids=[c["id"] for c in _gold_by_category("manual_review")],
    )
    def test_manual_review_cases_not_recovered(self, case):
        label, method = strict_remap_label(case["response"], case["choices"])
        assert label == case["expected_remap_label"], (
            f"{case['id']}: expected {case['expected_remap_label']!r}, got {label!r}"
        )
