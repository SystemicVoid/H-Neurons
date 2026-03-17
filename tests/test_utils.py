"""Tests for scripts/utils.py -- shared evaluation helpers."""

from __future__ import annotations

import sys
from pathlib import Path

# scripts/ uses flat sibling imports; add it to sys.path for test discovery.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from utils import extract_mc_answer, normalize_answer


# ---------------------------------------------------------------------------
# normalize_answer
# ---------------------------------------------------------------------------


class TestNormalizeAnswer:
    def test_lowercases(self):
        assert normalize_answer("PARIS") == "paris"

    def test_strips_articles(self):
        assert normalize_answer("The Eiffel Tower") == "eiffel tower"

    def test_strips_punctuation(self):
        assert normalize_answer("hello, world!") == "hello world"

    def test_collapses_whitespace(self):
        assert normalize_answer("  lots   of   spaces  ") == "lots of spaces"

    def test_underscores_to_spaces(self):
        assert normalize_answer("new_york_city") == "new york city"

    def test_curly_quotes_stripped(self):
        assert normalize_answer("\u2018quoted\u2019") == "quoted"

    def test_empty_string(self):
        assert normalize_answer("") == ""

    def test_none_returns_empty(self):
        assert normalize_answer(None) == ""

    def test_numeric_passthrough(self):
        assert normalize_answer("42") == "42"

    def test_combined_normalization(self):
        assert normalize_answer("The  'quick_brown' Fox!") == "quick brown fox"


# ---------------------------------------------------------------------------
# extract_mc_answer
# ---------------------------------------------------------------------------

VALID = ["A", "B", "C", "D"]


class TestExtractMcAnswer:
    def test_starts_with_letter(self):
        assert extract_mc_answer("B) some explanation", VALID) == "B"

    def test_answer_is_pattern(self):
        assert extract_mc_answer("The answer is C", VALID) == "C"

    def test_parenthesized(self):
        assert extract_mc_answer("(A)", VALID) == "A"

    def test_bold_markdown(self):
        assert extract_mc_answer("I think **D** is correct", VALID) == "D"

    def test_dot_pattern(self):
        assert extract_mc_answer("A. The first option", VALID) == "A"

    def test_standalone_letter_fallback(self):
        assert extract_mc_answer("Hmm, probably B here", VALID) == "B"

    def test_no_match_returns_none(self):
        assert extract_mc_answer("I have no idea", VALID) is None

    def test_empty_response(self):
        assert extract_mc_answer("", VALID) is None

    def test_invalid_letter_ignored(self):
        assert extract_mc_answer("E is my choice", VALID) is None

    def test_lowercase_input_still_matches(self):
        assert extract_mc_answer("the answer is b", VALID) == "B"

    def test_correct_is_pattern(self):
        assert extract_mc_answer("The correct answer is A", VALID) == "A"
