"""Tests for TriviaQA bridge benchmark grading logic."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from run_intervention import _triviaqa_bridge_prompt, grade_triviaqa_bridge


# ---------------------------------------------------------------------------
# grade_triviaqa_bridge
# ---------------------------------------------------------------------------


class TestGradeTriviaQaBridge:
    def test_exact_match(self):
        result = grade_triviaqa_bridge("David Seville", ["David Seville"])
        assert result["correct"] is True
        assert result["match_tier"] == "exact"
        assert result["matched_alias"] == "David Seville"

    def test_exact_match_with_normalization(self):
        result = grade_triviaqa_bridge("THE EIFFEL TOWER", ["Eiffel Tower"])
        assert result["correct"] is True
        assert result["match_tier"] == "exact"
        assert result["matched_alias"] == "Eiffel Tower"

    def test_boundary_match(self):
        result = grade_triviaqa_bridge("I think it was Paris France", ["Paris"])
        assert result["correct"] is True
        assert result["match_tier"] == "boundary"
        assert result["matched_alias"] == "Paris"

    def test_ambiguous_multi_answer_is_deferred_to_judge(self):
        result = grade_triviaqa_bridge("It could be Paris or London", ["Paris"])
        assert result["correct"] is False
        assert result["match_tier"] == "no_match"
        assert result["matched_alias"] is None

    def test_exact_alias_with_or_is_still_accepted(self):
        result = grade_triviaqa_bridge("To Have or Have Not", ["To Have or Have Not"])
        assert result["correct"] is True
        assert result["match_tier"] == "exact"
        assert result["matched_alias"] == "To Have or Have Not"

    def test_short_alias_rejected(self):
        result = grade_triviaqa_bridge("I think it was 10 items", ["10"])
        assert result["correct"] is False
        assert result["match_tier"] == "no_match"
        assert result["matched_alias"] is None

    def test_substring_rejection_mars_in_marshall(self):
        result = grade_triviaqa_bridge("marshall mcluhan was media theorist", ["Mars"])
        assert result["correct"] is False
        assert result["match_tier"] == "no_match"
        assert result["matched_alias"] is None

    def test_digit_containment_rejection(self):
        result = grade_triviaqa_bridge("answer is 1024 bytes", ["10"])
        assert result["correct"] is False
        assert result["match_tier"] == "no_match"
        assert result["matched_alias"] is None

    def test_no_match(self):
        result = grade_triviaqa_bridge("I have no idea", ["David Seville"])
        assert result["correct"] is False
        assert result["match_tier"] == "no_match"
        assert result["matched_alias"] is None

    def test_empty_response(self):
        result = grade_triviaqa_bridge("", ["test"])
        assert result["correct"] is False
        assert result["match_tier"] == "no_match"
        assert result["matched_alias"] is None

    def test_multiple_aliases_first_matches(self):
        result = grade_triviaqa_bridge(
            "David Seville", ["David Seville", "Ross Bagdasarian"]
        )
        assert result["correct"] is True
        assert result["match_tier"] == "exact"
        assert result["matched_alias"] == "David Seville"

    def test_multiple_aliases_second_matches_boundary(self):
        result = grade_triviaqa_bridge(
            "it was Bagdasarian who did it",
            ["David Seville", "Bagdasarian"],
        )
        assert result["correct"] is True
        assert result["match_tier"] == "boundary"
        assert result["matched_alias"] == "Bagdasarian"

    def test_exact_match_takes_priority_over_boundary(self):
        result = grade_triviaqa_bridge("Paris", ["Paris"])
        assert result["correct"] is True
        assert result["match_tier"] == "exact"

    def test_word_boundary_at_start_of_string(self):
        result = grade_triviaqa_bridge("paris is beautiful", ["Paris"])
        assert result["correct"] is True
        assert result["match_tier"] == "boundary"
        assert result["matched_alias"] == "Paris"

    def test_word_boundary_at_end_of_string(self):
        result = grade_triviaqa_bridge("the city is paris", ["Paris"])
        assert result["correct"] is True
        assert result["match_tier"] == "boundary"
        assert result["matched_alias"] == "Paris"

    def test_john_vs_johnson(self):
        result = grade_triviaqa_bridge("johnson was president", ["John"])
        assert result["correct"] is False
        assert result["match_tier"] == "no_match"
        assert result["matched_alias"] is None


# ---------------------------------------------------------------------------
# _triviaqa_bridge_prompt
# ---------------------------------------------------------------------------


class TestTriviaQaBridgePrompt:
    def test_format(self):
        sample = {"question": "Who painted the Mona Lisa?"}
        prompt = _triviaqa_bridge_prompt(sample)
        assert (
            prompt
            == "Question: Who painted the Mona Lisa?\nAnswer with a single short factual phrase only."
        )
