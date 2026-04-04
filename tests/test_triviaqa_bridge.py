"""Tests for TriviaQA bridge benchmark grading logic."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import build_triviaqa_bridge_manifest as manifest_builder
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


class TestTriviaQaBridgeManifest:
    def test_main_applies_exclusions_and_records_metadata(self, tmp_path, monkeypatch):
        pandas = pytest.importorskip("pandas")

        parquet_path = tmp_path / "validation.parquet"
        output_dir = tmp_path / "manifests"
        exclusion_path = tmp_path / "exclude.json"
        exclusion_path.write_text(json.dumps(["q5"]), encoding="utf-8")

        dataframe = pandas.DataFrame(
            [
                {
                    "question_id": f"q{idx}",
                    "question": f"Question {idx}?",
                    "answer": {"aliases": [f"Answer {idx}"]},
                }
                for idx in range(1, 6)
            ]
        )

        monkeypatch.setattr(pandas, "read_parquet", lambda _path: dataframe)
        monkeypatch.setattr(
            manifest_builder,
            "_parse_args",
            lambda: argparse.Namespace(
                seed=42,
                pilot_n=1,
                dev_n=1,
                test_n=1,
                reserve_n=1,
                parquet_path=str(parquet_path),
                output_dir=str(output_dir),
                exclude_qids_path=[str(exclusion_path)],
            ),
        )

        manifest_builder.main()

        metadata = json.loads(
            (output_dir / "triviaqa_bridge_metadata_seed42.json").read_text(
                encoding="utf-8"
            )
        )
        sampled_ids = set()
        for split in ("pilot", "dev", "test", "reserve"):
            sampled_ids.update(
                json.loads(
                    (
                        output_dir
                        / f"triviaqa_bridge_{split}{1 if split != 'reserve' else 1}_seed42.json"
                    ).read_text(encoding="utf-8")
                )
            )

        assert "q5" not in sampled_ids
        assert metadata["exclusion_sets"]["n_sources"] == 1
        assert metadata["exclusion_sets"]["n_excluded_qids"] == 1
        assert metadata["exclusion_sets"]["candidate_pool_overlap"]["n_qids"] == 1
        assert metadata["exclusion_sets"]["sources"][0]["path"] == str(exclusion_path)

    def test_rejects_overlapping_exclusion_sets(self, tmp_path):
        first = tmp_path / "first.json"
        second = tmp_path / "second.json"
        first.write_text(json.dumps(["q1", "q2"]), encoding="utf-8")
        second.write_text(json.dumps(["q2", "q3"]), encoding="utf-8")

        with pytest.raises(ValueError, match="Exclusion set overlap detected"):
            manifest_builder._load_exclusion_qid_sets([str(first), str(second)])
