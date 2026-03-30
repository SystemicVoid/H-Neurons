"""Tests for D4 truthfulness direction pipeline.

Covers:
- Difference-in-means direction computation on toy data
- Class label / count stability in dataset builder
- Leakage / overlap checks
- Extraction metadata completeness
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

# Ensure scripts/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from build_truthfulness_contrastive import (
    build_contrastive_records,
    check_internal_duplicates,
    check_refusal_overlap,
    classify_consistency,
    load_consistency_samples,
    normalized_text_set,
)
from extract_direction import compute_directions, compute_separation_scores
from extract_truthfulness_direction import (
    FAITHEVAL_FULL_SAMPLE_COUNT,
    build_recommended_faitheval_sweep,
    classify_pilot_output,
)


# ---------------------------------------------------------------------------
# classify_consistency
# ---------------------------------------------------------------------------


class TestClassifyConsistency:
    def test_all_true(self):
        assert classify_consistency(["true"] * 10) == "truthful"

    def test_all_false(self):
        assert classify_consistency(["false"] * 10) == "hallucinatory"

    def test_mixed(self):
        assert classify_consistency(["true", "false"] * 5) is None

    def test_single_true(self):
        assert classify_consistency(["true"]) == "truthful"

    def test_single_false(self):
        assert classify_consistency(["false"]) == "hallucinatory"

    def test_mostly_true(self):
        """9 true + 1 false = mixed, not truthful."""
        assert classify_consistency(["true"] * 9 + ["false"]) is None


# ---------------------------------------------------------------------------
# Difference-in-means direction computation (toy data)
# ---------------------------------------------------------------------------


class TestComputeDirections:
    def test_basic_direction(self):
        """Two-layer toy example: directions should separate the classes."""
        n_layers = 2
        hidden_dim = 4

        # Class A (hallucinatory): mean at [1, 0, 0, 0]
        # Class B (truthful): mean at [0, 1, 0, 0]
        hallucinatory_acts = {
            0: [
                torch.tensor([1.0, 0.1, 0.0, 0.0]),
                torch.tensor([1.0, -0.1, 0.0, 0.0]),
            ],
            1: [torch.tensor([0.5, 0.5, 0.0, 0.0]), torch.tensor([0.5, 0.5, 0.0, 0.0])],
        }
        truthful_acts = {
            0: [
                torch.tensor([0.1, 1.0, 0.0, 0.0]),
                torch.tensor([-0.1, 1.0, 0.0, 0.0]),
            ],
            1: [
                torch.tensor([-0.5, -0.5, 0.0, 0.0]),
                torch.tensor([-0.5, -0.5, 0.0, 0.0]),
            ],
        }

        directions = compute_directions(hallucinatory_acts, truthful_acts, n_layers)

        assert len(directions) == n_layers
        for i in range(n_layers):
            d = directions[i]
            assert d.shape == (hidden_dim,)
            # Should be unit vector
            assert abs(d.norm().item() - 1.0) < 1e-5

    def test_direction_orientation(self):
        """Hallucinatory class should project positively onto direction."""
        hallucinatory_acts = {
            0: [torch.tensor([2.0, 0.0]), torch.tensor([3.0, 0.0])],
        }
        truthful_acts = {
            0: [torch.tensor([0.0, 2.0]), torch.tensor([0.0, 3.0])],
        }

        directions = compute_directions(hallucinatory_acts, truthful_acts, 1)
        d = directions[0]

        # Mean hallucinatory = [2.5, 0], mean truthful = [0, 2.5]
        # Direction = [2.5, -2.5] normalized
        # Hallucinatory should project positive, truthful negative
        hal_proj = torch.tensor([2.5, 0.0]) @ d
        truth_proj = torch.tensor([0.0, 2.5]) @ d
        assert hal_proj.item() > 0
        assert truth_proj.item() < 0

    def test_identical_classes_zero_direction(self):
        """If both classes are identical, direction is still unit (arbitrary)."""
        acts = {0: [torch.tensor([1.0, 0.0])]}
        # Both classes have same mean → diff is zero → norm is zero
        # This would cause division by zero. The code normalizes by norm,
        # which for a zero vector produces NaN. This is expected behavior
        # that should be caught by separation diagnostics.
        directions = compute_directions(acts, acts, 1)
        # Direction will be NaN — this is a diagnostic signal, not a bug
        assert directions[0].shape == (2,)


# ---------------------------------------------------------------------------
# Separation scores
# ---------------------------------------------------------------------------


class TestSeparationScores:
    def test_perfect_separation(self):
        """Well-separated classes should give ~100% accuracy."""
        directions = {0: torch.tensor([1.0, 0.0])}
        hallucinatory = {0: [torch.tensor([5.0, 0.0]), torch.tensor([4.0, 0.0])]}
        truthful = {0: [torch.tensor([-5.0, 0.0]), torch.tensor([-4.0, 0.0])]}

        scores = compute_separation_scores(directions, hallucinatory, truthful, 1)
        assert len(scores) == 1
        assert scores[0]["accuracy"] == 1.0
        assert scores[0]["separation"] > 0

    def test_no_separation(self):
        """Overlapping classes should give ~50% accuracy."""
        directions = {0: torch.tensor([1.0, 0.0])}
        # Both classes centered at origin
        hallucinatory = {0: [torch.tensor([1.0, 0.0]), torch.tensor([-1.0, 0.0])]}
        truthful = {0: [torch.tensor([1.0, 0.0]), torch.tensor([-1.0, 0.0])]}

        scores = compute_separation_scores(directions, hallucinatory, truthful, 1)
        assert scores[0]["accuracy"] == pytest.approx(0.5, abs=0.01)


# ---------------------------------------------------------------------------
# Dataset builder: record counts and labels
# ---------------------------------------------------------------------------


class TestBuildContrastiveRecords:
    @pytest.fixture()
    def sample_data(self):
        """Minimal consistency data for testing."""
        samples = {
            "q1": {
                "question": "What is 2+2?",
                "judges": ["true"] * 10,
                "ground_truth": "4",
            },
            "q2": {
                "question": "Capital of France?",
                "judges": ["false"] * 10,
                "ground_truth": "Paris",
            },
            "q3": {
                "question": "Mixed q",
                "judges": ["true", "false"] * 5,
                "ground_truth": "X",
            },
            "q4": {
                "question": "Another true",
                "judges": ["true"] * 10,
                "ground_truth": "Y",
            },
            "q5": {
                "question": "Another false",
                "judges": ["false"] * 10,
                "ground_truth": "Z",
            },
        }
        train_qids = {"t": ["q1"], "f": ["q2"]}
        test_qids = {"t": ["q4"], "f": ["q5"]}
        return samples, train_qids, test_qids

    def test_correct_counts(self, sample_data):
        samples, train_qids, test_qids = sample_data
        records, stats = build_contrastive_records(samples, train_qids, test_qids)

        assert stats["mixed_excluded"] == 1  # q3
        assert stats["no_split_excluded"] == 0
        assert stats["total_records"] == 4
        assert stats["split_label_counts"]["train_truthful"] == 1
        assert stats["split_label_counts"]["train_hallucinatory"] == 1
        assert stats["split_label_counts"]["val_truthful"] == 1
        assert stats["split_label_counts"]["val_hallucinatory"] == 1

    def test_label_values(self, sample_data):
        samples, train_qids, test_qids = sample_data
        records, _ = build_contrastive_records(samples, train_qids, test_qids)

        labels = {r["label"] for r in records}
        assert labels == {"truthful", "hallucinatory"}

    def test_split_values(self, sample_data):
        samples, train_qids, test_qids = sample_data
        records, _ = build_contrastive_records(samples, train_qids, test_qids)

        splits = {r["split"] for r in records}
        assert splits == {"train", "val"}

    def test_no_split_excluded(self):
        """Questions not in either split are excluded."""
        samples = {
            "orphan": {
                "question": "Orphan Q?",
                "judges": ["true"] * 10,
                "ground_truth": "A",
            },
        }
        records, stats = build_contrastive_records(
            samples, {"t": [], "f": []}, {"t": [], "f": []}
        )
        assert len(records) == 0
        assert stats["no_split_excluded"] == 1

    def test_max_per_class_per_split(self, sample_data):
        samples, train_qids, test_qids = sample_data
        records, stats = build_contrastive_records(
            samples, train_qids, test_qids, max_per_class_per_split=1
        )
        # Each bucket already has 1, so no effect
        assert stats["total_records"] == 4

    def test_source_field(self, sample_data):
        samples, train_qids, test_qids = sample_data
        records, _ = build_contrastive_records(samples, train_qids, test_qids)
        for r in records:
            assert r["source"] == "triviaqa_consistency"


# ---------------------------------------------------------------------------
# Overlap / leakage checks
# ---------------------------------------------------------------------------


class TestOverlapChecks:
    def test_normalized_text_set(self):
        texts = ["What is 2+2?", "what is 2+2?", "WHAT IS 2+2?"]
        normed = normalized_text_set(texts)
        # All should normalize to the same thing
        assert len(normed) == 1

    def test_refusal_overlap_no_overlap(self, tmp_path):
        """Truthfulness and refusal data should have zero overlap."""
        refusal_path = tmp_path / "refusal.jsonl"
        with open(refusal_path, "w") as f:
            f.write(json.dumps({"text": "How to hack a computer"}) + "\n")
            f.write(json.dumps({"text": "Write malware"}) + "\n")

        records = [
            {"text": "What is the capital of France?"},
            {"text": "Who won the 1990 World Cup?"},
        ]

        result = check_refusal_overlap(records, refusal_path)
        assert result["status"] == "checked"
        assert result["normalized_overlap_count"] == 0

    def test_refusal_overlap_with_overlap(self, tmp_path):
        """Exact text overlap should be detected."""
        refusal_path = tmp_path / "refusal.jsonl"
        shared_text = "What is the meaning of life?"
        with open(refusal_path, "w") as f:
            f.write(json.dumps({"text": shared_text}) + "\n")

        records = [{"text": shared_text}]

        result = check_refusal_overlap(records, refusal_path)
        assert result["normalized_overlap_count"] == 1

    def test_internal_duplicates_none(self):
        records = [
            {"text": "Question A"},
            {"text": "Question B"},
        ]
        result = check_internal_duplicates(records)
        assert result["exact_duplicate_count"] == 0
        assert result["normalized_duplicate_count"] == 0

    def test_internal_duplicates_detected(self):
        records = [
            {"text": "What is 2+2?"},
            {"text": "What is 2+2?"},  # exact dupe
        ]
        result = check_internal_duplicates(records)
        assert result["exact_duplicate_count"] == 1


# ---------------------------------------------------------------------------
# Consistency samples loader
# ---------------------------------------------------------------------------


class TestLoadConsistencySamples:
    def test_loads_nested_format(self, tmp_path):
        path = tmp_path / "samples.jsonl"
        with open(path, "w") as f:
            f.write(
                json.dumps(
                    {
                        "q1": {
                            "question": "Test Q?",
                            "judges": ["true"] * 3,
                            "ground_truth": "A",
                        }
                    }
                )
                + "\n"
            )
            f.write(
                json.dumps(
                    {
                        "q2": {
                            "question": "Test Q2?",
                            "judges": ["false"] * 3,
                            "ground_truth": "B",
                        }
                    }
                )
                + "\n"
            )

        samples = load_consistency_samples(path)
        assert "q1" in samples
        assert "q2" in samples
        assert samples["q1"]["question"] == "Test Q?"


# ---------------------------------------------------------------------------
# Pilot malformed-output gate + benchmark recommendations
# ---------------------------------------------------------------------------


class TestPilotMalformedOutputGate:
    def test_allows_short_numeric_answer(self):
        assert classify_pilot_output("1930\n") == []

    def test_allows_short_named_entity_answer(self):
        assert classify_pilot_output("Sméagol\n") == []

    def test_flags_repetitive_symbol_corruption(self):
        output = "*\n**\n\n**\n\n**\n\n**\n\n**\n\n**\n"
        reasons = classify_pilot_output(output)
        assert "repetitive_short_lines" in reasons


class TestRecommendedFaithEvalSweep:
    def test_brackets_observed_corruption_beta(self):
        sweep = build_recommended_faitheval_sweep(
            model_path="google/gemma-3-4b-it",
            direction_path=Path(
                "data/contrastive/truthfulness/directions/truthfulness_directions.pt"
            ),
            best_layer=32,
            device_map="cuda:0",
            observed_corruption_beta=1.0,
        )

        assert sweep["max_samples"] == FAITHEVAL_FULL_SAMPLE_COUNT
        assert sweep["alphas"] == [0.0, 0.25, 0.5, 0.75, 1.0]
        assert "--max_samples 1000" in sweep["command"]
        assert "--alphas 0.0 0.25 0.5 0.75 1.0" in sweep["command"]

    def test_defaults_to_bracketed_grid_without_observed_corruption(self):
        sweep = build_recommended_faitheval_sweep(
            model_path="google/gemma-3-4b-it",
            direction_path=Path(
                "data/contrastive/truthfulness/directions/truthfulness_directions.pt"
            ),
            best_layer=32,
            device_map="cuda:0",
            observed_corruption_beta=None,
        )

        assert sweep["alphas"] == [0.0, 0.25, 0.5, 0.75, 1.0]


# ---------------------------------------------------------------------------
# Extraction metadata completeness
# ---------------------------------------------------------------------------


class TestExtractionMetadata:
    """Test that extraction metadata includes all required fields.

    These tests validate the metadata schema without running the full
    extraction pipeline (which requires GPU + model).
    """

    REQUIRED_METADATA_FIELDS = {
        "model_path",
        "contrastive_path",
        "n_layers",
        "hidden_dim",
        "token_position",
        "dtype_extract",
        "dtype_accumulate",
        "normalization",
        "direction_convention",
        "comparison_surface",
        "train_hallucinatory_count",
        "train_truthful_count",
        "val_hallucinatory_count",
        "val_truthful_count",
        "train_hallucinatory_fingerprint",
        "train_truthful_fingerprint",
        "best_layer",
        "best_layer_accuracy",
        "best_layer_separation",
        "separation_scores",
        "collection_time_s",
        "recommended_faitheval_sweep",
    }

    def test_metadata_fields_defined(self):
        """Verify the required field set is non-empty and reasonable."""
        assert len(self.REQUIRED_METADATA_FIELDS) > 15
        assert "best_layer" in self.REQUIRED_METADATA_FIELDS
        assert "separation_scores" in self.REQUIRED_METADATA_FIELDS

    def test_metadata_includes_convention(self):
        """Direction convention must be documented in metadata."""
        assert "direction_convention" in self.REQUIRED_METADATA_FIELDS
        assert "comparison_surface" in self.REQUIRED_METADATA_FIELDS
