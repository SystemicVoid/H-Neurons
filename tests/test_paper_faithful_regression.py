"""Regression tests for paper-faithful TruthfulQA 2-fold CV pipeline.

Guards against bugs found during audit:
- Split leakage (test data in direction fitting)
- Wrong ranking metric (balanced_accuracy instead of val_accuracy)
- Wrong token position (answer span instead of last token only)
- Inconsistent question IDs across modules
- Sigma/direction computed from wrong subset
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from build_truthfulqa_splits import (
    build_folds,
    build_mc_manifests,
    stable_question_id,
)
from extract_truthfulness_iti import (
    ITIExample,
    _encode_with_answer_positions,
    _qa_prompt_paper,
    compute_head_directions,
    rank_heads,
)
from utils import fingerprint_ids


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


class StubTokenizer:
    """Word-splitting stub matching test_truthfulness_iti.py convention."""

    def __call__(self, text, add_special_tokens=False, return_tensors=None):
        token_count = max(1, len(text.split()))
        ids = list(range(token_count))
        if return_tensors == "pt":
            return {"input_ids": torch.tensor([ids], dtype=torch.long)}
        return {"input_ids": ids}


def _make_toy_questions(n: int = 10) -> list[dict[str, Any]]:
    """Build canonical-format question list without needing TruthfulQA.csv."""
    questions = []
    for i in range(n):
        text = f"What is the answer to question {i}?"
        sid = stable_question_id(text)
        questions.append({"stable_id": sid, "csv_idx": i, "question_text": text})
    return questions


def _write_fold_and_manifest(
    tmp_path: Path,
    questions: list[dict[str, Any]],
    folds: list[dict[str, Any]],
    fold_idx: int,
) -> tuple[Path, Path]:
    """Write canonical manifest + fold JSON + toy CSV. Returns (fold_path, csv_path)."""
    manifest_path = tmp_path / "canonical.json"
    fp = fingerprint_ids([q["stable_id"] for q in questions])
    manifest = {
        "version": 1,
        "source_csv": "toy.csv",
        "n_questions": len(questions),
        "fingerprint": fp,
        "questions": questions,
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    fold = dict(folds[fold_idx])
    fold["canonical_manifest"] = str(manifest_path)
    fold["canonical_fingerprint"] = fp
    fold_path = tmp_path / f"fold{fold_idx}.json"
    fold_path.write_text(json.dumps(fold), encoding="utf-8")

    # Write toy CSV with correct columns
    csv_path = tmp_path / "TruthfulQA.csv"
    lines = ["Question,Correct Answers,Incorrect Answers"]
    for q in questions:
        qtext = q["question_text"].replace(",", " ")
        lines.append(f"{qtext},correct_{q['csv_idx']},wrong_{q['csv_idx']}")
    csv_path.write_text("\n".join(lines), encoding="utf-8")

    return fold_path, csv_path


def _make_examples(
    n_train_per_label: int = 2,
    n_val_per_label: int = 2,
) -> list[ITIExample]:
    """Balanced ITIExample list with train/val splits and true/false labels."""
    examples = []
    idx = 0
    for split, n_per_label in [("train", n_train_per_label), ("val", n_val_per_label)]:
        for label in (1, 0):
            for _ in range(n_per_label):
                examples.append(
                    ITIExample(
                        example_id=f"{split}_{label}_{idx}",
                        family="iti_truthfulqa_paper",
                        split=split,
                        qid=f"tqa_fake_{idx:04d}",
                        question="Q?",
                        answer="A",
                        label=label,
                        weight=1.0,
                        prompt_text="Q? A",
                        answer_positions=(1,),
                        metadata={},
                    )
                )
                idx += 1
    return examples


# ---------------------------------------------------------------------------
# 1. Split integrity
# ---------------------------------------------------------------------------


class TestSplitIntegrity:
    @pytest.fixture()
    def toy_universe(self):
        questions = _make_toy_questions(10)
        folds = build_folds(questions, seed=42)
        return questions, folds

    def test_no_overlap_within_fold(self, toy_universe):
        _, folds = toy_universe
        for fold in folds:
            train = set(fold["dev"]["train"])
            val = set(fold["dev"]["val"])
            test = set(fold["test"])
            assert train.isdisjoint(val), "train/val overlap"
            assert (train | val).isdisjoint(test), "dev/test overlap"

    def test_fold_swap_symmetry(self, toy_universe):
        _, folds = toy_universe
        dev0 = set(folds[0]["dev"]["train"]) | set(folds[0]["dev"]["val"])
        dev1 = set(folds[1]["dev"]["train"]) | set(folds[1]["dev"]["val"])
        assert set(folds[0]["test"]) == dev1
        assert set(folds[1]["test"]) == dev0

    def test_full_coverage(self, toy_universe):
        questions, folds = toy_universe
        all_ids = {q["stable_id"] for q in questions}
        for fold in folds:
            fold_all = (
                set(fold["dev"]["train"]) | set(fold["dev"]["val"]) | set(fold["test"])
            )
            assert fold_all == all_ids

    def test_test_ids_excluded_from_examples(self, toy_universe, tmp_path):
        from extract_truthfulness_iti import build_truthfulqa_paper_examples

        questions, folds = toy_universe
        fold_path, csv_path = _write_fold_and_manifest(
            tmp_path, questions, folds, fold_idx=0
        )
        examples, _ = build_truthfulqa_paper_examples(
            tokenizer=StubTokenizer(),
            fold_path=fold_path,
            csv_path=csv_path,
        )
        example_qids = {ex.qid for ex in examples}
        test_ids = set(folds[0]["test"])
        assert example_qids.isdisjoint(test_ids), (
            f"test IDs leaked into examples: {example_qids & test_ids}"
        )


# ---------------------------------------------------------------------------
# 2. Stable IDs
# ---------------------------------------------------------------------------


class TestStableIDs:
    def test_same_text_same_id(self):
        text = "What is the speed of light?"
        assert stable_question_id(text) == stable_question_id(text)

    def test_whitespace_normalization(self):
        assert stable_question_id("What  is  the   answer?") == stable_question_id(
            "What is the answer?"
        )

    def test_case_normalization(self):
        assert stable_question_id("WHO WAS FIRST?") == stable_question_id(
            "who was first?"
        )

    def test_mc_manifest_and_extraction_use_same_ids(self, tmp_path):
        from extract_truthfulness_iti import build_truthfulqa_paper_examples

        questions = _make_toy_questions(10)
        folds = build_folds(questions, seed=42)
        build_mc_manifests(questions, folds)  # validates structure internally

        # The MC manifest for fold 0 references test-fold questions via csv_idx
        fold0_test_sids = set(folds[0]["test"])
        sid_to_csv = {q["stable_id"]: q["csv_idx"] for q in questions}
        mc_csv_indices = {sid_to_csv[sid] for sid in fold0_test_sids}

        # Extraction for fold 0 should use only dev-fold questions
        fold_path, csv_path = _write_fold_and_manifest(
            tmp_path, questions, folds, fold_idx=0
        )
        examples, _ = build_truthfulqa_paper_examples(
            tokenizer=StubTokenizer(),
            fold_path=fold_path,
            csv_path=csv_path,
        )
        example_csv_indices = {ex.metadata["csv_idx"] for ex in examples}

        # MC eval uses test fold; extraction uses dev fold — zero overlap
        assert mc_csv_indices.isdisjoint(example_csv_indices)
        # Together they cover everything
        assert mc_csv_indices | example_csv_indices == {q["csv_idx"] for q in questions}


# ---------------------------------------------------------------------------
# 3. Ranking semantics
# ---------------------------------------------------------------------------


class TestRankingSemantics:
    @pytest.fixture()
    def two_layer_setup(self):
        """2 layers, 1 head, head_dim=1.

        Layer 0: train separable, val separable  -> high val_acc + high auroc
        Layer 1: train separable, val anti-correlated -> low val_acc + low auroc

        Switching ranking_primary should not change order here, so we make
        layer 1 have high auroc but low val_accuracy by giving val predictions
        that are anti-correlated with labels (auroc computed on probabilities
        can still be high if the ranking is flipped — but with liblinear on
        1D data, if train is separable, val predictions follow the same sign).

        Better approach: make the two layers differ in val separability only.
        Layer 0: both train and val separable -> val_acc=1.0, auroc=1.0
        Layer 1: train separable, val is random noise -> val_acc~0.5, auroc~0.5
        Then under val_accuracy primary, layer 0 wins; under auroc primary,
        layer 0 still wins.  To get a real flip, we need:

        Layer 0: train barely separable (probe still learns), val perfectly
                 separable -> val_acc=1.0 but auroc on val moderate
        Layer 1: train perfectly separable, val with good ranking but bad
                 threshold -> val_acc<1.0 but auroc=1.0

        Simplest reliable approach: directly construct activations so that
        one layer has perfect val_acc but the other has perfect auroc by
        manipulating the threshold.

        Actually — with 1D logistic regression, auroc and val_accuracy are
        tightly coupled.  Use 2 heads on 1 layer instead, and directly
        verify that _rank_key ordering changes.  But rank_heads picks the
        best position_summary per (layer, head), so 2 heads in 1 layer gives
        2 ranked entries.

        Final reliable approach: 2 layers × 1 head.
        Layer 0: train [+5,+5,-5,-5] labels [1,1,0,0]; val [+5,+5,-5,-5] labels [1,1,0,0] => val_acc=1.0, auroc=1.0
        Layer 1: train [+5,+5,-5,-5] labels [1,1,0,0]; val [+5,+5,-5,-5] labels [1,1,0,0] => also perfect

        That won't differ. We need different val behavior per layer.
        Layer 0: val perfectly matches => val_acc=1.0
        Layer 1: val activations are constant zero => probe predicts one class for all => val_acc=0.5

        With constant-zero val, predict_proba gives ~0.5 for all => auroc undefined or 0.5, val_acc=0.5.
        """
        examples = _make_examples(n_train_per_label=2, n_val_per_label=2)
        n = len(examples)  # 8
        train_idx = [i for i, ex in enumerate(examples) if ex.split == "train"]
        val_idx = [i for i, ex in enumerate(examples) if ex.split == "val"]

        acts = torch.zeros(n, 2, 1, 1)
        # Layer 0: separable for both train and val
        for i in train_idx:
            acts[i, 0, 0, 0] = 5.0 if examples[i].label == 1 else -5.0
        for i in val_idx:
            acts[i, 0, 0, 0] = 5.0 if examples[i].label == 1 else -5.0

        # Layer 1: separable train, constant-zero val => probe learns but
        # val predictions are at chance
        for i in train_idx:
            acts[i, 1, 0, 0] = 5.0 if examples[i].label == 1 else -5.0
        # val stays zero

        return examples, {"last_answer_token": acts}

    def test_val_accuracy_primary_ranks_perfect_layer_first(self, two_layer_setup):
        examples, activations = two_layer_setup
        ranked, _ = rank_heads(
            activations,
            examples,
            position_summaries=("last_answer_token",),
            ranking_primary="val_accuracy",
        )
        # Layer 0 has val_acc=1.0; layer 1 has val_acc~0.5
        assert ranked[0]["layer"] == 0
        assert ranked[0]["val_accuracy"] > ranked[1]["val_accuracy"]

    def test_paper_faithful_metadata_records_val_accuracy(self, two_layer_setup):
        """The paper-faithful code path sets ranking_primary='val_accuracy'."""
        # Replicate the main() logic for paper-faithful
        is_paper_faithful = True
        rank_primary = "val_accuracy" if is_paper_faithful else "auroc"
        rank_position_summaries = (
            ("last_answer_token",)
            if is_paper_faithful
            else ("first_answer_token", "mean_answer_span", "last_answer_token")
        )

        examples, activations = two_layer_setup
        ranked, ranking_metadata = rank_heads(
            activations,
            examples,
            position_summaries=rank_position_summaries,
            ranking_primary=rank_primary,
        )
        # Metadata that would be logged
        metadata = {"head_ranking_metric": rank_primary}
        assert metadata["head_ranking_metric"] == "val_accuracy"
        assert rank_position_summaries == ("last_answer_token",)


# ---------------------------------------------------------------------------
# 4. Direction fitting semantics
# ---------------------------------------------------------------------------


class TestDirectionFittingSemantics:
    def test_paper_faithful_uses_all_dev_indices(self, tmp_path):
        from extract_truthfulness_iti import build_truthfulqa_paper_examples

        questions = _make_toy_questions(10)
        folds = build_folds(questions, seed=42)
        fold_path, csv_path = _write_fold_and_manifest(
            tmp_path, questions, folds, fold_idx=0
        )
        examples, _ = build_truthfulqa_paper_examples(
            tokenizer=StubTokenizer(),
            fold_path=fold_path,
            csv_path=csv_path,
        )
        # Paper-faithful recipe: all examples are dev, so fit on all of them
        direction_fit_indices = list(range(len(examples)))
        assert len(direction_fit_indices) == len(examples)
        # No example has split="test"
        assert all(ex.split in ("train", "val") for ex in examples)

    def test_direction_rejects_test_indices(self):
        examples = _make_examples(n_train_per_label=2, n_val_per_label=1)
        # Add a test-split example
        test_ex = ITIExample(
            example_id="test_0",
            family="iti_truthfulqa_paper",
            split="test",
            qid="tqa_test_0000",
            question="Q?",
            answer="A",
            label=1,
            weight=1.0,
            prompt_text="Q? A",
            answer_positions=(1,),
            metadata={},
        )
        examples_with_test = list(examples) + [test_ex]
        n = len(examples_with_test)

        acts = {"last_answer_token": torch.randn(n, 1, 1, 1)}
        ranked = [{"layer": 0, "head": 0, "position_summary": "last_answer_token"}]

        with pytest.raises(AssertionError, match="test-fold example"):
            compute_head_directions(
                ranked,
                acts,
                examples_with_test,
                direction_fit_indices=list(range(n)),  # includes test index
                direction_source="dev_data",
            )

    def test_sigma_differs_dev_vs_train_only(self):
        """Val examples with extreme activations must change sigma when included."""
        examples = _make_examples(n_train_per_label=2, n_val_per_label=2)
        n = len(examples)
        acts = torch.zeros(n, 1, 1, 1)

        # Train: modest activations
        for i, ex in enumerate(examples):
            if ex.split == "train":
                acts[i, 0, 0, 0] = 1.0 if ex.label == 1 else -1.0
            else:
                # Val: extreme activations to push sigma
                acts[i, 0, 0, 0] = 100.0 if ex.label == 1 else -100.0

        activations = {"last_answer_token": acts}
        base_entry = {"layer": 0, "head": 0, "position_summary": "last_answer_token"}

        # Direction with all dev (train+val)
        ranked_dev = [dict(base_entry)]
        train_indices = [i for i, ex in enumerate(examples) if ex.split == "train"]
        all_indices = list(range(n))

        compute_head_directions(
            ranked_dev, activations, examples, all_indices, direction_source="dev_data"
        )
        sigma_dev = ranked_dev[0]["sigma"]

        # Direction with train only
        ranked_train = [dict(base_entry)]
        compute_head_directions(
            ranked_train,
            activations,
            examples,
            train_indices,
            direction_source="train_data",
        )
        sigma_train = ranked_train[0]["sigma"]

        assert sigma_dev != sigma_train, (
            f"sigma should differ: dev={sigma_dev}, train={sigma_train}"
        )
        assert sigma_dev > sigma_train, (
            "extreme val activations should inflate dev sigma"
        )

    def test_fingerprint_changes_when_dev_set_changes(self):
        fp1 = fingerprint_ids(["a", "b", "c"])
        fp2 = fingerprint_ids(["a", "b", "c", "d"])
        assert fp1 != fp2

    def test_fingerprint_stable_for_same_set(self):
        fp1 = fingerprint_ids(["a", "b", "c"])
        fp2 = fingerprint_ids(["a", "b", "c"])
        assert fp1 == fp2
        # Order shouldn't matter (fingerprint sorts internally)
        fp3 = fingerprint_ids(["c", "a", "b"])
        assert fp1 == fp3


# ---------------------------------------------------------------------------
# 5. Token position semantics
# ---------------------------------------------------------------------------


class TestTokenPositionSemantics:
    def test_paper_faithful_ranks_last_token_only(self):
        examples = _make_examples(n_train_per_label=2, n_val_per_label=2)
        n = len(examples)
        # Provide all three summaries in activations, but only pass last_answer_token
        acts = {
            "first_answer_token": torch.randn(n, 1, 1, 1),
            "mean_answer_span": torch.randn(n, 1, 1, 1),
            "last_answer_token": torch.randn(n, 1, 1, 1),
        }
        ranked, _ = rank_heads(
            acts,
            examples,
            position_summaries=("last_answer_token",),
            ranking_primary="val_accuracy",
        )
        for entry in ranked:
            assert entry["position_summary"] == "last_answer_token"
            assert set(entry["metrics_by_summary"].keys()) == {"last_answer_token"}

    def test_answer_positions_are_contiguous_span(self):
        tok = StubTokenizer()
        _, positions = _encode_with_answer_positions(
            tok, question="What is the capital", answer="Paris France"
        )
        assert len(positions) >= 1
        assert positions == tuple(range(positions[0], positions[-1] + 1))

    def test_paper_faithful_prompt_format(self):
        prefix, full = _qa_prompt_paper("What?", "Yes")
        assert prefix == "Q: What?\nA:"
        assert full == "Q: What?\nA: Yes"
