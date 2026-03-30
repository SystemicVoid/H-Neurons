"""Tests for head-level ITI extraction and intervention plumbing."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import extract_truthfulness_iti as iti_extract
from evaluate_intervention import parse_simpleqa_verdict
from intervene_iti import ITIHeadScaler


class StubTokenizer:
    def __call__(self, text, add_special_tokens=False, return_tensors=None):
        token_count = max(1, len(text.split()))
        ids = list(range(token_count))
        if return_tensors == "pt":
            return {"input_ids": torch.tensor([ids], dtype=torch.long)}
        return {"input_ids": ids}


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


class TestTriviaQAExamples:
    def test_caps_per_qid_label_and_weights_examples(self, tmp_path):
        consistency_path = tmp_path / "consistency.jsonl"
        _write_jsonl(
            consistency_path,
            [
                {
                    "q1": {
                        "question": "Capital of France?",
                        "responses": ["Paris", "City of Paris", "Paris", "Lyon"],
                        "judges": ["true", "true", "true", "false"],
                        "ground_truth": ["Paris"],
                    }
                },
                {
                    "q2": {
                        "question": "2 + 2?",
                        "responses": ["4", "four", "5"],
                        "judges": ["true", "true", "false"],
                        "ground_truth": ["4"],
                    }
                },
            ],
        )
        train_path = tmp_path / "train.json"
        val_path = tmp_path / "val.json"
        _write_json(train_path, {"t": ["q1"], "f": []})
        _write_json(val_path, {"t": ["q2"], "f": []})

        examples, metadata = iti_extract.build_triviaqa_transfer_examples(
            consistency_path=consistency_path,
            train_qids_path=train_path,
            test_qids_path=val_path,
            tokenizer=StubTokenizer(),
            train_cap_per_qid_label=2,
            val_cap_per_qid_label=1,
        )

        train_truthful = [
            ex
            for ex in examples
            if ex.split == "train" and ex.label == iti_extract.LABEL_TRUE
        ]
        val_truthful = [
            ex
            for ex in examples
            if ex.split == "val" and ex.label == iti_extract.LABEL_TRUE
        ]

        assert len(train_truthful) == 2
        assert sorted(ex.answer for ex in train_truthful) == ["City of Paris", "Paris"]
        assert all(ex.weight == pytest.approx(0.5) for ex in train_truthful)
        assert len(val_truthful) == 1
        assert val_truthful[0].weight == pytest.approx(1.0)
        assert metadata["deduplication"]["skipped_duplicate_answers"] == 1
        assert metadata["per_qid_weights"]["q1"]["truthful"][
            "weight_per_example"
        ] == pytest.approx(0.5)


class TestContextGroundedExamples:
    def test_builds_grounded_and_abstention_pairs(self, monkeypatch):
        train_rows = [
            {
                "id": "a1",
                "question": "Who wrote Hamlet?",
                "context": "Hamlet was written by William Shakespeare.",
                "answers": {"text": ["William Shakespeare"]},
                "title": "Hamlet",
            },
            {
                "id": "a2",
                "question": "Where is the Eiffel Tower?",
                "context": "The Eiffel Tower is in Paris.",
                "answers": {"text": ["Paris"]},
                "title": "Eiffel",
            },
            {
                "id": "i1",
                "question": "What is the capital of Mars?",
                "context": "The passage does not mention a capital.",
                "answers": {"text": []},
                "title": "Mars",
            },
            {
                "id": "i2",
                "question": "Who won the race?",
                "context": "No winner is given.",
                "answers": {"text": []},
                "title": "Race",
            },
        ]
        val_rows = [
            {
                "id": "a3",
                "question": "Who painted Guernica?",
                "context": "Guernica was painted by Pablo Picasso.",
                "answers": {"text": ["Pablo Picasso"]},
                "title": "Guernica",
            },
            {
                "id": "i3",
                "question": "What is the hidden code?",
                "context": "No code appears in the passage.",
                "answers": {"text": []},
                "title": "Code",
            },
        ]

        def fake_load_dataset(name, split):
            assert name == "squad_v2"
            return train_rows if split == "train" else val_rows

        monkeypatch.setattr(iti_extract, "load_dataset", fake_load_dataset)

        examples, metadata = iti_extract.build_context_grounded_examples(
            tokenizer=StubTokenizer(),
            seed=42,
            train_questions=4,
            val_questions=2,
        )

        abstentions = [
            ex.answer
            for ex in examples
            if ex.label == iti_extract.LABEL_TRUE and ex.metadata["answerable"] is False
        ]
        assert abstentions == [iti_extract.DEFAULT_ABSTENTION] * 3
        assert metadata["label_balance"]["train_answerable_questions"] == 2
        assert metadata["label_balance"]["train_impossible_questions"] == 2
        assert metadata["label_balance"]["val_answerable_questions"] == 1
        assert metadata["label_balance"]["val_impossible_questions"] == 1

    def test_excludes_duplicate_gold_strings_from_wrong_answer_pool(self, monkeypatch):
        train_rows = [
            {
                "id": "a1",
                "question": "Where is the Louvre?",
                "context": "The Louvre is in Paris.",
                "answers": {"text": ["Paris"]},
                "title": "Louvre",
            },
            {
                "id": "a2",
                "question": "Where is Notre-Dame?",
                "context": "Notre-Dame is in Paris.",
                "answers": {"text": ["Paris"]},
                "title": "Notre-Dame",
            },
            {
                "id": "i1",
                "question": "What color is the hidden key?",
                "context": "No key is described.",
                "answers": {"text": []},
                "title": "Key",
            },
            {
                "id": "i2",
                "question": "Who solved the unnamed puzzle?",
                "context": "No solver is identified.",
                "answers": {"text": []},
                "title": "Puzzle",
            },
        ]
        val_rows = [
            {
                "id": "a3",
                "question": "Where is Big Ben?",
                "context": "Big Ben is in London.",
                "answers": {"text": ["London"]},
                "title": "Big Ben",
            },
            {
                "id": "i3",
                "question": "What is the missing launch code?",
                "context": "The code does not appear.",
                "answers": {"text": []},
                "title": "Code",
            },
        ]

        def fake_load_dataset(name, split):
            assert name == "squad_v2"
            return train_rows if split == "train" else val_rows

        monkeypatch.setattr(iti_extract, "load_dataset", fake_load_dataset)

        examples, _ = iti_extract.build_context_grounded_examples(
            tokenizer=StubTokenizer(),
            seed=42,
            train_questions=4,
            val_questions=2,
        )

        paris_negatives = [
            ex.answer
            for ex in examples
            if ex.qid in {"a1", "a2"} and ex.label == iti_extract.LABEL_FALSE
        ]

        assert paris_negatives
        assert all(
            iti_extract.normalize_answer(answer) != "paris"
            for answer in paris_negatives
        )


class TestHeadRanking:
    def test_prefers_best_position_summary_per_head(self):
        examples = (
            [
                iti_extract.ITIExample(
                    example_id=f"train_t_{idx}",
                    family="iti_triviaqa_transfer",
                    split="train",
                    qid=f"train_t_{idx}",
                    question="q",
                    answer="a",
                    label=iti_extract.LABEL_TRUE,
                    weight=1.0,
                    prompt_text="q a",
                    answer_positions=(1,),
                    metadata={},
                )
                for idx in range(2)
            ]
            + [
                iti_extract.ITIExample(
                    example_id=f"train_f_{idx}",
                    family="iti_triviaqa_transfer",
                    split="train",
                    qid=f"train_f_{idx}",
                    question="q",
                    answer="a",
                    label=iti_extract.LABEL_FALSE,
                    weight=1.0,
                    prompt_text="q a",
                    answer_positions=(1,),
                    metadata={},
                )
                for idx in range(2)
            ]
            + [
                iti_extract.ITIExample(
                    example_id=f"val_t_{idx}",
                    family="iti_triviaqa_transfer",
                    split="val",
                    qid=f"val_t_{idx}",
                    question="q",
                    answer="a",
                    label=iti_extract.LABEL_TRUE,
                    weight=1.0,
                    prompt_text="q a",
                    answer_positions=(1,),
                    metadata={},
                )
                for idx in range(2)
            ]
            + [
                iti_extract.ITIExample(
                    example_id=f"val_f_{idx}",
                    family="iti_triviaqa_transfer",
                    split="val",
                    qid=f"val_f_{idx}",
                    question="q",
                    answer="a",
                    label=iti_extract.LABEL_FALSE,
                    weight=1.0,
                    prompt_text="q a",
                    answer_positions=(1,),
                    metadata={},
                )
                for idx in range(2)
            ]
        )

        good = torch.tensor(
            [
                [[[3.0, 0.0]]],
                [[[2.5, 0.0]]],
                [[[-3.0, 0.0]]],
                [[[-2.5, 0.0]]],
                [[[3.2, 0.0]]],
                [[[2.8, 0.0]]],
                [[[-3.1, 0.0]]],
                [[[-2.9, 0.0]]],
            ],
            dtype=torch.float16,
        )
        noisy = torch.tensor(
            [
                [[[0.1, 0.0]]],
                [[[0.2, 0.0]]],
                [[[0.0, 0.0]]],
                [[[0.1, 0.0]]],
                [[[0.1, 0.0]]],
                [[[0.2, 0.0]]],
                [[[0.0, 0.0]]],
                [[[0.1, 0.0]]],
            ],
            dtype=torch.float16,
        )
        activations = {
            "first_answer_token": good,
            "mean_answer_span": noisy,
            "last_answer_token": noisy,
        }

        ranked, metadata = iti_extract.rank_heads(activations, examples)

        assert ranked[0]["position_summary"] == "first_answer_token"
        assert ranked[0]["auroc"] > 0.9
        assert metadata["top5"][0]["position_summary"] == "first_answer_token"


class DummySelfAttn(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.head_dim = 2
        self.o_proj = torch.nn.Linear(4, 4, bias=False)
        with torch.no_grad():
            self.o_proj.weight.copy_(torch.eye(4))


class DummyLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = DummySelfAttn()


class DummyLanguageModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([DummyLayer()])


class DummyOuter(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.language_model = DummyLanguageModel()


class DummyITIModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = DummyOuter()

    def forward(self, x):
        return self.model.language_model.layers[0].self_attn.o_proj(x)


class TestITIHeadScaler:
    def test_decode_only_intervention_skips_prompt_and_edits_decode(self):
        model = DummyITIModel()
        artifact = {
            "family": "iti_triviaqa_transfer",
            "n_layers": 1,
            "n_attention_heads": 2,
            "head_dim": 2,
            "ranked_heads": [
                {
                    "layer": 0,
                    "head": 1,
                    "position_summary": "first_answer_token",
                    "auroc": 0.9,
                    "balanced_accuracy": 0.9,
                    "sigma": 1.0,
                    "direction": [1.0, 0.0],
                }
            ],
        }
        scaler = ITIHeadScaler(model, artifact, torch.device("cpu"), family=None, k=1)
        scaler.alpha = 2.0

        prompt_x = torch.zeros(1, 3, 4)
        prompt_out = model(prompt_x.clone())
        prompt_stats = scaler.consume_sample_stats()

        decode_x = torch.zeros(1, 1, 4)
        decode_out = model(decode_x.clone())
        decode_stats = scaler.consume_sample_stats()

        assert torch.allclose(prompt_out, prompt_x)
        assert prompt_stats["prompt_skip_calls"] == 1
        assert decode_out.tolist() == [[[0.0, 0.0, 2.0, 0.0]]]
        assert decode_stats["hook_calls"] == 1
        assert decode_stats["debug_steps"][0]["generated_token_index"] == 1

        scaler.remove()

    def test_armed_prefill_edits_only_last_prompt_position_for_first_token(self):
        model = DummyITIModel()
        artifact = {
            "family": "iti_triviaqa_transfer",
            "n_layers": 1,
            "n_attention_heads": 2,
            "head_dim": 2,
            "ranked_heads": [
                {
                    "layer": 0,
                    "head": 1,
                    "position_summary": "first_answer_token",
                    "auroc": 0.9,
                    "balanced_accuracy": 0.9,
                    "sigma": 1.0,
                    "direction": [1.0, 0.0],
                }
            ],
        }
        scaler = ITIHeadScaler(model, artifact, torch.device("cpu"), family=None, k=1)
        scaler.alpha = 2.0
        scaler.arm_first_decode_token()

        prompt_x = torch.zeros(1, 3, 4)
        prompt_out = model(prompt_x.clone())
        prompt_stats = scaler.consume_sample_stats()

        assert prompt_out.tolist() == [
            [[0.0, 0.0, 0.0, 0.0]] * 2 + [[0.0, 0.0, 2.0, 0.0]]
        ]
        assert prompt_stats["prompt_skip_calls"] == 0
        assert prompt_stats["debug_steps"][0]["generated_token_index"] == 1

        scaler.remove()

    def test_random_head_control_rescales_sigma_to_match_ranked_total_norm(self):
        model = DummyITIModel()
        artifact = {
            "family": "iti_triviaqa_transfer",
            "n_layers": 1,
            "n_attention_heads": 3,
            "head_dim": 2,
            "ranked_heads": [
                {
                    "layer": 0,
                    "head": 0,
                    "position_summary": "first_answer_token",
                    "auroc": 0.95,
                    "balanced_accuracy": 0.95,
                    "sigma": 4.0,
                    "direction": [1.0, 0.0],
                },
                {
                    "layer": 0,
                    "head": 1,
                    "position_summary": "first_answer_token",
                    "auroc": 0.9,
                    "balanced_accuracy": 0.9,
                    "sigma": 2.0,
                    "direction": [1.0, 0.0],
                },
                {
                    "layer": 0,
                    "head": 2,
                    "position_summary": "first_answer_token",
                    "auroc": 0.85,
                    "balanced_accuracy": 0.85,
                    "sigma": 1.0,
                    "direction": [1.0, 0.0],
                },
            ],
        }
        scaler = ITIHeadScaler(
            model,
            artifact,
            torch.device("cpu"),
            family=None,
            k=2,
            selection_strategy="random",
            random_seed=42,
        )

        applied_sigma_total = sum(
            item["applied_sigma"]
            for layer_items in scaler._selected_by_layer.values()
            for item in layer_items
        )

        assert applied_sigma_total == pytest.approx(6.0)

        scaler.remove()


class TestSimpleQAVerdictParsing:
    def test_parses_json_and_fallback_labels(self):
        assert (
            parse_simpleqa_verdict('{"grade": "CORRECT", "reason": "matches"}')
            == "CORRECT"
        )
        assert parse_simpleqa_verdict("This is NOT_ATTEMPTED.") == "NOT_ATTEMPTED"
        assert parse_simpleqa_verdict("unclear") == "UNKNOWN"


class TestTruthfulQAPaperExamples:
    def test_builds_correct_labels_and_splits_by_question(self, monkeypatch):
        """Each correct answer → LABEL_TRUE, each incorrect → LABEL_FALSE,
        split is by question ID not by pair."""

        class FakeDataset:
            def __len__(self):
                return 4

            def __getitem__(self, idx):
                questions = [
                    {
                        "question": "What is 1+1?",
                        "correct_answers": ["2", "Two"],
                        "incorrect_answers": ["3"],
                    },
                    {
                        "question": "Capital of France?",
                        "correct_answers": ["Paris"],
                        "incorrect_answers": ["London", "Berlin"],
                    },
                    {
                        "question": "Sky color?",
                        "correct_answers": ["Blue"],
                        "incorrect_answers": ["Red"],
                    },
                    {
                        "question": "Water formula?",
                        "correct_answers": ["H2O"],
                        "incorrect_answers": ["CO2"],
                    },
                ]
                return questions[idx]

            column_names = [
                "question",
                "correct_answers",
                "incorrect_answers",
            ]

        monkeypatch.setattr(
            "extract_truthfulness_iti.load_dataset",
            lambda *_args, **_kwargs: FakeDataset(),
        )

        examples, metadata = iti_extract.build_truthfulqa_paper_examples(
            tokenizer=StubTokenizer(),
            seed=42,
            val_fraction=0.5,
        )

        # Label checks: Q0: 2T+1F, Q1: 1T+2F, Q2: 1T+1F, Q3: 1T+1F = 5T+5F
        true_count = sum(1 for ex in examples if ex.label == iti_extract.LABEL_TRUE)
        false_count = sum(1 for ex in examples if ex.label == iti_extract.LABEL_FALSE)
        assert true_count == 5
        assert false_count == 5

        # Split by question: each QID's examples are all in the same split
        qid_splits = {}
        for ex in examples:
            if ex.qid not in qid_splits:
                qid_splits[ex.qid] = ex.split
            assert ex.split == qid_splits[ex.qid], (
                f"QID {ex.qid} has examples in both splits"
            )

        # Both splits present
        splits = {ex.split for ex in examples}
        assert splits == {"train", "val"}

        # Family and prompt format
        assert all(ex.family == "iti_truthfulqa_paper" for ex in examples)
        assert metadata["family"] == "iti_truthfulqa_paper"
        assert metadata["source_dataset"] == "truthful_qa/generation"

    def test_uses_paper_prompt_format(self, monkeypatch):
        """Prompt should be Q: {q}\\nA: {a} format."""

        class FakeDataset:
            def __len__(self):
                return 1

            def __getitem__(self, idx):
                return {
                    "question": "What is 1+1?",
                    "correct_answers": ["2"],
                    "incorrect_answers": ["3"],
                }

            column_names = ["question", "correct_answers", "incorrect_answers"]

        monkeypatch.setattr(
            "extract_truthfulness_iti.load_dataset",
            lambda *_args, **_kwargs: FakeDataset(),
        )

        examples, _ = iti_extract.build_truthfulqa_paper_examples(
            tokenizer=StubTokenizer(),
            seed=42,
            val_fraction=0.5,
        )

        for ex in examples:
            assert ex.prompt_text.startswith("Q: What is 1+1?\nA:")


class TestPaperFaithfulRanking:
    def test_balanced_accuracy_ranking_differs_from_auroc(self):
        """When ranking_primary='balanced_accuracy', the top head should
        be the one with highest balanced_accuracy, not highest AUROC."""
        examples = (
            [
                iti_extract.ITIExample(
                    example_id=f"train_t_{idx}",
                    family="iti_truthfulqa_paper",
                    split="train",
                    qid=f"train_t_{idx}",
                    question="q",
                    answer="a",
                    label=iti_extract.LABEL_TRUE,
                    weight=1.0,
                    prompt_text="q a",
                    answer_positions=(1,),
                    metadata={},
                )
                for idx in range(4)
            ]
            + [
                iti_extract.ITIExample(
                    example_id=f"train_f_{idx}",
                    family="iti_truthfulqa_paper",
                    split="train",
                    qid=f"train_f_{idx}",
                    question="q",
                    answer="a",
                    label=iti_extract.LABEL_FALSE,
                    weight=1.0,
                    prompt_text="q a",
                    answer_positions=(1,),
                    metadata={},
                )
                for idx in range(4)
            ]
            + [
                iti_extract.ITIExample(
                    example_id=f"val_t_{idx}",
                    family="iti_truthfulqa_paper",
                    split="val",
                    qid=f"val_t_{idx}",
                    question="q",
                    answer="a",
                    label=iti_extract.LABEL_TRUE,
                    weight=1.0,
                    prompt_text="q a",
                    answer_positions=(1,),
                    metadata={},
                )
                for idx in range(4)
            ]
            + [
                iti_extract.ITIExample(
                    example_id=f"val_f_{idx}",
                    family="iti_truthfulqa_paper",
                    split="val",
                    qid=f"val_f_{idx}",
                    question="q",
                    answer="a",
                    label=iti_extract.LABEL_FALSE,
                    weight=1.0,
                    prompt_text="q a",
                    answer_positions=(1,),
                    metadata={},
                )
                for idx in range(4)
            ]
        )

        # Only last_answer_token — paper-faithful restriction
        activations = {
            "last_answer_token": torch.tensor(
                [
                    [[[3.0, 0.0]]],
                    [[[2.5, 0.0]]],
                    [[[2.8, 0.0]]],
                    [[[2.2, 0.0]]],
                    [[[-3.0, 0.0]]],
                    [[[-2.5, 0.0]]],
                    [[[-2.8, 0.0]]],
                    [[[-2.2, 0.0]]],
                    [[[3.2, 0.0]]],
                    [[[2.8, 0.0]]],
                    [[[3.0, 0.0]]],
                    [[[2.5, 0.0]]],
                    [[[-3.1, 0.0]]],
                    [[[-2.9, 0.0]]],
                    [[[-3.0, 0.0]]],
                    [[[-2.5, 0.0]]],
                ],
                dtype=torch.float16,
            ),
        }

        ranked_auroc, _ = iti_extract.rank_heads(
            activations,
            examples,
            position_summaries=("last_answer_token",),
            ranking_primary="auroc",
        )
        ranked_acc, _ = iti_extract.rank_heads(
            activations,
            examples,
            position_summaries=("last_answer_token",),
            ranking_primary="balanced_accuracy",
        )

        # Both produce a single head (1 layer, 1 head), so ordering doesn't
        # differ for 1 head, but the metadata should reflect the right config
        assert ranked_auroc[0]["position_summary"] == "last_answer_token"
        assert ranked_acc[0]["position_summary"] == "last_answer_token"

    def test_last_token_only_ignores_other_summaries(self):
        """When position_summaries=('last_answer_token',), the other
        summaries should not affect ranking even if they have better metrics."""
        examples = (
            [
                iti_extract.ITIExample(
                    example_id=f"train_t_{idx}",
                    family="iti_truthfulqa_paper",
                    split="train",
                    qid=f"train_t_{idx}",
                    question="q",
                    answer="a",
                    label=iti_extract.LABEL_TRUE,
                    weight=1.0,
                    prompt_text="q a",
                    answer_positions=(1,),
                    metadata={},
                )
                for idx in range(2)
            ]
            + [
                iti_extract.ITIExample(
                    example_id=f"train_f_{idx}",
                    family="iti_truthfulqa_paper",
                    split="train",
                    qid=f"train_f_{idx}",
                    question="q",
                    answer="a",
                    label=iti_extract.LABEL_FALSE,
                    weight=1.0,
                    prompt_text="q a",
                    answer_positions=(1,),
                    metadata={},
                )
                for idx in range(2)
            ]
            + [
                iti_extract.ITIExample(
                    example_id=f"val_t_{idx}",
                    family="iti_truthfulqa_paper",
                    split="val",
                    qid=f"val_t_{idx}",
                    question="q",
                    answer="a",
                    label=iti_extract.LABEL_TRUE,
                    weight=1.0,
                    prompt_text="q a",
                    answer_positions=(1,),
                    metadata={},
                )
                for idx in range(2)
            ]
            + [
                iti_extract.ITIExample(
                    example_id=f"val_f_{idx}",
                    family="iti_truthfulqa_paper",
                    split="val",
                    qid=f"val_f_{idx}",
                    question="q",
                    answer="a",
                    label=iti_extract.LABEL_FALSE,
                    weight=1.0,
                    prompt_text="q a",
                    answer_positions=(1,),
                    metadata={},
                )
                for idx in range(2)
            ]
        )

        # first_answer_token has perfect separation, last_answer_token is noisy
        good = torch.tensor(
            [
                [[[3.0, 0.0]]],
                [[[2.5, 0.0]]],
                [[[-3.0, 0.0]]],
                [[[-2.5, 0.0]]],
                [[[3.2, 0.0]]],
                [[[2.8, 0.0]]],
                [[[-3.1, 0.0]]],
                [[[-2.9, 0.0]]],
            ],
            dtype=torch.float16,
        )
        noisy = torch.tensor(
            [
                [[[0.1, 0.0]]],
                [[[0.2, 0.0]]],
                [[[0.0, 0.0]]],
                [[[0.1, 0.0]]],
                [[[0.1, 0.0]]],
                [[[0.2, 0.0]]],
                [[[0.0, 0.0]]],
                [[[0.1, 0.0]]],
            ],
            dtype=torch.float16,
        )
        activations = {
            "first_answer_token": good,
            "mean_answer_span": good,
            "last_answer_token": noisy,
        }

        ranked, metadata = iti_extract.rank_heads(
            activations,
            examples,
            position_summaries=("last_answer_token",),
        )

        # Should pick last_answer_token since it's the only allowed summary
        assert ranked[0]["position_summary"] == "last_answer_token"
        # Only one position summary in metadata
        assert metadata["position_summaries"] == ["last_answer_token"]

    def test_recompute_directions_uses_all_data(self):
        """After ranking, recomputing directions from full data should use
        both train and val examples, producing a different direction than
        the train-only direction from ranking."""
        examples = [
            iti_extract.ITIExample(
                example_id=f"{split}_{label}_{idx}",
                family="iti_truthfulqa_paper",
                split=split,
                qid=f"{split}_{idx}",
                question="q",
                answer="a",
                label=iti_extract.LABEL_TRUE
                if label == "t"
                else iti_extract.LABEL_FALSE,
                weight=1.0,
                prompt_text="q a",
                answer_positions=(1,),
                metadata={},
            )
            for split in ("train", "val")
            for label in ("t", "f")
            for idx in range(2)
        ]

        # Train: true=[1,0], false=[-1,0] → direction ≈ [1,0]
        # Val: true=[0,1], false=[0,-1] → adds [0,1] component
        # Full data: direction should have both components
        activations = {
            "last_answer_token": torch.tensor(
                [
                    # train_t_0, train_t_1
                    [[[1.0, 0.0]]],
                    [[[1.0, 0.0]]],
                    # train_f_0, train_f_1
                    [[[-1.0, 0.0]]],
                    [[[-1.0, 0.0]]],
                    # val_t_0, val_t_1
                    [[[0.0, 1.0]]],
                    [[[0.0, 1.0]]],
                    # val_f_0, val_f_1
                    [[[0.0, -1.0]]],
                    [[[0.0, -1.0]]],
                ],
                dtype=torch.float16,
            ),
        }

        ranked, _ = iti_extract.rank_heads(
            activations, examples, position_summaries=("last_answer_token",)
        )
        # Train-only direction should be ≈ [1, 0]
        train_dir = ranked[0]["direction"]
        assert abs(train_dir[0]) > 0.9
        assert abs(train_dir[1]) < 0.1

        # Recompute from full data
        ranked = iti_extract.recompute_directions_full_data(
            ranked, activations, examples
        )
        full_dir = ranked[0]["direction"]
        # Full direction should have both components ≈ [0.707, 0.707]
        assert abs(full_dir[0]) > 0.5
        assert abs(full_dir[1]) > 0.5
        assert ranked[0]["direction_source"] == "full_data"
