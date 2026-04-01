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

    def test_first_token_only_scope_stops_after_token_one(self):
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
        scaler = ITIHeadScaler(
            model,
            artifact,
            torch.device("cpu"),
            family=None,
            k=1,
            decode_scope="first_token_only",
        )
        scaler.alpha = 2.0

        scaler.arm_first_decode_token()
        prefill_out = model(torch.zeros(1, 3, 4))
        prefill_stats = scaler.consume_sample_stats()

        decode_out = model(torch.zeros(1, 1, 4))
        decode_stats = scaler.consume_sample_stats()

        assert prefill_out.tolist() == [
            [[0.0, 0.0, 0.0, 0.0]] * 2 + [[0.0, 0.0, 2.0, 0.0]]
        ]
        assert prefill_stats["decode_scope"] == "first_token_only"
        assert prefill_stats["scope_skip_calls"] == 0
        assert decode_out.tolist() == [[[0.0, 0.0, 0.0, 0.0]]]
        assert decode_stats["scope_skip_calls"] == 1

        scaler.remove()

    def test_arming_new_decode_sequence_restarts_scope_counting(self):
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
        scaler = ITIHeadScaler(
            model,
            artifact,
            torch.device("cpu"),
            family=None,
            k=1,
            decode_scope="first_token_only",
        )
        scaler.alpha = 2.0

        scaler.arm_first_decode_token()
        first_out = model(torch.zeros(1, 3, 4))
        first_stats = scaler.consume_sample_stats()

        scaler.arm_first_decode_token()
        second_out = model(torch.zeros(1, 2, 4))
        second_stats = scaler.consume_sample_stats()

        assert first_out.tolist() == [
            [[0.0, 0.0, 0.0, 0.0]] * 2 + [[0.0, 0.0, 2.0, 0.0]]
        ]
        assert second_out.tolist() == [[[0.0, 0.0, 0.0, 0.0]] + [[0.0, 0.0, 2.0, 0.0]]]
        assert first_stats["debug_steps"][0]["generated_token_index"] == 1
        assert second_stats["debug_steps"][0]["generated_token_index"] == 1

        scaler.remove()

    def test_first_three_tokens_scope_stops_after_third_token(self):
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
        scaler = ITIHeadScaler(
            model,
            artifact,
            torch.device("cpu"),
            family=None,
            k=1,
            decode_scope="first_3_tokens",
        )
        scaler.alpha = 2.0

        scaler.arm_first_decode_token()
        outputs = [model(torch.zeros(1, 3, 4))]
        stats = [scaler.consume_sample_stats()]
        for _ in range(3):
            outputs.append(model(torch.zeros(1, 1, 4)))
            stats.append(scaler.consume_sample_stats())

        assert outputs[0].tolist() == [
            [[0.0, 0.0, 0.0, 0.0]] * 2 + [[0.0, 0.0, 2.0, 0.0]]
        ]
        assert outputs[1].tolist() == [[[0.0, 0.0, 2.0, 0.0]]]
        assert outputs[2].tolist() == [[[0.0, 0.0, 2.0, 0.0]]]
        assert outputs[3].tolist() == [[[0.0, 0.0, 0.0, 0.0]]]
        assert stats[2]["debug_steps"][0]["generated_token_index"] == 3
        assert stats[3]["scope_skip_calls"] == 1

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


def _write_fold_fixtures(tmp_path, questions, train_idxs, val_idxs, test_idxs):
    """Write canonical manifest, CSV, and fold file for test fixtures.

    ``train_idxs``, ``val_idxs``, ``test_idxs`` are positional indices into
    the ``questions`` list.  Returns (fold_path, csv_path).
    """
    import csv as csv_module

    from build_truthfulqa_splits import fingerprint_ids, stable_question_id

    # Write authoritative CSV
    csv_path = tmp_path / "questions.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv_module.writer(f)
        writer.writerow(["Question", "Correct Answers", "Incorrect Answers"])
        for q in questions:
            writer.writerow(
                [
                    q["question"],
                    "; ".join(q["correct_answers"]),
                    "; ".join(q["incorrect_answers"]),
                ]
            )

    canon_questions = []
    for i, q in enumerate(questions):
        canon_questions.append(
            {
                "stable_id": stable_question_id(q["question"]),
                "csv_idx": i,
                "question_text": q["question"],
            }
        )

    sid = {i: canon_questions[i]["stable_id"] for i in range(len(questions))}
    all_ids = [c["stable_id"] for c in canon_questions]
    canon = {
        "version": 1,
        "source_csv": str(csv_path),
        "n_questions": len(questions),
        "fingerprint": fingerprint_ids(all_ids),
        "questions": canon_questions,
    }
    canon_path = tmp_path / "canonical.json"
    canon_path.write_text(json.dumps(canon), encoding="utf-8")

    fold = {
        "version": 1,
        "seed": 42,
        "fold": 0,
        "canonical_manifest": str(canon_path),
        "canonical_fingerprint": canon["fingerprint"],
        "dev": {
            "train": [sid[i] for i in train_idxs],
            "val": [sid[i] for i in val_idxs],
        },
        "test": [sid[i] for i in test_idxs],
        "counts": {
            "train": len(train_idxs),
            "val": len(val_idxs),
            "test": len(test_idxs),
        },
    }
    fold_path = tmp_path / "fold.json"
    fold_path.write_text(json.dumps(fold), encoding="utf-8")
    return fold_path, csv_path


class TestTruthfulQAPaperExamples:
    _questions = [
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

    def test_builds_correct_labels_and_splits_by_question(self, tmp_path):
        """Each correct answer → LABEL_TRUE, each incorrect → LABEL_FALSE,
        split is by question ID not by pair."""
        # Q0, Q1 = dev (Q0 train, Q1 val), Q2, Q3 = test (excluded)
        fold_path, csv_path = _write_fold_fixtures(
            tmp_path,
            self._questions,
            train_idxs=[0],
            val_idxs=[1],
            test_idxs=[2, 3],
        )

        examples, metadata = iti_extract.build_truthfulqa_paper_examples(
            tokenizer=StubTokenizer(),
            fold_path=fold_path,
            csv_path=csv_path,
        )

        # Only dev questions: Q0: 2T+1F, Q1: 1T+2F = 3T+3F
        true_count = sum(1 for ex in examples if ex.label == iti_extract.LABEL_TRUE)
        false_count = sum(1 for ex in examples if ex.label == iti_extract.LABEL_FALSE)
        assert true_count == 3
        assert false_count == 3

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

        # Family and source metadata
        assert all(ex.family == "iti_truthfulqa_paperfaithful" for ex in examples)
        assert metadata["family"] == "iti_truthfulqa_paperfaithful"
        assert "questions.csv" in metadata["source_dataset"]

    def test_excludes_test_fold_questions(self, tmp_path):
        """Questions in the test fold must not appear in examples."""
        fold_path, csv_path = _write_fold_fixtures(
            tmp_path,
            self._questions,
            train_idxs=[0],
            val_idxs=[1],
            test_idxs=[2, 3],
        )

        examples, metadata = iti_extract.build_truthfulqa_paper_examples(
            tokenizer=StubTokenizer(),
            fold_path=fold_path,
            csv_path=csv_path,
        )

        # No examples should reference test-fold questions
        test_questions = {"Sky color?", "Water formula?"}
        for ex in examples:
            assert ex.question not in test_questions, (
                f"Test-fold question {ex.question!r} leaked into examples"
            )
        assert metadata["question_counts"]["test_excluded"] == 2

    def test_uses_paper_prompt_format(self, tmp_path):
        """Prompt should be Q: {q}\\nA: {a} format."""
        questions = [self._questions[0]]
        fold_path, csv_path = _write_fold_fixtures(
            tmp_path,
            questions,
            train_idxs=[0],
            val_idxs=[],
            test_idxs=[],
        )

        examples, _ = iti_extract.build_truthfulqa_paper_examples(
            tokenizer=StubTokenizer(),
            fold_path=fold_path,
            csv_path=csv_path,
        )

        for ex in examples:
            assert ex.prompt_text.startswith("Q: What is 1+1?\nA:")


class TestPaperFaithfulRanking:
    def test_val_accuracy_ranking_available(self):
        """When ranking_primary='val_accuracy', rank_heads should use plain
        accuracy (not balanced_accuracy) as the primary sort key."""
        examples = (
            [
                iti_extract.ITIExample(
                    example_id=f"train_t_{idx}",
                    family="iti_truthfulqa_paperfaithful",
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
                    family="iti_truthfulqa_paperfaithful",
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
                    family="iti_truthfulqa_paperfaithful",
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
                    family="iti_truthfulqa_paperfaithful",
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
        ranked_val_acc, _ = iti_extract.rank_heads(
            activations,
            examples,
            position_summaries=("last_answer_token",),
            ranking_primary="val_accuracy",
        )

        # Both produce a single head (1 layer, 1 head), so ordering doesn't
        # differ for 1 head, but the metadata should reflect the right fields
        assert ranked_auroc[0]["position_summary"] == "last_answer_token"
        assert ranked_val_acc[0]["position_summary"] == "last_answer_token"
        # val_accuracy field should be present
        assert "val_accuracy" in ranked_val_acc[0]

    def test_last_token_only_ignores_other_summaries(self):
        """When position_summaries=('last_answer_token',), the other
        summaries should not affect ranking even if they have better metrics."""
        examples = (
            [
                iti_extract.ITIExample(
                    example_id=f"train_t_{idx}",
                    family="iti_truthfulqa_paperfaithful",
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
                    family="iti_truthfulqa_paperfaithful",
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
                    family="iti_truthfulqa_paperfaithful",
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
                    family="iti_truthfulqa_paperfaithful",
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

    def test_compute_head_directions_with_explicit_indices(self):
        """rank_heads returns no direction; compute_head_directions adds it
        using only the specified fit indices."""
        import copy

        examples = [
            iti_extract.ITIExample(
                example_id=f"{split}_{label}_{idx}",
                family="iti_truthfulqa_paperfaithful",
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
        # rank_heads must not include direction or sigma
        assert "direction" not in ranked[0]
        assert "sigma" not in ranked[0]

        # Train-only direction ≈ [1, 0]
        train_indices = [i for i, ex in enumerate(examples) if ex.split == "train"]
        ranked_train = copy.deepcopy(ranked)
        iti_extract.compute_head_directions(
            ranked_train,
            activations,
            examples,
            train_indices,
            direction_source="train_data",
        )
        train_dir = ranked_train[0]["direction"]
        assert abs(train_dir[0]) > 0.9
        assert abs(train_dir[1]) < 0.1
        assert ranked_train[0]["direction_source"] == "train_data"

        # Dev direction (train+val) ≈ [0.707, 0.707]
        all_indices = list(range(len(examples)))
        ranked_dev = copy.deepcopy(ranked)
        iti_extract.compute_head_directions(
            ranked_dev,
            activations,
            examples,
            all_indices,
            direction_source="dev_data",
        )
        dev_dir = ranked_dev[0]["direction"]
        assert abs(dev_dir[0]) > 0.5
        assert abs(dev_dir[1]) > 0.5
        assert ranked_dev[0]["direction_source"] == "dev_data"


class TestCanonicalManifest:
    """Tests for stable question ID generation."""

    def test_stable_id_deterministic(self):
        from build_truthfulqa_splits import stable_question_id

        id1 = stable_question_id("What happens if you eat watermelon seeds?")
        id2 = stable_question_id("What happens if you eat watermelon seeds?")
        assert id1 == id2

    def test_stable_id_case_insensitive(self):
        from build_truthfulqa_splits import stable_question_id

        assert stable_question_id("Hello?") == stable_question_id("hello?")

    def test_stable_id_whitespace_normalized(self):
        from build_truthfulqa_splits import stable_question_id

        assert stable_question_id("  Hello? ") == stable_question_id("Hello?")
        assert stable_question_id("a  b") == stable_question_id("a b")

    def test_stable_id_format(self):
        from build_truthfulqa_splits import stable_question_id

        sid = stable_question_id("test question?")
        assert sid.startswith("tqa_")
        assert len(sid) == 16  # "tqa_" + 12 hex chars


class TestFoldIntegrity:
    """Tests for 2-fold CV split files."""

    def test_no_overlap_between_dev_and_test(self):
        from build_truthfulqa_splits import build_folds

        questions = [{"stable_id": f"tqa_{i:012x}", "csv_idx": i} for i in range(20)]
        folds = build_folds(questions, seed=42)
        for f in folds:
            dev = set(f["dev"]["train"]) | set(f["dev"]["val"])
            test = set(f["test"])
            assert dev.isdisjoint(test), f"Fold {f['fold']}: dev/test overlap"

    def test_fold_symmetry(self):
        from build_truthfulqa_splits import build_folds

        questions = [{"stable_id": f"tqa_{i:012x}", "csv_idx": i} for i in range(20)]
        folds = build_folds(questions, seed=42)
        dev0 = set(folds[0]["dev"]["train"]) | set(folds[0]["dev"]["val"])
        dev1 = set(folds[1]["dev"]["train"]) | set(folds[1]["dev"]["val"])
        assert set(folds[0]["test"]) == dev1
        assert set(folds[1]["test"]) == dev0

    def test_full_coverage(self):
        from build_truthfulqa_splits import build_folds

        questions = [{"stable_id": f"tqa_{i:012x}", "csv_idx": i} for i in range(20)]
        folds = build_folds(questions, seed=42)
        all_ids = {q["stable_id"] for q in questions}
        for f in folds:
            covered = set(f["dev"]["train"]) | set(f["dev"]["val"]) | set(f["test"])
            assert covered == all_ids

    def test_train_val_disjoint(self):
        from build_truthfulqa_splits import build_folds

        questions = [{"stable_id": f"tqa_{i:012x}", "csv_idx": i} for i in range(20)]
        folds = build_folds(questions, seed=42)
        for f in folds:
            assert set(f["dev"]["train"]).isdisjoint(set(f["dev"]["val"]))


class TestDirectionFitScope:
    """Tests for direction recomputation scope."""

    def test_rejects_test_split_in_fit_indices(self):
        """compute_head_directions must reject fit indices pointing to test examples."""
        examples = [
            iti_extract.ITIExample(
                example_id="train_0",
                family="test",
                split="train",
                qid="q0",
                question="q",
                answer="a",
                label=iti_extract.LABEL_TRUE,
                weight=1.0,
                prompt_text="q a",
                answer_positions=(1,),
                metadata={},
            ),
            iti_extract.ITIExample(
                example_id="test_0",
                family="test",
                split="test",
                qid="q1",
                question="q",
                answer="a",
                label=iti_extract.LABEL_TRUE,
                weight=1.0,
                prompt_text="q a",
                answer_positions=(1,),
                metadata={},
            ),
        ]
        with pytest.raises(AssertionError, match="test-fold example"):
            iti_extract.compute_head_directions(
                [], {}, examples, [0, 1], direction_source="dev_data"
            )


class TestLeakageBarrierMetadata:
    """Tests for question ID lists and audit in extraction metadata."""

    _questions = [
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

    def test_metadata_contains_question_id_lists(self, tmp_path):
        fold_path, csv_path = _write_fold_fixtures(
            tmp_path,
            self._questions,
            train_idxs=[0],
            val_idxs=[1],
            test_idxs=[2, 3],
        )

        _, metadata = iti_extract.build_truthfulqa_paper_examples(
            tokenizer=StubTokenizer(),
            fold_path=fold_path,
            csv_path=csv_path,
        )

        assert "question_ids_train" in metadata
        assert "question_ids_val" in metadata
        assert "question_ids_dev" in metadata
        assert "question_ids_test" in metadata
        assert len(metadata["question_ids_train"]) == 1
        assert len(metadata["question_ids_val"]) == 1
        assert len(metadata["question_ids_dev"]) == 2
        assert len(metadata["question_ids_test"]) == 2

        # dev = train ∪ val
        assert set(metadata["question_ids_dev"]) == (
            set(metadata["question_ids_train"]) | set(metadata["question_ids_val"])
        )
        # dev ∩ test = ∅
        assert set(metadata["question_ids_dev"]).isdisjoint(
            set(metadata["question_ids_test"])
        )

    def test_metadata_has_fit_source_fields(self, tmp_path):
        fold_path, csv_path = _write_fold_fixtures(
            tmp_path,
            self._questions,
            train_idxs=[0],
            val_idxs=[1],
            test_idxs=[2, 3],
        )

        _, metadata = iti_extract.build_truthfulqa_paper_examples(
            tokenizer=StubTokenizer(),
            fold_path=fold_path,
            csv_path=csv_path,
        )

        assert metadata["direction_fit_source"] == "dev_only"
        assert metadata["sigma_fit_source"] == "dev_only"
        assert metadata["head_ranking_metric"] == "val_accuracy"
        assert metadata["truthfulqa_manifest_fingerprint"] != ""
        assert metadata["fold_file_fingerprint"] != ""

    def test_metadata_split_fingerprints_include_test(self, tmp_path):
        fold_path, csv_path = _write_fold_fixtures(
            tmp_path,
            self._questions,
            train_idxs=[0],
            val_idxs=[1],
            test_idxs=[2, 3],
        )

        _, metadata = iti_extract.build_truthfulqa_paper_examples(
            tokenizer=StubTokenizer(),
            fold_path=fold_path,
            csv_path=csv_path,
        )

        fp = metadata["split_fingerprint"]
        assert "test_qids" in fp
        assert "dev_qids" in fp
        assert len(fp["test_qids"]) == 16  # hex fingerprint


class TestAuditSplitLeakage:
    """Tests for the audit_split_leakage utility."""

    def test_clean_metadata_passes_audit(self):
        from utils import audit_split_leakage

        meta = {
            "question_ids_train": ["q1", "q2"],
            "question_ids_val": ["q3"],
            "question_ids_dev": ["q1", "q2", "q3"],
            "question_ids_test": ["q4", "q5"],
            "direction_fit_source": "dev_only",
            "sigma_fit_source": "dev_only",
        }
        audit = audit_split_leakage(meta)
        assert audit["leakage_detected"] is False
        assert audit["counts"]["train"] == 2
        assert audit["counts"]["val"] == 1
        assert audit["counts"]["dev"] == 3
        assert audit["counts"]["test"] == 2
        assert audit["overlap"]["dev_test"] == 0
        assert audit["overlap"]["train_test"] == 0
        assert audit["direction_fit_ids_equal_dev_ids"] is True

    def test_leakage_detected_aborts(self):
        from utils import audit_split_leakage

        meta = {
            "question_ids_train": ["q1", "q2"],
            "question_ids_val": ["q3"],
            "question_ids_dev": ["q1", "q2", "q3", "q4"],  # q4 is in test!
            "question_ids_test": ["q4", "q5"],
            "direction_fit_source": "dev_only",
            "sigma_fit_source": "dev_only",
        }
        with pytest.raises(RuntimeError, match="FATAL.*test question"):
            audit_split_leakage(meta)

    def test_sample_ids_capped_at_five(self):
        from utils import audit_split_leakage

        train = [f"q{i}" for i in range(20)]
        meta = {
            "question_ids_train": train,
            "question_ids_val": [],
            "question_ids_dev": train,
            "question_ids_test": [f"t{i}" for i in range(10)],
            "direction_fit_source": "dev_only",
            "sigma_fit_source": "dev_only",
        }
        audit = audit_split_leakage(meta)
        assert len(audit["sample_ids"]["train"]) == 5
        assert len(audit["sample_ids"]["test"]) == 5

    def test_direction_fit_mismatch_flagged(self):
        from utils import audit_split_leakage

        meta = {
            "question_ids_train": ["q1", "q2"],
            "question_ids_val": ["q3"],
            "question_ids_dev": ["q1", "q2"],  # missing q3 — mismatch
            "question_ids_test": ["q4"],
            "direction_fit_source": "dev_only",
            "sigma_fit_source": "dev_only",
        }
        audit = audit_split_leakage(meta)
        assert audit["direction_fit_ids_equal_dev_ids"] is False
