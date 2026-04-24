from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from analyze_simid import index_rows, require_paired_panel, summarize_condition
from build_simid_manifest import (
    BridgeItem,
    OptionRecord,
    build_manifest_row,
    build_truthfulqa_rows,
    select_bridge_distractors,
    validate_manifest,
)
from build_truthfulqa_splits import stable_question_id
from run_simid import (
    assert_noop_equivalence,
    compute_mc_margin,
    load_existing_sample_ids,
    simid_output_id,
)


def _summary(
    *,
    first: float,
    first3: float,
    full: float,
    avg: float,
) -> dict:
    return {
        "first_token": {"sum_logprob": first, "n_tokens": 1},
        "first3": {"sum_logprob": first3, "n_tokens": 3},
        "full": {"sum_logprob": full, "avg_logprob": avg, "n_tokens": 3},
    }


def _analysis_row(
    *,
    sample_id: str,
    base_sample_id: str | None = None,
    condition: str = "selected",
    alpha: float = 0.0,
    option_order_replicate: int = 0,
    mc_letter_correct: bool = True,
    mc_option_text_correct: bool = False,
    open_correct: bool = True,
) -> dict:
    return {
        "sample_id": sample_id,
        "base_sample_id": base_sample_id or sample_id,
        "condition": condition,
        "alpha": alpha,
        "option_order_replicate": option_order_replicate,
        "dataset": "fixture",
        "question": "Question?",
        "mc_options": ["Gold", "Wrong"],
        "mc_letter_likelihood": {
            "chosen_is_gold": mc_letter_correct,
            "full": {"margin": 1.0 if mc_letter_correct else -1.0},
            "avg": {"margin": 1.0 if mc_letter_correct else -1.0},
        },
        "mc_generated_letter": {"chosen_is_gold": mc_letter_correct},
        "mc_likelihood": {
            "full": {
                "chosen_is_gold": mc_option_text_correct,
                "margin": 1.0 if mc_option_text_correct else -1.0,
            },
            "avg": {
                "chosen_is_gold": mc_option_text_correct,
                "margin": 1.0 if mc_option_text_correct else -1.0,
            },
        },
        "open_grade": {
            "correct": open_correct,
            "attempted": True,
            "failure_type": None if open_correct else "alias_or_other",
        },
        "open_generation": {"response": "Gold" if open_correct else "Wrong"},
        "open_margins": {
            "first_token": {"margin": 0.5},
            "first3": {"margin": 0.5},
            "full": {"margin": 0.5},
            "avg": {"margin": 0.5},
        },
        "gold_aliases": ["Gold"],
    }


def _write_truthfulqa_csv(path: Path, questions: list[str]) -> None:
    path.write_text(
        "Question,Best Answer,Correct Answers,Incorrect Answers,Category,Type,Source\n"
        + "\n".join(
            f"{question},Gold,Gold,Wrong,cat,type,src" for question in questions
        )
        + "\n",
        encoding="utf-8",
    )


def test_manifest_row_schema_and_option_order_are_deterministic() -> None:
    options = [
        OptionRecord("Paris", True, {"source": "gold"}),
        OptionRecord("London", False, {"source": "distractor"}),
        OptionRecord("Berlin", False, {"source": "distractor"}),
    ]

    row_a = build_manifest_row(
        sample_id="simid_test_1",
        dataset="fixture",
        source_id="q1",
        question="Capital of France?",
        options=options,
        gold_aliases=["Paris"],
        seed=42,
        replicate_idx=0,
        model_path="model",
        tokenizer_path="model",
        iti_artifact_path="iti.pt",
        iti_artifact_sha256="abc",
    )
    row_b = build_manifest_row(
        sample_id="simid_test_1",
        dataset="fixture",
        source_id="q1",
        question="Capital of France?",
        options=options,
        gold_aliases=["Paris"],
        seed=42,
        replicate_idx=0,
        model_path="model",
        tokenizer_path="model",
        iti_artifact_path="iti.pt",
        iti_artifact_sha256="abc",
    )

    assert row_a["mc_options"] == row_b["mc_options"]
    assert row_a["gold_option_indices"] == row_b["gold_option_indices"]
    validate_manifest({"schema_version": "simid_manifest/v1", "rows": [row_a]})


def test_bridge_distractor_selection_excludes_own_gold_aliases() -> None:
    item = BridgeItem("q1", "Question?", ["Paris", "City of Paris"])
    candidates = [
        ("q1", "Paris"),
        ("q1", "City of Paris"),
        ("q2", "London"),
        ("q3", "Berlin"),
        ("q4", "Rome"),
    ]

    distractors = select_bridge_distractors(
        item,
        all_candidates=candidates,
        baseline_wrong_by_qid={"q1": "Paris"},
        n_distractors=2,
        seed=7,
    )

    texts = {record.text for record in distractors}
    assert "Paris" not in texts
    assert "City of Paris" not in texts
    assert len(texts) == 2


def test_bridge_distractor_selection_preserves_seeded_randomness() -> None:
    item = BridgeItem("q1", "Question?", ["Paris"])
    candidates = [
        ("q2", "London"),
        ("q3", "Berlin"),
        ("q4", "Rome"),
        ("q5", "Madrid"),
        ("q6", "Vienna"),
        ("q7", "Prague"),
        ("q8", "Dublin"),
    ]

    seed_1 = select_bridge_distractors(
        item,
        all_candidates=candidates,
        baseline_wrong_by_qid={},
        n_distractors=3,
        seed=1,
    )
    seed_2 = select_bridge_distractors(
        item,
        all_candidates=candidates,
        baseline_wrong_by_qid={},
        n_distractors=3,
        seed=2,
    )

    assert [record.text for record in seed_1] != [record.text for record in seed_2]


def test_mc_margin_uses_correct_minus_best_incorrect_and_avg_sensitivity() -> None:
    option_summaries = [
        _summary(first=-5.0, first3=-9.0, full=-3.0, avg=-1.0),
        _summary(first=-1.0, first3=-3.0, full=-2.0, avg=-2.0),
        _summary(first=-4.0, first3=-8.0, full=-10.0, avg=-0.5),
    ]

    margins = compute_mc_margin(option_summaries, [1])

    assert margins["full"]["margin"] == pytest.approx(1.0)
    assert margins["full"]["best_incorrect_index"] == 0
    assert margins["avg"]["margin"] == pytest.approx(-1.5)
    assert margins["avg"]["best_incorrect_index"] == 2
    assert margins["full"]["chosen_is_gold"] is True
    assert margins["avg"]["chosen_is_gold"] is False


def test_output_ids_are_unique_and_resume_uses_sample_ids(tmp_path: Path) -> None:
    ids = {
        simid_output_id("s1", "selected", 0.0),
        simid_output_id("s1", "selected", 8.0),
        simid_output_id("s1", "random_head_seed1", 8.0),
    }
    assert len(ids) == 3

    out_path = tmp_path / "selected" / "alpha_0.0.jsonl"
    out_path.parent.mkdir(parents=True)
    out_path.write_text(
        json.dumps({"sample_id": "s1", "output_id": next(iter(ids))}) + "\n",
        encoding="utf-8",
    )

    assert load_existing_sample_ids(out_path) == {"s1"}


def test_analyzer_refuses_unpaired_condition_alpha_rows() -> None:
    rows = [
        {"condition": "selected", "alpha": 0.0, "sample_id": "s1"},
        {"condition": "selected", "alpha": 0.0, "sample_id": "s2"},
        {"condition": "selected", "alpha": 8.0, "sample_id": "s1"},
    ]
    indexed = index_rows(rows)

    with pytest.raises(ValueError, match="Unpaired SIMID rows"):
        require_paired_panel(indexed, condition="selected", alphas=[0.0, 8.0])


def test_analyzer_uses_lettered_mc_as_primary_mc_behavior() -> None:
    rows = [
        _analysis_row(sample_id="s1", alpha=0.0, mc_letter_correct=True),
        _analysis_row(sample_id="s1", alpha=8.0, mc_letter_correct=True),
    ]
    summary = summarize_condition(
        index_rows(rows),
        condition="selected",
        alphas=[0.0, 8.0],
        baseline_alpha=0.0,
        n_resamples=100,
        seed=1,
    )

    assert (
        summary["rates"]["0.0"]["mc_letter_likelihood_correct"]["estimate"]
        == pytest.approx(1.0)
    )
    assert (
        summary["rates"]["0.0"]["mc_likelihood_full_correct"]["estimate"]
        == pytest.approx(0.0)
    )


def test_analyzer_groups_option_order_replicates_by_base_item() -> None:
    rows = [
        _analysis_row(
            sample_id="s1",
            base_sample_id="base1",
            alpha=0.0,
            option_order_replicate=0,
            mc_letter_correct=True,
        ),
        _analysis_row(
            sample_id="s1__ord1",
            base_sample_id="base1",
            alpha=0.0,
            option_order_replicate=1,
            mc_letter_correct=False,
        ),
        _analysis_row(
            sample_id="s1",
            base_sample_id="base1",
            alpha=8.0,
            option_order_replicate=0,
            mc_letter_correct=True,
        ),
        _analysis_row(
            sample_id="s1__ord1",
            base_sample_id="base1",
            alpha=8.0,
            option_order_replicate=1,
            mc_letter_correct=True,
        ),
        _analysis_row(sample_id="s2", base_sample_id="base2", alpha=0.0),
        _analysis_row(sample_id="s2", base_sample_id="base2", alpha=8.0),
    ]
    summary = summarize_condition(
        index_rows(rows),
        condition="selected",
        alphas=[0.0, 8.0],
        baseline_alpha=0.0,
        n_resamples=100,
        seed=1,
    )

    assert summary["n_paired_items"] == 2
    assert summary["n_rows_at_baseline"] == 3
    assert (
        summary["rates"]["0.0"]["mc_letter_likelihood_correct"]["estimate"]
        == pytest.approx(0.75)
    )


def test_truthfulqa_heldout_policy_excludes_fitted_items(tmp_path: Path) -> None:
    csv_path = tmp_path / "truthfulqa.csv"
    questions = ["Held out?", "Fitted?"]
    _write_truthfulqa_csv(csv_path, questions)
    metadata_path = tmp_path / "extraction_metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "question_ids_train": [stable_question_id("Fitted?")],
                "question_ids_val": [],
                "question_ids_dev": [stable_question_id("Fitted?")],
                "question_ids_test": [stable_question_id("Held out?")],
            }
        ),
        encoding="utf-8",
    )

    rows = build_truthfulqa_rows(
        csv_path=csv_path,
        seed=42,
        n_rows=None,
        option_order_replicates=1,
        model_path="model",
        tokenizer_path="model",
        iti_artifact_path=str(tmp_path / "iti_heads.pt"),
        iti_artifact_sha256="abc",
        leakage_policy="heldout_only",
        split_metadata_path=metadata_path,
    )

    assert [row["truthfulqa_stable_id"] for row in rows] == [
        stable_question_id("Held out?")
    ]
    assert rows[0]["truthfulqa_seen_in_iti_fit"] is False


def test_truthfulqa_heldout_policy_refuses_artifact_without_test_ids(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "truthfulqa.csv"
    _write_truthfulqa_csv(csv_path, ["Fitted?"])
    metadata_path = tmp_path / "extraction_metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "question_ids_train": [stable_question_id("Fitted?")],
                "question_ids_val": [],
                "question_ids_dev": [stable_question_id("Fitted?")],
                "question_ids_test": [],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="no held-out TruthfulQA test IDs"):
        build_truthfulqa_rows(
            csv_path=csv_path,
            seed=42,
            n_rows=None,
            option_order_replicates=1,
            model_path="model",
            tokenizer_path="model",
            iti_artifact_path=str(tmp_path / "iti_heads.pt"),
            iti_artifact_sha256="abc",
            leakage_policy="heldout_only",
            split_metadata_path=metadata_path,
        )


def test_noop_equivalence_check_fails_on_hooked_alpha0_divergence() -> None:
    unhooked = [{"sample_id": "s1", "mc_likelihood": {"full": {"margin": 0.0}}}]
    hooked = [{"sample_id": "s1", "mc_likelihood": {"full": {"margin": 0.01}}}]

    with pytest.raises(AssertionError, match="alpha-0 no-op check failed"):
        assert_noop_equivalence(
            unhooked,
            hooked,
            tolerance=1e-5,
            compare_paths=(("mc_likelihood", "full", "margin"),),
        )
