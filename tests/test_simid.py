from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import run_simid as simid_runner
from analyze_simid import (
    build_alias_audit_queue,
    index_rows,
    load_run_rows,
    main as analyze_main,
    option_order_stability_gate,
    require_paired_panel,
    selected_minus_control_slope_summaries,
    summarize_condition,
    write_report,
)
from build_simid_manifest import (
    BridgeItem,
    OptionRecord,
    build_manifest_row,
    build_truthfulqa_rows,
    main as build_manifest_main,
    select_bridge_distractors,
    validate_manifest,
)
from build_truthfulqa_splits import stable_question_id
from run_simid import (
    assert_noop_equivalence,
    build_iti_config,
    build_run_config,
    compute_mc_margin,
    load_existing_sample_ids,
    load_or_create_locked_manifest,
    simid_output_id,
    write_or_validate_run_config,
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
    dataset: str = "fixture",
    mc_endpoint: str = "fixture_mc1",
    mc_letter_correct: bool = True,
    mc_option_text_correct: bool = False,
    open_correct: bool = True,
) -> dict:
    chosen_index = 0 if mc_letter_correct else 1
    return {
        "sample_id": sample_id,
        "base_sample_id": base_sample_id or sample_id,
        "condition": condition,
        "alpha": alpha,
        "option_order_replicate": option_order_replicate,
        "dataset": dataset,
        "mc_endpoint": mc_endpoint,
        "question": "Question?",
        "mc_options": ["Gold", "Wrong"],
        "gold_option_indices": [0],
        "mc_letter_likelihood": {
            "chosen_letter": chr(ord("A") + chosen_index),
            "chosen_index": chosen_index,
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


def _manifest_row(sample_id: str, question: str) -> dict:
    return build_manifest_row(
        sample_id=sample_id,
        dataset="fixture",
        source_id=sample_id,
        question=question,
        options=[
            OptionRecord("Gold", True, {"source": "gold"}),
            OptionRecord("Wrong", False, {"source": "distractor"}),
        ],
        gold_aliases=["Gold"],
        seed=42,
        replicate_idx=0,
        model_path="model",
        tokenizer_path="model",
        iti_artifact_path="iti.pt",
        iti_artifact_sha256="abc",
    )


def _run_args(**overrides: object) -> argparse.Namespace:
    defaults: dict[str, object] = {
        "manifest": "manifest.json",
        "model_path": "model",
        "device_map": "cuda:0",
        "iti_artifact_path": "iti.pt",
        "iti_family": "truthfulqa_paperfaithful",
        "iti_k": 12,
        "decode_scope": "first_3_tokens",
        "alphas": [0.0, 8.0],
        "conditions": ["selected"],
        "include_unhooked": False,
        "max_items": None,
        "top_k_first_token": 10,
        "mc_max_new_tokens": 4,
        "open_max_new_tokens": 64,
        "collect_debug_stats": False,
        "seed": 42,
        "noop_check": False,
        "noop_tolerance": 1e-5,
        "allow_analyzed_overwrite": False,
        "output_dir": None,
        "run_name": None,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


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


def test_manifest_validation_requires_option_order_replicate() -> None:
    row = _manifest_row("simid_test_1", "Question?")
    del row["option_order_replicate"]

    with pytest.raises(ValueError, match="option_order_replicate"):
        validate_manifest({"schema_version": "simid_manifest/v1", "rows": [row]})


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


def test_bridge_distractor_selection_rejects_options_containing_gold_alias() -> None:
    item = BridgeItem("q1", "Question?", ["OIL"])
    candidates = [
        ("q2", "Oil refining"),
        ("q3", "Natural gas"),
        ("q4", "Solar power"),
    ]

    distractors = select_bridge_distractors(
        item,
        all_candidates=candidates,
        baseline_wrong_by_qid={"q1": "Oil refining"},
        n_distractors=2,
        seed=7,
    )

    texts = {record.text for record in distractors}
    assert "Oil refining" not in texts
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


def test_resume_uses_existing_locked_manifest_when_source_changes(
    tmp_path: Path,
) -> None:
    source = tmp_path / "manifest.json"
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    row_a = _manifest_row("simid_a", "Original?")
    source.write_text(
        json.dumps({"schema_version": "simid_manifest/v1", "rows": [row_a]}) + "\n",
        encoding="utf-8",
    )

    initial_rows = load_or_create_locked_manifest(
        source_manifest=str(source),
        output_dir=output_dir,
        max_items=None,
    )

    row_b = _manifest_row("simid_b", "Changed?")
    source.write_text(
        json.dumps({"schema_version": "simid_manifest/v1", "rows": [row_b]}) + "\n",
        encoding="utf-8",
    )
    resumed_rows = load_or_create_locked_manifest(
        source_manifest=str(source),
        output_dir=output_dir,
        max_items=None,
    )

    assert [row["sample_id"] for row in initial_rows] == ["simid_a"]
    assert [row["sample_id"] for row in resumed_rows] == ["simid_a"]
    assert resumed_rows[0]["question"] == "Original?"


def test_resume_refuses_runtime_config_change_with_existing_rows(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    rows = [_manifest_row("simid_a", "Question?")]
    conditions = [simid_runner.CONDITION_SPECS["selected"]]
    iti_config = build_iti_config(
        model_path="model",
        tokenizer_path="model",
        iti_artifact_path="iti.pt",
        iti_artifact_sha256="abc",
        iti_family="truthfulqa_paperfaithful",
        iti_k=12,
        decode_scope="first_3_tokens",
    )
    run_config_path = output_dir / "run_config.json"
    locked_manifest_path = output_dir / "manifest.locked.json"
    run_config = build_run_config(
        args=_run_args(),
        output_dir=output_dir,
        rows=rows,
        conditions=conditions,
        iti_config=iti_config,
        locked_manifest_path=locked_manifest_path,
    )
    write_or_validate_run_config(
        path=run_config_path,
        run_config=run_config,
        output_dir=output_dir,
    )
    alpha_path = output_dir / "selected" / "alpha_0.0.jsonl"
    alpha_path.parent.mkdir()
    alpha_path.write_text(json.dumps({"sample_id": "simid_a"}) + "\n", encoding="utf-8")

    changed_config = dict(iti_config)
    changed_config["k"] = 7
    changed_run_config = build_run_config(
        args=_run_args(iti_k=7),
        output_dir=output_dir,
        rows=rows,
        conditions=conditions,
        iti_config=changed_config,
        locked_manifest_path=locked_manifest_path,
    )

    with pytest.raises(ValueError, match="Cannot resume"):
        write_or_validate_run_config(
            path=run_config_path,
            run_config=changed_run_config,
            output_dir=output_dir,
        )


def test_resume_refuses_configless_directory_with_existing_rows(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "run"
    alpha_path = output_dir / "selected" / "alpha_0.0.jsonl"
    alpha_path.parent.mkdir(parents=True)
    alpha_path.write_text(json.dumps({"sample_id": "simid_a"}) + "\n", encoding="utf-8")
    rows = [_manifest_row("simid_a", "Question?")]
    conditions = [simid_runner.CONDITION_SPECS["selected"]]
    iti_config = build_iti_config(
        model_path="model",
        tokenizer_path="model",
        iti_artifact_path="iti.pt",
        iti_artifact_sha256="abc",
        iti_family="truthfulqa_paperfaithful",
        iti_k=12,
        decode_scope="first_3_tokens",
    )
    run_config = build_run_config(
        args=_run_args(),
        output_dir=output_dir,
        rows=rows,
        conditions=conditions,
        iti_config=iti_config,
        locked_manifest_path=output_dir / "manifest.locked.json",
    )

    with pytest.raises(ValueError, match="run_config.json is missing"):
        write_or_validate_run_config(
            path=output_dir / "run_config.json",
            run_config=run_config,
            output_dir=output_dir,
        )


def test_run_main_refuses_configless_rows_before_writing_locked_manifest(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "run"
    alpha_path = output_dir / "selected" / "alpha_0.0.jsonl"
    alpha_path.parent.mkdir(parents=True)
    alpha_path.write_text(json.dumps({"sample_id": "simid_a"}) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="run_config.json is missing"):
        simid_runner.main(
            [
                "--output-dir",
                str(output_dir),
                "--manifest",
                str(tmp_path / "manifest.json"),
            ]
        )

    assert not (output_dir / "manifest.locked.json").exists()


def test_iti_config_records_runtime_overrides() -> None:
    config = build_iti_config(
        model_path="override-model",
        tokenizer_path="override-tokenizer",
        iti_artifact_path="override/iti_heads.pt",
        iti_artifact_sha256="abc123",
        iti_family="custom_family",
        iti_k=7,
        decode_scope="full_decode",
    )

    assert config == {
        "model_path": "override-model",
        "tokenizer_path": "override-tokenizer",
        "iti_artifact_path": "override/iti_heads.pt",
        "iti_artifact_sha256": "abc123",
        "family": "custom_family",
        "effective_family": "iti_custom_family",
        "k": 7,
        "decode_scope": "full_decode",
    }


def test_run_condition_alpha_passes_iti_config_to_scoring(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    iti_config = {"model_path": "runtime-model", "k": 3}
    captured_configs = []

    def fake_score_simid_item(
        model: object,
        tokenizer: object,
        row: dict,
        *,
        scaler: object,
        alpha: float,
        condition: object,
        iti_config: dict,
        top_k_first_token: int,
        mc_max_new_tokens: int,
        open_max_new_tokens: int,
    ) -> dict:
        captured_configs.append(dict(iti_config))
        return {"sample_id": row["sample_id"], "alpha": alpha}

    monkeypatch.setattr(simid_runner, "score_simid_item", fake_score_simid_item)

    simid_runner.run_condition_alpha(
        model=object(),
        tokenizer=object(),
        rows=[{"sample_id": "s1"}],
        scaler=None,
        alpha=0.0,
        condition=simid_runner.CONDITION_SPECS["selected"],
        iti_config=iti_config,
        output_dir=tmp_path,
        top_k_first_token=5,
        mc_max_new_tokens=4,
        open_max_new_tokens=64,
    )

    assert captured_configs == [iti_config]


def test_score_simid_item_includes_option_order_replicate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    row = _manifest_row("simid_a", "Question?")
    row["option_order_replicate"] = 1
    responses = iter(["A", "Gold"])

    monkeypatch.setattr(
        simid_runner,
        "score_mc_likelihood",
        lambda *args, **kwargs: ({"endpoint": "option_text_teacher_forced"}, 0.1),
    )
    monkeypatch.setattr(
        simid_runner,
        "score_mc_letters",
        lambda *args, **kwargs: (
            {
                "endpoint": "letter_teacher_forced",
                "valid_letters": ["A", "B"],
                "chosen_letter": "A",
                "chosen_index": 0,
                "chosen_is_gold": True,
                "full": {"margin": 1.0},
                "avg": {"margin": 1.0},
            },
            object(),
            0.1,
        ),
    )
    monkeypatch.setattr(
        simid_runner,
        "generate_response",
        lambda *args, **kwargs: (next(responses), {"total_s": 0.1}),
    )
    monkeypatch.setattr(simid_runner, "tokenize_chat", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        simid_runner,
        "score_open_margins",
        lambda *args, **kwargs: (
            {
                "endpoint": "open_prompt_gold_vs_distractor_teacher_forced",
                "full": {"margin": 1.0},
                "avg": {"margin": 1.0},
            },
            0.1,
        ),
    )
    monkeypatch.setattr(
        simid_runner,
        "topk_first_token_logprobs",
        lambda *args, **kwargs: [],
    )
    monkeypatch.setattr(simid_runner, "get_git_sha", lambda: "abc123")

    scored = simid_runner.score_simid_item(
        model=object(),
        tokenizer=object(),
        row=row,
        scaler=None,
        alpha=0.0,
        condition=simid_runner.CONDITION_SPECS["selected"],
        iti_config={"model_path": "model"},
        top_k_first_token=5,
        mc_max_new_tokens=4,
        open_max_new_tokens=64,
    )

    assert scored["option_order_seed"] == row["option_order_seed"]
    assert scored["option_order_replicate"] == 1


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

    assert summary["rates"]["0.0"]["mc_letter_likelihood_correct"][
        "estimate"
    ] == pytest.approx(1.0)
    assert summary["rates"]["0.0"]["mc_likelihood_full_correct"][
        "estimate"
    ] == pytest.approx(0.0)


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
    assert summary["rates"]["0.0"]["mc_letter_likelihood_correct"][
        "estimate"
    ] == pytest.approx(0.75)


def test_option_order_gate_computes_global_spread_by_replicate_index() -> None:
    groups = {
        "base1": [
            _analysis_row(
                sample_id="base1_ord0",
                base_sample_id="base1",
                option_order_replicate=0,
                mc_letter_correct=True,
            ),
            _analysis_row(
                sample_id="base1_ord1",
                base_sample_id="base1",
                option_order_replicate=1,
                mc_letter_correct=True,
            ),
        ],
        "base2": [
            _analysis_row(
                sample_id="base2_ord0",
                base_sample_id="base2",
                option_order_replicate=0,
                mc_letter_correct=True,
            ),
            _analysis_row(
                sample_id="base2_ord1",
                base_sample_id="base2",
                option_order_replicate=1,
                mc_letter_correct=False,
            ),
        ],
    }

    gate = option_order_stability_gate(groups)

    assert gate["replicate_rates"] == {"ord:0": 1.0, "ord:1": 0.5}
    assert gate["global_max_rate_spread"] == pytest.approx(0.5)
    assert gate["passed"] is False


def test_option_order_gate_reports_item_flips_separately_from_global_spread() -> None:
    groups = {
        "base1": [
            _analysis_row(
                sample_id="base1_ord0",
                base_sample_id="base1",
                option_order_replicate=0,
                mc_letter_correct=True,
            ),
            _analysis_row(
                sample_id="base1_ord1",
                base_sample_id="base1",
                option_order_replicate=1,
                mc_letter_correct=False,
            ),
        ],
        "base2": [
            _analysis_row(
                sample_id="base2_ord0",
                base_sample_id="base2",
                option_order_replicate=0,
                mc_letter_correct=False,
            ),
            _analysis_row(
                sample_id="base2_ord1",
                base_sample_id="base2",
                option_order_replicate=1,
                mc_letter_correct=True,
            ),
        ],
    }

    gate = option_order_stability_gate(groups)

    assert gate["replicate_rates"] == {"ord:0": 0.5, "ord:1": 0.5}
    assert gate["global_max_rate_spread"] == pytest.approx(0.0)
    assert gate["item_flip_count"] == 2
    assert gate["item_flip_rate"] == pytest.approx(1.0)
    assert gate["item_flip_base_sample_ids"] == ["base1", "base2"]
    assert gate["passed"] is True


def test_option_order_gate_fails_clearly_without_replicate_metadata() -> None:
    row = _analysis_row(sample_id="base1_ord0", base_sample_id="base1")
    del row["option_order_replicate"]

    gate = option_order_stability_gate({"base1": [row]})

    assert gate["passed"] is False
    assert "missing option_order_replicate metadata" in gate["reason"]


def test_analyzer_reports_dataset_endpoint_strata() -> None:
    rows = [
        _analysis_row(
            sample_id="truthful_1",
            alpha=0.0,
            dataset="truthfulqa",
            mc_endpoint="truthfulqa_mc1",
            mc_letter_correct=True,
        ),
        _analysis_row(
            sample_id="truthful_1",
            alpha=8.0,
            dataset="truthfulqa",
            mc_endpoint="truthfulqa_mc1",
            mc_letter_correct=True,
        ),
        _analysis_row(
            sample_id="bridge_1",
            alpha=0.0,
            dataset="triviaqa_bridge",
            mc_endpoint="synthetic_mc1",
            mc_letter_correct=False,
        ),
        _analysis_row(
            sample_id="bridge_1",
            alpha=8.0,
            dataset="triviaqa_bridge",
            mc_endpoint="synthetic_mc1",
            mc_letter_correct=False,
        ),
    ]

    summary = summarize_condition(
        index_rows(rows),
        condition="selected",
        alphas=[0.0, 8.0],
        baseline_alpha=0.0,
        n_resamples=100,
        seed=1,
    )

    strata = summary["dataset_endpoint_summaries"]
    truthfulqa = strata["truthfulqa::truthfulqa_mc1"]
    bridge = strata["triviaqa_bridge::synthetic_mc1"]
    assert summary["summary_scope"] == "pooled_across_dataset_and_mc_endpoint"
    assert truthfulqa["n_paired_items"] == 1
    assert bridge["n_paired_items"] == 1
    assert truthfulqa["rates"]["0.0"]["mc_letter_likelihood_correct"][
        "estimate"
    ] == pytest.approx(1.0)
    assert bridge["rates"]["0.0"]["mc_letter_likelihood_correct"][
        "estimate"
    ] == pytest.approx(0.0)


def test_analyzer_splits_truthfulqa_strata_by_leakage_metadata(
    tmp_path: Path,
) -> None:
    rows = []
    for sample_id, split, seen in [
        ("truthful_heldout", "test", False),
        ("truthful_fitted", "train", True),
    ]:
        for alpha in [0.0, 8.0]:
            row = _analysis_row(
                sample_id=sample_id,
                alpha=alpha,
                dataset="truthfulqa",
                mc_endpoint="truthfulqa_mc1",
            )
            row.update(
                {
                    "truthfulqa_artifact_split": split,
                    "truthfulqa_seen_in_iti_fit": seen,
                    "truthfulqa_leakage_policy": "allow_fitted",
                }
            )
            rows.append(row)

    summary = summarize_condition(
        index_rows(rows),
        condition="selected",
        alphas=[0.0, 8.0],
        baseline_alpha=0.0,
        n_resamples=100,
        seed=1,
    )
    report_path = tmp_path / "report.md"
    write_report({"conditions": {"selected": summary}}, report_path)

    strata = summary["dataset_endpoint_summaries"]
    heldout_key = (
        "truthfulqa::truthfulqa_mc1::truthfulqa_artifact_split=test::"
        "truthfulqa_seen_in_iti_fit=false::truthfulqa_leakage_policy=allow_fitted"
    )
    fitted_key = (
        "truthfulqa::truthfulqa_mc1::truthfulqa_artifact_split=train::"
        "truthfulqa_seen_in_iti_fit=true::truthfulqa_leakage_policy=allow_fitted"
    )
    assert sorted(strata) == [heldout_key, fitted_key]
    assert strata[heldout_key]["truthfulqa_leakage_metadata"] == {
        "truthfulqa_artifact_split": "test",
        "truthfulqa_seen_in_iti_fit": "false",
        "truthfulqa_leakage_policy": "allow_fitted",
    }
    report_text = report_path.read_text(encoding="utf-8")
    assert "artifact_split=test" in report_text
    assert "seen_in_iti_fit=false" in report_text
    assert "leakage_policy=allow_fitted" in report_text


def test_load_run_rows_recovers_mc_endpoint_from_locked_manifest(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    selected_dir = run_dir / "selected"
    selected_dir.mkdir(parents=True)
    manifest_row = _manifest_row("simid_a", "Question?")
    manifest_row["dataset"] = "truthfulqa"
    manifest_row["mc_endpoint"] = "truthfulqa_mc1"
    manifest_row["option_order_replicate"] = 1
    manifest_row["truthfulqa_artifact_split"] = "test"
    manifest_row["truthfulqa_seen_in_iti_fit"] = False
    manifest_row["truthfulqa_leakage_policy"] = "heldout_only"
    (run_dir / "manifest.locked.json").write_text(
        json.dumps(
            {"schema_version": "simid_locked_manifest/v1", "rows": [manifest_row]}
        )
        + "\n",
        encoding="utf-8",
    )
    (selected_dir / "alpha_0.0.jsonl").write_text(
        json.dumps({"sample_id": "simid_a"}) + "\n",
        encoding="utf-8",
    )

    rows = load_run_rows(run_dir)

    assert rows[0]["dataset"] == "truthfulqa"
    assert rows[0]["mc_endpoint"] == "truthfulqa_mc1"
    assert rows[0]["option_order_replicate"] == 1
    assert rows[0]["truthfulqa_artifact_split"] == "test"
    assert rows[0]["truthfulqa_seen_in_iti_fit"] is False
    assert rows[0]["truthfulqa_leakage_policy"] == "heldout_only"


def test_control_slope_requires_matching_replicates_across_controls() -> None:
    rows = [
        _analysis_row(
            sample_id="s1",
            base_sample_id="base1",
            alpha=0.0,
            option_order_replicate=0,
        ),
        _analysis_row(
            sample_id="s1__ord1",
            base_sample_id="base1",
            alpha=0.0,
            option_order_replicate=1,
        ),
        _analysis_row(
            sample_id="s1",
            base_sample_id="base1",
            alpha=8.0,
            option_order_replicate=0,
        ),
        _analysis_row(
            sample_id="s1__ord1",
            base_sample_id="base1",
            alpha=8.0,
            option_order_replicate=1,
        ),
        _analysis_row(
            sample_id="control_s1",
            base_sample_id="base1",
            condition="random_head_seed1",
            alpha=0.0,
            option_order_replicate=0,
        ),
        _analysis_row(
            sample_id="control_s1",
            base_sample_id="base1",
            condition="random_head_seed1",
            alpha=8.0,
            option_order_replicate=0,
        ),
    ]

    with pytest.raises(ValueError, match="Selected/control replicate sets differ"):
        selected_minus_control_slope_summaries(
            index_rows(rows),
            selected_condition="selected",
            control_conditions=["random_head_seed1"],
            alphas=[0.0, 8.0],
            n_resamples=100,
            seed=1,
        )


def test_report_baseline_rates_include_ci(tmp_path: Path) -> None:
    path = tmp_path / "report.md"
    write_report(
        {
            "conditions": {
                "selected": {
                    "n_paired_items": 2,
                    "baseline_alpha": 0.0,
                    "rates": {
                        "0.0": {
                            "mc_letter_likelihood_correct": {
                                "estimate": 0.75,
                                "ci": {"lower": 0.5, "upper": 1.0},
                            },
                            "open_correct": {
                                "estimate": 0.5,
                                "ci": {"lower": 0.25, "upper": 0.75},
                            },
                        }
                    },
                    "paired_deltas_vs_baseline": {},
                }
            }
        },
        path,
    )

    text = path.read_text(encoding="utf-8")
    assert "deterministic alias-grader correctness" in text
    assert "Pooled aggregate scope" in text
    assert "lettered MC=0.7500 [0.5000, 1.0000]" in text
    assert "open=0.5000 [0.2500, 0.7500]" in text


def test_report_phase0_option_order_details_are_human_readable(tmp_path: Path) -> None:
    path = tmp_path / "report.md"
    write_report(
        {
            "conditions": {},
            "phase0_gates": {
                "bridge_option_order_stability": {
                    "passed": True,
                    "global_max_rate_spread": 0.125,
                    "item_flip_count": 1,
                    "item_flip_rate": 0.125,
                    "item_flip_base_sample_ids": ["simid_bridge_1"],
                    "replicate_rates": {"ord:0": 0.875, "ord:1": 1.0},
                    "chosen_letter_distribution": {
                        "counts": {"A": 1, "B": 2},
                    },
                    "gold_position_correctness": {
                        "positions": {
                            "0": {"n": 2, "correct": 1, "rate": 0.5},
                            "1": {"n": 1, "correct": 1, "rate": 1.0},
                        }
                    },
                    "n_base_items": 8,
                }
            },
        },
        path,
    )

    text = path.read_text(encoding="utf-8")
    assert (
        "bridge_option_order_stability: PASS "
        "(global replicate spread=0.1250; item flips=1/8, rate=0.1250)"
    ) in text
    assert "replicate rates: ord:0=0.8750, ord:1=1.0000" in text
    assert "chosen letters: A=1, B=2" in text
    assert "gold-position correctness: pos 0: 1/2=0.5000" in text
    assert "flipped base items: simid_bridge_1" in text


def test_analysis_refuses_existing_outputs_by_default(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "results.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        analyze_main(["--run-dir", str(run_dir)])


def test_open_grading_is_labeled_and_audited_for_non_bridge_rows() -> None:
    grade = simid_runner.grade_open_response(
        {
            "dataset": "truthfulqa",
            "gold_aliases": ["Gold"],
            "mc_options": ["Gold", "Wrong"],
            "gold_option_indices": [0],
        },
        "Gold",
    )
    row = _analysis_row(
        sample_id="truthful_1",
        dataset="truthfulqa",
        mc_endpoint="truthfulqa_mc1",
        open_correct=True,
    )
    row["open_grade"] = grade

    queue = build_alias_audit_queue([row])

    assert grade["grader"]["name"] == "deterministic_alias_grader"
    assert queue[0]["reason"] == (
        "non_bridge_deterministic_alias_correct_requires_adjudication"
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


def test_manifest_builder_can_require_truthfulqa_rows(tmp_path: Path) -> None:
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

    with pytest.raises(ValueError, match="enough TruthfulQA rows"):
        build_manifest_main(
            [
                "--truthfulqa-csv",
                str(csv_path),
                "--truthfulqa-split-metadata",
                str(metadata_path),
                "--truthfulqa-leakage-policy",
                "drop_if_no_heldout",
                "--truthfulqa-n",
                "1",
                "--bridge-n",
                "0",
                "--min-truthfulqa-rows",
                "1",
                "--output",
                str(tmp_path / "simid.json"),
            ]
        )


def test_mvp_default_requires_truthfulqa_rows() -> None:
    script = Path(__file__).resolve().parent.parent / "scripts/infra/simid.sh"
    text = script.read_text(encoding="utf-8")

    assert "SIMID_TRUTHFULQA_LEAKAGE_POLICY:-heldout_only" in text
    assert "SIMID_MIN_TRUTHFULQA_ROWS:-1" in text


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
