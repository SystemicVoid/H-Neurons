from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import run_simid as simid_runner
from analyze_simid import (
    attach_open_adjudications,
    attach_prospective_open_labels,
    build_alias_audit_queue,
    build_open_calibration_queue,
    build_simid_open_adjudication_request,
    build_simid_open_adjudication_requests,
    effective_open_grade,
    fanout_existing_open_adjudications,
    index_rows,
    load_open_adjudications,
    load_open_calibration_summary,
    load_prospective_open_authority_manifest,
    load_run_rows,
    make_open_adjudication_row,
    main as analyze_main,
    normalize_open_calibration_summary,
    option_order_stability_gate,
    parse_simpleqa_verdict,
    prospective_effect_run_row_sha256,
    require_paired_panel,
    selected_minus_control_slope_summaries,
    simid_open_adjudication_custom_id,
    summarize_open_grading,
    summarize_condition,
    validate_complete_run_outputs,
    validate_open_adjudication_matches_row,
    validate_primary_open_judge_model,
    write_report,
)
from build_simid_manifest import (
    BridgeItem,
    OptionRecord,
    build_manifest_row,
    build_truthfulqa_rows,
    load_excluded_sample_ids,
    main as build_manifest_main,
    select_bridge_distractors,
    validate_manifest,
)
from build_truthfulqa_splits import stable_question_id
from finalize_simid_open_calibration import finalize_open_calibration
from label_simid_open_calibration import (
    CalibrationQueueLaunchError,
    build_adjudication_requests,
    build_secondary_requests,
    case_custom_id,
    chat_completion_kwargs_for_model,
    disagreement_rows,
    make_adjudication_row,
    make_secondary_row,
    parse_adjudication_payload,
    parse_args as label_parse_args,
    queue_row_sha256,
    raise_after_partial_write,
    validate_calibration_queue_for_launch,
    validate_existing_adjudication_context,
    validate_existing_output_rows,
    validate_secondary_queue_context,
)
from run_simid import (
    assert_noop_equivalence,
    build_iti_config,
    build_run_config,
    compute_mc_margin,
    load_existing_sample_ids,
    resolve_conditions,
    load_or_create_locked_manifest,
    simid_output_id,
    write_or_validate_run_config,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def _open_adjudication_row_for(
    row: dict,
    *,
    judge_grade: str = "CORRECT",
    valid: bool = True,
) -> dict:
    return {
        "schema_version": "simid_open_adjudication/v1",
        "sample_id": row["sample_id"],
        "condition": row["condition"],
        "alpha": row["alpha"],
        "base_sample_id": row["base_sample_id"],
        "dataset": row["dataset"],
        "mc_endpoint": row["mc_endpoint"],
        "question": row["question"],
        "gold_aliases": row["gold_aliases"],
        "response": row["open_generation"]["response"],
        "deterministic_open_grade": row["open_grade"],
        "judge_grade": judge_grade,
        "parse_result": {"verdict": judge_grade, "valid": valid},
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


def test_manifest_exclusion_loader_collects_sample_and_base_ids(tmp_path: Path) -> None:
    exclusions = tmp_path / "exclude.jsonl"
    exclusions.write_text(
        "\n".join(
            [
                json.dumps({"sample_id": "simid_a", "base_sample_id": "base_a"}),
                json.dumps({"sample_id": "simid_b"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    assert load_excluded_sample_ids([exclusions]) == {
        "simid_a",
        "base_a",
        "simid_b",
    }


def test_run_simid_supports_multi_seed_random_controls() -> None:
    conditions = resolve_conditions(
        ["selected", "random_head_seed5", "random_direction_seed5"],
        include_unhooked=True,
    )

    assert [condition.name for condition in conditions] == [
        "unhooked",
        "selected",
        "random_head_seed5",
        "random_direction_seed5",
    ]
    assert conditions[2].random_seed == 5
    assert conditions[3].direction_random_seed == 5


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _prospective_authority_fixture(
    tmp_path: Path,
    *,
    low_metrics: bool = False,
    include_source_manifest_sha256: bool = True,
    include_sample_design: bool = True,
    include_planned_manifest_sha256: bool = True,
    include_full_exclusion_provenance: bool = True,
) -> tuple[Path, Path, dict]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    run_dir = tmp_path / "future_run"
    run_dir.mkdir()
    manifest_row = {
        **_manifest_row("simid_new", "Fresh?"),
        "dataset": "truthfulqa",
        "truthfulqa_leakage_policy": "heldout_only",
        "truthfulqa_seen_in_iti_fit": False,
    }
    planned_manifest = tmp_path / "planned_manifest.json"
    _write_json(
        planned_manifest,
        {"schema_version": "simid_manifest/v1", "rows": [manifest_row]},
    )
    locked_payload = {
        "schema_version": "simid_locked_manifest/v1",
        "source_manifest": str(planned_manifest),
        "max_items": None,
        "rows": [manifest_row],
    }
    if include_source_manifest_sha256:
        locked_payload["source_manifest_sha256"] = _sha256(planned_manifest)
    _write_json(run_dir / "manifest.locked.json", locked_payload)

    open_gate = tmp_path / "open_gate"
    open_gate.mkdir()
    labels = tmp_path / "labels.jsonl"
    _write_jsonl(labels, [{"blind_case_id": "b1"}])
    raw_agreement = 0.5 if low_metrics else 0.92
    kappa = 0.1 if low_metrics else 0.87
    ac1 = 0.1 if low_metrics else 0.88
    policy = {
        "target": "simid_open_correctness_future_grading",
        "min_cases": 150,
        "min_raw_agreement": 0.9,
        "min_cohen_kappa": 0.8,
        "min_gwet_ac1": 0.8,
        "max_rule_gap_cases": 0,
    }
    metrics = {
        "raw_agreement": {
            "estimate": raw_agreement,
            "count": int(raw_agreement * 150),
            "n": 150,
        },
        "cohen_kappa": kappa,
        "gwet_ac1": ac1,
        "rule_gap_count": 0,
    }
    analysis = tmp_path / "analysis.json"
    _write_json(
        analysis,
        {
            "schema_version": "simid_prospective_open_calibration_analysis/v1",
            "package_dir": str(open_gate),
            "label_file": str(labels),
            "label_file_sha256": _sha256(labels),
            "n_cases": 150,
            "outcome": {"passed": True, "blockers": []},
            "policy": policy,
            "metrics": metrics,
        },
    )
    rubric = open_gate / "rubric.md"
    rubric.write_text("rubric\n", encoding="utf-8")
    review_manifest = open_gate / "review_manifest.json"
    _write_json(review_manifest, {})
    exclusions = tmp_path / "exclusions.jsonl"
    exclusion_rows = [
        {
            "source": "historical_mvp_locked_manifest",
            "sample_id": "simid_old",
            "base_sample_id": "simid_old",
        }
    ]
    if include_full_exclusion_provenance:
        exclusion_rows.append(
            {
                "source": "prospective_open_calibration_private_case_map",
                "sample_id": "simid_cal",
                "base_sample_id": "simid_cal",
            }
        )
    _write_jsonl(exclusions, exclusion_rows)
    authority = tmp_path / "authority.json"
    sample_design = {
        "manifest_path": str(planned_manifest),
        "exclusion_file": str(exclusions),
    }
    if include_planned_manifest_sha256:
        sample_design["manifest_sha256"] = _sha256(planned_manifest)
    authority_payload = {
        "schema_version": "simid_prospective_effect_run_gate/v2",
        "protocol_version": "fixture",
        "output_paths": {
            "effect_run_dir": str(run_dir),
            "package_dir": str(authority.parent),
        },
        "planned_run": {
            "sample_design": sample_design if include_sample_design else {}
        },
        "exclusions": {
            "path": str(exclusions),
            "content_sha256": _sha256(exclusions),
        },
        "frozen_open_grading_authority": {
            "authority_id": "fixture_authority",
            "authority_scope": "future_only",
            "requires_external_returned_labels": True,
            "gate_dir": {"path": str(open_gate)},
            "passing_analysis": {
                "path": str(analysis),
                "content_sha256": _sha256(analysis),
            },
            "rubric": {
                "path": str(rubric),
                "content_sha256": _sha256(rubric),
            },
            "passing_label_file": {
                "path": str(labels),
                "content_sha256": _sha256(labels),
            },
            "review_manifest": {
                "path": str(review_manifest),
                "content_sha256": _sha256(review_manifest),
            },
            "metrics": metrics,
            "policy": policy,
        },
    }
    _write_json(authority, authority_payload)
    return run_dir, authority, manifest_row


def test_prospective_open_authority_binds_run_dir_and_exclusions(
    tmp_path: Path,
) -> None:
    run_dir, authority, _manifest_row_payload = _prospective_authority_fixture(tmp_path)

    loaded = load_prospective_open_authority_manifest(authority, run_dir=run_dir)

    assert loaded is not None
    assert loaded["authority_id"] == "fixture_authority"
    assert loaded["content_sha256"] == _sha256(authority)
    assert loaded["requires_external_returned_labels"] is True

    exclusions = tmp_path / "exclusions.jsonl"
    _write_jsonl(
        exclusions,
        [
            {
                "source": "historical_mvp_locked_manifest",
                "sample_id": "simid_new",
                "base_sample_id": "simid_new",
            },
            {
                "source": "prospective_open_calibration_private_case_map",
                "sample_id": "simid_cal",
                "base_sample_id": "simid_cal",
            },
        ],
    )
    authority_payload = json.loads(authority.read_text(encoding="utf-8"))
    authority_payload["exclusions"]["content_sha256"] = _sha256(exclusions)
    authority.write_text(json.dumps(authority_payload) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="reuses excluded"):
        load_prospective_open_authority_manifest(authority, run_dir=run_dir)


def test_prospective_open_authority_rejects_low_metrics_even_if_passed(
    tmp_path: Path,
) -> None:
    run_dir, authority, _manifest_row_payload = _prospective_authority_fixture(
        tmp_path,
        low_metrics=True,
    )

    with pytest.raises(ValueError, match="raw_agreement below frozen threshold"):
        load_prospective_open_authority_manifest(authority, run_dir=run_dir)


def test_prospective_open_authority_requires_manifest_hash_and_exclusions(
    tmp_path: Path,
) -> None:
    run_dir, authority, _manifest_row_payload = _prospective_authority_fixture(
        tmp_path,
        include_source_manifest_sha256=False,
    )
    with pytest.raises(ValueError, match="source_manifest_sha256"):
        load_prospective_open_authority_manifest(authority, run_dir=run_dir)

    run_dir, authority, _manifest_row_payload = _prospective_authority_fixture(
        tmp_path / "missing_planned_hash",
        include_planned_manifest_sha256=False,
    )
    with pytest.raises(ValueError, match="missing planned manifest hash"):
        load_prospective_open_authority_manifest(authority, run_dir=run_dir)

    run_dir, authority, _manifest_row_payload = _prospective_authority_fixture(
        tmp_path / "missing_exclusion",
        include_full_exclusion_provenance=False,
    )
    with pytest.raises(ValueError, match="exclusions missing expected provenance"):
        load_prospective_open_authority_manifest(authority, run_dir=run_dir)


def test_prospective_authority_does_not_make_judge_adjudication_claimable(
    tmp_path: Path,
) -> None:
    run_dir, authority, _manifest_row_payload = _prospective_authority_fixture(tmp_path)
    loaded = load_prospective_open_authority_manifest(authority, run_dir=run_dir)
    rows = [
        _analysis_row(sample_id="simid_new", dataset="truthfulqa", open_correct=True)
    ]

    summary = summarize_open_grading(
        rows,
        adjudication_path=tmp_path / "open_adjudication.jsonl",
        adjudication_run=None,
        prospective_open_authority=loaded,
    )

    assert summary["claimable_open_correctness"] is False
    assert (
        summary["claimability_blocker"] == "prospective_open_external_labels_not_loaded"
    )


def test_prospective_external_labels_validate_and_make_claimable(
    tmp_path: Path,
) -> None:
    run_dir, authority, _manifest_row_payload = _prospective_authority_fixture(tmp_path)
    loaded = load_prospective_open_authority_manifest(authority, run_dir=run_dir)
    assert loaded is not None
    rows = [
        _analysis_row(sample_id="simid_new", dataset="truthfulqa", open_correct=False)
    ]
    label_package = tmp_path / "external_label_package"
    blind_rows = [
        {
            "blind_case_id": "blind_001",
            "review_order": 1,
            "question": rows[0]["question"],
            "gold_aliases": rows[0]["gold_aliases"],
            "predicted_answer": rows[0]["open_generation"]["response"],
        }
    ]
    private_rows = [
        {
            "schema_version": "simid_prospective_effect_open_private_case/v1",
            "blind_case_id": "blind_001",
            "review_order": 1,
            "sample_id": rows[0]["sample_id"],
            "base_sample_id": rows[0]["base_sample_id"],
            "dataset": rows[0]["dataset"],
            "condition": rows[0]["condition"],
            "alpha": rows[0]["alpha"],
            "run_row_sha256": prospective_effect_run_row_sha256(rows[0]),
            "authority_manifest_sha256": loaded["content_sha256"],
        }
    ]
    blind_path = label_package / "review_cases_blind.jsonl"
    private_path = label_package / "private_case_map.jsonl"
    _write_jsonl(blind_path, blind_rows)
    _write_jsonl(private_path, private_rows)
    rubric_hash = loaded["rubric"]["content_sha256"]
    review_manifest = {
        "schema_version": "simid_prospective_effect_open_label_package/v1",
        "prospective_open_authority": {
            "path": str(authority),
            "content_sha256": loaded["content_sha256"],
        },
        "rubric": {
            "path": loaded["rubric"]["path"],
            "content_sha256": rubric_hash,
        },
        "files": {
            "review_cases_blind": {
                "path": str(blind_path),
                "content_sha256": _sha256(blind_path),
            },
            "private_case_map": {
                "path": str(private_path),
                "content_sha256": _sha256(private_path),
            },
        },
    }
    _write_json(label_package / "review_manifest.json", review_manifest)
    labels_path = label_package / "prospective_open_labels_external.jsonl"
    _write_jsonl(
        labels_path,
        [
            {
                "schema_version": "simid_prospective_effect_open_label/v1",
                "blind_case_id": "blind_001",
                "review_order": 1,
                "label": "CORRECT",
                "confidence": 5,
                "rule_gap": False,
                "flags": [],
                "notes": "fixture external label",
                "rater": {"type": "external_human", "id": "fixture_reviewer"},
                "blind_cases_file_sha256": _sha256(blind_path),
                "rubric_sha256": rubric_hash,
            }
        ],
    )

    label_summary = attach_prospective_open_labels(
        rows,
        package_dir=label_package,
        label_file=labels_path,
        prospective_open_authority=loaded,
    )
    summary = summarize_open_grading(
        rows,
        adjudication_path=tmp_path / "open_adjudication.jsonl",
        adjudication_run=None,
        prospective_open_authority=loaded,
        prospective_open_labels=label_summary,
    )

    assert label_summary is not None
    assert summary["claimable_open_correctness"] is True
    assert summary["effective_grade_source_counts"] == {"external_blind_label": 1}
    assert summary["claim_bearing_metric_correct"] == "external_blind_open_correct"
    assert summary["prospective_open_labels"]["n"] == 1


def test_prospective_external_labels_reject_private_key_leak(
    tmp_path: Path,
) -> None:
    run_dir, authority, _manifest_row_payload = _prospective_authority_fixture(tmp_path)
    loaded = load_prospective_open_authority_manifest(authority, run_dir=run_dir)
    assert loaded is not None
    rows = [_analysis_row(sample_id="simid_new", dataset="truthfulqa")]
    label_package = tmp_path / "leaky_label_package"
    blind_path = label_package / "review_cases_blind.jsonl"
    private_path = label_package / "private_case_map.jsonl"
    _write_jsonl(
        blind_path,
        [
            {
                "blind_case_id": "blind_001",
                "review_order": 1,
                "question": rows[0]["question"],
                "gold_aliases": rows[0]["gold_aliases"],
                "predicted_answer": rows[0]["open_generation"]["response"],
                "sample_id": rows[0]["sample_id"],
            }
        ],
    )
    _write_jsonl(private_path, [])
    _write_json(
        label_package / "review_manifest.json",
        {
            "schema_version": "simid_prospective_effect_open_label_package/v1",
            "prospective_open_authority": {
                "path": str(authority),
                "content_sha256": loaded["content_sha256"],
            },
            "rubric": loaded["rubric"],
            "files": {
                "review_cases_blind": {
                    "path": str(blind_path),
                    "content_sha256": _sha256(blind_path),
                },
                "private_case_map": {
                    "path": str(private_path),
                    "content_sha256": _sha256(private_path),
                },
            },
        },
    )

    with pytest.raises(ValueError, match="blind row exposes unexpected keys"):
        attach_prospective_open_labels(
            rows,
            package_dir=label_package,
            label_file=label_package / "labels.jsonl",
            prospective_open_authority=loaded,
        )


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
    locked_payload = json.loads(
        (output_dir / "manifest.locked.json").read_text(encoding="utf-8")
    )
    assert locked_payload["source_manifest_sha256"] == _sha256(source)

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


def test_validate_complete_run_outputs_rejects_partial_alpha_file(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    (run_dir / "selected").mkdir(parents=True)
    manifest_rows = [_manifest_row("simid_a", "Question A?")]
    (run_dir / "manifest.locked.json").write_text(
        json.dumps(
            {"schema_version": "simid_locked_manifest/v1", "rows": manifest_rows}
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "run_config.json").write_text(
        json.dumps(
            {
                "schema_version": "simid_run_config/v1",
                "n_rows": 1,
                "conditions": [{"name": "selected"}],
                "alphas": [0.0, 8.0],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "selected" / "alpha_0.0.jsonl").write_text(
        json.dumps({"sample_id": "simid_a"}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="incomplete or inconsistent"):
        validate_complete_run_outputs(run_dir)


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
    assert "open_correct=0.5000 [0.2500, 0.7500]" in text


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


def test_open_adjudication_rows_are_keyed_and_loaded(tmp_path: Path) -> None:
    path = tmp_path / "open_adjudication.jsonl"
    analysis_rows = [
        _analysis_row(sample_id="s1", condition="selected", alpha=0.0),
        _analysis_row(sample_id="s2", condition="unhooked", alpha=0.0),
    ]
    rows = [
        _open_adjudication_row_for(analysis_rows[0], judge_grade="CORRECT"),
        _open_adjudication_row_for(analysis_rows[1], judge_grade="INCORRECT"),
    ]
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    adjudications = load_open_adjudications(path)
    attached = attach_open_adjudications(analysis_rows, adjudications)

    assert attached == 2
    assert adjudications[("s1", "selected", 0.0)]["judge_grade"] == "CORRECT"
    assert analysis_rows[1]["open_adjudication"]["judge_grade"] == "INCORRECT"


def test_open_adjudication_content_mismatch_is_rejected() -> None:
    row = _analysis_row(sample_id="s1", condition="selected", alpha=0.0)
    stale_adjudication = _open_adjudication_row_for(row)
    stale_adjudication["response"] = "Different answer"

    with pytest.raises(ValueError, match="does not match the current run row"):
        validate_open_adjudication_matches_row(row, stale_adjudication)


def test_primary_open_judge_rejects_gpt5_until_kwargs_are_model_aware() -> None:
    with pytest.raises(ValueError, match="legacy SimpleQA"):
        validate_primary_open_judge_model("gpt-5.5")


def test_effective_open_grade_prefers_adjudication() -> None:
    row = _analysis_row(sample_id="s1", open_correct=False)
    row["open_adjudication"] = {"judge_grade": "CORRECT"}

    grade = effective_open_grade(row)

    assert grade["source"] == "adjudication"
    assert grade["correct"] is True
    assert grade["attempted"] is True
    assert grade["claimable"] is False
    assert grade["claimability_blocker"] == "calibration_evidence_not_recorded"


def test_effective_open_grade_falls_back_to_deterministic_without_adjudication() -> (
    None
):
    row = _analysis_row(sample_id="s1", open_correct=True)

    grade = effective_open_grade(row)

    assert grade["source"] == "deterministic_alias"
    assert grade["correct"] is True
    assert grade["claimable"] is False


def test_effective_open_grade_marks_malformed_adjudication_unknown() -> None:
    row = _analysis_row(sample_id="s1", open_correct=True)
    row["open_adjudication"] = {"parse_result": {"valid": False}}

    grade = effective_open_grade(row)

    assert grade["source"] == "adjudication_unknown"
    assert grade["correct"] is False
    assert grade["claimable"] is False
    assert grade["claimability_blocker"] == "judge_verdict_unknown_or_error"


def test_mixed_open_grade_sources_block_claimability(tmp_path: Path) -> None:
    adjudicated = _analysis_row(sample_id="s1", open_correct=False)
    adjudicated["open_adjudication"] = {"judge_grade": "CORRECT"}
    deterministic = _analysis_row(sample_id="s2", open_correct=True)

    summary = summarize_open_grading(
        [adjudicated, deterministic],
        adjudication_path=tmp_path / "open_adjudication.jsonl",
        adjudication_run=None,
    )

    assert summary["metric_correct"] == "adjudicated_open_correct"
    assert summary["effective_grade_source_counts"] == {
        "adjudication": 1,
        "deterministic_alias": 1,
    }
    assert summary["claimable_open_correctness"] is False
    assert (
        summary["claimability_blocker"]
        == "mixed_adjudication_and_deterministic_sources"
    )


def _valid_open_calibration_summary() -> dict:
    return {
        "schema_version": "simid_open_calibration/v1",
        "status": "adjudicated",
        "target": "simid_open_correctness",
        "n_cases": 100,
        "sampling": {
            "audit_queue_disagreement_n": 40,
            "stratified_panel_n": 60,
        },
        "rule": {
            "path": "data/judge_validation/simid_open/adjudication_rule.md",
            "git_commit": "abc123",
            "content_sha256": "deadbeef",
        },
        "irr": {
            "n_cases": 100,
            "raw_agreement": {"count": 99, "n": 100},
            "cohen_kappa": 0.91,
            "gwet_ac1": 0.94,
        },
        "adjudication": {
            "rule_gap_cases": {"count": 0, "n": 1},
        },
    }


def test_open_grading_becomes_claimable_with_valid_calibration(
    tmp_path: Path,
) -> None:
    rows = [
        _analysis_row(sample_id="s1", open_correct=False),
        _analysis_row(sample_id="s2", open_correct=True),
    ]
    rows[0]["open_adjudication"] = {"judge_grade": "CORRECT"}
    rows[1]["open_adjudication"] = {"judge_grade": "INCORRECT"}
    calibration = normalize_open_calibration_summary(_valid_open_calibration_summary())

    summary = summarize_open_grading(
        rows,
        adjudication_path=tmp_path / "open_adjudication.jsonl",
        adjudication_run=None,
        calibration_summary=calibration,
    )

    assert summary["claimable_open_correctness"] is True
    assert summary["claimability_blocker"] is None
    assert summary["open_metrics_scope"] == "claimable"
    assert summary["open_correctness_claim_contract"] == (
        "claimable_with_recorded_judge_calibration_evidence"
    )
    assert summary["calibration"]["claimability"]["passed"] is True


def test_open_calibration_blocks_below_minimum_case_count() -> None:
    calibration_payload = _valid_open_calibration_summary()
    calibration_payload["n_cases"] = 99
    calibration_payload["sampling"]["audit_queue_disagreement_n"] = 39
    calibration_payload["sampling"]["stratified_panel_n"] = 60
    calibration_payload["irr"]["n_cases"] = 99
    calibration_payload["irr"]["raw_agreement"] = {"count": 98, "n": 99}

    calibration = normalize_open_calibration_summary(calibration_payload)

    assert calibration["claimability"]["passed"] is False
    assert calibration["claimability"]["blocker"] == (
        "calibration_evidence_failed_thresholds"
    )
    assert "n_cases_below_minimum" in calibration["claimability"]["blockers"]
    assert calibration["claimability"]["policy"]["min_cases"] == 100


@pytest.mark.parametrize(
    "n_cases,kappa,ac1,expect_blockers,forbid_blockers,expect_passed",
    [
        # Right at the boundary on every threshold simultaneously: claimable.
        (
            100,
            0.80,
            0.80,
            set(),
            {
                "n_cases_below_minimum",
                "cohen_kappa_below_threshold",
                "gwet_ac1_below_threshold",
            },
            True,
        ),
        # n one below the boundary, IRR at threshold: only n blocks.
        (
            99,
            0.80,
            0.80,
            {"n_cases_below_minimum"},
            {"cohen_kappa_below_threshold", "gwet_ac1_below_threshold"},
            False,
        ),
        # n at boundary, kappa just below threshold: kappa blocks, n does not.
        (
            100,
            0.7999,
            0.80,
            {"cohen_kappa_below_threshold"},
            {"n_cases_below_minimum", "gwet_ac1_below_threshold"},
            False,
        ),
        # n at boundary, AC1 just below threshold: AC1 blocks, n does not.
        (
            100,
            0.80,
            0.7999,
            {"gwet_ac1_below_threshold"},
            {"n_cases_below_minimum", "cohen_kappa_below_threshold"},
            False,
        ),
    ],
)
def test_open_calibration_threshold_boundaries_pin_specific_blockers(
    n_cases: int,
    kappa: float,
    ac1: float,
    expect_blockers: set[str],
    forbid_blockers: set[str],
    expect_passed: bool,
) -> None:
    """Each threshold must surface its own blocker identity at the boundary."""
    payload = _valid_open_calibration_summary()
    payload["n_cases"] = n_cases
    payload["irr"]["n_cases"] = n_cases
    payload["irr"]["raw_agreement"] = {"count": n_cases, "n": n_cases}
    payload["irr"]["cohen_kappa"] = kappa
    payload["irr"]["gwet_ac1"] = ac1

    calibration = normalize_open_calibration_summary(payload)
    blockers = set(calibration["claimability"]["blockers"])

    assert expect_blockers <= blockers, (
        f"missing required blockers {expect_blockers - blockers} in {blockers}"
    )
    assert blockers.isdisjoint(forbid_blockers), (
        f"unexpected blockers {blockers & forbid_blockers} present"
    )
    assert calibration["claimability"]["passed"] is expect_passed


def test_open_calibration_summary_must_match_run_queue_case_ids(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "open_calibration_queue.jsonl").write_text(
        "\n".join(
            json.dumps({"calibration_case_id": case_id})
            for case_id in ["case_a", "case_b"]
        )
        + "\n",
        encoding="utf-8",
    )
    summary_path = run_dir / "open_calibration_summary.json"
    payload = _valid_open_calibration_summary()
    payload["n_cases"] = 2
    payload["irr"]["n_cases"] = 2
    payload["adjudicated_rows"] = [{"calibration_case_id": "case_a"}]
    summary_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="case IDs do not exactly match"):
        load_open_calibration_summary(summary_path, run_dir=run_dir)


def test_open_calibration_summary_must_record_run_queue_content_hash(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "open_calibration_queue.jsonl").write_text(
        "\n".join(
            json.dumps({"calibration_case_id": case_id})
            for case_id in ["case_a", "case_b"]
        )
        + "\n",
        encoding="utf-8",
    )
    summary_path = run_dir / "open_calibration_summary.json"
    payload = _valid_open_calibration_summary()
    payload["n_cases"] = 2
    payload["irr"]["n_cases"] = 2
    payload["adjudicated_rows"] = [
        {"calibration_case_id": "case_a"},
        {"calibration_case_id": "case_b"},
    ]
    summary_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="inputs.queue.content_sha256 is not recorded"):
        load_open_calibration_summary(summary_path, run_dir=run_dir)


def test_open_calibration_summary_must_match_run_queue_content_hash(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    queue_path = run_dir / "open_calibration_queue.jsonl"
    queue_path.write_text(
        "\n".join(
            json.dumps(row)
            for row in [
                {"calibration_case_id": "case_a", "primary_judge_grade": "CORRECT"},
                {"calibration_case_id": "case_b", "primary_judge_grade": "INCORRECT"},
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    summary_path = run_dir / "open_calibration_summary.json"
    payload = _valid_open_calibration_summary()
    payload["n_cases"] = 2
    payload["irr"]["n_cases"] = 2
    payload["adjudicated_rows"] = [
        {"calibration_case_id": "case_a"},
        {"calibration_case_id": "case_b"},
    ]
    payload["inputs"] = {
        "queue": {
            "path": str(queue_path),
            "content_sha256": "0" * 64,
        },
    }
    summary_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="content_sha256 does not match"):
        load_open_calibration_summary(summary_path, run_dir=run_dir)

    payload["inputs"]["queue"]["content_sha256"] = hashlib.sha256(
        queue_path.read_bytes()
    ).hexdigest()
    summary_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    summary = load_open_calibration_summary(summary_path, run_dir=run_dir)

    assert summary is not None
    assert summary["sampling"]["n_cases"] == 2


def test_malformed_open_adjudication_blocks_claimability_with_valid_calibration(
    tmp_path: Path,
) -> None:
    row = _analysis_row(sample_id="s1", open_correct=True)
    row["open_adjudication"] = {"parse_result": {"valid": False}}
    calibration = normalize_open_calibration_summary(_valid_open_calibration_summary())

    summary = summarize_open_grading(
        [row],
        adjudication_path=tmp_path / "open_adjudication.jsonl",
        adjudication_run=None,
        calibration_summary=calibration,
    )

    assert summary["effective_grade_source_counts"] == {"adjudication_unknown": 1}
    assert summary["claimable_open_correctness"] is False
    assert summary["claimability_blocker"] == "judge_verdict_unknown_or_error"


def test_open_grading_blocks_when_calibration_records_rule_gap(
    tmp_path: Path,
) -> None:
    row = _analysis_row(sample_id="s1", open_correct=False)
    row["open_adjudication"] = {"judge_grade": "CORRECT"}
    calibration_payload = _valid_open_calibration_summary()
    calibration_payload["adjudication"]["rule_gap_cases"] = {"count": 1, "n": 1}
    calibration = normalize_open_calibration_summary(calibration_payload)

    summary = summarize_open_grading(
        [row],
        adjudication_path=tmp_path / "open_adjudication.jsonl",
        adjudication_run=None,
        calibration_summary=calibration,
    )

    assert summary["claimable_open_correctness"] is False
    assert summary["claimability_blocker"] == "calibration_rule_gap_recorded"
    assert summary["calibration"]["claimability"]["blockers"] == [
        "calibration_rule_gap_recorded"
    ]


@pytest.mark.parametrize(
    ("field", "value", "blocker"),
    [
        ("cohen_kappa", float("nan"), "cohen_kappa_not_recorded"),
        ("gwet_ac1", float("inf"), "gwet_ac1_not_recorded"),
    ],
)
def test_open_calibration_rejects_non_finite_irr_statistics(
    field: str,
    value: float,
    blocker: str,
) -> None:
    calibration_payload = _valid_open_calibration_summary()
    calibration_payload["irr"][field] = value

    calibration = normalize_open_calibration_summary(calibration_payload)

    assert calibration["irr"][field] is None
    assert calibration["claimability"]["passed"] is False
    assert blocker in calibration["claimability"]["blockers"]


def test_open_calibration_queue_deduplicates_disagreements_and_stratifies() -> None:
    disagreement_a = _analysis_row(
        sample_id="truthful_ord0",
        base_sample_id="truthful_base",
        condition="selected",
        option_order_replicate=0,
        dataset="truthfulqa",
        open_correct=False,
    )
    disagreement_b = _analysis_row(
        sample_id="truthful_ord1",
        base_sample_id="truthful_base",
        condition="unhooked",
        option_order_replicate=1,
        dataset="truthfulqa",
        open_correct=False,
    )
    for row in [disagreement_a, disagreement_b]:
        row["open_generation"]["response"] = "A correct paraphrase."
        row["open_adjudication"] = {"judge_grade": "CORRECT"}

    bridge_correct = _analysis_row(
        sample_id="bridge_correct",
        base_sample_id="bridge_correct",
        dataset="triviaqa_bridge",
        open_correct=True,
    )
    bridge_correct["open_adjudication"] = {"judge_grade": "CORRECT"}
    bridge_incorrect = _analysis_row(
        sample_id="bridge_incorrect",
        base_sample_id="bridge_incorrect",
        dataset="triviaqa_bridge",
        open_correct=False,
    )
    bridge_incorrect["open_adjudication"] = {"judge_grade": "INCORRECT"}

    queue = build_open_calibration_queue(
        [disagreement_a, disagreement_b, bridge_correct, bridge_incorrect],
        stratified_per_cell=1,
        seed=1,
    )

    assert [row["sample_source"] for row in queue].count(
        "audit_queue_disagreement"
    ) == 1
    assert len(queue) == 3
    assert len({row["calibration_case_id"] for row in queue}) == 3
    assert {
        row["primary_judge_grade"]
        for row in queue
        if row["sample_source"].startswith("stratified_panel::")
    } == {"CORRECT", "INCORRECT"}


def _open_calibration_queue_rows(grades: list[str]) -> list[dict]:
    return [
        {
            "calibration_case_id": f"case_{idx:03d}",
            "sample_source": (
                "audit_queue_disagreement"
                if idx == 0
                else f"stratified_panel::grade={grade}"
            ),
            "sample_id": f"s{idx}",
            "condition": "selected",
            "alpha": 0.0,
            "dataset": "truthfulqa" if idx % 2 == 0 else "triviaqa_bridge",
            "primary_judge_grade": grade,
        }
        for idx, grade in enumerate(grades)
    ]


def _open_calibration_rule_metadata(rule_path: Path) -> dict[str, str]:
    return {
        "path": str(rule_path),
        "git_commit": "test",
        "content_sha256": hashlib.sha256(rule_path.read_bytes()).hexdigest(),
    }


def _open_calibration_secondary_rows(
    queue_rows: list[dict], rule_metadata: dict[str, str], *, grade: str | None = None
) -> list[dict]:
    return [
        {
            "schema_version": "simid_open_calibration_secondary_label/v1",
            "calibration_case_id": row["calibration_case_id"],
            "queue_row_sha256": queue_row_sha256(row),
            "judge_grade": row["primary_judge_grade"] if grade is None else grade,
            "rater": {
                "type": "llm",
                "model": "gpt-5.5",
                "prompt_version": "simid_open_calibration_rule/v1",
            },
            "rule": rule_metadata,
        }
        for row in queue_rows
    ]


def _open_calibration_adjudication_row(
    queue_row: dict,
    rule_metadata: dict[str, str],
    *,
    judge_grade: str = "INCORRECT",
    secondary_grade: str = "CORRECT",
    queue_hash: str | None = None,
    model: str = "gpt-5.5",
    prompt_version: str = "simid_open_calibration_rule/v1",
    schema_version: str = "simid_open_calibration_adjudication/v1",
) -> dict:
    return {
        "schema_version": schema_version,
        "calibration_case_id": queue_row["calibration_case_id"],
        "queue_row_sha256": (
            queue_row_sha256(queue_row) if queue_hash is None else queue_hash
        ),
        "judge_grade": judge_grade,
        "primary_grade": queue_row["primary_judge_grade"],
        "secondary_grade": secondary_grade,
        "adjudicator": {
            "type": "llm",
            "model": model,
            "prompt_version": prompt_version,
        },
        "rule": rule_metadata,
    }


def test_finalize_open_calibration_records_metrics_and_claimability(
    tmp_path: Path,
) -> None:
    rule_path = tmp_path / "simid_open_rule.md"
    rule_path.write_text("# Frozen SIMID open calibration rule\n", encoding="utf-8")
    grades = ["CORRECT", "INCORRECT", "NOT_ATTEMPTED"]
    queue_rows = _open_calibration_queue_rows(
        [grades[idx % len(grades)] for idx in range(100)]
    )
    rule_metadata = _open_calibration_rule_metadata(rule_path)
    secondary_rows = _open_calibration_secondary_rows(queue_rows, rule_metadata)

    summary = finalize_open_calibration(
        queue_rows=queue_rows,
        secondary_rows=secondary_rows,
        adjudication_rows=[],
        rule_path=rule_path,
        primary_rater_name="gpt-4o",
        secondary_rater_name="human_subset",
    )

    assert summary["schema_version"] == "simid_open_calibration/v1"
    assert summary["sampling"]["audit_queue_disagreement_n"] == 1
    assert summary["sampling"]["stratified_panel_n"] == 99
    assert summary["irr"]["cohen_kappa"] == pytest.approx(1.0)
    assert summary["irr"]["gwet_ac1"] == pytest.approx(1.0)
    assert summary["adjudication"]["rule_gap_cases"]["count"] == 0
    assert summary["raters"]["secondary_model"] == "gpt-5.5"
    assert summary["raters"]["adjudicator_model"] is None
    assert summary["raters"]["prompt_version"] == "simid_open_calibration_rule/v1"
    assert summary["claimability"]["passed"] is True


def test_finalize_open_calibration_keeps_small_diagnostic_summary(
    tmp_path: Path,
) -> None:
    rule_path = tmp_path / "simid_open_rule.md"
    rule_path.write_text("# Frozen SIMID open calibration rule\n", encoding="utf-8")
    queue_rows = _open_calibration_queue_rows(["CORRECT", "INCORRECT", "NOT_ATTEMPTED"])
    rule_metadata = _open_calibration_rule_metadata(rule_path)
    secondary_rows = _open_calibration_secondary_rows(queue_rows, rule_metadata)

    summary = finalize_open_calibration(
        queue_rows=queue_rows,
        secondary_rows=secondary_rows,
        adjudication_rows=[],
        rule_path=rule_path,
        primary_rater_name="gpt-4o",
        secondary_rater_name="human_subset",
    )

    assert summary["n_cases"] == 3
    assert summary["irr"]["cohen_kappa"] == pytest.approx(1.0)
    assert summary["claimability"]["passed"] is False
    assert summary["claimability"]["blocker"] == (
        "calibration_evidence_failed_thresholds"
    )
    assert summary["claimability"]["blockers"] == ["n_cases_below_minimum"]


def test_finalize_open_calibration_leaves_single_grade_kappa_unrecorded(
    tmp_path: Path,
) -> None:
    rule_path = tmp_path / "simid_open_rule.md"
    rule_path.write_text("# Frozen SIMID open calibration rule\n", encoding="utf-8")
    queue_rows = _open_calibration_queue_rows(["CORRECT"] * 100)
    rule_metadata = _open_calibration_rule_metadata(rule_path)
    secondary_rows = _open_calibration_secondary_rows(queue_rows, rule_metadata)

    summary = finalize_open_calibration(
        queue_rows=queue_rows,
        secondary_rows=secondary_rows,
        adjudication_rows=[],
        rule_path=rule_path,
        primary_rater_name="gpt-4o",
        secondary_rater_name="human_subset",
    )

    assert summary["irr"]["cohen_kappa"] is None
    assert summary["irr"]["gwet_ac1"] == pytest.approx(1.0)
    assert summary["claimability"]["passed"] is False
    assert "cohen_kappa_not_recorded" in summary["claimability"]["blockers"]
    assert "n_cases_below_minimum" not in summary["claimability"]["blockers"]


def test_open_calibration_labeler_uses_gpt5_chat_kwargs() -> None:
    kwargs = chat_completion_kwargs_for_model(
        "gpt-5.5",
        max_output_tokens=32,
        reasoning_effort="auto",
    )

    assert kwargs == {"max_completion_tokens": 32, "reasoning_effort": "none"}


def test_open_calibration_labeler_omits_auto_reasoning_for_legacy_gpt5() -> None:
    kwargs = chat_completion_kwargs_for_model(
        "gpt-5",
        max_output_tokens=32,
        reasoning_effort="auto",
    )

    assert kwargs == {"max_completion_tokens": 32}


def test_open_calibration_labeler_rejects_unsupported_gpt5_none() -> None:
    with pytest.raises(ValueError, match="does not support --reasoning-effort none"):
        chat_completion_kwargs_for_model(
            "gpt-5",
            max_output_tokens=32,
            reasoning_effort="none",
        )


def test_open_calibration_labeler_uses_high_reasoning_for_gpt5_pro_auto() -> None:
    kwargs = chat_completion_kwargs_for_model(
        "gpt-5.5-pro",
        max_output_tokens=32,
        reasoning_effort="auto",
    )

    assert kwargs == {"max_completion_tokens": 32, "reasoning_effort": "high"}


def test_open_calibration_labeler_uses_legacy_chat_kwargs() -> None:
    kwargs = chat_completion_kwargs_for_model(
        "gpt-4.1",
        max_output_tokens=32,
        reasoning_effort="none",
    )

    assert kwargs == {"temperature": 0.0, "max_tokens": 32}


def test_open_calibration_disagreement_rows_only_primary_secondary_mismatches() -> None:
    queue_rows = _open_calibration_queue_rows(["CORRECT", "INCORRECT", "NOT_ATTEMPTED"])
    secondary_labels = {
        "case_000": "CORRECT",
        "case_001": "CORRECT",
        "case_002": "NOT_ATTEMPTED",
    }

    rows = disagreement_rows(queue_rows, secondary_labels=secondary_labels)

    assert [row["calibration_case_id"] for row in rows] == ["case_001"]


def test_open_calibration_custom_id_depends_on_rule_hash() -> None:
    first = case_custom_id(
        "case_000",
        mode="secondary",
        model="gpt-5.5",
        rule_metadata={"content_sha256": "rule-a"},
        queue_row_fingerprint="row-a",
    )
    second = case_custom_id(
        "case_000",
        mode="secondary",
        model="gpt-5.5",
        rule_metadata={"content_sha256": "rule-b"},
        queue_row_fingerprint="row-a",
    )

    assert first != second


def test_open_calibration_custom_id_depends_on_queue_row_hash() -> None:
    first = case_custom_id(
        "case_000",
        mode="secondary",
        model="gpt-5.5",
        rule_metadata={"content_sha256": "rule-a"},
        queue_row_fingerprint="row-a",
    )
    second = case_custom_id(
        "case_000",
        mode="secondary",
        model="gpt-5.5",
        rule_metadata={"content_sha256": "rule-a"},
        queue_row_fingerprint="row-b",
    )

    assert first != second


def test_open_calibration_secondary_request_id_depends_on_queue_row_hash() -> None:
    queue_row = {
        "calibration_case_id": "case_000",
        "question": "What is the capital of France?",
        "gold_aliases": ["Paris"],
        "response": "Paris",
        "sample_source": "audit_queue_disagreement",
    }
    changed_row = {**queue_row, "sample_source": "stratified_panel::grade=CORRECT"}
    rule_metadata = {"content_sha256": "rule-a"}

    first_requests, _ = build_secondary_requests(
        [queue_row],
        existing_rows={},
        rule_text="rule",
        rule_metadata=rule_metadata,
        model="gpt-5.5",
        max_output_tokens=32,
        reasoning_effort="auto",
    )
    second_requests, _ = build_secondary_requests(
        [changed_row],
        existing_rows={},
        rule_text="rule",
        rule_metadata=rule_metadata,
        model="gpt-5.5",
        max_output_tokens=32,
        reasoning_effort="auto",
    )

    assert first_requests[0]["custom_id"] != second_requests[0]["custom_id"]


def test_open_calibration_adjudication_custom_id_depends_on_labels() -> None:
    rule_metadata = {"content_sha256": "rule-a"}
    first = case_custom_id(
        "case_000",
        mode="adjudicate",
        model="gpt-5.5",
        rule_metadata=rule_metadata,
        queue_row_fingerprint="row-a",
        prompt_inputs={"primary_label": "CORRECT", "secondary_label": "INCORRECT"},
    )
    second = case_custom_id(
        "case_000",
        mode="adjudicate",
        model="gpt-5.5",
        rule_metadata=rule_metadata,
        queue_row_fingerprint="row-a",
        prompt_inputs={"primary_label": "CORRECT", "secondary_label": "CORRECT"},
    )

    assert first != second


def test_open_calibration_adjudication_request_id_depends_on_queue_row_hash() -> None:
    queue_row = _open_calibration_queue_rows(["CORRECT"])[0]
    changed_row = {**queue_row, "sample_source": "stratified_panel::grade=CORRECT"}
    rule_metadata = {"content_sha256": "rule-a"}

    first_requests, _ = build_adjudication_requests(
        [queue_row],
        secondary_labels={"case_000": "INCORRECT"},
        existing_rows={},
        rule_text="rule",
        rule_metadata=rule_metadata,
        model="gpt-5.5",
        max_output_tokens=32,
        reasoning_effort="auto",
    )
    second_requests, _ = build_adjudication_requests(
        [changed_row],
        secondary_labels={"case_000": "INCORRECT"},
        existing_rows={},
        rule_text="rule",
        rule_metadata=rule_metadata,
        model="gpt-5.5",
        max_output_tokens=32,
        reasoning_effort="auto",
    )

    assert first_requests[0]["custom_id"] != second_requests[0]["custom_id"]


def test_open_calibration_existing_rows_validate_rule_hash() -> None:
    rule_metadata = {
        "path": "data/judge_validation/simid_open_calibration/adjudication_rule.md",
        "git_commit": "abc123",
        "content_sha256": "current-rule",
    }
    row = make_secondary_row(
        {"calibration_case_id": "case_000"},
        model="gpt-5.5",
        rule_metadata=rule_metadata,
        raw_content="CORRECT",
    )

    validate_existing_output_rows(
        {"case_000": row},
        output_kind="secondary-rater",
        actor_key="rater",
        model="gpt-5.5",
        rule_metadata=rule_metadata,
    )

    stale_rule = {**rule_metadata, "content_sha256": "stale-rule"}
    with pytest.raises(ValueError, match="current frozen rule hash"):
        validate_existing_output_rows(
            {"case_000": row},
            output_kind="secondary-rater",
            actor_key="rater",
            model="gpt-5.5",
            rule_metadata=stale_rule,
        )


def test_open_calibration_secondary_rows_record_queue_row_hash() -> None:
    rule_metadata = {
        "path": "data/judge_validation/simid_open_calibration/adjudication_rule.md",
        "git_commit": "abc123",
        "content_sha256": "current-rule",
    }
    queue_row = {
        "calibration_case_id": "case_000",
        "question": "What is the capital of France?",
        "gold_aliases": ["Paris"],
        "response": "Paris",
        "judge_grade": "CORRECT",
    }

    row = make_secondary_row(
        queue_row,
        model="gpt-5.5",
        rule_metadata=rule_metadata,
        raw_content="CORRECT",
    )

    assert row["queue_row_sha256"] == queue_row_sha256(queue_row)


def test_open_calibration_secondary_queue_context_detects_drift() -> None:
    rule_metadata = {
        "path": "data/judge_validation/simid_open_calibration/adjudication_rule.md",
        "git_commit": "abc123",
        "content_sha256": "current-rule",
    }
    queue_row = {
        "calibration_case_id": "case_000",
        "question": "What is the capital of France?",
        "gold_aliases": ["Paris"],
        "response": "Paris",
        "judge_grade": "CORRECT",
    }
    secondary = make_secondary_row(
        queue_row,
        model="gpt-5.5",
        rule_metadata=rule_metadata,
        raw_content="CORRECT",
    )

    validate_secondary_queue_context(
        {"case_000": secondary},
        queue_by_case={"case_000": queue_row},
    )
    with pytest.raises(ValueError, match="different calibration queue row"):
        validate_secondary_queue_context(
            {"case_000": secondary},
            queue_by_case={"case_000": {**queue_row, "response": "Lyon"}},
        )


def test_open_calibration_secondary_queue_context_rejects_missing_hash() -> None:
    unfingerprinted_secondary = {
        "calibration_case_id": "case_000",
        "judge_grade": "CORRECT",
    }
    with pytest.raises(ValueError, match="missing queue_row_sha256"):
        validate_secondary_queue_context(
            {"case_000": unfingerprinted_secondary},
            queue_by_case={"case_000": {"calibration_case_id": "case_000"}},
        )


def test_open_calibration_adjudication_rows_record_queue_row_hash() -> None:
    rule_metadata = {
        "path": "data/judge_validation/simid_open_calibration/adjudication_rule.md",
        "git_commit": "abc123",
        "content_sha256": "current-rule",
    }
    queue_row = {
        "calibration_case_id": "case_000",
        "question": "What is the capital of France?",
        "gold_aliases": ["Paris"],
        "response": "Paris",
        "primary_judge_grade": "CORRECT",
    }

    row = make_adjudication_row(
        queue_row,
        secondary_label="INCORRECT",
        model="gpt-5.5",
        rule_metadata=rule_metadata,
        raw_content=json.dumps(
            {"label": "CORRECT", "rule_gap": False, "notes": "primary was right"}
        ),
    )

    assert row["queue_row_sha256"] == queue_row_sha256(queue_row)


def test_open_calibration_adjudication_queue_context_detects_drift() -> None:
    queue_row = {
        "calibration_case_id": "case_000",
        "question": "What is the capital of France?",
        "response": "Paris",
        "primary_judge_grade": "CORRECT",
    }
    adjudication = {
        "calibration_case_id": "case_000",
        "queue_row_sha256": queue_row_sha256(queue_row),
        "judge_grade": "CORRECT",
        "primary_grade": "CORRECT",
        "secondary_grade": "INCORRECT",
    }

    validate_existing_adjudication_context(
        {"case_000": adjudication},
        queue_by_case={"case_000": queue_row},
        secondary_labels={"case_000": "INCORRECT"},
    )
    with pytest.raises(ValueError, match="different calibration queue row"):
        validate_existing_adjudication_context(
            {"case_000": adjudication},
            queue_by_case={"case_000": {**queue_row, "response": "Lyon"}},
            secondary_labels={"case_000": "INCORRECT"},
        )


def test_open_calibration_adjudication_queue_context_rejects_missing_hash() -> None:
    with pytest.raises(ValueError, match="missing queue_row_sha256"):
        validate_existing_adjudication_context(
            {
                "case_000": {
                    "calibration_case_id": "case_000",
                    "judge_grade": "CORRECT",
                    "primary_grade": "CORRECT",
                    "secondary_grade": "INCORRECT",
                }
            },
            queue_by_case={
                "case_000": {
                    "calibration_case_id": "case_000",
                    "primary_judge_grade": "CORRECT",
                }
            },
            secondary_labels={"case_000": "INCORRECT"},
        )


def test_finalize_open_calibration_rejects_stale_secondary_queue_hash(
    tmp_path: Path,
) -> None:
    rule_path = tmp_path / "simid_open_rule.md"
    rule_path.write_text("# Frozen SIMID open calibration rule\n", encoding="utf-8")
    queue_rows = _open_calibration_queue_rows(["CORRECT", "INCORRECT"])
    rule_metadata = _open_calibration_rule_metadata(rule_path)
    secondary_rows = _open_calibration_secondary_rows(queue_rows, rule_metadata)

    with pytest.raises(ValueError, match="different calibration queue row"):
        finalize_open_calibration(
            queue_rows=[queue_rows[0], {**queue_rows[1], "sample_id": "changed"}],
            secondary_rows=secondary_rows,
            adjudication_rows=[],
            rule_path=rule_path,
            primary_rater_name="gpt-5.5",
            secondary_rater_name="gpt-5.5",
        )


def test_finalize_open_calibration_rejects_missing_secondary_queue_hash(
    tmp_path: Path,
) -> None:
    rule_path = tmp_path / "simid_open_rule.md"
    rule_path.write_text("# Frozen SIMID open calibration rule\n", encoding="utf-8")
    queue_rows = _open_calibration_queue_rows(["CORRECT", "INCORRECT"])
    rule_metadata = _open_calibration_rule_metadata(rule_path)
    secondary_rows = _open_calibration_secondary_rows(queue_rows, rule_metadata)
    del secondary_rows[0]["queue_row_sha256"]

    with pytest.raises(ValueError, match="missing queue_row_sha256"):
        finalize_open_calibration(
            queue_rows=queue_rows,
            secondary_rows=secondary_rows,
            adjudication_rows=[],
            rule_path=rule_path,
            primary_rater_name="gpt-5.5",
            secondary_rater_name="gpt-5.5",
        )


def test_finalize_open_calibration_rejects_stale_rule_hash(
    tmp_path: Path,
) -> None:
    rule_path = tmp_path / "simid_open_rule.md"
    rule_path.write_text("# Frozen SIMID open calibration rule\n", encoding="utf-8")
    queue_rows = _open_calibration_queue_rows(["CORRECT", "INCORRECT"])
    stale_rule_metadata = {
        "path": str(rule_path),
        "git_commit": "test",
        "content_sha256": "stale-rule",
    }
    secondary_rows = _open_calibration_secondary_rows(
        queue_rows,
        stale_rule_metadata,
    )

    with pytest.raises(ValueError, match="current frozen rule hash"):
        finalize_open_calibration(
            queue_rows=queue_rows,
            secondary_rows=secondary_rows,
            adjudication_rows=[],
            rule_path=rule_path,
            primary_rater_name="gpt-5.5",
            secondary_rater_name="gpt-5.5",
        )


def test_finalize_open_calibration_rejects_stale_secondary_metadata(
    tmp_path: Path,
) -> None:
    rule_path = tmp_path / "simid_open_rule.md"
    rule_path.write_text("# Frozen SIMID open calibration rule\n", encoding="utf-8")
    queue_rows = _open_calibration_queue_rows(["CORRECT", "INCORRECT"])
    rule_metadata = _open_calibration_rule_metadata(rule_path)
    secondary_rows = _open_calibration_secondary_rows(queue_rows, rule_metadata)
    secondary_rows[0]["rater"] = {
        "type": "llm",
        "model": "wrong-model",
        "prompt_version": "simid_open_calibration_rule/v1",
    }

    with pytest.raises(ValueError, match="expected 'gpt-5.5'"):
        finalize_open_calibration(
            queue_rows=queue_rows,
            secondary_rows=secondary_rows,
            adjudication_rows=[],
            rule_path=rule_path,
            primary_rater_name="gpt-5.5",
            secondary_rater_name="gpt-5.5",
        )


def test_finalize_open_calibration_rejects_stale_adjudication_queue_hash(
    tmp_path: Path,
) -> None:
    rule_path = tmp_path / "simid_open_rule.md"
    rule_path.write_text("# Frozen SIMID open calibration rule\n", encoding="utf-8")
    queue_rows = _open_calibration_queue_rows(["CORRECT", "INCORRECT"])
    rule_metadata = _open_calibration_rule_metadata(rule_path)
    secondary_rows = _open_calibration_secondary_rows(
        queue_rows,
        rule_metadata,
        grade="CORRECT",
    )

    with pytest.raises(ValueError, match="different calibration queue row"):
        finalize_open_calibration(
            queue_rows=queue_rows,
            secondary_rows=secondary_rows,
            adjudication_rows=[
                _open_calibration_adjudication_row(
                    queue_rows[1],
                    rule_metadata,
                    queue_hash=queue_row_sha256(queue_rows[0]),
                )
            ],
            rule_path=rule_path,
            primary_rater_name="gpt-5.5",
            secondary_rater_name="gpt-5.5",
        )


def test_finalize_open_calibration_rejects_missing_adjudication_queue_hash(
    tmp_path: Path,
) -> None:
    rule_path = tmp_path / "simid_open_rule.md"
    rule_path.write_text("# Frozen SIMID open calibration rule\n", encoding="utf-8")
    queue_rows = _open_calibration_queue_rows(["CORRECT", "INCORRECT"])
    rule_metadata = _open_calibration_rule_metadata(rule_path)
    secondary_rows = _open_calibration_secondary_rows(
        queue_rows,
        rule_metadata,
        grade="CORRECT",
    )

    with pytest.raises(ValueError, match="missing queue_row_sha256"):
        finalize_open_calibration(
            queue_rows=queue_rows,
            secondary_rows=secondary_rows,
            adjudication_rows=[
                {
                    "schema_version": "simid_open_calibration_adjudication/v1",
                    "calibration_case_id": "case_001",
                    "judge_grade": "INCORRECT",
                    "primary_grade": "INCORRECT",
                    "secondary_grade": "CORRECT",
                    "adjudicator": {
                        "type": "llm",
                        "model": "gpt-5.5",
                        "prompt_version": "simid_open_calibration_rule/v1",
                    },
                    "rule": rule_metadata,
                }
            ],
            rule_path=rule_path,
            primary_rater_name="gpt-5.5",
            secondary_rater_name="gpt-5.5",
        )


def test_finalize_open_calibration_rejects_stale_adjudication_rule_hash(
    tmp_path: Path,
) -> None:
    rule_path = tmp_path / "simid_open_rule.md"
    rule_path.write_text("# Frozen SIMID open calibration rule\n", encoding="utf-8")
    queue_rows = _open_calibration_queue_rows(["CORRECT", "INCORRECT"])
    rule_metadata = _open_calibration_rule_metadata(rule_path)
    secondary_rows = _open_calibration_secondary_rows(
        queue_rows,
        rule_metadata,
        grade="CORRECT",
    )
    stale_rule_metadata = {**rule_metadata, "content_sha256": "stale-rule"}

    with pytest.raises(ValueError, match="current frozen rule hash"):
        finalize_open_calibration(
            queue_rows=queue_rows,
            secondary_rows=secondary_rows,
            adjudication_rows=[
                _open_calibration_adjudication_row(
                    queue_rows[1],
                    stale_rule_metadata,
                )
            ],
            rule_path=rule_path,
            primary_rater_name="gpt-5.5",
            secondary_rater_name="gpt-5.5",
        )


def test_finalize_open_calibration_rejects_stale_adjudication_metadata(
    tmp_path: Path,
) -> None:
    rule_path = tmp_path / "simid_open_rule.md"
    rule_path.write_text("# Frozen SIMID open calibration rule\n", encoding="utf-8")
    queue_rows = _open_calibration_queue_rows(["CORRECT", "INCORRECT"])
    rule_metadata = _open_calibration_rule_metadata(rule_path)
    secondary_rows = _open_calibration_secondary_rows(
        queue_rows,
        rule_metadata,
        grade="CORRECT",
    )

    with pytest.raises(ValueError, match="prompt_version"):
        finalize_open_calibration(
            queue_rows=queue_rows,
            secondary_rows=secondary_rows,
            adjudication_rows=[
                _open_calibration_adjudication_row(
                    queue_rows[1],
                    rule_metadata,
                    prompt_version="stale-prompt/v0",
                )
            ],
            rule_path=rule_path,
            primary_rater_name="gpt-5.5",
            secondary_rater_name="gpt-5.5",
        )


def test_finalize_open_calibration_rejects_adjudication_context_drift(
    tmp_path: Path,
) -> None:
    rule_path = tmp_path / "simid_open_rule.md"
    rule_path.write_text("# Frozen SIMID open calibration rule\n", encoding="utf-8")
    queue_rows = _open_calibration_queue_rows(["CORRECT", "INCORRECT"])
    rule_metadata = _open_calibration_rule_metadata(rule_path)
    secondary_rows = _open_calibration_secondary_rows(
        queue_rows,
        rule_metadata,
        grade="CORRECT",
    )

    with pytest.raises(ValueError, match="secondary_grade"):
        finalize_open_calibration(
            queue_rows=queue_rows,
            secondary_rows=secondary_rows,
            adjudication_rows=[
                _open_calibration_adjudication_row(
                    queue_rows[1],
                    rule_metadata,
                    secondary_grade="NOT_ATTEMPTED",
                )
            ],
            rule_path=rule_path,
            primary_rater_name="gpt-5.5",
            secondary_rater_name="gpt-5.5",
        )


def test_open_calibration_partial_batch_errors_raise_after_successes(
    tmp_path: Path,
) -> None:
    output = tmp_path / "secondary.jsonl"

    with pytest.raises(RuntimeError, match="could not be materialized"):
        raise_after_partial_write(
            ["case_001: missing batch result"], output_path=output
        )


def test_open_calibration_adjudication_parser_requires_rule_gap_boolean() -> None:
    with pytest.raises(ValueError, match="rule_gap must be boolean"):
        parse_adjudication_payload(
            json.dumps(
                {
                    "label": "CORRECT",
                    "rule_gap": "false",
                    "notes": "R1",
                }
            )
        )


def test_report_labels_adjudicated_open_metrics(tmp_path: Path) -> None:
    path = tmp_path / "report.md"
    write_report(
        {
            "open_grading": {
                "metric_correct": "adjudicated_open_correct",
                "metric_attempted": "adjudicated_open_attempted",
                "effective_grade_source_counts": {"adjudication": 2},
                "claimable_open_correctness": False,
                "claimability_blocker": "calibration_evidence_not_recorded",
            },
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
                            "adjudicated_open_correct": {
                                "estimate": 0.5,
                                "ci": {"lower": 0.25, "upper": 0.75},
                            },
                        }
                    },
                    "paired_deltas_vs_baseline": {},
                }
            },
        },
        path,
    )

    text = path.read_text(encoding="utf-8")
    assert "reported as adjudicated_open_correct" in text
    assert "diagnostic-only" in text
    assert "adjudicated_open_correct=0.5000 [0.2500, 0.7500]" in text
    assert "calibration_evidence_not_recorded" in text


def test_report_labels_claimable_open_metrics_after_calibration(tmp_path: Path) -> None:
    path = tmp_path / "report.md"
    calibration = normalize_open_calibration_summary(
        _valid_open_calibration_summary(),
        path=tmp_path / "open_calibration_summary.json",
    )
    write_report(
        {
            "open_grading": {
                "metric_correct": "adjudicated_open_correct",
                "metric_attempted": "adjudicated_open_attempted",
                "effective_grade_source_counts": {"adjudication": 2},
                "claimable_open_correctness": True,
                "claimability_blocker": None,
                "calibration": calibration,
            },
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
                            "adjudicated_open_correct": {
                                "estimate": 0.5,
                                "ci": {"lower": 0.25, "upper": 0.75},
                            },
                        }
                    },
                    "paired_deltas_vs_baseline": {},
                }
            },
        },
        path,
    )

    text = path.read_text(encoding="utf-8")
    assert "Open correctness claimability: passed." in text
    assert "kappa=0.9100" in text
    assert "AC1=0.9400" in text
    assert "diagnostic-only" not in text


def test_report_distinguishes_failed_from_missing_open_calibration(
    tmp_path: Path,
) -> None:
    path = tmp_path / "report.md"
    calibration_payload = _valid_open_calibration_summary()
    calibration_payload["irr"]["cohen_kappa"] = 0.79
    calibration = normalize_open_calibration_summary(
        calibration_payload,
        path=tmp_path / "open_calibration_summary.json",
    )
    write_report(
        {
            "open_grading": {
                "metric_correct": "adjudicated_open_correct",
                "metric_attempted": "adjudicated_open_attempted",
                "effective_grade_source_counts": {"adjudication": 2},
                "claimable_open_correctness": False,
                "claimability_blocker": "calibration_evidence_failed_thresholds",
                "calibration": calibration,
            },
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
                            "adjudicated_open_correct": {
                                "estimate": 0.5,
                                "ci": {"lower": 0.25, "upper": 0.75},
                            },
                        }
                    },
                    "paired_deltas_vs_baseline": {},
                }
            },
        },
        path,
    )

    text = path.read_text(encoding="utf-8")
    assert "calibration_evidence_failed_thresholds" in text
    assert "recorded calibration evidence did not pass" in text
    assert "kappa=0.7900" in text
    assert "calibration_evidence_not_recorded" not in text


def test_unknown_judge_verdict_does_not_become_claimable_correctness() -> None:
    row = _analysis_row(sample_id="s1", open_correct=True)
    row["open_adjudication"] = {"judge_grade": "MAYBE"}

    grade = effective_open_grade(row)

    assert grade["source"] == "adjudication_unknown"
    assert grade["correct"] is False
    assert grade["claimable"] is False


def test_unknown_judge_verdict_records_null_deterministic_disagreement(
    tmp_path: Path,
) -> None:
    row = _analysis_row(sample_id="s1", open_correct=True)

    adjudication = make_open_adjudication_row(
        row,
        judge_model="gpt-test",
        batch_custom_id="batch-1",
        batch_state_path=tmp_path / "open_adjudication.batch_state.json",
        batch_result_entry=None,
        raw_content="MAYBE",
    )

    assert adjudication["parse_result"]["valid"] is False
    assert adjudication["judge_disagrees_with_deterministic"] is None


def test_simid_open_batch_request_contains_question_aliases_and_response() -> None:
    row = _analysis_row(sample_id="s1", open_correct=False)
    row["question"] = "Who wrote Hamlet?"
    row["gold_aliases"] = ["William Shakespeare", "Shakespeare"]
    row["open_generation"]["response"] = "The answer is Shakespeare."

    request = build_simid_open_adjudication_request(row, judge_model="gpt-test")
    body = request["body"]
    prompt = body["messages"][0]["content"]

    assert body["model"] == "gpt-test"
    assert "Who wrote Hamlet?" in prompt
    assert "William Shakespeare; Shakespeare" in prompt
    assert "The answer is Shakespeare." in prompt


def test_simid_open_primary_custom_id_depends_on_judge_model() -> None:
    row = _analysis_row(
        sample_id="s1",
        dataset="truthfulqa",
        mc_endpoint="truthfulqa_mc1",
        open_correct=False,
    )

    first = simid_open_adjudication_custom_id(row, judge_model="gpt-4o")
    second = simid_open_adjudication_custom_id(row, judge_model="gpt-test")

    assert first != second


def test_simid_open_batch_deduplicates_identical_base_alpha_responses() -> None:
    rows = [
        _analysis_row(
            sample_id="s1_order0",
            base_sample_id="base1",
            condition="selected",
            option_order_replicate=0,
            dataset="truthfulqa",
            mc_endpoint="truthfulqa_mc1",
            open_correct=True,
        ),
        _analysis_row(
            sample_id="s1_order1",
            base_sample_id="base1",
            condition="selected",
            option_order_replicate=1,
            dataset="truthfulqa",
            mc_endpoint="truthfulqa_mc1",
            open_correct=True,
        ),
        _analysis_row(
            sample_id="s1_unhooked_order0",
            base_sample_id="base1",
            condition="unhooked",
            option_order_replicate=0,
            dataset="truthfulqa",
            mc_endpoint="truthfulqa_mc1",
            open_correct=True,
        ),
    ]

    requests, request_map = build_simid_open_adjudication_requests(
        rows,
        existing_adjudications={},
        judge_model="gpt-test",
    )

    assert len(requests) == 1
    assert len(request_map[requests[0]["custom_id"]]) == 3


def test_simid_open_batch_retries_non_final_existing_adjudication_rows() -> None:
    row = _analysis_row(
        sample_id="s1",
        dataset="truthfulqa",
        mc_endpoint="truthfulqa_mc1",
        open_correct=False,
    )
    key = ("s1", "selected", 0.0)

    requests, request_map = build_simid_open_adjudication_requests(
        [row],
        existing_adjudications={
            key: _open_adjudication_row_for(row, judge_grade="CORRECT")
        },
        judge_model="gpt-test",
    )
    assert requests == []
    assert request_map == {}

    requests, request_map = build_simid_open_adjudication_requests(
        [row],
        existing_adjudications={
            key: _open_adjudication_row_for(
                row,
                judge_grade="ERROR",
                valid=False,
            )
        },
        judge_model="gpt-test",
    )

    assert len(requests) == 1
    assert set(request_map) == {requests[0]["custom_id"]}
    assert request_map[requests[0]["custom_id"]] == [row]


def test_simid_open_batch_reuses_final_existing_dedup_adjudications() -> None:
    adjudicated = _analysis_row(
        sample_id="s1_order0",
        base_sample_id="base1",
        condition="selected",
        option_order_replicate=0,
        dataset="truthfulqa",
        mc_endpoint="truthfulqa_mc1",
        open_correct=False,
    )
    missing_sibling = _analysis_row(
        sample_id="s1_order1",
        base_sample_id="base1",
        condition="unhooked",
        option_order_replicate=1,
        dataset="truthfulqa",
        mc_endpoint="truthfulqa_mc1",
        open_correct=False,
    )

    requests, request_map = build_simid_open_adjudication_requests(
        [adjudicated, missing_sibling],
        existing_adjudications={
            ("s1_order0", "selected", 0.0): _open_adjudication_row_for(
                adjudicated,
                judge_grade="CORRECT",
            )
        },
        judge_model="gpt-test",
    )

    assert requests == []
    assert request_map == {}


def test_simid_open_existing_dedup_adjudication_fans_out_to_missing_sibling() -> None:
    adjudicated = _analysis_row(
        sample_id="s1_order0",
        base_sample_id="base1",
        condition="selected",
        option_order_replicate=0,
        dataset="truthfulqa",
        mc_endpoint="truthfulqa_mc1",
        open_correct=False,
    )
    missing_sibling = _analysis_row(
        sample_id="s1_order1",
        base_sample_id="base1",
        condition="unhooked",
        option_order_replicate=1,
        dataset="truthfulqa",
        mc_endpoint="truthfulqa_mc1",
        open_correct=False,
    )
    existing = {
        ("s1_order0", "selected", 0.0): {
            "schema_version": "simid_open_adjudication/v1",
            "sample_id": "s1_order0",
            "condition": "selected",
            "alpha": 0.0,
            "base_sample_id": "base1",
            "dataset": "truthfulqa",
            "mc_endpoint": "truthfulqa_mc1",
            "question": "Question?",
            "gold_aliases": ["Gold"],
            "response": "Wrong",
            "deterministic_open_grade": adjudicated["open_grade"],
            "judge_grade": "CORRECT",
            "parse_result": {"verdict": "CORRECT", "valid": True},
            "batch_custom_id": "simid_open_existing",
            "adjudicated_correct": True,
            "adjudicated_attempted": True,
            "judge_disagrees_with_deterministic": True,
        }
    }

    fanout_rows = fanout_existing_open_adjudications(
        [missing_sibling],
        existing,
    )

    assert len(fanout_rows) == 1
    fanout = fanout_rows[0]
    assert fanout["sample_id"] == "s1_order1"
    assert fanout["condition"] == "unhooked"
    assert fanout["judge_grade"] == "CORRECT"
    assert fanout["adjudicated_correct"] is True
    assert fanout["dedup_fanout_source"] == {
        "sample_id": "s1_order0",
        "condition": "selected",
        "alpha": 0.0,
    }
    assert existing[("s1_order1", "unhooked", 0.0)] == fanout


def test_simpleqa_parser_handles_word_and_letter_outputs() -> None:
    assert parse_simpleqa_verdict("CORRECT") == "CORRECT"
    assert parse_simpleqa_verdict("INCORRECT") == "INCORRECT"
    assert parse_simpleqa_verdict("NOT_ATTEMPTED") == "NOT_ATTEMPTED"
    assert parse_simpleqa_verdict("A") == "CORRECT"
    assert parse_simpleqa_verdict("B") == "INCORRECT"
    assert parse_simpleqa_verdict("C") == "NOT_ATTEMPTED"


def test_simpleqa_parser_does_not_match_correct_inside_incorrect() -> None:
    assert (
        parse_simpleqa_verdict("The judge marked this response INCORRECT.")
        == "INCORRECT"
    )


def test_simpleqa_parser_free_text_incorrect_without_letter_token() -> None:
    assert parse_simpleqa_verdict("Final verdict: INCORRECT") == "INCORRECT"


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
        "unadjudicated_non_bridge_deterministic_alias_correct_requires_adjudication"
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


def test_truthfulqa_builder_skips_rows_without_unique_distractors(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "truthfulqa.csv"
    csv_path.write_text(
        "Question,Best Answer,Correct Answers,Incorrect Answers,Category,Type,Source\n"
        "Duplicate distractor?,The Gold,The Gold,Gold,cat,type,src\n"
        "Usable?,Gold,Gold,Wrong,cat,type,src\n",
        encoding="utf-8",
    )
    metadata_path = tmp_path / "extraction_metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "question_ids_train": [],
                "question_ids_val": [],
                "question_ids_dev": [],
                "question_ids_test": [
                    stable_question_id("Duplicate distractor?"),
                    stable_question_id("Usable?"),
                ],
            }
        ),
        encoding="utf-8",
    )

    rows = build_truthfulqa_rows(
        csv_path=csv_path,
        seed=42,
        n_rows=1,
        option_order_replicates=1,
        model_path="model",
        tokenizer_path="model",
        iti_artifact_path=str(tmp_path / "iti_heads.pt"),
        iti_artifact_sha256="abc",
        leakage_policy="heldout_only",
        split_metadata_path=metadata_path,
    )

    assert len(rows) == 1
    assert rows[0]["question"] == "Usable?"
    assert rows[0]["truthfulqa_stable_id"] == stable_question_id("Usable?")


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

    assert (
        "data/contrastive/truthfulness/iti_truthfulqa_paperfaithful/"
        "final_fold0/iti_heads.pt"
    ) in text
    assert "SIMID_TRUTHFULQA_LEAKAGE_POLICY:-heldout_only" in text
    assert "SIMID_OPTION_ORDER_REPLICATES:-2" in text
    assert "SIMID_TRUTHFULQA_N:-100" in text
    assert "SIMID_BRIDGE_N:-100" in text
    assert "DEFAULT_MIN_TRUTHFULQA_ROWS" in text
    assert "DEFAULT_MIN_BRIDGE_ROWS" in text
    assert "analyze-adjudicated" in text
    assert "--adjudicate-open" in text
    assert "--phase0-gates" in text
    assert "--open-calibration-queue-output" in text
    assert "open_calibration_queue.jsonl" in text
    assert "SIMID_OPEN_CALIBRATION_STRATIFIED_PER_CELL:-20" in text


def test_simid_canary_synthetic_queue_satisfies_launch_validator() -> None:
    script = (
        Path(__file__).resolve().parent.parent
        / "scripts/infra/simid_canary_calibration.sh"
    )
    text = script.read_text(encoding="utf-8")
    _, after_marker = text.split("cat > \"$QUEUE_PATH\" <<'JSONL'\n", maxsplit=1)
    queue_text, _ = after_marker.split("\nJSONL\n", maxsplit=1)
    rows = [json.loads(line) for line in queue_text.splitlines() if line.strip()]

    assert len(rows) == 2
    validate_calibration_queue_for_launch(rows, allow_undersized=True)
    assert {row["sample_source"] for row in rows} == {
        "audit_queue_disagreement",
        "stratified_panel::canary",
    }
    assert {row["judge_grade"] for row in rows} == {"CORRECT", "INCORRECT"}
    assert "--allow-undersized-queue" in text
    assert "preserving temp dir for resume/inspection" in text
    assert "active-run-status" in text


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


# ---------------------------------------------------------------------------
# validate_calibration_queue_for_launch — pre-launch queue sanity checks.
# These tests live at the bottom of the file by convention; do not fold them
# into the boundary-test cluster above.
# ---------------------------------------------------------------------------


def _make_queue_row(
    case_id: str,
    *,
    sample_source: str,
    judge_grade: str = "CORRECT",
) -> dict:
    return {
        "calibration_case_id": case_id,
        "sample_source": sample_source,
        "sample_id": f"sample-{case_id}",
        "condition": "selected",
        "alpha": 1.0,
        "dataset": "simpleqa",
        "judge_grade": judge_grade,
    }


def _balanced_full_queue(n: int = 100) -> list[dict]:
    rows: list[dict] = []
    # Half from audit_queue_disagreement, half from stratified_panel::*.
    for i in range(n // 2):
        grade = "CORRECT" if i % 2 == 0 else "INCORRECT"
        rows.append(
            _make_queue_row(
                f"audit-{i:03d}",
                sample_source="audit_queue_disagreement",
                judge_grade=grade,
            )
        )
    for i in range(n - n // 2):
        grade = "CORRECT" if i % 2 == 0 else "NOT_ATTEMPTED"
        rows.append(
            _make_queue_row(
                f"strat-{i:03d}",
                sample_source="stratified_panel::cell_a",
                judge_grade=grade,
            )
        )
    return rows


def test_queue_validator_accepts_valid_full_queue() -> None:
    rows = _balanced_full_queue(100)
    # Must not raise.
    validate_calibration_queue_for_launch(rows)


def test_queue_validator_rejects_undersized_queue_and_honors_min_cases() -> None:
    rows = _balanced_full_queue(99)
    with pytest.raises(CalibrationQueueLaunchError) as exc_info:
        validate_calibration_queue_for_launch(rows)
    assert "n_cases" in str(exc_info.value)
    # Same queue passes when min_cases is lowered.
    validate_calibration_queue_for_launch(rows, min_cases=50)


def test_queue_validator_allow_undersized_bypasses_size_check_only() -> None:
    rows = _balanced_full_queue(20)
    # 20 rows: undersized but otherwise valid → bypass succeeds.
    validate_calibration_queue_for_launch(rows, allow_undersized=True)
    # But the bypass does not override duplicate / source / grade checks.
    rows_dup = list(rows)
    rows_dup.append(rows_dup[0])  # duplicate calibration_case_id
    with pytest.raises(CalibrationQueueLaunchError, match="duplicate"):
        validate_calibration_queue_for_launch(rows_dup, allow_undersized=True)


def test_queue_validator_rejects_duplicate_calibration_case_ids() -> None:
    rows = _balanced_full_queue(100)
    rows[5] = dict(rows[5], calibration_case_id=rows[0]["calibration_case_id"])
    with pytest.raises(CalibrationQueueLaunchError) as exc_info:
        validate_calibration_queue_for_launch(rows)
    assert "duplicate" in str(exc_info.value).lower()
    assert rows[0]["calibration_case_id"] in str(exc_info.value)


def test_queue_validator_rejects_queue_missing_audit_queue_source() -> None:
    rows = [
        _make_queue_row(
            f"strat-{i:03d}",
            sample_source="stratified_panel::cell_a",
            judge_grade="CORRECT" if i % 2 == 0 else "INCORRECT",
        )
        for i in range(100)
    ]
    with pytest.raises(CalibrationQueueLaunchError) as exc_info:
        validate_calibration_queue_for_launch(rows)
    assert "audit_queue" in str(exc_info.value)


def test_queue_validator_rejects_queue_missing_stratified_panel_source() -> None:
    rows = [
        _make_queue_row(
            f"audit-{i:03d}",
            sample_source="audit_queue_disagreement",
            judge_grade="CORRECT" if i % 2 == 0 else "INCORRECT",
        )
        for i in range(100)
    ]
    with pytest.raises(CalibrationQueueLaunchError) as exc_info:
        validate_calibration_queue_for_launch(rows)
    assert "stratified_panel" in str(exc_info.value)


def test_queue_validator_rejects_collapsed_class_queue() -> None:
    # All 100 rows have judge_grade=CORRECT → only 1 distinct primary grade.
    rows = _balanced_full_queue(100)
    rows = [dict(row, judge_grade="CORRECT") for row in rows]
    with pytest.raises(CalibrationQueueLaunchError) as exc_info:
        validate_calibration_queue_for_launch(rows)
    msg = str(exc_info.value).lower()
    assert "distinct" in msg and "grade" in msg


def test_queue_validator_reports_all_failures_in_one_message() -> None:
    # Multi-failure: undersized AND missing audit_queue AND duplicates AND
    # collapsed grades.
    rows = [
        _make_queue_row(
            "dup-id",
            sample_source="stratified_panel::cell_a",
            judge_grade="CORRECT",
        ),
        _make_queue_row(
            "dup-id",
            sample_source="stratified_panel::cell_a",
            judge_grade="CORRECT",
        ),
        _make_queue_row(
            "another",
            sample_source="stratified_panel::cell_b",
            judge_grade="CORRECT",
        ),
    ]
    with pytest.raises(CalibrationQueueLaunchError) as exc_info:
        validate_calibration_queue_for_launch(rows)
    msg = str(exc_info.value)
    assert "n_cases" in msg
    assert "duplicate" in msg.lower()
    assert "audit_queue" in msg
    assert "distinct" in msg.lower()


def test_queue_validator_rejects_invalid_primary_grade_before_downstream_use(
    tmp_path: Path,
) -> None:
    rows = _balanced_full_queue(100)
    rows[3] = dict(rows[3], judge_grade="MAYBE")
    with pytest.raises(CalibrationQueueLaunchError) as exc_info:
        validate_calibration_queue_for_launch(rows)
    msg = str(exc_info.value)
    assert "invalid primary grade" in msg
    assert "audit-003" in msg

    secondary_labels = {str(row["calibration_case_id"]): "CORRECT" for row in rows}
    with pytest.raises(ValueError, match="Invalid primary queue SIMID open grade"):
        disagreement_rows(rows, secondary_labels=secondary_labels)

    rule_path = tmp_path / "simid_open_rule.md"
    rule_path.write_text("# Frozen SIMID open calibration rule\n", encoding="utf-8")
    rule_metadata = _open_calibration_rule_metadata(rule_path)
    secondary_rows = _open_calibration_secondary_rows(
        rows,
        rule_metadata,
        grade="CORRECT",
    )
    with pytest.raises(ValueError, match="Invalid primary queue SIMID open grade"):
        finalize_open_calibration(
            queue_rows=rows,
            secondary_rows=secondary_rows,
            adjudication_rows=[],
            rule_path=rule_path,
            primary_rater_name="gpt-5.5",
            secondary_rater_name="gpt-5.5",
        )


def test_label_parse_args_exposes_allow_undersized_queue_flag() -> None:
    args = label_parse_args(
        [
            "--mode",
            "secondary",
            "--queue-path",
            "/tmp/q.jsonl",
            "--rule-path",
            "/tmp/rule.md",
            "--output",
            "/tmp/out.jsonl",
        ]
    )
    assert args.allow_undersized_queue is False
    args2 = label_parse_args(
        [
            "--mode",
            "secondary",
            "--queue-path",
            "/tmp/q.jsonl",
            "--rule-path",
            "/tmp/rule.md",
            "--output",
            "/tmp/out.jsonl",
            "--allow-undersized-queue",
        ]
    )
    assert args2.allow_undersized_queue is True
