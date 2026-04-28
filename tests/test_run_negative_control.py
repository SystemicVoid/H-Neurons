from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from run_intervention import _faitheval_prompt  # noqa: E402
from run_negative_control import (  # noqa: E402
    _jailbreak_csv2_eval_dir,
    assert_or_write_negative_control_run_contract,
    build_negative_control_run_contract,
    negative_control_schedule,
    parse_args,
)


_FAKE_SAMPLE = {
    "id": "fake_0",
    "context": "Paris is the capital of France.",
    "question": "What is the capital of France?",
    "choices_text": "A. Paris\nB. London\nC. Madrid",
    "valid_letters": ["A", "B", "C"],
    "counterfactual_key": "B",
}


def _stage_block(script: str, stage_name: str) -> str:
    start = script.index(f"run_stage {stage_name} ")
    end = script.find("\nrun_stage ", start + 1)
    return script[start:] if end == -1 else script[start:end]


def _set_argv(monkeypatch: pytest.MonkeyPatch, *extra: str) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_negative_control.py",
            "--benchmark",
            "faitheval",
            "--model_key",
            "mistral24b",
            *extra,
        ],
    )


def test_parse_args_prompt_style_default_is_anti_compliance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_argv(monkeypatch)
    args = parse_args()
    assert args.prompt_style == "anti_compliance"


def test_parse_args_prompt_style_accepts_standard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_argv(monkeypatch, "--prompt_style", "standard")
    args = parse_args()
    assert args.prompt_style == "standard"


def test_parse_args_prompt_style_rejects_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_argv(monkeypatch, "--prompt_style", "made_up")
    with pytest.raises(SystemExit):
        parse_args()


def test_faitheval_prompt_standard_differs_from_anti_compliance() -> None:
    standard = _faitheval_prompt(_FAKE_SAMPLE, "standard")
    anti = _faitheval_prompt(_FAKE_SAMPLE, "anti_compliance")
    assert standard != anti
    assert "expert in retrieval question answering" in standard
    assert "answer based on your own knowledge" in anti


def test_mistral_wrapper_controls_reuse_baseline_prompt_and_sample_selection() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    script = (repo_root / "scripts/infra/mistral24b_replication.sh").read_text()

    faitheval_baseline = _stage_block(script, "faitheval")
    faitheval_control = _stage_block(script, "faitheval_controls")
    falseqa_baseline = _stage_block(script, "falseqa")
    falseqa_control = _stage_block(script, "falseqa_controls")

    shared_sample_cap = '--max_samples "${INTERVENTION_MAX_SAMPLES}"'
    assert "--prompt_style standard" in faitheval_baseline
    assert "--prompt_style standard" in faitheval_control
    assert shared_sample_cap in faitheval_baseline
    assert shared_sample_cap in faitheval_control
    assert shared_sample_cap in falseqa_baseline
    assert shared_sample_cap in falseqa_control


def test_negative_control_contract_rejects_legacy_outputs_without_guard(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    classifier_path = tmp_path / "classifier.pkl"
    classifier_path.write_bytes(b"classifier-v1")
    output_base = tmp_path / "control"
    seed_dir = output_base / "seed_0_unconstrained"
    seed_dir.mkdir(parents=True)
    (seed_dir / "alpha_0.0.jsonl").write_text('{"id": "old"}\n')
    _set_argv(
        monkeypatch,
        "--classifier_path",
        str(classifier_path),
        "--output_base",
        str(output_base),
        "--prompt_style",
        "standard",
        "--max_samples",
        "100",
    )
    args = parse_args()
    alphas, configs = negative_control_schedule(args.benchmark, args.quick)
    contract = build_negative_control_run_contract(args, alphas, configs)

    with pytest.raises(RuntimeError, match="without control_run_config.json"):
        assert_or_write_negative_control_run_contract(
            str(output_base),
            contract,
            alphas,
            configs,
            write_if_missing=True,
        )


def test_negative_control_contract_rejects_prompt_style_drift(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    classifier_path = tmp_path / "classifier.pkl"
    classifier_path.write_bytes(b"classifier-v1")
    output_base = tmp_path / "control"

    _set_argv(
        monkeypatch,
        "--classifier_path",
        str(classifier_path),
        "--output_base",
        str(output_base),
        "--prompt_style",
        "standard",
        "--max_samples",
        "100",
    )
    args = parse_args()
    alphas, configs = negative_control_schedule(args.benchmark, args.quick)
    contract = build_negative_control_run_contract(args, alphas, configs)
    output_base.mkdir(parents=True)
    assert_or_write_negative_control_run_contract(
        str(output_base),
        contract,
        alphas,
        configs,
        write_if_missing=True,
    )

    _set_argv(
        monkeypatch,
        "--classifier_path",
        str(classifier_path),
        "--output_base",
        str(output_base),
        "--prompt_style",
        "anti_compliance",
        "--max_samples",
        "100",
    )
    drifted_args = parse_args()
    drifted_contract = build_negative_control_run_contract(
        drifted_args, alphas, configs
    )

    with pytest.raises(RuntimeError, match="benchmark_config.prompt_style"):
        assert_or_write_negative_control_run_contract(
            str(output_base),
            drifted_contract,
            alphas,
            configs,
            write_if_missing=True,
        )


def test_parse_args_max_samples_accepts_int(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_argv(monkeypatch, "--max_samples", "100")
    args = parse_args()
    assert args.max_samples == 100


def test_parse_args_max_samples_default_is_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_argv(monkeypatch)
    args = parse_args()
    assert args.max_samples is None


def test_parse_args_sample_manifest_accepts_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(["a", "b", "c"]))
    _set_argv(monkeypatch, "--sample_manifest", str(manifest))
    args = parse_args()
    assert args.sample_manifest == str(manifest)


def test_jailbreak_csv2_eval_dir_uses_baseline_override() -> None:
    result = _jailbreak_csv2_eval_dir(
        user_baseline="data/custom/intervention/jailbreak/experiment",
        model_key="mistral24b",
        model_path=None,
    )
    assert result == "data/custom/intervention/jailbreak/csv2_evaluation"


def test_jailbreak_csv2_eval_dir_strips_trailing_slash() -> None:
    result = _jailbreak_csv2_eval_dir(
        user_baseline="data/custom/intervention/jailbreak/experiment/",
        model_key=None,
        model_path=None,
    )
    assert result == "data/custom/intervention/jailbreak/csv2_evaluation"


def test_jailbreak_csv2_eval_dir_handles_results_json_file() -> None:
    result = _jailbreak_csv2_eval_dir(
        user_baseline="data/custom/intervention/jailbreak/experiment/results.json",
        model_key=None,
        model_path=None,
    )
    assert result == "data/custom/intervention/jailbreak/csv2_evaluation"


def test_jailbreak_csv2_eval_dir_falls_back_to_registry() -> None:
    result = _jailbreak_csv2_eval_dir(
        user_baseline=None,
        model_key="mistral24b",
        model_path=None,
    )
    assert result == "data/mistral24b/intervention/jailbreak/csv2_evaluation"


def test_jailbreak_csv2_eval_dir_unregistered_returns_placeholder() -> None:
    result = _jailbreak_csv2_eval_dir(
        user_baseline=None,
        model_key=None,
        model_path="/workspace/models/some-unregistered-model",
    )
    assert result.startswith("<")
    assert "h_neuron_baseline" in result
