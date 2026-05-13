from __future__ import annotations

import json
import hashlib
from pathlib import Path
import re
import sys
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from run_intervention import (  # noqa: E402
    INTERVENTION_RUN_CONFIG_FILENAME,
    _faitheval_prompt,
    assert_or_write_intervention_run_contract,
    build_generation_fingerprint,
    build_intervention_run_contract,
    describe_sample_manifest,
    parse_args as parse_intervention_args,
)
from run_negative_control import (  # noqa: E402
    CANONICAL_JAILBREAK_GENERATION,
    _jailbreak_csv2_eval_dir,
    _run_faitheval_alphas,
    assert_h_neuron_baseline_matches_control_contract,
    assert_or_write_negative_control_run_contract,
    build_comparison_summary,
    build_negative_control_run_contract,
    load_h_neuron_baseline_binding,
    negative_control_triage,
    negative_control_schedule,
    parse_args,
)
from utils import fingerprint_ids  # noqa: E402


_FAKE_SAMPLE = {
    "id": "fake_0",
    "context": "Paris is the capital of France.",
    "question": "What is the capital of France?",
    "choices_text": "A. Paris\nB. London\nC. Madrid",
    "valid_letters": ["A", "B", "C"],
    "counterfactual_key": "B",
}


def _stage_block(script: str, stage_name: str) -> str:
    match = re.search(rf"(?m)^run_stage {re.escape(stage_name)}(?:\s|$)", script)
    if match is None:
        raise AssertionError(f"stage not found: {stage_name}")
    start = match.start()
    next_match = re.search(r"(?m)^run_stage ", script[start + 1 :])
    end = -1 if next_match is None else start + 1 + next_match.start()
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


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_h_baseline_artifacts(
    tmp_path: Path,
    *,
    classifier_path: Path,
    manifest_path: Path,
    benchmark: str = "faitheval",
    benchmark_config: dict[str, Any] | None = None,
    prompt_style: str | None = "standard",
    alphas: list[float] | None = None,
) -> Path:
    alphas = [0.0, 1.0, 3.0] if alphas is None else alphas
    baseline_dir = tmp_path / "experiment"
    baseline_dir.mkdir()
    sample_manifest = describe_sample_manifest(str(manifest_path))
    if benchmark_config is None:
        benchmark_config = {"prompt_style": prompt_style}
    run_contract = {
        "schema_version": "intervention_run_config/v1",
        "benchmark": benchmark,
        "model": {
            "key": "mistral24b",
            "path": "mistralai/Mistral-Small-24B-Instruct-2501",
        },
        "classifier": {
            "path": str(classifier_path),
            "content_sha256": _sha256(classifier_path),
        },
        "sample_selection": {
            "max_samples": None,
            "sample_manifest": sample_manifest,
        },
        "schedule": {"alphas": alphas},
        "benchmark_config": dict(benchmark_config),
        "intervention": {"mode": "neuron"},
    }
    (baseline_dir / INTERVENTION_RUN_CONFIG_FILENAME).write_text(
        json.dumps(run_contract, indent=2)
    )
    results = {
        "benchmark": benchmark,
        "model_key": "mistral24b",
        "model": "mistralai/Mistral-Small-24B-Instruct-2501",
        "classifier": str(classifier_path),
        "intervention_mode": "neuron",
        "sample_manifest": sample_manifest,
        "prompt_style": prompt_style,
        "results": {
            str(alpha): {
                "compliance_rate": 0.1,
                "n_compliant": 2,
                "n_total": 20,
                "parse_failures": 0,
            }
            for alpha in alphas
        },
    }
    (baseline_dir / "results.20260429_000000.json").write_text(
        json.dumps(results, indent=2)
    )
    return baseline_dir


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


def test_faitheval_negative_control_adapter_uses_shared_sweep(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class DummyScaler:
        alpha = -1.0

    responses = iter(["B", "unparseable"])
    monkeypatch.setattr(
        "run_negative_control.generate_response",
        lambda *_args, **_kwargs: (next(responses), {}),
    )

    results = _run_faitheval_alphas(
        model=object(),
        tokenizer=object(),
        samples=[
            {
                "id": "faith_1",
                "context": "Context",
                "question": "Question?",
                "choices_text": "A) One\nB) Two",
                "valid_letters": ["A", "B"],
                "counterfactual_key": "B",
            },
            {
                "id": "faith_2",
                "context": "Context",
                "question": "Question?",
                "choices_text": "A) One\nB) Two",
                "valid_letters": ["A", "B"],
                "counterfactual_key": "B",
            },
        ],
        scaler=DummyScaler(),
        alphas=[0.0],
        output_dir=str(tmp_path),
        config_name="seed_0_unconstrained",
        prompt_style="standard",
    )

    assert results["0.0"]["n_total"] == 2
    assert results["0.0"]["n_compliant"] == 1
    assert results["0.0"]["parse_failures"] == 1
    assert results["0.0"]["parse_failure"]["count"] == 1
    records = [
        json.loads(line)
        for line in (tmp_path / "alpha_0.0.jsonl").read_text().splitlines()
    ]
    assert [record["alpha"] for record in records] == [0.0, 0.0]
    assert [record["compliance"] for record in records] == [True, False]


def test_mistral_wrapper_controls_reuse_baseline_prompt_and_sample_selection() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    script = (repo_root / "scripts/infra/mistral24b_replication.sh").read_text()

    faitheval_pilot_baseline = _stage_block(script, "faitheval_pilot")
    faitheval_pilot_control = _stage_block(script, "faitheval_pilot_controls")
    faitheval_baseline = _stage_block(script, "faitheval")
    faitheval_control = _stage_block(script, "faitheval_controls")
    falseqa_baseline = _stage_block(script, "falseqa")
    falseqa_control = _stage_block(script, "falseqa_controls")

    pilot_manifest = '--sample_manifest "${FAITHEVAL_PILOT_MANIFEST}"'
    shared_faitheval_manifest = '--sample_manifest "${FAITHEVAL_SAMPLE_MANIFEST}"'
    falseqa_sample_cap = '--max_samples "${INTERVENTION_MAX_SAMPLES}"'
    assert "--prompt_style standard" in faitheval_pilot_baseline
    assert "--prompt_style standard" in faitheval_pilot_control
    assert pilot_manifest in faitheval_pilot_baseline
    assert pilot_manifest in faitheval_pilot_control
    assert '--alphas "${FAITHEVAL_PILOT_ALPHAS[@]}"' in faitheval_pilot_baseline
    assert "--quick" in faitheval_pilot_control
    assert (
        '--output_dir "${FAITHEVAL_PILOT_OUTPUT_ROOT}/experiment"'
        in faitheval_pilot_baseline
    )
    assert (
        '--h_neuron_baseline "${FAITHEVAL_PILOT_OUTPUT_ROOT}/experiment"'
        in faitheval_pilot_control
    )
    assert "--prompt_style standard" in faitheval_baseline
    assert "--prompt_style standard" in faitheval_control
    assert shared_faitheval_manifest in faitheval_baseline
    assert shared_faitheval_manifest in faitheval_control
    assert falseqa_sample_cap in falseqa_baseline
    assert falseqa_sample_cap in falseqa_control


def test_mistral_wrapper_blocks_cp5_until_cp4_review() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    script = (repo_root / "scripts/infra/mistral24b_replication.sh").read_text()

    assert 'CP4_REVIEWED="${CP4_REVIEWED:-0}"' in script
    assert "require_cp4_review_for_cp5" in script
    assert script.index(
        "if should_run_any_stage faitheval faitheval_controls; then"
    ) < script.index("run_stage faitheval ")


def test_mistral_wrapper_controls_require_complete_pilot_baseline() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    script = (repo_root / "scripts/infra/mistral24b_replication.sh").read_text()

    control_guard = (
        "if should_run_stage faitheval_pilot_controls && "
        "! should_run_stage faitheval_pilot; then\n"
        "    require_stage_complete \\\n"
        '        "${FAITHEVAL_PILOT_OUTPUT_ROOT}/experiment"'
    )
    assert control_guard in script
    assert script.index(control_guard) < script.index(
        "run_stage faitheval_pilot_controls "
    )


def test_intervention_contract_rejects_legacy_outputs_without_guard(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "experiment"
    output_dir.mkdir()
    (output_dir / "alpha_0.0.jsonl").write_text('{"id": "old"}\n')
    contract = {
        "schema_version": "intervention_run_config/v1",
        "benchmark": "faitheval",
        "schedule": {"alphas": [0.0]},
    }

    with pytest.raises(RuntimeError, match="without intervention_run_config.json"):
        assert_or_write_intervention_run_contract(
            str(output_dir),
            contract,
            [0.0],
            write_if_missing=True,
        )


def _write_bridge_lock(path: Path, ids: list[str]) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": "sample_manifest/v2",
                "benchmark": "triviaqa_bridge",
                "split_name": "unit",
                "ids": ids,
                "n_ids": len(ids),
                "fingerprint": fingerprint_ids(ids),
                "source_path": "unit-source.json",
                "source_sha256": hashlib.sha256(
                    json.dumps(ids).encode("utf-8")
                ).hexdigest(),
                "lock_context": "unit",
            }
        ),
        encoding="utf-8",
    )


def _build_bridge_contract(manifest_path: Path) -> dict[str, Any]:
    args = parse_intervention_args(
        [
            "--model_key",
            "mistral24b",
            "--benchmark",
            "triviaqa_bridge",
            "--triviaqa_bridge_manifest",
            str(manifest_path),
            "--triviaqa_bridge_parquet",
            "data/TriviaQA/rc.nocontext/validation-00000-of-00001.parquet",
            "--intervention_mode",
            "iti_head",
            "--iti_head_path",
            "models/unit_iti_heads.pt",
            "--iti_family",
            "truthfulqa_paperfaithful",
            "--iti_decode_scope",
            "first_3_tokens",
            "--alphas",
            "0.0",
            "8.0",
            "--run_profile",
            "canonical",
        ]
    )
    return build_intervention_run_contract(
        args,
        sample_manifest=None,
        run_profile="canonical",
        comparability_class="claimable",
        generation_fingerprint=build_generation_fingerprint(
            args=args,
            jailbreak_generation=None,
        ),
        jailbreak_generation=None,
    )


def test_triviaqa_bridge_contract_binds_same_path_manifest_content(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "triviaqa_bridge.lock.json"
    output_dir = tmp_path / "experiment"
    output_dir.mkdir()

    _write_bridge_lock(manifest_path, ["q1"])
    first_contract = _build_bridge_contract(manifest_path)
    _write_bridge_lock(manifest_path, ["q2"])
    second_contract = _build_bridge_contract(manifest_path)

    first_identity = first_contract["benchmark_config"][
        "triviaqa_bridge_manifest_identity"
    ]
    second_identity = second_contract["benchmark_config"][
        "triviaqa_bridge_manifest_identity"
    ]
    assert first_contract["benchmark_config"]["triviaqa_bridge_manifest"] == str(
        manifest_path
    )
    assert second_contract["benchmark_config"]["triviaqa_bridge_manifest"] == str(
        manifest_path
    )
    assert first_identity["path"] == second_identity["path"] == str(manifest_path)
    assert first_identity["content_sha256"] != second_identity["content_sha256"]
    assert first_identity["fingerprint"] != second_identity["fingerprint"]
    assert first_identity["lock_context"] == "unit"

    assert_or_write_intervention_run_contract(
        str(output_dir),
        first_contract,
        [0.0, 8.0],
        write_if_missing=True,
    )
    with pytest.raises(RuntimeError, match="triviaqa_bridge_manifest_identity"):
        assert_or_write_intervention_run_contract(
            str(output_dir),
            second_contract,
            [0.0, 8.0],
            write_if_missing=True,
        )


def test_negative_control_contract_binds_h_baseline_identity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    classifier_path = tmp_path / "classifier.pkl"
    classifier_path.write_bytes(b"classifier-v1")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(["fake_0", "fake_1"]))
    baseline_dir = _write_h_baseline_artifacts(
        tmp_path,
        classifier_path=classifier_path,
        manifest_path=manifest_path,
        prompt_style="standard",
    )
    _set_argv(
        monkeypatch,
        "--quick",
        "--classifier_path",
        str(classifier_path),
        "--sample_manifest",
        str(manifest_path),
        "--prompt_style",
        "standard",
        "--h_neuron_baseline",
        str(baseline_dir),
    )
    args = parse_args()
    alphas, configs = negative_control_schedule(args.benchmark, args.quick)
    baseline_binding = load_h_neuron_baseline_binding(str(baseline_dir), alphas)
    contract = build_negative_control_run_contract(
        args,
        alphas,
        configs,
        baseline_binding=baseline_binding,
    )

    assert_h_neuron_baseline_matches_control_contract(contract)
    assert contract["h_neuron_baseline"]["results_path"].endswith(
        "results.20260429_000000.json"
    )


def test_jailbreak_negative_control_baseline_allows_control_only_batch_size(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    classifier_path = tmp_path / "classifier.pkl"
    classifier_path.write_bytes(b"classifier-v1")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(["fake_0", "fake_1"]))
    baseline_dir = _write_h_baseline_artifacts(
        tmp_path,
        classifier_path=classifier_path,
        manifest_path=manifest_path,
        benchmark="jailbreak",
        benchmark_config={
            "jailbreak_source": "jailbreakbench",
            "jailbreak_path": None,
            "n_templates": 5,
            "jailbreak_generation": dict(CANONICAL_JAILBREAK_GENERATION),
        },
        prompt_style=None,
        alphas=[0.0, 1.0, 1.5, 3.0],
    )
    _set_argv(
        monkeypatch,
        "--benchmark",
        "jailbreak",
        "--classifier_path",
        str(classifier_path),
        "--sample_manifest",
        str(manifest_path),
        "--jailbreak_batch_size",
        "4",
        "--h_neuron_baseline",
        str(baseline_dir),
    )
    args = parse_args()
    alphas, configs = negative_control_schedule(args.benchmark, args.quick)
    baseline_binding = load_h_neuron_baseline_binding(str(baseline_dir), alphas)
    contract = build_negative_control_run_contract(
        args,
        alphas,
        configs,
        baseline_binding=baseline_binding,
    )

    assert contract["benchmark_config"]["jailbreak_batch_size"] == 4
    assert_h_neuron_baseline_matches_control_contract(contract)


def test_negative_control_contract_rejects_baseline_prompt_drift(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    classifier_path = tmp_path / "classifier.pkl"
    classifier_path.write_bytes(b"classifier-v1")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(["fake_0", "fake_1"]))
    baseline_dir = _write_h_baseline_artifacts(
        tmp_path,
        classifier_path=classifier_path,
        manifest_path=manifest_path,
        prompt_style="anti_compliance",
    )
    _set_argv(
        monkeypatch,
        "--quick",
        "--classifier_path",
        str(classifier_path),
        "--sample_manifest",
        str(manifest_path),
        "--prompt_style",
        "standard",
        "--h_neuron_baseline",
        str(baseline_dir),
    )
    args = parse_args()
    alphas, configs = negative_control_schedule(args.benchmark, args.quick)
    baseline_binding = load_h_neuron_baseline_binding(str(baseline_dir), alphas)
    contract = build_negative_control_run_contract(
        args,
        alphas,
        configs,
        baseline_binding=baseline_binding,
    )

    with pytest.raises(RuntimeError, match="benchmark_config.prompt_style"):
        assert_h_neuron_baseline_matches_control_contract(contract)


def test_comparison_summary_preserves_h_baseline_parse_failures(
    tmp_path: Path,
) -> None:
    alphas = [0.0, 1.0, 3.0]
    baseline = {
        "n_h_neurons": 10,
        "results": {
            "0.0": {
                "compliance_rate": 0.1,
                "n_compliant": 2,
                "n_total": 20,
                "parse_failures": 1,
            },
            "1.0": {
                "compliance_rate": 0.2,
                "n_compliant": 4,
                "n_total": 20,
                "parse_failures": 3,
            },
            "3.0": {
                "compliance_rate": 0.3,
                "n_compliant": 6,
                "n_total": 20,
                "parse_failures": 5,
            },
        },
    }
    baseline_path = tmp_path / "baseline_results.json"
    baseline_path.write_text(json.dumps(baseline))
    control_results = {}
    for n_compliant, alpha in enumerate(alphas, start=1):
        control_results[str(alpha)] = {
            "compliance_rate": n_compliant / 20,
            "n_compliant": n_compliant,
            "n_total": 20,
            "parse_failures": 2,
        }

    summary = build_comparison_summary(
        {"seed_0_unconstrained": control_results},
        alphas,
        baseline_path=str(baseline_path),
    )

    assert summary["h_neuron_baseline"]["parse_failures"] == [1, 3, 5]
    assert summary["unconstrained_random"]["per_seed"]["seed_0_unconstrained"][
        "parse_failures"
    ] == [2, 2, 2]


def _negative_control_triage_summary(
    *,
    alpha_0_h_rate_pct: float,
    alpha_3_h_rate_pct: float,
    alpha_0_interval: tuple[float, float] = (40.0, 60.0),
    alpha_3_interval: tuple[float, float] = (45.0, 65.0),
    slope_h_pp_per_alpha: float = 3.0,
    slope_interval: tuple[float, float] = (-1.0, 1.0),
) -> dict[str, Any]:
    return {
        "comparison_to_h_neurons": {
            "alpha_0_h_rate_pct": alpha_0_h_rate_pct,
            "alpha_0_random_percentile_interval_pct": {
                "lower": alpha_0_interval[0],
                "upper": alpha_0_interval[1],
            },
            "alpha_3_h_rate_pct": alpha_3_h_rate_pct,
            "alpha_3_random_percentile_interval_pct": {
                "lower": alpha_3_interval[0],
                "upper": alpha_3_interval[1],
            },
            "slope_h_pp_per_alpha": slope_h_pp_per_alpha,
            "slope_random_percentile_interval": {
                "lower": slope_interval[0],
                "upper": slope_interval[1],
            },
        }
    }


def test_negative_control_triage_flags_baseline_mismatch(
    tmp_path: Path,
) -> None:
    alphas = [0.0, 1.0, 3.0]
    baseline = {
        "n_h_neurons": 10,
        "results": {
            "0.0": {
                "compliance_rate": 0.75,
                "n_compliant": 15,
                "n_total": 20,
                "parse_failures": 0,
            },
            "1.0": {
                "compliance_rate": 0.65,
                "n_compliant": 13,
                "n_total": 20,
                "parse_failures": 0,
            },
            "3.0": {
                "compliance_rate": 0.65,
                "n_compliant": 13,
                "n_total": 20,
                "parse_failures": 0,
            },
        },
    }
    baseline_path = tmp_path / "baseline_results.json"
    baseline_path.write_text(json.dumps(baseline))
    control_results_a = {
        "0.0": {
            "compliance_rate": 0.65,
            "n_compliant": 65,
            "n_total": 100,
            "parse_failures": 0,
        },
        "1.0": {
            "compliance_rate": 0.66,
            "n_compliant": 66,
            "n_total": 100,
            "parse_failures": 0,
        },
        "3.0": {
            "compliance_rate": 0.65,
            "n_compliant": 65,
            "n_total": 100,
            "parse_failures": 0,
        },
    }
    control_results_b = {
        "0.0": {
            "compliance_rate": 0.65,
            "n_compliant": 65,
            "n_total": 100,
            "parse_failures": 0,
        },
        "1.0": {
            "compliance_rate": 0.64,
            "n_compliant": 64,
            "n_total": 100,
            "parse_failures": 0,
        },
        "3.0": {
            "compliance_rate": 0.65,
            "n_compliant": 65,
            "n_total": 100,
            "parse_failures": 0,
        },
    }

    summary = build_comparison_summary(
        {
            "seed_0_unconstrained": control_results_a,
            "seed_1_unconstrained": control_results_b,
        },
        alphas,
        baseline_path=str(baseline_path),
    )
    status, note, action = negative_control_triage(summary)

    comparison = summary["comparison_to_h_neurons"]
    assert comparison["alpha_0_h_minus_random_mean_pp"] == 10.0
    assert comparison["alpha_3_h_rate_pct"] == comparison["alpha_3_random_mean_pct"]
    assert status == "review_baseline_mismatch"
    assert "alpha=0 baseline differs" in note
    assert "no-op/rerun stability" in action


def test_negative_control_triage_flags_constant_offset() -> None:
    summary = _negative_control_triage_summary(
        alpha_0_h_rate_pct=62.0,
        alpha_3_h_rate_pct=66.5,
    )

    status, note, action = negative_control_triage(summary)

    assert status == "review_constant_offset"
    assert "same-direction, similar-magnitude offset" in note
    assert "baseline/control rerun offsets" in action


@pytest.mark.parametrize(
    ("alpha_0_h_rate_pct", "alpha_3_h_rate_pct"),
    [
        (38.0, 66.5),
        (62.0, 69.5),
        (50.0, 66.5),
    ],
)
def test_negative_control_triage_requires_constant_offset_predicate(
    alpha_0_h_rate_pct: float,
    alpha_3_h_rate_pct: float,
) -> None:
    summary = _negative_control_triage_summary(
        alpha_0_h_rate_pct=alpha_0_h_rate_pct,
        alpha_3_h_rate_pct=alpha_3_h_rate_pct,
    )

    status, _, _ = negative_control_triage(summary)

    assert status == "specificity_supported"


def test_mistral_wrapper_has_token_span_preflight_before_claim_stages() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    script = (repo_root / "scripts/infra/mistral24b_replication.sh").read_text()

    preflight = _stage_block(script, "preflight")
    model_smoke = _stage_block(script, "model_smoke")
    splits = _stage_block(script, "splits")

    assert "scripts/audit_token_spans.py" in preflight
    assert '--input_path "${ANSWER_TOKENS}"' in preflight
    assert '--output_path "${TOKEN_SPAN_AUDIT}"' in preflight
    assert '--rendered_chat_path "${RENDERED_CHAT_EXAMPLES}"' in preflight
    assert "scripts/model_load_smoke.py" in model_smoke
    assert '--device_map "${DEVICE_MAP}"' in model_smoke
    assert '--output_path "${MODEL_LOAD_SMOKE}"' in model_smoke
    assert "--max_new_tokens 1" in model_smoke
    assert script.index("run_stage preflight ") < script.index("run_stage model_smoke ")
    assert script.index("run_stage model_smoke ") < script.index("run_stage splits ")
    assert "scripts/audit_token_spans.py" not in splits
    assert "scripts/model_load_smoke.py" not in preflight


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


def test_negative_control_jailbreak_contract_binds_decode_semantics(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    classifier_path = tmp_path / "classifier.pkl"
    classifier_path.write_bytes(b"classifier-v1")
    output_base = tmp_path / "control"

    _set_argv(
        monkeypatch,
        "--benchmark",
        "jailbreak",
        "--classifier_path",
        str(classifier_path),
        "--output_base",
        str(output_base),
        "--jailbreak_batch_size",
        "1",
    )
    args = parse_args()
    alphas, configs = negative_control_schedule(args.benchmark, args.quick)
    contract = build_negative_control_run_contract(args, alphas, configs)

    assert contract["benchmark_config"]["jailbreak_batch_size"] == 1
    assert contract["benchmark_config"]["jailbreak_generation"] == dict(
        CANONICAL_JAILBREAK_GENERATION
    )


def test_negative_control_jailbreak_contract_rejects_decode_drift(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    classifier_path = tmp_path / "classifier.pkl"
    classifier_path.write_bytes(b"classifier-v1")
    output_base = tmp_path / "control"

    _set_argv(
        monkeypatch,
        "--benchmark",
        "jailbreak",
        "--classifier_path",
        str(classifier_path),
        "--output_base",
        str(output_base),
        "--jailbreak_batch_size",
        "1",
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
    seed_dir = output_base / "seed_0_unconstrained"
    seed_dir.mkdir()
    (seed_dir / "alpha_0.0.jsonl").write_text('{"id": "old"}\n')

    monkeypatch.setitem(CANONICAL_JAILBREAK_GENERATION, "top_p", 0.9)
    drifted_contract = build_negative_control_run_contract(args, alphas, configs)

    with pytest.raises(
        RuntimeError,
        match=r"benchmark_config\.jailbreak_generation\.top_p",
    ):
        assert_or_write_negative_control_run_contract(
            str(output_base),
            drifted_contract,
            alphas,
            configs,
            write_if_missing=True,
        )


def test_negative_control_jailbreak_contract_rejects_legacy_missing_decode_binding(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    classifier_path = tmp_path / "classifier.pkl"
    classifier_path.write_bytes(b"classifier-v1")
    output_base = tmp_path / "control"

    _set_argv(
        monkeypatch,
        "--benchmark",
        "jailbreak",
        "--classifier_path",
        str(classifier_path),
        "--output_base",
        str(output_base),
        "--jailbreak_batch_size",
        "1",
    )
    args = parse_args()
    alphas, configs = negative_control_schedule(args.benchmark, args.quick)
    contract = build_negative_control_run_contract(args, alphas, configs)
    legacy_contract = json.loads(json.dumps(contract))
    del legacy_contract["benchmark_config"]["jailbreak_generation"]

    output_base.mkdir(parents=True)
    (output_base / "control_run_config.json").write_text(
        json.dumps(legacy_contract),
        encoding="utf-8",
    )
    seed_dir = output_base / "seed_0_unconstrained"
    seed_dir.mkdir()
    (seed_dir / "alpha_0.0.jsonl").write_text('{"id": "old"}\n')

    with pytest.raises(
        RuntimeError,
        match=r"benchmark_config\.jailbreak_generation",
    ):
        assert_or_write_negative_control_run_contract(
            str(output_base),
            contract,
            alphas,
            configs,
            write_if_missing=True,
        )


def test_negative_control_non_jailbreak_contract_shape_remains_compatible(
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

    assert "jailbreak_generation" not in contract["benchmark_config"]
    assert_or_write_negative_control_run_contract(
        str(output_base),
        contract,
        alphas,
        configs,
        write_if_missing=True,
    )
    assert_or_write_negative_control_run_contract(
        str(output_base),
        contract,
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
