"""Tests for the FaithEval SAE utility-selector pipeline."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from intervene_sae import load_sae_feature_manifest  # noqa: E402
from report_faitheval_sae_utility_selector import (  # noqa: E402
    FAMILY_ORDER,
    build_audit_note,
    build_heldout_summary,
)
from run_intervention import (  # noqa: E402
    _choice_token_ids,
    _build_faitheval_sample,
    load_sample_manifest_ids,
    misleading_margin,
    resolve_sae_target_features,
)
from select_faitheval_sae_utility_features import (  # noqa: E402
    build_stratified_faitheval_split,
    layer_histogram,
    match_random_zero_weight_features,
)


def _extraction_metadata() -> dict[str, object]:
    return {
        "hook_point": "post_feedforward_layernorm",
        "sae_release": "gemma-scope-2-4b-it-mlp-all",
        "sae_width": "16k",
        "sae_l0": "small",
        "layer_indices": [5, 9],
        "d_in": 8,
        "d_sae": 4,
        "aggregation_method": "mean",
    }


def _make_split_samples(visible_labels: list[str]) -> list[dict[str, object]]:
    samples: list[dict[str, object]] = []
    canonical_labels = list("ABCDE")
    for idx in range(24):
        num_options = 4 if idx < 18 else 5
        canonical_key = canonical_labels[idx % num_options]
        display_key = visible_labels[canonical_labels.index(canonical_key)]
        samples.append(
            {
                "id": f"s{idx}",
                "num_options": num_options,
                "counterfactual_key": display_key,
                "counterfactual_key_canonical": canonical_key,
            }
        )
    return samples


def _family_rows(
    ids: list[str],
    compliant_ids: set[str] | None = None,
    *,
    metric_values: dict[str, float] | None = None,
    metric_name: str = "misleading_minus_preferred_logprob_margin",
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for sample_id in ids:
        row: dict[str, object] = {"id": sample_id}
        if compliant_ids is not None:
            row["compliance"] = sample_id in compliant_ids
        if metric_values is not None:
            row["metric_name"] = metric_name
            row["metric_value"] = float(metric_values[sample_id])
        rows.append(row)
    return rows


def _set_mtime(path: Path, timestamp: int) -> None:
    path.touch(exist_ok=True)
    os.utime(path, (timestamp, timestamp))


class SpaceSensitiveTokenizer:
    def __call__(self, text, add_special_tokens=False, return_tensors=None):
        del add_special_tokens
        token_id = 10 if text.startswith(" ") else 20
        ids = torch.tensor([[token_id]], dtype=torch.long)
        if return_tensors == "pt":
            return {"input_ids": ids}
        return {"input_ids": ids.squeeze(0).tolist()}


def test_manifest_loading_and_run_intervention_precedence(tmp_path: Path) -> None:
    classifier_path = tmp_path / "sae_detector.pkl"
    joblib.dump(
        SimpleNamespace(coef_=np.array([[0.9, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0]])),
        classifier_path,
    )

    manifest_path = tmp_path / "utility_selected_features.json"
    manifest_payload = {
        "schema_version": "sae_feature_manifest/v1",
        "extraction_metadata": _extraction_metadata(),
        "features": [
            {"layer": 5, "feature": 1, "flat_idx": 1},
            {"layer": 9, "feature": 3, "flat_idx": 7},
        ],
    }
    manifest_path.write_text(json.dumps(manifest_payload), encoding="utf-8")

    manifest = load_sae_feature_manifest(manifest_path)
    assert manifest["target_features"] == {"5": [1], "9": [3]}

    args = Namespace(
        sae_feature_manifest=str(manifest_path),
        sae_classifier_path=str(classifier_path),
        sae_classifier_summary=None,
        extraction_dir=None,
    )
    target_features, source = resolve_sae_target_features(args)

    assert target_features == {5: [1], 9: [3]}
    assert source["feature_count"] == 2


def test_sample_manifest_loader_accepts_metadata_wrapped_ids(tmp_path: Path) -> None:
    manifest_path = tmp_path / "test_manifest.json"
    manifest_path.write_text(
        json.dumps({"schema_version": "sample_manifest/v2", "ids": ["a", "b"]}),
        encoding="utf-8",
    )

    assert load_sample_manifest_ids(str(manifest_path)) == {"a", "b"}


def test_stratified_faitheval_split_is_deterministic_and_label_invariant() -> None:
    alpha_samples = _make_split_samples(["A", "B", "C", "D", "E"])
    numeric_samples = _make_split_samples(["1", "2", "3", "4", "5"])

    val_alpha, test_alpha, meta_alpha = build_stratified_faitheval_split(
        alpha_samples,
        validation_size=8,
        seed=42,
    )
    val_numeric, test_numeric, meta_numeric = build_stratified_faitheval_split(
        numeric_samples,
        validation_size=8,
        seed=42,
    )
    val_repeat, test_repeat, meta_repeat = build_stratified_faitheval_split(
        alpha_samples,
        validation_size=8,
        seed=42,
    )

    assert [row["id"] for row in val_alpha] == [row["id"] for row in val_repeat]
    assert [row["id"] for row in test_alpha] == [row["id"] for row in test_repeat]
    assert meta_alpha == meta_repeat

    assert [row["id"] for row in val_alpha] == [row["id"] for row in val_numeric]
    assert [row["id"] for row in test_alpha] == [row["id"] for row in test_numeric]
    assert meta_alpha == meta_numeric
    assert meta_alpha["stratify_by"] == ["num_options", "counterfactual_key_canonical"]
    assert sorted(meta_alpha["strata_validation_counts"]) == [
        "4::A",
        "4::B",
        "4::C",
        "4::D",
        "5::A",
        "5::B",
        "5::C",
        "5::D",
        "5::E",
    ]


def test_stratified_faitheval_split_requires_canonical_answer_key() -> None:
    samples = [
        {"id": "s0", "num_options": 4, "counterfactual_key": "A"},
        {"id": "s1", "num_options": 4, "counterfactual_key": "B"},
    ]

    with pytest.raises(ValueError, match="counterfactual_key_canonical"):
        build_stratified_faitheval_split(samples, validation_size=1, seed=42)


def test_margin_metric_uses_preferred_option_and_positive_is_good() -> None:
    sample = {"id": "s0", "counterfactual_key": "D", "preferred_key": "A"}
    baseline_scores = {"A": -0.2, "B": -0.9, "C": -1.1, "D": -0.1}
    ablated_scores = {"A": -0.1, "B": -0.8, "C": -1.2, "D": -0.6}

    baseline_margin = misleading_margin(sample, baseline_scores)
    ablated_margin = misleading_margin(sample, ablated_scores)
    utility = baseline_margin - ablated_margin

    assert baseline_margin == 0.1
    assert ablated_margin == -0.5
    assert utility == 0.6


def test_margin_metric_normalizes_numeric_display_labels() -> None:
    sample = {
        "id": "n0",
        "valid_letters": ["1", "2", "3", "4"],
        "counterfactual_key": "A",
        "preferred_key": "2",
    }
    scores = {"1": -0.3, "2": -0.8, "3": -1.1, "4": -1.5}

    assert misleading_margin(sample, scores) == 0.5


def test_choice_token_ids_score_first_answer_token_without_leading_space() -> None:
    tokenizer = SpaceSensitiveTokenizer()

    token_ids = _choice_token_ids(tokenizer, ["A", "B"])

    assert token_ids["A"].tolist() == [20]
    assert token_ids["B"].tolist() == [20]


def test_build_faitheval_sample_uses_arc_answer_for_preferred_key() -> None:
    sample = _build_faitheval_sample(
        {
            "id": "NYSEDREGENTS_2008_8_34",
            "context": "Counterfactual context.",
            "question": (
                "Which statement best describes the energy changes that occur "
                "while a child is riding on a sled down a steep, snow-covered hill?"
            ),
            "answer": "Kinetic energy decreases and potential energy increases.",
            "answerKey": "A",
            "choices": {
                "label": ["1", "2", "3", "4"],
                "text": [
                    "Kinetic energy decreases and potential energy increases.",
                    "Kinetic energy increases and potential energy decreases.",
                    "Both potential energy and kinetic energy decrease.",
                    "Both potential energy and kinetic energy increase.",
                ],
            },
            "num of options": 4,
        },
        preferred_answer_key="2",
    )

    assert sample["counterfactual_key"] == "1"
    assert sample["preferred_key"] == "2"
    assert sample["counterfactual_key_canonical"] == "A"
    assert sample["preferred_key_canonical"] == "2"
    assert (
        sample["preferred_answer_text"]
        == "Kinetic energy increases and potential energy decreases."
    )


def test_random_matching_is_layer_exact_without_replacement() -> None:
    selected = [
        {"layer": 5, "feature": 0, "flat_idx": 0},
        {"layer": 5, "feature": 1, "flat_idx": 1},
        {"layer": 9, "feature": 0, "flat_idx": 4},
    ]
    zero_pool = [
        {"layer": 5, "feature": 2, "flat_idx": 2},
        {"layer": 5, "feature": 3, "flat_idx": 3},
        {"layer": 5, "feature": 4, "flat_idx": 7},
        {"layer": 9, "feature": 1, "flat_idx": 5},
        {"layer": 9, "feature": 2, "flat_idx": 6},
        {"layer": 9, "feature": 3, "flat_idx": 8},
    ]
    feature_stats = {
        2: {"token_activation_rate": 0.50},
        3: {"token_activation_rate": 0.00},
        7: {"token_activation_rate": 0.20},
        5: {"token_activation_rate": 0.70},
        6: {"token_activation_rate": 0.40},
        8: {"token_activation_rate": 0.10},
    }

    matched = match_random_zero_weight_features(
        selected,
        zero_pool,
        feature_stats,
        seed=0,
    )

    assert len(matched) == len(selected)
    assert layer_histogram(matched) == {"5": 2, "9": 1}
    assert {row["flat_idx"] for row in matched}.isdisjoint(
        row["flat_idx"] for row in selected
    )
    assert len({row["flat_idx"] for row in matched}) == len(matched)
    assert all(float(row["token_activation_rate"]) > 0.0 for row in matched)


def test_random_matching_is_deterministic_for_fixed_seed() -> None:
    selected = [
        {"layer": 5, "feature": 0, "flat_idx": 0},
        {"layer": 9, "feature": 0, "flat_idx": 4},
    ]
    zero_pool = [
        {"layer": 5, "feature": 2, "flat_idx": 2},
        {"layer": 5, "feature": 3, "flat_idx": 3},
        {"layer": 9, "feature": 1, "flat_idx": 5},
        {"layer": 9, "feature": 2, "flat_idx": 6},
    ]
    feature_stats = {
        2: {"token_activation_rate": 0.6},
        3: {"token_activation_rate": 0.4},
        5: {"token_activation_rate": 0.7},
        6: {"token_activation_rate": 0.3},
    }

    matched_a = match_random_zero_weight_features(
        selected,
        zero_pool,
        feature_stats,
        seed=11,
    )
    matched_b = match_random_zero_weight_features(
        selected,
        zero_pool,
        feature_stats,
        seed=11,
    )

    assert [row["flat_idx"] for row in matched_a] == [
        row["flat_idx"] for row in matched_b
    ]


def test_random_matching_uses_only_positive_token_activation_rate() -> None:
    selected = [{"layer": 5, "feature": 0, "flat_idx": 0}]
    zero_pool = [
        {"layer": 5, "feature": 1, "flat_idx": 1},
        {"layer": 5, "feature": 2, "flat_idx": 2},
    ]
    feature_stats = {
        1: {"token_activation_rate": 0.0},
        2: {"token_activation_rate": 0.9},
    }

    matched = match_random_zero_weight_features(
        selected,
        zero_pool,
        feature_stats,
        seed=0,
    )

    assert [row["flat_idx"] for row in matched] == [2]


def test_random_matching_changes_across_seeds_on_nondegenerate_fixture() -> None:
    selected = [{"layer": 9, "feature": 0, "flat_idx": 4}]
    zero_pool = [
        {"layer": 9, "feature": 1, "flat_idx": 5},
        {"layer": 9, "feature": 2, "flat_idx": 6},
        {"layer": 9, "feature": 3, "flat_idx": 8},
    ]
    feature_stats = {
        5: {"token_activation_rate": 0.6},
        6: {"token_activation_rate": 0.5},
        8: {"token_activation_rate": 0.4},
    }

    matched_seed_0 = match_random_zero_weight_features(
        selected,
        zero_pool,
        feature_stats,
        seed=0,
    )
    matched_seed_1 = match_random_zero_weight_features(
        selected,
        zero_pool,
        feature_stats,
        seed=1,
    )

    assert [row["flat_idx"] for row in matched_seed_0] != [
        row["flat_idx"] for row in matched_seed_1
    ]


def test_report_summary_covers_full_bundle_schema() -> None:
    ordered_ids = ["a", "b", "c", "d"]
    selector_summary = {
        "selector_design": {
            "layer_coverage_note": "Partial L3 closure only.",
            "target_families": {
                "readout_selected": "probe-positive",
                "utility_selected": "validation margin",
                "matched_random": "zero-weight matched",
            },
        },
        "candidate_pool": {
            "n_features": 5,
            "layer_histogram": {"5": 3, "9": 2},
            "weight_sign_counts": {"negative": 2, "positive": 3},
        },
        "families": {
            "utility_selected": {
                "k": 2,
                "layer_histogram": {"5": 1, "9": 1},
                "weight_sign_counts": {"negative": 1, "positive": 1},
                "outside_old_shortlist": {"count": 1, "fraction": 0.5},
            },
            "readout_selected": {
                "k": 2,
                "layer_histogram": {"5": 2},
                "weight_sign_counts": {"positive": 2},
            },
        },
        "family_overlap": {
            "utility_selected_vs_readout_selected": {
                "intersection_count": 1,
                "union_count": 3,
                "jaccard": 1 / 3,
            }
        },
    }
    heldout_rows_by_benchmark = {
        "faitheval": {
            "noop": _family_rows(ordered_ids, {"a", "c"}),
            "readout_selected": _family_rows(ordered_ids, {"a"}),
            "utility_selected": _family_rows(ordered_ids, {"c"}),
            "matched_random_seed_0": _family_rows(ordered_ids, {"a", "d"}),
            "matched_random_seed_1": _family_rows(ordered_ids, {"b"}),
            "matched_random_seed_2": _family_rows(ordered_ids, {"a", "b"}),
        },
        "faitheval_anti_compliance_margin": {
            "noop": _family_rows(
                ordered_ids,
                metric_values={"a": 0.5, "b": 0.5, "c": 0.5, "d": 0.5},
            ),
            "readout_selected": _family_rows(
                ordered_ids,
                metric_values={"a": 0.0, "b": 0.0, "c": 0.0, "d": 0.0},
            ),
            "utility_selected": _family_rows(
                ordered_ids,
                metric_values={"a": 1.0, "b": 0.5, "c": -0.5, "d": 0.0},
            ),
            "matched_random_seed_0": _family_rows(
                ordered_ids,
                metric_values={"a": 0.25, "b": 0.25, "c": 0.25, "d": 0.25},
            ),
            "matched_random_seed_1": _family_rows(
                ordered_ids,
                metric_values={"a": -0.5, "b": -0.25, "c": -0.25, "d": 0.0},
            ),
            "matched_random_seed_2": _family_rows(
                ordered_ids,
                metric_values={"a": 0.75, "b": 0.25, "c": 0.0, "d": -0.25},
            ),
        },
    }

    summary = build_heldout_summary(
        selector_summary=selector_summary,
        ordered_ids=ordered_ids,
        heldout_rows_by_benchmark=heldout_rows_by_benchmark,
    )

    assert summary["schema_version"] == "faitheval_sae_utility_selector_report/v4"
    assert summary["heldout_compliance"]["families"]["noop"]["n_compliant"] == 2
    assert (
        summary["heldout_compliance"]["families"]["utility_selected"]["n_compliant"]
        == 1
    )
    assert (
        summary["heldout_compliance"]["paired_deltas_pp"]["utility_minus_noop"][
            "estimate_pp"
        ]
        == -25.0
    )
    assert summary["heldout_anti_compliance_margin"]["families"]["utility_selected"][
        "estimate"
    ] == pytest.approx(0.25)
    assert set(summary["heldout_compliance"]["families"]) == set(FAMILY_ORDER)
    assert set(summary["heldout_anti_compliance_margin"]["families"]) == set(
        FAMILY_ORDER
    )
    expected_delta_keys = {
        "utility_minus_readout",
        "utility_minus_noop",
        "utility_minus_matched_random_seed_0",
        "utility_minus_matched_random_seed_1",
        "utility_minus_matched_random_seed_2",
    }
    assert set(summary["heldout_compliance"]["paired_deltas_pp"]) == expected_delta_keys
    assert (
        summary["heldout_anti_compliance_margin"]["metric_name"]
        == "misleading_minus_preferred_logprob_margin"
    )
    assert (
        summary["heldout_anti_compliance_margin"]["margin_definition"]
        == "logp(counterfactual_key) - logp(preferred_key)"
    )
    assert summary["heldout_anti_compliance_margin"]["delta_units"] == "logprob_margin"
    assert set(summary["heldout_anti_compliance_margin"]["paired_deltas"]) == (
        expected_delta_keys
    )
    assert summary["heldout_anti_compliance_margin"]["paired_deltas"][
        "utility_minus_readout"
    ]["estimate"] == pytest.approx(0.25)
    assert summary["heldout_anti_compliance_margin"]["paired_deltas"][
        "utility_minus_noop"
    ]["estimate"] == pytest.approx(-0.25)
    assert summary["selector_diagnostics"]["candidate_pool_n"] == 5
    assert summary["selector_diagnostics"]["utility_weight_sign_counts"] == {
        "negative": 1,
        "positive": 1,
    }
    assert summary["selector_diagnostics"]["target_families"]["matched_random"] == (
        "zero-weight matched"
    )
    assert summary["selector_diagnostics"]["layer_coverage_note"] == (
        "Partial L3 closure only."
    )
    assert summary["selector_diagnostics"]["outside_old_shortlist_fraction"] == 0.5
    assert summary["selector_diagnostics"]["calibration_to_heldout_gap"] is None

    audit_note = build_audit_note(summary)
    assert "FaithEval anti-compliance margin families" in audit_note
    assert "FaithEval anti-compliance margin paired deltas" in audit_note
    assert "Candidate pool scope" in audit_note
    assert "Layer-coverage status" in audit_note
    assert "FaithEval MC-logprob" not in audit_note
    margin_line = next(
        line
        for line in audit_note.splitlines()
        if line.startswith("- FaithEval anti-compliance margin paired deltas:")
    )
    assert " pp " not in margin_line


def test_report_summary_requires_exact_sample_id_parity() -> None:
    ordered_ids = ["a", "b"]
    selector_summary = {
        "families": {
            "utility_selected": {
                "k": 2,
                "layer_histogram": {"5": 2},
                "outside_old_shortlist": {"count": 0, "fraction": 0.0},
            },
            "readout_selected": {
                "k": 2,
                "layer_histogram": {"5": 2},
            },
        },
        "family_overlap": {
            "utility_selected_vs_readout_selected": {
                "intersection_count": 2,
                "union_count": 2,
                "jaccard": 1.0,
            }
        },
    }
    benchmark_rows = {
        family: _family_rows(ordered_ids, {"a"}) for family in FAMILY_ORDER
    }
    benchmark_rows["readout_selected"] = [{"id": "a", "compliance": True}]

    heldout_rows_by_benchmark = {
        "faitheval": benchmark_rows,
        "faitheval_anti_compliance_margin": {
            family: _family_rows(
                ordered_ids,
                metric_values={"a": 0.0, "b": 0.0},
            )
            for family in FAMILY_ORDER
        },
    }

    with pytest.raises(
        ValueError,
        match="Held-out sample-ID parity failed for faitheval/readout_selected",
    ):
        build_heldout_summary(
            selector_summary=selector_summary,
            ordered_ids=ordered_ids,
            heldout_rows_by_benchmark=heldout_rows_by_benchmark,
        )


def test_wrapper_skips_expanded_bundle_when_all_outputs_exist(
    tmp_path: Path,
) -> None:
    dep_timestamp = 1_700_000_000
    repo_root = Path(__file__).resolve().parents[1]
    workdir = tmp_path
    script_path = repo_root / "scripts/infra/faitheval_sae_utility_selector.sh"

    (workdir / "scripts/infra").mkdir(parents=True)
    for rel_path in [
        "scripts/select_faitheval_sae_utility_features.py",
        "scripts/report_faitheval_sae_utility_selector.py",
        "scripts/run_intervention.py",
        "scripts/lib/pipeline.py",
        "models/sae_detector.pkl",
        "data/gemma3_4b/pipeline/classifier_sae_summary.json",
    ]:
        path = workdir / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# stub\n", encoding="utf-8")
        _set_mtime(path, dep_timestamp)

    selector_dir = (
        workdir / "data/gemma3_4b/intervention/faitheval_sae_utility_selector/selector"
    )
    selector_dir.mkdir(parents=True, exist_ok=True)
    for name in [
        "validation_manifest.json",
        "test_manifest.json",
        "candidate_pool.json",
        "feature_stats.json",
        "full_sequence_feature_stats.json",
        "utility_scores.jsonl",
        "utility_selected_features.json",
        "readout_selected_features.json",
        "matched_random_seed_0_features.json",
        "matched_random_seed_1_features.json",
        "matched_random_seed_2_features.json",
        "selector_summary.json",
    ]:
        artifact_path = selector_dir / name
        artifact_path.write_text("{}", encoding="utf-8")
        _set_mtime(artifact_path, dep_timestamp)

    heldout_root = (
        workdir / "data/gemma3_4b/intervention/faitheval_sae_utility_selector/heldout"
    )
    for benchmark in ["faitheval", "faitheval_anti_compliance_margin"]:
        for family in FAMILY_ORDER:
            alpha_label = "1.0" if family == "noop" else "0.0"
            family_dir = heldout_root / benchmark / family / "experiment"
            family_dir.mkdir(parents=True, exist_ok=True)
            alpha_path = family_dir / f"alpha_{alpha_label}.jsonl"
            alpha_path.write_text(
                "{}\n",
                encoding="utf-8",
            )
            _set_mtime(alpha_path, dep_timestamp)
            result_path = family_dir / "results.20260421_010203.json"
            result_path.write_text(
                "{}",
                encoding="utf-8",
            )
            _set_mtime(result_path, dep_timestamp)

    report_dir = (
        workdir / "data/gemma3_4b/intervention/faitheval_sae_utility_selector/report"
    )
    report_dir.mkdir(parents=True, exist_ok=True)
    summary_path = report_dir / "heldout_summary.json"
    summary_path.write_text("{}", encoding="utf-8")
    _set_mtime(summary_path, dep_timestamp)
    audit_path = report_dir / "audit_note.md"
    audit_path.write_text("# stub\n", encoding="utf-8")
    _set_mtime(audit_path, dep_timestamp)

    bin_dir = workdir / "bin"
    bin_dir.mkdir()
    (bin_dir / "nvitop").write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    (bin_dir / "systemd-inhibit").write_text(
        '#!/usr/bin/env bash\necho systemd-inhibit >>"$LOG_FILE"\nshift 2\n"$@"\n',
        encoding="utf-8",
    )
    (bin_dir / "uv").write_text(
        '#!/usr/bin/env bash\necho "uv $*" >>"$LOG_FILE"\nif [[ "$*" == *"check-sentinel"* ]]; then\n  exit 1\nfi\nexit 0\n',
        encoding="utf-8",
    )
    for executable in [bin_dir / "nvitop", bin_dir / "systemd-inhibit", bin_dir / "uv"]:
        executable.chmod(0o755)

    log_file = workdir / "tool.log"
    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    env["LOG_FILE"] = str(log_file)

    completed = subprocess.run(
        ["bash", str(script_path)],
        cwd=workdir,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "Skipping selector stage; found complete selector bundle" in completed.stdout
    assert completed.stdout.count("Skipping held-out stage; found") == 12
    assert "Skipping report stage; found fresh report outputs" in completed.stdout
    assert not log_file.exists() or "run_intervention.py" not in log_file.read_text(
        encoding="utf-8"
    )
    if log_file.exists():
        assert "log-run" not in log_file.read_text(encoding="utf-8")


def test_wrapper_reruns_stale_heldout_outputs_when_manifest_is_newer(
    tmp_path: Path,
) -> None:
    old_timestamp = 1_700_000_000
    new_timestamp = old_timestamp + 60
    repo_root = Path(__file__).resolve().parents[1]
    workdir = tmp_path
    script_path = repo_root / "scripts/infra/faitheval_sae_utility_selector.sh"

    (workdir / "scripts/infra").mkdir(parents=True)
    for rel_path in [
        "scripts/select_faitheval_sae_utility_features.py",
        "scripts/report_faitheval_sae_utility_selector.py",
        "scripts/run_intervention.py",
        "scripts/lib/pipeline.py",
        "models/sae_detector.pkl",
        "data/gemma3_4b/pipeline/classifier_sae_summary.json",
    ]:
        path = workdir / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# stub\n", encoding="utf-8")
        _set_mtime(path, old_timestamp)

    selector_dir = (
        workdir / "data/gemma3_4b/intervention/faitheval_sae_utility_selector/selector"
    )
    selector_dir.mkdir(parents=True, exist_ok=True)
    selector_files = [
        "validation_manifest.json",
        "test_manifest.json",
        "candidate_pool.json",
        "feature_stats.json",
        "full_sequence_feature_stats.json",
        "utility_scores.jsonl",
        "utility_selected_features.json",
        "readout_selected_features.json",
        "matched_random_seed_0_features.json",
        "matched_random_seed_1_features.json",
        "matched_random_seed_2_features.json",
        "selector_summary.json",
    ]
    for name in selector_files:
        artifact_path = selector_dir / name
        artifact_path.write_text("{}", encoding="utf-8")
        _set_mtime(artifact_path, old_timestamp)

    heldout_root = (
        workdir / "data/gemma3_4b/intervention/faitheval_sae_utility_selector/heldout"
    )
    for benchmark in ["faitheval", "faitheval_anti_compliance_margin"]:
        for family in FAMILY_ORDER:
            alpha_label = "1.0" if family == "noop" else "0.0"
            family_dir = heldout_root / benchmark / family / "experiment"
            family_dir.mkdir(parents=True, exist_ok=True)
            alpha_path = family_dir / f"alpha_{alpha_label}.jsonl"
            alpha_path.write_text(
                "{}\n",
                encoding="utf-8",
            )
            _set_mtime(alpha_path, old_timestamp)
            result_path = family_dir / "results.20260421_010203.json"
            result_path.write_text(
                "{}",
                encoding="utf-8",
            )
            _set_mtime(result_path, old_timestamp)

    report_dir = (
        workdir / "data/gemma3_4b/intervention/faitheval_sae_utility_selector/report"
    )
    report_dir.mkdir(parents=True, exist_ok=True)
    summary_path = report_dir / "heldout_summary.json"
    summary_path.write_text("{}", encoding="utf-8")
    _set_mtime(summary_path, old_timestamp)
    audit_path = report_dir / "audit_note.md"
    audit_path.write_text("# stub\n", encoding="utf-8")
    _set_mtime(audit_path, old_timestamp)

    stale_manifest = selector_dir / "utility_selected_features.json"
    stale_manifest.write_text('{"updated": true}\n', encoding="utf-8")
    _set_mtime(stale_manifest, new_timestamp)

    bin_dir = workdir / "bin"
    bin_dir.mkdir()
    (bin_dir / "nvitop").write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    (bin_dir / "systemd-inhibit").write_text(
        '#!/usr/bin/env bash\necho "systemd-inhibit $*" >>"$LOG_FILE"\nshift 2\n"$@"\n',
        encoding="utf-8",
    )
    (bin_dir / "uv").write_text(
        '#!/usr/bin/env bash\necho "uv $*" >>"$LOG_FILE"\nif [[ "$*" == *"check-sentinel"* ]]; then\n  exit 1\nfi\nexit 0\n',
        encoding="utf-8",
    )
    for executable in [bin_dir / "nvitop", bin_dir / "systemd-inhibit", bin_dir / "uv"]:
        executable.chmod(0o755)

    log_file = workdir / "tool.log"
    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    env["LOG_FILE"] = str(log_file)

    completed = subprocess.run(
        ["bash", str(script_path)],
        cwd=workdir,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    log_text = log_file.read_text(encoding="utf-8")
    assert (
        "run_intervention.py --model_path google/gemma-3-4b-it --device_map cuda:0 --benchmark faitheval --prompt_style anti_compliance --intervention_mode sae --sae_feature_manifest data/gemma3_4b/intervention/faitheval_sae_utility_selector/selector/utility_selected_features.json"
        in log_text
    )
    assert (
        "run_intervention.py --model_path google/gemma-3-4b-it --device_map cuda:0 --benchmark faitheval_anti_compliance_margin --prompt_style anti_compliance --intervention_mode sae --sae_feature_manifest data/gemma3_4b/intervention/faitheval_sae_utility_selector/selector/utility_selected_features.json"
        in log_text
    )
    assert "python -m scripts.lib.pipeline log-run" in log_text
