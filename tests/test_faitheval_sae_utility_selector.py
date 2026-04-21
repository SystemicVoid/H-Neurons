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
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from intervene_sae import load_sae_feature_manifest  # noqa: E402
from report_faitheval_sae_utility_selector import build_heldout_summary  # noqa: E402
from run_intervention import (  # noqa: E402
    _build_faitheval_sample,
    load_sample_manifest_ids,
    resolve_sae_target_features,
)
from select_faitheval_sae_utility_features import (  # noqa: E402
    _choice_token_ids,
    build_stratified_faitheval_split,
    match_random_zero_weight_features,
    misleading_margin,
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


def test_stratified_faitheval_split_is_deterministic_and_preserves_shape() -> None:
    samples = []
    for idx in range(24):
        samples.append(
            {
                "id": f"s{idx}",
                "num_options": 4 if idx < 18 else 5,
                "counterfactual_key": "A" if idx % 2 == 0 else "B",
            }
        )

    val_a, test_a, meta_a = build_stratified_faitheval_split(
        samples,
        validation_size=8,
        seed=42,
    )
    val_b, test_b, meta_b = build_stratified_faitheval_split(
        samples,
        validation_size=8,
        seed=42,
    )

    assert [row["id"] for row in val_a] == [row["id"] for row in val_b]
    assert [row["id"] for row in test_a] == [row["id"] for row in test_b]
    assert meta_a == meta_b
    assert len(val_a) == 8
    assert len(test_a) == 16
    assert set(row["id"] for row in val_a).isdisjoint(row["id"] for row in test_a)


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


def test_random_matching_is_layer_exact_and_avoids_selected_overlap() -> None:
    selected = [
        {"layer": 5, "feature": 0, "flat_idx": 0},
        {"layer": 9, "feature": 0, "flat_idx": 4},
    ]
    zero_pool = [
        {"layer": 5, "feature": 1, "flat_idx": 1},
        {"layer": 5, "feature": 2, "flat_idx": 2},
        {"layer": 9, "feature": 1, "flat_idx": 5},
        {"layer": 9, "feature": 2, "flat_idx": 6},
    ]
    feature_stats = {
        0: {"activation_frequency": 0.10, "decoder_norm": 1.00},
        1: {"activation_frequency": 0.11, "decoder_norm": 1.01},
        2: {"activation_frequency": 0.40, "decoder_norm": 1.90},
        4: {"activation_frequency": 0.30, "decoder_norm": 0.80},
        5: {"activation_frequency": 0.31, "decoder_norm": 0.82},
        6: {"activation_frequency": 0.70, "decoder_norm": 1.70},
    }

    matched = match_random_zero_weight_features(
        selected,
        zero_pool,
        feature_stats,
        seed=0,
    )

    assert {row["flat_idx"] for row in matched} == {1, 5}
    assert [row["layer"] for row in matched] == [5, 9]
    assert {row["flat_idx"] for row in matched}.isdisjoint(
        row["flat_idx"] for row in selected
    )


def test_report_summary_math_and_field_names() -> None:
    selector_summary = {
        "utility_selected": {
            "k": 2,
            "outside_old_shortlist_count": 1,
            "outside_old_shortlist_fraction": 0.5,
            "layer_histogram": {"5": 1, "9": 1},
        },
        "readout_reference": {"layer_histogram": {"5": 2}},
        "overlap_with_readout_positive": {"intersection_count": 1, "jaccard": 1 / 3},
    }
    baseline_rows = [
        {"id": "a", "compliance": True},
        {"id": "b", "compliance": False},
        {"id": "c", "compliance": True},
        {"id": "d", "compliance": False},
    ]
    intervention_rows = [
        {"id": "a", "compliance": False},
        {"id": "b", "compliance": False},
        {"id": "c", "compliance": True},
        {"id": "d", "compliance": False},
    ]

    summary = build_heldout_summary(
        selector_summary=selector_summary,
        baseline_rows=baseline_rows,
        intervention_rows=intervention_rows,
        baseline_alpha=1.0,
        intervention_alpha=0.0,
    )

    assert summary["heldout_compliance"]["noop"]["n_compliant"] == 2
    assert summary["heldout_compliance"]["utility_selected"]["n_compliant"] == 1
    assert (
        summary["heldout_compliance"]["utility_minus_noop_pp"]["estimate_pp"] == -25.0
    )
    assert summary["selector_diagnostics"]["outside_old_shortlist_fraction"] == 0.5
    assert summary["selector_diagnostics"]["utility_layer_histogram"] == {
        "5": 1,
        "9": 1,
    }


def test_wrapper_skips_heldout_stage_when_timestamped_results_exist(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    workdir = tmp_path
    script_path = repo_root / "scripts/infra/faitheval_sae_utility_selector.sh"

    (workdir / "scripts/infra").mkdir(parents=True)
    for rel_path in [
        "scripts/select_faitheval_sae_utility_features.py",
        "scripts/report_faitheval_sae_utility_selector.py",
        "scripts/run_intervention.py",
        "models/sae_detector.pkl",
        "data/gemma3_4b/pipeline/classifier_sae_summary.json",
    ]:
        path = workdir / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# stub\n", encoding="utf-8")

    selector_dir = (
        workdir / "data/gemma3_4b/intervention/faitheval_sae_utility_selector/selector"
    )
    selector_dir.mkdir(parents=True, exist_ok=True)
    (selector_dir / "selector_summary.json").write_text("{}", encoding="utf-8")

    heldout_dir = (
        workdir
        / "data/gemma3_4b/intervention/faitheval_sae_utility_selector/heldout"
        / "utility_selected/experiment"
    )
    heldout_dir.mkdir(parents=True, exist_ok=True)
    (heldout_dir / "results.20260421_010203.json").write_text(
        "{}",
        encoding="utf-8",
    )

    report_dir = (
        workdir / "data/gemma3_4b/intervention/faitheval_sae_utility_selector/report"
    )
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "heldout_summary.json").write_text("{}", encoding="utf-8")

    bin_dir = workdir / "bin"
    bin_dir.mkdir()
    (bin_dir / "nvitop").write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    (bin_dir / "systemd-inhibit").write_text(
        '#!/usr/bin/env bash\necho systemd-inhibit >>"$LOG_FILE"\nshift 2\n"$@"\n',
        encoding="utf-8",
    )
    (bin_dir / "uv").write_text(
        '#!/usr/bin/env bash\necho "uv $*" >>"$LOG_FILE"\nexit 0\n',
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
    assert "Skipping held-out stage; found existing results summary" in completed.stdout
    assert not log_file.exists() or "run_intervention.py" not in log_file.read_text(
        encoding="utf-8"
    )
