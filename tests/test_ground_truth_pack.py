from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
PACK_DIR = ROOT / "notes" / "ground-truth"
MANIFEST_PATH = PACK_DIR / "_sources.json"
BUILD_SCRIPT = ROOT / "scripts" / "build_ground_truth_pack.py"


def load_builder_module():
    spec = importlib.util.spec_from_file_location(
        "ground_truth_pack_builder", BUILD_SCRIPT
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PACK = load_builder_module()


def run_builder() -> None:
    subprocess.run([sys.executable, str(BUILD_SCRIPT)], cwd=ROOT, check=True)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict]:
    return PACK.load_jsonl(path)


@pytest.fixture(scope="module", autouse=True)
def rebuild_pack() -> None:
    run_builder()


def strip_authored_markdown(text: str) -> list[str]:
    lines: list[str] = []
    fence: str | None = None
    for line in text.splitlines():
        stripped = line.strip()
        fence_match = re.match(r"^(`{3,})", stripped)
        if fence_match:
            if fence is None:
                fence = fence_match.group(1)
            elif len(fence_match.group(1)) >= len(fence):
                fence = None
            continue
        if fence is not None:
            continue
        if not stripped:
            continue
        if stripped in {"python", "json", "text"}:
            continue
        if stripped.startswith("|"):
            continue
        lines.append(stripped)
    return lines


def select_quantile_ids(
    rows: list[dict], score_key: str, tie_key: str, fractions: list[float]
) -> list[str]:
    scores = sorted(float(row[score_key]) for row in rows)
    picked: list[str] = []
    used: set[str] = set()
    for fraction in fractions:
        target = PACK.percentile(scores, fraction)
        ranked = sorted(
            rows,
            key=lambda row: (abs(float(row[score_key]) - target), str(row[tie_key])),
        )
        for row in ranked:
            tie_value = str(row[tie_key])
            if tie_value not in used:
                picked.append(tie_value)
                used.add(tie_value)
                break
    return picked


def response_text(row: dict) -> str:
    return str(row.get("response", row.get("chosen", "")))


def response_length(row: dict) -> int:
    return len(response_text(row))


def load_keyed(path: Path) -> dict[str, dict]:
    return PACK.load_keyed_jsonl(path)


def merge_rows_by_id(paths: list[Path]) -> dict[str, dict]:
    merged: dict[str, dict] = {}
    for path in paths:
        for row in load_jsonl(path):
            sample_id = str(row["id"])
            assert sample_id not in merged, (sample_id, path)
            merged[sample_id] = row
    return merged


def test_pack_outputs_exist_and_rebuild_is_byte_stable() -> None:
    expected = {
        "README.md",
        "01_readout_surfaces.md",
        "02_intervention_surfaces.md",
        "03_measurement_surfaces.md",
        "04_transfer_externality_surfaces.md",
        "05_mechanism_diagnostic_surfaces.md",
        "artifact_manifest.jsonl",
        "metric_ledger.jsonl",
        "example_ledger.jsonl",
        "surface_crosswalk.jsonl",
        "_sources.json",
    }
    assert expected.issubset({path.name for path in PACK_DIR.iterdir()})
    before = {
        name: (PACK_DIR / name).read_bytes()
        for name in expected
        if (PACK_DIR / name).exists()
    }
    run_builder()
    after = {
        name: (PACK_DIR / name).read_bytes()
        for name in expected
        if (PACK_DIR / name).exists()
    }
    assert before == after


def test_multiline_jsonl_recovery_handles_split_records() -> None:
    answer_rows = load_keyed(
        ROOT / "data" / "gemma3_4b" / "pipeline" / "answer_tokens.jsonl"
    )
    consistency_rows = load_keyed(
        ROOT / "data" / "gemma3_4b" / "pipeline" / "consistency_samples.jsonl"
    )
    split_answer = json.dumps(
        {"qz_2447": answer_rows["qz_2447"]}, indent=2, ensure_ascii=False
    )
    split_consistency = json.dumps(
        {"tc_1": consistency_rows["tc_1"]}, indent=2, ensure_ascii=False
    )
    tmp = ROOT / "notes" / "ground-truth" / "_jsonl_recovery_test.jsonl"
    try:
        tmp.write_text(f"{split_answer}\n{split_consistency}\n", encoding="utf-8")
        rows = PACK.load_jsonl(tmp)
        ids = {next(iter(row)) if len(row) == 1 else row["id"] for row in rows}
        assert "qz_2447" in ids
        assert "tc_1" in ids
    finally:
        tmp.unlink(missing_ok=True)


def test_artifact_metric_and_example_ledgers_have_new_schema_and_declared_sources() -> (
    None
):
    manifest = load_json(MANIFEST_PATH)
    artifact_ids = {artifact["artifact_id"] for artifact in manifest["artifacts"]}
    artifact_rows = load_jsonl(PACK_DIR / "artifact_manifest.jsonl")
    metric_rows = load_jsonl(PACK_DIR / "metric_ledger.jsonl")
    example_rows = load_jsonl(PACK_DIR / "example_ledger.jsonl")

    assert len(artifact_rows) == len(manifest["artifacts"])

    metric_keys = {
        "metric_id",
        "family",
        "act_tags",
        "benchmark",
        "model",
        "intervention_family",
        "metric_name",
        "comparison_type",
        "condition_a",
        "condition_b",
        "estimate",
        "unit",
        "n",
        "paired",
        "ci_lower",
        "ci_upper",
        "ci_level",
        "ci_method",
        "source_artifact_ids",
        "source_format",
        "source_locators",
    }
    example_keys = {
        "example_id",
        "family",
        "act_tags",
        "benchmark",
        "source_artifact_id",
        "source_artifact_ids",
        "sample_id",
        "selection_stratum",
        "selection_rank",
        "condition_metadata",
        "prompt_or_question",
        "response_text",
        "raw_label_fields",
        "judge_fields",
        "paired_example_ids",
    }

    for row in metric_rows:
        assert set(row) == metric_keys
        assert PACK.is_numeric_scalar(row["estimate"])
        for field_name in ("ci_lower", "ci_upper", "ci_level"):
            if row[field_name] is not None:
                assert PACK.is_numeric_scalar(row[field_name]), (
                    row["metric_id"],
                    field_name,
                )
        assert row["source_artifact_ids"]
        assert set(row["source_artifact_ids"]) <= artifact_ids
        assert row["source_locators"]
        for locator in row["source_locators"]:
            assert set(locator) == {
                "artifact_id",
                "source_path",
                "locator",
                "derivation",
            }
            assert locator["artifact_id"] in artifact_ids
    for row in example_rows:
        assert set(row) == example_keys
        assert row["source_artifact_id"] in artifact_ids
        assert set(row["source_artifact_ids"]) <= artifact_ids
        assert row["prompt_or_question"]
        assert row["response_text"]
        assert "selection_rule" in row["condition_metadata"]
        assert "selection_score" in row["condition_metadata"]
        assert "selection_tiebreak" in row["condition_metadata"]


def test_surface_crosswalk_covers_declared_scalars_and_structured_claims() -> None:
    manifest = load_json(MANIFEST_PATH)
    crosswalk = load_jsonl(PACK_DIR / "surface_crosswalk.jsonl")
    by_id = {row["surface_id"]: row for row in crosswalk}

    for artifact in manifest["artifacts"]:
        assert f"source:{artifact['artifact_id']}" in by_id

    for surface_path in manifest["coverage_surfaces"]:
        if not surface_path.endswith(".json"):
            continue
        for locator, _ in PACK.iter_surface_scalars(load_json(ROOT / surface_path)):
            assert f"surface:{surface_path}:{locator}" in by_id

    provenance_rows = [
        row for row in crosswalk if row["surface_kind"] == "manuscript_provenance_row"
    ]
    unresolved = {
        "contamination-bug-borderline-records-reclassified",
        "contamination-bug-strict-harmfulness-inflation",
        "false-negatives-recovered-by-strongreject-model-upgrade",
        "gradient-ranked-pilot-effect",
        "h-minus-random-permutation-test",
        "historical-april-8-legacy-ruler-causal-full-500-effect",
        "holdout-gap-after-strongreject-model-upgrade",
        "persistent-strongreject-false-negatives-after-upgrade",
        "probe-causal-jaccard",
        "probe-selector-pilot-best-effect",
        "strongreject-gpt-4o-dev-accuracy-after-rerun",
        "strongreject-gpt-4o-mini-dev-accuracy-prior",
    }
    seen_unresolved = set()
    for row in provenance_rows:
        slug = row["surface_id"].split("provenance:", 1)[1]
        if slug in unresolved:
            assert row["coverage_status"] == "unresolved"
            seen_unresolved.add(slug)
        else:
            assert row["coverage_status"] == "exact_metric"
            assert row["ledger_metric_ids"]
    assert seen_unresolved == unresolved


def test_sae_utility_selector_metrics_use_april_22_surfaces_not_legacy_control() -> (
    None
):
    metrics = {
        row["metric_id"]: row for row in load_jsonl(PACK_DIR / "metric_ledger.jsonl")
    }
    legacy_prefix = "data/gemma3_4b/intervention/faitheval_sae/control/"
    expected = {
        "intervention.faitheval_sae_utility.readout_overlap_jaccard": (
            "faitheval_sae_utility_selector_summary",
            "data/gemma3_4b/intervention/faitheval_sae_utility_selector/selector/selector_summary.json",
            "family_overlap.utility_selected_vs_readout_selected.jaccard",
        ),
        "intervention.faitheval_sae_utility.utility_minus_readout_margin": (
            "faitheval_sae_utility_selector_heldout",
            "data/gemma3_4b/intervention/faitheval_sae_utility_selector/report/heldout_summary.json",
            "heldout_anti_compliance_margin.paired_deltas.utility_minus_readout.estimate",
        ),
        "intervention.faitheval_sae_utility_positive.utility_positive_minus_noop_margin": (
            "faitheval_sae_utility_selector_augment",
            "data/gemma3_4b/intervention/faitheval_sae_utility_selector/report_augment/augment_heldout_summary.json",
            "heldout_anti_compliance_margin.paired_deltas.utility_positive_minus_noop.estimate",
        ),
    }

    for metric_id, (artifact_id, source_path, locator) in expected.items():
        metric = metrics[metric_id]
        assert metric["source_artifact_ids"] == [artifact_id]
        assert metric["source_locators"] == [
            {
                "artifact_id": artifact_id,
                "source_path": source_path,
                "locator": locator,
                "derivation": "direct",
            }
        ]
        assert all(
            not ref["source_path"].startswith(legacy_prefix)
            for ref in metric["source_locators"]
        )


def test_new_sae_utility_surfaces_have_manifest_and_crosswalk_coverage() -> None:
    manifest = load_json(MANIFEST_PATH)
    artifact_ids = {artifact["artifact_id"] for artifact in manifest["artifacts"]}
    assert {
        "faitheval_sae_utility_selector_summary",
        "faitheval_sae_utility_selector_heldout",
        "faitheval_sae_utility_selector_audit",
        "faitheval_sae_utility_selector_augment",
        "faitheval_sae_utility_selector_augment_audit",
    } <= artifact_ids

    crosswalk = {
        row["surface_id"]: row
        for row in load_jsonl(PACK_DIR / "surface_crosswalk.jsonl")
    }
    expected = {
        "surface:data/gemma3_4b/intervention/faitheval_sae_utility_selector/selector/selector_summary.json:family_overlap.utility_selected_vs_readout_selected.jaccard": "intervention.faitheval_sae_utility.readout_overlap_jaccard",
        "surface:data/gemma3_4b/intervention/faitheval_sae_utility_selector/report/heldout_summary.json:heldout_anti_compliance_margin.paired_deltas.utility_minus_readout.estimate": "intervention.faitheval_sae_utility.utility_minus_readout_margin",
        "surface:data/gemma3_4b/intervention/faitheval_sae_utility_selector/report_augment/augment_heldout_summary.json:heldout_anti_compliance_margin.paired_deltas.utility_positive_minus_noop.estimate": "intervention.faitheval_sae_utility_positive.utility_positive_minus_noop_margin",
    }
    for surface_id, metric_id in expected.items():
        assert crosswalk[surface_id]["coverage_status"] == "exact_metric"
        assert crosswalk[surface_id]["ledger_metric_ids"] == [metric_id]

    assert (
        crosswalk["source:faitheval_sae_utility_selector_audit"]["coverage_status"]
        == "markdown_fallback_only"
    )
    assert (
        crosswalk["source:faitheval_sae_utility_selector_augment_audit"][
            "coverage_status"
        ]
        == "markdown_fallback_only"
    )


def test_faitheval_mean_slope_difference_uses_seed_bootstrap_mean_contract() -> None:
    summary = load_json(
        ROOT
        / "data/gemma3_4b/intervention/faitheval/control/slope_difference_summary.json"
    )
    metric = next(
        row
        for row in load_jsonl(PACK_DIR / "metric_ledger.jsonl")
        if row["metric_id"]
        == "intervention.faitheval.mean_slope_difference_pp_per_alpha"
    )
    seed_estimates = [
        row["slope_difference_pp_per_alpha"]["estimate"]
        for row in summary["per_seed"].values()
    ]
    expected = PACK.bootstrap_mean_ci(seed_estimates)

    assert metric["estimate"] == pytest.approx(expected["estimate"])
    assert metric["n"] == 8
    assert metric["paired"] is True
    assert metric["ci_method"] == "bootstrap_percentile_mean"
    assert metric["ci_lower"] == pytest.approx(expected["ci_lower"])
    assert metric["ci_upper"] == pytest.approx(expected["ci_upper"])
    assert metric["source_locators"] == [
        {
            "artifact_id": "faitheval_control_slope_difference",
            "source_path": "data/gemma3_4b/intervention/faitheval/control/slope_difference_summary.json",
            "locator": "per_seed.*.slope_difference_pp_per_alpha.estimate",
            "derivation": "direct",
        }
    ]


def test_d7_metric_contracts_use_real_raw_locators_and_exact_site_scalars() -> None:
    metrics = {
        row["metric_id"]: row for row in load_jsonl(PACK_DIR / "metric_ledger.jsonl")
    }
    crosswalk = {
        row["surface_id"]: row
        for row in load_jsonl(PACK_DIR / "surface_crosswalk.jsonl")
    }

    token_cap = metrics["mechanism.d7.current_panel.causal_token_cap_count"]
    assert token_cap["estimate"] == 112
    assert token_cap["source_locators"] == [
        {
            "artifact_id": "d7_current_state_summary",
            "source_path": "data/gemma3_4b/intervention/jailbreak_d7/full500_canonical/d7_full500_current_state_summary.json",
            "locator": "artifact_status.causal_locked.token_cap.count",
            "derivation": "direct",
        },
        {
            "artifact_id": "d7_comparison_site",
            "source_path": "site/data/d7_comparison.json",
            "locator": "token_cap.causal_hits",
            "derivation": "mirrored_surface",
        },
    ]

    expected_gaps = {
        "mechanism.d7.current_panel.causal_vs_probe_gap_pp": (
            "current_panel.direct_comparisons.probe_vs_causal.strict_harmfulness_normalized.estimate_pp",
            "current_state.current_panel.direct_probe_vs_causal.strict_harmfulness_normalized.estimate",
            "surface:site/data/d7_comparison.json:current_state.current_panel.direct_probe_vs_causal.strict_harmfulness_normalized.estimate",
        ),
        "mechanism.d7.current_panel.causal_vs_random_layer_seed1_gap_pp": (
            "current_panel.direct_comparisons.random_layer_seed1_vs_causal.strict_harmfulness_normalized.estimate_pp",
            "current_state.current_panel.direct_random_layer_seed1_vs_causal.strict_harmfulness_normalized.estimate",
            "surface:site/data/d7_comparison.json:current_state.current_panel.direct_random_layer_seed1_vs_causal.strict_harmfulness_normalized.estimate",
        ),
        "mechanism.d7.current_panel.causal_vs_random_layer_seed2_gap_pp": (
            "current_panel.direct_comparisons.random_layer_seed2_vs_causal.strict_harmfulness_normalized.estimate_pp",
            "current_state.current_panel.direct_random_layer_seed2_vs_causal.strict_harmfulness_normalized.estimate",
            "surface:site/data/d7_comparison.json:current_state.current_panel.direct_random_layer_seed2_vs_causal.strict_harmfulness_normalized.estimate",
        ),
    }
    for metric_id, (raw_locator, site_locator, surface_id) in expected_gaps.items():
        metric = metrics[metric_id]
        assert metric["paired"] is True
        assert {locator["locator"] for locator in metric["source_locators"]} == {
            raw_locator,
            site_locator,
        }
        assert "d7_comparison_site" in metric["source_artifact_ids"]
        assert crosswalk[surface_id]["coverage_status"] == "exact_metric"
        assert crosswalk[surface_id]["ledger_metric_ids"] == [metric_id]


def test_bridge_margins_results_are_present_and_crosswalked() -> None:
    results = load_json(
        ROOT / "data/gemma3_4b/analysis/bridge_margins/test/results.json"
    )
    metrics = {
        row["metric_id"]: row for row in load_jsonl(PACK_DIR / "metric_ledger.jsonl")
    }
    crosswalk = {
        row["surface_id"]: row
        for row in load_jsonl(PACK_DIR / "surface_crosswalk.jsonl")
    }

    sample_size = metrics["transfer.bridge_margins.sample_size"]
    assert sample_size["estimate"] == results["n_records"]
    assert sample_size["source_locators"] == [
        {
            "artifact_id": "bridge_margins_test_results",
            "source_path": "data/gemma3_4b/analysis/bridge_margins/test/results.json",
            "locator": "n_records",
            "derivation": "direct",
        }
    ]

    a_shift = results["cohorts"]["A_rw_substitution"]["first3"]["shift_nats"]
    a_metric = metrics["transfer.bridge_margins.a_rw_substitution.first3_shift_nats"]
    assert a_metric["estimate"] == pytest.approx(a_shift["estimate"])
    assert a_metric["n"] == a_shift["n"]
    assert a_metric["paired"] is True
    assert a_metric["ci_lower"] == pytest.approx(a_shift["ci"]["lower"])
    assert a_metric["ci_upper"] == pytest.approx(a_shift["ci"]["upper"])
    assert a_metric["source_locators"] == [
        {
            "artifact_id": "bridge_margins_test_results",
            "source_path": "data/gemma3_4b/analysis/bridge_margins/test/results.json",
            "locator": "cohorts.A_rw_substitution.first3.shift_nats.estimate",
            "derivation": "direct",
        }
    ]

    comparison = results["between_cohort"]["A_vs_D"]["first3"]["shift_nats"]
    gap_metric = metrics["transfer.bridge_margins.a_vs_d.first3_shift_nats_gap"]
    p_metric = metrics["transfer.bridge_margins.a_vs_d.first3_shift_nats_p"]
    assert gap_metric["estimate"] == pytest.approx(comparison["estimate"])
    assert gap_metric["ci_lower"] == pytest.approx(comparison["ci"]["lower"])
    assert gap_metric["ci_upper"] == pytest.approx(comparison["ci"]["upper"])
    assert gap_metric["n"] == comparison["n_a"] + comparison["n_b"]
    assert p_metric["estimate"] == pytest.approx(
        comparison["permutation_test"]["p_value"]
    )
    assert p_metric["source_locators"] == [
        {
            "artifact_id": "bridge_margins_test_results",
            "source_path": "data/gemma3_4b/analysis/bridge_margins/test/results.json",
            "locator": "between_cohort.A_vs_D.first3.shift_nats.permutation_test.p_value",
            "derivation": "direct",
        }
    ]

    expected = {
        "surface:data/gemma3_4b/analysis/bridge_margins/test/results.json:n_records": "transfer.bridge_margins.sample_size",
        "surface:data/gemma3_4b/analysis/bridge_margins/test/results.json:cohorts.A_rw_substitution.first3.shift_nats.estimate": "transfer.bridge_margins.a_rw_substitution.first3_shift_nats",
        "surface:data/gemma3_4b/analysis/bridge_margins/test/results.json:between_cohort.A_vs_D.first3.shift_nats.estimate": "transfer.bridge_margins.a_vs_d.first3_shift_nats_gap",
        "surface:data/gemma3_4b/analysis/bridge_margins/test/results.json:between_cohort.A_vs_D.first3.shift_nats.permutation_test.p_value": "transfer.bridge_margins.a_vs_d.first3_shift_nats_p",
    }
    for surface_id, metric_id in expected.items():
        assert crosswalk[surface_id]["coverage_status"] == "exact_metric"
        assert crosswalk[surface_id]["ledger_metric_ids"] == [metric_id]


def test_provenance_same_file_and_same_audit_path_state_resolves_correctly() -> None:
    builder = PACK.Builder(load_json(MANIFEST_PATH))
    provenance_rows = {
        row["claim_slug"]: row for row in builder.parse_number_provenance()
    }
    all_eight = provenance_rows["random-neuron-null-summary-all-eight-seeds"]

    same_file_refs = [
        ref for ref in all_eight["source_refs"] if "same file" in ref["raw"].lower()
    ]
    assert same_file_refs
    assert {ref["path"] for ref in same_file_refs} == {
        "data/gemma3_4b/intervention/faitheval/control/comparison_summary.json"
    }

    same_audit_refs = [
        ref for ref in all_eight["source_refs"] if "same audit" in ref["raw"].lower()
    ]
    assert same_audit_refs == [
        {
            "path": "notes/act3-reports/2026-04-13-faitheval-slope-difference-reporting-audit.md",
            "locator": None,
            "raw": "verified in same audit",
        }
    ]


def test_readout_examples_recompute_from_probability_quantiles_with_full_support() -> (
    None
):
    summary = load_json(
        ROOT / "data/gemma3_4b/pipeline/classifier_disjoint_summary.json"
    )
    answers = load_keyed(ROOT / "data/gemma3_4b/pipeline/answer_tokens.jsonl")
    consistency = load_keyed(ROOT / "data/gemma3_4b/pipeline/consistency_samples.jsonl")
    expected: dict[str, list[str]] = defaultdict(list)
    candidates: dict[str, list[dict]] = defaultdict(list)
    for row in summary["evaluation"]["examples"]:
        qid = row["qid"]
        if qid not in answers or qid not in consistency:
            continue
        if row["prediction"] and row["label"]:
            stratum = "tp"
        elif (not row["prediction"]) and (not row["label"]):
            stratum = "tn"
        elif row["prediction"] and (not row["label"]):
            stratum = "fp"
        else:
            stratum = "fn"
        candidates[stratum].append({"qid": qid, "score": row["probability"]})
    for stratum, rows in candidates.items():
        expected[stratum] = select_quantile_ids(
            [
                {"selection_score": row["score"], "selection_tiebreak": row["qid"]}
                for row in rows
            ],
            "selection_score",
            "selection_tiebreak",
            [0.25, 0.75],
        )
    examples = [
        row
        for row in load_jsonl(PACK_DIR / "example_ledger.jsonl")
        if row["family"] == "readout_surfaces"
    ]
    by_stratum: dict[str, list[dict]] = defaultdict(list)
    for row in examples:
        by_stratum[row["selection_stratum"]].append(row)
        assert {"pipeline_answer_tokens", "pipeline_consistency_samples"} <= set(
            row["source_artifact_ids"]
        )
    for stratum, rows in by_stratum.items():
        rows = sorted(rows, key=lambda row: row["selection_rank"])
        assert [row["sample_id"] for row in rows] == expected[stratum]


def test_measurement_examples_recompute_from_complete_join_and_use_neutral_phenotypes() -> (
    None
):
    gold_by_key = {
        (row["id"], float(row["alpha"])): row
        for row in load_jsonl(
            ROOT / "tests/gold_labels/jailbreak_cross_alpha_gold.jsonl"
        )
    }
    merged = {
        "binary": PACK.Builder._merge_alpha_files(  # type: ignore[misc]
            PACK.Builder.__new__(PACK.Builder),  # type: ignore[misc]
            {
                0.0: ROOT
                / "data/gemma3_4b/intervention/jailbreak/experiment/alpha_0.0.jsonl",
                1.5: ROOT
                / "data/gemma3_4b/intervention/jailbreak/experiment/alpha_1.5.jsonl",
                3.0: ROOT
                / "data/gemma3_4b/intervention/jailbreak/experiment/alpha_3.0.jsonl",
            },
        ),
        "csv2v2": PACK.Builder._merge_alpha_files(  # type: ignore[misc]
            PACK.Builder.__new__(PACK.Builder),  # type: ignore[misc]
            {
                0.0: ROOT
                / "data/gemma3_4b/intervention/jailbreak/csv2_evaluation/alpha_0.0.jsonl",
                1.5: ROOT
                / "data/gemma3_4b/intervention/jailbreak/csv2_evaluation/alpha_1.5.jsonl",
                3.0: ROOT
                / "data/gemma3_4b/intervention/jailbreak/csv2_evaluation/alpha_3.0.jsonl",
            },
        ),
        "csv2v3": PACK.Builder._merge_alpha_files(  # type: ignore[misc]
            PACK.Builder.__new__(PACK.Builder),  # type: ignore[misc]
            {
                0.0: ROOT / "data/judge_validation/csv2_v3/alpha_0.0.jsonl",
                1.5: ROOT / "data/judge_validation/csv2_v3/alpha_1.5.jsonl",
                3.0: ROOT / "data/judge_validation/csv2_v3/alpha_3.0.jsonl",
            },
        ),
        "sr": PACK.Builder._merge_alpha_files(  # type: ignore[misc]
            PACK.Builder.__new__(PACK.Builder),  # type: ignore[misc]
            {
                0.0: ROOT / "data/judge_validation/strongreject/alpha_0.0.jsonl",
                1.5: ROOT / "data/judge_validation/strongreject/alpha_1.5.jsonl",
                3.0: ROOT / "data/judge_validation/strongreject/alpha_3.0.jsonl",
            },
        ),
    }
    candidates: dict[str, list[dict]] = defaultdict(list)
    for key, gold_row in gold_by_key.items():
        if any(key not in merged[name] for name in merged):
            continue
        sample_id, alpha = key
        phenotype = (
            f"human_{PACK.normalize_token(gold_row['human_label'])}"
            f"__binary_{PACK.normalize_token(merged['binary'][key].get('judge'))}"
            f"__csv2v2_{PACK.normalize_token(merged['csv2v2'][key].get('csv2', {}).get('harmful_binary'))}"
            f"__csv2v3_{PACK.normalize_token(merged['csv2v3'][key].get('csv2', {}).get('harmful_binary'))}"
            f"__sr_{PACK.normalize_token(merged['sr'][key].get('strongreject', {}).get('harmful_binary'))}"
            f"__{PACK.alpha_code(alpha)}"
        )
        candidates[phenotype].append(
            {"sample_id": sample_id, "score": len(merged["csv2v3"][key]["response"])}
        )
    examples = [
        row
        for row in load_jsonl(PACK_DIR / "example_ledger.jsonl")
        if row["example_id"].startswith("example.measurement.")
    ]
    assert len(examples) == 7
    for row in examples:
        assert row["selection_stratum"].startswith("human_")
        for token in PACK.NEUTRAL_STRATUM_BANNED_TOKENS:
            assert token not in row["selection_stratum"]
        stratum = row["selection_stratum"]
        expected_id = select_quantile_ids(
            [
                {
                    "selection_score": candidate["score"],
                    "selection_tiebreak": candidate["sample_id"],
                }
                for candidate in candidates[stratum]
            ],
            "selection_score",
            "selection_tiebreak",
            [0.5],
        )[0]
        assert row["sample_id"].split("@", 1)[0] == expected_id


def test_paired_examples_recompute_from_median_length_delta_and_bridge_irr_examples_from_median_incorrect_length() -> (
    None
):
    paired_configs = {
        ("intervention_surfaces", "faitheval"): (
            [ROOT / "data/gemma3_4b/intervention/faitheval/experiment/alpha_0.0.jsonl"],
            [ROOT / "data/gemma3_4b/intervention/faitheval/experiment/alpha_3.0.jsonl"],
            lambda base, comp: (
                f"{int(bool(base['compliance']))}_to_{int(bool(comp['compliance']))}"
            ),
        ),
        ("intervention_surfaces", "falseqa"): (
            [ROOT / "data/gemma3_4b/intervention/falseqa/experiment/alpha_0.0.jsonl"],
            [ROOT / "data/gemma3_4b/intervention/falseqa/experiment/alpha_3.0.jsonl"],
            lambda base, comp: (
                f"{int(bool(base['compliance']))}_to_{int(bool(comp['compliance']))}"
            ),
        ),
        ("intervention_surfaces", "bioasq"): (
            [ROOT / "data/gemma3_4b/intervention/bioasq/experiment/alpha_0.0.jsonl"],
            [ROOT / "data/gemma3_4b/intervention/bioasq/experiment/alpha_3.0.jsonl"],
            lambda base, comp: (
                f"{int(bool(base['compliance']))}_to_{int(bool(comp['compliance']))}"
            ),
        ),
        ("intervention_surfaces", "jailbreak"): (
            [ROOT / "data/gemma3_4b/intervention/jailbreak/experiment/alpha_0.0.jsonl"],
            [ROOT / "data/gemma3_4b/intervention/jailbreak/experiment/alpha_3.0.jsonl"],
            lambda base, comp: (
                f"{int(bool(base['compliance']))}_to_{int(bool(comp['compliance']))}"
            ),
        ),
        ("transfer_externality_surfaces", "triviaqa_bridge"): (
            [
                ROOT
                / "data/gemma3_4b/intervention/triviaqa_bridge/test_experiment/alpha_1.0.jsonl"
            ],
            [
                ROOT
                / "data/gemma3_4b/intervention/triviaqa_bridge_iti_e0_paperfaithful_k12_first-3-tokens/test_experiment/alpha_8.0.jsonl"
            ],
            lambda base, comp: (
                "wrong_to_right"
                if float(base.get("metric_value", 0.0)) == 0.0
                and float(comp.get("metric_value", 0.0)) == 1.0
                else "right_to_wrong"
                if float(base.get("metric_value", 0.0)) == 1.0
                and float(comp.get("metric_value", 0.0)) == 0.0
                else ""
            ),
        ),
        ("transfer_externality_surfaces", "truthfulqa_mc"): (
            [
                ROOT
                / "data/gemma3_4b/intervention/truthfulqa_mc_mc1_iti-head_truthfulqa-paperfaithful_k-12_ranked_seed-42_final-fold0-iti-heads_7723b7d6d7/experiment/alpha_0.0.jsonl",
                ROOT
                / "data/gemma3_4b/intervention/truthfulqa_mc_mc1_iti-head_truthfulqa-paperfaithful_k-12_ranked_seed-42_final-fold1-iti-heads_9a10b5307d/experiment/alpha_0.0.jsonl",
            ],
            [
                ROOT
                / "data/gemma3_4b/intervention/truthfulqa_mc_mc1_iti-head_truthfulqa-paperfaithful_k-12_ranked_seed-42_final-fold0-iti-heads_7723b7d6d7/experiment/alpha_8.0.jsonl",
                ROOT
                / "data/gemma3_4b/intervention/truthfulqa_mc_mc1_iti-head_truthfulqa-paperfaithful_k-12_ranked_seed-42_final-fold1-iti-heads_9a10b5307d/experiment/alpha_8.0.jsonl",
            ],
            lambda base, comp: (
                "wrong_to_right"
                if float(base.get("metric_value", 0.0)) == 0.0
                and float(comp.get("metric_value", 0.0)) == 1.0
                else "right_to_wrong"
                if float(base.get("metric_value", 0.0)) == 1.0
                and float(comp.get("metric_value", 0.0)) == 0.0
                else ""
            ),
        ),
        ("transfer_externality_surfaces", "simpleqa"): (
            [
                ROOT
                / "data/gemma3_4b/intervention/simpleqa_factual_phrase_iti-head_truthfulqa-paperfaithful_k-12_ranked_seed-42_iti-truthfulqa-paperfaithful-production-iti-head_79f3852513/experiment/alpha_0.0.jsonl"
            ],
            [
                ROOT
                / "data/gemma3_4b/intervention/simpleqa_factual_phrase_iti-head_truthfulqa-paperfaithful_k-12_ranked_seed-42_iti-truthfulqa-paperfaithful-production-iti-head_79f3852513/experiment/alpha_8.0.jsonl"
            ],
            lambda base, comp: (
                "correct_to_incorrect"
                if bool(base.get("compliance")) and not bool(comp.get("compliance"))
                else "incorrect_to_correct"
                if (not bool(base.get("compliance"))) and bool(comp.get("compliance"))
                else ""
            ),
        ),
        ("mechanism_diagnostic_surfaces", "jailbreak_d7_full500"): (
            [
                ROOT
                / "data/gemma3_4b/intervention/jailbreak_d7/full500_canonical/baseline_noop/csv2_evaluation/alpha_1.0.jsonl"
            ],
            [
                ROOT
                / "data/gemma3_4b/intervention/jailbreak_d7/full500_canonical/causal_locked/csv2_evaluation/alpha_4.0.jsonl"
            ],
            lambda base, comp: (
                "wrong_to_right"
                if (not bool(base.get("compliance"))) and bool(comp.get("compliance"))
                else "right_to_wrong"
                if bool(base.get("compliance")) and (not bool(comp.get("compliance")))
                else "stayed_wrong"
                if (not bool(base.get("compliance")))
                and (not bool(comp.get("compliance")))
                else "stayed_right"
            ),
        ),
    }

    examples = load_jsonl(PACK_DIR / "example_ledger.jsonl")
    for (family, benchmark), (
        baseline_paths,
        comparison_paths,
        transition_fn,
    ) in paired_configs.items():
        baseline = merge_rows_by_id(baseline_paths)
        comparison = merge_rows_by_id(comparison_paths)
        expected_by_stratum: dict[str, str] = {}
        candidates: dict[str, list[dict]] = defaultdict(list)
        for sample_id in sorted(set(baseline) & set(comparison)):
            base = baseline[sample_id]
            comp = comparison[sample_id]
            stratum = transition_fn(base, comp)
            if not stratum:
                continue
            candidates[stratum].append(
                {
                    "selection_score": abs(
                        response_length(comp) - response_length(base)
                    ),
                    "selection_tiebreak": sample_id,
                }
            )
        for stratum, rows in candidates.items():
            expected_by_stratum[stratum] = select_quantile_ids(
                rows, "selection_score", "selection_tiebreak", [0.5]
            )[0]
        selected = [
            row
            for row in examples
            if row["family"] == family
            and row["benchmark"] == benchmark
            and row["condition_metadata"].get("pair_role") == "baseline"
        ]
        for row in selected:
            assert row["sample_id"] == expected_by_stratum[row["selection_stratum"]]
            if (
                family == "transfer_externality_surfaces"
                and benchmark == "truthfulqa_mc"
            ):
                assert row["source_artifact_ids"] == [
                    "truthfulqa_mc1_iti_fold0_alpha0",
                    "truthfulqa_mc1_iti_fold0_alpha8",
                    "truthfulqa_mc1_iti_fold1_alpha0",
                    "truthfulqa_mc1_iti_fold1_alpha8",
                ]

    truthfulqa_selected = {
        row["selection_stratum"]: row["sample_id"]
        for row in examples
        if row["family"] == "transfer_externality_surfaces"
        and row["benchmark"] == "truthfulqa_mc"
        and row["condition_metadata"].get("pair_role") == "baseline"
    }
    assert truthfulqa_selected == {
        "right_to_wrong": "truthfulqa_mc1_35",
        "wrong_to_right": "truthfulqa_mc1_373",
    }

    bridge_cases = {
        row["case_id"]: row
        for row in load_jsonl(
            ROOT / "data/judge_validation/bridge_irr/test_queue_key.jsonl"
        )
    }
    labels = {
        row["case_id"]: row
        for row in load_jsonl(
            ROOT / "data/judge_validation/bridge_irr/adjudicated_labels.jsonl"
        )
    }
    candidates: dict[str, list[dict]] = defaultdict(list)
    for case_id in sorted(set(bridge_cases) & set(labels)):
        case = bridge_cases[case_id]
        label = labels[case_id]
        stratum = f"{PACK.normalize_token(case['transition'])}_{PACK.normalize_token(label['label'])}"
        candidates[stratum].append(
            {
                "selection_score": len(case["incorrect_response"]),
                "selection_tiebreak": case_id,
            }
        )
    bridge_examples = [
        row
        for row in examples
        if row["example_id"].startswith("example.transfer.bridge_irr.")
    ]
    for row in bridge_examples:
        expected = select_quantile_ids(
            candidates[row["selection_stratum"]],
            "selection_score",
            "selection_tiebreak",
            [0.5],
        )[0]
        assert row["sample_id"] == expected


def test_generated_markdown_matches_neutral_template_allowlist() -> None:
    allowed = [
        re.compile(r"^# .+$"),
        re.compile(r"^## (Sources|Metrics|Examples)$"),
        re.compile(r"^### `.+`$"),
        re.compile(r"^Family id: `.+`$"),
        re.compile(r"^Act tags: `.+`$"),
        re.compile(r"^Template: neutral structured surface summary\.$"),
        re.compile(r"^Sample id: `.+`$"),
        re.compile(r"^Selection stratum: `.+`$"),
        re.compile(r"^Selection rank: `\d+`$"),
        re.compile(r"^Primary source id: `.+`$"),
        re.compile(r"^Source ids: `.*`$"),
        re.compile(r"^Question$"),
        re.compile(r"^Response$"),
        re.compile(r"^Condition metadata$"),
        re.compile(r"^Raw label fields$"),
        re.compile(r"^Judge fields$"),
        re.compile(r"^Paired example ids: `.*`$"),
        re.compile(r"^Pack mode: structured-first neutral evidence pack\.$"),
        re.compile(r"^Destination: `notes/ground-truth`\.$"),
        re.compile(r"^Ledgers: .+$"),
        re.compile(r"^Coverage statuses: .+$"),
    ]
    doc_paths = [
        PACK_DIR / "README.md",
        PACK_DIR / "01_readout_surfaces.md",
        PACK_DIR / "02_intervention_surfaces.md",
        PACK_DIR / "03_measurement_surfaces.md",
        PACK_DIR / "04_transfer_externality_surfaces.md",
        PACK_DIR / "05_mechanism_diagnostic_surfaces.md",
    ]
    for path in doc_paths:
        for line in strip_authored_markdown(path.read_text(encoding="utf-8")):
            assert any(pattern.match(line) for pattern in allowed), (path, line)
