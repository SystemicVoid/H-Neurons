from argparse import Namespace
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


class StubTokenizer:
    """Whitespace-split tokenizer for cohort-assembly tests.

    Each word maps to a stable int id via hash; enough to exercise
    bucketing/length logic without loading HF."""

    def __call__(self, text: str, **kwargs: Any) -> dict[str, list[int]]:
        if not text:
            return {"input_ids": []}
        words = text.strip().split()
        return {"input_ids": [(hash(w) & 0xFFFF) or 1 for w in words]}


def test_cohort_for_case_partitions_all_transitions() -> None:
    from score_bridge_margins import cohort_for_case

    assert (
        cohort_for_case("right_to_wrong", "wrong_entity_substitution")
        == "A_rw_substitution"
    )
    assert (
        cohort_for_case("right_to_wrong", "evasion_or_factual_denial")
        == "B_rw_nonsubstitution"
    )
    assert (
        cohort_for_case("right_to_wrong", "answer_dilution") == "B_rw_nonsubstitution"
    )
    assert cohort_for_case("right_to_wrong", "formal_refusal") == "B_rw_nonsubstitution"
    assert (
        cohort_for_case("wrong_to_right", "wrong_entity_substitution") == "C_wr_rescue"
    )
    assert cohort_for_case("wrong_to_right", "answer_dilution") == "C_wr_rescue"
    with pytest.raises(ValueError):
        cohort_for_case("stable_correct", "any")


def test_cohort_assignment_over_fixture_is_disjoint_and_exhaustive(
    tmp_path: Path,
) -> None:
    from score_bridge_margins import build_cohort_records

    adjudicated = [
        {
            "case_id": "c_a1",
            "question_id": "q_a1",
            "transition": "right_to_wrong",
            "label": "wrong_entity_substitution",
        },
        {
            "case_id": "c_a2",
            "question_id": "q_a2",
            "transition": "right_to_wrong",
            "label": "wrong_entity_substitution",
        },
        {
            "case_id": "c_b1",
            "question_id": "q_b1",
            "transition": "right_to_wrong",
            "label": "evasion_or_factual_denial",
        },
        {
            "case_id": "c_c1",
            "question_id": "q_c1",
            "transition": "wrong_to_right",
            "label": "wrong_entity_substitution",
        },
    ]
    queue_key = [
        {"case_id": "c_a1", "incorrect_response": "Berlin"},
        {"case_id": "c_a2", "incorrect_response": "Munich"},
        {"case_id": "c_b1", "incorrect_response": "I do not know"},
        {"case_id": "c_c1", "incorrect_response": "Paris"},
    ]
    question_pool = {
        "q_a1": {"question": "Q1?", "aliases": ["Alpha"], "alias_token_ids": [(1,)]},
        "q_a2": {"question": "Q2?", "aliases": ["Bravo"], "alias_token_ids": [(2,)]},
        "q_b1": {"question": "Q3?", "aliases": ["Charlie"], "alias_token_ids": [(3,)]},
        "q_c1": {"question": "Q4?", "aliases": ["Delta"], "alias_token_ids": [(4,)]},
    }

    records = build_cohort_records(
        adjudicated,
        queue_key,
        question_pool,
        StubTokenizer(),
        n_controls=3,
        seed=123,
    )

    cohorts = [r["cohort"] for r in records]
    # A=2, B=1, C=1, D=3
    assert cohorts.count("A_rw_substitution") == 2
    assert cohorts.count("B_rw_nonsubstitution") == 1
    assert cohorts.count("C_wr_rescue") == 1
    assert cohorts.count("D_random_control") == 3
    # Each adjudicated case appears exactly once, and no control leaks an
    # adjudicated case_id.
    adj_ids = {r["case_id"] for r in records if r["source"] == "adjudicated"}
    ctrl_ids = {r["case_id"] for r in records if r["source"] == "control"}
    assert adj_ids & ctrl_ids == set()
    assert len(adj_ids) == 4


def test_control_pool_excludes_self_question(tmp_path: Path) -> None:
    from score_bridge_margins import build_cohort_records

    adjudicated = [
        {
            "case_id": "c_a1",
            "question_id": "q_only",
            "transition": "right_to_wrong",
            "label": "wrong_entity_substitution",
        },
    ]
    queue_key = [{"case_id": "c_a1", "incorrect_response": "Berlin"}]
    question_pool = {
        "q_only": {"question": "Q?", "aliases": ["Alpha"], "alias_token_ids": [(1,)]},
        "q_other_a": {
            "question": "X?",
            "aliases": ["Omega"],
            "alias_token_ids": [(10,)],
        },
        "q_other_b": {
            "question": "Y?",
            "aliases": ["Sigma"],
            "alias_token_ids": [(20,)],
        },
    }

    records = build_cohort_records(
        adjudicated,
        queue_key,
        question_pool,
        StubTokenizer(),
        n_controls=5,
        seed=7,
    )
    controls = [r for r in records if r["cohort"] == "D_random_control"]
    assert len(controls) == 5
    for control in controls:
        # Wrong entity must come from a DIFFERENT question than the seed.
        assert control["wrong_entity_text"] not in {"Alpha"}
        # Seed case_id points back to an adjudicated R→W substitution case.
        assert control["seed_case_id"] == "c_a1"


def test_compute_shifts_matches_manual_arithmetic() -> None:
    from score_bridge_margins import compute_shifts, summarize_per_token

    # Gold preferred under baseline, less preferred under ITI.
    base_gold = summarize_per_token([-1.0, -2.0, -3.0, -5.0])  # first3 sum = -6.0
    base_wrong = summarize_per_token([-4.0, -5.0, -6.0, -1.0])  # first3 sum = -15.0
    iti_gold = summarize_per_token([-3.0, -4.0, -5.0, -5.0])  # first3 sum = -12.0
    iti_wrong = summarize_per_token([-2.0, -3.0, -4.0, -1.0])  # first3 sum = -9.0

    shifts = compute_shifts(base_gold, base_wrong, iti_gold, iti_wrong)
    # delta_base_f3 = -6 - (-15) = 9
    # delta_iti_f3  = -12 - (-9) = -3
    # shift_nats_f3 = -3 - 9 = -12
    assert shifts["first3"]["delta_base"] == pytest.approx(9.0)
    assert shifts["first3"]["delta_iti"] == pytest.approx(-3.0)
    assert shifts["first3"]["shift_nats"] == pytest.approx(-12.0)
    # Per-token normalization: gold_avg_base = -6/3 = -2, wrong_avg_base = -15/3 = -5
    # delta_base_per_token = -2 - (-5) = 3
    # gold_avg_iti = -12/3 = -4, wrong_avg_iti = -9/3 = -3, delta_iti_per_token = -1
    # shift_per_token = -1 - 3 = -4
    assert shifts["first3"]["shift_per_token"] == pytest.approx(-4.0)

    # Full window uses n=4 tokens each.
    # delta_base_full = -11 - (-16) = 5; delta_iti_full = -17 - (-10) = -7
    # shift_nats_full = -7 - 5 = -12
    assert shifts["full"]["shift_nats"] == pytest.approx(-12.0)


def test_unpaired_bootstrap_permutation_detects_separated_groups() -> None:
    from analyze_bridge_margins import unpaired_bootstrap_mean_difference

    rng = np.random.default_rng(0)
    a = rng.normal(loc=-2.0, scale=0.3, size=40)
    b = rng.normal(loc=0.0, scale=0.3, size=40)
    result = unpaired_bootstrap_mean_difference(
        a, b, n_resamples=2000, permutation_resamples=2000, seed=0
    )
    # A is more negative than B — estimate is negative and permutation rejects.
    assert result["estimate"] < -1.0
    assert result["permutation_test"]["p_value"] < 0.01
    # CI excludes 0.
    assert result["ci"]["upper"] < 0.0


def test_bootstrap_mean_ci_on_zero_mean_sample_contains_zero() -> None:
    from analyze_bridge_margins import bootstrap_mean_ci

    rng = np.random.default_rng(1)
    values = rng.normal(loc=0.0, scale=1.0, size=200)
    result = bootstrap_mean_ci(values, n_resamples=2000, seed=0)
    assert result["n"] == 200
    assert result["ci"]["lower"] < 0.0 < result["ci"]["upper"]


def test_pick_gold_alias_picks_highest_per_token_logp() -> None:
    from score_bridge_margins import pick_gold_alias

    # Two aliases: alias0 has high per-token avg, alias1 is long & low per-token.
    class FakeTokenizer:
        def __call__(self, text, **kwargs):
            return {"input_ids": list(range(len(text.split())))}

    tok = FakeTokenizer()

    # We stub score_continuations by monkey-patching the module.
    import score_bridge_margins as mod

    call_count = {"n": 0}

    def fake_score(
        model: torch.nn.Module,
        prompt_ids: torch.Tensor,
        targets: dict[str, tuple[int, ...]],
        *,
        scaler: Any,
    ) -> dict[str, list[float]]:
        del model, prompt_ids, targets, scaler
        call_count["n"] += 1
        # alias0 = "Alpha" (1 token, avg -0.5); alias1 = "A bravo name" (3 tokens, avg -2.0)
        return {
            "alias_0": [-0.5],
            "alias_1": [-2.0, -2.0, -2.0],
        }

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(mod, "score_continuations", fake_score)
    try:
        idx, text, ids, per_token = pick_gold_alias(
            ["Alpha", "A bravo name"],
            tok,
            baseline_prompt_ids=torch.tensor([1, 2, 3], dtype=torch.long),
            model=torch.nn.Linear(1, 1),
            scaler=None,
        )
    finally:
        monkeypatch.undo()
    assert idx == 0
    assert text == "Alpha"
    assert len(ids) == 1
    assert call_count["n"] == 1


def test_summarize_per_token_handles_empty_and_short() -> None:
    from score_bridge_margins import summarize_per_token

    empty = summarize_per_token([])
    assert empty["n_tokens"] == 0
    assert empty["sum_logprob"] == 0.0
    assert empty["sum_logprob_first3"] == 0.0

    short = summarize_per_token([-1.5, -2.5])
    assert short["n_tokens"] == 2
    assert short["n_tokens_first3"] == 2
    assert short["sum_logprob"] == pytest.approx(-4.0)
    assert short["sum_logprob_first3"] == pytest.approx(-4.0)

    longer = summarize_per_token([-1.0, -2.0, -3.0, -4.0])
    assert longer["n_tokens_first3"] == 3
    assert longer["sum_logprob_first3"] == pytest.approx(-6.0)
    assert longer["sum_logprob"] == pytest.approx(-10.0)


def test_build_provenance_serializes_path_args_and_relative_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from score_bridge_margins import ROOT, build_provenance

    monkeypatch.chdir(ROOT)

    args = Namespace(
        adjudicated=Path("data/input/adjudicated.jsonl"),
        queue_key=Path("data/input/queue_key.jsonl"),
        baseline_rows=Path("data/input/baseline.jsonl"),
        manifest=Path("data/input/manifest.json"),
        iti_artifact=Path("data/input/iti_heads.pt"),
        output_dir=Path("data/output"),
        model_path="google/gemma-3-4b-it",
    )
    provenance = build_provenance(
        args=args,
        output_path=Path("data/output/margins.jsonl"),
        iti_artifact_sha="abc123",
        n_records=7,
        n_skipped=2,
        elapsed_s=1.5,
    )

    assert provenance["args"]["adjudicated"] == "data/input/adjudicated.jsonl"
    assert provenance["args"]["output_dir"] == "data/output"
    assert provenance["output_path"] == "data/output/margins.jsonl"
    json.dumps(provenance)


def test_cohort_table_threads_cli_resample_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import analyze_bridge_margins as mod

    calls: list[tuple[int, int]] = []

    def fake_bootstrap_mean_ci(
        values: np.ndarray,
        *,
        n_resamples: int,
        seed: int,
        confidence: float = 0.95,
    ) -> dict[str, Any]:
        calls.append((n_resamples, seed))
        return {
            "estimate": float(np.mean(values)),
            "n": int(values.size),
            "ci": {
                "lower": 0.0,
                "upper": 0.0,
                "level": confidence,
                "method": "fake",
            },
            "bootstrap": {"n_resamples": n_resamples, "seed": seed},
        }

    monkeypatch.setattr(mod, "bootstrap_mean_ci", fake_bootstrap_mean_ci)

    by_cohort = {
        "A_rw_substitution": [
            {
                "cohort": "A_rw_substitution",
                "first3": {
                    "shift_nats": -1.0,
                    "shift_per_token": -0.5,
                    "delta_base": 1.0,
                    "delta_iti": -0.5,
                },
                "full": {
                    "shift_nats": -2.0,
                    "shift_per_token": -1.0,
                    "delta_base": 2.0,
                    "delta_iti": 0.0,
                },
            }
        ]
    }

    summary = mod.cohort_table(by_cohort, n_resamples=17, seed=9)

    assert summary["A_rw_substitution"]["first3"]["shift_nats"]["bootstrap"] == {
        "n_resamples": 17,
        "seed": 9,
    }
    assert calls == [(17, 9), (17, 9), (17, 9), (17, 9)]


def test_between_cohort_tests_threads_cli_resample_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import analyze_bridge_margins as mod

    calls: list[tuple[int, int, int]] = []

    def fake_unpaired_bootstrap_mean_difference(
        a: np.ndarray,
        b: np.ndarray,
        *,
        n_resamples: int,
        seed: int,
        permutation_resamples: int,
        confidence: float = 0.95,
    ) -> dict[str, Any]:
        calls.append((n_resamples, seed, permutation_resamples))
        return {
            "estimate": float(np.mean(a) - np.mean(b)),
            "n_a": int(a.size),
            "n_b": int(b.size),
            "ci": {
                "lower": 0.0,
                "upper": 0.0,
                "level": confidence,
                "method": "fake",
            },
            "bootstrap": {"n_resamples": n_resamples, "seed": seed},
            "permutation_test": {"n_permutations": permutation_resamples},
        }

    monkeypatch.setattr(
        mod,
        "unpaired_bootstrap_mean_difference",
        fake_unpaired_bootstrap_mean_difference,
    )

    by_cohort = {
        "A_rw_substitution": [
            {
                "cohort": "A_rw_substitution",
                "first3": {"shift_nats": -1.0, "shift_per_token": -0.5},
                "full": {"shift_nats": -2.0, "shift_per_token": -1.0},
            }
        ],
        "B_rw_nonsubstitution": [
            {
                "cohort": "B_rw_nonsubstitution",
                "first3": {"shift_nats": 0.5, "shift_per_token": 0.25},
                "full": {"shift_nats": 1.0, "shift_per_token": 0.5},
            }
        ],
    }

    results = mod.between_cohort_tests(by_cohort, n_resamples=23, seed=11)

    assert results["A_vs_B"]["first3"]["shift_nats"]["bootstrap"] == {
        "n_resamples": 23,
        "seed": 11,
    }
    assert calls == [(23, 11, 23), (23, 11, 23), (23, 11, 23), (23, 11, 23)]


def test_format_path_for_metadata_keeps_repo_relative_paths() -> None:
    from analyze_bridge_margins import ROOT
    from utils import format_path_for_metadata

    assert (
        format_path_for_metadata(ROOT / "data/bridge/results.json", root=ROOT)
        == "data/bridge/results.json"
    )
