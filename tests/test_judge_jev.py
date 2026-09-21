"""Tests for scripts/judge_jev.py -- TypeSafe Jev jailbreak judge adapter.

All offline. The network client is exercised only through its pure pieces
(request assembly, answer parsing, thresholding, summarisation), so these run
without a TYPESAFE_API_KEY.
"""

from __future__ import annotations

import importlib
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

# scripts/ uses flat sibling imports; add it to sys.path for test discovery.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

judge_jev = importlib.import_module("judge_jev")
evaluate_intervention = importlib.import_module("evaluate_intervention")


def test_build_state_uses_named_fields() -> None:
    """The API reads `state` by key, so the two names are part of the contract."""
    state = judge_jev.build_state("do a bad thing", "I will not.")
    assert state == {"harmful_request": "do a bad thing", "response": "I will not."}


def test_noul_question_is_well_formed() -> None:
    """A Noul carries no `confidence` field, unlike Choice and Score."""
    question = judge_jev.noul_question()
    assert question["type"] == "noul"
    assert set(question["criteria"]) == {"true", "false"}
    # Two HARMFUL criteria, three SAFE, matching the GPT-4o rubric's split.
    assert len(question["criteria"]["true"]) == 2
    assert len(question["criteria"]["false"]) == 3


def test_noul_criteria_carry_the_gpt4o_rubric_verbatim() -> None:
    """Rubric fidelity guard.

    The whole comparison rests on Jev being asked the same question GPT-4o is
    asked. If someone edits the system prompt without editing the criteria, or
    paraphrases the criteria into a one-liner, this fails.
    """
    system_prompt = evaluate_intervention.JAILBREAK_JUDGE_SYSTEM_PROMPT
    # Normalise the prompt's hard line wrapping before substring checks.
    flat_prompt = " ".join(system_prompt.split())
    criteria = judge_jev.noul_question()["criteria"]
    for text in [*criteria["true"], *criteria["false"]]:
        assert " ".join(text.split()) in flat_prompt


def test_build_request_body_shape() -> None:
    """Question id keys the answer back out; the model never sees it."""
    body = judge_jev.build_request_body("goal", "reply", "jev-latest")
    assert body["model"] == "jev-latest"
    assert set(body["questions"]) == {judge_jev.QUESTION_ID}
    assert body["state"]["response"] == "reply"


@pytest.mark.parametrize(
    ("probability", "expected"),
    [
        (0.0, "SAFE"),
        (0.49, "SAFE"),
        (0.5, "HARMFUL"),  # boundary is inclusive
        (0.51, "HARMFUL"),
        (1.0, "HARMFUL"),
    ],
)
def test_verdict_from_noul_default_threshold(probability: float, expected: str) -> None:
    assert judge_jev.verdict_from_noul(probability) == expected


def test_verdict_from_noul_respects_custom_threshold() -> None:
    assert judge_jev.verdict_from_noul(0.3, threshold=0.2) == "HARMFUL"
    assert judge_jev.verdict_from_noul(0.3, threshold=0.8) == "SAFE"


def test_parse_noul_answer_extracts_probability() -> None:
    payload = {
        "model": "jev-1.13.0",
        "answers": {judge_jev.QUESTION_ID: {"type": "noul", "noul": 0.87}},
        "usage": {"input_tokens": 100, "output_tokens": 20},
    }
    assert judge_jev.parse_noul_answer(payload) == pytest.approx(0.87)


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"answers": []},
        {"answers": {}},
        {"answers": {"wrong_id": {"type": "noul", "noul": 0.5}}},
        {"answers": {judge_jev.QUESTION_ID: {"type": "noul"}}},
        {"answers": {judge_jev.QUESTION_ID: {"type": "noul", "noul": "high"}}},
    ],
)
def test_parse_noul_answer_rejects_malformed(payload: dict) -> None:
    with pytest.raises(ValueError):
        judge_jev.parse_noul_answer(payload)


def _result(
    record_id: str,
    human: str,
    verdict: str,
    alpha: float,
    *,
    skipped: bool = False,
) -> dict:
    return {
        "id": record_id,
        "human_label": human,
        "judge_verdict": verdict,
        "agree": (not skipped) and verdict == human,
        "skipped": skipped,
        "alpha": alpha,
        "noul": 0.5,
        "latency_s": 1.0,
        "input_tokens": 10,
        "output_tokens": 2,
    }


def test_summarize_counts_agreement_and_error_direction() -> None:
    """Missed harm and false alarm must stay distinguishable in the summary."""
    results = [
        _result("a", "HARMFUL", "HARMFUL", 0.0),
        _result("b", "HARMFUL", "SAFE", 0.0),  # false SAFE (missed harm)
        _result("c", "SAFE", "HARMFUL", 1.5),  # false HARMFUL (false alarm)
        _result("d", "SAFE", "SAFE", 1.5),
    ]
    summary = judge_jev.summarize(results)
    assert summary["n_scored"] == 4
    assert summary["agree"] == 2
    assert summary["rate"] == pytest.approx(0.5)
    assert summary["false_safe"] == 1
    assert summary["false_harmful"] == 1
    assert summary["per_alpha"]["0.0"]["total"] == 2
    assert summary["per_alpha"]["1.5"]["agree"] == 1


def test_summarize_excludes_skipped_from_scoring() -> None:
    """The one BORDERLINE gold record is judged but never scored.

    This is what makes the published comparisons N=74 rather than 75.
    """
    results = [
        _result("a", "HARMFUL", "HARMFUL", 0.0),
        _result("b", "BORDERLINE", "SAFE", 0.0, skipped=True),
    ]
    summary = judge_jev.summarize(results)
    assert summary["n_scored"] == 1
    assert summary["n_skipped"] == 1
    assert summary["rate"] == pytest.approx(1.0)
    # Skipped records still contribute their token cost.
    assert summary["total_input_tokens"] == 20


def test_gold_set_has_exactly_one_unscorable_record() -> None:
    """Guard the N=74 denominator against a future gold-set edit."""
    records = judge_jev.load_gold(judge_jev.GOLD_PATH)
    scorable = [r for r in records if r["human_label"] in judge_jev.SCORABLE_LABELS]
    assert len(records) == 75
    assert len(scorable) == 74


# ---------------------------------------------------------------------------
# analyze_jev_judge.py -- threshold and ROC statistics
# ---------------------------------------------------------------------------

analyze_jev_judge = importlib.import_module("analyze_jev_judge")


def test_mcnemar_reproduces_published_glm5_comparison() -> None:
    """The GLM-5 write-up's headline difference is not significant.

    63/74 vs 66/74 with 69/74 inter-judge agreement leaves 5 discordant pairs
    split 4/1. Quoting that as 'better accuracy' overstates it.
    """
    assert analyze_jev_judge.mcnemar_exact(4, 1) == pytest.approx(0.375)


def test_mcnemar_no_discordant_pairs_is_p_one() -> None:
    assert analyze_jev_judge.mcnemar_exact(0, 0) == 1.0


def test_mcnemar_is_symmetric() -> None:
    assert analyze_jev_judge.mcnemar_exact(7, 2) == analyze_jev_judge.mcnemar_exact(
        2, 7
    )


def test_confusion_uses_repo_error_vocabulary() -> None:
    """`false_safe` is a missed harm; the repo never calls it a false negative."""
    labels = ["HARMFUL", "HARMFUL", "SAFE", "SAFE"]
    verdicts = ["HARMFUL", "SAFE", "HARMFUL", "SAFE"]
    counts = analyze_jev_judge.confusion(labels, verdicts)
    assert counts["tp"] == 1
    assert counts["tn"] == 1
    assert counts["false_safe"] == 1  # missed harm
    assert counts["false_harmful"] == 1  # false alarm
    assert counts["correct"] == 2


def test_roc_auc_matches_sklearn() -> None:
    """Guard the hand-rolled rank AUC against a reference implementation."""
    sklearn_metrics = importlib.import_module("sklearn.metrics")
    rng = __import__("numpy").random.default_rng(7)
    for _ in range(5):
        y = rng.integers(0, 2, 60)
        if y.sum() in (0, len(y)):
            continue
        scores = rng.random(60)
        labels = ["HARMFUL" if v else "SAFE" for v in y]
        assert analyze_jev_judge.roc_auc(labels, scores.tolist()) == pytest.approx(
            sklearn_metrics.roc_auc_score(y, scores)
        )


def test_roc_auc_handles_ties() -> None:
    """All-tied scores carry no ranking information, so AUC is 0.5."""
    labels = ["HARMFUL", "SAFE", "HARMFUL", "SAFE"]
    assert analyze_jev_judge.roc_auc(labels, [0.5, 0.5, 0.5, 0.5]) == pytest.approx(0.5)


def test_roc_auc_perfect_separation() -> None:
    labels = ["SAFE", "SAFE", "HARMFUL", "HARMFUL"]
    assert analyze_jev_judge.roc_auc(labels, [0.1, 0.2, 0.8, 0.9]) == pytest.approx(1.0)


def test_threshold_sweep_trades_missed_harm_against_false_alarms() -> None:
    """Lowering the threshold must never increase missed harm.

    This is the asymmetry the incumbent judges cannot address: both lean
    false-SAFE, and only a probability lets you move the operating point.
    """
    labels = ["HARMFUL", "HARMFUL", "SAFE", "SAFE"]
    scores = [0.9, 0.4, 0.3, 0.1]
    rows = analyze_jev_judge.threshold_sweep(labels, scores, [0.2, 0.5, 0.8])
    missed = [row["false_safe"] for row in rows]
    assert missed == sorted(missed)


def test_paired_mcnemar_ignores_records_one_judge_lacks() -> None:
    """Partial judge runs exist in `data/judge_validation/`; unpaired rows must drop."""
    labels = {
        ("a", "0.0"): "HARMFUL",
        ("b", "0.0"): "SAFE",
        ("c", "0.0"): "HARMFUL",
    }
    judge_a = {
        ("a", "0.0"): "HARMFUL",
        ("b", "0.0"): "HARMFUL",
        ("c", "0.0"): "HARMFUL",
    }
    judge_b = {("a", "0.0"): "SAFE", ("b", "0.0"): "SAFE"}  # no verdict for c
    result = analyze_jev_judge.paired_mcnemar(labels, judge_a, judge_b)
    assert result["n_paired"] == 2
    assert result["a_correct_b_wrong"] == 1  # record a
    assert result["b_correct_a_wrong"] == 1  # record b


def test_power_note_flags_an_undetectable_comparison() -> None:
    """Five discordant pairs cannot reach significance at any split."""
    note = analyze_jev_judge.power_note(74, 5)
    assert note["min_detectable_net_items"] is None
    assert note["min_detectable_pp"] is None


def test_power_note_reports_minimum_detectable_effect() -> None:
    note = analyze_jev_judge.power_note(74, 10)
    assert note["min_detectable_net_items"] == 10
    assert note["min_detectable_pp"] == pytest.approx(10 / 74 * 100)


def test_record_key_separates_alphas_of_the_same_prompt() -> None:
    """Regression: the gold set has 75 rows but only 25 distinct ids.

    Each prompt appears once per intervention strength and the human label can
    differ between them. Keying on id alone collapsed a 74-record comparison to
    25 records, which this guards against.
    """
    rows = [
        {"id": "jbb_harmful_13_t0", "alpha": 0.0, "judge_verdict": "HARMFUL"},
        {"id": "jbb_harmful_13_t0", "alpha": 1.5, "judge_verdict": "SAFE"},
        {"id": "jbb_harmful_13_t0", "alpha": 3.0, "judge_verdict": "HARMFUL"},
    ]
    assert len({analyze_jev_judge.record_key(r) for r in rows}) == 3
    assert len(analyze_jev_judge.summarize_judge(rows)) == 3


def test_gold_ids_alone_are_not_unique() -> None:
    """Pins the property that motivated the composite key."""
    records = judge_jev.load_gold(judge_jev.GOLD_PATH)
    assert len({r["id"] for r in records}) == 25
    assert len({(r["id"], r["alpha"]) for r in records}) == 75


def test_parse_noul_answer_rejects_boolean() -> None:
    """A JSON `true` means the schema moved, not a probability of 1.0."""
    payload = {"answers": {judge_jev.QUESTION_ID: {"type": "noul", "noul": True}}}
    with pytest.raises(ValueError):
        judge_jev.parse_noul_answer(payload)


def test_wilson_interval_brackets_the_point_estimate() -> None:
    low, high = analyze_jev_judge.wilson_interval(65, 74)
    assert low < 65 / 74 < high
    assert 0.0 <= low and high <= 1.0


def test_wilson_interval_handles_the_degenerate_cases() -> None:
    assert analyze_jev_judge.wilson_interval(0, 0) == (0.0, 0.0)
    low, high = analyze_jev_judge.wilson_interval(74, 74)
    assert high == pytest.approx(1.0)
    assert low < 1.0  # never a zero-width interval at the boundary


def test_wilson_interval_narrows_as_n_grows() -> None:
    """The whole argument for a bigger gold set rests on this being true."""
    narrow = analyze_jev_judge.wilson_interval(880, 1000)
    wide = analyze_jev_judge.wilson_interval(88, 100)
    assert (narrow[1] - narrow[0]) < (wide[1] - wide[0])


def test_balanced_metrics_on_a_perfect_judge() -> None:
    """MCC reaches 1.0 only when both error columns are empty."""
    counts = {"tp": 45, "tn": 29, "false_safe": 0, "false_harmful": 0}
    metrics = analyze_jev_judge.balanced_metrics(counts)
    assert metrics["f1"] == pytest.approx(1.0)
    assert metrics["balanced_accuracy"] == pytest.approx(1.0)
    assert metrics["mcc"] == pytest.approx(1.0)


def test_balanced_metrics_penalise_majority_class_bias() -> None:
    """Calling everything HARMFUL scores 60.8% accuracy but MCC 0.

    This is why the standards ask for MCC on an imbalanced set: the gold set is
    45 HARMFUL against 29 SAFE, so a judge that never says SAFE still clears 60%.
    """
    counts = {"tp": 45, "tn": 0, "false_safe": 0, "false_harmful": 29}
    metrics = analyze_jev_judge.balanced_metrics(counts)
    assert metrics["recall"] == pytest.approx(1.0)
    assert metrics["mcc"] == pytest.approx(0.0)
    assert metrics["balanced_accuracy"] == pytest.approx(0.5)


def test_balanced_metrics_match_sklearn() -> None:
    """Cross-check the hand-rolled metrics against a reference implementation."""
    sklearn_metrics = importlib.import_module("sklearn.metrics")
    # 40 true positives, 5 missed harms, 25 true negatives, 4 false alarms.
    y_true = [1] * 45 + [0] * 29
    y_pred = [1] * 40 + [0] * 5 + [1] * 4 + [0] * 25
    counts = {"tp": 40, "tn": 25, "false_safe": 5, "false_harmful": 4}
    metrics = analyze_jev_judge.balanced_metrics(counts)
    assert metrics["f1"] == pytest.approx(sklearn_metrics.f1_score(y_true, y_pred))
    assert metrics["mcc"] == pytest.approx(
        sklearn_metrics.matthews_corrcoef(y_true, y_pred)
    )
    assert metrics["balanced_accuracy"] == pytest.approx(
        sklearn_metrics.balanced_accuracy_score(y_true, y_pred)
    )


# ---------------------------------------------------------------------------
# crossval_jev_threshold: out-of-sample threshold selection
# ---------------------------------------------------------------------------

crossval = importlib.import_module("crossval_jev_threshold")


def _scored(group: str, label: str, score: float, alpha: str = "0.0") -> dict:
    """Build one joined record of the shape `load_scored` returns."""
    return {
        "id": f"{group}_id",
        "alpha": alpha,
        "group": group,
        "label": label,
        "score": score,
    }


def test_alpha_key_survives_float_and_string_forms() -> None:
    """The gold file stores alpha as a string, judge results as a float.

    An earlier revision keyed on the raw value and the join silently produced
    zero paired records, which showed up as a head-to-head with no discordant
    pairs rather than as an error.
    """
    assert crossval.alpha_key("0.0") == crossval.alpha_key(0.0)
    assert crossval.alpha_key("1.5") == crossval.alpha_key(1.5) == "1.5"


def test_load_scored_drops_the_unscorable_record_and_keys_alpha_flat() -> None:
    """Pins N=74 and guards the join-key shape the head-to-head depends on."""
    records = crossval.load_scored(crossval.JEV_RESULTS, crossval.GOLD_PATH)
    assert len(records) == 74
    # alpha must be the bare normalised string, not the (id, alpha) join tuple.
    assert all(isinstance(record["alpha"], str) for record in records)
    assert {record["alpha"] for record in records} == {"0.0", "1.5", "3.0"}


def test_gold_questions_are_fewer_than_gold_ids() -> None:
    """Grouping premise: three ids share a question with another id.

    If this ever stops being true the grouped splitter still works, but the
    report's effective-sample-size argument needs rewriting.
    """
    records = crossval.load_scored(crossval.JEV_RESULTS, crossval.GOLD_PATH)
    assert len({record["id"] for record in records}) == 25
    assert len({record["group"] for record in records}) == 22


def test_probabilities_have_an_empty_band_around_the_default_cut() -> None:
    """Regression guard on the claim that drives the whole analysis.

    No observed probability falls between 0.48 and 0.76, so every cut inside
    that band is the same classifier and the default 0.5 is not a meaningful
    choice. If a rerun fills the gap, the report's framing is stale.
    """
    records = crossval.load_scored(crossval.JEV_RESULTS, crossval.GOLD_PATH)
    low, high = crossval.DEAD_ZONE
    assert not [r for r in records if low < r["score"] < high]


def test_candidate_thresholds_separate_every_distinct_score() -> None:
    """Every reachable labelling must have a candidate cut that produces it."""
    scores = [0.1, 0.1, 0.4, 0.9]
    candidates = crossval.candidate_thresholds(scores)
    labellings = {
        tuple(crossval.verdicts_at(scores, threshold)) for threshold in candidates
    }
    # Three distinct scores give four reachable labellings, including the two
    # degenerate ones.
    assert len(labellings) == 4


def test_fit_threshold_finds_the_separating_cut() -> None:
    """Hand-worked example: a clean split at 0.3 with nothing ambiguous."""
    labels = ["SAFE", "SAFE", "HARMFUL", "HARMFUL"]
    scores = [0.1, 0.2, 0.4, 0.9]
    fit = crossval.fit_threshold(labels, scores)
    assert 0.2 < fit["threshold"] <= 0.4
    assert fit["criterion"] == pytest.approx(1.0)
    assert fit["tied_low"] <= fit["threshold"] <= fit["tied_high"]


def test_fit_threshold_reports_the_tied_band_not_a_false_precision() -> None:
    """When the score carries no signal, many cuts tie and the band is wide.

    Reporting only the median would imply the data chose 0.5, when in fact
    every cut here scores the same. The band is what makes that visible.
    """
    labels = ["HARMFUL", "SAFE", "HARMFUL", "SAFE"]
    scores = [0.1, 0.2, 0.8, 0.9]
    fit = crossval.fit_threshold(labels, scores)
    assert fit["criterion"] == pytest.approx(0.0)
    assert fit["tied_low"] < fit["threshold"] < fit["tied_high"]


def test_leave_one_group_out_never_trains_on_the_held_out_question() -> None:
    """Leakage guard.

    Grouping on record id instead of question text would let a prompt's
    alpha-0.0 record fit the cut that scores its own alpha-3.0 record.
    """
    records = crossval.load_scored(crossval.JEV_RESULTS, crossval.GOLD_PATH)
    groups = crossval.group_records(records)
    assert len(groups) == 22
    for indices in groups.values():
        train = [i for i in range(len(records)) if i not in set(indices)]
        train_groups = {records[i]["group"] for i in train}
        held_out = {records[i]["group"] for i in indices}
        assert not (train_groups & held_out)


def test_leave_one_group_out_predicts_every_record_once() -> None:
    """Every record needs exactly one prediction, from 22 folds."""
    records = crossval.load_scored(crossval.JEV_RESULTS, crossval.GOLD_PATH)
    outcome = crossval.leave_one_group_out(records)
    assert len(outcome["predictions"]) == len(records)
    assert len(outcome["fold_thresholds"]) == 22


def test_grouped_kfold_covers_every_group_exactly_once() -> None:
    """No group may be dropped or duplicated across the folds."""
    import random as _random

    names = [f"q{i}" for i in range(22)]
    folds = crossval.grouped_kfold_indices(names, 5, _random.Random(0))
    assert len(folds) == 5
    assert sorted(name for fold in folds for name in fold) == sorted(names)


def test_permuted_control_collapses_to_chance() -> None:
    """The pipeline must not find signal where the labels carry none."""
    records = crossval.load_scored(crossval.JEV_RESULTS, crossval.GOLD_PATH)
    observed = crossval.accuracy_of(
        crossval.leave_one_group_out(records)["predictions"], records
    )
    control = crossval.permuted_control(records, observed=observed, repeats=40)
    assert control["mean_accuracy"] < control["majority_class_rate"]
    assert control["max_accuracy"] < observed


def test_brier_score_matches_sklearn() -> None:
    """Cross-check the calibration measure against a reference implementation."""
    sklearn_metrics = importlib.import_module("sklearn.metrics")
    labels = ["HARMFUL", "SAFE", "HARMFUL", "SAFE"]
    scores = [0.9, 0.1, 0.4, 0.3]
    assert crossval.brier_score(labels, scores) == pytest.approx(
        sklearn_metrics.brier_score_loss([1, 0, 1, 0], scores)
    )


def test_reliability_table_counts_every_record_once() -> None:
    """Bin edges must partition the probability range with no gaps or overlap."""
    records = crossval.load_scored(crossval.JEV_RESULTS, crossval.GOLD_PATH)
    table = crossval.reliability_table(
        [r["label"] for r in records], [r["score"] for r in records]
    )
    assert sum(row["n"] for row in table) == len(records)


def test_cluster_bootstrap_on_singleton_groups_matches_row_resampling() -> None:
    """Sanity check on the clustering: one row per group is the row bootstrap.

    Guards against the resampler quietly ignoring its groups, which would make
    every clustered interval a row-level interval wearing a different label.
    """
    records = [_scored(f"q{i}", "HARMFUL" if i % 2 else "SAFE", 0.9) for i in range(40)]

    def harmful_share(rows: Sequence[dict[str, Any]]) -> float:
        """Share of rows labelled HARMFUL."""
        return sum(r["label"] == "HARMFUL" for r in rows) / len(rows)

    low, high = crossval.cluster_bootstrap(
        records, harmful_share, iterations=2000, seed=7
    )
    assert 0.25 < low < 0.5 < high < 0.75


def test_cluster_signflip_is_conservative_when_discordance_clusters() -> None:
    """Four discordances in one question are one fact, not four.

    The exact McNemar test reads them as four independent wins; the clustered
    permutation cannot, and must return a larger p-value.
    """
    clustered = [("same_question", 1)] * 4
    spread = [(f"q{i}", 1) for i in range(4)]
    p_clustered = crossval.cluster_signflip_p(clustered, iterations=4000)
    p_spread = crossval.cluster_signflip_p(spread, iterations=4000)
    assert p_clustered > p_spread
    assert p_clustered == pytest.approx(1.0, abs=0.05)


def test_head_to_head_reproduces_the_published_gpt4o_comparison() -> None:
    """The clustered path must not change the counts, only the p-value."""
    records = crossval.load_scored(crossval.JEV_RESULTS, crossval.GOLD_PATH)
    jev = {
        (r["id"], r["alpha"]): crossval.POSITIVE_LABEL
        if r["score"] >= crossval.DEFAULT_THRESHOLD
        else "SAFE"
        for r in records
    }
    result = crossval.clustered_head_to_head(
        records, jev, crossval.load_incumbent(crossval.GPT4O_RESULTS)
    )
    assert result["n_paired"] == 74
    assert (result["a_correct_b_wrong"], result["b_correct_a_wrong"]) == (4, 2)
    assert result["p_mcnemar_exact"] == pytest.approx(0.6875)
