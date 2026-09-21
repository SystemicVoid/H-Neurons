"""Tests for scripts/judge_jev.py -- TypeSafe Jev jailbreak judge adapter.

All offline. The network client is exercised only through its pure pieces
(request assembly, answer parsing, thresholding, summarisation), so these run
without a TYPESAFE_API_KEY.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

# scripts/ uses flat sibling imports; add it to sys.path for test discovery.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

judge_jev = importlib.import_module("judge_jev")
evaluate_intervention = importlib.import_module("evaluate_intervention")


def test_build_state_uses_named_fields() -> None:
    state = judge_jev.build_state("do a bad thing", "I will not.")
    assert state == {"harmful_request": "do a bad thing", "response": "I will not."}


def test_noul_question_is_well_formed() -> None:
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
    narrow = analyze_jev_judge.wilson_interval(880, 1000)
    wide = analyze_jev_judge.wilson_interval(88, 100)
    assert (narrow[1] - narrow[0]) < (wide[1] - wide[0])


def test_balanced_metrics_on_a_perfect_judge() -> None:
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
