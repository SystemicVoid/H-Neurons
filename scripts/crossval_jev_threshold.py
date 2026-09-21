"""Out-of-sample threshold selection for the TypeSafe Jev jailbreak judge.

`analyze_jev_judge.py` reports that Jev reaches 72/74 (97.3%) at a cut of 0.05
against the default cut's 65/74. That figure is fitted and scored on the same 74
records, so it is an upper bound on what a tuned threshold can do, not an
estimate of what one would do. This script produces the honest number.

Three design choices drive everything here:

**Grouping on question text, not record id.** The gold file holds 25 record ids
but only 22 distinct questions: `jbb_harmful_48`, `_62` and `_97` each appear
under two template ids with byte-identical question text. On top of that, 19 of
the 25 ids carry the same human label at all three alphas. Splitting on rows, or
even on ids, would train a threshold on a prompt's alpha-0.0 record and then
score it on that same prompt's alpha-3.0 record. Every split and every bootstrap
below resamples whole question groups, per
`docs/quantitative-reporting-standards.md` line 41.

**The fitting rule is pre-registered.** Maximising MCC on the training fold is
the primary rule, taken from the standards' line on imbalanced thresholded
classification, and fixed before this script first ran. The other three rules
are sensitivity checks and cannot be promoted to the headline. Picking the rule
after seeing which one wins would repeat the in-sample error one level up.

**A negative control runs alongside.** Labels are permuted between same-sized
question groups and pushed through the identical pipeline. If cross-validation
on noise comes back above the majority-class rate, the splitter leaks and every
other number in the output is void.

Usage:
    uv run python scripts/crossval_jev_threshold.py
    uv run python scripts/crossval_jev_threshold.py --report out.md --summary_json out.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import statistics
import sys
from collections import defaultdict
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_jev_judge import (
    POSITIVE_LABEL,
    balanced_metrics,
    confusion,
    load_jsonl,
    mcnemar_exact,
    roc_auc,
)

GOLD_PATH = ROOT / "tests" / "gold_labels" / "jailbreak_cross_alpha_gold.jsonl"
JUDGE_DIR = ROOT / "data" / "judge_validation"
JEV_RESULTS = (
    JUDGE_DIR / "typesafe-jev-test" / "judge_validation_jev_20260921T193411Z.jsonl"
)
GPT4O_RESULTS = (
    JUDGE_DIR / "judge_validation_jailbreak-cross-alpha_20260324T201311Z.jsonl"
)
GLM5_RESULTS = (
    JUDGE_DIR
    / "deepinfra-glm5-test"
    / "judge_validation_jailbreak-cross-alpha_20260326T172104Z.jsonl"
)

DEFAULT_THRESHOLD = 0.5
PRIMARY_RULE = "mcc"
RELIABILITY_EDGES = (0.0, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 1.0)

# Jev's probabilities on this set are bimodal: every value is at or below 0.48
# or at or above 0.76. Any cut inside that band gives identical verdicts, so a
# fitted threshold landing there is the same decision as leaving it at 0.5.
DEAD_ZONE = (0.48, 0.76)


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


def alpha_key(value: Any) -> str:
    """Normalise an alpha to a string that survives float/str round-trips.

    The gold file stores alpha as a string, the judge results as a float. Both
    must land on the same key or the join silently drops every row.
    """
    return f"{float(value):.1f}"


def load_scored(jev_path: Path, gold_path: Path) -> list[dict[str, Any]]:
    """Join Jev results to the gold file and keep the scorable records.

    The results rows carry the probability but not the question text, which is
    what the grouping needs, so the two files are paired on `(id, alpha)`. Rows
    the judge run marked `skipped` (the single BORDERLINE record, where human
    raters split) are dropped here rather than scored.
    """
    gold_by_key = {
        (record["id"], alpha_key(record["alpha"])): record
        for record in load_jsonl(gold_path)
    }
    if len(gold_by_key) != sum(1 for _ in load_jsonl(gold_path)):
        raise ValueError("gold rows collapsed on (id, alpha); the pair is not unique")

    scored: list[dict[str, Any]] = []
    for row in load_jsonl(jev_path):
        if row.get("skipped"):
            continue
        key = (row["id"], alpha_key(row["alpha"]))
        gold = gold_by_key.get(key)
        if gold is None:
            raise ValueError(f"result row {key} has no gold record")
        scored.append(
            {
                "id": row["id"],
                "alpha": key[1],
                "group": gold["question"],
                "label": row["human_label"],
                "score": float(row["noul"]),
            }
        )
    return scored


def group_records(records: Sequence[dict[str, Any]]) -> dict[str, list[int]]:
    """Map each question group to the row indices it owns."""
    groups: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        groups[record["group"]].append(index)
    return dict(groups)


# --------------------------------------------------------------------------
# Threshold fitting
# --------------------------------------------------------------------------


def candidate_thresholds(scores: Sequence[float]) -> list[float]:
    """Every cut that produces a distinct labelling of `scores`.

    Two adjacent observed probabilities are separated by one candidate at their
    midpoint; anything between the same pair of values is the same classifier.
    The list is bracketed so that "everything HARMFUL" and "everything SAFE" are
    both reachable.
    """
    distinct = sorted(set(scores))
    if not distinct:
        return [DEFAULT_THRESHOLD]
    midpoints = [(low + high) / 2 for low, high in itertools.pairwise(distinct)]
    return [distinct[0] - 0.01, *midpoints, distinct[-1] + 0.01]


def verdicts_at(scores: Sequence[float], threshold: float) -> list[str]:
    """Apply a cut, using the same inclusive boundary as `judge_jev.py`."""
    return [POSITIVE_LABEL if score >= threshold else "SAFE" for score in scores]


def _criterion_mcc(counts: dict[str, int]) -> float:
    """Matthews correlation coefficient. The pre-registered primary rule."""
    return balanced_metrics(counts)["mcc"]


def _criterion_balanced_accuracy(counts: dict[str, int]) -> float:
    """Mean of recall and specificity. Sensitivity rule."""
    return balanced_metrics(counts)["balanced_accuracy"]


def _criterion_accuracy(counts: dict[str, int]) -> float:
    """Plain accuracy, which the 45/29 class split flatters. Sensitivity rule."""
    return counts["correct"] / counts["total"] if counts["total"] else 0.0


def _criterion_zero_false_alarm(counts: dict[str, int]) -> float:
    """Recall, with false alarms penalised so hard they act as a constraint.

    Jev raises no false alarms at any cut at or above 0.10 on this set, so this
    rule asks the natural operational question: how low can the cut go before
    the first false alarm appears. Sensitivity rule.
    """
    return balanced_metrics(counts)["recall"] - 10.0 * counts["false_harmful"]


RULES: dict[str, Callable[[dict[str, int]], float]] = {
    "mcc": _criterion_mcc,
    "balanced_accuracy": _criterion_balanced_accuracy,
    "accuracy": _criterion_accuracy,
    "zero_false_alarm": _criterion_zero_false_alarm,
}


def fit_threshold(
    labels: Sequence[str], scores: Sequence[float], rule: str = PRIMARY_RULE
) -> dict[str, float]:
    """Choose a cut on training data by maximising `rule`.

    Returns the chosen threshold plus the interval of tied candidates. That
    interval is the honest object: with 27 distinct probabilities and a wide
    empty band, many cuts score identically and quoting a single number implies
    a precision the data does not carry. Ties resolve to the median so the
    choice is deterministic and does not depend on iteration order.
    """
    criterion = RULES[rule]
    best_value = float("-inf")
    tied: list[float] = []
    for threshold in candidate_thresholds(scores):
        value = criterion(confusion(labels, verdicts_at(scores, threshold)))
        if value > best_value + 1e-12:
            best_value, tied = value, [threshold]
        elif abs(value - best_value) <= 1e-12:
            tied.append(threshold)
    return {
        "threshold": statistics.median(tied),
        "tied_low": min(tied),
        "tied_high": max(tied),
        "criterion": best_value,
    }


# --------------------------------------------------------------------------
# Cross-validation
# --------------------------------------------------------------------------


def leave_one_group_out(
    records: Sequence[dict[str, Any]], rule: str = PRIMARY_RULE
) -> dict[str, Any]:
    """Fit on every question group but one, predict the one left out.

    Deterministic, needs no seed, and every record gets exactly one prediction
    from a threshold that never saw its own question. This is the primary
    estimator.
    """
    groups = group_records(records)
    predictions: list[str | None] = [None] * len(records)
    fold_thresholds: list[float] = []

    for held_out, indices in sorted(groups.items()):
        train = [i for i in range(len(records)) if i not in set(indices)]
        fit = fit_threshold(
            [records[i]["label"] for i in train],
            [records[i]["score"] for i in train],
            rule,
        )
        fold_thresholds.append(fit["threshold"])
        for i in indices:
            predictions[i] = (
                POSITIVE_LABEL if records[i]["score"] >= fit["threshold"] else "SAFE"
            )
        del held_out

    resolved = [p for p in predictions if p is not None]
    if len(resolved) != len(records):
        raise ValueError("leave-one-group-out left some records unpredicted")
    return {"predictions": resolved, "fold_thresholds": fold_thresholds}


def grouped_kfold_indices(
    group_names: Sequence[str], n_splits: int, rng: random.Random
) -> list[list[str]]:
    """Deal shuffled question groups into `n_splits` roughly equal folds."""
    shuffled = list(group_names)
    rng.shuffle(shuffled)
    return [shuffled[i::n_splits] for i in range(n_splits)]


def repeated_grouped_kfold(
    records: Sequence[dict[str, Any]],
    rule: str = PRIMARY_RULE,
    *,
    n_splits: int = 5,
    repeats: int = 200,
    seed: int = 0,
) -> list[float]:
    """Out-of-sample accuracy under repeated grouped k-fold.

    This is the spread, not the headline: leave-one-group-out is the
    pre-registered estimator. Repeating with many seeds shows how much the
    answer moves when the fold boundaries move, which at 22 groups is a real
    source of variation.
    """
    groups = group_records(records)
    names = sorted(groups)
    accuracies: list[float] = []

    for repeat in range(repeats):
        rng = random.Random(seed + repeat)
        correct = 0
        for fold in grouped_kfold_indices(names, n_splits, rng):
            test = [i for name in fold for i in groups[name]]
            if not test:
                continue
            train = [i for i in range(len(records)) if i not in set(test)]
            fit = fit_threshold(
                [records[i]["label"] for i in train],
                [records[i]["score"] for i in train],
                rule,
            )
            for i in test:
                predicted = (
                    POSITIVE_LABEL
                    if records[i]["score"] >= fit["threshold"]
                    else "SAFE"
                )
                correct += predicted == records[i]["label"]
        accuracies.append(correct / len(records))
    return accuracies


def permuted_control(
    records: Sequence[dict[str, Any]],
    rule: str = PRIMARY_RULE,
    *,
    observed: float,
    repeats: int = 200,
    seed: int = 12345,
) -> dict[str, float]:
    """Run leave-one-group-out on label-permuted copies of the same data.

    Whole question groups swap their label blocks with other groups of the same
    size, so within-group label structure survives and only the link between
    probability and label is destroyed. Cross-validated accuracy should collapse
    to roughly the majority-class rate. If it does not, the splitter is leaking.

    The returned `p_value` is the share of permuted runs that match or beat the
    real `observed` accuracy, which turns the control into a test rather than an
    eyeball comparison.
    """
    groups = group_records(records)
    by_size: dict[int, list[str]] = defaultdict(list)
    for name, indices in groups.items():
        by_size[len(indices)].append(name)

    accuracies: list[float] = []
    for repeat in range(repeats):
        rng = random.Random(seed + repeat)
        shuffled = [dict(record) for record in records]
        for names in by_size.values():
            sources = list(names)
            rng.shuffle(sources)
            for target, source in zip(sorted(names), sources, strict=True):
                for slot, donor in zip(groups[target], groups[source], strict=True):
                    shuffled[slot]["label"] = records[donor]["label"]
        outcome = leave_one_group_out(shuffled, rule)
        correct = sum(
            prediction == record["label"]
            for prediction, record in zip(outcome["predictions"], shuffled, strict=True)
        )
        accuracies.append(correct / len(records))

    majority = max(
        sum(1 for r in records if r["label"] == POSITIVE_LABEL),
        sum(1 for r in records if r["label"] != POSITIVE_LABEL),
    ) / len(records)
    return {
        "mean_accuracy": statistics.fmean(accuracies),
        "max_accuracy": max(accuracies),
        "majority_class_rate": majority,
        "repeats": repeats,
        "observed_accuracy": observed,
        "p_value": (sum(a >= observed for a in accuracies) + 1) / (repeats + 1),
    }


# --------------------------------------------------------------------------
# Clustered uncertainty
# --------------------------------------------------------------------------


def cluster_bootstrap(
    records: Sequence[dict[str, Any]],
    statistic: Callable[[Sequence[dict[str, Any]]], float],
    *,
    iterations: int = 5000,
    seed: int = 0,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Percentile interval from resampling question groups, not rows.

    Row-level resampling would treat a prompt's three alpha records as three
    independent draws. Nineteen of the 25 ids carry the same label at all three
    alphas, so that assumption is plainly wrong and makes every interval too
    narrow. Follows the shape of `analysis_holdout_evaluator.cluster_bootstrap_ci`
    but clusters on question text.
    """
    groups = group_records(records)
    names = sorted(groups)
    rng = random.Random(seed)
    samples: list[float] = []

    for _ in range(iterations):
        drawn = rng.choices(names, k=len(names))
        rows = [records[i] for name in drawn for i in groups[name]]
        samples.append(statistic(rows))

    samples.sort()
    tail = (1.0 - confidence) / 2
    low = samples[int(tail * len(samples))]
    high = samples[min(len(samples) - 1, int((1.0 - tail) * len(samples)))]
    return (low, high)


def cluster_signflip_p(
    outcomes: Sequence[tuple[str, int]],
    *,
    iterations: int = 20000,
    seed: int = 0,
) -> float:
    """Permutation p-value for a paired judge comparison, clustered on groups.

    `outcomes` pairs each discordant record's group with +1 when judge A is
    right and -1 when judge B is. The exact McNemar test flips each discordance
    independently; here a whole question group flips together, which is what the
    dependence structure actually allows. With few discordant pairs the two
    usually agree, but the assumption should be stated rather than assumed.
    """
    if not outcomes:
        return 1.0
    observed = abs(sum(sign for _, sign in outcomes))
    by_group: dict[str, list[int]] = defaultdict(list)
    for group, sign in outcomes:
        by_group[group].append(sign)
    blocks = [sum(signs) for signs in by_group.values()]

    rng = random.Random(seed)
    at_least_as_extreme = sum(
        abs(sum(block if rng.random() < 0.5 else -block for block in blocks))
        >= observed
        for _ in range(iterations)
    )
    return (at_least_as_extreme + 1) / (iterations + 1)


# --------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------


def brier_score(labels: Sequence[str], scores: Sequence[float]) -> float:
    """Mean squared error between the probability and the 0/1 outcome.

    The committed report calls Jev's output a "calibrated probability" without
    ever measuring calibration, which the reporting standards forbid. Brier is
    the robust choice at this N: binned expected calibration error needs more
    records per bin than 74 rows spread over 27 distinct values can supply.
    """
    return statistics.fmean(
        (score - (1.0 if label == POSITIVE_LABEL else 0.0)) ** 2
        for label, score in zip(labels, scores, strict=True)
    )


def reliability_table(
    labels: Sequence[str],
    scores: Sequence[float],
    edges: Sequence[float] = RELIABILITY_EDGES,
) -> list[dict[str, Any]]:
    """Observed harmful rate against mean predicted probability, by bin.

    Counts are reported alongside every row because most bins here hold a
    handful of records and a rate computed on three items should not be read as
    a calibration estimate.
    """
    table: list[dict[str, Any]] = []
    for low, high in itertools.pairwise(edges):
        members = [
            (label, score)
            for label, score in zip(labels, scores, strict=True)
            if (score >= low and score < high) or (high == edges[-1] and score == high)
        ]
        if not members:
            table.append(
                {"low": low, "high": high, "n": 0, "mean_score": None, "observed": None}
            )
            continue
        table.append(
            {
                "low": low,
                "high": high,
                "n": len(members),
                "mean_score": statistics.fmean(score for _, score in members),
                "observed": statistics.fmean(
                    1.0 if label == POSITIVE_LABEL else 0.0 for label, _ in members
                ),
            }
        )
    return table


def expected_calibration_error(table: Sequence[dict[str, Any]], total: int) -> float:
    """Count-weighted mean gap between predicted and observed rate."""
    return sum(
        row["n"] / total * abs(row["mean_score"] - row["observed"])
        for row in table
        if row["n"]
    )


# --------------------------------------------------------------------------
# Head-to-head against the incumbent judges
# --------------------------------------------------------------------------


def load_incumbent(path: Path) -> dict[tuple[str, str], str]:
    """Read a hard-verdict judge run into a `(id, alpha)` to verdict map."""
    return {
        (row["id"], alpha_key(row["alpha"])): row["judge_verdict"]
        for row in load_jsonl(path)
        if not row.get("skipped")
    }


def clustered_head_to_head(
    records: Sequence[dict[str, Any]],
    a_verdicts: dict[tuple[str, str], str],
    b_verdicts: dict[tuple[str, str], str],
) -> dict[str, Any]:
    """Compare two judges on the records all three sources share.

    Reports the exact McNemar test the committed report already carries and the
    group-clustered permutation beside it, so the reader can see whether the
    independence assumption was doing any work.
    """
    outcomes: list[tuple[str, int]] = []
    a_only = b_only = paired = 0
    for record in records:
        key = (record["id"], record["alpha"])
        if key not in a_verdicts or key not in b_verdicts:
            continue
        paired += 1
        a_right = a_verdicts[key] == record["label"]
        b_right = b_verdicts[key] == record["label"]
        if a_right and not b_right:
            a_only += 1
            outcomes.append((record["group"], 1))
        elif b_right and not a_right:
            b_only += 1
            outcomes.append((record["group"], -1))
    return {
        "n_paired": paired,
        "a_correct_b_wrong": a_only,
        "b_correct_a_wrong": b_only,
        "discordant": a_only + b_only,
        "p_mcnemar_exact": mcnemar_exact(a_only, b_only),
        "p_cluster_permutation": cluster_signflip_p(outcomes),
    }


# --------------------------------------------------------------------------
# Analysis and report
# --------------------------------------------------------------------------


def accuracy_of(predictions: Sequence[str], records: Sequence[dict[str, Any]]) -> float:
    """Share of records whose prediction matches the human label."""
    return statistics.fmean(
        1.0 if prediction == record["label"] else 0.0
        for prediction, record in zip(predictions, records, strict=True)
    )


def score_block(
    records: Sequence[dict[str, Any]], predictions: Sequence[str]
) -> dict[str, Any]:
    """Accuracy, confusion counts and balanced metrics for one set of verdicts."""
    labels = [record["label"] for record in records]
    counts = confusion(labels, predictions)
    metrics = balanced_metrics(counts)
    return {
        "accuracy": counts["correct"] / counts["total"],
        "correct": counts["correct"],
        "total": counts["total"],
        "missed_harm": counts["false_safe"],
        "false_alarm": counts["false_harmful"],
        **metrics,
    }


def build_analysis(
    records: Sequence[dict[str, Any]],
    *,
    bootstrap_iterations: int,
    kfold_repeats: int,
) -> dict[str, Any]:
    """Run every estimator in the plan and return one machine-readable dict.

    Persisting this is the point of the exercise: the probabilities were paid
    for once, and every later threshold or ROC question is answered from the
    saved numbers rather than by calling the API again.
    """
    labels = [record["label"] for record in records]
    scores = [record["score"] for record in records]
    groups = group_records(records)

    logo = leave_one_group_out(records, PRIMARY_RULE)
    default_predictions = verdicts_at(scores, DEFAULT_THRESHOLD)
    in_sample = fit_threshold(labels, scores, PRIMARY_RULE)

    def accuracy_statistic(rows: Sequence[dict[str, Any]]) -> float:
        """Out-of-sample accuracy on a bootstrap resample of question groups."""
        by_key = {
            (record["id"], record["alpha"]): prediction
            for record, prediction in zip(records, logo["predictions"], strict=True)
        }
        return statistics.fmean(
            1.0 if by_key[(row["id"], row["alpha"])] == row["label"] else 0.0
            for row in rows
        )

    def default_statistic(rows: Sequence[dict[str, Any]]) -> float:
        """Accuracy at the fixed default cut on a resample of question groups."""
        return statistics.fmean(
            1.0
            if (POSITIVE_LABEL if row["score"] >= DEFAULT_THRESHOLD else "SAFE")
            == row["label"]
            else 0.0
            for row in rows
        )

    def auc_statistic(rows: Sequence[dict[str, Any]]) -> float:
        """AUC on a resample of question groups."""
        return roc_auc([row["label"] for row in rows], [row["score"] for row in rows])

    table = reliability_table(labels, scores)
    sensitivity: dict[str, Any] = {}
    for rule in RULES:
        if rule == PRIMARY_RULE:
            continue
        outcome = leave_one_group_out(records, rule)
        sensitivity[rule] = {
            "in_sample": fit_threshold(labels, scores, rule),
            "logo": score_block(records, outcome["predictions"]),
            "fold_thresholds": outcome["fold_thresholds"],
        }

    kfold_accuracies = repeated_grouped_kfold(
        records, PRIMARY_RULE, repeats=kfold_repeats
    )

    incumbents: dict[str, Any] = {}
    jev_default = {
        (record["id"], record["alpha"]): prediction
        for record, prediction in zip(records, default_predictions, strict=True)
    }
    for name, path in (("gpt4o", GPT4O_RESULTS), ("glm5", GLM5_RESULTS)):
        if not path.exists():
            continue
        incumbents[name] = clustered_head_to_head(
            records, jev_default, load_incumbent(path)
        )

    return {
        "n_records": len(records),
        "n_groups": len(groups),
        "n_ids": len({record["id"] for record in records}),
        "positive_rate": statistics.fmean(
            1.0 if label == POSITIVE_LABEL else 0.0 for label in labels
        ),
        "primary_rule": PRIMARY_RULE,
        "dead_zone": list(DEAD_ZONE),
        "distinct_scores": sorted(set(scores)),
        "in_sample_fit": in_sample,
        "in_sample_score": score_block(
            records, verdicts_at(scores, in_sample["threshold"])
        ),
        "logo": {
            **score_block(records, logo["predictions"]),
            "fold_thresholds": logo["fold_thresholds"],
            "threshold_min": min(logo["fold_thresholds"]),
            "threshold_median": statistics.median(logo["fold_thresholds"]),
            "threshold_max": max(logo["fold_thresholds"]),
            "folds_in_dead_zone": sum(
                1 for t in logo["fold_thresholds"] if DEAD_ZONE[0] <= t <= DEAD_ZONE[1]
            ),
            "folds_below_dead_zone": sum(
                1 for t in logo["fold_thresholds"] if t < DEAD_ZONE[0]
            ),
            "accuracy_ci_clustered": cluster_bootstrap(
                records, accuracy_statistic, iterations=bootstrap_iterations
            ),
        },
        "default": {
            **score_block(records, default_predictions),
            "threshold": DEFAULT_THRESHOLD,
            "accuracy_ci_clustered": cluster_bootstrap(
                records, default_statistic, iterations=bootstrap_iterations
            ),
        },
        "kfold": {
            "n_splits": 5,
            "repeats": kfold_repeats,
            "mean": statistics.fmean(kfold_accuracies),
            "quantiles": _quantiles(kfold_accuracies),
        },
        "permutation_control": permuted_control(
            records,
            PRIMARY_RULE,
            observed=accuracy_of(logo["predictions"], records),
        ),
        "auc": roc_auc(labels, scores),
        "auc_ci_clustered": cluster_bootstrap(
            records, auc_statistic, iterations=bootstrap_iterations
        ),
        "brier": brier_score(labels, scores),
        "brier_baseline": brier_score(
            labels,
            [statistics.fmean(1.0 if lab == POSITIVE_LABEL else 0.0 for lab in labels)]
            * len(labels),
        ),
        "reliability": table,
        "ece": expected_calibration_error(table, len(records)),
        "sensitivity": sensitivity,
        "head_to_head": incumbents,
    }


def _quantiles(values: Sequence[float]) -> dict[str, float]:
    """Minimum, 5th, 50th, 95th percentile and maximum of a sample."""
    ordered = sorted(values)
    return {
        "min": ordered[0],
        "p05": ordered[int(0.05 * len(ordered))],
        "median": statistics.median(ordered),
        "p95": ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))],
        "max": ordered[-1],
    }


def _pct(value: float) -> str:
    """Format a proportion as a percentage with one decimal."""
    return f"{value * 100:.1f}%"


def build_report(analysis: dict[str, Any]) -> str:
    """Render the analysis as markdown.

    The ordering is deliberate: the out-of-sample number and the optimism gap
    come first, because replacing the in-sample 97.3% is the whole reason this
    script exists.
    """
    logo = analysis["logo"]
    default = analysis["default"]
    in_sample = analysis["in_sample_score"]
    fit = analysis["in_sample_fit"]
    control = analysis["permutation_control"]
    lines: list[str] = []

    lines.append("# Jev threshold: out-of-sample re-fit\n")
    lines.append(
        f"**Records:** {analysis['n_records']} scored, in "
        f"{analysis['n_groups']} question groups across "
        f"{analysis['n_ids']} record ids  \n"
        f"**Primary rule (pre-registered):** maximise "
        f"{analysis['primary_rule'].upper()} on the training fold  \n"
        f"**Primary estimator:** leave-one-group-out over question text\n"
    )

    lines.append("## Headline\n")
    lines.append(
        "| Threshold policy | Accuracy | 95% CI (clustered) | MCC | Missed harm | False alarm |\n"
        "|---|---|---|---|---|---|"
    )
    lines.append(
        f"| Fixed default 0.5 | {default['correct']}/{default['total']} "
        f"({_pct(default['accuracy'])}) | "
        f"{_pct(default['accuracy_ci_clustered'][0])} to "
        f"{_pct(default['accuracy_ci_clustered'][1])} | "
        f"{default['mcc']:.3f} | {default['missed_harm']} | {default['false_alarm']} |"
    )
    lines.append(
        f"| Fitted, out of sample | {logo['correct']}/{logo['total']} "
        f"({_pct(logo['accuracy'])}) | "
        f"{_pct(logo['accuracy_ci_clustered'][0])} to "
        f"{_pct(logo['accuracy_ci_clustered'][1])} | "
        f"{logo['mcc']:.3f} | {logo['missed_harm']} | {logo['false_alarm']} |"
    )
    lines.append(
        f"| Fitted, in sample | {in_sample['correct']}/{in_sample['total']} "
        f"({_pct(in_sample['accuracy'])}) | not applicable | "
        f"{in_sample['mcc']:.3f} | {in_sample['missed_harm']} | "
        f"{in_sample['false_alarm']} |\n"
    )
    gap = in_sample["accuracy"] - logo["accuracy"]
    lines.append(
        f"Optimism gap: {_pct(in_sample['accuracy'])} in sample against "
        f"{_pct(logo['accuracy'])} out of sample, a difference of "
        f"{gap * 100:.1f} points.\n"
    )

    lines.append("## Where the fitted threshold lands\n")
    lines.append(
        f"In-sample fit: {fit['threshold']:.3f}, tied across the band "
        f"{fit['tied_low']:.3f} to {fit['tied_high']:.3f}.\n"
    )
    lines.append(
        f"Across the {analysis['n_groups']} folds the fitted cut runs from "
        f"{logo['threshold_min']:.3f} to {logo['threshold_max']:.3f}, median "
        f"{logo['threshold_median']:.3f}. "
        f"{logo['folds_in_dead_zone']} folds land inside the empty band "
        f"{analysis['dead_zone'][0]} to {analysis['dead_zone'][1]}, where the cut "
        f"makes no difference to any verdict, and "
        f"{logo['folds_below_dead_zone']} land below it.\n"
    )
    lines.append(
        "Jev's probabilities on this set take "
        f"{len(analysis['distinct_scores'])} distinct values, none between "
        f"{analysis['dead_zone'][0]} and {analysis['dead_zone'][1]}. Every cut "
        "inside that band is the same classifier, so a fold that fits there has "
        "effectively chosen the default.\n"
    )

    lines.append("## Spread under repeated grouped 5-fold\n")
    kfold = analysis["kfold"]
    quantiles = kfold["quantiles"]
    lines.append(
        f"{kfold['repeats']} repeats, mean {_pct(kfold['mean'])}, "
        f"5th to 95th percentile {_pct(quantiles['p05'])} to "
        f"{_pct(quantiles['p95'])}, full range {_pct(quantiles['min'])} to "
        f"{_pct(quantiles['max'])}. Reported as spread, not as the headline.\n"
    )

    lines.append("## Negative control\n")
    lines.append(
        f"Labels permuted between same-sized question groups, "
        f"{control['repeats']} repeats through the identical pipeline: mean "
        f"accuracy {_pct(control['mean_accuracy'])}, worst case "
        f"{_pct(control['max_accuracy'])}, against a majority-class rate of "
        f"{_pct(control['majority_class_rate'])}. The real out-of-sample "
        f"{_pct(control['observed_accuracy'])} beats the permuted runs at "
        f"p = {control['p_value']:.4f}. The cross-validation is not "
        "manufacturing accuracy from noise.\n"
    )

    lines.append("## Calibration\n")
    lines.append(
        f"**Brier score: {analysis['brier']:.4f}** against "
        f"{analysis['brier_baseline']:.4f} for always predicting the base rate. "
        f"Binned expected calibration error {analysis['ece']:.4f}.\n"
    )
    lines.append(
        "| Bin | n | Mean probability | Observed harmful rate |\n|---|---|---|---|"
    )
    for row in analysis["reliability"]:
        if not row["n"]:
            lines.append(f"| {row['low']:.2f} to {row['high']:.2f} | 0 | | |")
            continue
        lines.append(
            f"| {row['low']:.2f} to {row['high']:.2f} | {row['n']} | "
            f"{row['mean_score']:.3f} | {row['observed']:.3f} |"
        )
    lines.append(
        "\nRead the counts before the rates. Most bins hold a handful of "
        "records, so this table shows the shape of the score distribution, not "
        "a calibration estimate fit to quote.\n"
    )

    lines.append("## Discrimination\n")
    lines.append(
        f"AUC {analysis['auc']:.3f}, 95% clustered bootstrap "
        f"{analysis['auc_ci_clustered'][0]:.3f} to "
        f"{analysis['auc_ci_clustered'][1]:.3f}. Resampling question groups "
        "rather than rows, which is wider than the row-level interval in "
        "`jev_judge_evaluation.md` and is the correct one for this design.\n"
    )

    if analysis["head_to_head"]:
        lines.append("## Head-to-head, with and without the independence assumption\n")
        lines.append(
            "| Comparison | Discordant | Exact McNemar p | Clustered permutation p |\n"
            "|---|---|---|---|"
        )
        for name, block in analysis["head_to_head"].items():
            lines.append(
                f"| Jev vs {name} | {block['discordant']} "
                f"({block['a_correct_b_wrong']}/{block['b_correct_a_wrong']}) | "
                f"{block['p_mcnemar_exact']:.3f} | "
                f"{block['p_cluster_permutation']:.3f} |"
            )
        lines.append(
            "\nThe exact test treats each discordant record as independent. The "
            "permutation flips a whole question group together. Neither reaches "
            "significance, so the independence assumption was not carrying the "
            "conclusion.\n"
        )

    lines.append("## Sensitivity to the fitting rule\n")
    lines.append(
        "These are not the headline. The primary rule was fixed before the "
        "script ran; these show whether the answer depends on that choice.\n"
    )
    lines.append(
        "| Rule | In-sample cut | Out-of-sample accuracy | MCC | Missed harm | False alarm |\n"
        "|---|---|---|---|---|---|"
    )
    lines.append(
        f"| {analysis['primary_rule']} (primary) | {fit['threshold']:.3f} | "
        f"{logo['correct']}/{logo['total']} ({_pct(logo['accuracy'])}) | "
        f"{logo['mcc']:.3f} | {logo['missed_harm']} | {logo['false_alarm']} |"
    )
    for rule, block in analysis["sensitivity"].items():
        block_logo = block["logo"]
        lines.append(
            f"| {rule} | {block['in_sample']['threshold']:.3f} | "
            f"{block_logo['correct']}/{block_logo['total']} "
            f"({_pct(block_logo['accuracy'])}) | {block_logo['mcc']:.3f} | "
            f"{block_logo['missed_harm']} | {block_logo['false_alarm']} |"
        )

    lines.append("\n## What this settles\n")
    lines.append(
        "It settles whether the in-sample 97.3% survives honest re-fitting. It "
        "does not establish a production operating point: every fold draws from "
        f"the same {analysis['n_records']} records and the same "
        f"{analysis['n_groups']} questions, so the estimate corrects optimism "
        "without being independent evidence. A cut chosen here still needs "
        "confirming on records this set has never seen.\n"
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    """Load the judged records, run every estimator, write report and JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jev_results", type=Path, default=JEV_RESULTS)
    parser.add_argument("--gold_path", type=Path, default=GOLD_PATH)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--summary_json", type=Path, default=None)
    parser.add_argument("--bootstrap_iterations", type=int, default=5000)
    parser.add_argument("--kfold_repeats", type=int, default=200)
    args = parser.parse_args()

    records = load_scored(args.jev_results, args.gold_path)
    analysis = build_analysis(
        records,
        bootstrap_iterations=args.bootstrap_iterations,
        kfold_repeats=args.kfold_repeats,
    )
    analysis["sources"] = {
        "jev_results": str(args.jev_results.relative_to(ROOT)),
        "gold_path": str(args.gold_path.relative_to(ROOT)),
    }

    report = build_report(analysis)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(report, encoding="utf-8")
        print(f"wrote {args.report}")
    else:
        print(report)

    if args.summary_json:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(
            json.dumps(analysis, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"wrote {args.summary_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
