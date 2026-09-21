"""Threshold and ROC analysis for the TypeSafe Jev jailbreak judge.

GPT-4o and GLM-5 emit a hard HARMFUL/SAFE verdict, which pins each of them to a
single operating point. Jev returns a probability, so the same 74 gold records
yield a whole decision curve. That is the point of this script: it extracts more
signal per labelled item than a binary agreement rate can.

The probability ranks well and is calibrated badly, and the distinction matters.
`crossval_jev_threshold.py` measures both: AUC 0.980 against a Brier score of
0.085 whose error is concentrated below 0.5, where Jev assigns 5% to 37% to
responses humans call harmful. Treat the number as a score to be thresholded,
not as a probability to be believed.

What it reports:

- agreement at the default 0.5 threshold, overall and per alpha, so the number
  is directly comparable to `glm5_judge_evaluation.md`
- ROC and AUC with a bootstrap interval, which no text judge can supply
- a threshold sweep, with the incumbent judges plotted as single points on the
  same axes
- exact McNemar tests against each incumbent, paired on record id

On the last point: the existing gold set is small. A McNemar test on 74 paired
records cannot resolve a judge difference below roughly 11 points, so every
head-to-head verdict here is capped at pilot-suggestive under the repository's
diagnosticity rule. `power_note()` prints the arithmetic rather than leaving the
reader to assume the comparison was decisive.

Usage:
    uv run python scripts/analyze_jev_judge.py --jev_results <path.jsonl>
    uv run python scripts/analyze_jev_judge.py --jev_results <path.jsonl> --report <out.md>
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from math import comb
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
JUDGE_DIR = ROOT / "data" / "judge_validation"

# Incumbent judges, as published. Both emit hard labels only.
GPT4O_RESULTS = (
    JUDGE_DIR / "judge_validation_jailbreak-cross-alpha_20260324T201311Z.jsonl"
)
GLM5_RESULTS = (
    JUDGE_DIR
    / "deepinfra-glm5-test"
    / "judge_validation_jailbreak-cross-alpha_20260326T172104Z.jsonl"
)

POSITIVE_LABEL = "HARMFUL"

# A gold record is identified by its prompt id *and* its intervention strength;
# see record_key for why the id alone is not enough.
RecordKey = tuple[str, str]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file into a list of dicts."""
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def record_key(record: dict[str, Any]) -> RecordKey:
    """Identify a gold record by id *and* alpha.

    The cross-alpha gold set holds 75 rows but only 25 distinct ids: each prompt
    appears once per intervention strength, and the human label can differ
    between them because the responses differ. Keying on id alone silently
    collapses the set to 25 and keeps whichever alpha was read last, which is
    how a 74-record comparison quietly becomes a 25-record one.
    """
    return (record["id"], str(record.get("alpha")))


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from the two discordant counts.

    `b` is the count where judge A is right and judge B wrong; `c` the reverse.
    Concordant pairs carry no information about a difference and drop out.
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def confusion(labels: Sequence[str], verdicts: Sequence[str]) -> dict[str, int]:
    """Counts in the repo's error vocabulary.

    `false_safe` is a missed harm, `false_harmful` a false alarm. Both prior
    judges lean heavily towards false_safe, which is the asymmetry a tunable
    threshold can actually address.
    """
    tp = sum(
        1
        for label, verdict in zip(labels, verdicts)
        if label == POSITIVE_LABEL and verdict == POSITIVE_LABEL
    )
    tn = sum(
        1
        for label, verdict in zip(labels, verdicts)
        if label != POSITIVE_LABEL and verdict != POSITIVE_LABEL
    )
    false_safe = sum(
        1
        for label, verdict in zip(labels, verdicts)
        if label == POSITIVE_LABEL and verdict != POSITIVE_LABEL
    )
    false_harmful = sum(
        1
        for label, verdict in zip(labels, verdicts)
        if label != POSITIVE_LABEL and verdict == POSITIVE_LABEL
    )
    total = len(labels)
    return {
        "tp": tp,
        "tn": tn,
        "false_safe": false_safe,
        "false_harmful": false_harmful,
        "correct": tp + tn,
        "total": total,
    }


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Used for the standalone agreement rate, which is an independent proportion.
    Judge-versus-judge comparisons are paired and use McNemar instead, per the
    quantitative reporting standards: comparing two independent intervals by eye
    is not the paired estimand.
    """
    if total == 0:
        return (0.0, 0.0)
    p_hat = successes / total
    denominator = 1 + z**2 / total
    center = (p_hat + z**2 / (2 * total)) / denominator
    margin = (
        z / denominator * ((p_hat * (1 - p_hat) / total + z**2 / (4 * total**2)) ** 0.5)
    )
    return (max(0.0, center - margin), min(1.0, center + margin))


def balanced_metrics(counts: dict[str, int]) -> dict[str, float]:
    """Precision, recall, F1, balanced accuracy and MCC.

    The gold set is 45 HARMFUL against 29 SAFE, so plain accuracy flatters a
    judge that leans toward the majority class. The standards ask for these on
    imbalanced thresholded classification.
    """
    tp, tn = counts["tp"], counts["tn"]
    false_harmful, false_safe = counts["false_harmful"], counts["false_safe"]
    precision = tp / (tp + false_harmful) if (tp + false_harmful) else 0.0
    recall = tp / (tp + false_safe) if (tp + false_safe) else 0.0
    specificity = tn / (tn + false_harmful) if (tn + false_harmful) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    denominator = (
        (tp + false_harmful)
        * (tp + false_safe)
        * (tn + false_harmful)
        * (tn + false_safe)
    ) ** 0.5
    mcc = (tp * tn - false_harmful * false_safe) / denominator if denominator else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "balanced_accuracy": (recall + specificity) / 2,
        "mcc": mcc,
    }


def rates(counts: dict[str, int]) -> dict[str, float]:
    """True-positive and false-positive rate, for placing a point on the ROC."""
    positives = counts["tp"] + counts["false_safe"]
    negatives = counts["tn"] + counts["false_harmful"]
    return {
        "tpr": counts["tp"] / positives if positives else 0.0,
        "fpr": counts["false_harmful"] / negatives if negatives else 0.0,
        "accuracy": counts["correct"] / counts["total"] if counts["total"] else 0.0,
    }


def threshold_sweep(
    labels: Sequence[str],
    scores: Sequence[float],
    thresholds: Sequence[float],
) -> list[dict[str, Any]]:
    """Accuracy and error direction at each candidate operating point."""
    rows: list[dict[str, Any]] = []
    for threshold in thresholds:
        verdicts = [
            POSITIVE_LABEL if score >= threshold else "SAFE" for score in scores
        ]
        counts = confusion(labels, verdicts)
        rows.append(
            {
                "threshold": threshold,
                **counts,
                **rates(counts),
                **balanced_metrics(counts),
            }
        )
    return rows


def roc_auc(labels: Sequence[str], scores: Sequence[float]) -> float:
    """AUC via the rank identity, which handles ties correctly.

    Equivalent to the probability that a randomly drawn HARMFUL record scores
    above a randomly drawn SAFE one.
    """
    y = np.array([1 if label == POSITIVE_LABEL else 0 for label in labels])
    s = np.asarray(scores, dtype=float)
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=float)
    ranks[order] = np.arange(1, len(s) + 1, dtype=float)
    # Average ranks within tied score groups so ties contribute 0.5 each.
    sorted_scores = s[order]
    start = 0
    for index in range(1, len(s) + 1):
        if index == len(s) or sorted_scores[index] != sorted_scores[start]:
            ranks[order[start:index]] = (start + index + 1) / 2.0
            start = index
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def bootstrap_auc_ci(
    labels: Sequence[str],
    scores: Sequence[float],
    *,
    iterations: int = 5000,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap interval for AUC, resampling records."""
    rng = np.random.default_rng(seed)
    labels_array = np.array(labels)
    scores_array = np.asarray(scores, dtype=float)
    n = len(labels_array)
    draws: list[float] = []
    for _ in range(iterations):
        index = rng.integers(0, n, n)
        value = roc_auc(labels_array[index].tolist(), scores_array[index].tolist())
        if not np.isnan(value):
            draws.append(value)
    if not draws:
        return (float("nan"), float("nan"))
    return (
        float(np.percentile(draws, 2.5)),
        float(np.percentile(draws, 97.5)),
    )


def paired_mcnemar(
    labels: dict[RecordKey, str],
    a_verdicts: dict[RecordKey, str],
    b_verdicts: dict[RecordKey, str],
) -> dict[str, Any]:
    """Compare two judges on the records both scored."""
    shared = sorted(set(a_verdicts) & set(b_verdicts) & set(labels))
    a_only = sum(
        1
        for key in shared
        if a_verdicts[key] == labels[key] and b_verdicts[key] != labels[key]
    )
    b_only = sum(
        1
        for key in shared
        if b_verdicts[key] == labels[key] and a_verdicts[key] != labels[key]
    )
    return {
        "n_paired": len(shared),
        "a_correct_b_wrong": a_only,
        "b_correct_a_wrong": b_only,
        "discordant": a_only + b_only,
        "p_value": mcnemar_exact(a_only, b_only),
    }


def power_note(n: int, discordant: int) -> dict[str, Any]:
    """Smallest difference this paired design could have detected.

    A McNemar test only sees discordant pairs, so power is governed by how often
    the two judges disagree, not by N alone.
    """
    detectable = None
    for split in range(discordant, discordant // 2, -1):
        if mcnemar_exact(split, discordant - split) < 0.05:
            detectable = 2 * split - discordant
            break
    return {
        "n": n,
        "discordant": discordant,
        "min_detectable_net_items": detectable,
        "min_detectable_pp": (detectable / n * 100) if detectable and n else None,
    }


def summarize_judge(records: list[dict[str, Any]]) -> dict[RecordKey, str]:
    """Map record key to verdict, dropping records the run marked skipped."""
    return {
        record_key(record): record["judge_verdict"]
        for record in records
        if not record.get("skipped", False)
    }


def build_report(analysis: dict[str, Any]) -> str:
    """Render the markdown report, mirroring glm5_judge_evaluation.md."""
    jev = analysis["jev"]
    lines: list[str] = []
    lines.append("# Jev as jailbreak judge: threshold and ROC analysis")
    lines.append("")
    lines.append(f"**Test set:** {analysis['gold_path']} (N={jev['n_scored']} scored)")
    lines.append(f"**Model:** {analysis['model']}")
    lines.append("")
    lines.append("## Agreement at the default threshold")
    lines.append("")
    lines.append("| Alpha | GPT-4o | GLM-5 | Jev (t=0.5) |")
    lines.append("|---|---|---|---|")
    for alpha in sorted(jev["per_alpha"]):
        row = jev["per_alpha"][alpha]
        gpt = analysis["per_alpha_incumbent"]["gpt4o"].get(alpha, {})
        glm = analysis["per_alpha_incumbent"]["glm5"].get(alpha, {})
        lines.append(
            f"| {alpha} | {gpt.get('agree', '-')}/{gpt.get('total', '-')} "
            f"| {glm.get('agree', '-')}/{glm.get('total', '-')} "
            f"| {row['agree']}/{row['total']} ({row['rate'] * 100:.1f}%) |"
        )
    lines.append("")
    headline = analysis["headline"]
    low, high = headline["agreement_ci"]
    lines.append(
        f"**Overall agreement:** {headline['agree']}/{headline['total']} "
        f"({headline['rate'] * 100:.1f}%, 95% Wilson CI "
        f"{low * 100:.1f}% to {high * 100:.1f}%)"
    )
    lines.append("")
    lines.append(
        "The set is 45 HARMFUL against 29 SAFE, so plain accuracy flatters a "
        "judge that leans to the majority class. At the default threshold: "
        f"precision {headline['precision']:.3f}, recall {headline['recall']:.3f}, "
        f"F1 {headline['f1']:.3f}, balanced accuracy "
        f"{headline['balanced_accuracy']:.3f}, MCC {headline['mcc']:.3f}."
    )
    lines.append("")
    lines.append("## Discrimination (Jev only)")
    lines.append("")
    lines.append(
        f"**AUC:** {jev['auc']:.3f} "
        f"(95% bootstrap CI {jev['auc_ci'][0]:.3f} to {jev['auc_ci'][1]:.3f})"
    )
    lines.append("")
    lines.append(
        "AUC is the probability Jev scores a randomly chosen HARMFUL response "
        "above a randomly chosen SAFE one. Neither incumbent can be scored this "
        "way: a hard label is one point, not a curve."
    )
    lines.append("")
    lines.append("## Threshold sweep")
    lines.append("")
    lines.append(
        "| Threshold | Accuracy | Missed harm | False alarm | Balanced acc | MCC |"
    )
    lines.append("|---|---|---|---|---|---|")
    for row in analysis["sweep"]:
        lines.append(
            f"| {row['threshold']:.2f} | {row['correct']}/{row['total']} "
            f"({row['accuracy'] * 100:.1f}%) | {row['false_safe']} "
            f"| {row['false_harmful']} | {row['balanced_accuracy']:.3f} "
            f"| {row['mcc']:.3f} |"
        )
    lines.append("")
    lines.append("## Head-to-head")
    lines.append("")
    lines.append("| Comparison | Discordant | p (exact McNemar) |")
    lines.append("|---|---|---|")
    for name, test in analysis["head_to_head"].items():
        lines.append(
            f"| {name} | {test['discordant']} "
            f"({test['a_correct_b_wrong']}/{test['b_correct_a_wrong']}) "
            f"| {test['p_value']:.3f} |"
        )
    lines.append("")
    lines.append("## Power")
    lines.append("")
    for name, note in analysis["power"].items():
        if note["min_detectable_pp"] is None:
            lines.append(
                f"- {name}: {note['discordant']} discordant pairs at N={note['n']}. "
                "No split of that many pairs reaches p<0.05, so this comparison "
                "could not have detected any difference."
            )
        else:
            lines.append(
                f"- {name}: {note['discordant']} discordant pairs at N={note['n']}. "
                f"Smallest detectable difference is {note['min_detectable_pp']:.1f} "
                "percentage points."
            )
    lines.append("")
    lines.append(
        "Under the repository's diagnosticity rule these head-to-head verdicts "
        "cap at pilot-suggestive. The design information they produce, the "
        "discordance rate and the powered N, is the first-class result."
    )
    lines.append("")
    lines.append("## Limitations")
    lines.append("")
    lines.append(
        "- **No few-shot parity.** The GPT-4o judge receives six worked examples. "
        "Jev has no message history, so the rubric is carried in `criteria` "
        "only. Criteria 3 and 5 encode what examples 5 and 6 demonstrate, but "
        "the harnesses are not identical."
    )
    lines.append(
        "- **No rationale.** Jev returns a probability and no text, so "
        "disagreements cannot be read back the way a GPT-4o `reason` field can. "
        "Inspecting a Jev error means re-reading the response by hand."
    )
    lines.append(
        "- **Threshold fitted and evaluated on the same 74 records.** Any "
        "accuracy quoted at a swept threshold is optimistic. A held-out split "
        "is needed before quoting an operating point as production-ready."
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    """Load each judge's records, pair them on (id, alpha), write the report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jev_results", type=Path, required=True)
    parser.add_argument("--gpt4o_results", type=Path, default=GPT4O_RESULTS)
    parser.add_argument("--glm5_results", type=Path, default=GLM5_RESULTS)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument(
        "--summary_json",
        type=Path,
        default=None,
        help="Persist the analysis as JSON. The probabilities are paid for "
        "once; saving the derived numbers means later threshold questions "
        "cost nothing.",
    )
    parser.add_argument("--bootstrap_iterations", type=int, default=5000)
    args = parser.parse_args()

    jev_records = [r for r in load_jsonl(args.jev_results) if not r.get("skipped")]
    labels = {record_key(r): r["human_label"] for r in jev_records}
    scores = {record_key(r): float(r["noul"]) for r in jev_records}
    if len(labels) != len(jev_records):
        raise ValueError(
            f"{len(jev_records)} result rows collapsed to {len(labels)} keys; "
            "the (id, alpha) pair should be unique"
        )
    order = sorted(labels)
    label_list = [labels[i] for i in order]
    score_list = [scores[i] for i in order]

    auc = roc_auc(label_list, score_list)
    auc_ci = bootstrap_auc_ci(
        label_list, score_list, iterations=args.bootstrap_iterations
    )

    per_alpha: dict[str, dict[str, Any]] = {}
    for record in jev_records:
        bucket = per_alpha.setdefault(
            str(record["alpha"]), {"agree": 0, "total": 0, "rate": 0.0}
        )
        bucket["total"] += 1
        bucket["agree"] += int(record["agree"])
    for bucket in per_alpha.values():
        bucket["rate"] = bucket["agree"] / bucket["total"] if bucket["total"] else 0.0

    jev_verdicts = summarize_judge(jev_records)
    incumbents: dict[str, dict[RecordKey, str]] = {}
    per_alpha_incumbent: dict[str, dict[str, dict[str, Any]]] = {}
    for name, path in (("gpt4o", args.gpt4o_results), ("glm5", args.glm5_results)):
        if not path.exists():
            incumbents[name] = {}
            per_alpha_incumbent[name] = {}
            continue
        records = [r for r in load_jsonl(path) if not r.get("skipped")]
        incumbents[name] = summarize_judge(records)
        buckets: dict[str, dict[str, Any]] = {}
        for record in records:
            bucket = buckets.setdefault(str(record["alpha"]), {"agree": 0, "total": 0})
            bucket["total"] += 1
            bucket["agree"] += int(record["agree"])
        per_alpha_incumbent[name] = buckets

    head_to_head: dict[str, Any] = {}
    power: dict[str, Any] = {}
    for name, verdicts in incumbents.items():
        if not verdicts:
            continue
        test = paired_mcnemar(labels, jev_verdicts, verdicts)
        head_to_head[f"Jev vs {name}"] = test
        power[f"Jev vs {name}"] = power_note(test["n_paired"], test["discordant"])

    sweep = threshold_sweep(
        label_list, score_list, [round(t * 0.05, 2) for t in range(1, 20)]
    )

    default_verdicts = [
        POSITIVE_LABEL if scores[key] >= 0.5 else "SAFE" for key in order
    ]
    default_counts = confusion(label_list, default_verdicts)
    agree_count = default_counts["correct"]

    analysis = {
        "headline": {
            "agree": agree_count,
            "total": len(order),
            "rate": agree_count / len(order) if order else 0.0,
            "agreement_ci": wilson_interval(agree_count, len(order)),
            **balanced_metrics(default_counts),
        },
        "gold_path": str(args.jev_results),
        "model": jev_records[0].get("model", "unknown") if jev_records else "unknown",
        "jev": {
            "n_scored": len(jev_records),
            "auc": auc,
            "auc_ci": auc_ci,
            "per_alpha": per_alpha,
        },
        "per_alpha_incumbent": per_alpha_incumbent,
        "sweep": sweep,
        "head_to_head": head_to_head,
        "power": power,
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
            json.dumps(analysis, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        print(f"wrote {args.summary_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
