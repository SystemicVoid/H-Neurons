#!/usr/bin/env python3
"""Summarize available jailbreak evaluator comparisons from local artifacts only."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = (
    ROOT / "data/judge_validation/available_jailbreak_evaluator_comparison.json"
)

CONTAMINATED_IDS = frozenset(
    {
        "jbb_harmful_62_t2",
        "jbb_harmful_97_t0",
        "jbb_harmful_14_t3",
        "jbb_harmful_91_t4",
        "jbb_harmful_36_t4",
        "jbb_harmful_34_t0",
        "jbb_harmful_19_t0",
        "jbb_harmful_3_t2",
    }
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def repo_relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def rate_summary(yes: int, total: int) -> dict[str, Any]:
    return {
        "yes": yes,
        "total": total,
        "rate": 0.0 if total == 0 else yes / total,
    }


def confusion(records: list[dict[str, Any]], field: str) -> dict[str, int]:
    tp = tn = fp = fn = 0
    for record in records:
        pred = record[field]
        gold = record["gold"]
        if gold and pred:
            tp += 1
        elif gold and not pred:
            fn += 1
        elif not gold and pred:
            fp += 1
        else:
            tn += 1
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def metrics_from_confusion(c: dict[str, int]) -> dict[str, float]:
    total = c["tp"] + c["tn"] + c["fp"] + c["fn"]
    precision = c["tp"] / (c["tp"] + c["fp"]) if (c["tp"] + c["fp"]) else 0.0
    recall = c["tp"] / (c["tp"] + c["fn"]) if (c["tp"] + c["fn"]) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "accuracy": 0.0 if total == 0 else (c["tp"] + c["tn"]) / total,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def summarize_binary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    harmful = sum(row.get("judge") == "HARMFUL" for row in rows)
    return rate_summary(harmful, len(rows))


def summarize_v2(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid_rows = [row for row in rows if "harmful_binary" in row.get("csv2", {})]
    yes = sum(row["csv2"]["harmful_binary"] == "yes" for row in valid_rows)
    borderline = sum(
        row["csv2"]["harmful_binary"] == "borderline" for row in valid_rows
    )
    summary = rate_summary(yes, len(valid_rows))
    summary["valid"] = len(valid_rows)
    summary["errors"] = len(rows) - len(valid_rows)
    summary["borderline"] = borderline
    summary["borderline_rate"] = 0.0 if not valid_rows else borderline / len(valid_rows)
    return summary


def summarize_v3(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid_rows = [row for row in rows if not row.get("csv2", {}).get("error")]
    yes = sum(row["csv2"]["harmful_binary"] == "yes" for row in valid_rows)
    substantive = sum(
        row["csv2"].get("primary_outcome") == "substantive_compliance"
        for row in valid_rows
    )
    summary = rate_summary(yes, len(valid_rows))
    summary["valid"] = len(valid_rows)
    summary["errors"] = len(rows) - len(valid_rows)
    summary["substantive_compliance"] = {
        "yes": substantive,
        "total": len(valid_rows),
        "rate": 0.0 if not valid_rows else substantive / len(valid_rows),
    }
    return summary


def v2_v3_transition(
    v2_rows: list[dict[str, Any]], v3_rows: list[dict[str, Any]]
) -> dict[str, int]:
    v3_by_id = {row["id"]: row for row in v3_rows}
    counts = {
        "yes_to_yes": 0,
        "yes_to_no": 0,
        "borderline_to_yes": 0,
        "borderline_to_no": 0,
        "no_to_yes": 0,
        "no_to_no": 0,
        "v2_errors": 0,
        "v3_errors": 0,
    }
    for row in v2_rows:
        row_v3 = v3_by_id[row["id"]]
        payload_v2 = row.get("csv2", {})
        payload_v3 = row_v3.get("csv2", {})
        if "harmful_binary" not in payload_v2:
            counts["v2_errors"] += 1
            continue
        if payload_v3.get("error"):
            counts["v3_errors"] += 1
            continue
        source = payload_v2["harmful_binary"]
        target = payload_v3["harmful_binary"]
        if source == "borderline":
            source = "borderline"
        key = f"{source}_to_{target}"
        counts[key] += 1
    return counts


def compute_gold_summary() -> dict[str, Any]:
    gold_rows = [
        row
        for row in load_jsonl(
            ROOT / "tests/gold_labels/jailbreak_cross_alpha_gold.jsonl"
        )
        if row["human_label"] != "BORDERLINE"
    ]
    gold_by_key = {(row["id"], row["alpha"]): row for row in gold_rows}

    v3_by_key = {}
    for alpha in ("0.0", "1.5", "3.0"):
        for row in load_jsonl(
            ROOT / "data/judge_validation/csv2_v3" / f"alpha_{alpha}.jsonl"
        ):
            v3_by_key[(row["id"], row["alpha"])] = row

    v2_by_key = {}
    for alpha in ("0.0", "1.5", "3.0"):
        for row in load_jsonl(
            ROOT
            / "data/gemma3_4b/intervention/jailbreak/csv2_evaluation"
            / f"alpha_{alpha}.jsonl"
        ):
            key = (row["id"], row["alpha"])
            if key in gold_by_key:
                v2_by_key[key] = row

    binary_by_key = {}
    for alpha in ("0.0", "1.5", "3.0"):
        for row in load_jsonl(
            ROOT
            / "data/gemma3_4b/intervention/jailbreak/experiment"
            / f"alpha_{alpha}.jsonl"
        ):
            key = (row["id"], row["alpha"])
            if key in gold_by_key:
                binary_by_key[key] = row

    sr_by_key = {}
    for row in load_jsonl(
        ROOT / "data/judge_validation/strongreject/strongreject_gold_results.jsonl"
    ):
        sr_by_key[(row["id"], row["alpha"])] = row

    joined: list[dict[str, Any]] = []
    for key, gold in sorted(gold_by_key.items()):
        joined.append(
            {
                "id": gold["id"],
                "alpha": gold["alpha"],
                "gold": gold["human_label"] == "HARMFUL",
                "split": "dev" if gold["id"] in CONTAMINATED_IDS else "holdout",
                "v3": v3_by_key[key]["csv2"]["harmful_binary"] == "yes",
                "v2": v2_by_key[key]["csv2"]["harmful_binary"] == "yes",
                "binary": binary_by_key[key].get("judge") == "HARMFUL",
                "sr4o": sr_by_key[key]["strongreject"]["harmful_binary"] == "yes",
            }
        )

    splits = {
        "combined": joined,
        "dev": [row for row in joined if row["split"] == "dev"],
        "holdout": [row for row in joined if row["split"] == "holdout"],
    }
    out: dict[str, Any] = {"n_total": len(joined), "splits": {}}
    for split_name, records in splits.items():
        split_summary: dict[str, Any] = {
            "n_records": len(records),
            "n_harmful": sum(record["gold"] for record in records),
            "n_safe": sum(not record["gold"] for record in records),
            "evaluators": {},
        }
        for field in ("v3", "sr4o", "v2", "binary"):
            c = confusion(records, field)
            split_summary["evaluators"][field] = {**c, **metrics_from_confusion(c)}
        out["splits"][split_name] = split_summary

    holdout = splits["holdout"]
    discordant = sum(row["v3"] != row["sr4o"] for row in holdout)
    out["holdout_v3_vs_sr4o"] = {"discordant_records": discordant}
    return out


def compute_h_sweep_summary() -> dict[str, Any]:
    base = ROOT / "data/gemma3_4b/intervention/jailbreak"
    out: dict[str, Any] = {"alphas": {}}
    for alpha in ("0.0", "1.0", "1.5", "3.0"):
        binary_rows = load_jsonl(base / "experiment" / f"alpha_{alpha}.jsonl")
        v2_rows = load_jsonl(base / "csv2_evaluation" / f"alpha_{alpha}.jsonl")
        v3_rows = load_jsonl(base / "csv2_v3_evaluation" / f"alpha_{alpha}.jsonl")
        out["alphas"][alpha] = {
            "binary": summarize_binary(binary_rows),
            "v2": summarize_v2(v2_rows),
            "v3": summarize_v3(v3_rows),
            "v2_to_v3_transition": v2_v3_transition(v2_rows, v3_rows),
        }
    return out


def compute_control_summary() -> dict[str, Any]:
    seed0 = (
        ROOT
        / "data/gemma3_4b/intervention/jailbreak/control/seed_0_unconstrained/csv2_evaluation"
    )
    seed1 = (
        ROOT
        / "data/gemma3_4b/intervention/jailbreak/control/seed_1_unconstrained_csv2_v3"
    )
    out: dict[str, Any] = {"seed0_v2": {}, "seed1_v3": {}}
    for alpha in ("0.0", "1.0", "1.5", "3.0"):
        out["seed0_v2"][alpha] = summarize_v2(
            load_jsonl(seed0 / f"alpha_{alpha}.jsonl")
        )
        out["seed1_v3"][alpha] = summarize_v3(
            load_jsonl(seed1 / f"alpha_{alpha}.jsonl")
        )
    return out


def compute_d7_summary() -> dict[str, Any]:
    root = ROOT / "data/gemma3_4b/intervention/jailbreak_d7/full500_canonical"
    surfaces = {
        "baseline_noop": ("1.0", True),
        "l1_neuron": ("3.0", True),
        "causal_locked": ("4.0", True),
        "probe_locked": ("1.0", False),
        "causal_random_head_layer_matched/seed_1": ("4.0", False),
        "causal_random_head_layer_matched/seed_2": ("4.0", False),
    }
    out: dict[str, Any] = {}
    for surface, (alpha, has_v3) in surfaces.items():
        surface_root = root / surface
        binary_rows = load_jsonl(surface_root / "experiment" / f"alpha_{alpha}.jsonl")
        v2_rows = load_jsonl(surface_root / "csv2_evaluation" / f"alpha_{alpha}.jsonl")
        summary: dict[str, Any] = {
            "alpha": float(alpha),
            "binary": summarize_binary(binary_rows),
            "v2": summarize_v2(v2_rows),
            "has_v3": has_v3,
        }
        if has_v3:
            v3_rows = load_jsonl(
                surface_root / "csv2_v3_evaluation" / f"alpha_{alpha}.jsonl"
            )
            summary["v3"] = summarize_v3(v3_rows)
            summary["v2_to_v3_transition"] = v2_v3_transition(v2_rows, v3_rows)
        out[surface] = summary
    return out


def compute_coverage_matrix() -> dict[str, Any]:
    root = ROOT / "data/gemma3_4b/intervention"
    coverage = {
        "gold_subset": {
            "binary": True,
            "v2": True,
            "v3": True,
            "strongreject": True,
        },
        "h_neuron_sweep": {
            "binary": True,
            "v2": True,
            "v3": True,
            "strongreject": False,
        },
        "seed0_control": {
            "binary": True,
            "v2": True,
            "v3": False,
            "strongreject": False,
        },
        "seed1_control": {
            "binary": False,
            "v2": False,
            "v3": True,
            "strongreject": False,
        },
        "d7_baseline_l1_causal": {
            "binary": True,
            "v2": True,
            "v3": True,
            "strongreject": False,
        },
        "d7_probe_randoms": {
            "binary": True,
            "v2": True,
            "v3": False,
            "strongreject": False,
        },
        "available_paths_checked": {
            "jailbreak_root": repo_relative(root / "jailbreak"),
            "d7_root": repo_relative(root / "jailbreak_d7/full500_canonical"),
        },
    }
    return coverage


def main() -> None:
    summary = {
        "coverage_matrix": compute_coverage_matrix(),
        "gold_comparison": compute_gold_summary(),
        "h_neuron_sweep": compute_h_sweep_summary(),
        "controls": compute_control_summary(),
        "d7": compute_d7_summary(),
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
