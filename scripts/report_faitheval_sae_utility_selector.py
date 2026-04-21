"""Report the held-out FaithEval utility-selector ablation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from uncertainty import build_rate_summary, paired_bootstrap_binary_rate_difference
from utils import (
    finish_run_provenance,
    format_alpha_label,
    provenance_error_message,
    provenance_status_for_exception,
    start_run_provenance,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--selector_dir",
        type=Path,
        default=Path(
            "data/gemma3_4b/intervention/faitheval_sae_utility_selector/selector"
        ),
    )
    parser.add_argument(
        "--heldout_dir",
        type=Path,
        default=Path(
            "data/gemma3_4b/intervention/"
            "faitheval_sae_utility_selector/heldout/utility_selected/experiment"
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data/gemma3_4b/intervention/faitheval_sae_utility_selector"),
    )
    parser.add_argument("--baseline_alpha", type=float, default=1.0)
    parser.add_argument("--intervention_alpha", type=float, default=0.0)
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _alpha_path(root: Path, alpha: float) -> Path:
    return root / f"alpha_{format_alpha_label(alpha)}.jsonl"


def _rows_by_id(
    rows: list[dict[str, Any]], *, context: str
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        sample_id = str(row["id"])
        if sample_id in out:
            raise ValueError(f"{context}: duplicate sample id {sample_id!r}")
        out[sample_id] = row
    return out


def build_heldout_summary(
    *,
    selector_summary: dict[str, Any],
    baseline_rows: list[dict[str, Any]],
    intervention_rows: list[dict[str, Any]],
    baseline_alpha: float,
    intervention_alpha: float,
) -> dict[str, Any]:
    baseline_by_id = _rows_by_id(baseline_rows, context="noop baseline")
    intervention_by_id = _rows_by_id(intervention_rows, context="utility selected")
    baseline_ids = set(baseline_by_id)
    intervention_ids = set(intervention_by_id)
    if baseline_ids != intervention_ids:
        raise ValueError(
            "Held-out FaithEval parity failed between noop and utility-selected runs"
        )
    ordered_ids = sorted(baseline_ids)
    baseline_compliance = np.array(
        [
            bool(baseline_by_id[sample_id].get("compliance"))
            for sample_id in ordered_ids
        ],
        dtype=bool,
    )
    intervention_compliance = np.array(
        [
            bool(intervention_by_id[sample_id].get("compliance"))
            for sample_id in ordered_ids
        ],
        dtype=bool,
    )
    return {
        "schema_version": "faitheval_sae_utility_selector_report/v1",
        "benchmark": "faitheval",
        "n_samples": len(ordered_ids),
        "baseline_alpha": float(baseline_alpha),
        "intervention_alpha": float(intervention_alpha),
        "heldout_compliance": {
            "noop": build_rate_summary(
                int(baseline_compliance.sum()),
                int(len(baseline_compliance)),
                count_key="n_compliant",
                total_key="n_total",
            ),
            "utility_selected": build_rate_summary(
                int(intervention_compliance.sum()),
                int(len(intervention_compliance)),
                count_key="n_compliant",
                total_key="n_total",
            ),
            "utility_minus_noop_pp": paired_bootstrap_binary_rate_difference(
                baseline_compliance,
                intervention_compliance,
            ),
        },
        "selector_diagnostics": {
            "selected_k": selector_summary["utility_selected"]["k"],
            "outside_old_shortlist_count": selector_summary["utility_selected"][
                "outside_old_shortlist_count"
            ],
            "outside_old_shortlist_fraction": selector_summary["utility_selected"][
                "outside_old_shortlist_fraction"
            ],
            "utility_layer_histogram": selector_summary["utility_selected"][
                "layer_histogram"
            ],
            "readout_layer_histogram": selector_summary["readout_reference"][
                "layer_histogram"
            ],
            "overlap_with_readout_positive": selector_summary[
                "overlap_with_readout_positive"
            ],
        },
    }


def build_audit_note(summary: dict[str, Any]) -> str:
    noop = summary["heldout_compliance"]["noop"]
    utility = summary["heldout_compliance"]["utility_selected"]
    delta = summary["heldout_compliance"]["utility_minus_noop_pp"]
    diagnostics = summary["selector_diagnostics"]
    return "\n".join(
        [
            "# FaithEval SAE Utility Selector Audit",
            "",
            f"- Held-out n: {summary['n_samples']}",
            (
                f"- No-op compliance: {noop['estimate']:.4f} "
                f"({noop['n_compliant']}/{noop['n_total']})"
            ),
            (
                f"- Utility-selected compliance: {utility['estimate']:.4f} "
                f"({utility['n_compliant']}/{utility['n_total']})"
            ),
            (
                f"- Utility minus no-op: {delta['estimate_pp']:+.3f} pp "
                f"[{delta['ci_pp']['lower']:+.3f}, {delta['ci_pp']['upper']:+.3f}]"
            ),
            (
                "- Outside old |w|>1e-3 shortlist: "
                f"{diagnostics['outside_old_shortlist_count']} / {diagnostics['selected_k']} "
                f"({diagnostics['outside_old_shortlist_fraction']:.4f})"
            ),
            f"- Utility layer histogram: {json.dumps(diagnostics['utility_layer_histogram'], sort_keys=True)}",
            f"- Readout layer histogram: {json.dumps(diagnostics['readout_layer_histogram'], sort_keys=True)}",
        ]
    )


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir / "report"
    summary_path = output_dir / "heldout_summary.json"
    audit_path = output_dir / "audit_note.md"
    provenance_handle = start_run_provenance(
        args,
        primary_target=output_dir,
        output_targets=[output_dir, summary_path, audit_path],
        primary_target_is_dir=True,
    )
    provenance_status = "completed"
    provenance_extra: dict[str, Any] = {}
    try:
        selector_summary = _load_json(args.selector_dir / "selector_summary.json")
        baseline_rows = _load_jsonl(_alpha_path(args.heldout_dir, args.baseline_alpha))
        intervention_rows = _load_jsonl(
            _alpha_path(args.heldout_dir, args.intervention_alpha)
        )
        summary = build_heldout_summary(
            selector_summary=selector_summary,
            baseline_rows=baseline_rows,
            intervention_rows=intervention_rows,
            baseline_alpha=args.baseline_alpha,
            intervention_alpha=args.intervention_alpha,
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        audit_path.write_text(build_audit_note(summary) + "\n", encoding="utf-8")
        provenance_extra["n_samples"] = summary["n_samples"]
    except BaseException as exc:
        provenance_status = provenance_status_for_exception(exc)
        provenance_extra["error"] = provenance_error_message(exc)
        raise
    finally:
        finish_run_provenance(provenance_handle, provenance_status, provenance_extra)


if __name__ == "__main__":
    main()
