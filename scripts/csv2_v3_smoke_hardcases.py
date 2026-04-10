#!/usr/bin/env python3
"""Build and audit the CSV v3 hard-case smoke-test cohort."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from utils import format_alpha_label

DEFAULT_EXPERIMENT_DIR = Path("data/gemma3_4b/intervention/jailbreak/experiment")
DEFAULT_LEGACY_DIR = Path("data/gemma3_4b/intervention/jailbreak/csv2_evaluation")
DEFAULT_GOLD_PATH = Path("tests/gold_labels/jailbreak_cross_alpha_gold.jsonl")
DEFAULT_SUBSET_DIR = Path(
    "data/gemma3_4b/intervention/jailbreak/experiment_csv2_v3_smoke_hardcases"
)
DEFAULT_V3_DIR = Path("data/gemma3_4b/intervention/jailbreak/csv2_v3_smoke_hardcases")
DEFAULT_MANIFEST_PATH = DEFAULT_SUBSET_DIR / "smoke_selection_manifest.json"
DEFAULT_AUDIT_DIR = DEFAULT_V3_DIR / "audit"

_GOLD_BACKBONE_ALPHAS = (0.0, 1.5, 3.0)
_GOLD_BACKBONE_IDS = (
    "jbb_harmful_62_t2",
    "jbb_harmful_97_t0",
    "jbb_harmful_14_t3",
    "jbb_harmful_91_t4",
    "jbb_harmful_36_t4",
    "jbb_harmful_34_t0",
    "jbb_harmful_19_t0",
    "jbb_harmful_3_t2",
)
_EXTREME_PIVOT_CASES = (
    ("jbb_harmful_82_t2", 1.0),
    ("jbb_harmful_44_t2", 1.5),
    ("jbb_harmful_71_t2", 1.5),
    ("jbb_harmful_92_t2", 0.0),
    ("jbb_harmful_29_t2", 0.0),
    ("jbb_harmful_16_t2", 3.0),
)


@dataclass(frozen=True)
class SmokeSelection:
    group: str
    prompt_id: str
    alpha: float
    selection_reason: str


def default_smoke_selections() -> list[SmokeSelection]:
    selections: list[SmokeSelection] = []
    for prompt_id in _GOLD_BACKBONE_IDS:
        reason = "gold_backbone_cross_alpha_failure_surface"
        for alpha in _GOLD_BACKBONE_ALPHAS:
            selections.append(
                SmokeSelection(
                    group="gold_backbone",
                    prompt_id=prompt_id,
                    alpha=alpha,
                    selection_reason=reason,
                )
            )
    for prompt_id, alpha in _EXTREME_PIVOT_CASES:
        selections.append(
            SmokeSelection(
                group="extreme_pivot",
                prompt_id=prompt_id,
                alpha=alpha,
                selection_reason="extreme_delayed_pivot_non_gold_expansion",
            )
        )
    return selections


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="Materialize the 24+6 smoke subset")
    build.add_argument(
        "--experiment_dir",
        type=Path,
        default=DEFAULT_EXPERIMENT_DIR,
    )
    build.add_argument(
        "--legacy_dir",
        type=Path,
        default=DEFAULT_LEGACY_DIR,
    )
    build.add_argument(
        "--gold_labels",
        type=Path,
        default=DEFAULT_GOLD_PATH,
    )
    build.add_argument(
        "--output_dir",
        type=Path,
        default=DEFAULT_SUBSET_DIR,
    )
    build.add_argument(
        "--manifest_path",
        type=Path,
        default=DEFAULT_MANIFEST_PATH,
    )

    report = subparsers.add_parser(
        "report",
        help="Build the joined audit workbook from smoke-test outputs",
    )
    report.add_argument(
        "--manifest_path",
        type=Path,
        default=DEFAULT_MANIFEST_PATH,
    )
    report.add_argument(
        "--legacy_dir",
        type=Path,
        default=DEFAULT_LEGACY_DIR,
    )
    report.add_argument(
        "--gold_labels",
        type=Path,
        default=DEFAULT_GOLD_PATH,
    )
    report.add_argument(
        "--v3_dir",
        type=Path,
        default=DEFAULT_V3_DIR,
    )
    report.add_argument(
        "--output_dir",
        type=Path,
        default=DEFAULT_AUDIT_DIR,
    )

    return parser.parse_args()


def _alpha_path(base: Path, alpha: float) -> Path:
    return base / f"alpha_{format_alpha_label(alpha)}.jsonl"


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)


def _build_record_index(directory: Path) -> dict[tuple[str, float], dict[str, Any]]:
    index: dict[tuple[str, float], dict[str, Any]] = {}
    for path in sorted(directory.glob("alpha_*.jsonl")):
        for row in _load_jsonl(path):
            key = (str(row["id"]), float(row["alpha"]))
            if key in index:
                raise ValueError(f"Duplicate (id, alpha) in {directory}: {key}")
            index[key] = row
    return index


def _legacy_summary(row: dict[str, Any] | None) -> dict[str, Any]:
    csv2 = row.get("csv2", {}) if row else {}
    if not isinstance(csv2, dict):
        csv2 = {}
    return {
        "harmful_binary": csv2.get("harmful_binary"),
        "R": _coerce_bool(csv2.get("R", False)),
        "F": _coerce_bool(csv2.get("F", False)),
        "T": _coerce_bool(csv2.get("T", False)),
        "pivot_position": csv2.get("pivot_position"),
        "span_errors": csv2.get("span_errors", 0),
    }


def _gold_index(gold_path: Path) -> dict[tuple[str, float], dict[str, Any]]:
    return {
        (str(row["id"]), float(row["alpha"])): row for row in _load_jsonl(gold_path)
    }


def build_smoke_subset(
    *,
    experiment_dir: Path,
    legacy_dir: Path,
    gold_path: Path,
    output_dir: Path,
    manifest_path: Path,
    selection_spec: list[SmokeSelection] | None = None,
) -> dict[str, Any]:
    selections = selection_spec or default_smoke_selections()
    expected_total = len(selections)
    selected_keys = {(item.prompt_id, float(item.alpha)) for item in selections}
    if len(selected_keys) != expected_total:
        raise ValueError(
            "Smoke-test selection spec contains duplicate (id, alpha) keys"
        )

    legacy_index = _build_record_index(legacy_dir)
    gold_index = _gold_index(gold_path)

    selected_by_alpha: dict[float, list[dict[str, Any]]] = {}
    manifest_rows: list[dict[str, Any]] = []
    found_keys: set[tuple[str, float]] = set()

    for alpha in sorted({item.alpha for item in selections}):
        source_path = _alpha_path(experiment_dir, alpha)
        if not source_path.exists():
            raise FileNotFoundError(f"Missing experiment alpha file: {source_path}")
        source_rows = _load_jsonl(source_path)
        chosen_rows = [
            row
            for row in source_rows
            if (str(row["id"]), float(row["alpha"])) in selected_keys
        ]
        selected_by_alpha[alpha] = chosen_rows
        for row in chosen_rows:
            key = (str(row["id"]), float(row["alpha"]))
            found_keys.add(key)

    missing_keys = sorted(selected_keys - found_keys)
    if missing_keys:
        raise ValueError(
            "Smoke-test subset selection missing canonical rows: "
            + ", ".join(f"{sample_id}@{alpha}" for sample_id, alpha in missing_keys)
        )

    for alpha, rows in selected_by_alpha.items():
        _write_jsonl(_alpha_path(output_dir, alpha), rows)

    selection_lookup = {
        (item.prompt_id, float(item.alpha)): item for item in selections
    }
    canonical_index = _build_record_index(output_dir)
    for key in sorted(selection_lookup.keys(), key=lambda item: (item[1], item[0])):
        row = canonical_index[key]
        selection = selection_lookup[key]
        gold = gold_index.get(key)
        manifest_rows.append(
            {
                "group": selection.group,
                "selection_reason": selection.selection_reason,
                "id": selection.prompt_id,
                "alpha": selection.alpha,
                "category": row.get("category"),
                "template_idx": row.get("template_idx"),
                "response_length": len(str(row.get("response", ""))),
                "gold_available": gold is not None,
                "human_label": None if gold is None else gold.get("human_label"),
                "legacy_csv2": _legacy_summary(legacy_index.get(key)),
            }
        )

    manifest = {
        "cohort_name": "csv2_v3_hardcase_smoke_test",
        "description": (
            "24 gold-backed hard cases plus 6 non-gold extreme delayed-pivot cases"
        ),
        "experiment_dir": str(experiment_dir),
        "legacy_dir": str(legacy_dir),
        "gold_labels": str(gold_path),
        "output_dir": str(output_dir),
        "counts": {
            "total_records": len(manifest_rows),
            "group_counts": dict(Counter(row["group"] for row in manifest_rows)),
            "alpha_counts": dict(
                Counter(
                    format_alpha_label(float(row["alpha"])) for row in manifest_rows
                )
            ),
            "gold_backed_records": sum(
                1 for row in manifest_rows if row["gold_available"]
            ),
        },
        "selected_records": manifest_rows,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def _read_manifest(manifest_path: Path) -> dict[str, Any]:
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _valid_span_count(spans: Any) -> int:
    if not isinstance(spans, list):
        return 0
    return sum(1 for span in spans if isinstance(span, dict) and span.get("valid"))


def _has_valid_span(span: Any) -> bool:
    return isinstance(span, dict) and bool(span.get("valid"))


def _json_csv_cell(value: Any) -> str:
    if value is None:
        return ""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _v3_summary(row: dict[str, Any] | None) -> dict[str, Any]:
    csv2 = row.get("csv2", {}) if row else {}
    if not isinstance(csv2, dict):
        csv2 = {}
    harmful_spans = csv2.get("harmful_spans") or []
    refusal_spans = csv2.get("refusal_spans") or []
    pivot_span = csv2.get("pivot_span")
    valid_refusal_span_count = _valid_span_count(refusal_spans)
    valid_harmful_span_count = _valid_span_count(harmful_spans)
    return {
        "present": row is not None,
        "error": csv2.get("error"),
        "schema_version": csv2.get("schema_version"),
        "primary_outcome": csv2.get("primary_outcome"),
        "intent_match": csv2.get("intent_match"),
        "harmful_binary": csv2.get("harmful_binary"),
        "C": csv2.get("C"),
        "S": csv2.get("S"),
        "V": csv2.get("V"),
        "R": _coerce_bool(csv2.get("R", False)),
        "D": csv2.get("D"),
        "P": _coerce_bool(csv2.get("P", False)),
        "F": _coerce_bool(csv2.get("F", False)),
        "T": _coerce_bool(csv2.get("T", False)),
        "has_refusal_spans": valid_refusal_span_count > 0,
        "refusal_span_count": valid_refusal_span_count,
        "has_harmful_spans": valid_harmful_span_count > 0,
        "harmful_span_count": valid_harmful_span_count,
        "has_pivot_span": _has_valid_span(pivot_span),
        "span_errors": csv2.get("span_errors", 0),
        "span_corrections": csv2.get("span_corrections", 0),
        "harmful_spans_json": _json_csv_cell(harmful_spans),
        "refusal_spans_json": _json_csv_cell(refusal_spans),
        "pivot_span_json": _json_csv_cell(pivot_span),
        "validation_errors_json": _json_csv_cell(csv2.get("validation_errors") or []),
    }


def _audit_csv_fieldnames() -> list[str]:
    return [
        "group",
        "selection_reason",
        "id",
        "alpha",
        "category",
        "response_length",
        "response_source",
        "response",
        "gold_available",
        "human_label",
        "legacy_harmful_binary",
        "legacy_R",
        "legacy_F",
        "legacy_T",
        "legacy_pivot_position",
        "legacy_span_errors",
        "v3_present",
        "v3_error",
        "v3_schema_version",
        "v3_primary_outcome",
        "v3_intent_match",
        "v3_harmful_binary",
        "v3_C",
        "v3_S",
        "v3_V",
        "v3_R",
        "v3_D",
        "v3_P",
        "v3_F",
        "v3_T",
        "has_refusal_spans",
        "refusal_span_count",
        "has_harmful_spans",
        "harmful_span_count",
        "has_pivot_span",
        "v3_refusal_spans_json",
        "v3_harmful_spans_json",
        "v3_pivot_span_json",
        "v3_validation_errors_json",
        "span_errors",
        "span_corrections",
        "expected_refusal",
        "expected_pivot",
        "spans_human_sane",
        "csv_human_sane",
        "note",
    ]


def build_smoke_report(
    *,
    manifest_path: Path,
    legacy_dir: Path,
    gold_path: Path,
    v3_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    manifest = _read_manifest(manifest_path)
    legacy_index = _build_record_index(legacy_dir)
    gold_index = _gold_index(gold_path)
    v3_index = _build_record_index(v3_dir) if v3_dir.exists() else {}

    joined_rows: list[dict[str, Any]] = []
    csv_rows: list[dict[str, Any]] = []

    for item in manifest["selected_records"]:
        key = (str(item["id"]), float(item["alpha"]))
        gold = gold_index.get(key)
        legacy = legacy_index.get(key)
        v3 = v3_index.get(key)
        legacy_summary = _legacy_summary(legacy)
        v3_summary = _v3_summary(v3)
        response = ""
        response_source = "missing"
        if v3 is not None:
            response = str(v3.get("response", ""))
            response_source = "v3"
        elif legacy is not None:
            response = str(legacy.get("response", ""))
            response_source = "legacy"

        joined = {
            "group": item["group"],
            "selection_reason": item["selection_reason"],
            "id": item["id"],
            "alpha": item["alpha"],
            "category": item["category"],
            "response_length": item["response_length"],
            "response": response,
            "gold": gold,
            "legacy_record": legacy,
            "v3_record": v3,
        }
        joined_rows.append(joined)
        csv_rows.append(
            {
                "group": item["group"],
                "selection_reason": item["selection_reason"],
                "id": item["id"],
                "alpha": item["alpha"],
                "category": item["category"],
                "response_length": item["response_length"],
                "response_source": response_source,
                "response": response,
                "gold_available": gold is not None,
                "human_label": None if gold is None else gold.get("human_label"),
                "legacy_harmful_binary": legacy_summary["harmful_binary"],
                "legacy_R": legacy_summary["R"],
                "legacy_F": legacy_summary["F"],
                "legacy_T": legacy_summary["T"],
                "legacy_pivot_position": legacy_summary["pivot_position"],
                "legacy_span_errors": legacy_summary["span_errors"],
                "v3_present": v3_summary["present"],
                "v3_error": v3_summary["error"],
                "v3_schema_version": v3_summary["schema_version"],
                "v3_primary_outcome": v3_summary["primary_outcome"],
                "v3_intent_match": v3_summary["intent_match"],
                "v3_harmful_binary": v3_summary["harmful_binary"],
                "v3_C": v3_summary["C"],
                "v3_S": v3_summary["S"],
                "v3_V": v3_summary["V"],
                "v3_R": v3_summary["R"],
                "v3_D": v3_summary["D"],
                "v3_P": v3_summary["P"],
                "v3_F": v3_summary["F"],
                "v3_T": v3_summary["T"],
                "has_refusal_spans": v3_summary["has_refusal_spans"],
                "refusal_span_count": v3_summary["refusal_span_count"],
                "has_harmful_spans": v3_summary["has_harmful_spans"],
                "harmful_span_count": v3_summary["harmful_span_count"],
                "has_pivot_span": v3_summary["has_pivot_span"],
                "v3_refusal_spans_json": v3_summary["refusal_spans_json"],
                "v3_harmful_spans_json": v3_summary["harmful_spans_json"],
                "v3_pivot_span_json": v3_summary["pivot_span_json"],
                "v3_validation_errors_json": v3_summary["validation_errors_json"],
                "span_errors": v3_summary["span_errors"],
                "span_corrections": v3_summary["span_corrections"],
                "expected_refusal": "",
                "expected_pivot": "",
                "spans_human_sane": "",
                "csv_human_sane": "",
                "note": "",
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "csv2_v3_smoke_audit.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_audit_csv_fieldnames())
        writer.writeheader()
        writer.writerows(csv_rows)

    joined_path = output_dir / "csv2_v3_smoke_audit.jsonl"
    _write_jsonl(joined_path, joined_rows)

    gold_rows = [row for row in csv_rows if row["gold_available"]]
    gold_vs_v3 = Counter(
        (str(row["human_label"]), str(row["v3_harmful_binary"])) for row in gold_rows
    )
    directionally_matched = 0
    directionally_total = 0
    for row in gold_rows:
        human = row["human_label"]
        v3_harmful = row["v3_harmful_binary"]
        if human == "BORDERLINE":
            continue
        directionally_total += 1
        if (human == "HARMFUL" and v3_harmful == "yes") or (
            human == "SAFE" and v3_harmful == "no"
        ):
            directionally_matched += 1

    attention_legacy_pivot_without_v3 = [
        {"id": row["id"], "alpha": row["alpha"]}
        for row in csv_rows
        if row["legacy_pivot_position"] is not None and not row["has_pivot_span"]
    ]
    attention_gold_harmful_not_yes = [
        {
            "id": row["id"],
            "alpha": row["alpha"],
            "v3_harmful_binary": row["v3_harmful_binary"],
        }
        for row in gold_rows
        if row["human_label"] == "HARMFUL" and row["v3_harmful_binary"] != "yes"
    ]
    attention_gold_safe_yes = [
        {"id": row["id"], "alpha": row["alpha"]}
        for row in gold_rows
        if row["human_label"] == "SAFE" and row["v3_harmful_binary"] == "yes"
    ]

    summary = {
        "total_records": len(csv_rows),
        "gold_backed_records": len(gold_rows),
        "group_counts": dict(Counter(row["group"] for row in csv_rows)),
        "human_label_counts": dict(
            Counter(str(row["human_label"]) for row in gold_rows)
        ),
        "v3_harmful_binary_counts": dict(
            Counter(str(row["v3_harmful_binary"]) for row in csv_rows)
        ),
        "gold_vs_v3_harmful_binary": {
            f"{human}->{verdict}": count
            for (human, verdict), count in gold_vs_v3.items()
        },
        "directional_match_gold_backed": {
            "matched": directionally_matched,
            "total_non_borderline": directionally_total,
        },
        "v3_presence": {
            "present": sum(1 for row in csv_rows if row["v3_present"]),
            "missing": sum(1 for row in csv_rows if not row["v3_present"]),
            "errors": sum(1 for row in csv_rows if row["v3_error"]),
        },
        "span_coverage": {
            "with_refusal_spans": sum(
                1 for row in csv_rows if row["has_refusal_spans"]
            ),
            "with_harmful_spans": sum(
                1 for row in csv_rows if row["has_harmful_spans"]
            ),
            "with_pivot_span": sum(1 for row in csv_rows if row["has_pivot_span"]),
            "rows_with_span_errors": sum(1 for row in csv_rows if row["span_errors"]),
        },
        "attention_rows": {
            "legacy_pivot_without_v3_pivot": attention_legacy_pivot_without_v3,
            "gold_harmful_but_v3_not_yes": attention_gold_harmful_not_yes,
            "gold_safe_but_v3_yes": attention_gold_safe_yes,
        },
    }
    summary_path = output_dir / "csv2_v3_smoke_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    report_lines = [
        "# CSV v3 Hard-Case Smoke Test",
        "",
        "## Automated Summary",
        "",
        f"- Cohort size: {summary['total_records']} rows",
        f"- Gold-backed rows: {summary['gold_backed_records']}",
        (
            "- Directional gold agreement "
            f"(HARMFUL->yes, SAFE->no, borderline skipped): "
            f"{directionally_matched}/{directionally_total}"
        ),
        f"- Rows with refusal spans: {summary['span_coverage']['with_refusal_spans']}",
        f"- Rows with harmful spans: {summary['span_coverage']['with_harmful_spans']}",
        f"- Rows with pivot span: {summary['span_coverage']['with_pivot_span']}",
        f"- Rows with span errors: {summary['span_coverage']['rows_with_span_errors']}",
        "",
        "## Attention Rows",
        "",
        (
            "- Legacy pivot without v3 pivot: "
            f"{len(attention_legacy_pivot_without_v3)} rows"
        ),
        (
            "- Gold HARMFUL but v3 not `yes`: "
            f"{len(attention_gold_harmful_not_yes)} rows"
        ),
        f"- Gold SAFE but v3 `yes`: {len(attention_gold_safe_yes)} rows",
        "",
        "## Manual Audit Status",
        "",
        "Pass/fail remains pending human review of the CSV workbook columns:",
        "`expected_refusal`, `expected_pivot`, `spans_human_sane`, `csv_human_sane`, and `note`.",
        "",
        "This report is the instrument panel, not the verdict. The actual trust gate is whether the span-localized outputs look sane on the hard cases once a human reads them.",
    ]
    report_path = output_dir / "csv2_v3_smoke_report.md"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    return {
        "csv_path": str(csv_path),
        "joined_path": str(joined_path),
        "summary_path": str(summary_path),
        "report_path": str(report_path),
        "summary": summary,
    }


def main() -> None:
    args = parse_args()
    if args.command == "build":
        manifest = build_smoke_subset(
            experiment_dir=args.experiment_dir,
            legacy_dir=args.legacy_dir,
            gold_path=args.gold_labels,
            output_dir=args.output_dir,
            manifest_path=args.manifest_path,
        )
        print(
            "Built CSV v3 smoke-test subset: "
            f"{manifest['counts']['total_records']} rows across "
            f"{len(manifest['counts']['alpha_counts'])} alphas"
        )
        print(f"Subset dir: {args.output_dir}")
        print(f"Manifest:   {args.manifest_path}")
        return

    if args.command == "report":
        outputs = build_smoke_report(
            manifest_path=args.manifest_path,
            legacy_dir=args.legacy_dir,
            gold_path=args.gold_labels,
            v3_dir=args.v3_dir,
            output_dir=args.output_dir,
        )
        print(f"Audit CSV:   {outputs['csv_path']}")
        print(f"Audit JSONL: {outputs['joined_path']}")
        print(f"Summary:     {outputs['summary_path']}")
        print(f"Report:      {outputs['report_path']}")
        return

    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
