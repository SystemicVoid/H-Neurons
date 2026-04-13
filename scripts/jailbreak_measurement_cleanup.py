"""Helpers for staged jailbreak measurement cleanup and canary validation."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_STATE_ROOT = Path("data/judge_validation/jailbreak_measurement_cleanup")
DEFAULT_H_NEURON_SOURCE = Path("data/gemma3_4b/intervention/jailbreak/experiment")
DEFAULT_SEED1_SOURCE = Path(
    "data/gemma3_4b/intervention/jailbreak/control/seed_1_unconstrained"
)
DEFAULT_ALPHAS = [0.0, 1.0, 1.5, 3.0]
DEFAULT_CANARY_ROWS = 20
CSV2_SCHEMA_VERSION = "csv2_v3"


def _job_specs(h_neuron_source: Path, seed1_source: Path) -> dict[str, Path]:
    return {
        "h_neuron": h_neuron_source,
        "seed_1_control": seed1_source,
    }


def _alpha_filename(alpha: float) -> str:
    return f"alpha_{alpha:.1f}.jsonl"


def _state_subdir(state_root: Path, group: str, job_name: str) -> Path:
    return state_root / group / job_name


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _response_excerpt(text: str, limit: int = 220) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


def build_canary(
    state_root: Path,
    *,
    h_neuron_source: Path = DEFAULT_H_NEURON_SOURCE,
    seed1_source: Path = DEFAULT_SEED1_SOURCE,
    alphas: list[float] | None = None,
    canary_rows: int = DEFAULT_CANARY_ROWS,
) -> Path:
    """Materialize first-N canary subsets for both source jobs."""
    alphas = list(DEFAULT_ALPHAS if alphas is None else alphas)
    jobs = _job_specs(h_neuron_source, seed1_source)

    manifest: dict[str, Any] = {
        "schema_version": "jailbreak_measurement_cleanup_canary_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "canary_rows_per_alpha": canary_rows,
        "alphas": alphas,
        "jobs": {},
    }

    for job_name, source_dir in jobs.items():
        job_entry = {
            "source_dir": str(source_dir),
            "input_dir": str(_state_subdir(state_root, "canary_inputs", job_name)),
            "per_alpha_counts": {},
            "selected_ids": {},
        }
        for alpha in alphas:
            source_path = source_dir / _alpha_filename(alpha)
            records = _load_jsonl(source_path)[:canary_rows]
            output_path = _state_subdir(
                state_root, "canary_inputs", job_name
            ) / _alpha_filename(alpha)
            _write_jsonl(output_path, records)
            alpha_key = f"{alpha:.1f}"
            job_entry["per_alpha_counts"][alpha_key] = len(records)
            job_entry["selected_ids"][alpha_key] = [record["id"] for record in records]
        manifest["jobs"][job_name] = job_entry

    manifest_path = state_root / "canary_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest_path


def _summarize_outcomes(records: list[dict[str, Any]]) -> dict[str, int]:
    counter = Counter()
    for record in records:
        csv2 = record.get("csv2") or {}
        if not isinstance(csv2, dict):
            counter["missing_csv2"] += 1
            continue
        if csv2.get("error"):
            counter[f"error:{csv2['error']}"] += 1
        outcome = csv2.get("primary_outcome")
        if outcome:
            counter[f"primary_outcome:{outcome}"] += 1
        harmful = csv2.get("harmful_binary")
        if harmful:
            counter[f"harmful_binary:{harmful}"] += 1
    return dict(sorted(counter.items()))


def _collect_manual_examples(
    output_records: list[dict[str, Any]], limit: int = 6
) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for record in output_records:
        csv2 = record.get("csv2") or {}
        if not isinstance(csv2, dict):
            continue
        if not (
            csv2.get("R")
            or csv2.get("primary_outcome")
            in {"partial_compliance", "deflection", "refusal"}
            or csv2.get("D", 0) > 0
        ):
            continue
        examples.append(
            {
                "id": record.get("id"),
                "alpha": record.get("alpha"),
                "primary_outcome": csv2.get("primary_outcome"),
                "harmful_binary": csv2.get("harmful_binary"),
                "intent_match": csv2.get("intent_match"),
                "response_excerpt": _response_excerpt(record.get("response", "")),
            }
        )
        if len(examples) >= limit:
            break
    return examples


def _validate_pair(
    input_records: list[dict[str, Any]],
    output_records: list[dict[str, Any]],
) -> tuple[list[str], dict[str, int]]:
    failures: list[str] = []
    parse_stats = Counter()

    if len(output_records) != len(input_records):
        failures.append(
            f"row_count_mismatch:{len(output_records)}!={len(input_records)}"
        )
        return failures, dict(parse_stats)

    input_keys = [(record.get("id"), record.get("alpha")) for record in input_records]
    output_keys = [(record.get("id"), record.get("alpha")) for record in output_records]
    if output_keys != input_keys:
        failures.append("join_key_order_mismatch")

    for idx, output_record in enumerate(output_records):
        csv2 = output_record.get("csv2")
        if not isinstance(csv2, dict):
            failures.append(f"missing_csv2:{idx}")
            parse_stats["missing_csv2"] += 1
            continue
        if csv2.get("error"):
            failures.append(f"csv2_error:{idx}:{csv2['error']}")
            parse_stats[f"error:{csv2['error']}"] += 1
        if csv2.get("schema_version") != CSV2_SCHEMA_VERSION:
            failures.append(f"schema_version:{idx}:{csv2.get('schema_version')!r}")
        validation_errors = csv2.get("validation_errors") or []
        if validation_errors:
            failures.append(f"validation_errors:{idx}:{','.join(validation_errors)}")
            parse_stats["validation_errors"] += 1
        if csv2.get("span_errors", 0):
            parse_stats["span_errors"] += int(csv2["span_errors"])

    return failures, dict(parse_stats)


def validate_canary(
    state_root: Path,
    *,
    alphas: list[float] | None = None,
) -> dict[str, Any]:
    """Validate canary outputs and emit machine + human-readable summaries."""
    alphas = list(DEFAULT_ALPHAS if alphas is None else alphas)
    summary: dict[str, Any] = {
        "schema_version": "jailbreak_measurement_cleanup_validation_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "passed": True,
        "alphas": alphas,
        "jobs": {},
    }

    for job_name in ("h_neuron", "seed_1_control"):
        job_summary: dict[str, Any] = {
            "input_dir": str(_state_subdir(state_root, "canary_inputs", job_name)),
            "output_dir": str(_state_subdir(state_root, "canary_v3", job_name)),
            "alphas": {},
            "overall_failures": [],
            "manual_examples": [],
        }
        collected_examples: list[dict[str, Any]] = []

        for alpha in alphas:
            input_path = _state_subdir(
                state_root, "canary_inputs", job_name
            ) / _alpha_filename(alpha)
            output_path = _state_subdir(
                state_root, "canary_v3", job_name
            ) / _alpha_filename(alpha)
            if not input_path.exists():
                failures = [f"missing_input:{input_path}"]
                alpha_summary = {
                    "expected_rows": 0,
                    "actual_rows": 0,
                    "failures": failures,
                    "parse_stats": {},
                    "outcomes": {},
                }
                summary["passed"] = False
                job_summary["overall_failures"].extend(failures)
                job_summary["alphas"][f"{alpha:.1f}"] = alpha_summary
                continue
            if not output_path.exists():
                input_records = _load_jsonl(input_path)
                failures = [f"missing_output:{output_path}"]
                alpha_summary = {
                    "expected_rows": len(input_records),
                    "actual_rows": 0,
                    "failures": failures,
                    "parse_stats": {"missing_output": 1},
                    "outcomes": {},
                }
                summary["passed"] = False
                job_summary["overall_failures"].extend(failures)
                job_summary["alphas"][f"{alpha:.1f}"] = alpha_summary
                continue

            input_records = _load_jsonl(input_path)
            output_records = _load_jsonl(output_path)
            failures, parse_stats = _validate_pair(input_records, output_records)
            outcomes = _summarize_outcomes(output_records)
            alpha_summary = {
                "expected_rows": len(input_records),
                "actual_rows": len(output_records),
                "failures": failures,
                "parse_stats": parse_stats,
                "outcomes": outcomes,
            }
            if failures:
                summary["passed"] = False
                job_summary["overall_failures"].extend(failures)
            if len(collected_examples) < 6:
                remaining = 6 - len(collected_examples)
                collected_examples.extend(
                    _collect_manual_examples(output_records, limit=remaining)
                )
            job_summary["alphas"][f"{alpha:.1f}"] = alpha_summary

        job_summary["manual_examples"] = collected_examples
        summary["jobs"][job_name] = job_summary

    summary_path = state_root / "canary_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    report_path = state_root / "canary_report.md"
    report_path.write_text(_build_canary_report(summary))
    return summary


def validate_scored_dir(
    input_dir: Path,
    output_dir: Path,
    *,
    alphas: list[float] | None = None,
) -> dict[str, Any]:
    """Validate a scored CSV2 directory against its source records."""
    alphas = list(DEFAULT_ALPHAS if alphas is None else alphas)
    summary: dict[str, Any] = {
        "schema_version": "jailbreak_measurement_cleanup_scored_dir_validation_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "passed": True,
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "alphas": alphas,
        "per_alpha": {},
        "overall_failures": [],
    }

    for alpha in alphas:
        alpha_key = f"{alpha:.1f}"
        input_path = input_dir / _alpha_filename(alpha)
        output_path = output_dir / _alpha_filename(alpha)

        if not input_path.exists():
            failures = [f"missing_input:{input_path}"]
            alpha_summary = {
                "expected_rows": 0,
                "actual_rows": 0,
                "failures": failures,
                "parse_stats": {},
                "outcomes": {},
            }
            summary["passed"] = False
            summary["overall_failures"].extend(failures)
            summary["per_alpha"][alpha_key] = alpha_summary
            continue

        input_records = _load_jsonl(input_path)
        if not output_path.exists():
            failures = [f"missing_output:{output_path}"]
            alpha_summary = {
                "expected_rows": len(input_records),
                "actual_rows": 0,
                "failures": failures,
                "parse_stats": {"missing_output": 1},
                "outcomes": {},
            }
            summary["passed"] = False
            summary["overall_failures"].extend(failures)
            summary["per_alpha"][alpha_key] = alpha_summary
            continue

        output_records = _load_jsonl(output_path)
        failures, parse_stats = _validate_pair(input_records, output_records)
        outcomes = _summarize_outcomes(output_records)
        alpha_summary = {
            "expected_rows": len(input_records),
            "actual_rows": len(output_records),
            "failures": failures,
            "parse_stats": parse_stats,
            "outcomes": outcomes,
        }
        if failures:
            summary["passed"] = False
            summary["overall_failures"].extend(failures)
        summary["per_alpha"][alpha_key] = alpha_summary

    return summary


def _build_canary_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Jailbreak Measurement Cleanup Canary",
        "",
        f"- Generated: `{summary['generated_at']}`",
        f"- Passed: `{summary['passed']}`",
        f"- Alphas: `{', '.join(f'{alpha:.1f}' for alpha in summary['alphas'])}`",
        "",
    ]

    for job_name, job_summary in summary["jobs"].items():
        lines.extend(
            [
                f"## {job_name}",
                "",
                f"- Input dir: `{job_summary['input_dir']}`",
                f"- Output dir: `{job_summary['output_dir']}`",
                f"- Overall failures: `{len(job_summary['overall_failures'])}`",
                "",
                "| Alpha | Rows | Failures | Parse stats | Outcomes |",
                "|---|---:|---|---|---|",
            ]
        )
        for alpha_key, alpha_summary in job_summary["alphas"].items():
            failures = ", ".join(alpha_summary["failures"]) or "none"
            parse_stats = (
                ", ".join(
                    f"{key}={value}"
                    for key, value in sorted(alpha_summary["parse_stats"].items())
                )
                or "none"
            )
            outcomes = (
                ", ".join(
                    f"{key}={value}"
                    for key, value in sorted(alpha_summary["outcomes"].items())
                )
                or "none"
            )
            lines.append(
                f"| {alpha_key} | {alpha_summary['actual_rows']}/{alpha_summary['expected_rows']} | "
                f"{failures} | {parse_stats} | {outcomes} |"
            )

        lines.extend(["", "### Manual sanity slice", ""])
        examples = job_summary["manual_examples"]
        if not examples:
            lines.append(
                "_No refusal-like or borderline-like examples captured in canary._"
            )
        else:
            for example in examples:
                lines.extend(
                    [
                        (
                            f"- `{example['id']}` alpha={example['alpha']:.1f} "
                            f"outcome=`{example['primary_outcome']}` "
                            f"harmful=`{example['harmful_binary']}` "
                            f"intent_match=`{example['intent_match']}`"
                        ),
                        f"  Raw excerpt: {example['response_excerpt']}",
                    ]
                )
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def require_canary_pass(state_root: Path) -> dict[str, Any]:
    summary_path = state_root / "canary_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(
            f"Canary summary not found: {summary_path}. Run validate-canary first."
        )
    summary = json.loads(summary_path.read_text())
    if not summary.get("passed"):
        raise ValueError(
            f"Canary did not pass. Review {state_root / 'canary_report.md'} before continuing."
        )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Helpers for staged jailbreak measurement cleanup"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build-canary", help="Materialize canary subsets")
    build.add_argument("--state-root", type=Path, default=DEFAULT_STATE_ROOT)
    build.add_argument("--h-neuron-source", type=Path, default=DEFAULT_H_NEURON_SOURCE)
    build.add_argument("--seed1-source", type=Path, default=DEFAULT_SEED1_SOURCE)
    build.add_argument("--alphas", type=float, nargs="+", default=DEFAULT_ALPHAS)
    build.add_argument("--canary-rows", type=int, default=DEFAULT_CANARY_ROWS)

    validate = subparsers.add_parser(
        "validate-canary", help="Validate canary v3 outputs and write reports"
    )
    validate.add_argument("--state-root", type=Path, default=DEFAULT_STATE_ROOT)
    validate.add_argument("--alphas", type=float, nargs="+", default=DEFAULT_ALPHAS)

    validate_scored = subparsers.add_parser(
        "validate-scored-dir",
        help="Validate a scored CSV2 directory against its source records",
    )
    validate_scored.add_argument("--input-dir", type=Path, required=True)
    validate_scored.add_argument("--output-dir", type=Path, required=True)
    validate_scored.add_argument(
        "--alphas", type=float, nargs="+", default=DEFAULT_ALPHAS
    )

    require = subparsers.add_parser(
        "require-canary-pass", help="Exit non-zero unless canary_summary says pass"
    )
    require.add_argument("--state-root", type=Path, default=DEFAULT_STATE_ROOT)

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "build-canary":
        manifest_path = build_canary(
            args.state_root,
            h_neuron_source=args.h_neuron_source,
            seed1_source=args.seed1_source,
            alphas=list(args.alphas),
            canary_rows=args.canary_rows,
        )
        print(f"Built canary manifest: {manifest_path}")
        return 0
    if args.command == "validate-canary":
        summary = validate_canary(args.state_root, alphas=list(args.alphas))
        print(f"Canary passed: {summary['passed']}")
        print(f"Summary: {args.state_root / 'canary_summary.json'}")
        print(f"Report: {args.state_root / 'canary_report.md'}")
        return 0 if summary["passed"] else 1
    if args.command == "validate-scored-dir":
        summary = validate_scored_dir(
            args.input_dir,
            args.output_dir,
            alphas=list(args.alphas),
        )
        print(f"Scored dir passed: {summary['passed']}")
        print(f"Input: {args.input_dir}")
        print(f"Output: {args.output_dir}")
        for alpha_key, alpha_summary in summary["per_alpha"].items():
            failure_text = ", ".join(alpha_summary["failures"]) or "none"
            print(
                f"alpha={alpha_key} rows={alpha_summary['actual_rows']}/"
                f"{alpha_summary['expected_rows']} failures={failure_text}"
            )
        return 0 if summary["passed"] else 1

    require_canary_pass(args.state_root)
    print(f"Canary pass confirmed: {args.state_root / 'canary_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
