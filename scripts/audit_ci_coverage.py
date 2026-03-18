from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "docs/ci_manifest.json"
PROVENANCE_PATTERN = re.compile(r"<!--\s*from:\s*([A-Za-z0-9_.-]+)\s*-->")


def load_json(path: str) -> Any:
    file_path = ROOT / path
    if not file_path.exists():
        raise FileNotFoundError(path)
    return json.loads(file_path.read_text())


def load_manifest() -> dict[str, Any]:
    return json.loads(MANIFEST_PATH.read_text())


def resolve_path(data: Any, path: tuple[str | int, ...]) -> Any:
    current = data
    for key in path:
        current = current[key]
    return current


def ensure_ci_block(
    errors: list[str],
    data: Any,
    path: tuple[str | int, ...],
    label: str,
) -> None:
    try:
        node = resolve_path(data, path)
    except (KeyError, IndexError, TypeError):
        errors.append(f"{label}: missing path {'/'.join(map(str, path))}")
        return

    if not isinstance(node, dict):
        errors.append(f"{label}: expected object at {'/'.join(map(str, path))}")
        return

    ci = node.get("ci")
    if not isinstance(ci, dict):
        errors.append(f"{label}: missing ci block at {'/'.join(map(str, path))}")
        return

    for key in ("lower", "upper", "level", "method"):
        if key not in ci:
            errors.append(f"{label}: ci missing '{key}' at {'/'.join(map(str, path))}")


def audit_classifier_summary(path: str, label: str, errors: list[str]) -> None:
    data = load_json(path)
    evaluation = data["evaluation"] if "evaluation" in data else data
    for metric in ("accuracy", "precision", "recall", "f1", "auroc"):
        ensure_ci_block(
            errors,
            evaluation,
            ("metrics", metric),
            f"{label}::{metric}",
        )


def audit_intervention_result(path: str, label: str, errors: list[str]) -> None:
    data = load_json(path)
    for alpha in ("0.0", "0.5", "1.0", "1.5", "2.0", "2.5", "3.0"):
        ensure_ci_block(
            errors,
            data,
            ("results", alpha, "compliance"),
            f"{label}::alpha={alpha}::compliance",
        )
    ensure_ci_block(
        errors,
        data,
        ("effects", "compliance_curve", "delta_0_to_max_pp"),
        f"{label}::delta_0_to_max_pp",
    )
    ensure_ci_block(
        errors,
        data,
        ("effects", "compliance_curve", "slope_pp_per_alpha"),
        f"{label}::slope_pp_per_alpha",
    )


def audit_site_intervention(path: str, errors: list[str]) -> None:
    data = load_json(path)
    for series_name in ("anti_compliance", "standard_raw"):
        series = data["series"][series_name]
        for idx, _point in enumerate(series["points"]):
            ensure_ci_block(
                errors,
                series,
                ("points", idx),
                f"site_intervention::{series_name}::point_{idx}",
            )
        ensure_ci_block(
            errors,
            series,
            ("effects", "delta_0_to_max_pp"),
            f"site_intervention::{series_name}::delta",
        )
        ensure_ci_block(
            errors,
            series,
            ("effects", "slope_pp_per_alpha"),
            f"site_intervention::{series_name}::slope",
        )

    for idx, _point in enumerate(data["parse_failures"]["points"]):
        ensure_ci_block(
            errors,
            data,
            ("parse_failures", "points", idx),
            f"site_intervention::parse_failures::point_{idx}",
        )

    ensure_ci_block(
        errors,
        data,
        (
            "series",
            "standard_text_remap",
            "by_alpha",
            "3.0",
            "strict_recovered_rate_summary",
        ),
        "site_intervention::standard_text_remap::strict_recovered_rate_summary",
    )
    ensure_ci_block(
        errors,
        data,
        (
            "series",
            "standard_text_remap",
            "by_alpha",
            "3.0",
            "strict_rescored_compliance_summary",
        ),
        "site_intervention::standard_text_remap::strict_rescored_compliance_summary",
    )

    for bucket in ("always_compliant", "never_compliant", "swing"):
        ensure_ci_block(
            errors,
            data,
            ("population", "anti_compliance", bucket),
            f"site_intervention::population::{bucket}",
        )


def audit_negative_control(path: str, errors: list[str]) -> None:
    data = load_json(path)
    comparison = data["comparison_to_h_neurons"]
    for path_key in (
        ("alpha_3_random_percentile_interval_pct",),
        ("slope_random_percentile_interval",),
    ):
        node = comparison[path_key[0]]
        if not isinstance(node, dict):
            errors.append(f"negative_control::{path_key[0]} missing interval object")
            continue
        for key in ("lower", "upper", "level", "method"):
            if key not in node:
                errors.append(f"negative_control::{path_key[0]} missing '{key}'")


def audit_text_surfaces(errors: list[str]) -> None:
    manifest = load_manifest()
    banned_phrases = tuple(manifest.get("banned_phrases", []))
    files = (
        "site/index.html",
        "site/story.html",
        "site/results/gemma-3-4b.html",
        "site/methods.html",
        "data/gemma3_4b/pipeline/pipeline_report.md",
        "data/gemma3_4b/intervention_findings.md",
        "docs/bluedot-rapid-grant-2026.md",
    )
    for file_path in files:
        text = (ROOT / file_path).read_text()
        for phrase in banned_phrases:
            if phrase in text:
                errors.append(f"{file_path}: contains banned phrase '{phrase}'")


def ensure_interval_block(errors: list[str], node: Any, label: str) -> None:
    if not isinstance(node, dict):
        errors.append(f"{label}: expected interval object")
        return
    for key in ("lower", "upper", "level", "method"):
        if key not in node:
            errors.append(f"{label}: missing '{key}'")


def ensure_estimate_like(errors: list[str], node: Any, label: str) -> None:
    if not isinstance(node, dict):
        errors.append(f"{label}: expected object with estimate-like field")
        return
    estimate_keys = {
        "estimate",
        "pct",
        "rate",
        "compliance_rate",
        "compliance_pct",
    }
    if not any(key in node for key in estimate_keys):
        errors.append(
            f"{label}: missing estimate-like field "
            f"(expected one of {sorted(estimate_keys)})"
        )


def audit_manifest_claims(errors: list[str]) -> None:
    manifest = load_manifest()
    claims = manifest.get("claims", [])
    claim_ids = {
        claim["id"] for claim in claims if isinstance(claim, dict) and "id" in claim
    }
    surface_files: set[str] = set()

    for claim in claims:
        claim_id = claim["id"]
        status = claim["status"]

        if status in {"blocked_data", "not_applicable"}:
            if not claim.get("reason"):
                errors.append(f"{claim_id}: missing reason for status '{status}'")
            continue

        if status != "required":
            errors.append(f"{claim_id}: unsupported status '{status}'")
            continue

        source = claim.get("source")
        if not isinstance(source, dict):
            errors.append(f"{claim_id}: missing source")
            continue

        file_path = source.get("file")
        path = tuple(source.get("path", []))
        if not isinstance(file_path, str) or not path:
            errors.append(f"{claim_id}: invalid source definition")
            continue

        try:
            node = resolve_path(load_json(file_path), path)
        except FileNotFoundError:
            errors.append(f"{claim_id}: missing source file '{file_path}'")
            continue
        except (KeyError, IndexError, TypeError):
            errors.append(f"{claim_id}: missing source path {'/'.join(map(str, path))}")
            continue

        kind = claim.get("kind", "estimate_with_ci")
        if kind == "estimate_with_ci":
            ensure_estimate_like(errors, node, claim_id)
            if isinstance(node, dict):
                ensure_interval_block(errors, node.get("ci"), f"{claim_id}::ci")
        elif kind == "interval_only":
            ensure_interval_block(errors, node, claim_id)
        else:
            errors.append(f"{claim_id}: unsupported kind '{kind}'")

        for surface in claim.get("surfaces", []):
            surface_file = surface.get("file")
            mode = surface.get("mode")
            if not isinstance(surface_file, str) or not isinstance(mode, str):
                errors.append(f"{claim_id}: invalid surface entry")
                continue
            surface_files.add(surface_file)
            if mode == "comment":
                text = (ROOT / surface_file).read_text()
                comment = f"<!-- from: {claim_id} -->"
                if comment not in text:
                    errors.append(
                        f"{claim_id}: missing provenance comment in '{surface_file}'"
                    )
            else:
                errors.append(f"{claim_id}: unsupported surface mode '{mode}'")

    for surface_file in surface_files:
        text = (ROOT / surface_file).read_text()
        for referenced_claim in PROVENANCE_PATTERN.findall(text):
            if referenced_claim not in claim_ids:
                errors.append(
                    f"{surface_file}: provenance comment references unknown claim "
                    f"'{referenced_claim}'"
                )


def main() -> int:
    errors: list[str] = []

    try:
        audit_classifier_summary(
            "data/gemma3_4b/pipeline/classifier_disjoint_summary.json",
            "classifier_disjoint",
            errors,
        )
        audit_classifier_summary(
            "data/gemma3_4b/pipeline/classifier_overlap_summary.json",
            "classifier_overlap",
            errors,
        )
        audit_classifier_summary(
            "data/gemma3_4b/probing/bioasq13b_factoid/classifier_summary.json",
            "bioasq_recovery",
            errors,
        )
        audit_classifier_summary(
            "site/data/classifier_summary.json",
            "site_classifier",
            errors,
        )
        audit_intervention_result(
            "data/gemma3_4b/intervention/faitheval/experiment/results.json",
            "faitheval_anti",
            errors,
        )
        audit_intervention_result(
            "data/gemma3_4b/intervention/falseqa/experiment/results.json",
            "falseqa",
            errors,
        )
        audit_site_intervention("site/data/intervention_sweep.json", errors)
        audit_negative_control(
            "data/gemma3_4b/intervention/faitheval/control/comparison_summary.json",
            errors,
        )
        audit_text_surfaces(errors)
        audit_manifest_claims(errors)
    except FileNotFoundError as exc:
        errors.append(f"missing required artifact: {exc}")

    if errors:
        print("CI audit failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    print("CI audit passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
