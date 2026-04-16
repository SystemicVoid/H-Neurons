from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, cast


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "docs/ci_manifest.json"
PROVENANCE_PATTERN = re.compile(r"<!--\s*from:\s*([A-Za-z0-9_.-]+)\s*-->")

WARN_PREFIX = "[WARN]"

# Fallback surface list for manifest schema_version < 2 (backwards compat).
_LEGACY_TEXT_SURFACES = (
    "site/index.html",
    "site/story.html",
    "site/results/gemma-3-4b.html",
    "site/methods.html",
    "site/extensions.html",
    "site/deep-dives/neuron-4288.html",
    "site/deep-dives/swing-characterization.html",
    "data/gemma3_4b/pipeline/pipeline_report.md",
    "data/gemma3_4b/intervention_findings.md",
)


# ---------------------------------------------------------------------------
# Low-level utilities
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# CI block primitives
# ---------------------------------------------------------------------------


def _validate_interval_values(
    errors: list[str],
    lower: Any,
    upper: Any,
    level: Any,
    method: Any,
    label: str,
) -> None:
    """Semantic checks shared by ensure_ci_block and ensure_interval_block."""
    if not isinstance(lower, (int, float)) or not isinstance(upper, (int, float)):
        errors.append(
            f"{label}: ci lower/upper must be numeric, got {lower!r}/{upper!r}"
        )
    elif lower > upper:
        errors.append(f"{label}: ci inverted — lower ({lower}) > upper ({upper})")

    if not isinstance(level, (int, float)) or not (0 < level < 1):
        errors.append(f"{label}: ci level must be float in (0,1), got {level!r}")

    if not isinstance(method, str) or not method.strip():
        errors.append(f"{label}: ci method must be a non-empty string, got {method!r}")


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

    _validate_interval_values(
        errors,
        ci.get("lower"),
        ci.get("upper"),
        ci.get("level"),
        ci.get("method"),
        label,
    )


def ensure_interval_block(errors: list[str], node: Any, label: str) -> None:
    if not isinstance(node, dict):
        errors.append(f"{label}: expected interval object")
        return
    for key in ("lower", "upper", "level", "method"):
        if key not in node:
            errors.append(f"{label}: missing '{key}'")
    _validate_interval_values(
        errors,
        node.get("lower"),
        node.get("upper"),
        node.get("level"),
        node.get("method"),
        label,
    )


def ensure_interval_array(errors: list[str], node: Any, label: str) -> None:
    if not isinstance(node, list) or len(node) != 2:
        errors.append(f"{label}: expected 2-item interval array")
        return
    if not all(isinstance(value, (int, float)) for value in node):
        errors.append(f"{label}: interval array must contain numeric values")
    elif node[0] > node[1]:
        errors.append(f"{label}: interval array inverted — [{node[0]}, {node[1]}]")


def ensure_estimate_like(errors: list[str], node: Any, label: str) -> None:
    if not isinstance(node, dict):
        errors.append(f"{label}: expected object with estimate-like field")
        return
    estimate_keys = {
        "estimate",
        "estimate_pp",
        "pct",
        "proportion",
        "rate",
        "compliance_rate",
        "compliance_pct",
    }
    if not any(key in node for key in estimate_keys):
        errors.append(
            f"{label}: missing estimate-like field "
            f"(expected one of {sorted(estimate_keys)})"
        )


# ---------------------------------------------------------------------------
# Classifier checks
# ---------------------------------------------------------------------------


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


def audit_classifier_structure_payload(
    structure: Any,
    selected_h_neurons: Any,
    label: str,
    errors: list[str],
    *,
    n_layers: int = 34,
    top_n_neurons: int = 10,
) -> None:
    if not isinstance(selected_h_neurons, int):
        errors.append(f"{label} selected_h_neurons must be an int")
        return

    if not isinstance(structure, dict):
        errors.append(f"{label} missing classifier structure payload")
        return

    counts = structure.get("positive_counts_by_layer")
    if not isinstance(counts, list) or len(counts) != n_layers:
        errors.append(f"{label} positive_counts_by_layer must be length {n_layers}")
        return
    if not all(isinstance(count, int) for count in counts):
        errors.append(f"{label} positive_counts_by_layer must contain ints")
        return
    typed_counts = [int(count) for count in counts]

    if sum(typed_counts) != selected_h_neurons:
        errors.append(
            f"{label} positive_counts_by_layer does not sum to selected_h_neurons"
        )

    nonzero_layers = structure.get("nonzero_layers")
    if not isinstance(nonzero_layers, list):
        errors.append(f"{label} missing nonzero_layers")
    else:
        expected_nonzero = [
            {"layer": layer, "count": count}
            for layer, count in enumerate(typed_counts)
            if count > 0
        ]
        if nonzero_layers != expected_nonzero:
            errors.append(
                f"{label} nonzero_layers disagrees with positive_counts_by_layer"
            )

    bands = structure.get("bands")
    if not isinstance(bands, dict):
        errors.append(f"{label} missing bands")
    else:
        band_total = 0
        for name in ("early", "middle", "late"):
            band = bands.get(name)
            if not isinstance(band, dict):
                errors.append(f"{label} missing bands.{name}")
                continue
            count = band.get("count")
            if not isinstance(count, int):
                errors.append(f"{label} bands.{name}.count invalid")
                continue
            band_total += count
        if band_total != selected_h_neurons:
            errors.append(f"{label} band counts do not sum to selected_h_neurons")

    top_positive_neurons = structure.get("top_positive_neurons")
    if (
        not isinstance(top_positive_neurons, list)
        or len(top_positive_neurons) != top_n_neurons
    ):
        errors.append(f"{label} top_positive_neurons must have length {top_n_neurons}")
        return

    previous_weight: float | None = None
    for idx, neuron in enumerate(top_positive_neurons):
        if not isinstance(neuron, dict):
            errors.append(f"{label} top_positive_neurons[{idx}] invalid")
            continue
        typed_neuron = cast(dict[str, object], neuron)
        weight = typed_neuron.get("weight")
        if not isinstance(weight, (int, float)) or weight <= 0:
            errors.append(
                f"{label} top_positive_neurons weights must be positive numbers"
            )
            continue
        if previous_weight is not None and weight > previous_weight:
            errors.append(
                f"{label} top_positive_neurons must be sorted descending by weight"
            )
        previous_weight = float(weight)


def audit_tracked_classifier_structure(
    path: str,
    errors: list[str],
    model_cfg: dict[str, Any] | None = None,
) -> None:
    if model_cfg is None:
        model_cfg = {}
    data = load_json(path)
    structure = data.get("structure")
    n_layers = model_cfg.get("n_layers", 34)
    top_n_neurons = model_cfg.get("top_n_neurons", 10)
    audit_classifier_structure_payload(
        structure,
        data.get("selected_h_neurons"),
        "tracked_classifier_structure::structure",
        errors,
        n_layers=n_layers,
        top_n_neurons=top_n_neurons,
    )

    for field in (
        "generated_at",
        "generated_by",
        "model",
        "model_path",
        "generation_script",
    ):
        if not isinstance(data.get(field), str) or not data[field]:
            errors.append(f"tracked_classifier_structure::{field} missing or invalid")
    coefficient_sha256 = data.get("coefficient_sha256")
    if not isinstance(coefficient_sha256, str) or len(coefficient_sha256) != 64:
        errors.append(
            "tracked_classifier_structure::coefficient_sha256 missing or invalid"
        )


def audit_site_classifier_structure(
    path: str,
    errors: list[str],
    model_cfg: dict[str, Any] | None = None,
) -> None:
    if model_cfg is None:
        model_cfg = {}
    data = load_json(path)
    n_layers = model_cfg.get("n_layers", 34)
    top_n_neurons = model_cfg.get("top_n_neurons", 10)
    audit_classifier_structure_payload(
        data.get("selected_h_neuron_structure"),
        data.get("selected_h_neurons"),
        "site_classifier::structure",
        errors,
        n_layers=n_layers,
        top_n_neurons=top_n_neurons,
    )


def audit_site_classifier_structure_consistency(
    tracked_path: str,
    site_path: str,
    errors: list[str],
) -> None:
    tracked = load_json(tracked_path)
    site = load_json(site_path)
    if tracked.get("structure") != site.get("selected_h_neuron_structure"):
        errors.append(
            "site_classifier::structure does not match tracked classifier structure summary"
        )


# ---------------------------------------------------------------------------
# Top-neuron artifact checks
# ---------------------------------------------------------------------------


def audit_top_neuron_artifact_summary(
    summary: Any,
    label: str,
    errors: list[str],
) -> None:
    if not isinstance(summary, dict):
        errors.append(f"{label} missing top-neuron artifact summary")
        return

    target = summary.get("target_neuron")
    if not isinstance(target, dict):
        errors.append(f"{label} missing target_neuron")
    else:
        for field in ("layer", "neuron", "label", "weight", "weight_rank"):
            if field not in target:
                errors.append(f"{label} target_neuron missing '{field}'")

    verdict = summary.get("verdict")
    if not isinstance(verdict, dict):
        errors.append(f"{label} missing verdict")
    else:
        for field in (
            "status",
            "supporting_tests",
            "total_tests",
            "summary",
            "ci_status",
        ):
            if field not in verdict:
                errors.append(f"{label} verdict missing '{field}'")

    tests = summary.get("tests")
    if not isinstance(tests, list) or not tests:
        errors.append(f"{label} missing tests")
    else:
        slugs: set[str] = set()
        for idx, test in enumerate(tests):
            if not isinstance(test, dict):
                errors.append(f"{label} tests[{idx}] invalid")
                continue
            typed_test = cast(dict[str, Any], test)
            for field in ("slug", "label", "display_value", "threshold", "verdict"):
                if field not in typed_test:
                    errors.append(f"{label} tests[{idx}] missing '{field}'")
            slug = typed_test.get("slug")
            if isinstance(slug, str):
                if slug in slugs:
                    errors.append(f"{label} duplicate test slug '{slug}'")
                slugs.add(slug)

    context = summary.get("distributed_detector_context")
    if not isinstance(context, dict):
        errors.append(f"{label} missing distributed_detector_context")
    else:
        for bucket in ("sparse_baseline", "broader_detector", "loosest_detector"):
            if not isinstance(context.get(bucket), dict):
                errors.append(f"{label} missing distributed_detector_context.{bucket}")


def audit_tracked_top_neuron_artifact_summary(path: str, errors: list[str]) -> None:
    data = load_json(path)
    audit_top_neuron_artifact_summary(data, "tracked_top_neuron_artifact", errors)


def audit_site_top_neuron_artifact_summary(path: str, errors: list[str]) -> None:
    data = load_json(path)
    audit_top_neuron_artifact_summary(
        data.get("top_neuron_artifact_summary"),
        "site_top_neuron_artifact",
        errors,
    )

    site_structure = data.get("selected_h_neuron_structure", {})
    top_neurons = site_structure.get("top_positive_neurons", [])
    artifact = data.get("top_neuron_artifact_summary", {})
    target = artifact.get("target_neuron", {}) if isinstance(artifact, dict) else {}
    if top_neurons and isinstance(target, dict):
        if target.get("label") != top_neurons[0].get("label"):
            errors.append(
                "site_top_neuron_artifact target label does not match classifier top neuron"
            )


def audit_site_top_neuron_artifact_consistency(
    tracked_path: str,
    site_path: str,
    errors: list[str],
) -> None:
    tracked = load_json(tracked_path)
    site = load_json(site_path)
    if tracked != site.get("top_neuron_artifact_summary"):
        errors.append(
            "site_top_neuron_artifact does not match tracked neuron_4288 summary"
        )


# ---------------------------------------------------------------------------
# Composite site-classifier check
# ---------------------------------------------------------------------------


def audit_site_classifier_full(
    check: dict[str, Any],
    errors: list[str],
    model_cfg: dict[str, Any],
) -> None:
    """All sub-checks for site/data/classifier_summary.json in one place."""
    site_path = check["file"]
    tracked_structure_file = check["tracked_structure_file"]
    tracked_neuron_file = check["tracked_neuron_file"]

    audit_classifier_summary(site_path, "site_classifier", errors)
    audit_site_classifier_structure(site_path, errors, model_cfg)
    audit_site_classifier_structure_consistency(
        tracked_structure_file, site_path, errors
    )
    audit_site_top_neuron_artifact_summary(site_path, errors)
    audit_site_top_neuron_artifact_consistency(tracked_neuron_file, site_path, errors)


# ---------------------------------------------------------------------------
# Intervention checks
# ---------------------------------------------------------------------------


def audit_intervention_result(
    path: str,
    label: str,
    errors: list[str],
    alpha_grid: list[str] | None = None,
) -> None:
    data = load_json(path)
    if alpha_grid is None:
        # Derive from data so any alpha grid works without manifest changes.
        alpha_grid = list(data.get("results", {}).keys())
    for alpha in alpha_grid:
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
    comparison = data.get("comparison_to_h_neurons")
    if not isinstance(comparison, dict):
        errors.append("negative_control: missing comparison_to_h_neurons")
        return
    for key in (
        "alpha_3_random_percentile_interval_pct",
        "slope_random_percentile_interval",
    ):
        node = comparison.get(key)
        if not isinstance(node, dict):
            errors.append(f"negative_control::{key} missing interval object")
            continue
        for field in ("lower", "upper", "level", "method"):
            if field not in node:
                errors.append(f"negative_control::{key} missing '{field}'")


# ---------------------------------------------------------------------------
# Text surface scan
# ---------------------------------------------------------------------------


def audit_text_surfaces(errors: list[str], manifest: dict[str, Any]) -> None:
    banned_phrases = tuple(manifest.get("banned_phrases", []))
    files = manifest.get("text_surfaces", _LEGACY_TEXT_SURFACES)
    for file_path in files:
        full = ROOT / file_path
        if not full.exists():
            errors.append(f"text_surfaces: missing file '{file_path}'")
            continue
        text = full.read_text()
        for phrase in banned_phrases:
            if phrase in text:
                errors.append(f"{file_path}: contains banned phrase '{phrase}'")


# ---------------------------------------------------------------------------
# Manifest claim checks
# ---------------------------------------------------------------------------


def audit_manifest_claims(errors: list[str], manifest: dict[str, Any]) -> None:
    claims = manifest.get("claims", [])
    claim_ids = {
        claim["id"] for claim in claims if isinstance(claim, dict) and "id" in claim
    }
    surface_files: set[str] = set()
    surface_text_cache: dict[str, str] = {}

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

        if not claim.get("surfaces"):
            errors.append(
                f"{WARN_PREFIX} {claim_id}: required claim has no registered surfaces"
            )

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
                if "ci" in node:
                    ensure_interval_block(errors, node.get("ci"), f"{claim_id}::ci")
                elif "ci_pp" in node:
                    ensure_interval_block(
                        errors, node.get("ci_pp"), f"{claim_id}::ci_pp"
                    )
                elif "ci_95" in node:
                    ensure_interval_array(
                        errors, node.get("ci_95"), f"{claim_id}::ci_95"
                    )
                else:
                    errors.append(f"{claim_id}: missing ci block")
        elif kind == "interval_only":
            ensure_interval_block(errors, node, claim_id)
        elif kind == "descriptive_value":
            if not isinstance(node, (str, int, float, bool)):
                errors.append(f"{claim_id}: descriptive_value must resolve to a scalar")
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
                if surface_file not in surface_text_cache:
                    surface_text_cache[surface_file] = (ROOT / surface_file).read_text()
                text = surface_text_cache[surface_file]
                comment = f"<!-- from: {claim_id} -->"
                if comment not in text:
                    errors.append(
                        f"{claim_id}: missing provenance comment in '{surface_file}'"
                    )
            else:
                errors.append(f"{claim_id}: unsupported surface mode '{mode}'")

    for surface_file in surface_files:
        if surface_file not in surface_text_cache:
            surface_text_cache[surface_file] = (ROOT / surface_file).read_text()
        for referenced_claim in PROVENANCE_PATTERN.findall(
            surface_text_cache[surface_file]
        ):
            if referenced_claim not in claim_ids:
                errors.append(
                    f"{surface_file}: provenance comment references unknown claim "
                    f"'{referenced_claim}'"
                )


# ---------------------------------------------------------------------------
# Manifest-driven dispatch
# ---------------------------------------------------------------------------


def dispatch_structural_check(
    check: dict[str, Any],
    manifest: dict[str, Any],
    errors: list[str],
) -> None:
    t = check["type"]
    model_cfg: dict[str, Any] = manifest.get("model_config", {}).get(
        check.get("model", ""), {}
    )

    if t == "classifier_summary":
        audit_classifier_summary(check["file"], check["label"], errors)
    elif t == "tracked_classifier_structure":
        audit_tracked_classifier_structure(check["file"], errors, model_cfg)
    elif t == "tracked_top_neuron_artifact":
        audit_tracked_top_neuron_artifact_summary(check["file"], errors)
    elif t == "site_classifier":
        audit_site_classifier_full(check, errors, model_cfg)
    elif t == "intervention_result":
        alpha_grid: list[str] | None = check.get(
            "alpha_grid_override"
        ) or model_cfg.get("canonical_alpha_grid")
        audit_intervention_result(check["file"], check["label"], errors, alpha_grid)
    elif t == "site_intervention":
        audit_site_intervention(check["file"], errors)
    elif t == "negative_control":
        audit_negative_control(check["file"], errors)
    else:
        errors.append(f"unknown structural check type: {t!r}")


def run_check(
    check: dict[str, Any],
    manifest: dict[str, Any],
    errors: list[str],
) -> None:
    """Run one structural check. Appends errors; never raises."""
    check_label = check.get("label", check.get("type", "unknown"))
    try:
        dispatch_structural_check(check, manifest, errors)
    except FileNotFoundError as exc:
        errors.append(f"[{check_label}] missing file: {exc}")
    except Exception as exc:  # noqa: BLE001
        errors.append(f"[{check_label}] unexpected error: {exc}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> int:
    errors: list[str] = []
    manifest = load_manifest()

    for check in manifest.get("structural_checks", []):
        run_check(check, manifest, errors)

    try:
        audit_text_surfaces(errors, manifest)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"[text_surfaces] unexpected error: {exc}")

    try:
        audit_manifest_claims(errors, manifest)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"[manifest_claims] unexpected error: {exc}")

    warnings = [e for e in errors if e.startswith(WARN_PREFIX)]
    blocking = [e for e in errors if not e.startswith(WARN_PREFIX)]

    if blocking:
        print("CI audit failed:", file=sys.stderr)
        for error in blocking:
            print(f"  - {error}", file=sys.stderr)
    if warnings:
        print("CI audit warnings:", file=sys.stderr)
        for warning in warnings:
            print(f"  - {warning}", file=sys.stderr)

    if not blocking and not warnings:
        print("CI audit passed.")
    elif not blocking:
        print("CI audit passed (with warnings).")

    return 1 if blocking else 0


if __name__ == "__main__":
    raise SystemExit(main())
