"""Shared Run Contract helpers for claim-bearing measurement outputs."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from utils import fingerprint_ids, format_alpha_label

INTERVENTION_RUN_CONFIG_FILENAME = "intervention_run_config.json"
CONTROL_RUN_CONFIG_FILENAME = "control_run_config.json"
INTERVENTION_RUN_CONFIG_SCHEMA_VERSION = "intervention_run_config/v1"
NEGATIVE_CONTROL_RUN_CONFIG_SCHEMA_VERSION = "negative_control_run_config/v1"
H_NEURON_BASELINE_BINDING_SCHEMA_VERSION = "h_neuron_baseline_binding/v1"
H_NEURON_BASELINE_INTERVENTION_MODE = "neuron"
MISMATCH_PREVIEW_LIMIT = 8
CONTROL_ONLY_BASELINE_BENCHMARK_CONFIG_KEYS = {
    "jailbreak": frozenset({"jailbreak_batch_size"}),
}
INTERVENTION_ONLY_BASELINE_BENCHMARK_CONFIG_KEYS = {
    "jailbreak": frozenset({"jailbreak_source", "jailbreak_path", "n_templates"}),
}


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def optional_file_sha256(path: str | Path | None) -> str | None:
    if not path:
        return None
    expanded = Path(path).expanduser()
    if not expanded.is_file():
        return None
    return file_sha256(expanded)


def describe_sample_manifest(path: str) -> dict[str, Any]:
    """Return content identity for a sample manifest, validating declared fields."""
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)

    if isinstance(payload, list):
        ids = [str(item) for item in payload]
        metadata: dict[str, Any] = {}
    elif isinstance(payload, dict):
        metadata = {}
        for key in ("ids", "sample_ids"):
            manifest_ids = payload.get(key)
            if isinstance(manifest_ids, list):
                ids = [str(item) for item in manifest_ids]
                break
        else:
            raise ValueError(
                "sample manifest must be a JSON list or an object with an 'ids' list"
            )
        if "schema_version" in payload:
            metadata["schema_version"] = payload["schema_version"]
        if "source_path" in payload:
            metadata["source_path"] = payload["source_path"]
        if "source_sha256" in payload:
            metadata["source_sha256"] = payload["source_sha256"]
        if "lock_context" in payload:
            metadata["lock_context"] = payload["lock_context"]
    else:
        raise ValueError(
            "sample manifest must be a JSON list or an object with an 'ids' list"
        )

    computed_fingerprint = fingerprint_ids(ids)
    if isinstance(payload, dict):
        declared_n_ids = payload.get("n_ids")
        if declared_n_ids is not None and declared_n_ids != len(ids):
            raise ValueError("sample manifest n_ids does not match the number of ids")
        declared_fingerprint = payload.get("fingerprint")
        if (
            declared_fingerprint is not None
            and str(declared_fingerprint) != computed_fingerprint
        ):
            raise ValueError("sample manifest fingerprint does not match ids")

    return {
        "path": path,
        "content_sha256": file_sha256(path),
        "n_ids": len(ids),
        "fingerprint": computed_fingerprint,
        **metadata,
    }


def build_run_contract(
    *,
    schema_version: str,
    benchmark: str,
    model_key: str | None,
    model_path: str | None,
    classifier_path: str | None,
    max_samples: int | None,
    sample_manifest: dict[str, Any] | None,
    alphas: Sequence[float],
    benchmark_config: Mapping[str, Any],
    after_benchmark_fields: Mapping[str, Any] | None = None,
    schedule_fields: Mapping[str, Any] | None = None,
    tail_fields: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the shared Run Contract scaffold used by measurement adapters."""
    schedule: dict[str, Any] = {"alphas": [float(alpha) for alpha in alphas]}
    if schedule_fields:
        schedule.update(schedule_fields)

    contract: dict[str, Any] = {
        "schema_version": schema_version,
        "benchmark": benchmark,
    }
    if after_benchmark_fields:
        contract.update(after_benchmark_fields)
    contract.update(
        {
            "model": {
                "key": model_key,
                "path": model_path,
            },
            "classifier": {
                "path": classifier_path,
                "content_sha256": optional_file_sha256(classifier_path),
            },
            "sample_selection": {
                "max_samples": max_samples,
                "sample_manifest": sample_manifest,
            },
            "schedule": schedule,
            "benchmark_config": dict(benchmark_config),
        }
    )
    if tail_fields:
        contract.update(tail_fields)
    return contract


def contract_mismatch_paths(
    expected: Any,
    observed: Any,
    *,
    prefix: str = "",
) -> list[str]:
    if isinstance(expected, dict) and isinstance(observed, dict):
        mismatches: list[str] = []
        for key in sorted(set(expected) | set(observed)):
            key_path = f"{prefix}.{key}" if prefix else str(key)
            if key not in expected:
                mismatches.append(f"{key_path} (unexpected stored value)")
            elif key not in observed:
                mismatches.append(f"{key_path} (missing stored value)")
            else:
                mismatches.extend(
                    contract_mismatch_paths(
                        expected[key],
                        observed[key],
                        prefix=key_path,
                    )
                )
        return mismatches
    if expected != observed:
        return [prefix or "<root>"]
    return []


def _coerce_alpha_sequence(value: Any) -> list[float] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return None
    try:
        return [float(alpha) for alpha in value]
    except (TypeError, ValueError):
        return None


def _summary_result_alphas(results: Mapping[str, Any]) -> list[float]:
    observed_alphas: list[float] = []
    for key in results:
        try:
            observed_alphas.append(float(key))
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"H-neuron baseline summary has a non-numeric alpha result key: {key!r}"
            ) from exc
    return observed_alphas


def _alpha_subset_mismatch(
    *,
    path: str,
    expected: Any,
    observed: Any,
) -> str | None:
    """Allow H-baseline reuse only when observed alphas cover required alphas.

    Extra observed alpha points are reusable evidence for a subset control
    comparison, but a missing required alpha means the control curve has no
    matching H-neuron measurement. Keep this exception local so every non-alpha
    binding field still flows through strict contract equality.
    """
    expected_alphas = _coerce_alpha_sequence(expected)
    observed_alphas = _coerce_alpha_sequence(observed)
    if expected_alphas is None or observed_alphas is None:
        return (
            f"{path} (expected required alphas {expected!r}; "
            f"observed alphas {observed!r})"
        )

    observed_alpha_set = set(observed_alphas)
    missing_alphas = [
        alpha for alpha in expected_alphas if alpha not in observed_alpha_set
    ]
    if not missing_alphas:
        return None
    return (
        f"{path} (missing required alphas {missing_alphas}; "
        f"expected required alphas {expected_alphas}; "
        f"observed alphas {observed_alphas})"
    )


def _without_schedule_alphas(contract_subset: Mapping[str, Any]) -> dict[str, Any]:
    stripped = dict(contract_subset)
    schedule = stripped.get("schedule")
    if isinstance(schedule, dict):
        stripped_schedule = dict(schedule)
        stripped_schedule.pop("alphas", None)
        stripped["schedule"] = stripped_schedule
    return stripped


def _without_summary_alphas(summary_identity: Mapping[str, Any]) -> dict[str, Any]:
    stripped = dict(summary_identity)
    stripped.pop("alphas", None)
    return stripped


def format_mismatch_preview(
    mismatches: Sequence[str],
    *,
    limit: int = MISMATCH_PREVIEW_LIMIT,
) -> str:
    mismatch_preview = ", ".join(mismatches[:limit])
    if len(mismatches) > limit:
        mismatch_preview += f", ... ({len(mismatches)} total)"
    return mismatch_preview


def existing_intervention_output_paths(
    output_dir: str,
    alphas: Sequence[float],
) -> list[str]:
    paths: list[str] = []
    for alpha in alphas:
        alpha_path = os.path.join(
            output_dir, f"alpha_{format_alpha_label(alpha)}.jsonl"
        )
        if os.path.exists(alpha_path):
            paths.append(alpha_path)
    return sorted(paths)


def existing_negative_control_output_paths(
    output_base: str,
    alphas: Sequence[float],
    configs: Iterable[tuple[str, str, int]],
) -> list[str]:
    paths: list[str] = []
    for name, _, _ in configs:
        seed_dir = os.path.join(output_base, name)
        results_path = os.path.join(seed_dir, "results.json")
        if os.path.exists(results_path):
            paths.append(results_path)
        for alpha in alphas:
            alpha_path = os.path.join(seed_dir, f"alpha_{alpha:.1f}.jsonl")
            if os.path.exists(alpha_path):
                paths.append(alpha_path)
    return sorted(paths)


def write_or_validate_run_contract(
    *,
    output_dir: str,
    contract: dict[str, Any],
    contract_filename: str,
    existing_output_paths: Sequence[str],
    output_kind: str,
    contract_label: str,
    fresh_output_flag: str,
    write_if_missing: bool,
) -> str:
    contract_path = os.path.join(output_dir, contract_filename)
    if os.path.exists(contract_path):
        with open(contract_path, encoding="utf-8") as handle:
            try:
                stored_contract = json.load(handle)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Invalid {contract_label} at {contract_path}: {exc}"
                ) from exc
        mismatches = contract_mismatch_paths(contract, stored_contract)
        if mismatches:
            raise RuntimeError(
                f"Refusing to reuse {output_kind} with an incompatible "
                f"run contract at {contract_path}. Mismatched fields: "
                f"{format_mismatch_preview(mismatches)}. Use a fresh "
                f"{fresh_output_flag} or archive the existing directory."
            )
        return contract_path

    if existing_output_paths:
        examples = [
            os.path.relpath(path, output_dir) for path in existing_output_paths[:5]
        ]
        example_text = ", ".join(examples)
        if len(existing_output_paths) > len(examples):
            example_text += f", ... ({len(existing_output_paths)} total)"
        raise RuntimeError(
            f"Refusing to reuse {output_kind} without {contract_filename}. "
            "Existing files may come from an incompatible pre-guard run: "
            f"{example_text}. Use a fresh {fresh_output_flag} or archive the "
            "existing directory."
        )

    if write_if_missing:
        with open(contract_path, "w", encoding="utf-8") as handle:
            json.dump(contract, handle, indent=2, sort_keys=True)
            handle.write("\n")
    return contract_path


def assert_or_write_intervention_run_contract(
    output_dir: str,
    contract: dict[str, Any],
    alphas: Sequence[float],
    *,
    write_if_missing: bool,
) -> str:
    return write_or_validate_run_contract(
        output_dir=output_dir,
        contract=contract,
        contract_filename=INTERVENTION_RUN_CONFIG_FILENAME,
        existing_output_paths=existing_intervention_output_paths(output_dir, alphas),
        output_kind="intervention outputs",
        contract_label="intervention run contract",
        fresh_output_flag="--output_dir",
        write_if_missing=write_if_missing,
    )


def assert_or_write_negative_control_run_contract(
    output_base: str,
    contract: dict[str, Any],
    alphas: Sequence[float],
    configs: Iterable[tuple[str, str, int]],
    *,
    write_if_missing: bool,
) -> str:
    return write_or_validate_run_contract(
        output_dir=output_base,
        contract=contract,
        contract_filename=CONTROL_RUN_CONFIG_FILENAME,
        existing_output_paths=existing_negative_control_output_paths(
            output_base, alphas, configs
        ),
        output_kind="negative-control outputs",
        contract_label="negative-control run contract",
        fresh_output_flag="--output_base",
        write_if_missing=write_if_missing,
    )


def resolve_results_json(path: str) -> str:
    candidate = os.path.expanduser(path)
    if os.path.isfile(candidate):
        return candidate
    if not os.path.isdir(candidate):
        return candidate

    stable_path = os.path.join(candidate, "results.json")
    if os.path.isfile(stable_path):
        return stable_path

    timestamped = [
        os.path.join(candidate, name)
        for name in os.listdir(candidate)
        if name.startswith("results.") and name.endswith(".json")
    ]
    if not timestamped:
        return stable_path
    return str(max(timestamped, key=lambda item: os.path.getmtime(item)))


def load_h_neuron_baseline_binding(
    baseline_path: str,
    alphas: Sequence[float],
) -> dict[str, Any]:
    resolved_results_path = resolve_results_json(baseline_path)
    if not os.path.isfile(resolved_results_path):
        raise RuntimeError(
            f"H-neuron baseline not found at {resolved_results_path}. Run the "
            "H-neuron intervention first."
        )

    with open(resolved_results_path, encoding="utf-8") as handle:
        try:
            baseline_summary = json.load(handle)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Invalid H-neuron baseline summary at {resolved_results_path}: {exc}"
            ) from exc

    results = baseline_summary.get("results")
    if not isinstance(results, dict):
        raise RuntimeError(
            f"H-neuron baseline summary at {resolved_results_path} has no results object"
        )
    observed_summary_alphas = _summary_result_alphas(results)
    missing_summary_alpha_mismatch = _alpha_subset_mismatch(
        path="alphas",
        expected=alphas,
        observed=observed_summary_alphas,
    )
    if missing_summary_alpha_mismatch is not None:
        raise RuntimeError(
            "H-neuron baseline summary is missing required alpha results. "
            f"Mismatched fields: {missing_summary_alpha_mismatch}."
        )

    contract_path = os.path.join(
        os.path.dirname(resolved_results_path),
        INTERVENTION_RUN_CONFIG_FILENAME,
    )
    if not os.path.isfile(contract_path):
        raise RuntimeError(
            f"H-neuron baseline is missing {INTERVENTION_RUN_CONFIG_FILENAME} at "
            f"{contract_path}. Use a fresh baseline produced by the guarded "
            "run_intervention.py contract path."
        )
    with open(contract_path, encoding="utf-8") as handle:
        try:
            baseline_contract = json.load(handle)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Invalid H-neuron baseline contract at {contract_path}: {exc}"
            ) from exc

    return {
        "schema_version": H_NEURON_BASELINE_BINDING_SCHEMA_VERSION,
        "results_path": resolved_results_path,
        "results_content_sha256": file_sha256(resolved_results_path),
        "contract_path": contract_path,
        "contract_content_sha256": file_sha256(contract_path),
        "summary_identity": {
            "benchmark": baseline_summary.get("benchmark"),
            "model_key": baseline_summary.get("model_key"),
            "model_path": baseline_summary.get("model"),
            "classifier_path": baseline_summary.get("classifier"),
            "intervention_mode": baseline_summary.get("intervention_mode"),
            "sample_manifest": baseline_summary.get("sample_manifest"),
            "prompt_style": baseline_summary.get("prompt_style"),
            "alphas": observed_summary_alphas,
        },
        "run_contract": baseline_contract,
    }


def _baseline_comparison_subset(
    contract: dict[str, Any],
    *,
    expect_h_neuron_intervention: bool,
) -> dict[str, Any]:
    return {
        "schema_version": (
            INTERVENTION_RUN_CONFIG_SCHEMA_VERSION
            if expect_h_neuron_intervention
            else contract.get("schema_version")
        ),
        "benchmark": contract.get("benchmark"),
        "model": contract.get("model"),
        "classifier": contract.get("classifier"),
        "sample_selection": contract.get("sample_selection"),
        "schedule": {
            "alphas": contract.get("schedule", {}).get("alphas"),
        },
        "benchmark_config": contract.get("benchmark_config"),
        "intervention": (
            {"mode": H_NEURON_BASELINE_INTERVENTION_MODE}
            if expect_h_neuron_intervention
            else contract.get("intervention")
        ),
    }


def assert_h_neuron_baseline_matches_control_contract(
    contract: dict[str, Any],
) -> None:
    baseline_binding = contract.get("h_neuron_baseline")
    if not isinstance(baseline_binding, dict):
        raise RuntimeError("Negative-control contract is missing h_neuron_baseline")
    baseline_contract = baseline_binding.get("run_contract")
    if not isinstance(baseline_contract, dict):
        raise RuntimeError("H-neuron baseline binding is missing run_contract")

    expected = _baseline_comparison_subset(
        contract,
        expect_h_neuron_intervention=True,
    )
    observed = _baseline_comparison_subset(
        baseline_contract,
        expect_h_neuron_intervention=False,
    )
    expected_benchmark_config = expected.get("benchmark_config") or {}
    observed_benchmark_config = observed.get("benchmark_config") or {}
    control_only_config_keys = CONTROL_ONLY_BASELINE_BENCHMARK_CONFIG_KEYS.get(
        str(contract.get("benchmark")),
        frozenset(),
    )
    intervention_only_config_keys = (
        INTERVENTION_ONLY_BASELINE_BENCHMARK_CONFIG_KEYS.get(
            str(contract.get("benchmark")),
            frozenset(),
        )
    )
    comparable_config_keys = sorted(
        (set(expected_benchmark_config) | set(observed_benchmark_config))
        - control_only_config_keys
        - intervention_only_config_keys
    )
    expected["benchmark_config"] = {
        key: expected_benchmark_config.get(key) for key in comparable_config_keys
    }
    observed["benchmark_config"] = {
        key: observed_benchmark_config.get(key) for key in comparable_config_keys
    }
    mismatches = contract_mismatch_paths(
        _without_schedule_alphas(expected),
        _without_schedule_alphas(observed),
    )
    schedule_alpha_mismatch = _alpha_subset_mismatch(
        path="schedule.alphas",
        expected=expected.get("schedule", {}).get("alphas"),
        observed=observed.get("schedule", {}).get("alphas"),
    )
    if schedule_alpha_mismatch is not None:
        mismatches.append(schedule_alpha_mismatch)
    if mismatches:
        raise RuntimeError(
            "H-neuron baseline contract does not match negative-control contract. "
            f"Mismatched fields: {format_mismatch_preview(mismatches)}."
        )

    summary_identity = baseline_binding.get("summary_identity", {})
    expected_summary = {
        "benchmark": contract.get("benchmark"),
        "model_key": contract.get("model", {}).get("key"),
        "model_path": contract.get("model", {}).get("path"),
        "classifier_path": contract.get("classifier", {}).get("path"),
        "intervention_mode": H_NEURON_BASELINE_INTERVENTION_MODE,
        "sample_manifest": contract.get("sample_selection", {}).get("sample_manifest"),
        "prompt_style": contract.get("benchmark_config", {}).get("prompt_style"),
        "alphas": contract.get("schedule", {}).get("alphas"),
    }
    if isinstance(summary_identity, Mapping):
        summary_mismatches = contract_mismatch_paths(
            _without_summary_alphas(expected_summary),
            _without_summary_alphas(summary_identity),
        )
        observed_summary_alphas = summary_identity.get("alphas")
    else:
        summary_mismatches = contract_mismatch_paths(expected_summary, summary_identity)
        observed_summary_alphas = None
    summary_alpha_mismatch = _alpha_subset_mismatch(
        path="alphas",
        expected=expected_summary.get("alphas"),
        observed=observed_summary_alphas,
    )
    if summary_alpha_mismatch is not None:
        summary_mismatches.append(summary_alpha_mismatch)
    if summary_mismatches:
        raise RuntimeError(
            "H-neuron baseline summary does not match negative-control contract. "
            f"Mismatched fields: {format_mismatch_preview(summary_mismatches)}."
        )
