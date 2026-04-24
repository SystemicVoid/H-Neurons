"""Select SAE steering features for FaithEval via validation logprob margins.

Locked selector for this diagnostic:
- candidate pool: all 509 non-zero SAE probe-support features
- prompt format: FaithEval anti-compliance prompt wrapped in the same chat template
- operator/sign family: SAE delta-only ablation at alpha=0.0
- utility metric: mean reduction in misleading-vs-preferred logprob margin
- selected set size: k=266

This is an in-family target-selection ablation for the L2/L3 reviewer critique:
it asks whether a utility-aware selector can recover a steerable SAE set within
the existing extraction scope. It is not a blind wider-layer or wider-width SAE
sweep, so layer-coverage closure remains partial by design.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from extract_sae_activations import load_saes
from intervene_sae import (
    SAEFeatureScaler,
    build_sae_feature_entries,
    build_sae_feature_manifest,
    get_positive_sae_features_from_classifier,
    get_zero_weight_sae_feature_indices,
    load_sae_classifier_coefficients,
)
from run_intervention import (
    FAITHEVAL_ANSWER_SPAN_PRIMARY_WINDOW_TOKENS,
    FAITHEVAL_CANONICAL_LABELS,
    _choice_token_ids,
    _faitheval_prompt,
    build_faitheval_answer_text_target_token_cache,
    load_faitheval,
    load_model_and_tokenizer,
    misleading_margin,
    score_faitheval_answer_text_targets_from_prompt_ids,
    score_choice_logprobs_from_prompt_ids,
    tokenize_chat,
)
from utils import (
    finish_run_provenance,
    fingerprint_ids,
    json_dumps,
    provenance_error_message,
    provenance_status_for_exception,
    start_run_provenance,
)


DEFAULT_MODEL_PATH = "google/gemma-3-4b-it"
DEFAULT_DEVICE_MAP = "cuda:0"
DEFAULT_CLASSIFIER_PATH = "models/sae_detector.pkl"
DEFAULT_CLASSIFIER_SUMMARY = "data/gemma3_4b/pipeline/classifier_sae_summary.json"
DEFAULT_OUTPUT_DIR = (
    "data/gemma3_4b/intervention/faitheval_sae_utility_selector/selector"
)
DEFAULT_VALIDATION_SIZE = 160
DEFAULT_SELECTION_SEED = 42
DEFAULT_TOP_K = 266
DEFAULT_N_RANDOM_SEEDS = 10
DEFAULT_PROMPT_STYLE = "anti_compliance"
DEFAULT_ALPHA = 0.0
DEFAULT_OLD_SHORTLIST_THRESHOLD = 1e-3
PATH_DRIFT_CONTROL_FAMILY = "matched_zero_dead"
PATH_DRIFT_CONTROL_LAYER = 20
INCREMENTAL_SCORE_CACHE_SCHEMA_VERSION = "selector_score_cache/v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--device_map", type=str, default=DEFAULT_DEVICE_MAP)
    parser.add_argument(
        "--classifier_path",
        type=str,
        default=DEFAULT_CLASSIFIER_PATH,
    )
    parser.add_argument(
        "--classifier_summary",
        type=str,
        default=DEFAULT_CLASSIFIER_SUMMARY,
    )
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--validation_size", type=int, default=DEFAULT_VALIDATION_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SELECTION_SEED)
    parser.add_argument("--top_k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--n_random_seeds", type=int, default=DEFAULT_N_RANDOM_SEEDS)
    parser.add_argument("--prompt_style", type=str, default=DEFAULT_PROMPT_STYLE)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument(
        "--old_shortlist_threshold",
        type=float,
        default=DEFAULT_OLD_SHORTLIST_THRESHOLD,
    )
    return parser.parse_args()


def _load_classifier_summary(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


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


def _feature_key(feature: dict[str, Any]) -> tuple[int, int]:
    return int(feature["layer"]), int(feature["feature"])


def _feature_id(feature: dict[str, Any]) -> str:
    return f"L{int(feature['layer'])}F{int(feature['feature'])}"


def _rows_by_id(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        sample_id = str(row["id"])
        if sample_id in out:
            raise ValueError(f"Duplicate FaithEval sample id {sample_id!r}")
        out[sample_id] = row
    return out


def _canonical_answer_position(sample: dict[str, Any]) -> str:
    raw_value = sample.get("counterfactual_key_canonical")
    if raw_value is None or str(raw_value).strip() == "":
        raise ValueError(
            "FaithEval sample is missing required counterfactual_key_canonical "
            f"for split construction (sample_id={sample.get('id')!r})"
        )

    canonical_key = str(raw_value).strip()
    num_options = int(sample["num_options"])
    valid_positions = FAITHEVAL_CANONICAL_LABELS[:num_options]
    if canonical_key not in valid_positions:
        raise ValueError(
            "FaithEval sample has invalid counterfactual_key_canonical "
            f"{canonical_key!r} for num_options={num_options} "
            f"(sample_id={sample.get('id')!r})"
        )
    return canonical_key


def build_stratified_faitheval_split(
    samples: list[dict[str, Any]],
    *,
    validation_size: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if validation_size <= 0 or validation_size >= len(samples):
        raise ValueError("validation_size must be between 1 and len(samples)-1")

    strata: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        key = (int(sample["num_options"]), _canonical_answer_position(sample))
        strata[key].append(sample)

    rng = random.Random(seed)
    for group in strata.values():
        rng.shuffle(group)

    total_n = len(samples)
    quotas: list[tuple[float, tuple[int, str], int]] = []
    base_total = 0
    for key, group in strata.items():
        exact = validation_size * len(group) / total_n
        base = int(np.floor(exact))
        quotas.append((exact - base, key, base))
        base_total += base

    remainder = validation_size - base_total
    quotas.sort(key=lambda item: (-item[0], item[1][0], item[1][1]))
    target_counts = {key: base for _, key, base in quotas}
    for _, key, _ in quotas[:remainder]:
        target_counts[key] += 1

    validation: list[dict[str, Any]] = []
    test: list[dict[str, Any]] = []
    validation_counts: dict[str, int] = {}
    test_counts: dict[str, int] = {}
    for key in sorted(strata):
        group = strata[key]
        n_val = target_counts[key]
        validation.extend(group[:n_val])
        test.extend(group[n_val:])
        str_key = f"{key[0]}::{key[1]}"
        validation_counts[str_key] = n_val
        test_counts[str_key] = len(group) - n_val

    validation.sort(key=lambda item: str(item["id"]))
    test.sort(key=lambda item: str(item["id"]))
    metadata = {
        "seed": int(seed),
        "validation_size": int(len(validation)),
        "test_size": int(len(test)),
        "stratify_by": ["num_options", "counterfactual_key_canonical"],
        "strata_validation_counts": validation_counts,
        "strata_test_counts": test_counts,
        "validation_fingerprint": fingerprint_ids(
            [str(sample["id"]) for sample in validation]
        ),
        "test_fingerprint": fingerprint_ids([str(sample["id"]) for sample in test]),
    }
    return validation, test, metadata


def build_split_manifest(
    *,
    split_name: str,
    samples: list[dict[str, Any]],
    split_metadata: dict[str, Any],
) -> dict[str, Any]:
    ids = [str(sample["id"]) for sample in samples]
    return {
        "schema_version": "sample_manifest/v2",
        "benchmark": "faitheval",
        "split_name": split_name,
        "n_ids": len(ids),
        "ids": ids,
        "fingerprint": fingerprint_ids(ids),
        "split_metadata": split_metadata,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json_dumps(payload)
    if path.exists() and path.read_text(encoding="utf-8") == content:
        return
    tmp_path = path.with_name(f".{path.name}.tmp")
    tmp_path.write_text(content, encoding="utf-8")
    os.replace(tmp_path, path)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    if path.exists() and path.read_text(encoding="utf-8") == content:
        return
    tmp_path = path.with_name(f".{path.name}.tmp")
    tmp_path.write_text(content, encoding="utf-8")
    os.replace(tmp_path, path)


def _append_jsonl_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _reset_jsonl(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")


def _sha256_hexdigest(payload: dict[str, Any]) -> str:
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _file_sha256_hexdigest(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifacts_are_fresh_against_dependencies(
    artifact_paths: list[Path],
    *,
    dependency_paths: list[Path],
) -> bool:
    if not artifact_paths or not dependency_paths:
        return False
    newest_dependency_mtime_ns = max(
        path.stat().st_mtime_ns for path in dependency_paths if path.exists()
    )
    oldest_artifact_mtime_ns = min(path.stat().st_mtime_ns for path in artifact_paths)
    return oldest_artifact_mtime_ns >= newest_dependency_mtime_ns


def _completed_selector_provenance(output_dir: Path) -> dict[str, Any] | None:
    candidates = sorted(
        output_dir.glob("select_faitheval_sae_utility_features.provenance.*.json"),
        reverse=True,
    )
    for path in candidates:
        payload = _load_json(path)
        if payload.get("status") == "completed":
            payload["_path"] = str(path)
            return payload
    return None


def _selector_scoring_input_payload(
    args: argparse.Namespace,
    *,
    extraction_metadata: dict[str, Any],
    candidate_pool: list[dict[str, Any]],
    split_metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "benchmark": "faitheval",
        "selector_bundle_schema": "v6",
        "model_path": str(args.model_path),
        "device_map": str(args.device_map),
        "classifier_path": str(args.classifier_path),
        "classifier_sha256": _file_sha256_hexdigest(Path(args.classifier_path)),
        "classifier_summary": str(args.classifier_summary),
        "classifier_summary_sha256": _sha256_hexdigest(
            _load_classifier_summary(args.classifier_summary)
        ),
        "validation_size": int(args.validation_size),
        "seed": int(args.seed),
        "top_k": int(args.top_k),
        "prompt_style": str(args.prompt_style),
        "alpha": float(args.alpha),
        "answer_span_primary_window_tokens": int(
            FAITHEVAL_ANSWER_SPAN_PRIMARY_WINDOW_TOKENS
        ),
        "old_shortlist_threshold": float(args.old_shortlist_threshold),
        "layer_indices": [int(layer) for layer in extraction_metadata["layer_indices"]],
        "d_sae": int(extraction_metadata["d_sae"]),
        "candidate_pool_fingerprint": fingerprint_ids(
            [_feature_id(feature) for feature in candidate_pool]
        ),
        "candidate_pool_n": int(len(candidate_pool)),
        "validation_fingerprint": str(split_metadata["validation_fingerprint"]),
        "test_fingerprint": str(split_metadata["test_fingerprint"]),
    }


def _provenance_args_match_current(
    provenance_payload: dict[str, Any] | None,
    args: argparse.Namespace,
) -> bool:
    if provenance_payload is None:
        return False
    recorded_args = provenance_payload.get("args")
    if not isinstance(recorded_args, dict):
        return False
    expected_args = {
        "model_path": str(args.model_path),
        "device_map": str(args.device_map),
        "classifier_path": str(args.classifier_path),
        "classifier_summary": str(args.classifier_summary),
        "output_dir": str(args.output_dir),
        "validation_size": int(args.validation_size),
        "seed": int(args.seed),
        "top_k": int(args.top_k),
        "prompt_style": str(args.prompt_style),
        "alpha": float(args.alpha),
        "old_shortlist_threshold": float(args.old_shortlist_threshold),
    }
    return recorded_args == expected_args


def _resolve_selector_scoring_cache(
    *,
    output_dir: Path,
    feature_stats_path: Path,
    full_sequence_stats_path: Path,
    utility_scores_path: Path,
    answer_span_scores_path: Path,
    utility_selected_manifest_path: Path,
    answer_span_selected_manifest_path: Path,
    readout_selected_manifest_path: Path,
    current_input_hash: str,
    current_input_payload: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    required_paths = [
        feature_stats_path,
        full_sequence_stats_path,
        utility_scores_path,
        answer_span_scores_path,
        utility_selected_manifest_path,
        answer_span_selected_manifest_path,
        readout_selected_manifest_path,
    ]
    if not all(path.exists() for path in required_paths):
        return {
            "cache_status": "computed",
            "cache_mode": "fresh_compute",
            "missing_artifacts": [
                str(path.name) for path in required_paths if not path.exists()
            ],
        }

    selector_summary_path = output_dir / "selector_summary.json"
    legacy_summary = (
        _load_json(selector_summary_path) if selector_summary_path.exists() else {}
    )
    selector_scoring = legacy_summary.get("selector_scoring")
    if (
        isinstance(selector_scoring, dict)
        and str(selector_scoring.get("input_hash", "")) == current_input_hash
    ):
        return {
            "cache_status": "reused",
            "cache_mode": "exact_input_hash",
            "input_hash": current_input_hash,
            "reused_artifacts": [path.name for path in required_paths],
        }

    legacy_split = legacy_summary.get("split_metadata")
    legacy_candidate_pool = legacy_summary.get("candidate_pool")
    provenance_payload = _completed_selector_provenance(output_dir)
    cache_artifacts_are_fresh = (
        provenance_payload is not None
        and _artifacts_are_fresh_against_dependencies(
            [
                *required_paths,
                selector_summary_path,
                Path(str(provenance_payload["_path"])),
            ],
            dependency_paths=[
                Path(args.classifier_path),
                Path(args.classifier_summary),
            ],
        )
    )
    if (
        isinstance(legacy_split, dict)
        and isinstance(legacy_candidate_pool, dict)
        and provenance_payload is not None
        and cache_artifacts_are_fresh
        and str(legacy_split.get("validation_fingerprint", ""))
        == str(current_input_payload["validation_fingerprint"])
        and str(legacy_split.get("test_fingerprint", ""))
        == str(current_input_payload["test_fingerprint"])
        and str(legacy_candidate_pool.get("fingerprint", ""))
        == str(current_input_payload["candidate_pool_fingerprint"])
        and int(legacy_candidate_pool.get("n_features", -1))
        == int(current_input_payload["candidate_pool_n"])
        and _provenance_args_match_current(provenance_payload, args)
    ):
        return {
            "cache_status": "reused",
            "cache_mode": "legacy_provenance_args_and_frozen_metadata",
            "input_hash": current_input_hash,
            "reused_artifacts": [path.name for path in required_paths],
            "legacy_source_provenance": provenance_payload["_path"],
        }

    return {
        "cache_status": "computed",
        "cache_mode": "fresh_compute",
        "input_hash": current_input_hash,
    }


def _prepare_incremental_score_cache(
    *,
    output_dir: Path,
    current_input_hash: str,
    score_paths: list[Path],
) -> dict[str, Any]:
    state_path = output_dir / "selector_scoring_state.json"
    expected_state = {
        "schema_version": INCREMENTAL_SCORE_CACHE_SCHEMA_VERSION,
        "input_hash": current_input_hash,
    }
    if state_path.exists():
        try:
            state = _load_json(state_path)
        except json.JSONDecodeError:
            state = {}
        if state == expected_state:
            return {
                "incremental_cache_status": "resumed",
                "incremental_state_path": state_path.name,
            }

    for path in score_paths:
        _reset_jsonl(path)
    _write_json(state_path, expected_state)
    return {
        "incremental_cache_status": "initialized",
        "incremental_state_path": state_path.name,
    }


def _candidate_rows_by_flat_idx(
    candidate_pool: list[dict[str, Any]],
) -> dict[int, dict[str, Any]]:
    return {int(feature["flat_idx"]): feature for feature in candidate_pool}


def _row_matches_candidate(row: dict[str, Any], candidate: dict[str, Any]) -> bool:
    return (
        int(row.get("flat_idx", -1)) == int(candidate["flat_idx"])
        and int(row.get("layer", -1)) == int(candidate["layer"])
        and int(row.get("feature", -1)) == int(candidate["feature"])
    )


def _load_completed_score_rows(
    path: Path,
    *,
    candidate_pool: list[dict[str, Any]],
    validation_n: int,
    required_score_fields: tuple[str, ...],
) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    candidate_by_flat_idx = _candidate_rows_by_flat_idx(candidate_pool)
    rows_by_flat_idx: dict[int, dict[str, Any]] = {}
    for row in _load_jsonl(path):
        flat_idx = int(row.get("flat_idx", -1))
        candidate = candidate_by_flat_idx.get(flat_idx)
        if candidate is None or not _row_matches_candidate(row, candidate):
            continue
        if int(row.get("validation_n", -1)) != validation_n:
            continue
        if not all(field in row for field in required_score_fields):
            continue
        rows_by_flat_idx[flat_idx] = row

    return [
        rows_by_flat_idx[int(feature["flat_idx"])]
        for feature in candidate_pool
        if int(feature["flat_idx"]) in rows_by_flat_idx
    ]


def build_prompt_cache(
    tokenizer,
    samples: list[dict[str, Any]],
    *,
    prompt_style: str,
) -> dict[str, torch.Tensor]:
    cache: dict[str, torch.Tensor] = {}
    for sample in samples:
        prompt = _faitheval_prompt(sample, prompt_style)
        cache[str(sample["id"])] = tokenize_chat(
            tokenizer,
            [{"role": "user", "content": prompt}],
        )
    return cache


class PromptEndFeatureCollector:
    """Capture prompt-end SAE activations for the selected layers."""

    def __init__(self, model, saes: dict[int, Any], layer_indices: list[int]):
        self.layer_indices = sorted(layer_indices)
        self.saes = saes
        self.captured: dict[int, torch.Tensor] = {}
        self.hooks: list[Any] = []
        for name, module in model.named_modules():
            if "post_feedforward_layernorm" not in name:
                continue
            layer_idx = self._extract_layer_idx(name)
            if layer_idx is None or layer_idx not in self.layer_indices:
                continue
            self.hooks.append(module.register_forward_hook(self._make_hook(layer_idx)))

    @staticmethod
    def _extract_layer_idx(name: str) -> int | None:
        for part in name.split("."):
            if part.isdigit():
                return int(part)
        return None

    def _make_hook(self, layer_idx: int):
        def hook_fn(module, inputs, output):
            self.captured[layer_idx] = output[:, -1:, :].detach()

        return hook_fn

    def clear(self) -> None:
        self.captured.clear()

    def encode_prompt_end(self) -> dict[int, torch.Tensor]:
        encoded: dict[int, torch.Tensor] = {}
        for layer_idx in self.layer_indices:
            hidden = self.captured[layer_idx].float().to(self.saes[layer_idx].device)
            with torch.inference_mode():
                features = (
                    self.saes[layer_idx]
                    .encode(hidden)
                    .detach()
                    .cpu()
                    .squeeze(0)
                    .squeeze(0)
                )
            encoded[layer_idx] = features
        return encoded

    def remove(self) -> None:
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()


class FullSequenceFeatureCollector:
    """Capture full-sequence SAE activations for the selected layers."""

    def __init__(self, model, saes: dict[int, Any], layer_indices: list[int]):
        self.layer_indices = sorted(layer_indices)
        self.saes = saes
        self.captured: dict[int, torch.Tensor] = {}
        self.hooks: list[Any] = []
        for name, module in model.named_modules():
            if "post_feedforward_layernorm" not in name:
                continue
            layer_idx = self._extract_layer_idx(name)
            if layer_idx is None or layer_idx not in self.layer_indices:
                continue
            self.hooks.append(module.register_forward_hook(self._make_hook(layer_idx)))

    @staticmethod
    def _extract_layer_idx(name: str) -> int | None:
        for part in name.split("."):
            if part.isdigit():
                return int(part)
        return None

    def _make_hook(self, layer_idx: int):
        def hook_fn(module, inputs, output):
            self.captured[layer_idx] = output.detach()

        return hook_fn

    def clear(self) -> None:
        self.captured.clear()

    def encode_full_sequence(self) -> dict[int, torch.Tensor]:
        encoded: dict[int, torch.Tensor] = {}
        for layer_idx in self.layer_indices:
            hidden = self.captured[layer_idx].float().to(self.saes[layer_idx].device)
            with torch.inference_mode():
                features = self.saes[layer_idx].encode(hidden).detach().cpu()
            encoded[layer_idx] = features.squeeze(0)
        return encoded

    def remove(self) -> None:
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()


def collect_prompt_end_feature_stats(
    model,
    saes: dict[int, Any],
    samples: list[dict[str, Any]],
    prompt_cache: dict[str, torch.Tensor],
    *,
    layer_indices: list[int],
    d_sae: int,
) -> dict[int, dict[str, float | int]]:
    collector = PromptEndFeatureCollector(model, saes, layer_indices)
    activation_counts = {
        layer_idx: np.zeros(d_sae, dtype=np.int64) for layer_idx in layer_indices
    }
    activation_sums = {
        layer_idx: np.zeros(d_sae, dtype=np.float64) for layer_idx in layer_indices
    }

    try:
        for sample in tqdm(samples, desc="Prompt-end SAE stats"):
            collector.clear()
            input_ids = prompt_cache[str(sample["id"])].to(model.device)
            with torch.inference_mode():
                model(input_ids)
            encoded = collector.encode_prompt_end()
            for layer_idx, features in encoded.items():
                values = features.numpy()
                activation_counts[layer_idx] += (values > 0).astype(np.int64)
                activation_sums[layer_idx] += values
    finally:
        collector.remove()

    stats: dict[int, dict[str, float | int]] = {}
    n_samples = len(samples)
    for layer_pos, layer_idx in enumerate(layer_indices):
        sae = saes[layer_idx]
        decoder = sae.W_dec.detach().float().cpu().numpy()
        decoder_norms = np.linalg.norm(decoder, axis=1)
        for feature_idx in range(d_sae):
            flat_idx = layer_pos * d_sae + feature_idx
            stats[flat_idx] = {
                "flat_idx": int(flat_idx),
                "layer": int(layer_idx),
                "feature": int(feature_idx),
                "activation_count": int(activation_counts[layer_idx][feature_idx]),
                "activation_frequency": float(
                    activation_counts[layer_idx][feature_idx] / n_samples
                ),
                "mean_activation": float(
                    activation_sums[layer_idx][feature_idx] / n_samples
                ),
                "decoder_norm": float(decoder_norms[feature_idx]),
            }
    return stats


def collect_full_sequence_feature_stats(
    model,
    saes: dict[int, Any],
    samples: list[dict[str, Any]],
    prompt_cache: dict[str, torch.Tensor],
    *,
    layer_indices: list[int],
    d_sae: int,
) -> dict[int, dict[str, float | int]]:
    collector = FullSequenceFeatureCollector(model, saes, layer_indices)
    token_active_counts = {
        layer_idx: np.zeros(d_sae, dtype=np.int64) for layer_idx in layer_indices
    }
    activation_sums = {
        layer_idx: np.zeros(d_sae, dtype=np.float64) for layer_idx in layer_indices
    }
    n_validation_tokens = 0

    try:
        for sample in tqdm(samples, desc="Full-sequence SAE stats"):
            collector.clear()
            input_ids = prompt_cache[str(sample["id"])].to(model.device)
            with torch.inference_mode():
                model(input_ids)
            encoded = collector.encode_full_sequence()
            for layer_idx, features in encoded.items():
                values = features.numpy()
                n_tokens = values.shape[0]
                n_validation_tokens += n_tokens if layer_idx == layer_indices[0] else 0
                token_active_counts[layer_idx] += (
                    (values > 0).sum(axis=0).astype(np.int64)
                )
                activation_sums[layer_idx] += values.sum(axis=0)
    finally:
        collector.remove()

    stats: dict[int, dict[str, float | int]] = {}
    for layer_pos, layer_idx in enumerate(layer_indices):
        sae = saes[layer_idx]
        decoder = sae.W_dec.detach().float().cpu().numpy()
        decoder_norms = np.linalg.norm(decoder, axis=1)
        for feature_idx in range(d_sae):
            flat_idx = layer_pos * d_sae + feature_idx
            stats[flat_idx] = {
                "flat_idx": int(flat_idx),
                "layer": int(layer_idx),
                "feature": int(feature_idx),
                "token_active_count": int(token_active_counts[layer_idx][feature_idx]),
                "token_activation_rate": float(
                    token_active_counts[layer_idx][feature_idx]
                    / max(1, n_validation_tokens)
                ),
                "n_validation_tokens": int(n_validation_tokens),
                "mean_activation": float(
                    activation_sums[layer_idx][feature_idx]
                    / max(1, n_validation_tokens)
                ),
                "decoder_norm": float(decoder_norms[feature_idx]),
            }
    return stats


def build_candidate_pool(
    *,
    coefficients: np.ndarray,
    layer_indices: list[int],
    d_sae: int,
    old_shortlist_threshold: float,
) -> list[dict[str, Any]]:
    nonzero = np.flatnonzero(coefficients != 0)
    entries = build_sae_feature_entries(
        nonzero.tolist(),
        layer_indices=layer_indices,
        d_sae=d_sae,
    )
    out: list[dict[str, Any]] = []
    for entry in entries:
        weight = float(coefficients[int(entry["flat_idx"])])
        out.append(
            {
                **entry,
                "weight": weight,
                "weight_sign": "positive" if weight > 0 else "negative",
                "abs_weight": abs(weight),
                "in_old_shortlist": abs(weight) > old_shortlist_threshold,
            }
        )
    return out


def baseline_margins(
    model,
    validation_samples: list[dict[str, Any]],
    prompt_cache: dict[str, torch.Tensor],
    choice_token_cache: dict[str, dict[str, torch.Tensor]],
) -> dict[str, float]:
    margins: dict[str, float] = {}
    for sample in tqdm(validation_samples, desc="Baseline margins"):
        sample_id = str(sample["id"])
        scores = score_choice_logprobs_from_prompt_ids(
            model,
            prompt_cache[sample_id],
            choice_token_cache[sample_id],
        )
        margins[sample_id] = misleading_margin(sample, scores)
    return margins


def baseline_answer_span_margins(
    model,
    validation_samples: list[dict[str, Any]],
    prompt_cache: dict[str, torch.Tensor],
    answer_text_token_cache: dict[str, dict[str, torch.Tensor]],
) -> dict[str, dict[str, float]]:
    margins: dict[str, dict[str, float]] = {}
    for sample in tqdm(validation_samples, desc="Baseline answer-span margins"):
        sample_id = str(sample["id"])
        diagnostics = score_faitheval_answer_text_targets_from_prompt_ids(
            model,
            prompt_cache[sample_id],
            answer_text_token_cache[sample_id],
            primary_window_tokens=FAITHEVAL_ANSWER_SPAN_PRIMARY_WINDOW_TOKENS,
        )
        margins[sample_id] = {
            "first3": float(diagnostics["margins"]["first3"]),
            "full": float(diagnostics["margins"]["full"]),
        }
    return margins


def score_candidate_feature(
    model,
    saes: dict[int, Any],
    validation_samples: list[dict[str, Any]],
    prompt_cache: dict[str, torch.Tensor],
    choice_token_cache: dict[str, dict[str, torch.Tensor]],
    baseline_margin_by_id: dict[str, float],
    *,
    feature: dict[str, Any],
    alpha: float,
) -> dict[str, Any]:
    feature_map = {int(feature["layer"]): [int(feature["feature"])]}
    scaler = SAEFeatureScaler(
        model,
        saes,
        feature_map,
        next(model.parameters()).device,
        mode="delta_only",
    )
    scaler.alpha = alpha
    reductions: list[float] = []
    try:
        for sample in validation_samples:
            sample_id = str(sample["id"])
            scores = score_choice_logprobs_from_prompt_ids(
                model,
                prompt_cache[sample_id],
                choice_token_cache[sample_id],
                scaler=scaler,
            )
            ablated_margin = misleading_margin(sample, scores)
            reductions.append(baseline_margin_by_id[sample_id] - ablated_margin)
    finally:
        scaler.remove()

    reductions_arr = np.asarray(reductions, dtype=np.float64)
    return {
        **feature,
        "selector_score": float(reductions_arr.mean()),
        "selector_score_std": float(reductions_arr.std(ddof=0)),
        "selector_score_sum": float(reductions_arr.sum()),
        "validation_n": int(len(reductions)),
    }


def score_candidate_feature_answer_span(
    model,
    saes: dict[int, Any],
    validation_samples: list[dict[str, Any]],
    prompt_cache: dict[str, torch.Tensor],
    answer_text_token_cache: dict[str, dict[str, torch.Tensor]],
    baseline_margin_by_id: dict[str, dict[str, float]],
    *,
    feature: dict[str, Any],
    alpha: float,
) -> dict[str, Any]:
    feature_map = {int(feature["layer"]): [int(feature["feature"])]}
    scaler = SAEFeatureScaler(
        model,
        saes,
        feature_map,
        next(model.parameters()).device,
        mode="delta_only",
    )
    scaler.alpha = alpha
    first3_reductions: list[float] = []
    full_reductions: list[float] = []
    try:
        for sample in validation_samples:
            sample_id = str(sample["id"])
            diagnostics = score_faitheval_answer_text_targets_from_prompt_ids(
                model,
                prompt_cache[sample_id],
                answer_text_token_cache[sample_id],
                scaler=scaler,
                primary_window_tokens=FAITHEVAL_ANSWER_SPAN_PRIMARY_WINDOW_TOKENS,
            )
            first3_reductions.append(
                baseline_margin_by_id[sample_id]["first3"]
                - float(diagnostics["margins"]["first3"])
            )
            full_reductions.append(
                baseline_margin_by_id[sample_id]["full"]
                - float(diagnostics["margins"]["full"])
            )
    finally:
        scaler.remove()

    first3_arr = np.asarray(first3_reductions, dtype=np.float64)
    full_arr = np.asarray(full_reductions, dtype=np.float64)
    return {
        **feature,
        "selector_score": float(first3_arr.mean()),
        "selector_score_std": float(first3_arr.std(ddof=0)),
        "selector_score_sum": float(first3_arr.sum()),
        "selector_score_full_span": float(full_arr.mean()),
        "selector_score_full_span_std": float(full_arr.std(ddof=0)),
        "selector_score_full_span_sum": float(full_arr.sum()),
        "validation_n": int(len(first3_reductions)),
    }


def layer_histogram(features: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(int(feature["layer"]) for feature in features)
    return {str(layer): int(counts[layer]) for layer in sorted(counts)}


def weight_sign_counts(features: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(str(feature.get("weight_sign", "unknown")) for feature in features)
    return {sign: int(counts[sign]) for sign in sorted(counts)}


def jaccard_overlap(
    left: list[dict[str, Any]],
    right: list[dict[str, Any]],
) -> dict[str, Any]:
    left_ids = {int(feature["flat_idx"]) for feature in left}
    right_ids = {int(feature["flat_idx"]) for feature in right}
    intersection = left_ids & right_ids
    union = left_ids | right_ids
    return {
        "intersection_count": int(len(intersection)),
        "union_count": int(len(union)),
        "jaccard": float(len(intersection) / len(union)) if union else 0.0,
    }


def match_random_zero_weight_features(
    selected_features: list[dict[str, Any]],
    zero_weight_pool: list[dict[str, Any]],
    feature_stats: dict[int, dict[str, float | int]],
    *,
    seed: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    pool_by_layer: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for feature in zero_weight_pool:
        merged = {**feature, **feature_stats[int(feature["flat_idx"])]}
        if float(merged.get("token_activation_rate", 0.0)) <= 0.0:
            continue
        pool_by_layer[int(feature["layer"])].append(merged)

    selected_by_layer: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for feature in selected_features:
        selected_by_layer[int(feature["layer"])].append(feature)

    matched: list[dict[str, Any]] = []
    for layer in sorted(selected_by_layer):
        pool = list(pool_by_layer[layer])
        if len(pool) < len(selected_by_layer[layer]):
            raise ValueError(
                f"Layer {layer} only has {len(pool)} eligible zero-weight controls "
                f"(token_activation_rate > 0) for "
                f"{len(selected_by_layer[layer])} selected features"
            )
        priorities: list[tuple[float, int, dict[str, Any]]] = []
        for candidate in pool:
            weight = float(candidate["token_activation_rate"])
            if weight <= 0.0:
                continue
            uniform = rng.random()
            while uniform <= 0.0:
                uniform = rng.random()
            priorities.append(
                (
                    math.log(uniform) / weight,
                    int(candidate["flat_idx"]),
                    candidate,
                )
            )
        priorities.sort(key=lambda item: (-item[0], item[1]))
        matched.extend(
            candidate for _, _, candidate in priorities[: len(selected_by_layer[layer])]
        )
    matched.sort(key=lambda feature: int(feature["flat_idx"]))
    return matched


def feature_manifest_with_selector_metadata(
    features: list[dict[str, Any]],
    *,
    extraction_metadata: dict[str, Any],
    selector_name: str,
    split_metadata: dict[str, Any],
    alpha: float,
    prompt_style: str,
) -> dict[str, Any]:
    return build_sae_feature_manifest(
        features,
        extraction_metadata=extraction_metadata,
        extra={
            "selector_name": selector_name,
            "benchmark": "faitheval",
            "prompt_style": prompt_style,
            "sae_steering_mode": "delta_only",
            "alpha": float(alpha),
            "split_metadata": split_metadata,
        },
    )


def select_zero_dead_path_drift_control(
    zero_weight_pool: list[dict[str, Any]],
    full_sequence_feature_stats: dict[int, dict[str, float | int]],
    *,
    target_layer: int = PATH_DRIFT_CONTROL_LAYER,
) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    for feature in zero_weight_pool:
        merged = {**feature, **full_sequence_feature_stats[int(feature["flat_idx"])]}
        if int(merged["layer"]) != target_layer:
            continue
        if float(merged.get("token_activation_rate", 0.0)) != 0.0:
            continue
        candidates.append(merged)
    if not candidates:
        raise ValueError(
            "No zero-weight dead-feature path-drift control found for "
            f"layer {target_layer}"
        )
    return min(candidates, key=lambda feature: int(feature["flat_idx"]))


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    matched_manifest_paths = [
        output_dir / f"matched_random_seed_{seed}_features.json"
        for seed in range(args.n_random_seeds)
    ]
    matched_answer_span_manifest_paths = [
        output_dir / f"matched_random_answer_span_seed_{seed}_features.json"
        for seed in range(args.n_random_seeds)
    ]
    output_targets = [
        output_dir,
        output_dir / "candidate_pool.json",
        output_dir / "validation_manifest.json",
        output_dir / "test_manifest.json",
        output_dir / "feature_stats.json",
        output_dir / "full_sequence_feature_stats.json",
        output_dir / "selector_scoring_state.json",
        output_dir / "utility_scores.jsonl",
        output_dir / "answer_span_scores.jsonl",
        output_dir / "utility_selected_features.json",
        output_dir / "answer_span_selected_features.json",
        output_dir / "readout_selected_features.json",
        *matched_manifest_paths,
        *matched_answer_span_manifest_paths,
        output_dir / f"{PATH_DRIFT_CONTROL_FAMILY}_features.json",
        output_dir / "selector_summary.json",
    ]
    provenance_handle = start_run_provenance(
        args,
        primary_target=output_dir,
        output_targets=[str(path) for path in output_targets],
        primary_target_is_dir=True,
    )
    provenance_status = "completed"
    provenance_extra: dict[str, Any] = {}

    try:
        classifier_summary = _load_classifier_summary(args.classifier_summary)
        extraction_metadata = classifier_summary["extraction_metadata"]
        layer_indices = [int(layer) for layer in extraction_metadata["layer_indices"]]
        d_sae = int(extraction_metadata["d_sae"])
        coefficients = load_sae_classifier_coefficients(args.classifier_path)

        samples = load_faitheval()
        missing_preferred = [
            str(sample["id"]) for sample in samples if not sample.get("preferred_key")
        ]
        if missing_preferred:
            raise ValueError(
                "FaithEval preferred-key resolution failed for "
                f"{len(missing_preferred)} samples (sample={missing_preferred[:5]})"
            )
        validation_samples, test_samples, split_metadata = (
            build_stratified_faitheval_split(
                samples,
                validation_size=args.validation_size,
                seed=args.seed,
            )
        )
        validation_manifest = build_split_manifest(
            split_name="validation",
            samples=validation_samples,
            split_metadata=split_metadata,
        )
        test_manifest = build_split_manifest(
            split_name="test",
            samples=test_samples,
            split_metadata=split_metadata,
        )
        _write_json(output_dir / "validation_manifest.json", validation_manifest)
        _write_json(output_dir / "test_manifest.json", test_manifest)

        candidate_pool = build_candidate_pool(
            coefficients=coefficients,
            layer_indices=layer_indices,
            d_sae=d_sae,
            old_shortlist_threshold=args.old_shortlist_threshold,
        )
        _write_json(
            output_dir / "candidate_pool.json",
            build_sae_feature_manifest(
                candidate_pool,
                extraction_metadata=extraction_metadata,
                extra={
                    "candidate_pool_name": "classifier_nonzero_support",
                    "n_positive": int(np.sum(coefficients > 0)),
                    "n_negative": int(np.sum(coefficients < 0)),
                    "old_shortlist_threshold": float(args.old_shortlist_threshold),
                },
            ),
        )

        feature_stats_path = output_dir / "feature_stats.json"
        full_sequence_stats_path = output_dir / "full_sequence_feature_stats.json"
        utility_scores_path = output_dir / "utility_scores.jsonl"
        answer_span_scores_path = output_dir / "answer_span_scores.jsonl"
        utility_selected_manifest_path = output_dir / "utility_selected_features.json"
        answer_span_selected_manifest_path = (
            output_dir / "answer_span_selected_features.json"
        )
        readout_selected_manifest_path = output_dir / "readout_selected_features.json"
        selector_scoring_input_payload = _selector_scoring_input_payload(
            args,
            extraction_metadata=extraction_metadata,
            candidate_pool=candidate_pool,
            split_metadata=split_metadata,
        )
        selector_scoring_input_hash = _sha256_hexdigest(selector_scoring_input_payload)
        selector_scoring_cache = _resolve_selector_scoring_cache(
            output_dir=output_dir,
            feature_stats_path=feature_stats_path,
            full_sequence_stats_path=full_sequence_stats_path,
            utility_scores_path=utility_scores_path,
            answer_span_scores_path=answer_span_scores_path,
            utility_selected_manifest_path=utility_selected_manifest_path,
            answer_span_selected_manifest_path=answer_span_selected_manifest_path,
            readout_selected_manifest_path=readout_selected_manifest_path,
            current_input_hash=selector_scoring_input_hash,
            current_input_payload=selector_scoring_input_payload,
            args=args,
        )

        if selector_scoring_cache["cache_status"] == "reused":
            feature_stats_payload = _load_json(feature_stats_path)
            stats_by_flat_idx = {
                int(row["flat_idx"]): row for row in feature_stats_payload["features"]
            }
            full_sequence_stats_payload = _load_json(full_sequence_stats_path)
            full_sequence_stats_by_flat_idx = {
                int(row["flat_idx"]): row
                for row in full_sequence_stats_payload["features"]
            }
            utility_rows = _load_jsonl(utility_scores_path)
            answer_span_rows = _load_jsonl(answer_span_scores_path)
            utility_selected = _load_json(utility_selected_manifest_path)["features"]
            answer_span_selected = _load_json(answer_span_selected_manifest_path)[
                "features"
            ]
            readout_selected = _load_json(readout_selected_manifest_path)["features"]
        else:
            model, tokenizer = load_model_and_tokenizer(
                args.model_path, args.device_map
            )
            prompt_cache = build_prompt_cache(
                tokenizer,
                validation_samples,
                prompt_style=args.prompt_style,
            )

            sae_layers = sorted({int(feature["layer"]) for feature in candidate_pool})
            saes = load_saes(
                extraction_metadata["sae_release"],
                sae_layers,
                extraction_metadata["sae_width"],
                extraction_metadata["sae_l0"],
                str(next(model.parameters()).device),
            )
            stats_by_flat_idx = collect_prompt_end_feature_stats(
                model,
                saes,
                validation_samples,
                prompt_cache,
                layer_indices=layer_indices,
                d_sae=d_sae,
            )
            _write_json(
                feature_stats_path,
                {
                    "schema_version": "sae_feature_stats/v1",
                    "benchmark": "faitheval",
                    "prompt_style": args.prompt_style,
                    "n_validation_samples": len(validation_samples),
                    "features": [
                        stats_by_flat_idx[flat_idx]
                        for flat_idx in range(len(stats_by_flat_idx))
                    ],
                },
            )

            choice_token_cache = {
                str(sample["id"]): _choice_token_ids(
                    tokenizer, list(sample["valid_letters"])
                )
                for sample in validation_samples
            }
            answer_text_token_cache = build_faitheval_answer_text_target_token_cache(
                tokenizer,
                validation_samples,
                prompt_style=args.prompt_style,
                prompt_cache=prompt_cache,
            )
            baseline_margin_by_id = baseline_margins(
                model,
                validation_samples,
                prompt_cache,
                choice_token_cache,
            )
            baseline_answer_span_margin_by_id = baseline_answer_span_margins(
                model,
                validation_samples,
                prompt_cache,
                answer_text_token_cache,
            )
            selector_scoring_cache.update(
                _prepare_incremental_score_cache(
                    output_dir=output_dir,
                    current_input_hash=selector_scoring_input_hash,
                    score_paths=[utility_scores_path, answer_span_scores_path],
                )
            )
            utility_rows = _load_completed_score_rows(
                utility_scores_path,
                candidate_pool=candidate_pool,
                validation_n=len(validation_samples),
                required_score_fields=(
                    "selector_score",
                    "selector_score_std",
                    "selector_score_sum",
                ),
            )
            answer_span_rows = _load_completed_score_rows(
                answer_span_scores_path,
                candidate_pool=candidate_pool,
                validation_n=len(validation_samples),
                required_score_fields=(
                    "selector_score",
                    "selector_score_std",
                    "selector_score_sum",
                    "selector_score_full_span",
                    "selector_score_full_span_std",
                    "selector_score_full_span_sum",
                ),
            )
            completed_utility_flat_idxs = {int(row["flat_idx"]) for row in utility_rows}
            completed_answer_span_flat_idxs = {
                int(row["flat_idx"]) for row in answer_span_rows
            }
            reused_utility_score_rows = len(completed_utility_flat_idxs)
            reused_answer_span_score_rows = len(completed_answer_span_flat_idxs)
            for feature in tqdm(candidate_pool, desc="Utility scoring"):
                flat_idx = int(feature["flat_idx"])
                if flat_idx not in completed_utility_flat_idxs:
                    scored = score_candidate_feature(
                        model,
                        saes,
                        validation_samples,
                        prompt_cache,
                        choice_token_cache,
                        baseline_margin_by_id,
                        feature=feature,
                        alpha=args.alpha,
                    )
                    scored.update(stats_by_flat_idx[flat_idx])
                    utility_rows.append(scored)
                    completed_utility_flat_idxs.add(flat_idx)
                    _append_jsonl_row(utility_scores_path, scored)

                if flat_idx not in completed_answer_span_flat_idxs:
                    answer_span_scored = score_candidate_feature_answer_span(
                        model,
                        saes,
                        validation_samples,
                        prompt_cache,
                        answer_text_token_cache,
                        baseline_answer_span_margin_by_id,
                        feature=feature,
                        alpha=args.alpha,
                    )
                    answer_span_scored.update(stats_by_flat_idx[flat_idx])
                    answer_span_rows.append(answer_span_scored)
                    completed_answer_span_flat_idxs.add(flat_idx)
                    _append_jsonl_row(answer_span_scores_path, answer_span_scored)

            utility_rows.sort(
                key=lambda row: (-float(row["selector_score"]), int(row["flat_idx"]))
            )
            selector_scoring_cache["utility_score_rows_reused"] = (
                reused_utility_score_rows
            )
            selector_scoring_cache["utility_score_rows_computed"] = (
                len(completed_utility_flat_idxs) - reused_utility_score_rows
            )
            _write_jsonl(utility_scores_path, utility_rows)
            answer_span_rows.sort(
                key=lambda row: (-float(row["selector_score"]), int(row["flat_idx"]))
            )
            selector_scoring_cache["answer_span_score_rows_reused"] = (
                reused_answer_span_score_rows
            )
            selector_scoring_cache["answer_span_score_rows_computed"] = (
                len(completed_answer_span_flat_idxs) - reused_answer_span_score_rows
            )
            _write_jsonl(answer_span_scores_path, answer_span_rows)

            utility_selected = utility_rows[: args.top_k]
            utility_manifest = feature_manifest_with_selector_metadata(
                utility_selected,
                extraction_metadata=extraction_metadata,
                selector_name="validation_logprob_margin",
                split_metadata=split_metadata,
                alpha=args.alpha,
                prompt_style=args.prompt_style,
            )
            _write_json(utility_selected_manifest_path, utility_manifest)
            answer_span_selected = answer_span_rows[: args.top_k]
            answer_span_manifest = feature_manifest_with_selector_metadata(
                answer_span_selected,
                extraction_metadata=extraction_metadata,
                selector_name="validation_answer_span_margin_first3",
                split_metadata=split_metadata,
                alpha=args.alpha,
                prompt_style=args.prompt_style,
            )
            _write_json(answer_span_selected_manifest_path, answer_span_manifest)

            readout_selected = get_positive_sae_features_from_classifier(
                args.classifier_path,
                layer_indices=layer_indices,
                d_sae=d_sae,
            )
            if len(readout_selected) != args.top_k:
                raise ValueError(
                    "FaithEval held-out comparison requires matched feature-set "
                    f"sizes: expected {args.top_k} readout-selected SAE features, "
                    f"found {len(readout_selected)}"
                )
            readout_manifest = feature_manifest_with_selector_metadata(
                readout_selected,
                extraction_metadata=extraction_metadata,
                selector_name="classifier_positive_readout",
                split_metadata=split_metadata,
                alpha=args.alpha,
                prompt_style=args.prompt_style,
            )
            _write_json(readout_selected_manifest_path, readout_manifest)
            full_sequence_stats_by_flat_idx = collect_full_sequence_feature_stats(
                model,
                saes,
                validation_samples,
                prompt_cache,
                layer_indices=layer_indices,
                d_sae=d_sae,
            )
            _write_json(
                full_sequence_stats_path,
                {
                    "schema_version": "sae_full_sequence_feature_stats/v1",
                    "benchmark": "faitheval",
                    "prompt_style": args.prompt_style,
                    "n_validation_samples": len(validation_samples),
                    "n_validation_tokens": int(
                        next(
                            iter(full_sequence_stats_by_flat_idx.values()),
                        )["n_validation_tokens"]
                    ),
                    "features": [
                        full_sequence_stats_by_flat_idx[flat_idx]
                        for flat_idx in range(len(full_sequence_stats_by_flat_idx))
                    ],
                },
            )
        zero_weight_pool = [
            {
                **entry,
                **full_sequence_stats_by_flat_idx[int(entry["flat_idx"])],
            }
            for entry in build_sae_feature_entries(
                get_zero_weight_sae_feature_indices(args.classifier_path).tolist(),
                layer_indices=layer_indices,
                d_sae=d_sae,
            )
        ]
        matched_random_seed_diagnostics: dict[str, Any] = {}
        matched_random_answer_span_seed_diagnostics: dict[str, Any] = {}
        for seed in range(args.n_random_seeds):
            matched = match_random_zero_weight_features(
                utility_selected,
                zero_weight_pool,
                full_sequence_stats_by_flat_idx,
                seed=seed,
            )
            matched_fingerprint = fingerprint_ids(
                [_feature_id(feature) for feature in matched]
            )
            matched_manifest = feature_manifest_with_selector_metadata(
                matched,
                extraction_metadata=extraction_metadata,
                selector_name=f"layer_matched_zero_weight_seed_{seed}",
                split_metadata=split_metadata,
                alpha=args.alpha,
                prompt_style=args.prompt_style,
            )
            _write_json(matched_manifest_paths[seed], matched_manifest)
            matched_random_seed_diagnostics[f"matched_random_seed_{seed}"] = {
                "k": len(matched),
                "fingerprint": matched_fingerprint,
                "layer_histogram": layer_histogram(matched),
            }
            matched_answer_span = match_random_zero_weight_features(
                answer_span_selected,
                zero_weight_pool,
                full_sequence_stats_by_flat_idx,
                seed=seed,
            )
            matched_answer_span_fingerprint = fingerprint_ids(
                [_feature_id(feature) for feature in matched_answer_span]
            )
            matched_answer_span_manifest = feature_manifest_with_selector_metadata(
                matched_answer_span,
                extraction_metadata=extraction_metadata,
                selector_name=f"layer_matched_zero_weight_answer_span_seed_{seed}",
                split_metadata=split_metadata,
                alpha=args.alpha,
                prompt_style=args.prompt_style,
            )
            _write_json(
                matched_answer_span_manifest_paths[seed],
                matched_answer_span_manifest,
            )
            matched_random_answer_span_seed_diagnostics[
                f"matched_random_answer_span_seed_{seed}"
            ] = {
                "k": len(matched_answer_span),
                "fingerprint": matched_answer_span_fingerprint,
                "layer_histogram": layer_histogram(matched_answer_span),
            }

        matched_zero_dead_feature = select_zero_dead_path_drift_control(
            zero_weight_pool,
            full_sequence_stats_by_flat_idx,
        )
        matched_zero_dead_manifest = feature_manifest_with_selector_metadata(
            [matched_zero_dead_feature],
            extraction_metadata=extraction_metadata,
            selector_name="layer20_zero_weight_dead_feature_path_drift_control",
            split_metadata=split_metadata,
            alpha=args.alpha,
            prompt_style=args.prompt_style,
        )
        _write_json(
            output_dir / f"{PATH_DRIFT_CONTROL_FAMILY}_features.json",
            matched_zero_dead_manifest,
        )

        outside_old_shortlist = [
            feature
            for feature in utility_selected
            if not bool(feature["in_old_shortlist"])
        ]
        answer_span_outside_old_shortlist = [
            feature
            for feature in answer_span_selected
            if not bool(feature["in_old_shortlist"])
        ]
        utility_readout_overlap = jaccard_overlap(utility_selected, readout_selected)
        answer_span_readout_overlap = jaccard_overlap(
            answer_span_selected,
            readout_selected,
        )
        answer_span_utility_overlap = jaccard_overlap(
            answer_span_selected,
            utility_selected,
        )
        eligible_zero_weight_pool = [
            feature
            for feature in zero_weight_pool
            if float(feature.get("token_activation_rate", 0.0)) > 0.0
        ]
        selector_summary = {
            "schema_version": "faitheval_sae_utility_selector/v6",
            "benchmark": "faitheval",
            "prompt_style": args.prompt_style,
            "sae_steering_mode": "delta_only",
            "alpha": float(args.alpha),
            "selector_scoring": {
                **selector_scoring_cache,
                "input_hash": selector_scoring_input_hash,
            },
            "selector_design": {
                "review_limitation_targets": ["L2", "L3"],
                "question": (
                    "Does an intervention-aware SAE selector recover a steerable "
                    "FaithEval feature set where readout-selected features fail?"
                ),
                "validation_policy": (
                    "Freeze a single stratified validation/test split; use "
                    "validation only for feature scoring and selection."
                ),
                "heldout_policy": (
                    "Run one locked held-out evaluation bundle on the test manifest."
                ),
                "layer_coverage_note": (
                    "Partial L3 closure only: selection searches all non-zero "
                    "probe-support features within the existing SAE extraction "
                    "layers, not a wider SAE sweep."
                ),
                "target_families": {
                    "readout_selected": (
                        "Top positive probe-weight SAE features from the original "
                        "FaithEval readout."
                    ),
                    "utility_selected": (
                        "Top-k SAE features ranked by validation reduction in "
                        "misleading-minus-preferred logprob margin."
                    ),
                    "answer_span_selected": (
                        "Top-k SAE features ranked by validation reduction in "
                        "counterfactual-minus-preferred answer-text logprob "
                        "margin over the first 3 assistant-content tokens."
                    ),
                    "matched_random": (
                        "Zero-weight SAE features sampled without replacement "
                        "from the token-active pool, exact-matched to the "
                        "utility-selected layer histogram, with within-layer "
                        "weights proportional to full-sequence token activation "
                        "rate on the frozen validation split."
                    ),
                    "matched_random_answer_span": (
                        "Zero-weight SAE features sampled without replacement "
                        "from the token-active pool, exact-matched to the "
                        "answer-span-selected layer histogram, with within-layer "
                        "weights proportional to full-sequence token activation "
                        "rate on the frozen validation split."
                    ),
                },
            },
            "selector_metric": {
                "name": "validation_logprob_margin",
                "margin_definition": "logp(counterfactual_key) - logp(preferred_key)",
                "utility_definition": "baseline_margin - ablated_margin",
                "aggregation": "validation_mean",
            },
            "answer_span_selector_metric": {
                "name": "validation_answer_span_margin_first3",
                "margin_definition": (
                    "logp(counterfactual_answer_text first 3 assistant-content "
                    "tokens) - logp(preferred_answer_text first 3 "
                    "assistant-content tokens)"
                ),
                "utility_definition": "baseline_margin_first3 - ablated_margin_first3",
                "aggregation": "validation_mean",
                "sensitivity_metric": (
                    "counterfactual_minus_preferred_answer_text_logprob_margin_full"
                ),
            },
            "split_metadata": split_metadata,
            "candidate_pool": {
                "name": "classifier_nonzero_support",
                "n_features": len(candidate_pool),
                "n_positive": int(np.sum(coefficients > 0)),
                "n_negative": int(np.sum(coefficients < 0)),
                "layer_histogram": layer_histogram(candidate_pool),
                "weight_sign_counts": weight_sign_counts(candidate_pool),
                "layer_indices": list(layer_indices),
                "fingerprint": fingerprint_ids(
                    [_feature_id(feature) for feature in candidate_pool]
                ),
            },
            "matched_random_controls": {
                "pool_name": "classifier_zero_weight",
                "eligibility_rule": "token_activation_rate > 0 on frozen validation split",
                "layer_matching": "exact utility_selected layer histogram",
                "sampling": "weighted_without_replacement_by_token_activation_rate_within_layer",
                "source_stats_artifact": "full_sequence_feature_stats.json",
                "n_random_seeds": int(args.n_random_seeds),
                "eligible_pool_n": len(eligible_zero_weight_pool),
                "eligible_pool_layer_histogram": layer_histogram(
                    eligible_zero_weight_pool
                ),
                "seed_families": matched_random_seed_diagnostics,
            },
            "matched_random_answer_span_controls": {
                "pool_name": "classifier_zero_weight",
                "eligibility_rule": "token_activation_rate > 0 on frozen validation split",
                "layer_matching": "exact answer_span_selected layer histogram",
                "sampling": "weighted_without_replacement_by_token_activation_rate_within_layer",
                "source_stats_artifact": "full_sequence_feature_stats.json",
                "n_random_seeds": int(args.n_random_seeds),
                "eligible_pool_n": len(eligible_zero_weight_pool),
                "eligible_pool_layer_histogram": layer_histogram(
                    eligible_zero_weight_pool
                ),
                "seed_families": matched_random_answer_span_seed_diagnostics,
            },
            "path_drift_control": {
                "family": PATH_DRIFT_CONTROL_FAMILY,
                "k": 1,
                "fingerprint": fingerprint_ids(
                    [_feature_id(matched_zero_dead_feature)]
                ),
                "selection_rule": (
                    "Smallest layer-20 flat_idx with zero classifier weight and "
                    "token_activation_rate == 0 on the frozen validation split."
                ),
                "feature": {
                    "flat_idx": int(matched_zero_dead_feature["flat_idx"]),
                    "layer": int(matched_zero_dead_feature["layer"]),
                    "feature": int(matched_zero_dead_feature["feature"]),
                    "token_activation_rate": float(
                        matched_zero_dead_feature["token_activation_rate"]
                    ),
                },
            },
            "families": {
                "utility_selected": {
                    "k": len(utility_selected),
                    "fingerprint": fingerprint_ids(
                        [_feature_id(feature) for feature in utility_selected]
                    ),
                    "layer_histogram": layer_histogram(utility_selected),
                    "mean_selector_score": float(
                        np.mean(
                            [
                                float(feature["selector_score"])
                                for feature in utility_selected
                            ]
                        )
                    ),
                    "weight_sign_counts": weight_sign_counts(utility_selected),
                    "outside_old_shortlist": {
                        "count": int(len(outside_old_shortlist)),
                        "fraction": (
                            float(len(outside_old_shortlist) / len(utility_selected))
                            if utility_selected
                            else 0.0
                        ),
                        "threshold": float(args.old_shortlist_threshold),
                    },
                },
                "answer_span_selected": {
                    "k": len(answer_span_selected),
                    "fingerprint": fingerprint_ids(
                        [_feature_id(feature) for feature in answer_span_selected]
                    ),
                    "layer_histogram": layer_histogram(answer_span_selected),
                    "mean_selector_score": float(
                        np.mean(
                            [
                                float(feature["selector_score"])
                                for feature in answer_span_selected
                            ]
                        )
                    ),
                    "mean_selector_score_full_span": float(
                        np.mean(
                            [
                                float(feature["selector_score_full_span"])
                                for feature in answer_span_selected
                            ]
                        )
                    ),
                    "weight_sign_counts": weight_sign_counts(answer_span_selected),
                    "outside_old_shortlist": {
                        "count": int(len(answer_span_outside_old_shortlist)),
                        "fraction": (
                            float(
                                len(answer_span_outside_old_shortlist)
                                / len(answer_span_selected)
                            )
                            if answer_span_selected
                            else 0.0
                        ),
                        "threshold": float(args.old_shortlist_threshold),
                    },
                },
                "readout_selected": {
                    "k": len(readout_selected),
                    "fingerprint": fingerprint_ids(
                        [_feature_id(feature) for feature in readout_selected]
                    ),
                    "layer_histogram": layer_histogram(readout_selected),
                    "weight_sign_counts": weight_sign_counts(readout_selected),
                    "old_shortlist_threshold": float(args.old_shortlist_threshold),
                    "old_shortlist_size": int(
                        np.sum(np.abs(coefficients) > args.old_shortlist_threshold)
                    ),
                },
            },
            "family_overlap": {
                "utility_selected_vs_readout_selected": utility_readout_overlap,
                "answer_span_selected_vs_readout_selected": (
                    answer_span_readout_overlap
                ),
                "answer_span_selected_vs_utility_selected": (
                    answer_span_utility_overlap
                ),
            },
        }
        _write_json(output_dir / "selector_summary.json", selector_summary)

        provenance_extra["validation_n"] = len(validation_samples)
        provenance_extra["test_n"] = len(test_samples)
        provenance_extra["selected_k"] = len(utility_selected)
        provenance_extra["answer_span_selected_k"] = len(answer_span_selected)
        provenance_extra["candidate_n"] = len(candidate_pool)
        provenance_extra["n_random_seeds"] = int(args.n_random_seeds)
        provenance_extra["selector_scoring_cache_status"] = selector_scoring_cache[
            "cache_status"
        ]
    except BaseException as exc:
        provenance_status = provenance_status_for_exception(exc)
        provenance_extra["error"] = provenance_error_message(exc)
        raise
    finally:
        finish_run_provenance(provenance_handle, provenance_status, provenance_extra)


if __name__ == "__main__":
    main()
