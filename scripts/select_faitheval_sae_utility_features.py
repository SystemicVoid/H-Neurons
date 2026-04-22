"""Select SAE steering features for FaithEval via validation logprob margins.

Locked selector for this diagnostic:
- candidate pool: all 509 non-zero SAE probe-support features
- prompt format: FaithEval anti-compliance prompt wrapped in the same chat template
- operator/sign family: SAE delta-only ablation at alpha=0.0
- utility metric: mean reduction in misleading-vs-preferred logprob margin
- selected set size: k=266
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
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
    FAITHEVAL_CANONICAL_LABELS,
    _choice_token_ids,
    _faitheval_prompt,
    load_faitheval,
    load_model_and_tokenizer,
    misleading_margin,
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
DEFAULT_PROMPT_STYLE = "anti_compliance"
DEFAULT_ALPHA = 0.0
DEFAULT_OLD_SHORTLIST_THRESHOLD = 1e-3


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
    path.write_text(json_dumps(payload), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True))
            handle.write("\n")


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


def layer_histogram(features: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(int(feature["layer"]) for feature in features)
    return {str(layer): int(counts[layer]) for layer in sorted(counts)}


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
        pool_by_layer[int(feature["layer"])].append(feature)

    selected_by_layer: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for feature in selected_features:
        selected_by_layer[int(feature["layer"])].append(feature)

    matched: list[dict[str, Any]] = []
    for layer in sorted(selected_by_layer):
        pool = list(pool_by_layer[layer])
        if len(pool) < len(selected_by_layer[layer]):
            raise ValueError(
                f"Layer {layer} only has {len(pool)} zero-weight controls for "
                f"{len(selected_by_layer[layer])} selected features"
            )
        targets = list(selected_by_layer[layer])
        rng.shuffle(targets)
        for target in targets:
            target_stats = feature_stats[int(target["flat_idx"])]
            best_idx = None
            best_distance = None
            for idx, candidate in enumerate(pool):
                candidate_stats = feature_stats[int(candidate["flat_idx"])]
                distance = float(
                    np.hypot(
                        float(candidate_stats["activation_frequency"])
                        - float(target_stats["activation_frequency"]),
                        float(candidate_stats["decoder_norm"])
                        - float(target_stats["decoder_norm"]),
                    )
                )
                candidate_key = (distance, int(candidate["flat_idx"]))
                if best_distance is None or candidate_key < best_distance:
                    best_distance = candidate_key
                    best_idx = idx
            assert best_idx is not None
            matched.append(pool.pop(best_idx))
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


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_targets = [
        output_dir,
        output_dir / "candidate_pool.json",
        output_dir / "validation_manifest.json",
        output_dir / "test_manifest.json",
        output_dir / "feature_stats.json",
        output_dir / "utility_scores.jsonl",
        output_dir / "utility_selected_features.json",
        output_dir / "readout_selected_features.json",
        output_dir / "matched_random_seed_0_features.json",
        output_dir / "matched_random_seed_1_features.json",
        output_dir / "matched_random_seed_2_features.json",
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

        model, tokenizer = load_model_and_tokenizer(args.model_path, args.device_map)
        prompt_cache = build_prompt_cache(
            tokenizer,
            validation_samples,
            prompt_style=args.prompt_style,
        )
        choice_token_cache = {
            str(sample["id"]): _choice_token_ids(
                tokenizer, list(sample["valid_letters"])
            )
            for sample in validation_samples
        }

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
            output_dir / "feature_stats.json",
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

        baseline_margin_by_id = baseline_margins(
            model,
            validation_samples,
            prompt_cache,
            choice_token_cache,
        )
        utility_rows: list[dict[str, Any]] = []
        for feature in tqdm(candidate_pool, desc="Utility scoring"):
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
            scored.update(stats_by_flat_idx[int(feature["flat_idx"])])
            utility_rows.append(scored)

        utility_rows.sort(
            key=lambda row: (-float(row["selector_score"]), int(row["flat_idx"]))
        )
        _write_jsonl(output_dir / "utility_scores.jsonl", utility_rows)

        utility_selected = utility_rows[: args.top_k]
        utility_manifest = feature_manifest_with_selector_metadata(
            utility_selected,
            extraction_metadata=extraction_metadata,
            selector_name="validation_logprob_margin",
            split_metadata=split_metadata,
            alpha=args.alpha,
            prompt_style=args.prompt_style,
        )
        _write_json(output_dir / "utility_selected_features.json", utility_manifest)

        zero_weight_pool = [
            {
                **entry,
                **stats_by_flat_idx[int(entry["flat_idx"])],
            }
            for entry in build_sae_feature_entries(
                get_zero_weight_sae_feature_indices(args.classifier_path).tolist(),
                layer_indices=layer_indices,
                d_sae=d_sae,
            )
        ]
        readout_selected = get_positive_sae_features_from_classifier(
            args.classifier_path,
            layer_indices=layer_indices,
            d_sae=d_sae,
        )
        if len(readout_selected) != args.top_k:
            raise ValueError(
                "FaithEval held-out comparison requires matched feature-set sizes: "
                f"expected {args.top_k} readout-selected SAE features, "
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
        _write_json(output_dir / "readout_selected_features.json", readout_manifest)

        for seed in range(3):
            matched = match_random_zero_weight_features(
                utility_selected,
                zero_weight_pool,
                stats_by_flat_idx,
                seed=seed,
            )
            matched_manifest = feature_manifest_with_selector_metadata(
                matched,
                extraction_metadata=extraction_metadata,
                selector_name=f"layer_matched_zero_weight_seed_{seed}",
                split_metadata=split_metadata,
                alpha=args.alpha,
                prompt_style=args.prompt_style,
            )
            _write_json(
                output_dir / f"matched_random_seed_{seed}_features.json",
                matched_manifest,
            )

        outside_old_shortlist = [
            feature
            for feature in utility_selected
            if not bool(feature["in_old_shortlist"])
        ]
        overlap = jaccard_overlap(utility_selected, readout_selected)
        selector_summary = {
            "schema_version": "faitheval_sae_utility_selector/v2",
            "benchmark": "faitheval",
            "prompt_style": args.prompt_style,
            "sae_steering_mode": "delta_only",
            "alpha": float(args.alpha),
            "selector_metric": {
                "name": "validation_logprob_margin",
                "margin_definition": "logp(counterfactual_key) - logp(preferred_key)",
                "utility_definition": "baseline_margin - ablated_margin",
                "aggregation": "validation_mean",
            },
            "split_metadata": split_metadata,
            "candidate_pool": {
                "name": "classifier_nonzero_support",
                "n_features": len(candidate_pool),
                "n_positive": int(np.sum(coefficients > 0)),
                "n_negative": int(np.sum(coefficients < 0)),
                "fingerprint": fingerprint_ids(
                    [_feature_id(feature) for feature in candidate_pool]
                ),
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
                "readout_selected": {
                    "k": len(readout_selected),
                    "fingerprint": fingerprint_ids(
                        [_feature_id(feature) for feature in readout_selected]
                    ),
                    "layer_histogram": layer_histogram(readout_selected),
                    "old_shortlist_threshold": float(args.old_shortlist_threshold),
                    "old_shortlist_size": int(
                        np.sum(np.abs(coefficients) > args.old_shortlist_threshold)
                    ),
                },
            },
            "family_overlap": {
                "utility_selected_vs_readout_selected": overlap,
            },
        }
        _write_json(output_dir / "selector_summary.json", selector_summary)

        provenance_extra["validation_n"] = len(validation_samples)
        provenance_extra["test_n"] = len(test_samples)
        provenance_extra["selected_k"] = len(utility_selected)
        provenance_extra["candidate_n"] = len(candidate_pool)
    except BaseException as exc:
        provenance_status = provenance_status_for_exception(exc)
        provenance_extra["error"] = provenance_error_message(exc)
        raise
    finally:
        finish_run_provenance(provenance_handle, provenance_status, provenance_extra)


if __name__ == "__main__":
    main()
