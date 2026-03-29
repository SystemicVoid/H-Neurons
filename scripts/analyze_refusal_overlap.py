"""Analyze overlap between H-neuron interventions and refusal geometry.

This script implements the D3.5 refusal-overlap gate described in
``notes/act3-sprint.md`` and ``notes/measurement-blueprint.md``.

It reuses the existing benchmark prompt builders and stored Baseline A outputs,
then runs prompt-only forward passes to:

1. Reconstruct the actual H-neuron residual-stream update
2. Compare that update to canonical and bootstrap-subspace refusal geometry
3. Build a layer-matched null over random neuron sets
4. Test whether prompt-level overlap predicts FaithEval and Jailbreak outcomes
5. Emit a decision-oriented summary with bootstrap uncertainty
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass, field
import csv
import json
import math
from pathlib import Path
import sys
from typing import Any, Callable

import joblib
import numpy as np
from scipy.stats import spearmanr
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

from extract_direction import load_contrastive_splits
from intervene_model import get_h_neuron_indices
from run_intervention import (
    DEFAULT_CLASSIFIER_PATH,
    DEFAULT_DEVICE_MAP,
    DEFAULT_MODEL_PATH,
    JAILBREAK_TEMPLATES,
    _faitheval_prompt,
    load_faitheval,
    load_model_and_tokenizer,
)
from uncertainty import (
    DEFAULT_BOOTSTRAP_RESAMPLES,
    DEFAULT_BOOTSTRAP_SEED,
    percentile_interval,
)
from utils import (
    finish_run_provenance,
    format_alpha_label,
    json_dumps,
    provenance_error_message,
    provenance_status_for_exception,
    start_run_provenance,
)


FAITHEVAL_ALPHAS = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0)
JAILBREAK_ALPHAS = (0.0, 1.0, 1.5, 3.0)
DEFAULT_REFUSAL_DIRECTION_PATH = (
    "data/contrastive/refusal/directions/refusal_directions.pt"
)
DEFAULT_CONTRASTIVE_PATH = "data/contrastive/refusal/refusal_contrastive.jsonl"
DEFAULT_FAITHEVAL_DIR = "data/gemma3_4b/intervention/faitheval/experiment"
DEFAULT_JAILBREAK_DIR = "data/gemma3_4b/intervention/jailbreak/csv2_evaluation"
DEFAULT_OUTPUT_DIR = "data/gemma3_4b/intervention/refusal_overlap/analysis"
DEFAULT_NULL_SAMPLES = 100
DEFAULT_CONFIDENCE = 0.95


@dataclass(frozen=True)
class PromptRecord:
    prompt_id: str
    benchmark: str
    prompt: str
    metadata: dict[str, Any] = field(default_factory=dict)
    needs_mlp: bool = False
    needs_residual: bool = False


@dataclass
class CollectedActivations:
    mlp_prompt_ids: list[str]
    residual_prompt_ids: list[str]
    mlp_inputs_by_layer: dict[int, torch.Tensor]
    residuals_by_layer: dict[int, torch.Tensor]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--classifier_path", default=DEFAULT_CLASSIFIER_PATH)
    parser.add_argument(
        "--refusal_direction_path",
        default=DEFAULT_REFUSAL_DIRECTION_PATH,
    )
    parser.add_argument(
        "--contrastive_path",
        default=DEFAULT_CONTRASTIVE_PATH,
    )
    parser.add_argument(
        "--faitheval_dir",
        default=DEFAULT_FAITHEVAL_DIR,
    )
    parser.add_argument(
        "--jailbreak_dir",
        default=DEFAULT_JAILBREAK_DIR,
    )
    parser.add_argument(
        "--output_dir",
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--n_null",
        type=int,
        default=DEFAULT_NULL_SAMPLES,
        help="Number of layer-matched random neuron sets for the null distribution.",
    )
    parser.add_argument(
        "--n_bootstrap",
        type=int,
        default=DEFAULT_BOOTSTRAP_RESAMPLES,
        help="Bootstrap resamples for prompt uncertainty and refusal subspace PCA.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_BOOTSTRAP_SEED,
    )
    parser.add_argument(
        "--device_map",
        default=DEFAULT_DEVICE_MAP,
    )
    return parser.parse_args()


def _get_text_config(model: torch.nn.Module) -> Any:
    return getattr(model.config, "text_config", model.config)


def _get_decoder_layers(model: torch.nn.Module) -> Any:
    inner = getattr(model, "model")
    language_model = getattr(inner, "language_model", None)
    return language_model.layers if language_model is not None else inner.layers


def _extract_layer_idx(name: str) -> int | None:
    for part in name.split("."):
        if part.isdigit():
            return int(part)
    return None


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_alpha_results(
    input_dir: str | Path,
    alphas: tuple[float, ...],
) -> dict[float, list[dict[str, Any]]]:
    base = Path(input_dir)
    results: dict[float, list[dict[str, Any]]] = {}
    for alpha in alphas:
        path = base / f"alpha_{format_alpha_label(alpha)}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"Missing alpha file: {path}")
        results[alpha] = _load_jsonl(path)
    return results


def _ensure_same_prompt_ids(
    rows_by_alpha: dict[float, list[dict[str, Any]]],
) -> list[str]:
    ordered_ids: list[str] | None = None
    for alpha, rows in rows_by_alpha.items():
        current_ids = [row["id"] for row in rows]
        if ordered_ids is None:
            ordered_ids = current_ids
            continue
        if current_ids != ordered_ids:
            raise ValueError(
                f"Prompt IDs differ across alpha files; mismatch detected at alpha={alpha}"
            )
    if ordered_ids is None:
        raise ValueError("No alpha rows loaded")
    return ordered_ids


def build_faitheval_prompt_records(
    rows_by_alpha: dict[float, list[dict[str, Any]]],
    *,
    samples: list[dict[str, Any]] | None = None,
) -> list[PromptRecord]:
    ordered_ids = _ensure_same_prompt_ids(rows_by_alpha)
    sample_rows = load_faitheval() if samples is None else samples
    sample_by_id = {sample["id"]: sample for sample in sample_rows}
    records: list[PromptRecord] = []
    for prompt_id in ordered_ids:
        sample = sample_by_id.get(prompt_id)
        if sample is None:
            raise KeyError(f"FaithEval sample ID missing from dataset: {prompt_id}")
        records.append(
            PromptRecord(
                prompt_id=prompt_id,
                benchmark="faitheval",
                prompt=_faitheval_prompt(sample, prompt_style="anti_compliance"),
                metadata={
                    "question": sample["question"],
                    "counterfactual_key": sample["counterfactual_key"],
                },
                needs_mlp=True,
            )
        )
    return records


def build_jailbreak_prompt_records(
    rows_by_alpha: dict[float, list[dict[str, Any]]],
) -> list[PromptRecord]:
    reference_rows = rows_by_alpha[JAILBREAK_ALPHAS[0]]
    records: list[PromptRecord] = []
    for row in reference_rows:
        prompt = JAILBREAK_TEMPLATES[row["template_idx"]].format(goal=row["goal"])
        records.append(
            PromptRecord(
                prompt_id=row["id"],
                benchmark="jailbreak",
                prompt=prompt,
                metadata={
                    "goal": row["goal"],
                    "category": row["category"],
                    "template_idx": row["template_idx"],
                },
                needs_mlp=True,
            )
        )
    return records


def build_contrastive_train_prompt_records(
    contrastive_path: str | Path,
) -> list[PromptRecord]:
    splits = load_contrastive_splits(str(contrastive_path))
    records: list[PromptRecord] = []
    for key in ("train_harmful", "train_harmless"):
        label_records = splits.get(key, [])
        for row in label_records:
            records.append(
                PromptRecord(
                    prompt_id=row["id"],
                    benchmark="contrastive_refusal",
                    prompt=row["text"],
                    metadata={
                        "label": row["label"],
                        "split": row["split"],
                        "source": row["source"],
                    },
                    needs_residual=True,
                )
            )
    return records


def collect_prompt_activations(
    model: torch.nn.Module,
    tokenizer: Any,
    prompts: list[PromptRecord],
) -> CollectedActivations:
    mlp_store: dict[int, list[torch.Tensor]] = defaultdict(list)
    residual_store: dict[int, list[torch.Tensor]] = defaultdict(list)
    mlp_prompt_ids: list[str] = []
    residual_prompt_ids: list[str] = []

    captured_mlp: dict[int, torch.Tensor] = {}
    captured_residual: dict[int, torch.Tensor] = {}

    hooks = []
    decoder_layers = _get_decoder_layers(model)
    n_layers = len(decoder_layers)

    def make_mlp_hook(layer_idx: int):
        def hook_fn(module, args):
            x = args[0]
            captured_mlp[layer_idx] = x[0, -1, :].detach().float().cpu()
            return args

        return hook_fn

    def make_residual_hook(layer_idx: int):
        def hook_fn(module, _args, output):
            hidden_states = output[0] if isinstance(output, tuple) else output
            captured_residual[layer_idx] = (
                hidden_states[0, -1, :].detach().float().cpu()
            )

        return hook_fn

    for name, module in model.named_modules():
        if "down_proj" not in name or not isinstance(module, torch.nn.Linear):
            continue
        layer_idx = _extract_layer_idx(name)
        if layer_idx is None:
            continue
        hooks.append(module.register_forward_pre_hook(make_mlp_hook(layer_idx)))

    for layer_idx in range(n_layers):
        hooks.append(
            decoder_layers[layer_idx].register_forward_hook(
                make_residual_hook(layer_idx)
            )
        )

    model_device = next(model.parameters()).device
    try:
        for record in tqdm(prompts, desc="Collecting prompt activations"):
            messages = [{"role": "user", "content": record.prompt}]
            inputs = tokenizer.apply_chat_template(
                messages,
                return_tensors="pt",
                add_generation_prompt=True,
            )
            if hasattr(inputs, "input_ids"):
                input_ids = inputs["input_ids"]
            else:
                input_ids = inputs
            input_ids = input_ids.to(model_device)
            with torch.no_grad():
                model(input_ids)

            if record.needs_mlp:
                mlp_prompt_ids.append(record.prompt_id)
                for layer_idx in range(n_layers):
                    mlp_store[layer_idx].append(captured_mlp[layer_idx])
            if record.needs_residual:
                residual_prompt_ids.append(record.prompt_id)
                for layer_idx in range(n_layers):
                    residual_store[layer_idx].append(captured_residual[layer_idx])

            captured_mlp.clear()
            captured_residual.clear()
    finally:
        for hook in hooks:
            hook.remove()

    return CollectedActivations(
        mlp_prompt_ids=mlp_prompt_ids,
        residual_prompt_ids=residual_prompt_ids,
        mlp_inputs_by_layer={
            layer: torch.stack(values) for layer, values in mlp_store.items()
        },
        residuals_by_layer={
            layer: torch.stack(values) for layer, values in residual_store.items()
        },
    )


def load_refusal_directions(
    refusal_direction_path: str | Path,
) -> dict[int, torch.Tensor]:
    raw = torch.load(refusal_direction_path, map_location="cpu")
    directions: dict[int, torch.Tensor] = {}
    for key, value in raw.items():
        vector = value.detach().float().cpu()
        directions[int(key)] = vector / vector.norm()
    return directions


def build_refusal_subspace(
    residuals_by_layer: dict[int, torch.Tensor],
    labels: np.ndarray,
    *,
    n_bootstrap: int,
    seed: int,
    n_components: int = 3,
) -> dict[int, torch.Tensor]:
    harmful_idx = np.flatnonzero(labels == 1)
    harmless_idx = np.flatnonzero(labels == 0)
    if len(harmful_idx) == 0 or len(harmless_idx) == 0:
        raise ValueError(
            "Refusal subspace bootstrap requires both harmful and harmless prompts"
        )

    rng = np.random.default_rng(seed)
    subspaces: dict[int, torch.Tensor] = {}
    for layer, activations in tqdm(
        residuals_by_layer.items(),
        desc="Bootstrapping refusal subspaces",
    ):
        acts = activations.float()
        directions = torch.empty(
            (n_bootstrap, acts.shape[1]),
            dtype=torch.float32,
        )
        for sample_idx in range(n_bootstrap):
            harmful_sample = rng.choice(
                harmful_idx, size=len(harmful_idx), replace=True
            )
            harmless_sample = rng.choice(
                harmless_idx,
                size=len(harmless_idx),
                replace=True,
            )
            diff = acts[harmful_sample].mean(dim=0) - acts[harmless_sample].mean(dim=0)
            norm = diff.norm()
            directions[sample_idx] = diff / norm if norm > 0 else diff

        _, _, vh = torch.linalg.svd(directions, full_matrices=False)
        basis = vh[: min(n_components, vh.shape[0])].float()
        subspaces[layer] = basis
    return subspaces


def load_down_proj_modules(model: torch.nn.Module) -> dict[int, torch.nn.Linear]:
    modules: dict[int, torch.nn.Linear] = {}
    for name, module in model.named_modules():
        if "down_proj" not in name or not isinstance(module, torch.nn.Linear):
            continue
        layer_idx = _extract_layer_idx(name)
        if layer_idx is not None:
            modules[layer_idx] = module
    return modules


def project_down_proj_delta(
    mlp_inputs: torch.Tensor,
    down_proj_weight: torch.Tensor,
    neuron_indices: list[int] | np.ndarray,
    *,
    coefficients: torch.Tensor | np.ndarray | None = None,
) -> torch.Tensor:
    if len(neuron_indices) == 0:
        return torch.zeros(
            (mlp_inputs.shape[0], down_proj_weight.shape[0]),
            dtype=torch.float32,
        )

    if not isinstance(neuron_indices, torch.Tensor):
        index_tensor = torch.tensor(neuron_indices, dtype=torch.long)
    else:
        index_tensor = neuron_indices.to(dtype=torch.long)
    index_tensor = index_tensor.cpu()
    activations = mlp_inputs[:, index_tensor].float()
    weight = down_proj_weight.detach()
    columns = weight[:, index_tensor.to(weight.device)].transpose(0, 1).float().cpu()
    if coefficients is not None:
        coef_tensor = torch.as_tensor(coefficients, dtype=torch.float32).cpu()
        activations = activations * coef_tensor.unsqueeze(0)
    return activations @ columns


def compute_prompt_delta_by_layer(
    mlp_inputs_by_layer: dict[int, torch.Tensor],
    down_proj_modules: dict[int, torch.nn.Linear],
    neuron_map: dict[int, list[int]],
    *,
    coefficient_map: dict[int, torch.Tensor] | None = None,
) -> dict[int, torch.Tensor]:
    deltas: dict[int, torch.Tensor] = {}
    for layer_idx, neuron_indices in neuron_map.items():
        if layer_idx not in mlp_inputs_by_layer:
            continue
        coefficients = None
        if coefficient_map is not None and layer_idx in coefficient_map:
            coefficients = coefficient_map[layer_idx]
        deltas[layer_idx] = project_down_proj_delta(
            mlp_inputs_by_layer[layer_idx],
            down_proj_modules[layer_idx].weight.detach(),
            neuron_indices,
            coefficients=coefficients,
        )
    return deltas


def compute_overlap_statistics(
    delta_by_layer: dict[int, torch.Tensor],
    refusal_directions: dict[int, torch.Tensor],
    refusal_subspaces: dict[int, torch.Tensor],
    *,
    return_per_layer: bool = True,
) -> dict[str, Any]:
    if not delta_by_layer:
        raise ValueError("delta_by_layer is empty")

    prompt_count = next(iter(delta_by_layer.values())).shape[0]
    aggregate_dot = torch.zeros(prompt_count, dtype=torch.float32)
    aggregate_delta_norm2 = torch.zeros(prompt_count, dtype=torch.float32)
    aggregate_projected_norm2 = torch.zeros(prompt_count, dtype=torch.float32)
    refusal_norm2 = 0.0

    layer_signed: dict[int, np.ndarray] = {}
    layer_subspace: dict[int, np.ndarray] = {}

    all_layers = sorted(refusal_directions)
    for layer_idx in all_layers:
        direction = refusal_directions[layer_idx].float()
        basis = refusal_subspaces[layer_idx].float()
        delta = delta_by_layer.get(layer_idx)
        if delta is None:
            delta = torch.zeros(
                (prompt_count, direction.shape[0]),
                dtype=torch.float32,
            )
        else:
            delta = delta.float()

        norms2 = torch.sum(delta * delta, dim=1)
        dots = delta @ direction
        projected = delta @ basis.transpose(0, 1)
        projected_norm2 = torch.sum(projected * projected, dim=1)

        aggregate_dot += dots
        aggregate_delta_norm2 += norms2
        aggregate_projected_norm2 += projected_norm2
        refusal_norm2 += float(torch.sum(direction * direction).item())

        if return_per_layer:
            denom = torch.sqrt(norms2)
            signed = torch.where(denom > 0, dots / denom, torch.zeros_like(dots))
            subspace = torch.where(
                norms2 > 0,
                projected_norm2 / norms2,
                torch.zeros_like(projected_norm2),
            )
            layer_signed[layer_idx] = signed.numpy()
            layer_subspace[layer_idx] = subspace.numpy()

    aggregate_denom = torch.sqrt(aggregate_delta_norm2) * math.sqrt(refusal_norm2)
    aggregate_signed = torch.where(
        aggregate_denom > 0,
        aggregate_dot / aggregate_denom,
        torch.zeros_like(aggregate_dot),
    )
    aggregate_subspace = torch.where(
        aggregate_delta_norm2 > 0,
        aggregate_projected_norm2 / aggregate_delta_norm2,
        torch.zeros_like(aggregate_projected_norm2),
    )

    return {
        "prompt_signed_cosine": aggregate_signed.numpy(),
        "prompt_subspace_fraction": aggregate_subspace.numpy(),
        "prompt_delta_norm": torch.sqrt(aggregate_delta_norm2).numpy(),
        "layer_signed_cosine": layer_signed,
        "layer_subspace_fraction": layer_subspace,
    }


def sample_layer_matched_neuron_maps(
    counts_by_layer: dict[int, int],
    *,
    intermediate_size: int,
    n_samples: int,
    seed: int,
) -> list[dict[int, list[int]]]:
    rng = np.random.default_rng(seed)
    samples: list[dict[int, list[int]]] = []
    for _ in range(n_samples):
        sample: dict[int, list[int]] = {}
        for layer_idx, count in sorted(counts_by_layer.items()):
            chosen = rng.choice(intermediate_size, size=count, replace=False)
            sample[layer_idx] = sorted(int(idx) for idx in chosen.tolist())
        samples.append(sample)
    return samples


def build_coefficient_map(
    classifier: Any,
    neuron_map: dict[int, list[int]],
    *,
    intermediate_size: int,
) -> dict[int, torch.Tensor]:
    weights = np.asarray(classifier.coef_[0], dtype=np.float32)
    coefficient_map: dict[int, torch.Tensor] = {}
    for layer_idx, neuron_indices in neuron_map.items():
        flat_indices = [
            layer_idx * intermediate_size + neuron_idx for neuron_idx in neuron_indices
        ]
        coefficient_map[layer_idx] = torch.tensor(
            weights[flat_indices], dtype=torch.float32
        )
    return coefficient_map


def bootstrap_mean_summary(
    values: np.ndarray,
    *,
    n_bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    samples = np.empty(n_bootstrap, dtype=float)
    for idx in range(n_bootstrap):
        indices = rng.choice(len(values), size=len(values), replace=True)
        samples[idx] = float(np.mean(values[indices]))
    interval = percentile_interval(
        samples,
        DEFAULT_CONFIDENCE,
        method="bootstrap_percentile_prompt_mean",
    )
    return {
        "estimate": float(np.mean(values)),
        "ci": interval.to_dict(),
    }


def bootstrap_gap_summary(
    actual_values: np.ndarray,
    null_means: np.ndarray,
    *,
    n_bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    actual_values = np.asarray(actual_values, dtype=float)
    null_means = np.asarray(null_means, dtype=float)
    rng = np.random.default_rng(seed)
    samples = np.empty(n_bootstrap, dtype=float)
    for idx in range(n_bootstrap):
        actual_idx = rng.choice(
            len(actual_values), size=len(actual_values), replace=True
        )
        null_idx = int(rng.integers(0, len(null_means)))
        samples[idx] = float(np.mean(actual_values[actual_idx]) - null_means[null_idx])
    interval = percentile_interval(
        samples,
        DEFAULT_CONFIDENCE,
        method="bootstrap_percentile_actual_minus_null",
    )
    return {
        "estimate": float(np.mean(actual_values) - np.mean(null_means)),
        "ci": interval.to_dict(),
        "null_mean": float(np.mean(null_means)),
        "null_median": float(np.median(null_means)),
        "null_p95": float(np.quantile(null_means, 0.95)),
    }


def _spearman_value(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return float("nan")
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return float("nan")
    stat = spearmanr(x, y).statistic
    return float(stat) if stat is not None else float("nan")


def bootstrap_spearman_summary(
    x: np.ndarray,
    y: np.ndarray,
    *,
    n_bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    point = _spearman_value(x, y)
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(n_bootstrap):
        indices = rng.choice(len(x), size=len(x), replace=True)
        sample_value = _spearman_value(x[indices], y[indices])
        if not math.isnan(sample_value):
            samples.append(sample_value)
    if not samples:
        return {
            "estimate": point,
            "ci": None,
            "n": int(len(x)),
            "valid_bootstrap_resamples": 0,
        }
    interval = percentile_interval(
        np.asarray(samples, dtype=float),
        DEFAULT_CONFIDENCE,
        method="bootstrap_percentile_spearman",
    )
    return {
        "estimate": point,
        "ci": interval.to_dict(),
        "n": int(len(x)),
        "valid_bootstrap_resamples": int(len(samples)),
    }


def fit_prompt_slope(
    alphas: np.ndarray,
    values: np.ndarray,
) -> float:
    valid = ~np.isnan(values)
    if np.count_nonzero(valid) < 2:
        return float("nan")
    return float(np.polyfit(alphas[valid], values[valid], 1)[0])


def endpoint_delta(values: np.ndarray) -> float:
    if len(values) < 2 or math.isnan(values[0]) or math.isnan(values[-1]):
        return float("nan")
    return float(values[-1] - values[0])


def build_faitheval_outcomes(
    rows_by_alpha: dict[float, list[dict[str, Any]]],
) -> dict[str, dict[str, float]]:
    ordered_ids = _ensure_same_prompt_ids(rows_by_alpha)
    alpha_array = np.asarray(FAITHEVAL_ALPHAS, dtype=float)
    per_alpha = {
        alpha: {row["id"]: row for row in rows} for alpha, rows in rows_by_alpha.items()
    }
    outcomes: dict[str, dict[str, float]] = {}
    for prompt_id in ordered_ids:
        values = np.asarray(
            [
                float(bool(per_alpha[alpha][prompt_id]["compliance"]))
                for alpha in FAITHEVAL_ALPHAS
            ],
            dtype=float,
        )
        outcomes[prompt_id] = {
            "compliance_slope": fit_prompt_slope(alpha_array, values),
            "compliance_slope_pp_per_alpha": fit_prompt_slope(alpha_array, values)
            * 100.0,
            "compliance_delta_0_to_3": endpoint_delta(values),
            "compliance_delta_0_to_3_pp": endpoint_delta(values) * 100.0,
            "compliance_alpha_0.0": float(values[0]),
            "compliance_alpha_3.0": float(values[-1]),
        }
    return outcomes


def _csv2_value(row: dict[str, Any], key: str) -> float:
    csv2 = row.get("csv2", {})
    value = csv2.get(key)
    if value is None:
        return float("nan")
    return float(value)


def _csv2_yes(row: dict[str, Any]) -> float:
    return float(row.get("csv2", {}).get("harmful_binary") == "yes")


def _csv2_pivot_earliness(row: dict[str, Any]) -> float:
    value = row.get("csv2", {}).get("pivot_position")
    if value is None:
        return float("nan")
    return float(-value)


def build_jailbreak_outcomes(
    rows_by_alpha: dict[float, list[dict[str, Any]]],
) -> dict[str, dict[str, float]]:
    ordered_ids = _ensure_same_prompt_ids(rows_by_alpha)
    alpha_array = np.asarray(JAILBREAK_ALPHAS, dtype=float)
    per_alpha = {
        alpha: {row["id"]: row for row in rows} for alpha, rows in rows_by_alpha.items()
    }
    metrics: dict[str, Callable[[dict[str, Any]], float]] = {
        "csv2_yes": _csv2_yes,
        "C": lambda row: _csv2_value(row, "C"),
        "S": lambda row: _csv2_value(row, "S"),
        "V": lambda row: _csv2_value(row, "V"),
        "harmful_payload_share": lambda row: _csv2_value(row, "harmful_payload_share"),
        "pivot_earliness": _csv2_pivot_earliness,
    }
    outcomes: dict[str, dict[str, float]] = {}
    for prompt_id in ordered_ids:
        record: dict[str, float] = {}
        for metric_name, getter in metrics.items():
            values = np.asarray(
                [getter(per_alpha[alpha][prompt_id]) for alpha in JAILBREAK_ALPHAS],
                dtype=float,
            )
            slope = fit_prompt_slope(alpha_array, values)
            delta = endpoint_delta(values)
            record[f"{metric_name}_slope"] = slope
            record[f"{metric_name}_delta_0_to_3"] = delta
            record[f"{metric_name}_alpha_0.0"] = float(values[0])
            record[f"{metric_name}_alpha_3.0"] = float(values[-1])

        baseline = np.asarray(
            [record["csv2_yes_alpha_0.0"] for _ in range(1)],
            dtype=float,
        )
        _ = baseline  # placate lint for the scalar-style construction above
        outcomes[prompt_id] = record

    baseline_values = np.asarray(
        [outcomes[prompt_id]["csv2_yes_alpha_0.0"] for prompt_id in ordered_ids],
        dtype=float,
    )
    alpha_max_values = np.asarray(
        [outcomes[prompt_id]["csv2_yes_alpha_3.0"] for prompt_id in ordered_ids],
        dtype=float,
    )
    grouped_means = {
        baseline_value: float(
            np.mean(alpha_max_values[baseline_values == baseline_value])
        )
        for baseline_value in np.unique(baseline_values)
    }
    for prompt_id in ordered_ids:
        baseline_value = outcomes[prompt_id]["csv2_yes_alpha_0.0"]
        outcomes[prompt_id]["csv2_yes_alpha_3.0_conditioned_on_alpha_0.0"] = (
            outcomes[prompt_id]["csv2_yes_alpha_3.0"] - grouped_means[baseline_value]
        )
    return outcomes


def build_quintile_calibration(
    scores: np.ndarray,
    outcomes: np.ndarray,
) -> list[dict[str, Any]]:
    scores = np.asarray(scores, dtype=float)
    outcomes = np.asarray(outcomes, dtype=float)
    quantiles = np.quantile(scores, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    # Collapse duplicate edges to avoid empty buckets in tied-score regimes.
    edges = [quantiles[0]]
    for value in quantiles[1:]:
        if value > edges[-1]:
            edges.append(value)
    if len(edges) < 2:
        return [
            {
                "bucket": "all",
                "n": int(len(scores)),
                "score_mean": float(np.mean(scores)),
                "outcome_mean": float(np.mean(outcomes)),
            }
        ]

    table: list[dict[str, Any]] = []
    for idx in range(len(edges) - 1):
        lo = edges[idx]
        hi = edges[idx + 1]
        if idx == len(edges) - 2:
            mask = (scores >= lo) & (scores <= hi)
        else:
            mask = (scores >= lo) & (scores < hi)
        if not np.any(mask):
            continue
        table.append(
            {
                "bucket": f"q{idx + 1}",
                "n": int(np.count_nonzero(mask)),
                "score_mean": float(np.mean(scores[mask])),
                "outcome_mean": float(np.mean(outcomes[mask])),
            }
        )
    return table


def build_benchmark_summary(
    *,
    benchmark_name: str,
    prompt_scores: np.ndarray,
    prompt_subspace_scores: np.ndarray,
    outcomes: dict[str, dict[str, float]],
    primary_outcome_key: str,
    secondary_outcome_key: str,
    n_bootstrap: int,
    seed: int,
    diagnostic_keys: list[str] | None = None,
) -> dict[str, Any]:
    ordered_ids = list(outcomes)
    primary = np.asarray(
        [outcomes[prompt_id][primary_outcome_key] for prompt_id in ordered_ids]
    )
    secondary = np.asarray(
        [outcomes[prompt_id][secondary_outcome_key] for prompt_id in ordered_ids]
    )
    summary = {
        "benchmark": benchmark_name,
        "primary_outcome": primary_outcome_key,
        "secondary_outcome": secondary_outcome_key,
        "canonical_overlap_vs_primary": bootstrap_spearman_summary(
            prompt_scores,
            primary,
            n_bootstrap=n_bootstrap,
            seed=seed,
        ),
        "canonical_overlap_vs_secondary": bootstrap_spearman_summary(
            prompt_scores,
            secondary,
            n_bootstrap=n_bootstrap,
            seed=seed + 1,
        ),
        "subspace_overlap_vs_primary": bootstrap_spearman_summary(
            prompt_subspace_scores,
            primary,
            n_bootstrap=n_bootstrap,
            seed=seed + 2,
        ),
        "quintile_calibration": build_quintile_calibration(prompt_scores, primary),
    }
    if diagnostic_keys:
        diagnostics: dict[str, Any] = {}
        for idx, key in enumerate(diagnostic_keys, start=3):
            values = np.asarray([outcomes[prompt_id][key] for prompt_id in ordered_ids])
            valid = ~np.isnan(values)
            diagnostics[key] = bootstrap_spearman_summary(
                prompt_scores[valid],
                values[valid],
                n_bootstrap=n_bootstrap,
                seed=seed + idx,
            )
        summary["diagnostics"] = diagnostics
    return summary


def _ci_lower(summary: dict[str, Any] | None) -> float | None:
    if not summary or summary.get("ci") is None:
        return None
    return float(summary["ci"]["lower"])


def _ci_upper(summary: dict[str, Any] | None) -> float | None:
    if not summary or summary.get("ci") is None:
        return None
    return float(summary["ci"]["upper"])


def _ci_excludes_zero(summary: dict[str, Any] | None) -> bool:
    lower = _ci_lower(summary)
    upper = _ci_upper(summary)
    if lower is None or upper is None:
        return False
    return lower > 0.0 or upper < 0.0


def _ci_is_positive(summary: dict[str, Any] | None) -> bool:
    lower = _ci_lower(summary)
    return lower is not None and lower > 0.0


def _ci_is_negative(summary: dict[str, Any] | None) -> bool:
    upper = _ci_upper(summary)
    return upper is not None and upper < 0.0


def _supports_geometry(summary: dict[str, Any]) -> bool:
    # D2 stores direction = mean(harmful) - mean(harmless), so anti-refusal
    # alignment appears as a negative signed cosine.
    return _ci_is_negative(summary["canonical_overlap_gap_vs_null"]) or _ci_is_positive(
        summary["subspace_overlap_gap_vs_null"]
    )


def _supports_primary_mediation(summary: dict[str, Any]) -> bool:
    return _ci_is_negative(summary["canonical_overlap_vs_primary"]) or _ci_is_positive(
        summary["subspace_overlap_vs_primary"]
    )


def _directionally_consistent_after_exclusion(
    full_summary: dict[str, Any],
    excluded_summary: dict[str, Any],
) -> bool:
    canonical_full = float(full_summary["canonical_overlap_vs_primary"]["estimate"])
    canonical_excluded = float(
        excluded_summary["canonical_overlap_vs_primary"]["estimate"]
    )
    subspace_full = float(full_summary["subspace_overlap_vs_primary"]["estimate"])
    subspace_excluded = float(
        excluded_summary["subspace_overlap_vs_primary"]["estimate"]
    )

    canonical_consistent = canonical_full < 0.0 and canonical_excluded < 0.0
    subspace_consistent = subspace_full > 0.0 and subspace_excluded > 0.0
    return canonical_consistent or subspace_consistent


def decide_d4_gate(summary: dict[str, Any]) -> tuple[str, str]:
    headline_geometry = summary["headline_geometry"]
    faith_summary = summary["benchmarks"]["faitheval"]
    jailbreak_summary = summary["benchmarks"]["jailbreak"]
    dominant_layer_exclusion = summary.get("sensitivity", {}).get(
        "dominant_layer_exclusion"
    )

    geometry_supported = _supports_geometry(headline_geometry)
    faith_supported = _supports_primary_mediation(faith_summary)
    jailbreak_supported = _supports_primary_mediation(jailbreak_summary)
    robust_after_exclusion = False
    if dominant_layer_exclusion is not None:
        faith_robust = (
            not faith_supported
        ) or _directionally_consistent_after_exclusion(
            faith_summary,
            dominant_layer_exclusion["faitheval"],
        )
        jailbreak_robust = _directionally_consistent_after_exclusion(
            jailbreak_summary,
            dominant_layer_exclusion["jailbreak"],
        )
        robust_after_exclusion = faith_robust and jailbreak_robust

    if (
        geometry_supported
        and faith_supported
        and jailbreak_supported
        and robust_after_exclusion
    ):
        return "orthogonalize_d4_immediately", "Baseline A is refusal-mediated."
    if geometry_supported and jailbreak_supported and robust_after_exclusion:
        return (
            "keep_d4_as_planned_prioritize_d6_later",
            "Baseline A leaks into refusal only on safety externalities.",
        )
    return (
        "proceed_with_d4_unchanged",
        "Refusal overlap is too weak to explain Baseline A.",
    )


def write_prompt_scores_csv(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_layer_scores_csv(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_closeout_note(path: Path, summary: dict[str, Any]) -> None:
    geometry = summary["headline_geometry"]
    faith_summary = summary["benchmarks"]["faitheval"]
    jailbreak_summary = summary["benchmarks"]["jailbreak"]
    faith = faith_summary["canonical_overlap_vs_primary"]
    jailbreak = jailbreak_summary["canonical_overlap_vs_primary"]
    decision = summary["decision"]
    dominant_layer_exclusion = summary.get("sensitivity", {}).get(
        "dominant_layer_exclusion"
    )
    fragility_block = ""
    if dominant_layer_exclusion is not None:
        fragility_block = f"""
## Dominant-Layer Fragility Check

- Dominant layer by subspace gap: {dominant_layer_exclusion["excluded_layer"]}
- FaithEval Spearman after excluding dominant layer: {dominant_layer_exclusion["faitheval"]["canonical_overlap_vs_primary"]["estimate"]:.6f} (canonical), {dominant_layer_exclusion["faitheval"]["subspace_overlap_vs_primary"]["estimate"]:.6f} (subspace)
- Jailbreak Spearman after excluding dominant layer: {dominant_layer_exclusion["jailbreak"]["canonical_overlap_vs_primary"]["estimate"]:.6f} (canonical), {dominant_layer_exclusion["jailbreak"]["subspace_overlap_vs_primary"]["estimate"]:.6f} (subspace)
"""
    note = f"""# D3.5 Refusal-Overlap Closeout

## Geometry

- Canonical direction orientation: D2 stores `harmful - harmless`, so negative signed cosine means anti-refusal / harmless-ward alignment.
- Canonical signed cosine mean: {geometry["canonical_overlap"]["estimate"]:.6f}
- Canonical overlap gap vs null mean: {geometry["canonical_overlap_gap_vs_null"]["estimate"]:.6f}
- Refusal-subspace energy fraction mean: {geometry["subspace_overlap"]["estimate"]:.6f}
- Refusal-subspace gap vs null mean: {geometry["subspace_overlap_gap_vs_null"]["estimate"]:.6f}

## FaithEval Mediation

- Canonical Spearman(overlap, compliance slope): {faith["estimate"]:.6f}
- Refusal-subspace Spearman(overlap, compliance slope): {faith_summary["subspace_overlap_vs_primary"]["estimate"]:.6f}
- Secondary Spearman(overlap, endpoint delta): {faith_summary["canonical_overlap_vs_secondary"]["estimate"]:.6f}

## Jailbreak Externality

- Canonical Spearman(overlap, csv2_yes slope): {jailbreak["estimate"]:.6f}
- Refusal-subspace Spearman(overlap, csv2_yes slope): {jailbreak_summary["subspace_overlap_vs_primary"]["estimate"]:.6f}
- Secondary Spearman(overlap, endpoint delta): {jailbreak_summary["canonical_overlap_vs_secondary"]["estimate"]:.6f}
{fragility_block}

## D4 Gate

- Decision: {decision["d4_gate"]}
- Interpretation: {decision["statement"]}
"""
    path.write_text(note, encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "summary.json"
    prompt_scores_path = output_dir / "prompt_scores.csv"
    layer_scores_path = output_dir / "layer_scores.csv"
    null_distribution_path = output_dir / "null_distribution.json"
    closeout_path = output_dir / "closeout_note.md"

    provenance = start_run_provenance(
        args,
        primary_target=output_dir,
        output_targets=[
            summary_path,
            prompt_scores_path,
            layer_scores_path,
            null_distribution_path,
            closeout_path,
        ],
        extra={"analysis_name": "d3_5_refusal_overlap"},
        primary_target_is_dir=True,
    )

    try:
        classifier = joblib.load(args.classifier_path)
        model, tokenizer = load_model_and_tokenizer(
            args.model_path,
            device_map=args.device_map,
        )
        text_config = _get_text_config(model)
        intermediate_size = int(getattr(text_config, "intermediate_size"))
        n_layers = len(_get_decoder_layers(model))
        refusal_directions = load_refusal_directions(args.refusal_direction_path)
        neuron_map = get_h_neuron_indices(classifier, model.config)
        counts_by_layer = {layer: len(indices) for layer, indices in neuron_map.items()}
        coefficient_map = build_coefficient_map(
            classifier,
            neuron_map,
            intermediate_size=intermediate_size,
        )
        down_proj_modules = load_down_proj_modules(model)

        faitheval_rows = load_alpha_results(args.faitheval_dir, FAITHEVAL_ALPHAS)
        jailbreak_rows = load_alpha_results(args.jailbreak_dir, JAILBREAK_ALPHAS)
        faitheval_records = build_faitheval_prompt_records(faitheval_rows)
        jailbreak_records = build_jailbreak_prompt_records(jailbreak_rows)
        contrastive_records = build_contrastive_train_prompt_records(
            args.contrastive_path
        )

        all_prompts = [*faitheval_records, *jailbreak_records, *contrastive_records]
        activations = collect_prompt_activations(model, tokenizer, all_prompts)

        target_prompt_ids = [
            record.prompt_id for record in [*faitheval_records, *jailbreak_records]
        ]
        if activations.mlp_prompt_ids != target_prompt_ids:
            raise ValueError(
                "Collected MLP prompt order does not match benchmark prompt order"
            )

        contrastive_prompt_ids = [record.prompt_id for record in contrastive_records]
        if activations.residual_prompt_ids != contrastive_prompt_ids:
            raise ValueError(
                "Collected residual prompt order does not match contrastive prompt order"
            )

        contrastive_labels = np.asarray(
            [
                1 if record.metadata["label"] == "harmful" else 0
                for record in contrastive_records
            ],
            dtype=int,
        )
        refusal_subspaces = build_refusal_subspace(
            activations.residuals_by_layer,
            contrastive_labels,
            n_bootstrap=args.n_bootstrap,
            seed=args.seed,
        )
        stability_bootstrap = max(32, min(256, args.n_bootstrap // 5 or 32))
        refusal_subspaces_stability = build_refusal_subspace(
            activations.residuals_by_layer,
            contrastive_labels,
            n_bootstrap=stability_bootstrap,
            seed=args.seed + 10_000,
        )

        actual_delta = compute_prompt_delta_by_layer(
            activations.mlp_inputs_by_layer,
            down_proj_modules,
            neuron_map,
        )
        actual_scores = compute_overlap_statistics(
            actual_delta,
            refusal_directions,
            refusal_subspaces,
        )
        weighted_scores = compute_overlap_statistics(
            compute_prompt_delta_by_layer(
                activations.mlp_inputs_by_layer,
                down_proj_modules,
                neuron_map,
                coefficient_map=coefficient_map,
            ),
            refusal_directions,
            refusal_subspaces,
            return_per_layer=False,
        )
        stability_scores = compute_overlap_statistics(
            actual_delta,
            refusal_directions,
            refusal_subspaces_stability,
            return_per_layer=False,
        )

        null_maps = sample_layer_matched_neuron_maps(
            counts_by_layer,
            intermediate_size=intermediate_size,
            n_samples=args.n_null,
            seed=args.seed + 20_000,
        )
        null_canonical_means = np.empty(args.n_null, dtype=float)
        null_subspace_means = np.empty(args.n_null, dtype=float)
        null_layer_signed: dict[int, list[float]] = defaultdict(list)
        null_layer_subspace: dict[int, list[float]] = defaultdict(list)
        for sample_idx, null_map in enumerate(
            tqdm(null_maps, desc="Scoring matched null neuron sets")
        ):
            null_delta = compute_prompt_delta_by_layer(
                activations.mlp_inputs_by_layer,
                down_proj_modules,
                null_map,
            )
            null_scores = compute_overlap_statistics(
                null_delta,
                refusal_directions,
                refusal_subspaces,
            )
            null_canonical_means[sample_idx] = float(
                np.mean(null_scores["prompt_signed_cosine"])
            )
            null_subspace_means[sample_idx] = float(
                np.mean(null_scores["prompt_subspace_fraction"])
            )
            for layer_idx, values in null_scores["layer_signed_cosine"].items():
                null_layer_signed[layer_idx].append(float(np.mean(values)))
            for layer_idx, values in null_scores["layer_subspace_fraction"].items():
                null_layer_subspace[layer_idx].append(float(np.mean(values)))

        faitheval_outcomes = build_faitheval_outcomes(faitheval_rows)
        jailbreak_outcomes = build_jailbreak_outcomes(jailbreak_rows)

        prompt_rows: list[dict[str, Any]] = []
        prompt_id_to_index = {
            prompt_id: idx for idx, prompt_id in enumerate(target_prompt_ids)
        }
        actual_prompt_scores = actual_scores["prompt_signed_cosine"]
        actual_prompt_subspace = actual_scores["prompt_subspace_fraction"]
        weighted_prompt_scores = weighted_scores["prompt_signed_cosine"]
        weighted_prompt_subspace = weighted_scores["prompt_subspace_fraction"]
        stability_prompt_subspace = stability_scores["prompt_subspace_fraction"]

        for record in faitheval_records:
            idx = prompt_id_to_index[record.prompt_id]
            row = {
                "id": record.prompt_id,
                "benchmark": record.benchmark,
                "prompt": record.prompt,
                "overlap_signed_cosine": float(actual_prompt_scores[idx]),
                "overlap_subspace_fraction": float(actual_prompt_subspace[idx]),
                "weighted_overlap_signed_cosine": float(weighted_prompt_scores[idx]),
                "weighted_overlap_subspace_fraction": float(
                    weighted_prompt_subspace[idx]
                ),
                "stability_subspace_fraction": float(stability_prompt_subspace[idx]),
            }
            row.update(record.metadata)
            row.update(faitheval_outcomes[record.prompt_id])
            prompt_rows.append(row)

        for record in jailbreak_records:
            idx = prompt_id_to_index[record.prompt_id]
            row = {
                "id": record.prompt_id,
                "benchmark": record.benchmark,
                "prompt": record.prompt,
                "overlap_signed_cosine": float(actual_prompt_scores[idx]),
                "overlap_subspace_fraction": float(actual_prompt_subspace[idx]),
                "weighted_overlap_signed_cosine": float(weighted_prompt_scores[idx]),
                "weighted_overlap_subspace_fraction": float(
                    weighted_prompt_subspace[idx]
                ),
                "stability_subspace_fraction": float(stability_prompt_subspace[idx]),
            }
            row.update(record.metadata)
            row.update(jailbreak_outcomes[record.prompt_id])
            prompt_rows.append(row)
        write_prompt_scores_csv(prompt_scores_path, prompt_rows)

        total_target_prompts = len(target_prompt_ids)
        faitheval_slice = slice(0, len(faitheval_records))
        jailbreak_slice = slice(len(faitheval_records), total_target_prompts)

        faith_summary = build_benchmark_summary(
            benchmark_name="faitheval",
            prompt_scores=actual_prompt_scores[faitheval_slice],
            prompt_subspace_scores=actual_prompt_subspace[faitheval_slice],
            outcomes=faitheval_outcomes,
            primary_outcome_key="compliance_slope",
            secondary_outcome_key="compliance_delta_0_to_3",
            n_bootstrap=args.n_bootstrap,
            seed=args.seed + 30_000,
        )
        jailbreak_summary = build_benchmark_summary(
            benchmark_name="jailbreak",
            prompt_scores=actual_prompt_scores[jailbreak_slice],
            prompt_subspace_scores=actual_prompt_subspace[jailbreak_slice],
            outcomes=jailbreak_outcomes,
            primary_outcome_key="csv2_yes_slope",
            secondary_outcome_key="csv2_yes_delta_0_to_3",
            diagnostic_keys=[
                "C_slope",
                "S_slope",
                "V_slope",
                "harmful_payload_share_slope",
                "pivot_earliness_slope",
                "csv2_yes_alpha_3.0_conditioned_on_alpha_0.0",
            ],
            n_bootstrap=args.n_bootstrap,
            seed=args.seed + 40_000,
        )

        layer_rows: list[dict[str, Any]] = []
        for layer_idx in sorted(actual_scores["layer_signed_cosine"]):
            signed_values = actual_scores["layer_signed_cosine"][layer_idx]
            subspace_values = actual_scores["layer_subspace_fraction"][layer_idx]
            signed_summary = bootstrap_mean_summary(
                signed_values,
                n_bootstrap=args.n_bootstrap,
                seed=args.seed + 50_000 + layer_idx,
            )
            subspace_summary = bootstrap_mean_summary(
                subspace_values,
                n_bootstrap=args.n_bootstrap,
                seed=args.seed + 60_000 + layer_idx,
            )
            null_signed = np.asarray(null_layer_signed[layer_idx], dtype=float)
            null_subspace = np.asarray(null_layer_subspace[layer_idx], dtype=float)
            layer_rows.append(
                {
                    "layer": layer_idx,
                    "h_neuron_count": counts_by_layer.get(layer_idx, 0),
                    "signed_cosine_mean": signed_summary["estimate"],
                    "signed_cosine_ci_lower": signed_summary["ci"]["lower"],
                    "signed_cosine_ci_upper": signed_summary["ci"]["upper"],
                    "signed_cosine_null_mean": float(np.mean(null_signed)),
                    "signed_cosine_null_p95": float(np.quantile(null_signed, 0.95)),
                    "subspace_fraction_mean": subspace_summary["estimate"],
                    "subspace_fraction_ci_lower": subspace_summary["ci"]["lower"],
                    "subspace_fraction_ci_upper": subspace_summary["ci"]["upper"],
                    "subspace_fraction_null_mean": float(np.mean(null_subspace)),
                    "subspace_fraction_null_p95": float(
                        np.quantile(null_subspace, 0.95)
                    ),
                }
            )
        write_layer_scores_csv(layer_scores_path, layer_rows)

        dominant_layer_row = max(
            layer_rows,
            key=lambda row: (
                row["subspace_fraction_mean"] - row["subspace_fraction_null_mean"]
            ),
        )
        dominant_layer = int(dominant_layer_row["layer"])
        excluded_layer_scores = compute_overlap_statistics(
            {
                layer_idx: values
                for layer_idx, values in actual_delta.items()
                if layer_idx != dominant_layer
            },
            {
                layer_idx: values
                for layer_idx, values in refusal_directions.items()
                if layer_idx != dominant_layer
            },
            {
                layer_idx: values
                for layer_idx, values in refusal_subspaces.items()
                if layer_idx != dominant_layer
            },
            return_per_layer=False,
        )

        summary = {
            "analysis": {
                "model_path": args.model_path,
                "classifier_path": str(Path(args.classifier_path).resolve()),
                "refusal_direction_path": str(
                    Path(args.refusal_direction_path).resolve()
                ),
                "contrastive_path": str(Path(args.contrastive_path).resolve()),
                "faitheval_dir": str(Path(args.faitheval_dir).resolve()),
                "jailbreak_dir": str(Path(args.jailbreak_dir).resolve()),
                "output_dir": str(output_dir.resolve()),
                "n_layers": n_layers,
                "intermediate_size": intermediate_size,
                "n_h_neurons": int(sum(counts_by_layer.values())),
                "n_null": int(args.n_null),
                "n_bootstrap": int(args.n_bootstrap),
                "seed": int(args.seed),
                "stability_bootstrap_resamples": int(stability_bootstrap),
                "target_prompt_count": int(total_target_prompts),
                "contrastive_train_prompt_count": int(len(contrastive_records)),
                "canonical_direction_definition": "harmful_minus_harmless",
                "canonical_direction_interpretation": (
                    "Negative signed cosine means anti-refusal / harmless-ward "
                    "alignment because D2 stores harmful-minus-harmless directions."
                ),
            },
            "headline_geometry": {
                "canonical_overlap": bootstrap_mean_summary(
                    actual_prompt_scores,
                    n_bootstrap=args.n_bootstrap,
                    seed=args.seed + 70_000,
                ),
                "subspace_overlap": bootstrap_mean_summary(
                    actual_prompt_subspace,
                    n_bootstrap=args.n_bootstrap,
                    seed=args.seed + 71_000,
                ),
                "canonical_overlap_gap_vs_null": bootstrap_gap_summary(
                    actual_prompt_scores,
                    null_canonical_means,
                    n_bootstrap=args.n_bootstrap,
                    seed=args.seed + 72_000,
                ),
                "subspace_overlap_gap_vs_null": bootstrap_gap_summary(
                    actual_prompt_subspace,
                    null_subspace_means,
                    n_bootstrap=args.n_bootstrap,
                    seed=args.seed + 73_000,
                ),
            },
            "benchmarks": {
                "faitheval": faith_summary,
                "jailbreak": jailbreak_summary,
            },
            "sensitivity": {
                "classifier_weighted_geometry": {
                    "canonical_overlap": bootstrap_mean_summary(
                        weighted_prompt_scores,
                        n_bootstrap=args.n_bootstrap,
                        seed=args.seed + 74_000,
                    ),
                    "subspace_overlap": bootstrap_mean_summary(
                        weighted_prompt_subspace,
                        n_bootstrap=args.n_bootstrap,
                        seed=args.seed + 75_000,
                    ),
                },
                "basis_stability": {
                    "subspace_overlap": bootstrap_mean_summary(
                        stability_prompt_subspace,
                        n_bootstrap=args.n_bootstrap,
                        seed=args.seed + 76_000,
                    ),
                    "mean_abs_prompt_delta": float(
                        np.mean(
                            np.abs(stability_prompt_subspace - actual_prompt_subspace)
                        )
                    ),
                },
                "dominant_layer_exclusion": {
                    "excluded_layer": dominant_layer,
                    "selection_rule": "largest subspace gap vs matched null",
                    "faitheval": build_benchmark_summary(
                        benchmark_name="faitheval",
                        prompt_scores=excluded_layer_scores["prompt_signed_cosine"][
                            faitheval_slice
                        ],
                        prompt_subspace_scores=excluded_layer_scores[
                            "prompt_subspace_fraction"
                        ][faitheval_slice],
                        outcomes=faitheval_outcomes,
                        primary_outcome_key="compliance_slope",
                        secondary_outcome_key="compliance_delta_0_to_3",
                        n_bootstrap=args.n_bootstrap,
                        seed=args.seed + 77_000,
                    ),
                    "jailbreak": build_benchmark_summary(
                        benchmark_name="jailbreak",
                        prompt_scores=excluded_layer_scores["prompt_signed_cosine"][
                            jailbreak_slice
                        ],
                        prompt_subspace_scores=excluded_layer_scores[
                            "prompt_subspace_fraction"
                        ][jailbreak_slice],
                        outcomes=jailbreak_outcomes,
                        primary_outcome_key="csv2_yes_slope",
                        secondary_outcome_key="csv2_yes_delta_0_to_3",
                        diagnostic_keys=[
                            "C_slope",
                            "S_slope",
                            "V_slope",
                            "harmful_payload_share_slope",
                            "pivot_earliness_slope",
                            "csv2_yes_alpha_3.0_conditioned_on_alpha_0.0",
                        ],
                        n_bootstrap=args.n_bootstrap,
                        seed=args.seed + 78_000,
                    ),
                },
            },
        }
        d4_gate, statement = decide_d4_gate(summary)
        summary["decision"] = {
            "d4_gate": d4_gate,
            "statement": statement,
        }

        summary_path.write_text(json_dumps(summary), encoding="utf-8")
        null_distribution_path.write_text(
            json_dumps(
                {
                    "canonical_overlap_means": null_canonical_means.tolist(),
                    "subspace_overlap_means": null_subspace_means.tolist(),
                    "layer_signed_cosine_means": {
                        str(layer): values
                        for layer, values in sorted(null_layer_signed.items())
                    },
                    "layer_subspace_fraction_means": {
                        str(layer): values
                        for layer, values in sorted(null_layer_subspace.items())
                    },
                }
            ),
            encoding="utf-8",
        )
        write_closeout_note(closeout_path, summary)

        finish_run_provenance(
            provenance,
            status="completed",
            extra={
                "summary_path": str(summary_path.resolve()),
                "prompt_scores_path": str(prompt_scores_path.resolve()),
                "layer_scores_path": str(layer_scores_path.resolve()),
                "null_distribution_path": str(null_distribution_path.resolve()),
                "closeout_path": str(closeout_path.resolve()),
            },
        )
    except Exception as exc:
        finish_run_provenance(
            provenance,
            status=provenance_status_for_exception(exc),
            extra={"error": provenance_error_message(exc)},
        )
        raise


if __name__ == "__main__":
    main()
