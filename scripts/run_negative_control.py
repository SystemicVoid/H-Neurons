"""
Negative control experiment for H-Neuron specificity.

Tests whether scaling arbitrary neuron sets of the same size as the current
classifier's positive-weight H-neurons produces comparable compliance effects.
Supports FaithEval (anti-compliance prompt, inline MC evaluation) and FalseQA
(bare-question prompt, deferred GPT-4o judging via evaluate_intervention.py).

Usage:
    # FaithEval (default)
    uv run python scripts/run_negative_control.py --quick
    uv run python scripts/run_negative_control.py

    # FalseQA
    uv run python scripts/run_negative_control.py --benchmark falseqa --quick
    uv run python scripts/run_negative_control.py --benchmark falseqa
"""

import argparse
from collections import Counter
from dataclasses import dataclass
import json
import os
import sys
from typing import Any, cast

import joblib
import matplotlib
import numpy as np
import torch
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import AutoConfig

sys.path.insert(0, os.path.dirname(__file__))
from run_intervention import (
    CANONICAL_JAILBREAK_GENERATION,
    HNeuronScaler,
    _bioasq_prompt,
    _faitheval_prompt,
    extract_mc_answer,
    generate_response,
    load_bioasq,
    load_existing_ids,
    load_faitheval,
    load_falseqa,
    load_jailbreak,
    load_model_and_tokenizer,
    load_sample_manifest_ids,
    normalize_answer,
    run_jailbreak,
    tokenize_chat,
)
from model_registry import (
    assert_causal_lm_supported,
    default_classifier_path_for,
    dimensions_from_config,
    model_output_root,
    model_path_for,
    registered_model_dimensions,
)
from utils import (
    define_wandb_metrics,
    finish_run_provenance,
    init_wandb_run,
    log_wandb_files_as_artifact,
    provenance_error_message,
    provenance_status_for_exception,
    start_run_provenance,
)
from uncertainty import build_rate_summary, percentile_interval

ALL_ALPHAS = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
QUICK_ALPHAS = [0.0, 1.0, 3.0]
JAILBREAK_ALPHAS = [0.0, 1.0, 1.5, 3.0]
ControlConfig = tuple[str, str, int]

MODEL_PATH = None
CLASSIFIER_PATH = os.environ.get("HNEURONS_CLASSIFIER_PATH")
FALSEQA_DATA_PATH = "data/benchmarks/falseqa_test.csv"
BIOASQ_DATA_PATH = "data/benchmarks/bioasq13b_factoid.parquet"


def default_output_base(
    benchmark: str, model_key: str | None, model_path: str | None
) -> str:
    return (
        f"{model_output_root(model_key, model_path)}/intervention/{benchmark}/control"
    )


def default_h_neuron_baseline_dir(
    benchmark: str, model_key: str | None, model_path: str | None
) -> str:
    return (
        f"{model_output_root(model_key, model_path)}/intervention/"
        f"{benchmark}/experiment"
    )


def _jailbreak_csv2_eval_dir(
    *,
    user_baseline: str | None,
    model_key: str | None,
    model_path: str | None,
) -> str:
    """Suggest the H-neuron csv2_evaluation directory for the analyzer hint.

    Prefer a path derived from the user's --h_neuron_baseline (so custom output
    layouts and unregistered models work). Fall back to the registry default
    only when the user did not override; if even that fails (unregistered model
    without a baseline override), return a human-readable placeholder so the
    informational print never crashes a successful generation run.
    """
    if user_baseline:
        baseline_root = os.path.expanduser(user_baseline).rstrip("/")
        if baseline_root.endswith(".json"):
            baseline_root = os.path.dirname(baseline_root)
        parent = os.path.dirname(baseline_root) or "."
        return os.path.join(parent, "csv2_evaluation")
    try:
        return (
            f"{model_output_root(model_key, model_path)}"
            "/intervention/jailbreak/csv2_evaluation"
        )
    except ValueError:
        return "<experiment_dir — pass --h_neuron_baseline>"


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


@dataclass(frozen=True)
class ClassifierGeometry:
    num_hidden_layers: int
    intermediate_size: int
    total_ffn_neurons: int
    source: str


def _classifier_weights(classifier) -> np.ndarray:
    return np.asarray(classifier.coef_[0])


def _model_dimensions(model_path: str, model_key: str | None = None):
    dimensions = registered_model_dimensions(model_key, model_path)
    if dimensions is not None:
        return dimensions
    return dimensions_from_config(AutoConfig.from_pretrained(model_path))


def derive_classifier_geometry(
    classifier,
    *,
    model_path: str | None = None,
    model_key: str | None = None,
    dimensions=None,
) -> ClassifierGeometry:
    weights = _classifier_weights(classifier)
    n_features = int(weights.shape[0])
    dims = dimensions
    if dims is None:
        if model_path is None:
            raise ValueError("model_path is required when dimensions are not provided")
        dims = _model_dimensions(model_path, model_key)

    if n_features == dims.total_ffn_neurons:
        return ClassifierGeometry(
            num_hidden_layers=int(dims.num_hidden_layers),
            intermediate_size=int(dims.intermediate_size),
            total_ffn_neurons=n_features,
            source="registered_or_hf_config",
        )

    raise ValueError(
        "Classifier coefficient length does not match registered/HF model "
        f"dimensions: n_features={n_features}, config_layers={dims.num_hidden_layers}, "
        f"config_intermediate_size={dims.intermediate_size}, "
        f"expected={dims.total_ffn_neurons}. Check --model_key/--model_path "
        "and --classifier_path."
    )


def derive_h_neuron_layer_distribution(
    classifier, intermediate_size: int
) -> dict[int, int]:
    selected_flat_indices = np.where(_classifier_weights(classifier) > 0)[0]
    layers = (selected_flat_indices // intermediate_size).astype(int)
    return dict(Counter(layers.tolist()))


def zero_weight_indices_from_classifier(classifier) -> np.ndarray:
    return np.where(_classifier_weights(classifier) == 0)[0]


def get_zero_weight_indices(classifier_path: str) -> np.ndarray:
    """Get flat indices of all neurons with zero classifier weight."""
    clf = joblib.load(classifier_path)
    return zero_weight_indices_from_classifier(clf)


def get_nonzero_flat_indices(classifier_path: str) -> set:
    """Get flat indices of all neurons with non-zero classifier weight."""
    clf = joblib.load(classifier_path)
    return set(np.where(_classifier_weights(clf) != 0)[0].tolist())


def flat_to_neuron_map(flat_indices: np.ndarray, intermediate_size: int) -> dict:
    """Convert flat classifier indices to {layer_idx: [neuron_indices]}."""
    neuron_map = {}
    for idx in flat_indices:
        layer = int(idx // intermediate_size)
        neuron = int(idx % intermediate_size)
        neuron_map.setdefault(layer, []).append(neuron)
    return neuron_map


def sample_unconstrained(zero_indices: np.ndarray, seed: int, n: int) -> np.ndarray:
    """Sample n neurons uniformly at random from zero-weight positions."""
    if len(zero_indices) < n:
        raise ValueError(
            f"Need {n} zero-weight neurons but only {len(zero_indices)} are available."
        )
    rng = np.random.RandomState(seed)
    chosen = rng.choice(zero_indices, size=n, replace=False)
    return np.sort(chosen)


def sample_layer_matched(
    zero_indices: np.ndarray,
    seed: int,
    *,
    layer_distribution: dict[int, int],
    intermediate_size: int,
) -> np.ndarray:
    """Sample random neurons matching the H-neuron layer distribution."""
    rng = np.random.RandomState(seed)
    zero_set = set(zero_indices.tolist())

    chosen = []
    for layer, count in sorted(layer_distribution.items()):
        # All neuron indices in this layer (flat)
        layer_start = layer * intermediate_size
        layer_end = layer_start + intermediate_size
        eligible = [i for i in range(layer_start, layer_end) if i in zero_set]
        if len(eligible) < count:
            raise ValueError(
                f"Layer {layer}: need {count} zero-weight neurons but only {len(eligible)} available"
            )
        picks = rng.choice(eligible, size=count, replace=False)
        chosen.extend(picks)
    return np.sort(np.array(chosen))


def run_single_config(
    model,
    tokenizer,
    samples,
    neuron_map: dict,
    alphas: list,
    output_dir: str,
    config_name: str,
    benchmark: str = "faitheval",
    jailbreak_batch_size: int = 1,
    prompt_style: str = "anti_compliance",
):
    """Run a benchmark for one neuron set across all alphas."""
    os.makedirs(output_dir, exist_ok=True)

    indices_path = os.path.join(output_dir, "neuron_indices.json")
    with open(indices_path, "w") as f:
        json.dump({str(k): v for k, v in neuron_map.items()}, f, indent=2)

    device = next(model.parameters()).device
    scaler = HNeuronScaler(model, neuron_map, device)
    print(
        f"\n[{config_name}] Installed {scaler.n_hooks} hooks on {scaler.n_neurons} neurons"
    )

    if benchmark == "faitheval":
        results = _run_faitheval_alphas(
            model,
            tokenizer,
            samples,
            scaler,
            alphas,
            output_dir,
            config_name,
            prompt_style=prompt_style,
        )
    elif benchmark == "falseqa":
        results = _run_falseqa_alphas(
            model, tokenizer, samples, scaler, alphas, output_dir, config_name
        )
    elif benchmark == "bioasq":
        results = _run_bioasq_alphas(
            model, tokenizer, samples, scaler, alphas, output_dir, config_name
        )
    elif benchmark == "jailbreak":
        results = _run_jailbreak_alphas(
            model,
            tokenizer,
            samples,
            scaler,
            alphas,
            output_dir,
            config_name,
            jailbreak_batch_size=jailbreak_batch_size,
        )
    else:
        raise ValueError(f"Unknown benchmark: {benchmark}")

    summary = {
        "config": config_name,
        "benchmark": benchmark,
        "n_neurons": sum(len(v) for v in neuron_map.values()),
        "results": results,
    }
    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(summary, f, indent=2)

    scaler.remove()
    return results


def _run_faitheval_alphas(
    model,
    tokenizer,
    samples,
    scaler,
    alphas,
    output_dir,
    config_name,
    *,
    prompt_style: str = "anti_compliance",
):
    """FaithEval inline MC evaluation under the chosen prompt style."""
    from tqdm import tqdm

    results = {}
    for alpha in alphas:
        out_path = os.path.join(output_dir, f"alpha_{alpha:.1f}.jsonl")
        existing_ids = load_existing_ids(out_path)
        scaler.alpha = alpha
        total = 0

        for sample in tqdm(samples, desc=f"[{config_name}] α={alpha:.1f}"):
            if sample["id"] in existing_ids:
                total += 1
                continue

            prompt = _faitheval_prompt(sample, prompt_style)
            messages = [{"role": "user", "content": prompt}]
            response, _ = generate_response(
                model,
                tokenizer,
                messages,
                do_sample=False,
                max_new_tokens=256,
            )

            chosen = extract_mc_answer(response, sample["valid_letters"])
            total += 1
            record = {
                "id": sample["id"],
                "alpha": alpha,
                "question": sample["question"],
                "counterfactual_key": sample["counterfactual_key"],
                "chosen": chosen,
                "response": response,
                "compliance": chosen == sample["counterfactual_key"],
                "parse_failure": chosen is None,
            }
            with open(out_path, "a") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        comp_total, n_total, pf_total = _count_compliance_and_pf(out_path)
        results[str(alpha)] = {
            "compliance_rate": round(comp_total / n_total, 4) if n_total else 0,
            "n_compliant": comp_total,
            "n_total": n_total,
            "parse_failures": pf_total,
            "compliance": build_rate_summary(
                comp_total,
                n_total,
                count_key="n_compliant",
                total_key="n_total",
            ),
            "parse_failure": build_rate_summary(
                pf_total,
                n_total,
                count_key="count",
                total_key="n_total",
            ),
        }
        print(
            f"  α={alpha:.1f}: {comp_total / n_total:.1%} compliance "
            f"({comp_total}/{n_total}), {pf_total} parse failures"
        )
    return results


def _run_falseqa_alphas(
    model, tokenizer, samples, scaler, alphas, output_dir, config_name
):
    """FalseQA: bare-question prompt, no inline compliance (deferred to GPT-4o)."""
    from tqdm import tqdm

    results = {}
    for alpha in alphas:
        out_path = os.path.join(output_dir, f"alpha_{alpha:.1f}.jsonl")
        existing_ids = load_existing_ids(out_path)
        scaler.alpha = alpha
        total = 0

        for sample in tqdm(samples, desc=f"[{config_name}] α={alpha:.1f}"):
            if sample["id"] in existing_ids:
                total += 1
                continue

            messages = [{"role": "user", "content": sample["question"]}]
            response, _ = generate_response(
                model,
                tokenizer,
                messages,
                do_sample=False,
                max_new_tokens=256,
            )
            total += 1
            record = {
                "id": sample["id"],
                "alpha": alpha,
                "question": sample["question"],
                "response": response,
            }
            with open(out_path, "a") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        n_total = _count_lines(out_path)
        results[str(alpha)] = {"n_total": n_total}
        print(f"  α={alpha:.1f}: {n_total} responses generated (judging deferred)")
    return results


def _run_bioasq_alphas(
    model, tokenizer, samples, scaler, alphas, output_dir, config_name
):
    """BioASQ factoid QA: inline accuracy evaluation via normalize_answer matching."""
    from tqdm import tqdm

    results = {}
    for alpha in alphas:
        out_path = os.path.join(output_dir, f"alpha_{alpha:.1f}.jsonl")
        existing_ids = load_existing_ids(out_path)
        scaler.alpha = alpha
        total = 0

        for sample in tqdm(samples, desc=f"[{config_name}] α={alpha:.1f}"):
            if sample["id"] in existing_ids:
                total += 1
                continue

            prompt = _bioasq_prompt(sample)
            messages = [{"role": "user", "content": prompt}]
            response, _ = generate_response(
                model,
                tokenizer,
                messages,
                do_sample=False,
                max_new_tokens=128,
            )

            norm_gts = [normalize_answer(gt) for gt in sample["ground_truth"]]
            norm_resp = normalize_answer(response)
            correct = any(gt in norm_resp for gt in norm_gts if gt)
            total += 1

            record = {
                "id": sample["id"],
                "alpha": alpha,
                "question": sample["question"],
                "response": response,
                "ground_truth": sample["ground_truth"],
                "compliance": correct,
            }
            with open(out_path, "a") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        comp_total, n_total, pf_total = _count_compliance_and_pf(out_path)
        results[str(alpha)] = {
            "compliance_rate": round(comp_total / n_total, 4) if n_total else 0,
            "n_compliant": comp_total,
            "n_total": n_total,
            "parse_failures": pf_total,
            "compliance": build_rate_summary(
                comp_total,
                n_total,
                count_key="n_compliant",
                total_key="n_total",
            ),
        }
        print(
            f"  α={alpha:.1f}: {comp_total / n_total:.1%} accuracy "
            f"({comp_total}/{n_total})"
        )
    return results


def _run_jailbreak_alphas(
    model,
    tokenizer,
    samples,
    scaler,
    alphas,
    output_dir,
    config_name,
    jailbreak_batch_size=1,
):
    """Jailbreak negative control: generate responses, defer CSV-v2 scoring."""
    prompt_cache = {}
    for s in samples:
        msgs = [{"role": "user", "content": s["full_prompt"]}]
        prompt_cache[s["id"]] = tokenize_chat(tokenizer, msgs)
    print(f"  [{config_name}] Pre-tokenized {len(prompt_cache)} jailbreak prompts")

    results = {}
    for alpha in alphas:
        canonical_do_sample = cast(bool, CANONICAL_JAILBREAK_GENERATION["do_sample"])
        canonical_temperature = CANONICAL_JAILBREAK_GENERATION["temperature"]
        canonical_top_k = cast(int, CANONICAL_JAILBREAK_GENERATION["top_k"])
        canonical_top_p = CANONICAL_JAILBREAK_GENERATION["top_p"]
        run_jailbreak(
            model,
            tokenizer,
            scaler,
            samples,
            alpha,
            output_dir,
            max_samples=None,
            max_new_tokens=CANONICAL_JAILBREAK_GENERATION["max_new_tokens"],
            prompt_cache=prompt_cache,
            wandb_module=None,
            alpha_idx=0,
            throughput_state=None,
            benchmark_name="jailbreak_control",
            do_sample=canonical_do_sample,
            temperature=canonical_temperature,
            top_k=canonical_top_k,
            top_p=canonical_top_p,
            jailbreak_batch_size=jailbreak_batch_size,
        )
        out_path = os.path.join(output_dir, f"alpha_{alpha:.1f}.jsonl")
        n_total = _count_lines(out_path)
        results[str(alpha)] = {"n_total": n_total}
        print(
            f"  [{config_name}] α={alpha:.1f}: {n_total} responses "
            f"(CSV-v2 scoring deferred)"
        )
    return results


def _count_lines(path: str) -> int:
    """Count valid JSONL lines in a file."""
    count = 0
    with open(path) as f:
        for line in f:
            try:
                json.loads(line)
                count += 1
            except json.JSONDecodeError:
                continue
    return count


def _count_compliance_and_pf(path: str):
    """Count compliance and parse failures from a JSONL file."""
    compliant = 0
    total = 0
    parse_failures = 0
    with open(path) as f:
        for line in f:
            try:
                rec = json.loads(line)
                total += 1
                if rec.get("compliance"):
                    compliant += 1
                if rec.get("parse_failure"):
                    parse_failures += 1
            except json.JSONDecodeError:
                continue
    return compliant, total, parse_failures


def build_comparison_summary(
    all_results: dict,
    alphas: list,
    benchmark: str = "faitheval",
    baseline_path: str | None = None,
) -> dict:
    """Build the comparison summary JSON."""
    if baseline_path is None:
        raise ValueError("baseline_path is required for comparison summaries")
    resolved_baseline_path = resolve_results_json(baseline_path)
    with open(resolved_baseline_path) as f:
        h_baseline_raw = json.load(f)
    h_neuron_count = h_baseline_raw.get("n_h_neurons") or h_baseline_raw.get(
        "n_neurons"
    )

    h_rates = []
    for a in alphas:
        key = str(float(a))
        h_rates.append(h_baseline_raw["results"][key]["compliance_rate"])

    alpha_arr = np.array(alphas)
    h_arr = np.array(h_rates)
    h_slope = np.polyfit(alpha_arr, h_arr * 100, 1)[0]
    h_rho = stats.spearmanr(alpha_arr, h_arr).statistic

    summary = {
        "h_neuron_baseline": {
            "path": resolved_baseline_path,
            "n_h_neurons": h_neuron_count,
            "compliance_rates": h_rates,
            "slope_per_alpha": round(h_slope, 2),
            "spearman_rho": round(h_rho, 4),
            "parse_failures": [0] * len(alphas),
            "compliance_ci_by_alpha": [
                h_baseline_raw["results"][str(float(a))].get("compliance", None)
                or build_rate_summary(
                    h_baseline_raw["results"][str(float(a))]["n_compliant"],
                    h_baseline_raw["results"][str(float(a))]["n_total"],
                    count_key="n_compliant",
                    total_key="n_total",
                )
                for a in alphas
            ],
        },
    }

    # Group results by strategy
    for strategy in ["unconstrained", "layer_matched"]:
        seed_results = {k: v for k, v in all_results.items() if strategy in k}
        if not seed_results:
            continue

        all_rates = []
        all_pf = []
        per_seed = {}

        for name, res in sorted(seed_results.items()):
            rates = [res[str(float(a))]["compliance_rate"] for a in alphas]
            pfs = [res[str(float(a))].get("parse_failures", 0) for a in alphas]
            all_rates.append(rates)
            all_pf.append(pfs)

            r_arr = np.array(rates)
            slope = np.polyfit(alpha_arr, r_arr * 100, 1)[0]
            rho = stats.spearmanr(alpha_arr, r_arr).statistic

            per_seed[name] = {
                "compliance_rates": rates,
                "slope_per_alpha": round(slope, 2),
                "spearman_rho": round(rho, 4),
                "parse_failures": pfs,
                "compliance_ci_by_alpha": [
                    res[str(float(a))].get("compliance", None)
                    or build_rate_summary(
                        res[str(float(a))]["n_compliant"],
                        res[str(float(a))]["n_total"],
                        count_key="n_compliant",
                        total_key="n_total",
                    )
                    for a in alphas
                ],
            }

        rates_matrix = np.array(all_rates)
        pf_matrix = np.array(all_pf)
        mean_rates = rates_matrix.mean(axis=0).tolist()
        std_rates = rates_matrix.std(axis=0).tolist()
        mean_pf = pf_matrix.mean(axis=0).tolist()

        slopes = [per_seed[k]["slope_per_alpha"] for k in per_seed]
        rhos = [per_seed[k]["spearman_rho"] for k in per_seed]
        monotonic_count = sum(
            1
            for k in per_seed
            if all(
                per_seed[k]["compliance_rates"][i]
                <= per_seed[k]["compliance_rates"][i + 1]
                for i in range(len(alphas) - 1)
            )
        )

        key = f"{strategy}_random"
        summary[key] = {
            "mean_compliance_rates": [round(r, 4) for r in mean_rates],
            "std_compliance_rates": [round(s, 4) for s in std_rates],
            "per_seed": per_seed,
            "mean_slope_per_alpha": round(np.mean(slopes), 2),
            "slope_percentile_interval": percentile_interval(
                np.array(slopes, dtype=float),
                method="empirical_random_set_percentile",
            ).to_dict(),
            "mean_spearman_rho": round(np.mean(rhos), 4),
            "any_seed_monotonic": monotonic_count > 0,
            "mean_parse_failures": [round(p, 1) for p in mean_pf],
            "alpha_3_compliance_percentile_interval_pct": percentile_interval(
                np.array(
                    [per_seed[k]["compliance_rates"][-1] * 100.0 for k in per_seed]
                ),
                method="empirical_random_set_percentile",
            ).to_dict(),
        }

    if "unconstrained_random" in summary:
        unc = cast(dict[str, Any], summary["unconstrained_random"])
        per_seed = cast(dict[str, dict[str, Any]], unc["per_seed"])
        seed_slopes = [per_seed[k]["slope_per_alpha"] for k in per_seed]
        seed_rates_3_pct = [
            per_seed[k]["compliance_rates"][-1] * 100.0 for k in per_seed
        ]
        summary["comparison_to_h_neurons"] = {
            "alpha_3_h_rate_pct": round(h_rates[-1] * 100.0, 1),
            "alpha_3_random_mean_pct": round(float(np.mean(seed_rates_3_pct)), 1),
            "alpha_3_random_percentile_interval_pct": percentile_interval(
                np.array(seed_rates_3_pct, dtype=float),
                method="empirical_random_set_percentile",
            ).to_dict(),
            "slope_h_pp_per_alpha": round(h_slope, 2),
            "slope_random_mean_pp_per_alpha": round(float(np.mean(seed_slopes)), 2),
            "slope_random_percentile_interval": percentile_interval(
                np.array(seed_slopes, dtype=float),
                method="empirical_random_set_percentile",
            ).to_dict(),
        }

    return summary


BENCHMARK_TITLES = {
    "faitheval": "H-Neuron Specificity: Negative Control (FaithEval Anti-Compliance)",
    "falseqa": "H-Neuron Specificity: Negative Control (FalseQA)",
    "bioasq": "H-Neuron Specificity: Negative Control (BioASQ Factoid)",
}


def plot_comparison(
    summary: dict, alphas: list, output_path: str, benchmark: str = "faitheval"
):
    """Generate the comparison plot."""
    fig, ax = plt.subplots(figsize=(10, 6))

    alpha_arr = np.array(alphas)

    # H-neuron baseline
    h_rates = np.array(summary["h_neuron_baseline"]["compliance_rates"]) * 100
    h_neuron_count = summary["h_neuron_baseline"].get("n_h_neurons")
    h_label = (
        f"H-neurons ({h_neuron_count})" if h_neuron_count is not None else "H-neurons"
    )
    ax.plot(
        alpha_arr,
        h_rates,
        "o-",
        color="tab:blue",
        linewidth=2.5,
        markersize=8,
        label=h_label,
        zorder=10,
    )

    # Unconstrained random
    if "unconstrained_random" in summary:
        unc = summary["unconstrained_random"]
        for name, seed_data in sorted(unc["per_seed"].items()):
            rates = np.array(seed_data["compliance_rates"]) * 100
            ax.plot(alpha_arr, rates, "-", color="gray", alpha=0.35, linewidth=1)

        mean = np.array(unc["mean_compliance_rates"]) * 100
        std = np.array(unc["std_compliance_rates"]) * 100
        ax.plot(
            alpha_arr,
            mean,
            "--",
            color="gray",
            linewidth=2,
            label=f"Unconstrained random (mean ± 1σ, n={len(unc['per_seed'])})",
        )
        ax.fill_between(alpha_arr, mean - std, mean + std, color="gray", alpha=0.15)

    # Layer-matched random
    if "layer_matched_random" in summary:
        lm = summary["layer_matched_random"]
        for name, seed_data in sorted(lm["per_seed"].items()):
            rates = np.array(seed_data["compliance_rates"]) * 100
            ax.plot(alpha_arr, rates, "-", color="tab:orange", alpha=0.35, linewidth=1)

        mean = np.array(lm["mean_compliance_rates"]) * 100
        std = np.array(lm["std_compliance_rates"]) * 100
        ax.plot(
            alpha_arr,
            mean,
            "--",
            color="tab:orange",
            linewidth=2,
            label=f"Layer-matched random (mean ± 1σ, n={len(lm['per_seed'])})",
        )
        ax.fill_between(
            alpha_arr, mean - std, mean + std, color="tab:orange", alpha=0.15
        )

    ax.set_xlabel("Scaling Factor (α)", fontsize=12)
    ax.set_ylabel("Compliance Rate (%)", fontsize=12)
    ax.set_title(
        BENCHMARK_TITLES.get(benchmark, f"Negative Control ({benchmark})"), fontsize=13
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(alpha_arr)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot to {output_path}")


def build_negative_control_curve_rows(
    summary: dict[str, Any], alphas: list[float], benchmark: str
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for alpha, rate, ci, parse_failures in zip(
        alphas,
        summary["h_neuron_baseline"]["compliance_rates"],
        summary["h_neuron_baseline"]["compliance_ci_by_alpha"],
        summary["h_neuron_baseline"]["parse_failures"],
        strict=True,
    ):
        rows.append(
            {
                "benchmark": benchmark,
                "strategy": "h_neuron_baseline",
                "seed": "baseline",
                "alpha": float(alpha),
                "compliance_rate": rate,
                "n_compliant": ci.get("n_compliant"),
                "n_total": ci.get("n_total"),
                "parse_failures": parse_failures,
                "ci_lower": ci["ci"]["lower"] if ci and ci.get("ci") else None,
                "ci_upper": ci["ci"]["upper"] if ci and ci.get("ci") else None,
                "slope_per_alpha": summary["h_neuron_baseline"]["slope_per_alpha"],
                "spearman_rho": summary["h_neuron_baseline"]["spearman_rho"],
                "is_h_baseline": True,
            }
        )

    for strategy_key in ("unconstrained_random", "layer_matched_random"):
        if strategy_key not in summary:
            continue
        strategy_summary = cast(dict[str, Any], summary[strategy_key])
        for seed_name, seed_data in sorted(strategy_summary["per_seed"].items()):
            seed_data = cast(dict[str, Any], seed_data)
            for alpha, rate, ci, parse_failures in zip(
                alphas,
                seed_data["compliance_rates"],
                seed_data["compliance_ci_by_alpha"],
                seed_data["parse_failures"],
                strict=True,
            ):
                rows.append(
                    {
                        "benchmark": benchmark,
                        "strategy": strategy_key,
                        "seed": seed_name,
                        "alpha": float(alpha),
                        "compliance_rate": rate,
                        "n_compliant": ci.get("n_compliant"),
                        "n_total": ci.get("n_total"),
                        "parse_failures": parse_failures,
                        "ci_lower": ci["ci"]["lower"] if ci and ci.get("ci") else None,
                        "ci_upper": ci["ci"]["upper"] if ci and ci.get("ci") else None,
                        "slope_per_alpha": seed_data["slope_per_alpha"],
                        "spearman_rho": seed_data["spearman_rho"],
                        "is_h_baseline": False,
                    }
                )
    return rows


def build_negative_control_strategy_rows(
    summary: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = [
        {
            "strategy": "h_neuron_baseline",
            "mean_slope_per_alpha": summary["h_neuron_baseline"]["slope_per_alpha"],
            "mean_spearman_rho": summary["h_neuron_baseline"]["spearman_rho"],
            "any_seed_monotonic": True,
            "alpha_3_interval_lower": None,
            "alpha_3_interval_upper": None,
            "slope_interval_lower": None,
            "slope_interval_upper": None,
        }
    ]
    for strategy_key in ("unconstrained_random", "layer_matched_random"):
        if strategy_key not in summary:
            continue
        strategy_summary = cast(dict[str, Any], summary[strategy_key])
        alpha_3_interval = strategy_summary[
            "alpha_3_compliance_percentile_interval_pct"
        ]
        slope_interval = strategy_summary["slope_percentile_interval"]
        rows.append(
            {
                "strategy": strategy_key,
                "mean_slope_per_alpha": strategy_summary["mean_slope_per_alpha"],
                "mean_spearman_rho": strategy_summary["mean_spearman_rho"],
                "any_seed_monotonic": strategy_summary["any_seed_monotonic"],
                "alpha_3_interval_lower": alpha_3_interval["lower"],
                "alpha_3_interval_upper": alpha_3_interval["upper"],
                "slope_interval_lower": slope_interval["lower"],
                "slope_interval_upper": slope_interval["upper"],
            }
        )
    return rows


def build_negative_control_seed_ranking_rows(
    summary: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for strategy_key in ("unconstrained_random", "layer_matched_random"):
        if strategy_key not in summary:
            continue
        per_seed = cast(dict[str, dict[str, Any]], summary[strategy_key]["per_seed"])
        for seed_name, seed_data in sorted(per_seed.items()):
            rows.append(
                {
                    "strategy": strategy_key,
                    "seed": seed_name,
                    "slope_per_alpha": seed_data["slope_per_alpha"],
                    "spearman_rho": seed_data["spearman_rho"],
                    "alpha_3_rate_pct": round(
                        seed_data["compliance_rates"][-1] * 100, 2
                    ),
                    "monotonic": all(
                        seed_data["compliance_rates"][i]
                        <= seed_data["compliance_rates"][i + 1]
                        for i in range(len(seed_data["compliance_rates"]) - 1)
                    ),
                }
            )
    return sorted(
        rows,
        key=lambda row: (row["slope_per_alpha"], row["alpha_3_rate_pct"]),
        reverse=True,
    )


def negative_control_triage(summary: dict[str, Any]) -> tuple[str, str, str]:
    comparison = cast(dict[str, Any] | None, summary.get("comparison_to_h_neurons"))
    if comparison is None:
        return (
            "needs_judging",
            "Compliance metrics are unavailable until judged results exist.",
            "Run `evaluate_intervention.py`, then rerun `run_negative_control.py --analysis_only`.",
        )

    slope_h = comparison["slope_h_pp_per_alpha"]
    slope_interval = comparison["slope_random_percentile_interval"]
    if slope_interval["lower"] <= slope_h <= slope_interval["upper"]:
        return (
            "review_specificity",
            "H-neuron slope overlaps the random-set interval.",
            "Inspect seed ranking and rerun with more seeds or a second benchmark.",
        )

    return (
        "specificity_supported",
        "H-neuron slope separates from the random-set interval.",
        "Use the strongest benchmark and seed spread views to choose the next validation run.",
    )


def _wandb_table_from_rows(wandb_module, rows: list[dict[str, Any]]):
    if not rows:
        return None
    columns = list(rows[0].keys())
    data = [[row.get(column) for column in columns] for row in rows]
    return wandb_module.Table(columns=columns, data=data)


def parse_args():
    p = argparse.ArgumentParser(
        description="Negative control experiment for H-neuron specificity"
    )
    p.add_argument(
        "--benchmark",
        type=str,
        default="faitheval",
        choices=["faitheval", "falseqa", "bioasq", "jailbreak"],
        help="Benchmark to run negative control on",
    )
    p.add_argument(
        "--quick",
        action="store_true",
        help="Quick falsification: 3 unconstrained seeds, α ∈ {0.0, 1.0, 3.0}",
    )
    p.add_argument(
        "--model_key",
        type=str,
        default=os.environ.get("HNEURONS_MODEL_KEY"),
        help="Registered model key. Used for dimensions, defaults, and output roots.",
    )
    p.add_argument("--model_path", type=str, default=MODEL_PATH)
    p.add_argument("--classifier_path", type=str, default=CLASSIFIER_PATH)
    p.add_argument("--device_map", type=str, default="cuda:0")
    p.add_argument(
        "--prompt_style",
        type=str,
        default="anti_compliance",
        choices=["standard", "anti_compliance"],
        help=(
            "FaithEval prompt style: 'standard' (pro-context retrieval QA) "
            "or 'anti_compliance' (resist misleading context). Must match the "
            "H-neuron baseline's --prompt_style for the comparison to be valid."
        ),
    )
    p.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help=(
            "Truncate the benchmark to the first N samples after manifest "
            "filtering. Must match the H-neuron baseline's --max_samples for "
            "the comparison to share a sample population."
        ),
    )
    p.add_argument(
        "--sample_manifest",
        type=str,
        default=None,
        help=(
            "Path to JSON file with list of sample IDs to include. Filters "
            "loaded samples to this subset before any --max_samples truncation."
        ),
    )
    p.add_argument(
        "--output_base",
        type=str,
        default=None,
        help="Override benchmark control output directory.",
    )
    p.add_argument(
        "--h_neuron_baseline",
        type=str,
        default=None,
        help=(
            "Path to H-neuron baseline results JSON or experiment directory. "
            "Defaults to the registered model output root."
        ),
    )
    p.add_argument("--falseqa_path", type=str, default=FALSEQA_DATA_PATH)
    p.add_argument("--bioasq_path", type=str, default=BIOASQ_DATA_PATH)
    p.add_argument(
        "--analysis_only",
        action="store_true",
        help="Skip generation, only run analysis/plotting on existing data",
    )
    p.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases run tracking",
    )
    p.add_argument(
        "--jailbreak_batch_size",
        type=int,
        default=1,
        help=(
            "Batch size for jailbreak generation (default: 1; larger values "
            "change sampling semantics)"
        ),
    )
    args = p.parse_args()
    args.model_path = model_path_for(args.model_key, args.model_path)
    assert_causal_lm_supported(args.model_key, args.model_path)
    if args.classifier_path is None:
        args.classifier_path = default_classifier_path_for(
            args.model_key, args.model_path
        )
        if args.classifier_path is None:
            raise ValueError(
                "--classifier_path is required for negative controls with an "
                "unregistered model."
            )
    return args


def negative_control_schedule(
    benchmark: str, quick: bool
) -> tuple[list[float], list[ControlConfig]]:
    if benchmark == "jailbreak":
        return JAILBREAK_ALPHAS, [
            ("seed_0_unconstrained", "unconstrained", 0),
            ("seed_1_unconstrained", "unconstrained", 1),
            ("seed_2_unconstrained", "unconstrained", 2),
        ]
    if quick:
        return QUICK_ALPHAS, [
            ("seed_0_unconstrained", "unconstrained", 0),
            ("seed_1_unconstrained", "unconstrained", 1),
            ("seed_2_unconstrained", "unconstrained", 2),
            ("seed_0_layer_matched", "layer_matched", 0),
        ]
    return ALL_ALPHAS, [
        ("seed_0_unconstrained", "unconstrained", 0),
        ("seed_1_unconstrained", "unconstrained", 1),
        ("seed_2_unconstrained", "unconstrained", 2),
        ("seed_3_unconstrained", "unconstrained", 3),
        ("seed_4_unconstrained", "unconstrained", 4),
        ("seed_0_layer_matched", "layer_matched", 0),
        ("seed_1_layer_matched", "layer_matched", 1),
        ("seed_2_layer_matched", "layer_matched", 2),
    ]


def main():
    args = parse_args()
    benchmark = args.benchmark
    output_base = args.output_base or default_output_base(
        benchmark, args.model_key, args.model_path
    )
    baseline_path = args.h_neuron_baseline or default_h_neuron_baseline_dir(
        benchmark, args.model_key, args.model_path
    )

    alphas, configs = negative_control_schedule(benchmark, args.quick)

    os.makedirs(output_base, exist_ok=True)
    resumed_from_existing = any(
        any(
            os.path.exists(os.path.join(output_base, name, f"alpha_{alpha:.1f}.jsonl"))
            for alpha in alphas
        )
        for name, _, _ in configs
    )
    summary_path = os.path.join(output_base, "comparison_summary.json")
    plot_path = os.path.join(output_base, "negative_control_comparison.png")
    provenance_handle = start_run_provenance(
        args,
        primary_target=output_base,
        output_targets=[output_base, summary_path, plot_path],
        primary_target_is_dir=True,
    )
    provenance_status = "completed"
    provenance_extra = {}
    wb_run = None
    wandb_module = None
    try:
        # W&B tracking (opt-in)
        if args.wandb:
            try:
                import wandb as wandb_module
            except ImportError as exc:
                raise ImportError(
                    "--wandb requested but wandb is not installed. "
                    "Install project dependencies with `uv sync` or add it with `uv add wandb`."
                ) from exc
            wb_run, wandb_provenance = init_wandb_run(
                wandb_module,
                args,
                job_type="run_negative_control",
                group=f"negative_control:{benchmark}",
                tags=[
                    "negative_control",
                    benchmark,
                    "quick" if args.quick else "full",
                    "analysis_only" if args.analysis_only else "generate_and_analyze",
                ],
                config_extra={
                    "alphas": [float(a) for a in alphas],
                    "n_configs": len(configs),
                },
            )
            provenance_extra["wandb"] = wandb_provenance
            define_wandb_metrics(
                wandb_module,
                step_metric="alpha",
                metrics=[
                    "curve/h_neurons/compliance_rate_pct",
                    "curve/unconstrained_random/mean_compliance_rate_pct",
                    "curve/unconstrained_random/std_compliance_rate_pct",
                    "curve/unconstrained_random/mean_parse_failures",
                    "curve/layer_matched_random/mean_compliance_rate_pct",
                    "curve/layer_matched_random/std_compliance_rate_pct",
                    "curve/layer_matched_random/mean_parse_failures",
                ],
            )

        # Load classifier zero-weight indices
        print("Loading classifier for neuron sampling...")
        classifier = joblib.load(args.classifier_path)
        geometry = derive_classifier_geometry(
            classifier,
            model_path=args.model_path,
            model_key=args.model_key,
        )
        layer_distribution = derive_h_neuron_layer_distribution(
            classifier,
            intermediate_size=geometry.intermediate_size,
        )
        target_neuron_count = sum(layer_distribution.values())
        if target_neuron_count == 0:
            raise ValueError(
                "Classifier has no positive-weight H-neurons; cannot build controls."
            )
        zero_indices = zero_weight_indices_from_classifier(classifier)
        print(f"  {len(zero_indices)} zero-weight neurons available for sampling")
        print(
            "  classifier geometry: "
            f"{geometry.num_hidden_layers} layers x "
            f"{geometry.intermediate_size} FFN neurons "
            f"({geometry.source}); H-neurons={target_neuron_count}"
        )

        # Pre-compute all neuron selections
        neuron_configs = {}
        for name, strategy, seed in configs:
            if strategy == "unconstrained":
                flat = sample_unconstrained(zero_indices, seed, n=target_neuron_count)
            else:
                flat = sample_layer_matched(
                    zero_indices,
                    seed,
                    layer_distribution=layer_distribution,
                    intermediate_size=geometry.intermediate_size,
                )
            nmap = flat_to_neuron_map(flat, geometry.intermediate_size)
            neuron_configs[name] = nmap
            n_layers = len(nmap)
            print(
                f"  {name}: {sum(len(v) for v in nmap.values())} neurons across {n_layers} layers"
            )

        if not args.analysis_only:
            print(f"\nLoading model: {args.model_path}")
            model, tokenizer = load_model_and_tokenizer(
                args.model_path, args.device_map, model_key=args.model_key
            )

            print(f"Loading benchmark: {benchmark}")
            if benchmark == "faitheval":
                samples = load_faitheval()
            elif benchmark == "falseqa":
                samples = load_falseqa(args.falseqa_path)
            elif benchmark == "bioasq":
                samples = load_bioasq(args.bioasq_path)
            elif benchmark == "jailbreak":
                samples = load_jailbreak()

            if args.sample_manifest:
                manifest_ids = load_sample_manifest_ids(args.sample_manifest)
                samples = [s for s in samples if s["id"] in manifest_ids]
                print(
                    f"  Filtered to {len(samples)} samples via manifest "
                    f"{args.sample_manifest}"
                )
            if args.max_samples:
                samples = samples[: args.max_samples]
                print(f"  Truncated to {len(samples)} samples (--max_samples)")
            print(f"  {len(samples)} samples")

            all_results = {}
            for name, strategy, seed in configs:
                nmap = neuron_configs[name]
                out_dir = os.path.join(output_base, name)
                results = run_single_config(
                    model,
                    tokenizer,
                    samples,
                    nmap,
                    alphas,
                    out_dir,
                    name,
                    benchmark=benchmark,
                    jailbreak_batch_size=args.jailbreak_batch_size,
                    prompt_style=args.prompt_style,
                )
                all_results[name] = results

            del model, tokenizer
            torch.cuda.empty_cache()
        else:
            all_results = {}
            for name, strategy, seed in configs:
                results_path = os.path.join(output_base, name, "results.json")
                if os.path.exists(results_path):
                    with open(results_path) as f:
                        data = json.load(f)
                    all_results[name] = data["results"]
                else:
                    print(f"  WARNING: no results for {name}")

        if not all_results:
            print("No results to analyze.")
            return

        # Jailbreak analysis requires CSV-v2 scoring (separate pipeline)
        if benchmark == "jailbreak":
            csv2_eval_dir = _jailbreak_csv2_eval_dir(
                user_baseline=args.h_neuron_baseline,
                model_key=args.model_key,
                model_path=args.model_path,
            )
            print("\n" + "=" * 60)
            print("Generation complete. Next steps:")
            print("  1. Run CSV-v2 scoring on each seed directory:")
            for name, _, _ in configs:
                seed_dir = os.path.join(output_base, name)
                alpha_str = " ".join(f"{a:.1f}" for a in alphas)
                print(
                    f"     uv run python scripts/evaluate_csv2.py "
                    f"--input_dir {seed_dir} --output_dir {seed_dir} "
                    f"--alphas {alpha_str}"
                )
            print("  2. Run comparative analysis:")
            print(
                f"     uv run python scripts/analyze_csv2_control.py "
                f"--control_base {output_base} "
                f"--experiment_dir {csv2_eval_dir} "
                f"--alphas {' '.join(f'{a:.1f}' for a in alphas)}"
            )
            provenance_extra["output_targets"] = [
                output_base,
                *[os.path.join(output_base, name) for name, _, _ in configs],
            ]
            finish_run_provenance(
                provenance_handle, provenance_status, provenance_extra
            )
            return

        # FalseQA analysis requires GPT-4o judging first
        if benchmark == "falseqa":
            has_compliance = any(
                "compliance_rate" in v.get(str(float(alphas[0])), {})
                for v in all_results.values()
            )
            if not has_compliance:
                print("\n" + "=" * 60)
                print("Generation complete. Run GPT-4o judging before analysis:")
                for name, _, _ in configs:
                    alpha_str = " ".join(f"{a:.1f}" for a in alphas)
                    print(
                        f"  uv run python scripts/evaluate_intervention.py "
                        f"--benchmark falseqa "
                        f"--input_dir {output_base}/{name} "
                        f"--alphas {alpha_str}"
                    )
                print(
                    f"\nThen re-run with --analysis_only:\n"
                    f"  uv run python scripts/run_negative_control.py "
                    f"--benchmark falseqa {'--quick' if args.quick else ''} --analysis_only"
                )
                return

        resolved_baseline_path = resolve_results_json(baseline_path)
        if not os.path.exists(resolved_baseline_path):
            print(f"\nH-neuron baseline not found at {resolved_baseline_path}")
            print(
                "Run the H-neuron intervention first, then re-run with --analysis_only"
            )
            return

        print("\n" + "=" * 60)
        print("Building comparison summary...")
        print("=" * 60)
        summary = build_comparison_summary(
            all_results,
            alphas,
            benchmark=benchmark,
            baseline_path=resolved_baseline_path,
        )
        triage_status, triage_note, recommended_next_action = negative_control_triage(
            summary
        )
        summary["triage"] = {
            "status": triage_status,
            "note": triage_note,
            "recommended_next_action": recommended_next_action,
        }

        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Saved summary to {summary_path}")

        print("\n--- KEY FINDINGS ---")
        h = summary["h_neuron_baseline"]
        print(f"H-neurons: slope={h['slope_per_alpha']}%/α, ρ={h['spearman_rho']}")

        for key in ["unconstrained_random", "layer_matched_random"]:
            if key in summary:
                d = summary[key]
                print(
                    f"{key}: mean_slope={d['mean_slope_per_alpha']}%/α, "
                    f"mean_ρ={d['mean_spearman_rho']}, "
                    f"any_monotonic={d['any_seed_monotonic']}"
                )

        if "comparison_to_h_neurons" in summary:
            st = summary["comparison_to_h_neurons"]
            print(
                "α=3 compliance: "
                f"H={st['alpha_3_h_rate_pct']}%, "
                f"random mean={st['alpha_3_random_mean_pct']}%, "
                f"random 95% interval=[{st['alpha_3_random_percentile_interval_pct']['lower']:.2f}, "
                f"{st['alpha_3_random_percentile_interval_pct']['upper']:.2f}]"
            )
            print(
                "Slope: "
                f"H={st['slope_h_pp_per_alpha']}pp/α, "
                f"random mean={st['slope_random_mean_pp_per_alpha']}pp/α, "
                f"random 95% interval=[{st['slope_random_percentile_interval']['lower']:.2f}, "
                f"{st['slope_random_percentile_interval']['upper']:.2f}]"
            )

        plot_comparison(summary, alphas, plot_path, benchmark=benchmark)

        provenance_extra["output_targets"] = [
            output_base,
            summary_path,
            plot_path,
            *[os.path.join(output_base, name) for name, _, _ in configs],
        ]
        finish_run_provenance(provenance_handle, provenance_status, provenance_extra)

        if wb_run is not None and wandb_module is not None:
            curve_rows = build_negative_control_curve_rows(summary, alphas, benchmark)
            strategy_rows = build_negative_control_strategy_rows(summary)
            seed_rows = build_negative_control_seed_ranking_rows(summary)

            for alpha in alphas:
                log_payload: dict[str, Any] = {"alpha": float(alpha)}
                baseline_row = summary["h_neuron_baseline"]
                baseline_idx = alphas.index(alpha)
                log_payload["curve/h_neurons/compliance_rate_pct"] = round(
                    baseline_row["compliance_rates"][baseline_idx] * 100, 2
                )

                for strategy_key in ("unconstrained_random", "layer_matched_random"):
                    if strategy_key not in summary:
                        continue
                    strategy_summary = cast(dict[str, Any], summary[strategy_key])
                    series_prefix = f"curve/{strategy_key}"
                    log_payload[f"{series_prefix}/mean_compliance_rate_pct"] = round(
                        strategy_summary["mean_compliance_rates"][baseline_idx] * 100, 2
                    )
                    log_payload[f"{series_prefix}/std_compliance_rate_pct"] = round(
                        strategy_summary["std_compliance_rates"][baseline_idx] * 100, 2
                    )
                    log_payload[f"{series_prefix}/mean_parse_failures"] = (
                        strategy_summary["mean_parse_failures"][baseline_idx]
                    )

                wandb_module.log(log_payload)

            wandb_module.log(
                {
                    "tables/curve_rows": _wandb_table_from_rows(
                        wandb_module, curve_rows
                    ),
                    "tables/strategy_summary": _wandb_table_from_rows(
                        wandb_module, strategy_rows
                    ),
                    "tables/seed_ranking": _wandb_table_from_rows(
                        wandb_module, seed_rows
                    ),
                }
            )

            comparison = cast(
                dict[str, Any] | None, summary.get("comparison_to_h_neurons")
            )
            wb_run.summary["run_health/status"] = "completed"
            wb_run.summary["run_health/resumed"] = resumed_from_existing or bool(
                args.analysis_only
            )
            wb_run.summary["run_health/output_path_count"] = 3 + len(configs)
            wb_run.summary["run_health/triage_status"] = triage_status
            wb_run.summary["run_health/triage_note"] = triage_note
            wb_run.summary["recommended_next_action"] = recommended_next_action
            wb_run.summary["science/h_slope_pp_per_alpha"] = summary[
                "h_neuron_baseline"
            ]["slope_per_alpha"]
            if comparison is not None:
                wb_run.summary["science/random_mean_slope_pp_per_alpha"] = comparison[
                    "slope_random_mean_pp_per_alpha"
                ]
                wb_run.summary["science/h_vs_random_alpha_3_gap_pct"] = round(
                    comparison["alpha_3_h_rate_pct"]
                    - comparison["alpha_3_random_mean_pct"],
                    2,
                )
                wb_run.summary["science/random_slope_interval_lower"] = comparison[
                    "slope_random_percentile_interval"
                ]["lower"]
                wb_run.summary["science/random_slope_interval_upper"] = comparison[
                    "slope_random_percentile_interval"
                ]["upper"]
            log_wandb_files_as_artifact(
                wb_run,
                wandb_module,
                name=f"negative-control-{benchmark}",
                artifact_type="negative_control_summary",
                paths=[summary_path, plot_path, provenance_handle["path"]]
                if provenance_handle is not None
                else [summary_path, plot_path],
            )
    except BaseException as exc:
        provenance_status = provenance_status_for_exception(exc)
        provenance_extra["error"] = provenance_error_message(exc)
        raise
    finally:
        if wb_run is not None and wandb_module is not None:
            wandb_module.finish()
        finish_run_provenance(provenance_handle, provenance_status, provenance_extra)


if __name__ == "__main__":
    main()
