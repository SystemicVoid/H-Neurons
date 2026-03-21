"""Verbosity confound test for the 38 H-Neurons.

Runs 400 forward passes (100 items x 4 conditions) through a 2x2 factorial
design (short/long x true/false) and measures whether H-Neuron CETT
activations encode truthfulness or response length.

Usage:
    uv run python scripts/run_verbosity_confound.py \
        --model_path google/gemma-3-4b-it \
        --classifier_path models/gemma3_4b_classifier_disjoint.pkl \
        --data_path data/gemma3_4b/intervention/verbosity_confound/verbosity_test_data.json \
        --output_dir data/gemma3_4b/intervention/verbosity_confound
"""

import argparse
import json
import os

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from scipy.stats import wilcoxon
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from extract_activations import CETTManager, unwrap_chat_template_output

CONDITIONS = ["short_true", "long_true", "short_false", "long_false"]

DEFAULT_MODEL_PATH = os.environ.get("HNEURONS_MODEL_PATH", "google/gemma-3-4b-it")
DEFAULT_CLASSIFIER_PATH = os.environ.get(
    "HNEURONS_CLASSIFIER_PATH", "models/gemma3_4b_classifier_disjoint.pkl"
)
DEFAULT_DEVICE_MAP = os.environ.get("HNEURONS_DEVICE_MAP", "cuda:0")


def parse_args():
    p = argparse.ArgumentParser(description="Verbosity confound test for H-Neurons.")
    p.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH)
    p.add_argument("--classifier_path", type=str, default=DEFAULT_CLASSIFIER_PATH)
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--device_map", type=str, default=DEFAULT_DEVICE_MAP)
    return p.parse_args()


def get_response_start(input_ids, tokenizer, question: str) -> int:
    """Find where the assistant response begins in the full sequence."""
    user_only = unwrap_chat_template_output(
        tokenizer.apply_chat_template(
            [{"role": "user", "content": question}],
            return_tensors="pt",
            add_generation_prompt=True,
        )
    )
    return user_only.shape[1]


def get_response_end(tokenizer, question: str, response_text: str) -> int:
    """Find where the assistant response content ends (excluding trailer tokens).

    Chat templates like Gemma's append trailer tokens (e.g. ``<end_of_turn>\\n``)
    after the assistant text.  We compute the end boundary by tokenizing the
    conversation *with* the generation prompt (user-only) and then encoding just
    the response text, giving us the exact span of content tokens.
    """
    user_only = unwrap_chat_template_output(
        tokenizer.apply_chat_template(
            [{"role": "user", "content": question}],
            return_tensors="pt",
            add_generation_prompt=True,
        )
    )
    resp_start = user_only.shape[1]
    resp_ids = tokenizer.encode(response_text, add_special_tokens=False)
    return resp_start + len(resp_ids)


def extract_condition_activations(
    model,
    tokenizer,
    cett_manager: CETTManager,
    question: str,
    response_text: str,
    inter_size: int,
):
    """Run a single forward pass and return aggregated CETT features."""
    cett_manager.clear()
    msgs = [
        {"role": "user", "content": question},
        {"role": "assistant", "content": response_text},
    ]
    input_ids = unwrap_chat_template_output(
        tokenizer.apply_chat_template(
            msgs, return_tensors="pt", add_generation_prompt=False
        )
    ).to(model.device)

    with torch.no_grad():
        model(input_ids)

    cett = cett_manager.get_cett_tensor(use_abs=True, use_mag=True)
    # cett shape: [layers, tokens, neurons]

    resp_start = get_response_start(input_ids, tokenizer, question)
    resp_end = get_response_end(tokenizer, question, response_text)
    resp_cett = cett[:, resp_start:resp_end, :]  # [layers, resp_tokens, neurons]

    mean_agg = resp_cett.mean(dim=1)  # [layers, neurons]
    max_agg = resp_cett.max(dim=1).values  # [layers, neurons]

    return (
        mean_agg.cpu().float().numpy().flatten(),
        max_agg.cpu().float().numpy().flatten(),
        int(resp_end - resp_start),
    )


def bootstrap_mean_ci(values, n_resamples=10_000, seed=42, confidence=0.95):
    rng = np.random.default_rng(seed)
    means = np.array(
        [
            rng.choice(values, size=len(values), replace=True).mean()
            for _ in range(n_resamples)
        ]
    )
    alpha = 1.0 - confidence
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return {
        "estimate": float(np.mean(values)),
        "ci_lower": float(lo),
        "ci_upper": float(hi),
    }


def cohens_d_paired(x, y):
    diff = x - y
    return float(np.mean(diff) / (np.std(diff, ddof=1) + 1e-12))


def analyse_effects(activations, neuron_labels, agg_label="mean"):
    """Compute 2x2 factorial effects across the 38 H-Neurons.

    activations: dict[condition_name -> np.array of shape [n_items, n_neurons]]
    """
    st = activations["short_true"]
    lt = activations["long_true"]
    sf = activations["short_false"]
    lf = activations["long_false"]

    n_items, n_neurons = st.shape

    # Aggregated (mean across neurons, weighted by nothing -- raw CETT)
    agg = {c: v.mean(axis=1) for c, v in activations.items()}

    agg_summary = {}
    for c in CONDITIONS:
        agg_summary[c] = bootstrap_mean_ci(agg[c])

    # Main effects (paired, across items)
    truth_diff = (agg["short_false"] + agg["long_false"]) / 2 - (
        agg["short_true"] + agg["long_true"]
    ) / 2
    length_diff = (agg["long_true"] + agg["long_false"]) / 2 - (
        agg["short_true"] + agg["short_false"]
    ) / 2

    truth_d = cohens_d_paired(
        (agg["short_false"] + agg["long_false"]) / 2,
        (agg["short_true"] + agg["long_true"]) / 2,
    )
    length_d = cohens_d_paired(
        (agg["long_true"] + agg["long_false"]) / 2,
        (agg["short_true"] + agg["short_false"]) / 2,
    )

    truth_p = float(wilcoxon(truth_diff, alternative="two-sided").pvalue)
    length_p = float(wilcoxon(length_diff, alternative="two-sided").pvalue)

    # Per-neuron breakdown
    per_neuron = []
    for j in range(n_neurons):
        false_vals = (sf[:, j] + lf[:, j]) / 2
        true_vals = (st[:, j] + lt[:, j]) / 2
        long_vals = (lt[:, j] + lf[:, j]) / 2
        short_vals = (st[:, j] + sf[:, j]) / 2

        t_diff = false_vals - true_vals
        l_diff = long_vals - short_vals

        t_d = cohens_d_paired(false_vals, true_vals)
        l_d = cohens_d_paired(long_vals, short_vals)

        t_p = (
            float(wilcoxon(t_diff, alternative="two-sided").pvalue)
            if np.any(t_diff != 0)
            else 1.0
        )
        l_p = (
            float(wilcoxon(l_diff, alternative="two-sided").pvalue)
            if np.any(l_diff != 0)
            else 1.0
        )

        per_neuron.append(
            {
                "label": neuron_labels[j],
                "truth_effect_d": round(t_d, 4),
                "truth_effect_p": round(t_p, 6),
                "length_effect_d": round(l_d, 4),
                "length_effect_p": round(l_p, 6),
                "mean_short_true": round(float(st[:, j].mean()), 6),
                "mean_long_true": round(float(lt[:, j].mean()), 6),
                "mean_short_false": round(float(sf[:, j].mean()), 6),
                "mean_long_false": round(float(lf[:, j].mean()), 6),
            }
        )

    # Verdict
    abs_truth = abs(truth_d)
    abs_length = abs(length_d)
    if abs_truth < 0.2 and abs_length < 0.2:
        verdict = "C_entangled_or_null"
        verdict_detail = "Neither effect reaches small-effect threshold (|d| < 0.2)."
    elif abs_length > 2 * abs_truth and abs_length >= 0.2:
        verdict = "A_verbosity"
        verdict_detail = (
            f"Length effect ({abs_length:.2f}) dominates truth effect "
            f"({abs_truth:.2f}) by >2:1 ratio."
        )
    elif abs_truth > 2 * abs_length and abs_truth >= 0.2:
        verdict = "B_semantic_truth"
        verdict_detail = (
            f"Truth effect ({abs_truth:.2f}) dominates length effect "
            f"({abs_length:.2f}) by >2:1 ratio."
        )
    else:
        verdict = "C_entangled"
        verdict_detail = (
            f"Both effects present (truth d={abs_truth:.2f}, length d={abs_length:.2f}) "
            "without clear dominance."
        )

    return {
        "aggregation": agg_label,
        "condition_means": agg_summary,
        "truth_effect": {
            "cohens_d": round(truth_d, 4),
            "p_value": round(truth_p, 6),
            "mean_diff": round(float(truth_diff.mean()), 6),
            "ci": bootstrap_mean_ci(truth_diff),
        },
        "length_effect": {
            "cohens_d": round(length_d, 4),
            "p_value": round(length_p, 6),
            "mean_diff": round(float(length_diff.mean()), 6),
            "ci": bootstrap_mean_ci(length_diff),
        },
        "verdict": verdict,
        "verdict_detail": verdict_detail,
        "per_neuron": per_neuron,
    }


def make_plots(mean_analysis, max_analysis, output_dir):
    matplotlib.use("Agg")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel (a): Classifier-level scores by condition
    ax = axes[0]
    cond_labels = ["short\ntrue", "long\ntrue", "short\nfalse", "long\nfalse"]
    means_vals = [mean_analysis["condition_means"][c]["estimate"] for c in CONDITIONS]
    ci_lo = [mean_analysis["condition_means"][c]["ci_lower"] for c in CONDITIONS]
    ci_hi = [mean_analysis["condition_means"][c]["ci_upper"] for c in CONDITIONS]
    colors = ["#2ecc71", "#27ae60", "#e74c3c", "#c0392b"]
    errs = [
        [m - lo for m, lo in zip(means_vals, ci_lo)],
        [hi - m for m, hi in zip(means_vals, ci_hi)],
    ]
    ax.bar(
        range(4),
        means_vals,
        yerr=errs,
        color=colors,
        capsize=5,
        edgecolor="black",
        linewidth=0.5,
    )
    ax.set_xticks(range(4))
    ax.set_xticklabels(cond_labels)
    ax.set_ylabel("Mean H-Neuron CETT (mean agg)")
    ax.set_title("(a) Aggregated H-Neuron Activation")

    # Panel (b): Heatmap of 38 neurons x 4 conditions (mean agg)
    ax = axes[1]
    n_neurons = len(mean_analysis["per_neuron"])
    heatmap_data = np.zeros((n_neurons, 4))
    labels = []
    for j, entry in enumerate(mean_analysis["per_neuron"]):
        heatmap_data[j, 0] = entry["mean_short_true"]
        heatmap_data[j, 1] = entry["mean_long_true"]
        heatmap_data[j, 2] = entry["mean_short_false"]
        heatmap_data[j, 3] = entry["mean_long_false"]
        labels.append(entry["label"])
    sns.heatmap(
        heatmap_data,
        ax=ax,
        xticklabels=["S+T", "L+T", "S+F", "L+F"],
        yticklabels=labels,
        cmap="YlOrRd",
        cbar_kws={"label": "CETT"},
    )
    ax.set_title("(b) Per-Neuron Activation Heatmap")
    ax.set_ylabel("H-Neuron")
    ax.tick_params(axis="y", labelsize=6)

    # Panel (c): Truth effect vs Length effect scatter (quadrant plot)
    ax = axes[2]
    truth_ds = [n["truth_effect_d"] for n in mean_analysis["per_neuron"]]
    length_ds = [n["length_effect_d"] for n in mean_analysis["per_neuron"]]
    ax.scatter(truth_ds, length_ds, s=40, alpha=0.7, edgecolors="black", linewidth=0.5)
    for j, lbl in enumerate(labels):
        ax.annotate(lbl, (truth_ds[j], length_ds[j]), fontsize=5, alpha=0.7)
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.axvline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.set_xlabel("Truth Effect (Cohen's d)")
    ax.set_ylabel("Length Effect (Cohen's d)")
    ax.set_title("(c) Per-Neuron Effect Quadrant")

    fig.suptitle(
        f"Verdict (mean agg): {mean_analysis['verdict']} | "
        f"Verdict (max agg): {max_analysis['verdict']}",
        fontsize=11,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    out_path = os.path.join(output_dir, "confound_analysis.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot to {out_path}")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.data_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    print(f"Loaded {len(dataset)} items with 4 conditions each.")

    classifier = joblib.load(args.classifier_path)
    weights = classifier.coef_[0]
    inter_size = 10240
    pos_indices = np.where(weights > 0)[0]
    n_neurons = len(pos_indices)
    neuron_labels = []
    for idx in pos_indices:
        layer = int(idx // inter_size)
        neuron = int(idx % inter_size)
        neuron_labels.append(f"L{layer}:N{neuron}")
    print(f"Classifier has {n_neurons} H-Neurons: {neuron_labels[:5]}...")

    print(f"Loading model {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, device_map=args.device_map
    )
    cett_manager = CETTManager(model)

    # Collect activations: [n_items, 4 conditions, n_neurons] for mean and max
    mean_acts = {c: [] for c in CONDITIONS}
    max_acts = {c: [] for c in CONDITIONS}
    resp_token_counts = {c: [] for c in CONDITIONS}

    for item in tqdm(dataset, desc="Items"):
        question = item["question"]
        for cond in CONDITIONS:
            response_text = item[cond]
            mean_flat, max_flat, n_resp_tokens = extract_condition_activations(
                model, tokenizer, cett_manager, question, response_text, inter_size
            )

            mean_h = mean_flat[pos_indices]
            max_h = max_flat[pos_indices]

            mean_acts[cond].append(mean_h)
            max_acts[cond].append(max_h)
            resp_token_counts[cond].append(n_resp_tokens)

    mean_arrays = {c: np.stack(v) for c, v in mean_acts.items()}
    max_arrays = {c: np.stack(v) for c, v in max_acts.items()}

    # Save raw results
    raw_results = {
        "n_items": len(dataset),
        "n_neurons": n_neurons,
        "neuron_labels": neuron_labels,
        "response_token_counts": resp_token_counts,
        "mean_activations": {c: v.tolist() for c, v in mean_arrays.items()},
        "max_activations": {c: v.tolist() for c, v in max_arrays.items()},
    }
    results_path = os.path.join(args.output_dir, "confound_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(raw_results, f, indent=2)
    print(f"Saved raw results to {results_path}")

    # Analyse
    print("\n=== Mean Aggregation Analysis ===")
    mean_analysis = analyse_effects(mean_arrays, neuron_labels, agg_label="mean")
    print(
        f"Truth effect: d={mean_analysis['truth_effect']['cohens_d']}, "
        f"p={mean_analysis['truth_effect']['p_value']}"
    )
    print(
        f"Length effect: d={mean_analysis['length_effect']['cohens_d']}, "
        f"p={mean_analysis['length_effect']['p_value']}"
    )
    print(f"Verdict: {mean_analysis['verdict']} -- {mean_analysis['verdict_detail']}")

    print("\n=== Max Aggregation Analysis ===")
    max_analysis = analyse_effects(max_arrays, neuron_labels, agg_label="max")
    print(
        f"Truth effect: d={max_analysis['truth_effect']['cohens_d']}, "
        f"p={max_analysis['truth_effect']['p_value']}"
    )
    print(
        f"Length effect: d={max_analysis['length_effect']['cohens_d']}, "
        f"p={max_analysis['length_effect']['p_value']}"
    )
    print(f"Verdict: {max_analysis['verdict']} -- {max_analysis['verdict_detail']}")

    # Full summary
    summary = {
        "mean_aggregation": mean_analysis,
        "max_aggregation": max_analysis,
        "response_token_count_means": {
            c: round(float(np.mean(v)), 1) for c, v in resp_token_counts.items()
        },
    }
    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary to {summary_path}")

    make_plots(mean_analysis, max_analysis, args.output_dir)


if __name__ == "__main__":
    main()
