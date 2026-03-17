"""
Negative control experiment for H-Neuron specificity.

Tests whether scaling arbitrary sets of 38 neurons produces comparable
compliance effects to scaling the identified H-neurons. Supports FaithEval
(anti-compliance prompt, inline MC evaluation) and FalseQA (bare-question
prompt, deferred GPT-4o judging via evaluate_intervention.py).

Usage:
    # FaithEval (default)
    uv run python scripts/run_negative_control.py --quick
    uv run python scripts/run_negative_control.py

    # FalseQA
    uv run python scripts/run_negative_control.py --benchmark falseqa --quick
    uv run python scripts/run_negative_control.py --benchmark falseqa
"""

import argparse
import json
import os
import sys

import joblib
import matplotlib
import numpy as np
import torch
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from run_intervention import (
    HNeuronScaler,
    _faitheval_prompt,
    extract_mc_answer,
    generate_response,
    load_existing_ids,
    load_faitheval,
    load_falseqa,
    load_model_and_tokenizer,
)

# H-neuron layer distribution: {layer: count}
H_NEURON_LAYER_DIST = {
    0: 2,
    2: 1,
    4: 3,
    5: 4,
    6: 2,
    7: 3,
    9: 1,
    10: 2,
    12: 1,
    13: 2,
    14: 2,
    15: 2,
    16: 2,
    20: 1,
    23: 1,
    24: 1,
    25: 1,
    26: 1,
    27: 1,
    28: 1,
    30: 1,
    31: 2,
    33: 1,
}

INTER_SIZE = 10240
N_LAYERS = 34
ALL_ALPHAS = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
QUICK_ALPHAS = [0.0, 1.0, 3.0]

MODEL_PATH = "google/gemma-3-4b-it"
CLASSIFIER_PATH = "models/gemma3_4b_classifier.pkl"
OUTPUT_BASES = {
    "faitheval": "data/gemma3_4b/intervention/negative_control",
    "falseqa": "data/gemma3_4b/intervention/negative_control_falseqa",
}
H_NEURON_BASELINES = {
    "faitheval": "data/gemma3_4b/intervention/faitheval/results.json",
    "falseqa": "data/gemma3_4b/intervention/falseqa/results.json",
}
FALSEQA_DATA_PATH = "data/benchmarks/falseqa_test.csv"


def get_zero_weight_indices(classifier_path: str) -> np.ndarray:
    """Get flat indices of all neurons with zero classifier weight."""
    clf = joblib.load(classifier_path)
    w = clf.coef_[0]
    return np.where(w == 0)[0]


def get_nonzero_flat_indices(classifier_path: str) -> set:
    """Get flat indices of all neurons with non-zero classifier weight."""
    clf = joblib.load(classifier_path)
    w = clf.coef_[0]
    return set(np.where(w != 0)[0].tolist())


def flat_to_neuron_map(flat_indices: np.ndarray) -> dict:
    """Convert flat classifier indices to {layer_idx: [neuron_indices]}."""
    neuron_map = {}
    for idx in flat_indices:
        layer = int(idx // INTER_SIZE)
        neuron = int(idx % INTER_SIZE)
        neuron_map.setdefault(layer, []).append(neuron)
    return neuron_map


def sample_unconstrained(
    zero_indices: np.ndarray, seed: int, n: int = 38
) -> np.ndarray:
    """Sample n neurons uniformly at random from zero-weight positions."""
    rng = np.random.RandomState(seed)
    chosen = rng.choice(zero_indices, size=n, replace=False)
    return np.sort(chosen)


def sample_layer_matched(zero_indices: np.ndarray, seed: int) -> np.ndarray:
    """Sample random neurons matching the H-neuron layer distribution."""
    rng = np.random.RandomState(seed)
    zero_set = set(zero_indices.tolist())

    chosen = []
    for layer, count in sorted(H_NEURON_LAYER_DIST.items()):
        # All neuron indices in this layer (flat)
        layer_start = layer * INTER_SIZE
        layer_end = layer_start + INTER_SIZE
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
            model, tokenizer, samples, scaler, alphas, output_dir, config_name
        )
    elif benchmark == "falseqa":
        results = _run_falseqa_alphas(
            model, tokenizer, samples, scaler, alphas, output_dir, config_name
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
    model, tokenizer, samples, scaler, alphas, output_dir, config_name
):
    """FaithEval anti-compliance: inline MC evaluation."""
    from tqdm import tqdm

    results = {}
    for alpha in alphas:
        out_path = os.path.join(output_dir, f"alpha_{alpha:.1f}.jsonl")
        existing_ids = load_existing_ids(out_path)
        scaler.alpha = alpha
        total = 0

        with open(out_path, "a") as f:
            for sample in tqdm(samples, desc=f"[{config_name}] α={alpha:.1f}"):
                if sample["id"] in existing_ids:
                    total += 1
                    continue

                prompt = _faitheval_prompt(sample, "anti_compliance")
                messages = [{"role": "user", "content": prompt}]
                response = generate_response(
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
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        comp_total, n_total, pf_total = _count_compliance_and_pf(out_path)
        results[str(alpha)] = {
            "compliance_rate": round(comp_total / n_total, 4) if n_total else 0,
            "n_compliant": comp_total,
            "n_total": n_total,
            "parse_failures": pf_total,
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

        with open(out_path, "a") as f:
            for sample in tqdm(samples, desc=f"[{config_name}] α={alpha:.1f}"):
                if sample["id"] in existing_ids:
                    total += 1
                    continue

                messages = [{"role": "user", "content": sample["question"]}]
                response = generate_response(
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
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        n_total = _count_lines(out_path)
        results[str(alpha)] = {"n_total": n_total}
        print(f"  α={alpha:.1f}: {n_total} responses generated (judging deferred)")
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
    all_results: dict, alphas: list, benchmark: str = "faitheval"
) -> dict:
    """Build the comparison summary JSON."""
    baseline_path = H_NEURON_BASELINES[benchmark]
    with open(baseline_path) as f:
        h_baseline_raw = json.load(f)

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
            "compliance_rates": h_rates,
            "slope_per_alpha": round(h_slope, 2),
            "spearman_rho": round(h_rho, 4),
            "parse_failures": [0] * len(alphas),
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
            pfs = [res[str(float(a))]["parse_failures"] for a in alphas]
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
            "mean_spearman_rho": round(np.mean(rhos), 4),
            "any_seed_monotonic": monotonic_count > 0,
            "mean_parse_failures": [round(p, 1) for p in mean_pf],
        }

    # Statistical tests (unconstrained vs H-neurons)
    if "unconstrained_random" in summary:
        unc = summary["unconstrained_random"]
        # t-test on compliance at alpha=3.0 (per-seed rates vs H-neuron rate)
        seed_rates_3 = [
            unc["per_seed"][k]["compliance_rates"][-1] for k in unc["per_seed"]
        ]
        h_rate_3 = h_rates[-1]
        if len(seed_rates_3) > 1:
            _, t_p = stats.ttest_1samp(seed_rates_3, h_rate_3)
        else:
            t_p = float("nan")

        # Slope comparison
        seed_slopes = [unc["per_seed"][k]["slope_per_alpha"] for k in unc["per_seed"]]
        if len(seed_slopes) > 1:
            _, s_p = stats.ttest_1samp(seed_slopes, h_slope)
        else:
            s_p = float("nan")

        summary["statistical_tests"] = {
            "compliance_at_alpha_3_h_vs_random_ttest_p": round(t_p, 6),
            "slope_h_vs_random_ttest_p": round(s_p, 6),
        }

    return summary


BENCHMARK_TITLES = {
    "faitheval": "H-Neuron Specificity: Negative Control (FaithEval Anti-Compliance)",
    "falseqa": "H-Neuron Specificity: Negative Control (FalseQA)",
}


def plot_comparison(
    summary: dict, alphas: list, output_path: str, benchmark: str = "faitheval"
):
    """Generate the comparison plot."""
    fig, ax = plt.subplots(figsize=(10, 6))

    alpha_arr = np.array(alphas)

    # H-neuron baseline
    h_rates = np.array(summary["h_neuron_baseline"]["compliance_rates"]) * 100
    ax.plot(
        alpha_arr,
        h_rates,
        "o-",
        color="tab:blue",
        linewidth=2.5,
        markersize=8,
        label="H-neurons (38)",
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


def parse_args():
    p = argparse.ArgumentParser(
        description="Negative control experiment for H-neuron specificity"
    )
    p.add_argument(
        "--benchmark",
        type=str,
        default="faitheval",
        choices=["faitheval", "falseqa"],
        help="Benchmark to run negative control on",
    )
    p.add_argument(
        "--quick",
        action="store_true",
        help="Quick falsification: 3 unconstrained seeds, α ∈ {0.0, 1.0, 3.0}",
    )
    p.add_argument("--model_path", type=str, default=MODEL_PATH)
    p.add_argument("--classifier_path", type=str, default=CLASSIFIER_PATH)
    p.add_argument("--device_map", type=str, default="cuda:0")
    p.add_argument("--falseqa_path", type=str, default=FALSEQA_DATA_PATH)
    p.add_argument(
        "--analysis_only",
        action="store_true",
        help="Skip generation, only run analysis/plotting on existing data",
    )
    return p.parse_args()


def main():
    args = parse_args()
    benchmark = args.benchmark
    alphas = QUICK_ALPHAS if args.quick else ALL_ALPHAS
    output_base = OUTPUT_BASES[benchmark]

    if args.quick:
        configs = [
            ("seed_0_unconstrained", "unconstrained", 0),
            ("seed_1_unconstrained", "unconstrained", 1),
            ("seed_2_unconstrained", "unconstrained", 2),
        ]
    else:
        configs = [
            ("seed_0_unconstrained", "unconstrained", 0),
            ("seed_1_unconstrained", "unconstrained", 1),
            ("seed_2_unconstrained", "unconstrained", 2),
            ("seed_3_unconstrained", "unconstrained", 3),
            ("seed_4_unconstrained", "unconstrained", 4),
            ("seed_0_layer_matched", "layer_matched", 0),
            ("seed_1_layer_matched", "layer_matched", 1),
            ("seed_2_layer_matched", "layer_matched", 2),
        ]

    os.makedirs(output_base, exist_ok=True)

    # Load classifier zero-weight indices
    print("Loading classifier for neuron sampling...")
    zero_indices = get_zero_weight_indices(args.classifier_path)
    print(f"  {len(zero_indices)} zero-weight neurons available for sampling")

    # Pre-compute all neuron selections
    neuron_configs = {}
    for name, strategy, seed in configs:
        if strategy == "unconstrained":
            flat = sample_unconstrained(zero_indices, seed)
        else:
            flat = sample_layer_matched(zero_indices, seed)
        nmap = flat_to_neuron_map(flat)
        neuron_configs[name] = nmap
        n_layers = len(nmap)
        print(
            f"  {name}: {sum(len(v) for v in nmap.values())} neurons across {n_layers} layers"
        )

    if not args.analysis_only:
        print(f"\nLoading model: {args.model_path}")
        model, tokenizer = load_model_and_tokenizer(args.model_path, args.device_map)

        print(f"Loading benchmark: {benchmark}")
        if benchmark == "faitheval":
            samples = load_faitheval()
        elif benchmark == "falseqa":
            samples = load_falseqa(args.falseqa_path)
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

    print("\n" + "=" * 60)
    print("Building comparison summary...")
    print("=" * 60)
    summary = build_comparison_summary(all_results, alphas, benchmark=benchmark)

    summary_path = os.path.join(output_base, "comparison_summary.json")
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

    if "statistical_tests" in summary:
        st = summary["statistical_tests"]
        print(
            f"t-test (compliance@α=3): p={st['compliance_at_alpha_3_h_vs_random_ttest_p']}"
        )
        print(f"t-test (slope): p={st['slope_h_vs_random_ttest_p']}")

    plot_path = os.path.join(output_base, "negative_control_comparison.png")
    plot_comparison(summary, alphas, plot_path, benchmark=benchmark)


if __name__ == "__main__":
    main()
