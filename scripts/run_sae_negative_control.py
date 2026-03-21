"""
Negative control experiment for SAE feature specificity.

Tests whether scaling arbitrary SAE features produces comparable compliance
effects to scaling the classifier-identified H-features. Analogous to
run_negative_control.py but operates in SAE feature space.

Usage:
    systemd-inhibit uv run python scripts/run_sae_negative_control.py \
        --benchmark faitheval \
        --sae_classifier_path models/sae_detector.pkl \
        --sae_classifier_summary data/gemma3_4b/pipeline/classifier_sae_summary.json \
        --extraction_dir data/gemma3_4b/pipeline/activations_sae_hlayers_16k_small/answer_tokens \
        --n_seeds 3 \
        --wandb
"""

import argparse
import json
import os
import sys
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats

sys.path.insert(0, os.path.dirname(__file__))
from extract_sae_activations import load_saes
from intervene_sae import (
    SAEFeatureScaler,
    build_sae_feature_map,
    get_control_sae_feature_indices,
    load_target_features_from_classifier,
)
from run_intervention import (
    _faitheval_prompt,
    extract_mc_answer,
    generate_response,
    load_existing_ids,
    load_faitheval,
    load_model_and_tokenizer,
)
from uncertainty import build_rate_summary, percentile_interval
from utils import (
    finish_run_provenance,
    init_wandb_run,
    log_wandb_files_as_artifact,
    provenance_error_message,
    provenance_status_for_exception,
    start_run_provenance,
)

ALL_ALPHAS = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]


def get_h_feature_info(classifier_path, classifier_summary_path, extraction_dir):
    """Load the classifier's positive-weight SAE features."""
    target_features = load_target_features_from_classifier(
        classifier_path,
        classifier_summary_path=classifier_summary_path,
        extraction_dir=extraction_dir,
    )
    n_features = sum(len(v) for v in target_features.values())
    return target_features, n_features


def get_sae_metadata(classifier_summary_path):
    """Load SAE metadata from the classifier summary."""
    with open(classifier_summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)
    return summary["extraction_metadata"]


def sample_random_sae_features(
    classifier_path,
    meta,
    n_features,
    seed,
):
    """Sample random SAE features from the cleanest available control pool."""
    layer_indices = meta["layer_indices"]
    d_sae = meta["d_sae"]
    eligible, feature_pool = get_control_sae_feature_indices(
        classifier_path,
        min_features=n_features,
    )

    rng = np.random.RandomState(seed)
    chosen_flat = rng.choice(eligible, size=n_features, replace=False)
    return (
        build_sae_feature_map(
            np.sort(chosen_flat),
            layer_indices=layer_indices,
            d_sae=d_sae,
        ),
        feature_pool,
    )


def _count_compliance_and_pf(path):
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


def run_single_sae_config(
    model,
    tokenizer,
    samples,
    saes,
    feature_map,
    alphas,
    output_dir,
    config_name,
    device,
    feature_pool=None,
    sae_steering_mode="full_replacement",
):
    """Run a benchmark for one SAE feature set across all alphas."""
    os.makedirs(output_dir, exist_ok=True)

    # Save feature indices
    indices_path = os.path.join(output_dir, "sae_feature_indices.json")
    with open(indices_path, "w") as f:
        json.dump({str(k): v for k, v in feature_map.items()}, f, indent=2)

    scaler = SAEFeatureScaler(model, saes, feature_map, device, mode=sae_steering_mode)
    n_features = scaler.n_features
    print(
        f"\n[{config_name}] Installed {scaler.n_hooks} SAE hooks on {n_features} features"
        f" (mode={sae_steering_mode})"
    )

    results = {}
    for alpha in alphas:
        out_path = os.path.join(output_dir, f"alpha_{alpha:.1f}.jsonl")
        existing_ids = load_existing_ids(out_path)
        scaler.alpha = alpha

        with open(out_path, "a") as f:
            from tqdm import tqdm

            for sample in tqdm(samples, desc=f"[{config_name}] a={alpha:.1f}"):
                if sample["id"] in existing_ids:
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
            "compliance": build_rate_summary(
                comp_total,
                n_total,
                count_key="n_compliant",
                total_key="n_total",
            ),
        }
        print(
            f"  a={alpha:.1f}: {comp_total / n_total:.1%} compliance "
            f"({comp_total}/{n_total}), {pf_total} parse failures"
        )

    summary = {
        "config": config_name,
        "benchmark": "faitheval",
        "sae_steering_mode": sae_steering_mode,
        "n_sae_features": n_features,
        "results": results,
    }
    if feature_pool is not None:
        summary["feature_pool"] = feature_pool
    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(summary, f, indent=2)

    scaler.remove()
    return results


def build_sae_comparison_summary(
    h_results, all_random_runs, alphas, h_baseline_path=None
):
    """Build comparison summary between H-features and random SAE features."""
    summary: dict[str, object] = {}

    # H-feature rates
    h_rates = [h_results[str(float(a))]["compliance_rate"] for a in alphas]
    alpha_arr = np.array(alphas)
    h_arr = np.array(h_rates)
    h_slope = np.polyfit(alpha_arr, h_arr * 100, 1)[0]
    h_rho = stats.spearmanr(alpha_arr, h_arr).statistic

    summary["h_sae_features"] = {
        "compliance_rates": h_rates,
        "slope_per_alpha": round(float(h_slope), 2),
        "spearman_rho": round(float(h_rho), 4),
        "compliance_ci_by_alpha": [
            h_results[str(float(a))].get("compliance", None)
            or build_rate_summary(
                h_results[str(float(a))]["n_compliant"],
                h_results[str(float(a))]["n_total"],
                count_key="n_compliant",
                total_key="n_total",
            )
            for a in alphas
        ],
    }

    # Random feature results
    per_seed = {}
    all_rates_list = []
    for name, run in sorted(all_random_runs.items()):
        res = run["results"]
        rates = [res[str(float(a))]["compliance_rate"] for a in alphas]
        all_rates_list.append(rates)
        r_arr = np.array(rates)
        slope = np.polyfit(alpha_arr, r_arr * 100, 1)[0]
        rho = stats.spearmanr(alpha_arr, r_arr).statistic
        per_seed[name] = {
            "compliance_rates": rates,
            "slope_per_alpha": round(float(slope), 2),
            "spearman_rho": round(float(rho), 4),
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
        if run.get("feature_pool") is not None:
            per_seed[name]["feature_pool"] = run["feature_pool"]

    if per_seed:
        rates_matrix = np.array(all_rates_list)
        mean_rates = rates_matrix.mean(axis=0).tolist()
        std_rates = rates_matrix.std(axis=0).tolist()
        slopes = [per_seed[k]["slope_per_alpha"] for k in per_seed]

        summary["random_sae_features"] = {
            "mean_compliance_rates": [round(r, 4) for r in mean_rates],
            "std_compliance_rates": [round(s, 4) for s in std_rates],
            "per_seed": per_seed,
            "mean_slope_per_alpha": round(float(np.mean(slopes)), 2),
            "slope_percentile_interval": percentile_interval(
                np.array(slopes, dtype=float),
                method="empirical_random_set_percentile",
            ).to_dict(),
        }

        summary["comparison_to_h_features"] = {
            "slope_h_pp_per_alpha": round(float(h_slope), 2),
            "slope_random_mean_pp_per_alpha": round(float(np.mean(slopes)), 2),
            "slope_random_percentile_interval": percentile_interval(
                np.array(slopes, dtype=float),
                method="empirical_random_set_percentile",
            ).to_dict(),
            "alpha_3_h_rate_pct": round(h_rates[-1] * 100.0, 1),
            "alpha_3_random_mean_pct": round(
                float(np.mean(rates_matrix[:, -1])) * 100, 1
            ),
        }

    # Include neuron baseline if available
    if h_baseline_path and os.path.exists(h_baseline_path):
        with open(h_baseline_path) as f:
            neuron_baseline = json.load(f)
        neuron_rates = [
            neuron_baseline["results"][str(float(a))]["compliance_rate"]
            for a in alphas
            if str(float(a)) in neuron_baseline["results"]
        ]
        if neuron_rates:
            neuron_slope = np.polyfit(
                alpha_arr[: len(neuron_rates)], np.array(neuron_rates) * 100, 1
            )[0]
            summary["neuron_baseline"] = {
                "compliance_rates": neuron_rates,
                "slope_per_alpha": round(float(neuron_slope), 2),
            }

    return summary


def plot_sae_comparison(summary, alphas, output_path):
    """Generate comparison plot: H-features vs random SAE features."""
    fig, ax = plt.subplots(figsize=(10, 6))
    alpha_arr = np.array(alphas)

    # H-features
    h_rates = np.array(summary["h_sae_features"]["compliance_rates"]) * 100
    ax.plot(
        alpha_arr,
        h_rates,
        "o-",
        color="tab:blue",
        linewidth=2.5,
        markersize=8,
        label="H SAE features",
        zorder=10,
    )

    # Random features
    if "random_sae_features" in summary:
        rand = summary["random_sae_features"]
        for name, seed_data in sorted(rand["per_seed"].items()):
            rates = np.array(seed_data["compliance_rates"]) * 100
            ax.plot(alpha_arr, rates, "-", color="gray", alpha=0.35, linewidth=1)

        mean = np.array(rand["mean_compliance_rates"]) * 100
        std = np.array(rand["std_compliance_rates"]) * 100
        ax.plot(
            alpha_arr,
            mean,
            "--",
            color="gray",
            linewidth=2,
            label=f"Random SAE features (mean +/- 1s, n={len(rand['per_seed'])})",
        )
        ax.fill_between(alpha_arr, mean - std, mean + std, color="gray", alpha=0.15)

    # Neuron baseline if available
    if "neuron_baseline" in summary:
        nb = summary["neuron_baseline"]
        nb_rates = np.array(nb["compliance_rates"]) * 100
        ax.plot(
            alpha_arr[: len(nb_rates)],
            nb_rates,
            "s--",
            color="tab:red",
            linewidth=2,
            markersize=6,
            label="Neuron baseline",
            alpha=0.7,
        )

    ax.set_xlabel("Scaling Factor (a)", fontsize=12)
    ax.set_ylabel("Compliance Rate (%)", fontsize=12)
    ax.set_title("SAE Feature Specificity: Negative Control (FaithEval)", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(alpha_arr)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot to {output_path}")


def parse_args():
    p = argparse.ArgumentParser(
        description="SAE negative control experiment for feature specificity"
    )
    p.add_argument(
        "--benchmark",
        type=str,
        default="faitheval",
        choices=["faitheval"],
        help="Benchmark to run (currently FaithEval only)",
    )
    p.add_argument("--model_path", type=str, default="google/gemma-3-4b-it")
    p.add_argument("--device_map", type=str, default="cuda:0")
    p.add_argument(
        "--sae_classifier_path",
        type=str,
        required=True,
        help="Path to SAE classifier .pkl",
    )
    p.add_argument(
        "--sae_classifier_summary",
        type=str,
        required=True,
        help="Path to SAE classifier summary JSON",
    )
    p.add_argument(
        "--extraction_dir",
        type=str,
        required=True,
        help="SAE extraction directory for metadata",
    )
    p.add_argument(
        "--n_seeds",
        type=int,
        default=3,
        help="Number of random seeds for negative control",
    )
    p.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=ALL_ALPHAS,
    )
    p.add_argument(
        "--sae_steering_mode",
        type=str,
        default="full_replacement",
        choices=["full_replacement", "delta_only"],
        help="SAE steering architecture: 'full_replacement' or 'delta_only'",
    )
    p.add_argument(
        "--analysis_only",
        action="store_true",
        help="Skip generation, only run analysis on existing data",
    )
    p.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases run tracking",
    )
    return p.parse_args()


def main():
    args = parse_args()
    benchmark = args.benchmark
    alphas = args.alphas
    if args.sae_steering_mode == "delta_only":
        output_base = f"data/gemma3_4b/intervention/{benchmark}_sae_delta/control"
    else:
        output_base = f"data/gemma3_4b/intervention/{benchmark}_sae/control"
    os.makedirs(output_base, exist_ok=True)

    summary_path = os.path.join(output_base, "comparison_summary.json")
    plot_path = os.path.join(output_base, "sae_negative_control_comparison.png")
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
        # Load H-feature info
        h_features, n_h_features = get_h_feature_info(
            args.sae_classifier_path,
            args.sae_classifier_summary,
            args.extraction_dir,
        )
        meta = get_sae_metadata(args.sae_classifier_summary)
        print(f"H SAE features: {n_h_features} across {len(h_features)} layers")

        # Pre-compute random feature sets
        random_configs = {}
        for seed in range(args.n_seeds):
            name = f"seed_{seed}_random"
            fmap, feature_pool = sample_random_sae_features(
                args.sae_classifier_path, meta, n_h_features, seed
            )
            random_configs[name] = {
                "feature_map": fmap,
                "feature_pool": feature_pool,
            }
            n = sum(len(v) for v in fmap.values())
            print(
                f"  {name}: {n} features across {len(fmap)} layers "
                f"(pool={feature_pool})"
            )

        if args.wandb:
            try:
                import wandb as wandb_module
            except ImportError as exc:
                raise ImportError(
                    "--wandb requested but wandb is not installed."
                ) from exc
            wb_run, wandb_provenance = init_wandb_run(
                wandb_module,
                args,
                job_type="run_sae_negative_control",
                group=f"sae_negative_control:{benchmark}",
                tags=["sae", "negative_control", benchmark],
                config_extra={
                    "n_h_features": n_h_features,
                    "n_seeds": args.n_seeds,
                    "alphas": [float(a) for a in alphas],
                },
            )
            provenance_extra["wandb"] = wandb_provenance

        if not args.analysis_only:
            print(f"\nLoading model: {args.model_path}")
            model, tokenizer = load_model_and_tokenizer(
                args.model_path, args.device_map
            )
            device = next(model.parameters()).device

            # Load all needed SAE layers (union of H-feature + random layers)
            all_layers = set(h_features.keys())
            for cfg in random_configs.values():
                all_layers.update(cfg["feature_map"].keys())
            all_layers_sorted = sorted(all_layers)
            print(f"Loading SAEs for layers: {all_layers_sorted}")
            saes = load_saes(
                meta["sae_release"],
                all_layers_sorted,
                meta["sae_width"],
                meta["sae_l0"],
                str(device),
            )

            print(f"\nLoading benchmark: {benchmark}")
            samples = load_faitheval()
            print(f"  {len(samples)} samples")

            # Run H-feature sweep
            h_out_dir = os.path.join(output_base, "h_features")
            h_results = run_single_sae_config(
                model,
                tokenizer,
                samples,
                saes,
                h_features,
                alphas,
                h_out_dir,
                "h_features",
                device,
                sae_steering_mode=args.sae_steering_mode,
            )

            # Run random feature sweeps
            all_random_runs = {}
            for name, config in random_configs.items():
                out_dir = os.path.join(output_base, name)
                results = run_single_sae_config(
                    model,
                    tokenizer,
                    samples,
                    saes,
                    config["feature_map"],
                    alphas,
                    out_dir,
                    name,
                    device,
                    feature_pool=config["feature_pool"],
                    sae_steering_mode=args.sae_steering_mode,
                )
                all_random_runs[name] = {
                    "results": results,
                    "feature_pool": config["feature_pool"],
                }

            del model, tokenizer
            torch.cuda.empty_cache()
        else:
            # Load existing results
            h_results_path = os.path.join(output_base, "h_features", "results.json")
            if os.path.exists(h_results_path):
                with open(h_results_path) as f:
                    h_results = json.load(f)["results"]
            else:
                print("No H-feature results found.")
                return

            all_random_runs = {}
            for name in random_configs:
                rpath = os.path.join(output_base, name, "results.json")
                if os.path.exists(rpath):
                    with open(rpath) as f:
                        payload = json.load(f)
                    all_random_runs[name] = {
                        "results": payload["results"],
                        "feature_pool": payload.get("feature_pool"),
                    }
                else:
                    print(f"  WARNING: no results for {name}")

        # Build comparison
        neuron_baseline_path = (
            f"data/gemma3_4b/intervention/{benchmark}/experiment/results.json"
        )
        print("\nBuilding comparison summary...")
        comparison = build_sae_comparison_summary(
            h_results,
            all_random_runs,
            alphas,
            h_baseline_path=neuron_baseline_path,
        )

        with open(summary_path, "w") as f:
            json.dump(comparison, f, indent=2)
        print(f"Saved summary to {summary_path}")

        # Print key findings
        print("\n--- KEY FINDINGS ---")
        h = comparison["h_sae_features"]
        print(
            f"H SAE features: slope={h['slope_per_alpha']}%/a, rho={h['spearman_rho']}"
        )

        if "random_sae_features" in comparison:
            r = comparison["random_sae_features"]
            print(f"Random SAE features: mean_slope={r['mean_slope_per_alpha']}%/a")
        if "comparison_to_h_features" in comparison:
            c = comparison["comparison_to_h_features"]
            print(
                f"Slope: H={c['slope_h_pp_per_alpha']}pp/a, "
                f"random mean={c['slope_random_mean_pp_per_alpha']}pp/a"
            )
        if "neuron_baseline" in comparison:
            nb = comparison["neuron_baseline"]
            print(f"Neuron baseline: slope={nb['slope_per_alpha']}%/a")

        plot_sae_comparison(comparison, alphas, plot_path)

        if wb_run is not None and wandb_module is not None:
            for alpha in alphas:
                log_payload = {
                    "alpha": float(alpha),
                    "curve/h_features/compliance_rate_pct": round(
                        h_results[str(float(alpha))]["compliance_rate"] * 100, 2
                    ),
                }
                if "random_sae_features" in comparison:
                    idx = alphas.index(alpha)
                    log_payload["curve/random/mean_compliance_rate_pct"] = round(
                        comparison["random_sae_features"]["mean_compliance_rates"][idx]
                        * 100,
                        2,
                    )
                wandb_module.log(log_payload)

            wb_run.summary["science/h_slope_pp_per_alpha"] = h["slope_per_alpha"]
            if "comparison_to_h_features" in comparison:
                c = comparison["comparison_to_h_features"]
                wb_run.summary["science/random_mean_slope_pp_per_alpha"] = c[
                    "slope_random_mean_pp_per_alpha"
                ]
            log_wandb_files_as_artifact(
                wb_run,
                wandb_module,
                name=f"sae-negative-control-{benchmark}",
                artifact_type="sae_negative_control_summary",
                paths=[summary_path, plot_path],
            )

        provenance_extra["output_targets"] = [output_base, summary_path, plot_path]
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
