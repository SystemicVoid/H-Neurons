"""
Investigate Neuron (Layer 20, Neuron 4288) — Hallucination Hub or L1 Artifact?

Six analyses to determine whether the dominant H-Neuron's weight (12.169, 1.65x
the runner-up) reflects genuine signal or L1 regularization artifact.

Usage:
    uv run python scripts/investigate_neuron_4288.py \
        --classifier models/gemma3_4b_classifier_disjoint.pkl \
        --train_ids data/gemma3_4b/train_qids.json \
        --test_ids data/gemma3_4b/test_qids_disjoint.json \
        --train_ans_acts data/gemma3_4b/activations/answer_tokens \
        --train_other_acts data/gemma3_4b/activations/all_except_answer_tokens \
        --test_acts data/gemma3_4b/activations/answer_tokens \
        --output_dir data/gemma3_4b/investigation_neuron_4288
"""

import argparse
import json
import os

import joblib
import matplotlib
import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Constants
NUM_LAYERS = 34
NUM_NEURONS = 10240
TOTAL_FEATURES = NUM_LAYERS * NUM_NEURONS  # 348,160

# Target neuron
TARGET_LAYER = 20
TARGET_NEURON = 4288
TARGET_IDX = TARGET_LAYER * NUM_NEURONS + TARGET_NEURON  # 209,168


def parse_args():
    p = argparse.ArgumentParser(description="Investigate Neuron (20, 4288)")
    p.add_argument("--classifier", required=True, help="Path to trained classifier .pkl")
    p.add_argument("--train_ids", required=True, help="Path to train_qids.json")
    p.add_argument("--test_ids", required=True, help="Path to test_qids.json")
    p.add_argument("--train_ans_acts", required=True, help="Dir of answer token activations")
    p.add_argument("--train_other_acts", required=True, help="Dir of other token activations (for 3-vs-1)")
    p.add_argument("--test_acts", required=True, help="Dir of test activations")
    p.add_argument("--output_dir", default="data/investigation_neuron_4288", help="Dir for plots")
    p.add_argument("--skip_csweep", action="store_true", help="Skip the slow C-sweep analysis")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data loading (replicates classifier.py pattern)
# ---------------------------------------------------------------------------

def load_data(ids_path, ans_acts_dir, other_acts_dir=None, mode="1-vs-1"):
    with open(ids_path, "r") as f:
        id_map = json.load(f)

    X, y, qids = [], [], []

    for qid in tqdm(id_map["f"], desc="Loading False Ans (Label 1)"):
        path = os.path.join(ans_acts_dir, f"act_{qid}.npy")
        if os.path.exists(path):
            X.append(np.load(path).flatten())
            y.append(1)
            qids.append(qid)

    for qid in tqdm(id_map["t"], desc="Loading True Ans (Label 0)"):
        path = os.path.join(ans_acts_dir, f"act_{qid}.npy")
        if os.path.exists(path):
            X.append(np.load(path).flatten())
            y.append(0)
            qids.append(qid)

    if mode == "3-vs-1":
        if not other_acts_dir:
            raise ValueError("other_acts_dir required for 3-vs-1 mode")
        for label_key in ["t", "f"]:
            for qid in tqdm(id_map[label_key], desc=f"Loading Other - {label_key} (Label 0)"):
                path = os.path.join(other_acts_dir, f"act_{qid}.npy")
                if os.path.exists(path):
                    X.append(np.load(path).flatten())
                    y.append(0)
                    qids.append(f"{qid}_other")

    return np.array(X), np.array(y), qids


def idx_to_loc(idx):
    return (idx // NUM_NEURONS, idx % NUM_NEURONS)


def loc_to_str(idx):
    layer, neuron = idx_to_loc(idx)
    return f"L{layer}:N{neuron}"


# ---------------------------------------------------------------------------
# Analysis 1: Single-Neuron Classification
# ---------------------------------------------------------------------------

def analysis_single_neuron(X_train_1v1, y_train_1v1, X_test, y_test, model):
    print("\n" + "=" * 60)
    print("ANALYSIS 1: Single-Neuron Classification")
    print("=" * 60)

    coef = model.coef_[0]
    positive_idxs = np.where(coef > 0)[0]
    top10_pos = positive_idxs[np.argsort(coef[positive_idxs])[::-1][:10]]

    rng = np.random.RandomState(42)
    zero_idxs = np.where(coef == 0)[0]
    random10 = rng.choice(zero_idxs, size=10, replace=False)

    all_idxs = np.concatenate([top10_pos, random10])
    labels = [f"{loc_to_str(i)} (w={coef[i]:.2f})" for i in top10_pos] + [
        f"{loc_to_str(i)} (random)" for i in random10
    ]

    results = {}
    for idx, label in zip(all_idxs, labels):
        x_tr = X_train_1v1[:, idx].reshape(-1, 1)
        x_te = X_test[:, idx].reshape(-1, 1)
        lr = LogisticRegression(max_iter=1000, random_state=42)
        lr.fit(x_tr, y_train_1v1)
        probs = lr.predict_proba(x_te)[:, 1]
        auc = roc_auc_score(y_test, probs)
        acc = accuracy_score(y_test, lr.predict(x_te))
        results[label] = {"auc": auc, "acc": acc, "idx": idx}
        is_target = "  <<<" if idx == TARGET_IDX else ""
        print(f"  {label:40s}  AUC={auc:.3f}  Acc={acc:.3f}{is_target}")

    target_label = [k for k, r in results.items() if r["idx"] == TARGET_IDX][0]
    target_auc = results[target_label]["auc"]
    random_aucs = [r["auc"] for k, r in results.items() if "random" in k]
    mean_random = np.mean(random_aucs)

    print(f"\n  Target AUC: {target_auc:.3f}")
    print(f"  Mean random AUC: {mean_random:.3f}")
    print(f"  Verdict: {'REAL SIGNAL' if target_auc > 0.60 else 'WEAK/ARTIFACT'}")

    return {"results": results, "target_auc": target_auc, "mean_random_auc": mean_random}


# ---------------------------------------------------------------------------
# Analysis 2: Activation Distribution Separation
# ---------------------------------------------------------------------------

def analysis_distributions(X_test, y_test, model):
    print("\n" + "=" * 60)
    print("ANALYSIS 2: Activation Distribution Separation")
    print("=" * 60)

    coef = model.coef_[0]
    positive_idxs = np.where(coef > 0)[0]
    runner_up_idx = positive_idxs[np.argsort(coef[positive_idxs])[::-1][1]]

    rng = np.random.RandomState(42)
    zero_idxs = np.where(coef == 0)[0]
    random_idx = rng.choice(zero_idxs)

    neurons = [
        ("Target (20:4288)", TARGET_IDX),
        (f"Runner-up ({loc_to_str(runner_up_idx)})", runner_up_idx),
        (f"Random ({loc_to_str(random_idx)})", random_idx),
    ]

    results = {}
    for name, idx in neurons:
        vals = X_test[:, idx]
        true_vals = vals[y_test == 0]
        false_vals = vals[y_test == 1]

        # Cohen's d
        pooled_std = np.sqrt((np.std(true_vals) ** 2 + np.std(false_vals) ** 2) / 2)
        d = (np.mean(false_vals) - np.mean(true_vals)) / pooled_std if pooled_std > 0 else 0

        # Mann-Whitney U
        u_stat, u_p = stats.mannwhitneyu(true_vals, false_vals, alternative="two-sided")

        # KS test
        ks_stat, ks_p = stats.ks_2samp(true_vals, false_vals)

        results[name] = {
            "idx": idx,
            "cohen_d": d,
            "mw_u": u_stat,
            "mw_p": u_p,
            "ks_stat": ks_stat,
            "ks_p": ks_p,
            "true_mean": np.mean(true_vals),
            "false_mean": np.mean(false_vals),
            "true_vals": true_vals,
            "false_vals": false_vals,
        }

        print(f"\n  {name}:")
        print(f"    Mean (true): {np.mean(true_vals):.4f}, Mean (false): {np.mean(false_vals):.4f}")
        print(f"    Cohen's d: {d:.3f}")
        print(f"    Mann-Whitney U p: {u_p:.2e}")
        print(f"    KS stat: {ks_stat:.3f}, p: {ks_p:.2e}")

    target_d = results["Target (20:4288)"]["cohen_d"]
    print(f"\n  Target Cohen's d: {target_d:.3f}")
    print(f"  Verdict: {'REAL SIGNAL' if abs(target_d) > 0.5 else 'WEAK/ARTIFACT'}")

    return results


# ---------------------------------------------------------------------------
# Analysis 3: C-Sweep Stability
# ---------------------------------------------------------------------------

def analysis_c_sweep(X_train_3v1, y_train_3v1, X_test, y_test):
    print("\n" + "=" * 60)
    print("ANALYSIS 3: C-Sweep Stability")
    print("=" * 60)

    C_values = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
    sweep_results = []

    for c_val in C_values:
        print(f"\n  Training C={c_val}...", end=" ", flush=True)
        lr = LogisticRegression(
            penalty="l1", C=c_val, solver="liblinear",
            max_iter=1000, random_state=42, verbose=0,
        )
        lr.fit(X_train_3v1, y_train_3v1)

        coef = lr.coef_[0]
        preds = lr.predict(X_test)
        probs = lr.predict_proba(X_test)[:, 1]
        acc = accuracy_score(y_test, preds)
        auc = roc_auc_score(y_test, probs)

        n_nonzero = np.sum(coef != 0)
        n_positive = np.sum(coef > 0)
        target_weight = coef[TARGET_IDX]

        # Rank among positive-weight neurons
        positive_idxs = np.where(coef > 0)[0]
        if target_weight > 0:
            rank = int(np.sum(coef[positive_idxs] > target_weight)) + 1
        else:
            rank = -1  # Not in positive set

        # Top 5 positive neurons for tracking
        top5 = {}
        if len(positive_idxs) > 0:
            sorted_pos = positive_idxs[np.argsort(coef[positive_idxs])[::-1][:5]]
            for idx in sorted_pos:
                top5[idx] = coef[idx]

        result = {
            "C": c_val,
            "acc": acc,
            "auc": auc,
            "n_nonzero": n_nonzero,
            "n_positive": n_positive,
            "target_weight": target_weight,
            "target_rank": rank,
            "top5": top5,
        }
        sweep_results.append(result)

        rank_str = f"rank {rank}" if rank > 0 else "NOT SELECTED"
        print(f"Acc={acc:.3f} AUC={auc:.3f} NZ={n_nonzero} Pos={n_positive} "
              f"4288_w={target_weight:.3f} ({rank_str})")

    # Summary
    c_selected = sum(1 for r in sweep_results if r["target_rank"] > 0)
    c_top3 = sum(1 for r in sweep_results if 0 < r["target_rank"] <= 3)
    print(f"\n  Neuron 4288 selected in {c_selected}/{len(C_values)} C values")
    print(f"  Neuron 4288 in top-3 for {c_top3}/{len(C_values)} C values")
    print(f"  Verdict: {'REAL SIGNAL' if c_selected >= 5 else 'L1 ARTIFACT'}")

    return sweep_results


# ---------------------------------------------------------------------------
# Analysis 4: Per-Example Contribution
# ---------------------------------------------------------------------------

def analysis_contributions(model, X_test, y_test):
    print("\n" + "=" * 60)
    print("ANALYSIS 4: Per-Example Contribution")
    print("=" * 60)

    coef = model.coef_[0]
    intercept = model.intercept_[0]

    # Per-example scores
    scores_total = X_test @ coef + intercept
    scores_4288 = coef[TARGET_IDX] * X_test[:, TARGET_IDX]

    # Fraction of total score from 4288
    # Use absolute contribution to handle sign issues
    abs_contributions = np.abs(coef) * np.abs(X_test)
    abs_total = abs_contributions.sum(axis=1)
    abs_4288 = abs_contributions[:, TARGET_IDX]
    frac_4288 = abs_4288 / (abs_total + 1e-12)

    # Is 4288 the single largest positive contributor?
    positive_idxs = np.where(coef > 0)[0]
    positive_contributions = coef[positive_idxs] * X_test[:, positive_idxs]
    max_pos_contributor = positive_idxs[np.argmax(positive_contributions, axis=1)]
    is_largest = max_pos_contributor == TARGET_IDX
    pct_largest = np.mean(is_largest) * 100

    preds = model.predict(X_test)
    correct = preds == y_test

    print(f"  Median abs contribution fraction: {np.median(frac_4288):.3f}")
    print(f"  Mean abs contribution fraction: {np.mean(frac_4288):.3f}")
    print(f"  Largest positive contributor for {pct_largest:.1f}% of examples")
    print(f"  Mean frac (correct predictions): {np.mean(frac_4288[correct]):.3f}")
    print(f"  Mean frac (incorrect predictions): {np.mean(frac_4288[~correct]):.3f}")
    print(f"  Verdict: {'REAL SIGNAL' if pct_largest > 30 else 'WEAK/ARTIFACT'}")

    return {
        "scores_total": scores_total,
        "scores_4288": scores_4288,
        "frac_4288": frac_4288,
        "pct_largest": pct_largest,
        "correct": correct,
        "y_test": y_test,
    }


# ---------------------------------------------------------------------------
# Analysis 5: Leave-One-Out Ablation
# ---------------------------------------------------------------------------

def analysis_ablation(model, X_test, y_test):
    print("\n" + "=" * 60)
    print("ANALYSIS 5: Leave-One-Out Ablation")
    print("=" * 60)

    coef = model.coef_[0]
    positive_idxs = np.where(coef > 0)[0]
    top10_pos = positive_idxs[np.argsort(coef[positive_idxs])[::-1][:10]]

    baseline_acc = accuracy_score(y_test, model.predict(X_test))
    baseline_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    print(f"  Baseline: Acc={baseline_acc:.4f}, AUC={baseline_auc:.4f}")

    results = {}
    for idx in top10_pos:
        X_abl = X_test.copy()
        X_abl[:, idx] = 0
        abl_acc = accuracy_score(y_test, model.predict(X_abl))
        abl_auc = roc_auc_score(y_test, model.predict_proba(X_abl)[:, 1])
        drop_acc = baseline_acc - abl_acc
        drop_auc = baseline_auc - abl_auc
        label = loc_to_str(idx)
        results[label] = {
            "idx": idx,
            "acc": abl_acc,
            "auc": abl_auc,
            "drop_acc": drop_acc,
            "drop_auc": drop_auc,
        }
        is_target = "  <<<" if idx == TARGET_IDX else ""
        print(f"  Ablate {label:12s}  Acc={abl_acc:.4f} (drop {drop_acc:+.4f})  "
              f"AUC={abl_auc:.4f} (drop {drop_auc:+.4f}){is_target}")

    # Ablate all positives except 4288
    X_only4288 = X_test.copy()
    for idx in positive_idxs:
        if idx != TARGET_IDX:
            X_only4288[:, idx] = 0
    only_acc = accuracy_score(y_test, model.predict(X_only4288))
    only_auc = roc_auc_score(y_test, model.predict_proba(X_only4288)[:, 1])
    print(f"\n  Only 4288 + negatives: Acc={only_acc:.4f}, AUC={only_auc:.4f}")
    results["only_4288"] = {"acc": only_acc, "auc": only_auc}

    target_drop = results[loc_to_str(TARGET_IDX)]["drop_acc"]
    print(f"\n  Target accuracy drop: {target_drop:+.4f}")
    print(f"  Verdict: {'REAL SIGNAL' if abs(target_drop) > 0.02 else 'WEAK/ARTIFACT'}")

    return {"baseline_acc": baseline_acc, "baseline_auc": baseline_auc, **results}


# ---------------------------------------------------------------------------
# Analysis 6: Correlation Structure
# ---------------------------------------------------------------------------

def analysis_correlations(X_train_1v1, model):
    print("\n" + "=" * 60)
    print("ANALYSIS 6: Correlation Structure")
    print("=" * 60)

    coef = model.coef_[0]
    positive_idxs = np.where(coef > 0)[0]
    top10_pos = positive_idxs[np.argsort(coef[positive_idxs])[::-1][:10]]

    # Find 10 zero-weight neurons from layers near layer 20 (18-22)
    zero_idxs = np.where(coef == 0)[0]
    nearby_zero = []
    for layer in range(18, 23):
        layer_start = layer * NUM_NEURONS
        layer_end = (layer + 1) * NUM_NEURONS
        layer_zeros = zero_idxs[(zero_idxs >= layer_start) & (zero_idxs < layer_end)]
        if len(layer_zeros) > 0:
            rng = np.random.RandomState(42 + layer)
            chosen = rng.choice(layer_zeros, size=min(2, len(layer_zeros)), replace=False)
            nearby_zero.extend(chosen)
    nearby_zero = np.array(nearby_zero[:10])

    all_idxs = np.concatenate([top10_pos, nearby_zero])
    labels = [f"{loc_to_str(i)} (w={coef[i]:.2f})" for i in top10_pos] + [
        f"{loc_to_str(i)} (zero)" for i in nearby_zero
    ]

    # Compute correlation matrix
    features = X_train_1v1[:, all_idxs]
    corr_matrix = np.corrcoef(features.T)

    # Report correlations with target
    target_pos_in_array = np.where(all_idxs == TARGET_IDX)[0][0]
    print(f"\n  Correlations with {loc_to_str(TARGET_IDX)}:")
    for i, (idx, label) in enumerate(zip(all_idxs, labels)):
        if idx == TARGET_IDX:
            continue
        r = corr_matrix[target_pos_in_array, i]
        flag = " ***" if abs(r) > 0.5 else ""
        print(f"    {label:40s}  r={r:+.3f}{flag}")

    # Summary stats
    top10_corrs = [corr_matrix[target_pos_in_array, i]
                   for i in range(len(top10_pos)) if all_idxs[i] != TARGET_IDX]
    zero_corrs = [corr_matrix[target_pos_in_array, i]
                  for i in range(len(top10_pos), len(all_idxs))]

    max_top10_r = max(abs(r) for r in top10_corrs) if top10_corrs else 0
    max_zero_r = max(abs(r) for r in zero_corrs) if zero_corrs else 0

    print(f"\n  Max |r| with other top-10: {max_top10_r:.3f}")
    print(f"  Max |r| with nearby zero-weight: {max_zero_r:.3f}")
    print(f"  Verdict: {'REAL SIGNAL' if max_top10_r < 0.3 and max_zero_r < 0.5 else 'POSSIBLE ARTIFACT'}")

    return {
        "corr_matrix": corr_matrix,
        "labels": labels,
        "all_idxs": all_idxs,
        "max_top10_r": max_top10_r,
        "max_zero_r": max_zero_r,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(output_dir, a1, a2, a3, a4, a5, a6):
    os.makedirs(output_dir, exist_ok=True)

    # --- Plot 1: Single-Neuron AUCs ---
    fig, ax = plt.subplots(figsize=(12, 5))
    labels = list(a1["results"].keys())
    aucs = [a1["results"][lb]["auc"] for lb in labels]
    colors = ["#d62728" if a1["results"][lb]["idx"] == TARGET_IDX
              else "#1f77b4" if "random" not in lb
              else "#7f7f7f" for lb in labels]
    short_labels = [lb.split(" (")[0] for lb in labels]
    ax.bar(range(len(labels)), aucs, color=colors)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=8)
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="Chance")
    ax.set_ylabel("AUC (disjoint test)")
    ax.set_title("Analysis 1: Single-Neuron Classification AUC")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "01_single_neuron_auc.png"), dpi=150)
    plt.close()

    # --- Plot 2: Distribution Separation ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (name, data) in zip(axes, a2.items()):
        true_v = data["true_vals"]
        false_v = data["false_vals"]
        ax.hist(true_v, bins=50, alpha=0.6, label="True (no hallucination)", color="#2ca02c", density=True)
        ax.hist(false_v, bins=50, alpha=0.6, label="False (hallucination)", color="#d62728", density=True)
        ax.set_title(f"{name}\nCohen's d={data['cohen_d']:.3f}, MW p={data['mw_p']:.1e}")
        ax.set_xlabel("CETT Activation")
        ax.legend(fontsize=7)
    fig.suptitle("Analysis 2: Activation Distributions (True vs False)", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "02_distributions.png"), dpi=150)
    plt.close()

    # --- Plot 3: C-Sweep ---
    if a3 is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        c_vals = [r["C"] for r in a3]
        # Track top-5 neurons across sweep
        all_top_neurons = set()
        for r in a3:
            all_top_neurons.update(r["top5"].keys())

        # For each neuron that ever appeared in top-5, track its weight
        tracked = {}
        for nidx in all_top_neurons:
            weights = []
            for r in a3:
                weights.append(r["top5"].get(nidx, 0.0))
            if max(abs(w) for w in weights) > 0.5:  # Only track meaningful neurons
                tracked[nidx] = weights

        for nidx, weights in sorted(tracked.items(), key=lambda x: -max(x[1])):
            label = loc_to_str(nidx)
            style = {"linewidth": 2.5, "color": "#d62728"} if nidx == TARGET_IDX else {"linewidth": 1, "alpha": 0.7}
            ax1.plot(c_vals, weights, "o-", label=label, **style)

        ax1.set_xscale("log")
        ax1.set_xlabel("C (regularization)")
        ax1.set_ylabel("Weight")
        ax1.set_title("Top Positive Neuron Weights Across C")
        ax1.legend(fontsize=7, loc="upper left")

        accs = [r["acc"] for r in a3]
        aucs_sweep = [r["auc"] for r in a3]
        ax2.plot(c_vals, accs, "o-", label="Accuracy", color="#1f77b4")
        ax2.plot(c_vals, aucs_sweep, "s-", label="AUC", color="#ff7f0e")
        ax2.set_xscale("log")
        ax2.set_xlabel("C (regularization)")
        ax2.set_ylabel("Score")
        ax2.set_title("Test Performance Across C")
        ax2.legend()

        fig.suptitle("Analysis 3: C-Sweep Stability", fontsize=12)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "03_c_sweep.png"), dpi=150)
        plt.close()

    # --- Plot 4: Contribution Scatter ---
    fig, ax = plt.subplots(figsize=(8, 6))
    y = a4["y_test"]
    correct = a4["correct"]
    for c_val, m, label in [("#2ca02c", "o", "True (correct)"), ("#d62728", "o", "False (correct)"),
                             ("#2ca02c", "x", "True (wrong)"), ("#d62728", "x", "False (wrong)")]:
        is_false = c_val == "#d62728"
        is_correct = m == "o"
        mask = (y == int(is_false)) & (correct == is_correct)
        if mask.sum() > 0:
            ax.scatter(a4["scores_total"][mask], a4["scores_4288"][mask],
                      c=c_val, marker=m, alpha=0.4, s=20, label=label)
    ax.axhline(0, color="gray", linestyle="--", alpha=0.3)
    ax.axvline(0, color="gray", linestyle="--", alpha=0.3)
    ax.set_xlabel("Total log-odds score")
    ax.set_ylabel("Neuron 4288 contribution")
    ax.set_title(f"Analysis 4: Per-Example Contribution\n"
                 f"4288 is largest positive contributor for {a4['pct_largest']:.1f}% of examples")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "04_contributions.png"), dpi=150)
    plt.close()

    # --- Plot 5: Ablation Bar Chart ---
    fig, ax = plt.subplots(figsize=(10, 5))
    ablation_items = [(k, v) for k, v in a5.items()
                      if k not in ("baseline_acc", "baseline_auc", "only_4288") and isinstance(v, dict)]
    ablation_items.sort(key=lambda x: -abs(x[1]["drop_acc"]))
    labels_abl = [k for k, _ in ablation_items]
    drops = [v["drop_acc"] for _, v in ablation_items]
    colors_abl = ["#d62728" if loc_to_str(TARGET_IDX) == k else "#1f77b4" for k in labels_abl]
    ax.bar(range(len(labels_abl)), drops, color=colors_abl)
    ax.set_xticks(range(len(labels_abl)))
    ax.set_xticklabels(labels_abl, rotation=45, ha="right")
    ax.set_ylabel("Accuracy Drop (pp)")
    ax.set_title("Analysis 5: Ablation — Accuracy Drop per Neuron")
    ax.axhline(0, color="gray", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "05_ablation.png"), dpi=150)
    plt.close()

    # --- Plot 6: Correlation Heatmap ---
    fig, ax = plt.subplots(figsize=(10, 8))
    short_labels_corr = [lb.split(" (")[0] for lb in a6["labels"]]
    im = ax.imshow(a6["corr_matrix"], cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(short_labels_corr)))
    ax.set_yticks(range(len(short_labels_corr)))
    ax.set_xticklabels(short_labels_corr, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(short_labels_corr, fontsize=7)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Pearson r")
    ax.set_title("Analysis 6: Correlation Structure (Top-10 + Nearby Zero-Weight)")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "06_correlations.png"), dpi=150)
    plt.close()

    print(f"\n  All plots saved to {output_dir}/")


# ---------------------------------------------------------------------------
# Final Verdict
# ---------------------------------------------------------------------------

def print_verdict(a1, a2, a3, a4, a5, a6):
    print("\n" + "=" * 60)
    print("FINAL VERDICT")
    print("=" * 60)

    votes = {"real": 0, "artifact": 0}

    # 1. Single-neuron AUC
    v1 = "real" if a1["target_auc"] > 0.60 else "artifact"
    votes[v1] += 1
    print(f"  1. Single-neuron AUC={a1['target_auc']:.3f}: {v1.upper()}")

    # 2. Distribution separation
    target_d = list(a2.values())[0]["cohen_d"]
    v2 = "real" if abs(target_d) > 0.5 else "artifact"
    votes[v2] += 1
    print(f"  2. Cohen's d={target_d:.3f}: {v2.upper()}")

    # 3. C-sweep
    if a3 is not None:
        c_selected = sum(1 for r in a3 if r["target_rank"] > 0)
        v3 = "real" if c_selected >= 5 else "artifact"
        votes[v3] += 1
        print(f"  3. C-sweep: selected in {c_selected}/9: {v3.upper()}")
    else:
        print("  3. C-sweep: SKIPPED")

    # 4. Contribution
    v4 = "real" if a4["pct_largest"] > 30 else "artifact"
    votes[v4] += 1
    print(f"  4. Largest contributor for {a4['pct_largest']:.1f}%: {v4.upper()}")

    # 5. Ablation
    target_key = loc_to_str(TARGET_IDX)
    if target_key in a5 and isinstance(a5[target_key], dict):
        drop = abs(a5[target_key]["drop_acc"])
        v5 = "real" if drop > 0.02 else "artifact"
        votes[v5] += 1
        print(f"  5. Ablation drop={drop:.4f}: {v5.upper()}")

    # 6. Correlation
    v6 = "real" if a6["max_top10_r"] < 0.3 and a6["max_zero_r"] < 0.5 else "artifact"
    votes[v6] += 1
    print(f"  6. Max |r| top10={a6['max_top10_r']:.3f}, zero={a6['max_zero_r']:.3f}: {v6.upper()}")

    total = votes["real"] + votes["artifact"]
    print(f"\n  SCORE: {votes['real']}/{total} analyses point to REAL SIGNAL")
    if votes["real"] >= 4:
        print("  CONCLUSION: Neuron (20, 4288) is likely a GENUINE HALLUCINATION HUB")
    elif votes["artifact"] >= 4:
        print("  CONCLUSION: Neuron (20, 4288) dominance is likely an L1 ARTIFACT")
    else:
        print("  CONCLUSION: INCONCLUSIVE — mixed evidence, further investigation needed")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading classifier...")
    model = joblib.load(args.classifier)

    print("Loading test data (1-vs-1)...")
    X_test, y_test, _ = load_data(args.test_ids, args.test_acts)

    print("Loading train data (1-vs-1)...")
    X_train_1v1, y_train_1v1, _ = load_data(args.train_ids, args.train_ans_acts)

    # Run analyses 1, 2, 4, 5, 6 (fast)
    a1 = analysis_single_neuron(X_train_1v1, y_train_1v1, X_test, y_test, model)
    a2 = analysis_distributions(X_test, y_test, model)
    a4 = analysis_contributions(model, X_test, y_test)
    a5 = analysis_ablation(model, X_test, y_test)
    a6 = analysis_correlations(X_train_1v1, model)

    # C-sweep (slow — load 3-vs-1 data)
    a3 = None
    if not args.skip_csweep:
        print("\nLoading train data (3-vs-1) for C-sweep...")
        X_train_3v1, y_train_3v1, _ = load_data(
            args.train_ids, args.train_ans_acts,
            other_acts_dir=args.train_other_acts, mode="3-vs-1",
        )
        a3 = analysis_c_sweep(X_train_3v1, y_train_3v1, X_test, y_test)

    # Plots
    plot_results(args.output_dir, a1, a2, a3, a4, a5, a6)

    # Final verdict
    print_verdict(a1, a2, a3, a4, a5, a6)


if __name__ == "__main__":
    main()
