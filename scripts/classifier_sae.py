"""Train and evaluate an L1 logistic regression probe on SAE features.

Mirrors the CETT classifier pipeline (scripts/classifier.py) but operates
on SAE feature vectors instead of CETT-normalized activations.

Usage (3-vs-1, with C sweep):
    uv run python scripts/classifier_sae.py \
        --model_path google/gemma-3-4b-it \
        --train_ids data/gemma3_4b/pipeline/train_qids.json \
        --train_ans_acts data/gemma3_4b/pipeline/activations_sae/answer_tokens \
        --train_other_acts data/gemma3_4b/pipeline/activations_sae/all_except_answer_tokens \
        --test_ids data/gemma3_4b/pipeline/test_qids_disjoint.json \
        --test_acts data/gemma3_4b/pipeline/activations_sae/answer_tokens \
        --c_values 0.001 0.01 0.1 1.0 10.0 \
        --metrics_out data/gemma3_4b/pipeline/classifier_sae_summary.json

1-vs-1 mode:
    uv run python scripts/classifier_sae.py \
        --train_mode 1-vs-1 \
        --train_ids data/gemma3_4b/pipeline/train_qids.json \
        --train_ans_acts data/gemma3_4b/pipeline/activations_sae/answer_tokens \
        --test_ids data/gemma3_4b/pipeline/test_qids_disjoint.json \
        --test_acts data/gemma3_4b/pipeline/activations_sae/answer_tokens \
        --metrics_out data/gemma3_4b/pipeline/classifier_sae_1v1_summary.json
"""

import argparse
import json
import os

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

from classifier import (
    compute_metrics,
    ensure_parent_dir,
    evaluate_model_with_uncertainty,
    print_metrics,
    select_candidate_score,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train/evaluate a hallucination detector on SAE features."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="google/gemma-3-4b-it",
        help="Model path (used only for config metadata).",
    )

    # Data paths
    parser.add_argument("--train_ids", type=str, help="Path to train_qids.json")
    parser.add_argument(
        "--train_ans_acts",
        type=str,
        help="Directory of SAE answer-token activations for training",
    )
    parser.add_argument(
        "--train_other_acts",
        type=str,
        help="Directory of SAE non-answer activations (for 3-vs-1)",
    )
    parser.add_argument("--test_ids", type=str, help="Path to held-out qids.json")
    parser.add_argument(
        "--test_acts",
        type=str,
        help="Directory of SAE held-out activations (answer tokens)",
    )

    # Model paths
    parser.add_argument("--save_model", type=str, default="models/sae_detector.pkl")
    parser.add_argument("--load_model", type=str, help="Load pre-trained model")
    parser.add_argument(
        "--metrics_out",
        type=str,
        default=None,
        help="JSON path for metrics output.",
    )

    # Training config
    parser.add_argument(
        "--train_mode",
        type=str,
        choices=["1-vs-1", "3-vs-1"],
        default="3-vs-1",
    )
    parser.add_argument("--penalty", type=str, choices=["l1", "l2"], default="l1")
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--c_values", type=float, nargs="+", default=None)
    parser.add_argument(
        "--selection_metric",
        type=str,
        choices=["accuracy", "precision", "recall", "f1", "auroc"],
        default="auroc",
    )
    parser.add_argument("--solver", type=str, default="liblinear")

    return parser.parse_args()


def load_sae_data(
    ids_path, ans_acts_dir, other_acts_dir=None, mode="1-vs-1", return_qids=False
):
    """Load SAE feature vectors, flattened to 1D per sample.

    SAE .npy files have shape [n_layers, d_sae]. We flatten to
    [n_layers * d_sae] to match the CETT classifier's flat feature vector.
    """
    with open(ids_path, "r", encoding="utf-8") as f:
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
            raise ValueError("train_other_acts directory required for 3-vs-1 mode.")

        for label_key in ["t", "f"]:
            for qid in tqdm(
                id_map[label_key],
                desc=f"Loading Other Tokens - {label_key} (Label 0)",
            ):
                path = os.path.join(other_acts_dir, f"act_{qid}.npy")
                if os.path.exists(path):
                    X.append(np.load(path).flatten())
                    y.append(0)
                    qids.append(f"{qid}_other")

    X_arr = np.array(X)
    y_arr = np.array(y)
    if return_qids:
        return X_arr, y_arr, qids
    return X_arr, y_arr


def get_sae_feature_info(X, n_layers=None, d_sae=None):
    """Infer SAE feature dimensions from flattened feature vector."""
    total_features = X.shape[1]
    if n_layers is not None and d_sae is not None:
        assert total_features == n_layers * d_sae
    return total_features


def summarize_sae_candidate(candidate, selection_metric, prefer_test):
    """Build summary dict for a trained candidate."""
    coef = candidate["model"].coef_[0]
    n_positive = int(np.sum(coef > 0))
    n_nonzero = int(np.sum(coef != 0))
    score = select_candidate_score(candidate, selection_metric, prefer_test)
    return {
        "C": candidate["C"],
        "score": float(score),
        "n_positive_features": n_positive,
        "n_nonzero_features": n_nonzero,
        "total_features": len(coef),
        "train_metrics": candidate["train_metrics"],
        "test_metrics": candidate["test_metrics"],
    }


def decode_sae_feature_indices(flat_indices, n_layers, d_sae, layer_indices):
    """Map flat feature indices back to (layer, sae_feature_idx) pairs."""
    decoded = []
    for idx in flat_indices:
        layer_pos = int(idx // d_sae)
        feature_idx = int(idx % d_sae)
        layer_id = layer_indices[layer_pos] if layer_indices else layer_pos
        decoded.append(
            {"layer": layer_id, "feature": feature_idx, "flat_idx": int(idx)}
        )
    return decoded


def main():
    args = parse_args()
    prefer_test = bool(args.test_ids and args.test_acts)

    # Load test data
    X_test = y_test = test_qids = None
    if prefer_test:
        print(f"Loading held-out SAE features from {args.test_ids}...")
        X_test, y_test, test_qids = load_sae_data(
            args.test_ids,
            args.test_acts,
            mode="1-vs-1",
            return_qids=True,
        )
        print(f"  Test set: {len(y_test)} samples, {X_test.shape[1]} features")

    if args.load_model:
        print(f"Loading pre-trained model: {args.load_model}")
        model = joblib.load(args.load_model)
        n_positive = int(np.sum(model.coef_[0] > 0))
        n_nonzero = int(np.sum(model.coef_[0] != 0))
        print(f"Loaded model: {n_positive} positive, {n_nonzero} nonzero features")

        if prefer_test:
            metrics = compute_metrics(model, X_test, y_test)
            print_metrics(metrics, "Test Set")
            evaluation = evaluate_model_with_uncertainty(
                model, X_test, y_test, test_qids
            )
        else:
            evaluation = None

        if args.metrics_out:
            ensure_parent_dir(args.metrics_out)
            payload = {
                "feature_type": "sae",
                "model_path": args.model_path,
                "loaded_model_path": args.load_model,
                "n_positive_features": n_positive,
                "n_nonzero_features": n_nonzero,
                "evaluation": evaluation,
            }
            with open(args.metrics_out, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            print(f"Saved metrics to {args.metrics_out}")
        return

    if not (args.train_ids and args.train_ans_acts):
        print("Please provide --load_model OR (--train_ids AND --train_ans_acts)")
        return

    # Load training data
    print(f"Training in {args.train_mode} mode...")
    X_train, y_train = load_sae_data(
        args.train_ids,
        args.train_ans_acts,
        other_acts_dir=args.train_other_acts,
        mode=args.train_mode,
    )
    total_features = get_sae_feature_info(X_train)
    print(f"  Train set: {len(y_train)} samples, {total_features} features")

    # C sweep
    c_values = args.c_values or [args.C]
    verbose = 1 if len(c_values) == 1 else 0
    candidates = []

    for c_value in c_values:
        print(f"\nTraining classifier with C={c_value}")
        model = LogisticRegression(
            penalty=args.penalty,
            C=c_value,
            solver=args.solver,
            max_iter=1000,
            random_state=42,
            verbose=verbose,
        )
        model.fit(X_train, y_train)
        train_metrics = compute_metrics(model, X_train, y_train)
        print_metrics(train_metrics, "Training Set")

        test_metrics = None
        if prefer_test:
            test_metrics = compute_metrics(model, X_test, y_test)
            print_metrics(test_metrics, "Test Set")

        candidates.append(
            {
                "C": float(c_value),
                "model": model,
                "train_metrics": train_metrics,
                "test_metrics": test_metrics,
            }
        )

    # Select best
    best_candidate = max(
        candidates,
        key=lambda c: (
            select_candidate_score(c, args.selection_metric, prefer_test),
            -np.sum(c["model"].coef_[0] > 0),
        ),
    )

    best_positive = int(np.sum(best_candidate["model"].coef_[0] > 0))
    best_nonzero = int(np.sum(best_candidate["model"].coef_[0] != 0))
    selection_split = "test" if prefer_test else "train"
    print(
        f"\nSelected best C={best_candidate['C']} by {selection_split} "
        f"{args.selection_metric}: "
        f"{select_candidate_score(best_candidate, args.selection_metric, prefer_test):.4f}"
    )
    print(
        f"SAE features: {best_positive} positive, {best_nonzero} nonzero "
        f"out of {total_features}"
    )

    # Save model
    if args.save_model:
        ensure_parent_dir(args.save_model)
        joblib.dump(best_candidate["model"], args.save_model)
        print(f"Model saved to {args.save_model}")

    # Save metrics
    if args.metrics_out:
        ensure_parent_dir(args.metrics_out)

        # Identify top features
        coef = best_candidate["model"].coef_[0]
        top_positive_idx = np.argsort(coef)[::-1][: min(50, best_positive)]
        top_positive_idx = top_positive_idx[coef[top_positive_idx] > 0]

        payload = {
            "feature_type": "sae",
            "model_path": args.model_path,
            "train_mode": args.train_mode,
            "penalty": args.penalty,
            "solver": args.solver,
            "selection_metric": args.selection_metric,
            "selection_split": selection_split,
            "total_sae_features": total_features,
            "best": summarize_sae_candidate(
                best_candidate, args.selection_metric, prefer_test
            ),
            "top_positive_features": [
                {
                    "flat_idx": int(idx),
                    "weight": float(coef[idx]),
                }
                for idx in top_positive_idx
            ],
            "candidates": [
                summarize_sae_candidate(c, args.selection_metric, prefer_test)
                for c in candidates
            ],
        }
        if prefer_test:
            payload["evaluation"] = evaluate_model_with_uncertainty(
                best_candidate["model"],
                X_test,
                y_test,
                test_qids,
            )
        with open(args.metrics_out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"Saved metrics to {args.metrics_out}")


if __name__ == "__main__":
    main()
