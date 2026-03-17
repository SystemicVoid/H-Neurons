import argparse
import json
import os

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from tqdm import tqdm
from transformers import AutoConfig

from uncertainty import (
    DEFAULT_BOOTSTRAP_RESAMPLES,
    DEFAULT_BOOTSTRAP_SEED,
    stratified_bootstrap_classifier_metrics,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train or evaluate an H-Neuron detector with optional C sweeps."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path or HF id used to load the model config for neuron counts.",
    )

    parser.add_argument("--train_ids", type=str, help="Path to train_qids.json")
    parser.add_argument(
        "--train_ans_acts",
        type=str,
        help="Directory of answer-token activations for training",
    )
    parser.add_argument(
        "--train_other_acts",
        type=str,
        help="Directory of non-answer activations (required for 3-vs-1)",
    )

    parser.add_argument("--test_ids", type=str, help="Path to held-out qids.json")
    parser.add_argument(
        "--test_acts",
        type=str,
        help="Directory of held-out activations (usually answer tokens)",
    )

    parser.add_argument("--save_model", type=str, default="models/detector.pkl")
    parser.add_argument(
        "--load_model",
        type=str,
        help="Load a pre-trained model for evaluation instead of training",
    )
    parser.add_argument(
        "--metrics_out",
        type=str,
        default=None,
        help="Optional JSON path for sweep metrics and selected checkpoint metadata.",
    )

    parser.add_argument(
        "--train_mode",
        type=str,
        choices=["1-vs-1", "3-vs-1"],
        default="3-vs-1",
    )
    parser.add_argument("--penalty", type=str, choices=["l1", "l2"], default="l1")
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument(
        "--c_values",
        type=float,
        nargs="+",
        default=None,
        help="Optional list of C values to sweep. Falls back to --C when omitted.",
    )
    parser.add_argument(
        "--selection_metric",
        type=str,
        choices=["accuracy", "precision", "recall", "f1", "auroc"],
        default="auroc",
        help="Metric used to pick the best C during a sweep.",
    )
    parser.add_argument("--solver", type=str, default="liblinear")

    return parser.parse_args()


def load_data(
    ids_path, ans_acts_dir, other_acts_dir=None, mode="1-vs-1", return_qids=False
):
    """
    Flexible data loader.
    1-vs-1: False answer tokens (label 1) vs true answer tokens (label 0).
    3-vs-1: False answer tokens (label 1) vs
            (true answer + true other + false other) (label 0).
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
            raise ValueError("train_other_acts directory is required for 3-vs-1 mode.")

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


def compute_metrics(model, X, y):
    preds = model.predict(X)
    probs = model.predict_proba(X)[:, 1]

    accuracy = accuracy_score(y, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y, preds, average="binary", zero_division=0
    )
    auroc = roc_auc_score(y, probs)

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auroc": float(auroc),
        "n_examples": int(len(y)),
        "n_positive": int(np.sum(y == 1)),
        "n_negative": int(np.sum(y == 0)),
    }


def evaluate_model_with_uncertainty(model, X, y, qids):
    preds = model.predict(X)
    probs = model.predict_proba(X)[:, 1]
    metrics = compute_metrics(model, X, y)
    ci_summary = stratified_bootstrap_classifier_metrics(
        y,
        probs,
        n_resamples=DEFAULT_BOOTSTRAP_RESAMPLES,
        seed=DEFAULT_BOOTSTRAP_SEED,
    )
    confusion = {
        "tp": int(np.sum((y == 1) & (preds == 1))),
        "tn": int(np.sum((y == 0) & (preds == 0))),
        "fp": int(np.sum((y == 0) & (preds == 1))),
        "fn": int(np.sum((y == 1) & (preds == 0))),
    }
    examples = [
        {
            "qid": qid,
            "label": int(label),
            "prediction": int(pred),
            "probability": float(prob),
            "correct": bool(label == pred),
        }
        for qid, label, pred, prob in zip(qids, y, preds, probs, strict=True)
    ]
    return {
        "n_examples": int(len(y)),
        "n_positive": int(np.sum(y == 1)),
        "n_negative": int(np.sum(y == 0)),
        "confusion_matrix": confusion,
        "metrics": ci_summary["metrics"],
        "bootstrap": ci_summary["bootstrap"],
        "examples": examples,
        "point_metrics": metrics,
    }


def print_metrics(metrics, dataset_name="Test"):
    print(f"\n--- Results: {dataset_name} ---")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1']:.4f}")
    print(f"AUROC:     {metrics['auroc']:.4f}")


def get_total_neurons(model_path):
    config = AutoConfig.from_pretrained(model_path)
    if hasattr(config, "intermediate_size"):
        intermediate_size = config.intermediate_size
    else:
        intermediate_size = config.text_config.intermediate_size

    if hasattr(config, "num_hidden_layers"):
        num_hidden_layers = config.num_hidden_layers
    else:
        num_hidden_layers = config.text_config.num_hidden_layers

    return int(intermediate_size * num_hidden_layers)


def fit_model(args, X_train, y_train, c_value, verbose):
    model = LogisticRegression(
        penalty=args.penalty,
        C=c_value,
        solver=args.solver,
        max_iter=1000,
        random_state=42,
        verbose=verbose,
    )
    model.fit(X_train, y_train)
    return model


def ensure_parent_dir(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def select_candidate_score(candidate, metric_name, prefer_test):
    dataset = candidate["test_metrics"] if prefer_test else candidate["train_metrics"]
    return dataset[metric_name]


def summarize_candidate(candidate, total_neurons, selection_metric, prefer_test):
    selected = int(np.sum(candidate["model"].coef_[0] > 0))
    ratio_per_mille = (selected / total_neurons * 1000) if total_neurons else 0.0
    score = select_candidate_score(candidate, selection_metric, prefer_test)
    return {
        "C": candidate["C"],
        "score": float(score),
        "selected_h_neurons": selected,
        "selected_ratio_per_mille": float(ratio_per_mille),
        "train_metrics": candidate["train_metrics"],
        "test_metrics": candidate["test_metrics"],
    }


def main():
    args = parse_args()
    total_neurons = get_total_neurons(args.model_path)
    prefer_test = bool(args.test_ids and args.test_acts)

    if args.load_model and args.c_values:
        raise ValueError("--c_values cannot be used with --load_model.")

    X_test = y_test = test_qids = None
    if prefer_test:
        print(f"Loading held-out evaluation data from {args.test_ids}...")
        X_test, y_test, test_qids = load_data(
            args.test_ids,
            args.test_acts,
            mode="1-vs-1",
            return_qids=True,
        )

    if args.load_model:
        print(f"Loading pre-trained model: {args.load_model}")
        model = joblib.load(args.load_model)
        selected = int(np.sum(model.coef_[0] > 0))
        ratio_per_mille = (selected / total_neurons * 1000) if total_neurons else 0.0
        print(
            "Loaded model selects "
            f"{selected} H-Neurons ({ratio_per_mille:.4f} per mille of FFN neurons)."
        )
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
                "model_path": args.model_path,
                "loaded_model_path": args.load_model,
                "selection_split": "pretrained_load",
                "total_ffn_neurons": total_neurons,
                "selected_h_neurons": selected,
                "selected_ratio_per_mille": float(ratio_per_mille),
                "evaluation": evaluation,
            }
            with open(args.metrics_out, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            print(f"Saved metrics to {args.metrics_out}")
        return

    if not (args.train_ids and args.train_ans_acts):
        print("Please provide --load_model OR (--train_ids AND --train_ans_acts)")
        return

    print(f"Training in {args.train_mode} mode...")
    X_train, y_train = load_data(
        args.train_ids,
        args.train_ans_acts,
        other_acts_dir=args.train_other_acts,
        mode=args.train_mode,
    )

    c_values = args.c_values or [args.C]
    verbose = 1 if len(c_values) == 1 else 0
    candidates = []

    for c_value in c_values:
        print(f"\nTraining classifier with C={c_value}")
        model = fit_model(args, X_train, y_train, c_value, verbose)
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

    best_candidate = max(
        candidates,
        key=lambda candidate: (
            select_candidate_score(candidate, args.selection_metric, prefer_test),
            -np.sum(candidate["model"].coef_[0] > 0),
        ),
    )

    best_selected = int(np.sum(best_candidate["model"].coef_[0] > 0))
    best_ratio_per_mille = (
        best_selected / total_neurons * 1000 if total_neurons else 0.0
    )
    selection_split = "test" if prefer_test else "train"
    print(
        "\nSelected best checkpoint with "
        f"C={best_candidate['C']} by {selection_split} {args.selection_metric}: "
        f"{select_candidate_score(best_candidate, args.selection_metric, prefer_test):.4f}"
    )
    print(
        "Best model identified "
        f"{best_selected} potential H-Neurons "
        f"({best_ratio_per_mille:.4f} per mille of FFN neurons)."
    )

    if args.save_model:
        ensure_parent_dir(args.save_model)
        joblib.dump(best_candidate["model"], args.save_model)
        print(f"Model saved to {args.save_model}")

    if args.metrics_out:
        ensure_parent_dir(args.metrics_out)
        payload = {
            "model_path": args.model_path,
            "train_mode": args.train_mode,
            "penalty": args.penalty,
            "solver": args.solver,
            "selection_metric": args.selection_metric,
            "selection_split": selection_split,
            "total_ffn_neurons": total_neurons,
            "best": summarize_candidate(
                best_candidate,
                total_neurons,
                args.selection_metric,
                prefer_test,
            ),
            "candidates": [
                summarize_candidate(
                    candidate,
                    total_neurons,
                    args.selection_metric,
                    prefer_test,
                )
                for candidate in candidates
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
