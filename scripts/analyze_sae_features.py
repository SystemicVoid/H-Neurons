"""Analyze top SAE features from the hallucination classifier.

For each positive-weight SAE feature, compute correlations with response
length to flag potential verbosity confounds, and extract max-activating
samples for manual interpretability review.

Usage:
    uv run python scripts/analyze_sae_features.py \
        --classifier_summary data/gemma3_4b/pipeline/classifier_sae_summary.json \
        --classifier_path models/sae_detector.pkl \
        --extraction_dir data/gemma3_4b/pipeline/activations_sae_hlayers_16k_small/answer_tokens \
        --samples_jsonl data/gemma3_4b/pipeline/answer_tokens.jsonl \
        --ids_path data/gemma3_4b/pipeline/test_qids_disjoint.json \
        --output data/gemma3_4b/pipeline/sae_feature_analysis.json
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
from scipy import stats

from classifier_sae import load_extraction_metadata
from intervene_sae import get_positive_sae_features_from_classifier


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze top SAE features for interpretability and confounds."
    )
    parser.add_argument(
        "--classifier_summary",
        type=str,
        required=True,
        help="Path to classifier_sae_summary.json",
    )
    parser.add_argument(
        "--classifier_path",
        type=str,
        default=None,
        help=(
            "Path to the trained SAE classifier .pkl. If omitted, the script "
            "will look for classifier_path in the summary JSON."
        ),
    )
    parser.add_argument(
        "--extraction_dir",
        type=str,
        required=True,
        help="Path to SAE answer_tokens extraction directory",
    )
    parser.add_argument(
        "--samples_jsonl",
        type=str,
        required=True,
        help="Path to answer_tokens.jsonl with response data",
    )
    parser.add_argument(
        "--ids_path",
        type=str,
        required=True,
        help="Path to test_qids_disjoint.json (or train_qids.json)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/gemma3_4b/pipeline/sae_feature_analysis.json",
        help="Output JSON path",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Number of top features for max-activating sample extraction",
    )
    parser.add_argument(
        "--max_activating_n",
        type=int,
        default=10,
        help="Number of max-activating samples per feature",
    )
    parser.add_argument(
        "--confound_threshold",
        type=float,
        default=0.3,
        help="Pearson |r| threshold to flag verbosity confounds",
    )
    return parser.parse_args()


def load_samples_jsonl(path):
    """Load answer_tokens.jsonl into a dict keyed by qid."""
    samples = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            for qid, data in row.items():
                samples[qid] = data
    return samples


def load_activations_for_ids(extraction_dir, qids):
    """Load SAE activation vectors for a list of qids.

    Returns:
        activations: np.ndarray of shape [n_samples, n_layers, d_sae]
        valid_qids: list of qids that had matching .npy files
    """
    arrays = []
    valid_qids = []
    for qid in qids:
        path = os.path.join(extraction_dir, f"act_{qid}.npy")
        if os.path.exists(path):
            arr = np.load(path)  # shape: [n_layers, d_sae]
            arrays.append(arr)
            valid_qids.append(qid)
    if not arrays:
        raise ValueError(f"No activation files found in {extraction_dir}")
    return np.stack(arrays), valid_qids


def compute_length_features(samples, qids):
    """Compute response length features for correlation analysis."""
    token_counts = []
    char_lengths = []
    question_lengths = []
    for qid in qids:
        sample = samples[qid]
        n_tokens = len(sample.get("answer_tokens", []))
        response_text = sample.get("response", "")
        question_text = sample.get("question", "")
        token_counts.append(n_tokens)
        char_lengths.append(len(response_text))
        question_lengths.append(len(question_text))
    return {
        "response_token_count": np.array(token_counts, dtype=float),
        "response_char_length": np.array(char_lengths, dtype=float),
        "question_char_length": np.array(question_lengths, dtype=float),
    }


def analyze_feature(
    feature_activations,
    length_features,
    confound_threshold,
):
    """Compute correlations between a single feature and length metrics."""
    correlations = {}
    for name, lengths in length_features.items():
        if np.std(feature_activations) < 1e-10 or np.std(lengths) < 1e-10:
            correlations[name] = {"r": 0.0, "p": 1.0}
            continue
        r, p = stats.pearsonr(feature_activations, lengths)
        correlations[name] = {"r": round(float(r), 4), "p": round(float(p), 6)}

    is_confound = any(abs(c["r"]) > confound_threshold for c in correlations.values())
    return correlations, is_confound


def extract_max_activating(feature_activations, qids, samples, n=10):
    """Find the n samples with highest activation for a feature."""
    top_indices = np.argsort(feature_activations)[::-1][:n]
    results = []
    for idx in top_indices:
        qid = qids[idx]
        sample = samples.get(qid, {})
        results.append(
            {
                "qid": qid,
                "activation": round(float(feature_activations[idx]), 4),
                "question": sample.get("question", "")[:200],
                "response": sample.get("response", "")[:200],
                "judge": sample.get("judge", ""),
                "n_answer_tokens": len(sample.get("answer_tokens", [])),
            }
        )
    return results


def resolve_classifier_path(explicit_path, summary, summary_path):
    """Resolve the classifier path needed to inspect the full coefficient vector."""
    if explicit_path:
        return str(Path(explicit_path).expanduser().resolve())

    classifier_path = summary.get("classifier_path")
    if not classifier_path:
        raise ValueError(
            "Full SAE analysis requires the trained classifier weights. "
            "Pass --classifier_path or retrain/export classifier_sae_summary.json "
            "with classifier_path metadata."
        )

    candidate_path = Path(classifier_path).expanduser()
    if candidate_path.is_absolute():
        return str(candidate_path.resolve())

    summary_dir = Path(summary_path).expanduser().resolve().parent
    summary_relative = (summary_dir / candidate_path).resolve()
    if summary_relative.exists():
        return str(summary_relative)

    # Backward-compatibility for summaries that stored repo-root-relative paths.
    repo_root = Path(__file__).resolve().parent.parent
    repo_relative = (repo_root / candidate_path).resolve()
    if repo_relative.exists():
        return str(repo_relative)

    raise FileNotFoundError(
        "Could not resolve classifier_path from the summary. Tried "
        f"{summary_relative} and {repo_relative}."
    )


def main():
    args = parse_args()

    # Load classifier summary
    with open(args.classifier_summary, "r", encoding="utf-8") as f:
        summary = json.load(f)

    extraction_metadata = summary.get("extraction_metadata")
    if extraction_metadata is None:
        extraction_metadata = load_extraction_metadata(args.extraction_dir)
    else:
        load_extraction_metadata(args.extraction_dir)  # validate metadata exists

    classifier_path = resolve_classifier_path(
        args.classifier_path,
        summary,
        args.classifier_summary,
    )
    positive_features = get_positive_sae_features_from_classifier(
        classifier_path,
        layer_indices=extraction_metadata["layer_indices"],
        d_sae=extraction_metadata["d_sae"],
    )
    if not positive_features:
        print("No positive classifier-weight SAE features found.")
        return

    summary_top_features = summary.get("top_positive_features", [])
    if len(summary_top_features) != len(positive_features):
        print(
            "Summary top_positive_features is truncated: "
            f"{len(summary_top_features)} listed vs {len(positive_features)} "
            "positive weights in classifier. Using full classifier coefficients."
        )

    # Load qids
    with open(args.ids_path, "r", encoding="utf-8") as f:
        id_map = json.load(f)
    all_qids = id_map.get("f", []) + id_map.get("t", [])

    print(f"Loading activations for {len(all_qids)} samples...")
    activations, valid_qids = load_activations_for_ids(args.extraction_dir, all_qids)
    print(f"  Loaded {len(valid_qids)} samples, shape {activations.shape}")

    # Build labels
    false_set = set(id_map.get("f", []))
    labels = np.array([1 if qid in false_set else 0 for qid in valid_qids])

    # Load response samples for length analysis
    print(f"Loading response data from {args.samples_jsonl}...")
    samples = load_samples_jsonl(args.samples_jsonl)
    length_features = compute_length_features(samples, valid_qids)

    # Flatten activations for indexing: [n_samples, n_layers * d_sae]
    n_samples = activations.shape[0]
    flat_acts = activations.reshape(n_samples, -1)

    # Analyze all positive features
    print(f"Analyzing {len(positive_features)} positive features...")
    feature_analyses = []
    n_confounds = 0

    for feat in positive_features:
        flat_idx = feat["flat_idx"]
        layer = feat["layer"]
        feature_idx = feat["feature"]
        weight = feat["weight"]

        feat_acts = flat_acts[:, flat_idx]
        correlations, is_confound = analyze_feature(
            feat_acts, length_features, args.confound_threshold
        )
        if is_confound:
            n_confounds += 1

        # Activation statistics by label
        false_acts = feat_acts[labels == 1]
        true_acts = feat_acts[labels == 0]

        analysis = {
            "layer": layer,
            "feature": feature_idx,
            "flat_idx": flat_idx,
            "weight": weight,
            "correlations": correlations,
            "is_verbosity_confound": is_confound,
            "activation_stats": {
                "false_mean": round(float(np.mean(false_acts)), 4)
                if len(false_acts) > 0
                else None,
                "true_mean": round(float(np.mean(true_acts)), 4)
                if len(true_acts) > 0
                else None,
                "false_std": round(float(np.std(false_acts)), 4)
                if len(false_acts) > 0
                else None,
                "true_std": round(float(np.std(true_acts)), 4)
                if len(true_acts) > 0
                else None,
                "separation": round(float(np.mean(false_acts) - np.mean(true_acts)), 4)
                if len(false_acts) > 0 and len(true_acts) > 0
                else None,
            },
        }
        feature_analyses.append(analysis)

    # Max-activating samples for top_k features
    print(f"Extracting max-activating samples for top {args.top_k} features...")
    max_activating = {}
    for feat in positive_features[: args.top_k]:
        flat_idx = feat["flat_idx"]
        feat_acts = flat_acts[:, flat_idx]
        key = f"L{feat['layer']}:F{feat['feature']}"
        max_activating[key] = extract_max_activating(
            feat_acts, valid_qids, samples, n=args.max_activating_n
        )

    # Summary
    output = {
        "classifier_summary_path": args.classifier_summary,
        "classifier_path": classifier_path,
        "extraction_dir": args.extraction_dir,
        "ids_path": args.ids_path,
        "feature_selection_method": "all_positive_classifier_weights",
        "summary_top_positive_feature_count": len(summary_top_features),
        "n_samples": len(valid_qids),
        "n_false": int(np.sum(labels == 1)),
        "n_true": int(np.sum(labels == 0)),
        "n_positive_features_analyzed": len(feature_analyses),
        "n_verbosity_confounds": n_confounds,
        "confound_threshold": args.confound_threshold,
        "feature_analyses": feature_analyses,
        "max_activating_samples": max_activating,
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved analysis to {args.output}")
    print(f"  {len(feature_analyses)} features analyzed")
    print(
        f"  {n_confounds} flagged as verbosity confounds (|r| > {args.confound_threshold})"
    )
    if feature_analyses:
        top = feature_analyses[0]
        print(
            f"  Top feature: L{top['layer']}:F{top['feature']} (weight={top['weight']:.4f})"
        )


if __name__ == "__main__":
    main()
