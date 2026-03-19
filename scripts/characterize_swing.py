#!/usr/bin/env python3
"""Characterize the 138 swing samples in the FaithEval intervention experiment.

Classifies the tri-modal population (always-compliant, never-compliant, swing),
analyzes swing trajectory subtypes (R→C, C→R, non-monotonic), and runs
structural/topic proxies to understand what makes swing samples different.

Usage:
    uv run python scripts/characterize_swing.py
    uv run python scripts/characterize_swing.py --use-llm --max-llm-samples 5
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from uncertainty import (
        DEFAULT_BOOTSTRAP_RESAMPLES,
        DEFAULT_BOOTSTRAP_SEED,
        build_rate_summary,
        stratified_bootstrap_classifier_metrics,
    )
except ModuleNotFoundError:
    from scripts.uncertainty import (
        DEFAULT_BOOTSTRAP_RESAMPLES,
        DEFAULT_BOOTSTRAP_SEED,
        build_rate_summary,
        stratified_bootstrap_classifier_metrics,
    )

ALPHAS = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

# Color scheme from plot_intervention.py
COLORS = {
    "teal": "#1D9E75",
    "purple": "#7F77DD",
    "coral": "#D85A30",
}
POPULATION_COLORS = {
    "always_compliant": COLORS["teal"],
    "never_compliant": COLORS["coral"],
    "swing": COLORS["purple"],
}
SUBTYPE_COLORS = {
    "R→C": "#9B59B6",
    "C→R": "#3498DB",
    "non-monotonic": "#95A5A6",
}
STRUCTURAL_NUMERIC_FEATURES = [
    "question_length",
    "context_length",
    "word_overlap",
    "num_options",
]
STRUCTURAL_CATEGORICAL_FEATURES = ["source", "topic"]
STRUCTURAL_FEATURE_SETS: dict[str, dict[str, list[str]]] = {
    "all_ex_ante": {
        "numeric": STRUCTURAL_NUMERIC_FEATURES,
        "categorical": STRUCTURAL_CATEGORICAL_FEATURES,
    },
    "structure_only": {
        "numeric": STRUCTURAL_NUMERIC_FEATURES,
        "categorical": ["topic"],
    },
    "source_only": {
        "numeric": [],
        "categorical": ["source"],
    },
}
STRUCTURAL_TASKS: dict[str, dict[str, str]] = {
    "swing_vs_non_swing": {
        "subset_population": "all",
        "positive_label": "swing",
        "description": "Predict whether an item belongs to the swing population.",
    },
    "r_to_c_vs_other_swing": {
        "subset_population": "swing",
        "positive_label": "R→C",
        "description": "Within swing samples, predict whether the subtype is R→C rather than C→R or non-monotonic.",
    },
}
STOP_WORDS = {
    "the",
    "a",
    "an",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "do",
    "does",
    "did",
    "will",
    "would",
    "shall",
    "should",
    "may",
    "might",
    "must",
    "can",
    "could",
    "of",
    "in",
    "to",
    "for",
    "with",
    "on",
    "at",
    "from",
    "by",
    "as",
    "or",
    "and",
    "but",
    "if",
    "not",
    "no",
    "so",
    "it",
    "its",
    "that",
    "this",
    "these",
    "those",
    "which",
    "what",
    "who",
    "whom",
    "how",
    "when",
    "where",
    "why",
    "all",
    "each",
    "every",
    "both",
    "few",
    "more",
    "most",
    "other",
    "some",
    "such",
    "than",
    "too",
    "very",
}
VALID_MC_ANSWERS = frozenset("ABCDEFGH")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_hf_faitheval() -> dict[str, dict[str, Any]]:
    """Load FaithEval counterfactual dataset from HuggingFace, keyed by ID."""
    from datasets import load_dataset

    ds = load_dataset("Salesforce/FaithEval-counterfactual-v1.0", split="test")
    by_id: dict[str, dict[str, Any]] = {}
    for row in ds:
        choices = row["choices"]
        labels = choices["label"]
        texts = choices["text"]
        by_id[row["id"]] = {
            "context": row["context"],
            "question": row["question"],
            "choices_labels": labels,
            "choices_texts": texts,
            "answer_key": row["answerKey"],
            "num_options": row["num of options"],
        }
    return by_id


def load_all_alphas(base_dir: Path) -> dict[float, dict[str, dict[str, Any]]]:
    """Load all alpha JSONL files, returning {alpha: {id: row}}."""
    rows_by_alpha: dict[float, dict[str, dict[str, Any]]] = {}
    for alpha in ALPHAS:
        path = base_dir / f"alpha_{alpha:.1f}.jsonl"
        rows_by_alpha[alpha] = {row["id"]: row for row in load_jsonl(path)}
    return rows_by_alpha


def classify_populations(
    rows_by_alpha: dict[float, dict[str, dict[str, Any]]],
) -> tuple[
    dict[str, list[bool]],
    list[str],
    list[str],
    list[str],
]:
    """Classify samples into always-compliant, never-compliant, swing.

    Returns (trajectories, always_ids, never_ids, swing_ids).
    """
    reference_ids = sorted(rows_by_alpha[ALPHAS[0]])
    trajectories: dict[str, list[bool]] = {}
    for sample_id in reference_ids:
        trajectories[sample_id] = [
            bool(rows_by_alpha[alpha][sample_id]["compliance"]) for alpha in ALPHAS
        ]

    always_ids = [sid for sid, vals in trajectories.items() if all(vals)]
    never_ids = [sid for sid, vals in trajectories.items() if not any(vals)]
    swing_ids = [
        sid for sid, vals in trajectories.items() if any(vals) and not all(vals)
    ]

    return trajectories, always_ids, never_ids, swing_ids


def classify_swing_subtypes(
    trajectories: dict[str, list[bool]], swing_ids: list[str]
) -> dict[str, str]:
    """Classify swing samples into R→C, C→R, or non-monotonic."""
    subtypes: dict[str, str] = {}
    for sid in swing_ids:
        traj = trajectories[sid]
        first = traj[0]
        last = traj[-1]
        # Check monotonicity: once it switches, does it stay?
        is_monotonic = True
        for i in range(1, len(traj)):
            if traj[i] != traj[i - 1]:
                # Found a transition; check remaining are same as traj[i]
                if any(t != traj[i] for t in traj[i + 1 :]):
                    is_monotonic = False
                break

        if not is_monotonic:
            subtypes[sid] = "non-monotonic"
        elif not first and last:
            subtypes[sid] = "R→C"  # Resistant → Compliant (knowledge override)
        elif first and not last:
            subtypes[sid] = "C→R"  # Compliant → Resistant (uncertainty resolution)
        else:
            # Monotonic transition but same start/end — shouldn't happen for swing
            subtypes[sid] = "non-monotonic"
    return subtypes


def find_transition_alpha(trajectories: dict[str, list[bool]], sample_id: str) -> float:
    """Find the first α where behavior changes."""
    traj = trajectories[sample_id]
    for i in range(1, len(traj)):
        if traj[i] != traj[i - 1]:
            return ALPHAS[i]
    return ALPHAS[-1]


def compute_word_overlap(question: str, context: str) -> float:
    """Fraction of non-stopword question tokens that also appear in the context."""
    q_words = set(re.findall(r"\w+", question.lower())) - STOP_WORDS
    c_words = (
        set(re.findall(r"\w+", context.lower())) - STOP_WORDS if context else set()
    )
    return len(q_words & c_words) / len(q_words) if q_words else 0.0


# ---------------------------------------------------------------------------
# Source dataset extraction
# ---------------------------------------------------------------------------


def extract_source(sample_id: str) -> str:
    """Extract source dataset from sample ID prefix."""
    # Known prefixes from ARC / AI2 datasets
    known_prefixes = [
        "Mercury_SC",
        "Mercury_LBS",
        "Mercury",
        "MCAS",
        "NYSEDREGENTS",
        "TIMSS",
        "MDSA",
        "ACTAAP",
        "AKDE&ED",
        "LEAP",
        "NCEOGA",
        "VASoL",
        "MEA",
        "AIMS",
        "MSA",
        "OHAT",
        "TAKS",
        "MEAP",
        "NAEP",
        "WASL",
        "FCAT",
    ]
    for prefix in known_prefixes:
        if sample_id.startswith(prefix + "_"):
            return prefix
    # Handle CSZ-style IDs (no separator: CSZ30169)
    if sample_id.startswith("CSZ"):
        return "CSZ"
    return "other"


# ---------------------------------------------------------------------------
# Topic classification (keyword-based)
# ---------------------------------------------------------------------------

TOPIC_KEYWORDS: dict[str, list[str]] = {
    "biology": [
        "cell",
        "organ",
        "plant",
        "animal",
        "species",
        "dna",
        "gene",
        "protein",
        "photosynthesis",
        "ecosystem",
        "food chain",
        "food web",
        "predator",
        "prey",
        "habitat",
        "population",
        "bacteria",
        "virus",
        "disease",
        "blood",
        "heart",
        "lung",
        "bone",
        "muscle",
        "digest",
        "nutrient",
        "evolution",
        "inherit",
        "trait",
        "offspring",
        "reproduce",
        "mitosis",
        "meiosis",
        "chromosome",
        "fossil",
        "extinct",
        "adaptation",
        "biome",
        "respiration",
        "ferment",
    ],
    "physics": [
        "force",
        "motion",
        "velocity",
        "speed",
        "acceleration",
        "gravity",
        "mass",
        "weight",
        "energy",
        "kinetic",
        "potential",
        "momentum",
        "friction",
        "magnet",
        "electric",
        "circuit",
        "current",
        "voltage",
        "wave",
        "frequency",
        "light",
        "sound",
        "reflect",
        "refract",
        "lens",
        "mirror",
        "heat",
        "temperature",
        "radiation",
        "conduct",
        "insulate",
        "newton",
        "joule",
        "watt",
    ],
    "earth_science": [
        "rock",
        "mineral",
        "soil",
        "erosion",
        "weather",
        "climate",
        "earthquake",
        "volcano",
        "plate",
        "tectonic",
        "fossil",
        "sediment",
        "layer",
        "atmosphere",
        "ocean",
        "tide",
        "moon",
        "sun",
        "star",
        "planet",
        "solar system",
        "orbit",
        "season",
        "rotation",
        "revolution",
        "water cycle",
        "evaporation",
        "condensation",
        "precipitation",
        "glacier",
        "continental",
        "crust",
        "mantle",
        "core",
    ],
    "chemistry": [
        "atom",
        "molecule",
        "element",
        "compound",
        "chemical",
        "reaction",
        "solution",
        "dissolve",
        "acid",
        "base",
        "ph",
        "periodic table",
        "metal",
        "nonmetal",
        "ion",
        "bond",
        "mixture",
        "pure substance",
        "gas",
        "liquid",
        "solid",
        "boiling",
        "melting",
        "freezing",
        "evaporat",
        "density",
        "matter",
        "property",
    ],
    "scientific_method": [
        "hypothesis",
        "experiment",
        "variable",
        "control",
        "data",
        "conclusion",
        "observation",
        "investigate",
        "procedure",
        "trial",
        "result",
        "evidence",
        "measure",
        "predict",
        "model",
        "scientific method",
        "test",
        "sample",
        "graph",
        "table",
    ],
}


def classify_topic(question: str) -> str:
    """Classify a question into a science topic using keyword matching."""
    q_lower = question.lower()
    scores: dict[str, int] = {topic: 0 for topic in TOPIC_KEYWORDS}
    for topic, keywords in TOPIC_KEYWORDS.items():
        for kw in keywords:
            if kw in q_lower:
                scores[topic] += 1
    best = max(scores, key=lambda t: scores[t])
    if scores[best] == 0:
        return "other"
    return best


def build_feature_table(
    df: pd.DataFrame,
    hf_data: dict[str, dict[str, Any]],
    standard_dir: Path | None,
) -> pd.DataFrame:
    """Build an auditable per-item feature table for descriptive and predictive analyses."""
    std_rows = {}
    if standard_dir and (standard_dir / "alpha_1.0.jsonl").exists():
        std_rows = {r["id"]: r for r in load_jsonl(standard_dir / "alpha_1.0.jsonl")}

    records = []
    for _, row in df.iterrows():
        hf = hf_data.get(row["id"], {})
        context = str(hf.get("context", ""))
        records.append(
            {
                "id": row["id"],
                "population": row["population"],
                "swing_subtype": row["swing_subtype"],
                "source": row["source"],
                "topic": row["topic"],
                "question_length": int(len(row["question"])),
                "context_length": int(len(context)),
                "word_overlap": round(
                    compute_word_overlap(row["question"], context), 4
                ),
                "num_options": int(hf.get("num_options", 0) or 0),
                "anti_compliance_response_length": int(len(row["response"])),
                "standard_response_length": int(
                    len(std_rows.get(row["id"], {}).get("response", ""))
                ),
            }
        )

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------


def wilson_ci(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score 95% CI for a binomial proportion."""
    if n == 0:
        return (0.0, 0.0)
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = z * math.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def kruskal_with_posthoc(
    groups: dict[str, list[float]],
) -> dict[str, Any]:
    """Kruskal-Wallis test with pairwise Mann-Whitney U posthoc (Bonferroni)."""
    group_names = sorted(groups)
    group_data = [groups[g] for g in group_names]
    # Filter out empty groups
    non_empty = [(n, d) for n, d in zip(group_names, group_data) if len(d) > 0]
    if len(non_empty) < 2:
        return {"kruskal_H": float("nan"), "kruskal_p": float("nan"), "posthoc": {}}

    h_stat, p_val = stats.kruskal(*[d for _, d in non_empty])
    result: dict[str, Any] = {"kruskal_H": float(h_stat), "kruskal_p": float(p_val)}

    # Pairwise posthoc
    n_comparisons = len(non_empty) * (len(non_empty) - 1) // 2
    posthoc: dict[str, dict[str, float]] = {}
    for i in range(len(non_empty)):
        for j in range(i + 1, len(non_empty)):
            name_i, data_i = non_empty[i]
            name_j, data_j = non_empty[j]
            u_stat, u_p = stats.mannwhitneyu(data_i, data_j, alternative="two-sided")
            key = f"{name_i}_vs_{name_j}"
            posthoc[key] = {
                "U": float(u_stat),
                "p": float(u_p),
                "p_bonferroni": float(min(1.0, u_p * n_comparisons)),
                "n1": len(data_i),
                "n2": len(data_j),
            }
    result["posthoc"] = posthoc
    return result


def chi_squared_test(
    contingency: pd.DataFrame,
) -> dict[str, Any]:
    """Chi-squared test of independence on a contingency table."""
    chi2, p_val, dof, expected = stats.chi2_contingency(contingency.values)
    # Cramér's V
    n = contingency.values.sum()
    k = min(contingency.shape) - 1
    v = math.sqrt(chi2 / (n * k)) if n * k > 0 else 0.0
    return {
        "chi2": float(chi2),
        "p": float(p_val),
        "dof": int(dof),
        "cramers_v": float(v),
    }


def mann_whitney_effect_size(x: list[float], y: list[float]) -> dict[str, Any]:
    """Mann-Whitney U with rank-biserial correlation as effect size."""
    if len(x) == 0 or len(y) == 0:
        return {"U": float("nan"), "p": float("nan"), "r": float("nan")}
    u_stat, p_val = stats.mannwhitneyu(x, y, alternative="two-sided")
    # Rank-biserial correlation: r = 1 - 2U/(n1*n2)
    r = 1 - 2 * u_stat / (len(x) * len(y))
    return {
        "U": float(u_stat),
        "p": float(p_val),
        "r": float(r),
        "n1": len(x),
        "n2": len(y),
    }


def build_structural_model(
    numeric_features: list[str],
    categorical_features: list[str],
) -> Pipeline:
    """Shared preprocessing + logistic baseline for held-out structural prediction."""
    transformers = []
    if numeric_features:
        transformers.append(
            (
                "num",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric_features,
            )
        )
    if categorical_features:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("one_hot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            )
        )
    if not transformers:
        raise ValueError("At least one feature is required for structural prediction")

    return Pipeline(
        [
            ("preprocess", ColumnTransformer(transformers=transformers)),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=42,
                ),
            ),
        ]
    )


def cross_validated_probabilities(
    feature_df: pd.DataFrame,
    labels: np.ndarray,
    numeric_features: list[str],
    categorical_features: list[str],
    *,
    n_splits: int = 5,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Return out-of-fold probabilities and hard predictions for a fixed feature set."""
    X = feature_df[numeric_features + categorical_features].copy()
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    probabilities = np.zeros(len(feature_df), dtype=float)

    for train_idx, test_idx in cv.split(X, labels):
        model = build_structural_model(numeric_features, categorical_features)
        model.fit(X.iloc[train_idx], labels[train_idx])
        probabilities[test_idx] = model.predict_proba(X.iloc[test_idx])[:, 1]

    predictions = (probabilities >= 0.5).astype(int)
    return probabilities, predictions


def permutation_test_classifier_metrics(
    feature_df: pd.DataFrame,
    labels: np.ndarray,
    numeric_features: list[str],
    categorical_features: list[str],
    *,
    observed_auroc: float,
    observed_balanced_accuracy: float,
    n_splits: int = 5,
    n_resamples: int = 100,
    seed: int = 42,
) -> dict[str, Any]:
    """Permutation test for held-out AUROC and balanced accuracy."""
    rng = np.random.default_rng(seed)
    auroc_samples = np.empty(n_resamples, dtype=float)
    balanced_accuracy_samples = np.empty(n_resamples, dtype=float)

    for sample_idx in range(n_resamples):
        permuted = rng.permutation(labels)
        probs, preds = cross_validated_probabilities(
            feature_df,
            permuted,
            numeric_features,
            categorical_features,
            n_splits=n_splits,
            seed=seed + sample_idx + 1,
        )
        auroc_samples[sample_idx] = float(roc_auc_score(permuted, probs))
        balanced_accuracy_samples[sample_idx] = float(
            balanced_accuracy_score(permuted, preds)
        )

    auroc_p = (1 + int(np.sum(auroc_samples >= observed_auroc))) / (n_resamples + 1)
    balanced_accuracy_p = (
        1 + int(np.sum(balanced_accuracy_samples >= observed_balanced_accuracy))
    ) / (n_resamples + 1)

    return {
        "n_resamples": int(n_resamples),
        "seed": int(seed),
        "metrics": {
            "auroc": {
                "p_value": float(auroc_p),
                "null_mean": float(np.mean(auroc_samples)),
                "null_std": float(np.std(auroc_samples)),
            },
            "balanced_accuracy": {
                "p_value": float(balanced_accuracy_p),
                "null_mean": float(np.mean(balanced_accuracy_samples)),
                "null_std": float(np.std(balanced_accuracy_samples)),
            },
        },
    }


def evaluate_prediction_feature_set(
    feature_df: pd.DataFrame,
    labels: np.ndarray,
    numeric_features: list[str],
    categorical_features: list[str],
    *,
    n_splits: int = 5,
    bootstrap_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    permutation_resamples: int = 100,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Evaluate one structural feature set with held-out metrics and permutation p-values."""
    probabilities, predictions = cross_validated_probabilities(
        feature_df,
        labels,
        numeric_features,
        categorical_features,
        n_splits=n_splits,
        seed=seed,
    )
    bootstrap = stratified_bootstrap_classifier_metrics(
        labels,
        probabilities,
        n_resamples=bootstrap_resamples,
        seed=seed,
    )
    permutation = permutation_test_classifier_metrics(
        feature_df,
        labels,
        numeric_features,
        categorical_features,
        observed_auroc=bootstrap["metrics"]["auroc"]["estimate"],
        observed_balanced_accuracy=bootstrap["metrics"]["balanced_accuracy"][
            "estimate"
        ],
        n_splits=n_splits,
        n_resamples=permutation_resamples,
        seed=seed,
    )

    return {
        "n_samples": int(len(labels)),
        "n_positive": int(np.sum(labels == 1)),
        "n_negative": int(np.sum(labels == 0)),
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "cv": {
            "n_splits": int(n_splits),
            "shuffle": True,
            "seed": int(seed),
            "prediction_mode": "out_of_fold",
        },
        "metrics": bootstrap["metrics"],
        "bootstrap": bootstrap["bootstrap"],
        "permutation_test": permutation,
        "confusion_matrix": {
            "tp": int(np.sum((labels == 1) & (predictions == 1))),
            "tn": int(np.sum((labels == 0) & (predictions == 0))),
            "fp": int(np.sum((labels == 0) & (predictions == 1))),
            "fn": int(np.sum((labels == 1) & (predictions == 0))),
        },
    }


def analyze_structural_predictability(
    feature_df: pd.DataFrame,
    *,
    n_splits: int = 5,
    bootstrap_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    permutation_resamples: int = 100,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Held-out prediction tests for whether surface features classify swing behavior."""
    tasks: dict[str, Any] = {}

    for task_name, task_spec in STRUCTURAL_TASKS.items():
        if task_spec["subset_population"] == "swing":
            task_df = feature_df[feature_df["population"] == "swing"].reset_index(
                drop=True
            )
            labels = (task_df["swing_subtype"] == task_spec["positive_label"]).astype(
                int
            )
        else:
            task_df = feature_df.reset_index(drop=True)
            labels = (task_df["population"] == task_spec["positive_label"]).astype(int)

        feature_set_results = {
            name: evaluate_prediction_feature_set(
                task_df,
                labels.to_numpy(),
                spec["numeric"],
                spec["categorical"],
                n_splits=n_splits,
                bootstrap_resamples=bootstrap_resamples,
                permutation_resamples=permutation_resamples,
                seed=seed,
            )
            for name, spec in STRUCTURAL_FEATURE_SETS.items()
        }

        tasks[task_name] = {
            "description": task_spec["description"],
            "positive_label": task_spec["positive_label"],
            "subset_population": task_spec["subset_population"],
            "feature_sets": feature_set_results,
        }

    return {
        "feature_sets": STRUCTURAL_FEATURE_SETS,
        "tasks": tasks,
    }


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------


def analyze_structural_proxies(
    feature_df: pd.DataFrame,
) -> dict[str, Any]:
    """Analyze question length, context length, number of options."""
    results: dict[str, Any] = {}

    # Question length
    q_lens = {
        pop: feature_df[feature_df["population"] == pop]["question_length"].tolist()
        for pop in ["always_compliant", "never_compliant", "swing"]
    }
    results["question_length"] = {
        "stats": {
            pop: {
                "mean": float(np.mean(vals)),
                "median": float(np.median(vals)),
                "std": float(np.std(vals)),
                "n": len(vals),
            }
            for pop, vals in q_lens.items()
        },
        "test": kruskal_with_posthoc(q_lens),
    }

    # Context length
    c_lens = {
        pop: feature_df[feature_df["population"] == pop]["context_length"].tolist()
        for pop in ["always_compliant", "never_compliant", "swing"]
    }
    results["context_length"] = {
        "stats": {
            pop: {
                "mean": float(np.mean(vals)) if vals else 0,
                "median": float(np.median(vals)) if vals else 0,
                "std": float(np.std(vals)) if vals else 0,
                "n": len(vals),
            }
            for pop, vals in c_lens.items()
        },
        "test": kruskal_with_posthoc(c_lens),
    }

    # Number of answer options
    n_opts = {
        pop: feature_df[feature_df["population"] == pop]["num_options"].tolist()
        for pop in ["always_compliant", "never_compliant", "swing"]
    }
    results["num_options"] = {
        "stats": {
            pop: {
                "mean": float(np.mean(vals)) if vals else 0,
                "unique_values": sorted(set(int(v) for v in vals)) if vals else [],
                "n": len(vals),
            }
            for pop, vals in n_opts.items()
        },
    }

    # Response length for anti-compliance (α=1.0) — expected null
    resp_lens = {
        pop: feature_df[feature_df["population"] == pop][
            "anti_compliance_response_length"
        ].tolist()
        for pop in ["always_compliant", "never_compliant", "swing"]
    }
    results["anti_compliance_response_length"] = {
        "stats": {
            pop: {
                "min": int(min(vals)) if vals else 0,
                "max": int(max(vals)) if vals else 0,
                "mean": float(np.mean(vals)),
                "n": len(vals),
            }
            for pop, vals in resp_lens.items()
        },
        "note": "Anti-compliance responses are single-letter; this is a null result.",
    }

    # Standard-prompt response lengths using anti-compliance population labels
    if "standard_response_length" in feature_df.columns:
        std_resp_lens = {
            pop: feature_df[feature_df["population"] == pop][
                "standard_response_length"
            ].tolist()
            for pop in ["always_compliant", "never_compliant", "swing"]
        }
        results["standard_response_length"] = {
            "stats": {
                pop: {
                    "mean": float(np.mean(vals)) if vals else 0,
                    "median": float(np.median(vals)) if vals else 0,
                    "std": float(np.std(vals)) if vals else 0,
                    "n": len(vals),
                }
                for pop, vals in std_resp_lens.items()
            },
            "test": kruskal_with_posthoc(std_resp_lens),
        }

    return results


def analyze_source_datasets(df: pd.DataFrame) -> dict[str, Any]:
    """Chi-squared test of source dataset vs population."""
    ct = pd.crosstab(df["source"], df["population"])
    # Only keep sources with >= 5 samples total for chi-squared validity
    ct_filtered = ct[ct.sum(axis=1) >= 5]
    test_result = chi_squared_test(ct_filtered) if len(ct_filtered) >= 2 else {}
    return {
        "counts": ct.to_dict(),
        "test": test_result,
        "filtered_sources": ct_filtered.index.tolist(),
    }


def analyze_topics(df: pd.DataFrame) -> dict[str, Any]:
    """Topic distribution by population, chi-squared test."""
    ct = pd.crosstab(df["topic"], df["population"])
    test_result = chi_squared_test(ct) if ct.shape[0] >= 2 and ct.shape[1] >= 2 else {}

    # Topic × swing subtype
    swing_df = df[df["population"] == "swing"]
    if len(swing_df) > 0:
        ct_subtype = pd.crosstab(swing_df["topic"], swing_df["swing_subtype"])
        subtype_counts = ct_subtype.to_dict()
    else:
        subtype_counts = {}

    return {
        "counts": ct.to_dict(),
        "test": test_result,
        "swing_subtype_counts": subtype_counts,
    }


def analyze_transitions(
    trajectories: dict[str, list[bool]],
    swing_ids: list[str],
    subtypes: dict[str, str],
) -> dict[str, Any]:
    """Analyze swing trajectory subtypes and transition dynamics."""
    results: dict[str, Any] = {}

    # Subtype counts with Wilson CIs
    subtype_counts = Counter(subtypes.values())
    n_swing = len(swing_ids)
    results["subtype_counts"] = {}
    for st in ["R→C", "C→R", "non-monotonic"]:
        count = subtype_counts.get(st, 0)
        prop = count / n_swing if n_swing > 0 else 0
        lo, hi = wilson_ci(prop, n_swing)
        results["subtype_counts"][st] = {
            "count": count,
            "proportion": round(prop, 4),
            "ci_95": [round(lo, 4), round(hi, 4)],
        }

    # Transition alpha distribution
    trans_alphas: dict[str, list[float]] = {"R→C": [], "C→R": [], "non-monotonic": []}
    for sid in swing_ids:
        ta = find_transition_alpha(trajectories, sid)
        trans_alphas[subtypes[sid]].append(ta)

    results["transition_alpha"] = {
        st: {
            "mean": round(float(np.mean(vals)), 3) if vals else None,
            "median": float(np.median(vals)) if vals else None,
            "values": sorted(vals),
            "counts_by_alpha": {
                f"{alpha:.1f}": int(Counter(vals).get(alpha, 0)) for alpha in ALPHAS[1:]
            },
        }
        for st, vals in trans_alphas.items()
    }

    for st in ["R→C", "C→R"]:
        vals = trans_alphas.get(st, [])
        if not vals:
            continue
        early_count = sum(alpha <= 1.5 for alpha in vals)
        results["transition_alpha"][st]["early_share_le_1_5"] = build_rate_summary(
            early_count,
            len(vals),
            count_key="count",
            total_key="n_total",
        )

    # Mann-Whitney U: R→C vs C→R transition alphas
    rc = trans_alphas.get("R→C", [])
    cr = trans_alphas.get("C→R", [])
    if rc and cr:
        results["rc_vs_cr_transition"] = mann_whitney_effect_size(rc, cr)

    # Resistance strength for R→C: transition alpha distribution
    if rc:
        results["rc_resistance_strength"] = {
            "description": "Alpha at which R→C samples first become compliant (higher = stronger original knowledge)",
            "mean": round(float(np.mean(rc)), 3),
            "median": float(np.median(rc)),
            "distribution": dict(Counter(rc)),
        }

    return results


def analyze_word_overlap(feature_df: pd.DataFrame) -> dict[str, Any]:
    """Lexical overlap between question words and context, by population."""
    overlaps = {
        pop: feature_df[feature_df["population"] == pop]["word_overlap"].tolist()
        for pop in ["always_compliant", "never_compliant", "swing"]
    }

    return {
        "stats": {
            pop: {
                "mean": round(float(np.mean(vals)), 4) if vals else 0,
                "median": round(float(np.median(vals)), 4) if vals else 0,
                "std": round(float(np.std(vals)), 4) if vals else 0,
                "n": len(vals),
            }
            for pop, vals in overlaps.items()
        },
        "test": kruskal_with_posthoc(overlaps),
    }


# ---------------------------------------------------------------------------
# LLM enrichment (optional)
# ---------------------------------------------------------------------------


def _llm_call(client: Any, model: str, prompt: str, max_retries: int = 5) -> str:
    """Call OpenAI chat completion with retry logic."""
    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=150,
            )
            return completion.choices[0].message.content.strip()
        except Exception as e:
            print(f"  LLM API error (attempt {attempt + 1}): {e}")
            time.sleep(2**attempt)
    return "ERROR"


def classify_knowledge(
    client: Any,
    model: str,
    question: str,
) -> str:
    """Ask LLM whether the question is answerable from common knowledge."""
    prompt = (
        f"Is the following science question answerable from common K-12 science "
        f"education knowledge, or does it require specialized/advanced knowledge?\n\n"
        f"Question: {question}\n\n"
        f"Respond with exactly one word: COMMON_KNOWLEDGE or SPECIALIZED or AMBIGUOUS"
    )
    result = _llm_call(client, model, prompt)
    for label in ["COMMON_KNOWLEDGE", "SPECIALIZED", "AMBIGUOUS"]:
        if label in result.upper():
            return label
    return "UNKNOWN"


def verify_answer(
    client: Any,
    model: str,
    question: str,
    choices_text: str,
) -> str:
    """Ask LLM to answer the question independently (no context)."""
    prompt = (
        f"Answer this multiple-choice question. Reply with ONLY the letter.\n\n"
        f"Question: {question}\n\n"
        f"{choices_text}\n\n"
        f"Answer:"
    )
    result = _llm_call(client, model, prompt)
    # Extract first letter
    for ch in result.strip():
        if ch.upper() in "ABCDEFGH":
            return ch.upper()
    return "UNKNOWN"


def rate_persuasiveness(
    client: Any,
    model: str,
    context: str,
    question: str,
) -> int:
    """Rate how convincing the misleading context is (1-5 scale)."""
    prompt = (
        f"You are evaluating a passage that contains misleading information.\n\n"
        f"Context: {context}\n\n"
        f"Question: {question}\n\n"
        f"On a scale of 1-5, how convincing is this misleading context? "
        f"(1 = obviously wrong, 5 = very convincing/subtle)\n\n"
        f"Respond with ONLY a number 1-5."
    )
    result = _llm_call(client, model, prompt)
    for ch in result.strip():
        if ch.isdigit() and 1 <= int(ch) <= 5:
            return int(ch)
    return 0


def is_valid_mc_answer(answer: str) -> bool:
    """Return whether an extracted answer is a usable MC option label."""
    return isinstance(answer, str) and answer.upper() in VALID_MC_ANSWERS


def summarize_llm_enrichment(results_list: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-sample LLM enrichment outputs into site-facing summaries."""
    if not results_list:
        return {
            "samples": [],
            "knowledge_by_population": {},
            "persuasiveness_by_population": {},
        }

    llm_df = pd.DataFrame(results_list)
    knowledge_by_pop: dict[str, dict[str, int]] = {}
    persuasiveness_by_population: dict[str, dict[str, float | int]] = {}

    for pop in llm_df["population"].unique():
        subset = llm_df[llm_df["population"] == pop]
        knowledge_by_pop[pop] = {
            str(k): int(v) for k, v in subset["knowledge_class"].value_counts().items()
        }
        persuasiveness_by_population[pop] = {
            "mean": round(float(subset["persuasiveness"].mean()), 2),
            "n": int(len(subset)),
        }

    comparable_agreement = [
        sample["answer_agrees_with_model_alpha0"]
        for sample in results_list
        if sample.get("answer_agrees_with_model_alpha0") is not None
    ]

    summary: dict[str, Any] = {
        "samples": results_list,
        "knowledge_by_population": knowledge_by_pop,
        "persuasiveness_by_population": persuasiveness_by_population,
    }
    if comparable_agreement:
        summary["verification_agreement"] = build_rate_summary(
            int(sum(comparable_agreement)),
            len(comparable_agreement),
            total_key="n_total",
        )

    return summary


def run_llm_enrichment(
    df: pd.DataFrame,
    hf_data: dict[str, dict[str, Any]],
    rows_by_alpha: dict[float, dict[str, dict[str, Any]]],
    max_samples: int,
    llm_model: str,
) -> dict[str, Any]:
    """Run GPT-4o-mini classification on swing + stratified frozen samples."""
    from dotenv import load_dotenv
    from openai import OpenAI

    load_dotenv()
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY required for --use-llm. Set in .env")
    client = OpenAI(api_key=api_key)

    # Select samples: all swing + stratified sample of frozen
    swing_ids = df[df["population"] == "swing"]["id"].tolist()
    always_ids = df[df["population"] == "always_compliant"]["id"].tolist()
    never_ids = df[df["population"] == "never_compliant"]["id"].tolist()

    rng = np.random.default_rng(42)
    n_swing = min(len(swing_ids), max_samples)
    n_frozen = min(max_samples - n_swing, (len(always_ids) + len(never_ids)))
    n_always_sample = min(len(always_ids), n_frozen // 2)
    n_never_sample = min(len(never_ids), n_frozen - n_always_sample)

    sample_ids = list(rng.choice(swing_ids, n_swing, replace=False))
    if n_always_sample > 0:
        sample_ids += list(rng.choice(always_ids, n_always_sample, replace=False))
    if n_never_sample > 0:
        sample_ids += list(rng.choice(never_ids, n_never_sample, replace=False))

    print(
        f"  LLM enrichment: {len(sample_ids)} samples ({n_swing} swing, "
        f"{n_always_sample} always, {n_never_sample} never)"
    )

    results_list = []
    for i, sid in enumerate(sample_ids):
        row = df[df["id"] == sid].iloc[0]
        hf = hf_data.get(sid, {})
        question = row["question"]
        choices_text = "\n".join(
            f"{lab}) {txt}"
            for lab, txt in zip(
                hf.get("choices_labels", []), hf.get("choices_texts", [])
            )
        )
        context = hf.get("context", "")

        print(f"  [{i + 1}/{len(sample_ids)}] {sid}")
        knowledge = classify_knowledge(client, llm_model, question)
        answer = verify_answer(client, llm_model, question, choices_text)
        persuasiveness = rate_persuasiveness(client, llm_model, context, question)

        # Get model's α=0 answer
        alpha0_row = rows_by_alpha[0.0].get(sid, {})
        model_alpha0_answer = alpha0_row.get("chosen", "")
        counterfactual_key = hf.get("answer_key", "")
        llm_answer_valid = is_valid_mc_answer(answer)
        model_answer_valid = is_valid_mc_answer(model_alpha0_answer)
        key_valid = is_valid_mc_answer(counterfactual_key)
        answer_agrees_with_model = (
            answer == model_alpha0_answer
            if llm_answer_valid and model_answer_valid
            else None
        )
        llm_answer_correct = (
            answer == counterfactual_key if llm_answer_valid and key_valid else None
        )
        model_alpha0_answer_correct = (
            model_alpha0_answer == counterfactual_key
            if model_answer_valid and key_valid
            else None
        )
        shared_error = (
            answer_agrees_with_model
            and llm_answer_correct is False
            and model_alpha0_answer_correct is False
            if answer_agrees_with_model is not None
            and llm_answer_correct is not None
            and model_alpha0_answer_correct is not None
            else None
        )

        results_list.append(
            {
                "id": sid,
                "population": row["population"],
                "swing_subtype": row.get("swing_subtype", ""),
                "knowledge_class": knowledge,
                "llm_answer": answer,
                "model_alpha0_answer": model_alpha0_answer,
                "counterfactual_key": counterfactual_key,
                "answer_agrees_with_model_alpha0": answer_agrees_with_model,
                "llm_answer_correct": llm_answer_correct,
                "model_alpha0_answer_correct": model_alpha0_answer_correct,
                "shared_error": shared_error,
                "persuasiveness": persuasiveness,
            }
        )

    return summarize_llm_enrichment(results_list)


# ---------------------------------------------------------------------------
# Plotting functions
# ---------------------------------------------------------------------------


def _save_fig(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path.name}")


def plot_population_overview(
    n_always: int,
    n_never: int,
    subtype_counts: dict[str, int],
    output_dir: Path,
) -> None:
    """Stacked bar showing swing subtypes within overall population split."""
    fig, ax = plt.subplots(figsize=(8, 5))

    # Main population bars
    categories = ["Always\ncompliant", "Never\ncompliant", "Swing"]
    # Swing is split into subtypes
    rc = subtype_counts.get("R→C", 0)
    cr = subtype_counts.get("C→R", 0)
    nm = subtype_counts.get("non-monotonic", 0)

    # Plot always and never as solid bars
    ax.barh(
        [0], [n_always], color=POPULATION_COLORS["always_compliant"], edgecolor="white"
    )
    ax.barh(
        [1], [n_never], color=POPULATION_COLORS["never_compliant"], edgecolor="white"
    )

    # Swing as stacked subtypes
    left = 0
    for label, count, color in [
        ("R→C", rc, SUBTYPE_COLORS["R→C"]),
        ("C→R", cr, SUBTYPE_COLORS["C→R"]),
        ("Non-mono", nm, SUBTYPE_COLORS["non-monotonic"]),
    ]:
        ax.barh(
            [2],
            [count],
            left=left,
            color=color,
            edgecolor="white",
            label=f"{label} ({count})",
        )
        if count > 0:
            ax.text(
                left + count / 2,
                2,
                str(count),
                ha="center",
                va="center",
                fontsize=10,
                fontweight="bold",
                color="white",
            )
        left += count

    # Annotate
    ax.text(
        n_always / 2,
        0,
        str(n_always),
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
        color="white",
    )
    ax.text(
        n_never / 2,
        1,
        str(n_never),
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
        color="white",
    )

    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(categories, fontsize=11)
    ax.set_xlabel("Number of samples", fontsize=11)
    ax.set_title("FaithEval population split & swing subtypes", fontsize=13)
    ax.legend(loc="lower right", fontsize=9)
    ax.invert_yaxis()
    sns.despine(ax=ax)

    _save_fig(fig, output_dir / "population_overview.png")


def plot_transition_alpha(
    trajectories: dict[str, list[bool]],
    swing_ids: list[str],
    subtypes: dict[str, str],
    output_dir: Path,
) -> None:
    """Histogram of first-transition α by subtype."""
    data = []
    for sid in swing_ids:
        ta = find_transition_alpha(trajectories, sid)
        data.append({"subtype": subtypes[sid], "transition_alpha": ta})
    tdf = pd.DataFrame(data)

    fig, ax = plt.subplots(figsize=(7, 4))
    for st in ["R→C", "C→R", "non-monotonic"]:
        subset = tdf[tdf["subtype"] == st]
        if len(subset) == 0:
            continue
        ax.hist(
            subset["transition_alpha"],
            bins=list(np.arange(0.25, 3.75, 0.5)),
            alpha=0.6,
            color=SUBTYPE_COLORS[st],
            label=f"{st} (n={len(subset)})",
            edgecolor="white",
        )
    ax.set_xlabel("First transition α", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("When do swing samples first change behavior?", fontsize=13)
    ax.legend(fontsize=9)
    ax.set_xticks(ALPHAS)
    sns.despine(ax=ax)

    _save_fig(fig, output_dir / "transition_alpha_distribution.png")


def plot_structural_proxies(
    feature_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Violin plots for question length, context length, word overlap."""
    plot_df = feature_df[
        ["population", "question_length", "context_length", "word_overlap"]
    ].copy()

    pop_order = ["always_compliant", "never_compliant", "swing"]
    pop_labels = {
        "always_compliant": "Always",
        "never_compliant": "Never",
        "swing": "Swing",
    }
    palette = [POPULATION_COLORS[p] for p in pop_order]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    for ax, col, title in zip(
        axes,
        ["question_length", "context_length", "word_overlap"],
        [
            "Question length (chars)",
            "Context length (chars)",
            "Question–context word overlap",
        ],
    ):
        sns.violinplot(
            data=plot_df,
            x="population",
            y=col,
            hue="population",
            order=pop_order,
            hue_order=pop_order,
            palette=palette,
            ax=ax,
            inner="box",
            cut=0,
            legend=False,
        )
        ax.set_xticks(range(len(pop_order)))
        ax.set_xticklabels([pop_labels[p] for p in pop_order], fontsize=10)
        ax.set_xlabel("")
        ax.set_ylabel(title, fontsize=10)
        sns.despine(ax=ax)

    fig.suptitle("Structural proxies by population", fontsize=13, y=1.02)
    fig.tight_layout()
    _save_fig(fig, output_dir / "structural_proxies.png")


def plot_source_datasets(df: pd.DataFrame, output_dir: Path) -> None:
    """Grouped horizontal bar chart of population fractions by source dataset."""
    ct = pd.crosstab(df["source"], df["population"])
    # Normalize by row (source)
    ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100
    # Only sources with >= 10 samples
    mask = ct.sum(axis=1) >= 10
    ct_pct = ct_pct[mask].sort_values("swing", ascending=True)

    if len(ct_pct) == 0:
        return

    fig, ax = plt.subplots(figsize=(8, max(4, len(ct_pct) * 0.5)))
    pop_order = ["always_compliant", "never_compliant", "swing"]
    pop_labels = {
        "always_compliant": "Always compliant",
        "never_compliant": "Never compliant",
        "swing": "Swing",
    }
    y_pos = np.arange(len(ct_pct))
    bar_height = 0.25

    for i, pop in enumerate(pop_order):
        if pop in ct_pct.columns:
            vals = ct_pct[pop].values
        else:
            vals = np.zeros(len(ct_pct))
        ax.barh(
            y_pos + i * bar_height,
            vals,
            height=bar_height,
            color=POPULATION_COLORS[pop],
            label=pop_labels[pop],
        )

    ax.set_yticks(y_pos + bar_height)
    # Add sample counts to labels
    ct_totals = ct.sum(axis=1)[mask]
    ct_totals = ct_totals.reindex(ct_pct.index)
    ax.set_yticklabels(
        [f"{src} (n={ct_totals[src]})" for src in ct_pct.index], fontsize=9
    )
    ax.set_xlabel("Percentage of source (%)", fontsize=11)
    ax.set_title("Population composition by source dataset", fontsize=13)
    ax.legend(loc="lower right", fontsize=9)
    sns.despine(ax=ax)

    _save_fig(fig, output_dir / "source_dataset_rates.png")


def plot_topic_by_population(df: pd.DataFrame, output_dir: Path) -> None:
    """Grouped bar chart of topic distribution within each population."""
    ct = pd.crosstab(df["population"], df["topic"])
    ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100
    pop_order = ["always_compliant", "never_compliant", "swing"]
    pop_labels = {
        "always_compliant": "Always",
        "never_compliant": "Never",
        "swing": "Swing",
    }

    topics = sorted(ct_pct.columns)
    x = np.arange(len(topics))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, pop in enumerate(pop_order):
        vals = [ct_pct.loc[pop, t] if t in ct_pct.columns else 0 for t in topics]
        ax.bar(
            x + i * width,
            vals,
            width,
            color=POPULATION_COLORS[pop],
            label=pop_labels[pop],
        )

    ax.set_xticks(x + width)
    ax.set_xticklabels(
        [t.replace("_", " ").title() for t in topics],
        rotation=30,
        ha="right",
        fontsize=10,
    )
    ax.set_ylabel("Percentage within population (%)", fontsize=11)
    ax.set_title("Topic distribution by population", fontsize=13)
    ax.legend(fontsize=9)
    sns.despine(ax=ax)
    fig.tight_layout()

    _save_fig(fig, output_dir / "topic_by_population.png")


def plot_trajectory_heatmap(
    trajectories: dict[str, list[bool]],
    swing_ids: list[str],
    subtypes: dict[str, str],
    output_dir: Path,
) -> None:
    """138×7 heatmap of per-sample compliance trajectories."""
    # Sort by subtype (R→C first, then C→R, then non-monotonic), then by transition α
    sorted_ids = sorted(
        swing_ids,
        key=lambda sid: (
            ["R→C", "C→R", "non-monotonic"].index(subtypes[sid]),
            find_transition_alpha(trajectories, sid),
        ),
    )

    matrix = np.array(
        [
            [1 if trajectories[sid][i] else 0 for i in range(len(ALPHAS))]
            for sid in sorted_ids
        ]
    )

    fig, ax = plt.subplots(figsize=(6, max(8, len(sorted_ids) * 0.06)))
    cmap = plt.cm.colors.ListedColormap([COLORS["coral"], COLORS["teal"]])  # type: ignore[attr-defined]
    ax.imshow(matrix, aspect="auto", cmap=cmap, interpolation="nearest")

    # Subtype boundaries
    subtypes_ordered = [subtypes[sid] for sid in sorted_ids]
    for i in range(1, len(subtypes_ordered)):
        if subtypes_ordered[i] != subtypes_ordered[i - 1]:
            ax.axhline(y=i - 0.5, color="white", linewidth=2)

    # Subtype labels on right
    counts = Counter(subtypes_ordered)
    y_pos = 0
    for st in ["R→C", "C→R", "non-monotonic"]:
        n = counts.get(st, 0)
        if n > 0:
            ax.text(
                len(ALPHAS) + 0.3,
                y_pos + n / 2 - 0.5,
                f"{st}\n(n={n})",
                va="center",
                fontsize=9,
                fontweight="bold",
            )
            y_pos += n

    ax.set_xticks(range(len(ALPHAS)))
    ax.set_xticklabels([f"α={a}" for a in ALPHAS], fontsize=9)
    ax.set_ylabel("Swing samples (sorted by subtype & transition α)", fontsize=10)
    ax.set_title("Per-sample compliance trajectories", fontsize=13)
    ax.set_yticks([])

    # Legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor=COLORS["teal"], label="Compliant"),
        Patch(facecolor=COLORS["coral"], label="Resistant"),
    ]
    ax.legend(handles=legend_elements, loc="lower left", fontsize=9)

    _save_fig(fig, output_dir / "trajectory_heatmap.png")


def plot_knowledge_classification(
    llm_results: dict[str, Any],
    output_dir: Path,
) -> None:
    """Stacked bar of GPT-4o-mini knowledge classification by population."""
    knowledge = llm_results["knowledge_by_population"]
    if not knowledge:
        return

    pop_order = [
        p for p in ["always_compliant", "never_compliant", "swing"] if p in knowledge
    ]
    pop_labels = {
        "always_compliant": "Always",
        "never_compliant": "Never",
        "swing": "Swing",
    }
    categories = ["COMMON_KNOWLEDGE", "SPECIALIZED", "AMBIGUOUS", "UNKNOWN"]
    cat_colors = ["#2ecc71", "#e74c3c", "#f39c12", "#95a5a6"]

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(pop_order))
    bottoms = np.zeros(len(pop_order))

    for cat, color in zip(categories, cat_colors):
        vals = [knowledge.get(pop, {}).get(cat, 0) for pop in pop_order]
        ax.bar(
            x,
            vals,
            bottom=bottoms,
            color=color,
            label=cat.replace("_", " ").title(),
            edgecolor="white",
        )
        bottoms += vals

    ax.set_xticks(x)
    ax.set_xticklabels([pop_labels[p] for p in pop_order], fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Knowledge classification by population (GPT-4o-mini)", fontsize=13)
    ax.legend(fontsize=9)
    sns.despine(ax=ax)

    _save_fig(fig, output_dir / "knowledge_classification.png")


# ---------------------------------------------------------------------------
# CLI & main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Characterize swing samples in FaithEval intervention experiment."
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/gemma3_4b/swing_characterization"),
        help="Output directory for results and figures.",
    )
    p.add_argument(
        "--use-llm",
        action="store_true",
        help="Enable GPT-4o-mini enrichment (requires OPENAI_API_KEY in .env).",
    )
    p.add_argument(
        "--llm-model",
        type=str,
        default="gpt-4o-mini",
        help="OpenAI model for LLM enrichment.",
    )
    p.add_argument(
        "--max-llm-samples",
        type=int,
        default=200,
        help="Max samples for LLM enrichment.",
    )
    p.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip figure generation.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    anti_dir = repo_root / "data/gemma3_4b/intervention/faitheval/experiment"
    standard_dir = (
        repo_root / "data/gemma3_4b/intervention/faitheval_standard/experiment"
    )
    output_dir = repo_root / args.output_dir
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Load data & classify populations ──────────────────────────
    print("Step 1: Loading data & classifying populations...")
    rows_by_alpha = load_all_alphas(anti_dir)
    trajectories, always_ids, never_ids, swing_ids = classify_populations(rows_by_alpha)

    print(f"  Always compliant: {len(always_ids)}")
    print(f"  Never compliant:  {len(never_ids)}")
    print(f"  Swing:            {len(swing_ids)}")

    assert len(always_ids) == 600, (
        f"Expected 600 always-compliant, got {len(always_ids)}"
    )
    assert len(never_ids) == 262, f"Expected 262 never-compliant, got {len(never_ids)}"
    assert len(swing_ids) == 138, f"Expected 138 swing, got {len(swing_ids)}"

    # Swing subtypes
    subtypes = classify_swing_subtypes(trajectories, swing_ids)
    subtype_counts = Counter(subtypes.values())
    print(f"  Swing subtypes: {dict(subtype_counts)}")

    # Load HuggingFace metadata
    print("  Loading HuggingFace FaithEval metadata...")
    hf_data = load_hf_faitheval()
    print(f"  HF dataset: {len(hf_data)} samples")

    # Build main DataFrame from α=0 rows
    alpha0 = rows_by_alpha[0.0]
    records = []
    for sid in sorted(alpha0):
        row = alpha0[sid]
        pop = (
            "always_compliant"
            if sid in always_ids
            else "never_compliant"
            if sid in never_ids
            else "swing"
        )
        records.append(
            {
                "id": sid,
                "question": row["question"],
                "response": row["response"],
                "counterfactual_key": row["counterfactual_key"],
                "chosen": row["chosen"],
                "compliance_alpha0": row["compliance"],
                "population": pop,
                "swing_subtype": subtypes.get(sid, ""),
                "source": extract_source(sid),
                "topic": classify_topic(row["question"]),
            }
        )
    df = pd.DataFrame(records)
    feature_df = build_feature_table(df, hf_data, standard_dir)

    # Save population IDs
    pop_ids = {
        "always_compliant": always_ids,
        "never_compliant": never_ids,
        "swing": swing_ids,
        "swing_subtypes": {
            "R→C": [s for s, st in subtypes.items() if st == "R→C"],
            "C→R": [s for s, st in subtypes.items() if st == "C→R"],
            "non-monotonic": [s for s, st in subtypes.items() if st == "non-monotonic"],
        },
    }
    (output_dir / "population_ids.json").write_text(
        json.dumps(pop_ids, indent=2) + "\n"
    )
    print("  Saved population_ids.json")
    (output_dir / "feature_table.jsonl").write_text(
        "\n".join(
            json.dumps(record)
            for record in feature_df.sort_values("id").to_dict(orient="records")
        )
        + "\n"
    )
    print("  Saved feature_table.jsonl")

    # ── Step 2: Structural proxies ────────────────────────────────────────
    print("\nStep 2: Analyzing structural proxies...")
    structural = analyze_structural_proxies(feature_df)

    print("\nStep 2b: Testing held-out structural predictability...")
    structural_predictability = analyze_structural_predictability(feature_df)

    # ── Step 3: Source datasets & topics ───────────────────────────────────
    print("\nStep 3: Analyzing source datasets & topics...")
    source_analysis = analyze_source_datasets(df)
    topic_analysis = analyze_topics(df)

    # ── Step 4: Transition dynamics ───────────────────────────────────────
    print("\nStep 4: Analyzing transition dynamics...")
    transition_analysis = analyze_transitions(trajectories, swing_ids, subtypes)

    # ── Step 5: Word overlap ──────────────────────────────────────────────
    print("\nStep 5: Analyzing question–context word overlap...")
    overlap_analysis = analyze_word_overlap(feature_df)

    # ── Step 6: LLM enrichment (optional) ─────────────────────────────────
    llm_results = None
    if args.use_llm:
        print("\nStep 6: Running LLM enrichment...")
        llm_results = run_llm_enrichment(
            df, hf_data, rows_by_alpha, args.max_llm_samples, args.llm_model
        )

    # ── Save summary ──────────────────────────────────────────────────────
    summary: dict[str, Any] = {
        "population_counts": {
            "always_compliant": len(always_ids),
            "never_compliant": len(never_ids),
            "swing": len(swing_ids),
            "total": len(always_ids) + len(never_ids) + len(swing_ids),
        },
        "structural_proxies": structural,
        "structural_predictability": structural_predictability,
        "source_datasets": source_analysis,
        "topics": topic_analysis,
        "transitions": transition_analysis,
        "word_overlap": overlap_analysis,
    }
    if llm_results:
        summary["llm_enrichment"] = llm_results

    class _NumpyEncoder(json.JSONEncoder):
        def default(self, o: Any) -> Any:
            if isinstance(o, (np.integer,)):
                return int(o)
            if isinstance(o, (np.floating,)):
                return float(o)
            if isinstance(o, (np.bool_,)):
                return bool(o)
            if isinstance(o, np.ndarray):
                return o.tolist()
            return super().default(o)

    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, cls=_NumpyEncoder) + "\n"
    )
    print(f"\nSaved summary.json to {output_dir}")

    # ── Figures ───────────────────────────────────────────────────────────
    if not args.skip_plots:
        print("\nGenerating figures...")
        plot_population_overview(
            len(always_ids), len(never_ids), dict(subtype_counts), figures_dir
        )
        plot_transition_alpha(trajectories, swing_ids, subtypes, figures_dir)
        plot_structural_proxies(feature_df, figures_dir)
        plot_source_datasets(df, figures_dir)
        plot_topic_by_population(df, figures_dir)
        plot_trajectory_heatmap(trajectories, swing_ids, subtypes, figures_dir)

        if llm_results:
            plot_knowledge_classification(llm_results, figures_dir)

    print("\nDone!")


if __name__ == "__main__":
    main()
