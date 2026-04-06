"""Extract head-level ITI artifacts for truthfulness steering.

This script builds one of seven explicit artifact families:

- ``iti_triviaqa_transfer``: closed-book factuality transfer from TriviaQA
  consistency responses.
- ``iti_context_grounded``: matched context-grounded answering + abstention
  from SQuAD-v2.
- ``iti_truthfulqa_paperfaithful``: paper-faithful extraction from TruthfulQA
  QA pairs following Li et al. (arXiv 2306.03341v6).  Last-token only,
  validation-accuracy ranking, 2-fold CV with dev/test split from fold files.
- ``iti_truthfulqa_exploratory``: same TruthfulQA data source, but with
  modernized ranking (AUROC primary, all position summaries).  Use this path
  to iterate on ranking metrics, candidate pruning, or alpha/K search without
  polluting the paper-faithful baseline.
- ``iti_truthfulqa_modernized``: TruthfulQA extraction with chat-template
  prompt matching, assistant-content token positions, AUROC ranking, and
  first/mean/last answer-position selection.
- ``iti_refusal_probe``: paired JBB harmful/benign prompt extraction with
  per-head logistic probes over prompt-final token activations.
- ``iti_refusal_causal``: paired JBB harmful/benign prompt extraction with
  head ranking via contrastive grad×activation scoring (∂logp/∂z · z per
  objective; motivated by GCM, not identical to GCM attribution patching).

The extraction surface is the tensor immediately before ``self_attn.o_proj``.
Heads are ranked by probe metrics and steered with a truthful-oriented
mass-mean direction for decode-time intervention.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from build_truthfulness_contrastive import (
    load_consistency_samples,
    load_id_split,
    validate_disjoint_qid_splits,
)
from intervene_iti import _get_decoder_layers
from utils import (
    audit_split_leakage,
    finish_run_provenance,
    fingerprint_ids,
    get_git_sha,
    json_dumps,
    normalize_answer,
    parse_semicolon_answers,
    provenance_error_message,
    provenance_status_for_exception,
    start_run_provenance,
)


POSITION_SUMMARIES = (
    "first_answer_token",
    "mean_answer_span",
    "last_answer_token",
)
LABEL_FALSE = 0
LABEL_TRUE = 1
DEFAULT_ABSTENTION = "The context does not contain the answer."
REFUSAL_FAMILIES = (
    "iti_refusal_probe",
    "iti_refusal_causal",
)


@dataclass(frozen=True)
class ITIExample:
    example_id: str
    family: str
    split: str
    qid: str
    question: str
    answer: str
    label: int
    weight: float
    prompt_text: str
    answer_positions: tuple[int, ...]
    metadata: dict[str, Any]
    prompt_input_ids: tuple[int, ...] | None = None


@dataclass(frozen=True)
class D7CausalPair:
    pair_id: str
    split: str
    behavior_index: int
    template_idx: int
    harmful_prompt: str
    harmful_response_target: str
    benign_response_target: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract head-level truthfulness ITI")
    parser.add_argument(
        "--family",
        type=str,
        required=True,
        choices=[
            "iti_triviaqa_transfer",
            "iti_context_grounded",
            "iti_truthfulqa_paperfaithful",
            "iti_truthfulqa_exploratory",
            "iti_truthfulqa_modernized",
            "iti_refusal_probe",
            "iti_refusal_causal",
        ],
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="google/gemma-3-4b-it",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Defaults to data/contrastive/truthfulness/<family>/",
    )
    parser.add_argument(
        "--device_map",
        type=str,
        default="cuda:0",
    )
    parser.add_argument(
        "--consistency_path",
        type=Path,
        default=Path("data/gemma3_4b/pipeline/consistency_samples.jsonl"),
    )
    parser.add_argument(
        "--train_qids_path",
        type=Path,
        default=Path("data/gemma3_4b/pipeline/train_qids.json"),
    )
    parser.add_argument(
        "--test_qids_path",
        type=Path,
        default=Path("data/gemma3_4b/pipeline/test_qids_disjoint.json"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--context_train_questions", type=int, default=400)
    parser.add_argument("--context_val_questions", type=int, default=100)
    parser.add_argument("--train_cap_per_qid_label", type=int, default=2)
    parser.add_argument("--val_cap_per_qid_label", type=int, default=1)
    parser.add_argument(
        "--fold_path",
        type=Path,
        default=None,
        help="Fold JSON from build_truthfulqa_splits.py. "
        "Required for iti_truthfulqa_paperfaithful, iti_truthfulqa_exploratory, "
        "and iti_truthfulqa_modernized.",
    )
    parser.add_argument(
        "--csv_path",
        type=Path,
        default=Path("data/benchmarks/TruthfulQA.csv"),
        help="Path to the official TruthfulQA CSV (canonical data source).",
    )
    parser.add_argument(
        "--ranking_metric_override",
        type=str,
        default=None,
        choices=["auroc", "balanced_accuracy", "val_accuracy"],
        help=(
            "Optional probe ranking-metric override. Default remains family-specific "
            "(paper-faithful: val_accuracy; others: auroc)."
        ),
    )
    parser.add_argument(
        "--position_policy_override",
        type=str,
        default=None,
        choices=["last_answer_token", "all_answer_positions"],
        help=(
            "Optional answer-position policy override. "
            "'last_answer_token' uses only the final answer token; "
            "'all_answer_positions' uses first/mean/last summaries."
        ),
    )
    parser.add_argument(
        "--d7_manifest_path",
        type=Path,
        default=Path("data/manifests/jbb_d7_extraction_pairs_seed42.jsonl"),
        help="Path to D7 extraction-pair JSONL manifest (required for refusal families).",
    )
    return parser.parse_args()


def default_output_dir(family: str) -> Path:
    suffix_map = {
        "iti_triviaqa_transfer": "iti_triviaqa",
        "iti_context_grounded": "iti_context",
        "iti_truthfulqa_paperfaithful": "iti_truthfulqa_paperfaithful",
        "iti_truthfulqa_exploratory": "iti_truthfulqa_exploratory",
        "iti_truthfulqa_modernized": "iti_truthfulqa_modernized",
        "iti_refusal_probe": "iti_refusal_probe",
        "iti_refusal_causal": "iti_refusal_causal",
    }
    suffix = suffix_map[family]
    return Path("data/contrastive/truthfulness") / suffix


def resolve_selector_policy(
    *,
    family: str,
    ranking_metric_override: str | None,
    position_policy_override: str | None,
) -> tuple[str, tuple[str, ...], str]:
    """Resolve ranking metric and position summaries for head selection."""
    if family in REFUSAL_FAMILIES:
        if position_policy_override and position_policy_override != "last_answer_token":
            raise ValueError(
                "Refusal families currently support only last_answer_token position policy"
            )
        rank_primary = ranking_metric_override or "auroc"
        return rank_primary, ("last_answer_token",), "last_answer_token"

    is_paper_faithful = family == "iti_truthfulqa_paperfaithful"
    default_rank_primary = "val_accuracy" if is_paper_faithful else "auroc"
    default_position_policy = (
        "last_answer_token" if is_paper_faithful else "all_answer_positions"
    )

    rank_primary = ranking_metric_override or default_rank_primary
    position_policy = position_policy_override or default_position_policy
    rank_position_summaries = (
        ("last_answer_token",)
        if position_policy == "last_answer_token"
        else POSITION_SUMMARIES
    )
    return rank_primary, rank_position_summaries, position_policy


def _qa_prompt(
    question: str, answer: str, *, context: str | None = None
) -> tuple[str, str]:
    if context:
        prefix = f"Context: {context}\n\nQuestion: {question}\nAnswer:"
    else:
        prefix = f"Question: {question}\nAnswer:"
    full = f"{prefix} {answer}".rstrip()
    return prefix, full


def _encode_with_answer_positions(
    tokenizer: Any,
    *,
    question: str,
    answer: str,
    context: str | None = None,
) -> tuple[str, tuple[int, ...]]:
    prefix, full = _qa_prompt(question, answer, context=context)
    prefix_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full, add_special_tokens=False)["input_ids"]
    if len(full_ids) <= len(prefix_ids):
        raise ValueError(
            "Answer encoding produced no answer tokens for "
            f"question={question!r}, answer={answer!r}"
        )
    positions = tuple(range(len(prefix_ids), len(full_ids)))
    return full, positions


_JBB_SAMPLE_ID_PATTERN = re.compile(r"^jbb_(harmful|benign)_(\d+)_t(\d+)$")


def _parse_jbb_sample_id(sample_id: str) -> tuple[str, int, int]:
    match = _JBB_SAMPLE_ID_PATTERN.match(sample_id)
    if not match:
        raise ValueError(f"Malformed JBB sample ID: {sample_id!r}")
    split, index, template_idx = match.groups()
    return split, int(index), int(template_idx)


def _encode_prompt_last_token(
    tokenizer: Any,
    *,
    prompt: str,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    if len(prompt_ids) == 0:
        raise ValueError(f"Prompt encodes to empty token sequence: {prompt!r}")
    prompt_ids_tuple = tuple(int(token_id) for token_id in prompt_ids)
    return prompt_ids_tuple, (len(prompt_ids_tuple) - 1,)


def _load_d7_manifest_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"D7 manifest row must be an object: {payload!r}")
            rows.append(payload)
    return rows


def build_refusal_d7_examples(
    *,
    tokenizer: Any,
    manifest_path: Path,
    family: str,
) -> tuple[list[ITIExample], list[D7CausalPair], dict[str, Any]]:
    rows = _load_d7_manifest_rows(manifest_path)
    if not rows:
        raise ValueError(f"D7 manifest has no rows: {manifest_path}")

    examples: list[ITIExample] = []
    causal_pairs: list[D7CausalPair] = []
    seen_pair_ids: set[str] = set()
    seen_harmful_ids: set[str] = set()
    seen_benign_ids: set[str] = set()

    for row in rows:
        pair_id = str(row["pair_id"])
        if pair_id in seen_pair_ids:
            raise ValueError(f"Duplicate pair_id in D7 manifest: {pair_id}")
        seen_pair_ids.add(pair_id)

        split = str(row.get("split", "train"))
        if split not in {"train", "val"}:
            raise ValueError(f"Invalid split for pair_id={pair_id}: {split!r}")

        harmful_sample_id = str(row["harmful_sample_id"])
        benign_sample_id = str(row["benign_sample_id"])
        h_split, h_index, h_template_idx = _parse_jbb_sample_id(harmful_sample_id)
        b_split, b_index, b_template_idx = _parse_jbb_sample_id(benign_sample_id)
        if h_split != "harmful" or b_split != "benign":
            raise ValueError(
                f"Pair {pair_id} has invalid harmful/benign ID split tags: "
                f"{harmful_sample_id!r}, {benign_sample_id!r}"
            )
        if h_index != b_index or h_template_idx != b_template_idx:
            raise ValueError(
                f"Pair {pair_id} is misaligned by index/template: "
                f"{harmful_sample_id!r} vs {benign_sample_id!r}"
            )
        declared_index = int(row["behavior_index"])
        declared_template_idx = int(row["template_idx"])
        if declared_index != h_index or declared_template_idx != h_template_idx:
            raise ValueError(
                f"Pair {pair_id} manifest index/template mismatch: "
                f"declared=({declared_index},{declared_template_idx}) "
                f"parsed=({h_index},{h_template_idx})"
            )
        if harmful_sample_id in seen_harmful_ids:
            raise ValueError(f"Duplicate harmful sample ID: {harmful_sample_id}")
        if benign_sample_id in seen_benign_ids:
            raise ValueError(f"Duplicate benign sample ID: {benign_sample_id}")
        seen_harmful_ids.add(harmful_sample_id)
        seen_benign_ids.add(benign_sample_id)

        harmful_prompt = str(row["harmful_prompt"]).strip()
        benign_prompt = str(row["benign_prompt"]).strip()
        harmful_target = str(row["harmful_response_target"]).strip()
        benign_target = str(row["benign_response_target"]).strip()
        if not harmful_prompt or not benign_prompt:
            raise ValueError(f"Pair {pair_id} has empty prompt field")
        if not harmful_target or not benign_target:
            raise ValueError(f"Pair {pair_id} has empty target field")

        harmful_prompt_ids, harmful_positions = _encode_prompt_last_token(
            tokenizer,
            prompt=harmful_prompt,
        )
        benign_prompt_ids, benign_positions = _encode_prompt_last_token(
            tokenizer,
            prompt=benign_prompt,
        )

        metadata_base = {
            "pair_id": pair_id,
            "behavior_index": declared_index,
            "behavior": str(row["behavior"]),
            "category": str(row["category"]),
            "source": str(row.get("source", "")),
            "template_idx": declared_template_idx,
            "harmful_sample_id": harmful_sample_id,
            "benign_sample_id": benign_sample_id,
        }

        examples.append(
            ITIExample(
                example_id=harmful_sample_id,
                family=family,
                split=split,
                qid=pair_id,
                question=harmful_prompt,
                answer=str(row["harmful_goal"]),
                label=LABEL_FALSE,
                weight=1.0,
                prompt_text=harmful_prompt,
                answer_positions=harmful_positions,
                metadata={**metadata_base, "prompt_role": "harmful"},
                prompt_input_ids=harmful_prompt_ids,
            )
        )
        examples.append(
            ITIExample(
                example_id=benign_sample_id,
                family=family,
                split=split,
                qid=pair_id,
                question=benign_prompt,
                answer=str(row["benign_goal"]),
                label=LABEL_TRUE,
                weight=1.0,
                prompt_text=benign_prompt,
                answer_positions=benign_positions,
                metadata={**metadata_base, "prompt_role": "benign"},
                prompt_input_ids=benign_prompt_ids,
            )
        )
        causal_pairs.append(
            D7CausalPair(
                pair_id=pair_id,
                split=split,
                behavior_index=declared_index,
                template_idx=declared_template_idx,
                harmful_prompt=harmful_prompt,
                harmful_response_target=harmful_target,
                benign_response_target=benign_target,
            )
        )

    metadata = {
        "family": family,
        "source_dataset": "JailbreakBench/JBB-Behaviors paired manifest",
        "manifest_path": str(manifest_path),
        "prompt_format": "templated_jbb_goal_prompt",
        "answer_token_policy": "last_prompt_token",
        "question_counts": {
            "train_pairs": sum(1 for pair in causal_pairs if pair.split == "train"),
            "val_pairs": sum(1 for pair in causal_pairs if pair.split == "val"),
        },
        "example_counts": {
            "train_harmful": sum(
                1 for ex in examples if ex.split == "train" and ex.label == LABEL_FALSE
            ),
            "train_benign": sum(
                1 for ex in examples if ex.split == "train" and ex.label == LABEL_TRUE
            ),
            "val_harmful": sum(
                1 for ex in examples if ex.split == "val" and ex.label == LABEL_FALSE
            ),
            "val_benign": sum(
                1 for ex in examples if ex.split == "val" and ex.label == LABEL_TRUE
            ),
        },
        "split_fingerprint": {
            "train_qids": fingerprint_ids(
                sorted({pair.pair_id for pair in causal_pairs if pair.split == "train"})
            ),
            "val_qids": fingerprint_ids(
                sorted({pair.pair_id for pair in causal_pairs if pair.split == "val"})
            ),
        },
        "question_ids_train": sorted(
            {pair.pair_id for pair in causal_pairs if pair.split == "train"}
        ),
        "question_ids_val": sorted(
            {pair.pair_id for pair in causal_pairs if pair.split == "val"}
        ),
        "question_ids_dev": sorted({pair.pair_id for pair in causal_pairs}),
        "question_ids_test": [],
        "direction_fit_scope": "dev_only",
        "direction_fit_source": "dev_only",
        "sigma_fit_source": "dev_only",
    }
    return examples, causal_pairs, metadata


def _weighted_mean(X: np.ndarray, weights: np.ndarray) -> np.ndarray:
    normalized = weights / np.clip(weights.sum(), 1e-8, None)
    return np.sum(X * normalized[:, None], axis=0)


def _truth_mass_mean_direction(
    X: np.ndarray, y: np.ndarray, weights: np.ndarray
) -> tuple[np.ndarray, float, dict[str, float], int]:
    true_mask = y == LABEL_TRUE
    false_mask = y == LABEL_FALSE
    truthful_mean = _weighted_mean(X[true_mask], weights[true_mask])
    false_mean = _weighted_mean(X[false_mask], weights[false_mask])
    direction = truthful_mean - false_mean
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-8:
        direction = np.zeros_like(direction)
    else:
        direction = direction / norm
    projections = X @ direction
    sigma = float(np.std(projections))
    class_means = {
        "truthful_projection_mean": float(np.mean(projections[true_mask])),
        "untruthful_projection_mean": float(np.mean(projections[false_mask])),
    }
    sign = (
        1
        if class_means["truthful_projection_mean"]
        >= class_means["untruthful_projection_mean"]
        else -1
    )
    if sign < 0:
        direction = -direction
        projections = X @ direction
        class_means = {
            "truthful_projection_mean": float(np.mean(projections[true_mask])),
            "untruthful_projection_mean": float(np.mean(projections[false_mask])),
        }
    return direction, sigma, class_means, sign


def compute_head_directions(
    ranked_heads: list[dict[str, Any]],
    activations: dict[str, torch.Tensor],
    examples: list[ITIExample],
    direction_fit_indices: list[int],
    *,
    direction_source: Literal["train_data", "dev_data"],
) -> list[dict[str, Any]]:
    """Compute mass-mean directions using an explicit set of fit indices.

    After head ranking selects which heads to steer (using val metrics),
    this function computes the steering direction from a specified subset
    of examples.  For paper-faithful mode, ``direction_fit_indices``
    covers all dev data (train+val); for other families, train only.

    No index in ``direction_fit_indices`` may point to an example with
    ``split='test'``.
    """
    for idx in direction_fit_indices:
        assert examples[idx].split != "test", (
            f"FATAL: test-fold example at index {idx} in direction_fit_indices. "
            "Test data must never be used for direction fitting."
        )
    y = np.array([examples[i].label for i in direction_fit_indices], dtype=np.int64)
    w = np.array([examples[i].weight for i in direction_fit_indices], dtype=np.float64)

    for head_entry in ranked_heads:
        summary_name = head_entry["position_summary"]
        layer_idx = head_entry["layer"]
        head_idx = head_entry["head"]
        X = (
            activations[summary_name][direction_fit_indices, layer_idx, head_idx]
            .float()
            .numpy()
        )
        direction, sigma, class_means, sign = _truth_mass_mean_direction(X, y, w)
        head_entry["direction"] = direction.tolist()
        head_entry["sigma"] = round(sigma, 6)
        head_entry["sign"] = sign
        head_entry["class_means"] = {
            key: round(value, 6) for key, value in class_means.items()
        }
        head_entry["direction_source"] = direction_source
    return ranked_heads


def build_triviaqa_transfer_examples(
    *,
    consistency_path: Path,
    train_qids_path: Path,
    test_qids_path: Path,
    tokenizer: Any,
    train_cap_per_qid_label: int = 2,
    val_cap_per_qid_label: int = 1,
) -> tuple[list[ITIExample], dict[str, Any]]:
    samples = load_consistency_samples(consistency_path)
    train_qids = load_id_split(train_qids_path)
    test_qids = load_id_split(test_qids_path)
    validate_disjoint_qid_splits(train_qids, test_qids)

    train_ids = set(train_qids.get("t", []) + train_qids.get("f", []))
    val_ids = set(test_qids.get("t", []) + test_qids.get("f", []))
    if train_ids & val_ids:
        raise ValueError("TriviaQA ITI requires disjoint QID splits")

    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    dedupe_seen: dict[tuple[str, str, int], set[str]] = defaultdict(set)
    skipped_duplicate_answers = 0

    for qid, inner in sorted(samples.items()):
        if qid in train_ids:
            split = "train"
            cap = train_cap_per_qid_label
        elif qid in val_ids:
            split = "val"
            cap = val_cap_per_qid_label
        else:
            continue

        question = inner["question"]
        responses = inner.get("responses", [])
        judges = inner.get("judges", [])
        if len(responses) != len(judges):
            continue

        per_label_counts: dict[int, int] = defaultdict(int)
        for idx, (answer, judge) in enumerate(zip(responses, judges, strict=False)):
            normalized = normalize_answer(answer)
            if not normalized:
                continue
            label = (
                LABEL_TRUE
                if judge == "true"
                else LABEL_FALSE
                if judge == "false"
                else None
            )
            if label is None:
                continue
            if normalized in dedupe_seen[(split, qid, label)]:
                skipped_duplicate_answers += 1
                continue
            if per_label_counts[label] >= cap:
                continue
            dedupe_seen[(split, qid, label)].add(normalized)
            grouped[(split, qid, label)].append(
                {
                    "answer": answer,
                    "normalized_answer": normalized,
                    "question": question,
                    "response_idx": idx,
                }
            )
            per_label_counts[label] += 1

    examples: list[ITIExample] = []
    per_qid_weights: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
    for (split, qid, label), items in sorted(grouped.items()):
        weight = 1.0 / len(items)
        label_name = "truthful" if label == LABEL_TRUE else "untruthful"
        per_qid_weights[qid][label_name] = {
            "count": len(items),
            "weight_per_example": weight,
        }
        for local_idx, item in enumerate(items):
            prompt_text, answer_positions = _encode_with_answer_positions(
                tokenizer,
                question=item["question"],
                answer=item["answer"],
            )
            examples.append(
                ITIExample(
                    example_id=f"{split}_{qid}_{label_name}_{local_idx}",
                    family="iti_triviaqa_transfer",
                    split=split,
                    qid=qid,
                    question=item["question"],
                    answer=item["answer"],
                    label=label,
                    weight=weight,
                    prompt_text=prompt_text,
                    answer_positions=answer_positions,
                    metadata={
                        "normalized_answer": item["normalized_answer"],
                        "response_idx": item["response_idx"],
                    },
                )
            )

    counts = {
        "train_truthful": sum(
            1 for ex in examples if ex.split == "train" and ex.label == LABEL_TRUE
        ),
        "train_untruthful": sum(
            1 for ex in examples if ex.split == "train" and ex.label == LABEL_FALSE
        ),
        "val_truthful": sum(
            1 for ex in examples if ex.split == "val" and ex.label == LABEL_TRUE
        ),
        "val_untruthful": sum(
            1 for ex in examples if ex.split == "val" and ex.label == LABEL_FALSE
        ),
    }
    metadata = {
        "family": "iti_triviaqa_transfer",
        "source_dataset": "triviaqa_consistency",
        "prompt_format": "Question: {question}\\nAnswer: {answer}",
        "answer_token_policy": "raw_prompt_answer_span",
        "split_fingerprint": {
            "train_qids": fingerprint_ids(sorted(train_ids)),
            "val_qids": fingerprint_ids(sorted(val_ids)),
        },
        "question_counts": {
            "train": len(train_ids),
            "val": len(val_ids),
        },
        "example_counts": counts,
        "unique_answer_strings": {
            "train": len(
                {normalize_answer(ex.answer) for ex in examples if ex.split == "train"}
            ),
            "val": len(
                {normalize_answer(ex.answer) for ex in examples if ex.split == "val"}
            ),
        },
        "deduplication": {
            "normalized_answers_within_split": True,
            "skipped_duplicate_answers": skipped_duplicate_answers,
            "train_cap_per_qid_label": train_cap_per_qid_label,
            "val_cap_per_qid_label": val_cap_per_qid_label,
        },
        "per_qid_weights": per_qid_weights,
    }
    return examples, metadata


def _bucket_answer(answer: str) -> int:
    n_words = max(1, len(normalize_answer(answer).split()))
    return min(n_words, 6)


def _sample_wrong_answer(
    *,
    rng: random.Random,
    bucketed_answers: dict[int, list[tuple[str, str, str]]],
    bucket: int,
    forbidden_qid: str,
    forbidden_answers: set[str],
) -> str:
    search_buckets = [bucket] + [b for b in sorted(bucketed_answers) if b != bucket]
    for current_bucket in search_buckets:
        pool = [
            answer
            for qid, answer, normalized_answer in bucketed_answers.get(
                current_bucket, []
            )
            if qid != forbidden_qid and normalized_answer not in forbidden_answers
        ]
        if pool:
            return rng.choice(pool)
    raise ValueError(f"No wrong-answer pool available for qid={forbidden_qid}")


def build_context_grounded_examples(
    *,
    tokenizer: Any,
    seed: int = 42,
    train_questions: int = 400,
    val_questions: int = 100,
    abstention_text: str = DEFAULT_ABSTENTION,
) -> tuple[list[ITIExample], dict[str, Any]]:
    train_ds = load_dataset("squad_v2", split="train")
    val_ds = load_dataset("squad_v2", split="validation")
    rng = random.Random(seed)

    def split_rows(rows: list[dict[str, Any]], n_total: int) -> list[dict[str, Any]]:
        answerable = [row for row in rows if row["answers"]["text"]]
        impossible = [row for row in rows if not row["answers"]["text"]]
        if n_total % 2 != 0:
            raise ValueError(
                "Context question counts must be even for 50/50 stratification"
            )
        half = n_total // 2
        return rng.sample(answerable, half) + rng.sample(impossible, half)

    sampled_train = split_rows(list(train_ds), train_questions)
    sampled_val = split_rows(list(val_ds), val_questions)

    answer_pool_rows = [
        row for row in list(train_ds) + list(val_ds) if row["answers"]["text"]
    ]
    bucketed_answers: dict[int, list[tuple[str, str, str]]] = defaultdict(list)
    for row in answer_pool_rows:
        answer = row["answers"]["text"][0]
        bucketed_answers[_bucket_answer(answer)].append(
            (row["id"], answer, normalize_answer(answer))
        )

    examples: list[ITIExample] = []
    for split, rows in (("train", sampled_train), ("val", sampled_val)):
        for row in rows:
            qid = row["id"]
            question = row["question"]
            context = row["context"]
            gold_answers = row["answers"]["text"]
            answerable = bool(gold_answers)
            truthful_answer = gold_answers[0] if answerable else abstention_text
            bucket = _bucket_answer(truthful_answer) if answerable else 1
            forbidden_answers = (
                {normalize_answer(answer) for answer in gold_answers}
                if answerable
                else set()
            )
            wrong_answer = _sample_wrong_answer(
                rng=rng,
                bucketed_answers=bucketed_answers,
                bucket=bucket,
                forbidden_qid=qid,
                forbidden_answers=forbidden_answers,
            )
            for label, answer in (
                (LABEL_TRUE, truthful_answer),
                (LABEL_FALSE, wrong_answer),
            ):
                prompt_text, answer_positions = _encode_with_answer_positions(
                    tokenizer,
                    question=question,
                    answer=answer,
                    context=context,
                )
                label_name = "truthful" if label == LABEL_TRUE else "untruthful"
                examples.append(
                    ITIExample(
                        example_id=f"{split}_{qid}_{label_name}",
                        family="iti_context_grounded",
                        split=split,
                        qid=qid,
                        question=question,
                        answer=answer,
                        label=label,
                        weight=1.0,
                        prompt_text=prompt_text,
                        answer_positions=answer_positions,
                        metadata={
                            "title": row.get("title"),
                            "answerable": answerable,
                            "source_answer": truthful_answer,
                            "wrong_answer": wrong_answer,
                        },
                    )
                )

    metadata = {
        "family": "iti_context_grounded",
        "source_dataset": "squad_v2",
        "prompt_format": "Context: {context}\\n\\nQuestion: {question}\\nAnswer: {answer}",
        "answer_token_policy": "raw_prompt_answer_span",
        "sampling": {
            "seed": seed,
            "train_questions": train_questions,
            "val_questions": val_questions,
            "abstention_text": abstention_text,
        },
        "question_counts": {
            "train": train_questions,
            "val": val_questions,
        },
        "label_balance": {
            "train_answerable_questions": sum(
                1 for row in sampled_train if row["answers"]["text"]
            ),
            "train_impossible_questions": sum(
                1 for row in sampled_train if not row["answers"]["text"]
            ),
            "val_answerable_questions": sum(
                1 for row in sampled_val if row["answers"]["text"]
            ),
            "val_impossible_questions": sum(
                1 for row in sampled_val if not row["answers"]["text"]
            ),
        },
        "split_fingerprint": {
            "train_qids": fingerprint_ids([row["id"] for row in sampled_train]),
            "val_qids": fingerprint_ids([row["id"] for row in sampled_val]),
        },
    }
    return examples, metadata


def _qa_prompt_paper(question: str, answer: str) -> tuple[str, str]:
    """Paper-faithful prompt: ``Q: {q}\\nA: {a}`` (Li et al. 2306.03341)."""
    prefix = f"Q: {question}\nA:"
    full = f"{prefix} {answer}".rstrip()
    return prefix, full


def _longest_common_prefix_len(a: list[int], b: list[int]) -> int:
    n = 0
    for x, y in zip(a, b, strict=False):
        if x != y:
            break
        n += 1
    return n


def _chat_prompt_with_answer_positions(
    tokenizer: Any,
    *,
    question: str,
    answer: str,
) -> tuple[str, tuple[int, ...], tuple[int, ...], dict[str, Any]]:
    """Render chat-formatted QA and return assistant-content token positions.

    Positions are computed on the fully-rendered chat sequence while excluding
    assistant control/suffix tokens (e.g. end-of-turn markers).
    """
    prompt_messages = [{"role": "user", "content": question}]
    full_messages = [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]
    empty_assistant_messages = [
        {"role": "user", "content": question},
        {"role": "assistant", "content": ""},
    ]

    prompt_ids = list(
        tokenizer.apply_chat_template(
            prompt_messages, tokenize=True, add_generation_prompt=True
        )
    )
    full_ids = list(
        tokenizer.apply_chat_template(
            full_messages, tokenize=True, add_generation_prompt=False
        )
    )
    full_text = str(
        tokenizer.apply_chat_template(
            full_messages, tokenize=False, add_generation_prompt=False
        )
    )
    empty_ids = list(
        tokenizer.apply_chat_template(
            empty_assistant_messages, tokenize=True, add_generation_prompt=False
        )
    )

    prefix_len = _longest_common_prefix_len(prompt_ids, full_ids)
    if prefix_len != len(prompt_ids):
        raise ValueError(
            "Chat-template mismatch: prompt IDs are not a full prefix of full IDs "
            f"(prefix_len={prefix_len}, prompt_len={len(prompt_ids)})."
        )

    suffix_ids = empty_ids[prefix_len:]
    tail_ids = full_ids[prefix_len:]
    if suffix_ids and tail_ids[-len(suffix_ids) :] == suffix_ids:
        content_tail = tail_ids[: -len(suffix_ids)]
    else:
        content_tail = tail_ids

    if not content_tail:
        raise ValueError(
            "Chat-template answer encoding produced no assistant-content tokens for "
            f"question={question!r}, answer={answer!r}"
        )

    answer_start = prefix_len
    answer_end = answer_start + len(content_tail)
    answer_positions = tuple(range(answer_start, answer_end))
    return (
        full_text,
        answer_positions,
        tuple(full_ids),
        {
            "prompt_len_tokens": len(prompt_ids),
            "full_len_tokens": len(full_ids),
            "assistant_suffix_len_tokens": len(suffix_ids),
            "assistant_content_len_tokens": len(content_tail),
        },
    )


TRUTHFULQA_FAMILIES = (
    "iti_truthfulqa_paperfaithful",
    "iti_truthfulqa_exploratory",
    "iti_truthfulqa_modernized",
)


def build_truthfulqa_paper_examples(
    *,
    tokenizer: Any,
    fold_path: Path,
    csv_path: Path = Path("data/benchmarks/TruthfulQA.csv"),
    family: str = "iti_truthfulqa_paperfaithful",
) -> tuple[list[ITIExample], dict[str, Any]]:
    """Build QA pairs from TruthfulQA following the paper's 2-fold CV protocol.

    Uses a fold file (from ``build_truthfulqa_splits.py``) that specifies which
    questions belong to train, val, and test.  **Test-fold questions are
    excluded entirely** so they never contaminate directions or ranking.

    TruthfulQA families share this builder for split logic and provenance
    handling. Prompt rendering differs by family:
    - ``iti_truthfulqa_paperfaithful`` / ``iti_truthfulqa_exploratory``:
      paper raw-text prompt ``Q: {question}\\nA: {answer}``.
    - ``iti_truthfulqa_modernized``: model chat-template prompt with
      assistant-content token spans.

    - Data: official TruthfulQA CSV (canonical source, 817 questions, dev-fold only)
    - Each Correct Answer → truthful pair, each Incorrect Answer → untruthful pair
    - Split assignment from fold file (train/val within dev)
    - Prompt: paper raw-text or modernized chat-template (family-dependent)
    - Token positions: family-dependent (paper span vs assistant-content span)
    """
    fold = json.loads(fold_path.read_text(encoding="utf-8"))
    canonical_path = Path(fold["canonical_manifest"])
    canonical = json.loads(canonical_path.read_text(encoding="utf-8"))

    csv_df = pd.read_csv(str(csv_path), encoding="utf-8-sig")
    max_csv_idx = max(int(q["csv_idx"]) for q in canonical["questions"])
    assert len(csv_df) > max_csv_idx, (
        f"CSV has {len(csv_df)} rows but canonical manifest references csv_idx={max_csv_idx}"
    )

    train_ids = set(fold["dev"]["train"])
    val_ids = set(fold["dev"]["val"])
    test_ids = set(fold["test"])

    dev_ids = train_ids | val_ids
    overlap = dev_ids & test_ids
    assert not overlap, f"FATAL: {len(overlap)} questions in both dev and test"

    examples: list[ITIExample] = []
    n_skipped_test = 0
    for q_entry in canonical["questions"]:
        csv_idx: int = int(q_entry["csv_idx"])
        stable_id: str = str(q_entry["stable_id"])

        # Exclude test-fold questions entirely
        if stable_id in test_ids:
            n_skipped_test += 1
            continue

        row = csv_df.iloc[csv_idx]
        question = str(row["Question"])
        qid = stable_id
        split = "val" if stable_id in val_ids else "train"

        for answer in parse_semicolon_answers(row["Correct Answers"]):
            if family == "iti_truthfulqa_modernized":
                full, answer_positions, full_ids, prompt_meta = (
                    _chat_prompt_with_answer_positions(
                        tokenizer,
                        question=question,
                        answer=answer,
                    )
                )
            else:
                prefix, full = _qa_prompt_paper(question, answer)
                prefix_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
                full_ids = tokenizer(full, add_special_tokens=False)["input_ids"]
                if len(full_ids) <= len(prefix_ids):
                    continue
                answer_positions = tuple(range(len(prefix_ids), len(full_ids)))
                prompt_meta = {}
            examples.append(
                ITIExample(
                    example_id=f"{split}_{qid}_truthful_{len(examples)}",
                    family=family,
                    split=split,
                    qid=qid,
                    question=question,
                    answer=answer,
                    label=LABEL_TRUE,
                    weight=1.0,
                    prompt_text=full,
                    answer_positions=answer_positions,
                    metadata={
                        "source": "csv",
                        "csv_idx": csv_idx,
                        **prompt_meta,
                    },
                    prompt_input_ids=tuple(full_ids),
                )
            )

        for answer in parse_semicolon_answers(row["Incorrect Answers"]):
            if family == "iti_truthfulqa_modernized":
                full, answer_positions, full_ids, prompt_meta = (
                    _chat_prompt_with_answer_positions(
                        tokenizer,
                        question=question,
                        answer=answer,
                    )
                )
            else:
                prefix, full = _qa_prompt_paper(question, answer)
                prefix_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
                full_ids = tokenizer(full, add_special_tokens=False)["input_ids"]
                if len(full_ids) <= len(prefix_ids):
                    continue
                answer_positions = tuple(range(len(prefix_ids), len(full_ids)))
                prompt_meta = {}
            examples.append(
                ITIExample(
                    example_id=f"{split}_{qid}_untruthful_{len(examples)}",
                    family=family,
                    split=split,
                    qid=qid,
                    question=question,
                    answer=answer,
                    label=LABEL_FALSE,
                    weight=1.0,
                    prompt_text=full,
                    answer_positions=answer_positions,
                    metadata={
                        "source": "csv",
                        "csv_idx": csv_idx,
                        **prompt_meta,
                    },
                    prompt_input_ids=tuple(full_ids),
                )
            )

    print(f"  Skipped {n_skipped_test} test-fold questions")

    train_qids = sorted(train_ids & {ex.qid for ex in examples})
    val_qids = sorted(val_ids & {ex.qid for ex in examples})
    test_qid_list = sorted(test_ids)
    dev_qid_list = sorted(train_ids | val_ids)

    metadata = {
        "family": family,
        "source_dataset": str(csv_path),
        "protocol": "li_et_al_2306.03341v6_2fold",
        "prompt_format": (
            "chat_template(user->assistant)"
            if family == "iti_truthfulqa_modernized"
            else "Q: {question}\\nA: {answer}"
        ),
        "fold_path": str(fold_path),
        "fold_id": fold["fold"],
        "fold_fingerprint": fold.get("canonical_fingerprint", ""),
        "direction_fit_scope": "dev_only",
        "direction_fit_source": "dev_only",
        "sigma_fit_source": "dev_only",
        "head_ranking_metric": (
            "auroc" if family == "iti_truthfulqa_modernized" else "val_accuracy"
        ),
        "answer_token_policy": (
            "assistant_content_only"
            if family == "iti_truthfulqa_modernized"
            else "raw_prompt_answer_span"
        ),
        "question_ids_train": sorted(train_qids),
        "question_ids_val": sorted(val_qids),
        "question_ids_dev": dev_qid_list,
        "question_ids_test": test_qid_list,
        "truthfulqa_manifest_fingerprint": canonical.get("fingerprint", ""),
        "fold_file_fingerprint": fingerprint_ids(
            fold["dev"]["train"] + fold["dev"]["val"] + fold["test"]
        ),
        "question_counts": {
            "total": len(canonical["questions"]),
            "dev": len(train_ids) + len(val_ids),
            "train": len(train_qids),
            "val": len(val_qids),
            "test_excluded": n_skipped_test,
        },
        "split_fingerprint": {
            "train_qids": fingerprint_ids(train_qids),
            "val_qids": fingerprint_ids(val_qids),
            "test_qids": fingerprint_ids(test_qid_list),
            "dev_qids": fingerprint_ids(dev_qid_list),
        },
    }
    return examples, metadata


def _get_text_config(model: torch.nn.Module) -> Any:
    return getattr(model.config, "text_config", model.config)


def collect_pre_o_proj_head_activations(
    model: torch.nn.Module,
    tokenizer: Any,
    examples: list[ITIExample],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    cfg = _get_text_config(model)
    n_layers = int(getattr(cfg, "num_hidden_layers"))
    n_heads = int(getattr(cfg, "num_attention_heads"))
    head_dim = int(getattr(_get_decoder_layers(model)[0].self_attn, "head_dim"))

    storage = {
        summary: torch.empty(
            (len(examples), n_layers, n_heads, head_dim),
            dtype=torch.float16,
            device="cpu",
        )
        for summary in POSITION_SUMMARIES
    }
    state: dict[str, Any] = {"example_idx": None, "answer_positions": ()}
    hooks = []

    def make_hook(layer_idx: int):
        def hook_fn(module, args):
            x = args[0]
            if x.shape[0] != 1:
                raise ValueError("ITI extraction currently expects batch size 1")
            positions = state["answer_positions"]
            head_view = x[0].detach().float().cpu().view(x.shape[1], n_heads, head_dim)
            first_idx = positions[0]
            last_idx = positions[-1]
            storage["first_answer_token"][state["example_idx"], layer_idx] = head_view[
                first_idx
            ].to(torch.float16)
            storage["last_answer_token"][state["example_idx"], layer_idx] = head_view[
                last_idx
            ].to(torch.float16)
            storage["mean_answer_span"][state["example_idx"], layer_idx] = (
                head_view[list(positions)].mean(dim=0).to(torch.float16)
            )
            return args

        return hook_fn

    decoder_layers = _get_decoder_layers(model)
    for layer_idx in range(n_layers):
        hooks.append(
            decoder_layers[layer_idx].self_attn.o_proj.register_forward_pre_hook(
                make_hook(layer_idx)
            )
        )

    try:
        for idx, example in enumerate(
            tqdm(examples, desc="Collecting ITI head activations")
        ):
            if example.prompt_input_ids is not None:
                input_ids = torch.tensor(
                    [list(example.prompt_input_ids)],
                    dtype=torch.long,
                    device=device,
                )
            else:
                input_ids = tokenizer(
                    example.prompt_text, return_tensors="pt", add_special_tokens=False
                )["input_ids"].to(device)
            state["example_idx"] = idx
            state["answer_positions"] = example.answer_positions
            with torch.no_grad():
                model(input_ids)
    finally:
        for hook in hooks:
            hook.remove()

    return storage


def _fit_probe_metrics(
    X_train: np.ndarray,
    y_train: np.ndarray,
    w_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> tuple[float, float, float]:
    probe = LogisticRegression(
        random_state=0,
        max_iter=2000,
        solver="liblinear",
    )
    try:
        probe.fit(X_train, y_train, sample_weight=w_train)
        val_probs = probe.predict_proba(X_val)[:, 1]
        val_preds = (val_probs >= 0.5).astype(int)
        auroc = float(roc_auc_score(y_val, val_probs))
        balanced_acc = float(balanced_accuracy_score(y_val, val_preds))
        val_acc = float(accuracy_score(y_val, val_preds))
    except Exception:
        auroc = 0.5
        balanced_acc = 0.5
        val_acc = 0.5
    return auroc, balanced_acc, val_acc


def rank_heads(
    activations: dict[str, torch.Tensor],
    examples: list[ITIExample],
    *,
    position_summaries: tuple[str, ...] = POSITION_SUMMARIES,
    ranking_primary: str = "auroc",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    train_indices = [i for i, ex in enumerate(examples) if ex.split == "train"]
    val_indices = [i for i, ex in enumerate(examples) if ex.split == "val"]
    y_train = np.array([examples[i].label for i in train_indices], dtype=np.int64)
    y_val = np.array([examples[i].label for i in val_indices], dtype=np.int64)
    w_train = np.array([examples[i].weight for i in train_indices], dtype=np.float64)
    qid_coverage = {
        "train": len({examples[i].qid for i in train_indices}),
        "val": len({examples[i].qid for i in val_indices}),
    }

    sample_tensor = next(iter(activations.values()))
    _, n_layers, n_heads, head_dim = sample_tensor.shape
    ranked_heads: list[dict[str, Any]] = []
    metrics_summary: dict[str, Any] = {"position_summaries": list(position_summaries)}

    def _rank_key(entry: dict[str, Any]) -> tuple[float, ...]:
        if ranking_primary == "balanced_accuracy":
            return (
                float(entry["balanced_accuracy"]),
                float(entry["auroc"]),
            )
        if ranking_primary == "val_accuracy":
            return (
                float(entry["val_accuracy"]),
                float(entry["auroc"]),
            )
        return (
            float(entry["auroc"]),
            float(entry["balanced_accuracy"]),
        )

    for layer_idx in range(n_layers):
        for head_idx in range(n_heads):
            best_entry: dict[str, Any] | None = None
            metrics_by_summary: dict[str, Any] = {}
            for summary_name in position_summaries:
                X_train = (
                    activations[summary_name][train_indices, layer_idx, head_idx]
                    .float()
                    .numpy()
                )
                X_val = (
                    activations[summary_name][val_indices, layer_idx, head_idx]
                    .float()
                    .numpy()
                )
                auroc, balanced_accuracy, val_accuracy = _fit_probe_metrics(
                    X_train, y_train, w_train, X_val, y_val
                )
                entry = {
                    "layer": layer_idx,
                    "head": head_idx,
                    "position_summary": summary_name,
                    "auroc": round(auroc, 6),
                    "balanced_accuracy": round(balanced_accuracy, 6),
                    "val_accuracy": round(val_accuracy, 6),
                    "qid_coverage": qid_coverage,
                }
                metrics_by_summary[summary_name] = {
                    key: value
                    for key, value in entry.items()
                    if key not in {"layer", "head"}
                }
                if best_entry is None:
                    best_entry = entry
                    continue
                if _rank_key(entry) > _rank_key(best_entry):
                    best_entry = entry

            assert best_entry is not None
            best_entry["metrics_by_summary"] = metrics_by_summary
            ranked_heads.append(best_entry)

    ranked_heads.sort(
        key=lambda item: (
            *_rank_key(item),
            -item["layer"],
            -item["head"],
        ),
        reverse=True,
    )
    metrics_summary["top5"] = [
        {
            key: item[key]
            for key in (
                "layer",
                "head",
                "position_summary",
                "auroc",
                "balanced_accuracy",
                "val_accuracy",
            )
        }
        for item in ranked_heads[:5]
    ]
    metrics_summary["head_dim"] = head_dim
    return ranked_heads, metrics_summary


def _encode_continuation_tokens(tokenizer: Any, text: str) -> tuple[int, ...]:
    continuation = text if text[:1].isspace() else f" {text}"
    token_ids = tokenizer(continuation, add_special_tokens=False)["input_ids"]
    if not token_ids:
        token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if not token_ids:
        raise ValueError(f"Continuation has no tokens: {text!r}")
    return tuple(int(token_id) for token_id in token_ids)


def _causal_branch_logprob_and_attribution(
    *,
    model: torch.nn.Module,
    prompt_ids: tuple[int, ...],
    continuation_ids: tuple[int, ...],
    device: torch.device,
    n_layers: int,
    n_heads: int,
    head_dim: int,
) -> tuple[float, np.ndarray]:
    prompt_pos = len(prompt_ids) - 1
    full_ids = torch.tensor(
        [list(prompt_ids + continuation_ids)],
        dtype=torch.long,
        device=device,
    )

    decoder_layers = _get_decoder_layers(model)
    hook_state: dict[int, torch.Tensor] = {}
    hooks = []

    def make_hook(layer_idx: int):
        def hook_fn(module, args):
            x = args[0]
            if x.ndim != 3 or x.shape[0] != 1:
                return args
            if prompt_pos >= x.shape[1]:
                raise ValueError(
                    f"Prompt position {prompt_pos} out of range for seq_len={x.shape[1]}"
                )
            hook_state[layer_idx] = x
            return args

        return hook_fn

    for layer_idx in range(n_layers):
        hooks.append(
            decoder_layers[layer_idx].self_attn.o_proj.register_forward_pre_hook(
                make_hook(layer_idx)
            )
        )

    try:
        model.zero_grad(set_to_none=True)
        logits = model(full_ids).logits
        pred_logits = logits[:, :-1, :]
        target_ids = full_ids[:, 1:]
        start = len(prompt_ids) - 1
        end = start + len(continuation_ids)
        token_logprobs = torch.log_softmax(pred_logits[:, start:end, :], dim=-1)
        gathered = token_logprobs.gather(
            dim=-1,
            index=target_ids[:, start:end].unsqueeze(-1),
        ).squeeze(-1)
        total_logprob = gathered.sum()
        if not torch.isfinite(total_logprob):
            raise ValueError("Non-finite causal branch objective")
        layer_inputs: list[torch.Tensor] = []
        for layer_idx in range(n_layers):
            layer_input = hook_state.get(layer_idx)
            if layer_input is None:
                raise ValueError(
                    "Missing causal hook captures for one or more decoder layers"
                )
            layer_inputs.append(layer_input)
        grads = torch.autograd.grad(
            total_logprob,
            layer_inputs,
            allow_unused=True,
        )

        attribution = np.zeros((n_layers, n_heads), dtype=np.float64)
        for layer_idx in range(n_layers):
            layer_input = layer_inputs[layer_idx]
            layer_grad = grads[layer_idx]
            if layer_grad is None:
                raise ValueError(
                    f"Missing prompt activation gradient for causal branch at layer {layer_idx}"
                )
            prompt_grad = layer_grad[:, prompt_pos, :].reshape(1, n_heads, head_dim)
            prompt_act = layer_input.detach()[:, prompt_pos, :].reshape(
                1, n_heads, head_dim
            )
            contrib = (prompt_grad[0] * prompt_act[0]).sum(dim=-1)
            contrib_np = contrib.detach().float().cpu().numpy().astype(np.float64)
            if not np.isfinite(contrib_np).all():
                raise ValueError(
                    f"NaN/Inf causal attribution detected at layer {layer_idx}"
                )
            attribution[layer_idx] = contrib_np
        return float(total_logprob.item()), attribution
    finally:
        for hook in hooks:
            hook.remove()


def rank_heads_causal(
    *,
    model: torch.nn.Module,
    tokenizer: Any,
    causal_pairs: list[D7CausalPair],
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not causal_pairs:
        raise ValueError("Causal ranking requires at least one extraction pair")

    cfg = _get_text_config(model)
    n_layers = int(getattr(cfg, "num_hidden_layers"))
    n_heads = int(getattr(cfg, "num_attention_heads"))
    decoder_layers = _get_decoder_layers(model)
    head_dim = int(getattr(decoder_layers[0].self_attn, "head_dim"))

    score_accumulator = np.zeros((n_layers, n_heads), dtype=np.float64)
    objective_deltas: list[float] = []

    for pair in tqdm(causal_pairs, desc="Scoring causal head attributions"):
        prompt_ids = tuple(
            int(token_id)
            for token_id in tokenizer(
                pair.harmful_prompt,
                add_special_tokens=False,
            )["input_ids"]
        )
        if not prompt_ids:
            raise ValueError(f"Pair {pair.pair_id} has empty harmful prompt encoding")

        benign_ids = _encode_continuation_tokens(tokenizer, pair.benign_response_target)
        harmful_ids = _encode_continuation_tokens(
            tokenizer,
            pair.harmful_response_target,
        )

        benign_logprob, benign_attr = _causal_branch_logprob_and_attribution(
            model=model,
            prompt_ids=prompt_ids,
            continuation_ids=benign_ids,
            device=device,
            n_layers=n_layers,
            n_heads=n_heads,
            head_dim=head_dim,
        )
        harmful_logprob, harmful_attr = _causal_branch_logprob_and_attribution(
            model=model,
            prompt_ids=prompt_ids,
            continuation_ids=harmful_ids,
            device=device,
            n_layers=n_layers,
            n_heads=n_heads,
            head_dim=head_dim,
        )
        delta_attr = benign_attr - harmful_attr
        if not np.isfinite(delta_attr).all():
            raise ValueError(f"NaN/Inf attribution delta for pair {pair.pair_id}")
        score_accumulator += delta_attr
        objective_deltas.append(benign_logprob - harmful_logprob)

    mean_scores = score_accumulator / len(causal_pairs)
    ranked_heads: list[dict[str, Any]] = []
    qid_coverage = {
        "train": len({pair.pair_id for pair in causal_pairs if pair.split == "train"}),
        "val": len({pair.pair_id for pair in causal_pairs if pair.split == "val"}),
    }
    for layer_idx in range(n_layers):
        for head_idx in range(n_heads):
            score = float(mean_scores[layer_idx, head_idx])
            ranked_heads.append(
                {
                    "layer": layer_idx,
                    "head": head_idx,
                    "position_summary": "last_answer_token",
                    "causal_effect": round(score, 6),
                    "auroc": 0.5,
                    "balanced_accuracy": 0.5,
                    "val_accuracy": 0.5,
                    "qid_coverage": qid_coverage,
                    "metrics_by_summary": {
                        "last_answer_token": {
                            "causal_effect": round(score, 6),
                            "auroc": 0.5,
                            "balanced_accuracy": 0.5,
                            "val_accuracy": 0.5,
                            "qid_coverage": qid_coverage,
                        }
                    },
                }
            )

    ranked_heads.sort(
        key=lambda item: (float(item["causal_effect"]), -item["layer"], -item["head"]),
        reverse=True,
    )
    objective_arr = np.array(objective_deltas, dtype=np.float64)
    ranking_metadata = {
        "position_summaries": ["last_answer_token"],
        "head_dim": head_dim,
        "objective_name": (
            "log_p(benign_response_target|harmful_prompt) - "
            "log_p(harmful_response_target|harmful_prompt)"
        ),
        "objective_summary": {
            "n_pairs": len(causal_pairs),
            "mean_delta": round(float(objective_arr.mean()), 6),
            "std_delta": round(float(objective_arr.std()), 6),
            "min_delta": round(float(objective_arr.min()), 6),
            "max_delta": round(float(objective_arr.max()), 6),
        },
        "top5": [
            {
                "layer": item["layer"],
                "head": item["head"],
                "position_summary": item["position_summary"],
                "causal_effect": item["causal_effect"],
                "auroc": item["auroc"],
                "balanced_accuracy": item["balanced_accuracy"],
                "val_accuracy": item["val_accuracy"],
            }
            for item in ranked_heads[:5]
        ],
    }
    return ranked_heads, ranking_metadata


def build_artifact(
    *,
    family: str,
    model: torch.nn.Module,
    ranked_heads: list[dict[str, Any]],
) -> dict[str, Any]:
    cfg = _get_text_config(model)
    decoder_layers = _get_decoder_layers(model)
    head_dim = int(getattr(decoder_layers[0].self_attn, "head_dim"))
    return {
        "family": family,
        "steering_direction_type": "mass_mean",
        "n_layers": int(getattr(cfg, "num_hidden_layers")),
        "n_attention_heads": int(getattr(cfg, "num_attention_heads")),
        "head_dim": head_dim,
        "ranked_heads": ranked_heads,
    }


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir = (
        Path(args.output_dir) if args.output_dir else default_output_dir(args.family)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = output_dir / "iti_heads.pt"
    metadata_path = output_dir / "extraction_metadata.json"

    provenance_handle = start_run_provenance(
        args,
        primary_target=str(output_dir),
        output_targets=[str(artifact_path), str(metadata_path)],
        primary_target_is_dir=True,
    )
    provenance_status = "completed"
    provenance_extra: dict[str, Any] = {}

    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        causal_pairs: list[D7CausalPair] = []
        if args.family == "iti_triviaqa_transfer":
            examples, family_metadata = build_triviaqa_transfer_examples(
                consistency_path=args.consistency_path,
                train_qids_path=args.train_qids_path,
                test_qids_path=args.test_qids_path,
                tokenizer=tokenizer,
                train_cap_per_qid_label=args.train_cap_per_qid_label,
                val_cap_per_qid_label=args.val_cap_per_qid_label,
            )
        elif args.family in TRUTHFULQA_FAMILIES:
            if args.fold_path is None:
                raise ValueError(
                    f"--fold_path is required for {args.family}. "
                    "Generate fold files with: "
                    "uv run python scripts/build_truthfulqa_splits.py --seed 42"
                )
            examples, family_metadata = build_truthfulqa_paper_examples(
                tokenizer=tokenizer,
                fold_path=args.fold_path,
                csv_path=args.csv_path,
                family=args.family,
            )
        elif args.family in REFUSAL_FAMILIES:
            examples, causal_pairs, family_metadata = build_refusal_d7_examples(
                tokenizer=tokenizer,
                manifest_path=args.d7_manifest_path,
                family=args.family,
            )
        else:
            examples, family_metadata = build_context_grounded_examples(
                tokenizer=tokenizer,
                seed=args.seed,
                train_questions=args.context_train_questions,
                val_questions=args.context_val_questions,
            )

        if args.family == "iti_refusal_causal":
            rank_primary = "causal_effect"
            rank_position_summaries = ("last_answer_token",)
            position_policy = "last_answer_token"
        else:
            rank_primary, rank_position_summaries, position_policy = (
                resolve_selector_policy(
                    family=args.family,
                    ranking_metric_override=args.ranking_metric_override,
                    position_policy_override=args.position_policy_override,
                )
            )

        print(f"Loaded {len(examples)} {args.family} examples")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, dtype=torch.bfloat16, device_map=args.device_map
        )
        model.eval()
        device = next(model.parameters()).device

        activations = collect_pre_o_proj_head_activations(
            model=model,
            tokenizer=tokenizer,
            examples=examples,
            device=device,
        )
        if args.family == "iti_refusal_causal":
            ranked_heads, ranking_metadata = rank_heads_causal(
                model=model,
                tokenizer=tokenizer,
                causal_pairs=causal_pairs,
                device=device,
            )
        else:
            ranked_heads, ranking_metadata = rank_heads(
                activations,
                examples,
                position_summaries=rank_position_summaries,
                ranking_primary=rank_primary,
            )

        use_dev_direction_fit = (
            args.family in TRUTHFULQA_FAMILIES or args.family in REFUSAL_FAMILIES
        )
        if use_dev_direction_fit:
            direction_fit_indices = list(range(len(examples)))
            direction_source = "dev_data"
        else:
            direction_fit_indices = [
                i for i, ex in enumerate(examples) if ex.split == "train"
            ]
            direction_source = "train_data"
        ranked_heads = compute_head_directions(
            ranked_heads,
            activations,
            examples,
            direction_fit_indices,
            direction_source=direction_source,
        )
        print(f"Computed directions from {direction_source}")
        artifact = build_artifact(
            family=args.family,
            model=model,
            ranked_heads=ranked_heads,
        )
        torch.save(artifact, artifact_path)

        metadata = {
            "family": args.family,
            "artifact_family": args.family,
            "model_path": args.model_path,
            "source_dataset": family_metadata["source_dataset"],
            "git_sha": get_git_sha(),
            "head_ranking_metric": rank_primary,
            "ranking_metric": rank_primary,
            "position_policy": position_policy,
            "prompt_format": family_metadata.get("prompt_format"),
            "answer_token_policy": family_metadata.get(
                "answer_token_policy", "raw_prompt_answer_span"
            ),
            "direction_fit_scope": family_metadata.get(
                "direction_fit_scope", "train_only"
            ),
            "direction_fit_source": family_metadata.get(
                "direction_fit_source", "train_only"
            ),
            "sigma_fit_source": family_metadata.get("sigma_fit_source", "train_only"),
            "fold_id": family_metadata.get("fold_id"),
            "fold_path": family_metadata.get("fold_path"),
            "fold_fingerprint": family_metadata.get("fold_fingerprint"),
            "truthfulqa_manifest_fingerprint": family_metadata.get(
                "truthfulqa_manifest_fingerprint"
            ),
            "fold_file_fingerprint": family_metadata.get("fold_file_fingerprint"),
            "d7_manifest_path": family_metadata.get("manifest_path"),
            "question_ids_train": family_metadata.get("question_ids_train", []),
            "question_ids_val": family_metadata.get("question_ids_val", []),
            "question_ids_dev": family_metadata.get("question_ids_dev", []),
            "question_ids_test": family_metadata.get("question_ids_test", []),
            "paths": {
                "artifact": str(artifact_path),
                "metadata": str(metadata_path),
            },
            "split_fingerprints": family_metadata.get("split_fingerprint", {}),
            "question_counts": family_metadata.get("question_counts", {}),
            "example_counts": {
                "train": sum(1 for ex in examples if ex.split == "train"),
                "val": sum(1 for ex in examples if ex.split == "val"),
            },
            "label_counts": {
                "train_truthful": sum(
                    1
                    for ex in examples
                    if ex.split == "train" and ex.label == LABEL_TRUE
                ),
                "train_untruthful": sum(
                    1
                    for ex in examples
                    if ex.split == "train" and ex.label == LABEL_FALSE
                ),
                "val_truthful": sum(
                    1 for ex in examples if ex.split == "val" and ex.label == LABEL_TRUE
                ),
                "val_untruthful": sum(
                    1
                    for ex in examples
                    if ex.split == "val" and ex.label == LABEL_FALSE
                ),
            },
            "family_metadata": family_metadata,
            "position_summaries": list(rank_position_summaries),
            "ranking_primary": rank_primary,
            "selector_overrides": {
                "ranking_metric_override": args.ranking_metric_override,
                "position_policy_override": args.position_policy_override,
            },
            "ranking": ranking_metadata,
            "selected_head_manifest": [
                {
                    "rank": idx + 1,
                    "layer": head["layer"],
                    "head": head["head"],
                    "position_summary": head["position_summary"],
                    "auroc": head["auroc"],
                    "balanced_accuracy": head["balanced_accuracy"],
                    "causal_effect": head.get("causal_effect"),
                    "sigma": head["sigma"],
                }
                for idx, head in enumerate(ranked_heads[:32])
            ],
            "steering_direction_type": "mass_mean",
            "direction_type": "mass_mean",
            "family_fingerprint": {
                "artifact_family": args.family,
                "source_dataset": family_metadata["source_dataset"],
                "ranking_metric": rank_primary,
                "position_policy": position_policy,
                "answer_token_policy": family_metadata.get(
                    "answer_token_policy", "raw_prompt_answer_span"
                ),
                "direction_type": "mass_mean",
            },
        }
        metadata_path.write_text(json_dumps(metadata), encoding="utf-8")

        # Post-build leakage audit — abort on overlap, embed in provenance
        split_audit = audit_split_leakage(metadata)
        metadata["split_audit"] = split_audit
        metadata_path.write_text(json_dumps(metadata), encoding="utf-8")

        provenance_extra["record_count"] = len(examples)
        provenance_extra["family"] = args.family
        provenance_extra["top_head"] = metadata["selected_head_manifest"][0]
        provenance_extra["split_audit"] = split_audit
        print(f"Saved ITI artifact to {artifact_path}")
        print(f"Saved metadata to {metadata_path}")
    except BaseException as exc:
        provenance_status = provenance_status_for_exception(exc)
        provenance_extra["error"] = provenance_error_message(exc)
        raise
    finally:
        finish_run_provenance(provenance_handle, provenance_status, provenance_extra)


if __name__ == "__main__":
    main()
