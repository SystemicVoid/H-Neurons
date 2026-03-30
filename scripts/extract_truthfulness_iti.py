"""Extract head-level ITI artifacts for closed-book and grounded truthfulness.

This script builds one of two explicit artifact families:

- ``iti_triviaqa_transfer``: closed-book factuality transfer from TriviaQA
  consistency responses.
- ``iti_context_grounded``: matched context-grounded answering + abstention
  from SQuAD-v2.

The extraction surface is the tensor immediately before ``self_attn.o_proj``.
For each physical attention head we evaluate three answer-span summaries
(``first_answer_token``, ``mean_answer_span``, ``last_answer_token``), rank the
head by probe AUROC first and balanced accuracy second, and store a truthful-
oriented mass-mean direction for decode-time steering.
"""

from __future__ import annotations

import argparse
import hashlib
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from build_truthfulness_contrastive import (
    load_consistency_samples,
    load_id_split,
    validate_disjoint_qid_splits,
)
from intervene_iti import _get_decoder_layers
from utils import (
    finish_run_provenance,
    get_git_sha,
    json_dumps,
    normalize_answer,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract head-level truthfulness ITI")
    parser.add_argument(
        "--family",
        type=str,
        required=True,
        choices=["iti_triviaqa_transfer", "iti_context_grounded"],
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
    return parser.parse_args()


def default_output_dir(family: str) -> Path:
    suffix = "iti_triviaqa" if family == "iti_triviaqa_transfer" else "iti_context"
    return Path("data/contrastive/truthfulness") / suffix


def _fingerprint_ids(values: list[str]) -> str:
    return hashlib.sha256("\n".join(sorted(values)).encode("utf-8")).hexdigest()[:16]


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
        "split_fingerprint": {
            "train_qids": _fingerprint_ids(sorted(train_ids)),
            "val_qids": _fingerprint_ids(sorted(val_ids)),
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
    bucketed_answers: dict[int, list[tuple[str, str]]],
    bucket: int,
    forbidden_qid: str,
) -> str:
    search_buckets = [bucket] + [b for b in sorted(bucketed_answers) if b != bucket]
    for current_bucket in search_buckets:
        pool = [
            answer
            for qid, answer in bucketed_answers.get(current_bucket, [])
            if qid != forbidden_qid
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
    bucketed_answers: dict[int, list[tuple[str, str]]] = defaultdict(list)
    for row in answer_pool_rows:
        answer = row["answers"]["text"][0]
        bucketed_answers[_bucket_answer(answer)].append((row["id"], answer))

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
            wrong_answer = _sample_wrong_answer(
                rng=rng,
                bucketed_answers=bucketed_answers,
                bucket=bucket,
                forbidden_qid=qid,
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
            "train_qids": _fingerprint_ids([row["id"] for row in sampled_train]),
            "val_qids": _fingerprint_ids([row["id"] for row in sampled_val]),
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
) -> tuple[float, float]:
    probe = LogisticRegression(
        random_state=0,
        max_iter=2000,
        solver="liblinear",
    )
    try:
        probe.fit(X_train, y_train, sample_weight=w_train)
        val_probs = probe.predict_proba(X_val)[:, 1]
        auroc = float(roc_auc_score(y_val, val_probs))
        balanced_accuracy = float(
            balanced_accuracy_score(y_val, (val_probs >= 0.5).astype(int))
        )
    except Exception:
        auroc = 0.5
        balanced_accuracy = 0.5
    return auroc, balanced_accuracy


def rank_heads(
    activations: dict[str, torch.Tensor],
    examples: list[ITIExample],
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
    metrics_summary: dict[str, Any] = {"position_summaries": list(POSITION_SUMMARIES)}

    for layer_idx in range(n_layers):
        for head_idx in range(n_heads):
            best_entry: dict[str, Any] | None = None
            metrics_by_summary: dict[str, Any] = {}
            for summary_name in POSITION_SUMMARIES:
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
                auroc, balanced_accuracy = _fit_probe_metrics(
                    X_train, y_train, w_train, X_val, y_val
                )
                direction, sigma, class_means, sign = _truth_mass_mean_direction(
                    X_train, y_train, w_train
                )
                entry = {
                    "layer": layer_idx,
                    "head": head_idx,
                    "position_summary": summary_name,
                    "auroc": round(auroc, 6),
                    "balanced_accuracy": round(balanced_accuracy, 6),
                    "sign": sign,
                    "sigma": round(sigma, 6),
                    "class_means": {
                        key: round(value, 6) for key, value in class_means.items()
                    },
                    "qid_coverage": qid_coverage,
                    "direction": direction.tolist(),
                }
                metrics_by_summary[summary_name] = {
                    key: value
                    for key, value in entry.items()
                    if key not in {"layer", "head", "direction"}
                }
                entry_rank = (
                    float(entry["auroc"]),
                    float(entry["balanced_accuracy"]),
                    -float(entry["sigma"]),
                )
                if best_entry is None:
                    best_entry = entry
                    continue
                best_rank = (
                    cast(float, best_entry["auroc"]),
                    cast(float, best_entry["balanced_accuracy"]),
                    -cast(float, best_entry["sigma"]),
                )
                if entry_rank > best_rank:
                    best_entry = entry

            assert best_entry is not None
            best_entry["metrics_by_summary"] = metrics_by_summary
            ranked_heads.append(best_entry)

    ranked_heads.sort(
        key=lambda item: (
            item["auroc"],
            item["balanced_accuracy"],
            -item["sigma"],
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
                "sigma",
            )
        }
        for item in ranked_heads[:5]
    ]
    metrics_summary["head_dim"] = head_dim
    return ranked_heads, metrics_summary


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
        if args.family == "iti_triviaqa_transfer":
            examples, family_metadata = build_triviaqa_transfer_examples(
                consistency_path=args.consistency_path,
                train_qids_path=args.train_qids_path,
                test_qids_path=args.test_qids_path,
                tokenizer=tokenizer,
                train_cap_per_qid_label=args.train_cap_per_qid_label,
                val_cap_per_qid_label=args.val_cap_per_qid_label,
            )
        else:
            examples, family_metadata = build_context_grounded_examples(
                tokenizer=tokenizer,
                seed=args.seed,
                train_questions=args.context_train_questions,
                val_questions=args.context_val_questions,
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
        ranked_heads, ranking_metadata = rank_heads(activations, examples)
        artifact = build_artifact(
            family=args.family,
            model=model,
            ranked_heads=ranked_heads,
        )
        torch.save(artifact, artifact_path)

        metadata = {
            "family": args.family,
            "model_path": args.model_path,
            "source_dataset": family_metadata["source_dataset"],
            "git_sha": get_git_sha(),
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
            "position_summaries": list(POSITION_SUMMARIES),
            "ranking": ranking_metadata,
            "selected_head_manifest": [
                {
                    "rank": idx + 1,
                    "layer": head["layer"],
                    "head": head["head"],
                    "position_summary": head["position_summary"],
                    "auroc": head["auroc"],
                    "balanced_accuracy": head["balanced_accuracy"],
                    "sigma": head["sigma"],
                }
                for idx, head in enumerate(ranked_heads[:32])
            ],
            "steering_direction_type": "mass_mean",
        }
        metadata_path.write_text(json_dumps(metadata), encoding="utf-8")
        provenance_extra["record_count"] = len(examples)
        provenance_extra["family"] = args.family
        provenance_extra["top_head"] = metadata["selected_head_manifest"][0]
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
