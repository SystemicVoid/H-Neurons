#!/usr/bin/env python3
"""Build locked same-item manifests for SIMID experiments.

SIMID pairs a forced-choice interface and an open-ended interface for the
same underlying item. The manifest stores the shared question, open prompt,
multiple-choice options, gold indices after deterministic option shuffling,
and provenance for every option.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
from typing import Any

from build_truthfulqa_splits import stable_question_id
from utils import (
    finish_run_provenance,
    format_path_for_metadata,
    normalize_answer,
    parse_semicolon_answers,
    provenance_error_message,
    provenance_status_for_exception,
    start_run_provenance,
)


ROOT = Path(__file__).resolve().parent.parent
SCHEMA_VERSION = "simid_manifest/v1"
BUILDER_VERSION = "2026-04-24"
DEFAULT_MODEL_PATH = "google/gemma-3-4b-it"
DEFAULT_ITI_ARTIFACT = (
    "data/contrastive/truthfulness/iti_truthfulqa_paperfaithful_production/iti_heads.pt"
)
DEFAULT_TRUTHFULQA_LEAKAGE_POLICY = "heldout_only"
DEFAULT_TRUTHFULQA_CSV = "data/benchmarks/TruthfulQA.csv"
DEFAULT_BRIDGE_MANIFEST = "data/manifests/triviaqa_bridge_test500_seed42.json"
DEFAULT_BRIDGE_PARQUET = "data/TriviaQA/rc.nocontext/validation-00000-of-00001.parquet"
DEFAULT_BRIDGE_BASELINE = (
    "data/gemma3_4b/intervention/triviaqa_bridge/test_experiment/alpha_1.0.jsonl"
)
REQUIRED_ROW_FIELDS = {
    "sample_id",
    "dataset",
    "question",
    "open_prompt",
    "mc_options",
    "gold_option_indices",
    "gold_aliases",
    "distractor_provenance",
    "option_order_seed",
    "option_order_replicate",
    "model_path",
    "tokenizer_path",
    "iti_artifact_path",
    "iti_artifact_sha256",
}
ABSTENTION_NORMALIZED = {
    "i do not know",
    "i don t know",
    "dont know",
    "do not know",
    "not sure",
    "no idea",
}


@dataclass(frozen=True)
class OptionRecord:
    text: str
    is_gold: bool
    provenance: dict[str, Any]


@dataclass(frozen=True)
class BridgeItem:
    question_id: str
    question: str
    aliases: list[str]


def stable_int_hash(*parts: object, modulo: int = 2**31 - 1) -> int:
    text = "\n".join(str(part) for part in parts)
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16) % modulo


def file_sha256(path: str | Path) -> str | None:
    resolved = Path(path)
    if not resolved.exists() or not resolved.is_file():
        return None
    h = hashlib.sha256()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dedupe_preserve_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        text = str(value).strip()
        norm = normalize_answer(text)
        if not text or not norm or norm in seen:
            continue
        seen.add(norm)
        output.append(text)
    return output


def word_len_bucket(text: str) -> int:
    return min(max(1, len(str(text).split())), 6)


def normalized_text_contains_alias(candidate_norm: str, alias_norms: set[str]) -> bool:
    candidate_tokens = candidate_norm.split()
    if not candidate_tokens:
        return False
    for alias_norm in alias_norms:
        alias_tokens = alias_norm.split()
        if not alias_tokens or len(alias_tokens) > len(candidate_tokens):
            continue
        for start in range(len(candidate_tokens) - len(alias_tokens) + 1):
            if candidate_tokens[start : start + len(alias_tokens)] == alias_tokens:
                return True
    return False


def open_prompt_for_question(question: str) -> str:
    return f"Question: {question}\nAnswer with a single short factual phrase only."


def option_order_seed(global_seed: int, sample_id: str, replicate_idx: int) -> int:
    return stable_int_hash("simid-option-order", global_seed, sample_id, replicate_idx)


def shuffle_options(
    options: list[OptionRecord],
    *,
    seed: int,
) -> tuple[list[str], list[int], list[dict[str, Any]]]:
    indexed = list(enumerate(options))
    rng = random.Random(seed)
    rng.shuffle(indexed)
    mc_options = [record.text for _, record in indexed]
    gold_indices = [
        new_idx for new_idx, (_, record) in enumerate(indexed) if record.is_gold
    ]
    provenance = []
    for original_idx, record in indexed:
        item = dict(record.provenance)
        item["option_text"] = record.text
        item["is_gold"] = bool(record.is_gold)
        item["pre_shuffle_index"] = int(original_idx)
        provenance.append(item)
    return mc_options, gold_indices, provenance


def build_manifest_row(
    *,
    sample_id: str,
    dataset: str,
    source_id: str,
    question: str,
    options: list[OptionRecord],
    gold_aliases: list[str],
    seed: int,
    replicate_idx: int,
    model_path: str,
    tokenizer_path: str,
    iti_artifact_path: str,
    iti_artifact_sha256: str | None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row_sample_id = sample_id
    if replicate_idx > 0:
        row_sample_id = f"{sample_id}__ord{replicate_idx}"
    row_seed = option_order_seed(seed, sample_id, replicate_idx)
    mc_options, gold_indices, provenance = shuffle_options(options, seed=row_seed)
    row = {
        "sample_id": row_sample_id,
        "base_sample_id": sample_id,
        "dataset": dataset,
        "source_id": source_id,
        "question": question,
        "open_prompt": open_prompt_for_question(question),
        "mc_options": mc_options,
        "gold_option_indices": gold_indices,
        "gold_aliases": gold_aliases,
        "distractor_provenance": provenance,
        "option_order_seed": row_seed,
        "option_order_replicate": replicate_idx,
        "model_path": model_path,
        "tokenizer_path": tokenizer_path,
        "iti_artifact_path": iti_artifact_path,
        "iti_artifact_sha256": iti_artifact_sha256,
        "manifest_seed": seed,
        "manifest_builder_version": BUILDER_VERSION,
    }
    if extra:
        row.update(extra)
    validate_manifest_row(row)
    return row


def default_metadata_path_for_artifact(artifact_path: str | Path) -> Path | None:
    candidate = Path(artifact_path).parent / "extraction_metadata.json"
    return candidate if candidate.exists() else None


def load_truthfulqa_split_metadata(
    *,
    iti_artifact_path: str | Path,
    metadata_path: str | Path | None,
) -> dict[str, Any]:
    resolved_metadata_path = (
        Path(metadata_path)
        if metadata_path is not None
        else default_metadata_path_for_artifact(iti_artifact_path)
    )
    if resolved_metadata_path is None or not resolved_metadata_path.exists():
        return {
            "metadata_path": None,
            "train_ids": set(),
            "val_ids": set(),
            "dev_ids": set(),
            "test_ids": set(),
            "metadata_available": False,
        }
    metadata = json.loads(resolved_metadata_path.read_text(encoding="utf-8"))
    train_ids = {str(qid) for qid in metadata.get("question_ids_train", [])}
    val_ids = {str(qid) for qid in metadata.get("question_ids_val", [])}
    dev_ids = {str(qid) for qid in metadata.get("question_ids_dev", [])}
    test_ids = {str(qid) for qid in metadata.get("question_ids_test", [])}
    if not dev_ids:
        dev_ids = train_ids | val_ids
    overlap = (train_ids | val_ids | dev_ids) & test_ids
    if overlap:
        raise ValueError(
            "TruthfulQA artifact metadata is internally inconsistent: "
            f"{len(overlap)} fitted/dev IDs also appear in held-out test IDs"
        )
    return {
        "metadata_path": str(resolved_metadata_path),
        "train_ids": train_ids,
        "val_ids": val_ids,
        "dev_ids": dev_ids,
        "test_ids": test_ids,
        "metadata_available": True,
        "fold_id": metadata.get("fold_id"),
        "fold_path": metadata.get("fold_path"),
        "direction_fit_source": metadata.get("direction_fit_source"),
        "sigma_fit_source": metadata.get("sigma_fit_source"),
    }


def truthfulqa_split_for_stable_id(
    stable_id: str,
    split_metadata: dict[str, Any],
) -> str:
    if stable_id in split_metadata["test_ids"]:
        return "test"
    if stable_id in split_metadata["val_ids"]:
        return "val"
    if stable_id in split_metadata["train_ids"]:
        return "train"
    if stable_id in split_metadata["dev_ids"]:
        return "dev"
    return "unknown"


def filter_truthfulqa_dataframe(
    df: Any,
    *,
    split_metadata: dict[str, Any],
    leakage_policy: str,
    n_rows: int | None,
) -> list[tuple[int, Any, str, str]]:
    if n_rows == 0:
        return []
    allowed_policies = {"heldout_only", "drop_if_no_heldout", "allow_fitted"}
    if leakage_policy not in allowed_policies:
        raise ValueError(
            f"Unsupported TruthfulQA leakage policy {leakage_policy!r}; "
            f"expected one of {sorted(allowed_policies)}"
        )
    if not split_metadata["metadata_available"] and leakage_policy != "allow_fitted":
        raise ValueError(
            "TruthfulQA leakage policy requires artifact split metadata, but none was "
            "found next to the ITI artifact. Use --truthfulqa-leakage-policy "
            "allow_fitted only for nonclaimable smoke/debug runs."
        )
    if (
        split_metadata["metadata_available"]
        and not split_metadata["test_ids"]
        and leakage_policy == "heldout_only"
    ):
        raise ValueError(
            "The selected ITI artifact exposes no held-out TruthfulQA test IDs. "
            "Refusing to build a claimable TruthfulQA SIMID block from fitted items. "
            "Use a fold/calibration artifact with question_ids_test, set "
            "--truthfulqa-n 0 for Bridge-only manifests, or use "
            "--truthfulqa-leakage-policy allow_fitted only for nonclaimable smoke/debug."
        )
    if (
        split_metadata["metadata_available"]
        and not split_metadata["test_ids"]
        and leakage_policy == "drop_if_no_heldout"
    ):
        return []

    rows: list[tuple[int, Any, str, str]] = []
    for csv_idx, raw in df.iterrows():
        question = str(raw["Question"]).strip()
        stable_id = stable_question_id(question)
        split = truthfulqa_split_for_stable_id(stable_id, split_metadata)
        if leakage_policy in {"heldout_only", "drop_if_no_heldout"} and split != "test":
            continue
        rows.append((int(csv_idx), raw, stable_id, split))
        if n_rows is not None and len(rows) >= n_rows:
            break
    return rows


def build_truthfulqa_rows(
    *,
    csv_path: Path,
    seed: int,
    n_rows: int | None,
    option_order_replicates: int,
    model_path: str,
    tokenizer_path: str,
    iti_artifact_path: str,
    iti_artifact_sha256: str | None,
    leakage_policy: str = DEFAULT_TRUTHFULQA_LEAKAGE_POLICY,
    split_metadata_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    import pandas as pd

    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    split_metadata = load_truthfulqa_split_metadata(
        iti_artifact_path=iti_artifact_path,
        metadata_path=split_metadata_path,
    )
    filtered_rows = filter_truthfulqa_dataframe(
        df,
        split_metadata=split_metadata,
        leakage_policy=leakage_policy,
        n_rows=n_rows,
    )

    rows: list[dict[str, Any]] = []
    for csv_idx, raw, stable_id, split in filtered_rows:
        question = str(raw["Question"]).strip()
        best = str(raw["Best Answer"]).strip()
        correct = dedupe_preserve_order(parse_semicolon_answers(raw["Correct Answers"]))
        incorrect = dedupe_preserve_order(
            parse_semicolon_answers(raw["Incorrect Answers"])
        )
        if best not in correct:
            correct = dedupe_preserve_order([best, *correct])
        incorrect = [
            answer
            for answer in incorrect
            if normalize_answer(answer)
            not in {normalize_answer(alias) for alias in correct}
        ]
        options = [
            OptionRecord(
                text=best,
                is_gold=True,
                provenance={
                    "source": "truthfulqa_best_answer",
                    "dataset": "truthfulqa",
                    "source_id": csv_idx,
                },
            )
        ]
        options.extend(
            OptionRecord(
                text=answer,
                is_gold=False,
                provenance={
                    "source": "truthfulqa_incorrect_answer",
                    "dataset": "truthfulqa",
                    "source_id": csv_idx,
                    "incorrect_answer_index": incorrect_idx,
                },
            )
            for incorrect_idx, answer in enumerate(incorrect)
        )
        sample_id = f"simid_truthfulqa_{csv_idx}"
        for replicate_idx in range(option_order_replicates):
            rows.append(
                build_manifest_row(
                    sample_id=sample_id,
                    dataset="truthfulqa",
                    source_id=str(csv_idx),
                    question=question,
                    options=options,
                    gold_aliases=correct,
                    seed=seed,
                    replicate_idx=replicate_idx,
                    model_path=model_path,
                    tokenizer_path=tokenizer_path,
                    iti_artifact_path=iti_artifact_path,
                    iti_artifact_sha256=iti_artifact_sha256,
                    extra={
                        "truthfulqa_category": raw.get("Category"),
                        "truthfulqa_type": raw.get("Type"),
                        "truthfulqa_source": raw.get("Source"),
                        "truthfulqa_csv_idx": csv_idx,
                        "truthfulqa_stable_id": stable_id,
                        "truthfulqa_artifact_split": split,
                        "truthfulqa_leakage_policy": leakage_policy,
                        "truthfulqa_split_metadata_path": split_metadata[
                            "metadata_path"
                        ],
                        "truthfulqa_seen_in_iti_fit": split in {"train", "val", "dev"},
                        "truthfulqa_best_answer": best,
                        "truthfulqa_correct_answers": correct,
                        "truthfulqa_incorrect_answers": incorrect,
                        "mc_endpoint": "truthfulqa_mc1",
                    },
                )
            )
    return rows


def load_bridge_items(manifest_path: Path, parquet_path: Path) -> list[BridgeItem]:
    import pandas as pd

    qids_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(qids_payload, list):
        raise ValueError("Bridge manifest must be a JSON list of question IDs")
    qids = [str(qid) for qid in qids_payload]
    df = pd.read_parquet(parquet_path)
    df = df.drop_duplicates(subset="question_id", keep="first")
    rows_by_qid = {str(row["question_id"]): row for _, row in df.iterrows()}
    missing = [qid for qid in qids if qid not in rows_by_qid]
    if missing:
        raise ValueError(
            f"{len(missing)} Bridge manifest IDs are missing from parquet; "
            f"examples={missing[:5]}"
        )

    items: list[BridgeItem] = []
    for qid in qids:
        row = rows_by_qid[qid]
        aliases = dedupe_preserve_order(list(row["answer"].get("aliases", [])))
        if not aliases:
            continue
        items.append(
            BridgeItem(
                question_id=qid,
                question=str(row["question"]).strip(),
                aliases=aliases,
            )
        )
    return items


def load_bridge_baseline_wrong_answers(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    wrong_by_qid: dict[str, str] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if bool(row.get("deterministic_correct")):
                continue
            response = str(row.get("response") or "").strip()
            if not response or normalize_answer(response) in ABSTENTION_NORMALIZED:
                continue
            qid = str(
                row.get("question_id")
                or str(row.get("id", "")).removeprefix("tqa_bridge_")
            )
            aliases = [str(alias) for alias in row.get("ground_truth_aliases") or []]
            alias_norms = {normalize_answer(alias) for alias in aliases}
            if normalized_text_contains_alias(normalize_answer(response), alias_norms):
                continue
            wrong_by_qid[qid] = response
    return wrong_by_qid


def bridge_alias_candidates(items: list[BridgeItem]) -> list[tuple[str, str]]:
    candidates: list[tuple[str, str]] = []
    for item in items:
        for alias in item.aliases:
            candidates.append((item.question_id, alias))
    return sorted(
        candidates, key=lambda pair: (word_len_bucket(pair[1]), pair[0], pair[1])
    )


def select_bridge_distractors(
    item: BridgeItem,
    *,
    all_candidates: list[tuple[str, str]],
    baseline_wrong_by_qid: dict[str, str],
    n_distractors: int,
    seed: int,
) -> list[OptionRecord]:
    own_norms = {normalize_answer(alias) for alias in item.aliases}
    chosen_norms = set(own_norms)
    distractors: list[OptionRecord] = []
    baseline_wrong = baseline_wrong_by_qid.get(item.question_id)
    if baseline_wrong:
        norm_wrong = normalize_answer(baseline_wrong)
        if norm_wrong and not normalized_text_contains_alias(norm_wrong, own_norms):
            distractors.append(
                OptionRecord(
                    text=baseline_wrong,
                    is_gold=False,
                    provenance={
                        "source": "bridge_baseline_wrong_answer",
                        "dataset": "triviaqa_bridge",
                        "source_question_id": item.question_id,
                    },
                )
            )
            chosen_norms.add(norm_wrong)

    target_bucket = word_len_bucket(item.aliases[0])
    rng = random.Random(seed)
    shuffled_candidates = list(all_candidates)
    rng.shuffle(shuffled_candidates)
    # Stable sort keeps the seeded shuffle order within each length-distance bucket.
    shuffled_candidates.sort(
        key=lambda pair: abs(word_len_bucket(pair[1]) - target_bucket)
    )
    for source_qid, alias in shuffled_candidates:
        if len(distractors) >= n_distractors:
            break
        norm_alias = normalize_answer(alias)
        if (
            source_qid == item.question_id
            or not norm_alias
            or normalized_text_contains_alias(norm_alias, own_norms)
            or norm_alias in chosen_norms
        ):
            continue
        source = (
            "bridge_gold_alias_length_matched"
            if word_len_bucket(alias) == target_bucket
            else "bridge_gold_alias_random_fallback"
        )
        distractors.append(
            OptionRecord(
                text=alias,
                is_gold=False,
                provenance={
                    "source": source,
                    "dataset": "triviaqa_bridge",
                    "source_question_id": source_qid,
                    "target_length_bucket": target_bucket,
                    "distractor_length_bucket": word_len_bucket(alias),
                },
            )
        )
        chosen_norms.add(norm_alias)

    if len(distractors) < n_distractors:
        raise ValueError(
            f"Unable to select {n_distractors} Bridge distractors for {item.question_id}; "
            f"selected {len(distractors)}"
        )
    return distractors


def build_bridge_rows(
    *,
    manifest_path: Path,
    parquet_path: Path,
    baseline_path: Path,
    seed: int,
    n_rows: int | None,
    n_options: int,
    option_order_replicates: int,
    model_path: str,
    tokenizer_path: str,
    iti_artifact_path: str,
    iti_artifact_sha256: str | None,
) -> list[dict[str, Any]]:
    if n_options < 2:
        raise ValueError("--n-options must be at least 2")
    all_items = load_bridge_items(manifest_path, parquet_path)
    items = list(all_items)
    if n_rows is not None:
        items = items[:n_rows]
    baseline_wrong_by_qid = load_bridge_baseline_wrong_answers(baseline_path)
    candidates = bridge_alias_candidates(all_items)

    rows: list[dict[str, Any]] = []
    for item in items:
        gold = item.aliases[0]
        distractors = select_bridge_distractors(
            item,
            all_candidates=candidates,
            baseline_wrong_by_qid=baseline_wrong_by_qid,
            n_distractors=n_options - 1,
            seed=stable_int_hash(seed, "bridge-distractors", item.question_id),
        )
        options = [
            OptionRecord(
                text=gold,
                is_gold=True,
                provenance={
                    "source": "bridge_gold_alias",
                    "dataset": "triviaqa_bridge",
                    "source_question_id": item.question_id,
                    "alias_index": 0,
                },
            ),
            *distractors,
        ]
        sample_id = f"simid_bridge_{item.question_id}"
        for replicate_idx in range(option_order_replicates):
            rows.append(
                build_manifest_row(
                    sample_id=sample_id,
                    dataset="triviaqa_bridge",
                    source_id=item.question_id,
                    question=item.question,
                    options=options,
                    gold_aliases=item.aliases,
                    seed=seed,
                    replicate_idx=replicate_idx,
                    model_path=model_path,
                    tokenizer_path=tokenizer_path,
                    iti_artifact_path=iti_artifact_path,
                    iti_artifact_sha256=iti_artifact_sha256,
                    extra={
                        "bridge_question_id": item.question_id,
                        "bridge_primary_gold_alias": gold,
                        "mc_endpoint": "synthetic_mc1",
                    },
                )
            )
    return rows


def validate_manifest_row(row: dict[str, Any]) -> None:
    missing = sorted(REQUIRED_ROW_FIELDS - set(row))
    if missing:
        raise ValueError(f"SIMID manifest row missing fields: {missing}")
    if not row["sample_id"]:
        raise ValueError("SIMID manifest row has empty sample_id")
    mc_options = row["mc_options"]
    if not isinstance(mc_options, list) or len(mc_options) < 2:
        raise ValueError(f"{row['sample_id']}: mc_options must contain >=2 options")
    if len({normalize_answer(option) for option in mc_options}) != len(mc_options):
        raise ValueError(
            f"{row['sample_id']}: mc_options contain normalized duplicates"
        )
    gold_indices = row["gold_option_indices"]
    if not isinstance(gold_indices, list) or not gold_indices:
        raise ValueError(f"{row['sample_id']}: gold_option_indices cannot be empty")
    if any(int(idx) < 0 or int(idx) >= len(mc_options) for idx in gold_indices):
        raise ValueError(f"{row['sample_id']}: gold option index out of range")
    provenance = row["distractor_provenance"]
    if not isinstance(provenance, list) or len(provenance) != len(mc_options):
        raise ValueError(
            f"{row['sample_id']}: distractor_provenance must align with mc_options"
        )
    replicate = row["option_order_replicate"]
    if not isinstance(replicate, int) or isinstance(replicate, bool):
        raise ValueError(
            f"{row['sample_id']}: option_order_replicate must be an integer"
        )
    if replicate < 0:
        raise ValueError(
            f"{row['sample_id']}: option_order_replicate must be non-negative"
        )
    if not isinstance(row["gold_aliases"], list) or not row["gold_aliases"]:
        raise ValueError(f"{row['sample_id']}: gold_aliases cannot be empty")


def validate_manifest(payload: dict[str, Any]) -> None:
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"Expected schema_version={SCHEMA_VERSION!r}, "
            f"got {payload.get('schema_version')!r}"
        )
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("SIMID manifest must contain a non-empty rows list")
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("SIMID manifest rows must be objects")
        validate_manifest_row(row)
        sample_id = str(row["sample_id"])
        if sample_id in seen:
            raise ValueError(f"Duplicate SIMID sample_id: {sample_id}")
        seen.add(sample_id)


def load_simid_manifest(path: str | Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, list):
        rows = payload
        payload = {"schema_version": SCHEMA_VERSION, "rows": rows}
    if not isinstance(payload, dict):
        raise ValueError("SIMID manifest must be a JSON object or row list")
    validate_manifest(payload)
    rows = payload["rows"]
    if not isinstance(rows, list):
        raise ValueError("SIMID manifest rows must be a list")
    return [dict(row) for row in rows]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--truthfulqa-csv", default=DEFAULT_TRUTHFULQA_CSV)
    parser.add_argument("--bridge-manifest", default=DEFAULT_BRIDGE_MANIFEST)
    parser.add_argument("--bridge-parquet", default=DEFAULT_BRIDGE_PARQUET)
    parser.add_argument("--bridge-baseline-jsonl", default=DEFAULT_BRIDGE_BASELINE)
    parser.add_argument("--truthfulqa-n", type=int, default=None)
    parser.add_argument(
        "--truthfulqa-leakage-policy",
        choices=["heldout_only", "drop_if_no_heldout", "allow_fitted"],
        default=DEFAULT_TRUTHFULQA_LEAKAGE_POLICY,
        help=(
            "heldout_only refuses fitted TruthfulQA items; allow_fitted is only for "
            "nonclaimable smoke/debug manifests"
        ),
    )
    parser.add_argument(
        "--truthfulqa-split-metadata",
        default=None,
        help=(
            "Optional extraction_metadata.json path used to identify fitted and "
            "held-out TruthfulQA question IDs. Defaults to the ITI artifact directory."
        ),
    )
    parser.add_argument("--bridge-n", type=int, default=None)
    parser.add_argument("--n-options", type=int, default=4)
    parser.add_argument("--option-order-replicates", type=int, default=1)
    parser.add_argument(
        "--min-truthfulqa-rows",
        type=int,
        default=0,
        help=(
            "Fail unless at least this many TruthfulQA manifest rows are built. "
            "Use this for claimable SIMID runs so held-out TruthfulQA cannot be "
            "silently dropped."
        ),
    )
    parser.add_argument(
        "--min-bridge-rows",
        type=int,
        default=0,
        help="Fail unless at least this many TriviaQA Bridge manifest rows are built.",
    )
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--tokenizer-path", default=None)
    parser.add_argument("--iti-artifact-path", default=DEFAULT_ITI_ARTIFACT)
    parser.add_argument(
        "--output",
        default="data/manifests/simid_truthfulqa_bridge_seed42.json",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.option_order_replicates < 1:
        raise ValueError("--option-order-replicates must be >= 1")
    tokenizer_path = args.tokenizer_path or args.model_path
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    iti_artifact_sha = file_sha256(args.iti_artifact_path)
    provenance = start_run_provenance(
        args,
        primary_target=output_path,
        output_targets=[output_path],
        extra={
            "schema_version": SCHEMA_VERSION,
            "builder_version": BUILDER_VERSION,
            "iti_artifact_sha256": iti_artifact_sha,
        },
    )
    status = "completed"
    extra: dict[str, Any] = {}
    try:
        rows = [
            *build_truthfulqa_rows(
                csv_path=Path(args.truthfulqa_csv),
                seed=args.seed,
                n_rows=args.truthfulqa_n,
                option_order_replicates=args.option_order_replicates,
                model_path=args.model_path,
                tokenizer_path=tokenizer_path,
                iti_artifact_path=args.iti_artifact_path,
                iti_artifact_sha256=iti_artifact_sha,
                leakage_policy=args.truthfulqa_leakage_policy,
                split_metadata_path=args.truthfulqa_split_metadata,
            ),
            *build_bridge_rows(
                manifest_path=Path(args.bridge_manifest),
                parquet_path=Path(args.bridge_parquet),
                baseline_path=Path(args.bridge_baseline_jsonl),
                seed=args.seed,
                n_rows=args.bridge_n,
                n_options=args.n_options,
                option_order_replicates=args.option_order_replicates,
                model_path=args.model_path,
                tokenizer_path=tokenizer_path,
                iti_artifact_path=args.iti_artifact_path,
                iti_artifact_sha256=iti_artifact_sha,
            ),
        ]
        n_truthfulqa_rows = sum(row["dataset"] == "truthfulqa" for row in rows)
        n_bridge_rows = sum(row["dataset"] == "triviaqa_bridge" for row in rows)
        if n_truthfulqa_rows < args.min_truthfulqa_rows:
            raise ValueError(
                "SIMID manifest does not contain enough TruthfulQA rows: "
                f"built {n_truthfulqa_rows}, required {args.min_truthfulqa_rows}. "
                "For a claimable same-item TruthfulQA block, use an ITI artifact "
                "whose extraction metadata exposes held-out question_ids_test, "
                "or lower --min-truthfulqa-rows only for an explicitly "
                "nonclaimable/debug run."
            )
        if n_bridge_rows < args.min_bridge_rows:
            raise ValueError(
                "SIMID manifest does not contain enough TriviaQA Bridge rows: "
                f"built {n_bridge_rows}, required {args.min_bridge_rows}."
            )
        payload = {
            "schema_version": SCHEMA_VERSION,
            "builder_version": BUILDER_VERSION,
            "metadata": {
                "seed": args.seed,
                "model_path": args.model_path,
                "tokenizer_path": tokenizer_path,
                "iti_artifact_path": args.iti_artifact_path,
                "iti_artifact_sha256": iti_artifact_sha,
                "truthfulqa_csv": format_path_for_metadata(
                    args.truthfulqa_csv, root=ROOT
                ),
                "bridge_manifest": format_path_for_metadata(
                    args.bridge_manifest, root=ROOT
                ),
                "bridge_parquet": format_path_for_metadata(
                    args.bridge_parquet, root=ROOT
                ),
                "bridge_baseline_jsonl": format_path_for_metadata(
                    args.bridge_baseline_jsonl, root=ROOT
                ),
                "truthfulqa_n": args.truthfulqa_n,
                "truthfulqa_leakage_policy": args.truthfulqa_leakage_policy,
                "truthfulqa_split_metadata": format_path_for_metadata(
                    args.truthfulqa_split_metadata, root=ROOT
                )
                if args.truthfulqa_split_metadata
                else None,
                "bridge_n": args.bridge_n,
                "n_options_bridge": args.n_options,
                "option_order_replicates": args.option_order_replicates,
                "n_rows": len(rows),
                "datasets": {
                    "truthfulqa": n_truthfulqa_rows,
                    "triviaqa_bridge": n_bridge_rows,
                },
                "minimum_rows": {
                    "truthfulqa": args.min_truthfulqa_rows,
                    "triviaqa_bridge": args.min_bridge_rows,
                },
            },
            "rows": rows,
        }
        validate_manifest(payload)
        output_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        extra["n_rows"] = len(rows)
        print(f"Wrote {len(rows)} SIMID rows to {output_path}")
    except BaseException as exc:
        status = provenance_status_for_exception(exc)
        extra["error"] = provenance_error_message(exc)
        raise
    finally:
        finish_run_provenance(provenance, status, extra)


if __name__ == "__main__":
    main()
