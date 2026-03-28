from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

from utils import (
    finish_run_provenance,
    get_git_sha,
    json_dumps,
    normalize_answer,
    provenance_error_message,
    provenance_status_for_exception,
    start_run_provenance,
)

UPSTREAM_REPO = "andyrdt/refusal_direction"
UPSTREAM_COMMIT = "9d852fae1a9121c78b29142de733cb1340770cc3"
UPSTREAM_RAW_BASE_URL = (
    f"https://raw.githubusercontent.com/{UPSTREAM_REPO}/{UPSTREAM_COMMIT}"
)

UPSTREAM_SPLIT_FILES = {
    "harmful_train": "dataset/splits/harmful_train.json",
    "harmful_val": "dataset/splits/harmful_val.json",
    "harmful_test": "dataset/splits/harmful_test.json",
    "harmless_train": "dataset/splits/harmless_train.json",
    "harmless_val": "dataset/splits/harmless_val.json",
    "harmless_test": "dataset/splits/harmless_test.json",
}

UPSTREAM_PROCESSED_FILES = {
    "advbench": "dataset/processed/advbench.json",
    "malicious_instruct": "dataset/processed/malicious_instruct.json",
    "tdc2023": "dataset/processed/tdc2023.json",
    "harmbench_val": "dataset/processed/harmbench_val.json",
    "alpaca": "dataset/processed/alpaca.json",
}

TRAIN_HARMFUL_COUNT = 128
VAL_HARMFUL_COUNT = 32
TRAIN_HARMLESS_COUNT = 128
VAL_HARMLESS_COUNT = 32
EVAL_HARMLESS_COUNT = 100

OFFICIAL_SAMPLING_ORDER = (
    "train_harmful",
    "train_harmless",
    "val_harmful",
    "val_harmless",
    "eval_harmless",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize the published refusal_direction dataset split pools at a pinned "
            "commit, then derive the exact seed-42 local contrastive sample used by the "
            "official pipeline before any model-specific refusal-metric filtering."
        )
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for deterministic sampling."
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data/contrastive/refusal"),
        help=(
            "Directory for refusal_contrastive.jsonl, metadata.json, alpaca_eval_holdout.jsonl, "
            "and a pinned upstream split snapshot."
        ),
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Build and validate the dataset in memory without writing output files.",
    )
    return parser.parse_args()


def fetch_text(url: str) -> str:
    request = Request(url, headers={"User-Agent": "h-neurons-refusal-curation/2.0"})
    with urlopen(request) as response:
        return response.read().decode("utf-8")


def raw_url(relative_path: str) -> str:
    return f"{UPSTREAM_RAW_BASE_URL}/{relative_path}"


def fetch_upstream_json(
    relative_path: str,
) -> tuple[list[dict[str, Any]], str, dict[str, Any]]:
    url = raw_url(relative_path)
    text = fetch_text(url)
    payload = json.loads(text)
    if not isinstance(payload, list):
        raise TypeError(
            f"Expected list payload for {relative_path}, found {type(payload)}"
        )
    return (
        payload,
        text,
        {
            "path": relative_path,
            "url": url,
            "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "record_count": len(payload),
        },
    )


def normalize_text(text: str) -> str:
    normalized = normalize_answer(text)
    if not normalized:
        raise ValueError(f"Encountered empty normalized text for input: {text!r}")
    return normalized


def extract_instruction(record: dict[str, Any], *, dataset_name: str) -> str:
    instruction = str(record.get("instruction", "")).strip()
    if not instruction:
        raise ValueError(f"Missing instruction in {dataset_name}: {record}")
    return instruction


def build_source_lookup(
    processed_payloads: dict[str, list[dict[str, Any]]],
) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for source_name, rows in processed_payloads.items():
        for row in rows:
            instruction = extract_instruction(row, dataset_name=source_name)
            prior = lookup.get(instruction)
            if prior is not None and prior != source_name:
                raise ValueError(
                    "Ambiguous exact source match for instruction "
                    f"{instruction!r}: {prior} vs {source_name}"
                )
            lookup[instruction] = source_name
    return lookup


def sample_rows(
    rows: list[dict[str, Any]], count: int, rng: random.Random
) -> list[dict[str, Any]]:
    if len(rows) < count:
        raise ValueError(f"Requested {count} rows, but only {len(rows)} are available.")
    return rng.sample(rows, count)


def build_minimally_audited_train_sample(
    *,
    official_train_harmful: list[dict[str, Any]],
    harmful_train_pool: list[dict[str, Any]],
    harmful_test_pool: list[dict[str, Any]],
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    harmful_test_norms = {
        normalize_text(text)
        for text in split_texts(harmful_test_pool, dataset_name="harmful_test_pool")
    }
    official_texts = split_texts(
        official_train_harmful, dataset_name="official_train_harmful"
    )
    leaked_positions = [
        index
        for index, text in enumerate(official_texts)
        if normalize_text(text) in harmful_test_norms
    ]
    if not leaked_positions:
        return official_train_harmful, {
            "train_harmful_rows_replaced": 0,
            "replaced_examples": [],
            "replacement_examples": [],
            "replacement_seed": seed,
        }

    selected_exact = set(official_texts)
    selected_norm = {normalize_text(text) for text in official_texts}
    replacement_pool = []
    for row in harmful_train_pool:
        instruction = extract_instruction(row, dataset_name="harmful_train_pool")
        normalized = normalize_text(instruction)
        if instruction in selected_exact:
            continue
        if normalized in harmful_test_norms:
            continue
        if normalized in selected_norm:
            continue
        replacement_pool.append(row)

    if len(replacement_pool) < len(leaked_positions):
        raise ValueError(
            "Not enough replacement candidates to remove normalized train/test leakage."
        )

    replacements = sample_rows(
        replacement_pool, len(leaked_positions), random.Random(seed)
    )
    audited_rows = list(official_train_harmful)
    replaced_examples: list[str] = []
    replacement_examples: list[str] = []
    for index, replacement in zip(leaked_positions, replacements, strict=True):
        replaced_examples.append(
            extract_instruction(
                audited_rows[index], dataset_name="official_train_harmful"
            )
        )
        replacement_examples.append(
            extract_instruction(replacement, dataset_name="replacement_train_harmful")
        )
        audited_rows[index] = replacement

    return audited_rows, {
        "train_harmful_rows_replaced": len(leaked_positions),
        "replaced_examples": replaced_examples,
        "replacement_examples": replacement_examples,
        "replacement_seed": seed,
    }


def build_output_records(
    *,
    train_harmful: list[dict[str, Any]],
    val_harmful: list[dict[str, Any]],
    train_harmless: list[dict[str, Any]],
    val_harmless: list[dict[str, Any]],
    train_harmful_source_lookup: dict[str, str],
) -> list[dict[str, str]]:
    output_records: list[dict[str, str]] = []

    def append_rows(rows: list[dict[str, Any]], *, split: str, label: str) -> None:
        for index, row in enumerate(rows):
            instruction = extract_instruction(
                row, dataset_name=f"{split}_{label}_sample"
            )
            if split == "train" and label == "harmful":
                source = train_harmful_source_lookup[instruction]
            elif label == "harmful":
                source = "harmbench_val"
            else:
                source = "alpaca"
            output_records.append(
                {
                    "id": f"refusal_{split}_{label}_{index:03d}",
                    "text": instruction,
                    "label": label,
                    "source": source,
                    "split": split,
                }
            )

    append_rows(train_harmful, split="train", label="harmful")
    append_rows(val_harmful, split="val", label="harmful")
    append_rows(train_harmless, split="train", label="harmless")
    append_rows(val_harmless, split="val", label="harmless")
    return output_records


def build_harmless_eval_holdout_records(
    eval_harmless: list[dict[str, Any]],
) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for index, row in enumerate(eval_harmless):
        instruction = extract_instruction(row, dataset_name="eval_harmless_sample")
        records.append(
            {
                "id": f"refusal_eval_harmless_{index:03d}",
                "text": instruction,
                "label": "harmless",
                "source": "alpaca",
                "split": "eval",
            }
        )
    return records


def split_texts(rows: list[dict[str, Any]], *, dataset_name: str) -> list[str]:
    return [extract_instruction(row, dataset_name=dataset_name) for row in rows]


def duplicate_audit(texts: list[str]) -> dict[str, Any]:
    exact_counter = Counter(texts)
    normalized_counter = Counter(normalize_text(text) for text in texts)

    exact_examples = [
        {"text": text, "count": count}
        for text, count in exact_counter.items()
        if count > 1
    ][:10]
    normalized_examples = [
        {"normalized_text": text, "count": count}
        for text, count in normalized_counter.items()
        if count > 1
    ][:10]

    return {
        "count": len(texts),
        "exact_duplicate_count": sum(
            count - 1 for count in exact_counter.values() if count > 1
        ),
        "normalized_duplicate_count": sum(
            count - 1 for count in normalized_counter.values() if count > 1
        ),
        "exact_duplicate_examples": exact_examples,
        "normalized_duplicate_examples": normalized_examples,
    }


def pairwise_overlap_audit(
    left_rows: list[dict[str, Any]],
    right_rows: list[dict[str, Any]],
    *,
    left_name: str,
    right_name: str,
) -> dict[str, Any]:
    left_texts = split_texts(left_rows, dataset_name=left_name)
    right_texts = split_texts(right_rows, dataset_name=right_name)

    left_exact = set(left_texts)
    right_exact = set(right_texts)
    exact_overlap = sorted(left_exact & right_exact)

    left_norm_map: defaultdict[str, list[str]] = defaultdict(list)
    right_norm_map: defaultdict[str, list[str]] = defaultdict(list)
    for text in left_texts:
        left_norm_map[normalize_text(text)].append(text)
    for text in right_texts:
        right_norm_map[normalize_text(text)].append(text)

    normalized_overlap_keys = sorted(set(left_norm_map) & set(right_norm_map))
    normalized_overlap_examples = [
        {
            "normalized_text": key,
            left_name: left_norm_map[key],
            right_name: right_norm_map[key],
        }
        for key in normalized_overlap_keys[:10]
    ]

    return {
        "left": left_name,
        "right": right_name,
        "exact_overlap_count": len(exact_overlap),
        "normalized_overlap_count": len(normalized_overlap_keys),
        "exact_overlap_examples": exact_overlap[:10],
        "normalized_overlap_examples": normalized_overlap_examples,
    }


def sample_fingerprint(rows: list[dict[str, Any]], *, dataset_name: str) -> str:
    texts = split_texts(rows, dataset_name=dataset_name)
    payload = json.dumps(texts, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def assert_zero_overlap(
    left_rows: list[dict[str, Any]],
    right_rows: list[dict[str, Any]],
    *,
    left_name: str,
    right_name: str,
) -> None:
    audit = pairwise_overlap_audit(
        left_rows, right_rows, left_name=left_name, right_name=right_name
    )
    if audit["exact_overlap_count"] != 0 or audit["normalized_overlap_count"] != 0:
        raise AssertionError(
            f"Expected {left_name} and {right_name} to be disjoint, found "
            f"exact={audit['exact_overlap_count']} normalized={audit['normalized_overlap_count']}."
        )


def validate_sampled_dataset(
    *,
    train_harmful: list[dict[str, Any]],
    val_harmful: list[dict[str, Any]],
    train_harmless: list[dict[str, Any]],
    val_harmless: list[dict[str, Any]],
    eval_harmless: list[dict[str, Any]],
) -> dict[str, Any]:
    if len(train_harmful) != TRAIN_HARMFUL_COUNT:
        raise AssertionError("Unexpected train harmful count.")
    if len(val_harmful) != VAL_HARMFUL_COUNT:
        raise AssertionError("Unexpected val harmful count.")
    if len(train_harmless) != TRAIN_HARMLESS_COUNT:
        raise AssertionError("Unexpected train harmless count.")
    if len(val_harmless) != VAL_HARMLESS_COUNT:
        raise AssertionError("Unexpected val harmless count.")
    if len(eval_harmless) != EVAL_HARMLESS_COUNT:
        raise AssertionError("Unexpected eval harmless count.")

    assert_zero_overlap(
        train_harmful, val_harmful, left_name="train_harmful", right_name="val_harmful"
    )
    assert_zero_overlap(
        train_harmless,
        val_harmless,
        left_name="train_harmless",
        right_name="val_harmless",
    )
    assert_zero_overlap(
        train_harmless,
        eval_harmless,
        left_name="train_harmless",
        right_name="eval_harmless",
    )
    assert_zero_overlap(
        val_harmless,
        eval_harmless,
        left_name="val_harmless",
        right_name="eval_harmless",
    )

    return {
        "train_val_harmful_exact_overlap": 0,
        "train_val_harmful_normalized_overlap": 0,
        "train_val_harmless_exact_overlap": 0,
        "train_val_harmless_normalized_overlap": 0,
        "train_eval_harmless_exact_overlap": 0,
        "train_eval_harmless_normalized_overlap": 0,
        "val_eval_harmless_exact_overlap": 0,
        "val_eval_harmless_normalized_overlap": 0,
    }


def summarize_counts(records: list[dict[str, str]]) -> dict[str, dict[str, int]]:
    summary: dict[str, dict[str, int]] = {
        "train": {"harmful": 0, "harmless": 0},
        "val": {"harmful": 0, "harmless": 0},
    }
    for record in records:
        summary[record["split"]][record["label"]] += 1
    return summary


def print_distribution(
    *, title: str, records: list[dict[str, str]], key: str = "source"
) -> None:
    counts = Counter(record[key] for record in records)
    print(f"\n{title}")
    print(f"{key:<20} {'count':>5}")
    print(f"{'-' * 20} {'-' * 5}")
    for value, count in sorted(counts.items()):
        print(f"{value:<20} {count:>5}")


def write_jsonl(records: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_text_file(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    jsonl_path = output_dir / "refusal_contrastive.jsonl"
    metadata_path = output_dir / "metadata.json"
    harmless_eval_holdout_path = output_dir / "alpaca_eval_holdout.jsonl"
    upstream_dir = output_dir / "upstream"
    upstream_manifest_path = upstream_dir / "manifest.json"
    upstream_split_paths = {
        name: upstream_dir / f"{name}.json" for name in UPSTREAM_SPLIT_FILES
    }

    provenance_handle: Any | None = None
    provenance_status = "completed"
    provenance_extra: dict[str, Any] = {}

    try:
        if not args.dry_run:
            provenance_handle = start_run_provenance(
                args,
                primary_target=output_dir,
                output_targets=[
                    jsonl_path,
                    metadata_path,
                    harmless_eval_holdout_path,
                    upstream_manifest_path,
                    *upstream_split_paths.values(),
                ],
                extra={
                    "dry_run": args.dry_run,
                    "upstream_repo": UPSTREAM_REPO,
                    "upstream_commit": UPSTREAM_COMMIT,
                },
                primary_target_is_dir=True,
            )

        split_payloads: dict[str, list[dict[str, Any]]] = {}
        split_texts_by_name: dict[str, str] = {}
        split_stats: dict[str, Any] = {}
        for name, relative_path in UPSTREAM_SPLIT_FILES.items():
            payload, raw_text, stats = fetch_upstream_json(relative_path)
            split_payloads[name] = payload
            split_texts_by_name[name] = raw_text
            split_stats[name] = stats

        processed_payloads: dict[str, list[dict[str, Any]]] = {}
        processed_stats: dict[str, Any] = {}
        for name, relative_path in UPSTREAM_PROCESSED_FILES.items():
            payload, _, stats = fetch_upstream_json(relative_path)
            processed_payloads[name] = payload
            processed_stats[name] = stats

        train_harmful_source_lookup = build_source_lookup(
            {
                "advbench": processed_payloads["advbench"],
                "malicious_instruct": processed_payloads["malicious_instruct"],
                "tdc2023": processed_payloads["tdc2023"],
            }
        )

        official_rng = random.Random(args.seed)
        official_train_harmful = sample_rows(
            split_payloads["harmful_train"], TRAIN_HARMFUL_COUNT, official_rng
        )
        train_harmless = sample_rows(
            split_payloads["harmless_train"], TRAIN_HARMLESS_COUNT, official_rng
        )
        val_harmful = sample_rows(
            split_payloads["harmful_val"], VAL_HARMFUL_COUNT, official_rng
        )
        val_harmless = sample_rows(
            split_payloads["harmless_val"], VAL_HARMLESS_COUNT, official_rng
        )
        eval_harmless = sample_rows(
            split_payloads["harmless_test"], EVAL_HARMLESS_COUNT, official_rng
        )
        train_harmful, local_train_harmful_audit = build_minimally_audited_train_sample(
            official_train_harmful=official_train_harmful,
            harmful_train_pool=split_payloads["harmful_train"],
            harmful_test_pool=split_payloads["harmful_test"],
            seed=args.seed,
        )

        validation = validate_sampled_dataset(
            train_harmful=train_harmful,
            val_harmful=val_harmful,
            train_harmless=train_harmless,
            val_harmless=val_harmless,
            eval_harmless=eval_harmless,
        )

        output_records = build_output_records(
            train_harmful=train_harmful,
            val_harmful=val_harmful,
            train_harmless=train_harmless,
            val_harmless=val_harmless,
            train_harmful_source_lookup=train_harmful_source_lookup,
        )
        harmless_eval_holdout_records = build_harmless_eval_holdout_records(
            eval_harmless
        )
        counts_by_split_label = summarize_counts(output_records)

        upstream_pool_audit = {
            "within_split_duplicates": {
                name: duplicate_audit(split_texts(payload, dataset_name=name))
                for name, payload in split_payloads.items()
            },
            "pairwise_overlaps": {
                "harmful_train__harmful_val": pairwise_overlap_audit(
                    split_payloads["harmful_train"],
                    split_payloads["harmful_val"],
                    left_name="harmful_train",
                    right_name="harmful_val",
                ),
                "harmful_train__harmful_test": pairwise_overlap_audit(
                    split_payloads["harmful_train"],
                    split_payloads["harmful_test"],
                    left_name="harmful_train",
                    right_name="harmful_test",
                ),
                "harmful_val__harmful_test": pairwise_overlap_audit(
                    split_payloads["harmful_val"],
                    split_payloads["harmful_test"],
                    left_name="harmful_val",
                    right_name="harmful_test",
                ),
                "harmless_train__harmless_val": pairwise_overlap_audit(
                    split_payloads["harmless_train"],
                    split_payloads["harmless_val"],
                    left_name="harmless_train",
                    right_name="harmless_val",
                ),
                "harmless_train__harmless_test": pairwise_overlap_audit(
                    split_payloads["harmless_train"],
                    split_payloads["harmless_test"],
                    left_name="harmless_train",
                    right_name="harmless_test",
                ),
                "harmless_val__harmless_test": pairwise_overlap_audit(
                    split_payloads["harmless_val"],
                    split_payloads["harmless_test"],
                    left_name="harmless_val",
                    right_name="harmless_test",
                ),
            },
        }

        sampled_audit = {
            "official_repo_sample_fingerprints": {
                "train_harmful": sample_fingerprint(
                    official_train_harmful, dataset_name="official_train_harmful"
                ),
                "train_harmless": sample_fingerprint(
                    train_harmless, dataset_name="official_train_harmless"
                ),
                "val_harmful": sample_fingerprint(
                    val_harmful, dataset_name="official_val_harmful"
                ),
                "val_harmless": sample_fingerprint(
                    val_harmless, dataset_name="official_val_harmless"
                ),
                "eval_harmless": sample_fingerprint(
                    eval_harmless, dataset_name="official_eval_harmless"
                ),
            },
            "sample_fingerprints": {
                "train_harmful": sample_fingerprint(
                    train_harmful, dataset_name="train_harmful"
                ),
                "train_harmless": sample_fingerprint(
                    train_harmless, dataset_name="train_harmless"
                ),
                "val_harmful": sample_fingerprint(
                    val_harmful, dataset_name="val_harmful"
                ),
                "val_harmless": sample_fingerprint(
                    val_harmless, dataset_name="val_harmless"
                ),
                "eval_harmless": sample_fingerprint(
                    eval_harmless, dataset_name="eval_harmless"
                ),
            },
            "train_harmful_leakage_fix": {
                **local_train_harmful_audit,
                "official_train_harmful_overlap_after_fix": pairwise_overlap_audit(
                    train_harmful,
                    official_train_harmful,
                    left_name="local_train_harmful",
                    right_name="official_train_harmful",
                ),
            },
            "sample_vs_upstream_harmful_test": {
                "train_harmful__harmful_test": pairwise_overlap_audit(
                    train_harmful,
                    split_payloads["harmful_test"],
                    left_name="train_harmful_sample",
                    right_name="harmful_test_pool",
                ),
                "val_harmful__harmful_test": pairwise_overlap_audit(
                    val_harmful,
                    split_payloads["harmful_test"],
                    left_name="val_harmful_sample",
                    right_name="harmful_test_pool",
                ),
            },
        }

        train_source_distribution = dict(
            sorted(
                Counter(
                    record["source"]
                    for record in output_records
                    if record["id"].startswith("refusal_train_harmful_")
                ).items()
            )
        )

        upstream_manifest = {
            "repo": UPSTREAM_REPO,
            "commit": UPSTREAM_COMMIT,
            "download_mode": "pinned_raw_files",
            "split_files": split_stats,
            "notes": {
                "why_not_submodule": (
                    "Use a pinned raw snapshot rather than a git submodule so the dataset "
                    "stays lightweight, reviewable, and stable at a specific upstream commit."
                ),
                "official_pipeline_sampling": (
                    "pipeline/run_pipeline.py seeds Python random with 42, then samples "
                    "128 harmful train, 128 harmless train, 32 harmful val, 32 harmless val, "
                    "and later 100 harmless test examples from these published pools."
                ),
            },
        }

        metadata = {
            "dataset_name": "refusal_contrastive",
            "curation_mode": "pinned_refusal_direction_snapshot",
            "paths": {
                "jsonl": str(jsonl_path),
                "metadata": str(metadata_path),
                "alpaca_eval_holdout": str(harmless_eval_holdout_path),
                "upstream_dir": str(upstream_dir),
                "upstream_manifest": str(upstream_manifest_path),
                "script_path": str(Path(__file__).resolve()),
            },
            "git_sha": get_git_sha(),
            "upstream": {
                "repo": UPSTREAM_REPO,
                "commit": UPSTREAM_COMMIT,
                "split_files": split_stats,
                "processed_files_used_for_source_resolution": processed_stats,
            },
            "faithfulness": {
                "paper_appendix_alignment": (
                    "Matches Appendix A.1/A.2 at the level of source pools and 128/32/100 "
                    "sample sizes, but anchors the reconstruction to the official published "
                    "split pools instead of rebuilding them by hand from raw corpora."
                ),
                "official_repo_alignment": {
                    "published_split_pools": True,
                    "seed": args.seed,
                    "sampling_order": list(OFFICIAL_SAMPLING_ORDER),
                    "matches_pipeline_run_pipeline_py_sampling": True,
                    "matches_official_default_seed": args.seed == 42,
                },
                "not_applied_here": (
                    "The official pipeline performs model-specific refusal-metric filtering "
                    "after sampling (Config.filter_train=True and Config.filter_val=True). "
                    "This script intentionally materializes the exact sampled pools before "
                    "that runtime filtering, because the filter depends on the model."
                ),
                "repo_vs_paper_differences": [
                    (
                        "The paper text discusses JBB and HarmBench test as harmful "
                        "evaluation datasets; the official repo's harmful_test pool also "
                        "includes StrongREJECT."
                    ),
                    (
                        "The official dataset notebook filters HarmBench using "
                        "FunctionalCategory for copyright and Tags for context, rather "
                        "than the alternative ContextString/SemanticCategory logic used in "
                        "the previous local reconstruction."
                    ),
                    (
                        "The official repo publishes fixed split pools, then the pipeline "
                        "samples from them with seed 42. Rebuilding from mutable upstream "
                        "raw corpora is therefore less faithful than pinning those pools."
                    ),
                    (
                        "This local working dataset makes one additional rigor fix beyond "
                        "the official sample: it replaces any train-harmful rows whose "
                        "normalized text still overlaps the published harmful_test pool."
                    ),
                ],
            },
            "sampling": {
                "seed": args.seed,
                "order": list(OFFICIAL_SAMPLING_ORDER),
                "local_train_harmful_policy": (
                    "Start from the official seed-42 sample, then minimally replace any "
                    "train-harmful prompts with normalized overlap against harmful_test."
                ),
                "counts": {
                    "train_harmful": len(train_harmful),
                    "train_harmless": len(train_harmless),
                    "val_harmful": len(val_harmful),
                    "val_harmless": len(val_harmless),
                    "eval_harmless": len(eval_harmless),
                },
                "fingerprints": sampled_audit["sample_fingerprints"],
            },
            "sampled_counts": {
                "train_harmful_by_source": train_source_distribution,
                "val_harmful_by_source": {"harmbench_val": len(val_harmful)},
                "harmless_by_source": {
                    "alpaca_train": len(train_harmless),
                    "alpaca_val": len(val_harmless),
                    "alpaca_eval": len(eval_harmless),
                },
            },
            "counts_by_split_label": counts_by_split_label,
            "validation": validation,
            "upstream_pool_audit": upstream_pool_audit,
            "sampled_audit": sampled_audit,
            "dry_run": args.dry_run,
        }

        provenance_extra.update(
            {
                "record_count": len(output_records),
                "counts_by_split_label": counts_by_split_label,
                "train_harmful_by_source": train_source_distribution,
                "eval_harmless_count": len(eval_harmless),
                "upstream_commit": UPSTREAM_COMMIT,
                "sample_fingerprints": sampled_audit["sample_fingerprints"],
            }
        )

        print(
            "Curated refusal contrastive dataset from pinned upstream split pools with "
            f"seed={args.seed}."
        )
        print(f"Upstream repo: {UPSTREAM_REPO}@{UPSTREAM_COMMIT}")
        print(
            "Materialized exact upstream split files into the local output directory."
        )
        print(
            "Train harmful leakage fix: replaced "
            f"{local_train_harmful_audit['train_harmful_rows_replaced']} official "
            "rows with normalized-nonoverlapping alternatives."
        )
        print_distribution(
            title="Train harmful source distribution",
            records=[
                record
                for record in output_records
                if record["id"].startswith("refusal_train_harmful_")
            ],
        )
        print_distribution(
            title="Sampled record counts by source",
            records=output_records + harmless_eval_holdout_records,
        )

        if args.dry_run:
            print("\nDry run enabled; skipping file writes.")
            return

        output_dir.mkdir(parents=True, exist_ok=True)
        write_jsonl(output_records, jsonl_path)
        write_jsonl(harmless_eval_holdout_records, harmless_eval_holdout_path)
        metadata_path.write_text(json_dumps(metadata), encoding="utf-8")

        for name, raw_text in split_texts_by_name.items():
            write_text_file(upstream_split_paths[name], raw_text)
        upstream_manifest_path.write_text(
            json_dumps(upstream_manifest), encoding="utf-8"
        )

        print(f"\nWrote dataset to {jsonl_path}")
        print(f"Wrote Alpaca eval holdout to {harmless_eval_holdout_path}")
        print(f"Wrote metadata to {metadata_path}")
        print(f"Wrote pinned upstream split snapshot to {upstream_dir}")
    except BaseException as exc:
        provenance_status = provenance_status_for_exception(exc)
        provenance_extra["error"] = provenance_error_message(exc)
        raise
    finally:
        if provenance_handle is not None:
            finish_run_provenance(
                provenance_handle, provenance_status, provenance_extra
            )


if __name__ == "__main__":
    main()
