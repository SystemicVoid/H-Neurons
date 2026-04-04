#!/usr/bin/env python3
"""Build 4 mutually disjoint manifest files for the TriviaQA bridge benchmark.

Produces pilot (150), dev (100), test (500), and reserve (200) splits from
the TriviaQA rc.nocontext validation parquet, using stratified proportional
sampling across QID prefix family, answer-length bucket, and alias-count bucket.

Usage::

    uv run python scripts/build_triviaqa_bridge_manifest.py \\
        --seed 42 \\
        --pilot_n 150 \\
        --dev_n 100 \\
        --test_n 500 \\
        --reserve_n 200
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any

from utils import (
    fingerprint_ids,
    finish_run_provenance,
    normalize_answer,
    provenance_error_message,
    provenance_status_for_exception,
    start_run_provenance,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_prefix(qid: str) -> str:
    """Extract QID source prefix family (up to first digit).

    E.g., "sfq_123" → "sfq_", "odql_45" → "odql_".
    """
    match = re.match(r"^([a-zA-Z_]+)", qid)
    return match.group(1) + "_" if match else "unknown_"


def _answer_length_bucket(first_alias: str) -> str:
    """Bucket by word count of the first alias."""
    n_words = len(first_alias.split())
    if n_words <= 1:
        return "1w"
    if n_words == 2:
        return "2w"
    return "3+w"


def _alias_count_bucket(n_aliases: int) -> str:
    """Bucket by number of aliases."""
    if n_aliases <= 3:
        return "1-3"
    if n_aliases <= 6:
        return "4-6"
    return "7-10"


def _stratum_key(prefix: str, ans_bucket: str, alias_bucket: str) -> str:
    return f"{prefix}{ans_bucket}|{alias_bucket}"


# ---------------------------------------------------------------------------
# Curation filters
# ---------------------------------------------------------------------------


def _apply_curation_filters(
    df: Any,
) -> tuple[Any, dict[str, dict[str, int | str]]]:
    """Apply metadata-only curation filters.  Returns filtered df and stats."""
    filter_stats: dict[str, dict[str, int | str]] = {}

    # Filter 1: Exclude questions where ALL aliases normalize to ≤2 chars
    def all_aliases_short(answer: dict[str, Any]) -> bool:
        aliases = answer.get("aliases", [])
        if len(aliases) == 0:
            return True
        return all(len(normalize_answer(a)) <= 2 for a in aliases)

    mask_short = df["answer"].apply(all_aliases_short)
    n_excluded_short = int(mask_short.sum())
    df = df[~mask_short].copy()
    filter_stats["all_aliases_le_2_chars"] = {
        "description": "Exclude questions where ALL aliases normalize to ≤2 characters",
        "excluded": n_excluded_short,
        "remaining": len(df),
    }
    print(
        f"  Filter 1 (short aliases): excluded {n_excluded_short}, remaining {len(df)}"
    )

    # Filter 2: Limit alias count ≤10
    def alias_count(answer: dict[str, Any]) -> int:
        return len(answer.get("aliases", []))

    mask_many = df["answer"].apply(alias_count) > 10
    n_excluded_many = int(mask_many.sum())
    df = df[~mask_many].copy()
    filter_stats["alias_count_gt_10"] = {
        "description": "Exclude questions with >10 aliases",
        "excluded": n_excluded_many,
        "remaining": len(df),
    }
    print(f"  Filter 2 (>10 aliases): excluded {n_excluded_many}, remaining {len(df)}")

    # Filter 3: First alias ≤5 words
    def first_alias_words(answer: dict[str, Any]) -> int:
        aliases = answer.get("aliases", [])
        if len(aliases) == 0:
            return 0
        return len(aliases[0].split())

    mask_long = df["answer"].apply(first_alias_words) > 5
    n_excluded_long = int(mask_long.sum())
    df = df[~mask_long].copy()
    filter_stats["first_alias_gt_5_words"] = {
        "description": "Exclude questions where the first alias has >5 words",
        "excluded": n_excluded_long,
        "remaining": len(df),
    }
    print(
        f"  Filter 3 (first alias >5 words): excluded {n_excluded_long}, remaining {len(df)}"
    )

    return df, filter_stats


# ---------------------------------------------------------------------------
# Stratified proportional sampling
# ---------------------------------------------------------------------------


def _assign_strata(df: Any) -> dict[str, list[str]]:
    """Assign each row to a stratum. Returns {stratum_key: [qid, ...]}."""
    strata: dict[str, list[str]] = {}
    for _, row in df.iterrows():
        qid = row["question_id"]
        answer = row["answer"]
        aliases = answer.get("aliases", [])

        prefix = _extract_prefix(qid)
        first_alias = aliases[0] if len(aliases) > 0 else ""
        ans_bucket = _answer_length_bucket(first_alias)
        alias_bucket = _alias_count_bucket(len(aliases))
        key = _stratum_key(prefix, ans_bucket, alias_bucket)

        strata.setdefault(key, []).append(qid)
    return strata


def _proportional_sample(
    strata: dict[str, list[str]],
    split_sizes: dict[str, int],
    seed: int,
) -> dict[str, list[str]]:
    """Proportionally allocate splits from strata, ensuring disjointness.

    Each stratum is shuffled with a seed-based RNG. Items are drawn in order:
    pilot, dev, test, reserve — so earlier splits get first pick from each
    stratum.
    """
    total_needed = sum(split_sizes.values())
    pool_size = sum(len(qids) for qids in strata.values())
    if total_needed > pool_size:
        raise ValueError(
            f"Total requested ({total_needed}) exceeds safe pool size ({pool_size})"
        )

    rng = random.Random(seed)

    # Shuffle within each stratum
    shuffled_strata: dict[str, list[str]] = {}
    for key in sorted(strata.keys()):
        qids = list(strata[key])
        rng.shuffle(qids)
        shuffled_strata[key] = qids

    # Compute proportional allocation per stratum per split.
    # Each split gets split_size items total, distributed across strata
    # proportionally to stratum size relative to the pool.
    split_names = list(split_sizes.keys())
    allocations: dict[str, dict[str, int]] = {name: {} for name in split_names}
    sorted_keys = sorted(shuffled_strata.keys())

    for name in split_names:
        target = split_sizes[name]
        for key in sorted_keys:
            stratum_size = len(shuffled_strata[key])
            share = (stratum_size / pool_size) * target
            allocations[name][key] = int(share)

        # Redistribute rounding shortfall one item at a time using
        # largest-remainder method
        current = sum(allocations[name].values())
        shortfall = target - current
        if shortfall > 0:
            remainders: list[tuple[float, str]] = []
            for key in sorted_keys:
                stratum_size = len(shuffled_strata[key])
                share = (stratum_size / pool_size) * target
                fractional = share - int(share)
                if fractional > 0:
                    remainders.append((fractional, key))
            remainders.sort(key=lambda x: (-x[0], x[1]))
            for _, key in remainders:
                if shortfall <= 0:
                    break
                allocations[name][key] += 1
                shortfall -= 1

    # Verify no stratum is over-allocated across all splits
    for key in sorted_keys:
        total_from_stratum = sum(allocations[name][key] for name in split_names)
        stratum_size = len(shuffled_strata[key])
        if total_from_stratum > stratum_size:
            # Trim from the last split that drew from this stratum
            excess = total_from_stratum - stratum_size
            for name in reversed(split_names):
                if excess <= 0:
                    break
                trim = min(excess, allocations[name][key])
                allocations[name][key] -= trim
                excess -= trim

    # Now draw from shuffled strata
    splits: dict[str, list[str]] = {name: [] for name in split_names}
    stratum_offset: dict[str, int] = {key: 0 for key in shuffled_strata}

    for name in split_names:
        for key in sorted_keys:
            count = allocations[name].get(key, 0)
            start = stratum_offset[key]
            drawn = shuffled_strata[key][start : start + count]
            splits[name].extend(drawn)
            stratum_offset[key] = start + count

    # Sort each split for deterministic output
    for name in split_names:
        splits[name] = sorted(splits[name])

    return splits


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--pilot_n", type=int, default=150)
    p.add_argument("--dev_n", type=int, default=100)
    p.add_argument("--test_n", type=int, default=500)
    p.add_argument("--reserve_n", type=int, default=200)
    p.add_argument(
        "--parquet_path",
        type=str,
        default="data/TriviaQA/rc.nocontext/validation-00000-of-00001.parquet",
    )
    p.add_argument("--output_dir", type=str, default="data/manifests")
    return p.parse_args()


def _write_json(path: Path, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def main() -> None:
    import pandas as pd

    args = _parse_args()
    seed = args.seed
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    split_sizes = {
        "pilot": args.pilot_n,
        "dev": args.dev_n,
        "test": args.test_n,
        "reserve": args.reserve_n,
    }
    split_file_names = {
        "pilot": f"triviaqa_bridge_pilot{args.pilot_n}_seed{seed}.json",
        "dev": f"triviaqa_bridge_dev{args.dev_n}_seed{seed}.json",
        "test": f"triviaqa_bridge_test{args.test_n}_seed{seed}.json",
        "reserve": f"triviaqa_bridge_reserve{args.reserve_n}_seed{seed}.json",
    }
    metadata_name = f"triviaqa_bridge_metadata_seed{seed}.json"

    output_files = [str(out / name) for name in split_file_names.values()]
    output_files.append(str(out / metadata_name))

    # --- Provenance ---
    prov_handle = start_run_provenance(
        args,
        primary_target=args.output_dir,
        output_targets=list(output_files),
        primary_target_is_dir=True,
    )

    try:
        # --- Load parquet ---
        print(f"Loading parquet from {args.parquet_path}...")
        df = pd.read_parquet(args.parquet_path)
        total_raw_rows = len(df)
        print(f"  Raw rows: {total_raw_rows}")

        # --- Deduplicate by question_id ---
        df = df.drop_duplicates(subset="question_id", keep="first").copy()
        unique_qids = len(df)
        print(f"  Unique question_ids after dedup: {unique_qids}")

        # --- Apply curation filters ---
        print("Applying curation filters...")
        df_safe, filter_stats = _apply_curation_filters(df)
        safe_pool_size = len(df_safe)
        print(f"  Safe pool size: {safe_pool_size}")

        total_needed = sum(split_sizes.values())
        if total_needed > safe_pool_size:
            raise ValueError(
                f"Total requested ({total_needed}) exceeds safe pool ({safe_pool_size})"
            )

        # --- Assign strata ---
        print("Assigning strata...")
        strata = _assign_strata(df_safe)
        print(f"  {len(strata)} strata, pool size {safe_pool_size}")

        strata_counts = {k: len(v) for k, v in sorted(strata.items())}

        # --- Proportional sampling ---
        print("Performing stratified proportional sampling...")
        splits = _proportional_sample(strata, split_sizes, seed)

        # --- Verify disjointness and sizes ---
        split_names = list(split_sizes.keys())
        for i, name_a in enumerate(split_names):
            for name_b in split_names[i + 1 :]:
                overlap = set(splits[name_a]) & set(splits[name_b])
                assert not overlap, (
                    f"{name_a} ∩ {name_b} = {len(overlap)} items: {sorted(overlap)[:5]}"
                )

        all_sampled = set()
        for name in split_names:
            all_sampled |= set(splits[name])
        assert len(all_sampled) == total_needed, (
            f"Total sampled {len(all_sampled)} != requested {total_needed}"
        )

        safe_pool_ids = set(df_safe["question_id"].tolist())
        assert all_sampled.issubset(safe_pool_ids), "Sampled IDs not in safe pool"

        for name in split_names:
            assert len(splits[name]) == split_sizes[name], (
                f"{name}: got {len(splits[name])}, expected {split_sizes[name]}"
            )
            print(f"  {name}: {len(splits[name])} questions")

        # --- Compute split strata counts ---
        split_strata_counts: dict[str, dict[str, int]] = {}
        for name in split_names:
            split_set = set(splits[name])
            counts: dict[str, int] = {}
            for stratum_key, qids in sorted(strata.items()):
                n = len(split_set & set(qids))
                if n > 0:
                    counts[stratum_key] = n
            split_strata_counts[name] = counts

        # --- Write split manifests ---
        for name in split_names:
            path = out / split_file_names[name]
            _write_json(path, sorted(splits[name]))
            print(f"  Wrote {path}")

        # --- Write metadata ---
        metadata = {
            "version": 1,
            "seed": seed,
            "parquet_path": args.parquet_path,
            "total_raw_rows": total_raw_rows,
            "unique_qids": unique_qids,
            "curation_filters": filter_stats,
            "safe_pool_size": safe_pool_size,
            "strata_counts": strata_counts,
            "split_strata_counts": split_strata_counts,
            "splits": {name: len(splits[name]) for name in split_names},
            "fingerprints": {
                "safe_pool": fingerprint_ids(list(safe_pool_ids)),
                **{name: fingerprint_ids(splits[name]) for name in split_names},
            },
            "exclusion_sets": {},
        }
        metadata_path = out / metadata_name
        _write_json(metadata_path, metadata)
        print(f"  Wrote {metadata_path}")

        print("Done. All splits verified disjoint.")
        finish_run_provenance(prov_handle, status="completed")

    except BaseException as exc:
        finish_run_provenance(
            prov_handle,
            status=provenance_status_for_exception(exc),
            extra={"error": provenance_error_message(exc)},
        )
        raise


if __name__ == "__main__":
    main()
