"""Materialize JBB_100, HarmBench_test_159, and StrongREJECT_313 eval sets.

Splits data/contrastive/refusal/upstream/harmful_test.json (572 records)
into three standalone JSONL files by category naming convention:

- JBB_100: 10 title-case categories with '/' separators (100 records)
- HarmBench_test_159: lowercase + underscore categories (159 records)
- StrongREJECT_313: title-case categories with 'and' (313 records)

Usage:
    uv run python scripts/materialize_harmful_eval_sets.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils import (
    finish_run_provenance,
    json_dumps,
    provenance_error_message,
    provenance_status_for_exception,
    start_run_provenance,
)

# Category → source mapping
JBB_CATEGORIES = {
    "Harassment/Discrimination",
    "Malware/Hacking",
    "Physical harm",
    "Economic harm",
    "Fraud/Deception",
    "Disinformation",
    "Sexual/Adult content",
    "Privacy",
    "Expert advice",
    "Government decision-making",
}

HARMBENCH_CATEGORIES = {
    "illegal",
    "cybercrime_intrusion",
    "chemical_biological",
    "misinformation_disinformation",
    "harmful",
    "harassment_bullying",
}


def classify_category(category: str) -> str:
    """Map a category string to its eval set name."""
    if category in JBB_CATEGORIES:
        return "jbb"
    if category in HARMBENCH_CATEGORIES:
        return "harmbench"
    # StrongREJECT: title-case with "and" (remainder)
    return "strongreject"


def main():
    parser = argparse.ArgumentParser(
        description="Materialize harmful eval sets from upstream pool"
    )
    parser.add_argument(
        "--input_path",
        type=str,
        default="data/contrastive/refusal/upstream/harmful_test.json",
        help="Path to harmful_test.json (572-record pool)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/contrastive/refusal/eval",
        help="Output directory for eval set JSONL files",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    jbb_path = output_dir / "jbb_100.jsonl"
    harmbench_path = output_dir / "harmbench_test_159.jsonl"
    strongreject_path = output_dir / "strongreject_313.jsonl"
    metadata_path = output_dir / "eval_sets_metadata.json"

    provenance_handle = start_run_provenance(
        args,
        primary_target=str(output_dir),
        output_targets=[
            str(jbb_path),
            str(harmbench_path),
            str(strongreject_path),
            str(metadata_path),
        ],
        primary_target_is_dir=True,
    )
    provenance_status = "completed"
    provenance_extra: dict = {}

    try:
        # Load upstream pool
        with open(args.input_path, "r", encoding="utf-8") as f:
            records = json.load(f)
        print(f"Loaded {len(records)} records from {args.input_path}")

        # Partition by category
        buckets: dict[str, list[dict]] = {
            "jbb": [],
            "harmbench": [],
            "strongreject": [],
        }
        for record in records:
            bucket = classify_category(record["category"])
            buckets[bucket].append(record)

        # Validate counts
        counts = {name: len(items) for name, items in buckets.items()}
        print(
            f"Counts: JBB={counts['jbb']}, HarmBench={counts['harmbench']}, "
            f"StrongREJECT={counts['strongreject']}"
        )

        expected = {"jbb": 100, "harmbench": 159, "strongreject": 313}
        for name, expected_count in expected.items():
            actual = counts[name]
            if actual != expected_count:
                raise ValueError(
                    f"{name}: expected {expected_count} records, got {actual}"
                )

        total = sum(counts.values())
        if total != len(records):
            raise ValueError(f"Partition total {total} != input total {len(records)}")

        # Write JSONL files
        source_map = {
            "jbb": "jailbreakbench",
            "harmbench": "harmbench",
            "strongreject": "strongreject",
        }
        id_prefix_map = {"jbb": "jbb", "harmbench": "hb", "strongreject": "sr"}
        path_map = {
            "jbb": jbb_path,
            "harmbench": harmbench_path,
            "strongreject": strongreject_path,
        }

        category_distributions: dict[str, dict[str, int]] = {}

        for bucket_name, items in buckets.items():
            source = source_map[bucket_name]
            prefix = id_prefix_map[bucket_name]
            out_path = path_map[bucket_name]

            cat_dist: dict[str, int] = {}
            with open(out_path, "w", encoding="utf-8") as f:
                for idx, item in enumerate(items):
                    record = {
                        "id": f"{prefix}_{idx:03d}",
                        "text": item["instruction"],
                        "category": item["category"],
                        "source": source,
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    cat_dist[item["category"]] = cat_dist.get(item["category"], 0) + 1

            category_distributions[bucket_name] = dict(sorted(cat_dist.items()))
            print(f"Wrote {len(items)} records to {out_path}")

        # Write metadata
        metadata = {
            "input_path": str(Path(args.input_path).resolve()),
            "input_total": len(records),
            "sets": {
                "jbb_100": {
                    "file": str(jbb_path),
                    "count": counts["jbb"],
                    "source": "jailbreakbench",
                    "categories": category_distributions["jbb"],
                },
                "harmbench_test_159": {
                    "file": str(harmbench_path),
                    "count": counts["harmbench"],
                    "source": "harmbench",
                    "categories": category_distributions["harmbench"],
                },
                "strongreject_313": {
                    "file": str(strongreject_path),
                    "count": counts["strongreject"],
                    "source": "strongreject",
                    "categories": category_distributions["strongreject"],
                },
            },
        }
        metadata_path.write_text(json_dumps(metadata), encoding="utf-8")
        print(f"Wrote metadata to {metadata_path}")

    except BaseException as exc:
        provenance_status = provenance_status_for_exception(exc)
        provenance_extra["error"] = provenance_error_message(exc)
        raise
    finally:
        finish_run_provenance(provenance_handle, provenance_status, provenance_extra)


if __name__ == "__main__":
    main()
