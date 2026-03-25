#!/usr/bin/env python3
"""Analyse smoke-test JSONL files for preflight validation.

Usage:
    uv run python scripts/analyse_smoke_test.py --input_dir <smoke_test_dir>
"""

import argparse
import json
import os
import sys
from pathlib import Path


def load_records(directory: str) -> list[dict]:
    """Load all JSONL records from alpha_*.jsonl files."""
    records = []
    for fname in sorted(Path(directory).glob("alpha_*.jsonl")):
        with open(fname) as f:
            for line in f:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", required=True, help="Smoke test output dir")
    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        print(f"ERROR: {args.input_dir} does not exist")
        sys.exit(1)

    records = load_records(args.input_dir)
    if not records:
        print("ERROR: No records found")
        sys.exit(1)

    print(f"{'=' * 60}")
    print(f"Smoke Test Analysis: {args.input_dir}")
    print(f"{'=' * 60}")
    print(f"Total records: {len(records)}")

    # Group by alpha
    by_alpha = {}
    for r in records:
        a = r["alpha"]
        by_alpha.setdefault(a, []).append(r)

    for alpha in sorted(by_alpha):
        recs = by_alpha[alpha]
        print(f"\n--- Alpha {alpha:.1f} ({len(recs)} records) ---")

        # Check timings presence
        has_timings = [r for r in recs if "timings" in r]
        if not has_timings:
            print("  WARNING: No records have timings!")
            continue

        print(f"  Records with timings: {len(has_timings)}/{len(recs)}")

        timings = [r["timings"] for r in has_timings]

        # Template time (should be ~0 if cached)
        template_s = [t["template_s"] for t in timings]
        print("\n  template_s (should be ~0 if cached):")
        print(
            f"    mean={sum(template_s) / len(template_s):.4f}s  "
            f"max={max(template_s):.4f}s"
        )

        # Generate time
        gen_s = [t["generate_s"] for t in timings]
        print("\n  generate_s:")
        print(
            f"    mean={sum(gen_s) / len(gen_s):.1f}s  "
            f"min={min(gen_s):.1f}s  max={max(gen_s):.1f}s"
        )

        # Total time
        total_s = [t["total_s"] for t in timings]
        print("\n  total_s:")
        print(
            f"    mean={sum(total_s) / len(total_s):.1f}s  "
            f"min={min(total_s):.1f}s  max={max(total_s):.1f}s"
        )

        # Token counts
        gen_tok = [t["generated_tokens"] for t in timings]
        print("\n  generated_tokens:")
        print(
            f"    mean={sum(gen_tok) / len(gen_tok):.0f}  "
            f"min={min(gen_tok)}  max={max(gen_tok)}"
        )

        # Cap hit rate
        cap_hits = [t["hit_token_cap"] for t in timings]
        hit_rate = sum(cap_hits) / len(cap_hits)
        print(f"\n  hit_token_cap: {sum(cap_hits)}/{len(cap_hits)} ({hit_rate:.0%})")
        if hit_rate > 0.3:
            print("  ⚠️  HIGH CAP-HIT RATE — truncation still a concern")
        elif hit_rate > 0.1:
            print("  ⚠  Moderate cap-hit rate — worth monitoring")
        else:
            print("  ✓  Low cap-hit rate")

        # Tokens/sec
        tok_per_s = [
            t["generated_tokens"] / t["generate_s"]
            for t in timings
            if t["generate_s"] > 0
        ]
        if tok_per_s:
            print("\n  throughput (tok/s):")
            print(
                f"    mean={sum(tok_per_s) / len(tok_per_s):.1f}  "
                f"min={min(tok_per_s):.1f}  max={max(tok_per_s):.1f}"
            )

    # Estimate full run time
    all_timings = [r["timings"] for r in records if "timings" in r]
    if all_timings:
        mean_total = sum(t["total_s"] for t in all_timings) / len(all_timings)
        n_alphas = len(by_alpha)
        total_samples = 500  # 100 behaviors × 5 templates
        est_hours = (mean_total * total_samples * n_alphas) / 3600
        print(f"\n{'=' * 60}")
        print(f"Full run estimate ({n_alphas} alphas × {total_samples} samples):")
        print(f"  Mean time/sample: {mean_total:.1f}s")
        print(f"  Estimated total: {est_hours:.1f} hours")
        print(f"{'=' * 60}")

    # Check for provenance sidecar
    prov_files = list(Path(args.input_dir).glob("*.provenance*.json"))
    if prov_files:
        print(f"\nProvenance files: {[p.name for p in prov_files]}")
    else:
        print("\nWARNING: No provenance sidecar found!")


if __name__ == "__main__":
    main()
