"""Derive the k=154 utility-positive augment family + 3 matched-random controls.

Robustness slice for the L2/L3 reviewer critique: the main k=266 utility_selected
set is size-matched to readout_selected but crosses selector_score=0 (only 154
of 509 candidate-pool features have strictly positive validation utility). A
reviewer can argue the size-matched utility family was diluted with neutral or
harmful features. This script:

  - selects every feature with selector_score > 0 from utility_scores.jsonl
  - layer-matches 3 random zero-weight SAE seeds to the positive selection
  - writes manifests consumable by run_intervention.py (same schema as
    utility_selected_features.json)

It reuses the helpers in select_faitheval_sae_utility_features.py so the
manifest schema, matching policy, and provenance handling stay bit-identical
to the main selector.

Inputs (from the completed selector run):
  - selector/utility_scores.jsonl
  - selector/feature_stats.json
  - selector/utility_selected_features.json (source of extraction_metadata)
  - selector/selector_summary.json (source of split_metadata / prompt_style / alpha)
  - models/sae_detector.pkl (classifier — enumerate zero-weight pool)

Outputs (written into selector_dir):
  - utility_positive_selected_features.json
  - matched_random_positive_seed_{0,1,2}_features.json
  - utility_positive_summary.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from intervene_sae import (
    build_sae_feature_entries,
    get_zero_weight_sae_feature_indices,
)
from select_faitheval_sae_utility_features import (
    _feature_id,
    feature_manifest_with_selector_metadata,
    layer_histogram,
    match_random_zero_weight_features,
    weight_sign_counts,
)
from utils import (
    finish_run_provenance,
    fingerprint_ids,
    json_dumps,
    provenance_error_message,
    provenance_status_for_exception,
    start_run_provenance,
)


DEFAULT_SELECTOR_DIR = (
    "data/gemma3_4b/intervention/faitheval_sae_utility_selector/selector"
)
DEFAULT_CLASSIFIER_PATH = "models/sae_detector.pkl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selector_dir", type=str, default=DEFAULT_SELECTOR_DIR)
    parser.add_argument("--classifier_path", type=str, default=DEFAULT_CLASSIFIER_PATH)
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json_dumps(payload), encoding="utf-8")


def main() -> None:
    args = parse_args()
    selector_dir = Path(args.selector_dir)

    positive_manifest_path = selector_dir / "utility_positive_selected_features.json"
    matched_manifest_paths = [
        selector_dir / f"matched_random_positive_seed_{seed}_features.json"
        for seed in range(3)
    ]
    augment_summary_path = selector_dir / "utility_positive_summary.json"
    output_targets = [
        positive_manifest_path,
        *matched_manifest_paths,
        augment_summary_path,
    ]

    provenance_handle = start_run_provenance(
        args,
        primary_target=selector_dir,
        output_targets=[str(path) for path in output_targets],
        primary_target_is_dir=True,
    )
    provenance_status = "completed"
    provenance_extra: dict[str, Any] = {}

    try:
        utility_selected_payload = _load_json(
            selector_dir / "utility_selected_features.json"
        )
        extraction_metadata = utility_selected_payload["extraction_metadata"]
        layer_indices = [int(idx) for idx in extraction_metadata["layer_indices"]]
        d_sae = int(extraction_metadata["d_sae"])

        selector_summary = _load_json(selector_dir / "selector_summary.json")
        split_metadata = selector_summary["split_metadata"]
        prompt_style = str(selector_summary["prompt_style"])
        alpha = float(selector_summary["alpha"])

        feature_stats_payload = _load_json(selector_dir / "feature_stats.json")
        stats_by_flat_idx: dict[int, dict[str, Any]] = {
            int(row["flat_idx"]): row for row in feature_stats_payload["features"]
        }

        utility_rows = _load_jsonl(selector_dir / "utility_scores.jsonl")
        positive_rows = [
            row for row in utility_rows if float(row["selector_score"]) > 0.0
        ]
        if not positive_rows:
            raise ValueError(
                "utility_scores.jsonl contains zero strictly-positive selector "
                "scores; cannot derive utility-positive augment"
            )
        positive_rows.sort(
            key=lambda row: (-float(row["selector_score"]), int(row["flat_idx"]))
        )

        positive_manifest = feature_manifest_with_selector_metadata(
            positive_rows,
            extraction_metadata=extraction_metadata,
            selector_name="validation_logprob_margin_positive",
            split_metadata=split_metadata,
            alpha=alpha,
            prompt_style=prompt_style,
        )
        _write_json(positive_manifest_path, positive_manifest)

        zero_weight_pool = [
            {**entry, **stats_by_flat_idx[int(entry["flat_idx"])]}
            for entry in build_sae_feature_entries(
                get_zero_weight_sae_feature_indices(args.classifier_path).tolist(),
                layer_indices=layer_indices,
                d_sae=d_sae,
            )
        ]

        matched_layer_histograms: list[dict[str, int]] = []
        matched_fingerprints: list[str] = []
        for seed in range(3):
            matched = match_random_zero_weight_features(
                positive_rows,
                zero_weight_pool,
                stats_by_flat_idx,
                seed=seed,
            )
            matched_manifest = feature_manifest_with_selector_metadata(
                matched,
                extraction_metadata=extraction_metadata,
                selector_name=f"layer_matched_zero_weight_positive_seed_{seed}",
                split_metadata=split_metadata,
                alpha=alpha,
                prompt_style=prompt_style,
            )
            _write_json(matched_manifest_paths[seed], matched_manifest)
            matched_layer_histograms.append(layer_histogram(matched))
            matched_fingerprints.append(
                fingerprint_ids([_feature_id(feature) for feature in matched])
            )

        positive_layer_histogram = layer_histogram(positive_rows)
        augment_summary = {
            "schema_version": "faitheval_sae_utility_positive_augment/v1",
            "benchmark": "faitheval",
            "prompt_style": prompt_style,
            "sae_steering_mode": "delta_only",
            "alpha": alpha,
            "augment_design": {
                "review_loophole_targeted": (
                    "Size-matched k=266 utility family crosses selector_score=0 "
                    "(only 154/509 features have strictly positive validation "
                    "utility). This slice selects only the positive-utility "
                    "features so paired comparisons cannot be attacked as "
                    "'diluted by neutral or harmful features to match readout "
                    "cardinality'."
                ),
                "selection_rule": "selector_score > 0",
                "size_matching_policy": (
                    "Cardinality = number of strictly-positive-utility features; "
                    "no size match to readout."
                ),
                "matched_random_policy": (
                    "Zero-weight SAE features layer-matched to the utility-"
                    "positive selection by activation frequency and decoder "
                    "norm. 3 seeds."
                ),
                "noop_reference": (
                    "Uses the shared noop (α=1.0) held-out bundle produced by "
                    "faitheval_sae_utility_selector.sh Phase 1."
                ),
            },
            "family_diagnostics": {
                "utility_positive_selected": {
                    "k": len(positive_rows),
                    "fingerprint": fingerprint_ids(
                        [_feature_id(feature) for feature in positive_rows]
                    ),
                    "layer_histogram": positive_layer_histogram,
                    "weight_sign_counts": weight_sign_counts(positive_rows),
                    "mean_selector_score": float(
                        sum(
                            float(feature["selector_score"])
                            for feature in positive_rows
                        )
                        / max(1, len(positive_rows))
                    ),
                    "min_selector_score": float(
                        min(
                            float(feature["selector_score"])
                            for feature in positive_rows
                        )
                    ),
                    "max_selector_score": float(
                        max(
                            float(feature["selector_score"])
                            for feature in positive_rows
                        )
                    ),
                },
                "matched_random_positive_layer_histograms": matched_layer_histograms,
                "matched_random_positive_fingerprints": matched_fingerprints,
            },
        }
        _write_json(augment_summary_path, augment_summary)

        provenance_extra["utility_positive_k"] = len(positive_rows)
        provenance_extra["matched_seeds"] = 3
        provenance_extra["utility_positive_layer_histogram"] = positive_layer_histogram
    except BaseException as exc:
        provenance_status = provenance_status_for_exception(exc)
        provenance_extra["error"] = provenance_error_message(exc)
        raise
    finally:
        finish_run_provenance(provenance_handle, provenance_status, provenance_extra)


if __name__ == "__main__":
    main()
