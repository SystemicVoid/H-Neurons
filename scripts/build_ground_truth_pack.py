"""Build the ground-truth evidence pack under notes/ground-truth/."""

from __future__ import annotations

import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


JsonDict = dict[str, Any]

ROOT = Path(__file__).resolve().parents[1]
GROUND_TRUTH_DIR = ROOT / "notes" / "ground-truth"
SOURCES_PATH = GROUND_TRUTH_DIR / "_sources.json"
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 42
NEUTRAL_STRATUM_BANNED_TOKENS = {
    "blind",
    "older",
    "universal",
    "solo",
    "extra",
    "recovered",
    "spot",
}
MARKDOWN_FALLBACK_CLAIMS = {
    "probe-selector-pilot-best-effect",
    "gradient-ranked-pilot-effect",
    "probe-causal-jaccard",
    "historical-april-8-legacy-ruler-causal-full-500-effect",
    "strongreject-gpt-4o-dev-accuracy-after-rerun",
    "strongreject-gpt-4o-mini-dev-accuracy-prior",
    "holdout-gap-after-strongreject-model-upgrade",
    "false-negatives-recovered-by-strongreject-model-upgrade",
    "persistent-strongreject-false-negatives-after-upgrade",
    "contamination-bug-borderline-records-reclassified",
    "contamination-bug-strict-harmfulness-inflation",
    "h-minus-random-permutation-test",
}
PROVENANCE_CLAIM_METRIC_IDS: dict[str, list[str]] = {
    "random-neuron-null-summary-five-unconstrained-seeds": [
        "intervention.faitheval.random_mean_slope_pp_per_alpha"
    ],
    "random-neuron-null-summary-all-eight-seeds": [
        "intervention.faitheval.random_max_slope_pp_per_alpha"
    ],
    "random-neuron-specificity": [
        "intervention.faitheval.seedwise_positive_differences"
    ],
    "faitheval-evaluation-size": ["intervention.faitheval.sample_size"],
    "bioasq-net-accuracy-effect": ["intervention.bioasq.delta_0_to_max_pp"],
    "bioasq-sample-size": ["intervention.bioasq.sample_size"],
    "bioasq-responses-behaviorally-changed": [
        "intervention.bioasq.changed_response_count"
    ],
    "iti-mc1-effect-held-out-truthfulqa-folds": [
        "transfer.truthfulqa_mc.mc1.iti.delta_pp"
    ],
    "iti-mc2-effect-held-out-truthfulqa-folds": [
        "transfer.truthfulqa_mc.mc2.iti.delta_pp"
    ],
    "truthfulqa-flip-counts": [
        "transfer.truthfulqa_mc.mc1.iti.wrong_to_right",
        "transfer.truthfulqa_mc.mc1.iti.right_to_wrong",
    ],
    "simpleqa-correct-answer-rate-at-0-0": ["transfer.simpleqa.compliance.0.0"],
    "simpleqa-correct-answer-rate-at-8-0": ["transfer.simpleqa.compliance.8.0"],
    "simpleqa-paired-compliance-delta-0-0-8-0": [
        "transfer.simpleqa.compliance_delta_0_to_8_pp"
    ],
    "simpleqa-attempt-rate-at-0-0": ["transfer.simpleqa.attempt_rate_alpha_0"],
    "simpleqa-attempt-rate-at-8-0": ["transfer.simpleqa.attempt_rate_alpha_8"],
    "simpleqa-paired-attempt-rate-delta-0-0-8-0": [
        "transfer.simpleqa.attempt_delta_0_to_8_pp"
    ],
    "bridge-baseline-adjudicated-accuracy": [
        "transfer.bridge.baseline_adjudicated_accuracy"
    ],
    "e0-8-0-adjudicated-delta": ["transfer.bridge.adjudicated_accuracy_delta_pp"],
    "e0-8-0-deterministic-delta": ["transfer.bridge.deterministic_accuracy_delta_pp"],
    "bridge-paired-significance": ["transfer.bridge.mcnemar_p"],
    "bridge-flip-counts": [
        "transfer.bridge.base_correct_iti_wrong",
        "transfer.bridge.base_wrong_iti_correct",
    ],
    "wrong-entity-substitution-adjudicated-r-w": [
        "measurement.bridge_irr.right_to_wrong.wrong_entity_substitution"
    ],
    "evasion-factual-denial-adjudicated-r-w": [
        "measurement.bridge_irr.right_to_wrong.evasion_or_factual_denial"
    ],
    "answer-dilution-verbosity-adjudicated-r-w": [
        "measurement.bridge_irr.right_to_wrong.answer_dilution"
    ],
    "formal-refusal-among-right-wrong-flips-adjudicated": [
        "measurement.bridge_irr.right_to_wrong.formal_refusal"
    ],
    "not-attempted-counts-baseline-iti": [
        "transfer.bridge.baseline_not_attempted_count",
        "transfer.bridge.iti_not_attempted_count",
    ],
    "bridge-test-set-size": ["transfer.bridge.sample_size"],
    "falseqa-sample-size": ["intervention.falseqa.sample_size"],
    "falseqa-slope": ["intervention.falseqa.slope_pp_per_alpha"],
    "falseqa-no-op-max-1-0-3-0": ["intervention.falseqa.delta_noop_to_max_pp"],
    "falseqa-full-sweep-0-0-3-0": ["intervention.falseqa.delta_0_to_max_pp"],
    "h-neuron-faitheval-slope": ["intervention.faitheval.anti.slope_pp_per_alpha"],
    "h-neuron-faitheval-no-op-max-1-0-3-0": [
        "intervention.faitheval.anti.delta_noop_to_max_pp"
    ],
    "h-neuron-faitheval-full-sweep-0-0-3-0": [
        "intervention.faitheval.anti.delta_0_to_max_pp"
    ],
    "binary-harmful-endpoint-shift-0-0-3-0": [
        "measurement.jailbreak.binary.alpha_0_count",
        "measurement.jailbreak.binary.alpha_3_count",
    ],
    "h-neuron-graded-harmfulness-slope": [
        "measurement.jailbreak.v2.h_slope_csv2_yes_pp_per_alpha"
    ],
    "h-neuron-graded-endpoint-shift-0-0-3-0": [
        "measurement.jailbreak.v2.endpoint_delta_csv2_yes_pp"
    ],
    "random-neuron-graded-slope-seed-0-control": [
        "measurement.jailbreak.v2.random_mean_slope_csv2_yes_pp_per_alpha"
    ],
    "h-minus-random-slope-difference": [
        "measurement.jailbreak.v2.gap_h_minus_random_mean_pp_per_alpha"
    ],
    "holdout-evaluator-sample-size": ["measurement.holdout.sample_size"],
    "evaluator-tie-statement-kept-in-main-text": [
        "measurement.holdout.csv2v3_vs_sr_mcnemar_p",
        "measurement.holdout.csv2v3_vs_sr_discordant_count",
    ],
    "pairwise-holdout-significance-summary": [
        "measurement.holdout.min_pairwise_mcnemar_p"
    ],
}
SURFACE_METADATA_PREFIXES = (
    "schema_version",
    "generated_at",
    "generated_by",
    "source_files",
    "runtime",
    "provenance.source_files",
    "provenance.notes",
)
SURFACE_METADATA_SEGMENTS = {
    "bootstrap",
}


def load_json(path: Path) -> JsonDict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[JsonDict]:
    rows: list[JsonDict] = []
    buffer = ""
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() and not buffer:
            continue
        buffer = f"{buffer}\n{line}" if buffer else line
        try:
            rows.append(json.loads(buffer, strict=False))
            buffer = ""
        except json.JSONDecodeError:
            continue
    if buffer.strip():
        rows.append(json.loads(buffer, strict=False))
    return rows


def load_keyed_jsonl(path: Path) -> dict[str, JsonDict]:
    keyed: dict[str, JsonDict] = {}
    for row in load_jsonl(path):
        if len(row) == 1:
            key, value = next(iter(row.items()))
            value = dict(value)
            value["id"] = key
            keyed[key] = value
        else:
            keyed[str(row["id"])] = dict(row)
    return keyed


def dump_jsonl(path: Path, rows: list[JsonDict]) -> None:
    text = "".join(
        json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n" for row in rows
    )
    path.write_text(text, encoding="utf-8")


def slugify(value: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return cleaned or "row"


def unique_list(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(values))


def normalize_token(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_") or "none"


def alpha_label(alpha: float | str) -> str:
    return f"alpha_{float(alpha):.1f}"


def alpha_code(alpha: float | str) -> str:
    return alpha_label(alpha).replace(".", "_")


def response_text(row: JsonDict) -> str:
    return str(row.get("response", row.get("chosen", "")))


def response_length(row: JsonDict) -> int:
    return len(response_text(row))


def is_numeric_scalar(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def should_include_surface_scalar(locator: str, value: Any) -> bool:
    if not is_numeric_scalar(value):
        return False
    if any(locator.startswith(prefix) for prefix in SURFACE_METADATA_PREFIXES):
        return False
    segments = re.split(r"[.\[]", locator)
    return not any(
        segment.rstrip("]") in SURFACE_METADATA_SEGMENTS for segment in segments
    )


def iter_surface_scalars(
    value: Any, prefix: str = ""
) -> Iterable[tuple[str, float | int]]:
    if isinstance(value, dict):
        for key in sorted(value):
            next_prefix = f"{prefix}.{key}" if prefix else key
            yield from iter_surface_scalars(value[key], next_prefix)
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            next_prefix = f"{prefix}[{index}]"
            yield from iter_surface_scalars(item, next_prefix)
        return
    if should_include_surface_scalar(prefix, value):
        yield prefix, value


def percentile(sorted_values: list[float], fraction: float) -> float:
    if not sorted_values:
        raise ValueError("percentile requires at least one value")
    if len(sorted_values) == 1:
        return sorted_values[0]
    index = fraction * (len(sorted_values) - 1)
    lower = int(index)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = index - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def bootstrap_paired_mean(
    values_a: list[float],
    values_b: list[float],
    *,
    scale: float = 1.0,
    seed: int = BOOTSTRAP_SEED,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
) -> JsonDict:
    if len(values_a) != len(values_b):
        raise ValueError("paired bootstrap requires equal-length vectors")
    if not values_a:
        raise ValueError("paired bootstrap requires at least one paired value")
    diffs = [(b - a) * scale for a, b in zip(values_a, values_b, strict=True)]
    estimate = sum(diffs) / len(diffs)
    rng = random.Random(seed)
    draws: list[float] = []
    for _ in range(n_resamples):
        sample = [diffs[rng.randrange(len(diffs))] for _ in range(len(diffs))]
        draws.append(sum(sample) / len(sample))
    draws.sort()
    return {
        "estimate": estimate,
        "ci_lower": percentile(draws, 0.025),
        "ci_upper": percentile(draws, 0.975),
        "ci_level": 0.95,
        "ci_method": "bootstrap_percentile_paired",
        "n": len(diffs),
        "paired": True,
    }


def metric_row(
    *,
    metric_id: str,
    family: str,
    act_tags: list[str],
    benchmark: str,
    model: str,
    intervention_family: str,
    metric_name: str,
    comparison_type: str,
    condition_a: str,
    condition_b: str,
    estimate: float | int,
    unit: str,
    n: int | None,
    paired: bool,
    ci_lower: float | None,
    ci_upper: float | None,
    ci_level: float | None,
    ci_method: str | None,
    source_artifact_ids: list[str],
    source_format: str,
    source_locators: list[JsonDict],
) -> JsonDict:
    return {
        "metric_id": metric_id,
        "family": family,
        "act_tags": act_tags,
        "benchmark": benchmark,
        "model": model,
        "intervention_family": intervention_family,
        "metric_name": metric_name,
        "comparison_type": comparison_type,
        "condition_a": condition_a,
        "condition_b": condition_b,
        "estimate": estimate,
        "unit": unit,
        "n": n,
        "paired": paired,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "ci_level": ci_level,
        "ci_method": ci_method,
        "source_artifact_ids": source_artifact_ids,
        "source_format": source_format,
        "source_locators": source_locators,
    }


def example_row(
    *,
    example_id: str,
    family: str,
    act_tags: list[str],
    benchmark: str,
    source_artifact_id: str,
    source_artifact_ids: list[str],
    sample_id: str,
    selection_stratum: str,
    selection_rank: int,
    condition_metadata: JsonDict,
    prompt_or_question: str,
    response_text: str,
    raw_label_fields: JsonDict,
    judge_fields: JsonDict,
    paired_example_ids: list[str],
) -> JsonDict:
    return {
        "example_id": example_id,
        "family": family,
        "act_tags": act_tags,
        "benchmark": benchmark,
        "source_artifact_id": source_artifact_id,
        "source_artifact_ids": source_artifact_ids,
        "sample_id": sample_id,
        "selection_stratum": selection_stratum,
        "selection_rank": selection_rank,
        "condition_metadata": condition_metadata,
        "prompt_or_question": prompt_or_question,
        "response_text": response_text,
        "raw_label_fields": raw_label_fields,
        "judge_fields": judge_fields,
        "paired_example_ids": paired_example_ids,
    }


def ci_fields(
    value: JsonDict,
) -> tuple[float | None, float | None, float | None, str | None]:
    ci = value.get("ci")
    if not isinstance(ci, dict):
        return None, None, None, None
    return (
        ci.get("lower"),
        ci.get("upper"),
        ci.get("level"),
        ci.get("method"),
    )


def first_sentence_count(text: str, pattern: str) -> str:
    match = re.search(pattern, text, flags=re.MULTILINE)
    if not match:
        raise ValueError(f"pattern not found: {pattern}")
    return match.group(1)


class Builder:
    def __init__(self, manifest: JsonDict) -> None:
        self.manifest = manifest
        self.artifact_specs: list[JsonDict] = manifest["artifacts"]
        self.family_specs: dict[str, JsonDict] = {
            family["id"]: family for family in manifest["families"]
        }
        self.artifact_by_id: dict[str, JsonDict] = {
            artifact["artifact_id"]: artifact for artifact in self.artifact_specs
        }
        self.metrics: list[JsonDict] = []
        self.examples: list[JsonDict] = []
        self.crosswalk: list[JsonDict] = []

    def path(self, relative_path: str) -> Path:
        return ROOT / relative_path

    def add_metric(self, row: JsonDict) -> None:
        self.metrics.append(row)

    def add_example(self, row: JsonDict) -> None:
        self.examples.append(row)

    def source_locator(
        self,
        artifact_id: str,
        locator: str,
        *,
        derivation: str = "direct",
    ) -> JsonDict:
        spec = self.artifact_by_id[artifact_id]
        return {
            "artifact_id": artifact_id,
            "source_path": spec["source_path"],
            "locator": locator,
            "derivation": derivation,
        }

    def metric_ids_by_source_locator(self) -> dict[tuple[str, str], list[str]]:
        mapping: dict[tuple[str, str], list[str]] = defaultdict(list)
        for row in self.metrics:
            for locator in row["source_locators"]:
                mapping[(locator["source_path"], locator["locator"])].append(
                    row["metric_id"]
                )
        return {key: sorted(set(value)) for key, value in mapping.items()}

    def match_metric_ids_for_ref(
        self,
        ref: JsonDict,
        metric_ids_by_locator: dict[tuple[str, str], list[str]],
    ) -> set[str]:
        if ref["path"] is None or ref["locator"] is None:
            return set()
        exact = metric_ids_by_locator.get((ref["path"], ref["locator"]))
        if exact:
            return set(exact)
        matches: set[str] = set()
        needle = str(ref["locator"]).strip("`")
        if needle.startswith("..."):
            needle = needle[3:]
        for (source_path, locator), metric_ids in metric_ids_by_locator.items():
            if source_path != ref["path"]:
                continue
            if (
                locator == needle
                or locator.startswith(f"{needle}.")
                or needle.startswith(f"{locator}.")
                or locator.endswith(needle)
            ):
                matches.update(metric_ids)
        return matches

    def _select_quantile_candidates(
        self,
        rows: list[JsonDict],
        *,
        score_key: str,
        tiebreak_key: str,
        fractions: list[float],
    ) -> list[JsonDict]:
        scores = sorted(float(row[score_key]) for row in rows)
        chosen: list[JsonDict] = []
        used_keys: set[str] = set()
        for fraction in fractions:
            target = percentile(scores, fraction)
            ranked = sorted(
                rows,
                key=lambda row: (
                    abs(float(row[score_key]) - target),
                    str(row[tiebreak_key]),
                ),
            )
            for row in ranked:
                if str(row[tiebreak_key]) not in used_keys:
                    chosen.append(row)
                    used_keys.add(str(row[tiebreak_key]))
                    break
        return chosen

    def artifact_rows(self) -> list[JsonDict]:
        rows: list[JsonDict] = []
        for spec in self.artifact_specs:
            source_path = self.path(spec["source_path"])
            if not source_path.exists():
                raise FileNotFoundError(source_path)
            rows.append(
                {
                    "artifact_id": spec["artifact_id"],
                    "family": spec["family"],
                    "act_tags": spec["act_tags"],
                    "source_path": spec["source_path"],
                    "content_type": spec["content_type"],
                    "granularity": spec["granularity"],
                    "benchmark": spec["benchmark"],
                    "model": spec["model"],
                    "intervention_family": spec["intervention_family"],
                    "canonical_role": spec["canonical_role"],
                    "replacement_status": spec["replacement_status"],
                }
            )
        return rows

    def build_metrics(self) -> None:
        extractors = {
            "pipeline_summary": self.extract_pipeline_summary,
            "classifier_disjoint": self.extract_classifier_disjoint,
            "classifier_overlap": self.extract_classifier_overlap,
            "classifier_sae": self.extract_classifier_sae,
            "classifier_structure": self.extract_classifier_structure,
            "intervention_sweep": self.extract_intervention_sweep,
            "faitheval_control_comparison": self.extract_faitheval_control_comparison,
            "faitheval_slope_difference": self.extract_faitheval_slope_difference,
            "faitheval_remap": self.extract_faitheval_remap,
            "faitheval_sae_comparison": self.extract_faitheval_sae_comparison,
            "faitheval_sae_difference": self.extract_faitheval_sae_difference,
            "falseqa_results": self.extract_falseqa_results,
            "bioasq_results": self.extract_bioasq_results,
            "bioasq_control_comparison": self.extract_bioasq_control_comparison,
            "bioasq_audit_gapfill": self.extract_bioasq_audit_gapfill,
            "jailbreak_results": self.extract_jailbreak_results,
            "jailbreak_control_v2": self.extract_jailbreak_control_v2,
            "jailbreak_control_v3": self.extract_jailbreak_control_v3,
            "holdout_comparison": self.extract_holdout_comparison,
            "csv2_v3_results": self.extract_csv2_v3_results,
            "strongreject_results": self.extract_strongreject_results,
            "available_jailbreak_evaluator_comparison": self.extract_available_jailbreak_evaluator_comparison,
            "bridge_irr_summary": self.extract_bridge_irr_summary,
            "bridge_phase3": self.extract_bridge_phase3,
            "truthfulqa_mc_pool": self.extract_truthfulqa_mc_pool,
            "simpleqa_results": self.extract_simpleqa_results,
            "neuron_4288": self.extract_neuron_4288,
            "verbosity_confound": self.extract_verbosity_confound,
            "refusal_overlap": self.extract_refusal_overlap,
            "swing_characterization": self.extract_swing_characterization,
            "d7_comparison": self.extract_d7_comparison,
        }
        for spec in self.artifact_specs:
            extractor_name = spec["extractor"]
            if extractor_name == "none":
                continue
            extractor = extractors[extractor_name]
            extractor(spec)

    def build_examples(self) -> None:
        self.build_readout_examples()
        self.build_intervention_examples()
        self.build_measurement_examples()
        self.build_transfer_examples()
        self.build_mechanism_examples()

    def add_metric_from_point(
        self,
        *,
        metric_id: str,
        spec: JsonDict,
        metric_name: str,
        comparison_type: str,
        condition_a: str,
        condition_b: str,
        point: float | int,
        unit: str,
        n: int | None = None,
        paired: bool = False,
        ci_lower: float | None = None,
        ci_upper: float | None = None,
        ci_level: float | None = None,
        ci_method: str | None = None,
        locator: str | None = None,
        source_artifact_ids: list[str] | None = None,
        source_locators: list[JsonDict] | None = None,
    ) -> None:
        if source_locators is None:
            if locator is None:
                locator = "root"
            source_locators = [
                self.source_locator(spec["artifact_id"], locator, derivation="direct")
            ]
        if source_artifact_ids is None:
            source_artifact_ids = unique_list(
                locator_row["artifact_id"] for locator_row in source_locators
            )
        self.add_metric(
            metric_row(
                metric_id=metric_id,
                family=spec["family"],
                act_tags=spec["act_tags"],
                benchmark=spec["benchmark"],
                model=spec["model"],
                intervention_family=spec["intervention_family"],
                metric_name=metric_name,
                comparison_type=comparison_type,
                condition_a=condition_a,
                condition_b=condition_b,
                estimate=point,
                unit=unit,
                n=n,
                paired=paired,
                ci_lower=ci_lower,
                ci_upper=ci_upper,
                ci_level=ci_level,
                ci_method=ci_method,
                source_artifact_ids=source_artifact_ids,
                source_format=spec["content_type"],
                source_locators=source_locators,
            )
        )

    def extract_pipeline_summary(self, spec: JsonDict) -> None:
        data = load_json(self.path(spec["source_path"]))
        for field, unit in [
            ("sampled_questions", "count"),
            ("consistent_total", "count"),
            ("extracted_answer_tokens", "count"),
            ("disjoint_evaluated_total", "count"),
            ("selected_h_neurons", "count"),
            ("total_ffn_neurons", "count"),
        ]:
            self.add_metric_from_point(
                metric_id=f"readout.pipeline.{field}",
                spec=spec,
                metric_name=field,
                comparison_type="summary_value",
                condition_a="pipeline",
                condition_b="pipeline",
                point=data["counts"][field],
                unit=unit,
                locator=f"counts.{field}",
                source_locators=(
                    [
                        self.source_locator(spec["artifact_id"], f"counts.{field}"),
                        self.source_locator(
                            "classifier_disjoint_summary",
                            "selected_h_neurons",
                            derivation="mirrored_structured_source",
                        ),
                    ]
                    if field == "selected_h_neurons"
                    else [
                        self.source_locator(spec["artifact_id"], f"counts.{field}"),
                        self.source_locator(
                            "classifier_structure_summary",
                            "total_ffn_neurons",
                            derivation="mirrored_structured_source",
                        ),
                    ]
                    if field == "total_ffn_neurons"
                    else None
                ),
            )

    def extract_classifier_disjoint(self, spec: JsonDict) -> None:
        data = load_json(self.path(spec["source_path"]))
        metrics = data["evaluation"]["metrics"]
        for key in ["accuracy", "precision", "recall", "f1", "auroc"]:
            lower, upper, level, method = ci_fields(metrics[key])
            self.add_metric_from_point(
                metric_id=f"readout.disjoint.{key}",
                spec=spec,
                metric_name=key,
                comparison_type="test_metric",
                condition_a="disjoint_test",
                condition_b="disjoint_test",
                point=metrics[key]["estimate"],
                unit="proportion",
                n=data["evaluation"]["n_examples"],
                ci_lower=lower,
                ci_upper=upper,
                ci_level=level,
                ci_method=method,
                locator=f"evaluation.metrics.{key}",
            )
        self.add_metric_from_point(
            metric_id="readout.disjoint.test_size",
            spec=spec,
            metric_name="test_size",
            comparison_type="summary_value",
            condition_a="disjoint_test",
            condition_b="disjoint_test",
            point=data["evaluation"]["n_examples"],
            unit="count",
            n=data["evaluation"]["n_examples"],
            locator="evaluation.n_examples",
        )
        for key, value in data["evaluation"]["confusion_matrix"].items():
            self.add_metric_from_point(
                metric_id=f"readout.disjoint.confusion.{key}",
                spec=spec,
                metric_name=f"confusion_{key}",
                comparison_type="confusion_count",
                condition_a="disjoint_test",
                condition_b="disjoint_test",
                point=value,
                unit="count",
                n=data["evaluation"]["n_examples"],
                locator=f"evaluation.confusion_matrix.{key}",
            )

    def extract_classifier_overlap(self, spec: JsonDict) -> None:
        data = load_json(self.path(spec["source_path"]))
        metrics = data["evaluation"]["metrics"]
        for key in ["accuracy", "auroc"]:
            lower, upper, level, method = ci_fields(metrics[key])
            self.add_metric_from_point(
                metric_id=f"readout.overlap.{key}",
                spec=spec,
                metric_name=key,
                comparison_type="test_metric",
                condition_a="overlap_test",
                condition_b="overlap_test",
                point=metrics[key]["estimate"],
                unit="proportion",
                n=data["evaluation"]["n_examples"],
                ci_lower=lower,
                ci_upper=upper,
                ci_level=level,
                ci_method=method,
            )

    def extract_classifier_sae(self, spec: JsonDict) -> None:
        data = load_json(self.path(spec["source_path"]))
        metrics = data["best"]["test_metrics"]
        for key in ["accuracy", "auroc"]:
            self.add_metric_from_point(
                metric_id=f"readout.sae.{key}",
                spec=spec,
                metric_name=key,
                comparison_type="test_metric",
                condition_a="sae_test",
                condition_b="sae_test",
                point=metrics[key],
                unit="proportion",
                n=metrics["n_examples"],
                locator=f"best.test_metrics.{key}",
            )
        self.add_metric_from_point(
            metric_id="readout.sae.test_size",
            spec=spec,
            metric_name="test_size",
            comparison_type="summary_value",
            condition_a="sae_test",
            condition_b="sae_test",
            point=metrics["n_examples"],
            unit="count",
            n=metrics["n_examples"],
            locator="best.test_metrics.n_examples",
        )
        for key in ["n_positive_features", "n_nonzero_features"]:
            self.add_metric_from_point(
                metric_id=f"readout.sae.{key}",
                spec=spec,
                metric_name=key,
                comparison_type="summary_value",
                condition_a="sae_selected",
                condition_b="sae_selected",
                point=data["best"][key],
                unit="count",
                locator=f"best.{key}",
            )
        self.add_metric_from_point(
            metric_id="readout.sae.layer_count",
            spec=spec,
            metric_name="layer_count",
            comparison_type="summary_value",
            condition_a="sae_selected",
            condition_b="sae_selected",
            point=len(data["extraction_metadata"]["layer_indices"]),
            unit="count",
            locator="extraction_metadata.layer_indices",
        )

    def extract_classifier_structure(self, spec: JsonDict) -> None:
        data = load_json(self.path(spec["source_path"]))
        for band_name in ["early", "middle", "late"]:
            band = data["structure"]["bands"][band_name]
            self.add_metric_from_point(
                metric_id=f"readout.structure.band.{band_name}",
                spec=spec,
                metric_name=f"{band_name}_count",
                comparison_type="band_count",
                condition_a=f"layers_{band['start_layer']}_{band['end_layer']}",
                condition_b=f"layers_{band['start_layer']}_{band['end_layer']}",
                point=band["count"],
                unit="count",
            )

    def extract_intervention_sweep(self, spec: JsonDict) -> None:
        data = load_json(self.path(spec["source_path"]))
        anti = data["series"]["anti_compliance"]["effects"]
        for field in [
            "delta_0_to_max_pp",
            "delta_noop_to_max_pp",
            "slope_pp_per_alpha",
        ]:
            value = anti[field]
            self.add_metric_from_point(
                metric_id=f"intervention.faitheval.anti.{field}",
                spec=spec,
                metric_name=field,
                comparison_type="paired_effect",
                condition_a="alpha_0.0"
                if field == "delta_0_to_max_pp"
                else "noop_alpha",
                condition_b="alpha_3.0",
                point=value["estimate"],
                unit="pp" if field != "slope_pp_per_alpha" else "pp_per_alpha",
                n=data["series"]["anti_compliance"]["points"][0]["n_total"],
                paired=True,
                ci_lower=value["ci"]["lower"],
                ci_upper=value["ci"]["upper"],
                ci_level=value["ci"]["level"],
                ci_method=value["ci"]["method"],
                locator=f"series.anti_compliance.effects.{field}.estimate",
            )
        self.add_metric_from_point(
            metric_id="intervention.faitheval.sample_size",
            spec=spec,
            metric_name="sample_size",
            comparison_type="summary_value",
            condition_a="alpha_0.0",
            condition_b="alpha_0.0",
            point=data["series"]["anti_compliance"]["points"][0]["n_total"],
            unit="count",
            n=data["series"]["anti_compliance"]["points"][0]["n_total"],
            locator="series.anti_compliance.points[0].n_total",
        )
        standard = data["series"]["standard_raw"]["effects"]
        self.add_metric_from_point(
            metric_id="measurement.faitheval.standard_raw.slope_pp_per_alpha",
            spec={**spec, "family": "measurement_surfaces"},
            metric_name="standard_raw_slope_pp_per_alpha",
            comparison_type="paired_effect",
            condition_a="alpha_0.0",
            condition_b="alpha_3.0",
            point=standard["slope_pp_per_alpha"]["estimate"],
            unit="pp_per_alpha",
            n=data["series"]["standard_raw"]["points"][0]["n_total"],
            paired=True,
            ci_lower=standard["slope_pp_per_alpha"]["ci"]["lower"],
            ci_upper=standard["slope_pp_per_alpha"]["ci"]["upper"],
            ci_level=standard["slope_pp_per_alpha"]["ci"]["level"],
            ci_method=standard["slope_pp_per_alpha"]["ci"]["method"],
            locator="series.standard_raw.effects.slope_pp_per_alpha.estimate",
        )

    def extract_faitheval_control_comparison(self, spec: JsonDict) -> None:
        data = load_json(self.path(spec["source_path"]))
        self.add_metric_from_point(
            metric_id="intervention.faitheval.random_mean_slope_pp_per_alpha",
            spec=spec,
            metric_name="random_mean_slope_pp_per_alpha",
            comparison_type="control_summary",
            condition_a="unconstrained_random",
            condition_b="unconstrained_random",
            point=round(
                sum(
                    seed["slope_per_alpha"]
                    for seed in data["unconstrained_random"]["per_seed"].values()
                )
                / len(data["unconstrained_random"]["per_seed"]),
                4,
            ),
            unit="pp_per_alpha",
            n=len(data["unconstrained_random"]["per_seed"]),
            locator="unconstrained_random.per_seed.*.slope_per_alpha",
        )
        self.add_metric_from_point(
            metric_id="intervention.faitheval.random_max_slope_pp_per_alpha",
            spec=spec,
            metric_name="random_max_slope_pp_per_alpha",
            comparison_type="control_summary",
            condition_a="all_random",
            condition_b="all_random",
            point=max(
                seed["slope_per_alpha"]
                for family_name in ["unconstrained_random", "layer_matched_random"]
                for seed in data[family_name]["per_seed"].values()
            ),
            unit="pp_per_alpha",
            n=8,
            locator="all_random.per_seed.*.slope_per_alpha",
        )
        self.add_metric_from_point(
            metric_id="intervention.faitheval.h_spearman",
            spec=spec,
            metric_name="h_spearman_rho",
            comparison_type="rank_correlation",
            condition_a="alpha",
            condition_b="compliance_rate",
            point=data["h_neuron_baseline"]["spearman_rho"],
            unit="rho",
            n=7,
            locator="h_neuron_baseline.spearman_rho",
        )

    def extract_faitheval_slope_difference(self, spec: JsonDict) -> None:
        data = load_json(self.path(spec["source_path"]))
        positive_count = sum(
            1
            for seed in data["per_seed"].values()
            if seed["slope_difference_pp_per_alpha"]["estimate"] > 0
        )
        first_row = next(iter(data["per_seed"].values()))
        estimate = sum(
            row["slope_difference_pp_per_alpha"]["estimate"]
            for row in data["per_seed"].values()
        ) / len(data["per_seed"])
        self.add_metric_from_point(
            metric_id="intervention.faitheval.seedwise_positive_differences",
            spec=spec,
            metric_name="seedwise_positive_differences",
            comparison_type="seed_count",
            condition_a="h_minus_random",
            condition_b="h_minus_random",
            point=positive_count,
            unit="count",
            n=len(data["per_seed"]),
            locator="per_seed.*.slope_difference_pp_per_alpha.estimate",
        )
        self.add_metric_from_point(
            metric_id="intervention.faitheval.mean_slope_difference_pp_per_alpha",
            spec=spec,
            metric_name="mean_slope_difference_pp_per_alpha",
            comparison_type="paired_effect",
            condition_a="h_neuron",
            condition_b="random_controls",
            point=estimate,
            unit="pp_per_alpha",
            n=first_row["n_items"],
            paired=True,
            ci_lower=data["per_seed"]["seed_0_unconstrained"][
                "slope_difference_pp_per_alpha"
            ]["ci"]["lower"],
            ci_upper=data["per_seed"]["seed_0_unconstrained"][
                "slope_difference_pp_per_alpha"
            ]["ci"]["upper"],
            ci_level=0.95,
            ci_method="bootstrap_percentile_paired_seedwise_reference",
            locator="aggregate.slope_difference_pp_per_alpha.estimate",
        )

    def extract_faitheval_remap(self, spec: JsonDict) -> None:
        data = load_json(self.path(spec["source_path"]))
        for field in [
            ("parse_failures", "count"),
            ("strict_recovered_count", "count"),
            ("strict_rescored_compliance_rate", "proportion"),
        ]:
            self.add_metric_from_point(
                metric_id=f"measurement.faitheval.standard_remap.{field[0]}",
                spec=spec,
                metric_name=field[0],
                comparison_type="scoring_remap",
                condition_a="alpha_3.0_raw",
                condition_b="alpha_3.0_strict_rescore",
                point=data[field[0]],
                unit=field[1],
                n=data["total_rows"],
            )

    def extract_faitheval_sae_comparison(self, spec: JsonDict) -> None:
        data = load_json(self.path(spec["source_path"]))
        self.add_metric_from_point(
            metric_id="intervention.faitheval_sae.h_slope_pp_per_alpha",
            spec=spec,
            metric_name="h_sae_slope_pp_per_alpha",
            comparison_type="paired_effect",
            condition_a="alpha_0.0",
            condition_b="alpha_3.0",
            point=data["h_sae_features"]["slope_per_alpha"],
            unit="pp_per_alpha",
            n=data["h_sae_features"]["compliance_ci_by_alpha"][0]["n_total"],
            locator="h_sae_features.slope_per_alpha",
        )
        self.add_metric_from_point(
            metric_id="intervention.faitheval_sae.h_spearman",
            spec=spec,
            metric_name="h_sae_spearman_rho",
            comparison_type="rank_correlation",
            condition_a="alpha",
            condition_b="compliance_rate",
            point=data["h_sae_features"]["spearman_rho"],
            unit="rho",
            n=7,
            locator="h_sae_features.spearman_rho",
        )
        mean_random_slope = sum(
            row["slope_per_alpha"]
            for row in data["random_sae_features"]["per_seed"].values()
        ) / len(data["random_sae_features"]["per_seed"])
        self.add_metric_from_point(
            metric_id="intervention.faitheval_sae.random_mean_slope_pp_per_alpha",
            spec=spec,
            metric_name="random_sae_mean_slope_pp_per_alpha",
            comparison_type="control_summary",
            condition_a="random_sae",
            condition_b="random_sae",
            point=mean_random_slope,
            unit="pp_per_alpha",
            n=len(data["random_sae_features"]["per_seed"]),
            locator="random_sae_features.per_seed.*.slope_per_alpha",
        )

    def extract_faitheval_sae_difference(self, spec: JsonDict) -> None:
        data = load_json(self.path(spec["source_path"]))
        diff = data["slope_difference_pp_per_alpha"]
        self.add_metric_from_point(
            metric_id="intervention.faitheval_sae.neuron_minus_sae_slope_pp_per_alpha",
            spec=spec,
            metric_name="neuron_minus_sae_slope_pp_per_alpha",
            comparison_type="paired_effect",
            condition_a="h_neuron",
            condition_b="h_sae_features",
            point=diff["estimate"],
            unit="pp_per_alpha",
            n=data["n_items"],
            paired=True,
            ci_lower=diff["ci"]["lower"],
            ci_upper=diff["ci"]["upper"],
            ci_level=diff["ci"]["level"],
            ci_method=diff["ci"]["method"],
            locator="slope_difference_pp_per_alpha.estimate",
        )
        self.add_metric_from_point(
            metric_id="intervention.faitheval_sae.permutation_p",
            spec=spec,
            metric_name="one_sided_permutation_p",
            comparison_type="permutation_test",
            condition_a="h_neuron",
            condition_b="h_sae_features",
            point=data["permutation_test"]["p_value"],
            unit="p_value",
            n=data["permutation_test"]["n_permutations"],
            locator="permutation_test.p_value",
        )

    def extract_falseqa_results(self, spec: JsonDict) -> None:
        data = load_json(self.path(spec["source_path"]))
        effects = data["effects"]["compliance_curve"]
        for field, unit in [
            ("slope_pp_per_alpha", "pp_per_alpha"),
            ("delta_0_to_max_pp", "pp"),
            ("delta_noop_to_max_pp", "pp"),
        ]:
            value = effects[field]
            self.add_metric_from_point(
                metric_id=f"intervention.falseqa.{field}",
                spec=spec,
                metric_name=field,
                comparison_type="paired_effect",
                condition_a="alpha_0.0",
                condition_b="alpha_3.0",
                point=value["estimate"],
                unit=unit,
                n=data["results"]["0.0"]["n_total"],
                paired=True,
                ci_lower=value["ci"]["lower"],
                ci_upper=value["ci"]["upper"],
                ci_level=value["ci"]["level"],
                ci_method=value["ci"]["method"],
                locator=f"effects.compliance_curve.{field}.estimate",
            )
        self.add_metric_from_point(
            metric_id="intervention.falseqa.sample_size",
            spec=spec,
            metric_name="sample_size",
            comparison_type="summary_value",
            condition_a="alpha_0.0",
            condition_b="alpha_0.0",
            point=data["results"]["0.0"]["n_total"],
            unit="count",
            n=data["results"]["0.0"]["n_total"],
            locator="results.0.0.n_total",
        )

    def extract_bioasq_results(self, spec: JsonDict) -> None:
        data = load_json(self.path(spec["source_path"]))
        effects = data["effects"]["compliance_curve"]
        for field, unit in [
            ("delta_0_to_max_pp", "pp"),
            ("slope_pp_per_alpha", "pp_per_alpha"),
        ]:
            value = effects[field]
            self.add_metric_from_point(
                metric_id=f"intervention.bioasq.{field}",
                spec=spec,
                metric_name=field,
                comparison_type="paired_effect",
                condition_a="alpha_0.0",
                condition_b="alpha_3.0",
                point=value["estimate"],
                unit=unit,
                n=data["results"]["0.0"]["n_total"],
                paired=True,
                ci_lower=value["ci"]["lower"],
                ci_upper=value["ci"]["upper"],
                ci_level=value["ci"]["level"],
                ci_method=value["ci"]["method"],
                locator=f"effects.compliance_curve.{field}.estimate",
            )
        self.add_metric_from_point(
            metric_id="intervention.bioasq.sample_size",
            spec=spec,
            metric_name="sample_size",
            comparison_type="summary_value",
            condition_a="alpha_0.0",
            condition_b="alpha_0.0",
            point=data["results"]["0.0"]["n_total"],
            unit="count",
            n=data["results"]["0.0"]["n_total"],
            locator="results.0.0.n_total",
        )

    def extract_bioasq_control_comparison(self, spec: JsonDict) -> None:
        data = load_json(self.path(spec["source_path"]))
        self.add_metric_from_point(
            metric_id="intervention.bioasq.random_mean_slope_pp_per_alpha",
            spec=spec,
            metric_name="random_mean_slope_pp_per_alpha",
            comparison_type="control_summary",
            condition_a="random_controls",
            condition_b="random_controls",
            point=data["unconstrained_random"]["mean_slope_per_alpha"],
            unit="pp_per_alpha",
            n=len(data["unconstrained_random"]["per_seed"]),
        )

    def extract_bioasq_audit_gapfill(self, spec: JsonDict) -> None:
        alpha0 = {
            row["id"]: row
            for row in load_jsonl(
                self.path(
                    "data/gemma3_4b/intervention/bioasq/experiment/alpha_0.0.jsonl"
                )
            )
        }
        alpha3 = {
            row["id"]: row
            for row in load_jsonl(
                self.path(
                    "data/gemma3_4b/intervention/bioasq/experiment/alpha_3.0.jsonl"
                )
            )
        }
        common_ids = sorted(set(alpha0) & set(alpha3))
        changed_count = sum(
            1
            for row_id in common_ids
            if response_text(alpha0[row_id]) != response_text(alpha3[row_id])
        )
        mean_alpha0 = sum(
            response_length(alpha0[row_id]) for row_id in common_ids
        ) / len(common_ids)
        mean_alpha3 = sum(
            response_length(alpha3[row_id]) for row_id in common_ids
        ) / len(common_ids)
        shared_locators = [
            self.source_locator(
                "bioasq_alpha_0_rows",
                "rows[*].response",
                derivation="paired_join_by_id",
            ),
            self.source_locator(
                "bioasq_alpha_3_rows",
                "rows[*].response",
                derivation="paired_join_by_id",
            ),
        ]
        self.add_metric_from_point(
            metric_id="intervention.bioasq.changed_response_count",
            spec=spec,
            metric_name="changed_response_count",
            comparison_type="text_change_count",
            condition_a="alpha_0.0",
            condition_b="alpha_3.0",
            point=changed_count,
            unit="count",
            n=len(common_ids),
            source_artifact_ids=["bioasq_alpha_0_rows", "bioasq_alpha_3_rows"],
            source_locators=shared_locators,
        )
        self.add_metric_from_point(
            metric_id="intervention.bioasq.response_length_mean_alpha0",
            spec=spec,
            metric_name="response_length_mean_alpha0",
            comparison_type="summary_value",
            condition_a="alpha_0.0",
            condition_b="alpha_0.0",
            point=mean_alpha0,
            unit="characters",
            n=len(common_ids),
            source_artifact_ids=["bioasq_alpha_0_rows"],
            source_locators=[
                self.source_locator(
                    "bioasq_alpha_0_rows",
                    "rows[*].response",
                    derivation="mean_response_length",
                )
            ],
        )
        self.add_metric_from_point(
            metric_id="intervention.bioasq.response_length_mean_alpha3",
            spec=spec,
            metric_name="response_length_mean_alpha3",
            comparison_type="summary_value",
            condition_a="alpha_3.0",
            condition_b="alpha_3.0",
            point=mean_alpha3,
            unit="characters",
            n=len(common_ids),
            source_artifact_ids=["bioasq_alpha_3_rows"],
            source_locators=[
                self.source_locator(
                    "bioasq_alpha_3_rows",
                    "rows[*].response",
                    derivation="mean_response_length",
                )
            ],
        )

    def extract_jailbreak_results(self, spec: JsonDict) -> None:
        data = load_json(self.path(spec["source_path"]))
        effects = data["effects"]["compliance_curve"]
        for field, unit in [
            ("delta_0_to_max_pp", "pp"),
            ("slope_pp_per_alpha", "pp_per_alpha"),
        ]:
            value = effects[field]
            self.add_metric_from_point(
                metric_id=f"intervention.jailbreak.binary.{field}",
                spec=spec,
                metric_name=field,
                comparison_type="paired_effect",
                condition_a="alpha_0.0",
                condition_b="alpha_3.0",
                point=value["estimate"],
                unit=unit,
                n=data["results"]["0.0"]["n_total"],
                paired=True,
                ci_lower=value["ci"]["lower"],
                ci_upper=value["ci"]["upper"],
                ci_level=value["ci"]["level"],
                ci_method=value["ci"]["method"],
                locator=f"effects.compliance_curve.{field}.estimate",
            )
        for alpha_key in ["0.0", "3.0"]:
            result = data["results"][alpha_key]["compliance"]
            self.add_metric_from_point(
                metric_id=f"measurement.jailbreak.binary.alpha_{alpha_key[0]}_count",
                spec={**spec, "family": "measurement_surfaces"},
                metric_name=f"binary_harmful_count_alpha_{alpha_key}",
                comparison_type="summary_value",
                condition_a=alpha_label(alpha_key),
                condition_b=alpha_label(alpha_key),
                point=result["n_compliant"],
                unit="count",
                n=result["n_total"],
                locator=f"results.{alpha_key}.compliance.n_compliant",
            )

    def extract_jailbreak_control_v2(self, spec: JsonDict) -> None:
        data = load_json(self.path(spec["source_path"]))
        self.add_metric_from_point(
            metric_id="measurement.jailbreak.v2.h_slope_csv2_yes_pp_per_alpha",
            spec=spec,
            metric_name="h_slope_csv2_yes_pp_per_alpha",
            comparison_type="graded_effect",
            condition_a="alpha_0.0",
            condition_b="alpha_3.0",
            point=data["h_neuron"]["slope_csv2_yes_pp_per_alpha"],
            unit="pp_per_alpha",
            n=data["h_neuron"]["total_record_counts"][0],
            locator="h_neuron.slope_csv2_yes_pp_per_alpha",
        )
        self.add_metric_from_point(
            metric_id="measurement.jailbreak.v2.random_mean_slope_csv2_yes_pp_per_alpha",
            spec=spec,
            metric_name="random_mean_slope_csv2_yes_pp_per_alpha",
            comparison_type="control_summary",
            condition_a="random_controls",
            condition_b="random_controls",
            point=data["random_aggregate"]["csv2_yes_slope_mean"],
            unit="pp_per_alpha",
            n=1,
            locator="random_aggregate.csv2_yes_slope_mean",
        )
        self.add_metric_from_point(
            metric_id="measurement.jailbreak.v2.gap_h_minus_random_mean_pp_per_alpha",
            spec=spec,
            metric_name="gap_h_minus_random_mean_pp_per_alpha",
            comparison_type="paired_difference",
            condition_a="h_neuron",
            condition_b="random_control",
            point=data["comparison"]["gap_h_minus_random_mean_pp"],
            unit="pp_per_alpha",
            n=data["h_neuron"]["total_record_counts"][0],
            locator="comparison.gap_h_minus_random_mean_pp",
        )
        self.add_metric_from_point(
            metric_id="measurement.jailbreak.v2.endpoint_delta_csv2_yes_pp",
            spec=spec,
            metric_name="endpoint_delta_csv2_yes_pp",
            comparison_type="paired_effect",
            condition_a="alpha_0.0",
            condition_b="alpha_3.0",
            point=(
                (
                    data["h_neuron"]["csv2_yes_rates"][-1]
                    - data["h_neuron"]["csv2_yes_rates"][0]
                )
                * 100.0
            ),
            unit="pp",
            n=data["h_neuron"]["total_record_counts"][0],
            locator="h_neuron.csv2_yes_rates",
        )

    def extract_jailbreak_control_v3(self, spec: JsonDict) -> None:
        data = load_json(self.path(spec["source_path"]))
        self.add_metric_from_point(
            metric_id="measurement.jailbreak.v3.h_slope_csv2_yes_pp_per_alpha",
            spec=spec,
            metric_name="h_slope_csv2_yes_pp_per_alpha",
            comparison_type="graded_effect",
            condition_a="alpha_0.0",
            condition_b="alpha_3.0",
            point=data["h_neuron"]["slope_csv2_yes_pp_per_alpha"],
            unit="pp_per_alpha",
            n=data["h_neuron"]["total_record_counts"][0],
            locator="h_neuron.slope_csv2_yes_pp_per_alpha",
        )
        self.add_metric_from_point(
            metric_id="measurement.jailbreak.v3.random_mean_slope_csv2_yes_pp_per_alpha",
            spec=spec,
            metric_name="random_mean_slope_csv2_yes_pp_per_alpha",
            comparison_type="control_summary",
            condition_a="random_controls",
            condition_b="random_controls",
            point=data["random_aggregate"]["csv2_yes_slope_mean"],
            unit="pp_per_alpha",
            n=1,
            locator="random_aggregate.csv2_yes_slope_mean",
        )
        self.add_metric_from_point(
            metric_id="measurement.jailbreak.v3.gap_h_minus_random_mean_pp_per_alpha",
            spec=spec,
            metric_name="gap_h_minus_random_mean_pp_per_alpha",
            comparison_type="paired_difference",
            condition_a="h_neuron",
            condition_b="random_control",
            point=data["comparison"]["gap_h_minus_random_mean_pp"],
            unit="pp_per_alpha",
            n=data["h_neuron"]["total_record_counts"][0],
            locator="comparison.gap_h_minus_random_mean_pp",
        )

    def extract_holdout_comparison(self, spec: JsonDict) -> None:
        data = load_json(self.path(spec["source_path"]))
        evaluators = data["bootstrap_cis"]["evaluators"]
        for evaluator in ["binary", "csv2v2", "csv2v3", "sr"]:
            accuracy = evaluators[evaluator]["accuracy"]
            self.add_metric_from_point(
                metric_id=f"measurement.holdout.{evaluator}.accuracy",
                spec=spec,
                metric_name=f"{evaluator}_accuracy",
                comparison_type="holdout_accuracy",
                condition_a="holdout",
                condition_b="holdout",
                point=accuracy["point"],
                unit="proportion",
                n=50,
                ci_lower=accuracy["ci_lower"],
                ci_upper=accuracy["ci_upper"],
                ci_level=0.95,
                ci_method=data["bootstrap_cis"]["method"],
                locator=f"bootstrap_cis.evaluators.{evaluator}.accuracy.point",
            )
        self.add_metric_from_point(
            metric_id="measurement.holdout.sample_size",
            spec=spec,
            metric_name="sample_size",
            comparison_type="summary_value",
            condition_a="holdout",
            condition_b="holdout",
            point=50,
            unit="count",
            n=50,
            locator="cross_alpha_holdout",
        )
        self.add_metric_from_point(
            metric_id="measurement.holdout.csv2v3_vs_sr_mcnemar_p",
            spec=spec,
            metric_name="csv2v3_vs_sr_mcnemar_p",
            comparison_type="paired_test",
            condition_a="csv2v3",
            condition_b="sr",
            point=data["mcnemar_tests"]["csv2v3_vs_sr"]["p_value"],
            unit="p_value",
            n=50,
            locator="mcnemar_tests.csv2v3_vs_sr.p_value",
        )
        self.add_metric_from_point(
            metric_id="measurement.holdout.csv2v3_vs_sr_discordant_count",
            spec=spec,
            metric_name="csv2v3_vs_sr_discordant_count",
            comparison_type="paired_test",
            condition_a="csv2v3",
            condition_b="sr",
            point=data["mcnemar_tests"]["csv2v3_vs_sr"]["n_discordant"],
            unit="count",
            n=50,
            locator="mcnemar_tests.csv2v3_vs_sr.n_discordant",
        )
        self.add_metric_from_point(
            metric_id="measurement.holdout.min_pairwise_mcnemar_p",
            spec=spec,
            metric_name="min_pairwise_mcnemar_p",
            comparison_type="paired_test",
            condition_a="all_pairs",
            condition_b="all_pairs",
            point=min(row["p_value"] for row in data["mcnemar_tests"].values()),
            unit="p_value",
            n=50,
            locator="mcnemar_tests",
        )

    def extract_csv2_v3_results(self, spec: JsonDict) -> None:
        data = load_json(self.path(spec["source_path"]))
        accuracy = data["accuracy"]
        self.add_metric_from_point(
            metric_id="measurement.csv2_v3.dev_accuracy",
            spec=spec,
            metric_name="dev_accuracy",
            comparison_type="gold_accuracy",
            condition_a="gold74",
            condition_b="gold74",
            point=accuracy["estimate"],
            unit="proportion",
            n=accuracy["n"],
            ci_lower=accuracy["ci"]["lower"],
            ci_upper=accuracy["ci"]["upper"],
            ci_level=accuracy["ci"]["level"],
            ci_method=accuracy["ci"]["method"],
        )

    def extract_strongreject_results(self, spec: JsonDict) -> None:
        data = load_json(self.path(spec["source_path"]))
        accuracy = data["accuracy"]
        self.add_metric_from_point(
            metric_id="measurement.strongreject.dev_accuracy",
            spec=spec,
            metric_name="dev_accuracy",
            comparison_type="gold_accuracy",
            condition_a="gold74",
            condition_b="gold74",
            point=accuracy["estimate"],
            unit="proportion",
            n=accuracy["n"],
            ci_lower=accuracy["ci"]["lower"],
            ci_upper=accuracy["ci"]["upper"],
            ci_level=accuracy["ci"]["level"],
            ci_method=accuracy["ci"]["method"],
        )

    def extract_available_jailbreak_evaluator_comparison(self, spec: JsonDict) -> None:
        data = load_json(self.path(spec["source_path"]))
        baseline = data["d7"]["baseline_noop"]
        causal = data["d7"]["causal_locked"]
        self.add_metric_from_point(
            metric_id="measurement.jailbreak.baseline_noop.v2_rate",
            spec=spec,
            metric_name="baseline_noop_v2_rate",
            comparison_type="ruler_rate",
            condition_a="baseline_noop_v2",
            condition_b="baseline_noop_v2",
            point=baseline["v2"]["rate"],
            unit="proportion",
            n=baseline["v2"]["total"],
        )
        self.add_metric_from_point(
            metric_id="measurement.jailbreak.baseline_noop.v3_rate",
            spec=spec,
            metric_name="baseline_noop_v3_rate",
            comparison_type="ruler_rate",
            condition_a="baseline_noop_v3",
            condition_b="baseline_noop_v3",
            point=baseline["v3"]["rate"],
            unit="proportion",
            n=baseline["v3"]["total"],
        )
        self.add_metric_from_point(
            metric_id="measurement.jailbreak.causal_locked.v2_rate",
            spec=spec,
            metric_name="causal_locked_v2_rate",
            comparison_type="ruler_rate",
            condition_a="causal_locked_v2",
            condition_b="causal_locked_v2",
            point=causal["v2"]["rate"],
            unit="proportion",
            n=causal["v2"]["total"],
        )
        self.add_metric_from_point(
            metric_id="measurement.jailbreak.causal_locked.v3_rate",
            spec=spec,
            metric_name="causal_locked_v3_rate",
            comparison_type="ruler_rate",
            condition_a="causal_locked_v3",
            condition_b="causal_locked_v3",
            point=causal["v3"]["rate"],
            unit="proportion",
            n=causal["v3"]["total"],
        )

    def extract_bridge_irr_summary(self, spec: JsonDict) -> None:
        data = load_json(self.path(spec["source_path"]))
        irr = data["irr"]
        raw = irr["raw_agreement"]
        self.add_metric_from_point(
            metric_id="measurement.bridge_irr.raw_agreement",
            spec=spec,
            metric_name="raw_agreement",
            comparison_type="irr",
            condition_a="rater_a",
            condition_b="rater_b",
            point=raw["estimate"],
            unit="proportion",
            n=raw["n"],
            ci_lower=raw["ci"]["lower"],
            ci_upper=raw["ci"]["upper"],
            ci_level=raw["ci"]["level"],
            ci_method=raw["ci"]["method"],
            locator="irr.raw_agreement.estimate",
        )
        self.add_metric_from_point(
            metric_id="measurement.bridge_irr.cohen_kappa",
            spec=spec,
            metric_name="cohen_kappa",
            comparison_type="irr",
            condition_a="rater_a",
            condition_b="rater_b",
            point=irr["cohen_kappa"],
            unit="kappa",
            n=irr["n_cases"],
            locator="irr.cohen_kappa",
        )
        self.add_metric_from_point(
            metric_id="measurement.bridge_irr.gwet_ac1",
            spec=spec,
            metric_name="gwet_ac1",
            comparison_type="irr",
            condition_a="rater_a",
            condition_b="rater_b",
            point=irr["gwet_ac1"],
            unit="ac1",
            n=irr["n_cases"],
            locator="irr.gwet_ac1",
        )
        self.add_metric_from_point(
            metric_id="measurement.bridge_irr.n_disagreements",
            spec=spec,
            metric_name="n_disagreements",
            comparison_type="irr",
            condition_a="rater_a",
            condition_b="rater_b",
            point=irr["n_disagreements"],
            unit="count",
            n=irr["n_cases"],
            locator="irr.n_disagreements",
        )
        right_to_wrong = data["direction_summaries"]["right_to_wrong"]["categories"]
        for category in [
            "wrong_entity_substitution",
            "evasion_or_factual_denial",
            "answer_dilution",
            "formal_refusal",
        ]:
            value = right_to_wrong[category]
            self.add_metric_from_point(
                metric_id=f"measurement.bridge_irr.right_to_wrong.{category}",
                spec=spec,
                metric_name=category,
                comparison_type="category_share",
                condition_a="right_to_wrong",
                condition_b=category,
                point=value["share"],
                unit="proportion",
                n=value["denominator"],
                ci_lower=value["share_ci"]["lower"],
                ci_upper=value["share_ci"]["upper"],
                ci_level=value["share_ci"]["level"],
                ci_method=value["share_ci"]["method"],
                locator=f"direction_summaries.right_to_wrong.categories.{category}.share",
            )

    def extract_bridge_phase3(self, spec: JsonDict) -> None:
        data = load_json(self.path(spec["source_path"]))
        baseline = data["conditions"]["baseline"]
        effects = data["effects"]
        self.add_metric_from_point(
            metric_id="transfer.bridge.baseline_adjudicated_accuracy",
            spec=spec,
            metric_name="baseline_adjudicated_accuracy",
            comparison_type="paired_surface",
            condition_a="baseline_alpha_1.0",
            condition_b="baseline_alpha_1.0",
            point=baseline["compliance"]["estimate"],
            unit="proportion",
            n=baseline["compliance"]["n_total"],
            ci_lower=baseline["compliance"]["ci"]["lower"],
            ci_upper=baseline["compliance"]["ci"]["upper"],
            ci_level=baseline["compliance"]["ci"]["level"],
            ci_method=baseline["compliance"]["ci"]["method"],
            locator="conditions.baseline.compliance.estimate",
        )
        self.add_metric_from_point(
            metric_id="transfer.bridge.sample_size",
            spec=spec,
            metric_name="sample_size",
            comparison_type="summary_value",
            condition_a="baseline_alpha_1.0",
            condition_b="baseline_alpha_1.0",
            point=baseline["compliance"]["n_total"],
            unit="count",
            n=baseline["compliance"]["n_total"],
            locator="conditions.baseline.compliance.n_total",
        )
        for key in [
            "adjudicated_accuracy_delta_pp",
            "deterministic_accuracy_delta_pp",
            "attempt_rate_delta_pp",
        ]:
            value = effects[key]
            self.add_metric_from_point(
                metric_id=f"transfer.bridge.{key}",
                spec=spec,
                metric_name=key,
                comparison_type="paired_effect",
                condition_a="baseline_alpha_1.0",
                condition_b="iti_e0_alpha_8.0",
                point=value["estimate"],
                unit="pp",
                n=500,
                paired=True,
                ci_lower=value["ci"]["lower"],
                ci_upper=value["ci"]["upper"],
                ci_level=value["ci"]["level"],
                ci_method=value["ci"]["method"],
                locator=f"effects.{key}.estimate",
            )
        self.add_metric_from_point(
            metric_id="transfer.bridge.mcnemar_p",
            spec=spec,
            metric_name="mcnemar_p",
            comparison_type="paired_test",
            condition_a="baseline_alpha_1.0",
            condition_b="iti_e0_alpha_8.0",
            point=effects["mcnemar_p"],
            unit="p_value",
            n=500,
            locator="effects.mcnemar_p",
        )
        flip_table = data["flip_table"]
        for key in ["base_correct_iti_wrong", "base_wrong_iti_correct"]:
            self.add_metric_from_point(
                metric_id=f"transfer.bridge.{key}",
                spec=spec,
                metric_name=key,
                comparison_type="flip_count",
                condition_a="baseline_alpha_1.0",
                condition_b="iti_e0_alpha_8.0",
                point=flip_table[key],
                unit="count",
                n=500,
                locator=f"flip_table.{key}",
            )
        self.add_metric_from_point(
            metric_id="transfer.bridge.baseline_not_attempted_count",
            spec=spec,
            metric_name="baseline_not_attempted_count",
            comparison_type="summary_value",
            condition_a="baseline_alpha_1.0",
            condition_b="baseline_alpha_1.0",
            point=baseline["n_not_attempted"],
            unit="count",
            n=500,
            locator="conditions.baseline.n_not_attempted",
        )
        self.add_metric_from_point(
            metric_id="transfer.bridge.iti_not_attempted_count",
            spec=spec,
            metric_name="iti_not_attempted_count",
            comparison_type="summary_value",
            condition_a="iti_e0_alpha_8.0",
            condition_b="iti_e0_alpha_8.0",
            point=data["conditions"]["iti_e0_alpha_8"]["n_not_attempted"],
            unit="count",
            n=500,
            locator="conditions.iti_e0_alpha_8.n_not_attempted",
        )

    def extract_truthfulqa_mc_pool(self, spec: JsonDict) -> None:
        if any(
            row["metric_id"].startswith("transfer.truthfulqa_mc.")
            for row in self.metrics
        ):
            return
        variants = {
            "mc1": {
                "h_fold0_alpha0": "truthfulqa_mc1_h_fold0_alpha0",
                "h_fold0_best": "truthfulqa_mc1_h_fold0_alpha1",
                "h_fold1_alpha0": "truthfulqa_mc1_h_fold1_alpha0",
                "h_fold1_best": "truthfulqa_mc1_h_fold1_alpha1",
                "iti_fold0_alpha0": "truthfulqa_mc1_iti_fold0_alpha0",
                "iti_fold0_best": "truthfulqa_mc1_iti_fold0_alpha8",
                "iti_fold1_alpha0": "truthfulqa_mc1_iti_fold1_alpha0",
                "iti_fold1_best": "truthfulqa_mc1_iti_fold1_alpha8",
            },
            "mc2": {
                "h_fold0_alpha0": "truthfulqa_mc2_h_fold0_alpha0",
                "h_fold0_best": "truthfulqa_mc2_h_fold0_alpha1",
                "h_fold1_alpha0": "truthfulqa_mc2_h_fold1_alpha0",
                "h_fold1_best": "truthfulqa_mc2_h_fold1_alpha1",
                "iti_fold0_alpha0": "truthfulqa_mc2_iti_fold0_alpha0",
                "iti_fold0_best": "truthfulqa_mc2_iti_fold0_alpha8",
                "iti_fold1_alpha0": "truthfulqa_mc2_iti_fold1_alpha0",
                "iti_fold1_best": "truthfulqa_mc2_iti_fold1_alpha8",
            },
        }
        for variant, ids in variants.items():
            for label, baseline_ids, best_ids, baseline_alpha, best_alpha in [
                (
                    "h_neuron",
                    [ids["h_fold0_alpha0"], ids["h_fold1_alpha0"]],
                    [ids["h_fold0_best"], ids["h_fold1_best"]],
                    "alpha_0.0",
                    "alpha_1.0",
                ),
                (
                    "iti",
                    [ids["iti_fold0_alpha0"], ids["iti_fold1_alpha0"]],
                    [ids["iti_fold0_best"], ids["iti_fold1_best"]],
                    "alpha_0.0",
                    "alpha_8.0",
                ),
            ]:
                baseline_rows = [
                    row
                    for artifact_id in baseline_ids
                    for row in load_jsonl(
                        self.path(self.artifact_by_id[artifact_id]["source_path"])
                    )
                ]
                best_rows = [
                    row
                    for artifact_id in best_ids
                    for row in load_jsonl(
                        self.path(self.artifact_by_id[artifact_id]["source_path"])
                    )
                ]
                baseline = {
                    row["id"]: float(row["metric_value"]) for row in baseline_rows
                }
                best = {row["id"]: float(row["metric_value"]) for row in best_rows}
                common_ids = sorted(set(baseline) & set(best))
                baseline_values = [baseline[row_id] for row_id in common_ids]
                best_values = [best[row_id] for row_id in common_ids]
                effect = bootstrap_paired_mean(
                    baseline_values, best_values, scale=100.0
                )
                spec_local = {**spec, "intervention_family": label}
                source_ids = baseline_ids + best_ids
                self.add_metric_from_point(
                    metric_id=f"transfer.truthfulqa_mc.{variant}.{label}.baseline_accuracy",
                    spec=spec_local,
                    metric_name=f"{variant}_{label}_baseline_accuracy",
                    comparison_type="pooled_rate",
                    condition_a=baseline_alpha,
                    condition_b=baseline_alpha,
                    point=sum(baseline_values) / len(baseline_values),
                    unit="proportion",
                    n=len(common_ids),
                    source_artifact_ids=source_ids,
                    source_locators=[
                        self.source_locator(
                            artifact_id,
                            "rows[*].metric_value",
                            derivation="pooled_mean",
                        )
                        for artifact_id in baseline_ids
                    ],
                )
                self.add_metric_from_point(
                    metric_id=f"transfer.truthfulqa_mc.{variant}.{label}.best_accuracy",
                    spec=spec_local,
                    metric_name=f"{variant}_{label}_best_accuracy",
                    comparison_type="pooled_rate",
                    condition_a=best_alpha,
                    condition_b=best_alpha,
                    point=sum(best_values) / len(best_values),
                    unit="proportion",
                    n=len(common_ids),
                    source_artifact_ids=source_ids,
                    source_locators=[
                        self.source_locator(
                            artifact_id,
                            "rows[*].metric_value",
                            derivation="pooled_mean",
                        )
                        for artifact_id in best_ids
                    ],
                )
                self.add_metric_from_point(
                    metric_id=f"transfer.truthfulqa_mc.{variant}.{label}.delta_pp",
                    spec=spec_local,
                    metric_name=f"{variant}_{label}_delta_pp",
                    comparison_type="paired_effect",
                    condition_a=baseline_alpha,
                    condition_b=best_alpha,
                    point=effect["estimate"],
                    unit="pp",
                    n=effect["n"],
                    paired=True,
                    ci_lower=effect["ci_lower"],
                    ci_upper=effect["ci_upper"],
                    ci_level=effect["ci_level"],
                    ci_method=effect["ci_method"],
                    source_artifact_ids=source_ids,
                    source_locators=[
                        self.source_locator(
                            artifact_id,
                            "rows[*].metric_value",
                            derivation="paired_pooled_effect",
                        )
                        for artifact_id in source_ids
                    ],
                )
                wrong_to_right = sum(
                    1
                    for row_id in common_ids
                    if baseline[row_id] == 0.0 and best[row_id] == 1.0
                )
                right_to_wrong = sum(
                    1
                    for row_id in common_ids
                    if baseline[row_id] == 1.0 and best[row_id] == 0.0
                )
                self.add_metric_from_point(
                    metric_id=f"transfer.truthfulqa_mc.{variant}.{label}.wrong_to_right",
                    spec=spec_local,
                    metric_name=f"{variant}_{label}_wrong_to_right",
                    comparison_type="flip_count",
                    condition_a=baseline_alpha,
                    condition_b=best_alpha,
                    point=wrong_to_right,
                    unit="count",
                    n=len(common_ids),
                    source_artifact_ids=source_ids,
                    source_locators=[
                        self.source_locator(
                            artifact_id, "rows[*].metric_value", derivation="flip_count"
                        )
                        for artifact_id in source_ids
                    ],
                )
                self.add_metric_from_point(
                    metric_id=f"transfer.truthfulqa_mc.{variant}.{label}.right_to_wrong",
                    spec=spec_local,
                    metric_name=f"{variant}_{label}_right_to_wrong",
                    comparison_type="flip_count",
                    condition_a=baseline_alpha,
                    condition_b=best_alpha,
                    point=right_to_wrong,
                    unit="count",
                    n=len(common_ids),
                    source_artifact_ids=source_ids,
                    source_locators=[
                        self.source_locator(
                            artifact_id, "rows[*].metric_value", derivation="flip_count"
                        )
                        for artifact_id in source_ids
                    ],
                )

    def extract_simpleqa_results(self, spec: JsonDict) -> None:
        data = load_json(self.path(spec["source_path"]))
        alpha_dir = self.path(spec["source_path"]).parent
        baseline_rows = load_jsonl(alpha_dir / "alpha_0.0.jsonl")
        max_rows = load_jsonl(alpha_dir / "alpha_8.0.jsonl")
        baseline_by_id = {row["id"]: row for row in baseline_rows}
        max_by_id = {row["id"]: row for row in max_rows}
        common_ids = sorted(set(baseline_by_id) & set(max_by_id))
        baseline_attempt = [
            1.0 if baseline_by_id[row_id]["judge"] != "NOT_ATTEMPTED" else 0.0
            for row_id in common_ids
        ]
        max_attempt = [
            1.0 if max_by_id[row_id]["judge"] != "NOT_ATTEMPTED" else 0.0
            for row_id in common_ids
        ]
        attempt_effect = bootstrap_paired_mean(
            baseline_attempt, max_attempt, scale=100.0
        )
        for alpha_key in ["0.0", "4.0", "8.0"]:
            result = data["results"][alpha_key]["compliance"]
            self.add_metric_from_point(
                metric_id=f"transfer.simpleqa.compliance.{alpha_key}",
                spec=spec,
                metric_name=f"compliance_{alpha_key}",
                comparison_type="rate",
                condition_a=f"alpha_{alpha_key}",
                condition_b=f"alpha_{alpha_key}",
                point=result["estimate"],
                unit="proportion",
                n=result["n_total"],
                ci_lower=result["ci"]["lower"],
                ci_upper=result["ci"]["upper"],
                ci_level=result["ci"]["level"],
                ci_method=result["ci"]["method"],
                locator=f"results.{alpha_key}.compliance.estimate",
            )
        effect = data["effects"]["compliance_curve"]["delta_0_to_max_pp"]
        self.add_metric_from_point(
            metric_id="transfer.simpleqa.compliance_delta_0_to_8_pp",
            spec=spec,
            metric_name="compliance_delta_0_to_8_pp",
            comparison_type="paired_effect",
            condition_a="alpha_0.0",
            condition_b="alpha_8.0",
            point=effect["estimate"],
            unit="pp",
            n=len(common_ids),
            paired=True,
            ci_lower=effect["ci"]["lower"],
            ci_upper=effect["ci"]["upper"],
            ci_level=effect["ci"]["level"],
            ci_method=effect["ci"]["method"],
            locator="effects.compliance_curve.delta_0_to_max_pp.estimate",
        )
        self.add_metric_from_point(
            metric_id="transfer.simpleqa.attempt_rate_alpha_0",
            spec=spec,
            metric_name="attempt_rate_alpha_0",
            comparison_type="rate",
            condition_a="alpha_0.0",
            condition_b="alpha_0.0",
            point=sum(baseline_attempt) / len(baseline_attempt),
            unit="proportion",
            n=len(common_ids),
            locator="alpha_0.0_rows[*].judge",
            source_artifact_ids=["simpleqa_alpha_0_rows"],
            source_locators=[
                self.source_locator(
                    "simpleqa_alpha_0_rows",
                    "rows[*].judge",
                    derivation="attempt_rate",
                )
            ],
        )
        self.add_metric_from_point(
            metric_id="transfer.simpleqa.attempt_rate_alpha_8",
            spec=spec,
            metric_name="attempt_rate_alpha_8",
            comparison_type="rate",
            condition_a="alpha_8.0",
            condition_b="alpha_8.0",
            point=sum(max_attempt) / len(max_attempt),
            unit="proportion",
            n=len(common_ids),
            locator="alpha_8.0_rows[*].judge",
            source_artifact_ids=["simpleqa_alpha_8_rows"],
            source_locators=[
                self.source_locator(
                    "simpleqa_alpha_8_rows",
                    "rows[*].judge",
                    derivation="attempt_rate",
                )
            ],
        )
        self.add_metric_from_point(
            metric_id="transfer.simpleqa.attempt_delta_0_to_8_pp",
            spec=spec,
            metric_name="attempt_delta_0_to_8_pp",
            comparison_type="paired_effect",
            condition_a="alpha_0.0",
            condition_b="alpha_8.0",
            point=attempt_effect["estimate"],
            unit="pp",
            n=attempt_effect["n"],
            paired=True,
            ci_lower=attempt_effect["ci_lower"],
            ci_upper=attempt_effect["ci_upper"],
            ci_level=attempt_effect["ci_level"],
            ci_method=attempt_effect["ci_method"],
            source_artifact_ids=["simpleqa_alpha_0_rows", "simpleqa_alpha_8_rows"],
            source_locators=[
                self.source_locator(
                    "simpleqa_alpha_0_rows",
                    "rows[*].judge",
                    derivation="paired_attempt_delta",
                ),
                self.source_locator(
                    "simpleqa_alpha_8_rows",
                    "rows[*].judge",
                    derivation="paired_attempt_delta",
                ),
            ],
        )

    def extract_neuron_4288(self, spec: JsonDict) -> None:
        data = load_json(self.path(spec["source_path"]))
        target = data["target_neuron"]
        tests = {test["slug"]: test for test in data["tests"]}
        self.add_metric_from_point(
            metric_id="mechanism.neuron_4288.weight",
            spec=spec,
            metric_name="weight",
            comparison_type="summary_value",
            condition_a="L20_N4288",
            condition_b="L20_N4288",
            point=target["weight"],
            unit="logit_weight",
        )
        for slug, metric_name, unit_key, unit in [
            ("single_neuron_auc", "single_neuron_auc", "value", "auroc"),
            ("distribution_separation", "distribution_separation", "value", "cohen_d"),
            ("ablation_accuracy_drop", "ablation_accuracy_drop", "value_pp", "pp"),
            ("max_top10_correlation", "max_top10_correlation", "value", "correlation"),
        ]:
            self.add_metric_from_point(
                metric_id=f"mechanism.neuron_4288.{slug}",
                spec=spec,
                metric_name=metric_name,
                comparison_type="diagnostic",
                condition_a="L20_N4288",
                condition_b="diagnostic_threshold",
                point=tests[slug][unit_key],
                unit=unit,
            )

    def extract_verbosity_confound(self, spec: JsonDict) -> None:
        data = load_json(self.path(spec["source_path"]))
        mean = data["mean_aggregation"]
        self.add_metric_from_point(
            metric_id="mechanism.verbosity.truth_effect_d",
            spec=spec,
            metric_name="truth_effect_d",
            comparison_type="effect_size",
            condition_a="truth",
            condition_b="falsehood",
            point=mean["truth_effect"]["cohens_d"],
            unit="cohen_d",
        )
        self.add_metric_from_point(
            metric_id="mechanism.verbosity.length_effect_d",
            spec=spec,
            metric_name="length_effect_d",
            comparison_type="effect_size",
            condition_a="short",
            condition_b="long",
            point=mean["length_effect"]["cohens_d"],
            unit="cohen_d",
        )
        self.add_metric_from_point(
            metric_id="mechanism.verbosity.truth_mean_diff",
            spec=spec,
            metric_name="truth_mean_diff",
            comparison_type="mean_difference",
            condition_a="truth",
            condition_b="falsehood",
            point=mean["truth_effect"]["mean_diff"],
            unit="activation",
            ci_lower=mean["truth_effect"]["ci"]["ci_lower"],
            ci_upper=mean["truth_effect"]["ci"]["ci_upper"],
            ci_level=0.95,
            ci_method="bootstrap_percentile_mean",
            n=None,
            paired=False,
        )
        self.add_metric_from_point(
            metric_id="mechanism.verbosity.length_mean_diff",
            spec=spec,
            metric_name="length_mean_diff",
            comparison_type="mean_difference",
            condition_a="short",
            condition_b="long",
            point=mean["length_effect"]["mean_diff"],
            unit="activation",
            ci_lower=mean["length_effect"]["ci"]["ci_lower"],
            ci_upper=mean["length_effect"]["ci"]["ci_upper"],
            ci_level=0.95,
            ci_method="bootstrap_percentile_mean",
            n=None,
            paired=False,
        )

    def extract_refusal_overlap(self, spec: JsonDict) -> None:
        data = load_json(self.path(spec["source_path"]))
        for benchmark in ["faitheval", "jailbreak"]:
            entry = data["benchmarks"][benchmark]
            canon = entry["canonical_overlap_vs_primary"]
            self.add_metric_from_point(
                metric_id=f"mechanism.refusal_overlap.{benchmark}.canonical_overlap_vs_primary",
                spec=spec,
                metric_name=f"{benchmark}_canonical_overlap_vs_primary",
                comparison_type="spearman",
                condition_a="canonical_overlap",
                condition_b=entry["primary_outcome"],
                point=canon["estimate"],
                unit="rho",
                n=canon["n"],
                ci_lower=canon["ci"]["lower"],
                ci_upper=canon["ci"]["upper"],
                ci_level=canon["ci"]["level"],
                ci_method=canon["ci"]["method"],
            )
            subspace = entry["subspace_overlap_vs_primary"]
            self.add_metric_from_point(
                metric_id=f"mechanism.refusal_overlap.{benchmark}.subspace_overlap_vs_primary",
                spec=spec,
                metric_name=f"{benchmark}_subspace_overlap_vs_primary",
                comparison_type="spearman",
                condition_a="subspace_overlap",
                condition_b=entry["primary_outcome"],
                point=subspace["estimate"],
                unit="rho",
                n=subspace["n"],
                ci_lower=subspace["ci"]["lower"],
                ci_upper=subspace["ci"]["upper"],
                ci_level=subspace["ci"]["level"],
                ci_method=subspace["ci"]["method"],
            )

    def extract_swing_characterization(self, spec: JsonDict) -> None:
        data = load_json(self.path(spec["source_path"]))
        population = data["population"]
        self.add_metric_from_point(
            metric_id="mechanism.swing.population.swing_count",
            spec=spec,
            metric_name="swing_count",
            comparison_type="population_count",
            condition_a="swing",
            condition_b="swing",
            point=population["swing"],
            unit="count",
            n=population["total"],
            locator="population.swing",
        )
        for subtype in ["R→C", "C→R", "non-monotonic"]:
            entry = data["subtypes"][subtype]
            metric_slug = subtype.replace("→", "_to_").replace("-", "_").lower()
            self.add_metric_from_point(
                metric_id=f"mechanism.swing.subtype.{metric_slug}",
                spec=spec,
                metric_name=f"{metric_slug}_share",
                comparison_type="subtype_share",
                condition_a="swing",
                condition_b=subtype,
                point=entry["proportion"],
                unit="proportion",
                n=population["swing"],
                ci_lower=entry["ci_95"][0],
                ci_upper=entry["ci_95"][1],
                ci_level=0.95,
                ci_method="wilson",
                locator=f"subtypes.{subtype}.proportion",
            )

    def extract_d7_comparison(self, spec: JsonDict) -> None:
        raw_artifact_id = "d7_current_state_summary"
        data = load_json(self.path(self.artifact_by_id[raw_artifact_id]["source_path"]))
        panel = data["current_panel"]
        for condition in [
            "baseline",
            "probe",
            "random_layer_seed1",
            "random_layer_seed2",
            "causal",
        ]:
            entry = panel["conditions"][condition]["strict_harmfulness_normalized"]
            self.add_metric_from_point(
                metric_id=f"mechanism.d7.current_panel.{condition}.strict_harmfulness_normalized",
                spec=spec,
                metric_name=f"{condition}_strict_harmfulness_normalized",
                comparison_type="panel_rate",
                condition_a=condition,
                condition_b=condition,
                point=entry["estimate"],
                unit="proportion",
                n=entry["n"],
                ci_lower=entry["ci"]["lower"],
                ci_upper=entry["ci"]["upper"],
                ci_level=entry["ci"]["level"],
                ci_method=entry["ci"]["method"],
                source_artifact_ids=[raw_artifact_id, spec["artifact_id"]],
                source_locators=[
                    self.source_locator(
                        raw_artifact_id,
                        f"current_panel.conditions.{condition}.strict_harmfulness_normalized.estimate",
                    ),
                    self.source_locator(
                        spec["artifact_id"],
                        f"current_state.current_panel.conditions.{condition}.strict_harmfulness_normalized.estimate",
                        derivation="mirrored_surface",
                    ),
                ],
            )
        causal_vs_probe = panel["direct_comparisons"]["probe_vs_causal"][
            "strict_harmfulness_normalized"
        ]
        self.add_metric_from_point(
            metric_id="mechanism.d7.current_panel.causal_vs_probe_gap_pp",
            spec=spec,
            metric_name="causal_vs_probe_gap_pp",
            comparison_type="paired_effect",
            condition_a="probe",
            condition_b="causal",
            point=-causal_vs_probe["estimate_pp"],
            unit="pp",
            n=500,
            ci_lower=-causal_vs_probe["ci_pp"]["upper"],
            ci_upper=-causal_vs_probe["ci_pp"]["lower"],
            ci_level=causal_vs_probe["ci_pp"]["level"],
            ci_method=causal_vs_probe["ci_pp"]["method"],
            source_artifact_ids=[raw_artifact_id, spec["artifact_id"]],
            source_locators=[
                self.source_locator(
                    raw_artifact_id,
                    "current_panel.direct_comparisons.probe_vs_causal.strict_harmfulness_normalized.estimate",
                ),
                self.source_locator(
                    spec["artifact_id"],
                    "current_state.current_panel.direct_probe_vs_causal.strict_harmfulness_normalized.estimate",
                    derivation="mirrored_surface",
                ),
            ],
        )
        for random_key in ["random_layer_seed1", "random_layer_seed2"]:
            gap = panel["direct_comparisons"][f"{random_key}_vs_causal"][
                "strict_harmfulness_normalized"
            ]
            self.add_metric_from_point(
                metric_id=f"mechanism.d7.current_panel.causal_vs_{random_key}_gap_pp",
                spec=spec,
                metric_name=f"causal_vs_{random_key}_gap_pp",
                comparison_type="paired_effect",
                condition_a=random_key,
                condition_b="causal",
                point=-gap["estimate_pp"],
                unit="pp",
                n=500,
                ci_lower=-gap["ci_pp"]["upper"],
                ci_upper=-gap["ci_pp"]["lower"],
                ci_level=gap["ci_pp"]["level"],
                ci_method=gap["ci_pp"]["method"],
                source_artifact_ids=[raw_artifact_id],
                source_locators=[
                    self.source_locator(
                        raw_artifact_id,
                        f"current_panel.direct_comparisons.{random_key}_vs_causal.strict_harmfulness_normalized.estimate",
                    )
                ],
            )
        self.add_metric_from_point(
            metric_id="mechanism.d7.current_panel.causal_token_cap_count",
            spec=spec,
            metric_name="causal_token_cap_count",
            comparison_type="summary_value",
            condition_a="causal",
            condition_b="causal",
            point=data["artifact_status"]["causal_locked"]["token_cap"],
            unit="count",
            n=500,
            source_artifact_ids=[raw_artifact_id, spec["artifact_id"]],
            source_locators=[
                self.source_locator(
                    raw_artifact_id, "artifact_status.causal_locked.token_cap"
                ),
                self.source_locator(
                    spec["artifact_id"],
                    "token_cap.causal_hits",
                    derivation="mirrored_surface",
                ),
            ],
        )

    def build_readout_examples(self) -> None:
        summary = load_json(
            self.path("data/gemma3_4b/pipeline/classifier_disjoint_summary.json")
        )
        answer_map = load_keyed_jsonl(
            self.path("data/gemma3_4b/pipeline/answer_tokens.jsonl")
        )
        consistency_map = load_keyed_jsonl(
            self.path("data/gemma3_4b/pipeline/consistency_samples.jsonl")
        )
        candidates: dict[str, list[JsonDict]] = defaultdict(list)
        for row in summary["evaluation"]["examples"]:
            qid = row["qid"]
            if qid not in answer_map or qid not in consistency_map:
                continue
            if row["prediction"] and row["label"]:
                stratum = "tp"
            elif (not row["prediction"]) and (not row["label"]):
                stratum = "tn"
            elif row["prediction"] and (not row["label"]):
                stratum = "fp"
            else:
                stratum = "fn"
            candidates[stratum].append(
                {
                    **row,
                    "selection_score": float(row["probability"]),
                    "selection_tiebreak": qid,
                }
            )
        for stratum in ["tp", "tn", "fp", "fn"]:
            selected_rows = self._select_quantile_candidates(
                candidates[stratum],
                score_key="selection_score",
                tiebreak_key="selection_tiebreak",
                fractions=[0.25, 0.75],
            )
            for rank, row in enumerate(selected_rows, start=1):
                qid = row["qid"]
                answer = answer_map[qid]
                consistency = consistency_map[qid]
                self.add_example(
                    example_row(
                        example_id=f"example.readout.{stratum}.{rank}",
                        family="readout_surfaces",
                        act_tags=["act1"],
                        benchmark="pipeline",
                        source_artifact_id="classifier_disjoint_summary",
                        source_artifact_ids=[
                            "classifier_disjoint_summary",
                            "pipeline_answer_tokens",
                            "pipeline_consistency_samples",
                        ],
                        sample_id=qid,
                        selection_stratum=stratum,
                        selection_rank=rank,
                        condition_metadata={
                            "prediction": row["prediction"],
                            "label": row["label"],
                            "probability": row["probability"],
                            "selection_rule": "quantiles_25_75_of_classifier_probability_with_answer_and_consistency_support",
                            "selection_score": row["selection_score"],
                            "selection_tiebreak": qid,
                        },
                        prompt_or_question=answer.get("question", ""),
                        response_text=answer.get("response", ""),
                        raw_label_fields={
                            "answer_token_judge": answer.get("judge"),
                            "consistency_judges": consistency.get("judges"),
                            "ground_truth": consistency.get("ground_truth"),
                        },
                        judge_fields={
                            "classifier_label": row["label"],
                            "classifier_prediction": row["prediction"],
                            "classifier_correct": row["correct"],
                        },
                        paired_example_ids=[],
                    )
                )

    def _paired_examples_from_rows(
        self,
        *,
        family: str,
        act_tags: list[str],
        benchmark: str,
        source_artifact_id: str,
        source_artifact_ids: list[str],
        baseline_path: str,
        comparison_path: str,
        strata: list[str],
        transition_fn: Any,
    ) -> None:
        baseline_rows = {row["id"]: row for row in load_jsonl(self.path(baseline_path))}
        comparison_rows = {
            row["id"]: row for row in load_jsonl(self.path(comparison_path))
        }
        grouped: dict[str, list[JsonDict]] = defaultdict(list)
        for sample_id in sorted(set(baseline_rows) & set(comparison_rows)):
            base = baseline_rows[sample_id]
            comp = comparison_rows[sample_id]
            stratum = transition_fn(base, comp)
            if stratum not in strata:
                continue
            grouped[stratum].append(
                {
                    "sample_id": sample_id,
                    "base": base,
                    "comp": comp,
                    "selection_score": abs(
                        response_length(comp) - response_length(base)
                    ),
                    "selection_tiebreak": sample_id,
                }
            )
        for stratum in strata:
            selected = self._select_quantile_candidates(
                grouped[stratum],
                score_key="selection_score",
                tiebreak_key="selection_tiebreak",
                fractions=[0.5],
            )
            for rank, candidate in enumerate(selected, start=1):
                sample_id = candidate["sample_id"]
                base = candidate["base"]
                comp = candidate["comp"]
                base_example_id = (
                    f"example.{family}.{benchmark}.{stratum}.{rank}.baseline"
                )
                comp_example_id = (
                    f"example.{family}.{benchmark}.{stratum}.{rank}.comparison"
                )
                raw_labels = {
                    "baseline_compliance": base.get("compliance"),
                    "comparison_compliance": comp.get("compliance"),
                }
                if "ground_truth" in base:
                    raw_labels["ground_truth"] = base["ground_truth"]
                if "counterfactual_key" in base:
                    raw_labels["counterfactual_key"] = base["counterfactual_key"]
                if "ground_truth_aliases" in base:
                    raw_labels["ground_truth_aliases"] = base["ground_truth_aliases"]
                for example_id, row, alpha_label, paired_id in [
                    (base_example_id, base, "baseline", comp_example_id),
                    (comp_example_id, comp, "comparison", base_example_id),
                ]:
                    response_text = row.get("response", row.get("chosen", ""))
                    self.add_example(
                        example_row(
                            example_id=example_id,
                            family=family,
                            act_tags=act_tags,
                            benchmark=benchmark,
                            source_artifact_id=source_artifact_id,
                            source_artifact_ids=source_artifact_ids,
                            sample_id=sample_id,
                            selection_stratum=stratum,
                            selection_rank=rank,
                            condition_metadata={
                                "alpha": row["alpha"],
                                "pair_role": alpha_label,
                                "transition": stratum,
                                "selection_rule": "median_abs_response_length_delta_with_sample_id_tiebreak",
                                "selection_score": candidate["selection_score"],
                                "selection_tiebreak": sample_id,
                            },
                            prompt_or_question=row.get("question")
                            or row.get("goal", ""),
                            response_text=response_text,
                            raw_label_fields=raw_labels,
                            judge_fields={
                                "judge": row.get("judge"),
                                "chosen": row.get("chosen"),
                                "metric_value": row.get("metric_value"),
                                "triviaqa_bridge_grade": row.get(
                                    "triviaqa_bridge_grade"
                                ),
                                "csv2_harmful_binary": row.get("csv2", {}).get(
                                    "harmful_binary"
                                ),
                            },
                            paired_example_ids=[paired_id],
                        )
                    )

    def build_intervention_examples(self) -> None:
        self._paired_examples_from_rows(
            family="intervention_surfaces",
            act_tags=["act2", "act3"],
            benchmark="faitheval",
            source_artifact_id="intervention_sweep_site",
            source_artifact_ids=[
                "intervention_sweep_site",
                "faitheval_alpha_0_rows",
                "faitheval_alpha_3_rows",
            ],
            baseline_path="data/gemma3_4b/intervention/faitheval/experiment/alpha_0.0.jsonl",
            comparison_path="data/gemma3_4b/intervention/faitheval/experiment/alpha_3.0.jsonl",
            strata=["0_to_1", "0_to_0"],
            transition_fn=lambda base, comp: (
                f"{int(bool(base['compliance']))}_to_{int(bool(comp['compliance']))}"
            ),
        )
        self._paired_examples_from_rows(
            family="intervention_surfaces",
            act_tags=["act2", "act3"],
            benchmark="falseqa",
            source_artifact_id="falseqa_results",
            source_artifact_ids=[
                "falseqa_results",
                "falseqa_alpha_0_rows",
                "falseqa_alpha_3_rows",
            ],
            baseline_path="data/gemma3_4b/intervention/falseqa/experiment/alpha_0.0.jsonl",
            comparison_path="data/gemma3_4b/intervention/falseqa/experiment/alpha_3.0.jsonl",
            strata=["1_to_1", "1_to_0"],
            transition_fn=lambda base, comp: (
                f"{int(bool(base['compliance']))}_to_{int(bool(comp['compliance']))}"
            ),
        )
        self._paired_examples_from_rows(
            family="intervention_surfaces",
            act_tags=["act2", "act3"],
            benchmark="bioasq",
            source_artifact_id="bioasq_results",
            source_artifact_ids=[
                "bioasq_results",
                "bioasq_alpha_0_rows",
                "bioasq_alpha_3_rows",
            ],
            baseline_path="data/gemma3_4b/intervention/bioasq/experiment/alpha_0.0.jsonl",
            comparison_path="data/gemma3_4b/intervention/bioasq/experiment/alpha_3.0.jsonl",
            strata=["0_to_1", "0_to_0"],
            transition_fn=lambda base, comp: (
                f"{int(bool(base['compliance']))}_to_{int(bool(comp['compliance']))}"
            ),
        )
        self._paired_examples_from_rows(
            family="intervention_surfaces",
            act_tags=["act3"],
            benchmark="jailbreak",
            source_artifact_id="jailbreak_results",
            source_artifact_ids=[
                "jailbreak_results",
                "jailbreak_alpha_0_rows",
                "jailbreak_alpha_3_rows",
            ],
            baseline_path="data/gemma3_4b/intervention/jailbreak/experiment/alpha_0.0.jsonl",
            comparison_path="data/gemma3_4b/intervention/jailbreak/experiment/alpha_3.0.jsonl",
            strata=["1_to_1", "1_to_0"],
            transition_fn=lambda base, comp: (
                f"{int(bool(base['compliance']))}_to_{int(bool(comp['compliance']))}"
            ),
        )

    def build_measurement_examples(self) -> None:
        gold_rows = load_jsonl(
            self.path("tests/gold_labels/jailbreak_cross_alpha_gold.jsonl")
        )
        gold_by_key = {(row["id"], row["alpha"]): row for row in gold_rows}
        binary_files = {
            0.0: self.path(
                "data/gemma3_4b/intervention/jailbreak/experiment/alpha_0.0.jsonl"
            ),
            1.5: self.path(
                "data/gemma3_4b/intervention/jailbreak/experiment/alpha_1.5.jsonl"
            ),
            3.0: self.path(
                "data/gemma3_4b/intervention/jailbreak/experiment/alpha_3.0.jsonl"
            ),
        }
        csv2_v2_files = {
            0.0: self.path(
                "data/gemma3_4b/intervention/jailbreak/csv2_evaluation/alpha_0.0.jsonl"
            ),
            1.5: self.path(
                "data/gemma3_4b/intervention/jailbreak/csv2_evaluation/alpha_1.5.jsonl"
            ),
            3.0: self.path(
                "data/gemma3_4b/intervention/jailbreak/csv2_evaluation/alpha_3.0.jsonl"
            ),
        }
        csv2_v3_files = {
            0.0: self.path("data/judge_validation/csv2_v3/alpha_0.0.jsonl"),
            1.5: self.path("data/judge_validation/csv2_v3/alpha_1.5.jsonl"),
            3.0: self.path("data/judge_validation/csv2_v3/alpha_3.0.jsonl"),
        }
        strongreject_files = {
            0.0: self.path("data/judge_validation/strongreject/alpha_0.0.jsonl"),
            1.5: self.path("data/judge_validation/strongreject/alpha_1.5.jsonl"),
            3.0: self.path("data/judge_validation/strongreject/alpha_3.0.jsonl"),
        }
        binary_by_key = self._merge_alpha_files(binary_files)
        csv2_v2_by_key = self._merge_alpha_files(csv2_v2_files)
        csv2_v3_by_key = self._merge_alpha_files(csv2_v3_files)
        strongreject_by_key = self._merge_alpha_files(strongreject_files)
        candidates: dict[str, list[JsonDict]] = defaultdict(list)
        for key, gold_row in gold_by_key.items():
            if (
                key not in binary_by_key
                or key not in csv2_v2_by_key
                or key not in csv2_v3_by_key
                or key not in strongreject_by_key
            ):
                continue
            sample_id, alpha = key
            phenotype = (
                f"human_{normalize_token(gold_row['human_label'])}"
                f"__binary_{normalize_token(binary_by_key[key].get('judge'))}"
                f"__csv2v2_{normalize_token(csv2_v2_by_key[key].get('csv2', {}).get('harmful_binary'))}"
                f"__csv2v3_{normalize_token(csv2_v3_by_key[key].get('csv2', {}).get('harmful_binary'))}"
                f"__sr_{normalize_token(strongreject_by_key[key].get('strongreject', {}).get('harmful_binary'))}"
                f"__{alpha_code(alpha)}"
            )
            primary = csv2_v3_by_key[key]
            candidates[phenotype].append(
                {
                    "key": key,
                    "primary": primary,
                    "selection_score": len(primary["response"]),
                    "selection_tiebreak": sample_id,
                }
            )
        target_strata = [
            "human_harmful__binary_safe__csv2v2_borderline__csv2v3_yes__sr_no__alpha_0_0",
            "human_harmful__binary_harmful__csv2v2_borderline__csv2v3_yes__sr_no__alpha_0_0",
            "human_harmful__binary_safe__csv2v2_borderline__csv2v3_no__sr_no__alpha_0_0",
            "human_safe__binary_harmful__csv2v2_no__csv2v3_no__sr_no__alpha_3_0",
            "human_harmful__binary_harmful__csv2v2_borderline__csv2v3_yes__sr_yes__alpha_0_0",
            "human_harmful__binary_safe__csv2v2_no__csv2v3_no__sr_no__alpha_1_5",
            "human_harmful__binary_safe__csv2v2_no__csv2v3_no__sr_no__alpha_0_0",
        ]
        for rank, stratum in enumerate(target_strata, start=1):
            selected = self._select_quantile_candidates(
                candidates[stratum],
                score_key="selection_score",
                tiebreak_key="selection_tiebreak",
                fractions=[0.5],
            )[0]
            sample_id, alpha = selected["key"]
            key = (sample_id, alpha)
            primary = selected["primary"]
            self.add_example(
                example_row(
                    example_id=f"example.measurement.{stratum}.{rank}",
                    family="measurement_surfaces",
                    act_tags=["act3"],
                    benchmark="jailbreak_gold",
                    source_artifact_id="csv2_v3_results",
                    source_artifact_ids=[
                        "jailbreak_gold_labels",
                        f"jailbreak_binary_{alpha_code(alpha)}_rows",
                        f"jailbreak_csv2_v2_{alpha_code(alpha)}_rows",
                        f"jailbreak_csv2_v3_{alpha_code(alpha)}_rows",
                        f"jailbreak_strongreject_{alpha_code(alpha)}_rows",
                    ],
                    sample_id=f"{sample_id}@{alpha}",
                    selection_stratum=stratum,
                    selection_rank=rank,
                    condition_metadata={
                        "alpha": alpha,
                        "selection_rule": "median_response_length_with_complete_evaluator_join_and_sample_id_tiebreak",
                        "selection_score": selected["selection_score"],
                        "selection_tiebreak": sample_id,
                    },
                    prompt_or_question=primary["question"],
                    response_text=primary["response"],
                    raw_label_fields={
                        "human_label": gold_by_key[key]["human_label"],
                        "binary_judge": binary_by_key[key].get("judge"),
                        "csv2_v2": csv2_v2_by_key[key]
                        .get("csv2", {})
                        .get("harmful_binary"),
                        "csv2_v3": csv2_v3_by_key[key]
                        .get("csv2", {})
                        .get("harmful_binary"),
                        "strongreject": strongreject_by_key[key]
                        .get("strongreject", {})
                        .get("harmful_binary"),
                    },
                    judge_fields={
                        "judge_label_at_creation": gold_by_key[key].get(
                            "judge_label_at_creation"
                        ),
                        "strongreject_score": strongreject_by_key[key]
                        .get("strongreject", {})
                        .get("score"),
                    },
                    paired_example_ids=[],
                )
            )

    def build_transfer_examples(self) -> None:
        self._paired_examples_from_rows(
            family="transfer_externality_surfaces",
            act_tags=["act3"],
            benchmark="triviaqa_bridge",
            source_artifact_id="bridge_phase3_site",
            source_artifact_ids=[
                "bridge_phase3_site",
                "triviaqa_bridge_alpha_1_rows",
                "triviaqa_bridge_iti_alpha_8_rows",
            ],
            baseline_path="data/gemma3_4b/intervention/triviaqa_bridge/test_experiment/alpha_1.0.jsonl",
            comparison_path="data/gemma3_4b/intervention/triviaqa_bridge_iti_e0_paperfaithful_k12_first-3-tokens/test_experiment/alpha_8.0.jsonl",
            strata=["wrong_to_right", "right_to_wrong"],
            transition_fn=lambda base, comp: (
                "wrong_to_right"
                if float(base.get("metric_value", 0.0)) == 0.0
                and float(comp.get("metric_value", 0.0)) == 1.0
                else "right_to_wrong"
                if float(base.get("metric_value", 0.0)) == 1.0
                and float(comp.get("metric_value", 0.0)) == 0.0
                else ""
            ),
        )
        self._paired_examples_from_rows(
            family="transfer_externality_surfaces",
            act_tags=["act3"],
            benchmark="truthfulqa_mc",
            source_artifact_id="truthfulqa_mc1_iti_fold1",
            source_artifact_ids=[
                "truthfulqa_mc1_iti_fold0_alpha0",
                "truthfulqa_mc1_iti_fold0_alpha8",
            ],
            baseline_path="data/gemma3_4b/intervention/truthfulqa_mc_mc1_iti-head_truthfulqa-paperfaithful_k-12_ranked_seed-42_final-fold0-iti-heads_7723b7d6d7/experiment/alpha_0.0.jsonl",
            comparison_path="data/gemma3_4b/intervention/truthfulqa_mc_mc1_iti-head_truthfulqa-paperfaithful_k-12_ranked_seed-42_final-fold0-iti-heads_7723b7d6d7/experiment/alpha_8.0.jsonl",
            strata=["wrong_to_right", "right_to_wrong"],
            transition_fn=lambda base, comp: (
                "wrong_to_right"
                if float(base.get("metric_value", 0.0)) == 0.0
                and float(comp.get("metric_value", 0.0)) == 1.0
                else "right_to_wrong"
                if float(base.get("metric_value", 0.0)) == 1.0
                and float(comp.get("metric_value", 0.0)) == 0.0
                else ""
            ),
        )
        self._paired_examples_from_rows(
            family="transfer_externality_surfaces",
            act_tags=["act3"],
            benchmark="simpleqa",
            source_artifact_id="simpleqa_ranked_results",
            source_artifact_ids=[
                "simpleqa_ranked_results",
                "simpleqa_alpha_0_rows",
                "simpleqa_alpha_8_rows",
            ],
            baseline_path="data/gemma3_4b/intervention/simpleqa_factual_phrase_iti-head_truthfulqa-paperfaithful_k-12_ranked_seed-42_iti-truthfulqa-paperfaithful-production-iti-head_79f3852513/experiment/alpha_0.0.jsonl",
            comparison_path="data/gemma3_4b/intervention/simpleqa_factual_phrase_iti-head_truthfulqa-paperfaithful_k-12_ranked_seed-42_iti-truthfulqa-paperfaithful-production-iti-head_79f3852513/experiment/alpha_8.0.jsonl",
            strata=["correct_to_incorrect", "incorrect_to_correct"],
            transition_fn=lambda base, comp: (
                "correct_to_incorrect"
                if bool(base.get("compliance")) and not bool(comp.get("compliance"))
                else "incorrect_to_correct"
                if (not bool(base.get("compliance"))) and bool(comp.get("compliance"))
                else ""
            ),
        )
        case_map = {
            row["case_id"]: row
            for row in load_jsonl(
                self.path("data/judge_validation/bridge_irr/test_queue_key.jsonl")
            )
        }
        label_map = {
            row["case_id"]: row
            for row in load_jsonl(
                self.path("data/judge_validation/bridge_irr/adjudicated_labels.jsonl")
            )
        }
        grouped: dict[str, list[JsonDict]] = defaultdict(list)
        for case_id in sorted(set(case_map) & set(label_map)):
            case = case_map[case_id]
            label = label_map[case_id]
            stratum = f"{normalize_token(case['transition'])}_{normalize_token(label['label'])}"
            grouped[stratum].append(
                {
                    "case_id": case_id,
                    "case": case,
                    "label": label,
                    "selection_score": len(case["incorrect_response"]),
                    "selection_tiebreak": case_id,
                }
            )
        strata = [
            "right_to_wrong_wrong_entity_substitution",
            "right_to_wrong_evasion_or_factual_denial",
            "right_to_wrong_answer_dilution",
            "wrong_to_right_wrong_entity_substitution",
        ]
        for rank, stratum in enumerate(strata, start=1):
            selected = self._select_quantile_candidates(
                grouped[stratum],
                score_key="selection_score",
                tiebreak_key="selection_tiebreak",
                fractions=[0.5],
            )[0]
            case_id = selected["case_id"]
            case = selected["case"]
            label = selected["label"]
            self.add_example(
                example_row(
                    example_id=f"example.transfer.bridge_irr.{stratum}.{rank}",
                    family="transfer_externality_surfaces",
                    act_tags=["act3"],
                    benchmark="triviaqa_bridge",
                    source_artifact_id="bridge_irr_key",
                    source_artifact_ids=["bridge_irr_key", "bridge_irr_labels"],
                    sample_id=case_id,
                    selection_stratum=stratum,
                    selection_rank=rank,
                    condition_metadata={
                        "transition": case["transition"],
                        "incorrect_condition": case["incorrect_condition"],
                        "selection_rule": "median_incorrect_response_length_with_case_id_tiebreak",
                        "selection_score": selected["selection_score"],
                        "selection_tiebreak": case_id,
                    },
                    prompt_or_question=case["question_id"],
                    response_text=case["incorrect_response"],
                    raw_label_fields={
                        "paired_correct_response": case["paired_correct_response"],
                        "baseline_response": case["baseline_response"],
                        "comparison_response": case["comparison_response"],
                    },
                    judge_fields={
                        "label": label["label"],
                        "label_source": label["label_source"],
                        "rater_a_label": label["rater_a_label"],
                        "rater_b_label": label["rater_b_label"],
                    },
                    paired_example_ids=[],
                )
            )

    def build_mechanism_examples(self) -> None:
        self._paired_examples_from_rows(
            family="mechanism_diagnostic_surfaces",
            act_tags=["act3"],
            benchmark="jailbreak_d7_full500",
            source_artifact_id="d7_comparison_site",
            source_artifact_ids=[
                "d7_comparison_site",
                "d7_baseline_alpha_1_rows",
                "d7_causal_alpha_4_rows",
            ],
            baseline_path="data/gemma3_4b/intervention/jailbreak_d7/full500_canonical/baseline_noop/csv2_evaluation/alpha_1.0.jsonl",
            comparison_path="data/gemma3_4b/intervention/jailbreak_d7/full500_canonical/causal_locked/csv2_evaluation/alpha_4.0.jsonl",
            strata=["wrong_to_right", "right_to_wrong", "stayed_wrong", "stayed_right"],
            transition_fn=lambda base, comp: (
                "wrong_to_right"
                if (not bool(base.get("compliance"))) and bool(comp.get("compliance"))
                else "right_to_wrong"
                if bool(base.get("compliance")) and (not bool(comp.get("compliance")))
                else "stayed_wrong"
                if (not bool(base.get("compliance")))
                and (not bool(comp.get("compliance")))
                else "stayed_right"
            ),
        )

    def _merge_alpha_files(
        self, mapping: dict[float, Path]
    ) -> dict[tuple[str, float], JsonDict]:
        merged: dict[tuple[str, float], JsonDict] = {}
        for alpha, path in mapping.items():
            for row in load_jsonl(path):
                merged[(str(row["id"]), float(alpha))] = row
        return merged

    def build_crosswalk(self) -> None:
        metric_ids_by_artifact: dict[str, list[str]] = defaultdict(list)
        example_ids_by_artifact: dict[str, list[str]] = defaultdict(list)
        metric_ids_by_locator = self.metric_ids_by_source_locator()
        for row in self.metrics:
            for artifact_id in row["source_artifact_ids"]:
                metric_ids_by_artifact[artifact_id].append(row["metric_id"])
        for row in self.examples:
            for artifact_id in row["source_artifact_ids"]:
                example_ids_by_artifact[artifact_id].append(row["example_id"])
        for spec in self.artifact_specs:
            metric_ids = sorted(
                set(metric_ids_by_artifact.get(spec["artifact_id"], []))
            )
            example_ids = sorted(
                set(example_ids_by_artifact.get(spec["artifact_id"], []))
            )
            self.crosswalk.append(
                {
                    "surface_id": f"source:{spec['artifact_id']}",
                    "surface_kind": "source_file",
                    "source_path": spec["source_path"],
                    "surface_locator": "root",
                    "family": spec["family"],
                    "act_tags": spec["act_tags"],
                    "ledger_metric_ids": metric_ids,
                    "ledger_example_ids": example_ids,
                    "matched_source_locators": [
                        locator
                        for row in self.metrics
                        for locator in row["source_locators"]
                        if locator["artifact_id"] == spec["artifact_id"]
                    ],
                    "provenance_refs": [],
                    "coverage_status": "exact_metric"
                    if metric_ids
                    else "exact_example"
                    if example_ids
                    else "markdown_fallback_only"
                    if spec["content_type"] == "markdown"
                    else "structured_surface_only",
                }
            )
        for source_path in self.manifest["coverage_surfaces"]:
            if not source_path.endswith(".json"):
                continue
            for locator, _ in iter_surface_scalars(load_json(self.path(source_path))):
                metric_ids = metric_ids_by_locator.get((source_path, locator), [])
                self.crosswalk.append(
                    {
                        "surface_id": f"surface:{source_path}:{locator}",
                        "surface_kind": "coverage_surface_scalar",
                        "source_path": source_path,
                        "surface_locator": locator,
                        "family": "",
                        "act_tags": [],
                        "ledger_metric_ids": metric_ids,
                        "ledger_example_ids": [],
                        "matched_source_locators": [
                            locator_row
                            for row in self.metrics
                            for locator_row in row["source_locators"]
                            if locator_row["source_path"] == source_path
                            and locator_row["locator"] == locator
                        ],
                        "provenance_refs": [],
                        "coverage_status": "exact_metric"
                        if metric_ids
                        else "structured_surface_only",
                    }
                )
        provenance_rows = self.parse_number_provenance()
        for row in provenance_rows:
            metric_ids: set[str] = set(
                PROVENANCE_CLAIM_METRIC_IDS.get(row["claim_slug"], [])
            )
            for ref in row["source_refs"]:
                metric_ids.update(
                    self.match_metric_ids_for_ref(ref, metric_ids_by_locator)
                )
            self.crosswalk.append(
                {
                    "surface_id": row["surface_id"],
                    "surface_kind": "manuscript_provenance_row",
                    "source_path": "paper/icml/number_provenance.md",
                    "surface_locator": row["surface_locator"],
                    "family": "",
                    "act_tags": [],
                    "ledger_metric_ids": sorted(metric_ids),
                    "ledger_example_ids": [],
                    "matched_source_locators": [
                        locator
                        for metric in self.metrics
                        if metric["metric_id"] in metric_ids
                        for locator in metric["source_locators"]
                    ],
                    "provenance_refs": row["source_refs"],
                    "coverage_status": "exact_metric"
                    if metric_ids
                    else "unresolved"
                    if row["claim_slug"] in MARKDOWN_FALLBACK_CLAIMS
                    else "markdown_fallback_only",
                }
            )

    def parse_number_provenance(self) -> list[JsonDict]:
        path = self.path("paper/icml/number_provenance.md")
        rows: list[JsonDict] = []
        last_explicit_path: str | None = None
        for line in path.read_text(encoding="utf-8").splitlines():
            if (
                not line.startswith("|")
                or line.startswith("|---")
                or line.startswith("| Claim ")
            ):
                continue
            parts = [part.strip() for part in line.strip().split("|")[1:-1]]
            if len(parts) != 4:
                continue
            claim, _, _, source_column = parts
            source_refs: list[JsonDict] = []
            current_path = last_explicit_path
            for segment in [part.strip() for part in source_column.split(";")]:
                path_match = re.findall(r"`([^`]+)`", segment)
                explicit_paths = [token for token in path_match if "/" in token]
                locators = [token for token in path_match if "/" not in token]
                section_match = re.search(r"(§\d+(?:\.\d+)?)", segment)
                if explicit_paths:
                    current_path = explicit_paths[0]
                    last_explicit_path = current_path
                if current_path is None and not explicit_paths and not section_match:
                    continue
                if not locators and section_match:
                    locators = [section_match.group(1)]
                if not locators:
                    locators = [None]
                for locator in locators:
                    source_refs.append(
                        {
                            "path": current_path,
                            "locator": locator,
                            "raw": segment,
                        }
                    )
            rows.append(
                {
                    "surface_id": f"provenance:{slugify(claim)}",
                    "surface_locator": f"claim:{slugify(claim)}",
                    "claim": claim,
                    "claim_slug": slugify(claim),
                    "source_refs": source_refs,
                }
            )
        return rows

    def build_docs(self, artifact_rows: list[JsonDict]) -> None:
        metrics_by_family: dict[str, list[JsonDict]] = defaultdict(list)
        examples_by_family: dict[str, list[JsonDict]] = defaultdict(list)
        artifacts_by_family: dict[str, list[JsonDict]] = defaultdict(list)
        for row in self.metrics:
            metrics_by_family[row["family"]].append(row)
        for row in self.examples:
            examples_by_family[row["family"]].append(row)
        for row in artifact_rows:
            artifacts_by_family[row["family"]].append(row)
        readme = self.build_readme()
        (GROUND_TRUTH_DIR / "README.md").write_text(readme, encoding="utf-8")
        title_map = {
            "readout_surfaces": "Readout Surfaces",
            "intervention_surfaces": "Intervention Surfaces",
            "measurement_surfaces": "Measurement Surfaces",
            "transfer_externality_surfaces": "Transfer Externality Surfaces",
            "mechanism_diagnostic_surfaces": "Mechanism Diagnostic Surfaces",
        }
        for family_id, family_spec in self.family_specs.items():
            path = GROUND_TRUTH_DIR / family_spec["doc"]
            lines = [
                f"# {title_map[family_id]}",
                "",
                f"Family id: `{family_id}`",
                f"Act tags: `{', '.join(family_spec['act_tags'])}`",
                "Template: neutral structured surface summary.",
                "",
                "## Sources",
                "",
                "| source_id | source_path | role | status |",
                "|---|---|---|---|",
            ]
            for row in sorted(
                artifacts_by_family[family_id], key=lambda item: item["artifact_id"]
            ):
                lines.append(
                    f"| `{row['artifact_id']}` | `{row['source_path']}` | `{row['canonical_role']}` | `{row['replacement_status']}` |"
                )
            lines.extend(
                [
                    "",
                    "## Metrics",
                    "",
                    "| metric_id | benchmark | metric_name | condition_a | condition_b | estimate | unit | n | ci | source_ids |",
                    "|---|---|---|---|---|---:|---|---:|---|---|",
                ]
            )
            for row in sorted(
                metrics_by_family[family_id], key=lambda item: item["metric_id"]
            ):
                ci_text = "—"
                if row["ci_lower"] is not None and row["ci_upper"] is not None:
                    ci_text = f"[{row['ci_lower']}, {row['ci_upper']}]"
                n_text = "—" if row["n"] is None else str(row["n"])
                lines.append(
                    f"| `{row['metric_id']}` | `{row['benchmark']}` | `{row['metric_name']}` | `{row['condition_a']}` | `{row['condition_b']}` | {row['estimate']} | `{row['unit']}` | {n_text} | `{ci_text}` | `{', '.join(row['source_artifact_ids'])}` |"
                )
            lines.extend(["", "## Examples", ""])
            for row in sorted(
                examples_by_family[family_id], key=lambda item: item["example_id"]
            ):
                lines.extend(
                    [
                        f"### `{row['example_id']}`",
                        "",
                        f"Sample id: `{row['sample_id']}`",
                        f"Selection stratum: `{row['selection_stratum']}`",
                        f"Selection rank: `{row['selection_rank']}`",
                        f"Primary source id: `{row['source_artifact_id']}`",
                        f"Source ids: `{', '.join(row['source_artifact_ids'])}`",
                        "",
                        "Question",
                        "````text",
                        row["prompt_or_question"],
                        "````",
                        "",
                        "Response",
                        "````text",
                        row["response_text"],
                        "````",
                        "",
                        "Condition metadata",
                        "````json",
                        json.dumps(
                            row["condition_metadata"],
                            ensure_ascii=True,
                            indent=2,
                            sort_keys=True,
                        ),
                        "````",
                        "",
                        "Raw label fields",
                        "````json",
                        json.dumps(
                            row["raw_label_fields"],
                            ensure_ascii=True,
                            indent=2,
                            sort_keys=True,
                        ),
                        "````",
                        "",
                        "Judge fields",
                        "````json",
                        json.dumps(
                            row["judge_fields"],
                            ensure_ascii=True,
                            indent=2,
                            sort_keys=True,
                        ),
                        "````",
                        "",
                        f"Paired example ids: `{', '.join(row['paired_example_ids'])}`",
                        "",
                    ]
                )
            path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")

    def build_readme(self) -> str:
        return (
            "\n".join(
                [
                    "# Ground-Truth Evidence Pack",
                    "",
                    "Pack mode: structured-first neutral evidence pack.",
                    "Destination: `notes/ground-truth`.",
                    "Ledgers: `artifact_manifest.jsonl`, `metric_ledger.jsonl`, `example_ledger.jsonl`, `surface_crosswalk.jsonl`.",
                    "Coverage statuses: `exact_metric`, `exact_example`, `structured_surface_only`, `markdown_fallback_only`, `unresolved`.",
                ]
            )
            + "\n"
        )


def main() -> None:
    manifest = load_json(SOURCES_PATH)
    builder = Builder(manifest)
    artifact_rows = sorted(builder.artifact_rows(), key=lambda row: row["artifact_id"])
    builder.build_metrics()
    builder.build_examples()
    builder.build_crosswalk()
    builder.metrics.sort(key=lambda row: row["metric_id"])
    builder.examples.sort(key=lambda row: row["example_id"])
    builder.crosswalk.sort(key=lambda row: row["surface_id"])
    dump_jsonl(GROUND_TRUTH_DIR / "artifact_manifest.jsonl", artifact_rows)
    dump_jsonl(GROUND_TRUTH_DIR / "metric_ledger.jsonl", builder.metrics)
    dump_jsonl(GROUND_TRUTH_DIR / "example_ledger.jsonl", builder.examples)
    dump_jsonl(GROUND_TRUTH_DIR / "surface_crosswalk.jsonl", builder.crosswalk)
    builder.build_docs(artifact_rows)


if __name__ == "__main__":
    main()
