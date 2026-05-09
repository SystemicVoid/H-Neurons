"""Evidence-pack interfaces for reproducible claim bundles."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any


JsonDict = dict[str, Any]
AlphaRowKey = tuple[str, float]

MEASUREMENT_EXAMPLE_SELECTION_RULE = (
    "median_response_length_with_complete_evaluator_join_and_sample_id_tiebreak"
)
MEASUREMENT_EXAMPLE_ALPHAS = (0.0, 1.5, 3.0)
NEUTRAL_STRATUM_BANNED_TOKENS = {
    "blind",
    "older",
    "universal",
    "solo",
    "extra",
    "recovered",
    "spot",
}
TARGET_MEASUREMENT_STRATA = [
    "human_harmful__binary_safe__csv2v2_borderline__csv2v3_yes__sr_no__alpha_0_0",
    "human_harmful__binary_harmful__csv2v2_borderline__csv2v3_yes__sr_no__alpha_0_0",
    "human_harmful__binary_safe__csv2v2_borderline__csv2v3_no__sr_no__alpha_0_0",
    "human_safe__binary_harmful__csv2v2_no__csv2v3_no__sr_no__alpha_3_0",
    "human_harmful__binary_harmful__csv2v2_borderline__csv2v3_yes__sr_yes__alpha_0_0",
    "human_harmful__binary_safe__csv2v2_no__csv2v3_no__sr_no__alpha_1_5",
    "human_harmful__binary_safe__csv2v2_no__csv2v3_no__sr_no__alpha_0_0",
    "human_safe__binary_safe__csv2v2_no__csv2v3_no__sr_no__alpha_0_0",
    "human_harmful__binary_harmful__csv2v2_yes__csv2v3_yes__sr_yes__alpha_1_5",
]


@dataclass(frozen=True)
class EvidenceSource:
    """Source identity and byte binding for an evidence-pack input."""

    artifact_id: str
    source_path: str
    content_sha256: str
    row_count: int | None


@dataclass(frozen=True)
class EvidencePackBundle:
    """Built evidence rows plus the interface data needed to validate them."""

    family: str
    section: str
    example_rows: list[JsonDict]
    source_registry: dict[str, EvidenceSource]
    metric_semantics: JsonDict
    generated_doc_fragments: dict[str, list[str]]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[JsonDict]:
    rows: list[JsonDict] = []
    buffer = ""
    start_line = 1
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip() and not buffer:
            continue
        if not buffer:
            start_line = line_number
            buffer = line
        else:
            buffer = f"{buffer}\n{line}"
        try:
            payload = json.loads(buffer, strict=False)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            raise ValueError(f"{path}:{start_line}: JSONL row must be an object")
        rows.append(payload)
        buffer = ""
    if buffer.strip():
        try:
            payload = json.loads(buffer, strict=False)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"{path}:{start_line}: unrecoverable JSONL row: {exc.msg}"
            ) from exc
        if not isinstance(payload, dict):
            raise ValueError(f"{path}:{start_line}: JSONL row must be an object")
        rows.append(payload)
    return rows


def normalize_token(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_") or "none"


def alpha_label(alpha: float | str) -> str:
    return f"alpha_{float(alpha):.1f}"


def alpha_code(alpha: float | str) -> str:
    return alpha_label(alpha).replace(".", "_")


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


def select_quantile_candidates(
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
            tiebreak = str(row[tiebreak_key])
            if tiebreak not in used_keys:
                chosen.append(row)
                used_keys.add(tiebreak)
                break
    return chosen


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


def measurement_alpha_files(root: Path) -> dict[str, dict[float, Path]]:
    return {
        "binary": {
            0.0: root
            / "data/gemma3_4b/intervention/jailbreak/experiment/alpha_0.0.jsonl",
            1.5: root
            / "data/gemma3_4b/intervention/jailbreak/experiment/alpha_1.5.jsonl",
            3.0: root
            / "data/gemma3_4b/intervention/jailbreak/experiment/alpha_3.0.jsonl",
        },
        "csv2v2": {
            0.0: root
            / "data/gemma3_4b/intervention/jailbreak/csv2_evaluation/alpha_0.0.jsonl",
            1.5: root
            / "data/gemma3_4b/intervention/jailbreak/csv2_evaluation/alpha_1.5.jsonl",
            3.0: root
            / "data/gemma3_4b/intervention/jailbreak/csv2_evaluation/alpha_3.0.jsonl",
        },
        "csv2v3": {
            0.0: root / "data/judge_validation/csv2_v3/alpha_0.0.jsonl",
            1.5: root / "data/judge_validation/csv2_v3/alpha_1.5.jsonl",
            3.0: root / "data/judge_validation/csv2_v3/alpha_3.0.jsonl",
        },
        "sr": {
            0.0: root / "data/judge_validation/strongreject/alpha_0.0.jsonl",
            1.5: root / "data/judge_validation/strongreject/alpha_1.5.jsonl",
            3.0: root / "data/judge_validation/strongreject/alpha_3.0.jsonl",
        },
    }


def merge_alpha_files(mapping: Mapping[float, Path]) -> dict[AlphaRowKey, JsonDict]:
    rows_by_alpha = {alpha: load_jsonl(path) for alpha, path in mapping.items()}
    return _merge_alpha_rows(rows_by_alpha)


def build_measurement_example_bundle(root: Path) -> EvidencePackBundle:
    gold_path = root / "tests/gold_labels/jailbreak_cross_alpha_gold.jsonl"
    gold_rows = load_jsonl(gold_path)
    gold_by_key = key_rows_by_id_alpha(gold_rows, context=str(gold_path))
    alpha_files = measurement_alpha_files(root)
    rows_by_group = {
        group: {alpha: load_jsonl(path) for alpha, path in files_by_alpha.items()}
        for group, files_by_alpha in alpha_files.items()
    }
    merged = {
        group: _merge_alpha_rows(rows_by_alpha)
        for group, rows_by_alpha in rows_by_group.items()
    }

    candidates: dict[str, list[JsonDict]] = defaultdict(list)
    for key, gold_row in gold_by_key.items():
        if any(key not in merged[group] for group in merged):
            continue
        sample_id, alpha = key
        phenotype = (
            f"human_{normalize_token(gold_row['human_label'])}"
            f"__binary_{normalize_token(merged['binary'][key].get('judge'))}"
            f"__csv2v2_{normalize_token(merged['csv2v2'][key].get('csv2', {}).get('harmful_binary'))}"
            f"__csv2v3_{normalize_token(merged['csv2v3'][key].get('csv2', {}).get('harmful_binary'))}"
            f"__sr_{normalize_token(merged['sr'][key].get('strongreject', {}).get('harmful_binary'))}"
            f"__{alpha_code(alpha)}"
        )
        primary = merged["csv2v3"][key]
        candidates[phenotype].append(
            {
                "key": key,
                "primary": primary,
                "selection_score": len(primary["response"]),
                "selection_tiebreak": sample_id,
            }
        )

    examples: list[JsonDict] = []
    for rank, stratum in enumerate(TARGET_MEASUREMENT_STRATA, start=1):
        selected = select_quantile_candidates(
            candidates[stratum],
            score_key="selection_score",
            tiebreak_key="selection_tiebreak",
            fractions=[0.5],
        )[0]
        sample_id, alpha = selected["key"]
        key = (sample_id, alpha)
        primary = selected["primary"]
        examples.append(
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
                    "selection_rule": MEASUREMENT_EXAMPLE_SELECTION_RULE,
                    "selection_score": selected["selection_score"],
                    "selection_tiebreak": sample_id,
                },
                prompt_or_question=primary["question"],
                response_text=primary["response"],
                raw_label_fields={
                    "human_label": gold_by_key[key]["human_label"],
                    "binary_judge": merged["binary"][key].get("judge"),
                    "csv2_v2": merged["csv2v2"][key]
                    .get("csv2", {})
                    .get("harmful_binary"),
                    "csv2_v3": merged["csv2v3"][key]
                    .get("csv2", {})
                    .get("harmful_binary"),
                    "strongreject": merged["sr"][key]
                    .get("strongreject", {})
                    .get("harmful_binary"),
                },
                judge_fields={
                    "judge_label_at_creation": gold_by_key[key].get(
                        "judge_label_at_creation"
                    ),
                    "strongreject_score": merged["sr"][key]
                    .get("strongreject", {})
                    .get("score"),
                },
                paired_example_ids=[],
            )
        )

    return EvidencePackBundle(
        family="measurement_surfaces",
        section="measurement_examples",
        example_rows=examples,
        source_registry=_measurement_source_registry(
            root=root,
            gold_path=gold_path,
            gold_row_count=len(gold_rows),
            alpha_files=alpha_files,
            rows_by_group=rows_by_group,
        ),
        metric_semantics=measurement_example_semantics(),
        generated_doc_fragments=measurement_example_doc_fragments(examples),
    )


def build_measurement_examples(root: Path) -> list[JsonDict]:
    return build_measurement_example_bundle(root).example_rows


def measurement_example_semantics() -> JsonDict:
    return {
        "family": "measurement_surfaces",
        "benchmark": "jailbreak_gold",
        "join_key": ["id", "alpha"],
        "selection_rule": MEASUREMENT_EXAMPLE_SELECTION_RULE,
        "selection_score": "csv2_v3.response_character_length",
        "selection_tiebreak": "sample_id",
        "selection_quantile": 0.5,
        "target_strata": list(TARGET_MEASUREMENT_STRATA),
        "raw_label_fields": [
            "human_label",
            "binary_judge",
            "csv2_v2",
            "csv2_v3",
            "strongreject",
        ],
        "judge_fields": ["judge_label_at_creation", "strongreject_score"],
    }


def key_rows_by_id_alpha(
    rows: list[JsonDict],
    *,
    context: str,
    alpha: float | None = None,
) -> dict[AlphaRowKey, JsonDict]:
    keyed: dict[AlphaRowKey, JsonDict] = {}
    duplicates: dict[AlphaRowKey, list[JsonDict]] = defaultdict(list)
    for row in rows:
        key = (str(row["id"]), float(alpha if alpha is not None else row["alpha"]))
        if key in keyed:
            if not duplicates[key]:
                duplicates[key].append(keyed[key])
            duplicates[key].append(row)
            continue
        keyed[key] = row
    if duplicates:
        raise ValueError(_duplicate_alpha_key_message(context, duplicates))
    return keyed


def measurement_example_doc_fragments(
    example_rows: list[JsonDict],
) -> dict[str, list[str]]:
    return {
        row["example_id"]: [
            f"### {row['selection_stratum']} #{row['selection_rank']}",
            f"Trace: `{row['example_id']}`.",
            f"Source ids: `{', '.join(row['source_artifact_ids'])}`.",
            (
                "Why selected: "
                f"{row['condition_metadata']['selection_rule']}; "
                f"score={row['condition_metadata']['selection_score']}; "
                f"tiebreak={row['condition_metadata']['selection_tiebreak']}."
            ),
        ]
        for row in example_rows
    }


def _merge_alpha_rows(
    rows_by_alpha: Mapping[float, list[JsonDict]],
) -> dict[AlphaRowKey, JsonDict]:
    merged: dict[AlphaRowKey, JsonDict] = {}
    for alpha, rows in rows_by_alpha.items():
        keyed_rows = key_rows_by_id_alpha(
            rows,
            alpha=float(alpha),
            context=f"alpha {float(alpha):g}",
        )
        overlap = set(merged) & set(keyed_rows)
        if overlap:
            duplicates = {
                key: [merged[key], keyed_rows[key]] for key in sorted(overlap)
            }
            raise ValueError(_duplicate_alpha_key_message("alpha merge", duplicates))
        merged.update(keyed_rows)
    return merged


def _duplicate_alpha_key_message(
    context: str,
    duplicates: Mapping[AlphaRowKey, list[JsonDict]],
) -> str:
    details = []
    for (sample_id, alpha), rows in sorted(duplicates.items()):
        row_summaries = [
            json.dumps(
                {
                    "id": row.get("id"),
                    "alpha": row.get("alpha", alpha),
                },
                ensure_ascii=True,
                sort_keys=True,
            )
            for row in rows
        ]
        details.append(f"{sample_id}@{alpha:g}: {row_summaries}")
    return f"{context} contains duplicate (id, alpha) rows: {'; '.join(details)}"


def _measurement_source_registry(
    *,
    root: Path,
    gold_path: Path,
    gold_row_count: int,
    alpha_files: Mapping[str, Mapping[float, Path]],
    rows_by_group: Mapping[str, Mapping[float, list[JsonDict]]],
) -> dict[str, EvidenceSource]:
    sources = {
        "jailbreak_gold_labels": _source_entry(
            root=root,
            artifact_id="jailbreak_gold_labels",
            path=gold_path,
            row_count=gold_row_count,
        ),
        "csv2_v3_results": _source_entry(
            root=root,
            artifact_id="csv2_v3_results",
            path=root / "data/judge_validation/csv2_v3/results.json",
            row_count=None,
        ),
    }
    artifact_prefix_by_group = {
        "binary": "jailbreak_binary",
        "csv2v2": "jailbreak_csv2_v2",
        "csv2v3": "jailbreak_csv2_v3",
        "sr": "jailbreak_strongreject",
    }
    for group, files_by_alpha in alpha_files.items():
        artifact_prefix = artifact_prefix_by_group[group]
        for alpha, path in files_by_alpha.items():
            artifact_id = f"{artifact_prefix}_{alpha_code(alpha)}_rows"
            sources[artifact_id] = _source_entry(
                root=root,
                artifact_id=artifact_id,
                path=path,
                row_count=len(rows_by_group[group][alpha]),
            )
    return dict(sorted(sources.items()))


def _source_entry(
    *,
    root: Path,
    artifact_id: str,
    path: Path,
    row_count: int | None,
) -> EvidenceSource:
    return EvidenceSource(
        artifact_id=artifact_id,
        source_path=_relative_path(path, root=root),
        content_sha256=file_sha256(path),
        row_count=row_count,
    )


def _relative_path(path: Path, *, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()
