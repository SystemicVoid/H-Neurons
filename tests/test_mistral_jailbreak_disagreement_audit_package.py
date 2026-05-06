from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any


sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from export_mistral_jailbreak_disagreement_audit_package import (
    DEFAULT_SOURCE_DIR,
    LABEL_SCHEMA_VERSION,
    MAIN_STRATUM,
    PROMPT_VERSION,
    REVERSE_STRATUM,
    RUBRIC_VERSION,
    build_rows,
    build_stats,
    export_package,
    file_sha256,
    parse_args,
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_disagreement_extraction_counts_and_key_matching() -> None:
    blind_rows, private_rows, verification = build_rows(DEFAULT_SOURCE_DIR)
    stats = build_stats(private_rows)

    assert len(blind_rows) == 164
    assert len(private_rows) == 164
    assert stats["by_stratum"] == {
        MAIN_STRATUM: 139,
        REVERSE_STRATUM: 25,
    }
    assert stats["by_stratum_alpha"] == {
        MAIN_STRATUM: {"0.0": 42, "1.0": 29, "1.5": 24, "3.0": 44},
        REVERSE_STRATUM: {"0.0": 1, "1.0": 5, "1.5": 11, "3.0": 8},
    }
    assert stats["csv3_primary_outcome_by_stratum"][MAIN_STRATUM] == {
        "benign_answer": 114,
        "deflection": 18,
        "refusal": 7,
    }
    assert (
        stats["priority_subset"][
            "main_stratum_benign_answer_strongreject_score_ge_0_75"
        ]
        == 103
    )

    for alpha_label, row in verification["key_matching"].items():
        assert alpha_label in {"0.0", "1.0", "1.5", "3.0"}
        assert row == {
            "experiment_rows": 500,
            "csv3_rows": 500,
            "strongreject_rows": 500,
            "matched_keys": 500,
            "matched_responses": 500,
        }


def test_blind_rows_omit_machine_judge_answer_key_fields(tmp_path: Path) -> None:
    output_dir = tmp_path / "audit_package"
    manifest = export_package(parse_args(["--output-dir", str(output_dir)]))

    blind_rows = load_jsonl(output_dir / "review_cases_blind.jsonl")
    private_rows = load_jsonl(output_dir / "private_analysis_key.jsonl")
    forbidden_blind_keys = {
        "stratum",
        "priority_reason",
        "priority_tuple",
        "csv3",
        "csv2",
        "strongreject",
        "harmful_binary",
        "primary_outcome",
        "score",
        "reasoning",
    }
    for row in blind_rows:
        assert not (forbidden_blind_keys & set(row))
        assert {
            "review_case_id",
            "source_id",
            "alpha",
            "prompt",
            "goal",
            "response",
        } <= set(row)

    assert {row["review_case_id"] for row in blind_rows} == {
        row["review_case_id"] for row in private_rows
    }
    assert {row["stratum"] for row in private_rows} == {
        MAIN_STRATUM,
        REVERSE_STRATUM,
    }
    assert (
        file_sha256(output_dir / "review_cases_blind.jsonl")
        == manifest["files"]["review_cases_blind.jsonl"]["content_sha256"]
    )


def test_priority_queue_starts_with_main_benign_high_score_rows(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "audit_package"
    export_package(parse_args(["--output-dir", str(output_dir)]))
    private_rows = load_jsonl(output_dir / "private_analysis_key.jsonl")

    high_priority_rows = private_rows[:103]
    assert all(row["stratum"] == MAIN_STRATUM for row in high_priority_rows)
    assert all(
        row["csv3"]["primary_outcome"] == "benign_answer" for row in high_priority_rows
    )
    assert all(
        float(row["strongreject"]["score"]) >= 0.75 for row in high_priority_rows
    )
    assert private_rows[103]["stratum"] == MAIN_STRATUM
    assert private_rows[103]["csv3"]["primary_outcome"] == "benign_answer"
    assert float(private_rows[103]["strongreject"]["score"]) < 0.75


def test_rater_readme_omits_stratum_ordering_details(tmp_path: Path) -> None:
    output_dir = tmp_path / "audit_package"
    export_package(parse_args(["--output-dir", str(output_dir)]))

    readme = (output_dir / "README.md").read_text(encoding="utf-8")
    assert "## Review Order" not in readme
    assert MAIN_STRATUM not in readme
    assert REVERSE_STRATUM not in readme
    assert "stratum" not in readme.lower()
    assert "CSV-v3 safe / StrongREJECT harmful" not in readme
    assert "StrongREJECT scores first" not in readme
    assert "review_order` as opaque queue identifiers" in readme


def test_static_ui_embeds_cases_schema_and_shortcuts(tmp_path: Path) -> None:
    output_dir = tmp_path / "audit_package"
    export_package(parse_args(["--output-dir", str(output_dir)]))

    index_html = (output_dir / "index.html").read_text(encoding="utf-8")
    label_schema = json.loads((output_dir / "label_schema.json").read_text())
    rule_sha256 = file_sha256(output_dir / "adjudication_rule.md")
    label_schema_sha256 = file_sha256(output_dir / "label_schema.json")
    assert "color-scheme: dark" in index_html
    assert 'addEventListener("keydown"' in index_html
    assert 'setLabel("HARMFUL")' in index_html
    assert 'setLabel("SAFE")' in index_html
    assert 'setLabel("AMBIGUOUS_OR_MALFORMED")' in index_html
    assert "const storageScope = [" in index_html
    assert "labelSchemaFileSha" in index_html
    assert "adjudicationRuleSha" in index_html
    assert "sanitizeStoredRows" in index_html
    assert "hasCurrentPackageProvenance" in index_html
    assert label_schema_sha256 in index_html
    assert rule_sha256 in index_html
    assert label_schema["properties"]["schema_version"]["const"] == (
        LABEL_SCHEMA_VERSION
    )
    assert label_schema["properties"]["blind_cases_file_sha256"]["const"] == (
        file_sha256(output_dir / "review_cases_blind.jsonl")
    )
    assert label_schema["properties"]["label_schema_version"]["const"] == (
        LABEL_SCHEMA_VERSION
    )
    assert label_schema["properties"]["label_schema_file_sha256"]["pattern"] == (
        "^[0-9a-f]{64}$"
    )
    assert label_schema["properties"]["adjudication_rule_sha256"]["const"] == (
        rule_sha256
    )
    assert label_schema["properties"]["rubric_version"]["const"] == RUBRIC_VERSION
    assert label_schema["properties"]["prompt_version"]["const"] == PROMPT_VERSION
    assert "label_schema_file_sha256" in label_schema["required"]
