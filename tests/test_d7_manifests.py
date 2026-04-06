"""Tests for D7 JBB manifest construction."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import build_d7_jbb_manifests as d7_manifest


def _fake_behavior_row(index: int, *, split: str) -> d7_manifest.BehaviorRow:
    return d7_manifest.BehaviorRow(
        index=index,
        behavior=f"behavior_{index}",
        category=f"category_{index % 5}",
        source="test",
        goal=f"{split}_goal_{index}",
        target=f"{split}_target_{index}",
    )


def _fake_maps(
    n: int = 140,
) -> tuple[dict[int, d7_manifest.BehaviorRow], dict[int, d7_manifest.BehaviorRow]]:
    harmful = {index: _fake_behavior_row(index, split="harmful") for index in range(n)}
    benign = {index: _fake_behavior_row(index, split="benign") for index in range(n)}
    return harmful, benign


def test_manifest_builder_is_deterministic_for_fixed_seed() -> None:
    harmful, benign = _fake_maps()

    payload_a = d7_manifest.build_d7_jbb_manifest_payload(
        harmful_by_index=harmful,
        benign_by_index=benign,
        seed=42,
        extraction_behaviors=10,
        pilot_behaviors=20,
        full_behaviors=100,
        n_templates=5,
        extraction_val_behaviors=2,
    )
    payload_b = d7_manifest.build_d7_jbb_manifest_payload(
        harmful_by_index=harmful,
        benign_by_index=benign,
        seed=42,
        extraction_behaviors=10,
        pilot_behaviors=20,
        full_behaviors=100,
        n_templates=5,
        extraction_val_behaviors=2,
    )

    assert payload_a["extraction_harmful_ids"] == payload_b["extraction_harmful_ids"]
    assert payload_a["pilot_harmful_ids"] == payload_b["pilot_harmful_ids"]
    assert payload_a["full_harmful_ids"] == payload_b["full_harmful_ids"]


def test_manifest_builder_enforces_disjoint_extraction_and_pilot_behaviors() -> None:
    harmful, benign = _fake_maps()
    payload = d7_manifest.build_d7_jbb_manifest_payload(
        harmful_by_index=harmful,
        benign_by_index=benign,
        seed=42,
        extraction_behaviors=10,
        pilot_behaviors=20,
        full_behaviors=100,
        n_templates=5,
        extraction_val_behaviors=2,
    )

    extraction = set(payload["splits"]["extraction_behavior_indices"])
    pilot = set(payload["splits"]["pilot_behavior_indices"])
    full = set(payload["splits"]["full_behavior_indices"])

    assert extraction.isdisjoint(pilot)
    assert pilot.isdisjoint(full)
    assert payload["splits"]["disjoint_extraction_vs_pilot"] is True
    assert payload["splits"]["disjoint_pilot_vs_full"] is True
    assert len(payload["extraction_pairs"]) == 50
    assert len(payload["pilot_harmful_ids"]) == 100
    assert len(payload["full_harmful_ids"]) == 500


def test_manifest_builder_rejects_full_size_larger_than_non_pilot_pool() -> None:
    harmful, benign = _fake_maps(n=100)

    try:
        d7_manifest.build_d7_jbb_manifest_payload(
            harmful_by_index=harmful,
            benign_by_index=benign,
            seed=42,
            extraction_behaviors=10,
            pilot_behaviors=20,
            full_behaviors=100,
            n_templates=5,
            extraction_val_behaviors=2,
        )
    except ValueError as exc:
        assert "exceeds non-pilot behavior count" in str(exc)
    else:
        raise AssertionError(
            "Expected ValueError when full split exceeds non-pilot pool"
        )


def test_extraction_pairs_preserve_harmful_benign_alignment() -> None:
    harmful, benign = _fake_maps()
    payload = d7_manifest.build_d7_jbb_manifest_payload(
        harmful_by_index=harmful,
        benign_by_index=benign,
        seed=42,
        extraction_behaviors=10,
        pilot_behaviors=20,
        full_behaviors=100,
        n_templates=5,
        extraction_val_behaviors=2,
    )

    for pair in payload["extraction_pairs"]:
        h_split, h_index, h_template = d7_manifest._parse_jbb_sample_id(
            pair["harmful_sample_id"]
        )
        b_split, b_index, b_template = d7_manifest._parse_jbb_sample_id(
            pair["benign_sample_id"]
        )
        assert h_split == "harmful"
        assert b_split == "benign"
        assert h_index == b_index == pair["behavior_index"]
        assert h_template == b_template == pair["template_idx"]


def test_jailbreak_templates_parity() -> None:
    """JAILBREAK_TEMPLATES must be identical in run_intervention.py and build_d7_jbb_manifests.py."""
    from run_intervention import JAILBREAK_TEMPLATES as RUN_TEMPLATES

    assert RUN_TEMPLATES == d7_manifest.JAILBREAK_TEMPLATES, (
        "JAILBREAK_TEMPLATES diverged between run_intervention.py and build_d7_jbb_manifests.py"
    )
