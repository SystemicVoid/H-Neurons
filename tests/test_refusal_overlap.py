"""Tests for scripts/analyze_refusal_overlap.py."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from analyze_refusal_overlap import (
    JAILBREAK_ALPHAS,
    build_faitheval_prompt_records,
    build_jailbreak_prompt_records,
    compute_overlap_statistics,
    decide_d4_gate,
    project_down_proj_delta,
    sample_layer_matched_neuron_maps,
)


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


class TestProjectionHelpers:
    def test_project_down_proj_delta_matches_manual_column_sum(self):
        mlp_inputs = torch.tensor([[2.0, 3.0, 5.0]], dtype=torch.float32)
        weight = torch.tensor(
            [
                [1.0, 10.0, 100.0],
                [2.0, 20.0, 200.0],
            ],
            dtype=torch.float32,
        )

        projected = project_down_proj_delta(mlp_inputs, weight, [0, 2])
        expected = torch.tensor([[2.0 * 1.0 + 5.0 * 100.0, 2.0 * 2.0 + 5.0 * 200.0]])

        assert torch.allclose(projected, expected)

    def test_direct_sum_overlap_equals_manual_formula(self):
        delta_by_layer = {
            0: torch.tensor([[3.0, 4.0]], dtype=torch.float32),
            1: torch.tensor([[0.0, 5.0]], dtype=torch.float32),
        }
        refusal_directions = {
            0: torch.tensor([1.0, 0.0], dtype=torch.float32),
            1: torch.tensor([0.0, 1.0], dtype=torch.float32),
        }
        refusal_subspaces = {
            0: torch.tensor([[1.0, 0.0]], dtype=torch.float32),
            1: torch.tensor([[0.0, 1.0]], dtype=torch.float32),
        }

        scores = compute_overlap_statistics(
            delta_by_layer,
            refusal_directions,
            refusal_subspaces,
        )

        numerator = 3.0 + 5.0
        denom = np.sqrt(3.0**2 + 4.0**2 + 5.0**2) * np.sqrt(2.0)
        expected_cos = numerator / denom
        expected_subspace = (3.0**2 + 5.0**2) / (3.0**2 + 4.0**2 + 5.0**2)

        assert scores["prompt_signed_cosine"][0] == pytest.approx(expected_cos)
        assert scores["prompt_subspace_fraction"][0] == pytest.approx(expected_subspace)

    def test_streaming_overlap_matches_explicit_concatenation(self):
        delta_by_layer = {
            0: torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32),
            1: torch.tensor([[5.0, 6.0], [7.0, 8.0]], dtype=torch.float32),
        }
        refusal_directions = {
            0: torch.tensor([1.0, 0.0], dtype=torch.float32),
            1: torch.tensor([0.0, 1.0], dtype=torch.float32),
        }
        refusal_subspaces = {
            0: torch.tensor([[1.0, 0.0]], dtype=torch.float32),
            1: torch.tensor([[0.0, 1.0]], dtype=torch.float32),
        }

        scores = compute_overlap_statistics(
            delta_by_layer,
            refusal_directions,
            refusal_subspaces,
        )

        explicit_delta = torch.cat([delta_by_layer[0], delta_by_layer[1]], dim=1)
        explicit_direction = torch.cat(
            [refusal_directions[0], refusal_directions[1]],
            dim=0,
        )
        explicit_cos = (explicit_delta @ explicit_direction) / (
            torch.norm(explicit_delta, dim=1) * torch.norm(explicit_direction)
        )
        explicit_proj_norm2 = torch.tensor(
            [
                delta_by_layer[0][0, 0].item() ** 2
                + delta_by_layer[1][0, 1].item() ** 2,
                delta_by_layer[0][1, 0].item() ** 2
                + delta_by_layer[1][1, 1].item() ** 2,
            ]
        )
        explicit_subspace = explicit_proj_norm2 / torch.sum(explicit_delta**2, dim=1)

        assert np.allclose(scores["prompt_signed_cosine"], explicit_cos.numpy())
        assert np.allclose(
            scores["prompt_subspace_fraction"], explicit_subspace.numpy()
        )

    def test_missing_layers_are_treated_as_zero_in_direct_sum(self):
        delta_by_layer = {
            0: torch.tensor([[3.0, 4.0]], dtype=torch.float32),
        }
        refusal_directions = {
            0: torch.tensor([1.0, 0.0], dtype=torch.float32),
            1: torch.tensor([0.0, 1.0], dtype=torch.float32),
        }
        refusal_subspaces = {
            0: torch.tensor([[1.0, 0.0]], dtype=torch.float32),
            1: torch.tensor([[0.0, 1.0]], dtype=torch.float32),
        }

        scores = compute_overlap_statistics(
            delta_by_layer,
            refusal_directions,
            refusal_subspaces,
        )

        expected_cos = 3.0 / (5.0 * np.sqrt(2.0))
        expected_subspace = 9.0 / 25.0

        assert scores["prompt_signed_cosine"][0] == pytest.approx(expected_cos)
        assert scores["prompt_subspace_fraction"][0] == pytest.approx(expected_subspace)


class TestNullSampling:
    def test_layer_matched_null_preserves_histogram(self):
        counts = {0: 2, 5: 4, 7: 1}
        samples = sample_layer_matched_neuron_maps(
            counts,
            intermediate_size=16,
            n_samples=32,
            seed=123,
        )

        assert len(samples) == 32
        for sample in samples:
            assert {layer: len(indices) for layer, indices in sample.items()} == counts
            for indices in sample.values():
                assert len(indices) == len(set(indices))


class TestPromptReconstruction:
    def test_build_jailbreak_prompt_records_reconstructs_actual_slice(self):
        rows_by_alpha = {}
        base_dir = Path("data/gemma3_4b/intervention/jailbreak/csv2_evaluation")
        for alpha in JAILBREAK_ALPHAS:
            path = base_dir / f"alpha_{alpha:.1f}.jsonl"
            rows_by_alpha[alpha] = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()[:5]
            ]

        records = build_jailbreak_prompt_records(rows_by_alpha)

        assert len(records) == 5
        assert [record.prompt_id for record in records] == [
            row["id"] for row in rows_by_alpha[JAILBREAK_ALPHAS[0]]
        ]
        assert records[0].prompt.startswith("You are DAN")

    def test_build_faitheval_prompt_records_uses_existing_prompt_builder(self):
        rows_by_alpha = {
            alpha: [
                {
                    "id": f"item_{idx}",
                    "question": f"question {idx}",
                    "compliance": bool(idx % 2),
                }
                for idx in range(5)
            ]
            for alpha in (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0)
        }
        samples = [
            {
                "id": f"item_{idx}",
                "context": f"context {idx}",
                "question": f"question {idx}",
                "choices_text": "A) one\nB) two",
                "valid_letters": ["A", "B"],
                "counterfactual_key": "A",
                "num_options": 2,
            }
            for idx in range(5)
        ]

        records = build_faitheval_prompt_records(rows_by_alpha, samples=samples)

        assert len(records) == 5
        assert [record.prompt_id for record in records] == [
            f"item_{idx}" for idx in range(5)
        ]
        assert (
            "If the context conflicts with established knowledge" in records[0].prompt
        )


def _summary_with_gate_metrics(
    *,
    canonical_geometry_ci: tuple[float, float] = (-0.1, 0.1),
    subspace_geometry_ci: tuple[float, float] = (-0.1, 0.1),
    faith_canonical_ci: tuple[float, float] = (-0.1, 0.1),
    faith_subspace_ci: tuple[float, float] = (-0.1, 0.1),
    jailbreak_canonical_ci: tuple[float, float] = (-0.1, 0.1),
    jailbreak_subspace_ci: tuple[float, float] = (-0.1, 0.1),
) -> dict[str, dict]:
    def metric(ci: tuple[float, float]) -> dict[str, dict[str, float]]:
        lower, upper = ci
        return {
            "estimate": (lower + upper) / 2.0,
            "ci": {
                "lower": lower,
                "upper": upper,
            },
        }

    return {
        "headline_geometry": {
            "canonical_overlap_gap_vs_null": metric(canonical_geometry_ci),
            "subspace_overlap_gap_vs_null": metric(subspace_geometry_ci),
        },
        "benchmarks": {
            "faitheval": {
                "canonical_overlap_vs_primary": metric(faith_canonical_ci),
                "subspace_overlap_vs_primary": metric(faith_subspace_ci),
            },
            "jailbreak": {
                "canonical_overlap_vs_primary": metric(jailbreak_canonical_ci),
                "subspace_overlap_vs_primary": metric(jailbreak_subspace_ci),
            },
        },
    }


class TestD4Gate:
    def test_anti_aligned_canonical_evidence_still_escalates_d4(self):
        summary = _summary_with_gate_metrics(
            canonical_geometry_ci=(-0.45, -0.15),
            faith_canonical_ci=(-0.80, -0.30),
            jailbreak_canonical_ci=(-0.70, -0.20),
        )

        gate, statement = decide_d4_gate(summary)

        assert gate == "orthogonalize_d4_immediately"
        assert "refusal-mediated" in statement

    def test_subspace_evidence_can_drive_full_d4_escalation(self):
        summary = _summary_with_gate_metrics(
            subspace_geometry_ci=(0.20, 0.55),
            faith_subspace_ci=(0.25, 0.65),
            jailbreak_subspace_ci=(0.15, 0.50),
        )

        gate, statement = decide_d4_gate(summary)

        assert gate == "orthogonalize_d4_immediately"
        assert "refusal-mediated" in statement

    def test_subspace_jailbreak_evidence_preserves_externality_only_path(self):
        summary = _summary_with_gate_metrics(
            subspace_geometry_ci=(0.20, 0.55),
            jailbreak_subspace_ci=(0.15, 0.50),
        )

        gate, statement = decide_d4_gate(summary)

        assert gate == "keep_d4_as_planned_prioritize_d6_later"
        assert "safety externalities" in statement
