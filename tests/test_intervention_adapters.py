"""Tests for intervention mode adapter construction."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import intervention_adapters
from intervention_adapters import (
    NOOP_ALPHA_BY_INTERVENTION_MODE,
    build_direction_output_suffix,
    get_intervention_adapter,
)


class _LifecycleScaler:
    def __init__(self) -> None:
        self.alpha: float | None = None
        self.removed = False

    def remove(self) -> None:
        self.removed = True


def test_direction_adapter_builds_metadata_and_lifecycle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, Any] = {}

    class _DirectionScaler(_LifecycleScaler):
        n_hooks = 3

        def __init__(
            self,
            model: Any,
            directions: Any,
            device: Any,
            *,
            mode: str,
            layers: list[int] | None,
        ) -> None:
            super().__init__()
            captured.update(
                {
                    "model": model,
                    "directions": directions,
                    "device": device,
                    "mode": mode,
                    "layers": layers,
                }
            )

    monkeypatch.setattr(
        intervention_adapters.torch,
        "load",
        lambda path, map_location, weights_only: {
            "path": path,
            "map_location": map_location,
            "weights_only": weights_only,
        },
    )
    monkeypatch.setattr(intervention_adapters, "DirectionScaler", _DirectionScaler)

    direction_path = tmp_path / "refusal_directions.pt"
    args = argparse.Namespace(
        direction_path=str(direction_path),
        direction_mode="add",
        direction_layers="1,3",
    )

    built = get_intervention_adapter("direction").build(
        args=args,
        model="model",
        device="cuda:0",
    )

    assert built.mode == "direction"
    assert built.noop_alpha == NOOP_ALPHA_BY_INTERVENTION_MODE["direction"] == 0.0
    assert built.total_units == 3
    assert "Installed 3 direction hooks" in built.install_message
    assert captured == {
        "model": "model",
        "directions": {
            "path": str(direction_path),
            "map_location": "cpu",
            "weights_only": True,
        },
        "device": "cuda:0",
        "mode": "add",
        "layers": [1, 3],
    }
    assert built.summary_fields == {
        "direction_path": str(direction_path),
        "direction_mode": "add",
        "direction_layers": "1,3",
        "direction_config_key": build_direction_output_suffix(
            str(direction_path), "add", "1,3"
        ),
    }

    built.set_alpha(built.noop_alpha)
    assert built.scaler.alpha == 0.0
    built.remove()
    assert built.scaler.removed is True


def test_iti_adapter_passes_debug_stats_and_lifecycle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, Any] = {}

    class _ITIHeadScaler(_LifecycleScaler):
        n_hooks = 5
        n_heads_selected = 2

        def __init__(
            self,
            model: Any,
            artifact: Any,
            device: Any,
            *,
            family: str,
            k: int,
            selection_strategy: str,
            random_seed: int,
            direction_mode: str,
            direction_random_seed: int | None,
            decode_scope: str,
            collect_debug_stats: bool,
        ) -> None:
            super().__init__()
            captured.update(
                {
                    "model": model,
                    "artifact": artifact,
                    "device": device,
                    "family": family,
                    "k": k,
                    "selection_strategy": selection_strategy,
                    "random_seed": random_seed,
                    "direction_mode": direction_mode,
                    "direction_random_seed": direction_random_seed,
                    "decode_scope": decode_scope,
                    "collect_debug_stats": collect_debug_stats,
                }
            )

    monkeypatch.setattr(
        intervention_adapters,
        "load_iti_artifact",
        lambda path: {"artifact_path": path},
    )
    monkeypatch.setattr(intervention_adapters, "ITIHeadScaler", _ITIHeadScaler)

    head_path = tmp_path / "heads.pt"
    args = argparse.Namespace(
        iti_head_path=str(head_path),
        iti_family="truthfulqa",
        iti_k=2,
        iti_selection_strategy="ranked",
        iti_random_seed=42,
        iti_direction_mode="random",
        iti_direction_random_seed=7,
        iti_decode_scope="generated_only",
        iti_collect_debug_stats=True,
    )

    built = get_intervention_adapter("iti_head").build(
        args=args,
        model="model",
        device="cuda:0",
    )

    assert built.mode == "iti_head"
    assert built.noop_alpha == NOOP_ALPHA_BY_INTERVENTION_MODE["iti_head"] == 0.0
    assert built.total_units == 2
    assert "debug_stats=on" in built.install_message
    assert captured == {
        "model": "model",
        "artifact": {"artifact_path": str(head_path)},
        "device": "cuda:0",
        "family": "iti_truthfulqa",
        "k": 2,
        "selection_strategy": "ranked",
        "random_seed": 42,
        "direction_mode": "random",
        "direction_random_seed": 7,
        "decode_scope": "generated_only",
        "collect_debug_stats": True,
    }
    assert built.summary_fields == {
        "iti_head_path": str(head_path),
        "iti_k": 2,
        "iti_selection_strategy": "ranked",
        "iti_random_seed": 42,
        "iti_family": "truthfulqa",
        "iti_decode_scope": "generated_only",
        "iti_collect_debug_stats": True,
    }

    built.set_alpha(built.noop_alpha)
    assert built.scaler.alpha == 0.0
    built.remove()
    assert built.scaler.removed is True


def test_direction_adapter_requires_artifact_path() -> None:
    args = argparse.Namespace(direction_path=None)

    with pytest.raises(ValueError, match="requires --direction_path"):
        get_intervention_adapter("direction").build(
            args=args,
            model=object(),
            device="cpu",
        )


def test_unknown_intervention_adapter_is_explicit() -> None:
    with pytest.raises(ValueError, match="No intervention adapter registered"):
        get_intervention_adapter("neuron")
