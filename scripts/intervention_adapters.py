"""Intervention mode adapters for hook-based run orchestration."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import hashlib
from pathlib import Path
import re
from typing import Any, Protocol

import torch

from intervene_direction import DirectionScaler
from intervene_iti import ITIHeadScaler, load_iti_artifact


NOOP_ALPHA_BY_INTERVENTION_MODE = {
    "neuron": 1.0,
    "sae": 1.0,
    "direction": 0.0,
    "iti_head": 0.0,
}


def _slugify_path_component(value: str, *, max_length: int = 48) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    if not slug:
        return "unnamed"
    return slug[:max_length].rstrip("-") or "unnamed"


def normalize_iti_family(iti_family: str) -> str:
    return iti_family if iti_family.startswith("iti_") else f"iti_{iti_family}"


def parse_direction_layers(direction_layers: str | None) -> list[int] | None:
    if not direction_layers:
        return None
    return [
        int(layer.strip()) for layer in direction_layers.split(",") if layer.strip()
    ]


def build_direction_output_suffix(
    direction_path: str,
    direction_mode: str,
    direction_layers: str | None,
) -> str:
    """Build a stable semantic suffix for default direction experiment dirs."""
    resolved_path = Path(direction_path).expanduser().resolve(strict=False)
    direction_name = _slugify_path_component(resolved_path.stem)
    layers_label = (
        "all-layers"
        if direction_layers is None
        else f"layers-{_slugify_path_component(direction_layers, max_length=32)}"
    )
    config_hash = hashlib.sha256(
        f"{resolved_path}|{direction_mode}|{direction_layers or 'all'}".encode("utf-8")
    ).hexdigest()[:10]
    return f"direction_{direction_mode}_{layers_label}_{direction_name}_{config_hash}"


def build_iti_output_suffix(
    iti_head_path: str,
    iti_family: str,
    iti_k: int,
    iti_selection_strategy: str,
    iti_random_seed: int,
    direction_mode: str = "artifact",
    direction_random_seed: int | None = None,
    iti_decode_scope: str = "full_decode",
) -> str:
    """Build a stable semantic suffix for default ITI experiment dirs."""
    normalized_family = normalize_iti_family(iti_family)
    resolved_path = Path(iti_head_path).expanduser().resolve(strict=False)
    artifact_name = _slugify_path_component(
        resolved_path.parent.name + "-" + resolved_path.stem
    )
    config_hash = hashlib.sha256(
        (
            f"{resolved_path}|{normalized_family}|{iti_k}|"
            f"{iti_selection_strategy}|{iti_random_seed}|"
            f"{direction_mode}|{direction_random_seed}|{iti_decode_scope}"
        ).encode("utf-8")
    ).hexdigest()[:10]
    steering_label = (
        f"k-{iti_k}_{_slugify_path_component(iti_selection_strategy)}"
        f"_seed-{iti_random_seed}"
    )
    if direction_mode != "artifact":
        steering_label += (
            f"_dir-{_slugify_path_component(direction_mode)}"
            f"_dirseed-{direction_random_seed}"
        )
    steering_label += f"_scope-{_slugify_path_component(iti_decode_scope)}"
    return (
        "iti-head_"
        f"{_slugify_path_component(normalized_family)}_{steering_label}_"
        f"{artifact_name}_{config_hash}"
    )


@dataclass
class BuiltIntervention:
    """A constructed intervention with common lifecycle helpers."""

    mode: str
    noop_alpha: float
    scaler: Any
    total_units: int
    install_message: str
    summary_fields: dict[str, Any] = field(default_factory=dict)

    def set_alpha(self, alpha: float) -> None:
        self.scaler.alpha = alpha

    def remove(self) -> None:
        self.scaler.remove()


class InterventionAdapter(Protocol):
    """Common construction interface for intervention modes."""

    mode: str
    noop_alpha: float

    def build(
        self,
        *,
        args: argparse.Namespace,
        model: Any,
        device: Any,
    ) -> BuiltIntervention:
        """Load artifacts, install hooks, and return lifecycle metadata."""


@dataclass(frozen=True)
class DirectionInterventionAdapter:
    mode: str = "direction"
    noop_alpha: float = NOOP_ALPHA_BY_INTERVENTION_MODE["direction"]

    def build(
        self,
        *,
        args: argparse.Namespace,
        model: Any,
        device: Any,
    ) -> BuiltIntervention:
        if not args.direction_path:
            raise ValueError("--intervention_mode direction requires --direction_path")
        print(f"Loading directions: {args.direction_path}")
        directions = torch.load(
            args.direction_path, map_location="cpu", weights_only=True
        )
        direction_layers = None
        if args.direction_layers:
            direction_layers = parse_direction_layers(args.direction_layers)
        scaler = DirectionScaler(
            model,
            directions,
            device,
            mode=args.direction_mode,
            layers=direction_layers,
        )
        return BuiltIntervention(
            mode=self.mode,
            noop_alpha=self.noop_alpha,
            scaler=scaler,
            total_units=scaler.n_hooks,
            install_message=(
                f"Installed {scaler.n_hooks} direction hooks "
                f"(mode={args.direction_mode})"
            ),
            summary_fields={
                "direction_path": args.direction_path,
                "direction_mode": args.direction_mode,
                "direction_layers": args.direction_layers,
                "direction_config_key": build_direction_output_suffix(
                    args.direction_path,
                    args.direction_mode,
                    args.direction_layers,
                ),
            },
        )


@dataclass(frozen=True)
class ITIHeadInterventionAdapter:
    mode: str = "iti_head"
    noop_alpha: float = NOOP_ALPHA_BY_INTERVENTION_MODE["iti_head"]

    def build(
        self,
        *,
        args: argparse.Namespace,
        model: Any,
        device: Any,
    ) -> BuiltIntervention:
        if not args.iti_head_path:
            raise ValueError("--intervention_mode iti_head requires --iti_head_path")
        artifact = load_iti_artifact(args.iti_head_path)
        family = normalize_iti_family(args.iti_family)
        scaler = ITIHeadScaler(
            model,
            artifact,
            device,
            family=family,
            k=args.iti_k,
            selection_strategy=args.iti_selection_strategy,
            random_seed=args.iti_random_seed,
            direction_mode=args.iti_direction_mode,
            direction_random_seed=args.iti_direction_random_seed,
            decode_scope=args.iti_decode_scope,
            collect_debug_stats=args.iti_collect_debug_stats,
        )
        return BuiltIntervention(
            mode=self.mode,
            noop_alpha=self.noop_alpha,
            scaler=scaler,
            total_units=scaler.n_heads_selected,
            install_message=(
                f"Installed {scaler.n_hooks} ITI hooks on "
                f"{scaler.n_heads_selected} heads "
                f"(selection={args.iti_selection_strategy}, "
                f"scope={args.iti_decode_scope}, "
                f"debug_stats={'on' if args.iti_collect_debug_stats else 'off'})"
            ),
            summary_fields={
                "iti_head_path": args.iti_head_path,
                "iti_k": args.iti_k,
                "iti_selection_strategy": args.iti_selection_strategy,
                "iti_random_seed": args.iti_random_seed,
                "iti_family": family,
                "iti_decode_scope": args.iti_decode_scope,
                "iti_collect_debug_stats": args.iti_collect_debug_stats,
            },
        )


INTERVENTION_ADAPTERS: dict[str, InterventionAdapter] = {
    "direction": DirectionInterventionAdapter(),
    "iti_head": ITIHeadInterventionAdapter(),
}


def get_intervention_adapter(intervention_mode: str) -> InterventionAdapter:
    try:
        return INTERVENTION_ADAPTERS[intervention_mode]
    except KeyError as exc:
        supported = ", ".join(sorted(INTERVENTION_ADAPTERS))
        raise ValueError(
            f"No intervention adapter registered for {intervention_mode!r}. "
            f"Supported adapters: {supported}"
        ) from exc
