"""Directional intervention on the residual stream for refusal steering.

Implements two modes:
- 'ablate': remove the refusal direction component from hidden states
- 'add': inject the refusal direction into hidden states

Designed for integration with run_intervention.py via --intervention_mode direction.

Usage:
    Integrated into run_intervention.py via --intervention_mode direction flag.
    Can also be used standalone for testing.
"""

from __future__ import annotations

import time
from typing import Any

import torch


VALID_DIRECTION_MODES = ("ablate", "add")


def _get_decoder_layers(model: torch.nn.Module) -> Any:
    inner = getattr(model, "model")
    language_model = getattr(inner, "language_model", None)
    return language_model.layers if language_model is not None else inner.layers


class DirectionScaler:
    """Hook-based directional intervention on residual stream.

    Supports two modes:
    - 'ablate': remove direction component: x = x - beta * (x @ d) * d
    - 'add': add direction: x = x + beta * d

    Convention: beta=0.0 is no-op (distinct from H-neuron alpha=1.0).
    The ``alpha`` property is the beta parameter, named for API compatibility
    with HNeuronScaler's sweep machinery.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        directions: dict[int, torch.Tensor],
        device: torch.device,
        *,
        mode: str = "ablate",
        layers: list[int] | None = None,
    ):
        if mode not in VALID_DIRECTION_MODES:
            raise ValueError(
                f"Invalid direction mode {mode!r}; expected one of {VALID_DIRECTION_MODES}"
            )
        self._alpha = 0.0  # beta=0 is no-op
        self.mode = mode
        self.hooks: list = []
        self.directions = directions
        self._target_layers = (
            layers if layers is not None else sorted(directions.keys())
        )
        self.reset_sample_stats()
        self._install(model, device)

    def _install(self, model: torch.nn.Module, device: torch.device):
        decoder_layers = _get_decoder_layers(model)
        for layer_idx in self._target_layers:
            if layer_idx not in self.directions:
                continue
            direction = self.directions[layer_idx].to(
                device=device, dtype=torch.float32
            )
            # Ensure unit vector
            direction = direction / direction.norm()
            layer_module = decoder_layers[layer_idx]

            def make_hook(d: torch.Tensor, dir_mode: str):
                def hook_fn(module, input, output):
                    hook_t0 = time.perf_counter()
                    if self._alpha == 0.0:
                        self._sample_hook_calls += 1
                        self._sample_hook_time_s += time.perf_counter() - hook_t0
                        return output

                    if isinstance(output, tuple):
                        hidden_states = output[0]
                    else:
                        hidden_states = output

                    original_dtype = hidden_states.dtype
                    h = hidden_states.float()
                    d_vec = d.to(h.device)

                    if dir_mode == "ablate":
                        # Remove projection onto direction, scaled by beta
                        proj = (h @ d_vec).unsqueeze(-1)  # [..., seq_len, 1]
                        h = h - self._alpha * proj * d_vec
                    else:
                        # Add direction, scaled by beta
                        h = h + self._alpha * d_vec

                    h = h.to(original_dtype)

                    self._sample_hook_calls += 1
                    self._sample_hook_time_s += time.perf_counter() - hook_t0

                    if isinstance(output, tuple):
                        return (h,) + output[1:]
                    return h

                return hook_fn

            self.hooks.append(
                layer_module.register_forward_hook(make_hook(direction, self.mode))
            )

    @staticmethod
    def _extract_layer_idx(name: str) -> int | None:
        for part in name.split("."):
            if part.isdigit():
                return int(part)
        return None

    @property
    def alpha(self) -> float:
        """Beta parameter (named alpha for API compatibility with sweep machinery)."""
        return self._alpha

    @alpha.setter
    def alpha(self, value: float):
        self._alpha = value

    @property
    def n_hooks(self) -> int:
        return len(self.hooks)

    def reset_sample_stats(self):
        self._sample_hook_time_s = 0.0
        self._sample_hook_calls = 0

    def consume_sample_stats(self) -> dict[str, float | int]:
        stats = {
            "hook_s": round(self._sample_hook_time_s, 6),
            "hook_calls": self._sample_hook_calls,
        }
        self.reset_sample_stats()
        return stats

    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()
