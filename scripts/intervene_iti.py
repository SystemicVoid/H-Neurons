"""Decode-only head-level ITI intervention at pre-o_proj attention outputs.

The intervention surface is the tensor immediately before ``self_attn.o_proj``.
That tensor is still head-structured, but HuggingFace flattens it to
``[batch, seq, num_heads * head_dim]`` before the projection. This module
reshapes it back to ``[batch, seq, num_heads, head_dim]``, edits selected head
slots, and lets ``o_proj`` mix the modified heads normally.
"""

from __future__ import annotations

from collections import Counter
import random
import time
from pathlib import Path
from typing import Any

import torch

ITI_DECODE_SCOPES = (
    "full_decode",
    "first_token_only",
    "first_3_tokens",
    "first_8_tokens",
)


def _decode_scope_limit(decode_scope: str) -> int | None:
    if decode_scope == "full_decode":
        return None
    if decode_scope == "first_token_only":
        return 1
    if decode_scope == "first_3_tokens":
        return 3
    if decode_scope == "first_8_tokens":
        return 8
    raise ValueError(
        f"decode_scope must be one of {ITI_DECODE_SCOPES}, got {decode_scope!r}"
    )


def _get_decoder_layers(model: torch.nn.Module) -> Any:
    inner = getattr(model, "model")
    language_model = getattr(inner, "language_model", None)
    return language_model.layers if language_model is not None else inner.layers


def load_iti_artifact(path: str | Path) -> dict[str, Any]:
    artifact = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(artifact, dict):
        raise TypeError(f"Expected ITI artifact dict at {path}, got {type(artifact)}")
    return artifact


def select_ranked_heads(
    artifact: dict[str, Any],
    *,
    family: str | None,
    k: int,
    selection_strategy: str = "ranked",
    random_seed: int = 42,
) -> list[dict[str, Any]]:
    ranked_heads = list(artifact.get("ranked_heads", []))
    if not ranked_heads:
        raise ValueError("ITI artifact is missing ranked_heads")

    artifact_family = artifact.get("family")
    if family is not None and artifact_family is not None and family != artifact_family:
        raise ValueError(
            f"ITI artifact family mismatch: expected {family!r}, found {artifact_family!r}"
        )

    if k <= 0:
        raise ValueError("iti_k must be positive")
    if k > len(ranked_heads):
        raise ValueError(
            f"Requested top-{k} heads but artifact only has {len(ranked_heads)} ranked heads"
        )

    if selection_strategy == "ranked":
        return ranked_heads[:k]
    if selection_strategy == "random":
        rng = random.Random(random_seed)
        chosen = rng.sample(ranked_heads, k)
        return sorted(chosen, key=lambda item: (item["layer"], item["head"]))
    if selection_strategy == "layer_matched_random":
        rng = random.Random(random_seed)
        ranked_top_k = ranked_heads[:k]
        layer_profile = Counter(int(item["layer"]) for item in ranked_top_k)
        ranked_top_k_keys = {
            (int(item["layer"]), int(item["head"])) for item in ranked_top_k
        }
        heads_by_layer: dict[int, list[dict[str, Any]]] = {}
        for item in ranked_heads:
            layer_idx = int(item["layer"])
            head_idx = int(item["head"])
            if (layer_idx, head_idx) in ranked_top_k_keys:
                continue
            heads_by_layer.setdefault(layer_idx, []).append(item)

        chosen: list[dict[str, Any]] = []
        for layer_idx, layer_k in sorted(layer_profile.items()):
            available = heads_by_layer.get(layer_idx, [])
            if layer_k > len(available):
                raise ValueError(
                    "Cannot sample layer-matched random heads: "
                    f"layer {layer_idx} needs {layer_k} held-out heads but only "
                    f"{len(available)} remain after excluding ranked top-{k}"
                )
            chosen.extend(rng.sample(available, layer_k))
        return sorted(chosen, key=lambda item: (item["layer"], item["head"]))
    raise ValueError(
        "selection_strategy must be 'ranked', 'random', or "
        f"'layer_matched_random', got {selection_strategy!r}"
    )


class ITIHeadScaler:
    """Decode-only ITI intervention on selected attention heads.

    Convention: ``alpha=0`` is a no-op. For compatibility with the existing
    sweep machinery, ``alpha`` is the intervention strength parameter.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        artifact: dict[str, Any],
        device: torch.device,
        *,
        family: str | None = None,
        k: int = 16,
        selection_strategy: str = "ranked",
        random_seed: int = 42,
        direction_mode: str = "artifact",
        direction_random_seed: int | None = None,
        decode_scope: str = "full_decode",
        collect_debug_stats: bool = False,
        max_debug_steps: int = 32,
    ):
        self._alpha = 0.0
        self.hooks: list[Any] = []
        self.artifact = artifact
        self.selected_heads = select_ranked_heads(
            artifact,
            family=family,
            k=k,
            selection_strategy=selection_strategy,
            random_seed=random_seed,
        )
        self.n_heads_selected = len(self.selected_heads)
        self.family = artifact.get("family", family)
        self.selection_strategy = selection_strategy
        self.random_seed = random_seed
        self.direction_mode = direction_mode
        self.direction_random_seed = direction_random_seed
        self.decode_scope = decode_scope
        self.collect_debug_stats = bool(collect_debug_stats)
        self._max_debug_steps_default = int(max_debug_steps)
        self._decode_scope_limit = _decode_scope_limit(decode_scope)
        self.n_layers = int(artifact["n_layers"])
        self.n_attention_heads = int(artifact["n_attention_heads"])
        self.head_dim = int(artifact["head_dim"])
        self._baseline_sigma_total = sum(
            float(item["sigma"]) for item in artifact["ranked_heads"][:k]
        )
        selected_sigma_total = sum(float(item["sigma"]) for item in self.selected_heads)
        self._random_sigma_scale = 1.0
        if (
            self.selection_strategy in {"random", "layer_matched_random"}
            and selected_sigma_total > 1e-8
            and self._baseline_sigma_total > 0.0
        ):
            self._random_sigma_scale = self._baseline_sigma_total / selected_sigma_total
        self._selected_by_layer = self._build_selected_by_layer(device=device)
        self._primary_step_layer = (
            min(self._selected_by_layer.keys()) if self._selected_by_layer else None
        )
        self._final_step_layer = (
            max(self._selected_by_layer.keys()) if self._selected_by_layer else None
        )
        self.reset_sample_stats()
        self._reset_decode_sequence_state()
        self._install(model)

    def _build_selected_by_layer(
        self, *, device: torch.device
    ) -> dict[int, list[dict[str, Any]]]:
        # Pre-generate random unit vectors if direction_mode == "random"
        random_gen: torch.Generator | None = None
        if self.direction_mode == "random":
            random_gen = torch.Generator(device="cpu")
            random_gen.manual_seed(
                self.direction_random_seed
                if self.direction_random_seed is not None
                else self.random_seed
            )

        by_layer: dict[int, list[dict[str, Any]]] = {}
        for item in self.selected_heads:
            if random_gen is not None:
                rand_dir = torch.randn(self.head_dim, generator=random_gen)
                direction = (rand_dir / rand_dir.norm()).to(device=device)
                direction_source = "random"
            else:
                direction = torch.tensor(
                    item["direction"], dtype=torch.float32, device=device
                )
                direction_source = "artifact"
            sigma = float(item["sigma"])
            applied_sigma = sigma * self._random_sigma_scale
            by_layer.setdefault(int(item["layer"]), []).append(
                {
                    "head": int(item["head"]),
                    "direction": direction,
                    "sigma": sigma,
                    "applied_sigma": applied_sigma,
                    "position_summary": item["position_summary"],
                    "auroc": float(item["auroc"]),
                    "balanced_accuracy": float(item["balanced_accuracy"]),
                    "direction_source": direction_source,
                }
            )
        return by_layer

    def arm_first_decode_token(self) -> None:
        """Treat the next prompt prefill as generation setup for token 1."""
        self._reset_decode_sequence_state()
        self._prefill_decode_token_armed = True

    def _reset_decode_sequence_state(self) -> None:
        self._generated_token_index = 0
        self._current_step_token_index: int | None = None
        self._prefill_decode_token_armed = False

    def _current_generated_token_index(self, layer_idx: int) -> int:
        if (
            self._current_step_token_index is None
            or layer_idx == self._primary_step_layer
        ):
            self._current_step_token_index = self._generated_token_index + 1
        return self._current_step_token_index

    def _scope_allows_token(self, generated_token_index: int) -> bool:
        if self._decode_scope_limit is None:
            return True
        return generated_token_index <= self._decode_scope_limit

    def _finalize_decode_step(
        self,
        layer_idx: int,
        *,
        prefill_decode_step: bool,
        generated_token_index: int,
    ) -> None:
        if layer_idx != self._final_step_layer:
            return
        self._generated_token_index = generated_token_index
        self._current_step_token_index = None
        if prefill_decode_step:
            self._prefill_decode_token_armed = False

    def _install(self, model: torch.nn.Module) -> None:
        decoder_layers = _get_decoder_layers(model)
        for layer_idx, layer_heads in sorted(self._selected_by_layer.items()):
            layer_module = decoder_layers[layer_idx].self_attn.o_proj
            self.hooks.append(
                layer_module.register_forward_pre_hook(
                    self._make_hook(layer_idx, layer_heads)
                )
            )

    def _make_hook(self, layer_idx: int, layer_heads: list[dict[str, Any]]):
        def hook_fn(module, args):
            hook_t0 = time.perf_counter()
            x = args[0]
            self._sample_hook_calls += 1

            if x.ndim != 3:
                self._sample_hook_time_s += time.perf_counter() - hook_t0
                return args

            if x.shape[-1] != self.n_attention_heads * self.head_dim:
                raise ValueError(
                    "Pre-o_proj tensor shape does not match artifact head layout: "
                    f"got trailing dim {x.shape[-1]}, expected "
                    f"{self.n_attention_heads * self.head_dim}"
                )

            prefill_decode_step = x.shape[1] > 1 and self._prefill_decode_token_armed

            # Decode-only ITI: edit decode steps plus the final prompt position
            # that produces the first generated token logits.
            if x.shape[1] != 1 and not prefill_decode_step:
                self._sample_prompt_skip_calls += 1
                self._sample_hook_time_s += time.perf_counter() - hook_t0
                return args

            generated_token_index = self._current_generated_token_index(layer_idx)
            if not self._scope_allows_token(generated_token_index):
                self._sample_scope_skip_calls += 1
                self._finalize_decode_step(
                    layer_idx,
                    prefill_decode_step=prefill_decode_step,
                    generated_token_index=generated_token_index,
                )
                self._sample_hook_time_s += time.perf_counter() - hook_t0
                return args

            if self._alpha == 0.0:
                self._finalize_decode_step(
                    layer_idx,
                    prefill_decode_step=prefill_decode_step,
                    generated_token_index=generated_token_index,
                )
                self._sample_hook_time_s += time.perf_counter() - hook_t0
                return args

            batch, seq_len, _ = x.shape
            reshaped = x.reshape(batch, seq_len, self.n_attention_heads, self.head_dim)
            position_slice = slice(-1, None) if prefill_decode_step else slice(None)
            collect_step_debug = self.collect_debug_stats and (
                len(self._debug_steps) < self._max_debug_steps
            )
            delta_norm_total = 0.0
            activation_norm_total = 0.0
            for item in layer_heads:
                head_idx = item["head"]
                target = reshaped[:, position_slice, head_idx, :]
                # Layers can run on different devices under device_map sharding.
                direction = item["direction"].to(device=target.device)
                delta = (self._alpha * item["applied_sigma"]) * direction
                if collect_step_debug:
                    before = target.float()
                    delta_norm_total += float(delta.norm().item())
                    activation_norm_total += float(before.norm(dim=-1).mean().item())
                target.add_(delta.to(dtype=reshaped.dtype))

            self._finalize_decode_step(
                layer_idx,
                prefill_decode_step=prefill_decode_step,
                generated_token_index=generated_token_index,
            )

            if collect_step_debug:
                self._debug_steps.append(
                    {
                        "generated_token_index": generated_token_index,
                        "layer": layer_idx,
                        "decode_scope": self.decode_scope,
                        "activation_norm_delta": round(delta_norm_total, 6),
                        "selected_head_norm_delta": round(
                            delta_norm_total / max(len(layer_heads), 1), 6
                        ),
                        "selected_head_activation_norm": round(
                            activation_norm_total / max(len(layer_heads), 1), 6
                        ),
                    }
                )

            self._sample_hook_time_s += time.perf_counter() - hook_t0
            return (reshaped.reshape(batch, seq_len, -1),) + args[1:]

        return hook_fn

    @property
    def alpha(self) -> float:
        return self._alpha

    @alpha.setter
    def alpha(self, value: float) -> None:
        self._alpha = value

    @property
    def n_hooks(self) -> int:
        return len(self.hooks)

    def reset_sample_stats(self) -> None:
        self._sample_hook_time_s = 0.0
        self._sample_hook_calls = 0
        self._sample_prompt_skip_calls = 0
        self._sample_scope_skip_calls = 0
        self._debug_steps: list[dict[str, Any]] = []
        self._max_debug_steps = self._max_debug_steps_default

    def consume_sample_stats(self) -> dict[str, Any]:
        stats = {
            "hook_s": round(self._sample_hook_time_s, 6),
            "hook_calls": self._sample_hook_calls,
            "prompt_skip_calls": self._sample_prompt_skip_calls,
            "scope_skip_calls": self._sample_scope_skip_calls,
            "debug_steps": list(self._debug_steps),
            "selection_strategy": self.selection_strategy,
            "direction_mode": self.direction_mode,
            "decode_scope": self.decode_scope,
            "n_heads_selected": self.n_heads_selected,
            "family": self.family,
            "debug_stats_enabled": self.collect_debug_stats,
        }
        self.reset_sample_stats()
        return stats

    def remove(self) -> None:
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
