"""SAE feature-space intervention for H-neuron steering experiments.

Instead of scaling raw neuron activations (HNeuronScaler in intervene_model.py),
this module encodes the MLP output through the SAE, scales target features
in SAE space, decodes back, and replaces the original activation.

The hook point is post_feedforward_layernorm output, matching the point where
Gemma Scope 2 SAEs are trained.

Usage:
    Integrated into run_intervention.py via --intervention_mode sae flag.
    Can also be used standalone for testing.
"""

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch


VALID_SAE_STEERING_MODES = ("full_replacement", "delta_only")


class SAEFeatureScaler:
    """Hook-based SAE feature scaler for intervention experiments.

    Registers forward hooks on post_feedforward_layernorm modules.

    Convention: α=1.0 is no-op (multiplicative identity, like H-neuron mode).
    This differs from ITI_head and direction modes where α=0.0 is baseline.

    Supports two steering modes:
      - ``full_replacement`` (default): encode -> scale -> decode, replacing the
        original activation entirely.  Subject to SAE reconstruction error.
      - ``delta_only``: compute the decoded *difference* from scaling and add it
        to the original activation, cancelling reconstruction error exactly.

    This operates in SAE feature space rather than neuron space.
    """

    def __init__(
        self, model, saes, target_features, device, *, mode="full_replacement"
    ):
        """
        Args:
            model: HuggingFace model.
            saes: dict mapping layer_idx -> loaded SAE object.
            target_features: dict mapping layer_idx -> list of SAE feature indices.
            device: torch device.
            mode: ``"full_replacement"`` or ``"delta_only"``.
        """
        if mode not in VALID_SAE_STEERING_MODES:
            raise ValueError(
                f"Invalid SAE steering mode {mode!r}; "
                f"expected one of {VALID_SAE_STEERING_MODES}"
            )
        self._alpha = 1.0
        self.mode = mode
        self.hooks = []
        self.saes = saes
        self.target_features = target_features
        self._install(model, device)

    def _install(self, model, device):
        for name, module in model.named_modules():
            if "post_feedforward_layernorm" not in name:
                continue
            layer_idx = self._extract_layer_idx(name)
            if layer_idx is None or layer_idx not in self.target_features:
                continue

            sae = self.saes[layer_idx]
            indices = torch.tensor(
                self.target_features[layer_idx], dtype=torch.long, device=device
            )

            def make_hook(sae_ref, idx, steering_mode):
                def hook_fn(module, input, output):
                    if self._alpha == 1.0:
                        return output

                    original_dtype = output.dtype
                    h = output.float().to(sae_ref.device)

                    features = sae_ref.encode(h)
                    f_modified = features.clone()
                    f_modified[:, :, idx] = f_modified[:, :, idx] * self._alpha

                    if steering_mode == "delta_only":
                        delta = sae_ref.decode(f_modified) - sae_ref.decode(features)
                        h_out = h + delta
                    else:
                        h_out = sae_ref.decode(f_modified)

                    return h_out.to(original_dtype)

                return hook_fn

            self.hooks.append(
                module.register_forward_hook(make_hook(sae, indices, self.mode))
            )

    @staticmethod
    def _extract_layer_idx(name):
        for part in name.split("."):
            if part.isdigit():
                return int(part)
        return None

    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, value):
        self._alpha = value

    @property
    def n_hooks(self):
        return len(self.hooks)

    @property
    def n_features(self):
        return sum(len(v) for v in self.target_features.values())

    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()


def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_extraction_metadata(extraction_dir: str) -> dict:
    path = Path(extraction_dir)
    candidates = [path / "metadata.json", path.parent / "metadata.json"]
    for candidate in candidates:
        if candidate.exists():
            return _load_json(str(candidate))
    raise FileNotFoundError(
        f"Could not find SAE extraction metadata for directory {extraction_dir}"
    )


def _validate_classifier_metadata(classifier_summary_path: str, extraction_dir: str):
    summary = _load_json(classifier_summary_path)
    classifier_metadata = summary.get("extraction_metadata")
    if classifier_metadata is None:
        raise ValueError(
            "Classifier summary is missing extraction_metadata; retrain the SAE "
            "classifier with the updated metadata-aware script."
        )

    extraction_metadata = _load_extraction_metadata(extraction_dir)
    for key in (
        "hook_point",
        "sae_release",
        "sae_width",
        "sae_l0",
        "layer_indices",
        "d_in",
        "d_sae",
        "aggregation_method",
    ):
        if classifier_metadata.get(key) != extraction_metadata.get(key):
            raise ValueError(
                f"Classifier/extraction metadata mismatch for {key}: "
                f"{classifier_metadata.get(key)!r} != {extraction_metadata.get(key)!r}"
            )

    return classifier_metadata


def decode_sae_feature_indices(flat_indices, *, layer_indices, d_sae):
    """Map flat SAE indices back to layer/feature coordinates."""
    decoded = []
    for flat_idx in flat_indices:
        layer_pos = int(flat_idx // d_sae)
        if layer_pos >= len(layer_indices):
            continue
        decoded.append(
            {
                "layer": int(layer_indices[layer_pos]),
                "feature": int(flat_idx % d_sae),
                "flat_idx": int(flat_idx),
            }
        )
    return decoded


def build_sae_feature_map(flat_indices, *, layer_indices, d_sae):
    """Convert flat SAE indices into {layer_idx: [feature_idx, ...]}."""
    feature_map = {}
    for decoded in decode_sae_feature_indices(
        flat_indices, layer_indices=layer_indices, d_sae=d_sae
    ):
        feature_map.setdefault(decoded["layer"], []).append(decoded["feature"])
    for layer_idx in feature_map:
        feature_map[layer_idx] = sorted(feature_map[layer_idx])
    return feature_map


def _metadata_from_summary_or_extraction(
    *,
    classifier_summary_path: str | None = None,
    extraction_dir: str | None = None,
) -> dict[str, Any] | None:
    if classifier_summary_path:
        summary = _load_json(classifier_summary_path)
        extraction_metadata = summary.get("extraction_metadata")
        if extraction_metadata is not None:
            if extraction_dir is not None:
                _validate_classifier_metadata(classifier_summary_path, extraction_dir)
            return extraction_metadata
    if extraction_dir is not None:
        return _load_extraction_metadata(extraction_dir)
    return None


def build_sae_feature_entries(flat_indices, *, layer_indices, d_sae):
    """Build normalized SAE feature dicts with layer/feature/flat index."""
    return decode_sae_feature_indices(
        flat_indices,
        layer_indices=layer_indices,
        d_sae=d_sae,
    )


def build_sae_feature_manifest(
    features: list[dict[str, Any]],
    *,
    extraction_metadata: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a normalized SAE feature manifest payload."""
    payload: dict[str, Any] = {
        "schema_version": "sae_feature_manifest/v1",
        "feature_space": "sae",
        "feature_count": len(features),
        "features": features,
    }
    if extraction_metadata is not None:
        payload["extraction_metadata"] = extraction_metadata
    if extra:
        payload.update(extra)
    return payload


def _coerce_feature_entries(
    raw_payload: dict[str, Any] | list[Any],
    *,
    layer_indices: list[int] | None,
    d_sae: int | None,
) -> list[dict[str, Any]]:
    if isinstance(raw_payload, list):
        raw_entries = raw_payload
    elif isinstance(raw_payload, dict) and isinstance(
        raw_payload.get("features"), list
    ):
        raw_entries = raw_payload["features"]
    elif isinstance(raw_payload, dict) and isinstance(
        raw_payload.get("target_features"), dict
    ):
        raw_entries = []
        for layer_key, features in raw_payload["target_features"].items():
            for feature in features:
                raw_entries.append({"layer": int(layer_key), "feature": int(feature)})
    elif isinstance(raw_payload, dict) and all(
        isinstance(key, (int, str)) and isinstance(value, list)
        for key, value in raw_payload.items()
    ):
        raw_entries = []
        for layer_key, features in raw_payload.items():
            for feature in features:
                raw_entries.append({"layer": int(layer_key), "feature": int(feature)})
    else:
        raise ValueError(
            "SAE feature manifest must be a list of features, an object with "
            "'features', an object with 'target_features', or a bare layer map."
        )

    entries: list[dict[str, Any]] = []
    seen_flat: set[int] = set()
    seen_pairs: set[tuple[int, int]] = set()
    valid_layers = set(layer_indices or [])
    for idx, raw_entry in enumerate(raw_entries):
        if not isinstance(raw_entry, dict):
            raise ValueError(f"Feature entry #{idx} must be an object")
        layer = raw_entry.get("layer")
        feature = raw_entry.get("feature")
        flat_idx = raw_entry.get("flat_idx")

        if flat_idx is not None:
            if layer_indices is None or d_sae is None:
                raise ValueError(
                    "Feature manifests with flat_idx entries require layer_indices and "
                    "d_sae metadata."
                )
            decoded = decode_sae_feature_indices(
                [int(flat_idx)],
                layer_indices=layer_indices,
                d_sae=d_sae,
            )
            if not decoded:
                raise ValueError(
                    f"Feature entry #{idx} has invalid flat_idx={flat_idx}"
                )
            decoded_entry = decoded[0]
            if layer is None:
                layer = decoded_entry["layer"]
            if feature is None:
                feature = decoded_entry["feature"]
            if (
                int(layer) != decoded_entry["layer"]
                or int(feature) != decoded_entry["feature"]
            ):
                raise ValueError(
                    f"Feature entry #{idx} disagrees with flat_idx={flat_idx}"
                )
        elif layer is None or feature is None:
            raise ValueError(
                f"Feature entry #{idx} must include either flat_idx or both layer and feature"
            )

        layer = int(layer)
        feature = int(feature)
        if layer_indices is not None and layer not in valid_layers:
            raise ValueError(f"Feature entry #{idx} uses unknown layer {layer}")
        if d_sae is not None and not (0 <= feature < int(d_sae)):
            raise ValueError(
                f"Feature entry #{idx} uses out-of-range feature {feature} for d_sae={d_sae}"
            )
        if flat_idx is None:
            if layer_indices is None or d_sae is None:
                raise ValueError(
                    "Feature manifests without flat_idx require layer_indices and d_sae metadata."
                )
            layer_pos = layer_indices.index(layer)
            flat_idx = layer_pos * int(d_sae) + feature
        flat_idx = int(flat_idx)

        pair = (layer, feature)
        if flat_idx in seen_flat or pair in seen_pairs:
            raise ValueError(
                f"Duplicate SAE feature entry for layer={layer}, feature={feature}"
            )
        seen_flat.add(flat_idx)
        seen_pairs.add(pair)

        entry = {"layer": layer, "feature": feature, "flat_idx": flat_idx}
        for key, value in raw_entry.items():
            if key not in entry:
                entry[key] = value
        entries.append(entry)

    entries.sort(key=lambda item: int(item["flat_idx"]))
    return entries


def load_sae_feature_manifest(
    manifest_path,
    *,
    classifier_summary_path=None,
    extraction_dir=None,
    layer_indices=None,
    d_sae=None,
):
    """Load and validate an explicit SAE feature manifest."""
    payload = _load_json(str(manifest_path))
    manifest_metadata = None
    if isinstance(payload, dict):
        manifest_metadata = payload.get("extraction_metadata")

    resolved_metadata = manifest_metadata or _metadata_from_summary_or_extraction(
        classifier_summary_path=classifier_summary_path,
        extraction_dir=extraction_dir,
    )
    if resolved_metadata is not None:
        layer_indices = resolved_metadata["layer_indices"]
        d_sae = resolved_metadata["d_sae"]
    elif layer_indices is None or d_sae is None:
        raise ValueError(
            "Explicit SAE feature manifests require extraction metadata in the manifest "
            "or caller-provided classifier_summary_path / extraction_dir / layer_indices / d_sae."
        )

    entries = _coerce_feature_entries(
        payload,
        layer_indices=layer_indices,
        d_sae=d_sae,
    )
    feature_map = build_sae_feature_map(
        [entry["flat_idx"] for entry in entries],
        layer_indices=layer_indices,
        d_sae=d_sae,
    )
    normalized_payload = (
        dict(payload)
        if isinstance(payload, dict)
        else build_sae_feature_manifest(entries)
    )
    normalized_payload["features"] = entries
    normalized_payload["feature_count"] = len(entries)
    normalized_payload["target_features"] = {
        str(layer): feature_map[layer] for layer in sorted(feature_map)
    }
    if resolved_metadata is not None:
        normalized_payload["extraction_metadata"] = resolved_metadata
    return normalized_payload


def load_target_features_from_manifest(
    manifest_path,
    *,
    classifier_summary_path=None,
    extraction_dir=None,
    layer_indices=None,
    d_sae=None,
):
    """Load target SAE features from an explicit feature manifest."""
    manifest = load_sae_feature_manifest(
        manifest_path,
        classifier_summary_path=classifier_summary_path,
        extraction_dir=extraction_dir,
        layer_indices=layer_indices,
        d_sae=d_sae,
    )
    return {
        int(layer): [int(feature) for feature in features]
        for layer, features in manifest["target_features"].items()
    }


def load_sae_classifier_coefficients(classifier_path):
    """Load the flattened SAE classifier coefficient vector."""
    model = joblib.load(classifier_path)
    return np.asarray(model.coef_[0], dtype=float)


def get_positive_sae_features_from_classifier(classifier_path, *, layer_indices, d_sae):
    """Load and decode all positive-weight SAE classifier features."""
    coef = load_sae_classifier_coefficients(classifier_path)
    decoded = decode_sae_feature_indices(
        np.flatnonzero(coef > 0),
        layer_indices=layer_indices,
        d_sae=d_sae,
    )
    for feature in decoded:
        feature["weight"] = float(coef[feature["flat_idx"]])
    decoded.sort(key=lambda feature: (-feature["weight"], feature["flat_idx"]))
    return decoded


def get_zero_weight_sae_feature_indices(classifier_path):
    """Return flat indices whose classifier weight is exactly zero."""
    coef = load_sae_classifier_coefficients(classifier_path)
    return np.flatnonzero(coef == 0)


def get_control_sae_feature_indices(classifier_path, *, min_features):
    """Return the cleanest available control pool for random SAE features.

    Prefer exact zero-weight features to avoid contaminating the control pool
    with classifier-selected directions. If the classifier is dense or only
    weakly sparse, fall back to the non-positive pool so the experiment still
    runs for valid `classifier_sae.py` configurations such as L2 probes.
    """
    coef = load_sae_classifier_coefficients(classifier_path)

    zero_weight = np.flatnonzero(coef == 0)
    if len(zero_weight) >= min_features:
        return zero_weight, "zero_weight_only"

    non_positive = np.flatnonzero(coef <= 0)
    if len(non_positive) >= min_features:
        return non_positive, "non_positive_weights"

    raise ValueError(
        f"Need {min_features} control SAE features but found only "
        f"{len(zero_weight)} zero-weight and {len(non_positive)} non-positive "
        "classifier coefficients."
    )


def load_target_features_from_classifier(
    classifier_path,
    *,
    classifier_summary_path=None,
    extraction_dir=None,
    layer_indices=None,
    d_sae=None,
):
    """Extract target SAE feature indices from a trained SAE classifier.

    The classifier's positive-weight features map back to
    (layer, sae_feature_idx) pairs via: layer_pos = flat_idx // d_sae,
    feature_idx = flat_idx % d_sae.

    Args:
        classifier_path: Path to saved sklearn model (.pkl).
        classifier_summary_path: Metrics JSON from classifier_sae.py.
        extraction_dir: Directory within the matching SAE extraction root.
        layer_indices: Fallback layer order if no metadata is provided.
        d_sae: Fallback number of SAE features per layer if no metadata is provided.

    Returns:
        dict mapping layer_idx -> list of SAE feature indices.
    """
    if classifier_summary_path and extraction_dir:
        classifier_metadata = _validate_classifier_metadata(
            classifier_summary_path, extraction_dir
        )
        layer_indices = classifier_metadata["layer_indices"]
        d_sae = classifier_metadata["d_sae"]

    if layer_indices is None or d_sae is None:
        raise ValueError(
            "Provide classifier_summary_path + extraction_dir or explicit "
            "layer_indices + d_sae."
        )

    positive_features = get_positive_sae_features_from_classifier(
        classifier_path,
        layer_indices=layer_indices,
        d_sae=d_sae,
    )
    return build_sae_feature_map(
        [feature["flat_idx"] for feature in positive_features],
        layer_indices=layer_indices,
        d_sae=d_sae,
    )
