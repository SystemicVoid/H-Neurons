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
    return feature_map


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
