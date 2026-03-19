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

import torch


class SAEFeatureScaler:
    """Hook-based SAE feature scaler for intervention experiments.

    Registers forward hooks on post_feedforward_layernorm modules.
    At each hooked layer:
      1. Encodes the activation through the SAE encoder
      2. Scales target SAE features by alpha
      3. Decodes back through the SAE decoder
      4. Replaces the original activation with the modified one

    This operates in SAE feature space rather than neuron space.
    """

    def __init__(self, model, saes, target_features, device):
        """
        Args:
            model: HuggingFace model.
            saes: dict mapping layer_idx -> loaded SAE object.
            target_features: dict mapping layer_idx -> list of SAE feature indices.
            device: torch device.
        """
        self._alpha = 1.0
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

            def make_hook(sae_ref, idx):
                def hook_fn(module, input, output):
                    # output shape: [batch, seq, hidden_size]
                    original_dtype = output.dtype
                    h = output.float()

                    # Encode through SAE
                    features = sae_ref.encode(h)

                    # Scale target features
                    features[:, :, idx] = features[:, :, idx] * self._alpha

                    # Decode back
                    h_modified = sae_ref.decode(features)

                    return h_modified.to(original_dtype)

                return hook_fn

            self.hooks.append(module.register_forward_hook(make_hook(sae, indices)))

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


def load_target_features_from_classifier(classifier_path, layer_indices, d_sae):
    """Extract target SAE feature indices from a trained SAE classifier.

    The classifier's positive-weight features map back to
    (layer, sae_feature_idx) pairs via: layer_pos = flat_idx // d_sae,
    feature_idx = flat_idx % d_sae.

    Args:
        classifier_path: Path to saved sklearn model (.pkl).
        layer_indices: Sorted list of layer indices used during SAE extraction.
        d_sae: Number of SAE features per layer.

    Returns:
        dict mapping layer_idx -> list of SAE feature indices.
    """
    import joblib
    import numpy as np

    model = joblib.load(classifier_path)
    coef = model.coef_[0]
    positive_flat = np.where(coef > 0)[0]

    feature_map = {}
    for flat_idx in positive_flat:
        layer_pos = int(flat_idx // d_sae)
        feature_idx = int(flat_idx % d_sae)
        if layer_pos >= len(layer_indices):
            continue
        layer_id = layer_indices[layer_pos]
        if layer_id not in feature_map:
            feature_map[layer_id] = []
        feature_map[layer_id].append(feature_idx)

    return feature_map
