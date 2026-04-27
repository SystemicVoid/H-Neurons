from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from model_registry import (  # noqa: E402
    ModelDimensions,
    assert_causal_lm_supported,
    default_classifier_path_for,
    get_model_spec,
    model_output_root,
    tokenizer_kwargs_for,
)
from classifier import validate_activation_width, validate_classifier_width  # noqa: E402
from intervene_model import get_h_neuron_indices  # noqa: E402
from run_negative_control import (  # noqa: E402
    derive_classifier_geometry,
    derive_h_neuron_layer_distribution,
    flat_to_neuron_map,
    negative_control_schedule,
    sample_layer_matched,
    sample_unconstrained,
    zero_weight_indices_from_classifier,
)


class _Classifier:
    def __init__(self, weights: list[float]):
        self.coef_ = np.array([weights], dtype=float)


class _Config:
    num_hidden_layers = 3
    intermediate_size = 4


def test_mistral_2501_registry_dimensions_and_tokenizer_kwargs() -> None:
    spec = get_model_spec("mistral24b")

    assert spec.hf_id == "mistralai/Mistral-Small-24B-Instruct-2501"
    assert spec.output_root == "data/mistral24b"
    assert spec.dimensions.num_hidden_layers == 40
    assert spec.dimensions.hidden_size == 5120
    assert spec.dimensions.intermediate_size == 32768
    assert spec.dimensions.num_attention_heads == 32
    assert spec.dimensions.num_key_value_heads == 8
    assert spec.dimensions.max_position_embeddings == 32768
    assert tokenizer_kwargs_for(model_path=spec.hf_id) == {"fix_mistral_regex": True}


def test_output_root_resolves_hf_id_and_local_dir_name() -> None:
    assert (
        model_output_root(None, "/workspace/models/Mistral-Small-24B-Instruct-2501")
        == "data/mistral24b"
    )
    assert (
        model_output_root(None, "mistralai/Mistral-Small-24B-Instruct-2501")
        == "data/mistral24b"
    )


def test_unregistered_model_path_requires_explicit_output_root() -> None:
    with pytest.raises(ValueError, match="Cannot infer an output root"):
        model_output_root(None, "/workspace/models/custom-causal-lm")


def test_default_classifier_path_tracks_registered_model() -> None:
    assert (
        default_classifier_path_for("mistral24b") == "models/mistral24b_classifier.pkl"
    )
    assert (
        default_classifier_path_for(None, "mistralai/Mistral-Small-24B-Instruct-2501")
        == "models/mistral24b_classifier.pkl"
    )


def test_mistral_2503_is_registered_but_not_causal_lm_supported() -> None:
    spec = get_model_spec("mistral24b_2503")

    assert spec.loader_family == "mistral3_multimodal"
    assert spec.supported is False
    with pytest.raises(ValueError, match="Mistral3ForConditionalGeneration"):
        assert_causal_lm_supported(model_key="mistral24b_2503")


def test_negative_control_geometry_uses_classifier_and_model_dimensions() -> None:
    classifier = _Classifier(
        [
            0.0,
            0.5,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.25,
            0.0,
            0.75,
            0.0,
        ]
    )
    dimensions = ModelDimensions(
        num_hidden_layers=3,
        hidden_size=8,
        intermediate_size=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=128,
        torch_dtype="bfloat16",
    )

    geometry = derive_classifier_geometry(classifier, dimensions=dimensions)
    layer_distribution = derive_h_neuron_layer_distribution(
        classifier,
        intermediate_size=geometry.intermediate_size,
    )

    assert geometry.num_hidden_layers == 3
    assert geometry.intermediate_size == 4
    assert geometry.total_ffn_neurons == 12
    assert layer_distribution == {0: 1, 2: 2}


def test_negative_control_geometry_rejects_classifier_model_mismatch() -> None:
    classifier = _Classifier([0.0] * 8)
    dimensions = ModelDimensions(
        num_hidden_layers=3,
        hidden_size=8,
        intermediate_size=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=128,
        torch_dtype="bfloat16",
    )

    with pytest.raises(ValueError, match="expected=12"):
        derive_classifier_geometry(classifier, dimensions=dimensions)


def test_layer_matched_sampling_uses_derived_layer_distribution() -> None:
    classifier = _Classifier(
        [0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25, 0.0, 0.75, 0.0]
    )
    zero_indices = zero_weight_indices_from_classifier(classifier)
    layer_distribution = {0: 1, 2: 2}

    sampled = sample_layer_matched(
        zero_indices,
        seed=7,
        layer_distribution=layer_distribution,
        intermediate_size=4,
    )
    neuron_map = flat_to_neuron_map(sampled, intermediate_size=4)

    assert {layer: len(indices) for layer, indices in neuron_map.items()} == {
        0: 1,
        2: 2,
    }
    assert not {1, 8, 10}.intersection(sampled.tolist())


def test_unconstrained_sampling_rejects_too_small_zero_pool() -> None:
    with pytest.raises(ValueError, match="Need 3 zero-weight neurons"):
        sample_unconstrained(np.array([0, 2]), seed=7, n=3)


def test_quick_negative_control_schedule_includes_layer_matched_seed() -> None:
    alphas, configs = negative_control_schedule("faitheval", quick=True)

    assert alphas == [0.0, 1.0, 3.0]
    assert ("seed_0_layer_matched", "layer_matched", 0) in configs


def test_intervention_classifier_mapping_rejects_wrong_model_geometry() -> None:
    classifier = _Classifier([0.0] * 8)

    with pytest.raises(ValueError, match="coef_width=8"):
        get_h_neuron_indices(classifier, _Config())


def test_classifier_width_validation_rejects_wrong_feature_count() -> None:
    classifier = _Classifier([0.0] * 8)

    with pytest.raises(ValueError, match="training activation width"):
        validate_activation_width(np.zeros((2, 8)), 12, context="training")
    with pytest.raises(ValueError, match="loaded classifier coefficient width"):
        validate_classifier_width(classifier, 12, context="loaded")
