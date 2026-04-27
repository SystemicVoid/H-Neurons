"""Central registry for model-specific H-neuron pipeline settings."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import re
from typing import Any


DEFAULT_MODEL_KEY = "gemma3_4b_it"


@dataclass(frozen=True)
class ModelDimensions:
    num_hidden_layers: int
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int | None
    max_position_embeddings: int
    torch_dtype: str | None

    @property
    def total_ffn_neurons(self) -> int:
        return self.num_hidden_layers * self.intermediate_size

    def to_dict(self) -> dict[str, int | str | None]:
        payload = asdict(self)
        payload["total_ffn_neurons"] = self.total_ffn_neurons
        return payload


@dataclass(frozen=True)
class ModelSpec:
    key: str
    slug: str
    hf_id: str
    output_root: str
    dimensions: ModelDimensions
    tokenizer_kwargs: dict[str, Any] = field(default_factory=dict)
    loader_family: str = "causal_lm"
    supported: bool = True
    aliases: tuple[str, ...] = ()
    local_dir_name: str | None = None
    default_classifier_path: str | None = None
    notes: str = ""

    def tokenizer_kwargs_copy(self) -> dict[str, Any]:
        return dict(self.tokenizer_kwargs)

    def to_metadata(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "slug": self.slug,
            "hf_id": self.hf_id,
            "output_root": self.output_root,
            "loader_family": self.loader_family,
            "supported": self.supported,
            "tokenizer_kwargs": self.tokenizer_kwargs_copy(),
            "dimensions": self.dimensions.to_dict(),
            "default_classifier_path": self.default_classifier_path,
            "notes": self.notes,
        }


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "gemma3_4b_it": ModelSpec(
        key="gemma3_4b_it",
        slug="gemma3_4b",
        hf_id="google/gemma-3-4b-it",
        output_root="data/gemma3_4b",
        local_dir_name="gemma-3-4b-it",
        default_classifier_path="models/gemma3_4b_classifier.pkl",
        aliases=(
            "gemma3_4b",
            "gemma-3-4b-it",
            "google/gemma-3-4b-it",
        ),
        dimensions=ModelDimensions(
            num_hidden_layers=34,
            hidden_size=2560,
            intermediate_size=10240,
            num_attention_heads=8,
            num_key_value_heads=4,
            max_position_embeddings=131072,
            torch_dtype="bfloat16",
        ),
    ),
    "mistral_small_24b_instruct_2501": ModelSpec(
        key="mistral_small_24b_instruct_2501",
        slug="mistral24b",
        hf_id="mistralai/Mistral-Small-24B-Instruct-2501",
        output_root="data/mistral24b",
        local_dir_name="Mistral-Small-24B-Instruct-2501",
        default_classifier_path="models/mistral24b_classifier.pkl",
        aliases=(
            "mistral24b",
            "mistral-small-24b-instruct-2501",
            "mistralai/Mistral-Small-24B-Instruct-2501",
        ),
        tokenizer_kwargs={"fix_mistral_regex": True},
        dimensions=ModelDimensions(
            num_hidden_layers=40,
            hidden_size=5120,
            intermediate_size=32768,
            num_attention_heads=32,
            num_key_value_heads=8,
            max_position_embeddings=32768,
            torch_dtype="bfloat16",
        ),
        notes=(
            "Canonical Mistral 24B second-model extension. Same-family 2501 "
            "checkpoint, not exact Mistral Small 3.1/2503 replication."
        ),
    ),
    "mistral_small_31_24b_instruct_2503": ModelSpec(
        key="mistral_small_31_24b_instruct_2503",
        slug="mistral24b_2503",
        hf_id="mistralai/Mistral-Small-3.1-24B-Instruct-2503",
        output_root="data/mistral24b_2503",
        local_dir_name="Mistral-Small-3.1-24B-Instruct-2503",
        aliases=(
            "mistral24b_2503",
            "mistral-small-3.1-24b-instruct-2503",
            "mistralai/Mistral-Small-3.1-24B-Instruct-2503",
        ),
        tokenizer_kwargs={"fix_mistral_regex": True},
        loader_family="mistral3_multimodal",
        supported=False,
        dimensions=ModelDimensions(
            num_hidden_layers=40,
            hidden_size=5120,
            intermediate_size=32768,
            num_attention_heads=32,
            num_key_value_heads=8,
            max_position_embeddings=131072,
            torch_dtype="bfloat16",
        ),
        notes=(
            "Deferred third-checkpoint extension. Requires a Mistral3 processor/"
            "Mistral3ForConditionalGeneration path before canonical runs."
        ),
    ),
}


def _identifier_candidates(value: str) -> set[str]:
    normalized = value.strip().rstrip("/")
    candidates = {normalized.lower()}
    if normalized:
        candidates.add(Path(normalized).name.lower())
    return candidates


def _alias_map() -> dict[str, str]:
    aliases: dict[str, str] = {}
    for key, spec in MODEL_REGISTRY.items():
        values = {
            key,
            spec.slug,
            spec.hf_id,
            *(spec.aliases or ()),
        }
        if spec.local_dir_name:
            values.add(spec.local_dir_name)
        for value in values:
            for candidate in _identifier_candidates(value):
                aliases[candidate] = key
    return aliases


def registered_model_keys() -> tuple[str, ...]:
    return tuple(MODEL_REGISTRY)


def get_model_spec(model_key: str) -> ModelSpec:
    key = _alias_map().get(model_key.strip().lower())
    if key is None:
        known = ", ".join(registered_model_keys())
        raise KeyError(f"Unknown model key {model_key!r}. Registered keys: {known}")
    return MODEL_REGISTRY[key]


def find_model_spec(
    model_key: str | None = None, model_path: str | None = None
) -> ModelSpec | None:
    if model_key:
        return get_model_spec(model_key)

    if model_path:
        aliases = _alias_map()
        for candidate in _identifier_candidates(model_path):
            key = aliases.get(candidate)
            if key is not None:
                return MODEL_REGISTRY[key]
        return None

    return MODEL_REGISTRY[DEFAULT_MODEL_KEY]


def model_path_for(model_key: str | None, model_path: str | None) -> str:
    if model_path:
        return model_path
    spec = find_model_spec(model_key=model_key)
    if spec is None:
        raise ValueError("model_path is required when no registered model key is set")
    return spec.hf_id


def slugify_model_id(model_path: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", model_path.lower()).strip("_")
    return slug or "model"


def model_output_root(model_key: str | None, model_path: str | None) -> str:
    spec = find_model_spec(model_key=model_key, model_path=model_path)
    if spec is not None:
        return spec.output_root
    if not model_path:
        return MODEL_REGISTRY[DEFAULT_MODEL_KEY].output_root
    raise ValueError(
        "Cannot infer an output root for unregistered model_path="
        f"{model_path!r}. Pass a registered --model_key or an explicit output "
        "directory for this script."
    )


def default_classifier_path_for(
    model_key: str | None = None, model_path: str | None = None
) -> str | None:
    spec = find_model_spec(model_key=model_key, model_path=model_path)
    return spec.default_classifier_path if spec is not None else None


def tokenizer_kwargs_for(
    model_key: str | None = None, model_path: str | None = None
) -> dict[str, Any]:
    spec = find_model_spec(model_key=model_key, model_path=model_path)
    return spec.tokenizer_kwargs_copy() if spec is not None else {}


def model_metadata(
    model_key: str | None = None, model_path: str | None = None
) -> dict[str, Any] | None:
    spec = find_model_spec(model_key=model_key, model_path=model_path)
    return spec.to_metadata() if spec is not None else None


def registered_model_dimensions(
    model_key: str | None = None, model_path: str | None = None
) -> ModelDimensions | None:
    spec = find_model_spec(model_key=model_key, model_path=model_path)
    return spec.dimensions if spec is not None else None


def dimensions_from_config(config: Any) -> ModelDimensions:
    text_config = getattr(config, "text_config", config)
    return ModelDimensions(
        num_hidden_layers=int(getattr(text_config, "num_hidden_layers")),
        hidden_size=int(getattr(text_config, "hidden_size")),
        intermediate_size=int(getattr(text_config, "intermediate_size")),
        num_attention_heads=int(getattr(text_config, "num_attention_heads")),
        num_key_value_heads=(
            int(getattr(text_config, "num_key_value_heads"))
            if getattr(text_config, "num_key_value_heads", None) is not None
            else None
        ),
        max_position_embeddings=int(getattr(text_config, "max_position_embeddings")),
        torch_dtype=str(getattr(text_config, "torch_dtype", None))
        if getattr(text_config, "torch_dtype", None) is not None
        else None,
    )


def assert_causal_lm_supported(
    model_key: str | None = None, model_path: str | None = None
) -> None:
    spec = find_model_spec(model_key=model_key, model_path=model_path)
    if spec is None or spec.supported and spec.loader_family == "causal_lm":
        return
    raise ValueError(
        f"{spec.hf_id} is registered as loader_family={spec.loader_family!r}, "
        "but this script only supports causal-LM text checkpoints. "
        f"Notes: {spec.notes}"
    )
