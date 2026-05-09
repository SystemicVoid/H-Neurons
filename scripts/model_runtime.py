"""Shared Model Runtime loading, diagnostics, and chat-template helpers."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import os
import re
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from model_registry import (
    assert_causal_lm_supported,
    model_metadata,
    model_path_for,
    tokenizer_kwargs_for,
)
from utils import provenance_error_message


DEFAULT_MODEL_DTYPE = torch.bfloat16
DEFAULT_GENERATION_SMOKE_PROMPT = "Reply with exactly one token: OK"
TARGET_GPU_NAME_RE = re.compile(r"H100|A100", re.IGNORECASE)
TARGET_MIN_MEMORY_BYTES = 75 * 1024**3


@dataclass(frozen=True)
class ModelRuntime:
    """Loaded model/tokenizer pair plus the runtime policy used to load it."""

    model: Any
    tokenizer: Any
    model_key: str | None
    model_path: str
    tokenizer_kwargs: dict[str, Any]
    requested_dtype: Any
    requested_device_map: str
    registered_model: dict[str, Any] | None

    def identity_metadata(self) -> dict[str, Any]:
        return {
            "model_key": self.model_key,
            "model_path": self.model_path,
            "registered_model": self.registered_model,
            "tokenizer_kwargs": dict(self.tokenizer_kwargs),
        }


def resolve_model_path(model_key: str | None, model_path: str | None) -> str:
    return model_path_for(model_key, model_path)


def load_model_runtime(
    model_path: str | None = None,
    *,
    model_key: str | None = None,
    device_map: str = "auto",
    dtype: Any = DEFAULT_MODEL_DTYPE,
) -> ModelRuntime:
    """Load a registered causal-LM text runtime with shared dtype/device policy."""
    resolved_model_path = resolve_model_path(model_key, model_path)
    assert_causal_lm_supported(model_key, resolved_model_path)
    tokenizer_kwargs = tokenizer_kwargs_for(model_key, resolved_model_path)
    tokenizer = AutoTokenizer.from_pretrained(resolved_model_path, **tokenizer_kwargs)
    model = AutoModelForCausalLM.from_pretrained(
        resolved_model_path,
        dtype=dtype,
        device_map=device_map,
    )
    model.eval()
    return ModelRuntime(
        model=model,
        tokenizer=tokenizer,
        model_key=model_key,
        model_path=resolved_model_path,
        tokenizer_kwargs=tokenizer_kwargs,
        requested_dtype=dtype,
        requested_device_map=device_map,
        registered_model=model_metadata(model_key, resolved_model_path),
    )


def load_model_and_tokenizer(
    model_path: str | None = None,
    device_map: str = "auto",
    model_key: str | None = None,
) -> tuple[Any, Any]:
    """Compatibility adapter returning the historical ``(model, tokenizer)`` pair."""
    runtime = load_model_runtime(
        model_path,
        model_key=model_key,
        device_map=device_map,
    )
    return runtime.model, runtime.tokenizer


def unwrap_chat_template_output(out: Any) -> Any:
    """Handle transformers returning either a tensor or a BatchEncoding."""
    if hasattr(out, "input_ids"):
        return out["input_ids"]
    return out


def tokenize_chat(tokenizer: Any, messages: list[dict[str, str]]) -> torch.Tensor:
    """Tokenize a chat message list once, returning a CPU tensor."""
    inputs = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        add_generation_prompt=True,
    )
    return unwrap_chat_template_output(inputs)


def _chat_template_token_ids(
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    add_generation_prompt: bool,
) -> list[int]:
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
    )
    rendered = unwrap_chat_template_output(rendered)
    if isinstance(rendered, torch.Tensor):
        ids = rendered.squeeze(0).tolist()
    else:
        ids = rendered
        if isinstance(ids, list) and ids and isinstance(ids[0], list):
            if len(ids) != 1:
                raise ValueError(
                    "Expected a single chat-template sequence when extracting "
                    "assistant continuation token IDs"
                )
            ids = ids[0]
    return [int(token_id) for token_id in ids]


def _longest_common_prefix_len(a: list[int], b: list[int]) -> int:
    n = 0
    for left, right in zip(a, b, strict=False):
        if left != right:
            break
        n += 1
    return n


def assistant_content_continuation_ids_from_chat_messages(
    tokenizer: Any,
    prompt_messages: list[dict[str, str]],
    assistant_content: str,
    *,
    prompt_input_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """Extract assistant-content continuation IDs under the active chat template."""
    if prompt_input_ids is None:
        prompt_ids = _chat_template_token_ids(
            tokenizer,
            prompt_messages,
            add_generation_prompt=True,
        )
    else:
        prompt_ids = [
            int(token_id) for token_id in prompt_input_ids.squeeze(0).tolist()
        ]

    full_messages = [
        *prompt_messages,
        {"role": "assistant", "content": assistant_content},
    ]
    empty_assistant_messages = [
        *prompt_messages,
        {"role": "assistant", "content": ""},
    ]
    full_ids = _chat_template_token_ids(
        tokenizer,
        full_messages,
        add_generation_prompt=False,
    )
    empty_ids = _chat_template_token_ids(
        tokenizer,
        empty_assistant_messages,
        add_generation_prompt=False,
    )

    prefix_len = _longest_common_prefix_len(prompt_ids, full_ids)
    if prefix_len != len(prompt_ids):
        raise ValueError(
            "Chat-template mismatch: generation prompt is not a full prefix of the "
            "assistant-completed sequence"
        )

    suffix_ids = empty_ids[prefix_len:]
    tail_ids = full_ids[prefix_len:]
    if (
        suffix_ids
        and len(tail_ids) >= len(suffix_ids)
        and tail_ids[-len(suffix_ids) :] == suffix_ids
    ):
        content_tail = tail_ids[: -len(suffix_ids)]
    else:
        content_tail = tail_ids

    if not content_tail:
        raise ValueError(
            "Assistant-content continuation extraction produced no content tokens "
            f"for assistant_content={assistant_content!r}"
        )
    return torch.tensor(content_tail, dtype=torch.long, device="cpu")


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple | set):
        return [_jsonable(item) for item in value]
    return str(value)


def _tensor_descriptor(tensor: Any) -> dict[str, Any]:
    shape = getattr(tensor, "shape", None)
    return {
        "dtype": str(getattr(tensor, "dtype", None)),
        "device": str(getattr(tensor, "device", None)),
        "shape": [int(dim) for dim in shape] if shape is not None else None,
        "numel": int(tensor.numel()) if hasattr(tensor, "numel") else None,
    }


def _counter_to_dict(counter: Counter[str]) -> dict[str, int]:
    return {key: int(counter[key]) for key in sorted(counter)}


def coerce_runtime_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


def _iter_tensors(model: Any, method_name: str) -> list[Any]:
    method = getattr(model, method_name, None)
    if method is None:
        return []
    return list(method())


def _device_counts(runtime: dict[str, Any]) -> dict[str, int]:
    value = runtime.get("device_counts")
    if not isinstance(value, dict):
        return {}
    counts: dict[str, int] = {}
    for key, count in value.items():
        parsed = coerce_runtime_int(count)
        if parsed is not None:
            counts[str(key)] = parsed
    return counts


def loaded_tensors_on_cuda(runtime: dict[str, Any]) -> bool:
    counts = {
        device: count for device, count in _device_counts(runtime).items() if count
    }
    return bool(counts) and all(device.startswith("cuda:") for device in counts)


def loaded_tensors_on_cuda0(runtime: dict[str, Any]) -> bool:
    counts = {
        device: count for device, count in _device_counts(runtime).items() if count
    }
    return (
        set(counts) == {"cuda:0"}
        and counts["cuda:0"] > 0
        and (runtime.get("loaded_primary_device") == "cuda:0")
    )


def _loaded_cuda_device_indexes(runtime: dict[str, Any]) -> set[int]:
    indexes: set[int] = set()
    for device, count in _device_counts(runtime).items():
        if count <= 0:
            continue
        match = re.fullmatch(r"cuda:(\d+)", device)
        if match:
            indexes.add(int(match.group(1)))
    return indexes


def _cuda_devices(cuda_memory: dict[str, Any]) -> list[dict[str, Any]]:
    devices = cuda_memory.get("devices")
    if not isinstance(devices, list):
        return []
    return [device for device in devices if isinstance(device, dict)]


def _cuda_devices_by_index(cuda_memory: dict[str, Any]) -> dict[int, dict[str, Any]]:
    indexed: dict[int, dict[str, Any]] = {}
    for fallback_index, device in enumerate(_cuda_devices(cuda_memory)):
        index = coerce_runtime_int(device.get("index"))
        indexed[fallback_index if index is None else index] = device
    return indexed


def _has_nonzero_cuda_allocation(device: dict[str, Any]) -> bool:
    allocation_keys = (
        "memory_allocated_bytes",
        "memory_reserved_bytes",
        "max_memory_allocated_bytes",
        "max_memory_reserved_bytes",
    )
    for key in allocation_keys:
        value = coerce_runtime_int(device.get(key))
        if value is not None and value > 0:
            return True
    return False


def _is_target_accelerator(device: dict[str, Any]) -> bool:
    name = str(device.get("name", ""))
    name_ok = bool(TARGET_GPU_NAME_RE.search(name))
    total_memory = coerce_runtime_int(device.get("total_memory_bytes"))
    if total_memory is None:
        return name_ok
    return name_ok and total_memory >= TARGET_MIN_MEMORY_BYTES


def _target_accelerator_present(cuda_memory: dict[str, Any]) -> bool:
    return any(_is_target_accelerator(device) for device in _cuda_devices(cuda_memory))


def loaded_cuda_devices_are_target_accelerators(
    runtime: dict[str, Any],
    cuda_memory: dict[str, Any],
) -> bool:
    indexes = _loaded_cuda_device_indexes(runtime)
    devices_by_index = _cuda_devices_by_index(cuda_memory)
    return bool(indexes) and all(
        index in devices_by_index and _is_target_accelerator(devices_by_index[index])
        for index in indexes
    )


def loaded_cuda_devices_have_nonzero_allocation(
    runtime: dict[str, Any],
    cuda_memory: dict[str, Any],
) -> bool:
    indexes = _loaded_cuda_device_indexes(runtime)
    devices_by_index = _cuda_devices_by_index(cuda_memory)
    return bool(indexes) and all(
        index in devices_by_index
        and _has_nonzero_cuda_allocation(devices_by_index[index])
        for index in indexes
    )


def derive_effective_device_map(runtime: dict[str, Any]) -> Any:
    hf_device_map = runtime.get("hf_device_map")
    if hf_device_map not in (None, {}):
        return hf_device_map
    if runtime.get("requested_device_map") == "cuda:0" and loaded_tensors_on_cuda0(
        runtime
    ):
        return {"": "cuda:0"}
    return None


def summarize_model_runtime(
    model: Any,
    *,
    requested_device_map: str,
    requested_dtype: Any = DEFAULT_MODEL_DTYPE,
) -> dict[str, Any]:
    parameters = _iter_tensors(model, "parameters")
    buffers = _iter_tensors(model, "buffers")
    tensor_list = [*parameters, *buffers]
    dtype_counts = Counter(
        str(getattr(tensor, "dtype", None)) for tensor in tensor_list
    )
    device_counts = Counter(
        str(getattr(tensor, "device", None)) for tensor in tensor_list
    )
    first_parameter = parameters[0] if parameters else None
    first_tensor = tensor_list[0] if tensor_list else None
    config = getattr(model, "config", None)
    config_dtype = getattr(config, "torch_dtype", None)

    summary = {
        "requested_dtype": str(requested_dtype),
        "requested_device_map": requested_device_map,
        "loaded_primary_dtype": (
            str(getattr(first_parameter, "dtype", None))
            if first_parameter is not None
            else None
        ),
        "loaded_primary_device": (
            str(getattr(first_parameter, "device", None))
            if first_parameter is not None
            else None
        ),
        "config_torch_dtype": str(config_dtype) if config_dtype is not None else None,
        "parameter_count": len(parameters),
        "buffer_count": len(buffers),
        "total_parameter_elements": int(
            sum(int(parameter.numel()) for parameter in parameters)
        ),
        "dtype_counts": _counter_to_dict(dtype_counts),
        "device_counts": _counter_to_dict(device_counts),
        "first_parameter": (
            _tensor_descriptor(first_parameter) if first_parameter is not None else None
        ),
        "first_tensor": _tensor_descriptor(first_tensor)
        if first_tensor is not None
        else None,
        "hf_device_map": _jsonable(getattr(model, "hf_device_map", None)),
    }
    summary["effective_device_map"] = derive_effective_device_map(summary)
    return summary


def build_no_quantization_check(model: Any) -> dict[str, Any]:
    config = getattr(model, "config", None)
    quantization_config = getattr(config, "quantization_config", None)
    is_loaded_in_4bit = bool(getattr(model, "is_loaded_in_4bit", False))
    is_loaded_in_8bit = bool(getattr(model, "is_loaded_in_8bit", False))
    has_quantization_config = quantization_config is not None
    status = (
        "needs_review"
        if is_loaded_in_4bit or is_loaded_in_8bit or has_quantization_config
        else "ok"
    )
    return {
        "status": status,
        "requested_no_quantization": True,
        "load_kwargs": {
            "load_in_4bit": False,
            "load_in_8bit": False,
            "quantization_config": None,
        },
        "model_is_loaded_in_4bit": is_loaded_in_4bit,
        "model_is_loaded_in_8bit": is_loaded_in_8bit,
        "config_has_quantization_config": has_quantization_config,
        "config_quantization_config": (
            str(quantization_config) if quantization_config is not None else None
        ),
    }


def cuda_memory_summary() -> dict[str, Any]:
    summary: dict[str, Any] = {"available": bool(torch.cuda.is_available())}
    if not torch.cuda.is_available():
        return summary
    try:
        device_count = int(torch.cuda.device_count())
        summary.update(
            {
                "device_count": device_count,
                "current_device": int(torch.cuda.current_device()),
                "bf16_supported": bool(torch.cuda.is_bf16_supported()),
            }
        )
        devices = []
        for index in range(device_count):
            devices.append(
                {
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "total_memory_bytes": int(
                        torch.cuda.get_device_properties(index).total_memory
                    ),
                    "memory_allocated_bytes": int(torch.cuda.memory_allocated(index)),
                    "memory_reserved_bytes": int(torch.cuda.memory_reserved(index)),
                    "max_memory_allocated_bytes": int(
                        torch.cuda.max_memory_allocated(index)
                    ),
                    "max_memory_reserved_bytes": int(
                        torch.cuda.max_memory_reserved(index)
                    ),
                }
            )
        summary["devices"] = devices
    except Exception as exc:  # pragma: no cover - hardware/driver dependent.
        summary["error"] = provenance_error_message(exc)
    return summary


def _first_model_device(model: Any) -> torch.device | None:
    for tensor in [
        *_iter_tensors(model, "parameters"),
        *_iter_tensors(model, "buffers"),
    ]:
        device = getattr(tensor, "device", None)
        if isinstance(device, torch.device) and device.type != "meta":
            return device
    return None


def _move_inputs_to_device(
    inputs: dict[str, Any],
    device: torch.device | None,
) -> dict[str, Any]:
    if device is None:
        return inputs
    return {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }


def _generation_inputs(tokenizer: Any) -> dict[str, Any]:
    messages = [{"role": "user", "content": DEFAULT_GENERATION_SMOKE_PROMPT}]
    if hasattr(tokenizer, "apply_chat_template"):
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        if hasattr(rendered, "items"):
            return {
                str(key): value
                for key, value in rendered.items()
                if hasattr(value, "shape")
            }
        if hasattr(rendered, "input_ids"):
            return {"input_ids": rendered["input_ids"]}
        return {"input_ids": rendered}
    encoded = tokenizer(DEFAULT_GENERATION_SMOKE_PROMPT, return_tensors="pt")
    return {
        str(key): value for key, value in encoded.items() if hasattr(value, "shape")
    }


def _decode_tokens(tokenizer: Any, token_ids: Any) -> str:
    try:
        return str(tokenizer.decode(token_ids, skip_special_tokens=True))
    except TypeError:
        return str(tokenizer.decode(token_ids))


def run_generation_smoke(
    model: Any,
    tokenizer: Any,
    *,
    max_new_tokens: int,
) -> dict[str, Any]:
    if max_new_tokens == 0:
        return {
            "attempted": False,
            "succeeded": None,
            "max_new_tokens": 0,
            "generated_new_tokens": 0,
            "decoded_new_text_preview": None,
        }

    model.eval()
    inputs = _generation_inputs(tokenizer)
    input_ids = inputs["input_ids"]
    input_length = int(input_ids.shape[-1])
    inputs = _move_inputs_to_device(inputs, _first_model_device(model))
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = getattr(tokenizer, "eos_token_id", None)
    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
    }
    if pad_token_id is not None:
        generation_kwargs["pad_token_id"] = pad_token_id

    with torch.inference_mode():
        output_ids = model.generate(**inputs, **generation_kwargs)

    generated_new_tokens = int(output_ids.shape[-1] - input_length)
    new_token_ids = output_ids[0, input_length:].detach().cpu().tolist()
    return {
        "attempted": True,
        "succeeded": True,
        "max_new_tokens": max_new_tokens,
        "generated_new_tokens": generated_new_tokens,
        "decoded_new_text_preview": _decode_tokens(tokenizer, new_token_ids)[:80],
    }


def summarize_chat_template_behavior(
    tokenizer: Any,
    *,
    prompt: str = DEFAULT_GENERATION_SMOKE_PROMPT,
    assistant_content: str = "OK",
) -> dict[str, Any]:
    """Return non-claim-bearing evidence about chat-template continuation behavior."""
    if not hasattr(tokenizer, "apply_chat_template"):
        return {
            "available": False,
            "generation_prompt_supported": False,
            "assistant_continuation_supported": False,
        }

    messages = [{"role": "user", "content": prompt}]
    try:
        user_ids = _chat_template_token_ids(
            tokenizer,
            messages,
            add_generation_prompt=False,
        )
        generation_ids = _chat_template_token_ids(
            tokenizer,
            messages,
            add_generation_prompt=True,
        )
        prompt_input_ids = torch.tensor([generation_ids], dtype=torch.long)
        continuation_ids = assistant_content_continuation_ids_from_chat_messages(
            tokenizer,
            messages,
            assistant_content,
            prompt_input_ids=prompt_input_ids,
        )
        return {
            "available": True,
            "generation_prompt_supported": True,
            "generation_prompt_preserves_user_prefix": (
                _longest_common_prefix_len(user_ids, generation_ids) == len(user_ids)
            ),
            "user_token_count": len(user_ids),
            "generation_prompt_token_count": len(generation_ids),
            "generation_prompt_added_token_count": max(
                0,
                len(generation_ids) - len(user_ids),
            ),
            "assistant_continuation_supported": True,
            "assistant_continuation_token_count": int(continuation_ids.numel()),
        }
    except Exception as exc:
        return {
            "available": True,
            "generation_prompt_supported": False,
            "assistant_continuation_supported": False,
            "error": provenance_error_message(exc),
        }
