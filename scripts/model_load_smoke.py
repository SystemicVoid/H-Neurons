"""Smoke-test model loading with BF16 and no quantization."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import re
import sys
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(__file__))
from model_registry import (
    assert_causal_lm_supported,
    model_metadata,
    model_path_for,
    tokenizer_kwargs_for,
)
from utils import (
    finish_run_provenance,
    json_dumps,
    provenance_error_message,
    provenance_status_for_exception,
    start_run_provenance,
)


SCHEMA_VERSION = "model_load_smoke/v1"
VALIDATION_SCHEMA_VERSION = "model_load_smoke_cp1_validation/v1"
DEFAULT_PROMPT = "Reply with exactly one token: OK"
TARGET_GPU_NAME_RE = re.compile(r"H100|A100", re.IGNORECASE)
TARGET_MIN_MEMORY_BYTES = 75 * 1024**3


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load a registered causal-LM checkpoint in BF16 without quantization "
            "and write a non-claim-bearing smoke artifact."
        )
    )
    parser.add_argument(
        "--model_key",
        type=str,
        default=os.environ.get("HNEURONS_MODEL_KEY"),
        help="Registered model key. Used for defaults and tokenizer quirks.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=os.environ.get("HNEURONS_MODEL_PATH"),
        help="Path or HF id for the model.",
    )
    parser.add_argument(
        "--device_map",
        type=str,
        default=os.environ.get("HNEURONS_DEVICE_MAP", "auto"),
        help="Hugging Face device_map value, e.g. 'auto' or 'cuda:0'.",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        default=None,
        help="Path for the JSON smoke summary.",
    )
    parser.add_argument(
        "--validate_summary",
        type=Path,
        default=None,
        help="Validate an existing model_load_smoke/v1 summary and exit.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=1,
        help="Tiny deterministic generation length; set 0 to skip generation.",
    )
    args = parser.parse_args(argv)
    if args.validate_summary is None and args.output_path is None:
        parser.error("--output_path is required unless --validate_summary is used")
    if args.validate_summary is None:
        args.model_path = model_path_for(args.model_key, args.model_path)
        assert_causal_lm_supported(args.model_key, args.model_path)
    if args.max_new_tokens < 0:
        raise ValueError("--max_new_tokens must be non-negative")
    return args


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


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


def _runtime(summary: dict[str, Any]) -> dict[str, Any]:
    value = summary.get("runtime")
    return value if isinstance(value, dict) else {}


def _cuda_memory(summary: dict[str, Any]) -> dict[str, Any]:
    value = summary.get("cuda_memory")
    return value if isinstance(value, dict) else {}


def _generation(summary: dict[str, Any]) -> dict[str, Any]:
    value = summary.get("generation")
    return value if isinstance(value, dict) else {}


def _quantization(summary: dict[str, Any]) -> dict[str, Any]:
    value = summary.get("no_quantization")
    return value if isinstance(value, dict) else {}


def _device_counts(runtime: dict[str, Any]) -> dict[str, int]:
    value = runtime.get("device_counts")
    if not isinstance(value, dict):
        return {}
    counts: dict[str, int] = {}
    for key, count in value.items():
        parsed = _as_int(count)
        if parsed is not None:
            counts[str(key)] = parsed
    return counts


def _loaded_tensors_on_cuda(runtime: dict[str, Any]) -> bool:
    counts = {
        device: count for device, count in _device_counts(runtime).items() if count
    }
    return bool(counts) and all(device.startswith("cuda:") for device in counts)


def _loaded_tensors_on_cuda0(runtime: dict[str, Any]) -> bool:
    counts = {
        device: count for device, count in _device_counts(runtime).items() if count
    }
    return (
        set(counts) == {"cuda:0"}
        and counts["cuda:0"] > 0
        and (runtime.get("loaded_primary_device") == "cuda:0")
    )


def _cuda_devices(cuda_memory: dict[str, Any]) -> list[dict[str, Any]]:
    devices = cuda_memory.get("devices")
    if not isinstance(devices, list):
        return []
    return [device for device in devices if isinstance(device, dict)]


def _has_nonzero_cuda_allocation(cuda_memory: dict[str, Any]) -> bool:
    allocation_keys = (
        "memory_allocated_bytes",
        "memory_reserved_bytes",
        "max_memory_allocated_bytes",
        "max_memory_reserved_bytes",
    )
    for device in _cuda_devices(cuda_memory):
        for key in allocation_keys:
            value = _as_int(device.get(key))
            if value is not None and value > 0:
                return True
    return False


def _is_target_accelerator(device: dict[str, Any]) -> bool:
    name = str(device.get("name", ""))
    name_ok = bool(TARGET_GPU_NAME_RE.search(name))
    total_memory = _as_int(device.get("total_memory_bytes"))
    if total_memory is None:
        return name_ok
    return name_ok and total_memory >= TARGET_MIN_MEMORY_BYTES


def _target_accelerator_present(cuda_memory: dict[str, Any]) -> bool:
    return any(_is_target_accelerator(device) for device in _cuda_devices(cuda_memory))


def derive_effective_device_map(runtime: dict[str, Any]) -> Any:
    hf_device_map = runtime.get("hf_device_map")
    if hf_device_map not in (None, {}):
        return hf_device_map
    if runtime.get("requested_device_map") == "cuda:0" and _loaded_tensors_on_cuda0(
        runtime
    ):
        return {"": "cuda:0"}
    return None


def _iter_tensors(model: Any, method_name: str) -> list[Any]:
    method = getattr(model, method_name, None)
    if method is None:
        return []
    return list(method())


def summarize_model_runtime(model: Any, *, requested_device_map: str) -> dict[str, Any]:
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
        "requested_dtype": str(torch.bfloat16),
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


def validate_cp1_smoke_summary(summary: dict[str, Any]) -> dict[str, Any]:
    runtime = _runtime(summary)
    cuda_memory = _cuda_memory(summary)
    quantization = _quantization(summary)
    generation = _generation(summary)
    requested_device_map = runtime.get("requested_device_map")
    hf_device_map = runtime.get("hf_device_map")
    effective_device_map = runtime.get("effective_device_map")
    if effective_device_map is None:
        effective_device_map = derive_effective_device_map(runtime)

    checks = {
        "schema_version": summary.get("schema_version") == SCHEMA_VERSION,
        "summary_status_ok": summary.get("status") == "ok",
        "model_loaded": summary.get("model_loaded") is True,
        "bf16_dtype_loaded": runtime.get("requested_dtype") == str(torch.bfloat16)
        and runtime.get("loaded_primary_dtype") == str(torch.bfloat16)
        and runtime.get("config_torch_dtype") in (None, str(torch.bfloat16)),
        "no_quantization": quantization.get("status") == "ok"
        and quantization.get("model_is_loaded_in_4bit") is False
        and quantization.get("model_is_loaded_in_8bit") is False
        and quantization.get("config_has_quantization_config") is False,
        "generation_succeeded": generation.get("attempted") is True
        and generation.get("succeeded") is True
        and (_as_int(generation.get("generated_new_tokens")) or 0) >= 1,
        "auto_load_has_hf_device_map": requested_device_map != "auto"
        or hf_device_map is not None,
        "device_map_evidence": effective_device_map is not None,
        "loaded_tensors_on_cuda": _loaded_tensors_on_cuda(runtime),
        "explicit_null_map_loaded_on_cuda0": hf_device_map is not None
        or (requested_device_map == "cuda:0" and _loaded_tensors_on_cuda0(runtime)),
        "cuda_available": cuda_memory.get("available") is True,
        "cuda_bf16_supported": cuda_memory.get("bf16_supported") is True,
        "target_cuda_hardware": _target_accelerator_present(cuda_memory),
        "nonzero_cuda_allocation": _has_nonzero_cuda_allocation(cuda_memory),
    }
    messages = {
        "schema_version": f"schema_version must be {SCHEMA_VERSION}",
        "summary_status_ok": "summary status must be ok",
        "model_loaded": "model_loaded must be true",
        "bf16_dtype_loaded": "requested/config/loaded dtype must be BF16",
        "no_quantization": "4-bit/8-bit/config quantization must be absent",
        "generation_succeeded": "one-token generation smoke must succeed",
        "auto_load_has_hf_device_map": "device_map=auto must report hf_device_map",
        "device_map_evidence": "missing hf_device_map or explicit cuda:0 evidence",
        "loaded_tensors_on_cuda": "loaded tensors must be on CUDA devices",
        "explicit_null_map_loaded_on_cuda0": (
            "null hf_device_map is accepted only for explicit cuda:0 loads"
        ),
        "cuda_available": "CUDA must be available",
        "cuda_bf16_supported": "CUDA BF16 support must be reported",
        "target_cuda_hardware": "GPU must be H100/A100-class with >=75 GiB when known",
        "nonzero_cuda_allocation": "CUDA allocation/reservation must be nonzero",
    }
    reasons = [messages[key] for key, passed in checks.items() if not passed]
    accepted = not reasons
    return {
        "schema_version": VALIDATION_SCHEMA_VERSION,
        "status": "ok" if accepted else "needs_review",
        "accepted": accepted,
        "checks": checks,
        "reasons": reasons,
        "effective_device_map": effective_device_map,
    }


def validate_cp1_smoke_summary_file(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return {
            "schema_version": VALIDATION_SCHEMA_VERSION,
            "status": "needs_review",
            "accepted": False,
            "checks": {},
            "reasons": [f"{path} must contain a JSON object"],
            "effective_device_map": None,
        }
    return validate_cp1_smoke_summary(payload)


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
    inputs: dict[str, Any], device: torch.device | None
) -> dict[str, Any]:
    if device is None:
        return inputs
    return {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }


def _generation_inputs(tokenizer: Any) -> dict[str, Any]:
    messages = [{"role": "user", "content": DEFAULT_PROMPT}]
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
    encoded = tokenizer(DEFAULT_PROMPT, return_tensors="pt")
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


def build_smoke_summary(
    args: argparse.Namespace,
    *,
    runtime: dict[str, Any],
    quantization: dict[str, Any],
    cuda_memory: dict[str, Any],
    generation: dict[str, Any],
) -> dict[str, Any]:
    bf16_dtype_loaded = runtime.get("loaded_primary_dtype") == str(torch.bfloat16)
    generation_ok = (
        generation.get("attempted") is False or generation.get("succeeded") is True
    )
    checks = {
        "bf16_dtype_loaded": bf16_dtype_loaded,
        "no_quantization": quantization.get("status") == "ok",
        "generation_smoke": generation_ok,
    }
    status = "ok" if all(checks.values()) else "needs_review"
    return {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "model_key": args.model_key,
        "model_path": args.model_path,
        "registered_model": model_metadata(args.model_key, args.model_path),
        "tokenizer_kwargs": tokenizer_kwargs_for(args.model_key, args.model_path),
        "output_path": str(args.output_path),
        "model_loaded": True,
        "gpu_required": True,
        "api_required": False,
        "checks": checks,
        "runtime": runtime,
        "no_quantization": quantization,
        "cuda_memory": cuda_memory,
        "generation": generation,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.validate_summary is not None:
        validation = validate_cp1_smoke_summary_file(args.validate_summary)
        print(json_dumps(validation))
        return 0 if validation["accepted"] else 1

    if args.output_path is None:
        raise RuntimeError("--output_path is required outside validation mode")
    provenance_handle = start_run_provenance(
        args,
        primary_target=args.output_path,
        output_targets=[args.output_path],
        extra={
            "model_key": args.model_key,
            "model_path": args.model_path,
            "gpu_required": True,
            "api_required": False,
        },
    )
    provenance_status = "completed"
    provenance_extra: dict[str, Any] = {}
    try:
        tokenizer_kwargs = tokenizer_kwargs_for(args.model_key, args.model_path)
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, **tokenizer_kwargs)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            dtype=torch.bfloat16,
            device_map=args.device_map,
        )
        generation = run_generation_smoke(
            model,
            tokenizer,
            max_new_tokens=args.max_new_tokens,
        )
        summary = build_smoke_summary(
            args,
            runtime=summarize_model_runtime(
                model,
                requested_device_map=args.device_map,
            ),
            quantization=build_no_quantization_check(model),
            cuda_memory=cuda_memory_summary(),
            generation=generation,
        )
        args.output_path.parent.mkdir(parents=True, exist_ok=True)
        args.output_path.write_text(json_dumps(summary), encoding="utf-8")
        provenance_extra["summary"] = {
            "status": summary["status"],
            "model_loaded": True,
            "gpu_required": True,
            "api_required": False,
            "checks": summary["checks"],
        }
    except BaseException as exc:
        provenance_status = provenance_status_for_exception(exc)
        provenance_extra["error"] = provenance_error_message(exc)
        raise
    finally:
        finish_run_provenance(provenance_handle, provenance_status, provenance_extra)
    return 0


if __name__ == "__main__":
    sys.exit(main())
