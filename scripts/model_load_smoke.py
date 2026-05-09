"""Smoke-test model loading with BF16 and no quantization."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

import torch

sys.path.insert(0, os.path.dirname(__file__))
from model_runtime import (
    DEFAULT_GENERATION_SMOKE_PROMPT,
    build_no_quantization_check,
    coerce_runtime_int,
    cuda_memory_summary,
    derive_effective_device_map,
    load_model_runtime,
    loaded_cuda_devices_are_target_accelerators,
    loaded_cuda_devices_have_nonzero_allocation,
    loaded_tensors_on_cuda,
    loaded_tensors_on_cuda0,
    run_generation_smoke,
    summarize_chat_template_behavior,
    summarize_model_runtime,
)
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
DEFAULT_PROMPT = DEFAULT_GENERATION_SMOKE_PROMPT


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
        "--expected_model_key",
        type=str,
        default=None,
        help="Expected model key when validating an existing summary.",
    )
    parser.add_argument(
        "--expected_model_path",
        type=str,
        default=None,
        help="Expected model path/HF id when validating an existing summary.",
    )
    parser.add_argument(
        "--expected_output_path",
        type=str,
        default=None,
        help="Expected summary output path when validating an existing summary.",
    )
    parser.add_argument(
        "--expected_device_map",
        type=str,
        default=None,
        help="Expected requested device_map when validating an existing summary.",
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


def _registered_model_key(metadata: Any) -> str | None:
    if isinstance(metadata, dict) and isinstance(metadata.get("key"), str):
        return metadata["key"]
    return None


def _registered_model_hf_id(metadata: Any) -> str | None:
    if isinstance(metadata, dict) and isinstance(metadata.get("hf_id"), str):
        return metadata["hf_id"]
    return None


def validate_cp1_smoke_summary(
    summary: dict[str, Any],
    *,
    expected_model_key: str | None = None,
    expected_model_path: str | None = None,
    expected_output_path: str | None = None,
    expected_device_map: str | None = None,
) -> dict[str, Any]:
    runtime = _runtime(summary)
    cuda_memory = _cuda_memory(summary)
    quantization = _quantization(summary)
    generation = _generation(summary)
    requested_device_map = runtime.get("requested_device_map")
    hf_device_map = runtime.get("hf_device_map")
    effective_device_map = derive_effective_device_map(runtime)
    expected_metadata = (
        model_metadata(expected_model_key, expected_model_path)
        if expected_model_key is not None or expected_model_path is not None
        else None
    )
    expected_tokenizer_kwargs = (
        tokenizer_kwargs_for(expected_model_key, expected_model_path)
        if expected_model_key is not None or expected_model_path is not None
        else None
    )
    registered_model = summary.get("registered_model")

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
        and (coerce_runtime_int(generation.get("generated_new_tokens")) or 0) >= 1,
        "auto_load_has_hf_device_map": requested_device_map != "auto"
        or hf_device_map not in (None, {}),
        "device_map_evidence": effective_device_map is not None,
        "loaded_tensors_on_cuda": loaded_tensors_on_cuda(runtime),
        "explicit_null_map_loaded_on_cuda0": hf_device_map not in (None, {})
        or (requested_device_map == "cuda:0" and loaded_tensors_on_cuda0(runtime)),
        "cuda_available": cuda_memory.get("available") is True,
        "cuda_bf16_supported": cuda_memory.get("bf16_supported") is True,
        "target_cuda_hardware": loaded_cuda_devices_are_target_accelerators(
            runtime,
            cuda_memory,
        ),
        "nonzero_cuda_allocation": loaded_cuda_devices_have_nonzero_allocation(
            runtime,
            cuda_memory,
        ),
    }
    if expected_model_key is not None:
        checks["expected_model_key"] = summary.get("model_key") == expected_model_key
        if expected_metadata is not None:
            checks["expected_registered_model_key"] = (
                _registered_model_key(registered_model) == expected_metadata["key"]
            )
    if expected_model_path is not None:
        checks["expected_model_path"] = summary.get("model_path") == expected_model_path
        if expected_metadata is not None:
            checks["expected_registered_model_hf_id"] = (
                _registered_model_hf_id(registered_model) == expected_metadata["hf_id"]
            )
    if expected_output_path is not None:
        checks["expected_output_path"] = (
            summary.get("output_path") == expected_output_path
        )
    if expected_device_map is not None:
        checks["expected_device_map"] = requested_device_map == expected_device_map
    if expected_tokenizer_kwargs is not None:
        checks["expected_tokenizer_kwargs"] = (
            summary.get("tokenizer_kwargs") == expected_tokenizer_kwargs
        )
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
        "target_cuda_hardware": (
            "loaded CUDA devices must be H100/A100-class with >=75 GiB when known"
        ),
        "nonzero_cuda_allocation": (
            "loaded CUDA devices must report nonzero allocation/reservation"
        ),
        "expected_model_key": "summary model_key does not match expected model key",
        "expected_registered_model_key": (
            "registered model key does not match expected model key"
        ),
        "expected_model_path": "summary model_path does not match expected model path",
        "expected_registered_model_hf_id": (
            "registered model HF id does not match expected model path"
        ),
        "expected_output_path": "summary output_path does not match expected path",
        "expected_device_map": "requested device_map does not match expected value",
        "expected_tokenizer_kwargs": (
            "summary tokenizer kwargs do not match expected model registry kwargs"
        ),
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


def validate_cp1_smoke_summary_file(
    path: Path,
    *,
    expected_model_key: str | None = None,
    expected_model_path: str | None = None,
    expected_output_path: str | None = None,
    expected_device_map: str | None = None,
) -> dict[str, Any]:
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
    return validate_cp1_smoke_summary(
        payload,
        expected_model_key=expected_model_key,
        expected_model_path=expected_model_path,
        expected_output_path=expected_output_path,
        expected_device_map=expected_device_map,
    )


def build_smoke_summary(
    args: argparse.Namespace,
    *,
    runtime: dict[str, Any],
    quantization: dict[str, Any],
    cuda_memory: dict[str, Any],
    generation: dict[str, Any],
    chat_template: dict[str, Any] | None = None,
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
        "chat_template": chat_template or {},
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.validate_summary is not None:
        validation = validate_cp1_smoke_summary_file(
            args.validate_summary,
            expected_model_key=args.expected_model_key,
            expected_model_path=args.expected_model_path,
            expected_output_path=args.expected_output_path,
            expected_device_map=args.expected_device_map,
        )
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
        model_runtime = load_model_runtime(
            args.model_path,
            model_key=args.model_key,
            device_map=args.device_map,
        )
        generation = run_generation_smoke(
            model_runtime.model,
            model_runtime.tokenizer,
            max_new_tokens=args.max_new_tokens,
        )
        summary = build_smoke_summary(
            args,
            runtime=summarize_model_runtime(
                model_runtime.model,
                requested_device_map=args.device_map,
                requested_dtype=model_runtime.requested_dtype,
            ),
            quantization=build_no_quantization_check(model_runtime.model),
            cuda_memory=cuda_memory_summary(),
            generation=generation,
            chat_template=summarize_chat_template_behavior(model_runtime.tokenizer),
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
