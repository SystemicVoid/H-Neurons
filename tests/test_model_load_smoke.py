from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import model_load_smoke  # noqa: E402


class FakeModel:
    def __init__(
        self,
        *,
        parameter: torch.Tensor | None = None,
        quantization_config: object | None = None,
        is_loaded_in_4bit: bool = False,
        is_loaded_in_8bit: bool = False,
    ) -> None:
        self._parameter = (
            torch.zeros(2, 3, dtype=torch.bfloat16) if parameter is None else parameter
        )
        self.config = SimpleNamespace(
            torch_dtype=torch.bfloat16,
            quantization_config=quantization_config,
        )
        self.hf_device_map = {"": "cuda:0"}
        self.is_loaded_in_4bit = is_loaded_in_4bit
        self.is_loaded_in_8bit = is_loaded_in_8bit

    def parameters(self):
        return iter([self._parameter])

    def buffers(self):
        return iter([torch.zeros(1, dtype=torch.float32)])


def test_parse_args_resolves_mistral_alias() -> None:
    args = model_load_smoke.parse_args(
        [
            "--model_key",
            "mistral24b",
            "--output_path",
            "data/mistral24b/preflight/model_load_smoke.json",
        ]
    )

    assert args.model_path == "mistralai/Mistral-Small-24B-Instruct-2501"
    assert args.max_new_tokens == 1


def test_runtime_summary_records_bf16_dtype_and_device_map() -> None:
    summary = model_load_smoke.summarize_model_runtime(
        FakeModel(),
        requested_device_map="cuda:0",
    )

    assert summary["requested_dtype"] == "torch.bfloat16"
    assert summary["requested_device_map"] == "cuda:0"
    assert summary["loaded_primary_dtype"] == "torch.bfloat16"
    assert summary["first_parameter"]["shape"] == [2, 3]
    assert summary["total_parameter_elements"] == 6
    assert summary["hf_device_map"] == {"": "cuda:0"}
    assert summary["effective_device_map"] == {"": "cuda:0"}


def test_no_quantization_check_flags_quantized_models() -> None:
    clean = model_load_smoke.build_no_quantization_check(FakeModel())
    quantized = model_load_smoke.build_no_quantization_check(
        FakeModel(quantization_config={"load_in_4bit": True}, is_loaded_in_4bit=True)
    )

    assert clean["status"] == "ok"
    assert clean["requested_no_quantization"] is True
    assert clean["load_kwargs"]["load_in_4bit"] is False
    assert quantized["status"] == "needs_review"
    assert quantized["model_is_loaded_in_4bit"] is True
    assert quantized["config_has_quantization_config"] is True


def test_smoke_summary_is_non_claim_bearing_and_uses_registry_metadata(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "model_load_smoke.json"
    args = argparse.Namespace(
        model_key="mistral24b",
        model_path="mistralai/Mistral-Small-24B-Instruct-2501",
        output_path=output_path,
    )
    summary = model_load_smoke.build_smoke_summary(
        args,
        runtime={
            "requested_dtype": "torch.bfloat16",
            "loaded_primary_dtype": "torch.bfloat16",
        },
        quantization={"status": "ok"},
        cuda_memory={"available": False},
        generation={"attempted": True, "succeeded": True},
    )

    assert summary["schema_version"] == "model_load_smoke/v1"
    assert summary["status"] == "ok"
    assert summary["model_loaded"] is True
    assert summary["gpu_required"] is True
    assert summary["api_required"] is False
    assert summary["tokenizer_kwargs"] == {"fix_mistral_regex": True}
    assert summary["registered_model"]["dimensions"]["total_ffn_neurons"] == 1310720


def _valid_cp1_summary() -> dict:
    return {
        "schema_version": "model_load_smoke/v1",
        "status": "ok",
        "model_loaded": True,
        "runtime": {
            "requested_dtype": "torch.bfloat16",
            "requested_device_map": "cuda:0",
            "loaded_primary_dtype": "torch.bfloat16",
            "loaded_primary_device": "cuda:0",
            "config_torch_dtype": "torch.bfloat16",
            "device_counts": {"cuda:0": 365},
            "hf_device_map": None,
        },
        "no_quantization": {
            "status": "ok",
            "model_is_loaded_in_4bit": False,
            "model_is_loaded_in_8bit": False,
            "config_has_quantization_config": False,
        },
        "cuda_memory": {
            "available": True,
            "bf16_supported": True,
            "devices": [
                {
                    "index": 0,
                    "name": "NVIDIA H100 80GB HBM3",
                    "total_memory_bytes": 80 * 1024**3,
                    "memory_allocated_bytes": 47 * 1024**3,
                    "memory_reserved_bytes": 47 * 1024**3,
                    "max_memory_allocated_bytes": 47 * 1024**3,
                    "max_memory_reserved_bytes": 47 * 1024**3,
                }
            ],
        },
        "generation": {
            "attempted": True,
            "succeeded": True,
            "generated_new_tokens": 1,
        },
    }


def test_existing_h100_cp1_artifact_validates() -> None:
    summary_path = (
        Path(__file__).resolve().parent.parent
        / "data/mistral24b/preflight/model_load_smoke.json"
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    result = model_load_smoke.validate_cp1_smoke_summary(summary)

    assert result["accepted"] is True
    assert result["effective_device_map"] == {"": "cuda:0"}


def test_cp1_validator_accepts_explicit_cuda0_null_hf_device_map() -> None:
    result = model_load_smoke.validate_cp1_smoke_summary(_valid_cp1_summary())

    assert result["accepted"] is True
    assert result["checks"]["explicit_null_map_loaded_on_cuda0"] is True


def test_cp1_validator_rejects_auto_load_with_null_hf_device_map() -> None:
    summary = _valid_cp1_summary()
    summary["runtime"]["requested_device_map"] = "auto"

    result = model_load_smoke.validate_cp1_smoke_summary(summary)

    assert result["accepted"] is False
    assert result["checks"]["auto_load_has_hf_device_map"] is False


def test_cp1_validator_rejects_wrong_dtype() -> None:
    summary = _valid_cp1_summary()
    summary["runtime"]["loaded_primary_dtype"] = "torch.float16"

    result = model_load_smoke.validate_cp1_smoke_summary(summary)

    assert result["accepted"] is False
    assert result["checks"]["bf16_dtype_loaded"] is False


def test_cp1_validator_rejects_quantized_load() -> None:
    summary = _valid_cp1_summary()
    summary["no_quantization"]["status"] = "needs_review"
    summary["no_quantization"]["model_is_loaded_in_4bit"] = True

    result = model_load_smoke.validate_cp1_smoke_summary(summary)

    assert result["accepted"] is False
    assert result["checks"]["no_quantization"] is False


def test_cp1_validator_rejects_failed_generation() -> None:
    summary = _valid_cp1_summary()
    summary["generation"]["succeeded"] = False

    result = model_load_smoke.validate_cp1_smoke_summary(summary)

    assert result["accepted"] is False
    assert result["checks"]["generation_succeeded"] is False


def test_cp1_validator_rejects_missing_cuda_allocation() -> None:
    summary = _valid_cp1_summary()
    for key in [
        "memory_allocated_bytes",
        "memory_reserved_bytes",
        "max_memory_allocated_bytes",
        "max_memory_reserved_bytes",
    ]:
        summary["cuda_memory"]["devices"][0][key] = 0

    result = model_load_smoke.validate_cp1_smoke_summary(summary)

    assert result["accepted"] is False
    assert result["checks"]["nonzero_cuda_allocation"] is False


def test_cp1_validator_rejects_local_16gb_cuda_gpu() -> None:
    summary = _valid_cp1_summary()
    device = summary["cuda_memory"]["devices"][0]
    device["name"] = "NVIDIA GeForce RTX 5060 Ti"
    device["total_memory_bytes"] = 16 * 1024**3

    result = model_load_smoke.validate_cp1_smoke_summary(summary)

    assert result["accepted"] is False
    assert result["checks"]["target_cuda_hardware"] is False


def test_cp1_validator_rejects_cpu_load() -> None:
    summary = copy.deepcopy(_valid_cp1_summary())
    summary["runtime"]["loaded_primary_device"] = "cpu"
    summary["runtime"]["device_counts"] = {"cpu": 365}

    result = model_load_smoke.validate_cp1_smoke_summary(summary)

    assert result["accepted"] is False
    assert result["checks"]["loaded_tensors_on_cuda"] is False
