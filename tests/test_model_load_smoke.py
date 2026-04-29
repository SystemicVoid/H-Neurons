from __future__ import annotations

import argparse
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
