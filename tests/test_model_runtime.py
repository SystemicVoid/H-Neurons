from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import model_runtime  # noqa: E402


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 9

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool = True,
        add_generation_prompt: bool = False,
        return_tensors: str | None = None,
        **_kwargs: Any,
    ) -> Any:
        ids = [10, 11]
        if messages and messages[-1]["role"] == "assistant":
            ids.append(12)
            if messages[-1]["content"]:
                ids.append(42)
            ids.append(13)
        elif add_generation_prompt:
            ids.append(12)

        if return_tensors == "pt":
            return torch.tensor([ids], dtype=torch.long)
        if tokenize:
            return ids
        return " ".join(str(token_id) for token_id in ids)

    def decode(self, token_ids: Any, *, skip_special_tokens: bool = True) -> str:
        return "OK" if list(token_ids) == [42] else str(list(token_ids))


class FakeModel:
    def __init__(self) -> None:
        self.config = SimpleNamespace(
            torch_dtype=torch.bfloat16,
            quantization_config=None,
        )
        self.hf_device_map = {"": "cuda:0"}
        self.is_loaded_in_4bit = False
        self.is_loaded_in_8bit = False
        self.eval_called = False
        self._parameter = torch.zeros(2, 3, dtype=torch.bfloat16)

    def eval(self) -> FakeModel:
        self.eval_called = True
        return self

    def parameters(self):
        return iter([self._parameter])

    def buffers(self):
        return iter([torch.zeros(1, dtype=torch.float32)])

    def generate(self, input_ids: torch.Tensor, **_kwargs: Any) -> torch.Tensor:
        new_token = torch.tensor([[42]], dtype=torch.long, device=input_ids.device)
        return torch.cat([input_ids, new_token], dim=-1)


def test_load_model_runtime_uses_registry_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, Any] = {}
    fake_model = FakeModel()
    fake_tokenizer = FakeTokenizer()

    def fake_tokenizer_from_pretrained(model_path: str, **kwargs: Any) -> FakeTokenizer:
        calls["tokenizer"] = {"model_path": model_path, "kwargs": kwargs}
        return fake_tokenizer

    def fake_model_from_pretrained(model_path: str, **kwargs: Any) -> FakeModel:
        calls["model"] = {"model_path": model_path, "kwargs": kwargs}
        return fake_model

    monkeypatch.setattr(
        model_runtime.AutoTokenizer,
        "from_pretrained",
        fake_tokenizer_from_pretrained,
    )
    monkeypatch.setattr(
        model_runtime.AutoModelForCausalLM,
        "from_pretrained",
        fake_model_from_pretrained,
    )

    runtime = model_runtime.load_model_runtime(
        model_key="mistral24b",
        device_map="cuda:0",
    )

    assert runtime.model is fake_model
    assert runtime.tokenizer is fake_tokenizer
    assert runtime.model_path == "mistralai/Mistral-Small-24B-Instruct-2501"
    assert runtime.tokenizer_kwargs == {"fix_mistral_regex": True}
    assert runtime.requested_dtype == torch.bfloat16
    assert runtime.requested_device_map == "cuda:0"
    assert runtime.registered_model is not None
    assert runtime.registered_model["key"] == "mistral_small_24b_instruct_2501"
    assert fake_model.eval_called is True
    assert calls["tokenizer"] == {
        "model_path": "mistralai/Mistral-Small-24B-Instruct-2501",
        "kwargs": {"fix_mistral_regex": True},
    }
    assert calls["model"] == {
        "model_path": "mistralai/Mistral-Small-24B-Instruct-2501",
        "kwargs": {"dtype": torch.bfloat16, "device_map": "cuda:0"},
    }


def test_load_model_runtime_rejects_unsupported_loader_before_hf_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        model_runtime.AutoTokenizer,
        "from_pretrained",
        lambda *_args, **_kwargs: pytest.fail("tokenizer load should not run"),
    )
    monkeypatch.setattr(
        model_runtime.AutoModelForCausalLM,
        "from_pretrained",
        lambda *_args, **_kwargs: pytest.fail("model load should not run"),
    )

    with pytest.raises(ValueError, match="loader_family='mistral3_multimodal'"):
        model_runtime.load_model_runtime(model_key="mistral24b_2503")


def test_runtime_summary_records_dtype_and_device_map() -> None:
    summary = model_runtime.summarize_model_runtime(
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


def test_chat_template_helpers_extract_assistant_content_continuation() -> None:
    tokenizer = FakeTokenizer()
    messages = [{"role": "user", "content": "question"}]

    continuation = model_runtime.assistant_content_continuation_ids_from_chat_messages(
        tokenizer,
        messages,
        "OK",
    )
    summary = model_runtime.summarize_chat_template_behavior(tokenizer)

    assert continuation.tolist() == [42]
    assert summary["available"] is True
    assert summary["generation_prompt_supported"] is True
    assert summary["generation_prompt_preserves_user_prefix"] is True
    assert summary["assistant_continuation_supported"] is True
    assert summary["assistant_continuation_token_count"] == 1


def test_generation_smoke_uses_chat_template_and_decodes_new_token() -> None:
    result = model_runtime.run_generation_smoke(
        FakeModel(),
        FakeTokenizer(),
        max_new_tokens=1,
    )

    assert result == {
        "attempted": True,
        "succeeded": True,
        "max_new_tokens": 1,
        "generated_new_tokens": 1,
        "decoded_new_text_preview": "OK",
    }
