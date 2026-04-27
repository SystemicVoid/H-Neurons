from __future__ import annotations

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from run_intervention import _faitheval_prompt  # noqa: E402
from run_negative_control import parse_args  # noqa: E402


_FAKE_SAMPLE = {
    "id": "fake_0",
    "context": "Paris is the capital of France.",
    "question": "What is the capital of France?",
    "choices_text": "A. Paris\nB. London\nC. Madrid",
    "valid_letters": ["A", "B", "C"],
    "counterfactual_key": "B",
}


def _set_argv(monkeypatch: pytest.MonkeyPatch, *extra: str) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_negative_control.py",
            "--benchmark",
            "faitheval",
            "--model_key",
            "mistral24b",
            *extra,
        ],
    )


def test_parse_args_prompt_style_default_is_anti_compliance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_argv(monkeypatch)
    args = parse_args()
    assert args.prompt_style == "anti_compliance"


def test_parse_args_prompt_style_accepts_standard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_argv(monkeypatch, "--prompt_style", "standard")
    args = parse_args()
    assert args.prompt_style == "standard"


def test_parse_args_prompt_style_rejects_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_argv(monkeypatch, "--prompt_style", "made_up")
    with pytest.raises(SystemExit):
        parse_args()


def test_faitheval_prompt_standard_differs_from_anti_compliance() -> None:
    standard = _faitheval_prompt(_FAKE_SAMPLE, "standard")
    anti = _faitheval_prompt(_FAKE_SAMPLE, "anti_compliance")
    assert standard != anti
    assert "expert in retrieval question answering" in standard
    assert "answer based on your own knowledge" in anti
