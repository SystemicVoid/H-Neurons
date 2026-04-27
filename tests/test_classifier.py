from __future__ import annotations

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from classifier import parse_args  # noqa: E402


def _set_argv(monkeypatch: pytest.MonkeyPatch, *extra: str) -> None:
    monkeypatch.setattr(sys, "argv", ["classifier.py", *extra])


def test_save_model_falls_back_to_registry_default_when_model_key_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_argv(monkeypatch, "--model_key", "mistral24b")
    args = parse_args()
    assert args.save_model == "models/mistral24b_classifier.pkl"


def test_save_model_legacy_default_when_no_registry_match(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_argv(
        monkeypatch,
        "--model_path",
        "/workspace/models/some-unregistered-model",
    )
    args = parse_args()
    assert args.save_model == "models/detector.pkl"


def test_save_model_explicit_override_wins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_argv(
        monkeypatch,
        "--model_key",
        "mistral24b",
        "--save_model",
        "custom/path.pkl",
    )
    args = parse_args()
    assert args.save_model == "custom/path.pkl"


def test_load_model_keeps_save_model_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_argv(
        monkeypatch,
        "--model_key",
        "mistral24b",
        "--load_model",
        "models/mistral24b_classifier.pkl",
    )
    args = parse_args()
    assert args.save_model is None


def test_save_model_uses_registry_default_for_gemma3_4b(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_argv(monkeypatch, "--model_key", "gemma3_4b_it")
    args = parse_args()
    assert args.save_model == "models/gemma3_4b_classifier.pkl"
