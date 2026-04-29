from __future__ import annotations

import argparse
from pathlib import Path
import sys
import warnings

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from classifier import fit_model, parse_args  # noqa: E402


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


def test_save_model_legacy_default_when_no_model_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("HNEURONS_MODEL_KEY", raising=False)
    monkeypatch.delenv("HNEURONS_MODEL_PATH", raising=False)
    _set_argv(monkeypatch)

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


def test_l1_liblinear_uses_l1_ratio_without_penalty_deprecation() -> None:
    args = argparse.Namespace(penalty="l1", solver="liblinear")
    X = np.array(
        [
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.1, 0.0, 0.8],
            [0.9, 1.0, 0.1],
        ]
    )
    y = np.array([0, 0, 1, 1, 0, 1])

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = fit_model(args, X, y, c_value=1.0, verbose=0)

    messages = [str(warning.message) for warning in caught]
    assert model.get_params()["penalty"] == "deprecated"
    assert model.get_params()["l1_ratio"] == 1.0
    assert not any(
        "penalty" in message and "deprecated" in message for message in messages
    )
    assert not any("Inconsistent values" in message for message in messages)
