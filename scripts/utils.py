"""Shared lightweight utilities used across pipeline scripts.

Keep this module free of heavy imports (torch, transformers, etc.)
so it can be used from any script without pulling in GPU dependencies.
"""

from __future__ import annotations

import re
import string
import subprocess
from typing import Any


_SENSITIVE_ARG_SUFFIXES = ("token", "secret", "password", "auth_key", "_api_key")
_SENSITIVE_ARG_NAMES = {"api_key", "sampling_api_key"}


def normalize_answer(s: str | None) -> str:
    """Standardize answer strings for comparison."""

    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def handle_punc(text: str) -> str:
        exclude = set(string.punctuation + "\u2018\u2019\u00b4`")
        return "".join(ch if ch not in exclude else " " for ch in text)

    if not s:
        return ""
    return white_space_fix(
        remove_articles(handle_punc(str(s).lower().replace("_", " ")))
    ).strip()


def extract_mc_answer(response: str, valid_letters: list[str]) -> str | None:
    """Extract a multiple-choice letter from model response."""
    text = response.strip()
    if text and text[0].upper() in valid_letters:
        return text[0].upper()
    for pattern in [
        r"\b(?:answer|correct)\s+(?:is|:)\s*\(?([A-Z])\)?",
        r"^\(?([A-Z])\)",
        r"^([A-Z])\.",
        r"\*\*([A-Z])\*\*",
    ]:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            letter = match.group(1).upper()
            if letter in valid_letters:
                return letter
    for letter in valid_letters:
        if re.search(rf"\b{letter}\b", text):
            return letter
    return None


def get_git_sha() -> str | None:
    """Return the current git SHA when available, otherwise None."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        check=False,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    if result.returncode != 0:
        return None
    sha = result.stdout.strip()
    return sha or None


def _is_sensitive_arg_name(name: str) -> bool:
    lowered = name.lower()
    if lowered in _SENSITIVE_ARG_NAMES:
        return True
    return lowered in _SENSITIVE_ARG_SUFFIXES or lowered.endswith(
        _SENSITIVE_ARG_SUFFIXES
    )


def sanitize_run_config(
    args_dict: dict[str, Any], extra: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Drop secret-like CLI args before uploading run metadata."""
    config = {
        key: value
        for key, value in args_dict.items()
        if key != "wandb" and not _is_sensitive_arg_name(key)
    }
    if extra:
        config.update({key: value for key, value in extra.items() if value is not None})
    return config
