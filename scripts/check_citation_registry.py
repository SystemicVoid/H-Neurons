#!/usr/bin/env python3
"""Validate citation registry integrity against local paper files."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from difflib import SequenceMatcher

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = REPO_ROOT / "paper/citations/registry.json"
TITLE_SIMILARITY_THRESHOLD = 0.75


@dataclass(frozen=True)
class ValidationError:
    key: str
    field: str
    message: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--registry",
        type=Path,
        default=DEFAULT_REGISTRY,
        help="Path to the citation registry JSON.",
    )
    return parser.parse_args()


def load_registry(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not payload:
        raise ValueError(f"{path} must contain a non-empty object")

    for key, entry in payload.items():
        if not isinstance(key, str) or not key:
            raise ValueError(f"{path} contains an invalid top-level key")
        if not isinstance(entry, dict):
            raise ValueError(f"{path} entry {key!r} must be an object")
    return payload


def extract_first_heading(path: Path) -> str | None:
    in_comment = False
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if in_comment:
            if "-->" in line:
                in_comment = False
            continue
        if line.startswith("<!--"):
            if "-->" not in line:
                in_comment = True
            continue
        if line.startswith("#"):
            return re.sub(r"^#+\s*", "", line).strip()
    return None


def normalize_title(text: str) -> str:
    normalized = text.lower()
    normalized = normalized.replace("&lt;", "<").replace("&gt;", ">")
    normalized = normalized.replace("–", "-").replace("—", "-")
    normalized = normalized.replace("’", "'").replace("“", '"')
    normalized = normalized.replace("”", '"').replace("≠", "!=")
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
    return " ".join(normalized.split())


def similarity(left: str, right: str) -> float:
    return SequenceMatcher(None, normalize_title(left), normalize_title(right)).ratio()


def validate_registry(
    registry: dict[str, dict[str, Any]], *, repo_root: Path
) -> list[ValidationError]:
    errors: list[ValidationError] = []

    for key, entry in registry.items():
        local_pdf = entry.get("local_pdf")
        local_md = entry.get("local_md")
        acquired = entry.get("acquired")
        md_converted = entry.get("md_converted")
        title = entry.get("title")

        for field_name, value in (("local_pdf", local_pdf), ("local_md", local_md)):
            if value is None:
                continue
            if not isinstance(value, str) or not value:
                errors.append(
                    ValidationError(
                        key, field_name, "must be null or a non-empty string"
                    )
                )
                continue
            if not (repo_root / value).exists():
                errors.append(
                    ValidationError(
                        key, field_name, f"local file does not exist: {value}"
                    )
                )

        if acquired is True and local_pdf is None:
            errors.append(
                ValidationError(
                    key, "acquired", "cannot be true when local_pdf is null"
                )
            )
        if acquired is False and local_pdf is not None:
            errors.append(
                ValidationError(
                    key, "acquired", "cannot be false when local_pdf is populated"
                )
            )
        if md_converted is True and local_md is None:
            errors.append(
                ValidationError(
                    key, "md_converted", "cannot be true when local_md is null"
                )
            )
        if md_converted is False and local_md is not None:
            errors.append(
                ValidationError(
                    key, "md_converted", "cannot be false when local_md is populated"
                )
            )

        if local_md is None:
            continue
        if not isinstance(local_md, str) or not local_md:
            continue
        local_md_path = repo_root / local_md
        if not local_md_path.exists():
            continue

        if not isinstance(title, str) or not title:
            errors.append(
                ValidationError(
                    key,
                    "title",
                    "must be a non-empty string when local_md is populated",
                )
            )
            continue

        heading = extract_first_heading(local_md_path)
        if not heading:
            errors.append(
                ValidationError(
                    key, "local_md", f"no markdown heading found in {local_md}"
                )
            )
            continue

        title_similarity = similarity(title, heading)
        if title_similarity < TITLE_SIMILARITY_THRESHOLD:
            errors.append(
                ValidationError(
                    key,
                    "title",
                    (
                        "registry title does not match markdown heading "
                        f"(similarity={title_similarity:.2f}; heading={heading!r})"
                    ),
                )
            )

    return errors


def main() -> None:
    args = parse_args()
    try:
        registry = load_registry(args.registry)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    errors = validate_registry(registry, repo_root=REPO_ROOT)
    if errors:
        for error in errors:
            print(f"{error.key}.{error.field}: {error.message}", file=sys.stderr)
        raise SystemExit(1)

    print(f"Validated {len(registry)} citation entries in {args.registry}")


if __name__ == "__main__":
    main()
