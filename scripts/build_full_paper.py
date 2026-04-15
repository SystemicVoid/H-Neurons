#!/usr/bin/env python3
"""Assemble the paper draft from section source files."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "paper/draft/assembly_manifest.json"
DEFAULT_OUTPUT = REPO_ROOT / "paper/draft/full_paper.md"
GENERATED_WARNING = """<!--
Generated file: do not edit directly.
Edit paper/draft source shards and rebuild with:
  uv run python scripts/build_full_paper.py
-->"""
RULE_AFTER = {"front_matter.md", "abstract.md", "section_9_conclusion.md"}


@dataclass(frozen=True)
class SourceSpec:
    path: Path
    expected_heading: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit nonzero if the compiled paper is stale",
    )
    return parser.parse_args()


def load_manifest(path: Path) -> list[SourceSpec]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_sources = payload.get("sources")
    if not isinstance(raw_sources, list) or not raw_sources:
        raise ValueError(f"{path} must define a non-empty 'sources' list")

    manifest_dir = path.parent
    seen: set[Path] = set()
    sources: list[SourceSpec] = []
    for entry in raw_sources:
        if not isinstance(entry, dict):
            raise ValueError(f"{path} contains a non-object manifest entry")
        raw_rel_path = entry.get("path")
        if not isinstance(raw_rel_path, str) or not raw_rel_path:
            raise ValueError(f"{path} contains an entry without a valid 'path'")
        source_path = manifest_dir / raw_rel_path
        if source_path in seen:
            raise ValueError(f"duplicate source entry in manifest: {raw_rel_path}")
        seen.add(source_path)
        expected_heading = entry.get("expected_heading")
        if expected_heading is not None and not isinstance(expected_heading, str):
            raise ValueError(
                f"{path} entry for {raw_rel_path} has a non-string expected_heading"
            )
        sources.append(SourceSpec(path=source_path, expected_heading=expected_heading))
    return sources


def extract_first_heading(text: str) -> str | None:
    in_comment = False
    for raw_line in text.splitlines():
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
            return line
        return None
    return None


def validate_source(spec: SourceSpec) -> str:
    if not spec.path.exists():
        raise FileNotFoundError(f"missing paper source: {spec.path}")
    text = spec.path.read_text(encoding="utf-8").rstrip()
    if spec.expected_heading is not None:
        found_heading = extract_first_heading(text)
        if found_heading != spec.expected_heading:
            raise ValueError(
                f"{spec.path} heading mismatch: expected "
                f"{spec.expected_heading!r}, found {found_heading!r}"
            )
    return text


def render_document(sources: list[SourceSpec]) -> str:
    if not sources:
        raise ValueError("cannot render document with no sources")

    parts: list[str] = []
    for idx, spec in enumerate(sources):
        parts.append(validate_source(spec))
        if idx == len(sources) - 1:
            continue
        separator = "\n\n---\n\n" if spec.path.name in RULE_AFTER else "\n\n"
        parts.append(separator)

    body = "".join(parts).rstrip() + "\n"
    return GENERATED_WARNING + "\n\n" + body


def build_full_paper(manifest_path: Path, output_path: Path, *, check: bool) -> None:
    sources = load_manifest(manifest_path)
    rendered = render_document(sources)
    if check:
        if not output_path.exists():
            raise SystemExit(f"{output_path} is missing; run the builder first")
        current = output_path.read_text(encoding="utf-8")
        if current != rendered:
            raise SystemExit(
                f"{output_path} is stale; run `uv run python scripts/build_full_paper.py`"
            )
        return

    output_path.write_text(rendered, encoding="utf-8")
    print(f"Built {output_path}")


def main() -> None:
    args = parse_args()
    try:
        build_full_paper(args.manifest, args.output, check=args.check)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
