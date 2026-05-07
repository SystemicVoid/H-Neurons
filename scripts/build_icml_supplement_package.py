#!/usr/bin/env python3
"""Assemble the anonymized ICML supplementary package from an allowlist."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import cast
from urllib.parse import unquote, urlparse
from zipfile import ZIP_DEFLATED, ZipFile

REPO_ROOT = Path(__file__).resolve().parents[1]
SUPPLEMENT_ROOT = REPO_ROOT / "paper/icml/supplement"
DEFAULT_MANIFEST = SUPPLEMENT_ROOT / "package_manifest.json"
DEFAULT_OUTPUT_DIR = SUPPLEMENT_ROOT / "build/icml_supplement_package"
DEFAULT_ZIP_PATH = SUPPLEMENT_ROOT / "build/icml_supplement_package.zip"

TEXT_SUFFIXES = {
    ".json",
    ".jsonl",
    ".md",
    ".py",
    ".tex",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}

PUBLIC_MANIFEST_PATH = Path("package_manifest.json")

MARKDOWN_LINK_RE = re.compile(r"(?<!!)\[[^\]\n]+\]\(([^)\n]+)\)")
EXTERNAL_LINK_SCHEMES = {"http", "https", "mailto", "tel"}

CODE_README = """# Curated Code Bundle

This directory contains the paper-critical code and test slice for the ICML 2026 supplementary package.
It is intentionally narrower than the full internal repository.

## Layout

- `scripts/` contains the curated Python entrypoints and their local dependencies for the paper's localization, externality, and measurement claims.
- `tests/` contains the safe unit-test slice used to verify the bundled code paths.
- `pyproject.toml` is the minimal environment file for `uv sync --dev`.
- `requirements.txt` is the pinned export from the main repository environment.

## Suggested Verification Flow

```bash
cd code
uv sync --dev
uv run pytest tests
```

## Scope Notes

- The bundled tests are regression/unit tests; they do not require access to model weights or raw experiment corpora.
- Some scripts in `scripts/` support larger reruns that require local model checkpoints, benchmark data, or API credentials. Those heavyweight assets are not part of this supplement package.
"""

CODE_PYPROJECT = """[project]
name = "icml-2026-supplement-code"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "datasets>=4.5.0",
    "joblib>=1.5.3",
    "matplotlib>=3.10.8",
    "openai>=2.28.0",
    "pandas>=2.3.3",
    "python-dotenv>=1.2.2",
    "sae-lens>=6.39.0",
    "scikit-learn>=1.8.0",
    "scipy>=1.17.1",
    "seaborn>=0.13.2",
    "torch>=2.9.1",
    "transformers>=4.57.6",
    "transformers-stream-generator>=0.0.5",
    "wandb>=0.24.0",
]

[dependency-groups]
dev = [
    "pytest>=9.0.2",
]

[tool.pytest.ini_options]
testpaths = ["tests"]
"""


@dataclass(frozen=True)
class FileCopySpec:
    src: Path
    dst: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--zip-path", type=Path, default=DEFAULT_ZIP_PATH)
    return parser.parse_args()


def load_manifest(path: Path, *, repo_root: Path = REPO_ROOT) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    copies = payload.get("copies")
    if not isinstance(copies, list) or not copies:
        raise ValueError(f"{path} must define a non-empty 'copies' list")
    for field in ("banned_path_prefixes", "banned_strings"):
        value = payload.get(field)
        if not isinstance(value, list) or any(
            not isinstance(item, str) for item in value
        ):
            raise ValueError(f"{path} field '{field}' must be a list of strings")
    materialize_copy_specs(copies, repo_root=repo_root)
    return payload


def _resolve_repo_path(path_str: str, *, repo_root: Path) -> Path:
    resolved = (repo_root / path_str).resolve()
    try:
        resolved.relative_to(repo_root.resolve())
    except ValueError as exc:
        raise ValueError(f"{path_str!r} resolves outside repo root") from exc
    return resolved


def _copy_entry_matches_banned_prefix(
    relative_src: Path, banned_prefixes: list[str]
) -> str | None:
    src_text = relative_src.as_posix()
    for prefix in banned_prefixes:
        if src_text.startswith(prefix):
            return prefix
    return None


def materialize_copy_specs(
    entries: list[object], *, repo_root: Path = REPO_ROOT
) -> list[FileCopySpec]:
    seen_dsts: set[Path] = set()
    specs: list[FileCopySpec] = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("manifest copy entries must be JSON objects")
        entry = cast(dict[str, object], entry)
        src_value = entry.get("src")
        glob_value = entry.get("glob")
        dst_value = entry.get("dst")
        dst_dir_value = entry.get("dst_dir")
        if (src_value is None) == (glob_value is None):
            raise ValueError("each copy entry must set exactly one of 'src' or 'glob'")
        if src_value is not None:
            if not isinstance(src_value, str) or not isinstance(dst_value, str):
                raise ValueError("'src' copy entries require string 'src' and 'dst'")
            src_path = _resolve_repo_path(src_value, repo_root=repo_root)
            if not src_path.is_file():
                raise FileNotFoundError(f"missing source file: {src_value}")
            dst_path = Path(dst_value)
            if dst_path in seen_dsts:
                raise ValueError(f"duplicate destination path in manifest: {dst_path}")
            seen_dsts.add(dst_path)
            specs.append(FileCopySpec(src=src_path, dst=dst_path))
            continue

        if not isinstance(glob_value, str) or not isinstance(dst_dir_value, str):
            raise ValueError("'glob' copy entries require string 'glob' and 'dst_dir'")
        matches = sorted(repo_root.glob(glob_value))
        if not matches:
            raise FileNotFoundError(f"glob did not match any files: {glob_value}")
        for match in matches:
            if not match.is_file():
                continue
            dst_path = Path(dst_dir_value) / match.name
            if dst_path in seen_dsts:
                raise ValueError(f"duplicate destination path in manifest: {dst_path}")
            seen_dsts.add(dst_path)
            specs.append(FileCopySpec(src=match.resolve(), dst=dst_path))
    return specs


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def _is_text_file(path: Path) -> bool:
    return path.suffix.lower() in TEXT_SUFFIXES


def _build_public_manifest(manifest: dict[str, object]) -> dict[str, object]:
    public_manifest = dict(manifest)
    banned_strings = cast(list[str], manifest["banned_strings"])
    public_manifest["banned_strings"] = [
        f"<redacted deanonymizing token {index}>"
        for index, _ in enumerate(banned_strings, start=1)
    ]
    public_manifest["banned_strings_redaction_note"] = (
        "Original banned_strings values are omitted from the bundled public "
        "manifest to preserve anonymization."
    )
    return public_manifest


def _write_public_manifest(output_dir: Path, manifest: dict[str, object]) -> None:
    public_manifest = _build_public_manifest(manifest)
    manifest_text = json.dumps(public_manifest, indent=2)
    _write(output_dir / PUBLIC_MANIFEST_PATH, manifest_text)


def scan_output_for_banned_strings(output_dir: Path, banned_strings: list[str]) -> None:
    violations: list[str] = []
    for path in sorted(output_dir.rglob("*")):
        if not path.is_file() or not _is_text_file(path):
            continue
        text = path.read_text(encoding="utf-8")
        for banned in banned_strings:
            if banned in text:
                violations.append(
                    f"{path.relative_to(output_dir)} contains banned string {banned!r}"
                )
    if violations:
        joined = "\n".join(violations)
        raise ValueError(f"anonymization scan failed:\n{joined}")


def _normalise_markdown_target(raw_target: str) -> str:
    target = raw_target.strip()
    if target.startswith("<"):
        end = target.find(">")
        if end != -1:
            return target[1:end]
    return target.split()[0] if target else ""


def validate_markdown_relative_links(output_dir: Path) -> None:
    """Ensure bundled Markdown links point to files inside the package."""

    root = output_dir.resolve()
    violations: list[str] = []
    for path in sorted(output_dir.rglob("*.md")):
        text = path.read_text(encoding="utf-8")
        for match in MARKDOWN_LINK_RE.finditer(text):
            line_no = text.count("\n", 0, match.start()) + 1
            target = _normalise_markdown_target(match.group(1))
            if not target or target.startswith("#"):
                continue
            parsed = urlparse(target)
            if parsed.scheme in EXTERNAL_LINK_SCHEMES or parsed.netloc:
                continue
            if parsed.scheme and parsed.scheme not in EXTERNAL_LINK_SCHEMES:
                violations.append(
                    f"{path.relative_to(output_dir)}:{line_no} "
                    f"uses unsupported link scheme {parsed.scheme!r}"
                )
                continue
            target_path = parsed.path
            if not target_path:
                continue
            candidate = (path.parent / unquote(target_path)).resolve()
            try:
                candidate.relative_to(root)
            except ValueError:
                violations.append(
                    f"{path.relative_to(output_dir)}:{line_no} "
                    f"links outside the package: {target}"
                )
                continue
            if not candidate.exists():
                violations.append(
                    f"{path.relative_to(output_dir)}:{line_no} "
                    f"links to missing file: {target}"
                )
    if violations:
        joined = "\n".join(violations)
        raise ValueError(f"markdown link validation failed:\n{joined}")


def _write_generated_code_files(output_dir: Path) -> None:
    _write(output_dir / "code/README.md", CODE_README)
    _write(output_dir / "code/pyproject.toml", CODE_PYPROJECT)


def _copy_sources(
    specs: list[FileCopySpec],
    output_dir: Path,
    *,
    repo_root: Path,
    banned_path_prefixes: list[str],
) -> None:
    for spec in specs:
        relative_src = spec.src.relative_to(repo_root)
        blocked_prefix = _copy_entry_matches_banned_prefix(
            relative_src, banned_path_prefixes
        )
        if blocked_prefix is not None:
            raise ValueError(
                f"manifest source {relative_src} is blocked by banned prefix {blocked_prefix!r}"
            )
        dst_path = output_dir / spec.dst
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(spec.src, dst_path)


def _write_zip(output_dir: Path, zip_path: Path) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    if zip_path.exists():
        zip_path.unlink()
    with ZipFile(zip_path, "w", compression=ZIP_DEFLATED) as archive:
        for path in sorted(output_dir.rglob("*")):
            if path.is_file():
                archive.write(path, arcname=path.relative_to(output_dir))


def build_package(
    manifest_path: Path,
    output_dir: Path,
    zip_path: Path,
    *,
    repo_root: Path = REPO_ROOT,
) -> None:
    manifest = load_manifest(manifest_path, repo_root=repo_root)
    copies = cast(list[object], manifest["copies"])
    specs = materialize_copy_specs(copies, repo_root=repo_root)
    banned_prefixes = cast(list[str], manifest["banned_path_prefixes"])
    banned_strings = cast(list[str], manifest["banned_strings"])

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _copy_sources(
        specs,
        output_dir,
        repo_root=repo_root,
        banned_path_prefixes=banned_prefixes,
    )
    _write_generated_code_files(output_dir)
    _write_public_manifest(output_dir, manifest)
    validate_markdown_relative_links(output_dir)
    scan_output_for_banned_strings(output_dir, banned_strings)
    _write_zip(output_dir, zip_path)


def main() -> None:
    args = parse_args()
    try:
        build_package(args.manifest, args.output_dir, args.zip_path)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    print(f"Built supplement package in {args.output_dir}")
    print(f"Wrote zip archive to {args.zip_path}")


if __name__ == "__main__":
    main()
