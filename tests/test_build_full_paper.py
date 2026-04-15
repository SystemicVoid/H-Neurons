"""Tests for the paper assembly builder."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

build_full_paper = importlib.import_module("build_full_paper")
REPO_ROOT = Path(__file__).resolve().parent.parent


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _manifest(tmp_path: Path, sources: list[dict[str, str]]) -> Path:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"sources": sources}), encoding="utf-8")
    return manifest


class TestBuildFullPaper:
    def test_default_paths_are_repo_root_relative(self, monkeypatch):
        monkeypatch.chdir(REPO_ROOT / "paper/draft")
        monkeypatch.setattr(sys, "argv", ["build_full_paper.py"])

        args = build_full_paper.parse_args()

        assert args.manifest == REPO_ROOT / "paper/draft/assembly_manifest.json"
        assert args.output == REPO_ROOT / "paper/draft/full_paper.md"

    def test_builds_from_manifest(self, tmp_path, monkeypatch):
        _write(tmp_path / "front_matter.md", "Title\n\nAuthor")
        _write(tmp_path / "abstract.md", "# Abstract\n\nAbstract body.")
        _write(tmp_path / "section_1.md", "# 1. Intro\n\nIntro body.")
        _write(tmp_path / "references.md", "# References\n\nRef.")

        manifest = _manifest(
            tmp_path,
            [
                {"path": "front_matter.md"},
                {"path": "abstract.md", "expected_heading": "# Abstract"},
                {"path": "section_1.md", "expected_heading": "# 1. Intro"},
                {"path": "references.md", "expected_heading": "# References"},
            ],
        )
        output = tmp_path / "full_paper.md"
        monkeypatch.setattr(
            build_full_paper,
            "parse_args",
            lambda: argparse.Namespace(
                manifest=manifest,
                output=output,
                check=False,
            ),
        )

        build_full_paper.main()

        rendered = output.read_text(encoding="utf-8")
        assert rendered.startswith(build_full_paper.GENERATED_WARNING)
        assert "Title\n\nAuthor\n\n---\n\n# Abstract" in rendered
        assert "# 1. Intro\n\nIntro body.\n\n# References" in rendered

    def test_check_mode_fails_on_stale_output(self, tmp_path, monkeypatch):
        _write(tmp_path / "front_matter.md", "Title")
        _write(tmp_path / "abstract.md", "# Abstract\n\nFresh.")
        manifest = _manifest(
            tmp_path,
            [
                {"path": "front_matter.md"},
                {"path": "abstract.md", "expected_heading": "# Abstract"},
            ],
        )
        output = tmp_path / "full_paper.md"
        output.write_text("stale", encoding="utf-8")
        monkeypatch.setattr(
            build_full_paper,
            "parse_args",
            lambda: argparse.Namespace(
                manifest=manifest,
                output=output,
                check=True,
            ),
        )

        with pytest.raises(SystemExit, match="is stale"):
            build_full_paper.main()

    def test_missing_source_fails(self, tmp_path):
        manifest = _manifest(
            tmp_path,
            [
                {"path": "missing.md", "expected_heading": "# Missing"},
            ],
        )

        with pytest.raises(FileNotFoundError, match="missing paper source"):
            build_full_paper.build_full_paper(
                manifest_path=manifest,
                output_path=tmp_path / "out.md",
                check=False,
            )

    def test_duplicate_source_fails(self, tmp_path):
        _write(tmp_path / "abstract.md", "# Abstract\n\nBody")
        manifest = _manifest(
            tmp_path,
            [
                {"path": "abstract.md", "expected_heading": "# Abstract"},
                {"path": "abstract.md", "expected_heading": "# Abstract"},
            ],
        )

        with pytest.raises(ValueError, match="duplicate source entry"):
            build_full_paper.load_manifest(manifest)

    def test_heading_validation_fails(self, tmp_path):
        _write(tmp_path / "abstract.md", "# Wrong\n\nBody")
        manifest = _manifest(
            tmp_path,
            [
                {"path": "abstract.md", "expected_heading": "# Abstract"},
            ],
        )

        with pytest.raises(ValueError, match="heading mismatch"):
            build_full_paper.build_full_paper(
                manifest_path=manifest,
                output_path=tmp_path / "out.md",
                check=False,
            )
