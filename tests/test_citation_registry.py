"""Tests for citation registry integrity validation."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

check_citation_registry = importlib.import_module("check_citation_registry")


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _make_entry(**overrides: object) -> dict[str, object]:
    entry: dict[str, object] = {
        "id": "S99",
        "title": "Correct Local Paper",
        "authors": "Test Author",
        "year": 2026,
        "venue": "TestConf",
        "arxiv": None,
        "doi": None,
        "url": "https://example.com",
        "local_pdf": "papers/correct-paper.pdf",
        "local_md": "papers/correct-paper.md",
        "acquired": True,
        "md_converted": True,
        "role": "Test role",
        "supports_claims": [],
        "tier": 1,
        "notes": "Test notes",
    }
    entry.update(overrides)
    return entry


def _write_registry(tmp_path: Path, payload: dict[str, dict[str, object]]) -> Path:
    registry = tmp_path / "paper/citations/registry.json"
    registry.parent.mkdir(parents=True, exist_ok=True)
    registry.write_text(json.dumps(payload), encoding="utf-8")
    return registry


class TestCitationRegistryValidation:
    def test_repo_registry_passes(self):
        registry = check_citation_registry.load_registry(
            check_citation_registry.DEFAULT_REGISTRY
        )

        errors = check_citation_registry.validate_registry(
            registry, repo_root=check_citation_registry.REPO_ROOT
        )

        assert errors == []

    def test_detects_wrong_markdown_mapping(self, tmp_path):
        _write(tmp_path / "papers/correct-paper.md", "# Correct Local Paper\n")
        _write(tmp_path / "papers/correct-paper.pdf", "pdf")
        _write(tmp_path / "papers/wrong-paper.md", "# Wrong Other Paper\n")
        registry = {
            "broken2026paper": _make_entry(local_md="papers/wrong-paper.md"),
        }

        errors = check_citation_registry.validate_registry(registry, repo_root=tmp_path)

        assert [error.key for error in errors] == ["broken2026paper"]
        assert errors[0].field == "title"
        assert "does not match markdown heading" in errors[0].message

    def test_detects_wrong_pdf_mapping(self, tmp_path):
        _write(tmp_path / "papers/correct-paper.md", "# Correct Local Paper\n")
        _write(tmp_path / "papers/correct-paper.pdf", "pdf")
        _write(tmp_path / "papers/wrong-paper.pdf", "pdf")
        registry = {
            "broken2026paper": _make_entry(local_pdf="papers/wrong-paper.pdf"),
        }

        errors = check_citation_registry.validate_registry(registry, repo_root=tmp_path)

        assert [error.key for error in errors] == ["broken2026paper"]
        assert errors[0].field == "local_pdf"
        assert "same paper basename as local_md" in errors[0].message

    def test_detects_missing_local_file(self, tmp_path):
        _write(tmp_path / "papers/correct-paper.pdf", "pdf")
        registry = {
            "missing2026paper": _make_entry(),
        }

        errors = check_citation_registry.validate_registry(registry, repo_root=tmp_path)

        assert [error.field for error in errors] == ["local_md"]
        assert "local file does not exist" in errors[0].message

    def test_detects_flag_contradictions(self, tmp_path):
        _write(tmp_path / "papers/correct-paper.md", "# Correct Local Paper\n")
        _write(tmp_path / "papers/correct-paper.pdf", "pdf")
        registry = {
            "flags2026paper": _make_entry(acquired=False, md_converted=False),
        }

        errors = check_citation_registry.validate_registry(registry, repo_root=tmp_path)

        assert {(error.field, error.message) for error in errors} == {
            ("acquired", "cannot be false when local_pdf is populated"),
            ("md_converted", "cannot be false when local_md is populated"),
        }

    def test_rejects_non_boolean_flags(self, tmp_path):
        _write(tmp_path / "papers/correct-paper.md", "# Correct Local Paper\n")
        _write(tmp_path / "papers/correct-paper.pdf", "pdf")
        registry = {
            "flags2026paper": _make_entry(acquired="true", md_converted="false"),
        }

        errors = check_citation_registry.validate_registry(registry, repo_root=tmp_path)

        assert {(error.field, error.message) for error in errors} == {
            ("acquired", "must be a boolean"),
            ("md_converted", "must be a boolean"),
        }

    def test_rejects_missing_boolean_flags(self, tmp_path):
        _write(tmp_path / "papers/correct-paper.md", "# Correct Local Paper\n")
        _write(tmp_path / "papers/correct-paper.pdf", "pdf")
        registry = {
            "flags2026paper": _make_entry(),
        }
        del registry["flags2026paper"]["acquired"]
        del registry["flags2026paper"]["md_converted"]

        errors = check_citation_registry.validate_registry(registry, repo_root=tmp_path)

        assert {(error.field, error.message) for error in errors} == {
            ("acquired", "must be a boolean"),
            ("md_converted", "must be a boolean"),
        }

    def test_allows_unacquired_entry_without_local_files(self, tmp_path):
        registry = {
            "pending2026paper": _make_entry(
                title="Pending Paper",
                local_pdf=None,
                local_md=None,
                acquired=False,
                md_converted=False,
            ),
        }

        errors = check_citation_registry.validate_registry(registry, repo_root=tmp_path)

        assert errors == []

    def test_load_registry_rejects_non_object_payload(self, tmp_path):
        registry = tmp_path / "paper/citations/registry.json"
        registry.parent.mkdir(parents=True, exist_ok=True)
        registry.write_text('["not-an-object"]', encoding="utf-8")

        with pytest.raises(ValueError, match="non-empty object"):
            check_citation_registry.load_registry(registry)
