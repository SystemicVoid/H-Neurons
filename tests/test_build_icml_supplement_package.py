"""Tests for the ICML supplement package builder."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from zipfile import ZipFile

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

build_icml_supplement_package = importlib.import_module("build_icml_supplement_package")
REPO_ROOT = Path(__file__).resolve().parent.parent


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


class TestBuildIcmlSupplementPackage:
    def test_repo_manifest_builds_bundle(self, tmp_path):
        output_dir = tmp_path / "bundle"
        zip_path = tmp_path / "bundle.zip"

        build_icml_supplement_package.build_package(
            build_icml_supplement_package.DEFAULT_MANIFEST,
            output_dir,
            zip_path,
            repo_root=REPO_ROOT,
        )

        assert (output_dir / "README.md").exists()
        assert (output_dir / "artifact_manifest.md").exists()
        assert (output_dir / "reference/main.tex").exists()
        assert (output_dir / "code/README.md").exists()
        assert (output_dir / "code/pyproject.toml").exists()
        assert (output_dir / "code/scripts/run_intervention.py").exists()
        assert (output_dir / "code/tests/test_utils.py").exists()
        assert (
            output_dir / "data/judge_validation/bridge_irr/adjudication_rule.md"
        ).exists()
        assert (
            output_dir / "data/judge_validation/bridge_irr/bridge_irr_summary.json"
        ).exists()
        assert (
            output_dir / "data/judge_validation/bridge_irr/adjudicated_labels.jsonl"
        ).exists()
        assert zip_path.exists()

        readme_text = (output_dir / "README.md").read_text(encoding="utf-8")
        assert "## Artifact Scope" in readme_text
        assert "does not enable full recomputation from raw generations" in readme_text
        assert "Raw response/scored JSONLs" in readme_text
        assert "harmful prompt gold labels" in readme_text
        assert "raw provenance sidecars" in readme_text
        assert "omitted for safety/anonymization" in readme_text
        assert (
            "`data/judge_validation/bridge_irr/bridge_irr_summary.json` is included"
            in readme_text
        )
        assert (
            "`data/judge_validation/bridge_irr/adjudicated_labels.jsonl` is redacted"
            in readme_text
        )

        with ZipFile(zip_path) as archive:
            names = set(archive.namelist())
        assert "README.md" in names
        assert "code/README.md" in names
        assert "code/scripts/evaluate_csv2.py" in names
        assert "data/judge_validation/bridge_irr/adjudication_rule.md" in names
        assert "data/judge_validation/bridge_irr/bridge_irr_summary.json" in names
        assert "data/judge_validation/bridge_irr/adjudicated_labels.jsonl" in names

    def test_scan_output_rejects_banned_strings(self, tmp_path):
        _write(tmp_path / "README.md", "keep /home/hugo out of the bundle")

        with pytest.raises(ValueError, match="anonymization scan failed"):
            build_icml_supplement_package.scan_output_for_banned_strings(
                tmp_path, ["/home/hugo"]
            )

    def test_scan_output_rejects_banned_strings_in_package_manifest(self, tmp_path):
        _write(tmp_path / "package_manifest.json", '{"banned": "/home/hugo"}')

        with pytest.raises(
            ValueError, match="package_manifest.json contains banned string"
        ):
            build_icml_supplement_package.scan_output_for_banned_strings(
                tmp_path, ["/home/hugo"]
            )

    def test_manifest_rejects_blocked_prefixes(self, tmp_path):
        repo_root = tmp_path / "repo"
        _write(repo_root / "notes/secret.md", "do not ship")
        manifest = repo_root / "manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "copies": [{"src": "notes/secret.md", "dst": "secret.md"}],
                    "banned_path_prefixes": ["notes/"],
                    "banned_strings": [],
                }
            ),
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match="blocked by banned prefix"):
            build_icml_supplement_package.build_package(
                manifest,
                repo_root / "out",
                repo_root / "out.zip",
                repo_root=repo_root,
            )

    def test_build_package_redacts_public_manifest_banned_strings(self, tmp_path):
        repo_root = tmp_path / "repo"
        manifest = repo_root / "manifest.json"
        _write(repo_root / "README.md", "safe bundle readme")
        manifest.write_text(
            json.dumps(
                {
                    "copies": [
                        {"src": "README.md", "dst": "README.md"},
                        {"src": "manifest.json", "dst": "package_manifest.json"},
                    ],
                    "banned_path_prefixes": [],
                    "banned_strings": ["/home/hugo", "Hugo Nguyen"],
                }
            ),
            encoding="utf-8",
        )

        output_dir = repo_root / "out"
        build_icml_supplement_package.build_package(
            manifest,
            output_dir,
            repo_root / "out.zip",
            repo_root=repo_root,
        )

        public_manifest = json.loads(
            (output_dir / "package_manifest.json").read_text(encoding="utf-8")
        )
        assert public_manifest["banned_strings"] == [
            "<redacted deanonymizing token 1>",
            "<redacted deanonymizing token 2>",
        ]
        assert (
            public_manifest["banned_strings_redaction_note"]
            == "Original banned_strings values are omitted from the bundled public "
            "manifest to preserve anonymization."
        )

    def test_validate_markdown_relative_links_rejects_missing_files(self, tmp_path):
        _write(tmp_path / "README.md", "[missing](missing.md)\n")

        with pytest.raises(ValueError, match="markdown link validation failed"):
            build_icml_supplement_package.validate_markdown_relative_links(tmp_path)

    def test_validate_markdown_relative_links_rejects_escaped_paths(self, tmp_path):
        _write(tmp_path / "support/summary.md", "[report](../../reports/full.md)\n")

        with pytest.raises(ValueError, match="links outside the package"):
            build_icml_supplement_package.validate_markdown_relative_links(tmp_path)
