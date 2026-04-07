"""Tests for scripts.lib.pipeline — incident-driven guard functions.

Each test traces to the invariant it protects. If a test here fails,
a GPU pipeline would silently skip incomplete data or lose work.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.lib.pipeline import (
    _count_lines,
    check_stage_complete,
    log_run,
    main,
    manifest_count,
)
from scripts.utils import format_alpha_label


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def manifest_file(tmp_path: Path) -> Path:
    """Create a 5-entry manifest."""
    p = tmp_path / "manifest.json"
    p.write_text(json.dumps(["id_0", "id_1", "id_2", "id_3", "id_4"]))
    return p


@pytest.fixture()
def stage_dir(tmp_path: Path) -> Path:
    d = tmp_path / "experiment"
    d.mkdir()
    return d


def _write_alpha(stage_dir: Path, alpha: float, n_lines: int) -> Path:
    """Write a fake alpha JSONL with n_lines records."""
    f = stage_dir / f"alpha_{format_alpha_label(alpha)}.jsonl"
    f.write_text(
        "".join(f'{{"id": "id_{i}", "alpha": {alpha}}}\n' for i in range(n_lines))
    )
    return f


# ---------------------------------------------------------------------------
# _count_lines
# ---------------------------------------------------------------------------


class TestCountLines:
    def test_normal_file(self, tmp_path: Path) -> None:
        f = tmp_path / "data.jsonl"
        f.write_text("line1\nline2\nline3\n")
        assert _count_lines(f) == 3

    def test_no_trailing_newline(self, tmp_path: Path) -> None:
        f = tmp_path / "data.jsonl"
        f.write_text("line1\nline2")
        # wc -l counts newline-terminated lines; we count all lines
        assert _count_lines(f) == 2

    def test_empty_file(self, tmp_path: Path) -> None:
        f = tmp_path / "data.jsonl"
        f.write_text("")
        assert _count_lines(f) == 0


# ---------------------------------------------------------------------------
# manifest_count
# ---------------------------------------------------------------------------


class TestManifestCount:
    def test_basic(self, manifest_file: Path) -> None:
        assert manifest_count(manifest_file) == 5

    def test_empty_list(self, tmp_path: Path) -> None:
        p = tmp_path / "empty.json"
        p.write_text("[]")
        assert manifest_count(p) == 0

    def test_non_list_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.json"
        p.write_text('{"not": "a list"}')
        with pytest.raises(ValueError, match="not a JSON list"):
            manifest_count(p)


# ---------------------------------------------------------------------------
# check_stage_complete — the core guard
# ---------------------------------------------------------------------------


class TestCheckStageComplete:
    """Incident: 2026-04-07 — alpha_8.0.jsonl had 64/100 lines after power-off.
    Old guard only checked file existence, so it skipped re-generation."""

    def test_all_complete(self, stage_dir: Path, manifest_file: Path) -> None:
        for alpha in [0.0, 1.0]:
            _write_alpha(stage_dir, alpha, 5)
        assert check_stage_complete(stage_dir, manifest_file, [0.0, 1.0]) is True

    def test_missing_file(self, stage_dir: Path, manifest_file: Path) -> None:
        _write_alpha(stage_dir, 0.0, 5)
        # alpha_1.0 is missing
        assert check_stage_complete(stage_dir, manifest_file, [0.0, 1.0]) is False

    def test_incomplete_file(self, stage_dir: Path, manifest_file: Path) -> None:
        """THE bug this library was created to catch."""
        _write_alpha(stage_dir, 0.0, 5)
        _write_alpha(stage_dir, 1.0, 3)  # only 3/5
        assert check_stage_complete(stage_dir, manifest_file, [0.0, 1.0]) is False

    def test_over_count_is_ok(self, stage_dir: Path, manifest_file: Path) -> None:
        """More records than expected (e.g. appended duplicates) is not a failure."""
        _write_alpha(stage_dir, 0.0, 7)
        assert check_stage_complete(stage_dir, manifest_file, [0.0]) is True

    def test_empty_alpha_list(self, stage_dir: Path, manifest_file: Path) -> None:
        """No alphas to check = vacuously complete."""
        assert check_stage_complete(stage_dir, manifest_file, []) is True

    def test_diagnostics_on_stderr(
        self, stage_dir: Path, manifest_file: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _write_alpha(stage_dir, 0.0, 5)
        _write_alpha(stage_dir, 1.0, 2)
        check_stage_complete(stage_dir, manifest_file, [0.0, 1.0])
        captured = capsys.readouterr()
        assert "incomplete" in captured.err
        assert "2/5" in captured.err

    def test_missing_file_diagnostics(
        self, stage_dir: Path, manifest_file: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        check_stage_complete(stage_dir, manifest_file, [0.0])
        captured = capsys.readouterr()
        assert "missing" in captured.err


# ---------------------------------------------------------------------------
# log_run
# ---------------------------------------------------------------------------


class TestLogRun:
    def test_creates_file(self, tmp_path: Path) -> None:
        notes = tmp_path / "notes" / "runs_to_analyse.md"
        log_run(
            "data/foo/experiment", "jailbreak + causal + alphas 0-8", notes_file=notes
        )
        content = notes.read_text()
        assert "data/foo/experiment" in content
        assert "jailbreak + causal + alphas 0-8" in content
        assert "awaiting analysis" in content

    def test_appends_to_existing(self, tmp_path: Path) -> None:
        notes = tmp_path / "runs.md"
        notes.write_text("# Existing\n")
        log_run("run1", "first run", notes_file=notes)
        log_run("run2", "second run", notes_file=notes)
        content = notes.read_text()
        assert "# Existing" in content
        assert "run1" in content
        assert "run2" in content

    def test_includes_iso_timestamp(self, tmp_path: Path) -> None:
        notes = tmp_path / "runs.md"
        log_run("run1", "desc", notes_file=notes)
        content = notes.read_text()
        # ISO timestamp has T separator and +00:00 or Z
        assert "T" in content


# ---------------------------------------------------------------------------
# CLI integration
# ---------------------------------------------------------------------------


class TestCli:
    def test_check_stage_complete_exit_0(
        self, stage_dir: Path, manifest_file: Path
    ) -> None:
        _write_alpha(stage_dir, 0.0, 5)
        rc = main(
            [
                "check-stage",
                "--output-dir",
                str(stage_dir),
                "--manifest",
                str(manifest_file),
                "--alphas",
                "0.0",
            ]
        )
        assert rc == 0

    def test_check_stage_incomplete_exit_1(
        self, stage_dir: Path, manifest_file: Path
    ) -> None:
        _write_alpha(stage_dir, 0.0, 2)
        rc = main(
            [
                "check-stage",
                "--output-dir",
                str(stage_dir),
                "--manifest",
                str(manifest_file),
                "--alphas",
                "0.0",
            ]
        )
        assert rc == 1

    def test_manifest_count(
        self, manifest_file: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        rc = main(["manifest-count", str(manifest_file)])
        assert rc == 0
        captured = capsys.readouterr()
        assert "5" in captured.out

    def test_log_run(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        (tmp_path / "notes").mkdir()
        rc = main(
            [
                "log-run",
                "--run-dir",
                "data/foo",
                "--description",
                "test run",
            ]
        )
        assert rc == 0
        assert "data/foo" in (tmp_path / "notes" / "runs_to_analyse.md").read_text()
