"""Regression tests for W&B safety and remote bootstrap wiring."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

# scripts/ uses flat sibling imports; add relevant dirs for test discovery.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "infra"))

from lambda_bootstrap_and_start_step1 import remote_setup_script  # noqa: E402
from utils import (  # noqa: E402
    finish_run_provenance,
    get_git_dirty,
    get_git_sha,
    provenance_error_message,
    provenance_status_for_exception,
    resolve_provenance_path,
    sanitize_run_config,
    start_run_provenance,
)


class TestGetGitSha:
    def test_returns_trimmed_sha_on_success(self, monkeypatch):
        def fake_run(*args, **kwargs):
            return subprocess.CompletedProcess(args[0], 0, stdout="abc123\n", stderr="")

        monkeypatch.setattr("utils.subprocess.run", fake_run)

        assert get_git_sha() == "abc123"

    def test_returns_none_when_git_metadata_unavailable(self, monkeypatch):
        def fake_run(*args, **kwargs):
            return subprocess.CompletedProcess(args[0], 128, stdout="", stderr="fatal")

        monkeypatch.setattr("utils.subprocess.run", fake_run)

        assert get_git_sha() is None

    def test_returns_none_when_subprocess_setup_is_invalid(self, monkeypatch):
        def fake_run(*args, **kwargs):
            raise ValueError("capture_output and stderr are mutually exclusive")

        monkeypatch.setattr("utils.subprocess.run", fake_run)

        assert get_git_sha() is None


class TestGetGitDirty:
    def test_returns_true_when_worktree_has_changes(self, monkeypatch):
        def fake_run(*args, **kwargs):
            return subprocess.CompletedProcess(args[0], 0, stdout=" M scripts/foo.py\n")

        monkeypatch.setattr("utils.subprocess.run", fake_run)

        assert get_git_dirty() is True

    def test_returns_false_when_worktree_is_clean(self, monkeypatch):
        def fake_run(*args, **kwargs):
            return subprocess.CompletedProcess(args[0], 0, stdout="", stderr="")

        monkeypatch.setattr("utils.subprocess.run", fake_run)

        assert get_git_dirty() is False

    def test_returns_none_when_git_metadata_unavailable(self, monkeypatch):
        def fake_run(*args, **kwargs):
            return subprocess.CompletedProcess(args[0], 128, stdout="", stderr="fatal")

        monkeypatch.setattr("utils.subprocess.run", fake_run)

        assert get_git_dirty() is None

    def test_returns_none_when_subprocess_setup_is_invalid(self, monkeypatch):
        def fake_run(*args, **kwargs):
            raise ValueError("capture_output and stderr are mutually exclusive")

        monkeypatch.setattr("utils.subprocess.run", fake_run)

        assert get_git_dirty() is None


class TestSanitizeRunConfig:
    def test_removes_secret_like_keys_and_preserves_safe_fields(self):
        config = sanitize_run_config(
            {
                "wandb": True,
                "api_key": "judge-secret",
                "sampling_api_key": "sampler-secret",
                "tailscale_auth_key": "ts-secret",
                "session_token": "token-secret",
                "db_password": "pw-secret",
                "base_url": "https://api.openai.com/v1",
                "model_path": "google/gemma-3-4b-it",
                "benchmark": "faitheval",
                "ssh_key_path": "~/.ssh/lambda",
            },
            extra={"git_sha": "abc123", "optional_none": None},
        )

        assert config == {
            "base_url": "https://api.openai.com/v1",
            "model_path": "google/gemma-3-4b-it",
            "benchmark": "faitheval",
            "ssh_key_path": "~/.ssh/lambda",
            "git_sha": "abc123",
        }


class TestRunProvenance:
    def test_resolve_provenance_path_for_file_and_directory(self):
        file_path = resolve_provenance_path(
            "data/run/results.json", "run_intervention", is_dir=False
        )
        dir_path = resolve_provenance_path(
            "data/run/experiment", "run_intervention", is_dir=True
        )

        assert file_path == Path("data/run/results.json.provenance.json")
        assert dir_path == Path("data/run/experiment/run_intervention.provenance.json")

    def test_start_and_finish_run_provenance_for_file_target(
        self, monkeypatch, tmp_path
    ):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr("utils.get_git_sha", lambda: "abc123")
        monkeypatch.setattr("utils.get_git_dirty", lambda: True)
        monkeypatch.setattr("utils.socket.gethostname", lambda: "lab-host")
        monkeypatch.setattr(
            "utils.sys.argv",
            ["scripts/collect_responses.py", "--output_path", "data/out.jsonl"],
        )
        monkeypatch.setattr(
            "utils.sys.orig_argv",
            [
                "uv",
                "run",
                "python",
                "scripts/collect_responses.py",
                "--output_path",
                "data/out.jsonl",
            ],
            raising=False,
        )
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
        monkeypatch.delenv("WANDB_MODE", raising=False)

        output_path = tmp_path / "data" / "out.jsonl"
        handle = start_run_provenance(
            args={"output_path": str(output_path), "api_key": "secret"},
            primary_target=output_path,
            output_targets=[output_path],
        )
        assert handle is not None

        provenance_path = output_path.with_name("out.jsonl.provenance.json")
        payload = json.loads(provenance_path.read_text())
        assert payload["status"] == "running"
        assert payload["git_sha"] == "abc123"
        assert payload["git_dirty"] is True
        assert payload["hostname"] == "lab-host"
        assert payload["safe_env"] == {"CUDA_VISIBLE_DEVICES": "0"}
        assert payload["args"] == {"output_path": str(output_path)}
        assert payload["output_targets"] == [str(output_path.resolve())]
        assert (
            payload["command"]
            == "uv run python scripts/collect_responses.py --output_path data/out.jsonl"
        )

        finish_run_provenance(handle, "completed", extra={"n_rows": 12})
        finished_payload = json.loads(provenance_path.read_text())
        assert finished_payload["status"] == "completed"
        assert finished_payload["n_rows"] == 12
        assert finished_payload["completed_at_utc"] is not None

    def test_start_and_finish_run_provenance_for_directory_target(
        self, monkeypatch, tmp_path
    ):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr("utils.get_git_sha", lambda: None)
        monkeypatch.setattr("utils.get_git_dirty", lambda: None)
        monkeypatch.setattr(
            "utils.sys.argv",
            ["scripts/run_intervention.py", "--benchmark", "faitheval"],
        )
        monkeypatch.setattr(
            "utils.sys.orig_argv",
            [
                "uv",
                "run",
                "python",
                "scripts/run_intervention.py",
                "--benchmark",
                "faitheval",
            ],
            raising=False,
        )
        monkeypatch.setenv("WANDB_MODE", "offline")

        output_dir = tmp_path / "data" / "experiment"
        handle = start_run_provenance(
            args={"benchmark": "faitheval", "wandb": True},
            primary_target=output_dir,
            output_targets=[output_dir, output_dir / "results.json"],
            extra={"wandb": {"project": "h-neurons", "mode": "offline"}},
            primary_target_is_dir=True,
        )
        assert handle is not None

        provenance_path = output_dir / "run_intervention.provenance.json"
        payload = json.loads(provenance_path.read_text())
        assert payload["status"] == "running"
        assert payload["git_sha"] is None
        assert payload["git_dirty"] is None
        assert payload["safe_env"] == {"WANDB_MODE": "offline"}
        assert payload["wandb"] == {"project": "h-neurons", "mode": "offline"}

        finish_run_provenance(
            handle,
            "failed",
            extra={
                "error": "boom",
                "output_targets": [str((output_dir / "results.json").resolve())],
            },
        )
        finished_payload = json.loads(provenance_path.read_text())
        assert finished_payload["status"] == "failed"
        assert finished_payload["error"] == "boom"
        assert finished_payload["output_targets"] == [
            str((output_dir / "results.json").resolve())
        ]

    def test_start_run_provenance_serializes_path_arguments(
        self, monkeypatch, tmp_path
    ):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr("utils.get_git_sha", lambda: "abc123")
        monkeypatch.setattr("utils.get_git_dirty", lambda: False)
        monkeypatch.setattr(
            "utils.sys.argv",
            ["scripts/prepare_bioasq_eval.py", "--output_parquet", "data/out.parquet"],
        )
        monkeypatch.setattr(
            "utils.sys.orig_argv",
            [
                "uv",
                "run",
                "python",
                "scripts/prepare_bioasq_eval.py",
                "--output_parquet",
                "data/out.parquet",
            ],
            raising=False,
        )

        input_path = tmp_path / "data" / "bioasq.json"
        output_path = tmp_path / "data" / "out.parquet"
        handle = start_run_provenance(
            args={
                "input_json": input_path,
                "output_parquet": output_path,
                "summary_out": output_path.with_suffix(".summary.json"),
            },
            primary_target=output_path,
            output_targets=[output_path],
        )
        assert handle is not None

        payload = json.loads(
            output_path.with_name("out.parquet.provenance.json").read_text()
        )
        assert payload["args"] == {
            "input_json": str(input_path),
            "output_parquet": str(output_path),
            "summary_out": str(output_path.with_suffix(".summary.json")),
        }

    def test_directory_targets_do_not_collide_between_scripts(
        self, monkeypatch, tmp_path
    ):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr("utils.get_git_sha", lambda: "abc123")
        monkeypatch.setattr("utils.get_git_dirty", lambda: False)

        output_dir = tmp_path / "data" / "shared"

        monkeypatch.setattr("utils.sys.argv", ["scripts/run_intervention.py"])
        monkeypatch.setattr(
            "utils.sys.orig_argv",
            ["uv", "run", "python", "scripts/run_intervention.py"],
            raising=False,
        )
        first = start_run_provenance(
            args={},
            primary_target=output_dir,
            output_targets=[output_dir],
            primary_target_is_dir=True,
        )
        finish_run_provenance(first, "completed")

        monkeypatch.setattr("utils.sys.argv", ["scripts/evaluate_intervention.py"])
        monkeypatch.setattr(
            "utils.sys.orig_argv",
            ["uv", "run", "python", "scripts/evaluate_intervention.py"],
            raising=False,
        )
        second = start_run_provenance(
            args={},
            primary_target=output_dir,
            output_targets=[output_dir],
            primary_target_is_dir=True,
        )
        finish_run_provenance(second, "completed")

        assert (output_dir / "run_intervention.provenance.json").exists()
        assert (output_dir / "evaluate_intervention.provenance.json").exists()

    def test_finish_run_provenance_is_noop_for_missing_handle(self):
        finish_run_provenance(None, "completed")


class TestProvenanceExceptionHelpers:
    def test_marks_interrupt_signals_as_interrupted(self):
        assert provenance_status_for_exception(KeyboardInterrupt()) == "interrupted"
        assert provenance_status_for_exception(SystemExit(130)) == "interrupted"

    def test_marks_regular_exceptions_as_failed(self):
        assert provenance_status_for_exception(RuntimeError("boom")) == "failed"

    def test_formats_error_messages_with_type_and_detail(self):
        assert provenance_error_message(RuntimeError("boom")) == "RuntimeError: boom"
        assert provenance_error_message(KeyboardInterrupt()) == "KeyboardInterrupt"


class TestRemoteBootstrapDependencies:
    def test_remote_setup_script_installs_wandb(self):
        script = remote_setup_script("mistralai/Mistral-7B-Instruct-v0.3", "model-dir")
        assert "uv add " in script
        assert " wandb" in script

    def test_lambda_bootstrap_installs_wandb_in_both_env_paths(self):
        script = (REPO_ROOT / "scripts" / "infra" / "lambda-bootstrap.sh").read_text()

        assert (
            "uv add torch transformers datasets accelerate scikit-learn joblib openai tqdm numpy wandb"
            in script
        )
        assert (
            "python -m pip install transformers datasets openai accelerate joblib scikit-learn wandb"
            in script
        )
