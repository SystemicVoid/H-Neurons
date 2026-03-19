"""Regression tests for W&B safety and remote bootstrap wiring."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

# scripts/ uses flat sibling imports; add relevant dirs for test discovery.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "infra"))

from lambda_bootstrap_and_start_step1 import remote_setup_script  # noqa: E402
from utils import get_git_sha, sanitize_run_config  # noqa: E402


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
