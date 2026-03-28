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
from collect_responses import (  # noqa: E402
    build_collect_review_row,
    build_collect_triage,
)
from extract_activations import (  # noqa: E402
    build_activation_summary_payload,
    build_metadata as build_activation_metadata,
    load_existing_activation_summary,
    scan_existing_activation_norms,
)
from run_intervention import (  # noqa: E402
    build_alpha_throughput_payload,
    define_run_intervention_wandb_metrics,
)
from utils import (  # noqa: E402
    define_wandb_metrics,
    finish_run_provenance,
    get_git_dirty,
    get_git_sha,
    init_wandb_run,
    log_wandb_files_as_artifact,
    provenance_error_message,
    provenance_status_for_exception,
    resolve_provenance_path,
    sanitize_run_config,
    summarize_numeric_values,
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


class TestWandbHelpers:
    class FakeArtifact:
        def __init__(self, name, type):
            self.name = name
            self.type = type
            self.files = []

        def add_file(self, path, name=None):
            self.files.append((path, name))

    class FakeRun:
        def __init__(self):
            self.summary = {}
            self.artifacts = []

        def log_artifact(self, artifact):
            self.artifacts.append(artifact)

    class FakeWandb:
        def __init__(self):
            self.init_kwargs = None
            self.defined_metrics = []
            self.Artifact = TestWandbHelpers.FakeArtifact
            self.run = TestWandbHelpers.FakeRun()

        def init(self, **kwargs):
            self.init_kwargs = kwargs
            return self.run

        def define_metric(self, name, **kwargs):
            self.defined_metrics.append((name, kwargs))

    def test_init_wandb_run_uses_group_job_type_and_redacted_config(self, monkeypatch):
        fake_wandb = self.FakeWandb()
        monkeypatch.setattr("utils.get_git_sha", lambda: "abc123")

        run, provenance = init_wandb_run(
            fake_wandb,
            args={"api_key": "secret", "model_path": "google/gemma-3-4b-it"},
            job_type="collect_responses",
            group="collect_responses:triviaqa",
            tags=["collect_responses", "rule"],
            config_extra={"sample_num": 10},
        )

        assert run is fake_wandb.run
        assert fake_wandb.init_kwargs["job_type"] == "collect_responses"
        assert fake_wandb.init_kwargs["group"] == "collect_responses:triviaqa"
        assert fake_wandb.init_kwargs["config"] == {
            "model_path": "google/gemma-3-4b-it",
            "sample_num": 10,
            "git_sha": "abc123",
        }
        assert provenance["job_type"] == "collect_responses"
        assert provenance["group"] == "collect_responses:triviaqa"

    def test_define_metrics_and_log_artifact_files(self, tmp_path):
        fake_wandb = self.FakeWandb()
        define_wandb_metrics(
            fake_wandb,
            step_metric="progress/processed_items",
            metrics=["progress/completed_qids"],
        )

        summary_path = tmp_path / "summary.json"
        summary_path.write_text("{}", encoding="utf-8")
        ignored_dir = tmp_path / "dir"
        ignored_dir.mkdir()
        log_wandb_files_as_artifact(
            fake_wandb.run,
            fake_wandb,
            name="collect-summary",
            artifact_type="collection_summary",
            paths=[summary_path, ignored_dir],
        )

        assert fake_wandb.defined_metrics == [
            ("progress/processed_items", {}),
            (
                "progress/completed_qids",
                {"step_metric": "progress/processed_items"},
            ),
        ]
        assert len(fake_wandb.run.artifacts) == 1
        assert fake_wandb.run.artifacts[0].files == [
            (str(summary_path), summary_path.name)
        ]

    def test_run_intervention_defines_sample_and_alpha_metrics(self):
        fake_wandb = self.FakeWandb()

        define_run_intervention_wandb_metrics(fake_wandb)

        assert fake_wandb.defined_metrics == [
            ("throughput/sample_idx", {}),
            (
                "throughput/sample/generate_s",
                {"step_metric": "throughput/sample_idx"},
            ),
            (
                "throughput/sample/wall_total_s",
                {"step_metric": "throughput/sample_idx"},
            ),
            (
                "throughput/sample/generated_tokens",
                {"step_metric": "throughput/sample_idx"},
            ),
            (
                "throughput/sample/prompt_tokens",
                {"step_metric": "throughput/sample_idx"},
            ),
            (
                "throughput/sample/tokens_per_s_generate",
                {"step_metric": "throughput/sample_idx"},
            ),
            (
                "throughput/sample/tokens_per_s_wall",
                {"step_metric": "throughput/sample_idx"},
            ),
            (
                "throughput/sample/hook_s",
                {"step_metric": "throughput/sample_idx"},
            ),
            (
                "throughput/sample/hook_calls",
                {"step_metric": "throughput/sample_idx"},
            ),
            (
                "throughput/sample/hook_frac_of_generate",
                {"step_metric": "throughput/sample_idx"},
            ),
            (
                "throughput/sample/hit_token_cap",
                {"step_metric": "throughput/sample_idx"},
            ),
            ("throughput/alpha_idx", {}),
            (
                "throughput/alpha/value",
                {"step_metric": "throughput/alpha_idx"},
            ),
            (
                "throughput/alpha/samples_completed",
                {"step_metric": "throughput/alpha_idx"},
            ),
            (
                "throughput/alpha/wall_total_s",
                {"step_metric": "throughput/alpha_idx"},
            ),
            (
                "throughput/alpha/samples_per_s_wall",
                {"step_metric": "throughput/alpha_idx"},
            ),
            (
                "throughput/alpha/tokens_per_s_wall",
                {"step_metric": "throughput/alpha_idx"},
            ),
            (
                "throughput/alpha/tokens_per_s_generate",
                {"step_metric": "throughput/alpha_idx"},
            ),
            (
                "throughput/alpha/mean_generated_tokens",
                {"step_metric": "throughput/alpha_idx"},
            ),
            (
                "throughput/alpha/inter_sample_gap_mean_s",
                {"step_metric": "throughput/alpha_idx"},
            ),
            (
                "throughput/alpha/hook_frac_of_generate",
                {"step_metric": "throughput/alpha_idx"},
            ),
            (
                "throughput/alpha/instrumented_coverage",
                {"step_metric": "throughput/alpha_idx"},
            ),
        ]

    def test_build_alpha_throughput_payload_is_scalar_only(self):
        payload = build_alpha_throughput_payload(
            alpha=3.0,
            alpha_idx=4,
            throughput_summary={
                "samples_completed": 12,
                "wall_total_s": 30.5,
                "samples_per_s_wall": 0.3934,
                "tokens_per_s_wall": 18.2,
                "tokens_per_s_generate": 19.7,
                "mean_generated_tokens": 555.0,
                "inter_sample_gap_mean_s": 0.021,
                "hook_frac_of_generate": 0.031,
                "instrumented_coverage": 1.0,
            },
        )

        assert payload == {
            "throughput/alpha_idx": 4,
            "throughput/alpha/value": 3.0,
            "throughput/alpha/samples_completed": 12,
            "throughput/alpha/wall_total_s": 30.5,
            "throughput/alpha/samples_per_s_wall": 0.3934,
            "throughput/alpha/tokens_per_s_wall": 18.2,
            "throughput/alpha/tokens_per_s_generate": 19.7,
            "throughput/alpha/mean_generated_tokens": 555.0,
            "throughput/alpha/inter_sample_gap_mean_s": 0.021,
            "throughput/alpha/hook_frac_of_generate": 0.031,
            "throughput/alpha/instrumented_coverage": 1.0,
        }

    def test_summarize_numeric_values_handles_empty_and_non_empty(self):
        assert summarize_numeric_values([])["count"] == 0
        summary = summarize_numeric_values([1, 2, 3, 4, 5])
        assert summary == {
            "count": 5,
            "min": 1.0,
            "max": 5.0,
            "mean": 3.0,
            "median": 3.0,
            "p95": 5.0,
        }


class TestRunProvenance:
    def test_resolve_provenance_path_for_file_and_directory(self):
        file_path = resolve_provenance_path(
            "data/run/results.json", "run_intervention", is_dir=False
        )
        dir_path = resolve_provenance_path(
            "data/run/experiment", "run_intervention", is_dir=True
        )

        assert file_path.parent == Path("data/run")
        assert file_path.name.startswith("results.json.provenance.")
        assert file_path.name.endswith(".json")
        assert dir_path.parent == Path("data/run/experiment")
        assert dir_path.name.startswith("run_intervention.provenance.")
        assert dir_path.name.endswith(".json")

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
                "--api_key",
                "sk-live-secret",
                "--sampling_api_key=sample-secret",
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

        provenance_path = handle["path"]
        payload = json.loads(provenance_path.read_text())
        assert payload["status"] == "running"
        assert payload["git_sha"] == "abc123"
        assert payload["git_dirty"] is True
        assert payload["hostname"] == "lab-host"
        assert payload["safe_env"] == {"CUDA_VISIBLE_DEVICES": "0"}
        assert payload["args"] == {"output_path": str(output_path)}
        assert payload["argv"] == [
            "uv",
            "run",
            "python",
            "scripts/collect_responses.py",
            "--api_key",
            "[REDACTED]",
            "--sampling_api_key=[REDACTED]",
            "--output_path",
            "data/out.jsonl",
        ]
        assert payload["output_targets"] == [str(output_path.resolve())]
        assert (
            payload["command"]
            == "uv run python scripts/collect_responses.py --api_key '[REDACTED]' "
            "'--sampling_api_key=[REDACTED]' --output_path data/out.jsonl"
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

        provenance_path = handle["path"]
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

        payload = json.loads(handle["path"].read_text())
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

        assert first["path"].exists()
        assert second["path"].exists()
        assert first["path"] != second["path"]

    def test_finish_run_provenance_is_noop_for_missing_handle(self):
        finish_run_provenance(None, "completed")

    def test_finish_run_provenance_is_idempotent_after_first_success(
        self, monkeypatch, tmp_path
    ):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr("utils.get_git_sha", lambda: "abc123")
        monkeypatch.setattr("utils.get_git_dirty", lambda: False)
        monkeypatch.setattr("utils.sys.argv", ["scripts/run_negative_control.py"])
        monkeypatch.setattr(
            "utils.sys.orig_argv",
            ["uv", "run", "python", "scripts/run_negative_control.py"],
            raising=False,
        )

        output_dir = tmp_path / "data" / "control"
        handle = start_run_provenance(
            args={},
            primary_target=output_dir,
            output_targets=[output_dir / "comparison_summary.json"],
            primary_target_is_dir=True,
        )
        assert handle is not None

        provenance_path = handle["path"]
        finish_run_provenance(
            handle,
            "completed",
            extra={
                "output_targets": [
                    str((output_dir / "comparison_summary.json").resolve())
                ]
            },
        )
        first_payload = json.loads(provenance_path.read_text())

        finish_run_provenance(handle, "failed", extra={"error": "should-not-override"})
        second_payload = json.loads(provenance_path.read_text())

        assert first_payload == second_payload
        assert second_payload["status"] == "completed"
        assert second_payload["completed_at_utc"] is not None
        assert "error" not in second_payload


class TestProvenanceExceptionHelpers:
    def test_marks_interrupt_signals_as_interrupted(self):
        assert provenance_status_for_exception(KeyboardInterrupt()) == "interrupted"
        assert provenance_status_for_exception(SystemExit(130)) == "interrupted"

    def test_marks_regular_exceptions_as_failed(self):
        assert provenance_status_for_exception(RuntimeError("boom")) == "failed"

    def test_formats_error_messages_with_type_and_detail(self):
        assert provenance_error_message(RuntimeError("boom")) == "RuntimeError: boom"
        assert provenance_error_message(KeyboardInterrupt()) == "KeyboardInterrupt"


class TestCollectObservabilityHelpers:
    def test_build_collect_review_row_and_triage_for_mixed_question(self):
        row = build_collect_review_row(
            "tc_1",
            "Who wrote Hamlet?",
            ["Shakespeare", "Marlowe", "Shakespeare"],
            ["true", "false", "true"],
            sample_num=3,
        )

        assert row["category"] == "mixed"
        assert row["true_count"] == 2
        assert row["false_count"] == 1
        assert row["unique_response_count"] == 2

        triage = build_collect_triage(
            {
                "completed_qids": 10,
                "mixed_count": 3,
                "uncertain_count": 1,
                "partial_failed_qids": 0,
            }
        )
        assert triage[0] == "review_sampling_or_judge"


class TestActivationObservabilityHelpers:
    def test_build_activation_metadata(self):
        class Args:
            model_path = "google/gemma-3-4b-it"
            input_path = "data/input.jsonl"
            train_ids_path = "data/train_ids.json"
            locations = ["answer_tokens", "output"]
            method = "mean"
            use_abs = True
            use_mag = True

        metadata = build_activation_metadata(Args(), 42)
        assert metadata == {
            "model_path": "google/gemma-3-4b-it",
            "input_path": "data/input.jsonl",
            "train_ids_path": "data/train_ids.json",
            "locations": ["answer_tokens", "output"],
            "aggregation_method": "mean",
            "use_abs": True,
            "use_mag": True,
            "n_target_ids": 42,
        }

    def test_build_activation_summary_payload_marks_high_missing_regions(self):
        class Args:
            model_path = "google/gemma-3-4b-it"
            input_path = "data/input.jsonl"
            train_ids_path = "data/train_ids.json"
            locations = ["answer_tokens", "output"]

        payload = build_activation_summary_payload(
            Args(),
            status="completed",
            total_input_rows=100,
            n_target_ids=20,
            target_qids_seen=20,
            skipped_complete_qids=5,
            existing_qids={"answer_tokens": {"a", "b"}, "output": {"a", "b", "c"}},
            extracted_counts={"answer_tokens": 1, "output": 1},
            missing_counts={"answer_tokens": 2, "output": 0},
            prompt_token_counts=[10, 12],
            response_token_counts=[20, 22],
            answer_token_counts=[2, 3],
            selected_token_counts={"answer_tokens": [2], "output": [20]},
            activation_norms={"answer_tokens": [1.5], "output": [4.0]},
            missing_examples=[{"qid": "tc_2", "location": "answer_tokens"}],
        )

        assert payload["per_location"]["answer_tokens"]["missing_region_count"] == 2
        assert payload["triage_status"] == "high_missing_regions"
        assert (
            "Inspect the missing-region examples" in payload["recommended_next_action"]
        )

    def test_resume_helpers_seed_existing_norms_and_prior_selected_token_summary(
        self, tmp_path
    ):
        output_root = tmp_path / "activations"
        answer_dir = output_root / "answer_tokens"
        answer_dir.mkdir(parents=True)
        npy_path = answer_dir / "act_tc_1.npy"

        import numpy as np

        np.save(npy_path, np.array([[3.0, 4.0]], dtype=np.float32))
        summary_payload = {
            "per_location": {
                "answer_tokens": {
                    "selected_token_count_summary": {
                        "count": 1,
                        "min": 2.0,
                        "max": 2.0,
                        "mean": 2.0,
                        "median": 2.0,
                        "p95": 2.0,
                    }
                }
            }
        }
        (output_root / "summary.json").write_text(
            json.dumps(summary_payload), encoding="utf-8"
        )

        class Args:
            model_path = "google/gemma-3-4b-it"
            input_path = "data/input.jsonl"
            train_ids_path = "data/train_ids.json"
            locations = ["answer_tokens"]

        loaded_summary = load_existing_activation_summary(str(output_root))
        activation_norms = scan_existing_activation_norms(
            str(output_root), ["answer_tokens"]
        )
        payload = build_activation_summary_payload(
            Args(),
            status="completed",
            total_input_rows=1,
            n_target_ids=1,
            target_qids_seen=1,
            skipped_complete_qids=1,
            existing_qids={"answer_tokens": {"tc_1"}},
            extracted_counts={"answer_tokens": 0},
            missing_counts={"answer_tokens": 0},
            prompt_token_counts=[],
            response_token_counts=[],
            answer_token_counts=[],
            selected_token_counts={"answer_tokens": []},
            activation_norms=activation_norms,
            missing_examples=[],
            prior_summary=loaded_summary,
        )

        assert payload["per_location"]["answer_tokens"][
            "selected_token_count_summary"
        ] == {
            "count": 1,
            "min": 2.0,
            "max": 2.0,
            "mean": 2.0,
            "median": 2.0,
            "p95": 2.0,
        }
        assert payload["per_location"]["answer_tokens"]["activation_norm_summary"] == {
            "count": 1,
            "min": 5.0,
            "max": 5.0,
            "mean": 5.0,
            "median": 5.0,
            "p95": 5.0,
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
