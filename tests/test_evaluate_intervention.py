"""Regression tests for scripts/evaluate_intervention.py."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import openai_batch
from evaluate_intervention import (
    build_alpha_batch_custom_id,
    build_alpha_file_path,
    evaluate_all_batch,
)


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


class TestAlphaLabelHelpers:
    def test_preserves_micro_beta_precision_in_paths_and_batch_ids(self, tmp_path):
        assert build_alpha_file_path(str(tmp_path), 0.0).endswith("alpha_0.0.jsonl")
        assert build_alpha_file_path(str(tmp_path), 0.02).endswith("alpha_0.02.jsonl")
        assert build_alpha_batch_custom_id(0.0, 7) == "a0.0_i7"
        assert build_alpha_batch_custom_id(0.02, 7) == "a0.02_i7"
        assert build_alpha_batch_custom_id(0.0, 7) != build_alpha_batch_custom_id(
            0.02, 7
        )


class TestEvaluateAllBatch:
    def test_evaluates_micro_beta_files_without_aliasing(self, tmp_path, monkeypatch):
        alpha_0_path = tmp_path / "alpha_0.0.jsonl"
        alpha_002_path = tmp_path / "alpha_0.02.jsonl"
        _write_jsonl(alpha_0_path, [{"id": "base", "question": "q0", "response": "r0"}])
        _write_jsonl(
            alpha_002_path,
            [{"id": "micro", "question": "q1", "response": "r1"}],
        )

        captured_custom_ids: list[str] = []

        def fake_build_chat_request(
            custom_id: str, model: str, messages: list[dict], **kwargs
        ):
            captured_custom_ids.append(custom_id)
            return {
                "custom_id": custom_id,
                "model": model,
                "messages": messages,
                "kwargs": kwargs,
            }

        def fake_resume_or_submit(
            client, batch_requests, state_path, metadata, max_enqueued_tokens
        ):
            return {
                request["custom_id"]: {
                    "custom_id": request["custom_id"],
                    "response": {
                        "status_code": 200,
                        "body": {"choices": [{"message": {"content": "ACCEPTED"}}]},
                    },
                }
                for request in batch_requests
            }

        monkeypatch.setattr(openai_batch, "build_chat_request", fake_build_chat_request)
        monkeypatch.setattr(openai_batch, "resume_or_submit", fake_resume_or_submit)

        evaluate_all_batch(
            str(tmp_path),
            [0.0, 0.02],
            "falseqa",
            client=object(),
            judge_model="gpt-4o",
        )

        assert captured_custom_ids == ["a0.0_i0", "a0.02_i0"]

        alpha_0_records = [
            json.loads(line) for line in alpha_0_path.read_text().splitlines()
        ]
        alpha_002_records = [
            json.loads(line) for line in alpha_002_path.read_text().splitlines()
        ]

        assert alpha_0_records[0]["judge"] == "ACCEPTED"
        assert alpha_0_records[0]["compliance"] is True
        assert alpha_002_records[0]["judge"] == "ACCEPTED"
        assert alpha_002_records[0]["compliance"] is True
