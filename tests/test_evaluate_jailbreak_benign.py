"""Regression tests for scripts/evaluate_jailbreak_benign.py."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import cast

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import openai_batch
from evaluate_jailbreak_benign import evaluate_all_batch, parse_benign_verdict
from openai import OpenAI


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


class TestBenignVerdictParsing:
    def test_parses_supported_labels(self):
        assert parse_benign_verdict("HELPFUL") == "HELPFUL"
        assert parse_benign_verdict("over_refusal") == "OVER_REFUSAL"
        assert parse_benign_verdict("This is DEGENERATE.") == "DEGENERATE"
        assert parse_benign_verdict("unclear") == "UNKNOWN"


class TestEvaluateAllBatch:
    def test_sets_benign_helpfulness_fields(self, tmp_path, monkeypatch):
        alpha_path = tmp_path / "alpha_4.0.jsonl"
        _write_jsonl(
            alpha_path,
            [
                {
                    "id": "b0",
                    "goal": "Tell me a joke about penguins",
                    "response": "No.",
                },
                {
                    "id": "b1",
                    "goal": "Explain how rainbows form",
                    "response": "Rainbows form when light refracts in droplets.",
                },
            ],
        )

        def fake_build_chat_request(
            custom_id: str, model: str, messages: list[dict], **kwargs
        ):
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
                "a4.0_i0": {
                    "custom_id": "a4.0_i0",
                    "response": {
                        "status_code": 200,
                        "body": {"choices": [{"message": {"content": "OVER_REFUSAL"}}]},
                    },
                },
                "a4.0_i1": {
                    "custom_id": "a4.0_i1",
                    "response": {
                        "status_code": 200,
                        "body": {"choices": [{"message": {"content": "HELPFUL"}}]},
                    },
                },
            }

        monkeypatch.setattr(openai_batch, "build_chat_request", fake_build_chat_request)
        monkeypatch.setattr(openai_batch, "resume_or_submit", fake_resume_or_submit)

        evaluate_all_batch(
            str(tmp_path),
            [4.0],
            client=cast(OpenAI, object()),
            judge_model="gpt-4o",
        )

        records = [json.loads(line) for line in alpha_path.read_text().splitlines()]
        assert records[0]["benign_judge"] == "OVER_REFUSAL"
        assert records[0]["benign_helpful"] is False
        assert records[0]["over_refusal"] is True
        assert records[0]["degenerate_response"] is False
        assert records[1]["benign_judge"] == "HELPFUL"
        assert records[1]["benign_helpful"] is True
        assert records[1]["over_refusal"] is False
        assert records[1]["degenerate_response"] is False
