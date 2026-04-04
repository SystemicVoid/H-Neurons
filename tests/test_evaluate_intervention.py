"""Regression tests for scripts/evaluate_intervention.py."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import openai_batch
from evaluate_intervention import (
    _load_bridge_baseline_alpha_hint,
    build_alpha_batch_custom_id,
    build_alpha_file_path,
    evaluate_all_batch,
    evaluate_triviaqa_bridge_batch,
    parse_simpleqa_verdict,
)
from run_intervention import resolve_triviaqa_bridge_baseline_alpha_for_mode


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

    def test_evaluates_simpleqa_batch_and_sets_grades(self, tmp_path, monkeypatch):
        alpha_path = tmp_path / "alpha_0.0.jsonl"
        _write_jsonl(
            alpha_path,
            [
                {
                    "id": "simple",
                    "question": "Who wrote Hamlet?",
                    "reference_answer": "William Shakespeare",
                    "response": "William Shakespeare",
                }
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
                request["custom_id"]: {
                    "custom_id": request["custom_id"],
                    "response": {
                        "status_code": 200,
                        "body": {
                            "choices": [
                                {
                                    "message": {
                                        "content": '{"grade": "CORRECT", "reason": "match"}'
                                    }
                                }
                            ]
                        },
                    },
                }
                for request in batch_requests
            }

        monkeypatch.setattr(openai_batch, "build_chat_request", fake_build_chat_request)
        monkeypatch.setattr(openai_batch, "resume_or_submit", fake_resume_or_submit)

        evaluate_all_batch(
            str(tmp_path),
            [0.0],
            "simpleqa",
            client=object(),
            judge_model="gpt-4o",
        )

        records = [json.loads(line) for line in alpha_path.read_text().splitlines()]
        assert records[0]["simpleqa_grade"] == "CORRECT"
        assert records[0]["compliance"] is True


class TestBridgeBaselineHints:
    def test_sae_mode_uses_one_point_zero_noop(self):
        assert resolve_triviaqa_bridge_baseline_alpha_for_mode("sae") == 1.0

    def test_reads_explicit_baseline_alpha_from_summary(self, tmp_path):
        (tmp_path / "results.20260404T120000Z.json").write_text(
            json.dumps(
                {
                    "benchmark": "triviaqa_bridge",
                    "intervention_mode": "direction",
                    "baseline_alpha": 0.0,
                }
            ),
            encoding="utf-8",
        )

        assert _load_bridge_baseline_alpha_hint(str(tmp_path)) == 0.0

    def test_recovers_legacy_direction_baseline_from_intervention_mode(self, tmp_path):
        (tmp_path / "results.20260404T120000Z.json").write_text(
            json.dumps(
                {
                    "benchmark": "triviaqa_bridge",
                    "intervention_mode": "direction",
                }
            ),
            encoding="utf-8",
        )

        assert _load_bridge_baseline_alpha_hint(str(tmp_path)) == 0.0

    def test_recovers_legacy_sae_baseline_from_intervention_mode(self, tmp_path):
        (tmp_path / "results.20260404T120000Z.json").write_text(
            json.dumps(
                {
                    "benchmark": "triviaqa_bridge",
                    "intervention_mode": "sae",
                }
            ),
            encoding="utf-8",
        )

        assert _load_bridge_baseline_alpha_hint(str(tmp_path)) == 1.0

    def test_skips_invalid_or_unrelated_summaries(self, tmp_path):
        (tmp_path / "results.20260404T110000Z.json").write_text(
            "{invalid json",
            encoding="utf-8",
        )
        (tmp_path / "results.20260404T120000Z.json").write_text(
            json.dumps(
                {
                    "benchmark": "simpleqa",
                    "intervention_mode": "sae",
                }
            ),
            encoding="utf-8",
        )
        (tmp_path / "results.20260404T130000Z.json").write_text(
            json.dumps(
                {
                    "benchmark": "triviaqa_bridge",
                    "intervention_mode": "unknown_mode",
                }
            ),
            encoding="utf-8",
        )

        assert _load_bridge_baseline_alpha_hint(str(tmp_path)) is None


class TestEvaluateTriviaQaBridgeBatch:
    def test_match_audit_sample_is_stable_across_reruns(self, tmp_path, monkeypatch):
        alpha_path = tmp_path / "alpha_0.0.jsonl"
        _write_jsonl(
            alpha_path,
            [
                {
                    "id": f"tqa_{idx}",
                    "question": f"Question {idx}?",
                    "response": f"Answer {idx}",
                    "ground_truth_aliases": [f"Answer {idx}"],
                    "match_tier": "exact",
                    "deterministic_correct": True,
                    "attempted": True,
                }
                for idx in range(50)
            ],
        )

        submitted_batches: list[list[str]] = []

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
            submitted_batches.append(
                [request["custom_id"] for request in batch_requests]
            )
            return {
                request["custom_id"]: {
                    "custom_id": request["custom_id"],
                    "response": {
                        "status_code": 200,
                        "body": {
                            "choices": [{"message": {"content": '{"grade":"CORRECT"}'}}]
                        },
                    },
                }
                for request in batch_requests
            }

        monkeypatch.setattr(openai_batch, "build_chat_request", fake_build_chat_request)
        monkeypatch.setattr(openai_batch, "resume_or_submit", fake_resume_or_submit)

        evaluate_triviaqa_bridge_batch(
            str(tmp_path),
            [0.0],
            client=object(),
            judge_model="gpt-4o",
            audit_seed=123,
        )

        first_pass_records = [
            json.loads(line) for line in alpha_path.read_text().splitlines()
        ]
        first_audit_stats = json.loads((tmp_path / "audit_stats.json").read_text())

        assert len(submitted_batches) == 1
        assert len(submitted_batches[0]) == 30
        assert (
            sum(
                rec.get("judge_audit_type") == "match_audit"
                for rec in first_pass_records
            )
            == 30
        )

        evaluate_triviaqa_bridge_batch(
            str(tmp_path),
            [0.0],
            client=object(),
            judge_model="gpt-4o",
            audit_seed=123,
        )

        second_pass_records = [
            json.loads(line) for line in alpha_path.read_text().splitlines()
        ]
        second_audit_stats = json.loads((tmp_path / "audit_stats.json").read_text())

        assert len(submitted_batches) == 1
        assert (
            sum(
                rec.get("judge_audit_type") == "match_audit"
                for rec in second_pass_records
            )
            == 30
        )
        assert first_pass_records == second_pass_records
        assert first_audit_stats == second_audit_stats
        assert (
            first_audit_stats["pilot_gate"]["by_alpha"]["0.0"]["status"]
            == "insufficient_pool"
        )

    def test_audit_stats_include_nonmatch_gate_and_disagree_rate(
        self, tmp_path, monkeypatch
    ):
        alpha_path = tmp_path / "alpha_0.0.jsonl"
        _write_jsonl(
            alpha_path,
            [
                *[
                    {
                        "id": f"match_{idx}",
                        "question": f"Question {idx}?",
                        "response": f"Answer {idx}",
                        "ground_truth_aliases": [f"Answer {idx}"],
                        "match_tier": "exact",
                        "deterministic_correct": True,
                        "attempted": True,
                    }
                    for idx in range(50)
                ],
                *[
                    {
                        "id": f"nonmatch_{idx}",
                        "question": f"Other question {idx}?",
                        "response": f"Guess {idx}",
                        "ground_truth_aliases": [f"Gold {idx}"],
                        "match_tier": "no_match",
                        "deterministic_correct": False,
                        "attempted": True,
                    }
                    for idx in range(25)
                ],
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
            payload = {}
            for request in batch_requests:
                custom_id = request["custom_id"]
                content = '{"grade":"CORRECT"}'
                if "_nm_" in custom_id:
                    content = '{"grade":"INCORRECT"}'
                payload[custom_id] = {
                    "custom_id": custom_id,
                    "response": {
                        "status_code": 200,
                        "body": {"choices": [{"message": {"content": content}}]},
                    },
                }
            return payload

        monkeypatch.setattr(openai_batch, "build_chat_request", fake_build_chat_request)
        monkeypatch.setattr(openai_batch, "resume_or_submit", fake_resume_or_submit)

        evaluate_triviaqa_bridge_batch(
            str(tmp_path),
            [0.0],
            client=object(),
            judge_model="gpt-4o",
            audit_seed=123,
        )

        audit_stats = json.loads((tmp_path / "audit_stats.json").read_text())
        pilot_gate = audit_stats["pilot_gate"]["by_alpha"]["0.0"]

        assert audit_stats["audit_disagree_rate_nonmatches"] == 0.0
        assert audit_stats["per_alpha"]["0.0"]["audit_disagree_rate_nonmatches"] == 0.0
        assert pilot_gate["status"] == "ready"
        assert pilot_gate["completed_total_n"] == 40
        assert pilot_gate["available_match_n"] == 20
        assert pilot_gate["available_nonmatch_n"] == 20
        assert pilot_gate["passes_agreement_threshold"] is True


class TestSimpleQAVerdict:
    def test_parses_json_and_keyword_fallback(self):
        assert parse_simpleqa_verdict('{"grade": "INCORRECT"}') == "INCORRECT"
        assert (
            parse_simpleqa_verdict("NOT_ATTEMPTED because abstained") == "NOT_ATTEMPTED"
        )
