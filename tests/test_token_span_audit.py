from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import audit_token_spans  # noqa: E402


class StubTokenizer:
    token_text = {
        1: "<s>",
        2: "[INST]",
        3: " question [/INST]",
        4: "York",
        5: "</s>",
        6: " assistant",
    }

    def decode(self, token_ids):
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        return "".join(self.token_text[int(token_id)] for token_id in token_ids)

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize=True,
        add_generation_prompt=False,
        return_tensors=None,
    ):
        roles = [message["role"] for message in messages]
        if roles == ["user"]:
            ids = [1, 2, 3]
            rendered = "<s>[INST] question [/INST]"
            if add_generation_prompt:
                ids = [*ids, 6]
                rendered = f"{rendered} assistant"
        elif roles == ["user", "assistant"]:
            ids = [1, 2, 3, 4, 5]
            rendered = "<s>[INST] question [/INST]York</s>"
        else:
            raise AssertionError(f"Unexpected roles: {roles!r}")

        if not tokenize:
            return rendered
        if return_tensors == "pt":
            return torch.tensor([ids], dtype=torch.long)
        return ids


def _record(answer_tokens: list[str]) -> dict:
    return {
        "qid": "tc_1",
        "line_no": 1,
        "question": "Where was Judi Dench born?",
        "response": "York",
        "answer_tokens": answer_tokens,
        "judge": "true",
    }


def test_audit_answer_token_record_resolves_contiguous_answer_span() -> None:
    row = audit_token_spans.audit_answer_token_record(
        StubTokenizer(),
        _record(["York"]),
        window_tokens=1,
    )

    assert row["status"] == "ok"
    assert row["regions"]["answer_tokens"] == {"start": 3, "end": 4, "length": 1}
    assert row["token_counts"]["full_chat"] == 5
    assert row["token_counts"]["user_no_generation_prompt"] == 3
    assert row["token_counts"]["user_generation_prompt"] == 4
    assert row["answer_window"]["decoded_text"] == "York"
    assert row["answer_window"]["tokens"] == [" question [/INST]", "York", "</s>"]


def test_audit_answer_token_record_flags_missing_answer_span() -> None:
    row = audit_token_spans.audit_answer_token_record(
        StubTokenizer(),
        _record(["London"]),
        window_tokens=1,
    )

    assert row["status"] == "missing_answer_region"
    assert row["regions"]["answer_tokens"] is None
    assert row["answer_window"]["decoded_text"] is None


def test_load_sample_ids_accepts_qid_split_manifest(tmp_path: Path) -> None:
    manifest = tmp_path / "split.json"
    manifest.write_text(json.dumps({"t": ["tc_1"], "f": ["qz_2"]}), encoding="utf-8")

    assert audit_token_spans.load_sample_ids(manifest) == {"tc_1", "qz_2"}


def test_build_summary_marks_empty_audit_as_needs_review(tmp_path: Path) -> None:
    summary = audit_token_spans.build_summary(
        argparse.Namespace(
            model_key="gemma3_4b_it",
            model_path="google/gemma-3-4b-it",
            input_path=tmp_path / "answer_tokens.jsonl",
            output_path=tmp_path / "token_span_audit.jsonl",
            rendered_chat_path=None,
            sample_ids_path=tmp_path / "sample_ids.json",
        ),
        rows=[],
        total_rows_scanned=3,
        sample_ids={"stale_id"},
    )

    assert summary["status"] == "needs_review"
    assert summary["audited_rows"] == 0
    assert summary["total_rows_scanned"] == 3
    assert summary["sample_id_filter_count"] == 1
