from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import finalize_gold_expansion
import prepare_gold_expansion_queue


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _base_record(sample_id: str, alpha: float) -> dict:
    return {
        "id": sample_id,
        "alpha": alpha,
        "goal": f"goal {sample_id}",
        "category": "cat",
        "response": f"response for {sample_id}",
    }


def _binary_row(sample_id: str, alpha: float, verdict: str) -> dict:
    return {
        **_base_record(sample_id, alpha),
        "judge": verdict,
    }


def _v2_row(
    sample_id: str,
    alpha: float,
    verdict: str,
    *,
    reasoning: str = "v2 reasoning",
) -> dict:
    return {
        **_base_record(sample_id, alpha),
        "judge": "SAFE",
        "csv2": {
            "harmful_binary": verdict,
            "reasoning": reasoning,
        },
    }


def _v3_row(
    sample_id: str,
    alpha: float,
    verdict: str | None,
    *,
    error: str | None = None,
    primary_outcome: str = "deflection",
) -> dict:
    csv2: dict[str, object] = {
        "primary_outcome": primary_outcome,
        "C": 1,
        "S": 2,
        "V": 3,
        "reasoning": "v3 reasoning",
        "harmful_spans": [{"start": 0, "end": 10}],
    }
    if error is not None:
        csv2["error"] = error
        csv2["raw"] = '{"schema_version":"csv2_v3","harmful_binary":"yes"}'
    else:
        csv2["harmful_binary"] = verdict
    return {
        **_base_record(sample_id, alpha),
        "judge": "SAFE",
        "csv2": csv2,
    }


class TestPrepareGoldExpansionQueue:
    def test_build_queue_excludes_gold_and_includes_disputes(self, tmp_path):
        binary_dir = tmp_path / "binary"
        v2_dir = tmp_path / "v2"
        v3_dir = tmp_path / "v3"
        gold_path = tmp_path / "gold.jsonl"

        alpha_0_binary = [
            _binary_row("gold_skip", 0.0, "HARMFUL"),
            _binary_row("agree_h_0", 0.0, "HARMFUL"),
            _binary_row("agree_h_1", 0.0, "HARMFUL"),
            _binary_row("agree_h_2", 0.0, "HARMFUL"),
            _binary_row("agree_h_3", 1.0, "HARMFUL"),
            _binary_row("agree_s_0", 0.0, "SAFE"),
            _binary_row("agree_s_1", 1.0, "SAFE"),
            _binary_row("binary_dissent", 0.0, "HARMFUL"),
            _binary_row("v2_border", 0.0, "SAFE"),
            _binary_row("v3_parse", 1.0, "SAFE"),
        ]
        alpha_0_v2 = [
            _v2_row("gold_skip", 0.0, "yes"),
            _v2_row("agree_h_0", 0.0, "yes"),
            _v2_row("agree_h_1", 0.0, "yes"),
            _v2_row("agree_h_2", 0.0, "yes"),
            _v2_row("agree_h_3", 1.0, "yes"),
            _v2_row("agree_s_0", 0.0, "no"),
            _v2_row("agree_s_1", 1.0, "no"),
            _v2_row("binary_dissent", 0.0, "no"),
            _v2_row("v2_border", 0.0, "borderline"),
            _v2_row("v3_parse", 1.0, "yes"),
        ]
        alpha_0_v3 = [
            _v3_row("gold_skip", 0.0, "yes", primary_outcome="substantive_compliance"),
            _v3_row("agree_h_0", 0.0, "yes", primary_outcome="substantive_compliance"),
            _v3_row("agree_h_1", 0.0, "yes", primary_outcome="substantive_compliance"),
            _v3_row("agree_h_2", 0.0, "yes", primary_outcome="substantive_compliance"),
            _v3_row("agree_h_3", 1.0, "yes", primary_outcome="substantive_compliance"),
            _v3_row("agree_s_0", 0.0, "no"),
            _v3_row("agree_s_1", 1.0, "no"),
            _v3_row("binary_dissent", 0.0, "no"),
            _v3_row("v2_border", 0.0, "no"),
            _v3_row("v3_parse", 1.0, None, error="parse_failed"),
        ]

        _write_jsonl(
            binary_dir / "alpha_0.0.jsonl",
            [row for row in alpha_0_binary if row["alpha"] == 0.0],
        )
        _write_jsonl(
            binary_dir / "alpha_1.0.jsonl",
            [row for row in alpha_0_binary if row["alpha"] == 1.0],
        )
        _write_jsonl(
            v2_dir / "alpha_0.0.jsonl",
            [row for row in alpha_0_v2 if row["alpha"] == 0.0],
        )
        _write_jsonl(
            v2_dir / "alpha_1.0.jsonl",
            [row for row in alpha_0_v2 if row["alpha"] == 1.0],
        )
        _write_jsonl(
            v3_dir / "alpha_0.0.jsonl",
            [row for row in alpha_0_v3 if row["alpha"] == 0.0],
        )
        _write_jsonl(
            v3_dir / "alpha_1.0.jsonl",
            [row for row in alpha_0_v3 if row["alpha"] == 1.0],
        )
        _write_jsonl(gold_path, [{"id": "gold_skip", "alpha": 0.0}])

        queue_rows, summary = prepare_gold_expansion_queue.build_labeling_queue(
            binary_dir=binary_dir,
            v2_dir=v2_dir,
            v3_dir=v3_dir,
            gold_path=gold_path,
            alphas=(0.0, 1.0),
            seed=42,
            calibration_per_label=1,
        )

        queue_ids = {(row["id"], row["queue_kind"]) for row in queue_rows}
        assert ("gold_skip", "disagreement") not in queue_ids
        assert summary["queue_rows"] == 5
        assert summary["calibration_rows"] == 2
        assert summary["disagreement_rows"] == 3
        assert ("binary_dissent", "disagreement") in queue_ids
        assert ("v2_border", "disagreement") in queue_ids
        assert ("v3_parse", "disagreement") in queue_ids

        v3_parse = next(row for row in queue_rows if row["id"] == "v3_parse")
        assert v3_parse["disagreement_type"] == "v3_parse_failed"
        assert v3_parse["v3_verdict"] is None
        assert v3_parse["v3_error"] == "parse_failed"

    def test_calibration_sampling_is_deterministic_and_alpha_stratified(self):
        rows = [
            {
                "id": "a0",
                "alpha": 0.0,
                "disagreement_type": "all_agree_harmful",
                "queue_kind": "disagreement",
                "calibration_consensus_label": None,
            },
            {
                "id": "a1",
                "alpha": 0.0,
                "disagreement_type": "all_agree_harmful",
                "queue_kind": "disagreement",
                "calibration_consensus_label": None,
            },
            {
                "id": "a2",
                "alpha": 0.0,
                "disagreement_type": "all_agree_harmful",
                "queue_kind": "disagreement",
                "calibration_consensus_label": None,
            },
            {
                "id": "b0",
                "alpha": 1.0,
                "disagreement_type": "all_agree_harmful",
                "queue_kind": "disagreement",
                "calibration_consensus_label": None,
            },
        ]

        first = prepare_gold_expansion_queue.sample_calibration_pool(
            rows,
            consensus_label="HARMFUL",
            total_requested=3,
            rng=__import__("random").Random(42),
        )
        second = prepare_gold_expansion_queue.sample_calibration_pool(
            rows,
            consensus_label="HARMFUL",
            total_requested=3,
            rng=__import__("random").Random(42),
        )

        assert [(row["id"], row["alpha"]) for row in first] == [
            (row["id"], row["alpha"]) for row in second
        ]
        counts = {}
        for row in first:
            counts[row["alpha"]] = counts.get(row["alpha"], 0) + 1
            assert row["queue_kind"] == "calibration"
            assert row["calibration_consensus_label"] == "HARMFUL"
        assert counts == {0.0: 2, 1.0: 1}


class TestFinalizeGoldExpansion:
    def test_finalize_ignores_stale_invalid_progress_rows_after_deduping(
        self, tmp_path
    ):
        queue_path = tmp_path / "queue.jsonl"
        progress_path = tmp_path / "progress.jsonl"
        expanded_path = tmp_path / "expanded.jsonl"
        uncertainties_path = tmp_path / "uncertainties.jsonl"
        summary_path = tmp_path / "summary.json"
        gold_path = tmp_path / "gold.jsonl"

        queue_rows = [
            {
                "id": "cal_0",
                "alpha": 0.0,
                "goal": "goal cal",
                "response": "response cal",
                "category": "cat",
                "binary_verdict": "SAFE",
                "v2_verdict": "no",
                "v2_reasoning": "r",
                "v3_verdict": "no",
                "v3_primary_outcome": "refusal",
                "v3_C": 0,
                "v3_S": 0,
                "v3_V": None,
                "v3_reasoning": "r",
                "v3_harmful_spans": [],
                "disagreement_type": "all_agree_safe",
                "queue_kind": "calibration",
                "calibration_consensus_label": "SAFE",
                "v3_error": None,
            },
            {
                "id": "dis_0",
                "alpha": 1.0,
                "goal": "goal dis",
                "response": "response dis",
                "category": "cat",
                "binary_verdict": "HARMFUL",
                "v2_verdict": "borderline",
                "v2_reasoning": "r",
                "v3_verdict": "yes",
                "v3_primary_outcome": "substantive_compliance",
                "v3_C": 3,
                "v3_S": 3,
                "v3_V": 2,
                "v3_reasoning": "r",
                "v3_harmful_spans": [],
                "disagreement_type": "v2_borderline",
                "queue_kind": "disagreement",
                "calibration_consensus_label": None,
                "v3_error": None,
            },
        ]
        progress_rows = [
            {
                "id": "cal_0",
                "alpha": 0.0,
                "label": "SAFE",
                "label_raw": "SAFE",
                "uncertainty": "low",
                "reasoning": "clear refusal",
            },
            {
                "id": "dis_0",
                "alpha": 1.0,
                "label": "SAFE",
                "label_raw": "SAFE",
                "uncertainty": "typo",
                "reasoning": "stale invalid row",
            },
            {
                "id": "dis_0",
                "alpha": 1.0,
                "label": "HARMFUL",
                "label_raw": "HARMFUL+",
                "uncertainty": "high",
                "reasoning": "corrected row",
            },
        ]

        _write_jsonl(queue_path, queue_rows)
        _write_jsonl(progress_path, progress_rows)
        _write_jsonl(gold_path, [])

        summary = finalize_gold_expansion.finalize_gold_expansion(
            queue_path=queue_path,
            progress_path=progress_path,
            gold_path=gold_path,
            expanded_path=expanded_path,
            uncertainties_path=uncertainties_path,
            summary_path=summary_path,
        )

        assert summary["status"] == "ready"
        expanded = finalize_gold_expansion.load_jsonl(expanded_path)
        dis_record = next(row for row in expanded if row["id"] == "dis_0")
        assert dis_record["label"] == "HARMFUL"
        assert dis_record["uncertainty"] == "high"

    def test_finalize_uses_last_progress_row_and_writes_uncertainties(self, tmp_path):
        queue_path = tmp_path / "queue.jsonl"
        progress_path = tmp_path / "progress.jsonl"
        expanded_path = tmp_path / "expanded.jsonl"
        uncertainties_path = tmp_path / "uncertainties.jsonl"
        summary_path = tmp_path / "summary.json"
        gold_path = tmp_path / "gold.jsonl"

        queue_rows = [
            {
                "id": "cal_0",
                "alpha": 0.0,
                "goal": "goal cal",
                "response": "response cal",
                "category": "cat",
                "binary_verdict": "SAFE",
                "v2_verdict": "no",
                "v2_reasoning": "r",
                "v3_verdict": "no",
                "v3_primary_outcome": "refusal",
                "v3_C": 0,
                "v3_S": 0,
                "v3_V": None,
                "v3_reasoning": "r",
                "v3_harmful_spans": [],
                "disagreement_type": "all_agree_safe",
                "queue_kind": "calibration",
                "calibration_consensus_label": "SAFE",
                "v3_error": None,
            },
            {
                "id": "dis_0",
                "alpha": 1.0,
                "goal": "goal dis",
                "response": "response dis",
                "category": "cat",
                "binary_verdict": "HARMFUL",
                "v2_verdict": "borderline",
                "v2_reasoning": "r",
                "v3_verdict": "yes",
                "v3_primary_outcome": "substantive_compliance",
                "v3_C": 3,
                "v3_S": 3,
                "v3_V": 2,
                "v3_reasoning": "r",
                "v3_harmful_spans": [],
                "disagreement_type": "v2_borderline",
                "queue_kind": "disagreement",
                "calibration_consensus_label": None,
                "v3_error": None,
            },
        ]
        progress_rows = [
            {
                "id": "cal_0",
                "alpha": 0.0,
                "label": "SAFE",
                "label_raw": "SAFE",
                "uncertainty": "low",
                "reasoning": "clear refusal",
            },
            {
                "id": "dis_0",
                "alpha": 1.0,
                "label": "SAFE",
                "label_raw": "SAFE",
                "uncertainty": "medium",
                "reasoning": "first pass",
            },
            {
                "id": "dis_0",
                "alpha": 1.0,
                "label": "HARMFUL",
                "label_raw": "HARMFUL+",
                "uncertainty": "high",
                "reasoning": "final call",
            },
        ]

        _write_jsonl(queue_path, queue_rows)
        _write_jsonl(progress_path, progress_rows)
        _write_jsonl(gold_path, [])

        summary = finalize_gold_expansion.finalize_gold_expansion(
            queue_path=queue_path,
            progress_path=progress_path,
            gold_path=gold_path,
            expanded_path=expanded_path,
            uncertainties_path=uncertainties_path,
            summary_path=summary_path,
        )

        assert summary["status"] == "ready"
        expanded = finalize_gold_expansion.load_jsonl(expanded_path)
        assert len(expanded) == 2
        dis_record = next(row for row in expanded if row["id"] == "dis_0")
        assert dis_record["label"] == "HARMFUL"
        assert dis_record["label_raw"] == "HARMFUL+"
        assert dis_record["label_source"] == "codex-agent"

        uncertainties = finalize_gold_expansion.load_jsonl(uncertainties_path)
        assert [row["id"] for row in uncertainties] == ["dis_0"]

    def test_finalize_counts_borderline_labels_in_v2_agreement_summary(self, tmp_path):
        queue_path = tmp_path / "queue.jsonl"
        progress_path = tmp_path / "progress.jsonl"
        expanded_path = tmp_path / "expanded.jsonl"
        uncertainties_path = tmp_path / "uncertainties.jsonl"
        summary_path = tmp_path / "summary.json"
        gold_path = tmp_path / "gold.jsonl"

        queue_rows = [
            {
                "id": "cal_0",
                "alpha": 0.0,
                "goal": "goal cal",
                "response": "response cal",
                "category": "cat",
                "binary_verdict": "SAFE",
                "v2_verdict": "no",
                "v2_reasoning": "r",
                "v3_verdict": "no",
                "v3_primary_outcome": "refusal",
                "v3_C": 0,
                "v3_S": 0,
                "v3_V": None,
                "v3_reasoning": "r",
                "v3_harmful_spans": [],
                "disagreement_type": "all_agree_safe",
                "queue_kind": "calibration",
                "calibration_consensus_label": "SAFE",
                "v3_error": None,
            },
            {
                "id": "dis_0",
                "alpha": 1.0,
                "goal": "goal dis",
                "response": "response dis",
                "category": "cat",
                "binary_verdict": "SAFE",
                "v2_verdict": "borderline",
                "v2_reasoning": "r",
                "v3_verdict": "no",
                "v3_primary_outcome": "refusal",
                "v3_C": 0,
                "v3_S": 0,
                "v3_V": None,
                "v3_reasoning": "r",
                "v3_harmful_spans": [],
                "disagreement_type": "v2_borderline",
                "queue_kind": "disagreement",
                "calibration_consensus_label": None,
                "v3_error": None,
            },
        ]
        progress_rows = [
            {
                "id": "cal_0",
                "alpha": 0.0,
                "label": "SAFE",
                "label_raw": "SAFE",
                "uncertainty": "low",
                "reasoning": "clear refusal",
            },
            {
                "id": "dis_0",
                "alpha": 1.0,
                "label": "BORDERLINE",
                "label_raw": "BORDERLINE",
                "uncertainty": "medium",
                "reasoning": "borderline call",
            },
        ]

        _write_jsonl(queue_path, queue_rows)
        _write_jsonl(progress_path, progress_rows)
        _write_jsonl(gold_path, [])

        summary = finalize_gold_expansion.finalize_gold_expansion(
            queue_path=queue_path,
            progress_path=progress_path,
            gold_path=gold_path,
            expanded_path=expanded_path,
            uncertainties_path=uncertainties_path,
            summary_path=summary_path,
        )

        assert summary["status"] == "ready"
        v2_summary = summary["per_evaluator_agreement_on_disagreement_rows"]["csv2_v2"]
        assert v2_summary["agree"] == 1
        assert v2_summary["total"] == 1
        assert v2_summary["excluded_borderline_labels"] == 0

    def test_finalize_blocks_when_calibration_disagreements_exceed_threshold(
        self, tmp_path
    ):
        queue_path = tmp_path / "queue.jsonl"
        progress_path = tmp_path / "progress.jsonl"
        expanded_path = tmp_path / "expanded.jsonl"
        uncertainties_path = tmp_path / "uncertainties.jsonl"
        summary_path = tmp_path / "summary.json"
        gold_path = tmp_path / "gold.jsonl"

        queue_rows = []
        progress_rows = []
        for idx in range(3):
            queue_rows.append(
                {
                    "id": f"cal_{idx}",
                    "alpha": 0.0,
                    "goal": "goal",
                    "response": "response",
                    "category": "cat",
                    "binary_verdict": "SAFE",
                    "v2_verdict": "no",
                    "v2_reasoning": "r",
                    "v3_verdict": "no",
                    "v3_primary_outcome": "refusal",
                    "v3_C": 0,
                    "v3_S": 0,
                    "v3_V": None,
                    "v3_reasoning": "r",
                    "v3_harmful_spans": [],
                    "disagreement_type": "all_agree_safe",
                    "queue_kind": "calibration",
                    "calibration_consensus_label": "SAFE",
                    "v3_error": None,
                }
            )
            progress_rows.append(
                {
                    "id": f"cal_{idx}",
                    "alpha": 0.0,
                    "label": "HARMFUL",
                    "label_raw": "HARMFUL",
                    "uncertainty": "high",
                    "reasoning": "disagree",
                }
            )

        _write_jsonl(queue_path, queue_rows)
        _write_jsonl(progress_path, progress_rows)
        _write_jsonl(gold_path, [])
        expanded_path.write_text("stale expanded\n", encoding="utf-8")
        uncertainties_path.write_text("stale uncertainties\n", encoding="utf-8")

        summary = finalize_gold_expansion.finalize_gold_expansion(
            queue_path=queue_path,
            progress_path=progress_path,
            gold_path=gold_path,
            expanded_path=expanded_path,
            uncertainties_path=uncertainties_path,
            summary_path=summary_path,
        )

        assert summary["status"] == "blocked_calibration"
        assert not expanded_path.exists()
        assert not uncertainties_path.exists()
        written_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        assert written_summary["calibration"]["n_disagree"] == 3

    def test_finalize_blocks_when_gold_overlap_disagrees_more_than_once(self, tmp_path):
        queue_path = tmp_path / "queue.jsonl"
        progress_path = tmp_path / "progress.jsonl"
        expanded_path = tmp_path / "expanded.jsonl"
        uncertainties_path = tmp_path / "uncertainties.jsonl"
        summary_path = tmp_path / "summary.json"
        gold_path = tmp_path / "gold.jsonl"

        queue_rows = []
        progress_rows = []
        gold_rows = []
        for idx in range(2):
            queue_rows.append(
                {
                    "id": f"g_{idx}",
                    "alpha": 0.0,
                    "goal": "goal",
                    "response": "response",
                    "category": "cat",
                    "binary_verdict": "SAFE",
                    "v2_verdict": "no",
                    "v2_reasoning": "r",
                    "v3_verdict": "no",
                    "v3_primary_outcome": "refusal",
                    "v3_C": 0,
                    "v3_S": 0,
                    "v3_V": None,
                    "v3_reasoning": "r",
                    "v3_harmful_spans": [],
                    "disagreement_type": "two_vs_one_binary_dissents",
                    "queue_kind": "disagreement",
                    "calibration_consensus_label": None,
                    "v3_error": None,
                }
            )
            progress_rows.append(
                {
                    "id": f"g_{idx}",
                    "alpha": 0.0,
                    "label": "HARMFUL",
                    "label_raw": "HARMFUL",
                    "uncertainty": "medium",
                    "reasoning": "disagree with gold",
                }
            )
            gold_rows.append(
                {
                    "id": f"g_{idx}",
                    "alpha": 0.0,
                    "human_label": "SAFE",
                    "human_label_raw": "SAFE",
                }
            )

        _write_jsonl(queue_path, queue_rows)
        _write_jsonl(progress_path, progress_rows)
        _write_jsonl(gold_path, gold_rows)
        expanded_path.write_text("stale expanded\n", encoding="utf-8")
        uncertainties_path.write_text("stale uncertainties\n", encoding="utf-8")

        summary = finalize_gold_expansion.finalize_gold_expansion(
            queue_path=queue_path,
            progress_path=progress_path,
            gold_path=gold_path,
            expanded_path=expanded_path,
            uncertainties_path=uncertainties_path,
            summary_path=summary_path,
        )

        assert summary["status"] == "blocked_gold_overlap"
        assert not expanded_path.exists()
        assert not uncertainties_path.exists()
        assert summary["overlap_with_existing_gold"]["n_disagree"] == 2
