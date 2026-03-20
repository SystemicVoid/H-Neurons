import os
import json
import argparse
import time
from pathlib import Path
from typing import Any, List, Set, cast

import torch
from tqdm import tqdm
from datasets import load_dataset
from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Consistency Filtering with Rule or LLM Judge."
    )
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to the model for sampling"
    )
    parser.add_argument(
        "--data_path", type=str, required=True, help="Path to the TriviaQA parquet file"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="data/consistency_samples.jsonl",
        help="Output path",
    )
    parser.add_argument(
        "--device_map",
        type=str,
        default="auto",
        help="Hugging Face device_map value, e.g. 'auto' or 'cuda:0'.",
    )

    parser.add_argument(
        "--sample_num", type=int, default=10, help="Samples per question"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of questions to process",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["transformers", "openai"],
        default="transformers",
        help="Generation backend: 'transformers' for local bf16 model, 'openai' for API endpoint",
    )
    parser.add_argument(
        "--sampling_base_url",
        type=str,
        default="http://127.0.0.1:8080/v1",
        help="Base URL for sampling LLM (openai backend)",
    )
    parser.add_argument(
        "--sampling_api_key",
        type=str,
        default="not-needed",
        help="API key for sampling LLM (openai backend)",
    )

    parser.add_argument(
        "--judge_type",
        type=str,
        choices=["rule", "llm"],
        default="rule",
        help="How to judge correctness",
    )
    parser.add_argument(
        "--api_key", type=str, default=None, help="API key for LLM Judge"
    )
    parser.add_argument(
        "--base_url", type=str, default="https://api.openai.com/v1", help="API base URL"
    )
    parser.add_argument(
        "--judge_model", type=str, default="gpt-4o", help="Model name for LLM Judge"
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases run tracking",
    )

    return parser.parse_args()


# ==========================================
# Utilities
# ==========================================

from utils import (  # noqa: E402
    define_wandb_metrics,
    finish_run_provenance,
    init_wandb_run,
    log_wandb_files_as_artifact,
    normalize_answer,
    provenance_error_message,
    provenance_status_for_exception,
    summarize_numeric_values,
    start_run_provenance,
)


PROGRESS_LOG_EVERY = 25
CURATED_TABLE_LIMIT = 50
RESPONSE_PREVIEW_LIMIT = 3


def summary_path_for_output(output_path: str) -> Path:
    return Path(f"{output_path}.summary.json")


def _trim_text(value: str, limit: int = 180) -> str:
    compact = " ".join(value.split())
    if len(compact) <= limit:
        return compact
    return f"{compact[: limit - 1]}..."


def build_collect_review_row(
    qid: str,
    question: str,
    responses: list[str],
    judges: list[str],
    sample_num: int,
    *,
    error_message: str | None = None,
) -> dict[str, Any]:
    judge_counts = {
        label: judges.count(label) for label in ("true", "false", "uncertain", "error")
    }
    unique_response_count = len(
        {response.strip() for response in responses if response}
    )
    consensus_count = max(judge_counts.values(), default=0)
    consensus_rate = consensus_count / sample_num if sample_num else None

    if error_message:
        category = "errors"
    elif judge_counts["error"] > 0 or len(responses) < sample_num:
        category = "errors"
    elif judge_counts["uncertain"] > 0:
        category = "uncertain"
    elif judge_counts["true"] == sample_num:
        category = "all_correct"
    elif judge_counts["false"] == sample_num:
        category = "all_incorrect"
    else:
        category = "mixed"

    return {
        "qid": qid,
        "question": _trim_text(question, limit=220),
        "question_preview": _trim_text(question, limit=120),
        "true_count": judge_counts["true"],
        "false_count": judge_counts["false"],
        "uncertain_count": judge_counts["uncertain"],
        "error_count": judge_counts["error"] + (1 if error_message else 0),
        "unique_response_count": unique_response_count,
        "consensus_rate": consensus_rate,
        "category": category,
        "example_responses": [
            _trim_text(response, limit=140)
            for response in responses[:RESPONSE_PREVIEW_LIMIT]
        ],
        "error_message": error_message,
    }


def build_collect_triage(summary_payload: dict[str, Any]) -> tuple[str, str, str]:
    completed_qids = summary_payload["completed_qids"]
    if completed_qids == 0:
        return (
            "needs_review",
            "No completed question summaries were produced.",
            "Inspect sampling errors before using this output downstream.",
        )

    mixed_share = summary_payload["mixed_count"] / completed_qids
    unstable_share = (
        summary_payload["uncertain_count"] + summary_payload["partial_failed_qids"]
    ) / completed_qids
    if mixed_share >= 0.15 or unstable_share >= 0.05:
        return (
            "review_sampling_or_judge",
            "A large share of questions are mixed, uncertain, or failed.",
            "Inspect the mixed and uncertain/error tables before deciding the next collection run.",
        )

    return (
        "stable",
        "Most questions cluster into stable consistency regimes.",
        "Use the stable-false and mixed tables to choose follow-up extraction or judge changes.",
    )


def _wandb_table_from_rows(wandb_module, rows: list[dict[str, Any]]):
    if not rows:
        return None
    columns = list(rows[0].keys())
    data = [[row.get(column) for column in columns] for row in rows]
    return wandb_module.Table(columns=columns, data=data)


class CollectRunTracker:
    def __init__(self, args, summary_path: Path, wb_run=None, wandb_module=None):
        self.args = args
        self.summary_path = summary_path
        self.wb_run = wb_run
        self.wandb_module = wandb_module
        self.total_dataset_rows = 0
        self.progress_seen_qids = 0
        self.skipped_existing_qids = 0
        self.partial_failed_qids = 0
        self.new_completed_qids = 0
        self.completed_qids = 0
        self.all_correct_count = 0
        self.all_incorrect_count = 0
        self.mixed_count = 0
        self.uncertain_count = 0
        self.judge_label_totals = {
            label: 0 for label in ("true", "false", "uncertain", "error")
        }
        self.consensus_histogram = {str(i): 0 for i in range(self.args.sample_num + 1)}
        self.response_diversity_values: list[int] = []
        self.review_rows: list[dict[str, Any]] = []
        self.curated_rows = {
            "mixed_questions": [],
            "uncertain_or_error_questions": [],
            "stable_false_questions": [],
        }

    def set_total_dataset_rows(self, total_rows: int) -> None:
        self.total_dataset_rows = total_rows

    def seed_from_existing_file(self, output_path: str) -> Set[str]:
        processed_qids = set()
        path = Path(output_path)
        if not path.exists():
            return processed_qids
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                qid, payload = next(iter(data.items()))
                processed_qids.add(qid)
                row = build_collect_review_row(
                    qid,
                    payload.get("question", ""),
                    list(payload.get("responses", [])),
                    list(payload.get("judges", [])),
                    self.args.sample_num,
                )
                self._record_row(row, is_existing=True)
        return processed_qids

    def record_existing_skip(self) -> None:
        self.progress_seen_qids += 1
        self.skipped_existing_qids += 1
        self._log_progress_if_needed()

    def record_completed(
        self,
        qid: str,
        question: str,
        responses: list[str],
        judges: list[str],
    ) -> None:
        self.progress_seen_qids += 1
        row = build_collect_review_row(
            qid, question, responses, judges, self.args.sample_num
        )
        self._record_row(row, is_existing=False)
        self.new_completed_qids += 1
        self._log_progress_if_needed()

    def record_partial_failure(
        self,
        qid: str,
        question: str,
        responses: list[str],
        judges: list[str],
        error_message: str,
    ) -> None:
        self.progress_seen_qids += 1
        self.partial_failed_qids += 1
        row = build_collect_review_row(
            qid,
            question,
            responses,
            judges,
            self.args.sample_num,
            error_message=error_message,
        )
        self.review_rows.append(row)
        self._append_curated_row("uncertain_or_error_questions", row)
        self._log_progress_if_needed()

    def _record_row(self, row: dict[str, Any], *, is_existing: bool) -> None:
        self.review_rows.append(row)
        self.completed_qids += 1
        self.response_diversity_values.append(row["unique_response_count"])
        self.judge_label_totals["true"] += row["true_count"]
        self.judge_label_totals["false"] += row["false_count"]
        self.judge_label_totals["uncertain"] += row["uncertain_count"]
        self.judge_label_totals["error"] += row["error_count"]
        self.consensus_histogram[str(row["true_count"])] += 1

        category = row["category"]
        if category == "all_correct":
            self.all_correct_count += 1
        elif category == "all_incorrect":
            self.all_incorrect_count += 1
            self._append_curated_row("stable_false_questions", row)
        elif category == "mixed":
            self.mixed_count += 1
            self._append_curated_row("mixed_questions", row)
        else:
            self.uncertain_count += 1
            self._append_curated_row("uncertain_or_error_questions", row)

    def _append_curated_row(self, key: str, row: dict[str, Any]) -> None:
        self.curated_rows[key].append(row)

    def build_summary_payload(
        self,
        *,
        status: str,
        error: str | None = None,
    ) -> dict[str, Any]:
        triage_status, triage_note, recommended_next_action = build_collect_triage(
            {
                "completed_qids": self.completed_qids,
                "mixed_count": self.mixed_count,
                "uncertain_count": self.uncertain_count,
                "partial_failed_qids": self.partial_failed_qids,
            }
        )
        return {
            "schema_version": 1,
            "status": status,
            "dataset_rows_considered": self.total_dataset_rows,
            "processed_qids": self.progress_seen_qids,
            "skipped_existing_qids": self.skipped_existing_qids,
            "completed_qids": self.completed_qids,
            "new_completed_qids": self.new_completed_qids,
            "partial_failed_qids": self.partial_failed_qids,
            "all_correct_count": self.all_correct_count,
            "all_incorrect_count": self.all_incorrect_count,
            "mixed_count": self.mixed_count,
            "uncertain_count": self.uncertain_count,
            "judge_label_totals": self.judge_label_totals,
            "response_diversity": summarize_numeric_values(
                self.response_diversity_values
            ),
            "consensus_true_count_histogram": self.consensus_histogram,
            "backend": self.args.backend,
            "judge_type": self.args.judge_type,
            "sample_num": self.args.sample_num,
            "triage_status": triage_status,
            "triage_note": triage_note,
            "recommended_next_action": recommended_next_action,
            "curated_examples": {
                "mixed_questions": sorted(
                    self.curated_rows["mixed_questions"],
                    key=lambda row: (
                        abs(0.5 - (row["consensus_rate"] or 0.0)),
                        -row["unique_response_count"],
                    ),
                )[:CURATED_TABLE_LIMIT],
                "uncertain_or_error_questions": sorted(
                    self.curated_rows["uncertain_or_error_questions"],
                    key=lambda row: (
                        row["error_count"] + row["uncertain_count"],
                        row["unique_response_count"],
                    ),
                    reverse=True,
                )[:CURATED_TABLE_LIMIT],
                "stable_false_questions": sorted(
                    self.curated_rows["stable_false_questions"],
                    key=lambda row: (row["unique_response_count"], row["false_count"]),
                    reverse=True,
                )[:CURATED_TABLE_LIMIT],
            },
            "error": error,
        }

    def write_summary(self, *, status: str, error: str | None = None) -> dict[str, Any]:
        payload = self.build_summary_payload(status=status, error=error)
        self.summary_path.write_text(
            f"{json.dumps(payload, indent=2, sort_keys=True)}\n", encoding="utf-8"
        )
        return payload

    def _log_progress_if_needed(self) -> None:
        if self.progress_seen_qids == 0:
            return
        if self.progress_seen_qids % PROGRESS_LOG_EVERY != 0:
            return
        self.log_progress()
        self.write_summary(status="running")

    def log_progress(self) -> None:
        if self.wb_run is None or self.wandb_module is None:
            return
        completed = max(self.completed_qids, 1)
        mixed_share = self.mixed_count / completed
        uncertain_share = self.uncertain_count / completed
        self.wandb_module.log(
            {
                "progress/processed_items": self.progress_seen_qids,
                "progress/completed_qids": self.completed_qids,
                "progress/skipped_existing_qids": self.skipped_existing_qids,
                "progress/partial_failed_qids": self.partial_failed_qids,
                "distribution/all_correct_share": self.all_correct_count / completed,
                "distribution/all_incorrect_share": self.all_incorrect_count
                / completed,
                "distribution/mixed_share": mixed_share,
                "distribution/uncertain_share": uncertain_share,
            }
        )


def load_existing_qids(path: str) -> Set[str]:
    if not os.path.exists(path):
        return set()
    qids = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line)
                qids.update(data.keys())
            except json.JSONDecodeError:
                continue
    return qids


# ==========================================
# Consistency Sampler with Multi-Judge Support
# ==========================================


class ConsistencySampler:
    def __init__(self, args, tracker: CollectRunTracker | None = None):
        self.args = args
        self.backend = args.backend
        self.tracker = tracker

        # 1. Init Sampling LLM
        if self.backend == "transformers":
            self.tokenizer = AutoTokenizer.from_pretrained(args.model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                args.model_path,
                torch_dtype=torch.bfloat16,
                device_map=args.device_map,
            )
            self.model.eval()
        else:
            self.sampling_client = OpenAI(
                api_key=args.sampling_api_key,
                base_url=args.sampling_base_url,
            )

        # 2. Init Judge Client (if needed)
        self.judge_client = None
        if args.judge_type == "llm":
            if not args.api_key:
                raise ValueError("API Key is required for LLM Judge.")
            self.judge_client = OpenAI(api_key=args.api_key, base_url=args.base_url)

    def rule_judge(self, response: str, norm_gts: List[str]) -> str:
        """Simple string matching judge."""
        norm_res = normalize_answer(response)
        for gt in norm_gts:
            if gt and gt in norm_res:
                return "true"
        return "false"

    def llm_judge(self, question: str, response: str, answer_list: List[str]) -> str:
        """LLM based judge with retry logic."""
        prompt = (
            f"Question: {question}\n"
            f"Response: {response}\n"
            f"Correct Answers: {answer_list}\n"
            f"Please judge whether the response is correct or not. "
            f"Return 't' if the response is correct, and 'f' if the response is incorrect. "
            f"Don't add any additional information."
        )

        for attempt in range(5):
            try:
                assert self.judge_client is not None
                completion = self.judge_client.chat.completions.create(
                    model=self.args.judge_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                )
                content = completion.choices[0].message.content or ""
                res = content.strip().lower()
                if "t" in res:
                    return "true"
                if "f" in res:
                    return "false"
            except Exception as e:
                print(f"Judge API failed (attempt {attempt + 1}): {e}")
                time.sleep(1)
        return "error"

    def process_data(self):
        dataset = load_dataset("parquet", data_files=self.args.data_path, split="train")
        if self.args.max_samples:
            dataset = dataset.select(range(self.args.max_samples))
        if self.tracker is not None:
            self.tracker.set_total_dataset_rows(len(dataset))
            processed_qids = self.tracker.seed_from_existing_file(self.args.output_path)
            self.tracker.write_summary(status="running")
        else:
            processed_qids = load_existing_qids(self.args.output_path)

        with open(self.args.output_path, "a", encoding="utf-8") as f:
            for item in tqdm(dataset, desc=f"Sampling ({self.args.judge_type} judge)"):
                qid = str(item.get("question_id", ""))
                if qid in processed_qids:
                    if self.tracker is not None:
                        self.tracker.record_existing_skip()
                    continue

                question = item.get("question", "")
                if not question or "answer" not in item:
                    continue

                # Get ground truth
                raw_aliases = []
                for col in ["aliases", "normalized_aliases"]:
                    val = item["answer"].get(col)
                    if val:
                        if isinstance(val, list):
                            raw_aliases.extend(val)
                        else:
                            raw_aliases.append(str(val))

                norm_gts = [normalize_answer(a) for a in set(raw_aliases) if a]
                if not norm_gts:
                    continue

                suffix = "Respond with the answer only, without any explanation."
                # Sampling
                messages = [{"role": "user", "content": f"{question.strip()} {suffix}"}]
                responses = []
                judges = []

                # Cache for LLM judge to avoid redundant API calls for the same response in 10 samples
                judge_cache = {}

                for _ in range(self.args.sample_num):
                    try:
                        if self.backend == "transformers":
                            inputs = self.tokenizer.apply_chat_template(
                                messages,
                                return_tensors="pt",
                                add_generation_prompt=True,
                            )
                            if hasattr(inputs, "input_ids"):
                                input_ids = inputs["input_ids"].to(self.model.device)
                            else:
                                input_ids = inputs.to(self.model.device)
                            with torch.no_grad():
                                output_ids = self.model.generate(
                                    input_ids,
                                    max_new_tokens=50,
                                    temperature=1.0,
                                    top_p=0.9,
                                    top_k=50,
                                    do_sample=True,
                                )
                            ans = self.tokenizer.decode(
                                output_ids[0][input_ids.shape[1] :],
                                skip_special_tokens=True,
                            ).strip()
                        else:
                            create_completion = cast(
                                Any,
                                self.sampling_client.chat.completions.create,
                            )
                            completion = create_completion(
                                model="local",
                                messages=messages,
                                temperature=1.0,
                                top_p=0.9,
                                top_k=50,
                                max_tokens=50,
                            )
                            content = completion.choices[0].message.content or ""
                            ans = content.strip()
                        responses.append(ans)

                        # 1. Uncertainty check (Rule-based pre-filter)
                        uncertain_terms = [
                            "don't know",
                            "cannot",
                            "not provided",
                            "no information",
                        ]
                        if any(term in ans.lower() for term in uncertain_terms):
                            judges.append("uncertain")
                            continue

                        # 2. Correctness check
                        if self.args.judge_type == "rule":
                            judges.append(self.rule_judge(ans, norm_gts))
                        else:
                            # Use cache to save tokens if model repeats the same answer
                            if ans not in judge_cache:
                                judge_cache[ans] = self.llm_judge(
                                    question, ans, raw_aliases
                                )
                            judges.append(judge_cache[ans])

                    except Exception as e:
                        print(f"Sampling error at {qid}: {e}")
                        if self.tracker is not None:
                            self.tracker.record_partial_failure(
                                qid,
                                question,
                                responses,
                                judges,
                                _trim_text(str(e), limit=180),
                            )
                        break

                if len(responses) < self.args.sample_num:
                    continue

                # Save record
                result = {
                    qid: {
                        "question": f"{question.strip()} {suffix}",
                        "responses": responses,
                        "judges": judges,
                        "ground_truth": list(set(raw_aliases)),
                    }
                }
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                processed_qids.add(qid)
                if self.tracker is not None:
                    self.tracker.record_completed(
                        qid,
                        f"{question.strip()} {suffix}",
                        responses,
                        judges,
                    )

                if len(processed_qids) % 10 == 0:
                    tqdm.write(
                        "Stats -> "
                        f"All-Correct: {self.tracker.all_correct_count if self.tracker else 0}, "
                        f"All-Incorrect: {self.tracker.all_incorrect_count if self.tracker else 0}"
                    )
        return self.tracker


if __name__ == "__main__":
    args = parse_args()
    summary_path = summary_path_for_output(args.output_path)
    provenance_handle = start_run_provenance(
        args,
        primary_target=args.output_path,
        output_targets=[args.output_path, summary_path],
    )
    provenance_status = "completed"
    provenance_extra = {}
    wb_run = None
    wandb = None

    try:
        if args.wandb:
            try:
                import wandb
            except ImportError as exc:
                raise ImportError(
                    "--wandb requested but wandb is not installed. "
                    "Install project dependencies with `uv sync` or add it with `uv add wandb`."
                ) from exc
            wb_run, wandb_provenance = init_wandb_run(
                wandb,
                args,
                job_type="collect_responses",
                group=f"collect_responses:{Path(args.data_path).stem}",
                tags=["collect_responses", args.backend, args.judge_type],
                config_extra={"output_summary_path": str(summary_path)},
            )
            provenance_extra["wandb"] = wandb_provenance
            define_wandb_metrics(
                wandb,
                step_metric="progress/processed_items",
                metrics=[
                    "progress/completed_qids",
                    "progress/skipped_existing_qids",
                    "progress/partial_failed_qids",
                    "distribution/all_correct_share",
                    "distribution/all_incorrect_share",
                    "distribution/mixed_share",
                    "distribution/uncertain_share",
                ],
            )

        tracker = CollectRunTracker(
            args, summary_path, wb_run=wb_run, wandb_module=wandb
        )
        sampler = ConsistencySampler(args, tracker=tracker)
        tracker = sampler.process_data()
        assert tracker is not None
        summary_payload = tracker.write_summary(status="completed")
        tracker.log_progress()
        finish_run_provenance(provenance_handle, provenance_status, provenance_extra)

        if wb_run is not None and wandb is not None:
            wb_run.summary["run_health/status"] = "completed"
            wb_run.summary["run_health/resumed"] = (
                summary_payload["skipped_existing_qids"] > 0
            )
            wb_run.summary["run_health/processed"] = summary_payload["processed_qids"]
            wb_run.summary["run_health/skipped_existing"] = summary_payload[
                "skipped_existing_qids"
            ]
            wb_run.summary["run_health/errors"] = summary_payload["partial_failed_qids"]
            wb_run.summary["run_health/output_path_count"] = 2
            wb_run.summary["run_health/triage_status"] = summary_payload[
                "triage_status"
            ]
            wb_run.summary["run_health/triage_note"] = summary_payload["triage_note"]
            wb_run.summary["recommended_next_action"] = summary_payload[
                "recommended_next_action"
            ]
            wb_run.summary["science/all_correct_share"] = (
                summary_payload["all_correct_count"] / summary_payload["completed_qids"]
                if summary_payload["completed_qids"]
                else 0.0
            )
            wb_run.summary["science/all_incorrect_share"] = (
                summary_payload["all_incorrect_count"]
                / summary_payload["completed_qids"]
                if summary_payload["completed_qids"]
                else 0.0
            )
            wb_run.summary["science/mixed_share"] = (
                summary_payload["mixed_count"] / summary_payload["completed_qids"]
                if summary_payload["completed_qids"]
                else 0.0
            )
            wandb.log(
                {
                    "tables/per_question_summary": _wandb_table_from_rows(
                        wandb, tracker.review_rows
                    ),
                    "tables/mixed_questions": _wandb_table_from_rows(
                        wandb, summary_payload["curated_examples"]["mixed_questions"]
                    ),
                    "tables/uncertain_or_error_questions": _wandb_table_from_rows(
                        wandb,
                        summary_payload["curated_examples"][
                            "uncertain_or_error_questions"
                        ],
                    ),
                    "tables/stable_false_questions": _wandb_table_from_rows(
                        wandb,
                        summary_payload["curated_examples"]["stable_false_questions"],
                    ),
                }
            )
            artifact_paths = [args.output_path, summary_path]
            if provenance_handle is not None:
                artifact_paths.append(provenance_handle["path"])
            log_wandb_files_as_artifact(
                wb_run,
                wandb,
                name=f"collect-responses-{Path(args.output_path).stem}",
                artifact_type="collection_summary",
                paths=artifact_paths,
            )
    except BaseException as exc:
        provenance_status = provenance_status_for_exception(exc)
        provenance_extra["error"] = provenance_error_message(exc)
        if "tracker" in locals() and tracker is not None:
            tracker.write_summary(
                status=provenance_status,
                error=provenance_error_message(exc),
            )
        if wb_run is not None:
            wb_run.summary["run_health/status"] = provenance_status
            wb_run.summary["run_health/triage_status"] = "needs_review"
        raise
    finally:
        if wb_run is not None and wandb is not None:
            wandb.finish()
        finish_run_provenance(provenance_handle, provenance_status, provenance_extra)
