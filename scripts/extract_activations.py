import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, cast

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import (
    define_wandb_metrics,
    finish_run_provenance,
    init_wandb_run,
    log_wandb_files_as_artifact,
    provenance_error_message,
    provenance_status_for_exception,
    summarize_numeric_values,
    start_run_provenance,
)


PROGRESS_LOG_EVERY = 25
CURATED_MISSING_LIMIT = 50


def unwrap_chat_template_output(chat_template_output):
    """Handle transformers returning either a tensor or a BatchEncoding."""
    if hasattr(chat_template_output, "input_ids"):
        return chat_template_output["input_ids"]
    return chat_template_output


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract CETT activations for multiple token positions."
    )
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument(
        "--input_path", type=str, required=True, help="Path to answer_tokens.jsonl"
    )
    parser.add_argument(
        "--train_ids_path", type=str, required=True, help="Path to train_qids.json"
    )
    parser.add_argument(
        "--output_root",
        type=str,
        required=True,
        help="Root directory for saving .npy files",
    )
    parser.add_argument(
        "--device_map",
        type=str,
        default="auto",
        help="Hugging Face device_map value, e.g. 'auto' or 'cuda:0'.",
    )

    # Extraction Parameters
    parser.add_argument(
        "--locations",
        nargs="+",
        default=["answer_tokens"],
        choices=["input", "output", "answer_tokens", "all_except_answer_tokens"],
        help="List of positions to extract activations from",
    )
    parser.add_argument("--method", type=str, choices=["mean", "max"], default="mean")
    parser.add_argument("--use_mag", action="store_true", default=True)
    parser.add_argument("--use_abs", action="store_true", default=True)
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases run tracking",
    )
    return parser.parse_args()


class CETTManager:
    def __init__(self, model):
        self.model = model
        self.activations = []  # Input to down_proj (neuron activations)
        self.output_norms = []  # Layer output norms for normalization
        self.hooks = []
        self._register_hooks()
        self.weight_norms = self._get_weight_norms()

    def _register_hooks(self):
        def hook_fn(module, input, output):
            self.activations.append(input[0].detach())
            self.output_norms.append(torch.norm(output.detach(), dim=-1, keepdim=True))

        for name, module in self.model.named_modules():
            if "down_proj" in name:
                self.hooks.append(module.register_forward_hook(hook_fn))

    def _get_weight_norms(self):
        norms = []
        for name, module in self.model.named_modules():
            if "down_proj" in name:
                norms.append(torch.norm(module.weight.data, dim=0))
        return torch.stack(norms).to(self.model.device)

    def clear(self):
        self.activations.clear()
        self.output_norms.clear()

    def get_cett_tensor(self, use_abs=True, use_mag=True):
        """Returns tensor of shape [layers, tokens, neurons]"""
        self.activations = [
            act.squeeze(0) if act.dim() == 3 and act.size(0) == 1 else act
            for act in self.activations
        ]
        self.output_norms = [
            norm.squeeze(0) if norm.dim() == 3 and norm.size(0) == 1 else norm
            for norm in self.output_norms
        ]

        acts = (
            torch.stack(self.activations).transpose(0, 1).to(self.model.device)
        )  # [tokens, layers, neurons]
        norms = (
            torch.stack(self.output_norms).transpose(0, 1).to(self.model.device)
        )  # [tokens, layers, 1]

        if use_abs:
            acts = torch.abs(acts)
        if use_mag:
            acts = acts * self.weight_norms.unsqueeze(0)

        return (acts / (norms + 1e-8)).transpose(0, 1)  # [layers, tokens, neurons]


def get_region_indices(
    full_ids: torch.Tensor,
    tokenizer,
    question: str,
    response: str,
    answer_tokens: List[str],
) -> Dict[str, Optional[Tuple[int, int]]]:
    """Identify token indices for different sequence regions."""
    full_tokens = [tokenizer.decode([tid]) for tid in full_ids[0]]
    answer_tokens = [
        token.replace("▁", " ").replace("Ġ", " ") for token in answer_tokens
    ]  # Normalize to tokenizer format

    # 1. Identify Input Region (User Prompt)
    user_ids = unwrap_chat_template_output(
        tokenizer.apply_chat_template(
            [{"role": "user", "content": question}], return_tensors="pt"
        )
    )
    input_len = user_ids.shape[1] - 1  # Exclude potential separator/header

    # 2. Identify Output Region (Assistant Response)
    # Usually starts after the assistant header
    output_start = input_len + 1
    output_end = len(full_tokens) - 1  # Exclude EOS

    # 3. Identify Answer Tokens Region
    ans_start: Optional[int] = None
    ans_end: Optional[int] = None
    m = len(answer_tokens)

    if m != 0:
        for i in range(output_start, len(full_tokens) - m + 1):
            if full_tokens[i : i + m] == answer_tokens:
                ans_start, ans_end = i, i + m
                break
        if ans_start is None:
            positions: List[int] = []
            search_start = output_start
            for token in answer_tokens:
                try:
                    position = full_tokens.index(token, search_start)
                except ValueError:
                    positions = []
                    break
                positions.append(position)
                search_start = position + 1
            if positions:
                ans_start, ans_end = positions[0], positions[-1] + 1
        if ans_start is not None:
            assert ans_end is not None

    answer_region: Optional[Tuple[int, int]] = None
    if ans_start is not None:
        assert ans_end is not None
        answer_region = (ans_start, ans_end)

    return {
        "input": (0, input_len),
        "output": (output_start, output_end),
        "answer_tokens": answer_region,
    }


def select_token_activations(
    activations: torch.Tensor,
    location: str,
    regions: Dict[str, Optional[Tuple[int, int]]],
) -> Optional[torch.Tensor]:
    """Select token activations for a named location.

    Args:
        activations: Tensor shaped [layers, tokens, features].
        location: One of input/output/answer_tokens/all_except_answer_tokens.
        regions: Region map from get_region_indices().

    Returns:
        Selected tensor shaped [layers, selected_tokens, features], or None when the
        requested region cannot be resolved.
    """
    if location in {"input", "output", "answer_tokens"}:
        indices = regions.get(location)
        if indices is None:
            return None
        start, end = indices
        return activations[:, start:end, :]

    if location == "all_except_answer_tokens":
        answer_region = regions.get("answer_tokens")
        if answer_region is None:
            return None
        ans_start, ans_end = answer_region
        return torch.cat(
            [activations[:, :ans_start, :], activations[:, ans_end:, :]],
            dim=1,
        )

    raise ValueError(f"Unsupported extraction location: {location}")


def aggregate_token_activations(activations: torch.Tensor, method: str) -> torch.Tensor:
    """Aggregate token activations across the token dimension."""
    if method == "mean":
        return activations.mean(dim=1)
    if method == "max":
        aggregated, _ = activations.max(dim=1)
        return aggregated
    raise ValueError(f"Unsupported aggregation method: {method}")


def metadata_path(output_root: str) -> Path:
    return Path(output_root) / "metadata.json"


def summary_path(output_root: str) -> Path:
    return Path(output_root) / "summary.json"


def build_metadata(args, n_target_ids: int) -> dict:
    return {
        "model_path": args.model_path,
        "input_path": args.input_path,
        "train_ids_path": args.train_ids_path,
        "locations": list(args.locations),
        "aggregation_method": args.method,
        "use_abs": args.use_abs,
        "use_mag": args.use_mag,
        "n_target_ids": n_target_ids,
    }


def ensure_metadata(output_root: str, metadata: dict) -> None:
    path = metadata_path(output_root)
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            existing = json.load(handle)
        keys_to_match = [
            "model_path",
            "input_path",
            "train_ids_path",
            "aggregation_method",
            "use_abs",
            "use_mag",
            "n_target_ids",
        ]
        mismatches = [
            key for key in keys_to_match if existing.get(key) != metadata.get(key)
        ]
        if mismatches:
            raise ValueError(
                f"Existing extraction metadata at {path} does not match this run. "
                "Use a fresh output directory or remove the stale metadata file."
            )
        merged_locations = list(
            dict.fromkeys(existing.get("locations", []) + metadata["locations"])
        )
        if merged_locations != existing.get("locations", []):
            existing["locations"] = merged_locations
            path.write_text(f"{json.dumps(existing, indent=2)}\n", encoding="utf-8")
        return
    path.write_text(f"{json.dumps(metadata, indent=2)}\n", encoding="utf-8")


def scan_existing_activation_qids(
    output_root: str, locations: list[str]
) -> dict[str, set[str]]:
    existing: dict[str, set[str]] = {}
    for location in locations:
        location_dir = Path(output_root) / location
        qids: set[str] = set()
        if location_dir.exists():
            for path in location_dir.glob("act_*.npy"):
                qids.add(path.stem.removeprefix("act_"))
        existing[location] = qids
    return existing


def load_existing_activation_summary(output_root: str) -> dict[str, object] | None:
    path = summary_path(output_root)
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def scan_existing_activation_norms(
    output_root: str, locations: list[str]
) -> dict[str, list[float]]:
    norms: dict[str, list[float]] = {}
    for location in locations:
        location_dir = Path(output_root) / location
        location_norms: list[float] = []
        if location_dir.exists():
            for path in sorted(location_dir.glob("act_*.npy")):
                try:
                    activation = np.load(path)
                except (OSError, ValueError):
                    continue
                location_norms.append(float(np.linalg.norm(activation)))
        norms[location] = location_norms
    return norms


def build_activation_triage(summary_payload: dict) -> tuple[str, str, str]:
    miss_rates = []
    for location_summary in summary_payload["per_location"].values():
        total = (
            location_summary["final_completed_count"]
            + location_summary["missing_region_count"]
        )
        if total:
            miss_rates.append(location_summary["missing_region_count"] / total)
    worst_miss_rate = max(miss_rates, default=0.0)
    if worst_miss_rate >= 0.1:
        return (
            "high_missing_regions",
            "At least one extraction location has a high unresolved token-region miss rate.",
            "Inspect the missing-region examples before trusting downstream classifier runs.",
        )
    return (
        "ready_for_downstream",
        "Extraction coverage is stable across requested locations.",
        "Use the completion and norm summaries to choose the next classifier or intervention step.",
    )


def build_activation_summary_payload(
    args,
    *,
    status: str,
    total_input_rows: int,
    n_target_ids: int,
    target_qids_seen: int,
    skipped_complete_qids: int,
    existing_qids: dict[str, set[str]],
    extracted_counts: dict[str, int],
    missing_counts: dict[str, int],
    prompt_token_counts: list[int],
    response_token_counts: list[int],
    answer_token_counts: list[int],
    selected_token_counts: dict[str, list[int]],
    activation_norms: dict[str, list[float]],
    missing_examples: list[dict[str, object]],
    prior_summary: dict[str, object] | None = None,
    error: str | None = None,
) -> dict:
    per_location = {}
    prior_per_location = (
        cast(dict[str, dict[str, object]], prior_summary.get("per_location", {}))
        if isinstance(prior_summary, dict)
        and isinstance(prior_summary.get("per_location"), dict)
        else {}
    )
    for location in args.locations:
        existing_count = len(existing_qids[location]) - extracted_counts[location]
        selected_token_summary = summarize_numeric_values(
            selected_token_counts[location]
        )
        if (
            existing_count > 0
            and not selected_token_counts[location]
            and location in prior_per_location
        ):
            prior_selected_summary = prior_per_location[location].get(
                "selected_token_count_summary"
            )
            if isinstance(prior_selected_summary, dict):
                selected_token_summary = dict(prior_selected_summary)
        per_location[location] = {
            "existing_count": max(existing_count, 0),
            "newly_extracted_count": extracted_counts[location],
            "missing_region_count": missing_counts[location],
            "final_completed_count": len(existing_qids[location]),
            "selected_token_count_summary": selected_token_summary,
            "activation_norm_summary": summarize_numeric_values(
                activation_norms[location]
            ),
        }
    payload = {
        "schema_version": 1,
        "status": status,
        "model_path": args.model_path,
        "input_path": args.input_path,
        "train_ids_path": args.train_ids_path,
        "locations": list(args.locations),
        "n_target_ids": n_target_ids,
        "total_input_rows": total_input_rows,
        "target_qids_seen": target_qids_seen,
        "skipped_complete_qids": skipped_complete_qids,
        "per_location": per_location,
        "token_span_diagnostics": {
            "prompt_token_count_summary": summarize_numeric_values(prompt_token_counts),
            "response_token_count_summary": summarize_numeric_values(
                response_token_counts
            ),
            "answer_token_count_summary": summarize_numeric_values(answer_token_counts),
        },
        "missing_region_examples": missing_examples[:CURATED_MISSING_LIMIT],
        "error": error,
    }
    triage_status, triage_note, recommended_next_action = build_activation_triage(
        payload
    )
    payload["triage_status"] = triage_status
    payload["triage_note"] = triage_note
    payload["recommended_next_action"] = recommended_next_action
    return payload


def _wandb_table_from_rows(wandb_module, rows: list[dict]):
    if not rows:
        return None
    columns = list(rows[0].keys())
    data = [[row.get(column) for column in columns] for row in rows]
    return wandb_module.Table(columns=columns, data=data)


def main():
    args = parse_args()
    output_targets = [
        args.output_root,
        metadata_path(args.output_root),
        summary_path(args.output_root),
        *[os.path.join(args.output_root, loc) for loc in args.locations],
    ]
    provenance_handle = start_run_provenance(
        args,
        primary_target=args.output_root,
        output_targets=output_targets,
        primary_target_is_dir=True,
    )
    provenance_status = "completed"
    provenance_extra = {}

    try:
        # W&B tracking (opt-in)
        wb_run = None
        wandb = None
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
                job_type="extract_activations",
                group=f"extract_activations:{Path(args.input_path).stem}",
                tags=[
                    "extract_activations",
                    Path(args.model_path).name,
                    *args.locations,
                ],
                config_extra={
                    "metadata_path": str(metadata_path(args.output_root)),
                    "summary_path": str(summary_path(args.output_root)),
                },
            )
            provenance_extra["wandb"] = wandb_provenance
            metrics = [
                "progress/target_qids_seen",
                "progress/skipped_complete_qids",
            ]
            for location in args.locations:
                metrics.extend(
                    [
                        f"coverage/{location}/final_completed_count",
                        f"coverage/{location}/missing_region_count",
                    ]
                )
            define_wandb_metrics(
                wandb,
                step_metric="progress/processed_items",
                metrics=metrics,
            )

        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            device_map=args.device_map,
        )
        cett_manager = CETTManager(model)

        with open(args.train_ids_path, "r") as f:
            id_map = json.load(f)
            target_ids = set(id_map["t"] + id_map["f"])
        print(f"Loaded {len(target_ids)} target IDs for extraction.")
        os.makedirs(args.output_root, exist_ok=True)
        ensure_metadata(args.output_root, build_metadata(args, len(target_ids)))
        existing_qids = scan_existing_activation_qids(args.output_root, args.locations)
        prior_summary = load_existing_activation_summary(args.output_root)
        extracted_counts = {location: 0 for location in args.locations}
        missing_counts = {location: 0 for location in args.locations}
        prompt_token_counts: list[int] = []
        response_token_counts: list[int] = []
        answer_token_counts: list[int] = []
        selected_token_counts = {location: [] for location in args.locations}
        activation_norms = scan_existing_activation_norms(
            args.output_root, args.locations
        )
        missing_examples: list[dict[str, object]] = []
        target_qids_seen = 0
        skipped_complete_qids = 0

        # Prepare directories
        for loc in args.locations:
            os.makedirs(os.path.join(args.output_root, loc), exist_ok=True)

        with open(args.input_path, "r", encoding="utf-8") as f:
            samples = [json.loads(line) for line in f]

        summary_payload = build_activation_summary_payload(
            args,
            status="running",
            total_input_rows=len(samples),
            n_target_ids=len(target_ids),
            target_qids_seen=target_qids_seen,
            skipped_complete_qids=skipped_complete_qids,
            existing_qids=existing_qids,
            extracted_counts=extracted_counts,
            missing_counts=missing_counts,
            prompt_token_counts=prompt_token_counts,
            response_token_counts=response_token_counts,
            answer_token_counts=answer_token_counts,
            selected_token_counts=selected_token_counts,
            activation_norms=activation_norms,
            missing_examples=missing_examples,
            prior_summary=prior_summary,
        )
        summary_path(args.output_root).write_text(
            f"{json.dumps(summary_payload, indent=2, sort_keys=True)}\n",
            encoding="utf-8",
        )
        progress_interval = max(
            PROGRESS_LOG_EVERY,
            max(1, int(np.ceil(len(target_ids) * 0.05))),
        )

        for sample_dict in tqdm(samples, desc="Processing"):
            qid = list(sample_dict.keys())[0]
            if qid not in target_ids:
                continue
            target_qids_seen += 1
            # Skip if all locations already extracted
            if all(
                os.path.exists(os.path.join(args.output_root, loc, f"act_{qid}.npy"))
                for loc in args.locations
            ):
                skipped_complete_qids += 1
                if (
                    wb_run is not None
                    and wandb is not None
                    and target_qids_seen % progress_interval == 0
                ):
                    wandb.log(
                        {
                            "progress/processed_items": target_qids_seen,
                            "progress/target_qids_seen": target_qids_seen,
                            "progress/skipped_complete_qids": skipped_complete_qids,
                        }
                    )
                continue
            data = sample_dict[qid]
            cett_manager.clear()

            # Forward Pass
            msgs = [
                {"role": "user", "content": data["question"]},
                {"role": "assistant", "content": data["response"]},
            ]
            input_ids = unwrap_chat_template_output(
                tokenizer.apply_chat_template(
                    msgs, return_tensors="pt", add_generation_prompt=False
                )
            ).to(model.device)
            with torch.no_grad():
                model(input_ids)

            cett_full = cett_manager.get_cett_tensor(
                use_abs=args.use_abs, use_mag=args.use_mag
            )
            regions = get_region_indices(
                input_ids,
                tokenizer,
                data["question"],
                data["response"],
                data["answer_tokens"],
            )
            prompt_region = regions["input"]
            output_region = regions["output"]
            answer_region = regions["answer_tokens"]
            assert prompt_region is not None
            assert output_region is not None
            prompt_token_counts.append(prompt_region[1] - prompt_region[0])
            response_token_counts.append(output_region[1] - output_region[0])
            if answer_region is not None:
                answer_token_counts.append(answer_region[1] - answer_region[0])
            for loc in args.locations:
                save_path = os.path.join(args.output_root, loc, f"act_{qid}.npy")
                if os.path.exists(save_path):
                    continue  # Resume support: skip already-extracted QIDs

                selected_cett = select_token_activations(cett_full, loc, regions)
                if selected_cett is None:
                    missing_counts[loc] += 1
                    if len(missing_examples) < CURATED_MISSING_LIMIT:
                        missing_examples.append(
                            {
                                "qid": qid,
                                "location": loc,
                                "question_preview": data["question"][:140],
                                "answer_tokens_preview": data["answer_tokens"][:8],
                                "response_preview": data["response"][:140],
                            }
                        )
                    continue

                selected_token_counts[loc].append(int(selected_cett.shape[1]))
                final_act = aggregate_token_activations(selected_cett, args.method)
                activation_norms[loc].append(float(final_act.norm().item()))
                np.save(save_path, final_act.cpu().float().numpy())
                extracted_counts[loc] += 1
                existing_qids[loc].add(qid)

            if target_qids_seen % progress_interval == 0:
                running_summary = build_activation_summary_payload(
                    args,
                    status="running",
                    total_input_rows=len(samples),
                    n_target_ids=len(target_ids),
                    target_qids_seen=target_qids_seen,
                    skipped_complete_qids=skipped_complete_qids,
                    existing_qids=existing_qids,
                    extracted_counts=extracted_counts,
                    missing_counts=missing_counts,
                    prompt_token_counts=prompt_token_counts,
                    response_token_counts=response_token_counts,
                    answer_token_counts=answer_token_counts,
                    selected_token_counts=selected_token_counts,
                    activation_norms=activation_norms,
                    missing_examples=missing_examples,
                    prior_summary=prior_summary,
                )
                summary_path(args.output_root).write_text(
                    f"{json.dumps(running_summary, indent=2, sort_keys=True)}\n",
                    encoding="utf-8",
                )
                if wb_run is not None and wandb is not None:
                    log_payload = {
                        "progress/processed_items": target_qids_seen,
                        "progress/target_qids_seen": target_qids_seen,
                        "progress/skipped_complete_qids": skipped_complete_qids,
                    }
                    for location in args.locations:
                        location_summary = running_summary["per_location"][location]
                        log_payload[f"coverage/{location}/final_completed_count"] = (
                            location_summary["final_completed_count"]
                        )
                        log_payload[f"coverage/{location}/missing_region_count"] = (
                            location_summary["missing_region_count"]
                        )
                    wandb.log(log_payload)

        summary_payload = build_activation_summary_payload(
            args,
            status="completed",
            total_input_rows=len(samples),
            n_target_ids=len(target_ids),
            target_qids_seen=target_qids_seen,
            skipped_complete_qids=skipped_complete_qids,
            existing_qids=existing_qids,
            extracted_counts=extracted_counts,
            missing_counts=missing_counts,
            prompt_token_counts=prompt_token_counts,
            response_token_counts=response_token_counts,
            answer_token_counts=answer_token_counts,
            selected_token_counts=selected_token_counts,
            activation_norms=activation_norms,
            missing_examples=missing_examples,
            prior_summary=prior_summary,
        )
        summary_path(args.output_root).write_text(
            f"{json.dumps(summary_payload, indent=2, sort_keys=True)}\n",
            encoding="utf-8",
        )
        finish_run_provenance(provenance_handle, provenance_status, provenance_extra)

        if wb_run is not None and wandb is not None:
            coverage_rows = []
            for location in args.locations:
                location_summary = summary_payload["per_location"][location]
                coverage_rows.append(
                    {
                        "location": location,
                        "existing_count": location_summary["existing_count"],
                        "newly_extracted_count": location_summary[
                            "newly_extracted_count"
                        ],
                        "missing_region_count": location_summary[
                            "missing_region_count"
                        ],
                        "final_completed_count": location_summary[
                            "final_completed_count"
                        ],
                    }
                )
            wb_run.summary["run_health/status"] = "completed"
            wb_run.summary["run_health/resumed"] = skipped_complete_qids > 0
            wb_run.summary["run_health/processed"] = target_qids_seen
            wb_run.summary["run_health/skipped_existing"] = skipped_complete_qids
            wb_run.summary["run_health/errors"] = 0
            wb_run.summary["run_health/output_path_count"] = 2 + len(args.locations)
            wb_run.summary["run_health/triage_status"] = summary_payload[
                "triage_status"
            ]
            wb_run.summary["run_health/triage_note"] = summary_payload["triage_note"]
            wb_run.summary["recommended_next_action"] = summary_payload[
                "recommended_next_action"
            ]
            wandb.log(
                {
                    "tables/coverage_by_location": _wandb_table_from_rows(
                        wandb, coverage_rows
                    ),
                    "tables/missing_region_examples": _wandb_table_from_rows(
                        wandb, summary_payload["missing_region_examples"]
                    ),
                }
            )
            artifact_paths = [
                metadata_path(args.output_root),
                summary_path(args.output_root),
            ]
            if provenance_handle is not None:
                artifact_paths.append(provenance_handle["path"])
            log_wandb_files_as_artifact(
                wb_run,
                wandb,
                name=f"extract-activations-{Path(args.output_root).name}",
                artifact_type="activation_summary",
                paths=artifact_paths,
            )

        if wb_run is not None and wandb is not None:
            wandb.finish()
    except BaseException as exc:
        provenance_status = provenance_status_for_exception(exc)
        provenance_extra["error"] = provenance_error_message(exc)
        if "samples" in locals() and "target_ids" in locals():
            failed_summary = build_activation_summary_payload(
                args,
                status=provenance_status,
                total_input_rows=len(samples),
                n_target_ids=len(target_ids),
                target_qids_seen=locals().get("target_qids_seen", 0),
                skipped_complete_qids=locals().get("skipped_complete_qids", 0),
                existing_qids=locals().get(
                    "existing_qids",
                    {location: set() for location in args.locations},
                ),
                extracted_counts=locals().get(
                    "extracted_counts",
                    {location: 0 for location in args.locations},
                ),
                missing_counts=locals().get(
                    "missing_counts",
                    {location: 0 for location in args.locations},
                ),
                prompt_token_counts=locals().get("prompt_token_counts", []),
                response_token_counts=locals().get("response_token_counts", []),
                answer_token_counts=locals().get("answer_token_counts", []),
                selected_token_counts=locals().get(
                    "selected_token_counts",
                    {location: [] for location in args.locations},
                ),
                activation_norms=locals().get(
                    "activation_norms",
                    {location: [] for location in args.locations},
                ),
                missing_examples=locals().get("missing_examples", []),
                prior_summary=locals().get("prior_summary"),
                error=provenance_error_message(exc),
            )
            summary_path(args.output_root).write_text(
                f"{json.dumps(failed_summary, indent=2, sort_keys=True)}\n",
                encoding="utf-8",
            )
        if "wb_run" in locals() and wb_run is not None:
            wb_run.summary["run_health/status"] = provenance_status
            wb_run.summary["run_health/triage_status"] = "needs_review"
        raise
    finally:
        finish_run_provenance(provenance_handle, provenance_status, provenance_extra)


if __name__ == "__main__":
    main()
