#!/usr/bin/env python3
"""Run SIMID same-item MC/open-ended ITI experiments.

Outputs are one JSONL per condition and alpha:

```
<output_dir>/<condition>/alpha_<alpha>.jsonl
```

Each row contains both interfaces for the same manifest item: option-text
likelihood, generated-letter MC, deterministic open generation and grading,
gold-vs-distractor token margins, timing, and intervention/control metadata.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
import torch

from build_simid_manifest import (
    DEFAULT_ITI_ARTIFACT,
    SCHEMA_VERSION,
    file_sha256,
    load_simid_manifest,
    validate_manifest,
)
from intervene_iti import ITIHeadScaler, load_iti_artifact
from run_intervention import (
    assistant_content_continuation_ids_from_chat_messages,
    generate_response,
    grade_triviaqa_bridge,
    load_model_and_tokenizer,
    tokenize_chat,
    triviaqa_bridge_attempted,
)
from utils import (
    extract_mc_answer,
    finish_run_provenance,
    format_alpha_label,
    get_git_sha,
    normalize_answer,
    provenance_error_message,
    provenance_status_for_exception,
    sanitize_run_config,
    start_run_provenance,
)


BASE_OUTPUT_ROOT = Path(
    "data/gemma3_4b/intervention/simid_iti_truthfulqa-paperfaithful_k12_first-3-tokens"
)
DEFAULT_MODEL_PATH = "google/gemma-3-4b-it"
DEFAULT_MANIFEST = "data/manifests/simid_truthfulqa_bridge_seed42.json"
DEFAULT_ALPHAS = [-8.0, 0.0, 4.0, 8.0]
PRIMARY_WINDOW_TOKENS = 3
OPEN_GRADING_METHOD = {
    "name": "deterministic_alias_grader",
    "implementation": "triviaqa_bridge_alias_matcher",
    "scope": "diagnostic_not_adjudicated",
    "claim_requirement": (
        "human_or_judge_adjudication_required_for_claimable_open_correctness"
    ),
}
NOOP_COMPARE_PATHS = (
    ("mc_likelihood", "full", "margin"),
    ("mc_likelihood", "avg", "margin"),
    ("mc_letter_likelihood", "chosen_letter"),
    ("mc_letter_likelihood", "chosen_is_gold"),
    ("mc_letter_likelihood", "full", "margin"),
    ("mc_letter_likelihood", "avg", "margin"),
    ("open_generation", "response"),
    ("open_grade", "correct"),
    ("open_grade", "attempted"),
    ("open_margins", "first_token", "margin"),
    ("open_margins", "first3", "margin"),
    ("open_margins", "full", "margin"),
)


@dataclass(frozen=True)
class ConditionSpec:
    name: str
    selection_strategy: str = "ranked"
    random_seed: int = 42
    direction_mode: str = "artifact"
    direction_random_seed: int | None = None
    control_type: str = "selected"


CONDITION_SPECS: dict[str, ConditionSpec] = {
    "selected": ConditionSpec(name="selected"),
    "random_head_seed1": ConditionSpec(
        name="random_head_seed1",
        selection_strategy="layer_matched_random",
        random_seed=1,
        control_type="layer_matched_random_head",
    ),
    "random_direction_seed1": ConditionSpec(
        name="random_direction_seed1",
        direction_mode="random",
        direction_random_seed=1,
        control_type="selected_heads_random_direction",
    ),
}
UNHOOKED_SPEC = ConditionSpec(name="unhooked", control_type="unhooked")


def set_reproducibility_seed(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def simid_output_id(sample_id: str, condition: str, alpha: float) -> str:
    alpha_label = format_alpha_label(alpha)
    return f"{sample_id}__{condition}__alpha_{alpha_label}"


def load_existing_sample_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            sample_id = row.get("sample_id") or row.get("id")
            if sample_id is not None:
                ids.add(str(sample_id))
    return ids


def normalize_iti_family(iti_family: str) -> str:
    return iti_family if iti_family.startswith("iti_") else f"iti_{iti_family}"


def build_iti_config(
    *,
    model_path: str,
    tokenizer_path: str,
    iti_artifact_path: str,
    iti_artifact_sha256: str | None,
    iti_family: str,
    iti_k: int,
    decode_scope: str,
) -> dict[str, Any]:
    return {
        "model_path": model_path,
        "tokenizer_path": tokenizer_path,
        "iti_artifact_path": iti_artifact_path,
        "iti_artifact_sha256": iti_artifact_sha256,
        "family": iti_family,
        "effective_family": normalize_iti_family(iti_family),
        "k": int(iti_k),
        "decode_scope": decode_scope,
    }


def stable_payload_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def alpha_output_has_rows(path: Path) -> bool:
    if not path.exists():
        return False
    with path.open(encoding="utf-8") as handle:
        return any(line.strip() for line in handle)


def output_dir_has_alpha_rows(output_dir: Path) -> bool:
    return any(
        alpha_output_has_rows(path) for path in output_dir.glob("*/alpha_*.jsonl")
    )


def build_resume_contract(
    *,
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    conditions: list[ConditionSpec],
    iti_config: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "simid_resume_contract/v1",
        "manifest_rows_sha256": stable_payload_sha256(rows),
        "iti_config": iti_config,
        "conditions": [asdict(condition) for condition in conditions],
        "alphas": [float(alpha) for alpha in args.alphas],
        "generation": {
            "top_k_first_token": int(args.top_k_first_token),
            "mc_max_new_tokens": int(args.mc_max_new_tokens),
            "open_max_new_tokens": int(args.open_max_new_tokens),
        },
        "runtime": {
            "device_map": args.device_map,
            "seed": args.seed,
            "collect_debug_stats": bool(args.collect_debug_stats),
        },
    }


def build_run_config(
    *,
    args: argparse.Namespace,
    output_dir: Path,
    rows: list[dict[str, Any]],
    conditions: list[ConditionSpec],
    iti_config: dict[str, Any],
    locked_manifest_path: Path,
) -> dict[str, Any]:
    resume_contract = build_resume_contract(
        args=args,
        rows=rows,
        conditions=conditions,
        iti_config=iti_config,
    )
    return {
        "schema_version": "simid_run_config/v1",
        "args": sanitize_run_config(vars(args)),
        "manifest": args.manifest,
        "locked_manifest": str(locked_manifest_path),
        "iti_config": iti_config,
        "n_rows": len(rows),
        "conditions": [asdict(condition) for condition in conditions],
        "alphas": [float(alpha) for alpha in args.alphas],
        "output_dir": str(output_dir),
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "resume_contract": resume_contract,
        "resume_fingerprint": stable_payload_sha256(resume_contract),
    }


def changed_contract_keys(left: dict[str, Any], right: dict[str, Any]) -> list[str]:
    changed: list[str] = []
    for key in sorted(set(left) | set(right)):
        if left.get(key) != right.get(key):
            changed.append(key)
    return changed


def write_or_validate_run_config(
    *,
    path: Path,
    run_config: dict[str, Any],
    output_dir: Path,
) -> None:
    if not path.exists():
        path.write_text(
            json.dumps(run_config, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return

    existing = json.loads(path.read_text(encoding="utf-8"))
    existing_fingerprint = existing.get("resume_fingerprint")
    new_fingerprint = run_config["resume_fingerprint"]
    has_partial_outputs = output_dir_has_alpha_rows(output_dir)
    if existing_fingerprint is None:
        if has_partial_outputs:
            raise ValueError(
                f"Cannot resume {output_dir}: existing run_config.json has no "
                "resume_fingerprint, so existing alpha rows cannot be proven "
                "compatible with the requested runtime config"
            )
        path.write_text(
            json.dumps(run_config, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return

    if existing_fingerprint != new_fingerprint:
        if not has_partial_outputs:
            path.write_text(
                json.dumps(run_config, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            return
        changed = changed_contract_keys(
            existing.get("resume_contract", {}),
            run_config["resume_contract"],
        )
        raise ValueError(
            f"Cannot resume {output_dir}: requested runtime config does not match "
            "the existing run_config.json resume_fingerprint. "
            f"Changed contract keys: {changed or ['unknown']}. "
            "Archive the old run or choose a new output directory."
        )


def load_locked_manifest(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must be a JSON object")
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise ValueError(f"{path} must contain a rows list")
    validate_manifest({"schema_version": SCHEMA_VERSION, "rows": rows})
    return [dict(row) for row in rows]


def load_or_create_locked_manifest(
    *,
    source_manifest: str,
    output_dir: Path,
    max_items: int | None,
) -> list[dict[str, Any]]:
    locked_manifest_path = output_dir / "manifest.locked.json"
    if locked_manifest_path.exists():
        return load_locked_manifest(locked_manifest_path)

    rows = load_simid_manifest(source_manifest)
    if max_items is not None:
        rows = rows[:max_items]
    if not rows:
        raise ValueError("No SIMID rows selected")
    locked_manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "simid_locked_manifest/v1",
                "source_manifest": source_manifest,
                "max_items": max_items,
                "rows": rows,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return rows


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def option_letters(n_options: int) -> list[str]:
    if n_options < 1 or n_options > 26:
        raise ValueError(f"SIMID supports 1..26 MC options, got {n_options}")
    return [chr(ord("A") + idx) for idx in range(n_options)]


def build_mc_prompt(row: dict[str, Any]) -> tuple[str, list[str]]:
    options = list(row["mc_options"])
    letters = option_letters(len(options))
    rendered_options = "\n".join(
        f"{letter}. {option}" for letter, option in zip(letters, options, strict=True)
    )
    prompt = (
        f"Question: {row['question']}\n"
        f"{rendered_options}\n"
        "Answer with the letter of the best answer only."
    )
    return prompt, letters


def build_answer_prompt(row: dict[str, Any]) -> str:
    return f"Question: {row['question']}\nAnswer:"


def prompt_messages(prompt: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": prompt}]


def continuation_token_ids(
    tokenizer: Any,
    messages: list[dict[str, str]],
    prompt_ids: torch.Tensor,
    continuations: dict[str, str],
) -> dict[str, torch.Tensor]:
    return {
        name: assistant_content_continuation_ids_from_chat_messages(
            tokenizer,
            messages,
            continuation,
            prompt_input_ids=prompt_ids,
        )
        for name, continuation in continuations.items()
    }


def arm_scaler_first_decode_token(scaler: ITIHeadScaler | None) -> None:
    if scaler is not None and hasattr(scaler, "arm_first_decode_token"):
        scaler.arm_first_decode_token()


def score_continuation_logprobs_fresh_prefill(
    model: Any,
    prompt_ids: torch.Tensor,
    continuation_ids_by_name: dict[str, torch.Tensor],
    *,
    scaler: ITIHeadScaler | None,
) -> dict[str, list[float]]:
    """Score each target from a fresh prompt prefill.

    Decode-scoped ITI keeps internal token counters. Re-prefilling for each
    target prevents one continuation's token steps from shifting another
    continuation outside the first-token/first-3-token intervention window.
    """
    prompt_ids = prompt_ids.to(model.device)
    if prompt_ids.ndim == 1:
        prompt_ids = prompt_ids.unsqueeze(0)

    scores: dict[str, list[float]] = {}
    for name, continuation_ids_cpu in continuation_ids_by_name.items():
        continuation_ids = continuation_ids_cpu.to(model.device)
        if continuation_ids.ndim != 1 or continuation_ids.numel() == 0:
            raise ValueError(
                f"Continuation token IDs for {name!r} must be a non-empty 1D tensor"
            )
        arm_scaler_first_decode_token(scaler)
        with torch.inference_mode():
            prompt_outputs = model(prompt_ids, use_cache=True)
        prompt_logprobs = torch.log_softmax(prompt_outputs.logits[:, -1, :], dim=-1)
        past_key_values = prompt_outputs.past_key_values

        first_token = int(continuation_ids[0].item())
        per_token = [float(prompt_logprobs[0, first_token].item())]
        for position in range(int(continuation_ids.numel()) - 1):
            current_token = continuation_ids[position : position + 1].view(1, 1)
            next_token = int(continuation_ids[position + 1].item())
            with torch.inference_mode():
                outputs = model(
                    current_token,
                    use_cache=True,
                    past_key_values=past_key_values,
                )
            past_key_values = outputs.past_key_values
            next_logprob = torch.log_softmax(outputs.logits[:, -1, :], dim=-1)[
                0, next_token
            ]
            per_token.append(float(next_logprob.item()))
        scores[name] = per_token
    return scores


def summarize_logprob_tokens(per_token_logprobs: list[float]) -> dict[str, Any]:
    values = [float(value) for value in per_token_logprobs]
    first3 = values[:PRIMARY_WINDOW_TOKENS]
    total = float(sum(values))
    n = len(values)
    return {
        "per_token_logprobs": [round(value, 6) for value in values],
        "first_token": {
            "sum_logprob": round(values[0], 6) if values else 0.0,
            "n_tokens": 1 if values else 0,
        },
        "first3": {
            "sum_logprob": round(float(sum(first3)), 6),
            "n_tokens": min(n, PRIMARY_WINDOW_TOKENS),
        },
        "full": {
            "sum_logprob": round(total, 6),
            "avg_logprob": round(total / n, 6) if n else 0.0,
            "n_tokens": n,
        },
    }


def _best_index_by_metric(
    summaries: list[dict[str, Any]],
    candidate_indices: list[int],
    *,
    window: str,
    metric: str,
) -> int:
    if not candidate_indices:
        raise ValueError("candidate_indices cannot be empty")
    return max(
        candidate_indices,
        key=lambda idx: float(summaries[idx][window][metric]),
    )


def compute_mc_margin(
    option_summaries: list[dict[str, Any]],
    gold_option_indices: list[int],
) -> dict[str, Any]:
    """Compute gold-minus-best-incorrect margins for ordered MC target scores."""
    if not option_summaries:
        raise ValueError("option_summaries cannot be empty")
    gold = [int(idx) for idx in gold_option_indices]
    all_indices = list(range(len(option_summaries)))
    incorrect = [idx for idx in all_indices if idx not in set(gold)]
    if not gold or not incorrect:
        raise ValueError(
            "MC margin requires at least one gold and one incorrect option"
        )

    best_gold_first = _best_index_by_metric(
        option_summaries, gold, window="first_token", metric="sum_logprob"
    )
    best_wrong_first = _best_index_by_metric(
        option_summaries, incorrect, window="first_token", metric="sum_logprob"
    )
    best_gold_first3 = _best_index_by_metric(
        option_summaries, gold, window="first3", metric="sum_logprob"
    )
    best_wrong_first3 = _best_index_by_metric(
        option_summaries, incorrect, window="first3", metric="sum_logprob"
    )
    best_gold_full = _best_index_by_metric(
        option_summaries, gold, window="full", metric="sum_logprob"
    )
    best_wrong_full = _best_index_by_metric(
        option_summaries, incorrect, window="full", metric="sum_logprob"
    )
    best_gold_avg = _best_index_by_metric(
        option_summaries, gold, window="full", metric="avg_logprob"
    )
    best_wrong_avg = _best_index_by_metric(
        option_summaries, incorrect, window="full", metric="avg_logprob"
    )
    chosen_full = max(
        all_indices,
        key=lambda idx: float(option_summaries[idx]["full"]["sum_logprob"]),
    )
    chosen_avg = max(
        all_indices,
        key=lambda idx: float(option_summaries[idx]["full"]["avg_logprob"]),
    )

    def margin(
        gold_idx: int,
        wrong_idx: int,
        *,
        window: str,
        metric: str,
    ) -> float:
        return round(
            float(option_summaries[gold_idx][window][metric])
            - float(option_summaries[wrong_idx][window][metric]),
            6,
        )

    return {
        "first_token": {
            "margin": margin(
                best_gold_first,
                best_wrong_first,
                window="first_token",
                metric="sum_logprob",
            ),
            "best_gold_index": best_gold_first,
            "best_incorrect_index": best_wrong_first,
        },
        "first3": {
            "margin": margin(
                best_gold_first3,
                best_wrong_first3,
                window="first3",
                metric="sum_logprob",
            ),
            "best_gold_index": best_gold_first3,
            "best_incorrect_index": best_wrong_first3,
        },
        "full": {
            "margin": margin(
                best_gold_full,
                best_wrong_full,
                window="full",
                metric="sum_logprob",
            ),
            "best_gold_index": best_gold_full,
            "best_incorrect_index": best_wrong_full,
            "chosen_index": chosen_full,
            "chosen_is_gold": chosen_full in set(gold),
        },
        "avg": {
            "margin": margin(
                best_gold_avg,
                best_wrong_avg,
                window="full",
                metric="avg_logprob",
            ),
            "best_gold_index": best_gold_avg,
            "best_incorrect_index": best_wrong_avg,
            "chosen_index": chosen_avg,
            "chosen_is_gold": chosen_avg in set(gold),
        },
        "margin_definition": "best_gold_logprob_minus_best_incorrect_logprob",
        "length_normalized_metric": "full.avg_logprob",
    }


def _target_margin(
    target_summaries: dict[str, dict[str, Any]],
    gold_names: list[str],
    distractor_names: list[str],
    *,
    window: str,
    metric: str,
) -> dict[str, Any]:
    best_gold = max(
        gold_names,
        key=lambda name: float(target_summaries[name][window][metric]),
    )
    best_distractor = max(
        distractor_names,
        key=lambda name: float(target_summaries[name][window][metric]),
    )
    return {
        "margin": round(
            float(target_summaries[best_gold][window][metric])
            - float(target_summaries[best_distractor][window][metric]),
            6,
        ),
        "best_gold_target": best_gold,
        "best_distractor_target": best_distractor,
    }


def compute_target_margins(
    target_summaries: dict[str, dict[str, Any]],
    gold_names: list[str],
    distractor_names: list[str],
) -> dict[str, Any]:
    return {
        "first_token": _target_margin(
            target_summaries,
            gold_names,
            distractor_names,
            window="first_token",
            metric="sum_logprob",
        ),
        "first3": _target_margin(
            target_summaries,
            gold_names,
            distractor_names,
            window="first3",
            metric="sum_logprob",
        ),
        "full": _target_margin(
            target_summaries,
            gold_names,
            distractor_names,
            window="full",
            metric="sum_logprob",
        ),
        "avg": _target_margin(
            target_summaries,
            gold_names,
            distractor_names,
            window="full",
            metric="avg_logprob",
        ),
        "margin_definition": "best_gold_logprob_minus_best_distractor_logprob",
    }


def score_continuation_summaries(
    model: Any,
    tokenizer: Any,
    prompt: str,
    continuations: dict[str, str],
    *,
    scaler: ITIHeadScaler | None,
) -> tuple[dict[str, dict[str, Any]], torch.Tensor, float]:
    messages = prompt_messages(prompt)
    prompt_ids = tokenize_chat(tokenizer, messages)
    token_ids = continuation_token_ids(tokenizer, messages, prompt_ids, continuations)
    t0 = time.perf_counter()
    per_target = score_continuation_logprobs_fresh_prefill(
        model,
        prompt_ids,
        token_ids,
        scaler=scaler,
    )
    elapsed = time.perf_counter() - t0
    return (
        {
            name: summarize_logprob_tokens(per_token)
            for name, per_token in per_target.items()
        },
        prompt_ids,
        elapsed,
    )


def score_mc_likelihood(
    model: Any,
    tokenizer: Any,
    row: dict[str, Any],
    *,
    scaler: ITIHeadScaler | None,
) -> tuple[dict[str, Any], float]:
    continuations = {
        f"option_{idx}": str(option) for idx, option in enumerate(row["mc_options"])
    }
    summaries, _, elapsed = score_continuation_summaries(
        model,
        tokenizer,
        build_answer_prompt(row),
        continuations,
        scaler=scaler,
    )
    ordered_summaries = [
        summaries[f"option_{idx}"] for idx in range(len(continuations))
    ]
    margins = compute_mc_margin(ordered_summaries, list(row["gold_option_indices"]))
    return {
        "endpoint": "option_text_teacher_forced",
        "prompt": build_answer_prompt(row),
        "options": row["mc_options"],
        "gold_option_indices": row["gold_option_indices"],
        "option_scores": ordered_summaries,
        **margins,
    }, elapsed


def score_mc_letters(
    model: Any,
    tokenizer: Any,
    row: dict[str, Any],
    *,
    scaler: ITIHeadScaler | None,
) -> tuple[dict[str, Any], torch.Tensor, float]:
    mc_prompt, letters = build_mc_prompt(row)
    continuations = {letter: letter for letter in letters}
    summaries, prompt_ids, elapsed = score_continuation_summaries(
        model,
        tokenizer,
        mc_prompt,
        continuations,
        scaler=scaler,
    )
    letter_logprobs = {
        letter: float(summaries[letter]["full"]["sum_logprob"]) for letter in letters
    }
    ordered_summaries = [summaries[letter] for letter in letters]
    margins = compute_mc_margin(ordered_summaries, list(row["gold_option_indices"]))
    chosen_letter = max(letters, key=lambda letter: letter_logprobs[letter])
    chosen_index = letters.index(chosen_letter)
    gold_indices = {int(idx) for idx in row["gold_option_indices"]}
    return (
        {
            "endpoint": "letter_teacher_forced",
            "prompt": mc_prompt,
            "valid_letters": letters,
            "letter_logprobs": {
                letter: round(float(value), 6)
                for letter, value in letter_logprobs.items()
            },
            "chosen_letter": chosen_letter,
            "chosen_index": chosen_index,
            "chosen_is_gold": chosen_index in gold_indices,
            "letter_scores": ordered_summaries,
            **margins,
        },
        prompt_ids,
        elapsed,
    )


def grade_open_response(row: dict[str, Any], response: str) -> dict[str, Any]:
    grade = grade_triviaqa_bridge(
        response, [str(alias) for alias in row["gold_aliases"]]
    )
    attempted = triviaqa_bridge_attempted(response)
    if grade["correct"]:
        failure_type = None
    elif not attempted:
        failure_type = "not_attempted"
    else:
        response_norm = normalize_answer(response)
        wrong_option_norms = {
            normalize_answer(option)
            for idx, option in enumerate(row["mc_options"])
            if idx not in {int(gold_idx) for gold_idx in row["gold_option_indices"]}
        }
        failure_type = (
            "wrong_entity" if response_norm in wrong_option_norms else "alias_or_other"
        )
    return {
        "correct": bool(grade["correct"]),
        "attempted": bool(attempted),
        "match_tier": grade["match_tier"],
        "matched_alias": grade["matched_alias"],
        "failure_type": failure_type,
        "grader": dict(OPEN_GRADING_METHOD),
        "grader_dataset": row.get("dataset"),
    }


def score_open_margins(
    model: Any,
    tokenizer: Any,
    row: dict[str, Any],
    *,
    scaler: ITIHeadScaler | None,
) -> tuple[dict[str, Any], float]:
    gold_aliases = [str(alias) for alias in row["gold_aliases"]]
    gold_norms = {normalize_answer(alias) for alias in gold_aliases}
    distractors = [
        str(option)
        for option in row["mc_options"]
        if normalize_answer(option) not in gold_norms
    ]
    continuations = {
        **{f"gold_{idx}": alias for idx, alias in enumerate(gold_aliases)},
        **{f"distractor_{idx}": option for idx, option in enumerate(distractors)},
    }
    summaries, _, elapsed = score_continuation_summaries(
        model,
        tokenizer,
        str(row["open_prompt"]),
        continuations,
        scaler=scaler,
    )
    gold_names = [name for name in continuations if name.startswith("gold_")]
    distractor_names = [
        name for name in continuations if name.startswith("distractor_")
    ]
    return {
        "endpoint": "open_prompt_gold_vs_distractor_teacher_forced",
        "target_scores": summaries,
        **compute_target_margins(summaries, gold_names, distractor_names),
    }, elapsed


def topk_first_token_logprobs(
    model: Any,
    tokenizer: Any,
    prompt_ids: torch.Tensor,
    *,
    scaler: ITIHeadScaler | None,
    k: int,
) -> list[dict[str, Any]]:
    if k <= 0:
        return []
    prompt_ids = prompt_ids.to(model.device)
    if prompt_ids.ndim == 1:
        prompt_ids = prompt_ids.unsqueeze(0)
    arm_scaler_first_decode_token(scaler)
    with torch.inference_mode():
        outputs = model(prompt_ids, use_cache=True)
    logprobs = torch.log_softmax(outputs.logits[:, -1, :], dim=-1)
    values, indices = torch.topk(logprobs[0], k=min(k, logprobs.shape[-1]))
    rows: list[dict[str, Any]] = []
    for value, token_id in zip(values.tolist(), indices.tolist(), strict=True):
        rows.append(
            {
                "token_id": int(token_id),
                "token_text": tokenizer.decode([int(token_id)]),
                "logprob": round(float(value), 6),
            }
        )
    return rows


def consume_scaler_stats(scaler: ITIHeadScaler | None) -> dict[str, Any] | None:
    if scaler is None or not hasattr(scaler, "consume_sample_stats"):
        return None
    return scaler.consume_sample_stats()


def reset_scaler_stats(scaler: ITIHeadScaler | None) -> None:
    if scaler is not None and hasattr(scaler, "reset_sample_stats"):
        scaler.reset_sample_stats()


def score_simid_item(
    model: Any,
    tokenizer: Any,
    row: dict[str, Any],
    *,
    scaler: ITIHeadScaler | None,
    alpha: float,
    condition: ConditionSpec,
    iti_config: dict[str, Any],
    top_k_first_token: int,
    mc_max_new_tokens: int,
    open_max_new_tokens: int,
) -> dict[str, Any]:
    wall_start = time.time()
    total_t0 = time.perf_counter()
    reset_scaler_stats(scaler)

    mc_likelihood, mc_likelihood_s = score_mc_likelihood(
        model, tokenizer, row, scaler=scaler
    )
    mc_letter_likelihood, mc_prompt_ids, mc_letter_likelihood_s = score_mc_letters(
        model, tokenizer, row, scaler=scaler
    )
    mc_response_t0 = time.perf_counter()
    mc_response, mc_generation_timings = generate_response(
        model,
        tokenizer,
        None,
        cached_input_ids=mc_prompt_ids,
        do_sample=False,
        max_new_tokens=mc_max_new_tokens,
        scaler=scaler,
    )
    mc_generation_s = time.perf_counter() - mc_response_t0
    generated_letter = extract_mc_answer(
        mc_response,
        list(mc_letter_likelihood["valid_letters"]),
    )
    generated_index = (
        mc_letter_likelihood["valid_letters"].index(generated_letter)
        if generated_letter in mc_letter_likelihood["valid_letters"]
        else None
    )
    gold_indices = {int(idx) for idx in row["gold_option_indices"]}
    mc_generated = {
        "endpoint": "generated_letter",
        "response": mc_response,
        "chosen_letter": generated_letter,
        "chosen_index": generated_index,
        "chosen_is_gold": generated_index in gold_indices
        if generated_index is not None
        else False,
        "timings": mc_generation_timings,
    }

    open_messages = prompt_messages(str(row["open_prompt"]))
    open_prompt_ids = tokenize_chat(tokenizer, open_messages)
    open_generation_t0 = time.perf_counter()
    open_response, open_generation_timings = generate_response(
        model,
        tokenizer,
        None,
        cached_input_ids=open_prompt_ids,
        do_sample=False,
        max_new_tokens=open_max_new_tokens,
        scaler=scaler,
    )
    open_generation_s = time.perf_counter() - open_generation_t0
    open_grade = grade_open_response(row, open_response)
    open_margins, open_margin_s = score_open_margins(
        model, tokenizer, row, scaler=scaler
    )
    topk_t0 = time.perf_counter()
    topk = topk_first_token_logprobs(
        model,
        tokenizer,
        open_prompt_ids,
        scaler=scaler,
        k=top_k_first_token,
    )
    topk_s = time.perf_counter() - topk_t0
    hook_stats = consume_scaler_stats(scaler)
    wall_end = time.time()

    condition_payload = asdict(condition)
    return {
        "id": str(row["sample_id"]),
        "sample_id": str(row["sample_id"]),
        "base_sample_id": row.get("base_sample_id", row["sample_id"]),
        "output_id": simid_output_id(str(row["sample_id"]), condition.name, alpha),
        "dataset": row["dataset"],
        "mc_endpoint": row.get("mc_endpoint", "unknown"),
        "source_id": row.get("source_id"),
        "alpha": float(alpha),
        "condition": condition.name,
        "question": row["question"],
        "open_prompt": row["open_prompt"],
        "mc_options": row["mc_options"],
        "gold_option_indices": row["gold_option_indices"],
        "gold_aliases": row["gold_aliases"],
        "distractor_provenance": row["distractor_provenance"],
        "option_order_seed": row["option_order_seed"],
        "mc_likelihood": mc_likelihood,
        "mc_letter_likelihood": mc_letter_likelihood,
        "mc_generated_letter": mc_generated,
        "open_generation": {
            "prompt": row["open_prompt"],
            "response": open_response,
            "timings": open_generation_timings,
        },
        "open_grade": open_grade,
        "open_margins": open_margins,
        "topk_first_token_logprobs": topk,
        "iti_config": dict(iti_config),
        "control_metadata": condition_payload,
        "timings": {
            "wall_start_ts": round(wall_start, 6),
            "wall_end_ts": round(wall_end, 6),
            "wall_total_s": round(wall_end - wall_start, 4),
            "total_s": round(time.perf_counter() - total_t0, 4),
            "mc_likelihood_s": round(mc_likelihood_s, 4),
            "mc_letter_likelihood_s": round(mc_letter_likelihood_s, 4),
            "mc_generation_s": round(mc_generation_s, 4),
            "open_generation_s": round(open_generation_s, 4),
            "open_margin_s": round(open_margin_s, 4),
            "topk_s": round(topk_s, 4),
            "hook_stats": hook_stats,
        },
        "git_sha": get_git_sha(),
    }


def get_path_value(row: dict[str, Any], path: tuple[str, ...]) -> Any:
    value: Any = row
    for key in path:
        if not isinstance(value, dict) or key not in value:
            raise KeyError(".".join(path))
        value = value[key]
    return value


def values_diverge(left: Any, right: Any, tolerance: float) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return bool(left) != bool(right)
    if isinstance(left, int | float) and isinstance(right, int | float):
        if math.isnan(float(left)) or math.isnan(float(right)):
            return True
        return abs(float(left) - float(right)) > tolerance
    return left != right


def assert_noop_equivalence(
    unhooked_rows: list[dict[str, Any]],
    hooked_rows: list[dict[str, Any]],
    *,
    tolerance: float = 1e-5,
    compare_paths: tuple[tuple[str, ...], ...] = NOOP_COMPARE_PATHS,
) -> None:
    """Fail loudly when hooked alpha=0 differs from unhooked outputs."""
    unhooked = {str(row["sample_id"]): row for row in unhooked_rows}
    hooked = {str(row["sample_id"]): row for row in hooked_rows}
    if set(unhooked) != set(hooked):
        raise AssertionError(
            "SIMID alpha-0 no-op check failed: unhooked and hooked sample IDs differ "
            f"(unhooked_only={sorted(set(unhooked) - set(hooked))[:5]}, "
            f"hooked_only={sorted(set(hooked) - set(unhooked))[:5]})"
        )

    divergences: list[str] = []
    for sample_id in sorted(unhooked):
        for path in compare_paths:
            left = get_path_value(unhooked[sample_id], path)
            right = get_path_value(hooked[sample_id], path)
            if values_diverge(left, right, tolerance):
                divergences.append(
                    f"{sample_id}:{'.'.join(path)} unhooked={left!r} hooked={right!r}"
                )
    if divergences:
        raise AssertionError(
            "SIMID alpha-0 no-op check failed; hooked alpha=0 diverged beyond "
            f"tolerance={tolerance}: {divergences[:10]}"
        )


def resolve_conditions(
    requested: list[str],
    *,
    include_unhooked: bool,
) -> list[ConditionSpec]:
    conditions = [CONDITION_SPECS[name] for name in requested]
    if include_unhooked:
        return [UNHOOKED_SPEC, *conditions]
    return conditions


def install_condition_scaler(
    *,
    model: Any,
    artifact: dict[str, Any],
    device: torch.device,
    condition: ConditionSpec,
    iti_family: str,
    iti_k: int,
    decode_scope: str,
    collect_debug_stats: bool,
) -> ITIHeadScaler | None:
    if condition.name == "unhooked":
        return None
    return ITIHeadScaler(
        model,
        artifact,
        device,
        family=normalize_iti_family(iti_family),
        k=iti_k,
        selection_strategy=condition.selection_strategy,
        random_seed=condition.random_seed,
        direction_mode=condition.direction_mode,
        direction_random_seed=condition.direction_random_seed,
        decode_scope=decode_scope,
        collect_debug_stats=collect_debug_stats,
    )


def run_condition_alpha(
    *,
    model: Any,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    scaler: ITIHeadScaler | None,
    alpha: float,
    condition: ConditionSpec,
    iti_config: dict[str, Any],
    output_dir: Path,
    top_k_first_token: int,
    mc_max_new_tokens: int,
    open_max_new_tokens: int,
) -> Path:
    if scaler is not None:
        scaler.alpha = float(alpha)
    alpha_label = format_alpha_label(alpha)
    out_path = output_dir / condition.name / f"alpha_{alpha_label}.jsonl"
    existing_ids = load_existing_sample_ids(out_path)
    for row in rows:
        sample_id = str(row["sample_id"])
        if sample_id in existing_ids:
            continue
        scored = score_simid_item(
            model,
            tokenizer,
            row,
            scaler=scaler,
            alpha=alpha,
            condition=condition,
            iti_config=iti_config,
            top_k_first_token=top_k_first_token,
            mc_max_new_tokens=mc_max_new_tokens,
            open_max_new_tokens=open_max_new_tokens,
        )
        append_jsonl(out_path, scored)
        existing_ids.add(sample_id)
    return out_path


def default_run_name() -> str:
    return datetime.now(timezone.utc).strftime("experiment_%Y%m%d_%H%M%S")


def resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir:
        return Path(args.output_dir)
    return BASE_OUTPUT_ROOT / (args.run_name or default_run_name())


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--device-map", default="cuda:0")
    parser.add_argument("--iti-artifact-path", default=DEFAULT_ITI_ARTIFACT)
    parser.add_argument("--iti-family", default="truthfulqa_paperfaithful")
    parser.add_argument("--iti-k", type=int, default=12)
    parser.add_argument("--decode-scope", default="first_3_tokens")
    parser.add_argument("--alphas", type=float, nargs="+", default=DEFAULT_ALPHAS)
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=["selected", "random_head_seed1", "random_direction_seed1"],
        choices=sorted(CONDITION_SPECS),
    )
    parser.add_argument("--include-unhooked", action="store_true")
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--top-k-first-token", type=int, default=10)
    parser.add_argument("--mc-max-new-tokens", type=int, default=4)
    parser.add_argument("--open-max-new-tokens", type=int, default=64)
    parser.add_argument("--collect-debug-stats", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--noop-check", action="store_true")
    parser.add_argument("--noop-tolerance", type=float, default=1e-5)
    parser.add_argument("--allow-analyzed-overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    set_reproducibility_seed(args.seed)
    output_dir = resolve_output_dir(args)
    if (output_dir / "results.json").exists() and not args.allow_analyzed_overwrite:
        raise FileExistsError(
            f"{output_dir} already contains results.json; archive or choose a new run name"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    locked_manifest_path = output_dir / "manifest.locked.json"
    rows = load_or_create_locked_manifest(
        source_manifest=args.manifest,
        output_dir=output_dir,
        max_items=args.max_items,
    )
    conditions = resolve_conditions(
        args.conditions, include_unhooked=args.include_unhooked
    )
    iti_config = build_iti_config(
        model_path=args.model_path,
        tokenizer_path=args.model_path,
        iti_artifact_path=args.iti_artifact_path,
        iti_artifact_sha256=file_sha256(args.iti_artifact_path),
        iti_family=args.iti_family,
        iti_k=args.iti_k,
        decode_scope=args.decode_scope,
    )
    run_config_path = output_dir / "run_config.json"
    provenance = start_run_provenance(
        args,
        primary_target=output_dir,
        output_targets=[output_dir, run_config_path, locked_manifest_path],
        extra={
            "simid_schema": "simid_run/v1",
            "n_manifest_rows": len(rows),
            "conditions": [condition.name for condition in conditions],
        },
        primary_target_is_dir=True,
    )
    status = "completed"
    extra: dict[str, Any] = {}
    scaler: ITIHeadScaler | None = None
    try:
        run_config = build_run_config(
            args=args,
            output_dir=output_dir,
            rows=rows,
            conditions=conditions,
            iti_config=iti_config,
            locked_manifest_path=locked_manifest_path,
        )
        write_or_validate_run_config(
            path=run_config_path,
            run_config=run_config,
            output_dir=output_dir,
        )
        extra["resume_fingerprint"] = run_config["resume_fingerprint"]

        print(f"Loading model: {args.model_path}")
        model, tokenizer = load_model_and_tokenizer(args.model_path, args.device_map)
        device = next(model.parameters()).device
        artifact = load_iti_artifact(args.iti_artifact_path)

        written_paths: list[Path] = []
        for condition in conditions:
            print(f"Running SIMID condition: {condition.name}")
            scaler = install_condition_scaler(
                model=model,
                artifact=artifact,
                device=device,
                condition=condition,
                iti_family=args.iti_family,
                iti_k=args.iti_k,
                decode_scope=args.decode_scope,
                collect_debug_stats=args.collect_debug_stats,
            )
            try:
                for alpha in args.alphas:
                    print(f"  alpha={format_alpha_label(alpha)}")
                    written_paths.append(
                        run_condition_alpha(
                            model=model,
                            tokenizer=tokenizer,
                            rows=rows,
                            scaler=scaler,
                            alpha=alpha,
                            condition=condition,
                            iti_config=iti_config,
                            output_dir=output_dir,
                            top_k_first_token=args.top_k_first_token,
                            mc_max_new_tokens=args.mc_max_new_tokens,
                            open_max_new_tokens=args.open_max_new_tokens,
                        )
                    )
            finally:
                if scaler is not None:
                    scaler.remove()
                scaler = None

        if args.noop_check:
            alpha0 = format_alpha_label(0.0)
            unhooked_path = output_dir / "unhooked" / f"alpha_{alpha0}.jsonl"
            selected_path = output_dir / "selected" / f"alpha_{alpha0}.jsonl"
            if not unhooked_path.exists() or not selected_path.exists():
                raise FileNotFoundError(
                    "--noop-check requires unhooked and selected alpha_0.0 outputs"
                )
            assert_noop_equivalence(
                load_jsonl(unhooked_path),
                load_jsonl(selected_path),
                tolerance=args.noop_tolerance,
            )

        extra["output_targets"] = [str(path) for path in written_paths]
        extra["n_rows"] = len(rows)
        print(f"SIMID run complete: {output_dir}")
    except BaseException as exc:
        status = provenance_status_for_exception(exc)
        extra["error"] = provenance_error_message(exc)
        raise
    finally:
        if scaler is not None:
            scaler.remove()
        finish_run_provenance(provenance, status, extra)


if __name__ == "__main__":
    main()
