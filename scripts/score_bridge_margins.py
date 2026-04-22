"""Score gold-vs-wrong log-likelihood margins on bridge discordant cases.

For each adjudicated bridge discordant case (plus length-matched random
wrong-entity controls), teacher-force the gold answer and the wrong-entity
string under two conditions:

  * ``baseline``: ITI hook installed but alpha=0 (no-op).
  * ``iti``:      ITI hook installed with production alpha (default 8.0).

Record first-3-token and full-continuation log-likelihoods for both targets
under both conditions. Primary per-case metric is

    shift_nats = (ell_iti_gold - ell_iti_wrong) - (ell_base_gold - ell_base_wrong)

for both the first-3-token window (primary) and the full-continuation window
(sensitivity). Negative shift = ITI moved probability mass toward the wrong
entity.

Output is an incrementally written JSONL. The script is idempotent: on
restart, cases whose ``case_id`` already appears in the output file are
skipped.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import torch

# scripts/ is importable as a flat namespace (see pyproject.toml tool.ty).
from intervene_iti import ITIHeadScaler, load_iti_artifact
from run_intervention import load_model_and_tokenizer, tokenize_chat
from utils import format_path_for_metadata, sanitize_run_config

ROOT = Path(__file__).resolve().parent.parent

DEFAULT_ADJUDICATED_LABELS = (
    ROOT / "data/judge_validation/bridge_irr/adjudicated_labels.jsonl"
)
DEFAULT_QUEUE_KEY = ROOT / "data/judge_validation/bridge_irr/test_queue_key.jsonl"
DEFAULT_BASELINE_ROWS = (
    ROOT / "data/gemma3_4b/intervention/triviaqa_bridge/test_experiment/alpha_1.0.jsonl"
)
DEFAULT_MANIFEST = ROOT / "data/manifests/triviaqa_bridge_test500_seed42.json"
DEFAULT_ITI_ARTIFACT = (
    ROOT
    / "data/contrastive/truthfulness/iti_truthfulqa_paperfaithful_production/iti_heads.pt"
)
DEFAULT_OUTPUT_DIR = ROOT / "data/gemma3_4b/analysis/bridge_margins/test"
DEFAULT_MODEL_PATH = "google/gemma-3-4b-it"

FIRST_N_TOKENS = 3
CONTROLS_PER_SEED = 4  # per R→W substitution case
BUCKET_CAP = 6  # same bucket ceiling as extract_truthfulness_iti._bucket_answer


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    # str.splitlines() splits on Unicode line-breaks (e.g. \u2028) that may
    # appear INSIDE JSON string values — iterate the file handle instead so
    # we split only on actual record terminators.
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_existing_case_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {row["case_id"] for row in load_jsonl(path)}


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_head_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


# ---------------------------------------------------------------------------
# Cohort assembly
# ---------------------------------------------------------------------------


def cohort_for_case(transition: str, label: str) -> str:
    if transition == "right_to_wrong":
        if label == "wrong_entity_substitution":
            return "A_rw_substitution"
        return "B_rw_nonsubstitution"
    if transition == "wrong_to_right":
        return "C_wr_rescue"
    raise ValueError(f"Unknown transition: {transition!r}")


def bucket_for_token_len(n_tokens: int) -> int:
    """Match ``extract_truthfulness_iti._bucket_answer`` bucketing."""
    return min(max(1, n_tokens), BUCKET_CAP)


def build_question_pool(
    baseline_rows: list[dict[str, Any]],
    tokenizer: Any,
) -> dict[str, dict[str, Any]]:
    """Map question_id -> {question, aliases, alias_token_ids_list}."""
    pool: dict[str, dict[str, Any]] = {}
    for row in baseline_rows:
        qid = str(row["id"])
        aliases = [a for a in row.get("ground_truth_aliases") or [] if a]
        if not aliases:
            continue
        alias_token_ids = [
            tuple(
                int(t) for t in tokenizer(alias, add_special_tokens=False)["input_ids"]
            )
            for alias in aliases
        ]
        pool[qid] = {
            "question": str(row["question"]),
            "aliases": aliases,
            "alias_token_ids": alias_token_ids,
        }
    return pool


def build_cohort_records(
    adjudicated: list[dict[str, Any]],
    queue_key: list[dict[str, Any]],
    question_pool: dict[str, dict[str, Any]],
    tokenizer: Any,
    *,
    n_controls: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Return a deterministic list of per-case input records (sorted by case_id).

    A/B/C cases come from the adjudicated artifact joined to queue_key.
    D (controls) cases are generated by seeding on the A-cohort prompts and
    swapping the wrong-entity string for a length-bucket-matched random gold
    answer drawn from a DIFFERENT question.
    """
    key_by_case = {row["case_id"]: row for row in queue_key}

    records: list[dict[str, Any]] = []
    a_cases: list[dict[str, Any]] = []
    for adj in adjudicated:
        case_id = str(adj["case_id"])
        qid = str(adj["question_id"])
        if case_id not in key_by_case:
            raise ValueError(f"queue_key missing case_id={case_id}")
        if qid not in question_pool:
            raise ValueError(
                f"question pool missing qid={qid} (baseline row not loaded?)"
            )
        key_row = key_by_case[case_id]
        pool_row = question_pool[qid]
        label = str(adj["label"])
        transition = str(adj["transition"])
        cohort = cohort_for_case(transition, label)
        # Strip incidental whitespace: tokenizers treat " Paris" and "Paris"
        # as different BPE sequences, and the queue_key is hand-curated so
        # leading/trailing spaces can slip in and silently distort margins.
        wrong_text = str(key_row["incorrect_response"]).strip()
        record = {
            "case_id": case_id,
            "question_id": qid,
            "cohort": cohort,
            "transition": transition,
            "label": label,
            "question": pool_row["question"],
            "gold_aliases": pool_row["aliases"],
            "wrong_entity_text": wrong_text,
            "source": "adjudicated",
        }
        records.append(record)
        if cohort == "A_rw_substitution":
            a_cases.append(record)

    # Build cohort D: length-bucketed random wrong-entity controls seeded on
    # the A cohort's prompts. We sample with a fixed RNG from the pool of
    # OTHER questions' gold aliases.
    if a_cases and n_controls > 0:
        rng = random.Random(seed)
        bucketed: dict[int, list[tuple[str, str, tuple[int, ...]]]] = defaultdict(list)
        for qid, row in question_pool.items():
            for alias, tok_ids in zip(row["aliases"], row["alias_token_ids"]):
                if not tok_ids:
                    continue
                bucketed[bucket_for_token_len(len(tok_ids))].append(
                    (qid, alias, tok_ids)
                )

        for idx in range(n_controls):
            seed_case = a_cases[idx % len(a_cases)]
            # Length-match on the first-3-token length of the original wrong entity.
            wrong_ids = tokenizer(
                seed_case["wrong_entity_text"], add_special_tokens=False
            )["input_ids"]
            target_len = min(len(wrong_ids), FIRST_N_TOKENS) or 1
            bucket = bucket_for_token_len(target_len)
            pool_at_bucket = [
                entry
                for entry in bucketed.get(bucket, [])
                if entry[0] != seed_case["question_id"]
                and entry[1] != seed_case["wrong_entity_text"]
            ]
            if not pool_at_bucket:
                # Fallback: any bucket.
                pool_at_bucket = [
                    entry
                    for entries in bucketed.values()
                    for entry in entries
                    if entry[0] != seed_case["question_id"]
                ]
            if not pool_at_bucket:
                raise ValueError(
                    "No control wrong-entity pool available for seed case "
                    f"{seed_case['case_id']}"
                )
            _, alias_text, _ = rng.choice(pool_at_bucket)
            synthetic_id = f"control_{seed_case['case_id']}_{idx:04d}"
            records.append(
                {
                    "case_id": synthetic_id,
                    "question_id": seed_case["question_id"],
                    "cohort": "D_random_control",
                    "transition": "right_to_wrong",
                    "label": "control_random_wrong_entity",
                    "question": seed_case["question"],
                    "gold_aliases": seed_case["gold_aliases"],
                    "wrong_entity_text": alias_text,
                    "source": "control",
                    "seed_case_id": seed_case["case_id"],
                }
            )

    records.sort(key=lambda r: (r["cohort"], r["case_id"]))
    return records


# ---------------------------------------------------------------------------
# Teacher-forcing
# ---------------------------------------------------------------------------


def build_prompt_ids(tokenizer: Any, question: str) -> torch.Tensor:
    """Reuse :func:`run_intervention._triviaqa_bridge_prompt` inline to avoid
    importing private generation helpers."""
    prompt_text = (
        f"Question: {question}\nAnswer with a single short factual phrase only."
    )
    messages = [{"role": "user", "content": prompt_text}]
    return tokenize_chat(tokenizer, messages)


def score_continuations(
    model: torch.nn.Module,
    prompt_input_ids: torch.Tensor,
    targets: dict[str, tuple[int, ...]],
    *,
    scaler: ITIHeadScaler | None,
) -> dict[str, list[float]]:
    """Return per-target per-token log-probability lists.

    Re-prefills the prompt for each target so the scaler's decode-scope
    counter is always fresh (``arm_first_decode_token`` + prefill).
    """
    device = next(model.parameters()).device
    prompt_input_ids = prompt_input_ids.to(device)
    if prompt_input_ids.ndim == 1:
        prompt_input_ids = prompt_input_ids.unsqueeze(0)

    results: dict[str, list[float]] = {}
    for name, tok_ids in targets.items():
        if not tok_ids:
            results[name] = []
            continue
        if scaler is not None:
            scaler.reset_sample_stats()
            scaler.arm_first_decode_token()
        with torch.inference_mode():
            prompt_out = model(prompt_input_ids, use_cache=True)
        logits_last = prompt_out.logits[:, -1, :]
        prompt_logprobs = torch.log_softmax(logits_last, dim=-1)
        past = prompt_out.past_key_values
        cont = torch.tensor(tok_ids, dtype=torch.long, device=device)
        first_lp = float(prompt_logprobs[0, int(cont[0])].item())
        _guard_finite_logprob(first_lp, target_name=name, position=0)
        per_token: list[float] = [first_lp]
        for position in range(cont.numel() - 1):
            cur = cont[position : position + 1].view(1, 1)
            nxt = int(cont[position + 1].item())
            with torch.inference_mode():
                out = model(cur, use_cache=True, past_key_values=past)
            past = out.past_key_values
            lp = torch.log_softmax(out.logits[:, -1, :], dim=-1)[0, nxt]
            lp_f = float(lp.item())
            _guard_finite_logprob(lp_f, target_name=name, position=position + 1)
            per_token.append(lp_f)
        results[name] = per_token
    return results


def _guard_finite_logprob(value: float, *, target_name: str, position: int) -> None:
    """Fail fast on NaN/Inf log-probs rather than letting them poison the
    downstream bootstrap silently. A surfaced exception during an overnight
    run is recoverable; an analysed report built on tainted numbers is not.
    """
    if math.isnan(value) or math.isinf(value):
        raise ValueError(
            f"Non-finite log-prob at target={target_name!r} position={position}: "
            f"{value!r}. Inspect the model/prompt rather than masking this record."
        )


def pick_gold_alias(
    gold_aliases: Iterable[str],
    tokenizer: Any,
    baseline_prompt_ids: torch.Tensor,
    model: torch.nn.Module,
    scaler: ITIHeadScaler | None,
) -> tuple[int, str, tuple[int, ...], list[float]]:
    """Score every alias under baseline, pick max per-token log-prob.

    Returns (index_in_aliases, alias_text, alias_token_ids, per_token_logprobs).
    """
    tokenized = {
        f"alias_{i}": tuple(
            int(t) for t in tokenizer(alias, add_special_tokens=False)["input_ids"]
        )
        for i, alias in enumerate(gold_aliases)
    }
    tokenized = {name: ids for name, ids in tokenized.items() if ids}
    if not tokenized:
        raise ValueError("No non-empty gold aliases")
    scores = score_continuations(model, baseline_prompt_ids, tokenized, scaler=scaler)
    best_name = max(
        scores.keys(),
        key=lambda name: (
            sum(scores[name]) / max(1, len(scores[name]))
            if scores[name]
            else float("-inf")
        ),
    )
    idx = int(best_name.split("_")[-1])
    aliases_list = list(gold_aliases)
    return (
        idx,
        aliases_list[idx],
        tokenized[best_name],
        scores[best_name],
    )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def build_provenance(
    *,
    args: argparse.Namespace,
    output_path: Path,
    iti_artifact_sha: str,
    n_records: int,
    n_skipped: int,
    elapsed_s: float,
) -> dict[str, Any]:
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return {
        "script": "scripts/score_bridge_margins.py",
        "timestamp_utc": ts,
        "git_commit": git_head_commit(),
        "args": sanitize_run_config(vars(args)),
        "iti_artifact_sha256": iti_artifact_sha,
        "model_path": args.model_path,
        "first_n_tokens": FIRST_N_TOKENS,
        "n_records_written": n_records,
        "n_records_skipped_existing": n_skipped,
        "elapsed_seconds": elapsed_s,
        "output_path": format_path_for_metadata(output_path, root=ROOT),
    }


def summarize_per_token(per_token: list[float]) -> dict[str, Any]:
    n = len(per_token)
    return {
        "per_token_logprobs": [float(x) for x in per_token],
        "n_tokens": int(n),
        "sum_logprob": float(sum(per_token)) if per_token else 0.0,
        "sum_logprob_first3": float(sum(per_token[:FIRST_N_TOKENS]))
        if per_token
        else 0.0,
        "n_tokens_first3": int(min(n, FIRST_N_TOKENS)),
    }


def compute_shifts(
    base_gold: dict[str, Any],
    base_wrong: dict[str, Any],
    iti_gold: dict[str, Any],
    iti_wrong: dict[str, Any],
) -> dict[str, dict[str, float]]:
    def per_token_avg(summary: dict[str, Any], key_sum: str, key_n: str) -> float:
        n = int(summary[key_n])
        if n == 0:
            return 0.0
        return float(summary[key_sum]) / n

    # Primary: first 3 tokens (sum-nats + per-token)
    base_gold_f3 = float(base_gold["sum_logprob_first3"])
    base_wrong_f3 = float(base_wrong["sum_logprob_first3"])
    iti_gold_f3 = float(iti_gold["sum_logprob_first3"])
    iti_wrong_f3 = float(iti_wrong["sum_logprob_first3"])
    delta_base_f3 = base_gold_f3 - base_wrong_f3
    delta_iti_f3 = iti_gold_f3 - iti_wrong_f3
    shift_nats_f3 = delta_iti_f3 - delta_base_f3

    # Per-token on first-3 window
    base_gold_f3_pt = per_token_avg(base_gold, "sum_logprob_first3", "n_tokens_first3")
    base_wrong_f3_pt = per_token_avg(
        base_wrong, "sum_logprob_first3", "n_tokens_first3"
    )
    iti_gold_f3_pt = per_token_avg(iti_gold, "sum_logprob_first3", "n_tokens_first3")
    iti_wrong_f3_pt = per_token_avg(iti_wrong, "sum_logprob_first3", "n_tokens_first3")
    shift_per_token_f3 = (iti_gold_f3_pt - iti_wrong_f3_pt) - (
        base_gold_f3_pt - base_wrong_f3_pt
    )

    # Sensitivity: full-continuation window
    delta_base_full = float(base_gold["sum_logprob"]) - float(base_wrong["sum_logprob"])
    delta_iti_full = float(iti_gold["sum_logprob"]) - float(iti_wrong["sum_logprob"])
    shift_nats_full = delta_iti_full - delta_base_full

    base_gold_full_pt = per_token_avg(base_gold, "sum_logprob", "n_tokens")
    base_wrong_full_pt = per_token_avg(base_wrong, "sum_logprob", "n_tokens")
    iti_gold_full_pt = per_token_avg(iti_gold, "sum_logprob", "n_tokens")
    iti_wrong_full_pt = per_token_avg(iti_wrong, "sum_logprob", "n_tokens")
    shift_per_token_full = (iti_gold_full_pt - iti_wrong_full_pt) - (
        base_gold_full_pt - base_wrong_full_pt
    )

    return {
        "first3": {
            "ell_base_gold": base_gold_f3,
            "ell_base_wrong": base_wrong_f3,
            "ell_iti_gold": iti_gold_f3,
            "ell_iti_wrong": iti_wrong_f3,
            "delta_base": delta_base_f3,
            "delta_iti": delta_iti_f3,
            "shift_nats": shift_nats_f3,
            "shift_per_token": shift_per_token_f3,
        },
        "full": {
            "ell_base_gold": float(base_gold["sum_logprob"]),
            "ell_base_wrong": float(base_wrong["sum_logprob"]),
            "ell_iti_gold": float(iti_gold["sum_logprob"]),
            "ell_iti_wrong": float(iti_wrong["sum_logprob"]),
            "delta_base": delta_base_full,
            "delta_iti": delta_iti_full,
            "shift_nats": shift_nats_full,
            "shift_per_token": shift_per_token_full,
        },
    }


def format_case_report(record: dict[str, Any]) -> str:
    f3 = record["first3"]
    return (
        f"[{record['cohort']}] {record['case_id']} "
        f"({record['transition']}, {record['label']})\n"
        f"  Q: {record['question'][:180]}\n"
        f"  gold: {record['gold_alias_used']!r} "
        f"(idx={record['gold_alias_index']}, tok_len={record['gold_first3_n_tokens']})\n"
        f"  wrong: {record['wrong_entity_text']!r} "
        f"(tok_len={record['wrong_first3_n_tokens']})\n"
        f"  first3: ell_base_gold={f3['ell_base_gold']:+.3f}  "
        f"ell_base_wrong={f3['ell_base_wrong']:+.3f}\n"
        f"          ell_iti_gold={f3['ell_iti_gold']:+.3f}  "
        f"ell_iti_wrong={f3['ell_iti_wrong']:+.3f}\n"
        f"          delta_base={f3['delta_base']:+.3f}  "
        f"delta_iti={f3['delta_iti']:+.3f}  "
        f"shift_nats={f3['shift_nats']:+.3f}  "
        f"shift/tok={f3['shift_per_token']:+.3f}\n"
    )


def score_one_case(
    case: dict[str, Any],
    *,
    model: torch.nn.Module,
    tokenizer: Any,
    scaler: ITIHeadScaler,
    iti_alpha: float,
) -> dict[str, Any]:
    prompt_ids = build_prompt_ids(tokenizer, case["question"])

    # Baseline condition (alpha=0, no-op).
    scaler.alpha = 0.0
    gold_idx, gold_text, gold_ids, _ = pick_gold_alias(
        case["gold_aliases"], tokenizer, prompt_ids, model, scaler
    )
    wrong_ids = tuple(
        int(t)
        for t in tokenizer(case["wrong_entity_text"], add_special_tokens=False)[
            "input_ids"
        ]
    )
    targets = {"gold": gold_ids, "wrong": wrong_ids}
    base_scores = score_continuations(model, prompt_ids, targets, scaler=scaler)

    # ITI condition.
    scaler.alpha = iti_alpha
    iti_scores = score_continuations(model, prompt_ids, targets, scaler=scaler)

    base_gold = summarize_per_token(base_scores["gold"])
    base_wrong = summarize_per_token(base_scores["wrong"])
    iti_gold = summarize_per_token(iti_scores["gold"])
    iti_wrong = summarize_per_token(iti_scores["wrong"])
    shifts = compute_shifts(base_gold, base_wrong, iti_gold, iti_wrong)

    record = {
        "case_id": case["case_id"],
        "question_id": case["question_id"],
        "cohort": case["cohort"],
        "transition": case["transition"],
        "label": case["label"],
        "source": case["source"],
        "question": case["question"],
        "prompt_sha256": hashlib.sha256(case["question"].encode("utf-8")).hexdigest(),
        "gold_alias_used": gold_text,
        "gold_alias_index": gold_idx,
        "gold_first3_n_tokens": int(min(len(gold_ids), FIRST_N_TOKENS)),
        "gold_full_n_tokens": int(len(gold_ids)),
        "wrong_entity_text": case["wrong_entity_text"],
        "wrong_first3_n_tokens": int(min(len(wrong_ids), FIRST_N_TOKENS)),
        "wrong_full_n_tokens": int(len(wrong_ids)),
        "base_gold": base_gold,
        "base_wrong": base_wrong,
        "iti_gold": iti_gold,
        "iti_wrong": iti_wrong,
        "first3": shifts["first3"],
        "full": shifts["full"],
        "iti_alpha": float(iti_alpha),
    }
    if "seed_case_id" in case:
        record["seed_case_id"] = case["seed_case_id"]
    return record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adjudicated", type=Path, default=DEFAULT_ADJUDICATED_LABELS)
    parser.add_argument("--queue-key", type=Path, default=DEFAULT_QUEUE_KEY)
    parser.add_argument("--baseline-rows", type=Path, default=DEFAULT_BASELINE_ROWS)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--iti-artifact", type=Path, default=DEFAULT_ITI_ARTIFACT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--iti-alpha", type=float, default=8.0)
    parser.add_argument("--iti-k", type=int, default=12)
    parser.add_argument("--iti-family", default="truthfulqa_paperfaithful")
    parser.add_argument("--iti-decode-scope", default="first_3_tokens")
    parser.add_argument("--iti-selection-strategy", default="ranked")
    parser.add_argument("--iti-random-seed", type=int, default=42)
    parser.add_argument("--n-controls", type=int, default=200)
    parser.add_argument("--control-seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--only-case-ids",
        nargs="*",
        default=None,
        help="If set, only score these case_ids (e.g. Gate 2 sanity run).",
    )
    parser.add_argument(
        "--dry-run-print-each-case",
        action="store_true",
        help="After scoring each case, print a human-readable report line.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    t0 = time.perf_counter()

    print("Loading adjudicated + queue-key + baseline rows...", file=sys.stderr)
    adjudicated = load_jsonl(args.adjudicated)
    queue_key = load_jsonl(args.queue_key)
    baseline_rows = load_jsonl(args.baseline_rows)

    print(f"Loading tokenizer + model from {args.model_path}...", file=sys.stderr)
    model, tokenizer = load_model_and_tokenizer(
        args.model_path, device_map=args.device_map
    )

    print("Building question pool...", file=sys.stderr)
    question_pool = build_question_pool(baseline_rows, tokenizer)

    print("Assembling cohorts...", file=sys.stderr)
    records = build_cohort_records(
        adjudicated,
        queue_key,
        question_pool,
        tokenizer,
        n_controls=args.n_controls,
        seed=args.control_seed,
    )

    if args.only_case_ids:
        keep = set(args.only_case_ids)
        records = [r for r in records if r["case_id"] in keep]
        print(
            f"Filtered to --only-case-ids ({len(records)}/{len(args.only_case_ids)} matched)",
            file=sys.stderr,
        )
    if args.limit is not None:
        records = records[: args.limit]

    output_path = args.output_dir / "margins.jsonl"
    existing_ids = load_existing_case_ids(output_path)
    n_skipped = 0

    print(f"Loading ITI artifact from {args.iti_artifact}...", file=sys.stderr)
    artifact = load_iti_artifact(args.iti_artifact)
    iti_artifact_sha = file_sha256(args.iti_artifact)

    device = model.device
    # The artifact's family field is ``iti_<family>``; ITIHeadScaler expects
    # the same name to match. ``run_intervention.py`` uses the short form on
    # CLI and prefixes with ``iti_`` when constructing the scaler — mirror
    # that convention here so the same ``--iti-family truthfulqa_paperfaithful``
    # CLI value accepts an artifact stamped ``iti_truthfulqa_paperfaithful``.
    scaler_family = (
        args.iti_family
        if args.iti_family.startswith("iti_")
        else f"iti_{args.iti_family}"
    )
    scaler = ITIHeadScaler(
        model,
        artifact,
        device=device,
        family=scaler_family,
        k=args.iti_k,
        selection_strategy=args.iti_selection_strategy,
        random_seed=args.iti_random_seed,
        direction_mode="artifact",
        decode_scope=args.iti_decode_scope,
    )

    print(
        f"Scoring {len(records)} cases (existing={len(existing_ids)}, alpha={args.iti_alpha})...",
        file=sys.stderr,
    )
    n_written = 0
    for case in records:
        if case["case_id"] in existing_ids:
            n_skipped += 1
            continue
        scored = score_one_case(
            case,
            model=model,
            tokenizer=tokenizer,
            scaler=scaler,
            iti_alpha=args.iti_alpha,
        )
        append_jsonl(output_path, scored)
        n_written += 1
        if args.dry_run_print_each_case:
            print(format_case_report(scored))

    elapsed_s = time.perf_counter() - t0
    prov_path = (
        args.output_dir
        / f"score_bridge_margins.provenance.{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    )
    prov_path.parent.mkdir(parents=True, exist_ok=True)
    prov_path.write_text(
        json.dumps(
            build_provenance(
                args=args,
                output_path=output_path,
                iti_artifact_sha=iti_artifact_sha,
                n_records=n_written,
                n_skipped=n_skipped,
                elapsed_s=elapsed_s,
            ),
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        f"Done. Wrote {n_written} (skipped {n_skipped}) in {elapsed_s:.1f}s.\n"
        f"Output:     {format_path_for_metadata(output_path, root=ROOT)}\n"
        f"Provenance: {format_path_for_metadata(prov_path, root=ROOT)}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
