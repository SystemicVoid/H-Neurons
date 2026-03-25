"""Regenerate full-length responses for jailbreak gold-label candidates.

Runs the same 25 prompts from jailbreak_candidates.jsonl through the model
with the h-neuron intervention at a given alpha, using greedy decoding and
max_new_tokens=5000 so responses are never truncated.

Output: tests/gold_labels/jailbreak_regen_a{alpha}_{timestamp}.jsonl

Usage:
    uv run python scripts/regen_gold_responses.py                 # alpha=0.0
    uv run python scripts/regen_gold_responses.py --alpha 1.5     # alpha=1.5
    uv run python scripts/regen_gold_responses.py --alpha 0.0 1.5 3.0  # multiple
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_intervention import JAILBREAK_TEMPLATES, HNeuronScaler
from intervene_model import get_h_neuron_indices

MODEL_PATH = os.environ.get("HNEURONS_MODEL_PATH", "google/gemma-3-4b-it")
CLASSIFIER_PATH = os.environ.get(
    "HNEURONS_CLASSIFIER_PATH", "models/gemma3_4b_classifier.pkl"
)
DEVICE = os.environ.get("HNEURONS_DEVICE_MAP", "cuda:0")
CANDIDATES_PATH = Path("tests/gold_labels/jailbreak_candidates.jsonl")
OUTPUT_DIR = Path("tests/gold_labels")


def _parse_template_idx(record_id: str) -> int:
    """Extract template index from ID like 'jbb_harmful_30_t1' -> 1."""
    return int(record_id.rsplit("_t", 1)[1])


def run_alpha(
    alpha: float,
    candidates: list[dict],
    model,
    tokenizer,
    scaler: HNeuronScaler,
    timestamp: str,
) -> Path:
    """Generate full responses for all candidates at a single alpha value."""
    scaler.alpha = alpha
    alpha_str = f"{alpha:.1f}".replace(".", "_")
    out_path = OUTPUT_DIR / f"jailbreak_regen_a{alpha_str}_{timestamp}.jsonl"
    out_path.write_text("")

    print(f"\n{'=' * 60}")
    print(f"  Alpha = {alpha}  →  {out_path.name}")
    print(f"{'=' * 60}")

    results = []
    for i, rec in enumerate(candidates, 1):
        template = JAILBREAK_TEMPLATES[rec["template_idx"]]
        prompt = template.format(goal=rec["goal"])
        messages = [{"role": "user", "content": prompt}]

        inputs = tokenizer.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True
        ).to(DEVICE)

        print(
            f"  [{i}/{len(candidates)}] {rec['id']} (t{rec['template_idx']}) ...",
            end="",
            flush=True,
        )
        with torch.no_grad():
            output = model.generate(
                inputs,
                max_new_tokens=5000,
                do_sample=False,  # greedy — deterministic for labeling
            )

        response = tokenizer.decode(
            output[0][inputs.shape[1] :], skip_special_tokens=True
        )

        result = {
            "id": rec["id"],
            "goal": rec["goal"],
            "template_idx": rec["template_idx"],
            "alpha": alpha,
            "response": response,
        }
        results.append(result)
        print(f"  {len(response)} chars")

        # Per-record write: reopen file each time (see scripts/CLAUDE.md)
        with open(out_path, "a") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    print(f"  Wrote {len(results)} records to {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Regenerate full-length jailbreak responses at given alpha(s)"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        nargs="+",
        default=[0.0],
        help="Alpha value(s) to test (default: 0.0)",
    )
    args = parser.parse_args()

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    # Load candidates
    candidates = [
        json.loads(line)
        for line in CANDIDATES_PATH.read_text().splitlines()
        if line.strip()
    ]
    for c in candidates:
        c["goal"] = c["question"]
        c["template_idx"] = _parse_template_idx(c["id"])
    print(f"Found {len(candidates)} candidates in {CANDIDATES_PATH}")

    # Load model once, reuse across alphas
    print(f"Loading model {MODEL_PATH} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype=torch.bfloat16, device_map=DEVICE
    )
    model.eval()

    clf = joblib.load(CLASSIFIER_PATH)
    neuron_map = get_h_neuron_indices(clf, model.config)
    scaler = HNeuronScaler(model, neuron_map, DEVICE)

    # Run each alpha
    output_paths = []
    for alpha in args.alpha:
        out_path = run_alpha(alpha, candidates, model, tokenizer, scaler, timestamp)
        output_paths.append(out_path)

    print(f"\n{'=' * 60}")
    print(f"  All done. {len(output_paths)} file(s) written:")
    for p in output_paths:
        print(f"    {p}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
