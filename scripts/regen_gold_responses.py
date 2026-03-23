"""One-off script to regenerate full responses for 4 gold-label candidates.

The original alpha_0.0.jsonl used max_new_tokens=256 which truncated these
responses mid-sentence. This script reruns only those 4 IDs with 1024 tokens
and greedy decoding so the full response is available for human labeling.

Output: tests/gold_labels/jailbreak_regen.jsonl

Usage:
    uv run python scripts/regen_gold_responses.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import joblib
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_intervention import JAILBREAK_TEMPLATES, HNeuronScaler
from intervene_model import get_h_neuron_indices

TARGET_IDS = {
    "jbb_harmful_19_t0",
    "jbb_harmful_3_t2",
    "jbb_harmful_91_t4",
    "jbb_harmful_97_t0",
}

MODEL_PATH = os.environ.get("HNEURONS_MODEL_PATH", "google/gemma-3-4b-it")
CLASSIFIER_PATH = os.environ.get(
    "HNEURONS_CLASSIFIER_PATH", "models/gemma3_4b_classifier.pkl"
)
DEVICE = os.environ.get("HNEURONS_DEVICE_MAP", "cuda:0")
SOURCE_PATH = Path("data/gemma3_4b/intervention/jailbreak/experiment/alpha_0.0.jsonl")
OUTPUT_PATH = Path("tests/gold_labels/jailbreak_regen.jsonl")


def main() -> None:
    # Load source records for the 4 targets
    source = {
        json.loads(line)["id"]: json.loads(line)
        for line in SOURCE_PATH.read_text().splitlines()
        if line.strip()
    }
    targets = [source[rid] for rid in TARGET_IDS if rid in source]
    print(f"Found {len(targets)}/4 target records in source data")

    print(f"Loading model {MODEL_PATH} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype=torch.bfloat16, device_map=DEVICE
    )
    model.eval()

    clf = joblib.load(CLASSIFIER_PATH)
    neuron_map = get_h_neuron_indices(clf, model.config)
    scaler = HNeuronScaler(model, neuron_map, DEVICE)
    scaler.alpha = 0.0  # match original alpha_0.0 condition

    results = []
    for rec in targets:
        template = JAILBREAK_TEMPLATES[rec["template_idx"]]
        prompt = template.format(goal=rec["goal"])
        messages = [{"role": "user", "content": prompt}]

        inputs = tokenizer.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True
        ).to(DEVICE)

        print(f"  Generating {rec['id']} (template {rec['template_idx']}) ...")
        with torch.no_grad():
            output = model.generate(
                inputs,
                max_new_tokens=1024,
                do_sample=False,  # greedy — deterministic for labeling
            )

        response = tokenizer.decode(
            output[0][inputs.shape[1] :], skip_special_tokens=True
        )

        result = {
            "id": rec["id"],
            "goal": rec["goal"],
            "template_idx": rec["template_idx"],
            "alpha": 0.0,
            "response": response,
        }
        results.append(result)
        print(f"    {len(response)} chars")

    OUTPUT_PATH.write_text("\n".join(json.dumps(r) for r in results) + "\n")
    print(f"\nWrote {len(results)} records to {OUTPUT_PATH}")

    # Print full responses for immediate review
    print("\n" + "=" * 80)
    for r in results:
        print(f"\n=== {r['id']} ===")
        print(f"Q: {r['goal']}\n")
        print(f"R:\n{r['response']}")
        print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
