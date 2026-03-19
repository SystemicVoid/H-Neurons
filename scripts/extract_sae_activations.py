"""Extract SAE feature activations for TriviaQA samples.

Hooks at post_feedforward_layernorm output (the exact point Gemma Scope 2
SAEs are trained on), projects through SAE encoders, and saves per-sample
sparse feature vectors aggregated over requested token locations.

Usage:
    uv run python scripts/extract_sae_activations.py \
        --model_path google/gemma-3-4b-it \
        --input_path data/gemma3_4b/pipeline/answer_tokens.jsonl \
        --train_ids_path data/gemma3_4b/pipeline/train_qids.json \
        --output_root data/gemma3_4b/pipeline/activations_sae \
        --locations answer_tokens all_except_answer_tokens \
        --sae_release gemma-scope-2-4b-it-mlp-all \
        --sae_width 16k \
        --sae_l0 small \
        --layers 0 5 6 7 13 14 15 16 17 20

For all layers:
    --layers all
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from extract_activations import (
    aggregate_token_activations,
    get_region_indices,
    select_token_activations,
    unwrap_chat_template_output,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract SAE feature activations for TriviaQA samples."
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
        help="Root directory for saving SAE feature .npy files.",
    )
    parser.add_argument(
        "--device_map",
        type=str,
        default="auto",
        help="Hugging Face device_map value, e.g. 'auto' or 'cuda:0'.",
    )
    parser.add_argument(
        "--sae_release",
        type=str,
        default="gemma-scope-2-4b-it-mlp-all",
        help="SAE-lens release name",
    )
    parser.add_argument(
        "--sae_width",
        type=str,
        default="16k",
        choices=["16k", "262k"],
        help="SAE width (16k or 262k)",
    )
    parser.add_argument(
        "--sae_l0",
        type=str,
        default="small",
        choices=["small", "big"],
        help="SAE L0 target (small=10-20, big=60-150)",
    )
    parser.add_argument(
        "--layers",
        nargs="+",
        default=["0", "5", "6", "7", "13", "14", "15", "16", "17", "20"],
        help="Layers to extract. Use 'all' for all layers.",
    )
    parser.add_argument(
        "--locations",
        nargs="+",
        default=["answer_tokens"],
        choices=["input", "output", "answer_tokens", "all_except_answer_tokens"],
        help="List of positions to extract SAE activations from.",
    )
    parser.add_argument("--method", type=str, choices=["mean", "max"], default="mean")
    return parser.parse_args()


def sae_cfg_to_dict(sae) -> dict:
    cfg = sae.cfg
    if hasattr(cfg, "to_dict"):
        return cfg.to_dict()
    return {key: value for key, value in vars(cfg).items() if not key.startswith("_")}


def resolve_layer_indices(model_path: str, layers: list[str]) -> list[int]:
    if layers == ["all"]:
        config = AutoConfig.from_pretrained(model_path)
        text_cfg = config.text_config if hasattr(config, "text_config") else config
        return list(range(text_cfg.num_hidden_layers))
    return [int(x) for x in layers]


def metadata_path(output_root: str) -> Path:
    return Path(output_root) / "metadata.json"


def build_metadata(
    args,
    layer_indices: list[int],
    sae_cfg: dict,
) -> dict:
    return {
        "model_path": args.model_path,
        "hook_point": "post_feedforward_layernorm",
        "sae_release": args.sae_release,
        "sae_width": args.sae_width,
        "sae_l0": args.sae_l0,
        "layer_indices": layer_indices,
        "d_in": sae_cfg.get("d_in"),
        "d_sae": sae_cfg.get("d_sae"),
        "aggregation_method": args.method,
        "locations": list(args.locations),
    }


def ensure_metadata(output_root: str, metadata: dict) -> None:
    path = metadata_path(output_root)
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            existing = json.load(handle)
        keys_to_match = [
            "model_path",
            "hook_point",
            "sae_release",
            "sae_width",
            "sae_l0",
            "layer_indices",
            "d_in",
            "d_sae",
            "aggregation_method",
        ]
        mismatches = [
            key for key in keys_to_match if existing.get(key) != metadata.get(key)
        ]
        if mismatches:
            raise ValueError(
                f"Existing SAE metadata at {path} does not match this extraction run. "
                "Use a fresh output directory or remove the stale metadata file."
            )
        merged_locations = list(
            dict.fromkeys(existing.get("locations", []) + metadata["locations"])
        )
        if merged_locations != existing.get("locations", []):
            existing["locations"] = merged_locations
            with path.open("w", encoding="utf-8") as handle:
                json.dump(existing, handle, indent=2)
        return

    with path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


class SAEExtractor:
    """Captures post_feedforward_layernorm outputs and projects through SAEs."""

    def __init__(self, model, saes, layer_indices):
        self.model = model
        self.saes = saes
        self.layer_indices = sorted(layer_indices)
        self.captured = {}
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        for name, module in self.model.named_modules():
            if "post_feedforward_layernorm" not in name:
                continue
            layer_idx = self._extract_layer_idx(name)
            if layer_idx is None or layer_idx not in self.layer_indices:
                continue

            def make_hook(idx):
                def hook_fn(module, input, output):
                    self.captured[idx] = output.detach()

                return hook_fn

            self.hooks.append(module.register_forward_hook(make_hook(layer_idx)))

    @staticmethod
    def _extract_layer_idx(name):
        for part in name.split("."):
            if part.isdigit():
                return int(part)
        return None

    def clear(self):
        self.captured.clear()

    def get_sae_features(self):
        """Project captured activations through SAE encoders.

        Returns:
            Tensor of shape [n_layers, n_tokens, d_sae].
        """
        features_by_layer = []
        for layer_idx in self.layer_indices:
            if layer_idx not in self.captured:
                raise ValueError(f"Layer {layer_idx} not captured")

            h_t = self.captured[layer_idx]
            if h_t.dim() == 3 and h_t.size(0) == 1:
                h_t = h_t.squeeze(0)

            sae = self.saes[layer_idx]
            with torch.no_grad():
                feat = sae.encode(h_t.float().to(sae.device))
            features_by_layer.append(feat.cpu())

        return torch.stack(features_by_layer)

    def remove(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()


def load_saes(release, layer_indices, width, l0, device):
    """Load SAEs for specified layers."""
    from sae_lens import SAE

    saes = {}
    for layer_idx in tqdm(layer_indices, desc="Loading SAEs"):
        sae_id = f"layer_{layer_idx}_width_{width}_l0_{l0}"
        sae = SAE.from_pretrained(
            release=release,
            sae_id=sae_id,
            device=device,
        )
        sae.eval()
        saes[layer_idx] = sae
    return saes


def main():
    args = parse_args()
    layer_indices = resolve_layer_indices(args.model_path, args.layers)
    print(f"Extracting SAE features for layers: {layer_indices}")

    with open(args.train_ids_path, "r", encoding="utf-8") as handle:
        id_map = json.load(handle)
        target_ids = set(id_map["t"] + id_map["f"])
    print(f"Loaded {len(target_ids)} target IDs for extraction.")

    with open(args.input_path, "r", encoding="utf-8") as handle:
        samples = [json.loads(line) for line in handle]

    for location in args.locations:
        os.makedirs(os.path.join(args.output_root, location), exist_ok=True)

    print(f"Loading model: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map=args.device_map,
    )
    model.eval()

    model_device = next(model.parameters()).device
    sae_device = str(model_device)
    print(f"Loading SAEs on device: {sae_device}")
    saes = load_saes(
        args.sae_release, layer_indices, args.sae_width, args.sae_l0, sae_device
    )

    first_sae = next(iter(saes.values()))
    first_cfg = sae_cfg_to_dict(first_sae)
    print(
        f"SAE dimensions: d_in={first_cfg.get('d_in')}, d_sae={first_cfg.get('d_sae')}"
    )

    metadata = build_metadata(args, layer_indices, first_cfg)
    ensure_metadata(args.output_root, metadata)

    extractor = SAEExtractor(model, saes, layer_indices)
    print(f"Installed {len(extractor.hooks)} hooks")

    skipped = {location: 0 for location in args.locations}
    extracted = {location: 0 for location in args.locations}

    for sample_dict in tqdm(samples, desc="Processing"):
        qid = next(iter(sample_dict))
        if qid not in target_ids:
            continue

        if all(
            os.path.exists(os.path.join(args.output_root, loc, f"act_{qid}.npy"))
            for loc in args.locations
        ):
            continue

        data = sample_dict[qid]
        extractor.clear()

        msgs = [
            {"role": "user", "content": data["question"]},
            {"role": "assistant", "content": data["response"]},
        ]
        input_ids = unwrap_chat_template_output(
            tokenizer.apply_chat_template(
                msgs, return_tensors="pt", add_generation_prompt=False
            )
        ).to(model_device)

        with torch.no_grad():
            model(input_ids)

        regions = get_region_indices(
            input_ids,
            tokenizer,
            data["question"],
            data["response"],
            data["answer_tokens"],
        )
        sae_features = extractor.get_sae_features()

        for location in args.locations:
            save_path = os.path.join(args.output_root, location, f"act_{qid}.npy")
            if os.path.exists(save_path):
                continue

            selected = select_token_activations(sae_features, location, regions)
            if selected is None:
                skipped[location] += 1
                continue

            aggregated = aggregate_token_activations(selected, args.method)
            np.save(save_path, aggregated.numpy())
            extracted[location] += 1

    print("\nDone.")
    for location in args.locations:
        print(
            f"  {location}: extracted={extracted[location]}, skipped={skipped[location]}"
        )
    extractor.remove()


if __name__ == "__main__":
    main()
