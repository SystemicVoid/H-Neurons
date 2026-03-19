"""Extract SAE feature activations for TriviaQA samples.

Hooks at post_feedforward_layernorm output (the exact point Gemma Scope 2
SAEs are trained on), projects through SAE encoder, and saves per-sample
sparse feature vectors aggregated over answer-token positions.

Usage:
    uv run python scripts/extract_sae_activations.py \
        --model_path google/gemma-3-4b-it \
        --input_path data/gemma3_4b/pipeline/answer_tokens.jsonl \
        --train_ids_path data/gemma3_4b/pipeline/train_qids.json \
        --output_root data/gemma3_4b/pipeline/activations_sae/answer_tokens \
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

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from extract_activations import get_region_indices, unwrap_chat_template_output


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
        help="Root directory for saving SAE feature .npy files",
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
    parser.add_argument("--method", type=str, choices=["mean", "max"], default="mean")
    return parser.parse_args()


class SAEExtractor:
    """Captures post_feedforward_layernorm outputs and projects through SAEs."""

    def __init__(self, model, saes, layer_indices):
        """
        Args:
            model: HuggingFace model.
            saes: dict mapping layer_idx -> loaded SAE object.
            layer_indices: list of layer indices to hook.
        """
        self.model = model
        self.saes = saes
        self.layer_indices = sorted(layer_indices)
        self.captured = {}  # layer_idx -> tensor
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        for name, module in self.model.named_modules():
            if "post_feedforward_layernorm" not in name:
                continue
            # Extract layer index from name like "model.layers.13.post_feedforward_layernorm"
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

    def get_sae_features(self, token_slice=None):
        """Project captured activations through SAE encoders.

        Args:
            token_slice: Optional (start, end) tuple to select token positions.

        Returns:
            Tensor of shape [n_layers, n_tokens, d_sae] with SAE feature activations.
        """
        features_by_layer = []
        for layer_idx in self.layer_indices:
            if layer_idx not in self.captured:
                raise ValueError(f"Layer {layer_idx} not captured")

            h_t = self.captured[layer_idx]  # [batch, seq, hidden_size]
            if h_t.dim() == 3 and h_t.size(0) == 1:
                h_t = h_t.squeeze(0)  # [seq, hidden_size]

            if token_slice is not None:
                start, end = token_slice
                h_t = h_t[start:end]  # [n_tokens, hidden_size]

            sae = self.saes[layer_idx]
            with torch.no_grad():
                # SAE expects float32 input
                feat = sae.encode(h_t.float().to(sae.device))
            features_by_layer.append(feat.cpu())

        return torch.stack(features_by_layer)  # [n_layers, n_tokens, d_sae]

    def remove(self):
        for h in self.hooks:
            h.remove()
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

    # Determine layer indices
    if args.layers == ["all"]:
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(args.model_path)
        text_cfg = config.text_config if hasattr(config, "text_config") else config
        layer_indices = list(range(text_cfg.num_hidden_layers))
    else:
        layer_indices = [int(x) for x in args.layers]

    print(f"Extracting SAE features for layers: {layer_indices}")

    # Load target sample IDs
    with open(args.train_ids_path, "r") as f:
        id_map = json.load(f)
        target_ids = set(id_map["t"] + id_map["f"])
    print(f"Loaded {len(target_ids)} target IDs for extraction.")

    # Load samples
    with open(args.input_path, "r", encoding="utf-8") as f:
        samples = [json.loads(line) for line in f]

    # Prepare output directory
    os.makedirs(args.output_root, exist_ok=True)

    # Load model
    print(f"Loading model: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map=args.device_map,
    )
    model.eval()

    # Load SAEs — put on same device as model
    model_device = next(model.parameters()).device
    sae_device = str(model_device)
    print(f"Loading SAEs on device: {sae_device}")
    saes = load_saes(
        args.sae_release, layer_indices, args.sae_width, args.sae_l0, sae_device
    )

    # Check dimensions match
    first_sae = next(iter(saes.values()))
    d_sae = first_sae.cfg.d_sae
    d_in = first_sae.cfg.d_in
    print(f"SAE dimensions: d_in={d_in}, d_sae={d_sae}")

    # Install hooks
    extractor = SAEExtractor(model, saes, layer_indices)
    print(f"Installed {len(extractor.hooks)} hooks")

    # Process samples
    skipped = 0
    extracted = 0
    for sample_dict in tqdm(samples, desc="Processing"):
        qid = list(sample_dict.keys())[0]
        if qid not in target_ids:
            continue

        save_path = os.path.join(args.output_root, f"act_{qid}.npy")
        if os.path.exists(save_path):
            continue  # Resume support

        data = sample_dict[qid]
        extractor.clear()

        # Forward pass
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

        # Find answer token region
        regions = get_region_indices(
            input_ids,
            tokenizer,
            data["question"],
            data["response"],
            data["answer_tokens"],
        )
        answer_region = regions.get("answer_tokens")
        if answer_region is None:
            skipped += 1
            continue

        # Extract SAE features for answer tokens
        sae_features = extractor.get_sae_features(
            token_slice=answer_region
        )  # [n_layers, n_answer_tokens, d_sae]

        # Aggregate over answer tokens
        if args.method == "mean":
            aggregated = sae_features.mean(dim=1)  # [n_layers, d_sae]
        else:
            aggregated, _ = sae_features.max(dim=1)

        np.save(save_path, aggregated.numpy())
        extracted += 1

    print(f"\nDone. Extracted: {extracted}, Skipped (no answer region): {skipped}")
    extractor.remove()


if __name__ == "__main__":
    main()
