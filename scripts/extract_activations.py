import argparse
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


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


def main():
    args = parse_args()
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

    # Prepare directories
    for loc in args.locations:
        os.makedirs(os.path.join(args.output_root, loc), exist_ok=True)

    with open(args.input_path, "r", encoding="utf-8") as f:
        samples = [json.loads(line) for line in f]

    for sample_dict in tqdm(samples, desc="Processing"):
        qid = list(sample_dict.keys())[0]
        if qid not in target_ids:
            continue
        # Skip if all locations already extracted
        if all(
            os.path.exists(os.path.join(args.output_root, loc, f"act_{qid}.npy"))
            for loc in args.locations
        ):
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
        for loc in args.locations:
            save_path = os.path.join(args.output_root, loc, f"act_{qid}.npy")
            if os.path.exists(save_path):
                continue  # Resume support: skip already-extracted QIDs

            selected_cett = select_token_activations(cett_full, loc, regions)
            if selected_cett is None:
                continue

            final_act = aggregate_token_activations(selected_cett, args.method)

            np.save(save_path, final_act.cpu().float().numpy())


if __name__ == "__main__":
    main()
