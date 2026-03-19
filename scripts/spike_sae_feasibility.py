"""Phase 1 feasibility spike: load Gemma Scope 2 SAE and verify dimensions.

Run on CPU (no model needed) to check SAE loads and dimensions match
Gemma-3-4B-IT architecture. GPU check (hook compatibility) is Phase 1.3.

Usage:
    uv run python scripts/spike_sae_feasibility.py
    uv run python scripts/spike_sae_feasibility.py --gpu  # include hook test
"""

import argparse

import torch


def sae_cfg_to_dict(sae) -> dict:
    """Convert an SAE cfg object into a regular dict for reporting/tests."""
    cfg = sae.cfg
    if hasattr(cfg, "to_dict"):
        return cfg.to_dict()
    return {key: value for key, value in vars(cfg).items() if not key.startswith("_")}


def inspect_sae(release: str, sae_id: str, device: str = "cpu"):
    """Load an SAE and print its configuration and dimensions."""
    from sae_lens import SAE

    print(f"Loading SAE: release={release!r}, sae_id={sae_id!r}")
    sae = SAE.from_pretrained(
        release=release,
        sae_id=sae_id,
        device=device,
    )
    cfg_dict = sae_cfg_to_dict(sae)
    print(f"\nSAE loaded successfully on device={device}")
    print(f"  Type: {type(sae).__name__}")
    print(f"  Config keys: {sorted(cfg_dict.keys())}")
    print("\n  cfg_dict contents:")
    for k, v in sorted(cfg_dict.items()):
        print(f"    {k}: {v}")

    # Key dimensions
    d_in = cfg_dict.get("d_in", None)
    d_sae = cfg_dict.get("d_sae", None)
    print(f"\n  d_in (SAE input dim): {d_in}")
    print(f"  d_sae (SAE feature count): {d_sae}")

    # Check encoder/decoder weight shapes
    if hasattr(sae, "W_enc"):
        print(f"  W_enc shape: {sae.W_enc.shape}")
    if hasattr(sae, "W_dec"):
        print(f"  W_dec shape: {sae.W_dec.shape}")
    if hasattr(sae, "b_enc"):
        print(f"  b_enc shape: {sae.b_enc.shape}")
    if hasattr(sae, "b_dec"):
        print(f"  b_dec shape: {sae.b_dec.shape}")

    # Test encode/decode with dummy data
    if d_in:
        dummy = torch.randn(1, 5, d_in, device=device)  # batch=1, seq=5
        print(f"\n  Dummy input shape: {dummy.shape}")
        features = sae.encode(dummy)
        print(f"  Encoded features shape: {features.shape}")
        reconstructed = sae.decode(features)
        print(f"  Reconstructed shape: {reconstructed.shape}")

        # Sparsity check
        n_active = (features > 0).float().sum(dim=-1).mean().item()
        print(f"  Mean active features per token: {n_active:.1f} / {d_sae}")

    return sae, cfg_dict


def check_model_dims(model_path: str):
    """Load model config and report MLP dimensions."""
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(model_path)

    # Handle nested text_config (common in multimodal models)
    if hasattr(config, "text_config"):
        text_cfg = config.text_config
    else:
        text_cfg = config

    hidden_size = text_cfg.hidden_size
    intermediate_size = text_cfg.intermediate_size
    num_hidden_layers = text_cfg.num_hidden_layers

    print(f"\nModel: {model_path}")
    print(f"  hidden_size (residual dim): {hidden_size}")
    print(f"  intermediate_size (MLP neurons): {intermediate_size}")
    print(f"  num_hidden_layers: {num_hidden_layers}")
    print(
        f"  MLP architecture: up_proj/gate_proj [{hidden_size}→{intermediate_size}]"
        f" → down_proj [{intermediate_size}→{hidden_size}]"
    )
    print(f"\n  post_feedforward_layernorm output dim: {hidden_size}")
    print(f"  down_proj output dim before RMSNorm: {hidden_size}")
    print(f"  down_proj input dim (CETT neuron activations): {intermediate_size}")
    print(
        "  Gemma Scope 2 MLP SAEs are trained on post_feedforward_layernorm output, "
        f"so SAE d_in should match hidden_size = {hidden_size}"
    )

    return {
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "num_hidden_layers": num_hidden_layers,
    }


def gpu_hook_test(model_path: str, release: str, sae_id: str):
    """Load model + SAE on GPU and verify hook captures match SAE input dim."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from sae_lens import SAE

    print("\n" + "=" * 60)
    print("GPU Hook Compatibility Test")
    print("=" * 60)

    device = "cuda:0"

    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    model.eval()

    print("Loading SAE...")
    sae = SAE.from_pretrained(
        release=release,
        sae_id=sae_id,
        device=device,
    )
    sae.eval()
    cfg_dict = sae_cfg_to_dict(sae)

    captured_outputs = []

    def capture_post_ff_output(module, input, output):
        captured_outputs.append(output.detach())

    target_layer = 13
    hooked_name = None
    for name, module in model.named_modules():
        if f"layers.{target_layer}" in name and "post_feedforward_layernorm" in name:
            module.register_forward_hook(capture_post_ff_output)
            hooked_name = name
            print(f"Hooked: {name}")
            break
    if hooked_name is None:
        raise RuntimeError(
            "Could not find target post_feedforward_layernorm module for GPU hook test"
        )

    # Run a single forward pass
    msgs = [{"role": "user", "content": "What is the capital of France?"}]
    inputs = tokenizer.apply_chat_template(
        msgs, return_tensors="pt", add_generation_prompt=True
    )
    if hasattr(inputs, "input_ids"):
        input_ids = inputs["input_ids"].to(device)
    else:
        input_ids = inputs.to(device)

    print(f"Input shape: {input_ids.shape}")
    with torch.no_grad():
        model(input_ids)

    # Check captured tensors
    print(f"\nCaptured {len(captured_outputs)} post-feedforward outputs")
    if captured_outputs:
        h_t = captured_outputs[0]
        print(f"  post_feedforward_layernorm output shape: {h_t.shape}")
        print(f"  post_feedforward_layernorm output dim: {h_t.shape[-1]}")
        print(f"  SAE d_in: {cfg_dict['d_in']}")
        print(f"  Dimensions match: {h_t.shape[-1] == cfg_dict['d_in']}")

        if h_t.shape[-1] == cfg_dict["d_in"]:
            h_t_float = h_t.float()
            features = sae.encode(h_t_float)
            print("\n  SAE encode on real activations:")
            print(f"    Input: {h_t_float.shape} → Features: {features.shape}")
            n_active = (features > 0).float().sum(dim=-1).mean().item()
            print(f"    Mean active features per token: {n_active:.1f}")

            reconstructed = sae.decode(features)
            error = (h_t_float - reconstructed).norm() / h_t_float.norm()
            print(f"    Reconstruction error (relative L2): {error.item():.4f}")
        else:
            print(
                "\n  *** DIMENSION MISMATCH — SAE cannot be used on "
                "post_feedforward_layernorm output ***"
            )

    print("\nFeasibility verdict:", end=" ")
    if captured_outputs and h_t.shape[-1] == cfg_dict["d_in"]:
        print("PASS — SAE is compatible with post_feedforward_layernorm output")
    else:
        print("FAIL — dimension mismatch")


def main():
    parser = argparse.ArgumentParser(description="SAE feasibility spike")
    parser.add_argument("--gpu", action="store_true", help="Run GPU hook test")
    parser.add_argument(
        "--model_path",
        default="google/gemma-3-4b-it",
        help="Model to check dimensions against",
    )
    parser.add_argument(
        "--release",
        default="gemma-scope-2-4b-it-mlp-all",
        help="SAE release name",
    )
    parser.add_argument(
        "--sae_id",
        default="layer_13_width_16k_l0_small",
        help="SAE ID within the release",
    )
    args = parser.parse_args()

    # Phase 1.2: Load and inspect SAE
    print("=" * 60)
    print("Phase 1.2: Load and Inspect SAE")
    print("=" * 60)
    sae, cfg_dict = inspect_sae(args.release, args.sae_id)

    # Check model dimensions
    print("\n" + "=" * 60)
    print("Model Dimension Check")
    print("=" * 60)
    model_dims = check_model_dims(args.model_path)

    # Compatibility check
    d_in = cfg_dict.get("d_in")
    if d_in:
        print(f"\n{'=' * 60}")
        print("Compatibility Summary")
        print(f"{'=' * 60}")
        print(f"  SAE d_in: {d_in}")
        print(f"  Model hidden_size: {model_dims['hidden_size']}")
        print(f"  Model intermediate_size: {model_dims['intermediate_size']}")

        if d_in == model_dims["hidden_size"]:
            print("  ✓ SAE d_in matches hidden_size")
            print("    → Hook at post_feedforward_layernorm output is correct")
        elif d_in == model_dims["intermediate_size"]:
            print("  ✗ SAE d_in matches intermediate_size instead of hidden_size")
            print("    → This would indicate the wrong hook point for Gemma Scope 2")
        else:
            print("  ✗ SAE d_in matches neither — BLOCKED")

    # Phase 1.3: GPU hook test
    if args.gpu:
        gpu_hook_test(args.model_path, args.release, args.sae_id)


if __name__ == "__main__":
    main()
