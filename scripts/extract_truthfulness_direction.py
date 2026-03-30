"""Extract truthfulness/hallucination directions via difference-in-means.

Reuses the core activation-collection and direction-computation logic from
extract_direction.py, but operates on the truthfulness contrastive dataset
(D4) instead of refusal data (D2).

For each decoder layer, computes:
  direction = normalize(mean(hallucinatory_acts) - mean(truthful_acts))

Convention: positive projection = more hallucinatory, negative = more truthful.
This parallels D2 where positive projection = more harmful/refusing.

Validates on the val split with projection-based separation scores.
Optionally computes cosine similarity with refusal directions to check for
direction independence.

Usage:
    uv run python scripts/extract_truthfulness_direction.py \\
        --model_path google/gemma-3-4b-it \\
        --contrastive_path data/contrastive/truthfulness/truthfulness_contrastive.jsonl \\
        --output_dir data/contrastive/truthfulness/directions \\
        --device_map cuda:0

    # With refusal-direction comparison:
    uv run python scripts/extract_truthfulness_direction.py \\
        --model_path google/gemma-3-4b-it \\
        --contrastive_path data/contrastive/truthfulness/truthfulness_contrastive.jsonl \\
        --output_dir data/contrastive/truthfulness/directions \\
        --refusal_directions_path data/contrastive/refusal/directions/refusal_directions.pt \\
        --device_map cuda:0
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, cast

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from extract_direction import (
    collect_activations,
    compute_directions,
    compute_separation_scores,
    fingerprint_records,
    _get_text_config,
)
from utils import (
    finish_run_provenance,
    format_alpha_label,
    json_dumps,
    provenance_error_message,
    provenance_status_for_exception,
    start_run_provenance,
)


# ---------------------------------------------------------------------------
# Data loading (truthfulness-specific)
# ---------------------------------------------------------------------------


def load_truthfulness_splits(
    path: str,
) -> dict[str, list[dict]]:
    """Load truthfulness contrastive JSONL and partition into splits + labels.

    Returns dict with keys like 'train_truthful', 'train_hallucinatory',
    'val_truthful', 'val_hallucinatory'.
    """
    splits: dict[str, list[dict]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            key = f"{record['split']}_{record['label']}"
            splits.setdefault(key, []).append(record)
    return splits


# ---------------------------------------------------------------------------
# Refusal direction comparison
# ---------------------------------------------------------------------------


def compare_with_refusal_directions(
    truthfulness_dirs: dict[int, torch.Tensor],
    refusal_dirs_path: str,
    n_layers: int,
) -> list[dict[str, Any]]:
    """Compute per-layer cosine similarity between truthfulness and refusal directions."""
    refusal_dirs = torch.load(refusal_dirs_path, map_location="cpu", weights_only=True)
    results = []
    for i in range(n_layers):
        if i not in refusal_dirs or i not in truthfulness_dirs:
            continue
        t_dir = truthfulness_dirs[i].float()
        r_dir = refusal_dirs[i].float()
        cos_sim = (t_dir @ r_dir).item() / (t_dir.norm().item() * r_dir.norm().item())
        results.append(
            {
                "layer": i,
                "cosine_similarity": round(cos_sim, 6),
                "abs_cosine_similarity": round(abs(cos_sim), 6),
            }
        )
    return results


# ---------------------------------------------------------------------------
# Pilot intervention check
# ---------------------------------------------------------------------------


FAITHEVAL_FULL_SAMPLE_COUNT = 1000
DEFAULT_FAITHEVAL_CALIBRATION_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]


def classify_pilot_output(text: str) -> list[str]:
    """Return malformation reasons for clearly corrupted generations.

    The pilot gate is meant to catch decoding corruption, not to enforce
    verbosity. One-token or one-word answers are valid on the truthfulness
    contrastive set and should not fail this check.
    """
    stripped = text.strip()
    if not stripped:
        return ["empty_output"]

    reasons: list[str] = []
    non_whitespace = [ch for ch in stripped if not ch.isspace()]
    if non_whitespace:
        alnum_fraction = sum(ch.isalnum() for ch in non_whitespace) / len(
            non_whitespace
        )
        if len(non_whitespace) >= 24 and alnum_fraction < 0.25:
            reasons.append("symbol_heavy_output")

    nonempty_lines = [line.strip() for line in stripped.splitlines() if line.strip()]
    if len(nonempty_lines) >= 4:
        unique_lines = set(nonempty_lines)
        longest_line = max(len(line) for line in nonempty_lines)
        if longest_line <= 4 and len(unique_lines) <= max(2, len(nonempty_lines) // 4):
            reasons.append("repetitive_short_lines")

    compact = "".join(non_whitespace)
    if len(compact) >= 24:
        longest_run = 0
        current_run = 0
        previous_char = None
        for ch in compact:
            if ch == previous_char:
                current_run += 1
            else:
                previous_char = ch
                current_run = 1
            longest_run = max(longest_run, current_run)
        if longest_run >= 8 and "long_character_run" not in reasons:
            reasons.append("long_character_run")

    return reasons


def build_recommended_faitheval_sweep(
    *,
    model_path: str,
    direction_path: Path,
    best_layer: int,
    device_map: str,
    observed_corruption_beta: float | None,
) -> dict[str, Any]:
    """Return the first full-power FaithEval sweep to run after extraction."""
    if observed_corruption_beta is None:
        alphas = DEFAULT_FAITHEVAL_CALIBRATION_GRID
        calibration_note = (
            "No malformed pilot outputs observed at beta=1.0; start with the "
            "default bracketed sweep before considering larger betas."
        )
    else:
        alphas = sorted(
            {
                0.0,
                round(observed_corruption_beta * 0.25, 3),
                round(observed_corruption_beta * 0.5, 3),
                round(observed_corruption_beta * 0.75, 3),
                round(observed_corruption_beta, 3),
            }
        )
        calibration_note = (
            "Pilot malformed output was observed at beta="
            f"{format_alpha_label(observed_corruption_beta)}; bracket the onset of "
            "corruption on the full benchmark instead of sweeping past it."
        )

    alpha_args = " ".join(format_alpha_label(alpha) for alpha in alphas)
    command = (
        "uv run python scripts/run_intervention.py "
        f"--model_path {model_path} "
        "--benchmark faitheval "
        "--intervention_mode direction "
        f"--direction_path {direction_path.as_posix()} "
        "--direction_mode ablate "
        f"--direction_layers {best_layer} "
        f"--alphas {alpha_args} "
        f"--max_samples {FAITHEVAL_FULL_SAMPLE_COUNT} "
        f"--device_map {device_map} "
        "--seed 42"
    )
    return {
        "benchmark": "faitheval",
        "max_samples": FAITHEVAL_FULL_SAMPLE_COUNT,
        "alphas": alphas,
        "direction_mode": "ablate",
        "direction_layers": [best_layer],
        "observed_corruption_beta": observed_corruption_beta,
        "calibration_note": calibration_note,
        "command": command,
    }


def run_pilot_check(
    model: torch.nn.Module,
    tokenizer: Any,
    directions: dict[int, torch.Tensor],
    val_records: list[dict],
    best_layer: int,
    device: torch.device,
    *,
    n_samples: int = 5,
    max_new_tokens: int = 256,
) -> dict[str, Any]:
    """Quick pilot: ablate truthfulness direction on a few val samples.

    Checks that the intervention doesn't produce malformed output (empty,
    repeated symbol patterns, garbage). This is NOT a full steering eval —
    just a basic sanity gate before the first full FaithEval sweep.
    """
    from intervene_direction import DirectionScaler

    results: list[dict[str, Any]] = []

    def generate(input_ids: torch.Tensor) -> str:
        with torch.no_grad():
            output_ids = cast(Any, model).generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        new_tokens = output_ids[0, input_ids.shape[1] :]
        return tokenizer.decode(new_tokens, skip_special_tokens=True)

    def tokenize(text: str) -> torch.Tensor:
        messages = [{"role": "user", "content": text}]
        inputs = tokenizer.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True
        )
        if hasattr(inputs, "input_ids"):
            return inputs["input_ids"].to(device)
        return inputs.to(device)

    pilot_samples = val_records[:n_samples]

    print(f"\n{'=' * 60}")
    print(f"Pilot Check — Ablate truthfulness direction (layer={best_layer})")
    print(f"{'=' * 60}")

    for record in pilot_samples:
        input_ids = tokenize(record["text"])

        # Baseline
        baseline = generate(input_ids)

        # Ablate (beta=1.0)
        scaler = DirectionScaler(
            model, directions, device, mode="ablate", layers=[best_layer]
        )
        scaler.alpha = 1.0
        ablated = generate(input_ids)
        scaler.remove()

        malformation_reasons = classify_pilot_output(ablated)
        is_malformed = bool(malformation_reasons)

        result = {
            "qid": record.get("qid", record["id"]),
            "label": record["label"],
            "prompt": record["text"][:80],
            "baseline_start": baseline[:200],
            "ablated_start": ablated[:200],
            "baseline_len": len(baseline),
            "ablated_len": len(ablated),
            "is_malformed": is_malformed,
            "malformation_reasons": malformation_reasons,
        }
        results.append(result)

        print(f"\n[{record['label']}] {record['text'][:60]}...")
        print(f"  Baseline ({len(baseline)} chars): {baseline[:150]}...")
        print(f"  Ablated  ({len(ablated)} chars): {ablated[:150]}...")
        if is_malformed:
            print(f"  ⚠ MALFORMED OUTPUT DETECTED: {', '.join(malformation_reasons)}")

    n_malformed = sum(1 for r in results if r["is_malformed"])
    return {
        "n_samples": len(results),
        "n_malformed": n_malformed,
        "malformed_rate": round(n_malformed / max(len(results), 1), 4),
        "passes": n_malformed == 0,
        "observed_corruption_beta": 1.0 if n_malformed else None,
        # Deprecated aliases retained for artifact compatibility.
        "n_degenerate": n_malformed,
        "degenerate_rate": round(n_malformed / max(len(results), 1), 4),
        "samples": results,
    }


# ---------------------------------------------------------------------------
# CLI and main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract truthfulness directions via difference-in-means"
    )
    p.add_argument(
        "--model_path",
        type=str,
        default="google/gemma-3-4b-it",
    )
    p.add_argument(
        "--contrastive_path",
        type=str,
        default="data/contrastive/truthfulness/truthfulness_contrastive.jsonl",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default="data/contrastive/truthfulness/directions",
    )
    p.add_argument(
        "--device_map",
        type=str,
        default="cuda:0",
    )
    p.add_argument(
        "--refusal_directions_path",
        type=str,
        default=None,
        help="Optional: path to refusal directions for cosine similarity comparison",
    )
    p.add_argument(
        "--pilot_check",
        action="store_true",
        help="Run a small pilot intervention to check for degenerate outputs",
    )
    p.add_argument(
        "--pilot_max_new_tokens",
        type=int,
        default=256,
        help="Max new tokens for pilot intervention generation",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    directions_path = output_dir / "truthfulness_directions.pt"
    metadata_path = output_dir / "extraction_metadata.json"

    provenance_handle = start_run_provenance(
        args,
        primary_target=str(output_dir),
        output_targets=[str(directions_path), str(metadata_path)],
        primary_target_is_dir=True,
    )
    provenance_status = "completed"
    provenance_extra: dict = {}

    try:
        # Load contrastive data
        splits = load_truthfulness_splits(args.contrastive_path)
        train_hallucinatory = splits.get("train_hallucinatory", [])
        train_truthful = splits.get("train_truthful", [])
        val_hallucinatory = splits.get("val_hallucinatory", [])
        val_truthful = splits.get("val_truthful", [])

        print(
            f"Train: {len(train_hallucinatory)} hallucinatory, "
            f"{len(train_truthful)} truthful"
        )
        print(
            f"Val:   {len(val_hallucinatory)} hallucinatory, "
            f"{len(val_truthful)} truthful"
        )

        if not train_hallucinatory or not train_truthful:
            raise ValueError(
                "Need both hallucinatory and truthful train samples for "
                "direction extraction"
            )

        # Load model
        print(f"\nLoading model: {args.model_path}")
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, dtype=torch.bfloat16, device_map=args.device_map
        )
        model.eval()
        device = next(model.parameters()).device
        cfg = _get_text_config(model)
        n_layers = int(getattr(cfg, "num_hidden_layers"))
        hidden_dim = int(getattr(cfg, "hidden_size"))
        print(f"Model loaded: {n_layers} layers, hidden_dim={hidden_dim}")

        # Collect train activations
        # Convention: hallucinatory = "harmful" position (positive direction)
        # truthful = "harmless" position (negative direction)
        print("\nCollecting hallucinatory train activations...")
        t0 = time.perf_counter()
        hallucinatory_acts = collect_activations(
            model,
            tokenizer,
            [r["text"] for r in train_hallucinatory],
            n_layers,
            device,
        )
        print("Collecting truthful train activations...")
        truthful_acts = collect_activations(
            model,
            tokenizer,
            [r["text"] for r in train_truthful],
            n_layers,
            device,
        )
        collection_time = round(time.perf_counter() - t0, 2)
        print(f"Activation collection: {collection_time}s")

        # Compute directions: hallucinatory - truthful (parallels harmful - harmless)
        directions = compute_directions(hallucinatory_acts, truthful_acts, n_layers)
        print(f"Computed {len(directions)} direction vectors")

        # Validate on val split
        print("\nCollecting val activations for separation diagnostics...")
        val_hallucinatory_acts = collect_activations(
            model,
            tokenizer,
            [r["text"] for r in val_hallucinatory],
            n_layers,
            device,
        )
        val_truthful_acts = collect_activations(
            model,
            tokenizer,
            [r["text"] for r in val_truthful],
            n_layers,
            device,
        )

        separation_scores = compute_separation_scores(
            directions, val_hallucinatory_acts, val_truthful_acts, n_layers
        )

        # Find best layer
        best = max(separation_scores, key=lambda x: x["accuracy"])
        best_layer = best["layer"]
        print(
            f"\nBest layer: {best_layer} (accuracy={best['accuracy']:.4f}, "
            f"separation={best['separation']:.4f})"
        )

        # Top 5 layers
        by_accuracy = sorted(separation_scores, key=lambda x: -x["accuracy"])
        print("\nTop 5 layers by separation accuracy:")
        for s in by_accuracy[:5]:
            print(
                f"  Layer {s['layer']:2d}: accuracy={s['accuracy']:.4f}, "
                f"separation={s['separation']:.4f}"
            )

        # Save directions
        torch.save(
            {i: directions[i] for i in range(n_layers)},
            str(directions_path),
        )
        print(f"\nSaved directions to {directions_path}")

        # Refusal direction comparison
        refusal_comparison = None
        if args.refusal_directions_path:
            refusal_path = Path(args.refusal_directions_path)
            if refusal_path.exists():
                print("\nComparing with refusal directions...")
                refusal_comparison = compare_with_refusal_directions(
                    directions, str(refusal_path), n_layers
                )
                # Report top cosine similarities
                by_abs_cos = sorted(
                    refusal_comparison,
                    key=lambda x: -x["abs_cosine_similarity"],
                )
                print("Top 5 layers by |cosine similarity| with refusal direction:")
                for s in by_abs_cos[:5]:
                    print(
                        f"  Layer {s['layer']:2d}: cos_sim={s['cosine_similarity']:.4f}"
                    )
                # Check if best layer is a refusal clone
                best_refusal = next(
                    (s for s in refusal_comparison if s["layer"] == best_layer), None
                )
                if best_refusal:
                    abs_cos = best_refusal["abs_cosine_similarity"]
                    if abs_cos > 0.5:
                        print(
                            f"\n⚠ WARNING: Best truthfulness layer {best_layer} has "
                            f"|cos_sim|={abs_cos:.4f} with refusal direction — "
                            f"possible refusal contamination"
                        )
                    else:
                        print(
                            f"\n✓ Best layer {best_layer} has low refusal overlap "
                            f"(|cos_sim|={abs_cos:.4f})"
                        )

        recommended_faitheval_sweep = build_recommended_faitheval_sweep(
            model_path=args.model_path,
            direction_path=directions_path,
            best_layer=best_layer,
            device_map=args.device_map,
            observed_corruption_beta=None,
        )

        # Pilot intervention check
        pilot_results = None
        if args.pilot_check:
            # Use a mix of truthful and hallucinatory val samples
            pilot_records = val_hallucinatory[:3] + val_truthful[:2]
            pilot_results = run_pilot_check(
                model,
                tokenizer,
                directions,
                pilot_records,
                best_layer,
                device,
                max_new_tokens=args.pilot_max_new_tokens,
            )
            recommended_faitheval_sweep = build_recommended_faitheval_sweep(
                model_path=args.model_path,
                direction_path=directions_path,
                best_layer=best_layer,
                device_map=args.device_map,
                observed_corruption_beta=pilot_results["observed_corruption_beta"],
            )
            if pilot_results["passes"]:
                print("\n✓ Pilot check passed: no malformed outputs")
            else:
                print(
                    f"\n⚠ Pilot check: {pilot_results['n_malformed']}/"
                    f"{pilot_results['n_samples']} malformed outputs"
                )
            print(
                "\nRecommended first FaithEval sweep:\n"
                f"  {recommended_faitheval_sweep['command']}"
            )

        # Save metadata
        metadata: dict[str, Any] = {
            "model_path": args.model_path,
            "contrastive_path": str(Path(args.contrastive_path).resolve()),
            "n_layers": n_layers,
            "hidden_dim": hidden_dim,
            "token_position": "last",
            "dtype_extract": "bfloat16",
            "dtype_accumulate": "float32",
            "normalization": "L2",
            "direction_convention": (
                "direction = normalize(mean(hallucinatory) - mean(truthful)). "
                "Positive projection = more hallucinatory."
            ),
            "comparison_surface": (
                "Last-token residual stream at decoder layer output. "
                "Same surface as D2 refusal direction extraction."
            ),
            "train_hallucinatory_count": len(train_hallucinatory),
            "train_truthful_count": len(train_truthful),
            "val_hallucinatory_count": len(val_hallucinatory),
            "val_truthful_count": len(val_truthful),
            "train_hallucinatory_fingerprint": fingerprint_records(train_hallucinatory),
            "train_truthful_fingerprint": fingerprint_records(train_truthful),
            "best_layer": best_layer,
            "best_layer_accuracy": best["accuracy"],
            "best_layer_separation": best["separation"],
            "separation_scores": separation_scores,
            "collection_time_s": collection_time,
            "recommended_faitheval_sweep": recommended_faitheval_sweep,
        }

        if refusal_comparison is not None:
            metadata["refusal_direction_comparison"] = {
                "refusal_directions_path": args.refusal_directions_path,
                "per_layer": refusal_comparison,
                "best_layer_cosine_similarity": next(
                    (
                        s["cosine_similarity"]
                        for s in refusal_comparison
                        if s["layer"] == best_layer
                    ),
                    None,
                ),
            }

        if pilot_results is not None:
            metadata["pilot_check"] = {
                "passes": pilot_results["passes"],
                "n_malformed": pilot_results["n_malformed"],
                "n_samples": pilot_results["n_samples"],
                "observed_corruption_beta": pilot_results["observed_corruption_beta"],
                "n_degenerate": pilot_results["n_degenerate"],
            }
            # Save full pilot results separately
            pilot_path = output_dir / "pilot_check_results.json"
            pilot_path.write_text(json_dumps(pilot_results), encoding="utf-8")
            print(f"Saved pilot check results to {pilot_path}")

        metadata_path.write_text(json_dumps(metadata), encoding="utf-8")
        print(f"Saved metadata to {metadata_path}")

        provenance_extra.update(
            {
                "best_layer": best_layer,
                "best_layer_accuracy": best["accuracy"],
                "best_layer_separation": best["separation"],
                "train_hallucinatory_count": len(train_hallucinatory),
                "train_truthful_count": len(train_truthful),
            }
        )
        if pilot_results is not None:
            provenance_extra["pilot_passes"] = pilot_results["passes"]

    except BaseException as exc:
        provenance_status = provenance_status_for_exception(exc)
        provenance_extra["error"] = provenance_error_message(exc)
        raise
    finally:
        finish_run_provenance(provenance_handle, provenance_status, provenance_extra)


if __name__ == "__main__":
    main()
