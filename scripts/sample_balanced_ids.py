import json
import argparse
import random
from tqdm import tqdm

from utils import (
    finish_run_provenance,
    provenance_error_message,
    provenance_status_for_exception,
    start_run_provenance,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sample balanced True and False IDs for training."
    )
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to extraction results (jsonl)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="data/train_qids.json",
        help="Path to save balanced IDs (json)",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1000,
        help="Number of samples per class (default: 1000)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--exclude_path",
        type=str,
        default=None,
        help="Path to a previously-sampled qids.json whose IDs should be excluded (for disjoint splits)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    provenance_handle = start_run_provenance(
        args,
        primary_target=args.output_path,
        output_targets=[args.output_path],
    )
    provenance_status = "completed"
    provenance_extra = {}
    random.seed(args.seed)

    try:
        true_ids = []
        false_ids = []

        # Load exclusion set if provided (for disjoint train/test splits)
        exclude_ids = set()
        if args.exclude_path:
            with open(args.exclude_path, "r") as f:
                excl = json.load(f)
                exclude_ids = set(excl["t"] + excl["f"])
            print(f"Excluding {len(exclude_ids)} IDs from {args.exclude_path}")

        # Categorize IDs based on labels
        with open(args.input_path, "r", encoding="utf-8") as f:
            for line in tqdm(f, desc="Reading IDs"):
                try:
                    data = json.loads(line)
                    qid = list(data.keys())[0]
                    if qid in exclude_ids:
                        continue
                    label = data[qid].get("judge")

                    if label == "true":
                        true_ids.append(qid)
                    elif label == "false":
                        false_ids.append(qid)
                except Exception as e:
                    print(f"Skipping line due to error: {e}")

        print(f"Total available - True: {len(true_ids)}, False: {len(false_ids)}")

        # Determine final sample count based on availability
        actual_samples = min(args.num_samples, len(true_ids), len(false_ids))
        if actual_samples < args.num_samples:
            print(
                f"Warning: Only {actual_samples} samples per class available. Sampling maximum possible."
            )

        # Randomly sample equal amounts
        sampled_t = random.sample(true_ids, actual_samples)
        sampled_f = random.sample(false_ids, actual_samples)

        output_data = {"t": sampled_t, "f": sampled_f}

        # Save to JSON
        with open(args.output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=4, ensure_ascii=False)

        provenance_extra["selected_counts"] = {
            "true": len(sampled_t),
            "false": len(sampled_f),
        }
        print(
            f"Successfully saved {actual_samples * 2} balanced IDs to {args.output_path}"
        )
    except BaseException as exc:
        provenance_status = provenance_status_for_exception(exc)
        provenance_extra["error"] = provenance_error_message(exc)
        raise
    finally:
        finish_run_provenance(provenance_handle, provenance_status, provenance_extra)


if __name__ == "__main__":
    main()
