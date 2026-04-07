"""Pipeline guard library — incident-driven guards for GPU pipeline orchestration.

Each function here exists because a specific incident burned GPU hours.
Do not add speculative utilities. Every function should trace to an incident
or a pattern duplicated across 5+ pipeline scripts.

CLI usage (from bash pipeline scripts):
    uv run python -m scripts.lib.pipeline check-stage ...
    uv run python -m scripts.lib.pipeline manifest-count ...
    uv run python -m scripts.lib.pipeline log-run ...
    uv run python -m scripts.lib.pipeline gpu-preflight

See scripts/AGENTS.md for integration guidance.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from scripts.utils import format_alpha_label


# Incident: 2026-04-07 — bash guard only checked file existence, not line count.
# A power-off left alpha_8.0.jsonl at 68/100 lines; guard skipped re-generation.
def check_stage_complete(
    output_dir: Path,
    manifest: Path,
    alphas: list[float],
) -> bool:
    """Return True only if every alpha file has >= expected records.

    Expected count is derived from the manifest (a JSON list of sample IDs).
    Prints diagnostics for incomplete files to stderr.
    """
    expected = manifest_count(manifest)
    all_complete = True
    for alpha in alphas:
        f = output_dir / f"alpha_{format_alpha_label(alpha)}.jsonl"
        if not f.exists():
            print(f"  missing: {f}", file=sys.stderr)
            all_complete = False
            continue
        n = _count_lines(f)
        if n < expected:
            print(f"  incomplete: {f} ({n}/{expected} lines)", file=sys.stderr)
            all_complete = False
    return all_complete


def manifest_count(manifest: Path) -> int:
    """Return the number of entries in a JSON manifest (a list of IDs)."""
    data = json.loads(manifest.read_text())
    if not isinstance(data, list):
        raise ValueError(
            f"Manifest {manifest} is not a JSON list (got {type(data).__name__})"
        )
    return len(data)


def gpu_preflight() -> None:
    """Print GPU status via nvitop. Non-fatal if nvitop is unavailable."""
    if shutil.which("nvitop") is None:
        print("gpu_preflight: nvitop not found, skipping", file=sys.stderr)
        return
    print("--- GPU preflight ---", file=sys.stderr)
    subprocess.run(["nvitop", "-1"], check=False)


# Incident: forgotten runs_to_analyse entries cause analysed data to be
# silently skipped or re-analysed. Standardise the format.
def log_run(
    run_dir: str,
    description: str,
    key_files: str = "results.json, *.provenance.json",
    notes_file: Path | None = None,
) -> None:
    """Append a run entry to notes/runs_to_analyse.md."""
    if notes_file is None:
        notes_file = Path("notes/runs_to_analyse.md")
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    entry = (
        f"\n## {ts} | {run_dir}\n"
        f"What: {description}\n"
        f"Key files: {key_files}\n"
        f"Status: awaiting analysis\n"
    )
    notes_file.parent.mkdir(parents=True, exist_ok=True)
    with open(notes_file, "a") as f:
        f.write(entry)
    print(f"log_run: appended {run_dir} to {notes_file}", file=sys.stderr)


def _count_lines(path: Path) -> int:
    """Count lines in a file (matches wc -l semantics)."""
    with open(path, "rb") as f:
        return sum(1 for _ in f)


# --- CLI — thin dispatch for bash integration ---


def _cli_check_stage(args: argparse.Namespace) -> int:
    ok = check_stage_complete(
        output_dir=Path(args.output_dir),
        manifest=Path(args.manifest),
        alphas=args.alphas,
    )
    # Exit 0 = complete (bash truthy), exit 1 = incomplete
    return 0 if ok else 1


def _cli_manifest_count(args: argparse.Namespace) -> int:
    for path in args.manifests:
        p = Path(path)
        print(f"{manifest_count(p)}\t{p}")
    return 0


def _cli_log_run(args: argparse.Namespace) -> int:
    log_run(
        run_dir=args.run_dir,
        description=args.description,
        key_files=args.key_files,
    )
    return 0


def _cli_gpu_preflight(_args: argparse.Namespace) -> int:
    gpu_preflight()
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="pipeline",
        description="Pipeline guard library CLI",
    )
    sub = p.add_subparsers(dest="command")
    sub.required = True

    cs = sub.add_parser(
        "check-stage",
        help="Exit 0 if all alpha files are complete, 1 otherwise",
    )
    cs.add_argument(
        "--output-dir", required=True, help="Directory containing alpha_*.jsonl files"
    )
    cs.add_argument(
        "--manifest", required=True, help="JSON manifest (list of sample IDs)"
    )
    cs.add_argument("--alphas", type=float, nargs="+", required=True)
    cs.set_defaults(func=_cli_check_stage)

    mc = sub.add_parser(
        "manifest-count",
        help="Print the number of entries in one or more JSON manifests",
    )
    mc.add_argument("manifests", nargs="+", help="Paths to JSON manifest files")
    mc.set_defaults(func=_cli_manifest_count)

    lr = sub.add_parser(
        "log-run", help="Append a run entry to notes/runs_to_analyse.md"
    )
    lr.add_argument(
        "--run-dir", required=True, help="Relative path to the run directory"
    )
    lr.add_argument(
        "--description", required=True, help="One-line: benchmark + method + alpha grid"
    )
    lr.add_argument(
        "--key-files",
        default="results.json, *.provenance.json",
        help="Comma-separated list of key files",
    )
    lr.set_defaults(func=_cli_log_run)

    gp = sub.add_parser("gpu-preflight", help="Print GPU status via nvitop")
    gp.set_defaults(func=_cli_gpu_preflight)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
