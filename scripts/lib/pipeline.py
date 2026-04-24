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
from collections.abc import Sequence
from dataclasses import dataclass
import json
import os
import shutil
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from scripts.utils import format_alpha_label

ProtectedPathKind = Literal["file", "directory"]


@dataclass(frozen=True)
class ProtectedPathSpec:
    path: Path
    kind: ProtectedPathKind


@dataclass(frozen=True)
class MalformedActiveRunLock:
    path: Path
    reason: str


@dataclass(frozen=True)
class ActiveRunGitViolation:
    staged_path: str
    run_id: str
    protected_path: Path
    protected_kind: ProtectedPathKind
    lock_path: Path | None = None


@dataclass(frozen=True)
class TrackedLiveOutputViolation:
    output_target: Path
    tracked_path: str


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
    """Return the number of entries in a JSON manifest.

    Accepts either a plain JSON list of IDs or a sample-manifest object with
    an ``ids`` list.
    """
    data = json.loads(manifest.read_text())
    if isinstance(data, list):
        return len(data)
    if isinstance(data, dict) and isinstance(data.get("ids"), list):
        return len(data["ids"])
    raise ValueError(
        f"Manifest {manifest} is neither a JSON list nor an object with an ids list "
        f"(got {type(data).__name__})"
    )


def gpu_preflight() -> None:
    """Print GPU status via nvitop. Non-fatal if nvitop is unavailable."""
    if shutil.which("nvitop") is None:
        print("gpu_preflight: nvitop not found, skipping", file=sys.stderr)
        return
    print("--- GPU preflight ---", file=sys.stderr)
    subprocess.run(["nvitop", "-1"], check=False)


# Incident: 2026-04-13 — measurement-cleanup runs needed a declarative
# stop point between canary, full scoring, and follow-up judge blocks.
def check_sentinel(directory: Path, name: str) -> str | None:
    """Return sentinel reason text if a named stop file exists, else None."""
    path = directory / name
    if not path.exists():
        return None
    return path.read_text().strip()


# Incident: 2026-04-24 — an agent committed live selector artifacts while a GPU
# run was still writing them. File-per-row append discipline prevented data loss,
# but Git workflows need structural live-output ownership checks.
def active_run_registry_dir(cwd: Path | None = None) -> Path | None:
    """Return the repo-local active-run registry directory, if inside Git."""
    git_cwd = _git_command_cwd(cwd)
    repo_root = _git_repo_root(git_cwd)
    git_common = _git_common_dir(git_cwd)
    if repo_root is None or git_common is None:
        return None
    return git_common / "h-neurons-active-runs"


def current_process_start_identity(pid: int | None = None) -> str | None:
    """Return Linux /proc start ticks for a process, or None when unavailable."""
    process_id = os.getpid() if pid is None else pid
    try:
        stat_text = Path("/proc") / str(process_id) / "stat"
        content = stat_text.read_text(encoding="utf-8")
    except OSError:
        return None
    return _parse_proc_stat_start_ticks(content)


def load_active_run_locks(
    registry_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """Load parseable active-run lock records from the registry.

    Malformed records are warned about and ignored, so stale/corrupt metadata
    cannot wedge commits.
    """
    locks, malformed = _scan_active_run_registry(registry_dir)
    for bad_lock in malformed:
        print(
            f"active-run: ignoring malformed lock {bad_lock.path}: {bad_lock.reason}",
            file=sys.stderr,
        )
    return locks


def is_lock_live(lock: dict[str, Any]) -> bool:
    """Return True only when lock ownership matches a currently live process."""
    if lock.get("hostname") != socket.gethostname():
        return False
    try:
        pid = int(lock["pid"])
    except (KeyError, TypeError, ValueError):
        return False
    if pid <= 0:
        return False

    expected_identity = _lock_process_start_identity(lock)
    if expected_identity is None:
        return False
    current_identity = current_process_start_identity(pid)
    return current_identity is not None and current_identity == expected_identity


def path_intersects_live_run(
    path: str | os.PathLike[str],
    lock: dict[str, Any],
    *,
    repo_root: Path | None = None,
) -> bool:
    """Return True if a candidate path is protected by a live run lock."""
    if not is_lock_live(lock):
        return False
    return _matching_protected_path(path, lock, repo_root=repo_root) is not None


def check_staged_paths_against_live_runs(
    paths: Sequence[str | os.PathLike[str]],
    *,
    registry_dir: Path | None = None,
    repo_root: Path | None = None,
) -> list[ActiveRunGitViolation]:
    """Return staged paths that intersect output targets of live runs."""
    resolved_repo_root = repo_root or _git_repo_root()
    violations: list[ActiveRunGitViolation] = []
    for lock in load_active_run_locks(registry_dir):
        if not is_lock_live(lock):
            continue
        for path in paths:
            protected = _matching_protected_path(
                path,
                lock,
                repo_root=resolved_repo_root,
            )
            if protected is None:
                continue
            violations.append(
                ActiveRunGitViolation(
                    staged_path=os.fspath(path),
                    run_id=_lock_run_id(lock),
                    protected_path=protected.path,
                    protected_kind=protected.kind,
                    lock_path=_lock_path(lock),
                )
            )
    return violations


def check_live_output_track_state(
    output_targets: Sequence[str | os.PathLike[str]],
    *,
    repo_root: Path | None = None,
) -> list[TrackedLiveOutputViolation]:
    """Return tracked files below proposed live output targets.

    Long runs should not write live outputs into paths Git already tracks:
    checkout/restore/reset can rewrite those files before pre-commit runs.
    """
    resolved_repo_root = (
        Path(repo_root).expanduser().resolve()
        if repo_root is not None
        else _git_repo_root()
    )
    if resolved_repo_root is None:
        return []

    violations: list[TrackedLiveOutputViolation] = []
    seen: set[tuple[str, str]] = set()
    for target in output_targets:
        target_path = _resolve_candidate_path(target, resolved_repo_root)
        rel_target = _relative_to_repo(target_path, resolved_repo_root)
        if rel_target is None:
            continue
        target_kind = _infer_protected_path_kind(target_path)
        tracked_paths = _git_ls_files(resolved_repo_root, rel_target)
        if target_kind == "file":
            tracked_paths = [path for path in tracked_paths if path == rel_target]
        for tracked_path in tracked_paths:
            key = (str(target_path), tracked_path)
            if key in seen:
                continue
            seen.add(key)
            violations.append(
                TrackedLiveOutputViolation(
                    output_target=target_path,
                    tracked_path=tracked_path,
                )
            )
    return violations


def format_tracked_live_output_error(
    violations: Sequence[TrackedLiveOutputViolation],
) -> str:
    """Return the operator-facing error for tracked live output targets."""
    grouped: dict[str, list[str]] = {}
    for violation in violations:
        grouped.setdefault(str(violation.output_target), []).append(
            violation.tracked_path
        )
    lines = [
        "Live output track-state check failed.",
        "",
        "These output targets contain files already tracked by Git:",
    ]
    for target, tracked_paths in grouped.items():
        lines.append(f"  {target}")
        for tracked_path in sorted(set(tracked_paths)):
            lines.append(f"    {tracked_path}")
    lines.extend(
        [
            "",
            "Long runs must write to untracked active output paths so Git "
            "checkout/restore/reset cannot rewrite files mid-run.",
            "Archive the old run or choose a new semantic output directory, "
            "unless this short script intentionally rewrites a tracked summary.",
        ]
    )
    return "\n".join(lines)


def build_active_run_lock(
    *,
    run_id: str,
    provenance_path: str | os.PathLike[str],
    output_targets: Sequence[str | os.PathLike[str]],
    started_at_utc: str,
    script: str,
    argv: Sequence[str],
    cwd: str | os.PathLike[str],
    directory_targets: Sequence[str | os.PathLike[str]] = (),
    pid: int | None = None,
    hostname: str | None = None,
) -> dict[str, Any]:
    """Build an active-run lock record for a provenance-tracked run."""
    process_id = os.getpid() if pid is None else pid
    protected_paths = _build_protected_path_payload(
        output_targets=output_targets,
        directory_targets=directory_targets,
        extra_file_targets=[provenance_path],
    )
    return {
        "schema_version": "active_run_lock/v1",
        "run_id": run_id,
        "hostname": hostname or socket.gethostname(),
        "pid": process_id,
        "process_start_identity": {
            "source": f"/proc/{process_id}/stat",
            "start_ticks": current_process_start_identity(process_id),
        },
        "cwd": str(Path(cwd).expanduser().resolve()),
        "script": script,
        "argv": list(argv),
        "started_at_utc": started_at_utc,
        "provenance_path": str(Path(provenance_path).expanduser().resolve()),
        "output_targets": [
            str(Path(target).expanduser().resolve()) for target in output_targets
        ],
        "protected_paths": protected_paths,
    }


def write_active_run_lock(
    lock: dict[str, Any],
    *,
    registry_dir: Path | None = None,
) -> Path | None:
    """Write an active-run lock atomically and return its path."""
    resolved_registry_dir = registry_dir or active_run_registry_dir()
    if resolved_registry_dir is None:
        return None
    resolved_registry_dir.mkdir(parents=True, exist_ok=True)
    run_id = _safe_lock_filename(_lock_run_id(lock))
    lock_path = resolved_registry_dir / f"{run_id}.json"
    _write_json_atomic(lock_path, lock)
    return lock_path


def create_active_run_lock(
    *,
    run_id: str,
    provenance_path: str | os.PathLike[str],
    output_targets: Sequence[str | os.PathLike[str]],
    started_at_utc: str,
    script: str,
    argv: Sequence[str],
    cwd: str | os.PathLike[str],
    directory_targets: Sequence[str | os.PathLike[str]] = (),
) -> Path | None:
    """Build and write an active-run lock for a provenance-tracked run."""
    lock = build_active_run_lock(
        run_id=run_id,
        provenance_path=provenance_path,
        output_targets=output_targets,
        started_at_utc=started_at_utc,
        script=script,
        argv=argv,
        cwd=cwd,
        directory_targets=directory_targets,
    )
    return write_active_run_lock(lock)


def remove_active_run_lock(lock_path: str | os.PathLike[str] | None) -> None:
    """Remove an active-run lock. Missing locks are already stale, so ignore."""
    if lock_path is None:
        return
    try:
        Path(lock_path).unlink(missing_ok=True)
    except OSError as exc:
        print(
            f"Warning: failed to remove active-run lock {lock_path}: {exc}",
            file=sys.stderr,
        )


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


def _git_stdout(args: Sequence[str], cwd: Path | None = None) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=cwd,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (OSError, ValueError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def _git_repo_root(cwd: Path | None = None) -> Path | None:
    output = _git_stdout(["rev-parse", "--show-toplevel"], cwd=cwd)
    return Path(output).resolve() if output else None


def _git_command_cwd(cwd: Path | None = None) -> Path:
    return Path.cwd() if cwd is None else Path(cwd).expanduser().resolve()


def _git_common_dir(cwd: Path | None = None) -> Path | None:
    output = _git_stdout(["rev-parse", "--git-common-dir"], cwd=cwd)
    if not output:
        return None
    git_common = Path(output).expanduser()
    if git_common.is_absolute():
        return git_common.resolve()
    return (_git_command_cwd(cwd) / git_common).resolve()


def _git_ls_files(repo_root: Path, rel_path: str) -> list[str]:
    try:
        result = subprocess.run(
            ["git", "ls-files", "-z", "--", rel_path],
            cwd=repo_root,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (OSError, ValueError):
        return []
    if result.returncode != 0:
        return []
    return [path for path in result.stdout.split("\0") if path]


def _parse_proc_stat_start_ticks(content: str) -> str | None:
    command_end = content.rfind(")")
    if command_end == -1:
        return None
    fields_after_command = content[command_end + 2 :].split()
    # Field 3 is first after "(comm)"; field 22 (starttime) is index 19.
    if len(fields_after_command) <= 19:
        return None
    return fields_after_command[19]


def _scan_active_run_registry(
    registry_dir: Path | None = None,
) -> tuple[list[dict[str, Any]], list[MalformedActiveRunLock]]:
    resolved_registry_dir = registry_dir or active_run_registry_dir()
    if resolved_registry_dir is None or not resolved_registry_dir.exists():
        return [], []

    locks: list[dict[str, Any]] = []
    malformed: list[MalformedActiveRunLock] = []
    for lock_path in sorted(resolved_registry_dir.glob("*.json")):
        try:
            payload = json.loads(lock_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            malformed.append(MalformedActiveRunLock(lock_path, str(exc)))
            continue
        if not isinstance(payload, dict):
            malformed.append(
                MalformedActiveRunLock(lock_path, "lock root is not a JSON object")
            )
            continue
        payload["_lock_path"] = str(lock_path)
        locks.append(payload)
    return locks, malformed


def _lock_process_start_identity(lock: dict[str, Any]) -> str | None:
    identity = lock.get("process_start_identity")
    if isinstance(identity, dict):
        value = (
            identity.get("start_ticks")
            or identity.get("proc_start_ticks")
            or identity.get("stat_start_ticks")
        )
    else:
        value = identity
    if value is None:
        return None
    return str(value)


def _lock_run_id(lock: dict[str, Any]) -> str:
    raw_run_id = lock.get("run_id") or lock.get("provenance_path") or "unknown-run"
    return str(raw_run_id)


def _lock_path(lock: dict[str, Any]) -> Path | None:
    raw_path = lock.get("_lock_path")
    if raw_path is None:
        return None
    return Path(str(raw_path))


def _resolve_candidate_path(
    path: str | os.PathLike[str],
    repo_root: Path | None = None,
) -> Path:
    raw_path = Path(path).expanduser()
    if raw_path.is_absolute():
        return raw_path.resolve()
    base = (
        Path(repo_root).expanduser().resolve() if repo_root is not None else Path.cwd()
    )
    return (base / raw_path).resolve()


def _relative_to_repo(path: Path, repo_root: Path) -> str | None:
    try:
        rel_path = path.resolve().relative_to(repo_root.resolve())
    except ValueError:
        return None
    rel_text = rel_path.as_posix()
    return rel_text or "."


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _infer_protected_path_kind(path: Path) -> ProtectedPathKind:
    if path.exists():
        return "directory" if path.is_dir() else "file"
    return "file" if path.suffix else "directory"


def _parse_protected_path_specs(lock: dict[str, Any]) -> list[ProtectedPathSpec]:
    raw_specs = lock.get("protected_paths")
    if not isinstance(raw_specs, list):
        return []

    specs: list[ProtectedPathSpec] = []
    for raw_spec in raw_specs:
        raw_path: Any
        raw_kind: Any
        if isinstance(raw_spec, str):
            raw_path = raw_spec
            raw_kind = None
        elif isinstance(raw_spec, dict):
            raw_path = raw_spec.get("path")
            raw_kind = raw_spec.get("kind")
        else:
            continue
        if not isinstance(raw_path, str) or not raw_path:
            continue
        path = Path(raw_path).expanduser().resolve()
        kind: ProtectedPathKind
        if raw_kind in ("file", "directory"):
            kind = raw_kind
        else:
            kind = _infer_protected_path_kind(path)
        specs.append(ProtectedPathSpec(path=path, kind=kind))
    return specs


def _matching_protected_path(
    path: str | os.PathLike[str],
    lock: dict[str, Any],
    *,
    repo_root: Path | None,
) -> ProtectedPathSpec | None:
    candidate = _resolve_candidate_path(path, repo_root)
    for protected in _parse_protected_path_specs(lock):
        if protected.kind == "file" and candidate == protected.path:
            return protected
        if protected.kind == "directory" and (
            candidate == protected.path or _is_relative_to(candidate, protected.path)
        ):
            return protected
    return None


def _build_protected_path_payload(
    *,
    output_targets: Sequence[str | os.PathLike[str]],
    directory_targets: Sequence[str | os.PathLike[str]],
    extra_file_targets: Sequence[str | os.PathLike[str]],
) -> list[dict[str, str]]:
    directory_paths = {
        str(Path(target).expanduser().resolve()) for target in directory_targets
    }
    protected_by_path: dict[str, ProtectedPathKind] = {}

    def add_path(path_like: str | os.PathLike[str], kind: ProtectedPathKind) -> None:
        path_text = str(Path(path_like).expanduser().resolve())
        existing = protected_by_path.get(path_text)
        if existing == "directory":
            return
        protected_by_path[path_text] = kind

    for target in directory_targets:
        add_path(target, "directory")
    for target in output_targets:
        target_path = Path(target).expanduser().resolve()
        target_kind: ProtectedPathKind
        if str(target_path) in directory_paths:
            target_kind = "directory"
        else:
            target_kind = _infer_protected_path_kind(target_path)
        add_path(target_path, target_kind)
    for target in extra_file_targets:
        add_path(target, "file")

    return [
        {"path": path, "kind": kind} for path, kind in sorted(protected_by_path.items())
    ]


def _safe_lock_filename(run_id: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in run_id)
    return safe[:180] or "unknown-run"


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
        _fsync_dir(path.parent)
    except Exception:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def _fsync_dir(path: Path) -> None:
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _staged_paths_from_git(repo_root: Path) -> list[str]:
    try:
        result = subprocess.run(
            ["git", "diff", "--cached", "--name-status", "-z"],
            cwd=repo_root,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"failed to inspect staged paths: {exc}") from exc
    if result.returncode != 0:
        detail = result.stderr.strip() or "git diff --cached failed"
        raise RuntimeError(detail)
    fields = [field for field in result.stdout.split("\0") if field]
    paths: list[str] = []
    index = 0
    while index < len(fields):
        status = fields[index]
        index += 1
        if not status:
            continue
        code = status[0]
        if code in {"R", "C"}:
            if index < len(fields):
                paths.append(fields[index])
                index += 1
            if index < len(fields):
                paths.append(fields[index])
                index += 1
            continue
        if index < len(fields):
            paths.append(fields[index])
            index += 1
    return paths


def _format_active_run_git_guard_error(
    violations: Sequence[ActiveRunGitViolation],
) -> str:
    grouped: dict[str, list[ActiveRunGitViolation]] = {}
    for violation in violations:
        grouped.setdefault(violation.run_id, []).append(violation)

    lines = ["Active run Git guard blocked this commit.", ""]
    for run_id, run_violations in sorted(grouped.items()):
        lines.append(f"The following staged paths are owned by live run {run_id}:")
        for violation in sorted(run_violations, key=lambda item: item.staged_path):
            lines.append(f"  {violation.staged_path}")
        lock_paths = {
            str(violation.lock_path)
            for violation in run_violations
            if violation.lock_path is not None
        }
        for lock_path in sorted(lock_paths):
            lines.append(f"  lock: {lock_path}")
        lines.append("")
    lines.extend(
        [
            "Run:",
            "  uv run python -m scripts.lib.pipeline active-run-status",
            "",
            "Wait for the run to finish, or explicitly stop/finalize it before "
            "committing these paths.",
        ]
    )
    return "\n".join(lines)


def _print_lock_summary(label: str, locks: Sequence[dict[str, Any]]) -> None:
    print(f"{label}: {len(locks)}")
    for lock in locks:
        protected_count = len(_parse_protected_path_specs(lock))
        print(
            f"  {_lock_run_id(lock)} "
            f"pid={lock.get('pid')} host={lock.get('hostname')} "
            f"protected_paths={protected_count}"
        )
        lock_path = _lock_path(lock)
        if lock_path is not None:
            print(f"    lock: {lock_path}")
        provenance_path = lock.get("provenance_path")
        if provenance_path:
            print(f"    provenance: {provenance_path}")


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


def _cli_check_sentinel(args: argparse.Namespace) -> int:
    reason = check_sentinel(Path(args.dir), args.name)
    if reason is None:
        return 1
    if reason:
        print(reason)
    return 0


def _cli_active_run_status(_args: argparse.Namespace) -> int:
    registry_dir = active_run_registry_dir()
    if registry_dir is None:
        print("Active run registry: unavailable (not inside a Git worktree)")
        return 0

    locks, malformed = _scan_active_run_registry(registry_dir)
    live = [lock for lock in locks if is_lock_live(lock)]
    stale = [lock for lock in locks if not is_lock_live(lock)]

    print(f"Active run registry: {registry_dir}")
    _print_lock_summary("Live locks", live)
    _print_lock_summary("Stale or remote locks", stale)
    print(f"Malformed locks: {len(malformed)}")
    for bad_lock in malformed:
        print(f"  {bad_lock.path}: {bad_lock.reason}")
    return 0


def _cli_check_active_run_git_guard(_args: argparse.Namespace) -> int:
    repo_root = _git_repo_root()
    if repo_root is None:
        return 0
    try:
        staged_paths = _staged_paths_from_git(repo_root)
    except RuntimeError as exc:
        print(f"active-run git guard: {exc}", file=sys.stderr)
        return 2
    violations = check_staged_paths_against_live_runs(
        staged_paths,
        repo_root=repo_root,
    )
    if not violations:
        return 0
    print(_format_active_run_git_guard_error(violations), file=sys.stderr)
    return 1


def _cli_check_live_output_track_state(args: argparse.Namespace) -> int:
    violations = check_live_output_track_state(args.output_target)
    if not violations:
        return 0
    print(format_tracked_live_output_error(violations), file=sys.stderr)
    return 1


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

    sentinel = sub.add_parser(
        "check-sentinel",
        help="Exit 0 if a named sentinel file exists, 1 otherwise",
    )
    sentinel.add_argument("--dir", required=True, help="Directory to search in")
    sentinel.add_argument(
        "--name", required=True, help="Sentinel filename to check for"
    )
    sentinel.set_defaults(func=_cli_check_sentinel)

    status = sub.add_parser(
        "active-run-status",
        help="Print live, stale, remote, and malformed active-run locks",
    )
    status.set_defaults(func=_cli_active_run_status)

    git_guard = sub.add_parser(
        "check-active-run-git-guard",
        help="Block commits whose staged paths intersect live run outputs",
    )
    git_guard.set_defaults(func=_cli_check_active_run_git_guard)

    track_state = sub.add_parser(
        "check-live-output-track-state",
        help="Exit non-zero when live output targets contain tracked files",
    )
    track_state.add_argument(
        "--output-target",
        action="append",
        required=True,
        help="File or directory a run plans to write while active",
    )
    track_state.set_defaults(func=_cli_check_live_output_track_state)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
