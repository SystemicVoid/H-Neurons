#!/usr/bin/env python3
"""Dry-run-first cloud control helpers for H-Neurons runs."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import tomllib
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
PROFILE_DIR = Path(__file__).resolve().parent / "cloud" / "profiles"
SPEND_EPSILON = 1e-9


class CloudctlError(RuntimeError):
    """Operator-facing cloudctl error."""


@dataclass(frozen=True)
class SshEndpoint:
    host: str
    port: int


@dataclass(frozen=True)
class StorageDecision:
    mode: str
    reason: str
    network_volume_size_gb: int | None = None


@dataclass(frozen=True)
class RunPodSnapshot:
    pods: list[dict[str, Any]]
    network_volumes: list[dict[str, Any]]
    user: dict[str, Any]


def _run(
    args: list[str],
    *,
    cwd: Path = REPO_ROOT,
    timeout: int | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=cwd,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout,
    )


def _require_success(result: subprocess.CompletedProcess[str], name: str) -> None:
    if result.returncode == 0:
        return
    detail = result.stderr.strip() or result.stdout.strip()
    raise CloudctlError(f"{name} failed with exit {result.returncode}: {detail}")


def _load_json_output(result: subprocess.CompletedProcess[str], name: str) -> Any:
    _require_success(result, name)
    text = result.stdout.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise CloudctlError(f"{name} did not return JSON: {exc}") from exc


def runpodctl_json(args: list[str]) -> Any:
    if shutil.which("runpodctl") is None:
        raise CloudctlError("runpodctl is not on PATH")
    return _load_json_output(_run(["runpodctl", *args, "-o", "json"]), "runpodctl")


def _profile_path(name: str) -> Path:
    candidate = Path(name)
    if candidate.suffix == ".toml" and candidate.exists():
        return candidate
    return PROFILE_DIR / f"{name}.toml"


def load_profile(name: str) -> dict[str, Any]:
    path = _profile_path(name)
    if not path.is_file():
        raise CloudctlError(f"profile not found: {path}")
    with path.open("rb") as handle:
        profile = tomllib.load(handle)
    if not isinstance(profile.get("name"), str):
        profile["name"] = path.stem
    if not isinstance(profile.get("provider"), str):
        raise CloudctlError(f"profile {path} must set provider")
    return profile


def _table(profile: dict[str, Any], key: str) -> dict[str, Any]:
    value = profile.get(key)
    if not isinstance(value, dict):
        raise CloudctlError(f"profile {profile.get('name')} must contain [{key}]")
    return value


def _optional_table(profile: dict[str, Any], key: str) -> dict[str, Any]:
    value = profile.get(key, {})
    if not isinstance(value, dict):
        raise CloudctlError(f"profile {profile.get('name')} [{key}] must be a table")
    return value


def _string(table: dict[str, Any], key: str, default: str | None = None) -> str:
    value = table.get(key, default)
    if not isinstance(value, str) or not value:
        raise CloudctlError(f"profile value {key} must be a non-empty string")
    return value


def _int(table: dict[str, Any], key: str, default: int | None = None) -> int:
    value = table.get(key, default)
    if not isinstance(value, int):
        raise CloudctlError(f"profile value {key} must be an integer")
    return value


def _bool(table: dict[str, Any], key: str, default: bool = False) -> bool:
    value = table.get(key, default)
    if not isinstance(value, bool):
        raise CloudctlError(f"profile value {key} must be a boolean")
    return value


def _strings(table: dict[str, Any], key: str) -> list[str]:
    value = table.get(key, [])
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise CloudctlError(f"profile value {key} must be a list of strings")
    return value


def _single_data_center_id(value: str, *, field: str = "data_center_id") -> str:
    data_center_id = value.strip()
    if (
        not data_center_id
        or "," in data_center_id
        or any(char.isspace() for char in data_center_id)
    ):
        raise CloudctlError(f"{field} must be a single datacenter id")
    if not re.fullmatch(r"[A-Za-z0-9-]+", data_center_id):
        raise CloudctlError(f"{field} contains unsupported characters")
    return data_center_id


def _parse_cp_number(stage: str) -> int | None:
    match = re.fullmatch(r"cp(\d+)", stage.strip().lower())
    if not match:
        return None
    return int(match.group(1))


def decide_storage(
    profile: dict[str, Any],
    *,
    stage: str,
    attempt: int,
) -> StorageDecision:
    storage = _optional_table(profile, "storage")
    cp_number = _parse_cp_number(stage)
    network_volume_size = storage.get("network_volume_size_gb")
    if network_volume_size is not None and not isinstance(network_volume_size, int):
        raise CloudctlError("storage.network_volume_size_gb must be an integer")

    if cp_number is not None and cp_number >= 2:
        return StorageDecision(
            mode="network_volume",
            reason="CP2+ runs need cached weights, uv cache, repo state, and artifacts",
            network_volume_size_gb=network_volume_size,
        )
    if stage.lower() == "cp1" and attempt > 1:
        return StorageDecision(
            mode="network_volume",
            reason="CP1 retry after paid setup friction should preserve caches",
            network_volume_size_gb=network_volume_size,
        )
    return StorageDecision(
        mode=_string(storage, "first_cp1_mode", "ephemeral"),
        reason="first bounded CP1 smoke should avoid idle storage billing",
        network_volume_size_gb=network_volume_size,
    )


def render_launch_command(
    profile: dict[str, Any],
    *,
    stage: str,
    attempt: int,
    pod_name: str | None = None,
    network_volume_id: str | None = None,
    data_center_id: str | None = None,
) -> list[str]:
    if profile["provider"] != "runpod":
        raise CloudctlError("render-launch currently supports RunPod profiles only")

    runpod = _table(profile, "runpod")
    storage = decide_storage(profile, stage=stage, attempt=attempt)
    resolved_volume_id = network_volume_id or runpod.get("network_volume_id")
    if storage.mode == "network_volume" and not isinstance(resolved_volume_id, str):
        raise CloudctlError(
            "storage decision requires a network volume; pass --network-volume-id"
        )
    if (
        storage.mode == "network_volume"
        and data_center_id is None
        and len(_strings(runpod, "data_center_ids")) != 1
    ):
        raise CloudctlError(
            "network-volume launches must pass --data-center-id for the volume's datacenter"
        )

    command = [
        "runpodctl",
        "pod",
        "create",
        "--name",
        pod_name or _string(runpod, "pod_name"),
        "--cloud-type",
        _string(runpod, "cloud_type", "SECURE"),
        "--gpu-id",
        _string(runpod, "gpu_id"),
        "--gpu-count",
        str(_int(runpod, "gpu_count", 1)),
        "--container-disk-in-gb",
        str(_int(runpod, "container_disk_gb", 100)),
        "--volume-mount-path",
        _string(runpod, "volume_mount_path", "/workspace"),
        "--ports",
        _string(runpod, "ports", "8888/http,22/tcp"),
    ]

    template_id = runpod.get("template_id")
    image = runpod.get("image")
    if isinstance(template_id, str) and template_id:
        command.extend(["--template-id", template_id])
    elif isinstance(image, str) and image:
        command.extend(["--image", image])
    else:
        raise CloudctlError("RunPod profile must set template_id or image")

    datacenters = (
        [_single_data_center_id(data_center_id)]
        if data_center_id is not None
        else _strings(runpod, "data_center_ids")
    )
    if datacenters:
        command.extend(["--data-center-ids", ",".join(datacenters)])

    if storage.mode == "network_volume":
        command.extend(["--network-volume-id", str(resolved_volume_id)])
    else:
        command.extend(["--volume-in-gb", str(_int(runpod, "volume_gb", 100))])

    if _bool(runpod, "ssh", True):
        command.append("--ssh")
    if _bool(runpod, "global_networking", False):
        command.append("--global-networking")
    if _bool(runpod, "public_ip", False):
        command.append("--public-ip")

    return command


def render_network_volume_create_command(
    profile: dict[str, Any],
    *,
    name: str | None = None,
    size_gb: int | None = None,
    data_center_id: str,
) -> list[str]:
    if profile["provider"] != "runpod":
        raise CloudctlError(
            "render-volume-create currently supports RunPod profiles only"
        )
    runpod = _table(profile, "runpod")
    storage = _optional_table(profile, "storage")
    resolved_size = size_gb
    if resolved_size is None:
        resolved_size = _int(storage, "network_volume_size_gb")
    if resolved_size <= 0:
        raise CloudctlError("network volume size must be positive")
    resolved_name = name or f"{_string(runpod, 'pod_name')}-cp2-cache"
    resolved_data_center_id = _single_data_center_id(data_center_id)
    return [
        "runpodctl",
        "network-volume",
        "create",
        "--name",
        resolved_name,
        "--size",
        str(resolved_size),
        "--data-center-id",
        resolved_data_center_id,
    ]


def shell_join(command: list[str]) -> str:
    return shlex.join(command)


def _data_list(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        data = payload.get("data")
        if isinstance(data, list):
            return [item for item in data if isinstance(item, dict)]
        if isinstance(data, dict):
            return [item for item in data.values() if isinstance(item, dict)]
    return []


def _pick(item: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = item.get(key)
        if value is not None:
            return str(value)
    return "?"


def _find_number(payload: Any, keys: set[str]) -> float | None:
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key in keys and isinstance(value, int | float):
                return float(value)
            found = _find_number(value, keys)
            if found is not None:
                return found
    if isinstance(payload, list):
        for item in payload:
            found = _find_number(item, keys)
            if found is not None:
                return found
    return None


def current_spend_per_hour(user_payload: dict[str, Any]) -> float:
    value = _find_number(
        user_payload,
        {"currentSpendPerHr", "current_spend_per_hr", "currentSpendPerHour"},
    )
    return 0.0 if value is None else value


def runpod_snapshot() -> RunPodSnapshot:
    return RunPodSnapshot(
        pods=_data_list(runpodctl_json(["pod", "list", "--all"])),
        network_volumes=_data_list(runpodctl_json(["network-volume", "list"])),
        user=runpodctl_json(["user"]),
    )


def print_snapshot(snapshot: RunPodSnapshot) -> None:
    spend = current_spend_per_hour(snapshot.user)
    print(f"pods: {len(snapshot.pods)}")
    for pod in snapshot.pods:
        print(
            "  "
            + " ".join(
                [
                    _pick(pod, "id", "podId"),
                    _pick(pod, "name"),
                    _pick(pod, "desiredStatus", "status"),
                    _pick(pod, "machineId", "gpuTypeId"),
                ]
            )
        )
    print(f"network_volumes: {len(snapshot.network_volumes)}")
    for volume in snapshot.network_volumes:
        print(
            "  "
            + " ".join(
                [
                    _pick(volume, "id", "networkVolumeId"),
                    _pick(volume, "name"),
                    _pick(volume, "dataCenterId", "data_center_id"),
                    _pick(volume, "size", "sizeInGb"),
                ]
            )
        )
    print(f"currentSpendPerHr: {spend:g}")


def _iter_dicts(payload: Any) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    if isinstance(payload, dict):
        found.append(payload)
        for value in payload.values():
            found.extend(_iter_dicts(value))
    elif isinstance(payload, list):
        for item in payload:
            found.extend(_iter_dicts(item))
    return found


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


def derive_ssh_endpoint(metadata: Any) -> SshEndpoint:
    if isinstance(metadata, dict):
        host = metadata.get("ip") or metadata.get("host") or metadata.get("publicIp")
        public_port = _as_int(metadata.get("port", metadata.get("publicPort")))
        if isinstance(host, str) and public_port is not None:
            return SshEndpoint(host=host, port=public_port)

    for item in _iter_dicts(metadata):
        private_port = _as_int(
            item.get("privatePort", item.get("private_port", item.get("containerPort")))
        )
        public_port = _as_int(item.get("publicPort", item.get("public_port")))
        host = item.get("ip") or item.get("host") or item.get("publicIp")
        is_public = item.get("isIpPublic", item.get("is_public", True))
        if private_port == 22 and public_port is not None and isinstance(host, str):
            if is_public is not False:
                return SshEndpoint(host=host, port=public_port)
    raise CloudctlError("could not derive public SSH host/port from pod metadata")


def _print_result(name: str, result: subprocess.CompletedProcess[str]) -> bool:
    ok = result.returncode == 0
    marker = "ok" if ok else "fail"
    print(f"[{marker}] {name}")
    if not ok:
        detail = result.stderr.strip() or result.stdout.strip()
        if detail:
            print(detail)
    return ok


def _git_dirty(status_output: str) -> bool:
    lines = [line for line in status_output.splitlines()[1:] if line.strip()]
    return bool(lines)


def _bundle_smoke(profile: dict[str, Any]) -> subprocess.CompletedProcess[str]:
    git_table = _optional_table(profile, "git")
    branch = _string(git_table, "bundle_branch", "main")
    with tempfile.TemporaryDirectory(prefix="hneurons-bundle-smoke-") as tmp:
        tmpdir = Path(tmp)
        bundle_path = tmpdir / "h-neurons.bundle"
        clone_path = tmpdir / "clone"
        create = _run(["git", "bundle", "create", str(bundle_path), branch])
        if create.returncode != 0:
            return create
        clone = _run(["git", "clone", "-b", branch, str(bundle_path), str(clone_path)])
        if clone.returncode != 0:
            return clone
        return _run(["git", "-C", str(clone_path), "rev-parse", "HEAD"])


def run_preflight(args: argparse.Namespace) -> int:
    profile = load_profile(args.profile)
    storage = decide_storage(profile, stage=args.stage, attempt=args.attempt)
    print(f"profile: {profile['name']} ({profile['provider']})")
    print(f"storage: {storage.mode} - {storage.reason}")
    if storage.network_volume_size_gb is not None:
        print(f"network_volume_size_gb: {storage.network_volume_size_gb}")

    failed = False

    status = _run(["git", "status", "--short", "--branch"])
    failed |= not _print_result("git status", status)
    if status.returncode == 0:
        print(status.stdout.rstrip())
        if _git_dirty(status.stdout) and not args.allow_dirty:
            print("[fail] working tree dirty; commit/stash before paid launch")
            failed = True

    checks = [
        (
            "active-run status",
            ["uv", "run", "python", "-m", "scripts.lib.pipeline", "active-run-status"],
        ),
        (
            "active-run git guard",
            [
                "uv",
                "run",
                "python",
                "-m",
                "scripts.lib.pipeline",
                "check-active-run-git-guard",
            ],
        ),
    ]
    if not args.skip_dependency_dry_run:
        checks.append(("dependency dry-run", ["uv", "sync", "--no-dev", "--dry-run"]))

    for name, command in checks:
        result = _run(command)
        failed |= not _print_result(name, result)

    if not args.skip_bundle_smoke:
        failed |= not _print_result("bundle smoke", _bundle_smoke(profile))

    return 1 if failed else 0


def run_status(_args: argparse.Namespace) -> int:
    print_snapshot(runpod_snapshot())
    return 0


def run_cleanup_check(args: argparse.Namespace) -> int:
    snapshot = runpod_snapshot()
    print_snapshot(snapshot)
    failed = False
    if snapshot.pods:
        print("[fail] pods still exist")
        failed = True
    if snapshot.network_volumes and not args.allow_network_volumes:
        print("[fail] network volumes still exist")
        failed = True
    spend = current_spend_per_hour(snapshot.user)
    if spend > SPEND_EPSILON:
        print("[fail] currentSpendPerHr is nonzero")
        failed = True
    if not failed:
        print("[ok] cleanup check passed")
    return 1 if failed else 0


def run_ssh_info(args: argparse.Namespace) -> int:
    if args.metadata_json:
        metadata = json.loads(Path(args.metadata_json).read_text(encoding="utf-8"))
        endpoint = derive_ssh_endpoint(metadata)
    else:
        metadata = runpodctl_json(
            [
                "pod",
                "get",
                args.pod_id,
                "--include-machine",
                "--include-network-volume",
            ]
        )
        try:
            endpoint = derive_ssh_endpoint(metadata)
        except CloudctlError as pod_get_exc:
            ssh_info = runpodctl_json(["ssh", "info", args.pod_id])
            try:
                endpoint = derive_ssh_endpoint(ssh_info)
            except CloudctlError as ssh_info_exc:
                raise CloudctlError(
                    "could not derive public SSH host/port from pod metadata "
                    f"or runpodctl ssh info: {pod_get_exc}; {ssh_info_exc}"
                ) from ssh_info_exc
    print(
        "ssh -o StrictHostKeyChecking=accept-new "
        f"-p {endpoint.port} root@{endpoint.host}"
    )
    return 0


def run_render_launch(args: argparse.Namespace) -> int:
    profile = load_profile(args.profile)
    command = render_launch_command(
        profile,
        stage=args.stage,
        attempt=args.attempt,
        pod_name=args.name,
        network_volume_id=args.network_volume_id,
        data_center_id=args.data_center_id,
    )
    print(shell_join(command))
    return 0


def run_render_volume_create(args: argparse.Namespace) -> int:
    profile = load_profile(args.profile)
    command = render_network_volume_create_command(
        profile,
        name=args.name,
        size_gb=args.size_gb,
        data_center_id=args.data_center_id,
    )
    print(shell_join(command))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cloudctl",
        description="H-Neurons cloud run adapter; status and dry-run by default.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    status = sub.add_parser("status", help="Show RunPod pods, volumes, and spend")
    status.set_defaults(func=run_status)

    preflight = sub.add_parser("preflight", help="Run local checks before paid launch")
    preflight.add_argument("--profile", required=True)
    preflight.add_argument("--stage", default="cp1")
    preflight.add_argument("--attempt", type=int, default=1)
    preflight.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Do not fail solely because the worktree has uncommitted changes.",
    )
    preflight.add_argument(
        "--skip-dependency-dry-run",
        action="store_true",
        help="Skip uv sync --no-dev --dry-run.",
    )
    preflight.add_argument(
        "--skip-bundle-smoke",
        action="store_true",
        help="Skip git bundle clone smoke test.",
    )
    preflight.set_defaults(func=run_preflight)

    ssh_info = sub.add_parser("ssh-info", help="Print direct root@ip SSH command")
    ssh_info.add_argument("--pod-id", required=True)
    ssh_info.add_argument(
        "--metadata-json",
        help="Read pod metadata from a JSON file instead of runpodctl.",
    )
    ssh_info.set_defaults(func=run_ssh_info)

    render = sub.add_parser(
        "render-launch",
        help="Print the paid RunPod launch command without executing it.",
    )
    render.add_argument("--profile", required=True)
    render.add_argument("--stage", default="cp1")
    render.add_argument("--attempt", type=int, default=1)
    render.add_argument("--name", help="Override pod name")
    render.add_argument(
        "--network-volume-id",
        help="Network volume id required for retry/CP2+ storage decisions.",
    )
    render.add_argument(
        "--data-center-id",
        help="Datacenter id for exact network-volume placement.",
    )
    render.set_defaults(func=run_render_launch)

    render_volume = sub.add_parser(
        "render-volume-create",
        help="Print the paid RunPod network-volume create command without executing it.",
    )
    render_volume.add_argument("--profile", required=True)
    render_volume.add_argument(
        "--name",
        help="Override network volume name; defaults to <pod_name>-cp2-cache.",
    )
    render_volume.add_argument(
        "--size-gb",
        type=int,
        default=None,
        help="Override network volume size in GB; defaults to profile storage.",
    )
    render_volume.add_argument(
        "--data-center-id",
        required=True,
        help="Datacenter id selected after live H100 stock check.",
    )
    render_volume.set_defaults(func=run_render_volume_create)

    cleanup = sub.add_parser(
        "cleanup-check",
        help="Verify no pods, no unexpected volumes, and zero hourly spend.",
    )
    cleanup.add_argument(
        "--allow-network-volumes",
        action="store_true",
        help="Allow intentional persistent network volumes to remain.",
    )
    cleanup.set_defaults(func=run_cleanup_check)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        func = args.func
        return int(func(args))
    except CloudctlError as exc:
        print(f"cloudctl: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
