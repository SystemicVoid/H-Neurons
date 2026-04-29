from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.infra import cloudctl


def test_load_mistral_runpod_profile() -> None:
    profile = cloudctl.load_profile("mistral24b-runpod")

    assert profile["provider"] == "runpod"
    assert profile["name"] == "mistral24b-runpod"


def test_storage_decision_switches_after_first_cp1_attempt() -> None:
    profile = cloudctl.load_profile("mistral24b-runpod")

    first = cloudctl.decide_storage(profile, stage="cp1", attempt=1)
    retry = cloudctl.decide_storage(profile, stage="cp1", attempt=2)
    cp2 = cloudctl.decide_storage(profile, stage="cp2", attempt=1)

    assert first.mode == "ephemeral"
    assert retry.mode == "network_volume"
    assert cp2.mode == "network_volume"
    assert retry.network_volume_size_gb == 200


def test_render_launch_default_is_dry_command_without_network_volume() -> None:
    profile = cloudctl.load_profile("mistral24b-runpod")

    command = cloudctl.render_launch_command(profile, stage="cp1", attempt=1)
    rendered = cloudctl.shell_join(command)

    assert command[:3] == ["runpodctl", "pod", "create"]
    assert "--template-id runpod-torch-v240" in rendered
    assert "--gpu-id 'NVIDIA H100 80GB HBM3'" in rendered
    assert "--volume-in-gb 100" in rendered
    assert "--network-volume-id" not in command


def test_render_launch_requires_volume_for_cp2() -> None:
    profile = cloudctl.load_profile("mistral24b-runpod")

    with pytest.raises(cloudctl.CloudctlError, match="network volume"):
        cloudctl.render_launch_command(profile, stage="cp2", attempt=1)

    command = cloudctl.render_launch_command(
        profile,
        stage="cp2",
        attempt=1,
        network_volume_id="nv-test",
    )
    assert "--network-volume-id" in command
    assert "nv-test" in command


def test_derive_direct_ssh_endpoint_from_pod_metadata() -> None:
    metadata = {
        "id": "pod-test",
        "runtime": {
            "ports": [
                {
                    "privatePort": 8888,
                    "isIpPublic": True,
                    "ip": "1.2.3.4",
                    "publicPort": 18888,
                },
                {
                    "privatePort": 22,
                    "isIpPublic": True,
                    "ip": "1.2.3.4",
                    "publicPort": 12222,
                },
            ]
        },
    }

    endpoint = cloudctl.derive_ssh_endpoint(metadata)

    assert endpoint.host == "1.2.3.4"
    assert endpoint.port == 12222


def test_runpod_snapshot_requests_all_pods(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []
    stopped_pod = {"id": "pod-stopped", "desiredStatus": "EXITED"}

    def fake_runpodctl_json(args: list[str]) -> object:
        calls.append(args)
        if args == ["pod", "list", "--all"]:
            return [stopped_pod]
        if args == ["network-volume", "list"]:
            return []
        if args == ["user"]:
            return {"currentSpendPerHr": 0}
        raise AssertionError(f"unexpected runpodctl args: {args}")

    monkeypatch.setattr(cloudctl, "runpodctl_json", fake_runpodctl_json)

    snapshot = cloudctl.runpod_snapshot()

    assert calls == [["pod", "list", "--all"], ["network-volume", "list"], ["user"]]
    assert snapshot.pods == [stopped_pod]


def test_ssh_info_can_read_metadata_json(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    metadata_path = tmp_path / "pod.json"
    metadata_path.write_text(
        json.dumps(
            {
                "runtime": {
                    "ports": [
                        {
                            "privatePort": 22,
                            "isIpPublic": True,
                            "ip": "5.6.7.8",
                            "publicPort": 43210,
                        }
                    ]
                }
            }
        ),
        encoding="utf-8",
    )

    code = cloudctl.main(
        ["ssh-info", "--pod-id", "pod-test", "--metadata-json", str(metadata_path)]
    )

    assert code == 0
    assert capsys.readouterr().out.strip() == (
        "ssh -o StrictHostKeyChecking=accept-new -p 43210 root@5.6.7.8"
    )


def test_cleanup_check_fails_on_leftover_paid_resources(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    snapshot = cloudctl.RunPodSnapshot(
        pods=[{"id": "pod-1", "name": "run", "desiredStatus": "RUNNING"}],
        network_volumes=[{"id": "nv-1", "name": "cache"}],
        user={"currentSpendPerHr": 2.99},
    )
    monkeypatch.setattr(cloudctl, "runpod_snapshot", lambda: snapshot)

    code = cloudctl.main(["cleanup-check"])

    output = capsys.readouterr().out
    assert code == 1
    assert "[fail] pods still exist" in output
    assert "[fail] network volumes still exist" in output
    assert "[fail] currentSpendPerHr is nonzero" in output


def test_cleanup_check_passes_for_zero_snapshot(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    snapshot = cloudctl.RunPodSnapshot(
        pods=[],
        network_volumes=[],
        user={"currentSpendPerHr": 0},
    )
    monkeypatch.setattr(cloudctl, "runpod_snapshot", lambda: snapshot)

    code = cloudctl.main(["cleanup-check"])

    assert code == 0
    assert "[ok] cleanup check passed" in capsys.readouterr().out
