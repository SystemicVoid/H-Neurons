from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from scripts.infra import cloudctl


EXPECTED_RUNPOD_REPOSITORY = "ghcr.io/systemicvoid/h-neurons-runpod"
EXPECTED_RUNPOD_DIGEST = (
    "sha256:15c9114da2420086a83ff85f261d07427e60ffd62835a177e587c3f2e431508e"
)
EXPECTED_RUNPOD_IMAGE = f"{EXPECTED_RUNPOD_REPOSITORY}@{EXPECTED_RUNPOD_DIGEST}"


def editable_profile(name: str) -> dict[str, Any]:
    profile = cloudctl.load_profile(name)
    return {**profile, "runpod": dict(profile["runpod"])}


def test_load_mistral_runpod_profile() -> None:
    profile = cloudctl.load_profile("mistral24b-runpod")

    assert profile["provider"] == "runpod"
    assert profile["name"] == "mistral24b-runpod"


def test_runpod_production_profiles_require_baked_template_pin() -> None:
    for name in ("mistral24b-runpod", "gemma3-4b-runpod"):
        profile = cloudctl.load_profile(name)
        runpod = profile["runpod"]

        assert runpod["template_id"] == "v88hqzvuxk"
        assert "image" not in runpod
        assert runpod["expected_image_repository"] == EXPECTED_RUNPOD_REPOSITORY
        assert runpod["expected_image_digest"] == EXPECTED_RUNPOD_DIGEST


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
    assert "--template-id v88hqzvuxk" in rendered
    assert "--image" not in command
    assert "--gpu-id 'NVIDIA A100-SXM4-80GB'" in rendered
    assert "--volume-in-gb 100" in rendered
    assert "--network-volume-id" not in command


def test_render_launch_rejects_profile_with_template_and_image() -> None:
    profile = editable_profile("mistral24b-runpod")
    profile["runpod"]["image"] = "ghcr.io/systemicvoid/h-neurons-runpod:latest"

    with pytest.raises(cloudctl.CloudctlError, match="exactly one"):
        cloudctl.render_launch_command(profile, stage="cp1", attempt=1)


def test_render_launch_rejects_direct_image_for_pinned_profile() -> None:
    profile = editable_profile("mistral24b-runpod")
    profile["runpod"].pop("template_id")
    profile["runpod"]["image"] = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1"

    with pytest.raises(cloudctl.CloudctlError, match="requires template_id"):
        cloudctl.render_launch_command(profile, stage="cp1", attempt=1)


def test_render_launch_rejects_unpinned_runpod_profile() -> None:
    profile = editable_profile("mistral24b-runpod")
    profile["runpod"].pop("expected_image_repository")
    profile["runpod"].pop("expected_image_digest")

    with pytest.raises(cloudctl.CloudctlError, match="expected_image_digest"):
        cloudctl.render_launch_command(profile, stage="cp1", attempt=1)


def test_render_launch_allows_direct_image_only_with_pin_ack() -> None:
    profile = editable_profile("mistral24b-runpod")
    profile["runpod"].pop("template_id")
    profile["runpod"]["image"] = EXPECTED_RUNPOD_IMAGE

    command = cloudctl.render_launch_command(
        profile,
        stage="cp1",
        attempt=1,
        allow_image_fallback=True,
        confirmed_image_pin=True,
    )

    assert "--image" in command
    assert EXPECTED_RUNPOD_IMAGE in command


def test_verify_runpod_image_pin_accepts_matching_template(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = cloudctl.load_profile("mistral24b-runpod")
    calls: list[list[str]] = []

    def fake_runpodctl_json(args: list[str]) -> object:
        calls.append(args)
        if args == ["template", "get", "v88hqzvuxk"]:
            return {"imageName": EXPECTED_RUNPOD_IMAGE}
        raise AssertionError(f"unexpected runpodctl args: {args}")

    monkeypatch.setattr(cloudctl, "runpodctl_json", fake_runpodctl_json)

    check = cloudctl.verify_runpod_image_pin(profile)

    assert calls == [["template", "get", "v88hqzvuxk"]]
    assert check is not None
    assert check.matched
    assert check.actual_image == EXPECTED_RUNPOD_IMAGE


def test_render_launch_cli_verifies_image_pin_without_polluting_stdout(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: list[list[str]] = []

    def fake_runpodctl_json(args: list[str]) -> object:
        calls.append(args)
        if args == ["template", "get", "v88hqzvuxk"]:
            return {"imageName": EXPECTED_RUNPOD_IMAGE}
        raise AssertionError(f"unexpected runpodctl args: {args}")

    monkeypatch.setattr(cloudctl, "runpodctl_json", fake_runpodctl_json)

    code = cloudctl.main(
        [
            "render-launch",
            "--profile",
            "mistral24b-runpod",
            "--stage",
            "cp1",
            "--attempt",
            "1",
        ]
    )

    output = capsys.readouterr()
    assert code == 0
    assert calls == [["template", "get", "v88hqzvuxk"]]
    assert output.err == ""
    assert output.out.startswith("runpodctl pod create ")
    assert "image pin" not in output.out


def test_verify_runpod_image_pin_rejects_template_digest_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = cloudctl.load_profile("mistral24b-runpod")

    def fake_runpodctl_json(args: list[str]) -> object:
        if args == ["template", "get", "v88hqzvuxk"]:
            return {
                "imageName": (
                    "ghcr.io/systemicvoid/h-neurons-runpod@"
                    "sha256:0000000000000000000000000000000000000000000000000000000000000000"
                )
            }
        raise AssertionError(f"unexpected runpodctl args: {args}")

    monkeypatch.setattr(cloudctl, "runpodctl_json", fake_runpodctl_json)

    with pytest.raises(cloudctl.CloudctlError, match="image pin mismatch"):
        cloudctl.verify_runpod_image_pin(profile)


def test_verify_runpod_image_pin_fallback_bypasses_drift_after_ack(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = cloudctl.load_profile("mistral24b-runpod")

    def fake_runpodctl_json(args: list[str]) -> object:
        if args == ["template", "get", "v88hqzvuxk"]:
            return {"imageName": "runpod/pytorch:2.4.0-py3.11-cuda12.4.1"}
        raise AssertionError(f"unexpected runpodctl args: {args}")

    monkeypatch.setattr(cloudctl, "runpodctl_json", fake_runpodctl_json)

    check = cloudctl.verify_runpod_image_pin(
        profile,
        allow_image_fallback=True,
        confirmed_image_pin=True,
    )

    assert check is not None
    assert not check.matched
    assert check.actual_image == "runpod/pytorch:2.4.0-py3.11-cuda12.4.1"


def test_image_fallback_flags_must_be_paired() -> None:
    profile = editable_profile("mistral24b-runpod")
    profile["runpod"].pop("template_id")
    profile["runpod"]["image"] = EXPECTED_RUNPOD_IMAGE

    with pytest.raises(cloudctl.CloudctlError, match="must be used together"):
        cloudctl.render_launch_command(
            profile,
            stage="cp1",
            attempt=1,
            allow_image_fallback=True,
        )


def test_render_launch_requires_volume_for_cp2() -> None:
    profile = cloudctl.load_profile("mistral24b-runpod")

    with pytest.raises(cloudctl.CloudctlError, match="network volume"):
        cloudctl.render_launch_command(profile, stage="cp2", attempt=1)

    with pytest.raises(cloudctl.CloudctlError, match="confirmed-gpu-stock"):
        cloudctl.render_launch_command(
            profile,
            stage="cp2",
            attempt=1,
            network_volume_id="nv-test",
            data_center_id="US-CA-2",
        )

    command = cloudctl.render_launch_command(
        profile,
        stage="cp2",
        attempt=1,
        network_volume_id="nv-test",
        data_center_id="US-CA-2",
        confirmed_gpu_stock=True,
    )
    rendered = cloudctl.shell_join(command)
    assert "--network-volume-id" in command
    assert "nv-test" in command
    assert "--data-center-ids US-CA-2" in rendered
    assert "US-OR-1" not in rendered


def test_render_launch_requires_exact_datacenter_for_network_volume() -> None:
    profile = cloudctl.load_profile("mistral24b-runpod")

    with pytest.raises(cloudctl.CloudctlError, match="data-center-id"):
        cloudctl.render_launch_command(
            profile,
            stage="cp2",
            attempt=1,
            network_volume_id="nv-test",
        )


def test_render_launch_rejects_multi_datacenter_override_for_network_volume() -> None:
    profile = cloudctl.load_profile("mistral24b-runpod")

    with pytest.raises(cloudctl.CloudctlError, match="single datacenter"):
        cloudctl.render_launch_command(
            profile,
            stage="cp2",
            attempt=1,
            network_volume_id="nv-test",
            data_center_id="US-CA-2,US-OR-1",
            confirmed_gpu_stock=True,
        )


def test_render_launch_requires_explicit_gpu_fallback_acknowledgement() -> None:
    profile = cloudctl.load_profile("mistral24b-runpod")

    with pytest.raises(cloudctl.CloudctlError, match="allow-gpu-fallback"):
        cloudctl.render_launch_command(
            profile,
            stage="cp2",
            attempt=1,
            network_volume_id="nv-test",
            data_center_id="US-CA-2",
            gpu_id=cloudctl.DEFAULT_RUNPOD_H100_GPU_ID,
            confirmed_gpu_stock=True,
        )

    command = cloudctl.render_launch_command(
        profile,
        stage="cp2",
        attempt=1,
        network_volume_id="nv-test",
        data_center_id="US-CA-2",
        gpu_id=cloudctl.DEFAULT_RUNPOD_H100_GPU_ID,
        allow_gpu_fallback=True,
        confirmed_gpu_stock=True,
    )
    rendered = cloudctl.shell_join(command)

    assert "--gpu-id 'NVIDIA H100 80GB HBM3'" in rendered
    assert "--data-center-ids US-CA-2" in rendered


def test_render_network_volume_create_uses_profile_size_and_datacenter() -> None:
    profile = cloudctl.load_profile("mistral24b-runpod")

    command = cloudctl.render_network_volume_create_command(
        profile,
        name="hneurons-mistral24b-cp2",
        data_center_id="US-CA-2",
    )
    rendered = cloudctl.shell_join(command)

    assert command[:3] == ["runpodctl", "network-volume", "create"]
    assert "--name hneurons-mistral24b-cp2" in rendered
    assert "--size 200" in rendered
    assert "--data-center-id US-CA-2" in rendered


def test_render_network_volume_create_rejects_multi_datacenter() -> None:
    profile = cloudctl.load_profile("mistral24b-runpod")

    with pytest.raises(cloudctl.CloudctlError, match="single datacenter"):
        cloudctl.render_network_volume_create_command(
            profile,
            data_center_id="US-CA-2,US-OR-1",
        )


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


def test_derive_direct_ssh_endpoint_from_runpod_ssh_info() -> None:
    metadata = {
        "id": "pod-test",
        "ip": "1.2.3.4",
        "port": 12222,
        "ssh_command": "ssh root@1.2.3.4 -p 12222",
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


def test_ssh_info_falls_back_to_runpod_ssh_info(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: list[list[str]] = []

    def fake_runpodctl_json(args: list[str]) -> object:
        calls.append(args)
        if args == [
            "pod",
            "get",
            "pod-test",
            "--include-machine",
            "--include-network-volume",
        ]:
            return {"ssh": {"error": "pod not ready"}}
        if args == ["ssh", "info", "pod-test"]:
            return {"ip": "5.6.7.8", "port": 43210}
        raise AssertionError(f"unexpected runpodctl args: {args}")

    monkeypatch.setattr(cloudctl, "runpodctl_json", fake_runpodctl_json)

    code = cloudctl.main(["ssh-info", "--pod-id", "pod-test"])

    assert code == 0
    assert calls == [
        [
            "pod",
            "get",
            "pod-test",
            "--include-machine",
            "--include-network-volume",
        ],
        ["ssh", "info", "pod-test"],
    ]
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


def test_cleanup_check_allows_storage_only_spend_when_volumes_allowed(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    snapshot = cloudctl.RunPodSnapshot(
        pods=[],
        network_volumes=[{"id": "nv-1", "name": "cache"}],
        user={"currentSpendPerHr": 0.019},
    )
    monkeypatch.setattr(cloudctl, "runpod_snapshot", lambda: snapshot)

    code = cloudctl.main(["cleanup-check", "--allow-network-volumes"])

    output = capsys.readouterr().out
    assert code == 0
    assert "[ok] currentSpendPerHr treated as retained network-volume storage" in output
    assert "[ok] cleanup check passed" in output


def test_cleanup_check_still_fails_on_spend_without_allowed_volume(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    snapshot = cloudctl.RunPodSnapshot(
        pods=[],
        network_volumes=[],
        user={"currentSpendPerHr": 0.019},
    )
    monkeypatch.setattr(cloudctl, "runpod_snapshot", lambda: snapshot)

    code = cloudctl.main(["cleanup-check", "--allow-network-volumes"])

    assert code == 1
    assert "[fail] currentSpendPerHr is nonzero" in capsys.readouterr().out


def test_cleanup_check_still_fails_when_pods_exist_even_if_volumes_allowed(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    snapshot = cloudctl.RunPodSnapshot(
        pods=[{"id": "pod-1", "desiredStatus": "RUNNING"}],
        network_volumes=[{"id": "nv-1", "name": "cache"}],
        user={"currentSpendPerHr": 0.019},
    )
    monkeypatch.setattr(cloudctl, "runpod_snapshot", lambda: snapshot)

    code = cloudctl.main(["cleanup-check", "--allow-network-volumes"])

    output = capsys.readouterr().out
    assert code == 1
    assert "[fail] pods still exist" in output
    assert "[fail] currentSpendPerHr is nonzero" in output
