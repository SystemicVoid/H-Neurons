from __future__ import annotations

import json
import os
from pathlib import Path
import shlex

from scripts.infra import coderabbit_review_watch as watch


def _write_executable(path: Path, body: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    path.chmod(0o755)
    return path


def _env_with_bin(bin_dir: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}{os.pathsep}{env.get('PATH', '')}"
    return env


def _parse_args(*args: str):
    return watch.build_parser().parse_args(list(args))


def _status(log_dir: Path) -> dict[str, object]:
    return json.loads((log_dir / "status.json").read_text(encoding="utf-8"))


def _codex_fake(bin_dir: Path) -> None:
    _write_executable(
        bin_dir / "codex",
        """#!/usr/bin/env bash
set -euo pipefail
report=""
while [[ "$#" -gt 0 ]]; do
  if [[ "$1" == "--output-last-message" ]]; then
    shift
    report="$1"
  fi
  shift || true
done
printf 'codex fallback review\\n' | tee "$report"
""",
    )


def test_parse_version_and_rate_limit_detection() -> None:
    assert watch.parse_version("0.4.5") == (0, 4, 5)
    assert watch.parse_version("coderabbit version 1.2.3") == (1, 2, 3)
    assert watch.parse_version("not a version") is None
    assert watch.version_at_least((0, 4, 0))
    assert not watch.version_at_least((0, 3, 9))
    assert watch.is_rate_limit_text("HTTP 429: too many requests")
    assert watch.is_rate_limit_text("rate limit exceeded")
    assert not watch.is_rate_limit_text("ordinary review output")


def test_builds_coderabbit_command_with_agent_and_passthrough() -> None:
    assert watch.build_coderabbit_review_command(
        "/bin/coderabbit",
        ["--type", "uncommitted"],
    ) == [
        "/bin/coderabbit",
        "review",
        "--agent",
        "--no-color",
        "--type",
        "uncommitted",
    ]


def test_run_quick_command_records_exec_error(tmp_path: Path) -> None:
    output_path = tmp_path / "quick.log"

    result = watch.run_quick_command(
        [str(tmp_path / "missing-tool")],
        cwd=tmp_path,
        env=os.environ,
        output_path=output_path,
    )

    assert result.exit_code is None
    assert result.reason == "exec_error"
    assert "Execution failed: FileNotFoundError:" in output_path.read_text(
        encoding="utf-8"
    )


def test_stream_command_records_exec_error(tmp_path: Path) -> None:
    output_path = tmp_path / "stream.log"

    result = watch.stream_command(
        [str(tmp_path / "missing-tool")],
        cwd=tmp_path,
        env=os.environ,
        output_path=output_path,
        timeout_seconds=30,
    )

    assert result.exit_code is None
    assert result.reason == "exec_error"
    assert "Execution failed: FileNotFoundError:" in output_path.read_text(
        encoding="utf-8"
    )


def test_coderabbit_success_writes_status(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    args_file = tmp_path / "coderabbit-args.txt"
    _write_executable(
        bin_dir / "coderabbit",
        f"""#!/usr/bin/env bash
set -euo pipefail
if [[ "$1" == "--version" ]]; then
  echo "0.4.5"
  exit 0
fi
if [[ "$1" == "review" && "${{2:-}}" == "--help" ]]; then
  echo "Usage: coderabbit review --agent --no-color"
  exit 0
fi
printf '%s\\n' "$*" > {shlex.quote(str(args_file))}
echo "CodeRabbit review complete"
""",
    )
    log_dir = tmp_path / "logs"

    exit_code = watch.run(
        _parse_args("--log-dir", str(log_dir), "--", "--type", "uncommitted"),
        cwd=tmp_path,
        env=_env_with_bin(bin_dir),
    )

    status = _status(log_dir)
    assert exit_code == 0
    assert status["schema_version"] == watch.SCHEMA_VERSION
    assert status["state"] == "coderabbit_completed"
    assert status["backend"] == "coderabbit"
    assert status["reason"] == "completed"
    assert "--agent --no-color --type uncommitted" in args_file.read_text(
        encoding="utf-8"
    )


def test_rate_limit_runs_codex_fallback(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    _write_executable(
        bin_dir / "coderabbit",
        """#!/usr/bin/env bash
set -euo pipefail
if [[ "$1" == "--version" ]]; then
  echo "0.4.5"
  exit 0
fi
if [[ "$1" == "review" && "${2:-}" == "--help" ]]; then
  echo "Usage: coderabbit review --agent --no-color"
  exit 0
fi
echo "HTTP 429 rate limit exceeded"
sleep 5
""",
    )
    _codex_fake(bin_dir)
    log_dir = tmp_path / "logs"

    exit_code = watch.run(
        _parse_args("--log-dir", str(log_dir), "--timeout-seconds", "20", "--"),
        cwd=tmp_path,
        env=_env_with_bin(bin_dir),
    )

    status = _status(log_dir)
    assert exit_code == 10
    assert status["state"] == "fallback_completed"
    assert status["backend"] == "codex"
    assert status["reason"] == "rate_limit"
    assert Path(str(status["report_path"])).read_text(encoding="utf-8").strip()


def test_timeout_runs_codex_fallback(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    _write_executable(
        bin_dir / "coderabbit",
        """#!/usr/bin/env bash
set -euo pipefail
if [[ "$1" == "--version" ]]; then
  echo "0.4.5"
  exit 0
fi
if [[ "$1" == "review" && "${2:-}" == "--help" ]]; then
  echo "Usage: coderabbit review --agent --no-color"
  exit 0
fi
sleep 5
""",
    )
    _codex_fake(bin_dir)
    log_dir = tmp_path / "logs"

    exit_code = watch.run(
        _parse_args(
            "--log-dir",
            str(log_dir),
            "--timeout-seconds",
            "1",
            "--fallback-timeout-seconds",
            "5",
            "--",
        ),
        cwd=tmp_path,
        env=_env_with_bin(bin_dir),
    )

    status = _status(log_dir)
    assert exit_code == 10
    assert status["state"] == "fallback_completed"
    assert status["reason"] == "timeout"


def test_coderabbit_error_runs_codex_fallback(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    _write_executable(
        bin_dir / "coderabbit",
        """#!/usr/bin/env bash
set -euo pipefail
if [[ "$1" == "--version" ]]; then
  echo "0.4.5"
  exit 0
fi
if [[ "$1" == "review" && "${2:-}" == "--help" ]]; then
  echo "Usage: coderabbit review --agent --no-color"
  exit 0
fi
echo "CodeRabbit failed"
exit 3
""",
    )
    _codex_fake(bin_dir)
    log_dir = tmp_path / "logs"

    exit_code = watch.run(
        _parse_args("--log-dir", str(log_dir), "--"),
        cwd=tmp_path,
        env=_env_with_bin(bin_dir),
    )

    status = _status(log_dir)
    assert exit_code == 10
    assert status["state"] == "fallback_completed"
    assert status["reason"] == "coderabbit_error"


def test_old_coderabbit_version_runs_codex_fallback(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    _write_executable(
        bin_dir / "coderabbit",
        """#!/usr/bin/env bash
set -euo pipefail
if [[ "$1" == "--version" ]]; then
  echo "0.3.9"
  exit 0
fi
if [[ "$1" == "review" && "${2:-}" == "--help" ]]; then
  echo "Usage: coderabbit review --agent"
  exit 0
fi
echo "should not run review"
exit 99
""",
    )
    _codex_fake(bin_dir)
    log_dir = tmp_path / "logs"

    exit_code = watch.run(
        _parse_args("--log-dir", str(log_dir), "--"),
        cwd=tmp_path,
        env=_env_with_bin(bin_dir),
    )

    status = _status(log_dir)
    assert exit_code == 10
    assert status["state"] == "fallback_completed"
    assert status["reason"] == "coderabbit_version_too_old"
    commands = status["commands"]
    assert isinstance(commands, dict)
    assert "coderabbit_review" not in commands
