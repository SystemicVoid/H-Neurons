#!/usr/bin/env python3
"""Run CodeRabbit review with a bounded unattended fallback.

This helper keeps the waiting inside one process so the agent does not need to
poll CodeRabbit with repeated inference turns. It writes a machine-readable
status file and falls back to headless Codex review when CodeRabbit times out,
rate limits, or exits with an error.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import queue
import re
import shlex
import shutil
import subprocess
import sys
import threading
import time
from typing import TextIO, cast


DEFAULT_TIMEOUT_SECONDS = 30 * 60
DEFAULT_FALLBACK_TIMEOUT_SECONDS = 30 * 60
MIN_CODERABBIT_VERSION = (0, 4, 0)
SCHEMA_VERSION = "coderabbit_review_watch/v1"

RATE_LIMIT_RE = re.compile(
    r"\b(429|rate[- ]?limit(?:ed|ing)?|too many requests|quota exceeded)\b",
    re.IGNORECASE,
)

FALLBACK_PROMPT = """\
CodeRabbit did not produce a usable review result. Run an independent headless
code review of the current uncommitted changes. Focus on correctness,
regressions, security, data loss, and test gaps. Put findings first, ordered by
severity, with file and line references when available. Ignore pure style nits.
"""


@dataclass(frozen=True)
class CommandResult:
    command: list[str]
    exit_code: int | None
    reason: str
    output_path: Path
    elapsed_seconds: float


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def parse_version(text: str) -> tuple[int, int, int] | None:
    match = re.search(r"(\d+)\.(\d+)\.(\d+)", text)
    if match is None:
        return None
    return (int(match.group(1)), int(match.group(2)), int(match.group(3)))


def version_at_least(
    version: tuple[int, int, int] | None,
    minimum: tuple[int, int, int] = MIN_CODERABBIT_VERSION,
) -> bool:
    return version is not None and version >= minimum


def is_rate_limit_text(text: str) -> bool:
    return RATE_LIMIT_RE.search(text) is not None


def command_display(command: Sequence[str]) -> str:
    return shlex.join(command)


def resolve_tool(name: str, env: Mapping[str, str]) -> str | None:
    path = shutil.which(name, path=env.get("PATH"))
    if path is not None:
        return path

    mise = shutil.which("mise", path=env.get("PATH"))
    if mise is None:
        return None

    try:
        result = subprocess.run(
            [mise, "which", name],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=10,
            env=dict(env),
        )
    except (OSError, subprocess.TimeoutExpired):
        return None

    if result.returncode != 0:
        return None

    resolved = result.stdout.strip()
    return resolved or None


def build_coderabbit_review_command(
    coderabbit_bin: str,
    passthrough_args: Sequence[str],
) -> list[str]:
    command = [coderabbit_bin, "review", "--agent", "--no-color"]
    command.extend(passthrough_args)
    return command


def build_codex_review_command(codex_bin: str, report_path: Path) -> list[str]:
    return [
        codex_bin,
        "exec",
        "review",
        "--uncommitted",
        "--output-last-message",
        str(report_path),
        FALLBACK_PROMPT,
    ]


def run_quick_command(
    command: Sequence[str],
    *,
    cwd: Path,
    env: Mapping[str, str],
    output_path: Path,
    timeout_seconds: int = 30,
) -> CommandResult:
    started = time.monotonic()
    try:
        result = subprocess.run(
            list(command),
            cwd=cwd,
            env=dict(env),
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout_seconds,
        )
        output_path.write_text(result.stdout, encoding="utf-8")
        return CommandResult(
            command=list(command),
            exit_code=result.returncode,
            reason="completed",
            output_path=output_path,
            elapsed_seconds=time.monotonic() - started,
        )
    except subprocess.TimeoutExpired as exc:
        output = exc.stdout or ""
        if isinstance(output, bytes):
            output = output.decode("utf-8", errors="replace")
        output_path.write_text(output, encoding="utf-8")
        return CommandResult(
            command=list(command),
            exit_code=None,
            reason="timeout",
            output_path=output_path,
            elapsed_seconds=time.monotonic() - started,
        )


def _reader(pipe: TextIO, output_queue: queue.Queue[str]) -> None:
    try:
        for line in iter(pipe.readline, ""):
            output_queue.put(line)
    finally:
        pipe.close()


def _stop_process(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def stream_command(
    command: Sequence[str],
    *,
    cwd: Path,
    env: Mapping[str, str],
    output_path: Path,
    timeout_seconds: int,
    stop_on_rate_limit: bool = False,
) -> CommandResult:
    started = time.monotonic()
    process = subprocess.Popen(
        list(command),
        cwd=cwd,
        env=dict(env),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None

    output_queue: queue.Queue[str] = queue.Queue()
    reader = threading.Thread(
        target=_reader,
        args=(process.stdout, output_queue),
        daemon=True,
    )
    reader.start()

    reason = "completed"
    with output_path.open("w", encoding="utf-8") as output:
        while True:
            elapsed = time.monotonic() - started
            if elapsed > timeout_seconds and process.poll() is None:
                reason = "timeout"
                _stop_process(process)

            try:
                line = output_queue.get(timeout=0.1)
            except queue.Empty:
                line = None

            if line is not None:
                output.write(line)
                output.flush()
                print(line, end="", flush=True)
                if stop_on_rate_limit and is_rate_limit_text(line):
                    reason = "rate_limit"
                    _stop_process(process)

            if process.poll() is not None and output_queue.empty():
                break

    reader.join(timeout=1)
    while not output_queue.empty():
        line = output_queue.get_nowait()
        with output_path.open("a", encoding="utf-8") as output:
            output.write(line)
        print(line, end="", flush=True)

    return CommandResult(
        command=list(command),
        exit_code=process.poll(),
        reason=reason,
        output_path=output_path,
        elapsed_seconds=time.monotonic() - started,
    )


def write_status(status_path: Path, status: Mapping[str, object]) -> None:
    status_path.write_text(
        json.dumps(status, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def initial_status(
    log_dir: Path, cwd: Path, review_args: Sequence[str]
) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "state": "started",
        "backend": None,
        "reason": None,
        "started_at": utc_now(),
        "finished_at": None,
        "cwd": str(cwd),
        "review_args": list(review_args),
        "log_dir": str(log_dir),
        "status_path": str(log_dir / "status.json"),
        "tools": {},
        "commands": {},
        "report_path": None,
    }


def result_payload(result: CommandResult) -> dict[str, object]:
    return {
        "command": command_display(result.command),
        "exit_code": result.exit_code,
        "reason": result.reason,
        "output_path": str(result.output_path),
        "elapsed_seconds": round(result.elapsed_seconds, 3),
    }


def finish(
    status: dict[str, object],
    *,
    state: str,
    backend: str | None,
    reason: str,
    report_path: Path | None,
) -> None:
    status["state"] = state
    status["backend"] = backend
    status["reason"] = reason
    status["finished_at"] = utc_now()
    status["report_path"] = str(report_path) if report_path is not None else None


def status_section(status: Mapping[str, object], key: str) -> dict[str, object]:
    section = status[key]
    if not isinstance(section, dict):
        raise TypeError(f"status[{key!r}] is not an object")
    return cast(dict[str, object], section)


def run_codex_fallback(
    *,
    reason: str,
    status: dict[str, object],
    status_path: Path,
    log_dir: Path,
    cwd: Path,
    env: Mapping[str, str],
    fallback_timeout_seconds: int,
) -> int:
    tools = status_section(status, "tools")
    commands = status_section(status, "commands")

    codex_bin = resolve_tool("codex", env)
    if codex_bin is None:
        finish(
            status,
            state="failed",
            backend=None,
            reason=f"{reason}; codex_not_found",
            report_path=None,
        )
        write_status(status_path, status)
        return 20

    tools["codex"] = {"path": codex_bin}
    report_path = log_dir / "codex-review.md"
    output_path = log_dir / "codex-review.log"
    command = build_codex_review_command(codex_bin, report_path)
    print(
        f"CodeRabbit {reason}; running headless Codex fallback: "
        f"{command_display(command)}",
        file=sys.stderr,
        flush=True,
    )
    result = stream_command(
        command,
        cwd=cwd,
        env=env,
        output_path=output_path,
        timeout_seconds=fallback_timeout_seconds,
    )
    commands["codex_review"] = result_payload(result)

    if result.exit_code == 0 and result.reason == "completed":
        if not report_path.exists() or report_path.stat().st_size == 0:
            report_path.write_text(
                output_path.read_text(encoding="utf-8"),
                encoding="utf-8",
            )
        finish(
            status,
            state="fallback_completed",
            backend="codex",
            reason=reason,
            report_path=report_path,
        )
        write_status(status_path, status)
        return 10

    finish(
        status,
        state="failed",
        backend="codex",
        reason=f"{reason}; codex_{result.reason}",
        report_path=output_path,
    )
    write_status(status_path, status)
    return 20


def run(args: argparse.Namespace, *, cwd: Path, env: Mapping[str, str]) -> int:
    review_args = list(args.review_args)
    if review_args and review_args[0] == "--":
        review_args = review_args[1:]

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    log_dir = (
        Path(args.log_dir) if args.log_dir else cwd / "logs" / "coderabbit" / timestamp
    )
    log_dir.mkdir(parents=True, exist_ok=True)
    status_path = log_dir / "status.json"
    status = initial_status(log_dir, cwd, review_args)
    write_status(status_path, status)

    tools = status_section(status, "tools")
    commands = status_section(status, "commands")

    coderabbit_bin = resolve_tool("coderabbit", env)
    if coderabbit_bin is None:
        return run_codex_fallback(
            reason="coderabbit_not_found",
            status=status,
            status_path=status_path,
            log_dir=log_dir,
            cwd=cwd,
            env=env,
            fallback_timeout_seconds=args.fallback_timeout_seconds,
        )

    version_result = run_quick_command(
        [coderabbit_bin, "--version"],
        cwd=cwd,
        env=env,
        output_path=log_dir / "coderabbit-version.log",
    )
    help_result = run_quick_command(
        [coderabbit_bin, "review", "--help"],
        cwd=cwd,
        env=env,
        output_path=log_dir / "coderabbit-review-help.log",
    )
    commands["coderabbit_version"] = result_payload(version_result)
    commands["coderabbit_review_help"] = result_payload(help_result)

    version_text = version_result.output_path.read_text(encoding="utf-8")
    help_text = help_result.output_path.read_text(encoding="utf-8")
    version = parse_version(version_text)
    tools["coderabbit"] = {
        "path": coderabbit_bin,
        "version_text": version_text.strip(),
        "version": ".".join(str(part) for part in version) if version else None,
    }
    write_status(status_path, status)

    if version_result.exit_code != 0 or help_result.exit_code != 0:
        return run_codex_fallback(
            reason="coderabbit_discovery_failed",
            status=status,
            status_path=status_path,
            log_dir=log_dir,
            cwd=cwd,
            env=env,
            fallback_timeout_seconds=args.fallback_timeout_seconds,
        )

    if not version_at_least(version):
        return run_codex_fallback(
            reason="coderabbit_version_too_old",
            status=status,
            status_path=status_path,
            log_dir=log_dir,
            cwd=cwd,
            env=env,
            fallback_timeout_seconds=args.fallback_timeout_seconds,
        )

    if "--agent" not in help_text:
        return run_codex_fallback(
            reason="coderabbit_agent_unsupported",
            status=status,
            status_path=status_path,
            log_dir=log_dir,
            cwd=cwd,
            env=env,
            fallback_timeout_seconds=args.fallback_timeout_seconds,
        )

    command = build_coderabbit_review_command(coderabbit_bin, review_args)
    print(
        f"Running CodeRabbit: {command_display(command)}", file=sys.stderr, flush=True
    )
    result = stream_command(
        command,
        cwd=cwd,
        env=env,
        output_path=log_dir / "coderabbit-review.log",
        timeout_seconds=args.timeout_seconds,
        stop_on_rate_limit=True,
    )
    commands["coderabbit_review"] = result_payload(result)
    write_status(status_path, status)

    if result.exit_code == 0 and result.reason == "completed":
        finish(
            status,
            state="coderabbit_completed",
            backend="coderabbit",
            reason="completed",
            report_path=result.output_path,
        )
        write_status(status_path, status)
        return 0

    if result.reason == "rate_limit":
        reason = "rate_limit"
    elif result.reason == "timeout":
        reason = "timeout"
    else:
        reason = "coderabbit_error"

    return run_codex_fallback(
        reason=reason,
        status=status,
        status_path=status_path,
        log_dir=log_dir,
        cwd=cwd,
        env=env,
        fallback_timeout_seconds=args.fallback_timeout_seconds,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run CodeRabbit review once, wait up to 30 minutes inside this "
            "process, and fall back to headless Codex review on timeout, "
            "rate limit, or error."
        )
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=DEFAULT_TIMEOUT_SECONDS,
        help="Maximum seconds to wait for CodeRabbit before Codex fallback.",
    )
    parser.add_argument(
        "--fallback-timeout-seconds",
        type=int,
        default=DEFAULT_FALLBACK_TIMEOUT_SECONDS,
        help="Maximum seconds to wait for the Codex fallback.",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help="Directory for status and logs. Defaults to logs/coderabbit/<timestamp>.",
    )
    parser.add_argument(
        "review_args",
        nargs=argparse.REMAINDER,
        help="Arguments passed to `coderabbit review` after `--`.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return run(args, cwd=Path.cwd().resolve(), env=os.environ)


if __name__ == "__main__":
    raise SystemExit(main())
