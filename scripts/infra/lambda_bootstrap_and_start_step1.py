#!/usr/bin/env python3
"""Bootstrap a launched Lambda instance and start Step 1 automatically.

This script is intended to be called after an auto-launch event from
`monitor_lambda_capacity.py`.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


API_BASE_DEFAULT = "https://cloud.lambda.ai/api/v1"

MODEL_CONFIGS = {
    "mistral7b": {
        "id": "mistralai/Mistral-7B-Instruct-v0.3",
        "dir": "Mistral-7B-Instruct-v0.3",
        "prefix": "mistral7b",
    },
    "mistral24b": {
        "id": "mistralai/Mistral-Small-24B-Instruct-2501",
        "dir": "Mistral-Small-24B-Instruct-2501",
        "prefix": "mistral24b",
    },
    "llama70b": {
        "id": "meta-llama/Llama-3.3-70B-Instruct",
        "dir": "Llama-3.3-70B-Instruct",
        "prefix": "llama70b",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bootstrap launched Lambda instance and start Step 1."
    )
    parser.add_argument(
        "--api-base", default=API_BASE_DEFAULT, help="Lambda API base URL."
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("LAMBDA_API_KEY", ""),
        help="Lambda API key (defaults to $LAMBDA_API_KEY).",
    )
    parser.add_argument(
        "--instance-id",
        default=os.getenv("LAMBDA_LAUNCH_INSTANCE_ID", ""),
        help="Launched instance ID (defaults to $LAMBDA_LAUNCH_INSTANCE_ID).",
    )
    parser.add_argument("--ssh-user", default="ubuntu", help="Remote SSH username.")
    parser.add_argument(
        "--ssh-key-path", default="", help="Optional SSH private key path."
    )
    parser.add_argument(
        "--local-repo",
        default=str(Path.cwd()),
        help="Local repo path to sync scripts from.",
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=10,
        help="Polling interval for instance readiness.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=1800,
        help="Timeout in seconds waiting for active instance.",
    )
    parser.add_argument(
        "--sample-num", type=int, default=10, help="Step 1 sampling count."
    )
    parser.add_argument(
        "--max-samples", type=int, default=3500, help="Step 1 max TriviaQA items."
    )
    parser.add_argument(
        "--model-choice",
        choices=["auto", "mistral7b", "mistral24b", "llama70b"],
        default="auto",
        help="Model selection for Step 1 and activation pipeline.",
    )
    return parser.parse_args()


def fetch_json(method: str, url: str, api_key: str) -> dict[str, Any]:
    req = urllib.request.Request(
        url,
        method=method,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
            "User-Agent": "hneurons-lambda-bootstrap/1.0",
        },
    )
    with urllib.request.urlopen(req, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def wait_for_active_instance(args: argparse.Namespace) -> dict[str, Any]:
    if not args.instance_id:
        raise ValueError(
            "Missing instance id. Set --instance-id or LAMBDA_LAUNCH_INSTANCE_ID."
        )

    deadline = time.time() + args.timeout
    last_status = "unknown"
    instance_url = f"{args.api_base}/instances/{args.instance_id}"

    while time.time() < deadline:
        payload = fetch_json("GET", instance_url, args.api_key)
        data = payload.get("data", {})
        status = str(data.get("status", "unknown"))
        ip = data.get("ip")
        if status != last_status:
            print(f"Instance {args.instance_id} status: {status}")
            last_status = status
        if status == "active" and ip:
            return data
        time.sleep(max(args.poll_interval, 5))

    raise TimeoutError(
        f"Timed out waiting for instance {args.instance_id} to become active."
    )


def choose_model(model_choice: str, instance_type_name: str) -> tuple[str, bool]:
    if model_choice in MODEL_CONFIGS:
        return model_choice, False
    lowered = instance_type_name.lower()
    if "gh200" in lowered or "h100" in lowered:
        return "llama70b", True
    return "mistral7b", False


def ssh_base(user: str, host: str, ssh_key_path: str) -> list[str]:
    cmd = ["ssh", "-o", "StrictHostKeyChecking=accept-new"]
    if ssh_key_path:
        cmd.extend(["-i", ssh_key_path])
    cmd.append(f"{user}@{host}")
    return cmd


def scp_base(ssh_key_path: str) -> list[str]:
    cmd = ["scp", "-o", "StrictHostKeyChecking=accept-new"]
    if ssh_key_path:
        cmd.extend(["-i", ssh_key_path])
    return cmd


def run_remote_script(user: str, host: str, ssh_key_path: str, script: str) -> None:
    cmd = ssh_base(user, host, ssh_key_path) + ["bash -s"]
    subprocess.run(cmd, input=script.encode("utf-8"), check=True)


def copy_scripts(local_repo: Path, user: str, host: str, ssh_key_path: str) -> None:
    scripts_dir = local_repo / "scripts"
    python_scripts = sorted(str(path) for path in scripts_dir.glob("*.py"))
    if not python_scripts:
        raise FileNotFoundError(f"No python scripts found in {scripts_dir}")

    subprocess.run(
        scp_base(ssh_key_path)
        + python_scripts
        + [f"{user}@{host}:~/h-neurons/scripts/"],
        check=True,
    )

    lambda_agents = scripts_dir / "lambda-AGENTS.md"
    if lambda_agents.exists():
        subprocess.run(
            scp_base(ssh_key_path)
            + [str(lambda_agents), f"{user}@{host}:~/h-neurons/AGENTS.md"],
            check=True,
        )


def remote_setup_script(model_id: str, model_dir: str) -> str:
    return f"""#!/usr/bin/env bash
set -euo pipefail

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"

mkdir -p ~/h-neurons/data/TriviaQA/rc.nocontext ~/h-neurons/data/activations ~/h-neurons/scripts ~/h-neurons/models ~/h-neurons/logs ~/models
cd ~/h-neurons

if [ ! -f pyproject.toml ]; then
  uv init --no-readme --python 3.12
fi

uv add torch transformers datasets accelerate scikit-learn joblib openai tqdm numpy huggingface-hub wandb

if [ ! -d ~/models/{model_dir} ]; then
  uv run huggingface-cli download {model_id} --local-dir ~/models/{model_dir} --local-dir-use-symlinks False
fi

if [ ! -f data/TriviaQA/rc.nocontext/triviaqa_train.parquet ]; then
  uv run python3 - <<'PY'
from datasets import load_dataset
ds = load_dataset('trivia_qa', 'rc.nocontext', split='train')
ds.to_parquet('data/TriviaQA/rc.nocontext/triviaqa_train.parquet')
print(f'TriviaQA saved: {{len(ds)}} questions')
PY
fi

tmux has-session -t hneurons 2>/dev/null || tmux new-session -d -s hneurons -c ~/h-neurons
"""


def remote_start_step1_script(
    model_dir: str, prefix: str, sample_num: int, max_samples: int
) -> str:
    output_file = f"data/{prefix}_TriviaQA_consistency_samples.jsonl"
    log_file = f"logs/{prefix}_step1_collect.log"
    window_name = f"step1-{prefix}"
    cmd = (
        "uv run python3 scripts/collect_responses.py "
        f"--model_path ~/models/{model_dir} "
        "--data_path data/TriviaQA/rc.nocontext/triviaqa_train.parquet "
        f"--output_path {output_file} "
        f"--sample_num {sample_num} --max_samples {max_samples} "
        "--backend transformers --judge_type rule"
    )
    return f"""#!/usr/bin/env bash
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
cd ~/h-neurons
mkdir -p logs
tmux has-session -t hneurons 2>/dev/null || tmux new-session -d -s hneurons -c ~/h-neurons
if tmux list-windows -t hneurons -F '#W' | grep -qx {shlex.quote(window_name)}; then
  tmux kill-window -t hneurons:{shlex.quote(window_name)}
fi
tmux new-window -t hneurons -n {shlex.quote(window_name)} {shlex.quote(f"cd ~/h-neurons && {cmd} 2>&1 | tee -a {log_file}")}
"""


def main() -> int:
    args = parse_args()
    if not args.api_key:
        print(
            "Missing API key. Set --api-key or export LAMBDA_API_KEY.", file=sys.stderr
        )
        return 2

    local_repo = Path(args.local_repo).expanduser().resolve()
    if not local_repo.exists():
        print(f"Local repo not found: {local_repo}", file=sys.stderr)
        return 2

    try:
        instance = wait_for_active_instance(args)
    except (
        urllib.error.URLError,
        urllib.error.HTTPError,
        ValueError,
        TimeoutError,
    ) as exc:
        print(f"Failed waiting for active instance: {exc}", file=sys.stderr)
        return 1

    instance_type_name = instance.get("instance_type", {}).get("name", "")
    ip = instance.get("ip")
    if not isinstance(ip, str) or not ip:
        print("Instance is active but has no public IP.", file=sys.stderr)
        return 1

    selected_key, has_fallback = choose_model(args.model_choice, instance_type_name)
    selected = MODEL_CONFIGS[selected_key]
    fallback = (
        MODEL_CONFIGS["mistral24b"]
        if has_fallback and selected_key != "mistral24b"
        else None
    )

    print(f"Instance active: {args.instance_id} ({instance_type_name}) @ {ip}")
    print(f"Primary model: {selected['id']}")

    try:
        run_remote_script(
            args.ssh_user,
            ip,
            args.ssh_key_path,
            remote_setup_script(selected["id"], selected["dir"]),
        )
    except subprocess.CalledProcessError as exc:
        if fallback is None:
            print(f"Remote setup failed: {exc}", file=sys.stderr)
            return 1
        print("Primary model setup failed, retrying with Mistral-7B fallback...")
        selected = fallback
        run_remote_script(
            args.ssh_user,
            ip,
            args.ssh_key_path,
            remote_setup_script(selected["id"], selected["dir"]),
        )

    copy_scripts(local_repo, args.ssh_user, ip, args.ssh_key_path)
    run_remote_script(
        args.ssh_user,
        ip,
        args.ssh_key_path,
        remote_start_step1_script(
            model_dir=selected["dir"],
            prefix=selected["prefix"],
            sample_num=args.sample_num,
            max_samples=args.max_samples,
        ),
    )

    print("Remote setup complete and Step 1 started in tmux session 'hneurons'.")
    print(f"Attach command: ssh {args.ssh_user}@{ip} -t 'tmux attach -t hneurons'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
