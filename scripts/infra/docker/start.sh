#!/usr/bin/env bash
set -euo pipefail

if [[ -n "${PUBLIC_KEY:-}" ]]; then
    mkdir -p /root/.ssh
    chmod 700 /root/.ssh
    grep -qxF "${PUBLIC_KEY}" /root/.ssh/authorized_keys 2>/dev/null \
        || echo "${PUBLIC_KEY}" >>/root/.ssh/authorized_keys
    chmod 600 /root/.ssh/authorized_keys
fi

ssh-keygen -A

driver="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || echo "0")"
echo "[start.sh] nvidia-driver=${driver} (CUDA 13.x needs R580+; CUDA 13.0 GA needs >=580.65.06)" >&2

exec /usr/sbin/sshd -D -e
