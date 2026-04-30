# Cloud Adapter

This directory is the local contract for adapting Zombuul-style cloud orchestration to this repo. `vendor/zombuul` is an inspectable upstream reference pinned as a submodule, not an installed plugin or active skill set.

Use `scripts/infra/cloudctl.py` for status-only and dry-run checks:

```bash
uv run python scripts/infra/cloudctl.py status
uv run python scripts/infra/cloudctl.py preflight --profile mistral24b-runpod
uv run python scripts/infra/cloudctl.py render-launch --profile mistral24b-runpod --stage cp1 --attempt 1
uv run python scripts/infra/cloudctl.py ssh-info --pod-id "$POD_ID"
uv run python scripts/infra/cloudctl.py cleanup-check
```

The CP1 RunPod incident context is concise and operational:
[2026-04-29-cp1-runpod-postmortem.md](../../../notes/icml/mistral24b/2026-04-29-cp1-runpod-postmortem.md).
Treat postmortem setup commands as historical; current RunPod launch/setup
instructions live in [runbooks/runpod.md](runbooks/runpod.md).
Use the Zombuul submodule to mine fixes for those failure modes, then encode
only adapted behavior here.

Profiles are TOML files in `scripts/infra/cloud/profiles/`. The current adapter maps Zombuul ideas into local pieces: profile sizing, bundle-smoke repo proof, direct SSH derivation, explicit artifact sync runbooks, status/cleanup checks, and Mistral wrapper/tmux discipline.

## Guardrails

- Default commands must not create pods, volumes, or other paid resources.
- Paid launch remains a rendered command until an operator deliberately runs it.
- Preserve the Mistral rules from the strategy memo: template-based RunPod launch with the baked `/opt/h-neurons/.venv`, no remote `uv sync` or tool installs on pods, `active-run-status` before staging outputs, no local 24B load, H100/BF16 assertions, explicit artifact sync, and the CP1-retry/CP2+ network-volume threshold.
- Do not run `/plugin install`, do not symlink vendored skills, and do not make Zombuul prompts authoritative for this repo.
