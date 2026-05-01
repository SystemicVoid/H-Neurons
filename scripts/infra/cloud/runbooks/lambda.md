# Lambda Runbook

Lambda support remains a wrapper around the existing scripts:

- `scripts/infra/monitor_lambda_capacity.py`
- `scripts/infra/lambda-bootstrap.sh`
- `scripts/infra/lambda_bootstrap_and_start_step1.py`
- `scripts/infra/lambda-run-python.sh`

Use Lambda credits for unfunded follow-ups or capacity opportunism. Do not replace the established bootstrap or runner from `cloudctl`; keep provider-specific behavior in the existing Lambda scripts and use the profile as a pointer layer.
