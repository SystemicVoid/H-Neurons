#!/usr/bin/env bash
# =============================================================================
# Lambda Cloud Instance Grabber — polls for capacity and auto-launches
# =============================================================================
# Polls the Lambda API every INTERVAL seconds for any matching instance type.
# When capacity appears, launches immediately and exits.
#
# Usage:
#   LAMBDA_API_KEY=... bash scripts/infra/lambda_grab.sh
#   # or source from .env:
#   source .env && bash scripts/infra/lambda_grab.sh
# =============================================================================
set -euo pipefail

# --- Configuration -----------------------------------------------------------
LAMBDA_API_KEY="${LAMBDA_API_KEY:?Set LAMBDA_API_KEY}"
API_BASE="https://cloud.lambda.ai/api/v1"
SSH_KEY_NAME="${LAMBDA_SSH_KEY:-excurio}"
INSTANCE_NAME="${LAMBDA_INSTANCE_NAME:-excurio-hneurons}"
INTERVAL="${LAMBDA_POLL_INTERVAL:-15}"

# Preferred types in priority order (cheapest single-GPU first)
# A6000 is no longer in Lambda's catalog; these are the closest alternatives
PREFERRED_TYPES=(
    "gpu_1x_gh200"        # 96GB $1.99/hr  — best value (ARM64)
    "gpu_1x_h100_pcie"    # 80GB $2.86/hr  — x86
)

# --- Helpers -----------------------------------------------------------------
api() {
    curl -sf -H "Authorization: Bearer $LAMBDA_API_KEY" "$@"
}

timestamp() {
    date '+%Y-%m-%d %H:%M:%S'
}

log() {
    echo "[$(timestamp)] $*"
}

# --- Pre-flight check --------------------------------------------------------
log "Lambda Grab starting"
log "Watching for: ${PREFERRED_TYPES[*]}"
log "SSH key: $SSH_KEY_NAME | Instance name: $INSTANCE_NAME"
log "Poll interval: ${INTERVAL}s"
log "---"

# Verify no instances already running
RUNNING=$(api "$API_BASE/instances" | python3 -c "
import json,sys
d = json.load(sys.stdin)['data']
active = [i for i in d if i.get('status') not in ('terminating','terminated')]
print(len(active))
")
if [[ "$RUNNING" -gt 0 ]]; then
    log "ERROR: $RUNNING active instance(s) found. Aborting to avoid double-billing."
    api "$API_BASE/instances" | python3 -m json.tool
    exit 1
fi

# --- Poll loop ---------------------------------------------------------------
ATTEMPT=0
while true; do
    ATTEMPT=$((ATTEMPT + 1))

    CAPACITY_JSON=$(api "$API_BASE/instance-types" 2>/dev/null || echo '{"data":{}}')

    # Check each preferred type in priority order
    for ITYPE in "${PREFERRED_TYPES[@]}"; do
        REGIONS=$(echo "$CAPACITY_JSON" | python3 -c "
import json, sys
data = json.load(sys.stdin).get('data', {})
info = data.get('$ITYPE', {})
regions = [r['name'] for r in info.get('regions_with_capacity_available', [])]
print(' '.join(regions))
" 2>/dev/null || echo "")

        if [[ -n "$REGIONS" ]]; then
            # Pick first available region
            REGION=$(echo "$REGIONS" | awk '{print $1}')
            log ">>> CAPACITY FOUND: $ITYPE in $REGION (attempt #$ATTEMPT)"
            log "Launching..."

            log "Image: Lambda Stack 24.04 (family selector)"

            LAUNCH_PAYLOAD=$(python3 -c "
import json
print(json.dumps({
    'region_name': '$REGION',
    'instance_type_name': '$ITYPE',
    'ssh_key_names': ['$SSH_KEY_NAME'],
    'name': '$INSTANCE_NAME',
    'quantity': 1,
    'image': {'family': 'lambda-stack-24-04'}
}))
")

            LAUNCH_RESULT=$(curl -sf -X POST \
                -H "Authorization: Bearer $LAMBDA_API_KEY" \
                -H "Content-Type: application/json" \
                -d "$LAUNCH_PAYLOAD" \
                "$API_BASE/instance-operations/launch" 2>&1) || {
                log "Launch request FAILED (likely race condition / capacity gone):"
                echo "$LAUNCH_RESULT"
                log "Resuming polling..."
                continue
            }

            INSTANCE_ID=$(echo "$LAUNCH_RESULT" | python3 -c "
import json, sys
data = json.load(sys.stdin)
ids = data.get('data', {}).get('instance_ids', [])
print(ids[0] if ids else '')
" 2>/dev/null || echo "")

            if [[ -z "$INSTANCE_ID" ]]; then
                log "Launch returned unexpected response:"
                echo "$LAUNCH_RESULT"
                log "Resuming polling..."
                continue
            fi

            log "=== INSTANCE LAUNCHED ==="
            log "Instance ID: $INSTANCE_ID"
            log "Type: $ITYPE | Region: $REGION"
            log ""
            log "Waiting for instance to become active..."

            # Poll for active status + IP
            for i in $(seq 1 120); do
                INSTANCE_INFO=$(api "$API_BASE/instances/$INSTANCE_ID" 2>/dev/null || echo '{}')
                STATUS=$(echo "$INSTANCE_INFO" | python3 -c "
import json, sys
d = json.load(sys.stdin).get('data', {})
print(d.get('status', 'unknown'))
" 2>/dev/null || echo "unknown")
                IP=$(echo "$INSTANCE_INFO" | python3 -c "
import json, sys
d = json.load(sys.stdin).get('data', {})
print(d.get('ip', '') or '')
" 2>/dev/null || echo "")

                if [[ "$STATUS" == "active" && -n "$IP" ]]; then
                    log "Instance ACTIVE at $IP"
                    log ""
                    log "=== CONNECT ==="
                    log "  ssh ubuntu@$IP"
                    log ""
                    log "=== BOOTSTRAP ==="
                    log "  scp scripts/infra/lambda-bootstrap.sh ubuntu@$IP:~/"
                    log "  ssh ubuntu@$IP 'bash ~/lambda-bootstrap.sh'"
                    log ""

                    # Write instance info for other scripts
                    cat > /tmp/lambda-instance.env <<EOF
LAMBDA_INSTANCE_ID=$INSTANCE_ID
LAMBDA_INSTANCE_IP=$IP
LAMBDA_INSTANCE_TYPE=$ITYPE
LAMBDA_INSTANCE_REGION=$REGION
EOF
                    log "Instance details written to /tmp/lambda-instance.env"
                    exit 0
                fi

                log "  status=$STATUS ip=${IP:-pending} (${i}/120)"
                sleep 5
            done

            log "WARNING: Instance did not become active within 10 minutes."
            log "Check manually: https://cloud.lambda.ai/instances"
            exit 2
        fi
    done

    # No capacity found — show a compact status line
    if (( ATTEMPT % 20 == 1 )); then
        log "No capacity (attempt #$ATTEMPT) — polling every ${INTERVAL}s..."
    else
        printf "\r[$(timestamp)] Poll #%-6d no capacity" "$ATTEMPT"
    fi

    sleep "$INTERVAL"
done
