#!/usr/bin/env bash
# Re-link 6 here.now sites under hugonguyen.com after DNS verification.
# Run after domain status flips to "active":
#   curl -sS https://here.now/api/v1/domains/hugonguyen.com \
#     -H "Authorization: Bearer $(cat ~/.herenow/credentials)" | jq .status

set -euo pipefail
API_KEY=$(cat ~/.herenow/credentials)
DOMAIN="hugonguyen.com"

# slug:location pairs (path under hugonguyen.com)
PAIRS=(
  "aware-fresco-4a2q:h-neurons-bluedot"
  "ivory-mirage-f33e:henry-schein-ai-sec"
  "jolly-banner-c6ke:josh-update"
  "lunar-turret-czz8:conference-v1"
  "graceful-vigil-3bgt:conference-v2"
  "silent-tablet-tg62:avl-deck"
)

for pair in "${PAIRS[@]}"; do
  slug="${pair%%:*}"
  loc="${pair##*:}"
  echo "=== Linking $slug -> $DOMAIN/$loc ==="
  curl -sS https://here.now/api/v1/links \
    -H "Authorization: Bearer $API_KEY" \
    -H "Content-Type: application/json" \
    -d "{\"location\": \"$loc\", \"slug\": \"$slug\", \"domain\": \"$DOMAIN\"}"
  echo
done

echo "=== Final link list ==="
curl -sS "https://here.now/api/v1/links?domain=$DOMAIN" \
  -H "Authorization: Bearer $API_KEY" | jq .
