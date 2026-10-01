#!/usr/bin/env bash
# Shared setup for the acceptance scripts.
#
# Run these from a laptop on the same network as the iPad, never on the iPad itself:
# the point is to prove the endpoint works for another device, which is how it will be used.
#
#   export BASE_URL=http://192.168.1.50:8080/v1
#   scripts/acceptance/run-all.sh
#
# Requires: curl, jq.

set -uo pipefail

BASE_URL="${BASE_URL:-http://localhost:8080/v1}"
API_KEY="${API_KEY:-not-checked}"
MODEL="${MODEL:-gemma3-4b-quality:latest}"

for tool in curl jq; do
  command -v "$tool" >/dev/null 2>&1 || { echo "missing required tool: $tool" >&2; exit 2; }
done

pass() { printf '  \033[32mPASS\033[0m %s\n' "$1"; }
fail() { printf '  \033[31mFAIL\033[0m %s\n' "$1"; FAILED=1; }
info() { printf '  ---- %s\n' "$1"; }
head1() { printf '\n== %s ==\n' "$1"; }

# POST a JSON body to /chat/completions. No timeout: a request may legitimately take
# twenty minutes, and curl's default would be the thing that broke the test.
chat() {
  curl -sS --no-progress-meter --max-time 0 \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer ${API_KEY}" \
    -X POST "${BASE_URL}/chat/completions" \
    --data-binary @-
}

# A prompt of roughly N tokens. Words map to tokens closely enough for a size target.
make_prompt() {
  local words="$1"
  awk -v n="$words" 'BEGIN {
    for (i = 0; i < n; i++) printf "item %d of the inventory list, ", i
  }'
}
