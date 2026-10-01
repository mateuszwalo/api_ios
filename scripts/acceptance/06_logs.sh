#!/usr/bin/env bash
# Criterion 6: the live log carries tokens, duration and tok/s for every request.
#
# Read over HTTP rather than off the screen: the iPad will be lying unlocked on the other
# side of the room for the length of a run, and a measurement that can only be read by
# walking over to it is a measurement nobody takes.

source "$(dirname "$0")/common.sh"
FAILED=0

head1 "6. Live log and stats"

ROOT="${BASE_URL%/v1}"

STATS=$(curl -sS --max-time 30 "${BASE_URL}/stats")
echo "$STATS" | jq . >/dev/null 2>&1 && pass "GET /v1/stats answers with JSON" || { fail "no stats"; exit 1; }

for key in model_loaded queue_depth served footprint_bytes peak_footprint_bytes thermal_state; do
  echo "$STATS" | jq -e "has(\"$key\")" >/dev/null \
    && pass "stats carry $key" || fail "stats are missing $key"
done
info "stats: $(echo "$STATS" | jq -c .)"

LOGS=$(curl -sS --max-time 60 "${ROOT}/logs.jsonl")
[ -n "$LOGS" ] && pass "GET /logs.jsonl returns entries" || fail "the log is empty — run 01_text.sh first"

LAST=$(echo "$LOGS" | tail -n 1)
if [ -n "$LAST" ]; then
  for field in promptTokens completionTokens totalMs prefillMs decodeMs status thermalState; do
    echo "$LAST" | jq -e "has(\"$field\")" >/dev/null \
      && pass "log entry carries $field" || fail "log entry is missing $field"
  done
  PROMPT=$(echo "$LAST" | jq -r '.promptTokens')
  PREFILL=$(echo "$LAST" | jq -r '.prefillMs')
  if [ "$PREFILL" -gt 0 ] 2>/dev/null; then
    info "last request: ${PROMPT} prompt tokens, prefill $(echo "$LAST" | jq -r '.prefillMs')ms, generation $(echo "$LAST" | jq -r '.decodeMs')ms"
    pass "timings are real, not placeholders"
  else
    fail "prefillMs is zero — the timings are not being measured"
  fi
fi

SELFTEST=$(curl -sS --max-time 300 "${BASE_URL}/selftest")
if echo "$SELFTEST" | jq . >/dev/null 2>&1; then
  pass "GET /v1/selftest answers"
  info "chat template ok: $(echo "$SELFTEST" | jq -r '.chat_template.ok')"
  info "tokens per image: $(echo "$SELFTEST" | jq -r '.image_tokens.tokens_attributable_to_image // "n/a"') (expected 256)"
  info "grammar ok: $(echo "$SELFTEST" | jq -r '.grammar.ok')"
else
  info "self-test did not answer; it needs a loaded model"
fi

exit $FAILED
