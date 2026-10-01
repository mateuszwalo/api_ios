#!/usr/bin/env bash
# Criterion 5: a request lasting more than ten minutes succeeds.
#
# Nothing in the path may impose a deadline: not the server, not the TCP keepalive, not a
# NAT table in a router that drops idle mappings after a few minutes. That last one is why
# the test runs over the network from another machine rather than against localhost — on
# localhost it would pass without ever exercising the thing most likely to fail.

source "$(dirname "$0")/common.sh"
FAILED=0

PROMPT_WORDS="${PROMPT_WORDS:-4000}"
MAX_TOKENS="${MAX_TOKENS:-8000}"
head1 "5. Long request (target: over 10 minutes)"

PROMPT="$(make_prompt "$PROMPT_WORDS")"
info "prompt of about $PROMPT_WORDS words, ceiling $MAX_TOKENS tokens"
info "this is expected to take a long time; nothing here will time out"

START=$(date +%s)
RESPONSE=$(jq -n --arg model "$MODEL" --arg prompt "$PROMPT" --argjson max "$MAX_TOKENS" '{
  model: $model,
  messages: [
    {role: "user", content: ($prompt + "\n\nSummarise the list above, then describe each item in one sentence.")}
  ],
  max_completion_tokens: $max,
  temperature: 0,
  stream: false
}' | chat)
ELAPSED=$(( $(date +%s) - START ))

echo "$RESPONSE" | jq . >/dev/null 2>&1 \
  || { fail "no valid response after ${ELAPSED}s"; echo "$RESPONSE" | head -c 1000; exit 1; }

CONTENT=$(echo "$RESPONSE" | jq -r '.choices[0].message.content // empty')
[ -n "$CONTENT" ] && pass "completed after ${ELAPSED}s" || fail "empty content after ${ELAPSED}s"

if [ "$ELAPSED" -ge 600 ]; then
  pass "exceeded ten minutes (${ELAPSED}s) and still completed"
else
  info "took ${ELAPSED}s, under the ten-minute mark — raise PROMPT_WORDS or MAX_TOKENS to"
  info "make the run longer if you want the criterion proven rather than merely plausible"
fi

echo "$RESPONSE" | jq -c '{usage, timings}'

exit $FAILED
