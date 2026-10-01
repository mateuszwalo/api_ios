#!/usr/bin/env bash
# Criterion 1: a text prompt from another device returns a well-formed OpenAI response with
# non-zero usage.
#
# `usage` is checked rather than glanced at because the numbers are the deliverable: a server
# that answers correctly but reports zeros leaves the exercise without a result.

source "$(dirname "$0")/common.sh"
FAILED=0

head1 "1. Text completion"

RESPONSE=$(jq -n --arg model "$MODEL" '{
  model: $model,
  messages: [
    {role: "system", content: "You are a concise assistant."},
    {role: "user", content: "Name three primary colours. Answer in one short sentence."}
  ],
  max_completion_tokens: 200,
  temperature: 0,
  stream: false
}' | chat)

if [ -z "$RESPONSE" ]; then
  fail "no response from ${BASE_URL}"
  exit 1
fi

echo "$RESPONSE" | jq . >/dev/null 2>&1 || { fail "response is not valid JSON:"; echo "$RESPONSE"; exit 1; }

[ "$(echo "$RESPONSE" | jq -r '.object')" = "chat.completion" ] \
  && pass "object is chat.completion" || fail "object field is wrong or missing"

[ "$(echo "$RESPONSE" | jq -r '.choices[0].message.role')" = "assistant" ] \
  && pass "assistant message present" || fail "no assistant message"

CONTENT=$(echo "$RESPONSE" | jq -r '.choices[0].message.content')
[ -n "$CONTENT" ] && pass "content is not empty" || fail "content is empty"

FINISH=$(echo "$RESPONSE" | jq -r '.choices[0].finish_reason')
case "$FINISH" in
  stop|length) pass "finish_reason is $FINISH" ;;
  *) fail "finish_reason is '$FINISH', expected stop or length" ;;
esac

PROMPT_TOKENS=$(echo "$RESPONSE" | jq -r '.usage.prompt_tokens')
COMPLETION_TOKENS=$(echo "$RESPONSE" | jq -r '.usage.completion_tokens')
TOTAL=$(echo "$RESPONSE" | jq -r '.usage.total_tokens')

[ "$PROMPT_TOKENS" -gt 0 ] 2>/dev/null \
  && pass "prompt_tokens = $PROMPT_TOKENS" || fail "prompt_tokens is zero or missing"
[ "$COMPLETION_TOKENS" -gt 0 ] 2>/dev/null \
  && pass "completion_tokens = $COMPLETION_TOKENS" || fail "completion_tokens is zero or missing"
[ "$TOTAL" -eq $((PROMPT_TOKENS + COMPLETION_TOKENS)) ] 2>/dev/null \
  && pass "total_tokens adds up" || fail "total_tokens does not equal the sum"

info "answer: $CONTENT"
info "timings: $(echo "$RESPONSE" | jq -c '.timings')"

exit $FAILED
