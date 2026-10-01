#!/usr/bin/env bash
# Criterion 4: twenty concurrent requests are all accepted, served one at a time, and none
# of the connections is dropped.
#
# A dropped connection is the failure that matters. The client does not retry transport
# errors, so one severed socket ends a run that may already have been going for half an
# hour — far worse than being slow. Hence the check is on how many came back, not on how
# long they took.

source "$(dirname "$0")/common.sh"
FAILED=0

COUNT="${COUNT:-20}"
head1 "4. $COUNT concurrent requests"

WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

info "firing $COUNT requests at once; they are expected to be served one after another"
START=$(date +%s)

for i in $(seq 1 "$COUNT"); do
  (
    BODY=$(jq -n --arg model "$MODEL" --arg n "$i" '{
      model: $model,
      messages: [{role: "user", content: ("Reply with exactly the number " + $n + " and nothing else.")}],
      max_completion_tokens: 20,
      temperature: 0,
      stream: false
    }')
    CODE=$(printf '%s' "$BODY" | curl -sS --max-time 0 -o "$WORK/$i.json" -w '%{http_code}' \
      -H "Content-Type: application/json" -H "Authorization: Bearer ${API_KEY}" \
      -X POST "${BASE_URL}/chat/completions" --data-binary @- 2>"$WORK/$i.err")
    echo "$CODE" > "$WORK/$i.code"
  ) &
done
wait

ELAPSED=$(( $(date +%s) - START ))
OK=0; BAD=0; DROPPED=0

for i in $(seq 1 "$COUNT"); do
  CODE=$(cat "$WORK/$i.code" 2>/dev/null || echo "000")
  if [ "$CODE" = "200" ]; then
    OK=$((OK + 1))
  elif [ "$CODE" = "000" ]; then
    DROPPED=$((DROPPED + 1))
    info "request $i: no response — $(head -c 200 "$WORK/$i.err" 2>/dev/null)"
  else
    BAD=$((BAD + 1))
    info "request $i: HTTP $CODE — $(head -c 200 "$WORK/$i.json" 2>/dev/null)"
  fi
done

[ "$OK" -eq "$COUNT" ] && pass "all $COUNT requests answered with 200" \
  || fail "$OK of $COUNT answered ($BAD error responses, $DROPPED dropped connections)"
[ "$DROPPED" -eq 0 ] && pass "no connection was dropped" \
  || fail "$DROPPED connections were dropped — this ends a client run outright"

info "wall clock: ${ELAPSED}s for $COUNT requests served serially"

exit $FAILED
