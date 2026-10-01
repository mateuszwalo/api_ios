#!/usr/bin/env bash
# Produces the measurement table for the README.
#
#   export BASE_URL=http://<ipad-ip>:8080/v1
#   scripts/bench.sh | tee bench.md
#
# One pass per prompt size. The 6k row is the one the exercise is about; the others exist to
# show how prefill scales, which is what a projection needs.
#
# Run it twice, once with "Reuse KV cache between requests" off and once with it on, and
# label the output. Off is the honest prefill cost. On is what a real deployment sees, where
# consecutive calls share a system prompt. The two differ by a lot, and quoting one when the
# other was measured is the easiest way to be wrong by an order of magnitude.

source "$(dirname "$0")/acceptance/common.sh"

SIZES="${SIZES:-700 4500 12000 24000}"      # words, roughly 1.35 tokens each
MAX_TOKENS="${MAX_TOKENS:-400}"

STATS=$(curl -sS --max-time 30 "${BASE_URL}/stats" 2>/dev/null)
MODEL_NAME=$(echo "$STATS" | jq -r '.model // "unknown"')
CONTEXT=$(echo "$STATS" | jq -r '.context_length // 0')

echo "# Measured performance"
echo
echo "Model: \`${MODEL_NAME}\` · context ${CONTEXT} · KV cache f16 · iSWA on · pan & scan off"
echo "Device: _fill in_ · iPadOS _fill in_ · llama.cpp \`$(cat "$(dirname "$0")/../LLAMA_CPP_TAG" 2>/dev/null || echo unknown)\`"
echo "KV reuse between requests: _off / on — state which_"
echo
echo "| prompt tokens | prefill tok/s | generation tok/s | prefill | generation | peak footprint | thermal |"
echo "|---:|---:|---:|---:|---:|---:|:--|"

for words in $SIZES; do
  PROMPT="$(make_prompt "$words")"
  RESPONSE=$(jq -n --arg model "$MODEL" --arg prompt "$PROMPT" --argjson max "$MAX_TOKENS" '{
    model: $model,
    messages: [{role: "user", content: ($prompt + "\n\nSummarise the list above in three sentences.")}],
    max_completion_tokens: $max,
    temperature: 0,
    stream: false
  }' | chat)

  if ! echo "$RESPONSE" | jq . >/dev/null 2>&1; then
    echo "| ~$words words | — | — | — | — | — | request failed |"
    continue
  fi

  PT=$(echo "$RESPONSE" | jq -r '.usage.prompt_tokens')
  PREFILL_TPS=$(echo "$RESPONSE" | jq -r '.timings.prefill_tps // 0')
  DECODE_TPS=$(echo "$RESPONSE" | jq -r '.timings.decode_tps // 0')
  PREFILL_MS=$(echo "$RESPONSE" | jq -r '.timings.prefill_ms // 0')
  DECODE_MS=$(echo "$RESPONSE" | jq -r '.timings.decode_ms // 0')

  AFTER=$(curl -sS --max-time 30 "${BASE_URL}/stats")
  PEAK=$(echo "$AFTER" | jq -r '.peak_footprint_bytes // 0')
  THERMAL=$(echo "$AFTER" | jq -r '.thermal_state // "unknown"')

  printf '| %s | %.1f | %.1f | %.1fs | %.1fs | %.2f GB | %s |\n' \
    "$PT" "$PREFILL_TPS" "$DECODE_TPS" \
    "$(echo "$PREFILL_MS" | awk '{print $1/1000}')" \
    "$(echo "$DECODE_MS" | awk '{print $1/1000}')" \
    "$(echo "$PEAK" | awk '{print $1/1073741824}')" \
    "$THERMAL"
done

echo
echo "Definitions: prefill tok/s = prompt_tokens / time to first token (tokenisation and image"
echo "encoding included). Generation tok/s = completion_tokens / (total − prefill)."
echo "Peak footprint is cumulative since the server started, and rises during generation"
echo "rather than during prefill."
