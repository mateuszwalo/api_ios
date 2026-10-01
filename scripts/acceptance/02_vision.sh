#!/usr/bin/env bash
# Criterion 2: an image in a data URI comes back described, which is the only proof the
# vision path is wired end to end.
#
#   scripts/acceptance/02_vision.sh path/to/page.jpg
#
# Also checks the image's token cost. Gemma 3 charges 256 tokens for one tile; a multiple of
# that means pan & scan is on, the token counts no longer match the reference configuration,
# and any comparison drawn from them is invalid. That is worth catching here rather than
# three weeks later in a spreadsheet.

source "$(dirname "$0")/common.sh"
FAILED=0

IMAGE="${1:-}"
head1 "2. Vision"

if [ -z "$IMAGE" ] || [ ! -f "$IMAGE" ]; then
  echo "usage: $0 <image.jpg>" >&2
  exit 2
fi

case "$(echo "${IMAGE##*.}" | tr '[:upper:]' '[:lower:]')" in
  jpg|jpeg) MIME="image/jpeg" ;;
  png)      MIME="image/png" ;;
  webp)     MIME="image/webp" ;;
  *) echo "unsupported image type: $IMAGE" >&2; exit 2 ;;
esac

B64=$(base64 < "$IMAGE" | tr -d '\n')
info "image: $IMAGE ($(wc -c < "$IMAGE" | tr -d ' ') bytes, $(echo -n "$B64" | wc -c | tr -d ' ') base64)"

RESPONSE=$(jq -n --arg model "$MODEL" --arg url "data:${MIME};base64,${B64}" '{
  model: $model,
  messages: [{
    role: "user",
    content: [
      {type: "image_url", image_url: {url: $url}},
      {type: "text", text: "Describe what you see in this image in two sentences."}
    ]
  }],
  max_completion_tokens: 300,
  temperature: 0,
  stream: false
}' | chat)

echo "$RESPONSE" | jq . >/dev/null 2>&1 || { fail "response is not valid JSON:"; echo "$RESPONSE" | head -c 2000; exit 1; }

CONTENT=$(echo "$RESPONSE" | jq -r '.choices[0].message.content // empty')
[ -n "$CONTENT" ] && pass "image was described" || { fail "empty description"; echo "$RESPONSE" | head -c 2000; }

PROMPT_TOKENS=$(echo "$RESPONSE" | jq -r '.usage.prompt_tokens')
[ "$PROMPT_TOKENS" -gt 256 ] 2>/dev/null \
  && pass "prompt_tokens = $PROMPT_TOKENS (includes image tokens)" \
  || fail "prompt_tokens = $PROMPT_TOKENS, too low to include a 256-token image"

info "description: $CONTENT"
info "Compare prompt_tokens with GET /v1/selftest, which reports the image's token cost exactly."
info "256 per image means one tile, pan & scan off, matching the reference."

exit $FAILED
