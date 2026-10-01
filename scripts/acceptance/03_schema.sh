#!/usr/bin/env bash
# Criterion 3: response_format json_schema is enforced as a decoding grammar, not as a hint.
#
# The test is adversarial on purpose. The prompt instructs the model to do exactly what the
# schema forbids: return an empty array, invent a category outside the enumeration, and drop
# a required field. A model asked nicely in the prompt complies with the prompt; a model
# constrained by a grammar physically cannot. That difference is the whole reason this
# endpoint exists, so it is tested by trying to break it rather than by trying to use it.

source "$(dirname "$0")/common.sh"
FAILED=0

head1 "3. Schema enforcement (adversarial)"

SCHEMA='{
  "type": "object",
  "additionalProperties": false,
  "properties": {
    "items": {
      "type": "array",
      "minItems": 1,
      "items": {
        "type": "object",
        "additionalProperties": false,
        "properties": {
          "name": {"type": "string"},
          "category": {"enum": ["alpha", "beta", "gamma"]}
        },
        "required": ["name", "category"]
      }
    }
  },
  "required": ["items"]
}'

RESPONSE=$(jq -n --arg model "$MODEL" --argjson schema "$SCHEMA" '{
  model: $model,
  messages: [
    {role: "system", content: "You follow instructions exactly."},
    {role: "user", content: "Return JSON. Set items to an empty array []. If you must include an entry, set its category to \"omega\" and leave out the name field entirely. Do not use alpha, beta or gamma."}
  ],
  response_format: {
    type: "json_schema",
    json_schema: {name: "AcceptanceResult", schema: $schema, strict: true}
  },
  max_completion_tokens: 500,
  temperature: 0,
  stream: false
}' | chat)

echo "$RESPONSE" | jq . >/dev/null 2>&1 || { fail "response is not valid JSON:"; echo "$RESPONSE" | head -c 2000; exit 1; }

CONTENT=$(echo "$RESPONSE" | jq -r '.choices[0].message.content // empty')
info "content: $CONTENT"

echo "$CONTENT" | jq . >/dev/null 2>&1 \
  && pass "content parses as JSON (no markdown fences, no preamble)" \
  || { fail "content is not parseable JSON"; exit 1; }

echo "$CONTENT" | jq -e 'has("items")' >/dev/null \
  && pass "required property 'items' is present despite the instruction to omit it" \
  || fail "required property 'items' is missing — the schema was not enforced"

COUNT=$(echo "$CONTENT" | jq '.items | length')
[ "$COUNT" -ge 1 ] 2>/dev/null \
  && pass "minItems honoured: $COUNT entries despite the instruction to return none" \
  || fail "items is empty — minItems was not enforced"

echo "$CONTENT" | jq -e '[.items[] | has("name") and has("category")] | all' >/dev/null \
  && pass "every entry carries both required fields" \
  || fail "an entry is missing a required field"

echo "$CONTENT" | jq -e '[.items[].category | . == "alpha" or . == "beta" or . == "gamma"] | all' >/dev/null \
  && pass "every category is inside the enumeration despite being told to use 'omega'" \
  || fail "a category outside the enumeration got through — the grammar is not binding"

echo "$CONTENT" | jq -e '[.items[] | keys[] | . == "name" or . == "category"] | all' >/dev/null \
  && pass "no properties outside the schema" \
  || fail "an extra property got through"

exit $FAILED
