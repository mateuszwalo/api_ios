#!/usr/bin/env bash
# Runs the acceptance criteria in order and prints a summary.
#
#   export BASE_URL=http://<ipad-ip>:8080/v1
#   scripts/acceptance/run-all.sh [image.jpg]
#
# Criterion 5 takes more than ten minutes by design; set SKIP_LONG=1 while iterating.

set -uo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
IMAGE="${1:-}"

declare -a NAMES=() RESULTS=()

run() {
  local name="$1"; shift
  "$@"
  local code=$?
  NAMES+=("$name")
  RESULTS+=("$code")
}

run "1. text"            "$DIR/01_text.sh"
if [ -n "$IMAGE" ]; then
  run "2. vision"        "$DIR/02_vision.sh" "$IMAGE"
else
  echo; echo "== 2. Vision == skipped: pass an image path to run it"
  NAMES+=("2. vision"); RESULTS+=("skip")
fi
run "3. schema"          "$DIR/03_schema.sh"
run "4. concurrency"     "$DIR/04_parallel.sh"
if [ "${SKIP_LONG:-0}" = "1" ]; then
  echo; echo "== 5. Long request == skipped (SKIP_LONG=1)"
  NAMES+=("5. long request"); RESULTS+=("skip")
else
  run "5. long request"  "$DIR/05_long_request.sh"
fi
run "6. logs"            "$DIR/06_logs.sh"

echo
echo "================ summary ================"
FAILURES=0
for i in "${!NAMES[@]}"; do
  case "${RESULTS[$i]}" in
    0)    printf '  \033[32mPASS\033[0m  %s\n' "${NAMES[$i]}" ;;
    skip) printf '  ----  %s (skipped)\n' "${NAMES[$i]}" ;;
    *)    printf '  \033[31mFAIL\033[0m  %s\n' "${NAMES[$i]}"; FAILURES=$((FAILURES + 1)) ;;
  esac
done
echo "========================================="
[ "$FAILURES" -eq 0 ] && echo "all criteria met" || echo "$FAILURES criteria failed"
exit "$FAILURES"
