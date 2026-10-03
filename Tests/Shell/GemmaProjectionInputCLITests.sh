#!/usr/bin/env bash
set -euo pipefail
PACKAGE_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
WICK_BINARY="${WICK_TEST_EXECUTABLE:-$PACKAGE_ROOT/.build/release/wick}"
STATS_BINARY="$(dirname "$WICK_BINARY")/wick-gemma-activation-stats"
TASK_TEMP="$(mktemp -d)"
trap 'rm -rf "$TASK_TEMP"' EXIT
"$STATS_BINARY" --help > "$TASK_TEMP/help.txt"
rg -q -- '--projection-input-plan' "$TASK_TEMP/help.txt"
rg -q -- '--projection-input-output' "$TASK_TEMP/help.txt"
if "$STATS_BINARY" /missing-source /missing-corpus "$TASK_TEMP/statistics.safetensors" \
    --projection-input-plan /missing-plan > "$TASK_TEMP/missing-output.log" 2>&1; then
    echo 'Expected the incomplete export option pair to fail.' >&2
    exit 1
fi
rg -q 'requires both --projection-input-plan and --projection-input-output' "$TASK_TEMP/missing-output.log"
if "$STATS_BINARY" /missing-source /missing-corpus "$TASK_TEMP/statistics.safetensors" \
    --projection-input-output "$TASK_TEMP/inputs" > "$TASK_TEMP/missing-plan.log" 2>&1; then
    echo 'Expected the incomplete export option pair to fail.' >&2
    exit 1
fi
rg -q 'requires both --projection-input-plan and --projection-input-output' "$TASK_TEMP/missing-plan.log"
test ! -e "$TASK_TEMP/statistics.safetensors"
test ! -e "$TASK_TEMP/inputs"
echo 'Projection input CLI help and pre-load validation passed.'
