#!/usr/bin/env bash
set -euo pipefail

QUANTIZATION_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
QUANTIZER="${MODEL_QUANTIZER_BIN:-$QUANTIZATION_ROOT/.build/release/model-runner-quantize}"
SOURCE="$QUANTIZATION_ROOT/Tests/Fixtures/TalkieTiny"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/talkie-quantizer-cli.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT

"$QUANTIZER" "$SOURCE" "$WORK/q4" --dry-run --standard-q4 > "$WORK/q4.log"
grep -Fq 'Profile: talkie-dense-affine' "$WORK/q4.log"
grep -Fq 'Quantizable modules: 9' "$WORK/q4.log"
test ! -e "$WORK/q4"

"$QUANTIZER" "$SOURCE" "$WORK/q8" --dry-run --standard-q8 > "$WORK/q8.log"
grep -Fq 'standard Q8 modules: 9' "$WORK/q8.log"
grep -Fq 'lm_head [requested]' "$WORK/q8.log"
grep -Fq 'model.blocks.0.attn.attn_query [requested]' "$WORK/q8.log"
test ! -e "$WORK/q8"

"$QUANTIZER" "$SOURCE" "$WORK/mixed" --dry-run --q8-module lm_head > "$WORK/mixed.log"
grep -Fq 'Q4 ScaleSearch Linear/SwitchLinear: 7' "$WORK/mixed.log"
grep -Fq 'standard Q8 modules: 1' "$WORK/mixed.log"
test ! -e "$WORK/mixed"

"$QUANTIZER" "$SOURCE" "$WORK/all-q8" --dry-run --q8-module '*' > "$WORK/all-q8.log"
grep -Fq 'standard Q8 modules: 9' "$WORK/all-q8.log"
test ! -e "$WORK/all-q8"

"$QUANTIZER" "$SOURCE" "$WORK/embedding-only" --dry-run --standard-q4 \
  --skip-module 'model.blocks.*' --skip-module lm_head > "$WORK/embedding-only.log"
grep -Fq 'standard Q4 embedding/custom modules: 1' "$WORK/embedding-only.log"
test ! -e "$WORK/embedding-only"

for SIZE in 8589934592 1e300 1e-20; do
  if "$QUANTIZER" "$SOURCE" "$WORK/invalid-size" --dry-run --max-shard-gib "$SIZE" > "$WORK/invalid-size.log" 2>&1; then
    echo "Invalid shard size was accepted: $SIZE" >&2
    exit 1
  fi
  grep -Fq 'signed 64-bit byte count' "$WORK/invalid-size.log"
done

mkdir "$WORK/existing"
if "$QUANTIZER" "$SOURCE" "$WORK/existing" --dry-run > "$WORK/existing.log" 2>&1; then
  echo 'Dry run accepted an existing destination without overwrite' >&2
  exit 1
fi
grep -Fq 'destination already exists' "$WORK/existing.log"

ln -s "$SOURCE" "$WORK/source-alias"
if "$QUANTIZER" "$SOURCE" "$WORK/source-alias" --dry-run --overwrite > "$WORK/alias.log" 2>&1; then
  echo 'Dry run accepted a source alias as destination' >&2
  exit 1
fi
grep -Fq 'must not overlap' "$WORK/alias.log"

if "$QUANTIZER" "$SOURCE" "$WORK/invalid" --dry-run --standard-q4 --standard-q8 > "$WORK/invalid.log" 2>&1; then
  echo 'Conflicting Q4/Q8 controls were accepted' >&2
  exit 1
fi
grep -Fq 'mutually exclusive' "$WORK/invalid.log"
test ! -e "$WORK/invalid"

echo 'Talkie quantizer CLI checks passed'
