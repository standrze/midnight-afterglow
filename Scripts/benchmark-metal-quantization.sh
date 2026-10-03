#!/usr/bin/env bash
set -euo pipefail

PACKAGE_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

WICK_BUILD_CONFIGURATION=release \
WICK_BUILD_PRODUCT=wick-metal-quant-bench \
  "$PACKAGE_ROOT/build-metal.sh"

BIN_DIR="$(
  cd "$PACKAGE_ROOT"
  swift build --configuration release --show-bin-path
)"
exec "$BIN_DIR/wick-metal-quant-bench" "$@"
