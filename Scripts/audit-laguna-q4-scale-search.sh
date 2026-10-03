#!/usr/bin/env bash
set -euo pipefail

PACKAGE_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PRODUCT="wick-q4-scale-search-audit"

WICK_BUILD_CONFIGURATION=release \
WICK_BUILD_PRODUCT="$PRODUCT" \
  "$PACKAGE_ROOT/build.sh"

case "$(uname -s)" in
  Darwin)
    BIN_DIR="$(
      cd "$PACKAGE_ROOT"
      swift build --configuration release --show-bin-path
    )"
    exec "$BIN_DIR/$PRODUCT" "$@"
    ;;
  Linux)
    source "$PACKAGE_ROOT/Scripts/swiftpm-scratch-path.sh"
    model_runner_configure_swiftpm_scratch "$PACKAGE_ROOT" Linux
    BIN_DIR="$(
      cd "$PACKAGE_ROOT"
      swift build --configuration release \
        "${MODEL_RUNNER_SWIFT_BUILD_SCRATCH_ARGS[@]}" --show-bin-path
    )"
    if [[ "${SPM_CUDA:-1}" == "1" ]]; then
      exec bash "$PACKAGE_ROOT/Scripts/cuda-runtime-environment.sh" \
        --package-root "$PACKAGE_ROOT" -- \
        "$BIN_DIR/$PRODUCT" "$@"
    fi
    exec "$BIN_DIR/$PRODUCT" "$@"
    ;;
  *)
    echo "Unsupported host: $(uname -s)" >&2
    exit 1
    ;;
esac
