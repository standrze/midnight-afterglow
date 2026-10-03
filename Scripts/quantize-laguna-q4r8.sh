#!/usr/bin/env bash
set -euo pipefail

# Q4R8: affine Q4 group-64 weights with affine Q8 group-64 MoE routers.

PACKAGE_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

WICK_BUILD_CONFIGURATION=release \
WICK_BUILD_PRODUCT=wick-laguna-quantize \
  "$PACKAGE_ROOT/build.sh"

case "$(uname -s)" in
  Darwin)
    BIN_DIR="$(
      cd "$PACKAGE_ROOT"
      swift build --configuration release --show-bin-path
    )"
    exec "$BIN_DIR/wick-laguna-quantize" "$@"
    ;;
  Linux)
    # Resolve the same optional scratch directory used by build.sh, then launch
    # through the verified CUDA environment when this is a CUDA build. `--cpu`
    # controls conversion placement; it does not change how the binary links.
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
        "$BIN_DIR/wick-laguna-quantize" "$@"
    fi
    exec "$BIN_DIR/wick-laguna-quantize" "$@"
    ;;
  *)
    echo "Unsupported host: $(uname -s)" >&2
    exit 1
    ;;
esac
