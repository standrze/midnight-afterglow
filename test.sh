#!/usr/bin/env bash
set -euo pipefail

QUANTIZATION_ROOT="$(cd "$(dirname "$0")" && pwd)"
BUILD_CONFIGURATION="${AFTERGLOW_BUILD_CONFIGURATION:-debug}"
BUILD_JOBS="${AFTERGLOW_BUILD_JOBS:-2}"
if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "test.sh currently configures the macOS Metal test bundle. Linux tests require a separately configured CPU/CUDA environment." >&2
  exit 2
fi
cd "$QUANTIZATION_ROOT"
AFTERGLOW_BUILD_CONFIGURATION="$BUILD_CONFIGURATION" ./build.sh
TEST_BUILD_ARGS=(--configuration "$BUILD_CONFIGURATION" --jobs "$BUILD_JOBS" --build-tests)
if [[ "$BUILD_CONFIGURATION" == "release" ]]; then
  TEST_BUILD_ARGS+=(-Xswiftc -enable-testing)
fi
# Remove the runtime resource from prior test bundles before Swift Build signs
# rebuilt executables. Restore it after building, beside each test executable.
BIN_DIR="$(swift build --configuration "$BUILD_CONFIGURATION" --show-bin-path)"
for TEST_BUNDLE in "$BIN_DIR"/*.xctest; do
  [[ -d "$TEST_BUNDLE/Contents/MacOS" ]] || continue
  rm -f "$TEST_BUNDLE/Contents/MacOS/mlx.metallib"
done
swift build "${TEST_BUILD_ARGS[@]}"
BIN_DIR="$(swift build --configuration "$BUILD_CONFIGURATION" --show-bin-path)"
# Native SwiftPM emits one package bundle; Swift Build emits one per test target.
for TEST_BUNDLE in "$BIN_DIR"/*.xctest; do
  [[ -d "$TEST_BUNDLE/Contents/MacOS" ]] || continue
  cp "$BIN_DIR/mlx.metallib" "$TEST_BUNDLE/Contents/MacOS/mlx.metallib"
done
AFTERGLOW_TEST_EXECUTABLE="$BIN_DIR/midnight-afterglow" swift test --configuration "$BUILD_CONFIGURATION" --skip-build --no-parallel "$@"
