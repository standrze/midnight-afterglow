#!/usr/bin/env bash
set -euo pipefail

AFTERGLOW_ROOT="$(cd "$(dirname "$0")" && pwd)"
PACKAGE_ROOT="$AFTERGLOW_ROOT"
cd "$PACKAGE_ROOT"
BUILD_CONFIGURATION="${AFTERGLOW_BUILD_CONFIGURATION:-debug}"
BUILD_PRODUCT="midnight-afterglow"
BUILD_JOBS="${AFTERGLOW_BUILD_JOBS:-2}"
case "$BUILD_CONFIGURATION" in debug|release) ;; *) echo "AFTERGLOW_BUILD_CONFIGURATION must be debug or release." >&2; exit 2 ;; esac
if ! [[ "$BUILD_JOBS" =~ ^[1-9][0-9]*$ ]]; then
  echo "AFTERGLOW_BUILD_JOBS must be a positive integer." >&2
  exit 2
fi

if ! xcrun -sdk macosx --find metal >/dev/null 2>&1; then
  echo "The Metal compiler is missing. Install it once with:"
  echo "  xcodebuild -downloadComponent MetalToolchain"
  exit 1
fi

"$AFTERGLOW_ROOT/prepare-dependencies.sh"
swift build --configuration "$BUILD_CONFIGURATION" --jobs "$BUILD_JOBS" --product "$BUILD_PRODUCT"

BIN_DIR="$(swift build --configuration "$BUILD_CONFIGURATION" --show-bin-path)"
MLX_SOURCE_ROOT="$PACKAGE_ROOT/.build/checkouts/mlx-swift/Source/Cmlx/mlx"
KERNEL_ROOT="$MLX_SOURCE_ROOT/mlx/backend/metal/kernels"
AIR_DIR="$PACKAGE_ROOT/.build/metal"


mkdir -p "$AIR_DIR"

SOURCES=(
  "$KERNEL_ROOT/steel/attn/kernels/steel_attention.metal"
  "$KERNEL_ROOT/arg_reduce.metal"
  "$KERNEL_ROOT/conv.metal"
  "$KERNEL_ROOT/dot.metal"
  "$KERNEL_ROOT/fence.metal"
  "$KERNEL_ROOT/rms_norm.metal"
  "$KERNEL_ROOT/random.metal"
  "$KERNEL_ROOT/scaled_dot_product_attention.metal"
  "$KERNEL_ROOT/layer_norm.metal"
  "$KERNEL_ROOT/rope.metal"
)

AIR_FILES=()
for source in "${SOURCES[@]}"; do
  name="$(basename "$source" .metal)"
  air_file="$AIR_DIR/$name.air"
  xcrun -sdk macosx metal \
    -std=metal4.0 \
    -Wno-c++20-extensions \
    -I "$MLX_SOURCE_ROOT" \
    -c "$source" \
    -o "$air_file"
  AIR_FILES+=("$air_file")
done

xcrun -sdk macosx metallib "${AIR_FILES[@]}" -o "$BIN_DIR/mlx.metallib"
echo "Built $BIN_DIR/$BUILD_PRODUCT and $BIN_DIR/mlx.metallib"
