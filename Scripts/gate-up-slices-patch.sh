#!/usr/bin/env bash
# Pinned Mac-only gate/up views avoid MLX's multi-output compiled-graph cycle.
model_runner_prepare_gate_up_slices() {
  local slices_host="$1"
  [[ "$slices_host" == "Darwin" ]] || return 0
  local slices_root="$2"
  local slices_checkout="$3"
  local slices_actual slices_patch
  slices_actual="$(git -C "$slices_checkout" rev-parse HEAD)" || return 1
  if [[ "$slices_actual" != "14414441fa44f45eee35a61e9fa0bab577cf9734" ]]; then
    echo "Refusing gate/up slices patch for unexpected mlx-swift-lm revision: $slices_actual" >&2
    return 1
  fi
  slices_patch="$slices_root/Patches/mlx-swift-lm-gate-up-slices.patch"
  if git -C "$slices_checkout" apply --reverse --check "$slices_patch" >/dev/null 2>&1; then
    echo "mlx-swift-lm gate/up slices patch already applied."
  elif git -C "$slices_checkout" apply --check "$slices_patch" >/dev/null 2>&1; then
    git -C "$slices_checkout" apply "$slices_patch" || return 1
    echo "mlx-swift-lm gate/up slices patch applied."
  else
    echo "Gate/up slices patch conflicts with mlx-swift-lm; checkout was not changed." >&2
    return 1
  fi
}
