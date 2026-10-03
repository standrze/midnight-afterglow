#!/usr/bin/env bash

# Bind every CUDA build step to the nvcc shipped by one resolved toolkit.
# A PATH-provided compiler is deliberately ignored so an older distro nvcc
# cannot be combined with headers and libraries from /usr/local/cuda.
model_runner_resolve_cuda_toolkit() {
  local toolkit_entry="$1"
  local toolkit_path
  local nvcc_candidate
  local nvcc_path
  local selected_nvcc

  if [[ "$toolkit_entry" != /* ]]; then
    echo "CUDA toolkit path must be absolute: $toolkit_entry" >&2
    return 2
  fi
  if [[ ! -d "$toolkit_entry" ]]; then
    echo "CUDA toolkit directory is missing: $toolkit_entry" >&2
    return 1
  fi

  toolkit_path="$(cd "$toolkit_entry" && pwd -P)" || return 1
  nvcc_candidate="$toolkit_path/bin/nvcc"
  if [[ ! -f "$nvcc_candidate" || ! -x "$nvcc_candidate" ]]; then
    echo "CUDA toolkit nvcc is missing or not executable: $nvcc_candidate" >&2
    return 1
  fi
  nvcc_path="$(readlink -f "$nvcc_candidate")" || return 1
  case "$nvcc_path" in
    "$toolkit_path"/*) ;;
    *)
      echo "CUDA toolkit nvcc resolves outside its toolkit: $nvcc_path" >&2
      return 1
      ;;
  esac

  CUDA_TOOLKIT_PATH="$toolkit_path"
  NVCC_PATH="$nvcc_path"
  export PATH="$(dirname "$NVCC_PATH"):$PATH"
  selected_nvcc="$(readlink -f "$(command -v nvcc)")" || return 1
  if [[ "$selected_nvcc" != "$NVCC_PATH" ]]; then
    echo "Failed to select the nvcc from $CUDA_TOOLKIT_PATH" >&2
    echo "Expected: $NVCC_PATH" >&2
    echo "Selected: $selected_nvcc" >&2
    return 1
  fi
}
