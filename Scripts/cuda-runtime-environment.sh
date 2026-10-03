#!/usr/bin/env bash

# Resolve the exact CUDA runtime environment used by the Linux MLX runner and
# publish the CUTLASS/CuTe headers that pinned MLX discovers relative to the
# executable while compiling kernels at runtime. This file is both sourceable
# and an executable launcher.

MODEL_RUNNER_CUDA_RUNTIME_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"

# shellcheck source=cuda-toolkit.sh
source "$MODEL_RUNNER_CUDA_RUNTIME_SCRIPT_DIR/cuda-toolkit.sh"
# shellcheck source=cuda-dependency-checks.sh
source "$MODEL_RUNNER_CUDA_RUNTIME_SCRIPT_DIR/cuda-dependency-checks.sh"
# shellcheck source=cuda-host-cxx.sh
source "$MODEL_RUNNER_CUDA_RUNTIME_SCRIPT_DIR/cuda-host-cxx.sh"

model_runner_cuda_runtime_reject_list_unsafe_path() {
  local label="$1"
  local value="$2"

  if [[ "$value" == *:* || "$value" == *$'\n'* || "$value" == *$'\r'* ]]; then
    echo "$label cannot contain a colon or newline: $value" >&2
    return 1
  fi
}

model_runner_cuda_runtime_resolve_directory() {
  local label="$1"
  local candidate="$2"
  local required_parent="$3"
  local resolved

  if [[ ! -d "$candidate" ]]; then
    echo "$label is missing or is not a directory: $candidate" >&2
    return 1
  fi
  resolved="$(cd "$candidate" && pwd -P)" || return 1
  if ! model_runner_path_is_within "$resolved" "$required_parent"; then
    echo "$label resolves outside the selected CUDA toolkit: $resolved" >&2
    return 1
  fi
  model_runner_cuda_runtime_reject_list_unsafe_path "$label" "$resolved" || return 1
  printf '%s\n' "$resolved"
}

model_runner_resolve_cuda_runtime_environment() {
  local toolkit_entry="${1:-/usr/local/cuda}"
  local runtime_bin
  local runtime_include
  local runtime_lib

  if ! command -v sha256sum >/dev/null 2>&1 \
    && ! command -v shasum >/dev/null 2>&1; then
    echo "A SHA-256 utility (sha256sum or shasum) is required." >&2
    return 1
  fi

  model_runner_resolve_cuda_toolkit "$toolkit_entry" || return 1
  model_runner_resolve_cuda_dependencies || return 1
  model_runner_resolve_cuda_host_cxx || return 1

  runtime_bin="$(model_runner_cuda_runtime_resolve_directory \
    "CUDA binary directory" "$CUDA_TOOLKIT_PATH/bin" "$CUDA_TOOLKIT_PATH")" || return 1
  runtime_include="$(model_runner_cuda_runtime_resolve_directory \
    "CUDA include directory" "$CUDA_TOOLKIT_PATH/include" "$CUDA_TOOLKIT_PATH")" || return 1
  runtime_lib="$(model_runner_cuda_runtime_resolve_directory \
    "CUDA library directory" "$CUDA_TOOLKIT_PATH/lib64" "$CUDA_TOOLKIT_PATH")" || return 1

  if [[ "$(readlink -f "$runtime_bin/nvcc")" != "$NVCC_PATH" ]]; then
    echo "CUDA runtime binary directory does not contain the verified nvcc." >&2
    return 1
  fi
  if [[ ! -e "$runtime_lib/libcudart.so" ]]; then
    echo "Verified CUDA runtime library was not found: $runtime_lib/libcudart.so" >&2
    return 1
  fi
  if [[ ! -e "$runtime_lib/libnvrtc.so" ]]; then
    echo "Verified CUDA NVRTC library was not found: $runtime_lib/libnvrtc.so" >&2
    return 1
  fi

  model_runner_cuda_runtime_reject_list_unsafe_path \
    "CUTLASS include directory" "$CUTLASS_INCLUDE_DIR" || return 1
  model_runner_cuda_runtime_reject_list_unsafe_path \
    "cudnn-frontend include directory" "$CUDNN_FRONTEND_INCLUDE_DIR" || return 1
  model_runner_cuda_runtime_reject_list_unsafe_path \
    "CUDA host C++ compiler" "$CUDA_HOST_CXX_PATH" || return 1

  CUDA_RUNTIME_BIN_DIR="$runtime_bin"
  CUDA_RUNTIME_INCLUDE_DIR="$runtime_include"
  CUDA_RUNTIME_LIBRARY_DIR="$runtime_lib"
  CUDA_HOME="$CUDA_TOOLKIT_PATH"
  CUDA_PATH="$CUDA_TOOLKIT_PATH"
  CUDACXX="$NVCC_PATH"
  CUDAHOSTCXX="$CUDA_HOST_CXX_PATH"
  MLX_CUDA_HOST_CXX="$CUDA_HOST_CXX_PATH"

  # Keep CUDA selection deterministic inside transient services. Standard
  # system locations remain available, but an inherited distro nvcc or CUDA
  # library path cannot precede the verified toolkit.
  PATH="$CUDA_RUNTIME_BIN_DIR:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
  LD_LIBRARY_PATH="$CUDA_RUNTIME_LIBRARY_DIR"
  LIBRARY_PATH="$CUDA_RUNTIME_LIBRARY_DIR"
  CPATH="$CUTLASS_INCLUDE_DIR:$CUDNN_FRONTEND_INCLUDE_DIR:$CUDA_RUNTIME_INCLUDE_DIR"
  MLX_CUDA_INCLUDE_PATHS="$CUTLASS_INCLUDE_DIR:$CUDNN_FRONTEND_INCLUDE_DIR"

  export CUDA_RUNTIME_BIN_DIR CUDA_RUNTIME_INCLUDE_DIR CUDA_RUNTIME_LIBRARY_DIR
  export CUDA_HOME CUDA_PATH CUDA_TOOLKIT_PATH NVCC_PATH CUDACXX
  export PATH LD_LIBRARY_PATH LIBRARY_PATH CPATH MLX_CUDA_INCLUDE_PATHS
  export CUDA_HOST_CXX_PATH CUDAHOSTCXX MLX_CUDA_HOST_CXX
  export CUDNN_FRONTEND_ROOT CUDNN_FRONTEND_INCLUDE_DIR
  export CUTLASS_ROOT CUTLASS_INCLUDE_DIR CUTLASS_VERSION
}

model_runner_cuda_runtime_tree_fingerprint() {
  local root="$1"
  local invalid_entry

  if [[ ! -d "$root" || -L "$root" ]]; then
    echo "Runtime header tree is missing, not a directory, or a symlink: $root" >&2
    return 1
  fi
  invalid_entry="$(find "$root" \( -type l -o \( ! -type d ! -type f \) \) -print -quit)"
  if [[ -n "$invalid_entry" ]]; then
    echo "Runtime header tree contains a symlink or special file: $invalid_entry" >&2
    return 1
  fi
  invalid_entry="$(find "$root" -name "*"$'\n'"*" -print -quit)"
  if [[ -n "$invalid_entry" ]]; then
    echo "Runtime header tree contains a newline in a path." >&2
    return 1
  fi

  (
    cd "$root" || exit 1
    LC_ALL=C find . -type f -print \
      | LC_ALL=C sort \
      | while IFS= read -r relative_path; do
          printf '%s  %s\n' \
            "$(model_runner_cuda_runtime_sha256_file "$relative_path")" \
            "$relative_path"
        done
  ) | model_runner_cuda_runtime_sha256_stream
}

model_runner_cuda_runtime_sha256_file() {
  local path="$1"

  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$path" | awk '{print $1}'
  elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$path" | awk '{print $1}'
  else
    echo "A SHA-256 utility (sha256sum or shasum) is required." >&2
    return 1
  fi
}

model_runner_cuda_runtime_sha256_stream() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum | awk '{print $1}'
  elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 | awk '{print $1}'
  else
    echo "A SHA-256 utility (sha256sum or shasum) is required." >&2
    return 1
  fi
}

model_runner_cuda_runtime_validated_headers_sha256() {
  local include_dir="$1"

  {
    printf 'cutlass/cutlass.h %s\n' \
      "$(model_runner_cuda_runtime_sha256_file "$include_dir/cutlass/cutlass.h")"
    printf 'cutlass/version.h %s\n' \
      "$(model_runner_cuda_runtime_sha256_file "$include_dir/cutlass/version.h")"
    printf 'cutlass/numeric_conversion.h %s\n' \
      "$(model_runner_cuda_runtime_sha256_file "$include_dir/cutlass/numeric_conversion.h")"
    printf 'cute/tensor.hpp %s\n' \
      "$(model_runner_cuda_runtime_sha256_file "$include_dir/cute/tensor.hpp")"
    printf 'cute/numeric/numeric_types.hpp %s\n' \
      "$(model_runner_cuda_runtime_sha256_file "$include_dir/cute/numeric/numeric_types.hpp")"
  } | model_runner_cuda_runtime_sha256_stream
}

model_runner_cuda_runtime_manifest_value() {
  local manifest="$1"
  local key="$2"
  local count

  count="$(awk -v prefix="$key=" \
    'index($0, prefix) == 1 { count++ } END { print count + 0 }' "$manifest")"
  if [[ "$count" != "1" ]]; then
    echo "CUDA runtime header manifest must contain exactly one $key field: $manifest" >&2
    return 1
  fi
  awk -v prefix="$key=" \
    'index($0, prefix) == 1 { print substr($0, length(prefix) + 1) }' "$manifest"
}

model_runner_verify_cuda_runtime_header_directory() {
  local include_dir="$1"
  local manifest="$include_dir/.model-runner-cuda-runtime-headers.manifest"
  local manifest_version
  local manifest_cutlass_version
  local manifest_source_include
  local manifest_validated_headers
  local manifest_validated_headers_sha256
  local manifest_cutlass_tree
  local manifest_cute_tree
  local actual_validated_headers
  local actual_validated_headers_sha256
  local actual_cutlass_tree
  local actual_cute_tree
  local source_cutlass_tree
  local source_cute_tree
  local -a fingerprint_files

  if [[ ! -d "$include_dir" || -L "$include_dir" ]]; then
    echo "Managed CUDA runtime include directory is missing or invalid: $include_dir" >&2
    return 1
  fi
  if [[ ! -f "$manifest" || -L "$manifest" ]]; then
    echo "Managed CUDA runtime header manifest is missing or invalid: $manifest" >&2
    return 1
  fi
  if [[ ! -d "$include_dir/cutlass" || -L "$include_dir/cutlass" \
    || ! -d "$include_dir/cute" || -L "$include_dir/cute" ]]; then
    echo "Managed CUDA runtime headers require regular cutlass/ and cute/ trees." >&2
    return 1
  fi

  manifest_version="$(model_runner_cuda_runtime_manifest_value "$manifest" manifest_version)" || return 1
  manifest_cutlass_version="$(model_runner_cuda_runtime_manifest_value "$manifest" cutlass_version)" || return 1
  manifest_source_include="$(model_runner_cuda_runtime_manifest_value "$manifest" source_include)" || return 1
  manifest_validated_headers="$(model_runner_cuda_runtime_manifest_value "$manifest" validated_headers_fingerprint)" || return 1
  manifest_validated_headers_sha256="$(model_runner_cuda_runtime_manifest_value "$manifest" validated_headers_sha256)" || return 1
  manifest_cutlass_tree="$(model_runner_cuda_runtime_manifest_value "$manifest" cutlass_tree_fingerprint)" || return 1
  manifest_cute_tree="$(model_runner_cuda_runtime_manifest_value "$manifest" cute_tree_fingerprint)" || return 1

  if [[ "$manifest_version" != "1" || "$manifest_cutlass_version" != "4.3.5" ]]; then
    echo "Unsupported CUDA runtime header manifest version or CUTLASS release." >&2
    return 1
  fi
  if [[ "$manifest_source_include" != "$CUTLASS_INCLUDE_DIR" ]]; then
    echo "Published CUDA runtime headers came from a different resolved CUTLASS tree." >&2
    return 1
  fi

  fingerprint_files=(
    "$include_dir/cutlass/cutlass.h"
    "$include_dir/cutlass/version.h"
    "$include_dir/cutlass/numeric_conversion.h"
    "$include_dir/cute/tensor.hpp"
    "$include_dir/cute/numeric/numeric_types.hpp"
  )
  local required_header
  for required_header in "${fingerprint_files[@]}"; do
    if [[ ! -f "$required_header" || -L "$required_header" ]]; then
      echo "Published CUDA runtime header is missing or is a symlink: $required_header" >&2
      return 1
    fi
  done
  if ! model_runner_header_has_integer_macro \
      "$include_dir/cutlass/version.h" CUTLASS_MAJOR 4 \
    || ! model_runner_header_has_integer_macro \
      "$include_dir/cutlass/version.h" CUTLASS_MINOR 3 \
    || ! model_runner_header_has_integer_macro \
      "$include_dir/cutlass/version.h" CUTLASS_PATCH 5
  then
    echo "Published CUDA runtime headers are not exact CUTLASS v4.3.5." >&2
    return 1
  fi

  actual_validated_headers="$(model_runner_content_fingerprint "${fingerprint_files[@]}")" || return 1
  actual_validated_headers_sha256="$(model_runner_cuda_runtime_validated_headers_sha256 "$include_dir")" || return 1
  actual_cutlass_tree="$(model_runner_cuda_runtime_tree_fingerprint "$include_dir/cutlass")" || return 1
  actual_cute_tree="$(model_runner_cuda_runtime_tree_fingerprint "$include_dir/cute")" || return 1
  source_cutlass_tree="$(model_runner_cuda_runtime_tree_fingerprint "$CUTLASS_INCLUDE_DIR/cutlass")" || return 1
  source_cute_tree="$(model_runner_cuda_runtime_tree_fingerprint "$CUTLASS_INCLUDE_DIR/cute")" || return 1

  if [[ "$manifest_validated_headers" != "$CUTLASS_HEADERS_FINGERPRINT" \
    || "$actual_validated_headers" != "$manifest_validated_headers" ]]; then
    echo "Published CUDA runtime header fingerprint does not match the verified CUTLASS source." >&2
    return 1
  fi
  if [[ ! "$manifest_validated_headers_sha256" =~ ^[0-9a-f]{64}$ \
    || "$actual_validated_headers_sha256" != "$manifest_validated_headers_sha256" \
    || "$(model_runner_cuda_runtime_validated_headers_sha256 "$CUTLASS_INCLUDE_DIR")" \
      != "$manifest_validated_headers_sha256" ]]; then
    echo "Published CUDA runtime header SHA-256 does not match the verified CUTLASS source." >&2
    return 1
  fi
  if [[ ! "$manifest_cutlass_tree" =~ ^[0-9a-f]{64}$ \
    || ! "$manifest_cute_tree" =~ ^[0-9a-f]{64}$ ]]; then
    echo "CUDA runtime header manifest contains an invalid tree SHA-256." >&2
    return 1
  fi
  if [[ "$manifest_cutlass_tree" != "$source_cutlass_tree" \
    || "$actual_cutlass_tree" != "$manifest_cutlass_tree" ]]; then
    echo "Published CUTLASS runtime tree is missing files or has drifted." >&2
    return 1
  fi
  if [[ "$manifest_cute_tree" != "$source_cute_tree" \
    || "$actual_cute_tree" != "$manifest_cute_tree" ]]; then
    echo "Published CuTe runtime tree is missing files or has drifted." >&2
    return 1
  fi
}

model_runner_verify_cuda_runtime_headers() {
  local package_root="$1"

  package_root="$(cd "$package_root" && pwd -P)" || return 1
  model_runner_verify_cuda_runtime_header_directory "$package_root/include"
}

model_runner_resolve_cuda_runtime_runner_include() {
  local package_root="$1"
  local runner_path="$2"
  local runner_realpath
  local runner_directory
  local runner_root
  local managed_include
  local runner_include

  package_root="$(cd "$package_root" && pwd -P)" || return 1
  if [[ "$runner_path" != /* || ! -f "$runner_path" || ! -x "$runner_path" ]]; then
    echo "CUDA runtime runner must be an absolute executable file: $runner_path" >&2
    return 1
  fi
  runner_realpath="$(readlink -f "$runner_path")" || return 1
  if [[ ! -f "$runner_realpath" || ! -x "$runner_realpath" || -L "$runner_realpath" ]]; then
    echo "CUDA runtime runner does not resolve to a regular executable: $runner_path" >&2
    return 1
  fi
  runner_directory="$(cd "$(dirname "$runner_realpath")" && pwd -P)" || return 1
  runner_root="$(cd "$runner_directory/.." && pwd -P)" || return 1
  runner_include="$runner_root/include"
  managed_include="$package_root/include"
  if [[ "$runner_include" != "$managed_include" ]]; then
    echo "Runner-relative MLX include directory is outside the managed package root." >&2
    echo "Runner:   $runner_realpath" >&2
    echo "Expected: $managed_include" >&2
    echo "Resolved: $runner_include" >&2
    echo "Use the verified package bin/midnight-rtx4090 publication or a link resolving to it." >&2
    return 1
  fi

  MODEL_RUNNER_CUDA_RUNTIME_RUNNER_REALPATH="$runner_realpath"
  MODEL_RUNNER_CUDA_RUNTIME_RUNNER_INCLUDE_DIR="$runner_include"
  export MODEL_RUNNER_CUDA_RUNTIME_RUNNER_REALPATH
  export MODEL_RUNNER_CUDA_RUNTIME_RUNNER_INCLUDE_DIR
}

model_runner_verify_cuda_runtime_headers_for_runner() {
  local package_root="$1"
  local runner_path="$2"
  local managed_include

  package_root="$(cd "$package_root" && pwd -P)" || return 1
  model_runner_resolve_cuda_runtime_runner_include \
    "$package_root" "$runner_path" || return 1
  model_runner_verify_cuda_runtime_headers "$package_root" || return 1
  managed_include="$(cd "$package_root/include" && pwd -P)" || return 1
  if [[ "$managed_include" != "$MODEL_RUNNER_CUDA_RUNTIME_RUNNER_INCLUDE_DIR" ]]; then
    echo "Managed CUDA runtime include directory resolves away from the runner lookup path." >&2
    return 1
  fi
}

model_runner_publish_cuda_runtime_headers() {
  local package_root="$1"
  local target
  local stage=""
  local cutlass_tree_fingerprint
  local cute_tree_fingerprint
  local validated_headers_sha256

  package_root="$(cd "$package_root" && pwd -P)" || return 1
  target="$package_root/include"
  if [[ -e "$target" || -L "$target" ]]; then
    # Never heal or overwrite an unmanaged/partial/drifted tree. An exact prior
    # publication is a no-op; anything else requires deliberate recovery.
    model_runner_verify_cuda_runtime_header_directory "$target" || {
      echo "Existing CUDA runtime include tree failed verification; refusing to overwrite it." >&2
      return 1
    }
    return 0
  fi

  if [[ ! -d "$CUTLASS_INCLUDE_DIR/cutlass" || -L "$CUTLASS_INCLUDE_DIR/cutlass" \
    || ! -d "$CUTLASS_INCLUDE_DIR/cute" || -L "$CUTLASS_INCLUDE_DIR/cute" ]]; then
    echo "Verified CUTLASS source does not contain regular cutlass/ and cute/ trees." >&2
    return 1
  fi
  cutlass_tree_fingerprint="$(model_runner_cuda_runtime_tree_fingerprint \
    "$CUTLASS_INCLUDE_DIR/cutlass")" || return 1
  cute_tree_fingerprint="$(model_runner_cuda_runtime_tree_fingerprint \
    "$CUTLASS_INCLUDE_DIR/cute")" || return 1
  validated_headers_sha256="$(model_runner_cuda_runtime_validated_headers_sha256 \
    "$CUTLASS_INCLUDE_DIR")" || return 1

  stage="$(mktemp -d "$package_root/.cuda-runtime-include.stage.XXXXXX")" || return 1
  if ! cp -RL "$CUTLASS_INCLUDE_DIR/cutlass" "$stage/cutlass" \
    || ! cp -RL "$CUTLASS_INCLUDE_DIR/cute" "$stage/cute"; then
    rm -rf "$stage"
    return 1
  fi
  if find "$stage" -type l -print -quit | grep -q .; then
    echo "Staged CUDA runtime headers unexpectedly contain a symlink." >&2
    rm -rf "$stage"
    return 1
  fi
  if ! {
    printf 'manifest_version=1\n'
    printf 'cutlass_version=4.3.5\n'
    printf 'source_include=%s\n' "$CUTLASS_INCLUDE_DIR"
    printf 'validated_headers_fingerprint=%s\n' "$CUTLASS_HEADERS_FINGERPRINT"
    printf 'validated_headers_sha256=%s\n' "$validated_headers_sha256"
    printf 'cutlass_tree_fingerprint=%s\n' "$cutlass_tree_fingerprint"
    printf 'cute_tree_fingerprint=%s\n' "$cute_tree_fingerprint"
  } > "$stage/.model-runner-cuda-runtime-headers.manifest"; then
    rm -rf "$stage"
    return 1
  fi
  chmod 0644 "$stage/.model-runner-cuda-runtime-headers.manifest"

  if ! model_runner_verify_cuda_runtime_header_directory "$stage"; then
    rm -rf "$stage"
    return 1
  fi
  if [[ -e "$target" || -L "$target" ]]; then
    echo "CUDA runtime include target appeared during publication; refusing to overwrite it." >&2
    rm -rf "$stage"
    return 1
  fi
  if ! mv "$stage" "$target"; then
    rm -rf "$stage"
    return 1
  fi
  stage=""
  model_runner_verify_cuda_runtime_header_directory "$target"
}

model_runner_cuda_runtime_systemd_setenv_args() {
  MODEL_RUNNER_CUDA_RUNTIME_SYSTEMD_SETENV_ARGS=(
    "--setenv=PATH=$PATH"
    "--setenv=CUDA_HOME=$CUDA_HOME"
    "--setenv=CUDA_PATH=$CUDA_PATH"
    "--setenv=CUDA_TOOLKIT_PATH=$CUDA_TOOLKIT_PATH"
    "--setenv=NVCC_PATH=$NVCC_PATH"
    "--setenv=CUDACXX=$CUDACXX"
    "--setenv=LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
    "--setenv=LIBRARY_PATH=$LIBRARY_PATH"
    "--setenv=CPATH=$CPATH"
    "--setenv=MLX_CUDA_INCLUDE_PATHS=$MLX_CUDA_INCLUDE_PATHS"
    "--setenv=MLX_CUDA_HOST_CXX=$MLX_CUDA_HOST_CXX"
    "--setenv=CUDAHOSTCXX=$CUDAHOSTCXX"
    "--setenv=CUDNN_FRONTEND_ROOT=$CUDNN_FRONTEND_ROOT"
    "--setenv=CUDNN_FRONTEND_INCLUDE_DIR=$CUDNN_FRONTEND_INCLUDE_DIR"
    "--setenv=CUTLASS_ROOT=$CUTLASS_ROOT"
    "--setenv=CUTLASS_INCLUDE_DIR=$CUTLASS_INCLUDE_DIR"
  )
}

model_runner_cuda_runtime_usage() {
  cat <<'USAGE'
Usage:
  Scripts/cuda-runtime-environment.sh [--toolkit ABSOLUTE_DIR] \
    [--package-root ABSOLUTE_DIR] -- COMMAND [ARG ...]
  Scripts/cuda-runtime-environment.sh [--toolkit ABSOLUTE_DIR] \
    [--package-root ABSOLUTE_DIR] --publish-only
  Scripts/cuda-runtime-environment.sh [--toolkit ABSOLUTE_DIR] \
    [--package-root ABSOLUTE_DIR] --verify-only

The launcher resolves exact CUDA, CUTLASS v4.3.5, cudnn-frontend v1.16.0,
and Clang 18-20 inputs, atomically publishes/verifies runner-relative runtime
headers, then execs COMMAND with only the verified CUDA runtime paths first.
USAGE
}

model_runner_cuda_runtime_main() {
  local toolkit="/usr/local/cuda"
  local package_root
  local action="launch"

  package_root="$(cd "$MODEL_RUNNER_CUDA_RUNTIME_SCRIPT_DIR/.." && pwd -P)"
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --toolkit)
        [[ $# -ge 2 ]] || { echo "--toolkit requires a value." >&2; return 2; }
        toolkit="$2"
        shift 2
        ;;
      --package-root)
        [[ $# -ge 2 ]] || { echo "--package-root requires a value." >&2; return 2; }
        package_root="$2"
        shift 2
        ;;
      --publish-only)
        action="publish"
        shift
        break
        ;;
      --verify-only)
        action="verify"
        shift
        break
        ;;
      --)
        shift
        break
        ;;
      --help|-h)
        model_runner_cuda_runtime_usage
        return 0
        ;;
      *)
        echo "Unknown argument: $1" >&2
        model_runner_cuda_runtime_usage >&2
        return 2
        ;;
    esac
  done
  if [[ "$package_root" != /* || ! -d "$package_root" ]]; then
    echo "--package-root must name an existing absolute directory: $package_root" >&2
    return 2
  fi

  model_runner_resolve_cuda_runtime_environment "$toolkit" || return 1
  case "$action" in
    publish)
      [[ $# -eq 0 ]] || { echo "--publish-only accepts no command." >&2; return 2; }
      model_runner_publish_cuda_runtime_headers "$package_root"
      ;;
    verify)
      [[ $# -eq 0 ]] || { echo "--verify-only accepts no command." >&2; return 2; }
      model_runner_verify_cuda_runtime_headers "$package_root"
      ;;
    launch)
      if [[ $# -eq 0 ]]; then
        echo "A command is required after --." >&2
        return 2
      fi
      model_runner_publish_cuda_runtime_headers "$package_root" || return 1
      model_runner_verify_cuda_runtime_headers "$package_root" || return 1
      exec "$@"
      ;;
  esac
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  set -euo pipefail
  model_runner_cuda_runtime_main "$@"
fi
