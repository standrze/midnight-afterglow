#!/usr/bin/env bash

# Resolve one CUDA host C++ compiler for both encuda phases. The pinned MLX
# plugin receives this absolute path through MLX_CUDA_HOST_CXX; CUDAHOSTCXX is
# accepted as the conventional lower-priority alias.
#
# encuda asks nvcc to emit host C++ and SwiftPM subsequently compiles that text
# with the Clang bundled in the Swift toolchain. GCC 13 must not be used for
# the first phase: glibc 2.39 lets GCC 13 emit native _FloatN C++ types, while
# Clang 21 cannot parse those types. A CUDA-supported Clang (< 21) makes glibc
# emit its compatible typedef forms, which remain valid for Swift's Clang 21.
model_runner_resolve_cuda_host_cxx() {
  local configured_name
  local configured_path
  local resolved_path
  local compiler_macros
  local clang_major

  if [[ -n "${MLX_CUDA_HOST_CXX+x}" ]]; then
    configured_name="MLX_CUDA_HOST_CXX"
    configured_path="$MLX_CUDA_HOST_CXX"
  elif [[ -n "${CUDAHOSTCXX+x}" ]]; then
    configured_name="CUDAHOSTCXX"
    configured_path="$CUDAHOSTCXX"
  else
    configured_name="linux-default"
    configured_path="/usr/bin/clang++-18"
  fi

  if [[ "$configured_path" != /* ]]; then
    echo "$configured_name must be an absolute path: $configured_path" >&2
    return 1
  fi
  if [[ ! -f "$configured_path" || ! -x "$configured_path" ]]; then
    echo "$configured_name must name an executable file: $configured_path" >&2
    return 1
  fi

  resolved_path="$(readlink -f "$configured_path")"
  if [[ -z "$resolved_path" || ! -f "$resolved_path" || ! -x "$resolved_path" ]]; then
    echo "Could not resolve an executable CUDA host C++ compiler: $configured_path" >&2
    return 1
  fi

  if ! compiler_macros="$("$resolved_path" -dM -E -x c++ /dev/null 2>/dev/null)"; then
    echo "Could not inspect CUDA host C++ compiler: $resolved_path" >&2
    return 1
  fi
  clang_major="$(
    awk '$2 == "__clang_major__" { print $3; exit }' <<< "$compiler_macros"
  )"
  if [[ ! "$clang_major" =~ ^[0-9]+$ ]]; then
    echo "CUDA host C++ compiler must be Clang 18, 19, or 20: $resolved_path" >&2
    echo "GCC 13 emits glibc _FloatN types that Swift Clang 21 cannot compile." >&2
    return 1
  fi
  if (( clang_major < 18 || clang_major >= 21 )); then
    echo "CUDA host Clang major must be 18, 19, or 20; found $clang_major: $resolved_path" >&2
    return 1
  fi

  CUDA_HOST_CXX_SOURCE="$configured_name"
  CUDA_HOST_CXX_PATH="$resolved_path"
  CUDA_HOST_CXX_FAMILY="clang"
  CUDA_HOST_CXX_MAJOR="$clang_major"
  CUDA_HOST_CXX_VERSION_FINGERPRINT="$("$CUDA_HOST_CXX_PATH" --version 2>&1 | cksum | awk '{print $1 "-" $2}')"
  if [[ -z "$CUDA_HOST_CXX_VERSION_FINGERPRINT" ]]; then
    echo "Could not fingerprint CUDA host C++ compiler: $CUDA_HOST_CXX_PATH" >&2
    return 1
  fi

  # Set the MLX-specific variable even when CUDAHOSTCXX supplied the value.
  # That gives the patched plugin one canonical, resolved input.
  MLX_CUDA_HOST_CXX="$CUDA_HOST_CXX_PATH"
  export MLX_CUDA_HOST_CXX CUDA_HOST_CXX_SOURCE CUDA_HOST_CXX_PATH
  export CUDA_HOST_CXX_FAMILY CUDA_HOST_CXX_MAJOR CUDA_HOST_CXX_VERSION_FINGERPRINT
}
