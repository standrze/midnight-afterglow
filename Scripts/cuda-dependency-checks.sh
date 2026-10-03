#!/usr/bin/env bash

# This file is sourced by build.sh. Keep dependency validation separate from
# compilation so it can be exercised with small, synthetic header trees.

model_runner_header_has_integer_macro() {
  local header="$1"
  local macro="$2"
  local value="$3"

  grep -Eq \
    "^[[:space:]]*#[[:space:]]*define[[:space:]]+${macro}[[:space:]]+[(]?${value}[)]?([uUlL]*)([[:space:]]|$|/)" \
    "$header"
}

model_runner_exact_version_file() {
  local file="$1"
  local expected="$2"
  local value

  value="$(sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//' "$file")"
  [[ "$value" == "$expected" ]]
}

model_runner_cmake_has_cudnn_frontend_version() {
  local file="$1"
  local flattened

  # Remove comments before flattening so a note mentioning 1.16.0 cannot make
  # an older project declaration pass. Accept the two common release metadata
  # forms and require the version token on that declaration.
  flattened="$(sed 's/#.*$//' "$file" | tr '\n' ' ')"
  printf '%s\n' "$flattened" | grep -Eiq \
    'project[[:space:]]*\([[:space:]]*cudnn[_-]?frontend([^)]*)VERSION[[:space:]]+1\.16\.0([[:space:])]|$)|set[[:space:]]*\([[:space:]]*CUDNN_FRONTEND_VERSION[[:space:]]+"?1\.16\.0"?[[:space:]]*\)'
}

model_runner_content_fingerprint() {
  # The inner cksums fingerprint each file. The outer cksum makes their
  # ordered aggregate compact enough for the Swift build-profile marker.
  cksum "$@" | awk '{print $1 "-" $2}' | cksum | awk '{print $1 "-" $2}'
}

model_runner_text_fingerprint() {
  printf '%s' "$1" | cksum | awk '{print $1 "-" $2}'
}

model_runner_path_is_within() {
  local child="${1%/}/"
  local parent="${2%/}/"
  [[ "$child" == "$parent"* ]]
}

model_runner_resolve_cuda_dependencies() {
  local expected_cudnn_frontend_version="1.16.0"
  local expected_cutlass_version="4.3.5"
  local cudnn_version_source=""
  local cudnn_version_evidence_file=""
  local cudnn_version_header=""
  local metadata_file
  local git_tag
  local git_commit
  local cutlass_header
  local -a cudnn_fingerprint_files
  local -a cutlass_fingerprint_files

  if [[ -n "${CUDNN_FRONTEND_INCLUDE_DIR:-}" ]]; then
    CUDNN_FRONTEND_INCLUDE_DIR="${CUDNN_FRONTEND_INCLUDE_DIR%/}"
    CUDNN_FRONTEND_ROOT="${CUDNN_FRONTEND_ROOT:-$(dirname "$CUDNN_FRONTEND_INCLUDE_DIR")}"
  else
    CUDNN_FRONTEND_ROOT="${CUDNN_FRONTEND_ROOT:-${HOME}/.local/opt/cudnn-frontend-v1.16.0}"
    CUDNN_FRONTEND_INCLUDE_DIR="${CUDNN_FRONTEND_ROOT%/}/include"
  fi

  if [[ ! -f "$CUDNN_FRONTEND_INCLUDE_DIR/cudnn_frontend.h" ]]; then
    echo "NVIDIA cudnn-frontend v1.16.0 was not found." >&2
    echo "Expected: $CUDNN_FRONTEND_INCLUDE_DIR/cudnn_frontend.h" >&2
    echo "Install v1.16.0 there or set CUDNN_FRONTEND_ROOT/CUDNN_FRONTEND_INCLUDE_DIR." >&2
    return 1
  fi

  CUDNN_FRONTEND_ROOT="$(readlink -f "$CUDNN_FRONTEND_ROOT")"
  CUDNN_FRONTEND_INCLUDE_DIR="$(readlink -f "$CUDNN_FRONTEND_INCLUDE_DIR")"
  if ! model_runner_path_is_within "$CUDNN_FRONTEND_INCLUDE_DIR" "$CUDNN_FRONTEND_ROOT"; then
    echo "CUDNN_FRONTEND_INCLUDE_DIR is not inside CUDNN_FRONTEND_ROOT." >&2
    echo "Refusing to validate metadata from one tree against headers from another." >&2
    return 1
  fi

  # Prefer version declarations shipped alongside the public headers. Support
  # both naming conventions used by frontend releases and the packed numeric
  # macro when present.
  for cudnn_version_header in \
    "$CUDNN_FRONTEND_INCLUDE_DIR/cudnn_frontend_version.h" \
    "$CUDNN_FRONTEND_INCLUDE_DIR/cudnn_frontend.h"
  do
    [[ -f "$cudnn_version_header" ]] || continue
    if model_runner_header_has_integer_macro "$cudnn_version_header" CUDNN_FRONTEND_VERSION_MAJOR 1 \
      && model_runner_header_has_integer_macro "$cudnn_version_header" CUDNN_FRONTEND_VERSION_MINOR 16 \
      && model_runner_header_has_integer_macro "$cudnn_version_header" CUDNN_FRONTEND_VERSION_PATCH 0
    then
      cudnn_version_source="header-macros:CUDNN_FRONTEND_VERSION_MAJOR/MINOR/PATCH"
      cudnn_version_evidence_file="$cudnn_version_header"
      break
    fi
    if model_runner_header_has_integer_macro "$cudnn_version_header" CUDNN_FRONTEND_MAJOR_VERSION 1 \
      && model_runner_header_has_integer_macro "$cudnn_version_header" CUDNN_FRONTEND_MINOR_VERSION 16 \
      && model_runner_header_has_integer_macro "$cudnn_version_header" CUDNN_FRONTEND_PATCH_VERSION 0
    then
      cudnn_version_source="header-macros:CUDNN_FRONTEND_MAJOR/MINOR/PATCH_VERSION"
      cudnn_version_evidence_file="$cudnn_version_header"
      break
    fi
    if model_runner_header_has_integer_macro "$cudnn_version_header" CUDNN_FRONTEND_VERSION 11600; then
      cudnn_version_source="header-macro:CUDNN_FRONTEND_VERSION=11600"
      cudnn_version_evidence_file="$cudnn_version_header"
      break
    fi
  done

  # Source archives do not always ship a generated version header. Accept only
  # exact, release-owned metadata or an exact git tag in that case. A versioned
  # directory name by itself is deliberately not evidence.
  if [[ -z "$cudnn_version_source" ]]; then
    for metadata_file in \
      "$CUDNN_FRONTEND_ROOT/VERSION" \
      "$CUDNN_FRONTEND_ROOT/VERSION.txt" \
      "$CUDNN_FRONTEND_ROOT/version.txt"
    do
      [[ -f "$metadata_file" ]] || continue
      if model_runner_exact_version_file "$metadata_file" "$expected_cudnn_frontend_version"; then
        cudnn_version_source="metadata:$(basename "$metadata_file")"
        cudnn_version_evidence_file="$metadata_file"
        break
      fi
    done
  fi

  if [[ -z "$cudnn_version_source" && -f "$CUDNN_FRONTEND_ROOT/CMakeLists.txt" ]] \
    && model_runner_cmake_has_cudnn_frontend_version "$CUDNN_FRONTEND_ROOT/CMakeLists.txt"
  then
    cudnn_version_source="metadata:CMakeLists.txt"
    cudnn_version_evidence_file="$CUDNN_FRONTEND_ROOT/CMakeLists.txt"
  fi

  if [[ -z "$cudnn_version_source" && -d "$CUDNN_FRONTEND_ROOT/.git" ]]; then
    git_tag="$(git -C "$CUDNN_FRONTEND_ROOT" describe --tags --exact-match HEAD 2>/dev/null || true)"
    if [[ "$git_tag" == "v1.16.0" || "$git_tag" == "1.16.0" ]]; then
      git_commit="$(git -C "$CUDNN_FRONTEND_ROOT" rev-parse HEAD)"
      cudnn_version_source="git:${git_tag}:${git_commit}"
    fi
  fi

  if [[ -z "$cudnn_version_source" ]]; then
    echo "Could not verify exact cudnn-frontend v1.16.0 at $CUDNN_FRONTEND_ROOT." >&2
    echo "A directory name is not version evidence. Keep the exact git tag, CMake release metadata," >&2
    echo "a VERSION file, or a public cudnn-frontend version header with exact 1.16.0 macros." >&2
    return 1
  fi

  if [[ -n "${CUTLASS_INCLUDE_DIR:-}" ]]; then
    CUTLASS_INCLUDE_DIR="${CUTLASS_INCLUDE_DIR%/}"
    CUTLASS_ROOT="${CUTLASS_ROOT:-$(dirname "$CUTLASS_INCLUDE_DIR")}"
  else
    CUTLASS_ROOT="${CUTLASS_ROOT:-${HOME}/.local/opt/cutlass-v4.3.5}"
    CUTLASS_INCLUDE_DIR="${CUTLASS_ROOT%/}/include"
  fi

  cutlass_fingerprint_files=(
    "$CUTLASS_INCLUDE_DIR/cutlass/cutlass.h"
    "$CUTLASS_INCLUDE_DIR/cutlass/version.h"
    "$CUTLASS_INCLUDE_DIR/cutlass/numeric_conversion.h"
    "$CUTLASS_INCLUDE_DIR/cute/tensor.hpp"
    "$CUTLASS_INCLUDE_DIR/cute/numeric/numeric_types.hpp"
  )
  for cutlass_header in "${cutlass_fingerprint_files[@]}"; do
    if [[ ! -f "$cutlass_header" ]]; then
      echo "NVIDIA CUTLASS v4.3.5 (including CuTe) was not found." >&2
      echo "Expected: $cutlass_header" >&2
      echo "Install v4.3.5 there or set CUTLASS_ROOT/CUTLASS_INCLUDE_DIR." >&2
      return 1
    fi
  done

  CUTLASS_ROOT="$(readlink -f "$CUTLASS_ROOT")"
  CUTLASS_INCLUDE_DIR="$(readlink -f "$CUTLASS_INCLUDE_DIR")"
  if ! model_runner_path_is_within "$CUTLASS_INCLUDE_DIR" "$CUTLASS_ROOT"; then
    echo "CUTLASS_INCLUDE_DIR is not inside CUTLASS_ROOT." >&2
    echo "Refusing to mix a CUTLASS root with headers from another tree." >&2
    return 1
  fi

  CUTLASS_VERSION_HEADER="$CUTLASS_INCLUDE_DIR/cutlass/version.h"
  if ! model_runner_header_has_integer_macro "$CUTLASS_VERSION_HEADER" CUTLASS_MAJOR 4 \
    || ! model_runner_header_has_integer_macro "$CUTLASS_VERSION_HEADER" CUTLASS_MINOR 3 \
    || ! model_runner_header_has_integer_macro "$CUTLASS_VERSION_HEADER" CUTLASS_PATCH 5
  then
    echo "Could not verify exact CUTLASS v4.3.5 at $CUTLASS_ROOT." >&2
    echo "Expected CUTLASS_MAJOR/MINOR/PATCH 4/3/5 in $CUTLASS_VERSION_HEADER." >&2
    return 1
  fi

  cudnn_fingerprint_files=("$CUDNN_FRONTEND_INCLUDE_DIR/cudnn_frontend.h")
  if [[ -n "$cudnn_version_evidence_file" \
    && "$cudnn_version_evidence_file" != "$CUDNN_FRONTEND_INCLUDE_DIR/cudnn_frontend.h" ]]
  then
    cudnn_fingerprint_files+=("$cudnn_version_evidence_file")
  fi

  CUDNN_FRONTEND_VERSION="$expected_cudnn_frontend_version"
  CUDNN_FRONTEND_VERSION_SOURCE="$cudnn_version_source"
  CUDNN_FRONTEND_VERSION_EVIDENCE_FINGERPRINT="$(model_runner_text_fingerprint "$cudnn_version_source")"
  CUDNN_FRONTEND_HEADERS_FINGERPRINT="$(model_runner_content_fingerprint "${cudnn_fingerprint_files[@]}")"
  CUTLASS_VERSION="$expected_cutlass_version"
  CUTLASS_HEADERS_FINGERPRINT="$(model_runner_content_fingerprint "${cutlass_fingerprint_files[@]}")"

  export CUDNN_FRONTEND_ROOT CUDNN_FRONTEND_INCLUDE_DIR
  export CUDNN_FRONTEND_VERSION CUDNN_FRONTEND_VERSION_SOURCE
  export CUDNN_FRONTEND_VERSION_EVIDENCE_FINGERPRINT CUDNN_FRONTEND_HEADERS_FINGERPRINT
  export CUTLASS_ROOT CUTLASS_INCLUDE_DIR CUTLASS_VERSION CUTLASS_VERSION_HEADER
  export CUTLASS_HEADERS_FINGERPRINT
}
