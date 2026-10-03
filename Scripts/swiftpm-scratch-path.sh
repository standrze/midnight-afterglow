#!/usr/bin/env bash

# Configure one SwiftPM scratch context for build, package resolution, cache
# markers, and executable lookup. Callers source this file, then invoke the
# function with the package root and host OS.
model_runner_configure_swiftpm_scratch() {
  local package_root="$1"
  local host_os="$2"
  local requested="${WICK_SCRATCH_PATH:-${FACET_SCRATCH_PATH:-${MODEL_RUNNER_SCRATCH_PATH:-}}}"
  local candidate
  local resolved

  MODEL_RUNNER_SWIFTPM_SCRATCH_PATH="$package_root/.build"
  MODEL_RUNNER_SWIFT_BUILD_SCRATCH_ARGS=()
  MODEL_RUNNER_SWIFT_PACKAGE_SCRATCH_ARGS=()

  # This opt-in exists for isolated Linux/CUDA builds. Metal retains its
  # established .build layout even if the variable leaks into a Mac shell.
  if [[ "$host_os" != "Linux" || -z "$requested" ]]; then
    return 0
  fi

  if [[ "$requested" == /* ]]; then
    candidate="$requested"
  else
    candidate="$package_root/$requested"
  fi

  case "${candidate%/}" in
    ""|"$package_root"|"${HOME:-}"|/tmp|/var/tmp)
      echo "WICK_SCRATCH_PATH (or a compatibility fallback) must name a dedicated build directory." >&2
      echo "Refusing broad scratch path: $candidate" >&2
      return 2
      ;;
  esac

  mkdir -p "$candidate"
  resolved="$(cd "$candidate" && pwd -P)"
  case "${resolved%/}" in
    ""|"$package_root"|"${HOME:-}"|/tmp|/var/tmp)
      echo "WICK_SCRATCH_PATH (or a compatibility fallback) resolves to a broad directory: $resolved" >&2
      return 2
      ;;
  esac

  MODEL_RUNNER_SWIFTPM_SCRATCH_PATH="$resolved"
  MODEL_RUNNER_SWIFT_BUILD_SCRATCH_ARGS=(--scratch-path "$resolved")
  MODEL_RUNNER_SWIFT_PACKAGE_SCRATCH_ARGS=(--scratch-path "$resolved")
}
