#!/usr/bin/env bash

# Report whether the MLX cross-thread stream compatibility overlay should be
# managed on this host. Linux enables it by default because Swift concurrency
# may resume model work on a different OS thread; upstream thread-affine MLX
# streams otherwise terminate the process with "There is no Stream(...) in
# current thread." Set the variable to exactly `0` only for upstream debugging.
# Darwin remains unmanaged because this overlay targets the pinned Linux/CUDA
# checkout only.
model_runner_mlx_cross_thread_stream_overlay_mode() {
  local host_os="$1"
  local requested="${MODEL_RUNNER_ENABLE_MLX_CROSS_THREAD_STREAM_OVERLAY:-}"

  case "$host_os" in
    Darwin)
      printf '%s\n' "unmanaged"
      ;;
    Linux)
      case "$requested" in
        "")
          printf '%s\n' "on"
          ;;
        1)
          printf '%s\n' "on"
          ;;
        0)
          printf '%s\n' "off"
          ;;
        *)
          echo \
            "MODEL_RUNNER_ENABLE_MLX_CROSS_THREAD_STREAM_OVERLAY must be unset, 1, or 0." \
            >&2
          return 2
          ;;
      esac
      ;;
    *)
      echo "Unsupported host for MLX cross-thread stream overlay: $host_os" >&2
      return 2
      ;;
  esac
}

# Reconcile one optional patch without disturbing any other checkout changes.
# The caller must verify the checkout's exact pinned revision before invoking
# this function. Exactly one direction must pass its dry run: forward means the
# overlay is absent, reverse means this exact overlay is applied. Any partial,
# drifted, or otherwise ambiguous state fails closed.
model_runner_reconcile_optional_dependency_patch() {
  local label="$1"
  local checkout="$2"
  local patch_file="$3"
  local desired_state="$4"
  local can_apply=0
  local can_reverse=0

  case "$desired_state" in
    on|off)
      ;;
    *)
      echo "Invalid desired state for $label overlay: $desired_state" >&2
      return 2
      ;;
  esac

  if git -C "$checkout" apply --check "$patch_file" >/dev/null 2>&1; then
    can_apply=1
  fi
  if git -C "$checkout" apply --reverse --check "$patch_file" >/dev/null 2>&1; then
    can_reverse=1
  fi

  if [[ "$can_apply" == "$can_reverse" ]]; then
    echo "Refusing ambiguous $label overlay state in $checkout" >&2
    echo "Expected exactly one of the forward or reverse patch checks to pass." >&2
    return 1
  fi

  if [[ "$desired_state" == "on" ]]; then
    if [[ "$can_reverse" == "1" ]]; then
      echo "$label overlay already enabled."
      return 0
    fi
    git -C "$checkout" apply "$patch_file"
    echo "Enabled $label overlay."
    return 0
  fi

  if [[ "$can_apply" == "1" ]]; then
    echo "$label overlay disabled and absent."
    return 0
  fi

  git -C "$checkout" apply --reverse "$patch_file"
  echo "Removed disabled $label overlay."
}
