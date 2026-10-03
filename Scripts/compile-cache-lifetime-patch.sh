#!/usr/bin/env bash
# Pinned Darwin MLX 0.32 compiler-cache lifetime fix. Source this helper.
# Preflight every layer before writing any checkout, then apply idempotently.
model_runner_prepare_compile_cache_lifetime() {
  local cache_host="$1"
  [[ "$cache_host" == "Darwin" ]] || return 0
  local cache_patch_root="$2"
  local cache_swift_checkout="$3"
  local cache_repositories=(
    "$cache_swift_checkout/Source/Cmlx/mlx"
    "$cache_swift_checkout/Source/Cmlx/mlx-c"
    "$cache_swift_checkout"
  )
  local cache_revisions=(
    "1f8e74e3f12f31365464a6867c6579f0e9b29d85"
    "c74db5307cc8ce122f48d97ef951b30578674e7f"
    "72f3c3ad8aeee39bfc94f8fbeb446cac89e3a798"
  )
  local cache_names=(mlx mlx-c mlx-swift)
  local cache_states=()
  local cache_index cache_actual cache_patch
  for cache_index in 0 1 2; do
    cache_actual="$(git -C "${cache_repositories[$cache_index]}" rev-parse HEAD)" || return 1
    if [[ "$cache_actual" != "${cache_revisions[$cache_index]}" ]]; then
      echo "Refusing compiler-cache lifetime patch for unexpected ${cache_names[$cache_index]} revision: $cache_actual" >&2
      return 1
    fi
    cache_patch="$cache_patch_root/Patches/${cache_names[$cache_index]}-compile-cache-lifetime.patch"
    if git -C "${cache_repositories[$cache_index]}" apply --reverse --check "$cache_patch" >/dev/null 2>&1; then
      cache_states[$cache_index]=applied
    elif git -C "${cache_repositories[$cache_index]}" apply --check "$cache_patch" >/dev/null 2>&1; then
      cache_states[$cache_index]=pending
    else
      echo "Compiler-cache lifetime patch conflicts with ${cache_names[$cache_index]}; no checkouts were changed." >&2
      return 1
    fi
  done
  for cache_index in 0 1 2; do
    if [[ "${cache_states[$cache_index]}" == "pending" ]]; then
      cache_patch="$cache_patch_root/Patches/${cache_names[$cache_index]}-compile-cache-lifetime.patch"
      git -C "${cache_repositories[$cache_index]}" apply "$cache_patch" || return 1
      echo "${cache_names[$cache_index]} compiler-cache lifetime patch applied."
    else
      echo "${cache_names[$cache_index]} compiler-cache lifetime patch already applied."
    fi
  done
}
