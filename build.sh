#!/usr/bin/env bash
set -euo pipefail
AFTERGLOW_ROOT="$(cd "$(dirname "$0")" && pwd)"
if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "The supported Afterglow build currently requires macOS and Metal." >&2
  exit 2
fi
exec "$AFTERGLOW_ROOT/build-metal.sh" "$@"
