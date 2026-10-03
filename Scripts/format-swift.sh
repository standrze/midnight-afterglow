#!/bin/sh
set -eu
package_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
exec swift format format --configuration "$package_root/.swift-format" \
    --in-place --recursive "$package_root/Package.swift" \
    "$package_root/Sources" "$package_root/Tests"
