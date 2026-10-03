#!/usr/bin/env bash
set -euo pipefail

PACKAGE_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BENCHMARK_BINARY="${1:-$PACKAGE_ROOT/.build/release/wick-metal-quant-bench}"

# These invocations must finish at CLI validation, before any GPU workload.
python3 - "$BENCHMARK_BINARY" <<'PY'
import pathlib
import subprocess
import sys

binary = pathlib.Path(sys.argv[1]).resolve(strict=True)

for arguments in (["--help"], ["-h"], ["--gpt-oss-20b", "--help"]):
    result = subprocess.run([str(binary), *arguments], text=True, capture_output=True, timeout=10)
    assert result.returncode == 0, (arguments, result.returncode, result.stderr)
    assert f"Usage: {binary.name}" in result.stdout, result.stdout
    assert "--gpt-oss-20b" in result.stdout, result.stdout
    assert "workload\tfirst\tsecond" not in result.stdout, result.stdout
    assert "Laguna-shaped MLX Q4R8 quantization benchmark" not in result.stdout, result.stdout

invalid = [
    ["--unknown-option"],
    ["--warmup"],
    ["--warmup", "0"],
    ["--iterations", "not-an-integer"],
    ["--iterations", "2"],
    ["--queue-depth", "1"],
    ["--queue-rounds", "2"],
    ["--warmup", "1", "--warmup", "2"],
    ["--gpt-oss-20b", "--format-ab"],
    ["--gpt-oss-20b", "--gpt-oss-20b"],
    ["--fused-gather-silu-candidate-first"],
    ["--fused-gather-silu-ab", "--fused-gather-silu-output"],
    ["--fused-gather-silu-output", "output.json"],
]
for arguments in invalid:
    result = subprocess.run([str(binary), *arguments], text=True, capture_output=True, timeout=10)
    assert result.returncode == 2, (arguments, result.returncode, result.stdout, result.stderr)
    assert result.stderr.startswith("Error:"), (arguments, result.stderr)
    assert not result.stdout, (arguments, result.stdout)

print(f"Metal quantization CLI checks passed ({3 + len(invalid)} invocations; no GPU benchmarks).")
PY
