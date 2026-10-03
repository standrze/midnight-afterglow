#!/usr/bin/env bash
set -euo pipefail

PACKAGE_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BENCHMARK_SOURCE="$PACKAGE_ROOT/Benchmarks/MetalQuantization/main.swift"

# Wick owns the benchmark options and workloads that exercise affine Q4 QMV.
# These checks inspect sources without launching a benchmark workload.
grep -Fq 'CommandLine.arguments.contains("--qmv-specialization")' "$BENCHMARK_SOURCE"
grep -Fq 'MLX_METAL_AFFINE_QMV_RESULTS_PER_SIMDGROUP' "$BENCHMARK_SOURCE"
grep -Fq 'dense-lm-head-2048x100352' "$BENCHMARK_SOURCE"
grep -Fq 'gather-gate-up-2048x1024' "$BENCHMARK_SOURCE"
grep -Fq 'gather-down-512x2048' "$BENCHMARK_SOURCE"
grep -Fq 'outputHash &*= 1_099_511_628_211' "$BENCHMARK_SOURCE"

echo "Metal quantization source contract checks passed."
