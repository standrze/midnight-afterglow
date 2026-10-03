#!/usr/bin/env python3
"""Analyze paired rounds from the synthetic GPT-OSS Metal format benchmark (CPU only)."""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import sys

HELPER = Path(__file__).resolve().with_name("benchmark-statistics.py")
spec = importlib.util.spec_from_file_location("wick_benchmark_statistics", HELPER)
benchmark_statistics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark_statistics)

EXPECTED = {
    ("gptoss-query-qmv", "affine-4bit-g64", "mxfp4-g32"),
    ("gptoss-output-qmv", "affine-4bit-g64", "mxfp4-g32"),
    ("gptoss-output-qmv", "affine-4bit-g64", "affine-4bit-g128"),
    ("gptoss-separate-gate-up-bias-clamp", "affine-4bit-g64", "mxfp4-g32"),
    ("gptoss-down-bias-weighted-reduction", "affine-4bit-g64", "mxfp4-g32"),
}
SETTINGS = re.compile(r"warmup=(\d+) iterations=(\d+) queue_depth=(\d+) queue_rounds=(\d+)")
ROUNDS = re.compile(r"rounds (\S+) (\S+) total_ms=(\[.*\]) execution_ms=(\[.*\])")


def positive(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0


def analyze(text, maximum_drift_fraction=0.1, seed=20260906, known_confounder=None):
    if not math.isfinite(maximum_drift_fraction) or not 0 <= maximum_drift_fraction < 1:
        raise ValueError("maximum_drift_fraction must be finite and in [0, 1)")
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines or lines[0] != "GPT-OSS-20B-shaped synthetic Metal format A/B":
        raise ValueError("expected GPT-OSS synthetic benchmark header")
    settings = [SETTINGS.fullmatch(line) for line in lines if SETTINGS.fullmatch(line)]
    if len(settings) != 1:
        raise ValueError("expected exactly one settings line")
    warmup, iterations, depth, count = map(int, settings[0].groups())
    if warmup < 1 or iterations < 3 or depth < 2 or count < 3:
        raise ValueError("invalid benchmark settings")
    comparisons, seen = [], set()
    pending = None
    for line in lines:
        if line.startswith("gptoss-"):
            if pending is not None:
                raise ValueError("comparison missing one or both round records")
            columns = line.split("\t")
            key = tuple(columns[:3])
            if len(columns) != 9 or key not in EXPECTED or key in seen:
                raise ValueError("unexpected or duplicate comparison row")
            # The formatted ratio-of-medians columns are deliberately not used.
            seen.add(key)
            pending = {"key": key, "arms": []}
        elif line.startswith("rounds "):
            match = ROUNDS.fullmatch(line)
            if pending is None or match is None:
                raise ValueError("orphaned or malformed round record")
            workload, arm, totals, execution = match.groups()
            key = pending["key"]
            if workload != key[0] or arm != key[len(pending["arms"]) + 1]:
                raise ValueError("round records do not match their ordered comparison arms")
            arrays = {"total_ms": json.loads(totals), "execution_ms": json.loads(execution)}
            if any(not isinstance(values, list) or len(values) != count or not all(map(positive, values))
                   for values in arrays.values()):
                raise ValueError("round counts must match settings and all timings must be finite and positive")
            if any(e > t for e, t in zip(arrays["execution_ms"], arrays["total_ms"])):
                raise ValueError("execution interval cannot exceed its encompassing total interval")
            pending["arms"].append(arrays)
            if len(pending["arms"]) == 2:
                comparisons.append(pending)
                pending = None
    if pending is not None or seen != EXPECTED:
        raise ValueError("incomplete five-comparison report")

    reasons = []
    if count < 4 or count % 2:
        reasons.append("require at least four pairs with equal AB/BA counts")
    if known_confounder:
        reasons.append("known concurrent workload or other confounder: " + known_confounder)
    result = {
        "format": 1, "status": "synthetic_primitive_diagnostics",
        "settings": {"warmup": warmup, "iterations": iterations, "queue_depth": depth, "queue_rounds": count},
        "orders": {"AB": (count + 1) // 2, "BA": count // 2},
        "thresholds": {"maximum_drift_fraction": maximum_drift_fraction, "seed": seed, "bootstrap_draws": 2000},
        "known_confounder": known_confounder, "comparisons": [],
        "limitations": [
            "Random BF16 source weights and synthetic inputs; this does not evaluate a model checkpoint, quantization quality, or tokens/s.",
            "Even-indexed rounds are AB and odd-indexed rounds BA per the current native benchmark implementation; the text log has no per-arm timestamps.",
            "Individual latency ratios are paired by round; reported intervals bootstrap pairs and do not account for temporal autocorrelation or multiple comparisons.",
            "Passing drift/order gates only makes a primitive measurement eligible; it does not prove a model speedup or support changing deployment defaults.",
            "The log lacks executable, Metal-library, hardware-state and source-array hashes; preserve external run provenance. Shape/finite checks are not numerical parity tests.",
        ],
    }
    for comparison in comparisons:
        workload, baseline, candidate = comparison["key"]
        first, second = comparison["arms"]
        paired = [({"measurement": {k: first[k][i] for k in first}, "baseline_first": i % 2 == 0},
                   {"measurement": {k: second[k][i] for k in second}}) for i in range(count)]
        metrics = {key: benchmark_statistics.metric_summary(paired, key, True,
                   {"maximum_drift_fraction": maximum_drift_fraction, "seed": seed}, reasons)
                   for key in first}
        result["comparisons"].append({"workload": workload, "baseline": baseline, "candidate": candidate,
                                     "valid_pairs": count, "metrics": metrics})
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--maximum-drift-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260906)
    parser.add_argument("--known-confounder", help="Record contamination and withhold every accepted estimate")
    args = parser.parse_args(argv)
    try:
        raw = args.report.read_bytes()
        result = analyze(raw.decode("utf-8"), args.maximum_drift_fraction, args.seed, args.known_confounder)
        result["provenance"] = {"native_log": str(args.report.resolve()),
            "native_log_sha256": hashlib.sha256(raw).hexdigest(),
            "analyzer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "statistics_helper_sha256": hashlib.sha256(HELPER.read_bytes()).hexdigest()}
        payload = json.dumps(result, indent=2, allow_nan=False) + "\n"
        if args.output:
            if args.output.resolve() == args.report.resolve():
                raise ValueError("output must not replace input log")
            with args.output.open("x") as stream:
                stream.write(payload)
        else:
            print(payload, end="")
        return 0
    except (OSError, UnicodeError, ValueError, TypeError, KeyError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
