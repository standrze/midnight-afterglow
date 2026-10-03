"""Paired benchmark statistics for Wick's validated synthetic measurements.

Uses the Python standard library only. The caller validates nonempty, finite,
positive measurements and supplies per-pair AB/BA ordering before calling.
"""
# Extracted from Midnight: Scripts/benchmark-campaign.py (Apache-2.0).
# Source snapshot SHA256: 30f16c104e83eb43c927b52e69b56b2c40a1b698c5504d4112337d72370ab889
import random
import statistics


def metric_summary(valid, key, lower_better, manifest, common_reasons):
    baseline = [a["measurement"][key] for a, _ in valid]
    candidate = [b["measurement"][key] for _, b in valid]
    ratios = [a / b if lower_better else b / a for a, b in zip(baseline, candidate)]
    half = len(valid) // 2
    drifts = [abs(statistics.median(values[half:]) / statistics.median(values[:half]) - 1) if half else None
              for values in (baseline, candidate)]
    orders = [[ratios[i] for i, (a, _) in enumerate(valid) if a["baseline_first"] == value] for value in [True, False]]
    order_effect = abs(statistics.median(orders[0]) / statistics.median(orders[1]) - 1) if all(orders) else None
    reasons = list(common_reasons)
    for arm, drift in zip(("baseline", "candidate"), drifts):
        if drift is None or drift > manifest["maximum_drift_fraction"]:
            reasons.append(f"{arm} drift exceeds threshold or is unavailable")
    if order_effect is None or order_effect > manifest["maximum_drift_fraction"]:
        reasons.append("AB/BA order effect exceeds threshold or is unavailable")
    rng = random.Random(manifest["seed"])
    bootstrap = sorted(statistics.median(rng.choices(ratios, k=len(ratios))) for _ in range(2000))
    median = statistics.median(ratios)
    return {"baseline_median": statistics.median(baseline), "candidate_median": statistics.median(candidate),
            "diagnostic_paired_ratios": ratios, "diagnostic_paired_median_ratio": median,
            "accepted_paired_median_ratio": None if reasons else median,
            "paired_ratio_min": min(ratios), "paired_ratio_max": max(ratios),
            "paired_ratio_median_absolute_deviation": statistics.median(abs(r - median) for r in ratios),
            "diagnostic_paired_bootstrap_95_percent_interval": [bootstrap[49], bootstrap[1949]],
            "accepted_paired_bootstrap_95_percent_interval": None if reasons else [bootstrap[49], bootstrap[1949]],
            "baseline_drift_fraction": drifts[0], "candidate_drift_fraction": drifts[1],
            "order_effect_fraction": order_effect, "acceptance_reasons": reasons,
            "ratio_direction": "baseline/candidate (latency)" if lower_better else "candidate/baseline (rate)",
            "interpretation": "Ratio >1 favors candidate. Accepted means eligible measurement, not proven improvement. Bootstrap resamples pairs (2000 draws); few pairs, thermal autocorrelation and multiple comparisons limit inference."}
