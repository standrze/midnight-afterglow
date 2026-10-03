"""CPU-only regressions for synthetic format log pairing and evidence gates."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import statistics
import shutil
import subprocess
import sys
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[2] / "Scripts/analyze-gptoss-metal-formats.py"
spec = importlib.util.spec_from_file_location("analyze_gptoss_formats", SCRIPT)
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


def fixture(first=None, second=None):
    first = [1.0] * 4 if first is None else first
    second = [0.5] * len(first) if second is None else second
    lines = ["GPT-OSS-20B-shaped synthetic Metal format A/B",
             f"warmup=8 iterations=20 queue_depth=16 queue_rounds={len(first)}"]
    for workload, baseline, candidate in sorted(analysis.EXPECTED):
        # Repeated O-projection baselines must stay attached to the right candidate.
        lines.append("\t".join([workload, baseline, candidate, "1", ".5", "2x", "1", ".5", "2x"]))
        for label, values in ((baseline, first), (candidate, second)):
            lines.append(f"rounds {workload} {label} total_ms={json.dumps(values)} execution_ms={json.dumps(values)}")
    return "\n".join(lines) + "\n"


class GPTOSSPrimitiveAnalysisTests(unittest.TestCase):
    def test_pairs_repeated_workload_baselines_with_their_own_candidate(self):
        result = analysis.analyze(fixture())
        self.assertEqual(result["orders"], {"AB": 2, "BA": 2})
        self.assertEqual(len(result["comparisons"]), 5)
        for comparison in result["comparisons"]:
            self.assertEqual(comparison["metrics"]["total_ms"]["accepted_paired_median_ratio"], 2)
        outputs = [c for c in result["comparisons"] if c["workload"] == "gptoss-output-qmv"]
        self.assertEqual({c["candidate"] for c in outputs}, {"mxfp4-g32", "affine-4bit-g128"})

    def test_uses_paired_ratios_not_ratio_of_marginal_medians_or_printed_summary(self):
        first, second = [1, 2, 100, 101], [1, 10, 10, 100]
        metric = analysis.analyze(fixture(first, second))["comparisons"][0]["metrics"]["total_ms"]
        expected = statistics.median(a / b for a, b in zip(first, second))
        self.assertAlmostEqual(metric["diagnostic_paired_median_ratio"], expected)
        self.assertNotAlmostEqual(expected, statistics.median(first) / statistics.median(second))

    def test_incomplete_duplicate_reordered_or_malformed_rounds_are_rejected(self):
        text = fixture()
        lines = text.splitlines()
        corruptions = ["\n".join(lines[:-1]), text + "\n".join(lines[2:5]),
                       text.replace("execution_ms=[0.5, 0.5, 0.5, 0.5]", "execution_ms=[0.5]", 1),
                       text.replace("total_ms=[0.5, 0.5, 0.5, 0.5]", "total_ms=[NaN, 0.5, 0.5, 0.5]", 1),
                       text.replace("total_ms=[0.5, 0.5, 0.5, 0.5]", "total_ms=[0, 0.5, 0.5, 0.5]", 1)]
        swapped = lines.copy()
        swapped[3], swapped[4] = swapped[4], swapped[3]
        corruptions.append("\n".join(swapped))
        for corrupted in corruptions:
            with self.subTest(corrupted=corrupted[:100]), self.assertRaises(ValueError):
                analysis.analyze(corrupted)

    def test_drift_unequal_order_counts_and_known_contamination_withhold_estimates(self):
        results = [analysis.analyze(fixture([1] * 6, [1, 1, 1, 2, 2, 2])),
                   analysis.analyze(fixture([1] * 5)),
                   analysis.analyze(fixture(), known_confounder="concurrent Swift compilation")]
        for result in results:
            for comparison in result["comparisons"]:
                for metric in comparison["metrics"].values():
                    self.assertIsNone(metric["accepted_paired_median_ratio"])
                    self.assertIsNotNone(metric["diagnostic_paired_median_ratio"])
                    self.assertTrue(metric["acceptance_reasons"])

    def test_analyzer_runs_from_an_isolated_checkout_without_midnight(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scripts = root / "Scripts"
            scripts.mkdir()
            isolated_analyzer = scripts / SCRIPT.name
            shutil.copy2(SCRIPT, isolated_analyzer)
            shutil.copy2(SCRIPT.with_name("benchmark-statistics.py"), scripts)
            source, output = root / "input.log", root / "analysis.json"
            source.write_text(fixture())
            result = subprocess.run(
                [sys.executable, str(isolated_analyzer), str(source), "--output", str(output)],
                cwd=root, text=True, capture_output=True, timeout=15,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads(output.read_text())
            self.assertEqual(len(report["comparisons"]), 5)
            self.assertEqual(
                report["comparisons"][0]["metrics"]["total_ms"]["accepted_paired_median_ratio"], 2,
            )
            self.assertIn("statistics_helper_sha256", report["provenance"])

    def test_cli_preserves_input_and_existing_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / "input.log", Path(directory) / "output.json"
            source.write_text(fixture())
            self.assertEqual(analysis.main([str(source), "--output", str(output)]), 0)
            saved = output.read_bytes()
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(analysis.main([str(source), "--output", str(output)]), 2)
                self.assertEqual(analysis.main([str(source), "--output", str(source)]), 2)
            self.assertEqual(output.read_bytes(), saved)
            self.assertEqual(source.read_text(), fixture())


if __name__ == "__main__":
    unittest.main()
