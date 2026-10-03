import copy
import importlib.util
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "Scripts/plan-gemma-quantization.py"
spec = importlib.util.spec_from_file_location("gemma_quantization_plan", SCRIPT)
planner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(planner)


def configuration():
    return {
        "model_type": "gemma4", "dtype": "bfloat16",
        "text_config": {"model_type": "gemma4_text", "dtype": "bfloat16",
                        "num_experts": 128, "top_k_experts": 8},
        "quantization": {"group_size": 64, "bits": 4, "mode": "affine"},
    }


def matrix(tensors, module, width, rows=24, experts=None, dtype="BF16"):
    prefix = [rows] if experts is None else [experts, rows]
    for suffix, shape, kind in [
        ("weight", prefix + [width // 8], "U32"),
        ("scales", prefix + [width // 64], dtype),
        ("biases", prefix + [width // 64], dtype),
    ]:
        size = planner.math.prod(shape) * planner.DTYPE_BYTES[kind]
        tensors[f"{module}.{suffix}"] = {"shape": shape, "dtype": kind, "bytes": size}


class GemmaQuantizationPlanTests(unittest.TestCase):
    def test_widths_active_experts_and_tail_interaction(self):
        tensors = {}
        for width in [640, 704, 2112, 2816, 5376, 21504]:
            matrix(tensors, f"model.layers.0.mlp.width{width}", width)
        matrix(tensors, "model.layers.0.experts.gate", 2816, experts=128)
        matrix(tensors, "model.embed_tokens", 640)
        matrix(tensors, "model.layers.0.router.proj", 2816)
        report = planner.analyze(configuration(), tensors)
        by_width = {r["input_width"]: r for r in report["modules"] if ".mlp." in r["module"]}
        for width in [704, 2112]:
            self.assertFalse(by_width[width]["g128_compatible"])
        for width in [640, 2816, 5376]:
            self.assertTrue(by_width[width]["g128_loses_opt_in_g64_tail"])
            self.assertEqual(by_width[width]["retain_tail_plan_group_size"], 64)
        self.assertEqual(by_width[21504]["retain_tail_plan_group_size"], 128)
        expert = next(r for r in report["modules"] if ".experts." in r["module"])
        self.assertEqual(expert["g128_saved_active_bytes"], expert["g128_saved_bytes"] / 16)
        protected = [r for r in report["modules"] if r["protected_embedding_head_or_router"]]
        self.assertEqual(len(protected), 2)
        self.assertTrue(all(r["g128_saved_bytes"] == 0 for r in protected))
        self.assertEqual(report["plans"]["retain_g64_tail_g128"]["changed_modules"], 1)
        self.assertGreater(report["plans"]["all_compatible_g128"]["saved_bytes"],
                           report["plans"]["retain_g64_tail_g128"]["saved_bytes"])
        self.assertEqual(report["warnings"], [])
        self.assertIn("Wick supports opt-in source-based Gemma 3/4 conversion", report["conversion_status"])
        self.assertIn("--gemma-group-policy", report["conversion_status"])
        self.assertIn("unquantized BF16/FP16 source", report["conversion_status"])
        self.assertIn("does not convert weights or establish quality/speed", report["conversion_status"])
        self.assertNotIn("not implemented", report["conversion_status"])

    def test_minimum_bits_foreign_formats_and_alias_disagreement(self):
        tensors = {}
        matrix(tensors, "model.layers.0.mlp.gate_proj", 640)
        for bits in [2, 3]:
            config = configuration()
            config["quantization"]["bits"] = bits
            with self.assertRaisesRegex(ValueError, "minimum bit width"):
                planner.analyze(config, tensors)
        config = configuration()
        config["quantization"]["unused.module"] = {"bits": 3}
        with self.assertRaisesRegex(ValueError, "minimum bit width"):
            planner.analyze(config, tensors)
        config = configuration()
        config["quantization"] = {"config_groups": {"group_0": {"weights": {"num_bits": 4}}}}
        with self.assertRaisesRegex(ValueError, "explicit import"):
            planner.analyze(config, tensors)
        config = configuration()
        config["quantization_config"] = {"bits": 4, "group_size": 128}
        with self.assertRaisesRegex(ValueError, "disagree"):
            planner.analyze(config, tensors)

    def test_dtype_promotion_and_grid_geometry(self):
        tensors = {}
        matrix(tensors, "model.layers.0.mlp.gate_proj", 640, dtype="F16")
        report = planner.analyze(configuration(), tensors)
        self.assertEqual(len(report["warnings"]), 1)
        self.assertIn("BF16 versus scale/bias F16/F16", report["warnings"][0])
        self.assertFalse(report["modules"][0]["activation_dtype_aligned"])
        bad = copy.deepcopy(tensors)
        bad["model.layers.0.mlp.gate_proj.scales"]["shape"][-1] -= 1
        with self.assertRaisesRegex(ValueError, "disagrees"):
            planner.analyze(configuration(), bad)
        bad = copy.deepcopy(tensors)
        del bad["model.layers.0.mlp.gate_proj.weight"]
        matrix(bad, "model.layers.1.mlp.gate_proj", 640)
        with self.assertRaisesRegex(ValueError, "no corresponding"):
            planner.analyze(configuration(), bad)

    def test_header_only_sparse_payload_and_exclusive_report(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tensors = {}
            matrix(tensors, "model.layers.0.mlp.gate_proj", 640)
            header, offset = {}, 0
            for name, tensor in tensors.items():
                header[name] = {"dtype": tensor["dtype"], "shape": tensor["shape"],
                                "data_offsets": [offset, offset + tensor["bytes"]]}
                offset += tensor["bytes"]
            encoded = json.dumps(header).encode()
            path = root / "model.safetensors"
            with path.open("wb") as output:
                output.write(struct.pack("<Q", len(encoded)) + encoded)
                output.truncate(8 + len(encoded) + offset)
            (root / "config.json").write_text(json.dumps(configuration()))
            actual, fingerprints = planner.read_headers(root)
            self.assertEqual(actual, tensors)
            self.assertIn("header_sha256", fingerprints[path.name])
            report_path = root / "plan.json"
            command = [sys.executable, str(SCRIPT), str(root), "--output", str(report_path)]
            subprocess.run(command, check=True, capture_output=True)
            original = report_path.read_bytes()
            self.assertNotEqual(subprocess.run(command, capture_output=True).returncode, 0)
            self.assertEqual(report_path.read_bytes(), original)
            path.write_bytes(struct.pack("<Q", len(encoded)) + encoded)
            with self.assertRaisesRegex(ValueError, "payload extent"):
                planner.read_headers(root)


if __name__ == "__main__":
    unittest.main()
