"""Optional CPU MLX tests for the bounded research probe (no model files needed)."""
import importlib.util
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("affine_probe", ROOT / "Scripts/probe-laguna-affine-gptq.py")
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


class AffineGPTQProbeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import mlx.core as mx
        except ImportError as error:
            raise unittest.SkipTest(f"CPU MLX runtime unavailable: {error}")
        mx.set_default_device(mx.cpu)
        cls.mx = mx

    def test_existing_output_is_preserved_before_loading_models(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "evidence.json"
            output.write_text("original evidence")
            result = subprocess.run([sys.executable, str(ROOT / "Scripts/probe-laguna-affine-gptq.py"),
                                     "--source", "missing", "--calibration", "missing",
                                     "--development", "missing", "--output", str(output)],
                                    text=True, capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("output already exists", result.stderr)
            self.assertEqual(output.read_text(), "original evidence")

    def test_selective_rows_survive_source_removal(self):
        mx = self.mx
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = struct.pack("<12f", *range(12))
            header = json.dumps({"a": {"dtype": "F32", "shape": [4, 3],
                                      "data_offsets": [0, len(payload)]}}).encode()
            shard = root / "model.safetensors"
            shard.write_bytes(struct.pack("<Q", len(header)) + header + payload)
            (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"a": shard.name}}))
            reader = probe.SelectedReader(root)
            selected = reader.read("a", [3, 1])
            shard.unlink()
            self.assertEqual(selected.tolist(), [[9, 10, 11], [3, 4, 5]])
            self.assertEqual(reader.evidence[0]["shape"], [2, 3])
            mx.eval(selected)

    def test_zero_constant_and_unobserved_inputs_remain_finite(self):
        mx = self.mx
        weights = mx.concatenate([mx.zeros((1, 128)), mx.ones((1, 128)) * 0.25,
                                  mx.sin(mx.arange(128).astype(mx.float32))[None, :]], axis=0).astype(mx.bfloat16)
        factor, _ = probe.inverse_factor(mx.zeros((4, 128)), 0.01)
        q = probe.gptq_style(weights, factor, 64)
        reconstruction = probe.unpack(q, 64)
        self.assertTrue(mx.all(mx.isfinite(reconstruction)).item())
        self.assertTrue(mx.all(reconstruction[0] == 0).item())
        self.assertTrue(mx.all(reconstruction[1] == 0.25).item())
        # Explicit zero scales must choose finite constant reconstruction.
        code = probe.nearest_code(mx.ones((2, 1)), mx.zeros((2, 1)), mx.ones((2, 1)))
        self.assertEqual(code.tolist(), [[0], [0]])
        for damping in [0, -1, float("nan"), float("inf")]:
            with self.assertRaisesRegex(ValueError, "damping"):
                probe.inverse_factor(mx.ones((2, 128)), damping)
        with self.assertRaisesRegex(ValueError, "non-finite"):
            probe.inverse_factor(mx.full((2, 128), float("nan")), 0.01)

    def test_source_offsets_are_checked_before_reading(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"a": "model.safetensors"}}))
            for offsets in [[-4, 4], [4, 0], [0, 12]]:
                header = json.dumps({"a": {"dtype": "F32", "shape": [1, 2], "data_offsets": offsets}}).encode()
                (root / "model.safetensors").write_bytes(struct.pack("<Q", len(header)) + header + b"\0" * 8)
                with self.assertRaisesRegex(ValueError, "offsets"):
                    probe.SelectedReader(root).read("a")

    def test_diagonal_covariance_matches_independent_nearest_grid(self):
        mx = self.mx
        weight = mx.sin(mx.arange(4 * 128).reshape(4, 128).astype(mx.float32) * 0.123).astype(mx.bfloat16)
        for group in [64, 128]:
            native = mx.quantize(weight, group_size=group, bits=4)
            q = probe.gptq_style(weight, mx.eye(128), group)
            self.assertTrue(mx.array_equal(native[1], q[1]).item())
            self.assertTrue(mx.array_equal(native[2], q[2]).item())
            # Explicit nearest-code objective avoids assuming undocumented native tie rounding.
            s = native[1].astype(mx.float32)[..., None]
            b = native[2].astype(mx.float32)[..., None]
            expected = mx.clip(mx.round((weight.reshape(4, -1, group).astype(mx.float32) - b) / s), 0, 15) * s + b
            self.assertTrue(mx.array_equal(probe.unpack(q, group), expected.reshape(4, 128)).item())

    def test_block_error_compensation_matches_unblocked_reference(self):
        mx = self.mx
        # Two blocks and four groups make missing cross-group/block updates observable.
        mx.random.seed(913)
        weight = (mx.random.normal((4, 256)) * 0.07).astype(mx.bfloat16)
        inputs = mx.random.normal((512, 256))
        inputs[:, 1:] += inputs[:, :-1] * 0.65
        factor, _ = probe.inverse_factor(inputs, 0.01)
        actual = probe.gptq_style(weight, factor, 64)
        corrected = weight.astype(mx.float32)
        codes = mx.zeros(weight.shape, dtype=mx.uint32)
        scales, biases = [], []
        for column in range(256):
            if column % 64 == 0:
                _, scale, bias = mx.quantize(corrected[:, column:column + 64].astype(weight.dtype), group_size=64, bits=4)
                scales.append(scale)
                biases.append(bias)
                scale, bias = scale.astype(mx.float32), bias.astype(mx.float32)
            w = corrected[:, column:column + 1]
            code = mx.clip(mx.round((w - bias) / scale), 0, 15).astype(mx.uint32)
            error = (w - (code.astype(mx.float32) * scale + bias)) / factor[column, column]
            codes[:, column:column + 1] = code
            corrected[:, column:] -= error @ factor[column:column + 1, column:]
            mx.eval(codes, corrected)
        expected = probe.packed_codes(codes), mx.concatenate(scales, axis=1), mx.concatenate(biases, axis=1)
        for first, second in zip(actual, expected):
            self.assertTrue(mx.array_equal(first, second).item())


if __name__ == "__main__":
    unittest.main()
