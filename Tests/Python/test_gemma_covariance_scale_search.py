"""CPU-only checks of the research primitive; no generated-quality claims."""
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("covariance_search", ROOT / "Scripts/gemma_covariance_scale_search.py")
search = importlib.util.module_from_spec(spec)
spec.loader.exec_module(search)


class CovarianceScaleSearchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import mlx.core as mx
        mx.set_default_device(mx.cpu)
        cls.mx = mx

    def weight(self, width=128):
        mx = self.mx
        return (mx.sin(mx.arange(3 * width).reshape(3, width).astype(mx.float32) * .123) * .1).astype(mx.bfloat16)

    def test_native_storage_and_diagonal_search_objective(self):
        mx = self.mx
        for dtype in (mx.bfloat16, mx.float16, mx.float32):
            for group in (64, 128):
                weight = self.weight().astype(dtype)
                q, report = search.quantize(weight, mx.eye(128), group)
                baseline, _ = search.quantize(weight, mx.eye(128), group, factors=(1.0,), refinement_steps=0)
                restored = mx.dequantize(*q, group_size=group, bits=4).astype(mx.float32)
                original = mx.dequantize(*baseline, group_size=group, bits=4).astype(mx.float32)
                # Score in FP32 on the stored grid rather than BF16 matmul rounding.
                s = mx.repeat(q[1].astype(mx.float32), group, axis=1)
                b = mx.repeat(q[2].astype(mx.float32), group, axis=1)
                shifts = mx.arange(0, 32, 4, dtype=mx.uint32)
                codes = ((q[0][..., None] >> shifts) & 15).reshape(weight.shape).astype(mx.float32)
                restored32 = codes * s + b
                bs = mx.repeat(baseline[1].astype(mx.float32), group, axis=1)
                bb = mx.repeat(baseline[2].astype(mx.float32), group, axis=1)
                bc = ((baseline[0][..., None] >> shifts) & 15).reshape(weight.shape).astype(mx.float32)
                self.assertLessEqual(float(mx.sum((restored32 - weight.astype(mx.float32))**2).item()),
                                     float(mx.sum((bc * bs + bb - weight.astype(mx.float32))**2).item()) + 1e-10)
                self.assertTrue(mx.all(mx.isfinite(restored)).item())
                self.assertEqual(q[0].dtype, mx.uint32)
                self.assertEqual(q[1].dtype, dtype)
                self.assertEqual(q[2].dtype, dtype)
                self.assertEqual(report['tensor_bytes'], sum(v.nbytes for v in baseline))
                for decision in report['decisions']:
                    self.assertLessEqual(decision['selected_conditional_error'], decision['native_conditional_error'])
                    self.assertEqual(sum(decision['factor_counts']), weight.shape[0])

    def test_correlated_factor_one_matches_independent_sequential_reference(self):
        mx = self.mx
        mx.random.seed(491)
        x = mx.random.normal((384, 128))
        x[:, 1:] += x[:, :-1] * .65
        factor, _ = search.inverse_factor(x)
        weight = self.weight()
        actual, _ = search.quantize(weight, factor, factors=(1.0,), refinement_steps=0)
        corrected = weight.astype(mx.float32)
        codes = mx.zeros(weight.shape, dtype=mx.uint32)
        scales, biases = [], []
        for column in range(128):
            if column % 64 == 0:
                _, s, b = mx.quantize(corrected[:, column:column + 64].astype(weight.dtype), group_size=64, bits=4)
                scales.append(s)
                biases.append(b)
                s, b = s.astype(mx.float32), b.astype(mx.float32)
            v = corrected[:, column:column + 1]
            c = mx.clip(mx.round((v - b) / s), 0, 15).astype(mx.uint32)
            error = (v - (c.astype(mx.float32) * s + b)) / factor[column, column]
            codes[:, column:column + 1] = c
            corrected[:, column:] -= error @ factor[column:column + 1, column:]
            mx.eval(corrected, codes)
        shifts = mx.arange(0, 32, 4, dtype=mx.uint32)
        expected = mx.sum(codes.reshape(3, -1, 8) << shifts, axis=-1).astype(mx.uint32)
        self.assertTrue(mx.array_equal(actual[0], expected).item())
        self.assertTrue(mx.array_equal(actual[1], mx.concatenate(scales, axis=1)).item())
        self.assertTrue(mx.array_equal(actual[2], mx.concatenate(biases, axis=1)).item())

    def test_correlated_search_keeps_native_option_and_finite_grid(self):
        mx = self.mx
        mx.random.seed(73)
        x = mx.random.normal((320, 128))
        x[:, 1:] += x[:, :-1] * .8
        factor, _ = search.inverse_factor(x)
        q, report = search.quantize(self.weight(), factor)
        self.assertTrue(mx.all(mx.isfinite(mx.dequantize(*q, group_size=64, bits=4))).item())
        for decision in report['decisions']:
            self.assertLessEqual(decision['selected_conditional_error'], decision['native_conditional_error'])

    def test_zero_constant_unobserved_channels(self):
        mx = self.mx
        factor, coverage = search.inverse_factor(mx.zeros((4, 128)))
        self.assertEqual(coverage['unobserved_channels'], 128)
        weight = mx.concatenate([mx.zeros((1, 128)), mx.full((1, 128), .25)], axis=0).astype(mx.bfloat16)
        q, _ = search.quantize(weight, factor)
        restored = mx.dequantize(*q, group_size=64, bits=4)
        self.assertTrue(mx.array_equal(restored, weight).item())

    def test_refinement_guards_complete_rows_with_cross_group_correlation(self):
        mx = self.mx
        mx.random.seed(91)
        x = mx.random.normal((320, 128))
        x[:, 64:] += 1.2 * x[:, :64]
        factor, _ = search.inverse_factor(x)
        weight = self.weight()
        legacy, _ = search.quantize(weight, factor, refinement_steps=0)
        refined, report = search.quantize(weight, factor)
        old = search._row_objective(mx, weight, factor, legacy, 64)
        new = search._row_objective(mx, weight, factor, refined, 64)
        self.assertTrue(mx.all(new <= old).item())
        self.assertGreater(report['refined_rows_accepted'], 0)
        self.assertLess(report['selected_row_objective'], report['baseline_row_objective'])
        self.assertEqual(report['algorithm'], 'greedy-native-affine-q4-covariance-scale-search-v2')
        # Independent covariance quadratic form checks orientation of the guard.
        shifts = mx.arange(0, 32, 4, dtype=mx.uint32)
        c = ((refined[0][..., None] >> shifts) & 15).reshape(weight.shape).astype(mx.float32)
        r = weight.astype(mx.float32) - c * mx.repeat(refined[1].astype(mx.float32), 64, axis=1)
        r -= mx.repeat(refined[2].astype(mx.float32), 64, axis=1)
        h = mx.linalg.inv(factor.T @ factor, stream=mx.cpu)
        reference = mx.sum((r @ h) * r, axis=1, keepdims=True)
        self.assertTrue(mx.allclose(new, reference, rtol=2e-4, atol=1e-8).item())

    def test_bf16_neighbors_and_refinement_bounds(self):
        mx = self.mx
        values = mx.array([0., -0., 1., -1.], dtype=mx.bfloat16)
        up = search._bf16_neighbor(mx, values, 1).view(mx.uint16).tolist()
        down = search._bf16_neighbor(mx, values, -1).view(mx.uint16).tolist()
        self.assertEqual(up, [1, 1, 0x3f81, 0xbf7f])
        self.assertEqual(down, [0x8001, 0x8001, 0x3f7f, 0xbf81])
        for steps in (-1, 9, 1.5, True):
            with self.assertRaises(ValueError):
                search.quantize(self.weight(), mx.eye(128), refinement_steps=steps)

    def test_invalid_geometry_inputs_and_factors(self):
        mx = self.mx
        for x in (mx.zeros((0, 128)), mx.ones((128,)), mx.full((4, 128), float('nan'))):
            with self.assertRaises(ValueError):
                search.inverse_factor(x)
        for damping in (0, -1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                search.inverse_factor(mx.ones((4, 128)), damping)
        for factors in ((), (.75, 1), (1, 0), (1, float('nan'))):
            with self.assertRaises(ValueError):
                search.quantize(self.weight(), mx.eye(128), factors=factors)
        for factor in (mx.eye(64), mx.zeros((128, 128)), mx.ones((128, 128))):
            with self.assertRaises(ValueError):
                search.quantize(self.weight(), factor)
        with self.assertRaises(ValueError):
            search.quantize(self.weight(), mx.eye(128), group_size=32)


if __name__ == '__main__':
    unittest.main()
