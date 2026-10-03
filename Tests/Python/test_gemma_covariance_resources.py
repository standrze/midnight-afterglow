import importlib.util
from pathlib import Path
import sys
import unittest

root = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('covariance_resources', root / 'Scripts/gemma_covariance_resources.py')
resources = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = resources
spec.loader.exec_module(resources)


class CovarianceResourceTests(unittest.TestCase):
    def test_wide_down_projection_requires_explicit_budget(self):
        with self.assertRaises(MemoryError):
            resources.preflight(4096, 21504, 128, 8 * 1024**3)
        plan = resources.preflight(4096, 21504, 128, 20 * 1024**3)
        self.assertEqual(plan.covariance_work_bytes, 8 * 21504**2 * 4)
        self.assertEqual(plan.planned_work_bytes, plan.covariance_work_bytes + plan.activation_work_bytes + plan.fitting_work_bytes)

    def test_attention_and_expert_geometry_and_budget_boundary(self):
        for width in (1024, 5376):
            plan = resources.preflight(4096, width, 128, 8 * 1024**3)
            self.assertLess(plan.planned_work_bytes, plan.budget_bytes)
            resources.preflight(4096, width, 128, plan.planned_work_bytes)
            with self.assertRaises(MemoryError):
                resources.preflight(4096, width, 128, plan.planned_work_bytes - 1)

    def test_float32_accounting_and_rejected_inputs(self):
        a = resources.preflight(4096, 1024, 128, 8 * 1024**3)
        b = resources.preflight(4096, 1024, 128, 8 * 1024**3, metadata_bytes=4)
        self.assertEqual(b.planned_work_bytes - a.planned_work_bytes, (4096 + 128) * 1024 * 2)
        for bad in (0, -1, 1.5, True):
            with self.assertRaises(ValueError):
                resources.preflight(bad, 1024, 128, 8 * 1024**3)
        with self.assertRaises(ValueError):
            resources.preflight(4096, 1025, 128, 8 * 1024**3)
        with self.assertRaises(ValueError):
            resources.preflight(4096, 1024, 128, 8 * 1024**3, metadata_bytes=1)


if __name__ == '__main__':
    unittest.main()
