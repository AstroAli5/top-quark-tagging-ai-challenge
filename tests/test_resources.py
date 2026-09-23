from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from estimate_resources import estimate, enforce_budget


class ResourceTests(unittest.TestCase):
    def test_planning_grows_and_full_scale_exceeds_current_file_format(self):
        small = estimate(50000); larger = estimate(100000); full = estimate(1211000)
        self.assertGreater(larger['planning_peak_gib'], small['planning_peak_gib'])
        self.assertTrue(larger['mat_v5_array_limit_ok'])
        self.assertFalse(full['mat_v5_array_limit_ok'])
        with self.assertRaisesRegex(ValueError, 'MAT-v5'): enforce_budget(full, None)

    def test_invalid_budgets_fail_before_allocating_data(self):
        for count in [0, -1, 1.5, True]:
            with self.assertRaises(ValueError): estimate(count)
        for budget in [-1, 0, .1, float('nan'), float('inf')]:
            with self.assertRaises(ValueError): enforce_budget(estimate(), budget)
        enforce_budget(estimate(), 16.)


if __name__ == '__main__': unittest.main()
