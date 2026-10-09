"""Focused invariants for the portable article comparison intervals."""

from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from recalculate_source_data_ci import auc_ci, median_ci, median_resamples, seed


class ComparisonIntervalsTest(unittest.TestCase):
    def test_deterministic_key_and_draws(self):
        key = ('Fig5', 'Nanog', 'TSA')
        self.assertEqual(seed(key), seed(key))
        self.assertEqual(median_ci([1, 2, 3], [0, 1, 2], key),
                         median_ci([1, 2, 3], [0, 1, 2], key))
        self.assertNotEqual(seed(key), seed(('Fig5', 'Sox2', 'TSA')))

    def test_even_and_odd_constant_medians(self):
        for values in ([3, 3], [3, 3, 3], [3, 3, 3, 3]):
            draws = median_resamples(values, np.random.default_rng(4), 100)
            np.testing.assert_array_equal(draws, np.full(100, 3.0))
        self.assertEqual(median_ci([3, 3], [1, 1], 'ties'), [2.0, 2.0, 2.0])

    def test_auc_direction_and_ties(self):
        self.assertEqual(auc_ci([1, 1], [1, 1]), [0.5, 0.5, 0.5])
        self.assertEqual(auc_ci([2, 3], [0, 1]), [1.0, 1.0, 1.0])
        self.assertEqual(auc_ci([0, 1], [2, 3]), [0.0, 0.0, 0.0])

    def test_insufficient_observations_fail_explicitly(self):
        with self.assertRaises(ValueError):
            median_resamples([np.nan, 1], np.random.default_rng(4))
        with self.assertRaises(ValueError):
            auc_ci([1], [1, 2])


if __name__ == '__main__':
    unittest.main()
