import unittest

import numpy as np
import pandas as pd

from src.features import (
    approximate_entropy,
    compute_features,
    enumerate_confirmatory_feature_sets,
    trailing_zero_share,
    verify_feature_identities,
)


class FeatureTests(unittest.TestCase):
    def test_exact_feature_identities(self) -> None:
        weeks = pd.date_range("2024-01-01", periods=8, freq="7D")
        weekly = pd.concat(
            [
                pd.DataFrame({"sku": "A", "week_start": weeks, "sales_v2": [0, 2, 0, 4, 0, 6, 0, 8]}),
                pd.DataFrame({"sku": "B", "week_start": weeks, "sales_v2": [1, 1, 0, 2, 0, 3, 0, 5]}),
            ],
            ignore_index=True,
        )
        checks = verify_feature_identities(compute_features(weekly))
        self.assertEqual(checks["mean_identity_failures"], 0)
        self.assertEqual(checks["adi_identity_failures"], 0)
        self.assertLess(checks["max_mean_identity_error"], 1e-10)
        self.assertLess(checks["max_adi_identity_error"], 1e-10)

    def test_mutually_exclusive_scale_features(self) -> None:
        sets = enumerate_confirmatory_feature_sets(
            ["ADI", "CV2"], ["mean_sales", "std_sales", "nonzero_mean", "acf1"]
        )
        self.assertTrue(all(not ({"mean_sales", "nonzero_mean"} <= set(item)) for item in sets))
        self.assertIn(("ADI", "CV2"), sets)
        self.assertIn(("ADI", "CV2", "nonzero_mean", "acf1"), sets)

    def test_approximate_entropy_is_zero_for_constant_series(self) -> None:
        self.assertEqual(approximate_entropy([2, 2, 2, 2, 2, 2]), 0.0)

    def test_approximate_entropy_is_scale_invariant_under_frozen_tolerance(self) -> None:
        values = [0, 1, 0, 3, 0, 1, 0, 4]
        scaled = [value * 7 for value in values]
        self.assertAlmostEqual(
            approximate_entropy(values), approximate_entropy(scaled), places=12
        )

    def test_trailing_zero_share_uses_only_terminal_run(self) -> None:
        self.assertEqual(trailing_zero_share([0, 2, 0, 0]), 0.5)
        self.assertEqual(trailing_zero_share([0, 2, 0, 3]), 0.0)
        self.assertEqual(trailing_zero_share([0, 0, 0, 0]), 1.0)

    def test_compute_features_includes_registry_admitted_supplementaries(self) -> None:
        weeks = pd.date_range("2024-01-01", periods=8, freq="7D")
        weekly = pd.DataFrame(
            {
                "sku": "A",
                "week_start": weeks,
                "sales_v2": [0, 2, 0, 4, 0, 6, 0, 0],
            }
        )
        row = compute_features(weekly).iloc[0]
        self.assertTrue(np.isfinite(row["approx_entropy"]))
        self.assertEqual(row["trailing_zero_share"], 0.25)


if __name__ == "__main__":
    unittest.main()
