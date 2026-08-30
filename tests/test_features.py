import unittest

import numpy as np
import pandas as pd

from src.features import compute_features, enumerate_confirmatory_feature_sets, verify_feature_identities


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


if __name__ == "__main__":
    unittest.main()
