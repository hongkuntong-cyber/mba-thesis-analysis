import unittest

import numpy as np
import pandas as pd

from src.features import (
    approximate_entropy,
    compute_features,
    enumerate_confirmatory_feature_sets,
    normalized_trend_coefficient,
    peak_ratio,
    promotion_exposure_features,
    promotion_week_weight,
    seasonal_lag_strength,
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

    def test_cross_month_promotion_week_uses_daily_average(self) -> None:
        # Monday 29 January 2024 contains three January days (W=0)
        # and four February days (W=1).
        self.assertAlmostEqual(promotion_week_weight("2024-01-29"), 4 / 7)

    def test_promotion_response_normalizes_observed_calendar(self) -> None:
        direct, window_mean, response = promotion_exposure_features(
            ["2024-01-01", "2024-07-01"], [1, 3]
        )
        self.assertAlmostEqual(direct, 2.25)
        self.assertAlmostEqual(window_mean, 1.5)
        self.assertAlmostEqual(response, 1.5)

    def test_business_features_have_frozen_interpretations(self) -> None:
        self.assertAlmostEqual(normalized_trend_coefficient([1, 2, 3, 4]), 1.2)
        self.assertLess(normalized_trend_coefficient([4, 3, 2, 1]), 0)
        self.assertAlmostEqual(peak_ratio([0, 2, 4, 0]), 4 / 3)

    def test_seasonality_requires_two_full_cycles(self) -> None:
        self.assertTrue(np.isnan(seasonal_lag_strength(np.ones(103), period=52)))
        seasonal = np.tile(np.arange(52, dtype=float), 2)
        self.assertGreater(seasonal_lag_strength(seasonal, period=52), 0.99)


if __name__ == "__main__":
    unittest.main()
