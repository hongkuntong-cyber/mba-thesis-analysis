import unittest

import numpy as np
import pandas as pd

from src.pxq_probability_v4_1 import (
    _probability_value_gate,
    backward_nonoverlapping_block_events,
    common_probability_sample,
    horizon_probability_from_weekly_rate,
    paired_bootstrap_brier_difference,
    pooled_block_probability,
    probability_metrics,
    reliability_bins,
)


class PxqProbabilityV41Tests(unittest.TestCase):
    def test_weekly_rate_converts_to_at_least_one_event_probability(self) -> None:
        self.assertAlmostEqual(
            horizon_probability_from_weekly_rate(0.10, 4),
            1.0 - 0.9**4,
        )
        self.assertEqual(horizon_probability_from_weekly_rate(0.0, 13), 0.0)
        self.assertEqual(horizon_probability_from_weekly_rate(1.0, 13), 1.0)
        with self.assertRaises(ValueError):
            horizon_probability_from_weekly_rate(1.1, 4)

    def test_blocks_align_backward_and_discard_leading_remainder(self) -> None:
        values = np.asarray([99, 0, 0, 2, 0, 0, 0, 0, 3], dtype=float)
        events = backward_nonoverlapping_block_events(values, 4)
        np.testing.assert_array_equal(events, np.asarray([1, 1]))

    def test_all_zero_blocks_remain_valid_probability_information(self) -> None:
        events = backward_nonoverlapping_block_events(np.zeros(8), 4)
        np.testing.assert_array_equal(events, np.asarray([0, 0]))
        self.assertEqual(
            pooled_block_probability(pd.Series([0]), pd.Series([2])),
            0.0,
        )

    def test_pooled_probability_weights_block_counts_not_sku_averages(self) -> None:
        probability = pooled_block_probability(
            pd.Series([1, 8]),
            pd.Series([2, 8]),
        )
        self.assertEqual(probability, 0.9)

    def test_probability_metrics_preserve_extreme_predictions_for_brier(self) -> None:
        metrics = probability_metrics(
            np.asarray([0, 1]),
            np.asarray([0.0, 1.0]),
            epsilon=1e-15,
        )
        self.assertEqual(metrics["brier_score"], 0.0)
        self.assertAlmostEqual(metrics["calibration_gap"], 0.0)
        self.assertTrue(np.isfinite(metrics["log_loss"]))

    def test_common_sample_requires_all_four_frozen_methods(self) -> None:
        rows = []
        for sku in ["complete", "missing"]:
            methods = [
                "PXQ_independence",
                "SKU_block_frequency",
                "Profile_block_frequency",
                "Overall_block_frequency",
            ]
            if sku == "missing":
                methods = methods[:-1]
            for method in methods:
                rows.append(
                    {
                        "horizon_label": "30_day_proxy",
                        "origin_index": 1,
                        "sku": sku,
                        "method": method,
                    }
                )
        common = common_probability_sample(pd.DataFrame(rows))
        self.assertEqual(set(common["sku"]), {"complete"})
        self.assertEqual(common["method"].nunique(), 4)

    def test_reliability_bins_include_probability_one_in_last_bin(self) -> None:
        frame = pd.DataFrame(
            {
                "horizon_label": ["x"] * 3,
                "horizon_weeks": [4] * 3,
                "method": ["m"] * 3,
                "probability": [0.0, 0.2, 1.0],
                "actual_event": [0, 1, 1],
            }
        )
        result = reliability_bins(frame, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        self.assertEqual(result["n_sku_origins"].sum(), 3)
        self.assertIn(4, set(result["bin_index"]))

    def test_paired_bootstrap_is_reproducible(self) -> None:
        frame = pd.DataFrame(
            [
                {"sku": sku, "method": method, "brier_loss": loss}
                for sku, primary, baseline in [("a", 0.1, 0.2), ("b", 0.2, 0.3)]
                for method, loss in [
                    ("PXQ_independence", primary),
                    ("Overall_block_frequency", baseline),
                ]
            ]
        )
        first = paired_bootstrap_brier_difference(
            frame,
            "PXQ_independence",
            "Overall_block_frequency",
            repetitions=50,
            seed=42,
        )
        second = paired_bootstrap_brier_difference(
            frame,
            "PXQ_independence",
            "Overall_block_frequency",
            repetitions=50,
            seed=42,
        )
        self.assertEqual(first, second)
        self.assertAlmostEqual(first["mean_difference"], -0.1)

    def test_gate_requires_origin_pairing_and_calibration(self) -> None:
        head = pd.DataFrame(
            [
                {
                    "horizon_label": "30_day_proxy",
                    "baseline": "Overall_block_frequency",
                    "pxq_independence_better": origin <= 4,
                }
                for origin in range(1, 7)
            ]
        )
        paired = pd.DataFrame(
            [
                {
                    "horizon_label": "30_day_proxy",
                    "baseline": "Overall_block_frequency",
                    "ci_high": -0.01,
                }
            ]
        )
        summary = pd.DataFrame(
            [
                {
                    "horizon_label": "30_day_proxy",
                    "horizon_weeks": 4,
                    "method": "PXQ_independence",
                    "calibration_gap": 0.04,
                }
            ]
        )
        result = _probability_value_gate(
            head,
            paired,
            summary,
            baseline="Overall_block_frequency",
            minimum_winning_origins=4,
            maximum_absolute_calibration_gap=0.05,
        )
        self.assertTrue(bool(result.loc[0, "historical_probability_value_supported"]))
        summary.loc[0, "calibration_gap"] = 0.051
        failed = _probability_value_gate(
            head,
            paired,
            summary,
            baseline="Overall_block_frequency",
            minimum_winning_origins=4,
            maximum_absolute_calibration_gap=0.05,
        )
        self.assertFalse(bool(failed.loc[0, "historical_probability_value_supported"]))


if __name__ == "__main__":
    unittest.main()
