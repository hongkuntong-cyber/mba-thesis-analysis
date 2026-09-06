import unittest

import numpy as np
import pandas as pd

from src.pxq_two_part_v4_2 import (
    _build_historical_components,
    _common_method_sample,
    add_leave_one_out_references,
    add_shrunk_estimates,
    backward_nonoverlapping_block_totals,
    credibility_conditional_quantity,
    credibility_probability,
    paired_bootstrap_method_difference,
)


class PxqTwoPartV42Tests(unittest.TestCase):
    def test_blocks_use_latest_lookback_and_align_backward(self) -> None:
        values = np.arange(1, 11, dtype=float)
        totals = backward_nonoverlapping_block_totals(values, horizon=4, lookback_weeks=8)
        np.testing.assert_array_equal(totals, np.asarray([18.0, 34.0]))

    def test_blocks_discard_oldest_remainder(self) -> None:
        values = np.asarray([99, 1, 2, 3, 4, 5, 6, 7, 8], dtype=float)
        totals = backward_nonoverlapping_block_totals(values, horizon=4, lookback_weeks=9)
        np.testing.assert_array_equal(totals, np.asarray([10.0, 26.0]))

    def test_credibility_formulas_are_exact(self) -> None:
        probability = credibility_probability(1, 4, 0.5, 4)
        quantity = credibility_conditional_quantity(10.0, 1, 6.0, 4)
        self.assertAlmostEqual(probability, 0.375)
        self.assertAlmostEqual(quantity, 6.8)
        self.assertAlmostEqual(probability * quantity, 2.55)

    def test_zero_own_history_uses_reference_without_zero_probability(self) -> None:
        self.assertAlmostEqual(credibility_probability(0, 0, 0.6, 4), 0.6)
        self.assertAlmostEqual(
            credibility_conditional_quantity(0.0, 0, 8.0, 4), 8.0
        )

    def test_leave_one_out_profile_and_enterprise_references(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "horizon_label": "h",
                    "origin_index": 1,
                    "sku": "a1",
                    "cluster_profile": "A",
                    "complete_history_blocks": 4,
                    "positive_history_blocks": 2,
                    "positive_history_volume": 10.0,
                },
                {
                    "horizon_label": "h",
                    "origin_index": 1,
                    "sku": "a2",
                    "cluster_profile": "A",
                    "complete_history_blocks": 6,
                    "positive_history_blocks": 3,
                    "positive_history_volume": 18.0,
                },
                {
                    "horizon_label": "h",
                    "origin_index": 1,
                    "sku": "b1",
                    "cluster_profile": "B",
                    "complete_history_blocks": 5,
                    "positive_history_blocks": 1,
                    "positive_history_volume": 8.0,
                },
            ]
        )
        result = add_leave_one_out_references(frame, minimum_peer_blocks=4).set_index("sku")
        self.assertAlmostEqual(result.loc["a1", "profile_reference_probability"], 0.5)
        self.assertAlmostEqual(
            result.loc["a1", "profile_reference_conditional_quantity"], 6.0
        )
        self.assertAlmostEqual(
            result.loc["a1", "enterprise_reference_probability"], 4.0 / 11.0
        )
        self.assertAlmostEqual(
            result.loc["a1", "enterprise_reference_conditional_quantity"], 6.5
        )
        self.assertEqual(
            result.loc["b1", "profile_probability_reference_source"],
            "enterprise_fallback",
        )
        self.assertAlmostEqual(result.loc["b1", "profile_reference_probability"], 0.5)

    def test_two_part_identity_is_preserved_after_shrinkage(self) -> None:
        frame = pd.DataFrame(
            {
                "complete_history_blocks": [4],
                "positive_history_blocks": [1],
                "positive_history_volume": [10.0],
                "profile_reference_probability": [0.5],
                "profile_reference_conditional_quantity": [6.0],
                "enterprise_reference_probability": [0.25],
                "enterprise_reference_conditional_quantity": [8.0],
            }
        )
        result = add_shrunk_estimates(
            frame, primary_strength=4, sensitivity_strengths=[1, 8]
        )
        self.assertAlmostEqual(
            result.loc[0, "profile_expected_quantity_l4"],
            result.loc[0, "profile_probability_l4"]
            * result.loc[0, "profile_conditional_quantity_l4"],
        )
        self.assertAlmostEqual(
            result.loc[0, "enterprise_expected_quantity_l4"],
            result.loc[0, "enterprise_probability_l4"]
            * result.loc[0, "enterprise_conditional_quantity_l4"],
        )

    def test_common_sample_requires_exact_method_set(self) -> None:
        rows = []
        for sku, methods in [("complete", ["a", "b"]), ("missing", ["a"])]:
            for method in methods:
                rows.append(
                    {
                        "horizon_label": "h",
                        "origin_index": 1,
                        "sku": sku,
                        "method": method,
                    }
                )
        common = _common_method_sample(pd.DataFrame(rows), ["a", "b"])
        self.assertEqual(set(common["sku"]), {"complete"})
        self.assertEqual(set(common["method"]), {"a", "b"})

    def test_paired_bootstrap_is_reproducible(self) -> None:
        frame = pd.DataFrame(
            [
                {"sku": sku, "method": method, "loss": loss}
                for sku, primary, baseline in [("a", 0.1, 0.2), ("b", 0.2, 0.4)]
                for method, loss in [("primary", primary), ("baseline", baseline)]
            ]
        )
        first = paired_bootstrap_method_difference(
            frame,
            "primary",
            "baseline",
            loss_column="loss",
            repetitions=50,
            seed=42,
        )
        second = paired_bootstrap_method_difference(
            frame,
            "primary",
            "baseline",
            loss_column="loss",
            repetitions=50,
            seed=42,
        )
        self.assertEqual(first, second)
        self.assertAlmostEqual(first["mean_difference"], -0.15)

    def test_future_values_do_not_change_historical_components(self) -> None:
        dates = pd.date_range("2024-01-01", periods=12, freq="7D")
        left = pd.DataFrame(
            {"sku": "x", "week_start": dates, "sales": [1.0] * 8 + [0.0] * 4}
        )
        right = left.copy()
        right.loc[right["week_start"].ge(pd.Timestamp("2024-02-26")), "sales"] = 999.0
        base = pd.DataFrame(
            [
                {
                    "horizon_label": "h",
                    "origin_index": 1,
                    "origin": pd.Timestamp("2024-02-26"),
                    "sku": "x",
                    "cluster": 1,
                    "cluster_profile": "A",
                    "actual_sum": 0.0,
                }
            ]
        )
        kwargs = {
            "v4_base": base,
            "horizons": [{"label": "h", "weeks": 4, "approximate_days": 28, "origins": 1}],
            "cleaning_parameters": {},
            "lookback_weeks": 52,
        }
        left_components, _ = _build_historical_components(left, **kwargs)
        right_components, _ = _build_historical_components(right, **kwargs)
        columns = [
            "complete_history_blocks",
            "positive_history_blocks",
            "positive_history_volume",
            "all_history_volume",
        ]
        pd.testing.assert_frame_equal(left_components[columns], right_components[columns])


if __name__ == "__main__":
    unittest.main()
