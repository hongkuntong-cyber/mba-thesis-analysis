import unittest

import numpy as np
import pandas as pd

from src.evaluation import evaluate_forecast
from src.forecasting import aggregate_fixed_calendar, forecast_adida, mase_scale


class ForecastingTests(unittest.TestCase):
    def test_adida2_aggregation_and_disaggregation(self) -> None:
        weeks = pd.Series(pd.date_range("2024-01-01", periods=8, freq="7D"))
        values = np.arange(1.0, 9.0)
        aggregated = aggregate_fixed_calendar(
            weeks, values, anchor=pd.Timestamp("2024-01-01"), aggregation_weeks=2
        )
        np.testing.assert_array_equal(aggregated, [3.0, 7.0, 11.0, 15.0])
        forecast, _ = forecast_adida(
            weeks,
            values,
            8,
            anchor=pd.Timestamp("2024-01-01"),
            aggregation_weeks=2,
        )
        self.assertEqual(len(forecast), 8)
        np.testing.assert_allclose(forecast[0::2], forecast[1::2])
        self.assertTrue(np.all(forecast >= 0))

    def test_partial_calendar_blocks_are_excluded(self) -> None:
        weeks = pd.Series(pd.date_range("2024-01-08", periods=4, freq="7D"))
        values = np.ones(4)
        aggregated = aggregate_fixed_calendar(
            weeks, values, anchor=pd.Timestamp("2024-01-01"), aggregation_weeks=2
        )
        np.testing.assert_array_equal(aggregated, [2.0])

    def test_non_divisible_sensitivity_horizon_is_truncated(self) -> None:
        weeks = pd.Series(pd.date_range("2024-01-01", periods=24, freq="7D"))
        forecast, _ = forecast_adida(
            weeks,
            np.ones(24),
            8,
            anchor=pd.Timestamp("2024-01-01"),
            aggregation_weeks=6,
        )
        self.assertEqual(len(forecast), 8)

    def test_zero_mase_denominator_is_missing(self) -> None:
        scale = mase_scale(np.ones(8))
        self.assertEqual(scale, 0.0)
        metrics = evaluate_forecast(np.ones(2), np.ones(2), scale)
        self.assertTrue(np.isnan(metrics["mase"]))
        self.assertEqual(metrics["mae"], 0.0)


if __name__ == "__main__":
    unittest.main()
