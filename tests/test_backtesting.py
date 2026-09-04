import unittest

import numpy as np
import pandas as pd

from src.backtesting import run_backtest


class BacktestingTests(unittest.TestCase):
    @staticmethod
    def _weekly() -> pd.DataFrame:
        weeks = pd.date_range("2024-01-01", periods=80, freq="7D")
        rows = []
        for sku_idx in range(8):
            for week_idx, week in enumerate(weeks):
                period = sku_idx + 2
                value = float((sku_idx + 1) * (1 + week_idx % 3)) if week_idx % period != 0 else 0.0
                rows.append({"sku": f"S{sku_idx}", "week_start": week, "sales": value})
        return pd.DataFrame(rows)

    def test_future_values_do_not_change_training_fit_or_forecast(self) -> None:
        left = self._weekly()
        origin = pd.Timestamp("2025-05-19")
        right = left.copy()
        right.loc[right["week_start"].ge(origin), "sales"] *= 100.0
        kwargs = dict(
            feature_names=["ADI", "CV2", "nonzero_mean", "acf1"],
            k=2,
            origins=[origin],
            horizon=8,
            minimum_positive_weeks=5,
            cleaning_parameters={},
            calendar_anchor=pd.Timestamp("2024-01-01"),
            adida_aggregation_weeks=2,
            include_pxq=True,
            pxq_lookback_weeks=8,
        )
        left_result = run_backtest(left, **kwargs)
        right_result = run_backtest(right, **kwargs)
        pd.testing.assert_frame_equal(left_result.assignments, right_result.assignments)
        keys = [
            "origin_index",
            "sku",
            "cluster",
            "model",
            "forecast_sum",
            "ses_alpha",
            "adida_ses_alpha",
            "pxq_p_hat",
            "pxq_q_hat",
            "pxq_weekly_rate",
        ]
        pd.testing.assert_frame_equal(
            left_result.predictions[keys].reset_index(drop=True),
            right_result.predictions[keys].reset_index(drop=True),
        )

    def test_raw_v2_rows_remain_aligned(self) -> None:
        result = run_backtest(
            self._weekly(),
            feature_names=["ADI", "CV2", "nonzero_mean", "acf1"],
            k=2,
            origins=[pd.Timestamp("2025-05-19")],
            horizon=8,
            minimum_positive_weeks=5,
            cleaning_parameters={},
            calendar_anchor=pd.Timestamp("2024-01-01"),
            adida_aggregation_weeks=2,
        )
        self.assertGreater(len(result.predictions), 0)
        self.assertEqual(result.predictions.groupby(["origin_index", "sku"])["model"].nunique().min(), 5)


if __name__ == "__main__":
    unittest.main()
