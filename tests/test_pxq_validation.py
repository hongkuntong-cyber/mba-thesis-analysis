import unittest

import numpy as np
import pandas as pd

from src.evaluation import (
    evaluate_forecast,
    paired_bootstrap_loss_difference,
)
from src.forecasting import forecast_pxq
from src.pxq_validation_v4 import (
    _predictability_gate,
    common_model_sample,
)


class PxqValidationTests(unittest.TestCase):
    def test_pxq_uses_recent_occurrence_and_expanding_positive_size(self) -> None:
        values = np.asarray([8, 0, 4, 0, 0, 12, 0, 0], dtype=float)
        forecast, parameters = forecast_pxq(
            values,
            horizon=4,
            occurrence_lookback_weeks=4,
        )
        self.assertEqual(parameters["pxq_p_hat"], 0.25)
        self.assertEqual(parameters["pxq_q_hat"], 8.0)
        self.assertEqual(parameters["pxq_weekly_rate"], 2.0)
        np.testing.assert_array_equal(forecast, np.repeat(2.0, 4))
        self.assertEqual(float(forecast.sum()), 8.0)

    def test_pxq_fails_closed_for_short_or_all_zero_training(self) -> None:
        with self.assertRaises(ValueError):
            forecast_pxq(np.asarray([1, 0, 1], dtype=float), horizon=4)
        with self.assertRaises(ValueError):
            forecast_pxq(np.zeros(4), horizon=4)

    def test_horizon_total_error_is_separate_from_weekly_path_error(self) -> None:
        metrics = evaluate_forecast(
            np.asarray([0, 4], dtype=float),
            np.asarray([2, 2], dtype=float),
            scale=1.0,
        )
        self.assertEqual(metrics["abs_error_sum"], 4.0)
        self.assertEqual(metrics["horizon_total_abs_error"], 0.0)
        self.assertEqual(metrics["underforecast_units"], 0.0)
        self.assertEqual(metrics["overforecast_units"], 0.0)

    def test_common_sample_requires_every_requested_model_and_valid_mase(self) -> None:
        models = ["PXQ", "MA4_proxy", "Naive", "SES", "ADIDA2"]
        rows = []
        for sku in ["complete", "missing", "zero_scale"]:
            used = models if sku != "missing" else models[:-1]
            for model in used:
                rows.append(
                    {
                        "horizon_label": "30_day_proxy",
                        "origin_index": 1,
                        "sku": sku,
                        "model": model,
                        "mase": np.nan if sku == "zero_scale" else 1.0,
                    }
                )
        frame = pd.DataFrame(rows)
        common = common_model_sample(frame, models)
        common_mase = common_model_sample(frame, models, require_mase=True)
        self.assertEqual(set(common["sku"]), {"complete", "zero_scale"})
        self.assertEqual(set(common_mase["sku"]), {"complete"})
        self.assertTrue(common.groupby("sku")["model"].nunique().eq(5).all())

    def test_empty_paired_comparison_is_reported_not_raised(self) -> None:
        empty = pd.DataFrame(columns=["sku", "model", "mase"])
        result = paired_bootstrap_loss_difference(
            empty,
            "PXQ",
            "Naive",
            loss_column="mase",
            repetitions=10,
            seed=42,
        )
        self.assertEqual(result["n_skus"], 0)
        self.assertTrue(np.isnan(result["mean_difference"]))

    def test_predictability_gate_requires_origin_wins_and_both_negative_cis(self) -> None:
        head = pd.DataFrame(
            [
                {
                    "horizon_label": "30_day_proxy",
                    "horizon_weeks": 4,
                    "scope": "all",
                    "baseline": baseline,
                    "pxq_better": origin <= wins,
                }
                for baseline, wins in [("MA4_proxy", 4), ("Naive", 5)]
                for origin in range(1, 7)
            ]
        )
        paired = pd.DataFrame(
            [
                {
                    "horizon_label": "30_day_proxy",
                    "scope": "all",
                    "baseline": baseline,
                    "loss": "horizon_total_abs_error",
                    "ci_high": ci_high,
                }
                for baseline, ci_high in [("MA4_proxy", -0.1), ("Naive", -0.2)]
            ]
        )
        gate = _predictability_gate(
            head,
            paired,
            required_baselines=["MA4_proxy", "Naive"],
            minimum_winning_origins=4,
        )
        self.assertTrue(bool(gate.loc[0, "pxq_value_supported"]))
        paired.loc[paired["baseline"].eq("Naive"), "ci_high"] = 0.01
        failed = _predictability_gate(
            head,
            paired,
            required_baselines=["MA4_proxy", "Naive"],
            minimum_winning_origins=4,
        )
        self.assertFalse(bool(failed.loc[0, "pxq_value_supported"]))


if __name__ == "__main__":
    unittest.main()
