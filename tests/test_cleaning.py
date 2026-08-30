import unittest

import numpy as np
import pandas as pd

from src.cleaning_v2 import apply_v2_cleaning


class CleaningV2Tests(unittest.TestCase):
    def test_internal_run_is_linearly_interpolated(self) -> None:
        values = [5, 5, 5, 0, 0, 7, 7, 7]
        weekly = pd.DataFrame(
            {
                "sku": "A",
                "week_start": pd.date_range("2024-01-01", periods=len(values), freq="7D"),
                "sales": values,
            }
        )
        result = apply_v2_cleaning(weekly)
        self.assertEqual(result.summary["corrected_intervals"], 1)
        np.testing.assert_allclose(result.weekly.loc[3:4, "sales_v2"], [5.6666666667, 6.3333333333])

    def test_leading_and_trailing_zero_runs_are_not_corrected(self) -> None:
        values = [0, 0, 5, 5, 5, 7, 7, 7, 0, 0]
        weekly = pd.DataFrame(
            {
                "sku": "B",
                "week_start": pd.date_range("2024-01-01", periods=len(values), freq="7D"),
                "sales": values,
            }
        )
        result = apply_v2_cleaning(weekly)
        self.assertEqual(result.summary["corrected_intervals"], 0)
        np.testing.assert_array_equal(result.weekly["sales_v2"], values)

    def test_calendar_gap_prevents_cross_gap_correction(self) -> None:
        dates = list(pd.date_range("2024-01-01", periods=4, freq="7D")) + list(
            pd.date_range("2024-02-12", periods=4, freq="7D")
        )
        weekly = pd.DataFrame(
            {"sku": "C", "week_start": dates, "sales": [4, 4, 4, 0, 0, 6, 6, 6]}
        )
        result = apply_v2_cleaning(weekly)
        self.assertEqual(result.summary["corrected_intervals"], 0)


if __name__ == "__main__":
    unittest.main()
