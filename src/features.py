from __future__ import annotations

from itertools import combinations
from typing import Iterable

import numpy as np
import pandas as pd


def _acf1_for_consecutive_weeks(frame: pd.DataFrame, value_column: str) -> tuple[float, bool]:
    ordered = frame.sort_values("week_start")
    values = ordered[value_column].to_numpy(dtype=float)
    dates = pd.to_datetime(ordered["week_start"]).to_numpy()
    if len(values) < 2:
        return 0.0, True
    consecutive = np.diff(dates).astype("timedelta64[D]").astype(int) == 7
    left = values[:-1][consecutive]
    right = values[1:][consecutive]
    if len(left) < 2 or np.std(left) == 0 or np.std(right) == 0:
        return 0.0, True
    return float(np.corrcoef(left, right)[0, 1]), False


def compute_features(weekly: pd.DataFrame, value_column: str = "sales_v2") -> pd.DataFrame:
    required = {"sku", "week_start", value_column}
    missing = required.difference(weekly.columns)
    if missing:
        raise ValueError(f"Missing columns for features: {sorted(missing)}")

    records: list[dict[str, float | int | str | bool]] = []
    for sku, frame in weekly.groupby("sku", sort=True):
        values = frame.sort_values("week_start")[value_column].to_numpy(dtype=float)
        if np.isnan(values).any():
            continue
        positive = values[values > 0]
        n_observed = len(values)
        n_positive = len(positive)
        n_zero = int(np.count_nonzero(values == 0))
        nonzero_mean = float(np.mean(positive)) if n_positive else np.nan
        positive_std = float(np.std(positive, ddof=1)) if n_positive >= 2 else np.nan
        cv2 = (positive_std / nonzero_mean) ** 2 if n_positive >= 2 and nonzero_mean > 0 else np.nan
        adi = n_observed / n_positive if n_positive else np.inf
        acf1, acf1_zero_variance = _acf1_for_consecutive_weeks(frame, value_column)
        records.append(
            {
                "sku": str(sku),
                "n_observed": n_observed,
                "n_positive": n_positive,
                "n_zero": n_zero,
                "total_sales": float(np.sum(values)),
                "mean_sales": float(np.mean(values)),
                "median_sales": float(np.median(values)),
                "std_sales": float(np.std(values, ddof=1)) if n_observed >= 2 else 0.0,
                "nonzero_mean": nonzero_mean,
                "CV2": cv2,
                "ADI": adi,
                "zero_ratio": n_zero / n_observed if n_observed else np.nan,
                "acf1": acf1,
                "acf1_zero_variance": acf1_zero_variance,
            }
        )
    return pd.DataFrame(records)


def enumerate_confirmatory_feature_sets(
    anchors: Iterable[str], optional: Iterable[str]
) -> list[tuple[str, ...]]:
    anchor_tuple = tuple(anchors)
    optional_tuple = tuple(optional)
    output: list[tuple[str, ...]] = []
    for size in range(len(optional_tuple) + 1):
        for subset in combinations(optional_tuple, size):
            if "mean_sales" in subset and "nonzero_mean" in subset:
                continue
            output.append((*anchor_tuple, *subset))
    return output


def verify_feature_identities(features: pd.DataFrame, tolerance: float = 1e-10) -> dict[str, float | int]:
    eligible = features.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["mean_sales", "nonzero_mean", "ADI", "zero_ratio"]
    )
    mean_error = np.abs(
        eligible["mean_sales"] - eligible["nonzero_mean"] / eligible["ADI"]
    )
    adi_error = np.abs(
        eligible["ADI"] - 1.0 / (1.0 - eligible["zero_ratio"])
    )
    return {
        "rows_checked": int(len(eligible)),
        "max_mean_identity_error": float(mean_error.max()) if len(mean_error) else np.nan,
        "max_adi_identity_error": float(adi_error.max()) if len(adi_error) else np.nan,
        "mean_identity_failures": int((mean_error > tolerance).sum()),
        "adi_identity_failures": int((adi_error > tolerance).sum()),
    }
