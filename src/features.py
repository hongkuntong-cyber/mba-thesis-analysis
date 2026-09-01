from __future__ import annotations

from itertools import combinations
from typing import Iterable

import numpy as np
import pandas as pd


PROMOTION_MONTH_WEIGHTS = {
    1: 0.0,
    2: 1.0,
    3: 2.0,
    4: 1.0,
    5: 0.0,
    6: 0.0,
    7: 3.0,
    8: 1.0,
    9: 0.0,
    10: 2.0,
    11: 3.0,
    12: 2.0,
}


def promotion_week_weight(week_start: object) -> float:
    """Average the frozen monthly promotion weights across seven calendar days."""
    start = pd.Timestamp(week_start)
    if pd.isna(start):
        raise ValueError("Promotion week start must be a valid date")
    daily = [
        PROMOTION_MONTH_WEIGHTS[int((start + pd.Timedelta(days=offset)).month)]
        for offset in range(7)
    ]
    return float(np.mean(daily))


def promotion_exposure_features(
    week_starts: Iterable[object], values: Iterable[float]
) -> tuple[float, float, float]:
    """Return direct exposure, observed-window mean weight, and normalized response.

    The direct exposure is sum(W_t * y_t) / sum(y_t). The normalized response
    divides that value by the mean W_t available in the SKU's own observation
    window, preventing unequal calendar coverage from being treated as demand
    response.
    """
    date_array = list(week_starts)
    value_array = np.asarray(list(values), dtype=float)
    if len(date_array) != len(value_array):
        raise ValueError("Promotion dates and demand values must have equal length")
    if len(value_array) == 0:
        return np.nan, np.nan, np.nan
    if not np.isfinite(value_array).all() or (value_array < 0).any():
        raise ValueError("Promotion exposure requires finite nonnegative demand")
    weights = np.asarray([promotion_week_weight(value) for value in date_array])
    total = float(value_array.sum())
    direct = float(np.dot(weights, value_array) / total) if total > 0 else np.nan
    window_mean = float(weights.mean())
    normalized = direct / window_mean if np.isfinite(direct) and window_mean > 0 else np.nan
    return direct, window_mean, float(normalized) if np.isfinite(normalized) else np.nan


def normalized_trend_coefficient(values: Iterable[float]) -> float:
    """OLS change over the observed window divided by mean positive demand."""
    array = np.asarray(list(values), dtype=float)
    if len(array) < 2:
        return np.nan
    if not np.isfinite(array).all() or (array < 0).any():
        raise ValueError("Trend coefficient requires finite nonnegative demand")
    positive = array[array > 0]
    if len(positive) == 0:
        return np.nan
    time = np.linspace(0.0, 1.0, len(array))
    slope = float(np.polyfit(time, array, 1)[0])
    return float(slope / np.mean(positive))


def peak_ratio(values: Iterable[float]) -> float:
    """Largest positive demand event divided by mean positive demand."""
    array = np.asarray(list(values), dtype=float)
    if not np.isfinite(array).all() or (array < 0).any():
        raise ValueError("Peak ratio requires finite nonnegative demand")
    positive = array[array > 0]
    if len(positive) == 0:
        return np.nan
    return float(np.max(positive) / np.mean(positive))


def seasonal_lag_strength(
    values: Iterable[float], *, period: int = 52, minimum_cycles: int = 2
) -> float:
    """Positive lag-period correlation after removing a linear trend.

    At least two complete cycles are required. This is a long-history
    sensitivity feature rather than a full-sample clustering feature.
    """
    array = np.asarray(list(values), dtype=float)
    if period < 2 or minimum_cycles < 2:
        raise ValueError("Seasonal period and minimum cycles are invalid")
    if len(array) < period * minimum_cycles:
        return np.nan
    if not np.isfinite(array).all() or (array < 0).any():
        raise ValueError("Seasonality strength requires finite nonnegative demand")
    time = np.arange(len(array), dtype=float)
    trend = np.polyval(np.polyfit(time, array, 1), time)
    residual = array - trend
    left = residual[:-period]
    right = residual[period:]
    if np.std(left) == 0 or np.std(right) == 0:
        return 0.0
    return float(np.clip(np.corrcoef(left, right)[0, 1], 0.0, 1.0))


def approximate_entropy(
    values: Iterable[float],
    *,
    embedding_dimension: int = 2,
    tolerance_scale: float = 0.5,
    minimum_length: int = 5,
) -> float:
    """Compute approximate entropy with self-matches and Chebyshev distance.

    The defaults reproduce the parameterization in the FIDE author
    implementation: m=2 and r=0.5 times the full-series sample standard
    deviation. The calculation is deterministic and uses only the supplied
    sequence.
    """
    array = np.asarray(list(values), dtype=float)
    if array.ndim != 1:
        raise ValueError("Approximate entropy requires a one-dimensional sequence")
    if len(array) < minimum_length:
        return np.nan
    if not np.isfinite(array).all():
        raise ValueError("Approximate entropy does not accept missing or infinite values")
    if (array < 0).any():
        raise ValueError("Demand values must be nonnegative")
    if embedding_dimension < 1:
        raise ValueError("embedding_dimension must be at least one")
    if tolerance_scale <= 0:
        raise ValueError("tolerance_scale must be positive")

    sample_std = float(np.std(array, ddof=1))
    if sample_std == 0:
        return 0.0
    tolerance = tolerance_scale * sample_std

    def phi(dimension: int) -> float:
        windows = np.lib.stride_tricks.sliding_window_view(array, dimension)
        distances = np.max(
            np.abs(windows[:, np.newaxis, :] - windows[np.newaxis, :, :]),
            axis=2,
        )
        match_share = np.mean(distances <= tolerance, axis=1)
        return float(np.mean(np.log(match_share)))

    return phi(embedding_dimension) - phi(embedding_dimension + 1)


def trailing_zero_share(values: Iterable[float]) -> float:
    """Return the terminal consecutive zero-run length divided by series length."""
    array = np.asarray(list(values), dtype=float)
    if array.ndim != 1:
        raise ValueError("Trailing-zero share requires a one-dimensional sequence")
    if len(array) == 0:
        return np.nan
    if not np.isfinite(array).all():
        raise ValueError("Trailing-zero share does not accept missing or infinite values")
    if (array < 0).any():
        raise ValueError("Demand values must be nonnegative")
    nonzero_positions = np.flatnonzero(array != 0)
    terminal_length = len(array) if len(nonzero_positions) == 0 else len(array) - 1 - int(nonzero_positions[-1])
    return float(terminal_length / len(array))


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
        ordered = frame.sort_values("week_start")
        values = ordered[value_column].to_numpy(dtype=float)
        if np.isnan(values).any():
            continue
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite demand values for SKU {sku}")
        if (values < 0).any():
            raise ValueError(f"Negative demand values for SKU {sku}")
        positive = values[values > 0]
        n_observed = len(values)
        n_positive = len(positive)
        n_zero = int(np.count_nonzero(values == 0))
        nonzero_mean = float(np.mean(positive)) if n_positive else np.nan
        positive_std = float(np.std(positive, ddof=1)) if n_positive >= 2 else np.nan
        cv2 = (positive_std / nonzero_mean) ** 2 if n_positive >= 2 and nonzero_mean > 0 else np.nan
        adi = n_observed / n_positive if n_positive else np.inf
        acf1, acf1_zero_variance = _acf1_for_consecutive_weeks(frame, value_column)
        promo_weight_ratio, promo_window_weight_mean, promo_response_index = (
            promotion_exposure_features(ordered["week_start"], values)
        )
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
                "approx_entropy": approximate_entropy(values),
                "trailing_zero_share": trailing_zero_share(values),
                "acf1": acf1,
                "acf1_zero_variance": acf1_zero_variance,
                "trend_coef": normalized_trend_coefficient(values),
                "seasonality_idx": seasonal_lag_strength(values),
                "peak_ratio": peak_ratio(values),
                "promo_weight_ratio": promo_weight_ratio,
                "promo_window_weight_mean": promo_window_weight_mean,
                "promo_response_index": promo_response_index,
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
