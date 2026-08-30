from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar


@dataclass(frozen=True)
class SESFit:
    alpha: float
    level: float
    sse: float


def fit_ses(values: np.ndarray) -> SESFit:
    y = np.asarray(values, dtype=float)
    if len(y) == 0 or not np.isfinite(y).all():
        raise ValueError("SES requires a finite, non-empty series")
    if len(y) == 1:
        return SESFit(alpha=1.0, level=float(y[-1]), sse=0.0)

    def evaluate(alpha: float) -> tuple[float, float]:
        level = float(y[0])
        squared_errors = 0.0
        for actual in y[1:]:
            error = float(actual) - level
            squared_errors += error * error
            level = alpha * float(actual) + (1.0 - alpha) * level
        return squared_errors, level

    result = minimize_scalar(
        lambda alpha: evaluate(float(alpha))[0],
        bounds=(1e-6, 1.0),
        method="bounded",
        options={"xatol": 1e-8},
    )
    alpha = float(result.x)
    sse, level = evaluate(alpha)
    return SESFit(alpha=alpha, level=max(0.0, float(level)), sse=float(sse))


def forecast_naive(values: np.ndarray, horizon: int) -> np.ndarray:
    y = np.asarray(values, dtype=float)
    if len(y) == 0:
        raise ValueError("Naive forecast requires at least one observation")
    return np.repeat(max(0.0, float(y[-1])), horizon)


def forecast_ma4(values: np.ndarray, horizon: int) -> np.ndarray:
    y = np.asarray(values, dtype=float)
    if len(y) < 4:
        raise ValueError("MA4 requires four observations")
    return np.repeat(max(0.0, float(np.mean(y[-4:]))), horizon)


def forecast_ses(values: np.ndarray, horizon: int) -> tuple[np.ndarray, SESFit]:
    fit = fit_ses(np.asarray(values, dtype=float))
    return np.repeat(fit.level, horizon), fit


def aggregate_fixed_calendar(
    weeks: pd.Series,
    values: np.ndarray,
    *,
    anchor: pd.Timestamp,
    aggregation_weeks: int,
) -> np.ndarray:
    frame = pd.DataFrame(
        {"week_start": pd.to_datetime(weeks).to_numpy(), "value": np.asarray(values, dtype=float)}
    ).sort_values("week_start")
    week_index = ((frame["week_start"] - pd.Timestamp(anchor)).dt.days // 7).astype(int)
    frame["block"] = np.floor_divide(week_index, aggregation_weeks)
    aggregates: list[float] = []
    for _, group in frame.groupby("block", sort=True):
        expected = pd.date_range(group["week_start"].min(), periods=aggregation_weeks, freq="7D")
        if len(group) != aggregation_weeks:
            continue
        if not np.array_equal(group["week_start"].to_numpy(), expected.to_numpy()):
            continue
        aggregates.append(float(group["value"].sum()))
    return np.asarray(aggregates, dtype=float)


def forecast_adida(
    weeks: pd.Series,
    values: np.ndarray,
    horizon: int,
    *,
    anchor: pd.Timestamp,
    aggregation_weeks: int = 2,
) -> tuple[np.ndarray, SESFit]:
    aggregated = aggregate_fixed_calendar(
        weeks,
        values,
        anchor=anchor,
        aggregation_weeks=aggregation_weeks,
    )
    if len(aggregated) < 2:
        raise ValueError("ADIDA requires at least two complete aggregate blocks")
    aggregate_horizon = int(np.ceil(horizon / aggregation_weeks))
    aggregate_forecast, fit = forecast_ses(aggregated, aggregate_horizon)
    disaggregated = np.repeat(aggregate_forecast / aggregation_weeks, aggregation_weeks)[:horizon]
    return np.maximum(disaggregated, 0.0), fit


def forecast_core_models(
    train_weeks: pd.Series,
    train_values: np.ndarray,
    *,
    horizon: int,
    calendar_anchor: pd.Timestamp,
    adida_aggregation_weeks: int = 2,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    values = np.asarray(train_values, dtype=float)
    forecasts: dict[str, np.ndarray] = {
        "MA4_proxy": forecast_ma4(values, horizon),
        "Naive": forecast_naive(values, horizon),
        "Zero": np.zeros(horizon, dtype=float),
    }
    ses_forecast, ses_fit = forecast_ses(values, horizon)
    forecasts["SES"] = ses_forecast
    parameters = {"ses_alpha": ses_fit.alpha, "adida_ses_alpha": np.nan}
    try:
        adida_forecast, adida_fit = forecast_adida(
            train_weeks,
            values,
            horizon,
            anchor=calendar_anchor,
            aggregation_weeks=adida_aggregation_weeks,
        )
        forecasts["ADIDA2"] = adida_forecast
        parameters["adida_ses_alpha"] = adida_fit.alpha
    except ValueError:
        # Other core models remain evaluable when a newly observed SKU lacks
        # two complete aggregation blocks. Availability is reported upstream.
        pass
    return forecasts, parameters


def mase_scale(train_values: np.ndarray) -> float:
    values = np.asarray(train_values, dtype=float)
    if len(values) < 2:
        return np.nan
    return float(np.mean(np.abs(np.diff(values))))
