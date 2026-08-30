from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def evaluate_forecast(actual: np.ndarray, forecast: np.ndarray, scale: float) -> dict[str, float]:
    y = np.asarray(actual, dtype=float)
    y_hat = np.asarray(forecast, dtype=float)
    if y.shape != y_hat.shape:
        raise ValueError("Actual and forecast arrays must have the same shape")
    errors = y_hat - y
    absolute = np.abs(errors)
    actual_sum = float(np.sum(y))
    abs_error_sum = float(np.sum(absolute))
    mase = float(np.mean(absolute) / scale) if np.isfinite(scale) and scale > 0 else np.nan
    return {
        "mase": mase,
        "mae": float(np.mean(absolute)),
        "wape": abs_error_sum / actual_sum if actual_sum > 0 else np.nan,
        "bias_units": float(np.mean(errors)),
        "bias_ratio": float(np.sum(errors) / actual_sum) if actual_sum > 0 else np.nan,
        "actual_sum": actual_sum,
        "forecast_sum": float(np.sum(y_hat)),
        "abs_error_sum": abs_error_sum,
        "signed_error_sum": float(np.sum(errors)),
    }


def summarize_predictions(predictions: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame()

    def summarize(group: pd.DataFrame) -> pd.Series:
        mase = group["mase"].dropna()
        actual_sum = group["actual_sum"].sum()
        return pd.Series(
            {
                "n_sku_origins": int(len(group)),
                "n_valid_mase": int(len(mase)),
                "mean_mase": float(mase.mean()) if len(mase) else np.nan,
                "median_mase": float(mase.median()) if len(mase) else np.nan,
                "mase_lt_1_share": float((mase < 1).mean()) if len(mase) else np.nan,
                "mean_mae": float(group["mae"].mean()),
                "mean_sku_wape": float(group["wape"].mean(skipna=True)),
                "median_sku_wape": float(group["wape"].median(skipna=True)),
                "aggregate_wape": float(group["abs_error_sum"].sum() / actual_sum)
                if actual_sum > 0
                else np.nan,
                "aggregate_bias": float(group["signed_error_sum"].sum() / actual_sum)
                if actual_sum > 0
                else np.nan,
                "actual_volume": float(actual_sum),
            }
        )

    return predictions.groupby(group_columns, dropna=False).apply(
        summarize, include_groups=False
    ).reset_index()


def paired_bootstrap_difference(
    predictions: pd.DataFrame,
    model: str,
    baseline: str,
    *,
    repetitions: int,
    seed: int,
) -> dict[str, Any]:
    pivot = (
        predictions.loc[predictions["model"].isin([model, baseline])]
        .groupby(["sku", "model"], as_index=False)["mase"]
        .mean()
        .pivot(index="sku", columns="model", values="mase")
        .dropna(subset=[model, baseline])
    )
    if pivot.empty:
        return {
            "model": model,
            "baseline": baseline,
            "n_skus": 0,
            "mean_difference": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
        }
    differences = (pivot[model] - pivot[baseline]).to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    boot = np.empty(repetitions, dtype=float)
    for idx in range(repetitions):
        boot[idx] = np.mean(rng.choice(differences, size=len(differences), replace=True))
    return {
        "model": model,
        "baseline": baseline,
        "n_skus": int(len(differences)),
        "mean_difference": float(np.mean(differences)),
        "ci_low": float(np.quantile(boot, 0.025)),
        "ci_high": float(np.quantile(boot, 0.975)),
    }


def model_wins(predictions: pd.DataFrame, loss_column: str = "mae") -> pd.DataFrame:
    """Count per-SKU and per-origin winners, breaking ties by maintenance order."""
    priority = {
        "MA4_proxy": 0,
        "Naive": 1,
        "SES": 2,
        "ADIDA2": 3,
        "SBA": 4,
        "Zero": 5,
    }
    frame = predictions.copy()
    frame["priority"] = frame["model"].map(priority).fillna(99)
    winners = (
        frame.sort_values(["origin_index", "sku", loss_column, "priority", "model"])
        .dropna(subset=[loss_column])
        .drop_duplicates(["origin_index", "sku"], keep="first")
    )
    return (
        winners.groupby(["sku", "model"], as_index=False)
        .agg(winning_origins=("origin_index", "nunique"), mean_winning_loss=(loss_column, "mean"))
        .sort_values(["sku", "winning_origins", "model"], ascending=[True, False, True])
    )
