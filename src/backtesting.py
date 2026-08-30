from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .cleaning_v2 import apply_v2_cleaning
from .clustering import fit_solution
from .evaluation import evaluate_forecast
from .features import compute_features
from .forecasting import forecast_core_models, mase_scale


@dataclass(frozen=True)
class BacktestResult:
    predictions: pd.DataFrame
    assignments: pd.DataFrame
    origin_audits: pd.DataFrame
    origins: list[pd.Timestamp]


def build_origins(weekly: pd.DataFrame, total_weeks: int, horizon: int) -> list[pd.Timestamp]:
    last_week = pd.Timestamp(weekly["week_start"].max())
    first_test_week = last_week - pd.Timedelta(weeks=total_weeks - 1)
    origins = [first_test_week + pd.Timedelta(weeks=horizon * idx) for idx in range(total_weeks // horizon)]
    if origins[-1] + pd.Timedelta(weeks=horizon - 1) != last_week:
        raise ValueError("Backtest windows do not align with the final observed week")
    return origins


def _ending_contiguous_training_segment(
    frame: pd.DataFrame, origin: pd.Timestamp, value_column: str
) -> pd.DataFrame:
    ordered = frame.loc[frame["week_start"] < origin].sort_values("week_start").copy()
    if ordered.empty or ordered.iloc[-1]["week_start"] != origin - pd.Timedelta(weeks=1):
        return ordered.iloc[0:0]
    dates = pd.to_datetime(ordered["week_start"]).reset_index(drop=True)
    breaks = np.where(np.diff(dates.to_numpy()).astype("timedelta64[D]").astype(int) != 7)[0]
    start = int(breaks[-1] + 1) if len(breaks) else 0
    return ordered.iloc[start:].copy()


def run_backtest(
    weekly_raw: pd.DataFrame,
    *,
    feature_names: Iterable[str],
    k: int,
    origins: list[pd.Timestamp],
    horizon: int,
    minimum_positive_weeks: int,
    cleaning_parameters: dict[str, Any],
    calendar_anchor: pd.Timestamp,
    adida_aggregation_weeks: int,
    include_sba: bool = False,
) -> BacktestResult:
    prediction_rows: list[dict[str, Any]] = []
    assignment_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    selected_features = list(feature_names)

    for origin_index, origin in enumerate(origins, start=1):
        train_raw = weekly_raw.loc[weekly_raw["week_start"] < origin].copy()
        clean_result = apply_v2_cleaning(train_raw, **cleaning_parameters)
        train_features = compute_features(clean_result.weekly)
        finite = np.isfinite(train_features[selected_features]).all(axis=1)
        eligible = train_features.loc[
            train_features["n_positive"].ge(minimum_positive_weeks) & finite
        ].copy()
        if len(eligible) <= k:
            raise ValueError(f"Origin {origin.date()} has too few clustering SKUs")
        labeled = fit_solution(eligible, selected_features, k)
        labeled_map = labeled.set_index("sku")["cluster"].to_dict()
        feature_map = train_features.set_index("sku")
        for sku, row in feature_map.iterrows():
            cluster = int(labeled_map.get(sku, 0))
            assignment_rows.append(
                {
                    "origin_index": origin_index,
                    "origin": origin,
                    "sku": sku,
                    "cluster": cluster,
                    "cluster_eligible": cluster > 0,
                    "n_positive": int(row["n_positive"]),
                    "ADI": float(row["ADI"]),
                    "CV2": float(row["CV2"]),
                    "nonzero_mean": float(row["nonzero_mean"]),
                    "acf1": float(row["acf1"]),
                }
            )

        forecasted_skus = 0
        skipped_incomplete_test = 0
        skipped_training_gap = 0
        skipped_model_error = 0
        unavailable_adida = 0
        unavailable_sba = 0
        expected_test_weeks = pd.date_range(origin, periods=horizon, freq="7D")

        forecast_pool = train_features["sku"].astype(str).tolist()
        low_information_forecasted = 0
        for sku in forecast_pool:
            cluster = int(labeled_map.get(sku, 0))
            train_sku = clean_result.weekly.loc[clean_result.weekly["sku"].eq(sku)]
            train_segment = _ending_contiguous_training_segment(train_sku, origin, "sales_v2")
            if len(train_segment) < 4:
                skipped_training_gap += 1
                continue
            test_sku = (
                weekly_raw.loc[
                    weekly_raw["sku"].eq(sku)
                    & weekly_raw["week_start"].isin(expected_test_weeks),
                    ["week_start", "sales"],
                ]
                .set_index("week_start")
                .reindex(expected_test_weeks)
            )
            if test_sku["sales"].isna().any():
                skipped_incomplete_test += 1
                continue
            train_values = train_segment["sales_v2"].to_numpy(dtype=float)
            try:
                forecasts, parameters = forecast_core_models(
                    train_segment["week_start"],
                    train_values,
                    horizon=horizon,
                    calendar_anchor=calendar_anchor,
                    adida_aggregation_weeks=adida_aggregation_weeks,
                    include_sba=include_sba,
                )
            except ValueError:
                skipped_model_error += 1
                continue
            if "ADIDA2" not in forecasts:
                unavailable_adida += 1
            if include_sba and "SBA" not in forecasts:
                unavailable_sba += 1
            scale = mase_scale(train_values)
            actual = test_sku["sales"].to_numpy(dtype=float)
            forecasted_skus += 1
            if cluster == 0:
                low_information_forecasted += 1
            for model, forecast in forecasts.items():
                metrics = evaluate_forecast(actual, forecast, scale)
                prediction_rows.append(
                    {
                        "origin_index": origin_index,
                        "origin": origin,
                        "sku": sku,
                        "cluster": int(cluster),
                        "demand_profile": f"cluster_{cluster}" if cluster > 0 else "low_information",
                        "model": model,
                        "mase_scale": scale,
                        **parameters,
                        **metrics,
                    }
                )

        audit_row = {
                "origin_index": origin_index,
                "origin": origin,
                "training_rows": int(len(train_raw)),
                "eligible_cluster_skus": int(len(labeled)),
                "forecasted_skus": forecasted_skus,
                "low_information_forecasted": low_information_forecasted,
                "skipped_incomplete_test": skipped_incomplete_test,
                "skipped_training_gap": skipped_training_gap,
                "skipped_model_error": skipped_model_error,
                "unavailable_adida": unavailable_adida,
                "v2_corrected_intervals": clean_result.summary["corrected_intervals"],
            }
        if include_sba:
            audit_row["unavailable_sba"] = unavailable_sba
        audit_rows.append(audit_row)

    return BacktestResult(
        predictions=pd.DataFrame(prediction_rows),
        assignments=pd.DataFrame(assignment_rows),
        origin_audits=pd.DataFrame(audit_rows),
        origins=origins,
    )


def derive_statistical_routes(
    development_predictions: pd.DataFrame,
    *,
    core_models: list[str],
    minimum_valid_origins: int = 3,
    minimum_winning_origins: int = 3,
    complex_min_relative_improvement: float = 0.05,
    simple_tie_relative_gap: float = 0.02,
    impact_quantile: float = 0.50,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    per_sku_volume = development_predictions.groupby("sku")["actual_sum"].first()
    total_volume = development_predictions.drop_duplicates(["origin_index", "sku"]).groupby("sku")[
        "actual_sum"
    ].sum()

    provisional: list[dict[str, Any]] = []
    for sku, frame in development_predictions.loc[
        development_predictions["model"].isin([*core_models, "Zero"])
    ].groupby("sku"):
        origin_pivot = frame.pivot_table(index="origin_index", columns="model", values="mae", aggfunc="first")
        means = origin_pivot.mean(axis=0, skipna=True)
        counts = origin_pivot.count(axis=0)
        eligible_models = [
            model for model in [*core_models, "Zero"]
            if model in origin_pivot and counts.get(model, 0) >= minimum_valid_origins
        ]
        if not eligible_models:
            provisional.append({"sku": sku, "stable": False, "candidate_model": None, "reason": "fewer than three valid development origins"})
            continue
        priority = {"MA4_proxy": 0, "Naive": 1, "SES": 2, "ADIDA2": 3, "Zero": 4}
        winner = min(eligible_models, key=lambda model: (means[model], priority.get(model, 99)))
        if {"MA4_proxy", "Naive"}.issubset(means.index):
            gap = abs(float(means["MA4_proxy"] - means["Naive"])) / max(
                float(means[["MA4_proxy", "Naive"]].min()), 1e-12
            )
            if gap < simple_tie_relative_gap and winner in {"MA4_proxy", "Naive"}:
                winner = "MA4_proxy"
        ranked = origin_pivot[eligible_models].copy()
        per_origin_winner = ranked.apply(
            lambda row: min(
                [model for model in eligible_models if row[model] == row.min(skipna=True)],
                key=lambda model: priority.get(model, 99),
            )
            if row.notna().any()
            else np.nan,
            axis=1,
        ).dropna()
        win_count = int((per_origin_winner == winner).sum())
        stable = win_count >= minimum_winning_origins
        if winner in {"MA4_proxy", "Naive"}:
            stable = stable and "Zero" in means and means[winner] < means["Zero"]
        if winner in {"SES", "ADIDA2"}:
            comparators = [model for model in ["MA4_proxy", "Naive", "Zero"] if model in means]
            stable = stable and len(comparators) == 3 and all(
                means[winner] <= means[model] * (1.0 - complex_min_relative_improvement)
                for model in comparators
            )
        if winner == "Zero":
            stable = False
        provisional.append(
            {
                "sku": sku,
                "stable": bool(stable),
                "candidate_model": winner,
                "candidate_mean_mae": float(means[winner]),
                "candidate_wins": win_count,
                "valid_origins": int(len(per_origin_winner)),
                "reason": "majority winner with required simple-baseline improvement" if stable else "model ranking not stable enough",
            }
        )

    provisional_frame = pd.DataFrame(provisional)
    unstable_skus = provisional_frame.loc[~provisional_frame["stable"], "sku"]
    unstable_median_volume = (
        float(total_volume.reindex(unstable_skus).quantile(impact_quantile))
        if len(unstable_skus)
        else np.nan
    )
    for row in provisional:
        sku = row["sku"]
        volume = float(total_volume.get(sku, per_sku_volume.get(sku, 0.0)))
        if row["stable"]:
            path = "预测管理"
            routed_model = row["candidate_model"]
        elif np.isfinite(unstable_median_volume) and volume > unstable_median_volume:
            path = "人工复核"
            routed_model = "MA4_proxy"
        else:
            path = "规则管理"
            routed_model = "Zero"
        rows.append(
            {
                **row,
                "development_actual_volume": volume,
                "unstable_volume_median": unstable_median_volume,
                "management_path": path,
                "routed_model": routed_model,
            }
        )
    return pd.DataFrame(rows)


def apply_routes_to_holdout(
    holdout_predictions: pd.DataFrame, routes: pd.DataFrame
) -> pd.DataFrame:
    merged = holdout_predictions.merge(
        routes[["sku", "management_path", "routed_model"]], on="sku", how="inner"
    )
    selected = merged.loc[merged["model"].eq(merged["routed_model"])].copy()
    selected["model"] = "Layered_mechanism"
    return selected
