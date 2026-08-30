from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .data_audit import WorkbookLoadResult
from .evaluation import summarize_predictions


def prepare_reporting_tables(
    loaded: WorkbookLoadResult,
    config: dict[str, Any],
    output_root: str | Path,
) -> None:
    output = Path(output_root)
    audit_root = output / "audit"
    forecast_root = output / "forecast"

    membership = (
        loaded.raw_long[["sku", "sheet"]]
        .drop_duplicates()
        .assign(value=1)
        .pivot(index="sku", columns="sheet", values="value")
        .fillna(0)
        .astype(int)
    )
    left, right = loaded.audit["sheet_names"][:2]
    membership["universe_status"] = np.select(
        [
            membership[left].eq(1) & membership[right].eq(1),
            membership[left].eq(1) & membership[right].eq(0),
            membership[left].eq(0) & membership[right].eq(1),
        ],
        ["overlap", "2024_2025_only", "2026_only"],
        default="unclassified",
    )
    membership.reset_index().to_csv(audit_root / "sku_universe_transition.csv", index=False)
    universe_summary = {
        "first_sheet": left,
        "second_sheet": right,
        "first_sheet_skus": int(membership[left].sum()),
        "second_sheet_skus": int(membership[right].sum()),
        "overlap_skus": int(membership["universe_status"].eq("overlap").sum()),
        "first_only_skus": int(membership["universe_status"].eq("2024_2025_only").sum()),
        "second_only_skus": int(membership["universe_status"].eq("2026_only").sum()),
        "union_skus": int(len(membership)),
    }
    (audit_root / "sku_universe_transition.json").write_text(
        json.dumps(universe_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    predictions = pd.read_csv(forecast_root / "rolling_origin_predictions.csv")
    assignments = pd.read_csv(forecast_root / "rolling_origin_cluster_assignments.csv")
    routes = pd.read_csv(forecast_root / "frozen_routes_before_holdout.csv")
    repeated = pd.read_csv(forecast_root / "adida2_repeated_improvements.csv")
    candidates = repeated.loc[
        repeated["better_naive_origins"].ge(3)
        & repeated["better_ma4_origins"].ge(3)
    ].copy()
    origin5 = assignments.loc[assignments["origin_index"].eq(5)].drop_duplicates("sku")
    candidate_profiles = candidates.merge(origin5, on="sku", how="left")
    candidate_profiles.to_csv(
        forecast_root / "adida2_repeated_candidate_profiles.csv", index=False
    )
    profile_columns = ["n_positive", "ADI", "CV2", "nonzero_mean", "acf1"]
    comparison_rows: list[dict[str, object]] = []
    for label, frame in [
        ("ADIDA2_repeated_candidates", candidate_profiles),
        ("other_origin5_skus", origin5.loc[~origin5["sku"].isin(candidates["sku"])]),
    ]:
        row: dict[str, object] = {"group": label, "n_skus": int(len(frame))}
        row.update(
            {f"{column}_median": float(frame[column].median()) for column in profile_columns}
        )
        comparison_rows.append(row)
    pd.DataFrame(comparison_rows).to_csv(
        forecast_root / "adida2_profile_summary.csv", index=False
    )

    holdout = predictions.loc[predictions["origin_index"].eq(config["forecast"]["origins"])]
    holdout_pivot = holdout.pivot_table(
        index="sku", columns="model", values="mae", aggfunc="first"
    ).dropna()
    holdout_both = holdout_pivot.loc[
        holdout_pivot["ADIDA2"].lt(holdout_pivot["Naive"])
        & holdout_pivot["ADIDA2"].lt(holdout_pivot["MA4_proxy"])
    ].reset_index()
    holdout_both.to_csv(forecast_root / "adida2_holdout_better_both.csv", index=False)

    sensitivity = pd.read_csv(
        forecast_root / "adida_aggregation_sensitivity_predictions.csv"
    )
    availability = sensitivity.groupby(["origin_index", "sku"])["aggregation_weeks"].nunique()
    common_keys = availability.loc[availability.eq(4)].index
    row_index = pd.MultiIndex.from_frame(sensitivity[["origin_index", "sku"]])
    sensitivity_common = sensitivity.loc[row_index.isin(common_keys)].copy()
    summarize_predictions(sensitivity_common, ["aggregation_weeks", "model"]).to_csv(
        forecast_root / "adida_aggregation_sensitivity_common_summary.csv", index=False
    )

    scheme = pd.read_csv(forecast_root / "holdout_scheme_comparison_predictions.csv")
    enterprise = scheme.loc[
        scheme["model"].eq("Enterprise_MA4"), ["sku", "abs_error_sum", "actual_sum"]
    ].rename(columns={"abs_error_sum": "enterprise_abs_error"})
    layered = scheme.loc[
        scheme["model"].eq("Layered_mechanism"),
        ["sku", "abs_error_sum", "actual_sum", "management_path", "routed_model"],
    ].rename(columns={"abs_error_sum": "layered_abs_error"})
    contribution = enterprise.merge(layered, on=["sku", "actual_sum"])
    contribution["absolute_error_improvement"] = (
        contribution["enterprise_abs_error"] - contribution["layered_abs_error"]
    )
    contribution.to_csv(forecast_root / "holdout_sku_contribution.csv", index=False)
    (
        contribution.groupby(["management_path", "routed_model"], as_index=False)
        .agg(
            n_skus=("sku", "size"),
            actual_volume=("actual_sum", "sum"),
            enterprise_abs_error=("enterprise_abs_error", "sum"),
            layered_abs_error=("layered_abs_error", "sum"),
            net_abs_error_improvement=("absolute_error_improvement", "sum"),
        )
        .to_csv(forecast_root / "holdout_path_contribution.csv", index=False)
    )

    holdout_skus = set(holdout["sku"].unique())
    origin6 = assignments.loc[
        assignments["origin_index"].eq(config["forecast"]["origins"]), ["sku", "cluster"]
    ]
    route_holdout = routes.loc[routes["sku"].isin(holdout_skus)].merge(
        origin6, on="sku", how="left"
    )
    route_holdout["demand_profile"] = route_holdout["cluster"].map(
        lambda value: f"cluster_{int(value)}" if value > 0 else "low_information"
    )
    (
        route_holdout.groupby(
            ["demand_profile", "management_path", "routed_model"], as_index=False
        )
        .size()
        .rename(columns={"size": "n_skus"})
        .to_csv(forecast_root / "holdout_route_counts.csv", index=False)
    )

    mase_counts = (
        predictions.assign(mase_computable=predictions["mase"].notna())
        .groupby(["origin_index", "model"], as_index=False)
        .agg(total=("sku", "size"), mase_computable=("mase_computable", "sum"))
    )
    mase_counts["mase_not_computable"] = (
        mase_counts["total"] - mase_counts["mase_computable"]
    )
    mase_counts.to_csv(forecast_root / "mase_availability_by_origin.csv", index=False)
