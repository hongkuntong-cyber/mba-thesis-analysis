from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .backtesting import build_origins, run_backtest
from .config import load_config
from .data_audit import load_workbook_long
from .evaluation import paired_bootstrap_difference, summarize_predictions


MODELS = ["MA4_proxy", "Naive", "SES", "ADIDA2", "SBA"]
POOLS = {
    1: ["MA4_proxy", "Naive", "SES", "ADIDA2"],
    2: ["MA4_proxy", "Naive", "ADIDA2", "SBA"],
}
PROFILE_NAMES = {
    0: "low_information",
    1: "active_moderate_intermittent",
    2: "highly_intermittent",
}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def _common_mase_sample(frame: pd.DataFrame) -> pd.DataFrame:
    relevant = frame.loc[frame["model"].isin(MODELS)].copy()
    available = (
        relevant.dropna(subset=["mase"])
        .groupby(["origin_index", "sku"])["model"]
        .nunique()
    )
    common_keys = available.loc[available.eq(len(MODELS))].reset_index()[
        ["origin_index", "sku"]
    ]
    return relevant.merge(common_keys, on=["origin_index", "sku"], how="inner")


def _rank_summary(frame: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    summary = summarize_predictions(frame, groups)
    rank_groups = [column for column in groups if column != "model"]
    if rank_groups:
        summary["mean_mase_rank"] = summary.groupby(rank_groups)["mean_mase"].rank(
            method="min", ascending=True
        )
        summary["median_mase_rank"] = summary.groupby(rank_groups)["median_mase"].rank(
            method="min", ascending=True
        )
    else:
        summary["mean_mase_rank"] = summary["mean_mase"].rank(method="min", ascending=True)
        summary["median_mase_rank"] = summary["median_mase"].rank(
            method="min", ascending=True
        )
    return summary


def _paired_rows(
    predictions: pd.DataFrame,
    *,
    period: str,
    repetitions: int,
    seed: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    scopes = [("all", predictions)]
    scopes.extend(
        (PROFILE_NAMES[cluster], predictions.loc[predictions["cluster"].eq(cluster)])
        for cluster in [1, 2]
    )
    for scope, frame in scopes:
        for model in ["Naive", "SES", "ADIDA2", "SBA"]:
            for baseline in ["MA4_proxy", "Naive"]:
                if model == baseline:
                    continue
                result = paired_bootstrap_difference(
                    frame,
                    model,
                    baseline,
                    repetitions=repetitions,
                    seed=seed,
                )
                rows.append({"period": period, "scope": scope, **result})
    return rows


def _reconcile_v1(new_predictions: pd.DataFrame, project_root: Path) -> dict[str, Any]:
    v1_path = project_root / "outputs/forecast/rolling_origin_predictions.csv"
    if not v1_path.exists():
        return {"status": "not_run", "reason": "V1 prediction file is unavailable"}
    v1 = pd.read_csv(v1_path)
    core = ["MA4_proxy", "Naive", "SES", "ADIDA2", "Zero"]
    left = v1.loc[v1["model"].isin(core)].copy()
    right = new_predictions.loc[new_predictions["model"].isin(core)].copy()
    keys = ["origin_index", "sku", "model"]
    metrics = [
        "cluster",
        "mase_scale",
        "mase",
        "mae",
        "actual_sum",
        "forecast_sum",
        "abs_error_sum",
        "signed_error_sum",
    ]
    merged = left[keys + metrics].merge(
        right[keys + metrics],
        on=keys,
        how="outer",
        suffixes=("_v1", "_v2"),
        indicator=True,
        validate="one_to_one",
    )
    differences: dict[str, float] = {}
    for metric in metrics:
        v1_values = pd.to_numeric(merged[f"{metric}_v1"], errors="coerce")
        v2_values = pd.to_numeric(merged[f"{metric}_v2"], errors="coerce")
        finite = np.isfinite(v1_values) & np.isfinite(v2_values)
        differences[metric] = (
            float(np.max(np.abs(v1_values[finite] - v2_values[finite])))
            if finite.any()
            else 0.0
        )
    return {
        "status": "passed"
        if merged["_merge"].eq("both").all() and max(differences.values()) <= 1e-12
        else "failed",
        "v1_rows": int(len(left)),
        "v2_core_rows": int(len(right)),
        "unmatched_rows": int(merged["_merge"].ne("both").sum()),
        "maximum_absolute_differences": differences,
    }


def run(config_path: str | Path) -> dict[str, Any]:
    config_file = Path(config_path).resolve()
    project_root = config_file.parent.parent
    config = load_config(config_file)
    workbook_path = project_root / config["input"]["workbook"]
    loaded = load_workbook_long(
        workbook_path,
        metadata_columns=config["input"]["metadata_columns"],
        sku_column=config["input"]["sku_column"],
        minimum_covered_days=config["input"]["calendar"]["minimum_covered_days"],
    )
    if loaded.audit["sha256"] != config["input"]["expected_sha256"]:
        raise RuntimeError("Raw workbook SHA256 does not match the frozen protocol")
    if loaded.audit["blockers"]:
        raise RuntimeError(f"Blocking audit findings: {loaded.audit['blockers']}")

    selection_path = project_root / "outputs/forecast/pre_first_origin_selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    origins = build_origins(
        loaded.weekly_complete,
        config["forecast"]["total_backtest_weeks"],
        config["forecast"]["horizon_weeks"],
    )
    backtest = run_backtest(
        loaded.weekly_complete,
        feature_names=selection["feature_names"],
        k=int(selection["k"]),
        origins=origins,
        horizon=config["forecast"]["horizon_weeks"],
        minimum_positive_weeks=config["samples"]["main_min_positive_weeks"],
        cleaning_parameters=config["cleaning_v2"],
        calendar_anchor=pd.Timestamp(loaded.weekly_complete["week_start"].min()),
        adida_aggregation_weeks=config["forecast"]["adida_aggregation_weeks"],
        include_sba=True,
    )
    predictions = backtest.predictions.copy()
    predictions["pool_profile"] = predictions["cluster"].map(PROFILE_NAMES)
    predictions["pool_eligible"] = predictions.apply(
        lambda row: row["model"] in POOLS.get(int(row["cluster"]), []), axis=1
    )

    formal = predictions.loc[predictions["model"].isin(MODELS)].copy()
    development = formal.loc[
        formal["origin_index"].le(config["forecast"]["development_origins"])
    ].copy()
    holdout = formal.loc[
        formal["origin_index"].eq(config["forecast"]["origins"])
    ].copy()
    common_all = _common_mase_sample(formal)
    common_holdout = _common_mase_sample(holdout)
    pool_holdout = holdout.loc[holdout["pool_eligible"] & holdout["cluster"].isin([1, 2])]
    pool_development = development.loc[
        development["pool_eligible"] & development["cluster"].isin([1, 2])
    ]

    output_root = project_root / "outputs/forecast_pool_v2_exploratory"
    output_root.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_root / "rolling_origin_predictions_with_sba.csv", index=False)
    predictions.loc[predictions["model"].eq("SBA")].to_csv(
        output_root / "rolling_origin_sba_predictions.csv", index=False
    )
    backtest.assignments.to_csv(output_root / "rolling_origin_cluster_assignments.csv", index=False)
    backtest.origin_audits.to_csv(output_root / "rolling_origin_audits.csv", index=False)

    summaries = {
        "all_origins_native": _rank_summary(formal, ["model"]),
        "all_origins_common_mase": _rank_summary(common_all, ["model"]),
        "by_origin_native": _rank_summary(
            formal, ["origin_index", "origin", "model"]
        ),
        "by_origin_common_mase": _rank_summary(
            common_all, ["origin_index", "origin", "model"]
        ),
        "holdout_native": _rank_summary(holdout, ["model"]),
        "holdout_common_mase": _rank_summary(common_holdout, ["model"]),
        "holdout_by_cluster_native": _rank_summary(
            holdout.loc[holdout["cluster"].isin([1, 2])], ["cluster", "pool_profile", "model"]
        ),
        "holdout_by_cluster_common_mase": _rank_summary(
            common_holdout.loc[common_holdout["cluster"].isin([1, 2])],
            ["cluster", "pool_profile", "model"],
        ),
        "holdout_prespecified_pools": _rank_summary(
            pool_holdout, ["cluster", "pool_profile", "model"]
        ),
        "development_prespecified_pools": _rank_summary(
            pool_development, ["cluster", "pool_profile", "model"]
        ),
        "by_origin_cluster_native": _rank_summary(
            formal.loc[formal["cluster"].isin([1, 2])],
            ["origin_index", "origin", "cluster", "pool_profile", "model"],
        ),
        "development_by_cluster_native": _rank_summary(
            development.loc[development["cluster"].isin([1, 2])],
            ["cluster", "pool_profile", "model"],
        ),
    }
    for name, frame in summaries.items():
        frame.to_csv(output_root / f"{name}.csv", index=False)

    paired = pd.DataFrame(
        _paired_rows(
            development,
            period="development_origins_1_5",
            repetitions=config["evaluation"]["paired_bootstrap_repetitions"],
            seed=config["project"]["seed"],
        )
        + _paired_rows(
            holdout,
            period="exploratory_origin_6",
            repetitions=config["evaluation"]["paired_bootstrap_repetitions"],
            seed=config["project"]["seed"],
        )
    )
    paired.to_csv(output_root / "paired_mase_bootstrap.csv", index=False)

    reconciliation = _reconcile_v1(predictions, project_root)
    _write_json(output_root / "v1_reconciliation.json", reconciliation)
    method = {
        "status": "exploratory_after_v1_holdout_was_viewed",
        "protocol": "protocol/forecast_pool_v2_exploratory_protocol.md",
        "source_sha256": loaded.audit["sha256"],
        "feature_names": selection["feature_names"],
        "k": int(selection["k"]),
        "origins": [str(origin.date()) for origin in origins],
        "models": MODELS,
        "pools": {str(key): value for key, value in POOLS.items()},
        "holdout_native_sku_records": {
            model: int(len(holdout.loc[holdout["model"].eq(model)])) for model in MODELS
        },
        "holdout_native_valid_mase": {
            model: int(holdout.loc[holdout["model"].eq(model), "mase"].notna().sum())
            for model in MODELS
        },
        "holdout_common_mase_skus": int(common_holdout["sku"].nunique()),
        "reconciliation": reconciliation,
    }
    _write_json(output_root / "method_and_availability.json", method)

    return {
        "output_root": str(output_root),
        "holdout_native": summaries["holdout_native"].to_dict(orient="records"),
        "holdout_common_mase": summaries["holdout_common_mase"].to_dict(orient="records"),
        "holdout_by_cluster_common_mase": summaries[
            "holdout_by_cluster_common_mase"
        ].to_dict(orient="records"),
        "reconciliation": reconciliation,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the frozen exploratory cluster-specific lightweight forecast pools."
    )
    parser.add_argument("--config", default="config/analysis.yaml")
    args = parser.parse_args()
    print(json.dumps(run(args.config), ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
