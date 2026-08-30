from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.cluster.hierarchy import linkage

from .backtesting import (
    apply_routes_to_holdout,
    build_origins,
    derive_statistical_routes,
    run_backtest,
)
from .cleaning_v2 import CleaningResult, apply_v2_cleaning
from .clustering import (
    algorithm_robustness,
    closest_cluster_profile_effects,
    cluster_profile_intervals,
    cluster_profiles,
    compare_solution_labels,
    evaluate_feature_grid,
    fit_solution,
    mark_pareto,
    select_operational_solution,
)
from .config import load_config
from .data_audit import WorkbookLoadResult, load_workbook_long
from .evaluation import (
    model_wins,
    paired_bootstrap_difference,
    summarize_predictions,
)
from .features import compute_features, verify_feature_identities
from .postprocessing import prepare_reporting_tables
from .reporting import write_final_report
from .stability import sku_assignment_stability, transform_features
from .visualization import render_clustering_charts, render_forecast_charts


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )


def _load_inputs(
    config: dict[str, Any], project_root: Path
) -> tuple[WorkbookLoadResult, CleaningResult, pd.DataFrame]:
    input_path = project_root / config["input"]["workbook"]
    loaded = load_workbook_long(
        input_path,
        metadata_columns=config["input"]["metadata_columns"],
        sku_column=config["input"]["sku_column"],
        minimum_covered_days=config["input"]["calendar"]["minimum_covered_days"],
    )
    if loaded.audit["sha256"] != config["input"]["expected_sha256"]:
        raise RuntimeError("Raw workbook SHA256 does not match the frozen protocol")
    if loaded.audit["blockers"]:
        raise RuntimeError(f"Blocking audit findings: {loaded.audit['blockers']}")
    cleaned = apply_v2_cleaning(loaded.weekly_complete, **config["cleaning_v2"])
    features = compute_features(cleaned.weekly)
    return loaded, cleaned, features


def _save_audit_and_features(
    loaded: WorkbookLoadResult,
    cleaned: CleaningResult,
    features: pd.DataFrame,
    config: dict[str, Any],
    output: Path,
) -> pd.DataFrame:
    audit_root = output / "audit"
    cluster_root = output / "clustering"
    audit_root.mkdir(parents=True, exist_ok=True)
    cluster_root.mkdir(parents=True, exist_ok=True)
    main_features = features.loc[
        features["n_positive"].ge(config["samples"]["main_min_positive_weeks"])
    ].reset_index(drop=True)
    features.to_csv(cluster_root / "features_v2_all_skus.csv", index=False)
    main_features.to_csv(cluster_root / "features_v2_main_sample.csv", index=False)
    cleaned.intervals.to_csv(audit_root / "v2_intervals.csv", index=False)
    loaded.weekly_all.loc[~loaded.weekly_all["is_complete_week"]].to_csv(
        audit_root / "incomplete_sku_weeks.csv", index=False
    )
    audit_payload = {
        **loaded.audit,
        "expected_sha256_match": True,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "scipy_version": scipy.__version__,
        "sklearn_version": sklearn.__version__,
        "random_seed": config["project"]["seed"],
        "v2": cleaned.summary,
        "sample_counts": {
            "positive_weeks_ge_2": int(features["n_positive"].ge(2).sum()),
            "positive_weeks_ge_5": int(features["n_positive"].ge(5).sum()),
            "positive_weeks_ge_10": int(features["n_positive"].ge(10).sum()),
        },
        "feature_identities": verify_feature_identities(features),
    }
    _write_json(audit_root / "audit_summary.json", audit_payload)
    return main_features


def _selection_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    rule = config["clustering"]["operational_selection"]
    return {
        "benchmark_features": rule["benchmark_features"],
        "benchmark_k": rule["benchmark_k"],
        "minimum_cluster_jaccard_median": rule["minimum_cluster_jaccard_median"],
        "minimum_cluster_size": rule["minimum_cluster_size"],
        "minimum_cluster_share": rule["minimum_cluster_share"],
    }


def _run_full_clustering(
    loaded: WorkbookLoadResult,
    cleaned: CleaningResult,
    features: pd.DataFrame,
    main_features: pd.DataFrame,
    config: dict[str, Any],
    output: Path,
    *,
    reuse_grid: bool,
) -> dict[str, Any]:
    root = output / "clustering"
    grid_path = root / "feature_k_grid_500.csv"
    started = time.perf_counter()
    if reuse_grid and grid_path.exists():
        grid = pd.read_csv(grid_path)
    else:
        grid = evaluate_feature_grid(
            main_features,
            anchors=config["features"]["anchors"],
            optional=config["features"]["sensitivity_optional"],
            k_values=config["clustering"]["k_values"],
            repetitions=config["clustering"]["stability"]["repetitions"],
            sample_fraction=config["clustering"]["stability"]["sample_fraction"],
            seed=config["project"]["seed"],
        ).rename(columns={"pareto": "sensitivity_pareto"})
        grid["confirmatory_eligible"] = ~grid["feature_set"].str.contains("median_sales")
        eligible_marked = mark_pareto(
            grid.rename(columns={"sensitivity_pareto": "pareto"}),
            eligibility=grid["confirmatory_eligible"],
        )
        grid["confirmatory_pareto"] = eligible_marked["pareto"]
        grid.to_csv(grid_path, index=False)
    confirmatory = grid.loc[grid["confirmatory_eligible"]].copy()
    selected = select_operational_solution(confirmatory, **_selection_kwargs(config))
    selection_payload = {
        "feature_names": list(selected.feature_names),
        "k": selected.k,
        "reason": selected.reason,
        "row": selected.row,
        "elapsed_seconds": time.perf_counter() - started,
    }
    _write_json(root / "operational_selection_full_period.json", selection_payload)

    feature_names = list(selected.feature_names)
    k = int(selected.k)
    v2_all = features
    raw_all = compute_features(cleaned.weekly, "sales_raw")
    main_min = config["samples"]["main_min_positive_weeks"]
    strict_min = config["samples"]["robustness_min_positive_weeks"]
    raw_main = raw_all.loc[raw_all["n_positive"].ge(main_min)].reset_index(drop=True)
    v2_strict = v2_all.loc[v2_all["n_positive"].ge(strict_min)].reset_index(drop=True)
    labeled_v2 = fit_solution(main_features, feature_names, k)
    labeled_raw = fit_solution(raw_main, feature_names, k)
    labeled_strict = fit_solution(v2_strict, feature_names, k)
    profiles = cluster_profiles(labeled_v2)
    intervals = cluster_profile_intervals(
        labeled_v2, repetitions=1000, seed=config["project"]["seed"]
    )
    assignment = sku_assignment_stability(
        main_features,
        feature_names,
        k,
        repetitions=config["clustering"]["stability"]["repetitions"],
        sample_fraction=config["clustering"]["stability"]["sample_fraction"],
        seed=config["project"]["seed"],
    )
    robustness = {
        **algorithm_robustness(
            labeled_v2,
            feature_names,
            k,
            n_init=config["clustering"]["kmeans_robustness"]["n_init"],
            seed=config["project"]["seed"],
        ),
        "raw_v2": compare_solution_labels(labeled_raw, labeled_v2),
        "main_strict": compare_solution_labels(labeled_v2, labeled_strict),
        "assignment_stability_median": float(assignment["assignment_stability"].median()),
        "assignment_stability_p10": float(assignment["assignment_stability"].quantile(0.10)),
        "assignment_stability_below_0_75": int(
            assignment["assignment_stability"].lt(0.75).sum()
        ),
    }

    labeled_v2_k3 = fit_solution(main_features, feature_names, 3)
    labeled_raw_k3 = fit_solution(raw_main, feature_names, 3)
    labeled_strict_k3 = fit_solution(v2_strict, feature_names, 3)
    effects_k3 = closest_cluster_profile_effects(labeled_v2_k3)
    selected_rows = grid.loc[grid["feature_set"].eq("+".join(feature_names))].set_index("k")
    row2, row3 = selected_rows.loc[2], selected_rows.loc[3]
    jaccards = {
        key: float(value)
        for key, value in json.loads(row3["cluster_jaccard_medians"]).items()
    }
    sizes = labeled_v2_k3["cluster"].value_counts().sort_index()
    thresholds = config["clustering"]["k3_acceptance"]
    raw_v2_k3 = compare_solution_labels(labeled_raw_k3, labeled_v2_k3)
    main_strict_k3 = compare_solution_labels(labeled_v2_k3, labeled_strict_k3)
    conditions = {
        "not_dominated_by_k2": not (
            float(row2["silhouette"]) >= float(row3["silhouette"])
            and float(row2["stability_ari_median"])
            >= float(row3["stability_ari_median"])
            and (
                float(row2["silhouette"]) > float(row3["silhouette"])
                or float(row2["stability_ari_median"])
                > float(row3["stability_ari_median"])
            )
        ),
        "all_cluster_jaccard_at_least_threshold": min(jaccards.values())
        >= thresholds["minimum_cluster_jaccard_median"],
        "all_clusters_large_enough": int(sizes.min())
        >= thresholds["minimum_cluster_size"]
        and float(sizes.min() / len(labeled_v2_k3))
        >= thresholds["minimum_cluster_share"],
        "all_clusters_profile_separated": float(effects_k3["max_effect"].min())
        >= thresholds["minimum_profile_effect_size"],
        "raw_v2_ari_at_least_threshold": float(raw_v2_k3["ari"])
        >= thresholds["minimum_raw_v2_ari"],
        "main_strict_ari_at_least_threshold": float(main_strict_k3["ari"])
        >= thresholds["minimum_main_strict_ari"],
    }
    conditions["independent_non_residual_interpretation"] = all(conditions.values())
    k3_payload = {
        "feature_set": "+".join(feature_names),
        "k2_silhouette": float(row2["silhouette"]),
        "k3_silhouette": float(row3["silhouette"]),
        "k2_stability_ari_median": float(row2["stability_ari_median"]),
        "k3_stability_ari_median": float(row3["stability_ari_median"]),
        "k3_cluster_jaccard_medians": jaccards,
        "k3_cluster_sizes": {str(int(idx)): int(value) for idx, value in sizes.items()},
        "raw_v2": raw_v2_k3,
        "main_strict": main_strict_k3,
        "conditions": conditions,
        "k3_passed": all(conditions.values()),
        "conditional_croston_allowed": all(conditions.values()),
        "interpretation_note": (
            "All numeric gates passed; structural interpretation still requires report review."
            if all(conditions.values())
            else "The least stable third cluster cannot be treated as an independent non-residual demand type."
        ),
    }

    labeled_v2.to_csv(root / "cluster_assignments_v2_k2.csv", index=False)
    labeled_v2_k3.to_csv(root / "cluster_assignments_v2_k3_exploratory.csv", index=False)
    profiles.to_csv(root / "cluster_profiles_v2_k2.csv", index=False)
    intervals.to_csv(root / "cluster_profile_intervals_v2_k2.csv", index=False)
    assignment.to_csv(root / "sku_assignment_stability_k2.csv", index=False)
    effects_k3.to_csv(root / "k3_profile_effects.csv", index=False)
    _write_json(root / "robustness_k2.json", robustness)
    _write_json(root / "k3_acceptance.json", k3_payload)
    transformed, _ = transform_features(main_features, feature_names)
    hierarchy = linkage(transformed, method="ward")
    merge_table = pd.DataFrame(
        hierarchy,
        columns=["left_node", "right_node", "merge_distance", "new_cluster_size"],
    )
    merge_table.insert(0, "merge_step", np.arange(1, len(merge_table) + 1))
    merge_table.to_csv(root / "ward_merge_distances.csv", index=False)
    return selection_payload


def _run_pre_origin_selection(
    loaded: WorkbookLoadResult,
    config: dict[str, Any],
    output: Path,
    *,
    reuse_grid: bool,
) -> tuple[dict[str, Any], list[pd.Timestamp]]:
    root = output / "forecast"
    root.mkdir(parents=True, exist_ok=True)
    origins = build_origins(
        loaded.weekly_complete,
        config["forecast"]["total_backtest_weeks"],
        config["forecast"]["horizon_weeks"],
    )
    first_origin = origins[0]
    train_raw = loaded.weekly_complete.loc[
        loaded.weekly_complete["week_start"].lt(first_origin)
    ]
    cleaned = apply_v2_cleaning(train_raw, **config["cleaning_v2"])
    features = compute_features(cleaned.weekly)
    columns = [
        *config["features"]["anchors"],
        *config["features"]["confirmatory_optional"],
    ]
    finite = np.isfinite(features[columns]).all(axis=1)
    eligible = features.loc[
        features["n_positive"].ge(config["samples"]["main_min_positive_weeks"])
        & finite
    ].reset_index(drop=True)
    grid_path = root / "pre_first_origin_feature_k_grid_500.csv"
    started = time.perf_counter()
    if reuse_grid and grid_path.exists():
        grid = pd.read_csv(grid_path)
    else:
        grid = evaluate_feature_grid(
            eligible,
            anchors=config["features"]["anchors"],
            optional=config["features"]["confirmatory_optional"],
            k_values=config["clustering"]["k_values"],
            repetitions=config["clustering"]["stability"]["repetitions"],
            sample_fraction=config["clustering"]["stability"]["sample_fraction"],
            seed=config["project"]["seed"],
        )
        grid.to_csv(grid_path, index=False)
    rule = config["clustering"]["operational_selection"]
    unconstrained = select_operational_solution(
        grid,
        benchmark_features=rule["benchmark_features"],
        benchmark_k=rule["benchmark_k"],
    )
    selected = select_operational_solution(grid, **_selection_kwargs(config))
    common = {
        "first_origin": str(first_origin.date()),
        "origins": [str(value.date()) for value in origins],
        "training_week_max": str(train_raw["week_start"].max().date()),
        "training_rows": int(len(train_raw)),
        "training_unique_skus": int(train_raw["sku"].nunique()),
        "eligible_cluster_skus": int(len(eligible)),
        "v2": cleaned.summary,
        "elapsed_seconds": time.perf_counter() - started,
    }
    unconstrained_payload = {
        **common,
        "feature_names": list(unconstrained.feature_names),
        "k": unconstrained.k,
        "reason": unconstrained.reason,
        "row": unconstrained.row,
        "status": "Preserved unconstrained result before protocol amendment 1.4 gate.",
    }
    selected_payload = {
        **common,
        "feature_names": list(selected.feature_names),
        "k": selected.k,
        "reason": selected.reason,
        "row": selected.row,
        "selection_correction": "Protocol amendment 1.4; the stability grid itself was not altered.",
        "unconstrained_selection_file": "pre_first_origin_selection_unconstrained.json",
    }
    _write_json(root / "pre_first_origin_selection_unconstrained.json", unconstrained_payload)
    _write_json(root / "pre_first_origin_selection.json", selected_payload)
    return selected_payload, origins


def _records(frame: pd.DataFrame) -> list[dict[str, object]]:
    return json.loads(frame.to_json(orient="records", date_format="iso"))


def _run_forecasts(
    loaded: WorkbookLoadResult,
    selection: dict[str, Any],
    origins: list[pd.Timestamp],
    config: dict[str, Any],
    output: Path,
) -> dict[str, Any]:
    root = output / "forecast"
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
    )
    predictions = backtest.predictions
    development = predictions.loc[
        predictions["origin_index"].le(config["forecast"]["development_origins"])
    ].copy()
    holdout = predictions.loc[
        predictions["origin_index"].eq(config["forecast"]["origins"])
    ].copy()
    routing = config["forecast"]["routing"]
    routes = derive_statistical_routes(
        development,
        core_models=config["forecast"]["models"],
        minimum_valid_origins=routing["minimum_valid_origins"],
        minimum_winning_origins=routing["minimum_winning_origins"],
        complex_min_relative_improvement=routing["complex_min_relative_improvement"],
        simple_tie_relative_gap=routing["simple_tie_relative_gap"],
        impact_quantile=routing["impact_quantile"],
    )
    layered = apply_routes_to_holdout(holdout, routes)
    common_skus = sorted(layered["sku"].unique())
    enterprise = holdout.loc[
        holdout["model"].eq("MA4_proxy") & holdout["sku"].isin(common_skus)
    ].copy()
    enterprise["model"] = "Enterprise_MA4"
    comparison = pd.concat([enterprise, layered], ignore_index=True)

    origin_summary = summarize_predictions(
        predictions, ["origin_index", "origin", "model"]
    )
    origin_summary["mean_mase_rank"] = origin_summary.groupby("origin_index")[
        "mean_mase"
    ].rank(method="min", ascending=True)
    overall_summary = summarize_predictions(predictions, ["model"])
    cluster_summary = summarize_predictions(
        predictions, ["origin_index", "cluster", "model"]
    )
    profile_summary = summarize_predictions(predictions, ["demand_profile", "model"])
    scheme_summary = summarize_predictions(comparison, ["model"])

    bootstrap_rows: list[dict[str, object]] = []
    for model in ["Naive", "SES", "ADIDA2"]:
        bootstrap_rows.append(
            paired_bootstrap_difference(
                development,
                model,
                "MA4_proxy",
                repetitions=config["evaluation"]["paired_bootstrap_repetitions"],
                seed=config["project"]["seed"],
            )
        )
    for model in ["SES", "ADIDA2"]:
        bootstrap_rows.append(
            paired_bootstrap_difference(
                development,
                model,
                "Naive",
                repetitions=config["evaluation"]["paired_bootstrap_repetitions"],
                seed=config["project"]["seed"],
            )
        )
    scheme_bootstrap = paired_bootstrap_difference(
        comparison,
        "Layered_mechanism",
        "Enterprise_MA4",
        repetitions=config["evaluation"]["paired_bootstrap_repetitions"],
        seed=config["project"]["seed"],
    )

    dev_pivot = development.pivot_table(
        index=["origin_index", "sku"], columns="model", values="mae", aggfunc="first"
    )
    holdout_pivot = holdout.pivot_table(
        index="sku", columns="model", values="mae", aggfunc="first"
    )
    adida_dev = dev_pivot.dropna(subset=["ADIDA2", "Naive", "MA4_proxy"])
    adida_holdout = holdout_pivot.dropna(subset=["ADIDA2", "Naive", "MA4_proxy"])
    better_naive = adida_dev["ADIDA2"].lt(adida_dev["Naive"])
    better_ma4 = adida_dev["ADIDA2"].lt(adida_dev["MA4_proxy"])
    repeated = pd.DataFrame(
        {
            "better_naive_origins": better_naive.groupby("sku").sum(),
            "better_ma4_origins": better_ma4.groupby("sku").sum(),
            "valid_origins": adida_dev.groupby("sku").size(),
        }
    ).reset_index()
    adida_summary = {
        "development_sku_origins": int(len(adida_dev)),
        "development_better_than_naive_sku_origins": int(better_naive.sum()),
        "development_better_than_ma4_sku_origins": int(better_ma4.sum()),
        "development_skus_better_than_naive_at_least_3_origins": int(
            repeated["better_naive_origins"].ge(3).sum()
        ),
        "development_skus_better_than_ma4_at_least_3_origins": int(
            repeated["better_ma4_origins"].ge(3).sum()
        ),
        "holdout_valid_skus": int(len(adida_holdout)),
        "holdout_better_than_naive_skus": int(
            adida_holdout["ADIDA2"].lt(adida_holdout["Naive"]).sum()
        ),
        "holdout_better_than_ma4_skus": int(
            adida_holdout["ADIDA2"].lt(adida_holdout["MA4_proxy"]).sum()
        ),
        "routed_to_adida2": int(routes["routed_model"].eq("ADIDA2").sum()),
    }

    sensitivity_frames = [
        predictions.loc[predictions["model"].eq("ADIDA2")].assign(aggregation_weeks=2)
    ]
    for aggregation_weeks in config["forecast"]["adida_sensitivity_weeks"]:
        run = run_backtest(
            loaded.weekly_complete,
            feature_names=selection["feature_names"],
            k=int(selection["k"]),
            origins=origins,
            horizon=config["forecast"]["horizon_weeks"],
            minimum_positive_weeks=config["samples"]["main_min_positive_weeks"],
            cleaning_parameters=config["cleaning_v2"],
            calendar_anchor=pd.Timestamp(loaded.weekly_complete["week_start"].min()),
            adida_aggregation_weeks=int(aggregation_weeks),
        )
        frame = run.predictions.loc[run.predictions["model"].eq("ADIDA2")].copy()
        frame["aggregation_weeks"] = int(aggregation_weeks)
        sensitivity_frames.append(frame)
    sensitivity = pd.concat(sensitivity_frames, ignore_index=True)
    sensitivity["model"] = sensitivity["aggregation_weeks"].map(
        lambda value: f"ADIDA{int(value)}"
    )

    predictions.to_csv(root / "rolling_origin_predictions.csv", index=False)
    backtest.assignments.to_csv(root / "rolling_origin_cluster_assignments.csv", index=False)
    backtest.origin_audits.to_csv(root / "rolling_origin_audits.csv", index=False)
    origin_summary.to_csv(root / "model_summary_by_origin.csv", index=False)
    overall_summary.to_csv(root / "model_summary_overall.csv", index=False)
    cluster_summary.to_csv(root / "model_summary_by_origin_cluster.csv", index=False)
    profile_summary.to_csv(root / "model_summary_by_profile.csv", index=False)
    model_wins(development).to_csv(root / "development_model_wins_by_sku.csv", index=False)
    repeated.to_csv(root / "adida2_repeated_improvements.csv", index=False)
    routes.to_csv(root / "frozen_routes_before_holdout.csv", index=False)
    comparison.to_csv(root / "holdout_scheme_comparison_predictions.csv", index=False)
    scheme_summary.to_csv(root / "holdout_scheme_comparison_summary.csv", index=False)
    sensitivity.to_csv(root / "adida_aggregation_sensitivity_predictions.csv", index=False)
    summarize_predictions(sensitivity, ["aggregation_weeks", "model"]).to_csv(
        root / "adida_aggregation_sensitivity_summary.csv", index=False
    )
    pd.DataFrame(bootstrap_rows).to_csv(
        root / "development_paired_bootstrap.csv", index=False
    )
    outcome = {
        "origins": [str(value.date()) for value in origins],
        "prediction_rows": int(len(predictions)),
        "forecasted_unique_skus": int(predictions["sku"].nunique()),
        "origin_audits": _records(backtest.origin_audits),
        "route_counts": {
            str(key): int(value)
            for key, value in routes["management_path"].value_counts().items()
        },
        "routed_model_counts": {
            str(key): int(value)
            for key, value in routes["routed_model"].value_counts().items()
        },
        "holdout_common_skus": int(len(common_skus)),
        "holdout_scheme_summary": _records(scheme_summary),
        "holdout_scheme_bootstrap": scheme_bootstrap,
        "adida2": adida_summary,
        "conditional_croston_used": False,
    }
    _write_json(root / "backtest_outcome.json", outcome)
    return outcome


def run_pipeline(
    config_path: str | Path,
    *,
    reuse_existing: bool = False,
    reuse_existing_grids: bool = False,
) -> dict[str, Any]:
    config_file = Path(config_path).resolve()
    project_root = config_file.parent.parent
    config = load_config(config_file)
    output = project_root / config["outputs"]["root"]
    required_existing = [
        output / "audit" / "audit_summary.json",
        output / "clustering" / "operational_selection_full_period.json",
        output / "forecast" / "backtest_outcome.json",
    ]
    if reuse_existing:
        missing = [str(path) for path in required_existing if not path.exists()]
        if missing:
            raise RuntimeError(f"Cannot reuse missing outputs: {missing}")
        loaded, _, _ = _load_inputs(config, project_root)
        prepare_reporting_tables(loaded, config, output)
        render_clustering_charts(output)
        render_forecast_charts(output, config)
        write_final_report(project_root, config)
        return _load_existing_summary(output)

    loaded, cleaned, features = _load_inputs(config, project_root)
    main_features = _save_audit_and_features(loaded, cleaned, features, config, output)
    full_selection = _run_full_clustering(
        loaded,
        cleaned,
        features,
        main_features,
        config,
        output,
        reuse_grid=reuse_existing_grids,
    )
    forecast_selection, origins = _run_pre_origin_selection(
        loaded,
        config,
        output,
        reuse_grid=reuse_existing_grids,
    )
    outcome = _run_forecasts(loaded, forecast_selection, origins, config, output)
    prepare_reporting_tables(loaded, config, output)
    render_clustering_charts(output)
    render_forecast_charts(output, config)
    write_final_report(project_root, config)
    return {
        "full_period": full_selection,
        "pre_first_origin": forecast_selection,
        "backtest": outcome,
    }


def _load_existing_summary(output: Path) -> dict[str, Any]:
    return {
        "full_period": json.loads(
            (output / "clustering" / "operational_selection_full_period.json").read_text(
                encoding="utf-8"
            )
        ),
        "pre_first_origin": json.loads(
            (output / "forecast" / "pre_first_origin_selection.json").read_text(
                encoding="utf-8"
            )
        ),
        "backtest": json.loads(
            (output / "forecast" / "backtest_outcome.json").read_text(encoding="utf-8")
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reproduce the frozen German Amazon SKU clustering and forecasting study."
    )
    parser.add_argument("--config", default="config/analysis.yaml")
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Rebuild tables, figures, and reports from existing validated outputs.",
    )
    parser.add_argument(
        "--reuse-existing-grids",
        action="store_true",
        help="Developer-only: reuse the two expensive 500-repeat grid CSVs.",
    )
    args = parser.parse_args()
    summary = run_pipeline(
        args.config,
        reuse_existing=args.reuse_existing,
        reuse_existing_grids=args.reuse_existing_grids,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
