from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .backtesting import build_origins, run_backtest
from .config import load_config
from .data_audit import load_workbook_long
from .evaluation import (
    model_wins,
    paired_bootstrap_loss_difference,
    summarize_predictions,
)


PROFILE_NAMES = {
    0: "low_information",
    1: "relative_active_persistent",
    2: "low_frequency_sparse",
}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def _attach_profiles(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["cluster_profile"] = output["cluster"].map(PROFILE_NAMES).fillna("unknown")
    return output


def common_model_sample(
    predictions: pd.DataFrame,
    models: Iterable[str],
    *,
    require_mase: bool = False,
) -> pd.DataFrame:
    requested = list(models)
    relevant = predictions.loc[predictions["model"].isin(requested)].copy()
    if require_mase:
        relevant = relevant.dropna(subset=["mase"])
    counts = relevant.groupby(["horizon_label", "origin_index", "sku"])["model"].nunique()
    keys = counts.loc[counts.eq(len(requested))].reset_index()[
        ["horizon_label", "origin_index", "sku"]
    ]
    return relevant.merge(
        keys,
        on=["horizon_label", "origin_index", "sku"],
        how="inner",
        validate="many_to_one",
    )


def _rank_summary(frame: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    summary = summarize_predictions(frame, groups)
    rank_groups = [column for column in groups if column != "model"]
    for metric in ["mean_mase", "median_mase", "horizon_total_wape", "aggregate_wape"]:
        rank_name = f"{metric}_rank"
        if rank_groups:
            summary[rank_name] = summary.groupby(rank_groups)[metric].rank(
                method="min", ascending=True
            )
        else:
            summary[rank_name] = summary[metric].rank(method="min", ascending=True)
    return summary


def _scope_frames(frame: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    scopes = [("all", frame)]
    scopes.extend(
        (PROFILE_NAMES[cluster], frame.loc[frame["cluster"].eq(cluster)])
        for cluster in [0, 1, 2]
    )
    return scopes


def _paired_comparisons(
    common: pd.DataFrame,
    common_mase: pd.DataFrame,
    *,
    baselines: list[str],
    repetitions: int,
    seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    horizons = common[["horizon_label", "horizon_weeks"]].drop_duplicates()
    for horizon in horizons.itertuples(index=False):
        total_frame = common.loc[common["horizon_label"].eq(horizon.horizon_label)]
        mase_frame = common_mase.loc[
            common_mase["horizon_label"].eq(horizon.horizon_label)
        ]
        total_scopes = dict(_scope_frames(total_frame))
        mase_scopes = dict(_scope_frames(mase_frame))
        for scope in ["all", *PROFILE_NAMES.values()]:
            for baseline in baselines:
                for loss, source in [
                    ("horizon_total_abs_error", total_scopes[scope]),
                    ("mase", mase_scopes[scope]),
                ]:
                    rows.append(
                        {
                            "horizon_label": horizon.horizon_label,
                            "horizon_weeks": int(horizon.horizon_weeks),
                            "scope": scope,
                            **paired_bootstrap_loss_difference(
                                source,
                                "PXQ",
                                baseline,
                                loss_column=loss,
                                repetitions=repetitions,
                                seed=seed,
                            ),
                        }
                    )
    return pd.DataFrame(rows)


def _origin_head_to_head(common: pd.DataFrame, baselines: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (horizon_label, horizon_weeks), horizon_frame in common.groupby(
        ["horizon_label", "horizon_weeks"], sort=False
    ):
        for scope, scope_frame in _scope_frames(horizon_frame):
            for origin_index, origin_frame in scope_frame.groupby("origin_index"):
                summary = summarize_predictions(origin_frame, ["model"]).set_index("model")
                if "PXQ" not in summary.index:
                    continue
                for baseline in baselines:
                    if baseline not in summary.index:
                        continue
                    rows.append(
                        {
                            "horizon_label": horizon_label,
                            "horizon_weeks": int(horizon_weeks),
                            "scope": scope,
                            "origin_index": int(origin_index),
                            "origin": str(pd.Timestamp(origin_frame["origin"].iloc[0]).date()),
                            "baseline": baseline,
                            "pxq_horizon_total_wape": float(
                                summary.loc["PXQ", "horizon_total_wape"]
                            ),
                            "baseline_horizon_total_wape": float(
                                summary.loc[baseline, "horizon_total_wape"]
                            ),
                            "pxq_better": bool(
                                summary.loc["PXQ", "horizon_total_wape"]
                                < summary.loc[baseline, "horizon_total_wape"]
                            ),
                            "n_sku_origins": int(summary.loc["PXQ", "n_sku_origins"]),
                        }
                    )
    return pd.DataFrame(rows)


def _predictability_gate(
    head_to_head: pd.DataFrame,
    paired: pd.DataFrame,
    *,
    required_baselines: list[str],
    minimum_winning_origins: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    keys = head_to_head[["horizon_label", "horizon_weeks", "scope"]].drop_duplicates()
    total_paired = paired.loc[paired["loss"].eq("horizon_total_abs_error")]
    for key in keys.itertuples(index=False):
        record: dict[str, Any] = {
            "horizon_label": key.horizon_label,
            "horizon_weeks": int(key.horizon_weeks),
            "scope": key.scope,
        }
        passed = True
        for baseline in required_baselines:
            origin_rows = head_to_head.loc[
                head_to_head["horizon_label"].eq(key.horizon_label)
                & head_to_head["scope"].eq(key.scope)
                & head_to_head["baseline"].eq(baseline)
            ]
            pair_rows = total_paired.loc[
                total_paired["horizon_label"].eq(key.horizon_label)
                & total_paired["scope"].eq(key.scope)
                & total_paired["baseline"].eq(baseline)
            ]
            wins = int(origin_rows["pxq_better"].sum())
            ci_high = float(pair_rows["ci_high"].iloc[0]) if len(pair_rows) == 1 else np.nan
            record[f"wins_vs_{baseline}"] = wins
            record[f"paired_ci_high_vs_{baseline}"] = ci_high
            passed = passed and wins >= minimum_winning_origins and np.isfinite(ci_high) and ci_high < 0
        record["pxq_value_supported"] = bool(passed)
        rows.append(record)
    return pd.DataFrame(rows)


def _sku_head_to_head(common: pd.DataFrame, baselines: list[str]) -> pd.DataFrame:
    pivot = common.pivot_table(
        index=[
            "horizon_label",
            "horizon_weeks",
            "origin_index",
            "origin",
            "sku",
            "cluster",
            "cluster_profile",
        ],
        columns="model",
        values="horizon_total_abs_error",
        aggfunc="first",
    ).reset_index()
    rows: list[pd.DataFrame] = []
    for baseline in baselines:
        current = pivot[
            [
                "horizon_label",
                "horizon_weeks",
                "origin_index",
                "origin",
                "sku",
                "cluster",
                "cluster_profile",
                "PXQ",
                baseline,
            ]
        ].copy()
        current["baseline"] = baseline
        current["pxq_better"] = current["PXQ"].lt(current[baseline])
        current["loss_difference"] = current["PXQ"] - current[baseline]
        rows.append(current.drop(columns=["PXQ", baseline]))
    return pd.concat(rows, ignore_index=True)


def run_pxq_validation(config_path: str | Path) -> dict[str, Any]:
    config_file = Path(config_path).resolve()
    project_root = config_file.parent.parent
    config = load_config(config_file)
    if config["project"]["analysis_mode"] != "retrospective_method_development":
        raise RuntimeError("V4.0 historical validation must remain retrospective")

    loaded = load_workbook_long(
        project_root / config["input"]["workbook"],
        metadata_columns=config["input"]["metadata_columns"],
        sku_column=config["input"]["sku_column"],
        minimum_covered_days=config["input"]["calendar"]["minimum_covered_days"],
    )
    if loaded.audit["sha256"] != config["input"]["expected_sha256"]:
        raise RuntimeError("Raw workbook SHA256 differs from the frozen V4.0 protocol")
    if loaded.audit["blockers"]:
        raise RuntimeError(f"Blocking audit findings: {loaded.audit['blockers']}")

    output_root = project_root / config["outputs"]["root"]
    output_root.mkdir(parents=True, exist_ok=True)
    _write_json(output_root / "input_audit.json", loaded.audit)
    calendar_anchor = pd.Timestamp(loaded.weekly_complete["week_start"].min())
    predictions_list: list[pd.DataFrame] = []
    assignments_list: list[pd.DataFrame] = []
    audits_list: list[pd.DataFrame] = []
    origin_map: dict[str, list[str]] = {}

    for horizon in config["forecast"]["horizons"]:
        weeks = int(horizon["weeks"])
        n_origins = int(horizon["origins"])
        origins = build_origins(
            loaded.weekly_complete,
            total_weeks=weeks * n_origins,
            horizon=weeks,
        )
        if len(origins) != n_origins:
            raise RuntimeError(f"Unexpected origin count for {horizon['label']}")
        result = run_backtest(
            loaded.weekly_complete,
            feature_names=config["clustering"]["feature_names"],
            k=int(config["clustering"]["k"]),
            origins=origins,
            horizon=weeks,
            minimum_positive_weeks=int(config["samples"]["main_min_positive_weeks"]),
            cleaning_parameters=config["cleaning_v2"],
            calendar_anchor=calendar_anchor,
            adida_aggregation_weeks=int(config["forecast"]["adida_aggregation_weeks"]),
            include_sba=False,
            include_pxq=True,
            pxq_lookback_weeks=weeks,
        )
        origin_map[str(horizon["label"])] = [str(value.date()) for value in origins]
        for frame, destination in [
            (result.predictions, predictions_list),
            (result.assignments, assignments_list),
            (result.origin_audits, audits_list),
        ]:
            enriched = frame.copy()
            enriched.insert(0, "horizon_label", str(horizon["label"]))
            enriched.insert(1, "horizon_weeks", weeks)
            enriched.insert(2, "approximate_days", int(horizon["approximate_days"]))
            destination.append(enriched)

    predictions = _attach_profiles(pd.concat(predictions_list, ignore_index=True))
    assignments = _attach_profiles(pd.concat(assignments_list, ignore_index=True))
    audits = pd.concat(audits_list, ignore_index=True)
    formal_models = [
        config["forecast"]["primary_model"],
        *config["forecast"]["comparison_models"],
    ]
    allowed_models = {*formal_models, config["forecast"]["value_baseline"]}
    unexpected = sorted(set(predictions["model"]) - allowed_models)
    if unexpected:
        raise RuntimeError(f"Unexpected models in V4.0 output: {unexpected}")

    formal = predictions.loc[predictions["model"].isin(formal_models)].copy()
    common = common_model_sample(formal, formal_models)
    common_mase = common_model_sample(formal, formal_models, require_mase=True)

    summaries = {
        "native_by_horizon_model": _rank_summary(
            formal, ["horizon_label", "horizon_weeks", "model"]
        ),
        "common_by_horizon_model": _rank_summary(
            common, ["horizon_label", "horizon_weeks", "model"]
        ),
        "common_mase_by_horizon_model": _rank_summary(
            common_mase, ["horizon_label", "horizon_weeks", "model"]
        ),
        "common_by_origin_model": _rank_summary(
            common,
            ["horizon_label", "horizon_weeks", "origin_index", "origin", "model"],
        ),
        "common_by_cluster_model": _rank_summary(
            common,
            [
                "horizon_label",
                "horizon_weeks",
                "cluster",
                "cluster_profile",
                "model",
            ],
        ),
        "historical_origin6_common": _rank_summary(
            common.loc[common["origin_index"].eq(6)],
            ["horizon_label", "horizon_weeks", "model"],
        ),
    }
    for name, frame in summaries.items():
        frame.to_csv(output_root / f"{name}.csv", index=False)

    baselines = list(config["forecast"]["comparison_models"])
    paired = _paired_comparisons(
        common,
        common_mase,
        baselines=baselines,
        repetitions=int(config["evaluation"]["paired_bootstrap_repetitions"]),
        seed=int(config["project"]["seed"]),
    )
    head_to_head = _origin_head_to_head(common, baselines)
    gate_config = config["evaluation"]["predictability_gate"]
    gate = _predictability_gate(
        head_to_head,
        paired,
        required_baselines=list(gate_config["required_baselines"]),
        minimum_winning_origins=int(gate_config["minimum_winning_origins"]),
    )
    sku_head_to_head = _sku_head_to_head(common, baselines)

    pxq_rows = formal.loc[formal["model"].eq("PXQ")].copy()
    pxq_components = (
        pxq_rows.groupby(
            ["horizon_label", "horizon_weeks", "cluster", "cluster_profile"],
            dropna=False,
        )
        .agg(
            n_sku_origins=("sku", "size"),
            p_median=("pxq_p_hat", "median"),
            p_mean=("pxq_p_hat", "mean"),
            q_median=("pxq_q_hat", "median"),
            q_mean=("pxq_q_hat", "mean"),
            weekly_rate_median=("pxq_weekly_rate", "median"),
            training_positive_weeks_median=("pxq_training_positive_weeks", "median"),
        )
        .reset_index()
    )

    wins_frames = []
    for horizon_label, frame in common.groupby("horizon_label"):
        wins = model_wins(frame, loss_column="horizon_total_abs_error")
        wins.insert(0, "horizon_label", horizon_label)
        wins_frames.append(wins)
    model_win_table = pd.concat(wins_frames, ignore_index=True)

    predictions.to_csv(output_root / "rolling_origin_predictions.csv", index=False)
    assignments.to_csv(output_root / "rolling_origin_cluster_assignments.csv", index=False)
    audits.to_csv(output_root / "rolling_origin_audits.csv", index=False)
    common.to_csv(output_root / "five_model_common_sample.csv", index=False)
    common_mase.to_csv(output_root / "five_model_common_mase_sample.csv", index=False)
    paired.to_csv(output_root / "paired_pxq_comparisons.csv", index=False)
    head_to_head.to_csv(output_root / "origin_pxq_head_to_head.csv", index=False)
    gate.to_csv(output_root / "pxq_predictability_gate.csv", index=False)
    sku_head_to_head.to_csv(output_root / "sku_origin_pxq_head_to_head.csv", index=False)
    pxq_rows.to_csv(output_root / "pxq_prediction_components.csv", index=False)
    pxq_components.to_csv(output_root / "pxq_component_summary.csv", index=False)
    model_win_table.to_csv(output_root / "model_wins_by_sku.csv", index=False)

    best_common = summaries["common_by_horizon_model"].sort_values(
        ["horizon_label", "horizon_total_wape", "mean_mase", "model"]
    ).drop_duplicates("horizon_label")
    outcome = {
        "analysis_mode": config["project"]["analysis_mode"],
        "confirmatory": False,
        "protocol_version": config["project"]["protocol_version"],
        "raw_sha256": loaded.audit["sha256"],
        "unique_skus": int(loaded.audit["unique_skus"]),
        "origin_map": origin_map,
        "models": formal_models,
        "prediction_rows": int(len(predictions)),
        "common_sku_origins": int(
            common.drop_duplicates(["horizon_label", "origin_index", "sku"]).shape[0]
        ),
        "common_mase_sku_origins": int(
            common_mase.drop_duplicates(
                ["horizon_label", "origin_index", "sku"]
            ).shape[0]
        ),
        "best_common_model_by_horizon_total_wape": {
            str(row.horizon_label): str(row.model)
            for row in best_common.itertuples(index=False)
        },
        "pxq_value_supported": json.loads(gate.to_json(orient="records")),
        "historical_origin6_is_independent_holdout": False,
        "inventory_or_profit_claims_supported": False,
    }
    _write_json(output_root / "pxq_validation_outcome.json", outcome)
    return outcome


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run frozen V4.0 unified p-times-q historical validation."
    )
    parser.add_argument("--config", default="config/pxq_validation_v4.yaml")
    args = parser.parse_args()
    print(
        json.dumps(
            run_pxq_validation(args.config),
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
