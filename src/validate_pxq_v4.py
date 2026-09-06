from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import load_config
from .data_audit import sha256_file


REQUIRED_OUTPUTS = [
    "input_audit.json",
    "rolling_origin_predictions.csv",
    "rolling_origin_cluster_assignments.csv",
    "rolling_origin_audits.csv",
    "five_model_common_sample.csv",
    "five_model_common_mase_sample.csv",
    "common_by_horizon_model.csv",
    "common_mase_by_horizon_model.csv",
    "common_by_origin_model.csv",
    "common_by_cluster_model.csv",
    "historical_origin6_common.csv",
    "paired_pxq_comparisons.csv",
    "origin_pxq_head_to_head.csv",
    "pxq_predictability_gate.csv",
    "sku_origin_pxq_head_to_head.csv",
    "pxq_prediction_components.csv",
    "pxq_component_summary.csv",
    "model_wins_by_sku.csv",
    "pxq_validation_outcome.json",
]


def _json_write(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    temporary.replace(path)


def _record(checks: list[dict[str, Any]], name: str, passed: bool, detail: Any) -> None:
    checks.append({"check": name, "passed": bool(passed), "detail": detail})


def _all_close(left: pd.Series, right: pd.Series) -> bool:
    return bool(
        np.allclose(
            left.to_numpy(dtype=float),
            right.to_numpy(dtype=float),
            rtol=1e-9,
            atol=1e-10,
            equal_nan=True,
        )
    )


def _independent_summary(common: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (label, weeks, model), frame in common.groupby(
        ["horizon_label", "horizon_weeks", "model"], sort=True
    ):
        valid_mase = frame["mase"].dropna()
        actual = float(frame["actual_sum"].sum())
        rows.append(
            {
                "horizon_label": label,
                "horizon_weeks": int(weeks),
                "model": model,
                "n_sku_origins": int(len(frame)),
                "n_valid_mase": int(len(valid_mase)),
                "mean_mase": float(valid_mase.mean()) if len(valid_mase) else np.nan,
                "median_mase": float(valid_mase.median()) if len(valid_mase) else np.nan,
                "aggregate_wape": float(frame["abs_error_sum"].sum() / actual),
                "horizon_total_wape": float(
                    frame["horizon_total_abs_error"].sum() / actual
                ),
                "actual_volume": actual,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["horizon_label", "horizon_weeks", "model"]
    ).reset_index(drop=True)


def validate_pxq_outputs(config_path: str | Path) -> dict[str, Any]:
    config_file = Path(config_path).resolve()
    project_root = config_file.parent.parent
    config = load_config(config_file)
    output_root = project_root / config["outputs"]["root"]
    checks: list[dict[str, Any]] = []

    partial_outputs = sorted(
        path.name for path in output_root.iterdir() if path.name.endswith(".tmp")
    ) if output_root.exists() else []
    _record(
        checks,
        "no_partial_output_files_remain",
        not partial_outputs,
        {"partial_outputs": partial_outputs},
    )

    missing = [name for name in REQUIRED_OUTPUTS if not (output_root / name).is_file()]
    _record(checks, "required_outputs_exist", not missing, {"missing": missing})
    if missing:
        summary = {"valid": False, "checks": checks}
        output_root.mkdir(parents=True, exist_ok=True)
        _json_write(output_root / "validation_summary.json", summary)
        return summary

    raw_path = project_root / config["input"]["workbook"]
    actual_sha = sha256_file(raw_path)
    expected_sha = config["input"]["expected_sha256"]
    _record(
        checks,
        "raw_sha256_matches_protocol",
        actual_sha == expected_sha,
        {"actual": actual_sha, "expected": expected_sha},
    )

    predictions = pd.read_csv(output_root / "rolling_origin_predictions.csv")
    common = pd.read_csv(output_root / "five_model_common_sample.csv")
    common_mase = pd.read_csv(output_root / "five_model_common_mase_sample.csv")
    audit = pd.read_csv(output_root / "rolling_origin_audits.csv")
    paired = pd.read_csv(output_root / "paired_pxq_comparisons.csv")
    gate = pd.read_csv(output_root / "pxq_predictability_gate.csv")
    outcome = json.loads((output_root / "pxq_validation_outcome.json").read_text(encoding="utf-8"))

    expected_prediction_rows = int(outcome["prediction_rows"])
    observed_horizons = set(predictions["horizon_label"].unique())
    expected_horizons = {str(item["label"]) for item in config["forecast"]["horizons"]}
    _record(
        checks,
        "prediction_detail_is_complete",
        len(predictions) == expected_prediction_rows and observed_horizons == expected_horizons,
        {
            "observed_rows": int(len(predictions)),
            "expected_rows": expected_prediction_rows,
            "observed_horizons": sorted(observed_horizons),
            "expected_horizons": sorted(expected_horizons),
        },
    )

    formal_models = [
        config["forecast"]["primary_model"],
        *config["forecast"]["comparison_models"],
    ]
    allowed_models = set(formal_models) | {config["forecast"]["value_baseline"]}
    observed_models = set(predictions["model"].unique())
    _record(
        checks,
        "only_frozen_models_present",
        observed_models.issubset(allowed_models) and set(formal_models).issubset(observed_models),
        {"observed": sorted(observed_models), "allowed": sorted(allowed_models)},
    )

    horizon_details: dict[str, Any] = {}
    origin_structure_passed = True
    for horizon in config["forecast"]["horizons"]:
        label = str(horizon["label"])
        weeks = int(horizon["weeks"])
        expected_origins = int(horizon["origins"])
        rows = audit.loc[audit["horizon_label"].eq(label)].sort_values("origin_index")
        origins = pd.to_datetime(rows["origin"])
        index_ok = rows["origin_index"].tolist() == list(range(1, expected_origins + 1))
        spacing_ok = len(origins) == expected_origins and (
            len(origins) <= 1
            or np.all(np.diff(origins.to_numpy()).astype("timedelta64[D]").astype(int) == 7 * weeks)
        )
        weeks_ok = len(rows) == expected_origins and rows["horizon_weeks"].eq(weeks).all()
        current = bool(index_ok and spacing_ok and weeks_ok)
        origin_structure_passed = origin_structure_passed and current
        horizon_details[label] = {
            "rows": int(len(rows)),
            "origins": origins.dt.date.astype(str).tolist(),
            "passed": current,
        }
    _record(checks, "six_nonoverlapping_origins_per_horizon", origin_structure_passed, horizon_details)

    pxq = predictions.loc[predictions["model"].eq("PXQ")].copy()
    expected_lookback = pxq["horizon_weeks"].astype(float)
    pxq_checks = {
        "p_in_unit_interval": bool(pxq["pxq_p_hat"].between(0.0, 1.0).all()),
        "q_positive_finite": bool(
            np.isfinite(pxq["pxq_q_hat"]).all() and pxq["pxq_q_hat"].gt(0).all()
        ),
        "lookback_equals_horizon": _all_close(pxq["pxq_lookback_weeks"], expected_lookback),
        "weekly_rate_equals_p_times_q": _all_close(
            pxq["pxq_weekly_rate"], pxq["pxq_p_hat"] * pxq["pxq_q_hat"]
        ),
        "forecast_sum_equals_h_times_p_times_q": _all_close(
            pxq["forecast_sum"],
            pxq["horizon_weeks"] * pxq["pxq_p_hat"] * pxq["pxq_q_hat"],
        ),
    }
    _record(checks, "pxq_frozen_formula_reconciles", all(pxq_checks.values()), pxq_checks)

    total_error = (predictions["forecast_sum"] - predictions["actual_sum"]).abs()
    total_metrics_ok = (
        _all_close(predictions["horizon_total_abs_error"], total_error)
        and _all_close(
            predictions["underforecast_units"],
            (predictions["actual_sum"] - predictions["forecast_sum"]).clip(lower=0),
        )
        and _all_close(
            predictions["overforecast_units"],
            (predictions["forecast_sum"] - predictions["actual_sum"]).clip(lower=0),
        )
    )
    _record(checks, "horizon_total_metrics_reconcile", total_metrics_ok, {"rows": len(predictions)})

    key_columns = ["horizon_label", "origin_index", "sku"]
    common_sets = common.groupby(key_columns)["model"].agg(lambda values: set(values))
    common_counts = common.groupby(key_columns).size()
    common_ok = bool(
        len(common_sets)
        and common_sets.map(lambda values: values == set(formal_models)).all()
        and common_counts.eq(len(formal_models)).all()
    )
    mase_sets = common_mase.groupby(key_columns)["model"].agg(lambda values: set(values))
    mase_counts = common_mase.groupby(key_columns).size()
    common_mase_ok = bool(
        len(mase_sets)
        and mase_sets.map(lambda values: values == set(formal_models)).all()
        and mase_counts.eq(len(formal_models)).all()
        and common_mase["mase"].notna().all()
    )
    row_identity = ["horizon_label", "origin_index", "sku", "model"]
    prediction_keys = set(map(tuple, predictions[row_identity].itertuples(index=False, name=None)))
    common_keys = set(map(tuple, common[row_identity].itertuples(index=False, name=None)))
    common_is_detail_subset = common_keys.issubset(prediction_keys)
    _record(
        checks,
        "five_model_common_samples_are_exact",
        common_ok and common_mase_ok and common_is_detail_subset,
        {
            "common_keys": int(len(common_sets)),
            "common_mase_keys": int(len(mase_sets)),
            "common_rows_are_prediction_subset": common_is_detail_subset,
        },
    )

    reported = pd.read_csv(output_root / "common_by_horizon_model.csv").sort_values(
        ["horizon_label", "horizon_weeks", "model"]
    ).reset_index(drop=True)
    independent = _independent_summary(common)
    summary_keys = ["horizon_label", "horizon_weeks", "model"]
    keys_ok = reported[summary_keys].equals(independent[summary_keys])
    numeric_columns = [
        "n_sku_origins",
        "n_valid_mase",
        "mean_mase",
        "median_mase",
        "aggregate_wape",
        "horizon_total_wape",
        "actual_volume",
    ]
    metrics_ok = keys_ok and all(
        _all_close(reported[column], independent[column]) for column in numeric_columns
    )
    _record(checks, "published_common_summary_reconciles", metrics_ok, {"rows": len(reported)})

    expected_scopes = {"all", "low_information", "relative_active_persistent", "low_frequency_sparse"}
    expected_losses = {"mase", "horizon_total_abs_error"}
    expected_baselines = set(config["forecast"]["comparison_models"])
    comparison_structure_ok = True
    gate_structure_ok = True
    for horizon in config["forecast"]["horizons"]:
        label = str(horizon["label"])
        subset = paired.loc[paired["horizon_label"].eq(label)]
        triples = set(zip(subset["scope"], subset["baseline"], subset["loss"]))
        expected_triples = {
            (scope, baseline, loss)
            for scope in expected_scopes
            for baseline in expected_baselines
            for loss in expected_losses
        }
        comparison_structure_ok = comparison_structure_ok and triples == expected_triples
        gate_scopes = set(gate.loc[gate["horizon_label"].eq(label), "scope"])
        gate_structure_ok = gate_structure_ok and gate_scopes == expected_scopes
    _record(
        checks,
        "paired_and_gate_tables_cover_all_frozen_scopes",
        comparison_structure_ok and gate_structure_ok,
        {"paired_rows": int(len(paired)), "gate_rows": int(len(gate))},
    )

    valid = bool(all(item["passed"] for item in checks))
    summary = {
        "valid": valid,
        "protocol_version": config["project"]["protocol_version"],
        "checks_passed": int(sum(item["passed"] for item in checks)),
        "checks_total": int(len(checks)),
        "checks": checks,
    }
    _json_write(output_root / "validation_summary.json", summary)

    manifest_rows = []
    for path in sorted(output_root.glob("*")):
        if path.is_file() and path.name != "manifest_sha256.csv":
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            manifest_rows.append(
                {"relative_path": str(path.relative_to(project_root)), "sha256": digest}
            )
    manifest_path = output_root / "manifest_sha256.csv"
    manifest_temporary = manifest_path.with_name(
        f".{manifest_path.name}.{os.getpid()}.tmp"
    )
    pd.DataFrame(manifest_rows).to_csv(manifest_temporary, index=False)
    manifest_temporary.replace(manifest_path)
    if not valid:
        failed = [item["check"] for item in checks if not item["passed"]]
        raise RuntimeError(f"V4.0 output validation failed: {failed}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate frozen V4.0 p-times-q outputs.")
    parser.add_argument("--config", default="config/pxq_validation_v4.yaml")
    args = parser.parse_args()
    print(json.dumps(validate_pxq_outputs(args.config), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
