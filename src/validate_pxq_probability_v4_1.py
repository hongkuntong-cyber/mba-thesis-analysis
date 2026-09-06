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
    "probability_components.csv",
    "rolling_probability_audits.csv",
    "probability_predictions.csv",
    "common_probability_sample.csv",
    "summary_by_horizon_model.csv",
    "summary_by_origin_model.csv",
    "summary_by_profile_model.csv",
    "native_coverage.csv",
    "reliability_bins.csv",
    "paired_brier_comparisons.csv",
    "origin_brier_head_to_head.csv",
    "probability_value_gate.csv",
    "probability_validation_outcome.json",
    "figures/pxq_independence_reliability.png",
]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
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
    for (label, weeks, method), frame in common.groupby(
        ["horizon_label", "horizon_weeks", "method"], sort=True
    ):
        target = frame["actual_event"].to_numpy(dtype=float)
        probability = frame["probability"].to_numpy(dtype=float)
        rows.append(
            {
                "horizon_label": label,
                "horizon_weeks": int(weeks),
                "method": method,
                "n_sku_origins": int(len(frame)),
                "n_unique_skus": int(frame["sku"].nunique()),
                "observed_event_rate": float(target.mean()),
                "mean_probability": float(probability.mean()),
                "calibration_gap": float(probability.mean() - target.mean()),
                "brier_score": float(np.square(probability - target).mean()),
                "zero_probability_share": float(np.mean(probability == 0.0)),
                "one_probability_share": float(np.mean(probability == 1.0)),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["horizon_label", "horizon_weeks", "method"]
    ).reset_index(drop=True)


def validate_probability_outputs(config_path: str | Path) -> dict[str, Any]:
    config_file = Path(config_path).resolve()
    project_root = config_file.parent.parent
    config = load_config(config_file)
    output_root = project_root / config["outputs"]["root"]
    checks: list[dict[str, Any]] = []

    partial_outputs = (
        sorted(path.name for path in output_root.iterdir() if path.name.endswith(".tmp"))
        if output_root.exists()
        else []
    )
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
        _write_json(output_root / "validation_summary.json", summary)
        return summary

    actual_sha = sha256_file(project_root / config["input"]["workbook"])
    expected_sha = str(config["input"]["expected_sha256"])
    _record(
        checks,
        "raw_sha256_matches_protocol",
        actual_sha == expected_sha,
        {"actual": actual_sha, "expected": expected_sha},
    )

    v4_outcome = json.loads(
        (project_root / config["input"]["v4_outcome"]).read_text(encoding="utf-8")
    )
    outcome = json.loads(
        (output_root / "probability_validation_outcome.json").read_text(encoding="utf-8")
    )
    v4_dependency_ok = (
        str(v4_outcome.get("protocol_version")) == "4.0"
        and str(outcome.get("protocol_version")) == "4.1"
        and outcome.get("reused_v4_quantity_models_without_rerun") is True
        and outcome.get("confirmatory") is False
    )
    _record(
        checks,
        "v4_dependency_and_retrospective_boundary",
        v4_dependency_ok,
        {
            "v4_protocol": v4_outcome.get("protocol_version"),
            "current_protocol": outcome.get("protocol_version"),
            "confirmatory": outcome.get("confirmatory"),
        },
    )

    components = pd.read_csv(output_root / "probability_components.csv")
    predictions = pd.read_csv(output_root / "probability_predictions.csv")
    common = pd.read_csv(output_root / "common_probability_sample.csv")
    audit = pd.read_csv(output_root / "rolling_probability_audits.csv")
    paired = pd.read_csv(output_root / "paired_brier_comparisons.csv")
    head = pd.read_csv(output_root / "origin_brier_head_to_head.csv")
    gate = pd.read_csv(output_root / "probability_value_gate.csv")
    reliability = pd.read_csv(output_root / "reliability_bins.csv")

    origin_details: dict[str, Any] = {}
    origin_structure_ok = True
    for horizon in config["probability"]["horizons"]:
        label = str(horizon["label"])
        weeks = int(horizon["weeks"])
        expected_origins = int(horizon["origins"])
        current = audit.loc[audit["horizon_label"].eq(label)].sort_values("origin_index")
        origins = pd.to_datetime(current["origin"])
        index_ok = current["origin_index"].tolist() == list(
            range(1, expected_origins + 1)
        )
        spacing_ok = len(origins) == expected_origins and (
            len(origins) <= 1
            or np.all(
                np.diff(origins.to_numpy()).astype("timedelta64[D]").astype(int)
                == 7 * weeks
            )
        )
        weeks_ok = len(current) == expected_origins and current[
            "horizon_weeks"
        ].eq(weeks).all()
        passed = bool(index_ok and spacing_ok and weeks_ok)
        origin_structure_ok = origin_structure_ok and passed
        origin_details[label] = {
            "origins": origins.dt.date.astype(str).tolist(),
            "passed": passed,
        }
    _record(
        checks,
        "six_frozen_nonoverlapping_origins_per_horizon",
        origin_structure_ok,
        origin_details,
    )

    probability_columns = [
        "pxq_independence_probability",
        "sku_block_probability",
        "profile_block_probability",
        "overall_block_probability",
    ]
    probability_bounds = bool(all(
        components[column].dropna().between(0.0, 1.0).all()
        for column in probability_columns
    ) and predictions["probability"].between(0.0, 1.0).all())
    expected_pxq_probability = 1.0 - np.power(
        1.0 - components["p_recent"], components["horizon_weeks"]
    )
    formula_ok = _all_close(
        components["pxq_independence_probability"], expected_pxq_probability
    )
    block_counts_ok = bool(
        components["positive_history_blocks"].ge(0).all()
        and components["complete_history_blocks"].ge(
            components["positive_history_blocks"]
        ).all()
        and components["complete_history_blocks"].ge(1).all()
    )
    _record(
        checks,
        "probability_formulas_and_bounds_reconcile",
        probability_bounds and formula_ok and block_counts_ok,
        {
            "probabilities_in_unit_interval": probability_bounds,
            "iid_conversion_matches": formula_ok,
            "block_counts_valid": block_counts_ok,
        },
    )

    target_ok = set(predictions["actual_event"].unique()).issubset({0, 1}) and bool(
        predictions["actual_event"].eq(predictions["actual_sum"].gt(0).astype(int)).all()
    )
    brier_ok = _all_close(
        predictions["brier_loss"],
        np.square(predictions["probability"] - predictions["actual_event"]),
    )
    _record(
        checks,
        "binary_target_and_row_losses_reconcile",
        target_ok and brier_ok,
        {"target_matches_raw_period_total": target_ok, "brier_matches": brier_ok},
    )

    methods = list(config["probability"]["methods"])
    key_columns = ["horizon_label", "origin_index", "sku"]
    method_sets = common.groupby(key_columns)["method"].agg(lambda values: set(values))
    method_counts = common.groupby(key_columns).size()
    common_ok = bool(
        len(method_sets)
        and method_sets.map(lambda values: values == set(methods)).all()
        and method_counts.eq(len(methods)).all()
    )
    row_identity = [*key_columns, "method"]
    detail_keys = set(
        map(tuple, predictions[row_identity].itertuples(index=False, name=None))
    )
    common_keys = set(map(tuple, common[row_identity].itertuples(index=False, name=None)))
    common_subset = common_keys.issubset(detail_keys)
    reported_common_keys = int(outcome["common_sku_origins"])
    observed_common_keys = int(len(method_sets))
    _record(
        checks,
        "four_method_common_sample_is_exact",
        common_ok and common_subset and reported_common_keys == observed_common_keys,
        {
            "observed_sku_origins": observed_common_keys,
            "reported_sku_origins": reported_common_keys,
            "common_is_detail_subset": common_subset,
        },
    )

    reported = pd.read_csv(output_root / "summary_by_horizon_model.csv").sort_values(
        ["horizon_label", "horizon_weeks", "method"]
    ).reset_index(drop=True)
    independent = _independent_summary(common)
    keys = ["horizon_label", "horizon_weeks", "method"]
    keys_ok = reported[keys].equals(independent[keys])
    numeric_columns = [
        "n_sku_origins",
        "n_unique_skus",
        "observed_event_rate",
        "mean_probability",
        "calibration_gap",
        "brier_score",
        "zero_probability_share",
        "one_probability_share",
    ]
    summaries_ok = keys_ok and all(
        _all_close(reported[column], independent[column]) for column in numeric_columns
    )
    _record(
        checks,
        "published_probability_summary_reconciles",
        summaries_ok,
        {"rows": int(len(reported))},
    )

    reliability_expected_rows = len(common)
    reliability_observed_rows = int(reliability["n_sku_origins"].sum())
    bin_bounds_ok = bool(
        reliability["bin_lower"].between(0.0, 1.0).all()
        and reliability["bin_upper"].between(0.0, 1.0).all()
        and reliability["bin_upper"].gt(reliability["bin_lower"]).all()
    )
    _record(
        checks,
        "reliability_bins_cover_common_predictions",
        reliability_observed_rows == reliability_expected_rows and bin_bounds_ok,
        {
            "observed_rows": reliability_observed_rows,
            "expected_rows": reliability_expected_rows,
            "valid_bounds": bin_bounds_ok,
        },
    )

    expected_baselines = set(methods) - {"PXQ_independence"}
    expected_horizons = {
        str(horizon["label"]) for horizon in config["probability"]["horizons"]
    }
    paired_pairs = set(zip(paired["horizon_label"], paired["baseline"]))
    expected_pairs = {
        (label, baseline)
        for label in expected_horizons
        for baseline in expected_baselines
    }
    head_counts = head.groupby(["horizon_label", "baseline"]).size()
    head_ok = bool(
        set(head_counts.index) == expected_pairs and head_counts.eq(6).all()
    )
    gate_ok = bool(
        set(gate["horizon_label"]) == expected_horizons
        and len(gate) == len(expected_horizons)
        and gate["baseline"].eq(
            config["evaluation"]["probability_value_gate"]["baseline"]
        ).all()
    )
    _record(
        checks,
        "paired_origin_and_gate_structures_are_complete",
        paired_pairs == expected_pairs and head_ok and gate_ok,
        {
            "paired_rows": int(len(paired)),
            "head_rows": int(len(head)),
            "gate_rows": int(len(gate)),
        },
    )

    reconciliation = outcome["v4_p_reconciliation"]
    reconciliation_ok = bool(
        reconciliation["reconciled"]
        and int(reconciliation["v4_nonmissing_p_rows_compared"]) > 0
        and float(reconciliation["v4_p_max_absolute_difference"]) <= 1e-12
    )
    _record(
        checks,
        "new_weekly_rates_reconcile_to_v4_when_available",
        reconciliation_ok,
        reconciliation,
    )

    expected_prediction_rows = int(outcome["probability_prediction_rows"])
    count_ok = expected_prediction_rows == len(predictions)
    _record(
        checks,
        "outcome_row_count_reconciles",
        count_ok,
        {"reported": expected_prediction_rows, "observed": int(len(predictions))},
    )

    valid = bool(all(item["passed"] for item in checks))
    summary = {
        "valid": valid,
        "protocol_version": config["project"]["protocol_version"],
        "checks_passed": int(sum(item["passed"] for item in checks)),
        "checks_total": int(len(checks)),
        "checks": checks,
    }
    _write_json(output_root / "validation_summary.json", summary)

    manifest_rows: list[dict[str, str]] = []
    for path in sorted(output_root.rglob("*")):
        if path.is_file() and path.name != "manifest_sha256.csv":
            manifest_rows.append(
                {
                    "relative_path": str(path.relative_to(project_root)),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
    manifest_path = output_root / "manifest_sha256.csv"
    temporary = manifest_path.with_name(f".{manifest_path.name}.{os.getpid()}.tmp")
    pd.DataFrame(manifest_rows).to_csv(temporary, index=False)
    temporary.replace(manifest_path)
    if not valid:
        failed = [item["check"] for item in checks if not item["passed"]]
        raise RuntimeError(f"V4.1 probability output validation failed: {failed}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate frozen V4.1 occurrence-probability outputs."
    )
    parser.add_argument("--config", default="config/pxq_probability_v4_1.yaml")
    args = parser.parse_args()
    print(
        json.dumps(
            validate_probability_outputs(args.config),
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
