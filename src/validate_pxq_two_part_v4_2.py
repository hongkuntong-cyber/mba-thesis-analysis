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
from .pxq_two_part_v4_2 import (
    PRIMARY_PROBABILITY_METHOD,
    PRIMARY_QUANTITY_METHOD,
    _common_method_sample,
    _summarize_probabilities,
    _summarize_quantities,
)


REQUIRED_OUTPUTS = [
    "input_audit.json",
    "rolling_origin_audits.csv",
    "two_part_components.csv",
    "probability_predictions.csv",
    "expected_quantity_predictions.csv",
    "probability_summary.csv",
    "quantity_summary.csv",
    "conditional_quantity_summary.csv",
    "summary_by_profile.csv",
    "reliability_bins.csv",
    "paired_probability_comparisons.csv",
    "paired_quantity_comparisons.csv",
    "origin_head_to_head.csv",
    "value_gates.csv",
    "two_part_outcome.json",
]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    temporary.replace(path)


def _record(
    checks: list[dict[str, Any]], check: str, passed: bool, details: dict[str, Any]
) -> None:
    checks.append({"check": check, "passed": bool(passed), "details": details})


def _all_close(left: pd.Series, right: pd.Series, tolerance: float = 1e-10) -> bool:
    a = pd.to_numeric(left, errors="coerce").to_numpy(dtype=float)
    b = pd.to_numeric(right, errors="coerce").to_numpy(dtype=float)
    return bool(np.allclose(a, b, rtol=tolerance, atol=tolerance, equal_nan=True))


def validate_two_part_outputs(config_path: str | Path) -> dict[str, Any]:
    config_file = Path(config_path).resolve()
    project_root = config_file.parent.parent
    config = load_config(config_file)
    output_root = project_root / config["outputs"]["root"]
    report_path = project_root / config["outputs"]["report"]
    checks: list[dict[str, Any]] = []

    partial = (
        sorted(path.name for path in output_root.iterdir() if path.name.endswith(".tmp"))
        if output_root.exists()
        else []
    )
    _record(checks, "no_partial_output_files_remain", not partial, {"partial": partial})

    missing = [name for name in REQUIRED_OUTPUTS if not (output_root / name).is_file()]
    _record(
        checks,
        "required_outputs_and_report_exist",
        not missing and report_path.is_file(),
        {"missing": missing, "report_exists": report_path.is_file()},
    )
    if missing or not report_path.is_file():
        summary = {
            "valid": False,
            "protocol_version": config["project"]["protocol_version"],
            "checks_passed": int(sum(item["passed"] for item in checks)),
            "checks_total": int(len(checks)),
            "checks": checks,
        }
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
    v41_outcome = json.loads(
        (project_root / config["input"]["v4_1_outcome"]).read_text(encoding="utf-8")
    )
    outcome = json.loads((output_root / "two_part_outcome.json").read_text(encoding="utf-8"))
    dependency_ok = bool(
        str(v4_outcome.get("protocol_version")) == "4.0"
        and str(v41_outcome.get("protocol_version")) == "4.1"
        and str(outcome.get("protocol_version")) == "4.2"
        and outcome.get("analysis_mode") == "retrospective_method_development"
        and outcome.get("confirmatory") is False
        and outcome.get("reused_prior_results_without_rerun") is True
    )
    _record(
        checks,
        "dependencies_and_retrospective_boundary_are_explicit",
        dependency_ok,
        {
            "v4": v4_outcome.get("protocol_version"),
            "v4_1": v41_outcome.get("protocol_version"),
            "current": outcome.get("protocol_version"),
            "confirmatory": outcome.get("confirmatory"),
        },
    )

    components = pd.read_csv(output_root / "two_part_components.csv")
    audits = pd.read_csv(output_root / "rolling_origin_audits.csv")
    probabilities = pd.read_csv(output_root / "probability_predictions.csv")
    quantities = pd.read_csv(output_root / "expected_quantity_predictions.csv")
    probability_summary = pd.read_csv(output_root / "probability_summary.csv")
    quantity_summary = pd.read_csv(output_root / "quantity_summary.csv")
    reliability = pd.read_csv(output_root / "reliability_bins.csv")
    paired_probability = pd.read_csv(output_root / "paired_probability_comparisons.csv")
    paired_quantity = pd.read_csv(output_root / "paired_quantity_comparisons.csv")
    origin_head = pd.read_csv(output_root / "origin_head_to_head.csv")
    gates = pd.read_csv(output_root / "value_gates.csv")

    origin_ok = True
    origin_details: dict[str, Any] = {}
    for horizon in config["two_part"]["horizons"]:
        label = str(horizon["label"])
        weeks = int(horizon["weeks"])
        expected_count = int(horizon["origins"])
        current = audits.loc[audits["horizon_label"].eq(label)].sort_values("origin_index")
        origin_dates = pd.to_datetime(current["origin"])
        passed = bool(
            len(current) == expected_count
            and current["origin_index"].tolist() == list(range(1, expected_count + 1))
            and current["horizon_weeks"].eq(weeks).all()
            and (
                len(origin_dates) <= 1
                or np.all(
                    np.diff(origin_dates.to_numpy())
                    .astype("timedelta64[D]")
                    .astype(int)
                    == 7 * weeks
                )
            )
        )
        origin_ok = origin_ok and passed
        origin_details[label] = {
            "origins": origin_dates.dt.date.astype(str).tolist(),
            "passed": passed,
        }
    _record(
        checks,
        "six_frozen_nonoverlapping_origins_per_horizon",
        origin_ok,
        origin_details,
    )

    lookback = int(config["two_part"]["training_lookback_weeks"])
    expected_blocks = (
        components[["training_weeks", "horizon_weeks"]]
        .assign(training=lambda frame: frame["training_weeks"].clip(upper=lookback))
        .eval("training // horizon_weeks")
        .astype(int)
    )
    blocks_ok = bool(
        components["complete_history_blocks"].astype(int).eq(expected_blocks).all()
        and components["positive_history_blocks"].ge(0).all()
        and components["positive_history_blocks"]
        .le(components["complete_history_blocks"])
        .all()
        and components["positive_history_volume"].ge(0).all()
        and components["all_history_volume"]
        .ge(components["positive_history_volume"] - 1e-12)
        .all()
    )
    _record(
        checks,
        "fixed_52_week_nonoverlapping_block_counts_reconcile",
        blocks_ok,
        {
            "lookback_weeks": lookback,
            "component_rows": int(len(components)),
            "zero_own_block_rows": int(components["complete_history_blocks"].eq(0).sum()),
        },
    )

    target_ok = bool(
        components["actual_event"]
        .eq(components["actual_sum"].gt(0).astype(int))
        .all()
        and set(probabilities["actual_event"].unique()).issubset({0, 1})
    )
    probability_loss_ok = _all_close(
        probabilities["brier_loss"],
        np.square(probabilities["probability"] - probabilities["actual_event"]),
    )
    quantity_loss_ok = _all_close(
        quantities["horizon_total_abs_error"],
        (quantities["forecast_sum"] - quantities["actual_sum"]).abs(),
    )
    _record(
        checks,
        "targets_and_row_losses_reconcile",
        target_ok and probability_loss_ok and quantity_loss_ok,
        {
            "target": target_ok,
            "brier": probability_loss_ok,
            "quantity_absolute_error": quantity_loss_ok,
        },
    )

    origin_keys = ["horizon_label", "origin_index"]
    profile_keys = [*origin_keys, "cluster_profile"]
    numeric = [
        "complete_history_blocks",
        "positive_history_blocks",
        "positive_history_volume",
    ]
    expected_overall = components.groupby(origin_keys)[numeric].transform("sum") - components[numeric]
    expected_profile = components.groupby(profile_keys)[numeric].transform("sum") - components[numeric]
    loo_ok = bool(
        _all_close(components["enterprise_peer_blocks"], expected_overall["complete_history_blocks"])
        and _all_close(
            components["enterprise_peer_positive_blocks"],
            expected_overall["positive_history_blocks"],
        )
        and _all_close(
            components["enterprise_peer_positive_volume"],
            expected_overall["positive_history_volume"],
        )
        and _all_close(components["profile_peer_blocks"], expected_profile["complete_history_blocks"])
        and _all_close(
            components["profile_peer_positive_blocks"],
            expected_profile["positive_history_blocks"],
        )
        and _all_close(
            components["profile_peer_positive_volume"],
            expected_profile["positive_history_volume"],
        )
    )
    _record(
        checks,
        "leave_one_sku_out_peer_totals_reconcile",
        loo_ok,
        {"leave_one_sku_out": config["two_part"]["leave_one_sku_out_reference"]},
    )

    strengths = [
        int(config["two_part"]["primary_prior_equivalent_blocks"]),
        *map(int, config["two_part"]["sensitivity_prior_equivalent_blocks"]),
    ]
    formula_ok = True
    for strength in strengths:
        expected_p = (
            components["positive_history_blocks"]
            + strength * components["profile_reference_probability"]
        ) / (components["complete_history_blocks"] + strength)
        expected_q = (
            components["positive_history_volume"]
            + strength * components["profile_reference_conditional_quantity"]
        ) / (components["positive_history_blocks"] + strength)
        formula_ok = formula_ok and _all_close(
            components[f"profile_probability_l{strength}"], expected_p
        )
        formula_ok = formula_ok and _all_close(
            components[f"profile_conditional_quantity_l{strength}"], expected_q
        )
        formula_ok = formula_ok and _all_close(
            components[f"profile_expected_quantity_l{strength}"],
            expected_p * expected_q,
        )
    primary_strength = int(config["two_part"]["primary_prior_equivalent_blocks"])
    enterprise_p = (
        components["positive_history_blocks"]
        + primary_strength * components["enterprise_reference_probability"]
    ) / (components["complete_history_blocks"] + primary_strength)
    enterprise_q = (
        components["positive_history_volume"]
        + primary_strength * components["enterprise_reference_conditional_quantity"]
    ) / (components["positive_history_blocks"] + primary_strength)
    formula_ok = formula_ok and _all_close(
        components[f"enterprise_probability_l{primary_strength}"], enterprise_p
    )
    formula_ok = formula_ok and _all_close(
        components[f"enterprise_expected_quantity_l{primary_strength}"],
        enterprise_p * enterprise_q,
    )
    _record(
        checks,
        "frozen_shrinkage_and_two_part_identities_reconcile",
        formula_ok,
        {"strengths": strengths},
    )

    bounds_ok = bool(
        probabilities["probability"].between(0.0, 1.0).all()
        and quantities["forecast_sum"].ge(0).all()
    )
    new_quantity_methods = [
        f"ProfileShrink_L{primary_strength}_expected",
        f"EnterpriseShrink_L{primary_strength}_expected",
        "SKU_recent_block_expected",
    ]
    no_invented_mase = bool(
        quantities.loc[quantities["method"].isin(new_quantity_methods), "existing_weekly_mase"]
        .isna()
        .all()
    )
    _record(
        checks,
        "predictions_have_valid_bounds_and_no_invented_weekly_mase",
        bounds_ok and no_invented_mase,
        {"bounds": bounds_ok, "new_method_mase_is_missing": no_invented_mase},
    )

    v41 = pd.read_csv(project_root / config["input"]["v4_1_probability_predictions"])
    v41 = v41.loc[v41["method"].eq("PXQ_independence")]
    keys = ["horizon_label", "origin_index", "sku"]
    merged = components.dropna(subset=["pxq_independence_probability"]).merge(
        v41[keys + ["probability"]], on=keys, how="inner", validate="one_to_one"
    )
    reuse_probability_ok = bool(
        len(merged) > 0
        and _all_close(merged["pxq_independence_probability"], merged["probability"])
    )
    v4 = pd.read_csv(project_root / config["input"]["v4_predictions"])
    existing_methods = {"PXQ", "MA4_proxy", "Naive", "SES", "ADIDA2"}
    existing_output = quantities.loc[quantities["method"].isin(existing_methods)]
    existing_merged = existing_output.merge(
        v4.loc[v4["model"].isin(existing_methods), keys + ["model", "forecast_sum"]],
        left_on=keys + ["method"],
        right_on=keys + ["model"],
        how="inner",
        suffixes=("_new", "_old"),
        validate="one_to_one",
    )
    reuse_quantity_ok = bool(
        len(existing_merged) == len(existing_output)
        and _all_close(existing_merged["forecast_sum_new"], existing_merged["forecast_sum_old"])
    )
    _record(
        checks,
        "v4_and_v4_1_predictions_are_reused_exactly",
        reuse_probability_ok and reuse_quantity_ok,
        {
            "probability_rows": int(len(merged)),
            "quantity_rows": int(len(existing_merged)),
        },
    )

    probability_methods = list(config["two_part"]["probability_methods"])
    quantity_methods = list(config["two_part"]["expected_quantity_methods"])
    probability_common = _common_method_sample(probabilities, probability_methods)
    quantity_common = _common_method_sample(quantities, quantity_methods)
    recomputed_probability = _summarize_probabilities(
        probability_common,
        ["horizon_label", "horizon_weeks", "method"],
        epsilon=float(config["evaluation"]["numerical_log_loss_epsilon"]),
    ).sort_values(["horizon_label", "method"]).reset_index(drop=True)
    reported_probability = probability_summary.sort_values(
        ["horizon_label", "method"]
    ).reset_index(drop=True)
    p_keys = ["horizon_label", "horizon_weeks", "method"]
    probability_summary_ok = bool(
        reported_probability[p_keys].equals(recomputed_probability[p_keys])
        and all(
            _all_close(reported_probability[column], recomputed_probability[column])
            for column in [
                "n_sku_origins",
                "n_unique_skus",
                "observed_event_rate",
                "mean_probability",
                "calibration_gap",
                "brier_score",
            ]
        )
    )
    recomputed_quantity = _summarize_quantities(
        quantity_common, ["horizon_label", "horizon_weeks", "method"]
    ).sort_values(["horizon_label", "method"]).reset_index(drop=True)
    reported_quantity = quantity_summary.sort_values(
        ["horizon_label", "method"]
    ).reset_index(drop=True)
    quantity_summary_ok = bool(
        reported_quantity[p_keys].equals(recomputed_quantity[p_keys])
        and all(
            _all_close(reported_quantity[column], recomputed_quantity[column])
            for column in [
                "n_sku_origins",
                "actual_volume",
                "forecast_volume",
                "mean_absolute_error",
                "median_absolute_error",
                "aggregate_wape",
                "aggregate_bias",
            ]
        )
    )
    _record(
        checks,
        "published_common_sample_summaries_reconcile",
        probability_summary_ok and quantity_summary_ok,
        {
            "probability_common_keys": int(
                probability_common[keys].drop_duplicates().shape[0]
            ),
            "quantity_common_keys": int(quantity_common[keys].drop_duplicates().shape[0]),
        },
    )

    reliability_expected = len(probability_common)
    reliability_actual = int(reliability["n_sku_origins"].sum())
    reliability_ok = bool(
        reliability_expected == reliability_actual
        and reliability["bin_lower"].between(0.0, 1.0).all()
        and reliability["bin_upper"].between(0.0, 1.0).all()
    )
    _record(
        checks,
        "reliability_bins_cover_probability_common_sample",
        reliability_ok,
        {"expected_rows": reliability_expected, "observed_rows": reliability_actual},
    )

    probability_baselines = set(probability_methods) - {PRIMARY_PROBABILITY_METHOD}
    quantity_baselines = set(quantity_methods) - {PRIMARY_QUANTITY_METHOD}
    horizon_labels = {str(item["label"]) for item in config["two_part"]["horizons"]}
    expected_probability_pairs = {
        (label, baseline) for label in horizon_labels for baseline in probability_baselines
    }
    expected_quantity_pairs = {
        (label, baseline) for label in horizon_labels for baseline in quantity_baselines
    }
    paired_probability_pairs = set(
        zip(paired_probability["horizon_label"], paired_probability["baseline"])
    )
    paired_quantity_pairs = set(
        zip(paired_quantity["horizon_label"], paired_quantity["baseline"])
    )
    origin_counts = origin_head.groupby(["target_type", "horizon_label", "baseline"]).size()
    origin_probability_ok = all(
        origin_counts.get(("probability", label, baseline), 0) == 6
        for label, baseline in expected_probability_pairs
    )
    origin_quantity_ok = all(
        origin_counts.get(("expected_quantity", label, baseline), 0) == 6
        for label, baseline in expected_quantity_pairs
    )
    comparisons_ok = bool(
        paired_probability_pairs == expected_probability_pairs
        and paired_quantity_pairs == expected_quantity_pairs
        and origin_probability_ok
        and origin_quantity_ok
    )
    _record(
        checks,
        "paired_and_origin_comparisons_are_complete",
        comparisons_ok,
        {
            "paired_probability_rows": int(len(paired_probability)),
            "paired_quantity_rows": int(len(paired_quantity)),
            "origin_rows": int(len(origin_head)),
        },
    )

    expected_gate_types = {
        "cluster_information_value",
        "probability_replacement_value",
        "expected_quantity_value",
    }
    gate_counts = gates.groupby("gate_type").size().to_dict()
    gates_ok = bool(
        set(gates["gate_type"]) == expected_gate_types
        and all(gate_counts.get(gate, 0) == len(horizon_labels) for gate in expected_gate_types)
        and set(gates["horizon_label"]) == horizon_labels
    )
    _record(
        checks,
        "all_frozen_value_gates_are_reported",
        gates_ok,
        {"gate_counts": gate_counts},
    )

    count_ok = bool(
        int(outcome["probability_rows"]) == len(probabilities)
        and int(outcome["expected_quantity_rows"]) == len(quantities)
        and int(outcome["probability_common_sku_origins"])
        == probability_common[keys].drop_duplicates().shape[0]
        and int(outcome["quantity_common_sku_origins"])
        == quantity_common[keys].drop_duplicates().shape[0]
        and int(outcome["pure_reference_sku_origins"])
        == components["complete_history_blocks"].eq(0).sum()
    )
    _record(
        checks,
        "outcome_counts_reconcile",
        count_ok,
        {
            "probability_rows": int(len(probabilities)),
            "quantity_rows": int(len(quantities)),
        },
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
    manifest_paths = [
        *sorted(path for path in output_root.rglob("*") if path.is_file()),
        report_path,
    ]
    for path in manifest_paths:
        if path.name == "manifest_sha256.csv":
            continue
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
        raise RuntimeError(f"V4.2 output validation failed: {failed}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate frozen V4.2 two-part outputs.")
    parser.add_argument("--config", default="config/pxq_two_part_v4_2.yaml")
    args = parser.parse_args()
    print(
        json.dumps(
            validate_two_part_outputs(args.config),
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
