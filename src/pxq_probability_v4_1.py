from __future__ import annotations

import argparse
import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from .backtesting import _ending_contiguous_training_segment
from .cleaning_v2 import apply_v2_cleaning
from .config import load_config
from .data_audit import load_workbook_long


METHODS = [
    "PXQ_independence",
    "SKU_block_frequency",
    "Profile_block_frequency",
    "Overall_block_frequency",
]
PRIMARY_METHOD = "PXQ_independence"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


@contextmanager
def _exclusive_output_lock(output_root: Path):
    output_root.mkdir(parents=True, exist_ok=True)
    stale = sorted(path.name for path in output_root.iterdir() if path.name.endswith(".tmp"))
    if stale:
        raise RuntimeError(f"Stale partial output exists; inspect before rerun: {stale}")
    lock_path = output_root / ".pxq_probability.lock"
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as error:
        raise RuntimeError(f"Another V4.1 run may already be writing {output_root}") from error
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(f"pid={os.getpid()}\n")
        yield
    finally:
        lock_path.unlink(missing_ok=True)


def horizon_probability_from_weekly_rate(p_recent: float, horizon: int) -> float:
    """Convert a weekly occurrence rate under the frozen iid-week assumption."""
    probability = float(p_recent)
    if horizon <= 0 or not np.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError("A positive horizon and a finite weekly rate in [0, 1] are required")
    return float(1.0 - (1.0 - probability) ** int(horizon))


def backward_nonoverlapping_block_events(values: np.ndarray, horizon: int) -> np.ndarray:
    """Return complete H-week event indicators, aligned backward from the origin."""
    series = np.asarray(values, dtype=float)
    if horizon <= 0 or not np.isfinite(series).all() or np.any(series < 0):
        raise ValueError("Blocks require a positive horizon and finite non-negative values")
    complete_blocks = len(series) // int(horizon)
    if complete_blocks == 0:
        return np.asarray([], dtype=int)
    used = series[-complete_blocks * int(horizon) :]
    return (used.reshape(complete_blocks, int(horizon)).sum(axis=1) > 0).astype(int)


def pooled_block_probability(successes: pd.Series, counts: pd.Series) -> float:
    total = int(counts.sum())
    if total <= 0:
        return np.nan
    positive = int(successes.sum())
    if not 0 <= positive <= total:
        raise ValueError("Positive block count must be between zero and total blocks")
    return float(positive / total)


def probability_metrics(
    target: np.ndarray, probability: np.ndarray, *, epsilon: float
) -> dict[str, float]:
    outcome = np.asarray(target, dtype=int)
    forecast = np.asarray(probability, dtype=float)
    if len(outcome) == 0 or outcome.shape != forecast.shape:
        raise ValueError("Target and probability must be non-empty and have equal shape")
    if not set(np.unique(outcome)).issubset({0, 1}):
        raise ValueError("Probability target must be binary")
    if not np.isfinite(forecast).all() or np.any((forecast < 0) | (forecast > 1)):
        raise ValueError("Forecast probabilities must be finite and in [0, 1]")
    protected = np.clip(forecast, epsilon, 1.0 - epsilon)
    brier = np.square(forecast - outcome)
    log_losses = -(outcome * np.log(protected) + (1 - outcome) * np.log(1 - protected))
    both_classes = len(np.unique(outcome)) == 2
    return {
        "observed_event_rate": float(np.mean(outcome)),
        "mean_probability": float(np.mean(forecast)),
        "calibration_gap": float(np.mean(forecast) - np.mean(outcome)),
        "brier_score": float(np.mean(brier)),
        "log_loss": float(np.mean(log_losses)),
        "roc_auc": float(roc_auc_score(outcome, forecast)) if both_classes else np.nan,
        "average_precision": (
            float(average_precision_score(outcome, forecast)) if both_classes else np.nan
        ),
        "zero_probability_share": float(np.mean(forecast == 0.0)),
        "one_probability_share": float(np.mean(forecast == 1.0)),
    }


def summarize_probabilities(
    predictions: pd.DataFrame, group_columns: list[str], *, epsilon: float
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, frame in predictions.groupby(group_columns, dropna=False, sort=True):
        values = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(group_columns, values))
        row.update(
            {
                "n_sku_origins": int(len(frame)),
                "n_unique_skus": int(frame["sku"].nunique()),
                **probability_metrics(
                    frame["actual_event"].to_numpy(dtype=int),
                    frame["probability"].to_numpy(dtype=float),
                    epsilon=epsilon,
                ),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def common_probability_sample(
    predictions: pd.DataFrame, methods: Iterable[str] = METHODS
) -> pd.DataFrame:
    requested = list(methods)
    relevant = predictions.loc[predictions["method"].isin(requested)].copy()
    key_columns = ["horizon_label", "origin_index", "sku"]
    counts = relevant.groupby(key_columns)["method"].nunique()
    keys = counts.loc[counts.eq(len(requested))].reset_index()[key_columns]
    return relevant.merge(keys, on=key_columns, how="inner", validate="many_to_one")


def reliability_bins(
    predictions: pd.DataFrame, edges: list[float]
) -> pd.DataFrame:
    if len(edges) < 2 or edges[0] != 0.0 or edges[-1] != 1.0:
        raise ValueError("Reliability bins must start at 0 and end at 1")
    frame = predictions.copy()
    probabilities = frame["probability"].to_numpy(dtype=float)
    bin_index = np.searchsorted(np.asarray(edges), probabilities, side="right") - 1
    frame["bin_index"] = np.minimum(bin_index, len(edges) - 2)
    rows: list[dict[str, Any]] = []
    group_columns = ["horizon_label", "horizon_weeks", "method", "bin_index"]
    for keys, group in frame.groupby(group_columns, sort=True):
        horizon_label, horizon_weeks, method, index = keys
        idx = int(index)
        mean_probability = float(group["probability"].mean())
        observed_rate = float(group["actual_event"].mean())
        rows.append(
            {
                "horizon_label": horizon_label,
                "horizon_weeks": int(horizon_weeks),
                "method": method,
                "bin_index": idx,
                "bin_lower": float(edges[idx]),
                "bin_upper": float(edges[idx + 1]),
                "n_sku_origins": int(len(group)),
                "mean_probability": mean_probability,
                "observed_event_rate": observed_rate,
                "calibration_gap": mean_probability - observed_rate,
            }
        )
    return pd.DataFrame(rows)


def paired_bootstrap_brier_difference(
    predictions: pd.DataFrame,
    method: str,
    baseline: str,
    *,
    repetitions: int,
    seed: int,
) -> dict[str, Any]:
    pivot = (
        predictions.loc[predictions["method"].isin([method, baseline])]
        .groupby(["sku", "method"], as_index=False)["brier_loss"]
        .mean()
        .pivot(index="sku", columns="method", values="brier_loss")
    )
    if method not in pivot.columns or baseline not in pivot.columns:
        pivot = pivot.iloc[0:0]
    else:
        pivot = pivot.dropna(subset=[method, baseline])
    if pivot.empty:
        return {
            "method": method,
            "baseline": baseline,
            "n_skus": 0,
            "mean_difference": np.nan,
            "median_difference": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
        }
    differences = (pivot[method] - pivot[baseline]).to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(int(repetitions), dtype=float)
    for index in range(int(repetitions)):
        bootstrap[index] = float(
            np.mean(rng.choice(differences, size=len(differences), replace=True))
        )
    return {
        "method": method,
        "baseline": baseline,
        "n_skus": int(len(differences)),
        "mean_difference": float(np.mean(differences)),
        "median_difference": float(np.median(differences)),
        "ci_low": float(np.quantile(bootstrap, 0.025)),
        "ci_high": float(np.quantile(bootstrap, 0.975)),
    }


def _origin_head_to_head(
    origin_summary: pd.DataFrame, baselines: list[str]
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, frame in origin_summary.groupby(
        ["horizon_label", "horizon_weeks", "origin_index", "origin"], sort=True
    ):
        horizon_label, horizon_weeks, origin_index, origin = keys
        indexed = frame.set_index("method")
        if PRIMARY_METHOD not in indexed.index:
            continue
        for baseline in baselines:
            if baseline not in indexed.index:
                continue
            primary_brier = float(indexed.loc[PRIMARY_METHOD, "brier_score"])
            baseline_brier = float(indexed.loc[baseline, "brier_score"])
            rows.append(
                {
                    "horizon_label": horizon_label,
                    "horizon_weeks": int(horizon_weeks),
                    "origin_index": int(origin_index),
                    "origin": str(pd.Timestamp(origin).date()),
                    "baseline": baseline,
                    "pxq_independence_brier": primary_brier,
                    "baseline_brier": baseline_brier,
                    "brier_difference": primary_brier - baseline_brier,
                    "pxq_independence_better": primary_brier < baseline_brier,
                    "n_sku_origins": int(indexed.loc[PRIMARY_METHOD, "n_sku_origins"]),
                }
            )
    return pd.DataFrame(rows)


def _probability_value_gate(
    head_to_head: pd.DataFrame,
    paired: pd.DataFrame,
    horizon_summary: pd.DataFrame,
    *,
    baseline: str,
    minimum_winning_origins: int,
    maximum_absolute_calibration_gap: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    primary_summary = horizon_summary.loc[horizon_summary["method"].eq(PRIMARY_METHOD)]
    for record in primary_summary.itertuples(index=False):
        origin_rows = head_to_head.loc[
            head_to_head["horizon_label"].eq(record.horizon_label)
            & head_to_head["baseline"].eq(baseline)
        ]
        paired_rows = paired.loc[
            paired["horizon_label"].eq(record.horizon_label)
            & paired["baseline"].eq(baseline)
        ]
        wins = int(origin_rows["pxq_independence_better"].sum())
        ci_high = float(paired_rows["ci_high"].iloc[0]) if len(paired_rows) == 1 else np.nan
        calibration_gap = float(record.calibration_gap)
        origin_gate = wins >= int(minimum_winning_origins)
        paired_gate = bool(np.isfinite(ci_high) and ci_high < 0)
        calibration_gate = abs(calibration_gap) <= float(maximum_absolute_calibration_gap)
        rows.append(
            {
                "horizon_label": record.horizon_label,
                "horizon_weeks": int(record.horizon_weeks),
                "baseline": baseline,
                "winning_origins": wins,
                "minimum_winning_origins": int(minimum_winning_origins),
                "paired_ci_high": ci_high,
                "calibration_gap": calibration_gap,
                "maximum_absolute_calibration_gap": float(
                    maximum_absolute_calibration_gap
                ),
                "origin_gate_passed": origin_gate,
                "paired_gate_passed": paired_gate,
                "calibration_gate_passed": calibration_gate,
                "historical_probability_value_supported": bool(
                    origin_gate and paired_gate and calibration_gate
                ),
            }
        )
    return pd.DataFrame(rows)


def _build_components(
    weekly_raw: pd.DataFrame,
    v4_base: pd.DataFrame,
    *,
    horizons: list[dict[str, Any]],
    cleaning_parameters: dict[str, Any],
    minimum_complete_blocks: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    reconciliation_count = 0
    reconciliation_max_difference = 0.0

    for horizon in horizons:
        label = str(horizon["label"])
        weeks = int(horizon["weeks"])
        horizon_base = v4_base.loc[v4_base["horizon_label"].eq(label)].copy()
        origins = horizon_base[["origin_index", "origin"]].drop_duplicates().sort_values(
            "origin_index"
        )
        if len(origins) != int(horizon["origins"]):
            raise RuntimeError(f"V4.0 origin count differs from frozen V4.1 protocol: {label}")
        for origin_record in origins.itertuples(index=False):
            origin = pd.Timestamp(origin_record.origin)
            current = horizon_base.loc[
                horizon_base["origin_index"].eq(origin_record.origin_index)
            ].copy()
            train_raw = weekly_raw.loc[weekly_raw["week_start"] < origin].copy()
            clean = apply_v2_cleaning(train_raw, **cleaning_parameters)
            training_by_sku = {
                str(sku): frame
                for sku, frame in clean.weekly.groupby("sku", sort=False)
            }
            eligible_recent = 0
            eligible_blocks = 0
            skipped_short_recent = 0
            for item in current.itertuples(index=False):
                sku = str(item.sku)
                sku_training = training_by_sku.get(sku)
                if sku_training is None:
                    skipped_short_recent += 1
                    continue
                segment = _ending_contiguous_training_segment(
                    sku_training, origin, "sales_v2"
                )
                if len(segment) < weeks:
                    skipped_short_recent += 1
                    continue
                values = segment["sales_v2"].to_numpy(dtype=float)
                p_recent = float(np.mean(values[-weeks:] > 0))
                p_horizon = horizon_probability_from_weekly_rate(p_recent, weeks)
                events = backward_nonoverlapping_block_events(values, weeks)
                complete_blocks = int(len(events))
                positive_blocks = int(events.sum())
                sku_block_probability = (
                    float(positive_blocks / complete_blocks)
                    if complete_blocks >= minimum_complete_blocks
                    else np.nan
                )
                eligible_recent += 1
                eligible_blocks += int(complete_blocks >= minimum_complete_blocks)
                existing_p = getattr(item, "pxq_p_hat", np.nan)
                if np.isfinite(existing_p):
                    difference = abs(float(existing_p) - p_recent)
                    reconciliation_count += 1
                    reconciliation_max_difference = max(
                        reconciliation_max_difference, difference
                    )
                rows.append(
                    {
                        "horizon_label": label,
                        "horizon_weeks": weeks,
                        "approximate_days": int(horizon["approximate_days"]),
                        "origin_index": int(origin_record.origin_index),
                        "origin": str(origin.date()),
                        "sku": sku,
                        "cluster": int(item.cluster),
                        "cluster_profile": str(item.cluster_profile),
                        "actual_sum": float(item.actual_sum),
                        "actual_event": int(float(item.actual_sum) > 0),
                        "training_weeks": int(len(values)),
                        "p_recent": p_recent,
                        "pxq_independence_probability": p_horizon,
                        "complete_history_blocks": complete_blocks,
                        "positive_history_blocks": positive_blocks,
                        "sku_block_probability": sku_block_probability,
                    }
                )
            audits.append(
                {
                    "horizon_label": label,
                    "horizon_weeks": weeks,
                    "origin_index": int(origin_record.origin_index),
                    "origin": str(origin.date()),
                    "v4_base_sku_origins": int(len(current)),
                    "eligible_recent_rate": eligible_recent,
                    "eligible_direct_blocks": eligible_blocks,
                    "skipped_short_recent": skipped_short_recent,
                    "v2_corrected_intervals": int(clean.summary["corrected_intervals"]),
                }
            )

    components = pd.DataFrame(rows)
    if components.empty:
        raise RuntimeError("No V4.1 probability components were created")
    components["overall_block_probability"] = np.nan
    components["profile_block_probability"] = np.nan
    pool_columns = ["horizon_label", "origin_index"]
    for _, index in components.groupby(pool_columns).groups.items():
        current = components.loc[index]
        pool = current.loc[
            current["complete_history_blocks"].ge(minimum_complete_blocks)
        ]
        overall_probability = pooled_block_probability(
            pool["positive_history_blocks"], pool["complete_history_blocks"]
        )
        components.loc[index, "overall_block_probability"] = overall_probability
        for profile, profile_index in current.groupby("cluster_profile").groups.items():
            profile_pool = components.loc[profile_index]
            profile_pool = profile_pool.loc[
                profile_pool["complete_history_blocks"].ge(minimum_complete_blocks)
            ]
            profile_probability = pooled_block_probability(
                profile_pool["positive_history_blocks"],
                profile_pool["complete_history_blocks"],
            )
            components.loc[profile_index, "profile_block_probability"] = profile_probability

    reconciliation = {
        "v4_nonmissing_p_rows_compared": reconciliation_count,
        "v4_p_max_absolute_difference": reconciliation_max_difference,
        "reconciled": bool(reconciliation_max_difference <= 1e-12),
    }
    return components, pd.DataFrame(audits), reconciliation


def _components_to_long(
    components: pd.DataFrame, *, epsilon: float
) -> pd.DataFrame:
    probability_columns = {
        "PXQ_independence": "pxq_independence_probability",
        "SKU_block_frequency": "sku_block_probability",
        "Profile_block_frequency": "profile_block_probability",
        "Overall_block_frequency": "overall_block_probability",
    }
    identifier_columns = [
        "horizon_label",
        "horizon_weeks",
        "approximate_days",
        "origin_index",
        "origin",
        "sku",
        "cluster",
        "cluster_profile",
        "actual_sum",
        "actual_event",
        "training_weeks",
        "p_recent",
        "complete_history_blocks",
        "positive_history_blocks",
    ]
    frames: list[pd.DataFrame] = []
    for method, column in probability_columns.items():
        current = components[identifier_columns + [column]].rename(
            columns={column: "probability"}
        )
        current = current.dropna(subset=["probability"]).copy()
        current["method"] = method
        current["brier_loss"] = np.square(
            current["probability"] - current["actual_event"]
        )
        protected = current["probability"].clip(epsilon, 1.0 - epsilon)
        current["log_loss_value"] = -(
            current["actual_event"] * np.log(protected)
            + (1 - current["actual_event"]) * np.log(1 - protected)
        )
        frames.append(current)
    return pd.concat(frames, ignore_index=True)


def _run_probability_validation_unlocked(config_path: str | Path) -> dict[str, Any]:
    config_file = Path(config_path).resolve()
    project_root = config_file.parent.parent
    config = load_config(config_file)
    if config["project"]["analysis_mode"] != "retrospective_method_development":
        raise RuntimeError("V4.1 historical validation must remain retrospective")

    v4_outcome_path = project_root / config["input"]["v4_outcome"]
    v4_outcome = json.loads(v4_outcome_path.read_text(encoding="utf-8"))
    if str(v4_outcome.get("protocol_version")) != "4.0":
        raise RuntimeError("V4.1 requires the frozen V4.0 output")
    v4_predictions = pd.read_csv(project_root / config["input"]["v4_predictions"])
    v4_predictions["origin"] = pd.to_datetime(v4_predictions["origin"])
    v4_base = v4_predictions.loc[v4_predictions["model"].eq("MA4_proxy")].copy()
    key_columns = ["horizon_label", "origin_index", "sku"]
    if v4_base.duplicated(key_columns).any():
        raise RuntimeError("V4.0 base predictions are not unique at SKU-origin grain")

    loaded = load_workbook_long(
        project_root / config["input"]["workbook"],
        metadata_columns=config["input"]["metadata_columns"],
        sku_column=config["input"]["sku_column"],
        minimum_covered_days=config["input"]["calendar"]["minimum_covered_days"],
    )
    if loaded.audit["sha256"] != config["input"]["expected_sha256"]:
        raise RuntimeError("Raw workbook SHA256 differs from the frozen V4.1 protocol")
    if loaded.audit["blockers"]:
        raise RuntimeError(f"Blocking audit findings: {loaded.audit['blockers']}")

    probability_config = config["probability"]
    minimum_blocks = int(
        probability_config["historical_blocks"]["minimum_complete_blocks"]
    )
    components, audits, reconciliation = _build_components(
        loaded.weekly_complete,
        v4_base,
        horizons=list(probability_config["horizons"]),
        cleaning_parameters=config["cleaning_v2"],
        minimum_complete_blocks=minimum_blocks,
    )
    epsilon = float(config["evaluation"]["numerical_log_loss_epsilon"])
    predictions = _components_to_long(components, epsilon=epsilon)
    observed_methods = set(predictions["method"].unique())
    if observed_methods != set(probability_config["methods"]):
        raise RuntimeError(f"Unexpected probability methods: {sorted(observed_methods)}")
    common = common_probability_sample(predictions, probability_config["methods"])

    summaries = {
        "summary_by_horizon_model": summarize_probabilities(
            common, ["horizon_label", "horizon_weeks", "method"], epsilon=epsilon
        ),
        "summary_by_origin_model": summarize_probabilities(
            common,
            [
                "horizon_label",
                "horizon_weeks",
                "origin_index",
                "origin",
                "method",
            ],
            epsilon=epsilon,
        ),
        "summary_by_profile_model": summarize_probabilities(
            common,
            [
                "horizon_label",
                "horizon_weeks",
                "cluster",
                "cluster_profile",
                "method",
            ],
            epsilon=epsilon,
        ),
        "native_coverage": (
            predictions.groupby(["horizon_label", "horizon_weeks", "method"])
            .agg(
                n_sku_origins=("sku", "size"),
                n_unique_skus=("sku", "nunique"),
            )
            .reset_index()
        ),
    }

    repetitions = int(config["evaluation"]["paired_bootstrap_repetitions"])
    seed = int(config["project"]["seed"])
    baselines = [method for method in METHODS if method != PRIMARY_METHOD]
    paired_rows: list[dict[str, Any]] = []
    for (label, weeks), frame in common.groupby(
        ["horizon_label", "horizon_weeks"], sort=True
    ):
        for baseline in baselines:
            paired_rows.append(
                {
                    "horizon_label": label,
                    "horizon_weeks": int(weeks),
                    **paired_bootstrap_brier_difference(
                        frame,
                        PRIMARY_METHOD,
                        baseline,
                        repetitions=repetitions,
                        seed=seed,
                    ),
                }
            )
    paired = pd.DataFrame(paired_rows)
    head_to_head = _origin_head_to_head(
        summaries["summary_by_origin_model"], baselines
    )
    gate_config = config["evaluation"]["probability_value_gate"]
    gate = _probability_value_gate(
        head_to_head,
        paired,
        summaries["summary_by_horizon_model"],
        baseline=str(gate_config["baseline"]),
        minimum_winning_origins=int(gate_config["minimum_winning_origins"]),
        maximum_absolute_calibration_gap=float(
            gate_config["maximum_absolute_calibration_gap"]
        ),
    )
    reliability = reliability_bins(
        common, [float(value) for value in config["evaluation"]["reliability_bin_edges"]]
    )

    output_root = project_root / config["outputs"]["root"]
    _write_json(output_root / "input_audit.json", loaded.audit)
    _write_csv(output_root / "probability_components.csv", components)
    _write_csv(output_root / "rolling_probability_audits.csv", audits)
    _write_csv(output_root / "probability_predictions.csv", predictions)
    _write_csv(output_root / "common_probability_sample.csv", common)
    for name, frame in summaries.items():
        _write_csv(output_root / f"{name}.csv", frame)
    _write_csv(output_root / "reliability_bins.csv", reliability)
    _write_csv(output_root / "paired_brier_comparisons.csv", paired)
    _write_csv(output_root / "origin_brier_head_to_head.csv", head_to_head)
    _write_csv(output_root / "probability_value_gate.csv", gate)

    horizon_summary = summaries["summary_by_horizon_model"]
    best = horizon_summary.sort_values(
        ["horizon_label", "brier_score", "method"]
    ).drop_duplicates("horizon_label")
    primary = horizon_summary.loc[horizon_summary["method"].eq(PRIMARY_METHOD)]
    overall = horizon_summary.loc[
        horizon_summary["method"].eq("Overall_block_frequency")
    ][["horizon_label", "brier_score"]].rename(
        columns={"brier_score": "overall_brier"}
    )
    skill = primary.merge(overall, on="horizon_label", validate="one_to_one")
    skill["brier_skill_vs_overall"] = 1.0 - skill["brier_score"] / skill["overall_brier"]
    outcome = {
        "analysis_mode": config["project"]["analysis_mode"],
        "confirmatory": False,
        "protocol_version": config["project"]["protocol_version"],
        "raw_sha256": loaded.audit["sha256"],
        "reused_v4_quantity_models_without_rerun": True,
        "v4_p_reconciliation": reconciliation,
        "probability_prediction_rows": int(len(predictions)),
        "common_sku_origins": int(
            common.drop_duplicates(key_columns).shape[0]
        ),
        "methods": METHODS,
        "best_common_method_by_brier": {
            str(row.horizon_label): str(row.method)
            for row in best.itertuples(index=False)
        },
        "pxq_independence_brier_skill_vs_overall": {
            str(row.horizon_label): float(row.brier_skill_vs_overall)
            for row in skill.itertuples(index=False)
        },
        "probability_value_gate": json.loads(gate.to_json(orient="records")),
        "management_thresholds_selected": False,
        "inventory_or_profit_claims_supported": False,
    }
    _write_json(output_root / "probability_validation_outcome.json", outcome)
    return outcome


def run_probability_validation(config_path: str | Path) -> dict[str, Any]:
    config_file = Path(config_path).resolve()
    project_root = config_file.parent.parent
    config = load_config(config_file)
    output_root = project_root / config["outputs"]["root"]
    with _exclusive_output_lock(output_root):
        return _run_probability_validation_unlocked(config_file)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run frozen V4.1 occurrence-probability historical validation."
    )
    parser.add_argument("--config", default="config/pxq_probability_v4_1.yaml")
    args = parser.parse_args()
    print(
        json.dumps(
            run_probability_validation(args.config),
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
