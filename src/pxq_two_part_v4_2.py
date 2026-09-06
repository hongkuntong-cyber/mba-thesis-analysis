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


PRIMARY_PROBABILITY_METHOD = "ProfileShrink_L4"
PRIMARY_QUANTITY_METHOD = "ProfileShrink_L4_expected"


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
    lock_path = output_root / ".pxq_two_part.lock"
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as error:
        raise RuntimeError(f"Another V4.2 run may already be writing {output_root}") from error
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(f"pid={os.getpid()}\n")
        yield
    finally:
        lock_path.unlink(missing_ok=True)


def backward_nonoverlapping_block_totals(
    values: np.ndarray, horizon: int, lookback_weeks: int
) -> np.ndarray:
    """Build complete H-week totals from at most the latest fixed lookback."""
    series = np.asarray(values, dtype=float)
    if (
        horizon <= 0
        or lookback_weeks <= 0
        or not np.isfinite(series).all()
        or np.any(series < 0)
    ):
        raise ValueError("Blocks require positive sizes and finite non-negative values")
    recent = series[-int(lookback_weeks) :]
    complete_blocks = len(recent) // int(horizon)
    if complete_blocks == 0:
        return np.asarray([], dtype=float)
    used = recent[-complete_blocks * int(horizon) :]
    return used.reshape(complete_blocks, int(horizon)).sum(axis=1)


def credibility_probability(
    positive_blocks: int,
    complete_blocks: int,
    reference_probability: float,
    prior_equivalent_blocks: float,
) -> float:
    n = int(complete_blocks)
    s = int(positive_blocks)
    reference = float(reference_probability)
    strength = float(prior_equivalent_blocks)
    if n < 0 or s < 0 or s > n:
        raise ValueError("Block counts are inconsistent")
    if strength <= 0 or not np.isfinite(strength):
        raise ValueError("Prior strength must be finite and positive")
    if not np.isfinite(reference) or not 0.0 <= reference <= 1.0:
        return np.nan
    return float((s + strength * reference) / (n + strength))


def credibility_conditional_quantity(
    positive_block_total: float,
    positive_blocks: int,
    reference_conditional_quantity: float,
    prior_equivalent_blocks: float,
) -> float:
    total = float(positive_block_total)
    s = int(positive_blocks)
    reference = float(reference_conditional_quantity)
    strength = float(prior_equivalent_blocks)
    if total < 0 or s < 0 or not np.isfinite(total):
        raise ValueError("Conditional-quantity inputs must be finite and non-negative")
    if strength <= 0 or not np.isfinite(strength):
        raise ValueError("Prior strength must be finite and positive")
    if not np.isfinite(reference) or reference < 0:
        return np.nan
    return float((total + strength * reference) / (s + strength))


def _probability_metrics(
    target: np.ndarray, probability: np.ndarray, *, epsilon: float
) -> dict[str, float]:
    outcome = np.asarray(target, dtype=int)
    forecast = np.asarray(probability, dtype=float)
    if len(outcome) == 0 or outcome.shape != forecast.shape:
        raise ValueError("Target and probability must be non-empty and aligned")
    if not set(np.unique(outcome)).issubset({0, 1}):
        raise ValueError("Probability target must be binary")
    if not np.isfinite(forecast).all() or np.any((forecast < 0) | (forecast > 1)):
        raise ValueError("Forecast probabilities must be finite and in [0, 1]")
    protected = np.clip(forecast, epsilon, 1.0 - epsilon)
    both_classes = len(np.unique(outcome)) == 2
    return {
        "observed_event_rate": float(np.mean(outcome)),
        "mean_probability": float(np.mean(forecast)),
        "calibration_gap": float(np.mean(forecast) - np.mean(outcome)),
        "brier_score": float(np.mean(np.square(forecast - outcome))),
        "log_loss": float(
            np.mean(
                -(outcome * np.log(protected) + (1 - outcome) * np.log(1 - protected))
            )
        ),
        "roc_auc": float(roc_auc_score(outcome, forecast)) if both_classes else np.nan,
        "average_precision": (
            float(average_precision_score(outcome, forecast)) if both_classes else np.nan
        ),
        "zero_probability_share": float(np.mean(forecast == 0.0)),
        "one_probability_share": float(np.mean(forecast == 1.0)),
    }


def _common_method_sample(
    frame: pd.DataFrame, methods: Iterable[str], *, method_column: str = "method"
) -> pd.DataFrame:
    requested = list(methods)
    key_columns = ["horizon_label", "origin_index", "sku"]
    relevant = frame.loc[frame[method_column].isin(requested)].copy()
    method_sets = relevant.groupby(key_columns)[method_column].agg(lambda values: set(values))
    valid = method_sets.loc[method_sets.map(lambda values: values == set(requested))]
    keys = valid.reset_index()[key_columns]
    return relevant.merge(keys, on=key_columns, how="inner", validate="many_to_one")


def _build_historical_components(
    weekly_raw: pd.DataFrame,
    v4_base: pd.DataFrame,
    *,
    horizons: list[dict[str, Any]],
    cleaning_parameters: dict[str, Any],
    lookback_weeks: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    for horizon in horizons:
        label = str(horizon["label"])
        weeks = int(horizon["weeks"])
        horizon_base = v4_base.loc[v4_base["horizon_label"].eq(label)].copy()
        origins = horizon_base[["origin_index", "origin"]].drop_duplicates().sort_values(
            "origin_index"
        )
        if len(origins) != int(horizon["origins"]):
            raise RuntimeError(f"Origin count differs from frozen protocol: {label}")
        for origin_record in origins.itertuples(index=False):
            origin = pd.Timestamp(origin_record.origin)
            current = horizon_base.loc[
                horizon_base["origin_index"].eq(origin_record.origin_index)
            ].copy()
            train_raw = weekly_raw.loc[weekly_raw["week_start"] < origin].copy()
            clean = apply_v2_cleaning(train_raw, **cleaning_parameters)
            training_by_sku = {
                str(sku): frame for sku, frame in clean.weekly.groupby("sku", sort=False)
            }
            with_own_blocks = 0
            pure_reference = 0
            for item in current.itertuples(index=False):
                sku = str(item.sku)
                sku_training = training_by_sku.get(sku)
                if sku_training is None:
                    segment = pd.DataFrame(columns=["sales_v2"])
                else:
                    segment = _ending_contiguous_training_segment(
                        sku_training, origin, "sales_v2"
                    )
                values = segment["sales_v2"].to_numpy(dtype=float)
                totals = backward_nonoverlapping_block_totals(
                    values, weeks, lookback_weeks
                )
                complete_blocks = int(len(totals))
                positive_mask = totals > 0
                positive_blocks = int(positive_mask.sum())
                positive_total = float(totals[positive_mask].sum())
                total_volume = float(totals.sum())
                with_own_blocks += int(complete_blocks > 0)
                pure_reference += int(complete_blocks == 0)
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
                        "lookback_weeks_used": int(min(len(values), lookback_weeks)),
                        "complete_history_blocks": complete_blocks,
                        "positive_history_blocks": positive_blocks,
                        "positive_history_volume": positive_total,
                        "all_history_volume": total_volume,
                        "sku_recent_block_probability": (
                            float(positive_blocks / complete_blocks)
                            if complete_blocks > 0
                            else np.nan
                        ),
                        "sku_recent_block_conditional_quantity": (
                            float(positive_total / positive_blocks)
                            if positive_blocks > 0
                            else np.nan
                        ),
                        "sku_recent_block_expected_quantity": (
                            float(total_volume / complete_blocks)
                            if complete_blocks > 0
                            else np.nan
                        ),
                    }
                )
            audits.append(
                {
                    "horizon_label": label,
                    "horizon_weeks": weeks,
                    "origin_index": int(origin_record.origin_index),
                    "origin": str(origin.date()),
                    "base_sku_origins": int(len(current)),
                    "sku_origins_with_own_blocks": with_own_blocks,
                    "pure_reference_sku_origins": pure_reference,
                    "v2_corrected_intervals": int(clean.summary["corrected_intervals"]),
                }
            )
    result = pd.DataFrame(rows)
    if result.empty:
        raise RuntimeError("No V4.2 historical components were created")
    return result, pd.DataFrame(audits)


def add_leave_one_out_references(
    components: pd.DataFrame, *, minimum_peer_blocks: int
) -> pd.DataFrame:
    """Add profile and enterprise peer rates, excluding the focal SKU."""
    frame = components.copy()
    numeric = [
        "complete_history_blocks",
        "positive_history_blocks",
        "positive_history_volume",
    ]
    origin_keys = ["horizon_label", "origin_index"]
    profile_keys = [*origin_keys, "cluster_profile"]
    overall = frame.groupby(origin_keys)[numeric].transform("sum") - frame[numeric]
    profile = frame.groupby(profile_keys)[numeric].transform("sum") - frame[numeric]

    frame["enterprise_peer_blocks"] = overall["complete_history_blocks"]
    frame["enterprise_peer_positive_blocks"] = overall["positive_history_blocks"]
    frame["enterprise_peer_positive_volume"] = overall["positive_history_volume"]
    frame["profile_peer_blocks"] = profile["complete_history_blocks"]
    frame["profile_peer_positive_blocks"] = profile["positive_history_blocks"]
    frame["profile_peer_positive_volume"] = profile["positive_history_volume"]

    enterprise_p = np.where(
        frame["enterprise_peer_blocks"] > 0,
        frame["enterprise_peer_positive_blocks"] / frame["enterprise_peer_blocks"],
        np.nan,
    )
    enterprise_q = np.where(
        frame["enterprise_peer_positive_blocks"] > 0,
        frame["enterprise_peer_positive_volume"]
        / frame["enterprise_peer_positive_blocks"],
        np.nan,
    )
    profile_p_native = np.where(
        frame["profile_peer_blocks"] > 0,
        frame["profile_peer_positive_blocks"] / frame["profile_peer_blocks"],
        np.nan,
    )
    profile_q_native = np.where(
        frame["profile_peer_positive_blocks"] > 0,
        frame["profile_peer_positive_volume"] / frame["profile_peer_positive_blocks"],
        np.nan,
    )
    profile_eligible = frame["profile_peer_blocks"].ge(int(minimum_peer_blocks))
    profile_q_eligible = profile_eligible & frame["profile_peer_positive_blocks"].gt(0)

    frame["enterprise_reference_probability"] = enterprise_p
    frame["enterprise_reference_conditional_quantity"] = enterprise_q
    frame["profile_reference_probability"] = np.where(
        profile_eligible, profile_p_native, enterprise_p
    )
    frame["profile_probability_reference_source"] = np.where(
        profile_eligible, "profile_leave_one_out", "enterprise_fallback"
    )
    frame["profile_reference_conditional_quantity"] = np.where(
        profile_q_eligible, profile_q_native, enterprise_q
    )
    frame["profile_quantity_reference_source"] = np.where(
        profile_q_eligible, "profile_leave_one_out", "enterprise_fallback"
    )
    return frame


def add_shrunk_estimates(
    components: pd.DataFrame,
    *,
    primary_strength: int,
    sensitivity_strengths: list[int],
) -> pd.DataFrame:
    frame = components.copy()
    strengths = sorted(set([int(primary_strength), *map(int, sensitivity_strengths)]))
    for strength in strengths:
        p_column = f"profile_probability_l{strength}"
        q_column = f"profile_conditional_quantity_l{strength}"
        frame[p_column] = [
            credibility_probability(s, n, reference, strength)
            for s, n, reference in zip(
                frame["positive_history_blocks"],
                frame["complete_history_blocks"],
                frame["profile_reference_probability"],
            )
        ]
        frame[q_column] = [
            credibility_conditional_quantity(total, s, reference, strength)
            for total, s, reference in zip(
                frame["positive_history_volume"],
                frame["positive_history_blocks"],
                frame["profile_reference_conditional_quantity"],
            )
        ]
        frame[f"profile_expected_quantity_l{strength}"] = frame[p_column] * frame[q_column]

    strength = int(primary_strength)
    frame[f"enterprise_probability_l{strength}"] = [
        credibility_probability(s, n, reference, strength)
        for s, n, reference in zip(
            frame["positive_history_blocks"],
            frame["complete_history_blocks"],
            frame["enterprise_reference_probability"],
        )
    ]
    frame[f"enterprise_conditional_quantity_l{strength}"] = [
        credibility_conditional_quantity(total, s, reference, strength)
        for total, s, reference in zip(
            frame["positive_history_volume"],
            frame["positive_history_blocks"],
            frame["enterprise_reference_conditional_quantity"],
        )
    ]
    frame[f"enterprise_expected_quantity_l{strength}"] = (
        frame[f"enterprise_probability_l{strength}"]
        * frame[f"enterprise_conditional_quantity_l{strength}"]
    )
    return frame


def _probability_long(
    components: pd.DataFrame,
    *,
    methods: list[str],
    primary_strength: int,
    sensitivity_strengths: list[int],
    epsilon: float,
) -> pd.DataFrame:
    mapping: dict[str, str] = {
        f"ProfileShrink_L{primary_strength}": f"profile_probability_l{primary_strength}",
        f"EnterpriseShrink_L{primary_strength}": f"enterprise_probability_l{primary_strength}",
        "SKU_recent_block": "sku_recent_block_probability",
        "PXQ_independence": "pxq_independence_probability",
    }
    for strength in sensitivity_strengths:
        mapping[f"ProfileShrink_L{int(strength)}"] = f"profile_probability_l{int(strength)}"
    if set(mapping) != set(methods):
        raise RuntimeError("Configured probability methods do not match frozen implementation")
    identifiers = [
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
        "lookback_weeks_used",
        "complete_history_blocks",
        "positive_history_blocks",
        "profile_probability_reference_source",
        "profile_quantity_reference_source",
    ]
    frames: list[pd.DataFrame] = []
    for method in methods:
        current = components[identifiers + [mapping[method]]].rename(
            columns={mapping[method]: "probability"}
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


def _expected_quantity_long(
    components: pd.DataFrame,
    v4_predictions: pd.DataFrame,
    *,
    methods: list[str],
    primary_strength: int,
) -> pd.DataFrame:
    new_mapping = {
        f"ProfileShrink_L{primary_strength}_expected": f"profile_expected_quantity_l{primary_strength}",
        f"EnterpriseShrink_L{primary_strength}_expected": f"enterprise_expected_quantity_l{primary_strength}",
        "SKU_recent_block_expected": "sku_recent_block_expected_quantity",
    }
    existing_methods = [method for method in methods if method not in new_mapping]
    identifiers = [
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
        "complete_history_blocks",
        "positive_history_blocks",
    ]
    frames: list[pd.DataFrame] = []
    for method, column in new_mapping.items():
        current = components[identifiers + [column]].rename(
            columns={column: "forecast_sum"}
        )
        current = current.dropna(subset=["forecast_sum"]).copy()
        current["method"] = method
        current["existing_weekly_mase"] = np.nan
        frames.append(current)

    existing = v4_predictions.loc[v4_predictions["model"].isin(existing_methods)].copy()
    existing = existing.rename(columns={"model": "method", "mase": "existing_weekly_mase"})
    existing["origin"] = pd.to_datetime(existing["origin"]).dt.date.astype(str)
    existing["actual_event"] = existing["actual_sum"].gt(0).astype(int)
    existing["training_weeks"] = np.nan
    existing["complete_history_blocks"] = np.nan
    existing["positive_history_blocks"] = np.nan
    frames.append(existing[identifiers + ["forecast_sum", "method", "existing_weekly_mase"]])
    result = pd.concat(frames, ignore_index=True)
    result["horizon_total_error"] = result["forecast_sum"] - result["actual_sum"]
    result["horizon_total_abs_error"] = result["horizon_total_error"].abs()
    result["underforecast_units"] = (-result["horizon_total_error"]).clip(lower=0)
    result["overforecast_units"] = result["horizon_total_error"].clip(lower=0)
    return result


def _summarize_probabilities(
    predictions: pd.DataFrame, group_columns: list[str], *, epsilon: float
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, frame in predictions.groupby(group_columns, dropna=False, sort=True):
        values = keys if isinstance(keys, tuple) else (keys,)
        rows.append(
            {
                **dict(zip(group_columns, values)),
                "n_sku_origins": int(len(frame)),
                "n_unique_skus": int(frame["sku"].nunique()),
                **_probability_metrics(
                    frame["actual_event"].to_numpy(dtype=int),
                    frame["probability"].to_numpy(dtype=float),
                    epsilon=epsilon,
                ),
            }
        )
    return pd.DataFrame(rows)


def _summarize_quantities(
    predictions: pd.DataFrame, group_columns: list[str]
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, frame in predictions.groupby(group_columns, dropna=False, sort=True):
        values = keys if isinstance(keys, tuple) else (keys,)
        actual_volume = float(frame["actual_sum"].sum())
        forecast_volume = float(frame["forecast_sum"].sum())
        absolute = frame["horizon_total_abs_error"]
        rows.append(
            {
                **dict(zip(group_columns, values)),
                "n_sku_origins": int(len(frame)),
                "n_unique_skus": int(frame["sku"].nunique()),
                "actual_volume": actual_volume,
                "forecast_volume": forecast_volume,
                "mean_absolute_error": float(absolute.mean()),
                "median_absolute_error": float(absolute.median()),
                "aggregate_wape": (
                    float(absolute.sum() / actual_volume) if actual_volume > 0 else np.nan
                ),
                "aggregate_bias": (
                    float((forecast_volume - actual_volume) / actual_volume)
                    if actual_volume > 0
                    else np.nan
                ),
                "underforecast_units": float(frame["underforecast_units"].sum()),
                "overforecast_units": float(frame["overforecast_units"].sum()),
            }
        )
    return pd.DataFrame(rows)


def _conditional_quantity_long(
    components: pd.DataFrame,
    *,
    primary_strength: int,
    sensitivity_strengths: list[int],
) -> pd.DataFrame:
    mapping = {
        f"ProfileShrink_L{primary_strength}": f"profile_conditional_quantity_l{primary_strength}",
        f"EnterpriseShrink_L{primary_strength}": f"enterprise_conditional_quantity_l{primary_strength}",
        "SKU_recent_block": "sku_recent_block_conditional_quantity",
    }
    for strength in sensitivity_strengths:
        mapping[f"ProfileShrink_L{int(strength)}"] = f"profile_conditional_quantity_l{int(strength)}"
    identifiers = [
        "horizon_label",
        "horizon_weeks",
        "origin_index",
        "origin",
        "sku",
        "cluster",
        "cluster_profile",
        "actual_sum",
        "actual_event",
    ]
    frames: list[pd.DataFrame] = []
    for method, column in mapping.items():
        current = components[identifiers + [column]].rename(
            columns={column: "conditional_quantity"}
        )
        current = current.loc[current["actual_event"].eq(1)].dropna(
            subset=["conditional_quantity"]
        )
        current = current.copy()
        current["method"] = method
        current["error"] = current["conditional_quantity"] - current["actual_sum"]
        current["absolute_error"] = current["error"].abs()
        frames.append(current)
    return pd.concat(frames, ignore_index=True)


def _summarize_conditional_quantities(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (label, weeks, method), frame in predictions.groupby(
        ["horizon_label", "horizon_weeks", "method"], sort=True
    ):
        actual = float(frame["actual_sum"].sum())
        rows.append(
            {
                "horizon_label": label,
                "horizon_weeks": int(weeks),
                "method": method,
                "n_positive_sku_origins": int(len(frame)),
                "n_unique_skus": int(frame["sku"].nunique()),
                "actual_positive_volume": actual,
                "predicted_conditional_volume": float(frame["conditional_quantity"].sum()),
                "mean_absolute_error": float(frame["absolute_error"].mean()),
                "median_absolute_error": float(frame["absolute_error"].median()),
                "aggregate_wape": (
                    float(frame["absolute_error"].sum() / actual) if actual > 0 else np.nan
                ),
                "aggregate_bias": (
                    float(frame["error"].sum() / actual) if actual > 0 else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def _reliability_bins(predictions: pd.DataFrame, edges: list[float]) -> pd.DataFrame:
    if len(edges) < 2 or edges[0] != 0.0 or edges[-1] != 1.0:
        raise ValueError("Reliability bins must span [0, 1]")
    frame = predictions.copy()
    index = np.searchsorted(np.asarray(edges), frame["probability"], side="right") - 1
    frame["bin_index"] = np.minimum(index, len(edges) - 2)
    rows: list[dict[str, Any]] = []
    keys = ["horizon_label", "horizon_weeks", "method", "bin_index"]
    for values, current in frame.groupby(keys, sort=True):
        label, weeks, method, bin_index = values
        idx = int(bin_index)
        predicted = float(current["probability"].mean())
        observed = float(current["actual_event"].mean())
        rows.append(
            {
                "horizon_label": label,
                "horizon_weeks": int(weeks),
                "method": method,
                "bin_index": idx,
                "bin_lower": float(edges[idx]),
                "bin_upper": float(edges[idx + 1]),
                "n_sku_origins": int(len(current)),
                "mean_probability": predicted,
                "observed_event_rate": observed,
                "calibration_gap": predicted - observed,
            }
        )
    return pd.DataFrame(rows)


def paired_bootstrap_method_difference(
    predictions: pd.DataFrame,
    method: str,
    baseline: str,
    *,
    loss_column: str,
    repetitions: int,
    seed: int,
) -> dict[str, Any]:
    pivot = (
        predictions.loc[predictions["method"].isin([method, baseline])]
        .groupby(["sku", "method"], as_index=False)[loss_column]
        .mean()
        .pivot(index="sku", columns="method", values=loss_column)
    )
    if method not in pivot.columns or baseline not in pivot.columns:
        pivot = pivot.iloc[0:0]
    else:
        pivot = pivot.dropna(subset=[method, baseline])
    if pivot.empty:
        return {
            "method": method,
            "baseline": baseline,
            "loss": loss_column,
            "n_skus": 0,
            "mean_difference": np.nan,
            "median_difference": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
        }
    differences = (pivot[method] - pivot[baseline]).to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(int(repetitions), dtype=float)
    for idx in range(int(repetitions)):
        bootstrap[idx] = float(
            np.mean(rng.choice(differences, size=len(differences), replace=True))
        )
    return {
        "method": method,
        "baseline": baseline,
        "loss": loss_column,
        "n_skus": int(len(differences)),
        "mean_difference": float(np.mean(differences)),
        "median_difference": float(np.median(differences)),
        "ci_low": float(np.quantile(bootstrap, 0.025)),
        "ci_high": float(np.quantile(bootstrap, 0.975)),
    }


def _paired_comparisons(
    predictions: pd.DataFrame,
    *,
    primary: str,
    baselines: list[str],
    loss_column: str,
    repetitions: int,
    seed: int,
    quantity: bool,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (label, weeks), frame in predictions.groupby(
        ["horizon_label", "horizon_weeks"], sort=True
    ):
        for baseline in baselines:
            paired = _common_method_sample(frame, [primary, baseline])
            result = paired_bootstrap_method_difference(
                paired,
                primary,
                baseline,
                loss_column=loss_column,
                repetitions=repetitions,
                seed=seed,
            )
            result.update({"horizon_label": label, "horizon_weeks": int(weeks)})
            if quantity:
                indexed = paired.groupby("method").agg(
                    total_abs_error=("horizon_total_abs_error", "sum"),
                    actual_volume=("actual_sum", "sum"),
                )
                for method_name, prefix in [(primary, "primary"), (baseline, "baseline")]:
                    if method_name in indexed.index:
                        actual = float(indexed.loc[method_name, "actual_volume"])
                        result[f"{prefix}_aggregate_wape"] = (
                            float(indexed.loc[method_name, "total_abs_error"] / actual)
                            if actual > 0
                            else np.nan
                        )
                result["wape_difference"] = (
                    result.get("primary_aggregate_wape", np.nan)
                    - result.get("baseline_aggregate_wape", np.nan)
                )
            rows.append(result)
    return pd.DataFrame(rows)


def _origin_head_to_head(
    predictions: pd.DataFrame,
    *,
    target_type: str,
    primary: str,
    baselines: list[str],
    loss_column: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_keys = ["horizon_label", "horizon_weeks", "origin_index", "origin"]
    for values, frame in predictions.groupby(group_keys, sort=True):
        label, weeks, origin_index, origin = values
        for baseline in baselines:
            paired = _common_method_sample(frame, [primary, baseline])
            indexed = paired.groupby("method")[loss_column].mean()
            if primary not in indexed.index or baseline not in indexed.index:
                continue
            primary_loss = float(indexed.loc[primary])
            baseline_loss = float(indexed.loc[baseline])
            rows.append(
                {
                    "target_type": target_type,
                    "horizon_label": label,
                    "horizon_weeks": int(weeks),
                    "origin_index": int(origin_index),
                    "origin": str(origin),
                    "primary_method": primary,
                    "baseline": baseline,
                    "primary_loss": primary_loss,
                    "baseline_loss": baseline_loss,
                    "loss_difference": primary_loss - baseline_loss,
                    "primary_better": primary_loss < baseline_loss,
                    "n_sku_origins": int(
                        paired[["horizon_label", "origin_index", "sku"]]
                        .drop_duplicates()
                        .shape[0]
                    ),
                }
            )
    return pd.DataFrame(rows)


def _profile_summary(
    probability_predictions: pd.DataFrame,
    quantity_predictions: pd.DataFrame,
    conditional_predictions: pd.DataFrame,
) -> pd.DataFrame:
    probability = probability_predictions.loc[
        probability_predictions["method"].eq(PRIMARY_PROBABILITY_METHOD)
    ]
    quantity = quantity_predictions.loc[
        quantity_predictions["method"].eq(PRIMARY_QUANTITY_METHOD)
    ]
    conditional = conditional_predictions.loc[
        conditional_predictions["method"].eq(PRIMARY_PROBABILITY_METHOD)
    ]
    keys = ["horizon_label", "horizon_weeks", "cluster", "cluster_profile"]
    rows: list[dict[str, Any]] = []
    for values, p_frame in probability.groupby(keys, sort=True):
        label, weeks, cluster, profile = values
        selector = (
            quantity["horizon_label"].eq(label)
            & quantity["cluster"].eq(cluster)
            & quantity["cluster_profile"].eq(profile)
        )
        q_frame = quantity.loc[selector]
        c_selector = (
            conditional["horizon_label"].eq(label)
            & conditional["cluster"].eq(cluster)
            & conditional["cluster_profile"].eq(profile)
        )
        c_frame = conditional.loc[c_selector]
        actual_volume = float(q_frame["actual_sum"].sum())
        metrics = _probability_metrics(
            p_frame["actual_event"].to_numpy(dtype=int),
            p_frame["probability"].to_numpy(dtype=float),
            epsilon=1e-15,
        )
        rows.append(
            {
                "horizon_label": label,
                "horizon_weeks": int(weeks),
                "cluster": int(cluster),
                "cluster_profile": profile,
                "n_sku_origins": int(len(p_frame)),
                "n_unique_skus": int(p_frame["sku"].nunique()),
                "observed_event_rate": metrics["observed_event_rate"],
                "mean_probability": metrics["mean_probability"],
                "calibration_gap": metrics["calibration_gap"],
                "brier_score": metrics["brier_score"],
                "roc_auc": metrics["roc_auc"],
                "actual_volume": actual_volume,
                "forecast_volume": float(q_frame["forecast_sum"].sum()),
                "expected_quantity_wape": (
                    float(q_frame["horizon_total_abs_error"].sum() / actual_volume)
                    if actual_volume > 0
                    else np.nan
                ),
                "expected_quantity_bias": (
                    float(q_frame["horizon_total_error"].sum() / actual_volume)
                    if actual_volume > 0
                    else np.nan
                ),
                "positive_test_cases": int(len(c_frame)),
                "conditional_quantity_mae": (
                    float(c_frame["absolute_error"].mean()) if len(c_frame) else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def _value_gates(
    probability_summary: pd.DataFrame,
    probability_predictions: pd.DataFrame,
    paired_probability: pd.DataFrame,
    paired_quantity: pd.DataFrame,
    origin_head: pd.DataFrame,
    *,
    minimum_winning_origins: int,
    maximum_absolute_calibration_gap: float,
    quantity_required_baselines: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    primary_summary = probability_summary.loc[
        probability_summary["method"].eq(PRIMARY_PROBABILITY_METHOD)
    ]
    probability_pairs = [
        ("cluster_information_value", "EnterpriseShrink_L4"),
        ("probability_replacement_value", "PXQ_independence"),
    ]
    for record in primary_summary.itertuples(index=False):
        label = str(record.horizon_label)
        weeks = int(record.horizon_weeks)
        primary_rows = probability_predictions.loc[
            probability_predictions["horizon_label"].eq(label)
            & probability_predictions["method"].eq(PRIMARY_PROBABILITY_METHOD)
        ]
        no_extremes = bool(
            (~primary_rows["probability"].isin([0.0, 1.0])).all()
        )
        for gate_type, baseline in probability_pairs:
            head = origin_head.loc[
                origin_head["target_type"].eq("probability")
                & origin_head["horizon_label"].eq(label)
                & origin_head["baseline"].eq(baseline)
            ]
            paired = paired_probability.loc[
                paired_probability["horizon_label"].eq(label)
                & paired_probability["baseline"].eq(baseline)
            ]
            wins = int(head["primary_better"].sum())
            ci_high = float(paired["ci_high"].iloc[0]) if len(paired) == 1 else np.nan
            origin_passed = wins >= int(minimum_winning_origins)
            ci_passed = bool(np.isfinite(ci_high) and ci_high < 0)
            calibration_passed = abs(float(record.calibration_gap)) <= float(
                maximum_absolute_calibration_gap
            )
            if gate_type == "cluster_information_value":
                passed = origin_passed and ci_passed
            else:
                passed = origin_passed and ci_passed and calibration_passed and no_extremes
            rows.append(
                {
                    "gate_type": gate_type,
                    "horizon_label": label,
                    "horizon_weeks": weeks,
                    "primary_method": PRIMARY_PROBABILITY_METHOD,
                    "baseline": baseline,
                    "winning_origins": wins,
                    "minimum_winning_origins": int(minimum_winning_origins),
                    "paired_ci_high": ci_high,
                    "calibration_gap": float(record.calibration_gap),
                    "maximum_absolute_calibration_gap": float(
                        maximum_absolute_calibration_gap
                    ),
                    "no_exact_zero_or_one": no_extremes,
                    "origin_gate_passed": origin_passed,
                    "paired_gate_passed": ci_passed,
                    "calibration_gate_passed": calibration_passed,
                    "wape_gate_passed": np.nan,
                    "gate_passed": bool(passed),
                }
            )

        baseline_results: list[dict[str, Any]] = []
        for baseline in quantity_required_baselines:
            head = origin_head.loc[
                origin_head["target_type"].eq("expected_quantity")
                & origin_head["horizon_label"].eq(label)
                & origin_head["baseline"].eq(baseline)
            ]
            paired = paired_quantity.loc[
                paired_quantity["horizon_label"].eq(label)
                & paired_quantity["baseline"].eq(baseline)
            ]
            wins = int(head["primary_better"].sum())
            ci_high = float(paired["ci_high"].iloc[0]) if len(paired) == 1 else np.nan
            wape_difference = (
                float(paired["wape_difference"].iloc[0]) if len(paired) == 1 else np.nan
            )
            baseline_results.append(
                {
                    "baseline": baseline,
                    "wins": wins,
                    "ci_high": ci_high,
                    "wape_difference": wape_difference,
                    "passed": bool(
                        wins >= int(minimum_winning_origins)
                        and np.isfinite(ci_high)
                        and ci_high < 0
                        and np.isfinite(wape_difference)
                        and wape_difference < 0
                    ),
                }
            )
        rows.append(
            {
                "gate_type": "expected_quantity_value",
                "horizon_label": label,
                "horizon_weeks": weeks,
                "primary_method": PRIMARY_QUANTITY_METHOD,
                "baseline": "+".join(quantity_required_baselines),
                "winning_origins": ";".join(
                    f"{item['baseline']}={item['wins']}" for item in baseline_results
                ),
                "minimum_winning_origins": int(minimum_winning_origins),
                "paired_ci_high": ";".join(
                    f"{item['baseline']}={item['ci_high']:.10g}" for item in baseline_results
                ),
                "calibration_gap": np.nan,
                "maximum_absolute_calibration_gap": np.nan,
                "no_exact_zero_or_one": np.nan,
                "origin_gate_passed": all(
                    item["wins"] >= int(minimum_winning_origins)
                    for item in baseline_results
                ),
                "paired_gate_passed": all(
                    np.isfinite(item["ci_high"]) and item["ci_high"] < 0
                    for item in baseline_results
                ),
                "calibration_gate_passed": np.nan,
                "wape_gate_passed": all(
                    np.isfinite(item["wape_difference"])
                    and item["wape_difference"] < 0
                    for item in baseline_results
                ),
                "gate_passed": all(item["passed"] for item in baseline_results),
            }
        )
    return pd.DataFrame(rows)


def _format_probability_table(summary: pd.DataFrame) -> str:
    selected = [
        PRIMARY_PROBABILITY_METHOD,
        "EnterpriseShrink_L4",
        "SKU_recent_block",
        "PXQ_independence",
        "ProfileShrink_L1",
        "ProfileShrink_L8",
    ]
    rows = [
        "| 周期 | 方法 | N | 实际发生率 | 平均概率 | 校准差 | Brier | AUC |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for record in summary.loc[summary["method"].isin(selected)].itertuples(index=False):
        rows.append(
            "| {days}天代理 | {method} | {n} | {observed:.1%} | {probability:.1%} | "
            "{gap:+.1%} | {brier:.3f} | {auc:.3f} |".format(
                days=int(record.horizon_weeks) * 7,
                method=record.method,
                n=int(record.n_sku_origins),
                observed=float(record.observed_event_rate),
                probability=float(record.mean_probability),
                gap=float(record.calibration_gap),
                brier=float(record.brier_score),
                auc=float(record.roc_auc),
            )
        )
    return "\n".join(rows)


def _format_quantity_table(summary: pd.DataFrame) -> str:
    rows = [
        "| 周期 | 方法 | N | WAPE | Mean AE | Median AE | Bias |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for record in summary.itertuples(index=False):
        rows.append(
            "| {days}天代理 | {method} | {n} | {wape:.3f} | {mae:.2f} | {median:.2f} | {bias:+.1%} |".format(
                days=int(record.horizon_weeks) * 7,
                method=record.method,
                n=int(record.n_sku_origins),
                wape=float(record.aggregate_wape),
                mae=float(record.mean_absolute_error),
                median=float(record.median_absolute_error),
                bias=float(record.aggregate_bias),
            )
        )
    return "\n".join(rows)


def _write_report(
    report_path: Path,
    *,
    probability_summary: pd.DataFrame,
    quantity_summary: pd.DataFrame,
    profile_summary: pd.DataFrame,
    gates: pd.DataFrame,
    outcome: dict[str, Any],
) -> None:
    primary_probability = probability_summary.loc[
        probability_summary["method"].eq(PRIMARY_PROBABILITY_METHOD)
    ].sort_values("horizon_weeks")
    primary_quantity = quantity_summary.loc[
        quantity_summary["method"].eq(PRIMARY_QUANTITY_METHOD)
    ].sort_values("horizon_weeks")
    probability_gate = gates.loc[gates["gate_type"].eq("probability_replacement_value")]
    cluster_gate = gates.loc[gates["gate_type"].eq("cluster_information_value")]
    quantity_gate = gates.loc[gates["gate_type"].eq("expected_quantity_value")]
    probability_result = "、".join(
        f"{int(row.horizon_weeks)*7}天={'通过' if row.gate_passed else '未通过'}"
        for row in probability_gate.itertuples(index=False)
    )
    cluster_result = "、".join(
        f"{int(row.horizon_weeks)*7}天={'通过' if row.gate_passed else '未通过'}"
        for row in cluster_gate.itertuples(index=False)
    )
    quantity_result = "、".join(
        f"{int(row.horizon_weeks)*7}天={'通过' if row.gate_passed else '未通过'}"
        for row in quantity_gate.itertuples(index=False)
    )
    best_probability = (
        probability_summary.sort_values(["horizon_weeks", "brier_score", "method"])
        .drop_duplicates("horizon_weeks")
        .sort_values("horizon_weeks")
    )
    best_probability_text = "、".join(
        f"{int(row.horizon_weeks)*7}天={row.method}（Brier {row.brier_score:.3f}）"
        for row in best_probability.itertuples(index=False)
    )
    ma4_quantity = quantity_summary.loc[
        quantity_summary["method"].eq("MA4_proxy")
    ][["horizon_weeks", "aggregate_wape"]].rename(
        columns={"aggregate_wape": "ma4_wape"}
    )
    quantity_comparison = primary_quantity.merge(
        ma4_quantity, on="horizon_weeks", validate="one_to_one"
    )
    quantity_comparison_text = "、".join(
        f"{int(row.horizon_weeks)*7}天={row.aggregate_wape:.3f} vs MA4 {row.ma4_wape:.3f}"
        for row in quantity_comparison.itertuples(index=False)
    )
    low_information = profile_summary.loc[
        profile_summary["cluster_profile"].eq("low_information")
    ].sort_values("horizon_weeks")
    low_information_text = "、".join(
        f"{int(row.horizon_weeks)*7}天{row.calibration_gap:+.1%}"
        for row in low_information.itertuples(index=False)
    )
    content = f"""# 簇级可信度收缩两部式需求验证报告

协议版本：V4.2；随机种子：42；生成日期：2026-09-05。  
本报告是查看 V4.0/V4.1 后开展的回顾性方法开发，不是新增样本确认。

## 一、结论先行

1. **聚类信息增量门槛：{cluster_result}。** 该门槛只比较同簇参考与完全相同的
   企业整体参考公式，回答聚类是否真正改善概率估计。
2. **替代 V4.1 风险评分门槛：{probability_result}。** 只有同时改善 Brier、
   配对区间、校准并消除精确 0/1，才能称为更可靠概率。
3. **周期总量价值门槛：{quantity_result}。** 该门槛要求新两部式期望量同时稳定
   优于 MA4_proxy 和 Naive；概率改善不自动等于数量改善。
4. V4.2 没有重跑 PXQ、MA4_proxy、Naive、SES、ADIDA2 或 V4.1 概率方法；只计算
   最近 52 周的直接周期统计和预冻结可信度收缩。
5. **失败原因不是收缩强度不足，而是参考组失配。** 同簇概率在活跃持续和低频
   稀疏画像中总体可解释，但低信息层仍大幅低估，三个周期校准差分别为
   {low_information_text}，拖累了总体表现。预冻结的 lambda=1 和 8 均未改变结论。
6. **当前不应把簇用于概率借力。** 各周期最低总体 Brier 分别为：
   {best_probability_text}。企业整体收缩在 28 天代理上校准较好，但它不是本轮
   预设主方案，只能列为下一次新增数据验证候选，不能事后升级为正式结论。

## 二、方法与样本

对每个 SKU、起点和 4/9/13 周周期，从起点向过去使用最多 52 个完整训练周，切成
互不重叠周期。主方法使用 leave-one-SKU-out 同画像参考，参考强度固定为 4 个
等效周期：

`P_H=(正需求周期数+4×同类概率)/(完整周期数+4)`

`Q_H+=(正需求周期总量+4×同类条件均量)/(正需求周期数+4)`

`E(D_H)=P_H×Q_H+`

同画像历史不足时回退到企业整体参考。训练期使用 V2 需求代理，测试目标使用原始
可观测销量。4/9/13 周只能称为 28/63/91 天代理。

## 三、发生概率结果

以下为六种预冻结概率方法的共同 SKU–起点样本；Brier 越低越好。

{_format_probability_table(probability_summary)}

主方法三个周期的 Brier 为 {', '.join(f'{row.brier_score:.3f}' for row in primary_probability.itertuples(index=False))}；
平均校准差为 {', '.join(f'{row.calibration_gap:+.1%}' for row in primary_probability.itertuples(index=False))}。

## 四、周期总量结果

以下为全部八种数量方法共同样本。V4.2 方法只预测周期总量，不预测具体发生周，
因此以周期总量 WAPE 和绝对误差评价，不事后均匀拆周制造 MASE。

{_format_quantity_table(quantity_summary)}

主方法三个周期 WAPE 为 {', '.join(f'{row.aggregate_wape:.3f}' for row in primary_quantity.itertuples(index=False))}。
与 MA4_proxy 的同样本比较为：{quantity_comparison_text}。因此新的两部式期望量
不能替代企业简单数量基准。

## 五、不同管理画像

| 周期 | 画像 | N | 实际发生率 | 平均概率 | 校准差 | Brier | 数量WAPE |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
"""
    for record in profile_summary.itertuples(index=False):
        content += (
            f"| {int(record.horizon_weeks)*7}天代理 | {record.cluster_profile} | "
            f"{int(record.n_sku_origins)} | {record.observed_event_rate:.1%} | "
            f"{record.mean_probability:.1%} | {record.calibration_gap:+.1%} | "
            f"{record.brier_score:.3f} | {record.expected_quantity_wape:.3f} |\n"
        )
    content += f"""

## 六、论文与企业解释

- 聚类仍只负责需求结构识别、管理分层和参考组信息借用，不绑定不同算法；
- 如果同簇参考不优于企业整体参考，应保留聚类的管理描述作用，但放弃其概率增益
  主张；
- 本轮证据支持这种收缩处理：保留 K=2 作为管理分层，不把簇均值用于概率计算；
- 数量端继续以 MA4_proxy 为企业默认回退，PXQ/SES等只保留历史对照，不因某一
  周期单项指标较低而形成簇专属算法；
- 28 天企业整体收缩可作为下一次确认性检验的单一概率候选；63/91 天目前继续只
  展示未经充分校准的风险信息，不设置自动阈值；
- 概率、条件需求量和期望需求量必须分列展示；期望需求量不是自动采购量；
- 没有库存、在途、提前期、MOQ、服务水平、成本和毛利，不能评价利润或补货阈值；
- 促销、断货、上新和退市没有可靠状态字段，仍需人工标记和复核；
- 本轮使用已经查看过的历史起点，只能形成新方法候选。新增数据确认前不得写成最终
  外部有效结论。

## 七、运行与复核

```bash
python -m src.pxq_two_part_v4_2 --config config/pxq_two_part_v4_2.yaml
python -m src.validate_pxq_two_part_v4_2 --config config/pxq_two_part_v4_2.yaml
pytest -q
```

原始数据 SHA256：`{outcome['raw_sha256']}`。  
V4.0/V4.1 既有模型未重跑：`{outcome['reused_prior_results_without_rerun']}`。
"""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = report_path.with_name(f".{report_path.name}.{os.getpid()}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(report_path)


def _run_two_part_unlocked(config_path: str | Path) -> dict[str, Any]:
    config_file = Path(config_path).resolve()
    project_root = config_file.parent.parent
    config = load_config(config_file)
    if config["project"]["analysis_mode"] != "retrospective_method_development":
        raise RuntimeError("V4.2 must remain retrospective method development")

    v4_outcome = json.loads(
        (project_root / config["input"]["v4_outcome"]).read_text(encoding="utf-8")
    )
    v41_outcome = json.loads(
        (project_root / config["input"]["v4_1_outcome"]).read_text(encoding="utf-8")
    )
    if str(v4_outcome.get("protocol_version")) != "4.0":
        raise RuntimeError("V4.2 requires frozen V4.0 outputs")
    if str(v41_outcome.get("protocol_version")) != "4.1":
        raise RuntimeError("V4.2 requires frozen V4.1 outputs")

    v4_predictions = pd.read_csv(project_root / config["input"]["v4_predictions"])
    v4_predictions["origin"] = pd.to_datetime(v4_predictions["origin"])
    v4_base = v4_predictions.loc[v4_predictions["model"].eq("MA4_proxy")].copy()
    key_columns = ["horizon_label", "origin_index", "sku"]
    if v4_base.duplicated(key_columns).any():
        raise RuntimeError("V4.0 base predictions are not unique at SKU-origin grain")

    v41_predictions = pd.read_csv(
        project_root / config["input"]["v4_1_probability_predictions"]
    )
    v41_primary = v41_predictions.loc[
        v41_predictions["method"].eq("PXQ_independence"),
        [*key_columns, "probability"],
    ].rename(columns={"probability": "pxq_independence_probability"})
    if v41_primary.duplicated(key_columns).any():
        raise RuntimeError("V4.1 probability rows are not unique at SKU-origin grain")

    loaded = load_workbook_long(
        project_root / config["input"]["workbook"],
        metadata_columns=config["input"]["metadata_columns"],
        sku_column=config["input"]["sku_column"],
        minimum_covered_days=config["input"]["calendar"]["minimum_covered_days"],
    )
    if loaded.audit["sha256"] != config["input"]["expected_sha256"]:
        raise RuntimeError("Raw workbook SHA256 differs from frozen V4.2 protocol")
    if loaded.audit["blockers"]:
        raise RuntimeError(f"Blocking audit findings: {loaded.audit['blockers']}")

    method_config = config["two_part"]
    components, audits = _build_historical_components(
        loaded.weekly_complete,
        v4_base,
        horizons=list(method_config["horizons"]),
        cleaning_parameters=config["cleaning_v2"],
        lookback_weeks=int(method_config["training_lookback_weeks"]),
    )
    components = components.merge(
        v41_primary, on=key_columns, how="left", validate="one_to_one"
    )
    components = add_leave_one_out_references(
        components,
        minimum_peer_blocks=int(method_config["minimum_peer_blocks"]),
    )
    primary_strength = int(method_config["primary_prior_equivalent_blocks"])
    sensitivity_strengths = [
        int(value) for value in method_config["sensitivity_prior_equivalent_blocks"]
    ]
    components = add_shrunk_estimates(
        components,
        primary_strength=primary_strength,
        sensitivity_strengths=sensitivity_strengths,
    )

    epsilon = float(config["evaluation"]["numerical_log_loss_epsilon"])
    probability_methods = list(method_config["probability_methods"])
    quantity_methods = list(method_config["expected_quantity_methods"])
    probability_predictions = _probability_long(
        components,
        methods=probability_methods,
        primary_strength=primary_strength,
        sensitivity_strengths=sensitivity_strengths,
        epsilon=epsilon,
    )
    quantity_predictions = _expected_quantity_long(
        components,
        v4_predictions,
        methods=quantity_methods,
        primary_strength=primary_strength,
    )
    conditional_predictions = _conditional_quantity_long(
        components,
        primary_strength=primary_strength,
        sensitivity_strengths=sensitivity_strengths,
    )

    probability_common = _common_method_sample(probability_predictions, probability_methods)
    quantity_common = _common_method_sample(quantity_predictions, quantity_methods)
    probability_summary = _summarize_probabilities(
        probability_common,
        ["horizon_label", "horizon_weeks", "method"],
        epsilon=epsilon,
    )
    quantity_summary = _summarize_quantities(
        quantity_common, ["horizon_label", "horizon_weeks", "method"]
    )
    conditional_summary = _summarize_conditional_quantities(conditional_predictions)
    profile_summary = _profile_summary(
        probability_predictions, quantity_predictions, conditional_predictions
    )
    reliability = _reliability_bins(
        probability_common,
        [float(value) for value in config["evaluation"]["reliability_bin_edges"]],
    )

    repetitions = int(config["evaluation"]["paired_bootstrap_repetitions"])
    seed = int(config["project"]["seed"])
    probability_baselines = [
        method for method in probability_methods if method != PRIMARY_PROBABILITY_METHOD
    ]
    quantity_baselines = [
        method for method in quantity_methods if method != PRIMARY_QUANTITY_METHOD
    ]
    paired_probability = _paired_comparisons(
        probability_predictions,
        primary=PRIMARY_PROBABILITY_METHOD,
        baselines=probability_baselines,
        loss_column="brier_loss",
        repetitions=repetitions,
        seed=seed,
        quantity=False,
    )
    paired_quantity = _paired_comparisons(
        quantity_predictions,
        primary=PRIMARY_QUANTITY_METHOD,
        baselines=quantity_baselines,
        loss_column="horizon_total_abs_error",
        repetitions=repetitions,
        seed=seed,
        quantity=True,
    )
    probability_head = _origin_head_to_head(
        probability_predictions,
        target_type="probability",
        primary=PRIMARY_PROBABILITY_METHOD,
        baselines=probability_baselines,
        loss_column="brier_loss",
    )
    quantity_head = _origin_head_to_head(
        quantity_predictions,
        target_type="expected_quantity",
        primary=PRIMARY_QUANTITY_METHOD,
        baselines=quantity_baselines,
        loss_column="horizon_total_abs_error",
    )
    origin_head = pd.concat([probability_head, quantity_head], ignore_index=True)
    gates = _value_gates(
        probability_summary,
        probability_predictions,
        paired_probability,
        paired_quantity,
        origin_head,
        minimum_winning_origins=int(config["evaluation"]["minimum_winning_origins"]),
        maximum_absolute_calibration_gap=float(
            config["evaluation"]["maximum_absolute_calibration_gap"]
        ),
        quantity_required_baselines=list(
            config["evaluation"]["quantity_required_baselines"]
        ),
    )

    outcome = {
        "analysis_mode": config["project"]["analysis_mode"],
        "confirmatory": False,
        "protocol_version": config["project"]["protocol_version"],
        "raw_sha256": loaded.audit["sha256"],
        "reused_prior_results_without_rerun": True,
        "training_lookback_weeks": int(method_config["training_lookback_weeks"]),
        "primary_prior_equivalent_blocks": primary_strength,
        "probability_rows": int(len(probability_predictions)),
        "expected_quantity_rows": int(len(quantity_predictions)),
        "probability_common_sku_origins": int(
            probability_common[key_columns].drop_duplicates().shape[0]
        ),
        "quantity_common_sku_origins": int(
            quantity_common[key_columns].drop_duplicates().shape[0]
        ),
        "pure_reference_sku_origins": int(
            components["complete_history_blocks"].eq(0).sum()
        ),
        "value_gates": json.loads(gates.to_json(orient="records")),
        "management_thresholds_selected": False,
        "inventory_or_profit_claims_supported": False,
    }

    output_root = project_root / config["outputs"]["root"]
    _write_json(output_root / "input_audit.json", loaded.audit)
    _write_csv(output_root / "rolling_origin_audits.csv", audits)
    _write_csv(output_root / "two_part_components.csv", components)
    _write_csv(output_root / "probability_predictions.csv", probability_predictions)
    _write_csv(output_root / "expected_quantity_predictions.csv", quantity_predictions)
    _write_csv(output_root / "probability_summary.csv", probability_summary)
    _write_csv(output_root / "quantity_summary.csv", quantity_summary)
    _write_csv(output_root / "conditional_quantity_summary.csv", conditional_summary)
    _write_csv(output_root / "summary_by_profile.csv", profile_summary)
    _write_csv(output_root / "reliability_bins.csv", reliability)
    _write_csv(output_root / "paired_probability_comparisons.csv", paired_probability)
    _write_csv(output_root / "paired_quantity_comparisons.csv", paired_quantity)
    _write_csv(output_root / "origin_head_to_head.csv", origin_head)
    _write_csv(output_root / "value_gates.csv", gates)
    _write_json(output_root / "two_part_outcome.json", outcome)
    _write_report(
        project_root / config["outputs"]["report"],
        probability_summary=probability_summary,
        quantity_summary=quantity_summary,
        profile_summary=profile_summary,
        gates=gates,
        outcome=outcome,
    )
    return outcome


def run_two_part_validation(config_path: str | Path) -> dict[str, Any]:
    config_file = Path(config_path).resolve()
    project_root = config_file.parent.parent
    config = load_config(config_file)
    output_root = project_root / config["outputs"]["root"]
    with _exclusive_output_lock(output_root):
        return _run_two_part_unlocked(config_file)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run frozen V4.2 cluster-shrunk horizon two-part validation."
    )
    parser.add_argument("--config", default="config/pxq_two_part_v4_2.yaml")
    args = parser.parse_args()
    print(
        json.dumps(
            run_two_part_validation(args.config),
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
