from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CleaningResult:
    weekly: pd.DataFrame
    intervals: pd.DataFrame
    summary: dict[str, Any]


def _contiguous_segments(dates: pd.Series) -> list[tuple[int, int]]:
    if dates.empty:
        return []
    values = pd.to_datetime(dates).reset_index(drop=True)
    starts = [0]
    for idx in range(1, len(values)):
        if values.iloc[idx] - values.iloc[idx - 1] != pd.Timedelta(days=7):
            starts.append(idx)
    starts.append(len(values))
    return list(zip(starts[:-1], starts[1:]))


def _zero_runs(values: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    idx = 0
    while idx < len(values):
        if values[idx] != 0:
            idx += 1
            continue
        start = idx
        while idx + 1 < len(values) and values[idx + 1] == 0:
            idx += 1
        runs.append((start, idx))
        idx += 1
    return runs


def apply_v2_cleaning(
    weekly: pd.DataFrame,
    *,
    zero_run_min: int = 2,
    zero_run_max: int = 12,
    context_window_max: int = 12,
    minimum_positive_share: float = 0.75,
    minimum_positive_weeks_each_side: int = 3,
    probability_cutoff: float = 0.05,
    correct_leading_or_trailing_runs: bool = False,
) -> CleaningResult:
    required = {"sku", "week_start", "sales"}
    missing = required.difference(weekly.columns)
    if missing:
        raise ValueError(f"Missing weekly columns: {sorted(missing)}")
    if correct_leading_or_trailing_runs:
        raise ValueError("Protocol forbids correcting leading or trailing zero runs")

    output = weekly.copy().sort_values(["sku", "week_start"]).reset_index(drop=True)
    output["sales_raw"] = output["sales"].astype(float)
    output["sales_v2"] = output["sales_raw"].copy()
    output["v2_corrected"] = False
    interval_records: list[dict[str, Any]] = []
    candidate_count = 0

    for sku, sku_frame in output.groupby("sku", sort=True):
        positions = sku_frame.index.to_numpy()
        local = sku_frame.reset_index(drop=True)
        for segment_start, segment_end in _contiguous_segments(local["week_start"]):
            segment = local.iloc[segment_start:segment_end]
            values = segment["sales_raw"].to_numpy(dtype=float)
            if np.isnan(values).any():
                continue
            for run_start, run_end in _zero_runs(values):
                length = run_end - run_start + 1
                internal = run_start > 0 and run_end < len(values) - 1
                if not internal or not (zero_run_min <= length <= zero_run_max):
                    continue
                if values[run_start - 1] <= 0 or values[run_end + 1] <= 0:
                    continue
                candidate_count += 1
                pre = values[max(0, run_start - context_window_max) : run_start]
                post = values[run_end + 1 : min(len(values), run_end + 1 + context_window_max)]
                pre_positive = pre[pre > 0]
                post_positive = post[post > 0]
                pre_share = len(pre_positive) / len(pre) if len(pre) else 0.0
                post_share = len(post_positive) / len(post) if len(post) else 0.0
                eligible_context = (
                    len(pre_positive) >= minimum_positive_weeks_each_side
                    and len(post_positive) >= minimum_positive_weeks_each_side
                    and pre_share >= minimum_positive_share
                    and post_share >= minimum_positive_share
                )
                p_hat = (
                    (len(pre_positive) + len(post_positive)) / (len(pre) + len(post))
                    if len(pre) + len(post)
                    else np.nan
                )
                p_zero = (1.0 - p_hat) ** length if np.isfinite(p_hat) else np.nan
                corrected = bool(eligible_context and p_zero < probability_cutoff)
                median_pre = float(np.median(pre_positive)) if len(pre_positive) else np.nan
                median_post = float(np.median(post_positive)) if len(post_positive) else np.nan

                global_local_start = segment_start + run_start
                global_local_end = segment_start + run_end
                affected_positions = positions[global_local_start : global_local_end + 1]
                if corrected:
                    interpolated = np.linspace(median_pre, median_post, length + 2)[1:-1]
                    output.loc[affected_positions, "sales_v2"] = interpolated
                    output.loc[affected_positions, "v2_corrected"] = True

                interval_records.append(
                    {
                        "sku": sku,
                        "start_week": segment.iloc[run_start]["week_start"],
                        "end_week": segment.iloc[run_end]["week_start"],
                        "length": length,
                        "pre_weeks": len(pre),
                        "post_weeks": len(post),
                        "pre_positive_weeks": len(pre_positive),
                        "post_positive_weeks": len(post_positive),
                        "pre_positive_share": pre_share,
                        "post_positive_share": post_share,
                        "p_hat": p_hat,
                        "p_zero": p_zero,
                        "median_pre": median_pre,
                        "median_post": median_post,
                        "corrected": corrected,
                    }
                )

    intervals = pd.DataFrame(interval_records)
    corrected_intervals = int(intervals["corrected"].sum()) if not intervals.empty else 0
    corrected_skus = int(intervals.loc[intervals.get("corrected", False), "sku"].nunique()) if not intervals.empty else 0
    corrected_weeks = int(output["v2_corrected"].sum())
    raw_total = float(output["sales_raw"].sum())
    v2_total = float(output["sales_v2"].sum())
    summary = {
        "candidate_intervals": candidate_count,
        "corrected_intervals": corrected_intervals,
        "corrected_skus": corrected_skus,
        "corrected_sku_weeks": corrected_weeks,
        "corrected_share": corrected_weeks / len(output) if len(output) else np.nan,
        "raw_total": raw_total,
        "v2_total": v2_total,
        "absolute_change": v2_total - raw_total,
        "relative_change": (v2_total - raw_total) / raw_total if raw_total else np.nan,
    }
    return CleaningResult(output, intervals, summary)
