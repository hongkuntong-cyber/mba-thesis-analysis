from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PERIOD_PATTERN = re.compile(r"^(\d{4}-\d{2}-\d{2})\s*~\s*(\d{4}-\d{2}-\d{2})$")


@dataclass(frozen=True)
class WorkbookLoadResult:
    weekly_all: pd.DataFrame
    weekly_complete: pd.DataFrame
    raw_long: pd.DataFrame
    audit: dict[str, Any]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_period_header(value: object) -> tuple[pd.Timestamp, pd.Timestamp]:
    match = PERIOD_PATTERN.match(str(value).strip())
    if not match:
        raise ValueError(f"Unrecognized weekly period header: {value!r}")
    start = pd.Timestamp(match.group(1))
    end = pd.Timestamp(match.group(2))
    if end < start:
        raise ValueError(f"Period end precedes start: {value!r}")
    if (end - start).days + 1 > 7:
        raise ValueError(f"Period spans more than seven days: {value!r}")
    return start, end


def monday_of_week(timestamp: pd.Timestamp) -> pd.Timestamp:
    return timestamp.normalize() - pd.Timedelta(days=int(timestamp.weekday()))


def _covered_days(group: pd.DataFrame) -> int:
    covered: set[pd.Timestamp] = set()
    for start, end in group[["period_start", "period_end"]].itertuples(index=False):
        covered.update(pd.date_range(start, end, freq="D"))
    return len(covered)


def load_workbook_long(
    path: str | Path,
    metadata_columns: int = 8,
    sku_column: str = "SKU",
    minimum_covered_days: int = 7,
) -> WorkbookLoadResult:
    workbook_path = Path(path)
    excel = pd.ExcelFile(workbook_path)
    frames: list[pd.DataFrame] = []
    sheet_details: list[dict[str, Any]] = []
    duplicate_rows: list[pd.DataFrame] = []

    for sheet_name in excel.sheet_names:
        frame = pd.read_excel(workbook_path, sheet_name=sheet_name)
        if sku_column not in frame.columns:
            raise ValueError(f"Sheet {sheet_name!r} does not contain {sku_column!r}")
        if frame.shape[1] <= metadata_columns:
            raise ValueError(f"Sheet {sheet_name!r} has no weekly columns")

        frame[sku_column] = frame[sku_column].astype("string").str.strip()
        weekly_columns = list(frame.columns[metadata_columns:])
        period_map = {column: parse_period_header(column) for column in weekly_columns}
        duplicate_skus = frame.loc[frame[sku_column].duplicated(keep=False), sku_column]
        if not duplicate_skus.empty:
            duplicate_rows.append(
                pd.DataFrame({"sheet": sheet_name, "sku": duplicate_skus.astype(str)})
            )

        long = frame[[sku_column, *weekly_columns]].melt(
            id_vars=[sku_column],
            value_vars=weekly_columns,
            var_name="period_label",
            value_name="sales_raw",
        )
        long = long.rename(columns={sku_column: "sku"})
        long["sheet"] = sheet_name
        long["period_start"] = long["period_label"].map(lambda value: period_map[value][0])
        long["period_end"] = long["period_label"].map(lambda value: period_map[value][1])
        long["period_days"] = (
            (long["period_end"] - long["period_start"]).dt.days + 1
        ).astype(int)
        long["iso_week_start"] = long["period_start"].map(monday_of_week)
        long["sales"] = pd.to_numeric(long["sales_raw"], errors="coerce")
        frames.append(long)

        sheet_details.append(
            {
                "sheet": sheet_name,
                "rows": int(frame.shape[0]),
                "columns": int(frame.shape[1]),
                "weekly_columns": int(len(weekly_columns)),
                "unique_skus": int(frame[sku_column].nunique(dropna=True)),
                "duplicate_sku_rows": int(duplicate_skus.shape[0]),
                "period_start": str(min(start for start, _ in period_map.values()).date()),
                "period_end": str(max(end for _, end in period_map.values()).date()),
            }
        )

    raw_long = pd.concat(frames, ignore_index=True)
    invalid_numeric = raw_long["sales_raw"].notna() & raw_long["sales"].isna()
    missing_sales = raw_long["sales_raw"].isna()
    negative_sales = raw_long["sales"].lt(0).fillna(False)
    duplicate_sku_period = raw_long.duplicated(
        subset=["sku", "period_start", "period_end"], keep=False
    )

    grouped_rows: list[dict[str, Any]] = []
    for (sku, iso_week_start), group in raw_long.groupby(
        ["sku", "iso_week_start"], sort=True, dropna=False
    ):
        grouped_rows.append(
            {
                "sku": str(sku),
                "week_start": pd.Timestamp(iso_week_start),
                "week_end": pd.Timestamp(iso_week_start) + pd.Timedelta(days=6),
                "sales": group["sales"].sum(min_count=1),
                "covered_days": _covered_days(group),
                "fragment_count": int(group.shape[0]),
                "source_sheets": ",".join(sorted(group["sheet"].astype(str).unique())),
                "source_periods": "|".join(sorted(group["period_label"].astype(str).unique())),
            }
        )
    weekly_all = pd.DataFrame(grouped_rows).sort_values(["sku", "week_start"])
    weekly_all["is_complete_week"] = weekly_all["covered_days"].eq(minimum_covered_days)
    weekly_complete = weekly_all.loc[weekly_all["is_complete_week"]].copy()

    duplicate_after = weekly_complete.duplicated(["sku", "week_start"], keep=False)
    incomplete = weekly_all.loc[~weekly_all["is_complete_week"]]
    overlap_excess = weekly_all["covered_days"].gt(7)

    blockers: list[str] = []
    if duplicate_rows:
        blockers.append("duplicate SKU rows exist within at least one worksheet")
    if int(duplicate_sku_period.sum()) > 0:
        blockers.append("duplicate SKU-period source rows exist")
    if int(invalid_numeric.sum()) > 0:
        blockers.append("non-numeric sales cells exist")
    if int(missing_sales.sum()) > 0:
        blockers.append("missing sales cells exist")
    if int(negative_sales.sum()) > 0:
        blockers.append("negative sales cells exist")
    if int(overlap_excess.sum()) > 0:
        blockers.append("overlapping period fragments cover more than seven days")
    if int(duplicate_after.sum()) > 0:
        blockers.append("duplicate SKU-week rows remain after consolidation")

    audit: dict[str, Any] = {
        "source_file": str(workbook_path),
        "sha256": sha256_file(workbook_path),
        "sheet_names": excel.sheet_names,
        "sheet_details": sheet_details,
        "raw_cells_at_sku_period_grain": int(raw_long.shape[0]),
        "unique_skus": int(raw_long["sku"].nunique(dropna=True)),
        "unique_period_fragments": int(
            raw_long[["period_start", "period_end"]].drop_duplicates().shape[0]
        ),
        "consolidated_sku_weeks": int(weekly_all.shape[0]),
        "complete_sku_weeks": int(weekly_complete.shape[0]),
        "incomplete_sku_weeks": int(incomplete.shape[0]),
        "split_week_sku_weeks": int(weekly_all["fragment_count"].gt(1).sum()),
        "invalid_numeric_cells": int(invalid_numeric.sum()),
        "missing_sales_cells": int(missing_sales.sum()),
        "negative_sales_cells": int(negative_sales.sum()),
        "duplicate_source_sku_period_rows": int(duplicate_sku_period.sum()),
        "duplicate_consolidated_sku_weeks": int(duplicate_after.sum()),
        "week_start_min": str(weekly_complete["week_start"].min().date()),
        "week_start_max": str(weekly_complete["week_start"].max().date()),
        "sales_sum_complete_weeks": float(weekly_complete["sales"].sum()),
        "blockers": blockers,
    }
    return WorkbookLoadResult(
        weekly_all=weekly_all.reset_index(drop=True),
        weekly_complete=weekly_complete.reset_index(drop=True),
        raw_long=raw_long.reset_index(drop=True),
        audit=audit,
    )


def save_audit(audit: dict[str, Any], output_path: str | Path) -> None:
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
