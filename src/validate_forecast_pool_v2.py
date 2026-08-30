from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .evaluation import summarize_predictions
from .forecast_pool_v2 import MODELS, _common_mase_sample


def validate(output_root: str | Path) -> dict[str, Any]:
    root = Path(output_root)
    predictions = pd.read_csv(root / "rolling_origin_predictions_with_sba.csv")
    formal = predictions.loc[predictions["model"].isin(MODELS)].copy()
    holdout = formal.loc[formal["origin_index"].eq(6)].copy()
    common_holdout = _common_mase_sample(holdout)

    key_duplicates = int(
        predictions.duplicated(["origin_index", "sku", "model"], keep=False).sum()
    )
    finite_scale = np.isfinite(formal["mase_scale"]) & formal["mase_scale"].gt(0)
    formula_error = np.abs(
        formal.loc[finite_scale, "mase"]
        - formal.loc[finite_scale, "mae"] / formal.loc[finite_scale, "mase_scale"]
    )
    scale_consistency = (
        predictions.groupby(["origin_index", "sku"])["mase_scale"]
        .nunique(dropna=False)
        .max()
    )
    actual_consistency = (
        predictions.groupby(["origin_index", "sku"])["actual_sum"].nunique().max()
    )
    saved_summary = pd.read_csv(root / "holdout_common_mase.csv").sort_values("model")
    recomputed = summarize_predictions(common_holdout, ["model"]).sort_values("model")
    summary_columns = [
        "mean_mase",
        "median_mase",
        "mase_lt_1_share",
        "mean_mae",
        "aggregate_wape",
        "aggregate_bias",
    ]
    summary_error = max(
        float(
            np.max(
                np.abs(
                    saved_summary[column].to_numpy(dtype=float)
                    - recomputed[column].to_numpy(dtype=float)
                )
            )
        )
        for column in summary_columns
    )
    reconciliation = json.loads(
        (root / "v1_reconciliation.json").read_text(encoding="utf-8")
    )
    checks = {
        "unique_sku_origin_model_rows": key_duplicates == 0,
        "mase_formula_matches_mae_over_training_scale": float(formula_error.max()) <= 1e-12,
        "one_training_scale_per_sku_origin": int(scale_consistency) == 1,
        "one_actual_volume_per_sku_origin": int(actual_consistency) == 1,
        "all_forecast_sums_nonnegative": bool(predictions["forecast_sum"].ge(0).all()),
        "saved_common_holdout_summary_recomputes": summary_error <= 1e-12,
        "v1_core_predictions_reconcile": reconciliation.get("status") == "passed",
    }
    result = {
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "diagnostics": {
            "duplicate_rows": key_duplicates,
            "maximum_mase_formula_error": float(formula_error.max()),
            "maximum_saved_summary_error": summary_error,
            "holdout_native_skus": int(holdout["sku"].nunique()),
            "holdout_common_mase_skus": int(common_holdout["sku"].nunique()),
            "holdout_sba_unavailable_skus": int(
                holdout["sku"].nunique()
                - holdout.loc[holdout["model"].eq("SBA"), "sku"].nunique()
            ),
        },
    }
    (root / "validation_checks.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if result["status"] != "passed":
        raise RuntimeError(f"Forecast-pool validation failed: {result}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the exploratory V2 forecast pool outputs.")
    parser.add_argument(
        "--output-root", default="outputs/forecast_pool_v2_exploratory"
    )
    args = parser.parse_args()
    print(json.dumps(validate(args.output_root), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
