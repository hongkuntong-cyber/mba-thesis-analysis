"""Exploratory fixed-120 weekly rolling 8-week Markov-PxQ vs MA4 backtest.

Methodological safeguards:
- fixed cohort = full_cluster 1 from the frozen business-feature cohort (expected 120 SKUs)
- weekly rolling origins only within the development period
- every 8-week target ends BEFORE the final 8-week holdout starts
- V2 cleaning, transition probabilities, Q and MA4 are re-estimated at every origin
- no structural pre-launch/post-discontinuation gaps are filled with zeros
- all 120 SKUs are accounted for at every origin; unavailable cases are audited
- primary comparable subset follows the frozen rolling-origin main threshold n_positive >= 5

This is exploratory method development and does not consume the final holdout.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .backtesting import _ending_contiguous_training_segment
from .cleaning_v2 import apply_v2_cleaning
from .data_audit import load_workbook_long

ROOT = Path(__file__).resolve().parent.parent
CFG = yaml.safe_load((ROOT / "config/pxq_fair_v4_4.yaml").read_text())
CLEAN_CFG = yaml.safe_load((ROOT / "config/pxq_two_part_v4_2.yaml").read_text())["cleaning_v2"]
COHORT = ROOT / CFG["cohort_file"]
OUT = ROOT / "outputs/markov_v4_19_fixed120_weekly_rolling"
H = 8
MIN_POS_MAIN = int(CFG["main_min_positive_weeks"])
BETA_A = 1.0
BETA_B = 1.0


def mase_scale(y: np.ndarray) -> float:
    if len(y) < 2:
        return float("nan")
    scale = float(np.mean(np.abs(np.diff(y))))
    return scale if scale > 0 else float("nan")


def frequency_band(rate: float) -> str:
    if rate >= 0.60:
        return "active_>=60%"
    if rate >= 0.30:
        return "medium_30-60%"
    if rate >= 0.10:
        return "lowfreq_10-30%"
    return "sparse_<10%"


def calc_markov(y: np.ndarray) -> dict:
    z = (y > 0).astype(int)
    a, b = z[:-1], z[1:]
    n00 = int(((a == 0) & (b == 0)).sum())
    n01 = int(((a == 0) & (b == 1)).sum())
    n10 = int(((a == 1) & (b == 0)).sum())
    n11 = int(((a == 1) & (b == 1)).sum())

    p01 = (n01 + BETA_A) / (n00 + n01 + BETA_A + BETA_B)
    p11 = (n11 + BETA_A) / (n10 + n11 + BETA_A + BETA_B)
    P = np.array([[1.0 - p01, p01], [1.0 - p11, p11]], dtype=float)
    state = np.array([1.0, 0.0]) if z[-1] == 0 else np.array([0.0, 1.0])
    probs: list[float] = []
    for _ in range(H):
        state = state @ P
        probs.append(float(state[1]))

    q = float(y[y > 0].mean()) if np.any(y > 0) else 0.0
    markov_weekly = np.asarray(probs, dtype=float) * q
    ma4_level = float(y[-4:].mean())
    ma4_weekly = np.repeat(ma4_level, H).astype(float)
    return {
        "current_state": int(z[-1]),
        "n_positive": int(z.sum()),
        "positive_rate": float(z.mean()),
        "n00": n00, "n01": n01, "n10": n10, "n11": n11,
        "p01": float(p01), "p11": float(p11),
        "state_dependence": float(p11 - p01),
        "q": q,
        "probs": np.asarray(probs, dtype=float),
        "markov_weekly": markov_weekly,
        "ma4_weekly": ma4_weekly,
    }


def summarize(frame: pd.DataFrame, label: str) -> dict:
    d = frame.copy()
    finite = d["mase_markov"].notna() & d["mase_ma4"].notna()
    x = d.loc[finite]
    if x.empty:
        return {"subset": label, "cases": 0, "unique_skus": int(d["sku"].nunique()) if len(d) else 0}
    diff = x["mase_ma4"] - x["mase_markov"]
    return {
        "subset": label,
        "cases": int(len(x)),
        "unique_skus": int(x["sku"].nunique()),
        "mean_mase_markov": float(x["mase_markov"].mean()),
        "mean_mase_ma4": float(x["mase_ma4"].mean()),
        "median_mase_markov": float(x["mase_markov"].median()),
        "median_mase_ma4": float(x["mase_ma4"].median()),
        "mase_lt1_markov": float((x["mase_markov"] < 1).mean()),
        "mase_lt1_ma4": float((x["mase_ma4"] < 1).mean()),
        "mean_weekly_mae_markov": float(x["mae_markov"].mean()),
        "mean_weekly_mae_ma4": float(x["mae_ma4"].mean()),
        "mean_total_abs_error_markov": float(x["total_abs_error_markov"].mean()),
        "mean_total_abs_error_ma4": float(x["total_abs_error_ma4"].mean()),
        "mean_mase_improvement_ma4_minus_markov": float(diff.mean()),
        "markov_win_rate_mase": float((diff > 0).mean()),
        "ma4_win_rate_mase": float((diff < 0).mean()),
        "tie_rate_mase": float((diff == 0).mean()),
        "mean_markov_brier_occurrence": float(x["brier_markov"].mean()),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    loaded = load_workbook_long(ROOT / CFG["input_workbook"])
    assert not loaded.audit["blockers"], loaded.audit["blockers"]
    raw = loaded.weekly_complete.copy()

    cohort = pd.read_csv(COHORT)
    fixed = sorted(cohort.loc[cohort["full_cluster"].eq(int(CFG["cohort_primary"])), "sku"].astype(str).unique())
    assert len(fixed) == int(CFG["expected_primary_skus"]), (len(fixed), CFG["expected_primary_skus"])

    last_week = pd.Timestamp(raw["week_start"].max())
    first_48 = last_week - pd.Timedelta(weeks=47)
    holdout_start = last_week - pd.Timedelta(weeks=7)
    dev_last_origin = holdout_start - pd.Timedelta(weeks=H)
    origins = list(pd.date_range(first_48, dev_last_origin, freq="7D"))
    assert origins and origins[-1] + pd.Timedelta(weeks=H - 1) < holdout_start

    raw_idx = raw.pivot(index="week_start", columns="sku", values="sales")
    rows: list[dict] = []
    audit_rows: list[dict] = []

    for oi, origin in enumerate(origins, start=1):
        clean = apply_v2_cleaning(raw.loc[raw["week_start"] < origin], **CLEAN_CFG)
        expected_test = pd.date_range(origin, periods=H, freq="7D")
        counts = {"forecastable": 0, "training_gap": 0, "incomplete_test": 0, "too_short": 0}

        for sku in fixed:
            g = clean.weekly.loc[clean.weekly["sku"].eq(sku)]
            seg = _ending_contiguous_training_segment(g, origin, "sales_v2")
            if seg.empty:
                counts["training_gap"] += 1
                continue
            if len(seg) < 4:
                counts["too_short"] += 1
                continue
            if sku not in raw_idx.columns:
                counts["incomplete_test"] += 1
                continue
            target = raw_idx[sku].reindex(expected_test)
            if not target.notna().all():
                counts["incomplete_test"] += 1
                continue

            y = seg["sales_v2"].to_numpy(float)
            actual = target.to_numpy(float)
            actual_z = (actual > 0).astype(float)
            s = calc_markov(y)
            scale = mase_scale(y)
            ae_m = np.abs(actual - s["markov_weekly"])
            ae_b = np.abs(actual - s["ma4_weekly"])
            mase_m = float(ae_m.mean() / scale) if np.isfinite(scale) else float("nan")
            mase_b = float(ae_b.mean() / scale) if np.isfinite(scale) else float("nan")
            counts["forecastable"] += 1

            rows.append({
                "origin_index": oi,
                "origin": str(pd.Timestamp(origin).date()),
                "sku": sku,
                "training_weeks": int(len(y)),
                "n_positive": s["n_positive"],
                "main_eligible_npos5": bool(s["n_positive"] >= MIN_POS_MAIN),
                "positive_rate": s["positive_rate"],
                "frequency_band": frequency_band(s["positive_rate"]),
                "current_state": s["current_state"],
                "n00": s["n00"], "n01": s["n01"], "n10": s["n10"], "n11": s["n11"],
                "p01": s["p01"], "p11": s["p11"],
                "state_dependence": s["state_dependence"],
                "q_positive_mean": s["q"],
                "recent4": json.dumps(y[-4:].tolist()),
                "weekly_probs_8": json.dumps([round(v, 6) for v in s["probs"]]),
                "expected_active_weeks_8": float(s["probs"].sum()),
                "actual_active_weeks_8": int(actual_z.sum()),
                "actual_next8": json.dumps(actual.tolist()),
                "markov_weekly_forecast": json.dumps([round(v, 6) for v in s["markov_weekly"]]),
                "ma4_weekly_forecast": json.dumps([round(v, 6) for v in s["ma4_weekly"]]),
                "actual_sum_8": float(actual.sum()),
                "markov_sum_8": float(s["markov_weekly"].sum()),
                "ma4_sum_8": float(s["ma4_weekly"].sum()),
                "mae_markov": float(ae_m.mean()),
                "mae_ma4": float(ae_b.mean()),
                "mase_scale": scale,
                "mase_markov": mase_m,
                "mase_ma4": mase_b,
                "total_abs_error_markov": float(abs(s["markov_weekly"].sum() - actual.sum())),
                "total_abs_error_ma4": float(abs(s["ma4_weekly"].sum() - actual.sum())),
                "bias_sum_markov": float(s["markov_weekly"].sum() - actual.sum()),
                "bias_sum_ma4": float(s["ma4_weekly"].sum() - actual.sum()),
                "brier_markov": float(np.mean((s["probs"] - actual_z) ** 2)),
            })

        audit_rows.append({
            "origin_index": oi,
            "origin": str(pd.Timestamp(origin).date()),
            "fixed_skus": len(fixed),
            **counts,
            "accounted_total": counts["forecastable"] + counts["training_gap"] + counts["incomplete_test"] + counts["too_short"],
        })

    out = pd.DataFrame(rows)
    audit = pd.DataFrame(audit_rows)
    assert (audit["accounted_total"] == len(fixed)).all()

    out["mase_diff_ma4_minus_markov"] = out["mase_ma4"] - out["mase_markov"]
    out["winner_mase"] = np.where(
        out["mase_diff_ma4_minus_markov"] > 0, "Markov",
        np.where(out["mase_diff_ma4_minus_markov"] < 0, "MA4", "Tie")
    )
    out["winner_total_ae"] = np.where(
        out["total_abs_error_markov"] < out["total_abs_error_ma4"], "Markov",
        np.where(out["total_abs_error_ma4"] < out["total_abs_error_markov"], "MA4", "Tie")
    )

    out.to_csv(OUT / "fixed120_weekly_rolling_cases.csv", index=False)
    audit.to_csv(OUT / "coverage_audit.csv", index=False)

    summary_rows = [
        summarize(out, "all_forecastable"),
        summarize(out.loc[out["main_eligible_npos5"]], "main_n_positive>=5"),
        summarize(out.loc[~out["main_eligible_npos5"]], "low_information_n_positive<5"),
    ]
    for band, g in out.groupby("frequency_band", sort=False):
        summary_rows.append(summarize(g, f"band:{band}"))
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT / "summary_overall.csv", index=False)

    by_origin = []
    for origin, g in out.groupby("origin", sort=True):
        row = summarize(g.loc[g["main_eligible_npos5"]], str(origin))
        row["origin"] = str(origin)
        row["forecastable_all"] = int(len(g))
        row["forecastable_main"] = int(g["main_eligible_npos5"].sum())
        by_origin.append(row)
    pd.DataFrame(by_origin).to_csv(OUT / "summary_by_origin.csv", index=False)

    by_sku = (
        out.loc[out["main_eligible_npos5"]]
        .groupby("sku")
        .agg(
            origins=("origin", "count"),
            mean_positive_rate=("positive_rate", "mean"),
            mean_state_dependence=("state_dependence", "mean"),
            mean_mase_markov=("mase_markov", "mean"),
            mean_mase_ma4=("mase_ma4", "mean"),
            markov_win_rate=("winner_mase", lambda x: float((x == "Markov").mean())),
            mean_total_ae_markov=("total_abs_error_markov", "mean"),
            mean_total_ae_ma4=("total_abs_error_ma4", "mean"),
        )
        .reset_index()
    )
    by_sku["mean_mase_improvement"] = by_sku["mean_mase_ma4"] - by_sku["mean_mase_markov"]
    by_sku.to_csv(OUT / "summary_by_sku.csv", index=False)

    meta = {
        "analysis_mode": "exploratory_method_development",
        "fixed_skus": len(fixed),
        "horizon_weeks": H,
        "rolling_origins": len(origins),
        "first_origin": str(pd.Timestamp(origins[0]).date()),
        "last_development_origin": str(pd.Timestamp(origins[-1]).date()),
        "holdout_start": str(holdout_start.date()),
        "last_observed_week": str(last_week.date()),
        "holdout_consumed": False,
        "main_min_positive_weeks": MIN_POS_MAIN,
        "probability_smoothing": "Beta(1,1)",
        "q_method": "mean positive weekly V2 demand over contiguous training history",
        "ma4_method": "mean of last 4 V2 training weeks, repeated for 8 weeks",
    }
    (OUT / "run_metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    main_summary = summary.loc[summary["subset"].eq("main_n_positive>=5")].iloc[0].to_dict()
    print("=== RUN METADATA ===")
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    print("\n=== COVERAGE ===")
    print(audit.to_string(index=False))
    print("\n=== OVERALL SUMMARY ===")
    print(summary.to_string(index=False))
    print("\n=== MAIN SUBSET ===")
    print(json.dumps(main_summary, ensure_ascii=False, indent=2))
    print("\n=== TOP 10 MARKOV IMPROVEMENTS BY SKU ===")
    print(by_sku.sort_values("mean_mase_improvement", ascending=False).head(10).to_string(index=False))
    print("\n=== BOTTOM 10 / MA4-FAVORED BY SKU ===")
    print(by_sku.sort_values("mean_mase_improvement", ascending=True).head(10).to_string(index=False))


if __name__ == "__main__":
    main()
