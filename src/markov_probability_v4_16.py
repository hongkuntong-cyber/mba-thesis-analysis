"""V4.16 exploratory two-state Markov occurrence-probability backtest.

Purpose
-------
Reuse the completed V4.4 fixed-cohort/tail-origin preparation without rerunning
clustering, cleaning, or benchmark generation.  The V4.4 components table stores
the origin-specific V2 training sequence (`train_values`) and observed future
event label.  This script adds only an equal-weight first-order two-state Markov
occurrence model with Beta(1,1) smoothing.

This is exploratory.  It does not select an information threshold by predictive
performance and does not modify the Q/quantity model.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "outputs/pxq_fair_v4_4"
OUT = ROOT / "outputs/markov_v4_16_exploratory"
KEY = ["horizon_weeks", "sku", "origin"]
COHORTS = {"active120": 1, "sparse77": 2}
SUPPORT_THRESHOLDS = [1, 2, 3, 5, 10]
SEED = 42
BOOTSTRAP_REPS = 1000


def _weighted_metrics(g: pd.DataFrame, probability_col: str) -> dict:
    p = g[probability_col].to_numpy(float)
    z = g["actual_event"].to_numpy(int)
    counts = g.groupby("sku")["sku"].transform("size").to_numpy(float)
    w = 1.0 / counts
    wsum = w.sum()
    mean_p = float(np.average(p, weights=w))
    actual_rate = float(np.average(z, weights=w))
    brier = float(np.average((p - z) ** 2, weights=w))
    bins = np.minimum((p * 5).astype(int), 4)
    ece = 0.0
    for b in np.unique(bins):
        m = bins == b
        ece += abs(float(np.average(p[m] - z[m], weights=w[m]))) * float(w[m].sum() / wsum)
    auc = float(roc_auc_score(z, p, sample_weight=w)) if len(np.unique(z)) == 2 else np.nan
    return {
        "n": int(len(g)),
        "n_skus": int(g.sku.nunique()),
        "mean_probability": mean_p,
        "actual_rate": actual_rate,
        "gap": mean_p - actual_rate,
        "brier": brier,
        "ece": float(ece),
        "auc": auc,
    }


def _transition_estimates(values: np.ndarray, horizon: int) -> dict:
    states = (values > 0).astype(np.int8)
    if len(states) < 2:
        return {
            "n00": 0, "n01": 0, "n10": 0, "n11": 0,
            "current_state": int(states[-1]) if len(states) else -1,
            "current_state_transitions": 0,
            "p01": np.nan, "p11": np.nan, "markov_probability": np.nan,
        }
    a, b = states[:-1], states[1:]
    n00 = int(np.sum((a == 0) & (b == 0)))
    n01 = int(np.sum((a == 0) & (b == 1)))
    n10 = int(np.sum((a == 1) & (b == 0)))
    n11 = int(np.sum((a == 1) & (b == 1)))
    p01 = (n01 + 1.0) / (n00 + n01 + 2.0)
    p11 = (n11 + 1.0) / (n10 + n11 + 2.0)
    p00 = 1.0 - p01
    p10 = 1.0 - p11
    current = int(states[-1])
    support = n00 + n01 if current == 0 else n10 + n11
    if current == 0:
        ph = 1.0 - p00 ** horizon
    else:
        ph = 1.0 - p10 * (p00 ** max(horizon - 1, 0))
    return {
        "n00": n00, "n01": n01, "n10": n10, "n11": n11,
        "current_state": current,
        "current_state_transitions": int(support),
        "p01": float(p01), "p11": float(p11),
        "p11_minus_p01": float(p11 - p01),
        "markov_probability": float(ph),
    }


def _bootstrap_pairwise(g: pd.DataFrame, baseline_col: str) -> dict:
    d = (g["markov_probability"] - g.actual_event) ** 2 - (g[baseline_col] - g.actual_event) ** 2
    sku_delta = d.groupby(g.sku).mean().to_numpy(float)
    rng = np.random.default_rng(SEED)
    boot = rng.choice(sku_delta, size=(BOOTSTRAP_REPS, len(sku_delta)), replace=True).mean(axis=1)
    markov_brier = _weighted_metrics(g, "markov_probability")["brier"]
    base_brier = _weighted_metrics(g, baseline_col)["brier"]
    return {
        "n_pairs": int(len(g)),
        "n_skus": int(g.sku.nunique()),
        "delta_brier_markov_minus_baseline": float(sku_delta.mean()),
        "ci_low": float(np.quantile(boot, 0.025)),
        "ci_high": float(np.quantile(boot, 0.975)),
        "brier_markov": markov_brier,
        "brier_baseline": base_brier,
        "brier_skill_score": float(1.0 - markov_brier / base_brier) if base_brier > 0 else np.nan,
        "pair_win_share": float((d < 0).mean()),
    }


def _reliability_rows(g: pd.DataFrame, probability_col: str, model: str, prefix: dict) -> list[dict]:
    p = g[probability_col].to_numpy(float)
    bins = np.minimum((p * 5).astype(int), 4)
    rows = []
    for b in sorted(np.unique(bins)):
        bg = g.loc[bins == b]
        rows.append(prefix | {
            "model": model,
            "bin": int(b),
            "n": int(len(bg)),
            "n_skus": int(bg.sku.nunique()),
            "mean_probability": float(bg[probability_col].mean()),
            "actual_rate": float(bg.actual_event.mean()),
        })
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    components = pd.read_csv(SOURCE / "components.csv")
    components["origin"] = pd.to_datetime(components.origin)
    required = set(KEY + ["full_cluster", "actual_event", "train_values", "ma4_forecast"])
    missing = required - set(components.columns)
    if missing:
        raise RuntimeError(f"components.csv missing columns: {sorted(missing)}")
    if components.duplicated(KEY).any():
        raise RuntimeError("components.csv has duplicate horizon/sku/origin keys")

    rows = []
    for r in components.itertuples(index=False):
        vals = np.asarray(json.loads(r.train_values), dtype=float)
        est = _transition_estimates(vals, int(r.horizon_weeks))
        rows.append({
            "horizon_weeks": int(r.horizon_weeks),
            "sku": r.sku,
            "origin": r.origin,
            "full_cluster": int(r.full_cluster),
            "calendar_group": getattr(r, "calendar_group", ""),
            "tail_rank": int(getattr(r, "tail_rank", 0)),
            "training_weeks": int(len(vals)),
            "actual_event": int(r.actual_event),
            "actual_sum": float(r.actual_sum),
            "ma4_forecast": float(r.ma4_forecast),
        } | est)
    pred = pd.DataFrame(rows)

    oldp = pd.read_csv(SOURCE / "probability_predictions.csv")
    oldp["origin"] = pd.to_datetime(oldp.origin)
    wide = oldp.pivot(index=KEY, columns="method", values="probability").reset_index()
    wide.columns.name = None
    rename = {
        "BlockFrequency": "block_frequency_probability",
        "Laplace": "laplace_probability",
        "EnterpriseFrequency": "enterprise_probability",
    }
    wide = wide.rename(columns=rename)
    pred = pred.merge(wide, on=KEY, how="left", validate="one_to_one")
    pred.to_csv(OUT / "markov_predictions.csv", index=False)

    coverage_rows = []
    metric_rows = []
    pair_rows = []
    reliability_rows = []
    diag_rows = []

    baseline_cols = [
        ("BlockFrequency", "block_frequency_probability"),
        ("Laplace", "laplace_probability"),
        ("EnterpriseFrequency", "enterprise_probability"),
    ]

    for cname, cid in COHORTS.items():
        cg = pred.loc[pred.full_cluster.eq(cid)].copy()
        for h, hg in cg.groupby("horizon_weeks"):
            diag_rows.append({
                "cohort": cname,
                "horizon_weeks": int(h),
                "n_cases": int(len(hg)),
                "n_skus": int(hg.sku.nunique()),
                "median_training_weeks": float(hg.training_weeks.median()),
                "median_current_state_transitions": float(hg.current_state_transitions.median()),
                "mean_p01": float(hg.p01.mean()),
                "mean_p11": float(hg.p11.mean()),
                "mean_p11_minus_p01": float(hg.p11_minus_p01.mean()),
                "median_p11_minus_p01": float(hg.p11_minus_p01.median()),
                "share_p11_gt_p01": float((hg.p11 > hg.p01).mean()),
                "current_state_1_share": float((hg.current_state == 1).mean()),
            })
            for threshold in SUPPORT_THRESHOLDS:
                sg = hg.loc[hg.current_state_transitions.ge(threshold) & np.isfinite(hg.markov_probability)].copy()
                coverage_rows.append({
                    "cohort": cname,
                    "horizon_weeks": int(h),
                    "min_current_state_transitions": int(threshold),
                    "planned_cases": int(len(hg)),
                    "planned_skus": int(hg.sku.nunique()),
                    "supported_cases": int(len(sg)),
                    "supported_skus": int(sg.sku.nunique()),
                    "case_coverage": float(len(sg) / len(hg)) if len(hg) else np.nan,
                    "sku_coverage": float(sg.sku.nunique() / hg.sku.nunique()) if hg.sku.nunique() else np.nan,
                })
                if sg.empty:
                    continue
                for scope, ss in [("all", sg), ("ma4_zero", sg.loc[np.isclose(sg.ma4_forecast, 0.0)])]:
                    if ss.empty:
                        continue
                    prefix = {
                        "cohort": cname,
                        "horizon_weeks": int(h),
                        "min_current_state_transitions": int(threshold),
                        "scope": scope,
                    }
                    metric_rows.append(prefix | {"model": "MarkovBeta11"} | _weighted_metrics(ss, "markov_probability"))
                    reliability_rows.extend(_reliability_rows(ss, "markov_probability", "MarkovBeta11", prefix))
                    for bname, bcol in baseline_cols:
                        common = ss.loc[np.isfinite(ss[bcol])].copy()
                        if common.empty:
                            continue
                        metric_rows.append(prefix | {"model": bname} | _weighted_metrics(common, bcol))
                        reliability_rows.extend(_reliability_rows(common, bcol, bname, prefix))
                        pair_rows.append(prefix | {"baseline": bname} | _bootstrap_pairwise(common, bcol))

    coverage = pd.DataFrame(coverage_rows)
    metrics = pd.DataFrame(metric_rows)
    pairs = pd.DataFrame(pair_rows)
    reliability = pd.DataFrame(reliability_rows)
    diagnostics = pd.DataFrame(diag_rows)

    coverage.to_csv(OUT / "support_coverage.csv", index=False)
    metrics.to_csv(OUT / "probability_metrics.csv", index=False)
    pairs.to_csv(OUT / "paired_brier.csv", index=False)
    reliability.to_csv(OUT / "reliability_bins.csv", index=False)
    diagnostics.to_csv(OUT / "transition_diagnostics.csv", index=False)

    # Primary display only: threshold=1 is the least restrictive evidence-supported
    # view.  No threshold is selected by error minimization; all thresholds are saved.
    primary = metrics.loc[
        metrics.min_current_state_transitions.eq(1)
        & metrics.scope.eq("all")
        & metrics.cohort.eq("active120")
    ].copy()
    pair_primary = pairs.loc[
        pairs.min_current_state_transitions.eq(1)
        & pairs.scope.eq("all")
        & pairs.cohort.eq("active120")
    ].copy()
    cov_primary = coverage.loc[
        coverage.min_current_state_transitions.eq(1)
        & coverage.cohort.eq("active120")
    ].copy()

    md = [
        "# V4.16 Exploratory Markov Probability Backtest",
        "",
        "This run reuses V4.4 origin-specific V2 training sequences and observed event labels.",
        "It adds only an equal-weight first-order two-state Markov occurrence model with Beta(1,1) smoothing.",
        "No Q model or recent-history weighting is changed, and no support threshold is selected by predictive performance.",
        "",
        "## Active120: threshold >= 1 current-state transition",
        "",
        primary.to_markdown(index=False),
        "",
        "## Pairwise Brier comparisons",
        "",
        pair_primary.to_markdown(index=False),
        "",
        "## Coverage",
        "",
        cov_primary.to_markdown(index=False),
        "",
        "## Interpretation boundary",
        "",
        "These are exploratory results on previously used retrospective origins. They are not an independent final holdout confirmation.",
    ]
    (OUT / "summary.md").write_text("\n".join(md), encoding="utf-8")

    audit = {
        "analysis": "v4.16 exploratory Markov probability",
        "source_components": "outputs/pxq_fair_v4_4/components.csv",
        "source_probability_predictions": "outputs/pxq_fair_v4_4/probability_predictions.csv",
        "model": "two-state first-order Markov; equal-weight transitions; Beta(1,1) smoothing",
        "support_thresholds_reported": SUPPORT_THRESHOLDS,
        "threshold_selection_by_validation_loss": False,
        "quantity_model_changed": False,
        "recent_weighting_used": False,
        "seed": SEED,
        "bootstrap_repetitions": BOOTSTRAP_REPS,
        "n_prediction_rows": int(len(pred)),
    }
    (OUT / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print((OUT / "summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
