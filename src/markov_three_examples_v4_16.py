"""Three representative SKU examples for an 8-week Markov-PxQ vs MA4 comparison.

Exploratory illustration only. Selection uses training characteristics only and
never future outcomes/errors. The existing V4.4 9-week tail-origin cases are
used solely because they contain at least 8 future observed weeks; the target is
re-cut to the first 8 weeks and MA4 is recomputed for an 8-week horizon.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "outputs/pxq_fair_v4_4/components.csv"
OUT = ROOT / "outputs/markov_v4_16_three_examples"
H = 8


def markov_stats(y: np.ndarray) -> dict:
    z = (y > 0).astype(int)
    a, b = z[:-1], z[1:]
    n00 = int(((a == 0) & (b == 0)).sum())
    n01 = int(((a == 0) & (b == 1)).sum())
    n10 = int(((a == 1) & (b == 0)).sum())
    n11 = int(((a == 1) & (b == 1)).sum())
    # Beta(1,1) smoothing, fixed before seeing outcomes.
    p01 = (n01 + 1.0) / (n00 + n01 + 2.0)
    p11 = (n11 + 1.0) / (n10 + n11 + 2.0)
    P = np.array([[1 - p01, p01], [1 - p11, p11]], float)
    state = np.array([1.0, 0.0]) if z[-1] == 0 else np.array([0.0, 1.0])
    probs = []
    for _ in range(H):
        state = state @ P
        probs.append(float(state[1]))
    q = float(y[y > 0].mean()) if np.any(y > 0) else 0.0
    return {
        "positive_rate": float(z.mean()),
        "recent4_positive_rate": float(z[-4:].mean()),
        "current_state": int(z[-1]),
        "n00": n00, "n01": n01, "n10": n10, "n11": n11,
        "p01": float(p01), "p11": float(p11),
        "state_dependence": float(p11 - p01),
        "expected_demand_weeks_8": float(sum(probs)),
        "weekly_probabilities": probs,
        "q_positive_week_mean": q,
        "markov_pxq_forecast_8": float(sum(probs) * q),
        "ma4_forecast_8": float(y[-4:].mean() * H),
    }


def pick_examples(df: pd.DataFrame) -> pd.DataFrame:
    # Only one tail-origin case per SKU, using horizon=9 so 8 complete future
    # weeks are available. We select without using test_values or actual_sum.
    d = df.loc[(df.horizon_weeks == 9) & (df.tail_rank == 1)].copy()
    stats = []
    for r in d.itertuples(index=False):
        y = np.asarray(json.loads(r.train_values), dtype=float)
        s = markov_stats(y)
        stats.append({"sku": r.sku, "full_cluster": int(r.full_cluster), "origin": r.origin,
                      "training_weeks": len(y), **s})
    s = pd.DataFrame(stats)

    # Minimum support: at least 12 historical transitions and >=3 positive weeks.
    s["positive_weeks"] = (s.positive_rate * s.training_weeks).round().astype(int)
    base = s.loc[(s.training_weeks >= 16) & (s.positive_weeks >= 3)].copy()

    # Representative targets are fixed by shape, not outcome:
    # active ~70% positive weeks, intermittent ~35%, sparse ~10%.
    specs = [
        ("活跃型", 1, 0.70, (0.55, 0.95)),
        ("间歇型", 1, 0.35, (0.20, 0.50)),
        ("稀疏型", 2, 0.10, (0.02, 0.20)),
    ]
    chosen = []
    used = set()
    for label, cluster, target, bounds in specs:
        cand = base.loc[(base.full_cluster == cluster)
                        & (base.positive_rate >= bounds[0])
                        & (base.positive_rate <= bounds[1])
                        & (~base.sku.isin(used))].copy()
        if cand.empty:
            cand = base.loc[(base.full_cluster == cluster) & (~base.sku.isin(used))].copy()
        cand["shape_distance"] = (cand.positive_rate - target).abs()
        # Prefer more historical support only as a deterministic tiebreaker.
        row = cand.sort_values(["shape_distance", "training_weeks", "sku"], ascending=[True, False, True]).iloc[0]
        used.add(row.sku)
        chosen.append({"type": label, **row.to_dict()})
    return pd.DataFrame(chosen)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(SRC)
    selected = pick_examples(df)
    rows = []
    for x in selected.itertuples(index=False):
        r = df.loc[(df.horizon_weeks == 9) & (df.tail_rank == 1) & (df.sku == x.sku)].iloc[0]
        y = np.asarray(json.loads(r.train_values), dtype=float)
        future9 = np.asarray(json.loads(r.test_values), dtype=float)
        future8 = future9[:H]
        s = markov_stats(y)
        actual = float(future8.sum())
        markov = s["markov_pxq_forecast_8"]
        ma4 = s["ma4_forecast_8"]
        rows.append({
            "type": x.type,
            "sku": x.sku,
            "origin": r.origin,
            "training_weeks": len(y),
            "positive_weeks": int((y > 0).sum()),
            "positive_rate": s["positive_rate"],
            "recent4_sales": json.dumps(y[-4:].tolist()),
            "current_state": s["current_state"],
            "p01": s["p01"],
            "p11": s["p11"],
            "expected_demand_weeks_8": s["expected_demand_weeks_8"],
            "weekly_probabilities_8": json.dumps([round(v, 4) for v in s["weekly_probabilities"]]),
            "q_positive_week_mean": s["q_positive_week_mean"],
            "markov_pxq_forecast_8": markov,
            "ma4_forecast_8": ma4,
            "actual_weekly_sales_8": json.dumps(future8.tolist()),
            "actual_sum_8": actual,
            "abs_error_markov": abs(markov - actual),
            "abs_error_ma4": abs(ma4 - actual),
            "ae_improvement_markov_vs_ma4": abs(ma4 - actual) - abs(markov - actual),
        })
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "three_examples.csv", index=False)

    lines = [
        "# Three representative SKU examples: 8-week Markov-PxQ vs MA4",
        "",
        "Selection is based only on pre-origin training shape; future outcomes are not used to choose SKUs.",
        "Q is fixed as the mean sales quantity among positive-demand training weeks. MA4 is recomputed as last-4-week mean × 8.",
        "The 8-week target is the first 8 observed weeks from an existing 9-week V4.4 test window.",
        "",
        out.to_markdown(index=False),
    ]
    (OUT / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print((OUT / "summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
