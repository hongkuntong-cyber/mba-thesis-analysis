"""Exploratory one-SKU rolling 8-week Markov-PxQ vs MA4 demonstration.

Re-estimates V2 training data, Markov transitions and Q at every weekly origin.
No future data are used in fitting. Future 8-week raw sales are used only for evaluation.
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
CFG = yaml.safe_load((ROOT / 'config/pxq_fair_v4_4.yaml').read_text())
CLEAN_CFG = yaml.safe_load((ROOT / 'config/pxq_two_part_v4_2.yaml').read_text())['cleaning_v2']
OUT = ROOT / 'outputs/markov_v4_17_rolling_one_sku'
SKU = 'JZJ0019'
START = pd.Timestamp('2026-06-01')
H = 8
MAX_ORIGINS = 8


def calc_markov(y: np.ndarray) -> dict:
    z = (y > 0).astype(int)
    a, b = z[:-1], z[1:]
    n00 = int(((a == 0) & (b == 0)).sum())
    n01 = int(((a == 0) & (b == 1)).sum())
    n10 = int(((a == 1) & (b == 0)).sum())
    n11 = int(((a == 1) & (b == 1)).sum())
    p01 = (n01 + 1.0) / (n00 + n01 + 2.0)
    p11 = (n11 + 1.0) / (n10 + n11 + 2.0)
    P = np.array([[1-p01, p01], [1-p11, p11]], dtype=float)
    state = np.array([1.0, 0.0]) if z[-1] == 0 else np.array([0.0, 1.0])
    probs = []
    for _ in range(H):
        state = state @ P
        probs.append(float(state[1]))
    q = float(y[y > 0].mean()) if np.any(y > 0) else 0.0
    return dict(
        current_state=int(z[-1]), n00=n00, n01=n01, n10=n10, n11=n11,
        p01=float(p01), p11=float(p11), q=q,
        expected_active_weeks=float(sum(probs)),
        weekly_probs=probs,
        markov_forecast=float(sum(probs)*q),
        ma4_forecast=float(y[-4:].mean()*H),
    )


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    loaded = load_workbook_long(ROOT / CFG['input_workbook'])
    assert not loaded.audit['blockers']
    raw = loaded.weekly_complete.copy()
    raw_sku = raw.loc[raw.sku.eq(SKU)].set_index('week_start')['sales'].sort_index()

    rows = []
    for k in range(MAX_ORIGINS):
        origin = START + pd.Timedelta(weeks=k)
        clean = apply_v2_cleaning(raw.loc[raw.week_start < origin], **CLEAN_CFG)
        g = clean.weekly.loc[clean.weekly.sku.eq(SKU)]
        seg = _ending_contiguous_training_segment(g, origin, 'sales_v2')
        if len(seg) < 4:
            continue
        y = seg.sales_v2.to_numpy(float)
        target_dates = pd.date_range(origin, periods=H, freq='7D')
        target = raw_sku.reindex(target_dates)
        if not target.notna().all():
            break
        s = calc_markov(y)
        actual = float(target.sum())
        rows.append(dict(
            origin=str(origin.date()),
            newly_observed_previous_week=(None if k == 0 else float(raw_sku.get(origin-pd.Timedelta(weeks=1), np.nan))),
            training_weeks=len(y),
            recent4=json.dumps(y[-4:].tolist()),
            current_state=s['current_state'],
            p01=s['p01'], p11=s['p11'],
            q_positive_mean=s['q'],
            expected_active_weeks_8=s['expected_active_weeks'],
            weekly_probs_8=json.dumps([round(x,4) for x in s['weekly_probs']]),
            markov_forecast_8=s['markov_forecast'],
            ma4_forecast_8=s['ma4_forecast'],
            actual_next8=json.dumps(target.tolist()),
            actual_sum_8=actual,
            abs_error_markov=abs(s['markov_forecast']-actual),
            abs_error_ma4=abs(s['ma4_forecast']-actual),
        ))

    out = pd.DataFrame(rows)
    out['winner'] = np.where(out.abs_error_markov < out.abs_error_ma4, 'Markov',
                     np.where(out.abs_error_ma4 < out.abs_error_markov, 'MA4', 'Tie'))
    out.to_csv(OUT/'rolling_JZJ0019.csv', index=False)

    cols = ['origin','newly_observed_previous_week','recent4','current_state','p01','p11',
            'expected_active_weeks_8','q_positive_mean','markov_forecast_8','ma4_forecast_8',
            'actual_sum_8','abs_error_markov','abs_error_ma4','winner']
    lines = [
        '# JZJ0019 rolling weekly 8-week forecast', '',
        'Each row re-estimates V2 history, Markov transition probabilities and Q using only data available before that origin.',
        'The forecast horizon stays 8 weeks while the origin advances by 1 week.', '',
        out[cols].to_markdown(index=False), '',
        '## Weekly probability paths', ''
    ]
    for r in out.itertuples(index=False):
        lines.append(f"- {r.origin}: recent4={r.recent4}; p1..p8={r.weekly_probs_8}")
    (OUT/'summary.md').write_text('\n'.join(lines), encoding='utf-8')
    print((OUT/'summary.md').read_text(encoding='utf-8'))


if __name__ == '__main__':
    main()
