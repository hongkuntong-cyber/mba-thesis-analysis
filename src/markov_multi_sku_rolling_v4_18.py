"""Exploratory multi-SKU rolling 8-week Markov-PxQ vs MA4 sample.

Six SKUs are selected deterministically from the fixed V4.4 evaluation universe using
ONLY pre-origin V2 training demand-frequency shape. No future outcomes or errors are
used in selection. Each SKU is evaluated at 2026-06-01 and, where available,
2026-06-08 after one new observed week.
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
COMP = ROOT / 'outputs/pxq_fair_v4_4/components.csv'
OUT = ROOT / 'outputs/markov_v4_18_multi_sku_rolling'
START = pd.Timestamp('2026-06-01')
H = 8
TARGET_RATES = [0.75, 0.55, 0.35, 0.20, 0.10, 0.05]
LABELS = ['高活跃','中高活跃','间歇','低频','稀疏','极稀疏']


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
    state = np.array([1.0,0.0]) if z[-1] == 0 else np.array([0.0,1.0])
    probs=[]
    for _ in range(H):
        state = state @ P
        probs.append(float(state[1]))
    q = float(y[y>0].mean()) if np.any(y>0) else 0.0
    return {
        'current_state': int(z[-1]), 'positive_rate': float(z.mean()),
        'positive_weeks': int(z.sum()), 'p01': float(p01), 'p11': float(p11),
        'state_dependence': float(p11-p01), 'q': q,
        'expected_active_weeks': float(sum(probs)), 'weekly_probs': probs,
        'markov_forecast': float(sum(probs)*q),
        'ma4_forecast': float(y[-4:].mean()*H),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    loaded = load_workbook_long(ROOT / CFG['input_workbook'])
    assert not loaded.audit['blockers']
    raw = loaded.weekly_complete.copy()

    # Candidate universe from existing V4.4 evaluation output; do not use outcome fields.
    comp = pd.read_csv(COMP)
    candidates = sorted(comp.loc[(comp.horizon_weeks == 9) & (comp.tail_rank == 1), 'sku'].astype(str).unique())

    # Build all training summaries at the first origin.
    clean0 = apply_v2_cleaning(raw.loc[raw.week_start < START], **CLEAN_CFG)
    stats=[]
    for sku in candidates:
        g = clean0.weekly.loc[clean0.weekly.sku.eq(sku)]
        seg = _ending_contiguous_training_segment(g, START, 'sales_v2')
        if len(seg) < 16:
            continue
        y = seg.sales_v2.to_numpy(float)
        s = calc_markov(y)
        if s['positive_weeks'] < 3:
            continue
        stats.append({'sku':sku,'training_weeks':len(y),**s})
    stats = pd.DataFrame(stats)

    # Select nearest to fixed target demand-frequency rates. No future outcomes used.
    chosen=[]; used=set()
    for label,target in zip(LABELS,TARGET_RATES):
        cand=stats.loc[~stats.sku.isin(used)].copy()
        cand['distance']=(cand.positive_rate-target).abs()
        row=cand.sort_values(['distance','training_weeks','sku'],ascending=[True,False,True]).iloc[0]
        used.add(row.sku)
        chosen.append((label,row.sku,float(row.positive_rate)))

    raw_index = raw.pivot(index='week_start', columns='sku', values='sales')
    rows=[]
    for label,sku,base_rate in chosen:
        for k in [0,1]:
            origin=START+pd.Timedelta(weeks=k)
            clean = apply_v2_cleaning(raw.loc[raw.week_start < origin], **CLEAN_CFG)
            g=clean.weekly.loc[clean.weekly.sku.eq(sku)]
            seg=_ending_contiguous_training_segment(g,origin,'sales_v2')
            if len(seg)<4:
                continue
            y=seg.sales_v2.to_numpy(float)
            target_dates=pd.date_range(origin,periods=H,freq='7D')
            if sku not in raw_index.columns:
                continue
            target=raw_index[sku].reindex(target_dates)
            if not target.notna().all():
                continue
            s=calc_markov(y); actual=float(target.sum())
            rows.append({
                'type':label,'sku':sku,'origin':str(origin.date()),
                'newly_observed_previous_week':None if k==0 else float(raw_index[sku].get(origin-pd.Timedelta(weeks=1),np.nan)),
                'training_weeks':len(y),'positive_rate':s['positive_rate'],'positive_weeks':s['positive_weeks'],
                'recent4':json.dumps(y[-4:].tolist()),'current_state':s['current_state'],
                'p01':s['p01'],'p11':s['p11'],'state_dependence':s['state_dependence'],
                'q_positive_mean':s['q'],'expected_active_weeks_8':s['expected_active_weeks'],
                'weekly_probs_8':json.dumps([round(x,4) for x in s['weekly_probs']]),
                'markov_forecast_8':s['markov_forecast'],'ma4_forecast_8':s['ma4_forecast'],
                'actual_next8':json.dumps(target.tolist()),'actual_sum_8':actual,
                'abs_error_markov':abs(s['markov_forecast']-actual),
                'abs_error_ma4':abs(s['ma4_forecast']-actual),
            })
    out=pd.DataFrame(rows)
    out['winner']=np.where(out.abs_error_markov<out.abs_error_ma4,'Markov',np.where(out.abs_error_ma4<out.abs_error_markov,'MA4','Tie'))
    out['ae_improvement_markov_vs_ma4']=out.abs_error_ma4-out.abs_error_markov
    out.to_csv(OUT/'multi_sku_rolling.csv',index=False)

    cols=['type','sku','origin','newly_observed_previous_week','recent4','current_state','p01','p11','q_positive_mean','expected_active_weeks_8','markov_forecast_8','ma4_forecast_8','actual_sum_8','abs_error_markov','abs_error_ma4','winner','ae_improvement_markov_vs_ma4']
    lines=['# Multi-SKU rolling weekly 8-week sample','',
           'Selection: six fixed target pre-origin positive-demand rates (75%,55%,35%,20%,10%,5%); nearest eligible SKU chosen without using future outcomes.','',
           out[cols].to_markdown(index=False),'', '## Aggregate', '']
    agg=out.groupby('origin').agg(cases=('sku','count'),markov_mae=('abs_error_markov','mean'),ma4_mae=('abs_error_ma4','mean'),markov_wins=('winner',lambda x:int((x=='Markov').sum())),ma4_wins=('winner',lambda x:int((x=='MA4').sum()))).reset_index()
    lines.append(agg.to_markdown(index=False))
    (OUT/'summary.md').write_text('\n'.join(lines),encoding='utf-8')
    print((OUT/'summary.md').read_text(encoding='utf-8'))

if __name__=='__main__':
    main()
