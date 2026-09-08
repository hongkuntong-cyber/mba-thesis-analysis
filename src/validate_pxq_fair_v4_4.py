"""Independent arithmetic, pairing, coverage and raw-target validation for V4.4."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .data_audit import sha256_file

ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'outputs/pxq_fair_v4_4'


def validate():
    c=pd.read_csv(OUT/'components.csv'); q=pd.read_csv(OUT/'quantity_predictions.csv')
    p=pd.read_csv(OUT/'probability_predictions.csv'); cov=pd.read_csv(OUT/'universe_coverage.csv')
    ids=pd.read_csv(OUT/'cohort_membership.csv'); summary=pd.read_csv(OUT/'quantity_comparisons.csv')
    key=['horizon_weeks','sku','origin']; checks={}
    def check(name,condition):
        checks[name]=bool(condition)
        assert condition,name
    check('cohort_120_and_77',ids.full_cluster.value_counts().to_dict()=={1:120,2:77})
    check('all_planned_cases_preserved',len(cov)==197*18 and not cov.duplicated(key).any())
    check('all_skus_have_18_plan_rows',cov.groupby('sku').size().eq(18).all())
    check('unique_prediction_keys',not q.duplicated(key+['method']).any() and not p.duplicated(key+['method']).any())
    check('eligible_set_exact',set(map(tuple,c[key].to_numpy()))==set(map(tuple,cov.loc[cov.eligibility.eq('eligible'),key].to_numpy())))
    check('training_before_origin',(pd.to_datetime(c.training_last_week)==pd.to_datetime(c.origin)-pd.Timedelta(weeks=1)).all())
    check('training_contiguous_length',((pd.to_datetime(c.training_last_week)-pd.to_datetime(c.training_start)).dt.days//7+1==c.training_weeks).all())
    # Reconstruct raw complete ISO-week sums independently of load_workbook_long.
    xlsx=ROOT/'01_原始数据/德国Amazon_SKU周度数据_原始合并版_未清洗 (1).xlsx'
    check('source_fingerprint',sha256_file(xlsx)=='ece008d42c9dd6ea11e4a0c8f6d828c2cb037df1d837ea233587f115491786b6')
    raw={};days={}
    for name,frame in pd.read_excel(xlsx,sheet_name=None).items():
        for col in frame.columns[8:]:
            a,b=[pd.Timestamp(x.strip()) for x in str(col).split('~')]
            week=a-pd.Timedelta(days=a.weekday()); covered=set(pd.date_range(a,b))
            for sku,val in zip(frame.SKU.astype(str).str.strip(),frame[col]):
                k=(sku,week);raw[k]=raw.get(k,0.0)+float(val);days.setdefault(k,set()).update(covered)
    raw={k:v for k,v in raw.items() if len(days[k])==7}
    qm=q.set_index(key+['method']); pm=p.set_index(key+['method'])
    for r in c.itertuples(index=False):
        y=np.asarray(json.loads(r.train_values));actual=np.asarray(json.loads(r.test_values));h=r.horizon_weeks
        period=min(52,len(y))//h
        totals=np.array([sum(y[len(y)-(j+1)*h:len(y)-j*h]) for j in range(period-1,-1,-1)])
        assert np.allclose(totals,np.asarray(json.loads(r.block_totals)))
        s=int((totals>0).sum()); assert r.n==period and r.s==s
        meanpos=float(totals[totals>0].mean()) if s else np.nan
        prob=s/period if period else np.nan
        lp=(s+1)/(period+2) if period else np.nan
        scale=float(np.abs(np.diff(totals)).mean()) if period>1 else np.nan
        squared=float(np.square(np.diff(totals)).mean()) if period>1 else np.nan
        k=(h,r.sku,r.origin)
        vals={'BlockFrequencyMean':prob*meanpos,'MatchedHistoryMean':float(totals.mean()) if period else np.nan,
              'LaplaceMean':lp*meanpos,'MA4_proxy':sum(y[-4:])/4*h,'Naive':y[-1]*h}
        for method,value in vals.items():
            rr=qm.loc[k+(method,)];assert np.isclose(rr.forecast_sum,value,equal_nan=True)
            err=abs(value-sum(actual));assert np.isclose(rr.ae,err,equal_nan=True)
            assert np.isclose(rr.mase_52,err/scale if scale>0 else np.nan,equal_nan=True)
            assert np.isclose(rr.scaled_squared_error,err**2/squared if squared>0 else np.nan,equal_nan=True)
        assert np.isclose(pm.loc[k+('BlockFrequency',),'probability'],prob,equal_nan=True)
        assert np.isclose(pm.loc[k+('Laplace',),'probability'],lp,equal_nan=True)
        assert np.isclose(r.tail4_identity,vals['MA4_proxy'])
        if s: assert np.isclose(vals['BlockFrequencyMean'],vals['MatchedHistoryMean'])
        dates=pd.date_range(r.origin,periods=h,freq='7D')
        assert len(actual)==h and np.allclose([raw[(r.sku,d)] for d in dates],actual)
        assert np.isclose(sum(actual),r.actual_sum) and r.actual_event==int(sum(actual)>0)
    checks['all_forecast_formula_and_period_scales']=True
    checks['all_raw_future_targets_independently_reconstructed']=True
    checks['probability_and_matched_mean_identities']=True
    for r in summary.itertuples(index=False):
        cid=1 if r.cohort=='active120' else 2
        sub=q.loc[q.full_cluster.eq(cid)&q.horizon_weeks.eq(r.horizon_weeks)]
        model=sub.loc[sub.method.eq(r.method),key+[r.metric]].set_index(key)
        base=sub.loc[sub.method.eq('MA4_proxy'),key+[r.metric]].set_index(key)
        pair=model.join(base,lsuffix='_m',rsuffix='_b',how='inner').dropna()
        assert len(pair)==r.n_pairs and pair.index.get_level_values('sku').nunique()==r.n_skus
        m=pair[r.metric+'_m'].groupby(level='sku').mean();b=pair[r.metric+'_b'].groupby(level='sku').mean()
        assert np.isclose(m.mean(),r.mean_model) and np.isclose(b.mean(),r.mean_ma4)
        delta=m-b;assert np.isclose(delta.mean(),r.delta)
        # Independent bootstrap algorithm, same deterministic draw order.
        rng=np.random.default_rng(42)
        sampled=[delta.to_numpy()[rng.integers(0,len(delta),len(delta))].mean() for _ in range(1000)]
        assert np.allclose(np.quantile(sampled,[.025,.975]),[r.ci_low,r.ci_high])
    checks['all_pair_intersections_and_macro_denominators']=True
    checks['all_bootstrap_intervals_independently_recomputed']=True
    check('ma4_not_probability_method',not p.method.str.contains('MA4',case=False).any())
    check('no_epsilon_for_zero_scale',q.loc[q.scale_52.eq(0),'mase_52'].isna().all())
    check('missing_conditional_quantity_preserved',q.loc[q.method.eq('BlockFrequencyMean')&q.s.eq(0),'forecast_sum'].isna().all())
    check('all_output_hashes',all(sha256_file(ROOT/f)==digest for f,digest in json.loads((OUT/'manifest_sha256.json').read_text()).items()))
    result=dict(passed=len(checks),failed=0,checks=checks,validated_cases=len(c),validated_forecast_rows=len(q),validated_summary_rows=len(summary))
    (OUT/'validation_summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))
    return result


if __name__=='__main__': validate()
