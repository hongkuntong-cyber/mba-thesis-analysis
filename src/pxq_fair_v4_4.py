"""Frozen supplementary experiment: SKU-tail origins, fixed cohorts, paired MA4."""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score

from .backtesting import _ending_contiguous_training_segment
from .cleaning_v2 import apply_v2_cleaning
from .clustering import fit_solution
from .data_audit import load_workbook_long, sha256_file
from .features import compute_features
from .pxq_two_part_v4_2 import backward_nonoverlapping_block_totals

ROOT = Path(__file__).resolve().parent.parent
CFG = yaml.safe_load((ROOT / 'config/pxq_fair_v4_4.yaml').read_text())
OUT = ROOT / CFG['output_root']
KEY = ['horizon_weeks', 'sku', 'origin']
COHORTS = {'active120': 1, 'sparse77': 2}


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding='utf-8')


def packet_data():
    with ZipFile(ROOT / CFG['prior_packet']) as z:
        manifest = json.loads(z.read('packet_manifest.json'))
        for name, digest in manifest.items():
            assert hashlib.sha256(z.read(name)).hexdigest() == digest, name
        tables = {name: pd.read_csv(io.BytesIO(z.read('outputs/pxq_simple_v4_3/' + name + '.csv')))
                  for name in ['components', 'quantity_predictions', 'probability_predictions', 'universe_coverage']}
        tables['legacy'] = pd.read_csv(io.BytesIO(z.read('outputs/pxq_validation_v4/rolling_origin_predictions.csv')))
    return tables, manifest


def scales(blocks):
    return ((float(np.abs(np.diff(blocks)).mean()), float(np.square(np.diff(blocks)).mean()))
            if len(blocks) >= 2 else (np.nan, np.nan))


def estimates(y, h):
    blocks = backward_nonoverlapping_block_totals(y, h, CFG['lookback_weeks'])
    full = backward_nonoverlapping_block_totals(y, h, len(y))
    n = len(blocks); positive = blocks[blocks > 0]; s = len(positive)
    p = s / n if n else np.nan
    q = float(positive.mean()) if s else np.nan
    laplace = (s + 1) / (n + 2) if n else np.nan
    scale, sqscale = scales(blocks)
    fullscale, _ = scales(full)
    positive_weeks = y[y > 0]
    legacy_p = float((y[-h:] > 0).mean()) if len(y) >= h else np.nan
    legacy_q = float(positive_weeks.mean()) if len(positive_weeks) else np.nan
    tail4 = y[-4:]; tpos = tail4[tail4 > 0]
    tail4_identity = float((tail4 > 0).mean() * tpos.mean() * h) if len(tpos) else 0.0
    pred = dict(BlockFrequencyMean=p*q, MA4_proxy=float(tail4.mean()*h),
                MatchedHistoryMean=float(blocks.mean()) if n else np.nan,
                LaplaceMean=laplace*q, LegacyPXQ=legacy_p*legacy_q*h,
                Naive=float(y[-1]*h), FullHistoryMean=float(full.mean()) if len(full) else np.nan)
    return dict(n=n, s=s, probability=p, laplace_probability=laplace, conditional_mean=q,
                scale_52=scale, square_scale_52=sqscale, scale_full=fullscale,
                full_n=len(full), legacy_week_probability=legacy_p, legacy_positive_week_mean=legacy_q,
                tail4_identity=tail4_identity, block_totals=json.dumps(blocks.tolist()),
                full_block_totals=json.dumps(full.tolist()), predictions=pred)


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    prior, packet_manifest = packet_data()
    loaded = load_workbook_long(ROOT / CFG['input_workbook'])
    assert not loaded.audit['blockers'] and loaded.audit['sha256'] == CFG['input_sha256']
    raw = loaded.weekly_complete
    cohort = pd.read_csv(ROOT / CFG['cohort_file'])
    assert not cohort.sku.duplicated().any()
    assert cohort.full_cluster.value_counts().to_dict() == {1: 120, 2: 77}
    rawmap = {s: g.set_index('week_start')['sales'] for s, g in raw.groupby('sku')}
    ids = cohort[['sku', 'full_cluster']].copy()
    bounds = raw.groupby('sku').week_start.agg(['min', 'max', 'size']).reset_index()
    ids = ids.merge(bounds, on='sku', validate='one_to_one')
    ids['calendar_group'] = np.select(
        [(ids['min'].dt.year < 2026) & (ids['max'].dt.year < 2026),
         (ids['min'].dt.year < 2026) & (ids['max'].dt.year == 2026), ids['min'].dt.year == 2026],
        ['only_2024_2025', 'across_2024_2026', 'only_2026'], default='other')
    ids.to_csv(OUT/'cohort_membership.csv', index=False)
    plan = []
    for r in ids.itertuples(index=False):
        for h in CFG['horizons_weeks']:
            for rank in range(6, 0, -1):
                origin = r.max - pd.Timedelta(weeks=rank*h-1)
                plan.append(dict(sku=r.sku, full_cluster=r.full_cluster, calendar_group=r.calendar_group,
                                 horizon_weeks=h, approximate_days=7*h, origin=origin,
                                 tail_rank=rank, origin_index=7-rank))
    plan = pd.DataFrame(plan)
    assert len(plan) == 197*3*6 and not plan.duplicated(KEY).any()
    oldc = prior['components'].copy(); oldc.origin = pd.to_datetime(oldc.origin)
    oldcm = oldc.set_index(KEY)
    oldq = prior['quantity_predictions'].copy(); oldq.origin = pd.to_datetime(oldq.origin)
    oldqm = oldq.set_index(KEY + ['method'])
    oldlegacy = prior['legacy'].copy(); oldlegacy.origin = pd.to_datetime(oldlegacy.origin)
    oldlm = oldlegacy.set_index(KEY + ['model'])
    # Old dynamic labels are taken from the verified packet on existing dates.
    old_labels = oldc[['origin','sku','cluster']].drop_duplicates()
    assert not old_labels.duplicated(['origin','sku']).any()
    label_map = old_labels.set_index(['origin','sku']).cluster.to_dict()
    clean_config = yaml.safe_load((ROOT/'config/pxq_two_part_v4_2.yaml').read_text())['cleaning_v2']
    records, forecasts, probabilities, coverage, train_audit = [], [], [], [], []
    reused_count = 0; new_count = 0
    old_names = {'BlockFrequencyMean':'SKU_recent_block_expected', 'MA4_proxy':'MA4_proxy',
                 'LaplaceMean':'Laplace_mean', 'LegacyPXQ':'PXQ', 'Naive':'Naive'}
    origin_groups = list(plan.groupby('origin', sort=True))
    for ix, (origin, requests) in enumerate(origin_groups, 1):
        clean = apply_v2_cleaning(raw.loc[raw.week_start < origin], **clean_config)
        trainmap = {s: g for s,g in clean.weekly.groupby('sku', sort=False)}
        segments = {s: _ending_contiguous_training_segment(g, origin, 'sales_v2') for s,g in trainmap.items()}
        eligible_requests = []
        for req in requests.to_dict('records'):
            sku=req['sku']; h=req['horizon_weeks']; dates=pd.date_range(origin, periods=h, freq='7D')
            target=rawmap[sku].reindex(dates)
            seg=segments.get(sku, pd.DataFrame(columns=['sales_v2']))
            reason='eligible' if len(seg)>=4 and target.notna().all() else 'training_missing_or_less_than_4_weeks' if len(seg)<4 else 'incomplete_future_observation'
            coverage.append(req | dict(training_weeks=len(seg), complete_future_weeks=int(target.notna().sum()), eligibility=reason))
            if reason=='eligible': eligible_requests.append((req,seg,target))
        needed = [req for req,_,_ in eligible_requests if (origin,req['sku']) not in label_map]
        newlabels={}
        if needed:
            features=compute_features(clean.weekly)
            finite=np.isfinite(features[CFG['features']]).all(axis=1)
            ef=features.loc[features.n_positive.ge(5) & finite]
            if len(ef)>2:
                labels=fit_solution(ef,CFG['features'],2)
                newlabels=labels.set_index('sku').cluster.to_dict()
        for h in sorted(requests.horizon_weeks.unique()):
            peer_counts={}
            for sku, seg in segments.items():
                totals=backward_nonoverlapping_block_totals(seg.sales_v2.to_numpy(float), h, 52)
                peer_counts[sku]=(len(totals), int((totals>0).sum()))
            pool_n=sum(v[0] for v in peer_counts.values()); pool_s=sum(v[1] for v in peer_counts.values())
            for req,seg,target in eligible_requests:
                if req['horizon_weeks']!=h: continue
                sku=req['sku']; y=seg.sales_v2.to_numpy(float); e=estimates(y,h); preds=e.pop('predictions')
                key=(h,sku,origin); actual=float(target.sum()); event=int(actual>0)
                old=oldcm.loc[key] if key in oldcm.index else None
                if old is not None:
                    assert np.isclose(old.actual_sum,actual)
                    for a,b in [('n','n'),('s','s'),('conditional_mean','q_mean'),('scale_52','scale_52'),('scale_full','scale_full')]:
                        assert np.isclose(e[a],old[b],equal_nan=True),(key,a,e[a],old[b])
                base=req | dict(actual_sum=actual,actual_event=event,training_weeks=len(y),
                                training_start=seg.week_start.min(),training_last_week=seg.week_start.max(),
                                dynamic_cluster=int(label_map.get((origin,sku),newlabels.get(sku,0))),
                                label_source='reused' if (origin,sku) in label_map else 'new_origin_training_only',
                                old_case_available=old is not None)
                en,es=peer_counts[sku]
                enterprise=(pool_s-es)/(pool_n-en) if pool_n>en else np.nan
                records.append(base | e | dict(enterprise_probability=enterprise,ma4_forecast=preds['MA4_proxy'],
                                               train_values=json.dumps(y.tolist()),test_values=json.dumps(target.tolist())))
                for method,value in preds.items():
                    source='new_formula'
                    oldkey=key+(old_names.get(method,''),)
                    if oldkey in oldqm.index and np.isfinite(value):
                        archived=float(oldqm.loc[oldkey,'forecast_sum'])
                        assert np.isclose(value,archived,rtol=1e-10,atol=1e-10),(oldkey,value,archived)
                        value=archived; source='verified_v4_3_reuse'; reused_count+=1
                    elif method in old_names and np.isfinite(value): new_count+=1
                    error=value-actual
                    forecast=base | dict(method=method,forecast_sum=value,error=error,ae=abs(error),
                        mase_52=abs(error)/e['scale_52'] if e['scale_52']>0 else np.nan,
                        mase_full=abs(error)/e['scale_full'] if e['scale_full']>0 else np.nan,
                        scaled_squared_error=error**2/e['square_scale_52'] if e['square_scale_52']>0 else np.nan,
                        scale_52=e['scale_52'],square_scale_52=e['square_scale_52'],scale_full=e['scale_full'],
                        source=source,n=e['n'],s=e['s'])
                    forecasts.append(forecast)
                for method,p in [('BlockFrequency',e['probability']),('Laplace',e['laplace_probability']),('EnterpriseFrequency',enterprise)]:
                    probabilities.append(base | dict(method=method,probability=p,brier=(p-event)**2,n=e['n'],s=e['s'],ma4_score=preds['MA4_proxy']))
        train_audit.append(dict(origin=origin,training_rows=len(clean.weekly),requests=len(requests),eligible=len(eligible_requests),
                                new_dynamic_labels=len(newlabels),v2_corrected_intervals=clean.summary['corrected_intervals']))
        print(f'Origin {ix}/{len(origin_groups)} {origin.date()}: {len(eligible_requests)}/{len(requests)} eligible',flush=True)
    tables=dict(components=pd.DataFrame(records),quantity_predictions=pd.DataFrame(forecasts),
                probability_predictions=pd.DataFrame(probabilities),universe_coverage=pd.DataFrame(coverage),
                origin_audit=pd.DataFrame(train_audit))
    for name,frame in tables.items(): frame.to_csv(OUT/(name+'.csv'),index=False)
    write_json(OUT/'input_audit.json',loaded.audit | dict(packet_files_verified=len(packet_manifest),
       prior_packet_sha256=sha256_file(ROOT/CFG['prior_packet']),cohort_sha256=sha256_file(ROOT/CFG['cohort_file']),
       verified_reused_forecast_values=reused_count,new_forecast_values=new_count,
       source_end_inclusive=str((raw.week_start.max()+pd.Timedelta(days=6)).date())))
    return tables,prior


def pair_rows(q, method, loss):
    cols=KEY+['full_cluster','calendar_group','dynamic_cluster','tail_rank','actual_sum',loss,'forecast_sum']
    a=q.loc[q.method.eq(method) & np.isfinite(q[loss]),cols]
    b=q.loc[q.method.eq('MA4_proxy') & np.isfinite(q[loss]),KEY+[loss,'forecast_sum']]
    return a.merge(b,on=KEY,suffixes=('_model','_ma4'),validate='one_to_one')


def boot_delta(pairs, loss):
    d=pairs[loss+'_model']-pairs[loss+'_ma4']
    ds=d.groupby(pairs.sku).mean()
    rng=np.random.default_rng(CFG['seed'])
    boot=rng.choice(ds.to_numpy(),(CFG['bootstrap_repetitions'],len(ds)),replace=True).mean(axis=1)
    return dict(delta=float(ds.mean()),ci_low=float(np.quantile(boot,.025)),ci_high=float(np.quantile(boot,.975)),
                pair_win_share=float((d < -1e-10).mean()),pair_tie_share=float((abs(d)<=1e-10).mean()))


def summarize_quantity(q):
    summaries=[]; pairs_out=[]; sku_out=[]
    for cname,cid in COHORTS.items():
        for h,g in q.loc[q.full_cluster.eq(cid)].groupby('horizon_weeks'):
            for method in CFG['quantity_methods']+['FullHistoryMean']:
                if method=='MA4_proxy': continue
                for loss in ['mase_52','mase_full','ae','scaled_squared_error']:
                    pp=pair_rows(g,method,loss)
                    if pp.empty: continue
                    stats=boot_delta(pp,loss)
                    row=dict(cohort=cname,horizon_weeks=h,method=method,metric=loss,n_pairs=len(pp),n_skus=pp.sku.nunique(),**stats)
                    for suffix in ['model','ma4']:
                        score=pp[loss+'_'+suffix]
                        row['mean_'+suffix]=float(score.groupby(pp.sku).mean().mean())
                        row['median_'+suffix]=float(score.median())
                        errors=pp['forecast_sum_'+suffix]-pp.actual_sum
                        row['bias_'+suffix]=float(errors.sum()/pp.actual_sum.sum()) if pp.actual_sum.sum()>0 else np.nan
                        row['under_units_'+suffix]=float(np.maximum(-errors,0).sum())
                        row['over_units_'+suffix]=float(np.maximum(errors,0).sum())
                        if loss=='mase_52': row['mase_lt1_'+suffix]=float((score<1).mean())
                        if loss=='scaled_squared_error': row['rmsse_'+suffix]=float(np.sqrt(row['mean_'+suffix]))
                    summaries.append(row)
                    if loss in ['mase_52','ae']:
                        pp['method']=method; pp['metric']=loss; pp['cohort']=cname
                        pp['delta']=pp[loss+'_model']-pp[loss+'_ma4']; pairs_out.append(pp)
                        ss=pp.groupby('sku').agg(n=('delta','size'),mean_delta=('delta','mean')).reset_index()
                        ss['method']=method;ss['metric']=loss;ss['cohort']=cname;ss['horizon_weeks']=h;sku_out.append(ss)
    return pd.DataFrame(summaries),pd.concat(pairs_out,ignore_index=True),pd.concat(sku_out,ignore_index=True)


def pmetrics(g):
    p=g.probability.to_numpy(); z=g.actual_event.to_numpy(); w=1/g.groupby('sku').sku.transform('size').to_numpy()
    bins=np.minimum((p*5).astype(int),4)
    ece=sum(abs(np.average(p[bins==b]-z[bins==b],weights=w[bins==b]))*w[bins==b].sum()/w.sum() for b in np.unique(bins))
    return dict(n=len(g),n_skus=g.sku.nunique(),mean_probability=float(np.average(p,weights=w)),
                actual_rate=float(np.average(z,weights=w)),gap=float(np.average(p-z,weights=w)),
                brier=float(np.average((p-z)**2,weights=w)),ece=float(ece),
                auc=float(roc_auc_score(z,p,sample_weight=w)) if len(np.unique(z))==2 else np.nan,
                ma4_score_auc=float(roc_auc_score(z,g.ma4_score,sample_weight=w)) if len(np.unique(z))==2 else np.nan)


def evaluate(tables,prior):
    q=tables['quantity_predictions'];p=tables['probability_predictions'];c=tables['components'];cov=tables['universe_coverage']
    summary,paired,sku=summarize_quantity(q)
    outputs=dict(quantity_comparisons=summary,paired_details=paired,sku_comparisons=sku)
    ps=[];bins=[];prob_pairs=[];sub=[]
    for cname,cid in COHORTS.items():
        for h,g in p.loc[p.full_cluster.eq(cid)].groupby('horizon_weeks'):
            commonkeys=g.loc[np.isfinite(g.probability)].groupby(KEY).method.nunique()
            valid=commonkeys[commonkeys.eq(3)].reset_index()[KEY]
            gg=g.merge(valid,on=KEY,validate='many_to_one')
            for method,mg in gg.groupby('method'):
                for scope, sg in [('all',mg),('ma4_zero',mg.loc[mg.ma4_score.eq(0)])]:
                    if not sg.empty: ps.append(dict(cohort=cname,horizon_weeks=h,method=method,scope=scope,**pmetrics(sg)))
                for dc,dg in mg.groupby('dynamic_cluster'):
                    sub.append(dict(cohort=cname,horizon_weeks=h,method=method,dynamic_cluster=dc,**pmetrics(dg)))
                mg=mg.copy(); mg['bin']=np.minimum((mg.probability*5).astype(int),4)
                for b,bg in mg.groupby('bin'):
                    bins.append(dict(cohort=cname,horizon_weeks=h,method=method,bin=b,n=len(bg),n_skus=bg.sku.nunique(),
                                     mean_probability=float(bg.probability.mean()),actual_rate=float(bg.actual_event.mean())))
            wide=gg.pivot(index=KEY,columns='method',values='brier')
            for base in ['Laplace','EnterpriseFrequency']:
                delta=(wide.BlockFrequency-wide[base]).groupby(level='sku').mean().to_numpy()
                rng=np.random.default_rng(42);boot=rng.choice(delta,(1000,len(delta)),replace=True).mean(axis=1)
                prob_pairs.append(dict(cohort=cname,horizon_weeks=h,method='BlockFrequency',baseline=base,
                                  n_pairs=len(wide),n_skus=len(delta),delta=delta.mean(),ci_low=np.quantile(boot,.025),ci_high=np.quantile(boot,.975)))
    outputs.update(probability_summary=pd.DataFrame(ps),reliability_bins=pd.DataFrame(bins),probability_by_dynamic_layer=pd.DataFrame(sub),paired_probability=pd.DataFrame(prob_pairs))
    coverage=[]
    for cname,cid in COHORTS.items():
        for h,g in cov.loc[cov.full_cluster.eq(cid)].groupby('horizon_weeks'):
            eligible=c.loc[c.full_cluster.eq(cid)&c.horizon_weeks.eq(h)]
            candidates=q.loc[q.full_cluster.eq(cid)&q.horizon_weeks.eq(h)&q.method.eq('BlockFrequencyMean')]
            validq=candidates[np.isfinite(candidates.forecast_sum)];validm=validq[np.isfinite(validq.mase_52)]
            coverage.append(dict(cohort=cname,horizon_weeks=h,planned_skus=g.sku.nunique(),planned_cases=len(g),
                         ma4_skus=eligible.sku.nunique(),ma4_cases=len(eligible),
                         probability_skus=eligible.loc[eligible.n>0].sku.nunique(),probability_cases=int((eligible.n>0).sum()),
                         pure_quantity_skus=validq.sku.nunique(),pure_quantity_cases=len(validq),mase_skus=validm.sku.nunique(),mase_cases=len(validm),
                         no_periods=int((eligible.n==0).sum()),no_positive_periods=int(((eligible.n>0)&(eligible.s==0)).sum()),
                         scale_insufficient=int((eligible.n<2).sum()),scale_zero=int((eligible.scale_52==0).sum())))
    outputs['coverage_summary']=pd.DataFrame(coverage)
    # Every planned SKU remains in a simple support table, including zero usable cases.
    sku_cov=cov.groupby(['full_cluster','sku','horizon_weeks']).agg(planned=('origin','size'),ma4_available=('eligibility',lambda s:int(s.eq('eligible').sum()))).reset_index()
    for m,col in [('forecast_sum','quantity_available'),('mase_52','mase_available')]:
        x=q.loc[q.method.eq('BlockFrequencyMean')].groupby(['sku','horizon_weeks'])[m].apply(lambda s:int(np.isfinite(s).sum())).reset_index(name=col)
        sku_cov=sku_cov.merge(x,on=['sku','horizon_weeks'],how='left',validate='one_to_one')
        sku_cov[col]=sku_cov[col].fillna(0).astype(int)
    outputs['sku_coverage']=sku_cov
    # Preserve old calendar-origin comparison as a distinct sensitivity, not merged with tail origins.
    old=prior['quantity_predictions'].copy();ids=pd.read_csv(OUT/'cohort_membership.csv')
    old=old.merge(ids[['sku','full_cluster','calendar_group']],on='sku',validate='many_to_one')
    old['dynamic_cluster']=0;old['tail_rank']=0;old['scaled_squared_error']=np.nan
    old.method=old.method.replace({'SKU_recent_block_expected':'BlockFrequencyMean','Laplace_mean':'LaplaceMean','PXQ':'LegacyPXQ'})
    old_summary,_,_=summarize_quantity(old)
    outputs['old_calendar_quantity_comparisons']=old_summary
    # Prespecified calendar and dynamic layer descriptions, with paired AE/MASE and no new model selection.
    rows=[]
    for dimension in ['calendar_group','dynamic_cluster','tail_rank']:
        for (cid,h,v),gg in q.groupby(['full_cluster','horizon_weeks',dimension]):
            for loss in ['ae','mase_52']:
                pair=pair_rows(gg,'BlockFrequencyMean',loss)
                if pair.empty: continue
                rows.append(dict(full_cluster=cid,horizon_weeks=h,dimension=dimension,layer=str(v),metric=loss,
                                 n=len(pair),n_skus=pair.sku.nunique(),mean_model=pair.groupby('sku')[loss+'_model'].mean().mean(),
                                 mean_ma4=pair.groupby('sku')[loss+'_ma4'].mean().mean(),**boot_delta(pair,loss)))
    outputs['quantity_by_layer']=pd.DataFrame(rows)
    for name,df in outputs.items(): df.to_csv(OUT/(name+'.csv'),index=False)
    return outputs


def main():
    if (OUT/'components.csv').exists():
        print('Reusing completed preparation; only recomputing summaries.',flush=True)
        tables={name:pd.read_csv(OUT/(name+'.csv')) for name in ['components','quantity_predictions','probability_predictions','universe_coverage','origin_audit']}
        prior,_=packet_data()
    else: tables,prior=prepare()
    outputs=evaluate(tables,prior)
    fingerprint={str(p.relative_to(ROOT)):sha256_file(p) for p in sorted(OUT.glob('*.csv'))}
    write_json(OUT/'manifest_sha256.json',fingerprint)
    cols=['horizon_weeks','method','n_pairs','n_skus','mean_model','mean_ma4','delta','ci_low','ci_high']
    main=outputs['quantity_comparisons']; main=main.loc[main.cohort.eq('active120') & main.metric.eq('mase_52')]
    print(main[cols].to_string(index=False))
    print(outputs['coverage_summary'].to_string(index=False))


if __name__=='__main__': main()
