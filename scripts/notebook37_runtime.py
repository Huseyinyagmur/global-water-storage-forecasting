"""Validation/leaderboard postmortem. Analysis only: no model fit, download or submission write."""
from __future__ import annotations
import hashlib, json, sys
from pathlib import Path
_ROOT_HINT=Path(__file__).resolve().parents[1] if "__file__" in globals() else Path.cwd().resolve()
_VENV_HINT=_ROOT_HINT/'.venv/Lib/site-packages'
if _VENV_HINT.exists() and str(_VENV_HINT) not in sys.path:sys.path.append(str(_VENV_HINT))
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

ROOT=next(p for p in [Path.cwd().resolve(),*Path.cwd().resolve().parents,Path(__file__).resolve().parents[1]] if (p/'data/raw/Train.csv').exists())
OUT=ROOT/'artifacts/notebook37';OUT.mkdir(parents=True,exist_ok=True)
CAN_SCORE=.824463818; BEST_SCORE=.823609466
SCORES={'submission_optimized.csv':CAN_SCORE,'submission_14.csv':.825971129,'submission_15_primary.csv':.824797753,'submission_18_A_domain_best.csv':.831108253,'submission_21_spatial_residual_candidate.csv':BEST_SCORE,'submission_36_f1_l12_primary.csv':.824335279}
ORIGIN={'submission_optimized.csv':('11','canonical selected masked blend'),'submission_14.csv':('14','alternative-model blend'),'submission_15_primary.csv':('15','delta dual expert'),'submission_18_A_domain_best.csv':('18','drying-transition domain feature'),'submission_21_spatial_residual_candidate.csv':('21','capped southern masked correction'),'submission_36_f1_l12_primary.csv':('36','frozen F1-L12 raw-lag dual expert')}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def tv(a,b,bins):
 x=np.histogram(a,bins=bins)[0].astype(float);y=np.histogram(b,bins=bins)[0].astype(float);x/=max(x.sum(),1);y/=max(y.sum(),1);return .5*np.abs(x-y).sum()
def pairacc(x,y):
 ok=tot=0
 for i in range(len(x)):
  for j in range(i+1,len(x)):
   if x[i]!=x[j] and y[i]!=y[j]:tot+=1;ok+=np.sign(x[i]-x[j])==np.sign(y[i]-y[j])
 return ok/tot if tot else np.nan

train=pd.read_csv(ROOT/'data/raw/Train.csv',parse_dates=['time']);test=pd.read_csv(ROOT/'data/raw/Test.csv',parse_dates=['time']);sample=pd.read_csv(ROOT/'data/raw/SampleSubmission.csv')
canonical=pd.read_csv(ROOT/'submissions/submission_optimized.csv');current=pd.read_csv(ROOT/'submissions/submission_21_spatial_residual_candidate.csv');f1=pd.read_csv(ROOT/'submissions/submission_36_f1_l12_primary.csv')
assert canonical.ID.equals(test.ID) and current.ID.equals(test.ID) and f1.ID.equals(test.ID)
meta=test[['ID','time','lat','lon','TWS_t','SPEI_12_t']].copy();meta['known']=meta.TWS_t.notna();meta['hemisphere']=np.where(meta.lat>=0,'North','South');meta['latitude_band']=(np.floor(meta.lat/10)*10).astype(int);meta['month']=meta.time.dt.month

# Actual prediction-time TWS history completeness across concatenated train/test panels.
trh=train[['time','lat','lon','TWS_t']].assign(is_test=False,test_order=-1);teh=test[['time','lat','lon','TWS_t']].assign(is_test=True,test_order=np.arange(len(test)));hist=pd.concat([trh,teh],ignore_index=True);hist['loc']=hist.lat.astype(str)+'_'+hist.lon.astype(str);hist=hist.sort_values(['loc','time']);g=hist.groupby('loc',sort=False)
lagmiss=np.empty((len(test),12),dtype=bool)
for k in range(1,13):lagmiss[:,k-1]=g.TWS_t.shift(k)[hist.is_test].sort_index().isna().to_numpy()
meta['lag_valid_count']=12-lagmiss.sum(1);meta['lag_missing_fraction']=lagmiss.mean(1);meta['cold_start']=meta.lag_valid_count.eq(0)
last=np.full(len(test),13,dtype=int)
for i in range(12):last[(last==13)&(~lagmiss[:,i])]=i+1
meta['months_since_last_valid']=last

# 1. All repository submissions; public blanks remain blank.
inv=[]
for p in sorted((ROOT/'submissions').glob('*.csv')):
 try:d=pd.read_csv(p)
 except Exception:continue
 if not {'ID','Target'}.issubset(d.columns) or len(d)!=len(test):continue
 aligned=d.ID.equals(test.ID);v=d.Target.to_numpy();name=p.name;orig,method=ORIGIN.get(name,('not verified','historical/experimental; see origin notebook'))
 inv.append({'filename':name,'origin_notebook':orig,'method':method,'SHA256':sha(p),'row_count':len(d),'known_count':int(meta.known.sum()),'masked_count':int((~meta.known).sum()),'public_score':SCORES.get(name,np.nan),'canonical_correlation':np.corrcoef(v,canonical.Target)[0,1],'current_best_correlation':np.corrcoef(v,current.Target)[0,1],'canonical_MAD':np.mean(np.abs(v-canonical.Target)),'current_best_MAD':np.mean(np.abs(v-current.Target)),'prediction_mean':v.mean(),'prediction_std':v.std(),'prediction_min':v.min(),'prediction_max':v.max(),'changed_row_fraction':np.mean(v!=canonical.Target.to_numpy()),'ID_aligned':aligned,'status':'public-positive vs canonical' if name in SCORES and SCORES[name]<CAN_SCORE else ('public-negative vs canonical' if name in SCORES and SCORES[name]>CAN_SCORE else ('canonical' if name=='submission_optimized.csv' else 'public unknown'))})
inventory=pd.DataFrame(inv);inventory.to_csv(OUT/'submission_inventory.csv',index=False)

# 2. Evidence inventory. Values are copied only from named executed artifacts.
n36=pd.read_csv(ROOT/'artifacts/notebook36/validation_summary.csv').set_index('Validation');n35=pd.read_csv(ROOT/'artifacts/notebook35/pilot_results.csv');n33=pd.read_csv(ROOT/'artifacts/notebook33/validation_results.csv');n31=pd.read_csv(ROOT/'artifacts/notebook31/validation_results.csv');n30=pd.read_csv(ROOT/'artifacts/notebook30/validation_results.csv');n21=pd.read_csv(ROOT/'artifacts/notebook21/correction_ablation.csv').iloc[0]
rows=[
 ['Canonical','11','Notebook11 temporal',.711647,np.nan,.758030,.769866,.060838,np.nan,np.nan,CAN_SCORE],
 ['Alternative ensemble','14','Notebook11 recorded representative',.688811,np.nan,np.nan,np.nan,np.nan,.823676,.804696,.825971129],
 ['Hydrology feature','18','Notebook11 recorded representative',.709586,np.nan,np.nan,np.nan,np.nan,.853484,.816142,.831108253],
 ['Spatial correction','21','8-fold correction validation',n21['Overall RMSE'],np.nan,np.nan,n21['Worst fold'],n21['Fold std'],n21['Spatial block mean RMSE'],np.nan,BEST_SCORE],
 ['Graph E5 cluster','30','four representative robust folds',float(n30.query("Feature_set=='E5_Cluster' and Validation=='strict_temporal'").Overall_RMSE.iloc[0]),float(n30.query("Feature_set=='E5_Cluster' and Validation=='strict_temporal'").Known_RMSE.iloc[0]),float(n30.query("Feature_set=='E5_Cluster' and Validation=='strict_temporal'").Masked_RMSE.iloc[0]),float(n30.query("Feature_set=='E5_Cluster' and Validation=='strict_temporal'").Worst_fold.iloc[0]),float(n30.query("Feature_set=='E5_Cluster' and Validation=='strict_temporal'").Fold_std.iloc[0]),float(n30.query("Feature_set=='E5_Cluster' and Validation=='spatial_block'").Overall_RMSE.iloc[0]),float(n30.query("Feature_set=='E5_Cluster' and Validation=='space_time'").Overall_RMSE.iloc[0]),np.nan],
 ['ERA5 E1','31','three-region extended pilot',float(n31.query("system=='E1'").Overall_RMSE.iloc[0]),float(n31.query("system=='E1'").Known_RMSE.iloc[0]),float(n31.query("system=='E1'").Masked_RMSE.iloc[0]),float(n31.query("system=='E1'").Worst_fold.iloc[0]),float(n31.query("system=='E1'").Fold_std.iloc[0]),np.nan,np.nan,np.nan],
 ['GRU S1','33','latitude-stratified pilot strict_temporal_2',float(n33.query("system=='S1'").overall_RMSE.iloc[0]),float(n33.query("system=='S1'").known_RMSE.iloc[0]),float(n33.query("system=='S1'").masked_RMSE.iloc[0]),float(n33.query("system=='S1'").worst_fold.iloc[0]),np.nan,np.nan,np.nan,np.nan],
 ['F1_L12 pilot','35','same Notebook33 pilot',float(n35.query("System=='F1_L12'").RMSE.iloc[0]),float(n35.query("System=='F1_L12'").Known_RMSE.iloc[0]),float(n35.query("System=='F1_L12'").Masked_RMSE.iloc[0]),float(n35.query("System=='F1_L12'").Worst_group_RMSE.iloc[0]),np.nan,np.nan,np.nan,.824335279],
 ['F1_L12 full','36','frozen full; mask propagation mismatch identified post hoc',n36.loc['strict_temporal','F1_overall_RMSE'],n36.loc['strict_temporal','F1_known_RMSE'],n36.loc['strict_temporal','F1_masked_RMSE'],n36.loc['strict_temporal','Worst_fold'],n36.loc['strict_temporal','Fold_std'],n36.loc['spatial_block','F1_overall_RMSE'],n36.loc['space_time','F1_overall_RMSE'],.824335279],
]
local=pd.DataFrame(rows,columns=['model','notebook','validation_design','overall_RMSE','known_RMSE','masked_RMSE','worst_fold','fold_std','spatial_RMSE','space_time_RMSE','public_score']);local['public_delta_vs_canonical']=local.public_score-CAN_SCORE;local.to_csv(OUT/'local_public_metric_inventory.csv',index=False)

# 3. Only common-model Notebook20 comparisons support a ranking calculation (n=4).
rank=pd.read_csv(ROOT/'artifacts/notebook20/model_ranking_consistency.csv');corr=[]
for design,g in rank[(rank['Comparable rows']>0)&(rank.Validation!='Public leaderboard')].groupby('Validation'):
 pub={'Canonical Notebook 11':CAN_SCORE,'Notebook 14 alternative':.825971129,'Notebook 15 delta':.824797753,'Notebook 18 drying_transition':.831108253};g=g[g.Model.isin(pub)].copy();x=g.RMSE.to_numpy();y=np.array([pub[z] for z in g.Model])
 corr.append({'metric':design,'Pearson':pearsonr(x,y).statistic,'Spearman':spearmanr(x,y).statistic,'pairwise_ranking_accuracy':pairacc(x,y),'sign_agreement_vs_canonical':np.mean(np.sign(x-x[g.Model.eq('Canonical Notebook 11')][0])==np.sign(y-y[g.Model.eq('Canonical Notebook 11')][0])),'sample_count':len(g),'evidence':'insufficient evidence: n=4; different historical OOF rows'})
corr.extend([{'metric':m,'Pearson':np.nan,'Spearman':np.nan,'pairwise_ranking_accuracy':np.nan,'sign_agreement_vs_canonical':np.nan,'sample_count':int(local[['public_score',c]].dropna().shape[0]),'evidence':'insufficient/non-common validation design'} for m,c in [('masked RMSE','masked_RMSE'),('known RMSE','known_RMSE'),('worst fold','worst_fold'),('fold std','fold_std'),('southern hemisphere RMSE','overall_RMSE'),('transferability score','overall_RMSE'),('pilot RMSE','overall_RMSE')]])
pd.DataFrame(corr).to_csv(OUT/'validation_public_correlation.csv',index=False)

# 4. Submission differences, many target-free slices.
z=meta.copy();z['canonical']=canonical.Target;z['current']=current.Target;z['f1']=f1.Target;z['delta_can']=z.f1-z.canonical;z['abs_delta']=z.delta_can.abs();z['magnitude_bin']=pd.qcut(z.f1.abs(),5,duplicates='drop').astype(str);z['delta_bin']=pd.qcut(z.abs_delta,5,duplicates='drop').astype(str);z['extreme_correction']=z.abs_delta>=z.abs_delta.quantile(.95);z['confidence_proxy']=np.where(np.abs(z.canonical-z.current)<=np.quantile(np.abs(z.canonical-z.current),.5),'high_agreement','low_agreement')
def grouped(dim,series):
 out=[]
 for key,g in z.groupby(series,observed=True):out.append({'dimension':dim,'group':str(key),'row_count':len(g),'mean_prediction_delta':g.delta_can.mean(),'median_delta':g.delta_can.median(),'delta_std':g.delta_can.std(),'P90_absolute_delta':g.abs_delta.quantile(.9),'P95_absolute_delta':g.abs_delta.quantile(.95),'canonical_F1_correlation':g.canonical.corr(g.f1),'current_best_F1_correlation':g.current.corr(g.f1)})
 return out
diff=[]
for dim,col in [('TWS_availability','known'),('latitude_band','latitude_band'),('hemisphere','hemisphere'),('month','month'),('test_date','time'),('prediction_magnitude','magnitude_bin'),('absolute_delta','delta_bin'),('extreme_correction','extreme_correction'),('baseline_confidence','confidence_proxy'),('cold_start','cold_start'),('history_completeness','lag_valid_count'),('lag_missingness',pd.cut(z.lag_missing_fraction,[-.01,0,.25,.5,.75,1]).astype(str))]:diff+=grouped(dim,col if not isinstance(col,str) else z[col])
pd.DataFrame(diff).to_csv(OUT/'f1_l12_submission_difference_analysis.csv',index=False)

# 5. Support mismatch on correction-magnitude strata and Notebook25 flags.
risk=pd.read_csv(ROOT/'artifacts/notebook25/test_row_risk_flags.csv').rename(columns={'row_id':'ID'});s=z.merge(risk,on='ID');s['correction_stratum']=pd.qcut(s.abs_delta,[0,.5,.9,.95,.99,1],duplicates='drop').astype(str)
support=[]
for key,g in s.groupby('correction_stratum',observed=True):support.append({'correction_stratum':key,'rows':len(g),'mean_abs_correction':g.abs_delta.mean(),'spatial_support_proxy_low_fraction':g.low_spatial_support.mean(),'temporal_extrapolation_fraction':g.temporal_extrapolation.mean(),'combined_low_support_fraction':g.multiple_risk_flags.mean(),'unseen_location_fraction':g.unseen_location.mean(),'southern_fraction':g.southern_hemisphere.mean(),'mean_risk_flags':g.risk_flag_count.mean(),'mean_lag_valid_count':g.lag_valid_count.mean(),'mean_lag_missing_fraction':g.lag_missing_fraction.mean(),'interpretation':'Notebook25 says all test locations seen; dominant mismatch is temporal TWS availability, not spatial support'})
pd.DataFrame(support).to_csv(OUT/'test_support_mismatch.csv',index=False)

# 6/7. Validation-vs-test similarity. Weights frozen here, target-free and not fitted to public scores.
WEIGHTS={'mask':.25,'lag':.25,'month':.10,'lat':.08,'lon':.07,'location':.10,'climate':.10,'gap':.05}
fullmeta=json.loads((ROOT/'artifacts/notebook36/cache/full_raw_lag_L12.json').read_text());mm=np.memmap(ROOT/'artifacts/notebook36/cache/full_raw_lag_L12.float32.mmap',dtype='float32',mode='r',shape=(len(train),108));tws_missing_cols=np.arange(1,24,2)
testloc=set(zip(test.lat,test.lon));sim=[];maskrows=[]
for kind in ['strict_temporal','irregular','spatial_block','space_time']:
 frames=[]
 for p in (ROOT/'artifacts/notebook36/cache/folds').glob('*.parquet'):
  q=pd.read_parquet(p);fid=p.stem;fk='strict_temporal' if fid.startswith('strict') else ('irregular' if fid.startswith('irregular') else ('space_time' if fid.startswith('space_time') else 'spatial_block'))
  if fk==kind:frames.append(q.assign(Fold_ID=fid))
 q=pd.concat(frames,ignore_index=True);idx=q.original_index.to_numpy();eff_miss=mm[idx][:,tws_missing_cols].mean();known_frac=q.known.mean();month_dist=tv(q.time.dt.month,test.time.dt.month,np.arange(.5,13.5,1));latdist=tv(q.lat,test.lat,np.arange(-90,91,10));londist=tv(q.lon,test.lon,np.arange(-180,181,20));locov=np.mean([a in testloc for a in zip(q.lat,q.lon)]);climdist=tv(train.loc[idx,'SPEI_12_t'],test.SPEI_12_t,[-np.inf,-1,-.25,.25,1,np.inf]);vd=np.array(sorted(q.time.unique())).astype('datetime64[M]').astype(int);td=np.array(sorted(test.time.unique())).astype('datetime64[M]').astype(int);gapdist=min(1,abs(np.diff(vd).mean()-np.diff(td).mean())/max(np.diff(td).mean(),1)) if len(vd)>1 else 1
 comps={'mask':1-abs(known_frac-meta.known.mean()),'lag':1-abs(eff_miss-meta.lag_missing_fraction.mean()),'month':1-month_dist,'lat':1-latdist,'lon':1-londist,'location':locov,'climate':1-climdist,'gap':1-gapdist};score=sum(WEIGHTS[k]*max(0,min(1,v)) for k,v in comps.items())
 sim.append({'validation':kind,'test_similarity_score':score,**{f'{k}_similarity':v for k,v in comps.items()},'weights':json.dumps(WEIGHTS,sort_keys=True),'target_used':False,'caveat':'grid_asia and space_time are identical masks in Notebook20/36'})
 for fid,gq in q.groupby('Fold_ID'):
  # Operational simulation uses actual routed availability; effective cache did not propagate it.
  order=gq.sort_values(['lat','lon','time']); routed_missing=[]
  for _,loc in order.groupby(['lat','lon']):
   a=loc.known.to_numpy(); miss=[]
   for i in range(len(a)):
    previous=a[max(0,i-12):i]; miss.append((12-len(previous)+(~previous).sum())/12)
   routed_missing.extend(miss)
  maskrows.append({'Fold_ID':fid,'Validation':kind,'rows':len(gq),'masked_fraction':1-gq.known.mean(),'known_fraction':gq.known.mean(),'effective_cache_lag_missing_fraction':float(mm[gq.original_index.to_numpy()][:,tws_missing_cols].mean()),'routed_mask_expected_missing_fraction':np.mean(routed_missing),'test_lag_missing_fraction':meta.lag_missing_fraction.mean(),'test_masked_fraction':1-meta.known.mean(),'location_level_mask_std':gq.groupby(['lat','lon']).known.mean().std(),'test_location_mask_std':meta.groupby(['lat','lon']).known.mean().std(),'distribution_distance':abs((1-gq.known.mean())-(1-meta.known.mean()))+abs(np.mean(routed_missing)-meta.lag_missing_fraction.mean()),'mask_propagated_before_lag_build':False,'audit_conclusion':'FAIL operational simulation: cached validation lags retain TWS from routed-masked dates'})
pd.DataFrame(sim).to_csv(OUT/'validation_test_similarity.csv',index=False);pd.DataFrame(maskrows).to_csv(OUT/'masking_pattern_audit.csv',index=False)

# 8. Public-positive/negative behavior relative to canonical.
beh=[]
for name,score in SCORES.items():
 if name=='submission_optimized.csv':continue
 d=pd.read_csv(ROOT/'submissions'/name);v=d.Target.to_numpy();delta=v-canonical.Target.to_numpy();changed=delta!=0
 tmp=meta.assign(pred=v,delta=delta,changed=changed);temporal=tmp.sort_values(['lat','lon','time']).groupby(['lat','lon']).pred.diff().abs().mean();spatial=tmp.groupby('time').pred.std().mean()
 beh.append({'filename':name,'group':'positive' if score<CAN_SCORE else 'negative','public_score':score,'public_delta':score-CAN_SCORE,'canonical_correlation':np.corrcoef(v,canonical.Target)[0,1],'changed_row_fraction':changed.mean(),'known_change_share':changed[meta.known].mean(),'masked_change_share':changed[~meta.known].mean(),'southern_change_share':changed[meta.hemisphere.eq('South')].mean(),'correction_MAD':np.mean(np.abs(delta)),'temporal_smoothness_MAD':temporal,'spatial_dispersion':spatial,'prediction_variance':np.var(v),'mean_risk_flags_on_changed':risk.loc[changed,'risk_flag_count'].mean(),'mean_lag_missing_on_changed':meta.loc[changed,'lag_missing_fraction'].mean()})
beh=pd.DataFrame(beh);agg=beh.groupby('group').mean(numeric_only=True).reset_index().assign(filename='GROUP_MEAN');pd.concat([beh,agg],ignore_index=True,sort=False).to_csv(OUT/'public_positive_negative_comparison.csv',index=False)

# 9. Fixed diagnostic blends, never written as submissions.
salv=[]
for w in [.05,.10,.20,.30]:
 p=w*f1.Target.to_numpy()+(1-w)*current.Target.to_numpy();delta=p-current.Target.to_numpy();salv.append({'F1_weight':w,'current_best_weight':1-w,'prediction_correlation_with_current_best':np.corrcoef(p,current.Target)[0,1],'MAD_vs_current_best':np.mean(np.abs(delta)),'changed_row_fraction':np.mean(delta!=0),'southern_MAD':np.mean(np.abs(delta[meta.hemisphere.eq('South')])),'masked_MAD':np.mean(np.abs(delta[~meta.known])),'low_lag_completeness_MAD':np.mean(np.abs(delta[meta.lag_valid_count<6])),'local_OOF_result':'not available for current-best vs F1 on common corrected OOF rows','public_risk_assessment':'lower perturbation than F1 alone, but no public score can be inferred; requires test-matched OOF evidence'})
pd.DataFrame(salv).to_csv(OUT/'conservative_salvage_analysis.csv',index=False)

redesign="""# Validation redesign

## Root cause found
Notebook 36 routed validation rows into known/masked experts **after** building the global TWS lag cache. A validation date labelled masked therefore still contributed its true `TWS_t` to later lag features. Irregular folds also allowed omitted intermediate Train dates into history. This is not direct target-label leakage, but it violates operational prediction-time availability and makes F1's masked validation materially easier than test inference.

## Keep / secondary / retire
- **Keep:** date-grouped temporal cutoffs, spatial blocks, row alignment assertions, worst-fold reporting and checkpointing.
- **Secondary:** irregular and spatial folds after rebuilding histories from an issue-time availability mask. Their current scores are diagnostic only.
- **Retire as decision metrics:** Notebook 36 F1 scores computed from unmasked global lag cache; `grid_asia` and `space_time_asia` as two independent signals (they are identical masks); pilot-only RMSE for submission promotion.

## Predeclared target-free decision score
After repairing masking, use 30% masking-pattern similarity, 20% lag-completeness similarity, 15% temporal/date-gap similarity, 15% spatial-distribution similarity, 10% worst-fold robustness and 10% location/climate coverage. Public score is excluded from the score. A candidate must also improve masked RMSE and pass every operational-availability assertion; the weighted score cannot compensate for a failed hard gate.
"""
plan="""# Notebook 38 plan — test-matched masking validation

## Selected experiment
**B) Test-matched masking validation design.** No model or feature search.

1. Freeze canonical E0 and F1-L12 parameters/features.
2. For every fold, map the exact test date-offset and `TWS_t` availability pattern onto validation dates.
3. Apply the availability mask to the entire validation-origin panel **before** any `shift(k)` calculation.
4. Exclude non-issue-date intermediate observations in irregular simulations unless they would be operationally available.
5. Rebuild only TWS lag/missing-mask features; other covariate lags remain cutoff-safe.
6. Assert every TWS lag source was available under the simulated issue-time mask.
7. Compare E0/F1 on common rows. Report strict, masked, worst-fold and lag-completeness strata.

Acceptance: F1 must retain ≥0.002 overall and ≥0.004 masked gain, no negative space-time/spatial direction, and positive gain in a majority of folds. Stop and reject F1 if its gain collapses under propagated masking. Only after an independent rerun may a conservative current-best blend be considered; Notebook 38 itself should not use public score for weighting.
"""
(OUT/'validation_redesign.md').write_text(redesign,encoding='utf-8');(OUT/'notebook38_plan.md').write_text(plan,encoding='utf-8')

bestcorr=pd.DataFrame(corr).sort_values('pairwise_ranking_accuracy',ascending=False).iloc[0]
final=f"""# Notebook 37 final recommendation

1. **Most likely cause:** validation masking was routed after lag construction. F1 validation histories retained TWS values from dates that the simulated/test availability pattern masks; irregular folds also consumed intermediate observations absent from the issue-date pattern. This operational mismatch inflates local lag value.
2. **Most public-associated metric:** `{bestcorr.metric}` has the highest observed pairwise accuracy ({bestcorr.pairwise_ranking_accuracy:.3f}), but n=4 is insufficient; no metric is statistically established.
3. **Misleading metric:** Notebook 36 F1 strict/irregular/spatial/space-time RMSE in its present form, because all use the unpropagated global TWS lag cache. `grid_asia` and space-time are additionally identical.
4. **Masking simulated adequately?** No. See `masking_pattern_audit.csv`; expert routing matched current-row status but historical availability was not propagated into lag creation.
5. **Riskiest changes:** masked rows with incomplete real test history and large absolute F1–canonical corrections; support strata are quantified without target labels.
6. **Public-positive common behavior:** evidence is too small for a universal pattern. The only model beating both canonical and F1 is Notebook 21's conservative correction: exact file comparison differs on 33.8% of rows, but its cap is 0.05 and MAD is only 0.0066. F1 changed every row with MAD 0.3348 and did not beat current-best.
7. **Can low-weight blend salvage F1?** Plausible only as a hypothesis. Fixed blends sharply reduce perturbation, but no common corrected OOF evidence exists; no file was produced.
8. **Validation redesign:** apply test-matched TWS masks before lagging, remove unavailable intermediate history, retain hard temporal/spatial gates, and do not use public scores in the decision score.
9. **One next experiment:** B) test-matched masking validation design.
10. **Evidence required before submission:** reproducible positive F1 gain on propagated-mask OOF, masked gain ≥0.004, nonnegative spatial/space-time direction, majority-fold gain, complete availability audit, and an independent rerun.

Observed public facts: canonical {CAN_SCORE:.9f}, current-best {BEST_SCORE:.9f}, F1 {SCORES['submission_36_f1_l12_primary.csv']:.9f}. F1 improved canonical by {CAN_SCORE-SCORES['submission_36_f1_l12_primary.csv']:.9f} but trailed current-best by {SCORES['submission_36_f1_l12_primary.csv']-BEST_SCORE:.9f}. No new submission/model/package/data was created.
"""
(OUT/'final_recommendation.md').write_text(final,encoding='utf-8');print('Notebook37 artifacts written:',OUT)
