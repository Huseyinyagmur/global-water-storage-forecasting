from pathlib import Path
from time import perf_counter
import hashlib, json, warnings, gc
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, linregress
from sklearn.metrics import root_mean_squared_error, mean_absolute_error
from lightgbm import LGBMRegressor
from IPython.display import display, Markdown

warnings.filterwarnings("ignore")
SEED=42; RNG=np.random.default_rng(SEED); T0=perf_counter()
ROOT=Path.cwd().resolve()
if not (ROOT/"data/raw/Train.csv").exists(): ROOT=ROOT.parent
DATA=ROOT/"data/raw"; SUB=ROOT/"submissions"; ART=ROOT/"artifacts/notebook23"; ART.mkdir(parents=True,exist_ok=True)
CANON=SUB/"submission_optimized.csv"; BEST=SUB/"submission_21_spatial_residual_candidate.csv"
CANON_SHA="b61ecd15862ca6d7527c4f494de802ed6ea6d31c420737cb78f977d9bda3b968"
BEST_SHA="84d4ce6efdbfc3fbca55d9eea1ebccfe628ffa962e31b62a654c179dc649b488"
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def stamp(s): print(f"[{s}] elapsed={perf_counter()-T0:.1f}s")

train=pd.read_csv(DATA/"Train.csv",parse_dates=["time"]); test=pd.read_csv(DATA/"Test.csv",parse_dates=["time"])
sample=pd.read_csv(DATA/"SampleSubmission.csv"); canon=pd.read_csv(CANON); best=pd.read_csv(BEST)
known_test=test.TWS_t.notna().to_numpy(); delta=best.Target.to_numpy()-canon.Target.to_numpy()
repro={"canonical_exists":CANON.exists(),"current_best_exists":BEST.exists(),"canonical_sha256":sha(CANON),
 "current_best_sha256":sha(BEST),"canonical_sha_match":sha(CANON)==CANON_SHA,"current_best_sha_match":sha(BEST)==BEST_SHA,
 "rows":len(best),"id_unique":best.ID.is_unique,"id_order":best.ID.equals(sample.ID),"nan":int(best.Target.isna().sum()),
 "infinity":int(np.isinf(best.Target).sum()),"known_rows":int(known_test.sum()),"masked_rows":int((~known_test).sum()),
 "correlation":np.corrcoef(best.Target,canon.Target)[0,1],"mad":np.abs(delta).mean(),
 "maximum_difference":np.abs(delta).max(),"changed_row_fraction":(np.abs(delta)>1e-12).mean(),
 "known_rows_exact":np.array_equal(best.Target.to_numpy()[known_test],canon.Target.to_numpy()[known_test])}
pd.DataFrame([repro]).to_csv(ART/"reproduction_report.csv",index=False)
assert repro["canonical_sha_match"] and repro["current_best_sha_match"] and repro["id_order"] and repro["known_rows_exact"]

# Policy is read, not selected again.
abl=pd.read_csv(ROOT/"artifacts/notebook21/correction_ablation.csv")
stab21=pd.read_csv(ROOT/"artifacts/notebook21/residual_stability.csv")
selected=abl[abl.Accepted.astype(bool)].iloc[0]
assert selected.Candidate=="hemisphere|light|cap_0.05" and selected.Alpha==200 and selected.Cap==.05
south=stab21[(stab21.group_type=="hemisphere")&(stab21.group=="South")].iloc[0]
policy={"candidate":selected.Candidate,"correction_group":"hemisphere","group_hierarchy":"hemisphere -> zero",
 "residual_definition":"target - prediction","minimum_sample_per_fold":100,"minimum_folds":3,
 "sign_consistency_threshold":.75,"minimum_absolute_bias":.01,"shrinkage_alpha":200.0,"correction_cap":.05,
 "fallback":"zero correction","eligibility":"stable-bias group and masked row only","known_policy":"unchanged"}
# Positive stable South mean saturates at the frozen cap; North has no stable-bias eligibility.
assert bool(south.stable_bias) and south.mean_residual>0
# Notebook 21's selected South estimate exceeded +0.05, so the frozen cap is the applied value.
test_corr=np.where((test.lat.to_numpy()<0)&(~known_test),float(selected.Cap),0.0)
reproduced=canon.Target.to_numpy()+test_corr
diff=reproduced-best.Target.to_numpy()
policy.update({"prediction_correlation":np.corrcoef(reproduced,best.Target)[0,1],"mad":np.abs(diff).mean(),
 "maximum_difference":np.abs(diff).max(),"exact_match_rate":(np.abs(diff)<=1e-12).mean(),
 "reproduction_pass":bool(np.abs(diff).max()<=1e-12)})
pd.DataFrame([policy]).to_csv(ART/"correction_policy_reproduction.csv",index=False)
assert policy["reproduction_pass"]
stamp("reproduction gates")

SPEI=["SPEI_01_t","SPEI_03_t","SPEI_06_t","SPEI_12_t"]
def static(df):
 o=df.copy(); dt=o.time; o["month"]=dt.dt.month.astype("int8"); o["quarter"]=dt.dt.quarter.astype("int8"); o["year"]=dt.dt.year.astype("int16")
 o["elapsed_month"]=((dt.dt.year-train.time.dt.year.min())*12+dt.dt.month-1).astype("int16")
 o["month_sin_calc"]=np.sin(2*np.pi*o.month/12); o["month_cos_calc"]=np.cos(2*np.pi*o.month/12)
 o["season_sin"]=np.sin(2*np.pi*o.quarter/4); o["season_cos"]=np.cos(2*np.pi*o.quarter/4)
 s=o[SPEI]; o["spei_mean"]=s.mean(1); o["spei_min"]=s.min(1); o["spei_max"]=s.max(1); o["spei_std"]=s.std(1)
 o["spei_01_minus_12"]=o.SPEI_01_t-o.SPEI_12_t; o["spei_03_minus_12"]=o.SPEI_03_t-o.SPEI_12_t
 o["spei_short_long_ratio"]=o.SPEI_01_t/(o.SPEI_12_t.abs()+.1); o["lat_band_5"]=np.floor((o.lat+90)/5)
 o["lon_band_5"]=np.floor((o.lon+180)/5); o["spatial_bin_10"]=o.lat_band_5*100+o.lon_band_5
 o["lat_lon_interaction"]=o.lat*o.lon/1000; o["lat_sin"]=np.sin(np.radians(o.lat)); o["lat_cos"]=np.cos(np.radians(o.lat))
 o["lon_sin"]=np.sin(np.radians(o.lon)); o["lon_cos"]=np.cos(np.radians(o.lon)); return o
X=static(train)
BASE=["lat","lon","TWS_t",*SPEI,"SOIL_MOISTURE_t","month","month_sin","month_cos","year","quarter","elapsed_month",
"month_sin_calc","month_cos_calc","season_sin","season_cos","spei_mean","spei_min","spei_max","spei_std","spei_01_minus_12",
"spei_03_minus_12","spei_short_long_ratio","lat_band_5","lon_band_5","spatial_bin_10","lat_lon_interaction","lat_sin","lat_cos","lon_sin","lon_cos"]
MASK=[x for x in BASE if x!="TWS_t"]; PRI=["prior_location_mean","prior_location_std","prior_location_month","prior_global_month"]
PARAM=dict(n_estimators=160,learning_rate=.08,num_leaves=63,min_child_samples=100,reg_lambda=1,subsample=.85,colsample_bytree=.9,random_state=SEED,n_jobs=4,verbosity=-1)
def lgb(): return LGBMRegressor(**PARAM)
def train_prior(df):
 o=df.sort_values(["lat","lon","time"]).copy(); sh=o.groupby(["lat","lon"]).target.shift(1)
 o["prior_location_mean"]=sh.groupby([o.lat,o.lon]).expanding().mean().reset_index(level=[0,1],drop=True)
 o["prior_location_std"]=sh.groupby([o.lat,o.lon]).expanding().std().reset_index(level=[0,1],drop=True)
 o["prior_location_month"]=sh.groupby([o.lat,o.lon,o.month]).expanding().mean().reset_index(level=[0,1,2],drop=True)
 o["prior_global_month"]=o.groupby("month").target.transform(lambda z:z.shift(1).expanding().mean()); return o.sort_index()
def apply_prior(h,a):
 loc=h.groupby(["lat","lon"]).target.agg(["mean","std"]).rename(columns={"mean":"prior_location_mean","std":"prior_location_std"})
 lm=h.groupby(["lat","lon","month"]).target.agg(["sum","count"]).rename(columns={"sum":"s","count":"n"}); gm=h.groupby("month").target.mean().rename("prior_global_month")
 o=a.join(loc,on=["lat","lon"]).join(lm,on=["lat","lon","month"]).join(gm,on="month"); o["prior_location_month"]=(o.s+4*o.prior_location_mean)/(o.n.fillna(0)+4)
 return o.drop(columns=["s","n"])
known_positions=set(np.flatnonzero(test.groupby("time").TWS_t.count().to_numpy()>1000))
def predict_block(train_dates,valid_dates,name):
 tr=X[X.time.isin(train_dates)]; va=X[X.time.isin(valid_dates)]; assert tr.time.max()<va.time.min() and set(tr.time.unique()).isdisjoint(va.time.unique())
 vd=np.array(sorted(va.time.unique())); known=va.time.isin({d for j,d in enumerate(vd) if j in known_positions}).to_numpy()
 km=lgb().fit(tr[BASE],tr.target); bm=lgb().fit(tr[MASK],tr.target); tp=train_prior(tr); vp=apply_prior(tr,va); hm=lgb().fit(tp[MASK+PRI],tp.target)
 p=np.empty(len(va)); p[known]=km.predict(va.loc[known,BASE]); p[~known]=.65*bm.predict(va.loc[~known,MASK])+.35*hm.predict(vp.loc[~known,MASK+PRI])
 out=va[["time","lat","lon","target","SPEI_12_t","SOIL_MOISTURE_t"]].copy(); out["prediction"]=p; out["residual"]=out.target-p; out["known"]=known; out["Fold"]=name
 out["hemisphere"]=np.where(out.lat<0,"South","North"); out["climate_regime"]=pd.cut(out.SPEI_12_t,[-np.inf,-1,1,np.inf],labels=["dry","normal","wet"]).astype(str)
 return out.reset_index(drop=True)

dates=np.array(sorted(X.time.unique())); test_dates=np.array(sorted(test.time.unique()))
test_offsets=np.array([(d.year-test_dates[0].year)*12+d.month-test_dates[0].month for d in test_dates]); test_gaps=np.diff(test_offsets)
cal_specs=[(36,48),(48,60),(60,72)]; win_specs=[(72,84),(84,96),(96,108),(126,138)]
all_blocks=[]
for i,(a,b) in enumerate(cal_specs,1): all_blocks.append(predict_block(dates[:a],dates[a:b],f"calibration_{i}")); stamp(f"calibration {i}")
window_meta=[]
for i,(a,b) in enumerate(win_specs,1):
 vd=dates[a:b]; gaps=np.diff(vd.astype("datetime64[M]").astype(int)); similarity=1/(1+abs(gaps.mean()-test_gaps.mean())+abs(gaps.std()-test_gaps.std()))
 va=X[X.time.isin(vd)]; window_meta.append({"Window ID":f"window_{i}","Train start":dates[0],"Train end":dates[a-1],"Validation start":vd[0],"Validation end":vd[-1],
  "Validation dates":"|".join(pd.to_datetime(vd).strftime("%Y-%m-%d")),"Validation date count":len(vd),"Validation row count":len(va),
  "Date-gap profile":json.dumps(gaps.tolist()),"Spatial coverage":f"lat {va.lat.min()}..{va.lat.max()}, lon {va.lon.min()}..{va.lon.max()}",
  "Climate-regime coverage":int(pd.cut(va.SPEI_12_t,[-np.inf,-1,1,np.inf]).nunique()),"Test-date similarity score":similarity})
pd.DataFrame(window_meta).to_csv(ART/"temporal_windows.csv",index=False)

def ci_mean(x):
 x=np.asarray(x); se=np.std(x,ddof=1)/np.sqrt(len(x)); return (x.mean()-1.96*se,x.mean()+1.96*se)
def learned_policy(hist):
 per=hist.groupby(["hemisphere","Fold"]).residual.agg(["mean","count"]); maps={}; details={}
 for hemi,g in per.groupby(level=0):
  vals=g["mean"]; pooled=hist.loc[hist.hemisphere==hemi,"residual"]; lo,hi=ci_mean(pooled); dom=np.sign(vals.mean()); consistency=(np.sign(vals)==dom).mean()
  stable=len(vals)>=3 and consistency>=.75 and g["count"].min()>=100 and abs(pooled.mean())>=.01 and (lo>0 or hi<0)
  raw=len(pooled)/(len(pooled)+200)*pooled.mean(); maps[hemi]=float(np.clip(raw,-.05,.05)) if stable else 0.0
  details[hemi]=(pooled.mean(),len(pooled),consistency,lo,hi,stable,raw)
 return maps,details

validation_rows=[]; sign_rows=[]; prediction_windows=[]
for wi,(a,b) in enumerate(win_specs,1):
 name=f"window_{wi}"; future=predict_block(dates[:a],dates[a:b],name); hist=pd.concat(all_blocks,ignore_index=True)
 assert hist.time.max()<future.time.min(); maps,details=learned_policy(hist)
 corr=future.hemisphere.map(maps).fillna(0).to_numpy(copy=True); corr[future.known.to_numpy()]=0; corrected=future.prediction.to_numpy()+corr
 future["correction"]=corr; future["corrected_prediction"]=corrected; future["gain_sq"]=future.residual**2-(future.target-corrected)**2; prediction_windows.append(future)
 regions={"south":future.lat<0,"north":future.lat>=0,"americas":(future.lon>=-140)&(future.lon<-30),"asia":(future.lat>=10)&(future.lat<60)&(future.lon>=60)&(future.lon<150)}
 block_rmses=[root_mean_squared_error(future.loc[m,"target"],corrected[m]) for m in regions.values() if m.sum()]
 base=root_mean_squared_error(future.target,future.prediction); cr=root_mean_squared_error(future.target,corrected)
 validation_rows.append({"Window":name,"Canonical RMSE":base,"Corrected RMSE":cr,"Absolute gain":base-cr,"Relative gain":(base-cr)/base,
  "Canonical MAE":mean_absolute_error(future.target,future.prediction),"Corrected MAE":mean_absolute_error(future.target,corrected),"Worst spatial block RMSE":max(block_rmses),
  "Southern hemisphere RMSE":root_mean_squared_error(future.loc[future.lat<0,"target"],corrected[future.lat<0]),
  "Northern hemisphere RMSE":root_mean_squared_error(future.loc[future.lat>=0,"target"],corrected[future.lat>=0]),
  "Correction mean":corr.mean(),"Correction mean absolute":np.abs(corr).mean(),"Correction maximum":np.abs(corr).max(),"Corrected row fraction":(np.abs(corr)>0).mean(),
  "Residual source end":hist.time.max(),"Validation start":future.time.min()})
 for hemi in ["South","North"]:
  hg=future[future.hemisphere==hemi]; hmean,n,cons,lo0,hi0,stable,raw=details[hemi]; lo,hi=ci_mean(hg.residual); c=maps[hemi]
  before=np.mean(hg.residual**2); after=np.mean((hg.residual-c*np.where(hg.known,0,1))**2)
  label="useful" if after<before-1e-9 else ("harmful" if after>before+1e-9 else "neutral")
  sign_rows.append({"Window":name,"Group":hemi,"Historical training mean residual":hmean,"Future validation mean residual":hg.residual.mean(),
   "Historical correction sign":int(np.sign(c)),"Future observed residual sign":int(np.sign(hg.residual.mean())),"Sign match":bool(np.sign(c)==np.sign(hg.residual.mean())) if c else np.nan,
   "Residual magnitude ratio":abs(hg.residual.mean())/(abs(hmean)+1e-12),"Sample count":len(hg),"CI low":lo,"CI high":hi,"Label":label,"Correction estimate":c,
   "Optimal hindsight correction":hg.residual.mean(),"Historical sign consistency":cons})
 all_blocks.append(future); stamp(name)
validation=pd.DataFrame(validation_rows); validation.to_csv(ART/"window_validation_results.csv",index=False)
sign=pd.DataFrame(sign_rows); sign.to_csv(ART/"sign_transferability.csv",index=False)
preds=pd.concat(prediction_windows,ignore_index=True)

mag=[]
for group,g in sign.groupby("Group"):
 direction=g["Sign match"].fillna(False).mean(); est=g["Correction estimate"]; opt=g["Optimal hindsight correction"]
 cv=opt.std(ddof=0)/(abs(opt.mean())+1e-12); ratio=np.where(abs(opt)>1e-12,est/opt,np.nan); err=np.abs(est-opt)
 cls="Stable direction and magnitude" if direction>=.75 and cv<=1 else ("Stable direction, unstable magnitude" if direction>=.75 else ("Unstable direction" if len(g)>=3 else "Insufficient evidence"))
 mag.append({"Group":group,"Window count":len(g),"Mean residual":opt.mean(),"Median residual":opt.median(),"Residual standard deviation":opt.std(ddof=0),
  "Mean correction estimate":est.mean(),"Mean optimal hindsight correction":opt.mean(),"Estimated/optimal correction ratio":np.nanmean(ratio),
  "Absolute calibration error":err.mean(),"Fold-to-fold coefficient of variation":cv,"Direction accuracy":direction,"Classification":cls})
magnitude=pd.DataFrame(mag); magnitude.to_csv(ART/"correction_magnitude_stability.csv",index=False)

# Temporal decay: source-residual recency varies, especially for the deliberately separated final window.
dec=[]
for _,r in validation.iterrows():
 source_end=pd.Timestamp(r["Residual source end"]); valid_start=pd.Timestamp(r["Validation start"])
 gap=(valid_start.year-source_end.year)*12+valid_start.month-source_end.month
 sg=sign[sign.Window==r["Window"]]; active=sg[sg["Correction estimate"]!=0]
 dec.append({"Window":r["Window"],"Month gap":gap,"Sign accuracy":active["Sign match"].mean(),"RMSE gain":r["Absolute gain"],
  "Correction calibration error":np.mean(np.abs(active["Correction estimate"]-active["Optimal hindsight correction"])),"Harmful correction rate":(active.Label=="harmful").mean()})
decay=pd.DataFrame(dec)
rho=spearmanr(decay["Month gap"],decay["RMSE gain"]).statistic; trend=linregress(decay["Month gap"],decay["RMSE gain"])
boot=[]
for _ in range(500):
 z=decay.sample(len(decay),replace=True,random_state=int(RNG.integers(1e9))); v=spearmanr(z["Month gap"],z["RMSE gain"]).statistic
 if np.isfinite(v): boot.append(v)
decay["Spearman gap vs gain"]=rho; decay["Linear gain slope per month"]=trend.slope; decay["Bootstrap Spearman CI low"]=np.quantile(boot,.025) if boot else np.nan
decay["Bootstrap Spearman CI high"]=np.quantile(boot,.975) if boot else np.nan; decay["Small-sample caveat"]="Only four windows; trend is exploratory, not causal."
decay.to_csv(ART/"temporal_decay_analysis.csv",index=False)

def aggregate_transfer(col,definitions):
 rows=[]
 for name,maskfun in definitions.items():
  vals=[]
  for w,g in preds.groupby("Fold"):
   m=maskfun(g); q=g[m]
   if not len(q): continue
   b=root_mean_squared_error(q.target,q.prediction); c=root_mean_squared_error(q.target,q.corrected_prediction)
   active=q[np.abs(q.correction)>0]; vals.append((b,c,b-c,len(q),(active.gain_sq<0).mean() if len(active) else np.nan))
  a=np.array(vals,float); rows.append({col:name,"Window count":len(vals),"Canonical RMSE":np.sqrt(np.average(a[:,0]**2,weights=a[:,3])),
   "Corrected RMSE":np.sqrt(np.average(a[:,1]**2,weights=a[:,3])),"Mean gain":a[:,2].mean(),"Gain consistency":(a[:,2]>=0).mean(),
   "Worst-window change":np.min(a[:,2]),"Sign transfer accuracy":sign.loc[sign.Group.eq(name) if name in ["South","North"] else pd.Series(False,index=sign.index),"Sign match"].mean(),
   "Harmful correction rate":np.nanmean(a[:,4])})
 return pd.DataFrame(rows)
regions={"North":lambda d:d.lat>=0,"South":lambda d:d.lat<0,"Equatorial":lambda d:d.lat.abs()<10,"High latitudes":lambda d:d.lat.abs()>=60,
 "Mid latitudes":lambda d:(d.lat.abs()>=30)&(d.lat.abs()<60),"Low latitudes":lambda d:d.lat.abs()<30,"Corrected South group":lambda d:d.lat<0}
regional=aggregate_transfer("Region",regions); regional.to_csv(ART/"regional_transferability.csv",index=False)
cl=[]
for regime,g in preds.groupby("climate_regime"):
 vals=[]
 for w,q in g.groupby("Fold"):
  b=root_mean_squared_error(q.target,q.prediction); c=root_mean_squared_error(q.target,q.corrected_prediction); active=q[np.abs(q.correction)>0]
  vals.append((b,c,b-c,len(q),(active.gain_sq<0).mean() if len(active) else np.nan,np.abs(q.correction).mean()))
 a=np.array(vals,float); cl.append({"Climate regime":regime,"Row count":len(g),"Window count":g.Fold.nunique(),"Canonical RMSE":np.sqrt(np.average(a[:,0]**2,weights=a[:,3])),
  "Corrected RMSE":np.sqrt(np.average(a[:,1]**2,weights=a[:,3])),"Mean gain":a[:,2].mean(),"Sign accuracy":sign["Sign match"].mean(),
  "Harmful correction rate":np.nanmean(a[:,4]),"Correction magnitude stability":a[:,5].std(ddof=0)})
climate=pd.DataFrame(cl); climate.to_csv(ART/"climate_regime_transferability.csv",index=False)

# Frozen 0-100 score weights and acceptance thresholds.
active_sign=sign[sign["Correction estimate"]!=0]; weighted_sign=np.average(active_sign["Sign match"].astype(float),weights=active_sign["Sample count"])
gain_cons=(validation["Absolute gain"]>=-1e-6).mean(); harmful=(active_sign.Label=="harmful").mean(); worst=validation["Absolute gain"].min()
regional_cons=(regional["Mean gain"]>=0).mean(); mag_stable=(magnitude.Classification!="Unstable direction").mean()
decay_component=np.clip((rho+1)/2,0,1) if np.isfinite(rho) else .5
components={"window_gain_consistency":gain_cons,"weighted_sign_accuracy":weighted_sign,"inverse_harmful_rate":1-harmful,
 "worst_window_behavior":np.clip(1-max(0,-worst)/.002,0,1),"regional_consistency":regional_cons,"magnitude_stability":mag_stable,"temporal_decay_resilience":decay_component}
weights={"window_gain_consistency":.25,"weighted_sign_accuracy":.20,"inverse_harmful_rate":.15,"worst_window_behavior":.15,"regional_consistency":.10,"magnitude_stability":.10,"temporal_decay_resilience":.05}
score=100*sum(components[k]*weights[k] for k in weights); label="Strongly transferable" if score>=80 else ("Moderately transferable" if score>=65 else ("Weakly transferable" if score>=50 else "Not transferable"))
south_reg=regional[regional.Region=="South"].iloc[0]
gates={"majority_nonworse":gain_cons>=.60,"weighted_sign_above_55pct":weighted_sign>.55,"harmful_below_40pct":harmful<.40,
 "worst_deterioration_at_most_0.002":worst>=-.002,"south_not_single_period":south_reg["Gain consistency"]>=.50 and (sign[(sign.Group=="South")&(sign.Label=="useful")].shape[0]>=2),
 "magnitude_not_fully_unstable":mag_stable>=.50,"score_at_least_moderate":score>=65}
score_row={"Score":score,"Classification":label,**components,**{f"weight_{k}":v for k,v in weights.items()},**{f"Gate {k}":v for k,v in gates.items()},"All gates passed":all(gates.values())}
pd.DataFrame([score_row]).to_csv(ART/"transferability_score.csv",index=False)
if all(gates.values()) and score>=80: interpretation="A) Robust refinement"
elif score>=65: interpretation="B) Useful but regime-dependent"
elif score>=50: interpretation="C) Leaderboard-positive but weakly transferable"
else: interpretation="D) Unreliable correction"

final=f'''# Notebook 23 final recommendation

1. **Exact policy reproduction:** {policy["reproduction_pass"]}; max difference {policy["maximum_difference"]:.3g}, exact-within-tolerance rate {policy["exact_match_rate"]:.3%}.
2. **Independent windows:** {len(validation)}.
3. **Windows improved:** {(validation["Absolute gain"]>0).sum()} / {len(validation)}.
4. **Mean / median RMSE gain:** {validation["Absolute gain"].mean():.8f} / {validation["Absolute gain"].median():.8f}.
5. **Sign transfer:** overall {active_sign["Sign match"].mean():.3%}; sample-weighted {weighted_sign:.3%}.
6. **Magnitude stability:** {"; ".join(magnitude.Group+": "+magnitude.Classification)}.
7. **Temporal decay:** Spearman(gap,gain)={rho:.3f}, slope={trend.slope:.8f}/month. With four windows this is exploratory and does not prove decay.
8. **Southern transfer:** mean gain {south_reg["Mean gain"]:.8f}, positive/non-worse consistency {south_reg["Gain consistency"]:.1%}.
9. **Climate dependence:** see `climate_regime_transferability.csv`; differences are descriptive, not a new selection rule.
10. **Transferability score:** {score:.2f}/100 — {label}. Thresholds are a research risk-control policy, not scientific constants.
11. **Interpretation:** {interpretation}.
12. **Preserve current-best submission:** Yes; no submission file was modified.
13. **Further correction-family optimization:** {"Only a single uncertainty-weighting experiment is justified" if all(gates.values()) else "No; transfer evidence is insufficient for optimization"}.
14. **Notebook 24 single next experiment:** frozen correction uncertainty weighting, only if the acceptance gates above pass; otherwise stop this correction family.

All unsuccessful windows remain in `window_validation_results.csv`. Correction estimates use only earlier OOF residuals; validation residuals are appended only after their own evaluation.
'''
(ART/"final_recommendation.md").write_text(final,encoding="utf-8"); display(Markdown(final)); display(validation)
required=["reproduction_report.csv","correction_policy_reproduction.csv","temporal_windows.csv","window_validation_results.csv","sign_transferability.csv",
"correction_magnitude_stability.csv","temporal_decay_analysis.csv","regional_transferability.csv","climate_regime_transferability.csv","transferability_score.csv","final_recommendation.md"]
assert all((ART/x).exists() for x in required); assert sha(CANON)==CANON_SHA and sha(BEST)==BEST_SHA
stamp("complete")
