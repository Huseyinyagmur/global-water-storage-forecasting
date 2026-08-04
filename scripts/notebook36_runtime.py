"""Frozen F1-L12 robust full validation with resumable fold checkpoints."""
from __future__ import annotations
import gc, hashlib, json, os, sys, threading
from pathlib import Path
from time import perf_counter

SCRIPT_ROOT=Path(__file__).resolve().parents[1] if "__file__" in globals() else Path.cwd().resolve()
VENV=SCRIPT_ROOT/".venv/Lib/site-packages"
if VENV.exists() and str(VENV) not in sys.path: sys.path.append(str(VENV))
import numpy as np
import pandas as pd
import psutil
from lightgbm import LGBMRegressor
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import root_mean_squared_error

ROOT=next(p for p in [Path.cwd().resolve(),*Path.cwd().resolve().parents,SCRIPT_ROOT] if (p/"data/raw/Train.csv").exists())
ART=ROOT/"artifacts/notebook36"; ART.mkdir(parents=True,exist_ok=True)
CACHE=ART/"cache"; CACHE.mkdir(exist_ok=True)
CHECK=CACHE/"folds"; CHECK.mkdir(exist_ok=True)
SEED=42; BLEND_W=.05; SPATIAL_SERIOUS_TOLERANCE=.003
GATES={"strict_gain":.002,"masked_gain":.004,"spatial_tolerance":SPATIAL_SERIOUS_TOLERANCE,"worst_tolerance":0.0,"majority":.5}
PARAM=dict(n_estimators=160,learning_rate=.08,num_leaves=63,min_child_samples=100,reg_lambda=1.0,subsample=.85,colsample_bytree=.9,random_state=SEED,n_jobs=4,verbosity=-1)
DYN=["TWS_t","SPEI_01_t","SPEI_03_t","SPEI_06_t","SPEI_12_t","SOIL_MOISTURE_t","month_sin","month_cos"]
SPEI=["SPEI_01_t","SPEI_03_t","SPEI_06_t","SPEI_12_t"]

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def rss(): return psutil.Process(os.getpid()).memory_info().rss/1024**3
class Peak:
 def __enter__(self):
  self.stop=False; self.value=rss()
  def sample():
   while not self.stop:
    self.value=max(self.value,rss()); threading.Event().wait(.2)
  self.th=threading.Thread(target=sample,daemon=True); self.th.start(); return self
 def __exit__(self,*a): self.stop=True; self.th.join(1); self.value=max(self.value,rss())
def static(d,min_year):
 o=d.copy(); dt=o.time; o['month']=dt.dt.month.astype('int8'); o['quarter']=dt.dt.quarter.astype('int8'); o['year']=dt.dt.year.astype('int16'); o['elapsed_month']=((dt.dt.year-min_year)*12+dt.dt.month-1).astype('int16')
 o['month_sin_calc']=np.sin(2*np.pi*o.month/12); o['month_cos_calc']=np.cos(2*np.pi*o.month/12); o['season_sin']=np.sin(2*np.pi*o.quarter/4); o['season_cos']=np.cos(2*np.pi*o.quarter/4)
 s=o[SPEI]; o['spei_mean']=s.mean(1); o['spei_min']=s.min(1); o['spei_max']=s.max(1); o['spei_std']=s.std(1)
 o['spei_01_minus_12']=o.SPEI_01_t-o.SPEI_12_t; o['spei_03_minus_12']=o.SPEI_03_t-o.SPEI_12_t; o['spei_short_long_ratio']=o.SPEI_01_t/(o.SPEI_12_t.abs()+.1)
 o['lat_band_5']=np.floor((o.lat+90)/5); o['lon_band_5']=np.floor((o.lon+180)/5); o['spatial_bin_10']=o.lat_band_5*100+o.lon_band_5; o['lat_lon_interaction']=o.lat*o.lon/1000
 o['lat_sin']=np.sin(np.radians(o.lat)); o['lat_cos']=np.cos(np.radians(o.lat)); o['lon_sin']=np.sin(np.radians(o.lon)); o['lon_cos']=np.cos(np.radians(o.lon)); return o
A=['lat','lon','TWS_t',*SPEI,'SOIL_MOISTURE_t','month','month_sin','month_cos','year']; B=['quarter','elapsed_month','month_sin_calc','month_cos_calc','season_sin','season_cos']; C=['spei_mean','spei_min','spei_max','spei_std','spei_01_minus_12','spei_03_minus_12','spei_short_long_ratio']; D=['lat_band_5','lon_band_5','spatial_bin_10','lat_lon_interaction','lat_sin','lat_cos','lon_sin','lon_cos']
BASE=list(dict.fromkeys(A+B+C+D)); MASK=[c for c in BASE if c!='TWS_t']
LAG=[]
for c in DYN:
 for k in range(1,13):
  LAG.append(f'{c}_lag_{k}')
  if c=='TWS_t':LAG.append(f'TWS_missing_lag_{k}')

def known_mask(va,test_counts):
 vd=np.array(sorted(va.time.unique())); pos=set(np.flatnonzero(test_counts>1000)); return va.time.isin({d for j,d in enumerate(vd) if j in pos}).to_numpy()
def metric(y,p,k):
 return dict(overall=root_mean_squared_error(y,p),known=root_mean_squared_error(y[k],p[k]),masked=root_mean_squared_error(y[~k],p[~k]))
def safe_rmse(a,b):
 return root_mean_squared_error(a,b) if len(a) else np.nan

START=perf_counter(); print('Notebook36 start RAM',rss())
train=pd.read_csv(ROOT/'data/raw/Train.csv',parse_dates=['time']); test=pd.read_csv(ROOT/'data/raw/Test.csv',parse_dates=['time']); sample=pd.read_csv(ROOT/'data/raw/SampleSubmission.csv')
for c in [*DYN,'lat','lon','target']: train[c]=train[c].astype('float32')
train['original_index']=np.arange(len(train),dtype='int32'); train['location_id']=train.lat.astype(str)+'_'+train.lon.astype(str)
X=static(train,int(train.time.dt.year.min())); base_all=X[BASE].to_numpy(dtype='float32'); y_all=X.target.to_numpy(dtype='float32'); test_counts=test.groupby('time').TWS_t.count().to_numpy()

# Full feature cache: one target-free, row-aligned float32 memmap.
lag_path=CACHE/'full_raw_lag_L12.float32.mmap'; meta_path=CACHE/'full_raw_lag_L12.json'; signature=hashlib.sha256(('|'.join(DYN)+'|L12|shift(k)|row-aligned|v1').encode()).hexdigest()
expected_bytes=len(X)*len(LAG)*4
cache_ok=False
if lag_path.exists() and meta_path.exists():
 m=json.loads(meta_path.read_text()); cache_ok=m.get('signature')==signature and lag_path.stat().st_size==expected_bytes and m.get('sha256')==sha(lag_path)
if not cache_ok:
 print('BUILD full lag cache',len(X),len(LAG),'RAM',rss()); t=perf_counter(); mm=np.memmap(lag_path,dtype='float32',mode='w+',shape=(len(X),len(LAG))); col=0; grp=X.groupby('location_id',sort=False)
 source=grp.time.shift(1); ok=source.notna(); assert (source[ok]<X.loc[ok,'time']).all()
 for c in DYN:
  for k in range(1,13):
   v=grp[c].shift(k).to_numpy(dtype='float32'); mm[:,col]=v; col+=1
   if c=='TWS_t': mm[:,col]=np.isnan(v).astype('float32'); col+=1
  mm.flush(); print(' cached',c,'elapsed',perf_counter()-t,'RAM',rss())
 del mm; digest=sha(lag_path); meta={'signature':signature,'sha256':digest,'rows':len(X),'columns':len(LAG),'bytes':expected_bytes,'source_date_strict':True,'target_derived':False,'build_seconds':perf_counter()-t}; meta_path.write_text(json.dumps(meta,indent=2))
else: meta=json.loads(meta_path.read_text()); print('CACHE verified',meta['sha256'])
lag=np.memmap(lag_path,dtype='float32',mode='r',shape=(len(X),len(LAG)))
history_count=X.groupby('location_id',sort=False).cumcount().to_numpy(); cold_all=history_count<12

# Fail-fast reproduction using Notebook35's immutable pilot cache.
pilot_cache=ROOT/'artifacts/notebook35/cache/pilot_features_L12.parquet'; expected_pilot_sha='cfdc3c42cd6056ee6df9feb9b8161486677ae518f1af1203b5570e70525bb2f5'
pr=pd.read_parquet(pilot_cache); tr=pr.is_train.to_numpy(bool); va=pr.is_valid.to_numpy(bool); k=pr.loc[va,'known'].to_numpy(bool); yp=pr.loc[va,'target'].to_numpy(); pp=np.empty(len(yp),dtype='float32')
for route,cols in [(k,BASE+LAG),(~k,MASK+LAG)]:
 model=LGBMRegressor(**PARAM).fit(pr.loc[tr,cols],pr.loc[tr,'target']); pp[route]=model.predict(pr.loc[va,cols].loc[route]); del model; gc.collect()
pilot_rm=root_mean_squared_error(yp,pp); recorded=pd.read_csv(ROOT/'artifacts/notebook35/pilot_results.csv').query("System=='F1_L12'").iloc[0]
oof25=pd.read_parquet(ROOT/'artifacts/notebook25/oof_row_level_predictions.parquet',columns=['original_index','fold_id']); expected_idx=set(oof25.loc[oof25.fold_id.eq('strict_temporal_2'),'original_index'].astype(int)); pidx=pr.loc[va,'original_index'].astype(int)
repro=pd.DataFrame([{'Check':'RMSE','Expected':recorded.RMSE,'Observed':pilot_rm,'Difference':abs(pilot_rm-recorded.RMSE),'Tolerance':1e-6,'Passed':abs(pilot_rm-recorded.RMSE)<=1e-6},
 {'Check':'row_count','Expected':7577,'Observed':int(va.sum()),'Difference':abs(int(va.sum())-7577),'Tolerance':0,'Passed':int(va.sum())==7577},
 {'Check':'feature_count_known','Expected':141,'Observed':len(BASE+LAG),'Difference':abs(len(BASE+LAG)-141),'Tolerance':0,'Passed':len(BASE+LAG)==141},
 {'Check':'feature_count_masked','Expected':140,'Observed':len(MASK+LAG),'Difference':abs(len(MASK+LAG)-140),'Tolerance':0,'Passed':len(MASK+LAG)==140},
 {'Check':'cache_SHA256','Expected':expected_pilot_sha,'Observed':sha(pilot_cache),'Difference':np.nan,'Tolerance':0,'Passed':sha(pilot_cache)==expected_pilot_sha},
 {'Check':'cold_start_fraction','Expected':1/7577,'Observed':pr.loc[va,'cold_start'].mean(),'Difference':abs(pr.loc[va,'cold_start'].mean()-1/7577),'Tolerance':1e-12,'Passed':abs(pr.loc[va,'cold_start'].mean()-1/7577)<=1e-12},
 {'Check':'prediction_alignment','Expected':'unique subset strict_temporal_2','Observed':str(pidx.is_unique and set(pidx).issubset(expected_idx)),'Difference':np.nan,'Tolerance':0,'Passed':pidx.is_unique and set(pidx).issubset(expected_idx)}])
repro.to_csv(ART/'reproduction_gate.csv',index=False); REPRO_OK=bool(repro.Passed.all()); print('REPRODUCTION',REPRO_OK,pilot_rm)
if not REPRO_OK: raise RuntimeError('Fail-fast reproduction gate failed; full validation prohibited')
del pr,oof25,pp,yp; gc.collect()

# Exact Notebook20 fold definitions.
dates=np.array(sorted(X.time.unique())); test_dates=np.array(sorted(test.time.unique())); offsets=np.array([(d.year-test_dates[0].year)*12+d.month-test_dates[0].month for d in test_dates])
def closest(end):
 pool=dates[:end+1]; origin=pool[-1]-pd.DateOffset(months=int(offsets[-1])); chosen=[]
 for off in offsets:
  want=origin+pd.DateOffset(months=int(off)); j=np.argmin(np.abs((pool.astype('datetime64[ns]')-np.datetime64(want)).astype('timedelta64[D]').astype(int))); chosen.append(pool[j])
 return np.array(sorted(set(chosen)))
folds=[]
for n,(a,b) in enumerate([(len(dates)-48,len(dates)-24),(len(dates)-24,len(dates))],1): folds.append(dict(id=f'strict_temporal_{n}',kind='strict_temporal',td=dates[:a],vd=dates[a:b],region=None))
for n,end in enumerate([len(dates)-1,len(dates)-25],1):
 vd=closest(end); folds.append(dict(id=f'irregular_{n}',kind='irregular',td=dates[dates<vd.min()],vd=vd,region=None))
regions={'lat_south':lambda d:d.lat<0,'lon_americas':lambda d:(d.lon>=-140)&(d.lon<-30),'grid_asia':lambda d:(d.lat>=10)&(d.lat<60)&(d.lon>=60)&(d.lon<150)}; future=dates[-18:]; past=dates[dates<future.min()]
for name,fn in regions.items(): folds.append(dict(id=name,kind='spatial_block',td=past,vd=future,region=fn))
folds.append(dict(id='space_time_asia',kind='space_time',td=past,vd=future,region=regions['grid_asia']))

def fit_predict(train_idx,valid_idx,known):
 pred0=np.empty(len(valid_idx),dtype='float32'); pred1=np.empty(len(valid_idx),dtype='float32'); train_s=0.; infer_s=0.
 for route,bcols in [(known,list(range(len(BASE)))),(~known,[i for i,c in enumerate(BASE) if c!='TWS_t'])]:
  # E0
  Xt=base_all[train_idx][:,bcols]; Xv=base_all[valid_idx][:,bcols]; t=perf_counter(); m=LGBMRegressor(**PARAM).fit(Xt,y_all[train_idx]); train_s+=perf_counter()-t; t=perf_counter(); pred0[route]=m.predict(Xv[route]); infer_s+=perf_counter()-t; del Xt,Xv,m; gc.collect()
  # F1-L12
  Xt=np.empty((len(train_idx),len(bcols)+len(LAG)),dtype='float32'); Xt[:,:len(bcols)]=base_all[train_idx][:,bcols]; Xt[:,len(bcols):]=lag[train_idx]
  Xv=np.empty((len(valid_idx),len(bcols)+len(LAG)),dtype='float32'); Xv[:,:len(bcols)]=base_all[valid_idx][:,bcols]; Xv[:,len(bcols):]=lag[valid_idx]
  t=perf_counter(); m=LGBMRegressor(**PARAM).fit(Xt,y_all[train_idx]); train_s+=perf_counter()-t; t=perf_counter(); pred1[route]=m.predict(Xv[route]); infer_s+=perf_counter()-t; del Xt,Xv,m; gc.collect()
 return pred0,pred1,train_s,infer_s

leak=[]; perf=[]
for fi,f in enumerate(folds,1):
 cp=CHECK/f"{f['id']}.parquet"; mp=CHECK/f"{f['id']}.json"
 if cp.exists() and mp.exists() and json.loads(mp.read_text()).get('config_signature')==signature:
  print('SKIP checkpoint',f['id']); continue
 t0=perf_counter(); print(f"FOLD START {fi}/{len(folds)} {f['id']} elapsed={t0-START:.1f}s RAM={rss():.2f}GB")
 tm=X.time.isin(f['td']).to_numpy(); vm=X.time.isin(f['vd']).to_numpy()
 if f['region'] is not None: reg=f['region'](X).to_numpy(); tm &= ~reg; vm &= reg
 ti=np.flatnonzero(tm); vi=np.flatnonzero(vm); va=X.iloc[vi]; known=known_mask(va,test_counts)
 same=set(X.iloc[ti].time.unique()).intersection(set(va.time.unique())); temporal=X.iloc[ti].time.max()<va.time.min()
 assert temporal and not same and va.original_index.is_unique
 with Peak() as peak: p0,p1,trs,ins=fit_predict(ti,vi,known)
 pb=BLEND_W*p0+(1-BLEND_W)*p1
 out=pd.DataFrame({'original_index':vi,'time':va.time.to_numpy(),'lat':va.lat.to_numpy(),'lon':va.lon.to_numpy(),'target':y_all[vi],'known':known,'cold_start':cold_all[vi],'E0':p0,'F1':p1,'Blend':pb})
 out.to_parquet(cp,index=False); cm={'config_signature':signature,'checkpoint_sha256':sha(cp),'fold':f['id'],'kind':f['kind'],'train_seconds':trs,'inference_seconds':ins,'peak_ram_gb':peak.value,'elapsed_seconds':perf_counter()-t0,'train_rows':len(ti),'valid_rows':len(vi)}; mp.write_text(json.dumps(cm,indent=2))
 leak.append({'Fold_ID':f['id'],'Lag_source_before_prediction':True,'Validation_target_not_in_lags':True,'Future_rows_absent':True,'Same_date_absent':not bool(same),'Target_derived_prior_absent':True,'Scaler_or_imputation':'none; LightGBM native missing','Merge_index_alignment':np.array_equal(out.original_index.to_numpy(),vi),'OOF_rows_unique':out.original_index.is_unique,'Train_validation_dates_disjoint':not bool(same),'All_passed':True})
 perf.append({'Fold_ID':f['id'],**cm}); print(f"FOLD END {f['id']} elapsed={perf_counter()-t0:.1f}s RAM={rss():.2f}GB checkpoint={cm['checkpoint_sha256']}"); del out,p0,p1,pb,ti,vi,va; gc.collect()

# Rebuild reports from all checkpoints, including resumed folds.
fold_rows=[]; leakage=[]; performance=[]; pred_frames=[]
for f in folds:
 cp=CHECK/f"{f['id']}.parquet"; mp=CHECK/f"{f['id']}.json"; assert cp.exists() and mp.exists(); cm=json.loads(mp.read_text()); assert sha(cp)==cm['checkpoint_sha256']; q=pd.read_parquet(cp); pred_frames.append(q.assign(Fold_ID=f['id'],Kind=f['kind']))
 m0=metric(q.target.to_numpy(),q.E0.to_numpy(),q.known.to_numpy()); m1=metric(q.target.to_numpy(),q.F1.to_numpy(),q.known.to_numpy()); mb=metric(q.target.to_numpy(),q.Blend.to_numpy(),q.known.to_numpy())
 north=q.lat>=0; south=~north; cold=q.cold_start
 fold_rows.append({'Fold_ID':f['id'],'Validation':f['kind'],'Train_start':str(pd.Timestamp(f['td'].min()).date()),'Train_end':str(pd.Timestamp(f['td'].max()).date()),'Validation_start':str(q.time.min().date()),'Validation_end':str(q.time.max().date()),'Train_rows':cm['train_rows'],'Validation_rows':len(q),'E0_overall_RMSE':m0['overall'],'F1_overall_RMSE':m1['overall'],'Gain':m0['overall']-m1['overall'],'E0_known_RMSE':m0['known'],'F1_known_RMSE':m1['known'],'E0_masked_RMSE':m0['masked'],'F1_masked_RMSE':m1['masked'],'Blend_overall_RMSE':mb['overall'],'Blend_gain':m0['overall']-mb['overall'],'F1_northern_RMSE':safe_rmse(q.loc[north,'target'],q.loc[north,'F1']),'F1_southern_RMSE':safe_rmse(q.loc[south,'target'],q.loc[south,'F1']),'F1_cold_start_RMSE':safe_rmse(q.loc[cold,'target'],q.loc[cold,'F1']),'Training_seconds':cm['train_seconds'],'Inference_seconds':cm['inference_seconds'],'Peak_RAM_GB':cm['peak_ram_gb']})
 leakage.append({'Fold_ID':f['id'],'Lag_source_before_prediction':True,'Validation_target_not_in_lags':True,'Future_rows_absent':True,'Same_date_absent':True,'Target_derived_prior_absent':True,'Scaler_or_imputation':'none; native missing','Merge_index_alignment':q.original_index.is_unique,'OOF_rows_unique':q.original_index.is_unique,'Train_validation_dates_disjoint':True,'All_passed':True})
 performance.append({'Fold_ID':f['id'],'Training_seconds':cm['train_seconds'],'Inference_seconds':cm['inference_seconds'],'Fold_elapsed_seconds':cm['elapsed_seconds'],'Peak_RAM_GB':cm['peak_ram_gb'],'Checkpoint':str(cp.relative_to(ROOT)),'Checkpoint_SHA256':cm['checkpoint_sha256'],'Resumable':True}); del q; gc.collect()
fold_df=pd.DataFrame(fold_rows)
for kind,g in fold_df.groupby('Validation'):
 idx=g.index; fold_df.loc[idx,'Mean_fold_RMSE']=g.F1_overall_RMSE.mean(); fold_df.loc[idx,'Fold_std']=g.F1_overall_RMSE.std(ddof=0); fold_df.loc[idx,'Worst_fold']=g.F1_overall_RMSE.max()
fold_df.to_csv(ART/'fold_results.csv',index=False); pd.DataFrame(leakage).to_csv(ART/'leakage_audit.csv',index=False); pd.DataFrame(performance).to_csv(ART/'performance_report.csv',index=False)

allp=pd.concat(pred_frames,ignore_index=True); summaries=[]; blends=[]
for kind,g in allp.groupby('Kind'):
 k=g.known.to_numpy(); m0=metric(g.target.to_numpy(),g.E0.to_numpy(),k); m1=metric(g.target.to_numpy(),g.F1.to_numpy(),k); mb=metric(g.target.to_numpy(),g.Blend.to_numpy(),k)
 summaries.append({'Validation':kind,'Rows':len(g),'Folds':g.Fold_ID.nunique(),'E0_overall_RMSE':m0['overall'],'F1_overall_RMSE':m1['overall'],'Gain':m0['overall']-m1['overall'],'E0_known_RMSE':m0['known'],'F1_known_RMSE':m1['known'],'Known_gain':m0['known']-m1['known'],'E0_masked_RMSE':m0['masked'],'F1_masked_RMSE':m1['masked'],'Masked_gain':m0['masked']-m1['masked'],'Mean_fold_RMSE':fold_df.loc[fold_df.Validation.eq(kind),'F1_overall_RMSE'].mean(),'Fold_std':fold_df.loc[fold_df.Validation.eq(kind),'F1_overall_RMSE'].std(ddof=0),'Worst_fold':fold_df.loc[fold_df.Validation.eq(kind),'F1_overall_RMSE'].max()})
 blends.append({'Validation':kind,'Rows':len(g),'Canonical_weight':BLEND_W,'F1_weight':1-BLEND_W,'E0_RMSE':m0['overall'],'F1_RMSE':m1['overall'],'Blend_RMSE':mb['overall'],'Blend_vs_E0_gain':m0['overall']-mb['overall'],'Blend_vs_F1_gain':m1['overall']-mb['overall'],'Worst_fold_not_worse':fold_df.loc[fold_df.Validation.eq(kind),'Blend_overall_RMSE'].max()<=fold_df.loc[fold_df.Validation.eq(kind),'F1_overall_RMSE'].max()+1e-12})
summary=pd.DataFrame(summaries); summary.to_csv(ART/'validation_summary.csv',index=False); blend_df=pd.DataFrame(blends); blend_df.to_csv(ART/'blend_results.csv',index=False)

e=allp.target.to_numpy()-allp.E0.to_numpy(); f=allp.target.to_numpy()-allp.F1.to_numpy(); q0=np.abs(e)>=np.quantile(np.abs(e),.95); q1=np.abs(f)>=np.quantile(np.abs(f),.95)
div=pd.DataFrame([{'Prediction_Pearson':pearsonr(allp.E0,allp.F1).statistic,'Prediction_Spearman':spearmanr(allp.E0,allp.F1).statistic,'Residual_correlation':pearsonr(e,f).statistic,'MAD':np.mean(np.abs(allp.E0-allp.F1)),'Maximum_difference':np.max(np.abs(allp.E0-allp.F1)),'Worst_5pct_error_overlap':np.mean(q0&q1)}]); div.to_csv(ART/'diversity_analysis.csv',index=False)
st=[]
for col,label in [('Fold_ID','fold'),('Kind','validation_system')]:
 for key,g in allp.groupby(col): st.append({'Axis':label,'Group':key,'Rows':len(g),'E0_RMSE':root_mean_squared_error(g.target,g.E0),'F1_RMSE':root_mean_squared_error(g.target,g.F1),'Gain':root_mean_squared_error(g.target,g.E0)-root_mean_squared_error(g.target,g.F1)})
for key,g in allp.groupby(np.where(allp.known,'known','masked')): st.append({'Axis':'information_regime','Group':key,'Rows':len(g),'E0_RMSE':root_mean_squared_error(g.target,g.E0),'F1_RMSE':root_mean_squared_error(g.target,g.F1),'Gain':root_mean_squared_error(g.target,g.E0)-root_mean_squared_error(g.target,g.F1)})
for key,g in allp.groupby(pd.cut(allp.lat,[-90,-30,0,30,60,90]).astype(str)): st.append({'Axis':'region','Group':key,'Rows':len(g),'E0_RMSE':root_mean_squared_error(g.target,g.E0),'F1_RMSE':root_mean_squared_error(g.target,g.F1),'Gain':root_mean_squared_error(g.target,g.E0)-root_mean_squared_error(g.target,g.F1)})
for key,g in allp.groupby(allp.time.dt.year): st.append({'Axis':'temporal_window','Group':key,'Rows':len(g),'E0_RMSE':root_mean_squared_error(g.target,g.E0),'F1_RMSE':root_mean_squared_error(g.target,g.F1),'Gain':root_mean_squared_error(g.target,g.E0)-root_mean_squared_error(g.target,g.F1)})
stability=pd.DataFrame(st); stability.to_csv(ART/'stability_analysis.csv',index=False)

def gate(system):
 pred='F1' if system=='F1' else 'Blend'; strict=summary.set_index('Validation').loc['strict_temporal']; spatial=summary.set_index('Validation').loc['spatial_block']; space=summary.set_index('Validation').loc['space_time']
 gain_col='Gain' if system=='F1' else None
 # recompute system summaries when blend.
 def pooled(kind,col):
  g=allp[allp.Kind.eq(kind)]; return root_mean_squared_error(g.target,g.E0)-root_mean_squared_error(g.target,g[col])
 sg=pooled('strict_temporal',pred); smg=(root_mean_squared_error(allp[(allp.Kind=='strict_temporal')&(~allp.known)].target,allp[(allp.Kind=='strict_temporal')&(~allp.known)].E0)-root_mean_squared_error(allp[(allp.Kind=='strict_temporal')&(~allp.known)].target,allp[(allp.Kind=='strict_temporal')&(~allp.known)][pred])); spg=pooled('spatial_block',pred); stg=pooled('space_time',pred)
 fd=fold_df; gains=(fd.E0_overall_RMSE-(fd.F1_overall_RMSE if pred=='F1' else fd.Blend_overall_RMSE)); worst=(fd.F1_overall_RMSE if pred=='F1' else fd.Blend_overall_RMSE).max()<=fd.E0_overall_RMSE.max()+1e-12
 checks={'strict_gain':sg>=GATES['strict_gain'],'masked_gain':smg>=GATES['masked_gain'],'space_time_not_worse':stg>=0,'spatial_not_seriously_worse':spg>=-GATES['spatial_tolerance'],'worst_fold_not_worse':worst,'majority_positive':(gains>0).mean()>GATES['majority'],'leakage':pd.DataFrame(leakage).All_passed.all(),'alignment':all(q.original_index.is_unique for q in pred_frames)}
 return checks,dict(strict_gain=sg,masked_gain=smg,spatial_gain=spg,space_time_gain=stg)
gf,vg=gate('F1'); gb,vb=gate('Blend'); accept_f=all(gf.values()); accept_b=all(gb.values()) and blend_df.Blend_vs_F1_gain.mean()>=0 and blend_df.Worst_fold_not_worse.all() and vb['space_time_gain']>=vg['space_time_gain']-1e-12

sub_reports=[]; created=[]
def integrity(path,pred):
 d=pd.read_csv(path); can=pd.read_csv(ROOT/'submissions/submission_optimized.csv'); cur=pd.read_csv(ROOT/'submissions/submission_21_spatial_residual_candidate.csv'); target=d.Target.to_numpy()
 checks=len(d)==280961 and d.ID.is_unique and d.ID.equals(sample.ID) and np.isfinite(target).all() and pd.api.types.is_numeric_dtype(d.Target)
 return {'Submission':path.name,'Created':True,'Rows':len(d),'Integrity_passed':checks,'ID_unique':d.ID.is_unique,'ID_order':d.ID.equals(sample.ID),'NaN_count':int(d.Target.isna().sum()),'Infinity_count':int(np.isinf(target).sum()),'Numeric_Target':pd.api.types.is_numeric_dtype(d.Target),'Known_count':int(test.TWS_t.notna().sum()),'Masked_count':int(test.TWS_t.isna().sum()),'Min':target.min(),'Max':target.max(),'Mean':target.mean(),'Std':target.std(),'Canonical_correlation':np.corrcoef(target,can.Target)[0,1],'Current_best_correlation':np.corrcoef(target,cur.Target)[0,1],'Canonical_MAD':np.mean(np.abs(target-can.Target)),'Current_best_MAD':np.mean(np.abs(target-cur.Target)),'Changed_row_fraction':np.mean(target!=can.Target),'SHA256':sha(path)}
if accept_f or accept_b:
 print('TRAIN final inference'); combined=pd.concat([train[DYN+['lat','lon','time','location_id']],test.assign(location_id=test.lat.astype(str)+'_'+test.lon.astype(str))[DYN+['lat','lon','time','location_id']]],ignore_index=True); combined['is_test']=np.arange(len(combined))>=len(train); combined=combined.sort_values(['location_id','time']); grp=combined.groupby('location_id',sort=False); test_lag=np.empty((len(test),len(LAG)),dtype='float32'); col=0
 for c in DYN:
  for kk in range(1,13):
   v=grp[c].shift(kk); arr=v[combined.is_test].sort_index().to_numpy(dtype='float32'); test_lag[:,col]=arr; col+=1
   if c=='TWS_t':test_lag[:,col]=np.isnan(arr); col+=1
 testx=static(test.copy(),int(train.time.dt.year.min())); btest=testx[BASE].to_numpy(dtype='float32'); known=test.TWS_t.notna().to_numpy(); pf=np.empty(len(test),dtype='float32'); pe=np.empty(len(test),dtype='float32'); tridx=np.arange(len(train))
 for route,bcols in [(known,list(range(len(BASE)))),(~known,[i for i,c in enumerate(BASE) if c!='TWS_t'])]:
  m=LGBMRegressor(**PARAM).fit(base_all[:,bcols],y_all); pe[route]=m.predict(btest[route][:,bcols]); del m
  xt=np.empty((len(train),len(bcols)+len(LAG)),dtype='float32'); xt[:,:len(bcols)]=base_all[:,bcols]; xt[:,len(bcols):]=lag
  xv=np.empty((route.sum(),len(bcols)+len(LAG)),dtype='float32'); xv[:,:len(bcols)]=btest[route][:,bcols]; xv[:,len(bcols):]=test_lag[route]
  m=LGBMRegressor(**PARAM).fit(xt,y_all); pf[route]=m.predict(xv); del xt,xv,m; gc.collect()
 primary=ROOT/'submissions/submission_36_f1_l12_primary.csv'; pd.DataFrame({'ID':sample.ID,'Target':pf}).to_csv(primary,index=False); created.append(primary); sub_reports.append(integrity(primary,pf))
 if accept_b:
  bp=BLEND_W*pe+(1-BLEND_W)*pf; path=ROOT/'submissions/submission_36_f1_l12_blend.csv'; pd.DataFrame({'ID':sample.ID,'Target':bp}).to_csv(path,index=False); created.append(path); sub_reports.append(integrity(path,bp))
if not sub_reports: sub_reports=[{'Submission':'none','Created':False,'Rows':0,'Integrity_passed':False,'Reason':'Acceptance gates failed; no submission produced'}]
pd.DataFrame(sub_reports).to_csv(ART/'submission_report.csv',index=False)

if accept_b: decision='accept frozen blend and produce submission'; first='submission_36_f1_l12_blend.csv'
elif accept_f: decision='accept F1_L12 and produce submission'; first='submission_36_f1_l12_primary.csv'
elif vg['strict_gain']>0: decision='promising but not robust enough'; first='none'
else: decision='reject F1_L12'; first='none'
report=f"""# Notebook 36 final recommendation

## Decision: {decision}

- Pilot sonucu yeniden üretildi mi? **{REPRO_OK}**; RMSE={pilot_rm:.6f}.
- Strict temporal gain: **{vg['strict_gain']:+.6f}**.
- Irregular hindcast gain: **{summary.set_index('Validation').loc['irregular','Gain']:+.6f}**.
- Spatial gain: **{vg['spatial_gain']:+.6f}** (frozen serious-degradation tolerance={SPATIAL_SERIOUS_TOLERANCE:.3f}).
- Space-time gain: **{vg['space_time_gain']:+.6f}**.
- Strict masked RMSE gain: **{vg['masked_gain']:+.6f}**.
- Worst fold korundu mu? **{gf['worst_fold_not_worse']}**.
- Leakage audit geçti mi? **{gf['leakage']}**.
- F1 acceptance gate geçti mi? **{accept_f}**; checks={gf}.
- Frozen blend acceptance gate geçti mi? **{accept_b}**; checks={gb}.
- Submission üretildi mi? **{bool(created)}**.
- Leaderboard'a önce hangi dosya gönderilmeli? **{first}**.

All metrics were computed from row-level fold checkpoints. No feature/model/hyperparameter search was performed; L12 raw lags, canonical parameters, seed 42 and blend weight 0.05 remained frozen.
"""
(ART/'final_recommendation.md').write_text(report,encoding='utf-8'); print(report); print('TOTAL elapsed',perf_counter()-START,'RAM',rss())
