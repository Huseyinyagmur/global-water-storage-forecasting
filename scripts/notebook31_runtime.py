"""Extended real ERA5-Land pilot: preparation, strict OOF validation, decision."""
from pathlib import Path
from time import perf_counter
import hashlib, json, warnings
import numpy as np, pandas as pd, xarray as xr
from lightgbm import LGBMRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import root_mean_squared_error
warnings.filterwarnings('ignore'); T0=perf_counter(); SEED=42
ROOT=Path.cwd().resolve(); ROOT=ROOT.parent if ROOT.name=='notebooks' else ROOT
ART=ROOT/'artifacts/notebook31'; ART.mkdir(parents=True,exist_ok=True)
PILOT_RAW=ROOT/'data/external/era5_land/pilot/raw'; EXT_RAW=ROOT/'data/external/era5_land/extended_pilot/raw'; OUT=ROOT/'data/external/era5_land/extended_pilot/processed/era5_land_extended_features.parquet'; OUT.parent.mkdir(parents=True,exist_ok=True)
REGIONS={'north':(35,50,-10,5),'equator':(-7.5,7.5,20,35),'south':(-35,-20,120,135)}; RAW=['tp','e','sro','ssro','swvl1','swvl3']; DER=['water_balance_proxy','total_runoff','soil_profile_gradient']
def stamp(x): print(f'[{x}] elapsed={perf_counter()-T0:.2f}s')
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

# Frozen acceptance thresholds, declared before results.
TH={'overall_gain':.0015,'masked_gain':.003,'known_deterioration_tolerance':.001,'overall_nonworse_tolerance':0.0}
manifest=pd.read_csv(ART/'download_extension_report.csv'); files=[]
for r in manifest.itertuples():
 p=ROOT/Path(r.target_path); files.append(p)
assert len(files)==27 and all(p.exists() and p.stat().st_size>0 for p in files)

# Validate and combine 27 real files.
nc_rows=[]; ds_by_region={}
for region in REGIONS:
 region_files=sorted(p for p in files if p.name.startswith(region+'_')); dsets=[]
 for p in region_files:
  year=int(p.stem[-4:]); ds=xr.open_dataset(p); dsets.append(ds); tm=pd.to_datetime(ds.valid_time.values)
  okvars=set(RAW)<=set(ds.data_vars)
  for v in RAW:
   a=np.asarray(ds[v].values); finite=a[np.isfinite(a)]; mn=float(finite.min()); mx=float(finite.max()); physical=(mn>=-1e-8 if v in ['tp','sro','ssro'] else -1e-8<=mn and mx<=1+1e-8 if v in ['swvl1','swvl3'] else True)
   nc_rows.append({'file':p.name,'region':region,'year':year,'size_bytes':p.stat().st_size,'sha256':sha(p),'variable':v,'unit':ds[v].attrs.get('units',''),'time_min':tm.min(),'time_max':tm.max(),'timestamp_count':len(tm),'missing_rate':float(np.isnan(a).mean()),'minimum':mn,'maximum':mx,'infinity_count':int(np.isinf(a).sum()),'physical_passed':physical,'valid':okvars and len(tm)==12 and tm.is_unique and not np.isinf(a).any() and physical})
 ds_by_region[region]=xr.concat(dsets,dim='valid_time').sortby('valid_time')
netcdf=pd.DataFrame(nc_rows); netcdf.to_csv(ART/'netcdf_validation.csv',index=False); assert netcdf.valid.all()

# Feature definitions, including verified evaporation convention.
evap_negative=(netcdf.loc[netcdf.variable.eq('e'),'maximum']<=0).mean()>.8 or netcdf.loc[netcdf.variable.eq('e'),'minimum'].median()<0
defs=pd.DataFrame([
['tp','raw','m/day monthly mean','total precipitation','identity','included'],['e','raw','m water equivalent/day','ERA5 evaporation; negative denotes upward loss','identity','included'],['sro','raw','m/day','surface runoff','identity','included'],['ssro','raw','m/day','sub-surface runoff','identity','included'],['swvl1','raw','m3/m3','0–7 cm soil water','identity','included'],['swvl3','raw','m3/m3','28–100 cm soil water','identity','included'],
['water_balance_proxy','derived','m/day','precipitation minus actual evaporation magnitude','tp + e' if evap_negative else 'tp - e','included'],['total_runoff','derived','m/day','combined runoff','sro + ssro','included'],['soil_profile_gradient','derived','m3/m3','deep minus surface soil water','swvl3 - swvl1','included']],columns=['feature','family','unit','physical_meaning','formula','decision'])
defs['cutoff']='strict t-1'; defs.to_csv(ART/'feature_definition.csv',index=False)

# Build real nearest-aligned train+test feature parquet. No future fill/backfill.
train=pd.read_csv(ROOT/'data/raw/Train.csv',parse_dates=['time']); test=pd.read_csv(ROOT/'data/raw/Test.csv',parse_dates=['time'])
frames=[]
for split,d,idcol in [('train',train,'sample_id'),('test',test,'ID')]:
 base=d[[idcol,'time','lat','lon']].rename(columns={idcol:'row_id','time':'date'}).copy(); base['region']=None
 for region,(s,n,w,e) in REGIONS.items(): base.loc[base.lat.between(s,n)&base.lon.between(w,e),'region']=region
 base=base[base.region.notna()].copy(); base['prediction_origin']=base.date.dt.to_period('M'); base['source_period']=base.prediction_origin-1; base=base[base.source_period.between(pd.Period('2010-01'),pd.Period('2018-12'))]
 parts=[]
 for region,g in base.groupby('region',sort=False):
  ds=ds_by_region[region]
  for month,h in g.groupby('source_period',sort=False):
   src=ds.sel(valid_time=np.datetime64(month.start_time)); q=src.interp(latitude=xr.DataArray(h.lat.to_numpy(),dims='row'),longitude=xr.DataArray(h.lon.to_numpy(),dims='row'),method='nearest'); z=q[RAW].to_dataframe().reset_index(drop=True); z['row_id']=h.row_id.to_numpy(); z['dataset']=split; z['date']=h.date.to_numpy(); z['lat']=h.lat.to_numpy(); z['lon']=h.lon.to_numpy(); z['region']=region; z['prediction_origin']=h.prediction_origin.astype(str).to_numpy(); z['era5_source_month']=str(month); z['era5_lag_months']=1; z['cutoff_safe_flag']=True; z['interpolation_method']='nearest'; parts.append(z)
 aligned=pd.concat(parts,ignore_index=True); frames.append(aligned)
features=pd.concat(frames,ignore_index=True); features['water_balance_proxy']=features.tp+features.e if evap_negative else features.tp-features.e; features['total_runoff']=features.sro+features.ssro; features['soil_profile_gradient']=features.swvl3-features.swvl1
keep=['row_id','dataset','date','lat','lon','region','prediction_origin','era5_source_month','era5_lag_months','cutoff_safe_flag','interpolation_method',*RAW,*DER]
features[keep].to_parquet(OUT,index=False); assert not features.duplicated(['dataset','row_id']).any()

orig=pd.PeriodIndex(features.prediction_origin,freq='M'); src=pd.PeriodIndex(features.era5_source_month,freq='M'); safe=src<orig
cutoff=pd.DataFrame([{'rows':len(features),'safe_rows':int(safe.sum()),'safe_fraction':safe.mean(),'future_rows':int((~safe).sum()),'minimum_lag_months':int(min((o-s).n for o,s in zip(orig,src))),'all_safe':bool(safe.all()),'policy':'strict t-1'}]); cutoff.to_csv(ART/'cutoff_audit.csv',index=False); assert safe.all()
merge=pd.DataFrame([{'processed_rows':len(features),'train_rows':int(features.dataset.eq('train').sum()),'test_rows':int(features.dataset.eq('test').sum()),'unique_rows':int((~features.duplicated(['dataset','row_id'])).sum()),'duplicates':int(features.duplicated(['dataset','row_id']).sum()),'missing_rate_raw':features[RAW].isna().mean().mean(),'missing_rate_derived':features[DER].isna().mean().mean(),'regions':features.region.nunique(),'date_min':features.date.min(),'date_max':features.date.max(),'source_months':pd.PeriodIndex(features.era5_source_month,freq='M').nunique(),'cutoff_safe':bool(safe.all()),'physical_consistency':bool(netcdf.physical_passed.all())}]); merge.to_csv(ART/'merge_report.csv',index=False); stamp('feature parquet prepared')

# Pilot train merge and Notebook11 canonical-equivalent model.
pf=features[features.dataset.eq('train')].copy(); pilot=train.merge(pf.drop(columns=['dataset','date','lat','lon']),left_on='sample_id',right_on='row_id',how='inner',validate='one_to_one')
SPEI=['SPEI_01_t','SPEI_03_t','SPEI_06_t','SPEI_12_t']
def static(df):
 o=df.copy(); dt=o.time; o['month']=dt.dt.month.astype('int8'); o['quarter']=dt.dt.quarter.astype('int8'); o['year']=dt.dt.year.astype('int16'); o['elapsed_month']=((dt.dt.year-train.time.dt.year.min())*12+dt.dt.month-1).astype('int16'); o['month_sin_calc']=np.sin(2*np.pi*o.month/12); o['month_cos_calc']=np.cos(2*np.pi*o.month/12); o['season_sin']=np.sin(2*np.pi*o.quarter/4); o['season_cos']=np.cos(2*np.pi*o.quarter/4); s=o[SPEI]; o['spei_mean']=s.mean(1); o['spei_min']=s.min(1); o['spei_max']=s.max(1); o['spei_std']=s.std(1); o['spei_01_minus_12']=o.SPEI_01_t-o.SPEI_12_t; o['spei_03_minus_12']=o.SPEI_03_t-o.SPEI_12_t; o['spei_short_long_ratio']=o.SPEI_01_t/(o.SPEI_12_t.abs()+.1); o['lat_band_5']=np.floor((o.lat+90)/5); o['lon_band_5']=np.floor((o.lon+180)/5); o['spatial_bin_10']=o.lat_band_5*100+o.lon_band_5; o['lat_lon_interaction']=o.lat*o.lon/1000; o['lat_sin']=np.sin(np.radians(o.lat)); o['lat_cos']=np.cos(np.radians(o.lat)); o['lon_sin']=np.sin(np.radians(o.lon)); o['lon_cos']=np.cos(np.radians(o.lon)); return o
X=static(pilot); BASE=['lat','lon','TWS_t',*SPEI,'SOIL_MOISTURE_t','month','month_sin','month_cos','year','quarter','elapsed_month','month_sin_calc','month_cos_calc','season_sin','season_cos','spei_mean','spei_min','spei_max','spei_std','spei_01_minus_12','spei_03_minus_12','spei_short_long_ratio','lat_band_5','lon_band_5','spatial_bin_10','lat_lon_interaction','lat_sin','lat_cos','lon_sin','lon_cos']; MASK=[c for c in BASE if c!='TWS_t']; PRI=['prior_location_mean','prior_location_std','prior_location_month','prior_global_month']; PARAM=dict(n_estimators=160,learning_rate=.08,num_leaves=63,min_child_samples=100,reg_lambda=1,subsample=.85,colsample_bytree=.9,random_state=SEED,n_jobs=4,verbosity=-1)
def mdl(): return LGBMRegressor(**PARAM)
def train_prior(df):
 o=df.sort_values(['lat','lon','time']).copy(); sh=o.groupby(['lat','lon']).target.shift(1); o['prior_location_mean']=sh.groupby([o.lat,o.lon]).expanding().mean().reset_index(level=[0,1],drop=True); o['prior_location_std']=sh.groupby([o.lat,o.lon]).expanding().std().reset_index(level=[0,1],drop=True); o['prior_location_month']=sh.groupby([o.lat,o.lon,o.month]).expanding().mean().reset_index(level=[0,1,2],drop=True); q=o.groupby(['month','time']).target.agg(['sum','count']).reset_index().sort_values(['month','time']); q['prior_global_month']=(q.groupby('month')['sum'].cumsum()-q['sum'])/(q.groupby('month')['count'].cumsum()-q['count']).replace(0,np.nan); return o.merge(q[['month','time','prior_global_month']],on=['month','time'],how='left',sort=False).sort_index()
def apply_prior(h,a):
 lp=h.groupby(['lat','lon']).target.agg(['mean','std']).rename(columns={'mean':'prior_location_mean','std':'prior_location_std'}); lm=h.groupby(['lat','lon','month']).target.agg(['sum','count']).rename(columns={'sum':'s','count':'n'}); gm=h.groupby('month').target.mean().rename('prior_global_month'); o=a.join(lp,on=['lat','lon']).join(lm,on=['lat','lon','month']).join(gm,on='month'); o['prior_location_month']=(o.s+4*o.prior_location_mean)/(o.n.fillna(0)+4); return o.drop(columns=['s','n'])
known_pos=set(np.flatnonzero(test.groupby('time').TWS_t.count().to_numpy()>1000)); dates=np.array(sorted(X.time.unique())); val_chunks=np.array_split(dates[len(dates)//2:],3)
sets={'E0':[],'E1':RAW,'E2':RAW+DER}; rows=[]; regrows=[]; improws=[]
for fold,vd in enumerate(val_chunks,1):
 tr=X.time<vd.min(); va=X.time.isin(vd); assert X.loc[tr].time.max()<X.loc[va].time.min(); known=X.loc[va].time.isin({d for j,d in enumerate(vd) if j in known_pos}).to_numpy(); tp=train_prior(X.loc[tr]); vp=apply_prior(X.loc[tr],X.loc[va])
 for system,extra in sets.items():
  t=perf_counter(); bf=BASE+extra; mf=MASK+extra; km=mdl().fit(X.loc[tr,bf],X.loc[tr].target); bm=mdl().fit(X.loc[tr,mf],X.loc[tr].target); hm=mdl().fit(tp[mf+PRI],tp.target); y=X.loc[va].target.to_numpy(); pred=np.empty(len(y)); pred[known]=km.predict(X.loc[va,bf].iloc[np.flatnonzero(known)]); pred[~known]=.65*bm.predict(X.loc[va,mf].iloc[np.flatnonzero(~known)])+.35*hm.predict(vp[mf+PRI].iloc[np.flatnonzero(~known)]); rm=root_mean_squared_error(y,pred)
  rows.append({'system':system,'fold':fold,'train_rows':int(tr.sum()),'validation_rows':int(va.sum()),'validation_dates':len(vd),'overall_RMSE':rm,'known_RMSE':root_mean_squared_error(y[known],pred[known]),'masked_RMSE':root_mean_squared_error(y[~known],pred[~known]),'runtime_seconds':perf_counter()-t,'feature_count':len(bf)})
  for region in REGIONS:
   q=X.loc[va].region.eq(region).to_numpy(); regrows.append({'system':system,'fold':fold,'region':region,'rows':int(q.sum()),'RMSE':root_mean_squared_error(y[q],pred[q])})
  if system in ['E1','E2']:
   rng=np.random.default_rng(SEED+fold)
   for expert,m,features_used,positions in [('known',km,bf,np.flatnonzero(known)),('masked',bm,mf,np.flatnonzero(~known))]:
    positions=positions[:5000]; sample=X.loc[va,features_used].iloc[positions].copy(); sy=X.loc[va].target.to_numpy()[positions]; base_direct=root_mean_squared_error(sy,m.predict(sample)); perm={}
    for feature in extra:
     deltas=[]
     for _ in range(2):
      q=sample.copy(); q[feature]=rng.permutation(q[feature].to_numpy()); deltas.append(root_mean_squared_error(sy,m.predict(q))-base_direct)
     perm[feature]=float(np.mean(deltas))
    improws.extend({'system':system,'fold':fold,'expert':expert,'feature':n,'gain_importance':g,'permutation_importance':perm.get(n,np.nan),'is_ERA5':n in RAW+DER,'family':'derived' if n in DER else 'raw' if n in RAW else 'canonical'} for n,g in zip(m.booster_.feature_name(),m.booster_.feature_importance('gain')))
validation=pd.DataFrame(rows); regional=pd.DataFrame(regrows); importance=pd.DataFrame(improws)

# E3 uses the three fold-stable highest-gain ERA features from OOF models; no test/LB information.
era_rank=importance[importance.is_ERA5].groupby('feature').agg(mean_gain=('gain_importance','mean'),positive_folds=('gain_importance',lambda x:(x>0).mean())).sort_values(['positive_folds','mean_gain'],ascending=False); subset=era_rank.head(3).index.tolist(); sets['E3']=subset
for fold,vd in enumerate(val_chunks,1):
 tr=X.time<vd.min(); va=X.time.isin(vd); known=X.loc[va].time.isin({d for j,d in enumerate(vd) if j in known_pos}).to_numpy(); tp=train_prior(X.loc[tr]); vp=apply_prior(X.loc[tr],X.loc[va]); extra=subset; bf=BASE+extra; mf=MASK+extra; t=perf_counter(); km=mdl().fit(X.loc[tr,bf],X.loc[tr].target); bm=mdl().fit(X.loc[tr,mf],X.loc[tr].target); hm=mdl().fit(tp[mf+PRI],tp.target); y=X.loc[va].target.to_numpy(); pred=np.empty(len(y)); pred[known]=km.predict(X.loc[va,bf].iloc[np.flatnonzero(known)]); pred[~known]=.65*bm.predict(X.loc[va,mf].iloc[np.flatnonzero(~known)])+.35*hm.predict(vp[mf+PRI].iloc[np.flatnonzero(~known)]); rows.append({'system':'E3','fold':fold,'train_rows':int(tr.sum()),'validation_rows':int(va.sum()),'validation_dates':len(vd),'overall_RMSE':root_mean_squared_error(y,pred),'known_RMSE':root_mean_squared_error(y[known],pred[known]),'masked_RMSE':root_mean_squared_error(y[~known],pred[~known]),'runtime_seconds':perf_counter()-t,'feature_count':len(bf)}); [regrows.append({'system':'E3','fold':fold,'region':r,'rows':int((q:=X.loc[va].region.eq(r).to_numpy()).sum()),'RMSE':root_mean_squared_error(y[q],pred[q])}) for r in REGIONS]
validation=pd.DataFrame(rows); regional=pd.DataFrame(regrows)
summary=validation.groupby('system').agg(Overall_RMSE=('overall_RMSE','mean'),Known_RMSE=('known_RMSE','mean'),Masked_RMSE=('masked_RMSE','mean'),Fold_std=('overall_RMSE','std'),Worst_fold=('overall_RMSE','max'),Runtime=('runtime_seconds','sum'),Feature_count=('feature_count','max')).reset_index(); base=summary[summary.system.eq('E0')].iloc[0]; summary['Overall_gain']=base.Overall_RMSE-summary.Overall_RMSE; summary['Known_gain']=base.Known_RMSE-summary.Known_RMSE; summary['Masked_gain']=base.Masked_RMSE-summary.Masked_RMSE; summary['Worst_fold_gain']=base.Worst_fold-summary.Worst_fold
validation=validation.merge(summary[['system','Overall_RMSE','Known_RMSE','Masked_RMSE','Fold_std','Worst_fold','Overall_gain','Known_gain','Masked_gain','Worst_fold_gain']],on='system',suffixes=('','_mean')); validation['E3_subset']='|'.join(subset); validation.to_csv(ART/'validation_results.csv',index=False)
regmean=regional.groupby(['system','region']).agg(RMSE=('RMSE','mean'),Fold_std=('RMSE','std')).reset_index(); breg=regmean[regmean.system.eq('E0')][['region','RMSE']].rename(columns={'RMSE':'E0_RMSE'}); regmean=regmean.merge(breg,on='region'); regmean['gain_vs_E0']=regmean.E0_RMSE-regmean.RMSE; regmean.to_csv(ART/'regional_results.csv',index=False)
importance['importance_share']=importance.groupby(['system','fold','expert']).gain_importance.transform(lambda x:x/x.sum() if x.sum() else 0); importance['positive_fold_share']=importance.groupby(['system','expert','feature']).gain_importance.transform(lambda x:(x>0).mean()); importance['E3_selected']=importance.feature.isin(subset); importance['note']='Gain plus two-repeat permutation RMSE increase on up to 5,000 OOF rows per expert/fold'; importance.to_csv(ART/'feature_importance.csv',index=False)

# Frozen gates use E1/E2 only.
candidates=summary[summary.system.isin(['E1','E2'])].copy(); region_pos=regmean[regmean.system.isin(['E1','E2'])].groupby('system').gain_vs_E0.apply(lambda x:int((x>0).sum()))
decisions=[]
for r in candidates.itertuples():
 A=bool(r.Overall_gain>=TH['overall_gain'] and r.Worst_fold_gain>=0); B=bool(r.Masked_gain>=TH['masked_gain'] and r.Known_gain>=-TH['known_deterioration_tolerance']); C=bool(region_pos[r.system]>=2 and r.Overall_gain>=TH['overall_nonworse_tolerance']); decisions.append((r.system,A,B,C,A or B or C))
passed=[x for x in decisions if x[-1]]
near=(candidates.Overall_gain.max()>0 or candidates.Masked_gain.max()>.001 or max(region_pos)>=2)
decision='proceed to full global download' if passed else 'expand pilot once more' if near else 'stop ERA5 research'
best=summary[summary.system.isin(['E1','E2','E3'])].sort_values('Overall_RMSE').iloc[0]; raw_imp=era_rank.loc[era_rank.index.intersection(RAW)].sort_values('mean_gain',ascending=False); der_imp=era_rank.loc[era_rank.index.intersection(DER)].sort_values('mean_gain',ascending=False)
expert_rank=importance[importance.is_ERA5].groupby(['expert','feature']).agg(mean_share=('importance_share','mean'),mean_permutation=('permutation_importance','mean'),positive_fold_share=('gain_importance',lambda x:(x>0).mean())).reset_index(); known_top=expert_rank[expert_rank.expert.eq('known')].sort_values(['mean_permutation','mean_share'],ascending=False).iloc[0].feature; masked_top=expert_rank[expert_rank.expert.eq('masked')].sort_values(['mean_permutation','mean_share'],ascending=False).iloc[0].feature
report=f'''# Notebook 31 final recommendation\n\n**Decision: {decision}**\n\n- Cached/downloaded files: 27 total; 9 reused, 18 newly downloaded; {sum(p.stat().st_size for p in files)/1024**2:.2f} MB.\n- NetCDF valid: {bool(netcdf.valid.all())}; strict cutoff safe: {cutoff.safe_fraction.iloc[0]:.3%}.\n- Processed rows: {len(features):,}; raw missing rate: {merge.missing_rate_raw.iloc[0]:.3%}.\n- E0 RMSE: {base.Overall_RMSE:.6f}; known {base.Known_RMSE:.6f}; masked {base.Masked_RMSE:.6f}; worst {base.Worst_fold:.6f}.\n- Best ERA system: {best.system}, RMSE {best.Overall_RMSE:.6f}, gain {best.Overall_gain:.6f}, masked gain {best.Masked_gain:.6f}, worst-fold gain {best.Worst_fold_gain:.6f}.\n- E3 OOF-selected subset: {', '.join(subset)} (exploratory; same OOF folds, not independent confirmation).\n- Most important raw: {raw_imp.index[0] if len(raw_imp) else 'none'}; derived: {der_imp.index[0] if len(der_imp) else 'none'}.\n- Known-expert top ERA5 feature: {known_top}; masked-expert top: {masked_top}. Fold stability is available in `feature_importance.csv`.\n- Frozen gates: {decisions}.\n- No full download was automatically started. No submission was produced.\n\nThis remains a three-region pilot. Proceeding means authorization for a separate future global download step, not evidence of public-leaderboard improvement.\n'''
(ART/'final_recommendation.md').write_text(report,encoding='utf-8'); stamp(decision); print(report)
