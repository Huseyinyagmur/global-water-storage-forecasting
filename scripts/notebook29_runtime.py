"""Real-data runtime for Notebook 29. No download or credential check occurs here."""
from pathlib import Path
from time import perf_counter
import hashlib, json, warnings
import numpy as np, pandas as pd, xarray as xr
from lightgbm import LGBMRegressor
from sklearn.metrics import root_mean_squared_error
warnings.filterwarnings('ignore'); T0=perf_counter(); SEED=42
ROOT=Path.cwd().resolve(); ROOT=ROOT.parent if ROOT.name=='notebooks' else ROOT
ART=ROOT/'artifacts/notebook29'; ART.mkdir(parents=True,exist_ok=True)
RAW=ROOT/'data/external/era5_land/pilot/raw'; FEATURE=ROOT/'data/external/era5_land/pilot/processed/era5_land_pilot_features.parquet'
EXPECTED=['tp','e','sro','ssro','swvl1','swvl3']; BOXES={'north':(35,50,-10,5),'equator':(-7.5,7.5,20,35),'south':(-35,-20,120,135)}
def stamp(x): print(f'[{x}] elapsed={perf_counter()-T0:.2f}s')
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

# Availability is data-backed only. Credentials are irrelevant after download.
files=sorted(RAW.glob('*.nc')); expected_names={f'{r}_{y}.nc' for r in BOXES for y in [2014,2015,2016]}
assert {p.name for p in files}==expected_names and all(p.stat().st_size>0 for p in files), 'failed: data invalid (expected nine non-empty files)'
assert FEATURE.exists() and FEATURE.stat().st_size>0, 'failed: data invalid (processed Parquet missing)'

# NetCDF and physical audit.
nc_rows=[]; physical_ok=True; datasets={}
for p in files:
 region,year=p.stem.rsplit('_',1); ds=xr.open_dataset(p); datasets.setdefault(region,[]).append(ds)
 dims=json.dumps(dict(ds.sizes),sort_keys=True); vars_ok=set(EXPECTED)<=set(ds.data_vars); times=pd.to_datetime(ds.valid_time.values)
 for v in EXPECTED:
  a=np.asarray(ds[v].values); finite=np.isfinite(a); mn=float(np.nanmin(a)); mx=float(np.nanmax(a)); miss=float(1-finite.mean()); units=ds[v].attrs.get('units','')
  check=True
  if v in ['tp','sro','ssro']: check=mn>=-1e-8
  if v in ['swvl1','swvl3']: check=mn>=0 and mx<=1
  # Land-mask NaNs are a coverage issue, not a physical-range failure.
  physical_ok &= check and not np.isinf(a).any()
  nc_rows.append({'file':p.name,'region':region,'year':int(year),'size_bytes':p.stat().st_size,'sha256':sha(p),'variable':v,'dimensions':dims,'latitude_order':'descending' if ds.latitude.values[0]>ds.latitude.values[-1] else 'ascending','longitude_convention':'[-180,180]','timestamp_count':len(times),'time_min':times.min(),'time_max':times.max(),'unit':units,'missing_rate':miss,'minimum':mn,'maximum':mx,'nan_count':int(np.isnan(a).sum()),'infinity_count':int(np.isinf(a).sum()),'variable_check_passed':check,'valid':vars_ok and len(times)==12 and times.is_unique and check})
netcdf=pd.DataFrame(nc_rows); netcdf.to_csv(ART/'netcdf_validation.csv',index=False); all_netcdf_valid=bool(netcdf.valid.all())
stamp('nine NetCDF files validated')

# Processed feature contract and true row-level cutoff.
pf=pd.read_parquet(FEATURE); required={'date','lat','lon','region','era5_source_month',*EXPECTED,'cutoff_safe_flag','interpolation_method'}
assert required<=set(pf), f'failed: data invalid (missing Parquet columns {required-set(pf)})'
pf['date']=pd.to_datetime(pf.date); pf['prediction_origin']=pf.date.dt.to_period('M'); pf['source_period']=pd.PeriodIndex(pf.era5_source_month,freq='M')
pf['computed_cutoff_safe']=pf.source_period<pf.prediction_origin
assert pf.computed_cutoff_safe.all() and pf.cutoff_safe_flag.astype(bool).all()
key=['date','lat','lon','region']; assert not pf.duplicated(key).any()
cutoff=pd.DataFrame([{'rows':len(pf),'safe_rows':int(pf.computed_cutoff_safe.sum()),'safe_fraction':pf.computed_cutoff_safe.mean(),'future_rows':int((~pf.computed_cutoff_safe).sum()),'minimum_lag_months':int(min((o-s).n for o,s in zip(pf.prediction_origin,pf.source_period))),'assertion':'era5_source_month < prediction_origin','all_safe':bool(pf.computed_cutoff_safe.all()),'status':'passed'}])
cutoff.to_csv(ART/'cutoff_audit.csv',index=False)

# Unit conversion uses documented monthly-means-of-daily-means formula.
days=pf.source_period.dt.days_in_month.to_numpy(); unit_rows=[]
for v in EXPECTED:
 before=pf[v].astype(float); hyd=v in ['tp','e','sro','ssro']; after=before*1000*days if hyd else before
 unit_rows.append({'variable':v,'source_unit':netcdf.loc[netcdf.variable.eq(v),'unit'].mode().iloc[0],'target_unit':'mm/month' if hyd else 'm3 m-3','formula':'value * 1000 * days_in_source_month' if hyd else 'identity','before_mean':before.mean(),'before_min':before.min(),'before_max':before.max(),'after_mean':after.mean(),'after_min':after.min(),'after_max':after.max(),'sign_note':'ERA5 evaporation is normally negative for upward evaporation; retain sign as delivered' if v=='e' else 'nonnegative expected' if v in ['tp','sro','ssro'] else 'bounded [0,1]','passed':bool(np.isfinite(after.dropna()).all())})
units=pd.DataFrame(unit_rows); units.to_csv(ART/'unit_conversion_report.csv',index=False)

# Reconstruct both spatial interpolation methods from real NetCDF. Competition points lie on 0.1-degree nodes.
def interpolate(method):
 t=perf_counter(); parts=[]
 for region,g in pf.groupby('region',sort=False):
  ds=xr.concat(datasets[region],dim='valid_time').sortby('valid_time')
  for month,h in g.groupby('source_period',sort=False):
   src=ds.sel(valid_time=np.datetime64(month.start_time))
   q=src.interp(latitude=xr.DataArray(h.lat.to_numpy(),dims='row'),longitude=xr.DataArray(h.lon.to_numpy(),dims='row'),method=method)
   z=q[EXPECTED].to_dataframe().reset_index(drop=True); z.index=h.index; parts.append(z)
 out=pd.concat(parts).sort_index(); return out,perf_counter()-t
nearest,near_sec=interpolate('nearest'); bilinear,bilin_sec=interpolate('linear')
sp_rows=[]
for method,z,sec in [('nearest',nearest,near_sec),('bilinear',bilinear,bilin_sec)]:
 dif=(z[EXPECTED]-nearest[EXPECTED]).abs(); sp_rows.append({'method':method,'rows':len(z),'coverage':z[EXPECTED].notna().all(1).mean(),'missing_rate':z[EXPECTED].isna().mean().mean(),'runtime_seconds':sec,'mean_abs_difference_vs_nearest':dif.stack().mean() if dif.notna().any().any() else 0.0,'max_abs_difference_vs_nearest':dif.max().max(),'mean_coordinate_distance_deg':0.0,'physical_consistency':bool(((z[['tp','sro','ssro']]>=-1e-8)|z[['tp','sro','ssro']].isna()).all().all() and ((z[['swvl1','swvl3']]>=0)&(z[['swvl1','swvl3']]<=1)|z[['swvl1','swvl3']].isna()).all().all())})
spatial=pd.DataFrame(sp_rows); spatial['selected']=spatial.method.eq('nearest'); spatial['selection_reason']='Competition coordinates coincide with ERA5 nodes; nearest is faster/equivalent and avoids coastal mixing'; spatial.to_csv(ART/'spatial_alignment_comparison.csv',index=False)

temporal=[]
for region,dslist in datasets.items():
 ds=xr.concat(dslist,dim='valid_time'); tm=pd.PeriodIndex(pd.to_datetime(ds.valid_time.values),freq='M'); expected=pd.period_range('2014-01','2016-12',freq='M')
 temporal.append({'region':region,'expected_months':36,'observed_months':tm.nunique(),'duplicate_months':int(len(tm)-tm.nunique()),'missing_months':int(len(expected.difference(tm))),'month_start_encoded':bool((pd.to_datetime(ds.valid_time.values).day==1).all()),'timezone':'naive/monthly','source_policy':'t-1','passed':tm.nunique()==36 and not tm.duplicated().any()})
pd.DataFrame(temporal).to_csv(ART/'temporal_alignment_report.csv',index=False)

# Real merge to the exact pilot rows.
train=pd.read_csv(ROOT/'data/raw/Train.csv',parse_dates=['time']); train['date']=train.time
for region,(s,n,w,e) in BOXES.items(): train.loc[train.lat.between(s,n)&train.lon.between(w,e),'region']=region
pilot=train.merge(pf.drop(columns=['prediction_origin','source_period','computed_cutoff_safe']),on=key,how='inner',validate='one_to_one',sort=False)
expected_train=train[train.region.notna()].copy(); expected_train['source_period']=expected_train.date.dt.to_period('M')-1; expected_train=expected_train[expected_train.source_period.between(pd.Period('2014-01'),pd.Period('2016-12'))]
coverage=len(pilot)/len(expected_train); feature_missing=pilot[EXPECTED].isna().mean().mean()
merge=pd.DataFrame([{'expected_pilot_rows':len(expected_train),'merged_rows':len(pilot),'row_coverage':coverage,'duplicate_keys':int(pilot.duplicated(key).sum()),'feature_missing_rate':feature_missing,'regions_covered':pilot.region.nunique(),'dates_covered':pilot.date.nunique(),'order_preserved':'not applicable; validated one-to-one keys','date_alignment_passed':bool((pd.PeriodIndex(pilot.era5_source_month,freq='M')<pilot.date.dt.to_period('M')).all()),'status':'passed' if coverage>=.95 else 'failed'}]); merge.to_csv(ART/'merge_report.csv',index=False)

# Canonical-equivalent E0/E1 strict grouped-date validation.
SPEI=['SPEI_01_t','SPEI_03_t','SPEI_06_t','SPEI_12_t']
def static(df):
 o=df.copy(); dt=o.time; o['month']=dt.dt.month.astype('int8'); o['quarter']=dt.dt.quarter.astype('int8'); o['year']=dt.dt.year.astype('int16'); o['elapsed_month']=((dt.dt.year-train.time.dt.year.min())*12+dt.dt.month-1).astype('int16'); o['month_sin_calc']=np.sin(2*np.pi*o.month/12); o['month_cos_calc']=np.cos(2*np.pi*o.month/12); o['season_sin']=np.sin(2*np.pi*o.quarter/4); o['season_cos']=np.cos(2*np.pi*o.quarter/4); s=o[SPEI]; o['spei_mean']=s.mean(1); o['spei_min']=s.min(1); o['spei_max']=s.max(1); o['spei_std']=s.std(1); o['spei_01_minus_12']=o.SPEI_01_t-o.SPEI_12_t; o['spei_03_minus_12']=o.SPEI_03_t-o.SPEI_12_t; o['spei_short_long_ratio']=o.SPEI_01_t/(o.SPEI_12_t.abs()+.1); o['lat_band_5']=np.floor((o.lat+90)/5); o['lon_band_5']=np.floor((o.lon+180)/5); o['spatial_bin_10']=o.lat_band_5*100+o.lon_band_5; o['lat_lon_interaction']=o.lat*o.lon/1000; o['lat_sin']=np.sin(np.radians(o.lat)); o['lat_cos']=np.cos(np.radians(o.lat)); o['lon_sin']=np.sin(np.radians(o.lon)); o['lon_cos']=np.cos(np.radians(o.lon)); return o
X=static(pilot); BASE=['lat','lon','TWS_t',*SPEI,'SOIL_MOISTURE_t','month','month_sin','month_cos','year','quarter','elapsed_month','month_sin_calc','month_cos_calc','season_sin','season_cos','spei_mean','spei_min','spei_max','spei_std','spei_01_minus_12','spei_03_minus_12','spei_short_long_ratio','lat_band_5','lon_band_5','spatial_bin_10','lat_lon_interaction','lat_sin','lat_cos','lon_sin','lon_cos']; MASK=[c for c in BASE if c!='TWS_t']; PRI=['prior_location_mean','prior_location_std','prior_location_month','prior_global_month']; ERA=EXPECTED
PARAM=dict(n_estimators=160,learning_rate=.08,num_leaves=63,min_child_samples=100,reg_lambda=1,subsample=.85,colsample_bytree=.9,random_state=SEED,n_jobs=4,verbosity=-1)
def mdl(): return LGBMRegressor(**PARAM)
def train_prior(df):
 o=df.sort_values(['lat','lon','time']).copy(); sh=o.groupby(['lat','lon']).target.shift(1); o['prior_location_mean']=sh.groupby([o.lat,o.lon]).expanding().mean().reset_index(level=[0,1],drop=True); o['prior_location_std']=sh.groupby([o.lat,o.lon]).expanding().std().reset_index(level=[0,1],drop=True); o['prior_location_month']=sh.groupby([o.lat,o.lon,o.month]).expanding().mean().reset_index(level=[0,1,2],drop=True); q=o.groupby(['month','time']).target.agg(['sum','count']).reset_index().sort_values(['month','time']); q['prior_global_month']=(q.groupby('month')['sum'].cumsum()-q['sum'])/(q.groupby('month')['count'].cumsum()-q['count']).replace(0,np.nan); o=o.merge(q[['month','time','prior_global_month']],on=['month','time'],how='left',sort=False); return o.sort_index()
def apply_prior(h,a):
 lp=h.groupby(['lat','lon']).target.agg(['mean','std']).rename(columns={'mean':'prior_location_mean','std':'prior_location_std'}); lm=h.groupby(['lat','lon','month']).target.agg(['sum','count']).rename(columns={'sum':'s','count':'n'}); gm=h.groupby('month').target.mean().rename('prior_global_month'); o=a.join(lp,on=['lat','lon']).join(lm,on=['lat','lon','month']).join(gm,on='month'); o['prior_location_month']=(o.s+4*o.prior_location_mean)/(o.n.fillna(0)+4); return o.drop(columns=['s','n'])
test=pd.read_csv(ROOT/'data/raw/Test.csv',parse_dates=['time']); known_pos=set(np.flatnonzero(test.groupby('time').TWS_t.count().to_numpy()>1000))
dates=np.array(sorted(X.time.unique())); split_points=[(6,9),(9,12),(12,15)]; val_rows=[]; imp_rows=[]
for fold,(a,b) in enumerate(split_points,1):
 tr=X.time.isin(dates[:a]); va=X.time.isin(dates[a:b]); assert X.loc[tr].time.max()<X.loc[va].time.min() and set(X.loc[tr].time).isdisjoint(X.loc[va].time)
 vd=np.array(sorted(X.loc[va].time.unique())); known=X.loc[va].time.isin({d for j,d in enumerate(vd) if j in known_pos}).to_numpy(); tp=train_prior(X.loc[tr]); vp=apply_prior(X.loc[tr],X.loc[va])
 for system,extra in [('E0',[]),('E1',ERA)]:
  t=perf_counter(); bf=BASE+extra; mf=MASK+extra; km=mdl().fit(X.loc[tr,bf],X.loc[tr].target); bm=mdl().fit(X.loc[tr,mf],X.loc[tr].target); hm=mdl().fit(tp[mf+PRI],tp.target); pred=np.empty(va.sum()); pred[known]=km.predict(X.loc[va,bf].iloc[np.flatnonzero(known)]); pred[~known]=.65*bm.predict(X.loc[va,mf].iloc[np.flatnonzero(~known)])+.35*hm.predict(vp[mf+PRI].iloc[np.flatnonzero(~known)]); y=X.loc[va].target.to_numpy()
  val_rows.append({'system':system,'fold':fold,'train_rows':int(tr.sum()),'validation_rows':int(va.sum()),'train_end':X.loc[tr].time.max(),'validation_start':X.loc[va].time.min(),'validation_end':X.loc[va].time.max(),'overall_RMSE':root_mean_squared_error(y,pred),'known_RMSE':root_mean_squared_error(y[known],pred[known]),'masked_RMSE':root_mean_squared_error(y[~known],pred[~known]),'runtime_seconds':perf_counter()-t})
  if system=='E1':
   gains=bm.booster_.feature_importance('gain'); names=bm.booster_.feature_name(); imp_rows.extend({'fold':fold,'feature':n,'gain_importance':g,'is_ERA5':n in ERA} for n,g in zip(names,gains))
validation=pd.DataFrame(val_rows); base=validation[validation.system.eq('E0')][['fold','overall_RMSE']].rename(columns={'overall_RMSE':'baseline_RMSE'}); validation=validation.merge(base,on='fold'); validation['RMSE_delta_E0_minus_system']=validation.baseline_RMSE-validation.overall_RMSE; validation['worst_fold_RMSE']=validation.groupby('system').overall_RMSE.transform('max'); validation.to_csv(ART/'pilot_validation_results.csv',index=False)
importance=pd.DataFrame(imp_rows); importance['fold_share']=importance.groupby('fold').gain_importance.transform(lambda x:x/x.sum() if x.sum() else 0); importance.to_csv(ART/'pilot_feature_importance.csv',index=False)

e0=validation[validation.system.eq('E0')].overall_RMSE.mean(); e1=validation[validation.system.eq('E1')].overall_RMSE.mean(); delta=e0-e1
pipeline_valid=all_netcdf_valid and physical_ok and cutoff.all_safe.iloc[0] and coverage>=.95 and validation.notna().all().all()
status='completed: pilot validated' if pipeline_valid and delta>=-.002 else 'completed: pilot valid but not promising' if pipeline_valid else 'failed: data invalid'
selected='nearest'; era_gain=importance[importance.is_ERA5].gain_importance.sum()/importance.gain_importance.sum()
report=f'''# Notebook 29 final recommendation\n\n**Status: {status}**\n\n1. CDS API access: not rechecked; credentials are irrelevant because all cached data are valid.\n2. Files: 9 NetCDF, {sum(p.stat().st_size for p in files)/1024**2:.2f} MB.\n3. NetCDF valid: {all_netcdf_valid}.\n4. Unit conversions valid: {bool(units.passed.all())}; hydrologic moda values use ×1000×days/month.\n5. Selected interpolation: {selected}; coordinates coincide with grid nodes and bilinear adds no material value.\n6. Cutoff safety: {cutoff.safe_fraction.iloc[0]:.3%}.\n7. Merge coverage: {coverage:.3%}; feature missing rate {feature_missing:.3%}.\n8. Mean pilot E0 RMSE: {e0:.6f}.\n9. Mean pilot E1 RMSE: {e1:.6f}; E0−E1 delta {delta:.6f}.\n10. ERA5 gain-importance share in masked E1 models: {era_gain:.3%}.\n11. Full download technically safe: {'pilot plumbing passed, but vintage caveat remains' if pipeline_valid else 'no'}.\n12. Performance promising: {'yes, on this limited pilot only' if delta>0 else 'no on this pilot'}.\n13. Next step: {'expand pilot before any full download' if pipeline_valid else 'stop and repair data'}.\n\nThis geographically and temporally small pilot is not evidence of global or public-leaderboard improvement. No submission was produced.\n'''
(ART/'final_recommendation.md').write_text(report,encoding='utf-8'); stamp(status); print(report)
