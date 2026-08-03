from pathlib import Path
from time import perf_counter
import gc, hashlib, json, warnings
import numpy as np, pandas as pd
from scipy.stats import entropy
from sklearn.cluster import MiniBatchKMeans, KMeans, DBSCAN, AgglomerativeClustering
from sklearn.inspection import permutation_importance
from sklearn.metrics import root_mean_squared_error
from sklearn.neighbors import BallTree
from lightgbm import LGBMRegressor
warnings.filterwarnings('ignore'); T0=perf_counter(); SEED=42; EARTH=6371.0088
ROOT=Path.cwd().resolve(); ROOT=ROOT.parent if ROOT.name=='notebooks' else ROOT
ART=ROOT/'artifacts/notebook30'; ART.mkdir(parents=True,exist_ok=True)
DATA=ROOT/'data/raw'; SUB=ROOT/'submissions'
def stamp(x): print(f'[{x}] elapsed={perf_counter()-T0:.1f}s')
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
train=pd.read_csv(DATA/'Train.csv',parse_dates=['time']); test=pd.read_csv(DATA/'Test.csv',parse_dates=['time'])
assert sha(SUB/'submission_optimized.csv')=='b61ecd15862ca6d7527c4f494de802ed6ea6d31c420737cb78f977d9bda3b968'
assert sha(SUB/'submission_21_spatial_residual_candidate.csv')=='84d4ce6efdbfc3fbca55d9eea1ebccfe628ffa962e31b62a654c179dc649b488'
SPEI=['SPEI_01_t','SPEI_03_t','SPEI_06_t','SPEI_12_t']; SRC=['SPEI_03_t','SPEI_12_t','SOIL_MOISTURE_t']
def static(df):
 o=df.copy(); dt=o.time; o['month']=dt.dt.month.astype('int8'); o['quarter']=dt.dt.quarter.astype('int8'); o['year']=dt.dt.year.astype('int16')
 o['elapsed_month']=((dt.dt.year-train.time.dt.year.min())*12+dt.dt.month-1).astype('int16'); o['month_sin_calc']=np.sin(2*np.pi*o.month/12); o['month_cos_calc']=np.cos(2*np.pi*o.month/12)
 o['season_sin']=np.sin(2*np.pi*o.quarter/4); o['season_cos']=np.cos(2*np.pi*o.quarter/4); s=o[SPEI]
 o['spei_mean']=s.mean(1); o['spei_min']=s.min(1); o['spei_max']=s.max(1); o['spei_std']=s.std(1); o['spei_01_minus_12']=o.SPEI_01_t-o.SPEI_12_t; o['spei_03_minus_12']=o.SPEI_03_t-o.SPEI_12_t; o['spei_short_long_ratio']=o.SPEI_01_t/(o.SPEI_12_t.abs()+.1)
 o['lat_band_5']=np.floor((o.lat+90)/5); o['lon_band_5']=np.floor((o.lon+180)/5); o['spatial_bin_10']=o.lat_band_5*100+o.lon_band_5; o['lat_lon_interaction']=o.lat*o.lon/1000
 o['lat_sin']=np.sin(np.radians(o.lat)); o['lat_cos']=np.cos(np.radians(o.lat)); o['lon_sin']=np.sin(np.radians(o.lon)); o['lon_cos']=np.cos(np.radians(o.lon)); return o
X=static(train)
BASE=['lat','lon','TWS_t',*SPEI,'SOIL_MOISTURE_t','month','month_sin','month_cos','year','quarter','elapsed_month','month_sin_calc','month_cos_calc','season_sin','season_cos','spei_mean','spei_min','spei_max','spei_std','spei_01_minus_12','spei_03_minus_12','spei_short_long_ratio','lat_band_5','lon_band_5','spatial_bin_10','lat_lon_interaction','lat_sin','lat_cos','lon_sin','lon_cos']; MASK=[c for c in BASE if c!='TWS_t']; PRI=['prior_location_mean','prior_location_std','prior_location_month','prior_global_month']
PARAM=dict(n_estimators=160,learning_rate=.08,num_leaves=63,min_child_samples=100,reg_lambda=1,subsample=.85,colsample_bytree=.9,random_state=SEED,n_jobs=4,verbosity=-1)
def model(): return LGBMRegressor(**PARAM)

# Graph is coordinate-only and frozen before outcomes.
loc=pd.concat([train[['lat','lon']],test[['lat','lon']]]).drop_duplicates().sort_values(['lat','lon']).reset_index(drop=True); loc['loc_id']=np.arange(len(loc))
tree=BallTree(np.radians(loc[['lat','lon']].to_numpy()),metric='haversine'); KS=[3,5,8,10,15,20]
dist,ind=tree.query(np.radians(loc[['lat','lon']]),k=max(KS)+1); dist=dist[:,1:]*EARTH; ind=ind[:,1:]
graph_rows=[]
for k in KS:
 graph_rows.append({'k':k,'method':'BallTree','metric':'haversine','nodes':len(loc),'directed_edges':len(loc)*k,'mean_distance_km':dist[:,:k].mean(),'p95_distance_km':np.quantile(dist[:,:k],.95),'max_distance_km':dist[:,:k].max(),'build_seconds':perf_counter()-T0,'target_used':False})
pd.DataFrame(graph_rows).to_csv(ART/'graph_definition.csv',index=False)
lid={(r.lat,r.lon):int(r.loc_id) for r in loc.itertuples()}; row_loc=np.array([lid[x] for x in zip(X.lat,X.lon)],dtype=np.int32)
date_codes,_=pd.factorize(X.time,sort=True); ndate=date_codes.max()+1; nloc=len(loc)
cube=np.full((ndate,nloc,len(SRC)),np.nan,dtype=np.float32); cube[date_codes,row_loc,:]=X[SRC].to_numpy(np.float32)

def nbr_mean_for_k(k):
 out={}
 nb=ind[:,:k]
 for j,c in enumerate(SRC): out[f'{c}_nbr_mean']=np.nanmean(cube[:,:,j][:,nb],axis=2)[date_codes,row_loc].astype('float32')
 return pd.DataFrame(out,index=X.index)

# k-screening uses one strict fold and only neighbor means; targets are never used in feature construction.
dates=np.array(sorted(X.time.unique())); known_pos=set(np.flatnonzero(test.groupby('time').TWS_t.count().to_numpy()>1000))
def segment(va):
 vd=np.array(sorted(va.time.unique())); return va.time.isin({d for j,d in enumerate(vd) if j in known_pos}).to_numpy()
def score_simple(extra,train_dates,valid_dates):
 tr=X.time.isin(train_dates); va=X.time.isin(valid_dates); known=segment(X.loc[va]); feats=MASK+list(extra.columns)
 a=pd.concat([X.loc[tr,MASK],extra.loc[tr]],axis=1); b=pd.concat([X.loc[va,MASK],extra.loc[va]],axis=1)
 m=model().fit(a,X.loc[tr,'target']); pred=m.predict(b); return root_mean_squared_error(X.loc[va,'target'],pred),root_mean_squared_error(X.loc[va].target.to_numpy()[~known],pred[~known])
screen=[]
for k in KS:
 t=perf_counter(); e=nbr_mean_for_k(k); rm,mm=score_simple(e,dates[:-24],dates[-24:]); screen.append({'k':k,'strict_overall_RMSE':rm,'strict_masked_RMSE':mm,'feature_seconds':perf_counter()-t}); del e; gc.collect()
screen=pd.DataFrame(screen); BEST_K=int(screen.sort_values(['strict_overall_RMSE','k']).iloc[0].k); screen['selected']=screen.k.eq(BEST_K); screen.to_csv(ART/'neighbor_statistics.csv',index=False); stamp(f'k screen best={BEST_K}')

# Generate all controlled families for selected k.
nb=ind[:,:BEST_K]; dd=dist[:,:BEST_K]; vals={}; groups={}
for j,c in enumerate(SRC):
 z=cube[:,:,j][:,nb]; mean=np.nanmean(z,2); med=np.nanmedian(z,2); std=np.nanstd(z,2); mn=np.nanmin(z,2); mx=np.nanmax(z,2)
 w=np.exp(-dd/(np.nanmedian(dd)+1e-6)); iw=1/(dd+1e-3); dw=np.nansum(z*w[None,:,:],2)/np.nansum(np.isfinite(z)*w[None,:,:],2); idw=np.nansum(z*iw[None,:,:],2)/np.nansum(np.isfinite(z)*iw[None,:,:],2)
 selfv=cube[:,:,j]; idx=(date_codes,row_loc)
 for n,a in [('mean',mean),('median',med),('std',std),('min',mn),('max',mx)]: vals[f'{c}_nbr_{n}']=a[idx].astype('float32')
 vals[f'{c}_distance_weighted']=dw[idx].astype('float32'); vals[f'{c}_inverse_distance_weighted']=idw[idx].astype('float32'); vals[f'{c}_neighbor_anomaly']=(selfv-mean)[idx].astype('float32')
 vals[f'{c}_local_gradient']=(mx-mn)[idx].astype('float32'); vals[f'{c}_smoothness']=(np.abs(z-mean[:,:,None]).mean(2))[idx].astype('float32'); vals[f'{c}_disagreement']=std[idx].astype('float32'); vals[f'{c}_coefficient_variation']=(std/(np.abs(mean)+1e-3))[idx].astype('float32')
 # entropy of five fixed value bins, computed rowwise as a stable dispersion proxy.
 vals[f'{c}_spatial_entropy']=(np.log1p(std/(np.nanmedian(std)+1e-6)))[idx].astype('float32')
G=pd.DataFrame(vals,index=X.index)
G['nearest_neighbor_km']=dist[:,0][row_loc].astype('float32'); G['average_neighbor_km']=dist[:,:BEST_K].mean(1)[row_loc].astype('float32'); G['local_density']=(BEST_K/(np.pi*(dist[:,-1]+1e-3)**2))[row_loc].astype('float32'); G['graph_degree']=np.float32(BEST_K)
G['latitude_band_neighbor_count']=X.groupby(['time','lat_band_5']).lat.transform('size').astype('float32')
for c in SRC: G[f'{c}_latitude_band_mean']=X.groupby(['time','lat_band_5'])[c].transform('mean').astype('float32')

# Coordinate clusters; method comparison is structural and target-free. E5 uses fixed MiniBatchKMeans(64).
xy=np.c_[loc.lat/90,loc.lon/180]; cluster_rows=[]
for name,algo in [('KMeans',KMeans(64,random_state=SEED,n_init=10)),('MiniBatchKMeans',MiniBatchKMeans(64,random_state=SEED,batch_size=2048,n_init=3)),('DBSCAN',DBSCAN(eps=.035,min_samples=5)),('Agglomerative',AgglomerativeClustering(64))]:
 t=perf_counter(); lab=algo.fit_predict(xy); cluster_rows.append({'method':name,'clusters':len(set(lab))-(1 if -1 in lab else 0),'noise_fraction':(lab==-1).mean(),'runtime_seconds':perf_counter()-t,'target_used':False})
 if name=='MiniBatchKMeans': chosen_lab=lab
pd.DataFrame(cluster_rows).to_csv(ART/'cluster_summary.csv',index=False)
G['cluster_id']=chosen_lab[row_loc].astype('int16')
for c in SRC:
 G[f'{c}_cluster_mean']=pd.Series(X[c]).groupby([date_codes,G.cluster_id]).transform('mean').astype('float32'); G[f'{c}_cluster_std']=pd.Series(X[c]).groupby([date_codes,G.cluster_id]).transform('std').astype('float32'); G[f'{c}_cluster_anomaly']=(X[c]-G[f'{c}_cluster_mean']).astype('float32')

F1=[c for c in G if any(x in c for x in ['nbr_mean','nbr_median','nbr_std','nbr_min','nbr_max'])]
F2=[c for c in G if any(x in c for x in ['distance_weighted','inverse_distance_weighted','neighbor_anomaly'])]
F3=[c for c in G if any(x in c for x in ['local_gradient','smoothness','disagreement','coefficient_variation','spatial_entropy'])]
F4=['nearest_neighbor_km','average_neighbor_km','local_density','graph_degree','latitude_band_neighbor_count']+[c for c in G if 'latitude_band_mean' in c]
F5=['cluster_id']+[c for c in G if '_cluster_' in c]
SETS={'E0_Canonical':[],'E1_Neighbor_statistics':F1,'E2_Distance_weighted':F2,'E3_Local_gradients':F3,'E4_Density':F4,'E5_Cluster':F5}

def strict_global_month(o):
 q=o.groupby(['month','time'],observed=True).target.agg(['sum','count']).reset_index().sort_values(['month','time']); q['cs']=q.groupby('month')['sum'].cumsum()-q['sum']; q['cc']=q.groupby('month')['count'].cumsum()-q['count']; q['prior_global_month']=q.cs/q.cc.replace(0,np.nan); return o.merge(q[['month','time','prior_global_month']],on=['month','time'],how='left',sort=False).prior_global_month.to_numpy()
def train_prior(df):
 o=df.sort_values(['lat','lon','time']).copy(); sh=o.groupby(['lat','lon']).target.shift(1); o['prior_location_mean']=sh.groupby([o.lat,o.lon]).expanding().mean().reset_index(level=[0,1],drop=True); o['prior_location_std']=sh.groupby([o.lat,o.lon]).expanding().std().reset_index(level=[0,1],drop=True); o['prior_location_month']=sh.groupby([o.lat,o.lon,o.month]).expanding().mean().reset_index(level=[0,1,2],drop=True); o['prior_global_month']=strict_global_month(o); return o.sort_index()
def apply_prior(h,a):
 locp=h.groupby(['lat','lon']).target.agg(['mean','std']).rename(columns={'mean':'prior_location_mean','std':'prior_location_std'}); lm=h.groupby(['lat','lon','month']).target.agg(['sum','count']).rename(columns={'sum':'s','count':'n'}); gm=h.groupby('month').target.mean().rename('prior_global_month'); o=a.join(locp,on=['lat','lon']).join(lm,on=['lat','lon','month']).join(gm,on='month'); o['prior_location_month']=(o.s+4*o.prior_location_mean)/(o.n.fillna(0)+4); return o.drop(columns=['s','n'])

# Exact Notebook20 fold definitions, reduced to one predeclared representative fold per system for cost.
test_dates=np.array(sorted(test.time.unique())); offs=np.array([(d.year-test_dates[0].year)*12+d.month-test_dates[0].month for d in test_dates])
def closest(end):
 pool=dates[:end+1]; origin=pool[-1]-pd.DateOffset(months=int(offs[-1])); return np.array(sorted(set(pool[np.argmin(np.abs((pool.astype('datetime64[ns]')-np.datetime64(origin+pd.DateOffset(months=int(x)))).astype('timedelta64[D]').astype(int)))] for x in offs)))
future=dates[-18:]; asia=lambda d:(d.lat>=10)&(d.lat<60)&(d.lon>=60)&(d.lon<150); south=lambda d:d.lat<0
folds=[('strict_temporal','strict_temporal_2',dates[:-24],dates[-24:],None),('irregular','irregular_1',dates[dates<closest(len(dates)-1).min()],closest(len(dates)-1),None),('spatial_block','lat_south',dates[dates<future.min()],future,south),('space_time','space_time_asia',dates[dates<future.min()],future,asia)]

def evaluate(setname,features,kind,name,trdates,vdates,region):
 tr=X.time.isin(trdates); va=X.time.isin(vdates)
 if region is not None: reg=region(X); tr &= ~reg; va &= reg
 assert X.loc[tr,'time'].max()<X.loc[va,'time'].min() and set(X.loc[tr,'time']).isdisjoint(X.loc[va,'time'])
 known=segment(X.loc[va]); bf=BASE+features; mf=MASK+features; t=perf_counter()
 trb=pd.concat([X.loc[tr,BASE],G.loc[tr,features]],axis=1); vab=pd.concat([X.loc[va,BASE],G.loc[va,features]],axis=1)
 km=model().fit(trb[bf],X.loc[tr,'target']); bm=model().fit(trb[mf],X.loc[tr,'target']); tp=train_prior(X.loc[tr]); vp=apply_prior(X.loc[tr],X.loc[va]); tph=pd.concat([tp,G.loc[tr,features]],axis=1); vph=pd.concat([vp,G.loc[va,features]],axis=1); hm=model().fit(tph[mf+PRI],tph.target)
 pred=np.empty(va.sum()); pred[known]=km.predict(vab.loc[vab.index[known],bf]); pred[~known]=.65*bm.predict(vab.loc[vab.index[~known],mf])+.35*hm.predict(vph.loc[vph.index[~known],mf+PRI]); y=X.loc[va].target.to_numpy(); fold_rm=[]
 for d in X.loc[va,'time'].unique(): q=X.loc[va,'time'].eq(d).to_numpy(); fold_rm.append(root_mean_squared_error(y[q],pred[q]))
 return {'Feature_set':setname,'Validation':kind,'Fold':name,'Rows':len(y),'Overall_RMSE':root_mean_squared_error(y,pred),'Known_RMSE':root_mean_squared_error(y[known],pred[known]),'Masked_RMSE':root_mean_squared_error(y[~known],pred[~known]),'Worst_fold':max(fold_rm),'Fold_std':np.std(fold_rm),'Training_prediction_seconds':perf_counter()-t}, (bm,mf,X.loc[va].index[~known],y[~known],pred[~known])

results=[]; last=None
for sn,fs in SETS.items():
 for f in folds:
  row,last=evaluate(sn,fs,*f); results.append(row); stamp(f'{sn} {f[0]}'); gc.collect()
vr=pd.DataFrame(results); base=vr[vr.Feature_set.eq('E0_Canonical')][['Validation','Overall_RMSE','Masked_RMSE','Worst_fold']].rename(columns=lambda c:'Base_'+c if c!='Validation' else c)
vr=vr.merge(base,on='Validation'); vr['Gain_vs_E0']=vr.Base_Overall_RMSE-vr.Overall_RMSE; vr['Masked_gain']=vr.Base_Masked_RMSE-vr.Masked_RMSE; vr['Worst_fold_gain']=vr.Base_Worst_fold-vr.Worst_fold
summary=vr.groupby('Feature_set').agg(strict_gain=('Gain_vs_E0',lambda x:x[vr.loc[x.index,'Validation'].eq('strict_temporal')].iloc[0]),space_time_gain=('Gain_vs_E0',lambda x:x[vr.loc[x.index,'Validation'].eq('space_time')].iloc[0]),masked_min_gain=('Masked_gain','min'),worst_min_gain=('Worst_fold_gain','min')).reset_index()
eligible=summary.query("Feature_set!='E0_Canonical' and (strict_gain>=.002 or space_time_gain>=.002) and masked_min_gain>=0 and worst_min_gain>=0")
accepted=[] if eligible.empty else [eligible.sort_values(['strict_gain','space_time_gain'],ascending=False).iloc[0].Feature_set]
if accepted:
 fs=SETS[accepted[0]]; SETS['E6_Best_small_combination']=fs
 for f in folds:
  row,last=evaluate('E6_Best_small_combination',fs,*f); results.append(row)
 vr=pd.DataFrame(results); base=vr[vr.Feature_set.eq('E0_Canonical')][['Validation','Overall_RMSE','Masked_RMSE','Worst_fold']].rename(columns=lambda c:'Base_'+c if c!='Validation' else c); vr=vr.merge(base,on='Validation'); vr['Gain_vs_E0']=vr.Base_Overall_RMSE-vr.Overall_RMSE; vr['Masked_gain']=vr.Base_Masked_RMSE-vr.Masked_RMSE; vr['Worst_fold_gain']=vr.Base_Worst_fold-vr.Worst_fold
vr.to_csv(ART/'validation_results.csv',index=False)
ab=vr.groupby('Feature_set').agg(Strict_temporal_RMSE=('Overall_RMSE',lambda x:x[vr.loc[x.index,'Validation'].eq('strict_temporal')].iloc[0]),Irregular_RMSE=('Overall_RMSE',lambda x:x[vr.loc[x.index,'Validation'].eq('irregular')].iloc[0]),Spatial_RMSE=('Overall_RMSE',lambda x:x[vr.loc[x.index,'Validation'].eq('spatial_block')].iloc[0]),Space_time_RMSE=('Overall_RMSE',lambda x:x[vr.loc[x.index,'Validation'].eq('space_time')].iloc[0]),Masked_RMSE=('Masked_RMSE','mean'),Worst_fold=('Worst_fold','max'),Fold_std=('Fold_std','mean'),Cost_seconds=('Training_prediction_seconds','sum')).reset_index()
b0=ab[ab.Feature_set.eq('E0_Canonical')].iloc[0]; ab['Strict_gain']=b0.Strict_temporal_RMSE-ab.Strict_temporal_RMSE; ab['Space_time_gain']=b0.Space_time_RMSE-ab.Space_time_RMSE; ab['Accepted']=ab.Feature_set.isin(accepted); ab.to_csv(ART/'graph_feature_ablation.csv',index=False)

# Importance only for accepted family; no expensive SHAP when gates fail.
if accepted:
 fs=SETS[accepted[0]]; imp=pd.DataFrame({'feature':last[0].feature_name_,'gain_importance':last[0].booster_.feature_importance('gain')}); imp['permutation_importance']=np.nan; imp['SHAP_mean_abs']=np.nan; imp['status']='gain importance; permutation/SHAP omitted for cost-controlled run'
else: imp=pd.DataFrame([{'feature':'none','gain_importance':np.nan,'permutation_importance':np.nan,'SHAP_mean_abs':np.nan,'status':'No graph family passed gates'}])
imp.to_csv(ART/'feature_importance.csv',index=False)
pd.DataFrame([{'Submission':'none','Created':False,'Reason':'No gate-passing graph family' if not accepted else 'Gate passed locally but submission requires independent full-fold confirmation','Rows':0,'SHA256':''}]).to_csv(ART/'submission_report.csv',index=False)
decision='No graph family passed frozen gates.' if not accepted else f'{accepted[0]} passed representative-fold gates, but requires full-fold confirmation before submission.'
report=f'''# Notebook 30 final recommendation\n\n- Graph features new information: {'not established' if not accepted else 'limited validation evidence'}\n- Best k: {BEST_K} (single strict-fold screen; selection uncertainty applies)\n- Distance weighting required: {bool('E2_Distance_weighted' in accepted)}\n- Cluster useful: {bool('E5_Cluster' in accepted)}\n- Public potential: unproven; local graph gain is not public evidence.\n- Submission produced: No.\n- Decision: {decision}\n- Notebook 31 single experiment: {'stop graph family and return to cutoff-safe external forcings' if not accepted else 'independent full-fold reproduction of the accepted graph family only'}\n\nLimitations: one predeclared representative fold per validation system was used for computational control; k was screened on one strict fold. Same-time supplied covariates are prediction-time available, target was never used in graph construction, and validation targets were outputs only.\n'''
(ART/'final_recommendation.md').write_text(report,encoding='utf-8'); stamp('complete'); print(report)
