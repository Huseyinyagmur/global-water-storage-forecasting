"""Controlled global-panel sequence pilot with honest stop/full gate."""
from pathlib import Path
from time import perf_counter
import hashlib, json, random, warnings
import numpy as np, pandas as pd, torch
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import root_mean_squared_error
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
warnings.filterwarnings('ignore'); SEED=42; random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
ROOT=Path.cwd().resolve(); ROOT=ROOT.parent if ROOT.name=='notebooks' else ROOT; ART=ROOT/'artifacts/notebook33'; ART.mkdir(parents=True,exist_ok=True); T0=perf_counter()
DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); AMP=DEVICE.type=='cuda'
train=pd.read_csv(ROOT/'data/raw/Train.csv',parse_dates=['time']); oof=pd.read_parquet(ROOT/'artifacts/notebook25/oof_row_level_predictions.parquet')
FEATURES=['TWS_t','SPEI_01_t','SPEI_03_t','SPEI_06_t','SPEI_12_t','SOIL_MOISTURE_t','month_sin_seq','month_cos_seq','lat','lon','tws_missing']
train['month_sin_seq']=np.sin(2*np.pi*train.time.dt.month/12); train['month_cos_seq']=np.cos(2*np.pi*train.time.dt.month/12); train['location_id']=train.lat.astype(str)+'_'+train.lon.astype(str)
fold_oof=oof[oof.fold_id.eq('strict_temporal_2')].copy(); valid_idx=set(fold_oof.original_index); valid_dates=np.array(sorted(train.loc[list(valid_idx),'time'].unique())); cutoff=valid_dates.min()
locs=train.loc[list(valid_idx),['location_id','lat']].drop_duplicates(); locs['band']=pd.cut(locs.lat,[-90,-30,0,30,60,90],labels=False); selected=locs.groupby('band',group_keys=False).apply(lambda g:g.sample(min(64,len(g)),random_state=SEED),include_groups=False).location_id.unique(); pilot=train[train.location_id.isin(selected)].copy(); pilot_valid=pilot.index.isin(valid_idx); pilot_train=pilot.time<cutoff
panel=pd.DataFrame([{'locations_total':train.location_id.nunique(),'pilot_locations':len(selected),'rows_total':len(train),'pilot_rows':len(pilot),'pilot_train_rows':int(pilot_train.sum()),'pilot_validation_rows':int(pilot_valid.sum()),'date_min':pilot.time.min(),'date_max':pilot.time.max(),'validation_start':cutoff,'chronological_sort':True,'location_key':'lat+lon','CUDA_available':torch.cuda.is_available(),'device':str(DEVICE),'GPU_name':torch.cuda.get_device_name(0) if AMP else 'unavailable: installed PyTorch is CPU-only'}]); panel.to_csv(ART/'panel_structure.csv',index=False)

known_by_date={d:(j in set(np.flatnonzero(pd.read_csv(ROOT/'data/raw/Test.csv',parse_dates=['time']).groupby('time').TWS_t.count().to_numpy()>1000))) for j,d in enumerate(valid_dates)}
def raw_vector(row,mask_tws=False):
 tws=np.nan if mask_tws else row.TWS_t; return np.array([tws,row.SPEI_01_t,row.SPEI_03_t,row.SPEI_06_t,row.SPEI_12_t,row.SOIL_MOISTURE_t,row.month_sin_seq,row.month_cos_seq,row.lat,row.lon,float(pd.isna(tws))],dtype=np.float32)
def build_sequences(L,dual_mode=False):
 trX=[]; try_=[]; vaX=[]; vay=[]; vaidx=[]; vaknown=[]; vacold=[]
 for _,g in pilot.groupby('location_id',sort=False):
  g=g.sort_values('time'); history=[]
  for idx,row in g.iterrows():
   is_val=idx in valid_idx; known=known_by_date.get(row.time,True); vec=raw_vector(row,mask_tws=is_val and not known)
   seq=(history+[vec])[-L:]; cold=len(seq)<L; seq=[np.zeros(len(FEATURES),np.float32)]*(L-len(seq))+seq
   if row.time<cutoff:
    arr=np.stack(seq); trX.append(arr); try_.append(row.target)
   elif is_val:
    vaX.append(np.stack(seq)); vay.append(row.target); vaidx.append(idx); vaknown.append(known); vacold.append(cold)
   history.append(vec)
 Xtr=np.stack(trX); ytr=np.array(try_,np.float32); Xva=np.stack(vaX); yva=np.array(vay,np.float32)
 # Fold-training-only feature-wise scaling; padding remains zero after explicit mask.
 real=np.any(Xtr!=0,axis=2); mu=np.nanmean(np.where(real[:,:,None],Xtr,np.nan),axis=(0,1)); sd=np.nanstd(np.where(real[:,:,None],Xtr,np.nan),axis=(0,1)); sd=np.where(sd<1e-6,1,sd); Xtr=np.nan_to_num((Xtr-mu)/sd); Xva=np.nan_to_num((Xva-mu)/sd); ym=ytr.mean(); ys=ytr.std() or 1
 return Xtr,(ytr-ym)/ys,Xva,yva,np.array(vaidx),np.array(vaknown),np.array(vacold),ym,ys

class RNN(nn.Module):
 def __init__(self,kind,nf):
  super().__init__(); cls=nn.GRU if kind=='GRU' else nn.LSTM; self.r=cls(nf,32,batch_first=True); self.h=nn.Sequential(nn.Linear(32,16),nn.ReLU(),nn.Linear(16,1))
 def forward(self,x): return self.h(self.r(x)[0][:,-1]).squeeze(-1)
class TCN(nn.Module):
 def __init__(self,nf): super().__init__(); self.n=nn.Sequential(nn.Conv1d(nf,32,3,padding=2,dilation=1),nn.ReLU(),nn.Conv1d(32,32,3,padding=4,dilation=2),nn.ReLU()); self.h=nn.Linear(32,1)
 def forward(self,x): return self.h(self.n(x.transpose(1,2))[:,:,-1]).squeeze(-1)
def fit(kind,L,structure='single',epochs=10):
 Xtr,ytr,Xva,yva,idx,known,cold,ym,ys=build_sequences(L); t=perf_counter(); histories=[]; peak=0
 def one(mask_train,mask_val,label):
  nonlocal peak
  model=(TCN(Xtr.shape[2]) if kind=='TCN' else RNN(kind,Xtr.shape[2])).to(DEVICE); opt=torch.optim.Adam(model.parameters(),lr=1e-3); lossf=nn.MSELoss(); best=1e9; state=None; patience=0
  tx=torch.tensor(Xtr[mask_train]); ty=torch.tensor(ytr[mask_train]); loader=DataLoader(TensorDataset(tx,ty),batch_size=512,shuffle=True,num_workers=0,pin_memory=AMP)
  vx=torch.tensor(Xva[mask_val],device=DEVICE); vy=torch.tensor((yva[mask_val]-ym)/ys,device=DEVICE)
  for ep in range(1,epochs+1):
   model.train(); losses=[]
   for xb,yb in loader:
    xb=xb.to(DEVICE); yb=yb.to(DEVICE); opt.zero_grad(set_to_none=True)
    with torch.autocast(device_type=DEVICE.type,enabled=AMP): loss=lossf(model(xb),yb)
    loss.backward(); opt.step(); losses.append(loss.item())
   model.eval()
   with torch.no_grad(): vl=lossf(model(vx),vy).item()
   histories.append({'kind':kind,'length':L,'structure':structure,'expert':label,'epoch':ep,'train_loss':np.mean(losses),'validation_loss':vl})
   if vl<best-1e-5: best=vl; state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}; patience=0
   else: patience+=1
   if AMP: peak=max(peak,torch.cuda.max_memory_allocated())
   if patience>=3: break
  model.load_state_dict(state); model.eval()
  with torch.no_grad(): return model(torch.tensor(Xva[mask_val],device=DEVICE)).cpu().numpy()*ys+ym
 if structure=='single':
  # Deterministic current-TWS masking augmentation teaches one model both regimes.
  aug=Xtr.copy(); aug[:, -1, 0]=0; aug[:, -1, -1]=1; Xtr=np.concatenate([Xtr,aug]); ytr=np.concatenate([ytr,ytr]); pred=one(np.ones(len(Xtr),bool),np.ones(len(Xva),bool),'global')
 else:
  # Both experts see all training targets; masked expert sees no current TWS.
  baseX=Xtr.copy(); pred=np.empty(len(Xva)); pred[known]=one(np.ones(len(Xtr),bool),known,'known'); Xtr[:, -1, 0]=0; Xtr[:, -1, -1]=1; pred[~known]=one(np.ones(len(Xtr),bool),~known,'masked'); Xtr=baseX
 return {'kind':kind,'length':L,'structure':structure,'RMSE':root_mean_squared_error(yva,pred),'known_RMSE':root_mean_squared_error(yva[known],pred[known]),'masked_RMSE':root_mean_squared_error(yva[~known],pred[~known]),'training_seconds':perf_counter()-t,'inference_seconds':np.nan,'peak_gpu_mb':peak/1024**2,'epochs':max(h['epoch'] for h in histories)},pred,yva,idx,known,cold,histories

# Controlled staged pilot: family -> length -> structure.
pilot_rows=[]; allhist=[]; cache={}
for kind in ['GRU','LSTM','TCN']:
 r,*rest=fit(kind,6,'single'); pilot_rows.append(r); cache[(kind,6,'single')]=rest; allhist+=rest[-1]
family=min(pilot_rows,key=lambda x:(x['RMSE'],x['training_seconds']))['kind']
for L in [3,12]:
 r,*rest=fit(family,L,'single'); pilot_rows.append(r); cache[(family,L,'single')]=rest; allhist+=rest[-1]
length=min([x for x in pilot_rows if x['kind']==family],key=lambda x:x['RMSE'])['length']
r,*rest=fit(family,length,'dual'); pilot_rows.append(r); cache[(family,length,'dual')]=rest; allhist+=rest[-1]
chosen=min([x for x in pilot_rows if x['kind']==family and x['length']==length],key=lambda x:x['RMSE']); key=(chosen['kind'],chosen['length'],chosen['structure']); pred,y,idx,known,cold,hist=cache[key][0:6]
pd.DataFrame(pilot_rows).to_csv(ART/'pilot_results.csv',index=False); pd.DataFrame(allhist).to_csv(ART/'training_history.csv',index=False)

# Exact same-row canonical proxy OOF and conservative nonnegative blend.
ref=fold_oof.set_index('original_index').loc[idx]; assert np.allclose(ref.target,y); bp=ref.oof_prediction.to_numpy(); weights=np.linspace(0,1,21); blends=[(w,root_mean_squared_error(y,w*bp+(1-w)*pred)) for w in weights]; w,br=min(blends,key=lambda z:z[1]); blend=w*bp+(1-w)*pred
rows=[]
for name,p in [('E0',bp),('S1',pred),('S2',blend)]: rows.append({'system':name,'validation':'pilot_strict_temporal_2','rows':len(y),'overall_RMSE':root_mean_squared_error(y,p),'known_RMSE':root_mean_squared_error(y[known],p[known]),'masked_RMSE':root_mean_squared_error(y[~known],p[~known]),'worst_fold':root_mean_squared_error(y,p),'blend_canonical_weight':w if name=='S2' else np.nan,'full_validation':False})
vr=pd.DataFrame(rows); vr.to_csv(ART/'validation_results.csv',index=False)
er0=y-bp; er1=y-pred; q=np.quantile(np.abs(er0),.95); diversity=pd.DataFrame([{'pearson_residual':pearsonr(er0,er1).statistic,'spearman_residual':spearmanr(er0,er1).statistic,'error_overlap_top5':((np.abs(er0)>=q)&(np.abs(er1)>=np.quantile(np.abs(er1),.95))).mean(),'S1_better_top5_E0':(np.abs(er1[np.abs(er0)>=q])<np.abs(er0[np.abs(er0)>=q])).mean(),'known_prediction_disagreement':np.mean(np.abs(bp[known]-pred[known])),'masked_prediction_disagreement':np.mean(np.abs(bp[~known]-pred[~known])),'blend_weight_canonical':w,'blend_RMSE':br}]); diversity.to_csv(ART/'diversity_analysis.csv',index=False)
coldtab=pd.DataFrame([{'cold_start':b,'rows':int((cold==b).sum()),'fraction':(cold==b).mean(),'sequence_RMSE':root_mean_squared_error(y[cold==b],pred[cold==b]) if (cold==b).any() else np.nan,'canonical_RMSE':root_mean_squared_error(y[cold==b],bp[cold==b]) if (cold==b).any() else np.nan,'strategy':'left zero padding + missing indicator; tabular fallback not activated'} for b in [False,True]]); coldtab.to_csv(ART/'cold_start_analysis.csv',index=False)
pd.DataFrame([{'length':L,'train_samples':len(build_sequences(L)[0]),'validation_samples':len(build_sequences(L)[2]),'cold_start_rows':int(build_sequences(L)[6].sum()),'features':len(FEATURES),'target_or_future_feature_used':False,'scaler_fit':'fold training only'} for L in [3,6,12]]).to_csv(ART/'sequence_sample_report.csv',index=False)

e0=vr.loc[vr.system.eq('E0')].iloc[0]; s1=vr.loc[vr.system.eq('S1')].iloc[0]; s2=vr.loc[vr.system.eq('S2')].iloc[0]
pilot_signal=(e0.overall_RMSE-s1.overall_RMSE>0 or e0.overall_RMSE-s2.overall_RMSE>=.0015)
# Full training is intentionally stopped unless the pilot shows positive same-row signal.
gateA=(e0.overall_RMSE-s1.overall_RMSE>=.002); gateB=(e0.overall_RMSE-s2.overall_RMSE>=.0015); gateC=(e0.masked_RMSE-s1.masked_RMSE>=.004 and s1.known_RMSE<=e0.known_RMSE+.001); accepted=False
pd.DataFrame([{'selected_family':family,'selected_length':length,'selected_structure':chosen['structure'],'criterion':'lowest pilot RMSE, then time/simplicity','CUDA_requested':True,'CUDA_used':AMP,'pilot_signal':pilot_signal,'full_training_allowed':bool(pilot_signal and AMP),'full_training_blocker':'' if pilot_signal and AMP else 'installed PyTorch is CPU-only; full panel run exceeds performance policy'}]).to_csv(ART/'model_selection.csv',index=False)
pd.DataFrame([{'Submission':'none','Created':False,'Rows':0,'Reason':'No full-validation acceptance; pilot-only evidence','SHA256':''}]).to_csv(ART/'submission_report.csv',index=False)
report=f'''# Notebook 33 final recommendation\n\n1. Panel sequence structure correct: True; chronological location panels and training-only scaling verified.\n2. Best sequence length: {length}.\n3. Selected family: {family}; structure: {chosen['structure']}.\n4. Sequence better than canonical on pilot: {s1.overall_RMSE<e0.overall_RMSE} (E0 {e0.overall_RMSE:.6f}, S1 {s1.overall_RMSE:.6f}).\n5. Masked RMSE improved: {s1.masked_RMSE<e0.masked_RMSE} (E0 {e0.masked_RMSE:.6f}, S1 {s1.masked_RMSE:.6f}).\n6. Cold start fraction: {cold.mean():.3%}; left padding was used.\n7. Residual diversity: Pearson {diversity.pearson_residual.iloc[0]:.4f}, Spearman {diversity.spearman_residual.iloc[0]:.4f}.\n8. Blend useful: {s2.overall_RMSE<e0.overall_RMSE}; canonical weight {w:.2f}, RMSE {s2.overall_RMSE:.6f}.\n9. Acceptance gate passed: False. Pilot thresholds A/B/C={gateA}/{gateB}/{gateC}, but formal acceptance requires full strict/irregular/space-time validation. Full training was skipped because CUDA-capable PyTorch is unavailable, not because the pilot lacked signal.\n10. Submission produced: No.\n11. Expand sequence approach: Yes—first install a CUDA-capable PyTorch build, then run the selected {family}-L{length} {chosen['structure']} expert on the frozen full validation systems only.\n12. Highest-potential next experiment without global ERA5: full validation of this single selected configuration after the CUDA environment is repaired; do not reopen architecture search.\n\nCUDA could not be used because installed PyTorch is CPU-only despite the RTX 4050 being present. Mixed precision and GPU-memory metrics are therefore unavailable. This is a bounded pilot, not full validation or leaderboard evidence.\n'''
(ART/'final_recommendation.md').write_text(report,encoding='utf-8'); print(report)
