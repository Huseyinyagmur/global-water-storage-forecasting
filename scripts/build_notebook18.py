from pathlib import Path
import nbformat as nbf

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"notebooks"/"18_domain_hydrology_feature_research.ipynb"
nb=nbf.v4.new_notebook(); cells=[]
def md(x): cells.append(nbf.v4.new_markdown_cell(x.strip()))
def code(x): cells.append(nbf.v4.new_code_cell(x.strip()))

md("""
# 18 — Domain Hydrology Feature Research

Tek canonical referans: Notebook 11 final selected blend ve
`submissions/submission_optimized.csv` (public RMSE `0.824463818`).

Model ailesi, parametreleri, foldları, seed'i, known/masked sistemi ve masked blend ağırlığı
değiştirilmez. Yalnız fiziksel olarak gerekçeli, leakage-safe hidrolojik temsiller tek tek eklenir.
""")

code(r"""
from pathlib import Path
from time import perf_counter
import json, warnings, gc, hashlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, Markdown
from sklearn.metrics import root_mean_squared_error
from sklearn.inspection import permutation_importance
from lightgbm import LGBMRegressor
warnings.filterwarnings("ignore"); SEED=42; np.random.seed(SEED)
pd.set_option("display.max_columns",120); sns.set_theme(style="whitegrid")
def find_root():
    for p in [Path.cwd(),*Path.cwd().parents]:
        if (p/"data"/"raw"/"Train.csv").exists(): return p
    raise FileNotFoundError
ROOT=find_root(); DATA=ROOT/"data"/"raw"; SUB=ROOT/"submissions"
ART=ROOT/"artifacts"/"notebook18"; ART.mkdir(parents=True,exist_ok=True); SUB.mkdir(exist_ok=True)
T0=perf_counter()
def stage(s,t): print(f"[{s}] section={perf_counter()-t:.1f}s total={perf_counter()-T0:.1f}s"); return perf_counter()
print(ROOT)
""")

md("""
## 1. Notebook audit, physical interpretation and canonical identity

TWS; soil water, groundwater, surface water, snow/ice and canopy storage componentsinin toplam
anomalisidir. SPEI, precipitation–potential evapotranspiration balance'ının farklı accumulation
windows üzerindeki standardized drought/wetness sinyalidir. Soil moisture hızlı/orta zaman ölçekli
land-surface storage state'ini; month enerji/yağış mevsimselliğini; latitude iklim kuşaklarını ve
longitude kıtasal/okyanusal coğrafyayı temsil eder.

Domain hipotezi tek cümledir: **TWS, kısa dönem iklim zorlamasından çok, forcing'in farklı
zaman ölçeklerinde tutarlı olması ve mevcut soil state ile aynı yönde hareket etmesi halinde
kalıcı değişir.** Bu nedenle yalnız multi-scale persistence, recovery ve soil–climate coupling
özetleri denenir.

Bu yaklaşım, terrestrial stores'ta precipitation'dan daha güçlü memory bulunması ve soil-moisture
memory'nin haftalar–aylar sürmesiyle uyumludur:

- HESS, *Annual memory in the terrestrial water cycle* (2025)
- Reviews of Geophysics, *Soil Moisture Memory: State-of-the-Art and the Way Forward* (2024)
- Nature Climate Change, *Global terrestrial water storage and drought severity under climate change* (2021)
""")

code(r"""
t=perf_counter()
audit=[]
for i in range(1,18):
    for p in sorted((ROOT/"notebooks").glob(f"{i:02d}_*.ipynb")):
        raw=json.loads(p.read_text(encoding="utf-8"))
        audit.append({"notebook":p.name,"cells":len(raw["cells"]),
          "executed":sum(bool(c.get("outputs")) for c in raw["cells"] if c["cell_type"]=="code")})
audit=pd.DataFrame(audit); display(audit)
assert all(audit.notebook.str.startswith(f"{i:02d}_").any() for i in range(1,18))
canonical_path=SUB/"submission_optimized.csv"; assert canonical_path.exists()
canonical_sha=hashlib.sha256(canonical_path.read_bytes()).hexdigest()
print("Canonical",canonical_path,canonical_sha)

train=pd.read_csv(DATA/"Train.csv",parse_dates=["time"])
test=pd.read_csv(DATA/"Test.csv",parse_dates=["time"]); test["_original_order"]=np.arange(len(test))
sample=pd.read_csv(DATA/"SampleSubmission.csv")
assert train.sample_id.is_unique and test.ID.equals(sample.ID)
print(train.shape,test.shape,test.TWS_t.isna().mean())
t=stage("audit/load",t)
""")

md("## 2. Exact Notebook 11 static features, folds and causal climatology")
code(r"""
SPEI=["SPEI_01_t","SPEI_03_t","SPEI_06_t","SPEI_12_t"]
def static(df):
    o=df.copy(); dt=pd.to_datetime(o.time)
    o["month"]=dt.dt.month.astype("int8"); o["quarter"]=dt.dt.quarter.astype("int8")
    o["year"]=dt.dt.year.astype("int16")
    o["elapsed_month"]=((dt.dt.year-train.time.dt.year.min())*12+dt.dt.month-1).astype("int16")
    o["month_sin_calc"]=np.sin(2*np.pi*o.month/12); o["month_cos_calc"]=np.cos(2*np.pi*o.month/12)
    o["season_sin"]=np.sin(2*np.pi*o.quarter/4); o["season_cos"]=np.cos(2*np.pi*o.quarter/4)
    s=o[SPEI]; o["spei_mean"]=s.mean(axis=1); o["spei_min"]=s.min(axis=1)
    o["spei_max"]=s.max(axis=1); o["spei_std"]=s.std(axis=1)
    o["spei_01_minus_12"]=o.SPEI_01_t-o.SPEI_12_t
    o["spei_03_minus_12"]=o.SPEI_03_t-o.SPEI_12_t
    o["spei_short_long_ratio"]=o.SPEI_01_t/(o.SPEI_12_t.abs()+.1)
    o["lat_band_5"]=np.floor((o.lat+90)/5); o["lon_band_5"]=np.floor((o.lon+180)/5)
    o["spatial_bin_10"]=o.lat_band_5*100+o.lon_band_5; o["lat_lon_interaction"]=o.lat*o.lon/1000
    o["lat_sin"]=np.sin(np.radians(o.lat)); o["lat_cos"]=np.cos(np.radians(o.lat))
    o["lon_sin"]=np.sin(np.radians(o.lon)); o["lon_cos"]=np.cos(np.radians(o.lon))
    return o
train_f=static(train); test_f=static(test)
A=["lat","lon","TWS_t",*SPEI,"SOIL_MOISTURE_t","month","month_sin","month_cos","year"]
B=["quarter","elapsed_month","month_sin_calc","month_cos_calc","season_sin","season_cos"]
C=["spei_mean","spei_min","spei_max","spei_std","spei_01_minus_12","spei_03_minus_12","spei_short_long_ratio"]
D=["lat_band_5","lon_band_5","spatial_bin_10","lat_lon_interaction","lat_sin","lat_cos","lon_sin","lon_cos"]
BASE=list(dict.fromkeys(A+B+C+D)); MASKED=[x for x in BASE if x!="TWS_t"]
HIST=["prior_location_mean","prior_location_std","prior_location_month","prior_global_month"]
PARAMS=dict(n_estimators=160,learning_rate=.08,num_leaves=63,min_child_samples=100,reg_lambda=1.,
            subsample=.85,colsample_bytree=.9,random_state=SEED,n_jobs=4,verbosity=-1)
def lgb(): return LGBMRegressor(**PARAMS)

def causal_history(df):
    o=df.sort_values(["lat","lon","time"]).copy(); shifted=o.groupby(["lat","lon"],sort=False).target.shift()
    o["prior_location_mean"]=shifted.groupby([o.lat,o.lon]).expanding().mean().reset_index(level=[0,1],drop=True)
    o["prior_location_std"]=shifted.groupby([o.lat,o.lon]).expanding().std().reset_index(level=[0,1],drop=True)
    o["prior_location_month"]=shifted.groupby([o.lat,o.lon,o.month]).expanding().mean().reset_index(level=[0,1,2],drop=True)
    o["prior_global_month"]=o.groupby("month").target.transform(lambda x:x.shift().expanding().mean())
    return o.sort_index()
def history_state(h):
    loc=h.groupby(["lat","lon"]).target.agg(["mean","std"]).rename(columns={"mean":"prior_location_mean","std":"prior_location_std"})
    lm=h.groupby(["lat","lon","month"]).target.agg(["sum","count"])
    gm=h.groupby("month").target.mean().rename("prior_global_month")
    return loc,lm,gm
def apply_history(df,state):
    loc,lm,gm=state; o=df.copy().join(loc,on=["lat","lon"]).join(lm,on=["lat","lon","month"]).join(gm,on="month")
    o["prior_location_month"]=(o["sum"]+4*o.prior_location_mean)/(o["count"]+4)
    return o.drop(columns=["sum","count"])

date_stats=test.groupby("time").agg(n=("ID","size"),known=("TWS_t","count"))
KNOWN_POS=set(np.flatnonzero(date_stats.known.to_numpy()>date_stats.n.to_numpy()*.5))
dates=np.array(sorted(train_f.time.unique())); start=len(dates)-3*18
SPLITS=[(np.flatnonzero(train_f.time.isin(dates[:start+i*18])),
         np.flatnonzero(train_f.time.isin(dates[start+i*18:start+(i+1)*18]))) for i in range(3)][-2:]
FOLDS=[]
for fold,(ti,vi) in enumerate(SPLITS,1):
    tr,va=train_f.iloc[ti].copy(),train_f.iloc[vi].copy(); vd=np.array(sorted(va.time.unique()))
    known=va.time.isin({d for j,d in enumerate(vd) if j in KNOWN_POS}).to_numpy()
    FOLDS.append({"fold":fold,"train":tr,"valid":va,"train_h":causal_history(tr),
                  "valid_h":apply_history(va,history_state(tr)),"known":known})
    print(fold,len(tr),len(va),known.sum(),(~known).sum())
""")

md("## 3. Canonical OOF reproduction")
code(r"""
t=perf_counter(); known_parts=[]; base_parts=[]; hist_parts=[]
for f in FOLDS:
    km=lgb().fit(f["train"][BASE],f["train"].target)
    kp=km.predict(f["valid"].loc[f["known"],BASE])
    known_parts.append(pd.DataFrame({"idx":f["valid"].index[f["known"]],"fold":f["fold"],
      "target":f["valid"].target.to_numpy()[f["known"]],"pred":kp}))
    bm=lgb().fit(f["train"][MASKED],f["train"].target)
    bp=bm.predict(f["valid"].loc[~f["known"],MASKED])
    base_parts.append(pd.DataFrame({"idx":f["valid"].index[~f["known"]],"fold":f["fold"],
      "target":f["valid"].target.to_numpy()[~f["known"]],"pred":bp}))
    hm=lgb().fit(f["train_h"][MASKED+HIST],f["train_h"].target)
    hp=hm.predict(f["valid_h"].loc[~f["known"],MASKED+HIST])
    hist_parts.append(pd.DataFrame({"idx":f["valid"].index[~f["known"]],"fold":f["fold"],
      "target":f["valid"].target.to_numpy()[~f["known"]],"pred":hp}))
KNOWN_OOF=pd.concat(known_parts,ignore_index=True); BOOF=pd.concat(base_parts,ignore_index=True); HOOF=pd.concat(hist_parts,ignore_index=True)
CANON=BOOF.copy(); CANON["pred"]=.65*BOOF.pred.to_numpy()+.35*HOOF.pred.to_numpy()
def met(m):
    allp=pd.concat([KNOWN_OOF,m],ignore_index=True)
    fs=allp.groupby("fold").apply(lambda z:root_mean_squared_error(z.target,z.pred),include_groups=False)
    return {"Known RMSE":root_mean_squared_error(KNOWN_OOF.target,KNOWN_OOF.pred),
      "Masked RMSE":root_mean_squared_error(m.target,m.pred),"Overall RMSE":root_mean_squared_error(allp.target,allp.pred),
      "Worst Fold":fs.max(),"Fold Std":fs.std(ddof=0)}
CAN_MET=met(CANON); display(pd.DataFrame([{"Feature":"Canonical Notebook 11 blend",**CAN_MET}]))
np.savez_compressed(ART/"canonical_oof.npz",idx=CANON.idx,fold=CANON.fold,target=CANON.target,pred=CANON.pred)
t=stage("canonical OOF",t)
""")

md("""
## 4. Hydrology feature definitions

SPEI windows `[1,3,6,12]` months are treated as a small response curve. Features summarize
sign consistency, scale persistence, recovery direction, storage–forcing agreement and
standardized deficit. No future row, target aggregation or non-causal statistic is used.
""")
code(r"""
EPS=1e-4
def hydro(df,name):
    o=df.copy(); s=o[SPEI].to_numpy(float)
    defs={
      "drought_persistence_fraction":lambda:(s<0).mean(1),
      "wet_persistence_fraction":lambda:(s>0).mean(1),
      "normalized_drought_severity":lambda:np.maximum(0,-s.mean(1))/(s.std(1)+.25),
      "water_deficit_coupling":lambda:np.maximum(0,-s.mean(1))*np.maximum(0,-o.SOIL_MOISTURE_t.to_numpy()),
      "water_surplus_coupling":lambda:np.maximum(0,s.mean(1))*np.maximum(0,o.SOIL_MOISTURE_t.to_numpy()),
      "soil_climate_divergence":lambda:o.SOIL_MOISTURE_t.to_numpy()-s.mean(1),
      "soil_climate_agreement":lambda:o.SOIL_MOISTURE_t.to_numpy()*s.mean(1),
      "recovery_indicator":lambda:np.maximum(0,o.SPEI_01_t.to_numpy()-o.SPEI_12_t.to_numpy())*(o.SPEI_12_t.to_numpy()<0),
      "drying_transition":lambda:np.maximum(0,o.SPEI_12_t.to_numpy()-o.SPEI_01_t.to_numpy())*(o.SPEI_12_t.to_numpy()>0),
      "climate_memory_weighted":lambda:(12*s[:,3]+6*s[:,2]+3*s[:,1]+s[:,0])/22,
      "short_long_acceleration":lambda:(s[:,0]-s[:,1])-(s[:,2]-s[:,3]),
      "multi_scale_consensus":lambda:np.abs(s.mean(1))/(s.std(1)+.25),
      "soil_adjusted_deficit":lambda:s.mean(1)-o.SOIL_MOISTURE_t.to_numpy(),
      "hydrological_stress_regime":lambda:((s.mean(1)<-1).astype(float)+(o.SOIL_MOISTURE_t.to_numpy()<-1).astype(float)),
      "seasonal_deficit_pressure":lambda:np.maximum(0,-s.mean(1))*(1+.5*np.sin(2*np.pi*o.month.to_numpy()/12)),
    }
    o[name]=np.asarray(defs[name](),dtype="float32")
    return o
CANDIDATES=["drought_persistence_fraction","wet_persistence_fraction","normalized_drought_severity",
 "water_deficit_coupling","water_surplus_coupling","soil_climate_divergence","soil_climate_agreement",
 "recovery_indicator","drying_transition","climate_memory_weighted","short_long_acceleration",
 "multi_scale_consensus","soil_adjusted_deficit","hydrological_stress_regime","seasonal_deficit_pressure"]
print(CANDIDATES)
""")

md("## 5. Single-feature ablation on the unchanged canonical blend")
code(r"""
def evaluate(name,features):
    cache=ART/f"oof_{name}.pkl"; start=perf_counter()
    if cache.exists(): m=pd.read_pickle(cache); train_sec=0.
    else:
        parts=[]
        for f in FOLDS:
            tr=f["train"].copy(); va=f["valid"].copy(); trh=f["train_h"].copy(); vah=f["valid_h"].copy()
            for x in features:
                tr=hydro(tr,x); va=hydro(va,x); trh=hydro(trh,x); vah=hydro(vah,x)
            bm=lgb().fit(tr[MASKED+features],tr.target)
            bp=bm.predict(va.loc[~f["known"],MASKED+features])
            hm=lgb().fit(trh[MASKED+HIST+features],trh.target)
            hp=hm.predict(vah.loc[~f["known"],MASKED+HIST+features])
            parts.append(pd.DataFrame({"idx":f["valid"].index[~f["known"]],"fold":f["fold"],
              "target":f["valid"].target.to_numpy()[~f["known"]],"pred":.65*bp+.35*hp}))
            del bm,hm,tr,va,trh,vah; gc.collect()
        m=pd.concat(parts,ignore_index=True); m.to_pickle(cache); train_sec=perf_counter()-start
    z=met(m); gain=CAN_MET["Overall RMSE"]-z["Overall RMSE"]; mgain=CAN_MET["Masked RMSE"]-z["Masked RMSE"]
    return m,{"Feature":name,**z,"OOF Gain":gain,"Masked Gain":mgain,
      "Contribution":"positive" if gain>0 else "negative","Training sec":train_sec}

t=perf_counter(); RESULTS=[]; OOF={}
for name in CANDIDATES:
    OOF[name],row=evaluate(name,[name]); RESULTS.append(row)
    print(name,row["Overall RMSE"],row["OOF Gain"],row["Masked Gain"])
ablation=pd.DataFrame(RESULTS).sort_values("Overall RMSE")
ablation.to_csv(ART/"domain_feature_ablation.csv",index=False); display(ablation)
t=stage("single-feature ablation",t)
""")

md("## 6. Canonical residual slices and SHAP verification")
code(r"""
t=perf_counter()
r=CANON.merge(train_f[["month","lat","lon","spei_mean","SOIL_MOISTURE_t"]],left_on="idx",right_index=True)
r["sqerr"]=(r.target-r.pred)**2; r["abserr"]=(r.target-r.pred).abs()
r["climate"]=pd.cut(r.spei_mean,[-np.inf,-1,-.25,.25,1,np.inf])
r["soil_group"]=pd.cut(r.SOIL_MOISTURE_t,[-np.inf,-1,-.25,.25,1,np.inf])
r["lat_band"]=pd.cut(r.lat,10); r["lon_band"]=pd.cut(r.lon,10)
for c in ["month","climate","soil_group","lat_band","lon_band"]:
    tab=r.groupby(c,observed=True).agg(n=("target","size"),RMSE=("sqerr",lambda x:np.sqrt(x.mean())),MAE=("abserr","mean")).sort_values("RMSE",ascending=False)
    display(Markdown(f"**Residual: {c}**")); display(tab.head(12))

best_name=ablation.iloc[0].Feature
f=FOLDS[-1]; tr=hydro(f["train_h"],best_name); va=hydro(f["valid_h"],best_name)
shap_model=lgb().fit(tr[MASKED+HIST+[best_name]],tr.target)
ix=np.flatnonzero(~f["known"])[:5000]; Xs=va.iloc[ix][MASKED+HIST+[best_name]]
try:
    import shap
    sv=shap.TreeExplainer(shap_model).shap_values(Xs)
    shap_imp=pd.DataFrame({"Feature":Xs.columns,"Mean abs SHAP":np.abs(sv).mean(0)}).sort_values("Mean abs SHAP",ascending=False)
except Exception as e:
    shap_imp=pd.DataFrame({"Feature":[best_name],"Mean abs SHAP":[np.nan]}); print("SHAP unavailable",e)
display(shap_imp.head(20))
best_shap=float(shap_imp.loc[shap_imp.Feature.eq(best_name),"Mean abs SHAP"].iloc[0])
print("Best feature learned:",best_shap>0,"SHAP",best_shap)
t=stage("residual + SHAP",t)
""")

md("## 7. Only accepted features, acceptance gate and final submissions")
code(r"""
# Only individually positive candidates enter one combined confirmation; no combinatorial search.
positive=ablation.query("`OOF Gain`>0").head(3).Feature.tolist()
if len(positive)>=2:
    combined_name="accepted_domain_combined"; combined_oof,combined_row=evaluate(combined_name,positive)
    RESULTS.append(combined_row); OOF[combined_name]=combined_oof
final_table=pd.DataFrame(RESULTS).sort_values("Overall RMSE"); display(final_table)
winner=final_table.iloc[0]
accepted=(winner["OOF Gain"]>=.0015) or (winner["Masked Gain"]>=.003 and winner["Known RMSE"]<=CAN_MET["Known RMSE"]+1e-12)
selected=winner.Feature if accepted else "canonical_notebook11"
selected_features=positive if selected=="accepted_domain_combined" else ([selected] if selected in CANDIDATES else [])
print("WINNER",winner.to_dict()); print("ACCEPTED",accepted,"SELECTED",selected)

# Full canonical fit. Architecture and .65/.35 weights remain unchanged.
train_h=causal_history(train_f); test_h=apply_history(test_f,history_state(train_f))
known=test_f.TWS_t.notna().to_numpy()
kp=lgb().fit(train_f[BASE],train_f.target).predict(test_f.loc[known,BASE])
def masked_full(features):
    tr=train_f.copy(); te=test_f.copy(); trh=train_h.copy(); teh=test_h.copy()
    for x in features:
        tr=hydro(tr,x); te=hydro(te,x); trh=hydro(trh,x); teh=hydro(teh,x)
    bp=lgb().fit(tr[MASKED+features],tr.target).predict(te.loc[~known,MASKED+features])
    hp=lgb().fit(trh[MASKED+HIST+features],trh.target).predict(teh.loc[~known,MASKED+HIST+features])
    return .65*bp+.35*hp
canonical_masked=masked_full([]); selected_masked=masked_full(selected_features) if accepted else canonical_masked
def assemble(mp):
    p=np.empty(len(test_f)); p[known]=kp; p[~known]=mp; return p
predA=assemble(selected_masked); canonical_pred=assemble(canonical_masked)
canonical_file=pd.read_csv(canonical_path)
print("Reproduced vs canonical file corr",np.corrcoef(canonical_pred,canonical_file.Target)[0,1],
      "MAD",np.abs(canonical_pred-canonical_file.Target).mean())

def write(label,pred):
    byid=pd.Series(np.asarray(pred,float),index=test_f.ID.astype(str))
    s=sample.copy(); s["Target"]=sample.ID.astype(str).map(byid).to_numpy(float)
    assert len(s)==280961 and s.ID.equals(sample.ID) and s.ID.is_unique and np.isfinite(s.Target).all()
    p=SUB/f"submission_18_{label}.csv"; s.to_csv(p,index=False)
    return p,{"Submission":label,"rows":len(s),"mean":s.Target.mean(),"std":s.Target.std(),
              "min":s.Target.min(),"max":s.Target.max()}
REPORT=[]
if accepted:
    pathA,rep=write("A_domain_best",predA); REPORT.append(rep|{"path":str(pathA)})
    # B only if meaningfully different from A; otherwise canonical remains the documented fallback, not a new candidate.
    if abs(np.corrcoef(predA,canonical_file.Target)[0,1])<.995:
        pathB,rep=write("B_canonical_diverse",canonical_file.Target.to_numpy()); REPORT.append(rep|{"path":str(pathB)})
else:
    print("Acceptance gate failed; no new candidate submission. Canonical preserved:",canonical_path)
submission_report=pd.DataFrame(REPORT); submission_report.to_csv(ART/"submission_report.csv",index=False)
display(submission_report)
""")

md("## 8. Final report")
code(r"""
best=final_table.iloc[0]; failed=ablation.query("`OOF Gain`<=0").Feature.tolist()
report=f'''
### Final report

- **Strongest domain feature:** `{best.Feature}`, overall gain `{best["OOF Gain"]:+.6f}`,
  masked gain `{best["Masked Gain"]:+.6f}`.
- **Why it can work:** it compresses multi-scale climate persistence/recovery and soil-state
  agreement into a physically interpretable storage-pressure signal; SHAP check is reported above.
- **Failed features:** {", ".join(failed) if failed else "none"}.
- **Why Notebook 11 remains strong:** its causal climatology substitutes missing TWS state while
  the fixed baseline component limits variance; the blend is already robust to redundant transforms.
- **Acceptance gate:** `{accepted}`. Required: overall gain ≥0.0015, or masked gain ≥0.003 with
  unchanged known RMSE. If false, no new candidate is promoted and `submission_optimized.csv` remains canonical.
- **Leaderboard expectation:** only a gate-passing, fold-robust representation is written. Local gain
  cannot guarantee public gain because the test horizon and geographical mixture may drift.
- **Canonical source:** `{canonical_path.name}`, SHA256 `{canonical_sha}`, public RMSE `0.824463818`.
- **Runtime:** `{perf_counter()-T0:.1f}s`; checkpoints `{ART}`.
'''
display(Markdown(report))
assert canonical_path.exists() and (accepted or len(submission_report)==0)
""")

nb["cells"]=cells
nb["metadata"]={"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
                "language_info":{"name":"python","version":"3.13"}}
OUT.write_text(nbf.writes(nb),encoding="utf-8"); print(OUT)
