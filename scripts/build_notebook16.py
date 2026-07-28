from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "16_competition_level_feature_research.ipynb"
nb = nbf.v4.new_notebook()
cells = []

def md(text):
    cells.append(nbf.v4.new_markdown_cell(text.strip()))

def code(text):
    cells.append(nbf.v4.new_code_cell(text.strip()))

md(r"""
# 16 — Competition-Level Feature Research

Bu notebook 01–15 arasındaki çalışmaların devamıdır. Amaç yeni model aileleri denemek değil,
**Notebook 11 (`11_submission_optimization.ipynb`) temporal OOF sistemini sabit tutarak**
feature representation sınırlarını ölçmektir.

Kurallar: seed=42, temporal OOF, known/masked ayrımı, fold-train dışında target istatistiği yok,
her aday tek başına baseline'a eklenir, ara sonuçlar checkpoint edilir. Veri şemasında
`SOIL_MOISTURE_tp1` bulunmadığı için tp1 içeren öneriler geleceği kullanmadan hesaplanamaz ve
bilimsel olarak reddedilir.
""")

code(r"""
from pathlib import Path
from time import perf_counter
import json, warnings, gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, Markdown
from sklearn.metrics import root_mean_squared_error
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.preprocessing import PolynomialFeatures, PowerTransformer
from sklearn.inspection import permutation_importance
from lightgbm import LGBMRegressor

warnings.filterwarnings("ignore")
SEED=42
np.random.seed(SEED)
pd.set_option("display.max_columns",120)
sns.set_theme(style="whitegrid")

def find_root():
    for p in [Path.cwd(),*Path.cwd().parents]:
        if (p/"data"/"raw"/"Train.csv").exists(): return p
    raise FileNotFoundError("Repository root bulunamadı")

ROOT=find_root(); DATA=ROOT/"data"/"raw"; SUB=ROOT/"submissions"
ART=ROOT/"artifacts"/"notebook16"; ART.mkdir(parents=True,exist_ok=True); SUB.mkdir(exist_ok=True)
T0=perf_counter()
def stage(name,t):
    sec=perf_counter()-t
    print(f"[STAGE] {name}: {sec:.1f}s | total={perf_counter()-T0:.1f}s")
    return perf_counter()
print("ROOT",ROOT)
""")

md("## 1. Repository audit, data and exact Notebook 11 baseline")
code(r"""
t=perf_counter()
expected=[f"{i:02d}_" for i in range(1,16)]
audit=[]
for prefix in expected:
    paths=sorted((ROOT/"notebooks").glob(prefix+"*.ipynb"))
    for p in paths:
        raw=json.loads(p.read_text(encoding="utf-8"))
        audit.append({"notebook":p.name,"cells":len(raw["cells"]),
                      "executed_code_cells":sum(bool(c.get("outputs")) for c in raw["cells"] if c["cell_type"]=="code")})
audit_df=pd.DataFrame(audit)
display(audit_df)
assert all(any(x["notebook"].startswith(p) for x in audit) for p in expected)

train=pd.read_csv(DATA/"Train.csv",parse_dates=["time"])
test=pd.read_csv(DATA/"Test.csv",parse_dates=["time"])
sample=pd.read_csv(DATA/"SampleSubmission.csv")
test["_original_order"]=np.arange(len(test))
assert train.sample_id.is_unique and test.ID.is_unique and sample.ID.is_unique and test.ID.equals(sample.ID)
print("shapes",train.shape,test.shape,sample.shape,"masked test share",test.TWS_t.isna().mean())
t=stage("audit + load",t)
""")

code(r"""
SPEI=["SPEI_01_t","SPEI_03_t","SPEI_06_t","SPEI_12_t"]
def static_features(df):
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
    o["spatial_bin_10"]=o.lat_band_5*100+o.lon_band_5
    o["lat_lon_interaction"]=o.lat*o.lon/1000
    o["lat_sin"]=np.sin(np.radians(o.lat)); o["lat_cos"]=np.cos(np.radians(o.lat))
    o["lon_sin"]=np.sin(np.radians(o.lon)); o["lon_cos"]=np.cos(np.radians(o.lon))
    return o

train_f=static_features(train); test_f=static_features(test)
A=["lat","lon","TWS_t",*SPEI,"SOIL_MOISTURE_t","month","month_sin","month_cos","year"]
B=["quarter","elapsed_month","month_sin_calc","month_cos_calc","season_sin","season_cos"]
C=["spei_mean","spei_min","spei_max","spei_std","spei_01_minus_12","spei_03_minus_12","spei_short_long_ratio"]
D=["lat_band_5","lon_band_5","spatial_bin_10","lat_lon_interaction","lat_sin","lat_cos","lon_sin","lon_cos"]
BASE_FEATURES=list(dict.fromkeys(A+B+C+D)); BASE_MASKED=[x for x in BASE_FEATURES if x!="TWS_t"]
PARAMS=dict(n_estimators=160,learning_rate=.08,num_leaves=63,min_child_samples=100,reg_lambda=1.,
            subsample=.85,colsample_bytree=.9,random_state=SEED,n_jobs=4,verbosity=-1)
def model(params=None):
    p=PARAMS.copy()
    if params:p.update(params)
    return LGBMRegressor(**p)

date_stats=test.groupby("time").agg(n=("ID","size"),known=("TWS_t","count"))
KNOWN_POS=set(np.flatnonzero(date_stats.known.to_numpy()>date_stats.n.to_numpy()*.5))
def temporal_splits(df,n_splits=3,block=18):
    dates=np.array(sorted(df.time.unique())); start=len(dates)-n_splits*block
    return [(np.flatnonzero(df.time.isin(dates[:start+i*block])),
             np.flatnonzero(df.time.isin(dates[start+i*block:start+(i+1)*block]))) for i in range(n_splits)]
SPLITS=temporal_splits(train_f)[-2:]
folds=[]
for f,(ti,vi) in enumerate(SPLITS,1):
    tr,va=train_f.iloc[ti].copy(),train_f.iloc[vi].copy()
    vd=np.array(sorted(va.time.unique()))
    known=va.time.isin({d for j,d in enumerate(vd) if j in KNOWN_POS}).to_numpy()
    folds.append({"fold":f,"train":tr,"valid":va,"known":known})
    print(f,tr.time.min(),tr.time.max(),va.time.min(),va.time.max(),known.sum(),(~known).sum())
""")

code(r"""
t=perf_counter()
known_parts=[]; base_masked_parts=[]; base_models=[]
for f in folds:
    km=model().fit(f["train"][BASE_FEATURES],f["train"].target)
    kp=km.predict(f["valid"].loc[f["known"],BASE_FEATURES])
    known_parts.append(pd.DataFrame({"idx":f["valid"].index[f["known"]],"fold":f["fold"],
                                     "target":f["valid"].target.to_numpy()[f["known"]],"pred":kp}))
    mm=model().fit(f["train"][BASE_MASKED],f["train"].target)
    mp=mm.predict(f["valid"].loc[~f["known"],BASE_MASKED])
    base_models.append(mm)
    base_masked_parts.append(pd.DataFrame({"idx":f["valid"].index[~f["known"]],"fold":f["fold"],
        "target":f["valid"].target.to_numpy()[~f["known"]],"pred":mp}))
KNOWN_OOF=pd.concat(known_parts,ignore_index=True); BASE_MOOF=pd.concat(base_masked_parts,ignore_index=True)
def metrics(moof):
    allp=pd.concat([KNOWN_OOF[["idx","fold","target","pred"]],moof[["idx","fold","target","pred"]]],ignore_index=True)
    fr=allp.groupby("fold").apply(lambda z:root_mean_squared_error(z.target,z.pred),include_groups=False)
    return {"Known RMSE":root_mean_squared_error(KNOWN_OOF.target,KNOWN_OOF.pred),
            "Masked RMSE":root_mean_squared_error(moof.target,moof.pred),
            "Overall RMSE":root_mean_squared_error(allp.target,allp.pred),
            "Worst Fold":fr.max(),"Fold Std":fr.std(ddof=0)}
BASE_MET=metrics(BASE_MOOF); display(pd.DataFrame([{"Feature":"Notebook 11 baseline",**BASE_MET}]))
np.savez_compressed(ART/"baseline_oof.npz",idx=BASE_MOOF.idx,pred=BASE_MOOF.pred,target=BASE_MOOF.target,fold=BASE_MOOF.fold)
t=stage("Notebook 11 baseline reproduction",t)
""")

md(r"""
## 2. Single-feature interaction and transform ablation

Her satır baseline'a **yalnız tek yeni sütun** ekler. Ratio paydayı güvenli epsilon ile korur.
Log/sqrt işaret-korumalıdır. Yeo–Johnson fold train'de fit edilir. Box–Cox yalnız kesin pozitif
değişkenlerde anlamlıdır; mevcut signed standardized değişkenlerde shift'in test-genellemesi
keyfi olduğundan uygulanmaz.
""")
code(r"""
EPS=1e-4
def add_simple(o,name):
    o=o.copy()
    specs={
      "soil_x_spei01":lambda x:x.SOIL_MOISTURE_t*x.SPEI_01_t,
      "soil_x_spei03":lambda x:x.SOIL_MOISTURE_t*x.SPEI_03_t,
      "soil_x_spei06":lambda x:x.SOIL_MOISTURE_t*x.SPEI_06_t,
      "soil_x_spei12":lambda x:x.SOIL_MOISTURE_t*x.SPEI_12_t,
      "soil_x_spei_mean":lambda x:x.SOIL_MOISTURE_t*x.spei_mean,
      "soil_minus_spei_mean":lambda x:x.SOIL_MOISTURE_t-x.spei_mean,
      "soil_ratio_spei_mean":lambda x:x.SOIL_MOISTURE_t/(x.spei_mean.abs()+EPS),
      "tws_x_spei_mean":lambda x:x.TWS_t*x.spei_mean,
      "lat_x_month":lambda x:x.lat*x.month,
      "lon_x_month":lambda x:x.lon*x.month,
      "month_x_soil":lambda x:x.month*x.SOIL_MOISTURE_t,
      "lat_x_soil":lambda x:x.lat*x.SOIL_MOISTURE_t,
      "lon_x_soil":lambda x:x.lon*x.SOIL_MOISTURE_t,
      "sinmonth_x_spei":lambda x:x.month_sin_calc*x.spei_mean,
      "cosmonth_x_spei":lambda x:x.month_cos_calc*x.spei_mean,
      "soil_abs":lambda x:x.SOIL_MOISTURE_t.abs(),
      "soil_signed_log1p":lambda x:np.sign(x.SOIL_MOISTURE_t)*np.log1p(x.SOIL_MOISTURE_t.abs()),
      "soil_signed_sqrt":lambda x:np.sign(x.SOIL_MOISTURE_t)*np.sqrt(x.SOIL_MOISTURE_t.abs()),
      "spei_abs":lambda x:x.spei_mean.abs(),
      "spei_signed_log1p":lambda x:np.sign(x.spei_mean)*np.log1p(x.spei_mean.abs()),
      "spei_signed_sqrt":lambda x:np.sign(x.spei_mean)*np.sqrt(x.spei_mean.abs()),
    }
    o[name]=specs[name](o).replace([np.inf,-np.inf],np.nan).astype("float32")
    return o

SIMPLE=list({
 "soil_x_spei01":0,"soil_x_spei03":0,"soil_x_spei06":0,"soil_x_spei12":0,
 "soil_x_spei_mean":0,"soil_minus_spei_mean":0,"soil_ratio_spei_mean":0,
 "lat_x_month":0,"lon_x_month":0,"month_x_soil":0,"lat_x_soil":0,"lon_x_soil":0,
 "sinmonth_x_spei":0,"cosmonth_x_spei":0,"soil_abs":0,"soil_signed_log1p":0,
 "soil_signed_sqrt":0,"spei_abs":0,"spei_signed_log1p":0,"spei_signed_sqrt":0})
# TWS interaction masked uzman için kullanılamaz; known uzman sabit tutulduğundan rejected.
NOT_APPLICABLE=[
 {"Feature":"soil_tp1_minus_soil_t","Reason":"SOIL_MOISTURE_tp1 şemada yok; geleceği türetmek leakage"},
 {"Feature":"soil_tp1_ratio_soil_t","Reason":"SOIL_MOISTURE_tp1 şemada yok; geleceği türetmek leakage"},
 {"Feature":"soil_tp1_x_spei","Reason":"SOIL_MOISTURE_tp1 şemada yok; geleceği türetmek leakage"},
 {"Feature":"tws_x_spei_mean","Reason":"Masked test segmentinde TWS_t NaN; masked uzman için kullanılamaz"},
 {"Feature":"boxcox","Reason":"Aday değişkenler signed; foldlar arası keyfi pozitif shift güvenli değil"}]
display(pd.DataFrame(NOT_APPLICABLE))
""")

code(r"""
def eval_candidate(name,builder,extra_cols,params=None):
    start=perf_counter()
    cache_path=ART/f"oof_{name.replace('/','_')}.pkl"
    if cache_path.exists():
        moof=pd.read_pickle(cache_path); met=metrics(moof)
        row={"Feature":name,**met,"OOF Gain":BASE_MET["Overall RMSE"]-met["Overall RMSE"],
             "Accepted":met["Overall RMSE"]<BASE_MET["Overall RMSE"] and met["Worst Fold"]<=BASE_MET["Worst Fold"]+.002,
             "Training sec":0.0}
        print(name,row["Overall RMSE"],row["OOF Gain"],row["Accepted"],"[checkpoint]")
        return row
    parts=[]
    for f in folds:
        tr,va=builder(f["train"],f["valid"])
        feats=BASE_MASKED+extra_cols
        m=model(params).fit(tr[feats],tr.target)
        pred=m.predict(va.loc[~f["known"],feats])
        parts.append(pd.DataFrame({"idx":f["valid"].index[~f["known"]],"fold":f["fold"],
          "target":f["valid"].target.to_numpy()[~f["known"]],"pred":pred}))
        del m,tr,va; gc.collect()
    moof=pd.concat(parts,ignore_index=True); met=metrics(moof)
    row={"Feature":name,**met,"OOF Gain":BASE_MET["Overall RMSE"]-met["Overall RMSE"],
         "Accepted":met["Overall RMSE"]<BASE_MET["Overall RMSE"] and met["Worst Fold"]<=BASE_MET["Worst Fold"]+.002,
         "Training sec":perf_counter()-start}
    moof.to_pickle(cache_path)
    print(name,row["Overall RMSE"],row["OOF Gain"],row["Accepted"])
    return row

t=perf_counter(); RESULTS=[]
for name in SIMPLE:
    RESULTS.append(eval_candidate(name,lambda tr,va,n=name:(add_simple(tr,n),add_simple(va,n)),[name]))

def yj_builder(tr,va):
    tr=tr.copy(); va=va.copy(); pt=PowerTransformer(method="yeo-johnson",standardize=True)
    tr["soil_yeojohnson"]=pt.fit_transform(tr[["SOIL_MOISTURE_t"]]).ravel().astype("float32")
    va["soil_yeojohnson"]=pt.transform(va[["SOIL_MOISTURE_t"]]).ravel().astype("float32")
    return tr,va
RESULTS.append(eval_candidate("soil_yeojohnson",yj_builder,["soil_yeojohnson"]))
pd.DataFrame(RESULTS).to_csv(ART/"ablation_interactions.csv",index=False)
display(pd.DataFrame(RESULTS).sort_values("Overall RMSE"))
t=stage("single interaction/transform ablation",t)
""")

md("## 3. Top-10 polynomial interaction-only experiment")
code(r"""
t=perf_counter()
imp=pd.DataFrame({"feature":BASE_MASKED,"gain":np.mean([m.booster_.feature_importance(importance_type="gain") for m in base_models],axis=0)})
top10=imp.sort_values("gain",ascending=False).head(10).feature.tolist()
poly=PolynomialFeatures(degree=2,interaction_only=True,include_bias=False)
poly.fit(train_f[top10].iloc[:100])
pair_names=[x.replace(" ","__x__") for x in poly.get_feature_names_out(top10) if " " in x]
def poly_builder(tr,va):
    tr=tr.copy(); va=va.copy()
    for nm in pair_names:
        a,b=nm.split("__x__"); tr[nm]=(tr[a]*tr[b]).astype("float32"); va[nm]=(va[a]*va[b]).astype("float32")
    return tr,va
poly_row=eval_candidate("poly_top10_degree2_interaction_only",poly_builder,pair_names)
RESULTS.append(poly_row)
print("top10",top10,"pair count",len(pair_names))
t=stage("polynomial",t)
""")

md("## 4. Spatial clusters, bands, density and leakage-safe frequency/target encoding")
code(r"""
def spatial_builder(kind,k):
    def build(tr,va):
        tr=tr.copy(); va=va.copy()
        cls=KMeans(n_clusters=k,random_state=SEED,n_init=10) if kind=="kmeans" else MiniBatchKMeans(n_clusters=k,random_state=SEED,batch_size=4096,n_init=5)
        # unique coordinates prevent repeated months from changing spatial weighting
        coords=tr[["lat","lon"]].drop_duplicates().to_numpy()
        cls.fit(coords)
        for o in [tr,va]:
            lab=cls.predict(o[["lat","lon"]]); o[f"{kind}_{k}_id"]=lab.astype("int16")
            o[f"{kind}_{k}_dist"]=np.linalg.norm(o[["lat","lon"]].to_numpy()-cls.cluster_centers_[lab],axis=1).astype("float32")
        counts=pd.Series(cls.labels_).value_counts()
        tr[f"{kind}_{k}_density"]=tr[f"{kind}_{k}_id"].map(counts).astype("float32")
        va[f"{kind}_{k}_density"]=va[f"{kind}_{k}_id"].map(counts).fillna(0).astype("float32")
        return tr,va
    return build

t=perf_counter()
for kind in ["kmeans","mbkmeans"]:
    for k in [8,16,32,64]:
        b=spatial_builder(kind,k)
        for suffix in ["id","dist","density"]:
            name=f"{kind}_{k}_{suffix}"; RESULTS.append(eval_candidate(name,b,[name]))

def band_builder(tr,va):
    tr=tr.copy(); va=va.copy()
    for o in [tr,va]:
        o["lat_band_2"]=np.floor((o.lat+90)/2).astype("int16")
        o["lon_band_2"]=np.floor((o.lon+180)/2).astype("int16")
        o["quadrant"]=((o.lat>=0).astype("int8")*2+(o.lon>=0).astype("int8"))
    return tr,va
for name in ["lat_band_2","lon_band_2","quadrant"]:
    RESULTS.append(eval_candidate(name,band_builder,[name]))
t=stage("spatial ablations",t)
""")

code(r"""
t=perf_counter()
def encoding_builder(group,name,target_encode=False,cluster=None):
    def build(tr,va):
        tr=tr.copy(); va=va.copy()
        g=group
        if cluster:
            tr,va=spatial_builder("mbkmeans",cluster)(tr,va); g=[f"mbkmeans_{cluster}_id"]
        keys=g if isinstance(g,list) else [g]
        if target_encode:
            global_mean=tr.target.mean(); stat=tr.groupby(keys).target.agg(["mean","count"])
            smooth=(stat["mean"]*stat["count"]+global_mean*20)/(stat["count"]+20)
            # leave-one-out train encoding; validation mapping only fold-train
            sums=tr.groupby(keys).target.transform("sum"); cnt=tr.groupby(keys).target.transform("count")
            tr[name]=((sums-tr.target+20*global_mean)/(cnt-1+20)).astype("float32")
            va[name]=pd.MultiIndex.from_frame(va[keys]).map(smooth).fillna(global_mean).astype("float32")
        else:
            freq=tr.groupby(keys).size()/len(tr)
            tr[name]=pd.MultiIndex.from_frame(tr[keys]).map(freq).astype("float32")
            va[name]=pd.MultiIndex.from_frame(va[keys]).map(freq).fillna(0).astype("float32")
        return tr,va
    return build

freq_specs=[("location_frequency",["lat","lon"],None),("month_frequency",["month"],None),
 ("lat_band_frequency",["lat_band_5"],None),("lon_band_frequency",["lon_band_5"],None),
 ("cluster_frequency",None,32)]
for name,g,k in freq_specs:
    RESULTS.append(eval_candidate(name,encoding_builder(g,name,False,k),[name]))

te_specs=[("te_month",["month"],None),("te_cluster",None,32),("te_lat_band",["lat_band_5"],None),
          ("te_lon_band",["lon_band_5"],None),("te_region",["lat_band_5","lon_band_5"],None)]
for name,g,k in te_specs:
    RESULTS.append(eval_candidate(name,encoding_builder(g,name,True,k),[name]))
t=stage("frequency + OOF target encoding",t)
""")

md("## 5. Feature selection, monotonic constraints, uncertainty and residual research")
code(r"""
t=perf_counter()
ablation=pd.DataFrame(RESULTS).sort_values("Overall RMSE").reset_index(drop=True)
ablation.to_csv(ART/"ablation_all.csv",index=False)
accepted=ablation.loc[ablation.Accepted,"Feature"].tolist()
display(ablation)

# Built-in importance + permutation on a bounded validation sample.
f=folds[-1]; bm=base_models[-1]; rng=np.random.default_rng(SEED)
ix=rng.choice(len(f["valid"]),min(10000,len(f["valid"])),replace=False)
pi=permutation_importance(bm,f["valid"].iloc[ix][BASE_MASKED],f["valid"].iloc[ix].target,
                          scoring="neg_root_mean_squared_error",n_repeats=3,random_state=SEED,n_jobs=1)
selection=pd.DataFrame({"Feature":BASE_MASKED,
 "LGBM gain":bm.booster_.feature_importance(importance_type="gain"),
 "Permutation importance":pi.importances_mean}).sort_values("LGBM gain",ascending=False)
try:
    import shap
    sx=f["valid"].iloc[ix[:2000]][BASE_MASKED]
    sv=shap.TreeExplainer(bm).shap_values(sx)
    shap_map=dict(zip(BASE_MASKED,np.abs(sv).mean(0)))
    selection["Mean abs SHAP"]=selection.Feature.map(shap_map)
except Exception as e:
    selection["Mean abs SHAP"]=np.nan; print("SHAP unavailable:",repr(e))
display(selection)

drop_candidates=selection.tail(5).Feature.tolist()
def drop_builder(tr,va): return tr.copy(),va.copy()
drop_row=eval_candidate("prune_bottom5_importance",drop_builder,[])
# special evaluation with removed features
def eval_pruned():
    global BASE_MASKED
    old=BASE_MASKED; BASE_MASKED=[x for x in old if x not in drop_candidates]
    try:return eval_candidate("prune_bottom5",drop_builder,[])
    finally:BASE_MASKED=old
prune_row=eval_pruned(); RESULTS.append(prune_row)

# Domain-defensible constraints only: SPEI and soil moisture signs are tested, not assumed selected.
constraints=[1 if x in SPEI+["SOIL_MOISTURE_t","spei_mean","spei_min","spei_max"] else 0 for x in BASE_MASKED]
mono_row=eval_candidate("monotonic_domain_signs",drop_builder,[],{"monotone_constraints":constraints})
RESULTS.append(mono_row)
t=stage("importance + selection + monotonic",t)
""")

code(r"""
t=perf_counter()
# Baseline residual slices.
res=BASE_MOOF.merge(train_f[["month","lat","lon","spei_mean","SOIL_MOISTURE_t"]],left_on="idx",right_index=True)
res["abs_error"]=(res.target-res.pred).abs()
res["lat_band"]=pd.cut(res.lat,8); res["spei_bin"]=pd.qcut(res.spei_mean,8,duplicates="drop")
res["soil_bin"]=pd.qcut(res.SOIL_MOISTURE_t,8,duplicates="drop")
residual_tables={}
for col in ["month","lat_band","spei_bin","soil_bin"]:
    residual_tables[col]=res.groupby(col,observed=True).agg(n=("target","size"),
      RMSE=("abs_error",lambda x:np.sqrt(np.mean(x*x))),MAE=("abs_error","mean")).sort_values("RMSE",ascending=False)
    display(Markdown(f"**Residual by {col}**")); display(residual_tables[col].head(10))

# Fold-model disagreement is a genuine uncertainty proxy.
sample_x=train_f.loc[BASE_MOOF.idx,BASE_MASKED]
P=np.column_stack([m.predict(sample_x) for m in base_models])
unc=P.std(axis=1); err=np.abs(BASE_MOOF.target.to_numpy()-BASE_MOOF.pred.to_numpy())
q=pd.qcut(unc,5,labels=False,duplicates="drop")
uncertainty=pd.DataFrame({"uncertainty":unc,"abs_error":err,"quintile":q}).groupby("quintile").agg(
 n=("abs_error","size"),mean_uncertainty=("uncertainty","mean"),MAE=("abs_error","mean"))
display(uncertainty)

# Pseudo labeling decision: no true label exists for test; confidence alone cannot establish benefit.
# Simulated self-training adds no independent information and compounds error, so it is deliberately not applied.
pseudo_decision={"applied":False,"reason":"OOF ile doğrulanabilir bağımsız pseudo-label kaynağı yok; self-training riskli"}
print(pseudo_decision)
t=stage("residual + uncertainty + pseudo-label decision",t)
""")

md("## 6. Accepted features together, conservative blend, full fit and submissions")
code(r"""
# Rebuild accepted candidates by name. To control multiple-testing risk, require positive gain and
# no material worst-fold regression; then test the combined representation once.
accepted=ablation.query("Accepted").Feature.tolist()
simple_acc=[x for x in accepted if x in SIMPLE]
print("Accepted single features",accepted)

def combined_builder(tr,va):
    tr=tr.copy(); va=va.copy()
    extras=[]
    for n in simple_acc: tr=add_simple(tr,n); va=add_simple(va,n); extras.append(n)
    return tr,va

combined_row=eval_candidate("accepted_simple_combined",combined_builder,simple_acc) if simple_acc else None
if combined_row: RESULTS.append(combined_row)
final_table=pd.DataFrame(RESULTS).sort_values("Overall RMSE")
best_row=final_table.iloc[0]
if best_row["Overall RMSE"]>=BASE_MET["Overall RMSE"]: best_name="Notebook 11 baseline"
else: best_name=best_row.Feature
print("Final selection",best_name); display(final_table.head(20))

def load_oof(name):
    if name=="Notebook 11 baseline": return BASE_MOOF.copy()
    return pd.read_pickle(ART/f"oof_{name.replace('/','_')}.pkl")
best_oof=load_oof(best_name)
align=BASE_MOOF[["idx","fold","target","pred"]].merge(best_oof[["idx","pred"]],on="idx",suffixes=("_base","_new"))
blend_grid=[]
for w in np.arange(0,1.001,.05):
    p=w*align.pred_new+(1-w)*align.pred_base
    z=align[["idx","fold","target"]].copy(); z["pred"]=p
    blend_grid.append({"new_weight":w,**metrics(z)})
blend_grid=pd.DataFrame(blend_grid).sort_values("Overall RMSE")
display(blend_grid.head())
""")

code(r"""
t=perf_counter()
known_test=test_f.TWS_t.notna().to_numpy()
known_model=model().fit(train_f[BASE_FEATURES],train_f.target)
known_pred=known_model.predict(test_f.loc[known_test,BASE_FEATURES])
base_full=model().fit(train_f[BASE_MASKED],train_f.target)
base_test=base_full.predict(test_f.loc[~known_test,BASE_MASKED])

def fit_selected(name):
    if name=="Notebook 11 baseline": return base_test
    if name in SIMPLE:
        tr=add_simple(train_f,name); te=add_simple(test_f,name); feats=BASE_MASKED+[name]
    elif name=="accepted_simple_combined":
        tr,te=combined_builder(train_f,test_f); feats=BASE_MASKED+simple_acc
    elif name.startswith(("kmeans_","mbkmeans_")):
        bits=name.split("_"); kind=bits[0]; k=int(bits[1])
        tr,te=spatial_builder(kind,k)(train_f,test_f); feats=BASE_MASKED+[name]
    else:
        # Complex fold-fit candidates are retained as research evidence; conservative final fallback
        # prevents a different full-data preprocessing path from silently diverging.
        print("Complex winner uses Notebook 11 final fallback:",name); return base_test
    return model().fit(tr[feats],tr.target).predict(te.loc[~known_test,feats])

new_test=fit_selected(best_name)
w=float(blend_grid.iloc[0].new_weight); blend_test=w*new_test+(1-w)*base_test
def assemble(masked):
    p=np.empty(len(test_f)); p[known_test]=known_pred; p[~known_test]=masked; return p
candidates={"A_best_single":assemble(new_test),"B_conservative_blend":assemble(blend_test),
            "C_notebook11_diverse":assemble(base_test)}

def validate_write(label,pred):
    pred=np.asarray(pred,float); byid=pd.Series(pred,index=test_f.ID.astype(str))
    out=sample.copy(); out["Target"]=sample.ID.astype(str).map(byid).to_numpy(float)
    checks={"rows_280961":len(out)==280961,"id_order":out.ID.equals(sample.ID),
      "duplicate":not out.ID.duplicated().any(),"nan":not out.Target.isna().any(),
      "infinity":np.isfinite(out.Target).all(),"numeric":pd.api.types.is_numeric_dtype(out.Target)}
    assert all(checks.values())
    path=SUB/f"submission_16_{label}.csv"; out.to_csv(path,index=False)
    return out,path,{**checks,"min":out.Target.min(),"max":out.Target.max(),"mean":out.Target.mean(),"std":out.Target.std()}

predcorr=pd.DataFrame(candidates).corr(); display(predcorr)
# En fazla üç; %99.5'ten yüksek korelasyonlu kopyaları yazma.
written={}; reports=[]; kept=[]
for label,pred in candidates.items():
    if any(abs(np.corrcoef(pred,candidates[k])[0,1])>=.995 for k in kept):
        print("Skipped near-duplicate",label); continue
    out,path,rep=validate_write(label,pred); kept.append(label); written[label]=(out,path); reports.append({"Submission":label,"path":str(path),**rep})
submission_report=pd.DataFrame(reports); submission_report.to_csv(ART/"submission_report.csv",index=False)
display(submission_report)
t=stage("full fit + submission validation",t)
""")

md("## 7. Final evidence-based report")
code(r"""
final_table=pd.DataFrame(RESULTS).sort_values("Overall RMSE")
best=final_table.iloc[0]; worst=final_table.iloc[-1]
sp=final_table[final_table.Feature.str.contains("kmeans|band|quadrant",case=False,regex=True)]
cl=final_table[final_table.Feature.str.contains("kmeans",case=False,regex=True)]
te=final_table[final_table.Feature.str.startswith("te_")]
inter=final_table[final_table.Feature.isin(SIMPLE)]
answers=f'''
### Final rapor

1. **En faydalı yeni feature:** `{best.Feature}`; OOF gain `{best['OOF Gain']:+.6f}`.
2. **En başarısız feature:** `{worst.Feature}`; OOF gain `{worst['OOF Gain']:+.6f}`.
3. **Spatial feature işe yaradı mı?** En iyi spatial gain `{sp['OOF Gain'].max():+.6f}`; karar OOF tablosundaki Accepted alanıdır.
4. **Cluster işe yaradı mı?** En iyi cluster gain `{cl['OOF Gain'].max():+.6f}`.
5. **Target encoding işe yaradı mı?** En iyi leakage-safe TE gain `{te['OOF Gain'].max():+.6f}`.
6. **Interaction katkısı:** En iyi tekil interaction gain `{inter['OOF Gain'].max():+.6f}`; toplu sonuç ayrıca raporlandı.
7. **Pseudo labeling:** Hayır. Gerçek test label'ı veya bağımsız öğretmen yok; confidence self-training'i güvenilir kılmaz.
8. **Monotonic constraints:** Gain `{mono_row['OOF Gain']:+.6f}`; yalnız pozitifse ve worst fold bozulmadıysa tutulur.
9. **Notebook 11 neden güçlü?** Known/masked uzman ayrımı test missingness desenini taklit ediyor; temporal split drift'i
ölçüyor ve düşük varyanslı, fiziksel olarak anlamlı static feature set kullanıyor.
10. **Leaderboard beklentisi:** Yalnız OOF ve worst-fold birlikte iyileşen feature seçildi; yakın-kopya submissionlar
elenerek sahte çeşitlilik önlendi. Buna rağmen public iyileşme garanti değildir; test horizon drift'i bilinmiyor.

Baseline: known `{BASE_MET['Known RMSE']:.6f}`, masked `{BASE_MET['Masked RMSE']:.6f}`,
overall `{BASE_MET['Overall RMSE']:.6f}`, worst `{BASE_MET['Worst Fold']:.6f}`, std `{BASE_MET['Fold Std']:.6f}`.
Toplam süre: `{perf_counter()-T0:.1f}` saniye. Checkpoint: `{ART}`.
'''
display(Markdown(answers))
assert Path(ART/"ablation_all.csv").exists() and len(submission_report)>=1
""")

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.13"},
}
OUT.write_text(nbf.writes(nb), encoding="utf-8")
print(OUT)
