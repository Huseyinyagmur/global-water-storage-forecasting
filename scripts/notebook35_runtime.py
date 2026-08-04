"""Notebook 35: bounded, leakage-safe flattened sequence LightGBM pilot."""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from time import perf_counter

# The repository venv already contains LightGBM and parquet support; no install/download.
_SCRIPT_ROOT = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path.cwd().resolve().parent
_VENV_SITE = _SCRIPT_ROOT / ".venv/Lib/site-packages"
if _VENV_SITE.exists() and str(_VENV_SITE) not in sys.path:
    sys.path.append(str(_VENV_SITE))

import numpy as np
import pandas as pd
import psutil
from lightgbm import LGBMRegressor
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import root_mean_squared_error

ROOT = Path.cwd().resolve()
if not (ROOT / "data/raw/Train.csv").exists():
    ROOT = ROOT.parent
ART = ROOT / "artifacts/notebook35"
ART.mkdir(parents=True, exist_ok=True)
CACHE = ART / "cache"
CACHE.mkdir(exist_ok=True)
SEED = 42
np.random.seed(SEED)

# Predeclared, immutable gates and performance stop.
GATES = {"A_overall_gain": .002, "B_masked_gain": .004, "C_blend_gain": .0015,
         "D_gru_gain_fraction": .60, "known_tolerance": .001, "space_time_tolerance": .002}
PERF = {"max_estimated_minutes": 30.0, "max_estimated_peak_ram_gb": 8.0}

DYN = ["TWS_t", "SPEI_01_t", "SPEI_03_t", "SPEI_06_t", "SPEI_12_t",
       "SOIL_MOISTURE_t", "month_sin", "month_cos"]
SPEI = ["SPEI_01_t", "SPEI_03_t", "SPEI_06_t", "SPEI_12_t"]
PARAMS = dict(n_estimators=160, learning_rate=.08, num_leaves=63, min_child_samples=100,
              reg_lambda=1.0, subsample=.85, colsample_bytree=.9,
              random_state=SEED, n_jobs=4, verbosity=-1)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def rss_gb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / 1024**3


def static_features(d: pd.DataFrame, min_year: int) -> pd.DataFrame:
    o = d.copy()
    dt = o.time
    o["month"] = dt.dt.month.astype("int8"); o["quarter"] = dt.dt.quarter.astype("int8")
    o["year"] = dt.dt.year.astype("int16")
    o["elapsed_month"] = ((dt.dt.year-min_year)*12+dt.dt.month-1).astype("int16")
    o["month_sin_calc"] = np.sin(2*np.pi*o.month/12).astype("float32")
    o["month_cos_calc"] = np.cos(2*np.pi*o.month/12).astype("float32")
    o["season_sin"] = np.sin(2*np.pi*o.quarter/4).astype("float32")
    o["season_cos"] = np.cos(2*np.pi*o.quarter/4).astype("float32")
    s=o[SPEI]; o["spei_mean"]=s.mean(axis=1); o["spei_min"]=s.min(axis=1); o["spei_max"]=s.max(axis=1); o["spei_std"]=s.std(axis=1)
    o["spei_01_minus_12"]=o.SPEI_01_t-o.SPEI_12_t; o["spei_03_minus_12"]=o.SPEI_03_t-o.SPEI_12_t
    o["spei_short_long_ratio"]=o.SPEI_01_t/(o.SPEI_12_t.abs()+.1)
    o["lat_band_5"]=np.floor((o.lat+90)/5); o["lon_band_5"]=np.floor((o.lon+180)/5)
    o["spatial_bin_10"]=o.lat_band_5*100+o.lon_band_5; o["lat_lon_interaction"]=o.lat*o.lon/1000
    o["lat_sin"]=np.sin(np.radians(o.lat)); o["lat_cos"]=np.cos(np.radians(o.lat))
    o["lon_sin"]=np.sin(np.radians(o.lon)); o["lon_cos"]=np.cos(np.radians(o.lon))
    return o


A=["lat","lon","TWS_t",*SPEI,"SOIL_MOISTURE_t","month","month_sin","month_cos","year"]
B=["quarter","elapsed_month","month_sin_calc","month_cos_calc","season_sin","season_cos"]
C=["spei_mean","spei_min","spei_max","spei_std","spei_01_minus_12","spei_03_minus_12","spei_short_long_ratio"]
D=["lat_band_5","lon_band_5","spatial_bin_10","lat_lon_interaction","lat_sin","lat_cos","lon_sin","lon_cos"]
BASE=list(dict.fromkeys(A+B+C+D)); BASE_MASK=[x for x in BASE if x != "TWS_t"]


def build_window(panel: pd.DataFrame, length: int) -> tuple[pd.DataFrame,list[str],list[str],dict]:
    """Every lag is shift(k) within location; current/same-date values are excluded."""
    cache = CACHE / f"pilot_features_L{length}.parquet"
    meta = CACHE / f"pilot_features_L{length}.json"
    signature = hashlib.sha256(("|".join(DYN)+f"|L={length}|shift_only|v2").encode()).hexdigest()
    if cache.exists() and meta.exists() and json.loads(meta.read_text())["signature"] == signature:
        x = pd.read_parquet(cache)
        if sha256(cache) == json.loads(meta.read_text())["sha256"]:
            raw=[c for c in x if "_lag_" in c or c.startswith("TWS_missing_lag_")]
            sums=[c for c in x if any(c.endswith("_"+z) for z in ["last_valid_value","window_mean","window_std","window_min","window_max","first_minus_last","linear_slope","valid_count","missing_fraction","months_since_last_valid"])]
            return x,raw,sums,json.loads(meta.read_text())
    t0=perf_counter(); before=rss_gb()
    x=panel.sort_values(["location_id","time","original_index"]).copy()
    grp=x.groupby("location_id",sort=False)
    source_max=grp.time.shift(1)
    eligible=source_max.notna()
    assert (source_max[eligible] < x.loc[eligible,"time"]).all(), "same/future date entered lag history"
    raw=[]; sums=[]
    for col in DYN:
        lcols=[]
        for k in range(1,length+1):
            name=f"{col}_lag_{k}"; x[name]=grp[col].shift(k).astype("float32"); raw.append(name); lcols.append(name)
            if col=="TWS_t":
                m=f"TWS_missing_lag_{k}"; x[m]=x[name].isna().astype("int8"); raw.append(m)
        vals=x[lcols]
        names={
            "last_valid_value": vals.bfill(axis=1).iloc[:,0],
            "window_mean": vals.mean(axis=1), "window_std": vals.std(axis=1),
            "window_min": vals.min(axis=1), "window_max": vals.max(axis=1),
            "first_minus_last": vals.bfill(axis=1).iloc[:,0]-vals.ffill(axis=1).iloc[:,-1],
            "valid_count": vals.notna().sum(axis=1).astype("int8"),
            "missing_fraction": vals.isna().mean(axis=1),
        }
        arr=vals.to_numpy(dtype="float32"); slopes=np.full(len(x),np.nan,dtype="float32"); axis=np.arange(length,dtype="float32")
        for i,row in enumerate(arr):
            ok=np.isfinite(row)
            if ok.sum()>=2: slopes[i]=np.polyfit(axis[ok],row[ok],1)[0]
        names["linear_slope"]=slopes
        if col=="TWS_t": names["months_since_last_valid"]=vals.notna().to_numpy().argmax(axis=1).astype("int8")+1
        for suffix,v in names.items():
            n=f"{col}_{suffix}"; x[n]=np.asarray(v,dtype="float32"); sums.append(n)
    x["history_count"]=grp.cumcount().clip(upper=length).astype("int8")
    x["cold_start"]=(x.history_count<length)
    keep=["original_index","time","lat","lon","location_id","target","is_train","is_valid","known","cold_start",*BASE,*raw,*sums]
    keep=list(dict.fromkeys(keep)); x=x[keep]
    x.to_parquet(cache,index=False)
    info={"signature":signature,"sha256":sha256(cache),"rows":len(x),"length":length,"elapsed_seconds":perf_counter()-t0,"peak_ram_gb":max(before,rss_gb()),"source_date_assertion":True,"same_date_forbidden":True,"target_features":False}
    meta.write_text(json.dumps(info,indent=2),encoding="utf-8")
    return x,raw,sums,info


def metrics(y,p,known,lat):
    out={"RMSE":root_mean_squared_error(y,p),"Known_RMSE":root_mean_squared_error(y[known],p[known]),"Masked_RMSE":root_mean_squared_error(y[~known],p[~known])}
    bands=pd.cut(lat,[-90,-30,0,30,60,90],labels=False)
    out["Worst_group_RMSE"]=max(root_mean_squared_error(y[bands==b],p[bands==b]) for b in np.unique(bands))
    return out


def fit_pair(x, features, tag):
    train_mask=x.is_train.to_numpy(bool); valid_mask=x.is_valid.to_numpy(bool); known=x.loc[valid_mask,"known"].to_numpy(bool)
    y=x.loc[valid_mask,"target"].to_numpy(); pred=np.empty(len(y),dtype="float32")
    t0=perf_counter(); peak=rss_gb(); fi=[]
    known_features=list(dict.fromkeys(BASE+features)); masked_features=[c for c in known_features if c!="TWS_t"]
    for expert,route,cols in [("known",known,known_features),("masked",~known,masked_features)]:
        m=LGBMRegressor(**PARAMS); m.fit(x.loc[train_mask,cols],x.loc[train_mask,"target"])
        ti=perf_counter(); pred[route]=m.predict(x.loc[valid_mask,cols].loc[route]); infer=perf_counter()-ti
        fi.extend([{"Feature_set":tag,"Expert":expert,"Feature":c,"Importance_gain":float(v)} for c,v in zip(cols,m.booster_.feature_name(),strict=False)] if False else
                  [{"Feature_set":tag,"Expert":expert,"Feature":c,"Importance_gain":float(v)} for c,v in zip(cols,m.booster_.feature_importance("gain"))])
        peak=max(peak,rss_gb())
    elapsed=perf_counter()-t0
    met=metrics(y,pred,known,x.loc[valid_mask,"lat"].to_numpy())
    met.update({"System":tag,"Training_seconds":elapsed-infer,"Inference_seconds":infer,"Peak_RAM_GB":peak,"Feature_count_known":len(known_features),"Feature_count_masked":len(masked_features)})
    return met,pred,y,known,valid_mask,pd.DataFrame(fi)


print("Notebook35 start RAM GB",rss_gb())
t_all=perf_counter()
train=pd.read_csv(ROOT/"data/raw/Train.csv",parse_dates=["time"])
for c in [*DYN,"lat","lon","target"]: train[c]=train[c].astype("float32")
train["original_index"]=np.arange(len(train)); train["location_id"]=train.lat.astype(str)+"_"+train.lon.astype(str)
oof=pd.read_parquet(ROOT/"artifacts/notebook25/oof_row_level_predictions.parquet")
foof=oof[oof.fold_id.eq("strict_temporal_2")]; valid_idx=set(foof.original_index.astype(int)); valid_dates=np.array(sorted(train.loc[list(valid_idx),"time"].unique())); cutoff=pd.Timestamp(valid_dates.min())
locs=train.loc[list(valid_idx),["location_id","lat"]].drop_duplicates(); locs["band"]=pd.cut(locs.lat,[-90,-30,0,30,60,90],labels=False)
selected=locs.groupby("band",group_keys=False).apply(lambda g:g.sample(min(64,len(g)),random_state=SEED),include_groups=False).location_id.unique()
panel=train[train.location_id.isin(selected)].copy(); panel=static_features(panel,int(train.time.dt.year.min()))
panel["is_valid"]=panel.original_index.isin(valid_idx); panel["is_train"]=panel.time.lt(cutoff)
test=pd.read_csv(ROOT/"data/raw/Test.csv",parse_dates=["time"]); test_counts=test.groupby("time").TWS_t.count().to_numpy(); known_pos=set(np.flatnonzero(test_counts>1000))
known_dates={d for j,d in enumerate(valid_dates) if j in known_pos}; panel["known"]=panel.time.isin(known_dates)
assert not panel.loc[panel.is_valid,"original_index"].duplicated().any()
assert panel.loc[panel.is_train,"time"].max() < panel.loc[panel.is_valid,"time"].min()

window_rows=[]; result_rows=[]; predictions={}; importance=[]
for L in [3,6,12]:
    x,raw,sums,info=build_window(panel,L); window_rows.append(info)
    configs={"F3":raw+sums} if L<12 else {"E0":[],"F1":raw,"F2":sums,"F3":raw+sums}
    for tag,cols in configs.items():
        name=f"{tag}_L{L}"; met,p,y,k,vm,fi=fit_pair(x,cols,name); result_rows.append(met); predictions[name]=(p,y,k,vm,x.loc[vm,["original_index","lat","time","cold_start"]].reset_index(drop=True)); importance.append(fi)

pilot=pd.DataFrame(result_rows)
e0_record=foof[foof.original_index.isin(panel.loc[panel.is_valid,"original_index"])].set_index("original_index")
ref=predictions["E0_L12"][4]; aligned=e0_record.loc[ref.original_index]
record_met=metrics(aligned.target.to_numpy(),aligned.oof_prediction.to_numpy(),predictions["E0_L12"][2],ref.lat.to_numpy())
record_met.update({"System":"E0_recorded_Notebook33","Training_seconds":np.nan,"Inference_seconds":np.nan,"Peak_RAM_GB":np.nan,"Feature_count_known":len(BASE),"Feature_count_masked":len(BASE_MASK)})
pilot=pd.concat([pilot,pd.DataFrame([record_met,
 {"System":"GRU_S1_recorded","RMSE":.696528,"Known_RMSE":.594448,"Masked_RMSE":.727309,"Worst_group_RMSE":np.nan,"Training_seconds":np.nan,"Inference_seconds":np.nan,"Peak_RAM_GB":np.nan,"Feature_count_known":11,"Feature_count_masked":11},
 {"System":"GRU_S2_recorded_blend","RMSE":.694025,"Known_RMSE":.596193,"Masked_RMSE":.723636,"Worst_group_RMSE":np.nan,"Training_seconds":np.nan,"Inference_seconds":np.nan,"Peak_RAM_GB":np.nan,"Feature_count_known":11,"Feature_count_masked":11}])],ignore_index=True)
pilot["Evidence"]="computed Notebook35 pilot"; pilot.loc[pilot.System.str.contains("recorded"),"Evidence"]="recorded result; not recomputed"
pilot.to_csv(ART/"pilot_results.csv",index=False)
pd.concat(importance,ignore_index=True).sort_values("Importance_gain",ascending=False).to_csv(ART/"feature_importance.csv",index=False)

best_len=int(pilot[pilot.System.str.startswith("F3_L")].sort_values("RMSE").iloc[0].System.split("L")[-1])
best_name=pilot[(pilot.System.str.match("F[123]_L12"))].sort_values("RMSE").iloc[0].System
fp,fy,fknown,_,fmeta=predictions[best_name]; cp=aligned.oof_prediction.to_numpy(); cy=aligned.target.to_numpy()
assert np.allclose(fy,cy,atol=1e-6)

# Conservative same-row OOF blend; reject endpoint/unstable weights (one pilot fold cannot prove stability).
grid=np.linspace(0,1,21); scores=np.array([root_mean_squared_error(fy,w*cp+(1-w)*fp) for w in grid]); w=float(grid[scores.argmin()]); blend=w*cp+(1-w)*fp
div=pd.DataFrame([{
 "Canonical_flattened_prediction_Pearson":pearsonr(cp,fp).statistic,
 "Canonical_flattened_prediction_Spearman":spearmanr(cp,fp).statistic,
 "Residual_correlation":pearsonr(fy-cp,fy-fp).statistic,
 "Mean_absolute_prediction_difference":np.mean(np.abs(cp-fp)),
 "Worst_5pct_overlap":np.mean((np.abs(fy-cp)>=np.quantile(np.abs(fy-cp),.95))&(np.abs(fy-fp)>=np.quantile(np.abs(fy-fp),.95))),
 "Known_disagreement":np.mean(np.abs(cp[fknown]-fp[fknown])),"Masked_disagreement":np.mean(np.abs(cp[~fknown]-fp[~fknown])),
 "Northern_disagreement":np.mean(np.abs(cp[fmeta.lat>=0]-fp[fmeta.lat>=0])),"Southern_disagreement":np.mean(np.abs(cp[fmeta.lat<0]-fp[fmeta.lat<0])),
 "Early_period_disagreement":np.mean(np.abs(cp[fmeta.time<=fmeta.time.median()]-fp[fmeta.time<=fmeta.time.median()])),"Late_period_disagreement":np.mean(np.abs(cp[fmeta.time>fmeta.time.median()]-fp[fmeta.time>fmeta.time.median()])),
 "OOF_blend_canonical_weight":w,"OOF_blend_RMSE":scores.min(),"Fold_weight_stability_verified":False}])
div.to_csv(ART/"diversity_analysis.csv",index=False)

cold=[]
for flag,g in fmeta.groupby("cold_start"):
 ix=g.index.to_numpy(); kk=fknown[ix]
 cold.append({"Cold_start":bool(flag),"Rows":len(ix),"Fraction":len(ix)/len(fmeta),"Flattened_RMSE":root_mean_squared_error(fy[ix],fp[ix]),"Masked_RMSE":root_mean_squared_error(fy[ix][~kk],fp[ix][~kk]) if (~kk).any() else np.nan,"Canonical_RMSE":root_mean_squared_error(fy[ix],cp[ix]),"Chosen_fallback":"flattened left-padding/native-missing; no canonical fallback selected"})
pd.DataFrame(cold).to_csv(ART/"cold_start_analysis.csv",index=False)

pd.DataFrame([{"Window_length":r["length"],"Rows":r["rows"],"Cache_SHA256":r["sha256"],"Build_seconds":r["elapsed_seconds"],"Peak_RAM_GB":r["peak_ram_gb"],"Source_date_assertion":r["source_date_assertion"],"Same_date_forbidden":r["same_date_forbidden"],"Target_derived_features":r["target_features"],"Left_padding":True} for r in window_rows]).to_csv(ART/"window_definition.csv",index=False)
pd.DataFrame([{"Feature_set":"F1","Definition":"raw lags 1..12 plus TWS missing masks","Feature_count":len(predictions["F1_L12"][4].columns)-5,"Allowed":True},
              {"Feature_set":"F2","Definition":"nine summaries per dynamic feature plus TWS months-since-last-valid","Feature_count":"see model feature counts","Allowed":True},
              {"Feature_set":"F3","Definition":"F1 + F2","Feature_count":"see model feature counts","Allowed":True},
              {"Feature_set":"F4","Definition":"OOF-stable small subset","Feature_count":0,"Allowed":False,"Reason":"one pilot fold cannot establish fold stability"}]).to_csv(ART/"lag_feature_report.csv",index=False)

e0m=metrics(fy,cp,fknown,fmeta.lat.to_numpy()); fm=metrics(fy,fp,fknown,fmeta.lat.to_numpy()); bm=metrics(fy,blend,fknown,fmeta.lat.to_numpy())
gru_gain=.772760-.696528; flat_gain=.772760-fm["RMSE"]; recovered=flat_gain/gru_gain
pd.DataFrame([{"Question":"Flattened gain vs recorded E0","Value":flat_gain,"Caveat":"Notebook35 model and Notebook33 recorded E0 differ in training protocol"},
              {"Question":"Fraction of GRU S1 recorded gain reproduced","Value":recovered,"Caveat":"Indicative only; direct causal attribution is invalid"},
              {"Question":"GRU architecture contribution","Value":pd.NA,"Caveat":"Cannot isolate without identical full folds/training population"},
              {"Question":"Full GRU validation still needed","Value":True,"Caveat":"Yes, if sequence path remains prioritized after robust flattened validation"}]).to_csv(ART/"gru_comparison.csv",index=False)

# Extrapolate pilot cost before full execution. Four systems require multiple very large feature matrices/models.
pilot_train=int(panel.is_train.sum()); full_train=int((train.time<cutoff).sum()); ratio=full_train/max(pilot_train,1)
f3_time=float(pilot.loc[pilot.System.eq("F3_L12"),"Training_seconds"].iloc[0]); estimated_minutes=f3_time*ratio*8/60
estimated_ram=max(r["peak_ram_gb"] for r in window_rows)*max(1,ratio*.30)
pilot_promising=fm["RMSE"] < e0m["RMSE"] or bm["RMSE"] < e0m["RMSE"]
full_allowed=pilot_promising and estimated_minutes<=PERF["max_estimated_minutes"] and estimated_ram<=PERF["max_estimated_peak_ram_gb"]
# This bounded runtime intentionally never launches the expensive branch; if under threshold it records that a separate frozen run is required.
full_status="not_run_performance_gate" if not full_allowed else "not_run_separate_frozen_full_run_required"
full=pd.DataFrame([{"Validation":"strict_date_grouped_temporal","Status":full_status,"Overall_RMSE":np.nan,"Known_RMSE":np.nan,"Masked_RMSE":np.nan,"Fold_RMSE":np.nan,"Mean_fold_RMSE":np.nan,"Fold_std":np.nan,"Worst_fold":np.nan,"Northern_RMSE":np.nan,"Southern_RMSE":np.nan,"Cold_start_RMSE":np.nan,"Training_seconds":np.nan,"Inference_seconds":np.nan,"Peak_RAM_GB":np.nan,"Estimated_total_minutes":estimated_minutes,"Estimated_peak_RAM_GB":estimated_ram},
 {"Validation":"irregular_date_hindcast","Status":full_status},{"Validation":"spatial_block","Status":full_status},{"Validation":"space_time","Status":full_status}])
full.to_csv(ART/"full_validation_results.csv",index=False)

gateA=(e0m["RMSE"]-fm["RMSE"]>=GATES["A_overall_gain"] and fm["Worst_group_RMSE"]<=e0m["Worst_group_RMSE"])
gateB=(e0m["Masked_RMSE"]-fm["Masked_RMSE"]>=GATES["B_masked_gain"] and fm["Known_RMSE"]<=e0m["Known_RMSE"]+GATES["known_tolerance"])
gateC=(e0m["RMSE"]-bm["RMSE"]>=GATES["C_blend_gain"] and False)
gateD=(recovered>=GATES["D_gru_gain_fraction"] and False)
accepted=bool(gateA or gateB or gateC or gateD) and False  # full robust validation is mandatory
pd.DataFrame([{"Submission":"none","Created":False,"Rows":0,"Reason":"Robust full validation not completed; acceptance gate cannot pass","Acceptance":accepted,"Gate_A_pilot":gateA,"Gate_B_pilot":gateB,"Gate_C":gateC,"Gate_D":gateD,"SHA256":""}]).to_csv(ART/"submission_report.csv",index=False)

raw_rm=float(pilot.loc[pilot.System.eq("F1_L12"),"RMSE"].iloc[0]); sum_rm=float(pilot.loc[pilot.System.eq("F2_L12"),"RMSE"].iloc[0])
report=f"""# Notebook 35 final recommendation

1. Flattened lag pipeline leakage-free kuruldu mu? **Evet, pilot kapsamında.** Tüm laglar location içi `shift(k)` ile üretildi; `max(source_date) < prediction_date` doğrulandı. Target-derived feature yoktur.
2. En iyi window length hangisi? Pilot F3 ablation sonucu **L={best_len}**; ana L12 karşılaştırması önceden sabitti.
3. Raw lag mı summary feature mı daha faydalı? L12 pilotunda **{'raw lag' if raw_rm<sum_rm else 'summary'}** (F1={raw_rm:.6f}, F2={sum_rm:.6f}).
4. Flattened LightGBM canonical modeli geçti mi? Aynı satırlardaki recorded canonical E0'a göre **{fm['RMSE']<e0m['RMSE']}** (E0={e0m['RMSE']:.6f}, {best_name}={fm['RMSE']:.6f}). Bu yalnız pilot kanıtıdır.
5. Masked RMSE ne kadar değişti? {e0m['Masked_RMSE']:.6f} -> {fm['Masked_RMSE']:.6f}; gain={e0m['Masked_RMSE']-fm['Masked_RMSE']:+.6f}.
6. Robust validation kapıları geçildi mi? **Hayır.** Full strict/irregular/spatial/space-time validation çalıştırılmadı; tahmin {estimated_minutes:.1f} dakika ve {estimated_ram:.1f} GB peak RAM, durum `{full_status}`.
7. GRU pilot kazancının ne kadarı yeniden üretildi? Gösterge oranı **{recovered:.1%}**; farklı eğitim protokolleri nedeniyle doğrudan nedensel karşılaştırma değildir.
8. GRU sequence mimarisinin ayrı katkısı var mı? **Belirlenemedi.** Aynı full foldlarda kontrollü GRU/flattened karşılaştırması gerekir.
9. Blend işe yaradı mı? Pilot OOF'ta **{bm['RMSE']<min(e0m['RMSE'],fm['RMSE'])}**; RMSE={bm['RMSE']:.6f}, canonical weight={w:.2f}. Tek fold olduğu için ağırlık stabilitesi kanıtlanmadı.
10. Submission üretildi mi? **Hayır.** Robust acceptance zorunludur.
11. Full GRU validation hâlâ öncelikli mi? **Evet, ancak önce seçilmiş flattened konfigürasyonun ayrı frozen full validation'ı tamamlanmalı.**
12. İnternet kullanmadan sonraki en güçlü deney nedir? **{best_name} için tek, frozen strict/irregular/spatial/space-time validation koşusu; yeni feature veya parametre araması yok.**

Pilot promising={pilot_promising}; formal acceptance=False. Notebook 33 GRU değerleri yeniden hesaplanmadı ve `recorded result` olarak etiketlendi. Canonical/current-best submission dosyaları değiştirilmedi.
"""
(ART/"final_recommendation.md").write_text(report,encoding="utf-8")
print(report)
print("Notebook35 elapsed seconds",perf_counter()-t_all,"RAM GB",rss_gb())
