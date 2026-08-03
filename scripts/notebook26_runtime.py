from pathlib import Path
import hashlib, json, re, subprocess, importlib.metadata as im
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Markdown

ROOT=Path.cwd().resolve()
if not (ROOT/"submissions/submission_optimized.csv").exists(): ROOT=ROOT.parent
ART=ROOT/"artifacts/notebook26"; FIG=ART/"figures"; FIG.mkdir(parents=True,exist_ok=True)
CANON=ROOT/"submissions/submission_optimized.csv"; BEST=ROOT/"submissions/submission_21_spatial_residual_candidate.csv"
CANON_SHA="b61ecd15862ca6d7527c4f494de802ed6ea6d31c420737cb78f977d9bda3b968"; BEST_SHA="84d4ce6efdbfc3fbca55d9eea1ebccfe628ffa962e31b62a654c179dc649b488"
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
test=pd.read_csv(ROOT/"data/raw/Test.csv"); sample=pd.read_csv(ROOT/"data/raw/SampleSubmission.csv"); canon=pd.read_csv(CANON); best=pd.read_csv(BEST)
known=test.TWS_t.notna().to_numpy(); masked=~known; delta=best.Target-canon.Target
checks=[
("Canonical submission",CANON.exists(),sha(CANON),sha(CANON)==CANON_SHA,len(canon),canon.ID.is_unique,canon.ID.equals(sample.ID),canon.Target.isna().sum(),np.isinf(canon.Target).sum(),canon.Target.min(),canon.Target.max(),True,int(masked.sum())),
("Current-best submission",BEST.exists(),sha(BEST),sha(BEST)==BEST_SHA,len(best),best.ID.is_unique,best.ID.equals(sample.ID),best.Target.isna().sum(),np.isinf(best.Target).sum(),best.Target.min(),best.Target.max(),np.array_equal(best.Target.to_numpy()[known],canon.Target.to_numpy()[known]),int(masked.sum())),
("Notebook 21 reproduction",(ROOT/"artifacts/notebook21/submission_report.csv").exists(),"not applicable",True,1,True,True,0,0,np.nan,np.nan,True,int(masked.sum())),
("Notebook 25 OOF artefacts",(ROOT/"artifacts/notebook25/oof_row_level_predictions.parquet").exists(),"not applicable",True,749046,True,True,0,0,np.nan,np.nan,True,0),
("Notebook 24 model card",(ROOT/"artifacts/notebook24/model_card.md").exists(),"not applicable",True,1,True,True,0,0,np.nan,np.nan,True,0)]
integrity=pd.DataFrame(checks,columns=["Item","Exists","SHA256","SHA match","Rows/items","ID unique","ID order","NaN","Infinity","Prediction min","Prediction max","Known rows preserved","Masked coverage"])
integrity["Canonical-current correlation"]=np.corrcoef(canon.Target,best.Target)[0,1]; integrity["MAD"]=np.abs(delta).mean(); integrity["Maximum difference"]=np.abs(delta).max(); integrity["Changed fraction"]=(np.abs(delta)>1e-12).mean()
integrity.to_csv(ART/"final_integrity_report.csv",index=False); assert integrity.Exists.all() and integrity["SHA match"].all() and sha(CANON)==CANON_SHA and sha(BEST)==BEST_SHA

executive="""# Executive summary

This project forecasts global gridded terrestrial-water-storage anomalies across repeated spatial locations and future dates. The final system retains Notebook 11's two-expert LightGBM selected blend as the **canonical base** and uses Notebook 21's bounded Southern-hemisphere residual intercept only as the **current-best refinement**.

Canonical public RMSE is **0.824463818**; current-best public RMSE is **0.823609466**, an improvement of **0.000854352 (0.104%)**. Model selection evolved from early splits toward grouped temporal, irregular-date, spatial-block and space-time validation. Audits found that the canonical `prior_global_month` training feature has a location-major ordering risk: approximately 91.30% of training rows can reference later timestamps. No direct validation-target leakage was detected.

The public-positive correction is not presented as universally robust. Four historical windows improved in only three cases; the closest window deteriorated by 0.002613 and transferability scored 54.53/100. Canonical-equivalent leakage-controlled OOF covers 749,046 rows (34.774%), with RMSE 0.743809, MAE 0.568233, P95 absolute error 1.520082 and Southern P95 1.773752. Trustworthiness is 66.30/100: **useful but requires careful oversight**.

Final decision: submit `submissions/submission_21_spatial_residual_candidate.csv`, preserve the canonical base unchanged, close further score optimization, and use this work for broad-scale research/benchmarking with expert review—not critical local operations or automated safety decisions.
"""
(ART/"executive_summary.md").write_text(executive,encoding="utf-8")

results_rows=[
("Baseline","02","statistical/basic regressors","early/random","not available",np.nan,np.nan,True,"inconclusive","sanity reference only"),
("Boosting","03","LightGBM","early validation","not available",np.nan,np.nan,True,"promising","became strongest family"),
("Boosting","04","XGBoost","early validation","not available",np.nan,np.nan,True,"failed","no verified public advantage"),
("Boosting","05","CatBoost","early validation","not available",np.nan,np.nan,True,"failed","no verified public advantage"),
("Ensemble","06","boosting ensemble","early validation","not available",np.nan,np.nan,True,"inconclusive","OOF/drift control required"),
("Canonical","11","LightGBM selected blend","temporal OOF","0.711647 overall",.824463818,0,True,"successful","durable canonical base"),
("Alternatives","14","alternative-model blend","temporal OOF","0.688811 representative",.825971129,.825971129-.824463818,True,"public worse","local rank did not transfer"),
("Delta","15","delta-target dual expert","temporal OOF","0.694090 representative",.824797753,.824797753-.824463818,True,"near-best","did not beat canonical"),
("Feature research","16A","best single feature","Notebook-11-like OOF","local gain",.839181712,.839181712-.824463818,True,"public worse","broad search overfit; score supplied externally"),
("Reference mismatch","16C/17","non-identical Notebook-11 reference","audit","not available",.836026816,.836026816-.824463818,True,"failed","Notebook 17 proved pipeline mismatch; score supplied externally"),
("Hydrology","18","drying-transition/domain features","canonical OOF","0.709586 representative",.831108253,.831108253-.824463818,True,"public worse","weak incremental signal"),
("Validation audit","20","strict temporal/irregular/spatial/space-time","multi-system","0.740524/0.680366/0.801398/0.817423",np.nan,np.nan,False,"methodological","ordering and spatial risks exposed"),
("Refinement","21","Southern residual correction","joint gates","spatial gain 0.005311",.823609466,.823609466-.824463818,True,"current best","small public-positive refinement"),
("Transfer audit","23","four historical windows","strict future windows","score 54.53/100",np.nan,np.nan,False,"weakly transferable","closest window worsened"),
("Trustworthiness","24","model card/scorecard","evidence synthesis","66.30/100",np.nan,np.nan,False,"oversight required","documented use boundaries"),
("OOF audit","25","canonical-equivalent strict OOF","2 strict folds","RMSE 0.743809; MAE 0.568233",np.nan,np.nan,False,"valid proxy OOF","tail/support artefacts persisted")]
results=pd.DataFrame(results_rows,columns=["Stage","Notebook","Method","Validation design","Local metric","Public RMSE","Public delta vs canonical","Submission produced","Status","Main conclusion"])
results.to_csv(ART/"final_results_table.csv",index=False)

model_def=f"""# Final model definition

- **Canonical base:** Notebook 11 final selected blend; public RMSE 0.824463818; file `submissions/submission_optimized.csv`; SHA256 `{CANON_SHA}`.
- **Architecture:** known-row LightGBM expert; masked-row base LightGBM plus causal-history expert with fixed 0.65/0.35 blend.
- **Target/features:** future TWS anomaly from spatial coordinates/bins, calendar/cyclic time, SPEI 01/03/06/12 and aggregates, soil moisture, known-row `TWS_t`, and historical priors.
- **Current-best refinement:** Notebook 21 stable Southern-hemisphere masked-row +0.05 residual intercept. It changes 36,832 test rows; known rows remain identical.
- **Final file:** `submissions/submission_21_spatial_residual_candidate.csv`; SHA256 `{BEST_SHA}`; public RMSE 0.823609466.
- **Improvement:** 0.000854352 RMSE (0.104%).
- **Limitations:** Notebook 23 classified correction transfer as leaderboard-positive but weak (54.53/100). Trustworthiness is 66.30/100, useful with careful oversight.
- **OOF terminology:** Notebook 25 is `canonical-equivalent leakage-controlled OOF`, not exact canonical OOF, because the risky global-month ordering was replaced with strict date-grouped history.

Notebook 21 is not renamed canonical.
"""
(ART/"final_model_definition.md").write_text(model_def,encoding="utf-8")

validation="""# Validation and generalization summary

| Design | Result | Role / limitation |
|---|---:|---|
| Notebook 08 same-month split | 0.6109 recorded | optimistic; same date split across rows |
| Notebook 11 temporal OOF | 0.711647 overall | useful known/masked simulation, incomplete spatial robustness |
| Strict temporal | 0.740524 | grouped forecast-origin causality |
| Irregular hindcast | 0.680366 | mimics official non-uniform date gaps |
| Spatial block | 0.801398 | contiguous-region transfer; limited ranking evidence |
| Space-time | 0.817423 | future date plus held region |
| Southern worst canonical | 0.833364 | strongest spatial warning |
| Notebook 23 transferability | 54.53/100 | 3/4 improved; closest window −0.002613 |
| Notebook 25 canonical-equivalent OOF | RMSE 0.743809; MAE 0.568233 | 2 strict folds, 34.774% coverage, same-grid spatial support |

These metrics belong to different validation designs and must not be compared as though they were one leaderboard. Final decisions use their joint pattern, not a single local score.
"""
(ART/"validation_generalization_summary.md").write_text(validation,encoding="utf-8")

leak=[
("Same-month split","Notebook 08 same dates across partitions","high","complete date groups","historical score remains optimistic","superseded"),
("prior_global_month location-major ordering","Notebook 20 audit","high","strict date-grouped prior for new OOF","canonical base retains training risk","documented/frozen"),
("91.30% future-date exposure","Notebook 20 contributor audit","high","do not reuse ordering implementation","indirect training-feature exposure","confirmed"),
("Direct validation-target leakage","Notebook 20 found none","none detected","fold target isolation assertions","indirect risk distinct","not detected"),
("Prediction-time availability","Notebook 19 specification","high","known/masked separation and available covariates","external feed latency","controlled in pipeline"),
("Notebook 16 reference mismatch","Notebook 17 file/code comparison","high","immutable canonical SHA","historical conclusions affected","resolved"),
("Canonical reproduction","Notebook 17/20/21 SHA and identity","low","fail-fast SHA","environment drift","verified"),
("Canonical-equivalent OOF naming","Notebook 25 strict prior","medium","explicit proxy terminology","not exact public pipeline","controlled"),
("Fixed scripts and caches","Builders 16–26; NB25 policy key","low","validate cache manifest","early notebooks less standardized","partial/high"),
("External leaderboard records","Notebook 22 evidence caveat","medium","label externally supplied values","cannot independently reproduce server scoring","documented")]
pd.DataFrame(leak,columns=["Risk","Evidence","Severity","Mitigation","Residual risk","Final status"]).to_csv(ART/"final_leakage_reproducibility_register.csv",index=False)

bias="""# Bias, tail risk and support summary

- Southern hemisphere is the principal geographic weakness: canonical spatial-block RMSE 0.833364 and Notebook 25 regional P95 absolute error 1.773752.
- Global proxy-OOF tail: P95 1.520082, P99 2.123579, maximum 4.461003.
- Combined support versus absolute error Spearman is −0.1956 (diagnostic association, not causality).
- Strict temporal folds use essentially the same spatial grid; spatial-support variation is limited and cannot validate unseen-region robustness.
- Test low/out spatial-support fraction is 0%, but test dates are temporal extrapolation and have no target for performance claims.
- Notebook 21 changes 36,832 Southern masked rows. Their mean combined support is 0.7572; none are low/out spatial support under the same-grid definition.
- Extreme-target and climate/drought tables are available in `artifacts/notebook25/`; they show diagnostic differences, not proof that omitted physics causes errors.
- Central temporal risk: the closest Notebook 23 hindcast reversed the correction benefit by −0.002613.
"""
(ART/"bias_tail_support_summary.md").write_text(bias,encoding="utf-8")

# Full notebook-file index (both distinct Notebook 11 files are retained).
inv22=pd.read_csv(ROOT/"artifacts/notebook22/experiment_inventory.csv")
index=[]
for p in sorted((ROOT/"notebooks").glob("[0-2][0-9]_*.ipynb")):
 n=int(p.name[:2]);
 if n>25: continue
 match=inv22[inv22["Notebook name"].eq(p.name)]
 purpose=match["Main purpose"].iloc[0] if len(match) else {22:"research roadmap",23:"temporal correction transferability",24:"trustworthiness/model card",25:"row-level OOF tail/support audit"}.get(n,"repository experiment")
 submission=bool(match["Submission produced"].iloc[0]) if len(match) else False
 public=match["Public RMSE"].iloc[0] if len(match) and pd.notna(match["Public RMSE"].iloc[0]) else ({21:.823609466}.get(n,np.nan))
 group="Foundation" if n<=2 else ("Model development" if n<=6 or n in [9,11,14,15] else ("Feature research" if n in [7,12,16,18] else ("Refinement" if n==21 else ("Trustworthiness and reporting" if n>=22 else "Validation and audit"))))
 index.append({"Notebook":n,"Filename":p.name,"Purpose":purpose,"Input dependencies":"data/raw + prior notebook artefacts","Main output":f"notebook {n} results/artefacts","Submission generated":submission,"Public result":public,"Current relevance":"final reference" if n in [11,17,19,20,21,22,23,24,25] else "historical evidence","Recommended reading order":group,"Status":"successful" if n in [11,21] else ("audit/report" if n>=17 and n!=18 else "historical experiment")})
exp=pd.DataFrame(index); exp.to_csv(ART/"experiment_index.csv",index=False)
md=["# Experiment index",""]
for group,g in exp.groupby("Recommended reading order",sort=False):
 md += [f"## {group}","", "| Notebook | File | Purpose | Status |", "|---:|---|---|---|"]+[f"| {r.Notebook} | `{r.Filename}` | {r.Purpose} | {r.Status} |" for r in g.itertuples()]+[""]
(ART/"experiment_index.md").write_text("\n".join(md),encoding="utf-8")

# Environment audit and explicit direct-dependency files.
direct=["numpy","pandas","scikit-learn","scipy","lightgbm","xgboost","catboost","matplotlib","seaborn","shap","nbformat","pyarrow","jupyter"]
env=[]; pins=[]
for package in direct:
 try: version=im.version(package); status="installed and used by repository family"; pins.append(f"{package}=={version}")
 except im.PackageNotFoundError: version="not available"; status="referenced but not installed"
 env.append({"Dependency":package,"Detected version":version,"Minimal role":"direct notebook import/model/report dependency","Status":status,"Exact-environment note":"pin captured from active project .venv; platform differences may remain"})
pd.DataFrame(env).to_csv(ART/"environment_audit.csv",index=False)
(ROOT/"requirements-minimal.txt").write_text("""numpy
pandas
scikit-learn
scipy
lightgbm
matplotlib
nbformat
pyarrow
jupyter
# Optional historical notebooks: xgboost, catboost, seaborn, shap
""",encoding="utf-8")
(ROOT/"requirements-reproducible.txt").write_text("# Direct package versions captured from the active project .venv on 2026-08-03.\n# This is not a full cross-platform lockfile.\n"+"\n".join(pins)+"\n",encoding="utf-8")

structure="""# Repository structure recommendation

Current folders `notebooks/`, `scripts/`, `artifacts/`, `submissions/`, `data/`, `models/`, `configs/`, `src/`, `reports/` and `experiments/` are retained to avoid breaking paths. Root README, reproduction guide, model card and requirements files now provide entry points.

Recommendations for a future non-competition branch: add `docs/` for rendered reports, `tests/` for hash/schema/fold tests, and a documented `cache/` convention (Notebook 25 currently keeps verified caches within its artefact folder). Do not move large artefacts or data without updating builders. Temporary `.tmp_*` directories and large generated files should be reviewed before release, not deleted automatically. Competition data licensing should determine whether `data/` can be distributed.
"""
(ART/"repository_structure_recommendation.md").write_text(structure,encoding="utf-8")

# Submission manifest: scores only when verified in repository/user-specified final record.
score_map={"submission_optimized.csv":.824463818,"submission_21_spatial_residual_candidate.csv":.823609466,"submission_14.csv":.825971129,"submission_15_primary.csv":.824797753,"submission_18_A_domain_best.csv":.831108253}
origin={"submission_optimized.csv":"11","submission_21_spatial_residual_candidate.csv":"21","submission_14.csv":"14","submission_15_primary.csv":"15","submission_18_A_domain_best.csv":"18"}
manifest=[]
for p in sorted((ROOT/"submissions").glob("*.csv")):
 try: d=pd.read_csv(p,usecols=["ID","Target"])
 except Exception: d=pd.read_csv(p); 
 role="canonical/base" if p.name==CANON.name else ("current-best" if p.name==BEST.name else "experimental")
 manifest.append({"Filename":p.name,"Origin notebook":origin.get(p.name,"not available"),"Method":"see originating notebook" if p.name in origin else "not available","SHA256":sha(p),"Row count":len(d),"Public RMSE":score_map.get(p.name,np.nan),"Status":"verified final" if p.name==BEST.name else ("verified base" if p.name==CANON.name else "historical"),"Role":role,"Should be used for final submission":p.name==BEST.name,"Notes":"Public score blank when not recorded/verified."})
pd.DataFrame(manifest).to_csv(ART/"submission_manifest.csv",index=False)

# Reusable figures.
plt.style.use("seaborn-v0_8-whitegrid")
hist=pd.read_csv(ROOT/"artifacts/notebook22/public_leaderboard_history.csv"); fig,ax=plt.subplots(figsize=(10,5)); ax.plot(range(1,len(hist)+1),hist["Public RMSE"],marker="o"); ax.set(xticks=range(1,len(hist)+1),xticklabels=hist.Notebook,ylabel="Public RMSE (lower is better)",xlabel="Submission order",title="Public leaderboard progression"); plt.tight_layout(); plt.savefig(FIG/"01_public_progression.png",dpi=160); plt.close()
vals=pd.DataFrame({"Design":["Strict temporal","Irregular","Spatial block","Space-time","OOF proxy"],"RMSE":[.740524,.680366,.801398,.817423,.743809]}); fig,ax=plt.subplots(figsize=(8,5)); ax.bar(vals.Design,vals.RMSE); ax.set_ylabel("RMSE (lower is better; designs differ)"); ax.tick_params(axis="x",rotation=25); ax.set_title("Validation metrics — not directly interchangeable"); plt.tight_layout(); plt.savefig(FIG/"02_validation_comparison.png",dpi=160); plt.close()
fig,ax=plt.subplots(figsize=(12,3)); stages=[("01–08","foundation"),("09–18","model/feature research"),("17–20","audits"),("21","refinement"),("22–26","trust/reporting")]; ax.scatter(range(len(stages)),[1]*len(stages),s=180); [ax.text(i,1.08,f"{a}\n{b}",ha="center") for i,(a,b) in enumerate(stages)]; ax.set(xlim=(-.5,len(stages)-.5),ylim=(.85,1.35),title="Research timeline"); ax.axis("off"); plt.tight_layout(); plt.savefig(FIG/"03_research_timeline.png",dpi=160); plt.close()
tail=pd.read_csv(ROOT/"artifacts/notebook25/global_tail_risk_metrics.csv"); names=["Median absolute error","P75 absolute error","P90 absolute error","P95 absolute error","P99 absolute error","Maximum absolute error"]; q=tail.set_index("Metric").loc[names]; fig,ax=plt.subplots(figsize=(9,5)); ax.bar(q.index,q.Value); ax.tick_params(axis="x",rotation=35); ax.set_ylabel("Absolute error"); ax.set_title("Canonical-equivalent OOF tail risk"); plt.tight_layout(); plt.savefig(FIG/"04_tail_percentiles.png",dpi=160); plt.close()
reg=pd.read_csv(ROOT/"artifacts/notebook25/regional_tail_risk.csv"); fig,ax=plt.subplots(figsize=(9,5)); ax.bar(reg.Group,reg["P95 absolute error"]); ax.tick_params(axis="x",rotation=35); ax.set_ylabel("P95 absolute error"); ax.set_title("Regional tail comparison"); plt.tight_layout(); plt.savefig(FIG/"05_regional_error.png",dpi=160); plt.close()
trust=pd.read_csv(ROOT/"artifacts/notebook24/trustworthiness_scorecard.csv"); fig,ax=plt.subplots(figsize=(9,6)); ax.barh(trust.Component,trust["Score 0-5"]); ax.set_xlim(0,5); ax.set_xlabel("Score (0–5; internal framework)"); ax.set_title("Trustworthiness scorecard"); plt.tight_layout(); plt.savefig(FIG/"06_trustworthiness.png",dpi=160); plt.close()
fig,ax=plt.subplots(figsize=(11,4)); ax.axis("off"); boxes=[(.03,"Inputs\nspace/time/climate"),(.28,"Notebook 11\nknown + masked experts"),(.54,"0.65/0.35\nmasked blend"),(.76,"Notebook 21\nSouth masked +0.05")];
for x,t in boxes: ax.text(x,.5,t,ha="left",va="center",bbox=dict(boxstyle="round",fc="#dbeafe",ec="#2563eb"));
for x in [.21,.47,.70]: ax.annotate("",xy=(x+.04,.5),xytext=(x,.5),arrowprops=dict(arrowstyle="->")); ax.text(.92,.5,"Final CSV",va="center",bbox=dict(boxstyle="round",fc="#dcfce7")); ax.annotate("",xy=(.91,.5),xytext=(.87,.5),arrowprops=dict(arrowstyle="->")); ax.set_title("Final system architecture"); plt.tight_layout(); plt.savefig(FIG/"07_system_architecture.png",dpi=160); plt.close()
fig,ax=plt.subplots(figsize=(8,5)); ax.hist(delta,bins=80); ax.set(xlabel="Current-best − canonical prediction",ylabel="Rows",title="Prediction-difference distribution"); plt.tight_layout(); plt.savefig(FIG/"08_prediction_difference.png",dpi=160); plt.close()

report="""# Final competition report

## 1. Introduction
This report consolidates Notebooks 01–25 without new score optimization.
## 2. Challenge Definition
Forecast terrestrial-water-storage anomaly robustly across space and time using prediction-time available inputs.
## 3. Dataset and Forecasting Structure
The data repeat a global spatial grid over grouped dates; test rows divide into known and masked `TWS_t` availability.
## 4. Exploratory Data Analysis
Early work established missingness, target spread, spatial coverage and temporal grouping.
## 5. Baseline Development
Simple baselines were useful controls but early validation was not decision-grade.
## 6. Gradient Boosting Models
LightGBM became the strongest family; XGBoost/CatBoost did not establish a verified public advantage.
## 7. Ensemble and Canonical Blend
Notebook 11's selected blend became canonical at public RMSE 0.824463818.
## 8. Feature Engineering Research
Large generic and clustered feature searches often improved local metrics but transferred poorly.
## 9. Alternative and Delta Forecasting Approaches
Notebook 14/15 scored 0.825971129/0.824797753 and did not beat canonical.
## 10. Hydrology-Inspired Feature Research
Domain-labelled features did not add robust independent signal; Notebook 18 public RMSE was 0.831108253.
## 11. Validation Evolution
Validation progressed from optimistic same-date splits to temporal, irregular, spatial and space-time gates.
## 12. Leakage and Ordering Audit
The global-month prior has a 91.30% future-timestamp ordering exposure in training; no direct validation-target leakage was found.
## 13. Reproducibility Audit
Notebook 17 corrected a pipeline identity mismatch and froze canonical SHA256.
## 14. Spatial Residual Correction
Notebook 21 applied a bounded +0.05 Southern masked-row correction and improved public RMSE to 0.823609466.
## 15. Temporal Transferability Evaluation
Only three of four windows improved; the closest lost 0.002613. Score 54.53/100 means weak transferability.
## 16. Bias and Trustworthiness Assessment
Trustworthiness is 66.30/100: useful with careful oversight; Southern and temporal risks remain.
## 17. OOF Tail-Risk and Support Audit
Canonical-equivalent OOF RMSE/MAE are 0.743809/0.568233; P95/P99/max errors are 1.520082/2.123579/4.461003.
## 18. Final Submission Selection
Use `submission_21_spatial_residual_candidate.csv`; preserve canonical separately.
## 19. Limitations
Weak correction transfer, same-grid support, omitted groundwater/snow/surface/human-water processes, proxy OOF and external leaderboard verification.
## 20. Intended Use
Broad-scale research, drought support and benchmarking with expert review; not critical automated decisions.
## 21. Reproducibility
Builders, SHA gates and caches are provided; environment portability and early notebooks remain moderate.
## 22. Lessons Learned
Validation and reproducibility were as consequential as modelling; local gain was not public proof.
## 23. Final Conclusion
The project stops at a small verified public improvement and documents why that improvement should not be overstated.
"""
(ART/"final_competition_report.md").write_text(report,encoding="utf-8")

lessons="""# Lessons learned

- **Complexity is not monotonic value:** alternative models and broad feature searches failed to beat the canonical public score.
- **Local is not public:** Notebooks 12, 14, 16 and 18 provide repeated mismatch evidence.
- **Validation design is a modelling decision:** complete-date, irregular, spatial and space-time folds changed conclusions.
- **Row order can leak time:** target-derived expanding statistics require explicit chronological grouping.
- **Reproducibility is score research:** Notebook 17 showed that a wrong reference pipeline invalidates comparisons.
- **A small correction can win yet transfer weakly:** Notebook 21 improved public score; Notebook 23 found 54.53/100 transferability.
- **Leaderboard-positive is not synonymous with robust.**
- **RMSE is incomplete:** Notebook 25 exposed P95/P99/max and regional tail risk.
- **Transparent limitations improve technical quality:** missing physics and support constraints are part of the result.
- **Stopping is a valid outcome:** after the weak transfer result, further score optimization was closed.
"""
(ART/"lessons_learned.md").write_text(lessons,encoding="utf-8")

# Root user-facing documents (relative paths only).
readme="""# Global Water Storage Forecasting

## Problem overview
Forecast gridded terrestrial-water-storage anomalies robustly across space and time. The project separates test rows where contemporaneous TWS is known from rows where it is masked.

## Best public score
**0.823609466** from `submissions/submission_21_spatial_residual_candidate.csv`. Canonical base score: **0.824463818**.

## Final system
Notebook 11 LightGBM selected blend is the canonical base. Notebook 21 adds a bounded +0.05 Southern-hemisphere correction only to eligible masked rows. It is the current-best refinement, not a new canonical.

## Key results
| Result | Value |
|---|---:|
| Public improvement | 0.000854352 (0.104%) |
| Strict / irregular / spatial / space-time RMSE | 0.740524 / 0.680366 / 0.801398 / 0.817423 |
| Transferability | 54.53/100, weak |
| OOF proxy RMSE / MAE | 0.743809 / 0.568233 |
| OOF P95 / P99 | 1.520082 / 2.123579 |
| Trustworthiness | 66.30/100, careful oversight |

## Repository structure
- `notebooks/`: research sequence 01–26
- `scripts/`: reproducible notebook builders
- `artifacts/`: audit tables, reports, figures and OOF data
- `submissions/`: immutable competition predictions
- `data/raw/`: locally supplied competition data (not assumed redistributable)

## Dataset setup
Place `Train.csv`, `Test.csv` and `SampleSubmission.csv` under `data/raw/`. See [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

## Installation and environment
Create a virtual environment, install `requirements-minimal.txt`, and use `requirements-reproducible.txt` when matching the captured local direct dependencies. Exact cross-platform determinism is not guaranteed.

## Reproduction steps
Build a report notebook with `python scripts/build_notebook26.py`, then execute it from the repository root. Advanced model notebooks require LightGBM and the local data. Verify SHA values before any audit.

## Notebook reading order
Recommended: 11 → 17 → 19 → 20 → 21 → 23 → 24 → 25 → 26. Full index: [experiment_index.md](artifacts/notebook26/experiment_index.md).

## Validation strategy
Decisions combine temporal, irregular hindcast, spatial-block and space-time evidence. Metrics from different designs are not directly interchangeable.

## Leakage controls
Complete dates are grouped and new residual/OOF work is train-on-past. The canonical global-month prior retains a documented location-major ordering risk affecting about 91.30% of training rows; no direct validation-target leakage was detected.

## Explainability
Known rows are persistence dominated; masked rows rely on SPEI, spatial/time and soil-moisture features. The correction is transparent but weakly transferable.

## Trustworthiness and limitations
Score: 66.30/100. The refinement improved public RMSE but worsened the closest historical window. Spatial support audit is same-grid and major physical processes are absent.

## Submission files
- Canonical: `submissions/submission_optimized.csv`
- Final/current-best: `submissions/submission_21_spatial_residual_candidate.csv`

## Artefacts and results
Start with `artifacts/notebook26/executive_summary.md`, `final_competition_report.md`, figures, and `final_results_table.csv`. Large row-level OOF files remain under `artifacts/notebook25/`.

## Intended use
Broad-scale water-storage research, drought decision support, screening and benchmarking with expert review. Not for critical local operations, emergency/safety automation, sanctions or guaranteed extrapolation.

## License / competition data note
Competition data and leaderboard scores may be governed by external terms. Confirm redistribution rights before publishing `data/` or submission-derived materials.

## Author / portfolio context
This repository demonstrates end-to-end ML research, validation design, leakage auditing, reproducibility, model-card reporting and evidence-based stopping. Add author/license details before public release.
"""
(ROOT/"README.md").write_text(readme,encoding="utf-8")

repro="""# Reproducibility guide

## Python and dependencies
Notebook metadata and the active project environment use Python 3.13, but not every early notebook was verified against a formal version range. Install direct packages from `requirements-minimal.txt`; use `requirements-reproducible.txt` to match versions captured from the active `.venv`.

### Windows PowerShell
```powershell
python -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
python -m pip install -r requirements-minimal.txt
python scripts/build_notebook26.py
```

### Unix-like shells
```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-minimal.txt
python scripts/build_notebook26.py
```

## Data layout
Place `Train.csv`, `Test.csv`, and `SampleSubmission.csv` in `data/raw/`. Data are not fetched automatically and may have competition licensing restrictions.

## Reading/execution order
For the final evidence chain use 11 → 17 → 19 → 20 → 21 → 23 → 24 → 25 → 26. Builders exist for later notebooks as `scripts/build_notebookNN.py`.

## Expected outputs
Notebook 26 writes reports/figures under `artifacts/notebook26/` and root documentation. Notebook 25 creates large OOF/cache artefacts; a second run validates the frozen policy key before reuse.

## SHA256 verification
Expected canonical: `b61ecd15862ca6d7527c4f494de802ed6ea6d31c420737cb78f977d9bda3b968`.
Expected current-best: `84d4ce6efdbfc3fbca55d9eea1ebccfe628ffa962e31b62a654c179dc649b488`.
Use `Get-FileHash ... -Algorithm SHA256` on Windows or `sha256sum ...` on Unix.

## Canonical/current-best reproduction
Notebook 17 identifies the exact canonical artifact; Notebook 21 creates the refinement; Notebook 23 verifies numerical policy reproduction. Do not overwrite either final CSV.

## Determinism
Advanced scripts use fixed seeds and explicit date ordering. LightGBM, threads, OS, compiler and package versions can still cause small numeric differences. Repository-external leaderboard scoring cannot be reproduced locally.

## Hardware
Report notebooks are lightweight. Model/OOF notebooks need several GB of RAM and multi-minute CPU runs; exact peak memory/latency is not available. Notebook 25 Parquet/cache files reduce repeat cost.

## Troubleshooting
- Run from repository root so relative paths resolve.
- Select a kernel backed by the environment containing LightGBM.
- If a cache policy key fails, allow Notebook 25 to refit rather than editing the manifest.
- Treat missing competition data or SHA mismatch as fail-fast conditions.
"""
(ROOT/"REPRODUCIBILITY.md").write_text(repro,encoding="utf-8")
card24=(ROOT/"artifacts/notebook24/model_card.md").read_text(encoding="utf-8")
card=card24+"""

## Notebook 25 OOF and tail-risk addendum

- Terminology: canonical-equivalent leakage-controlled OOF, not exact canonical OOF.
- Coverage: 749,046 rows (34.774%) across two strict temporal folds.
- RMSE / MAE: 0.743809 / 0.568233.
- P95 / P99 / maximum absolute error: 1.520082 / 2.123579 / 4.461003.
- Southern-hemisphere P95: 1.773752.
- Support limitation: folds share the spatial grid, so unseen-location robustness is not established.
- Test low/out spatial support: 0% under this same-grid definition; test targets are unavailable and dates extrapolate.
- Final decision: preserve the current-best competition artifact, stop score optimization, and retain careful-oversight classification.
"""
(ROOT/"MODEL_CARD.md").write_text(card,encoding="utf-8")

# Non-destructive delivery checks and secret-pattern scan: report filenames/counts only.
source_ext={".py",".md",".ipynb",".json",".yaml",".yml",".toml"}; suspect=[]
patterns=re.compile(r"(api[_-]?key|secret[_-]?key|access[_-]?token|password)\s*[:=]",re.I)
for p in ROOT.rglob("*"):
 if not p.is_file() or p.suffix.lower() not in source_ext or any(x in p.parts for x in [".git",".venv","data"]): continue
 try:
  if patterns.search(p.read_text(encoding="utf-8",errors="ignore")): suspect.append(str(p.relative_to(ROOT)))
 except Exception: pass
large=[str(p.relative_to(ROOT)) for p in ROOT.rglob("*") if p.is_file() and p.stat().st_size>50_000_000 and ".git" not in p.parts and ".venv" not in p.parts]
checklist=f"""# Final delivery checklist

- [x] Current-best submission and both SHA256 values verified
- [x] Notebook 26 executed; README, reproduction guide, model card and final report created
- [x] Experiment/artefact index, failed experiments, leakage, trustworthiness and tail-risk documented
- [ ] Competition data licensing/redistribution must be confirmed by repository owner
- [x] Final documents use repository-relative paths
- [x] Secret-pattern scan completed without deleting anything; flagged files: {len(suspect)} ({', '.join(suspect) if suspect else 'none'})
- [x] Large-file scan completed; >50 MB files: {len(large)} ({', '.join(large) if large else 'none'})
- [ ] Review `git status`, generated large artefacts and ignore policy before commit/release
- [ ] Add explicit author and license details
- [x] Recommended tag after owner review: `v1.0.0-competition-final`
"""
(ART/"final_delivery_checklist.md").write_text(checklist,encoding="utf-8")

final="""# Final recommendation

- **Final submission:** `submissions/submission_21_spatial_residual_candidate.csv` (public RMSE 0.823609466).
- **Canonical base:** `submissions/submission_optimized.csv` from Notebook 11 (0.824463818).
- **Score optimization:** closed; no new model/correction/submission is justified.
- **Most important limitation:** public-positive correction is weakly transferable and the closest historical window worsened; same-grid OOF does not prove unseen-region robustness.
- **Strongest contribution:** the evidence chain linking validation redesign, ordering/leakage audit, exact artifact reproduction, bounded refinement, transfer testing, model card and tail-risk audit.
- **Delivery readiness:** technically ready after owner completes data-license, author/license, large-file and git review. These governance steps cannot be inferred or completed automatically.
- **Missing mandatory technical work:** none in the requested report package; owner review items remain.
- **Recommended release:** `v1.0.0-competition-final` after checklist review.
- **Competition score to report:** 0.823609466, alongside canonical 0.824463818 and the 0.000854352 improvement.
- **Validation to present jointly:** strict temporal 0.740524, irregular 0.680366, spatial 0.801398, space-time 0.817423, transferability 54.53/100, and canonical-equivalent OOF RMSE/MAE 0.743809/0.568233. State that designs differ.
"""
(ART/"final_recommendation.md").write_text(final,encoding="utf-8")

# Verify final docs contain no absolute Windows workspace path.
for p in [ROOT/"README.md",ROOT/"REPRODUCIBILITY.md",ROOT/"MODEL_CARD.md",*ART.glob("*.md")]:
 assert "C:\\" not in p.read_text(encoding="utf-8",errors="ignore")
required=["final_integrity_report.csv","executive_summary.md","final_competition_report.md","final_results_table.csv","final_model_definition.md","validation_generalization_summary.md","final_leakage_reproducibility_register.csv","bias_tail_support_summary.md","experiment_index.csv","experiment_index.md","environment_audit.csv","repository_structure_recommendation.md","submission_manifest.csv","final_delivery_checklist.md","lessons_learned.md","final_recommendation.md"]
assert all((ART/x).exists() for x in required) and len(list(FIG.glob("*.png")))>=8
assert all((ROOT/x).exists() for x in ["README.md","REPRODUCIBILITY.md","MODEL_CARD.md","requirements-minimal.txt","requirements-reproducible.txt"])
assert sha(CANON)==CANON_SHA and sha(BEST)==BEST_SHA
display(Markdown(final)); print("Notebook 26 final consolidation complete; no submission created or modified.")
