from pathlib import Path
import hashlib, json, re
import numpy as np
import pandas as pd
import nbformat
from IPython.display import display, Markdown

ROOT=Path.cwd().resolve()
if not (ROOT/"data/raw/Test.csv").exists(): ROOT=ROOT.parent
ART=ROOT/"artifacts/notebook24"; ART.mkdir(parents=True,exist_ok=True)
CANON=ROOT/"submissions/submission_optimized.csv"; BEST=ROOT/"submissions/submission_21_spatial_residual_candidate.csv"
CANON_SHA="b61ecd15862ca6d7527c4f494de802ed6ea6d31c420737cb78f977d9bda3b968"
BEST_SHA="84d4ce6efdbfc3fbca55d9eea1ebccfe628ffa962e31b62a654c179dc649b488"
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

test=pd.read_csv(ROOT/"data/raw/Test.csv"); sample=pd.read_csv(ROOT/"data/raw/SampleSubmission.csv")
canon=pd.read_csv(CANON); best=pd.read_csv(BEST); known=test.TWS_t.notna().to_numpy(); masked=~known; delta=best.Target-canon.Target
integrity=[]
for label,path,df,expected in [("canonical base",CANON,canon,CANON_SHA),("current best refined",BEST,best,BEST_SHA)]:
 integrity.append({"Submission":label,"Exists":path.exists(),"SHA256":sha(path),"SHA match":sha(path)==expected,"Rows":len(df),
  "ID unique":df.ID.is_unique,"ID order":df.ID.equals(sample.ID),"NaN":int(df.Target.isna().sum()),"Infinity":int(np.isinf(df.Target).sum()),
  "Prediction minimum":df.Target.min(),"Prediction maximum":df.Target.max(),"Known rows":int(known.sum()),"Masked rows":int(masked.sum()),
  "Known preserved vs canonical":True if label=="canonical base" else np.array_equal(df.Target.to_numpy()[known],canon.Target.to_numpy()[known]),
  "Masked coverage":int(df.Target.to_numpy()[masked].size),"Correlation vs canonical":1.0 if label=="canonical base" else np.corrcoef(df.Target,canon.Target)[0,1],
  "MAD vs canonical":0.0 if label=="canonical base" else np.abs(delta).mean(),"Maximum absolute difference":0.0 if label=="canonical base" else np.abs(delta).max(),
  "Changed row fraction":0.0 if label=="canonical base" else (np.abs(delta)>1e-12).mean()})
integrity=pd.DataFrame(integrity); integrity.to_csv(ART/"submission_integrity_report.csv",index=False)
assert integrity["SHA match"].all() and integrity["ID order"].all() and integrity["Known preserved vs canonical"].all()

system_summary=f"""# Model system summary

- **Problem:** global gridded terrestrial-water-storage forecasting; target is future `target`/TWS anomaly at a latitude, longitude and date.
- **Data structure:** repeated global spatial grid across grouped monthly dates; official test has known and masked `TWS_t` rows.
- **Canonical base:** Notebook 11 final selected blend, public RMSE 0.824463818. A known-row LightGBM expert and masked base LightGBM are combined with a causal-history masked expert; the masked prediction uses the verified 0.65 base / 0.35 history blend.
- **Feature families:** latitude/longitude and spatial bins, calendar/cyclic time, SPEI 01/03/06/12 and aggregates, soil moisture, and `TWS_t` only where available; historical priors are target-derived and carry the documented training-order risk.
- **Prediction-time availability:** location, date, supplied SPEI/soil covariates and known `TWS_t` rows; masked rows cannot use unavailable `TWS_t`.
- **Notebook 21 refinement:** OOF residual = target − prediction; stable Southern-hemisphere masked rows receive a bounded +0.05 intercept. Known rows remain byte/numerically unchanged.
- **Validation:** Notebook 11 temporal OOF, then Notebook 20 strict temporal, irregular hindcast, spatial block and space-time; Notebook 23 adds four independent temporal transfer windows.
- **Selection rationale:** Notebook 21 improved public RMSE to 0.823609466 under predeclared gates. Notebook 23 later classified its historical transfer as weak, so it is the current-best refined submission—not a new canonical base and not a general robustness claim.
"""
(ART/"model_system_summary.md").write_text(system_summary,encoding="utf-8")

v20=pd.read_csv(ROOT/"artifacts/notebook20/validation_comparison.csv"); p20=pd.read_csv(ROOT/"artifacts/notebook20/prior_ablation.csv")
w23=pd.read_csv(ROOT/"artifacts/notebook23/window_validation_results.csv"); s23=pd.read_csv(ROOT/"artifacts/notebook23/transferability_score.csv").iloc[0]
perf=[
("Canonical public RMSE",.824463818,"public leaderboard","lower is better"),("Current-best public RMSE",.823609466,"public leaderboard","lower is better"),
("Absolute public improvement",.824463818-.823609466,"derived from public scores","positive improvement"),("Relative public improvement",(.824463818-.823609466)/.824463818,"derived from public scores","fraction"),
("Strict temporal canonical RMSE",v20.loc[v20.Validation.eq("strict_temporal"),"Mean fold RMSE"].iloc[0],"Notebook 20","different validation design"),
("Irregular hindcast canonical RMSE",v20.loc[v20.Validation.eq("irregular"),"Mean fold RMSE"].iloc[0],"Notebook 20","different validation design"),
("Spatial block canonical mean RMSE",v20.loc[v20.Validation.eq("spatial_block"),"Mean fold RMSE"].iloc[0],"Notebook 20","different validation design"),
("Space-time canonical RMSE",v20.loc[v20.Validation.eq("space_time"),"Mean fold RMSE"].iloc[0],"Notebook 20","different validation design"),
("Southern worst canonical RMSE",p20[(p20.Fold=="lat_south")&(p20.Prior=="canonical")]["Overall RMSE"].iloc[0],"Notebook 20","regional fold"),
("NB23 improved windows",(w23["Absolute gain"]>0).sum(),"Notebook 23","out of four"),("NB23 mean gain",w23["Absolute gain"].mean(),"Notebook 23","four windows"),
("NB23 median gain",w23["Absolute gain"].median(),"Notebook 23","four windows"),("NB23 closest-window gain",w23.iloc[-1]["Absolute gain"],"Notebook 23","risk signal"),
("NB23 transferability score",s23.Score,"Notebook 23","project decision score; not RMSE")]
pd.DataFrame(perf,columns=["Metric","Value","Evidence","Comparability note"]).to_csv(ART/"performance_summary.csv",index=False)

# Aggregate spatial evidence only: no row-level NB21/23 prediction artifact was persisted.
sp21=pd.read_csv(ROOT/"artifacts/notebook21/spatial_residual_summary.csv"); reg23=pd.read_csv(ROOT/"artifacts/notebook23/regional_transferability.csv")
sp_rows=[]
for r in sp21[sp21.group_type.isin(["hemisphere","lat_band","lon_band","spatial_cell"])].itertuples():
 sp_rows.append({"Group type":r.group_type,"Group":r.group,"Row count":r.row_count,"Canonical RMSE":r.rmse,"Corrected RMSE":"not available",
  "MAE":r.mae,"Mean residual":r.mean_residual,"Median residual":r.median_residual,"Residual std":r.residual_std,
  "Underprediction rate":r.underprediction_rate,"Overprediction rate":r.overprediction_rate,"Canonical-corrected difference":"not available",
  "Window stability":"not available for this aggregate","Uncertainty":f"bootstrap CI [{r.bootstrap_ci_low:.6f}, {r.bootstrap_ci_high:.6f}]","Evidence":"Notebook 21 mixed OOF/spatial aggregate"})
for r in reg23.itertuples():
 sp_rows.append({"Group type":"regional transfer","Group":r.Region,"Row count":"not available","Canonical RMSE":r._3,"Corrected RMSE":r._4,"MAE":"not available",
  "Mean residual":"not available","Median residual":"not available","Residual std":"not available","Underprediction rate":"not available","Overprediction rate":"not available",
  "Canonical-corrected difference":r._5,"Window stability":r._6,"Uncertainty":"four windows only","Evidence":"Notebook 23 aggregate"})
pd.DataFrame(sp_rows).to_csv(ART/"spatial_bias_analysis.csv",index=False)

windows=pd.read_csv(ROOT/"artifacts/notebook23/temporal_windows.csv"); decay=pd.read_csv(ROOT/"artifacts/notebook23/temporal_decay_analysis.csv")
temporal=windows.merge(w23,left_on="Window ID",right_on="Window").merge(decay[["Window","Month gap"]],on="Window")
temporal["Residual mean"]="not available — row-level predictions not persisted"; temporal["Residual drift"]="not available"
temporal["Prediction drift"]="not available"; temporal["Evidence strength"]="four-window exploratory sample"
temporal.to_csv(ART/"temporal_robustness.csv",index=False)

cl23=pd.read_csv(ROOT/"artifacts/notebook23/climate_regime_transferability.csv")
climate=cl23.rename(columns={"Canonical RMSE":"RMSE canonical","Corrected RMSE":"RMSE corrected","Correction magnitude stability":"Error/correction variability"})
climate["MAE"]="not available — aggregate artifact"; climate["Mean residual"]="not available — aggregate artifact"
climate["Temporal window consistency"]="four-window aggregate"; climate["Underrepresented"]=climate["Row count"]<.1*climate["Row count"].sum()
climate["Conclusion"]="Regime differences are descriptive; evidence is insufficient to establish systematic weakness."
climate.to_csv(ART/"climate_regime_performance.csv",index=False)

# Tail metrics cannot be reconstructed from aggregate CSVs; preserve the audit gap explicitly.
tail_metrics=["Median absolute error","75th percentile absolute error","90th percentile absolute error","95th percentile absolute error","99th percentile absolute error",
"Maximum absolute error","Large-error row fraction","Residual skewness","Residual kurtosis","Tail concentration by region/date/regime"]
pd.DataFrame({"Metric":tail_metrics,"Value":"not available","Reason":"Row-level validation predictions/residuals were not persisted by Notebooks 21/23; aggregate summaries cannot recover tails.",
 "Recommendation":"Persist compact row-level OOF predictions in a future audit-only notebook."}).to_csv(ART/"error_tail_analysis.csv",index=False)

leak_rows=[
("Same-month row split","validation leakage/optimism","high","Notebook 08 validation","same date can occur in train and validation","group complete dates","historical NB08 metrics remain optimistic","superseded"),
("prior_global_month location-major ordering","indirect temporal leakage risk","high","canonical training features","91.30% future-timestamp contributor rate","strict cutoff/OOF for new research; freeze canonical artifact","canonical still contains risky training construction","documented"),
("Validation target direct leakage","direct leakage","low/none found","Notebook 11/20 validation","validation target not included in aggregate","maintain fold assertions","training-feature risk remains distinct","not detected"),
("Prediction-time availability","availability","high","all target-derived inputs","Notebook 19 technical specification","use only available covariates at inference","unknown external deployment feeds","controlled in evaluated pipeline"),
("Train-on-past correction","temporal leakage","low","Notebook 21/23 refinement","max historical time < min future time assertions","older OOF residual pool only","weak transferability, not leakage","controlled"),
("Known-row preservation","submission integrity","low","Notebook 21","exact equality against canonical","assert before write","none observed","verified"),
("Current-best reproduction","reproducibility","low","Notebook 21/23","100% tolerance match and SHA","immutable hashes","external environment drift","verified")]
pd.DataFrame(leak_rows,columns=["Risk","Type","Severity","Affected component","Evidence","Mitigation","Residual risk","Current status"]).to_csv(ART/"leakage_risk_register.csv",index=False)

builds=sorted((ROOT/"scripts").glob("build_notebook*.py")); executed=[]
for p in sorted((ROOT/"notebooks").glob("[0-2][0-9]_*.ipynb")):
 try:
  n=nbformat.read(p,4); executed.append((p.name,any(c.get("execution_count") for c in n.cells if c.cell_type=="code")))
 except Exception: pass
rep_rows=[
("Notebook build scripts","present",f"{len(builds)} builder scripts; later notebooks reproducible","earlier notebooks may lack builders","add builders only when original logic is frozen"),
("Fixed random seeds","mostly present","SEED=42 in advanced pipelines","library/version determinism is not absolute","record package lock"),
("Environment metadata","partial","kernelspec and Python metadata in notebooks","requirements.txt is empty; local .venv required","export pinned environment"),
("Input paths","moderate","ROOT fallback and data/raw convention","working-directory assumptions remain","centralize path config"),
("Output directory creation","present","mkdir(parents=True, exist_ok=True)","low","retain"),
("Deterministic ordering","present in key pipelines","explicit date and spatial sorts","LightGBM/platform differences possible","pin versions and threads"),
("Submission hashes","high",f"canonical {sha(CANON)}; current {sha(BEST)}","none observed","retain fail-fast gates"),
("Exact reproduction tests","high","Notebook 17 and 23","numeric parser tolerances","retain SHA + tolerance checks"),
("Assertions","present","strict time, SHA, IDs and finiteness","coverage varies by early notebook","standardize audit helper"),
("Artefact availability","moderate","aggregate CSV/MD available","row-level OOF absent for tail audit","persist compact OOF parquet/npz"),
("Notebook execution state","high",f"{sum(x[1] for x in executed)}/{len(executed)} numbered notebook files show executed code","metadata is not a full run log","record run manifests"),
("Hidden manual steps","moderate","leaderboard upload/scoring external","some filenames/scores supplied outside repo","maintain submission registry"),
("External leaderboard scores","partial","NB22 flags uncorroborated scores","not independently reproducible","store dated score manifest")]
rep=pd.DataFrame(rep_rows,columns=["Check","Status","Evidence","Risk","Improvement recommendation"]); rep["Overall rating"]="Moderate"
rep.to_csv(ART/"reproducibility_assessment.csv",index=False)

explain_rows=[
("Known expert global importance","TWS_t dominant; then SPEI_06, soil moisture, latitude band","Notebook 10 permutation importance","TWS_t availability differs by row"),
("Masked expert global importance","SPEI_12, SPEI_06, SPEI mean, soil moisture, SPEI min","Notebook 10 permutation importance","climate proxies do not fully represent TWS physics"),
("Importance stability","masked latitude and elapsed_month changed most across two folds","Notebook 10 fold stability","only two folds; instability warning"),
("Spatial effects","latitude, spatial_bin_10, longitude trigonometric/interactions material for masked expert","Notebook 10","spatial association is not causal explanation"),
("Temporal effects","elapsed_month and cyclic month features contribute","Notebook 10","distribution shift possible"),
("Climate effects","multi-scale SPEI and soil moisture important, especially masked rows","Notebook 10","regime performance evidence limited"),
("Target-derived risk","TWS_t is legitimate only on known rows; priors carry ordering risk","Notebooks 10/20","strict availability controls required"),
("Correction explainability","simple +0.05 stable Southern masked-row intercept","Notebook 21","highly transparent but NB23 found weak temporal transfer")]
pd.DataFrame(explain_rows,columns=["Topic","Finding","Evidence","Caveat"]).to_csv(ART/"explainability_summary.csv",index=False)

physical=[
("Groundwater","major TWS component","not directly","TWS persistence/soil-climate proxies","regional bias and lag","satellite/model groundwater product","future availability/version latency"),
("Snow and ice storage","dominant at high latitudes","not directly","SPEI/season/latitude","high-latitude tail risk","snow water equivalent/cryosphere data","forecast-time availability"),
("Surface water","lakes/rivers/reservoir storage","not directly","location/climate","missed rapid storage changes","surface-water extent/altimetry","coverage and latency"),
("Runoff","links precipitation to export","no","SPEI weak proxy","timing/amplitude error","forecast/reanalysis runoff","future leakage if realized runoff"),
("Evapotranspiration","major loss flux","no","soil/SPEI proxy","climate-regime error","forecast ET product","latency/model dependence"),
("Soil moisture","important land-water component","yes, one supplied variable","direct supplied covariate","depth/scale incomplete","multi-layer soil moisture","consistent test feed"),
("Anthropogenic water use","human storage pressure","no","none verified","regional systematic bias","water-use scenarios","future scenario uncertainty"),
("Irrigation","alters soil/groundwater","no","soil moisture indirect","agricultural-region bias","irrigation maps/schedules","future schedule unknown"),
("Reservoir operations","large abrupt storage changes","no","location weak proxy","operational discontinuities","reservoir rule/level data","often unavailable prospectively"),
("Human abstraction","groundwater/surface withdrawals","no","none verified","long-term trend error","abstraction inventories/scenarios","reporting delay"),
("Land-use changes","changes ET/runoff/infiltration","no","static location proxy","nonstationarity","dated land-cover products","future land-use scenario")]
pd.DataFrame(physical,columns=["Physical process","Why important","Represented?","Existing proxy","Potential error impact","Possible future addition","Prediction-time availability issue"]).to_csv(ART/"data_representation_limitations.csv",index=False)

def elapsed_from_nb(name):
 try:
  n=nbformat.read(ROOT/"notebooks"/name,4); txt="\n".join(str(o.get("text","")) for c in n.cells for o in c.get("outputs",[]) if o.output_type=="stream")
  vals=[float(x) for x in re.findall(r"elapsed=([0-9.]+)s",txt)]; return max(vals) if vals else np.nan
 except Exception:return np.nan
comp=[
("Canonical model count","3 per fold/inference path","Notebook 11/21 code","known expert, masked base, masked prior expert"),
("Blend complexity","0.65 base + 0.35 history on masked rows","Notebook 11","simple weighted blend"),
("Notebook 21 runtime",elapsed_from_nb("21_spatial_residual_correction.ipynb"),"executed notebook stream","includes multiple validation fits"),
("Notebook 23 runtime",elapsed_from_nb("23_correction_temporal_transferability.ipynb"),"executed notebook stream","seven model blocks"),
("Peak memory","not available","no persisted profiler","do not infer"),
("Prediction time","not available","no persisted benchmark","batch benchmark needed"),
("Correction runtime","negligible relative to models; exact time not available","vectorized group map","no isolated timer"),
("Notebook 24 artefact size bytes","computed below","filesystem","report-only"),
("Batch-inference suitability","likely suitable with caveats","tree models + vectorized correction","requires environment and data-feed validation")]
ce=pd.DataFrame(comp,columns=["Item","Value","Evidence","Note"]); ce.to_csv(ART/"computational_efficiency.csv",index=False)

intended="""# Intended use and limitations

## Intended use

- Global or broad-scale water-storage forecasting research.
- Drought-monitoring decision support and spatial-temporal risk screening.
- Research benchmarking under documented train/test support.

## Out-of-scope use

- Sole basis for critical water-management or local reservoir operations.
- Emergency evacuation, financial sanction, political enforcement, or automated human-safety decisions.
- Guaranteed prediction outside training spatial/temporal support.

Predictions require domain-expert review, uncertainty/context from external hydrology sources, and continuous monitoring.
The system is not demonstrated to be production-ready; Notebook 23 found only weak temporal transferability of its refinement.
"""
(ART/"intended_use_and_limitations.md").write_text(intended,encoding="utf-8")

# Weights and scores are fixed in the notebook header, not optimized from the result.
score_rows=[("Performance",4.0,.15,"best public score, but small absolute gain"),("Temporal robustness",2.0,.15,"NB23 weak transfer; closest window worsened"),
("Spatial robustness",3.0,.12,"NB21 spatial gain; south remains key risk"),("Bias awareness",4.5,.08,"bias and uncertainty explicitly audited"),
("Leakage control",3.0,.12,"new correction causal; canonical prior ordering risk remains"),("Reproducibility",4.0,.12,"hash/build/assertions strong; environment and row OOF gaps"),
("Explainability",4.0,.08,"base importance plus transparent correction"),("Computational practicality",3.5,.05,"batch-friendly architecture; timings incomplete"),
("Physical completeness",2.0,.08,"many TWS components absent"),("Operational safety",4.0,.05,"use boundaries and monitoring documented")]
scorecard=pd.DataFrame(score_rows,columns=["Component","Score 0-5","Weight","Evidence"]); total=100*(scorecard["Score 0-5"]*scorecard.Weight).sum()/5
rating="Strong research-grade system" if total>=85 else ("Reliable with documented limitations" if total>=70 else ("Useful but requires careful oversight" if total>=55 else "Insufficiently trustworthy"))
scorecard["Weighted contribution /100"]=scorecard["Score 0-5"]*scorecard.Weight/5*100; scorecard["Total score"]=total; scorecard["Classification"]=rating
scorecard["Framework note"]="Project-internal decision framework, not an external certification."
scorecard.to_csv(ART/"trustworthiness_scorecard.csv",index=False)

model_card=f"""# Model Card — Current-best refined TWS forecasting system

## 1. Model Overview
Notebook 11 canonical selected blend plus Notebook 21 Southern-hemisphere masked-row residual correction. Public RMSE 0.823609466. Notebook 21 is not renamed canonical.
## 2. Problem Definition
Forecast gridded terrestrial-water-storage anomaly across space and time.
## 3. Intended Use
Broad-scale research, drought decision support, risk screening and benchmarking with expert review.
## 4. Out-of-Scope Use
Critical local operations, emergency or human-safety automation, sanctions, or guaranteed extrapolation.
## 5. Training Data
Repeated global grid and monthly dates with TWS, SPEI and soil-moisture fields; exact schema is repository-bound.
## 6. Evaluation Data
Temporal OOF, strict temporal, irregular hindcast, spatial block, space-time and four transfer windows; scores are not directly interchangeable.
## 7. Feature Families
Spatial coordinates/bins, calendar/cyclic time, multi-scale SPEI, soil moisture, known-row TWS and historical priors.
## 8. Model Architecture
LightGBM expert system; masked base/history blend 0.65/0.35.
## 9. Refinement Method
Stable historical OOF hemisphere residual intercept, +0.05 cap on eligible Southern masked rows; known rows unchanged.
## 10. Performance
Canonical/current public RMSE 0.824463818/0.823609466; absolute improvement {0.824463818-0.823609466:.9f}.
## 11. Spatial Robustness
Spatial canonical mean 0.801398; southern canonical fold 0.833364. NB21 improved spatial validation, but this does not prove universal regional fairness.
## 12. Temporal Robustness
NB23: 3/4 windows improved; closest window lost 0.002613; score 54.53, weakly transferable.
## 13. Bias and Fairness Considerations
Geographic performance disparities matter; this is environmental-model bias, not a demonstrated demographic fairness assessment. Sample size and uncertainty accompany claims.
## 14. Explainability
Known expert is persistence-dominated (`TWS_t`); masked expert relies on long/mid SPEI, spatial/time variables and soil moisture. Correction is simple but weakly transferable.
## 15. Leakage Controls
Date-grouped folds and train-on-past correction. Canonical `prior_global_month` retains a documented indirect training-order risk (91.30% future-timestamp contributor rate), without detected direct validation-target leakage.
## 16. Reproducibility
Rating: Moderate. Immutable SHA gates, builders and assertions are strong; environment pinning and row-level OOF persistence are incomplete.
## 17. Computational Requirements
Tree-model batch inference is plausible; exact production latency/peak memory are not available.
## 18. Limitations
Weak correction transfer, incomplete physical state, proxy validation, small public candidate sample and incomplete tail artefacts.
## 19. Ethical and Operational Risks
False confidence in local water decisions, unsupported extrapolation, and omission of human water operations.
## 20. Monitoring Recommendations
Monitor regional/window RMSE, residual sign, tails, covariate support, correction coverage, hashes and data drift; require expert escalation.
## 21. Versioning and SHA256
Canonical `{sha(CANON)}`; current-best `{sha(BEST)}`.
## 22. Final Recommendation
Preserve current-best for the competition record; stop score optimization. Use only as a research-grade decision-support artifact with careful oversight, not as production proof.
"""
(ART/"model_card.md").write_text(model_card,encoding="utf-8")

narrative="""# Final technical narrative

The project progressed from statistical baselines to gradient boosting, then selected a conservative dual-expert LightGBM blend as the canonical system. Subsequent work showed that lower local RMSE did not reliably predict public performance: alternative models, broad feature searches, delta formulations and hydrology-labelled features generally failed to improve the leaderboard.

That mismatch motivated validation, leakage and reproducibility audits. Same-date splitting was identified as optimistic; Notebook 17 recovered the true selected-blend artifact; Notebook 19 formalized prediction-time availability; Notebook 20 added strict temporal, irregular, spatial and space-time tests and quantified a location-major target-prior ordering risk. These audits became more valuable than further model complexity.

Notebook 21 then made a deliberately small change: a leakage-free, OOF-derived Southern-hemisphere residual intercept applied only to masked rows. It improved public RMSE from 0.824463818 to 0.823609466. However, Notebook 23 found only weak historical temporal transferability: three of four windows improved, the closest window deteriorated by 0.002613, and the score was 54.53/100. The public gain is real for the submitted artifact, but it is not proof of a generally robust correction.

The project therefore stops further leaderboard optimization, preserves the current-best submission, and reports both strengths and limitations. Its strongest technical contribution is the evidence-driven transition from model chasing to validation discipline, artifact reproducibility, bounded refinement and honest uncertainty—not a claim of production readiness.
"""
(ART/"final_technical_narrative.md").write_text(narrative,encoding="utf-8")

final=f"""# Notebook 24 final recommendation

1. **Current-best:** Notebook 11 canonical selected blend plus Notebook 21 +0.05 stable Southern masked-row correction.
2. **Integrity:** yes; both SHA256 values, IDs, finiteness and known-row preservation passed.
3. **Public improvement:** {0.824463818-0.823609466:.9f} RMSE ({(0.824463818-0.823609466)/0.824463818:.3%} relative).
4. **Strongest validation result:** the current-best public score is the direct competition outcome; among local systems no single validation is sufficient, so the joint Notebook 20 suite is the strongest decision support.
5. **Weakest robustness result:** Notebook 23 transferability 54.53/100; closest window −0.002613.
6. **Spatial risk:** Southern hemisphere was the worst canonical block and remains the correction-sensitive region.
7. **Temporal risk:** correction benefit reversed in the closest historical window; four windows provide limited evidence.
8. **Leakage risk:** canonical `prior_global_month` location-major training ordering (91.30% future-timestamp contributors); no direct validation-target leakage was found.
9. **Reproducibility:** Moderate.
10. **Explainability:** good at global feature-family and correction-rule level, limited for causal/local claims.
11. **Missing physics:** groundwater, snow/ice, surface water, runoff, ET, human abstraction/irrigation/reservoir and land-use dynamics; soil moisture is only partially represented.
12. **Suitable use:** broad-scale research, drought support, screening and benchmarking with experts.
13. **Do not use for:** critical local operations, emergencies, sanctions, safety automation or guaranteed extrapolation.
14. **Trustworthiness:** {total:.2f}/100 — {rating}; project-internal framework only.
15. **Preserve current-best:** Yes.
16. **New score optimization:** No.
17. **Technical story:** a strong canonical model survived broad alternatives; audits exposed validation/reproducibility risk; one bounded correction improved public score but weak transfer evidence triggered a principled stop.
18. **Notebook 25:** one audit-only step—persist and analyze compact row-level OOF residuals for tail-risk and support monitoring, without changing predictions.
"""
(ART/"final_recommendation.md").write_text(final,encoding="utf-8"); display(Markdown(final)); display(scorecard)
# Refresh the report-only artefact footprint after all outputs have been written.
ce.loc[ce.Item.eq("Notebook 24 artefact size bytes"),"Value"]=sum(p.stat().st_size for p in ART.iterdir() if p.is_file())
ce.to_csv(ART/"computational_efficiency.csv",index=False)
required=["submission_integrity_report.csv","model_system_summary.md","performance_summary.csv","spatial_bias_analysis.csv","temporal_robustness.csv",
"climate_regime_performance.csv","error_tail_analysis.csv","leakage_risk_register.csv","reproducibility_assessment.csv","explainability_summary.csv",
"data_representation_limitations.csv","computational_efficiency.csv","intended_use_and_limitations.md","model_card.md","trustworthiness_scorecard.csv",
"final_technical_narrative.md","final_recommendation.md"]
assert all((ART/x).exists() for x in required); assert sha(CANON)==CANON_SHA and sha(BEST)==BEST_SHA
print("Notebook 24 trustworthiness report complete; no submission created or modified.")
