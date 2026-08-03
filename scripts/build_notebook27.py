"""Build Notebook 27: research-only competition gap analysis.

The generated notebook writes only new artifacts under artifacts/notebook27.
It trains no model and creates no predictions or submissions.
"""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "27_competition_gap_analysis.ipynb"

nb = nbf.v4.new_notebook()
nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
               "language_info": {"name": "python", "version": "3"}}

def md(text): nb.cells.append(nbf.v4.new_markdown_cell(text.strip()))
def code(text): nb.cells.append(nbf.v4.new_code_cell(text.strip()))

md(r"""
# 27 — Competition Gap Analysis

**Scope:** research and strategy only. This notebook does not train a model, generate predictions, alter a correction, or create a submission. It treats Notebooks 01–26 and their frozen artifacts as evidence, not as a code base to optimize.

## Executive answer

The current system is approximately 125th **according to the user-provided snapshot; this rank was not independently frozen in the repository** because it extracts most of its value from persistence-like TWS state, generic climate proxies, tabular boosting and a small spatial correction. That is a strong baseline, but not a full water-balance forecasting system. The most credible top-10 gap is **prediction-time-legal, dynamic forcing/state data**—especially precipitation, evapotranspiration, runoff, snow and deeper soil-water memory—joined with strict issue-time provenance. A larger transformer on the same information is unlikely to close the gap.

The one next experiment recommended here is an **ERA5-Land operational water-budget residual model**: add only cutoff-safe monthly precipitation, evaporation, runoff, snow-water and four-layer soil-water changes to the frozen Notebook 11 prediction, then accept it only under predeclared temporal/spatial gates. This is a hypothesis, not a guaranteed leaderboard gain.
""")

code(r"""
from pathlib import Path
import json, platform
import pandas as pd

ROOT = Path.cwd().resolve()
if ROOT.name == 'notebooks': ROOT = ROOT.parent
ART = ROOT / 'artifacts' / 'notebook27'
ART.mkdir(parents=True, exist_ok=True)
print({'root': str(ROOT), 'artifact_dir': str(ART), 'python': platform.python_version()})
""")

md(r"""
## 1. Evidence boundary and competition specification

Official Zindi material says the target is one-month-ahead GRACE TWS, where time is relative to the last available GRACE observation. TWS includes groundwater, soil moisture, surface water and snow. Ranking uses RMSE; public/private test shares are approximately 30%/70%. External data must be free, operationally available within one month, available at prediction time, and must not directly or indirectly contain future TWS. Top-10 review also scores trustworthiness, innovation and practicality.

No public winning solution exists: the competition is still open. Discussion/FAQ/organizer answers were not accessible as a stable public corpus during this audit. Therefore every “top solution” statement below is explicitly a hypothesis. The leaderboard rank is volatile and the stated ~125 position is treated as user-supplied context.

Primary competition sources:

- [Official information and rules](https://zindi.world/competitions/one-step-ahead-of-drought-forecasting-global-water-storage-challenge)
- [Official data description](https://zindi.africa/competitions/one-step-ahead-of-drought-forecasting-global-water-storage-challenge/data)
- [Official leaderboard](https://zindi.africa/competitions/one-step-ahead-of-drought-forecasting-global-water-storage-challenge/leaderboard)

**Critical interpretation:** `SOIL_MOISTURE_tp1` is explicitly supplied and therefore legal, but any external monthly product must be reconstructed as it would have existed at each historical forecast issue date. A modern reanalysis download is not automatically leakage-free merely because its timestamp precedes the target.
""")

code(r"""
repo_evidence = pd.DataFrame([
['01–07','EDA and standard tabular models/features','LightGBM strongest early family; generic engineering not decisive'],
['08','Temporal validation','Validation design materially changes conclusions'],
['09–11','Advanced selection and canonical blend','Notebook 11 public RMSE 0.824463818; canonical reference'],
['12–13','Optimization/generalization','Large local gains did not reliably transfer to public'],
['14–15','Alternative families and delta target','0.825971129 / 0.824797753; neither beat canonical'],
['16','Competition-level feature search','Broad/clustered features transferred poorly'],
['17','Reproducibility audit','Pipeline identity error demonstrated comparison risk'],
['18','Hydrology-inspired proxies','Public 0.831108253; labels were not new physical observations'],
['19','Challenge specification','Prediction-time availability formalized'],
['20','Strict validation/leakage','global-month prior has 91.30% future-order exposure'],
['21','Spatial correction','Public 0.823609466; small +0.05 southern masked correction'],
['22','Research synthesis','Do-not-repeat register and stopping rules'],
['23','Correction transfer','3/4 windows; transfer score 54.53/100, weak'],
['24','Trustworthiness/model card','66.30/100; southern/temporal risk remains'],
['25','OOF tail/support','proxy OOF RMSE .743809; P95 1.520082; south P95 1.773752'],
['26','Final consolidation','Current-best frozen; optimization closed under existing evidence'],
], columns=['Evidence','Scope','Finding'])
repo_evidence
""")

md(r"""
## 2. Professional critique

### What is good

The repository is unusually strong on auditability: it froze canonical hashes, exposed a reference-pipeline mismatch, evolved beyond random validation, separated known/masked regimes, documented failed experiments, quantified tail risk, and did not pretend that a small public gain proved robustness.

### Why it is not a plausible top-10 system yet

1. **Information ceiling:** the system mostly recombines supplied TWS/SPEI/near-surface soil moisture. Feature engineering cannot manufacture missing groundwater, snow, runoff, evapotranspiration, reservoir or human-withdrawal states.
2. **Physical incompleteness:** Notebook 18 created hydrology-inspired transformations, not independent measurements of the omitted stores/fluxes. Its 0.831108253 public score is evidence against “domain naming” without new signal.
3. **Spatial dynamics are weakly represented:** latitude/longitude, clusters and a bounded regional correction do not model upstream connectivity, aquifers, basins or teleconnections.
4. **Sequence memory is shallow:** tree features summarize history but do not learn variable-delay accumulation and recession directly. Still, sequence learning is secondary until useful forcings exist.
5. **Validation remains a proxy:** OOF covers 34.774%, shares the same spatial grid, and does not emulate every hidden test mechanism. Public feedback has repeatedly disagreed with local selection.
6. **Correction is leaderboard-positive but scientifically fragile:** the Notebook 21 gain is small; Notebook 23 rates transfer only 54.53/100. It is a hedge, not the missing mechanism.
7. **Tail and regional bias matter under RMSE:** south P95 error is 1.773752 and max OOF error is 4.461003. A few poorly represented hydrologic regimes can dominate score.
8. **Submission strategy has low statistical power:** public is ~30%; repeated selection among variants can overfit the public leaderboard. Two private selections should emphasize mechanism diversity, not tiny public deltas.

The blunt conclusion is that further tuning of the current LightGBM feature space is exploitation of a saturated signal set. The repository has abundant model research and insufficient new, operational hydrologic information.
""")

code(r"""
gaps = pd.DataFrame([
['Dynamic precipitation/ET/runoff budget','Existing SPEI and soil moisture are incomplete proxies; NB18 added no observations','High','Medium'],
['Deep soil and groundwater memory','TWS explicitly includes groundwater; current covariates do not directly observe it','High','High'],
['Snow water equivalent','TWS explicitly includes snow; southern/northern seasonal tails remain','Medium–High','Medium'],
['Hydrologic connectivity','Coordinate/cluster features do not encode basin flow or aquifers','Medium','High'],
['Human regulation and withdrawal','Reservoirs/irrigation/pumping can dominate local storage','Medium','High'],
['True issue-time data vintages','Historical reanalysis can use information unavailable operationally','High (validity)','High'],
['Independent spatial holdout evidence','NB25 same-grid support cannot prove unseen-space robustness','High (selection)','Medium'],
['Mechanism-diverse ensemble','Existing blends are primarily related tabular learners','Medium','Medium'],
], columns=['Gap','Evidence','Expected impact','Difficulty'])
gaps
""")

md("## 3. External operational data audit")
code(r"""
external = pd.DataFrame([
['ERA5-Land','Copernicus CDS','precipitation, evaporation, runoff, SWE, 4-layer soil water','Yes; daily updates','Conditional—use only data/vintage available by issue cutoff','Reanalysis revisions and monthly aggregation can leak availability','CC-BY','Large (hourly 0.1°); subset/monthly aggregate','Medium','High','https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land'],
['ERA5','Copernicus CDS','atmospheric forcing, radiation, temperature, wind','Yes; near-real-time/final streams differ','Conditional on product stream and latency','Final reanalysis at historical timestamp may not equal operational vintage','Copernicus licence','Very large','High','Medium–High','https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels'],
['GDO/EDO indicators','Copernicus JRC','drought indicators, soil moisture anomaly, vegetation stress','Yes','Likely, if release date <= cutoff','Composite inputs and revision provenance','Copernicus terms','Medium','Medium','Medium–High','https://drought.emergency.copernicus.eu/'],
['CHIRPS preliminary','UCSB Climate Hazards Center','satellite-gauge precipitation','Yes','Likely within month; verify version calendar','Gauge assimilation/final replacement and land-only coverage','Free with attribution','Medium','Low–Medium','High','https://data.chc.ucsb.edu/products/CHIRPS/v3.0/prelim/'],
['GPM IMERG Early/Late','NASA GES DISC/LANCE','near-real-time precipitation','Yes','Yes for Early/Late products','Do not backfill Final product into historical issue time','NASA open data','Large','Medium','High','https://gpm.nasa.gov/data/imerg'],
['SMAP NRT','NASA NSIDC/LANCE','surface soil moisture / freeze-thaw','Yes; NRT can be hours','Yes','Short NRT retention; mission starts 2015; standard/NRT mismatch','NASA open data','Large','Medium–High','Medium','https://www.earthdata.nasa.gov/data/instruments/smap'],
['GLDAS 2.1','NASA GES DISC','soil layers, runoff, ET, snow, groundwater proxy','Yes; stated ~1.5 month latency for standard product','Often fails one-month rule unless a suitable NRT stream is verified','Assimilation latency; model dependence; overlaps target physics','NASA open data','Large','Medium','Medium','https://ldas.gsfc.nasa.gov/gldas'],
['MERRA-2','NASA GMAO/GES DISC','land/atmospheric reanalysis, precipitation, ET','Yes','Conditional; verify latency/vintage','Retrospective assimilation and revisions','NASA open data','Very large','High','Medium','https://gmao.gsfc.nasa.gov/reanalysis/MERRA-2/'],
['GLEAM','GLEAM consortium','evapotranspiration, root-zone moisture','Public research product','Unclear for strict one-month operational rule','Version/release delay; must audit timestamps','Research terms; verify version','Medium','Medium','Medium','https://www.gleam.eu/'],
['MODIS MOD13 NRT/VIIRS vegetation','NASA LP DAAC/LANCE','NDVI/EVI/vegetation state','Yes','NRT product likely; audit quality/reprocessing','Cloud compositing and final-vs-NRT revision','NASA open data','Large','Medium–High','Medium','https://www.earthdata.nasa.gov/data/instruments/modis'],
['MODIS snow / VIIRS snow','NASA NSIDC/LANCE','snow cover/fraction','Yes','NRT likely','Cloud gaps; snow cover is not SWE','NASA open data','Large','Medium','Medium–High','https://www.earthdata.nasa.gov/topics/cryosphere/snow'],
['NOAA climate indices','NOAA PSL/CPC','ENSO, PDO, NAO, teleconnections','Yes','Yes if lagged to release date','Revised index values; weak local specificity','Public domain','Tiny','Low','Medium','https://psl.noaa.gov/data/climateindices/list/'],
['NOAA OISST','NOAA NCEI','sea-surface temperature/teleconnections','Yes; daily','Conditional on preliminary/final vintage','Reanalysis/final revisions','Public domain','Large','Medium','Medium','https://www.ncei.noaa.gov/products/optimum-interpolation-sst'],
['HydroSHEDS / HydroBASINS','WWF/HydroSHEDS','static drainage direction, basins, river topology','Yes, static','Yes','Low; topology resolution/misalignment only','Free for non-commercial/scientific use; verify competition use','Medium','Medium','High','https://www.hydrosheds.org/'],
['MERIT Hydro','University of Tokyo','static flow direction/upstream area','Yes, static','Yes','Low; licence/attribution and grid mapping','CC-BY-NC 4.0—competition use must be checked','Large','Medium','High','http://hydro.iis.u-tokyo.ac.jp/~yamadai/MERIT_Hydro/'],
['SRTM/Copernicus DEM','NASA/Copernicus','elevation, slope, orographic context','Yes, static','Yes','Low','Open with attribution/terms','Large once, tiny aggregated','Low','Medium','https://dataspace.copernicus.eu/explore-data/data-collections/copernicus-contributing-missions/collections-description/COP-DEM'],
['ESA WorldCover','ESA','land cover, water/vegetation fractions','Yes, static/annual','Yes if vintage precedes cutoff','Land-cover year may postdate early training rows','CC-BY 4.0','Medium','Low–Medium','Medium','https://worldcover2021.esa.int/'],
['GRanD reservoirs','Global Water System Project','reservoir locations/capacity','Public static','Yes for inventory vintage','Inventory incompleteness; licence verification','Research licence; verify redistribution','Small','Low','Medium','https://www.globaldamwatch.org/grand/'],
['GRACE/JPL mascon auxiliaries','NASA JPL/PO.DAAC','alternative TWS/gridded mascons','Public','No for t+1; only lagged observations legal','Extremely high: direct/derived target contamination','NASA open data','Medium','Low','Low unless exact target-history reconciliation','https://grace.jpl.nasa.gov/data/get-data/jpl_global_mascons/'],
['Population/GHSL','European Commission JRC','static human-pressure proxy','Public','Static vintage can be legal','Temporal interpolation/postdated layers','Copernicus/JRC open terms','Medium','Low','Low–Medium','https://human-settlement.emergency.copernicus.eu/'],
['FAO AQUASTAT/irrigation','FAO','water withdrawal/irrigation context','Public','Static/lagged only','Sparse, revised, coarse, often non-operational','FAO terms; attribute','Small','High preprocessing','Medium'],
], columns=['Source','Provider','Signal','Accessible','Prediction_time_fit','Leakage_risk','License','Size_download','Difficulty','Contribution','URL'])
external.to_csv(ART/'external_data_sources.csv', index=False)
external
""")

md(r"""
## 4. Model and literature assessment

Recent forecasting architectures are relevant, but applicability is not equivalence to expected gain. PatchTST and TCN are plausible compact sequence baselines; TFT offers covariate gating and interpretability; graph models can encode basin adjacency; Fourier operators can learn global fields but demand careful masking and much more compute. Informer/Autoformer/FEDformer address long horizons, while this is one-step forecasting with a powerful observed state—so their headline advantages may not transfer. GraphCast/FourCastNet are weather-field systems trained at enormous scale and are inspirations, not sensible from-scratch baselines here. Physics-informed or hybrid residual learning is more credible because it can inject the missing water-budget structure without replacing the strong canonical predictor.

Primary method references: [TFT](https://doi.org/10.1016/j.ijforecast.2021.03.012), [Informer](https://ojs.aaai.org/index.php/AAAI/article/view/17325), [Autoformer](https://proceedings.neurips.cc/paper/2021/hash/bcc0d400288793e8bdcd7c19a8ac0c2b-Abstract.html), [FEDformer](https://proceedings.mlr.press/v162/zhou22g.html), [PatchTST](https://openreview.net/forum?id=Jbdc0vTOcol), [N-HiTS](https://ojs.aaai.org/index.php/AAAI/article/view/20590), [TiDE](https://openreview.net/forum?id=pCbC3aQB5W), [DeepAR](https://doi.org/10.1016/j.ijforecast.2019.07.001), [TCN evaluation](https://arxiv.org/abs/1803.01271), [DCRNN](https://openreview.net/forum?id=SJiHXGWAZ), [Graph WaveNet](https://www.ijcai.org/proceedings/2019/0264.pdf), [GraphCast](https://www.science.org/doi/10.1126/science.adi2336), [FourCastNet](https://arxiv.org/abs/2202.11214).
""")

code(r"""
models = pd.DataFrame([
['TFT','Conditional','covariate gating; static/dynamic fusion','heavy; overfit; global panel engineering','Small–medium','High'],
['Informer/Autoformer/FEDformer','Low–conditional','long-memory decomposition/attention','one-step horizon weakens premise; heavy','Small','High'],
['PatchTST','Conditional','strong channel-wise patch sequence baseline','panel/grid handling; needs long clean histories','Medium','Medium'],
['N-HiTS/TiDE','Conditional','efficient long-history forecasting with covariates','spatial interaction not native','Small–medium','Medium'],
['DeepAR/LSTM/GRU','Conditional','probabilistic/recurrent dynamics','slow global training; may lose to persistence','Small','Medium'],
['TCN','Yes as baseline','causal receptive field; efficient','spatial structure absent unless extended','Small–medium','Medium'],
['ST-GNN / Graph WaveNet','Conditional–strong','explicit basin/neighbor propagation','graph definition and leakage-safe temporal splits','Medium–high','High'],
['GraphCast-like','No from scratch','multiscale global message passing','data/compute mismatch; inspiration only','Unknown','Very high'],
['Fourier neural operator/FourCastNet-like','Low–conditional','global spectral coupling','regular-grid masks; compute; interpretability','Medium','Very high'],
['Physics-informed residual hybrid','Yes','targets water-budget correction around canonical','forcings/vintages must be valid; constraints imperfect','Medium–high','Medium'],
['Foundation model embeddings','Low now','possible transferable EO representation','availability, licence, alignment, opacity','Unknown','High'],
], columns=['Family','Applicable','Advantage','Disadvantage','Estimated_gain','Complexity'])
models
""")

md("## 5. Similar competitions and what transfers")
code(r"""
similar = pd.DataFrame([
['DrivenData Water Supply Forecast Rodeo','Seasonal water supply','Winning/community solutions emphasize hindcast validation, ensembles and basin-specific snow/climate signals','Transfer validation discipline and snow/teleconnection inputs; target/horizon differ','https://www.drivendata.org/competitions/254/reclamation-water-supply-forecast/'],
['Kaggle Flood Prediction / regression contests','Flood probability/tabular','Strong validation, diverse ensembles and target-aware transforms recur','Generic contest tactics transfer; hydrologic physics and metric differ','https://www.kaggle.com/competitions/playground-series-s4e5'],
['ECMWF WeatherBench / WeatherBench 2','Global weather fields','Graph/spectral models exploit grids, multivariate state and rollout evaluation','Spatial inductive bias transfers conceptually; scale is radically different','https://weatherbench2.readthedocs.io/'],
['AI4EO / ESA challenges','Earth observation','Pretrained EO representations and geospatial validation are common','Useful for static/vegetation context; no verified direct winner analogue','https://ai4eo.eu/'],
['CAMELS/Caravan hydrology benchmarks','Rainfall-runoff across basins','Entity-aware global models benefit from basin attributes and hydrologic forcings','Supports basin/entity modeling; TWS grid target differs','https://github.com/kratzert/Caravan'],
], columns=['Benchmark','Task','Observed lesson','Transfer limit','URL'])
similar
""")

code(r"""
comparison = pd.DataFrame([
['Target anchor','Strong TWS_t/persistence and dual-regime logic','Likely retains persistence anchor','No meaningful gap','High'],
['Dynamic forcings','Mostly supplied SPEI and near-surface soil moisture','Likely adds operational precipitation/ET/runoff/snow/deep-soil state','Major information gap','High'],
['Groundwater representation','Indirect through lagged TWS and proxies','May add deep-soil/land-model/aquifer proxies; direct current groundwater remains difficult','Major physical gap','Medium'],
['Spatial structure','Coordinates/clusters plus bounded hemispheric correction','May use basin, upstream, multiresolution or graph context','Moderate gap','Medium'],
['Temporal representation','Engineered lags/rolling statistics and trees','May use causal sequences, but a transformer is not required','Moderate gap','Low–Medium'],
['Validation','Temporal/irregular/spatial gates evolved; same-grid OOF limitation','Likely frozen issue-time, basin and regime stress tests','Selection gap','Medium–High'],
['Availability provenance','Audited conceptually; no external-data vintage pipeline','Must track release stream/version per feature','Major validity gap','High'],
['Ensemble diversity','Mostly related tabular models and conservative blends','May combine physically distinct mechanisms','Moderate gap','Medium'],
['Tail/regional behavior','South P95 1.773752; correction transfer weak','Likely explicitly controls high-error regimes under RMSE','Material gap','Medium'],
['Reproducibility/trust','Strong hashes, audits, model card; trust 66.30/100','Top 10 must reproduce and pass trust/practicality review','Current strength, still incomplete','High'],
], columns=['Dimension','Current_solution','Plausible_top10_solution','Gap_assessment','Confidence'])
comparison.to_csv(ART/'top_solution_comparison.csv', index=False)
comparison
""")

md("## 6. Top-solution hypotheses — not facts")
code(r"""
hyp = pd.DataFrame([
['Uses additional operational precipitation/land-state data','High','Official rules explicitly invite Copernicus covariates; current signal set has an information ceiling'],
['Uses release-date/vintage-safe data engineering','High','Top-10 code review and one-month availability make provenance essential'],
['Uses deeper soil/snow/runoff/ET variables','Medium–High','They map directly to omitted TWS stores/fluxes'],
['Uses strong lagged TWS persistence as anchor','High','One-step target and repository evidence make current state extremely informative'],
['Trains known/masked experts or missingness-aware model','High','Test contains materially different TWS_t availability regimes'],
['Uses basin/neighbor or multi-resolution spatial context','Medium','Global robustness favors spatial structure, but no winner evidence is public'],
['Uses conservative diverse ensemble','Medium','Often robust under RMSE; public/private split punishes variant chasing'],
['Uses a transformer','Low','No public solution evidence; one-step horizon and tabular strength reduce necessity'],
['Uses GraphCast/FourCastNet directly','Low','Compute/data mismatch and no public evidence'],
['Uses external GRACE t+1 or retrospectively revised proxy','Low and impermissible','Would violate availability/target-leakage requirements; not a legitimate hypothesis'],
['Optimizes public leaderboard with many corrections','Low','Top-10 private robustness and code review make this fragile'],
], columns=['Hypothesis','Confidence','Rationale'])
hyp.to_csv(ART/'top_solution_hypotheses.csv', index=False)
hyp
""")

md("## 7. Research idea pool and scoring")
code(r"""
ideas_raw = [
('ERA5-Land water-budget residual','Cutoff-safe precipitation, evaporation, runoff, SWE and layer-water deltas correct frozen baseline','0.005–0.025','Medium','Medium','ERA5-Land','5–8 days',5,3,3,4,5,5),
('IMERG precipitation forcing','Add lagged Early/Late monthly precipitation anomalies','0.002–0.015','Low–Medium','Medium','GPM IMERG','3–5 days',4,2,3,3,5,4),
('CHIRPS preliminary precipitation','Land precipitation accumulation/anomaly with release-safe archive','0.002–0.012','Low–Medium','Low–Medium','CHIRPS prelim','2–4 days',4,2,2,3,5,4),
('SMAP NRT root/surface state','Add issue-time soil moisture/freeze-thaw state','0.001–0.010','Medium','High','SMAP NRT','5–8 days',3,3,4,4,3,3),
('Snow water equivalent expert','Cold-region expert from ERA5-Land SWE/melt changes','0.001–0.012','Medium','Medium','ERA5-Land','4–6 days',4,3,3,4,4,4),
('Deep-soil layer memory','Four ERA5-Land soil layers and recession deltas','0.002–0.015','Medium','Medium','ERA5-Land','4–6 days',5,3,3,4,4,4),
('Runoff–ET balance feature','Lagged P-ET-runoff balance accumulated at 1/3/6 months','0.003–0.018','Medium','Medium','ERA5-Land','4–6 days',5,3,3,4,4,5),
('Vintage reconstruction audit','Record product/version/release date for every external feature','Validity enabler','Low','High','provider metadata','3–7 days',5,1,4,3,4,5),
('HydroBASINS aggregation','Aggregate forcings and residual context by drainage basin','0.002–0.015','Low','High','HydroBASINS + forcing','5–10 days',4,2,4,5,4,4),
('River-graph residual GNN','Causal upstream/downstream message passing on residuals','0.003–0.020','Medium','Very high','HydroSHEDS + forcings','2–3 weeks',4,3,5,5,2,3),
('Aquifer-region hierarchy','Random effects/expert groups based on aquifer maps','0.001–0.012','Medium','High','WHYMAP/aquifer data','1–2 weeks',3,3,4,5,2,3),
('Reservoir capacity context','Static dam count/capacity and regulated-basin interaction','0.000–0.008','Medium','Medium','GRanD','3–5 days',2,3,3,4,3,2),
('Irrigation/human-use proxy','Static irrigated fraction and withdrawal intensity','0.000–0.010','Medium–High','High','FAO/GHSL','1–2 weeks',3,4,4,5,2,2),
('ENSO/PDO/NAO regime gates','Release-safe lagged teleconnection indices gate experts','0.000–0.006','Low','Low','NOAA indices','1–2 days',2,1,2,3,5,2),
('OISST teleconnection embedding','Lagged basin-scale SST EOFs','0.001–0.010','Medium','High','NOAA OISST','1–2 weeks',3,3,4,5,2,3),
('Vegetation stress state','NRT NDVI/EVI anomaly and change','0.001–0.008','Medium','High','MODIS/VIIRS NRT','1–2 weeks',3,3,4,4,2,3),
('Static elevation/slope','DEM-derived terrain controls with interactions','0.000–0.005','Low','Low','Copernicus DEM','2–3 days',2,1,2,2,5,2),
('Land-cover fractions','Static water/wetland/forest/cropland fractions','0.000–0.006','Low–Medium','Low','ESA WorldCover','2–3 days',2,2,2,3,4,2),
('Masked-regime climatology audit','Diagnose whether masked rows represent specific months/regions','Selection enabler','Low','Low','provided data','1–2 days',4,1,1,2,5,3),
('Known/masked external-data experts','Separate residual experts with shared forcings','0.002–0.012','Low–Medium','Medium','external forcings','3–5 days',4,2,3,3,4,4),
('PatchTST residual sequence','Predict baseline residual from per-cell lag sequences','0.001–0.012','Low','High','provided + forcing','1–2 weeks',3,2,4,4,3,3),
('TCN residual sequence','Causal dilated convolution over histories','0.001–0.012','Low','Medium','provided + forcing','5–8 days',3,2,3,3,4,3),
('TFT residual model','Static/dynamic covariate-gated residual forecasting','0.001–0.015','Medium','Very high','provided + forcing','2–3 weeks',3,3,5,5,2,3),
('N-HiTS global panel','Entity-aware lag/covariate global forecast','0.000–0.010','Low–Medium','High','provided + forcing','1–2 weeks',3,2,4,4,3,3),
('TiDE residual model','Dense encoder-decoder with dynamic covariates','0.001–0.012','Low–Medium','High','provided + forcing','1–2 weeks',3,2,4,4,3,3),
('DeepAR probabilistic panel','Distributional forecast with cell embeddings','0.000–0.009','Low','High','provided + forcing','1–2 weeks',2,2,4,3,3,2),
('Graph WaveNet basin model','Adaptive adjacency plus hydrologic graph','0.002–0.018','Medium','Very high','topology + forcing','2–4 weeks',4,3,5,5,2,3),
('Fourier global residual operator','Learn low-frequency spatial residual evolution','0.001–0.015','Medium','Very high','regular grid + forcing','3–5 weeks',3,3,5,5,1,2),
('Multiresolution spatial pyramid','Cell/3°/basin/global lagged forcing summaries','0.002–0.014','Low–Medium','Medium','forcing + grid','5–8 days',4,2,3,4,4,4),
('Causal neighbor lags','Only prior-date neighbors; no same-target smoothing','0.001–0.010','Medium','Medium','provided data','4–7 days',3,3,3,3,4,3),
('Water-budget sign constraints','Constrain residual response signs/ranges where defensible','0.000–0.007','Low','Medium','forcing','3–5 days',3,1,3,4,4,3),
('Heteroscedastic uncertainty weighting','Downweight correction in unstable regimes via OOF variance','0.000–0.006','Low–Medium','Medium','OOF','3–4 days',3,2,3,3,4,3),
('Tail-aware training diagnostics','Report extreme-regime errors while keeping RMSE objective','Robustness enabler','Low','Low','OOF','1–2 days',4,1,1,2,5,3),
('Basin-held-out validation','Leave basins out in addition to time blocks','Selection enabler','Low','Medium','basin map','3–5 days',5,1,3,3,4,4),
('Vintage-held-out validation','Simulate data latency and missing product releases','Selection enabler','Low','Medium','metadata','3–5 days',5,1,3,4,4,4),
('Climate-regime stress folds','Hold out El Niño/La Niña and drought extremes','Selection enabler','Low','Medium','NOAA + provided','3–5 days',4,1,3,4,4,3),
('OOF diversity constrained blend','Blend only mechanisms with low residual correlation and stable folds','0.001–0.008','Low–Medium','Medium','OOF predictions','3–5 days',3,2,3,3,4,3),
('Seed/fold stability gate','Reject gains smaller than seed/fold variability','Prevents false gain','Low','Low','existing pipeline','1–2 days',5,1,1,2,5,4),
('Public-LB Bayesian restraint','Treat public score as noisy 30% observation; cap iteration','Prevents overfit','Low','Low','submission history','1 day',4,1,1,3,5,3),
('Two-private-selection diversity','Choose canonical and mechanism-diverse external-data model','Private risk reduction','Low','Low','validated models','1 day',4,1,1,3,5,3),
('GRACE target reconciliation audit','Ensure no external mascon version indirectly reconstructs target','Validity enabler','High if skipped','Medium','GRACE metadata','2–4 days',5,5,3,4,4,4),
('Missingness indicator audit','Test whether availability patterns encode hidden time/region','0.000–0.005','Medium','Low','provided data','1–2 days',3,3,1,2,5,2),
('Residual seasonal harmonics','Cross-fitted regional seasonal residual basis','0.000–0.006','Medium','Low','OOF','2–3 days',2,3,2,2,4,2),
('Conformal risk bands','Calibrate OOF uncertainty by region/time for trustworthiness','No direct gain','Low','Medium','OOF','3–5 days',2,1,3,4,4,2),
('Foundation EO embeddings','Frozen open model embeddings for land state','Unknown','Medium–High','Very high','open EO foundation model','3–5 weeks',2,4,5,5,1,1),
]
cols=['Name','Technical_description','Estimated_public_gain','Leakage_risk','Code_complexity','Data_need','Development_time','Impact','Risk','Difficulty','Novelty','Feasibility','Expected_LB_contribution']
ideas=pd.DataFrame(ideas_raw,columns=cols)
# Higher is better. Risk and difficulty are penalties; all inputs are transparent ordinal judgments, not measured effects.
ideas['Total_score']=(0.27*ideas.Impact + 0.23*ideas.Expected_LB_contribution + 0.18*ideas.Feasibility + 0.12*ideas.Novelty + 0.10*(6-ideas.Risk) + 0.10*(6-ideas.Difficulty)).round(3)
ideas['Priority']=pd.qcut(ideas['Total_score'].rank(method='first'),4,labels=['Low','Medium','High','Very High'])
ideas.to_csv(ART/'research_ideas.csv',index=False)
ranking=ideas.sort_values(['Total_score','Impact'],ascending=False).reset_index(drop=True)
ranking.insert(0,'Rank',ranking.index+1)
ranking.to_csv(ART/'research_ranking.csv',index=False)
ranking.head(15)
""")

md(r"""
## 8. Decision rules and roadmap

The estimated public gains are **ranges for prioritization, not forecasts**. Scores combine ordinal judgments: 27% impact, 23% expected leaderboard contribution, 18% feasibility, 12% novelty, and 10% each for low risk and low difficulty. They must never be presented as empirical results.

### Notebook 28 — one idea only

Implement **ERA5-Land operational water-budget residual learning**. Freeze Notebook 11 predictions. Build a feature manifest with source product, variable, valid time, publication/retrieval time, version and cutoff status. Aggregate only data provably available by each issue cutoff. Train one conservative residual learner using P, ET, runoff, SWE, snowmelt proxy and four soil layers; no new correction family, architecture search or ensemble search.

Acceptance gates, fixed before training:

- exact reproduction of Notebook 11 reference;
- zero availability violations in the vintage audit;
- improvement in at least 3/4 complete-date temporal windows;
- non-worse masked RMSE in at least 3/4 windows;
- improvement or non-inferiority in basin-held-out and south-region stress tests;
- mean improvement greater than fold/seed variability and at least 0.002 RMSE;
- no P95 absolute-error deterioration above 1%;
- prediction drift bounded and explained physically.

### Notebook 29

Only if Notebook 28 passes: add **basin topology and multiresolution forcing aggregation** to the same frozen residual target. Do not introduce a neural graph model until a simple basin aggregation establishes incremental value.

### Notebook 30

Only if Notebook 29 passes: compare a compact **causal TCN or PatchTST residual learner** against the tree residual learner, then create an OOF-diversity constrained blend only if it is independently stable.
""")

code(r"""
gap_md = '''# Competition gap analysis

## Direct answer
The current solution is strong but information-limited. Its approximate rank of 125 is user-provided and was not independently archived. Notebook 11 scores 0.824463818 public; Notebook 21 improves this only to 0.823609466 via a small southern masked correction whose transfer score is 54.53/100. The gap is unlikely to be closed by more tuning of the same supplied variables.

## Evidence-backed diagnosis
- Missing independent observations of precipitation, evapotranspiration, runoff, snow, deep soil water, groundwater and regulation.
- Hydrology-labelled engineered proxies did not work: Notebook 18 public RMSE 0.831108253.
- Local/public mismatch recurred in Notebooks 12, 14, 16 and 18.
- OOF is canonical-equivalent rather than exact canonical, covers 34.774%, and uses the same spatial grid.
- Tail risk is material: OOF P95 1.520082, south P95 1.773752, maximum 4.461003.
- The global-month prior has 91.30% future-order exposure, demonstrating why ordering provenance matters.

## Likely missing top-10 ingredients
High-confidence hypotheses are cutoff-safe operational covariates, persistence anchoring, and known/masked handling. Medium-confidence hypotheses are deeper soil/snow/runoff/ET state, basin context and mechanism-diverse ensembling. Transformer or GraphCast use is low-confidence; no winner solution is public.

## Harsh conclusion
The repository has over-invested in transformations and model variants relative to new causal information. The spatial correction is a small leaderboard hedge, not a hydrologic breakthrough. A bigger model on the same features is a poor next bet.
'''
strategy_md = '''# Competition strategy

1. Stop optimizing the existing feature space.
2. Freeze Notebook 11 as the scientific reference and Notebook 21 as a score artifact.
3. Spend the next experiment budget on one operationally legal data family: ERA5-Land water-budget variables.
4. Treat publication time/version as first-class columns; reject any row without proof of availability.
5. Select on complete-date temporal, basin-held-out, masked, southern and tail gates—not one aggregate RMSE.
6. Submit only after a predeclared robust gain; preserve one private slot for canonical diversity.

The challenge is still open and no winner method is public. Top-solution comparisons are hypotheses, not reverse-engineered facts.
'''
plan_md = '''# Notebook 28 plan — one main idea

## Hypothesis
Prediction-time-legal ERA5-Land water-budget states add missing physical information beyond SPEI and near-surface soil moisture, allowing a conservative residual model to improve Notebook 11 across time and space.

## Scope
Only ERA5-Land precipitation, total evaporation, runoff, snow-water equivalent/snowmelt proxy, and four soil-water layers. One controlled residual learner. No architecture search, correction optimization, or additional dataset.

## Protocol
1. Reproduce frozen Notebook 11 predictions and hashes.
2. Construct an issue-time vintage manifest and fail closed on unavailable data.
3. Aggregate to monthly/grid features in float32 with cached partitions.
4. Fit residual model using training-only fold histories.
5. Evaluate four complete-date temporal folds plus basin-held-out, known/masked, hemispheric and tail slices.
6. Compare against Notebook 11 under frozen gates.

## Acceptance
Pass only with zero vintage violations, >=3/4 temporal improvements, >=3/4 masked non-worse, basin/south non-inferiority, mean gain >=0.002 and above seed/fold noise, and <=1% P95 deterioration. Otherwise record failure and keep canonical.
'''
recommend_md = '''# Final recommendation

If this were my project, today I would implement one idea: **a cutoff-safe ERA5-Land water-budget residual model around the frozen Notebook 11 prediction**.

Why this one: TWS is the sum of groundwater, soil moisture, surface water and snow, but the current system mainly sees lagged TWS, SPEI and near-surface soil moisture. Notebooks 12–18 show that more transformations, hyperparameters and model families do not reliably transfer. ERA5-Land supplies independent, physically aligned precipitation, evaporation, runoff, snow and deep-soil states, is global, public and updated daily. That makes it a better information bet than a transformer on unchanged inputs.

The hard part is not downloading ERA5-Land; it is proving historical issue-time legality. Final reanalysis values can incorporate information or revisions unavailable at the forecast date. Notebook 28 must therefore store valid time, release/retrieval time, stream and version for every feature, fail closed on uncertain availability, and document the exact operational analogue.

I would freeze Notebook 11, predict only its OOF residual with one conservative LightGBM configuration, and pre-register temporal, basin, masked, southern and tail gates. I would require improvement in at least three of four temporal windows, masked non-inferiority in three of four, no meaningful basin/south degradation, mean RMSE gain of at least 0.002 beyond seed/fold noise, and no more than 1% P95 deterioration. If any gate fails, I would not submit it.

This is not a claim that ERA5-Land will reach top 10. It is the highest-value falsifiable experiment because it attacks the repository's information ceiling while preserving its strongest assets: the canonical predictor, strict chronology, reproducibility and conservative selection.
'''
for fn,txt in [('competition_gap_analysis.md',gap_md),('competition_strategy.md',strategy_md),('notebook28_plan.md',plan_md),('final_recommendation.md',recommend_md)]:
    (ART/fn).write_text(txt,encoding='utf-8')
print('Wrote:', sorted(p.name for p in ART.iterdir()))
""")

md(r"""
## 9. Final recommendation

**One idea:** a cutoff-safe ERA5-Land water-budget residual model around Notebook 11.

This choice follows the evidence rather than architecture fashion. The existing repository has already falsified many variants of the same information. Notebook 28 should test whether genuinely new, physically relevant and operationally available forcings break that ceiling. If the strict gates fail, retain Notebook 11/21 and conclude that ERA5-Land as implemented did not transfer—without adding another submission.

All generated outputs are written to `artifacts/notebook27/`. No pre-existing file is modified by notebook execution.
""")

OUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, OUT)
print(OUT)
