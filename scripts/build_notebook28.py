"""Build the fail-safe, single-hypothesis ERA5-Land integration notebook."""
from pathlib import Path
import nbformat as nbf

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'notebooks/28_era5_land_water_budget_integration.ipynb'
nb=nbf.v4.new_notebook(); nb.metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3'}}
def md(x): nb.cells.append(nbf.v4.new_markdown_cell(x.strip()))
def code(x): nb.cells.append(nbf.v4.new_code_cell(x.strip()))

md(r'''# 28 — ERA5-Land Water-Budget Integration

**Single hypothesis:** prediction-time-available, cutoff-safe ERA5-Land water-budget variables add independent hydrologic information to the frozen Notebook 11 feature space and improve temporal, irregular-date, spatial and space-time robustness.

This notebook changes no model family, parameter, expert segmentation, blend weight, target encoding or correction. Notebook 21 correction is out of scope until the primary ERA5 system passes frozen gates. It never fabricates external data, metrics or predictions.

## Research decision before download

The competition permits free, operational data available within one month of acquisition, but forbids future GRACE/TWS information. ERA5-Land-T is reported by ECMWF as roughly five days behind real time; its monthly means are available about five days after month-end and are later overwritten by consolidated ERA5-Land after about two months. The CDS monthly catalogue separately describes the consolidated monthly product as delayed 2–3 months. Therefore a present-day historical download is **not proof of historical operational availability**.

Two policies are frozen before validation:

- **A — strict operational (eligible for final selection):** prediction origin `t` uses source month `t-1`. This allows the complete prior calendar month plus the documented ~5-day ERA5-Land-T delay. Every row must carry release/vintage evidence; absent evidence fails closed.
- **B — less conservative research only:** `t` monthly data may be used only when forecast issue date is demonstrably at least five days after that month ends and the ERA5-Land-T vintage is archived. The supplied dates encode months, not intra-month issue days, so B is not eligible here without additional organizer clarification.

Final consolidated ERA5-Land is scientifically useful but cannot by itself reproduce ERA5-Land-T vintages. This is residual availability risk and is stated rather than hidden.

Official sources: [competition rules](https://zindi.world/competitions/one-step-ahead-of-drought-forecasting-global-water-storage-challenge), [ERA5-Land documentation](https://confluence.ecmwf.int/pages/viewpage.action?pageId=505384848), [monthly catalogue](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land-monthly-means), [CDS API setup](https://cds.climate.copernicus.eu/how-to-api).

Licence: the catalogue states CC-BY. Users must accept dataset terms before API download. A CDS account, personal access token and accepted terms are required. Tokens belong in the user profile configuration, never in this repository. Raw data are deliberately not committed.''')

code(r'''from pathlib import Path
import hashlib, importlib.util, json, platform, subprocess, sys, time
import numpy as np, pandas as pd
ROOT=Path.cwd().resolve()
if ROOT.name=='notebooks': ROOT=ROOT.parent
ART=ROOT/'artifacts/notebook28'; ART.mkdir(parents=True,exist_ok=True)
RAW=ROOT/'data/external/era5_land/raw'; PROCESSED=ROOT/'data/external/era5_land/processed'
cred=(Path.home()/'.cdsapirc').exists(); packages={x:bool(importlib.util.find_spec(x)) for x in ['cdsapi','xarray','netCDF4','pyarrow']}
raw_files=sorted(RAW.glob('era5_land_*.nc')) if RAW.exists() else []
STATUS='data-ready' if raw_files else 'blocked: external data unavailable'
print({'status':STATUS,'cds_credentials_detected':cred,'packages':packages,'raw_file_count':len(raw_files)})''')

md('''## Frozen design and acceptance gates

Feature sets are fixed as E0 canonical, E1 E0 + raw selected ERA5, E2 E1 + a small physical aggregate set, and E3 the smallest subset selected consistently across training folds. E3 selection is nested inside each fold and cannot look at test or validation targets.

Gates are fixed: A) strict temporal RMSE gain ≥0.002, irregular non-worse (tolerance 0.001), spatial deterioration ≤0.002; or B) masked gain ≥0.004, known deterioration ≤0.001 and worst fold non-worse (≤0.001); or C) space-time gain ≥0.003 and all other primary systems within 0.001. P95 deterioration must also be ≤1%, all cutoff flags true, and gain must exceed seed/fold noise. No gate may be changed after results.''')

code(r'''selection=pd.DataFrame([
['total_precipitation','m day-1 in moda','water input','positive storage forcing','t-1', 'Medium','~173 MB/month global upper bound','Included','core budget'],
['total_evaporation','m day-1 in moda; ECMWF sign convention may be negative','water loss','negative storage forcing after sign normalization','t-1','Medium','~173 MB/month','Included','core budget; verify sign'],
['surface_runoff','m day-1 in moda','surface drainage','storage loss/redistribution','t-1','Medium','~173 MB/month','Included','core budget'],
['sub_surface_runoff','m day-1 in moda','subsurface drainage','deep storage loss proxy','t-1','Medium','~173 MB/month','Included','core budget'],
['snowfall','m water equivalent day-1','frozen input','snow storage accumulation','t-1','Medium','~173 MB/month','Included','direct TWS component proxy'],
['snowmelt','m water equivalent day-1','frozen-to-liquid transfer','redistributes snow storage','t-1','Medium','~173 MB/month','Included','seasonal release'],
['volumetric_soil_water_layer_4','m3 m-3','100–289 cm soil water','deep memory proxy','t-1','Medium','~173 MB/month','Included','most independent soil layer'],
['volumetric_soil_water_layer_1','m3 m-3','0–7 cm soil water','surface state','t-1','Medium','~173 MB/month','Rejected initially','overlaps supplied near-surface soil moisture'],
['volumetric_soil_water_layer_2','m3 m-3','7–28 cm soil water','shallow root-zone state','t-1','Medium','~173 MB/month','Rejected initially','limit first download to 7 variables'],
['volumetric_soil_water_layer_3','m3 m-3','28–100 cm soil water','root-zone state','t-1','Medium','~173 MB/month','Rejected initially','add only if layer 4 proves useful'],
['snow_depth','m water equivalent','snow store','direct snow-storage proxy','t-1','Medium','~173 MB/month','Rejected initially','snowfall/melt chosen for limited pilot'],
['potential_evaporation','m day-1','atmospheric demand','drying pressure','t-1','Medium','~173 MB/month','Rejected initially','actual evaporation prioritized'],
['2m_temperature','K','air temperature','energy/snow phase proxy','t-1','Low–Medium','~173 MB/month','Rejected','not direct water budget'],
['skin_temperature','K','surface temperature','energy proxy','t-1','Low–Medium','~173 MB/month','Rejected','not direct water budget'],
['leaf_area_index_high_vegetation','m2 m-2','vegetation structure','ET modulation','t-1','Medium','~173 MB/month','Rejected','outside single core hypothesis'],
],columns=['ERA5_variable_name','Unit','Physical_meaning','Expected_relation_to_TWS','Chosen_lag','Leakage_risk','Download_size','Decision','Reason'])
selection.to_csv(ART/'era5_variable_selection.csv',index=False); selection''')

code(r'''# Reproducible dry-run manifest; no network call and no credential output.
r=subprocess.run([sys.executable,str(ROOT/'scripts/download_era5_land.py')],cwd=ROOT,text=True,capture_output=True)
print(r.stdout.strip()); assert r.returncode==0
manifest=pd.read_csv(ART/'era5_download_manifest.csv'); manifest.head()''')

code(r'''external=pd.DataFrame([
['ERA5-Land monthly means','Copernicus Climate Data Store','CC-BY; dataset terms must be accepted','monthly 0.1° NetCDF','ERA5-Land-T ~5 days; consolidated ~2–3 months','CDS account + personal token','data/external/era5_land/raw','Not committed; large and provider-governed','https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land-monthly-means'],
],columns=['Dataset','Provider','License','Format','Latency','Authentication','Local_path','Redistribution_policy','URL'])
external.to_csv(ART/'external_data_manifest.csv',index=False); external''')

md('''## Alignment contract

Coordinates are normalized to latitude ascending/descending agnostically and longitude `[-180,180)`. Competition coordinates remain in original row order. ERA5-Land is distributed on a regular 0.1° grid (native ≈9 km); competition cells are coarser and irregular over land. Nearest is the primary, reproducible pilot because storage/state fields and land masks can be discontinuous at coasts. Bilinear is a required secondary pilot and may be preferable for smooth fluxes, but it can mix land/ocean and will only be retained if physically valid, sufficiently covered, and not materially more costly. Monthly times are normalized to calendar periods; timezone is irrelevant after monthly aggregation.

Assertions: unchanged row count/order, unique IDs, one ERA5 match per ID, source month ≤ allowed cutoff, no backfill, no future month, and explicit missingness. Ocean/coastal missingness is never filled from future time.''')

code(r'''cutoff=pd.DataFrame([
['A_strict_operational','t-1','Eligible','Prior complete month; accommodates documented ~5-day ERA5-Land-T lag','pending row audit' if raw_files else STATUS],
['B_less_conservative','t conditional','Research only','Needs exact issue day and archived ERA5-Land-T vintage; neither is proven','ineligible without organizer/vintage evidence'],
],columns=['Scenario','Source_month','Final_selection','Rationale','Status'])
cutoff.to_csv(ART/'cutoff_safety_audit.csv',index=False)
alignment=pd.DataFrame([
['coordinate range','train/test lat -55.5..83.5; lon -179.5..179.5','pass from repository data'],
['longitude convention','normalize ERA5 to [-180,180)','pending external data'],['time convention','monthly Period[M], source=t-1','policy fixed'],
['nearest pilot','required','pending external data'],['bilinear pilot','required','pending external data'],
['row count/order/uniqueness','must be invariant','pending external data'],['future timestamp','must be zero','pending external data'],
['missing rate','report split/variable','pending external data'],
],columns=['Check','Expected','Status'])
alignment.to_csv(ART/'spatiotemporal_alignment_report.csv',index=False)
cutoff''')

code(r'''physical=pd.DataFrame([
['precipitation_nonnegative','total_precipitation >= 0','pending external data',STATUS],
['runoff_nonnegative','surface/subsurface runoff >= 0','pending external data',STATUS],
['soil_water_range','0 <= volumetric soil water <= plausible porosity; flag, do not clip silently','pending external data',STATUS],
['evaporation_sign','document ECMWF convention; convert loss magnitude explicitly','pending external data',STATUS],
['hydrology_units','moda m/day × 1000 × days_in_month = mm/month','pending external data',STATUS],
['temperature_units','K to °C only if later admitted','not applicable; rejected','complete'],
['accumulation_boundary','calendar month; no centered window/backfill','policy fixed','complete'],
],columns=['Check','Rule','Observed','Status'])
physical.to_csv(ART/'physical_consistency_report.csv',index=False)

val_cols=['Feature_set','Validation_system','Overall_RMSE','Known_RMSE','Masked_RMSE','Mean_fold_RMSE','Fold_std','Worst_fold','South_RMSE','North_RMSE','Dry_RMSE','Wet_RMSE','Training_seconds','Prediction_seconds','Gain_vs_E0','Status']
validations=['Notebook11_temporal','strict_date_grouped_temporal','irregular_date_hindcast','spatial_block','space_time']
validation=pd.DataFrame([['E0/E1/E2/E3',v]+[np.nan]*13+[STATUS] for v in validations],columns=val_cols)
validation.to_csv(ART/'validation_results.csv',index=False)
abl_cols=['Feature_group','Temporal_RMSE','Irregular_RMSE','Spatial_RMSE','Space_time_RMSE','Masked_RMSE','Worst_fold','Gain_vs_E0','Training_cost','Accepted','Status']
groups=['Precipitation','Evaporation','Runoff','Snow','Soil moisture profile','Temperature/energy proxy','Water-budget aggregates','Antecedent accumulation features']
ablation=pd.DataFrame([[g]+[np.nan]*8+['Not evaluated',STATUS] for g in groups],columns=abl_cols)
ablation.to_csv(ART/'era5_ablation.csv',index=False)
shift=pd.DataFrame(columns=['Feature','Split','Missing_rate','Mean','Std','Median','Q01','Q05','Q25','Q75','Q95','Q99','Min','Max','PSI_vs_train','KS_vs_train','Fold_stability','Status'])
shift.loc[0]=['not available','train/validation/test']+[np.nan]*15+[STATUS]; shift.to_csv(ART/'era5_distribution_shift.csv',index=False)
importance=pd.DataFrame([['not evaluated','none',np.nan,np.nan,np.nan,'Feature importance is gated on validation improvement',STATUS]],columns=['Feature','Expert','Gain_importance','Permutation_importance','SHAP_mean_abs','Interpretation','Status'])
importance.to_csv(ART/'era5_feature_importance.csv',index=False)
submission=pd.DataFrame([['none',False,0,'not evaluated','No submission: external data and validation unavailable',STATUS]],columns=['Submission','Created','Row_count','Integrity','Reason','Status'])
submission.to_csv(ART/'submission_report.csv',index=False)
print('All fail-safe audit schemas written; no metric or prediction was fabricated.')''')

md('''## Controlled feature construction once data are available

- E1: seven raw variables at strict `t-1`.
- E2: only `P-ET`, `P-ET-total_runoff`, total runoff, and—if extra layers are later admitted—a soil-profile mean/deep-minus-surface contrast. Antecedent 3/6/12-month sums are causal `shift(1).rolling(...)` relative to prediction origin; no centered rolling, backfill or validation-fitted climatology.
- E3: smallest fold-stable subset chosen inside training folds only.

Validation must reproduce Notebook 11 exactly before E1. The Notebook 08 split is diagnostic, never the decision metric. Canonical known/masked experts, prior, selected blend, seed and parameters remain frozen. Importance/SHAP runs only after an acceptance gate is passed.''')

code(r'''answers={
1:'Conditionally compatible: public, operational and CC-BY, but each vintage must satisfy the one-month rule.',
2:'Strict t-1.',3:'None; no CDS credentials or raw ERA5-Land files were available.',4:'Eight candidates were deferred/rejected to keep the pilot to seven direct budget variables; see CSV.',
5:'Not tested; alignment contract and scripts are ready.',6:'Residual risk remains because current consolidated files overwrite ERA5-Land-T; no future data were used here.',
7:'Not evaluated.',8:'Not evaluated.',9:'Not identifiable without data and passed gates.',10:'Not computed.',11:'Not computed.',12:'Not computed.',13:'Not computed.',14:'Not computed.',15:'No.',16:'No.',
17:'No new public submission exists.',18:'Preparation is reproducible; empirical pipeline is blocked pending authenticated data.',
19:'Accept the CDS dataset terms, configure ~/.cdsapirc, install the documented packages, run the pilot download, then pilot preparation.',
20:'Only if Notebook 28 passes: basin-level multiresolution aggregation; otherwise stop and retain canonical.'}
report='# Notebook 28 final recommendation\n\n**Status: blocked: external data unavailable. No validation or submission was produced.**\n\n'+''.join(f'{k}. {v}\n\n' for k,v in answers.items())
(ART/'final_recommendation.md').write_text(report,encoding='utf-8')
print(report)''')

md('''## Safe continuation commands

```powershell
pip install "cdsapi>=0.7.7" xarray netCDF4 pyarrow
python scripts/download_era5_land.py --pilot --execute
python scripts/prepare_era5_land_features.py --pilot --method nearest
python scripts/prepare_era5_land_features.py --pilot --method linear
```

Before execution, create a CDS account, accept the monthly ERA5-Land dataset terms, and configure the token in the user-level `.cdsapirc`. Do not paste the token into a notebook, command argument or repository file. After both pilots pass physical/alignment checks, execute the full download. Existing cached files are checksum-recorded and not downloaded twice.''')

OUT.parent.mkdir(parents=True,exist_ok=True); nbf.write(nb,OUT); print(OUT)
