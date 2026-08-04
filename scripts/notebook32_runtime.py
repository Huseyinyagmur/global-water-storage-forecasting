"""Notebook 32 orchestration. Preparation-only until explicit global-download approval."""
from pathlib import Path
import hashlib, json, subprocess, sys
import numpy as np, pandas as pd
ROOT=Path.cwd().resolve(); ROOT=ROOT.parent if ROOT.name=='notebooks' else ROOT; ART=ROOT/'artifacts/notebook32'; ART.mkdir(parents=True,exist_ok=True)
r=subprocess.run([sys.executable,str(ROOT/'scripts/download_era5_land_global.py')],cwd=ROOT,text=True,capture_output=True); print(r.stdout); assert r.returncode==0
m=pd.read_csv(ART/'download_manifest.csv'); disk=pd.read_csv(ART/'disk_space_report.csv'); incomplete=~m.status.isin(['cached','downloaded']); ready=bool(disk.disk_gate_passed.iloc[0])
cols=['tp','e','sro','ssro','swvl1','swvl3']; pending='pending controlled global download approval'
pd.DataFrame([{'files_total':len(m),'files_validated':int((~incomplete).sum()),'variables':'|'.join(cols),'time_start':'2002-04','time_end':'2018-11','status':pending}]).to_csv(ART/'netcdf_validation.csv',index=False)
pd.DataFrame([{'expected_rows':2_434_982,'audited_rows':0,'safe_fraction':np.nan,'future_rows':np.nan,'policy':'era5_source_month = prediction_origin - 1','status':pending}]).to_csv(ART/'cutoff_audit.csv',index=False)
pd.DataFrame([{'method':'nearest','tile_size_degrees':30,'used_tiles':m.tile.nunique(),'coordinate_mapping_cache':'data/external/era5_land/global/processed/coordinate_mapping.parquet','status':pending}]).to_csv(ART/'spatial_alignment_report.csv',index=False)
pd.DataFrame([{'variables':'tp,e,sro,ssro,swvl1,swvl3','checks':'finite ranges; evaporation sign; units; no infinity','status':pending}]).to_csv(ART/'physical_consistency_report.csv',index=False)
pd.DataFrame([{'expected_rows':2_434_982,'actual_rows':0,'train_coverage':np.nan,'test_coverage':np.nan,'duplicates':np.nan,'missing_rate':np.nan,'status':pending}]).to_csv(ART/'merge_report.csv',index=False)
valcols=['Feature_set','Validation_system','Overall_RMSE','Known_RMSE','Masked_RMSE','Mean_fold_RMSE','Fold_std','Worst_fold','North_RMSE','South_RMSE','Dry_RMSE','Wet_RMSE','Training_seconds','Prediction_seconds','Status']
pd.DataFrame([['E0/E1/E2/E3','Notebook11/strict/irregular/spatial/space-time']+[np.nan]*12+[pending]],columns=valcols).to_csv(ART/'validation_results.csv',index=False)
pd.DataFrame([['E0/E1/E2/E3','north/south/dry/wet',np.nan,pending]],columns=['Feature_set','Region_or_regime','RMSE','Status']).to_csv(ART/'regional_results.csv',index=False)
pd.DataFrame([['not evaluated','none',np.nan,np.nan,pending]],columns=['Feature','Expert','Gain_importance','Permutation_importance','Status']).to_csv(ART/'feature_importance.csv',index=False)
pd.DataFrame([['none',False,0,'Acceptance gates cannot be evaluated before download/validation',pending]],columns=['Submission','Created','Rows','Reason','Status']).to_csv(ART/'submission_report.csv',index=False)
status='ready for controlled global download' if ready else 'blocked: insufficient disk or access'
report=f'''# Notebook 32 final recommendation\n\n**Status: {status}**\n\n- Strategy: 30-degree tiles intersecting competition coordinates; no blind global bbox.\n- Used tiles: {m.tile.nunique()}; requests: {len(m)} total, {int(incomplete.sum())} pending.\n- Source months: {disk.source_start.iloc[0]} to {disk.source_end.iloc[0]}.\n- Estimated download: {disk.estimated_download_gb.iloc[0]:.2f} GB; uncompressed: {disk.estimated_uncompressed_gb.iloc[0]:.2f} GB.\n- Free disk: {disk.free_disk_gb.iloc[0]:.2f} GB; minimum required: {disk.minimum_required_free_gb.iloc[0]:.2f} GB; gate passed: {bool(disk.disk_gate_passed.iloc[0])}.\n- Estimated CDS duration: {disk.estimated_duration_hours.iloc[0]}.\n- Exact command after explicit approval: `{disk.exact_command.iloc[0]}`.\n- No global download, validation, prediction or submission was performed in this preparation run.\n\nAfter download completes, run `python scripts/prepare_era5_land_global_features.py`, then rerun Notebook 32 for empirical validation. Existing canonical/current-best submissions remain immutable.\n'''
(ART/'final_recommendation.md').write_text(report,encoding='utf-8'); print(report)
