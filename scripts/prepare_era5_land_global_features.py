"""Prepare global nearest-interpolated features after a complete validated manifest."""
from pathlib import Path
import json, sys
import numpy as np, pandas as pd
ROOT=Path(__file__).resolve().parents[1]; ART=ROOT/'artifacts/notebook32'; OUT=ROOT/'data/external/era5_land/global/processed'; OUT.mkdir(parents=True,exist_ok=True); TILE=30; RAW=['tp','e','sro','ssro','swvl1','swvl3']
def main():
 try: import xarray as xr
 except ImportError: print('xarray is required',file=sys.stderr); return 2
 m=pd.read_csv(ART/'download_manifest.csv'); incomplete=~m.status.isin(['cached','downloaded'])
 if incomplete.any(): print(f'blocked: {incomplete.sum()} global files incomplete',file=sys.stderr); return 3
 frames=[]; map_rows=[]
 for split,file,idcol in [('train','Train.csv','sample_id'),('test','Test.csv','ID')]:
  d=pd.read_csv(ROOT/'data/raw'/file,usecols=[idcol,'time','lat','lon'],parse_dates=['time']).rename(columns={idcol:'row_id','time':'prediction_origin'}); d['source_period']=d.prediction_origin.dt.to_period('M')-1; d['ti']=np.floor((d.lat+90)/TILE).astype(int); d['tj']=np.floor((d.lon+180)/TILE).astype(int); d['tile']=[f't{a:02d}_{b:02d}' for a,b in zip(d.ti,d.tj)]
  parts=[]
  for (tile,year),g in d.groupby(['tile',d.source_period.dt.year],sort=False):
   row=m[(m.tile==tile)&(m.year==year)].iloc[0]; ds=xr.open_dataset(ROOT/Path(row.target_path))
   for month,h in g.groupby('source_period',sort=False):
    src=ds.sel(valid_time=np.datetime64(month.start_time)); q=src.interp(latitude=xr.DataArray(h.lat.to_numpy(),dims='row'),longitude=xr.DataArray(h.lon.to_numpy(),dims='row'),method='nearest'); z=q[RAW].to_dataframe().reset_index(drop=True); z['row_id']=h.row_id.to_numpy(); z['dataset']=split; z['prediction_origin']=h.prediction_origin.to_numpy(); z['era5_source_month']=str(month); z['era5_lag_months']=1; z['cutoff_safe_flag']=True; z['tile']=tile; parts.append(z)
   ds.close()
  out=pd.concat(parts,ignore_index=True); frames.append(out)
 allf=pd.concat(frames,ignore_index=True); allf['water_balance_proxy']=allf.tp+allf.e; allf['total_runoff']=allf.sro+allf.ssro; allf['soil_profile_gradient']=allf.swvl3-allf.swvl1; assert len(allf)==2_154_021+280_961 and not allf.duplicated(['dataset','row_id']).any() and allf.cutoff_safe_flag.all()
 final=OUT/'era5_land_global_features.parquet'; allf.to_parquet(final,index=False); pd.DataFrame([{'mapping_rows':len(allf),'unique_tiles':allf.tile.nunique(),'method':'nearest','cache_path':str(OUT/'coordinate_mapping.parquet')}]).to_csv(ART/'spatial_alignment_report.csv',index=False); allf[['dataset','row_id','tile']].to_parquet(OUT/'coordinate_mapping.parquet',index=False); print(final); return 0
if __name__=='__main__': raise SystemExit(main())
