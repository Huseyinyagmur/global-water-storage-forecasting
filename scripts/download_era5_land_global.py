"""Manifest-first, resumable ERA5-Land global competition-grid downloader.

Dry-run is the default. Execution additionally requires both ``--execute`` and
the literal confirmation flag. Credentials are managed by cdsapi and never logged.
"""
from __future__ import annotations
import argparse, hashlib, json, re, shutil, sys, time, traceback
from pathlib import Path
import numpy as np, pandas as pd

ROOT=Path(__file__).resolve().parents[1]; ART=ROOT/'artifacts/notebook32'; ART.mkdir(parents=True,exist_ok=True)
RAW=ROOT/'data/external/era5_land/global/raw'; RAW.mkdir(parents=True,exist_ok=True)
DATASET='reanalysis-era5-land-monthly-means'; VARS=['total_precipitation','total_evaporation','surface_runoff','sub_surface_runoff','volumetric_soil_water_layer_1','volumetric_soil_water_layer_3']; SHORT={'tp','e','sro','ssro','swvl1','swvl3'}; TILE=30
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
 return h.hexdigest()
def safe(e):
 s=str(e)
 for pat,rep in [(r'(?i)(key|token|authorization|password)(\s*[:=]\s*)([^\s,;]+)',r'\1\2<redacted>'),(r'(?i)(https?://[^:/\s]+:)([^@/\s]+)(@)',r'\1<redacted>\3'),(r'(?i)(bearer\s+)([^\s,;]+)',r'\1<redacted>')]: s=re.sub(pat,rep,s)
 return s
def trace(e):
 print('Traceback (most recent call last):',file=sys.stderr)
 for x in traceback.format_tb(e.__traceback__): print(x,end='',file=sys.stderr)
 print(f'{type(e).__name__}: {safe(e)}',file=sys.stderr)
def months_needed():
 d=[]
 for f in ['Train.csv','Test.csv']: d.extend(pd.to_datetime(pd.read_csv(ROOT/'data/raw'/f,usecols=['time']).time).dt.to_period('M').tolist())
 return pd.period_range(min(d)-1,max(d)-1,freq='M')
def used_tiles():
 q=pd.concat([pd.read_csv(ROOT/'data/raw/Train.csv',usecols=['lat','lon']),pd.read_csv(ROOT/'data/raw/Test.csv',usecols=['lat','lon'])]).drop_duplicates()
 q['ti']=np.floor((q.lat+90)/TILE).astype(int); q['tj']=np.floor((q.lon+180)/TILE).astype(int)
 return sorted(set(zip(q.ti,q.tj)))
def valid_nc(path,expected_months):
 try:
  import xarray as xr
  with xr.open_dataset(path) as ds: return SHORT<=set(ds.data_vars) and ds.sizes.get('valid_time',0)==expected_months and all(ds.sizes.get(x,0)>0 for x in ['latitude','longitude'])
 except Exception: return False
def build_manifest():
 periods=months_needed(); rows=[]
 for ti,tj in used_tiles():
  south=-90+ti*TILE; north=min(90,south+TILE); west=-180+tj*TILE; east=min(180,west+TILE); tile=f't{ti:02d}_{tj:02d}'
  cells=(round((north-south)/.1)+1)*(round((east-west)/.1)+1)
  for year in sorted(set(periods.year)):
   months=sorted(periods[periods.year==year].month); dest=RAW/tile/f'{tile}_{year}.nc'; expected=len(months); good=dest.exists() and dest.stat().st_size>0 and valid_nc(dest,expected)
   uncompressed=cells*expected*len(VARS)*4/1024**2
   rows.append({'year':year,'tile':tile,'months':'|'.join(f'{m:02d}' for m in months),'variables':'|'.join(VARS),'date_start':f'{year}-{months[0]:02d}','date_end':f'{year}-{months[-1]:02d}','area':f'{north},{west},{south},{east}','target_path':str(dest.relative_to(ROOT)),'expected_uncompressed_mb':round(uncompressed,2),'expected_download_mb':round(uncompressed*.44,2),'actual_size_bytes':dest.stat().st_size if dest.exists() else 0,'sha256':sha(dest) if good else '','status':'cached' if good else 'partial_invalid' if dest.exists() else 'planned'})
 return pd.DataFrame(rows)
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--execute',action='store_true'); ap.add_argument('--confirm',default=''); ap.add_argument('--max-retries',type=int,default=4); args=ap.parse_args(); m=build_manifest(); path=ART/'download_manifest.csv'; m.to_csv(path,index=False)
 disk=shutil.disk_usage(ROOT); pending=m.status.ne('cached'); unc=m.loc[pending,'expected_uncompressed_mb'].sum()/1024; dl=m.loc[pending,'expected_download_mb'].sum()/1024; processed_gb=2_435_000*9*4/1024**3; minimum=dl+unc+processed_gb+5
 report=pd.DataFrame([{'strategy':'30-degree used-coordinate tiles','tile_count':m.tile.nunique(),'request_count_total':len(m),'request_count_pending':int(pending.sum()),'source_start':months_needed().min().strftime('%Y-%m'),'source_end':months_needed().max().strftime('%Y-%m'),'estimated_download_gb':dl,'estimated_uncompressed_gb':unc,'estimated_processed_gb':processed_gb,'minimum_required_free_gb':minimum,'free_disk_gb':disk.free/1024**3,'disk_gate_passed':disk.free/1024**3>=minimum,'estimated_duration_hours':'10-24 depending on CDS queue','exact_command':'.venv\\Scripts\\python.exe scripts\\download_era5_land_global.py --execute --confirm I_APPROVE_GLOBAL_ERA5_DOWNLOAD'}]); report.to_csv(ART/'disk_space_report.csv',index=False)
 print(report.to_string(index=False)); print(f'Manifest: {path}')
 if not args.execute: return 0
 if args.confirm!='I_APPROVE_GLOBAL_ERA5_DOWNLOAD': print('Refused: explicit confirmation literal missing.',file=sys.stderr); return 4
 if not report.disk_gate_passed.iloc[0]: print('Refused: insufficient disk.',file=sys.stderr); return 5
 try: import cdsapi; client=cdsapi.Client()
 except Exception as e: print(f'CDS initialization failed: {type(e).__name__}: {safe(e)}',file=sys.stderr); trace(e); return 3
 for i,r in m.iterrows():
  if r.status=='cached': continue
  dest=ROOT/Path(r.target_path); dest.parent.mkdir(parents=True,exist_ok=True); part=dest.with_suffix('.nc.part')
  if part.exists(): part.unlink()
  request={'product_type':'monthly_averaged_reanalysis','variable':VARS,'year':str(r.year),'month':r.months.split('|'),'time':'00:00','area':[float(x) for x in r.area.split(',')],'data_format':'netcdf','download_format':'unarchived'}
  for attempt in range(args.max_retries):
   try:
    client.retrieve(DATASET,request,str(part))
    if not valid_nc(part,len(r.months.split('|'))): raise ValueError('downloaded NetCDF failed structural validation')
    part.replace(dest); m.loc[i,['actual_size_bytes','sha256','status']]=[dest.stat().st_size,sha(dest),'downloaded']; m.to_csv(path,index=False); break
   except Exception as e:
    print(f'retrieve failed: tile={r.tile}, year={r.year}, destination={dest}, attempt={attempt+1}, {type(e).__name__}: {safe(e)}',file=sys.stderr); trace(e)
    if part.exists(): part.unlink()
    if attempt+1==args.max_retries: m.loc[i,'status']=f'failed: {type(e).__name__}'; m.to_csv(path,index=False); return 6
    time.sleep(min(300,15*2**attempt))
 return 0
if __name__=='__main__': raise SystemExit(main())
