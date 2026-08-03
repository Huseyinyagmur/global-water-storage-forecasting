"""Prepare cutoff-safe ERA5-Land features; never fabricates missing data."""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np, pandas as pd

def root(): return Path(__file__).resolve().parents[1]

PILOT_BOXES={'north':(35,50,-10,5),'equator':(-7.5,7.5,20,35),'south':(-35,-20,120,135)}

def prepare_pilot(xr, method, files):
    r=root(); out=r/'data/external/era5_land/pilot/processed'; out.mkdir(parents=True,exist_ok=True)
    train=pd.read_csv(r/'data/raw/Train.csv',usecols=['time','lat','lon'])
    train['date']=pd.to_datetime(train.time); train['prediction_origin']=train.date.dt.to_period('M')
    selected=[]
    for region,(s,n,w,e) in PILOT_BOXES.items():
        q=train[train.lat.between(s,n)&train.lon.between(w,e)].copy(); q['region']=region; selected.append(q)
    base=pd.concat(selected,ignore_index=True)
    base['era5_source_month']=base.prediction_origin-1
    base=base[base.era5_source_month.between(pd.Period('2014-01','M'),pd.Period('2016-12','M'))].copy()
    base['era5_lag_months']=1; base['cutoff_safe_flag']=base.era5_source_month<base.prediction_origin
    pieces=[]
    for region,g in base.groupby('region',sort=False):
        region_files=[p for p in files if p.name.startswith(region+'_')]
        datasets=[]
        for p in region_files:
            d=xr.open_dataset(p); rename={k:v for k,v in {'valid_time':'time','longitude':'lon','latitude':'lat'}.items() if k in d.coords}; d=d.rename(rename)
            d=d.assign_coords(lon=((d.lon+180)%360)-180).sortby('lon'); datasets.append(d.astype('float32'))
        era=xr.concat(datasets,dim='time').sortby('time')
        for month,h in g.groupby('era5_source_month',sort=False):
            src=era.sel(time=np.datetime64(month.start_time),method='nearest')
            q=src.interp(lat=xr.DataArray(h.lat.to_numpy(),dims='row'),lon=xr.DataArray(h.lon.to_numpy(),dims='row'),method=method)
            z=q.to_dataframe().reset_index(drop=True); z['date']=h.date.to_numpy(); z['lat']=h.lat.to_numpy(); z['lon']=h.lon.to_numpy(); z['region']=region
            z['era5_source_month']=str(month); z['cutoff_safe_flag']=True; z['interpolation_method']=method; pieces.append(z)
    result=pd.concat(pieces,ignore_index=True)
    assert result.cutoff_safe_flag.all() and not result.duplicated(['date','lat','lon','region']).any()
    keep=['date','lat','lon','region','era5_source_month']+[c for c in result if c not in ['date','lat','lon','region','era5_source_month','time','row']]+['cutoff_safe_flag','interpolation_method']
    keep=list(dict.fromkeys(keep)); result[keep].to_parquet(out/'era5_land_pilot_features.parquet',index=False)
    return result

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--method",choices=["nearest","linear"],default="nearest"); ap.add_argument("--pilot",action="store_true")
    args=ap.parse_args(); start=time.time(); r=root(); raw=r/"data/external/era5_land/raw"; out=r/"data/external/era5_land/processed"; out.mkdir(parents=True,exist_ok=True)
    if args.pilot: raw=r/'data/external/era5_land/pilot/raw'; files=sorted(raw.glob('*.nc'))
    else: files=sorted(raw.glob("era5_land_*.nc"))
    if not files: print("blocked: external data unavailable; run download_era5_land.py"); return 2
    try: import xarray as xr
    except ImportError: print("blocked: install xarray, netCDF4 and pyarrow"); return 3
    if args.pilot:
        if len(files)!=9: print(f'blocked: expected 9 pilot files, found {len(files)}'); return 4
        result=prepare_pilot(xr,args.method,files); print({'status':'complete','method':args.method,'rows':len(result),'elapsed_seconds':round(time.time()-start,2)}); return 0
    frames=[]
    for path in files[:1] if args.pilot else files:
        ds=xr.open_dataset(path)
        rename={k:v for k,v in {'valid_time':'time','longitude':'lon','latitude':'lat'}.items() if k in ds.coords}
        ds=ds.rename(rename); ds=ds.assign_coords(lon=((ds.lon+180)%360)-180).sortby('lon')
        frames.append(ds.astype('float32'))
    era=xr.concat(frames,dim='time').sortby('time')
    for split in ['Train','Test']:
        base=pd.read_csv(r/f"data/raw/{split}.csv",usecols=['ID','time','lat','lon'])
        if args.pilot: base=base.head(5000)
        base['prediction_origin']=pd.to_datetime(base.time).dt.to_period('M')
        base['era5_source_month']=base.prediction_origin-1; base['era5_lag_months']=1
        pieces=[]
        for month,g in base.groupby('era5_source_month',sort=False):
            src=era.sel(time=np.datetime64(month.start_time),method='nearest')
            q=src.interp(lat=xr.DataArray(g.lat.to_numpy(),dims='row'),lon=xr.DataArray(g.lon.to_numpy(),dims='row'),method=args.method)
            z=q.to_dataframe().reset_index(drop=True); z.insert(0,'ID',g.ID.to_numpy()); z['prediction_origin']=g.prediction_origin.astype(str).to_numpy(); z['era5_source_month']=str(month); z['era5_lag_months']=1; z['cutoff_safe_flag']=True; pieces.append(z)
        result=pd.concat(pieces,ignore_index=True).set_index('ID').loc[base.ID].reset_index()
        assert len(result)==len(base) and result.ID.is_unique and result.cutoff_safe_flag.all()
        result.to_parquet(out/f"{split.lower()}_{args.method}.parquet",index=False)
    print({'status':'complete','method':args.method,'elapsed_seconds':round(time.time()-start,2)})
    return 0
if __name__=='__main__': raise SystemExit(main())
