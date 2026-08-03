"""Create or execute a minimal ERA5-Land monthly download plan.

Safe defaults: dry-run, never prints credentials, never overwrites a valid cache.
Actual retrieval requires ``--execute``, cdsapi and a user-managed ~/.cdsapirc.
"""
from __future__ import annotations
import argparse, calendar, hashlib, json, re, sys, traceback
from pathlib import Path
import pandas as pd

DATASET = "reanalysis-era5-land-monthly-means"
VARIABLES = [
    "total_precipitation", "total_evaporation", "surface_runoff",
    "sub_surface_runoff", "snowfall", "snowmelt",
    "volumetric_soil_water_layer_4",
]
PILOT_VARIABLES = ["total_precipitation", "total_evaporation", "surface_runoff",
                   "sub_surface_runoff", "volumetric_soil_water_layer_1",
                   "volumetric_soil_water_layer_3"]
PILOT_REGIONS = {
    "north": (50.0, -10.0, 35.0, 5.0),
    "equator": (7.5, 20.0, -7.5, 35.0),
    "south": (-20.0, 120.0, -35.0, 135.0),
}

def root() -> Path: return Path(__file__).resolve().parents[1]

def periods() -> list[pd.Period]:
    raw = root()/"data"/"raw"
    dates=[]
    for name in ("Train.csv","Test.csv"):
        dates.extend(pd.to_datetime(pd.read_csv(raw/name,usecols=["time"])["time"]).tolist())
    # strict t-1 requires the month before the earliest origin.
    lo=pd.Period(min(dates),freq="M")-1; hi=pd.Period(max(dates),freq="M")-1
    return list(pd.period_range(lo,hi,freq="M"))

def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()

def safe_exception_message(exc: BaseException) -> str:
    """Return diagnostic text while redacting credential-like values."""
    message = str(exc)
    patterns = [
        (r"(?i)(key|token|api[_ -]?key|authorization|password)(\s*[:=]\s*)([^\s,;]+)", r"\1\2<redacted>"),
        (r"(?i)(https?://[^:/\s]+:)([^@/\s]+)(@)", r"\1<redacted>\3"),
        (r"(?i)(bearer\s+)([^\s,;]+)", r"\1<redacted>"),
    ]
    for pattern, replacement in patterns:
        message = re.sub(pattern, replacement, message)
    return message

def print_safe_traceback(exc: BaseException) -> None:
    """Print the full stack without allowing the raw exception text to expose secrets."""
    print("Traceback (most recent call last):", file=sys.stderr)
    for line in traceback.format_tb(exc.__traceback__):
        print(line, end="", file=sys.stderr)
    print(f"{type(exc).__name__}: {safe_exception_message(exc)}", file=sys.stderr)

def build_manifest() -> pd.DataFrame:
    out=[]
    for p in periods():
        target=root()/"data"/"external"/"era5_land"/"raw"/f"era5_land_{p}.nc"
        # Global 0.1° monthly float32 upper-bound approximation; compression varies.
        estimate_mb=7*1801*3600*4/1024**2
        out.append({"dataset":DATASET,"year":p.year,"month":p.month,"variables":"|".join(VARIABLES),
                    "area":"90,-180,-90,180","format":"netcdf","target_path":str(target.relative_to(root())),
                    "estimated_uncompressed_mb":round(estimate_mb,1),"exists":target.exists(),
                    "sha256":sha256(target) if target.exists() else "","status":"cached" if target.exists() else "planned"})
    return pd.DataFrame(out)

def build_pilot_manifest() -> pd.DataFrame:
    out=[]
    for region,area in PILOT_REGIONS.items():
        north,west,south,east=area
        # 0.1-degree cells x 12 months x 6 float32 variables, before compression/metadata.
        cells=(round((north-south)/.1)+1)*(round((east-west)/.1)+1)
        estimate_mb=cells*12*len(PILOT_VARIABLES)*4/1024**2
        for year in (2014,2015,2016):
            target=root()/"data/external/era5_land/pilot/raw"/f"{region}_{year}.nc"
            out.append({"dataset":DATASET,"region":region,"year":year,"months":"01-12",
                        "variables":"|".join(PILOT_VARIABLES),"area":",".join(map(str,area)),
                        "format":"netcdf","target_path":str(target.relative_to(root())),
                        "estimated_uncompressed_mb":round(estimate_mb,2),"exists":target.exists(),
                        "size_bytes":target.stat().st_size if target.exists() else 0,
                        "sha256":sha256(target) if target.exists() else "",
                        "status":"cached" if target.exists() else "planned"})
    return pd.DataFrame(out)

def main() -> int:
    ap=argparse.ArgumentParser(); ap.add_argument("--execute",action="store_true"); ap.add_argument("--pilot",action="store_true")
    args=ap.parse_args(); art=root()/"artifacts"/"notebook28"; art.mkdir(parents=True,exist_ok=True)
    manifest=build_pilot_manifest() if args.pilot else build_manifest()
    report=(root()/"artifacts/notebook29/download_report.csv") if args.pilot else (art/"era5_download_manifest.csv")
    report.parent.mkdir(parents=True,exist_ok=True)
    if not args.execute:
        manifest.to_csv(report,index=False)
        print(f"Dry run only: {len(manifest)} requests; estimated uncompressed {manifest.estimated_uncompressed_mb.sum()/1024:.1f} GiB")
        print("Run with --pilot --execute first. Credentials are read by cdsapi; never printed.")
        return 0
    try: import cdsapi
    except ImportError:
        manifest["status"]="blocked: cdsapi not installed"; manifest.to_csv(report,index=False); return 2
    try: client=cdsapi.Client()
    except Exception as e:
        print(
            f"cdsapi.Client() failed: {type(e).__name__}: {safe_exception_message(e)}",
            file=sys.stderr,
        )
        print_safe_traceback(e)
        manifest["status"]="blocked: CDS credentials/terms unavailable"; manifest.to_csv(report,index=False); return 3
    for i,row in manifest.iterrows():
        dest=root()/row.target_path; dest.parent.mkdir(parents=True,exist_ok=True)
        if dest.exists() and dest.stat().st_size>0: continue
        pilot=bool(args.pilot)
        request={"product_type":"monthly_averaged_reanalysis","variable":PILOT_VARIABLES if pilot else VARIABLES,
                 "year":str(row.year),"month":[f"{m:02d}" for m in range(1,13)] if pilot else f"{int(row.month):02d}",
                 "time":"00:00","data_format":"netcdf","download_format":"unarchived"}
        if pilot: request["area"]=[float(x) for x in row.area.split(',')]
        try: client.retrieve(DATASET,request,str(dest)); manifest.loc[i,["exists","sha256","status"]]=[True,sha256(dest),"downloaded"]
        except Exception as e:
            region = row.get("region", "global")
            print(
                "client.retrieve() failed: "
                f"region={region}, year={row.year}, destination={dest}, "
                f"exception={type(e).__name__}: {safe_exception_message(e)}",
                file=sys.stderr,
            )
            print_safe_traceback(e)
            manifest.loc[i,"status"]=f"failed: {type(e).__name__}"; break
        manifest.to_csv(report,index=False)
    manifest.to_csv(report,index=False); return 0

if __name__=="__main__": raise SystemExit(main())
