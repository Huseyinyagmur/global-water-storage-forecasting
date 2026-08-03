"""Build Notebook 29 from the real-data, credential-independent runtime."""
from pathlib import Path
import nbformat as nbf
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'notebooks/29_era5_land_pilot_validation.ipynb'
nb=nbf.v4.new_notebook(); nb.metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3'}}
nb.cells=[
 nbf.v4.new_markdown_cell('''# 29 — ERA5-Land Pilot Validation (real cached data)

This notebook never downloads data and never checks CDS credentials. Availability is determined only by nine expected, non-empty, xarray-readable NetCDF files and the processed Parquet contract. It validates physical units/ranges, nearest versus bilinear alignment, strict `t-1` cutoff, the real train merge, and E0/E1 models with complete-date temporal folds.

The model family and parameters are canonical-equivalent to Notebook 11: fixed LightGBM parameters, known/masked experts and the 0.65/0.35 masked prior blend. The pilot is only 15 training dates across three selected regions; its RMSE cannot establish global or leaderboard performance.'''),
 nbf.v4.new_code_cell("import sys\nfrom pathlib import Path\n_repo=Path.cwd().resolve().parent if Path.cwd().name=='notebooks' else Path.cwd().resolve()\n_site=_repo/'.venv/Lib/site-packages'\nif _site.exists() and str(_site) not in sys.path: sys.path.insert(0,str(_site))\n%run ../scripts/notebook29_runtime.py"),
 nbf.v4.new_markdown_cell('''## Decision rule

`completed: pilot validated` requires all nine NetCDF files valid, physical checks passed, 100% cutoff safety, at least 95% merge coverage and error-free E0/E1 validation. `completed: pilot valid but not promising` is used when plumbing passes but E1 materially worsens. Credentials are consulted only by the separate downloader when new data are explicitly requested.''')]
OUT.parent.mkdir(parents=True,exist_ok=True); nbf.write(nb,OUT); print(OUT)
