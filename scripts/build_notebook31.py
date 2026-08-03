"""Build Notebook 31 extended ERA5-Land pilot."""
from pathlib import Path
import nbformat as nbf
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'notebooks/31_era5_land_extended_pilot.ipynb'
nb=nbf.v4.new_notebook(); nb.metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3'}}
nb.cells=[nbf.v4.new_markdown_cell('''# 31 — ERA5-Land Extended Pilot

This notebook extends only time coverage (2010–2018) and three physical features while preserving the three Notebook 29 regions, nearest interpolation, strict `t-1` cutoff, canonical-equivalent Notebook 11 LightGBM parameters and complete-date validation.

Frozen gates: A) E1/E2 overall gain ≥0.0015 and worst fold non-worse; B) masked gain ≥0.003 and known deterioration ≤0.001; or C) at least two of three regions improve and overall does not worsen. The decision is fixed to one of `stop ERA5 research`, `expand pilot once more`, or `proceed to full global download`. This notebook never initiates a global download and never creates a submission.

The six raw variables remain unchanged. Derived features are only `water_balance_proxy`, `total_runoff`, and `soil_profile_gradient`. ERA5 evaporation sign is verified from real files before `water_balance_proxy` is formed.'''),nbf.v4.new_code_cell("import sys\nfrom pathlib import Path\n_repo=Path.cwd().resolve().parent if Path.cwd().name=='notebooks' else Path.cwd().resolve()\n_site=_repo/'.venv/Lib/site-packages'\nif _site.exists() and str(_site) not in sys.path: sys.path.insert(0,str(_site))\n%run ../scripts/notebook31_runtime.py"),nbf.v4.new_markdown_cell('''## Interpretation boundary

E3 is selected only from OOF gain stability and never from test or leaderboard information, but its score is still exploratory because selection and reporting share folds. Full-download permission, if a gate passes, is a future manual action—not an automatic operation or a claim of leaderboard gain.''')]
OUT.parent.mkdir(parents=True,exist_ok=True); nbf.write(nb,OUT); print(OUT)
