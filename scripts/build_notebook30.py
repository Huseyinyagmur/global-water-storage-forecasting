"""Build the executable Notebook 30 wrapper around its reproducible runtime."""
from pathlib import Path
import nbformat as nbf
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'notebooks/30_spatial_graph_feature_engineering.ipynb'
nb=nbf.v4.new_notebook(); nb.metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3'}}
nb.cells=[nbf.v4.new_markdown_cell('''# 30 — Leakage-Free Spatial Graph Feature Engineering

This notebook tests one change only: graph-inspired features derived from prediction-time-available supplied covariates. The Notebook 11 LightGBM family, parameters, known/masked segmentation, prior logic and 0.65/0.35 masked blend remain fixed. No target, future TWS, residual, validation outcome, external data, new model family or ensemble enters feature construction.

The coordinate graph is built once with BallTree haversine distance for k ∈ {3,5,8,10,15,20}. k is screened on one predeclared strict temporal fold using neighbor means only, then frozen. For computational control, ablations use one representative Notebook 20 fold for each of strict temporal, irregular hindcast, spatial block and space-time validation. This is weaker than full-fold confirmation and is reported as such.

Same-date `SPEI_03_t`, `SPEI_12_t` and `SOIL_MOISTURE_t` are legal graph sources because they are supplied prediction-time covariates. `target` and TWS-derived graph aggregates are prohibited. A spatial holdout's supplied covariates may be transformed among validation nodes, but its targets never enter graph calculations.'''),
nbf.v4.new_code_cell("import sys\nfrom pathlib import Path\n_repo=Path.cwd().resolve().parent if Path.cwd().name=='notebooks' else Path.cwd().resolve()\n_venv_site=_repo/'.venv/Lib/site-packages'\nif _venv_site.exists() and str(_venv_site) not in sys.path: sys.path.insert(0,str(_venv_site))\n%run ../scripts/notebook30_runtime.py"),
nbf.v4.new_markdown_cell('''## Acceptance policy

A family is accepted only if strict temporal or space-time RMSE improves by at least 0.002, masked RMSE never worsens, and worst-fold RMSE never worsens across the four representative systems. One-fold-only wins, spatial degradation, leakage ambiguity and excessive cost are rejected. Submission is conservative by default and requires independent full-fold confirmation even if representative gates pass.

Outputs are written under `artifacts/notebook30/`; existing notebooks and submissions remain unchanged.''')]
OUT.parent.mkdir(parents=True,exist_ok=True); nbf.write(nb,OUT); print(OUT)
