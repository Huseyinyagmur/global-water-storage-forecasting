"""Build the approval-gated Notebook 32 preparation notebook."""
from pathlib import Path
import nbformat as nbf
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'notebooks/32_era5_land_global_integration_and_submission.ipynb'
nb=nbf.v4.new_notebook(); nb.metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3'}}
nb.cells=[nbf.v4.new_markdown_cell('''# 32 — ERA5-Land Global Integration and Submission

## Approval-gated preparation run

Notebook 31 passed all frozen pilot gates, but the global download is deliberately not automatic. This notebook first creates a resumable manifest and disk-space audit. It does not download without a separate explicit confirmation literal.

Spatial choice is fixed before results: 30° tiles that intersect the 15,715 competition coordinates. This avoids a blind near-global rectangle while limiting CDS rate-limit exposure relative to 10° tiles. Required source months are computed as prediction origin minus one month. Only the six pilot-validated raw variables and three fixed derived features are permitted.

Future empirical gates are frozen: A) strict gain ≥0.002 with irregular non-worse and spatial deterioration ≤0.002; B) masked gain ≥0.004, known deterioration ≤0.001 and worst non-worse; C) space-time gain ≥0.003 with other systems within 0.001; or D) pilot direction reproduced for overall, masked and worst-fold. Notebook 21 correction is secondary, has historical transferability 54.53/100, and cannot run before primary acceptance.'''),nbf.v4.new_code_cell("import sys\nfrom pathlib import Path\n_repo=Path.cwd().resolve().parent if Path.cwd().name=='notebooks' else Path.cwd().resolve()\n_site=_repo/'.venv/Lib/site-packages'\nif _site.exists() and str(_site) not in sys.path: sys.path.insert(0,str(_site))\n%run ../scripts/notebook32_runtime.py"),nbf.v4.new_markdown_cell('''## Safety boundary

The exact command printed above is informational. Run it only after explicitly approving the estimated size, duration and disk use. The downloader uses per-file structural validation, `.part` files, SHA256, cache skipping, four retries and exponential backoff. Credentials are never printed. Raw/processed global data and cache directories are git-ignored.''')]
OUT.parent.mkdir(parents=True,exist_ok=True); nbf.write(nb,OUT); print(OUT)
