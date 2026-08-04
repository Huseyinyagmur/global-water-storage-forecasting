"""Build Notebook 33 controlled sequence pilot."""
from pathlib import Path
import nbformat as nbf
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'notebooks/33_global_panel_sequence_model.ipynb'
nb=nbf.v4.new_notebook(); nb.metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3'}}
nb.cells=[nbf.v4.new_markdown_cell('''# 33 — Global Panel Sequence Model

This notebook tests one bounded sequence hypothesis without external data. Prediction-time inputs are `TWS_t`, SPEI 1/3/6/12, soil moisture, calendar sin/cos, latitude/longitude and a TWS-missing indicator. Target-derived priors, future rows, validation targets, centered windows and teacher forcing are prohibited.

Frozen gates: A) S1 overall gain ≥0.002 and worst non-worse; B) OOF nonnegative blend gain ≥0.0015 with strict/space-time direction consistency; C) masked gain ≥0.004 with known deterioration ≤0.001. Full validation starts only after positive same-row pilot signal. Pilot selection is staged rather than factorial: GRU/LSTM/TCN at L=6, lengths 3/6/12 for the winning family, then single versus dual experts.'''),nbf.v4.new_code_cell("import torch, sys\nfrom pathlib import Path\n_repo=Path.cwd().resolve().parent if Path.cwd().name=='notebooks' else Path.cwd().resolve()\n_site=_repo/'.venv/Lib/site-packages'\nif _site.exists() and str(_site) not in sys.path: sys.path.append(str(_site))\n%run ../scripts/notebook33_runtime.py"),nbf.v4.new_markdown_cell('''## Scope boundary

Notebook 25 canonical-equivalent strict OOF supplies exact same-row E0 predictions. Pilot locations are fixed, latitude-stratified and selected without outcomes. If the pilot does not justify full training, irregular and space-time runs remain explicitly unavailable rather than fabricated. No ERA5 download or submission occurs.''')]
OUT.parent.mkdir(parents=True,exist_ok=True); nbf.write(nb,OUT); print(OUT)
