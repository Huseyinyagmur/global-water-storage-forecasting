import nbformat as nbf
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/"notebooks/25_oof_tail_risk_and_spatial_support_audit.ipynb"; runtime=(ROOT/"scripts/notebook25_runtime.py").read_text(encoding="utf-8")
nb=nbf.v4.new_notebook(); nb.cells=[
 nbf.v4.new_markdown_cell("""# 25 — OOF Tail Risk and Spatial Support Audit

This notebook creates no feature, correction or submission. It persists leakage-controlled row-level OOF predictions
and diagnoses tail risk and train/test support. Residual is fixed as **target − prediction**; positive means
underprediction.

## Frozen methodology

Primary strategy is two complete-date strict temporal folds. It is called **canonical-equivalent leakage-controlled
OOF**, not exact canonical OOF: the expert/blend structure is retained while the risky location-major
`prior_global_month` is replaced by strict date-grouped history. Large-error thresholds are OOF P90, OOF P95 and
2× median absolute error. Spatial support is `0.50 seen-location + 0.25 exp(-nearest_km/250) + 0.25 min(history/24,1)`;
temporal support is a fixed weighted combination of forecast gap, seasonal history and location recency. Combined
support is their arithmetic mean. No threshold is tuned against results."""),
 nbf.v4.new_markdown_cell("""## Input gate, cached OOF generation and audit

Fold caches are reused only when their frozen policy key matches. Each validation date is later than its training
maximum, validation targets are never feature inputs, and every primary OOF row is unique."""),
 nbf.v4.new_code_cell(runtime)]
nb.metadata={"kernelspec":{"display_name":"GWS project environment","language":"python","name":"gws"},"language_info":{"name":"python","version":"3.13"}}
nbf.write(nb,OUT); print(OUT)
