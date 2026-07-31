import nbformat as nbf
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/"notebooks/23_correction_temporal_transferability.ipynb"
runtime=(ROOT/"scripts/notebook23_runtime.py").read_text(encoding="utf-8")
nb=nbf.v4.new_notebook(); nb.cells=[
 nbf.v4.new_markdown_cell("""# 23 — Correction Temporal Transferability

This notebook changes no model, feature, grouping, shrinkage, cap or submission. It freezes Notebook 21's selected
`hemisphere | alpha=200 | cap=±0.05` masked-row correction and tests whether its direction and magnitude transfer
across four independent future hindcast windows.

## Frozen decision policy

Score weights are fixed before results: window consistency 25%, weighted sign accuracy 20%, inverse harmful rate
15%, worst-window behavior 15%, regional consistency 10%, magnitude stability 10%, temporal-decay resilience 5%.
Acceptance requires ≥60% non-worse windows, weighted sign accuracy >55%, harmful rate <40%, worst loss ≤0.002,
southern benefit in at least two periods, non-fully-unstable magnitude, and score ≥65. These are research risk-control
thresholds, not universal scientific rules."""),
 nbf.v4.new_markdown_cell("""## Reproduction, leakage-free hindcasts and transferability synthesis

Three early date-grouped OOF blocks initialize residual evidence. Each future window is predicted before its
residuals can enter the historical pool. Assertions enforce strict `max(history time) < min(validation time)`."""),
 nbf.v4.new_code_cell(runtime)]
nb.metadata={"kernelspec":{"display_name":"GWS project environment","language":"python","name":"gws"},"language_info":{"name":"python","version":"3.13"}}
nbf.write(nb,OUT); print(OUT)
