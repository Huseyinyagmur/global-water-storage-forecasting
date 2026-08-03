import nbformat as nbf
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/"notebooks/26_final_competition_report_and_repository_consolidation.ipynb"; runtime=(ROOT/"scripts/notebook26_runtime.py").read_text(encoding="utf-8")
nb=nbf.v4.new_notebook(); nb.cells=[
 nbf.v4.new_markdown_cell("""# 26 — Final Competition Report and Repository Consolidation

This notebook performs documentation and integrity consolidation only. It creates no model, feature, validation,
correction or submission. It fail-fast verifies the canonical/current-best SHA values, then produces final reports,
root documentation, environment manifests and eight reusable figures from existing evidence.

The final submission remains Notebook 21's current-best refined artifact. Its public improvement is retained without
hiding weak temporal transferability, same-grid support limitations, canonical ordering risk or tail-risk evidence."""),
 nbf.v4.new_markdown_cell("""## Final integrity gate, reports, repository documentation and delivery audit

All generated documentation uses repository-relative paths. Secret and large-file checks report filenames/risk only
and perform no deletion. Submission and dataset paths are never opened for writing."""),
 nbf.v4.new_code_cell(runtime)]
nb.metadata={"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python","version":"3.13"}}
nbf.write(nb,OUT); print(OUT)
