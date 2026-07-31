import nbformat as nbf
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/"notebooks/24_trustworthiness_bias_and_model_card.ipynb"
runtime=(ROOT/"scripts/notebook24_runtime.py").read_text(encoding="utf-8")
nb=nbf.v4.new_notebook(); nb.cells=[
 nbf.v4.new_markdown_cell("""# 24 — Trustworthiness, Bias and Model Card

This is an audit and reporting notebook. It trains no model, creates no feature/correction/submission, and preserves
the canonical and current-best artifacts. It consolidates performance, spatial/temporal bias, leakage,
reproducibility, explainability, physical completeness and operational-use evidence.

## Frozen trustworthiness framework

Scores and weights are fixed before generating the result: performance 15%, temporal robustness 15%, spatial
robustness 12%, bias awareness 8%, leakage control 12%, reproducibility 12%, explainability 8%, computational
practicality 5%, physical completeness 8%, operational safety 5%. The rating is project-internal, not certification.
Missing row-level/timing evidence is reported as `not available`, never reconstructed from aggregates."""),
 nbf.v4.new_markdown_cell("""## Integrity gate and consolidated technical assessment

The executable cell fail-fast validates both immutable SHA values, then writes all model-card and trustworthiness
artifacts from existing repository evidence. No path under `submissions/` is opened for writing."""),
 nbf.v4.new_code_cell(runtime)]
nb.metadata={"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python","version":"3.13"}}
nbf.write(nb,OUT); print(OUT)
