# Global Water Storage Forecasting

## Problem overview
Forecast gridded terrestrial-water-storage anomalies robustly across space and time. The project separates test rows where contemporaneous TWS is known from rows where it is masked.

## Best public score
**0.823609466** from `submissions/submission_21_spatial_residual_candidate.csv`. Canonical base score: **0.824463818**.

## Final system
Notebook 11 LightGBM selected blend is the canonical base. Notebook 21 adds a bounded +0.05 Southern-hemisphere correction only to eligible masked rows. It is the current-best refinement, not a new canonical.

## Key results
| Result | Value |
|---|---:|
| Public improvement | 0.000854352 (0.104%) |
| Strict / irregular / spatial / space-time RMSE | 0.740524 / 0.680366 / 0.801398 / 0.817423 |
| Transferability | 54.53/100, weak |
| OOF proxy RMSE / MAE | 0.743809 / 0.568233 |
| OOF P95 / P99 | 1.520082 / 2.123579 |
| Trustworthiness | 66.30/100, careful oversight |

## Repository structure
- `notebooks/`: research sequence 01–26
- `scripts/`: reproducible notebook builders
- `artifacts/`: audit tables, reports, figures and OOF data
- `submissions/`: immutable competition predictions
- `data/raw/`: locally supplied competition data (not assumed redistributable)

## Dataset setup
Place `Train.csv`, `Test.csv` and `SampleSubmission.csv` under `data/raw/`. See [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

## Installation and environment
Create a virtual environment, install `requirements-minimal.txt`, and use `requirements-reproducible.txt` when matching the captured local direct dependencies. Exact cross-platform determinism is not guaranteed.

## Reproduction steps
Build a report notebook with `python scripts/build_notebook26.py`, then execute it from the repository root. Advanced model notebooks require LightGBM and the local data. Verify SHA values before any audit.

## Notebook reading order
Recommended: 11 → 17 → 19 → 20 → 21 → 23 → 24 → 25 → 26. Full index: [experiment_index.md](artifacts/notebook26/experiment_index.md).

## Validation strategy
Decisions combine temporal, irregular hindcast, spatial-block and space-time evidence. Metrics from different designs are not directly interchangeable.

## Leakage controls
Complete dates are grouped and new residual/OOF work is train-on-past. The canonical global-month prior retains a documented location-major ordering risk affecting about 91.30% of training rows; no direct validation-target leakage was detected.

## Explainability
Known rows are persistence dominated; masked rows rely on SPEI, spatial/time and soil-moisture features. The correction is transparent but weakly transferable.

## Trustworthiness and limitations
Score: 66.30/100. The refinement improved public RMSE but worsened the closest historical window. Spatial support audit is same-grid and major physical processes are absent.

## Submission files
- Canonical: `submissions/submission_optimized.csv`
- Final/current-best: `submissions/submission_21_spatial_residual_candidate.csv`

## Artefacts and results
Start with `artifacts/notebook26/executive_summary.md`, `final_competition_report.md`, figures, and `final_results_table.csv`. Large row-level OOF files remain under `artifacts/notebook25/`.

## Intended use
Broad-scale water-storage research, drought decision support, screening and benchmarking with expert review. Not for critical local operations, emergency/safety automation, sanctions or guaranteed extrapolation.

## License / competition data note
Competition data and leaderboard scores may be governed by external terms. Confirm redistribution rights before publishing `data/` or submission-derived materials.

## Author / portfolio context
This repository demonstrates end-to-end ML research, validation design, leakage auditing, reproducibility, model-card reporting and evidence-based stopping. Add author/license details before public release.
