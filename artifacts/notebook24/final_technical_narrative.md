# Final technical narrative

The project progressed from statistical baselines to gradient boosting, then selected a conservative dual-expert LightGBM blend as the canonical system. Subsequent work showed that lower local RMSE did not reliably predict public performance: alternative models, broad feature searches, delta formulations and hydrology-labelled features generally failed to improve the leaderboard.

That mismatch motivated validation, leakage and reproducibility audits. Same-date splitting was identified as optimistic; Notebook 17 recovered the true selected-blend artifact; Notebook 19 formalized prediction-time availability; Notebook 20 added strict temporal, irregular, spatial and space-time tests and quantified a location-major target-prior ordering risk. These audits became more valuable than further model complexity.

Notebook 21 then made a deliberately small change: a leakage-free, OOF-derived Southern-hemisphere residual intercept applied only to masked rows. It improved public RMSE from 0.824463818 to 0.823609466. However, Notebook 23 found only weak historical temporal transferability: three of four windows improved, the closest window deteriorated by 0.002613, and the score was 54.53/100. The public gain is real for the submitted artifact, but it is not proof of a generally robust correction.

The project therefore stops further leaderboard optimization, preserves the current-best submission, and reports both strengths and limitations. Its strongest technical contribution is the evidence-driven transition from model chasing to validation discipline, artifact reproducibility, bounded refinement and honest uncertainty—not a claim of production readiness.
