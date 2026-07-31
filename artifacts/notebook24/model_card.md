# Model Card — Current-best refined TWS forecasting system

## 1. Model Overview
Notebook 11 canonical selected blend plus Notebook 21 Southern-hemisphere masked-row residual correction. Public RMSE 0.823609466. Notebook 21 is not renamed canonical.
## 2. Problem Definition
Forecast gridded terrestrial-water-storage anomaly across space and time.
## 3. Intended Use
Broad-scale research, drought decision support, risk screening and benchmarking with expert review.
## 4. Out-of-Scope Use
Critical local operations, emergency or human-safety automation, sanctions, or guaranteed extrapolation.
## 5. Training Data
Repeated global grid and monthly dates with TWS, SPEI and soil-moisture fields; exact schema is repository-bound.
## 6. Evaluation Data
Temporal OOF, strict temporal, irregular hindcast, spatial block, space-time and four transfer windows; scores are not directly interchangeable.
## 7. Feature Families
Spatial coordinates/bins, calendar/cyclic time, multi-scale SPEI, soil moisture, known-row TWS and historical priors.
## 8. Model Architecture
LightGBM expert system; masked base/history blend 0.65/0.35.
## 9. Refinement Method
Stable historical OOF hemisphere residual intercept, +0.05 cap on eligible Southern masked rows; known rows unchanged.
## 10. Performance
Canonical/current public RMSE 0.824463818/0.823609466; absolute improvement 0.000854352.
## 11. Spatial Robustness
Spatial canonical mean 0.801398; southern canonical fold 0.833364. NB21 improved spatial validation, but this does not prove universal regional fairness.
## 12. Temporal Robustness
NB23: 3/4 windows improved; closest window lost 0.002613; score 54.53, weakly transferable.
## 13. Bias and Fairness Considerations
Geographic performance disparities matter; this is environmental-model bias, not a demonstrated demographic fairness assessment. Sample size and uncertainty accompany claims.
## 14. Explainability
Known expert is persistence-dominated (`TWS_t`); masked expert relies on long/mid SPEI, spatial/time variables and soil moisture. Correction is simple but weakly transferable.
## 15. Leakage Controls
Date-grouped folds and train-on-past correction. Canonical `prior_global_month` retains a documented indirect training-order risk (91.30% future-timestamp contributor rate), without detected direct validation-target leakage.
## 16. Reproducibility
Rating: Moderate. Immutable SHA gates, builders and assertions are strong; environment pinning and row-level OOF persistence are incomplete.
## 17. Computational Requirements
Tree-model batch inference is plausible; exact production latency/peak memory are not available.
## 18. Limitations
Weak correction transfer, incomplete physical state, proxy validation, small public candidate sample and incomplete tail artefacts.
## 19. Ethical and Operational Risks
False confidence in local water decisions, unsupported extrapolation, and omission of human water operations.
## 20. Monitoring Recommendations
Monitor regional/window RMSE, residual sign, tails, covariate support, correction coverage, hashes and data drift; require expert escalation.
## 21. Versioning and SHA256
Canonical `b61ecd15862ca6d7527c4f494de802ed6ea6d31c420737cb78f977d9bda3b968`; current-best `84d4ce6efdbfc3fbca55d9eea1ebccfe628ffa962e31b62a654c179dc649b488`.
## 22. Final Recommendation
Preserve current-best for the competition record; stop score optimization. Use only as a research-grade decision-support artifact with careful oversight, not as production proof.
