# Final competition report

## 1. Introduction
This report consolidates Notebooks 01–25 without new score optimization.
## 2. Challenge Definition
Forecast terrestrial-water-storage anomaly robustly across space and time using prediction-time available inputs.
## 3. Dataset and Forecasting Structure
The data repeat a global spatial grid over grouped dates; test rows divide into known and masked `TWS_t` availability.
## 4. Exploratory Data Analysis
Early work established missingness, target spread, spatial coverage and temporal grouping.
## 5. Baseline Development
Simple baselines were useful controls but early validation was not decision-grade.
## 6. Gradient Boosting Models
LightGBM became the strongest family; XGBoost/CatBoost did not establish a verified public advantage.
## 7. Ensemble and Canonical Blend
Notebook 11's selected blend became canonical at public RMSE 0.824463818.
## 8. Feature Engineering Research
Large generic and clustered feature searches often improved local metrics but transferred poorly.
## 9. Alternative and Delta Forecasting Approaches
Notebook 14/15 scored 0.825971129/0.824797753 and did not beat canonical.
## 10. Hydrology-Inspired Feature Research
Domain-labelled features did not add robust independent signal; Notebook 18 public RMSE was 0.831108253.
## 11. Validation Evolution
Validation progressed from optimistic same-date splits to temporal, irregular, spatial and space-time gates.
## 12. Leakage and Ordering Audit
The global-month prior has a 91.30% future-timestamp ordering exposure in training; no direct validation-target leakage was found.
## 13. Reproducibility Audit
Notebook 17 corrected a pipeline identity mismatch and froze canonical SHA256.
## 14. Spatial Residual Correction
Notebook 21 applied a bounded +0.05 Southern masked-row correction and improved public RMSE to 0.823609466.
## 15. Temporal Transferability Evaluation
Only three of four windows improved; the closest lost 0.002613. Score 54.53/100 means weak transferability.
## 16. Bias and Trustworthiness Assessment
Trustworthiness is 66.30/100: useful with careful oversight; Southern and temporal risks remain.
## 17. OOF Tail-Risk and Support Audit
Canonical-equivalent OOF RMSE/MAE are 0.743809/0.568233; P95/P99/max errors are 1.520082/2.123579/4.461003.
## 18. Final Submission Selection
Use `submission_21_spatial_residual_candidate.csv`; preserve canonical separately.
## 19. Limitations
Weak correction transfer, same-grid support, omitted groundwater/snow/surface/human-water processes, proxy OOF and external leaderboard verification.
## 20. Intended Use
Broad-scale research, drought support and benchmarking with expert review; not critical automated decisions.
## 21. Reproducibility
Builders, SHA gates and caches are provided; environment portability and early notebooks remain moderate.
## 22. Lessons Learned
Validation and reproducibility were as consequential as modelling; local gain was not public proof.
## 23. Final Conclusion
The project stops at a small verified public improvement and documents why that improvement should not be overstated.
