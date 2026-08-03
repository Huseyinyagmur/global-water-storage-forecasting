# Executive summary

This project forecasts global gridded terrestrial-water-storage anomalies across repeated spatial locations and future dates. The final system retains Notebook 11's two-expert LightGBM selected blend as the **canonical base** and uses Notebook 21's bounded Southern-hemisphere residual intercept only as the **current-best refinement**.

Canonical public RMSE is **0.824463818**; current-best public RMSE is **0.823609466**, an improvement of **0.000854352 (0.104%)**. Model selection evolved from early splits toward grouped temporal, irregular-date, spatial-block and space-time validation. Audits found that the canonical `prior_global_month` training feature has a location-major ordering risk: approximately 91.30% of training rows can reference later timestamps. No direct validation-target leakage was detected.

The public-positive correction is not presented as universally robust. Four historical windows improved in only three cases; the closest window deteriorated by 0.002613 and transferability scored 54.53/100. Canonical-equivalent leakage-controlled OOF covers 749,046 rows (34.774%), with RMSE 0.743809, MAE 0.568233, P95 absolute error 1.520082 and Southern P95 1.773752. Trustworthiness is 66.30/100: **useful but requires careful oversight**.

Final decision: submit `submissions/submission_21_spatial_residual_candidate.csv`, preserve the canonical base unchanged, close further score optimization, and use this work for broad-scale research/benchmarking with expert review—not critical local operations or automated safety decisions.
