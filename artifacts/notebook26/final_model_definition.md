# Final model definition

- **Canonical base:** Notebook 11 final selected blend; public RMSE 0.824463818; file `submissions/submission_optimized.csv`; SHA256 `b61ecd15862ca6d7527c4f494de802ed6ea6d31c420737cb78f977d9bda3b968`.
- **Architecture:** known-row LightGBM expert; masked-row base LightGBM plus causal-history expert with fixed 0.65/0.35 blend.
- **Target/features:** future TWS anomaly from spatial coordinates/bins, calendar/cyclic time, SPEI 01/03/06/12 and aggregates, soil moisture, known-row `TWS_t`, and historical priors.
- **Current-best refinement:** Notebook 21 stable Southern-hemisphere masked-row +0.05 residual intercept. It changes 36,832 test rows; known rows remain identical.
- **Final file:** `submissions/submission_21_spatial_residual_candidate.csv`; SHA256 `84d4ce6efdbfc3fbca55d9eea1ebccfe628ffa962e31b62a654c179dc649b488`; public RMSE 0.823609466.
- **Improvement:** 0.000854352 RMSE (0.104%).
- **Limitations:** Notebook 23 classified correction transfer as leaderboard-positive but weak (54.53/100). Trustworthiness is 66.30/100, useful with careful oversight.
- **OOF terminology:** Notebook 25 is `canonical-equivalent leakage-controlled OOF`, not exact canonical OOF, because the risky global-month ordering was replaced with strict date-grouped history.

Notebook 21 is not renamed canonical.
