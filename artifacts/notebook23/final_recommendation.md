# Notebook 23 final recommendation

1. **Exact policy reproduction:** True; max difference 3e-16, exact-within-tolerance rate 100.000%.
2. **Independent windows:** 4.
3. **Windows improved:** 3 / 4.
4. **Mean / median RMSE gain:** 0.00098938 / 0.00089921.
5. **Sign transfer:** overall 66.667%; sample-weighted 58.208%.
6. **Magnitude stability:** North: Unstable direction; South: Stable direction, unstable magnitude.
7. **Temporal decay:** Spearman(gap,gain)=-0.775, slope=-0.00015492/month. With four windows this is exploratory and does not prove decay.
8. **Southern transfer:** mean gain 0.00073879, positive/non-worse consistency 75.0%.
9. **Climate dependence:** see `climate_regime_transferability.csv`; differences are descriptive, not a new selection rule.
10. **Transferability score:** 54.53/100 — Weakly transferable. Thresholds are a research risk-control policy, not scientific constants.
11. **Interpretation:** C) Leaderboard-positive but weakly transferable.
12. **Preserve current-best submission:** Yes; no submission file was modified.
13. **Further correction-family optimization:** No; transfer evidence is insufficient for optimization.
14. **Notebook 24 single next experiment:** frozen correction uncertainty weighting, only if the acceptance gates above pass; otherwise stop this correction family.

All unsuccessful windows remain in `window_validation_results.csv`. Correction estimates use only earlier OOF residuals; validation residuals are appended only after their own evaluation.
