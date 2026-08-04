# Notebook 38 plan — test-matched masking validation

## Selected experiment
**B) Test-matched masking validation design.** No model or feature search.

1. Freeze canonical E0 and F1-L12 parameters/features.
2. For every fold, map the exact test date-offset and `TWS_t` availability pattern onto validation dates.
3. Apply the availability mask to the entire validation-origin panel **before** any `shift(k)` calculation.
4. Exclude non-issue-date intermediate observations in irregular simulations unless they would be operationally available.
5. Rebuild only TWS lag/missing-mask features; other covariate lags remain cutoff-safe.
6. Assert every TWS lag source was available under the simulated issue-time mask.
7. Compare E0/F1 on common rows. Report strict, masked, worst-fold and lag-completeness strata.

Acceptance: F1 must retain ≥0.002 overall and ≥0.004 masked gain, no negative space-time/spatial direction, and positive gain in a majority of folds. Stop and reject F1 if its gain collapses under propagated masking. Only after an independent rerun may a conservative current-best blend be considered; Notebook 38 itself should not use public score for weighting.
