# Notebook 37 final recommendation

1. **Most likely cause:** validation masking was routed after lag construction. F1 validation histories retained TWS values from dates that the simulated/test availability pattern masks; irregular folds also consumed intermediate observations absent from the issue-date pattern. This operational mismatch inflates local lag value.
2. **Most public-associated metric:** `Spatial block` has the highest observed pairwise accuracy (0.667), but n=4 is insufficient; no metric is statistically established.
3. **Misleading metric:** Notebook 36 F1 strict/irregular/spatial/space-time RMSE in its present form, because all use the unpropagated global TWS lag cache. `grid_asia` and space-time are additionally identical.
4. **Masking simulated adequately?** No. See `masking_pattern_audit.csv`; expert routing matched current-row status but historical availability was not propagated into lag creation.
5. **Riskiest changes:** masked rows with incomplete real test history and large absolute F1–canonical corrections; support strata are quantified without target labels.
6. **Public-positive common behavior:** evidence is too small for a universal pattern. The only model beating both canonical and F1 is Notebook 21's conservative correction: exact file comparison differs on 33.8% of rows, but its cap is 0.05 and MAD is only 0.0066. F1 changed every row with MAD 0.3348 and did not beat current-best.
7. **Can low-weight blend salvage F1?** Plausible only as a hypothesis. Fixed blends sharply reduce perturbation, but no common corrected OOF evidence exists; no file was produced.
8. **Validation redesign:** apply test-matched TWS masks before lagging, remove unavailable intermediate history, retain hard temporal/spatial gates, and do not use public scores in the decision score.
9. **One next experiment:** B) test-matched masking validation design.
10. **Evidence required before submission:** reproducible positive F1 gain on propagated-mask OOF, masked gain ≥0.004, nonnegative spatial/space-time direction, majority-fold gain, complete availability audit, and an independent rerun.

Observed public facts: canonical 0.824463818, current-best 0.823609466, F1 0.824335279. F1 improved canonical by 0.000128539 but trailed current-best by 0.000725813. No new submission/model/package/data was created.
