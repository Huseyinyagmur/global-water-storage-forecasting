# Final recommendation

If this were my project, today I would implement one idea: **a cutoff-safe ERA5-Land water-budget residual model around the frozen Notebook 11 prediction**.

Why this one: TWS is the sum of groundwater, soil moisture, surface water and snow, but the current system mainly sees lagged TWS, SPEI and near-surface soil moisture. Notebooks 12–18 show that more transformations, hyperparameters and model families do not reliably transfer. ERA5-Land supplies independent, physically aligned precipitation, evaporation, runoff, snow and deep-soil states, is global, public and updated daily. That makes it a better information bet than a transformer on unchanged inputs.

The hard part is not downloading ERA5-Land; it is proving historical issue-time legality. Final reanalysis values can incorporate information or revisions unavailable at the forecast date. Notebook 28 must therefore store valid time, release/retrieval time, stream and version for every feature, fail closed on uncertain availability, and document the exact operational analogue.

I would freeze Notebook 11, predict only its OOF residual with one conservative LightGBM configuration, and pre-register temporal, basin, masked, southern and tail gates. I would require improvement in at least three of four temporal windows, masked non-inferiority in three of four, no meaningful basin/south degradation, mean RMSE gain of at least 0.002 beyond seed/fold noise, and no more than 1% P95 deterioration. If any gate fails, I would not submit it.

This is not a claim that ERA5-Land will reach top 10. It is the highest-value falsifiable experiment because it attacks the repository's information ceiling while preserving its strongest assets: the canonical predictor, strict chronology, reproducibility and conservative selection.
