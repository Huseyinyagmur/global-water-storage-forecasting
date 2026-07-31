# Model system summary

- **Problem:** global gridded terrestrial-water-storage forecasting; target is future `target`/TWS anomaly at a latitude, longitude and date.
- **Data structure:** repeated global spatial grid across grouped monthly dates; official test has known and masked `TWS_t` rows.
- **Canonical base:** Notebook 11 final selected blend, public RMSE 0.824463818. A known-row LightGBM expert and masked base LightGBM are combined with a causal-history masked expert; the masked prediction uses the verified 0.65 base / 0.35 history blend.
- **Feature families:** latitude/longitude and spatial bins, calendar/cyclic time, SPEI 01/03/06/12 and aggregates, soil moisture, and `TWS_t` only where available; historical priors are target-derived and carry the documented training-order risk.
- **Prediction-time availability:** location, date, supplied SPEI/soil covariates and known `TWS_t` rows; masked rows cannot use unavailable `TWS_t`.
- **Notebook 21 refinement:** OOF residual = target − prediction; stable Southern-hemisphere masked rows receive a bounded +0.05 intercept. Known rows remain byte/numerically unchanged.
- **Validation:** Notebook 11 temporal OOF, then Notebook 20 strict temporal, irregular hindcast, spatial block and space-time; Notebook 23 adds four independent temporal transfer windows.
- **Selection rationale:** Notebook 21 improved public RMSE to 0.823609466 under predeclared gates. Notebook 23 later classified its historical transfer as weak, so it is the current-best refined submission—not a new canonical base and not a general robustness claim.
