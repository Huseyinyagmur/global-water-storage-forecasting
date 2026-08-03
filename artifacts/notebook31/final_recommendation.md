# Notebook 31 final recommendation

**Decision: proceed to full global download**

- Cached/downloaded files: 27 total; 9 reused, 18 newly downloaded; 73.20 MB.
- NetCDF valid: True; strict cutoff safe: 100.000%.
- Processed rows: 37,779; raw missing rate: 0.540%.
- E0 RMSE: 0.765207; known 0.768114; masked 0.747487; worst 0.861263.
- Best ERA system: E1, RMSE 0.755780, gain 0.009427, masked gain 0.008345, worst-fold gain 0.017766.
- E3 OOF-selected subset: tp, water_balance_proxy, e (exploratory; same OOF folds, not independent confirmation).
- Most important raw: tp; derived: water_balance_proxy.
- Known-expert top ERA5 feature: tp; masked-expert top: tp. Fold stability is available in `feature_importance.csv`.
- Frozen gates: [('E1', True, True, True, True), ('E2', True, False, False, True)].
- No full download was automatically started. No submission was produced.

This remains a three-region pilot. Proceeding means authorization for a separate future global download step, not evidence of public-leaderboard improvement.
