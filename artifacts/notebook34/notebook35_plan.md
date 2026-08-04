# Notebook 35 plan — CPU-friendly flattened sequence boosting

## Decision
Choose **B) CPU-friendly flattened sequence boosting**. This is the only selected next experiment. It directly tests whether Notebook 33's strong L12 signal is temporal information rather than a GRU-specific effect, requires no download/new framework, and can reuse the canonical dual-expert LightGBM pipeline.

## Main hypothesis
A leakage-safe 12-month flattened panel window, trained as known-delta and masked-absolute LightGBM experts, improves the frozen canonical OOF system and masked RMSE without degrading worst-fold or space-time robustness.

## Required data/packages and internet cost
Existing train/test tables, Notebook 11/25 canonical OOF artefacts, LightGBM/pandas/numpy already installed. Internet cost: zero. No external data and no package installation.

## Implementation
1. Freeze Notebook 11 folds, feature availability rules, seed and baseline predictions.
2. Sort by location/date; construct only past/current-origin L12 features. Never use target or future rows.
3. Add raw lags, differences, missingness/age and robust window summaries; use float32.
4. Train one conservative LightGBM configuration, separately for known-delta and masked-absolute targets.
5. Persist per-fold row-level OOF predictions and provenance assertions.
6. Compare canonical, flattened model, and a nonnegative OOF blend. Do not tune on leaderboard.
7. Run strict temporal, irregular-time and space-time audits before any full-test fit.

## Validation and acceptance gate
Accept only if overall RMSE improves by at least 0.002, masked RMSE does not worsen, worst-fold improves or is within 0.001, every evaluated fold improves in at least one primary regime, and space-time RMSE does not worsen by more than 0.002. Confirm zero future timestamps and fold-training-only statistics.

## Submission policy
Produce a candidate only after all gates pass. Compare IDs/hash/schema to canonical. Make at most one leaderboard submission and record it before seeing the score.

## Stop condition
Stop after the single frozen configuration if overall gain is below 0.002, masked/worst-fold gates fail, runtime is disproportionate, or any provenance assertion fails. Do not start architecture or hyperparameter search.
