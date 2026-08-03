# Bias, tail risk and support summary

- Southern hemisphere is the principal geographic weakness: canonical spatial-block RMSE 0.833364 and Notebook 25 regional P95 absolute error 1.773752.
- Global proxy-OOF tail: P95 1.520082, P99 2.123579, maximum 4.461003.
- Combined support versus absolute error Spearman is −0.1956 (diagnostic association, not causality).
- Strict temporal folds use essentially the same spatial grid; spatial-support variation is limited and cannot validate unseen-region robustness.
- Test low/out spatial-support fraction is 0%, but test dates are temporal extrapolation and have no target for performance claims.
- Notebook 21 changes 36,832 Southern masked rows. Their mean combined support is 0.7572; none are low/out spatial support under the same-grid definition.
- Extreme-target and climate/drought tables are available in `artifacts/notebook25/`; they show diagnostic differences, not proof that omitted physics causes errors.
- Central temporal risk: the closest Notebook 23 hindcast reversed the correction benefit by −0.002613.
