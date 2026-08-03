# OOF artefact README

Purpose: persistent row-level errors for tail-risk and support audit. Strategy: **canonical-equivalent leakage-controlled strict temporal OOF** with two complete-date strict temporal folds. Residual is always `target - oof_prediction`; positive means underprediction.

The known/masked expert structure and 0.65/0.35 blend are canonical-equivalent, but `prior_global_month` is rebuilt with strict date-grouped history to remove the canonical location-major ordering risk. Therefore this artefact must not be described as exact canonical OOF. Validation targets are outputs only and never feature inputs.

Use it for aggregate error, tail, regional, temporal, regime and support diagnostics. Do not use it to tune a new leaderboard model, infer test accuracy, claim unseen-space robustness (same-grid locations dominate), or compare directly with public RMSE. Cache files are accepted only when the frozen policy key and required fold files match.
