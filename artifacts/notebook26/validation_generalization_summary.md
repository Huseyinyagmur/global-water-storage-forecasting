# Validation and generalization summary

| Design | Result | Role / limitation |
|---|---:|---|
| Notebook 08 same-month split | 0.6109 recorded | optimistic; same date split across rows |
| Notebook 11 temporal OOF | 0.711647 overall | useful known/masked simulation, incomplete spatial robustness |
| Strict temporal | 0.740524 | grouped forecast-origin causality |
| Irregular hindcast | 0.680366 | mimics official non-uniform date gaps |
| Spatial block | 0.801398 | contiguous-region transfer; limited ranking evidence |
| Space-time | 0.817423 | future date plus held region |
| Southern worst canonical | 0.833364 | strongest spatial warning |
| Notebook 23 transferability | 54.53/100 | 3/4 improved; closest window −0.002613 |
| Notebook 25 canonical-equivalent OOF | RMSE 0.743809; MAE 0.568233 | 2 strict folds, 34.774% coverage, same-grid spatial support |

These metrics belong to different validation designs and must not be compared as though they were one leaderboard. Final decisions use their joint pattern, not a single local score.
