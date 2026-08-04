# Validation redesign

## Root cause found
Notebook 36 routed validation rows into known/masked experts **after** building the global TWS lag cache. A validation date labelled masked therefore still contributed its true `TWS_t` to later lag features. Irregular folds also allowed omitted intermediate Train dates into history. This is not direct target-label leakage, but it violates operational prediction-time availability and makes F1's masked validation materially easier than test inference.

## Keep / secondary / retire
- **Keep:** date-grouped temporal cutoffs, spatial blocks, row alignment assertions, worst-fold reporting and checkpointing.
- **Secondary:** irregular and spatial folds after rebuilding histories from an issue-time availability mask. Their current scores are diagnostic only.
- **Retire as decision metrics:** Notebook 36 F1 scores computed from unmasked global lag cache; `grid_asia` and `space_time_asia` as two independent signals (they are identical masks); pilot-only RMSE for submission promotion.

## Predeclared target-free decision score
After repairing masking, use 30% masking-pattern similarity, 20% lag-completeness similarity, 15% temporal/date-gap similarity, 15% spatial-distribution similarity, 10% worst-fold robustness and 10% location/climate coverage. Public score is excluded from the score. A candidate must also improve masked RMSE and pass every operational-availability assertion; the weighted score cannot compensate for a failed hard gate.
