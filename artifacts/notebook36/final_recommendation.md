# Notebook 36 final recommendation

## Decision: accept F1_L12 and produce submission

- Pilot sonucu yeniden üretildi mi? **True**; RMSE=0.640994.
- Strict temporal gain: **+0.145670**.
- Irregular hindcast gain: **+0.125202**.
- Spatial gain: **+0.163781** (frozen serious-degradation tolerance=0.003).
- Space-time gain: **+0.108231**.
- Strict masked RMSE gain: **+0.175389**.
- Worst fold korundu mu? **True**.
- Leakage audit geçti mi? **True**.
- F1 acceptance gate geçti mi? **True**; checks={'strict_gain': True, 'masked_gain': True, 'space_time_not_worse': True, 'spatial_not_seriously_worse': True, 'worst_fold_not_worse': np.True_, 'majority_positive': np.True_, 'leakage': np.True_, 'alignment': True}.
- Frozen blend acceptance gate geçti mi? **False**; checks={'strict_gain': True, 'masked_gain': True, 'space_time_not_worse': True, 'spatial_not_seriously_worse': True, 'worst_fold_not_worse': np.True_, 'majority_positive': np.True_, 'leakage': np.True_, 'alignment': True}.
- Submission üretildi mi? **True**.
- Leaderboard'a önce hangi dosya gönderilmeli? **submission_36_f1_l12_primary.csv**.

All metrics were computed from row-level fold checkpoints. No feature/model/hyperparameter search was performed; L12 raw lags, canonical parameters, seed 42 and blend weight 0.05 remained frozen.
