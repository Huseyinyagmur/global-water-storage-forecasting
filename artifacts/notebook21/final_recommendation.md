# Notebook 21 final recommendation

Residual is defined as **target - prediction**; positive values mean underprediction.

1. **Is there systematic Southern Hemisphere bias?** yes. The historical OOF mean and CI are recorded in `residual_stability.csv`.
2. **Is its direction stable across folds?** True; the rule required all three folds, >=75% sign consistency, >=100 rows per fold, |bias|>=0.01, and a pooled bootstrap CI excluding zero.
3. **Groups with reliable correction evidence:** ['hemisphere:South', 'lat_band:-10', 'lat_band:-20', 'lat_band:30', 'lat_band:40', 'spatial_cell:-10|-80', 'spatial_cell:-10|0', 'spatial_cell:-10|100', 'spatial_cell:-20|0', 'spatial_cell:-20|20', 'spatial_cell:-20|40', 'spatial_cell:-30|100', 'spatial_cell:-30|120', 'spatial_cell:-40|160', 'spatial_cell:-50|160', 'spatial_cell:0|100', 'spatial_cell:0|80', 'spatial_cell:20|-100', 'spatial_cell:20|-20', 'spatial_cell:20|40', 'spatial_cell:30|-140', 'spatial_cell:30|-80', 'spatial_cell:30|20', 'spatial_cell:40|100', 'spatial_cell:40|20', 'spatial_cell:50|-100', 'spatial_cell:50|-140', 'spatial_cell:60|-40', 'spatial_cell:70|-100', 'spatial_cell:70|120'].
4. **Best correction method:** `hemisphere|light|cap_0.05` (best gate-compliant candidate if any; otherwise diagnostic best spatial candidate).
5. **Spatial-block RMSE gain:** 0.00531118.
6. **Strict temporal / irregular preserved?** gains 0.00084536 / 0.00127917; fixed tolerances were -0.0005 each.
7. **Space-time robustness:** gain 0.00000000; fixed tolerance was -0.0002.
8. **Prediction change:** mean absolute correction 0.00880744, maximum 0.05000000, corrected fraction 17.615%.
9. **New submission generated?** True.
10. **Keep canonical Notebook 11?** False. Canonical SHA remains `b61ecd15862ca6d7527c4f494de802ed6ea6d31c420737cb78f977d9bda3b968`.
11. **Next highest-priority experiment:** repeat this exact frozen correction policy on a new, later fully held-out space-time window; do not expand correction complexity before independent stability evidence.

All failed candidates remain in `correction_ablation.csv`. Spatial validation was not optimized alone: acceptance jointly required strict temporal, irregular, spatial, space-time, worst-fold, evidence, and magnitude gates.
