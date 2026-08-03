# Final recommendation

- **Final submission:** `submissions/submission_21_spatial_residual_candidate.csv` (public RMSE 0.823609466).
- **Canonical base:** `submissions/submission_optimized.csv` from Notebook 11 (0.824463818).
- **Score optimization:** closed; no new model/correction/submission is justified.
- **Most important limitation:** public-positive correction is weakly transferable and the closest historical window worsened; same-grid OOF does not prove unseen-region robustness.
- **Strongest contribution:** the evidence chain linking validation redesign, ordering/leakage audit, exact artifact reproduction, bounded refinement, transfer testing, model card and tail-risk audit.
- **Delivery readiness:** technically ready after owner completes data-license, author/license, large-file and git review. These governance steps cannot be inferred or completed automatically.
- **Missing mandatory technical work:** none in the requested report package; owner review items remain.
- **Recommended release:** `v1.0.0-competition-final` after checklist review.
- **Competition score to report:** 0.823609466, alongside canonical 0.824463818 and the 0.000854352 improvement.
- **Validation to present jointly:** strict temporal 0.740524, irregular 0.680366, spatial 0.801398, space-time 0.817423, transferability 54.53/100, and canonical-equivalent OOF RMSE/MAE 0.743809/0.568233. State that designs differ.
