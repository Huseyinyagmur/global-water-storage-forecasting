# Notebook 25 final audit recommendation

1. **Valid row-level OOF:** Yes—canonical-equivalent OOF, not exact canonical OOF; strict chronology and unique-row assertions passed.
2. **OOF coverage:** 34.774% (749,046/2,154,021).
3. **Key tail metric:** P95 absolute error 1.520082; maximum 4.461003.
4. **Largest regional tail:** South, P95 1.773752.
5. **Largest fold tail:** strict_temporal_2, P95 1.556778.
6. **Low-support error:** consult support tables; Spearman(combined support, absolute error)=-0.1956, bootstrap CI [-0.2097,-0.1824]. Association is diagnostic, not causal.
7. **Stronger support relationship:** compare spatial/temporal group tables; same-grid spatial support has limited variation, so temporal evidence is more informative here.
8. **Test low/out spatial support:** 0.000%; no target-based performance claim is made.
9. **Corrected test support:** 36,832 rows; mean combined support 0.7572; low/out spatial share 0.000%.
10. **Risk flags:** diagnostic only; see `test_row_risk_flags.csv`.
11. **Notebook 24 completion:** 9 previously unavailable tail metrics now populated from proxy OOF.
12. **Model card:** update with proxy-OOF tail numbers, same-grid spatial-support limitation, and test temporal-extrapolation profile.
13. **Preserve current-best:** Yes; submission hashes remain unchanged.
14. **New score optimization:** No.
15. **Report readiness:** Yes, with the canonical-equivalent/proxy terminology and limitations retained.

**Final decisions:** OOF artefact valid; tail/support audit complete; current-best retained; score optimization closed. Notebook 26 should be `26_final_competition_report_and_repository_consolidation.ipynb`, a documentation/consolidation step only.
