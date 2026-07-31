# Notebook 24 final recommendation

1. **Current-best:** Notebook 11 canonical selected blend plus Notebook 21 +0.05 stable Southern masked-row correction.
2. **Integrity:** yes; both SHA256 values, IDs, finiteness and known-row preservation passed.
3. **Public improvement:** 0.000854352 RMSE (0.104% relative).
4. **Strongest validation result:** the current-best public score is the direct competition outcome; among local systems no single validation is sufficient, so the joint Notebook 20 suite is the strongest decision support.
5. **Weakest robustness result:** Notebook 23 transferability 54.53/100; closest window −0.002613.
6. **Spatial risk:** Southern hemisphere was the worst canonical block and remains the correction-sensitive region.
7. **Temporal risk:** correction benefit reversed in the closest historical window; four windows provide limited evidence.
8. **Leakage risk:** canonical `prior_global_month` location-major training ordering (91.30% future-timestamp contributors); no direct validation-target leakage was found.
9. **Reproducibility:** Moderate.
10. **Explainability:** good at global feature-family and correction-rule level, limited for causal/local claims.
11. **Missing physics:** groundwater, snow/ice, surface water, runoff, ET, human abstraction/irrigation/reservoir and land-use dynamics; soil moisture is only partially represented.
12. **Suitable use:** broad-scale research, drought support, screening and benchmarking with experts.
13. **Do not use for:** critical local operations, emergencies, sanctions, safety automation or guaranteed extrapolation.
14. **Trustworthiness:** 66.30/100 — Useful but requires careful oversight; project-internal framework only.
15. **Preserve current-best:** Yes.
16. **New score optimization:** No.
17. **Technical story:** a strong canonical model survived broad alternatives; audits exposed validation/reproducibility risk; one bounded correction improved public score but weak transfer evidence triggered a principled stop.
18. **Notebook 25:** one audit-only step—persist and analyze compact row-level OOF residuals for tail-risk and support monitoring, without changing predictions.
