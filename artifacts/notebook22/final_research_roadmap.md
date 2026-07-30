# Final research roadmap

## Current best system definition

- **Canonical base:** Notebook 11 final selected blend, public RMSE 0.824463818.
- **Base components:** known-row LightGBM expert; masked base LightGBM plus causal-history expert conservative blend.
- **Refinement:** Notebook 21 stable hemisphere residual intercept, learned from older OOF residuals, shrinkage/cap controlled.
- **Application:** known rows unchanged; only masked rows with stable evidence are corrected.
- **Joint gates:** spatial mean and south improve; strict temporal, irregular, space-time and worst fold remain within predeclared tolerances; correction magnitude and support pass safety checks.
- **Integrity:** rows 280961, ID order retained, no duplicate/NaN/inf.
- **Correlation vs base:** 0.999512715393; MAD 0.006554646374; max difference 0.050000000000; changed fraction 13.109293%.
- **SHA256:** canonical `b61ecd15862ca6d7527c4f494de802ed6ea6d31c420737cb78f977d9bda3b968`; refined `84d4ce6efdbfc3fbca55d9eea1ebccfe628ffa962e31b62a654c179dc649b488`.

Notebook 21 is the **current best refined submission**, not a replacement name for the canonical base.


## Evidence synthesis

1. **Most successful main model:** Notebook 11 canonical LightGBM selected blend.
2. **Most successful refinement:** Notebook 21 small, stable hemisphere residual correction.
3. **Strongest local–public mismatch explanation:** validation mismatch plus repeated local-selection overfit under temporal/spatial distribution shift; specific mechanisms remain hypotheses except proven pipeline/leakage audits.
4. **Primary decision support:** joint strict temporal + irregular hindcast + spatial block + space-time gates, with spatial evidence weighted but never used alone.
5. **Never standalone:** Notebook 08 same-date row split, Notebook 11 local metric, or spatial ranking proxy alone.
6. **Critical leakage risk:** location-major `prior_global_month` ordering used future-timestamp training targets in 91.30% of rows. Notebook 20 did not find direct validation-target leakage.
7. **Current best:** Notebook 11 canonical base refined by Notebook 21; public RMSE 0.823609466.
8. **Do not repeat:** broad boosting/tuning/feature searches, uncontrolled blends, risky target encodings, single-metric selection, or unreproduced-base experiments.
9. **Highest priority:** frozen correction transferability across dates.
10. **Order:** NB23 transferability → NB24 uncertainty weighting if confirmed → NB25 spatial-support eligibility audit.
11. **Stop:** apply `research_stopping_rules.md`; especially stop a family after two public-worse submissions or failed independent sign/worst-fold replication.
12. **Strongest technical story:** a strong canonical boosting system survived extensive alternatives; audits exposed validation and reproducibility weaknesses; a bounded, OOF-derived spatial bias correction—rather than a new large model—produced the only verified improvement. This is the best supported explanation, not proof of causality.

## What actually worked

- **NB11:** score improvement; reusable frozen selected-blend base.
- **NB17:** methodology correction; reusable artifact identity/SHA gate.
- **NB19:** specification correction; reusable availability and robustness contract.
- **NB20:** validation correction; reusable joint temporal/spatial gate suite.
- **NB21:** score improvement; reusable bounded OOF residual correction policy.

The hypothesis that NB21 won because the main model was already strong and the residual correction removed a small,
stable spatial bias is consistent with its high base correlation, small MAD, known-row invariance, southern validation
gain and improved public score. It remains the strongest repository-supported explanation, not an experimentally proven causal fact.

## Public-score caveat

Notebook 16 A/C scores were explicitly supplied for this synthesis but exact score strings were not found in repository
artefacts; the leaderboard table flags this rather than presenting them as independently corroborated.
