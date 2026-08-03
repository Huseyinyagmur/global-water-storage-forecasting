# Competition gap analysis

## Direct answer
The current solution is strong but information-limited. Its approximate rank of 125 is user-provided and was not independently archived. Notebook 11 scores 0.824463818 public; Notebook 21 improves this only to 0.823609466 via a small southern masked correction whose transfer score is 54.53/100. The gap is unlikely to be closed by more tuning of the same supplied variables.

## Evidence-backed diagnosis
- Missing independent observations of precipitation, evapotranspiration, runoff, snow, deep soil water, groundwater and regulation.
- Hydrology-labelled engineered proxies did not work: Notebook 18 public RMSE 0.831108253.
- Local/public mismatch recurred in Notebooks 12, 14, 16 and 18.
- OOF is canonical-equivalent rather than exact canonical, covers 34.774%, and uses the same spatial grid.
- Tail risk is material: OOF P95 1.520082, south P95 1.773752, maximum 4.461003.
- The global-month prior has 91.30% future-order exposure, demonstrating why ordering provenance matters.

## Likely missing top-10 ingredients
High-confidence hypotheses are cutoff-safe operational covariates, persistence anchoring, and known/masked handling. Medium-confidence hypotheses are deeper soil/snow/runoff/ET state, basin context and mechanism-diverse ensembling. Transformer or GraphCast use is low-confidence; no winner solution is public.

## Harsh conclusion
The repository has over-invested in transformations and model variants relative to new causal information. The spatial correction is a small leaderboard hedge, not a hydrologic breakthrough. A bigger model on the same features is a poor next bet.
