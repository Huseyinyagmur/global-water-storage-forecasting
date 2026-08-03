# Notebook 28 plan — one main idea

## Hypothesis
Prediction-time-legal ERA5-Land water-budget states add missing physical information beyond SPEI and near-surface soil moisture, allowing a conservative residual model to improve Notebook 11 across time and space.

## Scope
Only ERA5-Land precipitation, total evaporation, runoff, snow-water equivalent/snowmelt proxy, and four soil-water layers. One controlled residual learner. No architecture search, correction optimization, or additional dataset.

## Protocol
1. Reproduce frozen Notebook 11 predictions and hashes.
2. Construct an issue-time vintage manifest and fail closed on unavailable data.
3. Aggregate to monthly/grid features in float32 with cached partitions.
4. Fit residual model using training-only fold histories.
5. Evaluate four complete-date temporal folds plus basin-held-out, known/masked, hemispheric and tail slices.
6. Compare against Notebook 11 under frozen gates.

## Acceptance
Pass only with zero vintage violations, >=3/4 temporal improvements, >=3/4 masked non-worse, basin/south non-inferiority, mean gain >=0.002 and above seed/fold noise, and <=1% P95 deterioration. Otherwise record failure and keep canonical.
