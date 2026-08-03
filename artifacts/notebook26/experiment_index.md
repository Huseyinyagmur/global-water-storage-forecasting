# Experiment index

## Foundation

| Notebook | File | Purpose | Status |
|---:|---|---|---|
| 1 | `01_eda.ipynb` | Data/schema and target exploration | historical experiment |
| 2 | `02_baseline_model.ipynb` | Establish simple prediction baseline | historical experiment |

## Model development

| Notebook | File | Purpose | Status |
|---:|---|---|---|
| 3 | `03_lightgbm.ipynb` | Test LightGBM | historical experiment |
| 4 | `04_xgboost.ipynb` | Test XGBoost | historical experiment |
| 5 | `05_catboost.ipynb` | Test CatBoost | historical experiment |
| 6 | `06_ensemble.ipynb` | Blend boosting predictions | historical experiment |
| 9 | `09_advanced_model_selection.ipynb` | Advanced model selection and submission | historical experiment |
| 11 | `11_submission_improvement.ipynb` | Dual-expert submission improvement | successful |
| 11 | `11_submission_optimization.ipynb` | Optimize submission with conservative blend | successful |
| 14 | `14_alternative_models.ipynb` | Test non-boosting families | historical experiment |
| 15 | `15_delta_spatiotemporal_forecasting.ipynb` | Forecast target delta with dual experts | historical experiment |

## Feature research

| Notebook | File | Purpose | Status |
|---:|---|---|---|
| 7 | `07_feature_engineering.ipynb` | General feature engineering | historical experiment |
| 12 | `12_advanced_submission_optimization.ipynb` | Expand optimization ideas | historical experiment |
| 16 | `16_competition_level_feature_research.ipynb` | Large single-feature and interaction research | historical experiment |
| 18 | `18_domain_hydrology_feature_research.ipynb` | Test hydrology-inspired features | historical experiment |

## Validation and audit

| Notebook | File | Purpose | Status |
|---:|---|---|---|
| 8 | `08_time_series_validation.ipynb` | Temporal validation redesign | historical experiment |
| 10 | `10_model_explainability.ipynb` | Explain strongest model and residuals | historical experiment |
| 13 | `13_generalization_optimization.ipynb` | Reduce local overfitting | historical experiment |
| 17 | `17_submission_reproducibility_audit.ipynb` | Audit submission identity and reproducibility | audit/report |
| 19 | `19_challenge_technical_specification.ipynb` | Translate challenge into technical specification | audit/report |
| 20 | `20_strict_temporal_spatial_validation.ipynb` | Audit leakage and generalization proxies | audit/report |

## Refinement

| Notebook | File | Purpose | Status |
|---:|---|---|---|
| 21 | `21_spatial_residual_correction.ipynb` | Correct stable spatial residual bias | successful |

## Trustworthiness and reporting

| Notebook | File | Purpose | Status |
|---:|---|---|---|
| 22 | `22_research_roadmap_and_experiment_synthesis.ipynb` | research roadmap | audit/report |
| 23 | `23_correction_temporal_transferability.ipynb` | temporal correction transferability | audit/report |
| 24 | `24_trustworthiness_bias_and_model_card.ipynb` | trustworthiness/model card | audit/report |
| 25 | `25_oof_tail_risk_and_spatial_support_audit.ipynb` | row-level OOF tail/support audit | audit/report |
